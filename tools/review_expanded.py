"""Export saved, pinned expanded E2 results to one offline review page."""

import argparse
import json
import math
from pathlib import Path
import re
import time

from sgfmill.boards import Board

from kenggo.checkpoints import file_hash, read_json


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(__file__).with_name("e2_expanded_review.html")
ARMS = ("baseline-conditioned", "baseline-unconditioned", "combined-conditioned", "combined-unconditioned")
ROLES = ("population_development", "validation_development")
SETS = tuple(f"{role}_komi{komi}" for role in ROLES for komi in (13, 1))
STAGES = ("preparation", *("probe-" + arm for arm in ARMS), *ARMS,
          *("evaluation-" + arm for arm in ARMS), "evaluation-human-reference",
          "matches-baseline", "matches-combined", "analysis")
MAX_JSON_BYTES = 32 * 1024**2
MAX_PAGE_BYTES = 32 * 1024**2
FINISH_STAGES = {"analysis", "matches-baseline", "matches-combined"}
REFERENCE_STAGE = "evaluation-human-reference"


def _path(reference, *, base=ROOT, owner=None):
    if (not isinstance(reference, dict) or not {"path", "sha256"} <= reference.keys()
            or not isinstance(reference["path"], str) or not reference["path"]
            or not isinstance(reference["sha256"], str)
            or re.fullmatch(r"[a-f0-9]{64}", reference["sha256"]) is None):
        raise ValueError("Saved artifacts need a path and SHA-256 pin")
    path = (base / reference["path"]).resolve()
    if owner is not None and not path.is_relative_to(owner.resolve()):
        raise ValueError("Saved artifact leaves its owning stage")
    return path


def _json(reference, *, base=ROOT, owner=None):
    path = _path(reference, base=base, owner=owner)
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("Saved JSON exceeds the review read limit")
    return read_json(path, reference["sha256"])


def _status(value, label, allowed=("completed", "incomplete", "failed")):
    if not isinstance(value, dict) or value.get("status") not in allowed:
        raise ValueError(f"{label} has no recognized explicit status")
    return value["status"]


def _display_frames(state, count):
    initial, moves = state.get("initial", {}), state.get("moves")
    if (state.get("origin") != "empty" or initial.get("board") != ["." * 19] * 19
            or initial.get("to_play") != "B" or initial.get("captured_black") != 0
            or initial.get("captured_white") != 0 or state.get("komi_halfpoints", 13) != 13
            or not isinstance(moves, list) or type(count) is not int or count != len(moves)
            or count > 801):
        raise ValueError("Review display requires bounded, empty-board 6.5-komi match records")
    board, previous, frames = Board(19), "." * 361, [[[], "B", 0, 0, 0]]
    captured_black = captured_white = passes = 0
    for index, move in enumerate(moves):
        if (not isinstance(move, list) or len(move) != 2 or move[0] != ("B" if index % 2 == 0 else "W")
                or type(move[1]) is not int or not 0 <= move[1] <= 361):
            raise ValueError("Invalid saved move in display reconstruction")
        color, action = move
        if action == 361:
            passes += 1
        else:
            row, column = divmod(action, 19)
            if board.get(row, column) is not None:
                raise ValueError("Display reconstruction encountered an occupied intersection")
            board.play(row, column, color.lower())
            passes = 0
        current = "".join((board.get(r, c) or ".").upper() for r in range(19) for c in range(19))
        captured_black += sum(a == "B" and b == "." for a, b in zip(previous, current))
        captured_white += sum(a == "W" and b == "." for a, b in zip(previous, current))
        frames.append([[[point, stone] for point, stone in enumerate(current) if stone != previous[point]],
                       "W" if color == "B" else "B", captured_black, captured_white, passes])
        previous = current
    return frames


def _games(summary, worker, cohort):
    if not summary.get("schedule_file_sha256"):
        return [], "Schedule unavailable"
    schedule = _json({"path": "schedule.json", "sha256": summary["schedule_file_sha256"]},
                     base=worker, owner=worker)
    slots = schedule.get("slots")
    if (not isinstance(slots, list) or len(slots) != summary.get("scheduled_games") or len(slots) > 12
            or len({slot["id"] for slot in slots}) != len(slots)):
        raise ValueError("Match schedule differs from its saved count")
    authorization = None
    if summary.get("evidence_authorization_sha256"):
        authorization = _json({"path": "evidence_authorization.json",
            "sha256": summary["evidence_authorization_sha256"]}, base=worker, owner=worker)
        if authorization.get("schedule_file_sha256") != summary["schedule_file_sha256"]:
            raise ValueError("Match authorization names a different schedule")
    records = authorization.get("records", {}) if authorization else {}
    unverified = set(summary.get("unverified_slots", []))
    if (not set(records) <= {slot["id"] for slot in slots} or not unverified <= set(records)
            or authorization and (len(records) != summary.get("recorded_games")
                or len(records) - len(unverified) != summary.get("records_replayed"))):
        raise ValueError("Saved replay counts disagree with the authorized match records")
    if summary["status"] == "completed" and (len(records) != len(slots) or unverified):
        raise ValueError("A completed match panel lacks verified scheduled games")
    result = []
    for slot in slots:
        identity, arm = slot["id"], slot["arm"]
        if (not isinstance(identity, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", identity) is None
                or arm not in {"conditioned", "unconditioned", "human_sl"}
                or slot["candidate_color"] not in {"B", "W"}):
            raise ValueError("Unexpected match slot identity or treatment")
        game = {"id": cohort + ":" + identity, "slot_id": identity, "cohort": cohort,
                "arm": "human-reference" if arm == "human_sl" else cohort + "-" + arm,
                "opponent": slot["profile_id"], "candidate_color": slot["candidate_color"]}
        if identity not in records:
            result.append(game | {"status": "unfilled" if authorization else "unavailable", "result": "U",
                "reason": "No authorized saved record", "worker_verified": False, "moves": [],
                "frames": [], "passes": [], "sgf": None, "behavior": {"unresolved": True}})
            continue
        pins = records[identity]
        record = _json({"path": f"{identity}.json", "sha256": pins["record_sha256"]},
                       base=worker / "matches", owner=worker / "matches")
        if record.get("slot") != slot or record.get("sgf_file") != identity + ".sgf":
            raise ValueError("Match record differs from its scheduled slot or SGF")
        sgf_path = _path({"path": record["sgf_file"], "sha256": pins["sgf_sha256"]},
                         base=worker / "matches", owner=worker / "matches")
        if (sgf_path.stat().st_size > 2 * 1024**2 or file_hash(sgf_path) != pins["sgf_sha256"]
                or record.get("sgf_sha256") != pins["sgf_sha256"]):
            raise ValueError("Saved match SGF changed")
        verified = identity not in unverified
        frames = _display_frames(record["state"], record["actual_actions"]) if verified else []
        result.append(game | {key: record.get(key) for key in
            ("status", "result", "reason", "censored", "actual_actions", "behavior", "outcome_kind",
             "winner", "disagreement", "fault", "context")} | {
                "conditioning_rank": record.get("candidate_profile", {}).get("conditioning_rank"),
                "worker_verified": verified, "moves": record["state"]["moves"] if verified else [],
            "frames": frames, "passes": [{key: item.get(key) for key in ("move_index", "verdict", "reason")}
                for item in record.get("candidate_passes", [])] if verified else [],
                "sgf": sgf_path.read_text(), "record_sha256": pins["record_sha256"],
                "sgf_sha256": pins["sgf_sha256"]})
    return result, "Saved worker replay verified" if not unverified and authorization else "Replay verification incomplete"


def _costs(main, run):
    costs = main.get("costs", {})
    ledger = costs.get("ledger")
    if ledger is None:
        return {"status": "unavailable"}
    entries = ledger.get("state", {}).get("entries", [])
    for entry in entries:
        if (not isinstance(entry.get("run_id"), str)
                or any(type(entry.get(key)) not in (int, float) or not math.isfinite(entry[key]) or entry[key] < 0
                       for key in ("gpu_seconds", "cpu_seconds", "wall_seconds"))):
            raise ValueError("Invalid saved resource charge")
    charged = math.fsum(entry["gpu_seconds"] for entry in entries)
    if not math.isclose(charged, ledger.get("charged_gpu_seconds", -1), abs_tol=1e-6):
        raise ValueError("Saved GPU ledger total differs from its entries")
    current = [entry for entry in entries if entry["run_id"].startswith(run.name + "-")]
    incremental = math.fsum(entry["gpu_seconds"] for entry in current)
    return {"status": "available", "cumulative_gpu_seconds": charged, "run_gpu_seconds": incremental,
            "prior_gpu_seconds": charged - incremental, "remaining_gpu_seconds": ledger.get("remaining_gpu_seconds"),
            "cpu_seconds": costs.get("cpu_seconds"), "elapsed_seconds": costs.get("elapsed_seconds"),
            "output_bytes": costs.get("output_bytes"), "launch_utc": costs.get("launch_utc"),
            "deadline_utc": costs.get("deadline_utc"), "budget_breached": ledger.get("budget_breached"),
            "reservations": ledger.get("state", {}).get("reservations", {}), "entries": current,
            "gpu_seconds_by_stage": {entry["run_id"][len(run.name) + 1:]: entry["gpu_seconds"]
                                     for entry in current},
            "scope": "Cumulative GPU includes earlier pilots. This run includes failed jobs, device ownership waits, and tracking charges. "
                + ("CPU includes both supervisors' ten-second closing reserves and the pinned completion preparation and verifier charge."
                   if main.get("completion_status") else "CPU includes the supervisor's retained ten-second closing reserve.")}


def _completion(main, run):
    if main.get("status") != "incomplete" or main.get("completion_status") not in {"completed", "failed"}:
        raise ValueError("Remaining-check completion cannot complete the original experiment")
    original_ref = main.get("original_failure")
    if _path(original_ref, owner=run) != run / "failure.json":
        raise ValueError("Completion must bind the original failure in this run")
    original = _json(original_ref, owner=run)
    if (original.get("schema") != "expanded_e2_pilot_v1" or original.get("status") != "incomplete"
            or original.get("promotion") != "not_promoted"
            or original.get("failed_stage", {}).get("name") != REFERENCE_STAGE
            or set(original.get("stages", {})) != set(STAGES) - FINISH_STAGES - {REFERENCE_STAGE}
            or main.get("plan") != original.get("plan")
            or main.get("reason") != original.get("reason")
            or main.get("failed_stage") != original.get("failed_stage")
            or any(main["stages"].get(name) != stage for name, stage in original["stages"].items())
            or set(main["stages"]) - set(original["stages"]) - FINISH_STAGES):
        raise ValueError("Completion changed the immutable original failure or stages")
    plan = _json(main.get("completion_plan"))
    if (plan.get("version") != "experiment_config_v1" or plan.get("purpose") != "expanded-finish"
            or plan.get("inputs", {}).get("original_failure") != original_ref
            or plan.get("inputs", {}).get("partial_reference_progress") != main.get("partial_reference_progress")):
        raise ValueError("Completion plan does not bind the predecessor and optional reference progress")
    if main["completion_status"] == "completed" and not FINISH_STAGES <= main["stages"].keys():
        raise ValueError("Completed remaining checks require analysis and both match panels")
    if main.get("completion_failed_stage") is not None and (
            main["completion_status"] != "failed" or main["completion_failed_stage"].get("name") not in FINISH_STAGES):
        raise ValueError("Unexpected failure in the remaining-check phase")
    original_costs, costs = original.get("costs", {}), main.get("costs", {})
    old_entries = original_costs.get("ledger", {}).get("state", {}).get("entries", [])
    entries = costs.get("ledger", {}).get("state", {}).get("entries", [])
    if (entries[:len(old_entries)] != old_entries
            or any(type(original_costs.get(key)) is not str or costs.get(key) != original_costs[key]
                   for key in ("launch_utc", "deadline_utc"))
            or costs.get("ledger", {}).get("charged_gpu_seconds")
                != original_costs.get("ledger", {}).get("charged_gpu_seconds")
            or any(type(costs.get(key)) not in (int, float) or not math.isfinite(costs[key])
                   or costs[key] < original_costs.get(key, 0) for key in ("cpu_seconds", "elapsed_seconds"))):
        raise ValueError("Completion cannot replace earlier resource charges or reset totals")
    if _path(main.get("training_decision"), owner=run) != run / "frozen-training.json":
        raise ValueError("Completion must retain the original fixed training decision")
    return original, plan


def _failed_stage(failure, run, reason, stage_status, summaries):
    if failure is None:
        return None
    name = failure.get("name")
    directory = (ROOT / failure["directory"]).resolve()
    if name not in STAGES or directory != run / name:
        raise ValueError("Failed stage leaves the expected run directory")
    checked = {}
    allowed = {"config.json", "provenance.json", "worker/job.json", "worker/summary.json", "worker/training_summary.json",
               "worker/failure.json", "worker/training_failure.json"}
    for relative, reference in failure.get("artifacts", {}).items():
        if relative not in allowed or _path(reference, owner=directory) != directory / relative:
            raise ValueError("Unexpected failed-stage artifact")
        checked[relative] = _json(reference, owner=directory)
    failed_status = next(item for item in stage_status if item["name"] == name)
    failed_status.update(status="failed", reason=reason)
    if name.startswith("matches-") and "worker/summary.json" in checked:
        _status(checked["worker/summary.json"], "Failed match worker")
        summaries[name] = checked["worker/summary.json"]
    return {"name": name, "artifacts": checked, "pins": failure["artifacts"]}


def _partial_reference(reference, failure, common_sets, run):
    worker = run / REFERENCE_STAGE / "worker"
    if _path(reference, owner=worker) != worker / "progress.json" or failure is None:
        raise ValueError("Partial reference progress must belong to the original failed worker")
    checked, pins = failure["artifacts"], failure["pins"]
    if not {"config.json", "provenance.json", "worker/job.json"} <= checked.keys():
        raise ValueError("Partial reference requires the pinned original config, provenance and failed job")
    config, provenance, job = (checked[name] for name in ("config.json", "provenance.json", "worker/job.json"))
    if (config.get("purpose") != "baseline-evaluation" or config.get("arm") != "human-reference"
            or provenance.get("inputs") != config.get("inputs")
            or job.get("schema") != "bounded_pilot_job_v1" or job.get("status") != "failed"
            or job.get("run_id") != run.name + "-" + REFERENCE_STAGE):
        raise ValueError("Partial reference does not match its original failed evaluation job")
    command = job.get("command", [])
    for flag, artifact in (("--config-sha256", "config.json"), ("--provenance-sha256", "provenance.json")):
        if (not isinstance(command, list) or command.count(flag) != 1
                or command.index(flag) + 1 >= len(command) or command[command.index(flag) + 1] != pins[artifact]["sha256"]):
            raise ValueError("Original reference job used different config or provenance pins")
    progress = _json(reference, owner=worker)
    if (progress.get("version") != "baseline_evaluation_v1" or progress.get("arm") != "human-reference"
            or progress.get("config_sha256") != pins["config.json"]["sha256"]
            or progress.get("provenance_sha256") != pins["provenance.json"]["sha256"]
            or progress.get("status") not in {"preparing", "running", "failed", "completed"}
            or not isinstance(progress.get("sets"), dict)):
        raise ValueError("Reference progress differs from the pinned original worker")
    specs = {spec["id"]: spec for spec in config.get("parameters", {}).get("common_sets", [])}
    reports, observations = [], []
    for key, state in progress["sets"].items():
        identity = key.removesuffix("/ordinary")
        spec, common = specs.get(identity), common_sets.get(identity)
        if (key != identity + "/ordinary" or spec is None or common is None
                or common["spec"]["komi_halfpoints"] != 13 or spec.get("role") != common["spec"]["role"]
                or config["inputs"].get(spec.get("manifest_input")) != common["manifest"]
                or type(state.get("expected_rows")) is not int or state["expected_rows"] != common["rows"]
                or type(state.get("completed_rows")) is not int or not 0 <= state["completed_rows"] <= state["expected_rows"]
                or state.get("common_set_sha256") != common["manifest"]["sha256"]
                or state.get("status") not in {"preparing", "running", "failed", "completed"}):
            raise ValueError("Saved reference progress differs from its frozen whole set")
        completed = state["status"] == "completed"
        if completed:
            summary_ref = state.get("summary")
            if (state["completed_rows"] != state["expected_rows"] or not isinstance(summary_ref, dict)
                    or Path(summary_ref.get("path", "")).is_absolute()
                    or _path(summary_ref, base=worker, owner=worker) != worker / identity / "ordinary" / "summary.json"):
                raise ValueError("Only a fully completed reference set can supply a report")
            report = _json(summary_ref, base=worker, owner=worker)
            if (report.get("set") != identity or report.get("role") != spec["role"]
                    or report.get("arm") != "human-reference" or report.get("rows") != state["expected_rows"]):
                raise ValueError("Completed reference summary differs from its saved progress")
            reports.append(report)
        elif "summary" in state:
            raise ValueError("An unfinished reference set cannot supply a completed report")
        observations.append({"set": identity, "status": state["status"], "completed_rows": state["completed_rows"],
                             "expected_rows": state["expected_rows"], "report_available": completed})
    return reports, {"reference": reference, "job_status": "failed", "job_reason": job.get("reason"),
        "progress_status": progress["status"], "sets": observations,
        "scope": "Last saved worker observations, not final native shutdown counts. Only complete pinned set reports are shown; the original reference job remains failed and was not restarted."}


def collect(source, *, input_sha256=None):
    source = Path(source).resolve()
    path = source / "summary.json" if source.is_dir() else source
    if path.name not in {"summary.json", "failure.json", "completion.json"}:
        raise ValueError("Supply a completed run directory, failure.json or completion.json")
    run, pin = path.parent, input_sha256 or file_hash(path)
    main = _json({"path": str(path), "sha256": pin}, owner=run)
    status = _status(main, "Expanded run")
    if (main.get("schema") != "expanded_e2_pilot_v1"
            or path.name == "summary.json" and status != "completed"
            or path.name == "failure.json" and status not in {"incomplete", "failed"}
            or main.get("promotion") != "not_promoted"):
        raise ValueError("The selected output is not an explicit expanded comparison result")
    plan = _json(main["plan"])
    if plan.get("purpose") != "expanded-pilot" or plan.get("seed") != 20261118:
        raise ValueError("Saved plan does not describe the expanded E2 comparison")
    stages = main.get("stages")
    if not isinstance(stages, dict) or set(stages) - set(STAGES):
        raise ValueError("Unexpected expanded stage names")
    if status == "completed" and set(stages) != set(STAGES):
        raise ValueError("Completed expanded output is missing a required stage")
    original, completion_plan = _completion(main, run) if path.name == "completion.json" else (None, None)
    if original is None and any(key in main for key in ("completion_status", "completion_plan", "partial_reference_progress")):
        raise ValueError("Remaining-check evidence requires an explicit completion.json")
    finish_completed = main.get("completion_status") == "completed"
    summaries, configs, evidence, stage_status = {}, {}, [], []
    for name in STAGES:
        if name not in stages:
            stage_status.append({"name": name, "status": "unavailable", "reason": "No completed pinned stage"})
            continue
        stage = stages[name]
        directory = (ROOT / stage["directory"]).resolve()
        if directory != run / name:
            raise ValueError("Stage directory differs from its declared name")
        config = _json(stage["config"], owner=directory)
        provenance = _json(stage["provenance"], owner=directory)
        summary = _json(stage["summary"], owner=directory)
        stage_state = _status(summary, name)
        if provenance.get("inputs") != config.get("inputs"):
            raise ValueError("Saved stage config and provenance disagree")
        if (summary.get("config_file_sha256", stage["config"]["sha256"]) != stage["config"]["sha256"]
                or summary.get("provenance_sha256", stage["provenance"]["sha256"]) != stage["provenance"]["sha256"]):
            raise ValueError("Saved worker summary belongs to different stage inputs")
        if ((status == "completed" or finish_completed) and stage_state != "completed"
                and not (name.startswith("matches-") and stage_state == "incomplete")):
            raise ValueError("Completed pipeline contains an unfinished required stage")
        summaries[name], configs[name] = summary, config
        stage_status.append({"name": name, "status": stage_state, "reason": summary.get("reason"),
                             "job": stage.get("job", {}), "wandb_url": stage.get("wandb_url")})
        evidence.append({"label": name, "summary": stage["summary"], "config": stage["config"],
                         "provenance": stage["provenance"], "wandb_url": stage.get("wandb_url")})
    original_reason = original.get("reason") if original is not None else main.get("reason")
    completion_reason = main.get("completion_reason", main.get("reason")) if original is not None else None
    failure = _failed_stage(main.get("failed_stage"), run, original_reason, stage_status, summaries)
    completion_failure = _failed_stage(main.get("completion_failed_stage"), run, completion_reason, stage_status, summaries)
    common_sets, selection = {}, None
    if "preparation" in summaries:
        prepared = summaries["preparation"]
        selection = _json(prepared["selection"], owner=run / "preparation")
        for item in prepared["common_sets"]:
            spec = item["spec"]
            if spec["id"] not in SETS or spec["id"] in common_sets:
                raise ValueError("Unexpected or repeated common set")
            manifest = _json(item["manifest"], owner=run / "preparation")
            sources, requested = manifest.get("source_ids"), spec.get("source_ids")
            if (not isinstance(sources, list) or not isinstance(requested, list)
                    or any(not isinstance(source, str) or not source for source in sources + requested)):
                raise ValueError("Frozen common set needs explicit source identifiers")
            if (manifest.get("role") != spec["role"] or len(manifest["rows"]) != item["rows"]
                    or len(sources) != len(set(sources)) or len(requested) != len(set(requested))
                    or set(sources) != set(requested)):
                raise ValueError("Frozen common set differs from its preparation summary")
            common_sets[spec["id"]] = {"spec": spec, "rows": item["rows"], "coverage": item["coverage"],
                                      "manifest": item["manifest"]}
    partial_reference = None
    if main.get("partial_reference_progress") is not None:
        reports, partial_reference = _partial_reference(main["partial_reference_progress"], failure, common_sets, run)
        summaries[REFERENCE_STAGE] = {"status": "failed", "reports": reports}
    evaluations, predictions = {identity: {} for identity in SETS}, {}
    for arm in (*ARMS, "human-reference"):
        name = "evaluation-" + arm
        if name not in summaries:
            continue
        worker = run / name / "worker"
        for report in summaries[name].get("reports", []):
            _status(report, "Recorded-position report", ("completed",))
            identity = report["set"]
            if (identity not in common_sets or report["variant"] != "ordinary" or arm in evaluations[identity]
                    or report["common_set_sha256"] != common_sets[identity]["manifest"]["sha256"]
                    or report["rows"] != common_sets[identity]["rows"]
                    or arm == "human-reference" and common_sets[identity]["spec"]["komi_halfpoints"] != 13):
                raise ValueError("Evaluation differs from the frozen comparison set")
            artifacts, output = report["artifacts"], {}
            for kind in ("metrics", "sampling", "timing"):
                output[kind] = _json(artifacts[kind + ".json"], base=worker, owner=worker)
            if output["metrics"].get("expected_outputs") != report["rows"]:
                raise ValueError("Metric row count differs from the evaluated common set")
            ref = artifacts["predictions.npz"]
            predicted = _path(ref, base=worker, owner=worker)
            if predicted.suffix != ".npz" or file_hash(predicted) != ref["sha256"]:
                raise ValueError("Saved prediction artifact changed")
            predictions[identity, arm] = {"path": str(predicted), "sha256": ref["sha256"]}
            output["metrics"].pop("by_komi", None)
            evaluations[identity][arm] = output | {"rows": report["rows"], "coverage": report.get("coverage"),
                "out_of_training_support_komi_halfpoints": report.get("out_of_training_support_komi_halfpoints", []),
                "prediction": predictions[identity, arm]}
    paired = {}
    if "analysis" in summaries:
        for identity, comparisons in summaries["analysis"]["comparisons"].items():
            if identity not in SETS:
                raise ValueError("Paired report names an unknown common set")
            paired[identity] = {}
            for contrast, item in comparisons.items():
                report = _json(item["report"], owner=run / "analysis")
                if report.get("kind") != "candidate_minus_control" or report.get("overall") != item["overall"]:
                    raise ValueError("Paired analysis summary differs from its saved report")
                paired[identity][contrast] = report
        for spec in configs["analysis"]["parameters"]["common_sets"]:
            for arm, input_name in spec["prediction_inputs"].items():
                reference = configs["analysis"]["inputs"][input_name]
                expected = predictions.get((spec["id"], arm))
                if expected is None or reference["sha256"] != expected["sha256"] or _path(reference) != Path(expected["path"]):
                    raise ValueError("Paired analysis used different saved predictions")
    training = {arm: summaries[arm] for arm in ARMS if arm in summaries}
    probes = {arm: summaries["probe-" + arm] for arm in ARMS if "probe-" + arm in summaries}
    matched = len(training) == 4 and all(summary["status"] == "completed" for summary in training.values())
    if matched:
        matched = all(len({summary.get(key) for summary in training.values()}) == 1
                      and all(summary.get(key) is not None for summary in training.values())
                      for key in ("initial_tensor_sha256", "completed_updates", "presented_examples"))
        matched = matched and all(summary.get("checkpoint_selection") == "fixed_final_update"
            and type(summary.get("completed_updates")) is int and summary["completed_updates"] > 0
            and summary["presented_examples"] == summary["completed_updates"] * 128
            for summary in training.values())
        for cohort in ("baseline", "combined"):
            left, right = (training[cohort + "-" + arm] for arm in ("conditioned", "unconditioned"))
            matched = matched and left.get("row_keys_sha256") == right.get("row_keys_sha256") and bool(left.get("update_metrics"))
            matched = matched and [block["order_sha256"] for block in left.get("update_metrics", [])] == [
                block["order_sha256"] for block in right.get("update_metrics", [])]
    if ((status == "completed" and main.get("matched_training") is not True)
            or (status == "completed" or original is not None) and not matched):
        raise ValueError("Completed comparison lacks matched training evidence")
    if original is not None and any(configs[arm].get("inputs", {}).get("training_decision") != main["training_decision"]
                                    for arm in ARMS):
        raise ValueError("Completion changed the original training decision")
    decision = _json(main["training_decision"], owner=run) if main.get("training_decision") else None
    panels, games = {}, []
    for cohort in ("baseline", "combined"):
        name = "matches-" + cohort
        if name in summaries:
            panel_games, replay = _games(summaries[name], run / name / "worker", cohort)
            games.extend(panel_games)
            panels[cohort] = {"summary": summaries[name], "replay_status": replay}
    return {"schema": "expanded_e2_review_v1", "bundle_id": pin, "input": {"path": str(path), "sha256": pin},
        "input_pin_scope": "Caller supplied" if input_sha256 else "Observed during export; downstream references checked against their saved pins",
        "status": status, "reason": original_reason, "plan": main["plan"], "plan_parameters": plan["parameters"],
        "completion_status": main.get("completion_status"), "completion_reason": completion_reason,
        "completion_plan": main.get("completion_plan"), "completion_parameters": completion_plan.get("parameters") if completion_plan else None,
        "original_failure": main.get("original_failure"), "partial_reference": partial_reference,
        "arms": list(ARMS), "sets": list(SETS), "common_sets": common_sets, "selection": selection,
        "evaluations": evaluations, "paired": paired, "training": training, "probes": probes,
        "training_decision": decision, "matched_training": matched, "costs": _costs(main, run),
        "stages": stage_status, "failure": failure, "completion_failure": completion_failure,
        "panels": panels, "games": games, "evidence": evidence,
        "promotion": "not_promoted", "feedback_recorded": False,
        "limitations": ["One training seed. This experiment cannot promote an opponent profile or claim a human rank.",
            "The addition changes both the source population and its komi mix; it does not isolate a komi effect.",
            "6.5-only models on 0.5-komi positions are outside their training support.",
            "Human SL is evaluated at 6.5 komi only, with its documented history and era/time defaults; training overlap is unknown.",
            "Missing ranks and unsupported uncertainty intervals remain unavailable or inconclusive. No pooled substitute replaces the declared aggregate.",
            "The two raw-play panels schedule 24 games in total against two weak policies. Repeated Human SL controls are not independent extra strength evidence.",
            "Display boards reconstruct pinned moves with capture handling only. Saved worker replay verification is separate; this export performs no legality, scoring, or referee checks.",
            "No inference runs here. Raw datasets and checkpoint contents are not opened. Confirmation and final targets remain sealed."]}


def render(data):
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    encoded = encoded.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    template = TEMPLATE.read_text()
    if template.count("__EXPANDED_DATA_JSON__") != 1:
        raise ValueError("The review template needs exactly one data marker")
    page = template.replace("__EXPANDED_DATA_JSON__", encoded)
    if len(page.encode()) > MAX_PAGE_BYTES:
        raise ValueError("Review page exceeds its 32 MiB allowance")
    return page


def export(source, output, *, input_sha256=None):
    started, cpu = time.monotonic(), time.process_time()
    output = Path(output)
    manifest_path = output.with_suffix(".manifest.json")
    if output.exists() or output.is_symlink() or manifest_path.exists() or manifest_path.is_symlink():
        raise ValueError("Review output and manifest paths must be unused")
    data = collect(source, input_sha256=input_sha256)
    page = render(data)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(page)
    manifest = {"schema": "expanded_e2_review_export_v1", "html": str(output.resolve()),
        "html_sha256": file_hash(output), "source": data["input"], "template_sha256": file_hash(TEMPLATE),
        "collector_sha256": file_hash(Path(__file__)), "status": data["status"], "completion_status": data["completion_status"],
        "games": len(data["games"]),
        "cpu_seconds": time.process_time() - cpu, "wall_seconds": time.monotonic() - started,
        "gpu_seconds": 0, "raw_dataset_reads": 0, "checkpoint_reads": 0, "inference_calls": 0}
    with manifest_path.open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Completed run directory, failure.json or completion.json")
    parser.add_argument("output", type=Path, help="Unused HTML output path")
    parser.add_argument("--input-sha256")
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.output, input_sha256=args.input_sha256), indent=2))
