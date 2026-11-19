"""Finish only the original E2 analysis and games after the reference CPU cap."""

import argparse
from datetime import datetime, timezone
import fcntl
from pathlib import Path
import signal

from kenggo.base_pilot import _config, _learned_bundle, _ref
from kenggo.checkpoints import file_hash, read_json, save_json
from kenggo.expanded_pilot import (ARMS, COHORTS, ROOT, Session, _match_config,
                                  matched_training, validate_plan)
from kenggo.resource_ledger import BudgetLedger
from kenggo.tracking import capture_provenance, validate_config


JOBS = ("analysis", "matches-baseline", "matches-combined")
OLD_STAGES = {"preparation", *ARMS, *("probe-" + arm for arm in ARMS),
              *("evaluation-" + arm for arm in ARMS)}


def _load(ref):
    return read_json(ROOT / ref["path"], ref["sha256"])


def validate_finish(config, output, ledger_state):
    validate_config(config)
    p, refs = config["parameters"], config["inputs"]
    if (config["purpose"] != "expanded-finish" or config["arm"] != "shared"
            or set(p) != {"jobs", "output", "ledger", "ledger_sha256", "deadline_utc",
                          "verification_directory", "preparation_cpu_reserve_seconds"}
            or p["jobs"] != list(JOBS) or p["preparation_cpu_reserve_seconds"] != 120):
        raise ValueError("Only the original three remaining jobs may run")
    plan = _load(refs["original_plan"])
    original = validate_plan(plan)
    origin, failure = _load(refs["original_session"]), _load(refs["original_failure"])
    output = Path(output).resolve()
    for key in ("version", "entity", "project", "base_url", "comparison", "seed", "budget_id"):
        if config[key] != plan[key]:
            raise ValueError("Completion must retain the original experiment identity")
    if (output != (ROOT / original["output"]).resolve() or p["output"] != original["output"]
            or not output.is_relative_to(ROOT / "runs") or p["ledger"] != original["ledger"]
            or (ROOT / refs["original_failure"]["path"]).resolve() != output / "failure.json"
            or (ROOT / refs["original_session"]["path"]).resolve() != output / "session.json"
            or origin["schema"] != "expanded_e2_session_v1" or origin["plan"] != refs["original_plan"]
            or origin["limits"] != original["limits"] or p["deadline_utc"] != origin["deadline_utc"]
            or failure["plan"] != refs["original_plan"] or failure["status"] != "incomplete"
            or set(failure["stages"]) != OLD_STAGES
            or failure["failed_stage"]["name"] != "evaluation-human-reference"):
        raise ValueError("This command requires the unchanged capped-reference experiment")
    for key in ("launch_utc", "deadline_utc"):
        if failure["costs"][key] != origin[key]:
            raise ValueError("The original production clock changed")
    if (ledger_state != failure["costs"]["ledger"]["state"] or ledger_state["reservations"]
            or ledger_state["budget_breached"]
            or ledger_state["entries"][:len(origin["ledger_before"]["state"]["entries"])]
                != origin["ledger_before"]["state"]["entries"]):
        raise ValueError("The original cumulative ledger must remain unchanged and settled")
    failed = failure["failed_stage"]
    job = _load(failed["artifacts"]["worker/job.json"])
    if job["status"] != "failed" or job["reason"] != "cpu_budget":
        raise ValueError("Only the recorded reference CPU cap permits this completion")
    for name in ("data", "split", "rules"):
        if refs[name] != plan["inputs"][name]:
            raise ValueError("Completion cannot change the original data contract")
    handoff = _load(refs["verification_handoff"])
    review = (ROOT / p["verification_directory"]).resolve()
    if (review == output or review.is_relative_to(output) or output.is_relative_to(review)
            or not review.is_relative_to(ROOT / "runs")
            or (ROOT / refs["verification_handoff"]["path"]).resolve() != review / "handoff.json"
            or handoff["verification_status"] != "passed_available_evidence"
            or handoff["verification_costs"]["gpu_seconds"] != 0
            or handoff["production_failure"]["sha256"] != refs["original_failure"]["sha256"]
            or Path(handoff["production_failure"]["path"]).resolve() != output / "failure.json"):
        raise ValueError("The completed saved-evidence verification must be pinned")
    if (ROOT / refs["partial_reference_progress"]["path"]).resolve() != output / "evaluation-human-reference/worker/progress.json":
        raise ValueError("Partial reference progress must belong to the failed job")
    _load(refs["partial_reference_progress"])
    for name in (*JOBS, "completion-session.json", "completion.json", "summary.json"):
        if (output / name).exists():
            raise ValueError("Remaining-check completion is one use only")
    return plan, origin, failure, handoff


def saved_work(plan, failure, output):
    summaries = {}
    for name, stage in failure["stages"].items():
        directory = output / name
        if Path(stage["directory"]).resolve() != directory:
            raise ValueError("Saved stage moved outside its original directory")
        for key in ("config", "provenance", "summary"):
            if not (ROOT / stage[key]["path"]).resolve().is_relative_to(directory):
                raise ValueError("Saved evidence must stay within its original stage")
        config, provenance = _load(stage["config"]), _load(stage["provenance"])
        summary = _load(stage["summary"])
        if (provenance["inputs"] != config["inputs"] or stage["job"]["status"] != "completed"
                or _load(_ref(directory / "worker/job.json")) != stage["job"]
                or summary["status"] != "completed"):
            raise ValueError("All retained training and learned evaluations must be complete")
        summaries[name] = summary
    decision_ref = _ref(output / "frozen-training.json")
    decision = _load(decision_ref)
    for arm in ARMS:
        if _load(failure["stages"][arm]["config"])["inputs"]["training_decision"] != decision_ref:
            raise ValueError("The fixed training decision changed")
    matched_training({arm: summaries[arm] for arm in ARMS}, expected_updates=decision["updates"])
    return summaries, decision_ref


def finish_jobs(plan, plan_ref, finish_ref, summaries, session):
    p = plan["parameters"]
    bases = {cohort: data["inputs"] | {"pilot_plan": plan_ref, "completion_plan": finish_ref}
             for cohort, data in p["datasets"].items()}
    refs, specs = dict(bases["combined"]), []
    for item in summaries["preparation"]["common_sets"]:
        spec = item["spec"]
        key = "common_" + spec["id"]
        refs[key] = item["manifest"]
        mapping = {}
        for arm in ARMS:
            reports = [report for report in summaries["evaluation-" + arm]["reports"]
                       if report["set"] == spec["id"] and report["variant"] == "ordinary"]
            if len(reports) != 1:
                raise ValueError("Every learned arm needs exactly one saved report per set")
            ref = reports[0]["artifacts"]["predictions.npz"]
            worker = Path(session.stages["evaluation-" + arm]["directory"]) / "worker"
            path = (worker / ref["path"]).resolve()
            if not path.is_relative_to(worker) or file_hash(path) != ref["sha256"]:
                raise ValueError("Saved evaluation predictions changed")
            input_key = spec["id"] + "_" + arm
            refs[input_key], mapping[arm] = _ref(path), input_key
        specs.append({key: spec[key] for key in ("id", "role", "source_ids", "aggregate_ranks")}
                     | {"manifest_input": "common_" + spec["id"], "prediction_inputs": mapping})
    config = _config(plan, purpose="expanded-analysis", arm="shared",
                     parameters={"common_sets": specs}, inputs=refs)
    session.stage("analysis", config, "analysis", module="kenggo.expanded_jobs")
    bundles = {arm: _learned_bundle(session.stages[arm], summaries[arm],
        {"inputs": bases[arm.split('-')[0]], "parameters": p}, arm.endswith("-conditioned"))
        for arm in ARMS}
    for cohort in COHORTS:
        config = _match_config(plan, bases[cohort], cohort, bundles, session.stages)
        session.stage("matches-" + cohort, config, "matches", module="kenggo.baseline_matches")


def run(config_path, config_pin, output):
    config = read_json(config_path, config_pin)
    output = Path(output).resolve()
    p, refs = config["parameters"], config["inputs"]
    lock_path = ROOT / ".cache/e2-expanded-pilot.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if file_hash(ROOT / p["ledger"]) != p["ledger_sha256"]:
            raise ValueError("The ledger changed before remaining-check completion")
        ledger = BudgetLedger.open(ROOT / p["ledger"], budget_id=config["budget_id"], gpu_seconds=86400)
        plan, origin, failure, handoff = validate_finish(config, output, ledger.snapshot()["state"])
        remaining = ((datetime.fromisoformat(origin["deadline_utc"]) - datetime.now(timezone.utc)).total_seconds()
                     - origin["limits"]["publication_seconds"])
        if remaining <= 120:
            raise ValueError("The original finishing cutoff has passed")
        finish_ref = _ref(config_path)
        save_json(output / "completion-session.json", {"schema": "expanded_e2_completion_session_v1",
            "plan": finish_ref, "original_session": refs["original_session"],
            "launch_utc": datetime.now(timezone.utc).isoformat(), "deadline_utc": origin["deadline_utc"]})
        overhead = (handoff["verification_costs"]["charged_cpu_seconds_including_reserve"]
                    + handoff["production_costs"]["wait_cpu_seconds"] + p["preparation_cpu_reserve_seconds"])
        charge_id = output.name + "-saved-verification"
        ledger.reserve(charge_id, estimate_gpu_seconds=0, shutdown_gpu_seconds=0,
                       metadata={"verification_handoff": refs["verification_handoff"],
                                 "preparation_cpu_reserve_seconds": p["preparation_cpu_reserve_seconds"]})
        ledger.record(charge_id, allocations=[], outcome="completed", arm_shares={"shared": 1},
                      cpu_seconds=overhead, wall_seconds=handoff["verification_costs"]["wall_seconds"])
        session = Session(output, plan, ledger, origin=origin, phase="completion",
                          extra_output_dirs=[ROOT / p["verification_directory"]])
        session.stages = dict(failure["stages"])
        previous = signal.getsignal(signal.SIGALRM)
        def expired(signum, frame):
            raise TimeoutError("The original expanded-comparison finishing cutoff was reached")
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, max(.001,
            (session.deadline - datetime.now(timezone.utc)).total_seconds() - session.limits["publication_seconds"]))
        result = dict(failure, original_failure=refs["original_failure"], completion_plan=finish_ref,
                      completion_status="failed", completion_reason="completion_did_not_finish",
                      training_decision=refs["training_decision"],
                      partial_reference_progress=refs["partial_reference_progress"], stages=session.stages)
        try:
            source, patch = capture_provenance(ROOT, refs)
            save_json(output / "completion-source.json", source)
            (output / "completion-source.patch").write_bytes(patch)
            capture_provenance(ROOT, plan["inputs"])
            for data in plan["parameters"]["datasets"].values():
                capture_provenance(ROOT, data["inputs"])
            summaries, decision_ref = saved_work(plan, failure, output)
            if decision_ref != refs["training_decision"]:
                raise ValueError("The original training decision pin changed")
            result.update(training_decision=decision_ref, matched_training=True)
            finish_jobs(plan, refs["original_plan"], finish_ref, summaries, session)
            current, _ = capture_provenance(ROOT, refs)
            if current["source"] != source["source"]:
                raise ValueError("Runtime source changed during remaining-check completion")
            result.update(completion_status="completed", completion_reason=None)
        except BaseException as error:
            result["completion_reason"] = type(error).__name__ + ": " + str(error)
            if session.failed_stage is not None:
                result["completion_failed_stage"] = session.failed_stage
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
            session.close(result["completion_status"], result["completion_reason"])
            result["costs"] = session.snapshot()
            save_json(output / "completion.json", result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config-sha256", required=True)
    args = parser.parse_args()
    run(args.config, args.config_sha256, args.output)
