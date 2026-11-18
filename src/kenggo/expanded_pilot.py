"""One-use, bounded four-arm comparison of the two frozen E2 populations."""

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import time

from kenggo.base_pilot import _config, _learned_bundle, _ref, _stage
from kenggo.checkpoints import file_hash, read_json, save_json
from kenggo.pilot_jobs import _output_bytes, _usage
from kenggo.resource_ledger import BudgetLedger
from kenggo.tracking import capture_provenance, validate_config


ROOT = Path(__file__).resolve().parents[2]
COHORTS = ("baseline", "combined")
TREATMENTS = ("conditioned", "unconditioned")
ARMS = tuple(f"{cohort}-{arm}" for cohort in COHORTS for arm in TREATMENTS)
DATA_KEYS = {"data", "canonical", "split", "examples", "normalizer", "profile", "rules",
             "cpu_environment", "cuda_environment"}
QUANTUM = 15_625


def validate_plan(plan):
    validate_config(plan)
    p = plan["parameters"]
    if (plan["purpose"] != "expanded-pilot" or plan["budget_id"] != "e0-sl-pilot-v1"
            or plan["seed"] != 20261118 or set(p["datasets"]) != set(COHORTS)
            or p["architecture"] != {"blocks": 8, "channels": 128}
            or p["selection"] != {"games_per_rank": 2, "max_games_per_set": 32,
                                   "max_rows_per_set": 12000}
            or p["probe_updates"] != 256 or p["max_updates"] != 187500
            or p["limits"] != {"wall_seconds": 86400, "cpu_seconds": 115200,
                               "output_bytes": 16 * 1024**3, "publication_seconds": 900}
            or p["ledger"] != "runs/e2-sl-budget-2026-11-14/resources.json"
            or not isinstance(p["prior_ledger_sha256"], str)
            or len(p["prior_ledger_sha256"]) != 64):
        raise ValueError("Expanded comparison differs from its approved scope or budget")
    for cohort, count in (("baseline", 1_000_000), ("combined", 2_000_000)):
        data = p["datasets"][cohort]
        if (set(data) != {"inputs", "training_rows", "training_komi_halfpoints"}
                or set(data["inputs"]) != DATA_KEYS or data["training_rows"] != count
                or data["training_komi_halfpoints"] != ([13] if cohort == "baseline" else [1, 13])):
            raise ValueError("Both frozen training populations must be explicit")
    required_jobs = {"preparation", "probe", "training", "learned_evaluation", "human_evaluation",
                     "matches", "analysis"}
    if set(p["job_limits"]) != required_jobs:
        raise ValueError("Every production phase requires an explicit allowance")
    ceilings = dict(preparation=3600, probe=900, training=9000, learned_evaluation=3600,
                    human_evaluation=9000, matches=5400, analysis=1800)
    from kenggo.pilot_jobs import JobLimits
    for key, values in p["job_limits"].items():
        limits = JobLimits(**values)
        if (limits.wall_seconds > ceilings[key] or limits.cpu_seconds > ceilings[key]
                or limits.output_bytes > 2 * 1024**3 or limits.shutdown_seconds != 10):
            raise ValueError("A job exceeds its frozen ceiling")
    return p


def freeze_training(probes, jobs, *, wall_seconds, maximum=187500):
    if set(probes) != set(ARMS) or set(jobs) != set(ARMS):
        raise ValueError("All four cost probes must complete before freezing training")
    rates, overheads = [], []
    initial = set()
    for arm in ARMS:
        summary, job = probes[arm], jobs[arm]
        blocks = summary["update_metrics"]
        if (summary["status"] != "completed" or summary["completed_updates"] != 256
                or summary["presented_examples"] != 256 * 128 or len(blocks) != 6
                or blocks[0]["phase"] != "warmup" or blocks[0]["updates"] != 6
                or any(b["phase"] != "measured" or b["updates"] != 50 for b in blocks[1:])
                or job["status"] != "completed"):
            raise ValueError("Probe timing needs six warmup and five complete measured blocks")
        measured = [b["seconds_per_update"] for b in blocks[1:]]
        if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in measured):
            raise ValueError("Probe times must be finite and positive")
        cursor = 0
        for block in blocks:
            if (block["start_step"] != cursor or block["next_step"] != cursor + block["updates"]
                    or block["presented_examples"] != 128 * block["updates"]
                    or not math.isfinite(block["seconds"]) or block["seconds"] <= 0
                    or not math.isclose(block["seconds"], block["seconds_per_update"] * block["updates"],
                                        rel_tol=1e-9)):
                raise ValueError("Probe block timing or work counts are inconsistent")
            cursor = block["next_step"]
        job_wall = job["costs"]["wall_seconds"]
        if not math.isfinite(job_wall) or job_wall < sum(b["seconds"] for b in blocks):
            raise ValueError("Probe job time must cover every timed update")
        rates.extend(measured)
        overheads.append(job_wall - sum(b["seconds"] for b in blocks))
        initial.add(summary["initial_tensor_sha256"])
    if len(initial) != 1:
        raise ValueError("Probe model initializations differ")
    worst, setup = max(rates), max(overheads)
    # Main training saves one more checkpoint than a probe; keep a fixed reserve too.
    overhead = max(300, 2 * setup)
    updates = min(maximum // QUANTUM, int(max(0, wall_seconds - overhead - 10) * .8 / worst) // QUANTUM) * QUANTUM
    if updates < QUANTUM:
        raise ValueError("Measured cost cannot fit one matched combined-data epoch")
    return {"schema": "expanded_training_decision_v1", "updates": updates,
            "presentations_per_arm": updates * 128, "batch_size": 128,
            "worst_measured_seconds_per_update": worst, "largest_observed_overhead_seconds": setup,
            "reserved_overhead_seconds": overhead, "work_fraction": .8,
            "per_arm_wall_seconds": wall_seconds, "maximum_updates": maximum,
            "save_updates": [0, updates // 2, updates], "checkpoint_selection": "fixed_final_update",
            "probe_weights_reused": False, "development_scores_used": False}


def matched_training(summaries, *, expected_updates=None):
    if set(summaries) != set(ARMS):
        raise ValueError("All four main training results are required")
    for summary in summaries.values():
        updates = summary["completed_updates"]
        if (summary["status"] != "completed" or type(updates) is not int or updates < 1
                or summary["presented_examples"] != updates * 128
                or expected_updates is not None and updates != expected_updates):
            raise ValueError("Main training did not finish the frozen full-batch work")
    for key in ("initial_tensor_sha256", "completed_updates", "presented_examples"):
        if len({s[key] for s in summaries.values()}) != 1:
            raise ValueError(f"Main training is not matched: {key}")
    for cohort in COHORTS:
        left, right = (summaries[f"{cohort}-{arm}"] for arm in TREATMENTS)
        if (left["row_keys_sha256"] != right["row_keys_sha256"]
                or [b["order_sha256"] for b in left["update_metrics"]]
                != [b["order_sha256"] for b in right["update_metrics"]]):
            raise ValueError("Rank treatments used different training order")
    return True


class Session:
    def __init__(self, output, plan, ledger):
        self.output, self.plan, self.ledger = output, plan, ledger
        self.started, self.cpu_started = time.monotonic(), time.process_time()
        self.tree_cpu_started = _usage(os.getpid())[0]
        self.stage_parent_cpu = self.stage_wall = 0.0
        self.closed = False
        self.limits = plan["parameters"]["limits"]
        self.utc = datetime.now(timezone.utc)
        self.deadline = self.utc + timedelta(seconds=self.limits["wall_seconds"])
        self.initial = ledger.snapshot()
        self.old_ids = {x["run_id"] for x in self.initial["state"]["entries"]}
        self.root_id = output.name + "-supervisor"
        ledger.reserve(self.root_id, estimate_gpu_seconds=0, shutdown_gpu_seconds=0,
                       metadata={"purpose": "expanded_supervisor", "launch_utc": self.utc.isoformat(),
                                 "final_write_cpu_reserve_seconds": 10})
        self.stages = {}
        self.failed_stage = None

    def _unreported_cpu(self, new_entries):
        own = time.process_time() - self.cpu_started - self.stage_parent_cpu
        tree = _usage(os.getpid())[0] - self.tree_cpu_started - sum(x["cpu_seconds"] for x in new_entries)
        return max(0, own, tree)

    def snapshot(self):
        state = self.ledger.snapshot()
        new = [x for x in state["state"]["entries"] if x["run_id"] not in self.old_ids]
        return {"launch_utc": self.utc.isoformat(), "deadline_utc": self.deadline.isoformat(),
                "elapsed_seconds": time.monotonic() - self.started,
                "cpu_seconds": sum(x["cpu_seconds"] for x in new)
                    + (0 if self.closed else self._unreported_cpu(new) + 10),
                "output_bytes": _output_bytes(self.output), "ledger": state}

    def stage(self, name, config, kind, *, module, device=None, register=False):
        state = self.snapshot()
        limits = dict(self.plan["parameters"]["job_limits"][kind])
        available_wall = min(self.limits["wall_seconds"] - state["elapsed_seconds"],
                             (self.deadline - datetime.now(timezone.utc)).total_seconds())
        available_wall -= self.limits["publication_seconds"] + 120
        available_cpu = self.limits["cpu_seconds"] - state["cpu_seconds"] - 120
        available_bytes = self.limits["output_bytes"] - state["output_bytes"] - 64 * 1024**2
        if (available_wall < limits["wall_seconds"] + limits["shutdown_seconds"]
                or available_cpu < limits["cpu_seconds"] or available_bytes < limits["output_bytes"]):
            raise RuntimeError("Remaining production allowance cannot fund the next complete job")
        if state["ledger"]["budget_breached"]:
            raise RuntimeError("Existing ledger breach blocks further jobs")
        directory = self.output / name
        save_json(self.output / "progress.json", {"status": "running", "active_stage": name,
            "stages": self.stages, "costs": state}, replace=True)
        try:
            result = _stage(self.output, name, config, limits, self.ledger, module=module,
                            device=device, register=register)
        except BaseException:
            artifacts = ("config.json", "provenance.json", "worker/job.json", "worker/summary.json",
                         "worker/failure.json", "worker/training_summary.json", "worker/training_failure.json")
            self.failed_stage = {"name": name, "directory": str(directory), "artifacts": {
                filename: _ref(directory / filename) for filename in artifacts
                if (directory / filename).is_file()}}
            raise
        finally:
            entries = self.ledger.snapshot()["state"]["entries"]
            overhead_id = self.output.name + "-" + name + "-tracking"
            overhead = next((entry for entry in entries if entry["run_id"] == overhead_id), None)
            if overhead is not None:
                self.stage_parent_cpu += overhead["cpu_seconds"]
                self.stage_wall += overhead["wall_seconds"]
            job_path = directory / "worker/job.json"
            if job_path.is_file():
                job = json.loads(job_path.read_text())
                if "costs" in job and "parent_cpu_seconds" in job:
                    self.stage_parent_cpu += job["parent_cpu_seconds"]
                    self.stage_wall += job["costs"]["wall_seconds"]
        self.stages[name] = result[0]
        save_json(self.output / "progress.json", {"status": "running", "last_stage": name,
            "stages": self.stages, "costs": self.snapshot()}, replace=True)
        return result

    def close(self, status, reason):
        entries = self.ledger.snapshot()["state"]["entries"]
        new = [entry for entry in entries if entry["run_id"] not in self.old_ids]
        self.ledger.record(self.root_id, allocations=[], outcome=status, arm_shares={"shared": 1},
            cpu_seconds=self._unreported_cpu(new) + 10,
            wall_seconds=max(0, time.monotonic() - self.started - self.stage_wall), reason=reason)
        self.closed = True


def _training_config(plan, base, cohort, arm, updates, saves):
    data = plan["parameters"]["datasets"][cohort]
    parameters = {"recipe": "expanded_e2_training_v1", "inference_contract": "komi_comparison_v1",
        "training_komi_halfpoints": data["training_komi_halfpoints"], "updates": updates,
        "save_updates": saves, "training_rows": data["training_rows"], "batch_size": 128,
        "learning_rate": .001, "clock_dropout": .1, "device": "cuda", "torch_threads": 1,
        "model": plan["parameters"]["architecture"] | {"rank_conditioned": arm == "conditioned"},
        "max_artifact_bytes": 256 * 1024**2}
    return _config(plan, purpose="baseline-training", arm=arm, parameters=parameters, inputs=base)


def _match_config(plan, base, cohort, bundles, stages):
    p = plan["parameters"]
    refs = base | {name: ref for name, ref in plan["inputs"].items()
                   if name.startswith(("human_", "referee_"))}
    for arm in TREATMENTS:
        label = f"{cohort}-{arm}"
        refs |= {arm + "_checkpoint": _ref(bundles[label]["checkpoint_manifest"]),
                 arm + "_config": stages[label]["config"], arm + "_provenance": stages[label]["provenance"]}
    parameters = dict(p["match_parameters"])
    parameters.update(recipe="expanded_e2_matches_v1", bootstrap_seed=plan["seed"],
                      learned_artifacts={arm: bundles[f"{cohort}-{arm}"] for arm in TREATMENTS})
    return _config(plan, purpose="baseline-matches", arm="shared", parameters=parameters, inputs=refs)


def execute(plan, plan_ref, output, session):
    p = plan["parameters"]
    bases = {cohort: data["inputs"] | {"pilot_plan": plan_ref} for cohort, data in p["datasets"].items()}
    base = bases["combined"]
    config = _config(plan, purpose="expanded-preparation", arm="shared",
                     parameters={"selection": p["selection"]}, inputs=base)
    _, prepared = session.stage("preparation", config, "preparation", module="kenggo.expanded_jobs")
    specs, common_refs = [], {}
    for item in prepared["common_sets"]:
        name = "common_" + item["spec"]["id"]
        specs.append({key: item["spec"][key] for key in ("id", "role", "source_ids", "aggregate_ranks")}
                     | {"manifest_input": name})
        common_refs[name] = item["manifest"]
    probes, probe_jobs, probe_refs = {}, {}, {}
    for cohort in COHORTS:
        for arm in TREATMENTS:
            label = f"{cohort}-{arm}"
            config = _training_config(plan, bases[cohort], cohort, arm, p["probe_updates"], [0, p["probe_updates"]])
            stage, summary = session.stage("probe-" + label, config, "probe",
                module="kenggo.baseline_training", device=p["device"], register=True)
            probes[label], probe_jobs[label], probe_refs[label] = summary, stage["job"], stage["summary"]
    decision = freeze_training(probes, probe_jobs, wall_seconds=p["job_limits"]["training"]["wall_seconds"],
                               maximum=p["max_updates"])
    decision["probe_summaries"] = probe_refs
    decision["prepared_common_sets"] = common_refs
    save_json(output / "frozen-training.json", decision)
    decision_ref = _ref(output / "frozen-training.json")
    bundles, training = {}, {}
    for cohort in COHORTS:
        for arm in TREATMENTS:
            label = f"{cohort}-{arm}"
            refs = bases[cohort] | {"training_decision": decision_ref}
            config = _training_config(plan, refs, cohort, arm, decision["updates"], decision["save_updates"])
            stage, training[label] = session.stage(label, config, "training",
                module="kenggo.baseline_training", device=p["device"], register=True)
            bundle_plan = {"inputs": refs, "parameters": p}
            bundles[label] = _learned_bundle(stage, training[label], bundle_plan, arm == "conditioned")
    matched_training(training, expected_updates=decision["updates"])
    evaluations = {}
    for cohort in COHORTS:
        for arm in TREATMENTS:
            label, bundle = f"{cohort}-{arm}", bundles[f"{cohort}-{arm}"]
            refs = base | common_refs | {
                "checkpoint": _ref(bundle["checkpoint_manifest"]),
                "training_config": session.stages[label]["config"],
                "training_provenance": session.stages[label]["provenance"],
                "training_examples": bases[cohort]["examples"],
                "training_normalizer": bases[cohort]["normalizer"]}
            parameters = {"common_sets": specs, "common_input_mode": "raw_komi_comparison_v1",
                "batch_size": 32, "samples_per_position": 32, "unknown_rank_diagnostic": False,
                "artifacts": bundle}
            config = _config(plan, purpose="baseline-evaluation", arm=arm, parameters=parameters, inputs=refs)
            _, evaluations[label] = session.stage("evaluation-" + label, config, "learned_evaluation",
                                                   module="kenggo.baseline_evaluation")
    human_specs = [spec for spec in specs if next(item for item in prepared["common_sets"]
                     if item["spec"]["id"] == spec["id"])["spec"]["komi_halfpoints"] == 13]
    human_rows = sum(item["rows"] for item in prepared["common_sets"]
                     if item["spec"]["komi_halfpoints"] == 13)
    if not 0 < human_rows <= p["reference_budget"]["queries"]:
        raise ValueError("Frozen Human SL positions exceed the reference query ceiling")
    refs = base | {spec["manifest_input"]: common_refs[spec["manifest_input"]] for spec in human_specs}
    refs |= {name: ref for name, ref in plan["inputs"].items() if name.startswith("human_")}
    parameters = {"common_sets": human_specs, "common_input_mode": "raw_komi_comparison_v1",
        "batch_size": 32, "samples_per_position": 32, "unknown_rank_diagnostic": False,
        "human_paths": p["human_paths"], "reference_budget": dict(p["reference_budget"], queries=human_rows)}
    config = _config(plan, purpose="baseline-evaluation", arm="human-reference", parameters=parameters, inputs=refs)
    session.stage("evaluation-human-reference", config, "human_evaluation", module="kenggo.baseline_evaluation")
    for cohort in COHORTS:
        config = _match_config(plan, bases[cohort], cohort, bundles, session.stages)
        session.stage("matches-" + cohort, config, "matches", module="kenggo.baseline_matches")
    refs, analysis_specs = base | common_refs, []
    for spec in specs:
        mapping = {}
        for label in ARMS:
            report = next(item for item in evaluations[label]["reports"]
                          if item["set"] == spec["id"] and item["variant"] == "ordinary")
            ref = report["artifacts"]["predictions.npz"]
            path = Path(session.stages["evaluation-" + label]["directory"]) / "worker" / ref["path"]
            if file_hash(path) != ref["sha256"]:
                raise ValueError("Saved evaluation predictions changed")
            key = spec["id"] + "_" + label
            refs[key], mapping[label] = _ref(path), key
        analysis_specs.append(spec | {"prediction_inputs": mapping})
    config = _config(plan, purpose="expanded-analysis", arm="shared",
                     parameters={"common_sets": analysis_specs}, inputs=refs)
    session.stage("analysis", config, "analysis", module="kenggo.expanded_jobs")
    return {"schema": "expanded_e2_pilot_v1", "status": "completed", "plan": plan_ref,
        "training_decision": decision_ref, "stages": session.stages, "matched_training": True,
        "promotion": "not_promoted", "scope": "Single-seed data and rank comparison; final targets sealed"}


def run(plan_path, plan_pin, output):
    plan = read_json(plan_path, plan_pin)
    p = validate_plan(plan)
    output = Path(output).resolve()
    if output != (ROOT / p["output"]).resolve() or not output.is_relative_to(ROOT / "runs"):
        raise ValueError("Production output must be the one-use path in the frozen plan")
    lock_path = ROOT / ".cache/e2-expanded-pilot.lock"
    lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if file_hash(ROOT / p["ledger"]) != p["prior_ledger_sha256"]:
            raise ValueError("Shared SL ledger changed before the one-use launch")
        ledger = BudgetLedger.open(ROOT / p["ledger"], budget_id=plan["budget_id"], gpu_seconds=86400)
        state = ledger.snapshot()
        if state["budget_breached"] or state["state"]["reservations"]:
            raise ValueError("An existing SL failure or allocation must be resolved first")
        output.mkdir(parents=True, exist_ok=False)
        session = Session(output, plan, ledger)
        save_json(output / "session.json", {"schema": "expanded_e2_session_v1", "plan": _ref(plan_path),
            "launch_utc": session.utc.isoformat(), "deadline_utc": session.deadline.isoformat(),
            "limits": session.limits, "ledger_before": state})
        previous = signal.getsignal(signal.SIGALRM)
        def expired(signum, frame):
            raise TimeoutError("Expanded comparison reached its absolute finishing cutoff")
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, p["limits"]["wall_seconds"] - p["limits"]["publication_seconds"])
        status, reason, result = "failed", "supervisor_did_not_finish", None
        try:
            # Dataset pins are checked after the durable launch marker and before any worker.
            source, patch = capture_provenance(ROOT, plan["inputs"])
            save_json(output / "source.json", source)
            (output / "source.patch").write_bytes(patch)
            for cohort in COHORTS:
                capture_provenance(ROOT, p["datasets"][cohort]["inputs"])
            result = execute(plan, _ref(plan_path), output, session)
            current, _ = capture_provenance(ROOT, plan["inputs"])
            if current["source"] != source["source"]:
                raise ValueError("Runtime source changed during the experiment")
            status, reason = "completed", None
        except BaseException as error:
            reason = type(error).__name__ + ": " + str(error)
            result = {"schema": "expanded_e2_pilot_v1", "status": "incomplete", "reason": reason,
                      "plan": _ref(plan_path), "stages": session.stages, "promotion": "not_promoted"}
            if session.failed_stage is not None:
                result["failed_stage"] = session.failed_stage
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
            session.close(status, reason)
            if result is not None:
                result["costs"] = session.snapshot()
                save_json(output / ("summary.json" if status == "completed" else "failure.json"), result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config-sha256", required=True)
    args = parser.parse_args()
    run(args.config, args.config_sha256, args.output)
