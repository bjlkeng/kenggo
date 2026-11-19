"""One-use E2 review rerun with saved models and a seven-day elapsed cap."""

import argparse
from datetime import datetime, timezone
import fcntl
from pathlib import Path
import signal

from kenggo.base_pilot import _config, _learned_bundle, _ref
from kenggo.baseline_evaluation import WEEK_REVIEW_RANKS, WEEK_REVIEW_SCOPE
from kenggo.checkpoints import file_hash, read_json, save_json
from kenggo.expanded_jobs import REVIEW_SELECTION
from kenggo.expanded_pilot import (ARMS, COHORTS, ROOT, Session, _match_config,
                                  matched_training, validate_plan)
from kenggo.resource_ledger import BudgetLedger
from kenggo.tracking import capture_provenance, validate_config


VERSION = "expanded_e2_review_rerun_v1"
STAGES = ("preparation", *("evaluation-" + arm for arm in ARMS), "analysis",
          "matches-baseline", "matches-combined", "evaluation-human-reference")
LIMITS = {"wall_seconds": 604800, "cpu_seconds": 518400,
          "output_bytes": 32 * 1024**3, "publication_seconds": 3600}
EXPECTED_SETS = {
    "population_development_komi13": {"games": 136, "rows": 31233},
    "population_development_komi1": {"games": 142, "rows": 31784},
    "validation_development_komi13": {"games": 127, "rows": 27949},
    "validation_development_komi1": {"games": 134, "rows": 29484},
}
JOB_SECONDS = {"preparation": 21600, "learned_evaluation": 21600, "analysis": 21600,
               "matches": 46800, "human_evaluation": 129600}
JOB_BYTES = {"preparation": 2, "learned_evaluation": 4, "analysis": 2,
             "matches": 2, "human_evaluation": 8}


def _load(ref):
    return read_json(ROOT / ref["path"], ref["sha256"])


def validate_review(plan):
    validate_config(plan)
    p, refs = plan["parameters"], plan["inputs"]
    original = _load(refs["original_plan"])
    previous = validate_plan(original)
    required = {"output", "ledger", "prior_ledger_sha256", "limits", "job_limits", "selection",
                "review_ranks", "expected_sets", "datasets", "architecture", "human_paths",
                "reference_budget", "match_parameters", "publication_cpu_seconds", "publication_bytes"}
    if (plan["purpose"] != "expanded-review" or plan["arm"] != "shared"
            or plan["comparison"] != "E2-expanded-review-v1" or set(p) != required
            or p["limits"] != LIMITS or p["selection"] != REVIEW_SELECTION
            or p["review_ranks"] != WEEK_REVIEW_RANKS or p["expected_sets"] != EXPECTED_SETS
            or p["publication_cpu_seconds"] != 1800 or p["publication_bytes"] != 512 * 1024**2
            or p["output"] != "runs/e2-review-rerun-2026-11-19"
            or len(p["prior_ledger_sha256"]) != 64):
        raise ValueError("Review rerun differs from its frozen scope or one-week allowance")
    for key in ("version", "entity", "project", "base_url", "seed", "budget_id"):
        if plan[key] != original[key]:
            raise ValueError("The rerun must retain the original model and accounting identity")
    for key in ("datasets", "architecture", "human_paths", "ledger"):
        if p[key] != previous[key]:
            raise ValueError("The rerun cannot change saved models' data or runtime inputs")
    for key, ref in original["inputs"].items():
        if key != "comparison_protocol" and refs.get(key) != ref:
            raise ValueError("The rerun must preserve original data and reference inputs")
    if set(p["job_limits"]) != set(JOB_SECONDS):
        raise ValueError("Every rerun phase requires its own bounded allowance")
    for kind, seconds in JOB_SECONDS.items():
        expected = {"wall_seconds": seconds, "cpu_seconds": seconds,
                    "output_bytes": JOB_BYTES[kind] * 1024**3, "shutdown_seconds": 10}
        if p["job_limits"][kind] != expected:
            raise ValueError("Rerun job limits differ from the reviewed allowances")
    reference = dict(previous["reference_budget"], queries=59182, cpu_seconds=86400,
        wall_seconds=86400, request_bytes=512 * 1024**2, response_bytes=1024**3, trace_bytes=2 * 1024**3)
    matches = {**previous["match_parameters"],
        "match_limits": dict(previous["match_parameters"]["match_limits"], game_seconds=3600),
        "runner_budget": {"cpu_seconds": 43200, "wall_seconds": 43200, "output_bytes": 1024**3},
        "human_budget": dict(previous["match_parameters"]["human_budget"], cpu_seconds=3600, wall_seconds=3600)}
    if p["reference_budget"] != reference or p["match_parameters"] != matches:
        raise ValueError("Rerun execution budgets changed the reference or game protocol")
    from kenggo.katago_human import ReferenceBudget
    ReferenceBudget(**reference)
    return original


def saved_models(plan, original):
    refs = plan["inputs"]
    prior, decision = _load(refs["previous_result"]), _load(refs["training_decision"])
    old_root = (ROOT / original["parameters"]["output"]).resolve()
    if (prior["schema"] != "expanded_e2_pilot_v1" or prior["status"] != "incomplete"
            or prior.get("completion_status") != "completed" or prior.get("matched_training") is not True
            or prior["plan"] != refs["original_plan"] or prior["training_decision"] != refs["training_decision"]
            or (ROOT / refs["previous_result"]["path"]).resolve() != old_root / "completion.json"
            or (ROOT / refs["training_decision"]["path"]).resolve() != old_root / "frozen-training.json"):
        raise ValueError("Rerun requires the preserved expanded result and fixed training decision")
    stages, summaries, bundles = {}, {}, {}
    for arm in ARMS:
        stage = prior["stages"][arm]
        directory = old_root / arm
        if Path(stage["directory"]).resolve() != directory:
            raise ValueError("Saved training moved outside its original directory")
        for key in ("config", "provenance", "summary"):
            if not (ROOT / stage[key]["path"]).resolve().is_relative_to(directory):
                raise ValueError("Saved training references must belong to their stage")
        config, provenance, summary = (_load(stage[key]) for key in ("config", "provenance", "summary"))
        if (summary["status"] != "completed" or stage["job"]["status"] != "completed"
                or stage["job"] != _load(_ref(directory / "worker/job.json"))
                or provenance["inputs"] != config["inputs"]
                or config["inputs"]["training_decision"] != refs["training_decision"]):
            raise ValueError("Saved training or provenance is incomplete or changed")
        cohort = arm.split("-")[0]
        bundle = _learned_bundle(stage, summary, {"inputs": plan["parameters"]["datasets"][cohort]["inputs"],
            "parameters": plan["parameters"]}, arm.endswith("-conditioned"))
        if file_hash(bundle["checkpoint_manifest"]) != bundle["checkpoint_sha256"]:
            raise ValueError("The saved final checkpoint manifest changed")
        stages[arm], summaries[arm], bundles[arm] = stage, summary, bundle
    matched_training(summaries, expected_updates=decision["updates"])
    return stages, bundles


def execute(plan, plan_ref, session, training, bundles):
    p = plan["parameters"]
    bases = {cohort: item["inputs"] | {"review_plan": plan_ref}
             for cohort, item in p["datasets"].items()}
    base = bases["combined"]
    config = _config(plan, purpose="expanded-review-preparation", arm="shared",
                     parameters={"selection": p["selection"]}, inputs=base)
    _, prepared = session.stage("preparation", config, "preparation", module="kenggo.expanded_jobs")
    observed = {item["spec"]["id"]: {"games": len(item["spec"]["source_ids"]), "rows": item["rows"]}
                for item in prepared["common_sets"]}
    if observed != p["expected_sets"]:
        raise ValueError("Prepared selection differs from the frozen metadata-only workload")
    specs, common_refs = [], {}
    for item in prepared["common_sets"]:
        spec = item["spec"]
        key = "common_" + spec["id"]
        specs.append({name: spec[name] for name in ("id", "role", "source_ids", "aggregate_ranks")}
                     | {"manifest_input": key})
        common_refs[key] = item["manifest"]
    evaluations = {}
    for arm in ARMS:
        cohort, treatment = arm.split("-")
        bundle = bundles[arm]
        refs = base | common_refs | {"checkpoint": _ref(bundle["checkpoint_manifest"]),
            "training_config": training[arm]["config"], "training_provenance": training[arm]["provenance"],
            "training_examples": bases[cohort]["examples"], "training_normalizer": bases[cohort]["normalizer"]}
        parameters = {"common_sets": specs, "common_input_mode": "raw_komi_comparison_v1",
            "evaluation_scope": WEEK_REVIEW_SCOPE, "batch_size": 32, "samples_per_position": 32,
            "unknown_rank_diagnostic": False, "artifacts": bundle}
        config = _config(plan, purpose="baseline-evaluation", arm=treatment, parameters=parameters, inputs=refs)
        _, evaluations[arm] = session.stage("evaluation-" + arm, config, "learned_evaluation",
                                            module="kenggo.baseline_evaluation")
    refs, analysis_specs = base | common_refs, []
    for spec in specs:
        mapping = {}
        for arm in ARMS:
            report = next(item for item in evaluations[arm]["reports"]
                          if item["set"] == spec["id"] and item["variant"] == "ordinary")
            ref = report["artifacts"]["predictions.npz"]
            worker = Path(session.stages["evaluation-" + arm]["directory"]) / "worker"
            path = (worker / ref["path"]).resolve()
            if not path.is_relative_to(worker) or file_hash(path) != ref["sha256"]:
                raise ValueError("Saved prediction archive changed or moved")
            key = spec["id"] + "_" + arm
            refs[key], mapping[arm] = _ref(path), key
        analysis_specs.append(spec | {"prediction_inputs": mapping})
    config = _config(plan, purpose="expanded-review-analysis", arm="shared",
                     parameters={"common_sets": analysis_specs}, inputs=refs)
    session.stage("analysis", config, "analysis", module="kenggo.expanded_jobs")
    for cohort in COHORTS:
        config = _match_config(plan, bases[cohort], cohort, bundles, training)
        config["parameters"]["recipe"] = "week_review_matches_v1"
        session.stage("matches-" + cohort, config, "matches", module="kenggo.baseline_matches")
    human_specs = [spec for spec in specs if spec["id"].endswith("komi13")]
    rows = sum(p["expected_sets"][spec["id"]]["rows"] for spec in human_specs)
    if rows != p["reference_budget"]["queries"]:
        raise ValueError("Reference query allowance differs from the frozen whole-game count")
    refs = base | {spec["manifest_input"]: common_refs[spec["manifest_input"]] for spec in human_specs}
    refs |= {key: ref for key, ref in plan["inputs"].items() if key.startswith("human_")}
    parameters = {"common_sets": human_specs, "common_input_mode": "raw_komi_comparison_v1",
        "evaluation_scope": WEEK_REVIEW_SCOPE, "batch_size": 32, "samples_per_position": 32,
        "unknown_rank_diagnostic": False, "human_paths": p["human_paths"],
        "reference_budget": p["reference_budget"]}
    config = _config(plan, purpose="baseline-evaluation", arm="human-reference", parameters=parameters, inputs=refs)
    session.stage("evaluation-human-reference", config, "human_evaluation", module="kenggo.baseline_evaluation")


def run(plan_path, plan_pin, output):
    plan = read_json(plan_path, plan_pin)
    original = validate_review(plan)
    p = plan["parameters"]
    output = Path(output).resolve()
    if output != (ROOT / p["output"]).resolve() or not output.is_relative_to(ROOT / "runs"):
        raise ValueError("Rerun output must be its one-use frozen path")
    lock_path = ROOT / ".cache/e2-expanded-pilot.lock"
    lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if file_hash(ROOT / p["ledger"]) != p["prior_ledger_sha256"]:
            raise ValueError("The cumulative SL ledger changed before launch")
        ledger = BudgetLedger.open(ROOT / p["ledger"], budget_id=plan["budget_id"], gpu_seconds=86400)
        before = ledger.snapshot()
        if before["budget_breached"] or before["state"]["reservations"]:
            raise ValueError("A previous allocation or breach must be settled first")
        output.mkdir(parents=True, exist_ok=False)
        session = Session(output, plan, ledger)
        save_json(output / "session.json", {"schema": VERSION, "plan": _ref(plan_path),
            "launch_utc": session.utc.isoformat(), "deadline_utc": session.deadline.isoformat(),
            "limits": p["limits"], "ledger_before": before,
            "publication_cpu_seconds": p["publication_cpu_seconds"], "publication_bytes": p["publication_bytes"]})
        # Hold the publication allowance aside while workers are running.
        session.limits = dict(session.limits, cpu_seconds=p["limits"]["cpu_seconds"] - p["publication_cpu_seconds"],
                             output_bytes=p["limits"]["output_bytes"] - p["publication_bytes"])
        previous_handler = signal.getsignal(signal.SIGALRM)
        def expired(signum, frame):
            raise TimeoutError("E2 review rerun reached its absolute production cutoff")
        signal.signal(signal.SIGALRM, expired)
        remaining = (session.deadline - datetime.now(timezone.utc)).total_seconds() - p["limits"]["publication_seconds"]
        signal.setitimer(signal.ITIMER_REAL, max(.001, remaining))
        status, reason = "failed", "supervisor_did_not_finish"
        result = {"schema": VERSION, "status": "incomplete", "plan": _ref(plan_path),
            "session": _ref(output / "session.json"),
            "previous_result": plan["inputs"]["previous_result"],
            "training_decision": plan["inputs"]["training_decision"], "reused_training": {},
            "matched_training": False, "stages": session.stages, "promotion": "not_promoted"}
        try:
            source, patch = capture_provenance(ROOT, plan["inputs"])
            save_json(output / "source.json", source)
            result["source"] = _ref(output / "source.json")
            (output / "source.patch").write_bytes(patch)
            for data in p["datasets"].values():
                capture_provenance(ROOT, data["inputs"])
            training, bundles = saved_models(plan, original)
            result.update(reused_training=training, matched_training=True)
            execute(plan, _ref(plan_path), session, training, bundles)
            current, _ = capture_provenance(ROOT, plan["inputs"])
            if current["source"] != source["source"]:
                raise ValueError("Rerun source changed during production")
            status, reason = "completed", None
            result["status"] = "completed"
        except BaseException as error:
            reason = type(error).__name__ + ": " + str(error)
            if session.failed_stage is not None:
                result["failed_stage"] = session.failed_stage
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
            session.close(status, reason)
            result.update(reason=reason, costs=session.snapshot())
            save_json(output / ("summary.json" if status == "completed" else "failure.json"), result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config-sha256", required=True)
    args = parser.parse_args()
    run(args.config, args.config_sha256, args.output)
