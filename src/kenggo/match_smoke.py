"""Run four fixed CPU weak-policy games and verify their saved replay records."""

import argparse
import json
from pathlib import Path
import resource
import time


PARAMETERS = {"pairs_per_profile": 1, "candidate_policy": "uniform",
              "opponent_policies": ["uniform", "capture_first"], "adjudication": "exact_only",
              "cpu_seconds": 600, "wall_seconds": 900, "output_bytes": 128 * 1024**2,
              "bootstrap_seed": 20261113, "bootstrap_resamples": 10000,
              "device": "cpu", "max_artifact_bytes": 10 * 1024**2}


def cpu_usage():
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return time.process_time() + children.ru_utime + children.ru_stime


def run(config_path, config_pin, output):
    started, cpu = time.monotonic(), cpu_usage()
    from kenggo.checkpoints import file_hash, read_json, save_json
    from kenggo.match_statistics import summarize
    from kenggo.matches import ExperimentBudget, make_schedule, read_records, run_schedule, weak_profile
    from kenggo.resource_ledger import BudgetLedger
    from kenggo.tracking import Experiment, capture_provenance

    root = Path(__file__).resolve().parents[2]
    config = read_json(config_path, config_pin)
    if config["purpose"] != "match-smoke" or config["parameters"] != PARAMETERS:
        raise ValueError("Match validation requires its frozen four-game CPU configuration")
    provenance, patch = capture_provenance(root, config["inputs"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    ledger = BudgetLedger.create(output / "resources.json", budget_id=config["budget_id"], gpu_seconds=0)
    ledger.reserve("match-smoke", estimate_gpu_seconds=0, shutdown_gpu_seconds=0,
                   metadata={"config_sha256": config_pin, "purpose": config["purpose"], "arm": config["arm"]})
    settled, match_charge, outside_before, after_matches = False, 0.0, 0.0, None

    def costs():
        known = cpu_usage() - cpu
        charged = (match_charge + outside_before + cpu_usage() - after_matches
                   if after_matches is not None else known)
        return {"cpu_seconds": max(known, charged), "wall_seconds": time.monotonic() - started}

    try:
        revision = provenance["source"]["commit"]
        candidate = weak_profile("uniform-candidate", "uniform", source_revision=revision)
        opponents = tuple(weak_profile(name + "-opponent", policy, source_revision=revision)
                          for name, policy in (("uniform", "uniform"), ("capture", "capture_first")))
        schedule = make_schedule(arms=(candidate,), profiles=opponents, pairs=1, seed=config["seed"],
            purpose="validation", scope="setup", provenance=provenance)
        with Experiment(output / "run", config, provenance, patch) as experiment:
            outside_before = cpu_usage() - cpu
            budget = ExperimentBudget(cpu_seconds=600 - outside_before - 30,
                                      wall_seconds=900 - (time.monotonic() - started) - 75,
                                      output_bytes=PARAMETERS["output_bytes"])
            records = run_schedule(schedule, output / "matches", budget=budget)
            after_matches = cpu_usage()
            actual = json.loads((output / "matches/run.json").read_text())
            match_charge = actual["cpu_seconds"]
            saved_schedule, saved_records = read_records(output / "matches")
            if saved_schedule != schedule or saved_records != records:
                raise ValueError("Saved matches differ from their independently replayed records")
            scores = summarize(schedule, records, bootstrap_seed=PARAMETERS["bootstrap_seed"],
                               resamples=PARAMETERS["bootstrap_resamples"])
            counts = scores["arms"][candidate.id]["counts"]
            if counts["N"] != 4 or counts["N"] != sum(counts[k] for k in ("W", "L", "D", "U")):
                raise ValueError("Scheduled-game accounting changed")
            save_json(output / "scores.json", scores)
            experiment.log(len(records), {f"matches/{key}": value for key, value in counts.items()})
            if len(records) != 4:
                raise RuntimeError("The four-game check is incomplete; unfilled slots remain unresolved")
            experiment.finish(summary={"scope": "Weak-only runner implementation check; no candidate qualification",
                                       "scheduled_games": 4, "records_replayed": len(saved_records),
                                       "result_counts": counts, "gpu_seconds": 0})
        measured = costs()
        if measured["cpu_seconds"] > 600 or measured["wall_seconds"] > 900:
            raise RuntimeError("The match implementation check exceeded its frozen resource budget")
        ledger.record("match-smoke", allocations=[], outcome="completed", arm_shares={config["arm"]: 1}, **measured)
        settled = True
        state = json.loads((output / "run/run.json").read_text())
        result = {"version": "match_smoke_v1", "status": "completed", "config_sha256": config_pin,
                  "wandb_url": state["wandb"]["url"], "provenance_sha256": state["provenance_sha256"],
                  "scheduled_games": 4, "recorded_games": len(records), "all_saved_replays_verified": True,
                  "counts": counts, "scores_sha256": file_hash(output / "scores.json"),
                  "schedule_sha256": file_hash(output / "matches/schedule.json"),
                  "matches_run_sha256": file_hash(output / "matches/run.json"),
                  "run_manifest_sha256": file_hash(output / "run/run.json"),
                  "ledger_sha256": file_hash(output / "resources.json"), "costs": measured,
                  "cpu_scope": "Known wrapper/child CPU and conservative runner charge; tracking service CPU is not fully measured.",
                  "gpu_seconds": 0, "qualification_claim": False,
                  "scope": "Four fixed unranked weak-policy games, including caps/unresolved outcomes; no BASE or full-panel result."}
        save_json(output / "summary.json", result)
        print(json.dumps(result, indent=2))
        return result
    except BaseException as error:
        measured = costs()
        if not settled:
            ledger.record("match-smoke", allocations=[], outcome="failed", arm_shares={config["arm"]: 1},
                          reason=type(error).__name__, **measured)
        save_json(output / "failure.json", {"status": "failed", "reason": type(error).__name__, "costs": measured})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config-sha256", required=True)
    arguments = parser.parse_args()
    run(arguments.config, arguments.config_sha256, arguments.output)
