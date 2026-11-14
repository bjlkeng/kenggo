"""Check a pinned Human SL reference on twelve fixed synthetic CPU queries."""

import argparse
import json
from pathlib import Path
import random
import resource
import time


PARAMETERS = {"queries": 12, "cpu_seconds": 180, "wall_seconds": 300,
              "output_bytes": 16 * 1024**2, "device": "cpu",
              "max_artifact_bytes": 1024**2}
QUERY_PIN = "3c4641b35097f46a13766abba5562b4cc0dc9e9a3b8a4d65c188dac7ae2fa194"


def cpu_usage():
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return time.process_time() + children.ru_utime + children.ru_stime


def run(config_path, config_pin, output):
    started, cpu = time.monotonic(), cpu_usage()
    from kenggo.checkpoints import file_hash, json_hash, read_json, save_json
    from kenggo.go import PASS
    from kenggo.katago_human import HumanProfile, HumanSL, ReferenceBudget
    from kenggo.resource_ledger import BudgetLedger
    from kenggo.rules_replay import load_game
    from kenggo.tracking import Experiment, capture_provenance

    root = Path(__file__).resolve().parents[2]
    config = read_json(config_path, config_pin)
    if (config["purpose"] != "human-reference-smoke" or config["parameters"] != PARAMETERS
            or config["inputs"]["queries"]["sha256"] != QUERY_PIN):
        raise ValueError("This check requires the frozen synthetic CPU queries and limits")
    refs = config["inputs"]
    specification = read_json(root / refs["queries"]["path"], QUERY_PIN)
    preflight = read_json(root / refs["preflight"]["path"], refs["preflight"]["sha256"])
    if (preflight["schema"] != "human_reference_preflight_v1"
            or preflight["budget_id"] != config["budget_id"] or preflight["analysis_queries"] != 0):
        raise ValueError("The twelve-query check requires its charged, zero-analysis preflight")
    prior = preflight["prior_costs"]
    provenance, patch = capture_provenance(root, refs)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    ledger = BudgetLedger.create(output / "resources.json", budget_id=config["budget_id"], gpu_seconds=0)
    ledger.reserve("human-reference-smoke", estimate_gpu_seconds=0, shutdown_gpu_seconds=0,
                   metadata={"config_sha256": config_pin, "purpose": config["purpose"], "arm": config["arm"]})
    settled, client, completed, trace_number = False, None, [], 0

    def costs():
        return {"cpu_seconds": cpu_usage() - cpu, "wall_seconds": time.monotonic() - started}

    def check_budget():
        measured = costs()
        if measured["cpu_seconds"] + prior["cpu_seconds"] >= 180 or measured["wall_seconds"] + prior["wall_seconds"] >= 300:
            raise RuntimeError("Human reference check exhausted its CPU/wall budget")
        if preflight["output_bytes"] + sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) > PARAMETERS["output_bytes"]:
            raise RuntimeError("Human reference check exhausted its output allowance")

    def trace(record):
        nonlocal trace_number
        trace_number += 1
        save_json(output / f"trace-{trace_number:03d}.json", record)
        check_budget()

    try:
        cases = {}
        for case in specification["cases"]:
            game = load_game(json.dumps(case["game"]), allow_fixture=True)
            if (game.snapshot != case["snapshot"] or sum(game.legal_mask()) != case["legal_count"]
                    or game.position.ko != case["ko"]):
                raise ValueError("Synthetic query failed independent replay")
            cases[case["id"]] = game
        reference = HumanProfile.load(root / refs["human_profile"]["path"], refs["human_profile"]["sha256"])
        previous = {}
        with Experiment(output / "run", config, provenance, patch) as experiment:
            remaining = costs()
            client = HumanSL(reference,
                engine=root / ".cache/katago-referee/squashfs-root/usr/bin/katago",
                main_model=root / ".cache/katago-referee/kata1-b10c128-s1141046784-d204142634.txt.gz",
                human_model=root / ".cache/katago-human/b18c384nbt-humanv0.bin.gz",
                config=root / refs["human_config"]["path"],
                runtime_root=root / ".cache/katago-referee/squashfs-root",
                budget=ReferenceBudget(queries=12, cpu_seconds=150 - prior["cpu_seconds"] - remaining["cpu_seconds"],
                    wall_seconds=225 - prior["wall_seconds"] - remaining["wall_seconds"], request_bytes=1024**2,
                    response_bytes=4 * 1024**2, stderr_bytes=1024**2, trace_bytes=8 * 1024**2),
                trace_sink=trace)
            with client:
                version = client.query_version()
                save_json(output / "engine-version.json", version)
                for number, query in enumerate(specification["queries"], 1):
                    check_budget()
                    game = cases[query["case"]]
                    prediction = client.predict(game, rank=query["rank"], allow_fixture=True)
                    legal = game.legal_mask()
                    if (prediction.snapshot != game.snapshot or prediction.rank != query["rank"]
                            or any((raw < 0) == allowed for raw, allowed in zip(prediction.raw_policy, legal))
                            or any(prob != (raw if allowed else 0) for prob, raw, allowed in
                                   zip(prediction.policy, prediction.raw_policy, legal))
                            or prediction.policy[PASS] != prediction.raw_policy[PASS]):
                        raise ValueError("Human policy changed legality, coordinates or pass mass")
                    key = (query["case"], query["rank"])
                    fingerprint = (prediction.raw_policy, prediction.value)
                    if query["repeat"] and previous.get(key) != fingerprint:
                        raise ValueError("Repeated deterministic Human SL query changed")
                    previous[key] = fingerprint
                    rng_seed = int(json_hash([config["seed"], *key])[:16], 16)
                    rng = random.Random(rng_seed)
                    actions = [prediction.sample(rng) for _ in range(specification["sampling_draws_per_query"])]
                    if not all(type(action) is int and 0 <= action <= PASS and legal[action] for action in actions):
                        raise ValueError("Raw policy sampling produced an illegal action")
                    row = {"query": query, "snapshot": game.snapshot, "to_play": game.position.to_play,
                           "raw_policy": list(prediction.raw_policy), "policy": list(prediction.policy),
                           "value_mover": prediction.value, "pass_probability": prediction.policy[PASS],
                           "sampling_seed": rng_seed, "sampled_actions": actions,
                           "legal_count": sum(legal), "repeat_exact": bool(query["repeat"])}
                    save_json(output / f"prediction-{number:03d}.json", row)
                    completed.append({"query": query, "snapshot": game.snapshot,
                                      "pass_probability": prediction.policy[PASS],
                                      "value_mover": prediction.value, "legal_count": sum(legal)})
                    experiment.log(number, {"check/queries_completed": number,
                        "check/pass_probability": prediction.policy[PASS],
                        "check/value_mover": prediction.value})
            adapter_cost = client.cost()
            save_json(output / "adapter-cost.json", adapter_cost)
            check_budget()
            experiment.finish(summary={"scope": "Synthetic CPU adapter check; no BASE or held-out comparison",
                "queries_completed": len(completed), "repeat_pairs_exact": 5,
                "pass_mass_preserved": True, "all_sampled_actions_legal": True, "gpu_seconds": 0})
        check_budget()
        measured = costs()
        ledger.record("human-reference-smoke", allocations=[], outcome="completed",
                      arm_shares={config["arm"]: 1}, **measured)
        settled = True
        state = json.loads((output / "run/run.json").read_text())
        result = {"version": "human_reference_smoke_v1", "status": "completed",
            "config_sha256": config_pin, "wandb_url": state["wandb"]["url"],
            "provenance_sha256": state["provenance_sha256"], "engine_version": version,
            "queries_completed": len(completed), "results": completed, "repeat_pairs_exact": 5,
            "pass_mass_preserved": True, "all_sampled_actions_legal": True,
            "adapter_cost": adapter_cost, "costs": measured, "gpu_seconds": 0,
            "costs_including_preflight": {key: measured[key] + prior[key] for key in measured},
            "prior_analysis_queries": preflight["analysis_queries"],
            "prior_metadata_requests": preflight["metadata_requests"],
            "network_calls": None, "network_calls_note": "Native cache hides actual neural forward counts",
            "run_manifest_sha256": file_hash(output / "run/run.json"),
            "ledger_sha256": file_hash(output / "resources.json"),
            "cpu_scope": "Calling process and reaped children; tracking service CPU is not fully measured",
            "scope": "Twelve fixed synthetic queries; no source game labels, strength or BASE result"}
        save_json(output / "summary.json", result)
        print(json.dumps(result, indent=2))
        return result
    except BaseException as error:
        if client is not None:
            client.close()
        measured = costs()
        if not settled:
            ledger.record("human-reference-smoke", allocations=[], outcome="failed",
                arm_shares={config["arm"]: 1}, reason=type(error).__name__, **measured)
        save_json(output / "failure.json", {"status": "failed", "reason": str(error),
            "error_type": type(error).__name__, "queries_completed": len(completed),
            "adapter_cost": getattr(error, "cost", None) if client is None else client.cost(),
            "costs": measured, "costs_including_preflight": {key: measured[key] + prior[key] for key in measured},
            "gpu_seconds": 0})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config-sha256", required=True)
    arguments = parser.parse_args()
    run(arguments.config, arguments.config_sha256, arguments.output)
