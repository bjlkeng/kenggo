"""Build examples from the reviewed frozen corpus and check bounded diagnostic positions."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from kenggo.canonical_data import digest, encoded
from kenggo.encoding import INPUT_KEYS, encode
from kenggo.examples import (CANONICAL_SHA256, CODE_FILES, SPLIT_SHA256, ExampleConfig, build_examples,
                             file_hash, fit_normalizer, load_corpus, replay_record, source_context)
from kenggo.go import Game, PASS
from kenggo.rules_replay import dump_game, load_game


KINDS = ("scored", "resignation", "timeout", "forfeit", "incomplete_or_unknown")


def diagnostic_sample(records, *, check_budget=lambda: None):
    chosen = []
    for kind in KINDS:
        candidates = sorted((r for r in records.values() if not r["structural_exclusions"]
                             and r["replay_validated"] and r["recorded_result"]["kind"] == kind),
                            key=lambda r: r["source"]["source_url"])
        if candidates:
            chosen.append(candidates[0])
    evidence = []
    for record in chosen:
        check_budget()
        final = replay_record(record)
        count = len(record["actions"])
        selected = {0, count // 2, count - 1}
        first_pass = next((i for i, m in enumerate(record["actions"]) if m["action"] == PASS), None)
        if first_pass is not None:
            selected.add(first_pass)
        game, positions = Game.new(), []
        for ply, move in enumerate(record["actions"]):
            check_budget()
            if ply in selected:
                context = source_context(record)
                encoded_views = {}
                rebuilt = load_game(dump_game(game))
                for window in (0, 8, 32, 64):
                    inputs = encode(game, context, history=window)
                    restored = encode(rebuilt, context, history=window)
                    for key in INPUT_KEYS:
                        np.testing.assert_array_equal(inputs[key], restored[key])
                    np.testing.assert_array_equal(inputs["legal"], game.legal_mask())
                    np.testing.assert_array_equal(inputs["board"][0].reshape(-1),
                                                  [c == move["color"] for c in game.position.board])
                    if (not inputs["legal"][move["action"]] or inputs["scalars"][5] != ply
                            or inputs["scalar_valid"][12:].any() or game.snapshot != rebuilt.snapshot):
                        raise ValueError("Sampled replay-to-input mismatch")
                    encoded_views[str(window)] = digest({k: v.tolist() for k, v in inputs.items()})
                positions.append({"ply": ply, "state_sha256": game.snapshot,
                                  "input_sha256_by_history": encoded_views})
            game = game.play(move["color"], move["action"])
        if game.snapshot != final.snapshot:
            raise ValueError("Diagnostic source history changed")
        evidence.append({"source_url": record["source"]["source_url"], "canonical_sha256": digest(record),
                         "result_kind": record["recorded_result"]["kind"], "actions": count,
                         "source_admitted": record["eligibility"]["training_eligible"], "positions": positions})
    return {"selection": "First source URL per supported result kind; first/middle/last/first-pass indices, deduplicated.",
            "games": len(evidence), "positions": sum(len(e["positions"]) for e in evidence),
            "history_windows": [0, 8, 32, 64], "source_actions_replayed": sum(e["actions"] for e in evidence),
            "evidence": evidence, "mismatches": 0, "training_examples_emitted": 0,
            "scope": "Offline encoding diagnostics only; no new native corpus comparison or source admission."}


def run(source, output, *, config=ExampleConfig()):
    started, cpu = time.monotonic(), time.process_time()
    started_utc = datetime.now(timezone.utc).isoformat()
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError("Example audit output already exists")

    def check_budget():
        if time.process_time() - cpu > 120 or time.monotonic() - started > 180:
            raise RuntimeError("Example audit time budget exceeded")

    def costs():
        return {"started_utc": started_utc, "cpu_seconds": time.process_time() - cpu,
                "wall_seconds": time.monotonic() - started, "workers": 1, "http_requests": 0, "gpu_hours": 0,
                "scope": "Pinned-input verification, shard/normalizer writes and sampled replay checks; excludes imports and final summary/failure write."}

    records, _, _ = load_corpus(source / "canonical.parquet", CANONICAL_SHA256,
                               source / "splits", SPLIT_SHA256)
    output.mkdir(parents=True)
    try:
        key = build_examples(source / "canonical.parquet", CANONICAL_SHA256, source / "splits", SPLIT_SHA256,
                             output / "examples", config=config, check_budget=check_budget)
        normalizer = fit_normalizer(output / "examples", key)
        (output / "normalizer.json").write_text(encoded(normalizer) + "\n")
        diagnostics = diagnostic_sample(records, check_budget=check_budget)
        (output / "diagnostics.json").write_text(encoded(diagnostics) + "\n")
        check_budget()
    except Exception as error:
        failure = {"status": "failed", "error_type": type(error).__name__, "error": str(error), "costs": costs(),
                   "code_sha256": {name: file_hash(Path(__file__).with_name(name)) for name in CODE_FILES}}
        (output / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise
    manifest = json.loads((output / "examples/manifest.json").read_text())
    summary = {"version": "example_audit_v1", "config": manifest["config"], "canonical_records": len(records),
               "canonical_sha256": CANONICAL_SHA256, "split_sha256": SPLIT_SHA256,
               "examples_manifest_sha256": key, "normalizer_sha256": file_hash(output / "normalizer.json"),
               "diagnostics_sha256": file_hash(output / "diagnostics.json"), "counts": manifest["counts"],
               "normalizer_fitted": normalizer["fitted"], "diagnostic_games": diagnostics["games"],
               "diagnostic_positions": diagnostics["positions"], "diagnostic_mismatches": diagnostics["mismatches"],
               "code_sha256": manifest["code_sha256"], "uv_lock_sha256": manifest["uv_lock_sha256"],
               "training_ready": False, "e1_ready": False,
               "costs": costs()}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("canonical_records", "normalizer_fitted", "diagnostic_games",
                                            "diagnostic_positions", "diagnostic_mismatches", "costs")}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--history", type=int, choices=(0, 8, 32, 64), default=0)
    args = parser.parse_args()
    run(args.source, args.output, config=ExampleConfig(history=args.history))


if __name__ == "__main__":
    main()
