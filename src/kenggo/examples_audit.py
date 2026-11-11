"""Build examples from the reviewed frozen corpus and check bounded diagnostic positions."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from kenggo.canonical_data import digest, encoded
from kenggo.encoding import INPUT_KEYS, encode
from kenggo.examples import (CANONICAL_SHA256, CODE_FILES, FINAL_ROLES, SPLIT_SHA256, ExampleConfig, build_examples,
                             examples_for_game, file_hash, fit_normalizer, load_corpus, load_dataset_profile,
                             replay_record, sealed_sources, source_context)
from kenggo.go import Game, PASS
from kenggo.rules_replay import dump_game, load_game


KINDS = ("scored", "resignation", "timeout", "forfeit", "incomplete_or_unknown")


def diagnostic_sample(records, *, sealed_source_ids=(), check_budget=lambda: None):
    chosen = []
    for kind in KINDS:
        candidates = sorted((r for r in records.values() if not r["structural_exclusions"]
                             and not r.get("sealed_final", False) and r["source"]["source_url"] not in sealed_source_ids
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


def run(source, output, *, config=ExampleConfig(), profile=None):
    started, cpu = time.monotonic(), time.process_time()
    started_utc = datetime.now(timezone.utc).isoformat()
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError("Example audit output already exists")

    def check_budget():
        if (time.process_time() - cpu > (120 if profile is None else profile.cpu_seconds)
                or time.monotonic() - started > (180 if profile is None else profile.wall_seconds)):
            raise RuntimeError("Example audit time budget exceeded")

    def costs():
        return {"started_utc": started_utc, "cpu_seconds": time.process_time() - cpu,
                "wall_seconds": time.monotonic() - started, "workers": 1, "http_requests": 0, "gpu_hours": 0,
                "scope": "Pinned-input verification, shard/normalizer writes and sampled replay checks; excludes imports and final summary/failure write."}

    canonical_sha256 = CANONICAL_SHA256 if profile is None else profile.canonical_sha256
    split_sha256 = SPLIT_SHA256 if profile is None else profile.split_sha256
    records, split, _ = load_corpus(source / "canonical.parquet", canonical_sha256,
                                   source / "splits", split_sha256, profile=profile)
    output.mkdir(parents=True)
    try:
        key = build_examples(source / "canonical.parquet", canonical_sha256, source / "splits", split_sha256,
                             output / "examples", config=config, profile=profile, check_budget=check_budget)
        normalizer = fit_normalizer(output / "examples", key)
        (output / "normalizer.json").write_text(encoded(normalizer) + "\n")
        diagnostics = diagnostic_sample(records, sealed_source_ids=sealed_sources(split), check_budget=check_budget)
        (output / "diagnostics.json").write_text(encoded(diagnostics) + "\n")
        check_budget()
    except Exception as error:
        failure = {"status": "failed", "error_type": type(error).__name__, "error": str(error), "costs": costs(),
                   "code_sha256": {name: file_hash(Path(__file__).with_name(name)) for name in CODE_FILES}}
        (output / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise
    manifest = json.loads((output / "examples/manifest.json").read_text())
    summary = {"version": "example_audit_v1", "config": manifest["config"], "canonical_records": len(records),
               "canonical_sha256": canonical_sha256, "split_sha256": split_sha256,
               "examples_manifest_sha256": key, "normalizer_sha256": file_hash(output / "normalizer.json"),
               "diagnostics_sha256": file_hash(output / "diagnostics.json"), "counts": manifest["counts"],
               "normalizer_fitted": normalizer["fitted"], "diagnostic_games": diagnostics["games"],
               "diagnostic_positions": diagnostics["positions"], "diagnostic_mismatches": diagnostics["mismatches"],
               "code_sha256": manifest["code_sha256"], "uv_lock_sha256": manifest["uv_lock_sha256"],
               "training_ready": False, "e1_ready": False,
               "costs": costs()}
    if profile is not None:
        summary["dataset_profile_sha256"] = profile.sha256
        summary["counts"] = {role: ({"examples": count["examples"], "sealed": True} if role in FINAL_ROLES else count)
                             for role, count in manifest["counts"].items()}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("canonical_records", "normalizer_fitted", "diagnostic_games",
                                            "diagnostic_positions", "diagnostic_mismatches", "costs")}, indent=2))
    return summary


def probe(source, *, profile, games=5, cpu_seconds=120, wall_seconds=180, config=ExampleConfig()):
    if type(games) is not int or not 1 <= games <= 5:
        raise ValueError("The encoding preflight permits one through five training games")
    if (type(cpu_seconds) is not int or type(wall_seconds) is not int
            or not 0 < cpu_seconds <= 120 or not 0 < wall_seconds <= 180):
        raise ValueError("The encoding preflight exceeds its 120 CPU / 180 wall second ceiling")
    started, cpu = time.monotonic(), time.process_time()

    def check_budget():
        if time.process_time() - cpu > cpu_seconds or time.monotonic() - started > wall_seconds:
            raise RuntimeError("Positive encoding preflight budget exceeded")

    source = Path(source)
    records, split, partitions = load_corpus(source / "canonical.parquet", profile.canonical_sha256,
                                            source / "splits", profile.split_sha256, profile=profile)
    selected = sorted(partitions["train"], key=lambda row: row["source_url"])[:games]
    examples = 0
    for membership in selected:
        check_budget()
        for _ in examples_for_game(records[membership["source_url"]], membership, split["plan"], config):
            examples += 1
            check_budget()
    check_budget()
    result = {"version": "positive_encoding_probe_v1", "dataset_profile_sha256": profile.sha256,
              "games": len(selected), "examples": examples, "artifacts_written": False,
              "cpu_seconds": time.process_time() - cpu, "wall_seconds": time.monotonic() - started,
              "limits": {"games": games, "cpu_seconds": cpu_seconds, "wall_seconds": wall_seconds},
              "http_requests": 0, "gpu_hours": 0, "scope": "Training encoding only; no final examples or diagnostic positions."}
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--history", type=int, choices=(0, 8, 32, 64), default=0)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--profile-sha256")
    parser.add_argument("--probe-train-games", type=int)
    args = parser.parse_args()
    if bool(args.profile) != bool(args.profile_sha256):
        parser.error("A production profile requires its independently supplied SHA-256")
    profile = load_dataset_profile(args.profile, args.profile_sha256) if args.profile else None
    config = ExampleConfig(history=args.history)
    if args.probe_train_games is not None:
        if profile is None or args.output is not None:
            parser.error("A training preflight requires a production profile and no output directory")
        probe(args.source, profile=profile, games=args.probe_train_games, config=config)
    else:
        if args.output is None:
            parser.error("An output directory is required")
        run(args.source, args.output, config=config, profile=profile)


if __name__ == "__main__":
    main()
