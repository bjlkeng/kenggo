"""Validate the pinned empirical referee on fixtures and the exposed E1 panel."""

import argparse
from collections import Counter
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import pyarrow.parquet as pq

from kenggo.adjudication import adjudicate_empirical, judge_pass
from kenggo.go import Game, PASS, canonical, digest
from kenggo.katago_referee import KataGoReferee, Profile
from kenggo.rules_replay import dump_game, load_game


CANONICAL_PATH = Path("runs/e1-canonical-2026-11-11-v4/canonical.parquet")
CANONICAL_SHA256 = "b3ec35e3ff97338f440e431e15e4ede8060158489903159108dede6f73ae1db0"
ADMISSION_PATH = Path("manifests/e1_kgs_admission_games_2026-11-11.parquet")
ADMISSION_SHA256 = "7e63e6c50a78a751dc0494533f6a5e80881019b23b58289a503e9257ceda7295"
PASS_SEED = "kenggo-6yr.10-pass-panel-v1"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cpu():
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return time.process_time() + children.ru_utime + children.ru_stime


def replay(record, stop=None):
    game = Game.new()
    for move in record["actions"][:stop]:
        game = game.play(move["color"], move["action"])
    return game


def source_panel():
    for path, pin in ((CANONICAL_PATH, CANONICAL_SHA256), (ADMISSION_PATH, ADMISSION_SHA256)):
        if file_hash(path) != pin:
            raise ValueError("Frozen exposed input differs from its independent pin")
    admitted = {r["source_url"] for r in pq.read_table(ADMISSION_PATH).to_pylist() if r["training_eligible"]}
    records = [json.loads(r["canonical_json"]) for r in pq.read_table(CANONICAL_PATH).to_pylist()]
    records = sorted((r for r in records if r["source"]["source_url"] in admitted), key=lambda r: r["source"]["source_url"])
    endings = [r for r in records if r["replay_end_status"] == "pending_adjudication"]
    passes = [(r, i) for r in records for i, move in enumerate(r["actions"]) if move["action"] == PASS]
    selected = sorted(passes, key=lambda item: digest([PASS_SEED, item[0]["source"]["source_url"], item[1]]))[:24]
    if len(records) != 241 or len(endings) != 108 or len(passes) != 266:
        raise ValueError("Original panel changed")
    return endings, selected


def run(profile_path, pin, output, fixtures_only=False, revision_probe=False):
    profile = Profile.load(profile_path, pin)
    output.mkdir(parents=True, exist_ok=False)
    start_cpu, started = cpu(), time.monotonic()
    try:
        return _run(profile, pin, output, fixtures_only, revision_probe)
    except BaseException as error:
        failure = {"status": "failed", "error_type": type(error).__name__, "error": str(error),
                   "cost": {"cpu_seconds": cpu() - start_cpu, "wall_seconds": time.monotonic() - started}}
        with (output / "failure.json").open("x") as stream:
            json.dump(failure, stream, indent=2)
            stream.write("\n")
        raise


def _run(profile, pin, output, fixtures_only=False, revision_probe=False):
    started = time.monotonic()
    start_cpu = cpu()
    code_paths = [Path(__file__), Path("src/kenggo/katago_referee.py"), Path("src/kenggo/go.py"),
                  Path("src/kenggo/adjudication.py"), Path("src/kenggo/rules_replay.py"), Path("tests/go_fixtures.py")]
    code_pins = {str(path): file_hash(path) for path in code_paths}
    endings, passes = ([], []) if fixtures_only else source_panel()
    diagnostic_cases = [2, 3, 4, 7, 10] if revision_probe else None
    if diagnostic_cases is not None:
        endings, passes = [endings[index] for index in diagnostic_cases], []
    cpu_limit, wall_limit = (600, 900) if revision_probe else (3600, 5400)
    write = lambda name, value: (output / name).write_text(json.dumps(value, indent=2) + "\n")
    write("panel.json", {"ending_source_ids": [r["source"]["source_url"] for r in endings],
                          "pass_seed": PASS_SEED,
                          "passes": [{"source_url": r["source"]["source_url"], "pre_move_index": i} for r, i in passes],
                          "canonical_sha256": CANONICAL_SHA256, "admission_sha256": ADMISSION_SHA256})
    write("run.json", {"profile_sha256": pin, "code_sha256": code_pins,
                       "limits": {"cpu_seconds": cpu_limit, "wall_seconds": wall_limit,
                                  "compressed_trace_bytes": 1024**3}, "fixtures_only": fixtures_only,
                       "revision_probe_original_case_indexes": diagnostic_cases})
    catalog, results, judgments, fixtures, pass_fixtures = [], [], [], [], []
    trace_path = output / "trace.jsonl.gz"
    with gzip.open(trace_path, "wt", compresslevel=1) as trace:
        def sink(kind, request, response):
            trace.write(canonical({"kind": kind, "request": request.decode(), "response": response.decode()}) + "\n")
            if trace_path.stat().st_size > 1024**3:
                raise RuntimeError("Compressed trace budget exceeded")

        with KataGoReferee(profile, engine=".cache/katago-referee/squashfs-root/usr/bin/katago",
                model=".cache/katago-referee/kata1-b10c128-s1141046784-d204142634.txt.gz",
                config="configs/katago_referee_v1.cfg", native=".cache/katago-reference/katago-reference",
                runtime_root=".cache/katago-referee/squashfs-root", trace_sink=sink) as referee:
            def check_budget():
                if referee._cpu() - start_cpu > cpu_limit or time.monotonic() - started > wall_limit:
                    raise RuntimeError("Total referee validation budget exhausted")
                if any(file_hash(path) != value for path, value in code_pins.items()):
                    raise RuntimeError("Referee implementation changed during frozen run")

            sys.path.insert(0, str(Path("tests").resolve()))
            from go_fixtures import dense, two_passes
            for kind, expected in (("settled", 13), ("dead", 3), ("seki", 13)):
                check_budget()
                game = two_passes(Game.from_fixture(dense(kind)))
                before = dump_game(game)
                ended, decision = adjudicate_empirical(game, referee)
                item = {"fixture": kind, "expected_halfpoints": expected, "decision": asdict(decision)}
                fixtures.append(item)
                write("fixtures.json", fixtures)
                evidence = json.loads(decision.evidence)
                if (decision.status != "resolved" or decision.white_minus_black_halfpoints != expected
                        or [run["terminal"]["white_minus_black_halfpoints"] for run in evidence["runs"]] != [expected, expected]):
                    raise ValueError("Real referee failed an exact scoring fixture: " + kind)
                authority = frozenset({digest(json.loads(decision.evidence))})
                restored = load_game(dump_game(ended), allow_fixture=True, empirical_referee=referee,
                                     authorized_hashes=authority)
                if restored != ended or dump_game(game) != before:
                    raise ValueError("Referee persistence or input preservation failed")
                catalog.extend(authority)
                print(json.dumps({"stage": "fixture", "name": kind, "passed": True}), flush=True)
            for kind, prior_pass in (("settled", False), ("save", False), ("settled", True)):
                for mover in ("B", "W"):
                    check_budget()
                    board = dense(kind)
                    if mover == "W":
                        board = [row.translate(str.maketrans("BW", "WB")) for row in board]
                    game = Game.from_fixture(board, to_play=("W" if mover == "B" else "B") if prior_pass else mover)
                    if prior_pass:
                        game = game.play(game.position.to_play, PASS)
                    expected = "premature" if kind == "save" else "acceptable"
                    proof, decision = judge_pass(game), referee.judge_pass(game)
                    item = {"fixture": kind, "mover": mover, "prior_pass": prior_pass,
                            "expected": expected, "certified_verdict": proof.verdict, "decision": asdict(decision)}
                    pass_fixtures.append(item)
                    write("pass_fixtures.json", pass_fixtures)
                    if proof.verdict != expected or decision.verdict not in (expected, "unjudgeable"):
                        raise ValueError("Pass fixture has a contradictory judgment")
            for index, record in enumerate(endings):
                check_budget()
                game = replay(record)
                original = dump_game(game)
                ended, decision = adjudicate_empirical(game, referee)
                evidence_hash = digest(json.loads(decision.evidence))
                item = {"source_url": record["source"]["source_url"], "recorded_result": record["recorded_result"],
                        "decision": asdict(decision), "evidence_sha256": evidence_hash,
                        "persistence_verified": False}
                write(f"ending-{index:03d}.json", item)
                if decision.status == "resolved":
                    restored = load_game(dump_game(ended), empirical_referee=referee,
                                         authorized_hashes=frozenset({evidence_hash}))
                    if restored != ended:
                        raise ValueError("Empirical score did not survive verified persistence")
                    item["persistence_verified"] = True
                    catalog.append(evidence_hash)
                if dump_game(game) != original or ended.moves != game.moves or ended.positions != game.positions:
                    raise ValueError("Referee changed actual play")
                results.append(item)
                write(f"ending-{index:03d}.json", item)
                print(json.dumps({"stage": "ending", "completed": index + 1, "total": len(endings),
                                  "status": decision.status, "reason": decision.reason}), flush=True)
            for index, (record, ply) in enumerate(passes):
                check_budget()
                game = replay(record, ply)
                original = dump_game(game)
                decision = referee.judge_pass(game)
                if dump_game(game) != original:
                    raise ValueError("Pass judgment changed actual play")
                item = {"source_url": record["source"]["source_url"], "pre_move_index": ply,
                        "color": game.position.to_play, "prior_pass_streak": game.position.pass_streak,
                        "decision": asdict(decision)}
                judgments.append(item)
                write(f"pass-{index:03d}.json", item)
                print(json.dumps({"stage": "pass", "completed": index + 1, "total": len(passes),
                                  "verdict": decision.verdict, "reason": decision.reason}), flush=True)
            check_budget()
    write("authorized_evidence.json", {"profile_sha256": pin, "evidence_sha256": sorted(catalog)})
    summary = {"profile_sha256": pin, "fixtures": len(fixtures), "endings": len(results),
               "pass_fixtures": len(pass_fixtures),
               "neural_pass_fixture_verdicts": dict(Counter(r["decision"]["verdict"] for r in pass_fixtures)),
               "ending_statuses": dict(Counter(r["decision"]["status"] for r in results)),
               "ending_reasons": dict(Counter(r["decision"]["reason"] for r in results)),
               "passes": len(judgments), "pass_verdicts": dict(Counter(r["decision"]["verdict"] for r in judgments)),
               "pass_reasons": dict(Counter(r["decision"]["reason"] for r in judgments)),
               "cost": {"cpu_seconds": cpu() - start_cpu, "wall_seconds": time.monotonic() - started,
                        "http_requests": 0, "gpu_seconds": 0},
               "trace_sha256": file_hash(trace_path), "trace_bytes": trace_path.stat().st_size,
               "authorization_sha256": file_hash(output / "authorized_evidence.json")}
    write("summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("profile_sha256")
    parser.add_argument("output", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fixtures-only", action="store_true")
    mode.add_argument("--revision-probe", action="store_true")
    args = parser.parse_args()
    run(args.profile, args.profile_sha256, args.output, args.fixtures_only, args.revision_probe)
