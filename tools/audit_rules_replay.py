#!/usr/bin/env python3
"""Offline validation of the frozen audit's 241 metadata-compatible records."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import time
import zipfile

import pyarrow.parquet as pq

from kenggo.adjudication import adjudicate
from kenggo.go import RULES
from kenggo.rules_replay import replay_sgf


AUDIT_SHA256 = "958147c8cd9a05e1db35553c1c8a7a789ec4fa1393c7b60d232047ffb066f30a"


def cpu():
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return time.process_time() + children.ru_utime + children.ru_stime


def run(args):
    started, cpu_start = time.monotonic(), cpu()
    audit_bytes = args.audit.read_bytes()
    if hashlib.sha256(audit_bytes).hexdigest() != AUDIT_SHA256:
        raise ValueError("Frozen audit hash mismatch")
    all_rows = [json.loads(row["audit_json"]) for row in pq.read_table(args.audit).to_pylist()]
    selected = [row for row in all_rows if not row["exclusions"]]
    if len(all_rows) != 1000 or len(selected) != 241:
        raise ValueError("Unexpected validation population")
    counts, records = Counter(), []
    input_bytes = output_bytes = 0
    for row in selected:
        if time.monotonic() - started > 600 or cpu() - cpu_start > 300:
            raise RuntimeError("Offline validation budget exceeded")
        with zipfile.ZipFile(args.source / (row["page_key"] + ".zip")) as archive:
            if archive.getinfo(row["zip_member"]).file_size > 8 * 1024 * 1024:
                raise ValueError("Selected SGF exceeds size budget")
            data = archive.read(row["zip_member"])
        input_bytes += len(data)
        if input_bytes > 64 * 1024 * 1024 or hashlib.sha256(data).hexdigest() != row["sgf_sha256"]:
            raise ValueError("SGF byte budget/hash mismatch")
        record = {"sgf_sha256": row["sgf_sha256"], "actions": row["counters"]["moves"]}
        try:
            replay = replay_sgf(data)
            case = {"moves": [{"color": m.color, "action": m.action} for m in replay.game.moves]}
            result = subprocess.run([str(args.reference.resolve())], input=json.dumps(case) + "\n",
                                    text=True, capture_output=True, check=True, timeout=30)
            output_bytes += len(result.stdout.encode())
            if output_bytes > 256 * 1024 * 1024:
                raise RuntimeError("Native output byte budget exceeded")
            native = json.loads(result.stdout)
            if not native["ok"] or result.stderr or len(native["steps"]) != len(replay.game.moves):
                raise ValueError("Invalid native reference response")
            for position, step in zip(replay.game.positions[1:], native["steps"]):
                state = step["state"]
                if (not step["accepted"] or tuple(state["board"]) != position.rows
                        or state["to_play"] != position.to_play or state["simple_ko"] != position.ko
                        or state["pass_streak"] != position.pass_streak
                        or state["captures"] != {"B": position.captured_black, "W": position.captured_white}):
                    raise ValueError("Native replay disagreement")
            counts["legally_replayed"] += 1
            counts["compared_actions"] += len(replay.game.moves)
            counts["passes"] += sum(m.action == 361 for m in replay.game.moves)
            record.update(status=replay.game.status, snapshot=replay.game.snapshot)
            counts[replay.game.status] += 1
            if replay.game.status == "pending_adjudication":
                adjudicated, decision = adjudicate(replay.game)
                counts["adjudication_" + adjudicated.status] += 1
                record["adjudication"] = decision.reason
        except (ValueError, subprocess.SubprocessError) as error:
            counts["failed_games"] += 1
            record.update(status="failed", reason=str(error))
        records.append(record)
    counts.setdefault("failed_games", 0)
    output = {"rules": RULES, "audit_sha256": AUDIT_SHA256, "selected_games": len(selected),
              "excluded_not_replayed": len(all_rows) - len(selected), "counts": dict(counts),
              "selected_sgf_bytes": input_bytes, "reference_output_bytes": output_bytes,
              "reference_executable_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
              "costs": {"wall_seconds": time.monotonic() - started, "cpu_seconds_including_children": cpu() - cpu_start,
                        "wall_limit_seconds": 600, "cpu_limit_seconds": 300, "workers": 1,
                        "http_requests": 0, "gpu_hours": 0,
                        "scope": "One validation run; excludes imports/startup/final write. Not a throughput benchmark."},
              "training_ready": False, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k != "records"}, indent=2))
    return 1 if counts["failed_games"] else 0


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=root / "manifests/e1_kgs_audit_games_2026-11-10.parquet")
    parser.add_argument("--source", type=Path, default=root / "runs/e1-kgs-audit-2026-11-10-v4")
    parser.add_argument("--reference", type=Path, default=root / ".cache/katago-reference/katago-reference")
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
