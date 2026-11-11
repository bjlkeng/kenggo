"""Summarize every case in a completed, frozen exposed-source referee panel."""

import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(directory):
    directory = Path(directory)
    panel = json.loads((directory / "panel.json").read_text())
    run = json.loads((directory / "run.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    if run["fixtures_only"] or len(panel["ending_source_ids"]) != 108 or len(panel["passes"]) != 24:
        raise ValueError("A completed full source panel is required")
    rows, comparable, winners, equal_margins = [], 0, 0, 0
    for index, source_id in enumerate(panel["ending_source_ids"]):
        path = directory / f"ending-{index:03d}.json"
        item = json.loads(path.read_text())
        decision, recorded = item["decision"], item["recorded_result"]
        evidence = json.loads(decision["evidence"])
        if item["source_url"] != source_id or evidence["profile_sha256"] != summary["profile_sha256"]:
            raise ValueError("Ending differs from the frozen panel or profile")
        score = decision["white_minus_black_halfpoints"]
        native_scores = [r["terminal"]["white_minus_black_halfpoints"] if r["terminal"] else None
                         for r in evidence.get("runs", [])]
        row = {"case": index, "source_url": source_id, "recorded_result": recorded,
               "status": decision["status"], "reason": decision["reason"],
               "native_scores_halfpoints": native_scores,
               "selected_white_minus_black_halfpoints": score,
               "cleanup_moves": [len(r["cleanup"]) for r in evidence.get("runs", [])],
               "evidence_sha256": item["evidence_sha256"], "artifact_sha256": file_hash(path),
               "cpu_seconds": evidence["cost"]["cpu_seconds"],
               "wall_seconds": evidence["cost"]["wall_seconds"]}
        if decision["status"] == "resolved" and recorded["kind"] == "scored":
            expected = Decimal(recorded["margin"]) * 2 * (1 if recorded["winner"] == "W" else -1)
            if expected != int(expected) or type(score) is not int or score % 2 != 1:
                raise ValueError("Invalid scored comparison")
            row.update(recorded_white_minus_black_halfpoints=int(expected),
                       margin_delta_halfpoints=score - int(expected),
                       winner_agrees=("W" if score > 0 else "B") == recorded["winner"])
            comparable += 1
            winners += row["winner_agrees"]
            equal_margins += score == int(expected)
        rows.append(row)
    passes = []
    for index, selected in enumerate(panel["passes"]):
        path = directory / f"pass-{index:03d}.json"
        item = json.loads(path.read_text())
        if any(item[key] != value for key, value in selected.items()):
            raise ValueError("Pass differs from the frozen panel")
        passes.append({key: value for key, value in item.items() if key != "decision"} | {
            "case": index, "verdict": item["decision"]["verdict"], "reason": item["decision"]["reason"],
            "gains_points": item["decision"]["gains_points"], "artifact_sha256": file_hash(path)})
    resolved = sum(row["status"] == "resolved" for row in rows)
    if (summary["endings"] != len(rows) or summary["passes"] != len(passes)
            or summary["ending_statuses"] != dict(Counter(row["status"] for row in rows))
            or summary["pass_verdicts"] != dict(Counter(row["verdict"] for row in passes))):
        raise ValueError("Completed summary does not account for every case")
    return {"version": "empirical_referee_comparison_v1", "directory": str(directory),
            "profile_sha256": summary["profile_sha256"],
            "artifacts": {name: file_hash(directory / name) for name in
                          ("run.json", "panel.json", "summary.json", "fixtures.json", "authorized_evidence.json")},
            "trace_sha256": summary["trace_sha256"], "trace_bytes": summary["trace_bytes"],
            "ending_counts": {"total": len(rows), "resolved": resolved, "unresolved": len(rows) - resolved,
                              "known_scored_comparisons": comparable, "winner_agrees": winners,
                              "exact_recorded_margin_agrees": equal_margins},
            "readiness_check": {"minimum_resolution": 0.9, "minimum_recorded_winner_agreement": 0.95,
                                "resolution_passes": resolved * 10 >= len(rows) * 9,
                                "winner_agreement_passes": comparable > 0 and winners * 100 >= comparable * 95},
            "ending_reasons": summary["ending_reasons"], "pass_verdicts": summary["pass_verdicts"],
            "pass_reasons": summary["pass_reasons"], "cost": summary["cost"],
            "endings": rows, "passes": passes,
            "claim": "Exposed-source implementation readiness only; no final evaluation or candidate qualification."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = summarize(args.directory)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("ending_counts", "readiness_check", "pass_verdicts", "cost")}, indent=2))
