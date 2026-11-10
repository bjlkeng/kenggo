"""Conservative eligibility and handle grouping for the frozen KGS audit."""

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import date
import hashlib
import json
from pathlib import Path
import re
import time

import pyarrow as pa
import pyarrow.parquet as pq

from kenggo.kgs_source import RANKS


POLICY = "kgs_eligibility_v1"
AUDIT_SHA256 = "958147c8cd9a05e1db35553c1c8a7a789ec4fa1393c7b60d232047ffb066f30a"
SAMPLE_SHA256 = "3beace1967c494dd8063fb2ce3d167388366020ccb376d054ea9220c3076284d"
HISTORICAL_BOT_HANDLES = ("gnugo2", "libertybot")
EXCLUDED_TYPES = ("Free", "Review", "Teaching", "Demo", "Demonstration", "Simul",
                  "Simultaneous", "Rengo", "Tournament", "Private", "KGS Plus")


def normalize_handle(handle):
    if not isinstance(handle, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9]{0,9}", handle) is None:
        raise ValueError("Expected a documented KGS ASCII handle, not an inferred identity")
    return handle.lower()


def handle_groups(handles, *, reviewed_aliases=()):
    """Aliases come from separately reviewed evidence, never an incoming game row."""
    parent = {normalize_handle(h): normalize_handle(h) for h in handles}

    def find(name):
        parent.setdefault(name, name)
        while parent[name] != name:
            name = parent[name]
        return name

    for left, right in reviewed_aliases:
        a, b = find(normalize_handle(left)), find(normalize_handle(right))
        parent[max(a, b)] = min(a, b)
    return {name: "kgs-handle:" + find(name) for name in sorted(parent)}


@dataclass(frozen=True)
class Exposure:
    grouping_sha256: str
    groups: frozenset[str]
    source_ids: frozenset[str]
    sgf_hashes: frozenset[str]
    sequence_hashes: frozenset[str]
    account_months: frozenset[tuple[str, str]]


def grouping_digest(groups):
    return hashlib.sha256(json.dumps(groups, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def exposure_registry(records, discovery_accounts, groups, *, probe_handles=("cu2",)):
    handles = {normalize_handle(h) for h in probe_handles}
    slots = {(normalize_handle(h), "2025-09") for h in probe_handles}
    for account in discovery_accounts:
        handle = normalize_handle(account["user"])
        handles.add(handle)
        slots.update((handle, f"2025-{month:02d}") for month in account["months"])
    for record in records:
        for field in ("PB", "PW"):
            handle = record.get("metadata", {}).get(field)
            try:
                handles.add(normalize_handle(handle))
            except ValueError:
                pass
    if any(handle not in groups for handle in handles):
        raise ValueError("Exposure registry omits a known handle group")
    return Exposure(grouping_digest(groups), frozenset(groups[h] for h in handles),
                    frozenset(r["source_url"] for r in records),
                    frozenset(r["sgf_sha256"] for r in records if r.get("sgf_sha256")),
                    frozenset(r["sequence_fingerprint"] for r in records if r.get("sequence_fingerprint")),
                    frozenset(slots))


def classify(record, archive_row, groups, exposure):
    """Version 1 has no reviewed affirmative human/ordinary/reuse evidence."""
    if grouping_digest(groups) != exposure.grouping_sha256:
        raise ValueError("Changed alias/group registry; rebuild exposure and affected splits")
    metadata = record.get("metadata", {})
    source = record.get("source_url")
    if not isinstance(source, str) or not source:
        raise ValueError("A source ID is required even for a failed retrieval")
    if archive_row is not None and archive_row.get("url") != source:
        raise ValueError("Archive evidence belongs to a different source ID")
    reasons = set("metadata:" + reason for reason in record.get("exclusions", ()))
    hard_exclusion = bool(reasons) or record.get("parse_status") != "parsed"
    if record.get("parse_status") != "parsed":
        reasons.add("parse_unverified")
    known_bots = {groups.get(h, "kgs-handle:" + h) for h in HISTORICAL_BOT_HANDLES}
    normalized, keys, humans = [], [], []
    for color, field in (("black", "PB"), ("white", "PW")):
        try:
            handle = normalize_handle(metadata.get(field))
        except ValueError:
            handle = None
        key = groups.get(handle)
        normalized.append(handle)
        keys.append(key)
        if key is None:
            humans.append("unverified")
            reasons.add(color + "_handle_unavailable")
            hard_exclusion = True
        elif key in known_bots:
            humans.append("historical_bot_handle_exclusion")
            reasons.add(color + "_historical_bot_handle")
            hard_exclusion = True
        else:
            humans.append("unverified")
            reasons.add(color + "_human_unverified")
    if keys[0] is not None and keys[0] == keys[1]:
        reasons.add("same_group_on_both_sides")
        hard_exclusion = True
    archive_type = None if archive_row is None else archive_row.get("type")
    if archive_type in EXCLUDED_TYPES:
        ordinary = "excluded_game_type"
        reasons.add("excluded_game_type")
        hard_exclusion = True
    else:
        ordinary = "unverified"
        reasons.add("ordinary_game_unverified")
    if metadata.get("EV") or metadata.get("RO"):
        reasons.add("event_or_round_requires_classification")
        hard_exclusion = True
    reasons.add("training_reuse_basis_unresolved")
    quarantine = set()
    if any(key is None for key in keys):
        quarantine.add("unknown_group")
    if any(key in exposure.groups for key in keys):
        quarantine.add("exposed_handle_group")
    if source in exposure.source_ids:
        quarantine.add("inspected_source_id")
    if record.get("sgf_sha256") in exposure.sgf_hashes:
        quarantine.add("inspected_sgf_bytes")
    if record.get("sequence_fingerprint") in exposure.sequence_hashes:
        quarantine.add("inspected_sequence")
    try:
        month = date.fromisoformat(metadata.get("DT", "")).strftime("%Y-%m")
    except (ValueError, TypeError):
        month = None
        quarantine.add("unknown_date")
    if any((handle, month) in exposure.account_months for handle in normalized):
        quarantine.add("discovery_account_month")
    return {"policy": POLICY, "source_url": source, "sgf_sha256": record.get("sgf_sha256"),
            "black_handle": metadata.get("PB"), "white_handle": metadata.get("PW"),
            "black_group": keys[0], "white_group": keys[1],
            "black_human_status": humans[0], "white_human_status": humans[1],
            "archive_type": archive_type, "ordinary_status": ordinary,
            "metadata_compatible": record.get("parse_status") == "parsed" and not record.get("exclusions"),
            "eligibility_status": "excluded" if hard_exclusion else "unverified",
            "eligibility_reasons": sorted(reasons), "training_eligible": False,
            "black_identity_valid": False, "white_identity_valid": False,
            "black_persona": "default", "white_persona": "default",
            "final_evaluation_excluded": bool(quarantine), "quarantine_reasons": sorted(quarantine)}


def coverage(records):
    by_rank = {rank: {"player_game_appearances": 0, "moves": 0, "explicit_passes": 0,
                      "two_pass_ending_appearances": 0, "result_counts": {}} for rank in RANKS}
    for record in records:
        for color, prop in (("b", "BR"), ("w", "WR")):
            rank = record["metadata"][prop]
            if rank not in by_rank:
                continue
            cell = by_rank[rank]
            cell["player_game_appearances"] += 1
            cell["moves"] += record["actions_by_color"][color]["moves"]
            cell["explicit_passes"] += record["actions_by_color"][color]["explicit_passes"]
            cell["two_pass_ending_appearances"] += record["final_two_explicit_passes"]
            result = record["result_kind"]
            cell["result_counts"][result] = cell["result_counts"].get(result, 0) + 1
    return {"games": len(records), "result_counts": dict(sorted(Counter(r["result_kind"] for r in records).items())),
            "by_player_rank": by_rank,
            "games_with_final_two_passes": sum(r["final_two_explicit_passes"] for r in records)}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(audit_path, sample_path, discovery_path, output):
    started, cpu_start = time.monotonic(), time.process_time()
    if sum(p.stat().st_size for p in (audit_path, sample_path, discovery_path)) > 64 * 1024**2:
        raise ValueError("Policy input byte budget exceeded")
    if sha256(audit_path) != AUDIT_SHA256:
        raise ValueError("Frozen audit hash mismatch")
    if sha256(sample_path) != SAMPLE_SHA256:
        raise ValueError("Frozen sample hash mismatch")
    sample = json.loads(sample_path.read_text())
    discovery = json.loads(discovery_path.read_text())
    if sample["config_sha256"] != sha256(discovery_path):
        raise ValueError("Discovery provenance does not match the frozen sample")
    records = [json.loads(row["audit_json"]) for row in pq.read_table(audit_path).to_pylist()]
    archived = {row["url"]: row for row in sample["candidates"]}
    if (len(records) != 1000 or len(archived) != 1000 or len(sample["candidates"]) != 1000
            or set(archived) != {r["source_url"] for r in records}):
        raise ValueError("Expected exactly the frozen 1000 distinct source IDs")
    handles = {r["metadata"][field] for r in records for field in ("PB", "PW")}
    groups = handle_groups(handles | {a["user"] for a in discovery["accounts"]} | {"cu2"})
    exposure = exposure_registry(records, discovery["accounts"], groups)
    decisions = [classify(r, archived[r["source_url"]], groups, exposure) for r in records]
    accepted = [r for r, d in zip(records, decisions) if d["training_eligible"]]
    compatible = [r for r in records if not r["exclusions"]]
    if time.monotonic() - started > 600 or time.process_time() - cpu_start > 300:
        raise RuntimeError("Policy audit budget exceeded")
    output.mkdir(parents=True, exist_ok=False)
    pq.write_table(pa.Table.from_pylist(decisions), output / "decisions.parquet", compression="zstd")
    (output / "decisions.json").write_text(json.dumps(decisions, indent=2) + "\n")
    summary = {"policy": POLICY, "audit_sha256": AUDIT_SHA256,
               "sample_sha256": sha256(sample_path), "discovery_sha256": sha256(discovery_path),
               "code_sha256": sha256(Path(__file__)), "decisions_parquet_sha256": sha256(output / "decisions.parquet"),
               "input_games": len(records), "metadata_compatible_games": len(compatible),
               "status_counts": dict(sorted(Counter(d["eligibility_status"] for d in decisions).items())),
               "reason_counts": dict(sorted(Counter(x for d in decisions for x in d["eligibility_reasons"]).items())),
               "historical_bot_exclusion_games": sum(any(d[c + "_human_status"] == "historical_bot_handle_exclusion"
                                                        for c in ("black", "white")) for d in decisions),
               "ordinary_verified_games": 0, "human_verified_games": 0, "training_eligible_games": len(accepted),
               "identity_valid_game_sides": 0, "reviewed_alias_edges": [],
               "grouping_sha256": exposure.grouping_sha256,
               "observed_handle_groups": len({groups[normalize_handle(h)] for h in handles}),
               "metadata_compatible_handle_groups": len({groups[normalize_handle(r["metadata"][p])]
                                                          for r in compatible for p in ("PB", "PW")}),
               "final_quarantine_groups": sorted(exposure.groups),
               "original_discovery_handles": len(discovery["accounts"]),
               "quarantine_reason_counts": dict(sorted(Counter(x for d in decisions for x in d["quarantine_reasons"]).items())),
               "final_eligible_sampled_games": sum(not d["final_evaluation_excluded"] for d in decisions),
               "metadata_compatible_coverage": coverage(compatible), "eligible_coverage": coverage(accepted),
               "source_use_scope": "local_audit_only; no affirmative training or redistribution basis established",
               "costs": {"wall_seconds": time.monotonic() - started, "cpu_seconds": time.process_time() - cpu_start,
                         "http_requests": 0, "raw_sgf_reads": 0, "gpu_hours": 0,
                         "scope": "One frozen-metadata policy pass and per-game writes; excludes imports and final summary write."}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("policy", "input_games", "metadata_compatible_games",
                     "status_counts", "training_eligible_games", "observed_handle_groups",
                     "metadata_compatible_handle_groups", "final_eligible_sampled_games", "costs")}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("sample", type=Path)
    parser.add_argument("discovery", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit(args.audit, args.sample, args.discovery, args.output)


if __name__ == "__main__":
    main()
