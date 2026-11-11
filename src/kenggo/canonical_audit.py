"""Build diagnostic canonical records and frozen splits from the cached KGS audit."""

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq

from kenggo.canonical_data import canonicalize, digest, encoded
from kenggo.data_splits import SplitPlan, build_splits, manifest_contents, write_membership
from kenggo.e1_audit import member_key, source_key
from kenggo.kgs_eligibility import (AUDIT_SHA256, SAMPLE_SHA256, POLICY, classify,
                                    exposure_registry, handle_groups)


AUDIT_MANIFEST_SHA256 = "3465ec3b374682b4f10bc479d02cccd9f77f96a4e0832456a64c6e9c3d46fe3f"
ELIGIBILITY_SHA256 = "57a3b6e520bf53d9b69bfac4db56d40108117e8d6438acaef141862082bd1dd7"
AUDIT_PATH = Path("manifests/e1_kgs_audit_games_2026-11-10.parquet")
AUDIT_MANIFEST = Path("manifests/e1_kgs_audit_2026-11-10.json")
ELIGIBILITY_PATH = Path("manifests/e1_kgs_eligibility_games_2026-11-10.parquet")
DISCOVERY_PATH = Path("manifests/e1_kgs_discovery_2026-11-10_v2.json")


def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def load_frozen(source, *, audit_path=AUDIT_PATH, eligibility_path=ELIGIBILITY_PATH):
    paths = (audit_path, eligibility_path, AUDIT_MANIFEST, DISCOVERY_PATH, source / "sample.json")
    if sum(p.stat().st_size for p in paths) > 64 * 1024**2:
        raise ValueError("Metadata input byte budget exceeded")
    for path, expected in ((audit_path, AUDIT_SHA256), (eligibility_path, ELIGIBILITY_SHA256),
                           (AUDIT_MANIFEST, AUDIT_MANIFEST_SHA256), (source / "sample.json", SAMPLE_SHA256)):
        if file_hash(path) != expected:
            raise ValueError("Frozen input hash mismatch: " + str(path))
    sample = json.loads((source / "sample.json").read_text())
    discovery = json.loads(DISCOVERY_PATH.read_text())
    if sample["config_sha256"] != file_hash(DISCOVERY_PATH):
        raise ValueError("Discovery provenance mismatch")
    audits = [json.loads(r["audit_json"]) for r in pq.read_table(audit_path).to_pylist()]
    decisions = pq.read_table(eligibility_path).to_pylist()
    archived = {r["url"]: r for r in sample["candidates"]}
    gates = {r["source_url"]: r for r in decisions}
    if (len(audits) != 1000 or len(decisions) != 1000 or len(archived) != 1000
            or len(gates) != 1000 or set(gates) != set(archived)
            or {r["source_url"] for r in audits} != set(gates)):
        raise ValueError("Expected exactly the frozen 1,000 source IDs")
    handles = {r["metadata"][p] for r in audits for p in ("PB", "PW")}
    groups = handle_groups(handles | {a["user"] for a in discovery["accounts"]} | {"cu2"})
    exposure = exposure_registry(audits, discovery["accounts"], groups)
    for record in audits:
        if classify(record, archived[record["source_url"]], groups, exposure) != gates[record["source_url"]]:
            raise ValueError("Eligibility decisions no longer reproduce under the pinned policy")
    return audits, archived, gates, exposure, json.loads(AUDIT_MANIFEST.read_text())


def run(source, config_path, output):
    started, cpu_start = time.monotonic(), time.process_time()
    started_utc = datetime.now(timezone.utc).isoformat()
    if output.exists():
        raise FileExistsError("Canonical output already exists")

    def check_budget():
        if time.monotonic() - started > 1800 or time.process_time() - cpu_start > 900:
            raise RuntimeError("Canonical audit time budget exceeded")

    audits, archived, gates, exposure, original = load_frozen(source)
    plan = SplitPlan.from_dict(json.loads(config_path.read_text()))
    if plan.grouping_sha256 != exposure.grouping_sha256:
        raise ValueError("Split configuration uses a different alias/group registry")
    records, entries, source_refs, adjudications = [], [], [], []
    selected_bytes = expanded_bytes = canonical_bytes = 0
    bundle_hashes = {}
    with ExitStack() as stack:
        archives = {}
        for bundle in original["summary"]["bundles"]:
            check_budget()
            key = bundle["page"]
            path = source / (key + ".zip")
            if path.stat().st_size > 8 * 1024**2 or file_hash(path) != bundle["sha256"]:
                raise ValueError("Original bundle size/hash mismatch")
            archive = stack.enter_context(zipfile.ZipFile(path))
            expanded_bytes += sum(info.file_size for info in archive.infolist())
            if expanded_bytes > 256 * 1024**2:
                raise ValueError("Expanded ZIP budget exceeded")
            archives[key] = archive
            bundle_hashes[key] = bundle["sha256"]
        for prior in sorted(audits, key=lambda r: r["source_url"]):
            check_budget()
            archive = archives[prior["page_key"]]
            matches = [info for info in archive.infolist() if member_key(info.filename) == source_key(prior["source_url"])]
            if len(matches) != 1 or matches[0].filename != prior["zip_member"] or matches[0].file_size > 1024**2:
                raise ValueError("Selected ZIP reference is ambiguous or oversized")
            with archive.open(matches[0]) as stream:
                data = stream.read(1024**2 + 1)
            selected_bytes += len(data)
            if len(data) > 1024**2 or selected_bytes > 64 * 1024**2:
                raise ValueError("Selected SGF byte budget exceeded")
            ref = {"source_url": prior["source_url"], "sgf_sha256": prior["sgf_sha256"],
                   "archive_path": str(source / (prior["page_key"] + ".zip")),
                   "archive_sha256": bundle_hashes[prior["page_key"]], "zip_member": prior["zip_member"]}
            record, call_cost = canonicalize(data, ref, archived[prior["source_url"]])
            gate = gates[prior["source_url"]]
            record["eligibility"] = gate
            if call_cost is not None:
                adjudications.append({"source_url": prior["source_url"], "sgf_sha256": prior["sgf_sha256"],
                                      "canonical_sha256": digest(record),
                                      "decision_sha256": digest(record["certification"]),
                                      "snapshot": record["certification"]["snapshot"],
                                      "method": record["certification"]["method"], "cost": call_cost})
            canonical_json = encoded(record)
            canonical_bytes += len(canonical_json.encode())
            if canonical_bytes > 256 * 1024**2:
                raise ValueError("Canonical output byte budget exceeded")
            records.append(record)
            source_refs.append(ref)
            entries.append({"source_url": prior["source_url"], "canonical_sha256": digest(record),
                            "grouping_sha256": exposure.grouping_sha256,
                            "sgf_sha256": prior["sgf_sha256"], "audit_sequence_sha256": prior.get("sequence_fingerprint"),
                            "trajectory_sha256": record["trajectory_sha256"], "semantic_sha256": record["semantic_sha256"],
                            "groups": [gate["black_group"], gate["white_group"]], "date": record["metadata"].get("DT"),
                            "admitted": gate["training_eligible"], "identity_valid": [gate["black_identity_valid"], gate["white_identity_valid"]],
                            "final_excluded": gate["final_evaluation_excluded"],
                            "exclusions": sorted(set(record["structural_exclusions"] + gate["eligibility_reasons"])),
                            "actions": len(record["actions"]),
                            "actions_by_color": [sum(a["color"] == c for a in record["actions"]) for c in ("B", "W")],
                            "value_supported": record["recorded_value_supported"]})
    rows = build_splits(entries, plan, exposed_groups=exposure.groups)
    exposure_hash = digest({"groups": sorted(exposure.groups), "source_ids": sorted(exposure.source_ids),
                            "bytes": sorted(exposure.sgf_hashes), "sequences": sorted(exposure.sequence_hashes),
                            "account_months": sorted(exposure.account_months)})
    code_names = ("canonical_audit.py", "canonical_data.py", "data_splits.py", "kgs_eligibility.py",
                  "kgs_source.py", "e1_audit.py", "sgf_audit.py", "rules_replay.py", "go.py", "adjudication.py")
    provenance = {"audit_sha256": AUDIT_SHA256, "sample_sha256": SAMPLE_SHA256,
                  "eligibility_sha256": ELIGIBILITY_SHA256, "eligibility_policy": POLICY,
                  "audit_manifest_sha256": AUDIT_MANIFEST_SHA256,
                  "discovery_sha256": file_hash(DISCOVERY_PATH), "config_file_sha256": file_hash(config_path),
                  "uv_lock_sha256": file_hash(Path("uv.lock")),
                  "source_references_sha256": digest(source_refs),
                  "code_sha256": {name: file_hash(Path(__file__).with_name(name)) for name in code_names}}
    split_manifest = manifest_contents(entries, rows, plan, exposure_sha256=exposure_hash, provenance=provenance)
    output.mkdir(parents=True, exist_ok=False)
    canonical_table = pa.Table.from_pylist([
        {"source_url": r["source"]["source_url"], "canonical_sha256": digest(r), "canonical_json": encoded(r)} for r in records])
    pq.write_table(canonical_table, output / "canonical.parquet", compression="zstd")
    split_hash = write_membership(output / "splits", rows, split_manifest)
    assigned = {r["source_url"]: r for r in rows}
    index = []
    for record in records:
        membership = assigned[record["source"]["source_url"]]
        gate = record["eligibility"]
        index.append({"source_url": record["source"]["source_url"], "canonical_sha256": digest(record),
                      "source_ref_json": encoded(record["source"]), "metadata_json": encoded(record["metadata"]),
                      "structural_exclusions": record["structural_exclusions"],
                      "replay_validated": record["replay_validated"], "replay_end_status": record["replay_end_status"],
                      "observed_actions": len(record["actions"]), "clock_nodes": len(record["clock_nodes"]),
                      "recorded_value_supported": record["recorded_value_supported"],
                      "source_admitted": gate["training_eligible"], "identity_valid": membership["identity_valid"],
                      "groups": membership["groups"], "role": membership["role"],
                      "split_exclusions": membership["exclusions"], "final_excluded": membership["final_excluded"],
                      "duplicate_representative": membership["duplicate_representative"]})
    pq.write_table(pa.Table.from_pylist(index), output / "index.parquet", compression="zstd")
    (output / "adjudications.json").write_text(json.dumps(adjudications, indent=2) + "\n")
    check_budget()
    summary = {"version": "canonical_audit_v1", "provenance": provenance, "records": len(records),
               "parsed_records": sum(r["audit"]["parse_status"] == "parsed" for r in records),
               "structurally_supported_games": sum(not r["structural_exclusions"] for r in records),
               "replay_validated_games": sum(r["replay_validated"] for r in records),
               "replay_validated_actions": sum(len(r["actions"]) for r in records if r["replay_validated"]),
               "observed_actions": sum(len(r["actions"]) for r in records),
               "structural_exclusion_counts": dict(sorted(Counter(x for r in records for x in r["structural_exclusions"]).items())),
               "recorded_value_supported_games": sum(r["recorded_value_supported"] for r in records),
               "source_admitted_games": sum(g["training_eligible"] for g in gates.values()),
               "duplicate_components": len({r["duplicate_representative"] for r in rows}),
               "duplicate_excess": sum(r["source_url"] != r["duplicate_representative"] for r in rows),
               "adjudication_calls": len(adjudications),
               "adjudication_execution": {"file": "adjudications.json", "sha256": file_hash(output / "adjudications.json"),
                                          "cpu_seconds": sum(a["cost"]["cpu_seconds"] for a in adjudications),
                                          "wall_seconds": sum(a["cost"]["wall_seconds"] for a in adjudications),
                                          "scope": "Full per-call limits/counts/costs, linked to canonical records and deterministic certificate hashes."},
               "split_manifest": split_manifest, "split_manifest_sha256": split_hash,
               "canonical_parquet_sha256": file_hash(output / "canonical.parquet"),
               "index_parquet_sha256": file_hash(output / "index.parquet"),
               "bytes": {"selected_sgf": selected_bytes, "expanded_bundles": expanded_bytes,
                         "canonical_json": canonical_bytes, "canonical_parquet": (output / "canonical.parquet").stat().st_size},
               "costs": {"started_utc": started_utc, "cpu_seconds": time.process_time() - cpu_start, "wall_seconds": time.monotonic() - started,
                         "workers": 1, "http_requests": 0, "gpu_hours": 0,
                         "scope": "Frozen-input verification, selected SGF parsing/replay and artifact writing; excludes imports and final summary write."},
               "training_ready": False, "e1_ready": False}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("records", "structurally_supported_games", "replay_validated_games",
                     "source_admitted_games", "duplicate_excess", "costs")}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path, default=Path("manifests/e1_split_config_v1.json"))
    args = parser.parse_args()
    run(args.source, args.config, args.output)


if __name__ == "__main__":
    main()
