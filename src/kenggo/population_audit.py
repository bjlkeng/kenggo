"""Build admitted population records while keeping a separately reserved final sample sealed."""

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import date, datetime, timezone
import json
from pathlib import Path
import re
import time
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq

from kenggo.canonical_audit import DISCOVERY_PATH, load_frozen
from kenggo.canonical_data import calendar_day, canonicalize, digest, encoded
from kenggo.data_splits import SplitPlan, build_splits, manifest_contents, write_membership
from kenggo.e1_audit import member_key, source_key
from kenggo.examples import file_hash, read_pinned_json
from kenggo.kgs_admission import DECISION_SHA256, POLICY, classify
from kenggo.kgs_eligibility import exposure_registry, handle_groups, normalize_handle
from kenggo.kgs_source import candidate_reasons, player


VERSION = "kgs_population_inputs_v1"
LIMIT_KEYS = {"max_records", "max_selected_sgf_bytes", "max_expanded_zip_bytes",
              "max_canonical_json_bytes", "cpu_seconds", "wall_seconds"}


def pinned_ref(reference, *, max_bytes=64 * 1024**2):
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("An input reference needs a path and an independent SHA-256")
    path = Path(reference["path"])
    expected = reference["sha256"]
    if (not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None
            or path.stat().st_size > max_bytes or file_hash(path) != expected):
        raise ValueError("Frozen population input hash or size mismatch: " + str(path))
    return path


def load_original(source):
    audits, archived, _, exposure, original = load_frozen(source)
    discovery = json.loads(DISCOVERY_PATH.read_text())
    references = []
    bundles = {bundle["page"]: bundle["sha256"] for bundle in original["summary"]["bundles"]}
    for audit in audits:
        references.append({"source_url": audit["source_url"], "sgf_sha256": audit["sgf_sha256"],
                           "archive_path": str(source / (audit["page_key"] + ".zip")),
                           "archive_sha256": bundles[audit["page_key"]], "zip_member": audit["zip_member"]})
    return audits, archived, exposure, discovery["accounts"], references


def read_inputs(path, expected_sha256):
    inputs = read_pinned_json(path, expected_sha256, max_bytes=256 * 1024)
    required = {"version", "original_source_directory", "admission_decisions", "source_policy",
                "final_scope_policy", "sealed_final", "split_config", "limits"}
    if not required <= set(inputs) or inputs["version"] != VERSION:
        raise ValueError("Unsupported population input manifest")
    limits = inputs["limits"]
    if (not isinstance(limits, dict) or set(limits) != LIMIT_KEYS
            or any(type(value) is not int or value <= 0 for value in limits.values())):
        raise ValueError("Population construction requires finite positive limits")
    paths = {name: pinned_ref(inputs[name]) for name in
             ("admission_decisions", "source_policy", "final_scope_policy", "sealed_final", "split_config")}
    if inputs["source_policy"]["sha256"] != DECISION_SHA256:
        raise ValueError("The owner source-policy decision differs from its independent pin")
    final = json.loads(paths["sealed_final"].read_text())
    scope = json.loads(paths["final_scope_policy"].read_text())
    if (final.get("version") != "sealed_kgs_index_v1" or not isinstance(final.get("records"), list)
            or scope.get("version") != "kgs_sealed_final_scope_v1"
            or scope.get("source_policy_sha256") != DECISION_SHA256
            or final.get("reservation_sha256") != scope.get("reservation_sha256")
            or re.fullmatch(r"[0-9a-f]{64}", final.get("reservation_sha256", "")) is None):
        raise ValueError("Sealed final index does not match its frozen scope")
    if not calendar_day(scope["date_from"]) < date.fromisoformat(scope["date_before"]) <= date(2026, 1, 1):
        raise ValueError("Final dates need a nonempty half-open window")
    fragments = scope["suspected_bot_handle_substrings"]
    if (not isinstance(fragments, list) or not fragments
            or any(not isinstance(value, str) or not value or value != value.lower() for value in fragments)):
        raise ValueError("Final screening needs explicit conservative handle-name exclusions")
    reserved = scope["reserved_groups"]
    if (not isinstance(reserved, list) or reserved != sorted(set(reserved))
            or final.get("test_groups") != reserved
            or any(not isinstance(group, str) or not group.startswith("kgs-handle:") for group in reserved)):
        raise ValueError("Final groups differ from the frozen reservation")
    for group in reserved:
        if group != "kgs-handle:" + normalize_handle(group.removeprefix("kgs-handle:")):
            raise ValueError("Reserved group is not a canonical whole-handle group")
    plan = SplitPlan.from_dict(json.loads(paths["split_config"].read_text()))
    if set(plan.test_groups) != set(reserved):
        raise ValueError("The split must preserve every reserved final group")
    return inputs, paths, final, scope, plan


def final_candidate(item, scope):
    if not isinstance(item, dict) or set(item) != {"source", "archive_row"}:
        raise ValueError("A sealed candidate requires exact source and archive references")
    source, archive = item["source"], item["archive_row"]
    if (set(source) != {"source_url", "sgf_sha256", "archive_path", "archive_sha256", "zip_member"}
            or source["source_url"] != archive["url"] or candidate_reasons(archive)):
        raise ValueError("Sealed source reference is outside the frozen archive population")
    key = source_key(source["source_url"])
    if key is None or not scope["date_from"] <= key[0] < scope["date_before"]:
        raise ValueError("Sealed source is outside the reserved dates")
    handles = [normalize_handle(player(archive[color])[0]) for color in ("black", "white")]
    if (not {"kgs-handle:" + handle for handle in handles} <= set(scope["reserved_groups"])
            or any(part in handle for handle in handles for part in scope["suspected_bot_handle_substrings"])):
        raise ValueError("Sealed archive participant fails the reserved-group screening")
    return handles


def split_entry(record, grouping_sha256):
    gate = record["eligibility"]
    return {"source_url": record["source"]["source_url"], "canonical_sha256": digest(record),
            "grouping_sha256": grouping_sha256, "sgf_sha256": record["source"]["sgf_sha256"],
            "audit_sequence_sha256": record["audit"].get("sequence_fingerprint"),
            "trajectory_sha256": record["trajectory_sha256"], "semantic_sha256": record["semantic_sha256"],
            "groups": [gate["black_group"], gate["white_group"]], "date": record["metadata"].get("DT"),
            "admitted": gate["training_eligible"],
            "identity_valid": [gate["black_identity_valid"], gate["white_identity_valid"]],
            "final_excluded": gate["final_evaluation_excluded"],
            "exclusions": sorted(set(record["structural_exclusions"] + gate["eligibility_reasons"])),
            "actions": len(record["actions"]),
            "actions_by_color": [sum(action["color"] == color for action in record["actions"]) for color in ("B", "W")],
            "value_supported": record["recorded_value_supported"]}


def run(inputs_path, expected_sha256, output):
    started, cpu = time.monotonic(), time.process_time()
    started_utc = datetime.now(timezone.utc).isoformat()
    output = Path(output)
    if output.exists():
        raise FileExistsError("Population output already exists")
    inputs, paths, final, scope, plan = read_inputs(inputs_path, expected_sha256)
    limits = inputs["limits"]

    def check_budget():
        if time.process_time() - cpu > limits["cpu_seconds"] or time.monotonic() - started > limits["wall_seconds"]:
            raise RuntimeError("Population construction budget exceeded")

    audits, archived, old_exposure, discovery, original_refs = load_original(Path(inputs["original_source_directory"]))
    if len(audits) + len(final["records"]) > limits["max_records"]:
        raise ValueError("Population record budget exceeded")
    handles = {record["metadata"][side] for record in audits for side in ("PB", "PW")}
    handles.update(account["user"] for account in discovery)
    handles.add("cu2")
    handles.update(group.removeprefix("kgs-handle:") for group in scope["reserved_groups"])
    for item in final["records"]:
        handles.update(final_candidate(item, scope))
    groups = handle_groups(handles)
    exposure = exposure_registry(audits, discovery, groups)
    if any(getattr(exposure, field) != getattr(old_exposure, field) for field in
           ("groups", "source_ids", "sgf_hashes", "sequence_hashes", "account_months")):
        raise ValueError("The original final-exposure quarantine changed")
    if plan.grouping_sha256 != exposure.grouping_sha256 or set(plan.test_groups) & exposure.groups:
        raise ValueError("Frozen group registry or final quarantine mismatch")
    decisions = pq.read_table(paths["admission_decisions"]).to_pylist()
    gates = {row["source_url"]: row for row in decisions}
    if len(decisions) != len(audits) or len(gates) != len(audits) or set(gates) != set(archived):
        raise ValueError("Admission decisions do not cover the original frozen sample")
    for audit in audits:
        if classify(audit, archived[audit["source_url"]], groups, exposure) != gates[audit["source_url"]]:
            raise ValueError("Presumed-human admission decisions do not reproduce")
    items = [{"source": ref, "archive_row": archived[ref["source_url"]], "sealed": False} for ref in original_refs]
    items.extend({**item, "sealed": True} for item in final["records"])
    source_ids = [item["source"]["source_url"] for item in items]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("Original and final candidates must have distinct source IDs")
    sealed_ids = sorted(item["source"]["source_url"] for item in items if item["sealed"])
    records, entries, adjudications = [], [], []
    selected_bytes = expanded_bytes = canonical_bytes = 0
    with ExitStack() as stack:
        archives = {}
        for item in sorted(items, key=lambda item: item["source"]["source_url"]):
            check_budget()
            ref, archive_row = item["source"], item["archive_row"]
            key = (ref["archive_path"], ref["archive_sha256"])
            if key not in archives:
                path = pinned_ref({"path": key[0], "sha256": key[1]}, max_bytes=8 * 1024**2)
                archive = stack.enter_context(zipfile.ZipFile(path))
                expanded_bytes += sum(info.file_size for info in archive.infolist())
                if expanded_bytes > limits["max_expanded_zip_bytes"]:
                    raise ValueError("Expanded ZIP budget exceeded")
                archives[key] = archive
            archive = archives[key]
            matches = [info for info in archive.infolist() if member_key(info.filename) == source_key(ref["source_url"])]
            if len(matches) != 1 or matches[0].filename != ref["zip_member"] or matches[0].file_size > 1024**2:
                raise ValueError("Selected ZIP reference is ambiguous or oversized")
            with archive.open(matches[0]) as stream:
                data = stream.read(1024**2 + 1)
            selected_bytes += len(data)
            if len(data) > 1024**2 or selected_bytes > limits["max_selected_sgf_bytes"]:
                raise ValueError("Selected SGF byte budget exceeded")
            record, call_cost = canonicalize(data, ref, archive_row)
            record["sealed_final"] = item["sealed"]
            if item["sealed"]:
                audit = {**record["audit"], "source_url": ref["source_url"]}
                gate = classify(audit, archive_row, groups, exposure)
                for color in ("black", "white"):
                    handle = gate[color + "_handle"]
                    if isinstance(handle, str) and any(part in handle.lower() for part in scope["suspected_bot_handle_substrings"]):
                        gate["eligibility_reasons"].append(color + "_suspected_bot_handle")
                        gate["training_eligible"] = False
                        gate["eligibility_status"] = "excluded"
                gate["eligibility_reasons"].sort()
                gate["final_scope_policy_sha256"] = inputs["final_scope_policy"]["sha256"]
            else:
                gate = gates[ref["source_url"]]
            record["eligibility"] = gate
            if call_cost is not None:
                adjudications.append({"source_url": ref["source_url"], "canonical_sha256": digest(record),
                                      "sealed_final": item["sealed"], "decision_sha256": digest(record["certification"]),
                                      "method": record["certification"]["method"], "cost": call_cost})
            canonical_bytes += len(encoded(record).encode())
            if canonical_bytes > limits["max_canonical_json_bytes"]:
                raise ValueError("Canonical JSON budget exceeded")
            records.append(record)
            entries.append(split_entry(record, exposure.grouping_sha256))
    rows = build_splits(entries, plan, exposed_groups=exposure.groups, group_registry=groups)
    if any(row["source_url"] in sealed_ids and row["role"] not in {"final", "test_prefix", "embargo", "excluded"}
           for row in rows):
        raise ValueError("Sealed source entered a nonfinal split")
    exposure_hash = digest({field: sorted(getattr(exposure, field)) for field in
                            ("groups", "source_ids", "sgf_hashes", "sequence_hashes", "account_months")})
    code_names = ("population_audit.py", "canonical_audit.py", "canonical_data.py", "data_splits.py",
                  "kgs_admission.py", "kgs_eligibility.py", "sgf_audit.py", "rules_replay.py", "go.py", "adjudication.py")
    provenance = {"population_inputs_sha256": expected_sha256, "source_policy": POLICY,
                  "input_references": {key: inputs[key] for key in paths},
                  "source_references_sha256": digest([item["source"] for item in items]),
                  "uv_lock_sha256": file_hash("uv.lock"),
                  "code_sha256": {name: file_hash(Path(__file__).with_name(name)) for name in code_names}}
    manifest = manifest_contents(entries, rows, plan, exposure_sha256=exposure_hash, provenance=provenance)
    manifest["sealed_source_ids"] = sealed_ids
    output.mkdir(parents=True, exist_ok=False)
    pq.write_table(pa.Table.from_pylist([{"source_url": record["source"]["source_url"],
                                         "canonical_sha256": digest(record), "canonical_json": encoded(record)}
                                        for record in records]), output / "canonical.parquet", compression="zstd")
    split_hash = write_membership(output / "splits", rows, manifest)
    assigned = {row["source_url"]: row for row in rows}
    index = []
    for record in records:
        membership = assigned[record["source"]["source_url"]]
        sealed = record["sealed_final"]
        index.append({"source_url": record["source"]["source_url"], "canonical_sha256": digest(record),
                      "sealed_final": sealed, "role": membership["role"], "groups": membership["groups"],
                      "metadata_json": None if sealed else encoded(record["metadata"]),
                      "source_admitted": record["eligibility"]["training_eligible"],
                      "replay_validated": record["replay_validated"], "split_exclusions": membership["exclusions"],
                      "final_excluded": membership["final_excluded"]})
    pq.write_table(pa.Table.from_pylist(index), output / "index.parquet", compression="zstd")
    (output / "adjudications.json").write_text(encoded(adjudications) + "\n")
    check_budget()
    summary = {"version": "population_audit_v1", "provenance": provenance, "records": len(records),
               "original_records": len(audits), "source_admitted_original_games": sum(g["training_eligible"] for g in gates.values()),
               "sealed_final_candidates": len(sealed_ids), "sealed_final_usable_games": sum(row["role"] == "final" for row in rows),
               "sealed_source_ids_sha256": digest(sealed_ids), "split_roles": dict(Counter(row["role"] for row in rows)),
               "canonical_parquet_sha256": file_hash(output / "canonical.parquet"), "split_manifest_sha256": split_hash,
               "index_parquet_sha256": file_hash(output / "index.parquet"),
               "adjudications_sha256": file_hash(output / "adjudications.json"),
               "bytes": {"selected_sgf": selected_bytes, "expanded_zip": expanded_bytes, "canonical_json": canonical_bytes},
               "costs": {"started_utc": started_utc, "cpu_seconds": time.process_time() - cpu,
                         "wall_seconds": time.monotonic() - started, "http_requests": 0, "gpu_hours": 0},
               "limits": limits, "training_ready": False, "e1_ready": False,
               "scope": "Mechanical construction; final targets remain sealed and no final diagnostics were selected."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("records", "source_admitted_original_games", "sealed_final_candidates",
                                                   "sealed_final_usable_games", "split_roles", "costs")}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--inputs-sha256", required=True)
    args = parser.parse_args()
    run(args.inputs, args.inputs_sha256, args.output)


if __name__ == "__main__":
    main()
