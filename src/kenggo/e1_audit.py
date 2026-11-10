"""Reproduce the E1 source audit offline from a frozen sample and KGS ZIPs."""

import argparse
from collections import Counter, defaultdict
from datetime import date
import hashlib
import json
from pathlib import Path, PurePosixPath
import time
from urllib.parse import urlsplit
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq

from kenggo.kgs_source import ArchivePage, candidate_reasons, select_candidates
from kenggo.sgf_audit import inspect_sgf


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def member_key(name: str):
    parts = PurePosixPath(name).parts
    if name.startswith("/") or ".." in parts or len(parts) < 4 or not name.endswith(".sgf"):
        return None
    try:
        day = date(*(int(value) for value in parts[-4:-1])).isoformat()
    except (ValueError, OverflowError):
        return None
    return day, parts[-1]


def source_key(url: str):
    return member_key(urlsplit(url).path.lstrip("/"))


def group_metrics(records):
    results = Counter()
    counters = Counter()
    by_rank = defaultdict(Counter)
    rank_handles = defaultdict(set)
    handles = Counter()
    pairs = Counter()
    clocks = Counter()
    rules = Counter()
    root_properties = Counter()
    node_properties = Counter()
    for record in records:
        result = record["result_kind"]
        results[result] += 1
        counters.update(record["counters"])
        metadata = record["metadata"]
        root_properties.update(record["root_property_names"])
        node_properties.update(record["node_property_counts"])
        pairs[f"{metadata['BR']} / {metadata['WR']}"] += 1
        clocks[json.dumps([metadata["TM"], metadata["OT"]])] += 1
        rules[json.dumps([metadata["RU"], metadata["KM"]])] += 1
        for color, name, rank in (("b", "PB", "BR"), ("w", "PW", "WR")):
            label = metadata[rank] or "unknown"
            by_rank[label]["player_game_appearances"] += 1
            by_rank[label]["result_" + result] += 1
            by_rank[label].update(record["actions_by_color"][color])
            by_rank[label]["games_with_final_two_passes"] += record["final_two_explicit_passes"]
            if metadata[name]:
                handle = metadata[name].lower()
                handles[handle] += 1
                rank_handles[label].add(handle)
    for rank, values in by_rank.items():
        values["distinct_handle_strings"] = len(rank_handles[rank])
    return dict(games=len(records), result_counts=dict(results), action_clock_counts=dict(counters),
                games_with_final_two_passes=sum(r["final_two_explicit_passes"] for r in records),
                by_player_rank={rank: dict(values) for rank, values in by_rank.items()},
                black_white_rank_pairs=dict(pairs), time_controls=dict(clocks), rules_komi=dict(rules),
                root_property_counts=dict(root_properties), mainline_property_counts=dict(node_properties),
                distinct_handle_strings=len(handles), games_per_handle_histogram=dict(Counter(handles.values())))


def audit(directory: Path, config_path: Path, output: Path):
    started = time.monotonic()
    cpu_start = time.process_time()
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads(config_path.read_text())
    sample = json.loads((directory / "sample.json").read_text())
    frame = json.loads((directory / "frame.json").read_text())
    candidates = sample["candidates"]
    if len(candidates) != 1000 or len({r["url"] for r in candidates}) != 1000:
        raise ValueError("Expected 1000 distinct frozen candidates")
    if sample["config_sha256"] != sha256(config_path) or select_candidates(frame["rows"]) != candidates:
        raise ValueError("Frozen sample does not match its discovery configuration and frame")
    unlinked = []
    page_provenance = []
    for path in sorted(directory.glob("*-2025-*.html")):
        metadata = json.loads(path.with_suffix(".json").read_text())
        page = ArchivePage(path.read_text())
        if page.rows != metadata["rows"] or sha256(path) != metadata["public_html_sha256"]:
            raise ValueError("Cached archive metadata changed")
        unlinked.extend(page.unlinked_rows)
        page_provenance.append(dict(page=path.stem, public_html_sha256=sha256(path),
                                    raw_response_sha256=metadata["body_sha256"],
                                    rows=len(page.rows), unlinked_rows=len(page.unlinked_rows)))
    archives = {}
    archive_errors = {}
    bundle_provenance = []
    expanded_total = 0
    try:
        for key in sorted({row["page_key"] for row in candidates}):
            path = directory / (key + ".zip")
            if not path.exists():
                continue
            try:
                archive = zipfile.ZipFile(path)
            except zipfile.BadZipFile:
                archive_errors[key] = "invalid_zip"
                bundle_provenance.append(dict(page=key, sha256=sha256(path), bytes=path.stat().st_size,
                                              error="invalid_zip"))
                continue
            index = defaultdict(list)
            for info in archive.infolist():
                expanded_total += info.file_size
                mapped = member_key(info.filename)
                if mapped:
                    index[mapped].append(info)
            if expanded_total > 256 * 1024**2:
                archive.close()
                raise ValueError("Expanded ZIP budget exceeded")
            archives[key] = (archive, index)
            bundle_provenance.append(dict(page=key, sha256=sha256(path), bytes=path.stat().st_size,
                                          members=len(archive.infolist()), sgf_members=sum(len(v) for v in index.values())))
        records = []
        raw_hashes = set()
        sequence_hashes = Counter()
        selected_bytes = 0
        for row in candidates:
            if time.monotonic() - started > 300 or time.process_time() - cpu_start > 120:
                raise RuntimeError("Offline audit time budget exceeded")
            record = dict(source_url=row["url"], page_key=row["page_key"],
                          retrieval_status="not_retrieved", parse_status="not_parsed",
                          exclusions=["source_record_unavailable"], training_ready=False)
            if row["page_key"] in archive_errors:
                record.update(retrieval_status="retrieval_failed", error=archive_errors[row["page_key"]])
            if row["page_key"] in archives:
                archive, index = archives[row["page_key"]]
                matches = index.get(source_key(row["url"]), [])
                if len(matches) != 1:
                    record["error"] = f"Expected one exact date/filename match; found {len(matches)}"
                elif matches[0].file_size > 1024**2:
                    record["error"] = "Selected SGF exceeds 1 MiB"
                else:
                    try:
                        with archive.open(matches[0]) as stream:
                            data = stream.read(1024**2 + 1)
                    except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
                        record.update(retrieval_status="retrieval_failed", error=type(exc).__name__)
                        records.append(record)
                        continue
                    selected_bytes += len(data)
                    if len(data) > 1024**2 or selected_bytes > 64 * 1024**2:
                        raise RuntimeError("Selected SGF byte budget exceeded")
                    record.update(inspect_sgf(data, row, config.get("excluded_discovery_accounts", {})))
                    record.update(retrieval_status="retrieved", zip_member=matches[0].filename)
                    if record["sgf_sha256"] in raw_hashes:
                        record["exclusions"].append("duplicate_sgf_bytes")
                    raw_hashes.add(record["sgf_sha256"])
                    if record["parse_status"] == "parsed":
                        sequence_hashes[record["sequence_fingerprint"]] += 1
            records.append(record)
    finally:
        for archive, _ in archives.values():
            archive.close()
    parsed = [r for r in records if r["parse_status"] == "parsed"]
    compatible = [r for r in parsed if not r["exclusions"]]
    summary = dict(
        schema_version=1, sample_sha256=sha256(directory / "sample.json"),
        frame_sha256=sha256(directory / "frame.json"), config_sha256=sha256(config_path),
        discovery=dict(pages=len(page_provenance), linked_rows=len(frame["rows"]),
                       distinct_linked_source_ids=len({r["url"] for r in frame["rows"]}),
                       unlinked_rows=len(unlinked), unlinked_types=dict(Counter(r["type"] for r in unlinked)),
                       scope_candidate_ids=len({r["url"] for r in frame["rows"] if not candidate_reasons(r)}),
                       row_exclusion_counts=frame["row_exclusion_counts"]),
        sampled_candidates=len(candidates), retrieved_selected=sum(r["retrieval_status"] == "retrieved" for r in records),
        parse_status_counts=dict(Counter(r["parse_status"] for r in records)),
        exclusion_counts=dict(Counter(reason for r in records for reason in r["exclusions"])),
        historical_human_status_counts=dict(Counter(r.get("human_status", "not_parsed") for r in records)),
        metadata_scope_compatible=len(compatible), fully_verified_training_games=0,
        validated_elapsed_labels=0, rules_replay_validated_games=0,
        duplicate_sequence_candidate_excess=sum(n - 1 for n in sequence_hashes.values() if n > 1),
        all_parsed=group_metrics(parsed), metadata_compatible=group_metrics(compatible),
        bundles=bundle_provenance, archive_pages=page_provenance,
        selected_sgf_bytes=selected_bytes, expanded_bundle_bytes=expanded_total,
        quarantine=dict(policy="Reserve no final evaluation games from these discovery account-months or the cu2 September2025 probe; all selected IDs remain excluded even if retrieval fails.",
                        sample_sha256=sha256(directory / "sample.json"), config_sha256=sha256(config_path)),
    )
    (output / "games.json").write_text(json.dumps(records, indent=2) + "\n")
    table = pa.Table.from_pylist([
        dict(source_url=r["source_url"], page_key=r["page_key"], zip_member=r.get("zip_member"),
             sgf_sha256=r.get("sgf_sha256"), sgf_bytes=r.get("sgf_bytes"),
             retrieval_status=r["retrieval_status"], parse_status=r["parse_status"], exclusions=r["exclusions"],
             audit_json=json.dumps(r, sort_keys=True)) for r in records])
    pq.write_table(table, output / "games.parquet", compression="zstd")
    summary["games_parquet_sha256"] = sha256(output / "games.parquet")
    summary["audit_costs"] = dict(wall_seconds=time.monotonic() - started,
                                 cpu_seconds=time.process_time() - cpu_start, gpu_hours=0,
                                 scope="Inspection and per-game artifact writing; excludes final summary write and interpreter startup")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("sampled_candidates", "retrieved_selected", "parse_status_counts",
                                                    "metadata_scope_compatible", "fully_verified_training_games", "audit_costs")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit(args.directory, args.config, args.output)


if __name__ == "__main__":
    main()
