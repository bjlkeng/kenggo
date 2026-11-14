"""Freeze and inspect a bounded, offline extension of the existing KGS cache."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
import time
import zipfile
import zlib

from sgfmill import sgf

from kenggo import e1_audit, kgs_admission, kgs_eligibility, kgs_source, sgf_audit

ROOT = Path.cwd().resolve()
MIB = 1024**2
VERSION = "e2_cached_coverage_audit_v1"
REQUEST_VERSION = "e2_cached_coverage_request_v1"
ROLES = ("train", "population_development", "validation_development")
EXPECTED_ROLES = {"train": 3031, "population_development": 986, "validation_development": 252}
EXPECTED_IDS = "d7ef1700e26310b5ce51be884156190eeac814ac45364a8de62f7ed476ded636"
PRODUCTION_INPUTS = {
    "frame": "e69698029b799a3779bbe460d16daad8ed6c90ffbceac9d80fe3ccb0122cead6",
    "original_sample": kgs_eligibility.SAMPLE_SHA256,
    "final_quarantine": "168010f08ad33256a6dcd9e8e268861fb41d9b02adc742dcd88697edf677a6ef",
    "source_policy": kgs_admission.DECISION_SHA256,
    "owner_exclusions": "44fc41edd839f53745bf28a8c315ef94b2e78e5ee3f049bcda491b9903b9b5dd",
    "environment": "967ee006a36704a3fc1f7df278127b46016be5e7e707b5efa2a01cef213e28f1",
    "split_config": "7138209f96063af222976b48cfd6ae1176c00eb75c93dd0186c9513965162917",
}


@dataclass(frozen=True)
class Limits:
    cpu_seconds: float = 300
    wall_seconds: float = 600
    selected_sgf_bytes: int = 64 * MIB
    member_bytes: int = MIB
    expanded_zip_bytes: int = 256 * MIB
    output_bytes: int = 128 * MIB

    def __post_init__(self):
        ceilings = {"cpu_seconds": 300, "wall_seconds": 600, "selected_sgf_bytes": 64 * MIB,
                    "member_bytes": MIB, "expanded_zip_bytes": 256 * MIB, "output_bytes": 128 * MIB}
        for key, limit in ceilings.items():
            value = getattr(self, key)
            if (type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= limit
                    or key.endswith("bytes") and type(value) is not int):
                raise ValueError("Invalid or enlarged offline audit limit: " + key)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return sha256(encoded(value)).hexdigest()


def file_hash(path):
    result = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(MIB), b""):
            result.update(chunk)
    return result.hexdigest()


def ref(path):
    path = Path(path).resolve()
    return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "sha256": file_hash(path)}


def read_ref(reference, *, maximum=64 * MIB, json_data=True):
    if (not isinstance(reference, dict) or set(reference) != {"path", "sha256"}
            or not isinstance(reference["path"], str)
            or not re.fullmatch(r"[a-f0-9]{64}", reference.get("sha256", ""))):
        raise ValueError("An input requires an explicit path and SHA-256")
    path = (ROOT / reference["path"]).resolve()
    if not path.is_file() or path.stat().st_size > maximum or file_hash(path) != reference["sha256"]:
        raise ValueError("Pinned metadata changed or exceeds its byte limit: " + str(path))
    return json.loads(path.read_bytes()) if json_data else path


def save_json(path, value, *, replace=False):
    path = Path(path)
    data = encoded(value) + b"\n"
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".writing-")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return sha256(data).hexdigest()


def source_identity():
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], timeout=10)
    if git("ls-files", "--others", "--exclude-standard").strip():
        raise ValueError("Stage intended new source files before freezing the audit")
    return {"commit": git("rev-parse", "HEAD").decode().strip(),
            "patch_sha256": sha256(git("diff", "--binary", "HEAD")).hexdigest()}


def code_refs():
    return {name: ref(path) for name, path in {
        "worker": Path(__file__), "e1_audit": e1_audit.__file__, "kgs_source": kgs_source.__file__,
        "sgf_audit": sgf_audit.__file__, "kgs_eligibility": kgs_eligibility.__file__,
        "kgs_admission": kgs_admission.__file__}.items()}


def owned(cache, name):
    path = (cache / name).resolve()
    if not path.is_relative_to(cache) or path.parent != cache:
        raise ValueError("Cache artifact leaves its owning directory")
    return path


def participants(row, groups):
    result = []
    for color in ("black", "white"):
        parsed = kgs_source.player(row[color])
        if parsed is None:
            raise ValueError("Unrecognized archive participant")
        handle = kgs_eligibility.normalize_handle(parsed[0])
        result.append((handle, groups.get(handle, "kgs-handle:" + handle)))
    return result


def provisional_role(row, groups, split):
    day = e1_audit.source_key(row["url"])[0]
    keys = {key for _, key in participants(row, groups)}
    if keys & set(split["test_groups"]):
        return None
    if day >= split["confirmation_from"]:
        return None
    if keys & set(split["validation_groups"]):
        return "validation_development" if day >= split["train_before"] else None
    return "train" if day < split["train_before"] else "population_development"


def select_rows(frame, original_ids, groups, split, cache):
    unique = {}
    for row in frame["rows"]:
        if row["url"] in unique:
            if any(row[key] != unique[row["url"]][key] for key in ("black", "white", "type", "setup", "result")):
                raise ValueError("Conflicting archive rows for one source ID")
        else:
            unique[row["url"]] = row
    bots = {groups.get(handle, "kgs-handle:" + handle) for handle in kgs_eligibility.HISTORICAL_BOT_HANDLES}
    selected = []
    for row in unique.values():
        if row["url"] in original_ids or kgs_source.candidate_reasons(row):
            continue
        sides = participants(row, groups)
        if any("bot" in handle or key in bots for handle, key in sides):
            continue
        role = provisional_role(row, groups, split)
        if role is not None and owned(cache, row["page_key"] + ".zip").is_file():
            selected.append({"archive_row": row, "provisional_role": role})
    return sorted(selected, key=lambda item: item["archive_row"]["url"])


def _successful_requests(references):
    successful = set()
    if not isinstance(references, list) or not references or len(references) > 8:
        raise ValueError("Provide the finite set of original request ledgers")
    for reference in references:
        path = read_ref(reference, maximum=4 * MIB, json_data=False)
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("method") == "GET" and row.get("status") == 200 and isinstance(row.get("sha256"), str):
                successful.add((row.get("url"), row["sha256"], row.get("bytes")))
    return successful


def _freeze_contents(request_path, request_pin, *, fixture_only=False):
    request = read_ref({"path": str(request_path), "sha256": request_pin}, maximum=8 * MIB)
    required = {"schema", "inputs", "request_ledgers", "cache_directory", "base_groups",
                "extension_scope_decision", "limits"}
    if set(request) != required or request["schema"] != REQUEST_VERSION:
        raise ValueError("Unsupported cached-coverage request")
    if not isinstance(request["extension_scope_decision"], str) or not request["extension_scope_decision"].strip():
        raise ValueError("Record the bounded extension scope before freezing candidates")
    limits = Limits(**request["limits"])
    if set(request["limits"]) != set(asdict(limits)):
        raise ValueError("Pin every offline audit limit explicitly")
    refs = request["inputs"]
    if set(refs) != set(PRODUCTION_INPUTS):
        raise ValueError("The frozen audit requires every source, quarantine and policy input")
    if not fixture_only and any(refs[key]["sha256"] != pin for key, pin in PRODUCTION_INPUTS.items()):
        raise ValueError("Production input differs from the original E1 evidence")
    values = {key: read_ref(reference) for key, reference in refs.items() if key != "environment"}
    read_ref(refs["environment"], json_data=False)
    split, groups = values["split_config"], dict(request["base_groups"])
    if (set(split) != {"grouping_sha256", "train_before", "confirmation_from", "validation_groups", "test_groups"}
            or split["train_before"] != "2025-09-01" or split["confirmation_from"] != "2025-11-01"
            or kgs_eligibility.grouping_digest(groups) != split["grouping_sha256"]
            or set(split["validation_groups"]) & set(split["test_groups"])
            or set(values["final_quarantine"]["test_groups"]) != set(split["test_groups"])):
        raise ValueError("The extension must preserve the frozen groups, cohorts and chronology")
    if any(kgs_eligibility.normalize_handle(handle) != handle or not isinstance(group, str)
           or not group.startswith("kgs-handle:") for handle, group in groups.items()):
        raise ValueError("Invalid base handle grouping")
    if not set(split["test_groups"] + split["validation_groups"]) <= set(groups.values()):
        raise ValueError("Base grouping omits a reserved cohort")
    if values["source_policy"].get("policy") != kgs_admission.POLICY:
        raise ValueError("The source-screening rule changed")
    original = values["original_sample"]["candidates"]
    original_ids = {row["url"] for row in original}
    if len(original_ids) != len(original) or not fixture_only and len(original) != 1000:
        raise ValueError("Original source exclusions are incomplete")
    cache = (ROOT / request["cache_directory"]).resolve()
    if not cache.is_dir():
        raise ValueError("Existing cache directory is required")
    candidates = select_rows(values["frame"], original_ids, groups, split, cache)
    counts = dict(Counter(item["provisional_role"] for item in candidates))
    ids_pin = digest([item["archive_row"]["url"] for item in candidates])
    if (not candidates or fixture_only and len(candidates) > 64
            or not fixture_only and (counts != EXPECTED_ROLES or ids_pin != EXPECTED_IDS)):
        raise ValueError("Frozen candidate identities or role counts differ from the declared audit")
    for item in candidates:
        for handle, group in participants(item["archive_row"], groups):
            groups.setdefault(handle, group)
    requests = _successful_requests(request["request_ledgers"])
    pages, bundles = {}, {}
    for item in candidates:
        row = item["archive_row"]
        key = row["page_key"]
        if key in pages:
            continue
        page_path, html_path = owned(cache, key + ".json"), owned(cache, key + ".html")
        page = read_ref(ref(page_path), maximum=4 * MIB)
        if html_path.stat().st_size > 2 * MIB or file_hash(html_path) != page["public_html_sha256"]:
            raise ValueError("Cached archive HTML changed")
        parsed = kgs_source.ArchivePage(html_path.read_text())
        if (parsed.rows != page["rows"] or page["bundle_url"] != row["bundle_url"]
                or page["bundle_url"] not in parsed.links
                or not any(url == page["url"] and pin == page["body_sha256"] for url, pin, _ in requests)):
            raise ValueError("Archive page lacks matching retained response provenance")
        for chosen in (x["archive_row"] for x in candidates if x["archive_row"]["page_key"] == key):
            if (chosen["bundle_url"] != page["bundle_url"] or
                    {k: v for k, v in chosen.items() if k not in ("page_key", "bundle_url")} not in page["rows"]):
                raise ValueError("Frozen candidate is absent from its retained archive page")
        bundle_path = owned(cache, key + ".zip")
        bundle_pin, bundle_bytes = file_hash(bundle_path), bundle_path.stat().st_size
        if bundle_bytes > 8 * MIB or (row["bundle_url"], bundle_pin, bundle_bytes) not in requests:
            raise ValueError("Cached ZIP lacks matching retained response provenance")
        pages[key] = {"metadata": ref(page_path), "html": ref(html_path),
                      "raw_response_sha256": page["body_sha256"], "url": page["url"]}
        bundles[key] = {"artifact": ref(bundle_path), "bytes": bundle_bytes, "url": row["bundle_url"]}
    excluded = values["owner_exclusions"]["excluded_games"]
    if any(item["archive_row"]["url"] in {row["source_url"] for row in excluded} for item in candidates):
        raise ValueError("An owner-excluded source entered the candidate list")
    return {"schema": VERSION, "fixture_only": fixture_only, "request": {"path": str(request_path), "sha256": request_pin},
        "inputs": refs, "request_ledgers": request["request_ledgers"], "cache_directory": str(cache),
        "extension_scope_decision": request["extension_scope_decision"], "source": source_identity(),
        "code": code_refs(), "limits": asdict(limits), "network_requests": 0, "gpu_seconds": 0, "workers": 1,
        "candidates": candidates, "candidate_ids_sha256": ids_pin, "candidate_counts": counts,
        "pages": pages, "bundles": bundles, "groups": groups,
        "extended_grouping_sha256": kgs_eligibility.grouping_digest(groups),
        "owner_excluded_sgf_sha256": sorted({row["sgf_sha256"] for row in excluded}),
        "screening_only": True, "deduplicated": False, "legal_replay_validated": False}


def freeze_inputs(request_path, request_pin, output_path, *, fixture_only=False):
    """Read cached metadata and hash ZIP bytes; do not open ZIP members or SGFs."""
    if type(fixture_only) is not bool:
        raise ValueError("Fixture scope must be explicit")
    frozen = _freeze_contents(request_path, request_pin, fixture_only=fixture_only)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pin = save_json(output_path, frozen)
    return {"path": str(output_path), "sha256": pin, "candidate_ids_sha256": frozen["candidate_ids_sha256"],
            "candidate_counts": frozen["candidate_counts"]}


def header_check(data, row, groups, reserved):
    """Read the root node only before admitting any action or result diagnostics."""
    start = re.match(br"\xef\xbb\xbf\s*\(\s*;|\s*\(\s*;", data)
    if start is None:
        raise ValueError("invalid_sgf_root")
    inside = escaped = False
    end = None
    for offset in range(start.end(), len(data)):
        char = data[offset]
        if inside:
            if escaped:
                escaped = False
            elif char == ord("\\"):
                escaped = True
            elif char == ord("]"):
                inside = False
        elif char == ord("["):
            inside = True
        elif char in b";()":
            end = offset
            break
    if end is None:
        raise ValueError("invalid_sgf_root")
    game = sgf.Sgf_game.from_bytes(data[:end] + b")")
    root = game.get_root()
    for color, prop, rank in (("black", "PB", "BR"), ("white", "PW", "WR")):
        declared = kgs_source.player(row[color])
        handle = sgf_audit.scalar(root, prop)
        if handle != declared[0] or sgf_audit.scalar(root, rank) != declared[1]:
            raise ValueError("archive_sgf_identity_or_rank_mismatch")
        normalized = kgs_eligibility.normalize_handle(handle)
        if groups.get(normalized, "kgs-handle:" + normalized) in reserved:
            raise ValueError("reserved_final_participant")
    if sgf_audit.scalar(root, "DT") != e1_audit.source_key(row["url"])[0]:
        raise ValueError("archive_sgf_date_mismatch")


class AuditStop(RuntimeError):
    def __init__(self, reason, record=None):
        super().__init__(reason)
        self.record = record


class Journal:
    def __init__(self, output, frozen, pin):
        self.output, self.frozen, self.pin = output, frozen, pin
        self.started, self.cpu = time.monotonic(), time.process_time()
        self.limits = Limits(**frozen["limits"])
        self.reserved = frozenset(read_ref(frozen["inputs"]["split_config"])["test_groups"])
        keys = {key for item in frozen["candidates"]
                for _, key in participants(item["archive_row"], frozen["groups"])}
        self.exposure = kgs_eligibility.Exposure(frozen["extended_grouping_sha256"], frozenset(keys),
            frozenset(item["archive_row"]["url"] for item in frozen["candidates"]),
            frozenset(), frozenset(), frozenset())
        self.state = {"schema": VERSION, "status": "running", "inputs_sha256": pin,
            "expected_records": len(frozen["candidates"]), "attempted_records": 0, "completed_records": 0,
            "current_source": None, "selected_sgf_bytes_charged": 0, "selected_sgf_bytes_returned": 0,
            "expanded_zip_bytes": 0, "network_requests": 0, "gpu_seconds": 0, "workers": 1,
            "byte_scope": "Selected bytes conservatively charge the declared member size before opening; failed reads retain their full charge."}
        self.records = []
        self.write()

    def write(self):
        self.state["costs"] = {"cpu_seconds": time.process_time() - self.cpu,
                               "wall_seconds": time.monotonic() - self.started}
        save_json(self.output / "progress.json", self.state, replace=True)

    def check(self):
        if (time.process_time() - self.cpu >= self.limits.cpu_seconds * .9
                or time.monotonic() - self.started >= self.limits.wall_seconds * .9):
            raise AuditStop("worker_time_reserve")
        if sum(p.stat().st_size for p in self.output.iterdir() if p.is_file()) > self.limits.output_bytes * .8:
            raise AuditStop("output_reserve")

    def append(self, record):
        with (self.output / "records.jsonl").open("ab") as stream:
            stream.write(encoded(record) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.records.append(record)
        self.state["completed_records"] += 1
        self.state["current_source"] = None
        self.write()


def _screen(data, item, frozen, exposure, reserved):
    row = item["archive_row"]
    pin = sha256(data).hexdigest()
    if pin in frozen["owner_excluded_sgf_sha256"]:
        return {"parse_status": "not_inspected", "source_admitted": False,
                "exclusions": ["owner_excluded_sgf_bytes"], "sgf_sha256": pin, "sgf_bytes": len(data)}
    try:
        header_check(data, row, frozen["groups"], reserved)
    except (ValueError, TypeError, KeyError, UnicodeError, LookupError) as error:
        reason = str(error) if str(error) in {"invalid_sgf_root", "archive_sgf_identity_or_rank_mismatch",
            "reserved_final_participant", "archive_sgf_date_mismatch"} else "invalid_sgf_header"
        return {"parse_status": "header_rejected", "source_admitted": False, "exclusions": [reason],
                "sgf_sha256": pin, "sgf_bytes": len(data)}
    audit = sgf_audit.inspect_sgf(data, row, kgs_eligibility.HISTORICAL_BOT_HANDLES)
    audit["source_url"] = row["url"]
    if "error" in audit:
        audit["error"] = "corrupt_or_unsupported_sgf"
    decision = kgs_admission.classify(audit, row, frozen["groups"], exposure)
    audit["source_admitted"] = decision.pop("training_eligible")
    audit["screening"] = decision
    audit["extension_scope_sha256"] = digest(frozen["extension_scope_decision"])
    audit["identity_valid"] = False
    audit["duration_targets_valid"] = False
    audit["dynamic_clock_inputs_valid"] = False
    if "metadata" in audit:
        audit["metadata"] = {key: audit["metadata"][key] for key in
            ("PB", "PW", "BR", "WR", "RU", "KM", "TM", "OT", "DT", "RE", "HA", "PL", "EV", "RO")}
    return audit


def _record(item, frozen, archives, archive_errors, journal):
    row = item["archive_row"]
    record = {"source_url": row["url"], "page_key": row["page_key"], "provisional_role": item["provisional_role"],
        "input_manifest_sha256": journal.pin, "bundle_sha256": frozen["bundles"][row["page_key"]]["artifact"]["sha256"],
        "expected_member_key": e1_audit.source_key(row["url"]), "zip_member": None,
        "retrieval_status": "failed", "parse_status": "not_parsed", "source_admitted": False,
        "exclusions": ["source_record_unavailable"], "training_ready": False,
        "deduplicated": False, "replay_validated": False}
    key = row["page_key"]
    if key in archive_errors:
        return record | {"error": archive_errors[key]}
    archive, index = archives[key]
    matches = index.get(e1_audit.source_key(row["url"]), [])
    if len(matches) != 1:
        return record | {"error": "missing_or_ambiguous_exact_member", "member_matches": len(matches)}
    info = matches[0]
    record["zip_member"] = info.filename
    mode = info.external_attr >> 16
    if info.is_dir() or stat.S_ISLNK(mode):
        return record | {"error": "selected_member_is_not_regular"}
    if info.file_size > journal.limits.member_bytes:
        return record | {"error": "member_byte_limit"}
    if journal.state["selected_sgf_bytes_charged"] + info.file_size > journal.limits.selected_sgf_bytes:
        raise AuditStop("selected_sgf_byte_limit", record | {"error": "selected_sgf_byte_limit"})
    journal.state["selected_sgf_bytes_charged"] += info.file_size
    journal.write()
    try:
        with archive.open(info) as stream:
            data = stream.read(info.file_size)
        journal.state["selected_sgf_bytes_returned"] += len(data)
        if len(data) != info.file_size:
            return record | {"error": "selected_member_size_mismatch"}
    except (zipfile.BadZipFile, RuntimeError, OSError, NotImplementedError, EOFError, zlib.error):
        return record | {"error": "selected_member_read_failure"}
    record.update(_screen(data, item, frozen, journal.exposure, journal.reserved))
    record["retrieval_status"] = "retrieved"
    return record


def _summary(journal, outcome, reason):
    records = journal.records
    parsed = [record for record in records if record["parse_status"] == "parsed"]
    admitted = [record for record in parsed if record["source_admitted"]]
    by_role = {}
    for role in ROLES:
        selected = [record for record in admitted if record["provisional_role"] == role]
        by_role[role] = e1_audit.group_metrics(selected)
    return {"schema": VERSION, "status": outcome, "reason": reason, "input_manifest_sha256": journal.pin,
        "candidate_ids_sha256": journal.frozen["candidate_ids_sha256"], "expected_records": len(journal.frozen["candidates"]),
        "attempted_records": journal.state["attempted_records"], "recorded_records": len(records),
        "attempted_unrecorded_records": journal.state["attempted_records"] - len(records),
        "current_source": journal.state["current_source"],
        "unattempted_records": len(journal.frozen["candidates"]) - journal.state["attempted_records"],
        "retrieved_records": sum(record["retrieval_status"] == "retrieved" for record in records),
        "source_admitted_records": len(admitted), "parse_status_counts": dict(Counter(record["parse_status"] for record in records)),
        "exclusion_counts": dict(Counter(reason for record in records for reason in record["exclusions"])),
        "retrieval_error_counts": dict(Counter(record["error"] for record in records if record.get("error"))),
        "all_parsed": e1_audit.group_metrics(parsed), "source_admitted": e1_audit.group_metrics(admitted),
        "source_admitted_by_provisional_role": by_role, "network_requests": 0, "gpu_seconds": 0,
        "deduplicated": False, "rules_replay_validated_games": 0, "promoted_profiles": [],
        "limits": journal.frozen["limits"], "costs": journal.state["costs"],
        "scope": "Provisional source screening and explicit-action coverage only; no legal replay, canonical records, new examples, training or qualification.",
        "cost_scope": "Worker body through final record, including retained failures; supervisor must charge startup, validation, final writes and cleanup.",
        "bytes": {key: journal.state[key] for key in ("selected_sgf_bytes_charged", "selected_sgf_bytes_returned", "expanded_zip_bytes")}}


def run_audit(inputs_path, inputs_pin, output, *, fixture_only=False):
    """Run once in an existing supervised worker directory; never acquire or retry."""
    if type(fixture_only) is not bool:
        raise ValueError("Fixture scope must be explicit")
    if Path(output).is_symlink():
        raise ValueError("Use the supervisor's real output directory")
    output = Path(output).resolve()
    if not output.is_dir() or any((output / name).exists() for name in ("progress.json", "records.jsonl", "summary.json", "exposure.json")):
        raise ValueError("Use an unused supervisor-owned output directory")
    frozen = read_ref({"path": str(inputs_path), "sha256": inputs_pin}, maximum=16 * MIB)
    if frozen.get("schema") != VERSION or frozen.get("fixture_only") is not fixture_only:
        raise ValueError("Worker scope differs from its frozen inputs")
    expected = _freeze_contents(frozen["request"]["path"], frozen["request"]["sha256"], fixture_only=fixture_only)
    if expected != frozen:
        raise ValueError("Frozen metadata, source, inputs or cache provenance changed")
    journal = Journal(output, frozen, inputs_pin)
    archives, archive_errors = {}, {}
    outcome, reason = "failed", None
    try:
        for key, bundle in frozen["bundles"].items():
            journal.check()
            path = read_ref(bundle["artifact"], maximum=8 * MIB, json_data=False)
            try:
                archive = zipfile.ZipFile(path)
            except zipfile.BadZipFile:
                archive_errors[key] = "invalid_zip"
                continue
            archives[key] = (archive, defaultdict(list))
            for info in archive.infolist():
                if journal.state["expanded_zip_bytes"] + info.file_size > journal.limits.expanded_zip_bytes:
                    raise AuditStop("expanded_zip_byte_limit")
                journal.state["expanded_zip_bytes"] += info.file_size
                mapped = e1_audit.member_key(info.filename)
                if mapped is not None:
                    archives[key][1][mapped].append(info)
            journal.write()
        for item in frozen["candidates"]:
            journal.check()
            journal.state["attempted_records"] += 1
            journal.state["current_source"] = item["archive_row"]["url"]
            journal.write()
            journal.append(_record(item, frozen, archives, archive_errors, journal))
        outcome = "completed"
    except AuditStop as error:
        if error.record is not None:
            journal.append(error.record)
        outcome, reason = "partial", str(error)
    except BaseException as error:
        reason = type(error).__name__
        raise
    finally:
        for archive, _ in archives.values():
            archive.close()
        journal.state.update(status=outcome, reason=reason)
        journal.write()
        exposure = {"input_manifest_sha256": inputs_pin, "grouping_sha256": frozen["extended_grouping_sha256"],
            "groups": sorted({key for item in frozen["candidates"] for _, key in participants(item["archive_row"], frozen["groups"])}),
            "source_ids": [item["archive_row"]["url"] for item in frozen["candidates"]],
            "sgf_sha256": sorted({record["sgf_sha256"] for record in journal.records if "sgf_sha256" in record}),
            "sequence_sha256": sorted({record["sequence_fingerprint"] for record in journal.records if "sequence_fingerprint" in record}),
            "scope": "Add these identities to the prior exposure registry; this artifact never replaces it."}
        exposure_pin = save_json(output / "exposure.json", exposure)
        summary = _summary(journal, outcome, reason)
        summary["exposure"] = {"path": "exposure.json", "sha256": exposure_pin}
        summary["records"] = {"path": "records.jsonl", "sha256": file_hash(output / "records.jsonl")} if journal.records else None
        save_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("freeze", "audit"))
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--inputs-sha256", required=True)
    args = parser.parse_args()
    result = (freeze_inputs if args.operation == "freeze" else run_audit)(args.inputs, args.inputs_sha256, args.output)
    print(json.dumps({key: value for key, value in result.items() if key in
        ("path", "sha256", "candidate_ids_sha256", "candidate_counts", "status", "reason",
         "expected_records", "recorded_records", "source_admitted_records", "costs")}))


if __name__ == "__main__":
    main()
