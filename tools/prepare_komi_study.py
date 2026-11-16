"""Seal one charged, metadata-only launch of the cached komi study."""

import time

IMPORT_CPU = time.process_time()

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import zipfile

from sgfmill import sgf

from kenggo.data_expansion import MIB, read_ref, reference, save_json
from kenggo.komi_expansion import CODE_PATHS, PRODUCTION_PINS, freeze_candidates
from kenggo.sgf_audit import scalar


FIELDS = ("RU", "KM", "BR", "WR", "TM", "OT", "PB", "PW", "DT", "SZ", "HA", "PL")
PREFLIGHT_BYTES = 512 * MIB


def root_metadata(data):
    start = re.match(br'\xef\xbb\xbf\s*\(\s*;|\s*\(\s*;', data)
    if start is None:
        raise ValueError("Missing SGF root")
    inside = escaped = False
    for end in range(start.end(), len(data)):
        char = data[end]
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
            break
    else:
        raise ValueError("Unterminated SGF root")
    prefix = data[:end] + b")"
    root = sgf.Sgf_game.from_bytes(prefix).get_root()
    return {field: scalar(root, field) for field in FIELDS}, sha256(prefix).hexdigest()


def exclusive_json(path, value):
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    if len(body.encode()) > 128 * MIB:
        raise ValueError("Preflight artifact exceeds 128 MiB")
    with Path(path).open("x") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    return reference(path)


def header_evidence(template, output, check):
    source_ref, summary_ref = template["previous_sources"], template["previous_summary"]
    sources, summary = read_ref(source_ref, maximum=128 * MIB), read_ref(summary_ref)
    audit = read_ref(template["scope_audit"])
    candidates = sources["candidates"]
    if (len(candidates) != 33817 or summary["source_completed"] != len(candidates)
            or summary["sources_sha256"] != source_ref["sha256"]
            or audit["inputs"]["sources"]["sha256"] != source_ref["sha256"]
            or audit["inputs"]["build_request"]["sha256"] != summary["request_sha256"]):
        raise ValueError("The header preflight requires the completed pinned source population")
    index = read_ref(summary["artifacts"]["base_index"])
    refs = index["provenance"]["work_units"]
    if len(refs) != len(candidates):
        raise ValueError("Prior work does not cover every source candidate")
    unit_dir = Path(summary_ref["path"]).resolve().parent / "work/canonical"
    by_archive = defaultdict(list)
    for number, item in enumerate(candidates):
        src = item["source"]
        by_archive[(src["archive_path"], src["archive_sha256"])].append((number, item))
    records = [None] * len(candidates)
    costs = {"selected_sgf_bytes": 0, "expanded_zip_bytes": 0, "archive_bytes_hashed": 0,
             "archives_verified": 0, "records": len(candidates)}
    for (archive_path, archive_pin), items in by_archive.items():
        check()
        pinned = read_ref({"path": archive_path, "sha256": archive_pin}, json_data=False, maximum=8 * MIB)
        costs["archive_bytes_hashed"] += pinned.stat().st_size
        costs["archives_verified"] += 1
        with zipfile.ZipFile(pinned) as archive:
            infos = archive.infolist()
            costs["expanded_zip_bytes"] += sum(info.file_size for info in infos)
            if costs["expanded_zip_bytes"] >= 4 * 1024**3:
                raise ValueError("Header preflight exhausted the cumulative ZIP expansion allowance")
            names = Counter(info.filename for info in infos)
            for number, item in items:
                check()
                unit_ref = refs[number]
                if Path(unit_ref["path"]).resolve() != unit_dir / f"{number:06d}.json":
                    raise ValueError("The source order differs from the old work-unit paths")
                unit = read_ref(unit_ref, maximum=256 * MIB)
                key, member = item["source"]["source_url"], item["source"]["zip_member"]
                if unit["source_url"] != key or names[member] != 1:
                    raise ValueError("Source identity or ZIP member is ambiguous")
                info = archive.getinfo(member)
                if info.file_size > MIB:
                    raise ValueError("An old selected member exceeds the frozen per-member limit")
                costs["selected_sgf_bytes"] += info.file_size
                if costs["selected_sgf_bytes"] >= 256 * MIB:
                    raise ValueError("Header preflight exhausted the cumulative SGF allowance")
                with archive.open(info) as stream:
                    data = stream.read(MIB + 1)
                pin = sha256(data).hexdigest()
                if len(data) != info.file_size or pin != unit["sgf_sha256"]:
                    raise ValueError("Cached SGF bytes differ from the old checked source")
                header, prefix_pin = root_metadata(data)
                records[number] = {"source_url": key, "source_index": number, "unit_ref": unit_ref,
                    "sgf_sha256": pin, "root_prefix_sha256": prefix_pin, "header": header}
    value = {"schema": "e2_komi_header_evidence_v1", "previous_sources": source_ref,
             "previous_summary": summary_ref, "scope_audit": template["scope_audit"],
             "records": records, "costs": costs}
    return exclusive_json(output, value), costs


def prepare(template_ref, preflight_directory, output_directory, *, cpu_start=None):
    cpu_start = time.process_time() if cpu_start is None else cpu_start
    wall_start, launch = time.monotonic(), datetime.now(timezone.utc)
    directory, output = Path(preflight_directory).resolve(), Path(output_directory).resolve()
    if directory == output or directory in output.parents or output in directory.parents:
        raise ValueError("Preflight and build outputs must be separate directories")
    if output.exists():
        raise FileExistsError("Production preparation requires a new build directory")
    directory.mkdir(parents=True, exist_ok=False)
    marker = {"schema": "e2_komi_production_session_v1", "launch_utc": launch.isoformat(),
              "preparation_deadline_utc": (launch + timedelta(hours=22)).isoformat(),
              "deadline_utc": (launch + timedelta(hours=24)).isoformat(),
              "output_directory": str(output), "workers": 1, "http_requests": 0,
              "gpu_seconds": 0, "training_performed": False, "template": template_ref,
              "preflight_cpu_reserved": 300}
    marker_ref = exclusive_json(directory / "session.json", marker)

    def check():
        if time.process_time() - cpu_start >= 290 or time.monotonic() - wall_start >= 590:
            raise RuntimeError("Metadata preflight reached its fixed budget")
        if sum(p.stat().st_size for p in directory.iterdir() if p.is_file()) > PREFLIGHT_BYTES:
            raise RuntimeError("Metadata preflight output exceeded 512 MiB")

    try:
        template = read_ref(template_ref, maximum=16 * MIB)
        if set(template) != {"schema", "request", "previous_sources", "previous_summary", "scope_audit"} \
                or template["schema"] != "e2_komi_launch_template_v1":
            raise ValueError("Unsupported komi launch template")
        if any(template[name]["sha256"] != pin for name, pin in PRODUCTION_PINS.items()):
            raise ValueError("The launch template changed the approved cached inputs")
        required_code = set(CODE_PATHS) | {"tools/prepare_komi_study.py"}
        if not required_code <= set(template["request"]["source_revision"]["code_sha256"]):
            raise ValueError("The launch template omitted a production dependency")
        for path, pin in template["request"]["source_revision"]["code_sha256"].items():
            check()
            read_ref({"path": path, "sha256": pin}, json_data=False, maximum=16 * MIB)
        headers, byte_costs = header_evidence(template, directory / "headers.json", check)
        frozen = freeze_candidates(template["previous_sources"], template["previous_summary"],
            template["scope_audit"], headers, directory / "sources.json", check_budget=check)
        sources = read_ref(frozen, maximum=128 * MIB)
        request = template["request"]
        baseline = read_ref(template["previous_summary"])
        request["base_groups"] = read_ref(baseline["artifacts"]["groups"])
        request["split_config"] = read_ref(baseline["artifacts"]["split_manifest"])["plan"]
        request.update({name: marker[name] for name in ("launch_utc", "preparation_deadline_utc", "deadline_utc")})
        request.update(initial_cpu_seconds=300, session_marker=marker_ref,
                       source_request_sha256=sources["request_sha256"])
        for name, cap in (("selected_sgf_bytes", 256 * MIB), ("expanded_zip_bytes", 4 * 1024**3)):
            request["limits"][name] = cap - byte_costs[name]
        request["limits"]["output_bytes"] = 31 * 1024**3
        check()
        report = {"schema": "e2_komi_preflight_v1", "status": "completed", "session_marker": marker_ref,
                  "sources": frozen, "header_evidence": headers, "cpu_charged_seconds": 300,
                  "cpu_seconds": time.process_time() - cpu_start,
                  "wall_seconds": time.monotonic() - wall_start, "byte_costs": byte_costs,
                  "selected_sgf_bytes": byte_costs["selected_sgf_bytes"],
                  "expanded_zip_bytes": byte_costs["expanded_zip_bytes"],
                  "timing_scope": "Through candidate freeze and request preparation; final manifest writes are covered by the conservative 300-second charge."}
        preflight_ref = exclusive_json(directory / "preflight.json", report)
        request["preflight"] = preflight_ref
        request_ref = exclusive_json(directory / "request.json", request)
        check()
        return {"request": request_ref, "sources": frozen, "session_marker": marker_ref,
                "preflight": preflight_ref, "output": str(output)}
    except BaseException as error:
        save_json(directory / "failure.json", {"status": "failed", "session_marker": marker_ref,
            "cpu_charged_seconds": 300, "cpu_seconds": time.process_time() - cpu_start,
            "wall_seconds": time.monotonic() - wall_start, "error_type": type(error).__name__, "error": str(error)})
        raise


def stop(signum, frame):
    raise RuntimeError("Metadata preflight reached its CPU or elapsed-time limit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True)
    parser.add_argument("--template-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("preflight_directory", type=Path)
    args = parser.parse_args()
    signal.signal(signal.SIGALRM, stop)
    signal.signal(signal.SIGXCPU, stop)
    signal.alarm(595)
    _, hard = resource.getrlimit(resource.RLIMIT_CPU)
    resource.setrlimit(resource.RLIMIT_CPU, (math.ceil(IMPORT_CPU + 295), hard))
    result = prepare({"path": args.template, "sha256": args.template_sha256},
                     args.preflight_directory, args.output, cpu_start=IMPORT_CPU)
    print(json.dumps(result, indent=2))
