"""Freeze and build the bounded cached Japanese 0.5-komi addition."""

import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import time

from kenggo import data_expansion as expansion
from kenggo.canonical_data import digest, encoded
from kenggo.cached_coverage_audit import provisional_role
from kenggo.kgs_eligibility import grouping_digest, normalize_handle
from kenggo.kgs_source import candidate_reasons, player
from kenggo.sgf_audit import finite_number, japanese_overtime


REQUEST_VERSION = "e2_komi_expansion_request_v1"
SOURCES_VERSION = "e2_komi_reconsideration_sources_v1"
VERSION = "e2_komi_expansion_v1"
OLD_REASON = "metadata:rules_or_komi_out_of_scope"
EXPECTED_COUNTS = {"train": 9525, "population_development": 1502, "validation_development": 2020}
FOCUS_RANKS = {"15k", "14k", "13k", "11k"}
CEILINGS = {**expansion.CEILINGS, "cpu_seconds": 72000, "preparation_cpu_seconds": 68400,
            "wall_seconds": 86400, "preparation_wall_seconds": 79200,
            "canonical_file_bytes": 128 * expansion.MIB, "output_bytes": 32 * 1024**3}
PRODUCTION_PINS = {
    "previous_summary": "f3209a278e9ab3ff7074913ca7f4e30377f60148f8b3ebd56c54fd5e53f2b136",
    "previous_sources": "0fe528cd35e00a402aa1a8617243ea58df618bcd7a3c3e9a9b8cd763ac54c8bc",
    "scope_audit": "126de24cbd290a81b1b39040d047d595200998c3d93de21e12c624a74e990844"}
CODE_PATHS = expansion.CODE_PATHS + ("src/kenggo/komi_expansion.py",)


def _same_ref(left, right):
    return left["sha256"] == right["sha256"] and Path(left["path"]).resolve() == Path(right["path"]).resolve()


def _utc(value):
    instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if instant.tzinfo is None or instant.utcoffset().total_seconds() != 0:
        raise ValueError("Production timestamps require UTC")
    return instant


def _valid_header(header, item):
    overtime = japanese_overtime(header.get("OT"))
    main = finite_number(header.get("TM"))
    if (header.get("RU") != "Japanese" or type(header.get("KM")) not in (int, float)
            or header["KM"] != 0.5 or header.get("SZ") != 19
            or header.get("HA") not in (None, 0) or header.get("PL") not in (None, "b")
            or main is None or not 600 <= main <= 1800 or overtime is None
            or not 3 <= overtime[0] <= 5 or not 20 <= overtime[1] <= 60):
        raise ValueError("A scope-only rejection has incompatible pinned root metadata")
    row = item["archive_row"]
    for color, name, rank in (("black", "PB", "BR"), ("white", "PW", "WR")):
        declared = player(row[color])
        if (declared is None or not isinstance(header.get(name), str)
                or normalize_handle(header[name]) != normalize_handle(declared[0])
                or header.get(rank) != declared[1]):
            raise ValueError("Pinned header identity or rank differs from its source row")
    if header.get("DT") != expansion.source_key(row["url"])[0]:
        raise ValueError("Pinned header date differs from its source row")


def _ordered(candidates):
    by_role = {role: [] for role in expansion.ALLOWED}
    for item in candidates:
        by_role[item["provisional_role"]].append(item)
    train = by_role["train"]
    focus = lambda item: any(player(item["archive_row"][color])[1] in FOCUS_RANKS
                             for color in ("black", "white"))
    by_role["train"] = [item for item in train if focus(item)] + [item for item in train if not focus(item)]
    return [by_role[role][index] for index in range(max(map(len, by_role.values()), default=0))
            for role in expansion.ALLOWED if index < len(by_role[role])]


def _select(previous_sources_ref, previous_summary_ref, scope_audit_ref, header_evidence_ref,
            check_budget=lambda: None):
    refs = {"previous_sources": previous_sources_ref, "previous_summary": previous_summary_ref,
            "scope_audit": scope_audit_ref}
    for name, ref in refs.items():
        if ref["sha256"] != PRODUCTION_PINS[name]:
            raise ValueError("The komi study requires its fixed production input: " + name)
    previous = expansion.read_ref(previous_sources_ref, maximum=128 * expansion.MIB)
    summary = expansion.read_ref(previous_summary_ref)
    scope = expansion.read_ref(scope_audit_ref)
    evidence = expansion.read_ref(header_evidence_ref, maximum=128 * expansion.MIB)
    if (summary["sources_sha256"] != previous_sources_ref["sha256"]
            or summary["source_completed"] != len(previous["candidates"])
            or scope["inputs"]["sources"]["sha256"] != previous_sources_ref["sha256"]
            or evidence.get("schema") != "e2_komi_header_evidence_v1"
            or any(not _same_ref(evidence[name], ref) for name, ref in refs.items())
            or len(evidence["records"]) != len(previous["candidates"])):
        raise ValueError("Header evidence does not cover the pinned completed source population")
    index = expansion.read_ref(summary["artifacts"]["base_index"])
    split = expansion.read_ref(summary["artifacts"]["split_manifest"])
    groups = expansion.read_ref(summary["artifacts"]["groups"])
    if (index["schema"] != expansion.INDEX_VERSION
            or index["provenance"]["sources_sha256"] != previous_sources_ref["sha256"]
            or len(index["provenance"]["work_units"]) != len(previous["candidates"])
            or groups != previous["groups"]):
        raise ValueError("The prior source collection and base index lineage disagree")
    old_rejected = {row["source_url"]: row for row in index["rejected"]}
    old_accepted = {row["source_url"] for row in index["records"]}
    selected = []
    for number, (item, header_row, pinned) in enumerate(zip(previous["candidates"], evidence["records"],
                                                           index["provenance"]["work_units"], strict=True)):
        check_budget()
        key = item["source"]["source_url"]
        if (header_row["source_url"] != key or header_row["source_index"] != number
                or not _same_ref(header_row["unit_ref"], pinned)):
            raise ValueError("Header evidence changed the original source order or work-unit pin")
        unit = expansion.read_ref(pinned, maximum=256 * expansion.MIB)
        if unit["source_url"] != key or unit.get("sgf_sha256") != header_row["sgf_sha256"]:
            raise ValueError("The pinned header member differs from the prior source decision")
        if (unit["status"] != "source_excluded" or unit.get("reason") != "source_policy"
                or unit.get("exclusions") != [OLD_REASON]
                or header_row["header"].get("RU") != "Japanese"
                or header_row["header"].get("KM") != 0.5):
            continue
        _valid_header(header_row["header"], item)
        rejected = old_rejected.get(key)
        expected = {"source_url": key, "reason": "source_policy", "sgf_sha256": unit["sgf_sha256"],
                    "audit_sequence_sha256": unit.get("sequence_fingerprint"),
                    "trajectory_sha256": unit.get("trajectory_sha256")}
        if (rejected != expected or key in old_accepted or candidate_reasons(item["archive_row"])
                or item["provisional_role"] not in expansion.ALLOWED
                or provisional_role(item["archive_row"], groups, split["plan"]) != item["provisional_role"]
                or unit["sgf_sha256"] in previous["owner_excluded_sgf_sha256"]):
            raise ValueError("A reconsideration conflicts with prior rejection, owner, or cohort evidence")
        selected.append({**deepcopy(item), "reconsideration": {
            **expected, "source_index": number, "prior_source_sha256": digest(item), "unit_ref": pinned,
            "prior_decision": {field: unit[field] for field in ("status", "reason", "exclusions")},
            "header_evidence": deepcopy(header_row), "new_komi_halfpoints": 1}})
    counts = dict(Counter(item["provisional_role"] for item in selected))
    if counts != EXPECTED_COUNTS:
        raise ValueError("The frozen 0.5-komi candidate counts did not reproduce: " + encoded(counts))
    candidates = _ordered(selected)
    provenance = {**refs, "header_evidence": header_evidence_ref,
                  "parent_index": summary["artifacts"]["base_index"]}
    request_pin = digest({"schema": SOURCES_VERSION, "provenance": provenance,
                          "candidate_order_sha256": digest([item["source"]["source_url"] for item in candidates])})
    return {"schema": SOURCES_VERSION, "request_sha256": request_pin,
            "base_grouping_sha256": grouping_digest(groups), "groups": groups,
            "candidates": candidates, "counts": counts, "provenance": provenance,
            "owner_excluded_sgf_sha256": previous["owner_excluded_sgf_sha256"],
            "exposure": previous["exposure"], "unavailable_records": 1360,
            "selection": "Pinned scope-only Japanese 0.5 rejections; focus ranks first within training; role round robin."}


def freeze_candidates(previous_sources_ref, previous_summary_ref, scope_audit_ref, header_evidence_ref,
                      output_path, *, check_budget=lambda: None):
    """Freeze every eligible prior scope-only rejection before reading any move path."""
    output = Path(output_path)
    if output.exists():
        raise ValueError("The reconsideration ledger is immutable")
    value = _select(previous_sources_ref, previous_summary_ref, scope_audit_ref, header_evidence_ref, check_budget)
    if len(encoded(value).encode()) + 1 > 128 * expansion.MIB:
        raise ValueError("The frozen reconsideration ledger exceeds 128 MiB")
    _exclusive_json(output, value)
    return expansion.reference(output)


def _exclusive_json(path, value):
    with Path(path).open("x") as stream:
        stream.write(encoded(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def validate_request(request, sources, output):
    if request.get("schema") != REQUEST_VERSION or set(request.get("limits", {})) != set(CEILINGS):
        raise ValueError("The komi study requires its versioned request and explicit limits")
    for name, ceiling in CEILINGS.items():
        if type(request["limits"][name]) is not int or not 0 < request["limits"][name] <= ceiling:
            raise ValueError("Invalid or enlarged komi build limit: " + name)
    limits = request["limits"]
    if (limits["cpu_seconds"] - limits["preparation_cpu_seconds"] < 3600
            or limits["wall_seconds"] - limits["preparation_wall_seconds"] < 7200):
        raise ValueError("Publication requires one CPU hour and two elapsed hours in reserve")
    launch, prep, final = map(_utc, (request["launch_utc"], request["preparation_deadline_utc"], request["deadline_utc"]))
    if (launch > datetime.now(timezone.utc) or (prep - launch).total_seconds() != limits["preparation_wall_seconds"]
            or (final - launch).total_seconds() != limits["wall_seconds"]):
        raise ValueError("Absolute study deadlines differ from the one-use launch budget")
    if type(request["initial_cpu_seconds"]) is not int or request["initial_cpu_seconds"] != 300:
        raise ValueError("The metadata preflight retains its full conservative CPU charge")
    marker, preflight = expansion.read_ref(request["session_marker"]), expansion.read_ref(request["preflight"])
    if (marker.get("schema") != "e2_komi_production_session_v1"
            or any(marker[name] != request[name] for name in ("launch_utc", "preparation_deadline_utc", "deadline_utc"))
            or Path(marker["output_directory"]).resolve() != Path(output).resolve()
            or marker.get("workers") != 1 or marker.get("http_requests") != 0
            or marker.get("gpu_seconds") != 0 or marker.get("training_performed") is not False
            or preflight.get("schema") != "e2_komi_preflight_v1" or preflight.get("status") != "completed"
            or not _same_ref(preflight["session_marker"], request["session_marker"])
            or preflight.get("cpu_charged_seconds") != request["initial_cpu_seconds"]
            or not _same_ref(preflight["header_evidence"], sources["provenance"]["header_evidence"])):
        raise ValueError("The request does not match its charged one-use production preflight")
    for field, ceiling in (("cpu_seconds", 300), ("wall_seconds", 600)):
        value = preflight[field]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= ceiling:
            raise ValueError("The production metadata preflight exceeded its reserved allowance")
    for field in ("selected_sgf_bytes", "expanded_zip_bytes"):
        value = preflight[field]
        if type(value) is not int or value < 0 or value + limits[field] > CEILINGS[field]:
            raise ValueError("Preflight and build must share the same byte allowance: " + field)
    if not set(CODE_PATHS) <= set(request["source_revision"]["code_sha256"]):
        raise ValueError("The komi request must pin every builder dependency")
    if sources["counts"] != EXPECTED_COUNTS or sources["schema"] != SOURCES_VERSION:
        raise ValueError("The request requires the full frozen candidate population")
    baseline = expansion.read_ref(sources["provenance"]["previous_summary"])
    for key in ("canonical", "split_manifest", "examples_manifest", "profile", "base_index"):
        if not _same_ref(request["refs"][key], baseline["artifacts"][key]):
            raise ValueError("The komi study base differs from the preserved control")
    if set(request["combined_example_caps"]) != set(expansion.ALLOWED):
        raise ValueError("Every combined role cap must be explicit")
    for role in expansion.ALLOWED:
        added = limits["max_train_examples" if role == "train" else "max_development_examples"]
        if request["combined_example_caps"][role] != baseline["counts"]["combined"][role]["examples"] + added:
            raise ValueError("Combined row limits must add the new allowance to the unchanged base")


class StudyBudget(expansion.Budget):
    """Keep failed work, process restarts and both absolute deadlines in one allowance."""

    def __init__(self, output, request, request_pin, sources_pin, started, started_utc):
        self.preparation_deadline = _utc(request["preparation_deadline_utc"])
        self.preparation_cpu = request["limits"]["preparation_cpu_seconds"]
        path = output / "progress.json"
        if path.exists():
            state = json.loads(path.read_text())
            if state.get("schema") != VERSION:
                raise ValueError("Existing progress is not a komi study")
            if state.get("started_utc") != _utc(request["launch_utc"]).isoformat():
                raise ValueError("A resume cannot reset the production launch")
            charged = state.get("cpu_charged_seconds")
            if (type(charged) not in (int, float) or not math.isfinite(charged)
                    or charged < request["initial_cpu_seconds"]):
                raise ValueError("Saved production CPU costs are invalid")
        super().__init__(output, request, request_pin, sources_pin, started, _utc(request["launch_utc"]),
                         version=VERSION, initial_cpu=request["initial_cpu_seconds"])

    def write(self, **updates):
        # Publication also needs a durable charge if the process exits between checkpoints.
        self.reserved = 60
        super().write(**updates)

    def check(self, *, finishing=False):
        if self.cpu >= (self.limits["cpu_seconds"] - 60 if finishing else self.preparation_cpu):
            raise expansion.BudgetStop("cpu_budget" if finishing else "preparation_cpu_budget")
        if datetime.now(timezone.utc) >= (self.deadline if finishing else self.preparation_deadline):
            raise expansion.BudgetStop("absolute_deadline" if finishing else "preparation_deadline")
        if time.process_time() - self.last_saved >= 10:
            self.write()


@contextmanager
def _writer_lock(request, request_pin, sources_pin, output):
    marker = Path(request["session_marker"]["path"])
    binding_path = marker.with_name(marker.name + ".build-binding.json")
    binding = {"schema": VERSION, "request_sha256": request_pin, "sources_sha256": sources_pin,
               "output_directory": str(output.resolve()), "session_marker": request["session_marker"]}
    lock_path = marker.with_name(marker.name + ".writer.lock")
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("A komi production writer is already active") from error
        try:
            if marker.with_name(marker.name + ".verification.json").exists():
                raise ValueError("A published study cannot resume building after verification has started")
            try:
                _exclusive_json(binding_path, binding)
                fresh = True
                if output.exists() and any(output.iterdir()):
                    raise ValueError("A new one-use build requires an empty output directory")
            except FileExistsError:
                fresh = False
                if json.loads(binding_path.read_text()) != binding:
                    raise ValueError("The one-use session already belongs to another build")
                if not (output / "progress.json").exists():
                    raise ValueError("A previous attempt lacks durable costs; the allowance cannot reset")
            yield fresh
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def build(request_path, request_sha256, sources_path, sources_sha256, output):
    started, started_utc = time.process_time(), datetime.now(timezone.utc)
    request = expansion.read_ref({"path": str(request_path), "sha256": request_sha256})
    output = Path(output)
    with _writer_lock(request, request_sha256, sources_sha256, output):
        output.mkdir(parents=True, exist_ok=True)
        budget = StudyBudget(output, request, request_sha256, sources_sha256, started, started_utc)
        try:
            budget.check(finishing=True)
            budget.write(phase="inputs")
            sources = expansion.read_ref({"path": str(sources_path), "sha256": sources_sha256}, maximum=128 * expansion.MIB)
            validate_request(request, sources, output)
            preflight = expansion.read_ref(request["preflight"])
            if not _same_ref(preflight["sources"], {"path": str(sources_path), "sha256": sources_sha256}):
                raise ValueError("Frozen candidates differ from the charged preflight")
            provenance = sources["provenance"]
            reproduced = _select(provenance["previous_sources"], provenance["previous_summary"],
                                 provenance["scope_audit"], provenance["header_evidence"],
                                 lambda: budget.check(finishing=True))
            if reproduced != sources:
                raise ValueError("The reconsideration ledger changed its candidates, order or prior evidence")
            ids = {item["source"]["source_url"] for item in sources["candidates"]}
            inputs = expansion._inputs(request, sources, komi_study=True, reconsidered_sources=ids)
            budget.end(phase="inputs_validated")
            return expansion._execute_build(output, request, sources, budget, inputs, request_sha256,
                                             sources_sha256, komi_study=True)
        except Exception as error:
            budget.write(phase="failed", status="failed", error_type=type(error).__name__, error=str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("sources", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--sources-sha256", required=True)
    args = parser.parse_args()
    result = build(args.request, args.request_sha256, args.sources, args.sources_sha256, args.output)
    print(json.dumps({name: result[name] for name in ("status", "new_encoded_games", "counts", "costs")}, indent=2))


if __name__ == "__main__":
    main()
