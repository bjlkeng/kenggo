"""Append checked training and development games to a frozen supervised corpus."""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import time
import zipfile
import zlib

import pyarrow as pa
import pyarrow.parquet as pq
from sgfmill import sgf

from kenggo.canonical_data import canonicalize, digest, encoded
from kenggo.cached_coverage_audit import participants, provisional_role
from kenggo.data_splits import (ROLES, SplitPlan, build_splits, duplicate_components,
                                read_membership, write_membership)
from kenggo.e1_audit import member_key, source_key
from kenggo.examples import (ExampleConfig, examples_for_game, fit_normalizer,
                             load_dataset_profile, read_shards, write_shard)
from kenggo.examples_audit import diagnostic_sample
from kenggo.kgs_admission import DECISION_SHA256, classify
from kenggo.kgs_eligibility import Exposure, HISTORICAL_BOT_HANDLES, grouping_digest, normalize_handle
from kenggo.kgs_source import candidate_reasons
from kenggo.population_audit import split_entry
from kenggo.sgf_audit import inspect_sgf, scalar
from kenggo.kgs_source import player


VERSION = "e2_data_expansion_v1"
REQUEST_VERSION = "e2_data_expansion_request_v1"
ALLOWED = ("train", "population_development", "validation_development")
REFS = {"canonical", "split_manifest", "examples_manifest", "profile", "original_audit",
        "owner_exclusions", "prior_exposure", "source_policy"}
MIB = 1024**2
CEILINGS = {"cpu_seconds": 68000, "selected_sgf_bytes": 256 * MIB, "member_bytes": MIB,
            "expanded_zip_bytes": 4096 * MIB, "canonical_file_bytes": 64 * MIB,
            "output_bytes": 16 * 1024**3, "max_train_examples": 1_000_000,
            "max_development_examples": 250_000, "max_records": 50_000}
# The pinned writer sorts keys; this suffix yields a hash without decoding old targets.
TRAJECTORY = re.compile(r',"trajectory_sha256":(?:"([0-9a-f]{64})"|null),"version":"canonical_kgs_v1"}$')
SEQUENCE = re.compile(r'(?<!\\)"sequence_fingerprint"\s*:\s*"([0-9a-f]{64})"')
CODE_PATHS = tuple("src/kenggo/" + name for name in (
    "data_expansion.py", "canonical_data.py", "data_splits.py", "encoding.py", "examples.py",
    "examples_audit.py", "go.py", "rules_replay.py", "adjudication.py", "sgf_audit.py",
    "kgs_admission.py", "kgs_eligibility.py", "kgs_source.py", "e1_audit.py",
    "population_audit.py", "cached_coverage_audit.py")) + ("uv.lock",)


def file_hash(path):
    result = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(MIB), b""):
            result.update(chunk)
    return result.hexdigest()


def read_ref(reference, *, json_data=True, maximum=64 * MIB):
    if (not isinstance(reference, dict) or set(reference) != {"path", "sha256"}
            or not isinstance(reference["path"], str)
            or re.fullmatch(r"[0-9a-f]{64}", reference.get("sha256", "")) is None):
        raise ValueError("A pinned path and SHA-256 are required")
    path = Path(reference["path"])
    if path.stat().st_size > maximum or file_hash(path) != reference["sha256"]:
        raise ValueError("Pinned input changed or exceeded its size limit: " + str(path))
    return json.loads(path.read_bytes()) if json_data else path


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("w") as stream:
        stream.write(encoded(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return file_hash(path)


def reference(path):
    return {"path": str(path), "sha256": file_hash(path)}


class BudgetStop(RuntimeError):
    pass


class Budget:
    def __init__(self, output, request, request_pin, sources_pin, started, started_utc):
        self.output, self.limits = output, request["limits"]
        self.deadline = datetime.fromisoformat(request["deadline_utc"].replace("Z", "+00:00"))
        if self.deadline.tzinfo is None:
            raise ValueError("The preparation deadline requires a UTC offset")
        self.started = started
        self.last_saved = self.started
        self.reserved = 0
        path = output / "progress.json"
        if path.exists():
            self.state = json.loads(path.read_text())
            if (self.state["request_sha256"] != request_pin or self.state["sources_sha256"] != sources_pin
                    or self.state["schema"] != VERSION):
                raise ValueError("Resume inputs differ from the saved build")
        else:
            self.state = {"schema": VERSION, "request_sha256": request_pin, "sources_sha256": sources_pin,
                          "started_utc": started_utc.isoformat(), "cpu_charged_seconds": 0,
                          "phase": "inputs", "canonical_completed": 0, "games_encoded": 0,
                          "selected_sgf_bytes": 0, "expanded_zip_bytes": 0}
        self.prior_cpu = self.state["cpu_charged_seconds"]
        for path in output.rglob("*.writing"):
            path.unlink()
        self.file_sizes = {str(path): path.stat().st_size for path in output.rglob("*") if path.is_file()}
        self.used_bytes = sum(self.file_sizes.values())
        self.write()

    @property
    def cpu(self):
        return self.prior_cpu + time.process_time() - self.started

    def write(self, **updates):
        self.state.update(updates)
        self.state.update(cpu_seconds=self.cpu, cpu_charged_seconds=self.cpu + self.reserved,
                          updated_utc=datetime.now(timezone.utc).isoformat(),
                          wall_seconds=(datetime.now(timezone.utc) - datetime.fromisoformat(self.state["started_utc"])).total_seconds())
        path = self.output / "progress.json"
        self.require_bytes(len(encoded(self.state).encode()) + 1)
        save_json(path, self.state)
        self.account(self.output / "progress.json")
        self.last_saved = time.process_time()

    def check(self, *, finishing=False):
        reserve = 0 if finishing else 120
        if self.cpu >= self.limits["cpu_seconds"] - reserve:
            raise BudgetStop("cpu_budget")
        if (self.deadline - datetime.now(timezone.utc)).total_seconds() <= reserve:
            raise BudgetStop("absolute_deadline")
        if time.process_time() - self.last_saved >= 20:
            self.write()

    def begin(self, **updates):
        self.check()
        # An interrupted unit retains this conservative CPU charge on resume.
        self.reserved = 60
        self.write(**updates)

    def end(self, **updates):
        self.reserved = 0
        self.write(**updates)

    def require_bytes(self, additional):
        if self.used_bytes + max(0, additional) > self.limits["output_bytes"]:
            raise BudgetStop("output_byte_budget")

    def account(self, path):
        key, size = str(path), Path(path).stat().st_size
        self.used_bytes += size - self.file_sizes.get(key, 0)
        self.file_sizes[key] = size

    def save(self, path, value):
        self.require_bytes(len(encoded(value).encode()) + 1)
        pin = save_json(path, value)
        self.account(path)
        return pin

    def remove_tree(self, directory):
        for path in directory.rglob("*"):
            if path.is_file():
                self.used_bytes -= self.file_sizes.pop(str(path), 0)
        shutil.rmtree(directory)

    def remove_file(self, path):
        if path.exists():
            self.used_bytes -= self.file_sizes.pop(str(path), 0)
            path.unlink()


def exposure_fields(value):
    aliases = {"sgf_hashes": "sgf_sha256", "sequence_hashes": "sequence_sha256"}
    result = {}
    for field in ("groups", "source_ids", "sgf_hashes", "sequence_hashes", "account_months"):
        items = value.get(field, value.get(aliases.get(field, ""), []))
        if not isinstance(items, list):
            raise ValueError("Exposure fields must be lists")
        result[field] = {tuple(item) if field == "account_months" else item for item in items}
    return result


def union_exposure(*values):
    result = {field: set() for field in exposure_fields({})}
    for value in values:
        for field, items in exposure_fields(value).items():
            result[field].update(items)
    return result


def _inputs(request, sources):
    if request.get("schema") != REQUEST_VERSION or set(request["refs"]) != REFS:
        raise ValueError("Unsupported expansion request or missing base references")
    limits = request["limits"]
    if set(limits) != set(CEILINGS):
        raise ValueError("Every build limit must be explicit")
    for name, ceiling in CEILINGS.items():
        if type(limits[name]) is not int or not 0 < limits[name] <= ceiling:
            raise ValueError("Invalid or enlarged build limit: " + name)
    if (sources.get("schema") != "e2_expansion_sources_v1"
            or sources["request_sha256"] != request["source_request_sha256"]):
        raise ValueError("The source collection does not match the build request")
    code = request["source_revision"].get("code_sha256", {})
    if not set(CODE_PATHS) <= set(code):
        raise ValueError("The source revision must pin every builder dependency")
    for path, pin in code.items():
        read_ref({"path": path, "sha256": pin}, json_data=False, maximum=16 * MIB)
    refs = request["refs"]
    profile = load_dataset_profile(refs["profile"]["path"], refs["profile"]["sha256"])
    canonical = read_ref(refs["canonical"], json_data=False)
    split_path = read_ref(refs["split_manifest"], json_data=False)
    split, partitions = read_membership(split_path.parent, refs["split_manifest"]["sha256"])
    examples = read_ref(refs["examples_manifest"])
    if (refs["canonical"]["sha256"] != profile.canonical_sha256
            or refs["split_manifest"]["sha256"] != profile.split_sha256
            or examples["canonical_sha256"] != profile.canonical_sha256
            or examples["split_sha256"] != profile.split_sha256
            or examples["dataset_profile_sha256"] != profile.sha256
            or split["plan"] != request["split_config"]
            or digest(split["plan"]) != split["plan_sha256"]
            or grouping_digest(request["base_groups"]) != split["plan"]["grouping_sha256"]
            or sources["base_grouping_sha256"] != split["plan"]["grouping_sha256"]):
        raise ValueError("The base profile, split, examples or groups do not agree")
    if (digest(sorted(split["sealed_source_ids"])) != profile.sealed_source_ids_sha256
            or examples["sealed_source_ids"] != sorted(split["sealed_source_ids"])):
        raise ValueError("The base sealing registry changed")
    for role in ROLES:
        if (examples["counts"][role]["examples"] != split["policy_actions_by_role"][role]
                or examples["counts"][role]["value_valid"] != split["value_actions_by_role"][role]):
            raise ValueError("Base example and split counts differ")
    for role in ALLOWED:
        cap = limits["max_train_examples" if role == "train" else "max_development_examples"]
        if examples["counts"][role]["examples"] > cap:
            raise ValueError("The requested row cap is smaller than the preserved base")
    groups = sources["groups"]
    if (any(groups.get(handle) != group for handle, group in request["base_groups"].items())
            or any(normalize_handle(handle) != handle or not isinstance(group, str)
                   or not group.startswith("kgs-handle:") for handle, group in groups.items())):
        raise ValueError("An existing handle group changed")
    owner = read_ref(refs["owner_exclusions"])
    if (split["provenance"].get("manual_review_sha256") is not None
            and split["provenance"]["manual_review_sha256"] != refs["owner_exclusions"]["sha256"]):
        raise ValueError("Owner exclusions differ from the reviewed base decision")
    owner_hashes = {row["sgf_sha256"] for row in owner["excluded_games"]}
    if set(sources["owner_excluded_sgf_sha256"]) != owner_hashes:
        raise ValueError("The source collection omitted an owner exclusion")
    rows = [row for role in ROLES for row in partitions[role]]
    by_id = {row["source_url"]: row for row in rows}
    for item in owner["excluded_games"]:
        row = by_id.get(item["source_url"])
        if row is None or row["role"] != "excluded" or row["canonical_sha256"] != item["canonical_sha256"]:
            raise ValueError("The base split does not preserve an owner exclusion")
    if refs["source_policy"]["sha256"] != DECISION_SHA256:
        raise ValueError("Source screening policy changed")
    read_ref(refs["source_policy"])
    exposure = union_exposure(read_ref(refs["prior_exposure"]), sources["exposure"])
    reserved = set(split["plan"]["test_groups"])
    if exposure["groups"] & reserved:
        raise ValueError("A reserved final group entered the exposure registry")
    sealed = set(split["sealed_source_ids"])
    old_groups = {group for row in rows if row["source_url"] not in sealed for group in row["groups"] if group}
    if not old_groups <= exposure["groups"]:
        raise ValueError("The exposure union dropped an old inspected group")
    candidates = sources["candidates"]
    ids = [item["source"]["source_url"] for item in candidates]
    if len(set(ids)) != len(ids) or set(ids) & set(by_id):
        raise ValueError("Source IDs are duplicated or already belong to the base corpus")
    if len(candidates) + len(rows) > limits["max_records"]:
        raise ValueError("Candidate record limit exceeded")
    for item in candidates:
        row, source = item["archive_row"], item["source"]
        if (source["source_url"] != row["url"] or item["provisional_role"] not in ALLOWED
                or provisional_role(row, groups, split["plan"]) != item["provisional_role"]
                or candidate_reasons(row) or any("bot" in handle for handle, _ in participants(row, groups))
                or row["url"] not in exposure["source_ids"]
                or any(group not in exposure["groups"] for _, group in participants(row, groups))):
            raise ValueError("Candidate identity, cohort or exposure does not agree")
    return canonical, split, partitions, examples, groups, exposure, owner_hashes


def base_fingerprints(canonical, rows, audit_ref):
    expected = {row["source_url"]: row["canonical_sha256"] for row in rows}
    fingerprints = {"sgf_sha256": set(), "audit_sequence_sha256": set(), "trajectory_sha256": set()}
    seen = set()
    for batch in pq.ParquetFile(canonical).iter_batches(batch_size=64):
        for row in batch.to_pylist():
            key, raw = row["source_url"], row["canonical_json"]
            if (key in seen or expected.get(key) != row["canonical_sha256"]
                    or sha256(raw.encode()).hexdigest() != row["canonical_sha256"]):
                raise ValueError("An opaque base canonical record changed")
            tail = TRAJECTORY.search(raw)
            if tail is None:
                raise ValueError("Unsupported opaque canonical record format")
            if tail[1]:
                fingerprints["trajectory_sha256"].add(tail[1])
            seen.add(key)
    if seen != set(expected):
        raise ValueError("Base canonical IDs differ from the base memberships")
    audit_path = read_ref(audit_ref, json_data=False)
    for batch in pq.ParquetFile(audit_path).iter_batches(batch_size=128,
            columns=["source_url", "sgf_sha256", "audit_json"]):
        for row in batch.to_pylist():
            if row["source_url"] not in expected:
                raise ValueError("Original audit contains a source outside the base")
            if row["sgf_sha256"]:
                fingerprints["sgf_sha256"].add(row["sgf_sha256"])
            matches = SEQUENCE.findall(row["audit_json"])
            if len(matches) > 1:
                raise ValueError("Ambiguous original sequence fingerprint")
            fingerprints["audit_sequence_sha256"].update(matches)
    return fingerprints


def _header_for_fingerprints(data, row, groups, reserved):
    # Rank availability affects admission, not whether a permitted copy can be identified.
    start = re.match(br"\xef\xbb\xbf\s*\(\s*;|\s*\(\s*;", data)
    if start is None:
        raise ValueError("invalid_sgf_root")
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
        raise ValueError("invalid_sgf_root")
    root = sgf.Sgf_game.from_bytes(data[:end] + b")").get_root()
    rank_conflict = False
    for color, name, rank in (("black", "PB", "BR"), ("white", "PW", "WR")):
        declared = player(row[color])
        handle = scalar(root, name)
        if declared is None or handle != declared[0]:
            raise ValueError("archive_sgf_identity_mismatch")
        normalized = normalize_handle(handle)
        if groups.get(normalized, "kgs-handle:" + normalized) in reserved:
            raise ValueError("reserved_final_participant")
        rank_conflict |= scalar(root, rank) != declared[1]
    if scalar(root, "DT") != source_key(row["url"])[0]:
        raise ValueError("archive_sgf_date_mismatch")
    return rank_conflict


def _canonical_unit(item, groups, exposure, owner_hashes, limits, budget, archives):
    row, source = item["archive_row"], dict(item["source"])
    key = source["archive_path"]
    if key not in archives:
        path = read_ref({"path": key, "sha256": source["archive_sha256"]}, json_data=False,
                        maximum=64 * MIB)
        archive = zipfile.ZipFile(path)
        expanded = sum(member.file_size for member in archive.infolist())
        charged = budget.state.setdefault("expanded_archives", {})
        previous = charged.get(key)
        if previous is not None and previous != {"sha256": source["archive_sha256"], "bytes": expanded}:
            archive.close()
            raise ValueError("A previously charged archive changed")
        if previous is None and budget.state["expanded_zip_bytes"] + expanded > limits["expanded_zip_bytes"]:
            archive.close()
            raise BudgetStop("expanded_zip_budget")
        archives[key] = (source["archive_sha256"], archive)
        if previous is None:
            budget.state["expanded_zip_bytes"] += expanded
            charged[key] = {"sha256": source["archive_sha256"], "bytes": expanded}
    pin, archive = archives[key]
    if pin != source["archive_sha256"]:
        raise ValueError("One archive path has conflicting pins")
    matches = [member for member in archive.infolist() if member_key(member.filename) == source_key(row["url"])]
    if len(matches) != 1 or matches[0].filename != source["zip_member"]:
        raise ValueError("Selected ZIP member is ambiguous or changed")
    info = matches[0]
    if info.file_size > limits["member_bytes"]:
        return {"status": "source_excluded", "reason": "member_size_limit"}
    if budget.state["selected_sgf_bytes"] + info.file_size > limits["selected_sgf_bytes"]:
        raise BudgetStop("selected_sgf_budget")
    budget.state["selected_sgf_bytes"] += info.file_size
    budget.write()
    try:
        with archive.open(info) as stream:
            data = stream.read(limits["member_bytes"] + 1)
    except (zipfile.BadZipFile, RuntimeError, EOFError, zlib.error):
        return {"status": "source_excluded", "reason": "selected_member_read_error",
                "selected_bytes": info.file_size}
    if len(data) != info.file_size or len(data) > limits["member_bytes"]:
        raise ValueError("Selected SGF size differs from its ZIP directory")
    source["sgf_sha256"] = sha256(data).hexdigest()
    result = {"sgf_sha256": source["sgf_sha256"], "selected_bytes": len(data)}
    if source["sgf_sha256"] in owner_hashes:
        return {**result, "status": "source_excluded", "reason": "owner_excluded_sgf_bytes"}
    try:
        rank_conflict = _header_for_fingerprints(data, row, groups, set(budget.state["test_groups"]))
    except (ValueError, TypeError, KeyError, UnicodeError, LookupError):
        return {**result, "status": "source_excluded", "reason": "source_header_rejected"}
    audit = inspect_sgf(data, row, HISTORICAL_BOT_HANDLES)
    audit["source_url"] = row["url"]
    gate = classify(audit, row, groups, Exposure(grouping_digest(groups), **{
        name: frozenset(items) for name, items in exposure.items()}))
    result["sequence_fingerprint"] = audit.get("sequence_fingerprint")
    record, cost = canonicalize(data, source, row)
    result.update(trajectory_sha256=record["trajectory_sha256"], adjudication_cost=cost)
    if rank_conflict or not gate["training_eligible"]:
        return {**result, "status": "source_excluded",
                "reason": "source_header_rejected" if rank_conflict else "source_policy",
                "exclusions": gate["eligibility_reasons"]}
    record.update(eligibility=gate, sealed_final=False)
    return {**result, "status": "canonical", "canonical_json": encoded(record),
            "entry": split_entry(record, grouping_digest(groups)), "adjudication_cost": cost,
            "replay_validated": record["replay_validated"]}


def _unit_descriptor(path, item, groups):
    pinned = json.loads(path.with_suffix(".ref.json").read_text())
    if pinned["path"] != str(path):
        raise ValueError("Saved unit reference points outside its expected path")
    unit = read_ref(pinned, maximum=256 * MIB)
    if unit["source_url"] != item["source"]["source_url"]:
        raise ValueError("Saved canonical unit belongs to another source")
    if unit["status"] == "canonical":
        record = json.loads(unit.pop("canonical_json"))
        if (record["source"]["source_url"] != unit["source_url"]
                or split_entry(record, grouping_digest(groups)) != unit["entry"]):
            raise ValueError("Saved canonical evidence changed")
    unit["unit_ref"] = pinned
    return unit


def _canonical_raw(unit):
    saved = read_ref(unit["unit_ref"], maximum=256 * MIB)
    raw = saved["canonical_json"]
    if sha256(raw.encode()).hexdigest() != unit["entry"]["canonical_sha256"]:
        raise ValueError("Saved canonical payload differs from its split evidence")
    return raw


def _canonical_phase(output, candidates, groups, exposure, owner_hashes, request, budget):
    work = output / "work/canonical"
    work.mkdir(parents=True, exist_ok=True)
    completed = output / "work/canonical-complete.json"
    archives, units, stop = {}, [], None
    if completed.exists():
        if file_hash(completed) != budget.state.get("canonical_phase_sha256"):
            raise ValueError("Saved canonical phase manifest changed")
        count = json.loads(completed.read_text())["completed"]
        units = [_unit_descriptor(work / f"{index:06d}.json", candidates[index], groups) for index in range(count)]
        return units, json.loads(completed.read_text())["stop_reason"]
    try:
        for number, item in enumerate(candidates):
            path = work / f"{number:06d}.json"
            if path.with_suffix(".ref.json").exists():
                units.append(_unit_descriptor(path, item, groups))
                continue
            try:
                budget.begin(phase="canonical", current_source=item["source"]["source_url"],
                             canonical_attempted=number + 1)
                unit = _canonical_unit(item, groups, exposure, owner_hashes, request["limits"], budget, archives)
                unit["source_url"] = item["source"]["source_url"]
                pin = budget.save(path, unit)
                budget.save(path.with_suffix(".ref.json"), {"path": str(path), "sha256": pin})
                units.append(_unit_descriptor(path, item, groups))
                budget.end(canonical_completed=len(units), current_source=None)
            except BudgetStop as error:
                stop = str(error)
                break
    finally:
        for _, archive in archives.values():
            archive.close()
    pin = budget.save(completed, {"completed": len(units), "stop_reason": stop})
    budget.write(canonical_phase_sha256=pin)
    return units, stop


def _new_memberships(units, fingerprints, groups, split, exposure):
    entries = [unit["entry"] for unit in units if unit["status"] == "canonical"]
    rejected = {"sgf_sha256": {unit["sgf_sha256"] for unit in units if unit["status"] != "canonical" and unit.get("sgf_sha256")},
                "audit_sequence_sha256": {unit["sequence_fingerprint"] for unit in units
                                          if unit["status"] != "canonical" and unit.get("sequence_fingerprint")},
                "trajectory_sha256": {unit["trajectory_sha256"] for unit in units
                                      if unit["status"] != "canonical" and unit.get("trajectory_sha256")}}
    for component in duplicate_components(entries):
        reasons = []
        if any(entry.get(field) in values for entry in component for field, values in fingerprints.items()):
            reasons.append("duplicate_of_frozen_corpus")
        if any(entry.get(field) in values for entry in component for field, values in rejected.items()):
            reasons.append("duplicate_of_rejected_source")
        if reasons:
            for entry in component:
                entry["exclusions"] = sorted(set(entry["exclusions"] + reasons))
    plan = {**split["plan"], "grouping_sha256": grouping_digest(groups)}
    rows = build_splits(entries, SplitPlan.from_dict(plan), exposed_groups=exposure["groups"], group_registry=groups)
    if any(row["role"] not in ALLOWED + ("excluded",) for row in rows):
        raise ValueError("A new game entered a forbidden role")
    order = {unit["source_url"]: index for index, unit in enumerate(units)}
    return entries, sorted(rows, key=lambda row: order[row["source_url"]]), plan


def _example_phase(output, units, rows, plan, examples, request, budget):
    config = ExampleConfig(**examples["config"])
    records = {unit["source_url"]: unit for unit in units if unit["status"] == "canonical"}
    counts = {role: examples["counts"][role]["examples"] for role in ALLOWED}
    games, stopped, capped = [], None, False
    work = output / "work/examples"
    work.mkdir(exist_ok=True)
    for row in rows:
        role = row["role"]
        if role not in ALLOWED:
            continue
        cap = request["limits"]["max_train_examples" if role == "train" else "max_development_examples"]
        reason = "example_row_limit" if counts[role] + row["policy_actions"] > cap else stopped
        capped |= reason == "example_row_limit"
        directory = work / digest(row["source_url"])
        manifest_path = directory / "game.json"
        if reason is None:
            try:
                if manifest_path.exists():
                    game = json.loads(manifest_path.read_text())
                    if game["membership_sha256"] != digest(row):
                        raise ValueError("Saved examples belong to another membership")
                    for artifact in game["artifacts"]:
                        for kind in ("arrays", "metadata"):
                            if file_hash(directory / artifact[kind]) != artifact[kind + "_sha256"]:
                                raise ValueError("Saved game shard changed")
                else:
                    budget.begin(phase="examples", current_source=row["source_url"])
                    generated = []
                    record = json.loads(_canonical_raw(records[row["source_url"]]))
                    for example in examples_for_game(record, row, plan, config):
                        budget.check()
                        generated.append(example)
                    if directory.exists():
                        budget.remove_tree(directory)
                    estimated = sum(4096 + 2 * len(encoded(example[2]).encode()) for example in generated)
                    estimated += 128 * 1024 * (1 + len(generated) // config.shard_size)
                    budget.require_bytes(estimated)
                    directory.mkdir()
                    artifacts = [write_shard(directory, role, index // config.shard_size,
                                             generated[index:index + config.shard_size], config.history)
                                 for index in range(0, len(generated), config.shard_size)]
                    for artifact in artifacts:
                        for kind in ("arrays", "metadata"):
                            budget.account(directory / artifact[kind])
                    count = {"examples": len(generated), "value_valid": sum(e[1]["value_valid"] for e in generated),
                             "duration_valid": sum(e[1]["duration_valid"] for e in generated),
                             "passes": sum(e[1]["action"] == 361 for e in generated),
                             "hidden_mover_rank": sum(e[2]["rank_hidden"][0] for e in generated),
                             "hidden_opponent_rank": sum(e[2]["rank_hidden"][1] for e in generated)}
                    game = {"source_url": row["source_url"], "role": role, "membership_sha256": digest(row),
                            "artifacts": artifacts, "counts": count}
                    budget.save(manifest_path, game)
                    budget.end(games_encoded=len(games) + 1, current_source=None,
                               combined_rows={**counts, role: counts[role] + count["examples"]})
                counts[role] += game["counts"]["examples"]
                games.append((directory, game))
            except BudgetStop as error:
                reason = stopped = str(error)
        if reason:
            row.update(role="excluded", policy_actions=0, value_actions=0,
                       adaptation_colors=[], known_persona_colors=[],
                       exclusions=sorted(set(row["exclusions"] + ["expansion_" + reason])))
    return games, stopped or ("example_row_limit" if capped else None)


def _copy_file(source, target, pin, budget):
    budget.check(finishing=True)
    if file_hash(source) != pin:
        raise ValueError("A source shard changed")
    target.parent.mkdir(parents=True, exist_ok=True)
    budget.require_bytes(source.stat().st_size - budget.file_sizes.get(str(target), 0))
    shutil.copyfile(source, target)
    budget.account(target)
    if file_hash(target) != pin:
        raise ValueError("A copied shard changed")


def _make_canonical(output, canonical, units, request, budget):
    budget.write(phase="canonical_file", current_source=None)
    target = output / "work/combined-canonical.parquet"
    budget.remove_file(target)
    schema = pq.ParquetFile(canonical).schema_arrow
    with pq.ParquetWriter(target, schema, compression="zstd") as writer:
        for batch in pq.ParquetFile(canonical).iter_batches(batch_size=64):
            budget.check(finishing=True)
            budget.require_bytes(batch.nbytes)
            writer.write_batch(batch)
            budget.account(target)
        batch = []
        for unit in units:
            if unit["status"] == "canonical":
                raw = _canonical_raw(unit)
                batch.append({"source_url": unit["source_url"], "canonical_sha256": sha256(raw.encode()).hexdigest(),
                              "canonical_json": raw})
            if len(batch) == 64:
                table = pa.Table.from_pylist(batch, schema=schema)
                budget.check(finishing=True)
                budget.require_bytes(table.nbytes)
                writer.write_table(table)
                budget.account(target)
                batch = []
        if batch:
            table = pa.Table.from_pylist(batch, schema=schema)
            budget.require_bytes(table.nbytes)
            writer.write_table(table)
    budget.account(target)
    if target.stat().st_size > request["limits"]["canonical_file_bytes"]:
        raise ValueError("Combined canonical file exceeds the explicit file limit")
    return target


def _publish(output, canonical, split, old_rows, examples, new_rows, units, games, plan,
             groups, exposure, request, request_pin, sources_pin, budget):
    published = output / "dataset"
    if published.exists():
        budget.remove_tree(published)
    published.mkdir()
    canonical_pin = file_hash(canonical)
    _copy_file(canonical, published / "canonical.parquet", canonical_pin, budget)
    rows = old_rows + new_rows
    derived = deepcopy(split)
    derived.pop("membership_sha256", None)
    derived.update(plan=plan, plan_sha256=digest(plan),
                   canonical_input_sha256=digest({"parent": split["canonical_input_sha256"],
                       "added": [unit["entry"] for unit in units if unit["status"] == "canonical"]}),
                   exposure_sha256=digest({key: sorted(value) for key, value in exposure.items()}))
    derived["provenance"] = {**derived["provenance"], "population_inputs_sha256": request_pin,
        "expansion_request_sha256": request_pin, "source_collection_sha256": sources_pin,
        "parent_split": request["refs"]["split_manifest"], "parent_profile": request["refs"]["profile"],
        "canonical_input_derivation": "Hash of parent input hash and added split evidence; old rows are unchanged."}
    derived["roles"] = {role: sum(row["role"] == role for row in rows) for role in ROLES}
    for output_key, field in (("policy_actions_by_role", "policy_actions"), ("value_actions_by_role", "value_actions")):
        derived[output_key] = {role: sum(row[field] for row in rows if row["role"] == role) for role in ROLES}
    derived["exclusion_counts"] = dict(sorted(Counter(reason for row in rows for reason in row["exclusions"]).items()))
    derived["adaptation_sides"] = sum(len(row["adaptation_colors"]) for row in rows)
    budget.require_bytes(4 * sum(len(encoded(row).encode()) for row in rows) + 16 * MIB)
    split_pin = write_membership(published / "splits", rows, derived)
    for path in (published / "splits").iterdir():
        budget.account(path)
    resulting, partitions = read_membership(published / "splits", split_pin)
    by_id = {row["source_url"]: row for role in ROLES for row in partitions[role]}
    if any(by_id[row["source_url"]] != row for row in old_rows):
        raise ValueError("An old membership changed during derivation")
    for role in ROLES:
        if role not in ALLOWED + ("excluded",) and resulting["membership_sha256"][role + ".parquet"] != split["membership_sha256"][role + ".parquet"]:
            raise ValueError("An untouched membership file changed")
    profile = {"version": "population_example_profile_v1", "canonical_sha256": canonical_pin,
        "split_sha256": split_pin, "population_inputs_sha256": request_pin,
        "sealed_source_ids_sha256": digest(sorted(split["sealed_source_ids"])),
        "limits": {"max_records": request["limits"]["max_records"],
                   "max_examples_per_role": max(request["limits"]["max_train_examples"],
                                                request["limits"]["max_development_examples"]),
                   "cpu_seconds": request["limits"]["cpu_seconds"], "wall_seconds": 86400}}
    profile_pin = budget.save(published / "profile.json", profile)
    manifest = deepcopy(examples)
    source_dir = Path(request["refs"]["examples_manifest"]["path"]).parent
    example_dir = published / "examples"
    example_dir.mkdir()
    for role in ROLES:
        for number, item in enumerate(examples["artifacts"][role]):
            for kind, suffix in (("arrays", "npz"), ("metadata", "parquet")):
                if item[kind] != f"{role}/{number:05d}.{suffix}":
                    raise ValueError("Unexpected base shard path")
                _copy_file(source_dir / item[kind], example_dir / item[kind], item[kind + "_sha256"], budget)
    for directory, game in games:
        role = game["role"]
        for item in game["artifacts"]:
            number = len(manifest["artifacts"][role])
            added = dict(item)
            for kind, suffix in (("arrays", "npz"), ("metadata", "parquet")):
                added[kind] = f"{role}/{number:05d}.{suffix}"
                _copy_file(directory / item[kind], example_dir / added[kind], item[kind + "_sha256"], budget)
            manifest["artifacts"][role].append(added)
        for key, value in game["counts"].items():
            manifest["counts"][role][key] += value
    for role in ROLES:
        if (manifest["counts"][role]["examples"] != resulting["policy_actions_by_role"][role]
                or manifest["counts"][role]["value_valid"] != resulting["value_actions_by_role"][role]):
            raise ValueError("Combined examples differ from the whole-game memberships")
    manifest.update(canonical_sha256=canonical_pin, split_sha256=split_pin, dataset_profile_sha256=profile_pin,
                    source_roles=resulting["roles"], exclusions=resulting["exclusion_counts"],
                    code_sha256={name: file_hash(Path(__file__).with_name(name)) for name in examples["code_sha256"]},
                    uv_lock_sha256=file_hash("uv.lock"),
                    derivation={"request_sha256": request_pin, "source_collection_sha256": sources_pin,
                                "parent_examples": request["refs"]["examples_manifest"],
                                "builder_sha256": file_hash(__file__)})
    examples_pin = budget.save(example_dir / "manifest.json", manifest)
    budget.write(phase="normalizer", combined_rows={role: manifest["counts"][role]["examples"] for role in ALLOWED})
    normalizer = fit_normalizer(example_dir, examples_pin)
    if normalizer["count"][5] != manifest["counts"]["train"]["examples"]:
        raise ValueError("Combined training normalizer count mismatch")
    budget.save(published / "normalizer.json", normalizer)
    budget.save(published / "groups.json", groups)
    budget.save(published / "exposure.json", {key: sorted(value) for key, value in exposure.items()})
    return published, manifest, examples_pin


def support_counts(directory, pin, check_budget=lambda: None):
    result = {}
    for role in ALLOWED:
        cells = defaultdict(lambda: {"rows": 0, "passes": 0, "value_valid": 0, "sides": set(), "accounts": set()})
        for arrays, metadata in read_shards(directory, pin, role):
            check_budget()
            for index, row in enumerate(metadata):
                phase = "1-60" if row["ply"] < 60 else "61-180" if row["ply"] < 180 else "181+"
                clock = "10-<20m" if arrays["scalars"][index, 9] < 1200 else "20-30m"
                for key in ((row["source_rank_mover"], "all", "all"), (row["source_rank_mover"], phase, clock)):
                    cell = cells[key]
                    cell["rows"] += 1
                    cell["passes"] += int(arrays["action"][index] == 361)
                    cell["value_valid"] += int(arrays["value_valid"][index])
                    cell["sides"].add((row["source_url"], row["mover"]))
                    cell["accounts"].add(row["group_mover"])
        result[role] = [{"rank": key[0], "move_bin": key[1], "time_bin": key[2],
                        "rows": value["rows"], "passes": value["passes"], "value_valid": value["value_valid"],
                        "game_sides": len(value["sides"]), "distinct_mover_accounts": len(value["accounts"]),
                        "enough_cell_support": len(value["sides"]) >= 100 and len(value["accounts"]) >= 30,
                        "profile_promoted": False} for key, value in sorted(cells.items())]
    return result


def build(request_path, request_sha256, sources_path, sources_sha256, output):
    started, started_utc = time.process_time(), datetime.now(timezone.utc)
    request = read_ref({"path": str(request_path), "sha256": request_sha256})
    sources = read_ref({"path": str(sources_path), "sha256": sources_sha256}, maximum=128 * MIB)
    canonical, split, partitions, examples, groups, exposure, owner_hashes = _inputs(request, sources)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()) and not (output / "progress.json").exists():
        raise ValueError("Existing output is not a resumable expansion build")
    budget = Budget(output, request, request_sha256, sources_sha256, started, started_utc)
    budget.remove_file(output / "summary.json")
    budget.state["test_groups"] = split["plan"]["test_groups"]
    try:
        old_rows = [row for role in ROLES for row in partitions[role]]
        fingerprints = base_fingerprints(canonical, old_rows, request["refs"]["original_audit"])
        fingerprints["sgf_sha256"].update(owner_hashes)
        units, canonical_stop = _canonical_phase(output, sources["candidates"], groups, exposure,
                                                owner_hashes, request, budget)
        _, new_rows, plan = _new_memberships(units, fingerprints, groups, split, exposure)
        deduplicated = sum(row["role"] in ALLOWED for row in new_rows)
        prepared_canonical = _make_canonical(output, canonical, units, request, budget)
        games, example_stop = _example_phase(output, units, new_rows, plan, examples, request, budget)
        for unit in units:
            if unit.get("sgf_sha256"):
                exposure["sgf_hashes"].add(unit["sgf_sha256"])
            if unit.get("sequence_fingerprint"):
                exposure["sequence_hashes"].add(unit["sequence_fingerprint"])
        published, manifest, example_pin = _publish(output, prepared_canonical, split, old_rows, examples,
            new_rows, units, games, plan, groups, exposure, request, request_sha256, sources_sha256, budget)
        budget.write(phase="verification")
        support = support_counts(published / "examples", example_pin, lambda: budget.check(finishing=True))
        budget.save(published / "support.json", support)
        selected = {game["source_url"] for _, game in games[:5]}
        records = {unit["source_url"]: json.loads(_canonical_raw(unit)) for unit in units if unit["source_url"] in selected}
        diagnostics = diagnostic_sample(records, check_budget=lambda: budget.check(finishing=True))
        budget.save(published / "diagnostics.json", diagnostics)
        total_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        if total_bytes > request["limits"]["output_bytes"]:
            raise ValueError("Generated output exceeds the explicit byte limit")
        budget.check(finishing=True)
        reasons = dict(Counter(unit.get("reason") for unit in units if unit["status"] != "canonical"))
        sample_ids = {role: [row["source_url"] for row in sorted(new_rows, key=lambda row: digest(row["source_url"]))
                             if row["role"] == role][:2 if role == "train" else 1] for role in ALLOWED}
        old_games = {role: len(partitions[role]) for role in ALLOWED}
        combined = {role: {"games": old_games[role] + sum(row["role"] == role for row in new_rows),
                           **manifest["counts"][role]} for role in ALLOWED}
        old = {role: {"games": old_games[role], **examples["counts"][role]} for role in ALLOWED}
        added = {role: {key: combined[role][key] - old[role][key] for key in old[role]} for role in ALLOWED}
        artifact_names = {"canonical": "canonical.parquet", "split_manifest": "splits/manifest.json",
                          "examples_manifest": "examples/manifest.json", "profile": "profile.json",
                          "normalizer": "normalizer.json", "coverage": "support.json",
                          "groups": "groups.json", "exposure": "exposure.json", "diagnostics": "diagnostics.json"}
        summary = {"schema": VERSION, "status": "budget_stopped" if canonical_stop or example_stop else "completed",
            "request_sha256": request_sha256, "sources_sha256": sources_sha256,
            "source_revision": request["source_revision"], "builder_sha256": file_hash(__file__),
            "stop_reasons": [reason for reason in (canonical_stop, example_stop) if reason],
            "stop_reason": example_stop or canonical_stop,
            "source_candidates": len(sources["candidates"]), "source_attempted": budget.state.get("canonical_attempted", 0),
            "source_completed": len(units),
            "source_unattempted": len(sources["candidates"]) - budget.state.get("canonical_attempted", 0),
            "source_exclusions": reasons,
            "source_exclusion_reasons": dict(Counter(reason for unit in units if unit["status"] != "canonical"
                                                      for reason in unit.get("exclusions", []))),
            "new_canonical_games": len(new_rows), "new_encoded_games": len(games),
            "new_deduplicated_games": deduplicated,
            "new_source_admitted": len(new_rows),
            "new_replay_validated": sum(unit["replay_validated"] for unit in units if unit["status"] == "canonical"),
            "new_split_exclusions": dict(Counter(reason for row in new_rows for reason in row["exclusions"])),
            "counts": {"old": old, "new": added, "combined": combined}, "review_source_ids": sample_ids,
            "minimum_train_rows": 100_000,
            "minimum_met": manifest["counts"]["train"]["examples"] >= 100_000,
            "training_performed": False, "held_out_targets_decoded": False, "final_targets_decoded": False,
            "base_memberships_unchanged": True, "base_shards_unchanged": True,
            "normalizer_fit_role": "train", "promoted_profiles": [], "output_bytes": total_bytes,
            "artifacts": {name: reference(published / path) for name, path in artifact_names.items()},
            "costs": {"cpu_seconds": budget.cpu,
                      "wall_seconds": (datetime.now(timezone.utc) - datetime.fromisoformat(budget.state["started_utc"])).total_seconds(),
                      "scope": "Build entry through input validation, preparation, verification and artifact hashing; final summary/progress writes excluded.",
                      "workers": 1, "http_requests": 0, "gpu_seconds": 0}}
        budget.state.pop("error", None)
        budget.state.pop("error_type", None)
        budget.remove_file(output / "failure.json")
        budget.end(phase="complete", status=summary["status"], current_source=None)
        summary["costs"].update(cpu_seconds=budget.cpu,
            wall_seconds=(datetime.now(timezone.utc) - datetime.fromisoformat(budget.state["started_utc"])).total_seconds(),
            scope="Build entry through input validation, preparation, verification, artifact hashing and final progress write; excludes final summary formatting/write.")
        for _ in range(3):
            summary["output_bytes"] = budget.used_bytes - budget.file_sizes.get(str(output / "summary.json"), 0) + len(encoded(summary).encode()) + 1
        budget.save(output / "summary.json", summary)
        return summary
    except Exception as error:
        try:
            budget.write(phase="failed", status="failed", error_type=type(error).__name__, error=str(error))
            budget.save(output / "failure.json", {"status": "failed", "error_type": type(error).__name__,
                                                  "error": str(error), "cpu_seconds": budget.cpu})
        except BudgetStop:
            pass
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
    print(json.dumps({key: result[key] for key in ("status", "source_attempted", "new_encoded_games", "minimum_met", "costs")}, indent=2))


if __name__ == "__main__":
    main()
