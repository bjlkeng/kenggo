"""Grow a frozen mixed-komi corpus without reopening its protected targets."""

from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import time
import zipfile
import zlib

import pyarrow as pa
import pyarrow.parquet as pq

from kenggo import data_expansion as expansion
from kenggo import expansion_source
from kenggo.canonical_data import canonicalize, digest, encoded
from kenggo.data_splits import ROLES, SplitPlan, build_splits, read_membership, write_membership
from kenggo.examples import (ExampleConfig, GROWTH_PROFILE_VERSION, KOMI_STUDY_VERSION,
                             examples_for_game, fit_normalizer, load_dataset_profile, write_shard)
from kenggo.go import KOMI_STUDY_RULES
from kenggo.kgs_admission import classify
from kenggo.kgs_eligibility import Exposure, HISTORICAL_BOT_HANDLES, grouping_digest, normalize_handle
from kenggo.population_audit import split_entry
from kenggo.sgf_audit import inspect_sgf


VERSION = "e2_data_growth_v1"
REQUEST_VERSION = "e2_data_growth_request_v1"
SOURCES_VERSION = "e2_data_growth_sources_v1"
INDEX_VERSION = "e2_data_growth_index_v1"
CAP_REASON = "expansion_example_row_limit"
MIB = 1024**2
LIMITS = {"cpu_seconds": 7 * 86400, "wall_seconds": 7 * 86400,
          "output_bytes": 128 * 1024**3, "max_records": 1_000_000,
          "canonical_bytes": 4 * 1024**3}
ARTIFACTS = ("canonical", "split_manifest", "examples_manifest", "profile", "normalizer",
             "base_index", "groups", "exposure")


class PreparationStop(expansion.BudgetStop):
    pass


class _Budget:
    def __init__(self, output, limits, external):
        self.output, self.limits, self.external = output, limits, external
        self.cpu_start, self.wall_start = time.process_time(), time.monotonic()
        self.last_progress = self.wall_start
        self.phase = "inputs"
        self.accounted = {}
        self.bytes = 0
        self.current_source = None
        self.selected_sgf_bytes = self.expanded_zip_bytes = 0

    @property
    def costs(self):
        return {"cpu_seconds": time.process_time() - self.cpu_start,
                "wall_seconds": time.monotonic() - self.wall_start,
                "workers": 1, "gpu_seconds": 0, "http_requests": 0,
                "selected_sgf_bytes": self.selected_sgf_bytes, "expanded_zip_bytes": self.expanded_zip_bytes}

    def check(self):
        self.external()
        costs = self.costs
        if any(costs[key] >= self.limits[key] for key in ("cpu_seconds", "wall_seconds")):
            raise expansion.BudgetStop("growth_build_budget")
        if time.monotonic() - self.last_progress >= 10:
            self.progress()

    def prepare(self):
        self.check()
        for key, reserve in (("cpu_seconds", 300), ("wall_seconds", 1800)):
            if self.costs[key] >= self.limits[key] - min(reserve, self.limits[key] * 0.05):
                raise PreparationStop("growth_preparation_budget")

    def progress(self, **values):
        self.last_progress = time.monotonic()
        expansion.save_json(self.output / "progress.json", {
            "schema": VERSION, "phase": self.phase, "current_source": self.current_source,
            "costs": self.costs, **values})

    def account(self, path):
        name, size = str(path), Path(path).stat().st_size
        self.bytes += size - self.accounted.get(name, 0)
        self.accounted[name] = size
        if self.bytes > self.limits["output_bytes"]:
            raise expansion.BudgetStop("growth_output_bytes")

    def save(self, path, value):
        self.check()
        expansion.save_json(path, value)
        self.account(path)
        return expansion.reference(path)


def _same_ref(left, right):
    return left["sha256"] == right["sha256"] and Path(left["path"]).resolve() == Path(right["path"]).resolve()


def _request(value):
    required = {"version", "parent_summary", "sources", "target_train_rows", "limits", "seed", "fixture_only"}
    if (not isinstance(value, dict) or set(value) != required or value["version"] != REQUEST_VERSION
            or type(value["fixture_only"]) is not bool or not isinstance(value["seed"], str) or not value["seed"]
            or type(value["target_train_rows"]) is not int or value["target_train_rows"] <= 0
            or set(value["limits"]) != set(LIMITS)):
        raise ValueError("Growth requires its complete versioned request")
    for name, maximum in LIMITS.items():
        if type(value["limits"][name]) is not int or not 0 < value["limits"][name] <= maximum:
            raise ValueError("Invalid growth limit: " + name)
    return deepcopy(value)


def _history(index):
    result = {}
    for field in ("rejected", "study_rejections"):
        for row in index.get(field, []):
            if row["source_url"] in result and result[row["source_url"]] != row:
                raise ValueError("Conflicting historical rejection evidence")
            result[row["source_url"]] = row
    return sorted(result.values(), key=lambda row: row["source_url"])


def _inputs(request, budget):
    parent = expansion.read_ref(request["parent_summary"])
    if (parent.get("schema") not in {"e2_komi_expansion_v1", VERSION}
            and not request["fixture_only"]):
        raise ValueError("Production growth requires the mixed-komi corpus or a prior growth round")
    refs = parent["artifacts"]
    if not set(ARTIFACTS) <= set(refs):
        raise ValueError("Parent summary omits a required dataset artifact")
    canonical = expansion.read_ref(refs["canonical"], json_data=False,
                                   maximum=request["limits"]["canonical_bytes"])
    split_path = expansion.read_ref(refs["split_manifest"], json_data=False)
    split, partitions = read_membership(split_path.parent, refs["split_manifest"]["sha256"])
    SplitPlan.from_dict(split["plan"])
    if (split["plan"]["train_before"] != "2025-09-01" or split["plan"]["confirmation_from"] != "2025-11-01"
            or not request["fixture_only"] and split.get("fixture_only") is not False):
        raise ValueError("Growth must retain the frozen production chronology and scope")
    examples = expansion.read_ref(refs["examples_manifest"])
    profile = load_dataset_profile(refs["profile"]["path"], refs["profile"]["sha256"])
    normalizer = expansion.read_ref(refs["normalizer"])
    groups = expansion.read_ref(refs["groups"])
    exposure = expansion.read_ref(refs["exposure"])
    index = expansion.read_ref(refs["base_index"], maximum=512 * MIB)
    if (profile.canonical_sha256 != refs["canonical"]["sha256"]
            or profile.split_sha256 != refs["split_manifest"]["sha256"]
            or examples["canonical_sha256"] != profile.canonical_sha256
            or examples["split_sha256"] != profile.split_sha256
            or examples["dataset_profile_sha256"] != refs["profile"]["sha256"]
            or profile.population_inputs_sha256 != split["provenance"]["population_inputs_sha256"]
            or not request["fixture_only"] and not profile.komi_study
            or index["canonical_sha256"] != profile.canonical_sha256
            or index["split_sha256"] != profile.split_sha256
            or not _same_ref(index["exposure"], refs["exposure"])
            or grouping_digest(groups) != split["plan"]["grouping_sha256"]
            or digest(exposure) != split["exposure_sha256"]
            or digest(split["plan"]) != split["plan_sha256"]
            or normalizer["examples_sha256"] != refs["examples_manifest"]["sha256"]
            or normalizer["fit_role"] != "train"
            or digest(sorted(split["sealed_source_ids"])) != profile.sealed_source_ids_sha256
            or examples["sealed_source_ids"] != sorted(split["sealed_source_ids"])):
        raise ValueError("Parent dataset pins, cohorts, or sealing registry disagree")
    rows = [row for role in ROLES for row in partitions[role]]
    by_id = {row["source_url"]: row for row in rows}
    if len(by_id) != len(rows) or len(rows) > request["limits"]["max_records"]:
        raise ValueError("Parent source IDs are duplicated or exceed the record limit")
    for role in ROLES:
        if (examples["counts"][role]["examples"] != split["policy_actions_by_role"][role]
                or examples["counts"][role]["value_valid"] != split["value_actions_by_role"][role]
                or sum(row["policy_actions"] for row in partitions[role]) != split["policy_actions_by_role"][role]):
            raise ValueError("Parent example and split counts disagree")
        if role in expansion.ALLOWED and parent["counts"]["combined"][role] != {
                "games": len(partitions[role]), **examples["counts"][role]}:
            raise ValueError("Parent summary does not reproduce its dataset counts")
    if normalizer["count"][5] != examples["counts"]["train"]["examples"]:
        raise ValueError("Parent normalizer count differs from its training data")
    if request["target_train_rows"] <= examples["counts"]["train"]["examples"]:
        raise ValueError("Growth target must exceed the preserved training count")
    records = expansion._canonical_index(canonical, rows, budget.check)
    if records != index["records"]:
        raise ValueError("Parent metadata index differs from its opaque canonical records")
    if index.get("schema") not in {expansion.INDEX_VERSION, "e2_komi_data_index_v1", INDEX_VERSION}:
        raise ValueError("Unsupported parent metadata index")
    owner_hashes = {record["sgf_sha256"] for record in records
                    if any(reason.startswith("owner_excluded_") for reason in by_id[record["source_url"]]["exclusions"])}
    sources = expansion.read_ref(request["sources"], maximum=256 * MIB)
    if (sources.get("schema") != SOURCES_VERSION or not _same_ref(sources["parent_summary"], request["parent_summary"])
            or sources["base_grouping_sha256"] != split["plan"]["grouping_sha256"]
            or set(sources["owner_excluded_sgf_sha256"]) != owner_hashes
            or any(sources["groups"].get(handle) != group for handle, group in groups.items())):
        raise ValueError("Growth sources changed their parent, aliases, or owner exclusions")
    if (not request["fixture_only"] and sources.get("fixture_only") is not False
            or sources.get("provenance", {}).get("selection_seed", request["seed"]) != request["seed"]):
        raise ValueError("Growth sources changed the request seed or production scope")
    groups = sources["groups"]
    if any(normalize_handle(handle) != handle or not isinstance(group, str)
           or not group.startswith("kgs-handle:") for handle, group in groups.items()):
        raise ValueError("Invalid source group registry")
    merged = expansion.union_exposure(exposure, sources["exposure"])
    if merged["groups"] & set(split["plan"]["test_groups"]):
        raise ValueError("A final group entered the exposure registry")
    blocked = set(by_id) | {row["source_url"] for row in _history(index)} | set(index.get("processed_source_ids", []))
    ids = []
    for item in sources["candidates"]:
        budget.check()
        row, source = item["archive_row"], item["source"]
        key = row["url"]
        if (key != source["source_url"] or key in blocked or item["provisional_role"] != "train"
                or expansion_source._eligible(row, groups, split["plan"], blocked) != "train"
                or key not in merged["source_ids"]
                or any(group not in merged["groups"] for _, group in expansion_source.cached.participants(row, groups))):
            raise ValueError("Fresh source violates the parent, cohort, or exposure contract")
        ids.append(key)
    if len(set(ids)) != len(ids) or len(ids) + len(rows) > request["limits"]["max_records"]:
        raise ValueError("Fresh IDs are duplicated or exceed the record allowance")
    return parent, canonical, split, partitions, examples, index, groups, merged, owner_hashes, sources


def _promotion_ids(canonical, rows, split, sources):
    reserved = set(split["plan"]["validation_groups"] + split["plan"]["test_groups"])
    eligible = {row["source_url"] for row in rows
                if row["role"] == "excluded" and row["exclusions"] == [CAP_REASON]
                and row["date"] < split["plan"]["train_before"] and row["date"].startswith("2025-")
                and not set(row["groups"]) & reserved and row["source_url"] not in split["sealed_source_ids"]}
    if "promotion_source_ids" in sources:
        selected = sources["promotion_source_ids"]
        if len(set(selected)) != len(selected) or not set(selected) <= eligible:
            raise ValueError("Only capacity-limited training games may be promoted")
        eligible = set(selected)
    headers = pq.read_table(canonical, columns=["source_url"])["source_url"].to_pylist()
    return [key for key in headers if key in eligible]


def _promote(record, old, plan, groups, exposure):
    entry = split_entry(record, grouping_digest(groups))
    regenerated = build_splits([entry], SplitPlan.from_dict(plan), exposed_groups=exposure["groups"], group_registry=groups)[0]
    fields = ("source_url", "canonical_sha256", "date", "groups", "actions", "actions_by_color", "identity_valid")
    if (old["exclusions"] != [CAP_REASON] or regenerated["role"] != "train"
            or regenerated["exclusions"] or any(regenerated[key] != old[key] for key in fields)
            or record.get("sealed_final") is not False or record.get("replay_validated") is not True):
        raise ValueError("Capacity promotion no longer matches its admitted whole game")
    return {**old, **{key: regenerated[key] for key in
                     ("role", "exclusions", "policy_actions", "value_actions", "adaptation_colors", "known_persona_colors")}}


def _raw_records(path, selected, check):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
        check()
        for row in batch.to_pylist():
            if row["source_url"] in selected:
                if sha256(row["canonical_json"].encode()).hexdigest() != row["canonical_sha256"]:
                    raise ValueError("Selected canonical bytes changed")
                yield row["source_url"], row["canonical_json"]


def _encode(record, row, plan, config, directory, budget):
    budget.prepare()
    generated = list(examples_for_game(record, row, plan, config, komi_study=True, check_budget=budget.prepare))
    directory.mkdir()
    artifacts = []
    for offset in range(0, len(generated), config.shard_size):
        budget.prepare()
        artifact = write_shard(directory, "train", offset // config.shard_size,
                               generated[offset:offset + config.shard_size], config.history)
        artifacts.append(artifact)
        for field in ("arrays", "metadata"):
            budget.account(directory / artifact[field])
    counts = {"examples": len(generated), "value_valid": sum(item[1]["value_valid"] for item in generated),
              "duration_valid": sum(item[1]["duration_valid"] for item in generated),
              "passes": sum(item[1]["action"] == 361 for item in generated),
              "hidden_mover_rank": sum(item[2]["rank_hidden"][0] for item in generated),
              "hidden_opponent_rank": sum(item[2]["rank_hidden"][1] for item in generated)}
    return {"directory": str(directory), "source_url": row["source_url"], "artifacts": artifacts, "counts": counts}


def _fresh_unit(item, groups, exposure, owner_hashes, split, archives, stack, budget):
    row, source = item["archive_row"], dict(item["source"])
    archive_key = (source["archive_path"], source["archive_sha256"])
    if archive_key not in archives:
        path = expansion.read_ref({"path": archive_key[0], "sha256": archive_key[1]}, json_data=False, maximum=64 * MIB)
        archive = stack.enter_context(zipfile.ZipFile(path))
        expanded = sum(info.file_size for info in archive.infolist())
        if expanded > budget.limits["output_bytes"]:
            raise ValueError("Declared expanded archive exceeds the build byte allowance")
        budget.expanded_zip_bytes += expanded
        archives[archive_key] = archive
    archive = archives[archive_key]
    matches = [info for info in archive.infolist()
               if expansion.member_key(info.filename) == expansion.source_key(source["source_url"])]
    if len(matches) != 1 or matches[0].filename != source["zip_member"] or matches[0].file_size > MIB:
        raise ValueError("Fresh ZIP member is missing, ambiguous, or oversized")
    budget.selected_sgf_bytes += matches[0].file_size
    try:
        data = archive.read(matches[0])
    except (zipfile.BadZipFile, RuntimeError, EOFError, zlib.error):
        return {"source_url": source["source_url"], "status": "source_excluded",
                "reason": "selected_member_read_error", "selected_bytes": matches[0].file_size}
    if len(data) != matches[0].file_size:
        raise ValueError("Fresh ZIP member size changed")
    source["sgf_sha256"] = sha256(data).hexdigest()
    result = {"source_url": source["source_url"], "sgf_sha256": source["sgf_sha256"], "selected_bytes": len(data)}
    if source["sgf_sha256"] in owner_hashes:
        return {**result, "status": "source_excluded", "reason": "owner_excluded_sgf_bytes"}
    try:
        if expansion._header_for_fingerprints(data, row, groups,
                set(split["plan"]["test_groups"] + split["plan"]["validation_groups"])):
            raise ValueError("rank mismatch")
    except (ValueError, TypeError, KeyError, UnicodeError, LookupError):
        return {**result, "status": "source_excluded", "reason": "source_header_rejected"}
    audit = inspect_sgf(data, row, HISTORICAL_BOT_HANDLES, komi_study=True)
    audit["source_url"] = source["source_url"]
    gate = classify(audit, row, groups, Exposure(grouping_digest(groups), **{
        name: frozenset(items) for name, items in exposure.items()}), komi_study=True)
    record, cost = canonicalize(data, source, row, komi_study=True, check_budget=budget.prepare)
    result.update(sequence_fingerprint=audit.get("sequence_fingerprint"),
                  trajectory_sha256=record["trajectory_sha256"], adjudication_cost=cost)
    if not gate["training_eligible"]:
        return {**result, "status": "source_excluded", "reason": "source_policy", "exclusions": gate["eligibility_reasons"]}
    record.update(eligibility=gate, sealed_final=False)
    return {**result, "status": "canonical", "canonical_json": encoded(record),
            "entry": split_entry(record, grouping_digest(groups)), "replay_validated": record["replay_validated"]}


def _link(source, target, pin, budget):
    budget.check()
    if expansion.file_hash(source) != pin:
        raise ValueError("Preserved artifact changed")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(Path(source).resolve())
    budget.account(target)


def _publish(output, request, parent, canonical, split, old_rows, rows, examples, index,
             groups, exposure, units, games, promotions, plan, budget):
    directory = output / "dataset"
    directory.mkdir()
    canonical_path = directory / "canonical.parquet"
    canonical_units = [unit for unit in units if unit["status"] == "canonical"]
    if not canonical_units:
        _link(canonical, canonical_path, parent["artifacts"]["canonical"]["sha256"], budget)
    else:
        schema = pq.ParquetFile(canonical).schema_arrow
        with pq.ParquetWriter(canonical_path, schema, compression="zstd") as writer:
            for batch in pq.ParquetFile(canonical).iter_batches(batch_size=32):
                budget.check()
                writer.write_batch(batch)
                budget.account(canonical_path)
            pending = []
            for unit in canonical_units:
                budget.check()
                raw = expansion.read_ref(unit["unit_ref"], maximum=256 * MIB)["canonical_json"]
                pending.append({"source_url": unit["source_url"],
                    "canonical_sha256": sha256(raw.encode()).hexdigest(), "canonical_json": raw})
                if len(pending) == 32:
                    writer.write_table(pa.Table.from_pylist(pending, schema=schema))
                    budget.account(canonical_path)
                    pending = []
            if pending:
                writer.write_table(pa.Table.from_pylist(pending, schema=schema))
        budget.account(canonical_path)
        if canonical_path.stat().st_size > request["limits"]["canonical_bytes"]:
            raise ValueError("Growth canonical output exceeds its explicit byte allowance")
    canonical_ref = expansion.reference(canonical_path)
    request_ref = expansion.reference(output / "request.json")
    derived = deepcopy(split)
    derived.pop("membership_sha256", None)
    derived.update(plan=plan, plan_sha256=digest(plan),
        canonical_input_sha256=digest({"parent": split["canonical_input_sha256"], "request": request_ref["sha256"]}),
        exposure_sha256=digest({key: sorted(value) for key, value in exposure.items()}))
    derived["provenance"] = {**derived["provenance"], "population_inputs_sha256": request_ref["sha256"],
        "growth_request": request_ref, "parent_summary": request["parent_summary"],
        "parent_split": parent["artifacts"]["split_manifest"], "promoted_source_ids": promotions}
    derived["roles"] = {role: sum(row["role"] == role for row in rows) for role in ROLES}
    for output_key, field in (("policy_actions_by_role", "policy_actions"), ("value_actions_by_role", "value_actions")):
        derived[output_key] = {role: sum(row[field] for row in rows if row["role"] == role) for role in ROLES}
    derived["exclusion_counts"] = dict(sorted(Counter(reason for row in rows for reason in row["exclusions"]).items()))
    derived["adaptation_sides"] = sum(len(row["adaptation_colors"]) for row in rows)
    split_pin = write_membership(directory / "splits", rows, derived)
    for path in (directory / "splits").iterdir():
        budget.account(path)
    resulting, parts = read_membership(directory / "splits", split_pin)
    by_id = {row["source_url"]: row for part in parts.values() for row in part}
    promoted = set(promotions)
    if any(by_id[row["source_url"]] != row for row in old_rows if row["source_url"] not in promoted):
        raise ValueError("An unapproved old membership changed")
    for role in ROLES:
        if role not in {"train", "excluded"} and resulting["membership_sha256"][role + ".parquet"] != split["membership_sha256"][role + ".parquet"]:
            raise ValueError("A development or protected membership file changed")
    profile_ref = budget.save(directory / "profile.json", {
        "version": GROWTH_PROFILE_VERSION, "canonical_sha256": canonical_ref["sha256"], "split_sha256": split_pin,
        "population_inputs_sha256": request_ref["sha256"], "sealed_source_ids_sha256": digest(sorted(split["sealed_source_ids"])),
        "limits": {"max_records": request["limits"]["max_records"], "max_examples_per_role": request["target_train_rows"],
                   **{name: request["limits"][name] for name in ("cpu_seconds", "wall_seconds", "canonical_bytes")}}})
    manifest = deepcopy(examples)
    example_dir = directory / "examples"
    example_dir.mkdir()
    source_dir = Path(parent["artifacts"]["examples_manifest"]["path"]).parent
    for role in ROLES:
        for number, item in enumerate(examples["artifacts"][role]):
            for kind, suffix in (("arrays", "npz"), ("metadata", "parquet")):
                if item[kind] != f"{role}/{number:05d}.{suffix}":
                    raise ValueError("Unexpected parent shard path")
                _link(source_dir / item[kind], example_dir / item[kind], item[kind + "_sha256"], budget)
    for game in games:
        for artifact in game["artifacts"]:
            number = len(manifest["artifacts"]["train"])
            added = dict(artifact)
            for kind, suffix in (("arrays", "npz"), ("metadata", "parquet")):
                added[kind] = f"train/{number:05d}.{suffix}"
                _link(Path(game["directory"]) / artifact[kind], example_dir / added[kind], artifact[kind + "_sha256"], budget)
            manifest["artifacts"]["train"].append(added)
        for key, value in game["counts"].items():
            manifest["counts"]["train"][key] += value
    for role in ROLES:
        if (manifest["counts"][role]["examples"] != resulting["policy_actions_by_role"][role]
                or manifest["counts"][role]["value_valid"] != resulting["value_actions_by_role"][role]):
            raise ValueError("Growth examples disagree with complete-game memberships")
    manifest.update(version=KOMI_STUDY_VERSION, rules=KOMI_STUDY_RULES,
        canonical_sha256=canonical_ref["sha256"], split_sha256=split_pin, dataset_profile_sha256=profile_ref["sha256"],
        source_roles=resulting["roles"], exclusions=resulting["exclusion_counts"],
        derivation={"growth_request": request_ref, "parent_summary": request["parent_summary"],
                    "parent_examples": parent["artifacts"]["examples_manifest"], "sources": request["sources"]},
        code_sha256={name: expansion.file_hash(Path(__file__).with_name(name)) for name in examples["code_sha256"]},
        uv_lock_sha256=expansion.file_hash("uv.lock"))
    example_ref = budget.save(example_dir / "manifest.json", manifest)
    normalizer = fit_normalizer(example_dir, example_ref["sha256"], komi_study=True, check_budget=budget.check)
    if normalizer["count"][5] != manifest["counts"]["train"]["examples"]:
        raise ValueError("Growth normalizer did not fit every training row")
    normalizer_ref = budget.save(directory / "normalizer.json", normalizer)
    groups_ref = budget.save(directory / "groups.json", groups)
    exposure_ref = budget.save(directory / "exposure.json", {key: sorted(value) for key, value in exposure.items()})
    metadata = expansion._canonical_index(canonical_path, rows, budget.check)
    processed = sorted(set(index.get("processed_source_ids", [])) | {unit["source_url"] for unit in units})
    grown_index = {"schema": INDEX_VERSION, "canonical_sha256": canonical_ref["sha256"], "split_sha256": split_pin,
        "records": metadata, "rejected": expansion._rejections(units, _history(index)),
        "exposure": exposure_ref, "processed_source_ids": processed,
        "provenance": {"parent_index": parent["artifacts"]["base_index"], "parent_summary": request["parent_summary"],
                       "request": request_ref, "sources": request["sources"], "promoted_source_ids": promotions,
                       "work_units": [unit["unit_ref"] for unit in units]}}
    index_ref = budget.save(directory / "base_index.json", grown_index)
    return {"canonical": canonical_ref, "split_manifest": expansion.reference(directory / "splits/manifest.json"),
            "examples_manifest": example_ref, "profile": profile_ref, "normalizer": normalizer_ref,
            "base_index": index_ref, "groups": groups_ref, "exposure": exposure_ref}, manifest, parts


def build_growth_round(request, output, *, check_budget=lambda: None):
    """Build one new immutable dataset; existing or failed output directories cannot be reused."""
    request = _request(request)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    budget = _Budget(output, request["limits"], check_budget)
    try:
        request_ref = budget.save(output / "request.json", request)
        budget.progress(request=request_ref, status="running")
        parent, canonical, split, partitions, examples, index, groups, exposure, owners, sources = _inputs(request, budget)
        old_rows = [row for role in ROLES for row in partitions[role]]
        old_by_id = {row["source_url"]: row for row in old_rows}
        plan = {**split["plan"], "grouping_sha256": grouping_digest(groups)}
        promotion_ids = _promotion_ids(canonical, old_rows, split, sources)
        budget.save(output / "promotion_candidates.json", {"parent_summary": request["parent_summary"], "source_ids": promotion_ids})
        work = output / "work"
        work.mkdir()
        config = ExampleConfig(**examples["config"])
        count = examples["counts"]["train"]["examples"]
        initial_count = count
        games, promotions, replacements, units = [], [], {}, []
        stop = None
        budget.phase = "promotions"
        for key, raw in _raw_records(canonical, set(promotion_ids), budget.check):
            if count + old_by_id[key]["actions"] > request["target_train_rows"]:
                continue
            try:
                budget.prepare()
                record = json.loads(raw)
                row = _promote(record, old_by_id[key], plan, groups, exposure)
                game = _encode(record, row, plan, config, work / digest(key), budget)
            except PreparationStop as error:
                stop = str(error)
                break
            games.append(game)
            count += game["counts"]["examples"]
            promotions.append(key)
            replacements[key] = row
        fingerprints = expansion._fingerprints(index["records"])
        fingerprints["sgf_sha256"].update(owners)
        new_rows = []
        archives = {}
        budget.phase = "fresh_canonical"
        with ExitStack() as stack:
            for number, item in enumerate(sources["candidates"]):
                if stop or count >= request["target_train_rows"]:
                    break
                try:
                    budget.prepare()
                    budget.current_source = item["source"]["source_url"]
                    unit = _fresh_unit(item, groups, exposure, owners, split, archives, stack, budget)
                except PreparationStop as error:
                    stop = str(error)
                    if budget.current_source is None:
                        break
                    unit = {"source_url": budget.current_source, "status": "source_excluded",
                            "reason": "growth_preparation_interrupted"}
                pin = budget.save(work / f"source-{number:06d}.json", unit)
                unit.pop("canonical_json", None)
                unit["unit_ref"] = pin
                units.append(unit)
                budget.current_source = None
                for field, exposed in (("sgf_sha256", "sgf_hashes"), ("sequence_fingerprint", "sequence_hashes")):
                    if unit.get(field):
                        exposure[exposed].add(unit[field])
                if len(units) % 128 == 0 or number == len(sources["candidates"]) - 1:
                    _, new_rows, _ = expansion._new_memberships(units, fingerprints, groups, split, exposure, _history(index))
                    available = request["target_train_rows"] - count
                    maximum = 0
                    for row in new_rows:
                        if row["role"] == "train":
                            maximum = max(maximum, row["actions"])
                            if row["actions"] <= available:
                                available -= row["actions"]
                    if maximum and available < maximum:
                        break
        if units:
            _, new_rows, _ = expansion._new_memberships(units, fingerprints, groups, split, exposure, _history(index))
        by_unit = {unit["source_url"]: unit for unit in units if unit["status"] == "canonical"}
        budget.phase = "fresh_examples"
        for row in new_rows:
            if row["role"] != "train":
                continue
            reason = CAP_REASON if count + row["actions"] > request["target_train_rows"] else stop
            if reason is None:
                try:
                    record = json.loads(expansion.read_ref(by_unit[row["source_url"]]["unit_ref"], maximum=256 * MIB)["canonical_json"])
                    game = _encode(record, row, plan, config, work / digest(row["source_url"]), budget)
                    games.append(game)
                    count += game["counts"]["examples"]
                except PreparationStop as error:
                    reason = stop = str(error)
            if reason:
                row.update(role="excluded", policy_actions=0, value_actions=0, adaptation_colors=[], known_persona_colors=[],
                           exclusions=sorted(set(row["exclusions"] + [reason])))
        budget.phase = "publication"
        rows = [replacements.get(row["source_url"], row) for row in old_rows] + new_rows
        artifacts, manifest, parts = _publish(output, request, parent, canonical, split, old_rows, rows,
            examples, index, groups, exposure, units, games, promotions, plan, budget)
        combined = {role: {"games": len(parts[role]), **manifest["counts"][role]} for role in expansion.ALLOWED}
        old = {role: {"games": len(partitions[role]), **examples["counts"][role]} for role in expansion.ALLOWED}
        added = {role: {key: combined[role][key] - old[role][key] for key in old[role]} for role in expansion.ALLOWED}
        interrupted = [unit["source_url"] for unit in units if unit.get("reason") == "growth_preparation_interrupted"]
        maximum_game = max((row["actions"] for row in rows if row["role"] == "train"), default=0)
        summary = {"schema": VERSION, "status": "budget_stopped" if stop else "completed",
            "parent_summary": request["parent_summary"], "request": request_ref, "sources": request["sources"],
            "request_sha256": request_ref["sha256"], "sources_sha256": request["sources"]["sha256"],
            "target_train_rows": request["target_train_rows"], "actual_train_rows": count,
            "added_train_rows": count - initial_count, "target_shortfall": request["target_train_rows"] - count,
            "whole_game_tolerance_rows": maximum_game,
            "target_within_one_game": request["target_train_rows"] - count <= maximum_game,
            "stop_reason": stop or ("target_reached" if count == request["target_train_rows"] else "whole_game_limit_or_sources_exhausted"),
            "promotion_candidates": len(promotion_ids), "promoted_source_ids": promotions,
            "source_candidates": len(sources["candidates"]), "source_completed": len(units) - len(interrupted),
            "source_attempted": len(units),
            "source_interrupted_ids": interrupted,
            "source_unattempted": len(sources["candidates"]) - len(units),
            "unprocessed_source_ids": [item["source"]["source_url"] for item in sources["candidates"][len(units):]],
            "source_exclusions": dict(Counter(unit["reason"] for unit in units if unit["status"] == "source_excluded")),
            "new_split_exclusions": dict(Counter(reason for row in new_rows for reason in row["exclusions"])),
            "new_canonical_games": len(new_rows), "new_encoded_games": len(games),
            "fresh_encoded_games": len(games) - len(promotions),
            "new_replay_validated": sum(unit.get("replay_validated", False) for unit in units),
            "counts": {"old": old, "new": added, "combined": combined}, "artifacts": artifacts,
            "old_training_shards_preserved": True, "protected_memberships_unchanged": True,
            "normalizer_fit_role": "train", "training_performed": False,
            "held_out_targets_decoded": False, "final_targets_decoded": False,
            "code_sha256": {name: expansion.file_hash(name) for name in
                            (*expansion.CODE_PATHS, "src/kenggo/data_growth.py")}, "costs": budget.costs}
        budget.check()
        budget.phase = "complete"
        budget.progress(status=summary["status"], actual_train_rows=count)
        summary["costs"] = {**budget.costs, "scope": "Build entry through publication and final progress; final summary write excluded."}
        budget.save(output / "summary.json", summary)
        return summary
    except Exception as error:
        budget.progress(status="failed", error_type=type(error).__name__, error=str(error))
        raise
