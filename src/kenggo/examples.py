"""Verified canonical games to bounded, hash-bound arrays and separate provenance."""

from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from kenggo.canonical_data import VERSION as CANONICAL_VERSION, digest, encoded
from kenggo.data_splits import ROLES, VERSION as SPLIT_VERSION, SplitPlan, read_membership
from kenggo.encoding import (Context, INPUT_KEYS, NORMALIZED, SCALARS, VERSION as ENCODING,
                             array_schema, encode)
from kenggo.go import Game, RULES
from kenggo.kgs_source import RANKS
from kenggo.sgf_audit import japanese_overtime


VERSION = "supervised_examples_v1"
CANONICAL_SHA256 = "b3ec35e3ff97338f440e431e15e4ede8060158489903159108dede6f73ae1db0"
SPLIT_SHA256 = "db064735e9a5f58525b67e1d3bc541be71c526cf98bbe9c178a3d59d38696a3d"
PREFIXES = {"validation_prefix", "test_prefix"}
UNUSED = {"excluded", "embargo"}
FINAL_ROLES = {"test_prefix", "final", "embargo"}
RECORDED_ROLES = frozenset({"train", "population_development", "validation_development"})
CODE_FILES = ("encoding.py", "examples.py", "examples_audit.py", "go.py", "rules_replay.py",
              "data_splits.py", "canonical_data.py", "kgs_source.py", "sgf_audit.py")
LABEL_SCHEMA = {"action": ("int16", ()), "policy_valid": ("bool", ()),
                "value": ("float32", ()), "value_valid": ("bool", ()),
                "duration": ("float32", ()), "duration_valid": ("bool", ())}
META_SCHEMA = pa.schema([
    *[(name, pa.string()) for name in ("source_url", "canonical_sha256", "sgf_sha256", "state_sha256",
        "role", "mover", "source_rank_mover", "source_rank_opponent", "group_mover", "group_opponent",
        "persona_group", "persona_mode", "row_key")],
    ("ply", pa.int32()), ("sgf_node", pa.int32()), ("rank_hidden", pa.list_(pa.bool_())),
    *[(name, pa.bool_()) for name in ("cohort_target", "personalization_target_valid", "identity_valid",
                                     "adaptation_allowed")]])


@dataclass(frozen=True)
class ExampleConfig:
    history: int = 0
    shard_size: int = 256
    rank_seed: int = 20261111
    hide_mover_per_mille: int = 100
    hide_opponent_per_mille: int = 200

    def __post_init__(self):
        array_schema(self.history)
        if type(self.shard_size) is not int or not 1 <= self.shard_size <= 4096:
            raise ValueError("Shard size must be between 1 and 4096")
        if type(self.rank_seed) is not int or not 0 <= self.rank_seed < 2**64:
            raise ValueError("Rank masking needs an unsigned 64-bit seed")
        if any(type(n) is not int or not 0 <= n <= 1000 for n in
               (self.hide_mover_per_mille, self.hide_opponent_per_mille)):
            raise ValueError("Rank mask probabilities must be integer thousandths")


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read_pinned_json(path, expected_sha256, *, max_bytes=64 * 1024):
    path = Path(path)
    if (not isinstance(expected_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
            or path.stat().st_size > max_bytes or file_hash(path) != expected_sha256):
        raise ValueError("JSON input differs from its independently supplied pin")
    return json.loads(path.read_text())


@dataclass(frozen=True)
class DatasetProfile:
    sha256: str
    canonical_sha256: str
    split_sha256: str
    population_inputs_sha256: str
    sealed_source_ids_sha256: str
    max_records: int
    max_examples_per_role: int
    cpu_seconds: int
    wall_seconds: int


def load_dataset_profile(path, expected_sha256):
    value = read_pinned_json(path, expected_sha256)
    hashes = {"canonical_sha256", "split_sha256", "population_inputs_sha256", "sealed_source_ids_sha256"}
    limits = {"max_records", "max_examples_per_role", "cpu_seconds", "wall_seconds"}
    if (set(value) != {"version", "limits"} | hashes or value["version"] != "population_example_profile_v1"
            or set(value["limits"]) != limits
            or any(not isinstance(value[k], str) or re.fullmatch(r"[0-9a-f]{64}", value[k]) is None for k in hashes)
            or any(type(value["limits"][k]) is not int or value["limits"][k] <= 0 for k in limits)):
        raise ValueError("Invalid production dataset profile")
    return DatasetProfile(expected_sha256, **{k: value[k] for k in hashes}, **value["limits"])


@dataclass(frozen=True)
class FinalRelease:
    sha256: str
    canonical_sha256: str
    split_sha256: str
    examples_sha256: str


def load_final_release(path, expected_sha256):
    value = read_pinned_json(path, expected_sha256)
    hashes = {"canonical_sha256", "split_sha256", "examples_sha256"}
    if (set(value) != {"version", "stage", "selection_frozen"} | hashes
            or value["version"] != "final_release_v1" or value["stage"] != "E7"
            or value["selection_frozen"] is not True
            or any(not isinstance(value[k], str) or re.fullmatch(r"[0-9a-f]{64}", value[k]) is None for k in hashes)):
        raise ValueError("Final access requires an explicit E7 release authorization")
    return FinalRelease(expected_sha256, **{k: value[k] for k in hashes})


def require_final_release(release, canonical_sha256, split_sha256, examples_sha256=None):
    if (not isinstance(release, FinalRelease)
            or (release.canonical_sha256, release.split_sha256) != (canonical_sha256, split_sha256)
            or examples_sha256 is not None and release.examples_sha256 != examples_sha256):
        raise PermissionError("Sealed final data requires a matching E7 release authorization")


def sealed_sources(manifest):
    sources = manifest.get("sealed_source_ids", [])
    if (not isinstance(sources, list) or any(not isinstance(key, str) or not key for key in sources)
            or sources != sorted(set(sources))):
        raise ValueError("Invalid sealed source registry")
    return frozenset(sources)


def source_context(record):
    metadata = record["metadata"]
    overtime = japanese_overtime(metadata["OT"])
    if (metadata["BR"] not in RANKS or metadata["WR"] not in RANKS
            or overtime is None
            or metadata["SZ"] != 19 or metadata["RU"] != "Japanese" or metadata["KM"] != 6.5):
        raise ValueError("Source input context is outside the v1 contract")
    return Context(metadata["BR"], metadata["WR"], (metadata["TM"], overtime[1], overtime[0]))


def replay_record(record):
    if (record["version"] != CANONICAL_VERSION or record["rules_contract"] != RULES
            or record["replay_validated"] is not True or record["structural_exclusions"]
            or record["duration_valid"] is not False or record["dynamic_clock_input_valid"] is not False):
        raise ValueError("Canonical record is not a supported replay under the v1 masks")
    source_context(record)
    game = Game.new()
    for move in record["actions"]:
        game = game.play(move["color"], move["action"])
    if (game.snapshot != record["replay_snapshot"] or game.status != record["replay_end_status"]
            or game.pending_cap != record["replay_pending_cap"]):
        raise ValueError("Canonical replay evidence changed")
    return game


def _load_corpus(canonical_path, canonical_sha256, split_dir, split_sha256, *, fixture_only=False,
                 profile=None, decode_sealed=True, role=None, source_ids=None):
    selected = None
    if role is not None or source_ids is not None:
        if role not in RECORDED_ROLES:
            raise ValueError("Recorded selection requires a training-debug or development role")
        if isinstance(source_ids, (str, bytes)) or source_ids is None:
            raise ValueError("Recorded selection requires explicit source IDs")
        source_ids = tuple(source_ids)
        if (not source_ids or any(not isinstance(key, str) or not key for key in source_ids)
                or len(set(source_ids)) != len(source_ids)):
            raise ValueError("Recorded selection requires nonempty, unique source IDs")
        selected = frozenset(source_ids)
    expected = ((CANONICAL_SHA256, SPLIT_SHA256) if profile is None else
                (profile.canonical_sha256, profile.split_sha256))
    if fixture_only and profile is not None:
        raise ValueError("Synthetic fixtures cannot use a production profile")
    if not fixture_only and (canonical_sha256, split_sha256) != expected:
        raise ValueError("Unreviewed production source; revise the source policy before admission")
    canonical_path = Path(canonical_path)
    if canonical_path.stat().st_size > 64 * 1024**2 or file_hash(canonical_path) != canonical_sha256:
        raise ValueError("Canonical file differs from its independent pin")
    manifest, partitions = read_membership(split_dir, split_sha256)
    if (manifest["version"] != SPLIT_VERSION or manifest["fixture_only"] is not fixture_only
            or set(manifest["roles"]) != set(ROLES)
            or digest(manifest["plan"]) != manifest["plan_sha256"]):
        raise ValueError("Split version or fixture scope mismatch")
    SplitPlan.from_dict(manifest["plan"])
    sealed = sealed_sources(manifest)
    if profile is not None and (digest(sorted(sealed)) != profile.sealed_source_ids_sha256
            or manifest["provenance"].get("population_inputs_sha256") != profile.population_inputs_sha256):
        raise ValueError("Dataset profile differs from its source or sealing provenance")
    headers = pq.read_table(canonical_path, columns=["source_url", "canonical_sha256"])
    if headers.num_rows > (1000 if profile is None else profile.max_records):
        raise ValueError("Canonical record budget exceeded")
    hashes = {row["source_url"]: row["canonical_sha256"] for row in headers.to_pylist()}
    if (len(hashes) != headers.num_rows or not sealed <= set(hashes)
            or any(not isinstance(key, str) or not key or not isinstance(value, str)
                   or re.fullmatch(r"[0-9a-f]{64}", value) is None for key, value in hashes.items())):
        raise ValueError("Canonical IDs are duplicated or omit a sealed source")
    seen, memberships = set(), {}
    for partition_role, rows in partitions.items():
        if len(rows) != manifest["roles"][partition_role]:
            raise ValueError("Split role count mismatch")
        for row in rows:
            key = row["source_url"]
            if key in seen or key not in hashes or row["role"] != partition_role:
                raise ValueError("Source IDs must have exactly one matching role")
            seen.add(key)
            memberships[key] = row
            if key in sealed and partition_role not in FINAL_ROLES | {"excluded"}:
                raise ValueError("Sealed final source entered a development or training role")
            if not fixture_only and partition_role in FINAL_ROLES and key not in sealed:
                raise ValueError("Final-role source is absent from the sealed registry")
            if row["canonical_sha256"] != hashes[key]:
                raise ValueError("Membership hash differs from its canonical record")
            if (any(type(row[name]) is not int or row[name] < 0 for name in
                    ("actions", "policy_actions", "value_actions"))
                    or not 0 <= row["value_actions"] <= row["policy_actions"] <= row["actions"]
                    or len(row["groups"]) != 2 or len(row["identity_valid"]) != 2
                    or any(type(flag) is not bool for flag in row["identity_valid"])
                    or len(row["actions_by_color"]) != 2
                    or any(type(count) is not int or count < 0 for count in row["actions_by_color"])
                    or sum(row["actions_by_color"]) != row["actions"]):
                raise ValueError("Invalid split membership counts or identities")
        if (sum(row["policy_actions"] for row in rows) != manifest["policy_actions_by_role"][partition_role]
                or sum(row["value_actions"] for row in rows) != manifest["value_actions_by_role"][partition_role]):
            raise ValueError("Split target count mismatch")
    if seen != set(hashes):
        raise ValueError("Canonical games are missing from the split")
    if selected is not None:
        if (not selected <= set(hashes) or any(key in sealed or memberships[key]["role"] != role
                or memberships[key]["exclusions"] for key in selected)):
            raise ValueError("Selected sources must be unsealed, nonexcluded members of the exact role")
        if any(row["policy_actions"] != row["actions"] or row["value_actions"] not in (0, row["actions"])
               or any(not isinstance(group, str) or not group for group in row["groups"])
               or len(set(row["groups"])) != 2 for row in (memberships[key] for key in selected)):
            raise ValueError("Selected complete games require full policy counts and distinct account groups")
        filters = [("source_url", "in", sorted(selected))]
    else:
        filters = [("source_url", "not in", sorted(sealed))] if sealed and not decode_sealed else None
    expected_records = selected if selected is not None else set(hashes) - (sealed if not decode_sealed else set())
    table = pq.read_table(canonical_path, columns=["source_url", "canonical_sha256", "canonical_json"], filters=filters)
    records, byte_count = {}, 0
    for row in table.to_pylist():
        key = row["source_url"]
        if key not in expected_records or row["canonical_sha256"] != hashes[key]:
            raise ValueError("Canonical payload is outside the authorized headers")
        byte_count += len(row["canonical_json"].encode())
        if byte_count > 256 * 1024**2:
            raise ValueError("Canonical JSON budget exceeded")
        record = json.loads(row["canonical_json"])
        if (key in records or record["source"]["source_url"] != key
                or digest(record) != row["canonical_sha256"]):
            raise ValueError("Canonical source ID or record digest mismatch")
        if (profile is not None and type(record.get("sealed_final")) is not bool
                or bool(record.get("sealed_final", False)) != (key in sealed)):
            raise ValueError("Canonical sealing marker differs from the frozen registry")
        records[key] = record
    if set(records) != expected_records:
        raise ValueError("Decoded canonical sources differ from the authorized selection")
    for partition_role, rows in partitions.items():
        for row in rows:
            key = row["source_url"]
            if key not in records:
                continue
            record = records[key]
            gate = record["eligibility"]
            if (digest(record) != row["canonical_sha256"] or len(record["actions"]) != row["actions"]
                    or row["groups"] != [gate["black_group"], gate["white_group"]]
                    or row["identity_valid"] != [gate["black_identity_valid"], gate["white_identity_valid"]]
                    or row["actions_by_color"] != [sum(m["color"] == c for m in record["actions"]) for c in ("B", "W")]):
                raise ValueError("Split membership differs from canonical evidence")
    if selected is not None:
        partitions = {name: [row for row in rows if row["source_url"] in selected]
                      for name, rows in partitions.items()}
    return records, manifest, partitions


def load_corpus(canonical_path, canonical_sha256, split_dir, split_sha256, *, fixture_only=False,
                profile=None, include_sealed=False, final_release=None, role=None, source_ids=None):
    """Load unsealed records or an explicit debug/development source selection."""
    if include_sealed:
        if role is not None or source_ids is not None:
            raise ValueError("Recorded selections cannot request a final release")
        require_final_release(final_release, canonical_sha256, split_sha256)
    records, manifest, partitions = _load_corpus(canonical_path, canonical_sha256, split_dir, split_sha256,
                                               fixture_only=fixture_only, profile=profile, decode_sealed=include_sealed,
                                               role=role, source_ids=source_ids)
    if not include_sealed:
        sealed = sealed_sources(manifest)
        records = {key: record for key, record in records.items() if key not in sealed}
        partitions = {role: [row for row in rows if row["source_url"] not in sealed]
                      for role, rows in partitions.items()}
    return records, manifest, partitions


def rank_hidden(source_url, ply, config):
    return [int(digest([config.rank_seed, source_url, ply, side]), 16) * 1000 < probability * 2**256
            for side, probability in enumerate((config.hide_mover_per_mille, config.hide_opponent_per_mille))]


def examples_for_game(record, membership, plan, config):
    if record.get("sealed_final", False):
        raise PermissionError("Sealed records are available only to the mechanical shard builder")
    yield from _examples_for_game(record, membership, plan, config)


def _examples_for_game(record, membership, plan, config):
    role = membership["role"]
    if role in UNUSED:
        return
    if (role not in ROLES or record["eligibility"]["training_eligible"] is not True
            or membership["exclusions"] or digest(record) != membership["canonical_sha256"]
            or membership["source_url"] != record["source"]["source_url"]):
        raise ValueError("Example emission requires admitted, hash-bound split membership")
    replay_record(record)  # Check the suffix before releasing even the first example.
    context = source_context(record)
    game = Game.new()
    produced = values = 0
    for ply, move in enumerate(record["actions"]):
        color, action = move["color"], move["action"]
        if role not in PREFIXES or color in membership["adaptation_colors"]:
            side = 0 if color == "B" else 1
            groups = membership["groups"]
            identity = membership["identity_valid"][side]
            adapting = role in PREFIXES
            if adapting and (not identity or groups[side] not in set(plan["validation_groups"] + plan["test_groups"])):
                raise ValueError("Adaptation requires the verified held-out target color")
            hidden = rank_hidden(membership["source_url"], ply, config) if role == "train" else [False, False]
            ranks = [context.black_rank, context.white_rank]
            for index, flag in zip((side, 1 - side), hidden):
                if flag:
                    ranks[index] = None
            inputs = encode(game, Context(*ranks, context.nominal_clock), history=config.history)
            if not inputs["legal"][action]:
                raise ValueError("Policy target is illegal in its pre-move observation")
            value_valid = membership["value_actions"] > 0 and not adapting
            if value_valid and (record["recorded_value_supported"] is not True or record["recorded_y_black"] not in (-1, 1)):
                raise ValueError("Value target is unsupported")
            labels = {"action": action, "policy_valid": True,
                      "value": record["recorded_y_black"] * (1 if color == "B" else -1) if value_valid else 0,
                      "value_valid": value_valid, "duration": 0, "duration_valid": False}
            known = color in membership["known_persona_colors"] and identity
            cohort_target = groups[side] in set(plan["validation_groups"] + plan["test_groups"])
            mode = "adaptation_target" if adapting else "known" if known else "default"
            metadata = {"source_url": membership["source_url"], "canonical_sha256": membership["canonical_sha256"],
                        "sgf_sha256": record["source"]["sgf_sha256"], "state_sha256": game.snapshot,
                        "role": role, "ply": ply, "sgf_node": move["node"], "mover": color,
                        "source_rank_mover": (context.black_rank, context.white_rank)[side],
                        "source_rank_opponent": (context.black_rank, context.white_rank)[1 - side],
                        "group_mover": groups[side], "group_opponent": groups[1 - side],
                        "persona_group": groups[side] if mode != "default" else None, "persona_mode": mode,
                        "identity_valid": identity, "cohort_target": cohort_target,
                        "personalization_target_valid": cohort_target and identity,
                        "adaptation_allowed": adapting, "rank_hidden": hidden,
                        "row_key": digest([membership["source_url"], ply])}
            produced += 1
            values += value_valid
            yield inputs, labels, metadata
        game = game.play(color, action)
    if produced != membership["policy_actions"] or values != membership["value_actions"]:
        raise ValueError("Emitted targets disagree with frozen per-head counts")


def write_shard(output, role, number, rows, history):
    relative = f"{role}/{number:05d}"
    prefix = output / relative
    prefix.parent.mkdir(exist_ok=True)
    schema = {**array_schema(history), **LABEL_SCHEMA}
    arrays = {}
    for key, (dtype, shape) in schema.items():
        items = [r[0 if key in INPUT_KEYS else 1][key] for r in rows]
        arrays[key] = np.asarray(items, dtype=dtype).reshape((len(rows),) + shape)
    with zipfile.ZipFile(prefix.with_suffix(".npz"), "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, array in sorted(arrays.items()):
            stream = BytesIO()
            np.lib.format.write_array(stream, array, allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, stream.getvalue())
    pq.write_table(pa.Table.from_pylist([r[2] for r in rows], schema=META_SCHEMA),
                   prefix.with_suffix(".parquet"), compression="zstd")
    return {"rows": len(rows), "arrays": relative + ".npz", "metadata": relative + ".parquet",
            "arrays_sha256": file_hash(prefix.with_suffix(".npz")),
            "metadata_sha256": file_hash(prefix.with_suffix(".parquet"))}


def build_examples(canonical_path, canonical_sha256, split_dir, split_sha256, output,
                   *, config=ExampleConfig(), fixture_only=False, profile=None, check_budget=lambda: None):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Example output already exists")
    records, split, partitions = _load_corpus(canonical_path, canonical_sha256, split_dir, split_sha256,
                                            fixture_only=fixture_only, profile=profile)
    output.mkdir(parents=True)
    artifacts, counts = {}, {}
    for role in ROLES:
        buffer, shards = [], []
        count = {"examples": 0, "value_valid": 0, "duration_valid": 0, "passes": 0,
                 "hidden_mover_rank": 0, "hidden_opponent_rank": 0}
        for membership in partitions[role]:
            check_budget()
            for example in _examples_for_game(records[membership["source_url"]], membership, split["plan"], config):
                check_budget()
                _, labels, metadata = example
                count["examples"] += 1
                if count["examples"] > (100_000 if profile is None else profile.max_examples_per_role):
                    raise ValueError("Example count budget exceeded")
                count["value_valid"] += labels["value_valid"]
                count["passes"] += labels["action"] == 361
                count["hidden_mover_rank"] += metadata["rank_hidden"][0]
                count["hidden_opponent_rank"] += metadata["rank_hidden"][1]
                buffer.append(example)
                if len(buffer) == config.shard_size:
                    shards.append(write_shard(output, role, len(shards), buffer, config.history))
                    buffer = []
        if buffer or not shards:
            shards.append(write_shard(output, role, len(shards), buffer, config.history))
        artifacts[role], counts[role] = shards, count
    schema = {k: {"dtype": d, "shape": list(s)} for k, (d, s) in
              {**array_schema(config.history), **LABEL_SCHEMA}.items()}
    manifest = {"version": VERSION, "encoding": ENCODING, "rules": RULES, "fixture_only": fixture_only,
                "canonical_sha256": canonical_sha256, "split_sha256": split_sha256,
                "config": asdict(config), "schema": schema, "input_keys": list(INPUT_KEYS),
                "scalars": list(SCALARS), "rank_vocabulary": list(RANKS), "counts": counts,
                "source_roles": split["roles"], "exclusions": split["exclusion_counts"],
                "execution_requirements": split["execution_requirements"], "artifacts": artifacts,
                "code_sha256": {name: file_hash(Path(__file__).with_name(name)) for name in CODE_FILES},
                "uv_lock_sha256": file_hash(Path("uv.lock"))}
    if profile is not None:
        manifest.update(dataset_profile_sha256=profile.sha256,
                        sealed_source_ids=sorted(sealed_sources(split)))
    (output / "manifest.json").write_text(encoded(manifest) + "\n")
    return file_hash(output / "manifest.json")


def read_shards(directory, expected_sha256, role, *, final_release=None):
    directory = Path(directory)
    if role not in ROLES or file_hash(directory / "manifest.json") != expected_sha256:
        raise ValueError("Examples require a known role and independently pinned manifest")
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("dataset_profile_sha256") is not None and manifest["fixture_only"] is not False:
        raise ValueError("A production dataset profile cannot become a synthetic fixture")
    sealed = sealed_sources(manifest)
    if not manifest["fixture_only"] and role in FINAL_ROLES and (sealed or manifest["counts"][role]["examples"]):
        require_final_release(final_release, manifest["canonical_sha256"], manifest["split_sha256"], expected_sha256)
    config = ExampleConfig(**manifest["config"])
    if manifest["version"] != VERSION or manifest["encoding"] != ENCODING or manifest["rules"] != RULES:
        raise ValueError("Unsupported example contract")
    schema = {**array_schema(config.history), **LABEL_SCHEMA}
    if (manifest["schema"] != {k: {"dtype": d, "shape": list(s)} for k, (d, s) in schema.items()}
            or manifest["input_keys"] != list(INPUT_KEYS) or manifest["scalars"] != list(SCALARS)
            or manifest["rank_vocabulary"] != RANKS):
        raise ValueError("Example schema differs from the preprocessing contract")
    for number, item in enumerate(manifest["artifacts"][role]):
        for kind, suffix in (("arrays", "npz"), ("metadata", "parquet")):
            if item[kind] != f"{role}/{number:05d}.{suffix}" or file_hash(directory / item[kind]) != item[kind + "_sha256"]:
                raise ValueError("Example shard path or content changed")
        metadata = pq.read_table(directory / item["metadata"]).to_pylist()
        if len(metadata) != item["rows"] or any(row["role"] != role for row in metadata):
            raise ValueError("Example provenance count or role mismatch")
        if role not in FINAL_ROLES and any(row["source_url"] in sealed for row in metadata):
            raise ValueError("Sealed final example entered a development or training shard")
        with np.load(directory / item["arrays"], allow_pickle=False) as data:
            if set(data.files) != set(schema):
                raise ValueError("Unexpected array fields")
            arrays = {name: data[name] for name in schema}
        for name, (dtype, shape) in schema.items():
            if (arrays[name].dtype != np.dtype(dtype) or arrays[name].shape != (item["rows"],) + shape
                    or not np.isfinite(arrays[name]).all()):
                raise ValueError("Example dtype, shape or numerical value mismatch")
        yield arrays, metadata


def fit_normalizer(directory, expected_sha256):
    count = np.zeros(len(SCALARS), dtype=np.int64)
    mean = np.zeros(len(SCALARS), dtype=np.float64)
    squared_deltas = mean.copy()
    for arrays, _ in read_shards(directory, expected_sha256, "train"):
        for index in NORMALIZED:
            selected = arrays["scalar_valid"][:, index] & arrays["policy_valid"]
            values = arrays["scalars"][selected, index].astype(np.float64)
            for value in values:
                count[index] += 1
                delta = value - mean[index]
                mean[index] += delta / count[index]
                squared_deltas[index] += delta * (value - mean[index])
    scale = np.sqrt(np.maximum(squared_deltas / np.maximum(count, 1), 0))
    scale[scale == 0] = 1
    manifest = json.loads((Path(directory) / "manifest.json").read_text())
    return {"version": "scalar_normalizer_v1", "encoding": ENCODING, "fit_role": "train",
            "examples_sha256": expected_sha256, "fixture_only": manifest["fixture_only"],
            "fitted": bool(count.any()), "scalars": list(SCALARS), "count": count.tolist(),
            "mean": mean.tolist(), "scale": scale.tolist()}
