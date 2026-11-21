"""Training-only continuation data and frozen diagnostic samples."""

from collections import Counter
from pathlib import Path

import numpy as np

from kenggo.checkpoints import file_hash, read_json, save_json
from kenggo.data_splits import read_membership
from kenggo.encoding import INPUT_KEYS, apply_normalizer
from kenggo.examples import LABEL_SCHEMA, load_dataset_profile, read_shards, sealed_sources
from kenggo.supervised import TrainingData


ROOT = Path(__file__).resolve().parents[2]
RECIPE = "e2_continuation_training_v1"
DIAGNOSTIC_VERSION = "continuation_diagnostic_v1"
ALLOWED = {"train", "population_development", "validation_development"}


def path(ref):
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise ValueError("An input needs a path and independent SHA-256")
    return (ROOT / ref["path"]).resolve()


def reference(filename):
    filename = Path(filename).resolve()
    return {"path": str(filename), "sha256": file_hash(filename)}


def read(ref):
    return read_json(path(ref), ref["sha256"])


def _shards(manifest):
    return Counter((item["arrays_sha256"], item["metadata_sha256"], item["rows"])
                   for item in manifest["artifacts"]["train"] if item["rows"])


def validate_lineage(inputs, *, fixture_only=False):
    """Keep an old training fit only when its exact training sources survive."""
    current, fitted = read(inputs["examples"]), read(inputs["normalization_examples"])
    normalizer = read(inputs["normalizer"])
    split, members = read_membership(path(inputs["split"]).parent, inputs["split"]["sha256"])
    old_split, old_members = read_membership(path(inputs["normalization_split"]).parent,
                                            inputs["normalization_split"]["sha256"])
    if (type(fixture_only) is not bool
            or any(item["fixture_only"] is not fixture_only for item in
                   (current, fitted, normalizer, split, old_split))
            or current["split_sha256"] != inputs["split"]["sha256"]
            or fitted["split_sha256"] != inputs["normalization_split"]["sha256"]
            or normalizer["examples_sha256"] != inputs["normalization_examples"]["sha256"]
            or current["config"] != fitted["config"] or current["config"]["history"] != 0
            or {key: value for key, value in split["plan"].items() if key != "grouping_sha256"}
                != {key: value for key, value in old_split["plan"].items() if key != "grouping_sha256"}
            or sealed_sources(current) != sealed_sources(fitted)
            or sealed_sources(split) != sealed_sources(old_split)):
        raise ValueError("Continuation requires the original fit and unchanged split/encoding policy")
    if not fixture_only:
        profile = load_dataset_profile(path(inputs["profile"]), inputs["profile"]["sha256"])
        if (not profile.komi_study or current["dataset_profile_sha256"] != profile.sha256
                or current["canonical_sha256"] != profile.canonical_sha256
                or current["split_sha256"] != profile.split_sha256):
            raise ValueError("Continuation requires a pinned mixed-komi production profile")
    existing = {row["source_url"]: row for row in members["train"]}
    if len(existing) != len(members["train"]) or not old_members["train"]:
        raise ValueError("Training memberships must be unique and nonempty")
    for row in old_members["train"]:
        if row["exclusions"] or existing.get(row["source_url"]) != row:
            raise ValueError("A normalizer fitting game left training or changed membership")
    for role in set(members) - {"train", "excluded"}:
        if members[role] != old_members[role]:
            raise ValueError("Growth changed a development or protected membership")
    if _shards(fitted) - _shards(current):
        raise ValueError("Growth lost or rewrote original training shards")
    for manifest, membership in ((current, members), (fitted, old_members)):
        if manifest["counts"]["train"]["examples"] != sum(row["policy_actions"] for row in membership["train"]):
            raise ValueError("Training example counts differ from membership")
    # Also validates masks, finite statistics and their training-only fitting role.
    from kenggo.encoding import encode
    from kenggo.go import Game
    apply_normalizer(encode(Game.new()), normalizer)
    fit_rows = fitted["counts"]["train"]["examples"]
    if (any(normalizer["count"][index] != fit_rows for index in (2, 3, 5))
            or any(count > fit_rows for count in normalizer["count"])):
        raise ValueError("Normalizer counts differ from the actual fitting population")
    return current, normalizer


def load_training(inputs, *, fixture_only=False):
    manifest, normalizer = validate_lineage(inputs, fixture_only=fixture_only)
    total = manifest["counts"]["train"]["examples"]
    if not 0 < total <= 32_000_000:
        raise ValueError("Continuation training rows exceed the bounded loader")
    parts = {key: [] for key in (*INPUT_KEYS, *LABEL_SCHEMA)}
    keys = []
    for arrays, metadata in read_shards(path(inputs["examples"]).parent,
                                       inputs["examples"]["sha256"], "train", komi_study=True):
        normalized = apply_normalizer({key: arrays[key] for key in INPUT_KEYS}, normalizer)
        for key in parts:
            parts[key].append(normalized[key] if key in INPUT_KEYS else arrays[key])
        keys.extend(row["row_key"] for row in metadata)
    if len(keys) != total or len(set(keys)) != total:
        raise ValueError("Training needs exactly the unique rows in its pinned manifest")
    arrays = {key: np.concatenate(values) for key, values in parts.items()}
    for array in arrays.values():
        array.flags.writeable = False
    return TrainingData({key: arrays[key] for key in INPUT_KEYS},
                        {key: arrays[key] for key in LABEL_SCHEMA}, tuple(keys),
                        inputs["examples"]["sha256"], inputs["normalizer"]["sha256"],
                        inputs["split"]["sha256"])


def prepare_diagnostic(inputs, output, *, purpose, role, source_ids, aggregate_ranks,
                       excluded_source_ids=(), expected_row_keys=None, check_budget=lambda: None):
    """Copy complete, verified allowed-role shards; never open sealed targets."""
    if role not in ALLOWED or purpose not in {"train_monitor", "epoch_validation", "assessment"}:
        raise ValueError("Diagnostics require an explicit fitting/development purpose")
    if (purpose == "train_monitor") != (role == "train"):
        raise ValueError("Validation and assessment cannot use fitting games")
    selected = set(source_ids)
    if not selected or len(selected) != len(source_ids) or selected.intersection(excluded_source_ids):
        raise ValueError("Diagnostic games must be unique and disjoint from reserved samples")
    manifest, normalizer = read(inputs["examples"]), read(inputs["normalizer"])
    split, members = read_membership(path(inputs["split"]).parent, inputs["split"]["sha256"])
    allowed = {row["source_url"]: row for row in members[role]}
    if (manifest["split_sha256"] != inputs["split"]["sha256"]
            or normalizer["examples_sha256"] != inputs["examples"]["sha256"]
            or not selected <= allowed.keys() or selected.intersection(sealed_sources(split))
            or any(allowed[key]["exclusions"] for key in selected)):
        raise ValueError("A diagnostic game is outside its pinned, unsealed whole-game role")
    expected = sum(allowed[key]["policy_actions"] for key in selected)
    parts, rows = {key: [] for key in (*INPUT_KEYS, *LABEL_SCHEMA)}, []
    for arrays, metadata in read_shards(path(inputs["examples"]).parent,
                                       inputs["examples"]["sha256"], role, komi_study=True):
        check_budget()
        indices = np.asarray([i for i, row in enumerate(metadata) if row["source_url"] in selected], dtype=int)
        if not len(indices):
            continue
        normalized = apply_normalizer({key: arrays[key][indices] for key in INPUT_KEYS}, normalizer)
        for key in parts:
            parts[key].append(normalized[key] if key in INPUT_KEYS else arrays[key][indices])
        for i in indices:
            raw = arrays["scalars"][i]
            clock = ([float(raw[9]), float(raw[10]), int(raw[11])]
                     if arrays["scalar_valid"][i, 9:12].all() else None)
            rows.append({**metadata[i], "nominal_clock": clock,
                         "komi_halfpoints": int(round(2 * abs(float(raw[1]))))})
    keys = [row["row_key"] for row in rows]
    if (len(rows) != expected or len(set(keys)) != len(keys)
            or {row["source_url"] for row in rows} != selected
            or expected_row_keys is not None and set(keys) != set(expected_row_keys)):
        raise ValueError("Diagnostic rows differ from the frozen complete games")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    arrays = {key: np.concatenate(values) for key, values in parts.items()}
    np.savez_compressed(output / "arrays.npz", **arrays)
    result = {"version": DIAGNOSTIC_VERSION, "purpose": purpose, "role": role,
              "arrays": reference(output / "arrays.npz"), "rows": rows,
              "aggregate_ranks": list(aggregate_ranks),
              "normalizer_sha256": inputs["normalizer"]["sha256"], "inputs_are_normalized": True,
              "source_ids": sorted(selected), "data": inputs,
              "excluded_rank_rows": sum(row["source_rank_mover"] not in aggregate_ranks for row in rows)}
    save_json(output / "manifest.json", result)
    return reference(output / "manifest.json")
