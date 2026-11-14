"""Purpose-restricted, frozen observations and targets for recorded-game comparisons."""

from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np

from kenggo.canonical_data import digest, encoded
from kenggo.encoding import (Context, INPUT_KEYS, SCALARS, VERSION as ENCODING,
                             apply_normalizer, array_schema)
from kenggo.examples import (DatasetProfile, ExampleConfig, LABEL_SCHEMA, RECORDED_ROLES,
                             VERSION as EXAMPLES_VERSION, examples_for_game, load_corpus,
                             read_pinned_json, sealed_sources, source_context)
from kenggo.go import Game, RULES
from kenggo.kgs_source import RANKS


VERSION = "recorded_common_set_v1"
MOVE_BINS = ("1-60", "61-180", "181+")
TIME_BINS = ("10-<20", "20-30", "missing")


@dataclass(frozen=True)
class CommonInputs:
    canonical_path: Path
    canonical_sha256: str
    split_dir: Path
    split_sha256: str
    examples_dir: Path
    examples_sha256: str
    normalizer_path: Path
    normalizer_sha256: str
    profile: DatasetProfile | None = None


@dataclass(frozen=True)
class RecordedObservation:
    row_key: str
    game: Game
    inputs: MappingProxyType
    context: Context


@dataclass(frozen=True)
class RecordedPosition:
    observation: RecordedObservation
    targets: MappingProxyType
    metadata: MappingProxyType


@dataclass(frozen=True)
class CommonSet:
    _manifest_json: str
    _positions: tuple[RecordedPosition, ...]

    @property
    def manifest(self):
        """Return a disposable JSON value; callers cannot change the frozen set."""
        return json.loads(self._manifest_json)

    @property
    def sha256(self):
        return sha256(self._manifest_json.encode()).hexdigest()

    def __len__(self):
        return len(self._positions)

    def positions(self):
        for position in self._positions:
            observation = position.observation
            inputs = MappingProxyType({key: value.view() for key, value in observation.inputs.items()})
            yield replace(position, observation=replace(observation, inputs=inputs))


def _immutable_inputs(inputs):
    # Bytes-backed arrays cannot regain write access through setflags or a base view.
    return MappingProxyType({name: np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
                             for name, value in inputs.items()})


def _inputs_hash(inputs):
    return digest({name: {"dtype": str(value.dtype), "shape": list(value.shape),
                          "sha256": sha256(value.tobytes()).hexdigest()} for name, value in inputs.items()})


def _example_contract(inputs, split, fixture_only):
    manifest = read_pinned_json(Path(inputs.examples_dir) / "manifest.json", inputs.examples_sha256,
                                max_bytes=16 * 1024**2)
    normalizer = read_pinned_json(inputs.normalizer_path, inputs.normalizer_sha256)
    config = ExampleConfig(**manifest["config"])
    schema = {k: {"dtype": d, "shape": list(s)} for k, (d, s) in
              {**array_schema(config.history), **LABEL_SCHEMA}.items()}
    if (manifest["version"] != EXAMPLES_VERSION or manifest["encoding"] != ENCODING
            or manifest["rules"] != RULES or config.history != 0
            or manifest["fixture_only"] is not fixture_only
            or manifest["canonical_sha256"] != inputs.canonical_sha256
            or manifest["split_sha256"] != inputs.split_sha256
            or manifest["schema"] != schema or manifest["input_keys"] != list(INPUT_KEYS)
            or manifest["scalars"] != list(SCALARS) or manifest["rank_vocabulary"] != RANKS
            or manifest["source_roles"] != split["roles"]
            or manifest["exclusions"] != split["exclusion_counts"]
            or sealed_sources(manifest) != sealed_sources(split)
            or manifest.get("dataset_profile_sha256") != (inputs.profile.sha256 if inputs.profile else None)
            or normalizer["examples_sha256"] != inputs.examples_sha256
            or normalizer["fixture_only"] is not fixture_only):
        raise ValueError("Recorded examples and normalizer must match the pinned zero-history source contract")
    for role in split["roles"]:
        if (manifest["counts"][role]["examples"] != split["policy_actions_by_role"][role]
                or manifest["counts"][role]["value_valid"] != split["value_actions_by_role"][role]
                or manifest["counts"][role]["duration_valid"] != 0):
            raise ValueError("Example target counts differ from the frozen split")
    return config, normalizer


def move_bin(ply):
    """Bins use the next one-based individual move, including recorded passes."""
    if type(ply) is not int or ply < 0:
        raise ValueError("A move bin needs a nonnegative zero-based ply")
    return MOVE_BINS[0 if ply < 60 else 1 if ply < 180 else 2]


def main_time_bin(clock):
    if clock is None:
        return "missing"
    Context(nominal_clock=clock)
    return TIME_BINS[0 if clock[0] < 1200 else 1]


def _counts(rows):
    movers = {row["group_mover"] for row in rows}
    sides = {(row["source_url"], row["mover"]) for row in rows}
    reasons = []
    if len(movers) < 30:
        reasons.append("fewer_than_30_mover_accounts")
    if len(sides) < 100:
        reasons.append("fewer_than_100_game_sides")
    return {"games": len({row["source_url"] for row in rows}), "game_sides": len(sides),
            "accounts": len(movers | {row["group_opponent"] for row in rows}),
            "mover_accounts": len(movers), "moves": len(rows),
            "promotion_supported": not reasons, "unsupported_reasons": reasons}


def _coverage(positions, records, memberships, aggregate_ranks):
    rows = [dict(position.metadata) for position in positions]
    selected = {row["source_url"]: row for row in memberships}
    no_actions = []
    for key, membership in selected.items():
        for index, color in enumerate(("B", "W")):
            if not membership["actions_by_color"][index]:
                no_actions.append({"source_url": key, "mover": color,
                                   "group_mover": membership["groups"][index],
                                   "source_rank_mover": records[key]["metadata"]["BR" if index == 0 else "WR"],
                                   "reason": "no_eligible_policy_actions"})
    dimensions = {
        "source_rank_mover": list(RANKS), "source_rank_opponent": list(RANKS),
        "move_bin": list(MOVE_BINS), "main_time_bin": list(TIME_BINS),
        "nominal_clock": sorted({row["nominal_clock"] for row in rows}, key=encoded),
        "nominal_clock_missing": [False, True],
    }
    tables = {name: [{name: value, **_counts([row for row in rows if row[name] == value])}
                     for value in values] for name, values in dimensions.items()}
    joint = defaultdict(list)
    for row in rows:
        joint[tuple(row[name] for name in dimensions)].append(row)
    joint_cells = [{**dict(zip(dimensions, values)), **_counts(cell)}
                   for values, cell in sorted(joint.items(), key=lambda item: encoded(item[0]))]
    present = {row["source_rank_mover"] for row in rows}
    missing = [rank for rank in aggregate_ranks if rank not in present]
    absent_all = [rank for rank in RANKS if rank not in present]
    rank_counts = {cell["source_rank_mover"]: cell for cell in tables["source_rank_mover"]}
    return {
        "candidate": {"games": len(selected), "game_sides": 2 * len(selected),
                      "accounts": len({group for row in memberships for group in row["groups"]}),
                      "moves": sum(row["actions"] for row in memberships)},
        "retained": _counts(rows), "by": tables, "joint_cells": joint_cells,
        "exclusions": {"game_sides": no_actions,
                       "value_masked_moves": sum(not position.targets["value_valid"] for position in positions),
                       "duration_masked_moves": sum(not position.targets["duration_valid"] for position in positions)},
        "aggregate": {"ranks": list(aggregate_ranks), "missing_ranks": missing,
                      "status": "unavailable" if missing else "available",
                      "promotion_supported": all(rank_counts[rank]["promotion_supported"] for rank in aggregate_ranks)},
        "all_rank_aggregate": {"ranks": list(RANKS), "missing_ranks": absent_all,
                               "status": "unavailable" if absent_all else "available",
                               "promotion_supported": all(cell["promotion_supported"] for cell in rank_counts.values())},
    }


def prepare_common_set(inputs, *, role, source_ids, aggregate_ranks, fixture_only=False):
    """Freeze selected complete games before any observation can reach a predictor.

    The pinned split's order determines row order. Aggregate ranks select reporting
    weights; every policy-valid row from the requested games remains in the set.
    No example arrays, confirmation payloads or final payloads are opened.
    """
    if not isinstance(inputs, CommonInputs) or role not in RECORDED_ROLES:
        raise ValueError("A common set requires pinned inputs and a training-debug or development role")
    if isinstance(aggregate_ranks, (str, bytes)):
        raise ValueError("Aggregate ranks must be an explicit nonempty unique sequence")
    aggregate_ranks = tuple(aggregate_ranks)
    if (not aggregate_ranks or len(set(aggregate_ranks)) != len(aggregate_ranks)
            or any(rank not in RANKS for rank in aggregate_ranks)):
        raise ValueError("Aggregate ranks must be an explicit nonempty unique supported sequence")
    records, split, partitions = load_corpus(
        inputs.canonical_path, inputs.canonical_sha256, inputs.split_dir, inputs.split_sha256,
        fixture_only=fixture_only, profile=inputs.profile, role=role, source_ids=source_ids)
    config, normalizer = _example_contract(inputs, split, fixture_only)
    memberships = partitions[role]
    limit = inputs.profile.max_examples_per_role if inputs.profile else 100_000
    if sum(row["policy_actions"] for row in memberships) > limit:
        raise ValueError("Recorded common set exceeds its example budget")
    positions = []
    for membership in memberships:
        record = records[membership["source_url"]]
        context = source_context(record)
        game = Game.new()
        for raw, targets, metadata in examples_for_game(record, membership, split["plan"], config):
            if len(game.moves) != metadata["ply"] or game.snapshot != metadata["state_sha256"]:
                raise ValueError("Recorded observation diverged from the shared example replay")
            side = 0 if metadata["mover"] == "B" else 1
            ranks = [context.black_rank, context.white_rank]
            for index, hidden in zip((side, 1 - side), metadata["rank_hidden"]):
                if hidden:
                    ranks[index] = None
            observation = RecordedObservation(metadata["row_key"], game,
                _immutable_inputs(apply_normalizer(raw, normalizer)), Context(*ranks, context.nominal_clock))
            metadata = {**metadata, "rank_hidden": tuple(metadata["rank_hidden"]),
                        "nominal_clock": context.nominal_clock, "nominal_clock_missing": context.nominal_clock is None,
                        "move_bin": move_bin(metadata["ply"]), "main_time_bin": main_time_bin(context.nominal_clock)}
            positions.append(RecordedPosition(observation, MappingProxyType(dict(targets)), MappingProxyType(metadata)))
            game = game.play(metadata["mover"], targets["action"])
    if not positions or len({p.observation.row_key for p in positions}) != len(positions):
        raise ValueError("A common set needs nonempty, unique eligible row keys")
    if {p.metadata["source_url"] for p in positions} != set(records):
        raise ValueError("A selected complete game has no eligible policy observations")
    rows = [{key: position.metadata[key] for key in
             ("row_key", "source_url", "canonical_sha256", "sgf_sha256", "state_sha256", "ply", "mover")}
            | {key: position.targets[key] for key in ("policy_valid", "value_valid", "duration_valid")}
            | {"rank_hidden": list(position.metadata["rank_hidden"]),
               "input_masks": {key: position.observation.inputs[key].tolist()
                               for key in ("scalar_valid", "rank_valid", "history_valid")},
               "inputs_sha256": _inputs_hash(position.observation.inputs)} for position in positions]
    manifest = {
        "version": VERSION, "fixture_only": fixture_only, "role": role,
        "purpose": "training_debug" if role == "train" else "exploratory_development",
        "encoding": ENCODING, "rules": RULES, "model_history": 0,
        "inputs": {"canonical_sha256": inputs.canonical_sha256, "split_sha256": inputs.split_sha256,
                   "examples_sha256": inputs.examples_sha256, "normalizer_sha256": inputs.normalizer_sha256,
                   "dataset_profile": asdict(inputs.profile) if inputs.profile else None,
                   "population_inputs_sha256": split["provenance"].get("population_inputs_sha256"),
                   "sealed_source_ids_sha256": digest(sorted(sealed_sources(split))),
                   "split_plan_sha256": split["plan_sha256"]},
        "source_ids": [row["source_url"] for row in memberships],
        "aggregate_ranks": list(aggregate_ranks), "rows": rows, "rows_sha256": digest(rows),
        "targets_sha256": digest([dict(position.targets) for position in positions]),
        "coverage": _coverage(positions, records, memberships, aggregate_ranks),
    }
    return CommonSet(encoded(manifest), tuple(positions))
