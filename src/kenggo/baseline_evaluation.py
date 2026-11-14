"""Pinned CPU recorded-game evaluation on predeclared, complete common sets."""

import argparse
from collections import defaultdict
from dataclasses import fields
from hashlib import sha256
import math
import os
from pathlib import Path
import random
import re
import subprocess
import time

import numpy as np
import torch

from kenggo.canonical_data import digest, encoded
from kenggo.checkpoints import file_hash, json_bytes, json_hash, read_json, save_json
from kenggo.encoding import INPUT_KEYS
from kenggo.examples import load_dataset_profile
from kenggo.go import PASS, move_category
from kenggo.katago_human import HumanProfile, HumanSL, ReferenceBudget
from kenggo.learned_agent import LearnedArtifacts, LearnedPolicy
from kenggo.learner import ModelConfig
from kenggo.recorded_data import CommonInputs, MOVE_BINS, TIME_BINS, prepare_common_set
from kenggo.recorded_statistics import (Prediction, evaluate, prediction_from_logits,
                                        prediction_from_raw_policy)
from kenggo.tracking import validate_config


ROOT = Path(__file__).resolve().parents[2]
VERSION = "baseline_evaluation_v1"
DEVELOPMENT_ROLES = {"population_development", "validation_development"}
SAMPLES = 32
CATEGORIES = ("pass", "capture", "save_atari", "give_atari", "self_atari", "other")


def _source_identity():
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], timeout=10)
    if git("ls-files", "--others", "--exclude-standard").strip():
        raise ValueError("Stage intended source files before recording evaluation provenance")
    return {"commit": git("rev-parse", "HEAD").decode().strip(),
            "patch_sha256": sha256(git("diff", "--binary", "HEAD")).hexdigest()}


def _path(ref):
    return (ROOT / ref["path"]).resolve()


def _validate(config, provenance, provenance_pin, fixture_only):
    validate_config(config)
    if (config["purpose"] != "baseline-evaluation"
            or config["arm"] not in {"conditioned", "unconditioned", "human-reference"}
            or type(fixture_only) is not bool
            or provenance.get("version") != "experiment_provenance_v1"
            or provenance.get("inputs") != config["inputs"]
            or not isinstance(provenance.get("environment"), dict)
            or json_hash(provenance) != provenance_pin
            or provenance.get("source") != _source_identity()):
        raise ValueError("Evaluation requires matching pinned configuration, provenance and source")
    refs, params = config["inputs"], config["parameters"]
    required = {"data", "canonical", "split", "examples", "normalizer", "rules"}
    if not fixture_only:
        required.add("profile")
    if not required <= refs.keys():
        raise ValueError("Evaluation requires the complete pinned data contract")
    for ref in refs.values():
        if (not isinstance(ref, dict) or set(ref) != {"path", "sha256"}
                or not isinstance(ref["path"], str) or not ref["path"]
                or file_hash(_path(ref)) != ref["sha256"]):
            raise ValueError("Evaluation input differs from its independent pin")
    batch_size = params.get("batch_size", 32)
    if (type(batch_size) is not int or not 1 <= batch_size <= 32
            or not fixture_only and batch_size != 32
            or params.get("samples_per_position", SAMPLES) != SAMPLES
            or type(params.get("unknown_rank_diagnostic", False)) is not bool
            or params.get("unknown_rank_diagnostic", False) and config["arm"] != "conditioned"):
        raise ValueError("Invalid evaluation batching or predeclared sampling/diagnostic settings")
    specs = params.get("common_sets")
    if not isinstance(specs, list) or not specs:
        raise ValueError("Evaluation needs explicit frozen common sets")
    ids, roles = set(), set()
    for spec in specs:
        if (not isinstance(spec, dict)
                or set(spec) != {"id", "role", "source_ids", "aggregate_ranks", "manifest_input"}
                or not isinstance(spec["id"], str)
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", spec["id"]) is None
                or spec["id"] in ids or spec["role"] in roles
                or spec["role"] not in DEVELOPMENT_ROLES | ({"train"} if fixture_only else set())
                or spec["manifest_input"] not in refs):
            raise ValueError("Common sets must have unique identifiers and separate development roles")
        ids.add(spec["id"])
        roles.add(spec["role"])


def _common_sets(config, *, fixture_only):
    refs = config["inputs"]
    profile = (load_dataset_profile(_path(refs["profile"]), refs["profile"]["sha256"])
               if "profile" in refs else None)
    inputs = CommonInputs(_path(refs["canonical"]), refs["canonical"]["sha256"],
        _path(refs["split"]).parent, refs["split"]["sha256"],
        _path(refs["examples"]).parent, refs["examples"]["sha256"],
        _path(refs["normalizer"]), refs["normalizer"]["sha256"], profile)
    result = []
    for spec in config["parameters"]["common_sets"]:
        ref = refs[spec["manifest_input"]]
        expected = read_json(_path(ref), ref["sha256"])
        common = prepare_common_set(inputs, role=spec["role"], source_ids=spec["source_ids"],
                                   aggregate_ranks=spec["aggregate_ranks"], fixture_only=fixture_only)
        if common.sha256 != ref["sha256"] or common.manifest != expected:
            raise ValueError("Prepared common set differs from its frozen manifest")
        result.append((spec, common))
    return result


def _learned_artifacts(config):
    refs, raw = config["inputs"], config["parameters"].get("artifacts")
    if not isinstance(raw, dict) or set(raw) != {field.name for field in fields(LearnedArtifacts)}:
        raise ValueError("A complete independently pinned learned artifact bundle is required")
    artifacts = LearnedArtifacts(**{**raw, "model_config": ModelConfig(**raw["model_config"])})
    for name, path, pin in (
            ("checkpoint", artifacts.checkpoint_manifest, artifacts.checkpoint_sha256),
            ("training_config", artifacts.config_path, artifacts.config_sha256),
            ("normalizer", artifacts.normalizer_path, artifacts.normalizer_sha256)):
        if (name not in refs or _path(refs[name]) != (ROOT / path).resolve()
                or refs[name]["sha256"] != pin):
            raise ValueError("Learned artifacts disagree with evaluation input pins")
    if (refs.get("training_provenance", {}).get("sha256") != artifacts.provenance_sha256
            or refs["examples"]["sha256"] != artifacts.examples_sha256
            or artifacts.model_config.rank_conditioned != (config["arm"] == "conditioned")):
        raise ValueError("Learned provenance, examples or conditioning differs from the declared arm")
    training_config = read_json(_path(refs["training_config"]), artifacts.config_sha256)
    training_provenance = read_json(_path(refs["training_provenance"]), artifacts.provenance_sha256)
    if (training_provenance.get("version") != "experiment_provenance_v1"
            or training_provenance.get("inputs") != training_config.get("inputs")):
        raise ValueError("Training provenance differs from the pinned training inputs")
    return LearnedArtifacts(**{**raw, "checkpoint_manifest": str(_path(refs["checkpoint"])),
        "config_path": str(_path(refs["training_config"])),
        "normalizer_path": str(_path(refs["normalizer"])), "model_config": artifacts.model_config})


def rows_for_common(common):
    return [{**dict(position.metadata), **dict(position.targets),
             "legal": position.observation.inputs["legal"].tolist()}
            for position in common.positions()]


def normalized_batch(positions, *, unknown_ranks=False):
    inputs = {key: torch.from_numpy(np.stack([position.observation.inputs[key]
                                             for position in positions])) for key in INPUT_KEYS}
    if unknown_ranks:
        inputs["ranks"].zero_()
        inputs["rank_valid"].zero_()
    return inputs


def sample_actions(prediction, *, seed, variant):
    private_seed = int(digest({"seed": seed, "row_key": prediction.row_key, "variant": variant}), 16)
    return random.Random(private_seed).choices(range(PASS + 1),
        weights=[math.exp(value) for value in prediction.log_policy], k=SAMPLES)


def _arrays(predictions, *, seed, variant):
    return {"row_keys": np.asarray([item.row_key for item in predictions], dtype=np.str_),
            "log_policy": np.asarray([item.log_policy for item in predictions], dtype=np.float64),
            "values": np.asarray([item.value for item in predictions], dtype=np.float64),
            "sampled_actions": np.asarray([sample_actions(item, seed=seed, variant=variant)
                                            for item in predictions], dtype=np.int16),
            "raw_pass_mass": np.asarray([math.exp(item.log_policy[PASS]) for item in predictions],
                                         dtype=np.float64)}


def _save_predictions(path, predictions, *, seed, variant):
    with Path(path).open("xb") as stream:
        np.savez(stream, **_arrays(predictions, seed=seed, variant=variant))
        stream.flush()
        os.fsync(stream.fileno())
    return file_hash(path)


def load_predictions(path, *, expected_row_keys):
    """Read the portable output while retaining its exact frozen row order."""
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    count = len(expected_row_keys)
    if (set(arrays) != {"row_keys", "log_policy", "values", "sampled_actions", "raw_pass_mass"}
            or arrays["row_keys"].dtype.kind != "U"
            or arrays["row_keys"].shape != (count,)
            or arrays["row_keys"].tolist() != list(expected_row_keys)
            or arrays["log_policy"].dtype != np.float64 or arrays["log_policy"].shape != (count, PASS + 1)
            or arrays["values"].dtype != np.float64 or arrays["values"].shape != (count,)
            or arrays["sampled_actions"].dtype != np.int16
            or arrays["sampled_actions"].shape != (count, SAMPLES)
            or ((arrays["sampled_actions"] < 0) | (arrays["sampled_actions"] > PASS)).any()
            or arrays["raw_pass_mass"].dtype != np.float64
            or arrays["raw_pass_mass"].shape != (count,)):
        raise ValueError("Prediction archive has incomplete, reordered or invalid output arrays")
    predictions = [Prediction(key, tuple(logs), value) for key, logs, value in
                   zip(arrays["row_keys"].tolist(), arrays["log_policy"], arrays["values"])]
    if not np.array_equal(arrays["raw_pass_mass"], [math.exp(item.log_policy[PASS]) for item in predictions]):
        raise ValueError("Saved raw pass mass differs from the raw policy")
    for prediction, actions in zip(predictions, arrays["sampled_actions"]):
        if any(prediction.log_policy[int(action)] == -math.inf for action in actions):
            raise ValueError("A sampled action has zero policy mass")
    return predictions


def _category_group(rows, observations, ranks):
    sides = defaultdict(list)
    for row in rows:
        sides[(row["source_rank_mover"], row["source_url"], row["mover"])].append(observations[row["row_key"]])
    missing = [rank for rank in ranks if not any(key[0] == rank for key in sides)]
    movers = {row["group_mover"] for row in rows}
    counts = {"games": len({row["source_url"] for row in rows}), "game_sides": len(sides),
              "moves": len(rows), "mover_accounts": len(movers),
              "accounts": len(movers | {row["group_opponent"] for row in rows})}
    frequencies = {kind: {category: None for category in CATEGORIES} for kind in ("model", "recorded")}
    if not missing:
        for kind in frequencies:
            for category in CATEGORIES:
                rank_means = []
                for rank in ranks:
                    selected = [values for key, values in sides.items() if key[0] == rank]
                    rank_means.append(math.fsum(math.fsum(row[kind][category] for row in values) / len(values)
                                                for values in selected) / len(selected))
                frequencies[kind][category] = math.fsum(rank_means) / len(ranks)
    reasons = ["single_seed_exploratory", "confidence_interval_not_estimated"]
    if missing:
        reasons.append("missing_aggregate_ranks")
    if counts["mover_accounts"] < 30:
        reasons.append("fewer_than_30_mover_accounts")
    if counts["game_sides"] < 100:
        reasons.append("fewer_than_100_game_sides")
    return {"status": "unavailable" if missing else "available", "counts": counts,
            "aggregate_ranks": list(ranks), "missing_ranks": missing, **frequencies,
            "total_variation": None if missing else .5 * math.fsum(
                abs(frequencies["model"][category] - frequencies["recorded"][category]) for category in CATEGORIES),
            "promotion_supported": False,
            "uncertainty": {"status": "inconclusive", "interval95": None, "reasons": reasons}}


def _sampling_report(rows, predictions, *, seed, variant, ranks, games):
    sides, observations = defaultdict(list), {}
    for row, prediction in zip(rows, predictions):
        actions = sample_actions(prediction, seed=seed, variant=variant)
        counts = np.bincount(actions, minlength=PASS + 1)
        categories = {action: move_category(games[row["row_key"]], action)
                      for action in set(actions) | {row["action"]}}
        observations[row["row_key"]] = {
            "model": {category: sum(categories[action] == category for action in actions) / SAMPLES
                      for category in CATEGORIES},
            "recorded": {category: float(categories[row["action"]] == category) for category in CATEGORIES}}
        sides[(row["source_rank_mover"], row["source_url"], row["mover"])].append({
            "unique_legal_actions": len(set(actions)),
            "pairwise_disagreement": 1 - int(np.sum(counts * (counts - 1))) / (SAMPLES * (SAMPLES - 1)),
            "raw_pass_mass": math.exp(prediction.log_policy[PASS]),
            "sampled_pass_fraction": actions.count(PASS) / SAMPLES})
    by_rank = {}
    for rank in ranks:
        selected = [values for key, values in sides.items() if key[0] == rank]
        by_rank[rank] = {name: math.fsum(math.fsum(row[name] for row in values) / len(values)
                                        for values in selected) / len(selected) if selected else None
                         for name in ("unique_legal_actions", "pairwise_disagreement",
                                      "raw_pass_mass", "sampled_pass_fraction")}
    overall = {name: math.fsum(by_rank[rank][name] for rank in ranks) / len(ranks)
               if all(by_rank[rank][name] is not None for rank in ranks) else None
               for name in next(iter(by_rank.values()))}
    category_report = {"categories": list(CATEGORIES), "classifier": "kenggo.go.move_category",
        "model_distribution": "32 independent temperature-1 samples per position",
        "overall": _category_group(rows, observations, ranks),
        "by_rank": {rank: _category_group([row for row in rows if row["source_rank_mover"] == rank], observations, [rank])
                    for rank in ranks},
        "by_phase": {phase: _category_group([row for row in rows if row["move_bin"] == phase], observations, ranks)
                     for phase in MOVE_BINS},
        "by_clock": {clock: _category_group([row for row in rows if row["main_time_bin"] == clock], observations, ranks)
                     for clock in TIME_BINS},
        "joint_cells": [{"rank": rank, "phase": phase, "clock": clock, **_category_group(
            [row for row in rows if row["source_rank_mover"] == rank and row["move_bin"] == phase
             and row["main_time_bin"] == clock], observations, [rank])}
            for rank in ranks for phase in MOVE_BINS for clock in TIME_BINS]}
    return {"draws_per_position": SAMPLES, "temperature": 1, "pass_correction": False,
            "weighting": "equal_ranks_equal_game_sides_equal_positions",
            "overall": overall, "by_rank": by_rank, "move_categories": category_report,
            "scope": "Recorded-position sampling diagnostics; no human conditional entropy or game qualification claim"}


class _Progress:
    def __init__(self, output, config_pin, provenance_pin):
        self.output = output
        self.started, self.cpu_started = time.monotonic(), time.process_time()
        self.state = {"version": VERSION, "status": "preparing", "config_sha256": config_pin,
            "provenance_sha256": provenance_pin, "attempted_rows": 0, "completed_rows": 0,
            "completed_units": 0, "learned_batches_attempted": 0, "analysis_queries_attempted": 0,
            "metadata_requests_attempted": 0, "metadata_requests_completed": 0,
            "trace_count": 0, "last_trace": None, "last_unit": None, "current": None,
            "sets": {}, "native_cost": None, "failures": []}
        self.write()

    def write(self):
        self.state["worker_cost"] = {"wall_seconds": time.monotonic() - self.started,
            "cpu_seconds": time.process_time() - self.cpu_started, "gpu_seconds": 0.0,
            "scope": "Worker process only; parent supervisor accounts for child/native CPU"}
        save_json(self.output / "progress.json", self.state, replace=True)

    def ref(self, path, pin=None):
        return {"path": str(path.relative_to(self.output)), "sha256": pin or file_hash(path)}

    def trace(self, record):
        path = self.output / "traces" / f"trace-{self.state['trace_count'] + 1:06d}.json"
        self.state["last_trace"] = self.ref(path, save_json(path, record))
        self.state["trace_count"] += 1
        self.state["native_cost"] = record["cost"]
        self.write()

    def unit(self, record):
        with (self.output / "units.jsonl").open("ab") as stream:
            stream.write(json_bytes(record))
            stream.flush()
            os.fsync(stream.fileno())
        self.state["completed_rows"] += record["rows"]
        self.state["completed_units"] += 1
        self.state["last_unit"] = record
        self.write()


def _learned_predictions(client, positions, inputs, normalizer_pin):
    logits, values = client.predict_batch(inputs, normalizer_sha256=normalizer_pin)
    if (not isinstance(logits, torch.Tensor) or not isinstance(values, torch.Tensor)
            or logits.device.type != "cpu" or values.device.type != "cpu"
            or logits.shape != (len(positions), PASS + 1) or values.shape != (len(positions),)
            or logits.dtype != torch.float32 or values.dtype != torch.float32
            or not torch.isneginf(logits[~inputs["legal"]]).all()):
        raise ValueError("Learned evaluation returned incomplete or invalid CPU output")
    return [prediction_from_logits(position.observation.row_key, row.tolist(), float(value),
                                   position.observation.inputs["legal"])
            for position, row, value in zip(positions, logits, values)]


def _evaluate_set(config, spec, common, variant, client, progress):
    directory = progress.output / spec["id"] / variant
    directory.mkdir()
    chunks = directory / "chunks"
    chunks.mkdir()
    positions, rows, manifest = list(common.positions()), rows_for_common(common), common.manifest
    predictions, timings = [], []
    human = config["arm"] == "human-reference"
    batch_size = 1 if human else config["parameters"].get("batch_size", 32)
    key = spec["id"] + "/" + variant
    progress.state["sets"][key] = {"expected_rows": len(common), "attempted_rows": 0, "completed_rows": 0,
                                    "common_set_sha256": common.sha256, "status": "running"}
    for offset in range(0, len(positions), batch_size):
        batch = positions[offset:offset + batch_size]
        progress.state["current"] = {"set": spec["id"], "variant": variant, "offset": offset}
        progress.state["attempted_rows"] += len(batch)
        progress.state["sets"][key]["attempted_rows"] += len(batch)
        progress.state["analysis_queries_attempted" if human else "learned_batches_attempted"] += 1
        progress.write()
        inputs = None if human else normalized_batch(batch, unknown_ranks=variant == "unknown_ranks")
        started = time.monotonic()
        if human:
            position = batch[0]
            native = client.predict(position.observation.game, position.metadata["source_rank_mover"],
                                    allow_fixture=manifest["fixture_only"])
            if (native.snapshot != position.observation.game.snapshot
                    or native.rank != position.metadata["source_rank_mover"]):
                raise ValueError("Human reference returned a stale position or conditioning rank")
            predicted = [prediction_from_raw_policy(position.observation.row_key, native.raw_policy,
                         native.value, position.observation.inputs["legal"])]
            progress.state["native_cost"] = client.cost()
        else:
            predicted = _learned_predictions(client, batch, inputs, config["inputs"]["normalizer"]["sha256"])
        timing = {"offset": offset, "rows": len(batch), "seconds": time.monotonic() - started,
                  "unit": "reference_query" if human else "learned_batch",
                  "network_calls": None if human else 1}
        path = chunks / f"{len(timings):06d}.npz"
        pin = _save_predictions(path, predicted, seed=config["seed"], variant=variant)
        predictions.extend(predicted)
        timings.append(timing)
        progress.state["sets"][key]["completed_rows"] += len(batch)
        progress.unit({"set": spec["id"], "variant": variant, "rows": len(batch),
                       "prediction_chunk": progress.ref(path, pin), "timing": timing})
    if [prediction.row_key for prediction in predictions] != [row["row_key"] for row in rows]:
        raise ValueError("Predictions must cover the frozen rows in their original order")
    metrics = evaluate(rows, predictions, aggregate_ranks=spec["aggregate_ranks"])
    artifacts = {}
    for name, value in (("metrics.json", metrics), ("timing.json", {
            "unit": "reference_query" if human else "learned_batch", "measurements": timings,
            "scope": "Human queries include history validation and trace persistence; learned batches include prediction and output validation. Prepared input construction and prediction persistence are excluded.",
            "per_position_latency_measured": human}), ("sampling.json", _sampling_report(
                rows, predictions, seed=config["seed"], variant=variant, ranks=spec["aggregate_ranks"],
                games={position.observation.row_key: position.observation.game for position in positions}))):
        path = directory / name
        artifacts[name] = progress.ref(path, save_json(path, value))
    path = directory / "predictions.npz"
    artifacts[path.name] = progress.ref(path, _save_predictions(path, predictions, seed=config["seed"], variant=variant))
    summary = {"version": VERSION, "status": "completed", "set": spec["id"], "role": spec["role"],
        "arm": config["arm"], "variant": variant, "rows": len(rows), "common_set_sha256": common.sha256,
        "coverage": manifest["coverage"], "artifacts": artifacts, "promotion": "not_promoted",
        "scope": "Exploratory recorded-position evaluation; no rank calibration or complete-game qualification"}
    path = directory / "summary.json"
    progress.state["sets"][key].update(status="completed", summary=progress.ref(path, save_json(path, summary)))
    progress.write()
    return summary


def run_evaluation(config_path, config_pin, provenance_path, provenance_pin, output, *, fixture_only=False):
    output = Path(output).resolve()
    if not output.is_dir() or any((output / name).exists() for name in
                                 ("progress.json", "summary.json", "failure.json", "units.jsonl", "traces")):
        raise ValueError("Evaluation needs a fresh parent-created job directory")
    progress = _Progress(output, config_pin, provenance_pin)
    client = None
    try:
        config = read_json(config_path, config_pin)
        provenance = read_json(provenance_path, provenance_pin)
        _validate(config, provenance, provenance_pin, fixture_only)
        progress.state.update(arm=config["arm"], input_hashes={name: ref["sha256"]
                                                             for name, ref in config["inputs"].items()})
        common_sets = _common_sets(config, fixture_only=fixture_only)
        for spec, common in common_sets:
            directory = output / spec["id"]
            directory.mkdir()
            with (directory / "common-set.json").open("x") as stream:
                stream.write(encoded(common.manifest))
                stream.flush()
                os.fsync(stream.fileno())
        human = config["arm"] == "human-reference"
        if human:
            refs, params = config["inputs"], config["parameters"]
            profile = HumanProfile.load(_path(refs["human_profile"]), refs["human_profile"]["sha256"])
            paths = params["human_paths"]
            if (set(paths) != {"engine", "main_model", "human_model", "runtime_root"}
                    or refs["human_config"]["sha256"] != profile.data["artifacts"]["config"]):
                raise ValueError("Native reference paths and config must match the pinned human profile")
            budget = ReferenceBudget(**params["reference_budget"])
            if budget.queries != sum(len(common) for _, common in common_sets):
                raise ValueError("The shared native query allowance must match the complete frozen sets")
            (output / "traces").mkdir()
            client = HumanSL(profile, **{name: str((ROOT / path).resolve()) for name, path in paths.items()},
                             config=_path(refs["human_config"]), budget=budget, trace_sink=progress.trace)
            progress.state["metadata_requests_attempted"] += 1
            progress.write()
            client.query_version()
            progress.state["metadata_requests_completed"] += 1
            progress.state["native_cost"] = client.cost()
        else:
            client = LearnedPolicy.load(_learned_artifacts(config))
        progress.state["status"] = "running"
        progress.write()
        variants = ("ordinary", "unknown_ranks") if config["parameters"].get("unknown_rank_diagnostic", False) else ("ordinary",)
        reports = [_evaluate_set(config, spec, common, variant, client, progress)
                   for spec, common in common_sets for variant in variants]
        client.close()
        if human:
            progress.state["native_cost"] = client.cost()
        client = None
        if provenance["source"] != _source_identity():
            raise ValueError("Source changed during evaluation")
        progress.state.update(status="completed", current=None)
        progress.write()
        result = {**progress.state, "reports": reports, "units": progress.ref(output / "units.jsonl"),
                  "native_neural_forwards": None if human else 0, "gpu_seconds": 0.0,
                  "promotion": "not_promoted", "automatic_retries": 0}
        save_json(output / "summary.json", result)
        return result
    except BaseException as error:
        progress.state["failures"].append({"type": type(error).__name__, "reason": str(error)})
        if getattr(error, "cost", None) is not None:
            progress.state["native_cost"] = error.cost
        if client is not None:
            try:
                client.close()
            except BaseException as close_error:
                progress.state["failures"].append({"type": type(close_error).__name__, "reason": str(close_error)})
            if config["arm"] == "human-reference":
                progress.state["native_cost"] = client.cost()
        progress.state["status"] = "failed"
        progress.write()
        save_json(output / "failure.json", {**progress.state, "automatic_retries": 0})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("provenance", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--provenance-sha256", required=True)
    parser.add_argument("--fixture-only", action="store_true")
    args = parser.parse_args()
    run_evaluation(args.config, args.config_sha256, args.provenance, args.provenance_sha256,
                   args.output, fixture_only=args.fixture_only)
