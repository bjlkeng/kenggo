"""One matched training round with isolated validation and recoverable checkpoints."""

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import math
import os
from pathlib import Path
import platform
import re
import time

import numpy as np
import torch

from kenggo.baseline_training import _source_identity
from kenggo.checkpoints import (file_hash, json_hash, load_checkpoint, read_json, restore_rng,
                                rng_state, save_checkpoint, save_json)
from kenggo.encoding import INPUT_KEYS, array_schema
from kenggo.examples import LABEL_SCHEMA
from kenggo.kgs_source import RANKS
from kenggo.learner import ModelConfig, PopulationModel
from kenggo.recorded_statistics import evaluate, prediction_from_logits
from kenggo.supervised import train_step
from kenggo.tracking import validate_config


ROOT = Path(__file__).resolve().parents[2]
RECIPE = "e2_continuation_training_v1"
DIAGNOSTIC_VERSION = "continuation_diagnostic_v1"
REVIEW_RANKS = tuple(RANKS[RANKS.index("12k"):])
DEFAULTS = {"batch_size": 128, "learning_rate": 0.001, "clock_dropout": 0.1,
            "recovery_updates": 1000, "min_delta": 0.002, "patience": 3,
            "max_rounds": 1000, "eval_batch_size": 128, "evaluate_only": False,
            "inference_contract": "komi_comparison_v1", "training_komi_halfpoints": [1, 13],
            "target_round": None}


def _path(ref):
    return (ROOT / ref["path"]).resolve()


def _reference(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_hash(path)}


def load_training(inputs, *, fixture_only=False):
    from kenggo.continuation_data import load_training as load
    return load(inputs, fixture_only=fixture_only)


@dataclass(frozen=True)
class Diagnostic:
    inputs: dict
    labels: dict
    rows: tuple
    ranks: tuple
    purpose: str
    sha256: str


def load_diagnostic(ref, purpose, normalizer_sha256, *, fixture_only=False):
    manifest = read_json(_path(ref), ref["sha256"])
    roles = {"train_monitor": {"train"},
             "epoch_validation": {"population_development", "validation_development"},
             "assessment": {"population_development", "validation_development"}}
    required = {"version", "purpose", "role", "arrays", "rows", "aggregate_ranks",
                "normalizer_sha256", "inputs_are_normalized", "source_ids"}
    if (purpose not in roles or not isinstance(manifest, dict) or not required <= manifest.keys()
            or manifest["version"] != DIAGNOSTIC_VERSION or manifest["purpose"] != purpose
            or manifest["role"] not in roles[purpose]
            or manifest["normalizer_sha256"] != normalizer_sha256
            or manifest["inputs_are_normalized"] is not True):
        raise ValueError("Diagnostic purpose, role or training-only normalizer differs")
    ranks, rows, sources = manifest["aggregate_ranks"], manifest["rows"], manifest["source_ids"]
    if (not isinstance(ranks, list) or not ranks or len(set(ranks)) != len(ranks)
            or any(rank not in RANKS for rank in ranks)
            or not fixture_only and tuple(ranks) != REVIEW_RANKS
            or not isinstance(rows, list) or not rows
            or not isinstance(sources, list) or not sources or len(set(sources)) != len(sources)):
        raise ValueError("Diagnostics require frozen ranks and nonempty unique source membership")
    metadata = {"row_key", "source_url", "mover", "group_mover", "group_opponent",
                "source_rank_mover", "source_rank_opponent", "ply", "nominal_clock"}
    if (any(not isinstance(row, dict) or not metadata <= row.keys()
            or row.get("role", manifest["role"]) != manifest["role"] for row in rows)
            or {row["source_url"] for row in rows} != set(sources)
            or len({row["row_key"] for row in rows}) != len(rows)
            or not set(ranks) <= {row["source_rank_mover"] for row in rows}
            or any(row["source_rank_mover"] not in RANKS for row in rows)):
        raise ValueError("Diagnostic rows must retain distinct keys, declared roles and every rank")
    arrays_ref = manifest["arrays"]
    if (not isinstance(arrays_ref, dict) or set(arrays_ref) != {"path", "sha256"}
            or file_hash(_path(arrays_ref)) != arrays_ref["sha256"]):
        raise ValueError("Diagnostic arrays differ from their pin")
    schema = {**array_schema(0), **LABEL_SCHEMA}
    with np.load(_path(arrays_ref), allow_pickle=False) as saved:
        if set(saved.files) != set(schema):
            raise ValueError("Diagnostic arrays require only shared inputs and masked labels")
        arrays = {name: saved[name] for name in schema}
    for name, (dtype, shape) in schema.items():
        if arrays[name].shape != (len(rows), *shape) or arrays[name].dtype != np.dtype(dtype):
            raise ValueError(f"Invalid diagnostic array shape or dtype: {name}")
        arrays[name].flags.writeable = False
    return Diagnostic({name: arrays[name] for name in INPUT_KEYS},
                      {name: arrays[name] for name in LABEL_SCHEMA}, tuple(rows), tuple(ranks),
                      purpose, ref["sha256"])


def evaluate_epoch(model, diagnostic, *, batch_size=128, device=None, check_budget=lambda: None,
                   include_predictions=False):
    """Measure one fixed checkpoint without changing modes, buffers or training RNGs."""
    if not isinstance(diagnostic, Diagnostic) or type(batch_size) is not int or batch_size < 1:
        raise ValueError("Epoch evaluation requires a pinned diagnostic and positive batch size")
    device = next(model.parameters()).device if device is None else torch.device(device)
    modes = [(module, module.training) for module in model.modules()]
    random_state = rng_state()
    predictions, rows = [], []
    try:
        model.eval()
        with torch.inference_mode():
            for offset in range(0, len(diagnostic.rows), batch_size):
                check_budget()
                stop = offset + batch_size
                inputs = {name: torch.from_numpy(value[offset:stop].copy()).to(device)
                          for name, value in diagnostic.inputs.items()}
                logits, values = model(inputs)
                logits, values = logits.cpu(), values.cpu()
                for index, metadata in enumerate(diagnostic.rows[offset:stop], start=offset):
                    legal = diagnostic.inputs["legal"][index]
                    targets = {name: value[index].item() for name, value in diagnostic.labels.items()}
                    row = {**metadata, **targets, "legal": legal.tolist()}
                    rows.append(row)
                    predictions.append(prediction_from_logits(row["row_key"], logits[index-offset].tolist(),
                                       float(values[index-offset]), legal))
        check_budget()
        all_rows = len(rows)
        original_rows, original_predictions = rows, predictions
        all_rank_report = evaluate(rows, predictions, aggregate_ranks=RANKS)
        selected = [(row, prediction) for row, prediction in zip(rows, predictions)
                    if row["source_rank_mover"] in diagnostic.ranks]
        rows, predictions = [item[0] for item in selected], [item[1] for item in selected]
        report = evaluate(rows, predictions, aggregate_ranks=diagnostic.ranks)
        metrics = {}
        for name in ("nll", "top1", "value_mse"):
            metric = report["overall"]["metrics"][name]
            value = metric["estimate"]["value"]
            if metric["status"] != "available" or type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"Every frozen rank must have finite supported {name}")
            metrics[name] = value
        result = {**metrics, "rows": len(rows), "evaluated_rows": all_rows,
                "excluded_rank_rows": all_rows - len(rows), "diagnostic_sha256": diagnostic.sha256,
                "aggregate_ranks": list(diagnostic.ranks),
                "weighting": "equal ranks, equal game sides within rank, equal valid moves within side",
                "policy_count": sum(row["policy_valid"] for row in rows),
                "value_count": sum(row["value_valid"] for row in rows),
                "by_rank": {rank: {name: group["metrics"][name]["estimate"]
                                    for name in metrics} for rank, group in report["by_rank"].items()},
                "all_rank_metrics": all_rank_report["overall"]["metrics"],
                "value_calibration": report["overall"]["value_calibration"],
                "scope": diagnostic.purpose}
        if include_predictions:
            result["report"] = report
            result["all_rank_report"] = all_rank_report
            result["predictions"] = [{**{key: value for key, value in row.items() if key != "legal"},
                "action": row["action"] if row["policy_valid"] else None,
                "value": row["value"] if row["value_valid"] else None,
                "nll": -prediction.log_policy[row["action"]] if row["policy_valid"] else None,
                "top1": float(max(range(362), key=prediction.log_policy.__getitem__) == row["action"])
                    if row["policy_valid"] else None,
                "prediction_value": prediction.value,
                "value_mse": (prediction.value - row["value"]) ** 2 if row["value_valid"] else None}
                for row, prediction in zip(original_rows, original_predictions)]
        check_budget()
        return result
    finally:
        for module, mode in modes:
            module.training = mode
        restore_rng(random_state)


def evaluate_diagnostic(model, ref, *, normalizer_sha256, device=None, batch_size=128,
                        purpose="epoch_validation", fixture_only=False, include_predictions=False,
                        check_budget=lambda: None):
    diagnostic = load_diagnostic(ref, purpose, normalizer_sha256, fixture_only=fixture_only)
    return evaluate_epoch(model, diagnostic, device=device, batch_size=batch_size,
                          include_predictions=include_predictions, check_budget=check_budget)


def observe_validation(stopping, score, *, min_delta=0.002):
    """Track the raw minimum separately from the meaningful-improvement reference."""
    if type(score) not in (int, float) or not math.isfinite(score):
        raise ValueError("Stopping requires a finite validation score")
    result = deepcopy(stopping)
    best = result.get("best_score") is None or score < result["best_score"]
    meaningful = result.get("tolerance_reference") is None or score <= result["tolerance_reference"] - min_delta
    if best:
        result["best_score"] = score
    if meaningful:
        result["tolerance_reference"] = score
        result["bad_rounds"] = 0
    else:
        result["bad_rounds"] += 1
    return result, best, meaningful


class BudgetStop(Exception):
    pass


class _Budget:
    def __init__(self, deadline, reserve):
        end = datetime.fromisoformat(deadline)
        if end.tzinfo is None:
            raise ValueError("The absolute deadline requires a timezone")
        self.end, self.reserve = end.timestamp(), reserve
        self.monotonic_end = time.monotonic() + self.end - time.time()

    def check(self):
        if min(self.end - time.time(), self.monotonic_end - time.monotonic()) <= self.reserve:
            raise BudgetStop("Checkpoint and shutdown reserve reached")


def _parameters(config, fixture_only):
    raw = config["parameters"]
    required = {"recipe", "stage_id", "round_updates", "model", "training_rows", "device",
                "deadline_utc", "checkpoint_reserve_seconds"}
    if not required <= raw.keys() or raw.keys() - required - set(DEFAULTS) - {"max_artifact_bytes"}:
        raise ValueError("Unexpected continuation training parameters")
    params = DEFAULTS | raw
    model = ModelConfig(**params["model"])
    counts = ("round_updates", "training_rows", "batch_size", "recovery_updates", "patience",
              "max_rounds", "eval_batch_size")
    if (params["recipe"] != RECIPE or not isinstance(params["stage_id"], str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", params["stage_id"]) is None
            or any(type(params[name]) is not int or params[name] < 1 for name in counts)
            or type(params["evaluate_only"]) is not bool or params["model"] != model.__dict__
            or params["target_round"] is not None and (type(params["target_round"]) is not int
                or not 0 <= params["target_round"] <= 1000
                or params["evaluate_only"] != (params["target_round"] == 0))
            or params["learning_rate"] != 0.001 or params["clock_dropout"] != 0.1
            or params["min_delta"] != 0.002 or params["patience"] != 3 or params["max_rounds"] != 1000
            or params["inference_contract"] != "komi_comparison_v1"
            or params["training_komi_halfpoints"] != [1, 13]
            or any(type(value) is not int for value in params["training_komi_halfpoints"])
            or type(params["checkpoint_reserve_seconds"]) not in (int, float)
            or not math.isfinite(params["checkpoint_reserve_seconds"])
            or params["checkpoint_reserve_seconds"] <= 0):
        raise ValueError("Continuation requires its frozen learning and stopping recipe")
    if fixture_only:
        if (params["device"] != "cpu" or params["training_rows"] > 64
                or params["round_updates"] > 64 or params["batch_size"] > 64
                or model.blocks != 1 or model.channels != 16):
            raise ValueError("Fixtures require bounded CPU data and a tiny model")
    elif (params["device"] != "cuda" or params["batch_size"] != 128
            or params["recovery_updates"] != 1000 or params["eval_batch_size"] > 128
            or model.blocks != 8 or model.channels != 128 or not model.rank_conditioned):
        raise ValueError("Production continuation requires its frozen CUDA model and batching")
    return params, model


def _contract(config, params, data):
    stable = {name: value for name, value in params.items()
              if name not in {"evaluate_only", "target_round", "deadline_utc", "checkpoint_reserve_seconds", "max_artifact_bytes"}}
    return json_hash({"parameters": stable, "seed": config["seed"], "arm": config["arm"],
                      "examples_sha256": data.examples_sha256,
                      "normalizer_sha256": data.normalizer_sha256, "split_sha256": data.split_sha256,
                      "row_keys_sha256": json_hash(list(data.row_keys)),
                      "diagnostics": {name: config["inputs"][name]["sha256"]
                                      for name in ("train_monitor", "epoch_validation")}})


def _empty_aggregate():
    return {"updates": 0, "presented_examples": 0, "policy_count": 0, "value_count": 0,
            "clock_missing_count": 0, "policy_sum": 0.0, "value_sum": 0.0}


def _parent_bundle(inputs):
    return {**inputs["parent_checkpoint"], "path": str(_path(inputs["parent_checkpoint"])),
            "config": inputs["parent_config"], "provenance": inputs["parent_provenance"]}


def _initial_progress(parent, parent_bundle, params, config, contract):
    progress = parent["progress"]
    if (progress.get("version") == RECIPE and progress.get("stage_id") == params["stage_id"]
            and progress.get("arm") == config["arm"]):
        if progress.get("stage_contract_sha256") != contract:
            raise ValueError("A resumed stage changed data, diagnostics, sampler or learning settings")
        result = deepcopy(progress)
        integer_keys = ("sampler_step", "lifetime_updates", "lifetime_examples", "completed_rounds")
        aggregate = result.get("partial_round", {})
        if (any(type(result.get(key)) is not int or result[key] < 0 for key in integer_keys)
                or result["lifetime_updates"] != parent["step"]
                or type(aggregate.get("updates")) is not int
                or not 0 <= aggregate["updates"] <= params["round_updates"]
                or result["sampler_step"] != result["completed_rounds"] * params["round_updates"] + aggregate["updates"]
                or aggregate.get("presented_examples") != aggregate["updates"] * params["batch_size"]
                or len(result.get("epoch_metrics", [])) != result["completed_rounds"]):
            raise ValueError("A resumed checkpoint has inconsistent sampler or round progress")
        if result["best_checkpoint"] == "self":
            result["best_checkpoint"] = {**parent_bundle, "step": parent["step"], "kind": "epoch"}
        return result
    lifetime = progress.get("lifetime_updates", progress.get("completed_updates", parent["step"]))
    examples = progress.get("lifetime_examples", progress.get("presented_examples"))
    if type(lifetime) is not int or lifetime != parent["step"] or type(examples) is not int or examples < 0:
        raise ValueError("A parent checkpoint requires consistent lifetime update and presentation counts")
    return {"version": RECIPE, "stage_id": params["stage_id"], "arm": config["arm"],
            "stage_contract_sha256": contract, "parent_checkpoint": parent_bundle,
            "sampler_step": 0, "lifetime_updates": lifetime, "lifetime_examples": examples,
            "completed_rounds": 0, "initial_evaluation_complete": False,
            "partial_round": _empty_aggregate(), "epoch_metrics": [], "last_epoch_metrics": None,
            "stopping": {"best_score": None, "tolerance_reference": None, "bad_rounds": 0},
            "best_checkpoint": None, "best_round": None, "best_sampler_step": None,
            "best_lifetime_updates": None}


def run(config_path, config_pin, provenance_path, provenance_pin, output, *, fixture_only=False):
    """Run an initial evaluation or finish exactly one matched round in a fresh job."""
    output = Path(output)
    if (type(fixture_only) is not bool or output.is_symlink() or not output.is_dir()
            or any(os.path.lexists(output / name) for name in
                   ("training_progress.json", "training_summary.json", "training_failure.json", "checkpoints", "commits"))):
        raise ValueError("Continuation requires an unused supervisor-owned job directory")
    output = output.resolve()
    started = time.monotonic()
    state = {"version": RECIPE, "status": "preparing", "checkpoints": [], "commits": [],
             "attempted_updates": 0, "completed_updates": 0, "uncertain_updates": 0}
    progress, model, optimizer, config, params = None, None, None, None, None

    def persist():
        state["wall_seconds"] = time.monotonic() - started
        if progress is not None:
            state["progress"] = deepcopy(progress)
        save_json(output / "training_progress.json", state, replace=True)

    def checkpoint(kind, *, best=False):
        if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
            raise ValueError("Nonfinite model state cannot become a recovery checkpoint")
        saved = deepcopy(progress)
        if best:
            saved["best_checkpoint"] = "self"
        ref = save_checkpoint(output / "checkpoints", model, optimizer,
            step=progress["lifetime_updates"], progress=saved,
            config_sha256=json_hash(config), provenance_sha256=provenance_pin)
        ref = {**ref, "path": str(Path(ref["path"]).resolve()),
               "config": _reference(config_path), "provenance": _reference(provenance_path),
               "step": progress["lifetime_updates"], "kind": kind}
        state["checkpoints"].append(ref)
        state["latest_checkpoint"] = ref
        if best:
            progress["best_checkpoint"] = ref
        persist()
        return ref

    def commit_epoch(metrics, *, initial=False):
        score = metrics["validation"]["nll"]
        stopping, best, meaningful = observe_validation(progress["stopping"], score,
                                                        min_delta=params["min_delta"])
        progress["stopping"] = stopping
        round_index = progress["completed_rounds"] + (0 if initial else 1)
        if best:
            progress.update(best_round=round_index, best_sampler_step=progress["sampler_step"],
                            best_lifetime_updates=progress["lifetime_updates"])
        metrics.update(round=round_index, sampler_step=progress["sampler_step"],
                       lifetime_updates=progress["lifetime_updates"],
                       meaningful_improvement=meaningful, new_raw_best=best)
        path = output / f"epoch-{round_index:04d}-metrics.json"
        save_json(path, metrics)
        progress["last_epoch_metrics"] = _reference(path)
        progress["initial_evaluation_complete"] = True
        progress["completed_rounds"] = round_index
        if not initial:
            online = deepcopy(progress["partial_round"])
            online.update(epoch=round_index, next_step=progress["sampler_step"],
                policy_loss=online["policy_sum"] / online["policy_count"] if online["policy_count"] else None,
                value_loss=online["value_sum"] / online["value_count"] if online["value_count"] else None)
            progress["epoch_metrics"].append(online)
            progress["partial_round"] = _empty_aggregate()
        ref = checkpoint("epoch", best=best)
        record = {"version": "continuation_epoch_commit_v1", "round": round_index,
                  "checkpoint": ref, "metrics": progress["last_epoch_metrics"],
                  "best_checkpoint": progress["best_checkpoint"], "stopping": stopping}
        path = output / "commits" / f"epoch-{round_index:04d}.json"
        save_json(path, record)
        state["commits"].append(_reference(path))
        persist()

    def commit(metrics, *, initial=False):
        before, saved_count = deepcopy(progress), len(state["checkpoints"])
        try:
            commit_epoch(metrics, initial=initial)
        except BaseException:
            if len(state["checkpoints"]) == saved_count:
                progress.clear()
                progress.update(before)
            raise

    def finish(status, reason=None):
        if status == "completed" and provenance["source"] != _source_identity():
            raise ValueError("Worker source changed during continuation")
        state.update(status=status, reason=reason,
            best_checkpoint=progress["best_checkpoint"] if progress else None,
            final_checkpoint=progress["best_checkpoint"] if progress else None,
            checkpoint_selection="minimum_epoch_validation_nll",
            patience_exhausted=bool(progress and progress["stopping"]["bad_rounds"] >= params["patience"]))
        if progress:
            state.update(epoch_metrics=progress["epoch_metrics"], training_rows=params["training_rows"],
                         presented_examples=state["completed_updates"] * params["batch_size"])
        persist()
        save_json(output / "training_summary.json", state)
        return state

    persist()
    try:
        config, provenance = read_json(config_path, config_pin), read_json(provenance_path, provenance_pin)
        validate_config(config)
        params, model_config = _parameters(config, fixture_only)
        if (config["purpose"] != "continuation-training" or provenance.get("version") != "experiment_provenance_v1"
                or provenance.get("inputs") != config["inputs"] or json_hash(provenance) != provenance_pin
                or provenance.get("source") != _source_identity()):
            raise ValueError("Continuation source, inputs and canonical provenance must agree")
        required = {"data", "split", "rules", "examples", "normalizer", "parent_checkpoint", "parent_config",
                    "parent_provenance", "train_monitor", "epoch_validation"}
        if not required <= config["inputs"].keys():
            raise ValueError("Continuation requires pinned training, parent and diagnostic inputs")
        for ref in config["inputs"].values():
            if (not isinstance(ref, dict) or set(ref) != {"path", "sha256"}
                    or file_hash(_path(ref)) != ref["sha256"]):
                raise ValueError("A continuation input differs from its independent pin")
        budget = _Budget(params["deadline_utc"], params["checkpoint_reserve_seconds"])
        torch.set_num_threads(1)
        device = torch.device("cuda:0" if params["device"] == "cuda" else params["device"])
        if device.type == "cuda" and (not torch.cuda.is_available()
                or os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}):
            raise ValueError("CUDA continuation requires the pinned deterministic device setup")
        state["runtime"] = {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "cuda_build": torch.version.cuda, "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "torch_cpu_threads": torch.get_num_threads()}
        if not fixture_only and (platform.python_version() != "3.12.13" or torch.__version__ != "2.15.0+cu132"
                                 or np.__version__ != "2.5.4"):
            raise ValueError("Continuation runtime differs from the pinned CUDA environment")
        data = load_training(config["inputs"], fixture_only=fixture_only)
        if len(data) != params["training_rows"]:
            raise ValueError("Training rows differ from the frozen stage")
        diagnostics = {name: load_diagnostic(config["inputs"][name], name, data.normalizer_sha256,
                                            fixture_only=fixture_only)
                       for name in ("train_monitor", "epoch_validation")}
        training_keys = set(data.row_keys)
        monitor_keys = {row["row_key"] for row in diagnostics["train_monitor"].rows}
        validation_keys = {row["row_key"] for row in diagnostics["epoch_validation"].rows}
        if not monitor_keys <= training_keys or validation_keys & training_keys:
            raise ValueError("Train monitoring must use training rows and validation must remain outside training")
        if ({row["source_url"] for row in diagnostics["train_monitor"].rows}
                & {row["source_url"] for row in diagnostics["epoch_validation"].rows}):
            raise ValueError("Training and validation diagnostics cannot share games")
        del training_keys
        model = PopulationModel(model_config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=0)
        parent_bundle = _parent_bundle(config["inputs"])
        parent_config = read_json(_path(parent_bundle["config"]), parent_bundle["config"]["sha256"])
        parent_provenance = read_json(_path(parent_bundle["provenance"]), parent_bundle["provenance"]["sha256"])
        if (parent_config["parameters"]["model"] != params["model"]
                or parent_provenance.get("inputs") != parent_config.get("inputs")):
            raise ValueError("The parent model and input provenance differ")
        parent = load_checkpoint(parent_bundle["path"], parent_bundle["sha256"], model, optimizer,
            config_sha256=json_hash(parent_config), provenance_sha256=json_hash(parent_provenance))
        parent_bundle["step"] = parent["step"]
        if any(group["lr"] != params["learning_rate"] or group.get("weight_decay", 0) != 0
               for group in optimizer.param_groups):
            raise ValueError("The parent optimizer differs from the frozen Adam recipe")
        progress = _initial_progress(parent, parent_bundle, params, config, _contract(config, params, data))
        (output / "commits").mkdir()
        if progress["last_epoch_metrics"] is not None:
            ref = progress["last_epoch_metrics"]
            read_json(_path(ref), ref["sha256"])
        checkpoint("initial_recovery")

        def measurements():
            state["status"] = "validating"
            persist()
            return {"training_monitor": evaluate_epoch(model, diagnostics["train_monitor"],
                        batch_size=params["eval_batch_size"], device=device, check_budget=budget.check),
                    "validation": evaluate_epoch(model, diagnostics["epoch_validation"],
                        batch_size=params["eval_batch_size"], device=device, check_budget=budget.check),
                    "online_training": deepcopy(progress["partial_round"])}

        if not progress["initial_evaluation_complete"]:
            commit(measurements(), initial=True)
        if params["evaluate_only"]:
            if progress["completed_rounds"] or progress["partial_round"]["updates"]:
                raise ValueError("Epoch-zero evaluation cannot replace a later or partial training round")
            return finish("completed", "initial_evaluation")
        target = params["target_round"]
        if target is not None:
            if progress["completed_rounds"] == target:
                if progress["partial_round"]["updates"] or progress["last_epoch_metrics"] is None:
                    raise ValueError("A completed target cannot contain a partial later round")
                path = output / "commits" / f"epoch-{target:04d}.json"
                save_json(path, {"version": "continuation_epoch_commit_v1", "round": target,
                    "checkpoint": state["latest_checkpoint"], "metrics": progress["last_epoch_metrics"],
                    "best_checkpoint": progress["best_checkpoint"], "stopping": progress["stopping"],
                    "recovered_completed_round": True})
                state["commits"].append(_reference(path))
                return finish("completed", "recovered_matched_round")
            if progress["completed_rounds"] + 1 != target:
                raise ValueError("The requested round does not follow the parent cursor")
        if progress["completed_rounds"] >= params["max_rounds"]:
            return finish("completed", "maximum_rounds")
        while progress["partial_round"]["updates"] < params["round_updates"]:
            budget.check()
            inputs, labels = data.batch(progress["sampler_step"], params["batch_size"], config["seed"],
                                        device=device, full_batch=True)
            state.update(status="training", attempted_updates=state["attempted_updates"] + 1, uncertain_updates=1)
            persist()
            metrics = train_step(model, optimizer, inputs, labels, clock_dropout=params["clock_dropout"])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if (metrics["examples"] != params["batch_size"]
                    or any(not math.isfinite(metrics[name]) for name in ("loss", "policy_loss", "value_loss"))):
                raise ValueError("Training returned nonfinite losses or incomplete batch counts")
            progress["sampler_step"] += 1
            progress["lifetime_updates"] += 1
            progress["lifetime_examples"] += metrics["examples"]
            aggregate = progress["partial_round"]
            aggregate["updates"] += 1
            aggregate["presented_examples"] += metrics["examples"]
            for name in ("policy_count", "value_count", "clock_missing_count"):
                aggregate[name] += metrics[name]
            aggregate["policy_sum"] += metrics["policy_loss"] * metrics["policy_count"]
            aggregate["value_sum"] += metrics["value_loss"] * metrics["value_count"]
            state.update(completed_updates=state["completed_updates"] + 1, uncertain_updates=0)
            persist()
            if progress["sampler_step"] % params["recovery_updates"] == 0:
                checkpoint("periodic_recovery")
        checkpoint("before_validation")
        commit(measurements())
        if provenance["source"] != _source_identity():
            raise ValueError("Worker source changed during continuation")
        return finish("completed", "matched_round")
    except BudgetStop as error:
        if progress is not None and not state["uncertain_updates"]:
            checkpoint("budget_recovery")
        return finish("partial", str(error))
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__, reason=str(error))
        persist()
        save_json(output / "training_failure.json", state)
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
    run(args.config, args.config_sha256, args.provenance, args.provenance_sha256, args.output,
        fixture_only=args.fixture_only)
