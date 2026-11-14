"""Finite BASE training worker; the caller owns supervision, tracking and costs."""

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import time

import torch

from kenggo.checkpoints import (file_hash, json_bytes, json_hash, read_json, save_checkpoint,
                                save_json, seed_all)
from kenggo.learner import ModelConfig, PopulationModel
from kenggo.supervised import TrainingData, train_step
from kenggo.tracking import validate_config


ROOT = Path(__file__).resolve().parents[2]
RECIPE = {"epochs": 20, "batch_size": 128, "learning_rate": 0.001, "clock_dropout": 0.1,
          "training_rows": 22916, "save_epochs": [0, 10, 20], "device": "cuda", "torch_threads": 1}
OUTPUTS = ("training_progress.json", "training_epochs.json", "training_summary.json",
           "training_failure.json", "checkpoints")


def _source_identity():
    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(ROOT), *arguments], timeout=10)
    if git("ls-files", "--others", "--exclude-standard").strip():
        raise ValueError("Stage intended new source files before running a pinned worker")
    return {"commit": git("rev-parse", "HEAD").decode().strip(),
            "patch_sha256": sha256(git("diff", "--binary", "HEAD")).hexdigest()}


def _validate(config, provenance, provenance_pin, fixture_only):
    validate_config(config)
    if config["purpose"] != "baseline-training":
        raise ValueError("The worker requires a baseline-training configuration")
    parameters = config["parameters"]
    if (not (set(RECIPE) | {"model"}) <= parameters.keys()
            or parameters.keys() - (set(RECIPE) | {"model", "max_artifact_bytes"})):
        raise ValueError("Unexpected baseline training parameters")
    model_config = ModelConfig(**parameters["model"])
    if parameters["model"] != model_config.__dict__:
        raise ValueError("Pin every model parameter explicitly")
    for name in ("epochs", "batch_size", "training_rows", "torch_threads"):
        if type(parameters[name]) is not int or parameters[name] < 1:
            raise ValueError("Training counts must be positive integers")
    for name in ("learning_rate", "clock_dropout"):
        if type(parameters[name]) not in (int, float) or not math.isfinite(parameters[name]):
            raise ValueError("Training rates must be finite numbers")
    saves = parameters["save_epochs"]
    if (not isinstance(saves, list) or any(type(epoch) is not int for epoch in saves)
            or saves != sorted(set(saves)) or not saves or saves[0] != 0
            or saves[-1] != parameters["epochs"]
            or parameters["learning_rate"] != RECIPE["learning_rate"]
            or parameters["clock_dropout"] != RECIPE["clock_dropout"]
            or parameters["torch_threads"] != 1):
        raise ValueError("Training requires its frozen rates and initial/final checkpoint epochs")
    if fixture_only:
        if (parameters["device"] != "cpu" or parameters["epochs"] > 4
                or parameters["training_rows"] > 64 or parameters["batch_size"] > 64
                or (model_config.blocks, model_config.channels) != (1, 16)):
            raise ValueError("Fixture training requires a tiny, bounded CPU configuration")
    elif (any(parameters[key] != value for key, value in RECIPE.items())
          or (model_config.blocks, model_config.channels) != (8, 128)
          or config["seed"] != 20261114):
        raise ValueError("Production training requires the frozen BASE pilot recipe")
    if (not isinstance(provenance, dict) or provenance.get("version") != "experiment_provenance_v1"
            or provenance.get("inputs") != config["inputs"]
            or not isinstance(provenance.get("environment"), dict)
            or json_hash(provenance) != provenance_pin):
        raise ValueError("Canonical provenance must bind the configured inputs")
    source = provenance.get("source", {})
    if (not isinstance(source, dict) or set(source) != {"commit", "patch_sha256"}
            or not isinstance(source["commit"], str)
            or re.fullmatch(r"[a-f0-9]{40}", source["commit"]) is None
            or not isinstance(source["patch_sha256"], str)
            or re.fullmatch(r"[a-f0-9]{64}", source["patch_sha256"]) is None
            or source != _source_identity()):
        raise ValueError("Worker source differs from the pinned provenance")
    refs = config["inputs"]
    if not {"data", "split", "rules", "examples", "normalizer", "cpu_environment", "cuda_environment"} <= refs.keys():
        raise ValueError("Training requires pinned data, rules, examples, normalizer and environments")
    for name, ref in refs.items():
        if ("checkpoint" in name.lower() or not isinstance(ref, dict)
                or set(ref) != {"path", "sha256"} or not isinstance(ref["path"], str)
                or not ref["path"] or not isinstance(ref["sha256"], str)
                or re.fullmatch(r"[a-f0-9]{64}", ref["sha256"]) is None
                or file_hash(ROOT / ref["path"]) != ref["sha256"]):
            raise ValueError("Training inputs must be pinned artifacts, without checkpoint initialization")
    if Path(refs["examples"]["path"]).name != "manifest.json":
        raise ValueError("The examples input must name its manifest.json")
    manifest = read_json(ROOT / refs["examples"]["path"], refs["examples"]["sha256"])
    if (manifest["fixture_only"] is not fixture_only
            or type(manifest["counts"]["train"]["examples"]) is not int
            or manifest["counts"]["train"]["examples"] != parameters["training_rows"]):
        raise ValueError("The full training population must match the pinned count and fixture scope")
    return model_config


def tensor_content_hash(model):
    """Hash named tensor values and shapes, independently of config and serialization."""
    digest = sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        metadata = json_bytes({"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)})
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _environment(device):
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "torch": str(torch.__version__), "torch_cuda_build": torch.version.cuda,
              "device": str(device), "torch_threads": torch.get_num_threads(),
              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
              "cudnn_deterministic": torch.backends.cudnn.deterministic,
              "cudnn_benchmark": torch.backends.cudnn.benchmark,
              "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
              "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
              "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
    if device.type == "cuda":
        result.update(cuda_device_name=torch.cuda.get_device_name(device),
                      cudnn_version=torch.backends.cudnn.version())
    return result


def run_training(config_path, config_pin, provenance_path, provenance_pin, output, *, fixture_only=False):
    """Start from random weights exactly once in an existing, caller-owned job directory."""
    if type(fixture_only) is not bool:
        raise ValueError("Fixture scope must be explicit")
    output = Path(output)
    if output.is_symlink() or not output.is_dir():
        raise ValueError("The supervisor must create the job output directory")
    output = output.resolve()
    if any(os.path.lexists(output / name) for name in OUTPUTS):
        raise ValueError("A training worker never resets or retries an existing output")
    started, cpu_started = time.monotonic(), time.process_time()
    state = {"version": "baseline_training_progress_v1", "status": "preparing",
             "fixture_only": fixture_only, "config_file_sha256": config_pin,
             "provenance_sha256": provenance_pin, "attempted_updates": 0, "completed_updates": 0,
             "attempted_examples": 0, "presented_examples": 0, "next_step": 0, "epoch": 0,
             "batch_in_epoch": 0, "uncertain_updates": 0, "uncertain_examples": 0,
             "last_loss": None, "checkpoints": [], "epoch_metrics": [],
             "phase_seconds": {"load": 0.0, "initialize": 0.0, "train": 0.0, "checkpoint": 0.0}}
    save_json(output / "training_progress.json", state)
    device, cuda_ready = None, False

    def persist():
        state["wall_seconds"] = time.monotonic() - started
        state["process_cpu_seconds"] = time.process_time() - cpu_started
        if cuda_ready:
            state["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            state["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        else:
            state["peak_cuda_allocated_bytes"] = state["peak_cuda_reserved_bytes"] = 0
        save_json(output / "training_progress.json", state, replace=True)

    try:
        config = read_json(config_path, config_pin)
        provenance = read_json(provenance_path, provenance_pin)
        model_config = _validate(config, provenance, provenance_pin, fixture_only)
        parameters, refs = config["parameters"], config["inputs"]
        state.update(config_sha256=json_hash(config), source=provenance["source"],
                     seed=config["seed"], parameters=parameters, status="loading")
        persist()
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        device = torch.device("cuda:0" if parameters["device"] == "cuda" else parameters["device"])
        if device.type == "cuda":
            if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
                raise ValueError("The supervisor must set deterministic CUBLAS_WORKSPACE_CONFIG")
            if not torch.cuda.is_available():
                raise RuntimeError("The pinned CUDA device is unavailable; no CPU fallback is allowed")
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
            cuda_ready = True
        state["environment"] = _environment(device)
        phase = time.monotonic()
        data = TrainingData.load(ROOT / Path(refs["examples"]["path"]).parent, refs["examples"]["sha256"],
            ROOT / refs["normalizer"]["path"], refs["normalizer"]["sha256"],
            split_sha256=refs["split"]["sha256"])
        state["phase_seconds"]["load"] = time.monotonic() - phase
        if len(data) != parameters["training_rows"]:
            raise ValueError("Loaded training rows differ from the frozen population")
        batches = (len(data) + parameters["batch_size"] - 1) // parameters["batch_size"]
        state.update(training_rows=len(data), row_keys_sha256=json_hash(list(data.row_keys)),
                     batches_per_epoch=batches, planned_updates=batches * parameters["epochs"],
                     planned_examples=len(data) * parameters["epochs"])
        phase = time.monotonic()
        seed_all(config["seed"])
        model = PopulationModel(model_config)
        state["initial_tensor_sha256"] = tensor_content_hash(model)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=parameters["learning_rate"], weight_decay=0)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        state["phase_seconds"]["initialize"] = time.monotonic() - phase
        (output / "checkpoints").mkdir()

        def checkpoint(epoch):
            state["status"] = "checkpointing"
            persist()
            phase = time.monotonic()
            progress = {key: state[key] for key in ("next_step", "epoch", "batch_in_epoch",
                        "completed_updates", "presented_examples", "row_keys_sha256", "initial_tensor_sha256")}
            ref = save_checkpoint(output / "checkpoints", model, optimizer, step=state["next_step"],
                progress=progress, config_sha256=state["config_sha256"], provenance_sha256=provenance_pin)
            ref = {"path": str(Path(ref["path"]).resolve().relative_to(output)), "sha256": ref["sha256"],
                   "epoch": epoch, "step": state["next_step"]}
            state["checkpoints"].append(ref)
            state["phase_seconds"]["checkpoint"] += time.monotonic() - phase
            persist()
            return ref

        checkpoint(0)
        save_json(output / "training_epochs.json", {"scope": "training_minibatches", "epochs": []})
        for epoch in range(parameters["epochs"]):
            aggregate = {"epoch": epoch + 1, "updates": 0, "presented_examples": 0,
                         "policy_count": 0, "value_count": 0, "clock_missing_count": 0}
            policy_sum = value_sum = 0.0
            order = sha256()
            for batch in range(batches):
                step = epoch * batches + batch
                phase = time.monotonic()
                indices = data.indices(step, parameters["batch_size"], config["seed"])
                inputs, labels = data.batch(step, parameters["batch_size"], config["seed"], device=device)
                state.update(status="training", attempted_updates=step + 1,
                             attempted_examples=state["attempted_examples"] + len(indices),
                             uncertain_updates=1, uncertain_examples=len(indices))
                persist()
                metrics = train_step(model, optimizer, inputs, labels, clock_dropout=parameters["clock_dropout"])
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                if (metrics["examples"] != len(indices) or any(not math.isfinite(metrics[key])
                        for key in ("loss", "policy_loss", "value_loss"))):
                    raise ValueError("A training update returned invalid counts or nonfinite losses")
                state["phase_seconds"]["train"] += time.monotonic() - phase
                state.update(completed_updates=step + 1, next_step=step + 1,
                             presented_examples=state["presented_examples"] + metrics["examples"],
                             epoch=(step + 1) // batches, batch_in_epoch=(step + 1) % batches,
                             uncertain_updates=0, uncertain_examples=0, last_loss=metrics)
                aggregate["updates"] += 1
                aggregate["presented_examples"] += metrics["examples"]
                for name in ("policy_count", "value_count", "clock_missing_count"):
                    aggregate[name] += metrics[name]
                policy_sum += metrics["policy_loss"] * metrics["policy_count"]
                value_sum += metrics["value_loss"] * metrics["value_count"]
                for index in indices:
                    order.update(json_bytes(data.row_keys[index]))
                persist()
            aggregate.update(next_step=state["next_step"], order_sha256=order.hexdigest(),
                             policy_loss=policy_sum / aggregate["policy_count"] if aggregate["policy_count"] else None,
                             value_loss=value_sum / aggregate["value_count"] if aggregate["value_count"] else None)
            state["epoch_metrics"].append(aggregate)
            save_json(output / "training_epochs.json", {"scope": "training_minibatches",
                      "epochs": state["epoch_metrics"]}, replace=True)
            persist()
            if epoch + 1 in parameters["save_epochs"]:
                checkpoint(epoch + 1)
        if provenance["source"] != _source_identity():
            raise ValueError("Worker source changed during training")
        state.update(status="completed", final_checkpoint=state["checkpoints"][-1],
                     checkpoint_selection="fixed_final_epoch", scope="training_minibatches")
        persist()
        result = dict(state, version="baseline_training_v1")
        save_json(output / "training_summary.json", result)
        return result
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
    arguments = parser.parse_args()
    print(json.dumps(run_training(arguments.config, arguments.config_sha256, arguments.provenance,
                                 arguments.provenance_sha256, arguments.output,
                                 fixture_only=arguments.fixture_only), indent=2))
