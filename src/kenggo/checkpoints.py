"""Immutable training checkpoints with explicit hashes and complete resume state."""

from hashlib import sha256
import json
import os
from pathlib import Path
import random
import re
import tempfile
import uuid

import numpy as np
import torch


VERSION = "training_checkpoint_v2"


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def json_hash(value):
    return sha256(json_bytes(value)).hexdigest()


def file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value, *, replace=False):
    path = Path(path)
    data = json_bytes(value)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".writing-")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return sha256(data).hexdigest()


def reject_constant(value):
    raise ValueError(f"Nonfinite JSON number: {value}")


def read_json(path, expected_sha256):
    if (not isinstance(expected_sha256, str) or re.fullmatch(r"[a-f0-9]{64}", expected_sha256) is None
            or file_hash(path) != expected_sha256):
        raise ValueError("Artifact differs from the caller's expected hash")
    return json.loads(Path(path).read_text(), parse_constant=reject_constant)


def seed_all(seed):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an unsigned 32-bit integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rng_state():
    numpy = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy[0], numpy[1].tolist(), int(numpy[2]), int(numpy[3]), float(numpy[4])],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }


def restore_rng(state):
    numpy = state["numpy"]
    random.setstate(state["python"])
    np.random.set_state((numpy[0], np.array(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    torch.use_deterministic_algorithms(state["deterministic"], warn_only=state["deterministic_warn_only"])
    torch.backends.cudnn.deterministic = state["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = state["cudnn_benchmark"]
    torch.backends.cudnn.allow_tf32 = state["cudnn_allow_tf32"]
    torch.backends.cuda.matmul.allow_tf32 = state["matmul_allow_tf32"]


def component_types(model, optimizer, scheduler, scaler):
    return {name: f"{type(value).__module__}.{type(value).__qualname__}" if value is not None else None
            for name, value in (("model", model), ("optimizer", optimizer),
                                ("scheduler", scheduler), ("scaler", scaler))}


def save_checkpoint(directory, model, optimizer, *, step, progress, config_sha256,
                    provenance_sha256, scheduler=None, scaler=None):
    if type(step) is not int or step < 0 or not isinstance(progress, dict):
        raise ValueError("Checkpoint needs a nonnegative step and progress object")
    for pin in (config_sha256, provenance_sha256):
        if not isinstance(pin, str) or re.fullmatch(r"[a-f0-9]{64}", pin) is None:
            raise ValueError("Checkpoint requires config and provenance hashes")
    progress = json.loads(json_bytes(progress))
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"step-{step:010d}-{uuid.uuid4().hex}"
    target.mkdir()
    payload = {"version": VERSION, "step": step, "progress": progress,
               "config_sha256": config_sha256, "provenance_sha256": provenance_sha256,
               "model": model.state_dict(),
               "module_training": {name: module.training for name, module in model.named_modules()},
               "component_types": component_types(model, optimizer, scheduler, scaler),
               "optimizer": optimizer.state_dict(),
               "scheduler": scheduler.state_dict() if scheduler is not None else None,
               "scaler": scaler.state_dict() if scaler is not None else None,
               "rng": rng_state()}
    with (target / "state.pt").open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    manifest = {"version": VERSION, "file": "state.pt", "sha256": file_hash(target / "state.pt"),
                "step": step, "config_sha256": config_sha256, "provenance_sha256": provenance_sha256}
    pin = save_json(target / "manifest.json", manifest)
    return {"path": str(target / "manifest.json"), "sha256": pin}


def load_checkpoint(manifest_path, expected_sha256, model, optimizer, *, config_sha256,
                    provenance_sha256, scheduler=None, scaler=None):
    path = Path(manifest_path)
    manifest = read_json(path, expected_sha256)
    if (set(manifest) != {"version", "file", "sha256", "step", "config_sha256", "provenance_sha256"}
            or manifest["version"] != VERSION or manifest["file"] != "state.pt"
            or manifest["config_sha256"] != config_sha256
            or manifest["provenance_sha256"] != provenance_sha256):
        raise ValueError("Checkpoint does not belong to this configuration and provenance")
    blob = path.parent / "state.pt"
    if file_hash(blob) != manifest["sha256"]:
        raise ValueError("Checkpoint state differs from its manifest")
    payload = torch.load(blob, map_location="cpu", weights_only=True)
    if any(payload[key] != manifest[key] for key in ("version", "step", "config_sha256", "provenance_sha256")):
        raise ValueError("Checkpoint payload and manifest disagree")
    if payload["component_types"] != component_types(model, optimizer, scheduler, scaler):
        raise ValueError("Resume component types differ from the checkpoint")
    for name, component in (("scheduler", scheduler), ("scaler", scaler)):
        if (payload[name] is None) != (component is None):
            raise ValueError(f"Resume must supply the saved {name} configuration")
    if payload["rng"]["cuda"] and (not torch.cuda.is_available()
            or len(payload["rng"]["cuda"]) != torch.cuda.device_count()):
        raise ValueError("CUDA RNG state requires the same available device count")
    current = model.state_dict()
    if current.keys() != payload["model"].keys() or any(
            current[key].shape != value.shape or current[key].dtype != value.dtype
            for key, value in payload["model"].items()):
        raise ValueError("Model structure differs from the checkpoint")
    modules = dict(model.named_modules())
    if (modules.keys() != payload["module_training"].keys()
            or any(type(mode) is not bool for mode in payload["module_training"].values())):
        raise ValueError("Model module modes differ from the checkpoint structure")
    model.load_state_dict(payload["model"], strict=True)
    for name, module in modules.items():
        module.training = payload["module_training"][name]
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None:
        scaler.load_state_dict(payload["scaler"])
    restore_rng(payload["rng"])
    return {"step": payload["step"], "progress": payload["progress"]}
