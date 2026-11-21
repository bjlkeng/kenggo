"""Owned CPU/GPU workers for the seven-day progressive E2 experiment."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import platform
import time

from kenggo.baseline_training import _source_identity
from kenggo.checkpoints import json_hash, read_json, save_json
from kenggo.continuation_data import path, read, reference
from kenggo.progressive_evaluation import assess, compare, prepare
from kenggo.tracking import validate_config


def readiness():
    import numpy as np
    import torch
    if (platform.python_version() != "3.12.13" or torch.__version__ != "2.15.0+cu132"
            or np.__version__ != "2.5.4" or not torch.cuda.is_available()):
        raise ValueError("The production CUDA environment differs from its pinned versions")
    value = torch.ones(8, device="cuda", requires_grad=True)
    value.square().sum().backward()
    torch.cuda.synchronize()
    if not torch.equal(value.grad, torch.full_like(value, 2)):
        raise ValueError("CUDA autograd readiness failed")
    return {"status": "completed", "python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "cuda_build": torch.version.cuda,
            "device": torch.cuda.get_device_name(0), "check": "cuda_tensor_autograd_synchronized"}


def run(config_path, config_pin, provenance_path, provenance_pin, output):
    config, provenance = read_json(config_path, config_pin), read_json(provenance_path, provenance_pin)
    validate_config(config)
    if (json_hash(provenance) != provenance_pin or provenance["inputs"] != config["inputs"]
            or provenance["source"] != _source_identity()):
        raise ValueError("Progressive worker configuration and frozen source must match")
    for ref in config["inputs"].values():
        if reference(path(ref)) != {"path": str(path(ref)), "sha256": ref["sha256"]}:
            raise ValueError("A progressive input differs from its pin")
    output = Path(output)
    deadline = datetime.fromisoformat(config["parameters"]["deadline_utc"])
    monotonic_end = time.monotonic() + (deadline - datetime.now(timezone.utc)).total_seconds()
    def check():
        if min((deadline - datetime.now(timezone.utc)).total_seconds(), monotonic_end - time.monotonic()) < 120:
            raise TimeoutError("Shared deadline reached; keep time for final evidence")
    check()
    purpose = config["purpose"]
    if purpose == "progressive-readiness":
        result = readiness()
    elif purpose == "progressive-preparation":
        result = prepare(config, output, check_budget=check)
    elif purpose == "progressive-assessment":
        result = assess(config, output, check_budget=check)
    elif purpose == "progressive-comparison":
        result = compare(config, output)
    elif purpose == "progressive-sources":
        from kenggo.data_growth_source import freeze_sources
        refs = config["inputs"]
        result = freeze_sources(refs["parent_summary"],
            [ref for key, ref in refs.items() if key.startswith("discovery_")],
            [ref for key, ref in refs.items() if key.startswith("source_")], output / "sources.json",
            seed=str(config["seed"]), check_budget=check)
        result = {"schema": "progressive_source_job_v1", "status": "completed", "result": result,
                  "sources": reference(output / "sources.json")}
    elif purpose == "progressive-growth":
        from kenggo.data_growth import build_growth_round
        request = read(config["inputs"]["request"])
        result = build_growth_round(request, output / "dataset", check_budget=check)
        result = {"schema": "progressive_growth_job_v1", "status": "completed", "growth": result,
                  "dataset_summary": reference(output / "dataset" / "summary.json")}
    elif purpose == "progressive-collection":
        from kenggo.data_growth_source import collect_more
        refs = config["inputs"]
        result = collect_more(refs["parent_summary"],
            [ref for key, ref in refs.items() if key.startswith("discovery_")],
            [ref for key, ref in refs.items() if key.startswith("source_")], output / "collection",
            seed=str(config["seed"]), deadline_utc=deadline.isoformat(),
            credentials_file=config["parameters"].get("credentials_file"), check_budget=check)
        result = {"schema": "progressive_collection_job_v1", "status": "completed", "result": result}
    else:
        raise ValueError("Unknown progressive job purpose")
    if provenance["source"] != _source_identity():
        raise ValueError("Worker source changed during production")
    save_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("provenance")
    parser.add_argument("output")
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--provenance-sha256", required=True)
    args = parser.parse_args()
    try:
        run(args.config, args.config_sha256, args.provenance, args.provenance_sha256, args.output)
    except BaseException as error:
        save_json(Path(args.output) / "failure.json", {"status": "failed", "type": type(error).__name__,
                                                      "reason": str(error)})
        raise


if __name__ == "__main__":
    main()
