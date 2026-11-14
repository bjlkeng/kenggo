"""A training-only cost probe; no probe checkpoint may initialize the SL pilot."""

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
PHASES = ("cuda_training", "cpu_inference", "human_reference")


def common_inputs(config):
    from kenggo.examples import load_dataset_profile
    from kenggo.recorded_data import CommonInputs

    refs = config["inputs"]
    return CommonInputs(ROOT / refs["canonical"]["path"], refs["canonical"]["sha256"],
        (ROOT / refs["split"]["path"]).parent, refs["split"]["sha256"],
        (ROOT / refs["examples"]["path"]).parent, refs["examples"]["sha256"],
        ROOT / refs["normalizer"]["path"], refs["normalizer"]["sha256"],
        load_dataset_profile(ROOT / refs["profile"]["path"], refs["profile"]["sha256"]))


def _positions(config, output):
    from kenggo.checkpoints import save_json
    from kenggo.kgs_source import RANKS
    from kenggo.recorded_data import prepare_common_set

    common = prepare_common_set(common_inputs(config), role="train",
        source_ids=[config["parameters"]["probe_source_id"]], aggregate_ranks=RANKS)
    save_json(output / "common-set.json", common.manifest)
    positions = {row.metadata["ply"]: row for row in common.positions()}
    plies = config["parameters"]["inference_plies"]
    warmup = config["parameters"]["reference_warmup_ply"]
    if set(plies + [warmup]) - positions.keys():
        raise ValueError("The frozen training game does not supply every probe position")
    return [positions[ply] for ply in plies], positions[warmup]


def _model(config, conditioned, device):
    import torch
    from kenggo.checkpoints import seed_all
    from kenggo.learner import ModelConfig, PopulationModel

    seed_all(config["seed"])
    model = PopulationModel(ModelConfig(**config["parameters"]["architecture"],
                                       rank_conditioned=conditioned)).to(device)
    return model, torch.optim.Adam(model.parameters(), lr=config["parameters"]["learning_rate"])


def _resume_check(config, data, output, conditioned):
    import torch
    from kenggo.checkpoints import (json_hash, load_checkpoint, save_checkpoint, seed_all)
    from kenggo.supervised import train_step

    model, optimizer = _model(config, conditioned, "cuda")
    batch_size = min(config["parameters"]["batch_sizes"])
    inputs, labels = data.batch(0, batch_size, config["seed"], "cuda")
    train_step(model, optimizer, inputs, labels)
    ref = save_checkpoint(output / ("resume-conditioned" if conditioned else "resume-unconditioned"),
        model, optimizer, step=1, progress={"next_step": 1}, config_sha256=json_hash(config),
        provenance_sha256=json_hash({"purpose": "discarded-throughput-check"}))
    inputs, labels = data.batch(1, batch_size, config["seed"], "cuda")
    expected_metrics = train_step(model, optimizer, inputs, labels)
    expected = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    del model, optimizer
    seed_all(config["seed"] + 1)
    model, optimizer = _model(config, conditioned, "cuda")
    restored = load_checkpoint(ref["path"], ref["sha256"], model, optimizer,
        config_sha256=json_hash(config), provenance_sha256=json_hash({"purpose": "discarded-throughput-check"}))
    if restored != {"step": 1, "progress": {"next_step": 1}}:
        raise ValueError("CUDA checkpoint cursor did not restore")
    actual_metrics = train_step(model, optimizer, inputs, labels)
    if actual_metrics != expected_metrics:
        raise ValueError("CUDA checkpoint continuation metrics differ")
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value.detach().cpu(), expected[key], rtol=0, atol=0)
    torch.cuda.synchronize()
    return {"rank_conditioned": conditioned, "exact": True, "actual_updates": 3,
            "actual_examples": 3 * batch_size, "checkpoint": ref,
            "scope": "Resume validation only; weights discarded before the pilot"}


def _training(config, output):
    import torch
    from kenggo.checkpoints import json_hash, save_json
    from kenggo.supervised import TrainingData, train_step

    params, refs = config["parameters"], config["inputs"]
    host = subprocess.run(["nvidia-smi", "-i", params["device"],
        "--query-gpu=uuid,name,driver_version,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=True)
    if not torch.cuda.is_available():
        raise RuntimeError("The CUDA probe cannot fall back to CPU")
    torch.cuda.set_device(0)
    torch.cuda.synchronize()
    load_started = time.monotonic()
    data = TrainingData.load((ROOT / refs["examples"]["path"]).parent, refs["examples"]["sha256"],
        ROOT / refs["normalizer"]["path"], refs["normalizer"]["sha256"],
        split_sha256=refs["split"]["sha256"])
    load_seconds = time.monotonic() - load_started
    if len(data) != params["training_rows"]:
        raise ValueError("Training row count differs from the frozen probe")
    cases = [(conditioned, batch) for batch in params["batch_sizes"] for conditioned in (True, False)]
    rows = []
    for repetition in range(params["repetitions"]):
        for conditioned, batch_size in (cases if repetition % 2 == 0 else list(reversed(cases))):
            case_started = time.monotonic()
            model, optimizer = _model(config, conditioned, "cuda")
            torch.cuda.reset_peak_memory_stats()
            warmup_started = time.monotonic()
            for step in range(params["warmup_updates"]):
                inputs, labels = data.batch(step, batch_size, config["seed"], "cuda")
                train_step(model, optimizer, inputs, labels)
            torch.cuda.synchronize()
            warmup_seconds = time.monotonic() - warmup_started
            started, examples = time.monotonic(), 0
            for step in range(params["warmup_updates"], params["warmup_updates"] + params["timed_updates"]):
                inputs, labels = data.batch(step, batch_size, config["seed"], "cuda")
                metrics = train_step(model, optimizer, inputs, labels)
                examples += metrics["examples"]
            torch.cuda.synchronize()
            seconds = time.monotonic() - started
            row = {"rank_conditioned": conditioned, "batch_size": batch_size, "repetition": repetition,
                "timed_updates": params["timed_updates"], "timed_examples": examples,
                "warmup_updates": params["warmup_updates"], "warmup_seconds": warmup_seconds,
                "timed_seconds": seconds, "examples_per_second": examples / seconds,
                "end_to_end_seconds": time.monotonic() - case_started,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "last_training_metrics": metrics}
            rows.append(row)
            save_json(output / f"measurement-{len(rows):03d}.json", row)
            del model, optimizer, inputs, labels
            torch.cuda.empty_cache()
    resumes = [_resume_check(config, data, output, conditioned) for conditioned in (True, False)]
    summaries = []
    for conditioned, batch in cases:
        selected = [row for row in rows if row["rank_conditioned"] == conditioned and row["batch_size"] == batch]
        rates = [row["examples_per_second"] for row in selected]
        summaries.append({"rank_conditioned": conditioned, "batch_size": batch, "repetitions": len(rates),
                          "examples_per_second": {"minimum": min(rates), "median": statistics.median(rates),
                                                   "maximum": max(rates)}})
    return {"phase": "cuda_training", "training_rows": len(data), "data_load_seconds": load_seconds,
            "row_keys_sha256": json_hash(list(data.row_keys)), "measurements": rows,
            "summaries": summaries, "resume_checks": resumes,
            "host_before_cuda_initialization": host.stdout.strip(),
            "device": {"name": torch.cuda.get_device_name(),
                       "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory},
            "scope": "Training/debug cost probe, not held-out quality or complete-game throughput"}


def _cpu_inference(config, output):
    import numpy as np
    import torch
    from kenggo.checkpoints import save_json

    positions, warmup = _positions(config, output)
    models = {conditioned: _model(config, conditioned, "cpu")[0].eval() for conditioned in (True, False)}
    rows = []
    for repetition in range(config["parameters"]["repetitions"]):
        for conditioned in ((True, False) if repetition % 2 == 0 else (False, True)):
            model = models[conditioned]
            with torch.inference_mode():
                inputs = {key: torch.from_numpy(np.array(value)[None]) for key, value in warmup.observation.inputs.items()}
                model(inputs)
                for position in positions:
                    started = time.monotonic()
                    inputs = {key: torch.from_numpy(np.array(value)[None]) for key, value in position.observation.inputs.items()}
                    logits, value = model(inputs)
                    if (not torch.isfinite(logits[inputs["legal"]]).all()
                            or not torch.isfinite(value).all()):
                        raise ValueError("CPU inference produced invalid output")
                    rows.append({"rank_conditioned": conditioned, "repetition": repetition,
                        "row_key": position.observation.row_key, "ply": position.metadata["ply"],
                        "seconds": time.monotonic() - started, "network_calls": 1})
    save_json(output / "measurements.json", rows)
    return {"phase": "cpu_inference", "measurements": rows, "timed_network_calls": len(rows),
            "warmup_network_calls": 2 * config["parameters"]["repetitions"],
            "scope": "Random weights, prepared normalized observations; full-game and replay cost excluded"}


def _reference(config, output):
    from kenggo.checkpoints import save_json
    from kenggo.katago_human import HumanProfile, HumanSL, ReferenceBudget

    positions, warmup = _positions(config, output)
    ref = config["inputs"]["human_profile"]
    profile = HumanProfile.load(ROOT / ref["path"], ref["sha256"])
    repetitions, rows, costs = config["parameters"]["repetitions"], [], []
    for repetition in range(repetitions):
        directory = output / f"repeat-{repetition}"
        directory.mkdir()
        trace_number = 0

        def trace(record):
            nonlocal trace_number
            trace_number += 1
            save_json(directory / f"trace-{trace_number:03d}.json", record)

        with HumanSL(profile,
            engine=ROOT / ".cache/katago-referee/squashfs-root/usr/bin/katago",
            main_model=ROOT / ".cache/katago-referee/kata1-b10c128-s1141046784-d204142634.txt.gz",
            human_model=ROOT / ".cache/katago-human/b18c384nbt-humanv0.bin.gz",
            config=ROOT / config["inputs"]["human_config"]["path"],
            runtime_root=ROOT / ".cache/katago-referee/squashfs-root",
            budget=ReferenceBudget(queries=len(positions) + 1, cpu_seconds=100, wall_seconds=150),
            trace_sink=trace) as client:
            client.query_version()
            client.predict(warmup.observation.game, warmup.metadata["source_rank_mover"])
            for position in positions:
                started = time.monotonic()
                prediction = client.predict(position.observation.game, position.metadata["source_rank_mover"])
                rows.append({"repetition": repetition, "row_key": position.observation.row_key,
                    "ply": position.metadata["ply"], "seconds": time.monotonic() - started,
                    "pass_probability": prediction.policy[-1], "analysis_requests": 1})
        costs.append(client.cost())
        save_json(directory / "cost.json", costs[-1])
    return {"phase": "human_reference", "measurements": rows, "client_costs": costs,
            "timed_analysis_requests": len(rows), "warmup_analysis_requests": repetitions,
            "metadata_requests": repetitions, "network_calls": None,
            "scope": "Fresh native process per repetition; query timing includes history validation and trace persistence"}


def worker(config_path, config_pin, output, phase):
    import torch
    from kenggo.checkpoints import read_json, save_json

    config = read_json(config_path, config_pin)
    output = Path(output)
    if config["purpose"] != "throughput-probe" or phase not in PHASES:
        raise ValueError("The worker requires a named training-only throughput phase")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        result = {"cuda_training": _training, "cpu_inference": _cpu_inference,
                  "human_reference": _reference}[phase](config, output)
    except BaseException as error:
        save_json(output / "worker-failure.json", {"phase": phase, "status": "failed",
            "error_type": type(error).__name__, "reason": str(error),
            "completed_units": "Retained measurement and trace files only; work after the last saved unit is unconfirmed"})
        raise
    result["environment"] = {"python": platform.python_version(), "torch": str(torch.__version__),
                             "torch_cuda_build": torch.version.cuda, "threads": torch.get_num_threads(),
                             "deterministic": True, "tf32": False}
    save_json(output / "summary.json", result)


def run(config_path, config_pin, output):
    from kenggo.checkpoints import file_hash, json_hash, read_json, save_json
    from kenggo.pilot_jobs import JobLimits, run_job
    from kenggo.resource_ledger import BudgetLedger
    from kenggo.tracking import Experiment, capture_provenance

    started, cpu_started = time.monotonic(), time.process_time()
    config = read_json(config_path, config_pin)
    if config["purpose"] != "throughput-probe" or config["budget_id"] != "e0-throughput-v1":
        raise ValueError("A separately budgeted E0 throughput configuration is required")
    params = config["parameters"]
    provenance, patch = capture_provenance(ROOT, config["inputs"])
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    ledger_path = ROOT / params["ledger"]
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = (BudgetLedger.open if ledger_path.exists() else BudgetLedger.create)(
        ledger_path, budget_id=config["budget_id"], gpu_seconds=1800)
    jobs = []

    def costs():
        return {"cpu_seconds": time.process_time() - cpu_started + sum(job["child_cpu_seconds"] for job in jobs),
                "wall_seconds": time.monotonic() - started}

    def remaining():
        measured = costs()
        result = {key: params["overall_" + key] - value for key, value in measured.items()}
        if min(result.values()) <= 0:
            raise RuntimeError("The throughput probe exhausted its overall CPU/wall allowance")
        return result

    try:
        with Experiment(output / "run", config, provenance, patch) as experiment:
            for number, phase in enumerate(PHASES, 1):
                available = remaining()
                limits = dict(params["job_limits"][phase])
                for key in ("wall_seconds", "cpu_seconds"):
                    limits[key] = min(limits[key], available[key] - limits["shutdown_seconds"] - 60)
                if min(limits["wall_seconds"], limits["cpu_seconds"]) <= 0:
                    raise RuntimeError("Insufficient allowance remains for the next throughput phase")
                environment = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="1",
                    MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES=params["device"] if phase == "cuda_training" else "")
                python = ROOT / (".cache/venvs/cuda/bin/python" if phase == "cuda_training" else ".venv/bin/python")
                job = run_job([str(python), "-m", "kenggo.throughput", str(Path(config_path).resolve()),
                    str(output / phase), "--config-sha256", config_pin, "--worker", phase],
                    output / phase, cwd=ROOT, limits=JobLimits(**limits),
                    ledger=ledger, run_id=f"{output.name}-{phase}",
                    arm_shares={"human_reference": 1} if phase == "human_reference" else {"conditioned": .5, "unconditioned": .5},
                    device=params["device"] if phase == "cuda_training" else None, environment=environment,
                    metadata={"config_sha256": config_pin, "provenance_sha256": json_hash(provenance), "purpose": phase})
                jobs.append(job)
                if job["status"] != "completed":
                    raise RuntimeError(f"Throughput phase failed: {phase}: {job['reason']}")
                experiment.log(number, {f"{phase}/wall_seconds": job["costs"]["wall_seconds"],
                                        f"{phase}/cpu_seconds": job["costs"]["cpu_seconds"]})
            remaining()
            experiment.finish(summary={"scope": "Training-only cost probe; all weights excluded from pilot initialization",
                                       "phases_completed": len(jobs), "gpu_seconds": ledger.snapshot()["charged_gpu_seconds"]})
        remaining()
        result = {"schema": "e2_throughput_v1", "status": "completed", "config_sha256": config_pin,
            "provenance_sha256": json_hash(provenance), "jobs": jobs,
            "phases": {phase: read_json(output / phase / "summary.json", file_hash(output / phase / "summary.json"))
                       for phase in PHASES}, "ledger": ledger.snapshot(),
            "costs": costs(),
            "wandb_url": json.loads((output / "run/run.json").read_text())["wandb"]["url"],
            "cpu_scope": "Calling process plus charged job children; tracking service CPU not fully measured"}
        save_json(output / "summary.json", result)
        return result
    except BaseException as error:
        save_json(output / "failure.json", {"status": "failed", "error": type(error).__name__,
            "reason": str(error), "jobs": jobs, "ledger": ledger.snapshot(),
            "wall_seconds": time.monotonic() - started})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--worker", choices=PHASES)
    args = parser.parse_args()
    if args.worker:
        worker(args.config, args.config_sha256, args.output, args.worker)
    else:
        run(args.config, args.config_sha256, args.output)
