"""Exercise online tracking and exact checkpoint continuation on synthetic CPU tensors."""

import argparse
import copy
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from kenggo.checkpoints import file_hash, load_checkpoint, read_json, save_json, seed_all
from kenggo.resource_ledger import BudgetLedger
from kenggo.tracking import Experiment, capture_provenance


ROOT = Path(__file__).resolve().parents[2]


def components():
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(0.25), torch.nn.Linear(4, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
    return model, optimizer, scheduler


def update(model, optimizer, scheduler):
    inputs = torch.randn(5, 3) + np.random.random() + random.random()
    optimizer.zero_grad()
    loss = (model(inputs) - inputs.sum(dim=1, keepdim=True)).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return float(loss.detach())


def run(config_path, config_pin, output):
    started, cpu = time.monotonic(), time.process_time()
    config = read_json(config_path, config_pin)
    if config["purpose"] != "tracking-smoke" or config["parameters"] != {
            "model": "synthetic-linear-dropout", "device": "cpu", "torch_threads": 1,
            "data_usage": "provenance-only", "cpu_seconds": 120, "wall_seconds": 180,
            "max_artifact_bytes": 10 * 1024**2}:
        raise ValueError("Smoke requires its bounded synthetic CPU configuration")
    provenance, patch = capture_provenance(ROOT, config["inputs"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    ledger = BudgetLedger.create(output / "resources.json", budget_id=config["budget_id"], gpu_seconds=0)
    ledger.reserve("tracking-smoke", estimate_gpu_seconds=0, shutdown_gpu_seconds=0,
                   metadata={"config_sha256": config_pin, "purpose": "tracking-smoke", "arm": config["arm"]})
    settled = False
    torch.set_num_threads(1)
    seed_all(config["seed"])
    run_path = output / "run"
    def check_budget():
        if time.monotonic() - started > 180 or time.process_time() - cpu > 120:
            raise RuntimeError("Tracking smoke exceeded its CPU/wall budget")
    try:
        model, optimizer, scheduler = components()
        with Experiment(run_path, config, provenance, patch) as experiment:
            first_loss = update(model, optimizer, scheduler)
            experiment.log(1, {"synthetic/loss": first_loss})
            ref = experiment.checkpoint(model, optimizer, step=1, progress={"next_batch": 1}, scheduler=scheduler)
            expected_loss = update(model, optimizer, scheduler)
            expected_state = copy.deepcopy(model.state_dict())
            experiment.finish("partial", reason="Deliberate checkpoint/resume verification")
        check_budget()
        seed_all(config["seed"] + 1)
        resumed_model, resumed_optimizer, resumed_scheduler = components()
        with Experiment(run_path, config, provenance, patch, resume=True) as resumed:
            restored = resumed.restore(ref, resumed_model, resumed_optimizer, scheduler=resumed_scheduler)
            if restored != {"step": 1, "progress": {"next_batch": 1}}:
                raise ValueError("Checkpoint progress did not restore")
            resumed_loss = update(resumed_model, resumed_optimizer, resumed_scheduler)
            if resumed_loss != expected_loss:
                raise ValueError("Resumed stochastic update differs from uninterrupted execution")
            for key, value in resumed_model.state_dict().items():
                torch.testing.assert_close(value, expected_state[key], rtol=0, atol=0)
            resumed.log(2, {"synthetic/loss": resumed_loss, "checks/resume_exact": True})
            latest = resumed.checkpoint(resumed_model, resumed_optimizer, step=2,
                                       progress={"next_batch": 2}, scheduler=resumed_scheduler)
            check_budget()
            resumed.finish(summary={"scope": "Synthetic tracking/resume check; no Go model or GPU work",
                                    "resume_exact": True, "checkpoint_sha256": latest["sha256"], "gpu_seconds": 0})
        import wandb
        api = wandb.Api(overrides={"base_url": config["base_url"]}, timeout=15)
        state = json.loads((run_path / "run.json").read_text())
        online = api.run(f"{config['entity']}/{config['project']}/{state['id']}")
        if (online.state != "finished" or online.summary.get("resume_exact") is not True
                or online.summary.get("attempts/count") != 2
                or online.summary.get("attempts/1/status") != "partial"
                or online.summary.get("attempts/2/status") != "completed"
                or online.summary.get("costs/incomplete_attempts") != 0
                or online.summary.get("gpu_seconds") != 0):
            raise ValueError("Online W&B run does not confirm successful resume")
        recorded_cpu = sum(a["cpu_seconds"] for a in state["attempts"])
        if online.summary.get("costs/process_cpu_seconds", -1) < recorded_cpu:
            raise ValueError("Online W&B run omits recorded attempt costs")
        artifact_ref = state["checkpoints"][-1]["wandb"]["name"]
        artifact = api.artifact(artifact_ref, type="checkpoint")
        downloaded = Path(artifact.download(root=str(output / "downloaded-checkpoint")))
        restored_model, restored_optimizer, restored_scheduler = components()
        loaded = load_checkpoint(downloaded / "manifest.json", latest["sha256"], restored_model,
            restored_optimizer, scheduler=restored_scheduler,
            config_sha256=state["config_sha256"], provenance_sha256=state["provenance_sha256"])
        for key, value in restored_model.state_dict().items():
            torch.testing.assert_close(value, expected_state[key], rtol=0, atol=0)
        if loaded["step"] != 2:
            raise ValueError("Downloaded checkpoint has the wrong step")
        check_budget()
        costs = {"cpu_seconds": time.process_time() - cpu, "wall_seconds": time.monotonic() - started}
        ledger.record("tracking-smoke", allocations=[], outcome="completed", arm_shares={config["arm"]: 1}, **costs)
        settled = True
        result = {"version": "tracking_smoke_v1", "status": "completed", "wandb_url": state["wandb"]["url"],
                  "wandb_run": f"{config['entity']}/{config['project']}/{state['id']}",
                  "artifact": artifact_ref, "config_sha256": config_pin,
                  "run_manifest_sha256": file_hash(run_path / "run.json"),
                  "checkpoint": {"path": str(Path(latest["path"]).relative_to(output)), "sha256": latest["sha256"]},
                  "provenance_sha256": state["provenance_sha256"], "ledger_sha256": file_hash(output / "resources.json"),
                  "resume_exact": True, "downloaded_checkpoint_verified": True, "attempts": 2,
                  "gpu_seconds": 0, "costs": costs, "artifact_bytes_attempted": state["artifact_bytes_attempted"],
                  "cost_scope": "Calling-process CPU and elapsed smoke body after imports; no total background-service CPU claim.",
                  "scope": "Synthetic CPU tracking/resume and artifact round trip; data pins are provenance only, no BASE result."}
        save_json(output / "summary.json", result)
        print(json.dumps(result, indent=2))
        return result
    except BaseException as error:
        costs = {"cpu_seconds": time.process_time() - cpu, "wall_seconds": time.monotonic() - started}
        if not settled:
            ledger.record("tracking-smoke", allocations=[], outcome="failed", arm_shares={config["arm"]: 1},
                          reason=type(error).__name__, **costs)
        save_json(output / "failure.json", {"status": "failed", "reason": type(error).__name__, "costs": costs})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config-sha256", required=True)
    arguments = parser.parse_args()
    run(arguments.config, arguments.config_sha256, arguments.output)
