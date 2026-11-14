"""Bounded CPU integration check on a fixed prefix of the training data."""

import argparse
import copy
import json
from pathlib import Path
import time

import torch

from kenggo.checkpoints import file_hash, json_hash, load_checkpoint, read_json, save_json, seed_all
from kenggo.learner import ModelConfig, PopulationModel
from kenggo.resource_ledger import BudgetLedger
from kenggo.supervised import TrainingData, train_step
from kenggo.tracking import Experiment, capture_provenance


ROOT = Path(__file__).resolve().parents[2]
PARAMETERS = {
    "model": {"blocks": 1, "channels": 16, "rank_conditioned": True},
    "training_rows": 64, "updates": 16, "batch_size": 8, "clock_dropout": 0.1,
    "learning_rate": 0.003, "device": "cpu", "torch_threads": 1,
    "cpu_seconds": 180, "wall_seconds": 300, "max_artifact_bytes": 10 * 1024**2,
}


def run(config_path, config_pin, output):
    started, cpu = time.monotonic(), time.process_time()
    config = read_json(config_path, config_pin)
    if config["purpose"] != "learner-smoke" or config["parameters"] != PARAMETERS:
        raise ValueError("This check requires its frozen, bounded CPU debug configuration")
    provenance, patch = capture_provenance(ROOT, config["inputs"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    ledger = BudgetLedger.create(output / "resources.json", budget_id=config["budget_id"], gpu_seconds=0)
    ledger.reserve("learner-smoke", estimate_gpu_seconds=0, shutdown_gpu_seconds=0,
                   metadata={"config_sha256": config_pin, "purpose": config["purpose"], "arm": config["arm"]})
    settled, actual_updates = False, 0
    torch.set_num_threads(1)
    seed_all(config["seed"])

    def check_budget():
        if time.monotonic() - started > 300 or time.process_time() - cpu > 180:
            raise RuntimeError("Learner debug check exceeded its CPU/wall budget")

    def components():
        model = PopulationModel(ModelConfig(**PARAMETERS["model"]))
        return model, torch.optim.Adam(model.parameters(), lr=PARAMETERS["learning_rate"])

    def update(model, optimizer, step):
        nonlocal actual_updates
        check_budget()
        inputs, labels = data.batch(step, PARAMETERS["batch_size"], config["seed"])
        metrics = train_step(model, optimizer, inputs, labels, clock_dropout=PARAMETERS["clock_dropout"])
        actual_updates += 1
        return metrics

    try:
        refs = config["inputs"]
        data = TrainingData.load(ROOT / Path(refs["examples"]["path"]).parent, refs["examples"]["sha256"],
            ROOT / refs["normalizer"]["path"], refs["normalizer"]["sha256"],
            split_sha256=refs["split"]["sha256"], max_rows=PARAMETERS["training_rows"])
        if len(data) != PARAMETERS["training_rows"]:
            raise ValueError("The frozen training prefix is unavailable")
        model, optimizer = components()
        with Experiment(output / "run", config, provenance, patch) as experiment:
            for step in range(8):
                experiment.log(step + 1, {f"debug/{k}": v for k, v in update(model, optimizer, step).items()})
            checkpoint = experiment.checkpoint(model, optimizer, step=8, progress={"next_step": 8})
            expected_metrics = update(model, optimizer, 8)
            expected_state = copy.deepcopy(model.state_dict())
            experiment.finish("partial", reason="Checkpoint continuation control; one extra update retained in costs")
        seed_all(config["seed"] + 1)
        model, optimizer = components()
        with Experiment(output / "run", config, provenance, patch, resume=True) as resumed:
            restored = resumed.restore(checkpoint, model, optimizer)
            if restored != {"step": 8, "progress": {"next_step": 8}}:
                raise ValueError("The batch cursor did not restore")
            metrics = update(model, optimizer, 8)
            if metrics != expected_metrics:
                raise ValueError("The resumed training metrics differ from the control")
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, expected_state[name], rtol=0, atol=0)
            resumed.log(9, {f"debug/{k}": v for k, v in metrics.items()})
            for step in range(9, PARAMETERS["updates"]):
                resumed.log(step + 1, {f"debug/{k}": v for k, v in update(model, optimizer, step).items()})
            latest = resumed.checkpoint(model, optimizer, step=16, progress={"next_step": 16})
            resumed.finish(summary={"scope": "Training-prefix implementation check; no BASE or held-out result",
                                    "resume_exact": True, "distinct_updates": 16,
                                    "actual_updates_including_control": actual_updates, "gpu_seconds": 0})
        import wandb
        state = json.loads((output / "run/run.json").read_text())
        api = wandb.Api(overrides={"base_url": config["base_url"]}, timeout=15)
        artifact_name = state["checkpoints"][-1]["wandb"]["name"]
        downloaded = Path(api.artifact(artifact_name, type="checkpoint").download(
            root=str(output / "downloaded-checkpoint")))
        restored_model, restored_optimizer = components()
        restored = load_checkpoint(downloaded / "manifest.json", latest["sha256"],
            restored_model, restored_optimizer, config_sha256=state["config_sha256"],
            provenance_sha256=state["provenance_sha256"])
        if restored != {"step": 16, "progress": {"next_step": 16}}:
            raise ValueError("Downloaded artifact has the wrong training cursor")
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, restored_model.state_dict()[name], rtol=0, atol=0)
        check_budget()
        costs = {"cpu_seconds": time.process_time() - cpu, "wall_seconds": time.monotonic() - started}
        ledger.record("learner-smoke", allocations=[], outcome="completed", arm_shares={config["arm"]: 1}, **costs)
        settled = True
        result = {"version": "learner_smoke_v1", "status": "completed", "config_sha256": config_pin,
                  "wandb_url": state["wandb"]["url"], "artifact": artifact_name,
                  "provenance_sha256": state["provenance_sha256"], "training_rows": len(data),
                  "training_row_keys_sha256": json_hash(list(data.row_keys)),
                  "distinct_updates": 16, "actual_updates_including_control": actual_updates,
                  "actual_examples_processed": actual_updates * PARAMETERS["batch_size"],
                  "resume_exact": True, "downloaded_checkpoint_verified": True,
                  "checkpoint": {"path": str(Path(latest["path"]).relative_to(output)), "sha256": latest["sha256"]},
                  "run_manifest_sha256": file_hash(output / "run/run.json"),
                  "ledger_sha256": file_hash(output / "resources.json"), "costs": costs,
                  "artifact_bytes_attempted": state["artifact_bytes_attempted"], "gpu_seconds": 0,
                  "scope": "Fixed training-prefix CPU debug check; no held-out evaluation or playing-strength claim."}
        save_json(output / "summary.json", result)
        print(json.dumps(result, indent=2))
        return result
    except BaseException as error:
        costs = {"cpu_seconds": time.process_time() - cpu, "wall_seconds": time.monotonic() - started}
        if not settled:
            ledger.record("learner-smoke", allocations=[], outcome="failed", arm_shares={config["arm"]: 1},
                          reason=type(error).__name__, **costs)
        save_json(output / "failure.json", {"status": "failed", "reason": type(error).__name__,
                  "actual_updates_including_control": actual_updates, "costs": costs})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config-sha256", required=True)
    arguments = parser.parse_args()
    run(arguments.config, arguments.config_sha256, arguments.output)
