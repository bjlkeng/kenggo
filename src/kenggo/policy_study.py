"""Four policy-only trials sharing one three-day allowance and validation sample."""

import argparse
import math
from pathlib import Path

from kenggo.checkpoints import file_hash, read_json, save_json
from kenggo.continuation_data import path, read, reference
from kenggo.policy_recipe import DEFAULTS, RECIPE, TRIALS
from kenggo.progressive_training import (Study, StopStudy, dataset_inputs, pin,
                                        require_completed)


VERSION = "e2_policy_study_v1"
STAGE_ID = "policy-tuning-20261126"
ROUND_UPDATES = 7813
PREPARATION_KEYS = ("fitting_summary", "previous_preparation", "inventory",
                    "prior_selection", "fixed_assessment", "supplements")
LIMITS = {"wall_seconds": 259200, "gpu_seconds": 259200, "cpu_seconds": 388800,
          "output_bytes": 96 * 1024**3, "publication_seconds": 21600}


def validate_plan(plan):
    params, inputs = plan["parameters"], plan["inputs"]
    required = {*PREPARATION_KEYS, "parent_checkpoint", "parent_config", "parent_provenance",
                "protocol", "prior_resources", "review_history"}
    if (plan["purpose"] != "policy-study" or plan["arm"] != "shared"
            or plan["seed"] != 20261126 or params["recipe"] != RECIPE
            or params["stage_id"] != STAGE_ID or params["round_updates"] != ROUND_UPDATES
            or params["trials"] != list(TRIALS) or params["limits"] != LIMITS
            or params["training_rows"] != 3999997 or not required <= inputs.keys()
            or params["output"] != "runs/e2-policy-tuning-2026-11-26"):
        raise ValueError("The policy study requires its frozen inputs, trials, output and allowance")
    for ref in [*inputs.values(), *params["base_inputs"].values()]:
        read_ref = reference(path(ref))
        if read_ref["sha256"] != ref["sha256"]:
            raise ValueError("A policy-study input differs from its pin")
    parent = read(inputs["parent_config"])
    if parent["parameters"]["model"] != params["model"]:
        raise ValueError("Policy trials must keep the parent architecture")


class PolicyStudy(Study):
    version = VERSION
    limit_description = "three-day"
    job_module = "kenggo.policy_jobs"
    frozen_limits = LIMITS
    final_reserve = {"cpu_seconds": 18000, "gpu_seconds": 12000,
                     "output_bytes": 8 * 1024**3}

    def __init__(self, plan, plan_ref, output):
        validate_plan(plan)
        super().__init__(plan, plan_ref, output)
        refs = plan["inputs"]
        self.state.pop("tiers")
        self.state.update(parent_checkpoint={**refs["parent_checkpoint"],
            "config": refs["parent_config"], "provenance": refs["parent_provenance"]},
            trials=[{"id": item["id"], "parameters": {k: v for k, v in item.items() if k != "id"},
                     "training": [], "status": "pending"} for item in TRIALS],
            assessments={}, objective="masked policy cross-entropy only",
            value_metrics="diagnostic only; not optimized or used for selection")
        self.persist()


def rotation(trials, epoch):
    active = [trial for trial in trials if trial["status"] not in ("stopped", "failed")]
    if active:
        offset = epoch % len(active)
        return active[offset:] + active[:offset]
    return []


def select_trial(trials):
    eligible = [trial for trial in trials if trial.get("selected") is not None]
    if len(eligible) != len(TRIALS):
        raise StopStudy("Every declared trial needs a complete validation checkpoint")
    return min(eligible, key=lambda trial: (trial["best_validation_nll"],
        next(i for i, item in enumerate(TRIALS) if item["id"] == trial["id"])))


def select_committed(trial, diagnostic, parent_step):
    """Select saved measurements independently of a job's eventual exit status."""
    measured = {}
    for item in trial["training"]:
        stage = item["stage"]
        if not stage.get("summary"):
            continue
        for commit_ref in read(stage["summary"]).get("commits", []):
            commit = read(commit_ref)
            metrics, checkpoint = read(commit["metrics"]), commit["checkpoint"]
            config, provenance = read(checkpoint["config"]), read(checkpoint["provenance"])
            manifest = read(pin(checkpoint))
            epoch, score = metrics["round"], metrics["validation"]["nll"]
            if (commit["version"] != "continuation_epoch_commit_v1"
                    or type(epoch) is not int or epoch < 0
                    or epoch != item["epoch"] or epoch != commit["round"]
                    or type(score) not in (int, float) or not math.isfinite(score) or score < 0
                    or metrics["validation"]["diagnostic_sha256"] != diagnostic.get("sha256")
                    or metrics["sampler_step"] != epoch * ROUND_UPDATES
                    or metrics["lifetime_updates"] != parent_step + metrics["sampler_step"]
                    or checkpoint["step"] != metrics["lifetime_updates"]
                    or checkpoint["kind"] != "epoch" or checkpoint["config"] != stage.get("config")
                    or checkpoint["provenance"] != stage.get("provenance")
                    or config["arm"] != trial["id"] or config["parameters"]["recipe"] != RECIPE
                    or config["parameters"]["stage_id"] != STAGE_ID
                    or config["inputs"]["epoch_validation"] != diagnostic
                    or provenance["inputs"] != config["inputs"]
                    or manifest["version"] != "training_checkpoint_v2" or manifest["file"] != "state.pt"
                    or manifest["step"] != checkpoint["step"]
                    or manifest["config_sha256"] != checkpoint["config"]["sha256"]
                    or manifest["provenance_sha256"] != checkpoint["provenance"]["sha256"]):
                raise ValueError("Committed policy selection changed its checkpoint, sample or trial")
            value = (score, checkpoint)
            if epoch in measured and measured[epoch] != value:
                raise ValueError("Conflicting committed measurements for one trial interval")
            measured[epoch] = value
    if not measured:
        trial.pop("selected", None)
        return
    epoch = min(measured, key=lambda key: (measured[key][0], key))
    score, checkpoint = measured[epoch]
    manifest = read(pin(checkpoint))
    if file_hash(path(pin(checkpoint)).parent / manifest["file"]) != manifest["sha256"]:
        raise ValueError("Selected policy checkpoint payload changed")
    trial.update(selected=checkpoint, best_validation_nll=score, completed_rounds=max(measured))


def train(study, preparation):
    plan, params = study.plan, study.plan["parameters"]
    fitting = dataset_inputs(plan["inputs"]["fitting_summary"], params["base_inputs"])
    if read(fitting["examples"])["counts"]["train"]["examples"] != params["training_rows"]:
        raise ValueError("Fitting row count changed")
    trials = study.state["trials"]
    latest = {trial["id"]: study.state["parent_checkpoint"] for trial in trials}
    interrupted = False
    for epoch in range(DEFAULTS["max_rounds"] + 1):
        active = rotation(trials, epoch)
        if not active:
            break
        wall = 1800 if epoch == 0 else 3000
        try:
            study.require(len(active) * (wall + 240), len(active) * (wall * 1.2 + 240),
                          len(active) * 1024**3, gpu=len(active) * (wall + 10))
        except StopStudy:
            if epoch == 0:
                raise
            for trial in active:
                trial.update(status="stopped", stop_reason="shared_budget")
            break
        for trial in active:
            identity, parent = trial["id"], latest[trial["id"]]
            inputs = {**fitting, "parent_checkpoint": pin(parent), "parent_config": parent["config"],
                      "parent_provenance": parent["provenance"],
                      "train_monitor": preparation["train_monitor"],
                      "epoch_validation": preparation["epoch_validation"]}
            config = study.config(purpose="continuation-training", arm=identity, inputs=inputs,
                parameters={"recipe": RECIPE, "stage_id": STAGE_ID, "round_updates": ROUND_UPDATES,
                    "model": params["model"], "training_rows": params["training_rows"], "device": "cuda",
                    "checkpoint_reserve_seconds": 300, "evaluate_only": epoch == 0, "target_round": epoch,
                    "inference_contract": "komi_comparison_v1", "training_komi_halfpoints": [1, 13],
                    **DEFAULTS, **trial["parameters"]})
            name = f"{identity}-epoch{epoch:02d}"
            try:
                stage, summary = study.stage(name, config, wall=wall, output_bytes=1024**3, gpu=True)
                trial["training"].append({"epoch": epoch, "stage": stage})
                require_completed(stage)
            except Exception as error:
                if not trial["training"] or trial["training"][-1]["epoch"] != epoch:
                    failed = next((item for item in reversed(study.state.get("failed_attempts", []))
                                   if item["name"] == name), {})
                    artifacts = failed.get("artifacts", {})
                    summary_ref = artifacts.get("worker/training_summary.json")
                    progress_ref = artifacts.get("worker/training_progress.json")
                    trial["training"].append({"epoch": epoch, "stage": {
                        "status": "failed", "reason": str(error), "directory": failed.get("directory"),
                        "config": artifacts.get("config.json"), "summary": summary_ref or progress_ref,
                        "provenance": artifacts.get("provenance.json"),
                        "summary_kind": "worker_summary" if summary_ref else "retained_progress",
                        "artifacts": artifacts}})
                trial.update(status="failed", stop_reason=type(error).__name__ + ": " + str(error))
                interrupted = True
                break
            latest[identity] = summary["latest_checkpoint"]
            progress = summary["progress"]
            trial.update(selected=summary["best_checkpoint"],
                best_validation_nll=progress["stopping"]["best_score"],
                completed_rounds=progress["completed_rounds"], status="training")
            if summary["patience_exhausted"] or epoch == DEFAULTS["max_rounds"]:
                trial.update(status="stopped", stop_reason=("validation_plateau"
                    if summary["patience_exhausted"] else "maximum_intervals"))
            study.persist()
        if interrupted:
            for trial in trials:
                if trial["status"] in ("pending", "training"):
                    trial.update(status="stopped", stop_reason="another_trial_interrupted")
            break
    parent = study.state["parent_checkpoint"]
    parent_step = parent.get("step")
    if parent_step is None:
        parent_step = read(pin(parent))["step"]
    for trial in trials:
        select_committed(trial, preparation["epoch_validation"], parent_step)
    chosen = select_trial(trials)
    study.state.update(selected_trial_id=chosen["id"], selection="minimum fixed stopping-validation NLL",
                       training_interrupted=interrupted)
    study.persist()


def execute(study):
    plan, params = study.plan, study.plan["parameters"]
    base = params["base_inputs"]
    stage, _ = study.stage("readiness", study.config(purpose="policy-readiness", arm="shared",
        inputs=base, parameters={}), wall=600, cpu=300, output_bytes=256 * 1024**2, gpu=True)
    require_completed(stage)
    stage, preparation = study.stage("preparation", study.config(purpose="policy-preparation", arm="shared",
        inputs={key: plan["inputs"][key] for key in PREPARATION_KEYS}, parameters={"base_inputs": base}),
        wall=3600, cpu=3600, output_bytes=2 * 1024**3)
    require_completed(stage)
    study.state["preparation"] = stage["summary"]
    study.persist()
    train(study, preparation)
    selection_path = study.output / "selection.json"
    save_json(selection_path, {"selected_trial_id": study.state["selected_trial_id"],
        "method": study.state["selection"], "trials": [{key: trial[key] for key in
        ("id", "selected", "best_validation_nll", "completed_rounds", "status", "stop_reason")}
        for trial in study.state["trials"]], "assessment_seen": False})
    study.state["selection_record"] = reference(selection_path)
    checkpoints = {"parent": study.state["parent_checkpoint"],
                   **{trial["id"]: trial["selected"] for trial in study.state["trials"]}}
    for identity, checkpoint in checkpoints.items():
        inputs = {**base, "checkpoint": pin(checkpoint), "training_config": checkpoint["config"],
                  "training_provenance": checkpoint["provenance"],
                  **{"assessment_" + key: ref for key, ref in preparation["assessments"].items()}}
        stage, _ = study.stage(f"assessment-{identity}", study.config(purpose="policy-assessment",
            arm=identity, inputs=inputs, parameters={"device": "cuda"}),
            wall=1800, cpu=1800, output_bytes=1024**3, gpu=True, final=True)
        require_completed(stage)
        study.state["assessments"][identity] = stage["summary"]
        study.persist()
    stage, _ = study.stage("comparison", study.config(purpose="policy-comparison", arm="shared",
        inputs={**study.state["assessments"], "selection": study.state["selection_record"]},
        parameters={}), wall=1800, cpu=1800, output_bytes=256 * 1024**2, final=True)
    require_completed(stage)
    study.state["comparison"] = stage["summary"]
    return "Training and fixed development comparisons finished; no strength qualification"


def publish(study, status, reason):
    snapshot = {**study.state, "status": status, "reason": reason, "costs": study.costs(),
                "costs_scope": "through training and comparison; final publication costs are in summary.json"}
    snapshot_path = study.output / "review-input.json"
    save_json(snapshot_path, snapshot)
    stage, report = study.stage("publication", study.config(purpose="policy-publication", arm="shared",
        inputs={"study": reference(snapshot_path), "review_history": study.plan["inputs"]["review_history"]},
        parameters={}), wall=1800, cpu=1800, output_bytes=512 * 1024**2, final=True)
    require_completed(stage)
    study.state["review"] = report["review"]
    study.state["review_input"] = reference(snapshot_path)


def run(plan_path, plan_pin, output):
    study = PolicyStudy(read_json(plan_path, plan_pin), reference(plan_path), output)
    status, reason = "failed", "The study did not finish"
    try:
        reason = execute(study)
        status = "partial_training" if study.state.get("training_interrupted") else "completed"
    except StopStudy as error:
        status, reason = "budget_or_evidence_stop", str(error)
    except BaseException as error:
        reason = type(error).__name__ + ": " + str(error)
        save_json(study.output / "failure.json", {"status": "failed", "reason": reason})
    try:
        publish(study, status, reason)
    except BaseException as error:
        study.state["publication_failure"] = type(error).__name__ + ": " + str(error)
        status = "publication_failed"
    return study.finish(status, reason)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan")
    parser.add_argument("output")
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    result = run(args.plan, args.sha256, args.output)
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
