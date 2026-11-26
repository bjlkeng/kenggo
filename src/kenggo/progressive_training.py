"""One deadline for data growth, matched continuation, comparisons and publication."""

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import time

from kenggo.checkpoints import file_hash, json_hash, read_json, save_json
from kenggo.continuation_data import RECIPE, path, read, reference
from kenggo.pilot_jobs import JobLimits, _output_bytes, run_job
from kenggo.resource_ledger import BudgetLedger
from kenggo.tracking import Experiment, capture_provenance, validate_config


ROOT = Path(__file__).resolve().parents[2]
VERSION = "e2_progressive_training_v1"
FINAL_RESERVE = {"cpu_seconds": 18000, "gpu_seconds": 12000, "output_bytes": 8 * 1024**3}


class StopStudy(RuntimeError):
    pass


def pin(ref):
    return {name: ref[name] for name in ("path", "sha256")}


def require_completed(stage):
    if stage["status"] != "completed":
        raise StopStudy("A required job was partial; its saved evidence is retained")


def dataset_inputs(summary_ref, base):
    artifacts = read(summary_ref)["artifacts"]
    result = {name: artifacts[key] for name, key in (
        ("canonical", "canonical"), ("split", "split_manifest"), ("examples", "examples_manifest"),
        ("profile", "profile"), ("dataset_normalizer", "normalizer"))}
    return {**result, **{key: base[key] for key in ("cpu_environment", "cuda_environment") if key in base},
            "data": summary_ref, "rules": base["rules"], "normalizer": base["normalizer"],
            "normalization_examples": base["examples"], "normalization_split": base["split"]}


def epoch_limit(rows):
    # Twice the old pilot's measured update cost, plus loading/checkpoint time.
    return math.ceil(rows / 128) * 0.36 + 1200


def memory_fits(rows, available):
    # Raw shard parts, concatenation, keys and temporary tensor/metadata copies.
    return rows * 5000 + 4 * 1024**3 <= available * 0.8


def available_memory():
    values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return int(values["MemAvailable"].split()[0]) * 1024


class Study:
    version = VERSION
    limit_description = "seven-day"
    job_module = "kenggo.progressive_jobs"
    supervisor_id = "supervisor"
    run_prefix = ""
    final_reserve = FINAL_RESERVE
    frozen_limits = {"wall_seconds": 604800, "gpu_seconds": 604800,
                     "cpu_seconds": 604800, "output_bytes": 160 * 1024**3,
                     "publication_seconds": 21600}

    def __init__(self, plan, plan_ref, output):
        validate_config(plan)
        self.plan, self.output = plan, Path(output).resolve()
        self.limits = plan["parameters"]["limits"]
        if self.limits != self.frozen_limits:
            raise ValueError(f"The study must use its frozen {self.limit_description} limits")
        intended = plan["parameters"].get("output")
        if not isinstance(intended, str) or not intended or self.output != (ROOT / intended).resolve():
            raise ValueError("The study must use its pinned output; another directory cannot reset the allowance")
        if self.output.exists():
            raise FileExistsError("This study has already launched. Whole-study restart is unsupported; preserve its session and ledger")
        self.output.mkdir(parents=True, exist_ok=False)
        self.started, self.cpu_started = time.monotonic(), time.process_time()
        self.accounted_cpu = 0.0
        self.closed = False
        self.launch = datetime.now(timezone.utc)
        self.deadline = self.launch + timedelta(seconds=self.limits["wall_seconds"])
        self.production_end = self.deadline - timedelta(seconds=self.limits["publication_seconds"])
        self.ledger = BudgetLedger.create(self.output / "resources.json", budget_id=plan["budget_id"],
                                          gpu_seconds=self.limits["gpu_seconds"])
        self.state = {"schema": self.version, "status": "running", "plan": plan_ref,
            "launch_utc": self.launch.isoformat(), "deadline_utc": self.deadline.isoformat(),
            "limits": self.limits, "finalization_reserve": self.final_reserve, "stages": {}, "tiers": [], "reason": None,
            "prior_costs": plan["inputs"]["prior_resources"], "promotion": False}
        save_json(self.output / "session.json", {key: self.state[key] for key in
                  ("schema", "plan", "launch_utc", "deadline_utc", "limits", "finalization_reserve", "prior_costs")})
        self.ledger.reserve(self.supervisor_id, estimate_gpu_seconds=0, shutdown_gpu_seconds=0,
                            metadata={"purpose": "progressive_supervisor", "cpu_reserve": 60})
        self.persist()

    def costs(self):
        ledger = self.ledger.snapshot()
        return {"elapsed_seconds": time.monotonic() - self.started,
                "cpu_seconds": sum(entry["cpu_seconds"] for entry in ledger["state"]["entries"])
                    + (0 if self.closed else max(0, time.process_time() - self.cpu_started - self.accounted_cpu) + 60),
                "output_bytes": _output_bytes(self.output), "ledger": ledger}

    def persist(self):
        self.state["costs"] = self.costs()
        save_json(self.output / "progress.json", self.state, replace=True)

    def require(self, wall, cpu, output_bytes, *, final=False, gpu=0):
        costs = self.costs()
        end = self.deadline if final else self.production_end
        remaining = min((end - datetime.now(timezone.utc)).total_seconds(),
                        self.limits["wall_seconds"] - costs["elapsed_seconds"]
                        - (0 if final else self.limits["publication_seconds"]))
        reserve = {name: 0 if final else value for name, value in self.final_reserve.items()}
        if (remaining < wall + 120
                or self.limits["cpu_seconds"] - costs["cpu_seconds"] - reserve["cpu_seconds"] < cpu + 600
                or self.limits["output_bytes"] - costs["output_bytes"] - reserve["output_bytes"] < output_bytes + 256 * 1024**2
                or costs["ledger"]["remaining_gpu_seconds"] - reserve["gpu_seconds"] < gpu + 30
                or costs["ledger"]["budget_breached"]):
            raise StopStudy("Remaining shared budget cannot fund the next complete unit")

    def config(self, *, purpose, arm, inputs, parameters):
        common = {key: self.plan[key] for key in
                  ("version", "entity", "project", "base_url", "comparison", "seed", "budget_id")}
        return {**common, "purpose": purpose, "arm": arm, "inputs": inputs, "parameters": parameters}

    def stage(self, name, config, *, wall, cpu=None, output_bytes=2 * 1024**3, gpu=False, final=False):
        cpu = wall * 1.2 if cpu is None else cpu
        self.require(wall + 180, cpu + 180, output_bytes, final=final, gpu=wall + 10 if gpu else 0)
        config = json.loads(json.dumps(config))
        end = min(datetime.now(timezone.utc) + timedelta(seconds=wall - 60),
                  self.deadline if final else self.production_end)
        config["parameters"]["deadline_utc"] = end.isoformat()
        config["parameters"]["max_artifact_bytes"] = min(output_bytes, 1024**3)
        started, cpu_started, job = time.monotonic(), time.process_time(), None
        directory = self.output / name
        self.state["active_stage"] = name
        self.persist()
        overhead_id = self.run_prefix + name + "-tracking"
        self.ledger.reserve(overhead_id, estimate_gpu_seconds=0, shutdown_gpu_seconds=0,
                            metadata={"purpose": "configuration_and_tracking"})
        outcome, reason = "failed", "Stage did not finish"
        try:
            provenance, patch = capture_provenance(ROOT, config["inputs"])
            with Experiment(directory, config, provenance, patch) as experiment:
                training = config["purpose"] == "continuation-training"
                module = "kenggo.continuation_training" if training else self.job_module
                python = ROOT / (".cache/venvs/cuda/bin/python" if gpu else ".venv/bin/python")
                command = [str(python), "-m", module, str(directory / "config.json"),
                    str(directory / "provenance.json"), str(directory / "worker"),
                    "--config-sha256", experiment.config_sha256,
                    "--provenance-sha256", experiment.provenance_sha256]
                env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="1",
                           MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", UV_OFFLINE="1",
                           CUDA_VISIBLE_DEVICES="0" if gpu else "")
                job = run_job(command, directory / "worker", cwd=ROOT,
                    limits=JobLimits(wall, cpu, output_bytes), ledger=self.ledger, run_id=self.run_prefix + name,
                    arm_shares={config["arm"]: 1}, device="0" if gpu else None, environment=env,
                    metadata={"config_sha256": experiment.config_sha256})
                summary_path = directory / "worker" / ("training_summary.json" if training else "summary.json")
                if not summary_path.is_file():
                    raise RuntimeError(f"{name}: {job['status']} / {job['reason']}; no completed summary")
                summary = read(reference(summary_path))
                if training:
                    owned = {ref["sha256"]: ref for ref in summary.get("checkpoints", [])}
                    wanted = {ref["sha256"] for ref in (summary.get("best_checkpoint"),
                              summary.get("latest_checkpoint")) if ref and ref["sha256"] in owned}
                    for checkpoint_hash in sorted(wanted):
                        experiment.register_checkpoint(pin(owned[checkpoint_hash]))
                    for ref in summary.get("commits", []):
                        metrics = read(read(ref)["metrics"])
                        flat = {scope + "/" + key: value for scope in ("training_monitor", "validation")
                                for key, value in metrics[scope].items() if type(value) in (int, float)}
                        experiment.log(metrics["lifetime_updates"], flat)
                experiment.log(max(experiment.attempt["last_step"] or 0, 1),
                    {"cost/cpu_seconds": job["costs"]["cpu_seconds"],
                     "cost/wall_seconds": job["costs"]["wall_seconds"], "job/status": job["status"]})
                outcome = "completed" if job["status"] == summary["status"] == "completed" else "partial"
                reason = None if outcome == "completed" else summary.get("reason") or job["reason"] or "Partial worker"
                experiment.finish(outcome, reason=reason,
                    summary={"worker_summary_sha256": file_hash(summary_path), "status": summary["status"]})
            stage = {"directory": str(directory), "config": reference(directory / "config.json"),
                "provenance": reference(directory / "provenance.json"), "summary": reference(summary_path),
                "job": job, "status": outcome}
            save_json(directory / "stage.json", stage)
            self.state["stages"][name] = stage
            return stage, summary
        except BaseException as error:
            outcome, reason = "failed", type(error).__name__ + ": " + str(error)
            evidence = {filename: reference(directory / filename) for filename in (
                "config.json", "provenance.json", "worker/job.json", "worker/summary.json",
                "worker/training_summary.json", "worker/training_progress.json", "worker/failure.json")
                if (directory / filename).is_file()}
            self.state.setdefault("failed_attempts", []).append({"name": name, "reason": reason,
                "directory": str(directory), "artifacts": evidence})
            raise
        finally:
            own_cpu = time.process_time() - cpu_started
            self.accounted_cpu += own_cpu
            self.ledger.record(overhead_id, allocations=[], outcome=outcome, arm_shares={config["arm"]: 1},
                cpu_seconds=max(0, own_cpu - (job["parent_cpu_seconds"] if job else 0)),
                wall_seconds=max(0, time.monotonic() - started - (job["costs"]["wall_seconds"] if job else 0)),
                reason=reason)
            self.persist()

    def finish(self, status, reason):
        self.state.update(status=status, reason=reason, active_stage=None)
        self.ledger.record(self.supervisor_id, allocations=[], outcome="completed" if status == "completed" else "partial",
            arm_shares={"shared": 1}, cpu_seconds=max(0, time.process_time() - self.cpu_started - self.accounted_cpu) + 60,
            wall_seconds=0, reason=None if status == "completed" else reason)
        self.accounted_cpu = time.process_time() - self.cpu_started + 60
        self.closed = True
        self.persist()
        save_json(self.output / "summary.json", self.state)
        return self.state


def initial_checkpoint(plan):
    old = read(plan["inputs"]["previous_training"])
    stage = old["stages"]["combined-conditioned"]
    summary = read(stage["summary"])
    ref = summary["final_checkpoint"]
    return {"path": str(Path(stage["directory"]) / "worker" / ref["path"]), "sha256": ref["sha256"],
            "config": stage["config"], "provenance": stage["provenance"], "step": ref["step"]}


def execute(study):
    plan, params = study.plan, study.plan["parameters"]
    base = params["base_inputs"]
    parent_data, parent_checkpoint = plan["inputs"]["parent_summary"], initial_checkpoint(plan)
    discovery = list(params["discovery_refs"])
    sources = list(params["source_refs"])
    preparation_refs = []
    tier_number = 0
    while True:
        tier_number += 1
        old_rows = read(read(parent_data)["artifacts"]["examples_manifest"])["counts"]["train"]["examples"]
        target = 2 * old_rows
        if not memory_fits(target, available_memory()):
            raise StopStudy("The next doubling exceeds measured host-memory headroom")
        study.require(36 * 3600 + 2 * epoch_limit(target) + 7200, 40 * 3600, 12 * 1024**3)
        if tier_number == 1:
            config = study.config(purpose="progressive-readiness", arm="shared", inputs=base, parameters={})
            stage, _ = study.stage("tier0-readiness", config, wall=600, cpu=300,
                                   output_bytes=256 * 1024**2, gpu=True)
            require_completed(stage)
        tier = {"tier": tier_number, "target_rows": target, "previous_rows": old_rows,
                "parent_data": parent_data, "parent_checkpoint": parent_checkpoint,
                "training": [], "assessments": {}, "status": "preparing"}
        study.state["tiers"].append(tier)
        source_inputs = {**base, "parent_summary": parent_data,
            **{f"discovery_{i}": ref for i, ref in enumerate(discovery)},
            **{f"source_{i}": ref for i, ref in enumerate(sources)}}
        source_config = study.config(purpose="progressive-sources", arm="shared", inputs=source_inputs,
                                     parameters={"tier": tier_number})
        stage, frozen = study.stage(f"tier{tier_number}-sources", source_config, wall=1800, cpu=1800)
        require_completed(stage)
        frozen_ref = frozen["sources"]
        sources.append(frozen_ref)
        request = {"version": "e2_data_growth_request_v1", "parent_summary": parent_data,
            "sources": frozen_ref, "target_train_rows": target, "seed": str(plan["seed"]),
            "fixture_only": False, "limits": {"cpu_seconds": 36 * 3600 - 600,
                "wall_seconds": 36 * 3600 - 600, "output_bytes": 24 * 1024**3,
                "max_records": 150000, "canonical_bytes": 1024**3}}
        request_path = study.output / f"tier{tier_number}-growth-request.json"
        save_json(request_path, request)
        growth_config = study.config(purpose="progressive-growth", arm="shared",
            inputs={**base, "request": reference(request_path)}, parameters={"tier": tier_number})
        stage, growth = study.stage(f"tier{tier_number}-growth", growth_config,
                               wall=36 * 3600, cpu=36 * 3600, output_bytes=24 * 1024**3)
        require_completed(stage)
        candidate_data = growth["dataset_summary"]
        candidate_inputs = dataset_inputs(candidate_data, base)
        candidate_rows = read(candidate_inputs["examples"])["counts"]["train"]["examples"]
        tier.update(dataset=candidate_data, training_rows=candidate_rows)
        cache_pass = 1
        while not growth["growth"]["target_within_one_game"] and (
                growth["growth"].get("source_unattempted", 0) > 0
                or growth["growth"].get("promotion_candidates", 0)
                    > len(growth["growth"].get("promoted_source_ids", []))):
            cache_pass += 1
            next_inputs = {**source_inputs, "parent_summary": candidate_data,
                           **{f"source_{i}": ref for i, ref in enumerate(sources)}}
            config = study.config(purpose="progressive-sources", arm="shared", inputs=next_inputs,
                                  parameters={"tier": tier_number})
            stage, frozen = study.stage(f"tier{tier_number}-sources-pass{cache_pass}", config, wall=1800, cpu=1800)
            require_completed(stage)
            sources.append(frozen["sources"])
            request.update(parent_summary=candidate_data, sources=frozen["sources"])
            request_path = study.output / f"tier{tier_number}-growth-request-pass{cache_pass}.json"
            save_json(request_path, request)
            growth_config["inputs"]["request"] = reference(request_path)
            stage, growth = study.stage(f"tier{tier_number}-growth-pass{cache_pass}", growth_config,
                                       wall=36 * 3600, cpu=36 * 3600, output_bytes=24 * 1024**3)
            require_completed(stage)
            candidate_data = growth["dataset_summary"]
            candidate_inputs = dataset_inputs(candidate_data, base)
            candidate_rows = read(candidate_inputs["examples"])["counts"]["train"]["examples"]
            tier.update(dataset=candidate_data, training_rows=candidate_rows)
        if not growth["growth"]["target_within_one_game"]:
            tier["cache_growth_shortfall"] = target - candidate_rows
            collection_config = study.config(purpose="progressive-collection", arm="shared",
                inputs={**source_inputs, "parent_summary": candidate_data,
                        **{f"source_{i}": ref for i, ref in enumerate(sources)}},
                parameters={"tier": tier_number, "credentials_file": params.get("credentials_file")})
            stage, collected = study.stage(f"tier{tier_number}-collection", collection_config,
                                       wall=22000, cpu=1200, output_bytes=2 * 1024**3)
            require_completed(stage)
            result = collected["result"]
            discovery.append(result["discovery"])
            sources.append(pin(result))
            request.update(parent_summary=candidate_data, sources=pin(result))
            request_path = study.output / f"tier{tier_number}-collection-growth-request.json"
            save_json(request_path, request)
            growth_config["inputs"]["request"] = reference(request_path)
            stage, growth = study.stage(f"tier{tier_number}-collection-growth", growth_config,
                wall=36 * 3600, cpu=36 * 3600, output_bytes=24 * 1024**3)
            require_completed(stage)
            candidate_data = growth["dataset_summary"]
            candidate_inputs = dataset_inputs(candidate_data, base)
            candidate_rows = read(candidate_inputs["examples"])["counts"]["train"]["examples"]
            tier.update(dataset=candidate_data, training_rows=candidate_rows)
        if not growth["growth"]["target_within_one_game"]:
            raise StopStudy("Usable whole games could not fill the next doubled dataset")
        prep_config = study.config(purpose="progressive-preparation", arm="shared",
            inputs={**base, "fixed_comparison": plan["inputs"]["fixed_comparison"],
                    **{f"previous_preparation_{i}": ref for i, ref in enumerate(preparation_refs)}},
            parameters={"tier": tier_number})
        prep_stage, preparation = study.stage(f"tier{tier_number}-preparation", prep_config, wall=3600, cpu=3600)
        require_completed(prep_stage)
        preparation_refs.append(prep_stage["summary"])
        tier["preparation"] = prep_stage["summary"]
        schedules = {"control": dataset_inputs(parent_data, base), "candidate": candidate_inputs}
        latest = {name: parent_checkpoint for name in schedules}
        pair = None
        for epoch in range(1001):
            wall = 1800 if epoch == 0 else epoch_limit(candidate_rows)
            try:
                study.require(2 * (wall + 240) + 3600, 2 * wall * 1.2 + 3600, 4 * 1024**3,
                              gpu=2 * (wall + 10))
            except StopStudy:
                if pair is None:
                    raise
                tier["training_stop"] = "shared_budget"
                break
            current_pair = {}
            for arm in (("control", "candidate") if epoch % 2 == 0 else ("candidate", "control")):
                model_inputs = {**schedules[arm], "parent_checkpoint": pin(latest[arm]),
                    "parent_config": latest[arm]["config"], "parent_provenance": latest[arm]["provenance"],
                    "train_monitor": preparation["train_monitor"], "epoch_validation": preparation["epoch_validation"]}
                training_config = study.config(purpose="continuation-training", arm=arm, inputs=model_inputs,
                    parameters={"recipe": RECIPE, "stage_id": f"tier{tier_number}",
                        "round_updates": math.ceil(candidate_rows / 128), "model": params["model"],
                        "training_rows": old_rows if arm == "control" else candidate_rows, "device": "cuda",
                        "checkpoint_reserve_seconds": 300, "evaluate_only": epoch == 0, "target_round": epoch,
                        "inference_contract": "komi_comparison_v1", "training_komi_halfpoints": [1, 13]})
                stage_name = f"tier{tier_number}-{arm}-epoch{epoch:02d}"
                try:
                    stage, summary = study.stage(stage_name, training_config,
                                                wall=wall, output_bytes=2 * 1024**3, gpu=True)
                except Exception as error:
                    tier.update(status="partial_interval", training_stop="interrupted_interval",
                        incomplete_interval={"epoch": epoch, "arm": arm, "status": "failed",
                            "reason": type(error).__name__ + ": " + str(error),
                            "attempt": next((item for item in reversed(study.state.get("failed_attempts", []))
                                             if item["name"] == stage_name), None)})
                    study.persist()
                    break
                tier["training"].append({"arm": arm, "epoch": epoch, "stage": stage})
                if stage["status"] != "completed":
                    tier.update(status="partial_interval", training_stop="interrupted_interval",
                        incomplete_interval={"epoch": epoch, "arm": arm, "stage": stage["summary"],
                            "status": stage["status"], "reason": summary.get("reason")
                                or stage["job"].get("reason") or "The interval did not finish"})
                    break
                latest[arm] = summary["latest_checkpoint"]
                current_pair[arm] = summary
            if len(current_pair) != 2:
                break
            pair = current_pair
            tier.update(status="training", completed_intervals=epoch,
                        selected={name: value["best_checkpoint"] for name, value in pair.items()})
            study.persist()
            if epoch and all(value["patience_exhausted"] for value in pair.values()):
                tier["training_stop"] = "validation_plateau"
                break
            if epoch == 1000:
                tier["training_stop"] = "maximum_intervals"
        if pair is None:
            raise StopStudy("No complete matched checkpoint pair was measured")
        for arm, checkpoint in {"parent": parent_checkpoint, **tier["selected"]}.items():
            inputs = {**base, "checkpoint": pin(checkpoint), "training_config": checkpoint["config"],
                      "training_provenance": checkpoint["provenance"],
                      **{"assessment_" + name: ref for name, ref in preparation["assessments"].items()}}
            config = study.config(purpose="progressive-assessment", arm=arm, inputs=inputs,
                                 parameters={"tier": tier_number, "device": "cuda"})
            stage, _ = study.stage(f"tier{tier_number}-assess-{arm}", config, wall=3600, cpu=3600,
                                   output_bytes=1024**3, gpu=True, final=True)
            require_completed(stage)
            tier["assessments"][arm] = stage["summary"]
        config = study.config(purpose="progressive-comparison", arm="shared",
            inputs={**base, **tier["assessments"]}, parameters={"tier": tier_number})
        stage, comparison = study.stage(f"tier{tier_number}-comparison", config, wall=600, cpu=600, final=True)
        require_completed(stage)
        interrupted = tier.get("training_stop") == "interrupted_interval"
        tier.update(status="partial_training" if interrupted else "completed", comparison_status="completed",
                    comparison=stage["summary"], decision=comparison["decision"])
        study.persist()
        if interrupted:
            raise StopStudy("An interval was interrupted. The last complete pair was compared; no further growth was started")
        if not comparison["decision"]["continue_growth"]:
            return "No useful supported data-growth gain; stop and report next-step hypotheses"
        parent_data, parent_checkpoint = candidate_data, tier["selected"]["candidate"]


def run(plan_path, plan_pin, output):
    plan = read_json(plan_path, plan_pin)
    study = Study(plan, reference(plan_path), output)
    status, reason = "failed", "Study did not finish"
    try:
        reason = execute(study)
        status = "completed"
    except StopStudy as error:
        status, reason = "budget_or_evidence_stop", str(error)
    except BaseException as error:
        reason = type(error).__name__ + ": " + str(error)
        save_json(study.output / "failure.json", {"type": type(error).__name__, "reason": str(error)})
        raise
    finally:
        result = study.finish(status, reason)
        from kenggo.progressive_review import export
        export(reference(study.output / "summary.json"), study.output / "review.html")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan")
    parser.add_argument("output")
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    run(args.plan, args.sha256, args.output)


if __name__ == "__main__":
    main()
