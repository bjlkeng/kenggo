"""Run the frozen single-seed BASE recipe through separately accounted jobs."""

import argparse
import json
import math
import os
from pathlib import Path
import time

from kenggo.checkpoints import file_hash, json_hash, read_json, save_json
from kenggo.pilot_jobs import JobLimits, run_job
from kenggo.resource_ledger import BudgetLedger
from kenggo.tracking import Experiment, capture_provenance


ROOT = Path(__file__).resolve().parents[2]
ARMS = ("conditioned", "unconditioned")
DATA_INPUTS = ("data", "canonical", "split", "examples", "normalizer", "profile", "rules",
               "cpu_environment", "cuda_environment")


def _ref(path):
    path = Path(path).resolve()
    return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "sha256": file_hash(path)}


def _data_inputs(plan, plan_ref):
    return {name: plan["inputs"][name] for name in DATA_INPUTS} | {"pilot_plan": plan_ref}


def _config(plan, *, purpose, arm, parameters, inputs):
    return {key: plan[key] for key in ("version", "entity", "project", "base_url", "comparison", "seed", "budget_id")} | {
        "purpose": purpose, "arm": arm, "parameters": parameters, "inputs": inputs}


def _common_inputs(refs):
    from kenggo.examples import load_dataset_profile
    from kenggo.recorded_data import CommonInputs

    return CommonInputs(ROOT / refs["canonical"]["path"], refs["canonical"]["sha256"],
        (ROOT / refs["split"]["path"]).parent, refs["split"]["sha256"],
        (ROOT / refs["examples"]["path"]).parent, refs["examples"]["sha256"],
        ROOT / refs["normalizer"]["path"], refs["normalizer"]["sha256"],
        load_dataset_profile(ROOT / refs["profile"]["path"], refs["profile"]["sha256"]))


def prepare(config, output):
    from kenggo.canonical_data import encoded
    from kenggo.recorded_data import prepare_common_set

    inputs, result = _common_inputs(config["inputs"]), []
    for spec in config["parameters"]["common_sets"]:
        common = prepare_common_set(inputs, role=spec["role"], source_ids=spec["source_ids"],
                                    aggregate_ranks=spec["aggregate_ranks"])
        if len(common) != spec["expected_rows"]:
            raise ValueError("Common-set row count differs from the prespecified scope")
        directory = output / spec["id"]
        directory.mkdir()
        path = directory / "manifest.json"
        with path.open("x") as stream:
            stream.write(encoded(common.manifest))
            stream.flush()
            os.fsync(stream.fileno())
        result.append({"spec": spec, "manifest": _ref(path), "rows": len(common),
                       "coverage": common.manifest["coverage"]})
    summary = {"schema": "base_preparation_v1", "status": "completed", "common_sets": result,
               "predictions": 0, "scope": "Frozen development inputs before any model output"}
    save_json(output / "summary.json", summary)
    return summary


def _predictions(path, pin, row_keys):
    from kenggo.baseline_evaluation import load_predictions

    if file_hash(path) != pin:
        raise ValueError("Saved predictions differ from their evaluation manifest")
    return load_predictions(path, expected_row_keys=row_keys)


def analyze(config, output):
    from kenggo.baseline_evaluation import rows_for_common
    from kenggo.recorded_data import prepare_common_set
    from kenggo.recorded_statistics import paired_report

    comparisons = (("conditioned", "unconditioned"), ("conditioned", "human-reference"),
                   ("unconditioned", "human-reference"), ("conditioned_unknown_ranks", "conditioned"))
    result = {}
    for spec in config["parameters"]["common_sets"]:
        common = prepare_common_set(_common_inputs(config["inputs"]), role=spec["role"],
                                    source_ids=spec["source_ids"], aggregate_ranks=spec["aggregate_ranks"])
        if common.sha256 != config["inputs"][spec["manifest_input"]]["sha256"]:
            raise ValueError("Paired analysis common set changed")
        rows = rows_for_common(common)
        predictions = {}
        for arm, input_name in spec["prediction_inputs"].items():
            ref = config["inputs"][input_name]
            predictions[arm] = _predictions(ROOT / ref["path"], ref["sha256"], [row["row_key"] for row in rows])
        result[spec["id"]] = {}
        for candidate, control in comparisons:
            name = candidate + "_minus_" + control
            report = paired_report(rows, predictions[candidate], predictions[control],
                                   aggregate_ranks=spec["aggregate_ranks"])
            path = output / f"{spec['id']}-{name}.json"
            save_json(path, report)
            result[spec["id"]][name] = {"report": _ref(path), "overall": report["overall"]}
    summary = {"schema": "base_paired_analysis_v1", "status": "completed", "comparisons": result,
               "promotion": "not_promoted", "scope": "Single seed and insufficient account coverage; final data sealed"}
    save_json(output / "summary.json", summary)
    return summary


def local_worker(config_path, config_pin, provenance_path, provenance_pin, output):
    config = read_json(config_path, config_pin)
    provenance = read_json(provenance_path, provenance_pin)
    current, patch = capture_provenance(ROOT, config["inputs"])
    if (provenance["inputs"] != config["inputs"] or json_hash(provenance) != provenance_pin
            or provenance["source"] != current["source"]):
        raise ValueError("Worker source or inputs differ from the pinned run")
    output = Path(output)
    try:
        if config["purpose"] == "baseline-preparation":
            result = prepare(config, output)
        elif config["purpose"] == "baseline-analysis":
            result = analyze(config, output)
        else:
            raise ValueError("Unexpected local BASE worker purpose")
        after, _ = capture_provenance(ROOT, config["inputs"])
        if after["source"] != current["source"]:
            raise ValueError("Worker source changed")
        return result
    except BaseException as error:
        save_json(output / "failure.json", {"status": "failed", "type": type(error).__name__, "reason": str(error)})
        raise


def _worker_reports(config, directory, summary):
    def read(ref, *, repository_relative=False):
        path = ((ROOT if repository_relative else directory) / ref["path"]).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("A worker report must stay inside its owned directory")
        return read_json(path, ref["sha256"])

    result = {}
    if config["purpose"] == "baseline-evaluation":
        for report in summary["reports"]:
            artifacts = report["artifacts"]
            timing = read(artifacts["timing.json"])
            seconds = sorted(item["seconds"] for item in timing["measurements"])
            result[report["set"] + "/" + report["variant"]] = {
                "metrics": read(artifacts["metrics.json"]), "sampling": read(artifacts["sampling.json"]),
                "timing": {"unit": timing["unit"], "measurements": len(seconds),
                    "median_seconds": seconds[math.ceil(len(seconds) * .5) - 1] if seconds else None,
                    "p95_seconds": seconds[math.ceil(len(seconds) * .95) - 1] if seconds else None,
                    "method": "nearest_rank", "scope": timing.get("scope")}}
    elif config["purpose"] == "baseline-analysis":
        for role, comparisons in summary["comparisons"].items():
            for name, comparison in comparisons.items():
                result[role + "/" + name] = read(comparison["report"], repository_relative=True)
    elif config["purpose"] == "baseline-matches":
        if "scores_sha256" in summary:
            result["scores"] = read({"path": "scores.json", "sha256": summary["scores_sha256"]})
        result["behavior"] = summary.get("behavior", {})
        result["counts"] = summary.get("counts", {})
    return result


def _report_metrics(reports):
    result = {}
    omitted = {"side_weights", "lengths", "sgfs", "disagreements", "coverage", "joint_cells", "diagnostics",
               "aggregate_ranks", "missing_ranks", "reasons", "scope", "method", "uncertainty_method",
               "weighting", "classifier", "categories", "model_distribution"}

    def visit(value, prefix):
        if isinstance(value, dict):
            for name, child in value.items():
                counts = name == "counts" and prefix != "reports" and not prefix.startswith(("reports/scores", "reports/behavior"))
                if name not in omitted and not counts:
                    visit(child, prefix + "/" + name)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, prefix + "/" + str(index))
        elif value is not None:
            result[prefix] = value

    visit(reports, "reports")
    return result


def _settled_job(directory):
    path = directory / "worker/job.json"
    if path.is_file():
        record = json.loads(path.read_text())
        if "costs" in record and "parent_cpu_seconds" in record:
            return record
    return None


def _log_job(experiment, job, ledger, run_id):
    snapshot = ledger.snapshot()
    entry = next(item for item in snapshot["state"]["entries"] if item["run_id"] == run_id)
    experiment.log(100, {"job/status": job["status"], "job/cpu_seconds": entry["cpu_seconds"],
        "job/wall_seconds": entry["wall_seconds"], "job/gpu_seconds": entry["gpu_seconds"],
        "budget/charged_gpu_seconds": snapshot["charged_gpu_seconds"],
        "budget/remaining_gpu_seconds": snapshot["remaining_gpu_seconds"]})


def _stage(output, name, config, limits, ledger, *, module, device=None, register=False):
    started, cpu_started, job = time.monotonic(), time.process_time(), None
    provenance, patch = capture_provenance(ROOT, config["inputs"])
    directory = output / name
    overhead_id = output.name + "-" + name + "-tracking"
    ledger.reserve(overhead_id, estimate_gpu_seconds=0, shutdown_gpu_seconds=0,
                   metadata={"purpose": "tracking_and_parent_setup", "config_sha256": json_hash(config)})
    status, reason = "failed", "stage_not_completed"
    try:
        with Experiment(directory, config, provenance, patch) as experiment:
            environment = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES=device or "")
            python = ROOT / (".cache/venvs/cuda/bin/python" if device else ".venv/bin/python")
            command = [str(python), "-m", module, str(directory / "config.json"),
                str(directory / "provenance.json"), str(directory / "worker"),
                "--config-sha256", experiment.config_sha256, "--provenance-sha256", experiment.provenance_sha256]
            if module == "kenggo.base_pilot":
                command.insert(3, "--worker")
            run_id = output.name + "-" + name
            try:
                job = run_job(command, directory / "worker", cwd=ROOT, limits=JobLimits(**limits),
                    ledger=ledger, run_id=run_id, arm_shares={config["arm"]: 1},
                    device=device, environment=environment,
                    metadata={"config_sha256": experiment.config_sha256, "provenance_sha256": experiment.provenance_sha256,
                              "purpose": config["purpose"]})
                if job["status"] != "completed":
                    raise RuntimeError(f"BASE job failed: {name}: {job['reason']}")
                path = directory / "worker" / ("training_summary.json" if register else "summary.json")
                summary = read_json(path, file_hash(path))
                incomplete_matches = module == "kenggo.baseline_matches" and summary["status"] == "incomplete"
                if summary["status"] != "completed" and not incomplete_matches:
                    raise ValueError("Worker did not complete its declared units")
                if register:
                    for ref in summary["checkpoints"]:
                        experiment.register_checkpoint({"path": str(directory / "worker" / ref["path"]), "sha256": ref["sha256"]})
                    for epoch in summary["epoch_metrics"]:
                        experiment.log(epoch["epoch"], {f"training/{key}": value for key, value in epoch.items()
                            if type(value) in (int, float) and key != "epoch"})
                reports = _worker_reports(config, directory / "worker", summary)
                if reports:
                    experiment.log(100, _report_metrics(reports))
            finally:
                if job is None:
                    job = _settled_job(directory)
                if job is not None:
                    _log_job(experiment, job, ledger, run_id)
            status = "partial" if incomplete_matches else "completed"
            reason = "Scheduled match slots remain unresolved under the fixed budget" if incomplete_matches else None
            experiment.finish(status, reason=reason,
                summary={"scope": config["parameters"].get("scope", config["purpose"]),
                         "worker_status": summary["status"], "worker_summary_sha256": file_hash(path),
                         "worker_reports": reports})
        result = {"directory": str(directory), "config": _ref(directory / "config.json"),
            "provenance": _ref(directory / "provenance.json"), "summary": _ref(path),
            "wandb_url": json.loads((directory / "run.json").read_text())["wandb"]["url"], "job": job}
        save_json(directory / "stage.json", result)
        return result, summary
    except BaseException as error:
        status, reason = "failed", type(error).__name__ + ": " + str(error)
        raise
    finally:
        ledger.record(overhead_id, allocations=[], outcome=status, arm_shares={config["arm"]: 1},
            cpu_seconds=max(0, time.process_time() - cpu_started - (job["parent_cpu_seconds"] if job else 0)),
            wall_seconds=max(0, time.monotonic() - started - (job["costs"]["wall_seconds"] if job else 0)), reason=reason)


def _learned_bundle(stage, summary, plan, conditioned):
    ref = summary["final_checkpoint"]
    checkpoint = Path(stage["directory"]) / "worker" / ref["path"]
    refs = plan["inputs"]
    return {"checkpoint_manifest": str(checkpoint), "checkpoint_sha256": ref["sha256"],
        "config_path": str(ROOT / stage["config"]["path"]), "config_sha256": stage["config"]["sha256"],
        "provenance_sha256": stage["provenance"]["sha256"],
        "normalizer_path": str(ROOT / refs["normalizer"]["path"]), "normalizer_sha256": refs["normalizer"]["sha256"],
        "examples_sha256": refs["examples"]["sha256"],
        "model_config": plan["parameters"]["architecture"] | {"rank_conditioned": conditioned}}


def run(plan_path, plan_pin, output):
    plan = read_json(plan_path, plan_pin)
    if (plan["purpose"] != "baseline-pilot" or plan["budget_id"] != "e0-sl-pilot-v1"
            or plan["seed"] != 20261114):
        raise ValueError("The run requires the frozen single-seed BASE pilot plan")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan_ref, params = _ref(plan_path), plan["parameters"]
    ledger_path = ROOT / params["ledger"]
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = (BudgetLedger.open if ledger_path.exists() else BudgetLedger.create)(
        ledger_path, budget_id=plan["budget_id"], gpu_seconds=86400)
    stages, bundles, trainings = {}, {}, {}
    base = _data_inputs(plan, plan_ref)
    try:
        config = _config(plan, purpose="baseline-preparation", arm="shared",
                         parameters={"common_sets": params["common_sets"]}, inputs=base)
        stages["preparation"], prepared = _stage(output, "preparation", config, params["job_limits"]["preparation"],
            ledger, module="kenggo.base_pilot")
        specs, common_refs = [], {}
        for common in prepared["common_sets"]:
            name = "common_" + common["spec"]["id"]
            specs.append({key: common["spec"][key] for key in ("id", "role", "source_ids", "aggregate_ranks")}
                         | {"manifest_input": name})
            common_refs[name] = common["manifest"]
        for arm in ARMS:
            config = _config(plan, purpose="baseline-training", arm=arm,
                parameters=params["training"] | {"model": params["architecture"] | {"rank_conditioned": arm == "conditioned"},
                                                   "max_artifact_bytes": params["max_artifact_bytes"]}, inputs=base)
            stages[arm], trainings[arm] = _stage(output, arm, config, params["job_limits"]["training"], ledger,
                module="kenggo.baseline_training", device=params["device"], register=True)
            bundles[arm] = _learned_bundle(stages[arm], trainings[arm], plan, arm == "conditioned")
        for key in ("initial_tensor_sha256", "row_keys_sha256", "completed_updates", "presented_examples"):
            if trainings["conditioned"][key] != trainings["unconditioned"][key]:
                raise ValueError("Learned arms did not complete the matched initialization and training units")
        evaluations = {}
        for arm in (*ARMS, "human-reference"):
            refs = base | common_refs
            parameters = {"common_sets": specs, "batch_size": params["evaluation_batch_size"],
                "samples_per_position": params["samples_per_position"],
                "unknown_rank_diagnostic": arm == "conditioned" and params["unknown_rank_diagnostic"]}
            if arm in ARMS:
                bundle = bundles[arm]
                parameters["artifacts"] = bundle
                refs |= {"checkpoint": _ref(bundle["checkpoint_manifest"]), "training_config": stages[arm]["config"],
                         "training_provenance": stages[arm]["provenance"]}
            else:
                refs |= {name: plan["inputs"][name] for name in
                         ("human_profile", "human_config", "human_engine", "human_main_model", "human_model")}
                parameters.update(reference_budget=params["reference_budget"], human_paths={
                    "engine": refs["human_engine"]["path"], "main_model": refs["human_main_model"]["path"],
                    "human_model": refs["human_model"]["path"], "runtime_root": ".cache/katago-referee/squashfs-root"})
            config = _config(plan, purpose="baseline-evaluation", arm=arm, parameters=parameters, inputs=refs)
            name = "evaluation-" + arm
            limit = "learned_evaluation" if arm in ARMS else "human_evaluation"
            stages[name], evaluations[arm] = _stage(output, name, config, params["job_limits"][limit], ledger,
                                                   module="kenggo.baseline_evaluation")
        refs = base | {name: plan["inputs"][name] for name in
            ("human_profile", "human_config", "human_engine", "human_main_model", "human_model",
             "referee_profile", "referee_config", "referee_engine", "referee_model", "referee_native")}
        for arm in ARMS:
            refs |= {arm + "_checkpoint": _ref(bundles[arm]["checkpoint_manifest"]),
                     arm + "_config": stages[arm]["config"], arm + "_provenance": stages[arm]["provenance"]}
        parameters = {"learned_artifacts": bundles, "human_paths": {
                "engine": refs["human_engine"]["path"], "main_model": refs["human_main_model"]["path"],
                "human_model": refs["human_model"]["path"], "config": refs["human_config"]["path"],
                "runtime_root": ".cache/katago-referee/squashfs-root"},
            "human_budget": params["human_match_budget"], "referee_paths": {
                "engine": refs["referee_engine"]["path"], "model": refs["referee_model"]["path"],
                "config": refs["referee_config"]["path"], "native": refs["referee_native"]["path"],
                "runtime_root": ".cache/katago-referee/squashfs-root"},
            "pairs_per_profile": params["pairs_per_profile"], "conditioning_rank": params["conditioning_rank"],
            "opponent_policies": params["opponent_policies"], "device": "cpu", "adjudication": "exact_first_empirical_v2",
            "match_limits": params["match_limits"], "runner_budget": params["runner_budget"],
            "bootstrap_seed": plan["seed"], "bootstrap_resamples": params["bootstrap_resamples"]}
        config = _config(plan, purpose="baseline-matches", arm="shared", parameters=parameters, inputs=refs)
        stages["matches"], matches = _stage(output, "matches", config, params["job_limits"]["matches"], ledger,
                                          module="kenggo.baseline_matches")
        refs, analysis_specs = base | common_refs, []
        for spec in specs:
            prediction_inputs = {}
            for arm in (*ARMS, "human-reference"):
                variants = ("ordinary", "unknown_ranks") if arm == "conditioned" else ("ordinary",)
                for variant in variants:
                    label = arm if variant == "ordinary" else "conditioned_unknown_ranks"
                    name = spec["id"] + "_" + label
                    report = next(row for row in evaluations[arm]["reports"]
                                  if row["set"] == spec["id"] and row["variant"] == variant)
                    ref = report["artifacts"]["predictions.npz"]
                    path = Path(stages["evaluation-" + arm]["directory"]) / "worker" / ref["path"]
                    if file_hash(path) != ref["sha256"]:
                        raise ValueError("Evaluation prediction evidence changed")
                    refs[name], prediction_inputs[label] = _ref(path), name
            analysis_specs.append(spec | {"prediction_inputs": prediction_inputs})
        config = _config(plan, purpose="baseline-analysis", arm="shared",
                         parameters={"common_sets": analysis_specs}, inputs=refs)
        stages["analysis"], analysis = _stage(output, "analysis", config, params["job_limits"]["analysis"], ledger,
                                            module="kenggo.base_pilot")
        result = {"schema": "base_pilot_v1", "status": "completed", "plan": plan_ref, "stages": stages,
                  "ledger": ledger.snapshot(), "matched_training": True, "promotion": "not_promoted",
                  "scope": params["scope"]}
        save_json(output / "summary.json", result)
        return result
    except BaseException as error:
        save_json(output / "failure.json", {"status": "failed", "error_type": type(error).__name__,
                  "reason": str(error), "plan": plan_ref, "stages": stages, "ledger": ledger.snapshot()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("config", type=Path)
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--provenance-sha256")
    args = parser.parse_args()
    if args.worker:
        if len(args.paths) != 2 or not args.provenance_sha256:
            parser.error("Worker requires provenance and output paths plus provenance hash")
        local_worker(args.config, args.config_sha256, args.paths[0], args.provenance_sha256, args.paths[1])
    elif len(args.paths) != 1:
        parser.error("Pilot requires one output path")
    else:
        run(args.config, args.config_sha256, args.paths[0])
