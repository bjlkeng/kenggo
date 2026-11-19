"""Offline review checks using small saved-result fixtures, without model or dataset reads."""

import copy
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from kenggo.checkpoints import file_hash, save_json


SPEC = importlib.util.spec_from_file_location("expanded_review", Path(__file__).resolve().parents[1] / "tools/review_expanded.py")
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


def ref(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    return {"path": str(path), "sha256": save_json(path, value, replace=True)}


def metric(value):
    return {"status": "available", "estimate": {"status": "finite", "value": value},
        "counts": {"moves": 2, "games": 2, "accounts": 2}, "masked_moves": 0,
        "uncertainty": {"status": "inconclusive", "interval95": None,
                        "reasons": ["fewer_than_30_accounts"]}}


def metrics(delta=False):
    group = {"metrics": {"nll": metric(-.04 if delta else 2.1), "top1": metric(.2),
                         "value_mse": metric(.3)}, "value_calibration": {"model": {"bins": []}}}
    return {"kind": "candidate_minus_control" if delta else "single_model", "expected_outputs": 2,
            "overall": group, "by_rank": {"5k": group}, "by_phase": {"opening": group},
            "by_clock": {"10-20": group}}


def review_metrics(delta=False):
    def report(ranks, rows, value):
        result = copy.deepcopy(metrics(delta))
        result.update(aggregate_ranks=ranks, expected_outputs=rows)
        group = result["overall"]
        group["counts"] = {"moves": rows}
        group["metrics"]["nll"]["estimate"]["value"] = value
        for item in group["metrics"].values():
            item["aggregate_ranks"] = ranks
        group["value_calibration"]["model"] = {"status": "available", "bins": [
            {"lower": -1, "upper": 1, "raw_count": rows,
             "mean_prediction": {"value": .125 if rows == 1 else .875}, "mean_result": {"value": 1}}]}
        result["by_rank"] = {rank: dict(copy.deepcopy(group), counts={"moves": int(rank in {"14k", "5k"})})
                             for rank in ranks}
        return result
    result = report(review.ALL_RANKS, 2, -.04 if delta else 2.1)
    result["review_scope"] = report(review.REVIEW_RANKS, 1, -.08 if delta else 1.25) | {"excluded_rows": 1}
    return result


def fixture(root, *, name="expanded-synthetic"):
    run = root / name
    run.mkdir(parents=True)
    plan_ref = ref(root / "plan.json", {"purpose": "expanded-pilot", "seed": 20261118,
                                       "parameters": {"scope": "synthetic fixture"}})
    stages, common, predicted = {}, {}, {}
    entries = [{"run_id": "earlier-pilot", "gpu_seconds": 1246.177333, "cpu_seconds": 1., "wall_seconds": 1.}]
    for name in review.STAGES:
        directory, summary = run / name, {"status": "completed"}
        worker = directory / "worker"
        worker.mkdir(parents=True)
        config = {"inputs": {"raw_data_never_opened": {"path": str(root / "missing-dataset.parquet"),
                                                        "sha256": "a" * 64}}, "parameters": {}}
        if name == "preparation":
            selected = []
            for identity in review.SETS:
                role, komi = identity.rsplit("_komi", 1)
                spec = {"id": identity, "role": role, "komi_halfpoints": int(komi),
                        "source_ids": ["https://example.invalid/" + identity + suffix for suffix in ("-a", "-z")],
                        "aggregate_ranks": ["5k"]}
                manifest = ref(worker / identity / "manifest.json", {"role": role,
                    "source_ids": list(reversed(spec["source_ids"])),
                    "rows": [{"row_key": identity + "-0"}, {"row_key": identity + "-1"}]})
                item = {"spec": spec, "rows": 2, "manifest": manifest, "coverage": {"games": 2, "accounts": 4}}
                common[identity] = item
                selected.append(item)
            summary.update(common_sets=selected, selection=ref(worker / "selection.json",
                {"schema": "expanded_selection_v1", "target_or_prediction_selection": False, "sets": selected}))
        elif name in review.ARMS or name.startswith("probe-"):
            arm = name.removeprefix("probe-")
            probe = name.startswith("probe-")
            updates = 256 if probe else 8
            summary.update(training_rows=1000000 if arm.startswith("baseline") else 2000000,
                completed_updates=updates, presented_examples=updates * 128,
                initial_tensor_sha256="a" * 64, row_keys_sha256="b" * 64 if arm.startswith("baseline") else "c" * 64,
                checkpoint_selection="fixed_final_update", final_checkpoint={"path": "never-open-checkpoint/manifest.json"},
                update_metrics=[{"phase": "warmup", "updates": 6, "seconds_per_update": .02, "order_sha256": "d" * 64}]
                    + [{"phase": "measured", "updates": 50 if probe else 2, "seconds_per_update": .01,
                        "order_sha256": "e" * 64} for _ in range(5 if probe else 1)])
        elif name.startswith("evaluation-"):
            arm, reports = name.removeprefix("evaluation-"), []
            for identity, common_item in common.items():
                if arm == "human-reference" and common_item["spec"]["komi_halfpoints"] != 13:
                    continue
                artifacts = {}
                for kind, value in (("metrics", metrics()), ("sampling", {"samples_per_position": 32}),
                                    ("timing", {"scope": "synthetic", "measurements": []})):
                    path = worker / identity / (kind + ".json")
                    artifact = ref(path, value)
                    artifacts[path.name] = {"path": str(path.relative_to(worker)), "sha256": artifact["sha256"]}
                prediction_path = worker / identity / "predictions.npz"
                prediction_path.write_bytes(b"Only the saved hash is read; this fixture is not an array file.")
                artifacts["predictions.npz"] = {"path": str(prediction_path.relative_to(worker)),
                                                "sha256": file_hash(prediction_path)}
                predicted[identity, arm] = {"path": str(prediction_path), "sha256": file_hash(prediction_path)}
                reports.append({"status": "completed", "set": identity, "variant": "ordinary", "rows": 2,
                    "common_set_sha256": common_item["manifest"]["sha256"], "artifacts": artifacts,
                    "out_of_training_support_komi_halfpoints": [1] if arm.startswith("baseline") and identity.endswith("komi1") else []})
            summary["reports"] = reports
        elif name.startswith("matches-"):
            slots = [{"id": f"slot-{index}", "arm": arm, "profile_id": "weak_uniform" if offset < 2 else "capture_first",
                      "candidate_color": "B" if offset % 2 == 0 else "W"}
                     for index, (arm, offset) in enumerate((arm, offset)
                         for arm in ("conditioned", "unconditioned", "human_sl") for offset in range(4))]
            schedule = ref(worker / "schedule.json", {"slots": slots})
            records = {}
            for slot in (slots[0], slots[4], slots[8]):
                identity = slot["id"]
                sgf = worker / "matches" / (identity + ".sgf")
                sgf.parent.mkdir(exist_ok=True)
                sgf.write_text("(;GM[1]SZ[19]KM[6.5];B[aa];W[ba];B[])")
                record = {"slot": slot, "state": {"origin": "empty", "initial": {"board": ["." * 19] * 19,
                    "to_play": "B", "captured_black": 0, "captured_white": 0}, "moves": [["B", 0], ["W", 1], ["B", 361]]},
                    "actual_actions": 3, "sgf_file": sgf.name, "sgf_sha256": file_hash(sgf), "result": "U",
                    "status": "truncated", "reason": "synthetic cap", "behavior": {"unresolved": True},
                    "candidate_profile": {"conditioning_rank": "5k"}, "censored": True,
                    "candidate_passes": [{"move_index": 2, "verdict": "unjudgeable"}]}
                record_pin = ref(worker / "matches" / (identity + ".json"), record)["sha256"]
                records[identity] = {"record_sha256": record_pin, "sgf_sha256": file_hash(sgf)}
            authorization = ref(worker / "evidence_authorization.json",
                {"schedule_file_sha256": schedule["sha256"], "records": records})
            summary.update(status="incomplete", scheduled_games=12, recorded_games=3, records_replayed=3,
                schedule_file_sha256=schedule["sha256"], evidence_authorization_sha256=authorization["sha256"],
                unverified_slots=[], missing_slots=[slot["id"] for slot in slots if slot["id"] not in records])
        elif name == "analysis":
            comparisons, specs = {}, []
            for identity in review.SETS:
                contrasts = {}
                for first, second in ((review.ARMS[0], review.ARMS[1]), (review.ARMS[2], review.ARMS[3]),
                                      (review.ARMS[2], review.ARMS[0]), (review.ARMS[3], review.ARMS[1])):
                    key = first + "_minus_" + second
                    report = metrics(delta=True)
                    contrasts[key] = {"report": ref(worker / (identity + "-" + key + ".json"), report),
                                      "overall": report["overall"]}
                comparisons[identity] = contrasts
                mapping = {}
                for arm in review.ARMS:
                    key = identity + "-" + arm
                    config["inputs"][key] = predicted[identity, arm]
                    mapping[arm] = key
                specs.append({"id": identity, "prediction_inputs": mapping})
            summary["comparisons"] = comparisons
            config["parameters"]["common_sets"] = specs
        config_ref = ref(directory / "config.json", config)
        provenance_ref = ref(directory / "provenance.json", {"inputs": config["inputs"]})
        summary.update(config_file_sha256=config_ref["sha256"], provenance_sha256=provenance_ref["sha256"])
        stages[name] = {"directory": str(directory), "config": config_ref, "provenance": provenance_ref,
            "summary": ref(worker / "summary.json", summary), "wandb_url": "https://forge.coreweave.com/wandb/example/run",
            "job": {"status": "completed", "costs": {"wall_seconds": 2., "cpu_seconds": 1.}}}
        entries.append({"run_id": run.name + "-" + name,
                        "gpu_seconds": 5000. if name in review.ARMS else 100. if name.startswith("probe-") else 0.,
                        "cpu_seconds": 1., "wall_seconds": 2.,
                        "outcome": "completed"})
    charged = sum(entry["gpu_seconds"] for entry in entries)
    main = {"schema": "expanded_e2_pilot_v1", "status": "completed", "promotion": "not_promoted", "plan": plan_ref,
        "stages": stages, "matched_training": True,
        "training_decision": ref(run / "frozen-training.json", {"updates": 8, "probe_weights_reused": False}),
        "costs": {"elapsed_seconds": 40., "cpu_seconds": 30., "output_bytes": 10000,
            "launch_utc": "2026-11-18T14:11:48.993557Z", "deadline_utc": "2026-11-19T14:11:48.993557Z",
            "ledger": {"charged_gpu_seconds": charged, "remaining_gpu_seconds": 86400 - charged,
                       "budget_breached": False, "state": {"entries": entries, "reservations": {}}}}}
    ref(run / "summary.json", main)
    return run, main


class ExpandedReviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run, self.main = fixture(self.root)

    def rerun_fixture(self):
        original_plan = json.loads(Path(self.main["plan"]["path"]).read_text())
        original_plan["parameters"]["output"] = str(self.run)
        self.main["plan"] = ref(Path(self.main["plan"]["path"]), original_plan)
        previous_path, previous, _ = self.completion_fixture()
        run, main = fixture(self.root / "new-evaluation", name="review-synthetic")
        main.update(schema=review.RERUN_SCHEMA, previous_result={"path": str(previous_path), "sha256": file_hash(previous_path)},
            training_decision=previous["training_decision"],
            reused_training={arm: copy.deepcopy(previous["stages"][arm]) for arm in review.ARMS},
            stages={name: stage for name, stage in main["stages"].items() if name in review.RERUN_STAGES})
        plan = {"purpose": "expanded-review", "seed": 20261118,
            "parameters": {"review_ranks": review.REVIEW_RANKS},
            "inputs": {"previous_result": main["previous_result"], "original_plan": previous["plan"],
                       "training_decision": main["training_decision"]}}
        main["plan"] = ref(run.parent / "plan.json", plan)
        old_entries = copy.deepcopy(previous["costs"]["ledger"]["state"]["entries"])
        new_entries = [entry for entry in main["costs"]["ledger"]["state"]["entries"]
                       if entry["run_id"] in {run.name + "-" + name for name in review.RERUN_STAGES}]
        main["costs"]["ledger"]["state"]["entries"] = old_entries + new_entries
        main["costs"].update(cpu_seconds=12., elapsed_seconds=20.)
        preparation = main["stages"]["preparation"]
        prepared = json.loads(Path(preparation["summary"]["path"]).read_text())
        common = {}
        for item in prepared["common_sets"]:
            item["spec"]["aggregate_ranks"] = review.ALL_RANKS
            item["coverage"] = {"by": {"source_rank_mover": [
                {"source_rank_mover": rank, "moves": int(rank in {"14k", "5k"})} for rank in review.ALL_RANKS]}}
            manifest_path = Path(item["manifest"]["path"])
            manifest = json.loads(manifest_path.read_text())
            manifest.update(aggregate_ranks=review.ALL_RANKS, coverage=item["coverage"])
            item["manifest"] = ref(manifest_path, manifest)
            common[item["spec"]["id"]] = item
        prepared["selection"] = ref(run / "preparation/worker/selection.json",
            {"review_ranks": review.REVIEW_RANKS, "sets": [item["spec"] for item in common.values()]})
        preparation["summary"] = ref(Path(preparation["summary"]["path"]), prepared)
        for name, stage in main["stages"].items():
            summary_path = Path(stage["summary"]["path"])
            summary = json.loads(summary_path.read_text())
            if name.startswith("evaluation-"):
                for report in summary["reports"]:
                    report["common_set_sha256"] = common[report["set"]]["manifest"]["sha256"]
                    artifact = report["artifacts"]["metrics.json"]
                    path = Path(stage["directory"]) / "worker" / artifact["path"]
                    artifact["sha256"] = ref(path, review_metrics())["sha256"]
            elif name == "analysis":
                for comparisons in summary["comparisons"].values():
                    for item in comparisons.values():
                        report = review_metrics(delta=True)
                        item.update(report=ref(Path(item["report"]["path"]), report), overall=report["overall"])
            stage["summary"] = ref(summary_path, summary)
        ref(run / "summary.json", main)
        return run, main, previous

    def completion_fixture(self, *, partial_reference=True):
        completion = copy.deepcopy(self.main)
        for arm in review.ARMS:
            stage = completion["stages"][arm]
            config = json.loads(Path(stage["config"]["path"]).read_text())
            config["inputs"]["training_decision"] = completion["training_decision"]
            stage["config"] = ref(Path(stage["config"]["path"]), config)
            stage["provenance"] = ref(Path(stage["provenance"]["path"]), {"inputs": config["inputs"]})
            summary = json.loads(Path(stage["summary"]["path"]).read_text())
            summary.update(config_file_sha256=stage["config"]["sha256"], provenance_sha256=stage["provenance"]["sha256"])
            stage["summary"] = ref(Path(stage["summary"]["path"]), summary)
        human = completion["stages"].pop(review.REFERENCE_STAGE)
        worker = self.run / review.REFERENCE_STAGE / "worker"
        preparation = json.loads(Path(completion["stages"]["preparation"]["summary"]["path"]).read_text())
        common = {item["spec"]["id"]: item for item in preparation["common_sets"] if item["spec"]["komi_halfpoints"] == 13}
        config = {"purpose": "baseline-evaluation", "arm": "human-reference", "inputs": {}, "parameters": {"common_sets": []}}
        for identity, item in common.items():
            key = "common_" + identity
            config["inputs"][key] = item["manifest"]
            config["parameters"]["common_sets"].append(item["spec"] | {"manifest_input": key})
        config_ref = ref(worker.parent / "config.json", config)
        provenance_ref = ref(worker.parent / "provenance.json", {"inputs": config["inputs"]})
        job_ref = ref(worker / "job.json", {"schema": "bounded_pilot_job_v1", "status": "failed", "reason": "cpu_budget",
            "run_id": self.run.name + "-" + review.REFERENCE_STAGE,
            "command": ["python", "-m", "kenggo.baseline_evaluation", config_ref["path"], provenance_ref["path"], str(worker),
                        "--config-sha256", config_ref["sha256"], "--provenance-sha256", provenance_ref["sha256"]]})
        failed_stage = {"name": review.REFERENCE_STAGE, "directory": str(worker.parent),
            "artifacts": {"config.json": config_ref, "provenance.json": provenance_ref, "worker/job.json": job_ref}}
        reports = json.loads(Path(human["summary"]["path"]).read_text())["reports"]
        sets = {}
        for index, report in enumerate(reports):
            identity = report["set"]
            state = {"status": "completed" if index == 0 else "running", "completed_rows": 2 if index == 0 else 1,
                     "expected_rows": 2, "common_set_sha256": common[identity]["manifest"]["sha256"]}
            if index == 0:
                report.update(role=common[identity]["spec"]["role"], arm="human-reference")
                summary_ref = ref(worker / identity / "ordinary" / "summary.json", report)
                state["summary"] = {"path": str(Path(summary_ref["path"]).relative_to(worker)), "sha256": summary_ref["sha256"]}
            sets[identity + "/ordinary"] = state
        progress_ref = ref(worker / "progress.json", {"version": "baseline_evaluation_v1", "status": "running",
            "arm": "human-reference", "config_sha256": config_ref["sha256"], "provenance_sha256": provenance_ref["sha256"], "sets": sets})
        original = copy.deepcopy(completion)
        original.update(status="incomplete", reason="Human reference reached its CPU cap", failed_stage=failed_stage,
            stages={name: stage for name, stage in completion["stages"].items() if name not in review.FINISH_STAGES})
        original.pop("training_decision")
        old_entries = [entry for entry in original["costs"]["ledger"]["state"]["entries"]
                       if entry["run_id"] not in {self.run.name + "-" + name for name in review.FINISH_STAGES}]
        original["costs"]["ledger"]["state"]["entries"] = old_entries
        original_ref = ref(self.run / "failure.json", original)
        finish_plan = {"version": "experiment_config_v1", "purpose": "expanded-finish",
                       "inputs": {"original_failure": original_ref}, "parameters": {"scope": "remaining checks only"}}
        completion.update(status="incomplete", completion_status="completed", completion_reason=None,
            reason=original["reason"], failed_stage=failed_stage, original_failure=original_ref)
        if partial_reference:
            completion["partial_reference_progress"] = finish_plan["inputs"]["partial_reference_progress"] = progress_ref
        completion["completion_plan"] = ref(self.root / "completion-plan.json", finish_plan)
        new_entries = [entry for entry in completion["costs"]["ledger"]["state"]["entries"]
                       if entry["run_id"] in {self.run.name + "-" + name for name in review.FINISH_STAGES}]
        completion["costs"]["ledger"]["state"]["entries"] = old_entries + [
            {"run_id": self.run.name + "-first-verification", "gpu_seconds": 0., "cpu_seconds": 10., "wall_seconds": 12.}
        ] + new_entries
        completion["costs"].update(cpu_seconds=60., elapsed_seconds=80.)
        path = self.run / "completion.json"
        ref(path, completion)
        return path, completion, original

    def test_rerun_retains_scopes_games_and_historical_training_without_recharging_it(self):
        from kenggo.expanded_review import STAGES, VERSION
        self.assertEqual((review.RERUN_STAGES, review.RERUN_SCHEMA), (STAGES, VERSION))
        run, main, previous = self.rerun_fixture()
        previous_bytes = Path(main["previous_result"]["path"]).read_bytes()
        result = review.collect(run)
        self.assertTrue(result["review_rerun"])
        self.assertTrue(result["matched_training"])
        self.assertEqual(result["default_scope"], "review_scope")
        self.assertEqual([stage["name"] for stage in result["stages"]], list(STAGES))
        self.assertEqual(len(result["training"]), 4)
        self.assertEqual(result["probes"], {})
        self.assertEqual(len(result["games"]), 24)
        self.assertEqual(result["costs"]["run_gpu_seconds"], 0)
        self.assertEqual(result["costs"]["cpu_seconds"], 12)
        self.assertEqual(result["historical_costs"]["cpu_seconds"], 60)
        self.assertEqual(result["historical_costs"]["gpu_seconds_by_stage"][review.ARMS[0]], 5000)
        self.assertEqual(result["costs"]["prior_gpu_seconds"], previous["costs"]["ledger"]["charged_gpu_seconds"])
        self.assertEqual(result["training_decision"]["updates"], 8)
        for common in result["common_sets"].values():
            self.assertEqual(common["review_scope"]["expected_outputs"], 1)
            self.assertEqual(common["review_scope"]["excluded_rows"], 1)
        for reports in result["evaluations"].values():
            for report in reports.values():
                self.assertEqual(report["metrics"]["aggregate_ranks"], review.ALL_RANKS)
                self.assertEqual(report["metrics"]["review_scope"]["aggregate_ranks"], review.REVIEW_RANKS)
                self.assertEqual(report["metrics"]["expected_outputs"], 2)
        self.assertNotIn('"row_key"', json.dumps(result))
        self.assertEqual(Path(main["previous_result"]["path"]).read_bytes(), previous_bytes)
        self.check_page_controls(result)

    def test_rerun_recovers_complete_reference_sets_after_worker_or_tracking_failure(self):
        run, main, _ = self.rerun_fixture()
        human = main["stages"].pop(review.REFERENCE_STAGE)
        worker = run / review.REFERENCE_STAGE / "worker"
        prepared = json.loads(Path(main["stages"]["preparation"]["summary"]["path"]).read_text())
        common = {item["spec"]["id"]: item for item in prepared["common_sets"] if item["spec"]["komi_halfpoints"] == 13}
        config = {"purpose": "baseline-evaluation", "arm": "human-reference", "inputs": {}, "parameters": {"common_sets": []}}
        for identity, item in common.items():
            key = "common_" + identity
            config["inputs"][key] = item["manifest"]
            config["parameters"]["common_sets"].append(item["spec"] | {"manifest_input": key})
        config_ref = ref(worker.parent / "config.json", config)
        provenance_ref = ref(worker.parent / "provenance.json", {"inputs": config["inputs"]})
        pins = {"config.json": config_ref, "provenance.json": provenance_ref}
        sets = {}
        reports = json.loads(Path(human["summary"]["path"]).read_text())["reports"]
        for report in reports:
            identity = report["set"]
            report.update(role=common[identity]["spec"]["role"], arm="human-reference")
            saved = ref(worker / identity / "ordinary/summary.json", report)
            sets[identity + "/ordinary"] = {"status": "completed", "completed_rows": 2, "expected_rows": 2,
                "common_set_sha256": common[identity]["manifest"]["sha256"],
                "summary": {"path": str(Path(saved["path"]).relative_to(worker)), "sha256": saved["sha256"]}}
        progress = {"version": "baseline_evaluation_v1", "status": "completed", "arm": "human-reference",
            "config_sha256": config_ref["sha256"], "provenance_sha256": provenance_ref["sha256"], "sets": sets}
        main.update(status="incomplete", reason="Late reference or tracking failure",
            failed_stage={"name": review.REFERENCE_STAGE, "directory": str(worker.parent), "artifacts": pins})
        for artifact, job_status in (("progress.json", "failed"), ("summary.json", "completed"), ("failure.json", "failed")):
            with self.subTest(artifact=artifact):
                for key in list(pins):
                    if key.startswith("worker/"):
                        del pins[key]
                pins["worker/" + artifact] = ref(worker / artifact, progress)
                pins["worker/job.json"] = ref(worker / "job.json", {"schema": "bounded_pilot_job_v1",
                    "status": job_status, "run_id": run.name + "-" + review.REFERENCE_STAGE,
                    "command": ["python", "--config-sha256", config_ref["sha256"],
                                "--provenance-sha256", provenance_ref["sha256"]]})
                path = run / "failure.json"
                ref(path, main)
                data = review.collect(path)
                self.assertEqual(sum("human-reference" in reports for reports in data["evaluations"].values()), 2)
                self.assertEqual(next(stage["status"] for stage in data["stages"] if stage["name"] == review.REFERENCE_STAGE), "failed")
                self.assertEqual(data["partial_reference"]["job_status"], job_status)
        state = next(iter(sets.values()))
        state.update(status="running", completed_rows=1)
        del state["summary"]
        pins["worker/failure.json"] = ref(worker / "failure.json", progress)
        ref(path, main)
        data = review.collect(path)
        self.assertEqual(sum("human-reference" in reports for reports in data["evaluations"].values()), 1)
        (worker / "failure.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "expected hash"):
            review.collect(path)

    def test_full_size_statistics_fit_after_compaction_without_losing_estimates(self):
        from kenggo.recorded_statistics import evaluate, paired_report
        from test_recorded_statistics import row, prediction
        run, _, _ = self.rerun_fixture()
        data = review.collect(run)
        rows = []
        for game in range(142):
            for ply in (0, 1, 60, 61, 180, 181):
                accounts = [f"{2 * game:064x}", f"{2 * game + 1:064x}"]
                if ply % 2:
                    accounts.reverse()
                rows.append(row(f"https://synthetic.invalid/{game:064x}", *accounts,
                    rank=review.ALL_RANKS[game % 16], mover="W" if ply % 2 else "B", ply=ply))
        predictions = [prediction(item, 1 + index % 3) for index, item in enumerate(rows)]
        report = evaluate(rows, predictions, aggregate_ranks=review.ALL_RANKS)
        selected = [(item, output) for item, output in zip(rows, predictions) if item["source_rank_mover"] in review.REVIEW_RANKS]
        scoped_rows, scoped_predictions = map(list, zip(*selected))
        report["review_scope"] = evaluate(scoped_rows, scoped_predictions, aggregate_ranks=review.REVIEW_RANKS)
        paired = paired_report(rows, predictions, predictions, aggregate_ranks=review.ALL_RANKS)
        paired["review_scope"] = paired_report(scoped_rows, scoped_predictions, scoped_predictions, aggregate_ranks=review.REVIEW_RANKS)
        original = json.dumps(report, sort_keys=True)
        compact, compact_paired = review._display_details(report), review._display_details(paired)
        self.assertEqual(json.dumps(report, sort_keys=True), original)
        for scope in (report, report["review_scope"]):
            for name, metric in scope["overall"]["metrics"].items():
                displayed = (compact if scope is report else compact["review_scope"])["overall"]["metrics"][name]
                for key in ("estimate", "counts", "status"):
                    self.assertEqual(displayed[key], metric[key])
                self.assertEqual(displayed["uncertainty"]["interval95"], metric["uncertainty"]["interval95"])
        for identity, reports in data["evaluations"].items():
            for item in reports.values():
                item["metrics"] = compact
            data["paired"][identity] = {name: compact_paired for name in data["paired"][identity]}
        page = review.render(data)
        self.assertLess(len(page.encode()), review.MAX_PAGE_BYTES)
        self.assertNotIn('"game_influences"', page)
        self.assertEqual(len(data["full_reports"]), 70)
        for reference in data["full_reports"]:
            self.assertEqual(file_hash(reference["path"]), reference["sha256"])

    def test_rerun_rejects_changed_reuse_predecessor_scope_and_historical_charges(self):
        run, main, _ = self.rerun_fixture()
        changes = []
        changed = copy.deepcopy(main)
        changed["reused_training"][review.ARMS[0]]["job"]["costs"]["cpu_seconds"] = 0
        changes.append(changed)
        changed = copy.deepcopy(main)
        del changed["reused_training"][review.ARMS[1]]
        changes.append(changed)
        changed = copy.deepcopy(main)
        changed["previous_result"]["sha256"] = "b" * 64
        changes.append(changed)
        changed = copy.deepcopy(main)
        changed["training_decision"]["sha256"] = "b" * 64
        changes.append(changed)
        changed = copy.deepcopy(main)
        changed["costs"]["ledger"]["state"]["entries"][0]["cpu_seconds"] = 0
        changes.append(changed)
        changed = copy.deepcopy(main)
        changed["costs"]["ledger"]["state"]["entries"][-1]["gpu_seconds"] = 1
        changes.append(changed)
        changed = copy.deepcopy(main)
        changed["matched_training"] = False
        changes.append(changed)
        changed = copy.deepcopy(main)
        del changed["stages"][review.REFERENCE_STAGE]
        changes.append(changed)
        changed = copy.deepcopy(main)
        changed["stages"][review.ARMS[0]] = changed["reused_training"][review.ARMS[0]]
        changes.append(changed)
        for index, changed in enumerate(changes):
            ref(run / "summary.json", changed)
            with self.subTest(change=index), self.assertRaises(ValueError):
                review.collect(run)
        changed = copy.deepcopy(main)
        plan = json.loads(Path(changed["plan"]["path"]).read_text())
        plan["parameters"]["review_ranks"] = review.ALL_RANKS
        changed["plan"] = ref(Path(changed["plan"]["path"]), plan)
        ref(run / "summary.json", changed)
        with self.assertRaisesRegex(ValueError, "rank scope"):
            review.collect(run)

    def test_rerun_rejects_wrong_metric_and_paired_scope_ranks_or_counts(self):
        run, main, _ = self.rerun_fixture()
        for name in ("evaluation-" + review.ARMS[0], "analysis"):
            original = main["stages"][name]
            summary_path = Path(original["summary"]["path"])
            saved_summary = json.loads(summary_path.read_text())
            for field in ("aggregate_ranks", "expected_outputs", "excluded_rows", "overall", "by_rank", "missing"):
                changed = copy.deepcopy(main)
                stage, summary = changed["stages"][name], copy.deepcopy(saved_summary)
                if name == "analysis":
                    item = next(iter(summary["comparisons"][review.SETS[0]].values()))
                    path = Path(item["report"]["path"])
                else:
                    item = summary["reports"][0]["artifacts"]["metrics.json"]
                    path = Path(stage["directory"]) / "worker" / item["path"]
                value = review_metrics(delta=name == "analysis")
                scoped = value["review_scope"]
                if field == "aggregate_ranks":
                    scoped[field] = review.ALL_RANKS
                elif field in {"expected_outputs", "excluded_rows"}:
                    scoped[field] += 1
                elif field == "overall":
                    scoped[field]["metrics"]["nll"]["aggregate_ranks"] = review.ALL_RANKS
                elif field == "by_rank":
                    scoped[field]["5k"]["counts"]["moves"] = 2
                else:
                    del value["review_scope"]
                pin = ref(path, value)
                if name == "analysis":
                    item["report"] = pin
                else:
                    item["sha256"] = pin["sha256"]
                stage["summary"] = ref(summary_path, summary)
                ref(run / "summary.json", changed)
                with self.subTest(stage=name, field=field), self.assertRaisesRegex(ValueError, "review"):
                    review.collect(run)
            if name != "analysis":
                artifact = saved_summary["reports"][0]["artifacts"]["metrics.json"]
                ref(Path(original["directory"]) / "worker" / artifact["path"], review_metrics())
                ref(summary_path, saved_summary)

    def test_rerun_fails_when_rank_coverage_or_prediction_binding_changes(self):
        run, main, _ = self.rerun_fixture()
        changed = copy.deepcopy(main)
        stage = changed["stages"]["preparation"]
        summary_path = Path(stage["summary"]["path"])
        original_summary = json.loads(summary_path.read_text())
        summary = copy.deepcopy(original_summary)
        item = summary["common_sets"][0]
        item["coverage"]["by"]["source_rank_mover"][0]["moves"] = 10
        stage["summary"] = ref(summary_path, summary)
        ref(run / "summary.json", changed)
        with self.assertRaisesRegex(ValueError, "rank coverage"):
            review.collect(run)
        ref(summary_path, original_summary)
        changed = copy.deepcopy(main)
        stage = changed["stages"]["analysis"]
        config_path = Path(stage["config"]["path"])
        config = json.loads(config_path.read_text())
        key = config["parameters"]["common_sets"][0]["prediction_inputs"][review.ARMS[0]]
        config["inputs"][key] = copy.deepcopy(config["inputs"][key])
        config["inputs"][key]["path"] = str(run / "wrong-predictions.npz")
        stage["config"] = ref(config_path, config)
        stage["provenance"] = ref(Path(stage["provenance"]["path"]), {"inputs": config["inputs"]})
        summary = json.loads(Path(stage["summary"]["path"]).read_text())
        summary.update(config_file_sha256=stage["config"]["sha256"], provenance_sha256=stage["provenance"]["sha256"])
        stage["summary"] = ref(Path(stage["summary"]["path"]), summary)
        ref(run / "summary.json", changed)
        with self.assertRaisesRegex(ValueError, "different saved predictions"):
            review.collect(run)

    def test_rerun_failure_before_training_reuse_is_explicit(self):
        run, main, _ = self.rerun_fixture()
        main.update(status="incomplete", reason="Saved-model validation failed", stages={}, reused_training={}, matched_training=False)
        path = run / "failure.json"
        ref(path, main)
        result = review.collect(path)
        self.assertEqual((result["status"], result["training"], result["games"]), ("incomplete", {}, []))
        self.assertEqual(len(result["stages"]), 9)
        self.assertTrue(all(stage["status"] == "unavailable" for stage in result["stages"]))
        self.assertFalse(result["matched_training"])

    def test_completed_remaining_checks_preserve_incomplete_reference_and_cumulative_costs(self):
        path, completion, original = self.completion_fixture()
        original_bytes = (self.run / "failure.json").read_bytes()
        result = review.collect(path)
        self.assertEqual((result["status"], result["completion_status"]), ("incomplete", "completed"))
        self.assertEqual(result["reason"], original["reason"])
        self.assertEqual(len(result["games"]), 24)
        self.assertEqual(sum(len(contrasts) for contrasts in result["paired"].values()), 16)
        self.assertIn("human-reference", result["evaluations"][review.SETS[0]])
        self.assertNotIn("human-reference", result["evaluations"][review.SETS[2]])
        self.assertEqual(result["partial_reference"]["job_status"], "failed")
        self.assertEqual([item["report_available"] for item in result["partial_reference"]["sets"]], [True, False])
        reference_stage = next(stage for stage in result["stages"] if stage["name"] == review.REFERENCE_STAGE)
        self.assertEqual(reference_stage["status"], "failed")
        self.assertEqual(result["costs"]["cumulative_gpu_seconds"], original["costs"]["ledger"]["charged_gpu_seconds"])
        self.assertEqual((result["costs"]["cpu_seconds"], result["costs"]["elapsed_seconds"]), (60., 80.))
        self.assertEqual((self.run / "failure.json").read_bytes(), original_bytes)
        finish_plan = json.loads(Path(completion["completion_plan"]["path"]).read_text())
        del finish_plan["inputs"]["partial_reference_progress"]
        del completion["partial_reference_progress"]
        completion["completion_plan"] = ref(Path(completion["completion_plan"]["path"]), finish_plan)
        ref(path, completion)
        without = review.collect(path)
        self.assertIsNone(without["partial_reference"])
        self.assertTrue(all("human-reference" not in reports for reports in without["evaluations"].values()))

    def test_completion_rejects_changed_predecessor_stage_failure_plan_or_ledger(self):
        path, completion, _ = self.completion_fixture()
        changes = []
        changed = copy.deepcopy(completion)
        changed["original_failure"]["sha256"] = "b" * 64
        changes.append(changed)
        changed = copy.deepcopy(completion)
        changed["stages"][review.ARMS[0]]["job"]["costs"]["cpu_seconds"] = 0
        changes.append(changed)
        changed = copy.deepcopy(completion)
        changed["failed_stage"]["artifacts"].pop("worker/job.json")
        changes.append(changed)
        changed = copy.deepcopy(completion)
        changed["costs"]["ledger"]["state"]["entries"][0]["wall_seconds"] = 0
        changes.append(changed)
        changed = copy.deepcopy(completion)
        changed["costs"]["cpu_seconds"] = 0
        changes.append(changed)
        changed = copy.deepcopy(completion)
        changed["reason"] = "The original failure disappeared"
        changes.append(changed)
        for key in ("launch_utc", "deadline_utc"):
            changed = copy.deepcopy(completion)
            changed["costs"][key] = "2026-11-20T14:11:48.993557Z"
            changes.append(changed)
        for index, changed in enumerate(changes):
            with self.subTest(change=index):
                ref(path, changed)
                with self.assertRaises(ValueError):
                    review.collect(path)
        finish_plan = json.loads(Path(completion["completion_plan"]["path"]).read_text())
        finish_plan["inputs"]["original_failure"]["sha256"] = "b" * 64
        completion["completion_plan"] = ref(Path(completion["completion_plan"]["path"]), finish_plan)
        ref(path, completion)
        with self.assertRaisesRegex(ValueError, "bind the predecessor"):
            review.collect(path)

    def test_completed_remaining_checks_require_all_three_new_stages(self):
        path, completion, _ = self.completion_fixture()
        for name in review.FINISH_STAGES:
            changed = copy.deepcopy(completion)
            del changed["stages"][name]
            ref(path, changed)
            with self.subTest(stage=name), self.assertRaisesRegex(ValueError, "analysis and both match panels"):
                review.collect(path)
        changed = copy.deepcopy(completion)
        changed["completion_status"] = "missing"
        ref(path, changed)
        with self.assertRaisesRegex(ValueError, "cannot complete"):
            review.collect(path)
        changed = copy.deepcopy(completion)
        stage = changed["stages"]["matches-combined"]
        summary_path = Path(stage["summary"]["path"])
        summary = json.loads(summary_path.read_text())
        summary["status"] = "failed"
        stage["summary"] = ref(summary_path, summary)
        ref(path, changed)
        with self.assertRaisesRegex(ValueError, "unfinished required stage"):
            review.collect(path)

    def test_partial_reference_requires_completed_full_sets_and_consistent_progress_pins(self):
        path, completion, _ = self.completion_fixture()
        progress_path = Path(completion["partial_reference_progress"]["path"])
        progress = json.loads(progress_path.read_text())
        finish_plan = json.loads(Path(completion["completion_plan"]["path"]).read_text())
        changes = []
        changed = copy.deepcopy(progress)
        changed["sets"][review.SETS[0] + "/ordinary"]["completed_rows"] = 1
        changes.append(changed)
        changed = copy.deepcopy(progress)
        changed["sets"][review.SETS[2] + "/ordinary"]["summary"] = changed["sets"][review.SETS[0] + "/ordinary"]["summary"]
        changes.append(changed)
        changed = copy.deepcopy(progress)
        changed["config_sha256"] = "b" * 64
        changes.append(changed)
        changed = copy.deepcopy(progress)
        changed["sets"][review.SETS[0] + "/ordinary"]["summary"]["sha256"] = "b" * 64
        changes.append(changed)
        for index, changed in enumerate(changes):
            tested = copy.deepcopy(completion)
            tested["partial_reference_progress"] = ref(progress_path, changed)
            plan = copy.deepcopy(finish_plan)
            plan["inputs"]["partial_reference_progress"] = tested["partial_reference_progress"]
            tested["completion_plan"] = ref(Path(tested["completion_plan"]["path"]), plan)
            ref(path, tested)
            with self.subTest(change=index), self.assertRaises(ValueError):
                review.collect(path)

    def test_remaining_check_failure_keeps_both_failures_and_missing_panel_unavailable(self):
        path, completion, original = self.completion_fixture()
        name = "matches-combined"
        del completion["stages"][name]
        artifact = ref(self.run / name / "worker" / "failure.json", {"status": "failed", "reason": "synthetic finishing cap"})
        completion.update(completion_status="failed", completion_reason="Remaining match panel reached its cap",
            completion_failed_stage={"name": name, "directory": str(self.run / name),
                                     "artifacts": {"worker/failure.json": artifact}})
        ref(path, completion)
        result = review.collect(path)
        self.assertEqual(result["reason"], original["reason"])
        self.assertEqual(result["completion_reason"], completion["completion_reason"])
        self.assertEqual(result["completion_failure"]["name"], name)
        self.assertEqual(result["failure"]["name"], review.REFERENCE_STAGE)
        self.assertNotIn("combined", result["panels"])
        self.assertEqual(len(result["games"]), 12)
        self.assertEqual(sum(stage["status"] == "failed" for stage in result["stages"]), 2)

    def test_completed_results_retain_all_arms_sets_slots_costs_and_display_frames(self):
        prepared = json.loads(Path(self.main["stages"]["preparation"]["summary"]["path"]).read_text())
        common = prepared["common_sets"][0]
        manifest_path = Path(common["manifest"]["path"])
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["source_ids"], list(reversed(common["spec"]["source_ids"])))
        data = review.collect(self.run)
        self.assertEqual(file_hash(manifest_path), common["manifest"]["sha256"])
        self.assertTrue(data["matched_training"])
        self.assertEqual(len(data["training"]), 4)
        self.assertEqual(len(data["games"]), 24)
        self.assertEqual(sum(game["worker_verified"] for game in data["games"]), 6)
        for identity, reports in data["evaluations"].items():
            self.assertEqual(len(reports), 5 if identity.endswith("komi13") else 4)
            self.assertEqual(reports[review.ARMS[0]]["metrics"]["overall"]["metrics"]["nll"]["uncertainty"]["status"], "inconclusive")
        game = data["games"][0]
        self.assertEqual(game["frames"], [[[], "B", 0, 0, 0], [[[0, "B"]], "W", 0, 0, 0],
                                         [[[1, "W"]], "B", 0, 0, 0], [[], "W", 0, 0, 1]])
        self.assertEqual(game["passes"][0]["verdict"], "unjudgeable")
        self.assertAlmostEqual(data["costs"]["prior_gpu_seconds"], 1246.177333)
        self.assertEqual(data["costs"]["run_gpu_seconds"], 20400)
        self.assertEqual(data["costs"]["gpu_seconds_by_stage"][review.ARMS[0]], 5000)
        self.assertEqual(data["costs"]["gpu_seconds_by_stage"]["probe-" + review.ARMS[0]], 100)
        self.assertFalse((self.root / "missing-dataset.parquet").exists())

    def test_duplicate_common_sources_are_rejected_even_when_membership_matches(self):
        main = copy.deepcopy(self.main)
        stage = main["stages"]["preparation"]
        path = Path(stage["summary"]["path"])
        prepared = json.loads(path.read_text())
        common = prepared["common_sets"][0]
        manifest_path = Path(common["manifest"]["path"])
        manifest = json.loads(manifest_path.read_text())
        manifest["source_ids"].append(manifest["source_ids"][0])
        common["manifest"] = ref(manifest_path, manifest)
        stage["summary"] = ref(path, prepared)
        ref(self.run / "summary.json", main)
        with self.assertRaisesRegex(ValueError, "Frozen common set differs"):
            review.collect(self.run)

    def test_changed_plan_report_prediction_or_sgf_pins_are_rejected(self):
        paths = [self.root / "plan.json", self.run / "analysis" / "worker" / (review.SETS[0] + "-" + review.ARMS[0] + "_minus_" + review.ARMS[1] + ".json"),
                 self.run / ("evaluation-" + review.ARMS[0]) / "worker" / review.SETS[0] / "predictions.npz",
                 self.run / "matches-baseline" / "worker" / "matches" / "slot-0.sgf"]
        for path in paths:
            old = path.read_bytes()
            try:
                path.write_bytes(b"changed")
                with self.subTest(path=path), self.assertRaises(ValueError):
                    review.collect(self.run)
            finally:
                path.write_bytes(old)

    def test_display_allows_the_white_candidates_last_reply_at_action_801(self):
        state = {"origin": "empty", "initial": {"board": ["." * 19] * 19,
            "to_play": "B", "captured_black": 0, "captured_white": 0},
            "moves": [["B" if index % 2 == 0 else "W", 361] for index in range(801)]}
        self.assertEqual(len(review._display_frames(state, 801)), 802)
        state["moves"].append(["W", 361])
        with self.assertRaises(ValueError):
            review._display_frames(state, 802)

    def test_missing_top_or_stage_status_is_not_treated_as_completion(self):
        changed = copy.deepcopy(self.main)
        del changed["status"]
        ref(self.run / "summary.json", changed)
        with self.assertRaisesRegex(ValueError, "explicit status"):
            review.collect(self.run)
        changed = copy.deepcopy(self.main)
        stage = changed["stages"][review.ARMS[0]]
        path = Path(stage["summary"]["path"])
        summary = json.loads(path.read_text())
        del summary["status"]
        stage["summary"] = ref(path, summary)
        ref(self.run / "summary.json", changed)
        with self.assertRaisesRegex(ValueError, "explicit status"):
            review.collect(self.run)

    def test_failure_exports_missing_metrics_and_checked_partial_stage_without_inventing_results(self):
        failed = copy.deepcopy(self.main)
        failed.update(status="incomplete", reason="Synthetic worker failure", stages={"preparation": self.main["stages"]["preparation"]})
        failed.pop("training_decision")
        name = review.ARMS[0]
        artifact = ref(self.run / name / "worker" / "training_failure.json",
                       {"status": "failed", "completed_updates": 2, "reason": "Synthetic interrupted update"})
        failed["failed_stage"] = {"name": name, "directory": str(self.run / name),
                                  "artifacts": {"worker/training_failure.json": artifact}}
        path = self.run / "failure.json"
        ref(path, failed)
        result = review.collect(path)
        self.assertFalse(result["matched_training"])
        self.assertEqual(result["training"], {})
        self.assertEqual(result["games"], [])
        self.assertTrue(all(not value for value in result["evaluations"].values()))
        self.assertEqual(next(stage for stage in result["stages"] if stage["name"] == name)["status"], "failed")
        self.assertEqual(result["failure"]["artifacts"]["worker/training_failure.json"]["completed_updates"], 2)

    def test_export_escapes_script_data_and_refuses_to_overwrite_outputs(self):
        data = review.collect(self.run)
        data["reason"] = "</script><img src=x onerror=alert(1)>\u2028\u2029"
        page = review.render(data)
        self.assertNotIn(data["reason"], page)
        self.assertIn("\\u003c/script>", page)
        payload = page.split('<script id="expanded-data" type="application/json">', 1)[1].split("</script>", 1)[0]
        self.assertEqual(json.loads(payload)["reason"], data["reason"])
        self.assertEqual(page.count("<script "), 2)
        self.assertNotIn("__EXPANDED_DATA_JSON__", page)
        self.assertIn("connect-src 'none'", page)
        self.assertNotIn("innerHTML", page)
        output = self.root / "review.html"
        manifest = review.export(self.run, output)
        self.assertEqual(manifest["html_sha256"], file_hash(output))
        self.assertEqual((manifest["raw_dataset_reads"], manifest["checkpoint_reads"], manifest["inference_calls"]), (0, 0, 0))
        with self.assertRaisesRegex(ValueError, "unused"):
            review.export(self.run, output)

    def test_page_script_selectors_replay_notes_and_downloads_without_a_browser(self):
        self.check_page_controls(review.collect(self.run))

    def test_completion_page_keeps_reference_failure_visible_and_controls_working(self):
        path, completion, _ = self.completion_fixture()
        completion.update(completion_status="failed", completion_reason="Synthetic publication failure")
        ref(path, completion)
        self.check_page_controls(review.collect(path))

    def check_page_controls(self, data):
        node = shutil.which("node")
        if node is None:
            candidates = sorted((Path.home() / ".nvm/versions/node").glob("*/bin/node"))
            node = str(candidates[-1]) if candidates else None
        if node is None:
            self.skipTest("A local Node runtime is unavailable; browser QA remains separate")
        page = review.render(data)
        script = page.split('<script id="expanded-script">', 1)[1].split("</script>", 1)[0]
        ids = re.findall(r'\bid="([^"]+)"', page.split('<script id="expanded-data"', 1)[0])
        harness = r'''
const assert = require('node:assert/strict');
class Element {
 constructor(tag){this.tagName=tag;this.children=[];this.attributes={};this.listeners={};this._value=undefined;this.textContent='';this.disabled=false;this.nodeType=1;}
 setAttribute(k,v){this.attributes[k]=v;if(k==='value')this._value=v;}
 append(...nodes){this.children.push(...nodes);}
 replaceChildren(...nodes){this.children=[...nodes];if(this.tagName==='select')this._value=undefined;}
 addEventListener(name,handler){this.listeners[name]=handler;}
 get value(){return this._value??(this.tagName==='select'?this.children[0]?.value??'':'');}
 set value(value){this._value=String(value);}
 click(){this.listeners.click?.();}
 remove(){}
}
const elements=Object.fromEntries(IDS.map(id=>[id,new Element(id.includes('select')||['game-cohort','game-arm'].includes(id)?'select':'div')]));
elements['expanded-data']=new Element('script');elements['expanded-data'].textContent=JSON.stringify(DATA);
elements['game-cohort'].value='all';elements['game-arm'].value='all';elements['breakdown-select'].value='overall';
const document={getElementById:id=>{assert.ok(elements[id],id);return elements[id];},createElement:tag=>new Element(tag),createElementNS:(ns,tag)=>new Element(tag),body:new Element('body')};
const localStorage={data:new Map(),getItem(key){return this.data.get(key)??null;},setItem(key,value){this.data.set(key,value);}};
let downloads=0;URL.createObjectURL=()=>{downloads++;return 'blob:synthetic';};URL.revokeObjectURL=()=>{};
const selected=report=>DATA.review_rerun?report.review_scope:report;
selected(DATA.evaluations[DATA.sets[0]]['human-reference'].metrics).overall.metrics.nll.estimate={status:'positive_infinity',value:null};
selected(Object.values(DATA.paired[DATA.sets[0]])[0]).overall.metrics.nll.estimate={status:'negative_infinity',value:null};
selected(DATA.evaluations[DATA.sets[0]]['combined-unconditioned'].metrics).overall.metrics.nll.estimate={status:'undefined',value:null};
elements['expanded-data'].textContent=JSON.stringify(DATA);
SCRIPT
const textOf=node=>[node.textContent,...node.children.map(textOf)].join(' ');
if(DATA.completion_status){
 assert.ok(elements['run-status'].textContent.includes('Experiment incomplete'));
 assert.ok(elements['run-status'].textContent.includes('remaining checks failed'));
 assert.equal(elements['completion-readout'].hidden,false);
 assert.ok(elements['completion-readout'].textContent.includes(DATA.completion_reason));
 assert.ok(textOf(elements['failure-readout']).includes(DATA.reason));
 assert.ok(textOf(elements['failure-readout']).includes(DATA.completion_reason));
 assert.ok(textOf(elements['reference-progress']).includes('Last saved worker observations'));
 assert.ok(textOf(elements['reference-progress']).includes('Unavailable'));
 assert.ok(!DATA.evaluations[DATA.sets[2]]['human-reference']);
}
assert.ok(textOf(elements['metric-cards']).includes('+∞'));
assert.ok(textOf(elements['metric-cards']).includes('Undefined'));
assert.ok(textOf(elements['metric-table']).includes('+∞'));
assert.ok(textOf(elements['comparison-chart']).includes('+∞ · outside finite scale'));
assert.ok(textOf(elements['paired-table']).includes('−∞'));
const trainingText=textOf(elements['training-table']),probeText=textOf(elements['probe-table']);
assert.ok(trainingText.includes('1.389 h'));
assert.ok(!trainingText.includes('0.028 h'));
if(DATA.review_rerun){
 assert.equal(elements['scope-select'].value,'review_scope');
 assert.equal(elements['scope-select'].children.length,2);
 assert.ok(elements['set-context'].textContent.includes('12k–1d'));
 assert.ok(elements['set-context'].textContent.includes('1 included positions; 1 weaker-rank positions excluded'));
 assert.ok(textOf(elements['metric-cards']).includes('1.25'));
 assert.ok(!textOf(elements['metric-cards']).includes('2.1'));
 assert.ok(textOf(elements['paired-table']).includes('-0.08'));
 assert.ok(textOf(elements['calibration']).includes('0.125'));
 assert.ok(textOf(elements['all-results']).includes('1.25'));
 assert.ok(trainingText.includes('Historical GPU allocation charge'));
 assert.ok(trainingText.includes('Historical CPU charge'));
 assert.equal(probeText.trim(),'');
 assert.ok(textOf(elements['cost-cards']).includes('New production CPU'));
 assert.ok(textOf(elements['training-context']).includes('not spent again'));
 assert.ok(textOf(elements['review-steps']).includes('kenggo-ydo.14'));
 assert.ok(textOf(elements['review-steps']).includes('kenggo-ydo.6'));
 elements['scope-select'].value='all_ranks';elements['scope-select'].listeners.change();
 assert.ok(elements['set-context'].textContent.includes('all 16 ranks (diagnostic)'));
 assert.ok(textOf(elements['metric-cards']).includes('2.1'));
 assert.ok(!textOf(elements['metric-cards']).includes('1.25'));
 assert.ok(textOf(elements['paired-table']).includes('-0.04'));
 assert.ok(textOf(elements['calibration']).includes('0.875'));
 assert.ok(textOf(elements['all-results']).includes('2.1'));
 elements['breakdown-select'].value='by_rank';elements['breakdown-select'].listeners.change();
 assert.ok(textOf(elements['metric-table']).includes('14k'));
 elements['scope-select'].value='review_scope';elements['scope-select'].listeners.change();
 assert.ok(!textOf(elements['metric-table']).includes('14k'));
 elements['breakdown-select'].value='overall';elements['breakdown-select'].listeners.change();
}else assert.ok(probeText.includes('0.028 h'));
assert.equal(elements['set-select'].children.length,4);
assert.equal(elements['game-select'].children.length,24);
elements['game-arm'].value='conditioned';elements['game-arm'].listeners.change();assert.equal(elements['game-select'].children.length,8);
elements['game-cohort'].value='baseline';elements['game-cohort'].listeners.change();assert.equal(elements['game-select'].children.length,4);
elements['next-move'].click();assert.equal(elements['move-slider'].value,'1');
elements['game-notes'].value='Inspect move 1';elements['game-notes'].listeners.input();assert.equal(localStorage.data.size,1);
elements['download-sgf'].click();elements['download-notes'].click();assert.equal(downloads,2);
elements['set-select'].value=DATA.sets[1];elements['set-select'].listeners.change();
assert.ok(elements['set-context'].textContent.includes('outside training support'));
elements['breakdown-select'].value='by_rank';elements['breakdown-select'].listeners.change();
assert.ok(elements['metric-table'].children.length);
assert.ok(elements['stage-table'].children.length);
console.log('selectors, replay, local notes, downloads, and breakdowns passed');
'''.replace("IDS", json.dumps(ids), 1).replace("SCRIPT", script, 1)
        harness = "const DATA = " + json.dumps(data) + ";\n" + harness
        path = self.root / "ui-check.cjs"
        path.write_text(harness)
        result = subprocess.run([node, str(path)], text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
