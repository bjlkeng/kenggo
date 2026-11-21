"""Synthetic orchestration checks without training, acquisition or tracking calls."""

from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from kenggo import progressive_jobs as jobs
from kenggo import progressive_evaluation as evaluation
from kenggo import progressive_training as training
from kenggo.checkpoints import json_bytes, save_json
from kenggo.continuation_data import read, reference


SOURCE = {"commit": "a" * 40, "patch_sha256": sha256(b"").hexdigest()}
LIMITS = {"wall_seconds": 604800, "cpu_seconds": 604800, "gpu_seconds": 604800,
          "output_bytes": 160 * 1024**3, "publication_seconds": 21600}


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        self.input = self.save("input.json", {"fixture": True})
        self.parent = self.dataset("parent", 128)
        artifacts = read(self.parent)["artifacts"]
        self.base = {"data": self.parent, "rules": self.input, "canonical": artifacts["canonical"],
            "split": artifacts["split_manifest"], "examples": artifacts["examples_manifest"],
            "profile": artifacts["profile"], "normalizer": artifacts["normalizer"]}
        old_directory = self.root / "old"
        old_checkpoint = self.save("old/worker/checkpoint.json", {"checkpoint": "original"})
        old_config = self.save("old/config.json", {"inputs": self.base})
        old_provenance = self.save("old/provenance.json", {"source": SOURCE})
        old_summary = self.save("old/worker/training_summary.json", {"final_checkpoint": {
            "path": "checkpoint.json", "sha256": old_checkpoint["sha256"], "step": 10}})
        old = self.save("old/summary.json", {"stages": {"combined-conditioned": {
            "directory": str(old_directory), "config": old_config,
            "provenance": old_provenance, "summary": old_summary}}})
        self.plan = {"version": "experiment_config_v1", "entity": "fixture", "project": "fixture",
            "base_url": "https://api.wandb.ai", "comparison": "progressive-fixture", "arm": "shared",
            "purpose": "progressive-study", "budget_id": "synthetic", "seed": 23,
            "inputs": {"parent_summary": self.parent, "previous_training": old,
                "fixed_comparison": self.input, "prior_resources": self.input, "data": self.parent,
                "split": self.base["split"], "rules": self.input},
            "parameters": {"base_inputs": self.base, "limits": dict(LIMITS),
                "output": str(self.root / "study"),
                "model": {"blocks": 1, "channels": 16, "rank_conditioned": True},
                "discovery_refs": [self.input], "source_refs": [self.input]}}
        self.plan_ref = self.save("plan.json", self.plan)

    def save(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(json_bytes(value))
        return reference(path)

    def dataset(self, name, rows):
        examples = self.save(name + "/examples.json", {"counts": {"train": {"examples": rows}}})
        return self.save(name + "/summary.json", {"artifacts": {
            "examples_manifest": examples, **{key: self.save(name + "/" + key + ".json", {"dataset": name})
                for key in ("canonical", "split_manifest", "profile", "normalizer")}}})


class SyntheticStudy:
    def __init__(self, fixture, *, decisions=(False,), plateau_epoch=2, shortfall=False,
                 partial_stage=None, pair_budget=None):
        self.fixture, self.plan = fixture, fixture.plan
        self.output = fixture.root / "synthetic-study"
        self.output.mkdir()
        self.state = {"tiers": [], "stages": {}}
        self.calls, self.requirements, self.checkpoints = [], [], {}
        self.decisions, self.plateau_epoch, self.shortfall = decisions, plateau_epoch, shortfall
        self.partial_stage, self.pair_budget, self.pair_checks = partial_stage, pair_budget, 0
        self.growth_requests, self.preparations = [], {}

    def config(self, **kwargs):
        return training.Study.config(self, **kwargs)

    def persist(self):
        pass

    def require(self, *args, **kwargs):
        self.requirements.append((args, kwargs))
        if kwargs.get("gpu", 0):
            self.pair_checks += 1
            if self.pair_budget is not None and self.pair_checks > self.pair_budget:
                raise training.StopStudy("Synthetic shared budget exhausted")

    def stage(self, name, config, **options):
        self.calls.append({"name": name, "config": deepcopy(config), "options": options})
        purpose, params = config["purpose"], config["parameters"]
        tier = int(re.search(r"tier(\d+)", name)[1])
        prefix = "synthetic-study/" + name
        config_ref = self.fixture.save(prefix + "/config.json", config)
        provenance_ref = self.fixture.save(prefix + "/provenance.json", {"source": SOURCE, "inputs": config["inputs"]})
        summary = {"status": "completed"}
        if purpose == "progressive-sources":
            summary["sources"] = self.fixture.save(prefix + "/sources.json", {"scope": "synthetic source freeze"})
        elif purpose == "progressive-growth":
            request = read(config["inputs"]["request"])
            self.growth_requests.append(request)
            complete = not self.shortfall or "collection-growth" in name
            rows = request["target_train_rows"] if complete else request["target_train_rows"] - 64
            summary.update(dataset_summary=self.fixture.dataset(prefix + "/dataset", rows),
                           growth={"target_within_one_game": complete})
        elif purpose == "progressive-collection":
            sources = self.fixture.save(prefix + "/sources.json", {"collected": True})
            discovery = self.fixture.save(prefix + "/discovery.json", {"rows": []})
            summary["result"] = {**sources, "discovery": discovery, "status": "partial", "counts": {"selected_candidates": 1}}
        elif purpose == "progressive-preparation":
            summary.update(train_monitor=self.fixture.input,
                epoch_validation=self.fixture.save(prefix + "/validation.json", {"tier": tier}),
                assessments={"validation_development_komi13": self.fixture.input})
            self.preparations[tier] = summary
        elif purpose == "continuation-training":
            epoch = int(name[-2:])
            arm = config["arm"]
            checkpoint = {**self.fixture.save(prefix + "/checkpoint.json", {"name": name}),
                          "config": config_ref, "provenance": provenance_ref, "step": 10 + epoch * 2}
            self.checkpoints[(tier, arm, epoch)] = checkpoint
            best = self.checkpoints[(tier, arm, min(epoch, 1))]
            summary.update(latest_checkpoint=checkpoint, best_checkpoint=best,
                           patience_exhausted=epoch >= self.plateau_epoch)
        elif purpose == "progressive-comparison":
            summary["decision"] = {"continue_growth": self.decisions[tier - 1]}
        elif purpose not in ("progressive-assessment", "progressive-readiness"):
            raise AssertionError("Unexpected synthetic stage: " + purpose)
        summary_ref = self.fixture.save(prefix + "/summary.json", summary)
        status = "partial" if name == self.partial_stage else "completed"
        stage = {"directory": str(self.output / name), "status": status, "summary": summary_ref,
                 "config": config_ref, "provenance": provenance_ref,
                 "job": {"status": "failed" if status == "partial" else "completed", "reason": None}}
        self.state["stages"][name] = stage
        return stage, summary

    def execute(self):
        with patch.object(training, "available_memory", return_value=1024**4):
            return training.execute(self)


class ExecuteTests(unittest.TestCase):
    def test_a_budget_limited_cache_build_uses_remaining_cache_before_collecting(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            study = SyntheticStudy(fixture)
            original = study.stage
            partial = fixture.dataset("partial-cache", 192)

            def stage(name, config, **options):
                result, summary = original(name, config, **options)
                if name == "tier1-growth":
                    summary.update(dataset_summary=partial, growth={"target_within_one_game": False,
                                                                   "source_unattempted": 2})
                return result, summary

            study.stage = stage
            study.execute()
            purposes = [item["config"]["purpose"] for item in study.calls]
            self.assertNotIn("progressive-collection", purposes)
            self.assertEqual(purposes.count("progressive-growth"), 2)
            self.assertEqual(study.growth_requests[1]["parent_summary"], partial)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.fixture = Fixture(temporary.name)

    def test_paired_updates_alternate_order_keep_epoch_zero_and_assess_restored_best(self):
        study = SyntheticStudy(self.fixture)
        study.execute()
        self.assertEqual(study.calls[0]["config"]["purpose"], "progressive-readiness")
        calls = [item for item in study.calls if item["config"]["purpose"] == "continuation-training"]
        self.assertEqual([item["config"]["arm"] for item in calls],
                         ["control", "candidate", "candidate", "control", "control", "candidate"])
        for index, call in enumerate(calls):
            config, epoch = call["config"], index // 2
            self.assertIs(config["parameters"]["evaluate_only"], epoch == 0)
            self.assertEqual(config["parameters"]["round_updates"], 2)
            self.assertEqual(config["parameters"]["training_rows"], 128 if config["arm"] == "control" else 256)
            self.assertEqual(config["inputs"]["normalizer"], self.fixture.base["normalizer"])
            self.assertEqual(config["inputs"]["normalization_examples"], self.fixture.base["examples"])
            if epoch:
                self.assertEqual(config["inputs"]["parent_checkpoint"],
                    training.pin(study.checkpoints[(1, config["arm"], epoch - 1)]))
        tier = study.state["tiers"][0]
        self.assertEqual(tier["completed_intervals"], 2)
        self.assertEqual(tier["training_stop"], "validation_plateau")
        assessments = [item for item in study.calls if item["config"]["purpose"] == "progressive-assessment"]
        self.assertEqual([item["config"]["arm"] for item in assessments], ["parent", "control", "candidate"])
        for item in assessments:
            self.assertTrue(item["options"]["final"])
            arm = item["config"]["arm"]
            expected = training.initial_checkpoint(study.plan) if arm == "parent" else study.checkpoints[(1, arm, 1)]
            self.assertEqual(item["config"]["inputs"]["checkpoint"], training.pin(expected))
            self.assertEqual(item["config"]["inputs"]["assessment_validation_development_komi13"], self.fixture.input)

    def test_budget_exhaustion_after_a_pair_still_runs_fixed_assessment(self):
        study = SyntheticStudy(self.fixture, pair_budget=2, plateau_epoch=20)
        study.execute()
        tier = study.state["tiers"][0]
        self.assertEqual(tier["training_stop"], "shared_budget")
        self.assertEqual(tier["completed_intervals"], 1)
        self.assertEqual(set(tier["assessments"]), {"parent", "control", "candidate"})
        self.assertEqual(tier["status"], "completed")
        self.assertEqual(len(tier["training"]), 4)

    def test_budget_exhaustion_before_any_pair_cannot_select_or_assess(self):
        study = SyntheticStudy(self.fixture, pair_budget=0)
        with self.assertRaises(training.StopStudy):
            study.execute()
        tier = study.state["tiers"][0]
        self.assertNotIn("selected", tier)
        self.assertFalse(tier["assessments"])

    def test_continuing_improvement_can_pass_twenty_intervals_until_the_shared_budget_stops(self):
        study = SyntheticStudy(self.fixture, pair_budget=22, plateau_epoch=1000)
        study.execute()
        tier = study.state["tiers"][0]
        self.assertEqual(tier["completed_intervals"], 21)
        self.assertEqual(len(tier["training"]), 44)
        self.assertEqual(tier["training_stop"], "shared_budget")
        self.assertEqual(set(tier["assessments"]), {"parent", "control", "candidate"})

    def test_partial_interval_keeps_the_last_complete_pair_for_assessment(self):
        study = SyntheticStudy(self.fixture, partial_stage="tier1-candidate-epoch02", plateau_epoch=20,
                               decisions=(True, False))
        with self.assertRaisesRegex(training.StopStudy, "interrupted"):
            study.execute()
        tier = study.state["tiers"][0]
        self.assertEqual(len(study.state["tiers"]), 1)
        self.assertEqual(tier["status"], "partial_training")
        self.assertEqual(tier["training_stop"], "interrupted_interval")
        self.assertEqual(tier["comparison_status"], "completed")
        self.assertEqual(tier["incomplete_interval"]["epoch"], 2)
        self.assertEqual(tier["incomplete_interval"]["arm"], "candidate")
        self.assertEqual(tier["incomplete_interval"]["reason"], "The interval did not finish")
        self.assertEqual(tier["completed_intervals"], 1)
        self.assertEqual(tier["training"][-1]["stage"]["status"], "partial")
        self.assertEqual(tier["selected"]["control"], study.checkpoints[(1, "control", 1)])
        self.assertEqual(tier["selected"]["candidate"], study.checkpoints[(1, "candidate", 1)])

    def test_worker_failure_without_a_summary_still_assesses_the_last_complete_pair(self):
        study = SyntheticStudy(self.fixture, plateau_epoch=20, decisions=(True, False))
        original = study.stage

        def stage(name, config, **options):
            if name == "tier1-candidate-epoch02":
                study.state["failed_attempts"] = [{"name": name, "reason": "worker killed before summary"}]
                raise RuntimeError("worker killed before summary")
            return original(name, config, **options)

        study.stage = stage
        with self.assertRaisesRegex(training.StopStudy, "interrupted"):
            study.execute()
        tier = study.state["tiers"][0]
        self.assertEqual(len(study.state["tiers"]), 1)
        self.assertEqual(tier["status"], "partial_training")
        self.assertEqual(tier["completed_intervals"], 1)
        self.assertEqual(tier["comparison_status"], "completed")
        self.assertEqual(set(tier["assessments"]), {"parent", "control", "candidate"})
        self.assertEqual(tier["incomplete_interval"]["attempt"], study.state["failed_attempts"][0])
        self.assertIn("worker killed", tier["incomplete_interval"]["reason"])
        for arm in ("control", "candidate"):
            self.assertEqual(tier["selected"][arm], study.checkpoints[(1, arm, 1)])

    def test_worker_failure_before_the_initial_pair_cannot_create_a_comparison(self):
        study = SyntheticStudy(self.fixture)
        original = study.stage

        def stage(name, config, **options):
            if name == "tier1-candidate-epoch00":
                raise RuntimeError("worker killed before summary")
            return original(name, config, **options)

        study.stage = stage
        with self.assertRaisesRegex(training.StopStudy, "No complete matched"):
            study.execute()
        tier = study.state["tiers"][0]
        self.assertEqual(tier["training_stop"], "interrupted_interval")
        self.assertFalse(tier["assessments"])
        self.assertNotIn("selected", tier)

    def test_only_a_useful_comparison_starts_the_next_tier_from_the_selected_candidate(self):
        study = SyntheticStudy(self.fixture, decisions=(True, False))
        study.execute()
        first, second = study.state["tiers"]
        self.assertEqual([first["target_rows"], second["target_rows"]], [256, 512])
        self.assertEqual(second["parent_data"], first["dataset"])
        self.assertEqual(second["parent_checkpoint"], study.checkpoints[(1, "candidate", 1)])
        self.assertNotEqual(second["parent_checkpoint"], study.checkpoints[(1, "candidate", 2)])
        preparation = next(item["config"] for item in study.calls if item["name"] == "tier2-preparation")
        self.assertEqual(preparation["inputs"]["previous_preparation_0"], first["preparation"])
        self.assertNotEqual(study.preparations[1]["epoch_validation"], study.preparations[2]["epoch_validation"])
        self.assertFalse(second["decision"]["continue_growth"])

    def test_collection_result_uses_path_hash_and_continues_the_valid_partial_dataset(self):
        study = SyntheticStudy(self.fixture, shortfall=True)
        study.execute()
        first, second = study.growth_requests
        collection = next(item for item in study.calls if item["name"] == "tier1-collection")
        self.assertEqual(second["parent_summary"], collection["config"]["inputs"]["parent_summary"])
        self.assertNotEqual(first["parent_summary"], second["parent_summary"])
        self.assertEqual(set(second["sources"]), {"path", "sha256"})
        self.assertEqual(read(second["sources"]), {"collected": True})
        self.assertEqual(first["target_train_rows"], second["target_train_rows"])
        self.assertEqual(second["seed"], str(study.plan["seed"]))
        self.assertEqual(len([item for item in study.calls if item["config"]["purpose"] == "progressive-collection"]), 1)

    def test_partial_required_stage_cannot_complete_a_tier_or_authorize_more_growth(self):
        for stage in ("tier1-sources", "tier1-growth", "tier1-preparation",
                      "tier1-assess-candidate", "tier1-comparison"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as temporary:
                study = SyntheticStudy(Fixture(temporary), partial_stage=stage, decisions=(True, False))
                with self.assertRaises(training.StopStudy):
                    study.execute()
                self.assertEqual(len(study.state["tiers"]), 1)
                self.assertNotEqual(study.state["tiers"][0]["status"], "completed")
                self.assertEqual(study.state["stages"][stage]["status"], "partial")

    def test_memory_refusal_starts_no_source_or_training_job(self):
        study = SyntheticStudy(self.fixture)
        with patch.object(training, "available_memory", return_value=1024**3):
            with self.assertRaisesRegex(training.StopStudy, "memory"):
                training.execute(study)
        self.assertFalse(study.calls)
        self.assertFalse(study.state["tiers"])


class Clock:
    def __init__(self):
        self.wall, self.cpu = 0., 10.
        self.utc_offset = 0.
        self.origin = datetime(2026, 11, 21, tzinfo=timezone.utc)

    def patches(self):
        clock = self
        class Time(datetime):
            @classmethod
            def now(cls, tz=None):
                return clock.origin + timedelta(seconds=clock.wall + clock.utc_offset)
        stack = ExitStack()
        stack.enter_context(patch.object(training, "datetime", Time))
        stack.enter_context(patch.object(training.time, "monotonic", side_effect=lambda: self.wall))
        stack.enter_context(patch.object(training.time, "process_time", side_effect=lambda: self.cpu))
        return stack


class LocalExperiment:
    def __init__(self, directory, config, provenance, source_patch):
        self.directory, self.config = Path(directory), config
        self.directory.mkdir(parents=True)
        self.config_sha256 = save_json(self.directory / "config.json", config)
        self.provenance_sha256 = save_json(self.directory / "provenance.json", provenance)
        self.attempt, self.registered, self.logged, self.finished = {"last_step": None}, [], [], None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def register_checkpoint(self, ref):
        self.registered.append(ref)

    def log(self, step, values):
        self.attempt["last_step"] = step
        self.logged.append((step, values))

    def finish(self, outcome, **kwargs):
        self.finished = outcome


class StageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.fixture = Fixture(temporary.name)

    def study(self):
        return training.Study(self.fixture.plan, self.fixture.plan_ref, self.fixture.root / "study")

    def test_failed_launch_cannot_reset_its_allowance_in_the_same_or_another_directory(self):
        clock = Clock()
        with clock.patches():
            study = self.study()
            study.finish("failed", "synthetic failed attempt")
            saved = {name: (study.output / name).read_bytes()
                     for name in ("session.json", "resources.json", "summary.json")}
            clock.wall = 3600
            with self.assertRaisesRegex(FileExistsError, "restart is unsupported"):
                self.study()
            alternate = self.fixture.root / "second-allowance"
            with self.assertRaisesRegex(ValueError, "cannot reset"):
                training.Study(self.fixture.plan, self.fixture.plan_ref, alternate)
            self.assertFalse(alternate.exists())
            self.assertEqual(saved, {name: (study.output / name).read_bytes() for name in saved})

    def test_pair_admission_preserves_all_three_assessments_comparison_and_publication(self):
        study = self.study()
        costs = deepcopy(study.costs())
        wall = training.epoch_limit(4_000_000)
        pair_cpu, pair_gpu = 2 * wall * 1.2 + 3600, 2 * (wall + 10)
        gib = 1024**3
        costs["cpu_seconds"] = LIMITS["cpu_seconds"] - pair_cpu - 600
        with patch.object(study, "costs", return_value=costs):
            with self.assertRaises(training.StopStudy):
                study.require(2 * (wall + 240) + 3600, pair_cpu, 4 * gib, gpu=pair_gpu)
            costs["cpu_seconds"] -= training.FINAL_RESERVE["cpu_seconds"] + 1
            costs["ledger"]["remaining_gpu_seconds"] = pair_gpu + training.FINAL_RESERVE["gpu_seconds"] + 31
            costs["output_bytes"] = LIMITS["output_bytes"] - 4 * gib - training.FINAL_RESERVE["output_bytes"] - gib / 4 - 1
            study.require(2 * (wall + 240) + 3600, pair_cpu, 4 * gib, gpu=pair_gpu)
            costs["cpu_seconds"] += 2 * wall * 1.2 + 360
            costs["ledger"]["remaining_gpu_seconds"] -= pair_gpu
            costs["output_bytes"] += 4 * gib
            for _ in range(3):
                study.require(3780, 3780, gib, final=True, gpu=3610)
                costs["cpu_seconds"] += 3780
                costs["ledger"]["remaining_gpu_seconds"] -= 3610
                costs["output_bytes"] += gib
            study.require(780, 780, 2 * gib, final=True)
            costs["cpu_seconds"] += 780
            costs["output_bytes"] += 2 * gib
            study.require(600, 600, gib, final=True)
        study.finish("completed", None)

    def test_exact_seven_day_limits_deadline_and_publication_reserve(self):
        clock = Clock()
        with clock.patches():
            study = self.study()
            self.assertEqual((study.deadline - study.launch).total_seconds(), 604800)
            self.assertEqual((study.deadline - study.production_end).total_seconds(), 21600)
            self.assertEqual(study.state["prior_costs"], self.fixture.plan["inputs"]["prior_resources"])
            clock.wall = 604800 - 21600 - 400
            with self.assertRaises(training.StopStudy):
                study.require(600, 10, 100)
            study.require(600, 10, 100, final=True)
            clock.wall = 604801
            clock.utc_offset = -clock.wall
            with self.assertRaises(training.StopStudy):
                study.require(1, 1, 1, final=True)
            clock.wall, clock.utc_offset = 0, 604801
            with self.assertRaises(training.StopStudy):
                study.require(1, 1, 1, final=True)
            study.finish("budget_or_evidence_stop", "synthetic deadline")
        self.assertFalse(study.ledger.snapshot()["state"]["reservations"])
        for field in LIMITS:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                changed = deepcopy(self.fixture.plan)
                changed["parameters"]["limits"][field] += 1
                with self.assertRaisesRegex(ValueError, "seven-day"):
                    training.Study(changed, self.fixture.plan_ref, Path(temporary) / "invalid")

    def test_stage_passes_pins_process_limits_and_device_and_counts_cpu_once(self):
        clock = Clock()
        experiments, jobs_seen = [], []
        with clock.patches():
            study = self.study()
            config = study.config(purpose="progressive-assessment", arm="candidate",
                inputs=self.fixture.base, parameters={"device": "cuda"})
            original = deepcopy(config)
            def experiment(*args):
                value = LocalExperiment(*args)
                experiments.append(value)
                return value
            def run_job(command, output, **kwargs):
                jobs_seen.append((command, kwargs))
                output.mkdir()
                save_json(output / "summary.json", {"status": "completed", "metric": 0.})
                ledger = kwargs["ledger"]
                ledger.reserve(kwargs["run_id"], estimate_gpu_seconds=600, shutdown_gpu_seconds=10, metadata={})
                ledger.record(kwargs["run_id"], allocations=[{"device": "0", "start_utc": clock.origin.isoformat(),
                    "stop_utc": (clock.origin + timedelta(seconds=60)).isoformat()}], outcome="completed",
                    arm_shares={"candidate": 1}, cpu_seconds=7, wall_seconds=60)
                clock.wall, clock.cpu = 90, 20
                return {"status": "completed", "reason": None, "costs": {"cpu_seconds": 7, "wall_seconds": 60},
                        "parent_cpu_seconds": 3}
            with patch.object(training, "capture_provenance", return_value=({"source": SOURCE}, b"")), \
                    patch.object(training, "Experiment", side_effect=experiment), \
                    patch.object(training, "run_job", side_effect=run_job):
                stage, summary = study.stage("assessment", config, wall=600, cpu=900, output_bytes=1024**3, gpu=True)
            self.assertEqual(config, original)
            self.assertEqual(stage["status"], "completed")
            self.assertEqual(summary["metric"], 0.)
            command, options = jobs_seen[0]
            self.assertIn("kenggo.progressive_jobs", command)
            self.assertEqual(options["limits"].wall_seconds, 600)
            self.assertEqual(options["limits"].cpu_seconds, 900)
            self.assertEqual(options["device"], "0")
            self.assertEqual(options["environment"]["CUDA_VISIBLE_DEVICES"], "0")
            self.assertEqual(options["environment"]["UV_OFFLINE"], "1")
            self.assertIn(experiments[0].config_sha256, command)
            self.assertIn(experiments[0].provenance_sha256, command)
            deadline = datetime.fromisoformat(experiments[0].config["parameters"]["deadline_utc"])
            self.assertEqual(deadline, clock.origin + timedelta(seconds=540))
            self.assertLessEqual(deadline, study.production_end)
            self.assertEqual(study.costs()["cpu_seconds"], 74)
            finished = study.finish("completed", None)
            self.assertEqual(finished["costs"]["cpu_seconds"], 74)
            self.assertFalse(study.ledger.snapshot()["state"]["reservations"])
            self.assertEqual(study.ledger.snapshot()["charged_gpu_seconds"], 60)

    def test_every_shared_resource_limit_stops_before_launch(self):
        study = self.study()
        baseline = study.costs()
        for resource in ("cpu", "output", "gpu", "breach", "elapsed"):
            with self.subTest(resource=resource):
                costs = deepcopy(baseline)
                if resource == "cpu":
                    costs["cpu_seconds"] = LIMITS["cpu_seconds"] - 500
                elif resource == "output":
                    costs["output_bytes"] = LIMITS["output_bytes"] - 256 * 1024**2
                elif resource == "gpu":
                    costs["ledger"]["remaining_gpu_seconds"] = 0
                elif resource == "breach":
                    costs["ledger"]["budget_breached"] = True
                else:
                    costs["elapsed_seconds"] = LIMITS["wall_seconds"] - 1
                with patch.object(study, "costs", return_value=costs), patch.object(training, "run_job") as launch:
                    with self.assertRaises(training.StopStudy):
                        study.stage("blocked", {}, wall=600)
                    launch.assert_not_called()
        study.finish("budget_or_evidence_stop", "synthetic resource limits")

    def test_training_stage_registers_only_owned_checkpoints_and_logs_both_diagnostics(self):
        study = self.study()
        config = study.config(purpose="continuation-training", arm="candidate",
            inputs=self.fixture.base, parameters={"round_updates": 2})
        experiments, commands = [], []
        def experiment(*args):
            value = LocalExperiment(*args)
            experiments.append(value)
            return value
        def run_job(command, output, **kwargs):
            commands.append(command)
            output.mkdir()
            save_json(output / "checkpoint.json", {"latest": True})
            latest = reference(output / "checkpoint.json")
            save_json(output / "metrics.json", {"lifetime_updates": 12,
                "training_monitor": {"nll": 0., "rows": 16, "not_a_metric": True},
                "validation": {"nll": 1.25, "top1": 0.4}})
            save_json(output / "commit.json", {"metrics": reference(output / "metrics.json")})
            save_json(output / "training_summary.json", {"status": "completed", "checkpoints": [latest],
                "latest_checkpoint": latest, "best_checkpoint": self.fixture.input,
                "commits": [reference(output / "commit.json")]})
            return {"status": "completed", "reason": None, "parent_cpu_seconds": 0,
                    "costs": {"cpu_seconds": 0, "wall_seconds": 0}}
        with patch.object(training, "capture_provenance", return_value=({"source": SOURCE}, b"")), \
                patch.object(training, "Experiment", side_effect=experiment), \
                patch.object(training, "run_job", side_effect=run_job):
            stage, summary = study.stage("training", config, wall=600, gpu=True)
        self.assertEqual(stage["status"], "completed")
        self.assertIn("kenggo.continuation_training", commands[0])
        self.assertEqual(experiments[0].registered, [summary["latest_checkpoint"]])
        metrics = experiments[0].logged[0]
        self.assertEqual(metrics[0], 12)
        self.assertEqual(metrics[1]["training_monitor/nll"], 0.)
        self.assertEqual(metrics[1]["validation/nll"], 1.25)
        self.assertNotIn("training_monitor/not_a_metric", metrics[1])
        study.finish("completed", None)

    def test_partial_or_missing_worker_summary_keeps_failure_costs_and_cannot_look_completed(self):
        for summary in ({"status": "partial", "completed_updates": 100}, {"status": 1, "completed_updates": 100}, None):
            with self.subTest(summary=summary), tempfile.TemporaryDirectory() as temporary:
                fixture = Fixture(temporary)
                study = training.Study(fixture.plan, fixture.plan_ref, fixture.root / "study")
                config = study.config(purpose="progressive-comparison", arm="shared", inputs=fixture.base, parameters={})
                def run_job(command, output, **kwargs):
                    output.mkdir()
                    if summary is not None:
                        save_json(output / "summary.json", summary)
                    else:
                        save_json(output / "failure.json", {"status": "failed", "reason": "synthetic worker failure"})
                    ledger, name = kwargs["ledger"], kwargs["run_id"]
                    ledger.reserve(name, estimate_gpu_seconds=0, shutdown_gpu_seconds=0, metadata={})
                    ledger.record(name, allocations=[], outcome="failed", arm_shares={"shared": 1},
                                  cpu_seconds=7, wall_seconds=60, reason="synthetic worker failure")
                    return {"status": "failed", "reason": "synthetic worker failure",
                            "costs": {"cpu_seconds": 7, "wall_seconds": 60}, "parent_cpu_seconds": 0}
                with patch.object(training, "capture_provenance", return_value=({"source": SOURCE}, b"")), \
                        patch.object(training, "Experiment", LocalExperiment), \
                        patch.object(training, "run_job", side_effect=run_job):
                    if summary is None:
                        with self.assertRaisesRegex(RuntimeError, "no completed summary"):
                            study.stage("failed", config, wall=600)
                    else:
                        stage, _ = study.stage("failed", config, wall=600)
                        self.assertEqual(stage["status"], "partial")
                study.finish("failed", "synthetic worker failure")
                entries = study.ledger.snapshot()["state"]["entries"]
                worker = next(item for item in entries if item["run_id"] == "failed")
                self.assertEqual(worker["cpu_seconds"], 7)
                self.assertEqual(worker["outcome"], "failed")
                self.assertFalse(study.ledger.snapshot()["state"]["reservations"])

    def test_run_records_failure_and_still_exports_retained_evidence(self):
        from kenggo import progressive_review
        with patch.object(training, "execute", side_effect=ValueError("synthetic input mismatch")), \
                patch.object(progressive_review, "export") as export:
            with self.assertRaisesRegex(ValueError, "synthetic input mismatch"):
                training.run(self.fixture.plan_ref["path"], self.fixture.plan_ref["sha256"], self.fixture.root / "study")
        failure = json.loads((self.fixture.root / "study/failure.json").read_text())
        summary = json.loads((self.fixture.root / "study/summary.json").read_text())
        self.assertEqual(failure["type"], "ValueError")
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(export.call_count, 1)
        self.assertEqual(export.call_args.args[0], reference(self.fixture.root / "study/summary.json"))


class WorkerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.fixture = Fixture(temporary.name)

    def config(self, purpose):
        config = deepcopy(self.fixture.plan)
        config.update(purpose=purpose, inputs={**self.fixture.base, "parent_summary": self.fixture.parent,
            "discovery_0": self.fixture.input, "source_0": self.fixture.input},
            parameters={"deadline_utc": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()})
        return config

    def run_worker(self, config, *, identity=None, output_name="worker"):
        config_ref = self.fixture.save(output_name + "-config.json", config)
        provenance_ref = self.fixture.save(output_name + "-provenance.json", {"source": SOURCE, "inputs": config["inputs"]})
        output = self.fixture.root / output_name
        output.mkdir()
        with patch.object(jobs, "_source_identity", side_effect=identity or (lambda: SOURCE)):
            result = jobs.run(config_ref["path"], config_ref["sha256"], provenance_ref["path"], provenance_ref["sha256"], output)
        return result, output

    def test_collection_worker_preserves_the_real_result_reference_and_budget_callback(self):
        from kenggo import data_growth_source
        config = self.config("progressive-collection")
        def collect(parent, discoveries, sources, output, **kwargs):
            self.assertEqual(parent, self.fixture.parent)
            self.assertEqual(discoveries, [self.fixture.input])
            self.assertEqual(sources, [self.fixture.input])
            self.assertEqual(kwargs["seed"], str(config["seed"]))
            self.assertEqual(kwargs["deadline_utc"], config["parameters"]["deadline_utc"])
            kwargs["check_budget"]()
            output.mkdir()
            save_json(output / "sources.json", {"candidates": []})
            save_json(output / "discovery.json", {"rows": []})
            return {**reference(output / "sources.json"), "discovery": reference(output / "discovery.json"),
                    "status": "partial", "counts": {"accessible_candidates": 0}}
        with patch.object(data_growth_source, "collect_more", side_effect=collect):
            result, output = self.run_worker(config)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["result"]["status"], "partial")
        self.assertEqual(training.pin(result["result"]), reference(output / "collection/sources.json"))
        self.assertEqual(read(reference(output / "summary.json")), result)

    def test_source_and_growth_workers_pin_the_published_artifacts(self):
        from kenggo import data_growth, data_growth_source
        def freeze(parent, discoveries, sources, output, **kwargs):
            kwargs["check_budget"]()
            save_json(output, {"candidates": []})
            return {**reference(output), "status": "cache_exhausted", "cache_exhausted": True}
        with patch.object(data_growth_source, "freeze_sources", side_effect=freeze):
            result, output = self.run_worker(self.config("progressive-sources"), output_name="sources")
        self.assertEqual(result["sources"], reference(output / "sources.json"))
        config = self.config("progressive-growth")
        request = {"parent_summary": self.fixture.parent, "target_train_rows": 256, "seed": "23"}
        config["inputs"]["request"] = self.fixture.save("request.json", request)
        def build(value, output, **kwargs):
            self.assertEqual(value, request)
            kwargs["check_budget"]()
            output.mkdir()
            save_json(output / "summary.json", {"status": "budget_stopped", "target_within_one_game": False})
            return {"status": "budget_stopped", "target_within_one_game": False}
        with patch.object(data_growth, "build_growth_round", side_effect=build):
            result, output = self.run_worker(config, output_name="growth")
        self.assertEqual(result["dataset_summary"], reference(output / "dataset/summary.json"))
        self.assertIs(result["growth"]["target_within_one_game"], False)

    def test_expired_deadline_or_mutated_input_prevents_any_worker(self):
        config = self.config("progressive-preparation")
        config["parameters"]["deadline_utc"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        with patch.object(jobs, "prepare") as prepare:
            with self.assertRaises(TimeoutError):
                self.run_worker(config, output_name="expired")
            prepare.assert_not_called()
        config = self.config("progressive-preparation")
        Path(self.fixture.input["path"]).write_text("changed")
        with patch.object(jobs, "prepare") as prepare:
            with self.assertRaises(ValueError):
                self.run_worker(config, output_name="changed")
            prepare.assert_not_called()

    def test_source_change_during_worker_prevents_a_completed_summary(self):
        with patch.object(jobs, "prepare", return_value={"status": "completed"}):
            with self.assertRaisesRegex(ValueError, "source changed"):
                self.run_worker(self.config("progressive-preparation"),
                    identity=iter([SOURCE, {**SOURCE, "commit": "b" * 40}]).__next__)
        self.assertFalse((self.fixture.root / "worker/summary.json").exists())


class DecisionTests(unittest.TestCase):
    def comparisons(self):
        return {opponent: {identity: {"metrics": {metric: {"status": "available",
            "estimate": {"value": -0.03 if metric == "nll" else 0.},
            "uncertainty": {"status": "available", "interval95": [-0.04, -0.02] if metric == "nll" else [-0.01, 0.01]}}
            for metric in ("nll", "value_mse")}} for identity in (
                "validation_development_komi13", "validation_development_komi1",
                "population_development_komi13", "population_development_komi1")}
            for opponent in ("parent", "control")}

    def test_next_tier_requires_useful_supported_gain_against_parent_and_control(self):
        self.assertIs(evaluation.decide(self.comparisons())["continue_growth"], True)
        for opponent in ("parent", "control"):
            for problem in ("small_gain", "partial", "missing_interval", "missing_metric"):
                with self.subTest(opponent=opponent, problem=problem):
                    value = self.comparisons()
                    metric = value[opponent]["validation_development_komi13"]["metrics"]["nll"]
                    if problem == "small_gain":
                        metric["uncertainty"]["interval95"] = [-0.04, -0.019]
                    elif problem == "partial":
                        metric["uncertainty"]["status"] = "partial"
                    elif problem == "missing_interval":
                        metric["uncertainty"]["interval95"] = None
                    else:
                        metric["status"] = "unsupported"
                    self.assertIs(evaluation.decide(value)["continue_growth"], False)

    def test_clear_regression_in_any_fixed_scorecard_blocks_the_next_tier(self):
        for opponent in ("parent", "control"):
            for metric_name in ("nll", "value_mse"):
                with self.subTest(opponent=opponent, metric=metric_name):
                    value = self.comparisons()
                    value[opponent]["population_development_komi1"]["metrics"][metric_name]["uncertainty"]["interval95"] = [0.021, 0.04]
                    self.assertIs(evaluation.decide(value)["continue_growth"], False)


if __name__ == "__main__":
    unittest.main()
