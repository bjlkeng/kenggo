"""Synthetic CPU checks for epoch selection, validation isolation and recovery."""

import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from kenggo import continuation_training as worker
from kenggo.checkpoints import (file_hash, json_hash, load_checkpoint, read_json, restore_rng,
                                rng_state, save_checkpoint, save_json, seed_all)
from kenggo.encoding import INPUT_KEYS
from kenggo.examples import LABEL_SCHEMA
from kenggo.learner import ModelConfig, PopulationModel
from kenggo.supervised import TrainingData, train_step
from test_learner import model_inputs


SOURCE = {"commit": "a" * 40, "patch_sha256": "b" * 64}


def reference(path):
    return {"path": str(path), "sha256": file_hash(path)}


def labels(count):
    return {"action": np.zeros(count, dtype=np.int16),
            "policy_valid": np.ones(count, dtype=bool),
            "value": np.asarray([1 if index % 2 else -1 for index in range(count)], dtype=np.float32),
            "value_valid": np.ones(count, dtype=bool),
            "duration": np.zeros(count, dtype=np.float32),
            "duration_valid": np.zeros(count, dtype=bool)}


class StatefulModel(PopulationModel):
    def __init__(self):
        super().__init__(ModelConfig(blocks=1, channels=16))
        self.observation_norm = torch.nn.BatchNorm1d(16)
        self.observation_dropout = torch.nn.Dropout(0.25)

    def forward(self, inputs):
        value = self.observation_dropout(self.observation_norm(inputs["scalars"]))
        return super().forward({**inputs, "scalars": value})


class ContinuationTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        original = rng_state()
        self.addCleanup(restore_rng, original)
        seed_all(19)
        tensors = model_inputs()
        # Both fixture actions remain legal; duplicated observations have distinct source keys.
        inputs = {name: np.repeat(value[:1].numpy(), 8, axis=0) for name, value in tensors.items()}
        inputs["scalars"][:, 9:12] = [0.5, 0.25, 0.75]
        inputs["scalar_valid"][:, 9:12] = True
        targets = labels(8)
        self.extra = self.root / "inputs.json"
        save_json(self.extra, {"scope": "synthetic only"})
        pin = file_hash(self.extra)
        self.data = TrainingData(inputs, targets, tuple(f"train-{i}" for i in range(8)), pin, pin, pin)
        self.monitor = self.diagnostic("monitor", "train_monitor", "train", ("5k", "1d"), keys=("train-0", "train-1"))
        self.validation = self.diagnostic("validation", "epoch_validation", "validation_development", ("5k", "1d"))
        inputs = {name: reference(self.extra) for name in ("data", "split", "rules", "examples", "normalizer")}
        self.parent_config = {"version": "experiment_config_v1", "entity": "fixture", "project": "fixture",
            "base_url": "https://api.wandb.ai", "comparison": "continuation-fixture", "arm": "parent",
            "purpose": "baseline-training", "budget_id": "synthetic", "seed": 19,
            "parameters": {"model": {"blocks": 1, "channels": 16, "rank_conditioned": True}},
            "inputs": inputs}
        config_path = self.root / "parent-config.json"
        save_json(config_path, self.parent_config)
        provenance = {"version": "experiment_provenance_v1", "source": SOURCE, "inputs": inputs,
                      "environment": {"scope": "synthetic CPU"}}
        provenance_path = self.root / "parent-provenance.json"
        provenance_pin = save_json(provenance_path, provenance)
        model = PopulationModel(ModelConfig(**self.parent_config["parameters"]["model"]))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        batch = self.data.batch(0, 4, 19, full_batch=True)
        train_step(model, optimizer, *batch)
        checkpoint = save_checkpoint(self.root / "parent", model, optimizer, step=1,
            progress={"completed_updates": 1, "presented_examples": 4},
            config_sha256=json_hash(self.parent_config), provenance_sha256=provenance_pin)
        self.config = {**self.parent_config, "arm": "candidate", "purpose": "continuation-training",
            "parameters": {"recipe": worker.RECIPE, "stage_id": "tier-4m", "round_updates": 4,
                "model": self.parent_config["parameters"]["model"], "training_rows": 8, "device": "cpu",
                "batch_size": 4, "eval_batch_size": 2, "recovery_updates": 2,
                "deadline_utc": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
                "checkpoint_reserve_seconds": 5},
            "inputs": {**inputs, "parent_checkpoint": checkpoint, "parent_config": reference(config_path),
                "parent_provenance": reference(provenance_path), "train_monitor": self.monitor,
                "epoch_validation": self.validation}}
        patch = mock.patch.object(worker, "_source_identity", return_value=SOURCE)
        patch.start()
        self.addCleanup(patch.stop)
        loader = mock.patch.object(worker, "load_training", return_value=self.data)
        self.loader = loader.start()
        self.addCleanup(loader.stop)

    def diagnostic(self, name, purpose, role, ranks, *, keys=None):
        count = len(ranks)
        arrays = {name: value[:count].copy() for name, value in self.data.inputs.items()}
        arrays.update({name: value[:count].copy() for name, value in self.data.labels.items()})
        path = self.root / (name + ".npz")
        np.savez(path, **arrays)
        rows = [{"row_key": keys[index] if keys else name + "-" + str(index),
                 "source_url": name + "-game-" + str(index), "mover": "B", "role": role,
                 "group_mover": name + "-black-" + str(index), "group_opponent": name + "-white-" + str(index),
                 "source_rank_mover": rank, "source_rank_opponent": rank, "ply": 0,
                 "nominal_clock": [600, 30, 3]} for index, rank in enumerate(ranks)]
        manifest = {"version": worker.DIAGNOSTIC_VERSION, "purpose": purpose, "role": role,
                    "arrays": reference(path), "rows": rows, "aggregate_ranks": list(dict.fromkeys(ranks)),
                    "source_ids": [row["source_url"] for row in rows], "inputs_are_normalized": True,
                    "normalizer_sha256": self.data.normalizer_sha256}
        path = self.root / (name + ".json")
        save_json(path, manifest)
        return reference(path)

    def execute(self, name, config=None):
        config = copy.deepcopy(self.config if config is None else config)
        path = self.root / (name + "-config.json")
        config_pin = save_json(path, config)
        provenance = {"version": "experiment_provenance_v1", "source": SOURCE,
                      "inputs": config["inputs"], "environment": {"scope": "synthetic CPU"}}
        provenance_path = self.root / (name + "-provenance.json")
        provenance_pin = save_json(provenance_path, provenance)
        output = self.root / name
        output.mkdir()
        return worker.run(path, config_pin, provenance_path, provenance_pin, output, fixture_only=True)

    def continued(self, checkpoint, **parameters):
        config = copy.deepcopy(self.config)
        config["inputs"].update(parent_checkpoint={key: checkpoint[key] for key in ("path", "sha256")},
            parent_config=checkpoint["config"], parent_provenance=checkpoint["provenance"])
        config["parameters"].update(parameters)
        return config

    def restore(self, checkpoint):
        config = read_json(checkpoint["config"]["path"], checkpoint["config"]["sha256"])
        provenance = read_json(checkpoint["provenance"]["path"], checkpoint["provenance"]["sha256"])
        model = PopulationModel(ModelConfig(**config["parameters"]["model"]))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        cursor = load_checkpoint(checkpoint["path"], checkpoint["sha256"], model, optimizer,
            config_sha256=json_hash(config), provenance_sha256=json_hash(provenance))
        return model, optimizer, cursor

    def test_protected_role_is_rejected_before_any_arrays_are_decoded(self):
        for role in ("final", "population_confirmation", "validation_confirmation", "train"):
            value = read_json(self.validation["path"], self.validation["sha256"])
            value["role"] = role
            path = self.root / (role + ".json")
            save_json(path, value)
            with self.subTest(role=role), mock.patch.object(worker.np, "load") as decoder:
                with self.assertRaisesRegex(ValueError, "role"):
                    worker.load_diagnostic(reference(path), "epoch_validation", self.data.normalizer_sha256,
                                           fixture_only=True)
                decoder.assert_not_called()

    def test_validation_leaves_rng_modes_buffers_and_next_optimizer_update_unchanged(self):
        diagnostic = worker.load_diagnostic(self.validation, "epoch_validation", self.data.normalizer_sha256,
                                             fixture_only=True)
        model = StatefulModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model.observation_dropout.eval()
        inputs, targets = self.data.batch(0, 4, 19)
        saved = copy.deepcopy(model.state_dict())
        saved_optimizer = copy.deepcopy(optimizer.state_dict())
        modes = [module.training for module in model.modules()]
        random_state = rng_state()
        expected = train_step(model, optimizer, inputs, targets, clock_dropout=0.1)
        expected_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(saved)
        optimizer.load_state_dict(saved_optimizer)
        for module, mode in zip(model.modules(), modes):
            module.training = mode
        restore_rng(random_state)
        result = worker.evaluate_epoch(model, diagnostic, batch_size=2)
        self.assertEqual(result["rows"], 2)
        self.assertEqual([module.training for module in model.modules()], modes)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, saved[name], rtol=0, atol=0)
        self.assertEqual(train_step(model, optimizer, inputs, targets, clock_dropout=0.1), expected)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, expected_state[name], rtol=0, atol=0)

    def test_validation_exception_also_restores_modes_and_rng(self):
        diagnostic = worker.load_diagnostic(self.validation, "epoch_validation", self.data.normalizer_sha256,
                                             fixture_only=True)
        model = StatefulModel()
        model.observation_dropout.eval()
        modes = [module.training for module in model.modules()]
        state = rng_state()
        expected = torch.rand(3)
        restore_rng(state)
        def fail(*args):
            torch.rand(5)
            raise RuntimeError("synthetic validation failure")
        with mock.patch.object(model, "forward", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "validation failure"):
                worker.evaluate_epoch(model, diagnostic)
        self.assertEqual([module.training for module in model.modules()], modes)
        torch.testing.assert_close(torch.rand(3), expected, rtol=0, atol=0)

    def test_small_gains_select_raw_best_but_accumulate_for_patience_reset(self):
        state = {"best_score": None, "tolerance_reference": None, "bad_rounds": 0}
        results = []
        for value in (2.0, 1.999, 1.998, 1.9975, 1.9985, 1.999):
            state, best, meaningful = worker.observe_validation(state, value)
            results.append((state["bad_rounds"], best, meaningful))
        self.assertEqual(results, [(0, True, True), (1, True, False), (0, True, True),
                                   (1, True, False), (2, False, False), (3, False, False)])
        self.assertEqual(state["best_score"], 1.9975)
        self.assertEqual(state["tolerance_reference"], 1.998)

    def test_epoch_zero_is_eligible_and_later_round_restores_stage_cursor(self):
        config = copy.deepcopy(self.config)
        config["parameters"]["evaluate_only"] = True
        initial = self.execute("initial", config)
        self.assertEqual(initial["completed_updates"], 0)
        self.assertEqual(initial["progress"]["completed_rounds"], 0)
        self.assertEqual(initial["best_checkpoint"], initial["latest_checkpoint"])
        self.assertEqual(initial["best_checkpoint"]["step"], 1)
        self.assertEqual(len(initial["commits"]), 1)
        result = self.execute("round", self.continued(initial["latest_checkpoint"]))
        self.assertEqual(result["completed_updates"], 4)
        self.assertEqual(result["progress"]["sampler_step"], 4)
        self.assertEqual(result["progress"]["lifetime_updates"], 5)
        self.assertEqual(result["progress"]["lifetime_examples"], 20)
        self.assertEqual(result["progress"]["completed_rounds"], 1)
        self.assertEqual(len(result["progress"]["epoch_metrics"]), 1)
        self.assertEqual(result["progress"]["partial_round"]["updates"], 0)
        record = read_json(result["commits"][0]["path"], result["commits"][0]["sha256"])
        self.assertEqual(record["checkpoint"], result["latest_checkpoint"])
        self.assertEqual(record["best_checkpoint"], result["best_checkpoint"])
        with mock.patch.object(worker, "train_step") as train:
            with self.assertRaisesRegex(ValueError, "Epoch-zero"):
                self.execute("invalid-epoch-zero", self.continued(result["latest_checkpoint"], evaluate_only=True))
            train.assert_not_called()

    def test_budget_stop_resumes_exact_optimizer_rng_and_partial_round_metrics(self):
        initial_config = copy.deepcopy(self.config)
        initial_config["parameters"]["evaluate_only"] = True
        initial = self.execute("epoch-zero", initial_config)
        config = self.continued(initial["latest_checkpoint"])
        uninterrupted = self.execute("uninterrupted", config)
        calls = []
        def observe(*args, **kwargs):
            result = train_step(*args, **kwargs)
            calls.append(result)
            return result
        def budget(_):
            if len(calls) >= 2:
                raise worker.BudgetStop("synthetic soft stop")
        with mock.patch.object(worker, "train_step", side_effect=observe), \
                mock.patch.object(worker._Budget, "check", budget):
            stopped = self.execute("stopped", config)
        self.assertEqual(stopped["status"], "partial")
        self.assertEqual(stopped["completed_updates"], 2)
        self.assertEqual(stopped["best_checkpoint"], initial["best_checkpoint"])
        self.assertEqual(stopped["progress"]["stopping"], initial["progress"]["stopping"])
        self.assertEqual(stopped["progress"]["partial_round"]["updates"], 2)
        resumed = self.execute("resumed", self.continued(stopped["latest_checkpoint"]))
        self.assertEqual(resumed["completed_updates"], 2)
        self.assertEqual(resumed["progress"]["epoch_metrics"], uninterrupted["progress"]["epoch_metrics"])
        self.assertEqual(resumed["progress"]["stopping"], uninterrupted["progress"]["stopping"])
        expected, expected_optimizer, _ = self.restore(uninterrupted["latest_checkpoint"])
        actual, actual_optimizer, _ = self.restore(resumed["latest_checkpoint"])
        for name, value in actual.state_dict().items():
            torch.testing.assert_close(value, expected.state_dict()[name], rtol=0, atol=0)
        batch = self.data.batch(4, 4, 19, full_batch=True)
        expected_loss = train_step(expected, expected_optimizer, *batch)
        self.restore(resumed["latest_checkpoint"])
        actual_loss = train_step(actual, actual_optimizer, *batch)
        self.assertEqual(actual_loss, expected_loss)

    def test_changed_stage_restarts_sampler_and_stopping_but_keeps_lifetime_cost_units(self):
        previous = self.execute("previous")
        config = self.continued(previous["latest_checkpoint"], stage_id="tier-8m", evaluate_only=True)
        result = self.execute("new-tier", config)
        self.assertEqual(result["progress"]["sampler_step"], 0)
        self.assertEqual(result["progress"]["completed_rounds"], 0)
        self.assertEqual(result["progress"]["lifetime_updates"], previous["progress"]["lifetime_updates"])
        self.assertEqual(result["progress"]["stopping"]["bad_rounds"], 0)

    def test_same_stage_rejects_changed_schedule_and_validation_before_training(self):
        previous = self.execute("original")
        config = self.continued(previous["latest_checkpoint"], round_updates=5)
        with mock.patch.object(worker, "train_step") as train:
            with self.assertRaisesRegex(ValueError, "resumed stage"):
                self.execute("changed", config)
            train.assert_not_called()
        failure = json.loads((self.root / "changed/training_failure.json").read_text())
        self.assertEqual(failure["completed_updates"], 0)

    def test_completed_target_recovers_interrupted_commit_without_an_extra_training_round(self):
        config = copy.deepcopy(self.config)
        config["parameters"]["target_round"] = 1
        original = worker.save_json
        def interrupted(path, *args, **kwargs):
            path = Path(path)
            if path.parent.name == "commits" and path.name == "epoch-0001.json":
                raise OSError("synthetic interrupted epoch commit")
            return original(path, *args, **kwargs)
        with mock.patch.object(worker, "save_json", side_effect=interrupted):
            with self.assertRaisesRegex(OSError, "epoch commit"):
                self.execute("commit-failure", config)
        failed = json.loads((self.root / "commit-failure/training_failure.json").read_text())
        self.assertEqual(failed["progress"]["completed_rounds"], 1)
        self.assertEqual(failed["completed_updates"], 4)
        self.assertFalse((self.root / "commit-failure/commits/epoch-0001.json").exists())
        resumed_config = self.continued(failed["latest_checkpoint"], target_round=1)
        with mock.patch.object(worker, "train_step") as train, \
                mock.patch.object(worker, "evaluate_epoch") as evaluate:
            result = self.execute("commit-recovery", resumed_config)
            train.assert_not_called()
            evaluate.assert_not_called()
        self.assertEqual(result["completed_updates"], 0)
        self.assertEqual(result["reason"], "recovered_matched_round")
        self.assertEqual(result["progress"]["sampler_step"], 4)
        self.assertEqual(result["progress"]["stopping"], failed["progress"]["stopping"])
        record = read_json(result["commits"][0]["path"], result["commits"][0]["sha256"])
        self.assertTrue(record["recovered_completed_round"])
        self.assertEqual(record["metrics"], failed["progress"]["last_epoch_metrics"])
        self.assertTrue(Path(record["metrics"]["path"]).is_relative_to(self.root / "commit-failure"))

    def test_failed_checkpoint_does_not_advance_patience_or_erase_partial_round_metrics(self):
        config = copy.deepcopy(self.config)
        config["parameters"]["target_round"] = 1
        original = worker.save_checkpoint
        def interrupted(*args, **kwargs):
            if kwargs["progress"]["completed_rounds"] == 1:
                raise OSError("synthetic checkpoint write failure")
            return original(*args, **kwargs)
        with mock.patch.object(worker, "save_checkpoint", side_effect=interrupted):
            with self.assertRaisesRegex(OSError, "checkpoint write"):
                self.execute("checkpoint-failure", config)
        failed = json.loads((self.root / "checkpoint-failure/training_failure.json").read_text())
        self.assertEqual(failed["progress"]["completed_rounds"], 0)
        self.assertEqual(failed["progress"]["partial_round"]["updates"], 4)
        self.assertEqual(failed["progress"]["stopping"]["bad_rounds"], 0)
        self.assertEqual(failed["latest_checkpoint"]["kind"], "before_validation")
        with mock.patch.object(worker, "train_step") as train:
            result = self.execute("checkpoint-recovery", self.continued(failed["latest_checkpoint"], target_round=1))
            train.assert_not_called()
        self.assertEqual(result["progress"]["completed_rounds"], 1)
        self.assertEqual(result["epoch_metrics"][0]["updates"], 4)
        self.assertEqual(result["completed_updates"], 0)

    def test_assessment_preserves_all_rows_and_explicitly_reports_excluded_ranks(self):
        ref = self.diagnostic("assessment", "assessment", "population_development", ("15k", "5k", "1d"))
        value = read_json(ref["path"], ref["sha256"])
        value["aggregate_ranks"] = ["5k", "1d"]
        save_json(ref["path"], value, replace=True)
        ref = reference(ref["path"])
        model = PopulationModel(ModelConfig(blocks=1, channels=16))
        result = worker.evaluate_diagnostic(model, ref, normalizer_sha256=self.data.normalizer_sha256,
            purpose="assessment", fixture_only=True, include_predictions=True)
        self.assertEqual((result["evaluated_rows"], result["rows"], result["excluded_rank_rows"]), (3, 2, 1))
        self.assertEqual(len(result["predictions"]), 3)
        self.assertEqual(result["report"]["aggregate_ranks"], ["5k", "1d"])
        self.assertEqual(result["all_rank_report"]["aggregate_ranks"], list(worker.RANKS))
        self.assertEqual(result["all_rank_report"]["overall"]["metrics"]["nll"]["status"], "unavailable")
        self.assertTrue(all("legal" not in row for row in result["predictions"]))
        self.assertTrue(all(np.isfinite(row["nll"]) for row in result["predictions"]))


if __name__ == "__main__":
    unittest.main()
