"""Small synthetic CPU worker checks; no production data or GPU allocation."""

import copy
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from kenggo.baseline_training import RECIPE, run_training, tensor_content_hash
from kenggo.checkpoints import (file_hash, json_hash, load_checkpoint, read_json, restore_rng,
                                rng_state, save_json)
from kenggo.examples import ExampleConfig, build_examples, fit_normalizer, read_shards
from kenggo.learner import ModelConfig, PopulationModel
from kenggo.supervised import TrainingData, fit, train_step
from encoding_fixtures import corpus, record


SOURCE = {"commit": "a" * 40, "patch_sha256": sha256(b"synthetic source").hexdigest()}


class BaselineTrainingTests(unittest.TestCase):
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
        source = mock.patch("kenggo.baseline_training._source_identity", return_value=SOURCE)
        source.start()
        self.addCleanup(source.stop)
        records = [record("a"), record("b", result="W+T"), record("c", result="W+2.5"),
                   record("development", day="2025-10-01"), record("final", day="2025-12-01")]
        args = corpus(self.root / "source", records)
        self.examples = self.root / "examples"
        examples_pin = build_examples(*args, self.examples, config=ExampleConfig(shard_size=4), fixture_only=True)
        normalizer = self.root / "normalizer.json"
        normalizer_pin = save_json(normalizer, fit_normalizer(self.examples, examples_pin))
        extra = self.root / "pinned-fixture.json"
        extra_pin = save_json(extra, {"purpose": "synthetic tests only"})
        inputs = {name: {"path": str(extra), "sha256": extra_pin}
                  for name in ("rules", "cpu_environment", "cuda_environment")}
        inputs.update(data={"path": str(args[0]), "sha256": args[1]},
                      split={"path": str(args[2] / "manifest.json"), "sha256": args[3]},
                      examples={"path": str(self.examples / "manifest.json"), "sha256": examples_pin},
                      normalizer={"path": str(normalizer), "sha256": normalizer_pin})
        self.config = {"version": "experiment_config_v1", "entity": "fixture", "project": "fixture",
            "base_url": "https://api.wandb.ai", "comparison": "baseline-fixture", "arm": "conditioned",
            "purpose": "baseline-training", "budget_id": "synthetic-cpu", "seed": 20261114,
            "parameters": {**RECIPE, "epochs": 2, "training_rows": 9, "batch_size": 4,
                           "save_epochs": [0, 1, 2], "device": "cpu",
                           "model": {"blocks": 1, "channels": 16, "rank_conditioned": True}},
            "inputs": inputs}
        self.provenance = {"version": "experiment_provenance_v1", "source": SOURCE,
                           "inputs": inputs, "environment": {"scope": "synthetic CPU fixture"}}
        self.config_path, self.provenance_path = self.root / "config.json", self.root / "provenance.json"
        self.write_contract()
        self.data = TrainingData.load(self.examples, examples_pin, normalizer, normalizer_pin,
                                      split_sha256=args[3])

    def write_contract(self):
        self.config_path.write_text(json.dumps(self.config, indent=2) + "\n")
        self.config_pin = file_hash(self.config_path)
        self.provenance_pin = save_json(self.provenance_path, self.provenance, replace=True)

    def run_worker(self, name="job", **options):
        output = self.root / name
        output.mkdir()
        result = run_training(self.config_path, self.config_pin, self.provenance_path,
                              self.provenance_pin, output, fixture_only=True, **options)
        return output, result

    def restore(self, output, ref):
        model = PopulationModel(ModelConfig(**self.config["parameters"]["model"]))
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config["parameters"]["learning_rate"])
        restored = load_checkpoint(output / ref["path"], ref["sha256"], model, optimizer,
            config_sha256=json_hash(self.config), provenance_sha256=self.provenance_pin)
        return model, optimizer, restored

    def test_full_epochs_preserve_short_batches_masks_order_and_weighted_training_losses(self):
        calls = []
        def observed_step(model, optimizer, inputs, labels, **options):
            logits, value = model(inputs)
            self.assertEqual(logits.shape, (labels["action"].shape[0], 362))
            self.assertEqual(value.shape, labels["action"].shape)
            result = train_step(model, optimizer, inputs, labels, **options)
            calls.append((copy.deepcopy(inputs), copy.deepcopy(labels), result))
            return result
        with mock.patch("kenggo.baseline_training.train_step", side_effect=observed_step):
            output, result = self.run_worker()
        self.assertEqual([metrics["examples"] for _, _, metrics in calls], [4, 4, 1, 4, 4, 1])
        self.assertEqual((result["completed_updates"], result["attempted_updates"], result["next_step"]), (6, 6, 6))
        self.assertEqual((result["presented_examples"], result["attempted_examples"]), (18, 18))
        self.assertEqual((result["uncertain_updates"], result["uncertain_examples"]), (0, 0))
        self.assertEqual(result["row_keys_sha256"], json_hash(list(self.data.row_keys)))
        self.assertEqual(result["config_sha256"], json_hash(self.config))
        self.assertNotEqual(result["config_sha256"], self.config_pin)
        self.assertEqual(result["config_file_sha256"], self.config_pin)
        self.assertEqual(result["provenance_sha256"], self.provenance_pin)
        self.assertEqual(result["checkpoint_selection"], "fixed_final_epoch")
        for step, (inputs, labels, _) in enumerate(calls):
            indices = self.data.indices(step, 4, self.config["seed"])
            for key in ("ranks", "rank_valid"):
                np.testing.assert_array_equal(inputs[key].numpy(), self.data.inputs[key][indices])
            for key in labels:
                np.testing.assert_array_equal(labels[key].numpy(), self.data.labels[key][indices])
        for epoch, metrics in enumerate(result["epoch_metrics"]):
            batches = [call[2] for call in calls[epoch * 3:(epoch + 1) * 3]]
            self.assertEqual((metrics["updates"], metrics["presented_examples"], metrics["policy_count"],
                              metrics["value_count"]), (3, 9, 9, 6))
            for loss, count in (("policy_loss", "policy_count"), ("value_loss", "value_count")):
                expected = sum(row[loss] * row[count] for row in batches) / sum(row[count] for row in batches)
                self.assertEqual(metrics[loss], expected)
        self.assertEqual([ref["epoch"] for ref in result["checkpoints"]], [0, 1, 2])
        self.assertEqual([ref["step"] for ref in result["checkpoints"]], [0, 3, 6])
        self.assertEqual(result["final_checkpoint"], result["checkpoints"][-1])
        for ref in result["checkpoints"]:
            self.assertTrue((output / ref["path"]).resolve().is_relative_to(output))
            self.assertEqual(read_json(output / ref["path"], ref["sha256"])["config_sha256"], json_hash(self.config))
        persisted = json.loads((output / "training_progress.json").read_text())
        self.assertEqual(persisted["completed_updates"], 6)
        self.assertEqual(json.loads((output / "training_epochs.json").read_text())["epochs"], result["epoch_metrics"])
        self.assertEqual(json.loads((output / "training_summary.json").read_text()), result)
        self.assertTrue(all(value >= 0 for value in result["phase_seconds"].values()))
        self.assertEqual(result["peak_cuda_allocated_bytes"], 0)
        self.assertEqual(result["peak_cuda_reserved_bytes"], 0)
        self.assertTrue(result["environment"]["deterministic_algorithms"])
        self.assertFalse(result["environment"]["cudnn_benchmark"])
        self.assertFalse(result["environment"]["matmul_allow_tf32"])

    def test_identical_seed_reproduces_metrics_and_final_tensors_and_checkpoints_resume_exactly(self):
        first_output, first = self.run_worker("first")
        second_output, second = self.run_worker("second")
        self.assertEqual(first["initial_tensor_sha256"], second["initial_tensor_sha256"])
        self.assertEqual(first["epoch_metrics"], second["epoch_metrics"])
        initial, _, initial_cursor = self.restore(first_output, first["checkpoints"][0])
        self.assertEqual(tensor_content_hash(initial), first["initial_tensor_sha256"])
        self.assertEqual(initial_cursor["progress"]["next_step"], 0)
        expected, _, final_cursor = self.restore(first_output, first["final_checkpoint"])
        again, _, _ = self.restore(second_output, second["final_checkpoint"])
        self.assertEqual(tensor_content_hash(expected), tensor_content_hash(again))
        self.assertEqual(final_cursor["progress"]["presented_examples"], 18)
        resumed, optimizer, cursor = self.restore(first_output, first["checkpoints"][1])
        self.assertEqual((cursor["step"], cursor["progress"]["next_step"], cursor["progress"]["epoch"]), (3, 3, 1))
        self.assertEqual(cursor["progress"]["batch_in_epoch"], 0)
        self.assertEqual(fit(resumed, optimizer, self.data, start_step=cursor["step"], steps=3,
            batch_size=4, seed=self.config["seed"], clock_dropout=0.1), 6)
        for name, value in expected.state_dict().items():
            torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)

    def test_conditioning_changes_neither_initial_tensors_nor_training_order(self):
        first_output, first = self.run_worker("conditioned")
        first_initial, _, _ = self.restore(first_output, first["checkpoints"][0])
        self.config["arm"] = "unconditioned"
        self.config["parameters"]["model"]["rank_conditioned"] = False
        self.write_contract()
        second_output, second = self.run_worker("unconditioned")
        second_initial, _, _ = self.restore(second_output, second["checkpoints"][0])
        self.assertEqual(first["initial_tensor_sha256"], second["initial_tensor_sha256"])
        self.assertEqual(first["row_keys_sha256"], second["row_keys_sha256"])
        self.assertEqual([row["order_sha256"] for row in first["epoch_metrics"]],
                         [row["order_sha256"] for row in second["epoch_metrics"]])
        for name, tensor in first_initial.state_dict().items():
            torch.testing.assert_close(tensor, second_initial.state_dict()[name], rtol=0, atol=0)

    def test_only_training_shards_are_opened(self):
        for role in ("population_development", "final", "test_prefix"):
            for path in (self.examples / role).glob("*.npz"):
                path.write_bytes(b"sealed or held-out payload must not be opened")
        with mock.patch("kenggo.supervised.read_shards", wraps=read_shards) as reader:
            _, result = self.run_worker()
        reader.assert_called_once_with(self.examples, self.config["inputs"]["examples"]["sha256"], "train")
        self.assertEqual(result["training_rows"], 9)

    def test_started_update_is_durable_and_failures_retain_uncertainty_without_retry(self):
        output = self.root / "job"
        output.mkdir()
        calls = []
        def fail_on_second(model, optimizer, inputs, labels, **options):
            progress = json.loads((output / "training_progress.json").read_text())
            self.assertEqual(progress["attempted_updates"], len(calls) + 1)
            self.assertEqual(progress["completed_updates"], len(calls))
            self.assertEqual(progress["uncertain_updates"], 1)
            if calls:
                raise RuntimeError("synthetic interrupted update")
            calls.append(train_step(model, optimizer, inputs, labels, **options))
            return calls[-1]
        with mock.patch("kenggo.baseline_training.train_step", side_effect=fail_on_second):
            with self.assertRaisesRegex(RuntimeError, "synthetic interrupted"):
                run_training(self.config_path, self.config_pin, self.provenance_path,
                             self.provenance_pin, output, fixture_only=True)
        failure = json.loads((output / "training_failure.json").read_text())
        self.assertEqual((failure["completed_updates"], failure["attempted_updates"], failure["next_step"]), (1, 2, 1))
        self.assertEqual((failure["presented_examples"], failure["attempted_examples"]), (4, 8))
        self.assertEqual((failure["uncertain_updates"], failure["uncertain_examples"]), (1, 4))
        self.assertEqual(failure["last_loss"], calls[0])
        self.assertEqual(len(failure["checkpoints"]), 1)
        self.assertFalse((output / "training_summary.json").exists())
        pin = file_hash(output / "training_progress.json")
        with self.assertRaisesRegex(ValueError, "never resets"):
            run_training(self.config_path, self.config_pin, self.provenance_path,
                         self.provenance_pin, output, fixture_only=True)
        self.assertEqual(pin, file_hash(output / "training_progress.json"))

    def test_contract_rejects_wrong_pins_purpose_source_inputs_counts_and_checkpoint_parameters(self):
        valid_config, valid_provenance = copy.deepcopy(self.config), copy.deepcopy(self.provenance)
        for mutation in ("config_pin", "provenance_pin", "purpose", "source", "inputs", "count", "resume"):
            with self.subTest(mutation=mutation):
                self.config, self.provenance = copy.deepcopy(valid_config), copy.deepcopy(valid_provenance)
                if mutation == "purpose":
                    self.config["purpose"] = "throughput-probe"
                elif mutation == "source":
                    self.provenance["source"]["patch_sha256"] = "b" * 64
                elif mutation == "inputs":
                    self.provenance["inputs"]["normalizer"]["sha256"] = "b" * 64
                elif mutation == "count":
                    self.config["parameters"]["training_rows"] = 8
                elif mutation == "resume":
                    self.config["parameters"]["initial_checkpoint"] = "forbidden"
                self.write_contract()
                if mutation == "config_pin":
                    self.config_pin = "b" * 64
                elif mutation == "provenance_pin":
                    self.provenance_pin = "b" * 64
                output = self.root / ("rejected-" + mutation)
                output.mkdir()
                with mock.patch("kenggo.baseline_training.TrainingData.load") as loader:
                    with self.assertRaises(ValueError):
                        run_training(self.config_path, self.config_pin, self.provenance_path,
                                     self.provenance_pin, output, fixture_only=True)
                    loader.assert_not_called()
                failure = json.loads((output / "training_failure.json").read_text())
                self.assertEqual(failure["completed_updates"], 0)

    def test_cuda_unavailability_is_a_failure_before_loading_without_cpu_fallback(self):
        self.config["parameters"] = {**RECIPE, "model": {"blocks": 8, "channels": 128, "rank_conditioned": True}}
        manifest_path = self.examples / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["fixture_only"] = False
        manifest["counts"]["train"]["examples"] = RECIPE["training_rows"]
        self.config["inputs"]["examples"]["sha256"] = save_json(manifest_path, manifest, replace=True)
        self.write_contract()
        output = self.root / "cuda-unavailable"
        output.mkdir()
        with mock.patch.dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8"), \
                mock.patch("torch.cuda.is_available", return_value=False), \
                mock.patch("kenggo.baseline_training.TrainingData.load") as loader:
            with self.assertRaisesRegex(RuntimeError, "no CPU fallback"):
                run_training(self.config_path, self.config_pin, self.provenance_path, self.provenance_pin, output)
        loader.assert_not_called()
        self.assertEqual(json.loads((output / "training_failure.json").read_text())["completed_updates"], 0)

    def test_output_requires_an_existing_unused_owned_directory(self):
        with self.assertRaisesRegex(ValueError, "supervisor must create"):
            run_training(self.config_path, self.config_pin, self.provenance_path,
                         self.provenance_pin, self.root / "missing", fixture_only=True)
        output = self.root / "unsafe"
        output.mkdir()
        outside = self.root / "outside"
        outside.mkdir()
        (output / "checkpoints").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "never resets"):
            run_training(self.config_path, self.config_pin, self.provenance_path,
                         self.provenance_pin, output, fixture_only=True)
        self.assertEqual(list(outside.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
