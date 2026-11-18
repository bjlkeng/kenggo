"""Synthetic CPU checks for matched expanded training and its explicit data contract."""

import copy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from kenggo.baseline_training import (EXPANDED_RECIPE, INFERENCE_CONTRACT, _expanded_updates,
                                      run_training, tensor_content_hash)
from kenggo.checkpoints import (json_hash, load_checkpoint, restore_rng, rng_state, save_json)
from kenggo.examples import ExampleConfig, build_examples, fit_normalizer, read_shards
from kenggo.learner import ModelConfig, PopulationModel
from kenggo.supervised import TrainingData, fit, train_step
from encoding_fixtures import corpus, record
from test_komi_data import production_profile, record as komi_record


SOURCE = {"commit": "a" * 40, "patch_sha256": sha256(b"expanded synthetic source").hexdigest()}


class ExpandedTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.addCleanup(restore_rng, rng_state())
        source = mock.patch("kenggo.baseline_training._source_identity", return_value=SOURCE)
        source.start()
        self.addCleanup(source.stop)

    def population(self, name, *, mixed=False):
        root = self.root / name
        root.mkdir()
        rows = [record("a"), record("b", result="W+T"), record("c", result="W+2.5")]
        if mixed:
            rows += [komi_record("low-a", moves=";B[aa];W[bb];B[]"),
                     komi_record("low-b", moves=";B[cc];W[dd];B[]"),
                     komi_record("low-c", moves=";B[ee];W[ff];B[]")]
        rows += [record("development", day="2025-10-01"), record("final", day="2025-12-01")]
        args = corpus(root / "source", rows)
        examples = root / "examples"
        pin = build_examples(*args, examples, config=ExampleConfig(shard_size=4),
                             fixture_only=True, komi_study=mixed)
        normalizer = root / "normalizer.json"
        normalizer_pin = save_json(normalizer, fit_normalizer(examples, pin, komi_study=mixed))
        load_args = (examples, pin, normalizer, normalizer_pin)
        options = {"split_sha256": args[3], "komi_study": mixed}
        data = TrainingData.load(*load_args, **options)
        extra = root / "inputs.json"
        extra_pin = save_json(extra, {"scope": "synthetic tests only"})
        inputs = {key: {"path": str(extra), "sha256": extra_pin}
                  for key in ("rules", "cpu_environment", "cuda_environment")}
        inputs.update(data={"path": str(args[0]), "sha256": args[1]},
            split={"path": str(args[2] / "manifest.json"), "sha256": args[3]},
            examples={"path": str(examples / "manifest.json"), "sha256": pin},
            normalizer={"path": str(normalizer), "sha256": normalizer_pin})
        config = {"version": "experiment_config_v1", "entity": "fixture", "project": "fixture",
            "base_url": "https://api.wandb.ai", "comparison": "expanded-fixture", "arm": "conditioned",
            "purpose": "baseline-training", "budget_id": "synthetic-cpu", "seed": 20261118,
            "parameters": {"recipe": EXPANDED_RECIPE, "inference_contract": INFERENCE_CONTRACT,
                "training_komi_halfpoints": [1, 13] if mixed else [13], "updates": 8,
                "batch_size": 4, "learning_rate": 0.001, "clock_dropout": 0.1,
                "training_rows": len(data), "save_updates": [0, 3, 8], "device": "cpu", "torch_threads": 1,
                "model": {"blocks": 1, "channels": 16, "rank_conditioned": True}}, "inputs": inputs}
        return root, args, load_args, options, data, config

    def run_worker(self, root, config, name="worker"):
        config_path = root / (name + "-config.json")
        config_pin = save_json(config_path, config)
        provenance = {"version": "experiment_provenance_v1", "source": SOURCE,
                      "inputs": config["inputs"], "environment": {"scope": "synthetic CPU"}}
        provenance_path = root / (name + "-provenance.json")
        provenance_pin = save_json(provenance_path, provenance)
        output = root / name
        output.mkdir()
        result = run_training(config_path, config_pin, provenance_path, provenance_pin,
                              output, fixture_only=True)
        return output, result

    def restore(self, output, config, summary, ref):
        model = PopulationModel(ModelConfig(**config["parameters"]["model"]))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        cursor = load_checkpoint(output / ref["path"], ref["sha256"], model, optimizer,
            config_sha256=json_hash(config), provenance_sha256=summary["provenance_sha256"])
        return model, optimizer, cursor

    def test_cached_legacy_order_is_exact_and_shuffles_only_once_per_epoch(self):
        _, _, _, _, data, _ = self.population("base")
        expected = [np.random.default_rng(np.random.SeedSequence([19, step // 3])).permutation(9)
                    [(step % 3) * 4:(step % 3 + 1) * 4] for step in range(9)]
        with mock.patch("kenggo.supervised.np.random.default_rng", wraps=np.random.default_rng) as generator:
            for step, indices in enumerate(expected):
                np.testing.assert_array_equal(data.indices(step, 4, 19), indices)
                data.batch(step, 4, 19)
            self.assertEqual(generator.call_count, 3)
        self.assertLessEqual(len(data._permutations), 2)

    def test_full_batches_cross_epoch_boundaries_and_resume_with_an_empty_cache(self):
        _, _, _, _, data, _ = self.population("base")
        expected = np.concatenate([np.random.default_rng(np.random.SeedSequence([19, epoch])).permutation(9)
                                   for epoch in range(4)])
        for step in range(9):
            indices = data.indices(step, 4, 19, full_batch=True)
            self.assertEqual(len(indices), 4)
            np.testing.assert_array_equal(indices, expected[step * 4:(step + 1) * 4])
            self.assertFalse(indices.flags.writeable)
        fresh = replace(data)
        self.assertEqual(fresh._permutations, {})
        for step in range(5, 9):
            np.testing.assert_array_equal(fresh.indices(step, 4, 19, full_batch=True),
                                          expected[step * 4:(step + 1) * 4])
        with self.assertRaises(ValueError):
            data.indices(0, 4, 19, full_batch=1)

    def test_mixed_loader_requires_opt_in_and_never_opens_other_roles(self):
        _, _, args, options, expected, _ = self.population("mixed", mixed=True)
        with self.assertRaisesRegex(ValueError, "Unsupported example contract"):
            TrainingData.load(*args, split_sha256=options["split_sha256"])
        for role in ("population_development", "final"):
            for path in (args[0] / role).glob("*.npz"):
                path.write_bytes(b"forbidden evaluation arrays")
        with mock.patch("kenggo.supervised.read_shards", wraps=read_shards) as reader:
            data = TrainingData.load(*args, **options)
        reader.assert_called_once_with(args[0], args[1], "train", komi_study=True)
        self.assertEqual(data.row_keys, expected.row_keys)
        self.assertEqual(set(abs(data.inputs["scalars"][:, 1])), {0.5, 6.5})
        self.assertTrue(np.array_equal(data.inputs["scalars"][:, 1] < 0, data.inputs["scalars"][:, 0] == 1))

    def test_production_profile_is_required_for_study_and_must_bind_the_exact_examples(self):
        root, source, args, _, _, _ = self.population("mixed", mixed=True)
        source, profile = production_profile(root, source)
        manifest = json.loads((args[0] / "manifest.json").read_text())
        manifest.update(fixture_only=False, split_sha256=source[3], dataset_profile_sha256=profile.sha256)
        pin = save_json(args[0] / "manifest.json", manifest, replace=True)
        normalizer = json.loads(args[2].read_text())
        normalizer.update(fixture_only=False, examples_sha256=pin)
        normalizer_pin = save_json(args[2], normalizer, replace=True)
        load_args = (args[0], pin, args[2], normalizer_pin)
        data = TrainingData.load(*load_args, split_sha256=source[3], profile=profile)
        self.assertEqual(len(data), 18)
        for options in ({"komi_study": True}, {"profile": replace(profile, sha256="b" * 64)},
                        {"profile": replace(profile, canonical_sha256="b" * 64)},
                        {"profile": replace(profile, split_sha256="b" * 64)}):
            with self.subTest(options=options), mock.patch("kenggo.supervised.read_shards") as reader:
                with self.assertRaises(ValueError):
                    TrainingData.load(*load_args, split_sha256=source[3], **options)
                reader.assert_not_called()

    def test_four_arms_match_initialization_update_counts_and_presentations(self):
        summaries = []
        for mixed in (False, True):
            root, _, _, _, _, original = self.population("mixed" if mixed else "base", mixed=mixed)
            pair = []
            for conditioned in (True, False):
                config = copy.deepcopy(original)
                config["arm"] = "conditioned" if conditioned else "unconditioned"
                config["parameters"]["model"]["rank_conditioned"] = conditioned
                output, summary = self.run_worker(root, config, config["arm"])
                summaries.append(summary)
                pair.append(summary)
                self.assertEqual(summary["version"], EXPANDED_RECIPE)
                self.assertEqual(summary["checkpoint_selection"], "fixed_final_update")
                self.assertEqual((summary["planned_updates"], summary["completed_updates"]), (8, 8))
                self.assertEqual((summary["planned_examples"], summary["presented_examples"]), (32, 32))
                self.assertEqual((summary["data_epoch"], summary["row_in_epoch"]), divmod(32, 18 if mixed else 9))
                self.assertEqual([ref["step"] for ref in summary["checkpoints"]], [0, 3, 8])
                self.assertNotIn("epoch_metrics", summary)
                self.assertFalse((output / "training_epochs.json").exists())
                self.assertEqual([row["updates"] for row in summary["update_metrics"]], [6, 2])
                self.assertEqual([row["phase"] for row in summary["update_metrics"]], ["warmup", "measured"])
                self.assertEqual(sum(row["presented_examples"] for row in summary["update_metrics"]), 32)
                self.assertEqual(json.loads((output / "training_summary.json").read_text()), summary)
            self.assertEqual(pair[0]["row_keys_sha256"], pair[1]["row_keys_sha256"])
            self.assertEqual([row["order_sha256"] for row in pair[0]["update_metrics"]],
                             [row["order_sha256"] for row in pair[1]["update_metrics"]])
        self.assertEqual(len({summary["initial_tensor_sha256"] for summary in summaries}), 1)

    def test_checkpoint_continuation_reproduces_exact_tensors_and_losses_across_epoch_tails(self):
        root, _, args, options, _, config = self.population("base")
        losses = []
        def observed(*args, **kwargs):
            result = train_step(*args, **kwargs)
            losses.append(result)
            return result
        with mock.patch("kenggo.baseline_training.train_step", side_effect=observed):
            output, summary = self.run_worker(root, config)
        final, _, _ = self.restore(output, config, summary, summary["final_checkpoint"])
        model, optimizer, cursor = self.restore(output, config, summary, summary["checkpoints"][1])
        self.assertEqual(cursor["progress"]["sampler"], "continuous_shuffled_epochs_v1")
        self.assertEqual((cursor["step"], cursor["progress"]["row_in_epoch"]), (3, 3))
        data = TrainingData.load(*args, **options)
        resumed = []
        fit(model, optimizer, data, start_step=cursor["step"], steps=5, batch_size=4,
            seed=config["seed"], full_batch=True, on_step=lambda step, metrics: resumed.append(metrics))
        self.assertEqual(resumed, losses[3:])
        self.assertEqual(tensor_content_hash(model), tensor_content_hash(final))

    def test_invalid_recipe_scope_and_initialization_are_rejected_before_loading(self):
        root, _, _, _, _, original = self.population("base")
        changes = [{"recipe": "unknown"}, {"inference_contract": "old"}, {"training_komi_halfpoints": [1, 13]},
                   {"training_komi_halfpoints": [13.0]}, {"updates": 200001, "save_updates": [0, 200001]},
                   {"batch_size": True}, {"learning_rate": 0.01}, {"clock_dropout": 0.2},
                   {"save_updates": [0, 9, 8]}, {"initial_checkpoint": "forbidden"}]
        for index, change in enumerate(changes):
            config = copy.deepcopy(original)
            config["parameters"].update(change)
            with self.subTest(change=change), mock.patch("kenggo.baseline_training.TrainingData.load") as loader:
                with self.assertRaises(ValueError):
                    self.run_worker(root, config, "invalid-" + str(index))
                loader.assert_not_called()

    def test_probe_timing_has_five_complete_blocks_and_includes_progress_writes(self):
        root, _, _, _, data, config = self.population("base")
        config["parameters"].update(updates=256, save_updates=[0, 256])
        state = {"attempted_examples": 0, "presented_examples": 0, "phase_seconds": {"train": 0.0}}
        elapsed, snapshots, checkpoints = [0.0], [], []
        original_batch = data.batch
        def batch(*args, **kwargs):
            elapsed[0] += 0.002
            return original_batch(*args, **kwargs)
        def update(*args, **kwargs):
            elapsed[0] += 0.01
            return {"examples": 4, "policy_count": 4, "value_count": 2,
                    "clock_missing_count": 0, "loss": 3.0, "policy_loss": 2.0, "value_loss": 1.0}
        def persist():
            snapshots.append(copy.deepcopy(state))
            elapsed[0] += 0.003
        with mock.patch("kenggo.baseline_training.time.monotonic", side_effect=lambda: elapsed[0]), \
                mock.patch("kenggo.baseline_training.train_step", side_effect=update), \
                mock.patch.object(TrainingData, "batch", side_effect=batch):
            blocks = _expanded_updates(data, None, None, config, state, root, torch.device("cpu"),
                                       persist, checkpoints.append)
        self.assertEqual([row["updates"] for row in blocks], [6, 50, 50, 50, 50, 50])
        self.assertEqual([row["start_step"] for row in blocks], [0, 6, 56, 106, 156, 206])
        for block in blocks:
            self.assertAlmostEqual(block["seconds_per_update"], 0.018, places=10)
        self.assertEqual(checkpoints, [256])
        self.assertEqual((snapshots[0]["attempted_updates"], snapshots[0]["uncertain_updates"]), (1, 1))
        self.assertEqual((snapshots[1]["completed_updates"], snapshots[1]["uncertain_updates"]), (1, 0))
        self.assertEqual((state["completed_updates"], state["presented_examples"]), (256, 1024))


if __name__ == "__main__":
    unittest.main()
