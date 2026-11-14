import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from kenggo.checkpoints import json_hash, load_checkpoint, save_checkpoint, save_json, seed_all
from kenggo.encoding import INPUT_KEYS
from kenggo.examples import (ExampleConfig, LABEL_SCHEMA, build_examples, examples_for_game, fit_normalizer,
                             read_shards)
from kenggo.go import PASS
from kenggo.learner import ModelConfig, PopulationModel
from kenggo.supervised import TrainingData, fit, hide_clocks, supervised_loss, train_step
from encoding_fixtures import corpus, record, split
from test_learner import model_inputs


def label_tensors(rows):
    return {key: torch.from_numpy(np.asarray([row[key] for row in rows], dtype=dtype))
            for key, (dtype, _) in LABEL_SCHEMA.items()}


def labels(actions=(0, PASS), values=(1, -1)):
    return label_tensors([{"action": action, "policy_valid": True, "value": value, "value_valid": True,
                          "duration": 0, "duration_valid": False} for action, value in zip(actions, values)])


def fixture_data(root, *, max_rows=None):
    records = [record("a"), record("b"), record("c"), record("development", day="2025-10-01")]
    args = corpus(root / "source", records)
    output = root / "examples"
    pin = build_examples(*args, output, config=ExampleConfig(shard_size=4), fixture_only=True)
    normalizer = fit_normalizer(output, pin)
    normalizer_path = root / "normalizer.json"
    normalizer_pin = save_json(normalizer_path, normalizer)
    options = dict(split_sha256=args[3], max_rows=max_rows)
    load_args = (output, pin, normalizer_path, normalizer_pin)
    return TrainingData.load(*load_args, **options), load_args, options


class SupervisedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        seed_all(31)

    def test_train_loader_pins_normalizer_examples_and_split_and_keeps_metadata_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            data, args, options = fixture_data(Path(temporary), max_rows=5)
            self.assertEqual(len(data), 5)
            self.assertEqual(set(data.inputs), set(INPUT_KEYS))
            self.assertEqual(set(data.labels), set(LABEL_SCHEMA))
            self.assertTrue(all(not array.flags.writeable for array in data.inputs.values()))
            raw = list(read_shards(args[0], args[1], "train"))
            for key in ("ranks", "rank_valid"):
                np.testing.assert_array_equal(data.inputs[key], np.concatenate([part[0][key] for part in raw])[:5])
            self.assertEqual(data.row_keys, tuple(row["row_key"] for _, meta in raw for row in meta)[:5])
            with self.assertRaises(ValueError):
                TrainingData.load(*args, **(options | {"split_sha256": json_hash("wrong split")}))
            wrong = (args[0], args[1], args[2], json_hash("wrong normalizer"))
            with self.assertRaises(ValueError):
                TrainingData.load(*wrong, **options)
            normalizer = fit_normalizer(args[0], args[1])
            for key, changed in (("examples_sha256", json_hash("other examples")), ("fit_role", "final")):
                replacement = Path(temporary) / (key + ".json")
                replacement_pin = save_json(replacement, normalizer | {key: changed})
                with self.subTest(key=key), self.assertRaises(ValueError):
                    TrainingData.load(args[0], args[1], replacement, replacement_pin, **options)

    def test_loader_never_opens_nontraining_roles(self):
        with tempfile.TemporaryDirectory() as temporary:
            data, args, options = fixture_data(Path(temporary))
            for role in ("population_development", "final", "test_prefix"):
                (args[0] / role / "00000.npz").write_bytes(b"sealed or evaluation arrays must not be read")
            with mock.patch("kenggo.supervised.read_shards", wraps=read_shards) as reader:
                again = TrainingData.load(*args, **options)
                reader.assert_called_once_with(args[0], args[1], "train")
            self.assertEqual(data.row_keys, again.row_keys)

    def test_epoch_permutations_cover_each_row_once_and_depend_only_on_seed_and_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            data, _, _ = fixture_data(Path(temporary))
            batches = [data.indices(step, 4, 19) for step in range(6)]
            self.assertEqual([len(batch) for batch in batches], [4, 4, 1, 4, 4, 1])
            for epoch in range(2):
                np.testing.assert_array_equal(np.sort(np.concatenate(batches[epoch * 3:epoch * 3 + 3])), np.arange(9))
            seed_all(999)
            for step in range(6):
                np.testing.assert_array_equal(data.indices(step, 4, 19), batches[step])
            self.assertFalse(np.array_equal(np.concatenate(batches[:3]), np.concatenate(batches[3:])))
            for step, batch_size, seed in ((-1, 4, 19), (0, 0, 19), (0, 4, -1)):
                with self.assertRaises(ValueError):
                    data.indices(step, batch_size, seed)

    def test_clock_dropout_hides_all_clock_payloads_and_masks_without_revealing_ranks(self):
        inputs = model_inputs()
        inputs["scalars"][:, 9:] = torch.arange(7)
        inputs["scalar_valid"][:, 9:] = True
        original = copy.deepcopy(inputs)
        hidden = hide_clocks(inputs, 1)
        self.assertFalse(hidden["scalar_valid"][:, 9:].any())
        self.assertTrue((hidden["scalars"][:, 9:] == 0).all())
        for key in INPUT_KEYS:
            torch.testing.assert_close(inputs[key], original[key], rtol=0, atol=0)
            if key not in ("scalars", "scalar_valid"):
                torch.testing.assert_close(hidden[key], inputs[key], rtol=0, atol=0)
        for key in ("scalars", "scalar_valid"):
            torch.testing.assert_close(hidden[key][:, :9], inputs[key][:, :9], rtol=0, atol=0)
        for probability in (-1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                hide_clocks(inputs, probability)

    def test_legal_cross_entropy_includes_pass_and_rejects_illegal_observed_targets(self):
        legal = torch.zeros((2, 362), dtype=torch.bool)
        legal[:, [0, PASS]] = True
        logits = torch.full((2, 362), 100.0, requires_grad=True)
        with torch.no_grad():
            logits[:, [0, PASS]] = 0
        value = torch.zeros(2, requires_grad=True)
        targets = labels()
        losses = supervised_loss(logits, value, targets, legal)
        self.assertAlmostEqual(float(losses["policy_loss"].detach()), np.log(2), places=6)
        losses["loss"].backward()
        self.assertTrue((logits.grad[~legal] == 0).all())
        for invalid in (1, 362, -1):
            changed = copy.deepcopy(targets)
            changed["action"][0] = invalid
            with self.subTest(action=invalid), self.assertRaises(ValueError):
                supervised_loss(logits, value, changed, legal)

    def test_source_outcomes_reach_loss_in_mover_view_and_unknown_timeout_forfeit_are_masked(self):
        for result, expected in (("B+R", [1, -1, 1]), ("W+2.5", [-1, 1, -1]),
                                 (None, None), ("W+T", None), ("B+F", None)):
            with self.subTest(result=result):
                source = record(result=result, moves=";B[aa];W[bb];B[]")
                memberships, manifest = split([source])
                rows = list(examples_for_game(source, memberships[0], manifest["plan"], ExampleConfig()))
                targets = label_tensors([row[1] for row in rows])
                legal = torch.from_numpy(np.stack([row[0]["legal"] for row in rows]))
                logits = torch.zeros((3, 362), requires_grad=True)
                value = torch.tensor([0.2, -0.4, 0.6], requires_grad=True)
                if expected is None:
                    self.assertFalse(targets["value_valid"].any())
                    targets["value"][:] = torch.nan
                else:
                    self.assertEqual(targets["value"].tolist(), expected)
                losses = supervised_loss(logits, value, targets, legal)
                if expected is None:
                    self.assertEqual(float(losses["value_loss"].detach()), 0)
                else:
                    mse = ((value.detach() - torch.tensor(expected)) ** 2).mean()
                    torch.testing.assert_close(losses["value_loss"], mse)
                losses["loss"].backward()
                expected_gradient = (torch.zeros(3) if expected is None
                                     else 2 * (value.detach() - torch.tensor(expected)) / 3)
                torch.testing.assert_close(value.grad, expected_gradient)
                self.assertGreater(float(logits.grad.abs().sum()), 0)

    def test_masked_bad_labels_are_excluded_and_value_mean_counts_only_supported_rows(self):
        targets = labels()
        targets["policy_valid"][1] = False
        targets["action"][1] = -999
        targets["value_valid"][1] = False
        targets["value"][1] = torch.nan
        logits = torch.zeros((2, 362), requires_grad=True)
        value = torch.tensor([0.5, -0.9], requires_grad=True)
        legal = torch.ones((2, 362), dtype=torch.bool)
        losses = supervised_loss(logits, value, targets, legal)
        self.assertEqual(float(losses["value_loss"].detach()), 0.25)
        losses["loss"].backward()
        self.assertEqual(value.grad.tolist(), [-1, 0])
        self.assertTrue((logits.grad[1] == 0).all())
        targets["policy_valid"][:] = False
        targets["value_valid"][:] = False
        empty = supervised_loss(logits, value, targets, legal)
        self.assertEqual(float(empty["loss"].detach()), 0)
        empty["loss"].backward()

    def test_small_board_problem_learns_actions_and_opposite_mover_values(self):
        seed_all(11)
        model = PopulationModel(ModelConfig(blocks=1, channels=16))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        inputs, targets = model_inputs(), labels()
        with torch.no_grad():
            initial = float(supervised_loss(*model(inputs), targets, inputs["legal"])["loss"])
        for _ in range(80):
            metrics = train_step(model, optimizer, inputs, targets, clock_dropout=1)
        model.eval()
        with torch.no_grad():
            logits, value = model(inputs)
            final = float(supervised_loss(logits, value, targets, inputs["legal"])["loss"])
        self.assertLess(final, initial * 0.35)
        self.assertEqual(logits.argmax(dim=1).tolist(), [0, PASS])
        self.assertGreater(float(value[0]), 0.5)
        self.assertLess(float(value[1]), -0.5)
        self.assertEqual((metrics["policy_count"], metrics["value_count"], metrics["clock_missing_count"]), (2, 2, 2))

    def test_checkpoint_resume_reproduces_batch_order_dropout_losses_and_model_updates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, _, _ = fixture_data(root)
            model = PopulationModel(ModelConfig(blocks=1, channels=16))
            optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
            options = dict(batch_size=4, seed=43, clock_dropout=0.5)
            cursor = fit(model, optimizer, data, start_step=0, steps=2, **options)
            config_pin = json_hash({"fixture": "population learner", **options})
            provenance_pin = json_hash({"examples": data.examples_sha256, "normalizer": data.normalizer_sha256})
            ref = save_checkpoint(root / "checkpoints", model, optimizer, step=cursor,
                                  progress={"next_step": cursor}, config_sha256=config_pin,
                                  provenance_sha256=provenance_pin)
            expected_metrics = []
            fit(model, optimizer, data, start_step=cursor, steps=5, **options,
                on_step=lambda step, metrics: expected_metrics.append((step, metrics)))
            expected = copy.deepcopy(model.state_dict())
            seed_all(999)
            restored = PopulationModel(ModelConfig(blocks=1, channels=16))
            restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.003)
            progress = load_checkpoint(ref["path"], ref["sha256"], restored, restored_optimizer,
                                       config_sha256=config_pin, provenance_sha256=provenance_pin)
            actual_metrics = []
            next_step = fit(restored, restored_optimizer, data, start_step=progress["step"], steps=5,
                            **options, on_step=lambda step, metrics: actual_metrics.append((step, metrics)))
            self.assertEqual(next_step, 7)
            self.assertEqual(actual_metrics, expected_metrics)
            for key, value in restored.state_dict().items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
