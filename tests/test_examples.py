import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from kenggo.canonical_data import digest
from kenggo.data_splits import ROLES
from kenggo.encoding import Context, INPUT_KEYS, SCALARS, apply_normalizer, encode
from kenggo.examples import (ExampleConfig, build_examples, examples_for_game, fit_normalizer,
                             load_corpus, rank_hidden, read_shards)
from kenggo.go import Game
from encoding_fixtures import corpus, record, split


def examples(row, config=ExampleConfig(hide_mover_per_mille=0, hide_opponent_per_mille=0), **cohorts):
    memberships, manifest = split([row], **cohorts)
    return list(examples_for_game(row, memberships[0], manifest["plan"], config))


class ExampleTests(unittest.TestCase):
    def test_constant_fractional_clock_uses_unit_scale_without_cancellation(self):
        actions = [i // 2 if i % 2 == 0 else 360 - i // 2 for i in range(167)]
        moves = "".join(f";{'B' if i % 2 == 0 else 'W'}[{chr(97 + a % 19)}{chr(97 + a // 19)}]"
                        for i, a in enumerate(actions))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = corpus(root / "source", [record(moves=moves, nominal_clock=(900.1, 30, 5))])
            key = build_examples(*args, root / "out", fixture_only=True)
            normalizer = fit_normalizer(root / "out", key)
            main = SCALARS.index("main_seconds")
            self.assertEqual(normalizer["count"][main], 167)
            self.assertEqual(normalizer["scale"][main], 1)
            self.assertAlmostEqual(normalizer["scale"][5], np.arange(167, dtype=np.float64).std())
            inputs = encode(Game.new(), Context(nominal_clock=(900.2, 30, 5)))
            transformed = apply_normalizer(inputs, normalizer)
            self.assertAlmostEqual(float(transformed["scalars"][main]),
                                   float(np.float32(900.2) - np.float32(900.1)))

    def test_source_clock_family_does_not_become_a_fixed_default(self):
        for clock in ((600, 60, 3), (1800, 20, 5), (900.5, 30.5, 4)):
            with self.subTest(clock=clock):
                rows = examples(record(nominal_clock=clock))
                self.assertEqual(rows[0][0]["scalars"][9:12].tolist(), list(clock))
                self.assertFalse(rows[0][0]["scalar_valid"][12:].any())

    def test_pre_move_inputs_never_include_target_future_clocks_or_results(self):
        original = record(moves=";B[aa]BL[100];W[bb]WL[99];B[cc]", result="B+R")
        changed = record(moves=";B[aa]BL[999]OB[1];W[cc]WL[bad];B[dd]", result="W+T")
        first, other = examples(original)[0], examples(changed)[0]
        for name in INPUT_KEYS:
            np.testing.assert_array_equal(first[0][name], other[0][name])
        self.assertFalse(first[0]["board"].any())
        self.assertTrue(first[1]["value_valid"])
        self.assertFalse(other[1]["value_valid"])
        self.assertFalse(first[1]["duration_valid"])
        self.assertFalse(first[0]["scalar_valid"][12:].any())
        self.assertNotIn("persona_group", first[0])
        self.assertEqual(first[2]["source_rank_mover"], "5k")
        self.assertEqual(first[2]["sgf_node"], 1)

    def test_value_signs_result_masks_eof_and_explicit_pass(self):
        for result, valid in (("B+R", True), ("W+2.5", True), ("W+T", False), ("B+F", False), (None, False)):
            with self.subTest(result=result):
                rows = examples(record(result=result, moves=";B[aa];W[bb];B[]"))
                self.assertEqual(len(rows), 3)
                self.assertEqual([r[1]["action"] for r in rows], [0, 20, 361])
                self.assertEqual([r[1]["value_valid"] for r in rows], [valid] * 3)
                expected = [1, -1, 1] if result == "B+R" else [-1, 1, -1] if valid else [0] * 3
                self.assertEqual([r[1]["value"] for r in rows], expected)
        self.assertEqual(len(examples(record(moves=";B[];W[]"))), 2)

    def test_entire_game_checked_before_first_example_and_closed_gates(self):
        row = record(moves=";B[aa];W[bb]")
        row["actions"].append({"color": "B", "action": 0, "node": 3})
        memberships, manifest = split([row])
        iterator = examples_for_game(row, memberships[0], manifest["plan"], ExampleConfig())
        with self.assertRaises(ValueError):
            next(iterator)
        self.assertEqual(examples(record(admitted=False)), [])
        row = record()
        memberships, manifest = split([row])
        row["eligibility"]["training_eligible"] = False
        memberships[0]["canonical_sha256"] = digest(row)
        with self.assertRaises(ValueError):
            list(examples_for_game(row, memberships[0], manifest["plan"], ExampleConfig()))

    def test_prefix_masks_target_colors_and_later_cohort_annotations(self):
        for groups, target, count in ((("held", "bob"), "B", 2), (("alice", "held"), "W", 1)):
            rows = examples(record(groups=groups), validation=("held",))
            self.assertEqual(len(rows), count)
            self.assertTrue(all(r[2]["mover"] == target and r[2]["adaptation_allowed"] for r in rows))
            self.assertTrue(all(not r[1]["value_valid"] for r in rows))
            self.assertTrue(all(r[2]["persona_mode"] == "adaptation_target" for r in rows))
            later = examples(record(groups=groups, day="2025-11-01"), test=("held",))
            self.assertEqual(len(later), 3)
            self.assertEqual(sum(r[2]["personalization_target_valid"] for r in later), count)
            self.assertTrue(all(not r[2]["adaptation_allowed"] for r in later))
            self.assertTrue(all(r[2]["persona_mode"] == "default" for r in later))
        self.assertEqual(examples(record(groups=("held", "bob"), identities=(False, True)), validation=("held",)), [])
        both = examples(record(groups=("held", "also")), validation=("held", "also"))
        self.assertEqual(len(both), 3)

    def test_rank_masking_is_seeded_train_only_and_source_labels_survive(self):
        config = ExampleConfig(rank_seed=9, hide_mover_per_mille=1000, hide_opponent_per_mille=1000)
        rows = examples(record(), config)
        self.assertTrue(all(not r[0]["rank_valid"].any() and r[2]["rank_hidden"] == [True, True] for r in rows))
        self.assertEqual(rows[1][2]["source_rank_mover"], "1d")
        later = examples(record(day="2025-10-01"), config)
        self.assertTrue(all(r[0]["rank_valid"].all() and r[2]["rank_hidden"] == [False, False] for r in later))
        first = [rank_hidden("fixture", i, ExampleConfig()) for i in range(100)]
        reverse = {i: rank_hidden("fixture", i, ExampleConfig()) for i in reversed(range(100))}
        self.assertEqual(first, [reverse[i] for i in range(100)])
        self.assertNotEqual(first, [rank_hidden("fixture", i, ExampleConfig(rank_seed=1)) for i in range(100)])

    def test_shards_reproduce_bind_hashes_and_preserve_empty_schemas(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = corpus(root / "source", [record(moves=";B[aa];W[bb];B[]")])
            config = ExampleConfig(history=8, shard_size=2)
            first = build_examples(*args, root / "a", config=config, fixture_only=True)
            second = build_examples(*args, root / "b", config=config, fixture_only=True)
            self.assertEqual(first, second)
            shards = list(read_shards(root / "a", first, "train"))
            self.assertEqual([len(m) for _, m in shards], [2, 1])
            expected = examples(record(moves=";B[aa];W[bb];B[]"), config)
            for index, (arrays, metadata) in enumerate(shards):
                for offset, meta in enumerate(metadata):
                    sample = expected[2 * index + offset]
                    self.assertEqual(meta, sample[2])
                    for name in INPUT_KEYS:
                        np.testing.assert_array_equal(arrays[name][offset], sample[0][name])
            for role in ROLES:
                if role != "train":
                    arrays, metadata = next(read_shards(root / "a", first, role))
                    self.assertEqual(arrays["board"].shape, (0, 3, 19, 19))
                    self.assertEqual(metadata, [])
            with self.assertRaises(FileExistsError):
                build_examples(*args, root / "a", fixture_only=True)
            with self.assertRaises(ValueError):
                load_corpus(*args)
            wrong = list(args)
            wrong[1] = digest("changed")
            with self.assertRaises(ValueError):
                load_corpus(*wrong, fixture_only=True)
            with self.assertRaises(ValueError):
                list(read_shards(root / "a", digest("changed"), "train"))
            (root / "a/train/00000.npz").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                list(read_shards(root / "a", first, "train"))

    def test_normalizer_fits_only_valid_training_inputs_and_never_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = [record("train", moves=";B[aa];W[bb];B[cc]"),
                       record("later", day="2025-10-01", moves=";B[dd];W[ee];B[ff];W[gg];B[hh]")]
            args = corpus(root / "source", records)
            key = build_examples(*args, root / "out", fixture_only=True)
            normalizer = fit_normalizer(root / "out", key)
            self.assertTrue(normalizer["fitted"])
            self.assertTrue(normalizer["fixture_only"])
            self.assertEqual(normalizer["mean"][SCALARS.index("move_number")], 1)
            self.assertEqual(normalizer["count"][SCALARS.index("move_number")], 3)
            self.assertTrue(all(n == 0 for n in normalizer["count"][12:]))
            arrays, _ = next(read_shards(root / "out", key, "train"))
            transformed = apply_normalizer({k: arrays[k] for k in INPUT_KEYS}, normalizer)
            self.assertAlmostEqual(float(transformed["scalars"][:, 5].mean()), 0)
            self.assertFalse(transformed["scalar_valid"][:, 12:].any())
            bad = copy.deepcopy(normalizer)
            bad["fit_role"] = "validation_prefix"
            with self.assertRaises(ValueError):
                apply_normalizer({k: arrays[k] for k in INPUT_KEYS}, bad)
            (root / "out/population_development/00000.npz").write_bytes(b"must not read for fitting")
            self.assertEqual(fit_normalizer(root / "out", key), normalizer)

    def test_zero_admissions_produce_no_fitted_statistics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = corpus(root / "source", [record(admitted=False)])
            key = build_examples(*args, root / "out", fixture_only=True)
            normalizer = fit_normalizer(root / "out", key)
            self.assertFalse(normalizer["fitted"])
            self.assertEqual(sum(normalizer["count"]), 0)
            manifest = json.loads((root / "out/manifest.json").read_text())
            self.assertEqual(sum(x["examples"] for x in manifest["counts"].values()), 0)
            arrays, _ = next(read_shards(root / "out", key, "train"))
            with self.assertRaises(ValueError):
                apply_normalizer({k: arrays[k] for k in INPUT_KEYS}, normalizer)


if __name__ == "__main__":
    unittest.main()
