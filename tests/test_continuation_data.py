from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from kenggo import continuation_data as data
from kenggo.canonical_data import digest
from kenggo.examples import ExampleConfig, read_shards
from encoding_fixtures import record
from test_recorded_data import bundle, change_membership, rebind, write_json


def refs(current, fitted=None):
    fitted = fitted or current
    return {"examples": data.reference(current.examples_dir / "manifest.json"),
            "split": data.reference(current.split_dir / "manifest.json"),
            "normalizer": data.reference(fitted.normalizer_path),
            "normalization_examples": data.reference(fitted.examples_dir / "manifest.json"),
            "normalization_split": data.reference(fitted.split_dir / "manifest.json")}


def fixture(root):
    originals = [record("fit", moves=";B[aa];W[bb]"),
        record("development", day="2025-09-04", groups=("held", "alice"), moves=";B[cc];W[dd]"),
        record("confirmation", day="2025-11-04", groups=("held", "alice"), moves=";B[ee];W[ff]"),
        record("final", day="2025-11-06", groups=("reserved", "alice"), moves=";B[gg];W[hh]")]
    config = ExampleConfig(shard_size=2)
    fitted = bundle(root / "fit", deepcopy(originals), config=config)
    current = bundle(root / "grown", deepcopy(originals) + [record("new", moves=";B[ii];W[jj]")], config=config)
    return fitted, current, originals


class ContinuationDataTests(unittest.TestCase):
    def test_old_fit_is_bound_to_exact_surviving_training_shards_and_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            fitted, current, _ = fixture(Path(temporary))
            inputs = refs(current, fitted)
            manifest, normalizer = data.validate_lineage(inputs, fixture_only=True)
            self.assertEqual(manifest["counts"]["train"]["examples"], 4)
            self.assertEqual(normalizer["count"][5], 2)
            self.assertEqual(normalizer["examples_sha256"], inputs["normalization_examples"]["sha256"])
            fit_manifest = data.read(inputs["normalization_examples"])
            self.assertEqual(manifest["artifacts"]["train"][:len(fit_manifest["artifacts"]["train"])],
                             fit_manifest["artifacts"]["train"])

    def test_additive_group_registry_digest_preserves_the_original_fit(self):
        with tempfile.TemporaryDirectory() as temporary:
            fitted, current, _ = fixture(Path(temporary))
            filename = current.split_dir / "manifest.json"
            split = json.loads(filename.read_text())
            split["plan"]["grouping_sha256"] = digest("old aliases preserved plus a fresh account")
            split["plan_sha256"] = digest(split["plan"])
            write_json(filename, split)
            current = rebind(current)
            data.validate_lineage(refs(current, fitted), fixture_only=True)

    def test_false_fit_binding_and_nontraining_fit_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fitted, current, _ = fixture(root)
            original = json.loads(fitted.normalizer_path.read_text())
            for change in ({"examples_sha256": current.examples_sha256}, {"fit_role": "validation_development"}, {"fitted": False}):
                with self.subTest(change=change):
                    filename = root / "invalid-normalizer.json"
                    write_json(filename, {**original, **change})
                    inputs = refs(current, fitted) | {"normalizer": data.reference(filename)}
                    with self.assertRaises(ValueError):
                        data.validate_lineage(inputs, fixture_only=True)

    def test_declared_fit_count_must_match_the_frozen_fitting_examples(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fitted, current, _ = fixture(root)
            normalizer = json.loads(fitted.normalizer_path.read_text())
            normalizer["count"][5] = 4
            filename = root / "false-fit-count.json"
            write_json(filename, normalizer)
            with self.assertRaises(ValueError):
                data.validate_lineage(refs(current, fitted) | {"normalizer": data.reference(filename)}, fixture_only=True)

    def test_old_training_membership_and_protected_memberships_cannot_change(self):
        for role in ("train", "validation_development", "validation_confirmation", "final"):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as temporary:
                fitted, current, _ = fixture(Path(temporary))
                change_membership(current, role, lambda rows: rows[0].update(groups=["changed", "alice"]))
                current = rebind(current)
                with self.assertRaises(ValueError):
                    data.validate_lineage(refs(current, fitted), fixture_only=True)

    def test_lost_original_training_shard_fails_lineage_before_loading_arrays(self):
        with tempfile.TemporaryDirectory() as temporary:
            fitted, current, _ = fixture(Path(temporary))
            filename = current.examples_dir / "manifest.json"
            manifest = json.loads(filename.read_text())
            manifest["artifacts"]["train"][0]["arrays_sha256"] = "0" * 64
            write_json(filename, manifest)
            with mock.patch("kenggo.continuation_data.read_shards") as opened:
                with self.assertRaisesRegex(ValueError, "training shards"):
                    data.load_training(refs(current, fitted), fixture_only=True)
                opened.assert_not_called()

    def test_loader_opens_only_training_and_keeps_original_normalizer_and_unique_keys(self):
        with tempfile.TemporaryDirectory() as temporary:
            fitted, current, _ = fixture(Path(temporary))
            inputs = refs(current, fitted)
            original_load = np.load

            def no_protected(path, *args, **kwargs):
                if any(role in str(path) for role in ("development/", "confirmation/", "final/")):
                    raise AssertionError("Loader opened a held-out array")
                return original_load(path, *args, **kwargs)

            with mock.patch("kenggo.continuation_data.read_shards", wraps=read_shards) as opened, \
                    mock.patch("numpy.load", side_effect=no_protected):
                loaded = data.load_training(inputs, fixture_only=True)
            self.assertEqual([call.args[2] for call in opened.call_args_list], ["train"])
            expected = tuple(row["row_key"] for _, rows in read_shards(current.examples_dir, current.examples_sha256, "train")
                             for row in rows)
            self.assertEqual(loaded.row_keys, expected)
            self.assertEqual(len(set(loaded.row_keys)), 4)
            self.assertEqual(loaded.normalizer_sha256, fitted.normalizer_sha256)
            self.assertTrue(all(not array.flags.writeable for array in (*loaded.inputs.values(), *loaded.labels.values())))

    def test_loader_rejects_duplicate_training_row_keys(self):
        with tempfile.TemporaryDirectory() as temporary:
            fitted, current, _ = fixture(Path(temporary))
            shards = list(read_shards(current.examples_dir, current.examples_sha256, "train"))
            shards[-1][1][-1]["row_key"] = shards[0][1][0]["row_key"]
            with mock.patch("kenggo.continuation_data.read_shards", return_value=iter(shards)):
                with self.assertRaisesRegex(ValueError, "unique"):
                    data.load_training(refs(current, fitted), fixture_only=True)

    def test_diagnostics_keep_complete_frozen_keys_and_a_disjoint_purpose(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fitted, _, original = fixture(root)
            inputs = refs(fitted)
            train_key, dev_key = (item["source"]["source_url"] for item in original[:2])
            expected = [row["row_key"] for _, rows in read_shards(fitted.examples_dir, fitted.examples_sha256,
                        "validation_development") for row in rows]
            with mock.patch("kenggo.continuation_data.read_shards", wraps=read_shards) as opened:
                pin = data.prepare_diagnostic(inputs, root / "diagnostic", purpose="epoch_validation",
                    role="validation_development", source_ids=[dev_key], aggregate_ranks=["5k", "1d"],
                    excluded_source_ids=[train_key], expected_row_keys=expected)
            saved = data.read(pin)
            self.assertEqual([call.args[2] for call in opened.call_args_list], ["validation_development"])
            self.assertEqual([row["row_key"] for row in saved["rows"]], expected)
            self.assertEqual(saved["source_ids"], [dev_key])
            self.assertEqual(len(saved["rows"]), 2)
            self.assertTrue(saved["inputs_are_normalized"])
            for number, (purpose, role, keys, excluded, frozen) in enumerate([
                    ("assessment", "train", [train_key], [], None),
                    ("train_monitor", "validation_development", [dev_key], [], None),
                    ("assessment", "final", [original[-1]["source"]["source_url"]], [], None),
                    ("epoch_validation", "validation_development", [dev_key], [dev_key], None),
                    ("assessment", "validation_development", [dev_key], [], expected[:-1]),
                    ("assessment", "validation_development", [dev_key, dev_key], [], None)]):
                with self.subTest(purpose=purpose, role=role, number=number), self.assertRaises(ValueError):
                    data.prepare_diagnostic(inputs, root / str(number), purpose=purpose, role=role,
                        source_ids=keys, aggregate_ranks=["5k", "1d"], excluded_source_ids=excluded,
                        expected_row_keys=frozen)

    def test_diagnostic_rejects_a_normalizer_fitted_to_different_examples(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fitted, current, original = fixture(root)
            with self.assertRaisesRegex(ValueError, "pinned"):
                data.prepare_diagnostic(refs(current, fitted), root / "diagnostic", purpose="assessment",
                    role="validation_development", source_ids=[original[1]["source"]["source_url"]],
                    aggregate_ranks=["5k", "1d"])


if __name__ == "__main__":
    unittest.main()
