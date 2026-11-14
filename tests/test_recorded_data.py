from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from kenggo.canonical_data import digest, encoded
from kenggo.encoding import INPUT_KEYS, apply_normalizer
from kenggo.examples import (DatasetProfile, ExampleConfig, build_examples, file_hash,
                             fit_normalizer, read_shards)
from kenggo.kgs_source import RANKS
from kenggo.recorded_data import (CommonInputs, main_time_bin, move_bin,
                                  prepare_common_set)
from encoding_fixtures import corpus, record


def write_json(path, value):
    path.write_text(encoded(value) + "\n")
    return file_hash(path)


def bundle(root, records, *, config=ExampleConfig(), production_profile=False):
    root.mkdir(parents=True, exist_ok=True)
    if production_profile:
        for item in records:
            item["sealed_final"] = "final" in item["source"]["source_url"]
    canonical, canonical_pin, splits, split_pin = corpus(
        root / "source", records, validation=("held",) if any(
            "held" in item["eligibility"].values() for item in records) else (),
        test=("reserved",) if any("reserved" in item["eligibility"].values() for item in records) else ())
    profile = None
    if production_profile:
        split = json.loads((splits / "manifest.json").read_text())
        sealed = sorted(item["source"]["source_url"] for item in records if item["sealed_final"])
        split.update(fixture_only=False, sealed_source_ids=sealed,
                     provenance={"population_inputs_sha256": digest("synthetic population inputs")})
        split_pin = write_json(splits / "manifest.json", split)
        profile = DatasetProfile(digest("synthetic profile"), canonical_pin, split_pin,
                                 split["provenance"]["population_inputs_sha256"], digest(sealed),
                                 1000, 100_000, 30, 60)
    examples = root / "examples"
    examples_pin = build_examples(canonical, canonical_pin, splits, split_pin, examples,
                                  fixture_only=not production_profile, profile=profile, config=config)
    normalizer_path = root / "normalizer.json"
    normalizer_pin = write_json(normalizer_path, fit_normalizer(examples, examples_pin))
    return CommonInputs(canonical, canonical_pin, splits, split_pin, examples, examples_pin,
                        normalizer_path, normalizer_pin, profile)


def rebind(inputs):
    canonical_pin = file_hash(inputs.canonical_path)
    split = json.loads((inputs.split_dir / "manifest.json").read_text())
    for name in split["membership_sha256"]:
        split["membership_sha256"][name] = file_hash(inputs.split_dir / name)
    split_pin = write_json(inputs.split_dir / "manifest.json", split)
    profile = replace(inputs.profile, canonical_sha256=canonical_pin, split_sha256=split_pin) if inputs.profile else None
    manifest = json.loads((inputs.examples_dir / "manifest.json").read_text())
    manifest.update(canonical_sha256=canonical_pin, split_sha256=split_pin)
    examples_pin = write_json(inputs.examples_dir / "manifest.json", manifest)
    normalizer = json.loads(inputs.normalizer_path.read_text())
    normalizer["examples_sha256"] = examples_pin
    normalizer_pin = write_json(inputs.normalizer_path, normalizer)
    return replace(inputs, canonical_sha256=canonical_pin, split_sha256=split_pin,
                   examples_sha256=examples_pin, normalizer_sha256=normalizer_pin, profile=profile)


def change_membership(inputs, role, change):
    path = inputs.split_dir / (role + ".parquet")
    table = pq.read_table(path)
    rows = table.to_pylist()
    change(rows)
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)


def source(item):
    return item["source"]["source_url"]


class RecordedDataTests(unittest.TestCase):
    def test_shared_arrays_masks_rank_dropout_keys_and_full_history_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            train = record("train", moves=";B[aa];W[bb];B[]")
            inputs = bundle(Path(temporary), [train], config=ExampleConfig(
                shard_size=2, hide_mover_per_mille=1000, hide_opponent_per_mille=1000))
            common = prepare_common_set(inputs, role="train", source_ids=[source(train)],
                                        aggregate_ranks=("5k", "1d"), fixture_only=True)
            normalizer = json.loads(inputs.normalizer_path.read_text())
            expected = []
            for arrays, metadata in read_shards(inputs.examples_dir, inputs.examples_sha256, "train"):
                normalized = apply_normalizer({key: arrays[key] for key in INPUT_KEYS}, normalizer)
                for index, row in enumerate(metadata):
                    expected.append((normalized, arrays, index, row))
            positions = list(common.positions())
            self.assertEqual(len(common), 3)
            for position, (normalized, arrays, index, row) in zip(positions, expected):
                self.assertEqual(position.observation.row_key, row["row_key"])
                self.assertEqual(position.observation.game.snapshot, row["state_sha256"])
                self.assertEqual(len(position.observation.game.moves), row["ply"])
                self.assertEqual(len(position.observation.game.positions), row["ply"] + 1)
                for key in INPUT_KEYS:
                    np.testing.assert_array_equal(position.observation.inputs[key], normalized[key][index])
                for key, value in position.targets.items():
                    self.assertEqual(value, arrays[key][index])
                for key, value in row.items():
                    self.assertEqual(position.metadata[key], tuple(value) if key == "rank_hidden" else value)
                self.assertEqual(position.observation.inputs["history_actions"].shape, (0,))
                self.assertIsNone(position.observation.context.black_rank)
                self.assertIsNone(position.observation.context.white_rank)
                self.assertFalse(position.observation.inputs["scalar_valid"][12:].any())
            self.assertEqual(positions[-1].targets["action"], 361)
            self.assertEqual(common.manifest["purpose"], "training_debug")
            self.assertEqual(common.sha256, digest(common.manifest))

    def test_frozen_set_does_not_allow_manifest_or_array_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            train = record("train")
            inputs = bundle(Path(temporary), [train])
            common = prepare_common_set(inputs, role="train", source_ids=[source(train)],
                                        aggregate_ranks=("5k",), fixture_only=True)
            manifest = common.manifest
            original = common.sha256
            manifest["rows"].clear()
            position = next(common.positions())
            with self.assertRaises(TypeError):
                position.targets["action"] = 12
            with self.assertRaises(TypeError):
                position.metadata["rank_hidden"][0] = True
            with self.assertRaises(TypeError):
                position.observation.inputs["board"] = np.zeros(1)
            with self.assertRaises(ValueError):
                position.observation.inputs["board"].setflags(write=True)
            position.observation.inputs["board"].shape = (3 * 19 * 19,)
            self.assertEqual(next(common.positions()).observation.inputs["board"].shape, (3, 19, 19))
            self.assertEqual(common.sha256, original)
            self.assertEqual(len(common.manifest["rows"]), 3)

    def test_validation_population_rows_do_not_require_personalization_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            validation = record("validation", day="2025-10-01", groups=("held", "alice"),
                                identities=(False, False), result="W+T")
            inputs = bundle(Path(temporary), [record("train"), validation])
            common = prepare_common_set(inputs, role="validation_development", source_ids=[source(validation)],
                                        aggregate_ranks=("5k", "1d"), fixture_only=True)
            self.assertEqual(len(common), 3)
            for position in common.positions():
                self.assertFalse(position.targets["value_valid"])
                self.assertFalse(position.metadata["identity_valid"])
                self.assertFalse(position.metadata["adaptation_allowed"])
                self.assertEqual(position.metadata["persona_mode"], "default")
                self.assertTrue(position.observation.inputs["rank_valid"].all())
            self.assertEqual(common.manifest["coverage"]["exclusions"]["value_masked_moves"], 3)

    def test_only_positive_selected_sources_are_decoded(self):
        with tempfile.TemporaryDirectory() as temporary:
            records = [record("train"), record("chosen", day="2025-10-01"),
                       record("unselected", day="2025-10-02"),
                       record("confirmation", day="2025-11-01"),
                       record("validation", day="2025-10-01", groups=("held", "alice")),
                       record("validation_confirmation", day="2025-11-02", groups=("held", "alice")),
                       record("excluded", admitted=False),
                       record("final", day="2025-11-03", groups=("reserved", "other"))]
            inputs = bundle(Path(temporary), records, production_profile=True)
            chosen = source(records[1])
            table = pq.read_table(inputs.canonical_path)
            rows = table.to_pylist()
            for row in rows:
                if row["source_url"] != chosen:
                    row["canonical_json"] = "unselected payload must remain closed"
            pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), inputs.canonical_path)
            inputs = rebind(inputs)
            decoded = []
            original = json.loads

            def loads(value, *args, **kwargs):
                result = original(value, *args, **kwargs)
                if isinstance(result, dict) and result.get("version") == "canonical_kgs_v1":
                    decoded.append(result["source"]["source_url"])
                return result

            with mock.patch("kenggo.examples.json.loads", side_effect=loads), mock.patch(
                    "kenggo.examples.pq.read_table", wraps=pq.read_table) as reader:
                common = prepare_common_set(inputs, role="population_development", source_ids=[chosen],
                                            aggregate_ranks=("5k", "1d"))
            self.assertEqual(decoded, [chosen])
            payload_calls = [call for call in reader.call_args_list if
                             "canonical_json" in call.kwargs.get("columns", [])]
            self.assertEqual(len(payload_calls), 1)
            self.assertEqual(payload_calls[0].kwargs["filters"], [("source_url", "in", [chosen])])
            self.assertEqual(common.manifest["source_ids"], [chosen])
            self.assertTrue(all(position.metadata["source_url"] == chosen for position in common.positions()))

    def test_empty_duplicate_missing_excluded_and_wrong_role_selections_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            train, development, excluded = record("train"), record("dev", day="2025-10-01"), record("excluded", admitted=False)
            inputs = bundle(Path(temporary), [train, development, excluded])
            for role, ids in (("train", []), ("train", [source(train)] * 2),
                              ("train", ["missing"]), ("train", [source(development)]),
                              ("train", [source(excluded)]), ("train", source(train)),
                              ("population_confirmation", [source(development)]),
                              ("validation_confirmation", [source(development)]),
                              ("final", [source(train)]), ("test_prefix", [source(train)])):
                with self.subTest(role=role, ids=ids), mock.patch("kenggo.examples.pq.read_table", wraps=pq.read_table) as reader:
                    with self.assertRaises(ValueError):
                        prepare_common_set(inputs, role=role, source_ids=ids,
                                           aggregate_ranks=("5k",), fixture_only=True)
                    self.assertFalse(any("canonical_json" in call.kwargs.get("columns", [])
                                         for call in reader.call_args_list))
            for ranks in ((), ("5k", "5k"), ("unknown",), "5k"):
                with self.subTest(ranks=ranks), self.assertRaises(ValueError):
                    prepare_common_set(inputs, role="train", source_ids=[source(train)],
                                       aggregate_ranks=ranks, fixture_only=True)

    def test_all_membership_hashes_are_checked_before_decoding(self):
        with tempfile.TemporaryDirectory() as temporary:
            train, closed = record("train"), record("confirmation", day="2025-11-01")
            inputs = bundle(Path(temporary), [train, closed])
            change_membership(inputs, "population_confirmation", lambda rows: rows[0].update(
                canonical_sha256=digest("wrong unselected record")))
            inputs = rebind(inputs)
            with mock.patch("kenggo.examples.pq.read_table", wraps=pq.read_table) as reader:
                with self.assertRaisesRegex(ValueError, "Membership hash"):
                    prepare_common_set(inputs, role="train", source_ids=[source(train)],
                                       aggregate_ranks=("5k",), fixture_only=True)
            self.assertFalse(any("canonical_json" in call.kwargs.get("columns", []) for call in reader.call_args_list))

    def test_unselected_membership_roles_and_counts_are_checked_before_payloads(self):
        for change in ({"role": "train"}, {"actions": 99}, {"policy_actions": 99}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                train, closed = record("train"), record("confirmation", day="2025-11-01")
                inputs = bundle(Path(temporary), [train, closed])
                change_membership(inputs, "population_confirmation", lambda rows: rows[0].update(change))
                inputs = rebind(inputs)
                with mock.patch("kenggo.examples.pq.read_table", wraps=pq.read_table) as reader:
                    with self.assertRaises(ValueError):
                        prepare_common_set(inputs, role="train", source_ids=[source(train)],
                                           aggregate_ranks=("5k",), fixture_only=True)
                self.assertFalse(any("canonical_json" in call.kwargs.get("columns", []) for call in reader.call_args_list))

    def test_header_duplicate_and_unassigned_sources_are_rejected_before_payloads(self):
        for corruption in ("duplicate", "unassigned"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as temporary:
                train = record("train")
                inputs = bundle(Path(temporary), [train])
                table = pq.read_table(inputs.canonical_path)
                rows = table.to_pylist()
                extra = dict(rows[0])
                if corruption == "unassigned":
                    extra["source_url"] += "-unassigned"
                extra["canonical_json"] = "closed"
                pq.write_table(pa.Table.from_pylist(rows + [extra], schema=table.schema), inputs.canonical_path)
                inputs = rebind(inputs)
                with mock.patch("kenggo.examples.pq.read_table", wraps=pq.read_table) as reader:
                    with self.assertRaises(ValueError):
                        prepare_common_set(inputs, role="train", source_ids=[source(train)],
                                           aggregate_ranks=("5k",), fixture_only=True)
                self.assertFalse(any("canonical_json" in call.kwargs.get("columns", []) for call in reader.call_args_list))

    def test_population_and_sealed_profile_pins_are_required_before_payloads(self):
        with tempfile.TemporaryDirectory() as temporary:
            train, final = record("train"), record("final", day="2025-11-01", groups=("reserved", "other"))
            inputs = bundle(Path(temporary), [train, final], production_profile=True)
            for name in ("population_inputs_sha256", "sealed_source_ids_sha256", "canonical_sha256", "split_sha256"):
                wrong = replace(inputs, profile=replace(inputs.profile, **{name: digest("wrong")}))
                with self.subTest(name=name), mock.patch("kenggo.examples.pq.read_table", wraps=pq.read_table) as reader:
                    with self.assertRaises(ValueError):
                        prepare_common_set(wrong, role="train", source_ids=[source(train)], aggregate_ranks=("5k",))
                    self.assertFalse(any("canonical_json" in call.kwargs.get("columns", []) for call in reader.call_args_list))
            manifest_path = inputs.split_dir / "manifest.json"
            split = json.loads(manifest_path.read_text())
            split["sealed_source_ids"] = sorted([source(train), source(final)])
            write_json(manifest_path, split)
            inputs = rebind(inputs)
            inputs = replace(inputs, profile=replace(inputs.profile,
                             sealed_source_ids_sha256=digest(split["sealed_source_ids"])))
            with self.assertRaisesRegex(ValueError, "Sealed final source"):
                prepare_common_set(inputs, role="train", source_ids=[source(train)], aggregate_ranks=("5k",))

    def test_illegal_selected_suffix_blocks_the_first_observation(self):
        with tempfile.TemporaryDirectory() as temporary:
            train = record("train", moves=";B[aa];W[bb];B[cc]")
            inputs = bundle(Path(temporary), [train])
            table = pq.read_table(inputs.canonical_path)
            rows = table.to_pylist()
            changed = json.loads(rows[0]["canonical_json"])
            changed["actions"][-1]["action"] = 0
            rows[0].update(canonical_json=encoded(changed), canonical_sha256=digest(changed))
            pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), inputs.canonical_path)
            change_membership(inputs, "train", lambda rows: rows[0].update(canonical_sha256=digest(changed)))
            inputs = rebind(inputs)
            with mock.patch("kenggo.recorded_data.RecordedObservation") as observations:
                with self.assertRaises(ValueError):
                    prepare_common_set(inputs, role="train", source_ids=[source(train)],
                                       aggregate_ranks=("5k",), fixture_only=True)
                observations.assert_not_called()

    def test_unsupported_selected_source_rank_never_reaches_observations(self):
        with tempfile.TemporaryDirectory() as temporary:
            train = record("train")
            inputs = bundle(Path(temporary), [train])
            table = pq.read_table(inputs.canonical_path)
            rows = table.to_pylist()
            changed = json.loads(rows[0]["canonical_json"])
            changed["metadata"]["BR"] = "unknown"
            rows[0].update(canonical_json=encoded(changed), canonical_sha256=digest(changed))
            pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), inputs.canonical_path)
            change_membership(inputs, "train", lambda rows: rows[0].update(canonical_sha256=digest(changed)))
            inputs = rebind(inputs)
            with mock.patch("kenggo.recorded_data.RecordedObservation") as observations:
                with self.assertRaisesRegex(ValueError, "outside the v1 contract"):
                    prepare_common_set(inputs, role="train", source_ids=[source(train)],
                                       aggregate_ranks=("5k",), fixture_only=True)
                observations.assert_not_called()

    def test_counts_finish_validation_before_prepared_set_can_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            train = record("train")
            inputs = bundle(Path(temporary), [train])
            change_membership(inputs, "train", lambda rows: rows[0].update(value_actions=1))
            split_path = inputs.split_dir / "manifest.json"
            split = json.loads(split_path.read_text())
            split["value_actions_by_role"]["train"] = 1
            write_json(split_path, split)
            examples_path = inputs.examples_dir / "manifest.json"
            examples = json.loads(examples_path.read_text())
            examples["counts"]["train"]["value_valid"] = 1
            write_json(examples_path, examples)
            inputs = rebind(inputs)
            with self.assertRaisesRegex(ValueError, "full policy counts"):
                prepare_common_set(inputs, role="train", source_ids=[source(train)],
                                   aggregate_ranks=("5k",), fixture_only=True)

    def test_manifest_and_normalizer_pins_and_history_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            train = record("train")
            inputs = bundle(Path(temporary) / "normal", [train])
            for field in ("examples_sha256", "normalizer_sha256", "canonical_sha256", "split_sha256"):
                with self.subTest(field=field), self.assertRaises(ValueError):
                    prepare_common_set(replace(inputs, **{field: digest("wrong")}), role="train",
                                       source_ids=[source(train)], aggregate_ranks=("5k",), fixture_only=True)
            original = json.loads(inputs.normalizer_path.read_text())
            for field, value in (("examples_sha256", digest("other")), ("fit_role", "population_development"),
                                 ("fitted", False), ("fixture_only", False)):
                changed = replace(inputs, normalizer_sha256=write_json(inputs.normalizer_path, original | {field: value}))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    prepare_common_set(changed, role="train", source_ids=[source(train)],
                                       aggregate_ranks=("5k",), fixture_only=True)
            history = bundle(Path(temporary) / "history", [train], config=ExampleConfig(history=8))
            with self.assertRaisesRegex(ValueError, "zero-history"):
                prepare_common_set(history, role="train", source_ids=[source(train)],
                                   aggregate_ranks=("5k",), fixture_only=True)

    def test_future_actions_clocks_and_results_never_enter_earlier_observations(self):
        with tempfile.TemporaryDirectory() as temporary:
            variants = [record("development", day="2025-10-01", moves=";B[aa]BL[100];W[bb]WL[99];B[cc]", result="B+R"),
                        record("development", day="2025-10-01", moves=";B[aa]BL[999]OB[1];W[dd]WL[bad];B[ee]", result="W+T")]
            positions = []
            for index, development in enumerate(variants):
                inputs = bundle(Path(temporary) / str(index), [record("train"), development])
                positions.append(list(prepare_common_set(inputs, role="population_development",
                    source_ids=[source(development)], aggregate_ranks=("5k", "1d"), fixture_only=True).positions()))
            for first, other in zip(positions[0][:2], positions[1][:2]):
                self.assertEqual(first.observation.row_key, other.observation.row_key)
                self.assertEqual(first.observation.game, other.observation.game)
                self.assertEqual(first.observation.context, other.observation.context)
                for key in INPUT_KEYS:
                    np.testing.assert_array_equal(first.observation.inputs[key], other.observation.inputs[key])
                self.assertTrue(first.targets["value_valid"])
                self.assertFalse(other.targets["value_valid"])
                self.assertIsNone(first.observation.game.outcome)
                self.assertFalse(hasattr(first.observation, "targets"))
                self.assertFalse(hasattr(first.observation, "metadata"))

    def test_coverage_joint_cells_missing_ranks_and_reporting_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            train = record("train")
            low = record("low", day="2025-10-01", nominal_clock=(1199.5, 30, 5), moves=";B[aa];W[];B[bb]")
            boundary = record("boundary", day="2025-10-02", nominal_clock=(1200, 30, 5), moves=";B[cc]")
            inputs = bundle(Path(temporary), [train, low, boundary])
            common = prepare_common_set(inputs, role="population_development",
                source_ids=[source(boundary), source(low)], aggregate_ranks=("5k", "1d"), fixture_only=True)
            coverage = common.manifest["coverage"]
            self.assertEqual(common.manifest["source_ids"], [source(low), source(boundary)])
            self.assertEqual(coverage["candidate"], {"games": 2, "game_sides": 4, "accounts": 2, "moves": 4})
            self.assertEqual(coverage["retained"]["game_sides"], 3)
            self.assertEqual(len(coverage["exclusions"]["game_sides"]), 1)
            self.assertEqual(coverage["aggregate"]["status"], "available")
            self.assertEqual(coverage["all_rank_aggregate"]["status"], "unavailable")
            self.assertEqual(set(coverage["all_rank_aggregate"]["missing_ranks"]), set(RANKS) - {"5k", "1d"})
            self.assertTrue(all(not cell["promotion_supported"] for cell in coverage["joint_cells"]))
            self.assertEqual(sum(cell["moves"] for cell in coverage["joint_cells"]), len(common))
            times = {cell["main_time_bin"]: cell["moves"] for cell in coverage["by"]["main_time_bin"]}
            self.assertEqual(times, {"10-<20": 3, "20-30": 1, "missing": 0})
            requested_absent = prepare_common_set(inputs, role="population_development", source_ids=[source(low)],
                aggregate_ranks=("5k", "9k"), fixture_only=True)
            self.assertEqual(requested_absent.manifest["coverage"]["aggregate"]["missing_ranks"], ["9k"])
            self.assertEqual(requested_absent.manifest["coverage"]["aggregate"]["status"], "unavailable")
        self.assertEqual([move_bin(n) for n in (0, 59, 60, 179, 180)], ["1-60", "1-60", "61-180", "61-180", "181+"])
        self.assertEqual(main_time_bin((600, 30, 5)), "10-<20")
        self.assertEqual(main_time_bin((1800, 30, 5)), "20-30")
        self.assertEqual(main_time_bin(None), "missing")


if __name__ == "__main__":
    unittest.main()
