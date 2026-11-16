import copy
from contextlib import redirect_stdout
from hashlib import sha256
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from kenggo.canonical_data import canonicalize, digest, encoded
from kenggo.encoding import INPUT_KEYS, SCALARS
from kenggo.examples import (KOMI_STUDY_PROFILE_VERSION, KOMI_STUDY_VERSION, PROFILE_VERSION, VERSION,
                             ExampleConfig, build_examples, examples_for_game, file_hash, fit_normalizer,
                             initial_game, load_corpus, load_dataset_profile, read_shards, replay_record,
                             source_context)
from kenggo.examples_audit import diagnostic_sample, probe
from kenggo.go import KOMI_STUDY_RULES, RULES
from kenggo.kgs_admission import KOMI_STUDY_POLICY, classify
from kenggo.kgs_eligibility import exposure_registry, handle_groups
from kenggo.sgf_audit import inspect_sgf
from encoding_fixtures import corpus, split
from test_canonical_data import source as old_source


def source(name="fixture", *, komi=0.5, **kwargs):
    data, ref, archive = old_source(**kwargs)
    data = data.replace(b"KM[6.5]", f"KM[{komi}]".encode())
    url = ref["source_url"].replace("fixture.sgf", name + ".sgf")
    ref.update(source_url=url, sgf_sha256=sha256(data).hexdigest(), zip_member=name + ".sgf")
    archive["url"] = url
    return data, ref, archive


def record(name="fixture", **kwargs):
    row, _ = canonicalize(*source(name, **kwargs), komi_study=True)
    row.update(sealed_final=False, eligibility={
        "training_eligible": True, "black_group": "alice", "white_group": "bob",
        "black_identity_valid": False, "white_identity_valid": False, "eligibility_reasons": []})
    return row


def emit(row, *, komi_study=True):
    memberships, manifest = split([row])
    config = ExampleConfig(hide_mover_per_mille=0, hide_opponent_per_mille=0)
    return list(examples_for_game(row, memberships[0], manifest["plan"], config, komi_study=komi_study))


def save(path, value):
    path.write_text(encoded(value) + "\n")
    return file_hash(path)


def production_profile(root, args, *, version=KOMI_STUDY_PROFILE_VERSION, sealed=()):
    canonical, canonical_pin, splits, _ = args
    manifest = json.loads((splits / "manifest.json").read_text())
    population_pin = digest("synthetic komi population")
    manifest.update(fixture_only=False, sealed_source_ids=sorted(sealed),
                    provenance={"population_inputs_sha256": population_pin})
    split_pin = save(splits / "manifest.json", manifest)
    profile = {"version": version, "canonical_sha256": canonical_pin, "split_sha256": split_pin,
               "population_inputs_sha256": population_pin, "sealed_source_ids_sha256": digest(sorted(sealed)),
               "limits": {"max_records": 10, "max_examples_per_role": 100,
                          "cpu_seconds": 30, "wall_seconds": 60}}
    path = root / (version + ".json")
    return (canonical, canonical_pin, splits, split_pin), load_dataset_profile(path, save(path, profile))


class KomiDataTests(unittest.TestCase):
    def test_opt_in_changes_only_the_allowed_komi(self):
        data, _, archive = source()
        self.assertIn("rules_or_komi_out_of_scope", inspect_sgf(data, archive)["exclusions"])
        self.assertEqual(inspect_sgf(data, archive, komi_study=True)["exclusions"], [])
        rejected, _ = canonicalize(*source())
        self.assertFalse(rejected["replay_validated"])
        self.assertIn("rules_or_komi_out_of_scope", rejected["structural_exclusions"])
        for before, after, reason in (
            (b"KM[0.5]", b"KM[7.5]", "rules_or_komi_out_of_scope"),
            (b"RU[Japanese]", b"RU[Chinese]", "rules_or_komi_out_of_scope"),
            (b"TM[1200]", b"TM[300]", "time_control_unknown_or_out_of_scope"),
            (b"BR[5k]", b"BR[16k]", "sgf_rank_unknown_or_out_of_scope"),
            (b";B[aa]", b"AB[cc];B[aa]", "setup_or_handicap"),
            (b";W[bb]", b"(;W[bb])(;W[cc])", "played_variation_unverified"),
        ):
            with self.subTest(reason=reason, replacement=after):
                audit = inspect_sgf(data.replace(before, after), archive, komi_study=True)
                self.assertIn(reason, audit["exclusions"])
        with self.assertRaises(ValueError):
            inspect_sgf(data, komi_study=1)

    def test_admission_keeps_bot_identity_exposure_and_scope_gates(self):
        data, ref, archive = source()
        audit = {**inspect_sgf(data, archive, komi_study=True), "source_url": ref["source_url"]}
        groups = handle_groups(["Alice", "Bob", "GnuGo2", "cu2"])
        exposure = exposure_registry([audit], [], groups)
        self.assertFalse(classify(audit, archive, groups, exposure)["training_eligible"])
        admitted = classify(audit, archive, groups, exposure, komi_study=True)
        self.assertTrue(admitted["training_eligible"])
        self.assertEqual(admitted["policy"], KOMI_STUDY_POLICY)
        self.assertTrue(admitted["final_evaluation_excluded"])
        self.assertFalse(admitted["black_identity_valid"])
        self.assertFalse(admitted["white_identity_valid"])
        for change in ({"exclusions": ["time_control_unknown_or_out_of_scope"]},
                       {"metadata": {**audit["metadata"], "PB": "GnuGo2"}}):
            self.assertFalse(classify({**audit, **change}, archive, groups, exposure,
                                      komi_study=True)["training_eligible"])

    def test_source_values_passes_and_masks_use_the_correct_mover_komi(self):
        for result, values in (("B+R", [1, -1, 1]), ("W+0.5", [-1, 1, -1]),
                               ("W+T", [0, 0, 0]), ("B+F", [0, 0, 0]), (None, [0, 0, 0])):
            with self.subTest(result=result):
                row = record(moves=";B[aa]BL[100];W[bb]WL[99];B[]", result=result)
                self.assertTrue(row["replay_validated"])
                self.assertEqual(row["rules_contract"], KOMI_STUDY_RULES)
                examples = emit(row)
                self.assertEqual([e[1]["value"] for e in examples], values)
                self.assertEqual([e[1]["value_valid"] for e in examples], [any(values)] * 3)
                self.assertEqual([e[1]["action"] for e in examples], [0, 20, 361])
                self.assertEqual([e[0]["scalars"][SCALARS.index("mover_komi")] for e in examples], [-0.5, 0.5, -0.5])
                for inputs, labels, metadata in examples:
                    self.assertTrue(inputs["legal"][labels["action"]])
                    self.assertFalse(inputs["scalar_valid"][12:].any())
                    self.assertFalse(labels["duration_valid"])
                    self.assertFalse(metadata["identity_valid"])
                    self.assertEqual(inputs["history_valid"].shape, (0,))
                with self.assertRaises(ValueError):
                    emit(row, komi_study=False)

    def test_uncertified_endings_remain_unresolved_without_losing_source_labels(self):
        row = record(moves=";B[];W[]", result="W+0.5")
        self.assertEqual(row["replay_end_status"], "pending_adjudication")
        self.assertEqual(row["certification"]["status"], "unresolved")
        self.assertTrue(row["recorded_value_supported"])
        self.assertEqual([e[1]["action"] for e in emit(row)], [361, 361])
        self.assertEqual([e[1]["value"] for e in emit(row)], [-1, 1])

    def test_komi_does_not_change_legality_or_leak_results_and_clocks(self):
        original = record(moves=";B[aa]BL[100];W[bb]WL[99];B[]")
        control = record(komi=6.5, moves=";B[aa]BL[100];W[bb]WL[99];B[]")
        first, old = replay_record(original, komi_study=True), replay_record(control)
        self.assertEqual(first.positions, old.positions)
        self.assertEqual(first.moves, old.moves)
        self.assertNotEqual(first.snapshot, old.snapshot)
        first_inputs = emit(original)[0][0]
        changed = record(moves=";B[aa]BL[999];W[cc]WL[bad];B[dd]", result="W+T")
        for name in INPUT_KEYS:
            np.testing.assert_array_equal(first_inputs[name], emit(changed)[0][0][name])
        invalid = copy.deepcopy(original)
        invalid["actions"].append({"color": "W", "action": 0, "node": 4})
        with self.assertRaises(ValueError):
            emit(invalid)

    def test_replay_rejects_rules_and_snapshot_komi_mismatches(self):
        row = record()
        with self.assertRaises(ValueError):
            source_context(row)
        with self.assertRaises(ValueError):
            initial_game(row)
        for field, value in (("rules_contract", RULES), ("replay_snapshot", replay_record(record(komi=6.5)).snapshot)):
            changed = {**row, field: value}
            with self.subTest(field=field), self.assertRaises(ValueError):
                replay_record(changed, komi_study=True)

    def test_budget_stop_aborts_canonicalization_without_salvaging_a_record(self):
        checks = []

        def check_budget():
            checks.append(None)
            if len(checks) >= 4:
                raise RuntimeError("study budget exhausted")

        with self.assertRaisesRegex(RuntimeError, "study budget exhausted"):
            canonicalize(*source(), komi_study=True, check_budget=check_budget)
        self.assertEqual(len(checks), 4)

    def test_example_budget_applies_before_validation_and_between_rows(self):
        row = record(moves=";B[aa];W[bb];B[]")
        memberships, manifest = split([row])
        expired = False

        def check_budget():
            if expired:
                raise RuntimeError("study budget exhausted")

        iterator = examples_for_game(row, memberships[0], manifest["plan"], ExampleConfig(),
                                     komi_study=True, check_budget=check_budget)
        first = next(iterator)
        self.assertEqual(first[1]["action"], 0)
        expired = True
        with self.assertRaisesRegex(RuntimeError, "study budget exhausted"):
            next(iterator)
        with mock.patch("kenggo.examples.Game.new") as new_game:
            with self.assertRaisesRegex(RuntimeError, "study budget exhausted"):
                replay_record(row, komi_study=True, check_budget=check_budget)
            new_game.assert_not_called()

    def test_old_canonical_records_and_example_bytes_are_unchanged_with_opt_in(self):
        args = source(komi=6.5, moves=";B[aa];W[bb];B[]")
        legacy, _ = canonicalize(*args)
        study, _ = canonicalize(*args, komi_study=True)
        self.assertEqual(encoded(legacy), encoded(study))
        row = record(komi=6.5, moves=";B[aa];W[bb];B[]")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = corpus(root / "source", [row])
            old_pin = build_examples(*inputs, root / "old", fixture_only=True)
            new_pin = build_examples(*inputs, root / "new", fixture_only=True, komi_study=True)
            old = json.loads((root / "old/manifest.json").read_text())
            new = json.loads((root / "new/manifest.json").read_text())
            self.assertEqual(old["schema"], new["schema"])
            self.assertEqual(old["config"], new["config"])
            for role, shards in old["artifacts"].items():
                self.assertEqual(shards, new["artifacts"][role])
                for shard in shards:
                    for kind in ("arrays", "metadata"):
                        self.assertEqual((root / "old" / shard[kind]).read_bytes(),
                                         (root / "new" / shard[kind]).read_bytes())
            self.assertEqual(len(list(read_shards(root / "old", old_pin, "train", komi_study=True))), 1)
            with self.assertRaisesRegex(ValueError, "Unsupported example contract"):
                list(read_shards(root / "new", new_pin, "train"))

    def test_versioned_profile_reads_both_komi_and_fits_only_training_rows(self):
        old = record("old", komi=6.5, moves=";B[aa];W[bb];B[]")
        added = record("added", moves=";B[cc];W[dd]")
        development = record("development", day="2025-10-01", moves=";B[ee];W[ff];B[gg];W[hh];B[]")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = corpus(root / "source", [old, added, development])
            pin = build_examples(*inputs, root / "examples", fixture_only=True, komi_study=True)
            inputs, profile = production_profile(root, inputs)
            _, legacy = production_profile(root, inputs, version=PROFILE_VERSION)
            self.assertTrue(profile.komi_study)
            self.assertFalse(legacy.komi_study)
            with self.assertRaisesRegex(ValueError, "versioned dataset profile"):
                load_corpus(*inputs, profile=legacy)
            keys = [row["source"]["source_url"] for row in (old, added)]
            records, _, _ = load_corpus(*inputs, profile=profile, role="train", source_ids=keys)
            self.assertEqual(records[old["source"]["source_url"]], old)
            with self.assertRaisesRegex(ValueError, "explicit allowed role"):
                load_corpus(*inputs, profile=profile)
            with self.assertRaisesRegex(ValueError, "reuse the unchanged base shards"):
                build_examples(*inputs, root / "rebuild", profile=profile)
            self.assertFalse((root / "rebuild").exists())
            manifest = json.loads((root / "examples/manifest.json").read_text())
            self.assertEqual(manifest["version"], KOMI_STUDY_VERSION)
            self.assertEqual(manifest["rules"], KOMI_STUDY_RULES)
            arrays, _ = next(read_shards(root / "examples", pin, "train", komi_study=True))
            self.assertEqual(set(abs(arrays["scalars"][:, 1])), {0.5, 6.5})
            with self.assertRaises(ValueError):
                fit_normalizer(root / "examples", pin)
            normalizer = fit_normalizer(root / "examples", pin, komi_study=True)
            move = SCALARS.index("move_number")
            self.assertEqual(normalizer["count"][move], 5)
            self.assertAlmostEqual(normalizer["mean"][move], 0.8)
            (root / "examples/population_development/00000.npz").write_bytes(b"must remain unread")
            self.assertEqual(fit_normalizer(root / "examples", pin, komi_study=True), normalizer)
            manifest.update(version=VERSION, rules=RULES)
            changed = save(root / "examples/manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "Example komi"):
                list(read_shards(root / "examples", changed, "train", komi_study=True))

    def test_study_readers_do_not_decode_confirmation_or_final_targets(self):
        train = record("train")
        confirmation = record("confirmation", komi=6.5, day="2025-11-01", moves=";B[cc];W[dd]")
        final = record("final", komi=6.5, day="2025-11-02", moves=";B[ee];W[ff]")
        final["eligibility"]["black_group"] = "reserved"
        final["sealed_final"] = True
        final_key = final["source"]["source_url"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = corpus(root / "source", [train, confirmation, final], test=("reserved",))
            inputs, profile = production_profile(root, inputs, sealed=[final_key])
            original_loads = json.loads

            def reject_held_out(value, *args, **kwargs):
                if isinstance(value, str) and '"source":{' in value:
                    if any(row["source"]["source_url"] in value for row in (confirmation, final)):
                        raise AssertionError("A komi study decoded a held-out target")
                return original_loads(value, *args, **kwargs)

            with mock.patch("kenggo.examples.json.loads", side_effect=reject_held_out):
                with self.assertRaisesRegex(ValueError, "explicit allowed role"):
                    load_corpus(*inputs, profile=profile)
                records, _, parts = load_corpus(*inputs, profile=profile, role="train",
                                                source_ids=[train["source"]["source_url"]])
                self.assertEqual(list(records.values()), [train])
                self.assertEqual(parts["population_confirmation"], [])
                self.assertEqual(parts["final"], [])
                for role, row in (("population_confirmation", confirmation), ("final", final)):
                    with self.assertRaisesRegex(ValueError, "training-debug or development"):
                        load_corpus(*inputs, profile=profile, role=role, source_ids=[row["source"]["source_url"]])
                with redirect_stdout(io.StringIO()):
                    checked = probe(root / "source", profile=profile, games=1)
                self.assertEqual(checked["games"], 1)
                self.assertEqual(checked["examples"], len(train["actions"]))
                self.assertFalse(checked["artifacts_written"])

    def test_new_komi_cannot_emit_confirmation_examples_or_read_unapproved_shards(self):
        with self.assertRaisesRegex(ValueError, "training and development"):
            emit(record(day="2025-11-01"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = corpus(root / "source", [record("old", komi=6.5)])
            build_examples(*inputs, root / "examples", fixture_only=True, komi_study=True)
            path = root / "examples/manifest.json"
            manifest = json.loads(path.read_text())
            manifest.update(fixture_only=False, dataset_profile_sha256=digest("synthetic profile"))
            pin = save(path, manifest)
            with mock.patch("kenggo.examples.np.load") as arrays:
                with self.assertRaisesRegex(PermissionError, "training and development"):
                    list(read_shards(root / "examples", pin, "population_confirmation", komi_study=True))
                arrays.assert_not_called()

    def test_normalizer_budget_stop_does_not_return_partial_statistics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = corpus(root / "source", [record()])
            pin = build_examples(*inputs, root / "examples", fixture_only=True, komi_study=True)

            def expired():
                raise RuntimeError("study budget exhausted")

            with mock.patch("kenggo.examples.read_shards") as reader:
                with self.assertRaisesRegex(RuntimeError, "study budget exhausted"):
                    fit_normalizer(root / "examples", pin, komi_study=True, check_budget=expired)
                reader.assert_not_called()

    def test_diagnostics_include_both_komi_without_decoding_sealed_records(self):
        old, added = record("old", komi=6.5), record("added")
        sealed = record("sealed")
        sealed["sealed_final"] = True
        records = {row["source"]["source_url"]: row for row in (old, added, sealed)}
        checked = diagnostic_sample(records, komi_study=True)
        self.assertEqual(checked["games"], 2)
        self.assertEqual(checked["mismatches"], 0)
        self.assertEqual({item["source_url"] for item in checked["evidence"]},
                         {old["source"]["source_url"], added["source"]["source_url"]})


if __name__ == "__main__":
    unittest.main()
