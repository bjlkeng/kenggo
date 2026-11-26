from collections import Counter
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from encoding_fixtures import corpus, entries, split
from kenggo import policy_validation as validation
from kenggo.canonical_data import canonicalize, digest
from kenggo.checkpoints import save_json
from kenggo.continuation_data import prepare_diagnostic, read, reference
from kenggo.continuation_training import load_diagnostic
from kenggo.encoding import INPUT_KEYS, apply_normalizer
from kenggo.examples import ExampleConfig, LABEL_SCHEMA, build_examples, examples_for_game, fit_normalizer
from kenggo.progressive_evaluation import inventory


def write(filename, value):
    filename.parent.mkdir(parents=True, exist_ok=True)
    save_json(filename, value, replace=True)
    return reference(filename)


def game(name, number, *, komi=6.5, role="validation", result="B+R"):
    groups = {"validation": ("held", "alice"), "train": ("alice", "bob"),
              "population": ("alice", "bob"), "final": ("reserved", "alice"),
              "confirmation": ("held", "alice")}[role]
    day = "2025-07-01" if role == "train" else "2025-11-05" if role in {"final", "confirmation"} else "2025-09-04"
    x = chr(97 + number)
    data = (f"(;FF[4]GM[1]SZ[19]PB[{groups[0]}]PW[{groups[1]}]BR[11k]WR[1d]"
            f"RU[Japanese]KM[{komi}]TM[1200]OT[5x30 byo-yomi]DT[{day}]RE[{result}]"
            f";B[{x}a];W[{x}c];B[{x}e])").encode()
    url = f"https://example.invalid/games/{day.replace('-', '/')}/{name}.sgf"
    record, _ = canonicalize(data, {"source_url": url, "sgf_sha256": sha256(data).hexdigest(), "zip_member": name + ".sgf"},
        {"url": url, "black": groups[0] + " [11k]", "white": groups[1] + " [1d]", "type": "Ranked", "result": result},
        komi_study=True)
    record.update(sealed_final=False, eligibility={"training_eligible": True, "black_group": groups[0],
        "white_group": groups[1], "black_identity_valid": False, "white_identity_valid": False, "eligibility_reasons": []})
    return record


def source(record):
    return record["source"]["source_url"]


class Fixture:
    ranks = ["11k", "1d"]

    def __init__(self, root):
        self.root = root
        records = [game("fit", 0, role="train"), game("keep-a", 1, komi=0.5), game("keep-b", 2, komi=0.5),
                   game("keep-c", 3), game("prior", 4, komi=0.5), game("pop1", 5, komi=0.5, role="population"),
                   game("pop13", 6, role="population"), game("val1", 7, komi=0.5), game("val13", 8),
                   game("final", 9, role="final"), game("confirmation", 10, role="confirmation")]
        self.original_records = records
        self.base = self.dataset(root / "base", records)
        grown_records = records + [game("fit-extra", 11, role="train")]
        self.grown = self.dataset(root / "grown", grown_records)
        parent = self.summary(root / "parent.json", self.base, records)
        fitting = self.summary(root / "fitting.json", self.grown, grown_records)
        self.monitor = prepare_diagnostic(self.base, root / "monitor", purpose="train_monitor", role="train",
                                          source_ids=[source(records[0])], aggregate_ranks=self.ranks)
        self.assessments, common = {}, []
        for record, role, komi in zip(records[5:9], ("population_development", "population_development",
                                                    "validation_development", "validation_development"), (1, 13, 1, 13)):
            identity = role + "_komi" + str(komi)
            ref = prepare_diagnostic(self.base, root / identity, purpose="assessment", role=role,
                                     source_ids=[source(record)], aggregate_ranks=self.ranks)
            self.assessments[identity] = ref
            frozen = write(root / (identity + "-common.json"), {"rows": read(ref)["rows"], "inputs": {
                "examples_sha256": self.base["examples"]["sha256"], "normalizer_sha256": self.base["normalizer"]["sha256"]}})
            common.append({"spec": {"id": identity, "role": role, "source_ids": [source(record)]}, "manifest": frozen})
        fixed = write(root / "fixed.json", {"common_sets": common})
        config = write(root / "original-config.json", {"parameters": {"base_inputs": self.base}})
        original_inventory = inventory(self.base, "validation_development")
        original_ref = write(root / "original-inventory.json", original_inventory)
        supplements = [game("supplement-a", 12), game("supplement-b", 13, result="W+T")]
        membership, _ = split(records + supplements, validation=("held",), test=("reserved",))
        by_source = {row["source_url"]: row for row in membership}
        self.accepted = []
        for record in supplements:
            member = deepcopy(by_source[source(record)])
            member["final_excluded"] = True
            self.accepted.append({"source_url": source(record), "status": "canonical", "membership": member,
                "canonical": write(root / (record["source"]["zip_member"] + ".json"), record),
                "sgf_sha256": record["source"]["sgf_sha256"], "audit_sequence_sha256": record["audit"]["sequence_fingerprint"],
                "trajectory_sha256": record["trajectory_sha256"]})
        accepted = write(root / "accepted.json", {"games": self.accepted})
        combined = write(root / "inventory.json", original_inventory + [{"source_url": source(record), "ranks": self.ranks,
            "rows": 3, "komi_halfpoints": 13} for record in supplements])
        exposure = write(root / "supplement-exposure.json", {"source_ids": [unit["source_url"] for unit in self.accepted],
            "sgf_hashes": [unit["sgf_sha256"] for unit in self.accepted],
            "sequence_hashes": [unit["audit_sequence_sha256"] for unit in self.accepted]})
        source_request = write(root / "source-request.json", {"inputs": {"parent_summary": parent, "grown_summary": fitting}})
        source_summary = write(root / "source-summary.json", {"status": "completed", "yield_status": "sufficient_yield",
            "accepted": self.accepted, "accepted_games": 2, "usable_games": 2, "request": source_request,
            "training_performed": False, "model_queries": 0, "protected_targets_decoded": False})
        source_review = write(root / "source-review.json", {"request_sha256": source_request["sha256"],
            "model": "gpt-6-astra", "reasoning": "max", "verdict": "no significant findings"})
        completion = write(root / "source-completion.json", {"source_summary": source_summary})
        phase = write(root / "source-phase.json", {"status": "completed", "source_summary": completion})
        self.old_request = {"inputs": {"parent_summary": parent, "grown_summary": fitting, "original_config": config,
            "original_inventory": original_ref, "fixed_comparison": fixed, "source_accepted": accepted,
            "source_summary": source_summary, "source_request": source_request, "source_review": source_review,
            "source_phase": phase, "source_exposure": exposure}}
        self.request_ref = write(root / "old-request.json", self.old_request)
        prior = write(root / "prior-selection.json", {"source_ids": [source(records[4])]})
        prior_diagnostic = write(root / "prior-diagnostic.json", {"source_ids": [source(records[4])],
            "purpose": "epoch_validation", "role": "validation_development", "normalizer_sha256": self.base["normalizer"]["sha256"]})
        self.previous = {"schema": "progressive_preparation_v1", "status": "completed", "protected_targets_decoded": False,
            "composite": {"request": self.request_ref, "selection": prior}, "validation_inventory": combined,
            "fixed_comparison": fixed, "supplemental_sources": {"accepted": accepted}, "assessments": self.assessments,
            "train_monitor": self.monitor, "epoch_validation": prior_diagnostic}
        previous_ref = write(root / "previous.json", self.previous)
        self.inputs = {"fitting_summary": fitting, "previous_preparation": previous_ref, "inventory": combined,
            "prior_selection": prior, "fixed_assessment": fixed, "supplements": accepted, "normalizer": self.base["normalizer"]}
        self.parameters = {"base_inputs": self.base, "ranks": self.ranks, "per_cell": 3}

    def dataset(self, root, records):
        root.mkdir()
        canonical, canonical_pin, splits, split_pin = corpus(root / "corpus", deepcopy(records), validation=("held",), test=("reserved",))
        examples = root / "examples"
        pin = build_examples(canonical, canonical_pin, splits, split_pin, examples, fixture_only=True,
                             config=ExampleConfig(shard_size=1), komi_study=True)
        normalizer = write(root / "normalizer.json", fit_normalizer(examples, pin, komi_study=True))
        return {"canonical": reference(canonical), "split": reference(splits / "manifest.json"),
            "examples": reference(examples / "manifest.json"), "normalizer": normalizer, "profile": write(root / "profile.json", {})}

    def summary(self, filename, data, records):
        exposure = write(filename.with_name(filename.stem + "-exposure.json"), {"source_ids": [], "sgf_hashes": [], "sequence_hashes": []})
        index = write(filename.with_name(filename.stem + "-index.json"), {"canonical_sha256": data["canonical"]["sha256"],
            "split_sha256": data["split"]["sha256"], "exposure": exposure,
            "records": [{name: row[name] for name in ("source_url", *validation.FINGERPRINTS)} for row in entries(records)]})
        return write(filename, {"artifacts": {"canonical": data["canonical"], "split_manifest": data["split"],
            "examples_manifest": data["examples"], "normalizer": data["normalizer"], "profile": data["profile"],
            "base_index": index, "exposure": exposure}})

    def update_previous(self):
        self.inputs["previous_preparation"] = write(self.root / "previous.json", self.previous)


class SelectionTests(unittest.TestCase):
    def test_reference_identity_requires_resolved_path_and_hash(self):
        ref = {"path": "tests/test_policy_validation.py", "sha256": "a" * 64}
        absolute = {"path": str(Path(ref["path"]).resolve()), "sha256": ref["sha256"]}
        self.assertTrue(validation.same_ref(ref, absolute))
        self.assertFalse(validation.same_ref(ref, absolute | {"sha256": "b" * 64}))
        self.assertFalse(validation.same_ref(ref, absolute | {"path": str(Path("tests/test_continuation_data.py").resolve())}))

    def population(self):
        games, sides = [], {}
        for komi in (1, 13):
            for number in range(4 if komi == 1 else 2):
                source = f"{komi}/{number}"
                games.append({"source_url": source, "ranks": ["11k", "1d"], "rows": 3, "komi_halfpoints": komi})
                sides[source] = {"B": ("11k", f"mover-{number}"), "W": ("1d", f"opponent-{number}")}
        return games, sides

    def test_order_invariance_sparse_cells_complete_games_and_exclusions(self):
        games, sides = self.population()
        result = validation.select_games(games, sides, excluded={"1/0"}, ranks=("11k", "1d"), per_cell=3)
        self.assertEqual(result, validation.select_games(games[::-1], sides, excluded={"1/0"}, ranks=("11k", "1d"), per_cell=3))
        self.assertEqual(result["unique_games"], 5)
        self.assertEqual(result["expected_rows"], 15)
        self.assertNotIn("1/0", result["source_ids"])
        self.assertEqual([cell["selected_games"] for cell in result["cells"]], [3, 3, 2, 2])
        self.assertEqual([cell["below_requested_quota"] for cell in result["cells"]], [False, False, True, True])

    def test_prefers_new_mover_accounts(self):
        games = [{"source_url": source, "ranks": ["1d"], "rows": 1, "komi_halfpoints": komi}
                 for komi in (1, 13) for source in (f"{komi}/shared-a", f"{komi}/shared-b", f"{komi}/fresh")]
        sides = {game["source_url"]: {"B": ("1d", "fresh" if game["source_url"].endswith("fresh") else "shared")}
                 for game in games}
        result = validation.select_games(games, sides, ranks=("1d",), per_cell=2)
        self.assertTrue(all(len(cell["selected_mover_accounts"]) == 2 for cell in result["cells"]))
        self.assertEqual(result["unique_games"], 4)

    def test_rejects_absent_cells_duplicate_games_and_false_side_ranks(self):
        games, sides = self.population()
        for changed, accounts in ((games + games[:1], sides), (games[:4], sides),
                                  (games, sides | {games[0]["source_url"]: {"B": ("12k", "wrong")}})):
            with self.subTest(), self.assertRaises(ValueError):
                validation.select_games(changed, accounts, ranks=("11k", "1d"))


class PreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.fixture = Fixture(self.root)

    def test_metadata_preflight_opens_no_arrays_or_replay_and_excludes_every_reserved_source(self):
        fixture = self.fixture
        with mock.patch("numpy.load", side_effect=AssertionError("array decoding")), \
                mock.patch.object(validation, "examples_for_game", side_effect=AssertionError("replay")), \
                mock.patch.object(validation, "prepare_diagnostic", side_effect=AssertionError("array preparation")):
            result = validation.preflight(fixture.inputs, fixture.parameters, fixture_only=True)
        self.assertEqual(result["status"], "metadata_verified")
        self.assertEqual(result["selection"]["unique_games"], 5)
        self.assertEqual(result["selection"]["expected_rows"], 15)
        excluded = {source(record) for record in fixture.original_records if not record["source"]["zip_member"].startswith("keep")}
        self.assertFalse(set(result["selection"]["source_ids"]) & excluded)
        self.assertFalse((self.root / "validation").exists())

    def test_prepare_preserves_normalization_whole_games_masks_and_reused_diagnostics(self):
        fixture = self.fixture
        refs = {"monitor": fixture.monitor, **fixture.assessments}
        before = {name: (validation.file_hash(ref["path"]), validation.file_hash(read(ref)["arrays"]["path"])) for name, ref in refs.items()}
        original_load = np.load

        def reject_protected(filename, *args, **kwargs):
            if any(role + "/" in str(filename) for role in validation.PROTECTED):
                raise AssertionError("Protected array was opened")
            return original_load(filename, *args, **kwargs)

        with mock.patch("numpy.load", side_effect=reject_protected):
            result = validation.prepare(fixture.inputs, fixture.parameters, self.root / "prepared", fixture_only=True)
        self.assertEqual(result["train_monitor"], fixture.monitor)
        self.assertEqual(result["assessments"], fixture.assessments)
        manifest = read(result["epoch_validation"])
        selection = read(result["selection"])
        loaded = load_diagnostic(result["epoch_validation"], "epoch_validation", fixture.base["normalizer"]["sha256"], fixture_only=True)
        self.assertEqual(len(loaded.rows), 15)
        self.assertEqual(Counter(row["source_url"] for row in loaded.rows), Counter({source: 3 for source in selection["source_ids"]}))
        original = read(manifest["preparation"]["original_component"])
        with np.load(original["arrays"]["path"], allow_pickle=False) as arrays:
            for name in (*INPUT_KEYS, *LABEL_SCHEMA):
                actual = loaded.inputs[name] if name in INPUT_KEYS else loaded.labels[name]
                np.testing.assert_array_equal(actual[:len(original["rows"])], arrays[name])
        offset = len(original["rows"])
        units = {unit["source_url"]: unit for unit in fixture.accepted}
        normalizer = read(fixture.base["normalizer"])
        context = validation._context(fixture.inputs, fixture.parameters, fixture_only=True)
        for source_id in selection["supplemental_source_ids"]:
            unit = units[source_id]
            for inputs, labels, metadata in examples_for_game(read(unit["canonical"]), unit["membership"], context["plan"],
                    ExampleConfig(**context["example_config"]), komi_study=True):
                expected = apply_normalizer(inputs, normalizer)
                for name in INPUT_KEYS:
                    np.testing.assert_array_equal(loaded.inputs[name][offset], expected[name])
                for name in LABEL_SCHEMA:
                    self.assertEqual(loaded.labels[name][offset].item(), labels[name])
                self.assertEqual({key: loaded.rows[offset][key] for key in metadata}, metadata)
                offset += 1
        self.assertEqual(int(loaded.labels["policy_valid"].sum()), 15)
        self.assertEqual(int(loaded.labels["value_valid"].sum()), 12)
        self.assertFalse(loaded.labels["duration_valid"].any())
        for name, ref in refs.items():
            self.assertEqual(before[name], (validation.file_hash(ref["path"]), validation.file_hash(read(ref)["arrays"]["path"])))
        with self.assertRaises(FileExistsError):
            validation.prepare(fixture.inputs, fixture.parameters, self.root / "prepared", fixture_only=True)

    def test_protected_supplement_rejected_before_reading_its_canonical_record(self):
        fixture = self.fixture
        unit = deepcopy(fixture.accepted[0])
        unit["membership"]["role"] = "validation_confirmation"
        with mock.patch.object(validation, "read", side_effect=AssertionError("canonical decoded")):
            with self.assertRaisesRegex(ValueError, "unsealed development"):
                validation.supplemental_metadata([unit], {"test_groups": ["reserved"], "validation_groups": ["held"],
                    "train_before": "2025-09-01", "confirmation_from": "2025-11-01"}, set(),
                    {name: set() for name in validation.FINGERPRINTS}, check_budget=lambda: None)

    def test_missing_review_rejects_before_any_array_decode_or_output(self):
        fixture = self.fixture
        changed = read(fixture.old_request["inputs"]["source_review"]) | {"verdict": "incomplete"}
        fixture.old_request["inputs"]["source_review"] = write(self.root / "bad-review.json", changed)
        fixture.previous["composite"]["request"] = write(self.root / "old-request.json", fixture.old_request)
        fixture.update_previous()
        with mock.patch("numpy.load", side_effect=AssertionError("array decoded")):
            with self.assertRaisesRegex(ValueError, "independent review"):
                validation.prepare(fixture.inputs, fixture.parameters, self.root / "failed", fixture_only=True)
        self.assertFalse((self.root / "failed").exists())

    def test_fitting_or_protected_duplicate_and_missing_fingerprint_fail(self):
        index = {name: {field: digest([name, field]) for field in validation.FINGERPRINTS}
                 for name in ("candidate", "fit", "protected")}
        validation.validate_fingerprints({"candidate": {}}, index, {"fit", "protected"})
        for forbidden in ("fit", "protected"):
            for field in validation.FINGERPRINTS:
                changed = deepcopy(index)
                changed["candidate"][field] = changed[forbidden][field]
                with self.subTest(forbidden=forbidden, field=field), self.assertRaisesRegex(ValueError, "forbidden fingerprint"):
                    validation.validate_fingerprints({"candidate": {}}, changed, {"fit", "protected"})
        del index["candidate"]["trajectory_sha256"]
        with self.assertRaisesRegex(ValueError, "Missing source fingerprint"):
            validation.validate_fingerprints({"candidate": {}}, index, {"fit"})
        index["candidate"]["trajectory_sha256"] = digest("candidate trajectory")
        del index["protected"]["trajectory_sha256"]
        with self.assertRaisesRegex(ValueError, "Missing source fingerprint"):
            validation.validate_fingerprints({"candidate": {}}, index, {"protected"})
        evidence = validation.validate_fingerprints({"candidate": {}}, index, {"protected"}, opaque_sources={"protected"})
        self.assertEqual(evidence["missing_sealed_trajectory_fingerprints"], 1)
        del index["protected"]["sgf_sha256"]
        with self.assertRaisesRegex(ValueError, "Missing source fingerprint"):
            validation.validate_fingerprints({"candidate": {}}, index, {"protected"}, opaque_sources={"protected"})

    def test_preparation_rejects_refitted_normalizer_or_changed_frozen_assessments(self):
        fixture = self.fixture
        with self.assertRaisesRegex(ValueError, "normalizer"):
            validation.preflight(fixture.inputs | {"normalizer": fixture.grown["normalizer"]}, fixture.parameters, fixture_only=True)
        fixture.previous["assessments"] = {key: value for key, value in fixture.assessments.items() if key != "population_development_komi1"}
        fixture.update_previous()
        with mock.patch("numpy.load", side_effect=AssertionError("array decoded")):
            with self.assertRaisesRegex(ValueError, "four fixed"):
                validation.preflight(fixture.inputs, fixture.parameters, fixture_only=True)


if __name__ == "__main__":
    unittest.main()
