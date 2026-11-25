"""Offline fixtures for the stopping-only composite, without model execution."""

from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
spec = importlib.util.spec_from_file_location("composite_validation", Path(__file__).with_name("composite-validation.py"))
composite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(composite)

from encoding_fixtures import corpus, entries, split
from kenggo.canonical_data import canonicalize, digest
from kenggo.continuation_data import prepare_diagnostic, read, reference
from kenggo.continuation_training import load_diagnostic
from kenggo.data_splits import read_membership
from kenggo.encoding import INPUT_KEYS, apply_normalizer
from kenggo.examples import ExampleConfig, LABEL_SCHEMA, build_examples, examples_for_game, fit_normalizer, read_shards
from kenggo.progressive_evaluation import inventory, choose_games


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    return reference(path)


def game(name, number, *, komi=6.5, role="validation", ranks=("11k", "1d"), result="B+R"):
    groups = {"validation": ("held", "alice"), "train": ("alice", "bob"),
              "population": ("alice", "bob"), "final": ("reserved", "alice")}[role]
    day = "2025-07-01" if role == "train" else "2025-11-05" if role == "final" else "2025-09-04"
    x = chr(97 + number % 17)
    moves = f";B[{x}a];W[{x}c];B[{x}e]"
    data = (f"(;FF[4]GM[1]SZ[19]PB[{groups[0]}]PW[{groups[1]}]BR[{ranks[0]}]WR[{ranks[1]}]"
            f"RU[Japanese]KM[{komi}]TM[1200]OT[5x30 byo-yomi]DT[{day}]RE[{result}]{moves})").encode()
    url = f"https://example.invalid/games/{day.replace('-', '/')}/{name}.sgf"
    source = {"source_url": url, "sgf_sha256": sha256(data).hexdigest(), "zip_member": name + ".sgf"}
    archive = {"url": url, "black": f"{groups[0]} [{ranks[0]}]", "white": f"{groups[1]} [{ranks[1]}]",
               "type": "Ranked", "result": result}
    record, _ = canonicalize(data, source, archive, komi_study=True)
    record.update(sealed_final=False, eligibility={"training_eligible": True,
        "black_group": groups[0], "white_group": groups[1], "black_identity_valid": False,
        "white_identity_valid": False, "eligibility_reasons": []})
    return record


class Fixture:
    def __init__(self, root):
        self.root = root
        records = [game("fit", 0, role="train"), game("old-a", 1, komi=0.5), game("old-b", 2, komi=0.5)]
        records += [game("fixed-pop-1", 3, komi=0.5, role="population"),
                    game("fixed-pop-13", 4, role="population"), game("fixed-val-1", 5, komi=0.5),
                    game("fixed-val-13", 6), game("final", 7, role="final")]
        self.records = records
        self.original = self.dataset(root / "original", records)
        self.grown = self.dataset(root / "grown", records + [game("fit-extra", 8, role="train")])
        self.base = {name: self.original[name] for name in ("canonical", "split", "examples", "normalizer")}
        self.parent = self.summary(root / "parent.json", self.original, records)
        self.growth = self.summary(root / "growth.json", self.grown, records + [game("fit-extra", 8, role="train")])
        self.split, self.members = read_membership(Path(self.base["split"]["path"]).parent, self.base["split"]["sha256"])
        ranks = ["11k", "1d"]
        self.monitor = prepare_diagnostic(self.base, root / "monitor", purpose="train_monitor", role="train",
            source_ids=[records[0]["source"]["source_url"]], aggregate_ranks=ranks)
        self.assessments, common = {}, []
        for record, role, komi in zip(records[3:7], ("population_development", "population_development",
                                                  "validation_development", "validation_development"), (1, 13, 1, 13)):
            identity = role + "_komi" + str(komi)
            sources = [record["source"]["source_url"]]
            ref = prepare_diagnostic(self.base, root / identity, purpose="assessment", role=role,
                                     source_ids=sources, aggregate_ranks=ranks)
            self.assessments[identity] = ref
            diag = read(ref)
            frozen = write(root / (identity + "-common.json"), {"rows": diag["rows"],
                "inputs": {"examples_sha256": self.base["examples"]["sha256"],
                           "normalizer_sha256": self.base["normalizer"]["sha256"]}})
            common.append({"spec": {"id": identity, "role": role, "source_ids": sources}, "manifest": frozen})
        fixed = write(root / "fixed.json", {"common_sets": common})
        original_config = write(root / "config.json", {"inputs": {"fixed_comparison": fixed},
                                                       "parameters": {"base_inputs": self.base}})
        original_inventory = write(root / "original-inventory.json", inventory(self.base, "validation_development"))
        self.supplement = [game("new-a", 9), game("new-b", 10, result="W+T")]
        rows, _ = split(records + self.supplement, validation=("held",), test=("reserved",))
        by_source = {row["source_url"]: row for row in rows}
        self.accepted = []
        for record in self.supplement:
            member = by_source[record["source"]["source_url"]]
            member["final_excluded"] = True
            self.accepted.append({"status": "canonical", "source_url": member["source_url"],
                "canonical": write(root / (record["source"]["zip_member"] + ".json"), record), "membership": member,
                "sgf_sha256": record["source"]["sgf_sha256"],
                "audit_sequence_sha256": record["audit"]["sequence_fingerprint"],
                "trajectory_sha256": record["trajectory_sha256"]})
        accepted_ref = write(root / "accepted.json", {"games": self.accepted})
        exposure = write(root / "supplement-exposure.json", {"source_ids": [u["source_url"] for u in self.accepted],
            "sgf_hashes": [u["sgf_sha256"] for u in self.accepted],
            "sequence_hashes": [u["audit_sequence_sha256"] for u in self.accepted]})
        source_request = write(root / "source-request.json", {"inputs": {"parent_summary": self.parent,
                                                                          "grown_summary": self.growth}})
        source_summary = write(root / "source-summary.json", {"status": "completed", "yield_status": "sufficient_yield",
            "accepted": self.accepted, "accepted_games": 2, "usable_games": 2, "request": source_request,
            "original_deadline_utc": "2026-11-28T14:13:03.148730+00:00", "training_performed": False,
            "model_queries": 0, "protected_targets_decoded": False})
        source_review = write(root / "source-review.json", {"request_sha256": source_request["sha256"],
            "model": "gpt-6-astra", "reasoning": "max", "verdict": "no significant findings"})
        completion = write(root / "source-completion.json", {"source_summary": source_summary})
        phase = write(root / "source-phase.json", {"status": "completed", "source_summary": completion})
        self.request = {"schema": composite.SCHEMA, "fixture_only": True, "code_sha256": {}, "ranks": ranks,
            "deadline_utc": "2026-11-28T14:13:03.148730+00:00", "assessments": self.assessments,
            "inputs": {"original_config": original_config, "fixed_comparison": fixed, "parent_summary": self.parent,
                "grown_summary": self.growth, "original_inventory": original_inventory, "train_monitor": self.monitor,
                "source_summary": source_summary, "source_accepted": accepted_ref, "source_exposure": exposure,
                "source_request": source_request, "source_review": source_review, "source_phase": phase},
            "expected": {"original_validation_games": 4, "supplemental_games": 2, "original_train_rows": 3,
                "grown_train_rows": 6, "grown_train_games": 2, "fixed_games": 4, "fixed_rows": 12, "fixed_scope_rows": 12}}
        self.request_ref = write(root / "request.json", self.request)

    def dataset(self, root, records):
        root.mkdir()
        canonical, _, splits, split_pin = corpus(root / "corpus", deepcopy(records), validation=("held",), test=("reserved",))
        config = ExampleConfig(shard_size=1)
        examples = root / "examples"
        examples_pin = build_examples(canonical, composite.file_hash(canonical), splits, split_pin, examples,
                                       fixture_only=True, config=config, komi_study=True)
        normalizer = write(root / "normalizer.json", fit_normalizer(examples, examples_pin, komi_study=True))
        return {"canonical": reference(canonical), "split": reference(splits / "manifest.json"),
                "examples": reference(examples / "manifest.json"), "normalizer": normalizer,
                "profile": write(root / "profile.json", {})}

    def summary(self, filename, data, records):
        exposure = write(filename.with_name(filename.stem + "-exposure.json"), {"source_ids": [], "sgf_hashes": [],
                                                                              "sequence_hashes": []})
        index = write(filename.with_name(filename.stem + "-index.json"), {"canonical_sha256": data["canonical"]["sha256"],
            "split_sha256": data["split"]["sha256"], "exposure": exposure,
            "records": [{name: row[name] for name in ("source_url", *composite.FINGERPRINTS)} for row in entries(records)]})
        return write(filename, {"artifacts": {"canonical": data["canonical"], "split_manifest": data["split"],
            "examples_manifest": data["examples"], "normalizer": data["normalizer"], "profile": data["profile"],
            "base_index": index, "exposure": exposure}})

    def context(self):
        return composite.context(self.request_ref["path"], self.request_ref["sha256"], fixture=True)


class SelectionTests(unittest.TestCase):
    def inventory(self):
        return [{"source_url": f"{komi}-{number}", "ranks": ["11k", "1d"], "rows": 3, "komi_halfpoints": komi}
                for komi in (1, 13) for number in range(8)]

    def test_unchanged_selector_deterministic_and_deduplicates_cells(self):
        games = [row for row in self.inventory() if int(row["source_url"].split("-")[1]) < 3]
        result = composite.freeze_selection(games, {"1-0"}, [row["source_url"] for row in games], [], ranks=["11k", "1d"])
        repeated = composite.freeze_selection(list(reversed(games)), {"1-0"}, [row["source_url"] for row in games], [],
                                               ranks=["11k", "1d"])
        self.assertEqual(result, repeated)
        self.assertEqual(result["source_ids"], choose_games(games, seed="stopping-20261121-tier-1", excluded={"1-0"},
                                                          ranks=["11k", "1d"]))
        self.assertEqual(result["cell_selections"], 8)
        self.assertLess(len(result["source_ids"]), 8)

    def test_missing_required_cell_stops(self):
        with self.assertRaisesRegex(ValueError, "Insufficient fresh validation"):
            composite.freeze_selection(self.inventory(), {"13-" + str(n) for n in range(7)}, [], [], ranks=["11k", "1d"])

    def test_repeated_inventory_source_stops(self):
        games = self.inventory()
        with self.assertRaisesRegex(ValueError, "Repeated"):
            composite.freeze_selection(games + games[:1], set(), [], [], ranks=["11k", "1d"])

    def test_later_tiers_change_seed_and_exclude_prior_stopping_games(self):
        games = self.inventory()
        first = composite.freeze_selection(games, set(), [row["source_url"] for row in games], [], ranks=["11k", "1d"])
        second = composite.freeze_selection(games, set(first["source_ids"]), [row["source_url"] for row in games], [],
                                             tier=2, ranks=["11k", "1d"])
        self.assertEqual(second["seed"], "stopping-20261121-tier-2")
        self.assertFalse(set(first["source_ids"]).intersection(second["source_ids"]))


class CompositeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fixture = Fixture(self.root)

    def test_metadata_preflight_never_builds_or_decodes_arrays(self):
        with mock.patch.object(composite, "prepare_diagnostic", side_effect=AssertionError("array build")), \
                mock.patch.object(composite, "examples_for_game", side_effect=AssertionError("replay")), \
                mock.patch("numpy.load", side_effect=AssertionError("array decode")):
            result = composite.preflight(self.fixture.request_ref["path"], self.fixture.request_ref["sha256"], fixture=True)
        self.assertEqual(result["status"], "metadata_verified")
        self.assertEqual(result["selection"]["expected_rows"], 12)
        self.assertEqual(result["inventory_games"], 6)
        self.assertFalse((self.root / "composite").exists())

    def test_composite_loads_unchanged_and_preserves_every_array_and_metadata_field(self):
        request = self.fixture.request_ref
        fixed_refs = {"monitor": self.fixture.monitor, **self.fixture.assessments}
        old_hashes = {name: (composite.file_hash(ref["path"]), composite.file_hash(read(ref)["arrays"]["path"]))
                      for name, ref in fixed_refs.items()}
        with mock.patch.object(composite, "prepare_diagnostic", wraps=prepare_diagnostic) as old_build:
            result = composite.build(request["path"], request["sha256"], self.root / "job", fixture=True)
        self.assertEqual(old_build.call_args.args[0], self.fixture.base)
        self.assertEqual(result["train_monitor"], self.fixture.monitor)
        self.assertEqual(result["assessments"], self.fixture.assessments)
        manifest = read(result["epoch_validation"])
        loaded = load_diagnostic(result["epoch_validation"], "epoch_validation",
                                  self.fixture.base["normalizer"]["sha256"], fixture_only=True)
        self.assertEqual(len(loaded.rows), 12)
        self.assertNotIn("split", manifest["data"])
        self.assertEqual(manifest["data"]["original"], self.fixture.base)
        self.assertEqual(read(self.fixture.base["normalizer"])["examples_sha256"], self.fixture.base["examples"]["sha256"])
        ctx = self.fixture.context()
        old_ref = manifest["composite"]["components"][0]["diagnostic"]
        old = read(old_ref)
        with np.load(old["arrays"]["path"], allow_pickle=False) as arrays:
            for name in (*INPUT_KEYS, *LABEL_SCHEMA):
                actual = loaded.inputs[name] if name in INPUT_KEYS else loaded.labels[name]
                np.testing.assert_array_equal(actual[:len(old["rows"])], arrays[name])
        start = len(old["rows"])
        units = {unit["source_url"]: unit for unit in self.fixture.accepted}
        for source in ctx["selection"]["supplemental_source_ids"]:
            unit = units[source]
            raw = list(examples_for_game(read(unit["canonical"]), unit["membership"], ctx["plan"],
                       ExampleConfig(**ctx["example_config"]), komi_study=True))
            for offset, (inputs, labels, metadata) in enumerate(raw, start=start):
                normalized = apply_normalizer(inputs, ctx["normalizer"])
                for name in INPUT_KEYS:
                    np.testing.assert_array_equal(loaded.inputs[name][offset], normalized[name])
                for name in LABEL_SCHEMA:
                    self.assertEqual(loaded.labels[name][offset].item(), labels[name])
                self.assertEqual({key: loaded.rows[offset][key] for key in metadata}, metadata)
                self.assertEqual(loaded.rows[offset]["nominal_clock"], [1200.0, 30.0, 5])
            start += len(raw)
        self.assertEqual(int(loaded.labels["policy_valid"].sum()), 12)
        self.assertEqual(int(loaded.labels["value_valid"].sum()), 9)
        self.assertEqual(loaded.labels["value"][-3:].tolist(), [0, 0, 0])
        self.assertEqual(loaded.labels["value"][-6:-3].tolist(), [1, -1, 1])
        self.assertFalse(loaded.labels["duration_valid"].any())
        for name, ref in fixed_refs.items():
            self.assertEqual(old_hashes[name], (composite.file_hash(ref["path"]), composite.file_hash(read(ref)["arrays"]["path"])))

    def test_selection_is_written_before_any_diagnostic_rows_are_built(self):
        ctx = self.fixture.context()
        work = self.root / "frozen"
        work.mkdir()
        def inspect(*args, **kwargs):
            self.assertEqual(json.loads((work / "selection.json").read_text()), ctx["selection"])
            self.assertEqual(json.loads((work / "inventory.json").read_text()), ctx["inventory"])
            self.assertTrue((work / "growth-exclusions.json").exists())
            return prepare_diagnostic(*args, **kwargs)
        with mock.patch.object(composite, "prepare_diagnostic", side_effect=inspect):
            composite.assemble(ctx, work, fixture=True)

    def test_duplicate_and_leaking_supplement_memberships_fail(self):
        ctx = self.fixture.context()
        cases = [self.fixture.accepted + self.fixture.accepted[:1]]
        for change in ({"role": "train"}, {"groups": ["reserved", "alice"]},
                       {"policy_actions": 2}, {"actions_by_color": [3, 0]}, {"exclusions": ["duplicate_copy"]}):
            accepted = deepcopy(self.fixture.accepted)
            accepted[0]["membership"].update(change)
            cases.append(accepted)
        for accepted in cases:
            with self.subTest(accepted=accepted[0]["membership"]), self.assertRaises(ValueError):
                composite.supplemental_inventory(accepted, ctx["plan"], set(), {key: set() for key in composite.FINGERPRINTS})
        with self.assertRaises(ValueError):
            composite.supplemental_inventory(self.fixture.accepted, ctx["plan"], {self.fixture.accepted[0]["source_url"]},
                                             {key: set() for key in composite.FINGERPRINTS})

    def test_each_historical_fingerprint_blocks_copies(self):
        ctx = self.fixture.context()
        for field in composite.FINGERPRINTS:
            hashes = {name: set() for name in composite.FINGERPRINTS}
            hashes[field].add(self.fixture.accepted[0][field])
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "duplicate"):
                composite.supplemental_inventory(self.fixture.accepted, ctx["plan"], set(), hashes)

    def test_original_inventory_count_and_source_membership_cannot_be_forged(self):
        ref = self.fixture.request["inputs"]["original_inventory"]
        original_read = composite.read
        changed = deepcopy(read(ref))
        changed[0]["rows"] -= 1
        def replaced(reference):
            return changed if reference == ref else original_read(reference)
        with mock.patch.object(composite, "read", side_effect=replaced):
            with self.assertRaisesRegex(ValueError, "complete memberships"):
                self.fixture.context()

    def test_per_head_count_mismatch_fails_before_publication(self):
        ctx = self.fixture.context()
        source = ctx["selection"]["supplemental_source_ids"][0]
        unit = next(unit for unit in ctx["accepted"] if unit["source_url"] == source)
        unit["membership"]["value_actions"] -= 1
        with self.assertRaisesRegex(ValueError, "per-head counts"):
            composite.supplemental_arrays(ctx, [source])

    def test_header_rank_without_played_actions_does_not_fill_a_cell(self):
        ctx = self.fixture.context()
        accepted = deepcopy(self.fixture.accepted[:1])
        record = read(accepted[0]["canonical"])
        record["actions"] = [dict(record["actions"][0], color="W")]
        record["metadata"].update(BR="11k", WR="1d")
        accepted[0]["canonical"] = write(self.root / "zero-11k.json", record)
        accepted[0]["membership"].update(canonical_sha256=digest(record), actions=1, policy_actions=1, value_actions=1,
                                           actions_by_color=[0, 1])
        with self.assertRaisesRegex(ValueError, "complete validation"):
            composite.supplemental_inventory(accepted, ctx["plan"], set(), {name: set() for name in composite.FINGERPRINTS})

    def test_wrong_normalizer_fit_is_rejected_before_sampling(self):
        bad = deepcopy(read(self.fixture.base["normalizer"]))
        bad["examples_sha256"] = self.fixture.grown["examples"]["sha256"]
        current = read(self.fixture.grown["examples"])
        with mock.patch.object(composite, "validate_lineage", return_value=(current, bad)), \
                mock.patch.object(composite, "freeze_selection", side_effect=AssertionError("selection opened")):
            with self.assertRaisesRegex(ValueError, "normalization fit"):
                self.fixture.context()

    def test_fixed_row_key_change_is_rejected(self):
        ref = next(iter(self.fixture.assessments.values()))
        expected = deepcopy(read(ref)["rows"])
        expected[0]["row_key"] = "wrong"
        with self.assertRaisesRegex(ValueError, "row keys"):
            composite.verify_reused(ref, "assessment", self.fixture.base, self.fixture.members, ["11k", "1d"],
                                    expected_rows=expected)

    def test_illegal_suffix_rejects_entire_supplement_and_preserves_failure(self):
        ctx = self.fixture.context()
        source = ctx["selection"]["supplemental_source_ids"][0]
        record = ctx["records"][source]
        record["actions"][-1]["action"] = record["actions"][0]["action"]
        next(unit for unit in ctx["accepted"] if unit["source_url"] == source)["membership"]["canonical_sha256"] = digest(record)
        with mock.patch.object(composite, "context", return_value=ctx), mock.patch.object(composite, "apply_normalizer") as norm:
            with self.assertRaises(ValueError):
                composite.build(self.fixture.request_ref["path"], self.fixture.request_ref["sha256"], self.root / "failure", fixture=True)
            norm.assert_not_called()
        failure = json.loads((self.root / "failure/summary.json").read_text())
        self.assertEqual(failure["status"], "failed")
        self.assertTrue((self.root / "failure/composite/selection.json").exists())
        self.assertFalse((self.root / "failure/composite/manifest.json").exists())
        self.assertTrue((self.root / "failure/composite/original/manifest.json").exists())
        with self.assertRaises(FileExistsError):
            composite.build(self.fixture.request_ref["path"], self.fixture.request_ref["sha256"], self.root / "failure", fixture=True)

    def test_precreated_job_logs_are_allowed_but_success_cannot_be_overwritten(self):
        output = self.root / "job"
        output.mkdir()
        (output / "job.json").write_text("{}")
        (output / "stdout.log").write_text("prior log")
        request = self.fixture.request_ref
        composite.build(request["path"], request["sha256"], output, fixture=True)
        self.assertEqual((output / "stdout.log").read_text(), "prior log")
        before = composite.file_hash(output / "summary.json")
        with self.assertRaises(FileExistsError):
            composite.build(request["path"], request["sha256"], output, fixture=True)
        self.assertEqual(before, composite.file_hash(output / "summary.json"))

    def test_later_tier_requires_complete_pinned_prior_samples_and_stops_without_coverage(self):
        request = self.fixture.request_ref
        with self.assertRaisesRegex(ValueError, "scope"):
            composite.preflight(request["path"], request["sha256"], tier=2, fixture=True)
        composite.build(request["path"], request["sha256"], self.root / "tier1", fixture=True)
        prior = reference(self.root / "tier1/summary.json")
        with self.assertRaisesRegex(ValueError, "Insufficient fresh validation"):
            composite.preflight(request["path"], request["sha256"], tier=2, previous=[prior], fixture=True)


if __name__ == "__main__":
    unittest.main()
