import copy
from contextlib import ExitStack
from hashlib import sha256
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from kenggo import expanded_jobs as jobs
from kenggo.baseline_evaluation import _save_predictions, rows_for_common
from kenggo.canonical_data import encoded
from kenggo.checkpoints import file_hash, save_json
from kenggo.examples import load_corpus
from kenggo.kgs_source import RANKS
from kenggo.recorded_data import prepare_common_set
from kenggo.recorded_statistics import Prediction

import test_expanded_evaluation as evaluation_fixtures
from test_recorded_data import rebind


SOURCE = {"commit": "a" * 40, "patch_sha256": sha256(b"").hexdigest()}


def candidate(source, *, ranks=("15k", "14k"), counts=(2, 1), komi=13,
              role="population_development"):
    return {"source_url": source, "role": role, "komi_halfpoints": komi,
            "ranks": list(ranks), "actions_by_color": list(counts), "actions": sum(counts),
            "policy_actions": sum(counts), "value_actions": sum(counts),
            "groups": [source + "-black", source + "-white"], "exclusions": []}


class SelectionTests(unittest.TestCase):
    def select(self, candidates, **kwargs):
        return jobs.select_games(candidates, role="population_development", komi_halfpoints=13, **kwargs)

    def test_selection_is_permutation_and_target_independent(self):
        candidates = [candidate(f"game-{index}") for index in range(12)]
        expected = self.select(candidates)
        self.assertEqual(expected, self.select(candidates[::-1]))
        changed = copy.deepcopy(candidates)
        for index, game in enumerate(changed):
            game.update(value_actions=0, winner="W" if index % 2 else "B",
                        prediction_nll=100 - index, action_targets=[361, 0, 1])
        self.assertEqual(expected, self.select(changed))
        self.assertEqual(expected["spec"]["aggregate_ranks"], RANKS)
        for rank in ("15k", "14k"):
            ordered = sorted((game["source_url"] for game in candidates), key=lambda source: sha256(
                json.dumps([20261118, "population_development", 13, rank, source],
                           sort_keys=True, separators=(",", ":")).encode()).hexdigest())
            self.assertEqual(expected["rank_choices"][rank]["chosen_source_ids"], ordered[:2])

    def test_review_selection_expands_whole_games_only_with_explicit_opt_in(self):
        candidates = [candidate(f"game-{index}") for index in range(40)]
        selected = self.select(candidates, review=True)
        self.assertEqual(selected, self.select(candidates[::-1], review=True))
        for rank in ("15k", "14k"):
            self.assertEqual(len(selected["rank_choices"][rank]["chosen_source_ids"]), 10)
        self.assertGreater(selected["selected_games"], self.select(candidates)["selected_games"])
        changed = copy.deepcopy(candidates)
        for row in changed:
            row.update(value_actions=0, prediction_nll=1, winner="B")
        self.assertEqual(selected, self.select(changed, review=True))
        with self.assertRaises(ValueError):
            self.select(candidates, selection=jobs.REVIEW_SELECTION)
        with self.assertRaises(ValueError):
            self.select(candidates, review=True, selection=dict(jobs.REVIEW_SELECTION, max_rows_per_set=50001))

    def test_shared_games_are_unioned_and_all_rank_shortfalls_retained(self):
        result = self.select([candidate("one"), candidate("two")])
        self.assertEqual(result["spec"]["source_ids"], ["one", "two"])
        self.assertEqual(result["spec"]["expected_rows"], 6)
        self.assertEqual(result["selected_games"], 2)
        self.assertEqual(list(result["rank_choices"]), RANKS)
        self.assertEqual([result["rank_choices"][rank]["shortfall"] for rank in RANKS], [0, 0] + [2] * 14)

    def test_rank_with_no_mover_actions_does_not_contribute_coverage(self):
        result = self.select([candidate("one", counts=(1, 0))])
        self.assertEqual(result["rank_choices"]["15k"]["eligible_games"], 1)
        self.assertEqual(result["rank_choices"]["14k"]["eligible_games"], 0)
        self.assertEqual(result["rank_choices"]["14k"]["selected_games_with_rank_moves"], 0)
        self.assertEqual(result["rank_choices"]["14k"]["shortfall"], 2)

    def test_caps_skip_whole_games_and_continue_to_fitting_games(self):
        limits = {"games_per_rank": 2, "max_games_per_set": 2, "max_rows_per_set": 5}
        result = self.select([candidate("too-large", counts=(4, 2)), candidate("three"),
                              candidate("two", counts=(1, 1))], selection=limits)
        self.assertEqual(result["spec"]["source_ids"], ["three", "two"])
        self.assertEqual(result["spec"]["expected_rows"], 5)
        limits["max_games_per_set"] = 1
        capped = self.select([candidate("one"), candidate("two")], selection=limits)
        self.assertEqual(capped["selected_games"], 1)
        self.assertEqual(capped["spec"]["expected_rows"], 3)

    def test_duplicate_or_incomplete_membership_is_rejected(self):
        with self.assertRaises(ValueError):
            self.select([candidate("duplicate"), candidate("duplicate")])
        for change in ({"policy_actions": 2}, {"value_actions": 2}, {"exclusions": ["excluded"]},
                       {"groups": ["same", "same"]}, {"actions_by_color": [3, 1]},
                       {"role": "population_confirmation"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.select([{**candidate("bad"), **change}])

    def test_sets_filter_roles_and_komi_without_reusing_sources(self):
        candidates = [candidate(f"{role}-{komi}", role=role, komi=komi)
                      for role in jobs.ROLES for komi in jobs.KOMI]
        selected = [jobs.select_games(candidates, role=role, komi_halfpoints=komi)
                    for role in jobs.ROLES for komi in jobs.KOMI]
        self.assertEqual([item["spec"]["source_ids"] for item in selected],
                         [[game["source_url"]] for game in candidates])


class ExpandedJobTests(unittest.TestCase):
    def setUp(self):
        self.fixture = evaluation_fixtures.ExpandedEvaluationTests()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.root = self.fixture.root
        self.inputs = self.fixture.combined

    def config(self, purpose="expanded-preparation"):
        refs = {"canonical": jobs._ref(self.inputs.canonical_path),
                "split": jobs._ref(self.inputs.split_dir / "manifest.json"),
                "examples": jobs._ref(self.inputs.examples_dir / "manifest.json"),
                "normalizer": jobs._ref(self.inputs.normalizer_path)}
        for name in ("data", "profile", "rules", "cpu_environment"):
            path = self.root / (name + ".json")
            save_json(path, {"fixture": name}, replace=path.exists())
            refs[name] = jobs._ref(path)
        return {"version": "experiment_config_v1", "entity": "fixture", "project": "fixture",
                "base_url": "https://api.wandb.ai", "comparison": "expanded-jobs", "seed": jobs.SEED,
                "arm": "preparation" if purpose == "expanded-preparation" else "analysis",
                "purpose": purpose, "budget_id": "fixture", "inputs": refs,
                "parameters": {"selection": dict(jobs.SELECTION)} if purpose == "expanded-preparation" else {}}

    def synthetic_inputs(self):
        stack = ExitStack()
        stack.enter_context(patch.object(jobs, "_common_inputs", return_value=self.inputs))
        stack.enter_context(patch.object(jobs, "load_corpus", side_effect=lambda *args, **kwargs:
            load_corpus(*args, fixture_only=True, **kwargs)))
        stack.enter_context(patch.object(jobs, "prepare_common_set", side_effect=lambda *args, **kwargs:
            prepare_common_set(*args, fixture_only=True, **kwargs)))
        return stack

    def prepare(self):
        output = self.root / "preparation"
        output.mkdir()
        with self.synthetic_inputs():
            summary = jobs.prepare(self.config(), output)
        return output, summary

    def analysis(self):
        _, prepared = self.prepare()
        config = self.config("expanded-analysis")
        specs = []
        for item in prepared["common_sets"]:
            spec = {key: item["spec"][key] for key in ("id", "role", "source_ids", "aggregate_ranks")}
            spec.update(manifest_input=spec["id"] + "-manifest", prediction_inputs={})
            config["inputs"][spec["manifest_input"]] = item["manifest"]
            common = prepare_common_set(self.inputs, role=spec["role"], source_ids=spec["source_ids"],
                aggregate_ranks=spec["aggregate_ranks"], fixture_only=True, raw_inputs=True, komi_study=True)
            rows = rows_for_common(common)
            for arm, loss in zip(jobs.ARMS, (2., 3., 1.5, 2.5)):
                predictions = []
                for row in rows:
                    remainder = math.log1p(-math.exp(-loss)) - math.log(sum(row["legal"]) - 1)
                    logs = [remainder if allowed else -math.inf for allowed in row["legal"]]
                    logs[row["action"]] = -loss
                    predictions.append(Prediction(row["row_key"], tuple(logs), 0.))
                name = spec["id"] + "-" + arm
                path = self.root / (name + ".npz")
                _save_predictions(path, predictions, seed=jobs.SEED, variant="ordinary")
                config["inputs"][name] = jobs._ref(path)
                spec["prediction_inputs"][arm] = name
            specs.append(spec)
        config["parameters"]["common_sets"] = specs
        output = self.root / "analysis"
        output.mkdir()
        return config, output

    def test_preparation_freezes_four_raw_sets_and_rank_shortfalls(self):
        output, summary = self.prepare()
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["predictions"], 0)
        self.assertEqual(len(summary["common_sets"]), 4)
        observed = set()
        for item in summary["common_sets"]:
            spec = item["spec"]
            self.assertEqual(spec["aggregate_ranks"], RANKS)
            self.assertEqual(spec["expected_rows"], item["rows"])
            manifest = json.loads(Path(item["manifest"]["path"]).read_text())
            self.assertEqual(file_hash(item["manifest"]["path"]), item["manifest"]["sha256"])
            self.assertEqual(manifest["version"], "recorded_raw_common_set_v1")
            self.assertEqual({row["komi_halfpoints"] for row in manifest["rows"]}, {spec["komi_halfpoints"]})
            self.assertFalse(observed.intersection(spec["source_ids"]))
            observed.update(spec["source_ids"])
        selection = json.loads((output / "selection.json").read_text())
        self.assertEqual(len(selection["sets"]), 4)
        self.assertTrue(all(item["rank_choices"]["15k"]["shortfall"] == 2 for item in selection["sets"]))

    def test_only_allowed_development_payloads_are_decoded(self):
        allowed = {item["source"]["source_url"] for _, _, item in self.fixture.selected}
        table = pq.read_table(self.inputs.canonical_path)
        rows = table.to_pylist()
        for row in rows:
            if row["source_url"] not in allowed:
                row["canonical_json"] = "TRAINING, CONFIRMATION AND FINAL PAYLOADS MUST STAY CLOSED"
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), self.inputs.canonical_path)
        self.inputs = rebind(self.inputs)
        with patch("kenggo.examples.pq.read_table", wraps=pq.read_table) as reader:
            self.prepare()
        payload_calls = [call for call in reader.call_args_list if "canonical_json" in call.kwargs.get("columns", [])]
        self.assertGreater(len(payload_calls), 0)
        for call in payload_calls:
            self.assertEqual(call.kwargs["filters"][0][:2], ("source_url", "in"))
            self.assertTrue(set(call.kwargs["filters"][0][2]) <= allowed)

    def test_membership_hash_is_checked_before_canonical_payloads(self):
        path = self.inputs.split_dir / "population_confirmation.parquet"
        path.write_bytes(path.read_bytes() + b"changed")
        with patch("kenggo.examples.pq.read_table", wraps=pq.read_table) as reader, self.synthetic_inputs():
            with self.assertRaisesRegex(ValueError, "membership changed"):
                jobs._development_candidates(self.inputs)
        self.assertFalse(any("canonical_json" in call.kwargs.get("columns", []) for call in reader.call_args_list))

    def test_analysis_preserves_contrast_signs_masks_and_missing_rank_status(self):
        config, output = self.analysis()
        with self.synthetic_inputs():
            result = jobs.analyze(config, output)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["promotion"], "not_promoted")
        self.assertEqual(len(result["comparisons"]), 4)
        expected = dict(zip((candidate + "_minus_" + control for candidate, control in jobs.CONTRASTS),
                            (-1., -1., -.5, -.5)))
        for identifier, reports in result["comparisons"].items():
            self.assertEqual(set(reports), set(expected))
            for name, item in reports.items():
                report = json.loads(Path(item["report"]["path"]).read_text())
                self.assertEqual(item["overall"], report["overall"])
                self.assertAlmostEqual(report["by_rank"]["5k"]["metrics"]["nll"]["estimate"]["value"], expected[name])
                self.assertIn("15k", report["overall"]["metrics"]["nll"]["missing_ranks"])
                if identifier.endswith("komi1"):
                    value = report["overall"]["metrics"]["value_mse"]
                    self.assertEqual(value["status"], "unavailable")
                    self.assertEqual(value["estimate"], {"status": "undefined", "value": None})

    def test_review_analysis_keeps_full_reports_and_a_named_rank_subset(self):
        config, output = self.analysis()
        config["purpose"] = "expanded-review-analysis"
        with self.synthetic_inputs():
            result = jobs.analyze(config, output)
        for reports in result["comparisons"].values():
            for item in reports.values():
                report = json.loads(Path(item["report"]["path"]).read_text())
                self.assertEqual(report["aggregate_ranks"], RANKS)
                self.assertEqual(report["review_scope"]["aggregate_ranks"], RANKS[3:])
                self.assertEqual(report["expected_outputs"], report["review_scope"]["expected_outputs"])
                self.assertEqual(report["review_scope"]["excluded_rows"], 0)

    def test_missing_arm_or_stratum_is_rejected_before_comparing(self):
        config, output = self.analysis()
        config["parameters"]["common_sets"][0]["prediction_inputs"].pop(jobs.ARMS[0])
        with patch.object(jobs, "paired_report", side_effect=AssertionError("Must not compare")):
            with self.assertRaises(ValueError):
                jobs.analyze(config, output)
        config["parameters"]["common_sets"].pop(0)
        with self.assertRaises(ValueError):
            jobs.analyze(config, output)

    def test_missing_or_reordered_prediction_rows_are_rejected_with_matching_file_pin(self):
        config, output = self.analysis()
        spec = config["parameters"]["common_sets"][0]
        name = spec["prediction_inputs"][jobs.ARMS[0]]
        path = Path(config["inputs"][name]["path"])
        manifest = json.loads(Path(config["inputs"][spec["manifest_input"]]["path"]).read_text())
        predictions = jobs.load_predictions(path, expected_row_keys=[row["row_key"] for row in manifest["rows"]])
        for changed in (predictions[::-1], predictions[:-1]):
            path.unlink()
            _save_predictions(path, changed, seed=jobs.SEED, variant="ordinary")
            config["inputs"][name] = jobs._ref(path)
            with self.subTest(rows=len(changed)), self.synthetic_inputs(), self.assertRaisesRegex(ValueError, "incomplete, reordered"):
                jobs.analyze(config, output)

    def test_changed_prediction_bytes_and_common_manifests_are_rejected(self):
        config, output = self.analysis()
        spec = config["parameters"]["common_sets"][0]
        name = spec["prediction_inputs"][jobs.ARMS[0]]
        path = Path(config["inputs"][name]["path"])
        path.write_bytes(path.read_bytes() + b"changed")
        with self.synthetic_inputs(), self.assertRaisesRegex(ValueError, "evaluation pin"):
            jobs.analyze(config, output)
        manifest_name = spec["manifest_input"]
        path = Path(config["inputs"][manifest_name]["path"])
        manifest = json.loads(path.read_text())
        manifest["targets_sha256"] = "0" * 64
        path.write_text(encoded(manifest))
        config["inputs"][manifest_name] = jobs._ref(path)
        with self.synthetic_inputs(), self.assertRaisesRegex(ValueError, "common set changed"):
            jobs.analyze(config, output)


class WorkerValidationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        refs = {}
        for name in ("data", "canonical", "split", "examples", "normalizer", "profile", "rules", "cpu_environment"):
            path = self.root / (name + ".json")
            save_json(path, {"fixture": name})
            refs[name] = jobs._ref(path)
        self.config = {"version": "experiment_config_v1", "entity": "fixture", "project": "fixture",
            "base_url": "https://api.wandb.ai", "comparison": "fixture", "arm": "prepare",
            "purpose": "expanded-preparation", "seed": jobs.SEED, "budget_id": "fixture", "inputs": refs,
            "parameters": {"selection": jobs.SELECTION}}
        self.provenance = {"version": "experiment_provenance_v1", "inputs": refs,
                           "source": SOURCE, "environment": {"fixture_only": True}}
        self.output = self.root / "output"
        self.output.mkdir()

    def invoke(self, *, revisions=None):
        revisions = iter(revisions or [SOURCE["commit"], SOURCE["commit"]])
        config_path, provenance_path = self.root / "config.json", self.root / "provenance.json"
        config_pin = save_json(config_path, self.config)
        provenance_pin = save_json(provenance_path, self.provenance)
        def git(arguments, **kwargs):
            return (next(revisions) + "\n").encode() if "rev-parse" in arguments else b""
        with patch("kenggo.tracking.subprocess.check_output", side_effect=git):
            return jobs.local_worker(config_path, config_pin, provenance_path, provenance_pin, self.output)

    def test_worker_checks_pins_and_records_validation_failure(self):
        Path(self.config["inputs"]["data"]["path"]).write_text("changed")
        with patch.object(jobs, "prepare", side_effect=AssertionError("Must not prepare")):
            with self.assertRaisesRegex(ValueError, "input differs"):
                self.invoke()
        self.assertEqual(json.loads((self.output / "failure.json").read_text())["status"], "failed")

    def test_worker_rejects_mismatched_provenance(self):
        self.provenance["source"] = {**SOURCE, "commit": "c" * 40}
        with patch.object(jobs, "prepare", side_effect=AssertionError("Must not prepare")):
            with self.assertRaisesRegex(ValueError, "pinned provenance"):
                self.invoke()
        self.assertTrue((self.output / "failure.json").is_file())

    def test_worker_rechecks_source_after_completion_and_marks_summary_failed(self):
        def prepare(config, output):
            summary = {"status": "completed", "fixture": True}
            save_json(output / "summary.json", summary)
            return summary
        with patch.object(jobs, "prepare", side_effect=prepare):
            with self.assertRaisesRegex(ValueError, "source changed"):
                self.invoke(revisions=[SOURCE["commit"], "c" * 40])
        self.assertEqual(json.loads((self.output / "summary.json").read_text())["status"], "failed")
        self.assertTrue((self.output / "failure.json").is_file())

    def test_worker_rechecks_input_pins_after_completion(self):
        def prepare(config, output):
            Path(config["inputs"]["rules"]["path"]).write_text("changed during preparation")
            return {"status": "completed"}
        with patch.object(jobs, "prepare", side_effect=prepare):
            with self.assertRaisesRegex(ValueError, "input differs"):
                self.invoke()
        self.assertTrue((self.output / "failure.json").is_file())

    def test_worker_keeps_previous_output_on_reentry(self):
        save_json(self.output / "summary.json", {"status": "completed"})
        before = (self.output / "summary.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "fresh parent-created"):
            self.invoke()
        self.assertEqual((self.output / "summary.json").read_bytes(), before)
        self.assertFalse((self.output / "failure.json").exists())


if __name__ == "__main__":
    unittest.main()
