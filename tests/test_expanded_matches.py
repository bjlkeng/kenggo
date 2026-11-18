import copy
from dataclasses import asdict
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from kenggo import baseline_matches as pilot
from kenggo.matches import AgentProfile, ExperimentBudget, validate_schedule

import test_baseline_matches as match_fixtures


EXPANDED_SEED = 20261118


class ExpandedMatchTests(unittest.TestCase):
    def setUp(self):
        self.fixture = match_fixtures.BaselineMatchTests()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()

    def training(self, arm):
        return json.loads(Path(self.fixture.refs[arm + "_config"]["path"]).read_text())

    def save_training(self, arm, training):
        self.fixture.save_ref(arm + "_config", training)
        self.fixture.save_ref(arm + "_provenance", {
            "version": "experiment_provenance_v1", "inputs": training["inputs"]})
        artifacts = self.fixture.artifacts[arm]
        artifacts["config_sha256"] = self.fixture.refs[arm + "_config"]["sha256"]
        artifacts["provenance_sha256"] = self.fixture.refs[arm + "_provenance"]["sha256"]

    def expanded(self, komi=(13,)):
        self.fixture.config["seed"] = EXPANDED_SEED
        self.fixture.config["parameters"].update(
            recipe="expanded_e2_matches_v1", bootstrap_seed=EXPANDED_SEED)
        for arm in pilot.ARMS[:2]:
            training = self.training(arm)
            training["seed"] = EXPANDED_SEED
            training["parameters"].update(recipe="expanded_e2_training_v1",
                inference_contract="komi_comparison_v1", training_komi_halfpoints=list(komi))
            self.save_training(arm, training)

    def test_each_training_cohort_accepts_its_matched_learned_bundles(self):
        for komi in ((13,), (1, 13)):
            with self.subTest(komi=komi):
                self.expanded(komi)
                learned, _, _, _, limits, budget = self.fixture.validate()
                self.assertEqual(set(learned), {"conditioned", "unconditioned"})
                self.assertEqual(learned["conditioned"].examples_sha256,
                                 learned["unconditioned"].examples_sha256)
                self.assertEqual(learned["conditioned"].normalizer_sha256,
                                 learned["unconditioned"].normalizer_sha256)
                self.assertEqual(asdict(limits), self.fixture.config["parameters"]["match_limits"])
                self.assertEqual(asdict(budget), self.fixture.config["parameters"]["runner_budget"])

    def test_expanded_matches_require_the_explicit_recipe_and_frozen_seeds(self):
        self.expanded()
        for field, value in (("recipe", None), ("recipe", "expanded_e2_matches_v2"),
                             ("bootstrap_seed", pilot.SEED), ("bootstrap_seed", 1)):
            config = copy.deepcopy(self.fixture.config)
            if value is None:
                config["parameters"].pop(field)
            else:
                config["parameters"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.fixture.validate(config)
        for seed in (pilot.SEED, 1):
            config = copy.deepcopy(self.fixture.config)
            config["seed"] = seed
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                self.fixture.validate(config)

    def test_old_training_cannot_be_relabelled_as_an_expanded_panel(self):
        self.fixture.config["seed"] = EXPANDED_SEED
        self.fixture.config["parameters"].update(
            recipe="expanded_e2_matches_v1", bootstrap_seed=EXPANDED_SEED)
        with self.assertRaises(ValueError):
            self.fixture.validate()
        for arm in pilot.ARMS[:2]:
            training = self.training(arm)
            training["seed"] = EXPANDED_SEED
            self.save_training(arm, training)
        with self.assertRaises(ValueError):
            self.fixture.validate()

    def test_expanded_training_contract_rejects_missing_or_unsupported_scope(self):
        self.expanded()
        original = self.training("conditioned")
        cases = [("recipe", None), ("recipe", "base_training_v1"),
                 ("inference_contract", None), ("inference_contract", "base_v1")]
        cases += [("training_komi_halfpoints", value) for value in
                  (None, [], [1], [13, 1], [13.0], [True, 13], [1, 13, 13], 13)]
        for field, value in cases:
            training = copy.deepcopy(original)
            if value is None:
                training["parameters"].pop(field)
            else:
                training["parameters"][field] = value
            self.save_training("conditioned", training)
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.fixture.validate()

    def test_both_learned_arms_must_share_the_training_scope(self):
        self.expanded()
        training = self.training("unconditioned")
        training["parameters"]["training_komi_halfpoints"] = [1, 13]
        self.save_training("unconditioned", training)
        with self.assertRaises(ValueError):
            self.fixture.validate()

    def test_training_inputs_must_bind_the_panel_cohort(self):
        self.expanded((1, 13))
        self.fixture.save_ref("other_examples", {"fixture_only": True, "cohort": "other"})
        training = self.training("unconditioned")
        training["inputs"]["examples"] = self.fixture.refs["other_examples"]
        self.save_training("unconditioned", training)
        with self.assertRaisesRegex(ValueError, "input binding: examples"):
            self.fixture.validate()

    def test_new_seed_reaches_schedule_and_old_schedule_default_is_preserved(self):
        self.expanded((1, 13))
        schedule, _, _, _ = self.fixture.components()
        self.assertEqual(schedule["seed"], EXPANDED_SEED)
        self.assertEqual(schedule, self.fixture.components()[0])
        candidates = tuple(AgentProfile.from_dict(profile) for profile in schedule["arm_profiles"])
        arguments = {"provenance": self.fixture.provenance(),
                     "referee_profile": schedule["referee_profile"]}
        legacy = pilot.build_schedule(candidates, **arguments)
        self.assertEqual(legacy, pilot.build_schedule(candidates, seed=pilot.SEED, **arguments))
        self.assertEqual(legacy["seed"], 20261114)
        self.assertNotEqual(schedule["slots"], legacy["slots"])
        self.assertEqual(len(schedule["slots"]), 12)
        for opponent in schedule["profiles"]:
            slots = [slot for slot in schedule["slots"] if slot["profile_id"] == opponent["id"]]
            self.assertEqual(len({(slot["candidate_seed"], slot["opponent_seed"]) for slot in slots}), 1)

    def test_combined_training_keeps_games_and_profiles_at_six_point_five(self):
        self.expanded((1, 13))
        schedule, _, _, _ = self.fixture.components()
        self.assertEqual(schedule["komi_halfpoints"], 13)
        self.assertEqual(schedule["scope"], "setup")
        self.assertEqual(schedule["context"]["opponent_rank"], "unknown")
        for profile in schedule["profiles"] + schedule["arm_profiles"]:
            self.assertEqual(profile["komi_halfpoints"], 13)
            self.assertIsNone(profile["human_rank"])
        schedule["komi_halfpoints"] = 1
        with self.assertRaises(ValueError):
            validate_schedule(schedule, for_execution=True)

    def test_synthetic_games_save_expanded_schedule_and_bootstrap_seeds(self):
        self.expanded((1, 13))
        result, output, _ = self.fixture.synthetic_run()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["records_replayed"], 12)
        self.assertEqual(result["seed"], EXPANDED_SEED)
        self.assertFalse(result["qualification_claim"])
        self.assertFalse(result["human_rank_claim"])
        scores = json.loads((output / "scores.json").read_text())
        schedule = json.loads((output / "schedule.json").read_text())
        self.assertEqual(scores["bootstrap_seed"], EXPANDED_SEED)
        self.assertEqual(scores["resamples"], 10000)
        self.assertEqual(schedule["seed"], EXPANDED_SEED)
        for path in (output / "matches").glob("*.sgf"):
            self.assertIn("KM[6.5]", path.read_text())

    def test_exhausted_expanded_budget_retains_unfilled_slots(self):
        self.expanded()
        self.fixture.config["parameters"]["runner_budget"] = asdict(ExperimentBudget(1, .1, 1024**2))
        with patch.object(pilot.RefereeFactory, "__call__", side_effect=AssertionError("No game should start")):
            result, _ = self.fixture.invoke()
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["records_replayed"], 0)
        self.assertEqual(len(result["missing_slots"]), 12)
        for counts in result["counts"].values():
            self.assertEqual(counts, {"N": 4, "W": 0, "L": 0, "D": 0, "U": 4, "missing": 4})

    def test_original_pilot_configuration_still_validates(self):
        self.assertNotIn("recipe", self.fixture.config["parameters"])
        self.fixture.validate()
        self.assertEqual(self.fixture.components()[0]["seed"], 20261114)


if __name__ == "__main__":
    unittest.main()
