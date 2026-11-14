"""Synthetic checks of frozen match denominators and uncertainty units."""

import copy
import json
import math
import unittest

import numpy as np

from kenggo.match_statistics import (clopper_pearson_lower, clopper_pearson_upper,
                                     human_score_interval, reliability_bounds, summarize)


def engine_schedule(profiles=(("p", "engine", 1., 20),), arms=("candidate", "control")):
    schedule = {"version": "match_schedule_v1", "purpose": "validation", "scope": "full",
                "arms": list(arms), "control_arm": "control" if "control" in arms else None,
                "profiles": [], "slots": []}
    for profile_index, (profile_id, family, weight, count) in enumerate(profiles):
        schedule["profiles"].append({"id": profile_id, "family": family, "weight": weight,
                                     "context_id": "untimed"})
        for block_index in range(count):
            for arm in arms:
                for color_index, color in enumerate(("B", "W")):
                    stream = 10000 * profile_index + 2 * block_index + color_index
                    schedule["slots"].append({"id": f"{profile_id}:{block_index}:{arm}:{color}",
                                              "block_id": f"{block_index:04d}", "profile_id": profile_id,
                                              "arm": arm, "candidate_color": color,
                                              "candidate_seed": 2 * stream, "opponent_seed": 2 * stream + 1})
    return schedule


def records_for(schedule, result="W", failure=False):
    return {slot["id"]: {"result": result, "reason": "synthetic", "behavior": {"failure": failure}}
            for slot in schedule["slots"]}


def human_schedule(counts):
    return {"version": "human_match_schedule_v1", "cohort_id": "declared-cohort",
            "slots": [{"id": f"{opponent}:{block}:{color}", "opponent_id": opponent,
                       "block_id": str(block), "candidate_color": color}
                      for opponent, blocks in counts.items() for block in range(blocks) for color in ("B", "W")]}


class ScoreTests(unittest.TestCase):
    def test_rounded_weights_are_normalized_for_scores_and_rate_bounds(self):
        schedule = engine_schedule(profiles=(("p", "engine", 1.0000000005, 20),), arms=("candidate",))
        records = records_for(schedule, failure=True)
        scores = summarize(schedule, records, bootstrap_seed=3)
        self.assertEqual(scores["profile_weights"], {"p": 1.0})
        rates = reliability_bounds(schedule, records, flag="failure", candidate_arm="candidate")
        self.assertEqual(rates["absolute_upper"], 1.0)

    def test_missing_and_unresolved_games_remain_in_every_denominator(self):
        schedule = engine_schedule(profiles=(("p", "engine", 1., 2),), arms=("candidate",))
        records = {"p:0:candidate:B": {"result": "W", "reason": "score"},
                   "p:0:candidate:W": {"result": "L", "reason": "forfeit"},
                   "p:1:candidate:B": {"result": "U", "reason": "cap"}}
        result = summarize(schedule, records, bootstrap_seed=3)
        arm = result["arms"]["candidate"]
        self.assertEqual(arm["counts"], {"N": 4, "W": 1, "L": 1, "D": 0, "U": 2, "missing": 1})
        self.assertEqual(arm["score_bounds"], [.25, .75])
        self.assertEqual(arm["resolved_score"], .5)
        self.assertEqual(arm["reasons"], {"score": 1, "forfeit": 1, "cap": 1, "missing_record": 1})
        self.assertEqual(arm["by_color"]["B"]["score_bounds"], [.5, 1])
        self.assertEqual(arm["by_color"]["W"]["score_bounds"], [0, .5])
        self.assertEqual(arm["by_profile"]["p"]["counts"], arm["counts"])
        self.assertEqual(arm["by_family"]["engine"]["counts"], arm["counts"])
        self.assertTrue(result["exploratory"])
        json.dumps(result, allow_nan=False)

    def test_fixed_profile_weights_survive_unequal_counts_and_missingness(self):
        schedule = engine_schedule(profiles=(("small", "same", .75, 1), ("large", "same", .25, 3)),
                                   arms=("candidate",))
        records = {slot["id"]: {"result": "W" if slot["profile_id"] == "small" else "L"}
                   for slot in schedule["slots"]}
        arm = summarize(schedule, records, bootstrap_seed=4)["arms"]["candidate"]
        self.assertEqual(arm["score_bounds"], [.75, .75])
        self.assertEqual(arm["resolved_score"], .75)
        self.assertEqual(arm["by_family"]["same"]["score_bounds"], [.75, .75])
        records = {key: value for key, value in records.items() if not key.startswith("small:")}
        arm = summarize(schedule, records, bootstrap_seed=4)["arms"]["candidate"]
        self.assertEqual(arm["score_bounds"], [0, .75])
        self.assertEqual(arm["resolved_score"], 0)
        self.assertIsNone(arm["by_profile"]["small"]["resolved_score"])

    def test_family_scores_use_conditional_frozen_weights(self):
        schedule = engine_schedule(profiles=(("p1", "first", .1, 2), ("p2", "first", .3, 2),
                                             ("p3", "second", .6, 2)), arms=("candidate",))
        records = {slot["id"]: {"result": "W" if slot["profile_id"] == "p1" else "L"}
                   for slot in schedule["slots"]}
        arm = summarize(schedule, records, bootstrap_seed=4)["arms"]["candidate"]
        self.assertAlmostEqual(arm["score_bounds"][0], .1)
        self.assertAlmostEqual(arm["by_family"]["first"]["score_bounds"][0], .25)
        self.assertEqual(arm["by_family"]["second"]["score_bounds"], [0, 0])

    def test_bootstrap_resamples_complete_pairs_with_shared_arm_indices(self):
        schedule = engine_schedule(profiles=(("p", "engine", 1., 4),))
        candidate = [("W", "L"), ("W", "W"), ("L", "L"), ("U", "W")]
        control = [("L", "L"), ("W", "L"), ("L", "W"), ("W", "U")]
        records = {slot["id"]: {"result": (candidate if slot["arm"] == "candidate" else control)
                                [int(slot["block_id"])][("B", "W").index(slot["candidate_color"])]}
                   for slot in schedule["slots"]}
        result = summarize(schedule, records, bootstrap_seed=42)
        lower = np.array([.5, 1, 0, .5]) - np.array([0, .5, .5, 1])
        upper = np.array([.5, 1, 0, 1]) - np.array([0, .5, .5, .5])
        indices = np.random.default_rng(42).integers(4, size=(10000, 4))
        expected = [np.quantile(lower[indices].mean(axis=1), .025),
                    np.quantile(upper[indices].mean(axis=1), .975)]
        np.testing.assert_allclose(result["comparisons"]["candidate"]["interval"], expected)
        self.assertEqual(result["comparisons"]["candidate"]["gain_bounds"], [float(lower.mean()), float(upper.mean())])
        self.assertEqual(result["comparisons"]["candidate"]["interval_methods"], ["paired_block_bootstrap"] * 2)

    def test_hoeffding_fallback_uses_independent_block_weights_and_range(self):
        schedule = engine_schedule(profiles=(("p1", "one", .75, 20), ("p2", "two", .25, 40)))
        records = records_for(schedule)
        for slot in schedule["slots"]:
            if slot["arm"] == "control":
                records[slot["id"]]["result"] = "L"
        result = summarize(schedule, records, bootstrap_seed=5)
        squared_weights = .75 ** 2 / 20 + .25 ** 2 / 40
        half_width = math.sqrt(math.log(40) * squared_weights / 2)
        self.assertEqual(result["weight_squares"], squared_weights)
        self.assertEqual(result["arms"]["candidate"]["interval"], [1 - half_width, 1])
        self.assertEqual(result["arms"]["control"]["interval"], [0, half_width])
        self.assertEqual(result["comparisons"]["candidate"]["interval"], [1 - 2 * half_width, 1])
        self.assertEqual(result["arms"]["candidate"]["interval_methods"], ["hoeffding"] * 2)
        self.assertFalse(result["exploratory"])

    def test_zero_variance_endpoint_falls_back_independently(self):
        schedule = engine_schedule(profiles=(("p", "engine", 1., 20),), arms=("candidate",))
        records = {slot["id"]: {"result": "L" if int(slot["block_id"]) % 2 else "U"}
                   for slot in schedule["slots"]}
        result = summarize(schedule, records, bootstrap_seed=10)["arms"]["candidate"]
        self.assertEqual(result["interval_methods"], ["hoeffding", "paired_block_bootstrap"])
        self.assertEqual(result["interval"][0], 0)
        self.assertGreater(result["interval"][1], .5)

    def test_no_results_is_unknown_and_keeps_full_interval(self):
        schedule = engine_schedule()
        result = summarize(schedule, {}, bootstrap_seed=1)
        for arm in result["arms"].values():
            self.assertEqual(arm["score_bounds"], [0, 1])
            self.assertEqual(arm["interval"], [0, 1])
            self.assertIsNone(arm["resolved_score"])
        self.assertEqual(result["comparisons"]["candidate"]["gain_bounds"], [-1, 1])
        self.assertEqual(result["comparisons"]["candidate"]["interval"], [-1, 1])

    def test_seed_is_reproducible_and_schedule_order_does_not_matter(self):
        schedule = engine_schedule()
        records = {slot["id"]: {"result": "W" if int(slot["block_id"]) % 3 else "L"}
                   for slot in schedule["slots"]}
        first = summarize(schedule, records, bootstrap_seed=87)
        schedule["slots"].reverse()
        schedule["arms"].reverse()
        self.assertEqual(first, summarize(schedule, records, bootstrap_seed=87))

    def test_unexpected_records_draws_and_bad_flags_are_rejected(self):
        schedule = engine_schedule()
        for record in ({"unexpected": {"result": "W"}},
                       {schedule["slots"][0]["id"]: {"result": "D"}},
                       {schedule["slots"][0]["id"]: {"behavior": {"failure": 0}}}):
            with self.subTest(record=record), self.assertRaises(ValueError):
                summarize(schedule, record, bootstrap_seed=0)

    def test_incomplete_schedule_duplicate_streams_and_unpaired_arms_are_rejected(self):
        for mutate in (lambda schedule: schedule["slots"].pop(),
                       lambda schedule: schedule["profiles"][0].update(weight=.9),
                       lambda schedule: schedule["slots"][0].update(candidate_seed=111),
                       lambda schedule: schedule["slots"][0].update(context_id="changed")):
            schedule = engine_schedule()
            mutate(schedule)
            with self.assertRaises(ValueError):
                summarize(schedule, {}, bootstrap_seed=1)
        schedule = engine_schedule()
        for slot in schedule["slots"]:
            if slot["block_id"] == "0001" and slot["candidate_color"] == "B":
                slot.update(candidate_seed=0, opponent_seed=1)
        with self.assertRaisesRegex(ValueError, "duplicates"):
            summarize(schedule, {}, bootstrap_seed=1)

    def test_distinct_agent_streams_and_valid_normalized_weights_required(self):
        mutations = [lambda s: s["slots"][0].update(opponent_seed=0),
                     lambda s: s["profiles"][0].update(weight=math.nan),
                     lambda s: s["profiles"][0].update(weight=math.inf),
                     lambda s: s["profiles"][0].update(weight=0),
                     lambda s: s["slots"].append(copy.deepcopy(s["slots"][0]))]
        for mutate in mutations:
            schedule = engine_schedule()
            mutate(schedule)
            with self.assertRaises(ValueError):
                summarize(schedule, {}, bootstrap_seed=0)

    def test_resample_count_resolves_adjusted_tails(self):
        schedule = engine_schedule()
        for settings in ({"resamples": 9999}, {"alpha": .01}, {"alpha": 0},
                         {"alpha": math.nan}, {"bootstrap_seed": -1}):
            arguments = {"bootstrap_seed": 0} | settings
            with self.assertRaises(ValueError):
                summarize(schedule, {}, **arguments)
        result = summarize(schedule, {}, bootstrap_seed=0, resamples=20000, alpha=.01)
        self.assertEqual(result["alpha"], .01)


class ExactReliabilityTests(unittest.TestCase):
    def test_protocol_rare_nonzero_failure_example(self):
        value = clopper_pearson_upper(1, 528, alpha=.005)
        self.assertAlmostEqual(value, .01399, places=5)
        self.assertGreater(value, .01)
        self.assertAlmostEqual((1 - value) ** 528 + 528 * value * (1 - value) ** 527, .005, places=13)

    def test_zero_all_and_no_trial_edges(self):
        self.assertEqual(clopper_pearson_upper(0, 0), 1)
        self.assertEqual(clopper_pearson_lower(0, 0), 0)
        self.assertEqual(clopper_pearson_lower(0, 528, alpha=.005), 0)
        self.assertEqual(clopper_pearson_upper(528, 528, alpha=.005), 1)
        self.assertAlmostEqual(clopper_pearson_upper(0, 528, alpha=.005), 1 - .005 ** (1 / 528), places=15)
        self.assertAlmostEqual(clopper_pearson_lower(528, 528, alpha=.005), .005 ** (1 / 528), places=15)

    def test_exact_bounds_satisfy_independent_binomial_sums(self):
        for n, x, alpha in ((10, 1, .05), (20, 9, .005), (21, 19, .025), (100, 40, 1e-8)):
            upper = clopper_pearson_upper(x, n, alpha=alpha)
            lower = clopper_pearson_lower(x, n, alpha=alpha)
            upper_tail = math.fsum(math.comb(n, k) * upper ** k * (1 - upper) ** (n - k) for k in range(x + 1))
            lower_tail = math.fsum(math.comb(n, k) * lower ** k * (1 - lower) ** (n - k) for k in range(x, n + 1))
            self.assertAlmostEqual(upper_tail, alpha, delta=1e-12 * alpha)
            self.assertAlmostEqual(lower_tail, alpha, delta=1e-12 * alpha)

    def test_invalid_binomial_inputs_are_rejected(self):
        for x, n, alpha in ((-1, 3, .05), (4, 3, .05), (0, -1, .05), (True, 3, .05),
                            (1., 3, .05), (0, 3, 0), (0, 3, 1), (0, 3, math.nan)):
            for bound in (clopper_pearson_lower, clopper_pearson_upper):
                with self.subTest(x=x, n=n, alpha=alpha), self.assertRaises(ValueError):
                    bound(x, n, alpha=alpha)

    def test_absolute_bounds_keep_profile_color_strata_and_fixed_weights(self):
        schedule = engine_schedule(profiles=(("p1", "same", .8, 20), ("p2", "same", .2, 50)))
        result = reliability_bounds(schedule, records_for(schedule), flag="failure", candidate_arm="candidate")
        self.assertEqual(result["marginal_alpha"], .05 / 4)
        self.assertEqual(result["marginal_bounds"], 4)
        expected = .8 * (1 - (.05 / 4) ** (1 / 20)) + .2 * (1 - (.05 / 4) ** (1 / 50))
        self.assertAlmostEqual(result["absolute_upper"], expected)
        self.assertEqual(result["strata"]["p1"]["by_color"]["B"]["candidate"]["trials"], 20)
        self.assertEqual(result["strata"]["p2"]["context_id"], "untimed")

    def test_nonzero_reliability_uses_exact_bounds(self):
        schedule = engine_schedule(profiles=(("p", "engine", 1., 528),), arms=("candidate",))
        records = records_for(schedule)
        records["p:0:candidate:B"]["behavior"]["failure"] = True
        records["p:0:candidate:W"]["behavior"]["failure"] = True
        result = reliability_bounds(schedule, records, flag="failure", candidate_arm="candidate", alpha=.01)
        self.assertAlmostEqual(result["absolute_upper"], .01399, places=5)

    def test_regression_missing_and_unknown_are_adverse_only_for_candidate_upper(self):
        schedule = engine_schedule(profiles=(("p", "engine", 1., 3),))
        records = records_for(schedule)
        for arm in ("candidate", "control"):
            records[f"p:0:{arm}:B"]["behavior"]["failure"] = True
            records[f"p:1:{arm}:B"]["behavior"]["failure"] = None
            del records[f"p:2:{arm}:B"]
        result = reliability_bounds(schedule, records, flag="failure", candidate_arm="candidate", control_arm="control")
        self.assertEqual(result["marginal_alpha"], .05 / 4)
        black = result["strata"]["p"]["by_color"]["B"]
        self.assertEqual(black["candidate"]["trials"], 3)
        self.assertEqual(black["candidate"]["failures"], 1)
        self.assertEqual(black["candidate"]["unknown"], 1)
        self.assertEqual(black["candidate"]["missing"], 1)
        self.assertEqual(black["candidate"]["adverse"], 3)
        self.assertEqual(black["candidate"]["upper"], 1)
        self.assertAlmostEqual(black["control"]["lower"], 1 - (1 - .0125) ** (1 / 3))
        expected = (1 - black["control"]["lower"] + clopper_pearson_upper(0, 3, alpha=.0125)) / 2
        self.assertAlmostEqual(result["regression_upper"], expected)

    def test_absent_flag_remains_unknown_and_complete_missingness_cannot_pass(self):
        schedule = engine_schedule()
        result = reliability_bounds(schedule, records_for(schedule), flag="not_judged", candidate_arm="candidate")
        self.assertEqual(result["absolute_upper"], 1)
        self.assertEqual(result["strata"]["p"]["by_color"]["W"]["candidate"]["unknown"], 20)
        result = reliability_bounds(schedule, {}, flag="failure", candidate_arm="candidate", control_arm="control")
        self.assertEqual(result["regression_upper"], 1)
        self.assertEqual(result["strata"]["p"]["by_color"]["W"]["control"]["lower"], 0)


class HumanScoreTests(unittest.TestCase):
    def test_repeated_opponents_receive_equal_weights_and_retain_missing_games(self):
        schedule = human_schedule({"short": 1, "repeated": 3})
        records = {slot["id"]: {"result": "W" if slot["opponent_id"] == "short" else "L"}
                   for slot in schedule["slots"]}
        result = human_score_interval(schedule, records, bootstrap_seed=11)
        self.assertEqual(result["score_bounds"], [.5, .5])
        self.assertEqual(result["counts"]["N"], 8)
        self.assertEqual(result["opponents"], 2)
        self.assertEqual(result["interval"], [0, 1])
        del records["short:0:W"]
        result = human_score_interval(schedule, records, bootstrap_seed=11)
        self.assertEqual(result["score_bounds"], [.25, .5])
        self.assertAlmostEqual(result["resolved_score"], 1 / 3)
        self.assertEqual(result["counts"]["missing"], 1)
        self.assertEqual(result["by_opponent"]["short"]["counts"]["U"], 1)

    def test_cluster_resampling_matches_whole_opponent_oracle(self):
        schedule = human_schedule({"a": 2, "b": 2, "c": 2, "d": 2})
        outcomes = {"a": ("W", "W", "W", "W"), "b": ("W", "L", "W", "L"),
                    "c": ("L", "L", "L", "L"), "d": ("U", "W", "U", "W")}
        records = {slot["id"]: {"result": outcomes[slot["opponent_id"]]
                                [2 * int(slot["block_id"]) + (slot["candidate_color"] == "W")]}
                   for slot in schedule["slots"]}
        result = human_score_interval(schedule, records, bootstrap_seed=19)
        values = np.array([[1, 1], [.5, .5], [0, 0], [.5, 1]])
        indices = np.random.default_rng(19).integers(4, size=(10000, 4))
        samples = values[indices].mean(axis=1)
        np.testing.assert_allclose(result["interval"], [np.quantile(samples[:, 0], .025), np.quantile(samples[:, 1], .975)])
        self.assertEqual(result["interval_methods"], ["opponent_cluster_bootstrap"] * 2)

    def test_constant_human_scores_use_opponent_count_for_hoeffding(self):
        schedule = human_schedule({str(i): 4 for i in range(20)})
        result = human_score_interval(schedule, records_for(schedule), bootstrap_seed=0)
        half_width = math.sqrt(math.log(40) / (2 * 20))
        self.assertEqual(result["interval"], [1 - half_width, 1])
        self.assertEqual(result["weight_squares"], 1 / 20)
        self.assertEqual(result["interval_methods"], ["hoeffding"] * 2)

    def test_human_and_engine_schedules_cannot_be_confused_or_mixed(self):
        human = human_schedule({"one": 2})
        with self.assertRaises(ValueError):
            reliability_bounds(human, {}, flag="failure", candidate_arm="candidate")
        with self.assertRaises(ValueError):
            human_score_interval(engine_schedule(), {}, bootstrap_seed=0)
        human["slots"][0]["cohort_id"] = "another-cohort"
        with self.assertRaises(ValueError):
            human_score_interval(human, {}, bootstrap_seed=0)


if __name__ == "__main__":
    unittest.main()
