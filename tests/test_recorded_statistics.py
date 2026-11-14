"""Synthetic oracles for rank/side weights, raw policy losses and dyadic CR0."""

from dataclasses import replace
import json
import math
from statistics import NormalDist
import unittest

from kenggo.recorded_statistics import (Prediction, evaluate, paired_report, prediction_from_logits,
                                       prediction_from_raw_policy)


LEGAL = (True, True) + (False,) * 360


def row(game, mover_account, opponent_account, *, rank="1k", mover="B", ply=0, value=1.,
        value_valid=True, clock=(600., 30., 5)):
    return {"row_key": f"{game}:{ply}", "source_url": game, "mover": mover,
            "group_mover": mover_account, "group_opponent": opponent_account,
            "source_rank_mover": rank, "source_rank_opponent": rank, "ply": ply,
            "nominal_clock": clock, "legal": LEGAL, "action": 0, "policy_valid": True,
            "value": value, "value_valid": value_valid}


def prediction(source, loss, value=0.):
    complement = math.log(-math.expm1(-loss)) if loss > 0 else -math.inf
    return Prediction(source["row_key"], (-loss, complement) + (-math.inf,) * 360, value)


def evaluate_losses(rows, losses, *, ranks=("1k",), values=None):
    values = [0.] * len(rows) if values is None else values
    return evaluate(rows, [prediction(source, loss, value) for source, loss, value in zip(rows, losses, values)],
                    aggregate_ranks=ranks)


def metric(report, name="nll"):
    return report["overall"]["metrics"][name]


def finite(number):
    if number["status"] != "finite":
        raise AssertionError(number)
    return number["value"]


class DyadicVarianceTests(unittest.TestCase):
    def test_repeated_unordered_pair_is_one_subtraction_cluster(self):
        rows = [row(str(i), *pair) for i, pair in enumerate((("a", "b"), ("b", "a"), ("a", "c"), ("d", "e")))]
        result = evaluate_losses(rows, [2., 1.5, .5, 0.])
        uncertainty = metric(result)["uncertainty"]
        self.assertEqual([finite(value) for value in uncertainty["game_influences"].values()], [.25, .125, -.125, -.25])
        self.assertEqual(finite(uncertainty["variance"]), 1 / 8)
        self.assertNotEqual(finite(uncertainty["variance"]), 3 / 16)
        self.assertEqual(uncertainty["status"], "inconclusive")

        swapped = [{**source, "mover": "W"} for source in rows]
        swapped_uncertainty = metric(evaluate_losses(swapped, [2., 1.5, .5, 0.]))["uncertainty"]
        self.assertEqual(uncertainty, swapped_uncertainty)
        self.assertEqual(result, evaluate_losses(rows[::-1], [0., .5, 1.5, 2.]))

    def test_negative_variance_remains_negative_and_inconclusive(self):
        rows = [row(str(i), *pair) for i, pair in enumerate((("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")))]
        uncertainty = metric(evaluate_losses(rows, [0., 1.5, 1.5, 1.]))["uncertainty"]
        self.assertEqual([finite(value) for value in uncertainty["game_influences"].values()], [-.25, .125, .125, 0.])
        self.assertEqual(finite(uncertainty["variance"]), -1 / 32)
        self.assertIn("nonpositive_variance", uncertainty["reasons"])
        self.assertIsNone(uncertainty["interval95"])

    def test_rank_balance_uses_global_centered_weighted_intercept(self):
        rows = [row(str(i), f"m{i}", f"o{i}", rank="1k" if i < 2 else "2k") for i in range(5)]
        result = metric(evaluate_losses(rows, [1., 3., 4., 6., 8.], ranks=("1k", "2k")))
        self.assertEqual(finite(result["estimate"]), 4)
        self.assertEqual([side["weight"] for side in result["side_weights"]], [.25, .25, 1 / 6, 1 / 6, 1 / 6])
        self.assertAlmostEqual(finite(result["uncertainty"]["variance"]), 85 / 72)
        self.assertNotAlmostEqual(finite(result["uncertainty"]["variance"]), 25 / 72)

    def test_two_sides_are_clustered_as_one_game(self):
        rows = [row("one", "a", "b"), row("one", "b", "a", mover="W", ply=1), row("two", "c", "d")]
        result = metric(evaluate_losses(rows, [0., 2., 1.]))
        uncertainty = result["uncertainty"]
        self.assertEqual(result["counts"]["game_sides"], 3)
        self.assertEqual(result["counts"]["games"], 2)
        self.assertEqual(finite(uncertainty["variance"]), 0)
        self.assertEqual(uncertainty["incident_account_weights"], {"a": 2 / 3, "b": 2 / 3, "c": 1 / 3, "d": 1 / 3})
        self.assertEqual(uncertainty["mover_contribution_weights"], {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3})

    def test_positive_variance_and_at_least_30_accounts_allow_normal_interval(self):
        rows = [row(str(i), f"m{i}", f"o{i}") for i in range(15)]
        result = metric(evaluate_losses(rows, list(range(1, 16))))
        uncertainty = result["uncertainty"]
        variance = sum((loss - 8) ** 2 / 15 ** 2 for loss in range(1, 16))
        radius = NormalDist().inv_cdf(.975) * math.sqrt(variance)
        self.assertEqual(uncertainty["status"], "available")
        self.assertAlmostEqual(finite(uncertainty["variance"]), variance)
        self.assertAlmostEqual(uncertainty["interval95"][0], 8 - radius)
        self.assertAlmostEqual(uncertainty["interval95"][1], 8 + radius)
        self.assertTrue(result["counts"]["insufficient"])
        small = metric(evaluate_losses(rows[:-1], list(range(1, 15))))["uncertainty"]
        self.assertIn("fewer_than_30_accounts", small["reasons"])

    def test_opponent_incident_weight_is_checked_not_only_mover_weight(self):
        rows = [row(str(i), f"m{i}", "hub" if i < 30 else f"o{i}") for i in range(36)]
        uncertainty = metric(evaluate_losses(rows, [0.] * 30 + [1., 2., 3., 4., 5., 6.]))["uncertainty"]
        self.assertGreaterEqual(uncertainty["accounts"], 30)
        self.assertAlmostEqual(uncertainty["max_incident_account_weight"], 30 / 36)
        self.assertAlmostEqual(uncertainty["max_mover_contribution_weight"], 1 / 36)
        self.assertGreater(finite(uncertainty["variance"]), 0)
        self.assertEqual(uncertainty["reasons"], ["incident_account_weight_over_10_percent"])

    def test_zero_variance_never_becomes_certainty_or_an_invariant(self):
        rows = [row(str(i), f"m{i}", f"o{i}") for i in range(30)]
        result = evaluate_losses(rows, [2.] * len(rows))
        uncertainty = metric(result)["uncertainty"]
        self.assertEqual(finite(uncertainty["variance"]), 0)
        self.assertEqual(uncertainty["reasons"], ["nonpositive_variance"])
        predictions = [prediction(source, 2.) for source in rows]
        paired = metric(paired_report(rows, predictions, predictions, aggregate_ranks=("1k",)))
        self.assertEqual(finite(paired["estimate"]), 0)
        self.assertIsNone(paired["uncertainty"]["interval95"])

    def test_constant_losses_with_different_side_lengths_have_zero_variance(self):
        rows = [row(str(game), f"m{game}", f"o{game}", ply=2 * move)
                for game in range(30) for move in range(5 + game % 3)]
        result = metric(evaluate_losses(rows, [.1] * len(rows)))
        self.assertEqual(finite(result["estimate"]), .1)
        self.assertEqual(finite(result["uncertainty"]["variance"]), 0)
        self.assertEqual(result["uncertainty"]["reasons"], ["nonpositive_variance"])

    def test_exact_ten_percent_incident_weight_is_allowed(self):
        rows = [row(str(i), f"m{i}", "shared" if i < 2 else f"o{i}") for i in range(20)]
        uncertainty = metric(evaluate_losses(rows, list(range(1, 21))))["uncertainty"]
        self.assertEqual(uncertainty["max_incident_account_weight"], .1)
        self.assertEqual(uncertainty["status"], "available")


class HierarchyTests(unittest.TestCase):
    def test_repeating_moves_within_one_side_preserves_primary(self):
        rows = [row("short", "a", "b"), row("long", "c", "d")]
        original = metric(evaluate_losses(rows, [3., 1.]))
        extended = [rows[0]] + [row("long", "c", "d", ply=2 * i) for i in range(10)]
        result = metric(evaluate_losses(extended, [3.] + [1.] * 10))
        self.assertEqual(finite(original["estimate"]), 2)
        self.assertEqual(result["estimate"], original["estimate"])
        self.assertEqual(result["uncertainty"], original["uncertainty"])
        self.assertAlmostEqual(finite(result["diagnostics"]["pooled_move"]), 13 / 11)

    def test_account_diagnostic_balances_mover_accounts_then_sides(self):
        rows = [row(str(i), "prolific" if i < 3 else "rare", f"o{i}") for i in range(4)]
        result = metric(evaluate_losses(rows, [1., 3., 5., 9.]))
        self.assertEqual(finite(result["estimate"]), 4.5)
        self.assertEqual(finite(result["diagnostics"]["account_balanced"]), 6)
        self.assertEqual(result["uncertainty"]["mover_contribution_weights"], {"prolific": .75, "rare": .25})

    def test_paired_differences_precede_hierarchy_and_keep_candidate_direction(self):
        rows = [row("one", "a", "b"), row("one", "a", "b", ply=2), row("two", "c", "d")]
        candidate = [prediction(source, loss, value) for source, loss, value in zip(rows, [1., 3., 6.], [1., -1., 0.])]
        control = [prediction(source, loss) for source, loss in zip(rows, [2., 1., 3.])]
        result = paired_report(rows, candidate, control, aggregate_ranks=("1k",))
        self.assertEqual(result["kind"], "candidate_minus_control")
        self.assertEqual(finite(metric(result)["estimate"]), 1.75)
        self.assertEqual(finite(metric(result, "value_mse")["estimate"]), .5)

    def test_missing_rank_keeps_scope_unavailable_and_diagnostics_explicit(self):
        rows = [row("one", "a", "b")]
        result = metric(evaluate_losses(rows, [2.], ranks=("1k", "2k")))
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["aggregate_ranks"], ["1k", "2k"])
        self.assertEqual(result["missing_ranks"], ["2k"])
        self.assertEqual(result["estimate"]["status"], "undefined")
        self.assertEqual(result["side_weights"], [])
        self.assertEqual(finite(result["diagnostics"]["pooled_move"]), 2)
        self.assertEqual(result["diagnostics"]["observed_ranks"], ["1k"])
        with self.assertRaises(ValueError):
            evaluate_losses(rows, [2.], ranks=("2k",))
        for ranks in ((), ("1k", "1k"), "1k"):
            with self.subTest(ranks=ranks), self.assertRaises(ValueError):
                evaluate_losses(rows, [2.], ranks=ranks)

    def test_empty_input_does_not_invent_a_score(self):
        result = evaluate([], [], aggregate_ranks=("1k",))
        self.assertEqual(result["expected_outputs"], 0)
        self.assertEqual(metric(result)["status"], "unavailable")
        self.assertEqual(metric(result)["diagnostics"]["pooled_move"]["status"], "undefined")
        json.dumps(result, allow_nan=False)

    def test_phase_clock_and_joint_counts_include_boundary_and_missing_rows(self):
        rows = [row(str(i), f"m{i}", f"o{i}", ply=ply, clock=clock) for i, (ply, clock) in enumerate(
            ((0, (600, 30, 5)), (59, (1199, 30, 5)), (60, (1200, 30, 5)),
             (179, (1800, 30, 5)), (180, None), (181, (599, 30, 5))))]
        result = evaluate_losses(rows, [1.] * 6)
        self.assertEqual([group["counts"]["moves"] for group in result["by_phase"].values()], [2, 2, 2])
        self.assertEqual([group["counts"]["moves"] for group in result["by_clock"].values()], [2, 2, 1, 1])
        self.assertEqual(sum(cell["counts"]["moves"] for cell in result["joint_cells"]), 6)
        self.assertTrue(all(cell["counts"]["insufficient"] for cell in result["joint_cells"]))
        self.assertEqual(len(result["coverage"]["by_nominal_clock"]), 6)
        self.assertEqual(result["coverage"]["by_rank_pair"][0]["counts"]["moves"], 6)

    def test_cell_support_uses_mover_accounts_and_game_sides(self):
        rows = [row(str(i), f"m{i % 30}", f"o{i}") for i in range(100)]
        result = evaluate_losses(rows, [1.] * 100)
        self.assertFalse(result["overall"]["counts"]["insufficient"])
        self.assertTrue(evaluate_losses(rows[:99], [1.] * 99)["overall"]["counts"]["insufficient"])
        concentrated = [{**source, "group_mover": f"m{i % 29}"} for i, source in enumerate(rows)]
        self.assertTrue(evaluate_losses(concentrated, [1.] * 100)["overall"]["counts"]["insufficient"])


class PolicyValidationTests(unittest.TestCase):
    def test_extreme_finite_logits_have_finite_loss_after_exp_underflow(self):
        source = row("one", "a", "b") | {"legal": (True,) * 3 + (False,) * 359, "action": 2}
        logits = [1000., 0., -1000.] + [0.] * 359
        output = prediction_from_logits(source["row_key"], logits, 0., source["legal"])
        self.assertEqual(math.exp(output.log_policy[2]), 0)
        self.assertEqual(output.log_policy[2], -2000)
        result = evaluate([source], [output], aggregate_ranks=("1k",))
        self.assertEqual(finite(metric(result)["estimate"]), 2000)
        self.assertEqual(finite(metric(result, "top1")["estimate"]), 0)

    def test_raw_rounding_is_not_renormalized(self):
        source = row("one", "a", "b")
        output = prediction_from_raw_policy(source["row_key"], [.2, .7998] + [-1.] * 360, 0., LEGAL)
        result = evaluate([source], [output], aggregate_ranks=("1k",))
        self.assertEqual(finite(metric(result)["estimate"]), -math.log(.2))
        self.assertAlmostEqual(math.fsum(math.exp(value) for value in output.log_policy), .9998)

    def test_raw_zero_and_infinite_contrasts_are_json_tagged(self):
        source = row("one", "a", "b")
        zero = prediction_from_raw_policy(source["row_key"], [0., 1.] + [-1.] * 360, 0., LEGAL)
        normal = prediction(source, 1.)
        result = evaluate([source], [zero], aggregate_ranks=("1k",))
        self.assertEqual(metric(result)["estimate"], {"status": "positive_infinity", "value": None})
        for candidate, control, status in ((zero, zero, "undefined"), (zero, normal, "positive_infinity"),
                                           (normal, zero, "negative_infinity")):
            with self.subTest(status=status):
                paired = paired_report([source], [candidate], [control], aggregate_ranks=("1k",))
                self.assertEqual(metric(paired)["estimate"]["status"], status)
                self.assertEqual(metric(paired)["uncertainty"]["status"], "inconclusive")
                json.dumps(paired, allow_nan=False)
        json.dumps(result, allow_nan=False)

    def test_missing_duplicate_extra_and_failed_outputs_fail_evaluation(self):
        source = row("one", "a", "b", value_valid=False)
        output = prediction(source, 1.)
        for outputs in ([], [output, output], [output, replace(output, row_key="extra")], [None]):
            with self.subTest(outputs=outputs), self.assertRaises(ValueError):
                evaluate([source], outputs, aggregate_ranks=("1k",))
        with self.assertRaises(ValueError):
            paired_report([source], [output], [], aggregate_ranks=("1k",))

    def test_complete_log_distribution_is_checked_even_for_masked_targets(self):
        source = row("one", "a", "b", value_valid=False) | {"policy_valid": False, "action": -1}
        for logs in ((-math.inf,) * 362, (0., -math.inf, 0.) + (-math.inf,) * 359,
                     (math.log(.2), math.log(.7)) + (-math.inf,) * 360):
            with self.subTest(logs=logs[:3]), self.assertRaises(ValueError):
                evaluate([source], [Prediction(source["row_key"], logs, 0.)], aggregate_ranks=("1k",))

    def test_invalid_log_probs_value_and_raw_policy_fail(self):
        valid_logs = (-math.log(2),) * 2 + (-math.inf,) * 360
        for value in (math.nan, math.inf, -math.inf, 1.01, -1.01, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Prediction("key", valid_logs, value)
        for first in (math.nan, math.inf, .01):
            with self.subTest(first=first), self.assertRaises(ValueError):
                Prediction("key", (first,) + valid_logs[1:], 0.)
        for probabilities in ([.2, .7] + [-1.] * 360, [.2, .8, 0.] + [-1.] * 359,
                              [math.nan, 1.] + [-1.] * 360, [-.1, 1.] + [-1.] * 360):
            with self.subTest(probabilities=probabilities[:3]), self.assertRaises(ValueError):
                prediction_from_raw_policy("key", probabilities, 0., LEGAL)
        for logits in ([math.nan, 0.] + [0.] * 360, [math.inf, 0.] + [0.] * 360,
                       [-math.inf, 0.] + [0.] * 360, [1e308, -1e308] + [0.] * 360):
            with self.subTest(logits=logits[:2]), self.assertRaises(ValueError):
                prediction_from_logits("key", logits, 0., LEGAL)

    def test_top1_ties_have_a_fixed_action_order(self):
        sources = [row("one", "a", "b"), row("two", "c", "d") | {"action": 1}]
        outputs = [prediction_from_logits(source["row_key"], [0.] * 362, 0., LEGAL) for source in sources]
        result = evaluate(sources, outputs, aggregate_ranks=("1k",))
        self.assertEqual(result["top1_ties"], "lowest_action_index")
        self.assertEqual(finite(metric(result, "top1")["estimate"]), .5)

    def test_inconsistent_game_metadata_and_duplicate_positions_fail(self):
        first = row("one", "a", "b")
        second = row("one", "a", "b", ply=2)
        for changed in (second | {"group_opponent": "c"}, second | {"nominal_clock": None},
                        second | {"source_rank_opponent": "2k"}, second | {"row_key": "different", "ply": 0}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                evaluate_losses([first, changed], [1., 1.])


class ValueCalibrationTests(unittest.TestCase):
    def test_all_fixed_bin_boundaries_and_plus_one(self):
        values = [-1., -.8, -.6, -.4, -.2, 0., .2, .4, .6, .8, 1.]
        rows = [row(str(i), f"m{i}", f"o{i}") for i in range(11)]
        report = evaluate_losses(rows, [1.] * 11, values=values)
        calibration = report["overall"]["value_calibration"]["model"]
        self.assertEqual([item["raw_count"] for item in calibration["bins"]], [1] * 9 + [2])
        self.assertAlmostEqual(sum(item["weighted_mass"] for item in calibration["bins"]), 1.)
        self.assertEqual(finite(calibration["bins"][-1]["mean_prediction"]), .9)
        self.assertTrue(calibration["bins"][-1]["upper_inclusive"])
        self.assertTrue(all(not item["upper_inclusive"] for item in calibration["bins"][:-1]))

    def test_bin_means_use_frozen_row_weights_and_show_raw_count(self):
        rows = [row("long", "a", "b", ply=2 * i) for i in range(3)]
        rows += [row("short", "c", "d", value=-1.), row("rank2", "e", "f", rank="2k")]
        report = evaluate_losses(rows, [1.] * 5, ranks=("1k", "2k"), values=[-1., -.8, 0., .8, 1.])
        calibration = report["overall"]["value_calibration"]["model"]
        last = calibration["bins"][-1]
        self.assertEqual(last["raw_count"], 2)
        self.assertEqual(last["weighted_mass"], .75)
        self.assertAlmostEqual(finite(last["mean_prediction"]), (.25 * .8 + .5) / .75)
        self.assertAlmostEqual(finite(last["mean_result"]), 1 / 3)
        self.assertEqual(calibration["bins"][0]["weighted_mass"], 1 / 12)
        self.assertEqual(calibration["bins"][2]["weighted_mass"], 0)
        self.assertEqual(calibration["bins"][2]["mean_prediction"]["status"], "undefined")

    def test_value_mask_changes_only_value_weights(self):
        rows = [row("one", "a", "b"), row("one", "a", "b", ply=2),
                row("two", "c", "d", value=math.nan, value_valid=False)]
        report = evaluate_losses(rows, [1., 3., 6.], values=[0., 1., -1.])
        nll, value = metric(report), metric(report, "value_mse")
        self.assertEqual(finite(nll["estimate"]), 4)
        self.assertEqual([side["weight"] for side in nll["side_weights"]], [.5, .5])
        self.assertEqual(finite(value["estimate"]), .5)
        self.assertEqual([side["weight"] for side in value["side_weights"]], [1.])
        self.assertEqual(value["counts"]["moves"], 2)
        self.assertEqual(value["masked_moves"], 1)
        self.assertEqual(value["excluded_game_sides_no_eligible_rows"], 1)
        self.assertEqual(sum(item["raw_count"] for item in report["overall"]["value_calibration"]["model"]["bins"]), 2)
        json.dumps(report, allow_nan=False)

    def test_missing_value_rank_leaves_value_aggregate_and_calibration_unavailable(self):
        rows = [row("one", "a", "b"), row("two", "c", "d", rank="2k", value_valid=False)]
        report = evaluate_losses(rows, [1., 3.], ranks=("1k", "2k"))
        self.assertEqual(metric(report)["status"], "available")
        self.assertEqual(metric(report, "value_mse")["missing_ranks"], ["2k"])
        calibration = report["overall"]["value_calibration"]["model"]
        self.assertEqual(calibration["status"], "unavailable")
        self.assertTrue(all(item["weighted_mass"] is None for item in calibration["bins"]))
        self.assertEqual(sum(item["raw_count"] for item in calibration["bins"]), 1)

    def test_policy_mask_does_not_discard_value_target(self):
        source = row("one", "a", "b") | {"policy_valid": False, "action": -1}
        report = evaluate_losses([source], [1.], values=[-1.])
        self.assertEqual(metric(report)["status"], "unavailable")
        self.assertEqual(finite(metric(report, "value_mse")["estimate"]), 4)


if __name__ == "__main__":
    unittest.main()
