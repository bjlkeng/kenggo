from copy import deepcopy
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from kenggo import progressive_evaluation as evaluation
from kenggo.checkpoints import save_json
from kenggo.recorded_statistics import paired_report
from test_recorded_statistics import prediction, row


def saved(rows, outputs, pin="a" * 64):
    return {"diagnostic_sha256": pin, "predictions": [
        {**{key: value for key, value in source.items() if key != "legal"},
         "action": source["action"] if source["policy_valid"] else None,
         "value": source["value"] if source["value_valid"] else None,
         "nll": -output.log_policy[source["action"]] if source["policy_valid"] else None,
         "top1": float(max(range(362), key=output.log_policy.__getitem__) == source["action"]) if source["policy_valid"] else None,
         "prediction_value": output.value,
         "value_mse": (output.value - source["value"]) ** 2 if source["value_valid"] else None}
        for source, output in zip(rows, outputs)]}


def metric(interval=(-0.08, -0.03), status="available", uncertainty="available"):
    return {"status": status, "uncertainty": {"status": uncertainty,
            "interval95": list(interval) if interval is not None else None}}


def comparisons():
    reports = {"validation_development_komi13": {"metrics": {
        "nll": metric(), "value_mse": metric((-.01, .01))}},
        "population_development_komi1": {"metrics": {
            "nll": metric((-.01, .01)), "value_mse": metric((-.01, .01))}}}
    return {"control": deepcopy(reports), "parent": deepcopy(reports)}


def games(ranks=("1k", "2k"), per_cell=8, prefix="validation"):
    return [{"source_url": f"{prefix}/{komi}/{rank}/{number}", "ranks": [rank],
             "komi_halfpoints": komi, "rows": 2 + number}
            for komi in (1, 13) for rank in ranks for number in range(per_cell)]


class PairedLossTests(unittest.TestCase):
    def test_saved_loss_comparison_exactly_matches_public_hierarchy_and_uncertainty(self):
        rows = []
        for number in range(60):
            rank = "1k" if number < 20 else "2k"
            pair = (("shared-a", "shared-b") if number < 2 else
                    ("shared-b", "shared-a") if number == 2 else (f"m{number}", f"o{number}"))
            rows.append(row(f"game{number}", *pair, rank=rank, value_valid=number % 7 != 0))
            if number % 3 == 0:
                rows.append(row(f"game{number}", *pair, rank=rank, ply=2, value_valid=number % 7 != 0))
            if number % 5 == 0:
                rows.append(row(f"game{number}", *pair[::-1], rank=rank, mover="W", ply=1,
                                value=-1, value_valid=number % 7 != 0))
        rows[3] = {**rows[3], "policy_valid": False, "action": -1}
        candidate = [prediction(source, .3 + (index % 9) * .17, value=-.5 + (index % 5) * .2)
                     for index, source in enumerate(rows)]
        control = [prediction(source, .4 + (index % 7) * .11, value=.4 - (index % 4) * .2)
                   for index, source in enumerate(rows)]
        ranks = ("1k", "2k")
        expected = paired_report(rows, candidate, control, aggregate_ranks=ranks)
        actual = evaluation.paired_losses(saved(rows, candidate), saved(rows[::-1], control[::-1]), ranks=ranks)
        self.assertEqual(actual["metrics"], expected["overall"]["metrics"])
        self.assertEqual(actual["metrics"]["nll"]["uncertainty"]["status"], "available")
        self.assertGreater(actual["metrics"]["value_mse"]["masked_moves"], 0)
        self.assertNotEqual(actual["metrics"]["nll"]["estimate"],
                            actual["metrics"]["nll"]["diagnostics"]["pooled_move"])

    def test_unknown_aggregate_rank_rows_are_reported_as_excluded(self):
        rows = [row("known", "a", "b", rank="1k"), row("outside", "c", "d", rank="3k")]
        candidate = [prediction(source, .5) for source in rows]
        control = [prediction(source, .8) for source in rows]
        actual = evaluation.paired_losses(saved(rows, candidate), saved(rows, control), ranks=("1k",))
        expected = paired_report(rows[:1], candidate[:1], control[:1], aggregate_ranks=("1k",))
        self.assertEqual(actual["metrics"], expected["overall"]["metrics"])
        self.assertEqual((actual["rows"], actual["excluded_rows"]), (1, 1))

    def test_mask_target_identity_key_and_diagnostic_mismatches_fail(self):
        rows = [row("one", "a", "b"), row("two", "c", "d")]
        baseline = saved(rows, [prediction(source, .5) for source in rows])
        for field, value in (("value_valid", False), ("policy_valid", False), ("value", -1.),
                             ("action", 1), ("group_opponent", "changed"), ("row_key", "other")):
            changed = deepcopy(baseline)
            changed["predictions"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                evaluation.paired_losses(changed, baseline, ranks=("1k",))
        for changed in (baseline | {"diagnostic_sha256": "b" * 64},
                        baseline | {"predictions": baseline["predictions"][:-1]},
                        baseline | {"predictions": baseline["predictions"] + baseline["predictions"][:1]}):
            with self.assertRaises(ValueError):
                evaluation.paired_losses(changed, baseline, ranks=("1k",))

    def test_masked_losses_are_ignored_but_nonfinite_valid_losses_fail(self):
        rows = [row("masked", "a", "b", value_valid=False), row("known", "c", "d")]
        baseline = saved(rows, [prediction(source, .5) for source in rows])
        changed = deepcopy(baseline)
        changed["predictions"][0]["value_mse"] = math.nan
        result = evaluation.paired_losses(changed, baseline, ranks=("1k",))
        self.assertEqual(result["metrics"]["value_mse"]["masked_moves"], 1)
        for name in ("nll", "top1", "value_mse"):
            changed = deepcopy(baseline)
            changed["predictions"][1][name] = math.nan
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "finite"):
                evaluation.paired_losses(changed, baseline, ranks=("1k",))


class GrowthDecisionTests(unittest.TestCase):
    def test_both_comparators_need_supported_useful_gain(self):
        self.assertTrue(evaluation.decide(comparisons())["continue_growth"])
        for opponent in ("control", "parent"):
            for unavailable in (metric((-.04, -.019)), metric(None), metric(status="unavailable"),
                                metric(uncertainty="inconclusive")):
                changed = comparisons()
                changed[opponent]["validation_development_komi13"]["metrics"]["nll"] = unavailable
                with self.subTest(opponent=opponent, metric=unavailable):
                    self.assertFalse(evaluation.decide(changed)["continue_growth"])

    def test_clear_regression_on_any_fixed_set_stops_growth_and_boundaries_are_explicit(self):
        for opponent in ("control", "parent"):
            for name in ("nll", "value_mse"):
                changed = comparisons()
                changed[opponent]["population_development_komi1"]["metrics"][name] = metric((.0201, .05))
                with self.subTest(opponent=opponent, name=name):
                    result = evaluation.decide(changed)
                    self.assertFalse(result["continue_growth"])
                    self.assertTrue(any("regression" in reason for reason in result["reasons"]))
        changed = comparisons()
        for opponent in changed:
            changed[opponent]["validation_development_komi13"]["metrics"]["nll"] = metric((-.04, -.02))
            changed[opponent]["population_development_komi1"]["metrics"]["value_mse"] = metric((.02, .05))
        self.assertTrue(evaluation.decide(changed)["continue_growth"])


class WholeGameSelectionTests(unittest.TestCase):
    def test_selection_is_deterministic_and_new_rounds_are_disjoint_whole_games(self):
        population = games()
        frozen = {item["source_url"] for item in population if item["source_url"].endswith("/0")}
        first = evaluation.choose_games(population, seed="first", excluded=frozen, ranks=("1k", "2k"))
        self.assertEqual(first, evaluation.choose_games(population[::-1], seed="first", excluded=frozen,
                                                       ranks=("1k", "2k")))
        second = evaluation.choose_games(population, seed="second", excluded=frozen | set(first), ranks=("1k", "2k"))
        self.assertFalse(set(first) & frozen)
        self.assertFalse(set(second) & (frozen | set(first)))
        lookup = {item["source_url"]: item for item in population}
        self.assertEqual(len(first), 8)
        self.assertEqual(len(second), 8)
        for selected in (first, second):
            self.assertTrue(all(lookup[key]["rows"] >= 2 for key in selected))
            for komi in (1, 13):
                for rank in ("1k", "2k"):
                    self.assertEqual(sum(lookup[key]["komi_halfpoints"] == komi and rank in lookup[key]["ranks"]
                                         for key in selected), 2)
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            evaluation.choose_games(population, seed="none", excluded={item["source_url"] for item in population},
                                    ranks=("1k", "2k"))

    def test_prepare_reuses_assessments_and_excludes_all_previous_stopping_games(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            validation = games(evaluation.REVIEW_RANKS)
            train = games(evaluation.REVIEW_RANKS, prefix="train")
            fixed_source = validation[0]["source_url"]
            fake_examples, fake_normalizer = {"path": "examples", "sha256": "a" * 64}, {"path": "norm", "sha256": "b" * 64}
            common_path = root / "common.json"
            save_json(common_path, {"inputs": {"examples_sha256": "a" * 64, "normalizer_sha256": "b" * 64},
                                    "rows": [{"row_key": "fixed-row"}]})
            comparison_path = root / "comparison.json"
            save_json(comparison_path, {"common_sets": [{"spec": {"id": "fixed", "role": "validation_development",
                "source_ids": [fixed_source]}, "manifest": evaluation.reference(common_path)}]})
            config = {"inputs": {"examples": fake_examples, "normalizer": fake_normalizer,
                "fixed_comparison": evaluation.reference(comparison_path)}, "parameters": {"tier": 1}}

            def diagnostic(inputs, output, **kwargs):
                output.mkdir(parents=True)
                filename = output / "manifest.json"
                save_json(filename, {"source_ids": kwargs["source_ids"], "purpose": kwargs["purpose"]})
                return evaluation.reference(filename)

            first_out = root / "first"
            first_out.mkdir()
            with mock.patch("kenggo.progressive_evaluation.inventory", side_effect=lambda _, role, **kwargs:
                            train if role == "train" else validation), \
                    mock.patch("kenggo.progressive_evaluation.prepare_diagnostic", side_effect=diagnostic) as prepared:
                first = evaluation.prepare(config, first_out)
            assessment_call = next(call for call in prepared.call_args_list if call.kwargs["purpose"] == "assessment")
            self.assertEqual(assessment_call.kwargs["expected_row_keys"], ["fixed-row"])
            first_ids = set(evaluation.read(first["epoch_validation"])["source_ids"])
            self.assertNotIn(fixed_source, first_ids)
            first_path = root / "first-preparation.json"
            save_json(first_path, first)
            second_out = root / "second"
            second_out.mkdir()
            config["inputs"]["previous_preparation_1"] = evaluation.reference(first_path)
            config["parameters"]["tier"] = 2
            with mock.patch("kenggo.progressive_evaluation.inventory", side_effect=AssertionError("Existing inventory must be reused")), \
                    mock.patch("kenggo.progressive_evaluation.prepare_diagnostic", side_effect=diagnostic) as prepared:
                second = evaluation.prepare(config, second_out)
            self.assertEqual(len(prepared.call_args_list), 1)
            self.assertEqual(second["assessments"], first["assessments"])
            self.assertEqual(second["train_monitor"], first["train_monitor"])
            second_ids = set(evaluation.read(second["epoch_validation"])["source_ids"])
            self.assertFalse(second_ids & (first_ids | {fixed_source}))


if __name__ == "__main__":
    unittest.main()
