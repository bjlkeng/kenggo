"""Saved-evidence presentation must preserve measurement scope and missingness."""

from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from kenggo import learning_review as review


def metric(value, *, status="available", ranks=None):
    return {"status": status, "estimate": {"status": "finite", "value": value},
        "counts": {"moves": 12, "games": 2, "accounts": 3}, "masked_moves": 4,
        "aggregate_ranks": ranks or ["5k"], "missing_ranks": [] if status == "available" else ["15k"],
        "uncertainty": {"status": "inconclusive", "interval95": None, "reasons": ["few_accounts"],
                        "large_map_not_for_display": [1, 2]}}


def fixture(identity="current"):
    m = {"nll": metric(2.5), "top1": metric(.3), "value_mse": metric(.7)}
    report = {"aggregate_ranks": ["5k"], "expected_outputs": 12, "overall": {"metrics": m}}
    training = {"completed_updates": 8, "presented_examples": 16, "training_rows": 16,
                "update_metrics": [{"next_step": 4, "updates": 4, "presented_examples": 8,
                                    "policy_count": 8, "value_count": 6, "policy_loss": 3., "value_loss": .5},
                                   {"next_step": 8, "updates": 4, "presented_examples": 8,
                                    "policy_count": 8, "value_count": 0, "policy_loss": 2., "value_loss": None}]}
    return {"bundle_id": identity, "training": {"conditioned": training},
            "evaluations": {"validation_development": {"conditioned": {"metrics": report}}}}


def saved(root, data):
    path = root / (data["bundle_id"] + ".html")
    path.write_text('<html><body><a href="evidence.json#cost">Costs</a><main></main>'
                    '<script id="pilot-data" type="application/json">' + json.dumps(data) + '</script></body></html>')
    return {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}


def continuation_fixture():
    data = fixture()
    training = data["training"]["conditioned"]
    training["epoch_metrics"] = training.pop("update_metrics")
    training["training_window"] = "matched interval"
    training["validation_metrics"] = []
    for index, step in enumerate((0, 4, 8)):
        diagnostics = {}
        for name, pin, loss in (("training_monitor", "a", 2.9-index*.2),
                                ("validation", "b", (2.8, 2.5, 2.6)[index])):
            diagnostics[name] = {"nll": loss, "top1": .3+index*.02, "value_mse": .7,
                "rows": 12, "policy_count": 12, "value_count": 8, "aggregate_ranks": ["5k"],
                "diagnostic_sha256": pin*64, "evaluated_rows": 14, "excluded_rank_rows": 2}
        training["validation_metrics"].append({"round": index, "sampler_step": step,
            "lifetime_updates": 31250+step, "new_raw_best": index < 2,
            "meaningful_improvement": index < 2, **diagnostics})
    training.update(selected_sampler_step=4, selected_lifetime_updates=31254, selected_round=1)
    return data


class LearningReviewTests(unittest.TestCase):
    def test_saved_training_and_validation_are_separate_with_no_fabricated_curve(self):
        data = fixture()
        result = review.build(data)["runs"][0]
        points = result["training"]["conditioned"]["points"]
        self.assertEqual([p["step"] for p in points], [4, 8])
        self.assertEqual(points[-1]["nll"], 2.)
        self.assertIsNone(points[-1]["value_mse"])
        self.assertEqual(points[-1]["value_count"], 0)
        self.assertNotIn("top1", points[-1])
        scopes = result["evaluations"]["validation_development_komi13"]["conditioned"]
        self.assertEqual(list(scopes), ["reported"])
        self.assertEqual(scopes["reported"]["metrics"]["nll"]["estimate"]["value"], 2.5)
        self.assertEqual(scopes["reported"]["metrics"]["value_mse"]["masked_moves"], 4)
        self.assertNotIn("large_map_not_for_display", scopes["reported"]["metrics"]["nll"]["uncertainty"])

    def test_rank_scopes_unavailable_and_infinite_metrics_keep_their_meaning(self):
        data = fixture()
        report = data["evaluations"]["validation_development"]["conditioned"]["metrics"]
        report["review_scope"] = deepcopy(report)
        report["review_scope"]["aggregate_ranks"] = ["4k"]
        report["review_scope"]["excluded_rows"] = 7
        report["overall"]["metrics"]["nll"] = metric(123., status="unavailable")
        report["review_scope"]["overall"]["metrics"]["nll"]["estimate"] = {"status": "positive_infinity", "value": None}
        result = review.normalize(data, {})["evaluations"]["validation_development_komi13"]["conditioned"]
        self.assertEqual(result["reported"]["metrics"]["nll"]["status"], "unavailable")
        self.assertEqual(result["reported"]["metrics"]["nll"]["missing_ranks"], ["15k"])
        self.assertEqual(result["review_scope"]["metrics"]["nll"]["estimate"]["status"], "positive_infinity")
        self.assertEqual(result["review_scope"]["ranks"], ["4k"])
        self.assertEqual(result["review_scope"]["excluded_rows"], 7)

    def test_epoch_records_do_not_change_the_update_axis(self):
        data = fixture()
        t = data["training"]["conditioned"]
        t["epoch_metrics"] = t.pop("update_metrics")
        for i, block in enumerate(t["epoch_metrics"], 1):
            block["epoch"] = i
        result = review.normalize(data, {})["training"]["conditioned"]
        self.assertEqual(result["window"], "epoch")
        self.assertEqual([p["step"] for p in result["points"]], [4, 8])
        t["epoch_metrics"][-1]["next_step"] = 4
        with self.assertRaisesRegex(ValueError, "increase"):
            review.normalize(data, {})

    def test_checkpoint_curves_keep_epoch_zero_training_agreement_and_selected_earlier_point(self):
        data = continuation_fixture()
        original = deepcopy(data)
        result = review.normalize(data, {})["training"]["conditioned"]
        self.assertEqual(data, original)
        self.assertEqual([point["step"] for point in result["checkpoints"]], [0, 4, 8])
        self.assertEqual([point["lifetime_updates"] for point in result["checkpoints"]], [31250, 31254, 31258])
        self.assertEqual(result["checkpoints"][1]["training_monitor"]["top1"], .32)
        self.assertEqual(result["checkpoints"][1]["validation"]["nll"], 2.5)
        self.assertEqual(result["checkpoints"][1]["validation"]["excluded_rank_rows"], 2)
        self.assertEqual(result["selected_step"], 4)
        self.assertEqual(result["selected_lifetime_updates"], 31254)
        self.assertEqual(result["selected_round"], 1)
        self.assertEqual(result["updates"], 8)
        self.assertEqual(result["window"], "matched interval")
        self.assertNotIn("top1", result["points"][1])

    def test_selected_epoch_zero_and_recorded_best_markers_are_preserved(self):
        data = continuation_fixture()
        training = data["training"]["conditioned"]
        training.update(selected_sampler_step=0, selected_lifetime_updates=31250, selected_round=0)
        result = review.normalize(data, {})["training"]["conditioned"]
        self.assertEqual((result["selected_step"], result["selected_round"]), (0, 0))
        for key in ("selected_sampler_step", "selected_lifetime_updates", "selected_round"):
            training.pop(key)
        self.assertEqual(review.normalize(data, {})["training"]["conditioned"]["selected_step"], 4)

    def test_old_runs_keep_absent_checkpoint_metrics_and_final_validation(self):
        result = review.normalize(fixture(), {})["training"]["conditioned"]
        self.assertEqual(result["checkpoints"], [])
        self.assertEqual(result["selected_step"], 8)
        self.assertIsNone(result["selected_round"])
        self.assertIsNone(result["selected_lifetime_updates"])

    def test_empty_and_partial_runs_preserve_absence_and_arbitrary_tier_arm_names(self):
        for evaluations in ({}, {name: {} for name in ("population_development_komi13",
                "population_development_komi1", "validation_development_komi13", "validation_development_komi1")}):
            data = {"bundle_id": "failed-before-training", "training": {}, "evaluations": evaluations}
            result = review.build(data)["runs"][0]
            self.assertEqual(result["training"], {})
            self.assertEqual(result["evaluations"], evaluations)
            self.assertTrue(result["current"])
            self.assertIn('id="learning-review"', review.insert("<main></main>", review.build(data)))
        data = continuation_fixture()
        data["training"]["tier1-candidate"] = data["training"].pop("conditioned")
        data["evaluations"] = {}
        result = review.build(data)["runs"][0]
        self.assertEqual(set(result["training"]), {"tier1-candidate"})
        self.assertEqual(len(result["training"]["tier1-candidate"]["checkpoints"]), 3)
        self.assertEqual(result["evaluations"], {})

    def test_checkpoint_curves_reject_changed_samples_missing_metrics_and_bad_cursors(self):
        changes = (
            lambda value: value["validation_metrics"][0].update(sampler_step=1),
            lambda value: value["validation_metrics"][1].update(lifetime_updates=31255),
            lambda value: value["validation_metrics"][2].update(sampler_step=4),
            lambda value: value["validation_metrics"][1]["validation"].update(diagnostic_sha256="c"*64),
            lambda value: value["validation_metrics"][1]["validation"].update(value_count=7),
            lambda value: value["validation_metrics"][1]["training_monitor"].pop("top1"),
            lambda value: value["validation_metrics"][1]["validation"].update(nll=float("nan")),
            lambda value: value.update(selected_sampler_step=7),
            lambda value: value.update(selected_lifetime_updates=31258),
        )
        for change in changes:
            data = continuation_fixture()
            change(data["training"]["conditioned"])
            with self.subTest(change=change), self.assertRaises(ValueError):
                review.normalize(data, {})

    def test_catalog_reads_only_earlier_pinned_pages_and_keeps_each_scope(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old = fixture("old")
            old["evaluations"]["validation_development"]["conditioned"]["metrics"]["aggregate_ranks"] = ["1k"]
            pin = saved(root, old)
            entries = [{"id": "old", "bundle_id": "old", "source": pin, "description": "Small pilot"},
                       {"id": "current", "bundle_id": "current", "source": {"path": "unused-current"}},
                       {"id": "future", "bundle_id": "future", "source": {"path": "must-not-open"}}]
            path = root / "catalog.json"
            path.write_text(json.dumps({"schema": "e2_review_history_v1", "runs": entries}))
            result = review.build(fixture(), path)
            self.assertEqual([r["id"] for r in result["runs"]], ["old", "current"])
            self.assertEqual(result["history_sources"], [pin])
            self.assertTrue(result["runs"][-1]["current"])
            self.assertEqual(result["runs"][0]["evaluations"]["validation_development_komi13"]["conditioned"]["reported"]["ranks"], ["1k"])
            Path(pin["path"]).write_text("changed")
            with self.assertRaisesRegex(ValueError, "pinned hash"):
                review.build(fixture(), path)

    def test_insert_escapes_embedded_data_preserves_existing_payload_and_links(self):
        original = '<html><body><main id="content"></main><script id="old-script">keep();</script></body></html>'
        learning = review.build(fixture())
        learning["runs"][0]["description"] = '</script><script>alert(1)</script>'
        page = review.insert(original, learning)
        self.assertIn('<script id="old-script">keep();</script>', page)
        self.assertNotIn('</script><script>alert(1)', page)
        self.assertIn('\\u003c/script>', page)
        with self.assertRaisesRegex(ValueError, "already contains"):
            review.insert(page, learning)
        links = review.preserve_links('<a href="../proof.json#cost">A</a><a href="#local">B</a>', Path('/tmp/old/review.html'))
        self.assertIn('href="file:///tmp/proof.json#cost"', links)
        self.assertIn('href="#local"', links)

    def test_enrichment_writes_a_new_page_without_changing_sealed_source(self):
        spec = importlib.util.spec_from_file_location("review_learning", review.ROOT / "tools/review_learning.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pin = saved(root, fixture())
            original = Path(pin["path"]).read_bytes()
            catalog = root / "catalog.json"
            catalog.write_text(json.dumps({"schema": "e2_review_history_v1", "runs": [
                {"id": "current", "bundle_id": "current", "source": pin}]}))
            output = root / "new/review.html"
            result = module.export(catalog, "current", output)
            self.assertEqual(Path(pin["path"]).read_bytes(), original)
            self.assertEqual(result["inference_calls"], 0)
            self.assertEqual(result["training_points"], {"current": {"conditioned": 2}})
            self.assertEqual(review.read_review({"path": str(output), "sha256": result["html_sha256"]})[0], fixture())
            with self.assertRaisesRegex(ValueError, "unused"):
                module.export(catalog, "current", output)


if __name__ == "__main__":
    unittest.main()
