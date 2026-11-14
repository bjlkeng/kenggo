from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from kenggo.data_expansion import reference
from kenggo.data_splits import read_membership
from test_data_expansion import fixture, run, write_json


spec = importlib.util.spec_from_file_location(
    "data_expansion_review_tool", Path(__file__).resolve().parents[1] / "tools/review_data_expansion.py")
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


class DataExpansionReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.request, cls.request_ref, sources, _ = fixture(cls.root)
        sources.update(status="authentication_blocked", stop_reason="archive_authentication_required",
                       counts={"frozen_candidates": 4, "accessible_candidates": 3,
                               "query_statuses": {"failed": 1, "unattempted": 285}})
        cls.source_ref = write_json(cls.root / "sources.json", sources)
        cls.summary = run(cls.root, cls.request_ref, cls.source_ref)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_export_counts_endings_and_coverage_without_old_target_decoding(self):
        _, parts = read_membership(self.root / "base/splits", self.request["refs"]["split_manifest"]["sha256"])
        old_ids = {row["source_url"] for rows in parts.values() for row in rows}
        loads, np_load = json.loads, np.load

        def guarded_json(value, *args, **kwargs):
            if isinstance(value, str) and value.startswith('{"actions":'):
                if any('"source_url":' + json.dumps(key) in value for key in old_ids):
                    raise AssertionError("An old canonical target was decoded")
            return loads(value, *args, **kwargs)

        def guarded_arrays(path, *args, **kwargs):
            if any(part in str(path) for part in ("/final/", "confirmation/")):
                raise AssertionError("A held-out array was decoded")
            return np_load(path, *args, **kwargs)

        output = self.root / "review"
        with mock.patch.object(review.json, "loads", side_effect=guarded_json), \
                mock.patch("numpy.load", side_effect=guarded_arrays), redirect_stdout(io.StringIO()):
            review.build(self.request_ref, reference(self.root / "out/summary.json"), self.source_ref, output)
        result = json.loads((output / "readout.json").read_text())
        self.assertFalse(result["checks"]["final_and_confirmation_targets_decoded"])
        self.assertEqual(result["collection"]["status"], "authentication_blocked")
        self.assertEqual(result["collection"]["counts"]["query_statuses"]["unattempted"], 285)
        self.assertEqual(result["coverage"]["train"]["endings"]["resignation"], 2)
        self.assertEqual(result["coverage"]["validation_development"]["endings"]["timeout"], 1)
        self.assertEqual(result["coverage"]["validation_development"]["value_valid"], 2)
        for coverage in result["coverage"].values():
            self.assertEqual(len(coverage["cells"]), 96)
            self.assertEqual(sum(coverage["opponent_ranks"].values()), coverage["examples"])
            for key in ("rank_pairs", "nominal_clocks"):
                self.assertEqual(sum(row["moves"] for row in coverage[key]), coverage["examples"])
            self.assertEqual(sum(v for k, v in coverage["endings"].items()
                                 if k != "two_explicit_pass_endings"), coverage["games"])
        self.assertTrue((output / "review.html").exists())

    def test_consistent_value_sign_reversal_is_rejected_against_the_source_winner(self):
        original = review.read_shards

        def reversed_values(*args, **kwargs):
            for arrays, metadata in original(*args, **kwargs):
                yield {**arrays, "value": -arrays["value"]}, metadata

        with mock.patch.object(review, "read_shards", side_effect=reversed_values):
            with self.assertRaisesRegex(ValueError, "Value label differs from the recorded winner"):
                review.inspect(self.request, self.summary, lambda: None)


if __name__ == "__main__":
    unittest.main()
