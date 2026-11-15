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
            self.assertTrue(all(v["games"] <= v["game_sides"] <= 2 * v["games"]
                                for v in coverage["ranks"].values()))
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

    def test_continuation_page_counts_include_both_rounds_without_doubling_candidates(self):
        parent = {"schema": "e2_expansion_sources_v1", "status": "completed",
                  "counts": {"query_statuses": {"retrieved": 1000}, "frozen_candidates": 35000}}
        parent_ref = write_json(self.root / "parent-source.json", parent)
        current = {"schema": parent["schema"], "status": "partial",
                   "provenance": {"previous": {"sources": parent_ref}},
                   "counts": {"query_statuses": {"retrieved": 198, "failed": 2},
                              "frozen_candidates": 37000}, "diagnostics": [{"reason": "fixture_failure"}]}
        result = review.collection_details(current)
        self.assertEqual(result["counts"]["query_statuses"], {"retrieved": 1198, "failed": 2})
        self.assertEqual(result["counts"]["frozen_candidates"], 37000)
        self.assertEqual(len(result["rounds"]), 2)
        self.assertEqual(result["diagnostics"], current["diagnostics"])

    def test_scope_candidates_exclude_other_failures_and_require_matching_evidence(self):
        group = {"records": 4, "status": {"canonical": 1, "source_excluded": 3},
                 "recorded_exclusion_groups": {"Only clock family excluded": 0},
                 "scope_only_cells": [
                     {"RU": "Japanese", "KM": .5, "records": 1,
                      "recorded_exclusions": "Only rules or komi excluded"},
                     {"RU": "Japanese", "KM": .5, "records": 1,
                      "recorded_exclusions": "Only rules, komi and clock family excluded"},
                     {"RU": "Chinese", "KM": 7.5, "records": 1,
                      "recorded_exclusions": "Only rules or komi excluded"}]}
        exclusions = {"metadata:rules_or_komi_out_of_scope": 3,
                      "metadata:time_control_unknown_or_out_of_scope": 1}
        audit = {"schema": "e2_sgf_scope_audit_v1", "status": "complete",
                 "inputs": {"build_request": self.request_ref, "sources": self.source_ref},
                 "scope": {"expected_records": 4, "processed_records": 4},
                 "aggregate": {**group, "exclusions": exclusions,
                               "rules_komi": [{"RU": "Japanese", "KM": .5, "records": 2}],
                               "rules_metadata": {}, "clock_metadata": {}},
                 "by_role": {"train": group}, "training_by_rank": {"15k": group}, "costs": {}}
        summary = {"source_completed": 4, "new_source_admitted": 1,
                   "source_exclusion_reasons": exclusions}
        audit_ref = write_json(self.root / "scope-audit.json", audit)
        result = review.read_scope_audit(audit_ref, self.request_ref, self.source_ref, summary)
        self.assertEqual(result["japanese_half_komi"], 2)
        self.assertEqual(result["training_half_komi_only"], 1)
        self.assertEqual(result["training_ranks"][0]["half_komi_only"], 1)
        with self.assertRaisesRegex(ValueError, "another run"):
            review.read_scope_audit(audit_ref, self.request_ref,
                                   {**self.source_ref, "sha256": "f" * 64}, summary)
        with self.assertRaisesRegex(ValueError, "screening disagree"):
            review.read_scope_audit(audit_ref, self.request_ref, self.source_ref,
                                   {**summary, "source_completed": 3})


if __name__ == "__main__":
    unittest.main()
