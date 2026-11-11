import copy
import json
from pathlib import Path
import tempfile
import unittest

from kenggo import kgs_admission as admission
from kenggo.kgs_eligibility import classify as legacy_classify, exposure_registry, handle_groups
from test_kgs_eligibility import archive, game


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.record = game()
        self.groups = handle_groups(["Alice", "Bob", "Carol", "cu2", "GnuGo2", "LibertyBot"])
        self.exposure = exposure_registry([self.record], [{"user": "Alice", "months": [1]}], self.groups)

    def decide(self, record=None, kind="Ranked"):
        record = self.record if record is None else record
        return admission.classify(record, archive(record, kind), self.groups, self.exposure)

    def test_presumed_admission_preserves_quarantine_identity_masks_and_original(self):
        before = copy.deepcopy(self.record)
        old = legacy_classify(self.record, archive(self.record), self.groups, self.exposure)
        result = self.decide()
        self.assertTrue(result["training_eligible"])
        self.assertEqual(result["eligibility_status"], "admitted_presumed_human")
        self.assertEqual(result["eligibility_reasons"], [])
        self.assertEqual(result["ordinary_status"], "presumed_ordinary_ranked")
        for color in ("black", "white"):
            self.assertEqual(result[color + "_human_status"], "presumed_human")
            self.assertFalse(result[color + "_identity_valid"])
            self.assertEqual(result[color + "_persona"], "default")
            self.assertEqual(result[color + "_group"], old[color + "_group"])
        self.assertTrue(result["final_evaluation_excluded"])
        self.assertEqual(result["quarantine_reasons"], old["quarantine_reasons"])
        self.assertEqual(self.record, before)
        self.assertFalse(old["training_eligible"])

    def test_known_bot_and_reviewed_alias_are_excluded_on_either_side(self):
        for black, white in (("GnUgO2", "Bob"), ("Alice", "LibertyBot")):
            with self.subTest(black=black, white=white):
                self.assertFalse(self.decide(game(black, white))["training_eligible"])
        groups = handle_groups(self.groups, reviewed_aliases=[("Carol", "GnuGo2")])
        exposure = exposure_registry([self.record], [], groups)
        record = game("Carol", "Bob")
        result = admission.classify(record, archive(record), groups, exposure)
        self.assertFalse(result["training_eligible"])
        self.assertEqual(result["black_human_status"], "historical_bot_handle_exclusion")

    def test_only_explicit_ranked_without_events_can_be_presumed_ordinary(self):
        for kind in (None, "Unknown", "ranked", "Tournament", "Simul", "Teaching", "Free", "Review"):
            with self.subTest(kind=kind):
                self.assertFalse(self.decide(kind=kind)["training_eligible"])
        self.assertFalse(admission.classify(self.record, None, self.groups, self.exposure)["training_eligible"])
        for prop in ("EV", "RO"):
            record = copy.deepcopy(self.record)
            record["metadata"][prop] = "Tournament evidence"
            result = self.decide(record)
            self.assertFalse(result["training_eligible"])
            self.assertIn("event_or_round_requires_classification", result["eligibility_reasons"])

    def test_other_exclusions_and_missing_groups_cannot_be_overridden_by_claims(self):
        records = [game("Unknown", "Bob"), game("Alice", "ALICE"), copy.deepcopy(self.record)]
        records[-1]["parse_status"] = "corrupt"
        for reason in ("rules_or_komi_out_of_scope", "time_control_unknown_or_out_of_scope",
                       "sgf_rank_unknown_or_out_of_scope", "archive_sgf_result_mismatch",
                       "played_variation_unverified"):
            record = copy.deepcopy(self.record)
            record["exclusions"] = [reason]
            records.append(record)
        for record in records:
            record.update(training_eligible=True, human_status="verified", ordinary_verified=True,
                          source_reuse_permitted=True, stable_account_ids_verified=True)
            with self.subTest(record=record):
                result = self.decide(record)
                self.assertFalse(result["training_eligible"])
                self.assertFalse(result["black_identity_valid"])

    def test_wrong_source_and_stale_alias_registry_are_rejected(self):
        with self.assertRaises(ValueError):
            admission.classify(self.record, {"url": "other", "type": "Ranked"}, self.groups, self.exposure)
        changed = handle_groups(self.groups, reviewed_aliases=[("Alice", "Carol")])
        with self.assertRaises(ValueError):
            admission.classify(self.record, archive(self.record), changed, self.exposure)

    def test_policy_document_cannot_be_forged_to_expand_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            changed = json.loads(admission.DECISION_PATH.read_text())
            changed["population"]["scope"] = "All games"
            policy = root / "policy.json"
            policy.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "Owner policy decision differs"):
                admission.audit(root / "missing_audit", root / "missing_sample", root / "missing_discovery",
                                root / "out", decision_path=policy)
            self.assertFalse((root / "out").exists())

    def test_production_audit_rejects_unpinned_source_before_admission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = [root / name for name in ("audit", "sample", "discovery")]
            for path in inputs:
                path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "Frozen audit hash mismatch"):
                admission.audit(*inputs, root / "out")
            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
