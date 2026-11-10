import copy
import unittest

from kenggo.kgs_eligibility import (classify, coverage, exposure_registry,
                                    handle_groups, normalize_handle)


def game(black="Alice", white="Bob", source="inspected", day="2025-01-02"):
    return {"source_url": source, "sgf_sha256": source + "_bytes", "sequence_fingerprint": source + "_moves",
            "metadata": {"PB": black, "PW": white, "BR": "5k", "WR": "1d", "DT": day},
            "parse_status": "parsed", "exclusions": [], "result_kind": "scored",
            "final_two_explicit_passes": True,
            "actions_by_color": {"b": {"moves": 4, "explicit_passes": 1},
                                 "w": {"moves": 3, "explicit_passes": 1}}}


def archive(record, kind="Ranked"):
    return {"url": record["source_url"], "type": kind}


class EligibilityTests(unittest.TestCase):
    def setUp(self):
        self.old = game()
        self.groups = handle_groups(["Alice", "Bob", "Carol", "Dave", "cu2", "GnuGo2", "LibertyBot"])
        self.exposure = exposure_registry([self.old], [{"user": "Alice", "months": [1, 4]}], self.groups)

    def decide(self, record, kind="Ranked"):
        return classify(record, archive(record, kind), self.groups, self.exposure)

    def test_case_and_documented_name_domain(self):
        self.assertEqual(normalize_handle("AliCe"), "alice")
        self.assertEqual(handle_groups(["Alice", "ALICE"]), {"alice": "kgs-handle:alice"})
        for value in (None, 17, "", "1Alice", "A_B", "A B", "Älice", "LongName123"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_handle(value)

    def test_alias_closure_is_deterministic_and_never_uses_opponents(self):
        aliases = [("Alice", "Carol"), ("Carol", "Dave")]
        first = handle_groups(["Dave", "Bob", "Alice"], reviewed_aliases=aliases)
        second = handle_groups(["Alice", "Bob", "Dave"], reviewed_aliases=list(reversed(aliases)))
        self.assertEqual(first, second)
        self.assertEqual(first["alice"], first["dave"])
        self.assertNotEqual(first["alice"], first["bob"])
        self.assertNotEqual(self.groups["alice"], self.groups["bob"])

    def test_gaps_reuse_and_personas_do_not_change_groups(self):
        early = self.decide(game("Alice", "Bob", "early", "2005-01-01"))
        late = self.decide(game("ALICE", "Bob", "late", "2026-11-10"))
        self.assertEqual(early["black_group"], late["black_group"])
        self.assertEqual(early["black_persona"], early["white_persona"])
        self.assertFalse(early["black_identity_valid"])
        self.assertFalse(early["white_identity_valid"])
        self.assertNotEqual(early["black_group"], early["white_group"])

    def test_ranked_flags_and_forged_claims_cannot_admit_humans(self):
        record = game()
        record.update(human_status="verified", stable_account_ids_verified=True, training_ready=True,
                      ordinary_verified=True, source_reuse_permitted=True, reviewed_aliases=[("Alice", "Bob")])
        record["metadata"].update(authLevel="normal", flags="=", verified_human=True)
        decision = self.decide(record)
        self.assertEqual(decision["eligibility_status"], "unverified")
        self.assertEqual(decision["black_human_status"], "unverified")
        self.assertEqual(decision["white_human_status"], "unverified")
        self.assertEqual(decision["ordinary_status"], "unverified")
        self.assertFalse(decision["training_eligible"])
        self.assertFalse(decision["black_identity_valid"])
        self.assertIn("training_reuse_basis_unresolved", decision["eligibility_reasons"])
        self.assertNotEqual(decision["black_group"], decision["white_group"])

    def test_historical_bot_group_is_excluded_without_claiming_generation(self):
        decision = self.decide(game("GnUgO2", "Bob", "bot"))
        self.assertEqual(decision["black_human_status"], "historical_bot_handle_exclusion")
        self.assertEqual(decision["white_human_status"], "unverified")
        self.assertEqual(decision["eligibility_status"], "excluded")
        groups = handle_groups(self.groups, reviewed_aliases=[("Carol", "GnuGo2")])
        exposure = exposure_registry([self.old], [], groups)
        aliased = game("Carol", "Bob", "aliased")
        result = classify(aliased, archive(aliased), groups, exposure)
        self.assertEqual(result["black_human_status"], "historical_bot_handle_exclusion")

    def test_ordinary_status_does_not_follow_absent_event_fields(self):
        for kind in ("Tournament", "Simul", "Teaching", "Review", "Free", "Rengo", "Private"):
            with self.subTest(kind=kind):
                result = self.decide(game(), kind)
                self.assertEqual(result["ordinary_status"], "excluded_game_type")
                self.assertEqual(result["eligibility_status"], "excluded")
        for kind in ("Ranked", None, "Unrecognized"):
            self.assertEqual(self.decide(game(), kind)["ordinary_status"], "unverified")
        event = game()
        event["metadata"]["EV"] = "Local cup"
        self.assertIn("event_or_round_requires_classification", self.decide(event)["eligibility_reasons"])

    def test_exposure_follows_later_games_but_does_not_spread_through_opponents(self):
        later = self.decide(game("Alice", "Carol", "later", "2026-11-10"))
        self.assertTrue(later["final_evaluation_excluded"])
        self.assertIn("exposed_handle_group", later["quarantine_reasons"])
        unrelated = self.decide(game("Carol", "Dave", "unrelated", "2026-11-11"))
        self.assertFalse(unrelated["final_evaluation_excluded"])
        self.assertFalse(unrelated["training_eligible"])

    def test_approved_alias_inherits_exposure_and_late_registry_changes_fail(self):
        groups = handle_groups(self.groups, reviewed_aliases=[("Alice", "Carol")])
        exposure = exposure_registry([self.old], [], groups)
        record = game("Carol", "Dave", "future")
        self.assertTrue(classify(record, archive(record), groups, exposure)["final_evaluation_excluded"])
        with self.assertRaises(ValueError):
            classify(record, archive(record), groups, self.exposure)

    def test_original_discovery_and_probe_groups_are_reserved_across_eras(self):
        exposure = exposure_registry([], [{"user": "Carol", "months": [7]}], self.groups)
        for black in ("Carol", "cu2"):
            record = game(black, "Dave", "unseen_id", "2030-02-02")
            result = classify(record, archive(record), self.groups, exposure)
            self.assertTrue(result["final_evaluation_excluded"])
        self.assertIn(("carol", "2025-07"), exposure.account_months)
        self.assertIn(("cu2", "2025-09"), exposure.account_months)

    def test_failed_or_relabelled_selected_records_and_duplicates_stay_reserved(self):
        failed = {"source_url": "inspected", "parse_status": "corrupt", "exclusions": ["corrupt"]}
        result = self.decide(failed)
        self.assertTrue(result["final_evaluation_excluded"])
        self.assertIn("inspected_source_id", result["quarantine_reasons"])
        self.assertIsNone(result["black_group"])
        for field in ("sgf_sha256", "sequence_fingerprint"):
            record = game("Carol", "Dave", "another_id")
            record[field] = self.old[field]
            self.assertTrue(self.decide(record)["final_evaluation_excluded"])
        selected = game("Carol", "Dave", "inspected")
        self.assertIn("inspected_source_id", self.decide(selected)["quarantine_reasons"])

    def test_unknown_groups_and_wrong_source_evidence_fail_closed(self):
        result = self.decide(game("Unknown", "Bob", "unknown"))
        self.assertEqual(result["eligibility_status"], "excluded")
        self.assertTrue(result["final_evaluation_excluded"])
        with self.assertRaises(ValueError):
            classify(self.old, {"url": "other", "type": "Ranked"}, self.groups, self.exposure)
        with self.assertRaises(ValueError):
            exposure_registry([self.old], [], {"alice": "kgs-handle:alice"})

    def test_coverage_retains_player_game_denominators_and_all_empty_ranks(self):
        same_rank = copy.deepcopy(self.old)
        same_rank["metadata"]["WR"] = "5k"
        counts = coverage([same_rank])
        self.assertEqual(counts["games"], 1)
        self.assertEqual(counts["by_player_rank"]["5k"]["player_game_appearances"], 2)
        self.assertEqual(counts["by_player_rank"]["5k"]["moves"], 7)
        empty = coverage([])
        self.assertEqual(len(empty["by_player_rank"]), 16)
        self.assertEqual(empty["games"], 0)
        self.assertEqual(empty["games_with_final_two_passes"], 0)
        self.assertEqual(sum(r["moves"] for r in empty["by_player_rank"].values()), 0)


if __name__ == "__main__":
    unittest.main()
