import unittest

from kenggo.sgf_audit import inspect_sgf, result_kind


ROOT = "FF[4]GM[1]CA[UTF-8]SZ[19]PB[a]PW[b]BR[5k]WR[5k]RU[Japanese]KM[6.5]TM[1200]OT[5x30 byo-yomi]DT[2025-01-01]"


def audit(moves=";B[aa];W[bb]", result="B+R", extra=""):
    return inspect_sgf(f"(;{ROOT}RE[{result}]{extra}{moves})".encode())


class SgfAuditTests(unittest.TestCase):
    def test_eof_does_not_invent_passes_or_a_result(self):
        record = audit(result="?")
        self.assertEqual(record["counters"]["explicit_passes"], 0)
        self.assertFalse(record["final_two_explicit_passes"])
        self.assertEqual(record["result_kind"], "incomplete_or_unknown")
        self.assertFalse(record["training_ready"])

    def test_explicit_passes_and_result_are_separate(self):
        record = audit(";B[aa];W[bb];B[];W[]", "B+2.5")
        self.assertEqual(record["parse_status"], "parsed")
        self.assertEqual(record["counters"]["explicit_passes"], 2)
        self.assertTrue(record["final_two_explicit_passes"])
        self.assertEqual(record["result_kind"], "scored")
        self.assertEqual(record["exclusions"], [])

    def test_missing_clock_breaks_elapsed_chain_and_overtime_is_ambiguous(self):
        record = audit(";B[aa]BL[1190];W[bb];B[cc];W[dd]WL[1180];B[ee]BL[1170];W[ff]WL[20]OW[5]")
        counts = record["counters"]
        self.assertEqual(counts["main_time_difference_candidates"], 1)
        self.assertEqual(counts["post_clock_missing"], 2)
        self.assertEqual(counts["overtime_marker_moves"], 1)
        self.assertEqual(record["validated_elapsed_labels"], 0)

    def test_collection_and_trailing_garbage_are_not_silently_accepted(self):
        for data in (b"(;GM[1])(;GM[1])", b"(;GM[1])garbage", b"(;GM[1];B[aa]", b"<html>login</html>"):
            self.assertEqual(inspect_sgf(data)["parse_status"], "corrupt")

    def test_escaped_comment_is_not_a_move_or_tree_boundary(self):
        record = audit(extra=r"C[text \] (; B[zz\] text]")
        self.assertEqual(record["parse_status"], "parsed")
        self.assertEqual(record["counters"]["moves"], 2)

    def test_unknown_human_status_and_historical_bots(self):
        self.assertEqual(audit()["human_status"], "unverified")
        record = inspect_sgf(f"(;{ROOT};B[aa])".encode(), bot_handles=["a"])
        self.assertIn("historical_bot_handle", record["exclusions"])

    def test_repeated_move_and_scalar_values_are_corrupt(self):
        self.assertEqual(audit(";B[aa][bb]")["parse_status"], "corrupt")
        self.assertEqual(audit(extra="KM[7.5]")["parse_status"], "corrupt")

    def test_variations_are_reported_and_leftmost_moves_counted(self):
        record = audit(";B[aa](;W[bb])(;W[cc];B[])")
        self.assertEqual(record["counters"]["moves"], 2)
        self.assertEqual(record["counters"]["explicit_passes"], 0)
        self.assertIn("played_variation_unverified", record["exclusions"])

    def test_timeout_and_forfeit_are_separate_from_scored_results(self):
        self.assertEqual(result_kind("W+Time"), "timeout")
        self.assertEqual(result_kind("B+Forfeit"), "forfeit")
        self.assertEqual(result_kind("W+Res."), "resignation")

    def test_invalid_clock_is_masked_without_discarding_moves(self):
        record = audit(";B[aa]BL[nan];W[bb]WL[garbage]")
        self.assertEqual(record["parse_status"], "parsed")
        self.assertEqual(record["counters"]["post_clock_invalid"], 2)
        self.assertEqual(record["counters"]["main_time_difference_candidates"], 0)

    def test_archive_result_comparison_includes_winner_and_margin(self):
        source = dict(white="b [5k]", black="a [5k]", result="W+Res.",
                      url="https://files.gokgs.com/games/2025/1/1/b-a.sgf")
        for sgf_result, archive_result in (("B+R", "W+Res."), ("B+1.5", "B+2.5")):
            record = inspect_sgf(f"(;{ROOT}RE[{sgf_result}];B[aa];W[bb])".encode(),
                                 {**source, "result": archive_result})
            self.assertIn("archive_sgf_result_mismatch", record["exclusions"])
        equivalent = inspect_sgf(f"(;{ROOT}RE[B+R];B[aa];W[bb])".encode(), {**source, "result": "B+Res."})
        self.assertNotIn("archive_sgf_result_mismatch", equivalent["exclusions"])


if __name__ == "__main__":
    unittest.main()
