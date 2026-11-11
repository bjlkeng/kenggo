import hashlib
import json
import unittest

from kenggo.adjudication import adjudicate
from kenggo.canonical_data import adjudication_evidence, canonicalize as canonicalize_with_cost
from kenggo.go import Game
from go_fixtures import dense, two_passes


def canonicalize(*args):
    return canonicalize_with_cost(*args)[0]


def source(moves=";B[aa];W[bb]", result="B+R", archive_result=None, day="2025-07-02", extra=""):
    re = "" if result is None else f"RE[{result}]"
    data = (f"(;FF[4]GM[1]SZ[19]PB[Alice]PW[Bob]BR[5k]WR[1d]RU[Japanese]KM[6.5]"
            f"TM[1200]OT[5x30 byo-yomi]DT[{day}]{re}{extra}{moves})").encode()
    url = "https://files.gokgs.com/games/" + day.replace("-", "/") + "/fixture.sgf"
    ref = {"source_url": url, "sgf_sha256": hashlib.sha256(data).hexdigest(), "zip_member": "fixture.sgf"}
    archive = {"url": url, "black": "Alice [5k]", "white": "Bob [1d]",
               "type": "Ranked", "result": result if archive_result is None else archive_result}
    return data, ref, archive


class CanonicalTests(unittest.TestCase):
    def test_complete_evidence_and_eof_without_synthetic_actions(self):
        data, ref, archive = source(archive_result="B+Res.")
        row = canonicalize(data, ref, archive)
        self.assertTrue(row["replay_validated"])
        self.assertEqual(row["structural_exclusions"], [])
        self.assertEqual(row["source"], ref)
        self.assertEqual(row["archive"], archive)
        self.assertEqual([(a["color"], a["action"]) for a in row["actions"]], [("B", 0), ("W", 20)])
        self.assertEqual(row["replay_end_status"], "playing")
        self.assertIsNone(row["certification"])
        self.assertEqual(row["recorded_y_black"], 1)
        self.assertTrue(row["recorded_value_supported"])
        self.assertIsNone(row["source_timezone"])
        self.assertEqual(row["date_precision"], "day")

    def test_raw_clock_presence_and_parse_masks_never_grant_timing_labels(self):
        row = canonicalize(*source(moves=";C[note]WL[1100];B[aa]BL[1000.5]OB[3]WL[bad];W[bb]WL[-1]OW[1][2]"))
        self.assertTrue(row["replay_validated"])
        self.assertFalse(row["duration_valid"])
        self.assertFalse(row["dynamic_clock_input_valid"])
        self.assertEqual(row["clock_nodes"][0]["node"], 1)
        clocks = row["clock_nodes"][1]["properties"]
        self.assertEqual(clocks["BL"]["value"], 1000.5)
        self.assertEqual(bytes.fromhex(clocks["WL"]["raw_hex"][0]), b"bad")
        self.assertFalse(clocks["WL"]["parse_valid"])
        self.assertNotIn("OW", clocks)
        self.assertFalse(row["clock_nodes"][2]["properties"]["OW"]["parse_valid"])

    def test_normalized_passes_preserve_original_encoding(self):
        empty = canonicalize(*source(moves=";B[];W[]"))
        legacy = canonicalize(*source(moves=";B[tt];W[tt]"))
        self.assertEqual(empty["trajectory_sha256"], legacy["trajectory_sha256"])
        self.assertEqual(empty["semantic_sha256"], legacy["semantic_sha256"])
        self.assertNotEqual(empty["actions"][0]["raw_hex"], legacy["actions"][0]["raw_hex"])
        self.assertEqual(empty["replay_end_status"], "pending_adjudication")
        self.assertEqual(empty["certification"]["status"], "unresolved")

    def test_adjudication_keeps_method_snapshot_and_budget_evidence(self):
        row, measured = canonicalize_with_cost(*source(moves=";B[];W[]"))
        evidence = row["certification"]
        self.assertEqual(evidence["method"], "singleton_eyes_v1")
        self.assertEqual(evidence["snapshot"], row["replay_snapshot"])
        self.assertIsNone(evidence["payload"])
        self.assertEqual(evidence["cost"]["node_limit"], 100000)
        self.assertGreater(evidence["cost"]["nodes"], 0)
        self.assertNotIn("cpu_seconds", evidence["cost"])
        self.assertNotIn("wall_seconds", evidence["cost"])
        self.assertGreaterEqual(measured["cpu_seconds"], 0)
        self.assertGreaterEqual(measured["wall_seconds"], 0)
        self.assertEqual(evidence["cost"], {k: v for k, v in measured.items()
                                          if k not in {"cpu_seconds", "wall_seconds"}})

    def test_successful_certificate_payload_survives_with_measured_costs(self):
        game = two_passes(Game.from_fixture(dense()))
        checked, decision = adjudicate(game)
        evidence, measured = adjudication_evidence(checked, decision)
        self.assertEqual(evidence["status"], "resolved")
        self.assertEqual(evidence["white_minus_black_halfpoints"], 13)
        self.assertEqual(evidence["payload"], decision.payload)
        self.assertEqual(evidence["snapshot"], game.snapshot)
        self.assertEqual(len(json.loads(evidence["payload"])["territory_black"]), 2)
        self.assertEqual(measured, json.loads(decision.cost))
        again, repeat_decision = adjudicate(game)
        self.assertEqual(adjudication_evidence(again, repeat_decision)[0], evidence)

    def test_illegal_or_ambiguous_records_never_salvage_a_training_prefix(self):
        for moves in (";B[aa];W[aa]", ";W[aa]", ";B[];W[];B[aa]", ";B[aa](;W[bb])(;W[cc])"):
            with self.subTest(moves=moves):
                row = canonicalize(*source(moves=moves))
                self.assertFalse(row["replay_validated"])
                self.assertIn("invalid_or_unsupported_replay", row["structural_exclusions"])
                self.assertFalse(row["recorded_value_supported"])
        setup = canonicalize(*source(extra="AB[dd]"))
        self.assertIn("setup_or_handicap", setup["structural_exclusions"])
        self.assertFalse(setup["replay_validated"])

    def test_outcome_masks_are_separate_from_policy_structure(self):
        for result, archived in ((None, "W+Forf."), ("W+T", "W+Time"), ("B+F", "B+Forfeit"), ("?", "?"), ("0", "0")):
            with self.subTest(result=result):
                row = canonicalize(*source(result=result, archive_result=archived))
                self.assertTrue(row["replay_validated"])
                self.assertEqual(row["structural_exclusions"], [])
                self.assertFalse(row["recorded_value_supported"])
                self.assertIsNone(row["recorded_y_black"])
        scored = canonicalize(*source(result="W+2.5"))
        self.assertEqual(scored["recorded_y_black"], -1)
        for result, archived in (("B+R", "W+R"), ("B+0.5", "B+1.5"), ("W+T", "W+R")):
            row = canonicalize(*source(result=result, archive_result=archived))
            self.assertIn("conflicting_known_results", row["structural_exclusions"])
            self.assertFalse(row["recorded_value_supported"])
        self.assertIn("score_margin_out_of_contract", canonicalize(*source(result="W+2"))["structural_exclusions"])

    def test_case_equivalence_and_conflicting_provenance(self):
        data, ref, archive = source()
        archive["black"] = "ALICE [5k]"
        self.assertEqual(canonicalize(data, ref, archive)["structural_exclusions"], [])
        archive["black"] = "Carol [5k]"
        self.assertIn("black_archive_sgf_mismatch", canonicalize(data, ref, archive)["structural_exclusions"])
        self.assertIn("date_unknown_or_out_of_scope", canonicalize(*source(day="2025-02-31"))["structural_exclusions"])
        with self.assertRaises(ValueError):
            canonicalize(data + b" ", ref, archive)
        with self.assertRaises(ValueError):
            canonicalize(data, ref, {**archive, "url": "other"})


if __name__ == "__main__":
    unittest.main()
