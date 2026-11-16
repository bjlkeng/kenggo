"""Synthetic checks for the opt-in 0.5-point contract and retained 6.5 evidence."""

from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import unittest
from unittest.mock import patch

import numpy as np

from kenggo.adjudication import adjudicate, adjudicate_empirical, judge_pass
from kenggo.encoding import Context, encode
from kenggo.go import Game, KOMI_STUDY_RULES, PASS, RULES, digest
from kenggo.rules_replay import dump_game, load_game, replay_sgf
from go_fixtures import CYCLE, dense, ko_patches, neutral_decision, point, two_passes


HEADER = b"(;GM[1]FF[4]SZ[19]RU[Japanese]KM[0.5]TM[1200]OT[5x30 byo-yomi]"


class KomiRulesTests(unittest.TestCase):
    def test_game_komi_is_exact_validated_and_immutable(self):
        for value in (True, False, 1.0, 13.0, "1", None, 0, 2, 7, 15, -1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    Game.new(komi_halfpoints=value)
                with self.assertRaises(ValueError):
                    Game.from_fixture(dense(), komi_halfpoints=value)
                with self.assertRaises(ValueError):
                    replace(Game.new(), komi_halfpoints=value)
        game = Game.new(komi_halfpoints=1)
        self.assertEqual(game.rules, KOMI_STUDY_RULES)
        self.assertEqual(Game.new().rules, RULES)
        with self.assertRaises(FrozenInstanceError):
            game.komi_halfpoints = 13

    def test_default_state_hashes_and_scalar_bytes_match_historical_pins(self):
        cases = [
            (Game.new(), "44eea4bf72c9797415716a867864000c08853d1dc9a16ca0650dc9b0aa369fe5",
             "00c47e5e3c768b39009dc209e1f5e5cfc9edc85860d0d193fae016735815f0c1",
             "63400ada6a1e88a698b3c33e905367be5b9bfc622706146384af5db459495972"),
            (Game.new().play("B", 1).play("W", 0).play("B", 19),
             "b0d64601accea8ba963a569d8170aca77b338eec41d6fe9e6f8723d79e17ea91",
             "366b7245ca9ee0b177e9b6913463f0409da94211900a498dc4de3ea2f2d2ff6a",
             "25fdbfa409d3dfe43b342573d771f3ed70dec3f60870aa961220512ed7075827"),
        ]
        for game, snapshot, state_hash, scalars_hash in cases:
            with self.subTest(moves=len(game.moves)):
                self.assertEqual(game.komi_halfpoints, 13)
                self.assertNotIn("komi_halfpoints", game.action_record())
                self.assertEqual(game.snapshot, snapshot)
                self.assertEqual(hashlib.sha256(dump_game(game).encode()).hexdigest(), state_hash)
                row = encode(game, Context("5k", "1d", (1200, 30, 5)), history=8)
                self.assertEqual(hashlib.sha256(row["scalars"].tobytes()).hexdigest(), scalars_hash)
                self.assertEqual(load_game(dump_game(game)), game)
                self.assertEqual(load_game(dump_game(game), komi_study=True), game)

    def test_both_movers_encode_game_komi_and_preserve_other_inputs(self):
        low, old = Game.new(komi_halfpoints=1), Game.new()
        for action in (1, 0, 19, PASS):
            low_row, old_row = encode(low, history=8), encode(old, history=8)
            sign = -1 if low.position.to_play == "B" else 1
            self.assertEqual(low_row["scalars"][1], sign * 0.5)
            self.assertEqual(old_row["scalars"][1], sign * 6.5)
            self.assertTrue(low_row["scalar_valid"][1])
            low_row["scalars"][1] = old_row["scalars"][1]
            for key in old_row:
                np.testing.assert_array_equal(low_row[key], old_row[key])
            low = low.play(low.position.to_play, action)
            old = old.play(old.position.to_play, action)

    def test_clones_replay_full_history_and_snapshot_separation(self):
        low, old = (Game.from_fixture(ko_patches(True), komi_halfpoints=komi) for komi in (1, 13))
        for action in CYCLE * 2:
            low = low.play(low.position.to_play, action)
            old = old.play(old.position.to_play, action)
        self.assertEqual(low.positions, old.positions)
        self.assertEqual(low.moves, old.moves)
        self.assertEqual(low.legal_mask(), old.legal_mask())
        self.assertEqual(low.pending_cap, "recurrence")
        self.assertNotEqual(low.snapshot, old.snapshot)
        self.assertEqual(low.action_record()["komi_halfpoints"], 1)
        restored = load_game(dump_game(low), allow_fixture=True, komi_study=True)
        self.assertEqual(restored, low)
        for window in (0, 8, 32, 64):
            encode(low, history=window)
            branch = low.clone().play("B", PASS)
            self.assertEqual(branch.komi_halfpoints, 1)
            self.assertEqual(branch.moves[:-1], low.moves)
            self.assertEqual(len(low.positions), 13)
            self.assertEqual(branch.segment_start, 13)
        with self.assertRaisesRegex(ValueError, "komi-study"):
            load_game(dump_game(low), allow_fixture=True)

    def test_serialized_rules_and_komi_cannot_disagree(self):
        low, old = json.loads(dump_game(Game.new(komi_halfpoints=1))), json.loads(dump_game(Game.new()))
        changes = [{**low, "komi_halfpoints": value} for value in (13, 1.0, True, 0)]
        changes += [{**low, "rules": RULES}, {**old, "komi_halfpoints": 1},
                    {key: value for key, value in low.items() if key != "komi_halfpoints"}]
        for record in changes:
            with self.subTest(record=record), self.assertRaises(ValueError):
                load_game(json.dumps(record), komi_study=True)

    def test_source_opt_in_is_narrow_and_preserves_actual_moves_and_results(self):
        data = HEADER + b"RE[B+R];B[aa];W[bb];B[];W[tt])"
        with self.assertRaises(ValueError):
            replay_sgf(data)
        result = replay_sgf(data, komi_study=True)
        self.assertEqual(result.game.komi_halfpoints, 1)
        self.assertEqual(result.game.status, "pending_adjudication")
        self.assertEqual(result.raw_moves, ("aa", "bb", "", "tt"))
        self.assertEqual((result.source_result, result.source_winner), ("B+R", "B"))
        self.assertIsNone(result.game.outcome)
        old_data = data.replace(b"KM[0.5]", b"KM[6.5]")
        self.assertEqual(replay_sgf(old_data), replay_sgf(old_data, komi_study=True))
        for invalid in (data.replace(b"0.5", b"7.5"), data.replace(b"Japanese", b"Chinese"),
                        data.replace(b"RE[B+R]", b"HA[2]RE[B+R]"),
                        data.replace(b";B[aa]", b";W[aa]")):
            with self.subTest(data=invalid), self.assertRaises(ValueError):
                replay_sgf(invalid, komi_study=True)
        for flag in (1, "yes", None):
            with self.assertRaises(ValueError):
                replay_sgf(data, komi_study=flag)
            with self.assertRaises(ValueError):
                load_game(dump_game(Game.new()), komi_study=flag)

    def test_source_budget_can_stop_mid_game_without_swallowing_the_deadline(self):
        data = HEADER + b";B[aa];W[bb];B[cc];W[dd])"
        original_play, played = Game.play, []
        deadline = TimeoutError("replay budget exhausted")

        def play(game, color, action):
            played.append((color, action))
            return original_play(game, color, action)

        def check_budget():
            if len(played) >= 2:
                raise deadline

        with patch.object(Game, "play", new=play):
            with self.assertRaises(TimeoutError) as caught:
                replay_sgf(data, komi_study=True, check_budget=check_budget)
        self.assertIs(caught.exception, deadline)
        self.assertEqual(len(played), 2)
        self.assertEqual(replay_sgf(data, komi_study=True),
                         replay_sgf(data, komi_study=True, check_budget=lambda: None))

    def test_territory_prisoners_and_both_value_signs_change_with_komi(self):
        board = list("".join(dense()))
        board[point(7, 14)] = "."
        rows = ["".join(board[i:i + 19]) for i in range(0, PASS, 19)]
        for komi, expected in ((1, -1), (13, 11)):
            frozen = two_passes(Game.from_fixture(rows, captured_black=2, captured_white=4,
                                                  komi_halfpoints=komi))
            ended, decision = adjudicate(frozen)
            self.assertEqual(ended.status, "resolved")
            self.assertEqual(decision.white_minus_black_halfpoints, expected)
            self.assertEqual(ended.outcome.value("W"), 1 if expected > 0 else -1)
            self.assertEqual(ended.outcome.value("B"), -ended.outcome.value("W"))
            self.assertEqual(ended.positions, frozen.positions)
            self.assertEqual(ended.moves, frozen.moves)
            self.assertEqual(load_game(dump_game(ended), allow_fixture=True, komi_study=True), ended)

    def test_old_evidence_cannot_score_the_new_contract(self):
        old, _ = adjudicate(two_passes(Game.from_fixture(dense())))
        low = two_passes(Game.from_fixture(dense(), komi_halfpoints=1))
        forged = json.loads(dump_game(low))
        forged["ending"] = json.loads(dump_game(old))["ending"]
        with self.assertRaisesRegex(ValueError, "snapshot"):
            load_game(json.dumps(forged), allow_fixture=True, komi_study=True)
        payload = neutral_decision(low, "settled")
        rejected, _ = adjudicate(low, neutral=payload, authorized_hashes={digest(payload)})
        self.assertEqual(rejected.status, "unresolved")
        self.assertIn("rules", rejected.outcome.reason)

    def test_dead_stones_seki_and_uncertified_endings(self):
        for kind, expected in (("dead", -9), ("seki", 1)):
            game = two_passes(Game.from_fixture(dense(kind), komi_halfpoints=1))
            unknown, _ = adjudicate(game)
            self.assertEqual(unknown.status, "unresolved")
            payload = neutral_decision(game, kind)
            payload["rules"] = game.rules
            authorized = {digest(payload)}
            ended, _ = adjudicate(game, neutral=payload, authorized_hashes=authorized)
            self.assertEqual(ended.outcome.white_minus_black_halfpoints, expected)
            self.assertEqual(load_game(dump_game(ended), allow_fixture=True, komi_study=True,
                                       authorized_hashes=authorized), ended)
        unknown, _ = adjudicate(two_passes(Game.new(komi_halfpoints=1)))
        self.assertEqual(unknown.status, "unresolved")
        self.assertIsNone(unknown.outcome.value("B"))

    def test_pass_proof_uses_both_komi_values_and_empirical_path_rejects_new_one(self):
        for komi in (1, 13):
            settled = Game.from_fixture(dense(), komi_halfpoints=komi)
            self.assertEqual(judge_pass(settled).verdict, "acceptable")
            saving = Game.from_fixture(dense("save"), komi_halfpoints=komi)
            judgment = judge_pass(saving)
            self.assertEqual(judgment.verdict, "premature")
            self.assertEqual(json.loads(judgment.witness)["gain_lower_halfpoints"], 4)
        with self.assertRaisesRegex(ValueError, "6.5-point komi"):
            adjudicate_empirical(two_passes(Game.new(komi_halfpoints=1)), object())


if __name__ == "__main__":
    unittest.main()
