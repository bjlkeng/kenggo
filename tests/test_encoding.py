import unittest

import numpy as np

from kenggo.encoding import Context, SCALARS, encode
from kenggo.go import Game, PASS
from go_fixtures import CYCLE, ko_patches, point


class EncodingTests(unittest.TestCase):
    def test_entire_approved_nominal_clock_family_retains_exact_tuple(self):
        for clock in ((600, 60, 3), (1800, 20, 5), (900.5, 30.5, 4)):
            with self.subTest(clock=clock):
                inputs = encode(Game.new(), Context(nominal_clock=clock))
                self.assertEqual(inputs["scalars"][9:12].tolist(), list(clock))
                self.assertTrue(inputs["scalar_valid"][9:12].all())
                self.assertFalse(inputs["scalar_valid"][12:].any())

    def test_corners_mover_view_and_action_alignment(self):
        game = Game.new()
        for action in (0, 18, 342, 360):
            before = encode(game)
            self.assertTrue(before["legal"][action])
            game = game.play(game.position.to_play, action)
        inputs = encode(game, Context("5k", "1d", (1200, 30, 5)))
        self.assertTrue(inputs["board"][0, 0, 0])
        self.assertTrue(inputs["board"][0, 18, 0])
        self.assertTrue(inputs["board"][1, 0, 18])
        self.assertTrue(inputs["board"][1, 18, 18])
        self.assertFalse(inputs["legal"][[0, 18, 342, 360]].any())
        self.assertTrue(inputs["legal"][PASS])
        self.assertEqual(inputs["scalars"][SCALARS.index("mover_komi")], -6.5)
        white = encode(game.play("B", PASS), Context("5k", "1d"))
        np.testing.assert_array_equal(inputs["board"][0], white["board"][1])
        np.testing.assert_array_equal(inputs["ranks"], white["ranks"][::-1])
        self.assertEqual(white["scalars"][1], 6.5)

    def test_prisoners_count_captured_opponent_stones(self):
        game = Game.new().play("B", 1).play("W", 0).play("B", 19)
        white = encode(game)
        self.assertEqual(white["scalars"][2:4].tolist(), [0, 1])
        black = encode(game.play("W", 200))
        self.assertEqual(black["scalars"][2:4].tolist(), [1, 0])

    def test_full_ko_recurrence_and_pass_state_survive_short_windows(self):
        game = Game.from_fixture(ko_patches(True))
        for action in CYCLE:
            game = game.play(game.position.to_play, action)
        snapshot = game.snapshot
        self.assertEqual(game.pending_cap, "recurrence")
        baseline = encode(game)
        for window in (0, 8, 32, 64):
            inputs = encode(game, history=window)
            for name in ("board", "legal", "scalars", "scalar_valid"):
                np.testing.assert_array_equal(inputs[name], baseline[name])
            self.assertEqual(int(inputs["history_valid"].sum()), min(window, 6))
            if window:
                self.assertEqual(inputs["history_actions"][-1], CYCLE[-1])
                self.assertEqual(inputs["history_colors"][-1], -1)
                self.assertTrue((inputs["history_actions"][~inputs["history_valid"]] == -1).all())
        self.assertEqual(game.snapshot, snapshot)
        self.assertFalse(baseline["legal"][point(10, 4)])
        passed = encode(game.play(game.position.to_play, PASS), history=8)
        self.assertEqual(passed["history_actions"][-1], PASS)
        self.assertTrue(passed["history_valid"][-1])
        self.assertEqual(passed["scalars"][4], 1)
        self.assertFalse(passed["board"][2].any())

    def test_explicit_missing_context_and_immutable_outputs(self):
        game = Game.new()
        unknown = encode(game)
        self.assertFalse(unknown["rank_valid"].any())
        self.assertTrue((unknown["ranks"] == 0).all())
        self.assertFalse(unknown["scalar_valid"][9:].any())
        known = encode(game, Context("15k", "1d", (1200, 30, 5)))
        self.assertEqual(known["ranks"].tolist(), [1, 16])
        self.assertEqual(known["scalars"][9:12].tolist(), [1200, 30, 5])
        self.assertFalse(known["scalar_valid"][12:].any())
        known["board"][:] = True
        self.assertFalse(encode(game)["board"].any())
        for context in (("5k?", "1d"), ("16k", None)):
            with self.assertRaises(ValueError):
                Context(*context)
        with self.assertRaises(ValueError):
            Context(nominal_clock=(1200, 5, 30))
        with self.assertRaises(ValueError):
            encode(game, history=1)

    def test_lifecycle_requires_pre_move_or_saved_bootstrap_state(self):
        game = Game.new(candidate_color="B").play("B", 0).play("W", 1)
        capped = game.truncate("wall_limit", candidate_color="B")
        inputs = encode(capped)
        self.assertEqual(inputs["scalars"][6:9].tolist(), [1, 1, 0])
        self.assertFalse(inputs["legal"].any())
        np.testing.assert_array_equal(inputs["board"], encode(game)["board"])
        for end in (game.resign("B"), game.play("B", PASS).play("W", PASS), game.fail("fault")):
            with self.assertRaises(ValueError):
                encode(end)


if __name__ == "__main__":
    unittest.main()
