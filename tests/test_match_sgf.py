import base64
import json
import unittest

from sgfmill import sgf

from kenggo.adjudication import adjudicate, adjudicate_empirical
from kenggo.go import Game, PASS, canonical, digest
from kenggo.match_sgf import export_match_sgf, replay_match_sgf
from go_fixtures import CYCLE, dense, ko_patches, two_passes
from test_matches import FakeReferee


class MatchSgfTests(unittest.TestCase):
    def test_coordinates_passes_and_untimed_metadata(self):
        game = Game.new(candidate_color="W")
        for color, action in (("B", 0), ("W", 342), ("B", 18), ("W", 360), ("B", PASS), ("W", PASS)):
            game = game.play(color, action)
        game, _ = adjudicate(game)
        data = export_match_sgf(game, metadata={"slot": "fixture"})
        for expected in (b";B[aa]", b";W[as]", b";B[sa]", b";W[ss]", b";B[]", b";W[]"):
            self.assertIn(expected, data)
        root = sgf.Sgf_game.from_bytes(data).get_root()
        self.assertFalse({"TM", "OT", "RE"}.intersection(root.properties()))
        self.assertEqual(replay_match_sgf(data), game)

    def test_exact_signed_halfpoints_resign_and_forfeit_are_distinct(self):
        for captures, expected in ((0, "W+6.5"), (8, "B+1.5")):
            game, _ = adjudicate(two_passes(Game.from_fixture(dense(), captured_white=captures, candidate_color="B")))
            data = export_match_sgf(game)
            self.assertEqual(sgf.Sgf_game.from_bytes(data).get_root().get("RE"), expected)
            self.assertEqual(replay_match_sgf(data, allow_fixture=True), game)
            with self.assertRaises(ValueError):
                replay_match_sgf(data)
        for game, expected in ((Game.new(candidate_color="B").resign("B"), "W+R"),
                               (Game.new(candidate_color="B").fail("illegal_action", responsible="B"), "W+F")):
            data = export_match_sgf(game)
            self.assertEqual(sgf.Sgf_game.from_bytes(data).get_root().get("RE"), expected)
            self.assertEqual(replay_match_sgf(data), game)
            self.assertEqual(len(game.moves), 0)

    def test_full_history_recurrence_and_candidate_color(self):
        game = Game.from_fixture(ko_patches(True), candidate_color="B")
        for action in CYCLE:
            game = game.play(game.position.to_play, action)
        restored = replay_match_sgf(export_match_sgf(game), allow_fixture=True)
        self.assertEqual(restored, game)
        self.assertEqual(restored.pending_cap, "recurrence")
        self.assertEqual(len(restored.positions), 7)

    def test_actions_outcome_candidate_metadata_and_variation_tampering(self):
        game = Game.new(candidate_color="B").play("B", 0).resign("W")
        original = export_match_sgf(game)
        for data in (original.replace(b";B[aa]", b";B[ab]"),
                     original.replace(b"RE[B+R]", b"RE[W+R]"),
                     original[:-2] + b"(;W[cc])(;W[dd]))",
                     original.replace(b"KM[6.5]", b"KM[6.5]TM[1200]")):
            with self.assertRaises(ValueError):
                replay_match_sgf(data)
        tree = sgf.Sgf_game.from_bytes(original)
        record = json.loads(base64.b64decode(tree.get_root().get_raw("KGS")))
        record["candidate_color"] = "W"
        tree.get_root().set_raw("KGS", base64.b64encode(canonical(record).encode()))
        with self.assertRaisesRegex(ValueError, "candidate color"):
            replay_match_sgf(tree.serialise())

    def test_empirical_sgf_requires_external_catalog_and_configured_referee(self):
        pending = two_passes(Game.new(candidate_color="B").play("B", 20).play("W", 21))
        game, _ = adjudicate_empirical(pending, FakeReferee())
        data = export_match_sgf(game)
        for kwargs in ({}, {"empirical_referee": FakeReferee()},
                       {"authorized_hashes": {digest(json.loads(game.outcome.evidence))}}):
            with self.assertRaises(ValueError):
                replay_match_sgf(data, **kwargs)
        restored = replay_match_sgf(data, empirical_referee=FakeReferee(),
                                    authorized_hashes={digest(json.loads(game.outcome.evidence))})
        self.assertEqual(restored, game)
        self.assertEqual(len(restored.moves), 4)
        self.assertEqual(sgf.Sgf_game.from_bytes(data).get_root().get("RE"), "W+5.5")


if __name__ == "__main__":
    unittest.main()
