"""Set KENGGO_KATAGO_REFERENCE to require the pinned native comparison."""

import json
import os
from pathlib import Path
import subprocess
import unittest

from kenggo.adjudication import adjudicate
from kenggo.go import Game, IllegalMove, PASS
from go_fixtures import CYCLE, cap_actions, dense, ko_patches, ordinary_cases, point, two_passes


REFERENCE = os.environ.get("KENGGO_KATAGO_REFERENCE")


@unittest.skipUnless(REFERENCE, "Set KENGGO_KATAGO_REFERENCE; native conformance is a separate required check")
class ReferenceTests(unittest.TestCase):
    def oracle(self, cases):
        result = subprocess.run([str(Path(REFERENCE).resolve())],
                                input="".join(json.dumps(c) + "\n" for c in cases),
                                text=True, capture_output=True, check=True, timeout=60)
        self.assertEqual(result.stderr, "")
        output = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(output), len(cases))
        return output

    def assert_position(self, game, state, *, mask=True):
        position = game.position
        self.assertEqual(position.rows, tuple(state["board"]))
        self.assertEqual(position.to_play, state["to_play"])
        self.assertEqual(position.ko, state["simple_ko"])
        self.assertEqual({"B": position.captured_black, "W": position.captured_white}, state["captures"])
        self.assertEqual(position.pass_streak, state["pass_streak"])
        self.assertEqual(len(game.moves), state["move_count"])
        if mask and game.status == "playing":
            self.assertEqual(game.legal_mask(), tuple(state["legal_mask"]))

    def test_ordinary_states_and_all_362_legal_actions(self):
        fixtures = ordinary_cases()
        cases = [{"board": c["board"], "include_legal_mask": True,
                  "moves": [{"color": color, "action": action} for color, action in c["moves"]]}
                 for c in fixtures]
        for fixture, output in zip(fixtures, self.oracle(cases)):
            with self.subTest(name=fixture["name"]):
                self.assertTrue(output["ok"])
                game = Game.from_fixture(fixture["board"])
                self.assert_position(game, output["initial"])
                for (color, action), native in zip(fixture["moves"], output["steps"]):
                    accepted = True
                    try:
                        game = game.play(color, action)
                    except IllegalMove:
                        accepted = False
                    self.assertEqual(accepted, native["accepted"])
                    self.assert_position(game, native["state"])

    def test_native_cleanup_is_diagnostic_and_scores_settled_fixture(self):
        for captured in ({"B": 0, "W": 0}, {"B": 3, "W": 1}, {"B": 0, "W": 8}):
            initial = Game.from_fixture(dense(), captured_black=captured["B"], captured_white=captured["W"])
            actual = two_passes(initial)
            ended, _ = adjudicate(actual)
            output = self.oracle([{"board": dense(), "captures": captured, "mode": "history",
                                   "moves": [{"color": "B" if i % 2 == 0 else "W", "action": PASS}
                                             for i in range(7)]}])[0]
            self.assertEqual([output["steps"][i]["state"]["encore_phase"] for i in (1, 3, 5)], [1, 2, 2])
            self.assertEqual([output["steps"][i]["state"]["finished"] for i in (1, 3, 5)], [False, False, True])
            self.assertFalse(output["steps"][6]["accepted"])
            self.assertEqual(ended.outcome.white_minus_black_halfpoints, output["final"]["white_minus_black_halfpoints"])
            self.assertEqual(len(actual.moves), 2)
            self.assertEqual(actual.status, "pending_adjudication")
        # Even a formally finished reference result cannot certify our empty ending.
        empty_native = self.oracle([{"mode": "history", "moves": [
            {"color": "B" if i % 2 == 0 else "W", "action": PASS} for i in range(6)]}])[0]
        self.assertTrue(empty_native["final"]["finished"])
        empty, _ = adjudicate(two_passes(Game.new()))
        self.assertEqual(empty.status, "unresolved")

    def test_native_history_repetition_and_clones_are_not_superko(self):
        prefix = [{"color": "B" if i % 2 == 0 else "W", "action": a} for i, a in enumerate(CYCLE)]
        history = self.oracle([{"board": ko_patches(True), "mode": "history", "moves": prefix,
                                "branches": [prefix, [{"color": "B", "action": PASS},
                                                       {"color": "W", "action": PASS}]]}])[0]
        self.assertFalse(history["final"]["finished"])
        self.assertTrue(history["branches"][0]["final"]["no_result"])
        self.assertIsNone(history["branches"][0]["final"]["white_minus_black_halfpoints"])
        self.assertFalse(history["branches"][1]["final"]["no_result"])
        self.assertEqual(history["final"]["move_count"], 6)
        adapter = Game.from_fixture(ko_patches(True))
        for action in CYCLE * 2:
            adapter = adapter.play(adapter.position.to_play, action)
        self.assertEqual(adapter.status, "playing")
        self.assertIsNone(adapter.outcome)

    def test_setup_suicide_and_801_action_capture_history(self):
        bad = self.oracle([{"board": ["B" * 19] * 19}])[0]
        self.assertFalse(bad["ok"])
        actions = cap_actions()[:801]
        native = self.oracle([{"board": ko_patches(), "moves": [
            {"color": "B" if i % 2 == 0 else "W", "action": a} for i, a in enumerate(actions)]}])[0]
        game = Game.from_fixture(ko_patches(), candidate_color="W")
        for action, step in zip(actions, native["steps"]):
            self.assertTrue(step["accepted"])
            game = game.play(game.position.to_play, action)
            self.assert_position(game, step["state"], mask=False)
        self.assertEqual(game.status, "truncated")
        self.assertFalse(native["final"]["finished"])

    def test_dead_chain_cleanup_and_seki_reference(self):
        cases = [
            {"board": dense("dead"), "mode": "history", "moves": [
                {"color": "B", "action": point(9, 6)}] + [
                {"color": "W" if i % 2 == 0 else "B", "action": PASS} for i in range(6)]},
            {"board": dense("seki"), "mode": "history", "moves": [
                {"color": "B" if i % 2 == 0 else "W", "action": PASS} for i in range(6)]},
        ]
        dead, seki = self.oracle(cases)
        self.assertEqual(dead["final"]["captures"]["W"], 2)
        # Playing the capture during ordinary play consumes a territory point.
        self.assertEqual(dead["final"]["white_minus_black_halfpoints"], 5)
        self.assertEqual(seki["final"]["white_minus_black_halfpoints"], 13)


if __name__ == "__main__":
    unittest.main()
