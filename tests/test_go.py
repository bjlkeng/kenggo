import copy
from dataclasses import FrozenInstanceError, replace
import json
import unittest

from kenggo.adjudication import Limits, adjudicate, judge_pass
from kenggo.go import Game, IllegalMove, Outcome, PASS, digest, move_category, require_valid_batch
from kenggo.rules_replay import dump_game, load_game, replay_sgf
from go_fixtures import (CYCLE, cap_actions, dense, ko_patches, neutral_decision,
                         ordinary_cases, point, rows, two_passes)


HEADER = b"(;GM[1]FF[4]SZ[19]RU[Japanese]KM[6.5]TM[1200]OT[5x30 byo-yomi]"


class RulesTests(unittest.TestCase):
    def test_capture_precedes_suicide(self):
        case = ordinary_cases()[0]
        game = Game.from_fixture(case["board"])
        following = game.play(*case["moves"][0])
        self.assertEqual(following.position.captured_white, 4)
        self.assertEqual(following.position.board[point(9, 9)], "B")
        self.assertEqual(game.position.captured_white, 0)
        self.assertEqual(move_category(game, point(9, 9)), "capture")

    def test_illegal_moves_are_atomic(self):
        for case in ordinary_cases()[1:3]:
            with self.subTest(case=case["name"]):
                game = Game.from_fixture(case["board"])
                before = dump_game(game)
                self.assertFalse(game.legal_mask()[point(9, 9)])
                with self.assertRaises(IllegalMove):
                    game.play(*case["moves"][0])
                self.assertEqual(dump_game(game), before)
        game = Game.new().play("B", 0)
        for color, action in [("B", 1), ("W", 0), ("W", -1), ("W", 362), ("W", True), ("W", 1.0)]:
            with self.assertRaises(IllegalMove):
                game.play(color, action)
        self.assertEqual(len(game.moves), 1)

    def test_ko_exchange_and_pass(self):
        game = Game.from_fixture(ko_patches()).play("B", point(3, 4))
        self.assertEqual(game.position.ko, point(3, 3))
        with self.assertRaises(IllegalMove):
            game.play("W", point(3, 3))
        self.assertEqual(game.position.ko, point(3, 3))
        passed = game.play("W", PASS)
        self.assertIsNone(passed.position.ko)
        self.assertTrue(passed.play("B", point(15, 15)).legal_mask()[point(3, 3)])
        exchanged = game.play("W", point(15, 15)).play("B", point(16, 15))
        recaptured = exchanged.play("W", point(3, 3))
        self.assertEqual(recaptured.position.captured_black, 1)
        self.assertEqual(recaptured.position.captured_white, 1)

    def test_recurrence_is_legal_and_caps_at_candidate_boundary(self):
        initial = Game.from_fixture(ko_patches(True), candidate_color="B")
        game = initial
        for action in CYCLE:
            game = game.play(game.position.to_play, action)
        self.assertEqual(game.position.key, initial.position.key)
        self.assertEqual((game.position.captured_black, game.position.captured_white), (3, 3))
        self.assertEqual((game.status, game.pending_cap), ("truncated", "recurrence"))
        self.assertEqual(game.match_result("B"), "U")
        replay = Game.from_fixture(ko_patches(True))
        for action in CYCLE * 3:
            replay = replay.play(replay.position.to_play, action)
        self.assertEqual(replay.status, "playing")
        self.assertEqual(len(replay.positions), 19)
        white = Game.from_fixture(ko_patches(True), candidate_color="W")
        for action in CYCLE:
            white = white.play(white.position.to_play, action)
        self.assertEqual(white.status, "playing")
        white = white.play("B", point(17, 17))
        self.assertEqual(white.status, "truncated")
        self.assertEqual(white.position.to_play, "W")

    def test_full_history_clone_and_serialization(self):
        game = Game.from_fixture(ko_patches(True))
        for action in CYCLE * 2:
            game = game.play(game.position.to_play, action)
        saved = dump_game(game)
        for count in (0, 8, 32, 64):
            self.assertEqual(len(game.recent_moves(count)), min(count, len(game.moves)))
            clone = game.clone()
            branch = clone.play("B", PASS)
            self.assertEqual(dump_game(game), saved)
            self.assertEqual(clone, game)
            self.assertEqual(len(branch.moves), 13)
            self.assertEqual(branch.segment_start, 13)
        restored = load_game(saved, allow_fixture=True)
        self.assertEqual(restored, game)
        self.assertEqual(restored.legal_mask(), game.legal_mask())
        with self.assertRaises(FrozenInstanceError):
            restored.position.to_play = "W"
        with self.assertRaises(ValueError):
            load_game(saved)
        changed = json.loads(saved)
        changed["origin"] = "empty"
        with self.assertRaises(ValueError):
            load_game(json.dumps(changed))

    def test_setup_never_repairs(self):
        with self.assertRaises(ValueError):
            Game.from_fixture(["B" * 19] * 19)
        with self.assertRaises(ValueError):
            Game.from_fixture(rows(), captured_black=True)
        with self.assertRaises(ValueError):
            Game.from_fixture(["." * 18] * 19)
        for setup in (b"AB[aa]", b"AW[aa]", b"AE[aa]", b"PL[W]", b"HA[2]"):
            with self.assertRaises(ValueError):
                replay_sgf(HEADER + setup + b";B[bb])")

    def test_pass_lifecycle_and_eof(self):
        empty = Game.new()
        first = empty.play("B", PASS)
        pending = first.play("W", PASS)
        self.assertEqual(first.status, "playing")
        self.assertEqual(pending.status, "pending_adjudication")
        self.assertEqual(sum(pending.legal_mask()), 0)
        for action in (0, PASS):
            with self.assertRaises(IllegalMove):
                pending.play("B", action)
        ending, decision = adjudicate(pending)
        self.assertEqual(ending.status, "unresolved")
        self.assertIsNone(decision.white_minus_black_halfpoints)
        self.assertEqual(len(ending.moves), 2)
        self.assertEqual(load_game(dump_game(ending)), ending)
        self.assertEqual(len(replay_sgf(HEADER + b";B[aa])").game.moves), 1)
        with self.assertRaises(ValueError):
            replay_sgf(HEADER + b";B[];W[];B[aa])")

    def test_strict_source_coordinate_and_pass_encoding(self):
        replay = replay_sgf(HEADER + b"RE[W+R]PL[B];B[aa];W[as];B[sa];W[ss];B[];W[tt])")
        self.assertEqual([m.action for m in replay.game.moves], [0, 342, 18, 360, 361, 361])
        self.assertEqual(replay.raw_moves[-2:], ("", "tt"))
        self.assertEqual(replay.source_result, "W+R")
        self.assertEqual(replay.source_result_kind, "resignation")
        self.assertIsNone(replay.game.outcome)
        self.assertFalse(replay.training_ready)
        for moves in (b";B[aa];W[aa]", b";B[aa]W[bb]", b";B[aa][bb]", b";W[aa]",
                      b";B[aa](;W[bb])(;W[cc])", b";B[aa];PL[B]B[bb]"):
            with self.subTest(moves=moves), self.assertRaises(ValueError):
                replay_sgf(HEADER + moves + b")")
        for suffix in (b"", b"RE[Void]", b"RE[0]", b"RE[B+T]", b"RE[W+F]"):
            replay = replay_sgf(HEADER + suffix + b";B[aa])")
            self.assertIsNone(replay.game.outcome)
            self.assertFalse(replay.training_ready)
        with self.assertRaises(ValueError):
            replay_sgf(HEADER.replace(b"SZ[19]", b"") + b";B[aa])")

    def test_resignation_and_player_perspectives(self):
        for resigning in ("B", "W"):
            game = Game.new()
            if resigning == "W":
                game = game.play("B", 0)
            ended = game.resign(resigning)
            self.assertEqual(ended.moves, game.moves)
            self.assertIsNone(ended.outcome.white_minus_black_halfpoints)
            for learner in ("B", "W"):
                boundary = ended.boundary(learner)
                self.assertEqual(boundary.reward, -1 if learner == resigning else 1)
                self.assertTrue(boundary.terminated)
                self.assertFalse(boundary.truncated)
                self.assertIsNone(boundary.bootstrap)
            self.assertEqual(load_game(dump_game(ended)), ended)
            with self.assertRaises(IllegalMove):
                ended.play(resigning, PASS)

    def test_actual_800_and_801_boundaries(self):
        for candidate, length in (("B", 800), ("W", 801)):
            game = Game.from_fixture(ko_patches(), candidate_color=candidate)
            for action in cap_actions()[:length]:
                game = game.play(game.position.to_play, action)
            self.assertEqual((len(game.moves), game.status, game.pending_cap),
                             (length, "truncated", "action_limit"))
            boundary = game.boundary(candidate)
            self.assertFalse(boundary.terminated)
            self.assertTrue(boundary.truncated)
            self.assertIs(boundary.bootstrap, game)
            self.assertEqual(game.position.to_play, candidate)
            self.assertEqual(load_game(dump_game(game), allow_fixture=True), game)
        game = Game.from_fixture(ko_patches(), candidate_color="W")
        for action in cap_actions()[:798]:
            game = game.play(game.position.to_play, action)
        pending = two_passes(game)
        self.assertEqual((len(pending.moves), pending.status), (800, "pending_adjudication"))
        self.assertFalse(pending.boundary("W").learning_valid)

    def test_white_opening_and_terminal_reply(self):
        game = Game.new(candidate_color="W")
        with self.assertRaises(ValueError):
            game.boundary("W")
        game = game.play("B", point(3, 3))
        self.assertEqual(len(game.boundary("W").bootstrap.moves), 1)
        learner_ends = game.resign("W")
        opponent_ends = game.play("W", point(15, 15)).resign("B")
        self.assertEqual(learner_ends.boundary("W").reward, -1)
        self.assertEqual(opponent_ends.boundary("W").reward, 1)
        self.assertEqual(len(opponent_ends.moves), 2)
        truncated = game.truncate("game_wall_limit", candidate_color="W")
        self.assertEqual(load_game(dump_game(truncated)), truncated)

    def test_fixed_learner_cannot_be_overridden(self):
        for color, other in (("B", "W"), ("W", "B")):
            game = Game.new(candidate_color=color)
            for state in (game, two_passes(game), game.fail("scheduler"), game.resign("B")):
                with self.subTest(color=color, status=state.status), self.assertRaises(ValueError):
                    state.boundary(other)
            self.assertEqual(game.resign("B").outcome.value(other), 1 if other == "W" else -1)

    def test_faults_and_unresolved_invalidate_entire_batch(self):
        good = Game.new().boundary("B")
        unknown, _ = adjudicate(two_passes(Game.new()))
        cases = [unknown, Game.new().fail("scheduler"), Game.new().fail("illegal_response", responsible="B")]
        self.assertEqual([g.match_result("B") for g in cases], ["U", "U", "L"])
        for game in cases:
            invalid = game.boundary("B")
            self.assertEqual((invalid.reward, invalid.terminated, invalid.truncated, invalid.bootstrap),
                             (None, None, None, None))
            updated = False
            with self.assertRaises(ValueError):
                require_valid_batch([good, invalid, good])
                updated = True
            self.assertFalse(updated)
        self.assertEqual(require_valid_batch([good]), (good,))

    def test_move_category_priority(self):
        self.assertEqual(move_category(Game.new(), PASS), "pass")
        self.assertEqual(move_category(Game.new(), point(3, 3)), "other")
        saving = Game.from_fixture(dense("save"))
        self.assertEqual(move_category(saving, point(9, 9)), "save_atari")
        giving = Game.from_fixture(rows({(9, 8), (8, 9)}, {(9, 9)}))
        self.assertEqual(move_category(giving, point(10, 9)), "give_atari")
        self_atari = Game.from_fixture(rows(white={(8, 9), (9, 8), (10, 9)}))
        self.assertEqual(move_category(self_atari, point(9, 9)), "self_atari")


class AdjudicationTests(unittest.TestCase):
    def test_empirical_outcomes_are_distinct_terminal_rewards(self):
        for score in (-3, 13):
            winner = "W" if score > 0 else "B"
            outcome = Outcome("empirical_score", "referee_fixture", winner, score, "fixture evidence")
            frozen = two_passes(Game.new())
            game = replace(frozen, status="resolved", outcome=outcome)
            self.assertEqual(game.moves, frozen.moves)
            self.assertEqual(game.positions, frozen.positions)
            for color in ("B", "W"):
                boundary = game.boundary(color)
                self.assertEqual(boundary.reward, 1 if color == winner else -1)
                self.assertTrue(boundary.terminated)
                self.assertFalse(boundary.truncated)
                self.assertIsNone(boundary.bootstrap)
            with self.assertRaisesRegex(ValueError, "independently configured referee"):
                load_game(dump_game(game))
        for invalid in (True, 13.0, 0, 2, None):
            with self.assertRaises(ValueError):
                Outcome("empirical_score", "fixture", "W", invalid, "fixture evidence")

    def test_persisted_score_rejects_boolean_and_float_halfpoints(self):
        ended, _ = adjudicate(two_passes(Game.from_fixture(dense(), captured_white=6)))
        self.assertEqual(ended.outcome.white_minus_black_halfpoints, 1)
        data = json.loads(dump_game(ended))
        for score in (True, 1.0, "1", 2, 0):
            data["ending"]["outcome"]["white_minus_black_halfpoints"] = score
            with self.subTest(score=score), self.assertRaises(ValueError):
                load_game(json.dumps(data), allow_fixture=True)

    def test_persisted_outcome_cannot_introduce_mutable_aliases(self):
        ended, _ = adjudicate(two_passes(Game.new()))
        record = json.loads(dump_game(ended))
        for field, value in (("reason", ["unsupported_position"]), ("evidence", {"classification": []}),
                             ("winner", []), ("kind", []), ("white_minus_black_halfpoints", [])):
            changed = copy.deepcopy(record)
            changed["ending"]["outcome"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                load_game(json.dumps(changed))
        loaded = load_game(dump_game(ended))
        clone = loaded.clone()
        self.assertIsInstance(clone.outcome.reason, str)
        self.assertIsInstance(clone.outcome.evidence, str)
        with self.assertRaises(FrozenInstanceError):
            loaded.outcome.reason = "changed"
        self.assertEqual(clone, ended)

    def test_exact_territory_captures_signs_and_immutable_scoring(self):
        for captured_white, expected in ((0, 13), (8, -3)):
            game = two_passes(Game.from_fixture(dense(), captured_white=captured_white))
            ended, decision = adjudicate(game)
            self.assertEqual(decision.white_minus_black_halfpoints, expected)
            self.assertEqual(ended.positions, game.positions)
            self.assertEqual(ended.moves, game.moves)
            self.assertEqual(game.status, "pending_adjudication")
            self.assertEqual(ended.outcome.value("W"), 1 if expected > 0 else -1)
            self.assertEqual(ended.outcome.value("B"), -ended.outcome.value("W"))
            self.assertEqual(load_game(dump_game(ended), allow_fixture=True), ended)
            forged = json.loads(dump_game(ended))
            forged["ending"]["outcome"]["white_minus_black_halfpoints"] += 2
            with self.assertRaises(ValueError):
                load_game(json.dumps(forged), allow_fixture=True)

    def test_large_invadable_region_and_dame_are_not_certified(self):
        cells = ["B"] * PASS
        for r, c in [(1, 1), (1, 17)] + [(r, c) for r in range(3, 16) for c in range(3, 16)]:
            cells[point(r, c)] = "."
        large = ["".join(cells[i:i + 19]) for i in range(0, PASS, 19)]
        dame = list("".join(dense()))
        dame[point(9, 8)] = "."
        dame = ["".join(dame[i:i + 19]) for i in range(0, PASS, 19)]
        for board in (rows(), large, dame, dense("dead"), dense("seki")):
            ended, _ = adjudicate(two_passes(Game.from_fixture(board)))
            self.assertEqual(ended.status, "unresolved")
            self.assertIsNone(ended.outcome.value("B"))

    def test_neutral_dead_seki_and_out_of_band_authorization(self):
        for kind, score in (("dead", 3), ("seki", 13)):
            game = two_passes(Game.from_fixture(dense(kind)))
            payload = neutral_decision(game, kind)
            unauthorized, _ = adjudicate(game, neutral=payload)
            self.assertEqual(unauthorized.status, "unresolved")
            authorized = frozenset({digest(payload)})
            ended, _ = adjudicate(game, neutral=payload, authorized_hashes=authorized)
            self.assertEqual(ended.outcome.white_minus_black_halfpoints, score)
            self.assertEqual(ended.positions, game.positions)
            self.assertEqual(ended.moves, game.moves)
            self.assertEqual(load_game(dump_game(ended), allow_fixture=True, authorized_hashes=authorized), ended)
            with self.assertRaises(ValueError):
                load_game(dump_game(ended), allow_fixture=True)

    def test_reject_incomplete_overlapping_wrong_snapshot_and_seki_territory(self):
        game = two_passes(Game.from_fixture(dense("dead")))
        original = neutral_decision(game, "dead")
        malformed = []
        p = copy.deepcopy(original)
        p["map"]["dead"][0].pop()
        malformed.append(p)
        p = copy.deepcopy(original)
        p["map"]["alive"].append(p["map"]["dead"][0])
        malformed.append(p)
        p = copy.deepcopy(original)
        p["map"]["territory_black"].remove(point(9, 4))
        malformed.append(p)
        p = copy.deepcopy(original)
        p["snapshot"] = "0" * 64
        malformed.append(p)
        p = copy.deepcopy(original)
        p["map"]["territory_white"].append(p["map"]["territory_black"][0])
        malformed.append(p)
        p = copy.deepcopy(original)
        p["budget"]["nodes"] = 100001
        malformed.append(p)
        for payload in malformed:
            ended, _ = adjudicate(game, neutral=payload, authorized_hashes={digest(payload)})
            self.assertEqual(ended.status, "unresolved")
        seki = two_passes(Game.from_fixture(dense("seki")))
        payload = neutral_decision(seki, "seki")
        payload["map"]["neutral"].remove(point(3, 3))
        payload["map"]["territory_black"].append(point(3, 3))
        ended, _ = adjudicate(seki, neutral=payload, authorized_hashes={digest(payload)})
        self.assertEqual(ended.status, "unresolved")

    def test_budget_exhaustion_and_pass_bounds(self):
        settled = Game.from_fixture(dense())
        ended, decision = adjudicate(two_passes(settled), limits=Limits(nodes=0))
        self.assertEqual(ended.status, "unresolved")
        self.assertEqual(decision.reason, "proof_budget_exhausted")
        self.assertEqual(judge_pass(settled).verdict, "acceptable")
        self.assertEqual(judge_pass(settled).alternative_gain_upper_halfpoints, 0)
        self.assertEqual(judge_pass(settled, limits=Limits(nodes=0)).verdict, "unjudgeable")
        saving = Game.from_fixture(dense("save"))
        premature = judge_pass(saving)
        self.assertEqual(premature.verdict, "premature")
        self.assertEqual(json.loads(premature.witness)["gain_lower_halfpoints"], 4)
        # A prior actual pass makes the response-to-pass witness unavailable.
        prior_pass = Game.from_fixture(dense("save"), to_play="W").play("W", PASS)
        self.assertEqual(judge_pass(prior_pass).verdict, "unjudgeable")
        self.assertEqual(judge_pass(Game.from_fixture(dense("dead"))).verdict, "unjudgeable")
        self.assertEqual(judge_pass(Game.new()).verdict, "unjudgeable")


if __name__ == "__main__":
    unittest.main()
