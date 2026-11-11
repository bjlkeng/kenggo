"""Referee protocol tests use fake neural answers, not neural quality evidence."""

from dataclasses import FrozenInstanceError, replace
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

from kenggo.go import Game, PASS, canonical, digest
from kenggo.katago_referee import (KataGoReferee, Profile, RefereeError, REQUIRED_CONFIG,
                                   _Budget, _JsonProcess, _action)
from go_fixtures import dense, two_passes


class FakeTransport:
    def __init__(self, handler):
        self.handler, self.closed, self.requests = handler, False, []

    def exchange(self, raw, budget):
        request = json.loads(raw)
        self.requests.append(request)
        result = self.handler(request)
        return result if isinstance(result, bytes) else (canonical(result) + "\n").encode()

    def cpu_seconds(self):
        return 0.0

    def close(self):
        self.closed = True


def analysis_reply(request, *, action=PASS, score=0.0):
    player = request["initialPlayer"] if not request["moves"] else (
        "W" if request["moves"][-1][0] == "B" else "B")
    move = "pass" if action == PASS else "ABCDEFGHJKLMNOPQRST"[action % 19] + str(19 - action // 19)
    return {"id": request["id"], "turnNumber": len(request["moves"]), "isDuringSearch": False,
            "rootInfo": {"currentPlayer": player, "visits": request["maxVisits"], "scoreLead": score},
            "moveInfos": [{"move": move, "order": 0, "visits": request["maxVisits"],
                           "scoreLead": score, "noResultValue": 0.0}]}


def native_reply(request):
    """A minimal phase stub for protocol tests; live native tests cover real rules."""
    initial = Game.from_fixture(request["board"], to_play=request["to_play"],
                                captured_black=request["captures"]["B"],
                                captured_white=request["captures"]["W"])
    game, phase, ending_passes, finished = initial, 0, 0, False

    def state():
        position = game.position
        score = 13 - (2 if position.board[0] == "B" else 0)
        return {"board": list(position.rows), "to_play": position.to_play,
                "captures": {"B": position.captured_black, "W": position.captured_white},
                "simple_ko": position.ko, "move_count": len(game.moves),
                "pass_streak": position.pass_streak, "encore_phase": phase,
                "finished": finished, "scored": finished, "no_result": False,
                "winner": ("W" if score > 0 else "B") if finished else None,
                "white_minus_black_halfpoints": score if finished else None,
                "legal_mask": [False] * 362 if finished else [c == "." for c in position.board] + [True]}

    result = {"ok": True, "mode": "history", "size": 19, "initial": state(), "steps": []}
    for move in request["moves"]:
        color, action = move["color"], move["action"]
        accepted = not finished
        if accepted:
            try:
                game = replace(game, status="playing").play(color, action)
            except ValueError:
                accepted = False
        if accepted:
            ending_passes = ending_passes + 1 if action == PASS else 0
            if ending_passes == 2:
                if phase == 2:
                    finished = True
                else:
                    phase += 1
                    ending_passes = 0
        result["steps"].append({"accepted": accepted, "color": color, "action": action, "state": state()})
    result["final"] = state()
    return result


def ending():
    return two_passes(Game.new().play("B", 20).play("W", 21))


class RefereeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = {}
        for name in ("engine", "model", "config", "native"):
            path = self.root / name
            path.write_text(name if name != "config" else "\n".join(
                f"{key} = {value}" for key, value in REQUIRED_CONFIG.items()) + "\nnnRandSeed = fixture\n")
            self.paths[name] = path
        self.settings = {
            "schema": "katago_referee_profile_v1", "profile_id": "protocol-fixture",
            "engine_version": "fake-neural", "native_revision": "fake-phase-stub", "model_name": "fake",
            "artifacts": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in self.paths.items()},
            "runtime_files": {}, "library_dirs": [], "cleanup_visits": [32, 128], "pass_visits": [32, 128],
            "cleanup_move_limit": 32, "acceptable_gain": 0.5, "premature_gain": 1.5, "no_result_limit": 0.01,
            "limits": {"nodes": 100_000, "cpu_seconds": 10, "wall_seconds": 20,
                       "request_bytes": 1024**2, "response_bytes": 1024**2,
                       "stderr_bytes": 65536, "evidence_bytes": 1024**2}}

    def profile(self):
        raw = (canonical(self.settings) + "\n").encode()
        path = self.root / "manifest.json"
        path.write_bytes(raw)
        return Profile.load(path, hashlib.sha256(raw).hexdigest())

    def referee(self, analysis=None, native=None):
        result = KataGoReferee(self.profile(), **self.paths,
            _analysis=FakeTransport(analysis or analysis_reply), _native=FakeTransport(native or native_reply))
        self.addCleanup(result.close)
        return result

    def test_cleanup_agreement_preserves_original_and_is_immutable(self):
        game = ending()
        original = game.action_record()
        referee = self.referee()
        result = referee.adjudicate(game)
        self.assertEqual((result.status, result.reason, result.white_minus_black_halfpoints),
                         ("resolved", "agreed_empirical_cleanup", 13))
        self.assertEqual(game.action_record(), original)
        self.assertEqual(game.status, "pending_adjudication")
        evidence = json.loads(result.evidence)
        self.assertEqual(evidence["original"], json.loads(canonical(original)))
        self.assertEqual(evidence["claim"], "empirical_not_certified")
        self.assertEqual([run["visits"] for run in evidence["runs"]], [32, 128])
        self.assertEqual([len(run["cleanup"]) for run in evidence["runs"]], [4, 4])
        self.assertEqual(evidence["cost"]["requests"], 17)
        self.assertEqual(len(evidence["calls"]), 17)
        self.assertTrue(all(len(call["query_sha256"]) == 64 for call in evidence["calls"]))
        with self.assertRaises(FrozenInstanceError):
            result.status = "unresolved"

    def test_cleanup_disagreement_and_move_limit_are_unresolved(self):
        def different(request):
            action = 0 if request["maxVisits"] == 32 and len(request["moves"]) == 4 else PASS
            return analysis_reply(request, action=action)
        result = self.referee(different).adjudicate(ending())
        self.assertEqual(result.reason, "cleanup_budget_disagreement")
        self.assertIsNone(result.white_minus_black_halfpoints)
        self.settings["cleanup_move_limit"] = 3
        result = self.referee().adjudicate(ending())
        self.assertEqual(result.reason, "cleanup_move_budget_exhausted")

    def test_winner_profile_selects_high_budget_margin_and_rechecks_each_transcript(self):
        self.settings.update(schema="katago_referee_profile_v2", cleanup_move_limit=64,
                             score_selection="higher_visit_when_winners_agree")

        def different(request):
            action = 0 if request["maxVisits"] == 128 and len(request["moves"]) == 4 else PASS
            return analysis_reply(request, action=action)

        referee = self.referee(different)
        game = ending()
        result = referee.adjudicate(game)
        self.assertEqual((result.status, result.reason, result.white_minus_black_halfpoints),
                         ("resolved", "winner_stable_empirical_cleanup", 11))
        evidence = json.loads(result.evidence)
        self.assertEqual([run["terminal"]["white_minus_black_halfpoints"] for run in evidence["runs"]], [13, 11])
        self.assertEqual(referee.verify(game, result.evidence, authorized_hashes={digest(evidence)}), result)
        evidence["white_minus_black_halfpoints"] = 13
        with self.assertRaises(ValueError):
            self.referee().verify(game, canonical(evidence), authorized_hashes={digest(evidence)})

    def test_winner_profile_does_not_resolve_opposite_winners(self):
        self.settings.update(schema="katago_referee_profile_v2", cleanup_move_limit=64,
                             score_selection="higher_visit_when_winners_agree")

        def different(request):
            action = 0 if request["maxVisits"] == 32 and len(request["moves"]) == 4 else PASS
            return analysis_reply(request, action=action)

        def opposite(request):
            result = native_reply(request)
            state = result["final"]
            if state["finished"] and state["board"][0][0] == "B":
                state.update(white_minus_black_halfpoints=-1, winner="B")
                result["steps"][-1]["state"] = state
            return result

        result = self.referee(different, opposite).adjudicate(ending())
        self.assertEqual((result.status, result.reason, result.white_minus_black_halfpoints),
                         ("unresolved", "cleanup_winner_disagreement", None))

    def test_extended_cleanup_persistence_accepts_its_full_request_ledger(self):
        self.settings.update(schema="katago_referee_profile_v2", cleanup_move_limit=64,
                             score_selection="higher_visit_when_winners_agree")

        def longer(request):
            index = len(request["moves"]) - 4
            return analysis_reply(request, action=100 + index if index < 36 else PASS)

        game = ending()
        referee = self.referee(longer)
        result = referee.adjudicate(game)
        self.assertEqual(result.status, "resolved")
        evidence = json.loads(result.evidence)
        self.assertEqual([len(run["cleanup"]) for run in evidence["runs"]], [40, 40])
        self.assertEqual(evidence["cost"]["requests"], 161)
        self.assertEqual(referee.verify(game, result.evidence, authorized_hashes={digest(evidence)}), result)

    def test_empty_ending_is_rejected_before_any_referee_query(self):
        referee = self.referee()
        game = two_passes(Game.new())
        result = referee.adjudicate(game)
        self.assertEqual((result.status, result.reason), ("unresolved", "empty_board_ending_unsupported"))
        self.assertIsNone(result.white_minus_black_halfpoints)
        self.assertEqual(json.loads(result.evidence)["calls"], [])
        with self.assertRaisesRegex(ValueError, "Empty-board"):
            referee.verify(game, result.evidence, authorized_hashes={digest(json.loads(result.evidence))})

    def test_stale_malformed_warnings_and_illegal_neural_output_fail(self):
        def altered(field, value):
            def reply(request):
                result = analysis_reply(request)
                result[field] = value
                return result
            return reply
        cases = [(altered("id", "wrong"), "invalid_or_stale_analysis"),
                 (altered("turnNumber", True), "invalid_or_stale_analysis"),
                 (altered("isDuringSearch", True), "invalid_or_stale_analysis"),
                 (altered("warning", "unsupported rules"), "analysis_error_or_warning"),
                 (lambda _: b'{"id":1,"id":2}\n', "invalid_json"),
                 (lambda _: b'{"x":NaN}\n', "invalid_json"),
                 (lambda _: b"[]\n", "response_is_not_object")]
        for handler, reason in cases:
            with self.subTest(reason=reason):
                referee = self.referee(handler)
                self.assertEqual(referee.adjudicate(ending()).reason, reason)
        occupied = two_passes(Game.new().play("B", 0).play("W", 1))
        result = self.referee(lambda q: analysis_reply(q, action=0)).adjudicate(occupied)
        self.assertEqual(result.reason, "illegal_referee_move")

    def test_native_no_result_and_invalid_score_types_fail(self):
        def changed(change):
            def reply(request):
                result = native_reply(request)
                if result["final"]["finished"]:
                    change(result["final"])
                    result["steps"][-1]["state"] = result["final"]
                return result
            return reply
        no_result = changed(lambda s: s.update(no_result=True, scored=False, winner=None,
                                               white_minus_black_halfpoints=None))
        self.assertEqual(self.referee(native=no_result).adjudicate(ending()).reason,
                         "native_no_result")
        for score in (True, 13.0, 0, 12):
            with self.subTest(score=score):
                native = changed(lambda s: s.update(white_minus_black_halfpoints=score))
                self.assertEqual(self.referee(native=native).adjudicate(ending()).reason,
                                 "invalid_native_score")

    def test_predicted_no_result_is_diagnostic_for_cleanup_but_blocks_pass_estimates(self):
        def uncertain(request):
            result = analysis_reply(request)
            result["moveInfos"][0]["noResultValue"] = 0.8
            return result

        trace = []
        referee = self.referee(uncertain)
        referee.trace_sink = lambda kind, query, response: trace.append((kind, response))
        decision = referee.adjudicate(ending())
        self.assertEqual((decision.status, decision.white_minus_black_halfpoints), ("resolved", 13))
        evidence = json.loads(decision.evidence)
        for call, (kind, response) in zip(evidence["calls"], trace):
            self.assertEqual(call["response_sha256"], hashlib.sha256(response).hexdigest())
            if kind == "analysis":
                self.assertEqual(json.loads(response)["moveInfos"][0]["noResultValue"], 0.8)
        judgment = referee.judge_pass(Game.new())
        self.assertEqual((judgment.verdict, judgment.reason),
                         ("unjudgeable", "analysis_no_result_uncertainty"))

        def no_result(request):
            result = native_reply(request)
            if result["final"]["finished"]:
                result["final"].update(no_result=True, scored=False, winner=None,
                                       white_minus_black_halfpoints=None)
                result["steps"][-1]["state"] = result["final"]
            return result

        decision = self.referee(uncertain, no_result).adjudicate(ending())
        self.assertEqual((decision.status, decision.reason, decision.white_minus_black_halfpoints),
                         ("unresolved", "native_no_result", None))

    def test_original_native_captures_and_full_history_are_checked(self):
        def mismatch(request):
            result = native_reply(request)
            result["initial"]["captures"]["B"] = 1
            return result
        game = two_passes(Game.new().play("B", 0).play("W", 1))
        result = self.referee(native=mismatch).adjudicate(game)
        self.assertEqual(result.reason, "original_native_state_mismatch")
        referee = self.referee()
        referee.adjudicate(game)
        first_query = referee.analysis.requests[0]
        self.assertEqual(first_query["moves"], [["B", "A19"], ["W", "B19"], ["B", "pass"], ["W", "pass"]])

    def test_pass_signs_first_and_second_pass_and_grey_band(self):
        for color, previous_pass in (("B", False), ("W", False), ("W", True)):
            game = Game.new() if color == "B" else Game.new().play("B", PASS if previous_pass else 0)
            for gain, expected in ((0.25, "acceptable"), (1.0, "unjudgeable"), (2.0, "premature")):
                def reply(request):
                    forced_pass = "allowMoves" in request
                    score = 0.0 if forced_pass else gain * (1 if color == "W" else -1)
                    return analysis_reply(request, action=PASS if forced_pass else 2, score=score)
                with self.subTest(color=color, previous_pass=previous_pass, gain=gain):
                    before = game.action_record()
                    judgment = self.referee(reply).judge_pass(game)
                    self.assertEqual(judgment.verdict, expected)
                    self.assertEqual(judgment.gains_points, (gain, gain))
                    self.assertEqual(game.action_record(), before)

    def test_pass_budget_disagreement_and_restriction_violation(self):
        def reply(request):
            forced_pass = "allowMoves" in request
            score = 0 if forced_pass or request["maxVisits"] == 32 else -2
            return analysis_reply(request, action=PASS if forced_pass else 0, score=score)
        judgment = self.referee(reply).judge_pass(Game.new())
        self.assertEqual((judgment.verdict, judgment.gains_points), ("unjudgeable", (0, 2)))
        judgment = self.referee().judge_pass(Game.new())
        self.assertEqual(judgment.reason, "analysis_violated_root_restriction")

    def test_limits_and_artifact_pins_fail_closed(self):
        self.settings["limits"]["nodes"] = 1
        result = self.referee().adjudicate(ending())
        self.assertEqual(result.reason, "node_budget_exhausted")
        self.settings["limits"]["nodes"] = 100_000
        referee = self.referee()
        self.paths["model"].write_text("changed")
        self.assertEqual(referee.adjudicate(ending()).reason, "artifact_hash_mismatch:model")
        with self.assertRaises(RefereeError):
            self.referee()
        manifest = self.root / "manifest.json"
        with self.assertRaises(ValueError):
            Profile.load(manifest, "0" * 64)

    def test_evidence_byte_limit_returns_bounded_failure_for_both_methods(self):
        self.settings["limits"]["evidence_bytes"] = 1024

        def reply(request):
            return analysis_reply(request, action=2 if "avoidMoves" in request else PASS)

        for method, game, expected in (("adjudicate", ending(), "unresolved"),
                                       ("judge_pass", Game.new(), "unjudgeable")):
            with self.subTest(method=method):
                before = game.action_record()
                referee = self.referee(reply)
                result = getattr(referee, method)(game)
                status = result.status if method == "adjudicate" else result.verdict
                self.assertEqual((status, result.reason), (expected, "evidence_byte_limit"))
                evidence = json.loads(result.evidence)
                self.assertEqual(evidence["snapshot"], game.snapshot)
                self.assertEqual(evidence["status"], expected)
                self.assertIsNone(evidence["white_minus_black_halfpoints"])
                self.assertLessEqual(len(result.evidence.encode()), 1024)
                self.assertEqual(game.action_record(), before)

    def test_runtime_library_pins_and_config_are_checked(self):
        directory = self.root / "lib"
        directory.mkdir()
        library = directory / "libfixture.so"
        library.write_bytes(b"fixture")
        self.settings["library_dirs"] = ["lib"]
        with self.assertRaisesRegex(RefereeError, "unpinned_runtime_library"):
            self.referee()
        self.settings["runtime_files"] = {"lib/libfixture.so": hashlib.sha256(b"fixture").hexdigest()}
        referee = self.referee()
        library.write_bytes(b"changed")
        self.assertEqual(referee.adjudicate(ending()).reason,
                         "runtime_hash_mismatch:lib/libfixture.so")
        library.write_bytes(b"fixture")
        changed = self.paths["config"].read_text().replace("ignoreAllHistory = false", "ignoreAllHistory = true")
        self.paths["config"].write_text(changed)
        self.settings["artifacts"]["config"] = hashlib.sha256(changed.encode()).hexdigest()
        with self.assertRaisesRegex(RefereeError, "unsafe_referee_config:ignoreAllHistory"):
            self.referee()

    def test_profile_rejects_invalid_budget_types(self):
        for key, value in (("cleanup_visits", [True, 128]), ("cleanup_visits", [128, 32]),
                           ("cleanup_move_limit", 33), ("acceptable_gain", 2)):
            original = self.settings[key]
            self.settings[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.profile()
            self.settings[key] = original

    def test_persistence_requires_external_authority_and_native_replay(self):
        game = ending()
        referee = self.referee()
        decision = referee.adjudicate(game)
        evidence = json.loads(decision.evidence)
        with self.assertRaisesRegex(ValueError, "independent authorization"):
            referee.verify(game, decision.evidence)
        analysis_requests = len(referee.analysis.requests)
        checked = referee.verify(game, decision.evidence, authorized_hashes={digest(evidence)})
        self.assertEqual(checked, decision)
        self.assertEqual(len(referee.analysis.requests), analysis_requests)
        for mutation in (lambda e: e.update(white_minus_black_halfpoints=True),
                         lambda e: e.update(snapshot="0" * 64),
                         lambda e: e["runs"][0]["cleanup"].pop(),
                         lambda e: e["runs"][0]["cleanup"][0].__setitem__(1, 0),
                         lambda e: e["calls"][0].update(query_sha256="unknown"),
                         lambda e: e["cost"].update(request_bytes=1),
                         lambda e: e["cost"].update(nodes=1),
                         lambda e: e["profile"].update(cleanup_move_limit=32.0)):
            altered = json.loads(decision.evidence)
            mutation(altered)
            with self.subTest(altered=altered["white_minus_black_halfpoints"]), self.assertRaises(ValueError):
                self.referee().verify(game, canonical(altered), authorized_hashes={digest(altered)})

    def test_concurrent_callers_share_a_serial_process_pair(self):
        referee = self.referee()
        game = ending()
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(referee.adjudicate, [game, game]))
        self.assertEqual([item.status for item in results], ["resolved", "resolved"])
        ids = [request["id"] for request in referee.analysis.requests]
        self.assertEqual(len(set(ids)), 16)

    def test_coordinate_conversion_rejects_noncanonical_values(self):
        self.assertEqual((_action("A19"), _action("T1"), _action("pass")), (0, 360, PASS))
        for value in (None, "resign", "I9", "A0", "A01", "T20", "a1"):
            with self.subTest(value=value), self.assertRaises(RefereeError):
                _action(value)

    @unittest.skipUnless(os.environ.get("KENGGO_KATAGO_REFERENCE"), "Requires pinned native oracle")
    def test_real_native_cleanup_arithmetic_with_fake_pass_policy(self):
        path = Path(os.environ["KENGGO_KATAGO_REFERENCE"]).resolve()
        self.paths["native"] = path
        self.settings["artifacts"]["native"] = hashlib.sha256(path.read_bytes()).hexdigest()
        referee = KataGoReferee(self.profile(), **self.paths, _analysis=FakeTransport(analysis_reply))
        self.addCleanup(referee.close)
        game = two_passes(Game.from_fixture(dense()))
        decision = referee.adjudicate(game)
        self.assertEqual((decision.status, decision.white_minus_black_halfpoints), ("resolved", 13))
        self.assertEqual(len(game.moves), 2)
        evidence = json.loads(decision.evidence)
        self.assertEqual(referee.verify(game, decision.evidence, authorized_hashes={digest(evidence)}), decision)


class ProcessTests(unittest.TestCase):
    def budget(self, process, **changes):
        limits = {"nodes": 1000, "cpu_seconds": 2, "wall_seconds": 2,
                  "request_bytes": 4096, "response_bytes": 4096, "stderr_bytes": 4096,
                  "evidence_bytes": 4096}
        limits.update(changes)
        return _Budget(limits, lambda: time.process_time() + process.cpu_seconds())

    def process(self, code):
        process = _JsonProcess([sys.executable, "-c", code])
        self.addCleanup(process.close)
        return process

    def test_serial_process_reuse(self):
        process = self.process("import sys\nfor line in sys.stdin:\n print(line.strip(), flush=True)")
        budget = self.budget(process)
        self.assertEqual(process.exchange(b'{"one":1}\n', budget), b'{"one":1}\n')
        self.assertEqual(process.exchange(b'{"two":2}\n', budget), b'{"two":2}\n')

    def test_wall_cpu_byte_and_exit_limits(self):
        cases = [
            ("import sys,time;sys.stdin.readline();time.sleep(1)", {"wall_seconds": 0.03}, "wall_budget_exhausted"),
            ("import sys;sys.stdin.readline()\nwhile True: pass", {"cpu_seconds": 0.03}, "cpu_budget_exhausted"),
            ("import sys;sys.stdin.readline();print('x'*3000,flush=True)", {"response_bytes": 1024}, "response_byte_limit"),
            ("import sys,time;sys.stdin.readline();sys.stderr.write('x'*3000);sys.stderr.flush();time.sleep(1)",
             {"stderr_bytes": 1024}, "stderr_byte_limit"),
            ("import sys;sys.stdin.readline()", {}, "process_output_closed"),
        ]
        for code, limits, reason in cases:
            with self.subTest(reason=reason):
                process = self.process(code)
                with self.assertRaisesRegex(RefereeError, reason):
                    process.exchange(b"{}\n", self.budget(process, **limits))
                process.close()


if __name__ == "__main__":
    unittest.main()
