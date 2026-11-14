import copy
from dataclasses import replace
import json
import os
import signal
import socket
import struct
import subprocess
import sys
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from kenggo.go import Game, PASS, canonical, digest
from kenggo.katago_referee import PassDecision, RefereeDecision
from kenggo.matches import (AgentProfile, Decision, ExperimentBudget, InfrastructureError,
                            MatchLimits, make_schedule, read_records, release_binding,
                            run_game, run_schedule, validate_schedule, weak_profile,
                            _Failure, _Meter, _Wire, _Worker, _FRAME_LIMIT, _tree_usage, _WeakAgent, _judgment,
                            _validate_pass_record, _write_new)
from kenggo.rules_replay import dump_game, load_game
from go_fixtures import CYCLE, cap_actions, dense, ko_patches, rows, point


REVISION = "a" * 40
REFEREE = {"sha256": "f" * 64, "device": "cpu"}
LIMITS = MatchLimits(response_seconds=1, game_seconds=30)
BUDGET = ExperimentBudget(cpu_seconds=20, wall_seconds=40)


def profile(id="candidate", policy="script", actions=None, **changes):
    data = weak_profile(id, source_revision=REVISION).data
    data.update(policy=policy, actions=actions or [PASS] * 4, **changes)
    return AgentProfile.from_dict(data)


def slot(color="B"):
    return {"id": "s00000000", "block_id": "b0", "arm": "candidate", "profile_id": "opponent",
            "candidate_color": color, "candidate_seed": 10, "opponent_seed": 20}


class ScriptAgent:
    def __init__(self, data, seed):
        self.data, self.seed = data, seed
        if data["policy"] == "initialization_exception":
            raise RuntimeError("broken_agent_initialization")
        if data["policy"] == "initialization_infrastructure":
            raise InfrastructureError("unattributed_initialization")
        if data["policy"] == "initialization_exit":
            os._exit(7)
        if data["policy"] == "initialization_hang":
            time.sleep(60)

    def choose(self, game):
        policy = self.data["policy"]
        if policy == "invalid_calls":
            return Decision(0, "bad", 0.0)
        if policy == "invalid_gpu":
            return Decision(0, 0, float("nan"))
        if policy == "invalid_action":
            return Decision({"bad": "action"}, 0, 0.0)
        if policy == "crash":
            os._exit(7)
        if policy == "hang":
            time.sleep(60)
        if policy == "child_hang":
            subprocess.Popen([sys.executable, "-c", "while True: pass"])
            time.sleep(60)
        if policy == "infrastructure":
            raise InfrastructureError("ambiguous_transport")
        if policy == "exception":
            raise RuntimeError("broken_agent")
        return Decision(self.data["actions"][len(game.moves) % len(self.data["actions"])], 0, 0.0)


def factory(data, seed):
    return ScriptAgent(data, seed)


FACTORIES = {name: factory for name in ("script", "crash", "hang", "child_hang", "infrastructure", "exception",
             "initialization_exception", "initialization_infrastructure", "initialization_exit", "initialization_hang",
             "invalid_calls", "invalid_gpu", "invalid_action")}


class FakeReferee:
    def __init__(self, scores=(13, 11)):
        self.profile = SimpleNamespace(sha256=REFEREE["sha256"])
        self.scores = scores

    def adjudicate(self, game):
        assert game.status == "pending_adjudication"
        same = (self.scores[0] > 0) == (self.scores[1] > 0)
        status = "resolved" if same else "unresolved"
        reason = "winner_stable_empirical_cleanup" if same else "cleanup_winner_disagreement"
        score = self.scores[1] if same else None
        evidence = canonical({"snapshot": game.snapshot, "original": game.action_record(),
                              "profile_sha256": REFEREE["sha256"], "status": status, "reason": reason,
                              "white_minus_black_halfpoints": score,
                              "runs": [{"terminal": {"white_minus_black_halfpoints": value}}
                                       for value in self.scores]})
        return RefereeDecision(status, reason, game.snapshot, score, evidence)

    def judge_pass(self, game):
        return PassDecision("unjudgeable", "fixture", game.snapshot, (), canonical({"snapshot": game.snapshot}))

    def verify(self, game, evidence, *, authorized_hashes=frozenset()):
        expected = self.adjudicate(game)
        if digest(json.loads(evidence)) not in authorized_hashes or evidence != expected.evidence:
            raise ValueError("Missing independent authorization or stale evidence")
        return expected


def referee_factory():
    return FakeReferee()


class KnownPassReferee(FakeReferee):
    def judge_pass(self, game):
        evidence = canonical({"snapshot": game.snapshot, "profile_sha256": self.profile.sha256,
                              "status": "acceptable", "reason": "fixture_acceptable"})
        return PassDecision("acceptable", "fixture_acceptable", game.snapshot, (0.0, 0.0), evidence)


def known_pass_referee_factory():
    return KnownPassReferee()


def disagreement_factory():
    return FakeReferee((13, -3))


def hanging_referee_factory():
    time.sleep(60)


def cpu_initializing_referee_factory():
    started = time.process_time()
    while time.process_time() - started < 0.3:
        pass
    return FakeReferee()


def exit_before_ready(channel, role, factory, profile, seed):
    os._exit(7)


def partial_peer(channel, role, factory, profile, seed):
    os.setsid()
    wire = _Wire(channel)
    wire.send(("ready", None))
    wire.send(("initialized", None))
    wire.recv()
    channel.sendall(struct.pack("!I", _FRAME_LIMIT + 1 if seed else 1000) + b"partial")
    time.sleep(60)


class WorkerTests(unittest.TestCase):
    def test_exit_before_worker_ready_keeps_infrastructure_attribution(self):
        meter = _Meter(BUDGET)
        worker = _Worker("agent", factory, profile().data, 10, meter, _target=exit_before_ready)
        try:
            result = worker.initialize(1, time.monotonic() + 3)
            self.assertIsInstance(result, _Failure)
            self.assertFalse(result.responsible)
        finally:
            worker.close()

    def test_stopped_reader_bounds_large_full_history_transmission(self):
        meter = _Meter(ExperimentBudget(10, 10))
        worker = _Worker("agent", factory, profile().data, 1, meter)
        self.addCleanup(worker.close)
        self.assertIsNone(worker.initialize(2, time.monotonic() + 5))
        worker.channel.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
        os.kill(worker.process.pid, signal.SIGSTOP)
        game = Game.from_fixture(ko_patches(), candidate_color="B")
        for action in cap_actions()[:798]:
            game = game.play(game.position.to_play, action)
        started = time.monotonic()
        failure, _ = worker.call("choose", game, LIMITS, 0.15, started + 5)
        self.assertIsInstance(failure, _Failure)
        self.assertEqual(failure.reason, "response_timeout")
        self.assertLess(time.monotonic() - started, 1)
        self.assertIsNotNone(worker.exitcode)

    def test_partial_and_oversized_reply_frames_are_bounded(self):
        for oversized in (False, True):
            meter = _Meter(ExperimentBudget(10, 10))
            worker = _Worker("agent", None, None, int(oversized), meter, _target=partial_peer)
            self.addCleanup(worker.close)
            self.assertIsNone(worker.initialize(2, time.monotonic() + 5))
            started = time.monotonic()
            failure, _ = worker.call("choose", Game.new(), LIMITS, 0.15, started + 5)
            self.assertIsInstance(failure, _Failure)
            self.assertEqual(failure.reason, "worker_transport" if oversized else "response_timeout")
            self.assertLess(time.monotonic() - started, 1)
            self.assertIsNotNone(worker.exitcode)

    def test_judgment_cpu_budget_includes_factory_initialization(self):
        meter = _Meter(ExperimentBudget(10, 10))
        game = Game.new(candidate_color="B").play("B", 0).play("W", 1).play("B", PASS).play("W", PASS)
        result, cost = _judgment(game, "end_empirical", limits=replace(LIMITS, empirical_cpu_seconds=0.08),
                                 meter=meter, deadline=time.monotonic() + 5,
                                 referee_factory=cpu_initializing_referee_factory, referee_profile=REFEREE)
        self.assertIsInstance(result, _Failure)
        self.assertEqual(result.reason, "judgment_cpu_budget")
        self.assertFalse(result.responsible)
        self.assertLess(cost["wall_seconds"], 1)

    def test_killed_native_descendant_cost_does_not_disappear(self):
        meter = _Meter(ExperimentBudget(30, 10))
        worker = _Worker("agent", factory, profile(policy="child_hang").data, 1, meter)
        self.addCleanup(worker.close)
        self.assertIsNone(worker.initialize(2, time.monotonic() + 5))
        worker.channel.setblocking(True)
        worker.wire.send(("choose", Game.new(), LIMITS))
        worker.channel.setblocking(False)
        deadline = time.monotonic() + 2
        while _tree_usage(worker.process.pid)[2] < 0.04 and time.monotonic() < deadline:
            time.sleep(0.01)
        observed = meter.cost()["cpu_seconds"]
        self.assertGreater(_tree_usage(worker.process.pid)[2], 0)
        worker.close()
        after = meter.cost()
        self.assertGreaterEqual(after["cpu_seconds"], observed)
        self.assertGreater(after["conservative_shutdown_cpu_seconds"], 0)
        self.assertIn("sampled", after["cpu_scope"])


class ScheduleTests(unittest.TestCase):
    def schedule(self, **kwargs):
        return make_schedule(arms=(profile("candidate"), profile("control")),
                             profiles=(profile("opponent"), profile("other")), pairs=3, seed=3,
                             control_arm="control", **kwargs)

    def test_color_balance_common_independent_streams_and_order(self):
        schedule = self.schedule()
        self.assertEqual(len(schedule["slots"]), 24)
        self.assertEqual([p["weight"] for p in schedule["profiles"]], [0.5, 0.5])
        first, second = schedule["slots"][:4], schedule["slots"][4:8]
        self.assertEqual([(s["arm"], s["candidate_color"]) for s in first],
                         [("candidate", "B"), ("control", "B"), ("candidate", "W"), ("control", "W")])
        self.assertEqual([(s["arm"], s["candidate_color"]) for s in second],
                         [("control", "W"), ("candidate", "W"), ("control", "B"), ("candidate", "B")])
        for group in (first, second):
            self.assertEqual(len({(s["candidate_seed"], s["opponent_seed"]) for s in group}), 1)
            self.assertNotEqual(group[0]["candidate_seed"], group[0]["opponent_seed"])
        validate_schedule(schedule)

    def test_duplicate_missing_and_seed_reused_blocks_rejected(self):
        original = self.schedule()
        for mutation in (lambda s: s["slots"].pop(),
                         lambda s: s["slots"].append(copy.deepcopy(s["slots"][0])),
                         lambda s: s["slots"][0].update(opponent_seed=s["slots"][0]["candidate_seed"]),
                         lambda s: s["slots"][4].update(candidate_seed=s["slots"][0]["candidate_seed"]),
                         lambda s: s["profiles"][0].update(weight=0.9)):
            changed = copy.deepcopy(original)
            mutation(changed)
            with self.assertRaises(ValueError):
                validate_schedule(changed)
        with self.assertRaises(ValueError):
            self.schedule(scope="full")

    def test_confirmation_uses_new_seed_streams(self):
        prior = self.schedule()
        with self.assertRaises(ValueError):
            self.schedule(purpose="confirmation", prior_schedules=(prior,))
        with self.assertRaises(ValueError):
            self.schedule(purpose="confirmation")
        fresh = make_schedule(arms=(profile(),), profiles=(profile("opponent"),), pairs=1, seed=19,
                              purpose="confirmation", prior_schedules=(prior,))
        validate_schedule(fresh)

    def test_final_binding_requires_external_pin_before_execution(self):
        opponents = tuple(profile(f"final-{family}", policy="local_liberties" if family == "weak" else "script",
                                  family=family, panel="final") for family in ("weak", "pachi", "fuego"))
        schedule = make_schedule(arms=(profile(),), profiles=opponents, pairs=1, seed=14,
                                 purpose="final", scope="full")
        with self.assertRaisesRegex(ValueError, "E7 release"):
            validate_schedule(schedule, for_execution=True)
        release = release_binding(schedule)
        schedule.update(release=release, release_sha256=digest(release))
        with self.assertRaisesRegex(ValueError, "E7 release"):
            validate_schedule(schedule, for_execution=True)
        validate_schedule(schedule, for_execution=True, release_sha256=digest(release))
        changed = copy.deepcopy(schedule)
        changed["arm_profiles"][0]["checkpoint_sha256"] = "b" * 64
        with self.assertRaises(ValueError):
            validate_schedule(changed, for_execution=True, release_sha256=digest(release))
        changed = copy.deepcopy(schedule)
        changed["profiles"][0]["panel"] = "validation"
        with self.assertRaises(ValueError):
            validate_schedule(changed)

    def test_profile_freezes_input_and_weak_policy_is_unranked(self):
        data = profile().data
        frozen = AgentProfile.from_dict(data)
        data["context"]["timed"] = True
        self.assertFalse(frozen.data["context"]["timed"])
        with self.assertRaises(ValueError):
            profile(human_rank="15k")
        self.assertEqual(len(weak_profile("weak", source_revision=REVISION).data["pins"]["implementation"]), 64)


class WeakPolicyTests(unittest.TestCase):
    def test_uniform_private_seed_and_pass_only_when_no_placement(self):
        data = weak_profile("uniform", source_revision=REVISION).data
        first, second = _WeakAgent(data, 17), _WeakAgent(data, 17)
        game = Game.new()
        expected = [first.choose(game).action for _ in range(4)]
        self.assertEqual(expected, [second.choose(game).action for _ in range(4)])
        self.assertTrue(all(0 <= action < PASS for action in expected))
        board = list("W" * PASS)
        board[point(3, 3)] = board[point(15, 15)] = "."
        no_placement = Game.from_fixture(["".join(board[i:i + 19]) for i in range(0, PASS, 19)])
        self.assertEqual(first.choose(no_placement).action, PASS)
        self.assertEqual(first.choose(game).network_calls, 0)

    def test_capture_first_samples_only_captures_with_uniform_ties(self):
        data = weak_profile("capture", "capture_first", source_revision=REVISION).data
        game = Game.from_fixture(rows({(2, 3), (4, 3), (3, 2), (14, 15), (16, 15), (15, 14)},
                                       {(3, 3), (15, 15)}))
        agent = _WeakAgent(data, 11)
        choices = [agent.choose(game).action for _ in range(12)]
        self.assertEqual(set(choices), {point(3, 4), point(15, 16)})
        self.assertLess(_WeakAgent(data, 11).choose(Game.new()).action, PASS)


class RunnerTests(unittest.TestCase):
    def run_fixture(self, initial=None, candidate=None, opponent=None, **kwargs):
        return run_game(slot(initial.candidate_color if initial is not None else "B"),
                        candidate or profile(), opponent or profile("opponent"),
                        budget=kwargs.pop("budget", BUDGET), limits=kwargs.pop("limits", LIMITS),
                        agent_factories=FACTORIES, initial=initial, allow_fixture=initial is not None, **kwargs)

    def test_exact_scores_resignation_and_empty_unresolved(self):
        for captures, score in ((0, 13), (8, -3)):
            initial = Game.from_fixture(dense(), captured_white=captures, candidate_color="B")
            result = self.run_fixture(initial)
            self.assertEqual(result.record["white_minus_black_halfpoints"], score)
            self.assertEqual(result.record["result"], "W" if score < 0 else "L")
            self.assertEqual(result.record["original_classification"], "pending_adjudication")
            self.assertEqual(result.record["behavior"]["premature_pass"], False)
            self.assertEqual(result.game, load_game(dump_game(result.game), allow_fixture=True))
            passed = result.record["candidate_passes"][0]
            _validate_pass_record(passed, initial, LIMITS, None, frozenset())
            changed = copy.deepcopy(passed)
            changed["verdict"] = changed["attempts"][-1]["verdict"] = "premature"
            with self.assertRaises(ValueError):
                _validate_pass_record(changed, initial, LIMITS, None, frozenset())
        resigned = self.run_fixture(candidate=profile(actions=["resign"]))
        self.assertEqual((resigned.record["result"], resigned.record["actual_actions"]), ("L", 0))
        self.assertIsNone(resigned.record["white_minus_black_halfpoints"])
        empty = self.run_fixture()
        self.assertEqual((empty.record["result"], empty.record["actual_actions"]), ("U", 2))
        self.assertIsNone(empty.record["behavior"]["premature_pass"])
        self.assertIsNone(empty.record["behavior"]["length_tail"])
        self.assertEqual(empty.record["candidate_passes"][0]["snapshot"], Game.new(candidate_color="B").snapshot)
        white = self.run_fixture(Game.new(candidate_color="W"))
        self.assertEqual(white.record["candidate_passes"][0]["snapshot"],
                         Game.new(candidate_color="W").play("B", PASS).snapshot)
        self.assertIsNone(white.record["behavior"]["premature_pass"])

    def test_white_gets_opponent_opening_and_candidate_snapshot(self):
        game = Game.new(candidate_color="W")
        result = self.run_fixture(game, candidate=profile(actions=[0, "resign"]),
                                  opponent=profile("opponent", actions=[0]))
        self.assertEqual([(m.color, m.action) for m in result.game.moves], [("B", 0)])
        self.assertEqual(result.record["result"], "L")
        self.assertEqual([d["color"] for d in result.record["decisions"]], ["B", "W"])

    def test_agent_faults_forfeit_unknown_faults_stay_unresolved(self):
        for policy, expected, runtime in (("exception", "L", True), ("crash", "L", True),
                                          ("hang", "L", True), ("infrastructure", "U", None)):
            started = time.monotonic()
            result = self.run_fixture(candidate=profile(policy=policy),
                                      limits=replace(LIMITS, response_seconds=0.3))
            self.assertLess(time.monotonic() - started, 3)
            self.assertEqual(result.record["result"], expected, result.record)
            self.assertEqual(result.record["behavior"]["runtime_failure"], runtime)
            self.assertEqual(result.game.status, "invalid")
        illegal = self.run_fixture(candidate=profile(actions=[True]))
        self.assertEqual((illegal.record["result"], illegal.record["reason"]), ("L", "invalid_agent_response"))
        occupied = self.run_fixture(candidate=profile(actions=[0]), opponent=profile("opponent", actions=[0]))
        self.assertEqual((occupied.record["result"], occupied.record["reason"]), ("W", "illegal_action"))
        self.assertIsNone(occupied.record["behavior"]["runtime_failure"])
        self.assertIsNone(occupied.record["behavior"]["length_tail"])
        self.assertEqual(occupied.record["fault"]["responsible"], "W")

    def test_budget_boundary_never_assigns_an_agent_forfeit(self):
        result = self.run_fixture(budget=ExperimentBudget(cpu_seconds=2, wall_seconds=0.2))
        self.assertEqual((result.game.status, result.record["result"]), ("truncated", "U"))
        self.assertEqual(result.game.position.to_play, "B")
        white = self.run_fixture(Game.new(candidate_color="W"), budget=ExperimentBudget(cpu_seconds=2, wall_seconds=0.2))
        self.assertEqual((white.game.status, white.record["result"]), ("invalid", "U"))
        self.assertIsNone(white.record["behavior"]["runtime_failure"])

    def test_empirical_original_pending_state_scores_spread_and_authorization(self):
        game = Game.new(candidate_color="B").play("B", 20).play("W", 21)
        result = self.run_fixture(game, referee_factory=referee_factory, referee_profile=REFEREE)
        self.assertEqual((result.record["result"], result.record["outcome_kind"]), ("L", "empirical_score"))
        self.assertEqual(result.record["disagreement"]["cleanup_scores_halfpoints"], [13, 11])
        self.assertEqual(result.record["disagreement"]["spread_halfpoints"], 2)
        self.assertIsNone(result.record["behavior"]["dispute"])
        self.assertEqual(len(result.game.moves), 4)
        self.assertEqual(len(result.record["adjudication_attempts"]), 2)
        with self.assertRaises(ValueError):
            load_game(dump_game(result.game), empirical_referee=FakeReferee())
        authorized = {digest(json.loads(result.game.outcome.evidence))}
        self.assertEqual(load_game(dump_game(result.game), empirical_referee=FakeReferee(),
                                   authorized_hashes=authorized), result.game)
        disputed = self.run_fixture(game, referee_factory=disagreement_factory, referee_profile=REFEREE)
        self.assertEqual(disputed.record["result"], "U")
        self.assertTrue(disputed.record["behavior"]["dispute"])
        self.assertEqual(disputed.record["disagreement"]["cleanup_scores_halfpoints"], [13, -3])

    def test_judgment_hang_is_bounded_unjudgeable_and_unresolved(self):
        game = Game.new(candidate_color="B").play("B", 20).play("W", 21)
        started = time.monotonic()
        result = self.run_fixture(game, referee_factory=hanging_referee_factory, referee_profile=REFEREE,
                                  limits=replace(LIMITS, empirical_wall_seconds=0.2))
        self.assertLess(time.monotonic() - started, 4)
        self.assertEqual(result.record["result"], "U")
        self.assertIsNone(result.record["behavior"]["premature_pass"])
        self.assertFalse(result.record["behavior"]["runtime_failure"])

    def test_cap_boundaries_full_history_and_terminal_precedence(self):
        actions = cap_actions()
        for color, prefix, expected in (("B", 798, 800), ("W", 799, 801)):
            game = Game.from_fixture(ko_patches(), candidate_color=color)
            for action in actions[:prefix]:
                game = game.play(game.position.to_play, action)
            continued = self.run_fixture(game, candidate=profile(actions=actions),
                                         opponent=profile("opponent", actions=actions))
            self.assertEqual((continued.game.status, len(continued.game.moves)), ("truncated", expected))
            self.assertEqual(continued.game.position.to_play, color)
            self.assertEqual(continued.record["result"], "U")
            self.assertEqual(load_game(dump_game(continued.game), allow_fixture=True), continued.game)
        game = Game.from_fixture(ko_patches(), candidate_color="B")
        for action in actions[:798]:
            game = game.play(game.position.to_play, action)
        result = self.run_fixture(game)
        self.assertEqual(result.record["original_classification"], "pending_adjudication")
        self.assertFalse(result.record["behavior"]["cap"])
        game = Game.from_fixture(ko_patches(True), candidate_color="B")
        result = self.run_fixture(game, candidate=profile(actions=CYCLE), opponent=profile("opponent", actions=CYCLE))
        self.assertEqual((result.record["reason"], len(result.game.moves)), ("recurrence", 6))
        self.assertEqual(load_game(dump_game(result.game), allow_fixture=True), result.game)

    def test_output_ownership_missing_slots_and_replay(self):
        schedule = make_schedule(arms=(profile(actions=["resign"]),), profiles=(profile("opponent"),),
                                 pairs=1, seed=3, limits=LIMITS)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "new"
            records = run_schedule(schedule, root, budget=BUDGET, agent_factories=FACTORIES)
            self.assertEqual(len(records), 2)
            loaded_schedule, loaded = read_records(root)
            self.assertEqual(loaded_schedule, schedule)
            self.assertEqual(loaded, records)
            with self.assertRaises(FileExistsError):
                run_schedule(schedule, root, budget=BUDGET, agent_factories=FACTORIES)
            limited = Path(temp) / "limited"
            self.assertEqual(run_schedule(schedule, limited, budget=ExperimentBudget(1, 0.1),
                                          agent_factories=FACTORIES), {})
            self.assertEqual(json.loads((limited / "run.json").read_text())["result_counts"]["U"], 2)
            record_path = root / "s00000000.json"
            changed = json.loads(record_path.read_text())
            changed["result"] = "W"
            record_path.write_text(canonical(changed))
            with self.assertRaises(ValueError):
                read_records(root)

    def test_production_rejects_fixture_and_unknown_gpu_stays_unknown(self):
        with self.assertRaises(ValueError):
            run_game(slot(), profile(), profile("opponent"), budget=BUDGET,
                     initial=Game.new(candidate_color="B"), agent_factories=FACTORIES)
        result = self.run_fixture(candidate=profile(actions=["resign"], device="unknown"))
        self.assertIsNone(result.record["cost"]["gpu_seconds"])
        self.assertGreater(result.record["cost"]["cpu_seconds"], 0)

    def test_malformed_decisions_persist_forfeits_and_complete_the_schedule(self):
        for policy in ("invalid_calls", "invalid_gpu", "invalid_action"):
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as temp:
                schedule = make_schedule(arms=(profile(policy=policy),), profiles=(profile("opponent"),),
                                         pairs=1, seed=3, limits=LIMITS)
                root = Path(temp) / "matches"
                records = run_schedule(schedule, root, budget=BUDGET, agent_factories=FACTORIES)
                self.assertEqual(len(records), 2)
                self.assertEqual(read_records(root)[1], records)
                for record in records.values():
                    self.assertEqual(record["result"], "L")
                    self.assertTrue(record["behavior"]["runtime_failure"])
                    self.assertEqual(record["reason"], "invalid_agent_response")
                    self.assertIsNone(record["decisions"][-1]["network_calls"])
                    self.assertIsNone(record["decisions"][-1]["gpu_seconds"])

    def test_agent_initialization_faults_retain_responsibility(self):
        for color in ("B", "W"):
            with self.subTest(color=color):
                candidate = self.run_fixture(candidate=profile(policy="initialization_exception"),
                                             initial=Game.new(candidate_color=color))
                self.assertEqual(candidate.record["result"], "L")
                self.assertTrue(candidate.record["behavior"]["runtime_failure"])
                opponent = self.run_fixture(opponent=profile("opponent", policy="initialization_exception"),
                                            initial=Game.new(candidate_color=color))
                self.assertEqual(opponent.record["result"], "W")
                self.assertIsNone(opponent.record["behavior"]["runtime_failure"])
        unknown = self.run_fixture(candidate=profile(policy="initialization_infrastructure"))
        self.assertEqual(unknown.record["result"], "U")
        self.assertIsNone(unknown.record["behavior"]["runtime_failure"])

    def test_abrupt_agent_initialization_exits_and_timeouts_are_attributed(self):
        for policy in ("initialization_exit", "initialization_hang"):
            for color in ("B", "W"):
                with self.subTest(policy=policy, color=color):
                    candidate = self.run_fixture(candidate=profile(policy=policy),
                                                 initial=Game.new(candidate_color=color))
                    self.assertEqual(candidate.record["result"], "L")
                    self.assertTrue(candidate.record["behavior"]["runtime_failure"])
                    opponent = self.run_fixture(opponent=profile("opponent", policy=policy),
                                                initial=Game.new(candidate_color=color))
                    self.assertEqual(opponent.record["result"], "W")
                    self.assertIsNone(opponent.record["behavior"]["runtime_failure"])

    def test_saved_behavior_and_observation_coverage_match_replay(self):
        schedule = make_schedule(arms=(profile(),), profiles=(profile("opponent"),), pairs=1, seed=3, limits=LIMITS)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "matches"
            records = run_schedule(schedule, root, budget=BUDGET, agent_factories=FACTORIES)
            name = next(iter(records)); original = records[name]; path = root / f"{name}.json"
            mutations = [lambda r: r["behavior"].update(cap=True),
                         lambda r: r["behavior"].update(unresolved=False),
                         lambda r: r["behavior"].update(length_tail=False),
                         lambda r: r["behavior"].update(premature_pass=False),
                         lambda r: r.update(decisions=[]), lambda r: r.update(candidate_passes=[]),
                         lambda r: r["candidate_passes"][0].update(verdict="acceptable")]
            for index, mutate in enumerate(mutations):
                with self.subTest(index=index):
                    changed = copy.deepcopy(original); mutate(changed)
                    path.write_text(canonical(changed))
                    with self.assertRaises(ValueError):
                        read_records(root)
            path.write_text(canonical(original))
            self.assertEqual(read_records(root)[1], records)

    def test_saved_empirical_referee_must_match_the_scheduled_pin(self):
        schedule = make_schedule(arms=(profile(),), profiles=(profile("opponent"),), pairs=1, seed=3,
                                 limits=LIMITS, referee_profile=REFEREE)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "matches"
            records = run_schedule(schedule, root, budget=BUDGET, agent_factories=FACTORIES,
                                   referee_factory=referee_factory)
            authorized = {digest(json.loads(record["state"]["ending"]["outcome"]["evidence"]))
                          for record in records.values()}
            self.assertEqual(read_records(root, authorized_hashes=authorized, empirical_referee=FakeReferee())[1],
                             json.loads(canonical(records)))
            from kenggo.match_sgf import export_match_sgf
            import hashlib
            changed = copy.deepcopy(schedule); changed["referee_profile"]["sha256"] = "e" * 64
            (root / "schedule.json").write_text(canonical(changed))
            for name, record in records.items():
                record["referee_profile"] = changed["referee_profile"]
                game = load_game(canonical(record["state"]), authorized_hashes=authorized,
                                 empirical_referee=FakeReferee())
                data = export_match_sgf(game, metadata={"slot": record["slot"], "schedule_sha256": digest(changed)})
                (root / record["sgf_file"]).write_bytes(data)
                record["sgf_sha256"] = hashlib.sha256(data).hexdigest()
                (root / f"{name}.json").write_text(canonical(record))
            with self.assertRaises(ValueError):
                read_records(root, authorized_hashes=authorized, empirical_referee=FakeReferee())

    def test_known_empirical_passes_need_independently_pinned_evidence(self):
        schedule = make_schedule(arms=(profile(),), profiles=(profile("opponent"),), pairs=1, seed=3,
                                 limits=LIMITS, referee_profile=REFEREE)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "matches"
            records = run_schedule(schedule, root, budget=BUDGET, agent_factories=FACTORIES,
                                   referee_factory=known_pass_referee_factory)
            ending_pins = {digest(json.loads(r["state"]["ending"]["outcome"]["evidence"])) for r in records.values()}
            pass_pins = {digest(p["attempts"][-1]["evidence"]) for r in records.values() for p in r["candidate_passes"]}
            with self.assertRaisesRegex(ValueError, "independently pinned"):
                read_records(root, authorized_hashes=ending_pins, empirical_referee=KnownPassReferee())
            pins = ending_pins | pass_pins
            self.assertEqual(read_records(root, authorized_hashes=pins, empirical_referee=KnownPassReferee())[1], records)
            name = next(iter(records)); changed = copy.deepcopy(records[name]); passed = changed["candidate_passes"][0]
            passed["verdict"] = passed["attempts"][-1]["verdict"] = "premature"
            passed["attempts"][-1]["evidence"]["status"] = "premature"
            changed["behavior"]["premature_pass"] = True
            (root / f"{name}.json").write_text(canonical(changed))
            with self.assertRaisesRegex(ValueError, "independently pinned"):
                read_records(root, authorized_hashes=pins, empirical_referee=KnownPassReferee())

    def test_unresolved_empirical_evidence_keeps_its_scheduled_profile(self):
        from kenggo.match_sgf import export_match_sgf
        import hashlib
        schedule = make_schedule(arms=(profile(),), profiles=(profile("opponent"),), pairs=1, seed=3,
                                 limits=LIMITS, referee_profile=REFEREE)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "matches"
            records = run_schedule(schedule, root, budget=BUDGET, agent_factories=FACTORIES,
                                   referee_factory=disagreement_factory)
            self.assertEqual(read_records(root)[1], records)
            changed = copy.deepcopy(schedule); changed["referee_profile"]["sha256"] = "e" * 64
            (root / "schedule.json").write_text(canonical(changed))
            for name, record in records.items():
                record["referee_profile"] = changed["referee_profile"]
                game = load_game(canonical(record["state"]))
                data = export_match_sgf(game, metadata={"slot": record["slot"], "schedule_sha256": digest(changed)})
                (root / record["sgf_file"]).write_bytes(data)
                record["sgf_sha256"] = hashlib.sha256(data).hexdigest()
                (root / f"{name}.json").write_text(canonical(record))
            with self.assertRaisesRegex(ValueError, "empirical score evidence"):
                read_records(root)

    def test_partial_result_write_keeps_completed_records_recoverable(self):
        schedule = make_schedule(arms=(profile(actions=["resign"]),), profiles=(profile("opponent"),),
                                 pairs=1, seed=3, limits=LIMITS)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "matches"
            records = run_schedule(schedule, root, budget=BUDGET, agent_factories=FACTORIES)
            first, interrupted = list(records)
            target = root / f"{interrupted}.json"
            payload = target.read_bytes()
            target.unlink()
            real_open = Path.open

            class PartialWrite:
                def __init__(self, stream):
                    self.stream = stream

                def __enter__(self):
                    return self

                def write(self, data):
                    self.stream.write(data[:128])
                    self.stream.flush()
                    raise OSError("injected interrupted write")

                def __exit__(self, *args):
                    self.stream.close()

            def failing_open(path, *args, **kwargs):
                stream = real_open(path, *args, **kwargs)
                return PartialWrite(stream) if args and args[0] == "xb" else stream

            with patch.object(Path, "open", failing_open):
                with self.assertRaisesRegex(OSError, "injected interrupted"):
                    _write_new(target, payload, BUDGET, 0)
            self.assertFalse(target.exists())
            (root / f".{interrupted}.json.abandoned.partial").write_bytes(payload[:128])
            loaded_schedule, loaded = read_records(root)
            self.assertEqual(loaded_schedule, schedule)
            self.assertEqual(loaded, {first: records[first]})
            self.assertEqual(len(schedule["slots"]) - len(loaded), 1)
            existing = (root / f"{first}.json").read_bytes()
            with self.assertRaises(FileExistsError):
                _write_new(root / f"{first}.json", b"{}", BUDGET, 0)
            self.assertEqual((root / f"{first}.json").read_bytes(), existing)


if __name__ == "__main__":
    unittest.main()
