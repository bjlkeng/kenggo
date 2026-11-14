"""Synthetic Human SL protocol tests; no neural-quality or strength evidence."""

from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from kenggo.go import Game, PASS, canonical
from kenggo.katago_human import (ENGINE_CONTEXT, ENGINE_REVISION, ENGINE_VERSION, POLICY,
    POLICY_MASS_TOLERANCE, RANKS, REQUIRED_CONFIG, SETTINGS, HumanMatchFactory, HumanProfile,
    HumanSL, ReferenceBudget, ReferenceBudgetExceeded, ReferenceError, human_agent_profile)
from kenggo.katago_referee import RULES, _action, _coordinate
from kenggo.matches import (ExperimentBudget, InfrastructureError, MatchLimits,
                            run_game, weak_profile)
from go_fixtures import ko_patches, point, two_passes


def replay_request(request):
    board = ["."] * PASS
    for color, coordinate in request["initialStones"]:
        board[_action(coordinate)] = color
    game = Game.from_fixture(["".join(board[i:i + 19]) for i in range(0, PASS, 19)],
                             to_play=request["initialPlayer"])
    for color, coordinate in request["moves"]:
        game = game.play(color, _action(coordinate))
    return game


def response(request, *, white_winrate=0.8):
    if request.get("action") == "query_version":
        return {**request, "version": ENGINE_VERSION, "git_hash": ENGINE_REVISION}
    game = replay_request(request)
    policy = [0.0 if legal else -1.0 for legal in game.legal_mask()]
    policy[PASS] = 1.0
    return {"id": request["id"], "turnNumber": len(request["moves"]), "isDuringSearch": False,
            "rootInfo": {"currentPlayer": game.position.to_play, "visits": 1,
                         "humanWinrate": white_winrate, "winrate": 0.03, "rawWinrate": 0.04},
            "humanPolicy": policy, "policy": [1 / 362] * 362, "moveInfos": [{"move": "A19"}]}


class FakeTransport:
    def __init__(self, handler=response):
        self.handler, self.requests, self.closed = handler, [], False
        self.last_response, self.stderr_tail = b"", b""

    def exchange(self, raw, budget):
        request = json.loads(raw)
        self.requests.append(request)
        result = self.handler(request)
        self.last_response = result if isinstance(result, bytes) else (canonical(result) + "\n").encode()
        budget.response_bytes += len(self.last_response)
        budget.check()
        return self.last_response

    def cpu_seconds(self):
        return 0.0

    def close(self):
        self.closed = True


class HumanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = {name: self.root / name for name in ("engine", "main_model", "human_model", "config")}
        for name, path in self.paths.items():
            path.write_text(name if name != "config" else "\n".join(
                f"{key} = {value}" for key, value in REQUIRED_CONFIG.items()) + "\n")
        self.settings = {"schema": "katago_human_profile_v1", "profile_id": POLICY,
            "engine_version": ENGINE_VERSION, "engine_revision": ENGINE_REVISION,
            "main_model_name": "kata1-b10c128-s1141046784-d204142634",
            "human_model_name": "b18c384nbt-humanv0", "artifacts": {}, "runtime_files": {},
            "library_dirs": [], "ranks": list(RANKS), "settings": dict(SETTINGS),
            "engine_context": dict(ENGINE_CONTEXT)}
        self.repin()

    def repin(self):
        self.settings["artifacts"] = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                                      for name, path in self.paths.items()}

    def profile(self):
        raw = (canonical(self.settings) + "\n").encode()
        path = self.root / "profile.json"
        path.write_bytes(raw)
        return HumanProfile.load(path, hashlib.sha256(raw).hexdigest())

    def client(self, handler=response, *, budget=None, trace_sink=None, real=False):
        client = HumanSL(self.profile(), **self.paths, runtime_root=self.root, trace_sink=trace_sink,
            budget=budget or ReferenceBudget(100, 10, 20),
            _analysis=None if real else FakeTransport(handler))
        self.addCleanup(client.close)
        return client

    def script(self, code):
        self.paths["engine"].write_text(f"#!{sys.executable}\n" + code)
        self.paths["engine"].chmod(0o700)
        self.repin()

    def factory(self, budget=None):
        return HumanMatchFactory(self.profile(), **{key: str(path) for key, path in self.paths.items()},
            budget=budget or ReferenceBudget(5, 3, 5), source_revision="c" * 40,
            runtime_root=str(self.root))

    def version_only_script(self):
        self.script("import json,sys\n"
            "for line in sys.stdin:\n"
            " r=json.loads(line)\n"
            " if r.get('action')=='query_version':\n"
            f"  r.update(version={ENGINE_VERSION!r},git_hash={ENGINE_REVISION!r})\n"
            " else: r={}\n"
            " print(json.dumps(r),flush=True)\n")

    def test_manifest_is_externally_pinned_and_immutable(self):
        profile = self.profile()
        self.assertEqual(profile.data["ranks"], list(RANKS))
        with self.assertRaisesRegex(ReferenceError, "profile_hash_mismatch"):
            HumanProfile.load(self.root / "profile.json", "0" * 64)
        with self.assertRaises(FrozenInstanceError):
            profile.sha256 = "0" * 64
        copied = profile.data
        copied["settings"]["pass_correction"] = True
        self.assertFalse(profile.data["settings"]["pass_correction"])
        duplicate = b'{"schema":1,"schema":2}'
        (self.root / "profile.json").write_bytes(duplicate)
        with self.assertRaises(ReferenceError):
            HumanProfile.load(self.root / "profile.json", hashlib.sha256(duplicate).hexdigest())

    def test_manifest_rejects_wrong_context_schema_and_pins(self):
        original = json.loads(canonical(self.settings))
        changes = [("engine_revision", "0" * 40), ("engine_version", "1.18.0"),
                   ("ranks", ["5k"]), ("artifacts", {}), ("runtime_files", {"../escape": "0" * 64}),
                   ("library_dirs", ["."]), ("settings", {**SETTINGS, "pass_correction": True}),
                   ("settings", {**SETTINGS, "max_visits": True}), ("engine_context", {})]
        for field, value in changes:
            with self.subTest(field=field, value=value):
                self.settings = {**original, field: value}
                with self.assertRaises(ReferenceError):
                    self.profile()
        self.settings = {**original, "extra": "unversioned"}
        with self.assertRaises(ReferenceError):
            self.profile()

    def test_artifacts_and_config_semantics_are_verified(self):
        for name in self.paths:
            with self.subTest(name=name):
                original = self.paths[name].read_bytes()
                self.paths[name].write_bytes(b"changed")
                with self.assertRaisesRegex(ReferenceError, "artifact_hash_mismatch:" + name) as error:
                    self.client()
                self.assertTrue(error.exception.cost["closed"])
                self.paths[name].write_bytes(original)
        for extra in ("humanSLProfile = rank_1d", "ignoreAllHistory = true", "not a config entry"):
            with self.subTest(extra=extra):
                self.paths["config"].write_text("\n".join(
                    f"{key} = {value}" for key, value in REQUIRED_CONFIG.items()) + "\n" + extra)
                self.repin()
                with self.assertRaises(ReferenceError):
                    self.client()

    def test_runtime_must_be_pinned_and_stay_within_root(self):
        directory = self.root / "lib"
        directory.mkdir()
        library = directory / "fixture.so"
        library.write_bytes(b"fixture")
        self.settings["library_dirs"] = ["lib"]
        with self.assertRaisesRegex(ReferenceError, "unpinned_runtime_library"):
            self.client()
        self.settings["runtime_files"] = {"lib/fixture.so": hashlib.sha256(b"fixture").hexdigest()}
        self.client()
        library.write_bytes(b"changed")
        with self.assertRaisesRegex(ReferenceError, "runtime_hash_mismatch"):
            self.client()
        library.unlink()
        library.symlink_to(sys.executable)
        self.settings["runtime_files"]["lib/fixture.so"] = hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()
        with self.assertRaisesRegex(ReferenceError, "runtime_hash_mismatch"):
            self.client()

    def test_mapping_full_history_pass_and_mover_value(self):
        client = self.client()
        game = Game.new().play("B", _action("D4")).play("W", _action("Q16")).play("B", PASS)
        before = game.action_record()
        prediction = client.predict(game, "5k")
        request = client.analysis.requests[-1]
        self.assertEqual(request["rules"], RULES)
        self.assertEqual(request["moves"], [["B", "D4"], ["W", "Q16"], ["B", "pass"]])
        self.assertEqual(request["initialStones"], [])
        self.assertEqual(request["initialPlayer"], "B")
        self.assertEqual((request["boardXSize"], request["boardYSize"], request["komi"]), (19, 19, 6.5))
        self.assertEqual(request["overrideSettings"], {"humanSLProfile": "rank_5k", "ignorePreRootHistory": False})
        self.assertTrue(request["includePolicy"])
        self.assertEqual(request["maxVisits"], 1)
        self.assertEqual(game.action_record(), before)
        self.assertEqual(prediction.snapshot, game.snapshot)
        self.assertAlmostEqual(prediction.value, 0.6)
        self.assertAlmostEqual(client.predict(Game.new(), "15k").value, -0.6)
        self.assertEqual(client.predict(Game.new(), "1d").rank, "1d")
        self.assertEqual(prediction.raw_policy[_action("D4")], -1)
        self.assertEqual(prediction.policy[_action("D4")], 0)

    def test_full_fixture_history_preserves_simple_ko_and_pass_clearing(self):
        client = self.client()
        fixture = Game.from_fixture(ko_patches()).play("B", point(3, 4))
        with self.assertRaisesRegex(ReferenceError, "fixture_requires_explicit_authorization"):
            client.predict(fixture, "5k")
        ko = client.predict(fixture, "5k", allow_fixture=True)
        self.assertEqual(ko.raw_policy[point(3, 3)], -1)
        self.assertEqual(client.analysis.requests[-1]["initialStones"],
                         [[color, _coordinate(action)] for action, color in enumerate(fixture.positions[0].board)
                          if color != "."])
        after_pass = fixture.play("W", PASS).play("B", point(16, 15))
        restored = client.predict(after_pass, "5k", allow_fixture=True)
        self.assertEqual(restored.raw_policy[point(3, 3)], 0)
        self.assertEqual(len(client.analysis.requests[-1]["moves"]), 3)

    def test_startup_profile_never_substitutes_for_required_query_rank(self):
        self.assertEqual(REQUIRED_CONFIG.get("humanSLProfile"), "rank_5k")
        client = self.client()
        with self.assertRaises(TypeError):
            client.predict(Game.new())
        for rank in ("15k", "1d"):
            prediction = client.predict(Game.new(), rank)
            self.assertEqual(prediction.rank, rank)
            self.assertEqual(client.analysis.requests[-1]["overrideSettings"]["humanSLProfile"],
                             "rank_" + rank)

    def test_reject_ended_states_offsets_and_forged_history(self):
        client = self.client()
        with self.assertRaises(TypeError):
            client.predict(Game.new())
        fixture = Game.from_fixture(ko_patches(), captured_black=1)
        with self.assertRaisesRegex(ReferenceError, "initial_capture_offset_unsupported"):
            client.predict(fixture, "5k", allow_fixture=True)
        for game in (two_passes(Game.new()), Game.new().resign("B"), Game.new().fail("fixture")):
            with self.assertRaisesRegex(ReferenceError, "requires_playing_game"):
                client.predict(game, "5k")
        moved = Game.new().play("B", 0)
        with self.assertRaisesRegex(ReferenceError, "game_history_mismatch"):
            client.predict(replace(moved, positions=Game.new().positions), "5k")
        for rank in ("16k", "2d", "rank_5k", None, 5):
            with self.assertRaisesRegex(ReferenceError, "unsupported_conditioning_rank"):
                client.predict(Game.new(), rank)
        self.assertEqual(client.cost()["queries"], 0)

    def test_policy_is_not_corrected_or_renormalized_and_sampling_is_private(self):
        def mixed(request):
            result = response(request)
            result["humanPolicy"][0], result["humanPolicy"][PASS] = 0.75, 0.25
            return result

        prediction = self.client(mixed).predict(Game.new(), "5k")
        state = random.getstate()
        first = random.Random(982)
        actual = [prediction.sample(first) for _ in range(1000)]
        second = random.Random(982)
        expected = [0 if second.random() < 0.75 else PASS for _ in range(1000)]
        self.assertEqual(actual, expected)
        self.assertIn(PASS, actual)
        self.assertEqual(random.getstate(), state)
        with self.assertRaises(TypeError):
            prediction.sample(random)
        with self.assertRaises(FrozenInstanceError):
            prediction.value = 0

        def rounded(request):
            result = response(request)
            result["humanPolicy"][PASS] = 1 - POLICY_MASS_TOLERANCE / 2
            return result

        output = self.client(rounded).predict(Game.new(), "5k")
        self.assertEqual(output.policy[PASS], 1 - POLICY_MASS_TOLERANCE / 2)
        self.assertEqual(output.sample(random.Random(1)), PASS)

    def test_strict_response_fields_legality_and_human_value(self):
        game = Game.new().play("B", 0)
        changes = [lambda r: r.update(id="stale"), lambda r: r.update(turnNumber=0),
                   lambda r: r.update(turnNumber=True), lambda r: r.update(isDuringSearch=True),
                   lambda r: r["rootInfo"].update(currentPlayer="B"),
                   lambda r: r["rootInfo"].update(visits=True),
                   lambda r: r["rootInfo"].update(visits=2),
                   lambda r: r["rootInfo"].pop("humanWinrate"),
                   lambda r: r["rootInfo"].update(humanWinrate=True),
                   lambda r: r["rootInfo"].update(humanWinrate=1.1),
                   lambda r: r.pop("humanPolicy"), lambda r: r["humanPolicy"].pop(),
                   lambda r: r["humanPolicy"].__setitem__(0, 0),
                   lambda r: r["humanPolicy"].__setitem__(1, -1),
                   lambda r: r["humanPolicy"].__setitem__(1, True),
                   lambda r: r["humanPolicy"].__setitem__(PASS, -1),
                   lambda r: r["humanPolicy"].__setitem__(PASS, 0),
                   lambda r: r.update(warning="changed profile")]
        for index, mutate in enumerate(changes):
            with self.subTest(index=index):
                def bad(request):
                    result = response(request)
                    mutate(result)
                    return result
                client = self.client(bad)
                with self.assertRaises(ReferenceError):
                    client.predict(game, "5k")
                self.assertTrue(client.analysis.closed)
                self.assertTrue(client.traces[-1]["error"])
                with self.assertRaisesRegex(ReferenceError, "reference_closed"):
                    client.predict(game, "5k")
                self.assertEqual(len(client.analysis.requests), 1)

    def test_bad_json_preserves_failed_raw_trace_and_cost(self):
        for raw in (b'{"id":1,"id":2}\n', b'{"humanPolicy":[NaN]}\n',
                    b'{"unused":1e999}\n', b"invalid\n", b"[]\n"):
            with self.subTest(raw=raw):
                sink = []
                client = self.client(lambda _: raw, trace_sink=sink.append)
                with self.assertRaises(ReferenceError) as error:
                    client.predict(Game.new(), "5k")
                self.assertEqual(client.cost()["queries"], 1)
                self.assertEqual(client.cost()["response_bytes"], len(raw))
                self.assertEqual(error.exception.cost, client.cost())
                self.assertEqual(sink[-1]["response_sha256"], hashlib.sha256(raw).hexdigest())
                if raw != b"[]\n":
                    self.assertEqual(sink[-1]["raw_response"], raw.decode())
                self.assertTrue(client.cost()["closed"])

    def test_oversized_numeric_replies_close_and_retain_both_fields(self):
        for field in ("humanWinrate", "humanPolicy"):
            for number in (10**400, -(10**400)):
                with self.subTest(field=field, sign=number > 0):
                    def oversized(request):
                        result = response(request)
                        if field == "humanWinrate":
                            result["rootInfo"][field] = number
                        else:
                            result[field][0] = number
                        return result
                    client = self.client(oversized)
                    with self.assertRaises(ReferenceError) as caught:
                        client.predict(Game.new(), "5k")
                    self.assertTrue(client.cost()["closed"])
                    self.assertTrue(client.analysis.closed)
                    self.assertEqual(caught.exception.cost, client.cost())
                    self.assertEqual(len(client.traces), 1)
                    self.assertTrue(client.traces[0]["error"])
                    with self.assertRaisesRegex(ReferenceError, "reference_closed"):
                        client.predict(Game.new(), "5k")
                    self.assertEqual(len(client.analysis.requests), 1)

    def test_numeric_overflow_kills_real_synthetic_child(self):
        reply = {"turnNumber": 0, "isDuringSearch": False,
                 "rootInfo": {"currentPlayer": "B", "visits": 1, "humanWinrate": 10**400},
                 "humanPolicy": [0.0] * PASS + [1.0]}
        self.script("import json,sys,time\n"
                    "request=json.loads(sys.stdin.readline())\n"
                    f"reply={reply!r}\n"
                    "reply['id']=request['id']\n"
                    "print(json.dumps(reply),flush=True)\n"
                    "time.sleep(10)\n")
        client = self.client(real=True)
        with self.assertRaises(ReferenceError) as caught:
            client.predict(Game.new(), "5k")
        self.assertTrue(client.cost()["closed"])
        self.assertIsNotNone(client.analysis.process.returncode)
        self.assertEqual(caught.exception.cost, client.cost())
        self.assertEqual(client.traces[0]["response"]["rootInfo"]["humanWinrate"], 10**400)
        with self.assertRaisesRegex(ReferenceError, "reference_closed"):
            client.predict(Game.new(), "5k")
        self.assertEqual(client.cost()["requests"], 1)

    def test_version_pins_do_not_consume_analysis_queries(self):
        client = self.client(budget=ReferenceBudget(1, 2, 3))
        self.assertEqual(client.query_version()["git_hash"], ENGINE_REVISION)
        self.assertEqual((client.cost()["queries"], client.cost()["requests"]), (0, 1))
        client.predict(Game.new(), "5k")
        with self.assertRaises(ReferenceBudgetExceeded):
            client.predict(Game.new(), "5k")
        self.assertEqual((client.cost()["queries"], client.cost()["requests"]), (1, 2))
        for field, value in (("version", "1.18.0"), ("git_hash", "<omitted>"), ("git_hash", "0" * 40),
                             ("id", "stale"), ("action", "analysis")):
            with self.subTest(field=field, value=value):
                client = self.client(lambda request: {**response(request), field: value})
                with self.assertRaisesRegex(ReferenceError, "engine_version_mismatch"):
                    client.query_version()

    def test_finite_budgets_and_cumulative_request_response_trace_limits(self):
        for kwargs in ({"queries": -1}, {"queries": True}, {"wall_seconds": float("inf")},
                       {"cpu_seconds": 0}, {"request_bytes": 0}, {"trace_bytes": 4095}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ReferenceBudget(**{"queries": 1, "cpu_seconds": 2, "wall_seconds": 3, **kwargs})
        client = self.client(budget=ReferenceBudget(10, 2, 3, request_bytes=1))
        with self.assertRaisesRegex(ReferenceBudgetExceeded, "request_bytes"):
            client.predict(Game.new(), "5k")
        self.assertEqual(client.cost()["requests"], 0)

        probe = self.client()
        probe.query_version()
        first_bytes = probe.cost()["response_bytes"]
        client = self.client(budget=ReferenceBudget(10, 2, 3, response_bytes=first_bytes + 1))
        client.query_version()
        with self.assertRaisesRegex(ReferenceBudgetExceeded, "response_bytes"):
            client.query_version()
        self.assertEqual(client.cost()["requests"], 2)

        client = self.client(budget=ReferenceBudget(10, 2, 3, trace_bytes=4096))
        with self.assertRaises(ReferenceBudgetExceeded):
            for _ in range(10):
                client.query_version()
        self.assertLessEqual(client.cost()["trace_bytes"], 4096)
        self.assertEqual(sum(len(canonical(record).encode()) for record in client.traces),
                         client.cost()["trace_bytes"])

    def test_sink_failure_stops_client_without_retry(self):
        def broken(record):
            raise OSError("sink unavailable")

        client = self.client(trace_sink=broken)
        with self.assertRaisesRegex(ReferenceError, "trace_sink_failed"):
            client.predict(Game.new(), "5k")
        self.assertTrue(client.cost()["closed"])
        self.assertEqual(len(client.traces), 1)

    def test_real_synthetic_process_startup_shutdown_and_cpu_retention(self):
        self.script("import json,sys,time\n"
            "start=time.process_time()\n"
            "while time.process_time()-start < 0.05: pass\n"
            "for line in sys.stdin:\n"
            " r=json.loads(line)\n"
            f" r.update(version={ENGINE_VERSION!r},git_hash={ENGINE_REVISION!r})\n"
            " print(json.dumps(r),flush=True)\n")
        client = self.client(real=True)
        client.query_version()
        before = client.cost()
        self.assertGreater(before["cpu_seconds"], 0.04)
        client.close()
        after = client.cost()
        self.assertGreaterEqual(after["cpu_seconds"], before["cpu_seconds"])
        self.assertTrue(after["closed"])
        self.assertIsNotNone(client.analysis.process.returncode)
        time.sleep(0.005)
        self.assertEqual(after, client.cost())

    def test_failed_startup_retains_native_cpu_and_failed_request(self):
        self.script("import time\n"
                    "start=time.process_time()\n"
                    "while time.process_time()-start < 0.05: pass\n")
        client = self.client(real=True)
        with self.assertRaisesRegex(ReferenceError, "process_output_closed") as error:
            client.predict(Game.new(), "5k")
        self.assertGreaterEqual(error.exception.cost["cpu_seconds"], 0.05)
        self.assertEqual(error.exception.cost["queries"], 1)
        self.assertEqual(client.traces[0]["request"]["overrideSettings"]["humanSLProfile"], "rank_5k")
        self.assertTrue(client.cost()["closed"])

    def test_unrelated_reaped_children_are_not_reference_cpu(self):
        client = self.client()
        before = client.cost()["cpu_seconds"]
        subprocess.run([sys.executable, "-c", "import time;t=time.process_time()\n"
                        "while time.process_time()-t < 0.1: pass"], check=True)
        self.assertLess(client.cost()["cpu_seconds"] - before, 0.05)

    def test_real_synthetic_process_deadlines_output_limits_and_partial_failure(self):
        cases = [
            ("import time;time.sleep(10)", {"wall_seconds": 0.1}, "wall_seconds"),
            ("while True: pass", {"cpu_seconds": 0.05}, "cpu_seconds"),
            ("import sys;sys.stdin.readline();sys.stdout.write('x'*10000);sys.stdout.flush()",
             {"response_bytes": 50}, "response_bytes"),
            ("import sys;sys.stderr.write('x'*10000);sys.stderr.flush()",
             {"stderr_bytes": 20}, "stderr_bytes"),
            ("import sys;sys.stdin.readline();sys.stdout.write('{broken');sys.stdout.flush()",
             {}, "process_output_closed"),
            ("import sys,time;sys.stderr.write('WARNING changed settings\\n');sys.stderr.flush();time.sleep(1)",
             {}, "process_warning"),
        ]
        for code, limits, message in cases:
            with self.subTest(message=message):
                self.script(code)
                client = self.client(real=True, budget=ReferenceBudget(**{
                    "queries": 2, "cpu_seconds": 2, "wall_seconds": 3, **limits}))
                with self.assertRaisesRegex(ReferenceError, message) as error:
                    client.query_version()
                if message.endswith("seconds") or message.endswith("bytes"):
                    self.assertIsInstance(error.exception, ReferenceBudgetExceeded)
                self.assertTrue(client.cost()["closed"])
                self.assertIsNotNone(client.analysis.process.returncode)
                self.assertEqual(client.cost()["requests"], 1)
                self.assertEqual(client.cost()["queries"], 0)
                self.assertTrue(client.traces[-1]["error"])
                if message == "process_output_closed":
                    self.assertEqual(client.traces[-1]["raw_response"], "{broken")
                if message == "process_warning":
                    self.assertIn("WARNING changed settings", client.traces[-1]["stderr_tail"])

    def test_match_profile_binding_pickling_rng_and_budget_attribution(self):
        factory = self.factory()
        profile = human_agent_profile("reference5k", factory.profile, "5k", source_revision="c" * 40, budget=factory.budget)
        self.assertIsNone(profile.data["human_rank"])
        self.assertEqual(profile.data["conditioning_rank"], "5k")
        self.assertEqual(profile.data["context"],
                         {"timed": False, "clock_inputs": "missing", "opponent_rank": "unknown"})
        restored = pickle.loads(pickle.dumps(factory))
        self.assertEqual(restored, factory)
        for field, value in (("source_revision", "0" * 40), ("checkpoint_sha256", "0" * 64),
                             ("pins", {}), ("human_rank", "5k"), ("sampling", "argmax"),
                             ("engine_context", {}), ("engine_revision", "0" * 40),
                             ("reference_limits", {})):
            with self.subTest(field=field):
                with self.assertRaisesRegex(InfrastructureError, "match_profile_pin_mismatch"):
                    factory({**profile.data, field: value}, 123)
        client = self.client(budget=ReferenceBudget(1, 2, 3))
        with patch("kenggo.katago_human.HumanSL", return_value=client):
            agent = factory(profile.data, 123)
        decision = agent.choose(Game.new())
        self.assertEqual(decision.action, PASS)
        self.assertIsNone(decision.network_calls)
        self.assertEqual(decision.gpu_seconds, 0.0)
        with self.assertRaisesRegex(InfrastructureError, "queries_budget_exhausted"):
            agent.choose(Game.new())

    def test_factory_accepts_runner_added_weight_without_weakening_pins(self):
        factory = self.factory()
        profile = human_agent_profile("reference5k", factory.profile, "5k", source_revision="c" * 40,
                                      budget=factory.budget).data
        client = self.client()
        with patch("kenggo.katago_human.HumanSL", return_value=client):
            agent = factory({**profile, "weight": 0.25}, 123)
        self.assertEqual(agent.choose(Game.new()).action, PASS)
        for weight in (0, -1, True, float("inf")):
            with self.subTest(weight=weight), self.assertRaises(InfrastructureError):
                factory({**profile, "weight": weight}, 123)

    def test_shutdown_accounts_pending_output_and_enforces_its_cap(self):
        self.script("import json,sys,time\n"
            "r=json.loads(sys.stdin.readline())\n"
            f"r.update(version={ENGINE_VERSION!r},git_hash={ENGINE_REVISION!r})\n"
            "print(json.dumps(r),flush=True)\n"
            "time.sleep(0.02)\n"
            "sys.stderr.write('x'*100);sys.stderr.flush()\n"
            "time.sleep(1)\n")
        client = self.client(real=True, budget=ReferenceBudget(1, 2, 3, stderr_bytes=50))
        client.query_version()
        time.sleep(0.04)
        with self.assertRaisesRegex(ReferenceBudgetExceeded, "stderr_bytes"):
            client.close()
        self.assertEqual(client.cost()["stderr_bytes"], 51)
        self.assertTrue(client.cost()["closed"])

    def run_match(self, factory, *, response_seconds=1):
        candidate = human_agent_profile("candidate", factory.profile, "5k", source_revision="c" * 40,
                                         budget=factory.budget)
        opponent = weak_profile("opponent", source_revision="c" * 40)
        slot = {"id": "synthetic", "block_id": "b0", "arm": candidate.id, "profile_id": opponent.id,
                "candidate_color": "B", "candidate_seed": 10, "opponent_seed": 20}
        return run_game(slot, candidate, opponent, budget=ExperimentBudget(10, 10),
                        limits=MatchLimits(response_seconds=response_seconds, game_seconds=5),
                        agent_factories={POLICY: factory})

    def test_runner_records_administrative_reference_exhaustion_as_unknown(self):
        self.version_only_script()
        result = self.run_match(self.factory(ReferenceBudget(0, 3, 5)))
        self.assertEqual(result.record["result"], "U")
        self.assertEqual(result.record["fault"]["source"], "infrastructure")
        self.assertIsNone(result.record["fault"]["responsible"])
        self.assertIn("queries_budget_exhausted", result.record["fault"]["reason"])

    def test_runner_keeps_missing_or_corrupt_reference_artifacts_unresolved(self):
        factory = self.factory()
        main_model = Path(factory.main_model)
        main_model.unlink()
        for state in ("missing", "corrupt"):
            with self.subTest(state=state):
                if state == "corrupt":
                    main_model.write_bytes(b"different checkpoint")
                result = self.run_match(factory)
                self.assertEqual(result.record["result"], "U")
                self.assertEqual(result.record["fault"]["source"], "infrastructure")
                self.assertIsNone(result.record["fault"]["responsible"])
                self.assertEqual(len(result.game.moves), 0)

    def test_runner_keeps_invalid_agent_replies_attributable(self):
        self.version_only_script()
        result = self.run_match(self.factory())
        self.assertEqual(result.record["result"], "L")
        self.assertEqual(result.record["fault"]["source"], "agent")
        self.assertEqual(result.record["fault"]["responsible"], "B")

    def test_runner_keeps_native_launch_permission_failure_unresolved(self):
        self.version_only_script()
        self.paths["engine"].chmod(0o600)
        result = self.run_match(self.factory())
        self.assertEqual(result.record["result"], "U")
        self.assertEqual(result.record["fault"]["source"], "infrastructure")
        self.assertIsNone(result.record["fault"]["responsible"])
        self.assertEqual(len(result.game.moves), 0)

    def test_runner_keeps_all_path_preparation_failures_unresolved(self):
        factory = self.factory()
        loop = self.root / "loop"
        loop.symlink_to(loop.name)
        for field in ("main_model", "runtime_root"):
            with self.subTest(field=field):
                result = self.run_match(replace(factory, **{field: str(loop)}))
                self.assertEqual(result.record["result"], "U")
                self.assertEqual(result.record["fault"]["source"], "infrastructure")
                self.assertIsNone(result.record["fault"]["responsible"])
                self.assertEqual(len(result.game.moves), 0)

    def test_path_preparation_failure_retains_closed_client_cost(self):
        loop = self.root / "loop"
        loop.symlink_to(loop.name)
        self.paths["main_model"] = loop
        with self.assertRaises(ReferenceError) as caught:
            self.client(real=True)
        self.assertTrue(caught.exception.cost["closed"])
        self.assertEqual(caught.exception.cost["requests"], 0)
        self.assertEqual(caught.exception.cost["queries"], 0)
        self.assertGreaterEqual(caught.exception.cost["cpu_seconds"], 0)

    def test_runner_kills_native_descendant_after_worker_timeout(self):
        pidfile = self.root / "native.pid"
        self.script("import os,time\n"
                    f"open({str(pidfile)!r},'w').write(str(os.getpid()))\n"
                    "time.sleep(10)\n")
        result = self.run_match(self.factory(), response_seconds=0.4)
        self.assertEqual(result.record["fault"]["reason"], "response_timeout")
        self.assertTrue(pidfile.is_file())
        stat = Path(f"/proc/{int(pidfile.read_text())}/stat")
        if stat.exists():
            self.assertEqual(stat.read_text().rsplit(")", 1)[1].split()[0], "Z")


if __name__ == "__main__":
    unittest.main()
