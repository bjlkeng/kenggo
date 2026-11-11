"""Bounded empirical KataGo judgments, with independently replayed cleanup."""

from dataclasses import dataclass
from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import resource
import selectors
import signal
import subprocess
import threading
import time

from kenggo.go import Game, PASS, canonical, digest


METHOD = "katago_cleanup_referee_v1"
PASS_METHOD = "katago_pass_referee_v1"
RULES = {"ko": "SIMPLE", "scoring": "TERRITORY", "tax": "SEKI", "suicide": False,
         "hasButton": False, "whiteHandicapBonus": "0", "friendlyPassOk": False}
REQUIRED_CONFIG = {
    "numAnalysisThreads": "1", "numSearchThreadsPerAnalysisThread": "1",
    "numEigenThreadsPerModel": "1", "nnMaxBatchSize": "1", "nnRandomize": "false",
    "nnForcedSymmetry": "0", "forDeterministicTesting": "true",
    "reportAnalysisWinratesAs": "WHITE", "preventCleanupPhase": "false",
    "conservativePass": "false", "winLossUtilityFactor": "0",
    "staticScoreUtilityFactor": "1", "dynamicScoreUtilityFactor": "0",
    "rootNoiseEnabled": "false", "wideRootNoise": "0",
    "playoutDoublingAdvantage": "0", "ignoreAllHistory": "false",
    "ignorePreRootHistory": "false", "assumeMultipleStartingBlackMovesAreHandicap": "false",
}


class RefereeError(ValueError):
    pass


def _integer(value, minimum, maximum):
    return type(value) is int and minimum <= value <= maximum


def _finite(value, minimum, maximum):
    return type(value) in (int, float) and math.isfinite(value) and minimum <= value <= maximum


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _file_sha(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            hasher.update(block)
    return hasher.hexdigest()


def _json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise RefereeError("duplicate_json_key")
            result[key] = value
        return result

    def constant(value):
        raise RefereeError("nonfinite_json_number")

    try:
        return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise RefereeError("invalid_json") from error


def _hash(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


def _relative(value):
    return (isinstance(value, str) and value and not PurePosixPath(value).is_absolute()
            and ".." not in PurePosixPath(value).parts and str(PurePosixPath(value)) == value)


@dataclass(frozen=True)
class Profile:
    """The manifest file is authorized by a digest supplied outside that file."""

    sha256: str
    payload: str

    @classmethod
    def load(cls, path, trusted_sha256):
        path = Path(path)
        if path.stat().st_size > 128_000 or not _hash(trusted_sha256):
            raise ValueError("Invalid referee manifest size or external pin")
        raw = path.read_bytes()
        if _sha(raw) != trusted_sha256:
            raise ValueError("Referee manifest differs from its external pin")
        data = _json(raw)
        expected = {"schema", "profile_id", "engine_version", "native_revision", "model_name",
                    "artifacts", "runtime_files", "library_dirs", "cleanup_visits", "pass_visits",
                    "cleanup_move_limit", "acceptable_gain", "premature_gain", "no_result_limit", "limits"}
        if not isinstance(data, dict):
            raise ValueError("Invalid referee profile schema")
        winner_profile = data.get("schema") == "katago_referee_profile_v2"
        if winner_profile:
            expected.add("score_selection")
        if (set(data) != expected or data["schema"] not in ("katago_referee_profile_v1", "katago_referee_profile_v2")
                or winner_profile and data["score_selection"] != "higher_visit_when_winners_agree"):
            raise ValueError("Invalid referee profile schema")
        for field in ("profile_id", "engine_version", "native_revision", "model_name"):
            if not isinstance(data[field], str) or not data[field] or len(data[field]) > 200:
                raise ValueError("Invalid referee provenance")
        pins = data["artifacts"]
        if (not isinstance(pins, dict) or set(pins) != {"engine", "model", "config", "native"}
                or not all(_hash(value) for value in pins.values())):
            raise ValueError("Invalid referee artifact pins")
        auxiliary = data["runtime_files"]
        if (not isinstance(auxiliary, dict) or len(auxiliary) > 1000
                or any(not _relative(name) or not _hash(pin) for name, pin in auxiliary.items())):
            raise ValueError("Invalid referee runtime pins")
        dirs = data["library_dirs"]
        if (not isinstance(dirs, list) or len(dirs) > 8 or any(not _relative(name) for name in dirs)
                or len(set(dirs)) != len(dirs)):
            raise ValueError("Invalid referee library directories")
        for field in ("cleanup_visits", "pass_visits"):
            values = data[field]
            if (not isinstance(values, list) or len(values) != 2
                    or not all(_integer(v, 1, 100_000) for v in values) or values[0] >= values[1]):
                raise ValueError("Two increasing, bounded visit budgets are required")
        if not _integer(data["cleanup_move_limit"], 1, 64 if winner_profile else 32):
            raise ValueError("Cleanup move limit exceeds the profile schema bound")
        if (not _finite(data["acceptable_gain"], 0, 1)
                or not _finite(data["premature_gain"], 1, 10)
                or data["acceptable_gain"] >= data["premature_gain"]
                or not _finite(data["no_result_limit"], 0, 0.1)):
            raise ValueError("Invalid empirical judgment thresholds")
        limits = data["limits"]
        ranges = {"nodes": (1, 10_000_000), "request_bytes": (1024, 4 * 1024**2),
                  "response_bytes": (1024, 32 * 1024**2), "stderr_bytes": (0, 4 * 1024**2),
                  "evidence_bytes": (1024, 8 * 1024**2)}
        if (not isinstance(limits, dict) or set(limits) != set(ranges) | {"cpu_seconds", "wall_seconds"}
                or any(not _integer(limits[key], *bounds) for key, bounds in ranges.items())
                or not _finite(limits["cpu_seconds"], 0.001, 3600)
                or not _finite(limits["wall_seconds"], 0.001, 3600)):
            raise ValueError("Invalid referee resource limits")
        return cls(trusted_sha256, canonical(data))

    @property
    def data(self):
        return json.loads(self.payload)


@dataclass(frozen=True)
class RefereeDecision:
    status: str
    reason: str
    snapshot: str
    white_minus_black_halfpoints: int | None
    evidence: str


@dataclass(frozen=True)
class PassDecision:
    verdict: str
    reason: str
    snapshot: str
    gains_points: tuple[float, ...]
    evidence: str


class _Budget:
    def __init__(self, limits, cpu):
        self.limits, self.cpu = limits, cpu
        self.started, self.cpu_started = time.monotonic(), cpu()
        self.nodes = self.requests = self.request_bytes = self.response_bytes = self.stderr_bytes = 0

    def check(self):
        if self.nodes > self.limits["nodes"]:
            raise RefereeError("node_budget_exhausted")
        if self.cpu() - self.cpu_started >= self.limits["cpu_seconds"]:
            raise RefereeError("cpu_budget_exhausted")
        if time.monotonic() - self.started >= self.limits["wall_seconds"]:
            raise RefereeError("wall_budget_exhausted")

    def reserve(self, nodes):
        self.nodes += nodes
        self.check()

    def record(self):
        return {"limits": self.limits, "nodes": self.nodes, "requests": self.requests,
                "request_bytes": self.request_bytes, "response_bytes": self.response_bytes,
                "stderr_bytes": self.stderr_bytes, "cpu_seconds": self.cpu() - self.cpu_started,
                "wall_seconds": time.monotonic() - self.started, "gpu_seconds": 0}


class _JsonProcess:
    def __init__(self, command, *, env=None):
        if not Path("/proc/self/stat").is_file():
            raise RefereeError("linux_child_cpu_accounting_required")
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, bufsize=0, start_new_session=True, env=env)
        for pipe in (self.process.stdin, self.process.stdout, self.process.stderr):
            os.set_blocking(pipe.fileno(), False)
        self.pending = bytearray()
        self.stderr_tail = b""

    def cpu_seconds(self):
        try:
            fields = Path(f"/proc/{self.process.pid}/stat").read_text().rsplit(")", 1)[1].split()
            return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
        except FileNotFoundError:
            return 0.0

    def exchange(self, request, budget):
        if self.pending:
            raise RefereeError("unsolicited_process_output")
        output, written = bytearray(), 0
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdin, selectors.EVENT_WRITE, "input")
            selector.register(self.process.stdout, selectors.EVENT_READ, "output")
            selector.register(self.process.stderr, selectors.EVENT_READ, "error")
            while True:
                budget.check()
                for key, _ in selector.select(0.01):
                    if key.data == "input":
                        try:
                            written += os.write(key.fd, request[written:])
                        except BrokenPipeError as error:
                            raise RefereeError("process_input_closed") from error
                        if written == len(request):
                            selector.unregister(key.fileobj)
                        continue
                    block = os.read(key.fd, 65536)
                    if not block:
                        selector.unregister(key.fileobj)
                        if key.data == "output":
                            raise RefereeError("process_output_closed")
                        continue
                    if key.data == "error":
                        budget.stderr_bytes += len(block)
                        if budget.stderr_bytes > budget.limits["stderr_bytes"]:
                            raise RefereeError("stderr_byte_limit")
                        self.stderr_tail = (self.stderr_tail + block)[-4096:]
                        if b"warning" in self.stderr_tail.lower():
                            raise RefereeError("process_warning")
                    else:
                        output.extend(block)
                        if len(output) > budget.limits["response_bytes"]:
                            raise RefereeError("response_byte_limit")
                        if b"\n" in output:
                            line, remainder = output.split(b"\n", 1)
                            self.pending.extend(remainder)
                            if self.pending or written != len(request):
                                raise RefereeError("unsolicited_process_output")
                            budget.check()
                            return bytes(line) + b"\n"

    def close(self):
        if self.process.returncode is None:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.process.wait()
        for pipe in (self.process.stdin, self.process.stdout, self.process.stderr):
            pipe.close()


def _coordinate(action):
    return "pass" if action == PASS else "ABCDEFGHJKLMNOPQRST"[action % 19] + str(19 - action // 19)


def _action(value):
    if value == "pass":
        return PASS
    if (not isinstance(value, str) or len(value) not in (2, 3)
            or value[0] not in "ABCDEFGHJKLMNOPQRST" or not value[1:].isdigit()):
        raise RefereeError("invalid_analysis_move")
    row = int(value[1:])
    if not 1 <= row <= 19 or str(row) != value[1:]:
        raise RefereeError("invalid_analysis_move")
    return (19 - row) * 19 + "ABCDEFGHJKLMNOPQRST".index(value[0])


def _serial(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return call


class KataGoReferee:
    def __init__(self, profile, *, engine, model, config, native, runtime_root=None, trace_sink=None,
                 _analysis=None, _native=None):
        if not isinstance(profile, Profile):
            raise ValueError("An externally pinned Profile is required")
        self.profile, self.settings = profile, profile.data
        self.paths = {name: Path(path).resolve() for name, path in
                      (("engine", engine), ("model", model), ("config", config), ("native", native))}
        self.runtime_root = Path(runtime_root).resolve() if runtime_root else self.paths["engine"].parent
        self.trace_sink, self.analysis, self.native = trace_sink, _analysis, _native
        self._sequence = 0
        self._lock = threading.RLock()
        self._verify_artifacts()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @_serial
    def close(self):
        for process in (self.analysis, self.native):
            if process is not None:
                process.close()
        self.analysis = self.native = None

    def _verify_artifacts(self):
        for name, path in self.paths.items():
            if _file_sha(path) != self.settings["artifacts"][name]:
                raise RefereeError("artifact_hash_mismatch:" + name)
        auxiliary = self.settings["runtime_files"]
        for relative, expected in auxiliary.items():
            path = (self.runtime_root / relative).resolve()
            if not path.is_relative_to(self.runtime_root) or _file_sha(path) != expected:
                raise RefereeError("runtime_hash_mismatch:" + relative)
        for relative in self.settings["library_dirs"]:
            directory = (self.runtime_root / relative).resolve()
            if not directory.is_relative_to(self.runtime_root) or not directory.is_dir():
                raise RefereeError("invalid_runtime_directory")
            for path in directory.rglob("*"):
                if path.is_file() and path.relative_to(self.runtime_root).as_posix() not in auxiliary:
                    raise RefereeError("unpinned_runtime_library")
        values = {}
        for line in self.paths["config"].read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            if "=" not in line:
                raise RefereeError("invalid_referee_config")
            key, value = (part.strip() for part in line.split("=", 1))
            if key in values:
                raise RefereeError("duplicate_config_key")
            values[key] = value
        for key, expected in REQUIRED_CONFIG.items():
            actual = values.get(key)
            matches = actual == expected
            if expected in ("0", "1") and actual is not None:
                try:
                    matches = float(actual) == int(expected)
                except ValueError:
                    pass
            if not matches:
                raise RefereeError("unsafe_referee_config:" + key)
        if not values.get("nnRandSeed"):
            raise RefereeError("missing_neural_seed")

    def _cpu(self):
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        return (time.process_time() + usage.ru_utime + usage.ru_stime
                + sum(process.cpu_seconds() for process in (self.analysis, self.native) if process is not None))

    def _process(self, kind):
        if kind == "native":
            if self.native is None:
                self.native = _JsonProcess([str(self.paths["native"])])
            return self.native
        if self.analysis is None:
            environment = dict(os.environ)
            environment.pop("LD_PRELOAD", None)
            environment.pop("LD_LIBRARY_PATH", None)
            environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            if self.settings["library_dirs"]:
                environment["LD_LIBRARY_PATH"] = ":".join(
                    str(self.runtime_root / relative) for relative in self.settings["library_dirs"])
            self.analysis = _JsonProcess([str(self.paths["engine"]), "analysis", "-config",
                                         str(self.paths["config"]), "-model", str(self.paths["model"])],
                                        env=environment)
        return self.analysis

    def _call(self, kind, request, budget, evidence, nodes):
        budget.reserve(nodes)
        raw = canonical(request).encode() + b"\n"
        if len(raw) > budget.limits["request_bytes"]:
            raise RefereeError("request_byte_limit")
        budget.requests += 1
        budget.request_bytes += len(raw)
        reply = self._process(kind).exchange(raw, budget)
        if not isinstance(reply, bytes) or len(reply) > budget.limits["response_bytes"]:
            raise RefereeError("response_byte_limit")
        budget.response_bytes += len(reply)
        evidence["calls"].append({"kind": kind, "query_sha256": _sha(raw), "response_sha256": _sha(reply),
                                  "request_bytes": len(raw), "response_bytes": len(reply)})
        if self.trace_sink is not None:
            self.trace_sink(kind, raw, reply)
        budget.check()
        result = _json(reply)
        if not isinstance(result, dict):
            raise RefereeError("response_is_not_object")
        return result

    def _native_state(self, game, moves, budget, evidence, *, original=False):
        initial = game.positions[0]
        result = self._call("native", {"mode": "history", "board": initial.rows,
            "captures": {"B": initial.captured_black, "W": initial.captured_white},
            "to_play": initial.to_play, "moves": [{"color": c, "action": a} for c, a in moves],
            "include_legal_mask": True}, budget, evidence, len(moves) + 1)
        if (result.get("ok") is not True or result.get("mode") != "history" or result.get("size") != 19
                or not isinstance(result.get("steps"), list) or len(result["steps"]) != len(moves)):
            raise RefereeError("invalid_native_response")
        for step, (color, action) in zip(result["steps"], moves):
            if (not isinstance(step, dict) or step.get("accepted") is not True
                    or step.get("color") != color or type(step.get("action")) is not int
                    or step["action"] != action):
                raise RefereeError("native_rejected_move")
        states = [result.get("initial")] + [step.get("state") for step in result["steps"]]
        for index, state in enumerate(states):
            self._validate_native_state(state, index)
        if result.get("final") != states[-1]:
            raise RefereeError("inconsistent_native_final")
        if original:
            if len(states) != len(game.positions):
                raise RefereeError("original_history_length_mismatch")
            for state, position in zip(states, game.positions):
                if (state["board"] != list(position.rows) or state["to_play"] != position.to_play
                        or state["simple_ko"] != position.ko or state["pass_streak"] != position.pass_streak
                        or state["captures"] != {"B": position.captured_black, "W": position.captured_white}):
                    raise RefereeError("original_native_state_mismatch")
            expected_phase = 1 if game.status == "pending_adjudication" else 0
            if states[-1]["encore_phase"] != expected_phase:
                raise RefereeError("original_native_phase_mismatch")
        return states[-1]

    @staticmethod
    def _validate_native_state(state, move_count):
        fields = {"board", "to_play", "captures", "simple_ko", "move_count", "pass_streak", "encore_phase",
                  "finished", "scored", "no_result", "winner", "white_minus_black_halfpoints", "legal_mask"}
        if not isinstance(state, dict) or set(state) != fields:
            raise RefereeError("invalid_native_state")
        board, captures, mask = state.get("board"), state.get("captures"), state.get("legal_mask")
        if (not isinstance(board, list) or len(board) != 19
                or any(not isinstance(row, str) or len(row) != 19 or set(row) - set(".BW") for row in board)
                or state.get("to_play") not in ("B", "W")
                or not isinstance(captures, dict) or set(captures) != {"B", "W"}
                or any(not _integer(v, 0, 1_000_000) for v in captures.values())
                or not isinstance(mask, list) or len(mask) != 362 or any(type(v) is not bool for v in mask)
                or not _integer(state.get("move_count"), move_count, move_count)
                or not _integer(state.get("pass_streak"), 0, 2000)
                or not _integer(state.get("encore_phase"), 0, 2)
                or (state.get("simple_ko") is not None and not _integer(state["simple_ko"], 0, 360))
                or any(type(state.get(key)) is not bool for key in ("finished", "scored", "no_result"))):
            raise RefereeError("invalid_native_state")
        score, winner = state.get("white_minus_black_halfpoints"), state.get("winner")
        if state["scored"]:
            if (not state["finished"] or state["no_result"] or type(score) is not int or score % 2 != 1
                    or winner != ("W" if score > 0 else "B")):
                raise RefereeError("invalid_native_score")
        elif score is not None or winner is not None:
            raise RefereeError("unexpected_native_result")
        if state["no_result"] and not state["finished"]:
            raise RefereeError("invalid_native_no_result")

    def _analysis(self, game, moves, to_play, visits, budget, evidence, *, restriction=None):
        if game.positions[0].captured_black or game.positions[0].captured_white:
            raise RefereeError("initial_capture_offset_unsupported")
        self._sequence += 1
        request_id = f"{game.snapshot}:{self._sequence}"
        initial = game.positions[0]
        request = {"id": request_id, "initialStones": [[c, _coordinate(a)]
                   for a, c in enumerate(initial.board) if c != "."],
                   "initialPlayer": initial.to_play, "moves": [[c, _coordinate(a)] for c, a in moves],
                   "rules": RULES, "komi": 6.5, "boardXSize": 19, "boardYSize": 19,
                   "maxVisits": visits, "includeNoResultValue": True}
        if restriction is not None:
            field, allowed = restriction
            request[field] = [{"player": to_play, "moves": allowed, "untilDepth": 1}]
        result = self._call("analysis", request, budget, evidence, visits)
        if "error" in result or "warning" in result or "warnings" in result:
            raise RefereeError("analysis_error_or_warning")
        root, infos = result.get("rootInfo"), result.get("moveInfos")
        if (result.get("id") != request_id or type(result.get("turnNumber")) is not int
                or result["turnNumber"] != len(moves) or result.get("isDuringSearch") is not False
                or not isinstance(root, dict) or root.get("currentPlayer") != to_play
                or not _integer(root.get("visits"), visits, visits)
                or not _finite(root.get("scoreLead"), -100_000, 100_000)
                or not isinstance(infos, list) or not 1 <= len(infos) <= 362):
            raise RefereeError("invalid_or_stale_analysis")
        candidates, seen, orders = [], set(), set()
        for info in infos:
            if not isinstance(info, dict):
                raise RefereeError("invalid_analysis_candidate")
            action = _action(info.get("move"))
            if (action in seen or not _integer(info.get("order"), 0, 361) or info["order"] in orders
                    or not _integer(info.get("visits"), 0, visits)
                    or not _finite(info.get("scoreLead"), -100_000, 100_000)
                    or not _finite(info.get("noResultValue"), 0, 1)):
                raise RefereeError("invalid_analysis_candidate")
            seen.add(action)
            orders.add(info["order"])
            candidates.append((info["order"], action, info))
        order, action, best = min(candidates)
        if order != 0 or best["visits"] < 1:
            raise RefereeError("analysis_has_no_searched_best_move")
        if restriction is not None:
            if best["noResultValue"] > self.settings["no_result_limit"]:
                raise RefereeError("analysis_no_result_uncertainty")
            field, allowed = restriction
            if any(((_coordinate(a) not in allowed) if field == "allowMoves" else (_coordinate(a) in allowed))
                   for _, a, _ in candidates):
                raise RefereeError("analysis_violated_root_restriction")
        return action, float(best["scoreLead"])

    def _begin(self, game, method):
        if not isinstance(game, Game):
            raise ValueError("A shared Game is required")
        budget = _Budget(self.settings["limits"], self._cpu)
        evidence = {"method": method, "claim": "empirical_not_certified", "snapshot": game.snapshot,
                    "original": game.action_record(), "profile_sha256": self.profile.sha256,
                    "profile": self.settings, "calls": [], "runs": []}
        return budget, evidence

    def _finish(self, evidence, budget, status, reason, score=None):
        accepted = status in ("resolved", "acceptable", "premature")
        if accepted:
            budget.check()
        evidence.update(status=status, reason=reason, white_minus_black_halfpoints=score, cost=budget.record())
        payload = canonical(evidence)
        if len(payload.encode()) > self.settings["limits"]["evidence_bytes"]:
            raise RefereeError("evidence_byte_limit")
        if accepted:
            budget.check()
        return payload

    def _failure_evidence(self, evidence, budget, status, reason):
        cost = budget.record()
        cost.pop("limits")
        return canonical({"method": evidence["method"], "claim": "empirical_not_certified",
            "snapshot": evidence["snapshot"], "profile_sha256": self.profile.sha256,
            "status": status, "reason": reason, "white_minus_black_halfpoints": None,
            "omitted_evidence_sha256": digest(evidence), "cost": cost})

    def _select_cleanup_score(self, runs):
        first, second = (run["terminal"]["white_minus_black_halfpoints"] for run in runs)
        if self.settings["schema"] == "katago_referee_profile_v2":
            if (first > 0) == (second > 0):
                return second, "winner_stable_empirical_cleanup"
            return None, "cleanup_winner_disagreement"
        return (first, "agreed_empirical_cleanup") if first == second else (None, "cleanup_budget_disagreement")

    @_serial
    def adjudicate(self, game):
        if game.status != "pending_adjudication":
            raise ValueError("Referee adjudication requires a frozen two-pass ending")
        budget, evidence = self._begin(game, METHOD)
        reason, score = "cleanup_budget_disagreement", None
        try:
            self._verify_artifacts()
            if all(color == "." for color in game.position.board):
                raise RefereeError("empty_board_ending_unsupported")
            actual = [[move.color, move.action] for move in game.moves]
            original = self._native_state(game, actual, budget, evidence, original=True)
            if original["finished"]:
                raise RefereeError("original_native_finished")
            for visits in self.settings["cleanup_visits"]:
                moves, state = list(actual), original
                run = {"visits": visits, "cleanup": [], "terminal": None}
                evidence["runs"].append(run)
                while not state["finished"]:
                    if len(run["cleanup"]) >= self.settings["cleanup_move_limit"]:
                        raise RefereeError("cleanup_move_budget_exhausted")
                    color = state["to_play"]
                    action, _ = self._analysis(game, moves, color, visits, budget, evidence)
                    if not state["legal_mask"][action]:
                        raise RefereeError("illegal_referee_move")
                    moves.append([color, action])
                    run["cleanup"].append([color, action])
                    state = self._native_state(game, moves, budget, evidence)
                run["terminal"] = {key: value for key, value in state.items() if key != "legal_mask"}
                if state["no_result"]:
                    raise RefereeError("native_no_result")
                if not state["scored"]:
                    raise RefereeError("native_finished_without_score")
            score, reason = self._select_cleanup_score(evidence["runs"])
            budget.check()
        except (RefereeError, OSError, UnicodeError) as error:
            reason, score = str(error), None
            self.close()
        status = "resolved" if score is not None else "unresolved"
        try:
            payload = self._finish(evidence, budget, status, reason, score)
        except RefereeError as error:
            self.close()
            status, reason, score = "unresolved", str(error), None
            payload = self._failure_evidence(evidence, budget, status, reason)
        return RefereeDecision(status, reason, game.snapshot, score, payload)

    @_serial
    def judge_pass(self, game):
        if game.status != "playing":
            raise ValueError("Pass judgment requires the exact pre-pass playing state")
        budget, evidence = self._begin(game, PASS_METHOD)
        verdict, reason, gains = "unjudgeable", "empirical_threshold_disagreement", []
        try:
            self._verify_artifacts()
            if game.pending_cap:
                raise RefereeError("pending_administrative_cap")
            moves = [[move.color, move.action] for move in game.moves]
            state = self._native_state(game, moves, budget, evidence, original=True)
            if state["finished"]:
                raise RefereeError("original_native_finished")
            if not any(state["legal_mask"][:PASS]):
                raise RefereeError("no_legal_nonpass_comparison")
            for visits in self.settings["pass_visits"]:
                passed, pass_score = self._analysis(game, moves, state["to_play"], visits, budget, evidence,
                                                    restriction=("allowMoves", ["pass"]))
                alternative, alternative_score = self._analysis(
                    game, moves, state["to_play"], visits, budget, evidence,
                    restriction=("avoidMoves", ["pass"]))
                if passed != PASS or alternative == PASS or not state["legal_mask"][alternative]:
                    raise RefereeError("illegal_pass_comparison_move")
                gain = (alternative_score - pass_score) * (1 if state["to_play"] == "W" else -1)
                gains.append(gain)
                evidence["runs"].append({"visits": visits, "alternative": alternative,
                    "pass_white_score_lead": pass_score, "alternative_white_score_lead": alternative_score,
                    "mover_gain_points": gain})
            if all(gain <= self.settings["acceptable_gain"] for gain in gains):
                verdict, reason = "acceptable", "empirical_loss_below_threshold"
            elif all(gain >= self.settings["premature_gain"] for gain in gains):
                verdict, reason = "premature", "empirical_loss_above_threshold"
            budget.check()
        except (RefereeError, OSError, UnicodeError) as error:
            verdict, reason = "unjudgeable", str(error)
            self.close()
        try:
            payload = self._finish(evidence, budget, verdict, reason)
        except RefereeError as error:
            self.close()
            verdict, reason = "unjudgeable", str(error)
            payload = self._failure_evidence(evidence, budget, verdict, reason)
        return PassDecision(verdict, reason, game.snapshot, tuple(gains), payload)

    @_serial
    def verify(self, game, evidence_json, *, authorized_hashes=frozenset()):
        if game.status != "pending_adjudication":
            raise ValueError("Referee verification requires the original frozen ending")
        if all(color == "." for color in game.position.board):
            raise ValueError("Empty-board endings are outside this referee profile")
        if not isinstance(evidence_json, str) or len(evidence_json.encode()) > self.settings["limits"]["evidence_bytes"]:
            raise ValueError("Invalid referee evidence size")
        evidence = _json(evidence_json)
        if not isinstance(evidence, dict) or digest(evidence) not in authorized_hashes:
            raise ValueError("Referee evidence lacks independent authorization")
        expected = {"method", "claim", "snapshot", "original", "profile_sha256", "profile", "calls", "runs",
                    "status", "reason", "white_minus_black_halfpoints", "cost"}
        score = evidence.get("white_minus_black_halfpoints")
        if (set(evidence) != expected or evidence["method"] != METHOD
                or evidence["claim"] != "empirical_not_certified" or evidence["snapshot"] != game.snapshot
                or canonical(evidence["original"]) != canonical(game.action_record())
                or evidence["profile_sha256"] != self.profile.sha256
                or canonical(evidence["profile"]) != self.profile.payload
                or evidence["status"] != "resolved"
                or evidence["reason"] not in ("agreed_empirical_cleanup", "winner_stable_empirical_cleanup")
                or type(score) is not int or score % 2 != 1
                or not isinstance(evidence["runs"], list) or len(evidence["runs"]) != 2):
            raise ValueError("Referee evidence does not match the authorized profile and game")
        cost = evidence["cost"]
        cost_fields = {"limits", "nodes", "requests", "request_bytes", "response_bytes", "stderr_bytes",
                       "cpu_seconds", "wall_seconds", "gpu_seconds"}
        calls = evidence["calls"]
        maximum_requests = 1 + 4 * self.settings["cleanup_move_limit"]
        if (not isinstance(cost, dict) or set(cost) != cost_fields
                or canonical(cost.get("limits")) != canonical(self.settings["limits"])
                or not _integer(cost.get("nodes"), 0, self.settings["limits"]["nodes"])
                or not _integer(cost.get("requests"), 1, maximum_requests)
                or not _integer(cost.get("request_bytes"), 1, maximum_requests * self.settings["limits"]["request_bytes"])
                or not _integer(cost.get("response_bytes"), 1, maximum_requests * self.settings["limits"]["response_bytes"])
                or not _integer(cost.get("stderr_bytes"), 0, self.settings["limits"]["stderr_bytes"])
                or not _finite(cost.get("cpu_seconds"), 0, self.settings["limits"]["cpu_seconds"])
                or not _finite(cost.get("wall_seconds"), 0, self.settings["limits"]["wall_seconds"])
                or type(cost.get("gpu_seconds")) is not int or cost["gpu_seconds"] != 0
                or not isinstance(calls, list) or len(calls) != cost["requests"]):
            raise ValueError("Referee evidence exceeded its authorized budget")
        for call in calls:
            if (not isinstance(call, dict) or set(call) != {
                    "kind", "query_sha256", "response_sha256", "request_bytes", "response_bytes"}
                    or call["kind"] not in ("analysis", "native")
                    or not _hash(call["query_sha256"]) or not _hash(call["response_sha256"])
                    or not _integer(call["request_bytes"], 1, self.settings["limits"]["request_bytes"])
                    or not _integer(call["response_bytes"], 1, self.settings["limits"]["response_bytes"])):
                raise ValueError("Invalid referee request evidence")
        if any(sum(call[field] for call in calls) != cost[field] for field in ("request_bytes", "response_bytes")):
            raise ValueError("Referee byte counts do not match its request evidence")
        self._verify_artifacts()
        budget, verification = self._begin(game, METHOD)
        actual = [[move.color, move.action] for move in game.moves]
        try:
            self._native_state(game, actual, budget, verification, original=True)
            expected_nodes, expected_kinds = len(actual) + 1, ["native"]
            for visits, run in zip(self.settings["cleanup_visits"], evidence["runs"]):
                if (not isinstance(run, dict) or set(run) != {"visits", "cleanup", "terminal"}
                        or type(run["visits"]) is not int or run["visits"] != visits
                        or not isinstance(run["cleanup"], list)
                        or not 1 <= len(run["cleanup"]) <= self.settings["cleanup_move_limit"]):
                    raise RefereeError("invalid_cleanup_transcript")
                for move in run["cleanup"]:
                    if (not isinstance(move, list) or len(move) != 2 or move[0] not in ("B", "W")
                            or not _integer(move[1], 0, PASS)):
                        raise RefereeError("invalid_cleanup_transcript")
                for index in range(1, len(run["cleanup"]) + 1):
                    expected_nodes += visits + len(actual) + index + 1
                    expected_kinds.extend(("analysis", "native"))
                state = self._native_state(game, actual + run["cleanup"], budget, verification)
                terminal = {key: value for key, value in state.items() if key != "legal_mask"}
                if (not state["finished"] or not state["scored"] or state["no_result"] or terminal != run["terminal"]):
                    raise RefereeError("cleanup_revalidation_failed")
            if self._select_cleanup_score(evidence["runs"]) != (score, evidence["reason"]):
                raise RefereeError("cleanup_score_selection_mismatch")
            if cost["nodes"] != expected_nodes or [call["kind"] for call in calls] != expected_kinds:
                raise RefereeError("referee_work_counts_mismatch")
            budget.check()
        except (RefereeError, OSError, UnicodeError):
            self.close()
            raise
        return RefereeDecision("resolved", evidence["reason"], game.snapshot, score, canonical(evidence))
