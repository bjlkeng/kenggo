"""Pinned, bounded KataGo Human SL inference without search-policy substitution."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import selectors
import signal
import threading
import time

from kenggo.go import Game, PASS, RULES as GAME_RULES, canonical
from kenggo.katago_referee import (RULES, RefereeError, _JsonProcess, _coordinate,
                                   _file_sha, _finite, _hash, _integer, _json,
                                   _relative, _serial)


ENGINE_VERSION = "1.18.1"
ENGINE_REVISION = "92ee95c0a4b25fec214da00951ab69e97e207729"
RANKS = tuple(f"{rank}k" for rank in range(15, 0, -1)) + ("1d",)
POLICY = "katago_human_raw_v1"
# Native decimal output may round each of 362 probabilities; keep stored values intact.
POLICY_MASS_TOLERANCE = 0.0005
SETTINGS = {"max_visits": 1, "history": "full", "report_view": "WHITE",
            "temperature": 1, "pass_correction": False, "device": "cpu"}
ENGINE_CONTEXT = {"source": "KGS", "profile_date": "2020-03-01", "main_time_seconds": 1200,
                  "byoyomi_periods": 5, "byoyomi_period_seconds": 30,
                  "opponent_rank": "same_as_profile"}
REQUIRED_CONFIG = {
    "numAnalysisThreads": "1", "numSearchThreadsPerAnalysisThread": "1",
    "numEigenThreadsPerModel": "1", "nnMaxBatchSize": "1",
    "nnCacheSizePowerOfTwo": "16", "nnMutexPoolSizePowerOfTwo": "12",
    "nnRandomize": "false", "nnForcedSymmetry": "0", "nnRandSeed": "kenggo-human-reference-v1",
    "forDeterministicTesting": "true", "ignoreAllHistory": "false",
    "ignorePreRootHistory": "false", "assumeMultipleStartingBlackMovesAreHandicap": "false",
    "reportAnalysisWinratesAs": "WHITE", "preventCleanupPhase": "false",
    "conservativePass": "false", "winLossUtilityFactor": "1", "staticScoreUtilityFactor": "0",
    "dynamicScoreUtilityFactor": "0", "rootNoiseEnabled": "false", "wideRootNoise": "0",
    "playoutDoublingAdvantage": "0", "rootNumSymmetriesToSample": "1", "maxVisits": "1",
    "logToStderr": "true", "humanSLProfile": "rank_5k",
}


class ReferenceError(ValueError):
    """Invalid reference input or output; a failed client is never retried."""

    cost = None


class ReferenceBudgetExceeded(ReferenceError):
    """An administrative reference cap, not evidence of a playing loss."""


def _validate_profile(data):
    fields = {"schema", "profile_id", "engine_version", "engine_revision", "main_model_name",
              "human_model_name", "artifacts", "runtime_files", "library_dirs", "ranks",
              "settings", "engine_context"}
    if (not isinstance(data, dict) or set(data) != fields
            or data["schema"] != "katago_human_profile_v1"
            or not isinstance(data["profile_id"], str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", data["profile_id"])
            or data["engine_version"] != ENGINE_VERSION or data["engine_revision"] != ENGINE_REVISION
            or data["main_model_name"] != "kata1-b10c128-s1141046784-d204142634"
            or data["human_model_name"] != "b18c384nbt-humanv0"
            or data["ranks"] != list(RANKS) or canonical(data["settings"]) != canonical(SETTINGS)
            or canonical(data["engine_context"]) != canonical(ENGINE_CONTEXT)):
        raise ReferenceError("invalid_human_profile_schema")
    pins = data["artifacts"]
    if (not isinstance(pins, dict) or set(pins) != {"engine", "main_model", "human_model", "config"}
            or not all(_hash(pin) for pin in pins.values())):
        raise ReferenceError("invalid_artifact_pins")
    runtime, directories = data["runtime_files"], data["library_dirs"]
    if (not isinstance(runtime, dict) or len(runtime) > 1000
            or any(not _relative(name) or name == "." or not _hash(pin) for name, pin in runtime.items())
            or not isinstance(directories, list) or len(directories) > 8
            or any(not _relative(name) or name == "." for name in directories)
            or len(set(directories)) != len(directories)):
        raise ReferenceError("invalid_runtime_pins")


@dataclass(frozen=True)
class HumanProfile:
    sha256: str
    payload: str

    @classmethod
    def load(cls, path, trusted_sha256):
        path = Path(path)
        if path.stat().st_size > 128_000 or not _hash(trusted_sha256):
            raise ReferenceError("invalid_manifest_size_or_pin")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != trusted_sha256:
            raise ReferenceError("profile_hash_mismatch")
        try:
            data = _json(raw)
        except RefereeError as error:
            raise ReferenceError(str(error)) from error
        _validate_profile(data)
        return cls(trusted_sha256, canonical(data))

    @property
    def data(self):
        return json.loads(self.payload)


@dataclass(frozen=True)
class ReferenceBudget:
    queries: int
    cpu_seconds: float
    wall_seconds: float
    request_bytes: int = 4 * 1024**2
    response_bytes: int = 16 * 1024**2
    stderr_bytes: int = 1024**2
    trace_bytes: int = 16 * 1024**2

    def __post_init__(self):
        if (not _integer(self.queries, 0, 10_000_000)
                or not _finite(self.cpu_seconds, 0.001, 86400)
                or not _finite(self.wall_seconds, 0.001, 86400)
                or any(not _integer(getattr(self, field), minimum, 1024**3)
                       for field, minimum in (("request_bytes", 1), ("response_bytes", 1),
                                              ("stderr_bytes", 0), ("trace_bytes", 4096)))):
            raise ValueError("Invalid finite reference budget")


class _Meter:
    def __init__(self, limits, cpu):
        self.limits, self.cpu = asdict(limits), cpu
        self.started, self.cpu_started = time.monotonic(), cpu()
        self.queries = self.requests = self.request_bytes = self.response_bytes = 0
        self.stderr_bytes = self.trace_bytes = 0
        self.finished = self.final_cpu = None

    def check(self):
        for field in ("queries", "request_bytes", "response_bytes", "stderr_bytes", "trace_bytes"):
            if getattr(self, field) > self.limits[field]:
                raise ReferenceBudgetExceeded(field + "_budget_exhausted")
        cost = self.record()
        for field in ("cpu_seconds", "wall_seconds"):
            if cost[field] >= self.limits[field]:
                raise ReferenceBudgetExceeded(field + "_budget_exhausted")

    def record(self):
        return {"limits": dict(self.limits), "queries": self.queries, "requests": self.requests,
                **{field: getattr(self, field) for field in
                   ("request_bytes", "response_bytes", "stderr_bytes", "trace_bytes")},
                "cpu_seconds": max(0.0, (self.cpu() if self.final_cpu is None else self.final_cpu)
                                   - self.cpu_started),
                "wall_seconds": (time.monotonic() if self.finished is None else self.finished) - self.started,
                "gpu_seconds": 0.0, "network_calls": None, "closed": self.finished is not None}


class _HumanProcess(_JsonProcess):
    """The existing pipe transport, with cumulative bytes and per-child reaping costs."""

    def __init__(self, command, *, env=None):
        super().__init__(command, env=env)
        self.retained_cpu = 0.0
        self.last_response = bytearray()
        self.budget = None

    def cpu_seconds(self):
        self.retained_cpu = max(self.retained_cpu, super().cpu_seconds())
        return self.retained_cpu

    def exchange(self, request, budget):
        self.budget = budget
        self.last_response = bytearray()
        output, written = self.last_response, 0
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
                            raise ReferenceError("process_input_closed") from error
                        if written == len(request):
                            selector.unregister(key.fileobj)
                        continue
                    field = "stderr_bytes" if key.data == "error" else "response_bytes"
                    remaining = budget.limits[field] - getattr(budget, field)
                    block = os.read(key.fd, max(1, min(65536, remaining + 1)))
                    if not block:
                        selector.unregister(key.fileobj)
                        if key.data == "output":
                            raise ReferenceError("process_output_closed")
                        continue
                    setattr(budget, field, getattr(budget, field) + len(block))
                    if key.data == "error":
                        self.stderr_tail = (self.stderr_tail + block)[-4096:]
                        budget.check()
                        if b"warning" in self.stderr_tail.lower():
                            raise ReferenceError("process_warning")
                    else:
                        output.extend(block)
                        budget.check()
                        if b"\n" in output:
                            line, remainder = output.split(b"\n", 1)
                            if remainder or written != len(request):
                                raise ReferenceError("unsolicited_process_output")
                            return bytes(line) + b"\n"

    def close(self):
        if self.process.returncode is None:
            self.cpu_seconds()
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            _, status, usage = os.wait4(self.process.pid, 0)
            self.process.returncode = os.waitstatus_to_exitcode(status)
            self.retained_cpu = max(self.retained_cpu, usage.ru_utime + usage.ru_stime)
        error = None
        if self.budget is not None:
            for pipe, field in ((self.process.stdout, "response_bytes"), (self.process.stderr, "stderr_bytes")):
                if pipe.closed:
                    continue
                while getattr(self.budget, field) <= self.budget.limits[field]:
                    remaining = self.budget.limits[field] - getattr(self.budget, field)
                    try:
                        block = os.read(pipe.fileno(), max(1, min(65536, remaining + 1)))
                    except BlockingIOError:
                        break
                    if not block:
                        break
                    setattr(self.budget, field, getattr(self.budget, field) + len(block))
                    if field == "response_bytes":
                        self.last_response.extend(block)
                        error = ReferenceError("unsolicited_process_output")
                    else:
                        self.stderr_tail = (self.stderr_tail + block)[-4096:]
                        if b"warning" in self.stderr_tail.lower():
                            error = ReferenceError("process_warning")
        for pipe in (self.process.stdin, self.process.stdout, self.process.stderr):
            pipe.close()
        if self.budget is not None:
            self.budget.check()
        if error is not None:
            raise error


@dataclass(frozen=True)
class HumanPrediction:
    raw_policy: tuple[float, ...]
    policy: tuple[float, ...]
    value: float
    snapshot: str
    rank: str

    def sample(self, rng):
        if not isinstance(rng, random.Random):
            raise TypeError("Sampling requires a private random.Random")
        target, cumulative = rng.random() * math.fsum(self.policy), 0.0
        last = None
        for action, probability in enumerate(self.policy):
            if probability:
                last = action
            cumulative += probability
            if target < cumulative:
                return action
        return last


class HumanSL:
    def __init__(self, profile, *, engine, main_model, human_model, config, budget,
                 runtime_root=None, trace_sink=None, _analysis=None):
        if not isinstance(profile, HumanProfile) or not isinstance(budget, ReferenceBudget):
            raise TypeError("An externally pinned HumanProfile and finite ReferenceBudget are required")
        self._lock = threading.RLock()
        self.analysis, self._failure, self._sequence = _analysis, None, 0
        self._traces, self.trace_sink = [], trace_sink
        self._meter = _Meter(budget, self._cpu)
        try:
            self.profile, self.settings = profile, profile.data
            self.paths = {name: Path(path).resolve() for name, path in
                          (("engine", engine), ("main_model", main_model),
                           ("human_model", human_model), ("config", config))}
            self.runtime_root = Path(runtime_root).resolve() if runtime_root else self.paths["engine"].parent
            _validate_profile(self.settings)
            self._verify_artifacts()
            self._meter.check()
        except Exception as error:
            raise self._fail(error) from error

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _cpu(self):
        return time.process_time() + (self.analysis.cpu_seconds() if self.analysis is not None else 0.0)

    @_serial
    def cost(self):
        return self._meter.record()

    @property
    def traces(self):
        return tuple(json.loads(record) for record in self._traces)

    @_serial
    def close(self):
        if self._meter.finished is not None:
            return
        error = None
        try:
            if self.analysis is not None:
                self.analysis.close()
        except (ValueError, OSError) as caught:
            error = caught if isinstance(caught, ReferenceError) else ReferenceError(str(caught))
        self._meter.final_cpu, self._meter.finished = self._cpu(), time.monotonic()
        if self._failure is None:
            try:
                self._meter.check()
            except ReferenceBudgetExceeded as caught:
                error = caught
            if error is not None:
                self._failure = str(error)
                error.cost = self.cost()
                raise error

    def _fail(self, error):
        if not isinstance(error, ReferenceError):
            error = ReferenceError(str(error))
        self._failure = str(error)
        self.close()
        error.cost = self.cost()
        return error

    def _verify_artifacts(self):
        for name, path in self.paths.items():
            if _file_sha(path) != self.settings["artifacts"][name]:
                raise ReferenceError("artifact_hash_mismatch:" + name)
        pins = self.settings["runtime_files"]
        for relative, expected in pins.items():
            path = (self.runtime_root / relative).resolve()
            if not path.is_relative_to(self.runtime_root) or _file_sha(path) != expected:
                raise ReferenceError("runtime_hash_mismatch:" + relative)
        for relative in self.settings["library_dirs"]:
            directory = (self.runtime_root / relative).resolve()
            if not directory.is_relative_to(self.runtime_root) or not directory.is_dir():
                raise ReferenceError("invalid_runtime_directory")
            for path in directory.rglob("*"):
                if path.is_file() and path.relative_to(self.runtime_root).as_posix() not in pins:
                    raise ReferenceError("unpinned_runtime_library")
        values = {}
        for line in self.paths["config"].read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            if "=" not in line:
                raise ReferenceError("invalid_human_config")
            key, value = (part.strip() for part in line.split("=", 1))
            if key in values:
                raise ReferenceError("duplicate_config_key")
            values[key] = value
        if values != REQUIRED_CONFIG:
            raise ReferenceError("unsafe_human_config")

    def _process(self):
        if self.analysis is None:
            environment = dict(os.environ)
            environment.pop("LD_PRELOAD", None)
            environment.pop("LD_LIBRARY_PATH", None)
            environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            if self.settings["library_dirs"]:
                environment["LD_LIBRARY_PATH"] = ":".join(
                    str(self.runtime_root / relative) for relative in self.settings["library_dirs"])
            self.analysis = _HumanProcess([str(self.paths["engine"]), "analysis", "-config",
                str(self.paths["config"]), "-model", str(self.paths["main_model"]),
                "-human-model", str(self.paths["human_model"])], env=environment)
        return self.analysis

    def _trace(self, request, raw, result, error):
        stderr = getattr(self.analysis, "stderr_tail", b"").decode("utf-8", errors="replace")
        record = {"request": request, "response": result,
                  "request_sha256": hashlib.sha256((canonical(request) + "\n").encode()).hexdigest(),
                  "raw_response": raw.decode("utf-8", errors="replace") if result is None else None,
                  "response_sha256": hashlib.sha256(raw).hexdigest(),
                  "response_bytes": len(raw), "error": str(error) if error else None,
                  "stderr_tail": stderr,
                  "cost": self.cost()}
        try:
            encoded = canonical(record)
        except (ValueError, RecursionError):
            record.update(response=None, raw_response=raw.decode("utf-8", errors="replace"))
            encoded = canonical(record)
        remaining = self._meter.limits["trace_bytes"] - self._meter.trace_bytes
        exceeded = len(encoded.encode()) > remaining
        if exceeded:
            record.update(response=None, raw_response=None, stderr_tail=stderr[-256:], truncated=True,
                          error=str(error) if error else "trace_bytes_budget_exhausted")
            encoded = canonical(record)
        if len(encoded.encode()) <= remaining:
            self._meter.trace_bytes += len(encoded.encode())
            self._traces.append(encoded)
            if self.trace_sink is not None:
                try:
                    self.trace_sink(json.loads(encoded))
                except Exception as sink_error:
                    raise ReferenceError("trace_sink_failed") from sink_error
        if exceeded:
            raise ReferenceBudgetExceeded("trace_bytes_budget_exhausted")

    def _call(self, request, *, analysis, validate):
        if self._meter.finished is not None:
            raise ReferenceError("reference_closed:" + (self._failure or "closed"))
        raw, result, error, value = b"", None, None, None
        try:
            self._meter.check()
            encoded = (canonical(request) + "\n").encode()
            if len(encoded) + 2048 > self._meter.limits["trace_bytes"] - self._meter.trace_bytes:
                raise ReferenceBudgetExceeded("trace_bytes_budget_exhausted")
            if analysis and self._meter.queries >= self._meter.limits["queries"]:
                raise ReferenceBudgetExceeded("queries_budget_exhausted")
            if len(encoded) + self._meter.request_bytes > self._meter.limits["request_bytes"]:
                raise ReferenceBudgetExceeded("request_bytes_budget_exhausted")
            self._meter.queries += int(analysis)
            self._meter.requests += 1
            self._meter.request_bytes += len(encoded)
            raw = self._process().exchange(encoded, self._meter)
            if not isinstance(raw, bytes):
                raise ReferenceError("response_is_not_bytes")
            result = _json(raw)
            if not isinstance(result, dict):
                raise ReferenceError("response_is_not_object")
            canonical(result)
            if any(field in result for field in ("error", "warning", "warnings")):
                raise ReferenceError("analysis_error_or_warning")
            value = validate(result)
            self._meter.check()
        except Exception as caught:
            if not raw and self.analysis is not None:
                raw = bytes(getattr(self.analysis, "last_response", b""))
            error = self._fail(caught)
        try:
            self._trace(request, raw, result, error)
        except ReferenceError as caught:
            if error is None:
                error = self._fail(caught)
        if error is not None:
            self._meter.final_cpu, self._meter.finished = self._cpu(), time.monotonic()
            error.cost = self.cost()
            raise error
        try:
            self._meter.check()
        except ReferenceBudgetExceeded as caught:
            raise self._fail(caught) from caught
        return value

    @_serial
    def query_version(self):
        self._sequence += 1
        request = {"id": f"version:{self._sequence}", "action": "query_version"}

        def validate(result):
            if (result.get("id") != request["id"] or result.get("action") != "query_version"
                    or result.get("version") != ENGINE_VERSION
                    or result.get("git_hash") != ENGINE_REVISION):
                raise ReferenceError("engine_version_mismatch")
            return result

        return self._call(request, analysis=False, validate=validate)

    @staticmethod
    def _validate_game(game, allow_fixture):
        if not isinstance(game, Game) or game.status != "playing":
            raise ReferenceError("reference_requires_playing_game")
        initial = game.positions[0]
        if initial.captured_black or initial.captured_white:
            raise ReferenceError("initial_capture_offset_unsupported")
        if game.origin == "fixture" and allow_fixture is True:
            replay = Game.from_fixture(initial.rows, to_play=initial.to_play,
                                       candidate_color=game.candidate_color)
        elif game.origin == "empty":
            replay = Game.new(candidate_color=game.candidate_color)
        else:
            raise ReferenceError("fixture_requires_explicit_authorization")
        for move in game.moves:
            replay = replay.play(move.color, move.action)
        if replay != game:
            raise ReferenceError("game_history_mismatch")

    @_serial
    def predict(self, game, rank, *, allow_fixture=False):
        if rank not in RANKS:
            raise ReferenceError("unsupported_conditioning_rank")
        self._validate_game(game, allow_fixture)
        initial, mask = game.positions[0], game.legal_mask()
        self._sequence += 1
        request = {"id": f"{game.snapshot}:{self._sequence}",
                   "initialStones": [[color, _coordinate(action)]
                                     for action, color in enumerate(initial.board) if color != "."],
                   "initialPlayer": initial.to_play,
                   "moves": [[move.color, _coordinate(move.action)] for move in game.moves],
                   "rules": RULES, "komi": 6.5, "boardXSize": 19, "boardYSize": 19,
                   "includePolicy": True, "maxVisits": 1,
                   "overrideSettings": {"humanSLProfile": "rank_" + rank, "ignorePreRootHistory": False}}

        def validate(result):
            root, raw_policy = result.get("rootInfo"), result.get("humanPolicy")
            if (result.get("id") != request["id"] or type(result.get("turnNumber")) is not int
                    or result["turnNumber"] != len(game.moves) or result.get("isDuringSearch") is not False
                    or not isinstance(root, dict) or root.get("currentPlayer") != game.position.to_play
                    or not _integer(root.get("visits"), 1, 1)
                    or not _finite(root.get("humanWinrate"), 0, 1)
                    or not isinstance(raw_policy, list) or len(raw_policy) != PASS + 1):
                raise ReferenceError("invalid_or_stale_human_analysis")
            policy = []
            for legal, probability in zip(mask, raw_policy):
                if not _finite(probability, 0 if legal else -1, 1 if legal else -1):
                    raise ReferenceError("human_policy_legality_mismatch")
                policy.append(probability if legal else 0.0)
            if abs(math.fsum(policy) - 1.0) > POLICY_MASS_TOLERANCE:
                raise ReferenceError("human_policy_mass_mismatch")
            white_value = 2 * root["humanWinrate"] - 1
            return HumanPrediction(tuple(raw_policy), tuple(policy),
                                   white_value if game.position.to_play == "W" else -white_value,
                                   game.snapshot, rank)

        return self._call(request, analysis=True, validate=validate)


def human_agent_profile(id, reference_profile, rank, *, source_revision, budget, panel="validation"):
    from kenggo.matches import AgentProfile

    if (not isinstance(reference_profile, HumanProfile) or rank not in RANKS
            or not isinstance(budget, ReferenceBudget)):
        raise ValueError("A pinned profile, conditioning rank and reference budget are required")
    return AgentProfile.from_dict({
        "id": id, "family": "katago_human", "policy": POLICY, "panel": panel,
        "context_id": "untimed_missing_clock_unknown_rank_v1",
        "context": {"timed": False, "clock_inputs": "missing", "opponent_rank": "unknown"},
        "device": "cpu", "human_rank": None, "conditioning_rank": rank,
        "source_revision": source_revision,
        "checkpoint_sha256": reference_profile.data["artifacts"]["human_model"],
        "pins": {**reference_profile.data["artifacts"], "reference_profile": reference_profile.sha256,
                 "implementation": _file_sha(__file__)},
        "engine_revision": ENGINE_REVISION, "engine_context": reference_profile.data["engine_context"],
        "reference_limits": asdict(budget),
        "rules": GAME_RULES, "komi_halfpoints": 13, "sampling": "humanPolicy_temperature_1",
        "pass": "raw_humanPolicy", "resignation": "never", "history": "full",
        "ending": "exact_first_optional_pinned_empirical_v2"})


@dataclass(frozen=True)
class HumanMatchFactory:
    profile: HumanProfile
    engine: str
    main_model: str
    human_model: str
    config: str
    budget: ReferenceBudget
    source_revision: str
    runtime_root: str | None = None

    def __call__(self, profile, seed):
        from kenggo.matches import InfrastructureError

        try:
            expected = human_agent_profile(profile["id"], self.profile, profile["conditioning_rank"],
                source_revision=self.source_revision, budget=self.budget, panel=profile["panel"]).data
            if ("weight" in profile and (not _finite(profile["weight"], 0, 1) or profile["weight"] == 0)
                    or {key: value for key, value in profile.items() if key != "weight"} != expected):
                raise ValueError("match_profile_pin_mismatch")
        except Exception as error:
            raise InfrastructureError("match_profile_pin_mismatch") from error
        try:
            client = HumanSL(self.profile, engine=self.engine, main_model=self.main_model,
                            human_model=self.human_model, config=self.config, budget=self.budget,
                            runtime_root=self.runtime_root)
            client.query_version()
        except Exception as error:
            raise InfrastructureError(str(error)) from error
        return _HumanAgent(client, profile["conditioning_rank"], seed)


class _HumanAgent:
    def __init__(self, client, rank, seed):
        self.client, self.rank, self.rng = client, rank, random.Random(seed)

    def choose(self, game):
        from kenggo.matches import Decision, InfrastructureError

        try:
            return Decision(self.client.predict(game, self.rank).sample(self.rng), None, 0.0)
        except ReferenceBudgetExceeded as error:
            raise InfrastructureError(str(error)) from error

    def close(self):
        self.client.close()
