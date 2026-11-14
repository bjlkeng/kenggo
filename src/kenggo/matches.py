"""Frozen match schedules, killable agents, and all-scheduled-game records."""

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import multiprocessing
import os
import pickle
from pathlib import Path
import random
import re
import resource
import signal
import select
import socket
import struct
import time

from kenggo.adjudication import Limits, adjudicate, adjudicate_empirical, judge_pass
from kenggo.go import Game, IllegalMove, Outcome, PASS, RULES, canonical, digest
from kenggo.match_sgf import export_match_sgf
from kenggo.rules_replay import dump_game, load_game


_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


@dataclass(frozen=True)
class AgentProfile:
    payload: str

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("Profile must be an object")
        data = json.loads(canonical(value))
        for field in ("id", "family", "policy", "context_id"):
            if not isinstance(data.get(field), str) or not _ID.fullmatch(data[field]):
                raise ValueError(f"Invalid profile {field}")
        if (data.get("panel") not in ("validation", "final", "candidate")
                or data.get("device") not in ("cpu", "gpu", "unknown")
                or data.get("rules") != RULES or data.get("komi_halfpoints") != 13
                or data.get("context") != {"timed": False, "clock_inputs": "missing",
                                            "opponent_rank": "unknown"}):
            raise ValueError("Profile needs pinned ordinary rules and the supported untimed context")
        pins = data.get("pins")
        if (not isinstance(pins, dict) or not pins
                or any(not isinstance(k, str) or not k or not isinstance(v, str)
                       or not _HASH.fullmatch(v) for k, v in pins.items())):
            raise ValueError("Profile requires artifact SHA-256 pins")
        revision = data.get("source_revision")
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Profile requires an immutable source revision")
        if data["family"] == "weak" and data.get("human_rank") is not None:
            raise ValueError("Weak policies are unranked")
        checkpoint = data.get("checkpoint_sha256")
        if checkpoint is not None and (not isinstance(checkpoint, str) or not _HASH.fullmatch(checkpoint)):
            raise ValueError("Invalid checkpoint pin")
        return cls(canonical(data))

    @property
    def data(self):
        return json.loads(self.payload)

    @property
    def id(self):
        return self.data["id"]


def weak_profile(id, policy="uniform", *, source_revision):
    if policy not in ("uniform", "capture_first"):
        raise ValueError("Unknown initial weak policy")
    return AgentProfile.from_dict({
        "id": id, "family": "weak", "policy": policy, "panel": "validation",
        "context_id": "untimed_missing_clock_unknown_rank_v1",
        "context": {"timed": False, "clock_inputs": "missing", "opponent_rank": "unknown"},
        "device": "cpu", "human_rank": None, "checkpoint_sha256": None,
        "source_revision": source_revision,
        "pins": {"implementation": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "rules": RULES, "komi_halfpoints": 13,
        "sampling": "uniform_among_capturing_placements_else_legal_placements" if policy == "capture_first"
                    else "uniform_among_legal_placements",
        "pass": "only_if_no_legal_placement", "resignation": "never",
        "ending": "exact_first_optional_pinned_empirical_v2"})


@dataclass(frozen=True)
class MatchLimits:
    action_limit: int = 800
    reply_grace_actions: int = 1
    response_seconds: float = 60.0
    game_seconds: float = 3600.0
    shutdown_seconds: float = 0.25
    exact_wall_seconds: float = 10.0
    exact_cpu_seconds: float = 2.0
    empirical_wall_seconds: float = 180.0
    empirical_cpu_seconds: float = 120.0

    def __post_init__(self):
        if (any(not _positive(value) for value in asdict(self).values())
                or type(self.action_limit) is not int or self.action_limit != 800
                or type(self.reply_grace_actions) is not int or self.reply_grace_actions != 1
                or self.response_seconds > 60 or self.game_seconds > 3600
                or self.exact_wall_seconds > 10 or self.exact_cpu_seconds > 2
                or self.empirical_wall_seconds > 180 or self.empirical_cpu_seconds > 120
                or self.shutdown_seconds >= self.game_seconds):
            raise ValueError("Match limits must fit the ordinary-play and judgment budgets")


@dataclass(frozen=True)
class ExperimentBudget:
    cpu_seconds: float
    wall_seconds: float
    output_bytes: int = 128 * 1024**2

    def __post_init__(self):
        if (not _positive(self.cpu_seconds) or not _positive(self.wall_seconds)
                or type(self.output_bytes) is not int or self.output_bytes <= 0):
            raise ValueError("An explicit positive total experiment budget is required")


def _profile_fingerprint(profile):
    return digest({k: v for k, v in profile.items() if k not in ("id", "panel", "weight")})


def _prior_catalog(prior_schedules):
    catalog = []
    for prior in prior_schedules:
        validate_schedule(prior)
        catalog.append({"sha256": digest(prior), "purpose": prior["purpose"],
                        "seeds": sorted({slot[key] for slot in prior["slots"]
                                         for key in ("candidate_seed", "opponent_seed")}),
                        "profiles": [_profile_fingerprint(p) for p in prior["profiles"]]})
    return catalog


def release_binding(schedule):
    """Fields an independently pinned E7 release must bind before execution."""
    return {"schema": "match_final_release_v1", "stage": "E7", "selection_frozen": True,
            "selection_sha256": digest(schedule["arm_profiles"]),
            "panel_sha256": digest(schedule["profiles"]),
            "seed_schedule_sha256": digest(schedule["slots"]),
            "protocol_sha256": digest({key: schedule[key] for key in
                                       ("rules", "limits", "context", "referee_profile", "provenance")})}


def make_schedule(*, arms, profiles, pairs, seed, purpose="validation", scope="setup",
                  control_arm=None, prior_schedules=(), release=None, release_sha256=None,
                  limits=MatchLimits(), referee_profile=None, provenance=None):
    if type(pairs) is not int or not 1 <= pairs <= 100_000 or type(seed) is not int or seed < 0:
        raise ValueError("Positive pair count and nonnegative integer schedule seed required")
    candidates = [profile.data for profile in arms]
    opponents = [profile.data for profile in profiles]
    families = {p["family"] for p in opponents}
    for profile in opponents:
        profile["weight"] = 1 / len(families) / sum(p["family"] == profile["family"] for p in opponents)
    rng = random.Random(seed)
    used, slots = set(), []

    def fresh_seed():
        while (value := rng.randrange(2**63)) in used:
            pass
        used.add(value)
        return value

    for profile_index, profile in enumerate(opponents):
        for pair in range(pairs):
            block = f"p{profile_index:03d}-b{pair:06d}"
            seeds = fresh_seed(), fresh_seed()
            order = candidates if pair % 2 == 0 else candidates[::-1]
            colors = ("B", "W") if pair % 2 == 0 else ("W", "B")
            for color in colors:
                for arm in order:
                    slots.append({"id": f"s{len(slots):08d}", "block_id": block, "arm": arm["id"],
                                  "profile_id": profile["id"], "candidate_color": color,
                                  "candidate_seed": seeds[0], "opponent_seed": seeds[1]})
    schedule = {"version": "match_schedule_v1", "purpose": purpose, "scope": scope,
                "arms": [p["id"] for p in candidates], "control_arm": control_arm,
                "arm_profiles": candidates, "profiles": opponents, "slots": slots,
                "rules": RULES, "komi_halfpoints": 13, "seed": seed,
                "limits": asdict(limits), "context": {"timed": False, "clock_inputs": "missing",
                                                       "opponent_rank": "unknown"},
                "referee_profile": referee_profile, "provenance": provenance or {},
                "prior_catalog": _prior_catalog(prior_schedules),
                "release": release, "release_sha256": release_sha256}
    validate_schedule(schedule)
    return json.loads(canonical(schedule))


def validate_schedule(schedule, *, for_execution=False, release_sha256=None):
    if (not isinstance(schedule, dict) or schedule.get("version") != "match_schedule_v1"
            or schedule.get("purpose") not in ("validation", "confirmation", "final")
            or schedule.get("scope") not in ("setup", "full")
            or schedule.get("rules") != RULES or schedule.get("komi_halfpoints") != 13):
        raise ValueError("Invalid match schedule")
    MatchLimits(**schedule["limits"])
    candidates = [AgentProfile.from_dict(p) for p in schedule["arm_profiles"]]
    opponents = [AgentProfile.from_dict(p) for p in schedule["profiles"]]
    ids = [p.id for p in candidates]
    if (not ids or len(set(ids)) != len(ids) or schedule["arms"] != ids
            or schedule["control_arm"] is not None and schedule["control_arm"] not in ids
            or not opponents or len({p.id for p in opponents}) != len(opponents)):
        raise ValueError("Empty or duplicate arms/profiles or unknown control")
    profiles = {p.id: p.data for p in opponents}
    families = {p["family"] for p in profiles.values()}
    purpose = schedule["purpose"]
    expected = {"weak", "pachi", "fuego"} if purpose == "final" else {"weak", "gnugo", "katago_selfplay"}
    if schedule["scope"] == "full" and families != expected:
        raise ValueError("A full panel needs all three frozen opponent families")
    if schedule["scope"] == "setup" and families != {"weak"}:
        raise ValueError("Setup scope supports only the explicitly narrower weak panel")
    for profile in profiles.values():
        weight = profile.get("weight")
        expected_weight = 1 / len(families) / sum(p["family"] == profile["family"] for p in profiles.values())
        if not _positive(weight) or not math.isclose(weight, expected_weight, abs_tol=1e-12):
            raise ValueError("Profiles must have equal family and within-family weights")
        if profile["panel"] != ("final" if purpose == "final" else "validation"):
            raise ValueError("Validation and final profiles must remain separate")
        if purpose == "final" and profile["family"] == "weak" and profile["policy"] != "local_liberties":
            raise ValueError("The final weak policy is the held-out local-liberties adapter")
        if profile["context"] != schedule["context"]:
            raise ValueError("A profile's context must match its scheduled stratum")
    if any(profile.data["context"] != schedule["context"] for profile in candidates):
        raise ValueError("Candidate context differs from the frozen schedule")
    slots = schedule["slots"]
    if not isinstance(slots, list) or not slots:
        raise ValueError("A schedule needs fixed slots")
    seen, blocks, block_seeds, streams = set(), {}, {}, set()
    for slot in slots:
        if (not isinstance(slot, dict) or set(slot) != {"id", "block_id", "arm", "profile_id",
                                                      "candidate_color", "candidate_seed", "opponent_seed"}
                or any(not isinstance(slot[key], str) or not _ID.fullmatch(slot[key])
                       for key in ("id", "block_id"))
                or slot["id"] in seen or slot["arm"] not in ids or slot["profile_id"] not in profiles
                or slot["candidate_color"] not in ("B", "W")
                or any(type(slot[k]) is not int or not 0 <= slot[k] < 2**63
                       for k in ("candidate_seed", "opponent_seed"))
                or slot["candidate_seed"] == slot["opponent_seed"]):
            raise ValueError("Malformed or duplicate slot, arm, color or seed")
        seen.add(slot["id"])
        key = slot["profile_id"], slot["block_id"]
        seeds = slot["candidate_seed"], slot["opponent_seed"]
        if key not in blocks:
            if any(seed in streams for seed in seeds):
                raise ValueError("Duplicate random-input blocks or shared agent streams")
            streams.update(seeds)
            block_seeds[key] = seeds
            blocks[key] = set()
        if seeds != block_seeds[key] or (slot["arm"], slot["candidate_color"]) in blocks[key]:
            raise ValueError("Block seeds must be common across arms and colors without duplicate games")
        blocks[key].add((slot["arm"], slot["candidate_color"]))
    if any(value != {(arm, color) for arm in ids for color in ("B", "W")} for value in blocks.values()):
        raise ValueError("Every block needs both colors for every arm")
    if {key[0] for key in blocks} != set(profiles):
        raise ValueError("Every weighted profile needs scheduled blocks")
    catalog = schedule["prior_catalog"]
    if purpose == "confirmation" and not catalog:
        raise ValueError("Confirmation requires its previous validation schedule catalog")
    if purpose in ("confirmation", "final"):
        if any(streams.intersection(prior["seeds"]) for prior in catalog):
            raise ValueError("Confirmation/final seeds overlap previously opened schedules")
    if purpose == "final":
        if any(_profile_fingerprint(profile) in prior["profiles"]
               for profile in profiles.values() for prior in catalog):
            raise ValueError("Final configurations overlap an opened panel")
        release, pin = schedule["release"], schedule["release_sha256"]
        if for_execution or release is not None or pin is not None:
            if (not isinstance(release, dict) or release != release_binding(schedule)
                    or not isinstance(pin, str) or not _HASH.fullmatch(pin) or digest(release) != pin
                    or for_execution and release_sha256 != pin):
                raise ValueError("Final candidate execution requires an independently pinned E7 release")
    elif schedule["release"] is not None or schedule["release_sha256"] is not None:
        raise ValueError("An E7 release belongs only to the final panel")
    canonical(schedule)
    return schedule


@dataclass(frozen=True)
class Decision:
    action: int | str
    network_calls: int | None = None
    gpu_seconds: float | None = None


class InfrastructureError(RuntimeError):
    """An adapter cannot attribute this failure to an agent."""


class _WeakAgent:
    def __init__(self, profile, seed):
        self.policy = profile["policy"]
        self.rng = random.Random(seed)

    def choose(self, game):
        placements, captures = [], []
        before = game.position
        for action, stone in enumerate(before.board):
            if stone != ".":
                continue
            try:
                after = before.place(action)
            except IllegalMove:
                continue
            placements.append(action)
            if (before.captured_black, before.captured_white) != (after.captured_black, after.captured_white):
                captures.append(action)
        choices = captures if self.policy == "capture_first" and captures else placements
        return Decision(self.rng.choice(choices) if choices else PASS, 0, 0.0)


def _weak_factory(profile, seed):
    return _WeakAgent(profile, seed)


def _own_cpu():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def _children_cpu():
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def _self_cpu():
    return _own_cpu() + _children_cpu()


def _tree_usage(pid):
    cpu, rss, descendants = 0.0, 0, 0.0
    queue, seen = [pid], set()
    ticks = os.sysconf("SC_CLK_TCK")
    while queue:
        current = queue.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            fields = Path(f"/proc/{current}/stat").read_text().rsplit(")", 1)[1].split()
            value = sum(int(fields[index]) for index in (11, 12, 13, 14)) / ticks
            cpu += value
            if current != pid:
                descendants += value
            rss += int(fields[21]) * os.sysconf("SC_PAGE_SIZE")
            for children in Path(f"/proc/{current}/task").glob("*/children"):
                queue.extend(int(child) for child in children.read_text().split())
        except (OSError, ValueError):
            continue
    return cpu, rss, descendants, seen


class _Meter:
    def __init__(self, budget):
        self.budget, self.started, self.cpu_started = budget, time.monotonic(), _own_cpu()
        self.children_started = _children_cpu()
        self.workers = []
        self.peak_memory_bytes = 0
        self.last_cpu = 0.0

    def cost(self):
        worker_cpu = sum(worker.cpu() for worker in self.workers)
        reaped_elsewhere = max(0.0, _children_cpu() - self.children_started
                              - sum(worker.reaped_cpu for worker in self.workers))
        cpu = _own_cpu() - self.cpu_started + worker_cpu + reaped_elsewhere
        self.last_cpu = max(self.last_cpu, cpu)
        own_rss = _tree_usage(os.getpid())[1]
        self.peak_memory_bytes = max(self.peak_memory_bytes, own_rss)
        return {"cpu_seconds": self.last_cpu, "wall_seconds": time.monotonic() - self.started,
                "peak_memory_bytes": self.peak_memory_bytes,
                "cpu_scope": "parent_and_reaped_children_plus_retained_sampled_descendants",
                "conservative_shutdown_cpu_seconds": sum(worker.shutdown_charge for worker in self.workers),
                "cpu_sampling_limit": "20ms polling; killed descendants retain observed CPU and conservative shutdown charge",
                "memory_scope": "sampled_parent_and_live_descendant_rss"}

    def remaining(self):
        cost = self.cost()
        return (self.budget.cpu_seconds - cost["cpu_seconds"],
                self.budget.wall_seconds - cost["wall_seconds"])


@dataclass(frozen=True)
class _Failure:
    reason: str
    responsible: bool


_FRAME_LIMIT = 16 * 1024**2


def _frame(value):
    payload = pickle.dumps(value, protocol=5)
    if len(payload) > _FRAME_LIMIT:
        raise ValueError("Worker payload exceeds the fixed frame limit")
    return struct.pack("!I", len(payload)) + payload


class _Wire:
    def __init__(self, channel):
        self.channel = channel
        self.buffer = bytearray()

    def send(self, value):
        self.channel.sendall(_frame(value))

    def recv(self):
        while True:
            value = self.extract()
            if value is not None:
                return value
            self.read()

    def read(self):
        needed = 4 - len(self.buffer) if len(self.buffer) < 4 else (
            4 + struct.unpack("!I", self.buffer[:4])[0] - len(self.buffer))
        data = self.channel.recv(min(65536, max(1, needed)))
        if not data:
            raise EOFError("Worker transport closed")
        self.buffer.extend(data)
        if len(self.buffer) >= 4 and struct.unpack("!I", self.buffer[:4])[0] > _FRAME_LIMIT:
            raise ValueError("Worker frame exceeds the fixed byte limit")

    def extract(self):
        if len(self.buffer) < 4:
            return None
        size = struct.unpack("!I", self.buffer[:4])[0]
        if size > _FRAME_LIMIT:
            raise ValueError("Worker frame exceeds the fixed byte limit")
        if len(self.buffer) < size + 4:
            return None
        raw = bytes(self.buffer[4:size + 4])
        del self.buffer[:size + 4]
        return pickle.loads(raw)


def _worker_main(channel, role, factory, profile, seed):
    os.setsid()
    connection, obj = _Wire(channel), None
    try:
        connection.send(("ready", None))
        if role == "agent":
            try:
                obj = factory(profile, seed)
            except InfrastructureError:
                raise
            except BaseException as error:
                connection.send(("failure", _Failure(f"initialization: {type(error).__name__}: {error}", True)))
                return
        elif role == "empirical":
            obj = factory()
            if obj.profile.sha256 != profile["sha256"]:
                raise InfrastructureError("referee_profile_pin_mismatch")
        connection.send(("initialized", None))
        while True:
            operation, game, limits = connection.recv()
            started, cpu = time.monotonic(), _self_cpu()
            try:
                if operation == "choose":
                    result = obj.choose(game)
                elif operation == "end_exact":
                    result = adjudicate(game, limits=Limits(cpu_seconds=limits.exact_cpu_seconds,
                                                           wall_seconds=limits.exact_wall_seconds))
                elif operation == "pass_exact":
                    result = judge_pass(game, limits=Limits(cpu_seconds=limits.exact_cpu_seconds,
                                                           wall_seconds=limits.exact_wall_seconds))
                elif operation == "end_empirical":
                    result = adjudicate_empirical(game, obj)
                else:
                    result = obj.judge_pass(game)
                    if result.snapshot != game.snapshot or result.verdict not in ("acceptable", "premature", "unjudgeable"):
                        raise InfrastructureError("invalid_or_stale_pass_judgment")
                    asdict(result)
                connection.send(("result", (result, {"cpu_seconds": _self_cpu() - cpu,
                                                   "wall_seconds": time.monotonic() - started})))
            except InfrastructureError as error:
                connection.send(("failure", _Failure(str(error), False)))
            except BaseException as error:
                connection.send(("failure", _Failure(f"{type(error).__name__}: {error}", role == "agent")))
    except (EOFError, BrokenPipeError):
        pass
    except BaseException as error:
        try:
            connection.send(("failure", _Failure(f"initialization: {type(error).__name__}: {error}", False)))
        except (OSError, EOFError):
            pass
    finally:
        if obj is not None and hasattr(obj, "close"):
            obj.close()
        channel.close()


class _Worker:
    def __init__(self, role, factory, profile, seed, meter, *, _target=_worker_main):
        context = multiprocessing.get_context("spawn")
        self.channel, child = socket.socketpair()
        self.channel.setblocking(False)
        self.wire = _Wire(self.channel)
        self.process = context.Process(target=_target, args=(child, role, factory, profile, seed))
        self.closed, self.role, self.meter = False, role, meter
        self.previous_cpu = self.retained_cpu = self.final_cpu = self.reaped_cpu = self.shutdown_charge = 0.0
        self.started = time.monotonic()
        self.process.start()
        child.close()
        meter.workers.append(self)

    def cpu(self):
        if self.closed:
            return self.final_cpu
        current = _tree_usage(self.process.pid)[0]
        if current < self.previous_cpu:
            self.retained_cpu += self.previous_cpu - current
        self.previous_cpu = current
        return current + self.retained_cpu

    def _check(self, stop, game_deadline, response, cpu_started, cpu_limit):
        cpu_left, wall_left = self.meter.remaining()
        if cpu_left <= 0 or wall_left <= 0:
            return _Failure("experiment_budget", False)
        if time.monotonic() >= game_deadline:
            return _Failure("game_deadline", False)
        if cpu_limit is not None and self.meter.cost()["cpu_seconds"] - cpu_started >= cpu_limit:
            return _Failure("judgment_cpu_budget", False)
        if time.monotonic() >= stop:
            return _Failure("response_timeout" if response else "judgment_or_startup_timeout", response)
        return None

    def exchange(self, seconds, game_deadline, *, request=None, response=False, cpu_limit=None, cpu_started=None):
        stop = time.monotonic() + seconds
        if cpu_started is None:
            cpu_started = self.meter.cost()["cpu_seconds"]
        try:
            outgoing = memoryview(_frame(request)) if request is not None else memoryview(b"")
            while True:
                failure = self._check(stop, game_deadline, response, cpu_started, cpu_limit)
                if failure is not None:
                    self.close()
                    return "failure", failure
                if not outgoing:
                    value = self.wire.extract()
                    if value is not None:
                        return value
                readable, writable, _ = select.select([self.channel] if not outgoing else [],
                                                       [self.channel] if outgoing else [], [],
                                                       max(0.0, min(0.02, stop - time.monotonic())))
                if writable:
                    try:
                        count = self.channel.send(outgoing[:65536])
                        outgoing = outgoing[count:]
                    except BlockingIOError:
                        pass
                if readable:
                    try:
                        self.wire.read()
                    except BlockingIOError:
                        pass
        except (EOFError, OSError, ValueError, pickle.UnpicklingError) as error:
            self.close()
            reason = "agent_crash" if isinstance(error, EOFError) and response else "worker_transport"
            return "failure", _Failure(reason, reason == "agent_crash")

    def initialize(self, seconds, game_deadline, *, cpu_limit=None, cpu_started=None):
        stop = self.started + seconds
        for expected in ("ready", "initialized"):
            kind, value = self.exchange(max(0.0, stop - time.monotonic()), game_deadline,
                                        response=self.role == "agent" and expected == "initialized",
                                        cpu_limit=cpu_limit, cpu_started=cpu_started)
            if kind != expected:
                return value if kind == "failure" else _Failure("invalid_worker_protocol", False)
        return None

    def call(self, operation, game, limits, seconds, game_deadline, *, cpu_limit=None, cpu_started=None):
        before = time.monotonic()
        kind, payload = self.exchange(seconds, game_deadline, request=(operation, game, limits),
                                      response=operation == "choose", cpu_limit=cpu_limit, cpu_started=cpu_started)
        if kind == "result":
            value, cost = payload
            cost["wall_seconds"] = time.monotonic() - before
            return value, cost
        return (payload if kind == "failure" else _Failure("invalid_worker_protocol", False),
                {"wall_seconds": time.monotonic() - before, "cpu_seconds": None})

    def close(self):
        if self.closed:
            return
        before, observed, reaped = time.monotonic(), self.cpu(), _children_cpu()
        _, _, descendants, pids = _tree_usage(self.process.pid)
        # Native grandchildren may become orphans when their owner is killed.
        # Retain their observed CPU and charge the shutdown interval conservatively.
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            try:
                os.kill(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for pid in pids - {self.process.pid}:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.process.join(timeout=0.25)
        if self.process.exitcode is None:
            self.process.kill()
            self.process.join(timeout=0.25)
        self.reaped_cpu = max(0.0, _children_cpu() - reaped)
        if len(pids) > 1:
            self.shutdown_charge = (time.monotonic() - before + 0.02) * len(os.sched_getaffinity(0))
        self.final_cpu = max(observed, self.reaped_cpu + descendants + self.retained_cpu) + self.shutdown_charge
        self.exitcode = self.process.exitcode
        if self.exitcode is not None:
            self.process.close()
        else:
            self.final_cpu += max(0, self.meter.budget.cpu_seconds - self.final_cpu)
        self.channel.close()
        self.closed = True


def _judgment(game, operation, *, limits, meter, deadline, referee_factory, referee_profile):
    empirical = operation.endswith("empirical")
    seconds = limits.empirical_wall_seconds if empirical else limits.exact_wall_seconds
    cpu_limit = limits.empirical_cpu_seconds if empirical else limits.exact_cpu_seconds
    cpu_left, wall_left = meter.remaining()
    if cpu_left <= limits.shutdown_seconds or wall_left <= limits.shutdown_seconds:
        return _Failure("judgment_experiment_budget", False), {"cpu_seconds": 0.0, "wall_seconds": 0.0}
    worker = None
    started, cpu = time.monotonic(), meter.cost()["cpu_seconds"]
    try:
        worker = _Worker("empirical" if empirical else "exact", referee_factory,
                         referee_profile, None, meter)
        stop = min(deadline, started + seconds)
        failure = worker.initialize(seconds, stop, cpu_limit=cpu_limit, cpu_started=cpu)
        if failure is None:
            result, _ = worker.call(operation, game, limits, max(0.001, stop - time.monotonic()),
                                    stop, cpu_limit=cpu_limit, cpu_started=cpu)
        else:
            result = failure
    except (OSError, ValueError) as error:
        result = _Failure(f"judgment_infrastructure: {error}", False)
    finally:
        if worker is not None:
            worker.close()
    cost = {"cpu_seconds": max(0.0, meter.cost()["cpu_seconds"] - cpu),
            "wall_seconds": time.monotonic() - started}
    if cost["cpu_seconds"] > cpu_limit or cost["wall_seconds"] > seconds:
        result = _Failure("judgment_cpu_budget" if cost["cpu_seconds"] > cpu_limit else "judgment_wall_budget", False)
    return result, cost


def _attempt(result, cost, method, snapshot):
    if isinstance(result, _Failure):
        return {"method": method, "snapshot": snapshot, "reason": result.reason,
                "status": "unjudgeable" if method.startswith("pass") else "unresolved",
                "worker_cost": cost}
    decision = result[1] if isinstance(result, tuple) else result
    value = asdict(decision)
    for key in ("cost", "payload", "evidence", "witness"):
        if isinstance(value.get(key), str):
            try:
                value[key] = json.loads(value[key])
            except ValueError:
                pass
    value.update(method=method, worker_cost=cost)
    return value


def _unknown_ending(game, reason, attempts):
    evidence = canonical({"method": "match_unresolved_v1", "snapshot": game.snapshot,
                          "reason": reason, "attempts": attempts})
    return replace(game, status="unresolved", outcome=Outcome("unresolved", reason, evidence=evidence))


def _end(game, *, limits, meter, deadline, referee_factory, referee_profile):
    pending, attempts = game, []
    exact, cost = _judgment(pending, "end_exact", limits=limits, meter=meter, deadline=deadline,
                            referee_factory=None, referee_profile=None)
    attempts.append(_attempt(exact, cost, "end_exact", pending.snapshot))
    if not isinstance(exact, _Failure) and exact[0].status == "resolved":
        return exact[0], attempts
    if referee_factory is not None:
        empirical, cost = _judgment(pending, "end_empirical", limits=limits, meter=meter, deadline=deadline,
                                    referee_factory=referee_factory, referee_profile=referee_profile)
        attempts.append(_attempt(empirical, cost, "end_empirical", pending.snapshot))
        if not isinstance(empirical, _Failure):
            return empirical[0], attempts
        return _unknown_ending(pending, empirical.reason, attempts), attempts
    if isinstance(exact, _Failure):
        return _unknown_ending(pending, exact.reason, attempts), attempts
    return exact[0], attempts


def _pass(game, *, limits, meter, deadline, referee_factory, referee_profile):
    attempts = []
    exact, cost = _judgment(game, "pass_exact", limits=limits, meter=meter, deadline=deadline,
                            referee_factory=None, referee_profile=None)
    attempts.append(_attempt(exact, cost, "pass_exact", game.snapshot))
    if not isinstance(exact, _Failure) and exact.verdict != "unjudgeable":
        return {"snapshot": game.snapshot, "move_index": len(game.moves), "verdict": exact.verdict,
                "method": "exact", "attempts": attempts}
    if referee_factory is not None:
        empirical, cost = _judgment(game, "pass_empirical", limits=limits, meter=meter, deadline=deadline,
                                    referee_factory=referee_factory, referee_profile=referee_profile)
        attempts.append(_attempt(empirical, cost, "pass_empirical", game.snapshot))
        if not isinstance(empirical, _Failure):
            if empirical.snapshot != game.snapshot or empirical.verdict not in ("acceptable", "premature", "unjudgeable"):
                raise ValueError("Pass referee returned a stale or invalid judgment")
            return {"snapshot": game.snapshot, "move_index": len(game.moves), "verdict": empirical.verdict,
                    "method": "empirical", "attempts": attempts}
    return {"snapshot": game.snapshot, "move_index": len(game.moves), "verdict": "unjudgeable",
            "method": "unresolved", "attempts": attempts}


def _disagreement(attempts):
    scores = []
    reason = None
    for attempt in attempts:
        evidence = attempt.get("evidence")
        if isinstance(evidence, dict):
            scores = [run.get("terminal", {}).get("white_minus_black_halfpoints")
                      for run in evidence.get("runs", []) if isinstance(run.get("terminal"), dict)]
            scores = [score for score in scores if type(score) is int]
        if "disagreement" in attempt.get("reason", ""):
            reason = attempt["reason"]
    if len(scores) == 2 and scores[0] != scores[1]:
        kind = "winner" if (scores[0] > 0) != (scores[1] > 0) else "margin"
    else:
        kind = "referee" if reason else None
    return {"kind": kind, "reason": reason, "cleanup_scores_halfpoints": scores,
            "spread_halfpoints": abs(scores[1] - scores[0]) if len(scores) == 2 else None,
            "gate_failure": None if kind == "margin" else bool(kind),
            "margin_gate_policy": "deferred_before_qualification"}


def _decision_valid(value):
    return (isinstance(value, Decision)
            and (type(value.action) is int and 0 <= value.action <= PASS
                 or type(value.action) is str and value.action == "resign")
            and (value.network_calls is None or type(value.network_calls) is int and value.network_calls >= 0)
            and (value.gpu_seconds is None or type(value.gpu_seconds) in (int, float)
                 and math.isfinite(value.gpu_seconds) and value.gpu_seconds >= 0))


@dataclass(frozen=True)
class MatchResult:
    game: Game
    record: dict


def _resolve_factory(profile, factories):
    if profile["policy"] in ("uniform", "capture_first"):
        if profile["pins"].get("implementation") != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
            raise ValueError("Weak implementation differs from its frozen profile")
        return _weak_factory
    if profile["policy"] not in factories:
        raise ValueError(f"No configured adapter for {profile['policy']}")
    return factories[profile["policy"]]


def run_game(slot, candidate_profile, opponent_profile, *, budget, limits=MatchLimits(),
             agent_factories=None, referee_factory=None, referee_profile=None,
             initial=None, allow_fixture=False, _meter=None):
    meter = _meter or _Meter(budget)
    start_cost, started = meter.cost(), time.monotonic()
    candidate, other = candidate_profile.data, opponent_profile.data
    slot = json.loads(canonical(slot))
    referee_profile = json.loads(canonical(referee_profile))
    if (slot.get("arm") != candidate["id"] or slot.get("profile_id") != other["id"]
            or slot.get("candidate_color") not in ("B", "W")
            or any(type(slot.get(key)) is not int or slot[key] < 0
                   for key in ("candidate_seed", "opponent_seed"))
            or slot["candidate_seed"] == slot["opponent_seed"]):
        raise ValueError("Slot and profiles do not agree")
    color = slot["candidate_color"]
    if initial is not None and not allow_fixture:
        raise ValueError("Production matches start empty; fixtures need explicit authorization")
    game = initial if initial is not None else Game.new(candidate_color=color)
    if game.candidate_color != color:
        raise ValueError("Game must preserve its configured candidate color")
    if (referee_factory is None) != (referee_profile is None):
        raise ValueError("Empirical judging needs both a factory and a frozen referee profile")
    if referee_profile is not None and (not isinstance(referee_profile.get("sha256"), str)
                                        or not _HASH.fullmatch(referee_profile["sha256"])):
        raise ValueError("Referee profile requires an independent SHA-256 pin")
    factories = agent_factories or {}
    definitions = {color: (candidate, slot["candidate_seed"]),
                   "W" if color == "B" else "B": (other, slot["opponent_seed"])}
    configured = {side: _resolve_factory(profile, factories) for side, (profile, _) in definitions.items()}
    deadline = min(started + limits.game_seconds, meter.started + budget.wall_seconds)
    workers, decisions, passes, attempts = {}, [], [], []
    initial_count, original_classification = len(game.moves), game.status
    runtime_failure, fault = False, None
    try:
        while game.status == "playing":
            side = game.position.to_play
            cpu_left, wall_left = meter.remaining()
            reserve = (2 if side == color else 1) * limits.response_seconds + limits.shutdown_seconds
            if wall_left < reserve or deadline - time.monotonic() < reserve or cpu_left <= limits.shutdown_seconds:
                reason = "experiment_budget" if wall_left < reserve or cpu_left <= limits.shutdown_seconds else "game_deadline"
                runtime_failure = None
                fault = {"responsible": None, "reason": reason, "source": "infrastructure"}
                game = (game.truncate(reason, candidate_color=color) if side == color
                        else game.fail(reason))
                break
            if side not in workers:
                profile, seed = definitions[side]
                try:
                    workers[side] = _Worker("agent", configured[side], profile, seed, meter)
                    failure = workers[side].initialize(limits.response_seconds, deadline)
                except (OSError, ValueError) as error:
                    failure = _Failure(f"worker_startup: {error}", False)
                if failure is not None:
                    runtime_failure = True if failure.responsible and side == color else None
                    fault = {"responsible": side if failure.responsible else None, "reason": failure.reason,
                             "source": "agent" if failure.responsible else "infrastructure"}
                    game = game.fail(failure.reason, responsible=side if failure.responsible else None)
                    break
            before_cpu = meter.cost()["cpu_seconds"]
            value, cost = workers[side].call("choose", game, limits, limits.response_seconds, deadline)
            cost["cpu_seconds"] = max(0.0, meter.cost()["cpu_seconds"] - before_cpu)
            valid = _decision_valid(value)
            decision = {"color": side, "move_index": len(game.moves), "snapshot": game.snapshot,
                        "action": value.action if valid else None,
                        "network_calls": value.network_calls if valid else None,
                        "gpu_seconds": value.gpu_seconds if valid else None,
                        "cost": cost}
            decisions.append(decision)
            if isinstance(value, _Failure):
                runtime_failure = True if value.responsible and side == color else None
                fault = {"responsible": side if value.responsible else None, "reason": value.reason,
                         "source": "agent" if value.responsible else "infrastructure"}
                decision["failure"] = value.reason
                game = game.fail(value.reason, responsible=side if value.responsible else None)
                break
            if not valid:
                runtime_failure = True if side == color else None
                fault = {"responsible": side, "reason": "invalid_agent_response", "source": "agent"}
                decision["failure"] = "invalid_agent_response"
                decision["action"] = None
                game = game.fail("invalid_agent_response", responsible=side)
                break
            try:
                if value.action == "resign":
                    game = game.resign(side)
                else:
                    if value.action == PASS and side == color:
                        passes.append(_pass(game, limits=limits, meter=meter, deadline=deadline,
                                            referee_factory=referee_factory, referee_profile=referee_profile))
                    game = game.play(side, value.action)
            except IllegalMove as error:
                runtime_failure = True if side == color else None
                fault = {"responsible": side, "reason": "illegal_action", "source": "agent"}
                decision["failure"] = str(error)
                game = game.fail("illegal_action", responsible=side)
                break
        original_classification = game.status
        if game.status == "pending_adjudication":
            pending = game
            game, attempts = _end(game, limits=limits, meter=meter, deadline=deadline,
                                  referee_factory=referee_factory, referee_profile=referee_profile)
            if len(dump_game(game)) > 2_000_000:
                game = _unknown_ending(pending, "replay_evidence_byte_limit",
                                       [{"attempt_sha256": digest(attempt)} for attempt in attempts])
    finally:
        for worker in workers.values():
            worker.close()
    cost = meter.cost()
    cost["cpu_seconds"] = max(0.0, cost["cpu_seconds"] - start_cost["cpu_seconds"])
    cost["wall_seconds"] = time.monotonic() - started
    calls = [decision["network_calls"] for decision in decisions]
    gpu = [decision["gpu_seconds"] for decision in decisions]
    known_cpu = all(profile["device"] == "cpu" for profile, _ in definitions.values())
    if referee_profile is not None:
        known_cpu = known_cpu and referee_profile.get("device") == "cpu"
    cost["gpu_seconds"] = 0.0 if known_cpu else None
    cost["gpu_scope"] = "cpu_only_profiles" if known_cpu else "caller_owned_external_allocation_ledger_required"
    cost["reported_agent_gpu_seconds"] = sum(gpu) if gpu and all(value is not None for value in gpu) else None
    cost["agent_network_calls"] = sum(calls) if all(value is not None for value in calls) else None
    all_attempts = attempts + [attempt for passed in passes for attempt in passed["attempts"]]
    empirical_calls = []
    for attempt in all_attempts:
        if attempt["method"].endswith("empirical"):
            evidence = attempt.get("evidence", {})
            calls = evidence.get("calls") if isinstance(evidence, dict) else None
            empirical_calls.append(sum(call.get("kind") == "analysis" for call in calls)
                                   if isinstance(calls, list) else None)
    cost["referee_network_calls"] = (sum(empirical_calls) if all(value is not None for value in empirical_calls)
                                     else None)
    latency = {}
    for role, side in (("candidate", color), ("opponent", "W" if color == "B" else "B")):
        values = sorted(decision["cost"]["wall_seconds"] for decision in decisions if decision["color"] == side)
        latency[role] = {"count": len(values), "method": "nearest_rank",
                         "median_seconds": values[math.ceil(len(values) * 0.5) - 1] if values else None,
                         "p95_seconds": values[math.ceil(len(values) * 0.95) - 1] if values else None}
    dispute = _disagreement(attempts)
    premature = True if any(p["verdict"] == "premature" for p in passes) else (
        None if any(p["verdict"] == "unjudgeable" for p in passes) else False)
    resolved = game.status == "resolved"
    reason = game.outcome.reason if game.outcome else game.pending_cap or game.status
    record = {"version": "match_game_v1", "slot": dict(slot), "result": game.match_result(color),
              "status": game.status, "reason": reason,
              "original_classification": original_classification,
              "outcome_kind": game.outcome.kind if game.outcome else None,
              "winner": game.outcome.winner if game.outcome else None,
              "white_minus_black_halfpoints": game.outcome.white_minus_black_halfpoints if game.outcome else None,
              "candidate_profile": candidate, "opponent_profile": other,
              "referee_profile": referee_profile, "limits": asdict(limits),
              "context": candidate["context"], "snapshot": game.snapshot,
              "prefix_actions": initial_count, "actual_actions": len(game.moves),
              "continuation_actions": len(game.moves) - initial_count, "censored": not resolved,
              "decisions": decisions, "candidate_passes": passes, "adjudication_attempts": attempts,
              "disagreement": dispute, "fault": fault,
              "behavior": {"runtime_failure": runtime_failure, "cap": game.status == "truncated",
                           "unresolved": game.match_result(color) == "U", "dispute": dispute["gate_failure"],
                           "premature_pass": premature,
                           "length_tail": True if len(game.moves) > 600 else False if resolved else None},
              "cost": cost, "latency": latency, "state": json.loads(dump_game(game))}
    return MatchResult(game, record)


def _write_new(path, payload, budget, written):
    if written + len(payload) > budget.output_bytes:
        raise ValueError("Match output byte budget exhausted")
    temporary = path.with_name(f".{path.name}.{os.urandom(16).hex()}.partial")
    created = False
    try:
        with temporary.open("xb") as stream:
            created = True
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if created:
            temporary.unlink(missing_ok=True)
    return written + len(payload)


def run_schedule(schedule, output_dir, *, budget, limits=None, agent_factories=None,
                 referee_factory=None, referee_profile=None, release_sha256=None):
    meter = _Meter(budget)
    schedule = json.loads(canonical(schedule))
    validate_schedule(schedule, for_execution=True, release_sha256=release_sha256)
    frozen_limits = MatchLimits(**schedule["limits"])
    if limits is not None and limits != frozen_limits:
        raise ValueError("Execution limits differ from the frozen schedule")
    if referee_profile is not None and referee_profile != schedule["referee_profile"]:
        raise ValueError("Execution referee differs from the frozen schedule")
    referee_profile = schedule["referee_profile"]
    if (referee_factory is None) != (referee_profile is None):
        raise ValueError("Frozen empirical judging requires its configured factory")
    candidates = {p["id"]: AgentProfile.from_dict(p) for p in schedule["arm_profiles"]}
    opponents = {p["id"]: AgentProfile.from_dict(p) for p in schedule["profiles"]}
    for profile in (*candidates.values(), *opponents.values()):
        _resolve_factory(profile.data, agent_factories or {})
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    written = _write_new(root / "schedule.json", (canonical(schedule) + "\n").encode(), budget, 0)
    written = _write_new(root / "budget.json", (canonical(asdict(budget)) + "\n").encode(), budget, written)
    records = {}
    for slot in schedule["slots"]:
        cpu_left, wall_left = meter.remaining()
        if cpu_left <= frozen_limits.shutdown_seconds or wall_left < 2 * frozen_limits.response_seconds + frozen_limits.shutdown_seconds:
            break
        result = run_game(slot, candidates[slot["arm"]], opponents[slot["profile_id"]], budget=budget,
                          limits=frozen_limits, agent_factories=agent_factories,
                          referee_factory=referee_factory, referee_profile=referee_profile, _meter=meter)
        sgf_bytes = export_match_sgf(result.game, metadata={"slot": slot, "schedule_sha256": digest(schedule)})
        record = json.loads(canonical(result.record | {"sgf_file": f"{slot['id']}.sgf",
                                  "sgf_sha256": hashlib.sha256(sgf_bytes).hexdigest()}))
        raw = (canonical(record) + "\n").encode()
        if written + len(raw) + len(sgf_bytes) > budget.output_bytes:
            raise ValueError("Match output byte budget exhausted; schedule retains every unfinished slot")
        written = _write_new(root / record["sgf_file"], sgf_bytes, budget, written)
        written = _write_new(root / f"{slot['id']}.json", raw, budget, written)
        records[slot["id"]] = record
    total = meter.cost()
    total.update(scheduled=len(schedule["slots"]), recorded=len(records), unfilled=len(schedule["slots"]) - len(records),
                 result_counts={key: sum(r["result"] == key for r in records.values())
                                + (len(schedule["slots"]) - len(records) if key == "U" else 0) for key in ("W", "L", "U")},
                 output_bytes_before_summary=written, qualification_claim=False)
    _write_new(root / "run.json", (canonical(total) + "\n").encode(), budget, written)
    return records


def _validate_pass_record(passed, game, limits, referee_profile, authorized_hashes):
    attempts = passed["attempts"]
    if (not isinstance(attempts, list) or not 1 <= len(attempts) <= 2
            or [a["method"] for a in attempts] != ["pass_exact", "pass_empirical"][:len(attempts)]
            or any(a["snapshot"] != game.snapshot for a in attempts)
            or len(attempts) == 2 and referee_profile is None):
        raise ValueError("Saved pass judgment attempts differ from their pre-pass state or referee")
    last = attempts[-1]
    verdict = last.get("verdict", "unjudgeable")
    if (verdict not in ("acceptable", "premature", "unjudgeable") or passed["verdict"] != verdict
            or passed["method"] != ("empirical" if last["method"] == "pass_empirical" and "verdict" in last
                                    else "exact" if verdict != "unjudgeable" else "unresolved")):
        raise ValueError("Saved pass verdict differs from its judgment evidence")
    if verdict != "unjudgeable" and last["method"] == "pass_exact":
        checked = judge_pass(game, limits=Limits(cpu_seconds=limits.exact_cpu_seconds,
                                               wall_seconds=limits.exact_wall_seconds))
        if (checked.verdict != verdict or checked.reason != last["reason"]
                or (json.loads(checked.witness) if checked.witness else None) != last.get("witness")):
            raise ValueError("Saved exact pass judgment failed revalidation")
    if verdict != "unjudgeable" and last["method"] == "pass_empirical":
        evidence = last["evidence"]
        if (digest(evidence) not in authorized_hashes or evidence.get("snapshot") != game.snapshot
                or evidence.get("profile_sha256") != referee_profile["sha256"]
                or evidence.get("status") != verdict or evidence.get("reason") != last["reason"]):
            raise ValueError("Saved empirical pass judgment requires independently pinned evidence")


def _validate_observations(game, record, authorized_hashes):
    decisions, passes, attempts = record["decisions"], record["candidate_passes"], record["adjudication_attempts"]
    n = len(game.moves)
    if (record["prefix_actions"] != 0 or record["continuation_actions"] != n
            or record["censored"] != (game.status != "resolved")
            or not isinstance(decisions, list) or len(decisions) not in (n, n + 1)
            or not isinstance(passes, list) or not isinstance(attempts, list)):
        raise ValueError("Saved observation counts or censoring differ from the replay")
    current, pass_states = Game.new(candidate_color=game.candidate_color), []
    for index, decision in enumerate(decisions):
        if (decision["move_index"] != index or decision["snapshot"] != current.snapshot
                or decision["color"] != current.position.to_play):
            raise ValueError("Saved decision differs from its pre-action state")
        if (decision["action"] is None and (decision["network_calls"] is not None or decision["gpu_seconds"] is not None)
                or decision["action"] is not None and not _decision_valid(
                    Decision(decision["action"], decision["network_calls"], decision["gpu_seconds"]))):
            raise ValueError("Saved decision contains invalid metadata")
        if index < n:
            move = game.moves[index]
            if (decision["action"] != move.action or decision["color"] != move.color or "failure" in decision
                    or not _decision_valid(Decision(decision["action"], decision["network_calls"], decision["gpu_seconds"]))):
                raise ValueError("Saved decision does not match the actual action")
            if move.action == PASS and move.color == game.candidate_color:
                pass_states.append((index, current))
            current = current.play(move.color, move.action)
        elif decision["action"] == "resign":
            if game.outcome is None or game.outcome.kind != "resignation" or "failure" in decision:
                raise ValueError("Saved resignation does not match the result")
        elif "failure" not in decision or record["fault"] is None:
            raise ValueError("An extra decision must explain the recorded failure")
    if game.outcome is not None and game.outcome.kind == "resignation" and len(decisions) != n + 1:
        raise ValueError("Saved resignation decision is missing")
    if len(passes) != len(pass_states):
        raise ValueError("Saved pass judgments must cover every actual candidate pass")
    for passed, (index, state) in zip(passes, pass_states):
        if passed["move_index"] != index or passed["snapshot"] != state.snapshot:
            raise ValueError("Saved pass judgment differs from the actual pass")
        _validate_pass_record(passed, state, MatchLimits(**record["limits"]),
                              record["referee_profile"], authorized_hashes)
    if current.status == "pending_adjudication":
        if (not 1 <= len(attempts) <= 2
                or [a["method"] for a in attempts] != ["end_exact", "end_empirical"][:len(attempts)]
                or any(a["snapshot"] != current.snapshot for a in attempts)
                or len(attempts) == 2 and record["referee_profile"] is None
                or record["original_classification"] != "pending_adjudication"):
            raise ValueError("Saved ending attempts differ from the frozen ending")
        evidence = json.loads(game.outcome.evidence)
        if evidence.get("method") == "match_unresolved_v1":
            retained = ([{"attempt_sha256": digest(a)} for a in attempts]
                        if evidence["reason"] == "replay_evidence_byte_limit" else attempts)
            if evidence["attempts"] != retained:
                raise ValueError("Saved unresolved ending attempts differ from the replay evidence")
        elif attempts[-1]["method"] == "end_empirical":
            if (attempts[-1].get("evidence") != evidence
                    or evidence.get("profile_sha256") != record["referee_profile"]["sha256"]):
                raise ValueError("Saved empirical score evidence differs from the replay")
        elif (attempts[-1].get("reason") != evidence["reason"]
              or attempts[-1].get("payload") != (json.loads(evidence["payload"]) if evidence["payload"] else None)
              or attempts[-1].get("white_minus_black_halfpoints") != game.outcome.white_minus_black_halfpoints):
            raise ValueError("Saved exact score evidence differs from the replay")
    elif attempts or record["original_classification"] != game.status:
        raise ValueError("Saved adjudication attempts have no frozen ending")
    fault = record["fault"]
    if fault is not None:
        responsible = ("W" if game.outcome.winner == "B" else "B") if game.outcome and game.outcome.winner else None
        reason = game.outcome.reason if game.outcome else game.pending_cap
        if (fault["responsible"] != responsible or fault["reason"] != reason
                or fault["source"] != ("agent" if responsible else "infrastructure")
                or game.outcome is not None and game.outcome.kind != "fault"):
            raise ValueError("Saved fault attribution differs from the replay")
    elif game.outcome is not None and game.outcome.kind == "fault":
        raise ValueError("Saved fault attribution is missing")
    dispute = _disagreement(attempts)
    premature = True if any(p["verdict"] == "premature" for p in passes) else (
        None if any(p["verdict"] == "unjudgeable" for p in passes) else False)
    expected = {"runtime_failure": False if fault is None else True if fault["responsible"] == game.candidate_color else None,
                "cap": game.status == "truncated", "unresolved": game.match_result(game.candidate_color) == "U",
                "dispute": dispute["gate_failure"], "premature_pass": premature,
                "length_tail": True if n > 600 else False if game.status == "resolved" else None}
    if record["behavior"] != expected or record["disagreement"] != dispute:
        raise ValueError("Saved behavior flags differ from the replay and judgment evidence")


def read_records(output_dir, *, authorized_hashes=frozenset(), empirical_referee=None):
    """Read completed records once; an interrupted or unfilled slot stays absent/U."""
    from kenggo.match_sgf import replay_match_sgf

    root = Path(output_dir)
    schedule = json.loads((root / "schedule.json").read_text())
    validate_schedule(schedule)
    if empirical_referee is not None and (schedule["referee_profile"] is None
            or empirical_referee.profile.sha256 != schedule["referee_profile"]["sha256"]):
        raise ValueError("Replay referee differs from the frozen schedule")
    slots = {slot["id"]: slot for slot in schedule["slots"]}
    records = {}
    for path in root.glob("*.json"):
        if path.name in ("schedule.json", "budget.json", "run.json"):
            continue
        if path.stem not in slots:
            raise ValueError("Result file has no scheduled slot")
        record = json.loads(path.read_text())
        canonical(record)
        if record["slot"] != slots[path.stem] or record["sgf_file"] != f"{path.stem}.sgf":
            raise ValueError("Saved result differs from its scheduled slot")
        data = (root / record["sgf_file"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != record["sgf_sha256"]:
            raise ValueError("Saved SGF digest mismatch")
        game = replay_match_sgf(data, authorized_hashes=authorized_hashes, empirical_referee=empirical_referee,
                                expected_metadata={"slot": slots[path.stem], "schedule_sha256": digest(schedule)})
        state = load_game(canonical(record["state"]), authorized_hashes=authorized_hashes,
                          empirical_referee=empirical_referee)
        if (game != state or game.candidate_color != record["slot"]["candidate_color"]
                or game.match_result(game.candidate_color) != record["result"]
                or game.snapshot != record["snapshot"] or game.status != record["status"]
                or len(game.moves) != record["actual_actions"]
                or (game.outcome.reason if game.outcome else game.pending_cap or game.status) != record["reason"]
                or (game.outcome.kind if game.outcome else None) != record["outcome_kind"]
                or (game.outcome.winner if game.outcome else None) != record["winner"]
                or (game.outcome.white_minus_black_halfpoints if game.outcome else None)
                    != record["white_minus_black_halfpoints"]
                or record["candidate_profile"] != next(p for p in schedule["arm_profiles"]
                                                        if p["id"] == record["slot"]["arm"])
                or record["opponent_profile"] != next(p for p in schedule["profiles"]
                                                       if p["id"] == record["slot"]["profile_id"])
                or record["limits"] != schedule["limits"]
                or record["context"] != schedule["context"]
                or record["referee_profile"] != schedule["referee_profile"]):
            raise ValueError("Saved result or state does not match authoritative replay")
        _validate_observations(game, record, authorized_hashes)
        records[path.stem] = record
    return schedule, records
