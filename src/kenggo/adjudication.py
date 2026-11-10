"""Exact scoring for a small proved subset or an authorized neutral decision."""

from dataclasses import dataclass, replace
import json
import math
import time

from kenggo.go import (RULES, PASS, IllegalMove, Outcome, canonical, components, digest,
                       liberties, neighbors, opponent)


@dataclass(frozen=True)
class Limits:
    nodes: int = 100_000
    cpu_seconds: float = 2.0
    wall_seconds: float = 10.0

    def __post_init__(self):
        if (type(self.nodes) is not int or not 0 <= self.nodes <= 100_000
                or not 0 <= self.cpu_seconds <= 2 or not 0 <= self.wall_seconds <= 10):
            raise ValueError("Adjudication limits must fit the contract budget")


class BudgetExceeded(ValueError):
    pass


class Meter:
    def __init__(self, limits):
        self.limits = limits
        self.nodes = 0
        self.cpu = time.process_time()
        self.wall = time.monotonic()

    def tick(self, count=1):
        self.nodes += count
        if (self.nodes > self.limits.nodes
                or time.process_time() - self.cpu >= self.limits.cpu_seconds
                or time.monotonic() - self.wall >= self.limits.wall_seconds):
            raise BudgetExceeded("proof_budget_exhausted")

    def record(self):
        return {"node_limit": self.limits.nodes, "cpu_limit": self.limits.cpu_seconds,
                "wall_limit": self.limits.wall_seconds, "nodes": self.nodes,
                "cpu_seconds": time.process_time() - self.cpu,
                "wall_seconds": time.monotonic() - self.wall, "gpu_seconds": 0}


@dataclass(frozen=True)
class Decision:
    method: str
    reason: str
    snapshot: str
    white_minus_black_halfpoints: int | None
    payload: str | None
    cost: str


def singleton_eyes(position, meter):
    """Every chain has two private singleton liberties; no other empties exist."""
    meter.tick(PASS)
    if position.ko is not None:
        return None
    board = position.board
    groups = components(board, "BW")
    if not groups:
        return None
    owner = {point: i for i, group in enumerate(groups) for point in group}
    eyes = [set() for _ in groups]
    for region in components(board, "."):
        meter.tick(len(region))
        if len(region) != 1:
            return None
        point = next(iter(region))
        bordering = {owner[n] for n in neighbors(point) if n in owner}
        if len(bordering) != 1:
            return None
        eyes[next(iter(bordering))].add(point)
    if any(len(points) < 2 for points in eyes):
        return None
    black = sorted(p for group, points in zip(groups, eyes)
                   if board[next(iter(group))] == "B" for p in points)
    white = sorted(p for group, points in zip(groups, eyes)
                   if board[next(iter(group))] == "W" for p in points)
    return {"alive": [sorted(g) for g in groups], "dead": [], "seki": [],
            "territory_black": black, "territory_white": white, "neutral": []}


def _points(value):
    if (not isinstance(value, list) or len(value) > PASS
            or any(type(p) is not int or not 0 <= p < PASS for p in value)
            or len(set(value)) != len(value)):
        raise ValueError("Invalid or repeated points")
    return frozenset(value)


def score_map(position, mapping, meter):
    """Check complete geometry and exact arithmetic; status truth needs authority."""
    if not isinstance(mapping, dict) or set(mapping) != {
            "alive", "dead", "seki", "territory_black", "territory_white", "neutral"}:
        raise ValueError("Incomplete classification")
    meter.tick(PASS)
    board = position.board
    remaining = set(components(board, "BW"))
    statuses = {}
    for status in ("alive", "dead", "seki"):
        if not isinstance(mapping[status], list) or len(mapping[status]) > PASS:
            raise ValueError("Invalid chain list")
        for raw in mapping[status]:
            chain = _points(raw)
            if chain not in remaining:
                raise ValueError("Overlapping, partial or nonexistent chain")
            remaining.remove(chain)
            statuses.update((point, status) for point in chain)
    if remaining:
        raise ValueError("Unclassified chain")
    dead_black = sum(status == "dead" and board[p] == "B" for p, status in statuses.items())
    dead_white = sum(status == "dead" and board[p] == "W" for p, status in statuses.items())
    view = tuple("." if statuses.get(p) == "dead" else color for p, color in enumerate(board))
    point_owner = {}
    for color, field in (("B", "territory_black"), ("W", "territory_white"), (".", "neutral")):
        for point in _points(mapping[field]):
            if view[point] != "." or point in point_owner:
                raise ValueError("Overlapping territory or living-stone territory")
            point_owner[point] = color
    if set(point_owner) != {p for p, c in enumerate(view) if c == "."}:
        raise ValueError("Unclassified empty point")
    for region in components(view, "."):
        meter.tick(len(region))
        assigned = {point_owner[p] for p in region}
        if len(assigned) != 1:
            raise ValueError("Inconsistent empty-region classification")
        color = next(iter(assigned))
        if color != ".":
            boundary = {n for p in region for n in neighbors(p) if view[n] != "."}
            if not boundary or {view[p] for p in boundary} != {color}:
                raise ValueError("Territory does not have one-color boundary")
            if any(statuses[p] == "seki" for p in boundary):
                raise ValueError("Seki-adjacent points cannot count as territory")
    return (2 * (len(mapping["territory_white"]) + position.captured_black + dead_black
                 - len(mapping["territory_black"]) - position.captured_white - dead_white) + 13)


def _neutral_payload(game, payload, authorized_hashes):
    if not isinstance(payload, dict) or len(canonical(payload)) > 128_000:
        raise ValueError("Invalid neutral payload")
    if digest(payload) not in authorized_hashes:
        raise ValueError("Neutral decision is not independently authorized")
    if (set(payload) != {"method", "rules", "snapshot", "referee", "provenance", "budget", "map"}
            or payload["method"] != "neutral_map_v1" or payload["rules"] != RULES
            or payload["snapshot"] != game.snapshot
            or not isinstance(payload["referee"], str) or not payload["referee"]
            or not isinstance(payload["provenance"], str) or not payload["provenance"]):
        raise ValueError("Neutral decision lacks exact state/rules/provenance")
    budget = payload["budget"]
    if not isinstance(budget, dict) or set(budget) != {
            "node_limit", "cpu_limit", "wall_limit", "nodes", "cpu_seconds", "wall_seconds", "gpu_seconds"}:
        raise ValueError("Neutral decision lacks a recorded budget")
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in budget.values()):
        raise ValueError("Invalid neutral budget")
    if (type(budget["nodes"]) is not int or type(budget["node_limit"]) is not int
            or budget["nodes"] > budget["node_limit"]
            or budget["cpu_seconds"] > budget["cpu_limit"]
            or budget["wall_seconds"] > budget["wall_limit"] or budget["gpu_seconds"] != 0):
        raise ValueError("Neutral adjudication exceeded its recorded budget")
    return payload["map"]


def adjudicate(game, *, neutral=None, authorized_hashes=frozenset(), limits=Limits()):
    if game.status != "pending_adjudication":
        raise ValueError("Only a frozen two-pass ending can be adjudicated")
    meter = Meter(limits)
    method = "singleton_eyes_v1" if neutral is None else "neutral_map_v1"
    reason, score, payload = "unsupported_position", None, None
    try:
        meter.tick()
        if neutral is None:
            mapping = singleton_eyes(game.position, meter)
        else:
            mapping = _neutral_payload(game, neutral, authorized_hashes)
        if mapping is not None:
            score = score_map(game.position, mapping, meter)
            meter.tick()
            payload = canonical(neutral if neutral is not None else mapping)
            reason = "exact_score"
    except ValueError as exc:
        reason, score, payload = str(exc), None, None
    decision = Decision(method, reason, game.snapshot, score, payload, canonical(meter.record()))
    evidence = canonical({"method": method, "reason": reason, "snapshot": game.snapshot,
                          "payload": payload, "cost": json.loads(decision.cost)})
    outcome = (Outcome("unresolved", reason, evidence=evidence) if score is None else
               Outcome("score", method, "W" if score > 0 else "B", score, evidence))
    return replace(game, status="unresolved" if score is None else "resolved", outcome=outcome), decision


@dataclass(frozen=True)
class PassJudgment:
    verdict: str
    reason: str
    snapshot: str
    alternative_gain_upper_halfpoints: int | None
    cost: str
    witness: str | None = None


def _settled_score(game, meter):
    if game.status != "playing" or game.pending_cap:
        return None
    mapping = singleton_eyes(game.position, meter)
    return None if mapping is None else score_map(game.position, mapping, meter)


def _atari_liberties(position, color):
    return sorted({p for group in components(position.board, color)
                   if len(points := liberties(position.board, group)) == 1 for p in points})


def judge_pass(game, *, limits=Limits()):
    if game.status != "playing":
        raise ValueError("Pass judgment requires a pre-action playing state")
    meter = Meter(limits)
    verdict, reason, bound = "unjudgeable", "unsupported_position", None
    witness = None
    try:
        if not game.pending_cap and singleton_eyes(game.position, meter) is not None:
            # An opponent that only passes keeps both eyes of every original chain.
            # The mover can only consume its own territory; it cannot force a gain.
            verdict, reason, bound = "acceptable", "singleton_eye_dominance_v1", 0
        elif game.position.pass_streak == 0 and not game.pending_cap:
            color = game.position.to_play
            sign = -1 if color == "B" else 1
            passed = game.play(color, PASS)
            if passed.status == "playing":
                alternatives, replies = [], []
                for action in _atari_liberties(game.position, color):
                    try:
                        alternative = game.play(color, action)
                        reply = passed.play(opponent(color), action)
                    except IllegalMove:
                        continue
                    lower = _settled_score(alternative, meter)
                    upper = _settled_score(reply, meter)
                    if lower is not None:
                        alternatives.append((sign * lower, action, alternative.snapshot))
                    if upper is not None:
                        replies.append((sign * upper, action, reply.snapshot))
                if alternatives and replies:
                    best, response = max(alternatives), min(replies)
                    if best[0] - response[0] >= 2:
                        verdict, reason = "premature", "settled_atari_bounds_v1"
                        witness = canonical({"alternative_action": best[1], "reply_to_pass": response[1],
                                             "alternative_snapshot": best[2], "reply_snapshot": response[2],
                                             "alternative_lower_halfpoints": best[0],
                                             "pass_upper_halfpoints": response[0],
                                             "gain_lower_halfpoints": best[0] - response[0]})
        meter.tick()
    except BudgetExceeded as exc:
        verdict, reason, bound, witness = "unjudgeable", str(exc), None, None
    return PassJudgment(verdict, reason, game.snapshot, bound, canonical(meter.record()), witness)
