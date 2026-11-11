"""Canonical source evidence; admission and dataset partitions remain separate."""

from datetime import date
from hashlib import sha256
import json

from sgfmill import sgf, sgf_grammar

from kenggo.adjudication import adjudicate
from kenggo.go import PASS, RULES
from kenggo.kgs_eligibility import normalize_handle
from kenggo.kgs_source import player
from kenggo.rules_replay import replay_sgf
from kenggo.sgf_audit import finite_number, inspect_sgf, result_signature, scalar


VERSION = "canonical_kgs_v1"
CLOCKS = ("BL", "WL", "OB", "OW")


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return sha256(encoded(value).encode()).hexdigest()


def signature(result):
    kind, winner, margin = result_signature(result)
    return {"kind": kind, "winner": winner, "margin": None if margin is None else str(margin.normalize())}


def calendar_day(value):
    if not isinstance(value, str):
        raise ValueError("Source date is missing")
    day = date.fromisoformat(value)
    if day.isoformat() != value or day.year != 2025:
        raise ValueError("Expected an exact calendar day in the frozen 2025 scope")
    return day


def adjudication_evidence(game, decision):
    """Separate measured timings from the reproducible certificate and work counters."""
    cost = json.loads(decision.cost)
    evidence = {"status": game.status, "method": decision.method, "reason": decision.reason,
                "snapshot": decision.snapshot, "payload": decision.payload,
                "winner": game.outcome.winner,
                "white_minus_black_halfpoints": decision.white_minus_black_halfpoints,
                "cost": {k: v for k, v in cost.items() if k not in {"cpu_seconds", "wall_seconds"}}}
    return evidence, cost


def canonicalize(data, source_ref, archive_row):
    """Return deterministic source evidence and separate adjudication execution costs."""
    if len(data) > 1024**2:
        raise ValueError("Selected SGF exceeds 1 MiB")
    if sha256(data).hexdigest() != source_ref["sgf_sha256"]:
        raise ValueError("Source SGF hash changed")
    if source_ref["source_url"] != archive_row["url"]:
        raise ValueError("Archive evidence belongs to another source ID")
    audit = inspect_sgf(data, archive_row)
    record = {"version": VERSION, "rules_contract": RULES, "source": dict(source_ref),
              "archive": dict(archive_row), "audit": audit, "metadata": audit.get("metadata", {}),
              "date_precision": "day", "source_timezone": None,
              "actions": [], "clock_nodes": [], "replay_validated": False,
              "replay_end_status": None, "replay_pending_cap": None, "replay_snapshot": None,
              "certification": None, "structural_exclusions": [],
              "recorded_y_black": None, "recorded_value_supported": False,
              "duration_valid": False, "dynamic_clock_input_valid": False,
              "trajectory_sha256": None, "semantic_sha256": None}
    call_cost = None
    reasons = set(audit["exclusions"]) - {"archive_sgf_result_mismatch", "historical_bot_handle"}
    if audit["parse_status"] != "parsed":
        record["structural_exclusions"] = sorted(reasons)
        return record, call_cost
    metadata = audit["metadata"]
    try:
        calendar_day(metadata["DT"])
    except (ValueError, TypeError):
        reasons.add("date_unknown_or_out_of_scope")
    handles = {}
    for color, name, rank in (("black", "PB", "BR"), ("white", "PW", "WR")):
        try:
            handles[color] = normalize_handle(metadata[name])
            archived = player(archive_row[color])
            if (archived is not None and normalize_handle(archived[0]) == handles[color]
                    and archived[1] == metadata[rank]):
                reasons.discard(color + "_archive_sgf_mismatch")
        except ValueError:
            reasons.add(color + "_handle_unavailable")
    if len(handles) == 2 and handles["black"] == handles["white"]:
        reasons.add("same_handle_on_both_sides")
    recorded, archived = signature(metadata["RE"]), signature(archive_row.get("result"))
    record["recorded_result"], record["archive_result"] = recorded, archived
    if recorded["winner"] is not None and archived["winner"] is not None and recorded != archived:
        reasons.add("conflicting_known_results")
    if recorded["kind"] == "scored":
        margin = result_signature(metadata["RE"])[2] * 2
        if margin <= 0 or margin != margin.to_integral_value() or int(margin) % 2 != 1:
            reasons.add("score_margin_out_of_contract")
    game = sgf.Sgf_game.from_coarse_game_tree(sgf_grammar.parse_sgf_collection(data)[0])
    for index, node in enumerate(game.get_main_sequence()):
        values = {}
        for prop in CLOCKS:
            if node.has_property(prop):
                try:
                    value = finite_number(scalar(node, prop))
                except (ValueError, TypeError):
                    value = None
                values[prop] = {"raw_hex": [v.hex() for v in node.get_raw_list(prop)],
                                "value": value, "parse_valid": value is not None}
        if values:
            record["clock_nodes"].append({"node": index, "properties": values})
        color, point = node.get_move()
        if color is not None:
            record["actions"].append({"node": index, "color": color.upper(),
                                      "action": PASS if point is None else (18 - point[0]) * 19 + point[1],
                                      "raw_hex": node.get_raw(color.upper()).hex()})
    normalized_actions = [[m["color"], m["action"]] for m in record["actions"]]
    if normalized_actions:
        record["trajectory_sha256"] = digest({"size": metadata["SZ"], "rules": metadata["RU"],
                                              "komi": metadata["KM"], "actions": normalized_actions})
    semantics = {p: metadata[p] for p in ("DT", "BR", "WR", "SZ", "RU", "KM", "TM", "OT", "HA", "PL")}
    semantics.update(handles=handles, result=recorded, archive_result=archived,
                     actions=normalized_actions, clocks=record["clock_nodes"])
    record["semantic_sha256"] = digest(semantics)
    try:
        replay = replay_sgf(data)
        record.update(replay_validated=True, replay_end_status=replay.game.status,
                      replay_pending_cap=replay.game.pending_cap, replay_snapshot=replay.game.snapshot)
        if replay.game.status == "pending_adjudication":
            checked, evidence = adjudicate(replay.game)
            outcome = checked.outcome
            record["certification"], call_cost = adjudication_evidence(checked, evidence)
            if outcome.kind == "score" and recorded["winner"] is not None:
                conflict = outcome.winner != recorded["winner"]
                if recorded["kind"] == "scored":
                    conflict |= abs(outcome.white_minus_black_halfpoints) != result_signature(metadata["RE"])[2] * 2
                if conflict:
                    reasons.add("certified_result_conflict")
    except ValueError as error:
        reasons.add("invalid_or_unsupported_replay")
        record["replay_error"] = str(error)
    record["structural_exclusions"] = sorted(reasons)
    record["recorded_value_supported"] = (
        not reasons and recorded == archived and recorded["kind"] in {"scored", "resignation"})
    if record["recorded_value_supported"]:
        record["recorded_y_black"] = 1 if recorded["winner"] == "B" else -1
    return record, call_cost
