"""Strict source replay and history-based persistence of the shared engine."""

from dataclasses import dataclass, replace
import json

from sgfmill import sgf, sgf_grammar

from kenggo.adjudication import adjudicate
from kenggo.go import RULES, PASS, Game, Outcome, canonical, opponent
from kenggo.sgf_audit import check_envelope, inspect_sgf, result_signature


@dataclass(frozen=True)
class SourceReplay:
    game: Game
    source_sha256: str
    raw_moves: tuple[str, ...]
    source_result: str | None
    source_result_kind: str
    source_winner: str | None
    source_margin: str | None
    training_ready: bool = False


def replay_sgf(data):
    audit = inspect_sgf(data)
    strict_flags = {
        "game_or_board_metadata_out_of_scope", "setup_or_handicap",
        "missing_moves_or_turn_order", "played_variation_unverified",
        "rules_or_komi_out_of_scope", "time_control_unknown_or_out_of_scope",
    }
    if audit["parse_status"] != "parsed":
        raise ValueError("SGF syntax/metadata rejected")
    if strict_flags.intersection(audit["exclusions"]):
        raise ValueError("SGF outside the strict ordinary-play contract")
    check_envelope(data)
    source = sgf.Sgf_game.from_coarse_game_tree(sgf_grammar.parse_sgf_collection(data)[0])
    game, raw_moves = Game.new(), []
    for node in source.get_main_sequence():
        color, point = node.get_move()
        if color is None:
            continue
        raw_moves.append(node.get_raw(color.upper()).decode("ascii"))
        action = PASS if point is None else (18 - point[0]) * 19 + point[1]
        game = game.play(color.upper(), action)
    result = audit["metadata"]["RE"]
    kind, winner, margin = result_signature(result)
    return SourceReplay(game, audit["sgf_sha256"], tuple(raw_moves), result, kind,
                        winner, None if margin is None else str(margin))


def dump_game(game):
    record = game.action_record()
    record["ending"] = {"status": game.status, "pending_cap": game.pending_cap,
                        "outcome": None if game.outcome is None else {
                            "kind": game.outcome.kind, "reason": game.outcome.reason,
                            "winner": game.outcome.winner,
                            "white_minus_black_halfpoints": game.outcome.white_minus_black_halfpoints,
                            "evidence": game.outcome.evidence}}
    return canonical(record)


def load_game(text, *, allow_fixture=False, authorized_hashes=frozenset(),
              empirical_referee=None):
    """Recompute board, ko, captures and recurrence; never trust serialized caches."""
    if not isinstance(text, str) or len(text) > 2_000_000:
        raise ValueError("Invalid state record size")
    record = json.loads(text)
    if (not isinstance(record, dict) or set(record) != {
            "rules", "origin", "initial", "candidate_color", "moves", "ending"}
            or record["rules"] != RULES):
        raise ValueError("Invalid state record or rules version")
    initial = record["initial"]
    if not isinstance(initial, dict) or set(initial) != {
            "board", "to_play", "captured_black", "captured_white"}:
        raise ValueError("Invalid initial state")
    if record["origin"] == "empty":
        game = Game.new(candidate_color=record["candidate_color"])
        if canonical(initial) != canonical(game.action_record()["initial"]):
            raise ValueError("Ordinary games must start empty with Black")
    elif record["origin"] == "fixture" and allow_fixture:
        game = Game.from_fixture(initial["board"], to_play=initial["to_play"],
                                 captured_black=initial["captured_black"],
                                 captured_white=initial["captured_white"],
                                 candidate_color=record["candidate_color"])
    else:
        raise ValueError("Setup origin is not permitted for ordinary games")
    if not isinstance(record["moves"], list) or len(record["moves"]) > 20_000:
        raise ValueError("Invalid move history")
    for move in record["moves"]:
        if not isinstance(move, list) or len(move) != 2:
            raise ValueError("Invalid action record")
        game = game.play(*move)
    ending = record["ending"]
    if not isinstance(ending, dict) or set(ending) != {"status", "pending_cap", "outcome"}:
        raise ValueError("Invalid ending record")
    raw = ending["outcome"]
    if raw is not None:
        if not isinstance(raw, dict) or set(raw) != {
                "kind", "reason", "winner", "white_minus_black_halfpoints", "evidence"}:
            raise ValueError("Invalid outcome record")
        outcome = Outcome(**raw)
        if outcome.kind == "resignation":
            game = game.resign(game.position.to_play)
            if game.outcome != outcome:
                raise ValueError("Invalid resignation record")
        elif outcome.kind == "fault":
            if outcome.white_minus_black_halfpoints is not None or outcome.evidence is not None:
                raise ValueError("A fault is not a scored result")
            game = game.fail(outcome.reason, responsible=(
                None if outcome.winner is None else opponent(outcome.winner)))
        elif outcome.kind == "empirical_score":
            if empirical_referee is None:
                raise ValueError("An empirical outcome requires an independently configured referee")
            decision = empirical_referee.verify(game, outcome.evidence,
                                                authorized_hashes=authorized_hashes)
            if (decision.status != "resolved" or decision.snapshot != game.snapshot
                    or decision.reason != outcome.reason
                    or decision.white_minus_black_halfpoints != outcome.white_minus_black_halfpoints
                    or decision.evidence != outcome.evidence):
                raise ValueError("Empirical adjudication failed independent revalidation")
            game = replace(game, status="resolved", outcome=outcome)
        elif outcome.kind == "score":
            evidence = json.loads(outcome.evidence)
            if evidence["snapshot"] != game.snapshot:
                raise ValueError("Adjudication snapshot mismatch")
            if evidence["method"] == "singleton_eyes_v1":
                checked, decision = adjudicate(game)
            elif evidence["method"] == "neutral_map_v1":
                checked, decision = adjudicate(game, neutral=json.loads(evidence["payload"]),
                                               authorized_hashes=authorized_hashes)
            else:
                raise ValueError("Unknown scoring method")
            if (checked.outcome.kind != "score" or checked.outcome.reason != outcome.reason
                    or checked.outcome.winner != outcome.winner
                    or checked.outcome.white_minus_black_halfpoints != outcome.white_minus_black_halfpoints
                    or decision.payload != evidence["payload"]):
                raise ValueError("Adjudication failed independent revalidation")
            game = replace(checked, outcome=outcome)
        elif outcome.kind == "unresolved":
            if (game.status != "pending_adjudication" or outcome.winner is not None
                    or outcome.white_minus_black_halfpoints is not None or not outcome.reason):
                raise ValueError("Invalid unresolved ending")
            game = replace(game, status="unresolved", outcome=outcome)
        else:
            raise ValueError("Unknown outcome kind")
    elif ending["status"] == "truncated" and game.status == "playing":
        game = game.truncate(ending["pending_cap"], candidate_color=game.candidate_color)
    if game.status != ending["status"] or game.pending_cap != ending["pending_cap"]:
        raise ValueError("Ending does not match reconstructed history")
    return game
