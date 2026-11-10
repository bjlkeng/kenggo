"""Inspect source metadata and explicit actions without claiming rules replay."""

from collections import Counter
from decimal import Decimal
import hashlib
import json
import math
import re

from sgfmill import sgf, sgf_grammar

from kenggo.kgs_source import RANKS, player


ROOT_PROPERTIES = ("FF", "GM", "CA", "SZ", "AP", "PB", "PW", "BR", "WR", "RU", "KM",
                   "TM", "OT", "DT", "RE", "HA", "PL", "EV", "RO", "PC", "SO", "GN")


def result_kind(result):
    if result in {"0", "Draw", "Jigo"}:
        return "draw"
    if not result or result in {"?", "Void", "Unfinished"}:
        return "incomplete_or_unknown"
    match = re.fullmatch(r"[BW]\+(.+)", result)
    if match:
        suffix = match[1].lower()
        if suffix in {"r", "resign", "res."}:
            return "resignation"
        if suffix in {"t", "time"}:
            return "timeout"
        if suffix in {"f", "forfeit", "forf."}:
            return "forfeit"
        if re.fullmatch(r"\d+(?:\.\d+)?", suffix):
            return "scored"
    return "unrecognized"


def result_signature(result):
    kind = result_kind(result)
    winner = result[0] if kind in {"resignation", "timeout", "forfeit", "scored"} else None
    margin = Decimal(result[2:]) if kind == "scored" else None
    return kind, winner, margin


def finite_number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) and number >= 0 else None
    except (TypeError, ValueError):
        return None


def japanese_overtime(value):
    match = re.fullmatch(r"(\d+)x(\d+(?:\.\d+)?) byo-yomi", value or "")
    if not match:
        return None
    periods, seconds = int(match[1]), float(match[2])
    return (periods, seconds) if periods > 0 and seconds > 0 else None


def check_envelope(data: bytes):
    depth = 0
    in_value = escaped = False
    for char in data.removeprefix(b"\xef\xbb\xbf"):
        if in_value:
            if escaped:
                escaped = False
            elif char == ord("\\"):
                escaped = True
            elif char == ord("]"):
                in_value = False
        else:
            if depth == 0 and char not in b" \t\r\n\v\f(":
                raise ValueError("Non-SGF data outside a game tree")
            if char == ord("("):
                depth += 1
            elif char == ord(")"):
                depth -= 1
            elif char == ord("["):
                in_value = True
            if depth < 0:
                raise ValueError("Unbalanced SGF")
    if depth or in_value:
        raise ValueError("Unclosed SGF tree or value")


def scalar(node, name):
    if not node.has_property(name):
        return None
    if len(node.get_raw_list(name)) != 1:
        raise ValueError(f"Repeated scalar property {name}")
    value = node.get(name)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Nonfinite property {name}")
    return value


def inspect_sgf(data: bytes, archive_row: dict | None = None, bot_handles=()) -> dict:
    record = {"sgf_sha256": hashlib.sha256(data).hexdigest(), "sgf_bytes": len(data),
              "parse_status": "corrupt", "exclusions": [], "training_ready": False}
    try:
        check_envelope(data)
        trees = sgf_grammar.parse_sgf_collection(data)
        if len(trees) != 1:
            raise ValueError("Expected one game, found an SGF collection")
        game = sgf.Sgf_game.from_coarse_game_tree(trees[0])
        root = game.get_root()
        metadata = {name: scalar(root, name) for name in ROOT_PROPERTIES}
        # AP is a composed SGF value; JSON retains both components.
        nodes = game.get_main_sequence()
        counters = Counter()
        property_counts = Counter()
        moves = []
        overtime = japanese_overtime(metadata["OT"])
        main_time = finite_number(metadata["TM"])
        previous_clock = {"b": main_time, "w": main_time}
        setup = (any(any(node.has_property(p) for p in ("AB", "AW", "AE")) for node in nodes)
                 or any(node.has_property("PL") for node in nodes[1:]))
        variations = sum(len(node) > 1 for node in nodes)
        expected_color = "b"
        turn_errors = 0
        canonical_nodes = []
        for node in nodes:
            properties = node.get_raw_property_map()
            property_counts.update(properties.keys())
            canonical_nodes.append({p: [v.hex() for v in properties[p]] for p in
                                    ("B", "W", "AB", "AW", "AE", "PL") if p in properties})
            if node.has_property("B") and node.has_property("W"):
                raise ValueError("Two move colors in one node")
            for name in ("B", "W"):
                if node.has_property(name) and len(node.get_raw_list(name)) != 1:
                    raise ValueError("Multiple move values")
            color, move = node.get_move()
            counters["scoring_markup_nodes"] += any(p in properties for p in ("TB", "TW"))
            if color is None:
                continue
            turn_errors += color != expected_color
            expected_color = "w" if color == "b" else "b"
            raw_move = node.get_raw(color.upper())
            if move is None:
                counters["explicit_passes"] += 1
                counters["empty_passes" if raw_move == b"" else "legacy_passes"] += 1
            moves.append((color, move))
            counters["moves"] += 1
            clock_name = "BL" if color == "b" else "WL"
            periods_name = "OB" if color == "b" else "OW"
            clock = None
            if node.has_property(clock_name):
                try:
                    clock = finite_number(scalar(node, clock_name))
                except ValueError:
                    clock = None
                counters["post_clock_present"] += 1
                counters["post_clock_invalid"] += clock is None
            else:
                counters["post_clock_missing"] += 1
            period_marker = node.has_property(periods_name)
            counters["overtime_marker_moves"] += period_marker
            if period_marker:
                try:
                    marker_valid = finite_number(scalar(node, periods_name)) is not None
                except ValueError:
                    marker_valid = False
                counters["overtime_marker_invalid"] += not marker_valid
            previous = previous_clock[color]
            if clock is not None and previous is not None and clock > previous:
                counters["clock_increases"] += 1
            # These differences are audit candidates, not validated duration labels.
            if (overtime and not period_marker and previous is not None and clock is not None
                    and previous >= clock > overtime[1]):
                counters["main_time_difference_candidates"] += 1
            elif clock is not None:
                counters["ambiguous_clock_moves"] += 1
            previous_clock[color] = None if period_marker else clock
        exclusions = []
        if metadata["GM"] != 1 or metadata["SZ"] != 19:
            exclusions.append("game_or_board_metadata_out_of_scope")
        if setup or metadata["HA"] not in {None, 0} or metadata["PL"] not in {None, "b"}:
            exclusions.append("setup_or_handicap")
        if not moves or moves[0][0] != "b" or turn_errors:
            exclusions.append("missing_moves_or_turn_order")
        if variations:
            exclusions.append("played_variation_unverified")
        if metadata["RU"] != "Japanese" or metadata["KM"] != 6.5:
            exclusions.append("rules_or_komi_out_of_scope")
        if (main_time is None or not 600 <= main_time <= 1800 or not overtime
                or not 3 <= overtime[0] <= 5 or not 20 <= overtime[1] <= 60):
            exclusions.append("time_control_unknown_or_out_of_scope")
        if any(metadata[p] not in RANKS for p in ("BR", "WR")):
            exclusions.append("sgf_rank_unknown_or_out_of_scope")
        if metadata["EV"] or metadata["RO"]:
            exclusions.append("event_or_round_requires_classification")
        handles = [metadata["PB"], metadata["PW"]]
        historical_bot = any(isinstance(h, str) and h.lower() in {b.lower() for b in bot_handles} for h in handles)
        if historical_bot:
            exclusions.append("historical_bot_handle")
        if not all(isinstance(h, str) and re.fullmatch(r"[A-Za-z0-9]+", h) for h in handles):
            exclusions.append("account_handle_missing_or_ambiguous")
        if archive_row:
            for color, prop, rank_prop in (("white", "PW", "WR"), ("black", "PB", "BR")):
                source = player(archive_row[color])
                if source is None or metadata[prop] != source[0] or metadata[rank_prop] != source[1]:
                    exclusions.append(color + "_archive_sgf_mismatch")
            date_parts = archive_row["url"].split("/")[4:7]
            date = "-".join(f"{int(value):02d}" for value in date_parts)
            if metadata["DT"] != date:
                exclusions.append("archive_sgf_date_mismatch")
            if result_signature(metadata["RE"]) != result_signature(archive_row["result"]):
                exclusions.append("archive_sgf_result_mismatch")
        elif not isinstance(metadata["DT"], str) or not re.fullmatch(r"2025-\d\d-\d\d", metadata["DT"]):
            exclusions.append("date_unknown_or_out_of_scope")
        canonical = json.dumps(dict(size=metadata["SZ"], rules=metadata["RU"], komi=metadata["KM"],
                                    nodes=[node for node in canonical_nodes if node]), sort_keys=True)
        fields = ("moves", "explicit_passes", "empty_passes", "legacy_passes", "scoring_markup_nodes",
                  "post_clock_present", "post_clock_invalid", "post_clock_missing", "overtime_marker_moves",
                  "clock_increases", "main_time_difference_candidates", "ambiguous_clock_moves", "overtime_marker_invalid")
        record.update(parse_status="parsed", metadata=metadata, exclusions=sorted(set(exclusions)),
                      counters={key: counters[key] for key in fields}, result_kind=result_kind(metadata["RE"]),
                      final_two_explicit_passes=len(moves) >= 2 and all(move is None for _, move in moves[-2:]),
                      variations_on_mainline=variations, setup_indicated=setup, turn_order_errors=turn_errors,
                      human_status="historical_bot_handle" if historical_bot else "unverified",
                      stable_account_ids_verified=False, validated_elapsed_labels=0,
                      root_property_names=sorted(root.get_raw_property_map()), node_property_counts=dict(property_counts),
                      replay_validated=False,
                      actions_by_color={color: {"moves": sum(c == color for c, _ in moves),
                                                "explicit_passes": sum(c == color and m is None for c, m in moves)}
                                        for color in ("b", "w")},
                      sequence_fingerprint=hashlib.sha256(canonical.encode()).hexdigest())
    except (ValueError, TypeError, KeyError, UnicodeError, LookupError) as exc:
        record["error"] = str(exc)
        record["exclusions"] = ["corrupt_or_unsupported_sgf"]
    return record
