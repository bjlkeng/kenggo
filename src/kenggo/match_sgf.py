"""Match SGFs carry actual actions and an independently checked state record."""

import base64
import json

from sgfmill import sgf, sgf_grammar

from kenggo.go import PASS, RULES, canonical
from kenggo.rules_replay import dump_game, load_game


def _result(game):
    outcome = game.outcome
    if outcome is None or outcome.winner is None:
        return None
    if outcome.kind in ("score", "empirical_score"):
        points, half = divmod(abs(outcome.white_minus_black_halfpoints), 2)
        return f"{outcome.winner}+{points}{'.5' if half else ''}"
    return f"{outcome.winner}+{'R' if outcome.kind == 'resignation' else 'F'}"


def export_match_sgf(game, *, metadata=None):
    tree = sgf.Sgf_game(size=19, encoding="UTF-8")
    root = tree.get_root()
    root.set("AP", ("kenggo", "match_v1"))
    root.set("RU", "Japanese")
    root.set("KM", 6.5)
    root.set_raw("KGS", base64.b64encode(dump_game(game).encode("utf-8")))
    root.set_raw("KGM", base64.b64encode(canonical({"schema": "match_sgf_v1", "rules": RULES,
                                  "context": "untimed", "candidate_color": game.candidate_color, "metadata": metadata or {}}).encode()))
    if game.origin == "fixture":
        for color, field in (("B", "AB"), ("W", "AW")):
            points = {(18 - i // 19, i % 19) for i, stone in enumerate(game.positions[0].board)
                      if stone == color}
            if points:
                root.set(field, points)
        root.set("PL", game.positions[0].to_play.lower())
    result = _result(game)
    if result is not None:
        root.set("RE", result)
    for move in game.moves:
        node = tree.extend_main_sequence()
        if move.action == PASS:
            node.set_raw(move.color, b"")
        else:
            node.set_move(move.color.lower(), (18 - move.action // 19, move.action % 19))
    return tree.serialise()


def replay_match_sgf(data, *, authorized_hashes=frozenset(), empirical_referee=None,
                     allow_fixture=False, expected_metadata=None):
    if not isinstance(data, bytes) or len(data) > 16 * 1024**2:
        raise ValueError("Invalid match SGF size")
    trees = sgf_grammar.parse_sgf_collection(data)
    if len(trees) != 1:
        raise ValueError("Exactly one match SGF is required")
    tree = sgf.Sgf_game.from_coarse_game_tree(trees[0])
    root = tree.get_root()
    required = {"GM", "FF", "SZ", "CA", "AP", "RU", "KM", "KGS", "KGM"}
    if not required <= set(root.properties()) or set(root.properties()) - required - {"RE", "AB", "AW", "PL"}:
        raise ValueError("Unexpected match SGF properties or missing state")
    if (root.get("GM") != 1 or root.get("FF") != 4 or tree.get_size() != 19
            or root.get("RU") != "Japanese" or root.get("KM") != 6.5
            or root.get("AP") != ("kenggo", "match_v1") or root.get("CA") != "UTF-8"):
        raise ValueError("Match SGF rules or format mismatch")
    for prop in root.properties():
        if prop not in ("AB", "AW") and len(root.get_raw_list(prop)) != 1:
            raise ValueError("Repeated match SGF property")
    envelope = json.loads(base64.b64decode(root.get_raw("KGM"), validate=True))
    if (not isinstance(envelope, dict) or set(envelope) != {"schema", "rules", "context", "candidate_color", "metadata"}
            or envelope["schema"] != "match_sgf_v1" or envelope["rules"] != RULES
            or envelope["context"] != "untimed" or not isinstance(envelope["metadata"], dict)):
        raise ValueError("Match SGF envelope mismatch")
    if expected_metadata is not None and envelope["metadata"] != expected_metadata:
        raise ValueError("Match SGF metadata differs from the frozen schedule")
    game = load_game(base64.b64decode(root.get_raw("KGS"), validate=True).decode("utf-8"), allow_fixture=allow_fixture,
                     authorized_hashes=authorized_hashes, empirical_referee=empirical_referee)
    if envelope["candidate_color"] != game.candidate_color:
        raise ValueError("Match SGF candidate color differs from the full-history state")
    expected = sgf.Sgf_game.from_bytes(export_match_sgf(game, metadata=envelope["metadata"]))
    expected_root = expected.get_root()
    if set(root.properties()) != set(expected_root.properties()):
        raise ValueError("Match SGF outcome or setup mismatch")
    for prop in root.properties():
        if prop in ("KGS", "KGM"):
            continue
        if root.get(prop) != expected_root.get(prop):
            raise ValueError("Match SGF outcome, rules or setup mismatch")
    actions = []
    node = root
    while len(node):
        if len(node) != 1:
            raise ValueError("Match SGF cannot have variations")
        node = node[0]
        props = node.properties()
        if len(props) != 1 or props[0] not in ("B", "W") or len(node.get_raw_list(props[0])) != 1:
            raise ValueError("Match SGF nodes must contain exactly one actual action")
        color, point = node.get_move()
        actions.append([color.upper(), PASS if point is None else (18 - point[0]) * 19 + point[1]])
    if actions != [[move.color, move.action] for move in game.moves]:
        raise ValueError("Match SGF actions differ from the full-history state")
    return game
