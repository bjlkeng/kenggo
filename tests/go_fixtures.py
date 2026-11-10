"""Small 19x19 fixtures, with top-left row-major action coordinates."""

from kenggo.go import PASS, RULES, components


def point(row, col):
    return row * 19 + col


def rows(black=(), white=(), fill="."):
    board = [fill] * PASS
    for color, stones in (("B", black), ("W", white)):
        for r, c in stones:
            if board[point(r, c)] != fill:
                raise ValueError("Overlapping fixture setup")
            board[point(r, c)] = color
    return ["".join(board[i:i + 19]) for i in range(0, PASS, 19)]


def dense(kind="settled"):
    board = list("BBBBBBBBBWWWWWWWWWW" * 19)
    empty = [(3, 3), (5, 3), (3, 14), (5, 14)]
    if kind == "dead":
        board[point(9, 4)] = board[point(9, 5)] = "W"
        empty.append((9, 6))
    elif kind == "seki":
        empty = [(3, 3), (3, 14), (9, 8)]
    elif kind == "save":
        board[point(9, 10)] = "B"
        empty.append((9, 9))
    for r, c in empty:
        board[point(r, c)] = "."
    return ["".join(board[i:i + 19]) for i in range(0, PASS, 19)]


def ko_patches(triple=False):
    black, white = set(), set()
    patches = [(3, 3, "W"), (3, 10, "B"), (10, 3, "W")] if triple else [(3, 3, "W")]
    for r, c, victim in patches:
        v = {(r, c), (r - 1, c + 1), (r + 1, c + 1), (r, c + 2)}
        a = {(r - 1, c), (r + 1, c), (r, c - 1)}
        black.update(v if victim == "B" else a)
        white.update(a if victim == "B" else v)
    return rows(black, white)


CYCLE = [point(3, 4), point(3, 11), point(10, 4), point(3, 3), point(3, 10), point(10, 3)]


def ordinary_cases():
    capture_black = {(7, 9), (8, 8), (8, 10), (11, 9), (10, 8), (10, 10), (9, 7), (9, 11)}
    capture_white = {(8, 9), (10, 9), (9, 8), (9, 10)}
    return [
        {"name": "capture_before_suicide", "board": rows(capture_black, capture_white),
         "moves": [("B", point(9, 9))]},
        {"name": "single_suicide", "board": rows(white=capture_white),
         "moves": [("B", point(9, 9))]},
        {"name": "multiple_suicide", "board": rows({(9, 8)},
         {(8, 8), (10, 8), (9, 7), (8, 9), (10, 9), (9, 10)}),
         "moves": [("B", point(9, 9))]},
        {"name": "ko_exchange", "board": ko_patches(), "moves": [
            ("B", point(3, 4)), ("W", point(3, 3)), ("W", point(15, 15)),
            ("B", point(16, 15)), ("W", point(3, 3))]},
        {"name": "ko_cleared_by_pass", "board": ko_patches(), "moves": [
            ("B", point(3, 4)), ("W", PASS), ("B", point(16, 15)), ("W", point(3, 3))]},
        {"name": "triple_ko", "board": ko_patches(True),
         "moves": [("B" if i % 2 == 0 else "W", a) for i, a in enumerate(CYCLE * 3)]},
        {"name": "invalid_moves", "board": rows(), "moves": [
            ("W", 0), ("B", 0), ("W", 0), ("W", True), ("W", 1.0), ("W", -1), ("W", 362), ("W", 1)]},
        {"name": "two_passes", "board": rows(), "moves": [("B", PASS), ("W", PASS)]},
    ]


def two_passes(game):
    return game.play(game.position.to_play, PASS).play(
        "W" if game.position.to_play == "B" else "B", PASS)


def neutral_decision(game, kind):
    board = game.position.board
    groups = components(board, "BW")
    dead = {point(9, 4), point(9, 5)} if kind == "dead" else set()
    empties = {i for i, c in enumerate(board) if c == "."} | dead
    mapping = {"alive": [sorted(g) for g in groups if not g & dead] if kind != "seki" else [],
               "dead": [sorted(dead)] if dead else [],
               "seki": [sorted(g) for g in groups] if kind == "seki" else [],
               "territory_black": sorted(p for p in empties if p % 19 < 9) if kind != "seki" else [],
               "territory_white": sorted(p for p in empties if p % 19 >= 9) if kind != "seki" else [],
               "neutral": sorted(empties) if kind == "seki" else []}
    return {"method": "neutral_map_v1", "rules": RULES, "snapshot": game.snapshot,
            "referee": "fixture-only-independent-analysis", "provenance": "tests/go_fixtures.py; not a production referee",
            "budget": {"node_limit": 100000, "cpu_limit": 2, "wall_limit": 10, "nodes": 361,
                       "cpu_seconds": 0, "wall_seconds": 0, "gpu_seconds": 0}, "map": mapping}


def cap_actions():
    black_threats = [point(r, c) for r in range(19) for c in range(9)
                     if not (1 <= r <= 5 and 1 <= c <= 6)]
    white_threats = [point(r, c) for r in range(19) for c in range(9, 19)]
    return [a for b, w in zip(black_threats, white_threats)
            for a in (point(3, 4), PASS, b, point(3, 3), PASS, w)]
