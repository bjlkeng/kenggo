"""Shared ordinary-play state for kgs_japanese_certified_v1."""

from dataclasses import dataclass, replace
import hashlib
import json

from sgfmill.boards import Board


RULES = "kgs_japanese_certified_v1"
SIZE = 19
PASS = SIZE * SIZE
ACTION_LIMIT = 800


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def opponent(color):
    if color not in ("B", "W"):
        raise ValueError("Color must be B or W")
    return "W" if color == "B" else "B"


def neighbors(point):
    row, col = divmod(point, SIZE)
    return tuple(r * SIZE + c for r, c in
                 ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))
                 if 0 <= r < SIZE and 0 <= c < SIZE)


def components(board, colors="BW."):
    remaining = {i for i, color in enumerate(board) if color in colors}
    groups = []
    while remaining:
        first = min(remaining)
        group, queue = {first}, [first]
        remaining.remove(first)
        while queue:
            for point in neighbors(queue.pop()):
                if point in remaining and board[point] == board[first]:
                    remaining.remove(point)
                    group.add(point)
                    queue.append(point)
        groups.append(frozenset(group))
    return tuple(groups)


def liberties(board, group):
    return frozenset(n for p in group for n in neighbors(p) if board[n] == ".")


class IllegalMove(ValueError):
    pass


@dataclass(frozen=True)
class Move:
    color: str
    action: int


@dataclass(frozen=True)
class Position:
    board: tuple[str, ...]
    to_play: str
    captured_black: int = 0
    captured_white: int = 0
    ko: int | None = None
    pass_streak: int = 0

    @property
    def key(self):
        return self.board, self.to_play

    @property
    def rows(self):
        return tuple("".join(self.board[i:i + SIZE]) for i in range(0, PASS, SIZE))

    def kernel(self):
        board = Board(SIZE)
        black = [divmod(i, SIZE) for i, color in enumerate(self.board) if color == "B"]
        white = [divmod(i, SIZE) for i, color in enumerate(self.board) if color == "W"]
        if not board.apply_setup(black, white, []):
            raise ValueError("Zero-liberty setup is unsupported; never repair it")
        return board

    def place(self, action):
        if type(action) is not int or not 0 <= action < PASS:
            raise IllegalMove("Intersection outside board")
        if self.board[action] != ".":
            raise IllegalMove("Occupied point")
        if action == self.ko:
            raise IllegalMove("Simple ko")
        scratch = self.kernel()
        row, col = divmod(action, SIZE)
        ko = scratch.play(row, col, self.to_play.lower())
        if scratch.get(row, col) != self.to_play.lower():
            raise IllegalMove("Suicide")
        cells = tuple((scratch.get(r, c) or ".").upper()
                      for r in range(SIZE) for c in range(SIZE))
        removed_black = sum(a == "B" and b == "." for a, b in zip(self.board, cells))
        removed_white = sum(a == "W" and b == "." for a, b in zip(self.board, cells))
        return Position(cells, opponent(self.to_play),
                        self.captured_black + removed_black,
                        self.captured_white + removed_white,
                        None if ko is None else ko[0] * SIZE + ko[1])


@dataclass(frozen=True)
class Outcome:
    kind: str
    reason: str
    winner: str | None = None
    white_minus_black_halfpoints: int | None = None
    evidence: str | None = None

    def __post_init__(self):
        if (self.kind not in ("score", "resignation", "fault", "unresolved")
                or not isinstance(self.reason, str) or not self.reason
                or self.winner not in (None, "B", "W")
                or (self.evidence is not None and not isinstance(self.evidence, str))):
            raise ValueError("Invalid outcome fields")
        if self.kind == "score":
            score = self.white_minus_black_halfpoints
            if (type(score) is not int or score % 2 != 1
                    or self.winner != ("W" if score > 0 else "B") or not self.evidence):
                raise ValueError("A score requires exact odd integer half-points and evidence")
        elif self.white_minus_black_halfpoints is not None:
            raise ValueError("Only scored outcomes have a territory margin")
        if self.kind == "unresolved" and (self.winner is not None or not self.evidence):
            raise ValueError("An unresolved outcome has evidence but no winner")
        if self.kind == "resignation" and self.winner is None:
            raise ValueError("Resignation requires a winner")
        if self.kind in ("resignation", "fault") and self.evidence is not None:
            raise ValueError("A control event has no adjudication evidence")

    def value(self, color):
        opponent(color)
        if self.kind not in ("score", "resignation"):
            return None
        return 1 if self.winner == color else -1


@dataclass(frozen=True)
class Game:
    positions: tuple[Position, ...]
    moves: tuple[Move, ...] = ()
    origin: str = "empty"
    candidate_color: str | None = None
    status: str = "playing"
    pending_cap: str | None = None
    segment_start: int = 0
    outcome: Outcome | None = None

    @classmethod
    def new(cls, *, candidate_color=None):
        if candidate_color is not None:
            opponent(candidate_color)
        return cls((Position((".",) * PASS, "B"),), candidate_color=candidate_color)

    @classmethod
    def from_fixture(cls, rows, *, to_play="B", captured_black=0, captured_white=0,
                     candidate_color=None):
        opponent(to_play)
        if candidate_color is not None:
            opponent(candidate_color)
        if (len(rows) != SIZE or any(not isinstance(r, str) or len(r) != SIZE for r in rows)
                or set("".join(rows)) - set(".BW")):
            raise ValueError("Expected 19 rows of 19 .BW points")
        if any(type(n) is not int or n < 0 for n in (captured_black, captured_white)):
            raise ValueError("Captured counts must be nonnegative integers")
        initial = Position(tuple("".join(rows)), to_play, captured_black, captured_white)
        initial.kernel()
        return cls((initial,), origin="fixture", candidate_color=candidate_color)

    @property
    def position(self):
        return self.positions[-1]

    def clone(self):
        # Every shared descendant is immutable; branches cannot change their parent.
        return replace(self)

    def recent_moves(self, count):
        if type(count) is not int or count < 0:
            raise ValueError("History window must be a nonnegative integer")
        return self.moves[-count:] if count else ()

    def action_record(self):
        initial = self.positions[0]
        return {"rules": RULES, "origin": self.origin,
                "initial": {"board": initial.rows, "to_play": initial.to_play,
                            "captured_black": initial.captured_black,
                            "captured_white": initial.captured_white},
                "candidate_color": self.candidate_color,
                "moves": [[m.color, m.action] for m in self.moves]}

    @property
    def snapshot(self):
        return digest(self.action_record())

    def play(self, color, action):
        if self.status != "playing":
            raise IllegalMove(f"Cannot play in {self.status}")
        if color != self.position.to_play:
            raise IllegalMove("Wrong player")
        if type(action) is not int or not 0 <= action <= PASS:
            raise IllegalMove("Action must be an integer in [0, 361]")
        current = self.position
        n = len(self.moves) + 1
        pending = self.pending_cap
        if action == PASS:
            following = replace(current, to_play=opponent(color), ko=None,
                                pass_streak=current.pass_streak + 1)
            segment_start = n
        else:
            following = current.place(action)
            segment_start = self.segment_start
            if any(p.key == following.key for p in self.positions[segment_start:]):
                pending = pending or "recurrence"
        if n >= ACTION_LIMIT:
            pending = pending or "action_limit"
        status = "pending_adjudication" if following.pass_streak == 2 else "playing"
        if status == "playing" and pending and following.to_play == self.candidate_color:
            status = "truncated"
        return replace(self, positions=self.positions + (following,),
                       moves=self.moves + (Move(color, action),), status=status,
                       pending_cap=pending, segment_start=segment_start)

    def legal_mask(self):
        if self.status != "playing":
            return (False,) * (PASS + 1)
        mask = []
        for action in range(PASS):
            try:
                self.position.place(action)
                mask.append(True)
            except IllegalMove:
                mask.append(False)
        return tuple(mask) + (True,)

    def resign(self, color):
        if self.status != "playing" or color != self.position.to_play:
            raise IllegalMove("Resignation requires the current player in playing")
        return replace(self, status="resolved", outcome=Outcome(
            "resignation", "explicit_resignation", opponent(color)))

    def fail(self, reason, *, responsible=None):
        if (self.status not in ("playing", "pending_adjudication")
                or not isinstance(reason, str) or not reason):
            raise ValueError("Fault requires an active game and a reason")
        winner = None if responsible is None else opponent(responsible)
        return replace(self, status="invalid", outcome=Outcome("fault", reason, winner))

    def truncate(self, reason, *, candidate_color):
        opponent(candidate_color)
        if (self.status != "playing" or self.position.to_play != candidate_color
                or self.candidate_color != candidate_color
                or not isinstance(reason, str) or not reason):
            raise ValueError("Truncation requires a saved candidate-to-move observation")
        return replace(self, status="truncated", pending_cap=reason)

    def boundary(self, learner):
        opponent(learner)
        if self.candidate_color is not None and learner != self.candidate_color:
            raise ValueError("Learning boundary must use the configured fixed learner")
        if self.status in ("pending_adjudication", "unresolved", "invalid"):
            return LearningBoundary(False, None, None, None, None)
        if self.status == "resolved":
            return LearningBoundary(True, self.outcome.value(learner), True, False, None)
        if self.position.to_play != learner:
            raise ValueError("Bootstrap observation must be learner-to-move")
        return LearningBoundary(True, 0, False, self.status == "truncated", self)

    def match_result(self, candidate):
        opponent(candidate)
        if self.outcome and self.outcome.winner:
            return "W" if self.outcome.winner == candidate else "L"
        return "U"


@dataclass(frozen=True)
class LearningBoundary:
    learning_valid: bool
    reward: int | None
    terminated: bool | None
    truncated: bool | None
    bootstrap: Game | None


def require_valid_batch(boundaries):
    batch = tuple(boundaries)
    if not batch or any(not item.learning_valid for item in batch):
        raise ValueError("Invalid collection batch; retain trajectories, do not update")
    return batch


def move_category(game, action):
    following = game.play(game.position.to_play, action)
    if action == PASS:
        return "pass"
    before, after = game.position, following.position
    if ((after.captured_black, after.captured_white)
            != (before.captured_black, before.captured_white)):
        return "capture"
    color = before.to_play
    own_after = next(g for g in components(after.board, color) if action in g)
    if len(liberties(after.board, own_after)) > 1 and any(
            len(liberties(before.board, g)) == 1 and g <= own_after
            for g in components(before.board, color)):
        return "save_atari"
    if any(len(liberties(after.board, g)) == 1 and action in
           {n for p in g for n in neighbors(p)} for g in components(after.board, opponent(color))):
        return "give_atari"
    if len(liberties(after.board, own_after)) == 1:
        return "self_atari"
    return "other"
