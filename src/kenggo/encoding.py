"""Pre-move inputs shared by supervised examples and inference."""

from dataclasses import dataclass
import math

import numpy as np

from kenggo.go import PASS, SIZE
from kenggo.kgs_source import RANKS


VERSION = "mover_encoding_v1"
SCALARS = ("mover_is_black", "mover_komi", "prisoners_for_mover", "prisoners_for_opponent",
           "pass_streak", "move_number", "cap_pending", "truncated", "cleanup_phase",
           "main_seconds", "period_seconds", "periods", "mover_clock", "opponent_clock",
           "mover_overtime", "opponent_overtime")
NORMALIZED = (2, 3, 5, 9, 10, 11)
INPUT_KEYS = ("board", "legal", "scalars", "scalar_valid", "ranks", "rank_valid",
              "history_actions", "history_colors", "history_valid")


@dataclass(frozen=True)
class Context:
    black_rank: str | None = None
    white_rank: str | None = None
    nominal_clock: tuple[float, float, int] | None = None

    def __post_init__(self):
        if any(rank is not None and rank not in RANKS for rank in (self.black_rank, self.white_rank)):
            raise ValueError("Rank must be an exact supported historical label or unknown")
        if self.nominal_clock is not None:
            clock = self.nominal_clock
            if (type(clock) is not tuple or len(clock) != 3
                    or any(type(n) not in (int, float) or not math.isfinite(n) for n in clock[:2])
                    or not 600 <= clock[0] <= 1800 or not 20 <= clock[1] <= 60
                    or type(clock[2]) is not int or not 3 <= clock[2] <= 5):
                raise ValueError("Nominal clocks must match the v1 main-time/byo-yomi family")


def array_schema(history):
    if type(history) is not int or history not in (0, 8, 32, 64):
        raise ValueError("Model history must be 0, 8, 32 or 64 actual moves")
    return {"board": ("bool", (3, SIZE, SIZE)), "legal": ("bool", (PASS + 1,)),
            "scalars": ("float32", (len(SCALARS),)), "scalar_valid": ("bool", (len(SCALARS),)),
            "ranks": ("int16", (2,)), "rank_valid": ("bool", (2,)),
            "history_actions": ("int16", (history,)), "history_colors": ("int8", (history,)),
            "history_valid": ("bool", (history,))}


def encode(game, context=Context(), *, history=0):
    """Color-relative planes without coordinate rotation; the engine keeps full history."""
    schema = array_schema(history)
    if game.status not in ("playing", "truncated"):
        raise ValueError("Only playable or saved truncation observations may be encoded")
    result = {name: np.zeros(shape, dtype=dtype) for name, (dtype, shape) in schema.items()}
    position = game.position
    black = position.to_play == "B"
    stones = np.array(position.board).reshape(SIZE, SIZE)
    result["board"][0] = stones == position.to_play
    result["board"][1] = (stones != ".") & (stones != position.to_play)
    if position.ko is not None:
        result["board"][2].flat[position.ko] = True
    result["legal"][:] = game.legal_mask()
    result["scalars"][:9] = [black, -6.5 if black else 6.5,
                             position.captured_white if black else position.captured_black,
                             position.captured_black if black else position.captured_white,
                             position.pass_streak, len(game.moves), game.pending_cap is not None,
                             game.status == "truncated", 0]
    result["scalar_valid"][:9] = True
    if context.nominal_clock is not None:
        result["scalars"][9:12] = context.nominal_clock
        result["scalar_valid"][9:12] = True
    ranks = (context.black_rank, context.white_rank) if black else (context.white_rank, context.black_rank)
    for index, rank in enumerate(ranks):
        if rank is not None:
            result["ranks"][index] = RANKS.index(rank) + 1
            result["rank_valid"][index] = True
    recent = game.recent_moves(history)
    result["history_actions"][:] = -1
    for index, move in enumerate(recent, start=history - len(recent)):
        result["history_actions"][index] = move.action
        result["history_colors"][index] = 1 if move.color == position.to_play else -1
        result["history_valid"][index] = True
    return result


def apply_normalizer(inputs, normalizer):
    if (set(inputs) != set(INPUT_KEYS) or normalizer["version"] != "scalar_normalizer_v1"
            or normalizer["encoding"] != VERSION or normalizer["scalars"] != list(SCALARS)
            or normalizer["fit_role"] != "train" or normalizer["fitted"] is not True):
        raise ValueError("An actually fitted training-only normalizer is required")
    result = {key: value.copy() for key, value in inputs.items()}
    mean, scale = np.asarray(normalizer["mean"]), np.asarray(normalizer["scale"])
    count = np.asarray(normalizer["count"])
    if (any(v.shape != (len(SCALARS),) for v in (mean, scale, count))
            or not np.isfinite(mean).all() or not np.isfinite(scale).all()
            or not np.issubdtype(count.dtype, np.integer)
            or (scale <= 0).any() or (count < 0).any()):
        raise ValueError("Invalid normalization statistics")
    for index in NORMALIZED:
        valid = result["scalar_valid"][..., index] & (count[index] > 0)
        result["scalar_valid"][..., index] = valid
        result["scalars"][..., index] = np.where(valid,
            (result["scalars"][..., index] - mean[index]) / scale[index], 0)
    return result
