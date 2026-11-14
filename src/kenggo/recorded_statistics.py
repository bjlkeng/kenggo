"""Recorded-game metrics under the frozen E0 hierarchy.

``weighted_intercept_dyadic_cr0_v1`` uses global centering, fixed rank/side
weights, and the dyadic sandwich of Aronow, Samii and Assenova
(https://arxiv.org/abs/1312.3398). Games sharing either account may covary;
disjoint account pairs are assumed independent. Repeated unordered dyads are
subtracted as clusters, without a degrees-of-freedom correction.
"""

from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Integral, Real
from statistics import NormalDist


METHOD = "weighted_intercept_dyadic_cr0_v1"
POLICY_MASS_TOLERANCE = 0.0005
PHASES = ("1-60", "61-180", "181+")
CLOCK_BINS = ("10-<20", "20-30", "missing", "outside_10_30_minutes")
VALUE_EDGES = (-1., -.8, -.6, -.4, -.2, 0., .2, .4, .6, .8, 1.)


def _identifier(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _real(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    return float(value)


def _boolean(value, name):
    if hasattr(value, "item"):
        value = value.item()
    if type(value) is not bool:
        raise ValueError(f"{name} must be boolean")
    return value


def _legal_mask(legal):
    try:
        result = tuple(_boolean(item, "legal mask") for item in legal)
    except TypeError as error:
        raise ValueError("A legal mask needs 362 boolean entries") from error
    if len(result) != 362 or not any(result):
        raise ValueError("A legal mask needs 362 entries and at least one legal action")
    return result


def _value(value):
    value = _real(value, "value")
    if not math.isfinite(value) or not -1 <= value <= 1:
        raise ValueError("Values must be finite and within [-1, 1]")
    return value


@dataclass(frozen=True)
class Prediction:
    row_key: str
    log_policy: tuple[float, ...]
    value: float

    def __post_init__(self):
        _identifier(self.row_key, "row_key")
        try:
            logs = tuple(_real(item, "log probability") for item in self.log_policy)
        except TypeError as error:
            raise ValueError("A policy needs 362 log probabilities") from error
        if len(logs) != 362 or any(math.isnan(item) or item > 0 for item in logs):
            raise ValueError("Log probabilities must be nonpositive, finite or negative infinity")
        object.__setattr__(self, "log_policy", logs)
        object.__setattr__(self, "value", _value(self.value))


def _validate_prediction(prediction, legal):
    if any(log != -math.inf for log, allowed in zip(prediction.log_policy, legal) if not allowed):
        raise ValueError("Illegal actions must have negative-infinity log probability")
    mass = math.fsum(math.exp(log) for log, allowed in zip(prediction.log_policy, legal) if allowed)
    if not math.isclose(mass, 1., rel_tol=0, abs_tol=POLICY_MASS_TOLERANCE):
        raise ValueError("Legal probabilities must sum to one within the raw-policy rounding tolerance")


def prediction_from_logits(row_key, logits, value, legal):
    """Mask illegal actions and retain finite log losses even after exp underflows."""
    legal = _legal_mask(legal)
    logits = tuple(_real(item, "logit") for item in logits)
    if len(logits) != 362 or any(math.isnan(item) or item == math.inf for item in logits):
        raise ValueError("A policy needs 362 valid logits")
    if any(not math.isfinite(item) for item, allowed in zip(logits, legal) if allowed):
        raise ValueError("Legal learned logits must be finite")
    maximum = max(item for item, allowed in zip(logits, legal) if allowed)
    shifted = tuple(item - maximum if allowed else -math.inf for item, allowed in zip(logits, legal))
    normalizer = math.log(math.fsum(math.exp(item) for item in shifted))
    logs = tuple(item - normalizer for item in shifted)
    if any(not math.isfinite(item) for item, allowed in zip(logs, legal) if allowed):
        raise ValueError("Finite logits exceed the representable log-probability range")
    prediction = Prediction(row_key, logs, value)
    _validate_prediction(prediction, legal)
    return prediction


def prediction_from_raw_policy(row_key, policy, value, legal):
    """Preserve raw legal probabilities, including zeros and reported rounding."""
    legal = _legal_mask(legal)
    policy = tuple(_real(item, "probability") for item in policy)
    if len(policy) != 362 or any(not math.isfinite(item) for item in policy):
        raise ValueError("A raw policy needs 362 finite entries")
    if any(not 0 <= item <= 1 if allowed else item != -1 for item, allowed in zip(policy, legal)):
        raise ValueError("Raw legal probabilities must be in [0, 1]; illegal entries must be -1")
    logs = tuple(math.log(item) if allowed and item > 0 else -math.inf
                 for item, allowed in zip(policy, legal))
    prediction = Prediction(row_key, logs, value)
    _validate_prediction(prediction, legal)
    return prediction


def _clock(value):
    if value is None:
        return None
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError("nominal_clock must be a (main seconds, period seconds, periods) tuple or None")
    main, period = (_real(item, "clock seconds") for item in value[:2])
    periods = value[2]
    if (not math.isfinite(main) or not math.isfinite(period) or main < 0 or period <= 0
            or isinstance(periods, bool) or not isinstance(periods, Integral) or periods <= 0):
        raise ValueError("Invalid nominal clock")
    return (main, period, int(periods))


def _rows_and_ranks(rows, aggregate_ranks):
    if isinstance(aggregate_ranks, str):
        raise ValueError("aggregate_ranks must be a frozen sequence of ranks")
    ranks = tuple(_identifier(rank, "aggregate rank") for rank in aggregate_ranks)
    if not ranks or len(set(ranks)) != len(ranks):
        raise ValueError("aggregate_ranks must be nonempty and unique")
    required = {"row_key", "source_url", "mover", "group_mover", "group_opponent", "source_rank_mover",
                "source_rank_opponent", "ply", "nominal_clock", "legal", "action", "policy_valid",
                "value", "value_valid"}
    result, keys, positions, games = [], set(), set(), {}
    for source in rows:
        if not isinstance(source, Mapping) or not required <= source.keys():
            raise ValueError("Each evaluation row must carry metadata, legality, targets and masks")
        row = dict(source)
        for name in ("row_key", "source_url", "group_mover", "group_opponent", "source_rank_mover"):
            _identifier(row[name], name)
        if row["source_rank_opponent"] is not None:
            _identifier(row["source_rank_opponent"], "source_rank_opponent")
        if row["source_rank_mover"] not in ranks:
            raise ValueError("An observed mover rank is outside the frozen aggregate; do not silently filter rows")
        if row["mover"] not in ("B", "W") or row["group_mover"] == row["group_opponent"]:
            raise ValueError("Rows need a B/W mover and two distinct source accounts")
        if isinstance(row["ply"], bool) or not isinstance(row["ply"], Integral) or row["ply"] < 0:
            raise ValueError("ply must be a nonnegative integer")
        row["ply"] = int(row["ply"])
        position = (row["source_url"], row["ply"])
        if row["row_key"] in keys or position in positions:
            raise ValueError("Evaluation rows must have unique row keys and game positions")
        keys.add(row["row_key"])
        positions.add(position)
        row["legal"] = _legal_mask(row["legal"])
        row["policy_valid"] = _boolean(row["policy_valid"], "policy_valid")
        row["value_valid"] = _boolean(row["value_valid"], "value_valid")
        if row["policy_valid"]:
            action = row["action"]
            if (isinstance(action, bool) or not isinstance(action, Integral)
                    or not 0 <= action < 362 or not row["legal"][action]):
                raise ValueError("A policy-valid target must name a legal action")
            row["action"] = int(action)
        if row["value_valid"]:
            row["value"] = _value(row["value"])
        row["nominal_clock"] = _clock(row["nominal_clock"])
        players = ((row["group_mover"], row["source_rank_mover"]),
                   (row["group_opponent"], row["source_rank_opponent"]))
        context = (players if row["mover"] == "B" else players[::-1], row["nominal_clock"])
        if games.setdefault(row["source_url"], context) != context:
            raise ValueError("Account, rank or nominal-clock metadata changes within a game")
        result.append(row)
    return sorted(result, key=lambda row: row["row_key"]), ranks


def _predictions(rows, predictions):
    expected = {row["row_key"]: row for row in rows}
    result = {}
    for prediction in predictions:
        if not isinstance(prediction, Prediction):
            raise ValueError("Each model output must be a Prediction")
        key = prediction.row_key
        if key not in expected or key in result:
            raise ValueError("Model outputs contain an extra or duplicate row key")
        _validate_prediction(prediction, expected[key]["legal"])
        result[key] = prediction
    if set(result) != set(expected):
        raise ValueError("Model outputs are missing expected rows; evaluation failed")
    return result


def _number(value):
    if value is None or math.isnan(value):
        return {"status": "undefined", "value": None}
    if math.isinf(value):
        return {"status": "positive_infinity" if value > 0 else "negative_infinity", "value": None}
    return {"status": "finite", "value": value}


def _sum(values):
    values = tuple(values)
    if any(value is None or math.isnan(value) for value in values):
        return None
    if math.inf in values and -math.inf in values:
        return None
    try:
        return math.fsum(values)
    except OverflowError:
        return None


def _weighted_mean(values):
    """Weights sum to one; centering arithmetic preserves constant scores exactly."""
    values = tuple(values)
    if not values or any(value is None for _, value in values):
        return None
    if all(math.isfinite(value) for _, value in values):
        center = values[0][1]
        terms = [weight * (value - center) for weight, value in values]
        if all(math.isfinite(term) for term in terms):
            return _sum((center, _sum(terms)))
    return _sum(weight * value for weight, value in values)


def _counts(rows):
    mover_accounts = {row["group_mover"] for row in rows}
    sides = {(row["source_url"], row["mover"]) for row in rows}
    return {"moves": len(rows), "games": len({row["source_url"] for row in rows}), "game_sides": len(sides),
            "mover_accounts": len(mover_accounts),
            "accounts": len(mover_accounts | {row["group_opponent"] for row in rows}),
            "insufficient": len(mover_accounts) < 30 or len(sides) < 100}


def _sides(rows):
    result = defaultdict(list)
    for row in rows:
        result[(row["source_url"], row["mover"])].append(row)
    return dict(sorted(result.items()))


def _side_weights(sides, ranks):
    counts = Counter(side[0]["source_rank_mover"] for side in sides.values())
    missing = [rank for rank in ranks if not counts[rank]]
    weights = {} if missing else {key: 1 / (len(ranks) * counts[side[0]["source_rank_mover"]])
                                  for key, side in sides.items()}
    return weights, missing


def _uncertainty(sides, weights, means, mean):
    incident, mover = defaultdict(list), defaultdict(list)
    game_weights, game_terms, game_accounts = defaultdict(list), defaultdict(list), {}
    for key, side in sides.items():
        row, weight = side[0], weights[key]
        source = row["source_url"]
        pair = tuple(sorted((row["group_mover"], row["group_opponent"])))
        game_accounts[source] = pair
        game_weights[source].append(weight)
        mover[row["group_mover"]].append(weight)
        game_terms[source].append(None if mean is None or means[key] is None else weight * (means[key] - mean))
    for source, terms in game_weights.items():
        weight = math.fsum(terms)
        for account in game_accounts[source]:
            incident[account].append(weight)
    incident = {account: math.fsum(terms) for account, terms in sorted(incident.items())}
    mover = {account: math.fsum(terms) for account, terms in sorted(mover.items())}
    influences = {source: _sum(terms) for source, terms in sorted(game_terms.items())}
    variance = None
    if mean is not None and math.isfinite(mean) and all(
            value is not None and math.isfinite(value) for value in influences.values()):
        account_terms, pair_terms = defaultdict(list), defaultdict(list)
        for source, influence in influences.items():
            pair = game_accounts[source]
            for account in pair:
                account_terms[account].append(influence)
            pair_terms[pair].append(influence)
        account_sums = [_sum(terms) for terms in account_terms.values()]
        pair_sums = [_sum(terms) for terms in pair_terms.values()]
        if all(value is not None for value in account_sums + pair_sums):
            variance = _sum([value * value for value in account_sums]
                            + [-value * value for value in pair_sums])
    reasons = []
    if len(incident) < 30:
        reasons.append("fewer_than_30_accounts")
    if max(incident.values(), default=0) > .1:
        reasons.append("incident_account_weight_over_10_percent")
    if mean is None or not math.isfinite(mean):
        reasons.append("nonfinite_or_undefined_estimate")
    if variance is None or not math.isfinite(variance):
        reasons.append("nonfinite_or_undefined_variance")
    elif variance <= 0:
        reasons.append("nonpositive_variance")
    bounds = None
    if not reasons:
        radius = NormalDist().inv_cdf(.975) * math.sqrt(variance)
        endpoints = (mean - radius, mean + radius)
        if all(math.isfinite(value) for value in endpoints):
            bounds = list(endpoints)
        else:
            reasons.append("nonfinite_interval")
    return {"method": METHOD, "status": "inconclusive" if reasons else "available", "reasons": reasons,
            "variance": _number(variance), "interval95": bounds,
            "accounts": len(incident), "max_incident_account_weight": max(incident.values(), default=0),
            "incident_account_weights": incident, "mover_contribution_weights": mover,
            "max_mover_contribution_weight": max(mover.values(), default=0),
            "game_influences": {key: _number(value) for key, value in influences.items()}}


def _estimate(rows, losses, ranks, mask):
    eligible = [row for row in rows if row[mask]]
    sides = _sides(eligible)
    weights, missing = _side_weights(sides, ranks)
    means = {key: _weighted_mean((1 / len(side), losses[row["row_key"]]) for row in side)
             for key, side in sides.items()}
    mean = _weighted_mean((weights[key], means[key]) for key in sides) if weights else None
    account_counts = Counter(side[0]["group_mover"] for side in sides.values())
    account_mean = _weighted_mean((1 / (len(account_counts) * account_counts[side[0]["group_mover"]]), means[key])
                                  for key, side in sides.items()) if sides else None
    pooled = _weighted_mean((1 / len(eligible), losses[row["row_key"]]) for row in eligible) if eligible else None
    return {"status": "unavailable" if missing else "undefined" if mean is None else "available",
            "aggregate_ranks": list(ranks), "missing_ranks": missing, "estimate": _number(mean),
            "counts": _counts(eligible), "masked_moves": len(rows) - len(eligible),
            "excluded_game_sides_no_eligible_rows": len(_sides(rows)) - len(sides),
            "uncertainty": _uncertainty(sides, weights, means, mean) if weights else {
                "method": METHOD, "status": "inconclusive", "reasons": ["missing_aggregate_ranks"],
                "variance": _number(None), "interval95": None},
            "diagnostics": {"scope": "available_eligible_rows", "observed_ranks": sorted(
                {row["source_rank_mover"] for row in eligible}), "pooled_move": _number(pooled),
                "account_balanced": _number(account_mean)},
            "side_weights": [{"source_url": key[0], "mover": key[1], "rank": sides[key][0]["source_rank_mover"],
                              "eligible_moves": len(sides[key]), "weight": weight} for key, weight in weights.items()]}


def _calibration(rows, predictions, ranks):
    eligible = [row for row in rows if row["value_valid"]]
    sides = _sides(eligible)
    weights, missing = _side_weights(sides, ranks)
    row_weights = {row["row_key"]: weights[key] / len(side) for key, side in sides.items() for row in side} if weights else {}
    bins = [[] for _ in range(10)]
    for row in eligible:
        index = min(9, bisect_right(VALUE_EDGES, predictions[row["row_key"]].value) - 1)
        bins[index].append(row)
    result = []
    for index, members in enumerate(bins):
        mass = math.fsum(row_weights[row["row_key"]] for row in members) if weights else None
        mean_prediction = _weighted_mean((row_weights[row["row_key"]] / mass, predictions[row["row_key"]].value)
                                         for row in members) if mass else None
        mean_result = _weighted_mean((row_weights[row["row_key"]] / mass, row["value"])
                                     for row in members) if mass else None
        result.append({"lower": VALUE_EDGES[index], "upper": VALUE_EDGES[index + 1], "upper_inclusive": index == 9,
                       "raw_count": len(members), "counts": _counts(members), "weighted_mass": mass,
                       "mean_prediction": _number(mean_prediction), "mean_result": _number(mean_result)})
    return {"status": "unavailable" if missing else "available", "aggregate_ranks": list(ranks),
            "missing_ranks": missing, "weighting": "frozen_value_row_weights", "bins": result}


def _phase(row):
    return PHASES[0 if row["ply"] < 60 else 1 if row["ply"] < 180 else 2]


def _clock_bin(row):
    clock = row["nominal_clock"]
    if clock is None:
        return "missing"
    return "10-<20" if 600 <= clock[0] < 1200 else "20-30" if 1200 <= clock[0] <= 1800 else "outside_10_30_minutes"


def _losses(rows, predictions):
    result = {"nll": {}, "top1": {}, "value_mse": {}}
    for row in rows:
        key = row["row_key"]
        prediction = predictions[key]
        if row["policy_valid"]:
            result["nll"][key] = -prediction.log_policy[row["action"]]
            top = max(range(362), key=prediction.log_policy.__getitem__)
            result["top1"][key] = float(top == row["action"])
        if row["value_valid"]:
            result["value_mse"][key] = (prediction.value - row["value"]) ** 2
    return result


def _group_report(rows, losses, ranks, predictions):
    metrics = {name: _estimate(rows, values, ranks, "value_valid" if name == "value_mse" else "policy_valid")
               for name, values in losses.items()}
    return {"counts": _counts(rows), "metrics": metrics,
            "value_calibration": {name: _calibration(rows, outputs, ranks) for name, outputs in predictions.items()}}


def _report(rows, losses, ranks, predictions, kind):
    rank_pairs, clocks = defaultdict(list), defaultdict(list)
    for row in rows:
        rank_pairs[(row["source_rank_mover"], row["source_rank_opponent"])].append(row)
        clocks[row["nominal_clock"]].append(row)
    return {"version": "recorded_statistics_v1", "kind": kind, "aggregate_ranks": list(ranks),
            "uncertainty_method": METHOD, "top1_ties": "lowest_action_index", "expected_outputs": len(rows),
            "overall": _group_report(rows, losses, ranks, predictions),
            "by_rank": {rank: _group_report([row for row in rows if row["source_rank_mover"] == rank],
                                            losses, (rank,), predictions) for rank in ranks},
            "by_phase": {phase: _group_report([row for row in rows if _phase(row) == phase], losses, ranks, predictions)
                         for phase in PHASES},
            "by_clock": {clock: _group_report([row for row in rows if _clock_bin(row) == clock], losses, ranks, predictions)
                         for clock in CLOCK_BINS},
            "coverage": {
                "by_rank_pair": [{"mover_rank": pair[0], "opponent_rank": pair[1], "counts": _counts(members)}
                                 for pair, members in sorted(rank_pairs.items(), key=lambda item: str(item[0]))],
                "by_nominal_clock": [{"nominal_clock": list(clock) if clock is not None else None,
                                      "counts": _counts(members)}
                                     for clock, members in sorted(clocks.items(), key=lambda item: str(item[0]))]},
            "joint_cells": [{"mover_rank": rank, "phase": phase, "clock_bin": clock,
                             "counts": _counts([row for row in rows if row["source_rank_mover"] == rank
                                                and _phase(row) == phase and _clock_bin(row) == clock])}
                            for rank in ranks for phase in PHASES for clock in CLOCK_BINS]}


def evaluate(rows, predictions, *, aggregate_ranks):
    """Validate exact row coverage and score one frozen rank scope; never filter it."""
    rows, ranks = _rows_and_ranks(rows, aggregate_ranks)
    predictions = _predictions(rows, predictions)
    return _report(rows, _losses(rows, predictions), ranks, {"model": predictions}, "single_model")


def paired_report(rows, candidate, control, *, aggregate_ranks):
    """Candidate-minus-control per-move contrasts, then the common hierarchy."""
    rows, ranks = _rows_and_ranks(rows, aggregate_ranks)
    candidate, control = _predictions(rows, candidate), _predictions(rows, control)
    candidate_losses, control_losses = _losses(rows, candidate), _losses(rows, control)
    differences = {name: {key: _sum((value, -control_losses[name][key])) for key, value in values.items()}
                   for name, values in candidate_losses.items()}
    return _report(rows, differences, ranks, {"candidate": candidate, "control": control}, "candidate_minus_control")
