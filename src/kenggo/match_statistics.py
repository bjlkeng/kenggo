"""Scheduled-game scores and uncertainty under the frozen E0 match protocol."""

from collections import Counter, defaultdict
from collections.abc import Mapping
import math

import numpy as np


COLORS = ("B", "W")


def _identifier(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _alpha(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < 1:
        raise ValueError("alpha must be between zero and one")
    return float(value)


def _bootstrap_settings(seed, resamples, alpha):
    alpha = _alpha(alpha)
    if type(seed) is not int or seed < 0:
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    if type(resamples) is not int or resamples < 10000 or resamples * alpha / 2 < 100:
        raise ValueError("Use at least 10,000 resamples and 100 resamples per requested tail")
    return alpha


def _validate_records(slots, records):
    if not isinstance(records, Mapping):
        raise ValueError("records must map scheduled slot IDs to records")
    if set(records) - set(slots):
        raise ValueError("Records contain unexpected slot IDs")
    for record in records.values():
        if not isinstance(record, Mapping):
            raise ValueError("Each record must be an object")
        if record.get("result") not in (None, "W", "L", "U"):
            raise ValueError("Results must be W, L or U; the current rules permit no draws")
        if "reason" in record and not isinstance(record["reason"], str):
            raise ValueError("A result reason must be a string")
        behavior = record.get("behavior", {})
        if not isinstance(behavior, Mapping) or any(
                not isinstance(key, str) or (value is not None and type(value) is not bool)
                for key, value in behavior.items()):
            raise ValueError("Behavior flags must map names to boolean or unknown values")


def _engine_schedule(schedule, records):
    if not isinstance(schedule, Mapping) or schedule.get("version") != "match_schedule_v1":
        raise ValueError("A fixed engine panel needs a match_schedule_v1 schedule")
    if schedule.get("purpose") not in ("validation", "confirmation", "final"):
        raise ValueError("The schedule must declare its evaluation purpose")
    if schedule.get("scope") not in ("setup", "full"):
        raise ValueError("The schedule must declare setup or full scope")
    arms = schedule.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ValueError("The schedule must declare at least one arm")
    for arm in arms:
        _identifier(arm, "arm")
    if len(set(arms)) != len(arms):
        raise ValueError("Arm IDs must be unique")
    arms = sorted(arms)
    control = schedule.get("control_arm")
    if control is not None and control not in arms:
        raise ValueError("control_arm must be a scheduled arm")

    declared = schedule.get("profiles")
    if not isinstance(declared, list) or not declared:
        raise ValueError("The schedule must declare its weighted profiles")
    profiles = {}
    for profile in declared:
        if not isinstance(profile, Mapping):
            raise ValueError("Profiles must be objects")
        profile_id = _identifier(profile.get("id"), "profile id")
        if profile_id in profiles:
            raise ValueError("Profile IDs must be unique; split contexts into distinct profiles")
        _identifier(profile.get("family"), "profile family")
        _identifier(profile.get("context_id"), "profile context_id")
        weight = profile.get("weight")
        if (isinstance(weight, bool) or not isinstance(weight, (int, float))
                or not math.isfinite(weight) or weight <= 0):
            raise ValueError("Profile weights must be positive and finite")
        profiles[profile_id] = profile
    total_weight = math.fsum(p["weight"] for p in profiles.values())
    if not math.isclose(total_weight, 1, rel_tol=0, abs_tol=1e-9):
        raise ValueError("Frozen profile weights must sum to one")
    profiles = {key: dict(value, weight=value["weight"] / total_weight)
                for key, value in sorted(profiles.items())}

    declared_slots = schedule.get("slots")
    if not isinstance(declared_slots, list) or not declared_slots:
        raise ValueError("The schedule must retain all scheduled game slots")
    slots, blocks = {}, defaultdict(dict)
    for slot in declared_slots:
        if not isinstance(slot, Mapping):
            raise ValueError("Slots must be objects")
        slot_id = _identifier(slot.get("id"), "slot id")
        block_id = _identifier(slot.get("block_id"), "block id")
        if slot_id in slots:
            raise ValueError("Slot IDs must be unique")
        profile_id, arm, color = slot.get("profile_id"), slot.get("arm"), slot.get("candidate_color")
        if profile_id not in profiles or arm not in arms or color not in COLORS:
            raise ValueError("Each slot must name a declared profile, arm and candidate color")
        if "context_id" in slot and slot["context_id"] != profiles[profile_id]["context_id"]:
            raise ValueError("Changed contexts need separately declared profile/context strata")
        for name in ("candidate_seed", "opponent_seed"):
            if type(slot.get(name)) is not int or slot[name] < 0:
                raise ValueError("Each agent seed must be a nonnegative integer")
        if slot["candidate_seed"] == slot["opponent_seed"]:
            raise ValueError("Candidate and opponent need distinct random streams")
        key = (arm, color)
        block = blocks[profile_id].setdefault(block_id, {})
        if key in block:
            raise ValueError("A block contains duplicate arm/color slots")
        block[key] = slot
        slots[slot_id] = slot

    expected = {(arm, color) for arm in arms for color in COLORS}
    for profile_id in profiles:
        if not blocks[profile_id]:
            raise ValueError("Every weighted profile needs scheduled blocks")
        seen = {color: set() for color in COLORS}
        for block in blocks[profile_id].values():
            if set(block) != expected:
                raise ValueError("Every block must schedule both colors for every arm")
            for color in COLORS:
                seeds = {(block[arm, color]["candidate_seed"], block[arm, color]["opponent_seed"])
                         for arm in arms}
                if len(seeds) != 1:
                    raise ValueError("Candidate/control arms must share each color's block seeds")
                seed_pair = seeds.pop()
                if seed_pair in seen[color]:
                    raise ValueError("Repeated random-input blocks are duplicates, not independent trials")
                seen[color].add(seed_pair)
    _validate_records(slots, records)
    return arms, control, profiles, slots, blocks


def _outcome(slot_id, records):
    return records.get(slot_id, {}).get("result") or "U"


def _counts(slot_ids, records):
    counts = {"N": len(slot_ids), "W": 0, "L": 0, "D": 0, "U": 0, "missing": 0}
    reasons = Counter()
    for slot_id in slot_ids:
        result = _outcome(slot_id, records)
        counts[result] += 1
        if slot_id not in records:
            counts["missing"] += 1
            reasons["missing_record"] += 1
        else:
            reasons[records[slot_id].get("reason", "unresolved" if result == "U" else "unspecified")] += 1
    return counts, dict(sorted(reasons.items()))


def _score_summary(groups, records):
    """Groups are (normalized weight, scheduled slot IDs), fixed before outcomes."""
    summaries = [(_counts(slot_ids, records), weight) for weight, slot_ids in groups]
    counts = {key: sum(summary[0][key] for summary, _ in summaries)
              for key in ("N", "W", "L", "D", "U", "missing")}
    reasons = Counter()
    lower, unknown, resolved = [], [], []
    for (group_counts, group_reasons), weight in summaries:
        reasons.update(group_reasons)
        lower.append(weight * group_counts["W"] / group_counts["N"])
        unknown.append(weight * group_counts["U"] / group_counts["N"])
        resolved.append(weight * (group_counts["N"] - group_counts["U"]) / group_counts["N"])
    lower, unknown = math.fsum(lower), math.fsum(unknown)
    resolved = math.fsum(resolved)
    return {"counts": counts, "reasons": dict(sorted(reasons.items())),
            "score_bounds": [lower, min(1.0, lower + unknown)],
            "resolved_score": lower / resolved if resolved > 0 else None}


def _resample_means(values, rng, resamples):
    count = len(values)
    result = np.empty((resamples,) + values.shape[1:], dtype=np.float64)
    batch = max(1, 1_000_000 // values.size)
    for start in range(0, resamples, batch):
        stop = min(resamples, start + batch)
        indices = rng.integers(count, size=(stop - start, count))
        result[start:stop] = values[indices].mean(axis=1)
    return result


def _interval(point, samples, constant, *, weight_squares, width, alpha, limits):
    half_width = width * math.sqrt(math.log(2 / alpha) * weight_squares / 2)
    endpoints, methods = [], []
    for endpoint, tail in enumerate((alpha / 2, 1 - alpha / 2)):
        if constant[endpoint]:
            value = point[endpoint] + (-half_width if endpoint == 0 else half_width)
            methods.append("hoeffding")
        else:
            value = float(np.quantile(samples[:, endpoint], tail))
            methods.append("paired_block_bootstrap")
        endpoints.append(min(limits[1], max(limits[0], value)))
    return {"interval": endpoints, "interval_methods": methods}


def summarize(schedule, records, *, bootstrap_seed, resamples=10000, alpha=.05):
    """Summarize a frozen engine panel; absent records remain unresolved scheduled games.

    Profiles each identify one fixed opponent/context stratum, with positive weights
    summing to one. Blocks contain both colors for every arm and matched per-color
    agent seeds. Repeated random-input trials are rejected. Intervals are conditional
    on these profiles, and do not describe unseen engine families or human ranks.
    """
    alpha = _bootstrap_settings(bootstrap_seed, resamples, alpha)
    arms, control, profiles, slots, blocks = _engine_schedule(schedule, records)
    rng = np.random.default_rng(bootstrap_seed)
    distributions = np.zeros((resamples, len(arms), 2), dtype=np.float64)
    values_by_profile = {}
    for profile_id, profile in profiles.items():
        values = np.array([[[
            [float(_outcome(block[arm, color]["id"], records) == "W"),
             float(_outcome(block[arm, color]["id"], records) != "L")]
            for color in COLORS] for arm in arms]
            for _, block in sorted(blocks[profile_id].items())]).mean(axis=2)
        values_by_profile[profile_id] = values
        distributions += profile["weight"] * _resample_means(values, rng, resamples)

    weight_squares = math.fsum(profile["weight"] ** 2 / len(blocks[profile_id])
                               for profile_id, profile in profiles.items())
    families = sorted({profile["family"] for profile in profiles.values()})
    results = {}
    for arm_index, arm in enumerate(arms):
        def groups(profile_ids, color=None):
            total_weight = math.fsum(profiles[p]["weight"] for p in profile_ids)
            return [(profiles[p]["weight"] / total_weight,
                     [slot_id for slot_id, slot in slots.items()
                      if slot["arm"] == arm and slot["profile_id"] == p
                      and (color is None or slot["candidate_color"] == color)]) for p in profile_ids]

        summary = _score_summary(groups(profiles), records)
        summary["by_color"] = {color: _score_summary(groups(profiles, color), records) for color in COLORS}
        summary["by_profile"] = {p: _score_summary(groups([p]), records) for p in profiles}
        summary["by_family"] = {
            family: _score_summary(groups([p for p in profiles if profiles[p]["family"] == family]), records)
            for family in families}
        constant = [all(np.all(values[:, arm_index, endpoint] == values[0, arm_index, endpoint])
                        for values in values_by_profile.values()) for endpoint in range(2)]
        summary.update(_interval(summary["score_bounds"], distributions[:, arm_index], constant,
                                 weight_squares=weight_squares, width=1, alpha=alpha, limits=(0, 1)))
        results[arm] = summary

    comparisons = {}
    if control is not None:
        control_index = arms.index(control)
        for arm_index, arm in enumerate(arms):
            if arm == control:
                continue
            candidate_bounds, control_bounds = results[arm]["score_bounds"], results[control]["score_bounds"]
            point = [candidate_bounds[0] - control_bounds[1], candidate_bounds[1] - control_bounds[0]]
            samples = distributions[:, arm_index] - distributions[:, control_index, ::-1]
            constant = [all(np.all((differences := values[:, arm_index, endpoint]
                                   - values[:, control_index, 1 - endpoint]) == differences[0])
                            for values in values_by_profile.values()) for endpoint in range(2)]
            comparisons[arm] = {"control_arm": control, "gain_bounds": point,
                                **_interval(point, samples, constant, weight_squares=weight_squares,
                                            width=2, alpha=alpha, limits=(-1, 1))}
    block_counts = {p: len(blocks[p]) for p in profiles}
    return {"version": "match_statistics_v1", "alpha": alpha, "bootstrap_seed": bootstrap_seed,
            "resamples": resamples, "uncertainty_unit": "independent_color_pair_block_within_profile",
            "weight_squares": weight_squares, "profile_weights": {p: profiles[p]["weight"] for p in profiles},
            "block_counts": block_counts, "exploratory": any(n < 20 for n in block_counts.values()),
            "arms": results, "comparisons": comparisons}


def _binomial_logcdf(n, x, probability):
    if probability == 0 or x == n:
        return 0.0
    if probability == 1:
        return -math.inf
    mode = min(x, int((n + 1) * probability))
    log_term = (math.lgamma(n + 1) - math.lgamma(mode + 1) - math.lgamma(n - mode + 1)
                + mode * math.log(probability) + (n - mode) * math.log1p(-probability))
    terms = [1.0]
    term = 1.0
    for k in range(mode, 0, -1):
        term *= k / (n - k + 1) * ((1 - probability) / probability)
        terms.append(term)
    term = 1.0
    for k in range(mode, x):
        term *= (n - k) / (k + 1) * (probability / (1 - probability))
        terms.append(term)
    return min(0.0, log_term + math.log(math.fsum(terms)))


def _binomial_arguments(failures, trials, alpha):
    alpha = _alpha(alpha)
    if type(trials) is not int or type(failures) is not int or not 0 <= failures <= trials:
        raise ValueError("Binomial counts must be integers with 0 <= failures <= trials")
    return alpha


def clopper_pearson_upper(failures, trials, *, alpha=.05):
    """One-sided exact upper bound; zero trials gives the uninformative bound 1."""
    alpha = _binomial_arguments(failures, trials, alpha)
    if failures == trials:
        return 1.0
    if failures == 0:
        return -math.expm1(math.log(alpha) / trials)
    low, high, target = 0.0, 1.0, math.log(alpha)
    for _ in range(64):
        middle = (low + high) / 2
        if middle == low or middle == high:
            break
        if _binomial_logcdf(trials, failures, middle) > target:
            low = middle
        else:
            high = middle
    return high


def clopper_pearson_lower(failures, trials, *, alpha=.05):
    """One-sided exact lower bound; zero trials gives the uninformative bound 0."""
    alpha = _binomial_arguments(failures, trials, alpha)
    if failures == 0:
        return 0.0
    if failures == trials:
        return math.exp(math.log(alpha) / trials)
    return max(0.0, 1 - clopper_pearson_upper(trials - failures, trials, alpha=alpha))


def _failure_counts(slot_ids, records, flag):
    counts = {"trials": len(slot_ids), "failures": 0, "nonfailures": 0, "missing": 0, "unknown": 0}
    for slot_id in slot_ids:
        if slot_id not in records:
            counts["missing"] += 1
        else:
            value = records[slot_id].get("behavior", {}).get(flag)
            counts["failures" if value is True else "nonfailures" if value is False else "unknown"] += 1
    counts["adverse"] = counts["failures"] + counts["missing"] + counts["unknown"]
    return counts


def reliability_bounds(schedule, records, *, flag, candidate_arm, control_arm=None, alpha=.05):
    """Exact engine-panel game-rate bound; callers own assertion allocation and gates.

    An absolute assertion divides alpha among 2P candidate upper bounds. A
    regression assertion uses 4P candidate-upper/control-lower bounds. Unknown
    flags are adverse only for the candidate upper bound; n remains scheduled n.
    """
    alpha = _alpha(alpha)
    _identifier(flag, "behavior flag")
    arms, _, profiles, slots, blocks = _engine_schedule(schedule, records)
    if candidate_arm not in arms or (control_arm is not None and control_arm not in arms):
        raise ValueError("Reliability arms must be scheduled")
    if candidate_arm == control_arm:
        raise ValueError("Candidate and control must be different arms")
    marginal_alpha = alpha / ((2 if control_arm is None else 4) * len(profiles))
    strata, contributions = {}, []
    for profile_id, profile in profiles.items():
        colors = {}
        for color in COLORS:
            def for_arm(arm):
                return _failure_counts([slot_id for slot_id, slot in slots.items()
                                        if slot["profile_id"] == profile_id and slot["arm"] == arm
                                        and slot["candidate_color"] == color], records, flag)

            candidate = for_arm(candidate_arm)
            candidate["upper"] = clopper_pearson_upper(candidate["adverse"], candidate["trials"], alpha=marginal_alpha)
            entry, contribution = {"candidate": candidate}, candidate["upper"]
            if control_arm is not None:
                control = for_arm(control_arm)
                control["lower"] = clopper_pearson_lower(control["failures"], control["trials"], alpha=marginal_alpha)
                entry["control"] = control
                contribution -= control["lower"]
            colors[color] = entry
            contributions.append(profile["weight"] * contribution / 2)
        strata[profile_id] = {"context_id": profile["context_id"], "family": profile["family"],
                              "weight": profile["weight"], "by_color": colors}
    return {"version": "match_reliability_v1", "flag": flag, "candidate_arm": candidate_arm,
            "control_arm": control_arm, "alpha": alpha, "marginal_alpha": marginal_alpha,
            "marginal_bounds": (2 if control_arm is None else 4) * len(profiles),
            "method": "one_sided_exact_clopper_pearson_union_bound", "strata": strata,
            "exploratory": any(len(blocks[p]) < 20 for p in profiles),
            "absolute_upper" if control_arm is None else "regression_upper": math.fsum(contributions)}


def human_score_interval(schedule, records, *, bootstrap_seed, resamples=10000, alpha=.05):
    """Equal-opponent score interval for one declared human cohort, retaining all slots.

    A human_match_schedule_v1 has cohort_id and slots with id, opponent_id,
    block_id and candidate_color. Each opponent's blocks contain B/W games. Whole
    opponents, including all their scheduled games, are the resampling units.
    """
    alpha = _bootstrap_settings(bootstrap_seed, resamples, alpha)
    if not isinstance(schedule, Mapping) or schedule.get("version") != "human_match_schedule_v1":
        raise ValueError("Human scores need a human_match_schedule_v1 schedule")
    cohort = _identifier(schedule.get("cohort_id"), "cohort_id")
    declared = schedule.get("slots")
    if not isinstance(declared, list) or not declared:
        raise ValueError("Human scores require scheduled opponent games")
    slots, pairs, opponents = {}, defaultdict(set), defaultdict(list)
    for slot in declared:
        if not isinstance(slot, Mapping):
            raise ValueError("Human slots must be objects")
        slot_id = _identifier(slot.get("id"), "slot id")
        opponent = _identifier(slot.get("opponent_id"), "opponent_id")
        block = _identifier(slot.get("block_id"), "block_id")
        color = slot.get("candidate_color")
        if slot_id in slots or color not in COLORS or color in pairs[opponent, block]:
            raise ValueError("Human slots require unique IDs and one game per block/color")
        if "cohort_id" in slot and slot["cohort_id"] != cohort:
            raise ValueError("Analyze human cohorts separately")
        slots[slot_id] = slot
        pairs[opponent, block].add(color)
        opponents[opponent].append(slot_id)
    if any(colors != set(COLORS) for colors in pairs.values()):
        raise ValueError("Each human opponent block must schedule both colors")
    _validate_records(slots, records)
    opponents = dict(sorted(opponents.items()))
    summaries = {opponent: _score_summary([(1, slot_ids)], records) for opponent, slot_ids in opponents.items()}
    values = np.array([summary["score_bounds"] for summary in summaries.values()])
    distribution = _resample_means(values, np.random.default_rng(bootstrap_seed), resamples)
    constant = np.all(values == values[0], axis=0)
    result = _score_summary([(1 / len(opponents), slots) for slots in opponents.values()], records)
    result.update(_interval(result["score_bounds"], distribution, constant, weight_squares=1 / len(opponents),
                            width=1, alpha=alpha, limits=(0, 1)))
    result["interval_methods"] = ["opponent_cluster_bootstrap" if method == "paired_block_bootstrap" else method
                                  for method in result["interval_methods"]]
    result.update({"version": "human_match_statistics_v1", "cohort_id": cohort, "alpha": alpha,
                   "bootstrap_seed": bootstrap_seed, "resamples": resamples, "opponents": len(opponents),
                   "uncertainty_unit": "whole_opponent", "weight_squares": 1 / len(opponents),
                   "by_opponent": summaries})
    return result
