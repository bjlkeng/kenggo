"""Policy-only continuation settings and a checkpointed plateau schedule."""

import math


RECIPE = "e2_policy_tuning_v1"
DEFAULTS = {"learning_rate": 0.0003, "weight_decay": 0.0, "value_loss_weight": 0.0,
            "lr_factor": 0.3, "lr_patience": 2, "min_lr": 0.000001,
            "min_rounds": 8, "patience": 6, "max_rounds": 48}
TRIALS = (
    {"id": "lr-1e-3", "learning_rate": 0.001, "weight_decay": 0.0},
    {"id": "lr-3e-4", "learning_rate": 0.0003, "weight_decay": 0.0},
    {"id": "lr-1e-4", "learning_rate": 0.0001, "weight_decay": 0.0},
    {"id": "lr-3e-4-decay", "learning_rate": 0.0003, "weight_decay": 0.0001},
)


def validate(params):
    if ((params["learning_rate"], params["weight_decay"]) not in {
                (trial["learning_rate"], trial["weight_decay"]) for trial in TRIALS}
            or any(type(params[key]) not in (int, float) or not math.isfinite(params[key])
                   for key in DEFAULTS)
            or any(params[key] != value for key, value in DEFAULTS.items()
                   if key not in ("learning_rate", "weight_decay"))
            or any(type(params[key]) is not int for key in
                   ("lr_patience", "min_rounds", "patience", "max_rounds"))):
        raise ValueError("Policy tuning requires its declared loss, optimizer and stopping recipe")


def update_schedule(progress, optimizer, params, score, *, initial):
    before = optimizer.param_groups[0]["lr"]
    if initial:
        state = {"reference": score, "bad_rounds": 0, "reductions": 0, "learning_rate": before}
    else:
        state = dict(progress["policy_schedule"])
        if score <= state["reference"] - params["min_delta"]:
            state.update(reference=score, bad_rounds=0)
        else:
            state["bad_rounds"] += 1
        if state["bad_rounds"] >= params["lr_patience"]:
            after = max(params["min_lr"], before * params["lr_factor"])
            if after < before:
                state["reductions"] += 1
            state.update(learning_rate=after, bad_rounds=0)
    for group in optimizer.param_groups:
        group["lr"] = state["learning_rate"]
    progress["policy_schedule"] = state
    return before, state["learning_rate"]


def stopped(progress, params):
    return (progress["completed_rounds"] >= params["min_rounds"]
            and progress["stopping"]["bad_rounds"] >= params["patience"])


def validate_schedule(progress, params):
    state = progress.get("policy_schedule")
    if not progress["initial_evaluation_complete"]:
        if state is not None:
            raise ValueError("An unevaluated checkpoint cannot have a policy schedule")
        return
    if (not isinstance(state, dict) or set(state) != {"reference", "bad_rounds", "reductions", "learning_rate"}
            or type(state["reference"]) not in (int, float) or not math.isfinite(state["reference"])
            or state["reference"] < 0
            or type(state["reductions"]) is not int
            or not 0 <= state["reductions"] <= progress["completed_rounds"] // params["lr_patience"]
            or type(state["bad_rounds"]) is not int
            or type(progress["stopping"]["bad_rounds"]) is not int
            or progress["stopping"]["bad_rounds"] < 0
            or state["bad_rounds"] != progress["stopping"]["bad_rounds"] % params["lr_patience"]
            or state["reference"] != progress["stopping"]["tolerance_reference"]
            or type(state["learning_rate"]) not in (int, float)
            or not math.isfinite(state["learning_rate"])
            or not math.isclose(state["learning_rate"], max(params["min_lr"],
                params["learning_rate"] * params["lr_factor"] ** state["reductions"]), rel_tol=1e-12)):
        raise ValueError("Policy checkpoint schedule is inconsistent with its stopping state")
