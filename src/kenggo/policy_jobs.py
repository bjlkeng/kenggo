"""Bounded preparation, assessment and publication for policy tuning."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

from kenggo.baseline_training import _source_identity
from kenggo.checkpoints import json_hash, read_json, save_json
from kenggo.continuation_data import path, read, reference
from kenggo.policy_recipe import TRIALS
from kenggo.progressive_evaluation import assess, paired_losses
from kenggo.progressive_jobs import readiness
from kenggo.tracking import validate_config


def compare(inputs, *, check_budget=lambda: None):
    selection = read(inputs["selection"])
    identities = [trial["id"] for trial in TRIALS]
    if selection["selected_trial_id"] not in identities or selection["assessment_seen"] is not False:
        raise ValueError("Trial selection must be sealed before assessment")
    assessments = {name: read(inputs[name]) for name in ["parent", *identities]}
    selected = {trial["id"]: trial["selected"] for trial in selection["trials"]}
    if len(selection["trials"]) != len(identities) or set(selected) != set(identities) or any(
            assessments[identity]["status"] != "completed"
            or assessments[identity]["checkpoint"] != {key: selected[identity][key]
                for key in ("path", "sha256", "config", "provenance")}
            for identity in identities):
        raise ValueError("Assessment checkpoints differ from the sealed validation selection")
    reports = {name: report["reports"] for name, report in assessments.items()}
    scopes = {role + "_komi" + str(komi) for role in
              ("population_development", "validation_development") for komi in (1, 13)}
    if assessments["parent"]["status"] != "completed" or any(set(refs) != scopes for refs in reports.values()):
        raise ValueError("Every trial must use the same complete fixed assessment")
    comparisons = {}
    for identity in identities:
        comparisons[identity] = {}
        for label, opponent in (("against_parent", "parent"), ("against_baseline", identities[0])):
            comparisons[identity][label] = {}
            for scope, ref in reports[identity].items():
                check_budget()
                comparisons[identity][label][scope] = paired_losses(read(ref), read(reports[opponent][scope]))
    return {"schema": "e2_policy_comparison_v1", "status": "completed",
        "selected_trial_id": selection["selected_trial_id"], "baseline_trial_id": identities[0],
        "comparisons": comparisons,
        "notes": ["Selection used stopping validation, before these assessment predictions existed.",
                  "All trials are exploratory continuations with independently selected stopping points.",
                  "Value error is diagnostic only and never vetoes a policy result.",
                  "Final and confirmation targets remain closed; no playing strength is established."]}


def run(config_path, config_pin, provenance_path, provenance_pin, output):
    config, provenance = read_json(config_path, config_pin), read_json(provenance_path, provenance_pin)
    validate_config(config)
    if (json_hash(provenance) != provenance_pin or provenance["inputs"] != config["inputs"]
            or provenance["source"] != _source_identity()):
        raise ValueError("Policy worker configuration and source must match")
    refs = [*config["inputs"].values(), *config["parameters"].get("base_inputs", {}).values()]
    for ref in refs:
        if reference(path(ref))["sha256"] != ref["sha256"]:
            raise ValueError("A policy input differs from its pin")
    output = Path(output)
    deadline = datetime.fromisoformat(config["parameters"]["deadline_utc"])
    if deadline.tzinfo is None:
        raise ValueError("The worker deadline requires a timezone")
    monotonic_end = time.monotonic() + (deadline - datetime.now(timezone.utc)).total_seconds()

    def check():
        if min((deadline - datetime.now(timezone.utc)).total_seconds(), monotonic_end - time.monotonic()) < 120:
            raise TimeoutError("The worker deadline was reached")

    check()
    purpose = config["purpose"]
    if purpose == "policy-readiness":
        result = readiness()
    elif purpose == "policy-preparation":
        from kenggo.policy_validation import prepare
        result = prepare(config["inputs"], config["parameters"], output, check_budget=check)
    elif purpose == "policy-assessment":
        result = assess(config, output, check_budget=check)
    elif purpose == "policy-comparison":
        result = compare(config["inputs"], check_budget=check)
    elif purpose == "policy-publication":
        from kenggo.policy_review import export
        review_path = output / "review.html"
        export(config["inputs"]["study"], review_path,
               history_path=path(config["inputs"]["review_history"]))
        result = {"status": "completed", "review": reference(review_path),
                  "costs_scope": "Page costs end before publication; final study summary includes publication"}
    else:
        raise ValueError("Unknown policy worker purpose")
    check()
    if provenance["source"] != _source_identity():
        raise ValueError("Worker source changed during production")
    save_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("provenance")
    parser.add_argument("output")
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--provenance-sha256", required=True)
    args = parser.parse_args()
    try:
        run(args.config, args.config_sha256, args.provenance, args.provenance_sha256, args.output)
    except BaseException as error:
        save_json(Path(args.output) / "failure.json", {"status": "failed", "reason": str(error)})
        raise


if __name__ == "__main__":
    main()
