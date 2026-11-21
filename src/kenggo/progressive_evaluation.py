"""Disjoint stopping samples and matched saved-loss comparisons for data growth."""

from collections import defaultdict
import math
from pathlib import Path

import torch

from kenggo.canonical_data import digest
from kenggo.checkpoints import save_json
from kenggo.continuation_data import path, prepare_diagnostic, read, reference
from kenggo.continuation_training import REVIEW_RANKS, evaluate_diagnostic
from kenggo.examples import read_shards
from kenggo.kgs_source import RANKS
from kenggo.learned_agent import LearnedArtifacts, _normalizer, _weights
from kenggo.learner import ModelConfig, PopulationModel
from kenggo.recorded_statistics import METHOD, _estimate


def inventory(inputs, role, *, check_budget=lambda: None):
    """Use only source ranks, komi and memberships to choose whole games."""
    games = {}
    for arrays, metadata in read_shards(path(inputs["examples"]).parent,
                                       inputs["examples"]["sha256"], role, komi_study=True):
        check_budget()
        for i, row in enumerate(metadata):
            item = games.setdefault(row["source_url"], {"source_url": row["source_url"],
                "ranks": set(), "rows": 0, "komi_halfpoints": int(round(abs(float(arrays["scalars"][i, 1])) * 2))})
            item["ranks"].add(row["source_rank_mover"])
            item["rows"] += 1
    return [{**item, "ranks": sorted(item["ranks"])} for item in games.values()]


def choose_games(games, *, seed, excluded=(), per_rank=2, ranks=REVIEW_RANKS):
    if type(per_rank) is not int or per_rank < 1:
        raise ValueError("A diagnostic needs a positive whole-game quota")
    excluded, selected = set(excluded), set()
    for komi in (1, 13):
        for rank in ranks:
            ordered = sorted((item for item in games if item["source_url"] not in excluded
                              and item["komi_halfpoints"] == komi and rank in item["ranks"]),
                             key=lambda item: digest([seed, komi, rank, item["source_url"]]))
            if len(ordered) < per_rank:
                raise ValueError(f"Insufficient fresh validation games: komi {komi}/2, rank {rank}")
            selected.update(item["source_url"] for item in ordered[:per_rank])
    return sorted(selected)


def prepare(config, output, *, check_budget=lambda: None):
    inputs, params, output = config["inputs"], config["parameters"], Path(output)
    old = read(inputs["fixed_comparison"])
    fixed_games = {key for common in old["common_sets"] for key in common["spec"]["source_ids"]}
    previous = [read(ref) for key, ref in inputs.items() if key.startswith("previous_preparation_")]
    used = fixed_games | {key for preparation in previous
                         for key in read(preparation["epoch_validation"])["source_ids"]}
    if previous:
        common = previous[0]
        train_monitor, assessments = common["train_monitor"], common["assessments"]
        games = read(common["validation_inventory"])
        inventory_ref = common["validation_inventory"]
    else:
        games = inventory(inputs, "validation_development", check_budget=check_budget)
        save_json(output / "validation-inventory.json", games)
        inventory_ref = reference(output / "validation-inventory.json")
        train_games = inventory(inputs, "train", check_budget=check_budget)
        train_ids = choose_games(train_games, seed="training-diagnostic-20261121")
        train_monitor = prepare_diagnostic(inputs, output / "train-monitor", purpose="train_monitor",
            role="train", source_ids=train_ids, aggregate_ranks=REVIEW_RANKS, check_budget=check_budget)
        assessments = {}
        for common in old["common_sets"]:
            check_budget()
            spec, frozen = common["spec"], read(common["manifest"])
            if (frozen["inputs"]["examples_sha256"] != inputs["examples"]["sha256"]
                    or frozen["inputs"]["normalizer_sha256"] != inputs["normalizer"]["sha256"]):
                raise ValueError("Fixed assessment differs from the earlier combined-data sample")
            assessments[spec["id"]] = prepare_diagnostic(inputs, output / spec["id"], purpose="assessment",
                role=spec["role"], source_ids=spec["source_ids"], aggregate_ranks=REVIEW_RANKS,
                expected_row_keys=[row["row_key"] for row in frozen["rows"]], check_budget=check_budget)
    ids = choose_games(games, seed=f"stopping-20261121-tier-{params['tier']}", excluded=used)
    validation = prepare_diagnostic(inputs, output / "epoch-validation", purpose="epoch_validation",
        role="validation_development", source_ids=ids, aggregate_ranks=REVIEW_RANKS,
        excluded_source_ids=used, check_budget=check_budget)
    return {"schema": "progressive_preparation_v1", "status": "completed", "tier": params["tier"],
            "train_monitor": train_monitor, "epoch_validation": validation, "assessments": assessments,
            "validation_inventory": inventory_ref, "fixed_comparison": inputs["fixed_comparison"],
            "selection_uses_targets_or_predictions": False,
            "previous_validation_sources": len(used - fixed_games), "fixed_comparison_sources": len(fixed_games)}


def artifact_bundle(checkpoint):
    config = read(checkpoint["config"])
    refs = config["inputs"]
    return LearnedArtifacts(str(path({key: checkpoint[key] for key in ("path", "sha256")})),
        checkpoint["sha256"], str(path(checkpoint["config"])),
        checkpoint["config"]["sha256"], checkpoint["provenance"]["sha256"],
        str(path(refs["normalizer"])), refs["normalizer"]["sha256"], refs["examples"]["sha256"],
        ModelConfig(**config["parameters"]["model"]))


def assess(config, output, *, check_budget=lambda: None):
    output, refs = Path(output), config["inputs"]
    bundle = {"path": refs["checkpoint"]["path"], "sha256": refs["checkpoint"]["sha256"],
              "config": refs["training_config"], "provenance": refs["training_provenance"]}
    artifacts = artifact_bundle(bundle)
    _normalizer(artifacts)
    model = PopulationModel(artifacts.model_config)
    _weights(artifacts, model)
    device = config["parameters"]["device"]
    model.to(device)
    reports = {}
    for name, ref in sorted(refs.items()):
        if not name.startswith("assessment_"):
            continue
        check_budget()
        result = evaluate_diagnostic(model, ref, normalizer_sha256=artifacts.normalizer_sha256,
            device=device, purpose="assessment", include_predictions=True, check_budget=check_budget)
        identity = name.removeprefix("assessment_")
        save_json(output / (identity + ".json"), result)
        reports[identity] = reference(output / (identity + ".json"))
    if not reports:
        raise ValueError("Assessment requires fixed development diagnostics")
    if device == "cuda":
        torch.cuda.synchronize()
    return {"schema": "progressive_assessment_v1", "status": "completed", "checkpoint": bundle,
            "reports": reports, "scope": "exploratory development; final and confirmation remain closed"}


def paired_losses(candidate, control, *, ranks=REVIEW_RANKS):
    if candidate["diagnostic_sha256"] != control["diagnostic_sha256"]:
        raise ValueError("A paired comparison requires the same frozen diagnostic")
    def keyed(value):
        rows = value["predictions"]
        result = {row["row_key"]: row for row in rows}
        if len(result) != len(rows):
            raise ValueError("Repeated comparison row")
        return result
    candidate_rows, control_rows = keyed(candidate), keyed(control)
    if candidate_rows.keys() != control_rows.keys():
        raise ValueError("Paired predictions need identical complete row coverage")
    outcomes = {"nll", "top1", "value_mse", "prediction_value"}
    selected, losses = [], {name: {} for name in ("nll", "top1", "value_mse")}
    for key in sorted(candidate_rows):
        row, other = candidate_rows[key], control_rows[key]
        if ({name: value for name, value in row.items() if name not in outcomes}
                != {name: value for name, value in other.items() if name not in outcomes}):
            raise ValueError("Paired metadata, targets or masks changed")
        if row["source_rank_mover"] not in ranks:
            continue
        selected.append(row)
        for metric in losses:
            valid = row["value_valid" if metric == "value_mse" else "policy_valid"]
            if valid:
                values = (row[metric], other[metric])
                if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
                    raise ValueError("Paired valid losses must be finite")
                losses[metric][key] = values[0] - values[1]
    return {"kind": "candidate_minus_control", "aggregate_ranks": list(ranks), "rows": len(selected),
            "excluded_rows": len(candidate_rows) - len(selected), "uncertainty_method": METHOD,
            "metrics": {name: _estimate(selected, values, ranks,
                        "value_valid" if name == "value_mse" else "policy_valid")
                        for name, values in losses.items()}}


def decide(comparisons, *, primary="validation_development_komi13"):
    """A further doubling requires useful paired progress and no clear regression."""
    reasons = []
    for opponent in ("control", "parent"):
        metric = comparisons[opponent][primary]["metrics"]["nll"]
        interval = metric["uncertainty"].get("interval95")
        if (metric["status"] != "available" or interval is None
                or metric["uncertainty"]["status"] != "available"
                or interval[1] > -0.02):
            reasons.append(f"No supported useful NLL improvement against {opponent}")
        for identity, report in comparisons[opponent].items():
            for name in ("nll", "value_mse"):
                metric = report["metrics"][name]
                interval = metric["uncertainty"].get("interval95")
                if metric["status"] == "available" and interval and interval[0] > 0.02:
                    reasons.append(f"{identity}: {name} regression against {opponent}")
    return {"continue_growth": not reasons, "reasons": reasons or ["Useful paired progress without a clear regression"],
            "minimum_useful_nll_gain": 0.02, "allowed_regression": 0.02,
            "selection_data": "fixed development comparison; not final testing"}


def compare(config, output):
    reports = {name: read(config["inputs"][name])["reports"] for name in ("candidate", "control", "parent")}
    if not reports["candidate"].keys() == reports["control"].keys() == reports["parent"].keys():
        raise ValueError("Every selected checkpoint needs every fixed comparison set")
    comparisons = {name: {identity: paired_losses(read(candidate_ref), read(reports[name][identity]))
                         for identity, candidate_ref in reports["candidate"].items()}
                   for name in ("control", "parent")}
    return {"schema": "progressive_comparison_v1", "status": "completed", "comparisons": comparisons,
            "decision": decide(comparisons)}
