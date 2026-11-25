"""Build one stopping sample from the original pool and reviewed source additions."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np

from kenggo.canonical_data import digest
from kenggo.checkpoints import file_hash, save_json
from kenggo.continuation_data import path, prepare_diagnostic, read, reference, validate_lineage
from kenggo.continuation_training import REVIEW_RANKS, load_diagnostic
from kenggo.data_splits import read_membership
from kenggo.encoding import INPUT_KEYS, apply_normalizer, array_schema
from kenggo.examples import ExampleConfig, LABEL_SCHEMA, examples_for_game, sealed_sources
from kenggo.kgs_source import RANKS
from kenggo.progressive_evaluation import choose_games


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "composite_stopping_validation_request_v1"
COMPOSITE = "composite_stopping_validation_v1"
FINGERPRINTS = ("sgf_sha256", "audit_sequence_sha256", "trajectory_sha256")


def unique_rows(rows, label):
    result = {row["source_url"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Repeated source membership: {label}")
    return result


def historical_barrier(summaries):
    blocked, hashes = set(), {name: set() for name in FINGERPRINTS}
    for summary in summaries:
        index = read(summary["artifacts"]["base_index"])
        exposure = read(summary["artifacts"]["exposure"])
        if (index["canonical_sha256"] != summary["artifacts"]["canonical"]["sha256"]
                or index["split_sha256"] != summary["artifacts"]["split_manifest"]["sha256"]
                or index["exposure"] != summary["artifacts"]["exposure"]):
            raise ValueError("Historical index does not describe its pinned corpus")
        blocked.update(index.get("processed_source_ids", []))
        for section in ("records", "rejected", "study_rejections"):
            for row in index.get(section, []):
                blocked.add(row["source_url"])
                for name in FINGERPRINTS:
                    if row.get(name):
                        hashes[name].add(row[name])
        blocked.update(exposure["source_ids"])
        hashes["sgf_sha256"].update(exposure["sgf_hashes"])
        hashes["audit_sequence_sha256"].update(exposure["sequence_hashes"])
    return blocked, hashes


def supplemental_inventory(accepted, plan, blocked, hashes, *, check=lambda: None):
    """Use played colors, ranks, komi and whole-game membership for selection."""
    unique_rows(accepted, "supplement")
    inventory, records = [], {}
    seen = {name: set() for name in FINGERPRINTS}
    for unit in accepted:
        check()
        record = read(unit["canonical"])
        member, source = unit["membership"], unit["source_url"]
        gate, metadata, actions = record["eligibility"], record["metadata"], record["actions"]
        colors = [sum(move["color"] == color for move in actions) for color in ("B", "W")]
        groups = [gate[side + "_group"] for side in ("black", "white")]
        identities = [gate[side + "_identity_valid"] for side in ("black", "white")]
        ranks = sorted({metadata[field] for field, count in zip(("BR", "WR"), colors) if count})
        if (unit["status"] != "canonical" or source in blocked
                or member["source_url"] != source or record["source"]["source_url"] != source
                or member["canonical_sha256"] != digest(record)
                or member["role"] != "validation_development" or member["exclusions"]
                or member["duplicate_representative"] != source or member["duplicate_sources"] != [source]
                or member["final_excluded"] is not True or record.get("sealed_final") is not False
                or gate["training_eligible"] is not True or member["groups"] != groups
                or member["identity_valid"] != identities or any(identities)
                or member["adaptation_colors"] or member["known_persona_colors"]
                or not set(groups).intersection(plan["validation_groups"])
                or set(groups).intersection(plan["test_groups"])
                or member["date"] != metadata["DT"]
                or not plan["train_before"] <= member["date"] < plan["confirmation_from"]
                or member["actions"] != len(actions) or member["policy_actions"] != len(actions)
                or not actions or member["actions_by_color"] != colors
                or member["value_actions"] != (len(actions) if record["recorded_value_supported"] else 0)
                or record["replay_validated"] is not True or record["replay_pending_cap"] is not None
                or metadata["KM"] != 6.5 or metadata["RU"] != "Japanese"
                or "11k" not in ranks or any(rank not in RANKS for rank in ranks)
                or record["dynamic_clock_input_valid"] or record["duration_valid"]):
            raise ValueError("Supplement differs from admitted, complete validation membership")
        actual = {"sgf_sha256": record["source"]["sgf_sha256"],
                  "audit_sequence_sha256": record["audit"].get("sequence_fingerprint"),
                  "trajectory_sha256": record["trajectory_sha256"]}
        for name in FINGERPRINTS:
            value = unit[name]
            if not value or value != actual[name] or value in hashes[name] or value in seen[name]:
                raise ValueError(f"Supplement has duplicate or inconsistent {name}")
            seen[name].add(value)
        inventory.append({"source_url": source, "ranks": ranks, "rows": len(actions), "komi_halfpoints": 13})
        records[source] = record
    return sorted(inventory, key=lambda row: row["source_url"]), records


def freeze_selection(inventory, excluded, original_ids, supplemental_ids, *, tier=1, ranks=REVIEW_RANKS):
    if type(tier) is not int or tier < 1:
        raise ValueError("A stopping sample needs a positive integer tier")
    seed = f"stopping-20261121-tier-{tier}"
    by_id = unique_rows(inventory, "combined inventory")
    selected = choose_games(inventory, seed=seed, excluded=excluded, per_rank=2, ranks=ranks)
    cells = []
    for komi in (1, 13):
        for rank in ranks:
            ordered = sorted((row for row in inventory if row["source_url"] not in excluded
                              and row["komi_halfpoints"] == komi and rank in row["ranks"]),
                             key=lambda row: digest([seed, komi, rank, row["source_url"]]))
            choices = [row["source_url"] for row in ordered[:2]]
            if len(choices) != 2 or len(set(choices)) != 2:
                raise ValueError("Every rank/komi cell needs two distinct games")
            cells.append({"rank": rank, "komi_halfpoints": komi, "source_ids": choices,
                          "eligible_games": len(ordered)})
    if (set(selected) != {source for cell in cells for source in cell["source_ids"]}
            or set(selected).intersection(excluded)
            or set(original_ids).intersection(supplemental_ids)
            or not set(selected) <= set(original_ids) | set(supplemental_ids)
            or any(by_id[source]["rows"] <= 0 for source in selected)):
        raise ValueError("Selection differs from the unchanged whole-game selector")
    return {"schema": "composite_stopping_selection_v1", "tier": tier, "seed": seed, "per_cell": 2,
            "ranks": list(ranks), "komi_halfpoints": [1, 13], "cells": cells,
            "cell_selections": sum(len(cell["source_ids"]) for cell in cells),
            "source_ids": selected, "unique_games": len(selected),
            "expected_rows": sum(by_id[source]["rows"] for source in selected),
            "excluded_source_ids": sorted(excluded),
            "original_source_ids": sorted(set(selected).intersection(original_ids)),
            "supplemental_source_ids": sorted(set(selected).intersection(supplemental_ids)),
            "selection_uses_targets_or_predictions": False}


def verify_reused(ref, purpose, base, members, ranks, *, expected_sources=None, expected_rows=None):
    manifest = read(ref)
    rows, sources = manifest["rows"], manifest["source_ids"]
    permitted = unique_rows(members[manifest["role"]], "reused diagnostic role")
    if (manifest["version"] != "continuation_diagnostic_v1" or manifest["purpose"] != purpose
            or manifest["inputs_are_normalized"] is not True
            or manifest["normalizer_sha256"] != base["normalizer"]["sha256"]
            or any(manifest["data"].get(key) != value for key, value in base.items())
            or manifest["aggregate_ranks"] != list(ranks)
            or len(sources) != len(set(sources)) or not set(sources) <= permitted.keys()
            or any(permitted[source]["exclusions"] for source in sources)
            or (manifest["role"] == "train") != (purpose == "train_monitor")
            or manifest["role"] not in {"train", "population_development", "validation_development"}
            or {row["source_url"] for row in rows} != set(sources)
            or Counter(row["source_url"] for row in rows)
                != Counter({source: permitted[source]["policy_actions"] for source in sources})
            or len({row["row_key"] for row in rows}) != len(rows)
            or any(row["role"] != manifest["role"] for row in rows)
            or manifest["excluded_rank_rows"] != sum(row["source_rank_mover"] not in ranks for row in rows)
            or file_hash(path(manifest["arrays"])) != manifest["arrays"]["sha256"]
            or expected_sources is not None and set(sources) != set(expected_sources)):
        raise ValueError("Reused diagnostic differs from its original fitting/development artifact")
    if expected_rows is not None:
        frozen = {row["row_key"]: row for row in expected_rows}
        if len(frozen) != len(expected_rows) or set(frozen) != {row["row_key"] for row in rows}:
            raise ValueError("Fixed assessment row keys changed")
        fields = ("canonical_sha256", "sgf_sha256", "state_sha256", "source_url", "mover", "ply",
                  "komi_halfpoints", "rank_hidden")
        if any(any(row[key] != frozen[row["row_key"]][key] for key in fields) for row in rows):
            raise ValueError("Fixed assessment row provenance changed")
    return manifest


def context(request_path, pin, *, tier=1, previous=(), fixture=False, check=lambda: None):
    request_ref = {"path": str(Path(request_path).resolve()), "sha256": pin}
    request = read(request_ref)
    if (request["schema"] != SCHEMA or request["fixture_only"] is not fixture
            or type(tier) is not int or tier < 1 or len(previous) != tier - 1
            or len({ref["sha256"] for ref in previous}) != len(previous)):
        raise ValueError("Wrong composite request scope")
    if not fixture:
        if (subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
                != request["source_commit"]
                or subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
                or request["ranks"] != list(REVIEW_RANKS)
                or request["expected"] != {"original_validation_games": 2254, "supplemental_games": 7,
                     "original_train_rows": 2000000, "grown_train_rows": 3999997, "grown_train_games": 18126,
                     "fixed_games": 539, "fixed_rows": 120450, "fixed_scope_rows": 111591}
                or request["deadline_utc"] != "2026-11-28T14:13:03.148730+00:00"):
            raise ValueError("Composite production scope or source revision changed")
    for name, expected in request["code_sha256"].items():
        check()
        if file_hash(ROOT / name) != expected:
            raise ValueError("Composite code pin changed: " + name)
    refs, ranks = request["inputs"], request["ranks"]
    original = read(refs["original_config"])
    base = original["parameters"]["base_inputs"]
    if refs["fixed_comparison"] != original["inputs"]["fixed_comparison"]:
        raise ValueError("Fixed comparison changed from the original study")
    parent, grown = read(refs["parent_summary"]), read(refs["grown_summary"])
    for key, artifact in (("examples", "examples_manifest"), ("split", "split_manifest"),
                          ("normalizer", "normalizer"), ("canonical", "canonical")):
        if base[key] != parent["artifacts"][artifact]:
            raise ValueError("Original fitting data differs from the study configuration")
    grown_inputs = {"examples": grown["artifacts"]["examples_manifest"],
                    "split": grown["artifacts"]["split_manifest"], "profile": grown["artifacts"]["profile"],
                    "normalization_examples": base["examples"], "normalization_split": base["split"],
                    "normalizer": base["normalizer"]}
    grown_examples, normalizer = validate_lineage(grown_inputs, fixture_only=fixture)
    examples = read(base["examples"])
    split, members = read_membership(path(base["split"]).parent, base["split"]["sha256"])
    _, grown_members = read_membership(path(grown_inputs["split"]).parent, grown_inputs["split"]["sha256"])
    if (normalizer["examples_sha256"] != base["examples"]["sha256"]
            or examples["config"]["history"] != 0
            or examples["counts"]["train"]["examples"] != request["expected"]["original_train_rows"]
            or grown_examples["counts"]["train"]["examples"] != request["expected"]["grown_train_rows"]
            or len(grown_members["train"]) != request["expected"]["grown_train_games"]):
        raise ValueError("Original normalization fit or fitting population changed")
    original_by_id = {}
    for role, rows in members.items():
        for source, member in unique_rows(rows, role).items():
            if source in original_by_id:
                raise ValueError("Source membership crosses original roles")
            original_by_id[source] = member
    train_sources = set(unique_rows(grown_members["train"], "grown training"))
    blocked, hashes = historical_barrier((parent, grown))
    summary, accepted = read(refs["source_summary"]), read(refs["source_accepted"])["games"]
    source_request, review = read(refs["source_request"]), read(refs["source_review"])
    phase = read(refs["source_phase"])
    phase_completion = read(phase["source_summary"])
    if (summary["status"] != "completed" or summary["yield_status"] != "sufficient_yield"
            or summary["accepted"] != accepted or summary["accepted_games"] != len(accepted)
            or summary["usable_games"] != len(accepted)
            or len(accepted) != request["expected"]["supplemental_games"]
            or summary["request"] != refs["source_request"]
            or summary["original_deadline_utc"] != request["deadline_utc"]
            or summary["training_performed"] or summary["model_queries"] or summary["protected_targets_decoded"]
            or source_request["inputs"]["parent_summary"] != refs["parent_summary"]
            or source_request["inputs"]["grown_summary"] != refs["grown_summary"]
            or review["request_sha256"] != refs["source_request"]["sha256"]
            or review["model"] != "gpt-6-astra" or review["reasoning"] != "max"
            or review["verdict"] != "no significant findings"
            or phase["status"] != "completed" or phase_completion["source_summary"] != refs["source_summary"]):
        raise ValueError("Supplement requires completed, independently reviewed source evidence")
    supplemental, records = supplemental_inventory(accepted, split["plan"], blocked, hashes, check=check)
    exposure = read(refs["source_exposure"])
    if any(unit["source_url"] not in exposure["source_ids"] or unit["sgf_sha256"] not in exposure["sgf_hashes"]
           or unit["audit_sequence_sha256"] not in exposure["sequence_hashes"] for unit in accepted):
        raise ValueError("Supplement source exposure is incomplete")
    original_inventory = read(refs["original_inventory"])
    inventory_by_id = unique_rows(original_inventory, "original inventory")
    allowed = {source: row for source, row in original_by_id.items()
               if row["role"] == "validation_development" and row["policy_actions"] > 0 and not row["exclusions"]}
    if (set(inventory_by_id) != set(allowed) or len(allowed) != request["expected"]["original_validation_games"]
            or any(row["rows"] != allowed[source]["policy_actions"]
                   or row["komi_halfpoints"] not in (1, 13) or not row["ranks"]
                   or len(set(row["ranks"])) != len(row["ranks"])
                   or any(rank not in RANKS for rank in row["ranks"])
                   for source, row in inventory_by_id.items())):
        raise ValueError("Original validation inventory differs from its complete memberships")
    monitor = verify_reused(refs["train_monitor"], "train_monitor", base, members, ranks)
    fixed = read(refs["fixed_comparison"])
    if set(request["assessments"]) != {item["spec"]["id"] for item in fixed["common_sets"]}:
        raise ValueError("All four original fixed assessments are required")
    fixed_sources, fixed_rows, scope_rows = set(), 0, 0
    for common in fixed["common_sets"]:
        check()
        spec, frozen = common["spec"], read(common["manifest"])
        if (frozen["inputs"]["examples_sha256"] != base["examples"]["sha256"]
                or frozen["inputs"]["normalizer_sha256"] != base["normalizer"]["sha256"]):
            raise ValueError("Fixed comparison used different examples or normalization")
        reused = verify_reused(request["assessments"][spec["id"]], "assessment", base, members, ranks,
                               expected_sources=spec["source_ids"], expected_rows=frozen["rows"])
        if fixed_sources.intersection(reused["source_ids"]):
            raise ValueError("Fixed assessment memberships overlap")
        fixed_sources.update(reused["source_ids"])
        fixed_rows += len(reused["rows"])
        scope_rows += len(reused["rows"]) - reused["excluded_rank_rows"]
    if (len(fixed_sources), fixed_rows, scope_rows) != tuple(request["expected"][key]
            for key in ("fixed_games", "fixed_rows", "fixed_scope_rows")):
        raise ValueError("Fixed assessment population changed")
    previous_sources, previous_tiers = set(), set()
    for ref in previous:
        prepared = read(ref)
        diagnostic = read(prepared["epoch_validation"])
        sources = set(diagnostic["source_ids"])
        if (prepared["schema"] != "progressive_preparation_v1" or prepared["status"] != "completed"
                or prepared["train_monitor"] != refs["train_monitor"]
                or prepared["assessments"] != request["assessments"]
                or diagnostic["purpose"] != "epoch_validation" or diagnostic["role"] != "validation_development"
                or diagnostic["normalizer_sha256"] != base["normalizer"]["sha256"]
                or sources.intersection(fixed_sources | train_sources | previous_sources)):
            raise ValueError("Previous stopping sample has invalid purpose or overlapping membership")
        previous_tiers.add(prepared["tier"])
        previous_sources.update(sources)
    if previous_tiers != set(range(1, tier)):
        raise ValueError("A new tier requires every previous stopping sample")
    supplemental_ids = {row["source_url"] for row in supplemental}
    if supplemental_ids.intersection(original_by_id.keys() | train_sources | fixed_sources):
        raise ValueError("Supplement leaks into an existing fitting or reserved membership")
    protected = sealed_sources(split) | sealed_sources(examples)
    excluded = train_sources | fixed_sources | previous_sources | protected
    inventory = sorted(original_inventory + supplemental, key=lambda row: row["source_url"])
    selection = freeze_selection(inventory, excluded, inventory_by_id, supplemental_ids, tier=tier, ranks=ranks)
    barrier = {"schema": "stopping_source_growth_exclusions_v1", "source_summary": refs["source_summary"],
               "source_exposure": refs["source_exposure"], "source_accepted": refs["source_accepted"],
               "source_ids": sorted(supplemental_ids),
               **{name: sorted(unit[name] for unit in accepted) for name in FINGERPRINTS},
               "scope": "All accepted supplemental games remain outside fitting, including unselected games."}
    return {"request": request, "request_ref": request_ref, "base": base, "normalizer": normalizer,
            "plan": split["plan"], "example_config": examples["config"], "accepted": accepted, "records": records,
            "inventory": inventory, "selection": selection, "barrier": barrier,
            "original_members": original_by_id, "fixed_sources": fixed_sources, "previous_sources": previous_sources,
            "tier": tier, "previous_preparations": list(previous),
            "train_monitor": refs["train_monitor"], "assessments": request["assessments"],
            "train_monitor_rows": len(monitor["rows"])}


def supplemental_arrays(ctx, selected, *, check=lambda: None):
    units = unique_rows(ctx["accepted"], "supplement")
    raw_rows = []
    config = ExampleConfig(**ctx["example_config"])
    for source in selected:
        check()
        raw_rows.extend(examples_for_game(ctx["records"][source], units[source]["membership"], ctx["plan"],
                                         config, komi_study=True, check_budget=check))
    if not raw_rows:
        raise ValueError("The coverage repair must include supplemental games")
    schema = {**array_schema(0), **LABEL_SCHEMA}
    arrays = {name: np.asarray([row[0 if name in INPUT_KEYS else 1][name] for row in raw_rows], dtype=dtype)
                        .reshape((len(raw_rows), *shape)) for name, (dtype, shape) in schema.items()}
    rows = []
    for inputs, _, row in raw_rows:
        scalar = inputs["scalars"]
        clock = ([float(scalar[9]), float(scalar[10]), int(scalar[11])]
                 if inputs["scalar_valid"][9:12].all() else None)
        rows.append({**row, "nominal_clock": clock, "komi_halfpoints": int(round(2 * abs(float(scalar[1]))))})
    expected_policy = sum(units[source]["membership"]["policy_actions"] for source in selected)
    expected_value = sum(units[source]["membership"]["value_actions"] for source in selected)
    if (len(rows) != expected_policy or int(arrays["policy_valid"].sum()) != expected_policy
            or int(arrays["value_valid"].sum()) != expected_value or arrays["duration_valid"].any()):
        raise ValueError("Supplemental complete-game masks/counts changed")
    arrays.update(apply_normalizer({name: arrays[name] for name in INPUT_KEYS}, ctx["normalizer"]))
    return arrays, rows


def assemble(ctx, work, *, fixture=False, check=lambda: None):
    selection, request, refs = ctx["selection"], ctx["request"], ctx["request"]["inputs"]
    saved = {}
    for name, value in (("inventory", ctx["inventory"]), ("selection", selection),
                        ("growth-exclusions", ctx["barrier"])):
        save_json(work / (name + ".json"), value)
        saved[name] = reference(work / (name + ".json"))
    check()
    old_ref = prepare_diagnostic(ctx["base"], work / "original", purpose="epoch_validation",
        role="validation_development", source_ids=selection["original_source_ids"], aggregate_ranks=request["ranks"],
        excluded_source_ids=selection["excluded_source_ids"], check_budget=check)
    old = read(old_ref)
    if file_hash(path(old["arrays"])) != old["arrays"]["sha256"]:
        raise ValueError("Original diagnostic arrays changed")
    with np.load(path(old["arrays"]), allow_pickle=False) as values:
        arrays = {name: values[name] for name in (*INPUT_KEYS, *LABEL_SCHEMA)}
    extra, extra_rows = supplemental_arrays(ctx, selection["supplemental_source_ids"], check=check)
    combined = {name: np.concatenate([arrays[name], extra[name]]) for name in arrays}
    rows = old["rows"] + extra_rows
    units = unique_rows(ctx["accepted"], "supplement")
    if (len(rows) != selection["expected_rows"] or len({row["row_key"] for row in rows}) != len(rows)
            or {row["source_url"] for row in rows} != set(selection["source_ids"])):
        raise ValueError("Composite rows differ from the frozen whole-game selection")
    check()
    with (work / "arrays.npz").open("xb") as output:
        np.savez_compressed(output, **combined)
    supplemental_refs = {"source_summary": refs["source_summary"], "accepted": refs["source_accepted"],
                         "exposure": refs["source_exposure"]}
    composite = {"version": COMPOSITE, "request": ctx["request_ref"], "tier": ctx["tier"],
        "previous_preparations": ctx["previous_preparations"], "selection": saved["selection"],
        "inventory": saved["inventory"], "growth_exclusions": saved["growth-exclusions"],
        "components": [{"kind": "original", "diagnostic": old_ref,
                        "source_ids": selection["original_source_ids"],
                        "memberships": [ctx["original_members"][source] for source in selection["original_source_ids"]]},
                       {"kind": "supplemental", "source_summary": refs["source_summary"],
                        "source_ids": selection["supplemental_source_ids"],
                        "games": [units[source] for source in selection["supplemental_source_ids"]]}],
        "normalization": {"normalizer": ctx["base"]["normalizer"], "fitting_examples": ctx["base"]["examples"],
                          "fitting_split": ctx["base"]["split"]}, "original_fitting_data_unchanged": True}
    manifest = {"version": "continuation_diagnostic_v1", "purpose": "epoch_validation",
        "role": "validation_development", "arrays": reference(work / "arrays.npz"), "rows": rows,
        "aggregate_ranks": request["ranks"], "normalizer_sha256": ctx["base"]["normalizer"]["sha256"],
        "inputs_are_normalized": True, "source_ids": selection["source_ids"],
        "excluded_rank_rows": sum(row["source_rank_mover"] not in request["ranks"] for row in rows),
        "data": {"original": ctx["base"], "supplemental": supplemental_refs}, "composite": composite}
    save_json(work / "manifest.json", manifest)
    diagnostic = reference(work / "manifest.json")
    loaded = load_diagnostic(diagnostic, "epoch_validation", ctx["base"]["normalizer"]["sha256"], fixture_only=fixture)
    if len(loaded.rows) != selection["expected_rows"]:
        raise ValueError("Unchanged diagnostic loader lost rows")
    check()
    return {"schema": "progressive_preparation_v1", "status": "completed", "tier": ctx["tier"],
        "train_monitor": ctx["train_monitor"], "epoch_validation": diagnostic, "assessments": ctx["assessments"],
        "validation_inventory": saved["inventory"], "fixed_comparison": refs["fixed_comparison"],
        "selection_uses_targets_or_predictions": False, "previous_validation_sources": len(ctx["previous_sources"]),
        "fixed_comparison_sources": len(ctx["fixed_sources"]), "composite": composite,
        "original_validation_inventory": refs["original_inventory"], "supplemental_sources": supplemental_refs,
        "growth_exclusions": saved["growth-exclusions"], "training_performed": False, "model_queries": 0,
        "protected_targets_decoded": False}


def preflight(request_path, request_pin, *, tier=1, previous=(), check_budget=lambda: None, fixture=False):
    ctx = context(request_path, request_pin, tier=tier, previous=previous, fixture=fixture, check=check_budget)
    return {"schema": "composite_stopping_preflight_v1", "status": "metadata_verified", "tier": tier,
            "selection": ctx["selection"], "inventory_games": len(ctx["inventory"]),
            "growth_exclusions": ctx["barrier"], "training_performed": False, "model_queries": 0}


def build(request_path, request_pin, output, *, tier=1, previous=(), check_budget=lambda: None, fixture=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError("Composite output already has a summary")
    work = output / "composite"
    work.mkdir(exist_ok=False)
    request_ref = {"path": str(Path(request_path).resolve()), "sha256": request_pin}
    save_json(work / "started.json", {"request": request_ref, "utc": datetime.now(timezone.utc).isoformat()})
    try:
        ctx = context(request_path, request_pin, tier=tier, previous=previous, fixture=fixture, check=check_budget)
        result = assemble(ctx, work, fixture=fixture, check=check_budget)
    except BaseException as error:
        save_json(output / "summary.json", {"schema": "progressive_preparation_v1", "status": "failed",
            "request": request_ref, "reason": f"{type(error).__name__}: {error}",
            "training_performed": False, "model_queries": 0, "failed_evidence_preserved": True})
        raise
    save_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "run"))
    parser.add_argument("request")
    parser.add_argument("sha256")
    parser.add_argument("output", nargs="?")
    args = parser.parse_args()
    request = read({"path": str(Path(args.request).resolve()), "sha256": args.sha256})
    def check():
        if datetime.now(timezone.utc) >= datetime.fromisoformat(request["deadline_utc"]):
            raise TimeoutError("Original study deadline reached")
    if args.command == "check":
        if args.output is not None:
            parser.error("check does not write output")
        result = preflight(args.request, args.sha256, check_budget=check)
    else:
        if args.output is None:
            parser.error("run requires OUTPUT")
        result = build(args.request, args.sha256, args.output, check_budget=check)
        result = {"status": result["status"], "summary": reference(Path(args.output) / "summary.json")}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
