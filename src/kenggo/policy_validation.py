"""Freeze larger policy stopping data without changing fitting or assessment data."""

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from kenggo.canonical_data import digest
from kenggo.checkpoints import file_hash, save_json
from kenggo.continuation_data import path, prepare_diagnostic, read, reference, validate_lineage
from kenggo.continuation_training import REVIEW_RANKS, load_diagnostic
from kenggo.data_splits import read_membership
from kenggo.encoding import INPUT_KEYS, apply_normalizer, array_schema
from kenggo.examples import ExampleConfig, LABEL_SCHEMA, examples_for_game, sealed_sources
from kenggo.kgs_source import RANKS


SEED = "policy-only-stopping-20261125-v1"
PER_CELL = 30
SELECTION_SHA256 = "a93fe207fdc8e538d2ebe0ba67628b179c33ac287440b662d18ef86ccd0828a3"
FINGERPRINTS = ("sgf_sha256", "audit_sequence_sha256", "trajectory_sha256")
DEVELOPMENT = "validation_development"
PROTECTED = {"final", "population_confirmation", "validation_confirmation", "test_prefix",
             "validation_prefix", "embargo"}


def same_ref(left, right):
    return left["sha256"] == right["sha256"] and path(left) == path(right)


def same_bundle(left, right):
    return left.keys() == right.keys() and all(same_ref(left[key], right[key]) for key in left)


def unique(rows, label):
    result = {row["source_url"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Repeated source in " + label)
    return result


def select_games(games, sides, *, excluded=(), ranks=REVIEW_RANKS, per_cell=PER_CELL, seed=SEED):
    """Fill sparse cells first, preferring new mover accounts before repeated ones."""
    by_source = unique(games, "validation inventory")
    if (type(per_cell) is not int or per_cell < 1 or not ranks or len(set(ranks)) != len(ranks)
            or any(rank not in RANKS for rank in ranks) or not isinstance(seed, str) or not seed):
        raise ValueError("Selection requires fixed ranks, seed and a positive game quota")
    for source, game in by_source.items():
        played = sides.get(source, {})
        if (type(game["rows"]) is not int or game["rows"] <= 0 or game["komi_halfpoints"] not in (1, 13)
                or not played or not set(played) <= {"B", "W"}
                or any(rank not in RANKS or not isinstance(account, str) or not account
                       for rank, account in played.values())
                or set(game["ranks"]) != {rank for rank, _ in played.values()}
                or len(game["ranks"]) != len(set(game["ranks"]))):
            raise ValueError("Inventory ranks, played accounts or complete-game counts differ")
    excluded = set(excluded)
    available = [game for game in games if game["source_url"] not in excluded]
    cells = {(komi, rank): [game for game in available
                           if game["komi_halfpoints"] == komi and rank in game["ranks"]]
             for komi in (1, 13) for rank in ranks}
    if any(not pool for pool in cells.values()):
        raise ValueError("Every declared rank/komi cell needs at least one unused game")
    selected, usage = set(), Counter()
    for komi, rank in sorted(cells, key=lambda cell: (len(cells[cell]), cell[0], ranks.index(cell[1]))):
        pool = cells[komi, rank]
        while sum(game["source_url"] in selected for game in pool) < min(per_cell, len(pool)):
            seen = {account for game in pool if game["source_url"] in selected
                    for mover_rank, account in sides[game["source_url"]].values() if mover_rank == rank}

            def ordering(game):
                source = game["source_url"]
                accounts = {account for _, account in sides[source].values()}
                movers = {account for mover_rank, account in sides[source].values() if mover_rank == rank}
                return (-len(movers - seen), max(usage[account] for account in accounts),
                        sum(usage[account] for account in accounts), digest([seed, komi, rank, source]))

            source = min((game for game in pool if game["source_url"] not in selected),
                         key=ordering)["source_url"]
            selected.add(source)
            usage.update({account for _, account in sides[source].values()})
    coverage = []
    for (komi, rank), pool in cells.items():
        chosen = [game["source_url"] for game in pool if game["source_url"] in selected]
        accounts = lambda sources: sorted({account for source in sources
            for mover_rank, account in sides[source].values() if mover_rank == rank})
        coverage.append({"rank": rank, "komi_halfpoints": komi, "available_games": len(pool),
            "available_mover_accounts": len(accounts(game["source_url"] for game in pool)),
            "target_games": min(per_cell, len(pool)), "selected_games": len(chosen),
            "selected_mover_accounts": accounts(chosen), "source_ids": sorted(chosen),
            "below_requested_quota": len(pool) < per_cell})
    sources = sorted(selected)
    return {"schema": "policy_stopping_selection_v1", "seed": seed, "per_cell": per_cell,
            "ranks": list(ranks), "komi_halfpoints": [1, 13], "cells": coverage,
            "source_ids": sources, "source_ids_sha256": digest(sources), "unique_games": len(sources),
            "expected_rows": sum(by_source[source]["rows"] for source in sources),
            "available_games": len(available), "available_rows": sum(game["rows"] for game in available),
            "accounts": len(usage), "maximum_games_per_account": max(usage.values()),
            "excluded_source_ids": sorted(excluded), "selection_uses_targets_or_predictions": False}


def historical_indexes(summaries):
    records, blocked = {}, set()
    hashes = {name: set() for name in FINGERPRINTS}
    for summary in summaries:
        artifacts = summary["artifacts"]
        index, exposure = read(artifacts["base_index"]), read(artifacts["exposure"])
        if (index["canonical_sha256"] != artifacts["canonical"]["sha256"]
                or index["split_sha256"] != artifacts["split_manifest"]["sha256"]
                or index["exposure"] != artifacts["exposure"]):
            raise ValueError("Historical index differs from its pinned corpus")
        for section in ("records", "rejected", "study_rejections"):
            for row in index.get(section, []):
                source = row["source_url"]
                blocked.add(source)
                if section == "records":
                    if source in records and any(records[source].get(name) != row.get(name) for name in FINGERPRINTS):
                        raise ValueError("Historical source fingerprints changed")
                    records[source] = row
                for name in FINGERPRINTS:
                    if row.get(name):
                        hashes[name].add(row[name])
        blocked.update(index.get("processed_source_ids", []))
        blocked.update(exposure["source_ids"])
        hashes["sgf_sha256"].update(exposure["sgf_hashes"])
        hashes["audit_sequence_sha256"].update(exposure["sequence_hashes"])
    return records, blocked, hashes


def validate_fingerprints(inventory, index, forbidden, *, opaque_sources=()):
    if not (set(inventory) | set(forbidden)) <= index.keys():
        raise ValueError("Missing historical fingerprint evidence")
    available = set(inventory) - set(forbidden)
    missing_sealed_trajectories = 0
    for name in FINGERPRINTS:
        missing = {source for source in forbidden if not index[source].get(name)}
        if (any(not index[source].get(name) for source in available)
                or missing and (name != "trajectory_sha256" or not missing <= set(opaque_sources))):
            raise ValueError("Missing source fingerprint: " + name)
        if name == "trajectory_sha256":
            missing_sealed_trajectories = len(missing)
        barrier = {index[source][name] for source in forbidden if index[source].get(name)}
        values = [index[source][name] for source in available]
        if len(values) != len(set(values)) or set(values).intersection(barrier):
            raise ValueError("Validation duplicate or forbidden fingerprint: " + name)
    return {"missing_sealed_trajectory_fingerprints": missing_sealed_trajectories,
            "sealed_sgf_and_audit_sequence_fingerprints_complete": True}


def original_sides(base, allowed, inventory, *, check_budget):
    manifest = read(base["examples"])
    sides, counts, color_counts, keys = defaultdict(dict), Counter(), Counter(), set()
    columns = ["source_url", "mover", "source_rank_mover", "group_mover", "role", "canonical_sha256", "row_key"]
    for shard in manifest["artifacts"][DEVELOPMENT]:
        check_budget()
        filename = path(base["examples"]).parent / shard["metadata"]
        if file_hash(filename) != shard["metadata_sha256"]:
            raise ValueError("Validation metadata differs from its pinned shard")
        rows = pq.read_table(filename, columns=columns).to_pylist()
        if len(rows) != shard["rows"]:
            raise ValueError("Validation metadata row count changed")
        for row in rows:
            source, color = row["source_url"], row["mover"]
            if (source not in allowed or row["role"] != DEVELOPMENT or color not in ("B", "W")
                    or row["canonical_sha256"] != allowed[source]["canonical_sha256"]
                    or row["group_mover"] != allowed[source]["groups"][("B", "W").index(color)]
                    or row["row_key"] in keys):
                raise ValueError("Validation metadata crosses roles or changes complete-game identity")
            value = (row["source_rank_mover"], row["group_mover"])
            if color in sides[source] and sides[source][color] != value:
                raise ValueError("A game side changes rank or account")
            sides[source][color] = value
            counts[source] += 1
            color_counts[source, color] += 1
            keys.add(row["row_key"])
    if (counts != Counter({source: member["policy_actions"] for source, member in allowed.items()})
            or set(sides) != set(inventory)
            or any(inventory[source]["rows"] != counts[source]
                   or set(inventory[source]["ranks"]) != {rank for rank, _ in sides[source].values()}
                   or [color_counts[source, color] for color in ("B", "W")] != member["actions_by_color"]
                   for source, member in allowed.items())):
        raise ValueError("Inventory differs from complete validation metadata")
    return dict(sides)


def reviewed_supplements(refs, accepted, parent_ref, fitting_ref):
    summary, request = read(refs["source_summary"]), read(refs["source_request"])
    review, phase, exposure = read(refs["source_review"]), read(refs["source_phase"]), read(refs["source_exposure"])
    if (summary["status"] != "completed" or summary["yield_status"] != "sufficient_yield"
            or summary["accepted"] != accepted or summary["accepted_games"] != len(accepted)
            or summary["usable_games"] != len(accepted) or summary["request"] != refs["source_request"]
            or summary["training_performed"] or summary["model_queries"] or summary["protected_targets_decoded"]
            or not same_ref(request["inputs"]["parent_summary"], parent_ref)
            or not same_ref(request["inputs"]["grown_summary"], fitting_ref)
            or review["request_sha256"] != refs["source_request"]["sha256"]
            or (review["model"], review["reasoning"], review["verdict"]) !=
                ("gpt-6-astra", "max", "no significant findings")
            or phase["status"] != "completed" or read(phase["source_summary"])["source_summary"] != refs["source_summary"]
            or any(unit["source_url"] not in exposure["source_ids"]
                   or unit["sgf_sha256"] not in exposure["sgf_hashes"]
                   or unit["audit_sequence_sha256"] not in exposure["sequence_hashes"] for unit in accepted)):
        raise ValueError("Supplemental source admission or independent review is incomplete")


def supplemental_metadata(accepted, plan, blocked, hashes, *, check_budget):
    units = unique(accepted, "supplement")
    records, inventory, sides = {}, {}, {}
    seen = {name: set() for name in FINGERPRINTS}
    for source, unit in units.items():
        check_budget()
        member = unit["membership"]
        if (source in blocked or unit["status"] != "canonical" or member["source_url"] != source
                or member["role"] != DEVELOPMENT or member["exclusions"] or not member["final_excluded"]
                or member["duplicate_representative"] != source or member["duplicate_sources"] != [source]
                or set(member["groups"]).intersection(plan["test_groups"])
                or not set(member["groups"]).intersection(plan["validation_groups"])
                or not plan["train_before"] <= member["date"] < plan["confirmation_from"]):
            raise ValueError("Supplement is outside its unsealed development membership")
        for name in FINGERPRINTS:
            value = unit[name]
            if not value or value in hashes[name] or value in seen[name]:
                raise ValueError("Supplement has a historical or repeated fingerprint: " + name)
            seen[name].add(value)
        record = read(unit["canonical"])
        metadata, gate, actions = record["metadata"], record["eligibility"], record["actions"]
        colors = [sum(action["color"] == color for action in actions) for color in ("B", "W")]
        actual = {"sgf_sha256": record["source"]["sgf_sha256"],
                  "audit_sequence_sha256": record["audit"].get("sequence_fingerprint"),
                  "trajectory_sha256": record["trajectory_sha256"]}
        if (record["source"]["source_url"] != source or member["canonical_sha256"] != digest(record)
                or record.get("sealed_final") is not False or gate["training_eligible"] is not True
                or member["groups"] != [gate[side + "_group"] for side in ("black", "white")]
                or member["identity_valid"] != [gate[side + "_identity_valid"] for side in ("black", "white")]
                or any(member["identity_valid"]) or member["adaptation_colors"] or member["known_persona_colors"]
                or not actions or member["actions"] != len(actions) or member["policy_actions"] != len(actions)
                or member["actions_by_color"] != colors or member["date"] != metadata["DT"]
                or member["value_actions"] != (len(actions) if record["recorded_value_supported"] else 0)
                or not record["replay_validated"] or record["replay_pending_cap"] is not None
                or metadata["KM"] != 6.5 or metadata["RU"] != "Japanese"
                or record["dynamic_clock_input_valid"] or record["duration_valid"]
                or any(actual[name] != unit[name] for name in FINGERPRINTS)):
            raise ValueError("Supplement differs from its admitted complete canonical game")
        sides[source] = {color: (metadata[field], member["groups"][index])
                        for index, (color, field) in enumerate((("B", "BR"), ("W", "WR"))) if colors[index]}
        ranks = sorted({rank for rank, _ in sides[source].values()})
        inventory[source] = {"source_url": source, "ranks": ranks, "rows": len(actions), "komi_halfpoints": 13}
        records[source] = record
    return units, records, inventory, sides


def verify_reused(ref, purpose, base, members, ranks, *, expected_sources=None, expected_rows=None):
    manifest = read(ref)
    rows, sources, role = manifest["rows"], manifest["source_ids"], manifest["role"]
    if role not in {"train", "population_development", DEVELOPMENT}:
        raise ValueError("Reused diagnostic has a protected role")
    allowed = unique(members[role], role)
    if (manifest["version"] != "continuation_diagnostic_v1" or manifest["purpose"] != purpose
            or not manifest["inputs_are_normalized"] or manifest["normalizer_sha256"] != base["normalizer"]["sha256"]
            or not base.keys() <= manifest["data"].keys()
            or any(not same_ref(manifest["data"][key], ref) for key, ref in base.items())
            or manifest["aggregate_ranks"] != list(ranks) or len(sources) != len(set(sources))
            or not set(sources) <= allowed.keys() or (role == "train") != (purpose == "train_monitor")
            or any(allowed[source]["exclusions"] for source in sources)
            or Counter(row["source_url"] for row in rows) != Counter({source: allowed[source]["policy_actions"] for source in sources})
            or len({row["row_key"] for row in rows}) != len(rows) or any(row["role"] != role for row in rows)
            or manifest["excluded_rank_rows"] != sum(row["source_rank_mover"] not in ranks for row in rows)
            or expected_sources is not None and set(sources) != set(expected_sources)):
        raise ValueError("Reused diagnostic changed fitting population, membership or normalization")
    if expected_rows is not None:
        frozen = {row["row_key"]: row for row in expected_rows}
        fields = ("canonical_sha256", "sgf_sha256", "state_sha256", "source_url", "mover", "ply", "komi_halfpoints", "rank_hidden")
        if (len(frozen) != len(expected_rows) or set(frozen) != {row["row_key"] for row in rows}
                or any(any(row[key] != frozen[row["row_key"]][key] for key in fields) for row in rows)):
            raise ValueError("Fixed assessment row provenance changed")
    if file_hash(path(manifest["arrays"])) != manifest["arrays"]["sha256"]:
        raise ValueError("Reused diagnostic arrays changed")
    return manifest


def _context(inputs, parameters, *, fixture_only=False, check_budget=lambda: None):
    base = parameters["base_inputs"]
    ranks, per_cell = tuple(parameters.get("ranks", REVIEW_RANKS)), parameters.get("per_cell", PER_CELL)
    if not fixture_only and (ranks != REVIEW_RANKS or per_cell != PER_CELL):
        raise ValueError("Production validation requires the frozen rank scope and quota")
    if "normalizer" in inputs and not same_ref(inputs["normalizer"], base["normalizer"]):
        raise ValueError("Preparation normalizer differs from its original fit")
    previous = read(inputs["previous_preparation"])
    request = read(previous["composite"]["request"])
    refs = request["inputs"]
    if (previous["schema"] != "progressive_preparation_v1" or previous["status"] != "completed"
            or previous["protected_targets_decoded"] or not same_ref(previous["validation_inventory"], inputs["inventory"])
            or not same_ref(previous["composite"]["selection"], inputs["prior_selection"])
            or not same_ref(previous["fixed_comparison"], inputs["fixed_assessment"])
            or not same_ref(previous["supplemental_sources"]["accepted"], inputs["supplements"])
            or not same_ref(refs["source_accepted"], inputs["supplements"])
            or not same_ref(refs["grown_summary"], inputs["fitting_summary"])
            or not same_ref(refs["fixed_comparison"], inputs["fixed_assessment"])
            or not same_bundle(base, read(refs["original_config"])["parameters"]["base_inputs"])):
        raise ValueError("Preparation differs from the reviewed original data lineage")
    parent, grown = read(refs["parent_summary"]), read(inputs["fitting_summary"])
    for key, artifact in (("examples", "examples_manifest"), ("split", "split_manifest"),
                          ("normalizer", "normalizer"), ("canonical", "canonical")):
        if not same_ref(base[key], parent["artifacts"][artifact]):
            raise ValueError("Original normalization fitting inputs changed")
    fitting = {key: grown["artifacts"][artifact] for key, artifact in
               (("examples", "examples_manifest"), ("split", "split_manifest"), ("profile", "profile"))}
    fitting.update(normalizer=base["normalizer"], normalization_examples=base["examples"], normalization_split=base["split"])
    grown_examples, normalizer = validate_lineage(fitting, fixture_only=fixture_only)
    examples = read(base["examples"])
    split, members = read_membership(path(base["split"]).parent, base["split"]["sha256"])
    grown_split, grown_members = read_membership(path(fitting["split"]).parent, fitting["split"]["sha256"])
    original = unique(members[DEVELOPMENT], "original validation")
    if any(member["exclusions"] or member["role"] != DEVELOPMENT or member["policy_actions"] <= 0 for member in original.values()):
        raise ValueError("Original validation contains excluded or protected memberships")
    index, historical_sources, historical_hashes = historical_indexes((parent, grown))
    accepted = read(inputs["supplements"])["games"]
    reviewed_supplements(refs, accepted, refs["parent_summary"], inputs["fitting_summary"])
    sealed = sealed_sources(split) | sealed_sources(grown_split) | sealed_sources(examples) | sealed_sources(grown_examples)
    protected = sealed | {row["source_url"] for role in PROTECTED for row in grown_members[role]}
    units, records, supplemental, extra_sides = supplemental_metadata(
        accepted, split["plan"], historical_sources | protected, historical_hashes, check_budget=check_budget)
    inventory = unique(read(inputs["inventory"]), "combined inventory")
    original_inventory = unique(read(refs["original_inventory"]), "original inventory")
    if (set(original_inventory) != set(original) or set(original_inventory).intersection(supplemental)
            or inventory != original_inventory | supplemental):
        raise ValueError("Combined inventory differs from admitted original and supplemental games")
    sides = original_sides(base, original, original_inventory, check_budget=check_budget) | extra_sides
    fixed, assessments = read(inputs["fixed_assessment"]), previous["assessments"]
    expected_assessments = {role + "_komi" + str(komi) for role in ("population_development", DEVELOPMENT) for komi in (1, 13)}
    if set(assessments) != expected_assessments or {item["spec"]["id"] for item in fixed["common_sets"]} != expected_assessments:
        raise ValueError("All four fixed assessment populations are required")
    fixed_sources, fixed_rows = set(), 0
    for common in fixed["common_sets"]:
        check_budget()
        spec, frozen = common["spec"], read(common["manifest"])
        if (frozen["inputs"]["examples_sha256"] != base["examples"]["sha256"]
                or frozen["inputs"]["normalizer_sha256"] != base["normalizer"]["sha256"]):
            raise ValueError("Fixed assessment normalization changed")
        saved = verify_reused(assessments[spec["id"]], "assessment", base, members, ranks,
                             expected_sources=spec["source_ids"], expected_rows=frozen["rows"])
        if saved["role"] != spec["role"] or fixed_sources.intersection(saved["source_ids"]):
            raise ValueError("Fixed assessment roles or source membership overlap")
        fixed_sources.update(saved["source_ids"])
        fixed_rows += len(saved["rows"])
    verify_reused(previous["train_monitor"], "train_monitor", base, members, ranks)
    prior, prior_diagnostic = read(inputs["prior_selection"]), read(previous["epoch_validation"])
    prior_sources = set(prior["source_ids"])
    if (len(prior_sources) != len(prior["source_ids"]) or prior_sources != set(prior_diagnostic["source_ids"])
            or not prior_sources <= inventory.keys() or prior_sources.intersection(fixed_sources)
            or prior_diagnostic["purpose"] != "epoch_validation" or prior_diagnostic["role"] != DEVELOPMENT
            or prior_diagnostic["normalizer_sha256"] != base["normalizer"]["sha256"]):
        raise ValueError("Previous stopping membership or normalization changed")
    forbidden = protected | fixed_sources | prior_sources | {row["source_url"]
        for role in ("train", "population_development") for row in grown_members[role]}
    index.update(units)
    fingerprint_evidence = validate_fingerprints(inventory, index, forbidden, opaque_sources=sealed)
    selection = select_games(list(inventory.values()), sides, excluded=forbidden, ranks=ranks, per_cell=per_cell)
    selection["original_source_ids"] = sorted(set(selection["source_ids"]) & original.keys())
    selection["supplemental_source_ids"] = sorted(set(selection["source_ids"]) & units.keys())
    complete_members = original | {source: unit["membership"] for source, unit in units.items()}
    selection["excluded_rank_rows"] = sum(count for source in selection["source_ids"]
        for color, count in zip(("B", "W"), complete_members[source]["actions_by_color"])
        if count and sides[source][color][0] not in ranks)
    counts = {"original_games": len(original), "supplemental_games": len(units),
              "fitting_games": len(grown_members["train"]), "fitting_rows": grown_examples["counts"]["train"]["examples"],
              "fixed_games": len(fixed_sources), "fixed_rows": fixed_rows, "prior_stopping_games": len(prior_sources)}
    if not fixture_only and (counts != {"original_games": 2254, "supplemental_games": 7, "fitting_games": 18126,
            "fitting_rows": 3999997, "fixed_games": 539, "fixed_rows": 120450, "prior_stopping_games": 52}
            or fingerprint_evidence["missing_sealed_trajectory_fingerprints"] != 3
            or selection["source_ids_sha256"] != SELECTION_SHA256
            or (selection["unique_games"], selection["expected_rows"], selection["excluded_rank_rows"],
                selection["available_games"], selection["available_rows"], selection["accounts"]) !=
                (577, 125998, 1147, 1948, 432918, 578)):
        raise ValueError("Production validation selection differs from its frozen metadata result")
    return {"base": base, "normalizer": normalizer, "plan": split["plan"], "example_config": examples["config"],
            "inputs": inputs, "parameters": parameters, "previous": previous, "units": units, "records": records,
            "members": complete_members, "selection": selection, "counts": counts, "ranks": ranks,
            "fingerprint_evidence": fingerprint_evidence}


def preflight(inputs, parameters, *, fixture_only=False, check_budget=lambda: None):
    context = _context(inputs, parameters, fixture_only=fixture_only, check_budget=check_budget)
    return {"status": "metadata_verified", "selection": context["selection"], "counts": context["counts"],
            "fingerprint_evidence": context["fingerprint_evidence"],
            "protected_targets_decoded": False, "model_queries": 0, "training_performed": False}


def supplemental_arrays(context, *, check_budget):
    raw = []
    for source in context["selection"]["supplemental_source_ids"]:
        check_budget()
        raw.extend(examples_for_game(context["records"][source], context["units"][source]["membership"],
            context["plan"], ExampleConfig(**context["example_config"]), komi_study=True, check_budget=check_budget))
    schema = {**array_schema(0), **LABEL_SCHEMA}
    arrays = {name: np.asarray([row[0 if name in INPUT_KEYS else 1][name] for row in raw], dtype=dtype)
                      .reshape((len(raw), *shape)) for name, (dtype, shape) in schema.items()}
    rows = []
    for inputs, _, metadata in raw:
        scalar = inputs["scalars"]
        clock = ([float(scalar[9]), float(scalar[10]), int(scalar[11])]
                 if inputs["scalar_valid"][9:12].all() else None)
        rows.append({**metadata, "nominal_clock": clock, "komi_halfpoints": int(round(2 * abs(float(scalar[1]))))})
    selected = context["selection"]["supplemental_source_ids"]
    if (len(rows) != sum(context["units"][source]["membership"]["policy_actions"] for source in selected)
            or int(arrays["policy_valid"].sum()) != len(rows)
            or int(arrays["value_valid"].sum()) != sum(context["units"][source]["membership"]["value_actions"] for source in selected)
            or arrays["duration_valid"].any()):
        raise ValueError("Supplemental whole-game labels or missing-data masks changed")
    arrays.update(apply_normalizer({name: arrays[name] for name in INPUT_KEYS}, context["normalizer"]))
    return arrays, rows


def prepare(inputs, parameters, output, *, fixture_only=False, check_budget=lambda: None):
    context = _context(inputs, parameters, fixture_only=fixture_only, check_budget=check_budget)
    selection, base, previous = context["selection"], context["base"], context["previous"]
    work = Path(output) / "validation"
    work.mkdir(parents=True, exist_ok=False)
    save_json(work / "selection.json", selection)
    selection_ref = reference(work / "selection.json")
    original_ref = prepare_diagnostic(base, work / "original", purpose="epoch_validation", role=DEVELOPMENT,
        source_ids=selection["original_source_ids"], aggregate_ranks=context["ranks"],
        excluded_source_ids=selection["excluded_source_ids"], check_budget=check_budget)
    original = read(original_ref)
    with np.load(path(original["arrays"]), allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in (*INPUT_KEYS, *LABEL_SCHEMA)}
    extra, extra_rows = supplemental_arrays(context, check_budget=check_budget)
    arrays = {name: np.concatenate([values, extra[name]]) for name, values in arrays.items()}
    rows = original["rows"] + extra_rows
    if (len(rows) != selection["expected_rows"] or len({row["row_key"] for row in rows}) != len(rows)
            or Counter(row["source_url"] for row in rows) != Counter({source: context["members"][source]["policy_actions"]
                                                                     for source in selection["source_ids"]})
            or sum(row["source_rank_mover"] not in context["ranks"] for row in rows) != selection["excluded_rank_rows"]):
        raise ValueError("Prepared rows differ from the frozen complete-game selection")
    check_budget()
    with (work / "arrays.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    manifest = {"version": "continuation_diagnostic_v1", "purpose": "epoch_validation", "role": DEVELOPMENT,
        "arrays": reference(work / "arrays.npz"), "rows": rows, "aggregate_ranks": list(context["ranks"]),
        "normalizer_sha256": base["normalizer"]["sha256"], "inputs_are_normalized": True,
        "source_ids": selection["source_ids"], "excluded_rank_rows": selection["excluded_rank_rows"],
        "data": {"original": base, "supplements": inputs["supplements"]},
        "preparation": {"inputs": inputs, "parameters": parameters, "selection": selection_ref,
            "original_component": original_ref, "supplemental_source_ids": selection["supplemental_source_ids"],
            "normalization": {"normalizer": base["normalizer"], "fitting_examples": base["examples"], "fitting_split": base["split"]}}}
    save_json(work / "manifest.json", manifest)
    diagnostic_ref = reference(work / "manifest.json")
    loaded = load_diagnostic(diagnostic_ref, "epoch_validation", base["normalizer"]["sha256"], fixture_only=fixture_only)
    if len(loaded.rows) != selection["expected_rows"]:
        raise ValueError("Diagnostic loader lost prepared rows")
    check_budget()
    return {"schema": "policy_preparation_v1", "status": "completed", "epoch_validation": diagnostic_ref,
            "train_monitor": previous["train_monitor"], "assessments": previous["assessments"],
            "selection": selection_ref, "inventory": inputs["inventory"], "fixed_assessment": inputs["fixed_assessment"],
            "counts": context["counts"], "fingerprint_evidence": context["fingerprint_evidence"],
            "selection_uses_targets_or_predictions": False,
            "protected_targets_decoded": False, "training_performed": False, "model_queries": 0}
