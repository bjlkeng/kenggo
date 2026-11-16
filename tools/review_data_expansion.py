"""Check a derived dataset and export one local review page."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
import math
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq

from kenggo.cached_coverage_audit import file_hash, read_ref, save_json
from kenggo.canonical_data import digest
from kenggo.data_splits import ROLES, read_membership
from kenggo.data_expansion import base_endings
from kenggo.encoding import NORMALIZED
from kenggo.examples import load_corpus, load_dataset_profile, read_shards, replay_record
from kenggo.go import PASS
from kenggo.kgs_source import RANKS
from kenggo.sgf_audit import result_signature


ROOT = Path(__file__).resolve().parents[1]
ALLOWED = ("train", "population_development", "validation_development")


def reference(path):
    return {"path": str(Path(path).resolve()), "sha256": file_hash(path)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def collection_details(source):
    if source.get("schema") == "e2_komi_reconsideration_sources_v1":
        parent = collection_details(read_ref(source["provenance"]["previous_sources"]))
        total = sum(source["counts"].values())
        return {"status": "completed", "stop_reason": None, "cached_only": True,
                "counts": {"frozen_candidates": total, "accessible_candidates": total, "query_statuses": {}},
                "unavailable_records": source["unavailable_records"], "diagnostics": parent["diagnostics"],
                "rounds": [], "prior_collection": parent}
    rounds = [source]
    previous = source.get("provenance", {}).get("previous")
    if previous is not None:
        parent = read_ref(previous["sources"])
        require(parent.get("schema") == source.get("schema")
                and not parent.get("provenance", {}).get("previous"),
                "Unsupported or repeated collection lineage")
        rounds.insert(0, parent)
    statuses = Counter()
    for item in rounds:
        statuses.update(item.get("counts", {}).get("query_statuses", {}))
    return {"status": source.get("status", "unreported"), "stop_reason": source.get("stop_reason"),
            "counts": {**source.get("counts", {}), "query_statuses": dict(statuses)},
            "diagnostics": [item for run in rounds for item in run.get("diagnostics", [])],
            "rounds": [{"status": run.get("status"),
                        "query_statuses": run.get("counts", {}).get("query_statuses", {})}
                       for run in rounds]}


def read_scope_audit(audit_ref, request_ref, source_ref, summary):
    if audit_ref is None:
        return None
    audit = read_ref(audit_ref)
    require(audit.get("schema") == "e2_sgf_scope_audit_v1"
            and audit.get("status") == "complete"
            and audit["inputs"]["build_request"]["sha256"] == request_ref["sha256"]
            and audit["inputs"]["sources"]["sha256"] == source_ref["sha256"],
            "Scope audit belongs to another run or is incomplete")
    total = audit["aggregate"]
    require(total["records"] == audit["scope"]["processed_records"]
            == audit["scope"]["expected_records"] == summary["source_completed"]
            and total["status"].get("canonical", 0) == summary["new_source_admitted"]
            and total["exclusions"] == summary["source_exclusion_reasons"],
            "Scope audit and completed source screening disagree")

    def half_komi_only(group):
        return sum(cell["records"] for cell in group["scope_only_cells"]
                   if cell["RU"] == "Japanese" and cell["KM"] == .5
                   and cell["recorded_exclusions"] == "Only rules or komi excluded")

    return {"reference": audit_ref, "records": total["records"],
            "rules_or_komi_excluded": total["exclusions"].get("metadata:rules_or_komi_out_of_scope", 0),
            "japanese_half_komi": sum(cell["records"] for cell in total["rules_komi"]
                                      if cell["RU"] == "Japanese" and cell["KM"] == .5),
            "missing_rules_or_komi": total["rules_metadata"].get("Missing rules or komi", 0),
            "clock_metadata": total["clock_metadata"],
            "training_half_komi_only": half_komi_only(audit["by_role"]["train"]),
            "training_ranks": [{"rank": rank, "candidates": group["records"],
                                "passed_source_checks": group["status"].get("canonical", 0),
                                "half_komi_only": half_komi_only(group),
                                "clock_only": group["recorded_exclusion_groups"].get("Only clock family excluded", 0)}
                               for rank in RANKS if (group := audit["training_by_rank"].get(rank)) is not None],
            "costs": audit["costs"]}


def inspect(request, summary, check_budget):
    refs = summary["artifacts"]
    base_split, base_parts = read_membership(
        Path(request["refs"]["split_manifest"]["path"]).parent,
        request["refs"]["split_manifest"]["sha256"])
    split, parts = read_membership(Path(refs["split_manifest"]["path"]).parent,
                                   refs["split_manifest"]["sha256"])
    profile = load_dataset_profile(refs["profile"]["path"], refs["profile"]["sha256"])
    base_profile = load_dataset_profile(request["refs"]["profile"]["path"],
                                        request["refs"]["profile"]["sha256"])
    require(not base_profile.komi_study, "This comparison requires the preserved 6.5-komi baseline")
    examples = read_ref(refs["examples_manifest"])
    normalizer = read_ref(refs["normalizer"])
    old_examples = read_ref(request["refs"]["examples_manifest"])
    require(profile.canonical_sha256 == refs["canonical"]["sha256"]
            and profile.split_sha256 == refs["split_manifest"]["sha256"]
            and examples["canonical_sha256"] == profile.canonical_sha256
            and examples["split_sha256"] == profile.split_sha256
            and examples["dataset_profile_sha256"] == profile.sha256
            and normalizer["examples_sha256"] == refs["examples_manifest"]["sha256"],
            "Dataset artifact pins disagree")
    require(file_hash(refs["canonical"]["path"]) == profile.canonical_sha256,
            "Canonical artifact differs from its pin")
    require(split["sealed_source_ids"] == base_split["sealed_source_ids"]
            and examples["sealed_source_ids"] == old_examples["sealed_source_ids"],
            "The sealed source registry changed")
    for key in ("train_before", "confirmation_from", "validation_groups", "test_groups"):
        require(split["plan"][key] == base_split["plan"][key], "The old split policy changed")
    old_rows = {r["source_url"]: r for rows in base_parts.values() for r in rows}
    rows = {r["source_url"]: r for values in parts.values() for r in values}
    ending_counts = {role: Counter() for role in ALLOWED}
    ending_ids = set()
    recorded_outcomes = {}
    recorded_komi = {}
    recorded_endings = {}
    recorded_actions = {}
    recorded_ranks = {}

    def add_ending(key, audit, komi=6.5):
        kind = audit.get("result_kind")
        require(key not in ending_ids and kind in
                {"scored", "resignation", "timeout", "forfeit", "incomplete_or_unknown", "draw", "unrecognized"}
                and type(audit.get("final_two_explicit_passes")) is bool, "Ending evidence is missing or duplicated")
        parsed_kind, winner, _ = result_signature(audit["metadata"]["RE"])
        require(parsed_kind == kind, "Ending type differs from the recorded result")
        require(not rows[key]["value_actions"] or kind in {"scored", "resignation"},
                "An unsupported recorded outcome supplies value labels")
        recorded_outcomes[key] = 1 if winner == "B" else -1 if winner == "W" else None
        require(komi == 6.5 or profile.komi_study and komi == .5,
                "A recorded game has an unsupported komi")
        recorded_komi[key] = komi
        recorded_endings[key] = (kind, audit["final_two_explicit_passes"])
        count = ending_counts[rows[key]["role"]]
        count[kind] += 1
        count["two_explicit_pass_endings"] += audit["final_two_explicit_passes"]
        ending_ids.add(key)

    require(len(rows) == sum(map(len, parts.values())), "A source has more than one role")
    require(all(rows.get(key) == row for key, row in old_rows.items()), "An old membership changed")
    for role in set(ROLES) - set(ALLOWED) - {"excluded"}:
        require(parts[role] == base_parts[role], "An untouched role changed")
    for name, ref in (("old", request["refs"]["canonical"]), ("new", refs["canonical"])):
        table = pq.read_table(ref["path"], columns=["source_url", "canonical_sha256"])
        headers = {r["source_url"]: r["canonical_sha256"] for r in table.to_pylist()}
        require(len(headers) == table.num_rows, "Canonical source IDs are duplicated")
        if name == "old":
            old_headers = headers
        else:
            require(all(headers.get(k) == v for k, v in old_headers.items()), "An old canonical hash changed")
            require(headers == {k: r["canonical_sha256"] for k, r in rows.items()},
                    "Canonical and membership rows differ")
            for batch in pq.ParquetFile(ref["path"]).iter_batches(batch_size=64):
                check_budget()
                for row in batch.to_pylist():
                    require(sha256(row["canonical_json"].encode()).hexdigest() == row["canonical_sha256"],
                            "An opaque canonical record differs from its hash")
                    key = row["source_url"]
                    if rows[key]["role"] in ALLOWED and (key not in old_rows or profile.komi_study):
                        record = json.loads(row["canonical_json"])
                        require(record["source"]["source_url"] == key and record["sealed_final"] is False,
                                "Canonical evidence belongs to a different or sealed game")
                        require(record["replay_validated"] is True and not record["structural_exclusions"]
                                and len(record["actions"]) == rows[key]["actions"]
                                and record["eligibility"]["training_eligible"] is True
                                and rows[key]["groups"] == [record["eligibility"][color + "_group"]
                                                           for color in ("black", "white")],
                                "Canonical trajectory lacks complete legal replay or matching membership")
                        recorded_actions[key] = [(move["color"], move["action"]) for move in record["actions"]]
                        recorded_ranks[key] = [record["metadata"][field] for field in ("BR", "WR")]
                        if key not in old_rows:
                            add_ending(key, record["audit"], record["metadata"]["KM"])
    for key, audit in base_endings(request, list(old_rows.values())).items():
        check_budget()
        add_ending(key, audit)
    require(ending_ids == {key for key, row in rows.items() if row["role"] in ALLOWED},
            "Ending coverage does not include every allowed game")
    require(not profile.komi_study or set(recorded_actions) == ending_ids,
            "The study must compare move labels against every allowed canonical game")
    directory = Path(refs["examples_manifest"]["path"]).parent
    copied = 0
    for role in ROLES:
        for index, item in enumerate(old_examples["artifacts"][role]):
            check_budget()
            require(examples["artifacts"][role][index] == item, "An old shard reference changed")
            for kind in ("arrays", "metadata"):
                require(file_hash(directory / item[kind]) == item[kind + "_sha256"], "An old shard changed")
                copied += 1
        if role not in ALLOWED:
            require(examples["artifacts"][role] == old_examples["artifacts"][role],
                    "Examples were added to an untouched role")
    excluded = {r["source_url"] for r in read_ref(request["refs"]["owner_exclusions"])["excluded_games"]}
    reserved = set(split["plan"]["test_groups"])
    validation = set(split["plan"]["validation_groups"])
    coverage = {}
    komi_coverage = {"6.5": {}, "0.5": {}}
    total_checked = 0
    moments = {i: [0, 0.0, 0.0] for i in NORMALIZED}
    selected = {}
    for role in ALLOWED:
        membership = {r["source_url"]: r for r in parts[role]}
        seen, outcomes = {}, {}
        cells = defaultdict(lambda: {"moves": 0, "passes": 0, "sides": set(), "accounts": set()})
        rank_cells = defaultdict(lambda: {"moves": 0, "passes": 0, "sides": set(), "accounts": set()})
        nominal_clocks = defaultdict(lambda: {"moves": 0, "passes": 0, "sides": set(), "accounts": set()})
        all_sides, all_accounts = set(), set()
        opponent_ranks, rank_pairs = Counter(), Counter()
        count = Counter()
        komi_groups = {label: {"counts": Counter(), "sides": set(), "accounts": set(),
            "games": set(), "ranks": defaultdict(lambda: {"moves": 0, "passes": 0, "sides": set(), "accounts": set()}),
            "cells": defaultdict(lambda: {"moves": 0, "passes": 0, "sides": set(), "accounts": set()})}
            for label in komi_coverage}
        shard_options = {"komi_study": True} if profile.komi_study else {}
        for arrays, metadata in read_shards(directory, refs["examples_manifest"]["sha256"], role, **shard_options):
            check_budget()
            n = len(metadata)
            require(np.all(arrays["policy_valid"]), "A recorded policy row is masked")
            require(np.all((arrays["action"] >= 0) & (arrays["action"] <= PASS))
                    and np.all(arrays["legal"][np.arange(n), arrays["action"]]), "An action label is illegal")
            require(not arrays["duration_valid"].any() and not arrays["scalar_valid"][:, 12:].any(),
                    "Unverified timing became visible")
            require(np.all(arrays["value"][~arrays["value_valid"]] == 0)
                    and np.all(np.isin(arrays["value"][arrays["value_valid"]], [-1, 1])),
                    "Value labels do not follow the missing-result contract")
            require(not (arrays["board"][:, 0] & arrays["board"][:, 1]).any(), "Board planes overlap")
            if role == "train":
                for index in NORMALIZED:
                    values = arrays["scalars"][arrays["scalar_valid"][:, index], index].astype(np.float64)
                    moments[index][0] += len(values)
                    moments[index][1] += float(values.sum())
                    moments[index][2] += float(np.square(values).sum())
            count.update(examples=n, value_valid=int(arrays["value_valid"].sum()),
                         passes=int((arrays["action"] == PASS).sum()))
            for i, meta in enumerate(metadata):
                key, ply = meta["source_url"], meta["ply"]
                require(key in membership and key not in excluded, "Unexpected or owner-excluded source")
                member = membership[key]
                require(type(ply) is int and 0 <= ply < member["actions"], "Move index is out of range")
                require(not seen.get(key, 0) & (1 << ply), "A game move is duplicated")
                seen[key] = seen.get(key, 0) | (1 << ply)
                require(meta["row_key"] == digest([key, ply])
                        and meta["canonical_sha256"] == member["canonical_sha256"]
                        and arrays["scalars"][i, 5] == ply
                        and bool(arrays["scalars"][i, 0]) == (meta["mover"] == "B"),
                        "Example provenance differs from its game")
                if key in recorded_actions:
                    require((meta["mover"], int(arrays["action"][i])) == recorded_actions[key][ply],
                            "Example label or mover differs from its canonical move")
                    side = 0 if meta["mover"] == "B" else 1
                    require(meta["group_mover"] == member["groups"][side]
                            and meta["group_opponent"] == member["groups"][1 - side]
                            and meta["source_rank_mover"] == recorded_ranks[key][side]
                            and meta["source_rank_opponent"] == recorded_ranks[key][1 - side],
                            "Coverage metadata differs from its canonical players")
                expected_komi = recorded_komi[key] * (-1 if meta["mover"] == "B" else 1)
                require(bool(arrays["scalar_valid"][i, 1])
                        and float(arrays["scalars"][i, 1]) == expected_komi,
                        "Komi input differs from the recorded game and mover")
                require(not meta["identity_valid"] and not meta["adaptation_allowed"]
                        and meta["persona_mode"] == "default", "Unsupported identity became usable")
                require(not reserved.intersection(member["groups"])
                        and (role != "train" or not validation.intersection(member["groups"])),
                        "A reserved account leaked into an allowed role")
                require(bool(arrays["value_valid"][i]) == (member["value_actions"] > 0),
                        "A value mask differs from its complete game")
                if arrays["value_valid"][i]:
                    outcome = float(arrays["value"][i]) * (1 if meta["mover"] == "B" else -1)
                    require(outcome == recorded_outcomes[key], "Value label differs from the recorded winner")
                    require(outcomes.setdefault(key, outcome) == outcome, "Value perspective changes within a game")
                rank = meta["source_rank_mover"]
                opponent_rank = meta["source_rank_opponent"]
                require(rank in RANKS and opponent_rank in RANKS,
                        "A source rank is outside the approved vocabulary")
                opponent_ranks[opponent_rank] += 1
                rank_pairs[(rank, opponent_rank)] += 1
                all_sides.add((key, meta["mover"]))
                all_accounts.add(meta["group_mover"])
                nominal = tuple(float(arrays["scalars"][i, j]) if arrays["scalar_valid"][i, j] else None
                                for j in (9, 10, 11))
                phase = "1–60" if ply < 60 else "61–180" if ply < 180 else "181+"
                clock = "10–<20 min" if arrays["scalars"][i, 9] < 1200 else "20–30 min"
                group = komi_groups[str(recorded_komi[key])]
                group["games"].add(key)
                group["sides"].add((key, meta["mover"]))
                group["accounts"].add(meta["group_mover"])
                group["counts"].update(examples=1, passes=int(arrays["action"][i] == PASS),
                                       value_valid=int(arrays["value_valid"][i]))
                for cell in (cells[(rank, phase, clock)], rank_cells[rank], nominal_clocks[nominal],
                             group["ranks"][rank], group["cells"][(rank, phase, clock)]):
                    cell["moves"] += 1
                    cell["passes"] += int(arrays["action"][i] == PASS)
                    cell["sides"].add((key, meta["mover"]))
                    cell["accounts"].add(meta["group_mover"])
        require(set(seen) == set(membership), "A complete game is missing from the examples")
        require(all(mask == (1 << membership[key]["actions"]) - 1 for key, mask in seen.items()),
                "A game was only partly encoded")
        for key in ("examples", "value_valid", "passes"):
            require(count[key] == examples["counts"][role][key], "Example counts disagree")
        require(count["examples"] == sum(r["policy_actions"] for r in membership.values()),
                "Move count differs from split membership")
        def cell_counts(cell):
            sides, accounts = len(cell["sides"]), len(cell["accounts"])
            return {"moves": cell["moves"], "passes": cell["passes"], "game_sides": sides,
                    "games": len({key for key, _ in cell["sides"]}),
                    "accounts": accounts, "enough_data": sides >= 100 and accounts >= 30}
        cell_keys = [(rank, phase, clock) for rank in RANKS
                     for phase in ("1–60", "61–180", "181+") for clock in ("10–<20 min", "20–30 min")]
        coverage[role] = {"games": len(membership), "game_sides": len(all_sides),
            "accounts": len(all_accounts), **count, "endings": dict(ending_counts[role]),
            "ranks": {r: cell_counts(rank_cells[r]) for r in RANKS},
            "opponent_ranks": {r: opponent_ranks[r] for r in RANKS},
            "rank_pairs": [{"mover_rank": r, "opponent_rank": o, "moves": rank_pairs[(r, o)]}
                           for r in RANKS for o in RANKS if rank_pairs[(r, o)]],
            "nominal_clocks": [{"main_seconds": key[0], "period_seconds": key[1], "periods": key[2],
                                **cell_counts(nominal_clocks[key])}
                               for key in sorted(nominal_clocks, key=json.dumps)],
            "cells": [{"rank": r, "phase": p, "clock": c, **cell_counts(cells[(r, p, c)])}
                      for r, p, c in cell_keys]}
        for label, group in komi_groups.items():
            endings = Counter()
            for key in group["games"]:
                kind, two_passes = recorded_endings[key]
                endings[kind] += 1
                endings["two_explicit_pass_endings"] += two_passes
            komi_coverage[label][role] = {"games": len(group["games"]),
                "game_sides": len(group["sides"]), "accounts": len(group["accounts"]),
                "endings": dict(endings),
                **{name: group["counts"][name] for name in ("examples", "passes", "value_valid")},
                "ranks": {r: cell_counts(group["ranks"][r]) for r in RANKS},
                "cells": [{"rank": r, "phase": p, "clock": c,
                           **cell_counts(group["cells"][(r, p, c)])} for r, p, c in cell_keys]}
        total_checked += count["examples"]
        fresh = sorted(set(membership) - set(old_rows), key=lambda key: digest(["e2-expansion-review-v1", key]))
        selected[role] = fresh[:2 if role == "train" else 1]
        if profile.komi_study:
            old = sorted(set(membership) & set(old_rows), key=lambda key: digest(["e2-komi-control-v1", key]))
            selected[role] = old[:1] + selected[role]
    require(normalizer["fit_role"] == "train" and normalizer["fitted"], "Normalizer was not fitted on train")
    for i, (count, total, squared) in moments.items():
        mean = total / count if count else 0.0
        scale = max(squared / max(count, 1) - mean * mean, 0) ** .5 or 1.0
        require(normalizer["count"][i] == count
                and np.isclose(normalizer["mean"][i], mean, rtol=1e-10, atol=1e-8)
                and np.isclose(normalizer["scale"][i], scale, rtol=1e-7, atol=1e-7),
                "Independent training statistics do not reproduce normalization")
    samples = []
    for role, keys in selected.items():
        if not keys:
            continue
        check_budget()
        records, _, _ = load_corpus(refs["canonical"]["path"], profile.canonical_sha256,
            Path(refs["split_manifest"]["path"]).parent, profile.split_sha256,
            profile=profile, role=role, source_ids=keys)
        for key in keys:
            record = records[key]
            game = replay_record(record, komi_study=profile.komi_study, check_budget=check_budget)
            first_pass = next((i for i, m in enumerate(record["actions"]) if m["action"] == PASS), None)
            indices = {0, len(game.moves) // 2, len(game.moves) - 1}
            if first_pass is not None:
                indices.add(first_pass)
            samples.append({"source": key, "role": role, "ranks": [record["metadata"][p] for p in ("BR", "WR")],
                "komi": record["metadata"]["KM"], "cohort": "baseline" if key in old_rows else "addition",
                "result": record["metadata"]["RE"], "value_valid": record["recorded_value_supported"],
                "moves": [[m.color, m.action] for m in game.moves],
                "boards": ["".join(p.board) for p in game.positions],
                "inspect_plies": sorted(indices),
                "canonical_sha256": digest(record)})
    return {"checks": {"train_development_rows_checked": total_checked,
            "recorded_policy_labels_checked": sum(map(len, recorded_actions.values())),
            "unchanged_old_memberships": len(old_rows), "unchanged_old_files": copied,
            "complete_games": True, "legal_policy_labels": True, "value_masks_and_perspective": True,
            "komi_inputs_and_perspective": True,
            "normalizer_reproduced": True, "final_and_confirmation_targets_decoded": False},
            "coverage": coverage, "komi_coverage": komi_coverage,
            "komi_study": profile.komi_study, "samples": samples,
            "before": {role: {"games": len(base_parts[role]), **old_examples["counts"][role]} for role in ALLOWED},
            "sealed_final": {"games": len(base_parts["final"]),
                             "examples": old_examples["counts"]["final"]["examples"]}}


def _build(request_ref, summary_ref, source_ref, output, scope_audit_ref=None, review_cpu_cap=900):
    started, cpu = time.monotonic(), time.process_time()
    output = Path(output)
    if output.exists():
        raise FileExistsError("Keep the earlier review; use a new output directory")
    final_deadline, remaining_cpu = None, 900
    def check_budget():
        if time.process_time() - cpu >= review_cpu_cap or time.monotonic() - started > 1800:
            raise RuntimeError("Readout verification exceeded its fixed budget")
        if final_deadline is not None and (datetime.now(timezone.utc) >= final_deadline
                or time.process_time() - cpu >= remaining_cpu):
            raise RuntimeError("Readout verification exhausted the shared study allowance")
    request, summary, source = (read_ref(ref) for ref in (request_ref, summary_ref, source_ref))
    require(summary["request_sha256"] == request_ref["sha256"]
            and summary["sources_sha256"] == source_ref["sha256"]
            and source["request_sha256"] == request["source_request_sha256"],
            "The request, collection and build are from different runs")
    if request["schema"] == "e2_komi_expansion_request_v1":
        final_deadline = datetime.fromisoformat(request["deadline_utc"])
        remaining_cpu = request["limits"]["cpu_seconds"] - summary["costs"]["cpu_seconds"]
        check_budget()
    scope_audit = read_scope_audit(scope_audit_ref, request_ref, source_ref, summary)
    result = inspect(request, summary, check_budget)
    result.update(schema="e2_data_review_v1", bead=request.get("bead", "kenggo-ydo.8"), request=request_ref,
                  build_summary=summary_ref, source_summary=source_ref,
                  status=summary["status"], stop_reason=summary.get("stop_reason"),
                  build_costs=summary["costs"], source_costs=source.get("costs", {}),
                  collection=collection_details(source),
                  scope_audit=scope_audit,
                  preparation={key: summary[key] for key in (
                      "source_candidates", "source_attempted", "source_completed", "source_unattempted",
                      "source_exclusions", "source_exclusion_reasons",
                      "new_source_admitted", "new_replay_validated", "new_encoded_games", "new_split_exclusions")},
                  provenance=summary["artifacts"],
                  pilot={"parameters": 2410515, "epochs": 20, "batch_size": 128,
                         "learning_rate": .001, "train_games": 106, "train_moves": 22916,
                         "development_games": 24, "development_moves": 5926},
                  limitations=["No model was trained on these new data.",
                    "The account survey is a selected sample; it does not represent every KGS player.",
                    "Accounts passed bot screening; human control was not independently verified.",
                    "Data support does not establish playing strength or a calibrated human rank.",
                    "Move durations, dynamic clocks and personalization inputs remain masked."])
    check_budget()
    result["verification_costs"] = {"cpu_seconds": time.process_time() - cpu,
                                    "wall_seconds": time.monotonic() - started, "network_requests": 0, "gpu_seconds": 0}
    output.mkdir(parents=True)
    payload_pin = save_json(output / "readout.json", result)
    payload = json.dumps(result, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    template = ROOT / "tools/data_expansion_review.html"
    page = template.read_text().replace("REVIEW_DATA", payload)
    require(len(payload.encode()) + len(page.encode()) + 1024 * 1024 < 512 * 1024**2,
            "The review output exceeds its reserved 512 MiB")
    (output / "review.html").write_text(page)
    export = {"payload": {"path": str(output / "readout.json"), "sha256": payload_pin},
              "page": reference(output / "review.html"), "template": reference(template),
              "tool": reference(__file__), "checks": result["checks"]}
    save_json(output / "export.json", export)
    check_budget()
    print(json.dumps(export, indent=2))


def build(request_ref, summary_ref, source_ref, output, scope_audit_ref=None):
    started = time.process_time()
    request = read_ref(request_ref)
    if request["schema"] != "e2_komi_expansion_request_v1":
        return _build(request_ref, summary_ref, source_ref, output, scope_audit_ref)
    summary = read_ref(summary_ref)
    marker = Path(request["session_marker"]["path"])
    read_ref(request["session_marker"])
    ledger_path = marker.with_name(marker.name + ".verification.json")
    with marker.with_name(marker.name + ".writer.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {
            "request_sha256": request_ref["sha256"], "summary_sha256": summary_ref["sha256"], "attempts": []}
        require(ledger["request_sha256"] == request_ref["sha256"]
                and ledger["summary_sha256"] == summary_ref["sha256"],
                "Verification ledger belongs to another study or build")
        require(isinstance(ledger["attempts"], list) and all(
            row.get("status") in {"running", "failed", "completed"}
            and type(row.get("cpu_charged_seconds")) in (int, float)
            and math.isfinite(row["cpu_charged_seconds"]) and row["cpu_charged_seconds"] >= 0
            for row in ledger["attempts"]), "Verification charges are invalid")
        prior = sum(row["cpu_charged_seconds"] for row in ledger["attempts"])
        remaining = request["limits"]["cpu_seconds"] - summary["costs"]["cpu_seconds"] - prior
        reserved = min(900, remaining)
        require(reserved > time.process_time() - started + 1,
                "Verification has no remaining shared CPU allowance")
        require(datetime.now(timezone.utc) < datetime.fromisoformat(request["deadline_utc"]),
                "Verification is past the shared deadline")
        attempt = {"output": str(Path(output).resolve()), "status": "running",
                   "cpu_charged_seconds": reserved}
        ledger["attempts"].append(attempt)
        save_json(ledger_path, ledger, replace=True)
        try:
            result = _build(request_ref, summary_ref, source_ref, output, scope_audit_ref,
                            review_cpu_cap=reserved - (time.process_time() - started))
            attempt["status"] = "completed"
            return result
        except BaseException:
            attempt["status"] = "failed"
            raise
        finally:
            attempt["cpu_seconds"] = time.process_time() - started
            attempt["cpu_charged_seconds"] = attempt["cpu_seconds"]
            save_json(ledger_path, ledger, replace=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("request", "summary", "source"):
        parser.add_argument("--" + name, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--scope-audit", help="Optional pinned root-metadata analysis for this run")
    parser.add_argument("--scope-audit-sha256")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if bool(args.scope_audit) != bool(args.scope_audit_sha256):
        parser.error("Pass both --scope-audit and --scope-audit-sha256")
    build(*({"path": getattr(args, name), "sha256": getattr(args, name + "_sha256")}
            for name in ("request", "summary", "source")), args.output,
          {"path": args.scope_audit, "sha256": args.scope_audit_sha256} if args.scope_audit else None)
