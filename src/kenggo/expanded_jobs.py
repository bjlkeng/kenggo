"""Freeze expanded E2 development sets and compare their pinned predictions."""

import argparse
import os
from pathlib import Path
import re

from kenggo.base_pilot import ROOT, _common_inputs, _ref
from kenggo.baseline_evaluation import load_predictions, rows_for_common
from kenggo.canonical_data import digest, encoded
from kenggo.checkpoints import file_hash, json_hash, read_json, save_json
from kenggo.data_splits import read_membership
from kenggo.examples import load_corpus
from kenggo.kgs_source import RANKS
from kenggo.recorded_data import RAW_VERSION, prepare_common_set
from kenggo.recorded_statistics import paired_report
from kenggo.tracking import capture_provenance, validate_config


SEED = 20261118
ROLES = ("population_development", "validation_development")
KOMI = (13, 1)
SELECTION = {"games_per_rank": 2, "max_games_per_set": 32, "max_rows_per_set": 12000}
ARMS = ("baseline-conditioned", "baseline-unconditioned", "combined-conditioned", "combined-unconditioned")
CONTRASTS = ((ARMS[0], ARMS[1]), (ARMS[2], ARMS[3]), (ARMS[2], ARMS[0]), (ARMS[3], ARMS[1]))


def _configuration(config, purpose):
    validate_config(config)
    if config["purpose"] != purpose or config["seed"] != SEED:
        raise ValueError("Expanded jobs require their declared purpose and frozen seed")


def select_games(candidates, *, role, komi_halfpoints, seed=SEED, selection=None):
    """Choose complete games from membership and input metadata alone."""
    selection = dict(SELECTION if selection is None else selection)
    if (role not in ROLES or type(komi_halfpoints) is not int or komi_halfpoints not in KOMI
            or type(seed) is not int or seed != SEED or set(selection) != set(SELECTION)
            or any(type(value) is not int or not 0 < value <= SELECTION[key]
                   for key, value in selection.items())):
        raise ValueError("Selection must fit the frozen development roles, seed and whole-game limits")
    candidates = list(candidates)
    seen = set()
    for row in candidates:
        source = row["source_url"]
        counts, groups = row["actions_by_color"], row["groups"]
        if (not isinstance(source, str) or not source or source in seen or row["role"] not in ROLES
                or type(row["komi_halfpoints"]) is not int or row["komi_halfpoints"] not in KOMI
                or len(row["ranks"]) != 2 or len(counts) != 2
                or any(type(count) is not int or count < 0 for count in counts)
                or type(row["actions"]) is not int or sum(counts) != row["actions"]
                or type(row["policy_actions"]) is not int or row["policy_actions"] != row["actions"]
                or type(row["value_actions"]) is not int or row["value_actions"] not in (0, row["actions"])
                or row["exclusions"] or len(groups) != 2 or len(set(groups)) != 2
                or any(not isinstance(group, str) or not group for group in groups)):
            raise ValueError("Selection requires unique complete games with valid counts and distinct account groups")
        seen.add(source)
    eligible = [row for row in candidates if row["role"] == role and row["komi_halfpoints"] == komi_halfpoints]

    def has_rank(row, rank):
        return any(value == rank and count > 0 for value, count in zip(row["ranks"], row["actions_by_color"]))

    selected, rank_choices, rows = {}, {}, 0
    for rank in RANKS:
        ordered = sorted((row for row in eligible if has_rank(row, rank)),
            key=lambda row: (digest([seed, role, komi_halfpoints, rank, row["source_url"]]), row["source_url"]))
        choices, skipped = [], []
        for row in ordered:
            if len(choices) == selection["games_per_rank"]:
                break
            source = row["source_url"]
            if source not in selected:
                if (len(selected) >= selection["max_games_per_set"]
                        or rows + row["actions"] > selection["max_rows_per_set"]):
                    skipped.append(source)
                    continue
                selected[source] = row
                rows += row["actions"]
            choices.append(source)
        rank_choices[rank] = {"eligible_games": len(ordered), "chosen_source_ids": choices,
                              "shortfall": selection["games_per_rank"] - len(choices),
                              "cap_skipped_source_ids": skipped}
    for rank in RANKS:
        rank_choices[rank]["selected_games_with_rank_moves"] = sum(has_rank(row, rank) for row in selected.values())
    spec = {"id": f"{role}_komi{komi_halfpoints}", "role": role, "source_ids": sorted(selected),
            "aggregate_ranks": list(RANKS), "komi_halfpoints": komi_halfpoints, "expected_rows": rows}
    return {"spec": spec, "eligible_games": len(eligible), "selected_games": len(selected),
            "rank_choices": rank_choices}


def _development_candidates(inputs):
    _, partitions = read_membership(inputs.split_dir, inputs.split_sha256)
    result = []
    for role in ROLES:
        source_ids = [row["source_url"] for row in partitions[role]]
        if not source_ids:
            continue
        records, _, selected = load_corpus(inputs.canonical_path, inputs.canonical_sha256,
            inputs.split_dir, inputs.split_sha256, profile=inputs.profile, role=role,
            source_ids=source_ids, komi_study=True)
        for membership in selected[role]:
            metadata = records[membership["source_url"]]["metadata"]
            if metadata.get("KM") not in (.5, 6.5):
                raise ValueError("Development metadata has unsupported komi")
            result.append({**membership, "ranks": [metadata.get("BR"), metadata.get("WR")],
                           "komi_halfpoints": int(metadata["KM"] * 2)})
    return result


def prepare(config, output):
    _configuration(config, "expanded-preparation")
    selection = config["parameters"].get("selection")
    if (selection != SELECTION or any(type(value) is not int for value in selection.values())):
        raise ValueError("Preparation requires the frozen two-game, 32-game, 12,000-row selection")
    output, inputs = Path(output), _common_inputs(config["inputs"])
    candidates = _development_candidates(inputs)
    chosen = [select_games(candidates, role=role, komi_halfpoints=komi,
                           seed=config["seed"], selection=selection) for role in ROLES for komi in KOMI]
    selection_path = output / "selection.json"
    save_json(selection_path, {"schema": "expanded_selection_v1", "seed": config["seed"],
        "limits": selection, "ranks": list(RANKS), "sets": chosen,
        "ordering": "SHA256 of canonical JSON [seed, role, komi_halfpoints, rank, source_url]",
        "target_or_prediction_selection": False})
    result = []
    for selected in chosen:
        spec = selected["spec"]
        if not spec["source_ids"]:
            raise ValueError(f"No eligible complete game fits the frozen set: {spec['id']}")
        common = prepare_common_set(inputs, role=spec["role"], source_ids=spec["source_ids"],
            aggregate_ranks=spec["aggregate_ranks"], raw_inputs=True, komi_study=True)
        if (len(common) != spec["expected_rows"]
                or {row["komi_halfpoints"] for row in common.manifest["rows"]} != {spec["komi_halfpoints"]}):
            raise ValueError("Common-set rows or komi differ from the complete selected games")
        directory = output / spec["id"]
        directory.mkdir()
        path = directory / "manifest.json"
        with path.open("x") as stream:
            stream.write(encoded(common.manifest))
            stream.flush()
            os.fsync(stream.fileno())
        if file_hash(path) != common.sha256:
            raise ValueError("Saved common-set manifest differs from its frozen content")
        result.append({"spec": spec, "manifest": _ref(path), "rows": len(common),
                       "coverage": common.manifest["coverage"]})
    summary = {"schema": "expanded_preparation_v1", "status": "completed", "common_sets": result,
               "selection": _ref(selection_path), "predictions": 0,
               "scope": "Four development sets frozen before model output; confirmation and final targets stay sealed"}
    save_json(output / "summary.json", summary)
    return summary


def _analysis_specs(config):
    specs, refs = config["parameters"].get("common_sets"), config["inputs"]
    if not isinstance(specs, list) or len(specs) != len(ROLES) * len(KOMI):
        raise ValueError("Analysis requires all four frozen development sets")
    ids, sources, strata, expected = set(), set(), set(), []
    for spec in specs:
        if (not isinstance(spec, dict)
                or set(spec) != {"id", "role", "source_ids", "aggregate_ranks", "manifest_input", "prediction_inputs"}
                or not isinstance(spec["id"], str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", spec["id"]) is None
                or spec["id"] in ids or spec["role"] not in ROLES or spec["aggregate_ranks"] != RANKS
                or not isinstance(spec["source_ids"], list) or not spec["source_ids"]
                or any(not isinstance(source, str) or not source for source in spec["source_ids"])
                or len(set(spec["source_ids"])) != len(spec["source_ids"])
                or sources.intersection(spec["source_ids"]) or spec["manifest_input"] not in refs
                or not isinstance(spec["prediction_inputs"], dict) or set(spec["prediction_inputs"]) != set(ARMS)
                or any(name not in refs for name in spec["prediction_inputs"].values())):
            raise ValueError("Analysis requires distinct frozen sets and all four pinned prediction arms")
        ref = refs[spec["manifest_input"]]
        manifest = read_json(ROOT / ref["path"], ref["sha256"])
        rows = manifest.get("rows", [])
        komi = {row.get("komi_halfpoints") for row in rows}
        if (manifest.get("version") != RAW_VERSION or manifest.get("role") != spec["role"]
                or manifest.get("aggregate_ranks") != RANKS
                or set(manifest.get("source_ids", [])) != set(spec["source_ids"])
                or not 0 < len(rows) <= SELECTION["max_rows_per_set"]
                or len(spec["source_ids"]) > SELECTION["max_games_per_set"]
                or len(komi) != 1 or not komi <= set(KOMI)):
            raise ValueError("Analysis manifest differs from the frozen role, rank, komi or row scope")
        stratum = spec["role"], next(iter(komi))
        if stratum in strata:
            raise ValueError("Analysis cannot repeat a role and komi stratum")
        ids.add(spec["id"])
        sources.update(spec["source_ids"])
        strata.add(stratum)
        expected.append((spec, manifest))
    if strata != {(role, komi) for role in ROLES for komi in KOMI}:
        raise ValueError("Analysis must retain every predeclared role and komi stratum")
    return expected


def analyze(config, output):
    _configuration(config, "expanded-analysis")
    expected = _analysis_specs(config)
    inputs, result, output = _common_inputs(config["inputs"]), {}, Path(output)
    for spec, manifest in expected:
        common = prepare_common_set(inputs, role=spec["role"], source_ids=spec["source_ids"],
            aggregate_ranks=spec["aggregate_ranks"], raw_inputs=True, komi_study=True)
        if (common.sha256 != config["inputs"][spec["manifest_input"]]["sha256"]
                or common.manifest != manifest):
            raise ValueError("Paired analysis common set changed")
        rows = rows_for_common(common)
        predictions = {}
        for arm, name in spec["prediction_inputs"].items():
            ref = config["inputs"][name]
            path = ROOT / ref["path"]
            if file_hash(path) != ref["sha256"]:
                raise ValueError("Saved predictions differ from their evaluation pin")
            predictions[arm] = load_predictions(path, expected_row_keys=[row["row_key"] for row in rows])
        result[spec["id"]] = {}
        for candidate, control in CONTRASTS:
            name = candidate + "_minus_" + control
            report = paired_report(rows, predictions[candidate], predictions[control],
                                   aggregate_ranks=spec["aggregate_ranks"])
            path = output / f"{spec['id']}-{name}.json"
            save_json(path, report)
            result[spec["id"]][name] = {"report": _ref(path), "overall": report["overall"]}
    summary = {"schema": "expanded_paired_analysis_v1", "status": "completed", "comparisons": result,
               "promotion": "not_promoted", "scope": "Single-seed development contrasts; final data sealed"}
    save_json(output / "summary.json", summary)
    return summary


def local_worker(config_path, config_pin, provenance_path, provenance_pin, output):
    output = Path(output)
    if (output.is_symlink() or not output.is_dir()
            or any(os.path.lexists(output / name) for name in ("summary.json", "selection.json", "failure.json"))):
        raise ValueError("Expanded workers require a fresh parent-created output directory")
    try:
        config = read_json(config_path, config_pin)
        provenance = read_json(provenance_path, provenance_pin)
        validate_config(config)
        if config["purpose"] not in ("expanded-preparation", "expanded-analysis"):
            raise ValueError("Unexpected expanded worker purpose")
        required = {"data", "canonical", "split", "examples", "normalizer", "profile", "rules", "cpu_environment"}
        if not required <= config["inputs"].keys() or any(
                not isinstance(ref, dict) or set(ref) != {"path", "sha256"}
                or not isinstance(ref["path"], str) or not ref["path"]
                or not isinstance(ref["sha256"], str) or re.fullmatch(r"[a-f0-9]{64}", ref["sha256"]) is None
                for ref in config["inputs"].values()):
            raise ValueError("Expanded workers require independently pinned data and environment inputs")
        current, _ = capture_provenance(ROOT, config["inputs"])
        if (provenance.get("version") != "experiment_provenance_v1"
                or provenance.get("inputs") != config["inputs"]
                or not isinstance(provenance.get("environment"), dict)
                or json_hash(provenance) != provenance_pin or provenance.get("source") != current["source"]):
            raise ValueError("Worker source or inputs differ from canonical pinned provenance")
        result = (prepare if config["purpose"] == "expanded-preparation" else analyze)(config, output)
        after, _ = capture_provenance(ROOT, config["inputs"])
        if after["source"] != current["source"]:
            raise ValueError("Worker source changed")
        return result
    except BaseException as error:
        failure = {"status": "failed", "type": type(error).__name__, "reason": str(error)}
        if (output / "summary.json").is_file():
            summary = read_json(output / "summary.json", file_hash(output / "summary.json"))
            save_json(output / "summary.json", {**summary, "status": "failed"}, replace=True)
        save_json(output / "failure.json", failure)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("provenance", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--provenance-sha256", required=True)
    args = parser.parse_args()
    local_worker(args.config, args.config_sha256, args.provenance, args.provenance_sha256, args.output)
