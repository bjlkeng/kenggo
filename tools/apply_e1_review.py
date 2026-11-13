"""Derive usable E1 splits and examples after the owner's two-game exclusion."""

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import shutil
import time

from kenggo.canonical_data import digest, encoded
from kenggo.data_splits import ROLES, read_membership, write_membership
from kenggo.encoding import INPUT_KEYS
from kenggo.examples import (file_hash, fit_normalizer, load_corpus, load_dataset_profile,
                             read_pinned_json, read_shards, write_shard)


ROOT = Path(__file__).resolve().parents[1]
DECISION = {"path": "manifests/e1_manual_review_v1.json",
            "sha256": "44fc41edd839f53745bf28a8c315ef94b2e78e5ee3f049bcda491b9903b9b5dd"}


def read_ref(ref):
    return read_pinned_json(ROOT / ref["path"], ref["sha256"], max_bytes=16 * 1024**2)


def exclude_memberships(records, split, partitions, exclusions, decision_sha256):
    rows = copy.deepcopy([row for role in ROLES for row in partitions[role]])
    by_id = {row["source_url"]: row for row in rows}
    selected = {item["source_url"] for item in exclusions}
    if not selected or len(selected) != len(exclusions):
        raise ValueError("Exclusions require distinct source IDs")
    for item in exclusions:
        key = item["source_url"]
        if key not in records or records[key].get("sealed_final", False):
            raise ValueError("Exclusions must name an existing nonfinal source")
        record, row = records[key], by_id[key]
        if (row["role"] != "train" or row["duplicate_sources"] != [key]
                or row["canonical_sha256"] != item["canonical_sha256"]
                or digest(record) != item["canonical_sha256"]
                or record["source"]["sgf_sha256"] != item["sgf_sha256"]
                or not isinstance(item["reason"], str) or not item["reason"]):
            raise ValueError("Exclusion differs from its pinned unique training game")
        row.update(role="excluded", policy_actions=0, value_actions=0,
                   adaptation_colors=[], known_persona_colors=[],
                   exclusions=sorted(set(row["exclusions"] + [item["reason"]])))
    updated = copy.deepcopy(split)
    updated["roles"] = {role: sum(row["role"] == role for row in rows) for role in ROLES}
    for field, count in (("policy_actions_by_role", "policy_actions"), ("value_actions_by_role", "value_actions")):
        updated[field] = {role: sum(row[count] for row in rows if row["role"] == role) for role in ROLES}
    updated["exclusion_counts"] = dict(sorted(Counter(reason for row in rows for reason in row["exclusions"]).items()))
    updated["adaptation_sides"] = sum(len(row["adaptation_colors"]) for row in rows)
    updated["provenance"]["manual_review_sha256"] = decision_sha256
    return rows, updated


def copy_shard(source, output, role, number, item):
    for field, suffix in (("arrays", "npz"), ("metadata", "parquet")):
        if item[field] != f"{role}/{number:05d}.{suffix}":
            raise ValueError("Unexpected shard path")
        original, target = source / item[field], output / item[field]
        if file_hash(original) != item[field + "_sha256"]:
            raise ValueError("Source shard differs from its pin")
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(original, target)
        if file_hash(target) != item[field + "_sha256"]:
            raise ValueError("Copied shard differs from its pin")


def filter_examples(source, pin, manifest, output, excluded, check_budget):
    output.mkdir(parents=True, exist_ok=False)
    artifacts, counts = copy.deepcopy(manifest["artifacts"]), copy.deepcopy(manifest["counts"])
    counts["train"] = {key: 0 for key in counts["train"]}
    removed = Counter()
    for role in ROLES:
        if role != "train":
            for number, item in enumerate(manifest["artifacts"][role]):
                check_budget()
                copy_shard(source, output, role, number, item)
            continue
        for number, (arrays, metadata) in enumerate(read_shards(source, pin, "train")):
            check_budget()
            indices = [i for i, meta in enumerate(metadata) if meta["source_url"] not in excluded]
            removed.update(meta["source_url"] for meta in metadata if meta["source_url"] in excluded)
            if len(indices) == len(metadata):
                copy_shard(source, output, role, number, manifest["artifacts"][role][number])
            else:
                rows = [({key: value[i] for key, value in arrays.items() if key in INPUT_KEYS},
                         {key: value[i] for key, value in arrays.items() if key not in INPUT_KEYS},
                         metadata[i]) for i in indices]
                artifacts[role][number] = write_shard(output, role, number, rows, manifest["config"]["history"])
            count = counts["train"]
            count["examples"] += len(indices)
            for key in ("value_valid", "duration_valid"):
                count[key] += int(arrays[key][indices].sum())
            count["passes"] += int((arrays["action"][indices] == 361).sum())
            count["hidden_mover_rank"] += sum(metadata[i]["rank_hidden"][0] for i in indices)
            count["hidden_opponent_rank"] += sum(metadata[i]["rank_hidden"][1] for i in indices)
    return artifacts, counts, dict(removed)


def write_json(path, value):
    path.write_text(encoded(value) + "\n")
    return file_hash(path)


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Review-derived output already exists")
    started, cpu = time.monotonic(), time.process_time()
    def check_budget():
        if time.monotonic() - started > 180 or time.process_time() - cpu > 120:
            raise RuntimeError("Manual-review derivation exceeded its time budget")

    decision = read_ref(DECISION)
    base = read_ref(decision["base_population"])
    comparison = read_ref(decision["historical_comparison"])
    original = base["artifacts"]
    profile_ref = base["inputs"]["example_profile"]
    profile_json = read_ref(profile_ref)
    profile = load_dataset_profile(ROOT / profile_ref["path"], profile_ref["sha256"])
    source_splits = (ROOT / original["split_manifest"]["path"]).parent
    records, _, _ = load_corpus(ROOT / original["canonical"]["path"], profile.canonical_sha256,
                               source_splits, profile.split_sha256, profile=profile)
    split, partitions = read_membership(source_splits, profile.split_sha256)
    for item in decision["excluded_games"]:
        case = next(row for row in comparison["endings"] if row["case"] == item["case"])
        if case["source_url"] != item["source_url"] or case.get("winner_agrees") is not False:
            raise ValueError("Exclusion does not match the reviewed winner disagreement")
    rows, updated = exclude_memberships(records, split, partitions, decision["excluded_games"], DECISION["sha256"])
    updated["provenance"]["parent_split_sha256"] = profile.split_sha256
    manifest = read_ref(original["examples_manifest"])
    if (manifest["canonical_sha256"] != profile.canonical_sha256 or manifest["split_sha256"] != profile.split_sha256
            or manifest["dataset_profile_sha256"] != profile.sha256):
        raise ValueError("Examples do not belong to the pinned population")
    check_budget()
    output.mkdir(parents=True, exist_ok=False)
    try:
        split_pin = write_membership(output / "splits", rows, updated)
        resulting, _ = read_membership(output / "splits", split_pin)
        unchanged = [role for role in ROLES if role not in {"train", "excluded"}]
        if any(resulting["membership_sha256"][role + ".parquet"] != split["membership_sha256"][role + ".parquet"]
               for role in unchanged):
            raise ValueError("An unrelated split file changed")
        profile_json["split_sha256"] = split_pin
        profile_pin = write_json(output / "profile.json", profile_json)
        excluded = {item["source_url"] for item in decision["excluded_games"]}
        artifacts, counts, removed = filter_examples((ROOT / original["examples_manifest"]["path"]).parent,
            original["examples_manifest"]["sha256"], manifest, output / "examples", excluded, check_budget)
        expected = {row["source_url"]: row["policy_actions"] for row in partitions["train"] if row["source_url"] in excluded}
        if removed != expected:
            raise ValueError("Removed examples do not cover exactly the excluded games")
        manifest.update(artifacts=artifacts, counts=counts, split_sha256=split_pin,
                        dataset_profile_sha256=profile_pin, source_roles=resulting["roles"],
                        exclusions=resulting["exclusion_counts"],
                        derivation={"manual_review": DECISION, "parent_examples": original["examples_manifest"],
                                    "tool_sha256": file_hash(Path(__file__))})
        example_pin = write_json(output / "examples/manifest.json", manifest)
        normalizer = fit_normalizer(output / "examples", example_pin)
        normalizer_pin = write_json(output / "normalizer.json", normalizer)
        if normalizer["count"][5] != counts["train"]["examples"]:
            raise ValueError("Training normalizer count mismatch")
        total_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        if total_bytes > 256 * 1024**2:
            raise ValueError("Manual-review derivation exceeds 256 MiB")
        check_budget()
    except Exception as error:
        write_json(output / "failure.json", {"status": "failed", "error": str(error),
            "cpu_seconds": time.process_time() - cpu, "wall_seconds": time.monotonic() - started})
        raise
    summary = {"version": "e1_manual_review_derivation_v1", "manual_review": DECISION,
               "base_population": decision["base_population"], "removed_examples": removed,
               "split_manifest_sha256": split_pin, "profile_sha256": profile_pin,
               "examples_manifest_sha256": example_pin, "normalizer_sha256": normalizer_pin,
               "roles": resulting["roles"], "train_counts": counts["train"],
               "unchanged_split_files": unchanged,
               "unchanged_example_roles": [role for role in ROLES if role != "train"],
               "final_targets_decoded": False, "canonical_unchanged": True,
               "historical_comparison_unchanged": True, "output_bytes": total_bytes,
               "tool_sha256": file_hash(Path(__file__)),
               "costs": {"cpu_seconds": time.process_time() - cpu, "wall_seconds": time.monotonic() - started,
                         "workers": 1, "http_requests": 0, "gpu_seconds": 0}}
    write_json(output / "summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("removed_examples", "roles", "train_counts", "costs")}, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
