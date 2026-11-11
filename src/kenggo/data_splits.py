"""Deterministic whole-game partitions over trusted canonical evidence."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
import re

import pyarrow as pa
import pyarrow.parquet as pq

from kenggo.canonical_data import calendar_day, digest, encoded


VERSION = "whole_game_splits_v1"
ROLES = ("train", "population_development", "population_confirmation",
         "validation_prefix", "validation_development", "validation_confirmation",
         "test_prefix", "final", "embargo", "excluded")


@dataclass(frozen=True)
class SplitPlan:
    grouping_sha256: str
    train_before: str
    confirmation_from: str
    validation_groups: tuple[str, ...] = ()
    test_groups: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.grouping_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", self.grouping_sha256) is None:
            raise ValueError("A frozen grouping digest is required")
        if not calendar_day(self.train_before) < calendar_day(self.confirmation_from):
            raise ValueError("Development must precede confirmation on separate calendar days")
        for groups in (self.validation_groups, self.test_groups):
            if (type(groups) is not tuple or len(set(groups)) != len(groups)
                    or any(not isinstance(g, str) or not g for g in groups)):
                raise ValueError("Cohorts require distinct immutable group keys")
        if set(self.validation_groups) & set(self.test_groups):
            raise ValueError("Validation and test groups overlap")

    @classmethod
    def from_dict(cls, value):
        if set(value) != {"grouping_sha256", "train_before", "confirmation_from",
                          "validation_groups", "test_groups"}:
            raise ValueError("Unexpected split configuration fields")
        return cls(**{**value, "validation_groups": tuple(value["validation_groups"]),
                      "test_groups": tuple(value["test_groups"])})

    def as_dict(self):
        return {"grouping_sha256": self.grouping_sha256, "train_before": self.train_before,
                "confirmation_from": self.confirmation_from,
                "validation_groups": sorted(self.validation_groups), "test_groups": sorted(self.test_groups)}


def duplicate_components(entries):
    """Sequence equality is collision evidence; disagreeing copies cannot choose a survivor."""
    parent = {r["source_url"]: r["source_url"] for r in entries}
    if len(parent) != len(entries):
        raise ValueError("Source IDs must be unique")

    def find(key):
        while parent[key] != key:
            key = parent[key]
        return key

    seen = {}
    for row in sorted(entries, key=lambda r: r["source_url"]):
        for field in ("sgf_sha256", "audit_sequence_sha256", "trajectory_sha256"):
            value = row.get(field)
            if not value or (field != "sgf_sha256" and not row["actions"]):
                continue
            key = (field, value)
            if key in seen:
                left, right = find(row["source_url"]), find(seen[key])
                parent[max(left, right)] = min(left, right)
            else:
                seen[key] = row["source_url"]
    components = defaultdict(list)
    for row in entries:
        components[find(row["source_url"])].append(row)
    return [sorted(rows, key=lambda r: r["source_url"]) for _, rows in sorted(components.items())]


def build_splits(entries, plan, *, exposed_groups, group_registry=None):
    """Entries are prepared by the source adapter; raw source assertions cannot grant admission."""
    exposed_groups = frozenset(exposed_groups)
    for row in entries:
        if (not isinstance(row["source_url"], str) or not row["source_url"]
                or len(row["groups"]) != 2 or len(row["identity_valid"]) != 2
                or any(type(flag) is not bool for flag in row["identity_valid"])
                or any(type(row[key]) is not bool for key in ("admitted", "value_supported", "final_excluded"))
                or type(row["actions"]) is not int or row["actions"] < 0
                or len(row["actions_by_color"]) != 2
                or any(type(n) is not int or n < 0 for n in row["actions_by_color"])
                or sum(row["actions_by_color"]) != row["actions"]):
            raise ValueError("Invalid canonical split entry")
    if any(r["grouping_sha256"] != plan.grouping_sha256 for r in entries):
        raise ValueError("Changed grouping registry; rebuild canonical inputs and splits")
    known_groups = {g for r in entries for g in r["groups"] if g is not None}
    if group_registry is not None:
        if (not isinstance(group_registry, dict) or digest(group_registry) != plan.grouping_sha256
                or any(not isinstance(key, str) or not isinstance(value, str) or not value
                       for key, value in group_registry.items())
                or known_groups - set(group_registry.values())):
            raise ValueError("Reserved-group registry differs from the frozen grouping pin")
        known_groups.update(group_registry.values())
    if (set(plan.validation_groups) | set(plan.test_groups)) - known_groups:
        raise ValueError("Reserved group is absent from the canonical frame")
    if set(plan.test_groups) & exposed_groups:
        raise ValueError("An inspected group cannot become an unseen final-test group")
    validation, test = set(plan.validation_groups), set(plan.test_groups)
    rows = []
    for component in duplicate_components(entries):
        representative = component[0]["source_url"]
        reasons = {x for r in component for x in r["exclusions"]}
        if any(r["admitted"] is not True for r in component):
            reasons.add("source_not_admitted")
        if any(r["actions"] == 0 for r in component):
            reasons.add("no_policy_actions")
        if len({r["semantic_sha256"] for r in component}) != 1:
            reasons.add("ambiguous_duplicate_component")
        groups = {g for r in component for g in r["groups"]}
        if None in groups:
            reasons.add("group_unavailable")
        if any(len(set(r["groups"])) != 2 for r in component):
            reasons.add("same_group_on_both_sides")
        if groups & validation and groups & test:
            reasons.add("reserved_cohort_bridge")
        final_exposed = any(r["final_excluded"] for r in component) or bool(groups & exposed_groups)
        for entry in component:
            own_reasons = set(reasons)
            if entry["source_url"] != representative:
                own_reasons.add("duplicate_copy")
            try:
                day = calendar_day(entry["date"])
            except (ValueError, TypeError):
                own_reasons.add("date_unavailable")
                day = None
            role = "excluded"
            if not own_reasons:
                early = day < date.fromisoformat(plan.train_before)
                late = day >= date.fromisoformat(plan.confirmation_from)
                if groups & test:
                    role = "test_prefix" if early else "final" if late else "embargo"
                    if final_exposed:
                        own_reasons.add("final_exposure")
                        role = "excluded"
                elif groups & validation:
                    role = "validation_prefix" if early else "validation_confirmation" if late else "validation_development"
                else:
                    role = "train" if early else "population_confirmation" if late else "population_development"
            rows.append({"source_url": entry["source_url"], "canonical_sha256": entry["canonical_sha256"],
                         "duplicate_representative": representative,
                         "duplicate_sources": [r["source_url"] for r in component],
                         "date": entry["date"], "groups": entry["groups"], "role": role,
                         "exclusions": sorted(own_reasons), "final_excluded": final_exposed,
                         "actions": entry["actions"], "policy_actions": entry["actions"] if role not in {"excluded", "embargo"} else 0,
                         "actions_by_color": entry["actions_by_color"],
                         "value_actions": entry["actions"] if role not in {"excluded", "embargo"} and entry["value_supported"] else 0,
                         "identity_valid": list(entry["identity_valid"]),
                         "adaptation_colors": [], "known_persona_colors": []})
    trained = {g for r in rows if r["role"] == "train"
               for g, valid, count in zip(r["groups"], r["identity_valid"], r["actions_by_color"]) if valid and count}
    for row in rows:
        for i, color in enumerate(("B", "W")):
            group, valid = row["groups"][i], row["identity_valid"][i]
            if valid and row["role"] not in {"excluded", "embargo"}:
                if (row["role"] in {"validation_prefix", "test_prefix"} and group in validation | test
                        and row["actions_by_color"][i] > 0):
                    row["adaptation_colors"].append(color)
                if group in trained:
                    row["known_persona_colors"].append(color)
        if row["role"] in {"validation_prefix", "test_prefix"}:
            row["policy_actions"] = sum(row["actions_by_color"][i] for i, color in enumerate(("B", "W"))
                                        if color in row["adaptation_colors"])
            row["value_actions"] = 0
    return sorted(rows, key=lambda r: (r["date"] or "", r["source_url"]))


def manifest_contents(entries, rows, plan, *, exposure_sha256, provenance, fixture_only=False):
    counts = {role: sum(r["role"] == role for r in rows) for role in ROLES}
    return {"version": VERSION, "fixture_only": fixture_only, "plan": plan.as_dict(),
            "plan_sha256": digest(plan.as_dict()),
            "canonical_input_sha256": digest(sorted(entries, key=lambda r: r["source_url"])),
            "exposure_sha256": exposure_sha256, "provenance": provenance,
            "roles": counts, "exclusion_counts": dict(sorted(Counter(x for r in rows for x in r["exclusions"]).items())),
            "policy_actions_by_role": {role: sum(r["policy_actions"] for r in rows if r["role"] == role) for role in ROLES},
            "value_actions_by_role": {role: sum(r["value_actions"] for r in rows if r["role"] == role) for role in ROLES},
            "adaptation_sides": sum(len(r["adaptation_colors"]) for r in rows),
            "execution_requirements": {
                "shared_training_and_normalizer_fit": "train role only",
                "selection": "development roles only; test-group data never used for selection",
                "confirmation": "frozen candidate; never add development/confirmation rows to adaptation prefixes",
                "adaptation": "only listed target colors; shared weights frozen; procedure selected on validation",
                "final": "sealed until deployment profiles and shared weights/procedure are frozen; hashes do not prove noninspection"}}


def write_membership(output, rows, manifest):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    schema = pa.schema([
        *[(field, pa.string()) for field in ("source_url", "canonical_sha256", "duplicate_representative", "date", "role")],
        *[(field, pa.list_(pa.string())) for field in ("duplicate_sources", "groups", "exclusions", "adaptation_colors", "known_persona_colors")],
        ("identity_valid", pa.list_(pa.bool_())), ("actions_by_color", pa.list_(pa.int64())), ("final_excluded", pa.bool_()),
        *[(field, pa.int64()) for field in ("actions", "policy_actions", "value_actions")]])
    for role in ROLES:
        contents = [r for r in rows if r["role"] == role]
        path = output / (role + ".parquet")
        pq.write_table(pa.Table.from_pylist(contents, schema=schema), path, compression="zstd")
        files[path.name] = sha256(path.read_bytes()).hexdigest()
    frozen = {**manifest, "membership_sha256": files}
    (output / "manifest.json").write_text(encoded(frozen) + "\n")
    return sha256((output / "manifest.json").read_bytes()).hexdigest()


def read_membership(directory, expected_sha256):
    directory = Path(directory)
    data = (directory / "manifest.json").read_bytes()
    manifest = json.loads(data)
    if sha256(data).hexdigest() != expected_sha256 or set(manifest["membership_sha256"]) != {r + ".parquet" for r in ROLES}:
        raise ValueError("Split manifest differs from the caller-pinned version")
    result = {}
    for role in ROLES:
        path = directory / (role + ".parquet")
        if sha256(path.read_bytes()).hexdigest() != manifest["membership_sha256"][role + ".parquet"]:
            raise ValueError("Frozen split membership changed")
        result[role] = pq.read_table(path).to_pylist()
    return manifest, result
