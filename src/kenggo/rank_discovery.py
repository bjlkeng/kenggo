"""Choose archive pages from pinned historical rank and account metadata."""

from collections import Counter, defaultdict, deque
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.parse import urlencode

import pyarrow.parquet as pq

from kenggo import e1_audit, kgs_eligibility, kgs_source


POLICY_SCHEMA = "e2_rank_discovery_policy_v1"
COUNTS_SCHEMA = "e2_rank_discovery_counts_v1"
SEED = "kenggo-ydo.10-rank-collection-20261115-v1"
ROLES = ("train", "population_development", "validation_development")
COUNT_FIELDS = ("train_games_by_rank", "population_development_game_sides_by_rank")


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash(*parts):
    return sha256(_encoded([SEED, *parts])).hexdigest()


def _reference(reference):
    if (not isinstance(reference, dict) or set(reference) != {"path", "sha256"}
            or not isinstance(reference["path"], str) or not reference["path"]
            or not isinstance(reference["sha256"], str)
            or re.fullmatch("[0-9a-f]{64}", reference["sha256"]) is None):
        raise ValueError("A discovery input needs a path and SHA-256")
    return Path(reference["path"])


def _read(reference, maximum=16 * 1024**2):
    path = _reference(reference)
    if not path.is_file() or path.stat().st_size > maximum:
        raise ValueError("Missing or oversized discovery input")
    body = path.read_bytes()
    if sha256(body).hexdigest() != reference["sha256"]:
        raise ValueError("Discovery input hash changed")
    return json.loads(body)


def freeze_baseline_counts(summary_ref, output_path):
    """Count active game sides using only training and development metadata."""
    summary = _read(summary_ref)
    if summary.get("schema") != "e2_data_expansion_v1":
        raise ValueError("Expected a data-expansion baseline summary")
    artifacts = summary["artifacts"]
    examples_ref, split_ref = artifacts["examples_manifest"], artifacts["split_manifest"]
    examples = _read(examples_ref)
    _read(split_ref)
    if (examples.get("version") != "supervised_examples_v1"
            or examples["split_sha256"] != split_ref["sha256"]):
        raise ValueError("Baseline examples and splits disagree")
    directory = _reference(examples_ref).resolve().parent
    sides = {role: {} for role in ROLES[:2]}
    for role in ROLES[:2]:
        observed_rows = 0
        for shard in examples["artifacts"][role]:
            relative = Path(shard["metadata"])
            path = (directory / relative).resolve()
            if (relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 2
                    or relative.parts[0] != role or relative.suffix != ".parquet"
                    or not path.is_relative_to(directory / role)):
                raise ValueError("Baseline metadata leaves its allowed role")
            if not path.is_file() or path.stat().st_size > 64 * 1024**2:
                raise ValueError("Missing or oversized baseline metadata")
            if sha256(path.read_bytes()).hexdigest() != shard["metadata_sha256"]:
                raise ValueError("Baseline metadata hash changed")
            rows = pq.read_table(path, columns=["source_url", "role", "mover", "source_rank_mover"])
            if rows.num_rows != shard["rows"]:
                raise ValueError("Baseline metadata row count changed")
            observed_rows += rows.num_rows
            for row in rows.to_pylist():
                source = row["source_url"]
                day = e1_audit.source_key(source) if kgs_source.is_sgf_url(source) else None
                expected_role = ("train" if day and "2025-01-01" <= day[0] < "2025-09-01"
                    else "population_development" if day and "2025-09-01" <= day[0] < "2025-11-01"
                    else None)
                if (row["role"] != role or expected_role != role or row["mover"] not in {"B", "W"}
                        or row["source_rank_mover"] not in kgs_source.RANKS):
                    raise ValueError("Baseline metadata violates its role or rank scope")
                key = (source, row["mover"])
                previous = sides[role].setdefault(key, row["source_rank_mover"])
                if previous != row["source_rank_mover"]:
                    raise ValueError("One baseline game side has conflicting source ranks")
        if (observed_rows != examples["counts"][role]["examples"]
                or observed_rows != summary["counts"]["combined"][role]["examples"]
                or len({source for source, _ in sides[role]})
                    != summary["counts"]["combined"][role]["games"]):
            raise ValueError("Baseline metadata totals disagree with the summary")
    counts = {
        "train_games_by_rank": {rank: len({source for (source, _), value in sides["train"].items()
            if value == rank}) for rank in kgs_source.RANKS},
        "population_development_game_sides_by_rank": {rank: sum(value == rank
            for value in sides["population_development"].values()) for rank in kgs_source.RANKS},
    }
    result = {"schema": COUNTS_SCHEMA, "baseline_counts": counts,
        "provenance": {"current_base": summary_ref, "examples_manifest": examples_ref,
                       "split_manifest": split_ref}}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = _encoded(result)
    path.write_bytes(body)
    return {"path": str(path.resolve()), "sha256": sha256(body).hexdigest()}


def _policy(policy):
    fields = {"schema", "seed", "page_counts", "target_train_games_per_rank",
              "target_population_development_sides_per_rank", "baseline_counts", "baseline_counts_ref"}
    if (not isinstance(policy, dict) or set(policy) != fields
            or policy["schema"] != POLICY_SCHEMA or policy["seed"] != SEED):
        raise ValueError("Use the frozen rank-discovery policy")
    pages = policy["page_counts"]
    if (not isinstance(pages, dict) or set(pages) != set(ROLES)
            or any(type(value) is not int or not 0 <= value <= 1150 for value in pages.values())
            or not 1 <= sum(pages.values()) <= 1150):
        raise ValueError("Freeze at most 1150 nonnegative discovery page counts")
    for key in ("target_train_games_per_rank", "target_population_development_sides_per_rank"):
        if type(policy[key]) is not int or not 1 <= policy[key] <= 1000:
            raise ValueError("Invalid discovery rank target")
    counts = policy["baseline_counts"]
    if not isinstance(counts, dict) or set(counts) != set(COUNT_FIELDS):
        raise ValueError("Missing baseline rank counts")
    for values in counts.values():
        if (not isinstance(values, dict) or set(values) != set(kgs_source.RANKS)
                or any(type(value) is not int or not 0 <= value <= 1000000 for value in values.values())):
            raise ValueError("Invalid baseline rank counts")
    evidence = _read(policy["baseline_counts_ref"])
    if (evidence.get("schema") != COUNTS_SCHEMA or evidence.get("baseline_counts") != counts
            or set(evidence.get("provenance", {})) != {"current_base", "examples_manifest", "split_manifest"}):
        raise ValueError("Baseline counts disagree with their frozen evidence")
    for reference in evidence["provenance"].values():
        _reference(reference)
    return pages, counts


def _context(groups, split, cache):
    if (split["train_before"] != "2025-09-01" or split["confirmation_from"] != "2025-11-01"
            or set(split["validation_groups"]) & set(split["test_groups"])):
        raise ValueError("Preserve the original dates and disjoint reserved groups")
    for handle, group in groups.items():
        if (kgs_eligibility.normalize_handle(handle) != handle or not isinstance(group, str)
                or not group.startswith("kgs-handle:")):
            raise ValueError("Invalid discovery account registry")
        kgs_eligibility.normalize_handle(group.removeprefix("kgs-handle:"))
    if not set(split["validation_groups"] + split["test_groups"]) <= set(groups.values()):
        raise ValueError("The discovery registry omits a reserved group")
    bots = {groups.get(handle, "kgs-handle:" + handle)
            for handle in set(groups) | set(kgs_eligibility.HISTORICAL_BOT_HANDLES)
            if "bot" in handle or handle in kgs_eligibility.HISTORICAL_BOT_HANDLES}
    directory = Path(cache)
    if not directory.is_dir():
        raise ValueError("The cached archive directory is missing")
    existing = set()
    for path in directory.iterdir():
        match = re.fullmatch(r"([A-Za-z0-9]+)-2025-(0[1-9]|1[0-2])\.html", path.name, re.I)
        if match and path.is_file():
            existing.add((match[1].lower(), int(match[2])))
    return set(split["validation_groups"]), set(split["test_groups"]), bots, existing


def _hints(frame, groups, validation, final, bots):
    hints = defaultdict(lambda: defaultdict(set))
    seen = {}
    for row in frame["rows"]:
        if not kgs_source.is_sgf_url(row["url"]):
            continue
        day = e1_audit.source_key(row["url"])
        if day is None or not "2025-01-01" <= day[0] < "2025-09-01":
            continue
        metadata = tuple(row[key] for key in ("black", "white", "setup", "type"))
        if seen.setdefault(row["url"], metadata) != metadata:
            raise ValueError("Conflicting historical account or rank metadata")
        if row["type"] != "Ranked" or not re.fullmatch(r"19×19(?: H[1-9]\d*)?", row["setup"]):
            continue
        players = [kgs_source.player(row[color]) for color in ("black", "white")]
        if any(player is None for player in players):
            continue
        handles = [player[0].lower() for player in players]
        cohort_groups = [groups.get(handle, "kgs-handle:" + handle) for handle in handles]
        if (set(cohort_groups) & (validation | final | bots)
                or any("bot" in handle or handle in kgs_eligibility.HISTORICAL_BOT_HANDLES
                       for handle in handles)):
            continue
        for handle, (_, rank) in zip(handles, players):
            if rank in kgs_source.RANKS:
                hints[rank][handle].add(int(day[0][5:7]))
    return hints


def _rank_pages(hints, existing, maximum, weights, role):
    queues = {}
    months = range(1, 9) if role == "train" else (9, 10)
    for rank in kgs_source.RANKS:
        if not weights[rank]:
            continue
        accounts = sorted(hints[rank], key=lambda handle: _hash("account", role, rank, handle))
        by_account = {}
        for handle in accounts:
            observed = hints[rank][handle]
            available = [month for month in months if (handle, month) not in existing]
            if role == "train":
                by_account[handle] = sorted(available,
                    key=lambda month: (month not in observed, min(abs(month - value) for value in observed),
                                       _hash("month", role, handle, month)))
            else:
                by_account[handle] = sorted(available, key=lambda month: _hash("month", role, handle, month))
        queue = [(handle, by_account[handle][depth]) for depth in range(len(months))
                 for handle in accounts if len(by_account[handle]) > depth]
        if queue:
            queues[rank] = deque(queue)
    counts, chosen, result = Counter(), set(), []
    while queues and len(result) < maximum:
        rank = min(queues, key=lambda value: (Fraction(counts[value], weights[value]), kgs_source.RANKS.index(value)))
        while queues[rank] and queues[rank][0] in chosen:
            queues[rank].popleft()
        if not queues[rank]:
            del queues[rank]
            continue
        handle, month = queues[rank].popleft()
        chosen.add((handle, month))
        counts[rank] += 1
        result.append(_query(handle, month, role, rank))
    return result


def _query(handle, month, role, rank=None):
    query = {"key": f"{handle}-2025-{month:02d}", "user": handle, "month": month,
        "url": "https://www.gokgs.com/gameArchives.jsp?" + urlencode(
            dict(user=handle, year=2025, month=month, oldAccounts="y")), "role": role}
    if rank is not None:
        query["discovery_rank"] = rank
    return query


def queries(frame, groups, split, cache, policy):
    """Freeze page requests; neither retrieval status nor game outcomes select them."""
    pages, counts = _policy(policy)
    validation, final, bots, existing = _context(groups, split, cache)
    hints = _hints(frame, groups, validation, final, bots)
    weights = {
        "train": {rank: max(0, policy["target_train_games_per_rank"] - counts[COUNT_FIELDS[0]][rank])
                  for rank in kgs_source.RANKS},
        "population_development": {rank: max(0, policy["target_population_development_sides_per_rank"]
            - counts[COUNT_FIELDS[1]][rank]) for rank in kgs_source.RANKS},
    }
    selected = {role: _rank_pages(hints, existing, pages[role], weights[role], role) for role in ROLES[:2]}
    handles = sorted((handle for handle, group in groups.items() if group in validation and group not in bots
        and "bot" not in handle and handle not in kgs_eligibility.HISTORICAL_BOT_HANDLES),
        key=lambda handle: _hash("validation", handle))
    months = {handle: sorted((month for month in (9, 10) if (handle, month) not in existing),
        key=lambda month: _hash("month", "validation", handle, month)) for handle in handles}
    selected["validation_development"] = [_query(handle, months[handle][depth], "validation_development")
        for depth in range(2) for handle in handles if len(months[handle]) > depth][:pages["validation_development"]]
    if any(len(selected[role]) != pages[role] for role in ROLES):
        raise ValueError("The frozen metadata frame cannot fill the requested page counts")
    pending = {role: deque(items) for role, items in selected.items() if items}
    used, result = Counter(), []
    while pending:
        role = min(pending, key=lambda value: (Fraction(used[value], pages[value]), ROLES.index(value)))
        result.append(pending[role].popleft())
        used[role] += 1
        if not pending[role]:
            del pending[role]
    return result
