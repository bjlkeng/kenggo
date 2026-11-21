"""Freeze training-only KGS sources, then collect bounded additional metadata."""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import time
import zipfile

from kenggo import expansion_source as source
from kenggo import rank_discovery


SCHEMA = "e2_data_growth_sources_v1"
MIB = 1024**2
_EXPOSURE_FIELDS = set(source.EXPOSURE_FIELDS)
_ROW_FIELDS = ("black", "white", "setup", "type", "result")


@dataclass(frozen=True)
class SourceLimits:
    metadata_bytes: int = 512 * MIB
    archive_bytes: int = 8 * MIB
    expanded_zip_bytes: int = 4 * 1024**3
    member_bytes: int = MIB
    output_bytes: int = 256 * MIB
    max_candidates: int | None = None

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key == "max_candidates" and value is None:
                continue
            if type(value) is not int or value <= 0:
                raise ValueError("A source limit must be a positive integer: " + key)
        if self.archive_bytes > 8 * MIB or self.member_bytes > MIB:
            raise ValueError("Preserve the existing archive and member size limits")


@dataclass(frozen=True)
class CollectionLimits:
    pages: int = 1000
    requests: int = 2400
    response_bytes: int = 1024 * MIB
    cpu_seconds: int = 900
    wall_seconds: int = 21600
    min_interval_seconds: int = 5
    retries: int = 2

    def __post_init__(self):
        if type(self.pages) is not int or not 1 <= self.pages <= 1150:
            raise ValueError("Collect at most 1150 frozen archive pages in one round")
        self.network_limits()

    def network_limits(self):
        return source.Limits(**{key: value for key, value in asdict(self).items() if key != "pages"})


class _Budget:
    def __init__(self, limits, check_budget, counters):
        self.limits = limits
        self.check = check_budget or (lambda: None)
        self.counters = counters if counters is not None else {}
        self.costs = Counter()
        self.objects = {}
        self.output_reserve = 0

    def charge(self, name, amount):
        self.costs[name] += amount
        key = "source_" + name
        self.counters[key] = self.counters.get(key, 0) + amount

    def read(self, reference):
        path = rank_discovery._reference(reference).resolve()
        identity = (str(path), reference["sha256"])
        if identity in self.objects:
            return self.objects[identity]
        self.check()
        if not path.is_file() or path.stat().st_size > self.limits.metadata_bytes - self.costs["metadata_bytes"]:
            raise ValueError("Missing metadata or metadata byte limit exceeded")
        body = path.read_bytes()
        self.charge("metadata_bytes", len(body))
        if sha256(body).hexdigest() != reference["sha256"]:
            raise ValueError("Pinned source metadata changed")
        self.check()
        result = json.loads(body)
        self.objects[identity] = result
        return result

    def save(self, path, value):
        self.check()
        body = source.encoded(value)
        if len(body) + 1 + self.costs["output_bytes"] + self.output_reserve > self.limits.output_bytes:
            raise source.SourceStop("source_output_byte_limit")
        pin = source.write_json(path, value)
        self.charge("output_bytes", Path(path).stat().st_size)
        return {"path": str(Path(path).resolve()), "sha256": pin}

    def save_bytes(self, path, body):
        self.check()
        if len(body) + self.costs["output_bytes"] + self.output_reserve > self.limits.output_bytes:
            raise source.SourceStop("source_output_byte_limit")
        with Path(path).open("xb") as stream:
            stream.write(body)
        self.charge("output_bytes", len(body))
        return {"path": str(Path(path).resolve()), "sha256": sha256(body).hexdigest()}


def _groups(value):
    if not isinstance(value, dict) or not value:
        raise ValueError("A source needs the frozen account registry")
    for handle, group in value.items():
        if (source.kgs_eligibility.normalize_handle(handle) != handle or not isinstance(group, str)
                or not group.startswith("kgs-handle:")):
            raise ValueError("Invalid source account group")
        source.kgs_eligibility.normalize_handle(group.removeprefix("kgs-handle:"))


def _exposure(value):
    if not isinstance(value, dict) or set(value) != _EXPOSURE_FIELDS:
        raise ValueError("Preserve every exposure field")
    result = {}
    for key, values in value.items():
        if not isinstance(values, list):
            raise ValueError("Exposure fields must be lists")
        if key == "account_months":
            for item in values:
                if (not isinstance(item, list) or len(item) != 2
                        or source.kgs_eligibility.normalize_handle(item[0]) != item[0]
                        or re.fullmatch(r"2025-(?:0[1-9]|1[0-2])", item[1]) is None):
                    raise ValueError("Invalid exposed account-month")
            result[key] = set(map(tuple, values))
        else:
            if any(not isinstance(item, str) or not item for item in values):
                raise ValueError("Invalid exposure entry")
            result[key] = set(values)
    return result


def _union_exposure(destination, extra):
    for key, values in _exposure(extra).items():
        destination[key].update(values)


def _add_bundle(bundles, row, item):
    if (not isinstance(item, dict)
            or set(item) != {"source_url", "archive_path", "archive_sha256", "zip_member"}
            or item["source_url"] != row["url"]
            or source.e1_audit.member_key(item["zip_member"]) != source.e1_audit.source_key(row["url"])):
        raise ValueError("A cached candidate changed its source identity")
    source.allowed_url(row["bundle_url"], bundle=True)
    ref = {"path": item["archive_path"], "sha256": item["archive_sha256"]}
    rank_discovery._reference(ref)
    bundle = {**ref, "url": row["bundle_url"]}
    _register_bundle(bundles, row["page_key"], bundle)


def _register_bundle(bundles, key, bundle):
    if (not re.fullmatch(r"[A-Za-z0-9]+-2025-(?:0[1-9]|1[0-2])", key)
            or set(bundle) != {"path", "sha256", "url"}):
        raise ValueError("Invalid cached archive reference")
    rank_discovery._reference({name: bundle[name] for name in ("path", "sha256")})
    source.allowed_url(bundle["url"], bundle=True)
    previous = bundles.setdefault(key, bundle)
    if previous["sha256"] != bundle["sha256"] or previous["url"] != bundle["url"]:
        raise ValueError("One cached archive page has conflicting bundle provenance")
    if str(Path(bundle["path"]).resolve()) < str(Path(previous["path"]).resolve()):
        bundles[key] = bundle


def _context(parent_ref, discovery_refs, prior_source_refs, budget, fixture_only):
    if type(fixture_only) is not bool or not prior_source_refs or not isinstance(discovery_refs, list):
        raise ValueError("Provide source provenance and an explicit fixture scope")
    parent = budget.read(parent_ref)
    if parent.get("schema") not in {"e2_komi_expansion_v1", "e2_data_growth_v1"}:
        raise ValueError("Growth must preserve the combined 6.5/0.5-komi parent")
    artifacts = parent["artifacts"]
    groups = dict(budget.read(artifacts["groups"]))
    _groups(groups)
    split_manifest = budget.read(artifacts["split_manifest"])
    split = split_manifest["plan"]
    if (split_manifest.get("fixture_only") is not fixture_only
            or set(split) != {"grouping_sha256", "train_before", "confirmation_from", "validation_groups", "test_groups"}
            or split["grouping_sha256"] != source.kgs_eligibility.grouping_digest(groups)
            or split["train_before"] != "2025-09-01" or split["confirmation_from"] != "2025-11-01"
            or set(split["validation_groups"]) & set(split["test_groups"])
            or not set(split["validation_groups"] + split["test_groups"]) <= set(groups.values())):
        raise ValueError("The parent changed its frozen groups, cohorts, dates or fixture scope")
    index = budget.read(artifacts["base_index"])
    if (index.get("schema") not in {"e2_komi_data_index_v1", "e2_data_growth_index_v1"}
            or index["canonical_sha256"] != artifacts["canonical"]["sha256"]
            or index["split_sha256"] != artifacts["split_manifest"]["sha256"]):
        raise ValueError("The source history index belongs to another parent")
    excluded = set(split_manifest.get("sealed_source_ids", []))
    for key in ("records", "rejected", "study_rejections"):
        for row in index.get(key, []):
            excluded.add(row["source_url"])
    excluded.update(index.get("processed_source_ids", []))
    exposure = _exposure(budget.read(artifacts["exposure"]))
    rows, bundles, owner_hashes = [], {}, None
    for reference in prior_source_refs:
        previous = budget.read(reference)
        if (previous.get("schema") not in {source.SOURCES_SCHEMA, SCHEMA}
                or previous.get("fixture_only", fixture_only) is not fixture_only):
            raise ValueError("Unsupported previous growth source scope")
        old_groups = previous["groups"]
        _groups(old_groups)
        for handle, group in old_groups.items():
            if handle in groups and groups[handle] != group:
                raise ValueError("A previous source changed an existing account group")
            groups.setdefault(handle, group)
        hashes = previous["owner_excluded_sgf_sha256"]
        if (not isinstance(hashes, list)
                or any(not isinstance(pin, str) or not re.fullmatch("[0-9a-f]{64}", pin) for pin in hashes)
                or owner_hashes is not None and set(hashes) != owner_hashes):
            raise ValueError("Previous sources disagree on owner exclusions")
        owner_hashes = set(hashes)
        _union_exposure(exposure, previous["exposure"])
        for item in previous["candidates"]:
            row = item["archive_row"]
            _add_bundle(bundles, row, item["source"])
            rows.append(row)
        for key, bundle in previous.get("cached_bundles", {}).items():
            _register_bundle(bundles, key, bundle)
        if previous["schema"] == source.SOURCES_SCHEMA:
            frozen = budget.read(previous["provenance"]["frozen_candidates"])
            frozen_by_id = {item["archive_row"]["url"]: item for item in frozen["candidates"]}
            if len(frozen_by_id) != len(frozen["candidates"]):
                raise ValueError("Historical frozen source identities are not unique")
            if any(item["archive_row"] != frozen_by_id.get(item["archive_row"]["url"], {}).get("archive_row")
                   for item in previous["candidates"]):
                raise ValueError("Historical source metadata left its frozen frame")
            excluded.update(frozen_by_id)
    for reference in discovery_refs:
        frame = budget.read(reference)
        if not isinstance(frame.get("rows"), list):
            raise ValueError("Discovery must retain its archive metadata rows")
        rows.extend(frame["rows"])
        for page in frame.get("pages", []):
            match = re.fullmatch(r"([A-Za-z0-9]+)-2025-(0[1-9]|1[0-2])", page["key"])
            if match:
                exposure["account_months"].add((match[1].lower(), "2025-" + match[2]))
    return {"parent_summary": parent_ref, "base_grouping_sha256": split["grouping_sha256"],
            "groups": groups, "split": split, "excluded": excluded, "exposure": exposure,
            "rows": rows, "bundles": bundles, "owner_hashes": owner_hashes,
            "discovery_refs": discovery_refs, "prior_source_refs": prior_source_refs}


def _candidate_rows(context, budget, extra_rows=()):
    aliases = defaultdict(dict)
    signatures = {}
    for number, row in enumerate([*context["rows"], *extra_rows]):
        if number % 256 == 0:
            budget.check()
        if not isinstance(row, dict) or any(not isinstance(row.get(key), str)
                for key in ("url", "page_key", "bundle_url", *_ROW_FIELDS)):
            raise ValueError("Incomplete archive metadata row")
        signature = tuple(row[key] for key in _ROW_FIELDS)
        if signatures.setdefault(row["url"], signature) != signature:
            raise ValueError("Conflicting archive metadata for one source ID")
        if source._eligible(row, context["groups"], context["split"], context["excluded"]) != "train":
            continue
        match = re.fullmatch(r"([A-Za-z0-9]+)-2025-(0[1-9]|1[0-2])", row["page_key"])
        day = source.e1_audit.source_key(row["url"])[0]
        if (not match or int(match[2]) != int(day[5:7])
                or match[1].lower() not in {handle for handle, _ in source.cached.participants(row, context["groups"])}):
            raise ValueError("Archive metadata leaves its account-month page")
        source.allowed_url(row["bundle_url"], bundle=True)
        aliases[row["url"]][source.encoded(row)] = row
    return {key: sorted(values.values(), key=lambda row: (row["page_key"], source.encoded(row)))
            for key, values in aliases.items()}


def _index_bundle(bundle, budget):
    budget.check()
    path = Path(bundle["path"])
    if not path.is_file():
        return None
    if path.stat().st_size > budget.limits.archive_bytes:
        raise ValueError("Cached archive exceeds its byte limit")
    pin = sha256()
    with path.open("rb") as stream:
        while block := stream.read(MIB):
            budget.check()
            pin.update(block)
            budget.charge("archive_bytes_read", len(block))
    if pin.hexdigest() != bundle["sha256"]:
        raise ValueError("Cached archive hash changed")
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
    expanded = sum(member.file_size for member in entries)
    if expanded + budget.costs["expanded_zip_bytes"] > budget.limits.expanded_zip_bytes:
        raise source.SourceStop("source_expanded_zip_limit")
    budget.charge("expanded_zip_bytes", expanded)
    budget.charge("archives_indexed", 1)
    result = defaultdict(list)
    for entry in entries:
        key = source.e1_audit.member_key(entry.filename)
        if key is not None:
            result[key].append(entry)
    return result


def _available(aliases, bundles, budget):
    indexes, candidates, diagnostics = {}, [], []
    for key, rows in sorted(aliases.items()):
        budget.check()
        reason = "missing_cached_archive"
        for row in rows:
            bundle = bundles.get(row["page_key"])
            if bundle is None:
                continue
            if row["bundle_url"] != bundle["url"]:
                raise ValueError("Discovery row and cached archive URL disagree")
            identity = (str(Path(bundle["path"]).resolve()), bundle["sha256"])
            if identity not in indexes:
                indexes[identity] = _index_bundle(bundle, budget)
            index = indexes[identity]
            if index is None:
                continue
            entries = index.get(source.e1_audit.source_key(key), [])
            if len(entries) != 1:
                reason = "missing_or_ambiguous_member"
                continue
            entry = entries[0]
            if entry.is_dir() or entry.flag_bits & 1 or entry.file_size > budget.limits.member_bytes:
                reason = "unsupported_or_oversized_member"
                continue
            candidates.append({"archive_row": row, "provisional_role": "train", "source": {
                "source_url": key, "archive_path": str(Path(bundle["path"]).resolve()),
                "archive_sha256": bundle["sha256"], "zip_member": entry.filename}})
            break
        else:
            diagnostics.append({"source_url": key, "reason": reason})
    return candidates, diagnostics


def _manifest(context, aliases, candidates, diagnostics, seed, budget, fixture_only, provenance=None):
    if not isinstance(seed, str) or not seed:
        raise ValueError("Freeze a nonempty source selection seed")
    ordered = source.balanced(candidates, len(candidates), seed=seed)
    maximum = budget.limits.max_candidates
    selected = ordered if maximum is None else ordered[:maximum]
    groups = dict(context["groups"])
    exposure = {key: set(values) for key, values in context["exposure"].items()}
    for row_set in aliases.values():
        for row in row_set:
            exposure["source_ids"].add(row["url"])
            for handle, group in source.cached.participants(row, groups):
                groups.setdefault(handle, group)
                exposure["groups"].add(group)
    return {"schema": SCHEMA, "fixture_only": fixture_only,
            "parent_summary": context["parent_summary"], "base_grouping_sha256": context["base_grouping_sha256"],
            "groups": groups, "candidates": selected,
            "cached_bundles": context["bundles"],
            "owner_excluded_sgf_sha256": sorted(context["owner_hashes"]),
            "exposure": {key: sorted(values) for key, values in exposure.items()},
            "provenance": {"discoveries": context["discovery_refs"], "prior_sources": context["prior_source_refs"],
                "selection_seed": seed, "selection": "Training-only; balance weaker rank, month and archive account before reading SGF members.",
                **(provenance or {})},
            "candidate_ids_sha256": source.digest([item["archive_row"]["url"] for item in selected]),
            "counts": {"eligible_source_ids": len(aliases), "accessible_candidates": len(candidates),
                "selected_candidates": len(selected), "deferred_candidates": len(candidates) - len(selected),
                "unavailable_candidates": len(diagnostics)},
            "cache_exhausted": not candidates, "status": "cache_ready" if selected else "cache_exhausted",
            "diagnostics": diagnostics, "costs": {**budget.costs, "http_requests": 0, "sgf_members_decoded": 0}}


def freeze_sources(parent_summary_ref, discovery_refs, prior_source_refs, output_path, *, seed,
                   limits=None, check_budget=None, counters=None, fixture_only=False):
    """Pin cached members without decoding SGFs; counters accumulate across calls.

    Byte and candidate limits apply to this invocation. The owner supplies the
    cumulative deadline and resource checks through check_budget(). Historical
    non-growth candidate IDs stay excluded; unattempted growth IDs remain usable.
    """
    budget = _Budget(limits or SourceLimits(), check_budget, counters)
    context = _context(parent_summary_ref, discovery_refs, prior_source_refs, budget, fixture_only)
    aliases = _candidate_rows(context, budget)
    candidates, diagnostics = _available(aliases, context["bundles"], budget)
    manifest = _manifest(context, aliases, candidates, diagnostics, seed, budget, fixture_only)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    reference = budget.save(path, manifest)
    return {**reference, "status": manifest["status"], "counts": manifest["counts"],
            "cache_exhausted": manifest["cache_exhausted"]}


def _queries(context, maximum):
    groups, split = context["groups"], context["split"]
    validation, final = set(split["validation_groups"]), set(split["test_groups"])
    bots = {group for handle, group in groups.items()
            if "bot" in handle or handle in source.kgs_eligibility.HISTORICAL_BOT_HANDLES}
    bots.update(groups.get(handle, "kgs-handle:" + handle)
                for handle in source.kgs_eligibility.HISTORICAL_BOT_HANDLES)
    hints = rank_discovery._hints({"rows": context["rows"]}, groups, validation, final, bots)
    existing = {(handle, int(month[-2:])) for handle, month in context["exposure"]["account_months"]}
    return rank_discovery._rank_pages(hints, existing, maximum,
                                     {rank: 1 for rank in source.kgs_source.RANKS}, "train")


def collect_more(parent_summary_ref, discovery_refs, prior_source_refs, output_directory, *, seed,
                 deadline_utc, limits=None, source_limits=None, credentials_file=None,
                 check_budget=None, counters=None, fixture_only=False, transport=None,
                 now=source.utc_now, monotonic=time.monotonic, cpu=time.process_time, sleep=time.sleep):
    """Freeze safe training queries and retrieve within a fresh, bounded collection.

    Pass the returned discovery and source references into later calls. Every
    attempted page, failure and request remains in this one-use directory. The
    caller's budget check must enforce the single deadline across all rounds.
    """
    if transport is not None and not fixture_only:
        raise ValueError("Replacement transport is allowed only for synthetic fixtures")
    source.deadline(deadline_utc)
    started, cpu_started = monotonic(), cpu()
    limits = limits or CollectionLimits()
    budget = _Budget(source_limits or SourceLimits(output_bytes=2 * 1024**3), check_budget, counters)
    if budget.limits.output_bytes < MIB:
        raise ValueError("Collection output must reserve one MiB for its recovery record")
    budget.output_reserve = MIB
    context = _context(parent_summary_ref, discovery_refs, prior_source_refs, budget, fixture_only)
    _candidate_rows(context, budget)
    queries = _queries(context, limits.pages)
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=False)
    frozen_ref = budget.save(output / "queries.json", {"schema": "e2_data_growth_queries_v1",
        "parent_summary": parent_summary_ref, "discoveries": discovery_refs, "prior_sources": prior_source_refs,
        "queries": queries, "queries_sha256": source.digest(queries), "query_selection_seed": rank_discovery.SEED,
        "candidate_selection_seed": seed, "limits": asdict(limits), "source_limits": asdict(budget.limits),
        "deadline_utc": deadline_utc, "fixture_only": fixture_only, "started_utc": now().isoformat()})

    class Client(source.Client):
        def check(self):
            for key in ("requests", "response_bytes"):
                current = getattr(self, key)
                budget.charge(key, current - budget.costs[key])
            log_bytes = self.log.tell()
            if log_bytes > budget.costs["request_log_bytes"]:
                amount = log_bytes - budget.costs["request_log_bytes"]
                budget.charge("request_log_bytes", amount)
                budget.charge("output_bytes", amount)
            if budget.costs["output_bytes"] + budget.output_reserve > budget.limits.output_bytes:
                raise source.SourceStop("source_output_byte_limit")
            budget.check()
            super().check()

    def bounded_sleep(seconds):
        end = monotonic() + seconds
        while monotonic() < end:
            budget.check()
            sleep(min(30, end - monotonic()))

    client = Client(output, limits.network_limits(), deadline_utc, transport=transport,
                    now=now, monotonic=monotonic, cpu=cpu, sleep=bounded_sleep)
    client.started, client.cpu_started = started, cpu_started
    pages, rows, problems = [], [], []
    bundles = dict(context["bundles"])
    stopped, authentication, result = None, "not_requested", None
    try:
        first_body = None
        if queries:
            try:
                client.wait(limits.min_interval_seconds)
            except source.SourceStop as error:
                stopped = str(error)
        if credentials_file is not None and queries and not stopped:
            try:
                first_body = client.authenticate(credentials_file, queries[0])
                authentication = "verified"
            except (source.SourceStop, source.RequestFailure, ValueError, OSError, UnicodeError) as error:
                authentication = "failed"
                stopped = str(error) if isinstance(error, source.SourceStop) else "authentication_failed"
        for number, query in enumerate(queries):
            handle = query["user"].lower()
            context["exposure"]["groups"].add(context["groups"].get(handle, "kgs-handle:" + handle))
            context["exposure"]["account_months"].add((handle, f"2025-{query['month']:02d}"))
            if stopped:
                pages.append({"key": query["key"], "status": "unattempted", "reason": stopped})
                continue
            try:
                body = first_body if number == 0 and first_body is not None else client.get(query["url"], 2 * MIB)
                parsed = source.parse_page(body, query)
                html_ref = budget.save_bytes(output / (query["key"] + ".html"), body)
                pages.append({**query, "status": "retrieved", "html": html_ref, "rows": parsed})
                rows.extend(parsed)
            except source.SourceStop as error:
                stopped = str(error)
                pages.append({"key": query["key"], "status": "failed", "reason": stopped})
            except (source.RequestFailure, ValueError, UnicodeError) as error:
                pages.append({"key": query["key"], "status": "failed", "reason": type(error).__name__})
        discovery_ref = budget.save(output / "discovery.json", {
            "schema": "e2_data_growth_discovery_v1", "queries_sha256": source.digest(queries), "pages": pages, "rows": rows})
        aliases = _candidate_rows(context, budget, rows)
        frozen_candidates = source.balanced([{"archive_row": alternatives[0], "provisional_role": "train"}
                                             for alternatives in aliases.values()], len(aliases), seed=seed)
        candidates_ref = budget.save(output / "frozen_candidates.json", {
            "schema": "e2_data_growth_candidates_v1", "queries": frozen_ref,
            "candidates": frozen_candidates, "candidate_ids_sha256": source.digest(
                [item["archive_row"]["url"] for item in frozen_candidates])})
        needed = {}
        for item in frozen_candidates:
            row = item["archive_row"]
            alternatives = aliases[row["url"]]
            if any(candidate["page_key"] in bundles and Path(bundles[candidate["page_key"]]["path"]).is_file()
                   for candidate in alternatives):
                continue
            needed[row["page_key"]] = row["bundle_url"]
        for key, url in sorted(needed.items()):
            if stopped:
                problems.append({"page_key": key, "reason": stopped, "status": "unattempted_bundle"})
                continue
            try:
                source.allowed_url(url, bundle=True)
                body = client.get(url, budget.limits.archive_bytes, bundle=True)
                if not body.startswith(b"PK"):
                    if re.search(br'name\s*=\s*[\"\']password[\"\']', body, re.I):
                        raise source.AuthenticationRequired("archive_authentication_required")
                    raise ValueError("Invalid ZIP response")
                ref = budget.save_bytes(output / (key + ".zip"), body)
                with zipfile.ZipFile(ref["path"]):
                    pass
                bundles[key] = {**ref, "url": url}
            except source.SourceStop as error:
                stopped = str(error)
                problems.append({"page_key": key, "reason": stopped, "status": "failed_bundle"})
            except (source.RequestFailure, ValueError, OSError, zipfile.BadZipFile) as error:
                problems.append({"page_key": key, "reason": type(error).__name__, "status": "failed_bundle"})
        candidates, diagnostics = _available(aliases, bundles, budget)
        context["discovery_refs"] = [*discovery_refs, discovery_ref]
        context["bundles"] = bundles
        manifest = _manifest(context, aliases, candidates, diagnostics, seed, budget, fixture_only,
                             {"queries": frozen_ref, "frozen_candidates": candidates_ref})
        manifest.update(authentication=authentication, stop_reason=stopped,
            status="authentication_blocked" if authentication == "failed" or stopped == "archive_authentication_required"
                else "partial" if stopped or diagnostics or problems or any(page["status"] != "retrieved" for page in pages)
                else "completed" if queries else "no_discovery_queries", collection_diagnostics=problems,
            costs={**budget.costs, **client.costs(), "sgf_members_decoded": 0})
        sources_ref = budget.save(output / "sources.json", manifest)
        result = {**sources_ref, "status": manifest["status"], "counts": manifest["counts"],
                  "cache_exhausted": manifest["cache_exhausted"], "discovery": discovery_ref,
                  "queries": frozen_ref, "queries_selected": len(queries)}
        return result
    finally:
        log_bytes = client.log.tell()
        if log_bytes > budget.costs["request_log_bytes"]:
            amount = log_bytes - budget.costs["request_log_bytes"]
            budget.charge("request_log_bytes", amount)
            budget.charge("output_bytes", amount)
        client.log.close()
        for key in ("requests", "response_bytes"):
            budget.charge(key, getattr(client, key) - budget.costs[key])
        progress = {
            "status": result["status"] if result else "interrupted", "stop_reason": stopped,
            "queries": frozen_ref, "pages": [{key: page[key] for key in ("key", "status", "reason") if key in page}
                                            for page in pages], "bundle_diagnostics": problems,
            "costs": {**budget.costs, **client.costs(), "sgf_members_decoded": 0}}
        body_size = len(source.encoded(progress)) + 1
        if body_size > budget.output_reserve:
            raise ValueError("Collection recovery record exceeds its reserved bytes")
        source.write_json(output / "collection_progress.json", progress)
        budget.charge("output_bytes", body_size)
