"""Collect a bounded second metadata round without resetting the first run."""

import argparse
from collections import Counter, deque
from dataclasses import asdict
from datetime import timedelta
import json
import math
from pathlib import Path
import time
from urllib.parse import parse_qsl, urlsplit
import zipfile

from kenggo import expansion_source as source
from kenggo import rank_discovery


REQUEST_SCHEMA = "e2_rank_continuation_request_v1"
QUERY_SCHEMA = "e2_rank_continuation_queries_v1"
FOCUS_RANKS = ("15k", "14k", "13k")
CODE_PATH = "src/kenggo/rank_continuation.py"


def _ledger(reference):
    path = rank_discovery._reference(reference)
    if (not path.is_file() or path.stat().st_size > 16 * source.MIB
            or source.file_hash(path) != reference["sha256"]):
        raise ValueError("The previous request ledger changed or exceeds its limit")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not rows or any(row.get("sequence") != index for index, row in enumerate(rows, 1)):
        raise ValueError("The previous request ledger is incomplete")
    if any(type(row.get("bytes", 0)) is not int or row.get("bytes", 0) < 0 for row in rows):
        raise ValueError("Invalid previous response-byte accounting")
    return rows


def _successful_metadata(ledger):
    successful, origins = set(), set()
    prior_key, continuing = None, False
    for row in ledger:
        source.allowed_url(row["url"])
        parsed = urlsplit(row["url"])
        key = tuple(sorted((name, value.lower() if name == "user" else value)
                           for name, value in parse_qsl(parsed.query, keep_blank_values=True)))
        if row.get("method") != "GET" or parsed.path not in {"/gameArchives.jsp", "/archives.jsp"}:
            origins, prior_key, continuing = set(), None, False
            continue
        if not continuing or key != prior_key:
            origins = set()
        origins.add(row["url"])
        if row.get("status") == 200 and not row.get("error"):
            successful.update((url, row.get("sha256"), row.get("bytes", 0)) for url in origins)
        # The serial ledger omits Location headers. Only consecutive redirects
        # and retries for the same account/month may establish a response alias.
        continuing = bool(row.get("error")) or row.get("status") in {
            301, 302, 303, 307, 308, 408, 429, 500, 502, 503, 504}
        prior_key = key
    return successful


def _context(request_ref, fixture_only=False):
    request = source.read_ref(request_ref, 4 * source.MIB)
    if (set(request) != {"schema", "scope_decision", "previous", "selection", "source_revision"}
            or request["schema"] != REQUEST_SCHEMA or not isinstance(request["scope_decision"], str)
            or not request["scope_decision"].strip()):
        raise ValueError("Freeze a separate continuation decision and request")
    selection = request["selection"]
    if (set(selection) != {"pages", "max_train_candidates", "focus_ranks"}
            or selection["focus_ranks"] != list(FOCUS_RANKS)
            or type(selection["pages"]) is not int or not 1 <= selection["pages"] <= 200
            or type(selection["max_train_candidates"]) is not int
            or not 1 <= selection["max_train_candidates"] <= 5000):
        raise ValueError("Use at most 200 weak-rank pages and 5000 additional training candidates")
    code = request["source_revision"].get("code_sha256", {})
    if not fixture_only and (CODE_PATH not in code or any(source.file_hash(path) != pin for path, pin in code.items())):
        raise ValueError("The continuation code differs from its frozen revision")
    refs = request["previous"]
    if set(refs) != {"sources", "requests", "costs", "execution"}:
        raise ValueError("Pin the previous sources, ledger, final costs and execution marker")
    previous, costs, execution = (source.read_ref(refs[name]) for name in ("sources", "costs", "execution"))
    ledger = _ledger(refs["requests"])
    if (previous.get("schema") != source.SOURCES_SCHEMA
            or previous.get("status") not in {"completed", "partial"}
            or previous.get("stop_reason") is not None):
        raise ValueError("The previous collector must finish without a global stop")
    provenance = previous["provenance"]
    queries = source.read_ref(provenance["queries"])
    if (queries.get("schema") != source.QUERY_SCHEMA or queries.get("fixture_only") is not fixture_only
            or previous["request_sha256"] != queries["request"]["sha256"]
            or queries != source._query_contents(queries["request"], fixture_only)):
        raise ValueError("The previous frozen queries or original context changed")
    original, objects, original_limits, cache = source._context(queries["request"], fixture_only)
    if not fixture_only and "discovery_policy" not in original:
        raise ValueError("Continue the approved rank-discovery run")
    directory = Path(refs["sources"]["path"]).resolve().parent
    expected_execution = Path(provenance["queries"]["path"]).with_name(
        Path(provenance["queries"]["path"]).name + ".execution.json").resolve()
    if (Path(refs["requests"]["path"]).resolve() != directory / "requests.jsonl"
            or Path(refs["costs"]["path"]).resolve() != directory / "costs.json"
            or Path(refs["execution"]["path"]).resolve() != expected_execution
            or execution.get("queries_sha256") != provenance["queries"]["sha256"]
            or execution.get("request") != queries["request"]
            or Path(execution["output_directory"]).resolve() != directory):
        raise ValueError("The previous accounting and one-use marker belong to another run")
    discovery = source.read_ref(provenance["discovery"])
    pages = discovery["pages"]
    if (discovery["queries_sha256"] != queries["queries_sha256"]
            or len(pages) != len(queries["queries"])
            or any(page.get("status") != "retrieved" or any(page.get(key) != value for key, value in query.items())
                   for query, page in zip(queries["queries"], pages))
            or discovery["rows"] != [row for page in pages for row in page["rows"]]):
        raise ValueError("The previous metadata round is incomplete or changed")
    successful = _successful_metadata(ledger)
    for page in pages:
        html = rank_discovery._reference(page["html"])
        if (html.resolve().parent != directory or not html.is_file()
                or html.stat().st_size > original_limits.html_bytes
                or source.file_hash(html) != page["html"]["sha256"]
                or (page["url"], page["html"]["sha256"], html.stat().st_size) not in successful
                or source.parse_page(html.read_bytes(), page) != page["rows"]):
            raise ValueError("The previous archive metadata lacks matching HTML and request provenance")
    frozen = source.read_ref(provenance["frozen_candidates"])
    if frozen != source.freeze_candidates(queries, original, objects, discovery["rows"]):
        raise ValueError("The previous candidate selection changed")
    candidates = previous["candidates"]
    available_ids = {item["archive_row"]["url"] for item in candidates}
    frozen_by_id = {item["archive_row"]["url"]: item for item in frozen["candidates"]}
    if ([{key: item[key] for key in ("archive_row", "provisional_role")} for item in candidates]
            != [item for item in frozen["candidates"] if item["archive_row"]["url"] in available_ids]
            or len(available_ids) != len(candidates) or previous["groups"] != frozen["groups"]
            or any(item["source"]["source_url"] != item["archive_row"]["url"] for item in candidates)
            or previous["counts"]["frozen_candidates"] != len(frozen["candidates"])
            or previous["counts"]["accessible_candidates"] != len(candidates)):
        raise ValueError("The previous accessible candidates changed or left their frozen order")
    missing_ids = set(frozen_by_id) - available_ids
    reported_missing, failed_pages = set(), set()
    frozen_pages = {item["archive_row"]["page_key"] for item in frozen["candidates"]}
    for problem in previous.get("diagnostics", []):
        if problem.get("status") == "unavailable_candidate":
            key = problem.get("source_url")
            if (key not in missing_ids or key in reported_missing
                    or problem.get("reason") != "missing_or_ambiguous_member"):
                raise ValueError("A prior missing candidate has inconsistent evidence")
            reported_missing.add(key)
        elif problem.get("status") == "failed_bundle":
            key = problem.get("page_key")
            if (key not in frozen_pages or key in failed_pages
                    or not isinstance(problem.get("reason"), str) or not problem["reason"]
                    or any(item["archive_row"]["page_key"] == key for item in candidates)):
                raise ValueError("A prior failed bundle has inconsistent evidence")
            failed_pages.add(key)
        else:
            raise ValueError("Unsupported previous collection failure")
    if (reported_missing != missing_ids
            or previous["status"] != ("partial" if previous.get("diagnostics") else "completed")):
        raise ValueError("The previous collector omitted an unavailable candidate or failure")
    base_groups = objects.get("current_groups", original["base_groups"])
    if (previous["base_grouping_sha256"] != source.kgs_eligibility.grouping_digest(base_groups)
            or previous["owner_excluded_sgf_sha256"] != sorted(row["sgf_sha256"]
                for row in objects["owner_exclusions"]["excluded_games"])
            or provenance["source_revision"] != original["source_revision"]):
        raise ValueError("The previous source changed its base, owner exclusions or code provenance")
    exposure = {key: set(map(tuple, value)) if key == "account_months" else set(value)
                for key, value in objects["prior_exposure"].items()}
    for item in frozen["candidates"]:
        exposure["source_ids"].add(item["archive_row"]["url"])
        exposure["groups"].update(group for _, group in source.cached.participants(item["archive_row"], frozen["groups"]))
    for query in queries["queries"]:
        handle = query["user"].lower()
        exposure["groups"].add(frozen["groups"].get(handle, "kgs-handle:" + handle))
        exposure["account_months"].add((handle, f"2025-{query['month']:02d}"))
    if source.digest(previous["exposure"]) != source.digest({key: sorted(values) for key, values in exposure.items()}):
        raise ValueError("The previous exposure union changed")
    probe = source.read_ref(original["access_probe"]) if "access_probe" in original else {}
    for key in ("requests", "response_bytes", "collection_requests", "collection_response_bytes",
                "cpu_seconds", "cpu_charged_seconds", "wall_seconds"):
        if (type(costs.get(key)) not in (int, float) or not math.isfinite(costs[key]) or costs[key] < 0
                or costs[key] < previous["costs"][key]):
            raise ValueError("The previous final costs are invalid or omit recorded work")
    if (costs["collection_requests"] != len(ledger)
            or costs["collection_response_bytes"] != sum(row.get("bytes", 0) for row in ledger)
            or costs["requests"] != len(ledger) + probe.get("requests", 0)
            or costs["response_bytes"] != costs["collection_response_bytes"] + probe.get("response_bytes", 0)
            or not math.isclose(costs["cpu_charged_seconds"], costs["cpu_seconds"] + probe.get("cpu_charged_seconds", 0))):
        raise ValueError("The previous ledger, probe and final costs disagree")
    totals = {"requests": original_limits.requests + probe.get("requests", 0),
              "response_bytes": original_limits.response_bytes + probe.get("response_bytes", 0),
              "cpu_charged_seconds": original_limits.cpu_seconds + probe.get("cpu_charged_seconds", 0),
              "wall_seconds": original_limits.wall_seconds,
              "expanded_zip_bytes": original_limits.expanded_zip_bytes}
    expanded = previous["counts"]["expanded_zip_bytes"]
    if type(expanded) is not int or expanded < 0:
        raise ValueError("Invalid previous ZIP-directory accounting")
    remaining = {**asdict(original_limits),
        "requests": totals["requests"] - costs["requests"],
        "response_bytes": totals["response_bytes"] - costs["response_bytes"],
        "cpu_seconds": math.floor(totals["cpu_charged_seconds"] - costs["cpu_charged_seconds"]),
        "wall_seconds": math.floor(totals["wall_seconds"] - costs["wall_seconds"]),
        "expanded_zip_bytes": totals["expanded_zip_bytes"] - expanded,
        "max_train_candidates": selection["max_train_candidates"]}
    limits = source.Limits(**remaining)
    if limits.requests < 2 * selection["pages"] + 2:
        raise ValueError("Insufficient remaining requests for the frozen pages, ZIPs and login")
    first_start = min(source.deadline(execution["started_utc"]),
        source.deadline(ledger[0]["started_utc"]) - timedelta(seconds=ledger[0]["start_offset_seconds"]))
    end = min(source.deadline(original["deadline_utc"]), first_start + timedelta(seconds=totals["wall_seconds"]))
    return {"request": request, "original": original, "objects": objects, "previous": previous,
            "discovery": discovery, "frozen": frozen, "queries": queries, "costs": costs,
            "cache": cache, "limits": limits, "totals": totals, "first_start": first_start,
            "deadline_utc": end.isoformat()}


def _query_contents(request_ref, fixture_only=False):
    context = _context(request_ref, fixture_only)
    request, previous, objects = context["request"], context["previous"], context["objects"]
    groups, split = previous["groups"], objects["split_config"]
    validation, final, bots, existing = rank_discovery._context(groups, split, context["cache"])
    existing.update((query["user"].lower(), query["month"]) for query in context["queries"]["queries"])
    hints = rank_discovery._hints({"rows": objects["frame"]["rows"] + context["discovery"]["rows"]},
                                  groups, validation, final, bots)
    policy = context["original"].get("discovery_policy")
    counts = policy["baseline_counts"]["train_games_by_rank"] if policy else {}
    target = policy["target_train_games_per_rank"] if policy else 200
    weights = {rank: max(0, target - counts.get(rank, 0)) if rank in FOCUS_RANKS else 0
               for rank in source.kgs_source.RANKS}
    queries = rank_discovery._rank_pages(hints, existing, request["selection"]["pages"], weights, "train")
    if len(queries) != request["selection"]["pages"]:
        raise ValueError("The safe historical metadata cannot fill the continuation pages")
    frozen = {"schema": QUERY_SCHEMA, "fixture_only": fixture_only, "request": request_ref,
              "queries": queries, "queries_sha256": source.digest(queries),
              "limits": asdict(context["limits"]), "total_limits": context["totals"],
              "deadline_utc": context["deadline_utc"],
              "first_source_started_utc": context["first_start"].isoformat(),
              "prior_costs": context["costs"]}
    return frozen, context


def freeze_queries(request_path, request_sha256, output_path, *, fixture_only=False):
    request_ref = {"path": str(Path(request_path).resolve()), "sha256": request_sha256}
    frozen, _ = _query_contents(request_ref, fixture_only)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pin = source.write_json(output, frozen)
    return {"path": str(output.resolve()), "sha256": pin, "queries": len(frozen["queries"]),
            "deadline_utc": frozen["deadline_utc"], "remaining_limits": frozen["limits"]}


def _freeze_candidates(context, rows, request_ref):
    previous, objects = context["previous"], context["objects"]
    groups = dict(previous["groups"])
    old = context["frozen"]["candidates"]
    excluded = {item["archive_row"]["url"] for item in old}
    excluded.update(objects.get("completed_ids", ()))
    excluded.update(row["url"] for row in objects["original_sample"]["candidates"])
    excluded.update(row["source_url"] for row in objects["owner_exclusions"]["excluded_games"])
    source._unique([item["archive_row"] for item in old] + rows)
    eligible = [{"archive_row": row, "provisional_role": "train"} for row in source._unique(rows).values()
        if source._eligible(row, groups, objects["split_config"], excluded) == "train"
        and any(source.kgs_source.player(row[side])[1] in FOCUS_RANKS for side in ("black", "white"))]
    additional = source.balanced(eligible, context["request"]["selection"]["max_train_candidates"],
                                 seed=rank_discovery.SEED)
    for item in additional:
        for handle, group in source.cached.participants(item["archive_row"], groups):
            groups.setdefault(handle, group)
    queues = [deque(source.balanced([item for item in old + additional if item["provisional_role"] == role],
                                    len(old) + len(additional), seed=rank_discovery.SEED))
              for role in source.cached.ROLES]
    combined = []
    while any(queues):
        for queue in queues:
            if queue:
                combined.append(queue.popleft())
    return {"schema": "e2_rank_continuation_candidates_v1", "request_sha256": request_ref["sha256"],
            "groups": groups, "additional_candidates": additional, "candidates": combined,
            "additional_candidate_ids_sha256": source.digest([item["archive_row"]["url"] for item in additional]),
            "candidate_ids_sha256": source.digest([item["archive_row"]["url"] for item in combined]),
            "counts": dict(Counter(item["provisional_role"] for item in combined)),
            "selection_seed": rank_discovery.SEED,
            "selection": "Retain all first-round IDs; add bounded weak-rank training candidates; balance rank, month and archive account and interleave roles before reading SGFs."}


def acquire(queries_path, queries_sha256, output_directory, *, credentials_file=None,
            fixture_only=False, transport=None, now=source.utc_now, monotonic=time.monotonic,
            cpu=time.process_time, sleep=time.sleep):
    started, cpu_started = monotonic(), cpu()
    if transport is not None and not fixture_only:
        raise ValueError("A replacement HTTP transport is allowed only for fixtures")
    frozen = source.read_ref({"path": str(queries_path), "sha256": queries_sha256})
    if frozen.get("schema") != QUERY_SCHEMA or frozen.get("fixture_only") is not fixture_only:
        raise ValueError("Wrong continuation query scope")
    expected, context = _query_contents(frozen["request"], fixture_only)
    if frozen != expected:
        raise ValueError("Continuation inputs changed after freezing")
    output = Path(output_directory).resolve()
    if output.exists():
        raise ValueError("Use a fresh continuation output directory")
    execution = Path(queries_path).with_name(Path(queries_path).name + ".execution.json")
    source.write_json(execution, {"queries_sha256": queries_sha256, "request": frozen["request"],
                                 "output_directory": str(output), "started_utc": now().isoformat()})
    output.mkdir(parents=True, exist_ok=False)
    limits, previous = context["limits"], context["previous"]
    client = source.Client(output, limits, frozen["deadline_utc"], transport=transport,
                           now=now, monotonic=monotonic, cpu=cpu, sleep=sleep)
    client.started, client.cpu_started = started, cpu_started
    pages, rows, diagnostics, bundles, indexes = [], [], [], {}, {}
    expanded, stopped, candidate_manifest = 0, None, None
    authentication, outcome = "not_requested", "failed"

    def accounted_costs():
        round_costs = client.costs()
        prior = context["costs"]
        costs = {**round_costs,
            "requests": prior["requests"] + round_costs["requests"],
            "response_bytes": prior["response_bytes"] + round_costs["response_bytes"],
            "collection_requests": prior["collection_requests"] + round_costs["requests"],
            "collection_response_bytes": prior["collection_response_bytes"] + round_costs["response_bytes"],
            "access_probe_requests": prior.get("access_probe_requests", 0),
            "cpu_seconds": prior["cpu_seconds"] + round_costs["cpu_seconds"],
            "cpu_charged_seconds": prior["cpu_charged_seconds"] + round_costs["cpu_seconds"],
            "wall_seconds": max(prior["wall_seconds"], (now() - context["first_start"]).total_seconds())}
        return costs, round_costs

    def progress(phase):
        costs, round_costs = accounted_costs()
        source.cached.save_json(output / "progress.json", {"phase": phase, "stop_reason": stopped,
            "queries_expected": len(frozen["queries"]), "queries_recorded": len(pages),
            "query_statuses": dict(Counter(page["status"] for page in pages)),
            "additional_candidates": 0 if candidate_manifest is None else len(candidate_manifest["additional_candidates"]),
            "bundles_available": len(bundles), "bundles_indexed": len(indexes),
            "expanded_zip_bytes": previous["counts"]["expanded_zip_bytes"] + expanded,
            "diagnostic_count": len(diagnostics), "costs": costs, "round_costs": round_costs}, replace=True)

    try:
        first_body = None
        try:
            client.wait(limits.min_interval_seconds)
            if credentials_file is not None:
                progress("authentication")
                first_body = client.authenticate(credentials_file, frozen["queries"][0])
                authentication = "verified"
        except (source.SourceStop, source.RequestFailure, ValueError, OSError, UnicodeError) as error:
            stopped = str(error) if isinstance(error, source.SourceStop) else "authentication_failed"
            authentication = "failed" if credentials_file is not None else "not_requested"
            diagnostics.append({"status": "initial_request_failed", "reason": stopped})
        for index, query in enumerate(frozen["queries"]):
            if stopped:
                pages.append({"key": query["key"], "status": "unattempted", "reason": stopped})
                continue
            try:
                body = first_body if index == 0 and first_body is not None else client.get(query["url"], limits.html_bytes)
                parsed = source.parse_page(body, query)
                path = output / (query["key"] + ".html")
                path.write_bytes(body)
                pages.append({**query, "status": "retrieved", "html": source.reference(path), "rows": parsed})
                rows.extend(parsed)
            except source.SourceStop as error:
                stopped = str(error)
                pages.append({"key": query["key"], "status": "failed", "reason": stopped})
            except (source.RequestFailure, ValueError, UnicodeError) as error:
                pages.append({"key": query["key"], "status": "failed", "reason": str(error)})
            progress("archive_metadata")
        discovery_path = output / "discovery.json"
        source.write_json(discovery_path, {"queries_sha256": frozen["queries_sha256"], "pages": pages, "rows": rows})
        try:
            candidate_manifest = _freeze_candidates(context, rows, frozen["request"])
        except ValueError as error:
            stopped = "candidate_freeze_failed"
            diagnostics.append({"status": stopped, "reason": str(error)})
            candidate_manifest = _freeze_candidates(context, [], frozen["request"])
        candidate_path = output / "frozen_candidates.json"
        source.write_json(candidate_path, candidate_manifest)
        needed = {item["archive_row"]["page_key"]: item["archive_row"]["bundle_url"]
                  for item in candidate_manifest["additional_candidates"]}
        for key, url in sorted(needed.items()):
            if stopped:
                diagnostics.append({"page_key": key, "status": "unattempted_bundle", "reason": stopped})
                continue
            try:
                source.allowed_url(url, bundle=True)
                body = client.get(url, limits.bundle_bytes, bundle=True)
                if not body.startswith(b"PK"):
                    source.parse_page(body, {"user": key.rsplit("-2025-", 1)[0], "month": int(key[-2:]), "key": key})
                    raise ValueError("invalid_zip_response")
                path = output / (key + ".zip")
                path.write_bytes(body)
                bundles[key] = {**source.reference(path), "bytes": len(body), "url": url}
                indexes[key], expanded = source._index_bundle(bundles[key], limits, expanded)
            except source.SourceStop as error:
                stopped = str(error)
                diagnostics.append({"page_key": key, "status": "failed_bundle", "reason": stopped})
            except (source.RequestFailure, ValueError, UnicodeError, zipfile.BadZipFile) as error:
                diagnostics.append({"page_key": key, "status": "failed_bundle", "reason": str(error)})
            progress("bundle_retrieval")
        accessible = {item["archive_row"]["url"]: item for item in previous["candidates"]}
        for item in candidate_manifest["additional_candidates"]:
            row = item["archive_row"]
            matches = indexes.get(row["page_key"], {}).get(source.e1_audit.source_key(row["url"]), [])
            if len(matches) != 1:
                diagnostics.append({"source_url": row["url"], "status": "unavailable_candidate",
                                    "reason": "missing_or_ambiguous_member"})
                continue
            bundle = bundles[row["page_key"]]
            accessible[row["url"]] = {**item, "source": {"source_url": row["url"], "archive_path": bundle["path"],
                "archive_sha256": bundle["sha256"], "zip_member": matches[0]}}
        exposure = {key: set(map(tuple, value)) if key == "account_months" else set(value)
                    for key, value in previous["exposure"].items()}
        for item in candidate_manifest["additional_candidates"]:
            row = item["archive_row"]
            exposure["source_ids"].add(row["url"])
            exposure["groups"].update(group for _, group in source.cached.participants(row, candidate_manifest["groups"]))
        for query in frozen["queries"]:
            exposure["groups"].add(candidate_manifest["groups"].get(query["user"], "kgs-handle:" + query["user"]))
            exposure["account_months"].add((query["user"], f"2025-{query['month']:02d}"))
        costs, round_costs = accounted_costs()
        result = {"schema": source.SOURCES_SCHEMA, "request_sha256": frozen["request"]["sha256"],
            "base_grouping_sha256": previous["base_grouping_sha256"], "groups": candidate_manifest["groups"],
            "candidates": [accessible[item["archive_row"]["url"]] for item in candidate_manifest["candidates"]
                           if item["archive_row"]["url"] in accessible],
            "owner_excluded_sgf_sha256": previous["owner_excluded_sgf_sha256"],
            "exposure": {key: sorted(value) for key, value in exposure.items()},
            "provenance": {"queries": {"path": str(Path(queries_path).resolve()), "sha256": queries_sha256},
                "discovery": source.reference(discovery_path), "frozen_candidates": source.reference(candidate_path),
                "previous": context["request"]["previous"], "scope_decision": context["request"]["scope_decision"],
                "source_revision": context["request"]["source_revision"]},
            "counts": {"frozen_candidates": len(candidate_manifest["candidates"]), "accessible_candidates": len(accessible),
                "previous_unavailable_candidates": previous["counts"]["frozen_candidates"] - len(previous["candidates"]),
                "additional_frozen_candidates": len(candidate_manifest["additional_candidates"]),
                "additional_accessible_candidates": len(accessible) - len(previous["candidates"]),
                "candidate_roles": candidate_manifest["counts"],
                "expanded_zip_bytes": previous["counts"]["expanded_zip_bytes"] + expanded,
                "query_statuses": dict(Counter(page["status"] for page in pages))},
            "authentication": authentication, "diagnostics": diagnostics,
            "status": "authentication_blocked" if authentication == "failed" or stopped == "archive_authentication_required"
                else "partial" if previous["status"] == "partial" or stopped or diagnostics or any(page["status"] != "retrieved" for page in pages)
                else "completed", "stop_reason": stopped, "costs": costs, "round_costs": round_costs}
        path = output / "sources.json"
        pin = source.write_json(path, result)
        outcome = result["status"]
        return {"path": str(path), "sha256": pin, "status": outcome, "counts": result["counts"]}
    finally:
        client.log.close()
        costs, round_costs = accounted_costs()
        source.write_json(output / "costs.json", {**costs, "round_costs": round_costs})
        progress(outcome)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("freeze", "acquire"))
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--inputs-sha256", required=True)
    parser.add_argument("--credentials-file", type=Path)
    args = parser.parse_args()
    if args.operation == "freeze" and args.credentials_file is not None:
        parser.error("Credentials are used only by acquire")
    result = (freeze_queries if args.operation == "freeze" else acquire)(args.inputs, args.inputs_sha256, args.output,
        **({"credentials_file": args.credentials_file} if args.operation == "acquire" else {}))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
