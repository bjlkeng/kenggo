"""One reviewed, bounded source check for the missing 11k/6.5 validation cell."""

import argparse
from collections import Counter
from hashlib import sha256
import html
import json
from pathlib import Path
import re
import subprocess
import time
from urllib.parse import parse_qs, quote, quote_plus, urlsplit
import zipfile

from kenggo import expansion_source as source
from kenggo import data_expansion, kgs_admission, kgs_eligibility, rank_discovery
from kenggo.canonical_data import canonicalize, digest
from kenggo.data_splits import SplitPlan, build_splits
from kenggo.population_audit import split_entry
from kenggo.sgf_audit import inspect_sgf


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SCHEMA = "e2_validation_gap_source_v1"
COMMIT = "59d29abf57d8c03eec783a30222fa288403c8e7b"
DEADLINE = "2026-11-28T14:13:03.148730+00:00"
MIB = 1024**2
SLOTS = (("cerepi", 9), ("dexmon", 9), ("ekw16", 9), ("iral", 9),
         ("kam2015", 9), ("le0nid", 10), ("mskcx5", 9), ("pachacamac", 10),
         ("rmgorbunov", 9), ("sirbernd", 9), ("vel777", 9), ("viandierer", 10))
LIMITS = {"cpu_seconds": 1800, "wall_seconds": 3600, "output_bytes": 1024 * MIB,
          "selected_sgf_bytes": 64 * MIB, "expanded_zip_bytes": 256 * MIB,
          "max_candidates": 256, "target_games": 8}
FINGERPRINTS = ("sgf_sha256", "audit_sequence_sha256", "trajectory_sha256")
FINAL_RESERVE = MIB
STOP_REASONS = {"absolute_deadline", "phase_wall_limit", "phase_cpu_limit", "phase_output_limit",
                "source_wall_limit", "source_cpu_limit", "request_or_response_limit", "response_byte_limit",
                "insufficient_time_for_request", "selected_sgf_limit", "expanded_zip_limit",
                "archive_authentication_required", "unsupported_archive_login_form", "login_post_redirect_refused"}


def read(ref, maximum=128 * MIB):
    return source.read_ref(ref, maximum)


def error_reason(error):
    # Parser and HTTP exception messages can contain untrusted response text.
    return str(error) if isinstance(error, source.SourceStop) and str(error) in STOP_REASONS else type(error).__name__


def queries():
    return [rank_discovery._query(handle, month, "validation_development", "11k")
            for handle, month in SLOTS]


def metadata_reason(row, groups, plan, blocked):
    if row["url"] in blocked:
        return "previous_source_id"
    reasons = source.kgs_source.candidate_reasons(row)
    if reasons:
        return "+".join(reasons)
    if any(handle not in groups for handle, _ in source.cached.participants(row, groups)):
        return "unmapped_participant"
    if source._eligible(row, groups, plan, blocked) != "validation_development":
        return "outside_frozen_validation_population"
    if not any(source.kgs_source.player(row[color])[1] == "11k" for color in ("black", "white")):
        return "no_11k_side"
    return None


def history_barrier(indexes, exposures, owner, sealed):
    blocked = set(sealed)
    hashes = {field: set() for field in FINGERPRINTS}
    for index in indexes:
        blocked.update(index.get("processed_source_ids", []))
        for name in ("records", "rejected", "study_rejections"):
            for row in index.get(name, []):
                blocked.add(row["source_url"])
                for field in FINGERPRINTS:
                    if row.get(field):
                        hashes[field].add(row[field])
    for exposure in exposures:
        blocked.update(exposure["source_ids"])
        hashes["sgf_sha256"].update(exposure["sgf_hashes"])
        hashes["audit_sequence_sha256"].update(exposure["sequence_hashes"])
    for row in owner["excluded_games"]:
        blocked.add(row["source_url"])
        hashes["sgf_sha256"].add(row["sgf_sha256"])
    return blocked, hashes


def context(request_path, pin, *, fixture=False, check=lambda: None):
    request = read({"path": str(Path(request_path).resolve()), "sha256": pin}, 4 * MIB)
    if request["schema"] != SCHEMA or request["fixture_only"] is not fixture:
        raise ValueError("Wrong request scope")
    if set(request["limits"]) != set(LIMITS) or any(
            type(value) is not int or not 0 < value <= LIMITS[key]
            for key, value in request["limits"].items()):
        raise ValueError("Invalid or enlarged phase limits")
    network = source.Limits(**request["network_limits"])
    if (network.requests > 80 or network.response_bytes > 128 * MIB
            or network.wall_seconds > request["limits"]["wall_seconds"]
            or network.cpu_seconds > request["limits"]["cpu_seconds"]):
        raise ValueError("Network limits exceed the bounded phase")
    if not fixture:
        if request["source_commit"] != COMMIT or request["deadline_utc"] != DEADLINE:
            raise ValueError("The source commit or original deadline changed")
        if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip() != COMMIT:
            raise ValueError("The tracked source revision changed")
        if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT).strip():
            raise ValueError("The source tree changed")
        for path, expected in request["code_sha256"].items():
            check()
            if source.file_hash(ROOT / path) != expected:
                raise ValueError("A reviewed source file changed")
        if request["code_sha256"].get(str(Path(__file__).resolve().relative_to(ROOT))) != source.file_hash(__file__):
            raise ValueError("The helper is not pinned")
    refs = request["inputs"]
    check()
    parent, grown = read(refs["parent_summary"]), read(refs["grown_summary"])
    if (parent["schema"] != "e2_komi_expansion_v1" or grown["schema"] != "e2_data_growth_v1"
            or grown["parent_summary"]["sha256"] != refs["parent_summary"]["sha256"]):
        raise ValueError("The two preserved corpora are unrelated")
    objects = []
    for summary in (parent, grown):
        check()
        artifacts = summary["artifacts"]
        item = {name: read(artifacts[name]) for name in ("groups", "split_manifest", "base_index", "exposure")}
        plan, index = item["split_manifest"]["plan"], item["base_index"]
        SplitPlan.from_dict(plan)
        if (plan["grouping_sha256"] != kgs_eligibility.grouping_digest(item["groups"])
                or plan["train_before"] != "2025-09-01" or plan["confirmation_from"] != "2025-11-01"
                or index["canonical_sha256"] != artifacts["canonical"]["sha256"]
                or index["split_sha256"] != artifacts["split_manifest"]["sha256"]
                or index["exposure"] != artifacts["exposure"]
                or digest(plan) != item["split_manifest"]["plan_sha256"]
                or item["split_manifest"]["fixture_only"] is not fixture
                or digest(item["exposure"]) != item["split_manifest"]["exposure_sha256"]):
            raise ValueError("Corpus cohort, index or exposure pins disagree")
        objects.append(item)
    a, b = objects
    plan = a["split_manifest"]["plan"]
    if (any(b["groups"].get(h) != g for h, g in a["groups"].items())
            or any(b["split_manifest"]["plan"][k] != plan[k] for k in plan if k != "grouping_sha256")):
        raise ValueError("The expanded corpus changed a frozen cohort")
    owner, policy = read(refs["owner_exclusions"]), read(refs["source_policy"])
    if (policy["policy"] != kgs_admission.POLICY or not fixture
            and refs["source_policy"]["sha256"] != kgs_admission.DECISION_SHA256):
        raise ValueError("The owner admission policy changed")
    if any(item["split_manifest"]["provenance"]["manual_review_sha256"] != refs["owner_exclusions"]["sha256"]
           for item in objects):
        raise ValueError("The owner exclusions changed")
    frozen = read(refs["queries"])
    expected = queries() if not fixture else frozen["queries"]
    if (frozen["queries"] != expected or frozen["queries_sha256"] != source.digest(expected)
            or not 1 <= len(expected) <= 12 or len({q["key"] for q in expected}) != len(expected)):
        raise ValueError("The frozen query list changed")
    prior_pages, hints = set(), set()
    for reference in frozen["historical_metadata"]:
        check()
        frame = read(reference)
        prior_pages.update(p["key"].lower() for p in frame.get("pages", []))
        for row in frame["rows"]:
            for color in ("black", "white"):
                player = source.kgs_source.player(row[color])
                if player and player[1] == "11k":
                    hints.add(player[0].lower())
    _, final, bots, cached = rank_discovery._context(a["groups"], plan, frozen["old_cache"])
    if frozen.get("old_cache_account_months_sha256", source.digest(sorted(cached)) if fixture else None) != source.digest(sorted(cached)):
        raise ValueError("The old cache inventory changed")
    for query in expected:
        handle, month = query["user"].lower(), query["month"]
        if (a["groups"].get(handle) not in plan["validation_groups"]
                or a["groups"].get(handle) in final | bots or handle not in hints or month not in (9, 10)
                or query != rank_discovery._query(handle, month, "validation_development", "11k")
                or query["key"].lower() in prior_pages or (handle, month) in cached):
            raise ValueError("A query is not a fresh allowed validation account-month")
    exposures = [item["exposure"] for item in objects]
    sealed = set().union(*(set(item["split_manifest"]["sealed_source_ids"]) for item in objects))
    blocked, hashes = history_barrier([item["base_index"] for item in objects], exposures, owner, sealed)
    exposure = data_expansion.union_exposure(*exposures)
    if exposure["groups"] & set(plan["test_groups"]):
        raise ValueError("A test group entered prior exposure")
    return {"request": request, "network": network, "queries": expected, "groups": dict(b["groups"]),
            "plan": plan, "blocked": blocked, "hashes": hashes, "exposure": exposure,
            "barrier_counts": {"source_ids": len(blocked), **{k: len(v) for k, v in hashes.items()}}}


class Budget:
    def __init__(self, limits, deadline, *, now=source.utc_now, clock=time.monotonic, cpu=time.process_time):
        self.limits, self.deadline = limits, source.deadline(deadline)
        self.now, self.clock, self.cpu = now, clock, cpu
        self.started, self.cpu_start = clock(), cpu()
        self.bytes = self.selected_bytes = self.expanded_bytes = 0

    def check(self):
        if self.now() >= self.deadline:
            raise source.SourceStop("absolute_deadline")
        if self.clock() - self.started >= self.limits["wall_seconds"]:
            raise source.SourceStop("phase_wall_limit")
        if self.cpu() - self.cpu_start >= self.limits["cpu_seconds"]:
            raise source.SourceStop("phase_cpu_limit")

    def save(self, path, value, *, final=False):
        body = value if isinstance(value, bytes) else source.encoded(value) + b"\n"
        if self.bytes + len(body) > self.limits["output_bytes"] - (0 if final else FINAL_RESERVE):
            raise source.SourceStop("phase_output_limit")
        with Path(path).open("xb") as stream:
            stream.write(body)
        self.bytes += len(body)
        return {"path": str(Path(path).resolve()), "sha256": sha256(body).hexdigest()}

    def costs(self):
        return {"cpu_seconds": self.cpu() - self.cpu_start, "wall_seconds": self.clock() - self.started,
                "output_bytes": self.bytes, "selected_sgf_bytes": self.selected_bytes,
                "expanded_zip_bytes": self.expanded_bytes, "gpu_seconds": 0, "workers": 1}


class EvidenceTransport:
    """Retain response evidence while the original Client owns cookies and retries."""

    def __init__(self, transport, output, budget, allowed_queries):
        self.transport, self.output, self.budget = transport, Path(output), budget
        self.query_parameters = [parse_qs(urlsplit(q["url"]).query) for q in allowed_queries]
        self.bundles, self.secrets = set(), set()
        self.attempts = self.response_bytes = 0

    def redact(self, value):
        for secret in sorted(self.secrets, key=len, reverse=True):
            for form in (secret, html.escape(secret), quote(secret, safe=""), quote_plus(secret),
                         json.dumps(secret)[1:-1]):
                value = value.replace(form, "[redacted]")
        return re.sub(r"You are currently logged in as .*?\.",
                      "You are currently logged in as [redacted].", value, flags=re.S)

    def __call__(self, url, timeout, maximum, **kwargs):
        self.budget.check()
        parsed = urlsplit(url)
        source.allowed_url(url)
        if parsed.path == "/login.jsp":
            if url != "https://www.gokgs.com/login.jsp":
                raise ValueError("Unexpected login URL")
        elif parsed.path in ("/gameArchives.jsp", "/archives.jsp"):
            if parse_qs(parsed.query) not in self.query_parameters:
                raise ValueError("Redirect changes the frozen account-month")
        elif url not in self.bundles:
            raise ValueError("ZIP request has no authorized candidate")
        if kwargs.get("data"):
            self.secrets.update(value for values in parse_qs(kwargs["data"].decode()).values() for value in values)
        self.attempts += 1
        item = {"attempt": self.attempts, "url": url, "maximum_bytes": maximum}
        status, headers, body = None, {}, b""
        try:
            status, headers, body = self.transport(url, timeout, maximum, **kwargs)
            return status, headers, body
        except source.InterruptedResponse as error:
            status, headers, body = error.status, error.headers, error.body
            item["error"] = type(error).__name__
            raise source.InterruptedResponse(status, headers, body, type(error).__name__) from None
        except Exception as error:
            item["error"] = type(error).__name__
            raise
        finally:
            self.response_bytes += len(body)
            keep = {"content-type", "content-length", "date", "retry-after", "last-modified", "etag"}
            item.update(status=status, bytes=len(body), sha256=sha256(body).hexdigest(),
                        headers={k.lower(): self.redact(str(v))[:1024]
                                 for k, v in headers.items() if k.lower() in keep})
            if parsed.path == "/login.jsp" or kwargs.get("data"):
                item["body_omitted"] = "authentication_response_may_echo_credentials"
            elif url in self.bundles and body.startswith(b"PK"):
                item["body_omitted"] = "archive_retained_separately"
            elif body:
                retained = self.redact(body.decode("utf-8", errors="replace")).encode()
                item["body"] = self.budget.save(self.output / f"response-{self.attempts:03d}.bin", retained)
                item["body_redacted"] = retained != body
            self.budget.save(self.output / f"response-{self.attempts:03d}.json", item)


class RedactedLog:
    def __init__(self, stream, redact):
        self.stream, self.redact = stream, redact

    def write(self, value):
        return self.stream.write(self.redact(value))

    def flush(self):
        self.stream.flush()

    def close(self):
        self.stream.close()


def candidate_record(data, row, source_ref, ctx, budget):
    pin = sha256(data).hexdigest()
    basic = {"source_url": row["url"], "sgf_sha256": pin}
    if pin in ctx["hashes"]["sgf_sha256"]:
        return {**basic, "status": "excluded", "reason": "historical_sgf_hash"}
    try:
        data_expansion._header_for_fingerprints(data, row, ctx["groups"], set(ctx["plan"]["test_groups"]))
    except (ValueError, TypeError, KeyError, UnicodeError, LookupError):
        return {**basic, "status": "excluded", "reason": "source_header_rejected"}
    budget.check()
    audit = inspect_sgf(data, row, kgs_eligibility.HISTORICAL_BOT_HANDLES)
    audit["source_url"] = row["url"]
    basic["audit_sequence_sha256"] = audit.get("sequence_fingerprint")
    if basic["audit_sequence_sha256"] in ctx["hashes"]["audit_sequence_sha256"]:
        return {**basic, "status": "excluded", "reason": "historical_sequence_hash"}
    record, adjudication_cost = canonicalize(data, {**source_ref, "sgf_sha256": pin}, row,
                                            check_budget=budget.check)
    basic["trajectory_sha256"] = record["trajectory_sha256"]
    if basic["trajectory_sha256"] in ctx["hashes"]["trajectory_sha256"]:
        return {**basic, "status": "excluded", "reason": "historical_trajectory_hash"}
    exposure = kgs_eligibility.Exposure(kgs_eligibility.grouping_digest(ctx["groups"]), **{
        name: frozenset(values) for name, values in ctx["exposure"].items()})
    record.update(eligibility=kgs_admission.classify(audit, row, ctx["groups"], exposure), sealed_final=False)
    entry = split_entry(record, exposure.grouping_sha256)
    return {**basic, "status": "canonical", "record": record, "entry": entry,
            "adjudication_cost": adjudication_cost}


def finalize_candidates(units, ctx):
    canonical = [u for u in units if u["status"] == "canonical"]
    plan = {**ctx["plan"], "grouping_sha256": kgs_eligibility.grouping_digest(ctx["groups"])}
    # A rejected fresh copy still invalidates a matching candidate. No survivor is chosen by outcome.
    poisoned = {f: {u[f] for u in units if u["status"] == "excluded" and u.get(f)} for f in FINGERPRINTS}
    entries = []
    for unit in canonical:
        entry = dict(unit["entry"])
        if any(unit.get(f) in poisoned[f] for f in FINGERPRINTS):
            entry["exclusions"] = [*entry["exclusions"], "fresh_rejected_duplicate"]
        record = unit["record"]
        if (record.get("recorded_result", {}).get("kind") not in {"scored", "resignation", "timeout", "forfeit"}
                or record["replay_pending_cap"] is not None):
            entry["exclusions"] = [*entry["exclusions"], "source_completion_unavailable"]
        entries.append(entry)
    memberships = build_splits(entries, SplitPlan.from_dict(plan),
                              exposed_groups=ctx["exposure"]["groups"], group_registry=ctx["groups"])
    by_id = {row["source_url"]: row for row in memberships}
    accepted, rejected = [], []
    for unit in units:
        if unit["status"] != "canonical":
            rejected.append(unit)
            continue
        row = by_id[unit["source_url"]]
        record = unit["record"]
        reason = None
        if row["role"] != "validation_development" or row["exclusions"] or not record["replay_validated"]:
            reason = "admission_replay_or_split"
        if (record["metadata"].get("KM") != 6.5 or record["metadata"].get("RU") != "Japanese"
                or not any(record["metadata"].get(r) == "11k" for r in ("BR", "WR"))):
            reason = "outside_11k_japanese_6_5"
        details = {k: v for k, v in unit.items() if k not in ("record", "entry")}
        if reason:
            rejected.append({**details, "status": "excluded", "reason": reason, "split_exclusions": row["exclusions"]})
        else:
            accepted.append({**unit, "membership": row})
    return accepted, rejected


def run(request_path, pin, output, *, credentials_file=None, fixture=False, transport=None,
        now=source.utc_now, monotonic=time.monotonic, cpu=time.process_time, sleep=time.sleep):
    initial = read({"path": str(Path(request_path).resolve()), "sha256": pin}, 4 * MIB)
    budget = Budget(initial["limits"], initial["deadline_utc"], now=now, clock=monotonic, cpu=cpu)
    output = Path(output).resolve()
    if output.exists() and any(p.name not in {"job.json", "stdout.log", "stderr.log"} for p in output.iterdir()):
        raise FileExistsError("The worker output was already used")
    output.mkdir(parents=True, exist_ok=True)
    execution = Path(request_path).resolve().with_name("source-execution.json")
    budget.save(execution, {"request_sha256": pin, "output": str(output), "started_utc": now().isoformat()})
    work = output / "source-work"
    work.mkdir()
    client, ctx, evidence = None, None, None
    pages, candidates, units, rejected, accepted = [], [], [], [], []
    status, reason = "failed", "setup_incomplete"
    yield_status, selection_stop = "insufficient_yield", None
    published_complete = False
    try:
        if transport is not None and not fixture:
            raise ValueError("Replacement transports are fixture-only")
        ctx = context(request_path, pin, fixture=fixture, check=budget.check)
        budget.check()
        client = source.Client(work, ctx["network"], initial["deadline_utc"], transport=transport,
                               now=now, monotonic=monotonic, cpu=cpu, sleep=sleep)
        evidence = EvidenceTransport(client.transport, work, budget, ctx["queries"])
        client.transport = evidence
        client.log = RedactedLog(client.log, evidence.redact)
        first = client.authenticate(credentials_file, ctx["queries"][0]) if credentials_file else None
        all_rows = []
        for number, query in enumerate(ctx["queries"]):
            budget.check()
            ctx["exposure"]["account_months"].add((query["user"].lower(), f"2025-{query['month']:02d}"))
            ctx["exposure"]["groups"].add(ctx["groups"][query["user"].lower()])
            try:
                body = first if number == 0 and first is not None else client.get(query["url"], ctx["network"].html_bytes)
                rows = source.parse_page(body, query)
                pages.append({**query, "status": "retrieved", "rows": len(rows)})
                all_rows.extend(rows)
            except (source.RequestFailure, ValueError, UnicodeError) as error:
                pages.append({**query, "status": "failed", "reason": error_reason(error)})
        unique = source._unique(all_rows)
        metadata_rejections = Counter()
        for row in sorted(unique.values(), key=lambda r: r["url"]):
            why = metadata_reason(row, ctx["groups"], ctx["plan"], ctx["blocked"])
            if why:
                metadata_rejections[why] += 1
            else:
                candidates.append(row)
        budget.save(work / "discovery.json", {"pages": pages, "rows": all_rows,
                    "metadata_rejections": dict(metadata_rejections), "eligible_candidates": len(candidates)})
        eligible_count = len(candidates)
        candidates = candidates[:initial["limits"]["max_candidates"]]
        for row in candidates:
            for handle, group in source.cached.participants(row, ctx["groups"]):
                ctx["exposure"]["groups"].add(group)
            ctx["exposure"]["source_ids"].add(row["url"])
        budget.save(work / "frozen-candidates.json", {"candidates": candidates,
            "ordering": "lexicographic source URL before SGF reads", "eligible_before_cap": eligible_count,
            "groups": ctx["groups"], "original_plan": ctx["plan"]})
        bundles = {}
        for row in candidates:
            budget.check()
            key, url = row["page_key"], row["bundle_url"]
            if key not in bundles:
                source.allowed_url(url, bundle=True)
                evidence.bundles.add(url)
                try:
                    body = client.get(url, ctx["network"].bundle_bytes, bundle=True)
                    bundle = budget.save(work / (key + ".zip"), body)
                    with zipfile.ZipFile(bundle["path"]) as archive:
                        expanded = sum(i.file_size for i in archive.infolist())
                        if budget.expanded_bytes + expanded > initial["limits"]["expanded_zip_bytes"]:
                            raise source.SourceStop("expanded_zip_limit")
                        budget.expanded_bytes += expanded
                    bundles[key] = bundle
                except (source.RequestFailure, zipfile.BadZipFile, ValueError) as error:
                    bundles[key] = None
                    rejected.append({"page_key": key, "reason": error_reason(error), "status": "bundle_failed"})
            bundle = bundles[key]
            if bundle is None:
                rejected.append({"source_url": row["url"], "reason": "bundle_unavailable", "status": "unattempted"})
                continue
            with zipfile.ZipFile(bundle["path"]) as archive:
                matches = [i for i in archive.infolist() if source.e1_audit.member_key(i.filename) == source.e1_audit.source_key(row["url"])]
                if len(matches) != 1 or matches[0].file_size > MIB:
                    rejected.append({"source_url": row["url"], "reason": "missing_ambiguous_or_large_member", "status": "excluded"})
                    continue
                info = matches[0]
                if budget.selected_bytes + info.file_size > initial["limits"]["selected_sgf_bytes"]:
                    raise source.SourceStop("selected_sgf_limit")
                budget.selected_bytes += info.file_size
                try:
                    with archive.open(info) as stream:
                        data = stream.read(MIB + 1)
                except (zipfile.BadZipFile, RuntimeError, EOFError) as error:
                    rejected.append({"source_url": row["url"], "reason": error_reason(error), "status": "excluded"})
                    continue
                if len(data) != info.file_size or len(data) > MIB:
                    raise ValueError("Selected member size mismatch")
            source_ref = {"source_url": row["url"], "archive_path": bundle["path"],
                          "archive_sha256": bundle["sha256"], "zip_member": info.filename}
            unit = candidate_record(data, row, source_ref, ctx, budget)
            units.append(unit)
            if unit.get("sgf_sha256"):
                ctx["exposure"]["sgf_hashes"].add(unit["sgf_sha256"])
            if unit.get("audit_sequence_sha256"):
                ctx["exposure"]["sequence_hashes"].add(unit["audit_sequence_sha256"])
            budget.save(work / f"unit-{len(units):04d}.json", unit)
            budget.check()
            provisional, _ = finalize_candidates(units, ctx)
            if len(provisional) >= initial["limits"]["target_games"]:
                selection_stop = "target_games"
                break
        accepted, exclusions = finalize_candidates(units, ctx)
        rejected.extend(exclusions)
        published = []
        for unit in accepted:
            canonical = budget.save(work / ("canonical-" + unit["sgf_sha256"] + ".json"), unit["record"])
            published.append({**{k: v for k, v in unit.items() if k not in ("record", "entry")},
                              "canonical": canonical})
        accepted = published
        budget.save(work / "accepted.json", {"games": accepted})
        budget.save(work / "exposure.json", {k: sorted(v) for k, v in ctx["exposure"].items()})
        published_complete = True
        status = "completed"
        yield_status = "sufficient_yield" if len(accepted) >= 2 else "insufficient_yield"
        reason = None if yield_status == "sufficient_yield" else "fewer_than_two_fresh_games"
        if any(page["status"] == "failed" for page in pages):
            status, reason = "partial", "metadata_requests_failed"
        elif any(item["status"] == "bundle_failed" for item in rejected):
            status, reason = "partial", "bundle_unavailable"
    except Exception as error:
        status, reason = ("partial" if isinstance(error, source.SourceStop) else "failed"), error_reason(error)
    finally:
        if client:
            client.log.close()
            budget.bytes += (work / "requests.jsonl").stat().st_size
        network_costs = client.costs() if client else {"requests": 0, "response_bytes": 0}
        if evidence:
            network_costs["response_bytes"] = max(network_costs["response_bytes"], evidence.response_bytes)
        if not published_complete:
            accepted, yield_status = [], "insufficient_yield"
        summary = {"schema": SCHEMA, "request": {"path": str(Path(request_path).resolve()), "sha256": pin},
            "status": status, "yield_status": yield_status, "reason": reason, "stop_reason": reason,
            "pages": pages, "candidates": len(candidates), "selection_stop": selection_stop or "candidate_list_exhausted",
            "candidates_processed": len(units), "accepted_games": len(accepted), "usable_games": len(accepted),
            "accepted": accepted,
            "rejected": rejected, "duplicate_barrier_counts": ctx["barrier_counts"] if ctx else None,
            "costs": {**budget.costs(), "http_requests": network_costs["requests"], "response_bytes": network_costs["response_bytes"],
                      "scope": "Helper entry through finalization; final summary writes excluded. Runner owns cumulative accounting."},
            "training_performed": False, "model_queries": 0, "protected_targets_decoded": False,
            "integration_performed": False, "original_deadline_utc": initial["deadline_utc"],
            "open_checks": ["Independent review and separate integration into stopping-only samples", "Original-budget cumulative accounting by the phase runner"]}
        summary_ref = budget.save(work / "summary.json", summary, final=True)
        completion = {k: summary[k] for k in ("schema", "status", "yield_status", "reason", "stop_reason",
                                               "accepted_games", "usable_games", "costs")}
        completion["source_summary"] = summary_ref
        budget.save(output / "summary.json", completion, final=True)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("check", "run"))
    parser.add_argument("request", type=Path)
    parser.add_argument("sha256")
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--credentials-file", type=Path)
    args = parser.parse_args()
    if args.operation == "check":
        if args.credentials_file or args.output:
            parser.error("Offline checking accepts only a request and pin")
        ctx = context(args.request, args.sha256)
        print(json.dumps({"status": "offline_pins_and_queries_verified", "queries": len(ctx["queries"]),
                          "duplicate_barrier_counts": ctx["barrier_counts"]}, sort_keys=True))
    else:
        if not args.output:
            parser.error("run requires a fresh worker output")
        result = run(args.request, args.sha256, args.output, credentials_file=args.credentials_file)
        print(json.dumps({k: result[k] for k in ("status", "stop_reason", "accepted_games", "costs")}, sort_keys=True))
        if result["status"] != "completed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
