"""Freeze a larger KGS archive frame before retrieving any new game records."""

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from hashlib import sha256
import http.client
import http.cookiejar
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

from kenggo import cached_coverage_audit as cached
from kenggo import e1_audit, kgs_admission, kgs_eligibility, kgs_source
from kenggo import rank_discovery


MIB = 1024**2
REQUEST_SCHEMA = "e2_data_collection_request_v1"
QUERY_SCHEMA = "e2_expansion_queries_v1"
SOURCES_SCHEMA = "e2_expansion_sources_v1"
SEED = "kenggo-ydo.8-data-expansion-20261114-v1"
EXPOSURE_FIELDS = ("groups", "source_ids", "sgf_hashes", "sequence_hashes", "account_months")
CODE_PATHS = {"src/kenggo/expansion_source.py", "src/kenggo/kgs_source.py",
              "src/kenggo/e1_audit.py", "src/kenggo/cached_coverage_audit.py",
              "src/kenggo/kgs_eligibility.py"}


@dataclass(frozen=True)
class Limits:
    requests: int = 900
    response_bytes: int = 512 * MIB
    html_bytes: int = 2 * MIB
    bundle_bytes: int = 8 * MIB
    expanded_zip_bytes: int = 4096 * MIB
    cpu_seconds: int = 900
    wall_seconds: int = 21600
    min_interval_seconds: int = 5
    retries: int = 2
    max_train_candidates: int = 25000
    max_population_development_candidates: int = 3000
    max_validation_development_candidates: int = 1500

    def __post_init__(self):
        maxima = dict(requests=2400, response_bytes=1024 * MIB, html_bytes=2 * MIB,
            bundle_bytes=8 * MIB, expanded_zip_bytes=4096 * MIB, cpu_seconds=900,
            wall_seconds=21600, retries=2, max_train_candidates=25000,
            max_population_development_candidates=5000, max_validation_development_candidates=5000)
        for key, maximum in maxima.items():
            value = getattr(self, key)
            if type(value) is not int or value < (0 if key == "retries" else 1) or value > maximum:
                raise ValueError("Invalid or enlarged source limit: " + key)
        if type(self.min_interval_seconds) is not int or not 5 <= self.min_interval_seconds <= 60:
            raise ValueError("Use at least five seconds between request starts")


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return sha256(encoded(value)).hexdigest()


def file_hash(path):
    value = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(MIB), b""):
            value.update(block)
    return value.hexdigest()


def reference(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_hash(path)}


def read_ref(ref, maximum=128 * MIB):
    if (not isinstance(ref, dict) or set(ref) != {"path", "sha256"}
            or not isinstance(ref["path"], str) or not re.fullmatch("[0-9a-f]{64}", ref["sha256"])):
        raise ValueError("An artifact needs an exact path and SHA-256")
    path = Path(ref["path"]).resolve()
    if not path.is_file() or path.stat().st_size > maximum or file_hash(path) != ref["sha256"]:
        raise ValueError("Source input changed or exceeds its limit: " + str(path))
    return json.loads(path.read_bytes())


def write_json(path, value):
    return cached.save_json(Path(path), value)


def utc_now():
    return datetime.now(timezone.utc)


def deadline(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("The collection deadline must include a timezone")
    return result


def allowed_url(url, *, bundle=False):
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != "https" or parsed.netloc not in {"www.gokgs.com", "files.gokgs.com"}
            or parsed.username or parsed.password or parsed.fragment):
        raise ValueError("Request outside the official KGS HTTPS origins")
    if bundle and (parsed.netloc != "www.gokgs.com" or not parsed.path.startswith("/servlet/archives/")
                   or not parsed.path.endswith(".zip") or parsed.query):
        raise ValueError("Expected an official monthly KGS ZIP URL")


def _owned(cache, name):
    path = (cache / name).resolve()
    if path.parent != cache:
        raise ValueError("Cache path leaves its directory")
    return path


def _context(request_ref, fixture_only=False):
    request = read_ref(request_ref, 16 * MIB)
    if request.get("schema") != REQUEST_SCHEMA or not isinstance(request.get("scope_decision"), str):
        raise ValueError("A separate expansion scope decision is required")
    if not request["scope_decision"].strip():
        raise ValueError("The expansion decision is empty")
    limits = Limits(**request["limits"])
    deadline(request["deadline_utc"])
    if "access_probe" in request:
        probe = read_ref(request["access_probe"], MIB)
        if (set(probe) != {"requests", "response_bytes", "cpu_charged_seconds", "records"}
                or any(type(probe[key]) is not int or probe[key] < 0
                       for key in ("requests", "response_bytes", "cpu_charged_seconds"))
                or probe["requests"] != len(probe["records"])
                or probe["response_bytes"] != sum(row["bytes"] for row in probe["records"])
                or probe["requests"] + limits.requests > 2400
                or probe["response_bytes"] + limits.response_bytes > 1024 * MIB
                or probe["cpu_charged_seconds"] + limits.cpu_seconds > 900):
            raise ValueError("The access probe and collection must share the frozen limits")
    refs = request["refs"]
    names = {"frame", "original_sample", "discovery_config", "split_config", "owner_exclusions",
             "source_policy", "old_coverage_frozen", "prior_exposure"}
    continuation = {"current_base", "base_index"}
    if set(refs) not in (names, names | continuation):
        raise ValueError("Unexpected expansion input references")
    if (set(refs) == names | continuation) != ("discovery_policy" in request):
        raise ValueError("Expanded account discovery requires the preserved current dataset")
    if not fixture_only:
        for name in names & cached.PRODUCTION_INPUTS.keys():
            if refs[name]["sha256"] != cached.PRODUCTION_INPUTS[name]:
                raise ValueError("The historical source contract changed: " + name)
        code = request["source_revision"]["code_sha256"]
        required_code = CODE_PATHS | ({"src/kenggo/rank_discovery.py"} if "discovery_policy" in request else set())
        if not required_code <= code.keys() or any(file_hash(path) != pin for path, pin in code.items()):
            raise ValueError("The expansion source code differs from its frozen revision")
    objects = {name: read_ref(ref) for name, ref in refs.items()}
    groups = request["base_groups"]
    split = objects["split_config"]
    if (kgs_eligibility.grouping_digest(groups) != split["grouping_sha256"]
            or split["train_before"] != "2025-09-01" or split["confirmation_from"] != "2025-11-01"
            or set(split["validation_groups"]) & set(split["test_groups"])):
        raise ValueError("Preserve the original account groups and split dates")
    if not set(split["validation_groups"] + split["test_groups"]) <= set(groups.values()):
        raise ValueError("A reserved group is absent from the fixed registry")
    for handle, group in groups.items():
        if kgs_eligibility.normalize_handle(handle) != handle or not group.startswith("kgs-handle:"):
            raise ValueError("Invalid base handle group")
        kgs_eligibility.normalize_handle(group.removeprefix("kgs-handle:"))
    if objects["source_policy"].get("policy") != kgs_admission.POLICY:
        raise ValueError("Preserve the approved source screening policy")
    old = objects["old_coverage_frozen"]
    for handle, group in groups.items():
        if old["groups"].get(handle) != group:
            raise ValueError("Cached coverage changed an existing group")
    prior = objects["prior_exposure"]
    if set(prior) != set(EXPOSURE_FIELDS) or any(not isinstance(prior[key], list) for key in EXPOSURE_FIELDS):
        raise ValueError("Provide the complete prior exposure union")
    old_ids = {item["archive_row"]["url"] for item in old["candidates"]}
    original_ids = {row["url"] for row in objects["original_sample"]["candidates"]}
    if not old_ids | original_ids <= set(prior["source_ids"]):
        raise ValueError("The exposure union omits previously inspected source IDs")
    prior_groups = {group for item in old["candidates"]
                    for _, group in cached.participants(item["archive_row"], old["groups"])}
    prior_groups.update(group for row in objects["original_sample"]["candidates"]
                        for _, group in cached.participants(row, groups))
    if not prior_groups <= set(prior["groups"]):
        raise ValueError("The exposure union omits previously inspected account groups")
    cache = Path(request["cache_directory"]).resolve()
    if not cache.is_dir():
        raise ValueError("Original cache is unavailable")
    expected = cached.select_rows(objects["frame"], original_ids, groups, split, cache)
    if expected != old["candidates"]:
        raise ValueError("The old cached candidate list changed")
    if not fixture_only and (len(expected) != 4269 or digest(sorted(old_ids)) != cached.EXPECTED_IDS):
        raise ValueError("Wrong historical coverage candidate list")
    if "current_base" in objects:
        base, index = objects["current_base"], objects["base_index"]
        if base.get("schema") != "e2_data_expansion_v1" or index.get("schema") != "e2_data_index_v1":
            raise ValueError("Unsupported prior dataset or data index")
        artifacts = base["artifacts"]
        counts = read_ref(request["discovery_policy"]["baseline_counts_ref"])
        if counts.get("provenance") != {"current_base": refs["current_base"],
                "examples_manifest": artifacts["examples_manifest"],
                "split_manifest": artifacts["split_manifest"]}:
            raise ValueError("Rank sampling counts belong to another base dataset")
        current_groups = read_ref(artifacts["groups"])
        current_plan = read_ref(artifacts["split_manifest"])["plan"]
        if (any(current_groups.get(key) != value for key, value in groups.items())
                or any(current_plan[key] != split[key] for key in split if key != "grouping_sha256")
                or current_plan["grouping_sha256"] != kgs_eligibility.grouping_digest(current_groups)
                or index["canonical_sha256"] != artifacts["canonical"]["sha256"]
                or index["split_sha256"] != artifacts["split_manifest"]["sha256"]
                or refs["prior_exposure"] != artifacts["exposure"]):
            raise ValueError("The continuation changed its base dataset, groups, splits or exposure")
        objects["current_groups"] = current_groups
        objects["completed_ids"] = {row["source_url"] for row in index["records"] + index["rejected"]}
    return request, objects, limits, cache


def _eligible(row, groups, split, excluded):
    if row["url"] in excluded or kgs_source.candidate_reasons(row):
        return None
    sides = cached.participants(row, groups)
    bots = {groups.get(handle, "kgs-handle:" + handle) for handle in kgs_eligibility.HISTORICAL_BOT_HANDLES}
    if any("bot" in handle or group in bots for handle, group in sides):
        return None
    return cached.provisional_role(row, groups, split)


def _unique(rows):
    result = {}
    for row in rows:
        previous = result.setdefault(row["url"], row)
        if any(previous[key] != row[key] for key in ("black", "white", "setup", "type", "result")):
            raise ValueError("Conflicting archive metadata for one source ID")
    return result


def _query_contents(request_ref, fixture_only=False):
    request, objects, limits, cache = _context(request_ref, fixture_only)
    groups, split = objects.get("current_groups", request["base_groups"]), objects["split_config"]
    discovery = objects["discovery_config"]
    if discovery.get("seed") != kgs_source.SEED:
        raise ValueError("Use the original discovery accounts")
    queries = []
    for account in discovery["accounts"]:
        handle = kgs_eligibility.normalize_handle(account["user"])
        group = groups.get(handle, "kgs-handle:" + handle)
        if (account["user"] in discovery.get("excluded_discovery_accounts", {}) or "bot" in handle
                or handle in kgs_eligibility.HISTORICAL_BOT_HANDLES or group in split["test_groups"]):
            continue
        months = range(9 if group in split["validation_groups"] else 1, 11)
        for month in months:
            key = f"{account['user']}-2025-{month:02d}"
            if _owned(cache, key + ".html").is_file():
                continue
            query = urllib.parse.urlencode(dict(user=account["user"], year=2025, month=month, oldAccounts="y"))
            queries.append({"key": key, "user": account["user"], "month": month,
                            "url": "https://www.gokgs.com/gameArchives.jsp?" + query})
    queries.sort(key=lambda item: (item["month"], item["user"].lower()))
    if "discovery_policy" in request:
        queries = rank_discovery.queries(objects["frame"], groups, split, cache, request["discovery_policy"])
    if (len({item["key"].lower() for item in queries}) != len(queries)
            or not fixture_only and "discovery_policy" not in request and len(queries) != 286):
        raise ValueError("Unexpected or duplicate archive queries")
    original_ids = {row["url"] for row in objects["original_sample"]["candidates"]}
    excluded = original_ids | {row["source_url"] for row in objects["owner_exclusions"]["excluded_games"]}
    excluded.update(objects.get("completed_ids", ()))
    rows = _unique(objects["frame"]["rows"])
    old_candidates = [{"archive_row": row, "provisional_role": role} for row in rows.values()
                      if (role := _eligible(row, groups, split, excluded)) is not None]
    old_candidates.sort(key=lambda item: item["archive_row"]["url"])
    successful = cached._successful_requests(objects["old_coverage_frozen"]["request_ledgers"])
    pages, bundles, page_rows = {}, {}, {}
    for item in old_candidates:
        row, key = item["archive_row"], item["archive_row"]["page_key"]
        if key in pages:
            continue
        metadata_path, html_path = _owned(cache, key + ".json"), _owned(cache, key + ".html")
        metadata = read_ref(reference(metadata_path), 4 * MIB)
        if html_path.stat().st_size > limits.html_bytes or file_hash(html_path) != metadata["public_html_sha256"]:
            raise ValueError("Cached public archive HTML changed")
        body = html_path.read_text()
        page = kgs_source.ArchivePage(body)
        if page.rows != metadata["rows"] or row["bundle_url"] != metadata["bundle_url"]:
            raise ValueError("Cached archive metadata disagrees with its HTML")
        if not any(url == metadata["url"] and pin == metadata["body_sha256"] for url, pin, _ in successful):
            raise ValueError("Cached page has no original successful request")
        pages[key] = {"metadata": reference(metadata_path), "html": reference(html_path)}
        page_rows[key] = {digest(row) for row in page.rows}
        path = _owned(cache, key + ".zip")
        if path.is_file():
            pin, size = file_hash(path), path.stat().st_size
            if size > limits.bundle_bytes or (row["bundle_url"], pin, size) not in successful:
                raise ValueError("Cached bundle has no original successful request")
            bundles[key] = {"path": str(path), "sha256": pin, "bytes": size, "url": row["bundle_url"]}
    for item in old_candidates:
        row = item["archive_row"]
        plain = {key: value for key, value in row.items() if key not in ("page_key", "bundle_url")}
        if digest(plain) not in page_rows[row["page_key"]]:
            raise ValueError("Frozen frame row is absent from its archive page")
    return {"schema": QUERY_SCHEMA, "fixture_only": fixture_only, "request": request_ref,
            "queries": queries, "queries_sha256": digest(queries), "old_candidates": old_candidates,
            "old_pages": pages, "old_bundles": bundles, "limits": asdict(limits)}


def freeze_queries(request_path, request_sha256, output_path, *, fixture_only=False):
    request_ref = {"path": str(Path(request_path).resolve()), "sha256": request_sha256}
    frozen = _query_contents(request_ref, fixture_only)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pin = write_json(output, frozen)
    return {"path": str(output.resolve()), "sha256": pin, "queries": len(frozen["queries"]),
            "old_candidates": len(frozen["old_candidates"])}


class SourceStop(RuntimeError):
    pass


class AuthenticationRequired(SourceStop):
    pass


class RequestFailure(RuntimeError):
    pass


class InterruptedResponse(TimeoutError):
    def __init__(self, status, headers, body, reason):
        super().__init__(reason)
        self.status, self.headers, self.body = status, headers, body


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def http_response(url, timeout, maximum, *, opener=None, data=None):
    request_deadline = time.monotonic() + timeout
    if opener is None:
        opener = urllib.request.build_opener(NoRedirect())
    request = urllib.request.Request(url, data=data, headers={
        "User-Agent": "kenggo-source-expansion/1.0 (serial; 5s minimum interval)", "Accept": "*/*"})
    try:
        response = opener.open(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        status, headers, blocks, total = response.code, dict(response.headers), [], 0
        wire = response.fp if isinstance(response, urllib.error.HTTPError) else response
        try:
            while total <= maximum:
                if wire.isclosed():
                    break
                remaining = request_deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("request_deadline")
                # urllib exposes no response timeout setter; its socket owns each blocking read.
                wire.fp.raw._sock.settimeout(remaining)
                block = wire.read1(min(64 * 1024, maximum + 1 - total))
                if not block:
                    break
                blocks.append(block)
                total += len(block)
            if total <= maximum and wire.length not in (None, 0):
                raise http.client.IncompleteRead(b"", wire.length)
        except (OSError, http.client.HTTPException) as error:
            if isinstance(error, http.client.IncompleteRead):
                blocks.append(error.partial)
            raise InterruptedResponse(status, headers, b"".join(blocks), type(error).__name__) from error
        return status, headers, b"".join(blocks)


class Client:
    def __init__(self, output, limits, deadline_utc, *, transport=None, now=utc_now,
                 monotonic=time.monotonic, cpu=time.process_time, sleep=time.sleep):
        self.limits, self.deadline = limits, deadline(deadline_utc)
        if transport is None:
            opener = urllib.request.build_opener(NoRedirect(),
                urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
            transport = lambda url, timeout, maximum, **kwargs: http_response(
                url, timeout, maximum, opener=opener, **kwargs)
        self.transport, self.now = transport, now
        self.monotonic, self.cpu, self.sleep = monotonic, cpu, sleep
        self.started, self.cpu_started, self.last_start = monotonic(), cpu(), None
        self.requests = self.response_bytes = 0
        self.log = (Path(output) / "requests.jsonl").open("x")

    def check(self):
        if self.now() >= self.deadline:
            raise SourceStop("absolute_deadline")
        if self.monotonic() - self.started >= self.limits.wall_seconds:
            raise SourceStop("source_wall_limit")
        if self.cpu() - self.cpu_started >= self.limits.cpu_seconds:
            raise SourceStop("source_cpu_limit")

    def wait(self, seconds):
        self.check()
        remaining = min((self.deadline - self.now()).total_seconds(),
                        self.limits.wall_seconds - (self.monotonic() - self.started))
        if seconds >= remaining:
            raise SourceStop("insufficient_time_for_request")
        self.sleep(max(0, seconds))
        self.check()

    def get(self, url, maximum, *, bundle=False):
        return self._fetch(url, maximum, bundle=bundle)

    def authenticate(self, credentials_file, query):
        path = Path(credentials_file).resolve()
        repository = Path(__file__).resolve().parents[2]
        if (path.is_relative_to(repository) or not path.is_file()
                or path.stat().st_uid != os.getuid() or path.stat().st_mode & 0o077
                or path.stat().st_size > 4096):
            raise ValueError("Use an owner-only credential file outside the repository")
        try:
            credentials = json.loads(path.read_bytes())
        except (ValueError, OSError):
            raise ValueError("Cannot read the local credential file") from None
        if (not isinstance(credentials, dict) or set(credentials) != {"username", "password"}
                or any(not isinstance(value, str) or not value or len(value) > 256
                       for value in credentials.values())):
            raise ValueError("Invalid local credential fields")
        body = self._fetch(query["url"], self.limits.html_bytes, allow_login=True)
        form = LoginForm()
        form.feed(body.decode("utf-8", errors="strict"))
        if form.password_forms:
            if form.password_forms != [{"action": "login.jsp", "method": "post",
                                        "fields": {"user", "password"}}]:
                raise AuthenticationRequired("unsupported_archive_login_form")
            payload = urllib.parse.urlencode({"user": credentials["username"],
                                              "password": credentials["password"]}).encode()
            self._fetch("https://www.gokgs.com/login.jsp", self.limits.html_bytes,
                        allow_login=True, data=payload)
            body = self.get(query["url"], self.limits.html_bytes)
        parse_page(body, query)
        return body

    def _fetch(self, url, maximum, *, bundle=False, allow_login=False, data=None):
        retries, redirects = 0, 0
        while True:
            allowed_url(url)
            path = urllib.parse.urlsplit(url).path
            if path == "/login.jsp" and not allow_login:
                raise AuthenticationRequired("archive_authentication_required")
            allowed_paths = ("/gameArchives.jsp", "/archives.jsp") + (("/login.jsp",) if allow_login else ())
            if (bundle and not path.endswith(".zip") or not bundle and path not in allowed_paths):
                raise ValueError("Redirect leaves the frozen archive request type")
            if data is not None and url != "https://www.gokgs.com/login.jsp":
                raise ValueError("Credentials may only be posted to the official HTTPS login")
            if path == "/login.jsp" and url != "https://www.gokgs.com/login.jsp":
                raise ValueError("Unexpected archive login address")
            if not bundle and set(urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)) - {
                    "user", "year", "month", "oldAccounts"}:
                raise ValueError("Unexpected archive query fields")
            if self.last_start is not None:
                self.wait(max(0, self.limits.min_interval_seconds - (self.monotonic() - self.last_start)))
            self.check()
            if self.requests >= self.limits.requests or self.response_bytes >= self.limits.response_bytes:
                raise SourceStop("request_or_response_limit")
            self.requests += 1
            self.last_start = self.monotonic()
            record = {"sequence": self.requests, "method": "POST" if data is not None else "GET", "url": url,
                      "started_utc": self.now().isoformat(), "start_offset_seconds": self.last_start - self.started}
            maximum = min(maximum, self.limits.response_bytes - self.response_bytes)
            timeout = min(30, (self.deadline - self.now()).total_seconds(),
                          self.limits.wall_seconds - (self.monotonic() - self.started))
            status, headers, body, retryable = None, {}, b"", False
            try:
                status, headers, body = (self.transport(url, timeout, maximum) if data is None
                                        else self.transport(url, timeout, maximum, data=data))
                self.response_bytes += len(body)
                headers = {key.lower(): value for key, value in headers.items()}
                record.update(status=status, bytes=len(body), sha256=sha256(body).hexdigest())
                if len(body) > maximum:
                    raise SourceStop("response_byte_limit")
                retryable = status in (408, 429, 500, 502, 503, 504)
            except InterruptedResponse as error:
                status, body = error.status, error.body
                headers = {key.lower(): value for key, value in error.headers.items()}
                self.response_bytes += len(body)
                record.update(status=status, bytes=len(body), sha256=sha256(body).hexdigest(), error=str(error))
                retryable = True
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                retryable = True
                record["error"] = type(error).__name__
            finally:
                record["elapsed_seconds"] = self.monotonic() - self.last_start
                self.log.write(json.dumps(record, sort_keys=True) + "\n")
                self.log.flush()
            self.check()
            if retryable:
                if retries >= self.limits.retries:
                    raise RequestFailure("transient_retries_exhausted")
                delay = 300 * (2 ** retries)
                retry_after = headers.get("retry-after", "")
                if retry_after.isdigit():
                    delay = max(delay, int(retry_after))
                elif retry_after:
                    try:
                        delay = max(delay, (parsedate_to_datetime(retry_after) - self.now()).total_seconds())
                    except (TypeError, ValueError, OverflowError):
                        pass
                self.wait(delay)
                retries += 1
                continue
            if status in (401, 403):
                raise AuthenticationRequired("archive_authentication_required")
            if status in (301, 302, 303, 307, 308):
                target = headers.get("location")
                if not target or redirects >= 5:
                    raise RequestFailure("invalid_or_excessive_redirects")
                if data is not None and status in (307, 308):
                    raise AuthenticationRequired("login_post_redirect_refused")
                url = urllib.parse.urljoin(url, target)
                allowed_url(url)
                data = None
                if urllib.parse.urlsplit(url).path == "/login.jsp" and not allow_login:
                    raise AuthenticationRequired("archive_authentication_required")
                redirects += 1
                continue
            if status != 200:
                raise RequestFailure("http_" + str(status))
            return body

    def costs(self):
        return {"requests": self.requests, "response_bytes": self.response_bytes,
                "wall_seconds": self.monotonic() - self.started,
                "cpu_seconds": self.cpu() - self.cpu_started, "gpu_seconds": 0, "workers": 1}


class LoginForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.current = None
        self.password_forms = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form":
            self.current = {"action": attrs.get("action"), "method": attrs.get("method", "get").lower(),
                            "fields": set()}
        elif tag == "input" and self.current is not None and attrs.get("name"):
            self.current["fields"].add(attrs["name"])

    def handle_endtag(self, tag):
        if tag == "form" and self.current is not None:
            if "password" in self.current["fields"]:
                self.password_forms.append(self.current)
            self.current = None


def parse_page(body, query):
    text = body.decode("utf-8", errors="strict")
    if re.search(r'name\s*=\s*[\"\']password[\"\']', text, re.I):
        raise AuthenticationRequired("archive_authentication_required")
    page = kgs_source.ArchivePage(text)
    if not page.rows and not page.unlinked_rows and "no games" not in text.lower() and "(0 games)" not in text:
        raise ValueError("unrecognized_empty_archive")
    for row in page.rows:
        key = e1_audit.source_key(row["url"])
        if key is None or not key[0].startswith(f"2025-{query['month']:02d}-"):
            raise ValueError("archive_date_mismatch")
        players = [kgs_source.player(row[color]) for color in ("black", "white")]
        if all(player is not None for player in players) and not any(
                player[0].lower() == query["user"].lower() for player in players):
            raise ValueError("archive_account_mismatch")
    links = sorted({url for url in page.links if urllib.parse.urlsplit(url).path.startswith("/servlet/archives/")
                    and urllib.parse.urlsplit(url).path.endswith(".zip")})
    if page.rows and len(links) != 1:
        raise ValueError("missing_or_ambiguous_archive_bundle")
    bundle_url = links[0] if links else None
    if bundle_url:
        allowed_url(bundle_url, bundle=True)
    return [{**row, "page_key": query["key"], "bundle_url": bundle_url} for row in page.rows]


def balanced(rows, maximum, *, seed=SEED):
    strata = defaultdict(lambda: defaultdict(list))
    for item in rows:
        row = item["archive_row"]
        ranks = [kgs_source.player(row[side])[1] for side in ("black", "white")]
        weaker = min(kgs_source.RANKS.index(rank) for rank in ranks)
        month = int(e1_audit.source_key(row["url"])[0][5:7])
        account = row["page_key"].rsplit("-2025-", 1)[0].lower()
        strata[(weaker, month)][account].append(item)
    queues = {}
    for key, accounts in sorted(strata.items()):
        queues[key] = deque((account, deque(sorted(items, key=lambda item: digest(
            [seed, "game", item["archive_row"]["url"]])))) for account, items in sorted(
                accounts.items(), key=lambda pair: digest([seed, "account", pair[0]])))
    result = []
    while queues and len(result) < maximum:
        for key in list(queues):
            account, queue = queues[key].popleft()
            result.append(queue.popleft())
            if queue:
                queues[key].append((account, queue))
            if not queues[key]:
                del queues[key]
            if len(result) >= maximum:
                break
    return result


def freeze_candidates(frozen, request, objects, rows):
    groups = dict(objects.get("current_groups", objects["old_coverage_frozen"]["groups"]))
    split = objects["split_config"]
    seed = request.get("discovery_policy", {}).get("seed", SEED)
    old_items = objects["old_coverage_frozen"]["candidates"]
    mandatory = {item["archive_row"]["url"]: item for item in old_items
                 if item["archive_row"]["url"] not in objects.get("completed_ids", ())}
    excluded = {row["url"] for row in objects["original_sample"]["candidates"]}
    excluded.update(row["source_url"] for row in objects["owner_exclusions"]["excluded_games"])
    excluded.update(objects.get("completed_ids", ()))
    unique = _unique([item["archive_row"] for item in frozen["old_candidates"]] + rows)
    by_role = defaultdict(list)
    for row in unique.values():
        role = _eligible(row, groups, split, excluded)
        if role is None:
            continue
        if row["url"] not in mandatory:
            by_role[role].append({"archive_row": row, "provisional_role": role})
    selected = list(mandatory.values())
    counts = Counter(item["provisional_role"] for item in selected)
    for role in cached.ROLES:
        maximum = request["limits"]["max_" + role + "_candidates"]
        if counts[role] > maximum:
            raise ValueError("Candidate cap would remove already inspected IDs")
        selected.extend(balanced(by_role[role], maximum - counts[role], seed=seed))
    role_queues = [deque(balanced([item for item in selected if item["provisional_role"] == role], len(selected), seed=seed))
                   for role in cached.ROLES]
    selected = []
    while any(role_queues):
        for queue in role_queues:
            if queue:
                selected.append(queue.popleft())
    for item in selected:
        for handle, group in cached.participants(item["archive_row"], groups):
            groups.setdefault(handle, group)
    return {"schema": "e2_expansion_candidates_v1", "request_sha256": frozen["request"]["sha256"],
            "groups": groups, "candidates": selected,
            "candidate_ids_sha256": digest([item["archive_row"]["url"] for item in selected]),
            "counts": dict(Counter(item["provisional_role"] for item in selected)),
            "selection_seed": seed, "selection": "Preserve the base dataset; skip previously completed records; balance rank, month and archive account; interleave the three allowed roles before reading SGFs."
            if "completed_ids" in objects else "Keep the old cached IDs; balance rank, month and archive account; interleave the three allowed roles before reading SGFs."}


def _index_bundle(bundle, limits, expanded):
    path = Path(bundle["path"])
    if path.stat().st_size > limits.bundle_bytes or file_hash(path) != bundle["sha256"]:
        raise ValueError("Bundle changed before member selection")
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
    size = sum(member.file_size for member in entries)
    if expanded + size > limits.expanded_zip_bytes:
        raise SourceStop("expanded_zip_limit")
    index = defaultdict(list)
    for member in entries:
        key = e1_audit.member_key(member.filename)
        if key is not None:
            index[key].append(member.filename)
    return index, expanded + size


def acquire(queries_path, queries_sha256, output_directory, *, credentials_file=None,
            fixture_only=False, transport=None,
            now=utc_now, monotonic=time.monotonic, cpu=time.process_time, sleep=time.sleep):
    started, cpu_started = monotonic(), cpu()
    if transport is not None and not fixture_only:
        raise ValueError("A replacement HTTP transport is allowed only for fixtures")
    frozen = read_ref({"path": str(queries_path), "sha256": queries_sha256})
    if frozen.get("schema") != QUERY_SCHEMA or frozen.get("fixture_only") is not fixture_only:
        raise ValueError("Wrong expansion query scope")
    if frozen != _query_contents(frozen["request"], fixture_only):
        raise ValueError("Query inputs or cached provenance changed after freezing")
    request, objects, limits, _ = _context(frozen["request"], fixture_only)
    output = Path(output_directory).resolve()
    if output.exists():
        raise ValueError("Use a fresh source output directory")
    execution = Path(queries_path).with_name(Path(queries_path).name + ".execution.json")
    write_json(execution, {"queries_sha256": queries_sha256, "request": frozen["request"],
                          "output_directory": str(output), "started_utc": now().isoformat()})
    output.mkdir(parents=True, exist_ok=False)
    client = Client(output, limits, request["deadline_utc"], transport=transport,
                    now=now, monotonic=monotonic, cpu=cpu, sleep=sleep)
    client.started, client.cpu_started = started, cpu_started
    rows, page_results, diagnostics = [], [], []
    stopped = None
    bundles = dict(frozen["old_bundles"])
    needed, indexes, expanded, candidate_manifest = {}, {}, 0, None
    outcome = "failed"
    authentication = "not_requested"
    first_body = None
    probe = read_ref(request["access_probe"]) if "access_probe" in request else {}

    def accounted_costs():
        costs = client.costs()
        costs["collection_requests"] = costs["requests"]
        costs["collection_response_bytes"] = costs["response_bytes"]
        costs["requests"] += probe.get("requests", 0)
        costs["response_bytes"] += probe.get("response_bytes", 0)
        costs["access_probe_requests"] = probe.get("requests", 0)
        costs["cpu_charged_seconds"] = costs["cpu_seconds"] + probe.get("cpu_charged_seconds", 0)
        return costs

    def progress(phase):
        cached.save_json(output / "progress.json", {
            "phase": phase, "stop_reason": stopped, "queries_expected": len(frozen["queries"]),
            "queries_recorded": len(page_results),
            "query_statuses": dict(Counter(item["status"] for item in page_results)),
            "frozen_candidates": 0 if candidate_manifest is None else len(candidate_manifest["candidates"]),
            "bundles_needed": len(needed), "bundles_available": len(bundles),
            "bundles_indexed": len(indexes), "expanded_zip_bytes": expanded,
            "diagnostic_count": len(diagnostics), "costs": accounted_costs()}, replace=True)

    try:
        if credentials_file is not None and frozen["queries"]:
            progress("authentication")
            try:
                first_body = client.authenticate(credentials_file, frozen["queries"][0])
                authentication = "verified"
            except (SourceStop, RequestFailure, ValueError, OSError, UnicodeError) as error:
                stopped = str(error) if isinstance(error, SourceStop) else "authentication_failed"
                authentication = "failed"
                diagnostics.append({"status": "authentication_failed", "reason": stopped})
        progress("archive_metadata")
        for query_index, query in enumerate(frozen["queries"]):
            if stopped:
                page_results.append({"key": query["key"], "status": "unattempted", "reason": stopped})
                continue
            try:
                body = first_body if query_index == 0 and first_body is not None else client.get(
                    query["url"], limits.html_bytes)
                parsed = parse_page(body, query)
                path = output / (query["key"] + ".html")
                path.write_bytes(body)
                page = {**query, "status": "retrieved", "html": reference(path), "rows": parsed}
                page_results.append(page)
                rows.extend(parsed)
            except SourceStop as error:
                stopped = str(error)
                page_results.append({"key": query["key"], "status": "failed", "reason": stopped})
            except (RequestFailure, ValueError, UnicodeError) as error:
                page_results.append({"key": query["key"], "status": "failed", "reason": str(error)})
            progress("archive_metadata")
        frame_path = output / "discovery.json"
        write_json(frame_path, {"queries_sha256": frozen["queries_sha256"], "pages": page_results, "rows": rows})
        candidate_manifest = freeze_candidates(frozen, request, objects, rows)
        candidate_path = output / "frozen_candidates.json"
        write_json(candidate_path, candidate_manifest)
        needed = {item["archive_row"]["page_key"]: item["archive_row"]["bundle_url"]
                  for item in candidate_manifest["candidates"]}
        progress("cached_bundle_index")
        for key, bundle in sorted(bundles.items()):
            if key not in needed:
                continue
            try:
                client.check()
                indexes[key], expanded = _index_bundle(bundle, limits, expanded)
            except SourceStop as error:
                stopped = str(error)
                diagnostics.append({"page_key": key, "status": "excluded_bundle", "reason": stopped})
            except zipfile.BadZipFile:
                diagnostics.append({"page_key": key, "status": "excluded_bundle", "reason": "invalid_zip"})
            progress("cached_bundle_index")
        for key, url in sorted(needed.items()):
            if key in bundles:
                continue
            if stopped:
                diagnostics.append({"page_key": key, "reason": stopped, "status": "unattempted_bundle"})
                continue
            try:
                allowed_url(url, bundle=True)
                body = client.get(url, limits.bundle_bytes, bundle=True)
                if not body.startswith(b"PK"):
                    if re.search(br'name\s*=\s*[\"\']password[\"\']', body, re.I):
                        raise AuthenticationRequired("archive_authentication_required")
                    raise ValueError("invalid_zip_response")
                path = output / (key + ".zip")
                path.write_bytes(body)
                bundles[key] = {**reference(path), "bytes": len(body), "url": url}
                indexes[key], expanded = _index_bundle(bundles[key], limits, expanded)
            except SourceStop as error:
                stopped = str(error)
                diagnostics.append({"page_key": key, "status": "failed_bundle", "reason": stopped})
            except (RequestFailure, ValueError, zipfile.BadZipFile) as error:
                diagnostics.append({"page_key": key, "status": "failed_bundle", "reason": str(error)})
            progress("bundle_retrieval")
        candidates = []
        for item in candidate_manifest["candidates"]:
            row = item["archive_row"]
            matches = indexes.get(row["page_key"], {}).get(e1_audit.source_key(row["url"]), [])
            if len(matches) != 1:
                diagnostics.append({"source_url": row["url"], "status": "unavailable_candidate",
                                    "reason": "missing_or_ambiguous_member"})
                continue
            bundle = bundles[row["page_key"]]
            candidates.append({**item, "source": {"source_url": row["url"], "archive_path": bundle["path"],
                               "archive_sha256": bundle["sha256"], "zip_member": matches[0]}})
        exposure = {key: set(map(tuple, value)) if key == "account_months" else set(value)
                    for key, value in objects["prior_exposure"].items()}
        for item in candidate_manifest["candidates"]:
            row = item["archive_row"]
            exposure["source_ids"].add(row["url"])
            exposure["groups"].update(group for _, group in cached.participants(row, candidate_manifest["groups"]))
        for query in frozen["queries"]:
            handle = kgs_eligibility.normalize_handle(query["user"])
            exposure["groups"].add(candidate_manifest["groups"].get(handle, "kgs-handle:" + handle))
            exposure["account_months"].add((handle, f"2025-{query['month']:02d}"))
        sources = {"schema": SOURCES_SCHEMA, "request_sha256": frozen["request"]["sha256"],
            "base_grouping_sha256": kgs_eligibility.grouping_digest(objects.get("current_groups", request["base_groups"])),
            "groups": candidate_manifest["groups"], "candidates": candidates,
            "owner_excluded_sgf_sha256": sorted(row["sgf_sha256"] for row in objects["owner_exclusions"]["excluded_games"]),
            "exposure": {key: sorted(value) for key, value in exposure.items()},
            "provenance": {"queries": {"path": str(Path(queries_path).resolve()), "sha256": queries_sha256},
                "discovery": reference(frame_path), "frozen_candidates": reference(candidate_path),
                "scope_decision": request["scope_decision"], "source_revision": request["source_revision"]},
            "counts": {"frozen_candidates": len(candidate_manifest["candidates"]), "accessible_candidates": len(candidates),
                "candidate_roles": candidate_manifest["counts"], "expanded_zip_bytes": expanded,
                "query_statuses": dict(Counter(item["status"] for item in page_results))},
            "authentication": authentication,
            "diagnostics": diagnostics, "status": "authentication_blocked" if authentication == "failed" or stopped == "archive_authentication_required"
                else "partial" if stopped or diagnostics or any(item["status"] != "retrieved" for item in page_results)
                else "completed", "stop_reason": stopped, "costs": accounted_costs()}
        path = output / "sources.json"
        pin = write_json(path, sources)
        outcome = sources["status"]
        return {"path": str(path), "sha256": pin, "status": sources["status"], "counts": sources["counts"]}
    finally:
        client.log.close()
        write_json(output / "costs.json", accounted_costs())
        progress(outcome)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("freeze", "acquire"))
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--inputs-sha256", required=True)
    parser.add_argument("--credentials-file", type=Path,
                        help="Owner-only username/password JSON outside the repository; never copied to output")
    args = parser.parse_args()
    if args.operation == "freeze" and args.credentials_file is not None:
        parser.error("Credentials are used only by acquire")
    result = (freeze_queries if args.operation == "freeze" else acquire)(
        args.inputs, args.inputs_sha256, args.output,
        **({"credentials_file": args.credentials_file} if args.operation == "acquire" else {}))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
