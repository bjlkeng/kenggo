"""Serial, bounded acquisition for a frozen KGS audit configuration."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import getpass
import hashlib
import http.cookiejar
import json
from pathlib import Path
import re
import resource
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

from kenggo.kgs_source import SEED, ArchivePage, candidate_reasons, select_candidates


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Client:
    def __init__(self, directory: Path, prior_costs=None, deadline_utc=None):
        self.prior = prior_costs or {}
        self.deadline = datetime.fromisoformat(deadline_utc) if deadline_utc else None
        self.log = (directory / "requests.jsonl").open("x")
        self.opener = urllib.request.build_opener(
            NoRedirect(), urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        self.started = time.monotonic()
        self.last_start = None
        self.count = self.prior.get("requests", 0)
        self.size = self.prior.get("response_bytes", 0)

    def get(self, url: str, *, data=None, limit=2 * 1024**2) -> bytes:
        parsed = urllib.parse.urlsplit(url)
        if (parsed.scheme != "https" or parsed.netloc not in {"www.gokgs.com", "files.gokgs.com"}
                or parsed.username or parsed.password or parsed.fragment):
            raise ValueError("Request outside the allowed KGS HTTPS origins")
        if data is not None and url != "https://www.gokgs.com/login.jsp":
            raise ValueError("POST allowed only for the official login")
        if self.last_start is not None:
            time.sleep(max(0, 5 - (time.monotonic() - self.last_start)))
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if (self.count >= 500 or time.monotonic() - self.started + self.prior.get("wall_seconds", 0) >= 3600
                or self.size >= 100 * 1024**2
                or usage.ru_utime + usage.ru_stime + self.prior.get("cpu_seconds", 0) >= 900
                or (self.deadline is not None and datetime.now(timezone.utc) >= self.deadline)):
            raise RuntimeError("Acquisition budget exhausted")
        self.count += 1
        self.last_start = time.monotonic()
        record = dict(sequence=self.count, method="POST" if data is not None else "GET",
                      url=url, started_utc=datetime.now(timezone.utc).isoformat(),
                      start_offset_seconds=self.last_start - self.started)
        request = urllib.request.Request(url, data=data, headers={
            "User-Agent": "kenggo-source-audit/0.1 (serial; 5s minimum interval)",
            "Accept": "*/*"})
        try:
            try:
                response = self.opener.open(request, timeout=30)
            except urllib.error.HTTPError as exc:
                response = exc
            with response:
                record["status"] = response.code
                location = response.headers.get("Location")
                maximum = min(limit, 100 * 1024**2 - self.size)
                body = response.read(maximum + 1)
            self.size += len(body)
            record.update(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
        except Exception as exc:
            record["error"] = type(exc).__name__
            raise
        finally:
            record["elapsed_seconds"] = time.monotonic() - self.last_start
            self.log.write(json.dumps(record) + "\n")
            self.log.flush()
        if len(body) > maximum:
            raise RuntimeError("Response byte budget exhausted")
        if record["status"] >= 400:
            raise RuntimeError(f"HTTP {record['status']}; stopping without retry")
        if record["status"] in {301, 302, 303, 307, 308}:
            if not location or (data is not None and record["status"] in {307, 308}):
                raise RuntimeError("Unsafe or missing redirect")
            return self.get(urllib.parse.urljoin(url, location), limit=limit)
        return body

    def login(self):
        user = getpass.getpass("KGS account (hidden): ")
        password = getpass.getpass("KGS password (hidden): ")
        self.get("https://www.gokgs.com/archives.jsp")
        body = self.get("https://www.gokgs.com/login.jsp", data=urllib.parse.urlencode(
            {"user": user, "password": password}).encode())
        if b"You are currently logged in as" not in body or b'name="password"' in body:
            raise RuntimeError("Login not confirmed; stopping after one attempt")
        del user, password, body


def write_json(path: Path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def cached_page(directory: Path | None, key: str, url: str):
    if directory is None or not (directory / (key + ".html")).exists():
        return None
    public = (directory / (key + ".html")).read_bytes().decode("utf-8")
    metadata = directory / (key + ".json")
    if metadata.exists():
        record = json.loads(metadata.read_text())
        if record["url"] != url:
            raise ValueError("Cached page belongs to a different query")
        if record["public_html_sha256"] != hashlib.sha256(public.encode()).hexdigest():
            raise ValueError("Cached page content changed")
        original_hash = record["body_sha256"]
    else:
        requests = [json.loads(line) for line in (directory / "requests.jsonl").read_text().splitlines()]
        matches = [r for r in requests if r["url"] == url and r.get("status") == 200 and "sha256" in r]
        if not matches:
            raise ValueError("Cached response has no successful request provenance")
        original_hash = matches[-1]["sha256"]
    return public, original_hash


def acquire(config_path: Path, output: Path, prior_attempt: Path | None = None):
    config = json.loads(config_path.read_text())
    if config["seed"] != SEED:
        raise ValueError("Configuration seed differs from the implemented selection rule")
    output.mkdir(parents=True, exist_ok=True)
    prior = (json.loads((prior_attempt / "costs.json").read_text()) if prior_attempt
             else config.get("prior_attempt_costs"))
    client = Client(output, prior, config.get("deadline_utc"))
    write_json(output / "run.json", dict(
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        prior_attempt=str(prior_attempt) if prior_attempt else None,
        started_utc=datetime.now(timezone.utc).isoformat(),
        code_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                     for name in ("kgs_acquire.py", "kgs_source.py")}))
    all_rows = []
    pages = []
    outcome = "failed"
    try:
        client.login()
        for month in range(1, 13):
            for account in config["accounts"]:
                if (month not in account["months"]
                        or account["user"] in config.get("excluded_discovery_accounts", {})):
                    continue
                user = account["user"]
                query = urllib.parse.urlencode(dict(user=user, year=2025, month=month, oldAccounts="y"))
                url = "https://www.gokgs.com/gameArchives.jsp?" + query
                key = f"{user}-2025-{month:02d}"
                cached = cached_page(prior_attempt, key, url)
                if cached is None:
                    body = client.get(url)
                    if b'name="password"' in body:
                        raise RuntimeError("Archive returned login form; session expired")
                    text, body_hash = body.decode("utf-8", errors="strict"), hashlib.sha256(body).hexdigest()
                else:
                    text, body_hash = cached
                public = re.sub(r"You are currently logged in as.*?(?=</p>|</body>|$)",
                                "You are currently logged in as [redacted].", text, flags=re.S)
                with (output / (key + ".html")).open("x") as stream:
                    stream.write(public)
                page = ArchivePage(text)
                if (not page.rows and not page.unlinked_rows and "no games" not in text.lower()
                        and re.search(r"\(0 games\)", text) is None):
                    raise RuntimeError("Unrecognized empty archive response")
                if any(urllib.parse.urlsplit(row["url"]).path.split("/")[2] != "2025"
                       or int(urllib.parse.urlsplit(row["url"]).path.split("/")[3]) != month
                       for row in page.rows):
                    raise RuntimeError("Archive returned a different date range")
                bundles = sorted(set(link for link in page.links if
                                     urllib.parse.urlsplit(link).path.startswith("/servlet/archives/")
                                     and urllib.parse.urlsplit(link).path.endswith(".zip")))
                if page.rows and len(bundles) != 1:
                    raise RuntimeError("Missing or ambiguous official ZIP link")
                record = dict(key=key, user=user, month=month, url=url,
                              body_sha256=body_hash, public_html_sha256=hashlib.sha256(public.encode()).hexdigest(),
                              cache_source=str(prior_attempt) if cached else None,
                              bundle_url=bundles[0] if bundles else None, rows=page.rows)
                write_json(output / (key + ".json"), record)
                pages.append(record)
                all_rows.extend({**row, "page_key": key, "bundle_url": record["bundle_url"]}
                                for row in page.rows)
                print(json.dumps(dict(stage="discovery", page=len(pages), key=key,
                                      rows=len(page.rows), total_rows=len(all_rows))), flush=True)
        selected = select_candidates(all_rows)
        exclusions = Counter(reason for row in all_rows for reason in candidate_reasons(row))
        write_json(output / "frame.json", dict(rows=all_rows, row_exclusion_counts=dict(exclusions)))
        write_json(output / "sample.json", dict(
            seed=config["seed"], config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
            frozen_utc=datetime.now(timezone.utc).isoformat(), candidates=selected))
        print(json.dumps(dict(stage="sample_frozen", candidates=len(selected),
                              total_rows=len(all_rows), exclusions=dict(exclusions))), flush=True)
        if len(selected) != 1000:
            outcome = "insufficient_frame"
            print("Insufficient frame; no SGFs downloaded. Stop and record the coverage decision.", flush=True)
            return
        print("Type retrieve to download official ZIPs for the frozen sample, or stop to exit.", flush=True)
        if input().strip() != "retrieve":
            outcome = "sample_only"
            return
        bundles = {row["page_key"]: row["bundle_url"] for row in selected}
        expanded = 0
        for index, (key, url) in enumerate(sorted(bundles.items()), 1):
            body = client.get(url, limit=8 * 1024**2)
            path = output / (key + ".zip")
            with path.open("xb") as stream:
                stream.write(body)
            with zipfile.ZipFile(path) as archive:
                expanded += sum(member.file_size for member in archive.infolist())
                if expanded > 256 * 1024**2:
                    raise RuntimeError("Expanded archive budget exhausted")
            print(json.dumps(dict(stage="retrieval", bundle=index, total_bundles=len(bundles),
                                  key=key, bytes=len(body), expanded_bytes=expanded)), flush=True)
        outcome = "retrieved"
    finally:
        client.log.close()
        write_json(output / "costs.json", dict(outcome=outcome, requests=client.count, response_bytes=client.size,
                   wall_seconds=time.monotonic() - client.started + client.prior.get("wall_seconds", 0),
                   cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime +
                               resource.getrusage(resource.RUSAGE_SELF).ru_stime +
                               client.prior.get("cpu_seconds", 0), gpu_hours=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prior-attempt", type=Path, help="Consume this stopped attempt's cumulative resource costs")
    args = parser.parse_args()
    acquire(args.config, args.output, args.prior_attempt)


if __name__ == "__main__":
    main()
