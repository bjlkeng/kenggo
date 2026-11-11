"""Freeze and acquire a sealed KGS final cohort without inspecting game targets."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import resource
import time
from urllib.parse import urlencode, urlsplit
import zipfile

from kenggo.canonical_audit import load_frozen
from kenggo.e1_audit import member_key, source_key
from kenggo.kgs_acquire import Client, write_json
from kenggo.kgs_eligibility import normalize_handle
from kenggo.kgs_source import ArchivePage, candidate_reasons, player


SEED = "kenggo-6yr.7-final-cohort-v1"
FRAME_SHA256 = "e69698029b799a3779bbe460d16daad8ed6c90ffbceac9d80fe3ccb0122cead6"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group(handle):
    return "kgs-handle:" + normalize_handle(handle)


def participants(row):
    parsed = [player(row[color]) for color in ("black", "white")]
    if any(p is None for p in parsed):
        raise ValueError("Unrecognized player metadata")
    return [(p[0], group(p[0])) for p in parsed]


def choose_groups(rows, exposed):
    candidates = {}
    for row in rows:
        if candidate_reasons(row) or int(urlsplit(row["url"]).path.split("/")[3]) not in (11, 12):
            continue
        for handle, key in participants(row):
            if key not in exposed and "bot" not in handle.lower():
                candidates.setdefault(key, handle)
    ordered = sorted(candidates, key=lambda key: hashlib.sha256(
        (SEED + ":group:" + key).encode()).hexdigest())
    if len(ordered) < 20:
        raise ValueError("Insufficient untouched discovery groups")
    return [{"group": key, "user": candidates[key]} for key in ordered[:20]]


def select_rows(rows, reservation):
    exposed = frozenset(reservation["exposed_groups"])
    seeds = {a["group"] for a in reservation["accounts"]}
    unique = {}
    for row in rows:
        if candidate_reasons(row):
            continue
        sides = participants(row)
        keys = {key for _, key in sides}
        if (keys & exposed or not keys & seeds or len(keys) != 2
                or any("bot" in handle.lower() for handle, _ in sides)
                or int(urlsplit(row["url"]).path.split("/")[3]) not in (11, 12)):
            continue
        if row["url"] in unique:
            prior = unique[row["url"]]
            if any(prior[k] != row[k] for k in ("white", "black", "type", "setup", "result")):
                raise ValueError("Conflicting duplicate archive row")
        else:
            unique[row["url"]] = row
    return sorted(unique.values(), key=lambda row: hashlib.sha256(
        (SEED + ":game:" + row["url"]).encode()).hexdigest())[:reservation["candidate_cap"]]


def reserve(source, output):
    audits, _, _, exposure, _ = load_frozen(source)
    frame = source / "frame.json"
    if sha(frame) != FRAME_SHA256:
        raise ValueError("Frozen discovery frame changed")
    accounts = choose_groups(json.loads(frame.read_text())["rows"], exposure.groups)
    value = {
        "version": "kgs_final_reservation_v1", "seed": SEED,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_frame_sha256": FRAME_SHA256,
        "original_sample_sha256": sha(source / "sample.json"),
        "original_audit_count": len(audits), "exposed_groups": sorted(exposure.groups),
        "accounts": accounts, "months": [11, 12], "year": 2025, "candidate_cap": 200,
        "selection": "First20 SHA256(seed+:group:+group) unexposed non-bot-named groups appearing in in-scope Nov/Dec archive metadata; first200 distinct SHA256(seed+:game:+url) eligible new IDs, no replacement.",
        "scope": "Owner-approved presumed-human ordinary-ranked KGS2025 research use; unchanged15k..1d, even19x19, Japanese6.5,600..1800seconds and3..5x20..60byoyomi after mechanical SGF checks.",
        "suspected_bot_policy": "Exclude either participant whose case-folded handle contains bot, in addition to existing documented-bot exclusions; this is conservative screening, not proof of bot control.",
        "final_policy": "Reserve every selected candidate participant group before SGF retrieval; reject either exposed side, reserved-cohort bridges and duplicate exposure; mechanically validate and seal targets, no model/referee selection or diagnostics from final rows before E7.",
        "limits": {"requests": 86, "response_bytes": 104857600, "wall_seconds": 900,
                   "cpu_seconds": 180, "workers": 1, "minimum_request_interval_seconds": 5,
                   "automatic_retries": 0, "expanded_zip_bytes": 268435456},
    }
    write_json(output, value)
    print(json.dumps({"reservation": str(output), "sha256": sha(output), "groups": len(accounts)}))


class HoldoutClient(Client):
    def get(self, url, *, data=None, limit=2 * 1024**2):
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if (self.count >= 86 or time.monotonic() - self.started >= 900
                or usage.ru_utime + usage.ru_stime >= 180):
            raise RuntimeError("Holdout acquisition budget exhausted")
        return super().get(url, data=data, limit=limit)


def acquire(reservation_path, expected, output):
    if sha(reservation_path) != expected:
        raise ValueError("Reservation differs from the independent pin")
    reservation = json.loads(reservation_path.read_text())
    if (reservation["version"] != "kgs_final_reservation_v1" or reservation["seed"] != SEED
            or reservation["year"] != 2025 or reservation["months"] != [11, 12]
            or reservation["candidate_cap"] != 200 or len(reservation["accounts"]) != 20):
        raise ValueError("Unsupported acquisition scope")
    output.mkdir(parents=True, exist_ok=False)
    output.chmod(0o700)
    client = HoldoutClient(output)
    outcome = "failed"
    try:
        client.login()
        rows, pages = [], {}
        for account in reservation["accounts"]:
            for month in reservation["months"]:
                user = account["user"]
                if group(user) != account["group"]:
                    raise ValueError("Reserved handle/group mismatch")
                key = f"{user}-2025-{month:02d}"
                url = "https://www.gokgs.com/gameArchives.jsp?" + urlencode(
                    dict(user=user, year=2025, month=month, oldAccounts="y"))
                body = client.get(url)
                text = body.decode("utf-8")
                if 'name="password"' in text:
                    raise RuntimeError("Session expired; no retry")
                page = ArchivePage(text)
                if (not page.rows and not page.unlinked_rows and "no games" not in text.lower()
                        and re.search(r"\(0 games\)", text) is None):
                    raise ValueError("Unrecognized empty archive response")
                if any(source_key(row["url"])[0][:7] != f"2025-{month:02d}" for row in page.rows):
                    raise ValueError("Archive date scope changed")
                bundles = sorted({link for link in page.links if urlsplit(link).path.startswith(
                    "/servlet/archives/") and urlsplit(link).path.endswith(".zip")})
                if page.rows and len(bundles) != 1:
                    raise ValueError("Missing or ambiguous archive bundle")
                public = re.sub(r"You are currently logged in as.*?(?=</p>|</body>|$)",
                                "You are currently logged in as [redacted].", text, flags=re.S)
                (output / (key + ".html")).write_text(public)
                record = {"key": key, "url": url, "body_sha256": hashlib.sha256(body).hexdigest(),
                          "public_html_sha256": sha(output / (key + ".html")),
                          "bundle_url": bundles[0] if bundles else None, "rows": page.rows}
                write_json(output / (key + ".json"), record)
                pages[key] = record
                rows.extend({**row, "page_key": key, "bundle_url": record["bundle_url"]} for row in page.rows)
                print(json.dumps({"stage": "metadata", "pages": len(pages), "planned_pages": 40}), flush=True)
        candidates = select_rows(rows, reservation)
        test_groups = sorted({key for row in candidates for _, key in participants(row)}
                             | {a["group"] for a in reservation["accounts"]})
        write_json(output / "sample.json", {"version": "sealed_kgs_sample_v1", "reservation_sha256": expected,
                    "frozen_utc": datetime.now(timezone.utc).isoformat(), "test_groups": test_groups,
                    "candidates": candidates})
        bundles = {row["page_key"]: row["bundle_url"] for row in candidates}
        expanded, hashes = 0, {}
        for key, url in sorted(bundles.items()):
            body = client.get(url, limit=8 * 1024**2)
            path = output / (key + ".zip")
            path.write_bytes(body)
            hashes[key] = sha(path)
            with zipfile.ZipFile(path) as z:
                expanded += sum(i.file_size for i in z.infolist())
            if expanded > 256 * 1024**2:
                raise RuntimeError("Expanded ZIP cap")
            print(json.dumps({"stage": "sealed_retrieval", "bundles": len(hashes), "planned_bundles": len(bundles)}), flush=True)
        records = []
        for row in candidates:
            path = output / (row["page_key"] + ".zip")
            with zipfile.ZipFile(path) as z:
                matches = [i for i in z.infolist() if member_key(i.filename) == source_key(row["url"])]
                if len(matches) != 1 or matches[0].file_size > 1024**2:
                    raise ValueError("Ambiguous or oversized selected SGF")
                data = z.read(matches[0])
            records.append({"archive_row": row, "source": {"source_url": row["url"],
                "sgf_sha256": hashlib.sha256(data).hexdigest(), "archive_path": str(path),
                "archive_sha256": hashes[row["page_key"]], "zip_member": matches[0].filename}})
        write_json(output / "sealed_index.json", {"version": "sealed_kgs_index_v1",
                   "reservation_sha256": expected, "sample_sha256": sha(output / "sample.json"),
                   "test_groups": test_groups, "records": records})
        outcome = "sealed"
        print(json.dumps({"stage": "sealed", "candidate_count": len(candidates),
                          "index_sha256": sha(output / "sealed_index.json"),
                          "test_group_count": len(test_groups)}), flush=True)
    finally:
        client.log.close()
        usage = resource.getrusage(resource.RUSAGE_SELF)
        write_json(output / "costs.json", {"outcome": outcome, "requests": client.count,
                   "response_bytes": client.size, "wall_seconds": time.monotonic() - client.started,
                   "cpu_seconds": usage.ru_utime + usage.ru_stime, "gpu_seconds": 0,
                   "reservation_sha256": expected, "tool_sha256": sha(Path(__file__))})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("reserve")
    plan.add_argument("source", type=Path)
    plan.add_argument("output", type=Path)
    run = commands.add_parser("acquire")
    run.add_argument("reservation", type=Path)
    run.add_argument("sha256")
    run.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "reserve":
        reserve(args.source, args.output)
    else:
        acquire(args.reservation, args.sha256, args.output)
