from dataclasses import replace
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

from kenggo import data_growth_source as growth
from kenggo import expansion_source as source


def row(name, *, black="Alice", white="Bob", month=1, rank="5k", **changes):
    key = f"{black.lower()}-2025-{month:02d}"
    return {"viewable": "Yes", "black": f"{black} [{rank}]", "white": f"{white} [{rank}]",
        "setup": "19×19", "start": f"{month}/1/25 1:00 PM", "type": "Ranked", "result": "W+R",
        "url": f"https://files.gokgs.com/games/2025/{month}/1/{name}.sgf", "page_key": key,
        "bundle_url": f"https://www.gokgs.com/servlet/archives/en_US/{key}.zip", **changes}


def member(item):
    return item["url"].split("/games/", 1)[1]


def zip_bytes(rows):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in rows:
            archive.writestr(member(item), b"SGF payload must stay unopened")
    return stream.getvalue()


def html_page(rows):
    import html
    return ("<table>" + "".join('<tr><td><a href="' + html.escape(item["url"], quote=True) + '">Yes</a></td>'
        + "".join("<td>" + html.escape(item[field]) + "</td>" for field in
                    ("white", "black", "setup", "start", "type", "result")) + "</tr>" for item in rows)
        + '</table><a href="' + html.escape(rows[0]["bundle_url"], quote=True) + '">ZIP</a>').encode()


class Clock:
    def __init__(self):
        self.seconds = 0
        self.origin = datetime(2026, 11, 21, tzinfo=timezone.utc)

    def now(self):
        return self.origin + timedelta(seconds=self.seconds)

    def monotonic(self):
        return self.seconds

    def sleep(self, seconds):
        self.seconds += seconds

    def kwargs(self):
        return {"now": self.now, "monotonic": self.monotonic, "cpu": lambda: 0, "sleep": self.sleep}


class Fixture:
    def __init__(self, directory, rows=None):
        self.root = Path(directory)
        self.rows = rows if rows is not None else [row("new")]
        self.anchor = row("old")
        self.archive = self.root / "alice-2025-01.zip"
        self.archive.write_bytes(zip_bytes([self.anchor, *self.rows]))
        self.groups = {name: "kgs-handle:" + name for name in ("alice", "bob", "valid", "final", "libertybot")}
        self.groups["alias"] = "kgs-handle:final"
        self.split = {"version": "whole_game_splits_v1", "fixture_only": True,
            "sealed_source_ids": [], "plan": {"grouping_sha256": source.kgs_eligibility.grouping_digest(self.groups),
                "train_before": "2025-09-01", "confirmation_from": "2025-11-01",
                "validation_groups": ["kgs-handle:valid"], "test_groups": ["kgs-handle:final"]}}
        self.exposure = {"groups": ["kgs-handle:alice"], "source_ids": [self.anchor["url"]],
            "sgf_hashes": ["a" * 64], "sequence_hashes": ["b" * 64],
            "account_months": [["alice", "2025-01"]]}
        self.index = {"schema": "e2_komi_data_index_v1", "canonical_sha256": "c" * 64,
                      "records": [], "rejected": [], "study_rejections": []}
        self.parent = {"schema": "e2_komi_expansion_v1", "artifacts": {}}
        self.frozen = {"candidates": [{"archive_row": self.anchor, "provisional_role": "train"}]}
        self.previous = {"schema": source.SOURCES_SCHEMA, "groups": self.groups,
            "owner_excluded_sgf_sha256": ["d" * 64], "exposure": self.exposure,
            "candidates": [{**self.frozen["candidates"][0], "source": {
                "source_url": self.anchor["url"], "archive_path": str(self.archive),
                "archive_sha256": source.file_hash(self.archive), "zip_member": member(self.anchor)}}]}
        self.discovery = {"rows": self.rows, "pages": [{"key": self.anchor["page_key"], "status": "retrieved"}]}
        self.refresh()

    def save(self, name, value):
        path = self.root / name
        path.write_bytes(source.encoded(value))
        return source.reference(path)

    def refresh(self):
        self.split["plan"]["grouping_sha256"] = source.kgs_eligibility.grouping_digest(self.groups)
        refs = {"groups": self.save("groups.json", self.groups), "split_manifest": self.save("split.json", self.split),
                "exposure": self.save("exposure.json", self.exposure),
                "canonical": {"path": str(self.root / "never-read.parquet"), "sha256": "c" * 64}}
        self.index["split_sha256"] = refs["split_manifest"]["sha256"]
        refs["base_index"] = self.save("index.json", self.index)
        self.parent["artifacts"] = refs
        self.parent_ref = self.save("summary.json", self.parent)
        self.previous["provenance"] = {"frozen_candidates": self.save("frozen.json", self.frozen)}
        self.previous_ref = self.save("previous.json", self.previous)
        self.discovery_ref = self.save("discovery.json", self.discovery)

    def freeze(self, name="sources.json", **kwargs):
        return growth.freeze_sources(self.parent_ref, [self.discovery_ref], [self.previous_ref],
            self.root / name, seed="fixture-growth", fixture_only=True, **kwargs)

    @staticmethod
    def read(reference):
        return json.loads(Path(reference["path"]).read_bytes())

    def collect(self, transport, **kwargs):
        clock = Clock()
        limits = kwargs.pop("limits", growth.CollectionLimits(pages=2))
        return growth.collect_more(self.parent_ref, [self.discovery_ref], [self.previous_ref], self.root / "collection",
            seed="fixture-growth", deadline_utc="2026-11-22T00:00:00Z",
            limits=limits, fixture_only=True, transport=transport,
            **clock.kwargs(), **kwargs)


class CachedGrowthTests(unittest.TestCase):
    def test_cached_freeze_preserves_scope_exposure_and_never_reads_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            counters = {"source_metadata_bytes": 7}
            with patch.object(zipfile.ZipFile, "open", side_effect=AssertionError("No SGF decoding")):
                result = fixture.freeze(counters=counters)
            data = fixture.read(result)
            self.assertEqual(data["schema"], growth.SCHEMA)
            self.assertEqual(data["parent_summary"], fixture.parent_ref)
            self.assertEqual(data["base_grouping_sha256"], fixture.split["plan"]["grouping_sha256"])
            self.assertEqual([item["source"]["source_url"] for item in data["candidates"]], [fixture.rows[0]["url"]])
            self.assertEqual(data["owner_excluded_sgf_sha256"], ["d" * 64])
            self.assertEqual(data["exposure"]["sgf_hashes"], ["a" * 64])
            self.assertEqual(data["exposure"]["sequence_hashes"], ["b" * 64])
            self.assertIn(fixture.anchor["url"], data["exposure"]["source_ids"])
            self.assertFalse(result["cache_exhausted"])
            self.assertGreater(counters["source_metadata_bytes"], 7)
            self.assertEqual(data["costs"]["sgf_members_decoded"], 0)

    def test_heldout_aliases_chronology_bots_and_wrong_ranks_are_filtered_before_zip(self):
        rows = [row("test", white="Final"), row("alias", white="Alias"), row("valid", white="Valid"),
                row("development", month=9), row("confirmation", month=11), row("bot", white="libertybot"),
                row("wrongrank", rank="2d"), row("handicap", setup="19×19 H2"), row("notranked", type="Free")]
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp, rows)
            with patch.object(growth, "_index_bundle", side_effect=AssertionError("No protected ZIP selection")):
                result = fixture.freeze()
            self.assertEqual(result["counts"]["eligible_source_ids"], 0)
            self.assertTrue(result["cache_exhausted"])

    def test_all_processing_history_and_old_frozen_ids_stay_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [row(str(i)) for i in range(6)]
            fixture = Fixture(tmp, rows)
            for field, item in zip(("records", "rejected", "study_rejections"), rows[:3], strict=True):
                fixture.index[field].append({"source_url": item["url"]})
            fixture.index["processed_source_ids"] = [rows[3]["url"]]
            fixture.frozen["candidates"].append({"archive_row": rows[4], "provisional_role": "train"})
            fixture.refresh()
            data = fixture.read(fixture.freeze())
            self.assertEqual([item["source"]["source_url"] for item in data["candidates"]], [rows[5]["url"]])

    def test_unused_growth_candidates_survive_the_next_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp, [row(str(index)) for index in range(10)])
            first = fixture.freeze()
            used = [item["source"]["source_url"] for item in fixture.read(first)["candidates"][:2]]
            fixture.parent["schema"] = "e2_data_growth_v1"
            fixture.index.update(schema="e2_data_growth_index_v1", processed_source_ids=used)
            fixture.refresh()
            second = growth.freeze_sources(fixture.parent_ref, [fixture.discovery_ref], [fixture.previous_ref,
                {key: first[key] for key in ("path", "sha256")}], fixture.root / "second.json",
                seed="fixture-growth", fixture_only=True)
            remaining = fixture.read(second)["candidates"]
            self.assertEqual(len(remaining), 8)
            self.assertTrue(set(used).isdisjoint(item["source"]["source_url"] for item in remaining))

    def test_metadata_aliases_are_deduplicated_and_order_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [row(str(i), rank="15k" if i % 2 else "1d") for i in range(8)]
            fixture = Fixture(tmp, rows)
            fixture.discovery["rows"] = rows + [rows[0]]
            fixture.refresh()
            first = fixture.read(fixture.freeze())
            fixture.discovery["rows"] = list(reversed(rows))
            fixture.refresh()
            second = fixture.read(fixture.freeze("reordered.json"))
            self.assertEqual(first["candidates"], second["candidates"])
            self.assertEqual(len(first["candidates"]), 8)
            self.assertEqual({item["archive_row"]["black"] for item in first["candidates"][:2]}, {"Alice [15k]", "Alice [1d]"})

    def test_conflicting_metadata_is_rejected_before_archive_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            fixture.discovery["rows"].append({**fixture.rows[0], "result": "B+R"})
            fixture.refresh()
            with patch.object(growth, "_index_bundle", side_effect=AssertionError("No ZIP selection")):
                with self.assertRaisesRegex(ValueError, "Conflicting archive"):
                    fixture.freeze()

    def test_changed_metadata_and_archive_hashes_are_rejected(self):
        for kind in ("metadata", "archive"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                fixture = Fixture(tmp)
                path = Path(fixture.discovery_ref["path"]) if kind == "metadata" else fixture.archive
                path.write_bytes(path.read_bytes() + b" ")
                with self.assertRaisesRegex(ValueError, "changed"):
                    fixture.freeze()

    def test_changed_group_split_or_owner_exclusion_is_rejected(self):
        for kind in ("group", "dates", "cohorts", "scope", "owner"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                fixture = Fixture(tmp)
                if kind == "group":
                    fixture.previous["groups"] = {**fixture.groups, "alice": "kgs-handle:other"}
                elif kind == "dates":
                    fixture.split["plan"]["train_before"] = "2025-10-01"
                elif kind == "cohorts":
                    fixture.split["plan"]["validation_groups"].append("kgs-handle:final")
                elif kind == "scope":
                    fixture.split["fixture_only"] = False
                else:
                    fixture.previous["owner_excluded_sgf_sha256"] = ["not-a-hash"]
                fixture.refresh()
                with self.assertRaises(ValueError):
                    fixture.freeze()

    def test_missing_cache_or_member_reports_exhaustion_without_hidden_limit(self):
        for kind in ("cache", "member", "duplicate"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                fixture = Fixture(tmp)
                if kind == "cache":
                    fixture.archive.unlink()
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        fixture.archive.write_bytes(zip_bytes([fixture.anchor] if kind == "member"
                            else [fixture.anchor, fixture.rows[0], fixture.rows[0]]))
                    fixture.previous["candidates"][0]["source"]["archive_sha256"] = source.file_hash(fixture.archive)
                    fixture.refresh()
                result = fixture.freeze()
                self.assertTrue(result["cache_exhausted"])
                self.assertEqual(result["counts"]["unavailable_candidates"], 1)
                self.assertEqual(result["counts"]["deferred_candidates"], 0)

    def test_candidate_limit_is_explicit_and_unused_candidates_remain_in_exposure(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp, [row(str(i)) for i in range(3)])
            result = fixture.freeze(limits=growth.SourceLimits(max_candidates=1))
            self.assertEqual(result["counts"]["selected_candidates"], 1)
            self.assertEqual(result["counts"]["deferred_candidates"], 2)
            self.assertFalse(result["cache_exhausted"])
            self.assertTrue({item["url"] for item in fixture.rows} <= set(fixture.read(result)["exposure"]["source_ids"]))

    def test_budget_checks_preserve_costs_and_do_not_publish_partial_source_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            counters = {}
            def check():
                if counters.get("source_metadata_bytes", 0) > 0:
                    raise RuntimeError("whole-experiment-deadline")
            with self.assertRaisesRegex(RuntimeError, "whole-experiment-deadline"):
                fixture.freeze(check_budget=check, counters=counters)
            self.assertGreater(counters["source_metadata_bytes"], 0)
            self.assertFalse((fixture.root / "sources.json").exists())

    def test_metadata_expansion_and_output_byte_limits(self):
        for field in ("metadata_bytes", "expanded_zip_bytes", "output_bytes"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                fixture = Fixture(tmp)
                with self.assertRaises((ValueError, source.SourceStop)):
                    fixture.freeze(limits=replace(growth.SourceLimits(), **{field: 1}))
                self.assertFalse((fixture.root / "sources.json").exists())

    def test_output_is_one_use(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            first = fixture.freeze()
            with self.assertRaises(FileExistsError):
                fixture.freeze()
            self.assertEqual(source.file_hash(first["path"]), first["sha256"])


class CollectionTests(unittest.TestCase):
    def test_queries_use_safe_rank_hints_and_skip_exposed_pages(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp, [row("normal"), row("test", black="Final"), row("valid", black="Valid"),
                row("alias", black="Alias"), row("bot", black="libertybot"), row("later", black="Later", month=10)])
            context = growth._context(fixture.parent_ref, [fixture.discovery_ref], [fixture.previous_ref],
                                     growth._Budget(growth.SourceLimits(), None, None), True)
            queries = growth._queries(context, 100)
            self.assertEqual({query["user"] for query in queries}, {"alice", "bob"})
            self.assertNotIn(("alice", 1), {(query["user"], query["month"]) for query in queries})
            self.assertTrue(all(1 <= query["month"] <= 8 and query["role"] == "train" for query in queries))
            self.assertEqual(queries, growth._queries(context, 100))

    def test_collection_freezes_candidates_before_zip_and_preserves_sources_for_later_rounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            calls = []
            new_rows = {}
            def transport(url, timeout, maximum):
                from urllib.parse import parse_qs, urlsplit
                calls.append(url)
                if url.endswith(".zip"):
                    self.assertTrue((fixture.root / "collection/frozen_candidates.json").is_file())
                    return 200, {}, zip_bytes(new_rows[url])
                query = parse_qs(urlsplit(url).query)
                user, month = query["user"][0], int(query["month"][0])
                item = row(f"fresh{user}{month}", black=user, white="Carol", month=month)
                protected = row(f"sealed{user}{month}", black=user, white="Final", month=month)
                new_rows[item["bundle_url"]] = [item, protected]
                return 200, {}, html_page([item, protected])
            counters = {}
            original_open = zipfile.ZipFile.open
            def no_member_reads(archive, name, mode="r", **kwargs):
                if mode != "w":
                    raise AssertionError("No SGF decoding")
                return original_open(archive, name, mode, **kwargs)
            with patch.object(zipfile.ZipFile, "open", no_member_reads):
                result = fixture.collect(transport, counters=counters)
            data = fixture.read(result)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["queries_selected"], 2)
            self.assertEqual(data["costs"]["requests"], 4)
            self.assertEqual(counters["source_requests"], 4)
            self.assertEqual(len(data["candidates"]), 3)
            self.assertTrue(all(item["archive_row"]["white"] != "Final [5k]" for item in data["candidates"]))
            source_ref = {key: result[key] for key in ("path", "sha256")}
            next_result = growth.freeze_sources(fixture.parent_ref, [fixture.discovery_ref, result["discovery"]],
                [fixture.previous_ref, source_ref], fixture.root / "next.json", seed="fixture-growth", fixture_only=True)
            self.assertEqual(fixture.read(next_result)["candidates"], data["candidates"])
            ledger = [json.loads(line) for line in (fixture.root / "collection/requests.jsonl").read_text().splitlines()]
            self.assertTrue(all(right["start_offset_seconds"] - left["start_offset_seconds"] >= 5
                                for left, right in zip(ledger, ledger[1:])))

    def test_request_failures_are_retained_and_query_pages_are_not_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            result = fixture.collect(lambda *args: (404, {}, b"missing"))
            data = fixture.read(result)
            self.assertEqual(data["status"], "partial")
            self.assertEqual(data["costs"]["requests"], 2)
            self.assertEqual(result["queries_selected"], 2)
            queries = fixture.read(result["queries"])["queries"]
            self.assertTrue({(item["user"], f"2025-{item['month']:02d}") for item in queries}
                            <= set(map(tuple, data["exposure"]["account_months"])))
            self.assertTrue(all(page["status"] == "failed" for page in fixture.read(result["discovery"])["pages"]))
            next_context = growth._context(fixture.parent_ref, [fixture.discovery_ref, result["discovery"]],
                [fixture.previous_ref, {key: result[key] for key in ("path", "sha256")}],
                growth._Budget(growth.SourceLimits(), None, None), True)
            next_queries = growth._queries(next_context, 100)
            self.assertTrue({item["key"] for item in queries}.isdisjoint(item["key"] for item in next_queries))

    def test_downloaded_archives_survive_a_candidate_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            archives = {}
            def transport(url, *args):
                from urllib.parse import parse_qs, urlsplit
                if url.endswith(".zip"):
                    return 200, {}, archives[url]
                query = parse_qs(urlsplit(url).query)
                user, month = query["user"][0], int(query["month"][0])
                item = row(f"fresh{user}{month}", black=user, white="Carol", month=month)
                archives[item["bundle_url"]] = zip_bytes([item])
                return 200, {}, html_page([item])
            result = fixture.collect(transport, source_limits=growth.SourceLimits(max_candidates=1))
            self.assertEqual(result["counts"]["selected_candidates"], 1)
            self.assertEqual(result["counts"]["deferred_candidates"], 2)
            next_result = growth.freeze_sources(fixture.parent_ref, [fixture.discovery_ref, result["discovery"]],
                [fixture.previous_ref, {key: result[key] for key in ("path", "sha256")}],
                fixture.root / "next.json", seed="fixture-growth", fixture_only=True)
            self.assertEqual(next_result["counts"]["selected_candidates"], 3)

    def test_protected_discovered_games_never_trigger_a_zip_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            fixture.index["processed_source_ids"] = [fixture.rows[0]["url"]]
            fixture.refresh()
            def transport(url, *args):
                from urllib.parse import parse_qs, urlsplit
                self.assertFalse(url.endswith(".zip"))
                query = parse_qs(urlsplit(url).query)
                item = row("protected", black=query["user"][0], white="Final", month=int(query["month"][0]))
                return 200, {}, html_page([item])
            result = fixture.collect(transport)
            self.assertTrue(result["cache_exhausted"])
            self.assertEqual(fixture.read(result)["costs"]["requests"], 2)

    def test_malformed_download_is_a_retained_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            def transport(url, *args):
                from urllib.parse import parse_qs, urlsplit
                if url.endswith(".zip"):
                    return 200, {}, b"PK broken archive"
                query = parse_qs(urlsplit(url).query)
                item = row("badzip" + query["user"][0], black=query["user"][0], white="Carol", month=int(query["month"][0]))
                return 200, {}, html_page([item])
            result = fixture.collect(transport)
            data = fixture.read(result)
            self.assertEqual(data["status"], "partial")
            self.assertEqual(len(data["collection_diagnostics"]), 2)
            self.assertTrue(all(problem["reason"] == "BadZipFile" for problem in data["collection_diagnostics"]))
            self.assertEqual(data["counts"]["unavailable_candidates"], 2)

    def test_network_caps_and_retries_keep_all_attempt_costs(self):
        for kind in ("requests", "bytes", "retries"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                fixture = Fixture(tmp)
                if kind == "requests":
                    limits = growth.CollectionLimits(pages=2, requests=1)
                    reply = (404, {}, b"bad")
                elif kind == "bytes":
                    limits = growth.CollectionLimits(pages=2, response_bytes=2)
                    reply = (200, {}, b"long")
                else:
                    limits = growth.CollectionLimits(pages=2)
                    reply = (503, {}, b"busy")
                result = fixture.collect(lambda *args: reply, limits=limits)
                data = fixture.read(result)
                self.assertEqual(data["status"], "partial")
                expected = 6 if kind == "retries" else 1
                self.assertEqual(data["costs"]["requests"], expected)
                self.assertEqual(data["costs"]["response_bytes"], expected * len(reply[2]))
                ledger = (fixture.root / "collection/requests.jsonl").read_text().splitlines()
                self.assertEqual(len(ledger), expected)

    def test_exhausted_query_universe_is_explicit_and_uses_no_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            fixture.exposure["account_months"] = [[handle, f"2025-{month:02d}"]
                for handle in ("alice", "bob") for month in range(1, 9)]
            fixture.refresh()
            result = fixture.collect(lambda *args: self.fail("No requests without safe new pages"))
            self.assertEqual(result["status"], "no_discovery_queries")
            self.assertEqual(result["queries_selected"], 0)
            self.assertEqual(fixture.read(result)["costs"]["requests"], 0)

    def test_authentication_block_is_explicit_and_does_not_open_secret_files_without_a_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            with patch.object(source.Client, "authenticate", side_effect=AssertionError("No credential read")):
                result = fixture.collect(lambda *args: (403, {}, b"denied"))
            data = fixture.read(result)
            self.assertEqual(data["status"], "authentication_blocked")
            self.assertEqual(data["costs"]["requests"], 1)
            self.assertEqual(data["authentication"], "not_requested")

    def test_global_budget_failure_retains_request_ledger_and_accumulated_counters(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            counters = {"source_requests": 11, "source_response_bytes": 7}
            def check():
                if counters["source_requests"] > 11:
                    raise RuntimeError("global-budget-ended")
            with self.assertRaisesRegex(RuntimeError, "global-budget-ended"):
                fixture.collect(lambda *args: (404, {}, b"bad"), counters=counters, check_budget=check)
            self.assertEqual(counters["source_requests"], 12)
            self.assertEqual(counters["source_response_bytes"], 10)
            progress = json.loads((fixture.root / "collection/collection_progress.json").read_text())
            self.assertEqual(progress["status"], "interrupted")
            self.assertEqual(progress["costs"]["requests"], 1)
            self.assertEqual(len((fixture.root / "collection/requests.jsonl").read_text().splitlines()), 1)

    def test_production_transport_override_and_enlarged_network_limits_are_rejected(self):
        for kwargs in ({"requests": 2401}, {"response_bytes": 1024 * growth.MIB + 1},
                       {"wall_seconds": 21601}, {"retries": 3}, {"min_interval_seconds": 4}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                growth.CollectionLimits(**kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            with self.assertRaisesRegex(ValueError, "synthetic fixtures"):
                growth.collect_more(fixture.parent_ref, [fixture.discovery_ref], [fixture.previous_ref],
                    fixture.root / "blocked", seed="fixture-growth", deadline_utc="2026-11-22T00:00:00Z",
                    transport=lambda *args: (200, {}, b""))
            self.assertFalse((fixture.root / "blocked").exists())


if __name__ == "__main__":
    unittest.main()
