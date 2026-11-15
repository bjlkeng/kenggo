from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit
import zipfile

from kenggo import expansion_source as source
from kenggo import rank_continuation as continuation
from test_expansion_source import Fixture as SourceFixture, html_page, zip_bytes
from test_cached_coverage_audit import row


def save(path, value):
    path.write_bytes(source.encoded(value))
    return source.reference(path)


def read_result(reference):
    return source.read_ref({key: reference[key] for key in ("path", "sha256")})


def weak_row(name, month, account, opponent="WeakPeer", rank="15k"):
    item = row(name, month, black=account, white=opponent,
        page_key=f"{account.lower()}-2025-{month:02d}",
        bundle_url=f"https://www.gokgs.com/servlet/archives/en_US/{account.lower()}-all-2025-{month}.zip")
    item.update(black=f"{account} [{rank}]", white=f"{opponent} [{rank}]")
    return item


class Fixture:
    def __init__(self, directory, *, pages=3, candidates=5, request_limit=2400,
                 fail_first_zip=False, redirect_metadata=False):
        self.root = Path(directory)
        self.first = SourceFixture(directory)
        self.clock = self.first.clock
        probe = {"requests": 4, "response_bytes": 4, "cpu_charged_seconds": 120,
                 "records": [{"bytes": 1} for _ in range(4)]}
        self.first.request["access_probe"] = save(self.root / "probe.json", probe)
        self.first.request["limits"].update(requests=request_limit - 4,
            response_bytes=1024 * source.MIB - 4, cpu_seconds=780,
            max_population_development_candidates=5000, max_validation_development_candidates=5000)
        self.seed_rows = [weak_row(f"seed{rank}", 2, "Alice", opponent, rank)
                          for opponent, rank in (("NewWeak", "15k"), ("OtherWeak", "14k"), ("ThirdWeak", "13k"))]
        prior = source.read_ref(self.first.inputs["prior_exposure"])
        prior["groups"].remove("kgs-handle:final")
        self.first.inputs["prior_exposure"] = save(self.root / "prior_exposure.json", prior)

        def transport(url, *args):
            if url.endswith(".zip"):
                if fail_first_zip:
                    return 503, {}, b"busy"
                return 200, {}, zip_bytes(self.seed_rows)
            if redirect_metadata and urlsplit(url).path == "/gameArchives.jsp":
                return 302, {"Location": url.replace("/gameArchives.jsp", "/archives.jsp")}, b""
            params = parse_qs(urlsplit(url).query)
            if params.get("user") == ["Alice"] and params.get("month") == ["2"]:
                return 200, {}, b"".join(html_page(item) for item in self.seed_rows)
            return 200, {}, b"No games"

        self.previous = self.first.run(transport)
        assert self.previous["status"] == ("partial" if fail_first_zip else "completed"), self.previous
        self.request = {"schema": continuation.REQUEST_SCHEMA,
            "scope_decision": "Synthetic continuation; amend screening only, preserve all shared budgets.",
            "previous": {"sources": source.reference(self.first.output / "sources.json"),
                "requests": source.reference(self.first.output / "requests.jsonl"),
                "costs": source.reference(self.first.output / "costs.json"),
                "execution": source.reference(self.first.queries_path.with_name("queries.json.execution.json"))},
            "selection": {"pages": pages, "max_train_candidates": candidates,
                          "focus_ranks": list(continuation.FOCUS_RANKS)},
            "source_revision": {"code_sha256": {}}}
        self.request_path = self.root / "continuation.json"
        self.queries_path = self.root / "continuation-queries.json"
        self.output = self.root / "continued"

    def freeze(self):
        save(self.request_path, self.request)
        return continuation.freeze_queries(self.request_path, source.file_hash(self.request_path),
                                           self.queries_path, fixture_only=True)

    def acquire(self, transport, **kwargs):
        return continuation.acquire(self.queries_path, source.file_hash(self.queries_path), self.output,
            fixture_only=True, transport=transport, **self.clock.kwargs(), **kwargs)

    def transport(self, url, *args, **kwargs):
        queries = json.loads(self.queries_path.read_text())["queries"]
        items = [weak_row(f"new{query['user']}{query['month']}", query["month"], query["user"], rank=query["discovery_rank"])
                 for query in queries]
        if url.endswith(".zip"):
            return 200, {}, zip_bytes([item for item in items if item["bundle_url"] == url])
        query = next(query for query in queries if query["url"] == url)
        item = next(item for item in items if item["page_key"] == query["key"])
        return 200, {}, html_page(item)


class ContinuationTests(unittest.TestCase):
    def test_previous_redirected_metadata_keeps_its_request_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory, redirect_metadata=True)
            frozen = read_result(fixture.freeze())
            self.assertEqual(len(frozen["queries"]), 3)
            self.assertEqual(frozen["prior_costs"]["requests"],
                             len(continuation._ledger(fixture.request["previous"]["requests"])) + 4)
            ledger_path = Path(fixture.request["previous"]["requests"]["path"])
            ledger = continuation._ledger(fixture.request["previous"]["requests"])
            response = next(item for item in ledger
                            if item.get("status") == 200 and "/archives.jsp" in item["url"])
            response["url"] += "&oldAccounts=changed"
            ledger_path.write_text("".join(json.dumps(item) + "\n" for item in ledger))
            fixture.request["previous"]["requests"] = source.reference(ledger_path)
            with self.assertRaisesRegex(ValueError, "request provenance"):
                fixture.freeze()

    def test_exhausted_prior_bundle_stays_unavailable_while_new_accounts_are_collected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory, fail_first_zip=True)
            old_pins = {name: source.file_hash(ref["path"]) for name, ref in fixture.request["previous"].items()}
            old_frozen = source.read_ref(fixture.previous["provenance"]["frozen_candidates"])
            old_ids = {item["archive_row"]["url"] for item in old_frozen["candidates"]}
            old_accessible = {item["archive_row"]["url"] for item in fixture.previous["candidates"]}
            missing = old_ids - old_accessible
            self.assertTrue(missing)
            fixture.freeze()
            result = read_result(fixture.acquire(fixture.transport))
            combined = source.read_ref(result["provenance"]["frozen_candidates"])
            self.assertLessEqual(old_ids, {item["archive_row"]["url"] for item in combined["candidates"]})
            self.assertFalse(missing & {item["archive_row"]["url"] for item in result["candidates"]})
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["counts"]["previous_unavailable_candidates"], len(missing))
            self.assertEqual(result["counts"]["additional_accessible_candidates"], 3)
            self.assertTrue(all(item in result["candidates"] for item in fixture.previous["candidates"]))
            self.assertLessEqual(missing, set(result["exposure"]["source_ids"]))
            self.assertEqual({name: source.file_hash(ref["path"]) for name, ref in fixture.request["previous"].items()}, old_pins)

    def test_queries_preserve_cohorts_skip_both_prior_rounds_and_reproduce(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            result = fixture.freeze()
            frozen = read_result(result)
            prior_queries = json.loads(fixture.first.queries_path.read_text())["queries"]
            prior_pairs = {(query["user"].lower(), query["month"]) for query in prior_queries}
            self.assertEqual(len(frozen["queries"]), 3)
            self.assertEqual(Counter(query["discovery_rank"] for query in frozen["queries"]),
                             {"15k": 1, "14k": 1, "13k": 1})
            for query in frozen["queries"]:
                self.assertNotIn((query["user"], query["month"]), prior_pairs)
                self.assertEqual(query["role"], "train")
                self.assertEqual(query["month"], 2)
                self.assertIn(query["user"], {"newweak", "otherweak", "thirdweak"})
            self.assertEqual(frozen, continuation._query_contents(frozen["request"], True)[0])

    def test_previous_final_costs_define_residuals_and_shared_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            costs_path = fixture.first.output / "costs.json"
            costs = json.loads(costs_path.read_text())
            costs.update(cpu_seconds=11.25, cpu_charged_seconds=131.25, wall_seconds=500)
            fixture.request["previous"]["costs"] = save(costs_path, costs)
            fixture.freeze()
            frozen = json.loads(fixture.queries_path.read_text())
            self.assertEqual(frozen["limits"]["requests"], 2400 - costs["requests"])
            self.assertEqual(frozen["limits"]["response_bytes"], 1024 * source.MIB - costs["response_bytes"])
            self.assertEqual(frozen["limits"]["cpu_seconds"], 768)
            self.assertEqual(frozen["limits"]["wall_seconds"], 21100)
            self.assertEqual(frozen["limits"]["expanded_zip_bytes"],
                             4096 * source.MIB - fixture.previous["counts"]["expanded_zip_bytes"])
            fixture.clock.seconds += 1800
            self.assertEqual(frozen, continuation._query_contents(frozen["request"], True)[0])
            self.assertEqual(source.deadline(frozen["deadline_utc"]) - fixture.clock.origin,
                             source.deadline("2026-11-14T06:00:00+00:00") - fixture.clock.origin)

    def test_partial_previous_run_and_cost_ledger_mismatch_are_rejected(self):
        for change in ("partial", "requests", "bytes", "probe", "ledger", "marker"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                fixture = Fixture(directory)
                if change == "partial":
                    fixture.previous["status"] = "partial"
                    fixture.request["previous"]["sources"] = save(fixture.first.output / "sources.json", fixture.previous)
                elif change == "ledger":
                    path = fixture.first.output / "requests.jsonl"
                    path.write_text(path.read_text().replace('"sequence": 1,', '"sequence": 999,', 1))
                    fixture.request["previous"]["requests"] = source.reference(path)
                elif change == "marker":
                    path = Path(fixture.request["previous"]["execution"]["path"])
                    data = json.loads(path.read_text()); data["queries_sha256"] = "a" * 64
                    fixture.request["previous"]["execution"] = save(path, data)
                else:
                    path = fixture.first.output / "costs.json"
                    data = json.loads(path.read_text())
                    data[{"requests": "requests", "bytes": "response_bytes", "probe": "cpu_charged_seconds"}[change]] += 1
                    fixture.request["previous"]["costs"] = save(path, data)
                with self.assertRaises(ValueError):
                    fixture.freeze()

    def test_stale_code_pin_and_policy_enlargement_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            fixture.request["source_revision"]["code_sha256"] = {continuation.CODE_PATH: "0" * 64}
            save(fixture.request_path, fixture.request)
            with self.assertRaisesRegex(ValueError, "code differs"):
                continuation.freeze_queries(fixture.request_path, source.file_hash(fixture.request_path), fixture.queries_path)
        for change in ("pages", "candidates", "ranks"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                fixture = Fixture(directory)
                if change == "pages": fixture.request["selection"]["pages"] = 201
                if change == "candidates": fixture.request["selection"]["max_train_candidates"] = 5001
                if change == "ranks": fixture.request["selection"]["focus_ranks"].append("12k")
                with self.assertRaisesRegex(ValueError, "at most 200"):
                    fixture.freeze()

    def test_candidates_are_training_weak_rank_only_bounded_and_keep_all_prior_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory, candidates=2)
            fixture.freeze()
            frozen = json.loads(fixture.queries_path.read_text())
            _, context = continuation._query_contents(frozen["request"], True)
            samples = [weak_row(f"extra{i}", 3, f"New{i}") for i in range(6)]
            samples += [weak_row("valid", 3, "Valid"), weak_row("final", 3, "Final"),
                        weak_row("bot", 3, "libertybot"), weak_row("later", 9, "Later"),
                        weak_row("strong", 3, "Strong", rank="4k")]
            result = continuation._freeze_candidates(context, samples, frozen["request"])
            self.assertEqual(len(result["additional_candidates"]), 2)
            self.assertTrue(all(item["archive_row"]["black"].startswith("New") for item in result["additional_candidates"]))
            self.assertLessEqual({item["archive_row"]["url"] for item in fixture.previous["candidates"]},
                                 {item["archive_row"]["url"] for item in result["candidates"]})
            self.assertEqual(result, continuation._freeze_candidates(context, list(reversed(samples)), frozen["request"]))

    def test_complete_round_freezes_before_zip_access_and_preserves_previous_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            fixture.freeze()
            old_pins = {name: source.file_hash(ref["path"]) for name, ref in fixture.request["previous"].items()}
            calls = []
            prior_time = fixture.clock.seconds
            queries = json.loads(fixture.queries_path.read_text())["queries"]
            responses = {query["url"]: fixture.transport(query["url"]) for query in queries}
            for query in queries:
                body = responses[query["url"]][2]
                url = source.parse_page(body, query)[0]["bundle_url"]
                responses[url] = fixture.transport(url)
            def pinned_transport(url, *args):
                calls.append((url, fixture.clock.seconds))
                if url.endswith(".zip"):
                    self.assertTrue((fixture.output / "frozen_candidates.json").is_file())
                return responses[url]
            with patch.object(zipfile.ZipFile, "read", side_effect=AssertionError("SGF contents must stay unopened")), \
                    patch.object(zipfile.ZipFile, "open", side_effect=AssertionError("No ZIP member streams")):
                result_ref = fixture.acquire(pinned_transport)
            result = read_result(result_ref)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["counts"]["additional_accessible_candidates"], 3)
            self.assertGreaterEqual(calls[0][1] - prior_time, 5)
            self.assertEqual(result["costs"]["requests"], fixture.previous["costs"]["requests"] + 6)
            self.assertEqual(result["round_costs"]["requests"], 6)
            self.assertGreater(result["counts"]["expanded_zip_bytes"], fixture.previous["counts"]["expanded_zip_bytes"])
            self.assertEqual({name: source.file_hash(ref["path"]) for name, ref in fixture.request["previous"].items()}, old_pins)
            for item in fixture.previous["candidates"]:
                self.assertIn(item, result["candidates"])
            for field in source.EXPOSURE_FIELDS:
                normalize = lambda values: {tuple(value) if isinstance(value, list) else value for value in values}
                self.assertLessEqual(normalize(fixture.previous["exposure"][field]), normalize(result["exposure"][field]))
            with self.assertRaises(FileExistsError):
                continuation.acquire(fixture.queries_path, source.file_hash(fixture.queries_path), fixture.root / "again",
                    fixture_only=True, transport=pinned_transport, **fixture.clock.kwargs())
            self.assertEqual(len(calls), 6)

    def test_failure_is_partial_without_replacement_and_keeps_exposure(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            fixture.freeze()
            queries = json.loads(fixture.queries_path.read_text())["queries"]
            calls = []
            def transport(url, *args):
                calls.append(url)
                if url == queries[0]["url"]:
                    return 404, {}, b"absent"
                return fixture.transport(url, *args)
            result = read_result(fixture.acquire(transport))
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["counts"]["additional_accessible_candidates"], 2)
            self.assertEqual([url for url in calls if not url.endswith(".zip")], [query["url"] for query in queries])
            self.assertIn([queries[0]["user"], f"2025-{queries[0]['month']:02d}"], result["exposure"]["account_months"])
            self.assertTrue(all(item in result["candidates"] for item in fixture.previous["candidates"]))

    def test_expired_shared_deadline_stops_before_network_without_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            fixture.freeze()
            fixture.clock.seconds = 21601
            calls = []
            result = read_result(fixture.acquire(lambda *args: calls.append(args)))
            self.assertEqual(calls, [])
            self.assertEqual(result["stop_reason"], "absolute_deadline")
            self.assertEqual(result["status"], "partial")
            self.assertEqual(len(result["candidates"]), len(fixture.previous["candidates"]))

    def test_retries_exhaust_the_shared_request_cap_without_losing_the_first_round(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory, pages=2, request_limit=23)
            fixture.freeze()
            seen = Counter()
            def transport(url, *args):
                seen[url] += 1
                if seen[url] == 1:
                    return 503, {}, b"busy"
                return fixture.transport(url, *args)
            result = read_result(fixture.acquire(transport))
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["stop_reason"], "request_or_response_limit")
            self.assertEqual(result["costs"]["requests"], 23)
            self.assertEqual(result["round_costs"]["requests"], 6)
            self.assertEqual(result["counts"]["additional_frozen_candidates"], 2)
            self.assertEqual(result["counts"]["additional_accessible_candidates"], 1)
            self.assertTrue(all(item in result["candidates"] for item in fixture.previous["candidates"]))
            self.assertTrue((fixture.output / "costs.json").is_file())

    def test_login_secret_is_not_copied_to_continuation_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory, pages=1)
            fixture.freeze()
            secret = fixture.root / "private.json"
            secret.write_text(json.dumps({"username": "continuation-user-secret", "password": "continuation-password-secret"}))
            secret.chmod(0o600)
            calls = []
            def transport(url, *args, **kwargs):
                calls.append(url)
                if len(calls) == 1:
                    return 200, {}, b'<form action="login.jsp" method="post"><input name="user"><input name="password"></form>'
                if url.endswith("login.jsp"):
                    self.assertIn(b"continuation-password-secret", kwargs["data"])
                    return 200, {}, b"logged in"
                return fixture.transport(url, *args)
            result = read_result(fixture.acquire(transport, credentials_file=secret))
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["round_costs"]["requests"], 4)
            for path in fixture.output.iterdir():
                if path.suffix != ".zip":
                    self.assertNotIn(b"continuation-password-secret", path.read_bytes())
                    self.assertNotIn(b"continuation-user-secret", path.read_bytes())


if __name__ == "__main__":
    unittest.main()
