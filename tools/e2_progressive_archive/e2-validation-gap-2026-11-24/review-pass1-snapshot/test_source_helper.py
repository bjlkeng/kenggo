"""Small synthetic source fixtures; no live requests, old targets, or models."""

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import html
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import quote_plus
import zipfile


SPEC = importlib.util.spec_from_file_location("validation_source_helper", Path(__file__).with_name("source-helper.py"))
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)
source = helper.source


class Clock:
    def __init__(self):
        self.origin = datetime(2026, 11, 24, tzinfo=timezone.utc)
        self.seconds = self.cpu_seconds = 0

    def now(self):
        return self.origin + timedelta(seconds=self.seconds)

    def clock(self):
        return self.seconds

    def cpu(self):
        return self.cpu_seconds

    def sleep(self, seconds):
        self.seconds += seconds

    def kwargs(self):
        return dict(now=self.now, monotonic=self.clock, cpu=self.cpu, sleep=self.sleep)


def row(name="one", **changes):
    return dict(viewable="Yes", white="Bob [5k]", black="Valid [11k]", setup="19×19",
                start="9/1/25 1:00 PM", type="Ranked", result="W+Res.",
                url=f"https://files.gokgs.com/games/2025/9/1/{name}.sgf",
                page_key="valid-2025-09", bundle_url="https://www.gokgs.com/servlet/archives/en_US/valid-2025-09.zip") | changes


def sgf(item, *, props=None, tail=";B[aa];W[bb];B[cc]"):
    black, br = source.kgs_source.player(item["black"])
    white, wr = source.kgs_source.player(item["white"])
    values = dict(FF="4", GM="1", CA="UTF-8", SZ="19", PB=black, PW=white, BR=br, WR=wr,
                  RU="Japanese", KM="6.5", TM="900", OT="5x30 byo-yomi",
                  DT=source.e1_audit.source_key(item["url"])[0], RE="W+R")
    values.update(props or {})
    return ("(;" + "".join(k + "[" + v + "]" for k, v in values.items()) + tail + ")").encode()


def page(rows):
    if not rows:
        return b"No games"
    return ("<table>" + "".join('<tr><td><a href="' + html.escape(r["url"], quote=True) + '">Yes</a></td>'
            + "".join("<td>" + html.escape(r[key]) + "</td>" for key in
                      ("white", "black", "setup", "start", "type", "result")) + "</tr>" for r in rows)
            + '</table><a href="' + rows[0]["bundle_url"] + '">ZIP</a>').encode()


def archive(rows, bodies):
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w", compression=zipfile.ZIP_DEFLATED) as stream:
        for item, data in zip(rows, bodies):
            stream.writestr(item["url"].split("files.gokgs.com/", 1)[1], data)
    return memory.getvalue()


class Fixture:
    def __init__(self, root, rows=None, bodies=None):
        self.root = Path(root)
        self.rows = [row()] if rows is None else rows
        self.bodies = [sgf(r) for r in self.rows] if bodies is None else bodies
        self.clock = Clock()
        self.calls = []
        self.groups = {key: "kgs-handle:" + key for key in ("valid", "bob", "alice", "final", "libertybot")}
        plan = dict(grouping_sha256=helper.digest(self.groups), train_before="2025-09-01",
                    confirmation_from="2025-11-01", validation_groups=["kgs-handle:valid"],
                    test_groups=["kgs-handle:final"])
        owner = self.save("owner.json", {"excluded_games": []})
        self.owner_ref = owner
        self.policy_ref = self.save("policy.json", {"policy": helper.kgs_admission.POLICY})
        self.artifacts = []
        self.summaries = []
        for label in ("parent", "grown"):
            exposure = {k: [] for k in source.EXPOSURE_FIELDS}
            group_ref = self.save(label + "-groups.json", self.groups)
            exposure_ref = self.save(label + "-exposure.json", exposure)
            split = dict(plan=plan, plan_sha256=helper.digest(plan), fixture_only=True,
                         exposure_sha256=helper.digest(exposure), sealed_source_ids=[],
                         provenance={"manual_review_sha256": owner["sha256"]})
            split_ref = self.save(label + "-split.json", split)
            canonical_ref = {"path": str(self.root / (label + "-canonical-MUST-NOT-READ.parquet")), "sha256": "a" * 64}
            index = dict(canonical_sha256=canonical_ref["sha256"], split_sha256=split_ref["sha256"],
                         exposure=exposure_ref, records=[], rejected=[], processed_source_ids=[])
            index_ref = self.save(label + "-index.json", index)
            artifacts = dict(groups=group_ref, split_manifest=split_ref, base_index=index_ref,
                             exposure=exposure_ref, canonical=canonical_ref)
            self.artifacts.append(artifacts)
            summary = dict(schema="e2_komi_expansion_v1" if label == "parent" else "e2_data_growth_v1", artifacts=artifacts)
            if label == "grown":
                summary["parent_summary"] = self.summaries[0]
            self.summaries.append(self.save(label + ".json", summary))
        self.cache = self.root / "cache"
        self.cache.mkdir()
        query = helper.rank_discovery._query("valid", 9, "validation_development", "11k")
        self.frozen = dict(queries=[query], queries_sha256=source.digest([query]), old_cache=str(self.cache),
                           historical_metadata=[self.save("history.json", {"rows": [row()], "pages": []})])
        frozen_ref = self.save("queries.json", self.frozen)
        self.request = dict(schema=helper.SCHEMA, fixture_only=True, limits=dict(helper.LIMITS),
                            network_limits=asdict(source.Limits(requests=80, response_bytes=128 * helper.MIB,
                                                              wall_seconds=1800, expanded_zip_bytes=256 * helper.MIB)),
                            source_commit=helper.COMMIT, deadline_utc=helper.DEADLINE, code_sha256={},
                            inputs=dict(parent_summary=self.summaries[0], grown_summary=self.summaries[1],
                                        owner_exclusions=owner, source_policy=self.policy_ref, queries=frozen_ref))
        self.request_path = self.root / "request.json"
        self.output = self.root / "output"

    def save(self, name, value):
        path = self.root / name
        path.write_bytes(source.encoded(value) + b"\n")
        return source.reference(path)

    def pin(self):
        self.save("request.json", self.request)
        return source.file_hash(self.request_path)

    def ctx(self):
        return helper.context(self.request_path, self.pin(), fixture=True)

    def transport(self, url, timeout, maximum, **kwargs):
        self.calls.append(url)
        return 200, {"Content-Type": "application/zip" if url.endswith(".zip") else "text/html"}, (
            archive(self.rows, self.bodies) if url.endswith(".zip") else page(self.rows))

    def run(self, **kwargs):
        return helper.run(self.request_path, self.pin(), self.output, fixture=True,
                          transport=kwargs.pop("transport", self.transport), **self.clock.kwargs(), **kwargs)

    def details(self):
        return json.loads((self.output / "source-work/summary.json").read_text())


class ContextTests(unittest.TestCase):
    def test_context_uses_metadata_only_and_checks_all_pins(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            ctx = fixture.ctx()
            self.assertEqual(ctx["groups"], fixture.groups)
            self.assertEqual(len(ctx["queries"]), 1)
            self.assertFalse(Path(fixture.artifacts[0]["canonical"]["path"]).exists())
            Path(fixture.artifacts[1]["base_index"]["path"]).write_text("{}")
            with self.assertRaisesRegex(ValueError, "changed"):
                fixture.ctx()

    def test_enlarged_caps_changed_scope_and_reused_month_are_rejected(self):
        for case in ("cpu", "requests", "html", "query", "month", "cohort", "dates", "policy"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                f = Fixture(tmp)
                if case == "cpu":
                    f.request["limits"]["cpu_seconds"] = 1801
                elif case == "requests":
                    f.request["network_limits"]["requests"] = 81
                elif case == "html":
                    f.request["network_limits"]["html_bytes"] = 2 * helper.MIB + 1
                elif case == "query":
                    q = helper.rank_discovery._query("final", 9, "validation_development", "11k")
                    f.frozen.update(queries=[q], queries_sha256=source.digest([q]))
                    f.request["inputs"]["queries"] = f.save("queries.json", f.frozen)
                elif case == "month":
                    (f.cache / "valid-2025-09.html").write_text("Already queried")
                elif case in ("cohort", "dates"):
                    ref = f.artifacts[1]["split_manifest"]
                    changed = json.loads(Path(ref["path"]).read_text())
                    changed["plan"]["validation_groups" if case == "cohort" else "train_before"] = (
                        ["kgs-handle:bob"] if case == "cohort" else "2025-08-01")
                    changed["plan_sha256"] = helper.digest(changed["plan"])
                    f.artifacts[1]["split_manifest"] = f.save("grown-split.json", changed)
                    summary = json.loads(Path(f.summaries[1]["path"]).read_text())
                    summary["artifacts"] = f.artifacts[1]
                    f.request["inputs"]["grown_summary"] = f.save("grown.json", summary)
                else:
                    f.request["inputs"]["source_policy"] = f.save("policy.json", {"policy": "unapproved"})
                with self.assertRaises(ValueError):
                    f.ctx()

    def test_both_indexes_ledgers_owner_and_sealed_ids_form_barrier(self):
        indexes = [{"records": [{"source_url": "parent", "sgf_sha256": "ps", "audit_sequence_sha256": "pq", "trajectory_sha256": "pt"}],
                    "rejected": [{"source_url": "rejected", "trajectory_sha256": "rt"}]},
                   {"records": [{"source_url": "grown", "sgf_sha256": "gs"}], "processed_source_ids": ["processed"],
                    "study_rejections": [{"source_url": "study", "audit_sequence_sha256": "sq"}]}]
        exposures = [dict(source_ids=["exposed1"], sgf_hashes=["e1s"], sequence_hashes=["e1q"]),
                     dict(source_ids=["exposed2"], sgf_hashes=["e2s"], sequence_hashes=["e2q"])]
        blocked, hashes = helper.history_barrier(indexes, exposures,
            {"excluded_games": [{"source_url": "owner", "sgf_sha256": "os"}]}, ["sealed"])
        self.assertEqual(blocked, {"parent", "rejected", "grown", "processed", "study", "exposed1", "exposed2", "owner", "sealed"})
        self.assertEqual(hashes, {"sgf_sha256": {"ps", "gs", "e1s", "e2s", "os"},
                                  "audit_sequence_sha256": {"pq", "sq", "e1q", "e2q"}, "trajectory_sha256": {"pt", "rt"}})


class AdmissionTests(unittest.TestCase):
    def test_metadata_rejects_other_dates_cohorts_accounts_and_ranks(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp)
            ctx = f.ctx()
            for item in (row(white="Final [5k]"), row(white="Unknown [5k]"), row(black="Alice [11k]"),
                         row(black="Valid [5k]"), row(type="Free"), row(white="libertybot [5k]"),
                         row(url="https://files.gokgs.com/games/2025/11/1/late.sgf"),
                         row(url="https://files.gokgs.com/games/2025/8/1/early.sgf")):
                self.assertIsNotNone(helper.metadata_reason(item, ctx["groups"], ctx["plan"], set()), item)
            self.assertIsNotNone(helper.metadata_reason(row(), ctx["groups"], ctx["plan"], {row()["url"]}))
            self.assertIsNone(helper.metadata_reason(row(), ctx["groups"], ctx["plan"], set()))

    def test_legality_policy_results_and_rank_mismatch_do_not_admit(self):
        cases = [(dict(KM="0.5"), ";B[aa];W[bb]"), (dict(HA="2"), ";B[aa];W[bb]"),
                 (dict(EV="Teaching"), ";B[aa];W[bb]"), (dict(TM="0", OT="none"), ";B[aa];W[bb]"),
                 (dict(BR="10k"), ";B[aa];W[bb]"), (dict(RE="?"), ";B[aa];W[bb]"),
                 ({}, ";B[aa];W[bb];B[aa]")]
        for props, tail in cases:
            with self.subTest(props=props, tail=tail), tempfile.TemporaryDirectory() as tmp:
                f = Fixture(tmp, bodies=[sgf(row(), props=props, tail=tail)])
                result = f.run()
                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["usable_games"], 0)

    def test_all_three_historical_fingerprint_types_reject_fresh_source_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp)
            ctx = f.ctx()
            budget = helper.Budget(helper.LIMITS, helper.DEADLINE)
            data = sgf(row())
            original = helper.candidate_record(data, row(), {"source_url": row()["url"]}, ctx, budget)
            for field in helper.FINGERPRINTS:
                with self.subTest(field=field):
                    ctx["hashes"] = {key: {original[key]} if key == field else set() for key in helper.FINGERPRINTS}
                    unit = helper.candidate_record(data, row(), {"source_url": row()["url"]}, ctx, budget)
                    self.assertEqual(unit["status"], "excluded")
                    self.assertIn("historical", unit["reason"])

    def test_duplicate_copies_and_conflicting_outcomes_are_not_extra_games(self):
        for conflict, expected in ((False, 1), (True, 0)):
            with self.subTest(conflict=conflict), tempfile.TemporaryDirectory() as tmp:
                rows = [row("a"), row("b", result="B+Res." if conflict else "W+Res.")]
                bodies = [sgf(rows[0]), sgf(rows[1], props={"RE": "B+R"} if conflict else {})]
                f = Fixture(tmp, rows, bodies)
                result = f.run()
                self.assertEqual(result["usable_games"], expected)

    def test_rejected_duplicate_poison_spreads_through_component(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp)
            ctx = f.ctx()
            budget = helper.Budget(helper.LIMITS, helper.DEADLINE)
            a = helper.candidate_record(sgf(row("a")), row("a"), {"source_url": row("a")["url"]}, ctx, budget)
            b = helper.candidate_record(sgf(row("b")), row("b"), {"source_url": row("b")["url"]}, ctx, budget)
            a["audit_sequence_sha256"] = a["entry"]["audit_sequence_sha256"] = "poison"
            b["audit_sequence_sha256"] = b["entry"]["audit_sequence_sha256"] = "different"
            rejected = dict(status="excluded", source_url="rejected", audit_sequence_sha256="poison")
            accepted, exclusions = helper.finalize_candidates([a, b, rejected], ctx)
            self.assertEqual(accepted, [])
            self.assertEqual(len(exclusions), 3)


class ExecutionTests(unittest.TestCase):
    def test_urls_freeze_before_sgf_reads_and_target_eight_stops_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [row(f"game-{i:02d}") for i in reversed(range(10))]
            bodies = [sgf(item, tail=f";B[aa];W[bb];B[{chr(99 + i)}c]") for i, item in enumerate(rows)]
            f = Fixture(tmp, rows, bodies)
            f.output.mkdir()
            for name in ("job.json", "stdout.log", "stderr.log"):
                (f.output / name).write_text("")
            original = helper.candidate_record
            observed = []
            def checked(data, item, *args):
                frozen = json.loads((f.output / "source-work/frozen-candidates.json").read_text())
                self.assertEqual([r["url"] for r in frozen["candidates"]], sorted(r["url"] for r in rows))
                observed.append(item["url"])
                return original(data, item, *args)
            with patch.object(helper, "candidate_record", side_effect=checked):
                result = f.run()
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["yield_status"], "sufficient_yield")
            self.assertEqual(result["accepted_games"], 8)
            self.assertEqual(result["usable_games"], 8)
            self.assertEqual(observed, sorted(r["url"] for r in rows)[:8])
            self.assertEqual(f.details()["selection_stop"], "target_games")
            self.assertEqual(source.file_hash(result["source_summary"]["path"]), result["source_summary"]["sha256"])
            self.assertEqual(f.details()["costs"]["gpu_seconds"], 0)

    def test_empty_or_one_game_is_a_completed_negative_result(self):
        for count in (0, 1):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as tmp:
                f = Fixture(tmp, [] if count == 0 else [row()])
                result = f.run()
                self.assertEqual((result["status"], result["yield_status"], result["usable_games"]),
                                 ("completed", "insufficient_yield", count))

    def test_failed_request_is_partial_and_cannot_retry_another_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp)
            f.request["network_limits"]["requests"] = 1
            result = f.run()
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["usable_games"], 0)
            self.assertEqual(result["reason"], "request_or_response_limit")
            self.assertTrue((f.output / "source-work/response-001.json").exists())
            f.output = f.root / "second-output"
            with self.assertRaises(FileExistsError):
                f.run()
            self.assertEqual(len(f.calls), 1)

    def test_metadata_failure_stays_explicit_with_two_usable_games(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [row("a"), row("b")]
            f = Fixture(tmp, rows, [sgf(rows[0]), sgf(rows[1], tail=";B[aa];W[bb];B[dc]")])
            extra = helper.rank_discovery._query("valid", 10, "validation_development", "11k")
            f.frozen["queries"].append(extra)
            f.frozen["queries_sha256"] = source.digest(f.frozen["queries"])
            f.request["inputs"]["queries"] = f.save("queries.json", f.frozen)
            def transport(url, *args, **kwargs):
                if "month=10" in url:
                    return 404, {}, b"not found"
                return f.transport(url, *args, **kwargs)
            result = f.run(transport=transport)
            self.assertEqual((result["status"], result["yield_status"], result["usable_games"]),
                             ("partial", "sufficient_yield", 2))
            self.assertEqual(result["reason"], "metadata_requests_failed")

    def test_deadline_wall_cpu_zip_and_sgf_caps(self):
        for case in ("deadline", "wall", "cpu", "zip", "sgf"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                f = Fixture(tmp)
                if case == "deadline":
                    f.clock.origin = source.deadline(helper.DEADLINE)
                elif case in ("wall", "cpu"):
                    def exhausted(url, timeout, maximum, **kwargs):
                        result = f.transport(url, timeout, maximum, **kwargs)
                        if case == "wall":
                            f.clock.seconds = helper.LIMITS["wall_seconds"]
                        else:
                            f.clock.cpu_seconds = helper.LIMITS["cpu_seconds"]
                        return result
                else:
                    f.request["limits"]["expanded_zip_bytes" if case == "zip" else "selected_sgf_bytes"] = 1
                result = f.run(transport=exhausted) if case in ("wall", "cpu") else f.run()
                self.assertEqual(result["status"], "partial")
                self.assertEqual(result["usable_games"], 0)
                self.assertEqual(result["yield_status"], "insufficient_yield")

    def test_unexpected_failure_retains_redacted_summary_and_disallows_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp)
            with patch.object(helper, "candidate_record", side_effect=RuntimeError("untrusted response password")):
                result = f.run()
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["reason"], "RuntimeError")
            self.assertNotIn("password", (f.output / "summary.json").read_text())
            self.assertTrue((f.root / "source-execution.json").exists())

    def test_response_and_output_caps_preserve_failure_summary(self):
        for case in ("response", "output"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                f = Fixture(tmp)
                if case == "response":
                    f.request["network_limits"]["response_bytes"] = 1
                else:
                    f.request["limits"]["output_bytes"] = helper.FINAL_RESERVE + 1000
                result = f.run()
                self.assertEqual(result["status"], "partial")
                self.assertEqual(result["usable_games"], 0)
                self.assertGreater(result["costs"]["response_bytes"], 0)
                self.assertTrue((f.output / "summary.json").exists())


class RedactionTests(unittest.TestCase):
    def test_login_body_headers_and_response_errors_do_not_leak_fixture_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp)
            username, password = "private-user-marker", "private & password-marker"
            secret = f.root / "synthetic-credentials.json"
            secret.write_text(json.dumps({"username": username, "password": password}))
            secret.chmod(0o600)
            login = b'<form action="login.jsp" method="post"><input name="user"><input name="password"></form>'
            calls = []
            def transport(url, timeout, maximum, **kwargs):
                calls.append(url)
                if len(calls) == 1:
                    return 200, {}, login
                if len(calls) == 2:
                    return 200, {"Set-Cookie": password}, (username + password).encode()
                if url.endswith(".zip"):
                    return f.transport(url, timeout, maximum, **kwargs)
                body = page(f.rows) + ("You are currently logged in as " + username + ". "
                                       + html.escape(password) + " " + quote_plus(password)).encode()
                return 200, {"Set-Cookie": password, "ETag": password}, body
            result = f.run(transport=transport, credentials_file=secret)
            self.assertEqual(result["status"], "completed")
            files = b"".join(p.read_bytes() for p in f.output.rglob("*") if p.is_file())
            for value in (username, password, html.escape(password), quote_plus(password)):
                self.assertNotIn(value.encode(), files)
            self.assertIn(b"authentication_response_may_echo_credentials", files)

    def test_interrupted_response_message_is_sanitized_before_client_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp)
            secret = "untrusted-response-secret"
            def interrupted(*args, **kwargs):
                raise source.InterruptedResponse(200, {"Set-Cookie": secret}, b"partial", secret)
            f.request["network_limits"]["retries"] = 0
            result = f.run(transport=interrupted)
            self.assertEqual(result["status"], "partial")
            files = b"".join(p.read_bytes() for p in f.output.rglob("*") if p.is_file())
            self.assertNotIn(secret.encode(), files)

    def test_redirect_to_unfrozen_account_never_reaches_transport(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp)
            calls = []
            def redirect(url, *args, **kwargs):
                calls.append(url)
                return 302, {"Location": "https://www.gokgs.com/gameArchives.jsp?user=final&year=2025&month=9&oldAccounts=y"}, b""
            result = f.run(transport=redirect)
            self.assertEqual(result["status"], "partial")
            self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
