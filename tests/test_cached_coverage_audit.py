import html
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

from kenggo import cached_coverage_audit as audit


DRAFT = Path(__file__).resolve().parents[1] / "runs"
DRAFT.mkdir(exist_ok=True)


def row(name="one", month=1, black="Alice", white="Bob", **changes):
    key = f"fixture-2025-{month}"
    return dict(viewable="Yes", white=white + " [5k]", black=black + " [5k]", setup="19×19",
        start=f"{month}/1/25 1:00 PM", type="Ranked", result="W+Res.",
        url=f"https://files.gokgs.com/games/2025/{month}/1/{name}.sgf", page_key=key,
        bundle_url=f"https://www.gokgs.com/servlet/archives/en_US/{key}.zip") | changes


def sgf_bytes(item, *, replace=None, tail=";B[aa]BL[898];W[bb]WL[897];B[];W[]"):
    black, rank_b = audit.kgs_source.player(item["black"])
    white, rank_w = audit.kgs_source.player(item["white"])
    properties = {"FF": "4", "GM": "1", "CA": "UTF-8", "SZ": "19", "PB": black, "PW": white,
        "BR": rank_b, "WR": rank_w, "RU": "Japanese", "KM": "6.5", "TM": "900", "OT": "5x30 byo-yomi",
        "DT": audit.e1_audit.source_key(item["url"])[0], "RE": "W+R", "C": "private-comment-marker"}
    properties.update(replace or {})
    return ("(;" + "".join(key + "[" + value + "]" for key, value in properties.items()) + tail + ")").encode()


def member(item):
    return item["url"].split("files.gokgs.com/", 1)[1]


class Fixture:
    def __init__(self, parent, rows=None, *, original=(), members=None, exclusions=(), limits=None):
        self.root = Path(parent)
        self.rows = rows or [row()]
        self.cache = self.root / "cache"
        self.cache.mkdir()
        page_requests, zip_requests = [], []
        by_page = {}
        for item in self.rows:
            by_page.setdefault(item["page_key"], []).append(item)
        for key, items in by_page.items():
            plain = [{k: v for k, v in item.items() if k not in ("page_key", "bundle_url")} for item in items]
            body = "<table>" + "".join("<tr><td><a href=\"" + html.escape(item["url"], quote=True) + "\">Yes</a></td>"
                + "".join("<td>" + html.escape(str(item[field])) + "</td>" for field in
                          ("white", "black", "setup", "start", "type", "result")) + "</tr>" for item in items)
            body += "</table><a href=\"" + items[0]["bundle_url"] + "\">ZIP</a>"
            html_path = self.cache / (key + ".html")
            html_path.write_text(body)
            page_url = f"https://www.gokgs.com/gameArchives.jsp?fixture={key}"
            raw_hash = audit.sha256(("raw response " + body).encode()).hexdigest()
            self.write(self.cache / (key + ".json"), {"url": page_url, "body_sha256": raw_hash,
                "public_html_sha256": audit.file_hash(html_path), "bundle_url": items[0]["bundle_url"],
                "rows": plain, "cache_source": "earlier-attempt"})
            page_requests.append({"method": "GET", "status": 200, "url": page_url, "sha256": raw_hash,
                                  "bytes": len(body.encode()) + 13})
            archive_path = self.cache / (key + ".zip")
            entries = [(member(item), sgf_bytes(item)) for item in items] if members is None else members[key]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                    for name, data in entries:
                        archive.writestr(name, data)
            zip_requests.append({"method": "GET", "status": 200, "url": items[0]["bundle_url"],
                "sha256": audit.file_hash(archive_path), "bytes": archive_path.stat().st_size})
        self.ledgers = []
        for name, requests in (("earlier-requests", page_requests), ("current-requests", zip_requests)):
            path = self.root / (name + ".jsonl")
            path.write_text("".join(json.dumps(item) + "\n" for item in requests))
            self.ledgers.append(path)
        handles = {"Alice", "Bob", "Carol", "Valid", "Final", "libertybot", "gnugo2"}
        for item in self.rows:
            handles.update(audit.kgs_source.player(item[side])[0] for side in ("black", "white"))
        groups = audit.kgs_eligibility.handle_groups(handles)
        objects = {
            "frame": {"rows": self.rows}, "original_sample": {"candidates": list(original)},
            "final_quarantine": {"test_groups": ["kgs-handle:final"], "records": [{"payload": "not for inspection"}]},
            "source_policy": {"policy": audit.kgs_admission.POLICY},
            "owner_exclusions": {"excluded_games": list(exclusions)},
            "split_config": {"grouping_sha256": audit.kgs_eligibility.grouping_digest(groups),
                "train_before": "2025-09-01", "confirmation_from": "2025-11-01",
                "validation_groups": ["kgs-handle:valid"], "test_groups": ["kgs-handle:final"]},
        }
        self.inputs = {}
        for name, value in objects.items():
            self.inputs[name] = self.root / (name + ".json")
            self.write(self.inputs[name], value)
        self.inputs["environment"] = self.root / "environment.lock"
        self.inputs["environment"].write_text("synthetic dependency lock\n")
        self.request = {"schema": audit.REQUEST_VERSION, "inputs": {}, "request_ledgers": [],
            "cache_directory": str(self.cache), "base_groups": groups,
            "extension_scope_decision": "Synthetic offline source-screening fixture only.",
            "limits": audit.asdict(audit.Limits()) | (limits or {})}
        self.request_path, self.frozen_path = self.root / "request.json", self.root / "frozen.json"
        self.output = self.root / "worker"
        self.output.mkdir()
        self.pin_request()

    @staticmethod
    def write(path, value):
        path.write_bytes(audit.encoded(value) + b"\n")

    def pin_request(self):
        self.request["inputs"] = {key: audit.ref(path) for key, path in self.inputs.items()}
        self.request["request_ledgers"] = [audit.ref(path) for path in self.ledgers]
        self.write(self.request_path, self.request)
        return audit.file_hash(self.request_path)

    def freeze(self):
        self.frozen = audit.freeze_inputs(self.request_path, self.pin_request(), self.frozen_path, fixture_only=True)
        return json.loads(self.frozen_path.read_text())

    def run(self):
        self.freeze()
        return audit.run_audit(self.frozen_path, self.frozen["sha256"], self.output, fixture_only=True)

    def records(self):
        path = self.output / "records.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


class OfflineCoverageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="synthetic-", dir=DRAFT)
        self.addCleanup(self.temporary.cleanup)

    def fixture(self, *args, **kwargs):
        return Fixture(self.temporary.name, *args, **kwargs)

    def test_freeze_uses_metadata_without_opening_members_and_preserves_roles(self):
        selected = [row("train"), row("population", 9), row("validation", 9, white="Valid")]
        excluded = [row("prefix", white="Valid"), row("confirmation", 11), row("final", white="Final"),
                    row("bot", black="LibertyBot"), row("namedbot", white="Robot"), row("original")]
        fixture = self.fixture(selected + excluded, original=[excluded[-1]])
        with patch.object(audit.zipfile, "ZipFile", side_effect=AssertionError("no ZIP opening during freeze")):
            frozen = fixture.freeze()
        self.assertEqual([item["archive_row"]["url"] for item in frozen["candidates"]],
                         sorted(item["url"] for item in selected))
        self.assertEqual(frozen["candidate_counts"], dict.fromkeys(audit.ROLES, 1))
        self.assertEqual(frozen["candidate_ids_sha256"], audit.digest(sorted(item["url"] for item in selected)))
        self.assertEqual(frozen["groups"]["valid"], "kgs-handle:valid")

    def test_audit_preserves_masks_and_reports_provisional_explicit_action_coverage(self):
        fixture = self.fixture([row("one"), row("two", 9, white="Valid")])
        result = fixture.run()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["source_admitted_records"], 2)
        self.assertEqual(result["source_admitted"]["action_clock_counts"]["explicit_passes"], 4)
        self.assertEqual(result["source_admitted_by_provisional_role"]["validation_development"]["games"], 1)
        self.assertEqual(result["network_requests"], 0)
        self.assertFalse(result["deduplicated"])
        self.assertEqual(result["rules_replay_validated_games"], 0)
        self.assertEqual(result["promoted_profiles"], [])
        for record in fixture.records():
            self.assertFalse(record["training_ready"])
            self.assertFalse(record["identity_valid"])
            self.assertFalse(record["duration_targets_valid"])
            self.assertFalse(record["dynamic_clock_inputs_valid"])
            self.assertEqual(record["screening"]["black_human_status"], "presumed_human")
            self.assertEqual(record["screening"]["ordinary_status"], "presumed_ordinary_ranked")
            self.assertEqual(record["screening"]["source_policy_decision_sha256"], audit.kgs_admission.DECISION_SHA256)
            self.assertNotIn("training_eligible", record["screening"])
            self.assertNotIn("private-comment-marker", json.dumps(record))
            self.assertNotIn("B[aa]", json.dumps(record))
        with self.assertRaisesRegex(ValueError, "unused"):
            audit.run_audit(fixture.frozen_path, fixture.frozen["sha256"], fixture.output, fixture_only=True)

    def test_duplicate_and_missing_members_are_retained_without_substitution(self):
        one, two, three = row("one"), row("two"), row("three")
        fixture = self.fixture([one, two, three], members={one["page_key"]:
            [(member(one), sgf_bytes(one)), ("other/" + member(one), sgf_bytes(one)), (member(three), sgf_bytes(three))]})
        result = fixture.run()
        self.assertEqual(result["expected_records"], 3)
        self.assertEqual(result["recorded_records"], 3)
        self.assertEqual(result["source_admitted_records"], 1)
        failures = {record["source_url"]: record for record in fixture.records() if record["retrieval_status"] == "failed"}
        self.assertEqual(failures[one["url"]]["member_matches"], 2)
        self.assertEqual(failures[two["url"]]["member_matches"], 0)
        self.assertNotIn("sgf_sha256", failures[one["url"]])

    def test_header_mismatch_blocks_diagnostics_even_with_corrupt_move_tail(self):
        one, two, three = row("identity"), row("rank"), row("date")
        entries = [(member(one), sgf_bytes(one, replace={"PB": "Final"}, tail=";B[invalid")),
                   (member(two), sgf_bytes(two, replace={"WR": "1d"})),
                   (member(three), sgf_bytes(three, replace={"DT": "2025-11-01"}))]
        fixture = self.fixture([one, two, three], members={one["page_key"]: entries})
        with patch.object(audit.sgf_audit, "inspect_sgf", side_effect=AssertionError("must not inspect actions")):
            result = fixture.run()
        self.assertEqual(result["parse_status_counts"], {"header_rejected": 3})
        self.assertEqual(result["source_admitted_records"], 0)
        for record in fixture.records():
            self.assertNotIn("metadata", record)
            self.assertNotIn("counters", record)
            self.assertNotIn("result_kind", record)

    def test_header_guard_handles_escaped_comment_delimiters_and_final_group(self):
        item = row()
        data = sgf_bytes(item, replace={"C": r"escaped \] ; ( ) text"})
        groups = audit.kgs_eligibility.handle_groups(("Alice", "Bob", "Final"))
        audit.header_check(data, item, groups, {"kgs-handle:final"})
        item = row(white="Final")
        with self.assertRaisesRegex(ValueError, "reserved_final_participant"):
            audit.header_check(sgf_bytes(item), item, groups, {"kgs-handle:final"})

    def test_owner_excluded_bytes_are_rejected_under_a_different_source_id(self):
        item = row()
        raw = sgf_bytes(item)
        fixture = self.fixture([item], exclusions=[{"source_url": "https://files.gokgs.com/games/2025/1/1/old.sgf",
            "sgf_sha256": audit.sha256(raw).hexdigest()}])
        with patch.object(audit.sgf_audit, "inspect_sgf", side_effect=AssertionError("owner exclusion must run first")):
            result = fixture.run()
        self.assertEqual(result["source_admitted_records"], 0)
        self.assertEqual(fixture.records()[0]["exclusions"], ["owner_excluded_sgf_bytes"])
        self.assertNotIn("metadata", fixture.records()[0])

    def test_changed_zip_after_freeze_stops_before_member_inspection(self):
        fixture = self.fixture()
        fixture.freeze()
        path = fixture.cache / (fixture.rows[0]["page_key"] + ".zip")
        with path.open("ab") as stream:
            stream.write(b"changed")
        with patch.object(audit.zipfile, "ZipFile", side_effect=AssertionError("provenance before ZIP opening")):
            with self.assertRaisesRegex(ValueError, "provenance"):
                audit.run_audit(fixture.frozen_path, fixture.frozen["sha256"], fixture.output, fixture_only=True)

    def test_retained_earlier_page_ledger_is_required(self):
        fixture = self.fixture()
        fixture.ledgers = fixture.ledgers[1:]
        with self.assertRaisesRegex(ValueError, "response provenance"):
            fixture.freeze()

    def test_corrupt_sgf_is_retained_without_raw_parser_error_or_comment(self):
        item = row()
        fixture = self.fixture([item], members={item["page_key"]:
            [(member(item), sgf_bytes(item, tail=";B[invalid"))]})
        result = fixture.run()
        self.assertEqual(result["parse_status_counts"], {"corrupt": 1})
        self.assertEqual(result["source_admitted_records"], 0)
        record = fixture.records()[0]
        self.assertEqual(record["error"], "corrupt_or_unsupported_sgf")
        self.assertNotIn("private-comment-marker", json.dumps(record))
        self.assertNotIn("counters", record)

    def test_invalid_zip_is_a_retained_failure_for_each_frozen_id(self):
        fixture = self.fixture([row("one"), row("two")])
        archive = fixture.cache / (fixture.rows[0]["page_key"] + ".zip")
        archive.write_bytes(b"synthetic invalid ZIP")
        ledger = json.loads(fixture.ledgers[1].read_text())
        ledger.update(sha256=audit.file_hash(archive), bytes=archive.stat().st_size)
        fixture.ledgers[1].write_text(json.dumps(ledger) + "\n")
        result = fixture.run()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["recorded_records"], 2)
        self.assertEqual(result["retrieval_error_counts"], {"invalid_zip": 2})
        self.assertEqual(result["bytes"]["selected_sgf_bytes_charged"], 0)

    def test_eligibility_v2_rules_are_reused_without_relaxing_komi_or_clock(self):
        one, two = row("one"), row("two")
        fixture = self.fixture([one, two], members={one["page_key"]:
            [(member(one), sgf_bytes(one, replace={"KM": "7.5"})),
             (member(two), sgf_bytes(two, replace={"TM": "300"}))]})
        result = fixture.run()
        self.assertEqual(result["source_admitted_records"], 0)
        self.assertEqual(result["all_parsed"]["games"], 2)
        reasons = {reason for record in fixture.records() for reason in record["screening"]["eligibility_reasons"]}
        self.assertIn("metadata:rules_or_komi_out_of_scope", reasons)
        self.assertIn("metadata:time_control_unknown_or_out_of_scope", reasons)

    def test_selected_byte_limit_retains_partial_records_and_budget_failure(self):
        one, two = row("one"), row("two")
        size = len(sgf_bytes(one))
        fixture = self.fixture([one, two], limits={"selected_sgf_bytes": size + 1})
        result = fixture.run()
        self.assertEqual((result["status"], result["reason"]), ("partial", "selected_sgf_byte_limit"))
        self.assertEqual(result["recorded_records"], 2)
        self.assertEqual(result["attempted_records"], 2)
        self.assertEqual(result["attempted_unrecorded_records"], 0)
        self.assertIsNone(result["current_source"])
        self.assertEqual(fixture.records()[-1]["source_url"], two["url"])
        self.assertEqual(fixture.records()[-1]["error"], "selected_sgf_byte_limit")
        self.assertEqual(result["bytes"]["selected_sgf_bytes_charged"], size)
        progress = json.loads((fixture.output / "progress.json").read_text())
        self.assertEqual(progress["status"], "partial")
        exposure = json.loads((fixture.output / "exposure.json").read_text())
        self.assertEqual(exposure["source_ids"], sorted([one["url"], two["url"]]))
        self.assertEqual(len(exposure["sgf_sha256"]), 1)

    def test_declared_expanded_limit_stops_before_any_member_opens(self):
        item = row()
        fixture = self.fixture([item], limits={"expanded_zip_bytes": len(sgf_bytes(item)) - 1})
        with patch.object(audit.zipfile.ZipFile, "open", side_effect=AssertionError("no member reads")):
            result = fixture.run()
        self.assertEqual((result["status"], result["reason"]), ("partial", "expanded_zip_byte_limit"))
        self.assertEqual(result["attempted_records"], 0)
        self.assertEqual(result["bytes"]["selected_sgf_bytes_charged"], 0)
        self.assertLessEqual(result["bytes"]["expanded_zip_bytes"], result["limits"]["expanded_zip_bytes"])

    def test_member_limit_rejects_id_without_opening_it(self):
        fixture = self.fixture(limits={"member_bytes": 10})
        with patch.object(audit.zipfile.ZipFile, "open", side_effect=AssertionError("no member reads")):
            result = fixture.run()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["retrieval_error_counts"], {"member_byte_limit": 1})
        self.assertEqual(result["bytes"]["selected_sgf_bytes_charged"], 0)


if __name__ == "__main__":
    unittest.main()
