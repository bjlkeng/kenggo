from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock
import zipfile

import numpy as np
from sgfmill import sgf

from kenggo import data_expansion as expansion
from kenggo import komi_expansion as study
from kenggo.canonical_data import digest
from kenggo.data_splits import ROLES, read_membership
from kenggo.examples import read_shards
from kenggo.sgf_audit import scalar
from test_data_expansion import fixture, run
from test_population_audit import write_json


def study_fixture(root, definitions=None):
    base = root / "base-run"
    base.mkdir()
    request, request_ref, sources, _ = fixture(base, definitions)
    bundle = Path(sources["candidates"][0]["source"]["archive_path"])
    with zipfile.ZipFile(bundle) as archive:
        payloads = {name: archive.read(name).replace(b"KM[6.5]", b"KM[0.5]") for name in archive.namelist()}
    with zipfile.ZipFile(bundle, "w") as archive:
        for name, data in payloads.items():
            archive.writestr(name, data)
    for item in sources["candidates"]:
        item["source"]["archive_sha256"] = expansion.file_hash(bundle)
    sources_ref = write_json(base / "sources.json", sources)
    with redirect_stdout(io.StringIO()):
        summary = run(base, request_ref, sources_ref)
    summary_ref = expansion.reference(base / "out/summary.json")
    index = expansion.read_ref(summary["artifacts"]["base_index"])
    headers = []
    for number, item in enumerate(sources["candidates"]):
        data = payloads[item["source"]["zip_member"]]
        root_node = sgf.Sgf_game.from_bytes(data).get_root()
        header = {field: scalar(root_node, field) for field in ("RU", "KM", "BR", "WR", "TM", "OT", "PB", "PW", "DT", "SZ", "HA", "PL")}
        headers.append({"source_url": item["source"]["source_url"], "source_index": number,
                        "unit_ref": index["provenance"]["work_units"][number],
                        "sgf_sha256": sha256(data).hexdigest(), "header": header})
    scope_ref = write_json(root / "scope.json", {"inputs": {"sources": sources_ref}})
    header_ref = write_json(root / "headers.json", {"schema": "e2_komi_header_evidence_v1",
        "previous_sources": sources_ref, "previous_summary": summary_ref, "scope_audit": scope_ref,
        "records": headers})
    pins = {"previous_sources": sources_ref["sha256"], "previous_summary": summary_ref["sha256"],
            "scope_audit": scope_ref["sha256"]}
    counts = {role: sum(item["provisional_role"] == role for item in sources["candidates"]) for role in expansion.ALLOWED}
    counts = {role: count for role, count in counts.items() if count}
    launch = datetime.now(timezone.utc) - timedelta(seconds=1)
    timestamps = {"launch_utc": launch.isoformat(), "preparation_deadline_utc": (launch + timedelta(hours=22)).isoformat(),
                  "deadline_utc": (launch + timedelta(hours=24)).isoformat()}
    marker_ref = write_json(root / "session.json", {"schema": "e2_komi_production_session_v1", **timestamps,
        "output_directory": str(root / "study-out"), "workers": 1, "http_requests": 0,
        "gpu_seconds": 0, "training_performed": False})
    with mock.patch.dict(study.PRODUCTION_PINS, pins, clear=True), mock.patch.object(study, "EXPECTED_COUNTS", counts):
        frozen_ref = study.freeze_candidates(sources_ref, summary_ref, scope_ref, header_ref, root / "frozen.json")
    frozen = expansion.read_ref(frozen_ref)
    preflight = write_json(root / "preflight.json", {"schema": "e2_komi_preflight_v1", "status": "completed",
        "session_marker": marker_ref, "sources": frozen_ref, "header_evidence": header_ref,
        "cpu_seconds": 1.0, "wall_seconds": 1.0, "cpu_charged_seconds": 300,
        "selected_sgf_bytes": 0, "expanded_zip_bytes": 0})
    request = {**request, "schema": study.REQUEST_VERSION, **timestamps, "limits": dict(study.CEILINGS),
        "source_request_sha256": frozen["request_sha256"], "initial_cpu_seconds": 300,
        "preflight": preflight, "session_marker": marker_ref,
        "combined_example_caps": {role: summary["counts"]["combined"][role]["examples"] +
            study.CEILINGS["max_train_examples" if role == "train" else "max_development_examples"] for role in expansion.ALLOWED}}
    request["refs"] = {**request["refs"], **{name: summary["artifacts"][name]
                       for name in ("canonical", "split_manifest", "examples_manifest", "profile", "base_index")},
                       "prior_exposure": summary["artifacts"]["exposure"]}
    request["base_groups"] = expansion.read_ref(summary["artifacts"]["groups"])
    request["split_config"] = expansion.read_ref(summary["artifacts"]["split_manifest"])["plan"]
    request["source_revision"]["code_sha256"] = {path: expansion.file_hash(path) for path in study.CODE_PATHS}
    request_ref = write_json(root / "study-request.json", request)
    return request, request_ref, frozen_ref, pins, counts, summary


def execute(root, request_ref, sources_ref, pins, counts):
    with mock.patch.dict(study.PRODUCTION_PINS, pins, clear=True), mock.patch.object(study, "EXPECTED_COUNTS", counts):
        return study.build(request_ref["path"], request_ref["sha256"], sources_ref["path"], sources_ref["sha256"], root / "study-out")


class KomiExpansionTests(unittest.TestCase):
    def test_study_preserves_control_and_old_decisions_and_emits_both_komi(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request, req, sources, pins, counts, base = study_fixture(root)
            old_index = expansion.read_ref(base["artifacts"]["base_index"])
            loads = json.loads
            old_ids = {row["source_url"] for row in old_index["records"]}

            def no_old_targets(value, *args, **kwargs):
                if isinstance(value, str) and value.startswith('{"actions":'):
                    if any('"source_url":' + json.dumps(key) in value for key in old_ids):
                        raise AssertionError("An old canonical target was decoded")
                return loads(value, *args, **kwargs)

            with mock.patch("kenggo.data_expansion.json.loads", side_effect=no_old_targets):
                result = execute(root, req, sources, pins, counts)
            self.assertEqual(result["new_encoded_games"], 3)
            self.assertGreaterEqual(result["costs"]["cpu_seconds"], 300)
            current_index = expansion.read_ref(result["artifacts"]["base_index"])
            self.assertEqual(current_index["rejected"], old_index["rejected"])
            self.assertEqual(current_index["study_rejections"], [])
            _, old_parts = read_membership(Path(base["artifacts"]["split_manifest"]["path"]).parent,
                                           base["artifacts"]["split_manifest"]["sha256"])
            _, new_parts = read_membership(Path(result["artifacts"]["split_manifest"]["path"]).parent,
                                           result["artifacts"]["split_manifest"]["sha256"])
            by_id = {row["source_url"]: row for part in new_parts.values() for row in part}
            self.assertTrue(all(by_id[row["source_url"]] == row for part in old_parts.values() for row in part))
            before = expansion.read_ref(base["artifacts"]["examples_manifest"])
            after = expansion.read_ref(result["artifacts"]["examples_manifest"])
            directory = Path(result["artifacts"]["examples_manifest"]["path"]).parent
            for role in ROLES:
                self.assertEqual(after["artifacts"][role][:len(before["artifacts"][role])], before["artifacts"][role])
                for artifact in before["artifacts"][role]:
                    for kind in ("arrays", "metadata"):
                        self.assertTrue((directory / artifact[kind]).is_symlink())
                        self.assertEqual(expansion.file_hash(directory / artifact[kind]), artifact[kind + "_sha256"])
            with self.assertRaises(ValueError):
                list(read_shards(directory, result["artifacts"]["examples_manifest"]["sha256"], "train"))
            values = set()
            for arrays, _ in read_shards(directory, result["artifacts"]["examples_manifest"]["sha256"], "train", komi_study=True):
                values.update(np.abs(arrays["scalars"][:, 1]).tolist())
            self.assertEqual(values, {0.5, 6.5})
            self.assertEqual(expansion.read_ref(result["artifacts"]["normalizer"])["count"][5],
                             result["counts"]["combined"]["train"]["examples"])
            units = [expansion.read_ref(pin) for pin in current_index["provenance"]["work_units"]]
            key = units[0]["source_url"]
            frozen = expansion.read_ref(sources)
            split = expansion.read_ref(base["artifacts"]["split_manifest"])
            exposure = expansion.union_exposure(expansion.read_ref(base["artifacts"]["exposure"]), frozen["exposure"])
            empty = {field: set() for field in expansion.FINGERPRINT_FIELDS}
            _, own_rows, _ = expansion._new_memberships(deepcopy(units[:1]), empty, frozen["groups"], split,
                exposure, old_index["rejected"], reconsidered_sources=[key])
            self.assertEqual(own_rows[0]["role"], "train")
            alias = {**next(row for row in old_index["rejected"] if row["source_url"] == key),
                     "source_url": "https://files.gokgs.com/games/2025/7/3/unselected-alias.sgf"}
            _, blocked, _ = expansion._new_memberships(deepcopy(units[:1]), empty, frozen["groups"], split,
                exposure, [*old_index["rejected"], alias], reconsidered_sources=[key])
            self.assertEqual(blocked[0]["role"], "excluded")
            self.assertIn("duplicate_of_rejected_source", blocked[0]["exclusions"])

    def test_default_builder_rejects_previous_sources_and_enlarged_limits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request, _, frozen_ref, _, _, _ = study_fixture(root)
            sources = expansion.read_ref(frozen_ref)
            old_sources = expansion.read_ref(sources["provenance"]["previous_sources"])
            old_request = {**request, "schema": expansion.REQUEST_VERSION,
                           "source_request_sha256": old_sources["request_sha256"], "limits": dict(expansion.CEILINGS)}
            old_sources["base_grouping_sha256"] = sources["base_grouping_sha256"]
            with self.assertRaisesRegex(ValueError, "already belong"):
                expansion._inputs(old_request, old_sources)
            old_request["limits"]["canonical_file_bytes"] = 128 * expansion.MIB
            with self.assertRaisesRegex(ValueError, "enlarged"):
                expansion._inputs(old_request, old_sources)

    def test_frozen_ledger_tampering_stops_before_replay_and_charges_failed_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request, _, frozen_ref, pins, counts, _ = study_fixture(root)
            frozen = expansion.read_ref(frozen_ref)
            frozen["candidates"][0]["reconsideration"]["prior_decision"]["exclusions"] = []
            changed_ref = write_json(root / "tampered.json", frozen)
            preflight = expansion.read_ref(request["preflight"])
            preflight["sources"] = changed_ref
            request["preflight"] = write_json(root / "preflight-tampered.json", preflight)
            req = write_json(root / "study-request.json", request)
            for attempt in range(2):
                with mock.patch("kenggo.data_expansion._canonical_phase") as replay:
                    with self.assertRaisesRegex(ValueError, "ledger changed"):
                        execute(root, req, changed_ref, pins, counts)
                    replay.assert_not_called()
                state = json.loads((root / "study-out/progress.json").read_text())
                self.assertGreaterEqual(state["cpu_charged_seconds"], 360 + 60 * attempt)
            replacement = root / "replacement-output"
            with self.assertRaisesRegex(ValueError, "another build"):
                study.build(req["path"], req["sha256"], changed_ref["path"], changed_ref["sha256"], replacement)

    def test_role_order_focus_order_and_own_rejection_exception_are_narrow(self):
        candidates = []
        for name, role, rank in (("a", "train", "5k"), ("b", "train", "15k"),
                                 ("c", "population_development", "5k"), ("d", "train", "14k"),
                                 ("e", "validation_development", "1d")):
            candidates.append({"source": {"source_url": name}, "provisional_role": role,
                               "archive_row": {"black": "a [" + rank + "]", "white": "b [1d]"}})
        self.assertEqual([row["source"]["source_url"] for row in study._ordered(candidates)], ["b", "c", "e", "d", "a"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request, req, frozen_ref, pins, counts, _ = study_fixture(root)
            frozen = expansion.read_ref(frozen_ref)
            header_ref = frozen["provenance"]["header_evidence"]
            headers = expansion.read_ref(header_ref)
            headers["records"][0]["source_index"] = 1
            altered = write_json(root / "altered-headers.json", headers)
            with mock.patch.dict(study.PRODUCTION_PINS, pins, clear=True), mock.patch.object(study, "EXPECTED_COUNTS", counts):
                with self.assertRaisesRegex(ValueError, "source order"):
                    study.freeze_candidates(frozen["provenance"]["previous_sources"],
                        frozen["provenance"]["previous_summary"], frozen["provenance"]["scope_audit"], altered,
                        root / "bad-freeze.json")

    def test_budget_retains_launch_and_preparation_reserve_on_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request, req, sources, _, _, _ = study_fixture(root)
            output = root / "budget"
            output.mkdir()
            budget = study.StudyBudget(output, request, req["sha256"], sources["sha256"], time.process_time(),
                                      datetime.now(timezone.utc))
            budget.begin(phase="synthetic_interrupted_work")
            charged = budget.state["cpu_charged_seconds"]
            resumed = study.StudyBudget(output, request, req["sha256"], sources["sha256"], time.process_time(),
                                       datetime.now(timezone.utc))
            self.assertGreaterEqual(resumed.cpu, charged)
            self.assertEqual(resumed.state["started_utc"], request["launch_utc"])
            resumed.prior_cpu = request["limits"]["preparation_cpu_seconds"]
            with self.assertRaisesRegex(expansion.BudgetStop, "preparation_cpu_budget"):
                resumed.check()
            resumed.check(finishing=True)
            resumed.prior_cpu = request["limits"]["cpu_seconds"]
            with self.assertRaisesRegex(expansion.BudgetStop, "cpu_budget"):
                resumed.check(finishing=True)

    def test_one_writer_and_missing_progress_cannot_reset_a_used_session(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = {"session_marker": {"path": str(root / "session.json"), "sha256": "a" * 64}}
            with study._writer_lock(request, "b" * 64, "c" * 64, root / "output"):
                with self.assertRaisesRegex(ValueError, "writer is already active"):
                    with study._writer_lock(request, "b" * 64, "c" * 64, root / "output"):
                        self.fail("Concurrent writer entered")
            with self.assertRaisesRegex(ValueError, "allowance cannot reset"):
                with study._writer_lock(request, "b" * 64, "c" * 64, root / "output"):
                    self.fail("Missing progress reset the session")

    def test_finishing_interruption_retains_uncheckpointed_cpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request, req, sources, _, _, _ = study_fixture(root)
            for phase in ("publication", "normalizer", "support", "complete"):
                with self.subTest(phase=phase):
                    output = root / phase
                    output.mkdir()
                    with mock.patch.object(study.time, "process_time", return_value=100):
                        budget = study.StudyBudget(output, request, req["sha256"], sources["sha256"],
                                                  100, datetime.now(timezone.utc))
                        budget.end(phase=phase)
                    with mock.patch.object(study.time, "process_time", return_value=109):
                        budget.check(finishing=True)
                        spent = budget.cpu
                    with mock.patch.object(study.time, "process_time", return_value=200):
                        resumed = study.StudyBudget(output, request, req["sha256"], sources["sha256"],
                                                   200, datetime.now(timezone.utc))
                        self.assertGreaterEqual(resumed.cpu, spent)

    def test_verification_attempt_prevents_builder_resume_without_losing_its_charge(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request, req, sources, pins, counts, _ = study_fixture(root)
            execute(root, req, sources, pins, counts)
            ledger = Path(request["session_marker"]["path"] + ".verification.json")
            ledger.write_text(json.dumps({"request_sha256": req["sha256"], "attempts": [
                {"status": "running", "cpu_charged_seconds": 900}]}))
            progress = (root / "study-out/progress.json").read_bytes()
            with mock.patch.object(expansion, "_execute_build") as build:
                with self.assertRaisesRegex(ValueError, "after verification"):
                    execute(root, req, sources, pins, counts)
                build.assert_not_called()
            self.assertEqual((root / "study-out/progress.json").read_bytes(), progress)
            self.assertEqual(json.loads(ledger.read_text())["attempts"][0]["cpu_charged_seconds"], 900)

    def test_interrupted_publication_resumes_after_preparation_cutoff_without_new_examples(self):
        for cutoff in ("cpu", "elapsed"):
            with self.subTest(cutoff=cutoff), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                request, req, sources, pins, counts, _ = study_fixture(root)
                with mock.patch.object(expansion, "_publish", side_effect=RuntimeError("interrupted publication")):
                    with self.assertRaisesRegex(RuntimeError, "interrupted publication"):
                        execute(root, req, sources, pins, counts)
                state_path = root / "study-out/progress.json"
                state = json.loads(state_path.read_text())
                if cutoff == "cpu":
                    state.update(cpu_charged_seconds=68461, cpu_seconds=68401)
                    state_path.write_text(json.dumps(state))
                instant = (datetime.fromisoformat(request["launch_utc"]) + timedelta(hours=22, seconds=1)
                           if cutoff == "elapsed" else datetime.now(timezone.utc))

                class Clock(datetime):
                    @classmethod
                    def now(cls, tz=None):
                        return instant.astimezone(tz) if tz else instant.replace(tzinfo=None)

                with mock.patch.object(study, "datetime", Clock), \
                        mock.patch.object(expansion, "datetime", Clock), \
                        mock.patch.object(expansion, "_canonical_unit", side_effect=AssertionError("New replay")), \
                        mock.patch.object(expansion, "examples_for_game", side_effect=AssertionError("New encoding")):
                    result = execute(root, req, sources, pins, counts)
                self.assertEqual(result["new_encoded_games"], 3)
                self.assertLess(result["costs"]["cpu_seconds"], 72000)
                self.assertEqual(result["launch_utc"], request["launch_utc"])
                self.assertEqual(result["deadline_utc"], request["deadline_utc"])

    def test_pinned_member_change_stops_before_any_encoding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, req, frozen_ref, pins, counts, _ = study_fixture(root)
            sources = expansion.read_ref(frozen_ref)
            bundle = Path(sources["candidates"][0]["source"]["archive_path"])
            with bundle.open("ab") as stream:
                stream.write(b"tampered")
            with mock.patch("kenggo.data_expansion._example_phase") as encoding:
                with self.assertRaisesRegex(ValueError, "Pinned input changed"):
                    execute(root, req, frozen_ref, pins, counts)
                encoding.assert_not_called()

    def test_whole_game_row_cap_preserves_partial_evidence_and_survives_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request, _, sources, pins, counts, baseline = study_fixture(root)
            request["limits"]["max_train_examples"] = 1
            request["combined_example_caps"]["train"] = baseline["counts"]["combined"]["train"]["examples"] + 1
            req = write_json(root / "study-request.json", request)
            first = execute(root, req, sources, pins, counts)
            self.assertEqual(first["status"], "budget_stopped")
            self.assertEqual(first["counts"]["new"]["train"]["examples"], 0)
            self.assertEqual(first["new_encoded_games"], 2)
            self.assertEqual(first["new_split_exclusions"]["expansion_example_row_limit"], 1)
            second = execute(root, req, sources, pins, counts)
            self.assertEqual(first["counts"], second["counts"])
            self.assertEqual(first["launch_utc"], second["launch_utc"])
            self.assertGreater(second["costs"]["cpu_seconds"], first["costs"]["cpu_seconds"])


if __name__ == "__main__":
    unittest.main()
