import importlib.util
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


spec = importlib.util.spec_from_file_location("komi_launch", Path(__file__).resolve().parents[1] / "tools/prepare_komi_study.py")
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


class KomiLaunchTests(unittest.TestCase):
    def test_successful_preflight_freezes_a_buildable_request_and_shared_limits(self):
        from kenggo import komi_expansion as study
        from test_komi_expansion import study_fixture

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request, _, source_ref, pins, counts, _ = study_fixture(root)
            provenance = launch.read_ref(source_ref)["provenance"]
            request = deepcopy(request)
            for field in ("base_groups", "split_config", "session_marker", "preflight",
                          "launch_utc", "preparation_deadline_utc", "deadline_utc",
                          "source_request_sha256", "initial_cpu_seconds"):
                request.pop(field)
            request["source_revision"]["code_sha256"]["tools/prepare_komi_study.py"] = \
                launch.reference("tools/prepare_komi_study.py")["sha256"]
            template = {"schema": "e2_komi_launch_template_v1", "request": request,
                        **{name: provenance[name] for name in pins}}
            template_ref = launch.exclusive_json(root / "template.json", template)
            byte_costs = {"selected_sgf_bytes": 1234, "expanded_zip_bytes": 5678}
            with mock.patch.dict(study.PRODUCTION_PINS, pins, clear=True), \
                    mock.patch.object(study, "EXPECTED_COUNTS", counts), \
                    mock.patch.object(launch, "header_evidence", return_value=(
                        provenance["header_evidence"], byte_costs)):
                result = launch.prepare(template_ref, root / "launch", root / "production")
                actual = launch.read_ref(result["request"])
                frozen = launch.read_ref(result["sources"])
                study.validate_request(actual, frozen, root / "production")
            self.assertEqual(actual["initial_cpu_seconds"], 300)
            self.assertEqual(actual["limits"]["selected_sgf_bytes"], 256 * launch.MIB - 1234)
            self.assertEqual(actual["limits"]["expanded_zip_bytes"], 4 * 1024**3 - 5678)
            self.assertEqual(actual["limits"]["output_bytes"], 31 * 1024**3)
            self.assertEqual(frozen["counts"], counts)
            self.assertFalse((root / "production").exists())

    def test_only_root_fields_are_decoded_and_escaped_separators_are_preserved(self):
        raw = (b"(;FF[4]GM[1]SZ[19]RU[Japanese]KM[0.5]PB[Alice]PW[bob]BR[15k]WR[14k]"
               b"TM[1200]OT[5x30 byo-yomi]DT[2025-07-01]C[escaped\\] ; ( ) text]"
               b"RE[deliberately unparsed]; THIS IS NOT A VALID MOVE TREE")
        header, pin = launch.root_metadata(raw)
        self.assertEqual(header["KM"], .5)
        self.assertEqual(header["PB"], "Alice")
        self.assertEqual(set(header), set(launch.FIELDS))
        self.assertNotIn("RE", header)
        self.assertEqual(len(pin), 64)
        changed = raw.replace(b"THIS IS NOT A VALID MOVE TREE", b"B[zz])")
        self.assertEqual(launch.root_metadata(changed), (header, pin))

    def test_failed_preflight_keeps_the_one_use_marker_and_full_cpu_charge(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "template.json"
            source.write_text("{}")
            marker_dir = root / "preflight"
            with mock.patch.object(launch, "header_evidence") as headers:
                with self.assertRaisesRegex(ValueError, "Pinned input changed"):
                    launch.prepare({"path": str(source), "sha256": "0" * 64}, marker_dir, root / "build")
                headers.assert_not_called()
            marker = (marker_dir / "session.json").read_bytes()
            failure = json.loads((marker_dir / "failure.json").read_text())
            self.assertEqual(failure["cpu_charged_seconds"], 300)
            self.assertEqual(failure["status"], "failed")
            with self.assertRaises(FileExistsError):
                launch.prepare(launch.reference(source), marker_dir, root / "build")
            self.assertEqual((marker_dir / "session.json").read_bytes(), marker)
            self.assertFalse((root / "build").exists())

    def test_existing_build_or_nested_output_is_rejected_without_overwriting_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sentinel = root / "keep.txt"
            sentinel.write_text("keep")
            with self.assertRaises(FileExistsError):
                launch.prepare({}, root.parent / "unused-komi-preflight", root)
            with self.assertRaisesRegex(ValueError, "separate directories"):
                launch.prepare({}, root / "preflight", root / "preflight/build")
            self.assertEqual(sentinel.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
