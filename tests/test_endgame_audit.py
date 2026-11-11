"""A rejected rerun cannot mutate another run's evidence."""

from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


class OutputOwnershipTests(unittest.TestCase):
    def test_a_new_owned_directory_records_a_genuine_failure(self):
        path = Path(__file__).parents[1] / "tools/audit_endgame_referee.py"
        spec = importlib.util.spec_from_file_location("audit_endgame_referee", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "new-run"
            with patch.object(module.Profile, "load", return_value=object()), \
                    patch.object(module, "_run", side_effect=RuntimeError("synthetic failure")):
                with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                    module.run("unused", "unused", output)
            failure = json.loads((output / "failure.json").read_text())
            self.assertEqual(failure["status"], "failed")
            self.assertEqual(failure["error_type"], "RuntimeError")
            self.assertEqual(failure["error"], "synthetic failure")
            self.assertGreaterEqual(failure["cost"]["cpu_seconds"], 0)
            self.assertGreaterEqual(failure["cost"]["wall_seconds"], 0)

    def test_existing_output_is_unchanged_with_or_without_a_failure_record(self):
        root = Path(__file__).parents[1]
        for failed in (False, True):
            with self.subTest(failed=failed), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                (output / "summary.json").write_bytes(b'{"original_run": true}\n')
                if failed:
                    (output / "failure.json").write_bytes(b'{"original_failure_cost": 123}\n')
                before = {path.name: path.read_bytes() for path in output.iterdir()}
                process = subprocess.run([
                    sys.executable, str(root / "tools/audit_endgame_referee.py"),
                    str(root / "manifests/e1_katago_referee_profile_v2.json"),
                    "2831d93dfbba8713a7e3fb00a1318bd4024ae9defd8032e39737f807ee4a1be4",
                    str(output), "--fixtures-only"], cwd=root, capture_output=True, timeout=10)
                self.assertNotEqual(process.returncode, 0)
                self.assertIn(b"FileExistsError", process.stderr)
                self.assertEqual({path.name: path.read_bytes() for path in output.iterdir()}, before)
