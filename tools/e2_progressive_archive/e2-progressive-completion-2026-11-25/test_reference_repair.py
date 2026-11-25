"""Parent paths can differ in spelling while identifying the same pinned file."""
import importlib.util
from copy import deepcopy
import os
from pathlib import Path
import sys
import unittest

ROOT = Path.cwd()
OLD = ROOT / "runs/e2-progressive-validation-verification-2026-11-24"
sys.path.insert(0, str(OLD))
import test_composite_checks as original
spec = importlib.util.spec_from_file_location("composite_v2", Path(__file__).with_name("composite_checks_v2.py"))
repaired = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repaired)
original.verifier = repaired

class ParentReferenceTests(original.CompositeVerifierTests):
    def setUp(self):
        super().setUp()
        config = original.read(self.fixture.request["inputs"]["original_config"])
        config["inputs"]["parent_summary"] = dict(self.fixture.parent,
            path=os.path.relpath(self.fixture.parent["path"], ROOT))
        self.fixture.request["inputs"]["original_config"] = original.fixture_module.write(self.root / "config.json", config)
        request = original.fixture_module.write(self.root / "request.json", self.fixture.request)
        built = original.fixture_module.composite.build(request["path"], request["sha256"], self.root / "job-relative", fixture=True)
        self.diagnostic = original.read(built["epoch_validation"])
        self.check.PLAN = config

    def test_same_parent_with_relative_and_absolute_paths(self):
        arrays, evidence = self.verify()
        self.assertEqual(evidence["selected_supplemental_games"], 2)
        self.assertEqual(len(arrays["action"]), 12)

    def replace_plan_parent(self, parent):
        config = deepcopy(self.check.PLAN)
        config["inputs"]["parent_summary"] = parent
        request = original.read(self.diagnostic["composite"]["request"])
        request["inputs"]["original_config"] = original.fixture_module.write(self.root / "other-config.json", config)
        self.diagnostic["composite"]["request"] = original.fixture_module.write(self.root / "other-request.json", request)
        self.check.PLAN = config

    def test_same_bytes_at_another_path_are_not_the_original_parent(self):
        other = self.root / "other-parent.json"
        other.write_bytes(Path(self.fixture.parent["path"]).read_bytes())
        self.replace_plan_parent(dict(self.fixture.parent, path=str(other)))
        with self.assertRaisesRegex(ValueError, "Original parent path"):
            self.verify()

    def test_changed_parent_hash_is_rejected(self):
        self.replace_plan_parent(dict(self.fixture.parent, sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "pin differs"):
            self.verify()

if __name__ == "__main__":
    unittest.main(verbosity=2)
