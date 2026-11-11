from pathlib import Path
import tempfile
import unittest

from kenggo.examples_audit import diagnostic_sample, run
from encoding_fixtures import corpus, record


class ExampleAuditTests(unittest.TestCase):
    def test_diagnostics_do_not_open_source_gate_or_emit_training_rows(self):
        game = record(moves=";B[aa];W[bb];B[]", admitted=False)
        before = game["eligibility"].copy()
        result = diagnostic_sample({game["source"]["source_url"]: game})
        self.assertEqual(result["games"], 1)
        self.assertEqual(result["positions"], 3)
        self.assertEqual(result["mismatches"], 0)
        self.assertEqual(result["training_examples_emitted"], 0)
        self.assertEqual(game["eligibility"], before)
        self.assertFalse(result["evidence"][0]["source_admitted"])
        self.assertEqual(result["history_windows"], [0, 8, 32, 64])

    def test_production_command_rejects_unreviewed_positive_fixture(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus(root / "source", [record()])
            with self.assertRaises(ValueError):
                run(root / "source", root / "out")
            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
