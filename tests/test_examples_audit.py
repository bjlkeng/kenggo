from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from kenggo.examples_audit import diagnostic_sample, probe, run
from kenggo.population_audit import run as build_population
from encoding_fixtures import corpus, record
from test_population_audit import population_fixture, profile_for


class ExampleAuditTests(unittest.TestCase):
    def test_diagnostics_never_choose_a_sealed_record_including_an_excluded_one(self):
        game = record(admitted=False)
        game["sealed_final"] = True
        self.assertEqual(diagnostic_sample({game["source"]["source_url"]: game})["games"], 0)
        game["sealed_final"] = False
        self.assertEqual(diagnostic_sample({game["source"]["source_url"]: game},
                                          sealed_source_ids=[game["source"]["source_url"]])["games"], 0)

    def test_production_audit_and_bounded_probe_use_only_development_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, ref, original = population_fixture(root)
            with mock.patch("kenggo.population_audit.load_original", return_value=original), redirect_stdout(io.StringIO()):
                summary = build_population(ref["path"], ref["sha256"], root / "source")
            profile = profile_for(root, summary)
            with redirect_stdout(io.StringIO()):
                cost = probe(root / "source", profile=profile, games=1)
                result = run(root / "source", root / "examples", profile=profile)
            self.assertEqual(cost["games"], 1)
            self.assertEqual(cost["examples"], 2)
            self.assertFalse(cost["artifacts_written"])
            self.assertTrue(result["normalizer_fitted"])
            self.assertEqual(result["counts"]["final"], {"examples": 2, "sealed": True})
            self.assertNotIn("sealed.sgf", (root / "examples/diagnostics.json").read_text())
            for games in (0, 6):
                with self.assertRaises(ValueError):
                    probe(root / "source", profile=profile, games=games)

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
