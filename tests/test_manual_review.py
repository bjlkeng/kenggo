import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from encoding_fixtures import record
from kenggo.encoding import encode
from kenggo.examples import replay_record
from kenggo.go import Game
from tools.build_e1_review import (EXAMPLE_ROLES, build, collect_examples, frames, render,
                                   selection, source_frames, stored_example)


class ManualReviewTests(unittest.TestCase):
    def test_frames_preserve_capture_and_pass_without_reimplementing_rules(self):
        board = ["." * 19 for _ in range(19)]
        board[0] = ".W." + "." * 16
        board[1] = ".B." + "." * 16
        game = Game.from_fixture(board)
        game = game.play("B", 0).play("W", 361).play("B", 2)
        packed = source_frames(game)
        reconstructed = ["."] * 361
        for frame, position in zip(packed, game.positions):
            for index, stone in frame[0]:
                reconstructed[index] = stone
            self.assertEqual(tuple(reconstructed), position.board)
            self.assertEqual(frame[2:5], [position.captured_black, position.captured_white,
                                         position.pass_streak])
        self.assertEqual(packed[-1][3], 1)
        self.assertEqual(packed[2][0], [])
        with self.assertRaises(ValueError):
            frames([{"board": ["bad"]}])

    def test_selection_excludes_final_confirmation_and_sealed_development(self):
        records = {key: {"actions": [{"action": 0}, {"action": 361}], "sealed_final": key == "sealed"}
                   for key in ("train", "dev", "confirmation", "final", "sealed")}
        parts = {"train": [{"source_url": "train"}], "population_development": [{"source_url": "dev"}],
                 "validation_development": [], "population_confirmation": [{"source_url": "confirmation"}],
                 "final": [{"source_url": "final"}]}
        diagnostics = {"evidence": [{"source_url": key} for key in records]}
        self.assertEqual(selection(records, parts, diagnostics),
                         {("train", 0), ("train", 1), ("dev", 0), ("dev", 1)})
        parts["train"].append({"source_url": "sealed"})
        with self.assertRaises(PermissionError):
            selection(records, parts, diagnostics)

    def example(self):
        source = record(result="B+R")
        game = replay_record(source)
        before = Game.new().play("B", source["actions"][0]["action"])
        inputs = encode(before)
        labels = {"action": source["actions"][1]["action"], "policy_valid": True,
                  "value": -1.0, "value_valid": True, "duration": 0.0, "duration_valid": False}
        arrays = {key: np.asarray([value]) for key, value in (inputs | labels).items()}
        metadata = {"role": "train", "ply": 1, "mover": "W", "source_url": source["source"]["source_url"]}
        return arrays, metadata, source, game

    def test_stored_example_uses_actual_mover_planes_masks_and_labels(self):
        arrays, meta, source, game = self.example()
        example = stored_example(arrays, 0, meta, source, game)
        self.assertEqual(example["board"], "".join(game.positions[1].board))
        self.assertEqual(example["value"], -1)
        self.assertFalse(example["duration_valid"])
        self.assertEqual(example["legal"][example["action"]], "1")
        self.assertFalse(example["scalar_valid"][-1])
        arrays["value"][0] = 1
        with self.assertRaisesRegex(ValueError, "source replay or labels"):
            stored_example(arrays, 0, meta, source, game)
        arrays["value"][0] = -1
        arrays["board"][0, 0, 0, 0] = True
        with self.assertRaises(ValueError):
            stored_example(arrays, 0, meta, source, game)

    def test_example_reader_never_requests_final_shards_and_rejects_missing_rows(self):
        calls = []
        def shards(directory, pin, role):
            calls.append(role)
            return iter(())
        with patch("tools.build_e1_review.read_shards", side_effect=shards):
            self.assertEqual(collect_examples("unused", "pin", set(), {}, {}), [])
            self.assertEqual(tuple(calls), EXAMPLE_ROLES)
            with self.assertRaisesRegex(ValueError, "missing"):
                collect_examples("unused", "pin", {("missing", 0)}, {}, {})
        arrays, meta, source, game = self.example()
        with self.assertRaises(PermissionError):
            stored_example(arrays, 0, meta | {"role": "final"}, source, game)
        with self.assertRaises(PermissionError):
            stored_example(arrays, 0, meta, source | {"sealed_final": True}, game)

    def test_source_text_cannot_escape_embedded_json(self):
        payload = {"name": '</script><script>alert("x")</script>&', "line": "\u2028"}
        page = render(payload)
        encoded = page.split('<script id="review-data" type="application/json">')[1].split("</script>")[0]
        self.assertEqual(json.loads(encoded), payload)
        self.assertNotIn("</script>", encoded)
        self.assertNotIn("/* REVIEW_JS */", page)
        self.assertNotIn("/* REVIEW_CSS */", page)

    def test_existing_export_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "index.html"
            output.write_text("existing user review")
            with self.assertRaises(FileExistsError):
                build(output)
            self.assertEqual(output.read_text(), "existing user review")


if __name__ == "__main__":
    unittest.main()
