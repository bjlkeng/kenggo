import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from kenggo.canonical_data import digest
from kenggo.data_splits import read_membership, write_membership
from kenggo.examples import build_examples, file_hash, fit_normalizer, load_corpus, read_shards
from kenggo.population_audit import run as build_population
from test_population_audit import population_fixture, profile_for
from encoding_fixtures import corpus, record
from tools.apply_e1_review import exclude_memberships, filter_examples


class ReviewExclusionTests(unittest.TestCase):
    def test_filter_keeps_surviving_train_rows_and_refits_their_normalizer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            excluded, kept = record("remove", result="B+R"), record("keep", result="W+R")
            args = corpus(root / "source", [excluded, kept])
            pin = build_examples(*args, root / "examples", fixture_only=True)
            manifest = json.loads((root / "examples/manifest.json").read_text())
            arrays, meta = next(read_shards(root / "examples", pin, "train"))
            selected = [i for i, row in enumerate(meta) if row["source_url"] == kept["source"]["source_url"]]
            artifacts, counts, _ = filter_examples(root / "examples", pin, manifest, root / "out",
                {excluded["source"]["source_url"]}, lambda: None)
            path = root / "out/manifest.json"
            path.write_text(json.dumps(manifest | {"artifacts": artifacts, "counts": counts}))
            output_pin = file_hash(path)
            after, metadata = next(read_shards(root / "out", output_pin, "train"))
            self.assertEqual(metadata, [meta[i] for i in selected])
            for key in arrays:
                np.testing.assert_array_equal(after[key], arrays[key][selected])
            self.assertEqual(counts["train"]["examples"], len(selected))
            self.assertEqual(fit_normalizer(root / "out", output_pin)["count"][5], len(selected))

    def test_whole_game_removal_preserves_other_memberships_and_canonical_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, ref, original = population_fixture(root)
            with patch("kenggo.population_audit.load_original", return_value=original), redirect_stdout(io.StringIO()):
                summary = build_population(ref["path"], ref["sha256"], root / "source")
            profile = profile_for(root, summary)
            records, _, _ = load_corpus(root / "source/canonical.parquet", profile.canonical_sha256,
                                        root / "source/splits", profile.split_sha256, profile=profile)
            split, partitions = read_membership(root / "source/splits", profile.split_sha256)
            before = copy.deepcopy((records, split, partitions))
            row = partitions["train"][0]
            key = row["source_url"]
            policy = [{"source_url": key, "canonical_sha256": row["canonical_sha256"],
                       "sgf_sha256": records[key]["source"]["sgf_sha256"], "reason": "owner_excluded"}]
            rows, updated = exclude_memberships(records, split, partitions, policy, digest("review"))
            changed = next(item for item in rows if item["source_url"] == key)
            self.assertEqual(changed["role"], "excluded")
            self.assertEqual(changed["policy_actions"], 0)
            self.assertEqual(changed["value_actions"], 0)
            self.assertIn("owner_excluded", changed["exclusions"])
            self.assertEqual(updated["roles"]["train"], split["roles"]["train"] - 1)
            old = {item["source_url"]: item for group in partitions.values() for item in group}
            self.assertTrue(all(item == old[item["source_url"]] for item in rows if item["source_url"] != key))
            self.assertEqual((records, split, partitions), before)
            changed_hash = write_membership(root / "new-splits", rows, updated)
            new_manifest, _ = read_membership(root / "new-splits", changed_hash)
            self.assertEqual(new_manifest["membership_sha256"]["final.parquet"], split["membership_sha256"]["final.parquet"])
            for change in ({"source_url": "sealed"}, {"sgf_sha256": digest("changed")},
                           {"canonical_sha256": digest("changed")}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    exclude_memberships(records, split, partitions, [policy[0] | change], digest("review"))

    def test_filter_preserves_surviving_arrays_and_never_decodes_final_shards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, ref, original = population_fixture(root)
            with patch("kenggo.population_audit.load_original", return_value=original), redirect_stdout(io.StringIO()):
                summary = build_population(ref["path"], ref["sha256"], root / "source")
            profile = profile_for(root, summary)
            pin = build_examples(root / "source/canonical.parquet", profile.canonical_sha256,
                                 root / "source/splits", profile.split_sha256, root / "examples", profile=profile)
            manifest = json.loads((root / "examples/manifest.json").read_text())
            arrays, metadata = next(read_shards(root / "examples", pin, "train"))
            excluded = {metadata[0]["source_url"]}
            old_np_load = np.load
            def reject_final(path, *args, **kwargs):
                self.assertNotIn("final", Path(path).parts)
                return old_np_load(path, *args, **kwargs)
            with patch("numpy.load", side_effect=reject_final):
                artifacts, counts, removed = filter_examples(root / "examples", pin, manifest,
                    root / "filtered", excluded, lambda: None)
            self.assertEqual(removed, {next(iter(excluded)): len(metadata)})
            self.assertEqual(counts["train"]["examples"], 0)
            self.assertEqual(artifacts["final"], manifest["artifacts"]["final"])
            for item in artifacts["final"]:
                for field in ("arrays", "metadata"):
                    self.assertEqual(file_hash(root / "filtered" / item[field]), item[field + "_sha256"])
            changed = manifest | {"artifacts": artifacts, "counts": counts}
            target = root / "filtered/manifest.json"
            target.write_text(json.dumps(changed))
            filtered_pin = file_hash(target)
            self.assertEqual(fit_normalizer(root / "filtered", filtered_pin)["count"][5], 0)
            before = list(read_shards(root / "examples", pin, "validation_development"))
            after = list(read_shards(root / "filtered", filtered_pin, "validation_development"))
            for (a, am), (b, bm) in zip(before, after):
                self.assertEqual(am, bm)
                for field in a:
                    np.testing.assert_array_equal(a[field], b[field])
            with self.assertRaises(PermissionError):
                list(read_shards(root / "filtered", filtered_pin, "final"))


if __name__ == "__main__":
    unittest.main()
