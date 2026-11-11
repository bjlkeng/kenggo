from contextlib import redirect_stdout
from hashlib import sha256
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq

from kenggo.canonical_data import digest, encoded
from kenggo.examples import (build_examples, file_hash, fit_normalizer, load_corpus,
                             load_dataset_profile, read_shards)
from kenggo.kgs_admission import DECISION_PATH, DECISION_SHA256, classify
from kenggo.kgs_eligibility import exposure_registry, grouping_digest, handle_groups
from kenggo.population_audit import run
from kenggo.sgf_audit import inspect_sgf


def write_json(path, value):
    path.write_text(encoded(value) + "\n")
    return {"path": str(path), "sha256": file_hash(path)}


def population_fixture(root):
    records = []
    definitions = [("train", "alice", "bob", "2025-07-01", "aa", "bb"),
                   ("prefix", "held", "bob", "2025-07-02", "cc", "dd"),
                   ("development", "held", "alice", "2025-09-01", "ee", "ff"),
                   ("confirmation", "alice", "held", "2025-11-01", "gg", "hh"),
                   ("sealed", "carol", "dave", "2025-11-02", "ii", "jj")]
    bundle = root / "games.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        for name, black, white, day, first, second in definitions:
            member = "games/" + day.replace("-", "/") + "/" + name + ".sgf"
            url = "https://files.gokgs.com/" + member
            data = (f"(;FF[4]GM[1]SZ[19]PB[{black}]PW[{white}]BR[5k]WR[1d]RU[Japanese]KM[6.5]"
                    f"TM[1200]OT[5x30 byo-yomi]DT[{day}]RE[B+R];B[{first}];W[{second}])").encode()
            row = {"url": url, "black": black + " [5k]", "white": white + " [1d]",
                   "type": "Ranked", "setup": "19×19", "result": "B+Res.", "viewable": "Yes"}
            source = {"source_url": url, "sgf_sha256": sha256(data).hexdigest(),
                      "archive_path": str(bundle), "zip_member": member}
            records.append({"archive_row": row, "source": source, "audit": inspect_sgf(data, row)})
            archive.writestr(member, data)
    for record in records:
        record["source"]["archive_sha256"] = file_hash(bundle)
    discovery = [{"user": "alice", "months": [7]}]
    original = records[:-1]
    audits = [{**record["audit"], "source_url": record["source"]["source_url"]} for record in original]
    archived = {record["archive_row"]["url"]: record["archive_row"] for record in original}
    groups = handle_groups(["alice", "bob", "held", "cu2"])
    exposure = exposure_registry(audits, discovery, groups)
    decisions = [classify(audit, archived[audit["source_url"]], groups, exposure) for audit in audits]
    decisions_path = root / "admission.parquet"
    pq.write_table(pa.Table.from_pylist(decisions), decisions_path)
    reserved = ["kgs-handle:carol", "kgs-handle:dave", "kgs-handle:unused"]
    all_groups = handle_groups([*groups, "carol", "dave", "unused"])
    final_record = {key: records[-1][key] for key in ("source", "archive_row")}
    final_ref = write_json(root / "final.json", {"version": "sealed_kgs_index_v1",
                           "reservation_sha256": digest("fixture reservation"), "sample_sha256": digest("fixture sample"),
                           "test_groups": reserved, "records": [final_record]})
    scope_ref = write_json(root / "scope.json", {"version": "kgs_sealed_final_scope_v1",
                           "source_policy_sha256": DECISION_SHA256, "reservation_sha256": digest("fixture reservation"),
                           "date_from": "2025-11-01", "date_before": "2026-01-01",
                           "suspected_bot_handle_substrings": ["bot"], "reserved_groups": reserved})
    split_ref = write_json(root / "split.json", {"grouping_sha256": grouping_digest(all_groups),
                           "train_before": "2025-09-01", "confirmation_from": "2025-11-01",
                           "validation_groups": ["kgs-handle:held"], "test_groups": reserved})
    inputs = {"version": "kgs_population_inputs_v1", "original_source_directory": str(root),
              "source_policy": {"path": str(DECISION_PATH), "sha256": DECISION_SHA256},
              "admission_decisions": {"path": str(decisions_path), "sha256": file_hash(decisions_path)},
              "sealed_final": final_ref, "final_scope_policy": scope_ref, "split_config": split_ref,
              "limits": {"max_records": 10, "max_selected_sgf_bytes": 1024**2,
                         "max_expanded_zip_bytes": 1024**2, "max_canonical_json_bytes": 1024**2,
                         "cpu_seconds": 30, "wall_seconds": 60}}
    ref = write_json(root / "inputs.json", inputs)
    original_inputs = (audits, archived, exposure, discovery, [record["source"] for record in original])
    return inputs, ref, original_inputs


def profile_for(root, summary):
    profile = {"version": "population_example_profile_v1", "canonical_sha256": summary["canonical_parquet_sha256"],
               "split_sha256": summary["split_manifest_sha256"],
               "population_inputs_sha256": summary["provenance"]["population_inputs_sha256"],
               "sealed_source_ids_sha256": summary["sealed_source_ids_sha256"],
               "limits": {"max_records": 10, "max_examples_per_role": 100, "cpu_seconds": 30, "wall_seconds": 60}}
    ref = write_json(root / "profile.json", profile)
    return load_dataset_profile(ref["path"], ref["sha256"])


class PopulationAuditTests(unittest.TestCase):
    def test_positive_population_and_reserved_final_reproduce_without_releasing_final(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, ref, original = population_fixture(root)
            with mock.patch("kenggo.population_audit.load_original", return_value=original), redirect_stdout(io.StringIO()):
                first = run(ref["path"], ref["sha256"], root / "a")
                second = run(ref["path"], ref["sha256"], root / "b")
            self.assertEqual(first["canonical_parquet_sha256"], second["canonical_parquet_sha256"])
            self.assertEqual(first["split_manifest_sha256"], second["split_manifest_sha256"])
            self.assertEqual(first["source_admitted_original_games"], 4)
            self.assertEqual(first["sealed_final_usable_games"], 1)
            profile = profile_for(root, first)
            args = (root / "a/canonical.parquet", profile.canonical_sha256, root / "a/splits", profile.split_sha256)
            records, manifest, partitions = load_corpus(*args, profile=profile)
            self.assertEqual(len(records), 4)
            self.assertEqual(partitions["final"], [])
            self.assertIn("kgs-handle:unused", manifest["plan"]["test_groups"])
            self.assertEqual(partitions["validation_prefix"][0]["policy_actions"], 0)
            self.assertEqual(partitions["validation_development"][0]["policy_actions"], 2)
            key = build_examples(*args, root / "examples", profile=profile)
            self.assertTrue(fit_normalizer(root / "examples", key)["fitted"])
            self.assertEqual(fit_normalizer(root / "examples", key)["count"][5], 2)
            arrays, metadata = next(read_shards(root / "examples", key, "validation_development"))
            self.assertEqual(len(metadata), 2)
            self.assertTrue(all(row["persona_mode"] == "default" and not row["identity_valid"] for row in metadata))
            self.assertEqual(arrays["value_valid"].tolist(), [True, True])
            with self.assertRaises(PermissionError):
                list(read_shards(root / "examples", key, "final"))
            with self.assertRaises(PermissionError):
                load_corpus(*args, profile=profile, include_sealed=True)
            index = pq.read_table(root / "a/index.parquet").to_pylist()
            self.assertTrue(all(row["metadata_json"] is None for row in index if row["sealed_final"]))

    def test_independent_manifest_pin_and_admission_recomputation_precede_sgf_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs, ref, original = population_fixture(root)
            with mock.patch("kenggo.population_audit.load_original") as loader:
                with self.assertRaisesRegex(ValueError, "independently supplied pin"):
                    run(ref["path"], digest("wrong"), root / "wrong")
                loader.assert_not_called()
            path = Path(inputs["admission_decisions"]["path"])
            rows = pq.read_table(path).to_pylist()
            rows[0]["black_identity_valid"] = True
            pq.write_table(pa.Table.from_pylist(rows), path)
            inputs["admission_decisions"]["sha256"] = file_hash(path)
            ref = write_json(root / "inputs.json", inputs)
            with mock.patch("kenggo.population_audit.load_original", return_value=original), mock.patch("zipfile.ZipFile") as opened:
                with self.assertRaisesRegex(ValueError, "do not reproduce"):
                    run(ref["path"], ref["sha256"], root / "forged")
                opened.assert_not_called()

    def test_final_reservation_rejects_exposure_and_wrong_dates_without_opening_sgfs(self):
        for change in ("exposed_group", "wrong_date", "suspected_bot"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                inputs, ref, original = population_fixture(root)
                final_path = Path(inputs["sealed_final"]["path"])
                final = json.loads(final_path.read_text())
                scope_path = Path(inputs["final_scope_policy"]["path"])
                scope = json.loads(scope_path.read_text())
                split_path = Path(inputs["split_config"]["path"])
                split = json.loads(split_path.read_text())
                if change == "wrong_date":
                    final["records"][0]["source"]["source_url"] = "https://files.gokgs.com/games/2025/10/2/sealed.sgf"
                    final["records"][0]["archive_row"]["url"] = final["records"][0]["source"]["source_url"]
                else:
                    handle = "alice" if change == "exposed_group" else "MichiBot"
                    final["records"][0]["archive_row"]["black"] = handle + " [5k]"
                    reserved = sorted(set(scope["reserved_groups"] + ["kgs-handle:" + handle.lower()]))
                    final["test_groups"] = scope["reserved_groups"] = split["test_groups"] = reserved
                inputs["sealed_final"] = write_json(final_path, final)
                inputs["final_scope_policy"] = write_json(scope_path, scope)
                inputs["split_config"] = write_json(split_path, split)
                ref = write_json(root / "inputs.json", inputs)
                with mock.patch("kenggo.population_audit.load_original", return_value=original), mock.patch("zipfile.ZipFile") as opened:
                    with self.assertRaises(ValueError):
                        run(ref["path"], ref["sha256"], root / "rejected")
                    opened.assert_not_called()


if __name__ == "__main__":
    unittest.main()
