from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np

from kenggo import data_growth as growth
from kenggo import data_growth_source
from kenggo import data_expansion as expansion
from kenggo.canonical_data import digest
from kenggo.data_splits import ROLES, read_membership
from kenggo.examples import (GROWTH_PROFILE_VERSION, DatasetProfile, load_dataset_profile,
                             load_corpus, read_shards)
from test_data_expansion import fixture, run
from test_population_audit import write_json


def parent_fixture(root):
    base = root / "base"
    base.mkdir()
    request, _, _, sources_ref = fixture(base, [
        ("capacity", "alice", "bob", "2025-07-03", "kk", "ll", "B+R", "train"),
        ("development_kept", "alice", "bob", "2025-09-03", "mm", "nn", "B+R", "population_development"),
        ("development_capacity", "alice", "bob", "2025-09-04", "ss", "qq", "B+R", "population_development")])
    request["limits"]["max_train_examples"] = 2
    request["limits"]["max_development_examples"] = 2
    run(base, write_json(base / "request.json", request), sources_ref)
    return expansion.reference(base / "out/summary.json")


def sources_for(root, parent_ref, definitions=()):
    parent = expansion.read_ref(parent_ref)
    groups = expansion.read_ref(parent["artifacts"]["groups"])
    split = expansion.read_ref(parent["artifacts"]["split_manifest"])
    exposure = expansion.read_ref(parent["artifacts"]["exposure"])
    candidates = []
    bundle = root / "fresh.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        for name, first, second, komi in definitions:
            day = "2025-07-09"
            member = "games/2025/7/9/" + name + ".sgf"
            key = "https://files.gokgs.com/" + member
            data = (f"(;FF[4]GM[1]SZ[19]PB[alice]PW[bob]BR[5k]WR[1d]RU[Japanese]KM[{komi}]"
                    f"TM[1200]OT[5x30 byo-yomi]DT[{day}]RE[B+R];B[{first}];W[{second}])").encode()
            archive.writestr(member, data)
            row = {"url": key, "black": "alice [5k]", "white": "bob [1d]", "type": "Ranked",
                   "setup": "19×19", "result": "B+Res.", "viewable": "Yes"}
            candidates.append({"archive_row": row, "provisional_role": "train", "source": {
                "source_url": key, "archive_path": str(bundle), "zip_member": member}})
            exposure["source_ids"].append(key)
    for item in candidates:
        item["source"]["archive_sha256"] = expansion.file_hash(bundle)
    sources = {"schema": growth.SOURCES_VERSION, "parent_summary": parent_ref,
        "base_grouping_sha256": split["plan"]["grouping_sha256"], "groups": groups,
        "candidates": candidates, "owner_excluded_sgf_sha256": [], "exposure": exposure, "provenance": {}}
    return sources, write_json(root / "sources.json", sources)


def request_for(parent, sources, target):
    return {"version": growth.REQUEST_VERSION, "parent_summary": parent, "sources": sources,
            "target_train_rows": target, "limits": {"cpu_seconds": 60, "wall_seconds": 60,
                "output_bytes": 64 * 1024**2, "max_records": 100, "canonical_bytes": 8 * 1024**2},
            "seed": "growth-fixture", "fixture_only": True}


class DataGrowthTests(unittest.TestCase):
    def test_promotions_and_mixed_additions_preserve_old_targets_and_protected_roles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent_ref = parent_fixture(root)
            parent = expansion.read_ref(parent_ref)
            old_split, old_parts = read_membership(Path(parent["artifacts"]["split_manifest"]["path"]).parent,
                                                   parent["artifacts"]["split_manifest"]["sha256"])
            sources, sources_ref = sources_for(root, parent_ref, [
                ("half", "oo", "pp", 0.5), ("six", "qq", "rr", 6.5)])
            request = request_for(parent_ref, sources_ref, 8)
            forbidden = {row["source_url"] for role, rows in old_parts.items()
                         for row in rows if not row["source_url"].endswith("/capacity.sgf")}
            loads, np_load = json.loads, np.load

            def no_protected_targets(value, *args, **kwargs):
                if isinstance(value, str) and value.startswith('{"actions":'):
                    if any('"source_url":' + json.dumps(key) in value for key in forbidden):
                        raise AssertionError("An old unselected canonical target was decoded")
                return loads(value, *args, **kwargs)

            def no_protected_arrays(path, *args, **kwargs):
                if any(part in str(path) for part in ("final/", "confirmation/")):
                    raise AssertionError("Protected arrays were decoded")
                return np_load(path, *args, **kwargs)

            with mock.patch("json.loads", side_effect=no_protected_targets), mock.patch("numpy.load", side_effect=no_protected_arrays):
                summary = growth.build_growth_round(request, root / "out")
            self.assertEqual(summary["actual_train_rows"], 8)
            self.assertEqual(len(summary["promoted_source_ids"]), 1)
            self.assertEqual(summary["fresh_encoded_games"], 2)
            new_split, new_parts = read_membership(Path(summary["artifacts"]["split_manifest"]["path"]).parent,
                                                   summary["artifacts"]["split_manifest"]["sha256"])
            for role in set(ROLES) - {"train", "excluded"}:
                self.assertEqual(old_parts[role], new_parts[role])
                self.assertEqual(old_split["membership_sha256"][role + ".parquet"],
                                 new_split["membership_sha256"][role + ".parquet"])
            self.assertTrue(any(row["source_url"].endswith("development_capacity.sgf") for row in new_parts["excluded"]))
            old_manifest = expansion.read_ref(parent["artifacts"]["examples_manifest"])
            new_manifest = expansion.read_ref(summary["artifacts"]["examples_manifest"])
            for role in ROLES:
                self.assertEqual(old_manifest["artifacts"][role],
                                 new_manifest["artifacts"][role][:len(old_manifest["artifacts"][role])])
            arrays = list(read_shards(Path(summary["artifacts"]["examples_manifest"]["path"]).parent,
                                     summary["artifacts"]["examples_manifest"]["sha256"], "train", komi_study=True))
            self.assertEqual({float(value) for data, _ in arrays for value in np.abs(data["scalars"][:, 1])}, {0.5, 6.5})
            self.assertEqual(expansion.read_ref(summary["artifacts"]["normalizer"])["count"][5], 8)
            profile_ref = summary["artifacts"]["profile"]
            profile = load_dataset_profile(profile_ref["path"], profile_ref["sha256"])
            selected = [new_parts["train"][0]["source_url"]]
            loaded, _, _ = load_corpus(summary["artifacts"]["canonical"]["path"],
                summary["artifacts"]["canonical"]["sha256"],
                Path(summary["artifacts"]["split_manifest"]["path"]).parent,
                summary["artifacts"]["split_manifest"]["sha256"], profile=profile, role="train", source_ids=selected)
            self.assertEqual(set(loaded), set(selected))
            self.assertEqual(expansion.file_hash(parent_ref["path"]), parent_ref["sha256"])

    def test_rounds_keep_unattempted_candidates_and_whole_game_caps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = parent_fixture(root)
            _, sources = sources_for(root, parent, [("fresh", "oo", "pp", 0.5)])
            first = growth.build_growth_round(request_for(parent, sources, 5), root / "first")
            self.assertEqual(first["actual_train_rows"], 4)
            self.assertEqual(first["target_shortfall"], 1)
            self.assertEqual(first["new_split_exclusions"], {growth.CAP_REASON: 1})
            later = root / "later"
            later.mkdir()
            next_parent = expansion.reference(root / "first/summary.json")
            _, next_sources = sources_for(later, next_parent)
            second = growth.build_growth_round(request_for(next_parent, next_sources, 6), root / "second")
            self.assertEqual(second["actual_train_rows"], 6)
            self.assertEqual(second["source_completed"], 0)
            self.assertEqual(len(second["promoted_source_ids"]), 1)
            self.assertEqual(expansion.read_ref(first["artifacts"]["base_index"])["processed_source_ids"],
                             expansion.read_ref(second["artifacts"]["base_index"])["processed_source_ids"])
            with self.assertRaises(FileExistsError):
                growth.build_growth_round(request_for(next_parent, next_sources, 8), root / "second")

    def test_duplicate_copy_is_excluded_and_rejection_history_survives(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = parent_fixture(root)
            sources, source_ref = sources_for(root, parent, [
                ("copy", "aa", "bb", 6.5), ("badclock", "oo", "pp", 6.5), ("fresh", "qq", "rr", 0.5)])
            bundle = root / "fresh.zip"
            with zipfile.ZipFile(bundle) as archive:
                payloads = {name: archive.read(name) for name in archive.namelist()}
            with zipfile.ZipFile(bundle, "w") as archive:
                for name, payload in payloads.items():
                    archive.writestr(name, payload.replace(b"TM[1200]", b"TM[30]") if "badclock" in name else payload)
            for item in sources["candidates"]:
                item["source"]["archive_sha256"] = expansion.file_hash(bundle)
            source_ref = write_json(root / "sources.json", sources)
            first = growth.build_growth_round(request_for(parent, source_ref, 8), root / "first")
            self.assertEqual(first["actual_train_rows"], 6)
            self.assertEqual(first["new_split_exclusions"]["duplicate_of_frozen_corpus"], 1)
            self.assertEqual(first["source_exclusions"], {"source_policy": 1})
            index = expansion.read_ref(first["artifacts"]["base_index"])
            self.assertEqual(len(index["rejected"]), 1)
            next_root = root / "next"
            next_root.mkdir()
            parent2 = expansion.reference(root / "first/summary.json")
            _, sources2 = sources_for(next_root, parent2, [("rejectedcopy", "oo", "pp", 6.5)])
            second = growth.build_growth_round(request_for(parent2, sources2, 8), root / "second")
            self.assertEqual(second["actual_train_rows"], 6)
            self.assertEqual(second["new_split_exclusions"]["duplicate_of_rejected_source"], 1)
            self.assertEqual(expansion.read_ref(second["artifacts"]["base_index"])["rejected"], index["rejected"])

    def test_forbidden_cohort_metadata_stops_before_zip_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = parent_fixture(root)
            sources, _ = sources_for(root, parent, [("fresh", "oo", "pp", 6.5)])
            for name in ("held", "carol"):
                changed = deepcopy(sources)
                changed["candidates"][0]["archive_row"]["black"] = name + " [5k]"
                pin = write_json(root / (name + ".json"), changed)
                with mock.patch("kenggo.data_growth.zipfile.ZipFile") as opened:
                    with self.assertRaisesRegex(ValueError, "cohort"):
                        growth.build_growth_round(request_for(parent, pin, 6), root / name)
                    opened.assert_not_called()

    def test_header_identity_conflict_stops_before_move_parsing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = parent_fixture(root)
            sources, _ = sources_for(root, parent, [("fresh", "oo", "pp", 6.5)])
            bundle = root / "fresh.zip"
            with zipfile.ZipFile(bundle) as archive:
                name = archive.namelist()[0]
                payload = archive.read(name).replace(b"PB[alice]", b"PB[carol]")
            with zipfile.ZipFile(bundle, "w") as archive:
                archive.writestr(name, payload)
            sources["candidates"][0]["source"]["archive_sha256"] = expansion.file_hash(bundle)
            pin = write_json(root / "sources.json", sources)
            with mock.patch("kenggo.data_growth.canonicalize") as replay, mock.patch("kenggo.data_growth.inspect_sgf") as inspect:
                result = growth.build_growth_round(request_for(parent, pin, 6), root / "out")
            replay.assert_not_called()
            inspect.assert_not_called()
            self.assertEqual(result["source_exclusions"], {"source_header_rejected": 1})

    def test_corrupt_member_is_retained_as_exclusion_and_following_game_builds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = parent_fixture(root)
            _, pin = sources_for(root, parent, [("broken", "oo", "pp", 6.5), ("good", "qq", "rr", 0.5)])
            read = zipfile.ZipFile.read

            def corrupt(archive, member, *args, **kwargs):
                name = member.filename if isinstance(member, zipfile.ZipInfo) else member
                if name.endswith("broken.sgf"):
                    raise zipfile.BadZipFile("synthetic CRC error")
                return read(archive, member, *args, **kwargs)

            with mock.patch("zipfile.ZipFile.read", new=corrupt):
                result = growth.build_growth_round(request_for(parent, pin, 6), root / "out")
            self.assertEqual(result["actual_train_rows"], 6)
            self.assertEqual(result["source_attempted"], 2)
            self.assertEqual(result["source_exclusions"], {"selected_member_read_error": 1})
            self.assertGreater(result["costs"]["selected_sgf_bytes"], 0)
            rejected = expansion.read_ref(result["artifacts"]["base_index"])["rejected"]
            self.assertEqual(rejected[0]["reason"], "selected_member_read_error")

    def test_only_exact_capacity_exclusions_can_be_requested_for_promotion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = parent_fixture(root)
            sources, _ = sources_for(root, parent)
            sources["promotion_source_ids"] = ["https://files.gokgs.com/games/2025/9/4/development_capacity.sgf"]
            pin = write_json(root / "sources.json", sources)
            with self.assertRaisesRegex(ValueError, "capacity-limited training"):
                growth.build_growth_round(request_for(parent, pin, 6), root / "out")

    def test_budget_callback_cannot_be_bypassed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = parent_fixture(root)
            _, sources = sources_for(root, parent)
            with self.assertRaisesRegex(RuntimeError, "shared deadline"):
                growth.build_growth_round(request_for(parent, sources, 6), root / "out",
                    check_budget=mock.Mock(side_effect=RuntimeError("shared deadline")))
            self.assertFalse((root / "out/summary.json").exists())
            self.assertEqual(json.loads((root / "out/progress.json").read_text())["status"], "failed")

    def test_interrupted_fresh_attempt_is_recorded_as_attempted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = parent_fixture(root)
            sources, pin = sources_for(root, parent, [("fresh", "oo", "pp", 6.5)])
            with mock.patch("kenggo.data_growth._fresh_unit", side_effect=growth.PreparationStop("growth_preparation_budget")):
                summary = growth.build_growth_round(request_for(parent, pin, 6), root / "out")
            key = sources["candidates"][0]["source"]["source_url"]
            self.assertEqual(summary["actual_train_rows"], 4)
            self.assertEqual(summary["source_attempted"], 1)
            self.assertEqual(summary["source_completed"], 0)
            self.assertEqual(summary["source_unattempted"], 0)
            self.assertEqual(summary["source_interrupted_ids"], [key])
            index = expansion.read_ref(summary["artifacts"]["base_index"])
            self.assertIn(key, index["processed_source_ids"])
            self.assertEqual(index["rejected"][0]["reason"], "growth_preparation_interrupted")

    def test_cached_source_adapter_output_builds_the_next_round(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = parent_fixture(root)
            _, empty = sources_for(root, original)
            growth.build_growth_round(request_for(original, empty, 4), root / "first")
            parent = expansion.reference(root / "first/summary.json")
            source_dir = root / "cache"
            source_dir.mkdir()
            sources, _ = sources_for(source_dir, parent, [("fresh", "oo", "pp", 0.5)])
            row = sources["candidates"][0]["archive_row"]
            row.update(page_key="alice-2025-07", start="7/9/25 1:00 PM",
                       bundle_url="https://www.gokgs.com/servlet/archives/en_US/alice-2025-07.zip")
            prior = write_json(source_dir / "sources.json", sources)
            frozen = data_growth_source.freeze_sources(parent, [], [prior], root / "frozen.json", seed="growth-fixture")
            pinned = {key: frozen[key] for key in ("path", "sha256")}
            result = growth.build_growth_round(request_for(parent, pinned, 6), root / "second")
            self.assertEqual(result["actual_train_rows"], 6)
            self.assertEqual(result["fresh_encoded_games"], 1)


class GrowthProfileTests(unittest.TestCase):
    def test_new_profile_owns_its_allowance_and_old_profiles_cannot_override(self):
        base = dict(sha256="a" * 64, canonical_sha256="b" * 64, split_sha256="c" * 64,
                    population_inputs_sha256="d" * 64, sealed_source_ids_sha256="e" * 64,
                    max_records=100, max_examples_per_role=1000, cpu_seconds=60, wall_seconds=60)
        self.assertTrue(DatasetProfile(**base, version=GROWTH_PROFILE_VERSION, canonical_bytes=256 * 1024**2).komi_study)
        with self.assertRaisesRegex(ValueError, "Historical"):
            DatasetProfile(**base, canonical_bytes=256 * 1024**2)
        with self.assertRaisesRegex(ValueError, "bounded"):
            DatasetProfile(**base, version=GROWTH_PROFILE_VERSION)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            profile = {"version": GROWTH_PROFILE_VERSION,
                **{key: value for key, value in base.items() if key.endswith("sha256") and key != "sha256"},
                "limits": {key: base[key] for key in ("max_records", "max_examples_per_role", "cpu_seconds", "wall_seconds")}}
            profile["limits"]["canonical_bytes"] = 256 * 1024**2
            pin = write_json(path, profile)
            self.assertEqual(load_dataset_profile(pin["path"], pin["sha256"]).canonical_bytes, 256 * 1024**2)


if __name__ == "__main__":
    unittest.main()
