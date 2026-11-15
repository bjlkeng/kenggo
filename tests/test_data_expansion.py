from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from kenggo.canonical_data import digest, encoded
from kenggo.data_expansion import (ALLOWED, CEILINGS, CODE_PATHS, REQUEST_VERSION, BudgetStop,
                                  base_endings, base_fingerprints, build, file_hash,
                                  prepare_base_index, reference)
from kenggo.data_splits import ROLES, read_membership
from kenggo.examples import build_examples, load_corpus, load_dataset_profile, read_shards
from kenggo.kgs_admission import DECISION_PATH, DECISION_SHA256
from kenggo.kgs_eligibility import handle_groups
from kenggo.population_audit import run as build_population
from kenggo.supervised import TrainingData
from test_population_audit import population_fixture, profile_for, write_json


def fixture(root, definitions=None):
    inputs, ref, original = population_fixture(root)
    with mock.patch("kenggo.population_audit.load_original", return_value=original), redirect_stdout(io.StringIO()):
        summary = build_population(ref["path"], ref["sha256"], root / "base")
    profile = profile_for(root, summary)
    example_pin = build_examples(root / "base/canonical.parquet", profile.canonical_sha256,
        root / "base/splits", profile.split_sha256, root / "base/examples", profile=profile)
    split = json.loads((root / "base/splits/manifest.json").read_text())
    audits = original[0]
    audit_path = root / "original-audit.parquet"
    pq.write_table(pa.Table.from_pylist([{"source_url": row["source_url"],
        "sgf_sha256": row["sgf_sha256"], "audit_json": encoded(row)} for row in audits]), audit_path)
    groups = handle_groups(["alice", "bob", "held", "cu2", "carol", "dave", "unused"])
    prior = {"groups": sorted(original[2].groups), "source_ids": sorted(original[2].source_ids),
             "sgf_hashes": sorted(original[2].sgf_hashes), "sequence_hashes": sorted(original[2].sequence_hashes),
             "account_months": sorted(original[2].account_months)}
    owner_ref = write_json(root / "owner.json", {"excluded_games": []})
    prior_ref = write_json(root / "prior.json", prior)
    definitions = definitions or [
        ("newtrain", "alice", "bob", "2025-07-03", "kk", "ll", "B+R", "train"),
        ("newpopulation", "alice", "bob", "2025-09-03", "mm", "nn", "B+R", "population_development"),
        ("newvalidation", "held", "bob", "2025-09-04", "oo", "pp", "W+T", "validation_development")]
    candidates = []
    bundle = root / "new.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        for name, black, white, day, first, second, result, role in definitions:
            member = "games/" + day.replace("-", "/") + "/" + name + ".sgf"
            url = "https://files.gokgs.com/" + member
            data = (f"(;FF[4]GM[1]SZ[19]PB[{black}]PW[{white}]BR[5k]WR[1d]RU[Japanese]KM[6.5]"
                    f"TM[1200]OT[5x30 byo-yomi]DT[{day}]RE[{result}];B[{first}];W[{second}])").encode()
            archive.writestr(member, data)
            row = {"url": url, "black": black + " [5k]", "white": white + " [1d]", "type": "Ranked",
                   "setup": "19×19", "result": "B+Res." if result == "B+R" else "W+Time", "viewable": "Yes"}
            candidates.append({"archive_row": row, "provisional_role": role,
                               "source": {"source_url": url, "archive_path": str(bundle), "zip_member": member}})
    for item in candidates:
        item["source"]["archive_sha256"] = file_hash(bundle)
    sources = {"schema": "e2_expansion_sources_v1", "request_sha256": digest("source request"),
               "base_grouping_sha256": split["plan"]["grouping_sha256"], "groups": groups,
               "candidates": candidates, "owner_excluded_sgf_sha256": [],
               "exposure": {"groups": [groups[name] for name in ("alice", "bob", "held")],
                            "source_ids": [item["source"]["source_url"] for item in candidates],
                            "sgf_hashes": [], "sequence_hashes": [], "account_months": []},
               "provenance": {}, "counts": {}, "diagnostics": []}
    sources_ref = write_json(root / "sources.json", sources)
    request = {"schema": REQUEST_VERSION, "source_request_sha256": sources["request_sha256"],
               "refs": {"canonical": reference(root / "base/canonical.parquet"),
                        "split_manifest": reference(root / "base/splits/manifest.json"),
                        "examples_manifest": reference(root / "base/examples/manifest.json"),
                        "profile": reference(root / "profile.json"), "original_audit": reference(audit_path),
                        "owner_exclusions": owner_ref, "prior_exposure": prior_ref,
                        "source_policy": {"path": str(DECISION_PATH), "sha256": DECISION_SHA256}},
               "base_groups": groups, "split_config": split["plan"], "deadline_utc": "2099-01-01T00:00:00Z",
               "limits": dict(CEILINGS), "source_revision": {"commit": "synthetic",
                   "code_sha256": {path: file_hash(path) for path in CODE_PATHS}}}
    request_ref = write_json(root / "request.json", request)
    return request, request_ref, sources, sources_ref


def run(root, request_ref, sources_ref):
    return build(request_ref["path"], request_ref["sha256"], sources_ref["path"], sources_ref["sha256"], root / "out")


class DataExpansionTests(unittest.TestCase):
    def test_bootstrap_index_keeps_allowed_endings_and_rejects_missing_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request, req, sources, _ = fixture(root)
            bundle = Path(sources["candidates"][0]["source"]["archive_path"])
            with zipfile.ZipFile(bundle) as archive:
                payloads = {name: archive.read(name) for name in archive.namelist()}
            with zipfile.ZipFile(bundle, "w") as archive:
                for name, data in payloads.items():
                    archive.writestr(name, data.replace(b"TM[1200]", b"TM[30]") if "newtrain" in name else data)
            for item in sources["candidates"]:
                item["source"]["archive_sha256"] = file_hash(bundle)
            summary = run(root, req, write_json(root / "sources.json", sources))
            _, parts = read_membership(Path(summary["artifacts"]["split_manifest"]["path"]).parent,
                                      summary["artifacts"]["split_manifest"]["sha256"])
            rows = [row for part in parts.values() for row in part]
            loads = json.loads

            def no_canonical_targets(value, *args, **kwargs):
                if isinstance(value, str) and value.startswith('{"actions":'):
                    raise AssertionError("A canonical target was decoded")
                return loads(value, *args, **kwargs)

            with mock.patch("kenggo.data_expansion.json.loads", side_effect=no_canonical_targets):
                pinned = prepare_base_index(reference(root / "out/summary.json"), req, root / "index.json")
                base_request = {**request, "refs": {**request["refs"], **summary["artifacts"], "base_index": pinned}}
                endings = base_endings(base_request, rows)
            self.assertEqual(set(endings), {row["source_url"] for row in rows if row["role"] in ALLOWED})
            self.assertEqual(endings[sources["candidates"][2]["source"]["source_url"]]["result_kind"], "timeout")
            index = loads((root / "index.json").read_text())
            self.assertEqual(len(index["rejected"]), 1)
            self.assertEqual(len(index["records"]), len(rows))
            for case in ("missing_membership", "wrong_fingerprint", "missing_ending", "sealed_ending", "missing_rejection"):
                with self.subTest(case=case):
                    changed = deepcopy(index)
                    if case == "missing_membership":
                        changed["records"].pop()
                    elif case == "wrong_fingerprint":
                        changed["records"][0]["sgf_sha256"] = "0" * 64
                    elif case == "missing_ending":
                        next(row for row in changed["records"] if row["ending"] is not None)["ending"] = None
                    elif case == "sealed_ending":
                        next(row for row in changed["records"] if row["ending"] is None)["ending"] = next(iter(endings.values()))
                    else:
                        changed["rejected"] = []
                    broken = {**base_request, "refs": {**base_request["refs"],
                              "base_index": write_json(root / "broken-index.json", changed)}}
                    with self.assertRaises(ValueError):
                        base_endings(broken, rows)
            changed_summary = {**summary, "source_completed": summary["source_completed"] - 1}
            with self.assertRaisesRegex(ValueError, "Previous work omits"):
                prepare_base_index(write_json(root / "out/broken-summary.json", changed_summary), req,
                                   root / "broken-bootstrap.json")

    def test_second_expansion_preserves_the_expanded_base_and_prior_rejections(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_root = root / "first"
            first_root.mkdir()
            request, req, sources, _ = fixture(first_root, [
                ("accepted", "alice", "bob", "2025-07-03", "kk", "ll", "B+R", "train"),
                ("rejected", "alice", "bob", "2025-07-04", "mm", "nn", "B+R", "train")])
            prior = json.loads(Path(request["refs"]["prior_exposure"]["path"]).read_text())
            prior["sgf_hashes"].append(digest("inspected but unused bytes"))
            prior["account_months"].append(["unused", "2025-03"])
            request["refs"]["prior_exposure"] = write_json(first_root / "prior.json", prior)
            req = write_json(first_root / "request.json", request)
            sources["groups"] = {**sources["groups"], "newalias": "kgs-handle:newalias"}
            bundle = Path(sources["candidates"][0]["source"]["archive_path"])
            with zipfile.ZipFile(bundle) as archive:
                payloads = {name: archive.read(name) for name in archive.namelist()}
            with zipfile.ZipFile(bundle, "w") as archive:
                for name, data in payloads.items():
                    if "rejected" in name:
                        data = data[:-1].replace(b"TM[1200]", b"TM[30]") + b";B[tt];W[tt])"
                    archive.writestr(name, data)
            for item in sources["candidates"]:
                item["source"]["archive_sha256"] = file_hash(bundle)
            first = run(first_root, req, write_json(first_root / "sources.json", sources))
            self.assertEqual(first["new_encoded_games"], 1)
            first_index = json.loads(Path(first["artifacts"]["base_index"]["path"]).read_text())
            self.assertEqual(len(first_index["rejected"]), 1)

            second_root = root / "second"
            second_root.mkdir()
            second_request, _, second_sources, _ = fixture(second_root, [
                ("oldcopy", "alice", "bob", "2025-07-05", "kk", "ll", "B+R", "train"),
                ("rejectedcopy", "alice", "bob", "2025-07-06", "mm", "nn", "B+R", "train"),
                ("fresh", "alice", "bob", "2025-07-07", "oo", "pp", "B+R", "train")])
            bundle = Path(second_sources["candidates"][0]["source"]["archive_path"])
            with zipfile.ZipFile(bundle) as archive:
                payloads = {name: archive.read(name) for name in archive.namelist()}
            with zipfile.ZipFile(bundle, "w") as archive:
                for name, data in payloads.items():
                    archive.writestr(name, data[:-1] + b";B[];W[])" if "rejectedcopy" in name else data)
            for item in second_sources["candidates"]:
                item["source"]["archive_sha256"] = file_hash(bundle)
            for key in ("canonical", "split_manifest", "examples_manifest", "profile", "base_index"):
                second_request["refs"][key] = first["artifacts"][key]
            second_request["refs"]["prior_exposure"] = first["artifacts"]["exposure"]
            second_request["refs"]["original_audit"] = request["refs"]["original_audit"]
            second_request["base_groups"] = json.loads(Path(first["artifacts"]["groups"]["path"]).read_text())
            split, first_parts = read_membership(Path(first["artifacts"]["split_manifest"]["path"]).parent,
                                                first["artifacts"]["split_manifest"]["sha256"])
            second_request["split_config"] = split["plan"]
            second_sources["base_grouping_sha256"] = split["plan"]["grouping_sha256"]
            second_sources["groups"] = dict(second_request["base_groups"])
            prior_ids = {row["source_url"] for part in first_parts.values() for row in part}
            loads, np_load = json.loads, np.load

            def no_old_targets(value, *args, **kwargs):
                if isinstance(value, str) and value.startswith('{"actions":'):
                    if any('"source_url":' + json.dumps(key) in value for key in prior_ids):
                        raise AssertionError("An old canonical target was decoded")
                return loads(value, *args, **kwargs)

            def no_heldout_arrays(path, *args, **kwargs):
                if any(role in str(path) for role in ("final/", "confirmation/")):
                    raise AssertionError("A held-out array was decoded")
                return np_load(path, *args, **kwargs)

            with mock.patch("kenggo.data_expansion.json.loads", side_effect=no_old_targets), \
                    mock.patch("numpy.load", side_effect=no_heldout_arrays):
                second = run(second_root, write_json(second_root / "request.json", second_request),
                             write_json(second_root / "sources.json", second_sources))
            self.assertEqual(second["new_encoded_games"], 1)
            self.assertEqual(second["new_split_exclusions"]["duplicate_of_frozen_corpus"], 1)
            self.assertEqual(second["new_split_exclusions"]["duplicate_of_rejected_source"], 1)
            second_index = json.loads(Path(second["artifacts"]["base_index"]["path"]).read_text())
            self.assertEqual(second_index["rejected"], first_index["rejected"])
            indexed = {row["source_url"]: row for row in second_index["records"]}
            self.assertTrue(all(indexed[row["source_url"]] == row for row in first_index["records"]))
            _, second_parts = read_membership(Path(second["artifacts"]["split_manifest"]["path"]).parent,
                                              second["artifacts"]["split_manifest"]["sha256"])
            memberships = {row["source_url"]: row for part in second_parts.values() for row in part}
            self.assertTrue(all(memberships[row["source_url"]] == row for part in first_parts.values() for row in part))
            for role in ROLES:
                if role not in ALLOWED + ("excluded",):
                    self.assertEqual(first_parts[role], second_parts[role])
            first_examples = loads(Path(first["artifacts"]["examples_manifest"]["path"]).read_text())
            second_examples = loads(Path(second["artifacts"]["examples_manifest"]["path"]).read_text())
            directory = Path(second["artifacts"]["examples_manifest"]["path"]).parent
            for role in ROLES:
                self.assertEqual(second_examples["artifacts"][role][:len(first_examples["artifacts"][role])],
                                 first_examples["artifacts"][role])
                for item in first_examples["artifacts"][role]:
                    for kind in ("arrays", "metadata"):
                        self.assertEqual(file_hash(directory / item[kind]), item[kind + "_sha256"])
            normalizer = loads(Path(second["artifacts"]["normalizer"]["path"]).read_text())
            self.assertEqual(normalizer["count"][5], second["counts"]["combined"]["train"]["examples"])
            old_exposure = loads(Path(first["artifacts"]["exposure"]["path"]).read_text())
            new_exposure = loads(Path(second["artifacts"]["exposure"]["path"]).read_text())
            self.assertTrue(all(all(item in new_exposure[field] for item in values)
                                for field, values in old_exposure.items()))
            for case in ("missing_index", "missing_exposure"):
                with self.subTest(case=case):
                    broken = deepcopy(second_request)
                    if case == "missing_index":
                        del broken["refs"]["base_index"]
                    else:
                        dropped = deepcopy(old_exposure)
                        dropped["sgf_hashes"].remove(digest("inspected but unused bytes"))
                        broken["refs"]["prior_exposure"] = write_json(second_root / "dropped-exposure.json", dropped)
                    with mock.patch("zipfile.ZipFile") as opened:
                        with self.assertRaisesRegex(ValueError, "pinned metadata|prior inspection evidence"):
                            run(second_root, write_json(second_root / "broken-request.json", broken),
                                reference(second_root / "sources.json"))
                        opened.assert_not_called()

    def test_append_preserves_base_and_heldout_payloads_and_learner_reads_train(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request, request_ref, sources, sources_ref = fixture(root)
            old_manifest = json.loads((root / "base/examples/manifest.json").read_text())
            loads, np_load = json.loads, np.load

            def no_old_targets(value, *args, **kwargs):
                if isinstance(value, str) and value.startswith('{"actions":'):
                    if '"sealed_final":true' in value or any(name in value for name in ("/confirmation.sgf", "/train.sgf")):
                        raise AssertionError("An old target record was decoded")
                return loads(value, *args, **kwargs)

            def no_heldout_arrays(path, *args, **kwargs):
                if any(role in str(path) for role in ("final/", "confirmation/")):
                    raise AssertionError("A held-out array was decoded")
                return np_load(path, *args, **kwargs)

            with mock.patch("kenggo.data_expansion.json.loads", side_effect=no_old_targets), mock.patch("numpy.load", side_effect=no_heldout_arrays):
                summary = run(root, request_ref, sources_ref)
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["new_encoded_games"], 3)
            self.assertEqual(summary["counts"]["combined"]["train"]["examples"], 4)
            self.assertFalse(summary["minimum_met"])
            self.assertFalse(summary["final_targets_decoded"])
            self.assertFalse(summary["training_performed"])
            out = root / "out/dataset"
            manifest = json.loads((out / "examples/manifest.json").read_text())
            for role in ROLES:
                self.assertEqual(manifest["artifacts"][role][:len(old_manifest["artifacts"][role])], old_manifest["artifacts"][role])
                for item in old_manifest["artifacts"][role]:
                    for kind in ("arrays", "metadata"):
                        self.assertEqual(file_hash(out / "examples" / item[kind]), item[kind + "_sha256"])
            old_split, old_rows = read_membership(root / "base/splits", request["refs"]["split_manifest"]["sha256"])
            _, new_rows = read_membership(out / "splits", summary["artifacts"]["split_manifest"]["sha256"])
            by_id = {row["source_url"]: row for role in ROLES for row in new_rows[role]}
            self.assertTrue(all(by_id[row["source_url"]] == row for role in ROLES for row in old_rows[role]))
            data = TrainingData.load(out / "examples", summary["artifacts"]["examples_manifest"]["sha256"],
                                     out / "normalizer.json", summary["artifacts"]["normalizer"]["sha256"],
                                     split_sha256=summary["artifacts"]["split_manifest"]["sha256"])
            self.assertEqual(len(data), 4)
            arrays, metadata = list(read_shards(out / "examples", summary["artifacts"]["examples_manifest"]["sha256"],
                                                "validation_development"))[-1]
            self.assertFalse(arrays["value_valid"].any())
            self.assertFalse(arrays["duration_valid"].any())
            self.assertFalse(arrays["scalar_valid"][:, 12:].any())
            self.assertTrue(all(not row["identity_valid"] for row in metadata))
            profile = load_dataset_profile(out / "profile.json", summary["artifacts"]["profile"]["sha256"])
            selected = [sources["candidates"][0]["source"]["source_url"]]
            records, _, _ = load_corpus(out / "canonical.parquet", profile.canonical_sha256, out / "splits",
                                        profile.split_sha256, profile=profile, role="train", source_ids=selected)
            self.assertEqual(set(records), set(selected))

    def test_changed_date_copy_of_old_trajectory_is_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, req, _, src = fixture(root, [("copy", "alice", "bob", "2025-07-03", "aa", "bb", "B+R", "train")])
            summary = run(root, req, src)
            self.assertEqual(summary["new_encoded_games"], 0)
            self.assertEqual(summary["new_split_exclusions"]["duplicate_of_frozen_corpus"], 1)

    def test_inspected_but_unused_cached_hash_does_not_exclude_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request, _, sources, src = fixture(root)
            prior = json.loads(Path(request["refs"]["prior_exposure"]["path"]).read_text())
            item = sources["candidates"][0]
            with zipfile.ZipFile(item["source"]["archive_path"]) as archive:
                prior["sgf_hashes"].append(sha256(archive.read(item["source"]["zip_member"])).hexdigest())
            request["refs"]["prior_exposure"] = write_json(root / "prior.json", prior)
            req = write_json(root / "request.json", request)
            summary = run(root, req, src)
            self.assertEqual(summary["new_encoded_games"], 3)

    def test_row_cap_excludes_a_whole_game(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request, _, _, src = fixture(root)
            request["limits"]["max_train_examples"] = 3
            req = write_json(root / "request.json", request)
            summary = run(root, req, src)
            self.assertEqual(summary["counts"]["combined"]["train"]["examples"], 2)
            self.assertEqual(summary["new_split_exclusions"]["expansion_example_row_limit"], 1)
            self.assertEqual(summary["status"], "budget_stopped")
            self.assertEqual(summary["stop_reason"], "example_row_limit")
            self.assertEqual(summary["counts"]["new"]["population_development"]["examples"], 2)

    def test_invalid_new_copy_excludes_the_matching_admitted_game(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request, req, sources, _ = fixture(root, [
                ("valid", "alice", "bob", "2025-07-03", "kk", "ll", "B+R", "train"),
                ("badclock", "alice", "bob", "2025-07-03", "kk", "ll", "B+R", "train")])
            archive_path = Path(sources["candidates"][0]["source"]["archive_path"])
            with zipfile.ZipFile(archive_path) as archive:
                payloads = {name: archive.read(name) for name in archive.namelist()}
            with zipfile.ZipFile(archive_path, "w") as archive:
                for name, data in payloads.items():
                    archive.writestr(name, data.replace(b"TM[1200]", b"TM[30]") if "badclock" in name else data)
            for item in sources["candidates"]:
                item["source"]["archive_sha256"] = file_hash(archive_path)
            src = write_json(root / "sources.json", sources)
            summary = run(root, req, src)
            self.assertEqual(summary["source_exclusions"], {"source_policy": 1})
            self.assertEqual(summary["new_encoded_games"], 0)
            self.assertEqual(summary["new_split_exclusions"]["duplicate_of_rejected_source"], 1)

    def test_rejected_pass_alias_or_rank_header_copy_quarantines_the_component(self):
        for variant in ("pass_alias", "rank_header", "missing_rank", "empty_rank"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                request, req, sources, _ = fixture(root, [
                    ("valid", "alice", "bob", "2025-07-03", "kk", "ll", "B+R", "train"),
                    ("rejected", "alice", "bob", "2025-07-03", "kk", "ll", "B+R", "train")])
                bundle = Path(sources["candidates"][0]["source"]["archive_path"])
                with zipfile.ZipFile(bundle) as archive:
                    payloads = {name: archive.read(name) for name in archive.namelist()}
                with zipfile.ZipFile(bundle, "w") as archive:
                    for name, data in payloads.items():
                        data = data[:-1] + b";B[];W[])"
                        if "rejected" in name and variant == "pass_alias":
                            data = data.replace(b"TM[1200]", b"TM[30]").replace(b"B[]", b"B[tt]").replace(b"W[]", b"W[tt]")
                        if "rejected" in name and variant == "rank_header":
                            data = data.replace(b"BR[5k]", b"BR[4k]")
                        if "rejected" in name and variant == "missing_rank":
                            data = data.replace(b"BR[5k]", b"")
                        if "rejected" in name and variant == "empty_rank":
                            data = data.replace(b"BR[5k]", b"BR[]")
                        archive.writestr(name, data)
                for item in sources["candidates"]:
                    item["source"]["archive_sha256"] = file_hash(bundle)
                src = write_json(root / "sources.json", sources)
                summary = run(root, req, src)
                self.assertEqual(summary["new_encoded_games"], 0)
                self.assertEqual(summary["new_split_exclusions"]["duplicate_of_rejected_source"], 1)

    def test_conflicting_rank_never_opens_reserved_or_wrong_date_actions(self):
        for variant in ("reserved", "confirmation_date"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                request, req, sources, _ = fixture(root, [
                    ("blocked", "alice", "bob", "2025-07-03", "kk", "ll", "B+R", "train")])
                bundle = Path(sources["candidates"][0]["source"]["archive_path"])
                with zipfile.ZipFile(bundle) as archive:
                    payloads = {name: archive.read(name) for name in archive.namelist()}
                with zipfile.ZipFile(bundle, "w") as archive:
                    for name, data in payloads.items():
                        data = data.replace(b"BR[5k]", b"BR[4k]")
                        data = (data.replace(b"PB[alice]", b"PB[carol]") if variant == "reserved"
                                else data.replace(b"DT[2025-07-03]", b"DT[2025-11-03]"))
                        archive.writestr(name, data)
                for item in sources["candidates"]:
                    item["source"]["archive_sha256"] = file_hash(bundle)
                src = write_json(root / "sources.json", sources)
                with mock.patch("kenggo.data_expansion.canonicalize", side_effect=AssertionError("Forbidden actions opened")), \
                        mock.patch("kenggo.data_expansion.inspect_sgf", side_effect=AssertionError("Forbidden actions opened")):
                    summary = run(root, req, src)
                self.assertEqual(summary["new_encoded_games"], 0)
                self.assertEqual(summary["source_exclusions"], {"source_header_rejected": 1})

    def test_byte_stop_retains_only_complete_canonical_games(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request, _, sources, src = fixture(root)
            item = sources["candidates"][0]
            with zipfile.ZipFile(item["source"]["archive_path"]) as archive:
                request["limits"]["selected_sgf_bytes"] = archive.getinfo(item["source"]["zip_member"]).file_size
            req = write_json(root / "request.json", request)
            summary = run(root, req, src)
            self.assertEqual(summary["source_completed"], 1)
            self.assertEqual(summary["source_attempted"], 2)
            self.assertEqual(summary["source_unattempted"], 1)
            self.assertEqual(summary["new_encoded_games"], 1)
            self.assertEqual(summary["stop_reason"], "selected_sgf_budget")

    def test_output_limit_is_checked_before_large_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request, _, _, src = fixture(root)
            request["limits"]["output_bytes"] = 32 * 1024
            req = write_json(root / "request.json", request)
            with self.assertRaisesRegex(BudgetStop, "output_byte_budget"):
                run(root, req, src)
            self.assertLessEqual(sum(path.stat().st_size for path in (root / "out").rglob("*") if path.is_file()), 32 * 1024)
            self.assertFalse((root / "out/summary.json").exists())

    def test_canonical_file_limit_precedes_expensive_example_encoding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request, _, _, src = fixture(root)
            request["limits"]["canonical_file_bytes"] = 1
            req = write_json(root / "request.json", request)
            with mock.patch("kenggo.data_expansion._example_phase") as encoding:
                with self.assertRaisesRegex(ValueError, "canonical file exceeds"):
                    run(root, req, src)
                encoding.assert_not_called()

    def test_encoding_uses_frozen_source_order_when_a_role_fills(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request, _, sources, src = fixture(root, [
                ("selectedfirst", "alice", "bob", "2025-07-20", "kk", "ll", "B+R", "train"),
                ("earlierdate", "alice", "bob", "2025-07-03", "mm", "nn", "B+R", "train")])
            request["limits"]["max_train_examples"] = 4
            req = write_json(root / "request.json", request)
            summary = run(root, req, src)
            self.assertEqual(summary["review_source_ids"]["train"], [sources["candidates"][0]["source"]["source_url"]])

    def test_interrupted_canonical_phase_resumes_only_remaining_games_and_checks_saved_pins(self):
        from kenggo.data_expansion import _canonical_unit as actual
        for corrupt in (False, True):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                _, req, _, src = fixture(root)
                calls = []

                def interrupt_second(*args, **kwargs):
                    calls.append(args[0]["source"]["source_url"])
                    if len(calls) == 2:
                        raise RuntimeError("synthetic interruption")
                    return actual(*args, **kwargs)

                with mock.patch("kenggo.data_expansion._canonical_unit", side_effect=interrupt_second):
                    with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                        run(root, req, src)
                if corrupt:
                    path = root / "out/work/canonical/000000.json"
                    path.write_text(path.read_text() + " ")
                    with mock.patch("zipfile.ZipFile") as opened:
                        with self.assertRaisesRegex(ValueError, "Pinned input changed"):
                            run(root, req, src)
                        opened.assert_not_called()
                else:
                    with mock.patch("kenggo.data_expansion._canonical_unit", wraps=actual) as resumed:
                        summary = run(root, req, src)
                    self.assertEqual(resumed.call_count, 2)
                    self.assertEqual(summary["new_encoded_games"], 3)
                    self.assertFalse((root / "out/failure.json").exists())

    def test_budget_stop_during_a_game_never_publishes_a_partial_game(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, req, _, src = fixture(root)
            from kenggo.data_expansion import examples_for_game as actual

            def interrupted(*args, **kwargs):
                iterator = actual(*args, **kwargs)
                yield next(iterator)
                raise BudgetStop("synthetic_stop")

            with mock.patch("kenggo.data_expansion.examples_for_game", side_effect=interrupted):
                summary = run(root, req, src)
            self.assertEqual(summary["status"], "budget_stopped")
            self.assertEqual(summary["new_encoded_games"], 0)
            self.assertEqual(summary["counts"]["combined"]["train"]["examples"], 2)

    def test_completed_build_resumes_without_reopening_members_or_reencoding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, req, _, src = fixture(root)
            first = run(root, req, src)
            with mock.patch("kenggo.data_expansion._canonical_unit", side_effect=AssertionError("read twice")), \
                    mock.patch("kenggo.data_expansion.examples_for_game", side_effect=AssertionError("encoded twice")):
                second = run(root, req, src)
            for key in ("canonical", "split_manifest", "examples_manifest", "profile", "normalizer"):
                self.assertEqual(first["artifacts"][key]["sha256"], second["artifacts"][key]["sha256"])

    def test_changed_groups_reserved_candidates_and_bad_pins_precede_payload_access(self):
        for case in ("group", "reserved", "pin", "owner", "code"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                request, req, sources, src = fixture(root)
                if case == "group":
                    sources["groups"]["alice"] = "kgs-handle:other"
                elif case == "reserved":
                    sources["candidates"][0]["archive_row"]["black"] = "carol [5k]"
                elif case == "owner":
                    sources["owner_excluded_sgf_sha256"] = [digest("missing owner")]
                elif case == "code":
                    request["source_revision"]["code_sha256"]["src/kenggo/data_expansion.py"] = digest("changed code")
                    req = write_json(root / "request.json", request)
                else:
                    req["sha256"] = digest("wrong")
                src = write_json(root / "sources.json", sources)
                with mock.patch("zipfile.ZipFile") as opened:
                    with self.assertRaises(ValueError):
                        run(root, req, src)
                    opened.assert_not_called()


if __name__ == "__main__":
    unittest.main()
