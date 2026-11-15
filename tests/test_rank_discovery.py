from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from kenggo import rank_discovery as discovery
from kenggo.kgs_source import RANKS


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(body)
    return {"path": str(path), "sha256": sha256(body).hexdigest()}


def archive_row(name, black="Alice", white="Betty", rank="15k", month=3, year=2025, **extra):
    return {"url": f"https://files.gokgs.com/games/{year}/{month}/1/{name}.sgf",
        "black": f"{black} [{rank}]", "white": f"{white} [{rank}]",
        "type": "Ranked", "setup": "19×19", "result": "B+R", **extra}


class SelectionFixture:
    def __init__(self, directory, *, pages=None, validation=2):
        self.root = Path(directory)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.groups = {f"valid{index}": f"kgs-handle:valid{index}" for index in range(validation)}
        self.groups.update(final="kgs-handle:final")
        self.split = {"train_before": "2025-09-01", "confirmation_from": "2025-11-01",
            "validation_groups": [group for handle, group in self.groups.items() if handle != "final"],
            "test_groups": ["kgs-handle:final"]}
        counts = {"train_games_by_rank": dict.fromkeys(RANKS, 0),
                  "population_development_game_sides_by_rank": dict.fromkeys(RANKS, 0)}
        provenance = save(self.root / "dummy.json", {})
        self.evidence = {"schema": discovery.COUNTS_SCHEMA, "baseline_counts": counts,
            "provenance": {key: provenance for key in ("current_base", "examples_manifest", "split_manifest")}}
        self.policy = {"schema": discovery.POLICY_SCHEMA, "seed": discovery.SEED,
            "page_counts": pages or {"train": 4, "population_development": 2, "validation_development": 2},
            "target_train_games_per_rank": 200, "target_population_development_sides_per_rank": 100,
            "baseline_counts": counts}
        self.frame = {"rows": [archive_row("safe")]}
        self.pin_counts()

    def pin_counts(self):
        self.policy["baseline_counts_ref"] = save(self.root / "counts.json", self.evidence)

    def run(self):
        return discovery.queries(self.frame, self.groups, self.split, self.cache, self.policy)


class QueryTests(unittest.TestCase):
    def test_dates_cohorts_aliases_and_bots_guard_every_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = SelectionFixture(directory)
            fixture.groups.update(finalalias="kgs-handle:final", validalias="kgs-handle:valid0",
                                  gnugo2="kgs-handle:gnugo2", cobalt="kgs-handle:gnugo2")
            fixture.frame["rows"] = [archive_row("safe", setup="19×19 H4")]
            fixture.frame["rows"].extend([
                archive_row("final", black="FinalAlias", white="FinalPeer"),
                archive_row("valid", black="ValidAlias", white="ValidPeer"),
                archive_row("bot", black="SomeBot", white="BotPeer"),
                archive_row("bot_alias", black="Cobalt", white="CobaltPeer"),
                archive_row("later", black="Later", white="LaterPeer", month=9),
                archive_row("confirmation", black="Secret", white="SecretPeer", month=11),
                archive_row("prior_year", black="Prior", white="PriorPeer", year=2024),
                archive_row("small", black="Small", white="SmallPeer", setup="9×9"),
                archive_row("free", black="Free", white="FreePeer", type="Free"),
            ])
            output = fixture.run()
            for item in output:
                if item["role"] == "validation_development":
                    self.assertIn(fixture.groups[item["user"]], fixture.split["validation_groups"])
                    self.assertIn(item["month"], (9, 10))
                else:
                    self.assertIn(item["user"], {"alice", "betty"})
                    self.assertIn(item["month"], range(1, 9) if item["role"] == "train" else (9, 10))
                self.assertTrue(item["url"].startswith("https://www.gokgs.com/gameArchives.jsp?"))

    def test_deterministic_page_order_ignores_results_and_frame_order(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = SelectionFixture(directory)
            fixture.frame["rows"].extend([archive_row("more", black="Carol", white="Donna", rank="14k"),
                                          archive_row("safe")])
            before = fixture.run()
            fixture.frame["rows"].reverse()
            for row in fixture.frame["rows"]:
                row["result"] = "unknown"
            self.assertEqual(fixture.run(), before)
            self.assertEqual(len({item["key"] for item in before}), len(before))

    def test_existing_pages_are_skipped_case_insensitively_without_reading_them(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = SelectionFixture(directory)
            (fixture.cache / "ALIce-2025-03.html").write_bytes(b"not inspected")
            (fixture.cache / "VALID0-2025-09.HTML").write_bytes(b"not inspected")
            output = fixture.run()
            self.assertNotIn(("alice", 3), {(item["user"], item["month"]) for item in output})
            self.assertNotIn(("valid0", 9), {(item["user"], item["month"]) for item in output})

    def test_rank_gaps_prioritize_requests_and_accounts_precede_repeat_months(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = SelectionFixture(directory, pages={"train": 24, "population_development": 0,
                                                         "validation_development": 0})
            fixture.policy["baseline_counts"]["train_games_by_rank"].update(dict.fromkeys(RANKS, 200))
            fixture.policy["baseline_counts"]["train_games_by_rank"].update({"15k": 190, "14k": 199})
            fixture.pin_counts()
            fixture.frame["rows"] = [archive_row(f"g{rank}{i}", black=f"a{rank}n{i}", white=f"b{rank}n{i}", rank=rank)
                for rank in ("15k", "14k", "4k") for i in range(20)]
            output = fixture.run()
            counts = Counter(item["discovery_rank"] for item in output)
            self.assertLessEqual(abs(counts["15k"] - 10 * counts["14k"]), 10)
            self.assertEqual(counts["4k"], 0)
            self.assertEqual(len({item["user"] for item in output}), len(output))
            self.assertTrue(all(item["month"] == 3 for item in output))

    def test_fixed_1000_page_plan_has_balanced_400_page_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = SelectionFixture(directory, validation=140, pages={"train": 750,
                "population_development": 110, "validation_development": 140})
            fixture.frame["rows"] = [archive_row(f"g{rank}{i}", black=f"a{rank}n{i}", white=f"b{rank}n{i}", rank=rank)
                for rank in RANKS for i in range(35)]
            original = deepcopy((fixture.frame, fixture.groups, fixture.split, fixture.policy))
            output = fixture.run()
            self.assertEqual(Counter(item["role"] for item in output), fixture.policy["page_counts"])
            self.assertEqual(Counter(item["role"] for item in output[:400]),
                {"train": 300, "population_development": 44, "validation_development": 56})
            validation = [item for item in output if item["role"] == "validation_development"]
            self.assertEqual(len({item["user"] for item in validation}), 140)
            self.assertEqual({item["month"] for item in output if item["role"] == "population_development"}, {9, 10})
            fixture.policy["page_counts"] = {"train": 300, "population_development": 44,
                                              "validation_development": 56}
            self.assertEqual(fixture.run(), output[:400])
            fixture.policy["page_counts"] = original[3]["page_counts"]
            self.assertEqual((fixture.frame, fixture.groups, fixture.split, fixture.policy), original)

    def test_counts_and_policy_changes_require_matching_pinned_evidence(self):
        for change in ("hash", "counts", "missing_rank", "negative", "boolean", "pages", "target", "seed", "provenance"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                fixture = SelectionFixture(directory)
                if change == "hash":
                    (fixture.root / "counts.json").write_text("{}")
                elif change == "counts":
                    fixture.policy["baseline_counts"]["train_games_by_rank"]["15k"] = 1
                elif change == "missing_rank":
                    del fixture.policy["baseline_counts"]["train_games_by_rank"]["15k"]
                elif change in {"negative", "boolean"}:
                    fixture.policy["baseline_counts"]["train_games_by_rank"]["15k"] = -1 if change == "negative" else True
                    fixture.pin_counts()
                elif change == "pages":
                    fixture.policy["page_counts"]["train"] = 1151
                elif change == "target":
                    fixture.policy["target_train_games_per_rank"] = 0
                elif change == "seed":
                    fixture.policy["seed"] = "replacement"
                else:
                    fixture.evidence["provenance"] = {}
                    fixture.pin_counts()
                with self.assertRaises(ValueError):
                    fixture.run()

    def test_inconsistent_rank_metadata_and_exhausted_frames_are_not_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = SelectionFixture(directory)
            fixture.frame["rows"].append(archive_row("safe", rank="14k"))
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                fixture.run()
            fixture.frame["rows"] = []
            with self.assertRaisesRegex(ValueError, "cannot fill"):
                fixture.run()


class BaselineFixture:
    def __init__(self, directory):
        self.root = Path(directory)
        self.directory = self.root / "examples"
        self.artifacts, self.counts, self.games = {}, {}, {}
        for role, month, players in [("train", 1, [("same", "B", "5k"), ("same", "B", "5k"),
            ("same", "W", "5k"), ("mixed", "B", "5k"), ("mixed", "W", "7k"), ("opening", "B", "15k")]),
            ("population_development", 9, [("same", "B", "5k"), ("same", "W", "5k"),
                ("mixed", "B", "5k"), ("mixed", "W", "7k")])]:
            rows = [{"source_url": f"https://files.gokgs.com/games/2025/{month}/1/{source}.sgf",
                "role": role, "mover": mover, "source_rank_mover": rank} for source, mover, rank in players]
            self.artifacts[role] = []
            for index, part in enumerate((rows[:2], rows[2:])):
                path = self.directory / role / f"{index:05d}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.Table.from_pylist(part), path)
                self.artifacts[role].append({"metadata": str(path.relative_to(self.directory)),
                    "metadata_sha256": sha256(path.read_bytes()).hexdigest(), "rows": len(part),
                    "arrays": f"{role}/forbidden.npz", "arrays_sha256": "0" * 64})
            self.counts[role] = {"examples": len(rows)}
            self.games[role] = len({row["source_url"] for row in rows})
        self.artifacts["final"] = [{"metadata": "final/forbidden.parquet", "arrays": "final/forbidden.npz"}]
        self.split_ref = save(self.root / "splits.json", {})
        self.manifest = {"version": "supervised_examples_v1", "artifacts": self.artifacts,
            "counts": self.counts, "split_sha256": self.split_ref["sha256"]}
        self.summary = {"schema": "e2_data_expansion_v1", "artifacts": {"split_manifest": self.split_ref,
            "canonical": {"path": str(self.root / "forbidden.parquet"), "sha256": "0" * 64}},
            "counts": {"combined": {role: {**counts, "games": self.games[role]} for role, counts in self.counts.items()}}}
        self.pin()

    def pin(self):
        self.summary["artifacts"]["examples_manifest"] = save(self.directory / "manifest.json", self.manifest)
        self.summary_ref = save(self.root / "summary.json", self.summary)

    def run(self):
        reference = discovery.freeze_baseline_counts(self.summary_ref, self.root / "baseline-counts.json")
        self.assert_hash = sha256(Path(reference["path"]).read_bytes()).hexdigest()
        return reference, json.loads(Path(reference["path"]).read_bytes())


class BaselineCountTests(unittest.TestCase):
    def test_counts_distinct_active_training_games_and_development_sides(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = BaselineFixture(directory)
            with patch.object(discovery.pq, "read_table", wraps=pq.read_table) as read:
                reference, result = fixture.run()
            self.assertEqual(reference["sha256"], fixture.assert_hash)
            self.assertEqual(result["baseline_counts"]["train_games_by_rank"],
                {**dict.fromkeys(RANKS, 0), "5k": 2, "7k": 1, "15k": 1})
            self.assertEqual(result["baseline_counts"]["population_development_game_sides_by_rank"],
                {**dict.fromkeys(RANKS, 0), "5k": 3, "7k": 1})
            self.assertEqual(result["provenance"]["current_base"], fixture.summary_ref)
            self.assertEqual(read.call_count, 4)
            for call in read.call_args_list:
                self.assertIn(Path(call.args[0]).parent.name, {"train", "population_development"})
                self.assertEqual(call.kwargs["columns"], ["source_url", "role", "mover", "source_rank_mover"])

    def test_modified_summary_manifest_or_metadata_is_rejected(self):
        for artifact in ("summary", "manifest", "metadata"):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as directory:
                fixture = BaselineFixture(directory)
                path = (Path(fixture.summary_ref["path"]) if artifact == "summary" else
                    fixture.directory / "manifest.json" if artifact == "manifest" else
                    fixture.directory / fixture.artifacts["train"][0]["metadata"])
                with path.open("ab") as stream:
                    stream.write(b" ")
                with self.assertRaisesRegex(ValueError, "hash changed"):
                    fixture.run()

    def test_metadata_cannot_redirect_to_a_held_out_role(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = BaselineFixture(directory)
            fixture.artifacts["train"][0]["metadata"] = "final/forbidden.parquet"
            fixture.pin()
            with self.assertRaisesRegex(ValueError, "allowed role"):
                fixture.run()

    def test_inconsistent_baseline_counts_or_mover_rank_fail(self):
        for error in ("games", "rows", "rank"):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                fixture = BaselineFixture(directory)
                if error == "games":
                    fixture.summary["counts"]["combined"]["train"]["games"] += 1
                elif error == "rows":
                    fixture.artifacts["train"][0]["rows"] += 1
                else:
                    shard = fixture.artifacts["train"][0]
                    path = fixture.directory / shard["metadata"]
                    rows = pq.read_table(path).to_pylist()
                    rows[1]["source_rank_mover"] = "6k"
                    pq.write_table(pa.Table.from_pylist(rows), path)
                    shard["metadata_sha256"] = sha256(path.read_bytes()).hexdigest()
                fixture.pin()
                with self.assertRaises(ValueError):
                    fixture.run()


if __name__ == "__main__":
    unittest.main()
