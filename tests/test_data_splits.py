import copy
from pathlib import Path
import tempfile
import unittest

import pyarrow.parquet as pq

from kenggo.canonical_data import digest
from kenggo.data_splits import (ROLES, SplitPlan, build_splits, duplicate_components,
                               manifest_contents, read_membership, write_membership)


GROUPING = digest("fixture grouping")


def entry(name, groups=("a", "b"), day="2025-07-01", *, admitted=True, identities=(True, True),
          trajectory=None, raw=None, exposed=False, exclusions=(), actions=10):
    trajectory = trajectory or name
    return {"source_url": name, "canonical_sha256": digest(name), "grouping_sha256": GROUPING,
            "sgf_sha256": digest(raw or name), "audit_sequence_sha256": digest("raw actions:" + trajectory),
            "trajectory_sha256": digest(trajectory),
            "semantic_sha256": digest([groups, day, trajectory]),
            "groups": list(groups), "date": day, "admitted": admitted,
            "identity_valid": list(identities), "final_excluded": exposed,
            "exclusions": list(exclusions), "actions": actions,
            "actions_by_color": [(actions + 1) // 2, actions // 2], "value_supported": True}


def plan(validation=(), test=()):
    return SplitPlan(GROUPING, "2025-09-01", "2025-11-01", validation, test)


def by_id(rows):
    return {r["source_url"]: r for r in rows}


class SplitTests(unittest.TestCase):
    def test_both_colors_cohort_bridges_and_whole_day_roles(self):
        entries = [entry("train"), entry("val-prefix", ("v", "a")), entry("test-prefix", ("a", "t")),
                   entry("v-dev", ("a", "v"), "2025-09-01"), entry("v-conf", ("v", "a"), "2025-11-01"),
                   entry("test-gap", ("t", "a"), "2025-10-31"), entry("test-final", ("a", "t"), "2025-11-01"),
                   entry("bridge-early", ("v", "t")), entry("bridge-late", ("t", "v"), "2025-11-01"),
                   entry("dev1", day="2025-09-01"), entry("dev2", day="2025-09-01"),
                   entry("last-train", day="2025-08-31"), entry("confirm", day="2025-11-01")]
        rows = by_id(build_splits(entries, plan(("v",), ("t",)), exposed_groups=[]))
        expected = {"train": "train", "val-prefix": "validation_prefix", "test-prefix": "test_prefix",
                    "v-dev": "validation_development", "v-conf": "validation_confirmation",
                    "test-gap": "embargo", "test-final": "final", "bridge-early": "excluded", "bridge-late": "excluded",
                    "dev1": "population_development", "dev2": "population_development",
                    "last-train": "train", "confirm": "population_confirmation"}
        self.assertEqual({k: r["role"] for k, r in rows.items()}, expected)
        self.assertEqual(rows["val-prefix"]["adaptation_colors"], ["B"])
        self.assertEqual(rows["test-prefix"]["adaptation_colors"], ["W"])
        self.assertEqual(rows["v-conf"]["adaptation_colors"], [])
        self.assertEqual(rows["v-dev"]["adaptation_colors"], [])
        for row in rows.values():
            if "t" in row["groups"]:
                self.assertNotIn(row["role"], {"train", "population_development", "validation_development"})
            if "v" in row["groups"]:
                self.assertNotEqual(row["role"], "train")

    def test_known_personas_require_training_and_missing_identity_stays_default(self):
        entries = [entry("train", identities=(True, False)),
                   entry("later", ("a", "new"), "2025-10-01"),
                   entry("masked", ("v", "a"), identities=(False, False)),
                   entry("eval", ("a", "v"), "2025-11-01", identities=(False, False))]
        rows = by_id(build_splits(entries, plan(("v",)), exposed_groups=[]))
        self.assertEqual(rows["later"]["known_persona_colors"], ["B"])
        self.assertEqual(rows["masked"]["role"], "validation_prefix")
        self.assertEqual(rows["masked"]["adaptation_colors"], [])
        self.assertNotEqual(rows["masked"]["groups"][0], rows["masked"]["groups"][1])
        self.assertEqual(rows["eval"]["known_persona_colors"], [])

    def test_prefix_actions_belong_only_to_the_supported_target_side(self):
        entries = [entry("single", actions=1), entry("later", day="2025-10-01"),
                   entry("prefix", ("v", "a")), entry("masked", ("a", "v"), identities=(False, False))]
        rows = by_id(build_splits(entries, plan(("v",)), exposed_groups=[]))
        self.assertEqual(rows["later"]["known_persona_colors"], ["B"])
        self.assertEqual(rows["prefix"]["policy_actions"], 5)
        self.assertEqual(rows["prefix"]["value_actions"], 0)
        self.assertEqual(rows["masked"]["policy_actions"], 0)

    def test_duplicate_components_precede_filtering_and_propagate_conflicts(self):
        a = entry("a", trajectory="same", raw="bytes1")
        b = entry("b", trajectory="same", raw="bytes2")
        c = entry("c", trajectory="same", raw="bytes2")
        c["audit_sequence_sha256"] = digest("different pass spelling")
        rows = build_splits([c, b, a], plan(), exposed_groups=[])
        self.assertEqual(rows, build_splits([a, c, b], plan(), exposed_groups=[]))
        self.assertEqual([r["source_url"] for r in rows if r["role"] == "train"], ["a"])
        self.assertEqual(rows[0]["duplicate_sources"], ["a", "b", "c"])
        b["admitted"] = False
        self.assertTrue(all(r["role"] == "excluded" for r in build_splits([a, b], plan(), exposed_groups=[])))
        conflicting = entry("d", day="2025-11-01", trajectory="same")
        rows = build_splits([a, conflicting], plan(), exposed_groups=[])
        self.assertTrue(all("ambiguous_duplicate_component" in r["exclusions"] for r in rows))
        invalid = entry("e", trajectory="same", exclusions=["illegal_replay"])
        self.assertTrue(all("illegal_replay" in r["exclusions"] for r in build_splits([a, invalid], plan(), exposed_groups=[])))

    def test_final_exposure_does_not_exclude_shared_training(self):
        row = entry("training", exposed=True)
        self.assertEqual(build_splits([row], plan(), exposed_groups=["a"])[0]["role"], "train")
        a = entry("a-copy", ("t", "b"), "2025-11-01", trajectory="same", exposed=False)
        b = entry("z-copy", ("t", "b"), "2025-11-01", trajectory="same", exposed=True)
        result = build_splits([a, b], plan(test=("t",)), exposed_groups=[])
        self.assertTrue(all(r["role"] == "excluded" and r["final_excluded"] for r in result))
        with self.assertRaises(ValueError):
            build_splits([a], plan(test=("t",)), exposed_groups=["t"])

    def test_zero_move_coincidences_are_not_claimed_as_duplicate_games(self):
        a, b = entry("a", actions=0), entry("b", actions=0)
        a["audit_sequence_sha256"] = b["audit_sequence_sha256"]
        a["trajectory_sha256"] = b["trajectory_sha256"]
        self.assertEqual(len(duplicate_components([a, b])), 2)
        self.assertTrue(all(r["role"] == "excluded" for r in build_splits([a, b], plan(), exposed_groups=[])))

    def test_invalid_configs_groups_dates_and_shape_fail_closed(self):
        with self.assertRaises(ValueError):
            plan(("a",), ("a",))
        with self.assertRaises(ValueError):
            SplitPlan(GROUPING, "2025-11-01", "2025-11-01")
        with self.assertRaises(ValueError):
            build_splits([entry("a")], plan(test=("missing",)), exposed_groups=[])
        row = entry("a")
        row["grouping_sha256"] = digest("changed aliases")
        with self.assertRaises(ValueError):
            build_splits([row], plan(), exposed_groups=[])
        row = entry("a", day="2025-02-31")
        self.assertEqual(build_splits([row], plan(), exposed_groups=[])[0]["role"], "excluded")
        for changes in ({"admitted": 1}, {"identity_valid": [True]}, {"groups": ["a"]}, {"actions": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                build_splits([{**entry("a"), **changes}], plan(), exposed_groups=[])
        with self.assertRaises(ValueError):
            build_splits([entry("a"), entry("a")], plan(), exposed_groups=[])

    def test_frozen_manifests_bind_content_and_preserve_empty_schemas(self):
        entries = [entry("a", admitted=False)]
        rows = build_splits(entries, plan(), exposed_groups=[])
        manifest = manifest_contents(entries, rows, plan(), exposure_sha256=digest([]),
                                     provenance={"source": "synthetic"}, fixture_only=True)
        changed = copy.deepcopy(entries)
        changed[0]["date"] = "2025-11-01"
        other = manifest_contents(changed, rows, plan(), exposure_sha256=digest([]), provenance={}, fixture_only=True)
        self.assertNotEqual(manifest["canonical_input_sha256"], other["canonical_input_sha256"])
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "splits"
            frozen_hash = write_membership(output, rows, manifest)
            restored, partitions = read_membership(output, frozen_hash)
            self.assertTrue(restored["fixture_only"])
            self.assertEqual(set(partitions), set(ROLES))
            self.assertEqual(partitions["excluded"], rows)
            self.assertEqual(pq.read_table(output / "train.parquet").num_rows, 0)
            self.assertIn("groups", pq.read_table(output / "train.parquet").column_names)
            with self.assertRaises(FileExistsError):
                write_membership(output, rows, manifest)
            with self.assertRaises(ValueError):
                read_membership(output, digest("another version"))
            (output / "train.parquet").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                read_membership(output, frozen_hash)
        empty = manifest_contents([], [], plan(), exposure_sha256=digest([]), provenance={}, fixture_only=True)
        self.assertEqual(sum(empty["roles"].values()), 0)


if __name__ == "__main__":
    unittest.main()
