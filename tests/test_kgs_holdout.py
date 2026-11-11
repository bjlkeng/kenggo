import copy
import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location("acquire_holdout", Path(__file__).parents[1] / "tools/acquire_kgs_holdout.py")
holdout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(holdout)


def row(name, white="alice", black="bob", month=11):
    return {"url": f"https://files.gokgs.com/games/2025/{month}/1/{name}.sgf",
            "white": white + " [1k]", "black": black + " [1k]", "setup": "19×19",
            "type": "Ranked", "result": "B+R"}


class HoldoutSelectionTests(unittest.TestCase):
    def test_both_colors_exposure_and_suspected_bots_are_excluded(self):
        reservation = {"exposed_groups": ["kgs-handle:exposed"],
                       "accounts": [{"group": "kgs-handle:alice"}], "candidate_cap": 200}
        rows = [row("good"), row("exposed-white", "exposed", "alice"),
                row("exposed-black", "alice", "exposed"), row("bot-white", "Michi1Bot", "alice"),
                row("bot-black", "alice", "AyaBot002"), row("wrong-month", month=10),
                row("no-seed", "new", "other")]
        self.assertEqual(holdout.select_rows(rows, reservation), [rows[0]])

    def test_order_and_result_do_not_select_favorable_final_games(self):
        reservation = {"exposed_groups": [], "accounts": [{"group": "kgs-handle:alice"}], "candidate_cap": 2}
        records = [row(str(i)) for i in range(10)]
        expected = [r["url"] for r in holdout.select_rows(records, reservation)]
        changed = copy.deepcopy(records)
        for r in changed:
            r["result"] = "W+100.5"
        self.assertEqual([r["url"] for r in holdout.select_rows(changed[::-1], reservation)], expected)
        duplicated = records + [copy.deepcopy(records[0])]
        self.assertEqual([r["url"] for r in holdout.select_rows(duplicated, reservation)], expected)
        duplicated[-1]["result"] = "W+R"
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            holdout.select_rows(duplicated, reservation)

    def test_group_reservation_is_stable_and_excludes_prior_exposure(self):
        records = [row(str(i), "p" + str(i), "exposed") for i in range(30)]
        selected = holdout.choose_groups(records, {"kgs-handle:exposed"})
        self.assertEqual(len(selected), 20)
        self.assertEqual(selected, holdout.choose_groups(records[::-1], {"kgs-handle:exposed"}))
        self.assertNotIn("kgs-handle:exposed", {r["group"] for r in selected})
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            holdout.choose_groups(records[:10], {"kgs-handle:exposed"})


if __name__ == "__main__":
    unittest.main()
