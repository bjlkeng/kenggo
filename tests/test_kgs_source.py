import unittest

from kenggo.kgs_source import ArchivePage, candidate_reasons, discovery_accounts, select_candidates


def row(name="a-b", month=1):
    return dict(viewable="Yes", white="a [5k]", black="b [5k]", setup="19×19",
                start="1/1/25 1:00 PM", type="Ranked", result="W+Res.",
                url=f"https://files.gokgs.com/games/2025/{month}/1/{name}.sgf")


class SourceTests(unittest.TestCase):
    def test_archive_rows_and_bundle(self):
        body = '''<table><tr><td><a href="https://files.gokgs.com/games/2025/1/1/a-b.sgf">Yes</a></td>
        <td><a href="gameArchives.jsp?user=a">a</a> [5k]</td><td>b [5k]</td>
        <td>19&times;19</td><td>1/1/25 1:00 PM</td><td>Ranked</td><td>W+Res.</td></tr></table>
        <a href="/servlet/archives/en_US/a-all-2025-1.zip">zip</a>'''
        parsed = ArchivePage(body)
        self.assertEqual(parsed.rows, [row()])
        self.assertIn("https://www.gokgs.com/servlet/archives/en_US/a-all-2025-1.zip", parsed.links)

    def test_metadata_is_not_silently_defaulted(self):
        for field, value in (("setup", "19×19 H2"), ("type", "Free"),
                             ("white", "a [5k?]"), ("black", "b [-]")):
            self.assertTrue(candidate_reasons({**row(), field: value}))
        self.assertEqual(candidate_reasons(row()), [])

    def test_optional_html_cell_and_row_end_tags(self):
        body = '''<table><tr><th>Header<tr><td><a href="https://files.gokgs.com/games/2025/1/1/a-b.sgf">Yes</a>
        <td>a [5k]<td>b [5k]<td>19×19<td>1/1/25 1:00 PM<td>Ranked<td>W+Res.</table>'''
        self.assertEqual(ArchivePage(body).rows, [row()])

    def test_review_row_spanning_player_columns_is_retained_as_excluded(self):
        body = '''<table><tr><td><a href="https://files.gokgs.com/games/2025/1/1/review.sgf">Yes</a></td>
        <td colspan="2">analyst [5k] (a [5k] vs. b [5k])</td><td>19×19</td>
        <td>1/1/25 1:00 PM</td><td>Review</td><td>Unfinished</td></tr></table>
        <table><tr><td colspan="13">Month navigation</td></tr></table>'''
        parsed = ArchivePage(body).rows
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["type"], "Review")
        self.assertEqual(parsed[0]["setup"], "19×19")
        self.assertIn("not_ranked", candidate_reasons(parsed[0]))

    def test_inaccessible_rows_are_counted_without_inventing_source_ids(self):
        body = '''<table><tr><td>No</td><td>a [5k]</td><td>b [5k]</td><td>19×19</td>
        <td>1/1/25 1:00 PM</td><td>Ranked</td><td>B+Res.</td></tr></table>'''
        page = ArchivePage(body)
        self.assertEqual(page.rows, [])
        self.assertEqual(len(page.unlinked_rows), 1)
        self.assertNotIn("url", page.unlinked_rows[0])

    def test_selection_deduplicates_and_is_stable(self):
        rows = [row(str(i), 1 + i % 12) for i in range(50)]
        expected = select_candidates(rows, 20)
        self.assertEqual(expected, select_candidates(list(reversed(rows)) + rows, 20))
        self.assertEqual(len({r["url"] for r in expected}), 20)
        self.assertEqual(len({r["url"].split("/")[5] for r in expected}), 12)

    def test_conflicting_duplicates_stop(self):
        with self.assertRaises(ValueError):
            select_candidates([row(), {**row(), "result": "B+Res."}])

    def test_sparse_frame_does_not_invent_replacements(self):
        self.assertEqual(select_candidates([row()], 1000), [row()])

    def test_invalid_date_is_excluded_before_sampling(self):
        invalid = {**row(), "url": "https://files.gokgs.com/games/2025/2/31/a-b.sgf"}
        self.assertIn("invalid_source_date", candidate_reasons(invalid))
        self.assertEqual(select_candidates([invalid]), [])

    def test_survey_is_discovery_only_and_months_cover_year(self):
        survey = "\n".join(f"| user{i} | 15k | 2026-04 |" for i in range(5))
        accounts = discovery_accounts(survey)
        self.assertEqual(len(accounts), 3)
        self.assertEqual(sorted(m for a in accounts for m in a["months"]), list(range(1, 13)))
        self.assertEqual(accounts, discovery_accounts("\n".join(reversed(survey.splitlines()))))


if __name__ == "__main__":
    unittest.main()
