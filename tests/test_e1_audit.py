from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import pyarrow.parquet as pq

from kenggo.e1_audit import audit, member_key
from kenggo.kgs_source import select_candidates


class OfflineAuditTests(unittest.TestCase):
    def test_zip_mapping_requires_an_exact_valid_date_and_filename(self):
        self.assertEqual(member_key("games/2025/01/02/b-a.sgf"), ("2025-01-02", "b-a.sgf"))
        for name in ("../2025/1/2/b-a.sgf", "/2025/1/2/b-a.sgf", "2025/2/31/b-a.sgf", "b-a.sgf"):
            self.assertIsNone(member_key(name))

    def test_offline_sample_preserves_duplicates_missingness_and_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            config = root / "config.json"
            config.write_text("{}")
            config_hash = hashlib.sha256(config.read_bytes()).hexdigest()
            rows = [dict(viewable="Yes", white="b [5k]", black="a [5k]", setup="19×19", type="Ranked",
                         start="1/1/25 1:00 PM", result="B+Res.", page_key="a-2025-01", bundle_url="unused",
                         url=f"https://files.gokgs.com/games/2025/1/1/game{i}.sgf") for i in range(1000)]
            (source / "frame.json").write_text(json.dumps(dict(rows=rows, row_exclusion_counts={})))
            (source / "sample.json").write_text(json.dumps(dict(candidates=select_candidates(rows), config_sha256=config_hash)))
            data = b"(;FF[4]GM[1]SZ[19]PB[a]PW[b]BR[5k]WR[5k]RU[Japanese]KM[6.5]TM[1200]OT[5x30 byo-yomi]DT[2025-01-01]RE[B+R];B[aa];W[bb])"
            with zipfile.ZipFile(source / "a-2025-01.zip", "w", zipfile.ZIP_DEFLATED) as archive:
                for i in range(1000):
                    archive.writestr(f"games/2025/1/1/game{i}.sgf", data)
            output = root / "audit"
            with redirect_stdout(io.StringIO()):
                audit(source, config, output)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["retrieved_selected"], 1000)
            self.assertEqual(summary["exclusion_counts"]["duplicate_sgf_bytes"], 999)
            self.assertEqual(summary["metadata_scope_compatible"], 1)
            self.assertEqual(summary["fully_verified_training_games"], 0)
            self.assertEqual(summary["all_parsed"]["action_clock_counts"]["explicit_passes"], 0)
            self.assertEqual(summary["all_parsed"]["action_clock_counts"]["post_clock_missing"], 2000)
            self.assertEqual(pq.read_table(output / "games.parquet").num_rows, 1000)
            self.assertEqual(summary["games_parquet_sha256"], hashlib.sha256((output / "games.parquet").read_bytes()).hexdigest())
            (source / "a-2025-01.zip").write_bytes(b"<html>not an archive</html>")
            with redirect_stdout(io.StringIO()):
                audit(source, config, root / "failed-audit")
            failed = json.loads((root / "failed-audit" / "summary.json").read_text())
            self.assertEqual(failed["sampled_candidates"], 1000)
            self.assertEqual(failed["retrieved_selected"], 0)
            self.assertEqual(failed["parse_status_counts"], {"not_parsed": 1000})
            self.assertEqual(failed["bundles"][0]["error"], "invalid_zip")


if __name__ == "__main__":
    unittest.main()
