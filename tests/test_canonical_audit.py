from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from kenggo.canonical_audit import ELIGIBILITY_PATH, load_frozen, run


class CanonicalBoundaryTests(unittest.TestCase):
    def test_changed_affirmative_source_claims_are_rejected_before_any_sgf_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "sample.json").write_text("{}")
            rows = pq.read_table(ELIGIBILITY_PATH).to_pylist()
            rows[0].update(training_eligible=True, black_identity_valid=True, ordinary_status="verified")
            changed = root / "forged.parquet"
            pq.write_table(pa.Table.from_pylist(rows), changed)
            with self.assertRaisesRegex(ValueError, "Frozen input hash mismatch.*forged.parquet"):
                load_frozen(root, eligibility_path=changed)

    def test_completed_destination_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            sentinel = output / "existing"
            sentinel.write_text("keep")
            with self.assertRaises(FileExistsError):
                run(output / "absent", output / "missing", output)
            self.assertEqual(sentinel.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
