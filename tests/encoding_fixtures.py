"""Affirmative admissions here are synthetic and never production policy evidence."""

from hashlib import sha256

import pyarrow as pa
import pyarrow.parquet as pq

from kenggo.canonical_data import canonicalize, digest, encoded
from kenggo.data_splits import SplitPlan, build_splits, manifest_contents, write_membership


GROUPING = digest("synthetic example groups")


def record(name="fixture", *, day="2025-07-02", groups=("alice", "bob"), identities=(True, True),
           moves=None, result="B+R", admitted=True, nominal_clock=(1200, 30, 5)):
    if moves is None:
        start = int(sha256(name.encode()).hexdigest(), 16) % 300
        points = [start, start + 20, start + 40]
        moves = "".join(f";{color}[{chr(97 + point % 19)}{chr(97 + point // 19)}]"
                        for color, point in zip("BWB", points))
    re = "" if result is None else f"RE[{result}]"
    main, seconds, periods = nominal_clock
    data = (f"(;FF[4]GM[1]SZ[19]PB[{groups[0]}]PW[{groups[1]}]BR[5k]WR[1d]RU[Japanese]KM[6.5]"
            f"TM[{main}]OT[{periods}x{seconds} byo-yomi]DT[{day}]{re}{moves})").encode()
    url = "https://example.invalid/games/" + day.replace("-", "/") + "/" + name + ".sgf"
    source = {"source_url": url, "sgf_sha256": sha256(data).hexdigest(), "zip_member": name + ".sgf"}
    archive = {"url": url, "black": groups[0] + " [5k]", "white": groups[1] + " [1d]",
               "type": "Ranked", "result": result}
    row, _ = canonicalize(data, source, archive)
    row["eligibility"] = {"training_eligible": admitted, "black_group": groups[0], "white_group": groups[1],
                          "black_identity_valid": identities[0], "white_identity_valid": identities[1],
                          "eligibility_reasons": [] if admitted else ["fixture_not_admitted"]}
    return row


def entries(records):
    return [{"source_url": r["source"]["source_url"], "canonical_sha256": digest(r),
             "grouping_sha256": GROUPING, "sgf_sha256": r["source"]["sgf_sha256"],
             "audit_sequence_sha256": r["audit"].get("sequence_fingerprint"),
             "trajectory_sha256": r["trajectory_sha256"], "semantic_sha256": r["semantic_sha256"],
             "groups": [r["eligibility"][p + "_group"] for p in ("black", "white")],
             "identity_valid": [r["eligibility"][p + "_identity_valid"] for p in ("black", "white")],
             "date": r["metadata"]["DT"], "admitted": r["eligibility"]["training_eligible"],
             "final_excluded": False,
             "exclusions": r["structural_exclusions"] + r["eligibility"]["eligibility_reasons"],
             "actions": len(r["actions"]),
             "actions_by_color": [sum(a["color"] == c for a in r["actions"]) for c in ("B", "W")],
             "value_supported": r["recorded_value_supported"]} for r in records]


def split(records, *, validation=(), test=()):
    plan = SplitPlan(GROUPING, "2025-09-01", "2025-11-01", validation, test)
    inputs = entries(records)
    rows = build_splits(inputs, plan, exposed_groups=[])
    manifest = manifest_contents(inputs, rows, plan, exposure_sha256=digest([]),
                                 provenance={"source": "synthetic tests only"}, fixture_only=True)
    return rows, manifest


def corpus(directory, records, **cohorts):
    directory.mkdir()
    path = directory / "canonical.parquet"
    pq.write_table(pa.Table.from_pylist([{"source_url": r["source"]["source_url"],
        "canonical_sha256": digest(r), "canonical_json": encoded(r)} for r in records]), path)
    rows, manifest = split(records, **cohorts)
    split_hash = write_membership(directory / "splits", rows, manifest)
    return path, sha256(path.read_bytes()).hexdigest(), directory / "splits", split_hash
