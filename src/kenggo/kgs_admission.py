"""Owner-approved presumed-human screening of the frozen KGS audit."""

import argparse
import json
from pathlib import Path

from kenggo import kgs_eligibility as legacy


POLICY = "kgs_eligibility_v2"
DECISION_PATH = Path("manifests/e1_kgs_admission_policy_v2.json")
DECISION_SHA256 = "461a7fed1ea637d8d39cfa1437a06f6d4e198a59af9bf1cabd6c245ef444c4bc"
REPLACED_REASONS = frozenset({"black_human_unverified", "white_human_unverified",
                              "ordinary_game_unverified", "training_reuse_basis_unresolved"})


def classify(record, archive_row, groups, exposure):
    """Screen an audited record; production admission also pins its source inputs."""
    decision = legacy.classify(record, archive_row, groups, exposure)
    reasons = set(decision["eligibility_reasons"]) - REPLACED_REASONS
    metadata = record.get("metadata", {})
    ranked = archive_row is not None and archive_row.get("type") == "Ranked"
    ordinary = ranked and not (metadata.get("EV") or metadata.get("RO"))
    if not ranked:
        reasons.add("ranked_archive_evidence_missing")
    if ordinary:
        decision["ordinary_status"] = "presumed_ordinary_ranked"
    for color in ("black", "white"):
        if (ranked and decision[color + "_group"] is not None
                and decision[color + "_human_status"] == "unverified"):
            decision[color + "_human_status"] = "presumed_human"
    admitted = (not reasons and decision["metadata_compatible"] and ordinary
                and all(decision[c + "_human_status"] == "presumed_human" for c in ("black", "white")))
    decision.update(policy=POLICY, source_policy_decision_sha256=DECISION_SHA256,
                    source_use_basis="owner_attested_research_permission",
                    eligibility_status="admitted_presumed_human" if admitted else "excluded",
                    eligibility_reasons=sorted(reasons), training_eligible=admitted,
                    eligibility_assumptions=["ranked_archive_known_bot_screen",
                                             "ordinary_ranked_without_event_metadata",
                                             "owner_attested_research_permission"])
    return decision


def audit(audit_path, sample_path, discovery_path, output, *, decision_path=DECISION_PATH):
    if decision_path.stat().st_size > 64 * 1024 or legacy.sha256(decision_path) != DECISION_SHA256:
        raise ValueError("Owner policy decision differs from its independent pin")
    decision = json.loads(decision_path.read_text())
    evidence = {"policy": POLICY, "decision_sha256": DECISION_SHA256,
                "classifier_code_sha256": legacy.sha256(Path(__file__)),
                "source_use_basis": decision["source_use_basis"]}
    return legacy.audit(audit_path, sample_path, discovery_path, output,
                        classifier=classify, policy_evidence=evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("sample", type=Path)
    parser.add_argument("discovery", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit(args.audit, args.sample, args.discovery, args.output)


if __name__ == "__main__":
    main()
