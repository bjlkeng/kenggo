# Screened KGS source admission, version 2

**11 November 2026, UTC — `kenggo-6yr.11`: 241 of the frozen 1,000 games pass
source admission under the owner-approved presumed-human policy.** The other
759 remain excluded. The [owner decision](../manifests/e1_kgs_admission_policy_v2.json),
[run manifest](../manifests/e1_kgs_admission_v2.json) and
[per-game decisions](../manifests/e1_kgs_admission_games_2026-11-11.parquet)
record the revised population and its limits.

The owner approved best-guess human screening and stated that permission to use
the games is available. Version 2 records that statement as the research-use
basis. It does not claim independent licence verification or permission to
redistribute the raw corpus. No further permission confirmation is required for
this bounded admission revision.

## Admission rules

[`kgs_admission`](../src/kenggo/kgs_admission.py) starts with the original
[eligibility classifier](../src/kenggo/kgs_eligibility.py). It preserves all
metadata, corruption, known-bot, account-group and explicit game-context
exclusions. It replaces only the three unresolved evidence requirements approved
by the owner:

| Evidence | Version-2 treatment |
| --- | --- |
| Explicit archive `Type=Ranked`, with no participant in a documented bot handle group | Label the participants `presumed_human`. This is a screening assumption, not verified historical control or proof of unaided moves. |
| Explicit `Ranked` without SGF event/round metadata | Label the context `presumed_ordinary_ranked`. Unknown types, non-Ranked types and event/round evidence cannot pass. The HTML mapping is not newly claimed to distinguish every tournament or simul. |
| Owner statement that permission to use is available | Record `owner_attested_research_permission` for this project's local research training and evaluation. |

Known bot exclusions still apply to both colors and through reviewed aliases.
There are no new alias links. A short blacklist is not a complete historical bot
census; undetected bots, assisted play and unmarked nonordinary games remain
possible. The human-like population claim must describe this screening method.
Later contradictory evidence requires exclusion of the affected groups and
rebuilding any affected data artifacts.

An accepted row has `eligibility_status=admitted_presumed_human` and
`training_eligible=true`. That existing field names the **source gate**; replay,
whole-game splits, masks and example validation still apply. Assumptions are
stored separately from exclusion reasons so they cannot silently erase a hard
exclusion. Incoming `verified_human`, reuse or identity flags do not select a
policy or override its filters.

The production command pins the owner-decision file, original audit, original
sample and discovery provenance before classification. It accepts exactly the
same 1,000 source IDs. The old version-1 command remains available and still
reproduces its original decision-table hash. No original audit, SGF, source ID,
rank, result, move, clock or pass was modified.

## Recomputed coverage and preserved boundaries

| Quantity | Result |
| --- | ---: |
| Frozen source records | 1,000 |
| Games with both players presumed human | 971 |
| Games presumed ordinary from retained Ranked metadata | 1,000 |
| Source-admitted games after all existing filters | 241 |
| Excluded games / games involving a documented bot handle | 759 / 29 |
| Distinct admitted handle groups | 216 |
| Recorded actions / explicit passes in admitted games | 52,900 / 266 |
| Admitted source results: score / resignation / timeout / forfeit / unknown | 102 / 114 / 16 / 6 / 3 |
| Two-pass endings / accepted automatic scores | 108 / 0 |
| Verified human games / verified ordinary-game classifications | 0 / 0 |
| Valid identity sides / final-eligible sampled games | 0 / 0 |

The 241 admitted IDs exactly match the earlier metadata-compatible subset. The
[existing rules comparison](../manifests/e1_rules_validation_v1.json) covered
their 52,900 actions with zero native discrepancies. This revision also checked
their replay-valid status and absence of structural exclusions against the
retained canonical index. It did not perform a new corpus replay or adjudication.
The two additional structurally supported records in the canonical report remain
excluded by the inherited archive/result-mismatch filter; this revision changes
the human/context/use policy only.

| Proposed rank label | Player-game appearances | Recorded actions | Explicit passes |
| --- | ---: | ---: | ---: |
| 15k | 10 | 897 | 5 |
| 10k | 22 | 3,091 | 14 |
| 5k | 5 | 448 | 1 |
| 1k | 56 | 6,350 | 27 |
| 1d | 80 | 8,031 | 13 |

The manifest gives all 16 exact ranks and result/ending counts. These are
convenience-sample counts, not KGS-wide estimates. No rank reaches E0's minimum
100 game sides per claimed rank/phase/time cell even before splitting, so no raw
profile is promoted by this admission decision.

All 695 exposed handle groups, selected source IDs, SGF hashes, sequence hashes
and discovery account-months retain their final-evaluation quarantine. Every
decision's original grouping, quarantine and identity/persona fields was compared
with version 1 and remained equal. Handle grouping still cannot identify
undocumented alternate accounts, account sharing or historical generations;
every persona stays default with identity validity false. Duration and dynamic
clock masks remain governed by the unchanged data contract.

**The current canonical/split/example artifacts still carry version-1 admission
and remain empty for training.** `kenggo-6yr.7` must regenerate them from these
reviewed decisions and update their production pins before use. The new table
cannot be substituted into an old manifest or used to bypass the example
loader's hash checks. Additional data/support and unseen cohorts are still
needed, and `kenggo-6yr.10` still owns usable ending/pass judgments. E1 remains
unready; admission is not model qualification.

## Reproduction and validation

From the repository root with retained inputs, use a fresh output directory:

```bash
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.cache/python"
uv run --locked python -m kenggo.kgs_admission \
  manifests/e1_kgs_audit_games_2026-11-10.parquet \
  runs/e1-kgs-audit-2026-11-10-v4/sample.json \
  manifests/e1_kgs_discovery_2026-11-10_v2.json \
  runs/e1-kgs-admission-reproduced
uv run --locked python -m unittest discover -s tests -p 'test_kgs*.py' -v
```

Two version-2 runs produced identical decision Parquet bytes with SHA-256
`7e63e6c50a78a751dc0494533f6a5e80881019b23b58289a503e9257ceda7295`.
A separate version-1 rerun reproduced its original SHA-256
`57a3b6e520bf53d9b69bfac4db56d40108117e8d6438acaef141862082bd1dd7`.
The three metadata passes used 1.44 process-CPU seconds and 0.58 wall seconds in
total, excluding imports and final summary writes. The manifest retains each
measurement; these are run costs, not performance comparisons. The recorded
audit allowance was 600 CPU / 900 wall seconds.

Tests cover positive presumptions, both-color and alias bot exclusions,
unknown/nonordinary/event cases, inherited hard exclusions, forged incoming
claims, changed source/policy evidence, stale group registries, quarantine and
identity preservation. No game was downloaded or newly parsed, no KGS request
was made, and no model or GPU job ran.

The full CPU suite passed **112 tests**, including all five enabled native
reference methods with no skips. Lock/environment checks, CPU smoke, package
build and whitespace checks also passed. The changed documents passed 79 local
link/anchor checks and 53 external-URL syntax checks. The validation logs and
their hashes are recorded in the manifest.
