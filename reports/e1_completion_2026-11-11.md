# E1 completion evidence and manual validation

11 November 2026 | `kenggo-6yr`, `kenggo-6yr.10`, `kenggo-6yr.7`

**13 November update:** the owner excluded cases 044 and 083, accepted the
admission/example subset checks and the stated limitations. Use the
[manual-review decision and current inputs](e1_manual_review_2026-11-13.md).
Counts and referee comparisons below preserve the historical 11 November run.

E1's data and endgame readiness checks pass for **exploratory E2 population/rank
baseline work**. The [readiness manifest](../manifests/e1_readiness_v2.json) binds
the populated data, complete referee comparison, accepted evidence catalog and
validation results. Independent review and source/tracker delivery are recorded
in Beads. No opponent profile or human rank is qualified by this work.

## Data available for baseline work

The [population manifest](../manifests/e1_population_v2.json) records 1,200 retained
records: the original 1,000 candidates and 200 separately reserved final
candidates. The approved presumed-human policy admits 241 original games. Fixed
whole-handle and chronological splits provide:

| Role | Games | Policy examples | Valid source value targets |
| --- | ---: | ---: | ---: |
| Training | 108 | 23,395 | 21,367 |
| Population development | 14 | 3,363 | 3,363 |
| Population confirmation | 28 | 5,954 | 5,575 |
| Unseen-account development | 10 | 2,563 | 2,563 |
| Unseen-account confirmation | 22 | 4,508 | 3,750 |
| Earlier validation-account prefix | 59 | 0 | 0 |
| Sealed final | 70 | 15,223 | Sealed |

The 59 prefix games cannot support personalization because stable identities are
missing. All original 695 exposed handle groups remain quarantined from final
evaluation. The new reservation contains 190 distinct groups, with both sides of
every selected game outside the exposure registry. The 130 excluded final
candidates remain sealed and are not replaced. No E7 release was created.

All 39,783 nonfinal actions match their source records; all 36,618 valid value
targets match the recorded winner in the mover's perspective. The 446 shard files
match their hashes. Final shards were hashed without decoding their targets.
Public final access fails without a separately pinned E7 release.

Canonical records, index, split manifest and all ten membership files reproduced
byte for byte. The train-only normalizer also reproduced byte for byte, with
23,395 observations per valid normalized scalar and zero dynamic-clock counts.
Five exposed diagnostic games passed 16 pre-move checks at history windows
0/8/32/64. Duration targets, dynamic clocks and identity/persona inputs remain
masked. Source SGF results are unchanged by the referee.

This supports **exploratory E2 population/rank baseline work**. Every proposed
raw rank profile remains deferred: the pooled usable nonfinal data has at most
62 game-sides and 25 handle groups at any exact rank, below the 100-side/30-account
promotion requirement. Training-only cells are smaller. No trained model, playing
strength, human-like behavior, human rank, personalization or timing capability is
claimed. B still starts from random weights when its later prerequisites pass.

## Ending method and development evidence

The [version-2 contract](../plan/e1_empirical_endgame_contract_v2.md) preserves
ordinary KGS Japanese play and original actions. A pinned CPU KataGo B10 referee
plays cleanup on two separate copies at 32 and 128 visits per decision. The
independent native verifier replays each proposed move and computes the exact
formal score. The [selected profile](../manifests/e1_katago_referee_profile_v2.json)
requires both runs to finish without no-result and agree on the winner; it always
selects the 128-visit transcript's margin. Both scores remain visible.

This is an empirical interpretation of the original ending. The two searches
share a model and can share mistakes. Their difference is not a confidence
interval, and a higher visit budget does not prove a more accurate margin.
Accepted outcomes are tagged `empirical_score`, separately from certified scores.
Loading a saved outcome requires an external evidence-hash catalog and native
replay. Unknowns and faults retain invalid learning boundaries; administrative
caps retain their learner-to-move bootstrap semantics. No failure becomes a draw.

The development record includes the unsuccessful attempts:

| Attempt | Result | Measured CPU seconds |
| --- | --- | ---: |
| Tiny B6 model, initial dense fixtures | Failed scoring fixtures | 12.39 |
| Tiny B6, corrected deterministic configuration | Still failed | 12.93 |
| B10 dense scoring probes, 128 then 32 visits | Settled/dead/seki scores 13/3/13 half-points at both budgets | 1.98 + 1.38 |
| Four hash-selected exposed source endings | Both budgets completed with equal scores; cost probe only | 45.96 |
| First integration smoke | Neural no-result prediction incorrectly blocked native-scored seki; fixed before source panel | 1.56 recorded decision CPU; total not recorded |
| Corrected integration smoke | All three scores and persistence passed | 2.21 |
| Initial full source panel | 85/108 resolved; coverage gate failed | 1,907.82 |
| Six dense neural pass fixtures | All unjudgeable; exact checker handles all six | 8.11 |
| First extended-cleanup diagnostic | Persistence ledger still assumed the old move cap; retained failure and fixed | 116.60 |
| Corrected fixed five-case diagnostic | All five resolved and persisted; exact scoring fixtures still pass | 191.93 |
| Revised full source panel | 108/108 resolved; 100/102 known scored winners agree | 2,282.04 |

The [initial full comparison](../manifests/e1_endgame_comparison_v1.json) reports
every ending and pass. Its 23 unresolved endings comprise 11 work-cap failures,
10 move-cap failures and two exact-margin disagreements. Among 82 resolved scored
comparisons, 81 winners and 68 exact margins matched the source. This failed the
predeclared 90% resolution gate; it was not reclassified as a success.

The revised policy targets the W/L reward while reporting the predetermined
higher-budget margin. It separately raises cleanup from 32 to 64 moves per run
and the combined visit/replay limit from 20,000 to 150,000. The 120 CPU/180 wall
second per-judgment limits remain. Source results and candidate identity never
select a transcript. The fixed diagnostic used original source-panel cases
002/003/004/007/010, representing two work caps, two move caps and one margin
disagreement. The full revised panel retains all 108 endings and the same
24 hash-selected actual passes. Acceptance still requires 90% resolution and
95% agreement among resolved known scored winners. This is exposed-source
implementation evidence, not held-out accuracy or candidate qualification.

The [revised full comparison](../manifests/e1_endgame_comparison_v2.json) accounts
for **108/108 resolved endings** and **100/102 recorded scored winners agreeing
(98.04%)**, passing both thresholds. Six source results lack a scored margin and
remain outside the winner/margin comparison denominator. Exact recorded margins
agree for **77/102**; all 25 differences are retained case by case. Six endings
have different 32/128-visit native margins, with the same winner. The selected
128-visit margin follows the declared rule in every case.

The two winner disagreements need manual inspection:

| Exposed source game | Recorded result | Both native cleanup scores |
| --- | --- | --- |
| [lorbas72–chika, 2 March](https://files.gokgs.com/games/2025/3/2/lorbas72-chika.sgf), case 044 | Black +15.5 | White +0.5 |
| [BasilDiug–yuumsuzuki, 29 July](https://files.gokgs.com/games/2025/7/29/BasilDiug-yuumsuzuki.sgf), case 083 | Black +53.5 | White +2.5 |

Source labels remain unchanged. All accepted endings passed saved-outcome native
revalidation; the separately pinned [evidence catalog](../manifests/e1_endgame_evidence_v2.json)
contains those 108 decisions and the three exact scoring fixtures. The complete
run used 2,282.04 CPU seconds and 2,257.71 wall seconds, with a 245,402,710-byte
compressed trace. It made zero HTTP requests and used no GPU.

Use exact pass certificates where available and retain their method tags. The
neural pass method compares forced pass with the best searched non-pass option
at both budgets. Gains at most 0.5 points are acceptable; gains at least 1.5 are
premature; disagreements or uncertain results are unjudgeable. The initial source
and revised source panels both judged 21/24 passes and left three unjudgeable.
The separate six dense fixtures demonstrate its limitations. Future 99%
candidate-pass coverage and
game-level behavior gates remain unchanged and unmet by this experiment.

## Reproduction and resource use

Use the [population commands](../docs/e1_population.md#reproduce-the-build) and
[referee setup/API](../docs/rules_engine.md#empirical-ending-referee). Inputs and
outputs have independent pins; old version-1 artifacts remain reproducible.
Models, SGFs, arrays and detailed traces stay under ignored cache/run paths.
Archived probe scripts and recorded code hashes preserve the failed attempts.
The full revised run also retains all six executed source files, verified against
its pre-run hashes; its CLI snapshot was recovered byte for byte after the
failure-record ownership fix.

With the original exposed canonical input and pinned referee cache present, rerun
the complete ending/pass panel into a new directory:

```bash
uv run --locked python tools/audit_endgame_referee.py \
  manifests/e1_katago_referee_profile_v2.json \
  2831d93dfbba8713a7e3fb00a1318bd4024ae9defd8032e39737f807ee4a1be4 \
  runs/e1-referee-reproduce

uv run --locked python tools/summarize_endgame_referee.py \
  runs/e1-referee-reproduce runs/e1-referee-reproduce/comparison.json
```

The full referee run reserves 3,600 CPU seconds, 5,400 wall seconds and 1 GiB of
compressed trace, with no HTTP requests or GPU work. It rejects an existing output
directory and checks its frozen code hashes between cases.

The new KGS acquisition made **72 serial HTTP requests**, with a measured minimum
5.00006 seconds between starts, and received **13,183,226 bytes**. It used 3.38 CPU
seconds and 494.34 wall seconds. There were no automatic retries. Candidate IDs
and all participant groups were reserved before retrieving SGF content. No final
targets entered model/referee selection or diagnostics.

Canonical construction used 31.49 CPU seconds, and its repeat used 31.86. The
five-training-game encoding probe used 54.25; full encoding, normalizer and
diagnostics used 2,362.27. Shard/normalizer integrity checks used 1.06, and source
action/value checks used 2.15. Each run stayed within its recorded budget. These
are single-run costs, not performance comparisons. GPU use was zero. Model-file
downloads and authoring/review were not comprehensively CPU-timed; there is no
claim of a complete project cost total.

All **170 tests pass with the pinned native reference enabled and zero skips**.
Lockfile/environment checks, CPU tensor/NumPy/Parquet smoke, package build,
whitespace, local documentation links, artifact pins and ignore rules pass.
Regression checks cover bounded failure evidence, the larger cleanup persistence
ledger, and preserving completed/failed output directories when a rerun is
rejected. The last change affects CLI failure recording only; it does not change
the referee decisions measured above.

## Manual validation

Use the [local review UI](../docs/e1_manual_review.md) to replay source/cleanup
games, inspect admission and stored examples, and export your review notes.

1. Inspect **both winner disagreements in the table above**, especially living/dead
   groups, seki, prisoners and cleanup changes. Then sample the comparison's other
   margin disagreements. Treat referee agreement as empirical evidence.
2. Spot-check presumed-human admission against familiar KGS accounts and the
   [recorded policy](../reports/e1_kgs_admission_2026-11-11.md). The research-use
   permission is owner-attested; no manual relabeling of all games is required.
   Report concrete known bots, aliases or provenance errors for quarantine.
3. Review a few **training/development** SGFs and their pre-move examples using
   the source IDs in the encoding diagnostics. Check board orientation, pass
   actions, mover-view values and missing-clock/persona masks. Leave final targets
   sealed until E7 selection is frozen.
4. Before GPU training, resolve the host driver mismatch tracked in
   `kenggo-ydo.7`. A read-only host check on 11 November still found NVML 595.99
   against kernel module 595.84; `nvidia-smi` exited 18. E1's CPU success does not
   establish CUDA readiness. No driver or system configuration was changed.
