# More data helped move prediction; value accuracy stopped further growth

Beads: `kenggo-ydo.16` (data and training), `.17` (recovery), and `.18`
(validation coverage).

The study doubled fitting data from **2,000,000 to 3,999,997 moves**, using
**18,126 complete games**. The larger-data model predicted human moves better
than both the previous model and the continued 2M-data control. Its predictions
of the eventual winner became worse in two development scopes. That triggered
the declared value-error guard, so another doubling did not run. The study
stopped on its decision rule, with time left in the original seven-day budget.

Open the [owner review page](../runs/e2-progressive-completion-2026-11-25/review.html)
for training and validation curves, earlier runs, all fixed comparisons and
recovery details. The [compact charts](../runs/e2-progressive-completion-2026-11-25/charts.html)
use the same saved measurements. The
[manifest](../manifests/e2_progressive_training_v1.json) pins the evidence.

## What ran

The [frozen plan](../plan/e2_progressive_training_v1.md) retained the eight-block,
128-channel rank-conditioned model, zero model-history window, Adam at 0.001,
0.1 clock dropout, human-move cross-entropy and masked winner-value MSE.
The model's inputs and the original 2M fitting-only normalizer were unchanged.
The rules adapter still retains full legality history. This is supervised
training of A; no weights were transferred to B.

The 2M control and 4M candidate started from the same parent weights, optimizer
state and random state. Both finished **six matched intervals**, each adding
31,250 updates. Each arm therefore received **187,500 new updates and
24 million move presentations**, in addition to the parent's 31,250 updates.
Execution order alternated. Epoch-zero measurements matched.

A separate **52-game / 11,420-position stopping sample** was frozen before
evaluating a checkpoint. It contains two games per required rank/komi cell.
Reporting uses 11,303 policy rows and 10,921 valid value rows in the 12k–1d
scope; 117 weaker-rank rows remain stored. The fixed training-monitor sample
has 11,273 positions, including 11,147 in-scope policy rows and 10,883 valid
value rows. Its curves describe a small sample, not the entire fitting set.
Online losses use all presented batches and changing weights; the monitor
and stopping curves use frozen checkpoints without clock dropout.

The lowest validation move loss selected **candidate interval 3** and
**control interval 1**. These contain 93,750 and 31,250 new updates respectively.
Both procedures had the same six checkpoint opportunities; their selected
checkpoints have different exposure. The pair stopped after both arms had at
least three intervals without a 0.002 validation-NLL decrease. The final
interval is retained but does not replace either selected checkpoint.

Weights & Biases (W&B) tracking identifies the
[previous model](https://forge.coreweave.com/wandb/bjlkeng/kenggo/runs/0f99998a454740d38c7a2784c4a43a6e),
[selected 2M control](https://forge.coreweave.com/wandb/bjlkeng/kenggo/runs/38b820c5362d48b78379974150e4bed5)
and [selected 4M candidate](https://forge.coreweave.com/wandb/bjlkeng/kenggo/runs/262a7ebf277c4af4b43c804ad896219b).
The [tracking index](../manifests/e2_progressive_tracking_v1.json) pins all
25 saved run receipts, including the parent and the
[validation-source collection](https://forge.coreweave.com/wandb/bjlkeng/kenggo/runs/275d89f479a344bba5166c5ca6e2c344),
and the selected checkpoint artifact names, digests
and recorded sync status. These are saved receipts; completion made no new
upload or remote-status query. The interrupted growth run still says
`running` in its sealed tracking snapshot. Its process stopped, and its
unknown original exit remains a partial attempt in the ledger.

## Fixed comparisons

The fixed development comparison retains **539 games / 120,450 positions**,
with **111,591 positions in the 12k–1d scope**. All three selected models
completed all four reports: **361,350 predictions** in total. These positions
are separate from the stopping sample. They guide data growth and are not
independent final testing. Confirmation and final targets stayed closed.

The primary unseen-account, 6.5-komi comparison uses the same 127 whole games
for every model, with 27,268 in-scope policy rows and 26,150 valid winner labels.
NLL measures move prediction loss; MSE measures error predicting the eventual
human-game winner from the mover's perspective. Lower is better for both.

| Selected model | Move NLL | Move matching | Value MSE |
| --- | ---: | ---: | ---: |
| Previous 2M model | 2.4924 | 37.18% | 1.1722 |
| Continued 2M control | 2.4963 | 37.14% | 1.3279 |
| Continued 4M candidate | 2.3948 | 39.31% | 1.3505 |

The candidate's paired NLL reduction is **0.1015** against the control
(95% interval **0.0763–0.1268**) and **0.0976** against the parent
(**0.0640–0.1311**). Both exceed the declared useful reduction of 0.02.
All four fixed scopes show a supported NLL reduction against both models.

Two value-error increases against the parent blocked further growth:

| Scope | Candidate minus parent value MSE | Paired 95% interval |
| --- | ---: | ---: |
| Population development, 0.5 komi | +0.2388 | +0.0678 to +0.4098 |
| Unseen accounts, 6.5 komi | +0.1782 | +0.0435 to +0.3129 |

Both lower bounds exceed the allowed +0.02 regression. The original summary's
generic reason, “No useful supported data-growth gain,” is retained as saved
evidence; the more precise result is **a useful move-prediction gain with a
value-error regression**.

Metrics average moves within a game side, sides within each mover rank, then
ranks equally. Policy and value masks are separate. Paired uncertainty accounts
for shared games/accounts, but not variation across training seeds. This is
one exploratory seed with development-based selection. It does not establish
stronger play, a human rank or E2 readiness. No new Human SL inference or
complete-game panel ran; earlier premature passing remains unresolved.

## Recovery, integrity and cost

- **.17 — Recover supervision:** the original supervisor was interrupted.
  Recovery reused the saved 4M dataset after checking its hashes, memberships,
  labels, masks and normalizer lineage. The original worker's exit code remains
  unknown; its attempt remains partial. Later failed recovery and preparation
  attempts, their charges and summaries are preserved. A reviewed durable
  service completed the training and comparisons under the original allowance.
- **.18 — Repair stopping coverage:** all three existing 11k/6.5-komi validation
  games were already in the fixed comparison. A separately reviewed source
  phase checked 12 previously unqueried archive months from the frozen account
  cohorts and admitted seven usable games. Two enter the stopping sample.
  All seven source IDs and duplicate fingerprints are barred from fitting.
  Cohorts, chronology, admission policy and normalizer were not relaxed.
- **Final verification:** ten original checks passed; one stopped on a
  relative-versus-absolute parent-path comparison. A new helper checks both
  content hashes and resolved file identity. Its regression test failed before
  the correction; all 14 affected tests passed afterward. The skipped composite
  check then passed, including independent replay/re-encoding of the two
  selected supplemental games, original arrays, masks, normalization and
  sample separation. The old failed result is unchanged. The full 4M audit was
  reused by hash, not repeated; this is not a fresh replay of every fitting game.

Production finished **99.81 elapsed hours** after the original launch on
21 November at 14:13:03 UTC, before the **28 November, 14:13:03 UTC deadline**.
The cumulative ledger charges **64.36 CPU hours** and **18.73 allocated GPU
hours**, including all retained attempts, verification and a conservative
**5-CPU-hour upper charge for root completion**. CPU charges include bounds
for lost timing and recovery work; they are not all measured CPU. The final
handoff records actual root timing against that bound and output size. No
allowance was reset or production work retried during completion.

Source stayed clean at `59d29abf57d8c03eec783a30222fa288403c8e7b` until every
production and verification worker stopped. Exact recovery/helper source is
now preserved in the [source archive](../tools/e2_progressive_archive/README.md).
Raw data, checkpoints, credentials and generated run output remain local.

Workspace lock, sync, CPU smoke, build and diff checks passed. Repository
validation has 750 passing tests and eight skipped native-reference checks;
completion changed no rules code. The first full run hit 12 workspace-guard
errors before new files were staged. All 14 tests in that module passed after
staging; the original failure and its resolution remain in the manifest.
The separate 14-test verifier regression check also passed.

## What to review and what to try next

1. **For .16, inspect the training and validation curves.** Check the selected
   intervals and the flat or rising validation move loss after them. This
   explains why training stopped and why the last checkpoint was not selected.
2. **For .16, compare move and value results on the fixed samples.** The data
   gain and value regression answer different questions. Together they explain
   why another doubling was blocked despite better move matching.
3. **For .17/.18, read the recovery note and record an E2 next-step preference.**
   The original allowance, failures and validation-only additions are visible
   so the result can be judged with its limitations.

The leading hypothesis is poor generalization of winner prediction: training
value error is low, while development value error remains high. This does not
identify the cause. Before collecting more data, inspect saved value errors
and calibration by komi, rank and phase, then test regularization or a smaller
learning rate with a checkpoint rule that also considers value accuracy.
Those are proposed experiments requiring their own plan and budget.

E2 and its readiness bead `kenggo-ydo.6` remain open. Complete-game behavior and
profile qualification still need separate evidence.
