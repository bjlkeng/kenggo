# Finish the planned E2 checks after the reference CPU cap

Bead: `kenggo-ydo.13`. This amendment to the
[expanded comparison](e2_expanded_comparison_v1.md) uses the remainder of the
owner's approved 24-hour window. It changes how the remaining jobs are reached
after one reference job failed. It adds no training, data collection or budget.

All four models completed 31,250 updates and four million presented examples
each. All 16 learned-model evaluation reports completed. The separate Human SL
reference evaluation hit its CPU cap after saving 11,204 of 13,022 positions.
The original runner stopped there, before its analysis and game panels.

Run only these already planned jobs, in this order:

1. Compute the 16 predeclared paired comparisons from the saved predictions.
   Analysis comes first so another game-panel cap cannot hide completed
   prediction evidence.
2. Run the original 12-game 6.5-komi panel for the baseline models.
3. Run the same 12-game panel for the combined-data models.

Keep the original seeds, models, final checkpoints, normalizers, positions,
opponents, inference settings, game limits and comparison rules. The game
panels still include the originally planned raw Human SL games. Do not retry
the failed recorded-position reference job. Do not fill missing ranks with
new games, select a new checkpoint or treat repeated reference games as
independent strength evidence. Incomplete or unresolved slots stay visible.

The absolute deadline remains **19 November 2026 at 14:11:48.993557 UTC**.
Expensive work stops 15 minutes earlier. Elapsed time includes the pause for
verification and this implementation. The cumulative 86,400-GPU-second ledger,
32-CPU-hour production cap and 16-GiB output cap remain unchanged. Analysis has
at most 1,800 CPU/elapsed seconds; each panel has at most 5,400. All three jobs
run on CPU. A job starts only if its full allowance and closing reserves fit.

Retain every original ledger entry. Add the first verifier's 5.984558735 charged
CPU seconds, its separately recorded 0.412146032 wait CPU seconds, and a
conservative 120-CPU-second reserve for saved-evidence inspection and launch
preparation. The first review directory counts toward output. New supervision,
tracking, workers and shutdown remain measured by the existing job ledger.
Synthetic implementation tests and independent code review are development
work; the original elapsed deadline still advances during them.

The one-use [completion config](../configs/e2_expanded_finish_v1.json) pins the
original plan, session, failure, fixed training decision, saved reference
progress and verification handoff. It also pins the current ledger before
any mutation. Reject a changed or unsettled ledger, changed saved evidence,
an unexpected failure, an already attempted completion or an expired deadline.
Use the same production lock. Obtain a fresh Astra/max review and deliver the
code before any remaining model query.

Keep `failure.json`, its stage records and the first review page unchanged.
Write separate `completion-session.json`, `completion-progress.json` and
`completion.json`. The last file retains the original failure and costs and
adds the new phase's status. Its overall experiment status stays **incomplete**,
even if all three remaining jobs finish, because the reference evaluation did
not finish. A later failure also keeps its own reason and partial artifacts.

The updated local review must show both statuses, the frozen training decision,
all available comparisons and game records, and the full completed population
reference report. Label the unfinished unseen-account reference counts as last
saved observations, not a completed comparison. Do not use its partial rows to
change the evaluation population. Preserve unsupported intervals, missing-rank
aggregates and sealed final/confirmation targets. This single-seed experiment
cannot promote a model or qualify a human rank; E2 readiness remains open.
