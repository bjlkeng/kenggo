# Finish a useful E2 review with the saved models

Bead: `kenggo-ydo.14`. The owner approved a rerun with up to **one week of
wall-clock time**. This plan keeps the four models from the
[expanded comparison](e2_expanded_comparison_v1.md) and starts a separate,
bounded evaluation. The old incomplete result remains unchanged.

## What this rerun answers

All four models finished training. The missing Human SL report and four empty
game slots came from smaller limits inside the earlier run. Some unavailable
statistics came from missing ranks or too few independent accounts. More time
alone cannot repair those data limits.

Reuse the four fixed final checkpoints, their training-only normalizers,
architecture, rank treatment and seed `20261118`. Do not train, download games,
change raw move sampling or open confirmation/final targets. The question is
whether the existing models predict held-out human moves and how their raw
games fail. This is exploratory development work after earlier results were
seen; it cannot promote a model or claim a human rank.

## Freeze a wider sample before querying models

Use the combined dataset's two development roles, separately at 6.5 and
0.5 komi. Reuse the original metadata-only hash ordering, taking up to ten
whole games with moves from each rank. Union shared games; retain every move
from each selected game. Limits are 160 games and 50,000 positions per set.
The seed, source, role, komi and ranks determine selection; outcomes and
predictions do not. The pinned inputs give:

| Development set | Games | Positions | Positions at 12k–1d |
| --- | ---: | ---: | ---: |
| Population, 6.5 komi | 136 | 31,233 | 29,119 |
| Population, 0.5 komi | 142 | 31,784 | 27,558 |
| Unseen accounts, 6.5 komi | 127 | 27,949 | 27,268 |
| Unseen accounts, 0.5 komi | 134 | 29,484 | 27,646 |

This gives 120,450 positions and 481,800 learned-model predictions. Human SL
evaluates the same 59,182 positions in the two 6.5 sets. Its pinned profile
does not support 0.5 komi; that reference remains explicitly out of scope.

Keep the full 15k–1d reports, including unavailable all-rank estimates.
There are no 15k games in population/6.5 or unseen-account/0.5. Add a separately
named **12k–1d review comparison**, filtering reporting rows only after all
selected outputs have been checked. Keep exact included and excluded row
counts, and the weaker-rank results. This does not change the original study
or assert that 12k is a playable difficulty setting.

This scope and sample size were chosen from metadata before new predictions.
Ten games per rank gives maximum account weights of 6.25–9.50% for policy and
7.29–9.42% for valid outcomes in the new aggregate, with at least 131 accounts.
The wider 14k–1d and 13k–1d scopes fail concentration checks. Taking twenty
games per rank still leaves an account above 10% in population/0.5; using the
full available data does not resolve that dependence. This is a fixed sample,
not an account-balanced reweighting or a guarantee of useful precision.

Use the same equal-rank, equal-game-side and equal-position means, paired
contrasts, dyadic account uncertainty, masked outcome MSE, calibration and
32 temperature-1 samples per position. Report ordinary intervals only when
their variance and support checks pass. All selected per-rank cells still
have fewer than 100 game sides; cell-level promotion remains unsupported.
Keep the [E0](e0_evaluation_protocol.md) gain and regression rules: 0.02 NLL
reduction is useful; allowed supported-group NLL/MSE regression is at most
0.02. One training seed remains insufficient for promotion.

## Complete the game schedule without changing the referee

Rerun the same two 12-slot panels: rank on, rank off and Human SL, both colors,
against uniform legal and capture-first opponents. Keep the original seeds,
raw temperature-1 policies, 5k input, 800-action cap, pass judgments and pinned
referee. Identical random-input blocks repeat earlier contexts; do not count
them as extra independent strength evidence.

Raise the execution allowance to one hour per game and twelve CPU/elapsed
hours per panel. Keep another hour for native replay verification and reports.
The Human SL per-game CPU/elapsed allowance rises to one hour; query and byte
limits stay explicit. Referee search visits, 64-move cleanup, thresholds and
per-judgment budgets stay unchanged. An unresolved ending or unjudgeable pass
is a valid reported limitation, not a missing job to retry until it passes.

The prior nine unresolved endings exhausted cleanup moves. Of 303 unknown
pass judgments, only two came from the experiment budget; the other 301 came
from disagreement, no legal non-pass comparison or uncertain no-result risk.
Changing those judgments would be a separate referee study.

## Limits and execution

The one-use launch starts a new hard **604,800-second elapsed deadline**,
including selection, queries, saved-evidence verification and publication.
Stop production at least one hour before it. Use at most 144 CPU hours and
32 GiB of new output. Reserve 30 CPU minutes and 512 MiB within those limits
for final verification/publication. Development tests and code review are
separate; no new production starts until prelaunch review is clean.

After the producer stops, `kenggo.expanded_review_finalize` runs a pinned,
saved-evidence helper under the same lock and original deadline. The helper
has 1,200 CPU seconds, 2,400 elapsed seconds and 384 MiB of output. Charge
another 600 CPU seconds conservatively for wrapper setup, monitoring and
closing writes; a parent CPU timer stops active work before that reserve is
spent. Total finalization output is capped at 512 MiB, including a 1 MiB
closing reserve. These costs remain inside the 144-hour/32-GiB limits.

The wrapper checks the unchanged source, producer pins and settled ledger.
It rejects an expired deadline, insufficient remaining allowance or an
existing finalization directory. It records failure without automatic retry.
The helper only verifies saved results, exports the page and checks it in an
offline browser; it must not query models, replay native games or open sealed
targets. Both `verification.json` and `browser-checks.json` must say `passed`.
The helper and all outputs remain local for completion review.

The finalizer command is:

```bash
.venv/bin/python -m kenggo.expanded_review_finalize \
  runs/e2-review-rerun-2026-11-19/summary.json /absolute/path/to/verified-helper.py \
  --producer-sha256 PRODUCER_SHA256 --helper-sha256 HELPER_SHA256
```

Use the pinned `failure.json` if production fails. Output goes to
`runs/e2-review-rerun-review-2026-11-19/worker/`. The page shows production
costs and links to `final-resources.json` beside it, which the wrapper writes
after export/browser checks with total costs, deadline checks and artifact
pins. Later versioned reporting and independent code review are development
work, separate from this production allowance.

| Job | CPU and elapsed allowance | New output allowance |
| --- | ---: | ---: |
| Prepare all four sets | 6 hours | 2 GiB |
| Each of four learned evaluations | 6 hours | 4 GiB |
| Paired analysis | 6 hours | 2 GiB |
| Each game panel, including replay/reports | 13 hours | 2 GiB |
| Human SL recorded-position evaluation | 36 hours | 8 GiB |

The Human SL client has its own 24-CPU/elapsed-hour allowance within its job,
59,182 queries, 512 MiB of requests, 1 GiB of responses and 2 GiB of trace data.
The outer job includes preparation and reporting. These are ceilings, not a
promise to spend the week. Budget failures remain visible and are not retried
automatically. Stop if a required job fails or its full remaining allowance
cannot fit. Run analysis and game panels before the external reference so a
late reference failure cannot hide already available model/game results.

Run one CPU worker at a time with no GPU allocation. Preserve all charges and
the cumulative 86,400-GPU-second SL ledger; its existing 22,270.127059-second
charge is not reset. Pin the settled ledger at launch. Record the new attempt,
supervisor, native children, tracking, failures and final verification costs.
Earlier production/output is historical and is not charged again to this new
CPU/output allowance. The global seven-day deadline overrides every job cap.

The supervisor keeps 20 ms process/CPU/wall checks. Full output-directory
scans occur at most once per second during work, plus a final scan. Output
limits are sampled with that delay; a final overrun fails the job. This removes
repeated scanning work while keeping resource failures explicit. Earlier
Human SL supervision used 4,979 CPU seconds versus 4,022 child CPU seconds.
No end-to-end speedup is claimed before measurement.

## Delivery and owner review

Freeze the [config](../configs/e2_review_rerun_v1.json), source, dataset and
checkpoint pins before launch. Reject changed inputs, an unsettled ledger or
reuse of the output directory. Obtain fresh independent Astra/max prelaunch
review, then deliver source and Beads before starting the command.

After the run, verify the saved evidence and generate one local HTML page.
It must distinguish the new 12k–1d comparison from all-rank diagnostics,
show the completed reference reports, expose every scheduled game and retain
real failures. Explain why any remaining result is unavailable.
Keep long account-weight and influence maps in their pinned source JSON;
the page links those files and embeds estimates, intervals and support
summaries. A late Human SL or tracking failure may still leave complete set
reports: validate and show those reports while keeping the stage failed.

1. For **the review rerun (`kenggo-ydo.14`)**, compare the four models and the
   Human SL reference on the declared positions. Check uncertainty before
   treating a point estimate as a useful improvement.
2. For **E2 readiness (`kenggo-ydo.6`)**, replay flagged passes and endings.
   Complete evaluation jobs do not establish that the models play reliably.
3. Download notes from the page and send concerns or the next-scope decision.
   No owner approval or readiness pass is inferred from producing the page.

Publish a concise report and pinned manifest, finish fresh independent
completion review, and sync source and Beads. E2 readiness stays open unless
its separate acceptance criteria pass.
