# Grow the data and stop training when validation levels off

Bead: `kenggo-ydo.16`. The owner approved **seven days of wall time** for
this study. The allowance covers every production attempt, data preparation,
download, training, evaluation and final evidence export. It starts when the
reviewed runner launches and never resets between rounds. This is a new
allowance; earlier E2 costs remain recorded separately.

## Question and comparison

The current rank-conditioned A model learned from 2,000,000 moves in 9,135
complete games. Its wider development evaluation suggests that more data
helped move prediction. We will try 4 million fitting moves first, then double
again only if the previous comparison supports continuing and time remains.
Keep Japanese 6.5- and 0.5-komi games, the existing admission policy, account
cohorts, date windows, owner exclusions, outcome masks and full replay history.
Complete games may leave the total slightly below its target. Report that
shortfall; never fill it with duplicated moves or partial games.

Use cached games first. Earlier row caps left 7,401 replayed training games
with 1,644,283 recorded actions unencoded. Further cached archive candidates
still need admission, duplicate and replay checks. A source count is not a
usable training yield. If a round exhausts the cache, bounded acquisition may
add candidates under the same policy and shared deadline.

For each data tier, start two rank-conditioned models from the same selected
parent checkpoint, including its Adam state. One continues on the previous
dataset; the other uses the enlarged dataset. Both receive the same scheduled
updates and checkpoint opportunities. Alternate their execution order. An
evaluation interval contains enough 128-move batches for one pass through the
larger dataset. The old-data control therefore usually sees two passes.
Record actual presentations and selected checkpoint steps for both models.
Selected checkpoints can occur at different steps: the comparison measures
the two training procedures with equal opportunities, not identical selected
exposure. Further rounds use the enlarged model's selected checkpoint only
after the comparison gate passes.

Keep the existing eight residual blocks, 128 channels, zero model history,
rank inputs, 0.1 clock dropout, Adam learning rate 0.001, and policy cross
entropy plus masked outcome-value mean squared error. This is supervised A
training. Nothing is transferred to B. Preserve the original 2M training-only
normalization statistics for both arms. Their true fitting manifest remains
pinned; verify that every fitting game and original training shard remains
inside the new training set. Do not relabel these statistics as a new fit.

## Three distinct data uses

1. **Fitting:** the previous or enlarged training dataset. Old fitting games
   never become validation games. New games must respect the frozen cohorts
   and chronology before their targets are opened.
2. **Epoch validation:** freeze a fresh whole-game sample from the existing
   unseen-account development pool for each data tier, before querying a
   checkpoint. It includes both komi values and the predeclared 12k–1d mover
   ranks. Exclude every fixed-comparison game and every earlier stopping game.
   Keep all these games outside fitting in every round. Freeze a small,
   separate training diagnostic sample from the original training pool too.
3. **Fixed comparison:** retain the four wider development samples from the
   19 November rerun: 539 games / 120,450 positions. Use the same row keys,
   masks and 12k–1d reporting scope (111,591 positions) across tiers. These
   development results guide the next round; they are not independent final
   testing. Confirmation and final targets stay closed.

Samples contain complete games. Reporting the 12k–1d scope explicitly excludes
weaker-rank positions without deleting them from the stored diagnostic sample.
Show that excluded count and all-rank diagnostics. Missing required rank
coverage or nonfinite metrics stops the study as insufficient evidence; it
must not silently switch to a pooled mean. Preserve the E0 average: positions
within a game side, game sides within a mover rank, then equal mover ranks.
Show valid outcome-label counts separately from policy counts.

## Checkpoints and stopping

Evaluate the parent at epoch zero. After every matched interval, record online
training losses and frozen-checkpoint training-diagnostic and validation NLL,
move agreement, and masked value MSE. Online losses use changing weights and
clock dropout; diagnostic losses use fixed weights with dropout disabled.
Do not label the diagnostic subset as the full training population. Evaluation
must preserve all training RNG states and module modes.

Select each arm's lowest finite validation NLL, including epoch zero. For the
patience counter, a decrease of at least **0.002 NLL** resets the reference.
Small improvements still update the best checkpoint. Stop the pair once both
arms have gone **three intervals** without a meaningful decrease or at the
remaining budget boundary. A defensive 1,000-interval ceiling exceeds what
this machine can run in a week; reaching it is a protocol stop, not evidence
of a plateau. Save periodic recovery checkpoints
and an immutable record linking each completed interval to its metrics and
stopping state. A partial interval cannot update patience. Retain failures and
their costs. The supervisor refuses a second launch of this plan, including
into a different output folder. Whole-study restart is not implemented. Worker
checkpoints support recovery, but any later recovery orchestration must first
preserve the saved session, original deadline and cumulative ledger; it cannot
start a new allowance. An interrupted pair gets an explicit partial result and
the last completed pair's comparison.

After a tier, evaluate both selected checkpoints on the fixed comparison
positions. Compare them with each other and the selected parent. Use paired
game/account uncertainty and E0's 0.02 useful NLL gain and 0.02 allowed value
MSE/NLL regression. Continue data growth only when the larger-data arm has a
useful paired NLL improvement against both control and parent in the primary
unseen-account 6.5-komi scope, and no supported fixed-comparison scope shows
a material NLL or value regression. Retain inconclusive comparisons as such;
do not spend another doubling on an assumed gain. All decisions are exploratory
and use one training seed. No profile promotion or human-rank claim follows.

## Limits and evidence

Use one RTX 3090. The hard overall limits are 604,800 elapsed seconds,
604,800 allocated GPU seconds, 604,800 measured CPU seconds and 160 GiB of
new output. Reserve the final six hours, 18,000 CPU seconds, 12,000 allocated
GPU seconds and 8 GiB for the three fixed assessments, paired comparison,
publication and verification. Before a complete paired interval, check that both jobs and
their checkpoint/evaluation reserves fit. Memory headroom can stop growth;
do not swap an oversized corpus into an unbounded job. Acquisition keeps the
existing per-request byte limits, at least five seconds between requests,
at most two retries, and at most 2,400 requests / 1 GiB / six hours per
collection attempt. The shared deadline and cumulative limits override them.

Retain original artifacts unchanged. Register configuration, source pins,
dataset lineage, sample IDs, losses, checkpoint choices, failures and costs.
Reuse saved Human SL evidence only for exactly matching fixed positions; no
new Human SL inference is required. Known premature passing and the earlier
0.5-komi value regression remain visible. This study does not repeat the
complete-game qualification panel or resolve E2 readiness by prediction alone.

Produce a new self-contained HTML review with training and validation curves,
overall metrics from every relevant earlier run, plain descriptions, paired
comparisons, budgets, missing evidence and next-step hypotheses. Use the shared
learning-review component. Obtain fresh independent Astra/max review before
production and again for completion. Deliver source and Beads; leave E2 and
its readiness gate open unless their separate acceptance evidence is met.
