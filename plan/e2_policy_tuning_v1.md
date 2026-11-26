# E2 policy-only tuning with larger validation

Bead: `kenggo-ydo.19`. The owner authorized this study on 25 November 2026:
use saved games for larger validation, remove winner-value loss, and tune a few
settings within three days of wall-clock training time. This protocol gives the
whole production study, including preparation, evaluation and publication, one
72-hour allowance. It does not extend the earlier seven-day study.

## Question and fixed inputs

Can further policy-only training improve prediction of human moves? Compare three
initial learning rates and one weight-decay setting. Cross-entropy, also called
negative log likelihood (NLL), measures the probability assigned to the recorded
human move; lower is better. Agreement is the fraction whose most likely move
matches the record; higher is better. Neither establishes playing strength.

All four trials start from the same selected 4M-data checkpoint from the completed
[progressive study](../reports/e2_progressive_training_2026-11-25.md): lifetime
125,000 updates and 16 million move presentations. Keep its eight residual blocks,
128 channels, rank input, board encoding, nominal-clock dropout and batch size 128.
This is continuation from joint-loss weights, not policy-only training from random
initialization. The architecture has no extra move-history planes; legal moves
still use the full rules history.

Fitting uses the same 3,999,997 moves from 18,126 complete games, including Japanese
0.5 and 6.5 komi. No acquisition, augmentation change or data growth is included.
Keep the original 2M training-only normalizer and verify the established 4M lineage.
The config pins the data, normalizer, model, source protocol and prior evidence.

## Larger stopping validation

Choose complete unseen-account development games using metadata only. Exclude
every fitting game, the fixed 539-game comparison, the previous 52 stopping games,
protected memberships, and matching source/SGF/action-sequence/trajectory identities.
The seven previously reviewed 11k/6.5 supplements remain validation-only.
Three historical protected records have no saved trajectory fingerprint. Compare
their existing SGF and action-sequence hashes and every available trajectory;
record the missing metadata without opening or replaying protected targets.

The fixed selector aims for 30 games in each rank/komi group from 12k through 1d.
Process the smallest groups first and prefer new mover accounts, then accounts
with fewer selected games, then a seeded hash. A whole game's two ranks may serve
two groups. Freeze 577 games / 125,998 moves, including 124,851 in-scope moves and
578 accounts. The source-ID digest is
`a93fe207fdc8e538d2ebe0ba67628b179c33ac287440b662d18ef86ccd0828a3`.
The seed is `policy-only-stopping-20261125-v1`.

Coverage remains uneven: 11k/6.5 has only five games, 7k/6.5 has 23, and 1d/6.5 has 19.
The 12k/0.5 group has only six mover accounts. Account spreading does not make
games independent. Retain complete games, show the 1,147 weaker-rank rows as
excluded from the declared aggregate, and disclose support by group.

Use this exact validation sample for every checkpoint and trial. A separate saved
training-game diagnostic gives a comparable training curve. Their evaluation
metrics weight ranks equally, then game sides equally within rank, then valid
moves equally within side. Online minibatch loss is separately labelled because
it uses a different sample, dropout and weighting. The stopping sample and final
comparison share no games; account overlap can remain. Final and confirmation
targets stay closed.

## Loss, settings and stopping

Only masked legal-move cross-entropy contributes gradients. Winner-value labels
cannot change an update. Freeze the value-head parameters and exclude them from
the optimizer; report value error only as an untrained diagnostic. Its predictions
can still change as the shared board representation changes. Do not use value
error to select a checkpoint or veto a policy improvement. This head is not
qualified for search.

Reset optimizer state once when each trial starts. All trials use AdamW, the same
parent weights, sampler seed, dropout RNG state and data order. Resume optimizer,
RNG, sampler, stopping and learning-rate state exactly from subsequent checkpoints.
The only initial differences are:

| Trial | Initial learning rate | Weight decay |
|---|---:|---:|
| lr-1e-3 |0.001|0|
| lr-3e-4 |0.0003|0|
| lr-1e-4 |0.0001|0|
| lr-3e-4-decay |0.0003|0.0001|

Evaluate the initial checkpoint and every 7,813 updates, or 1,000,064 presentations
(about one quarter of a pass through fitting data). Keep recovery checkpoints
every 1,000 updates. Choose the raw minimum validation NLL, including checkpoint 0.
For patience, a meaningful improvement is at least 0.002 NLL against the last
meaningful best. After two intervals without that improvement, multiply learning
rate by 0.3, down to 0.000001. Stop a trial after at least eight intervals and six
successive intervals without meaningful improvement, or after 48 intervals.
The shared budget can stop it earlier. Do not tune these rules after inspecting
results. No further hyperparameter search is authorized within this protocol.

Run trials sequentially on one GPU, rotating which active trial goes first.
Reserve enough budget for a full next cycle before starting it. Each trial stops
independently; report actual exposure and compute rather than claiming identical
training amounts for selected checkpoints. A failed interval stops the tuning
loop. Preserve partial evidence; no automatic retry or whole-study restart.

## Selection and separate assessment

Seal the selected trial and every trial's best checkpoint using stopping NLL
before computing the fixed comparison. Ties use the table order. Evaluate all
four selected checkpoints and the common parent on the unchanged 539-game sample:
120,450 positions total, 111,591 within 12k–1d, with population/unseen-account and
0.5/6.5-komi reports. Compare each trial with the parent and the 0.001 policy-only
baseline using matched saved positions and the existing game/account uncertainty
method. Report NLL and agreement, intervals and support. Do not choose a new winner
from these assessment results or feed them into further tuning.

This is exploratory development evidence. The parent and fixed assessment have
been used in earlier studies; four trials and one seed do not provide independent
confirmation. No new complete-game or human-rank qualification runs here. Earlier
premature-pass findings and E2 readiness remain unresolved.
Every new trial is policy-only. There is no equally trained joint-loss control,
so a difference from the parent combines further training, loss and optimizer
changes; it does not isolate the effect of removing value loss.

## Budget and durable execution

One immutable launch time starts a 259,200-second wall-clock deadline. CPU cap:
388,800 measured/charged seconds; GPU allocation cap: 259,200 seconds; local output:
96 GiB. These caps include failed jobs and tracking. Stop launching training six
hours before the deadline to leave time for assessment, review export and checks.
Publication reserves also protect 18,000 CPU seconds, 12,000 GPU seconds and 8 GiB.
No parallel training or external collection. Check both UTC and monotonic clocks.

Saved prior 4M intervals took 5,082–5,790 seconds for 31,250 updates. A new 7,813-update
interval has a conservative 3,000-second outer cap, including loading and the larger
validation. Initial evaluation gets 1,800 seconds; preparation gets 3,600; each fixed
assessment gets 1,800; comparisons and publication each get 1,800. These are caps,
not fresh speed claims. If a complete next cycle cannot fit, stop and report.

Use a durable user-systemd service with `Restart=no`, `KillMode=control-group` and
a runtime limit no later than the original deadline. The supervisor is the only
ledger writer. The output directory is unique and refuses reuse. Unexpected exits
need retained-evidence inspection and a fresh reviewed recovery; never reset the
allowance. Freeze source while any worker is live. Use small health checks of
actual process identity, stage progress, saved measurements and limits.

The owner page is exported as an accounted worker from an immutable pre-publication
snapshot. Its cost label names that boundary. The final `summary.json` and ledger
include publication and supervisor charges. Later review must reconcile them;
do not describe the page's earlier cost snapshot as the final total.

## Evidence and acceptance

Before launch, test policy-gradient independence, frozen value weights under
decay, unchanged legacy joint loss, exact checkpoint/schedule continuation,
whole-game selection and leakage barriers, budget stopping, failure preservation
and report rendering. Run the repository checks and a fresh independent
GPT-6-Astra/max review. A clean verdict is required before production.

The final self-contained HTML uses the shared learning-review component. Include
saved training and validation curves, learning-rate changes, all four trials,
selected checkpoints, separate assessment and relevant earlier runs. Mark missing
measurements, unequal exposure, sparse support and diagnostic-only value metrics.
After production, verify actual saved evidence and the page, obtain a separate
fresh completion review, and deliver the versioned report and manifest. A negative
or budget-limited result may complete this study; it cannot close E2 readiness.
