# Learning curves and run history in review pages

Model review pages show saved training curves, separate training and validation
metrics, and earlier runs with short descriptions. The BASE and expanded E2
exporters include the shared learning-review section. Their command-line default
is [the E2 history catalog](../configs/e2_review_history_v1.json).

For a completed, sealed page, create a new copy instead of overwriting the
original. From the repository root:

```bash
.venv/bin/python tools/review_learning.py rerun-2026-11-19 \
  runs/e2-learning-review-2026-11-20/review.html
```

The output path and its `.manifest.json` must be unused. The command checks
each source page's hash, reads its embedded saved results, and preserves the
original metrics, game controls and notes. Relative evidence links are made
absolute so they still work from the new folder. Open the HTML locally; it
needs no server or network connection.

For a new run, add its reviewed page to the catalog after export. List runs
oldest first, with a unique ID, source HTML path/hash, embedded `bundle_id`,
short chart label, description and relevant notes. Include training size,
training amount, model reuse, changed evaluation samples and incomplete stages.
When a run is already in the catalog, only entries before it enter its history.
For an unlisted new run, all supplied entries are treated as earlier runs;
use a dedicated catalog if that is not true. Both exporters accept `--history`.
Python callers of the expanded exporter include the default catalog unless
they explicitly pass `history=None` for an isolated fixture or a first run.

The current E2 evidence has training losses per epoch or update block, but
only final-checkpoint validation. The page therefore shows training lines and
final validation dots. It labels intermediate validation, full-training-set
evaluation and training agreement as unrecorded. It never fills those gaps
with zeros or invented curves. Plan checkpoint-level train/validation logging
before future training, within the approved experiment budget.

Training points average changing models on the batches used to learn, with
training-time dropout. Validation uses the frozen final model and the saved
rank/game-side weighting. Show these definitions and outcome masks alongside
the metrics; do not interpret the gap alone as overfitting. Keep each historical
run's original rank list, sample, missing metrics and uncertainty. Differences
across changed samples are historical context, not paired model gains.

For a data-only or planning review, state that no training occurred and that
learning curves/model metrics are not applicable. Do not borrow an earlier
model's curve as if it came from that data run. Preserve old sealed artifacts
and record the new page, inputs, code hashes and browser checks in the owning
bead and reproducibility manifest.
