# E2 review: learning curves and previous runs

Bead: `kenggo-ydo.15`.

The [updated review page](../runs/e2-learning-review-2026-11-20/review.html)
adds training curves, training/validation metrics and overall results from
the three relevant runs. Each run has a short description of its data,
training amount, evaluation changes and limitations. The original game
controls, notes, metric breakdowns and evidence links still work.

The small pilot has 20 saved epoch averages per model. Expanded training has
626 saved update-block averages per model. The wider evaluation rerun uses
those same four models and curves; it did not train again. The page checks
that reuse and keeps the original training records unchanged.

Validation was recorded only at the final checkpoint. The charts therefore
show training lines and final validation dots, with separate NLL, move
agreement and outcome-MSE fields. Intermediate validation, training move
agreement and full-training-set evaluation are explicitly unrecorded.
Training losses average the changing model's learning batches; validation
uses the frozen final model and rank/game-side weighting. Their gap alone
does not quantify overfitting.

The earlier runs used different development samples and rank coverage.
History keeps each run's own scope, masks, unavailable estimates and
uncertainty. It does not calculate a paired gain across runs. The experiment
conclusion is unchanged: added data helped move prediction in the supported
review scope, while passing, endings and the rank/value tradeoff keep E2
readiness (`kenggo-ydo.6`) open. This update ran no training or inference.

Both model-review exporters now use the same component. The data-only
template explicitly marks model metrics as not applicable. `AGENTS.md`
requires these sections in future experiment reviews and asks future
training plans to budget checkpoint-level train/validation logging.
[Export instructions](../docs/review_pages.md) explain the pinned history
catalog and how to add later runs.

The [manifest](../manifests/e2_learning_review_v1.json) pins the new page,
sources and checks. All three source pages remain unchanged, as do the
current page's original embedded experiment data and script. Direct saved
measurement checks passed 6,985 assertions. Offline browser checks passed
919 assertions at 1,000 and 390 pixels, including curves, tables, selectors,
end/pass controls for all 24 games, evidence links and notes download. No script errors,
external requests or page overflow occurred; automated notes were cleared.
The compact inline curve view also passed eight model/width cases.

Focused Python checks passed 36 tests. The first broad run encountered
12 setup errors because new source files had not yet been staged; all
14 tests in the affected audit module passed after staging. Further
validation and the fresh independent completion verdict are recorded in
Beads. No native replay, protected target access or new model evaluation
was needed for this presentation change.

1. For **`.15`**, select a run and model in the new top section. Inspect
   both loss curves and their final validation points to see what was
   actually measured during training.
2. For **`.15`**, inspect “Overall metrics across runs” and the descriptions
   below it. Check rank scopes before interpreting differences; the rerun
   used the same weights on a wider evaluation sample.
3. For **E2 readiness (`.6`)**, continue reviewing flagged passes and endings
   in the original controls below, then download and share notes. The
   new plots add context but do not resolve those behavior problems.
