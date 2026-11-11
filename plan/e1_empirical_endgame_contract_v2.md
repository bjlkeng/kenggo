# E1 empirical endgame contract

Version 2 | 11 November 2026 | Bead `kenggo-6yr.10`

Use a pinned KataGo referee on a separate copy of a two-pass ending. Its cleanup
decisions provide an **empirical judgment**, not a proof of optimal Japanese play.
The shared engine preserves the original board, captures, full history and actual
actions. This implements the owner's instruction to finish E1 after the
recommendation to reuse an existing engine.

This revision replaces version 1's requirement for exact life/death and minimax
pass proofs as the only automatic route to usable endgames. The
[ordinary-play, source, mask, recurrence, learning-boundary and qualification
contracts](e1_game_and_data_contract.md) remain in force. The existing narrow
certifier remains available with its original claim. No engine move becomes a
human label, candidate action or raw-policy completion heuristic.

## Referee and evidence

The [referee profile](../manifests/e1_katago_referee_profile_v2.json) pins the
executable, model, configuration, runtime libraries and native verifier by SHA-256.
The [configuration](../configs/katago_referee_v1.cfg) uses one search thread and one
CPU inference thread, fixed deterministic-testing initialization, no neural
symmetry randomization, no root noise and no handicap advantage. Preserve all
pre-root history. Optimize the engine's score utility with win/loss utility zero.

Use the official **KataGo v1.18.1 Eigen AVX2** executable, revision
`92ee95c0a4b25fec214da00951ab69e97e207729`, and
`kata1-b10c128-s1141046784-d204142634`. The v1.18.2 release supplies a CUDA change
and directs other backends to v1.18.1. The separate native verifier remains
**v1.18.2**, revision `fd0723fdbc0e9d82cf269c9630af8c27c57c07c4`.
These are two recorded roles; runtime replay checks their compatibility for each
accepted transcript. ([Release](https://github.com/lightvector/KataGo/releases/tag/v1.18.2),
[model catalogue](https://katagotraining.org/networks/))

Both use simple ko, territory scoring, seki tax, no suicide, no button, no handicap
bonus, `friendlyPassOk=false`, and White komi 6.5. The referee enables formal
cleanup with `preventCleanupPhase=false` and `conservativePass=false`. The native
verifier uses the explicitly pinned history modes from version 1. GTP
`final_score` is never an outcome source.

For each ending:

1. Require two actual passes and `pending_adjudication`. Bind the entire original
   action record and snapshot, including candidate color and origin.
2. Replay the original history through the native verifier and compare the
   original board, captures, player and accepted actions.
3. Run two separate cleanup trajectories at **32 and 128 visits per decision**, with at most
   **64 referee moves per run**. Verify every proposed move with native full
   history, retaining both transcripts outside the actual game.
4. Accept only if both native transcripts finish with a scored result, no
   no-result, and the same winner. Always select the **128-visit transcript's
   native score** as the empirical margin. Preserve both odd integer half-point
   scores and report their difference. Neither source `RE` nor candidate color
   may select the preferred transcript or change acceptance.
5. Return `empirical_score`, distinct from certified `score`. Missing output,
   incompatible state, illegal output, no-result, winner disagreement or exhausted
   budgets returns unresolved. Do not retry until a preferred answer appears.

Each judgment is limited to **150,000 combined reserved search visits and native
replay positions, 120 CPU seconds and
180 wall seconds**. CPU includes the Python owner and its engine/verifier
processes. The profile also bounds request, response, stderr and evidence bytes.
These per-judgment limits do not authorize an unbounded experiment; record a
separate total budget and fixed input set before a run.

Evidence records artifact pins, the original snapshot, request/response digests,
both native cleanup transcripts, exact terminal arithmetic, empirical claim,
resource limits, actual cost and failures. A saved empirical outcome requires
both an independently supplied evidence-hash catalog and replay through the
pinned native verifier when loaded. An embedded hash cannot authorize itself.
Evidence-size exhaustion returns bounded failure metadata with the original
snapshot/profile hashes, resource usage and a hash of the omitted evidence.

The native score is exact for its formal rules and played cleanup transcript.
Its interpretation as the original human ending remains empirical: finite play
can make mistakes, both trajectories share one model, and KataGo's formal Japanese
rules have documented exceptions. Their score spread is not a confidence interval;
the higher search budget does not prove that its margin is more accurate.
([Rules comparison](https://github.com/lightvector/KataGo/blob/fd0723fdbc0e9d82cf269c9630af8c27c57c07c4/docs/rules.html#L495-L502))

The [initial profile](../manifests/e1_katago_referee_profile_v1.json) required exact
margin agreement and allowed 32 cleanup moves with 20,000 combined work units.
Its full exposed-source run resolved 85/108 endings, failing the 90% gate:
11 exhausted the work cap, 10 exhausted the move cap and two disagreed on margin.
That negative result remains intact. The revised profile separates the W/L target
from transcript-specific margins and budgets longer cleanup. It is a new policy,
not a passing reinterpretation of the original experiment. Keep the same source
resolution/winner thresholds and exact fixture requirements.

## Empirical pass judgment

Judge the exact original pre-pass state. At both **32 and 128 visits**, compare a
search forced to pass at the root with one prohibited from passing at the root.
Report the searched best alternative and `scoreLead`, in White's perspective,
then convert the score difference to the mover's perspective.

- Gains at most **0.5 points at both budgets** are empirically acceptable.
- Gains at least **1.5 points at both budgets** are empirically premature.
- Borderline, inconsistent, incomplete, no-result-risk or over-budget judgments
  are unjudgeable.

The grey interval surrounds the original one-point criterion. These values are
search estimates, not minimax bounds; neural `scoreStdev` is not a confidence
interval. Actual first and second passes have different histories and must be
queried as such. Counterfactual moves never enter actual action counts.

Use an available exact pass certificate before requesting an empirical judgment;
retain the method tag. The neural method returned unjudgeable on all six dense
settled/save-atari fixtures that the exact checker can handle. It judged 21 of the
24 exposed actual passes in the initial panel; three were borderline across the
budgets. These are coverage limitations. Keep the neural no-result-risk threshold
at 0.01 and count unknown judgments as adverse; do not relax it to fit fixtures.

Keep the existing 99% pass-coverage, 100 pass-bearing games per color and group,
and game-level qualification thresholds for E2/E5/E6. They now refer explicitly
to this empirical referee. Source fixtures and the E1 source-ending panel do not
qualify any A profile or B checkpoint. Unjudgeable passes remain adverse outcomes.

## Source labels and learning boundaries

Preserve original SGF results separately. Referee predictions never become SL
policy or human-outcome labels. An empirical disagreement is a diagnostic requiring
inspection; it does not silently replace the source result or claim a certified
contradiction. Certified contradictions retain the version-1 exclusion policy.

An accepted empirical score supplies terminal W/L and fixed-learner reward ±1,
with zero bootstrap. An unresolved ending has no payoff or valid learning
boundary and invalidates the collection batch. Administrative caps retain the
saved learner-to-move bootstrap state. No-result, timeouts and failures never become
draw rewards. Existing all-scheduled-game match denominators remain unchanged.

## Validation before E1 acceptance

Require exact settled, dead-stone and seki fixture scores at both budgets; test
perspectives, full history, forged/stale evidence, malformed output, no-result,
disagreement and process/budget failure. Run the pinned native rules comparison.

Freeze the source panel to all **108 two-pass endings** in the original 241 admitted
games, and freeze a hash-selected **24 actual pre-pass states** for a pass-method
coverage/cost check. No final-cohort data enters either panel. Require at least
**90%** of the ending panel to resolve and at least **95%** winner agreement among
resolved records with a known recorded scored result; report every disagreement
and all score-margin differences. These are limited implementation-readiness
checks on an already exposed sample, not an accuracy guarantee or candidate
qualification. Keep E1 open if they fail and investigate the cause.
