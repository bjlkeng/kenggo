# Independent match runner

The runner uses the shared full-history game state for every move. A frozen
schedule owns all game slots, including slots that never start. A missing,
capped or unresolved game remains `U`; it is never discarded or counted as a
draw. Half-point komi and the current ending contract produce no scored draws.
Weak policies are unranked setup opponents, with no strength claim.

## Freeze and execute a schedule

`weak_profile` provides uniform legal placement and capture-first policies.
Each uses its own seeded RNG, samples ties uniformly, passes only when no legal
placement exists, and never resigns. Profiles pin the implementation and source
revision. They use missing clocks and an unknown opponent rank.

`make_schedule` freezes profiles, context, referee, deadlines, provenance and
seed/color blocks. Each block contains both candidate colors for every arm,
using the same candidate/opponent seed pair across arms. The two players have
distinct random streams. Families receive equal weight; profiles receive equal
weight within a family. The setup scope permits weak opponents alone. A full
validation panel requires the E0 families and is a separate experiment.

`run_schedule(schedule, output_dir, budget=ExperimentBudget(...))` requires a
new directory and explicit CPU, elapsed-time and output-byte limits. It saves
the schedule before playing, then writes one SGF and JSON record per game.
Files are flushed and synced before publication under their final names;
publication never replaces an existing file. An interrupted temporary file is
excluded from record discovery, so completed records remain readable and the
interrupted slot stays unresolved.
Production games start from the empty board. Custom agent factories receive
the pinned profile and seed and return an object with `choose(game)`. Its
`Decision` contains an action, network-call count and GPU seconds when known.
Adapters are trusted local Python code; process isolation enforces deadlines,
not an untrusted-code sandbox.

Agent startup, requests and replies use bounded worker processes. The runner
times out stalled sends, partial replies and oversized frames, and kills the
worker process group on shutdown. CPU charges retain observed descendant work
after process exit, with a conservative shutdown allowance. Memory is sampled
RSS, not an exact peak. GPU allocation intervals remain the caller's
[resource-ledger](experiments.md) responsibility; per-decision reports cannot
replace the full allocation charge.

Exact ending and pass judgments run within their own limits. An optional
pinned empirical referee receives the original state when exact judging cannot
resolve it. Records retain every attempt, score margin, pass judgment, original
classification, final result, latency and cost. An unknown or opponent-caused
runtime failure does not become a verified candidate failure for a control's
lower bound. Short censored games have unknown length-tail status.

Generated SGFs contain actual moves and passes, the game-state payload and
schedule metadata, without invented clock values. `read_records` verifies
digests and independently replays records against their scheduled profiles,
limits, result and state. Empirical outcomes require explicit referee/profile
authorization during replay.

Replay also checks decision and candidate-pass coverage against the actual
history and derives the behavior flags from that evidence. Known exact pass
verdicts are rechecked within the frozen judgment limits. Known empirical pass
verdicts require their evidence digests in `authorized_hashes`, alongside any
authorized ending evidence; the configured referee must match the schedule.
Malformed agent responses and attributable initialization exceptions remain
recorded forfeits, with invalid compute metadata marked unknown.
The ready handshake separates process startup from agent initialization:
abrupt exits and timeouts after that handshake are attributable to the agent;
earlier process failures and explicit infrastructure errors remain unresolved.

## Summaries and held-out boundaries

`match_statistics.summarize` reports the scheduled denominator, W/L/U counts,
reasons, conservative score bounds, and results by profile, family and color.
It uses 10,000 paired whole-block bootstrap resamples by default, sharing
resample indices across arms. Zero-variance endpoints use the E0 Hoeffding
fallback; small block counts remain exploratory.

`reliability_bounds` uses exact binomial bounds within profile/color strata,
with the protocol's marginal alpha allocation. Unknown and missing candidate
judgments are adverse; a control's lower bound uses only verified failures.
`human_score_interval` resamples whole human opponents within one cohort and
weights those opponents equally. It does not treat repeated games against one
person as independent trials.

Confirmation schedules require a prior validation catalog and fresh seeds.
Final schedules use the reserved panel and cannot execute without a separately
pinned E7 release binding the selected candidates, profiles, schedule and
protocol. These gates do not authorize final evaluation during E2.

The runner records typed empirical score disagreements. The precise treatment
of same-winner margin spread in the 1% dispute gate must be frozen before
qualification; unknown dispute judgments remain adverse. Setup summaries do
not establish complete-game qualification, calibrated rank or useful gain.

## Bounded four-game implementation check

With the pinned E1 v3 input manifests present, from the repository root:

```bash
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.cache/python"
timeout 900s uv run --locked python -m kenggo.match_smoke \
  configs/e2_match_smoke_v1.json runs/e2-match-smoke-repeat \
  --config-sha256 366113d6fb4f0e436263a6c12a5f1756fde445d3df605e63f20c9ae96e0d3597
```

Use a new output directory and stage intended new source files so provenance
includes them. The check runs one uniform candidate against uniform and
capture-first opponents, with one color pair each and exact-only adjudication.
It replays all four saved records and checks the scheduled score denominator.
It uses no human target rows, external engine models or GPU allocation. Caps
and unresolved endings remain part of the result.

The frozen allowance is 600 CPU seconds, 900 elapsed seconds and 128 MiB of
match output. Thirty CPU seconds and 75 elapsed seconds are reserved outside
the match budget for tracking and shutdown. W&B receives metadata and summary
metrics; generated SGFs stay local. All failed or incomplete outputs remain on
disk. CPU accounting covers known wrapper/child work and conservative runner
charges; the tracking service's CPU is not fully measured. This check is not
the E0 throughput probe or the full validation panel.
