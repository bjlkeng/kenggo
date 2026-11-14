# E2 Human SL adapter check

Bead: `kenggo-ydo.4`. Review base:
`c14262e03d48385cc4aaeed484b8023e5d982ad3`.

The pinned KataGo Human SL adapter passed its 12-query CPU implementation
check. It preserves raw policy probabilities, including pass, validates the
legal mask against the shared rules engine and converts the explicit human
value to the mover's view. Five repeated position/rank pairs returned exactly
the same policy, value and seeded samples. All 1,536 sampled actions were legal.

This supplies an external reference for the later BASE comparison. It supplies
no held-out imitation score, completed-game result, calibrated strength or
qualified opponent profile. No training, E0 throughput-probe allocation or SL
pilot GPU allowance was spent. Confirmation and final targets remain unopened.

## Reproduction and evidence

The [usage guide](../docs/human_reference.md) gives setup, API and bounded-check
commands. The [evidence manifest](../manifests/e2_human_reference_v1.json) pins
source, configuration, logs, raw replies and sampled actions. Large models and
run output stay in ignored paths.

The successful run is
[raw-human-cpu-b41712010ab248caa2f12698690ddc37](https://forge.coreweave.com/wandb/bjlkeng/kenggo/runs/b41712010ab248caa2f12698690ddc37),
with local output `runs/e2-human-reference-fixed-2026-11-13`. It ran source base
`c14262e` plus patch
`d8297cc87728c5998388030bc7e65a3e0d22207f0e8c811ce396f2ce466a8221`.

KataGo reports version 1.18.1 and revision
`92ee95c0a4b25fec214da00951ab69e97e207729`. The auxiliary checkpoint is
`b18c384nbt-humanv0`, SHA-256
`637746e44f0efe00ad1245a50aa9bbf0716efe364c43965ead97bd6835d84ab5`.
The [profile](../manifests/katago_human_profile_v1.json) also pins the b10 main
checkpoint, Eigen executable, config and bundled libraries. Main-model outputs
are never substituted for `humanPolicy` or `humanWinrate`.

The fixed queries cover an empty Black-to-play board, a one-move White-to-play
board, an explicit pass, an immediate simple-ko prohibition and the same ko
after a pass and remote move. Each is queried twice at `5k`; the empty board
also uses `15k` and `1d`. Full actual history and the native Japanese-rule object
are supplied. Exact repeat checks include cached native execution; they are
implementation evidence, not an inference-throughput measurement.

## Costs and retained failures

The official human-model download used three HTTP requests, 99,110,201 response
bytes, 0.615 calling-process CPU seconds, 2.646 elapsed seconds and zero GPU.
The [acquisition record](../manifests/e2_human_acquisition_v1.json) retains the
release metadata, 99,066,230-byte checkpoint and hash. It stayed within the
separate eight-request, 128 MiB, 30 CPU-second and 180 elapsed-second limits.

The CPU implementation check had a combined allowance of 12 analysis queries,
180 CPU seconds, 300 elapsed seconds, 16 MiB of output and zero GPU. All
attempts used **16.715 CPU seconds and 21.498 elapsed seconds**, including two
failed metadata requests and the successful run's startup/shutdown. There were
12 analysis requests and three version requests in total. The repaired run
used 7.846 CPU seconds and 12.468 elapsed seconds. Retained run output is below
one MiB. No analysis request was retried or discarded.

The first start stopped on KataGo's warning that no startup Human SL profile
was configured. That attempt exposed a missing diagnostic: the trace saved the
warning category but omitted stderr. A regression check reproduced that gap,
and the trace now retains a bounded stderr tail. One metadata-only diagnosis
then identified the warning. The config now supplies an explicit startup
profile; every prediction still requires a rank and overrides it. Other native
warnings still fail the client. The [preflight record](../manifests/e2_human_preflight_v1.json)
retains both failures and deducts their costs from the repaired check's budget.

CPU accounting includes the calling process and reaped native child. W&B's
background service CPU is not fully measured. Native forward-call counts remain
unknown because the interface exposes no exact cache-aware counter. These
limits prohibit a neural-call or throughput claim from this check.

## Validation and limits

Both pinned Python environments passed **315 tests**, with the native rules
reference enabled and zero skips. The 30 adapter tests cover malformed output,
pin/config binding, history and coordinate mapping, ko/pass behavior, both value
perspectives, seeded sampling, cumulative budgets, CPU retention after reaping,
shutdown output and actual match-worker cleanup. Budget exhaustion produces an
unresolved match result. Lock/sync checks, CPU smoke, package build, local links
and `git diff --check` passed. Independent review is recorded in the owning bead.

Independent review found that artifact/path preparation and native launch
failures could be scored as agent forfeits. The complete preparation phase now
retains failure costs and reports infrastructure errors, leaving the result U.
The factory starts and verifies the native process before returning a playing
agent. Regression tests cover missing/corrupted files, execute permissions,
artifact/runtime symlink loops and failed-client costs, while preserving
attribution for invalid replies from a running agent. A further review found
that oversized JSON numbers could bypass cleanup; protocol validation now
routes every ordinary exception through failure retention and closes the
client. Tests cover both numerical fields and a real synthetic child, checking
that its process stops and no second request is sent. The complete suites
passed again after the repairs. The actual 12-query run predates these setup
and protocol error-handling fixes;
its prediction path, original source patch and costs are retained unchanged.

The guide records the reference's fixed KGS/2020-era metadata, nominal clock and
same-rank-opponent defaults; its full-history input differs from the compact
learner's current zero-history input. The project's games are from 2025, after
the model release, but human control and training overlap remain unverified.
The reference rank labels are conditioning settings. No pass correction or
search-policy baseline was introduced, and recorded-outcome calibration remains
part of the later held-out comparison.
