# Resume the E2 policy study after its setup failure

This repairs `kenggo-ydo.19`; it does not start another experiment or allowance.
The original [study protocol](e2_policy_tuning_v1.md), fitting data, validation
selection, four trials and stopping rules remain in force.

The first service started on 26 November 2026 at 01:35:51 UTC. GPU readiness
passed. Preparation then stopped because the shared tracking interface required
three input pins that the controller had omitted: data, split and rules.
Publication met the same error. No training updates or validation preparation ran.
The failed service and all its workers stopped before this repair.

The configuration repair supplies these common pins for every job while keeping
job-specific overrides, such as the larger fitting dataset. A regression test
uses the real provenance interface for preparation, comparison, publication and
training configs; it reproduces the original failure and passes after the fix.

The recovery has one permitted output directory and accepts only the pinned
pre-training failure. It verifies that the old service is stopped, readiness is
the only completed stage, all four trials remain pending, the ledger is settled,
and the original allowance has time left. It refuses existing recovery output,
changed evidence, live workers, outstanding reservations or reused recovery IDs.

The controller opens the original ledger, keeps its charges, and uses distinct
IDs for new charges. It preserves the old summary, progress and artifacts. A
separate snapshot retains the ledger before new entries. Output accounting covers
both production directories. The completed readiness result is reused.

The original deadline remains **29 November 2026 at 01:35:51 UTC**. Recovery work
and downtime consume the same wall-clock allowance. The six-hour finalization
reserve and all CPU, GPU and output limits remain unchanged. The user-systemd
service has no automatic restart and a runtime limit within the remaining time.
The original service's monotonic start and invocation are pinned with the boot
identity. Recovery uses that earlier start, as well as UTC and saved elapsed time,
so a backward clock correction cannot erase downtime. A different boot or service
invocation requires another reviewed recovery rather than a new allowance.

Before resuming, run the affected behavior checks and a fresh independent
GPT-6-Astra/max review of the recovery and the actual frozen request. After
launch, check real progress rather than relying on the service state alone.
Final results, resource reconciliation, the owner page and completion review
remain unfinished. E2 readiness is separate.
