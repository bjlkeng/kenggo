# Seven-day study: recovery source

These are exact source snapshots from the E2 study launched on 21 November
2026. They preserve the recovery and verification work for `kenggo-ydo.17`
and `.18`. Production source stayed at commit `59d29ab` until every worker
finished; the helpers therefore originally lived in ignored `runs/` folders.

[sources.json](sources.json) maps each copy to its original path and SHA-256.
The copies include earlier failed attempts because later recovery depends on
their recorded behavior. They have not been refactored or relabeled as a new
supported runner. In particular, the original composite verifier retains its
relative-path comparison bug; the completion correction is recorded separately.

The scripts expect the original workspace paths and local pinned artifacts.
Read or compare these copies here. To reproduce a historical helper check,
use its original `runtime_path` and the exact command in the completion
manifest. Restoring a missing helper requires placing these exact bytes at
that path; preserve an existing file unless its hash matches. Do not launch
the old production services: their allowance and process identities belong to
the completed study.

No credentials, game data, checkpoint weights, live ledger or generated run
output are stored here. The versioned report and manifest explain the final
result and link the local evidence.

Historical code pins refer to production commit `59d29ab`; source delivered
after the workers stopped can differ. The completion manifest records the
later history-catalog update and preserves its previous bytes. Check those
pins against the recorded Git revision rather than silently accepting today's
files as the production source.
