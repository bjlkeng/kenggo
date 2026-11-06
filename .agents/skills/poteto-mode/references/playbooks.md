# Workflows

Use the relevant sequence. Keep task status in Beads. `AGENTS.md` owns the common review and delivery steps, so every workflow that changes the repository ends there.

## Feature or refactoring

1. Read `pstack-how` and inspect the affected callers and contracts.
2. Read `pstack-architect` when data shapes, interfaces, or ownership change. For a refactor, name the behavior that must remain unchanged.
3. Record blocking checks, independent workstreams, shared writes, and the smallest useful delegation. Keep shared Git and Beads operations with one owner.
4. Implement bounded changes. Delegate disjoint work through `pstack-swarm` when it improves coverage or separates implementation from design; otherwise keep one writer.
5. Exercise the actual behavior and relevant failure cases. Complete independent review and direct-main delivery.

## Bug fix

1. Reproduce the reported failure and trace the first wrong decision. Use instrumentation when the cause is unclear.
2. Read `pstack-tdd` when a cheap regression test is available. Confirm the check fails for the intended reason before fixing.
3. Fix the cause, rerun the reproduction and nearby checks, then complete independent review and delivery.

## Experiment or performance change

1. Read the relevant project-plan epic and `pstack-benchmark-checklist`. Record the hypothesis, metric, selection rule, comparison, and budget before running.
2. Validate the measurement on a small case and establish a baseline.
3. Change one relevant factor, measure under comparable conditions, and record all outcomes and costs.
4. Keep, defer, or reject the change based on evidence. Review the implementation and claims independently before delivery.

## Investigation or prototype

1. Read `pstack-how`, state the question, and identify evidence that would answer it.
2. Trace source and run read-only probes. If an empirical design choice remains, make the smallest isolated prototype that can settle it.
3. Report observed facts separately from inferences and remaining unknowns. Do not turn a prototype into production code without the feature workflow. A completed investigation bead still receives independent review of its evidence and conclusions.

## Author or modify a skill

1. Use Codex's skill-creator when available. Define the actual trigger and the decisions the skill changes.
2. Write valid frontmatter, resolve every local reference, and remove unavailable tool or model dependencies. Keep source attribution and licenses for adapted material.
3. Validate structure and exercise realistic scenarios for structural changes with a separate Codex instance. Small wording edits need proportionate checks.
4. Complete independent review and direct-main delivery. Reviewers must not recurse into the completion workflow.

## Resume or hand off

1. Run `bd prime`, inspect the bead, Git status, prior review scope, and live work that belongs to the task.
2. Verify what is actually complete before continuing. Reuse evidence only while its inputs remain unchanged.
3. If interrupted, record the remaining work, exact commands, findings, and blockers in the bead. Keep it open until validation and review finish.
