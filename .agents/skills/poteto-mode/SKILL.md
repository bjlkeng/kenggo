---
name: poteto-mode
description: Use the pstack workflow for implementation, bug fixes, refactoring, experiments, and agent workflow changes in kenggo. Grounds decisions in evidence, routes to focused skills, and finishes with independent Codex review and direct-main delivery.
---

# Poteto mode for kenggo

Read the repository's [AGENTS.md](../../../AGENTS.md). It owns task tracking, authority, review gates, and delivery. This [Codex port](../../../docs/pstack.md) adapts pstack; no Cursor plugin is required.

## Route the work

State the user's intended outcome and the smallest change likely to meet it. Claim a bead and record its acceptance criteria and review base. Choose the relevant workflow from [references/playbooks.md](references/playbooks.md); keep progress and evidence in the bead.

Read the supporting skill when its trigger applies:

| Trigger | Skill |
| --- | --- |
| Understand a subsystem before a nontrivial change | [pstack-how](../pstack-how/SKILL.md) |
| Choose data shapes, interfaces, ownership, or module boundaries | [pstack-architect](../pstack-architect/SKILL.md) |
| Fix a bug with a cheap local regression check | [pstack-tdd](../pstack-tdd/SKILL.md) |
| Measure performance or report experimental results | [pstack-benchmark-checklist](../pstack-benchmark-checklist/SKILL.md) |
| Delegate independent exploration, implementation, or verification | [pstack-swarm](../pstack-swarm/SKILL.md) |
| Write docs, instructions, or commit messages | [pstack-technical-writing](../pstack-technical-writing/SKILL.md) |
| Clean up prose before delivery | [pstack-unslop](../pstack-unslop/SKILL.md) |
| Inspect code comments before review | [pstack-no-comments](../pstack-no-comments/SKILL.md) |
| Complete any bead or challenge a contested design | [pstack-interrogate](../pstack-interrogate/SKILL.md) |

## Engineering principles

Name the data shape before writing stateful logic. Prefer structures that encode valid states over repeated conditionals. Keep boundaries small, validate external input, and remove needless layers. Fix the earliest wrong decision, not a downstream symptom.

Break work into units that each end with an observable check. Test behavior through the caller's interface. Verify the actual artifact or runtime output before claiming success. Record failed approaches when they change the next decision.

For multi-step work, record the blocking first checks, independent workstreams, shared mutable state, and smallest useful delegation in the bead. One implementation owner is appropriate when the work is tightly coupled. Use separate, fresh reviewers regardless of implementation size.

Questions about observable behavior call for a probe or small prototype. Ask the user about unresolved product choices, missing access, or authority. Do not invent permission from this skill.

## Finish

Inspect the complete diff, remove accidental complexity, and run relevant checks. Complete the independent review and fix loop in `AGENTS.md`, then its direct-main delivery procedure. Recheck any patch changed after review. Report the outcome, evidence, and material limitations in plain language.

When invoked as a read-only reviewer, inspect and report only. Do not claim a bead, change files, run delivery, or spawn another reviewer.
