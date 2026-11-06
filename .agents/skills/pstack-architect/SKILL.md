---
name: pstack-architect
description: Design caller usage, data structures, and ownership before a nontrivial implementation in kenggo. Use when changing interfaces, shared state, or module boundaries.
---

# Architect

Ground the affected system with [pstack-how](../pstack-how/SKILL.md). For greenfield work, start from the plan and acceptance criteria.

Write the caller's intended usage first. Derive the types, signatures, state transitions, and module responsibilities from that usage. Identify invalid states and who owns each mutation before writing logic.

For a consequential design choice, compare at least two structurally different candidates. Use independent Codex exploration through [pstack-swarm](../pstack-swarm/SKILL.md) when it adds useful separation. Prefer a small public interface that hides the relevant complexity. Keep exploratory sketches out of the shipped application unless they are useful design records.

Record the selected shape and decisive tradeoff in the bead. Proceed within the user's scope without an extra approval checkpoint unless one was requested. Implement against the sketch and check deviations against the actual requirement.

If unrelated cases repeatedly need the same workaround, revisit the model. Remove the wrong structure before adding another guard. A contested design can receive [pstack-interrogate](../pstack-interrogate/SKILL.md) review before implementation; it still needs final review after implementation.
