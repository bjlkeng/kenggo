---
name: pstack-no-comments
description: Inspect changed code comments before a kenggo review. Remove redundant narration and stale claims while preserving non-obvious reasons, contracts, provenance, and required notices.
---

# Comments before review

Inspect comments only within the task's changed code. Remove comments that repeat the code, narrate phases, or describe behavior that is no longer true. Prefer a clearer name or structure when it makes the explanation unnecessary.

Keep non-obvious reasons, mathematical assumptions, external contracts, source attribution, safety constraints, and license notices. Do not delete a constraint whose purpose is uncertain. Investigate it with [pstack-how](../pstack-how/SKILL.md).

If a comment exposes a real bug or an unenforced invariant, fix it within the bead's scope and test the behavior. Do not expand the task just to remove comments. The independent reviewer can check this pass as part of the full review; no separate comment-only agent is required.
