---
name: pstack-how
description: Ground a kenggo change by tracing how the affected subsystem works, where state belongs, and which callers and invariants constrain it. Use before nontrivial changes or for code walkthroughs.
---

# How

Read the relevant plan section, source, tests, configuration, and callers. Trace one concrete input through state changes and observable output. Name ownership, boundaries, failure paths, and the constraints the proposed change must preserve.

For a narrow question, inspect it directly. For a broad subsystem, use [pstack-swarm](../pstack-swarm/SKILL.md) to split read-only exploration by distinct questions. Give explorers file pointers and request source-backed findings. Verify their conclusions before using them.

Give the implementer a short account of the flow, where changes belong, and the checks that would expose a mistake. Cite actual paths or commands. Distinguish source evidence from inference. Do not invent an architecture for code that has not been written.
