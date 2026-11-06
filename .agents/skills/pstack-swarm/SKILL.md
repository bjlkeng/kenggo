---
name: pstack-swarm
description: Coordinate bounded, independent Codex workers for exploration, design comparison, implementation, or verification in kenggo while preserving one owner of shared state.
---

# Bounded delegation

Use native Codex subagents when available. Inherit the current model for ordinary workers unless the task specifies one. The mandatory review model is defined in [AGENTS.md](../../../AGENTS.md). If delegation is unavailable, perform implementation sequentially and use a separate Codex process for required review.

Before spawning, define the deliverable, done condition, and each worker's distinct scope. For competing designs, state the comparison criteria. Give each worker the user's goal, bead context, relevant paths, constraints, verification method, and required report.

Exploration and review workers are read-only. Implementation workers own disjoint files. Keep shared files, Git operations, Beads mutations, commits, and pushes with one integration owner. Do not create task branches or worktrees by default. Serialize work when ownership overlaps.

Start fresh workers for independent tasks. Collect every required result and inspect the actual artifacts. Reports distinguish pass, findings, and blocked checks with evidence. Missing results or tool failures are gaps, not passes. The owner verifies the integrated result and completes the independent review loop before delivery.
