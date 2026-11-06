---
name: pstack-interrogate
description: Independently review a kenggo bead or contested design with a fresh Codex instance. Find substantive defects, verify findings, and repeat review after fixes until the final patch is clean.
---

# Independent review and repair

[AGENTS.md](../../../AGENTS.md) defines reviewer commands, the model, significance, completion, and direct-main delivery. Read it before starting. This skill does not substitute self-review for an independent instance.

## Implementation owner

1. Freeze a reviewable patch after relevant checks pass. Include new files and all commits since the bead's recorded starting SHA. Check that the diff is nonempty when repository files changed.
2. Start a fresh Codex reviewer. Supply the bead ID and acceptance criteria, plan sections, exact scope/base, and validation evidence. Keep the working tree stable while it reviews.
3. Request concrete findings with paths/lines, impact, and a reproduction or direct contract violation. The reviewer reads and reports only.
4. Assess every substantive finding. Reproduce it when feasible, fix valid defects, and rerun affected checks. Document evidence for dismissed findings.
5. Start a new reviewer on the resulting full patch. Continue until a completed pass has no significant findings. An error or incomplete review leaves the gate unmet.
6. Record scope, reviewer/model, passes, verdict, and checks in the bead. Then follow `AGENTS.md` to close and deliver it.

## Reviewer brief

You are the independent reviewer, not the implementation owner. Read the scoped files and relevant contracts. Check correctness, data integrity, experimental validity, and likely regressions. For workflow changes, exercise or inspect the actual commands and routing.

Use small reproductions or read-only commands when practical. A direct, evidenced contract violation can be a finding even when expensive training or unavailable hardware prevents a full run. Label that limit. Do not discard a credible correctness defect merely because it lacks a cheap reproduction.

Report each significant finding with severity, location, why it fails the requirement, and evidence. Merge duplicate root causes. Omit taste-only suggestions. No significant findings is a valid verdict. State any unreviewed area or failed verification separately.

Do not edit project files, mutate Beads, commit, push, or spawn another reviewer. The owner performs fixes and decides whether the completion gate is met.
