# pstack for Codex

This repository carries a focused Codex port of [pstack](https://github.com/cursor/plugins/tree/2776ddd2355d6d6c8747d3964600f28daf62f049/pstack) by Lauren Tan, under its [MIT license](../third_party/pstack/LICENSE). The source revision is `2776ddd2355d6d6c8747d3964600f28daf62f049`. These are adapted skills, not an installation of the Cursor plugin.

## Use it

Start Codex in this repository. [Codex discovers repository skills](https://learn.chatgpt.com/docs/build-skills#where-codex-loads-local-skills) under `.agents/skills`. [AGENTS.md](../AGENTS.md) directs it to read `poteto-mode` automatically for changes. You can also invoke `$poteto-mode` explicitly. If skills do not appear in an already-running session, restart Codex; the files remain readable directly in the current session.

The entry point selects an implementation, bug-fix, refactoring, investigation, experiment, or skill-authoring workflow. It reads the relevant supporting skill, keeps progress in Beads, verifies the result, and finishes through the independent review loop in `AGENTS.md`.

## Adaptations

| Upstream mechanism | Codex port |
| --- | --- |
| `/poteto-mode` and Cursor skill commands | `$poteto-mode` and repository `SKILL.md` paths |
| Cursor `Task`, `poteto-agent`, model-specific panels | Codex subagents when available; a separate read-only Codex process for required review |
| Cursor/Grok/Claude model defaults and setup rules | Inherit the current Codex model for optional delegates; GPT-6-Astra at `max` for bead review |
| Cursor todo tools | Beads issues, dependencies, and notes |
| PRs, stacked branches, and shipping automation | Reviewed commits directly to `main`, then Beads sync and Git push |
| `create-skill`, `deslop`, `control-cli`, `control-ui` plugin dependencies | Codex skill-creator when available; local prose cleanup; actual shell/browser verification |
| Multi-model adversarial review | Fresh Codex review, concrete findings, fixes, and repeated review until clean |

The port keeps pstack's grounding, design-before-code, small verifiable changes, practical regression tests, measurement discipline, and separate review. It limits delegation to useful, bounded work and gives one owner control of the shared checkout. It does not grant authority to send messages, deploy services, or change unrelated projects.

## Included source mapping

Paths in the source column are relative to the pinned upstream `pstack/skills/` directory.

| Local skill | Upstream source |
| --- | --- |
| [poteto-mode](../.agents/skills/poteto-mode/SKILL.md) | `poteto-mode/SKILL.md`, implementation and skill-authoring playbooks, `principle-*` skills |
| [pstack-how](../.agents/skills/pstack-how/SKILL.md) | `how/SKILL.md` |
| [pstack-architect](../.agents/skills/pstack-architect/SKILL.md) | `architect/SKILL.md` |
| [pstack-interrogate](../.agents/skills/pstack-interrogate/SKILL.md) | `interrogate/SKILL.md`, plus the [SAT-playground review loop](https://github.com/bjlkeng/SAT-playground/blob/main/CLAUDE.md) |
| [pstack-tdd](../.agents/skills/pstack-tdd/SKILL.md) | `tdd/SKILL.md` |
| [pstack-no-comments](../.agents/skills/pstack-no-comments/SKILL.md) | `no-comments/SKILL.md` |
| [pstack-technical-writing](../.agents/skills/pstack-technical-writing/SKILL.md) | `technical-writing/SKILL.md` |
| [pstack-unslop](../.agents/skills/pstack-unslop/SKILL.md) | `unslop/SKILL.md` |
| [pstack-benchmark-checklist](../.agents/skills/pstack-benchmark-checklist/SKILL.md) | `benchmark-checklist/SKILL.md`, `principle-explain-the-number/SKILL.md` |
| [pstack-swarm](../.agents/skills/pstack-swarm/SKILL.md) | `swarm/SKILL.md` |

Unported workflows include Cursor automations, cloud fleets, PR babysitting, worktree cleanup, video generation, and the statechart toolchain. They are not dependencies of this port. Add another upstream workflow when the project needs it, with a corresponding Codex adaptation and validation.

## Update the port

Read the pinned source and the candidate upstream revision. Compare the specific mapped skills, then adapt the changes while retaining this repo's direct-main, Beads, and review rules. Update the revision and mapping above and preserve the license. Do not overwrite the port with raw Cursor files or enable auto-updates.

Validate frontmatter (`name` and `description`), relative links, and routing after an update. Exercise a realistic request with a separate Codex instance before adopting structural workflow changes. A reviewer must report findings without editing or recursively reviewing itself.
