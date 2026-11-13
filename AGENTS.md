# Working in kenggo

## Project contract

Read [plan/go_project_plan_v0.5.md](plan/go_project_plan_v0.5.md) and the relevant epic before implementation. This is a research project about the board game Go. Follow the selected [technology stack](plan/go_project_plan_v0.5.md#technology-stack).

- A learns human moves with rank conditioning. B starts from random weights and learns against frozen A profiles. Do not transfer A's trained weights to B.
- Adjustable difficulty is a project goal. Use supported, evaluated opponent profiles; establish the available range from data and calibration.
- Keep move prediction, strength, and human-like behavior separate. Do not claim a human rank without the plan's calibration evidence.
- Preserve the plan's data splits, missing-data masks, full legality history, endgame contract, player/value perspectives, and termination versus truncation semantics.
- Record experiment decisions, budgets, provenance, and acceptance evidence. A negative result or a decision to defer an extension can complete an experiment.

Use plain language in replies, documentation, and commit messages. Make the smallest change that meets the bead's acceptance criteria.

## Start a task

1. Read this file, run `git status --short --branch`, and inspect any existing changes. Preserve work that is not yours.
2. Run `bd prime` after starting or resuming a session, including after compaction. Run `bd ready` and `bd list --status=in_progress` to recover shared work.
3. Use an existing bead or create one before changing files. Inspect it with `bd show <id>` and claim it with `bd update <id> --claim`.
4. Record the starting commit from `git rev-parse HEAD` in the bead. This is the review base if the work spans commits on `main`.
5. Read [.agents/skills/poteto-mode/SKILL.md](.agents/skills/poteto-mode/SKILL.md) and follow the matching workflow. This applies automatically to implementation, fixes, refactors, and workflow changes; the user need not invoke it each time.

## Beads is the task record

Use `bd` for tasks, dependencies, blockers, acceptance criteria, review results, and handoff notes. Do not create a second task tracker in Markdown. Research plans and experiment reports still belong in versioned documents.

```bash
bd ready
bd show <id>
bd update <id> --claim
bd create "Bounded task" --type task --priority 2 \
  --description "Purpose, plan epic, inputs, and expected output" \
  --acceptance "Observable completion check"
bd dep add <blocked-id> <prerequisite-id>
bd update <id> --append-notes "Evidence, decisions, or handoff"
bd close <id> --reason "Completed and independently reviewed"
```

Use `--json` for machine-readable results. Use command flags rather than the interactive `bd edit`. Close a bead only after its acceptance criteria, validation, and independent review pass. File discovered follow-up work with dependencies; do not defer a significant defect in the current change just to close it.

This repo uses Beads 1.3.2 with embedded Dolt. The configured remote uses the same GitHub repository through `git+ssh`. Beads data lives in a separate Git data ref, not in source commits or a task branch. Run `bd sync` to pull, merge, and push it. Ordinary `git push` alone does not synchronize the tracker. Do not add JSONL export/import hooks or commit the live database. Setup and fresh-clone instructions are in [.beads/README.md](.beads/README.md).

## Direct commits to main

This repo opts into Beads' **team-maintainer** workflow. For assigned work, completing the review loop, committing, syncing Beads, and pushing to `origin/main` are authorized parts of completion. A current user instruction such as "do not push" overrides this default.

Work on `main`. Do not create a feature branch, PR, or stacked PR for each bead. In pstack, "open a PR", "ship a stack", and similar delivery steps mean the completion procedure below. Do not force-push or discard somebody else's work.

Keep one writer for shared files and Git operations. A reviewer reads the same stable patch without editing. If delegating implementation, give each worker disjoint files and retain one owner for integration, Beads updates, commits, and pushes. Resolve existing branch/worktree state without silently switching away from someone else's work.

## Independent Codex review for every bead

Every completed bead, including docs and tooling changes, needs a **fresh, separate Codex instance**. Self-review by the implementation session does not satisfy this requirement. Use GPT-6-Astra with maximum reasoning.

Read [.agents/skills/pstack-interrogate/SKILL.md](.agents/skills/pstack-interrogate/SKILL.md). Run the reviewer from the repository root. For a bead contained in the current uncommitted patch:

```bash
codex -s read-only -a never review --uncommitted \
  -c model='"gpt-6-astra"' -c model_reasoning_effort='"max"'
```

For a bead spanning commits, review against its recorded starting commit, not the current `main` tip. `main` would otherwise compare against itself:

```bash
codex -s read-only -a never review --base <starting-sha> \
  -c model='"gpt-6-astra"' -c model_reasoning_effort='"max"'
```

Include all changed files, including new files, in the review scope. Stage intended new files before a base review so they appear in the diff. Give the reviewer the bead's acceptance criteria, the relevant plan sections, and validation evidence. Use a separate read-only `codex exec` or a fresh Codex subagent if the review command cannot carry the necessary context. Keep the same model, reasoning level, and scope. Do not pass credentials or unrelated private context.

Repeat **review → assess findings → fix → rerun affected checks → fresh review** until a completed review returns no significant findings. The reviewer reports findings and never edits, closes beads, commits, or starts another review. The implementation owner applies fixes and assesses the final result.

Significance, in priority order:

1. **Correctness and integrity.** Illegal play, bad replay/scoring, leakage, invalid splits, sign/mask/return errors, wrong termination handling, lost state, races, crashes on valid input, security defects, and broken workflow commands.
2. **Research and metric impact.** Invalid comparisons, misleading strength claims, behavior changes in a behavior-preserving refactor, or regressions in prediction, strength, complete-game behavior, compute, or reproducibility.
3. **Style and taste.** Optional unless they conceal a substantive defect. Do not spend review iterations on naming preferences alone.

Verify findings against the implementation. Record evidence when dismissing a false positive and ask the next reviewer to reassess it. An unavailable reviewer, failed command, partial review, or exhausted session is not a clean verdict. Keep the bead open with the blocker if review cannot complete. If repeated fixes fail the same check, revisit the underlying assumption instead of adding another workaround.

Record the review scope, reviewer/model, pass count, final verdict, and validation commands/results in the bead. Mention review-driven fixes in the commit message when they changed the patch.

## Manual review by the owner

When the owner needs to manually review something, provide a single-page HTML/JS interface when applicable so the evidence is easy to inspect. Show the relevant results and the checks or decisions needed, and link directly to the generated page. Prefer a self-contained page that opens locally without a server. Record accepted feedback in the owning bead; the page does not replace independent Codex review or acceptance checks.

## Validation and completion

Follow [the workspace setup](docs/workspace.md) for the pinned Linux CPU environment. From the repository root, the current checks are:

```bash
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.cache/python"
uv lock --check
uv sync --locked --check
uv run --locked python -m kenggo.smoke
uv run --locked python -m unittest discover -s tests -v
uv build --no-sources
git diff --check
```

The smoke command checks CPU tensor/autograd, NumPy and Parquet installation behavior; the build produces the Python package in ignored `dist/`. Behavior tests cover acquisition/auditing, eligibility, canonical records, whole-game splits, masked examples, shared board encoding and the rules adapter. For rules changes, also run the [pinned native reference check](docs/rules_engine.md#reproduce-validation); a skipped comparison is not validation. Search and RL implementations/tests remain future work. Verify GPU readiness separately as documented; CPU success does not establish CUDA support. Also check changed Markdown links, skill frontmatter/routing when applicable, ignore rules, Beads commands/sync and the independent review. Add actual behavior-test commands as implementations arrive.

Use small behavior tests for rules, data splitting, search, and RL contracts before expensive runs. Apply the pstack benchmark checklist before trusting performance or evaluation claims. Keep raw datasets, checkpoints, and run output out of Git; keep small fixtures and reproducibility manifests in Git.

For each completed bead:

1. Meet acceptance criteria, run the relevant checks, and finish the independent review loop.
2. Record evidence and the clean verdict in the bead. Close it when complete, or leave it open with a clear blocker.
3. Inspect `git diff` and `git diff --cached`. Stage only intended files and commit on `main`, including the bead ID in the message.
4. Run `git pull --rebase origin main`. If integration changes the reviewed patch, rerun affected checks and independent review before pushing.
5. Run `bd sync`, then `git push origin main`. Resolve actual conflicts; never force a push. If either command fails, report what remains local and retain enough bead notes to resume.
6. Confirm `git status --short --branch` is clean and `HEAD` matches `origin/main`. Report the commit, bead, checks, and review outcome.

Work is delivered when both source and tracker sync succeed. Do not stop at an offer to push when pushing is already authorized.
