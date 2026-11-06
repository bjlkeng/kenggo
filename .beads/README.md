# Beads in kenggo

Use [Beads](https://github.com/gastownhall/beads) for the shared task record. This setup was verified with `bd` 1.3.2, embedded Dolt, and issue prefix `kenggo`. No separate Dolt server is needed.

The database is local and ignored. Its Dolt remote is `git+ssh://git@github.com/bjlkeng/kenggo.git`; `bd sync` exchanges database history through a Git data ref on that repository. Source commits and issue history sync separately. Do not use `.beads/issues.jsonl` as the database or add legacy export/import hooks.

## Set up a checkout

Install the verified Beads version using the [upstream installation instructions](https://github.com/gastownhall/beads#installation), then run from the repository root:

```bash
bd init --prefix kenggo --skip-agents --skip-hooks --non-interactive
bd hooks install --beads
bd dolt pull
bd prime
bd ready
```

Run `bd init` only when the checkout does not already have a local database. Preserve the tracked config and metadata. `--skip-agents` keeps this repository's custom `AGENTS.md` and skill setup intact. Do not use `--force`, `--reinit-local`, or `--discard-remote` to get past an initialization error.

The hooks are versioned under `.beads/hooks`; `bd hooks install --beads` sets the checkout-local `core.hooksPath`. They are the native hooks for this Beads version. They do not replace explicit tracker sync.

## Work and synchronize

```bash
bd ready
bd show <id>
bd update <id> --claim
bd update <id> --append-notes "Evidence and review verdict"
bd close <id> --reason "Completed and independently reviewed"
bd sync
```

Use the direct-to-`main` completion and review loop in [AGENTS.md](../AGENTS.md). `bd sync` detects conflicts and retries push races a bounded number of times. A transport error or unresolved conflict is a failed sync; do not claim delivery or discard database history.

`bd where`, `bd dolt status`, and `bd dolt remote list` inspect the local setup. In embedded mode, use supported checks such as `bd doctor --check=conventions`; the full `bd doctor` requires server mode in version 1.3.2.
