# Review E1 in a local browser

Generate a single offline page from the existing E1 artifacts, then open it:

```bash
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.cache/python"
uv run --locked python tools/build_e1_review.py
xdg-open runs/e1-manual-review/index.html
```

The exporter requires the pinned local corpus, example shards, original SGF
archives, saved referee decisions and native reference binary from the completed
E1 work. It reports missing or changed inputs instead of downloading replacements.
Run it from the repository root in the [CPU workspace](workspace.md). The page
works with a `file://` URL: no server, installation or network access is needed
after export. You can copy the generated HTML to another computer.

The generated page and its embedded source data stay in ignored `runs/`. The
exporter refuses to replace an existing page. Supply another filename when
rebuilding, for example:

```bash
uv run --locked python tools/build_e1_review.py runs/e1-manual-review-v2/index.html
```

## What to inspect

- **Endgames:** starts with cases 044 and 083, the two winner disagreements.
  Replay the original game and either saved cleanup budget independently. Orange
  squares show changes from the source ending. Filter all 108 endings by winner,
  score or between-budget margin differences. Source SGF downloads preserve the
  original bytes, comments and variations. The displayed replay follows the
  canonical played main line. Colors come from SGF metadata, not filenames.
- **Admission:** search all 1,000 previously exposed records by account, rank,
  date or exclusion. Inspect screening assumptions and metadata, and replay the
  241 admitted sources. Record concrete bot, alias or provenance findings.
- **Examples:** compare a deterministic sample of stored training/development
  rows with their source pre-move boards. Inspect mover/opponent planes, legal
  actions, explicit passes, source-value signs, rank fallbacks and missingness
  masks. Scalars are the stored values before normalization. Sampled games come
  from the existing diagnostics plus the first two source IDs in each allowed
  role; positions include the first two, middle, final and explicit-pass rows.
- **Sign-off checklist:** record the manual checks, capability limits and separate
  GPU follow-up. The listed validation/review results describe the recorded E1
  evidence; opening the page does not run tests or inspect the current host.

Use **Checked** or **Needs follow-up** with a note for each case you inspect.
Notes save independently per case under the evidence bundle's fingerprint, so
editing different cases in multiple tabs preserves both. Tabs receive each
other's updates; for edits to the same case, the latest save wins. Browser
storage for local files can vary; the page reports when it is unavailable.
**Export notes** saves a JSON backup. **Import notes** accepts the same evidence
bundle, merges entries and replaces matching entries. It rejects a different
bundle. Export before moving the HTML, clearing browser storage or closing a
session where storage is unavailable. Files opened in different browsers or at
different paths may have separate storage.

These annotations do not change source labels, admission, masks, profiles,
qualification or Beads. Transfer accepted findings to the relevant bead after
inspection. GPU commands must be run separately using [the CUDA setup](gpu_workspace.md).

## Data boundary and verification

The exporter uses the public nonfinal corpus reader with the independently pinned
population profile. It never requests a final release, decodes sealed canonical
records, or reads final example arrays. Only `train`, `population_development`
and `validation_development` shards supply the displayed examples. Original
exposed-source confirmation games remain visible in the admission/endgame views,
as they were in the E1 audit; their role labels remain visible.

Source frames use the shared Python rules adapter. Cleanup frames come from the
pinned native reference replaying the saved actions, with terminal states checked
against the saved evidence. No neural inference or new acquisition occurs.
Empirical cleanup scores remain separate from original human results. The
exporter uses one worker, a 300-second wall budget, a 10-second timeout per native
replay, and a 32 MiB HTML limit.

Run the exporter behavior checks with:

```bash
uv run --locked python -m unittest discover -s tests -p test_manual_review.py -v
```

For browser checks, use an existing Node/Playwright installation:

```bash
node tools/e1_review/check_browser.cjs runs/e1-manual-review/index.html
```

If necessary, set `NODE_PATH` to the directory containing Playwright and
`PLAYWRIGHT_CHROMIUM_EXECUTABLE` to the installed Chromium executable. This checks
the actual offline page, replay controls, filters, downloads, stored inputs,
notes persistence/import/export and narrow-screen layout. Screenshots stay next
to the generated HTML. The browser check uses a separate temporary browser
context and does not touch your personal notes.
