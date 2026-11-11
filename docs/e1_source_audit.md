# Reproduce the bounded KGS source audit

The E1 audit separates archive discovery, a frozen candidate sample, retrieval and offline inspection. It does not train a model or establish legal replay. The [E0 population](../plan/e0_scope_decision.md) and [E1 contract](../plan/go_project_plan_v0.5.md#e1-establish-the-data-and-game-contract) govern its exclusions and support decisions.

The [10 November report](../reports/e1_kgs_audit_2026-11-10.md) records the actual sample, costs and deferred capabilities. Its [manifest](../manifests/e1_kgs_audit_2026-11-10.json) and [per-game Parquet](../manifests/e1_kgs_audit_games_2026-11-10.parquet) are the versioned audit evidence.

Use the [locked CPU workspace](workspace.md). The audit uses [SGFmill 1.1.1](https://mjw.woodcraft.me.uk/sgfmill/doc/1.1.1/) for SGF syntax and property access. The separately validated [shared rules adapter](rules_engine.md) also uses its placement/capture primitives; parsing alone does not perform that replay.

The [version-1 eligibility policy](../reports/e1_kgs_eligibility_2026-11-10.md) adds a reproducible admission decision and conservative handle groups. It admits zero training games, keeps personas masked and reserves exposed groups from final evaluation. It does not change the frozen audit counts.

## Offline checks

```bash
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.cache/python"
uv run --locked python -m unittest discover -s tests -v
```

The source-audit fixtures cover source selection, HTML review rows and inaccessible records, request limits, error handling, ZIP mapping, duplicate bytes, results, passes and missing clocks. The suite also contains eligibility, canonical-record, whole-game split, masked-example, encoding and shared-rules tests; native-reference validation is documented separately in the [rules guide](rules_engine.md#reproduce-validation). The [canonical builder](../reports/e1_canonical_2026-11-11.md) and [example encoder](../reports/e1_examples_2026-11-11.md) preserve provenance and admission gates, with zero currently usable games or examples. Search and RL remain future work.

With the retained run files, reproduce the audit in a fresh output directory:

```bash
uv run --locked python -m kenggo.e1_audit \
  runs/e1-kgs-audit-2026-11-10-v4 \
  manifests/e1_kgs_discovery_2026-11-10_v2.json \
  runs/e1-kgs-audit-reproduced
```

This command makes no network requests. It verifies the configuration and frozen sample, reparses cached archive pages, matches selected source IDs to exact ZIP date/filename pairs, and writes `games.json`, `games.parquet` and `summary.json`. The JSON contains per-record audit evidence; Parquet preserves the source IDs, SGF hashes, retrieval/parse status, exclusions and the same audit evidence. Original SGFs remain in the retained ZIPs. Extra bundle members are not parsed.

The offline limits are 120 CPU seconds, 300 wall seconds, 64 MiB of selected SGFs, 1 MiB per selected SGF and 256 MiB of declared expanded ZIP content. A failed retrieval stays in the sample. No replacement games are selected.

## Acquisition controls

The [original discovery configuration](../manifests/e1_kgs_discovery_2026-11-10.json) pins the published account survey and selection seed. [Revision 2](../manifests/e1_kgs_discovery_2026-11-10_v2.json) excludes two historically documented bot handles without replacement and records the stopped first attempt. Survey ranks select discovery accounts; archive and SGF ranks describe the historical games.

The acquisition entry point is:

```text
uv run --locked python -m kenggo.kgs_acquire CONFIG FRESH_OUTPUT_DIRECTORY
```

It prompts for the account and password without echo. Credentials are not command arguments, environment variables or files. Cookies remain in memory. Request logs omit POST bodies and response headers. Cached archive HTML replaces the logged-in account footer with a redaction; login responses are not saved.

The client allows one request at a time and waits at least five seconds between starts, including redirects. Its ceilings are 500 requests, 100 MiB of response bodies, 2 MiB per HTML response, 8 MiB per ZIP, 60 minutes and 15 CPU minutes. Redirects stay on the two official KGS HTTPS hosts. HTTP errors, timeouts and limit violations stop the process; there are no automatic retries. A timeout can leave the server working on its request, so restarting immediately is not a safe substitute for the stop.

The historical configuration has a hard UTC deadline and is not a standing download job. Record a new finite execution budget before any future acquisition. After diagnosing a stopped local run, `--prior-attempt DIRECTORY` carries its cumulative costs forward and reuses its cached pages. It does not reset the deadline. Any deliberate continuation and additional login belong in Beads and the run provenance.

Discovery completes before `sample.json` is written. The command then waits for `retrieve` before fetching monthly ZIPs for those selected IDs. Use that boundary to verify the sample hash and counts. No game outcome or SGF exclusion causes resampling.

## Interpret the evidence

`metadata_scope_compatible` means the inspected metadata passes the specified filters. It does not certify a human opponent, a stable account generation, legal moves, correct scoring, a supported model profile or permission for a different use of the corpus. Human status without independent evidence stays `unverified`; previously documented bot handles remain a separate category. Account coverage counts handle strings, not verified people or persistent account IDs.

The parser selects the leftmost SGF variation and flags branching records as unverified played lines. It counts explicit pass actions, including legacy pass encoding, independently of the recorded result. EOF supplies no move or termination. Result categories distinguish score, resignation, timeout, forfeit and incomplete/unknown records. The archive/SGF comparison includes the winner and numeric margin.

SGF `BL` and `WL` describe post-move clocks. The audit counts observed, missing, invalid and overtime-marked values. A contiguous nonincreasing clock pair above the declared Japanese overtime period is a **main-time difference candidate**. It is not yet a validated thinking-time target. Missing or invalid clocks break the chain; overtime and transitions remain ambiguous. No missing duration becomes zero, and no future clock or result enters a model input. ([SGF properties](https://www.red-bean.com/sgf/properties.html))

Exact SGF byte duplicates are excluded after the first sampled occurrence. Matching move-sequence fingerprints are reported as possible duplicates for E1's later deduplication work; they are not proof that two source records are the same game. The audit does not produce training splits. The [eligibility policy](../reports/e1_kgs_eligibility_2026-11-10.md#final-evaluation-exposure) reserves both participant groups in inspected games, all original discovery handles and the `cu2` probe across all dates, while retaining the source-ID, byte/sequence and account-month predicates.
