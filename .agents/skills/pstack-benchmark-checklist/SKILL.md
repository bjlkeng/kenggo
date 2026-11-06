---
name: pstack-benchmark-checklist
description: Validate performance measurements and research comparisons in kenggo before using them to select an implementation or report a gain. Applies to throughput, latency, model evaluation, and training-cost claims.
---

# Explain the measurement

Read the relevant project-plan comparison before running. Record the claim, primary metric, minimum useful gain, allowed regressions, selection rule, and budget. Inspect what the measurement code counts, times, and excludes.

Check these points with evidence:

1. The intended work actually ran and produced correct results. Count errors, timeouts, caps, and skipped cases explicitly.
2. Each side uses comparable data, versions, settings, hardware, load, and compute. Separate configuration differences from implementation differences.
3. Identify the measured resource bottleneck for performance claims. Check that the claimed gain is plausible given the fraction of work changed.
4. Repeat and alternate cheap performance runs to expose variation and drift; use at least five runs per side when affordable. Follow the plan's seeds, game/account-level uncertainty, held-out cohorts, and compute budgets for training experiments instead of multiplying expensive runs blindly.
5. Confirm that a microbenchmark gain matters to the complete operation. Report actual network calls, latency, and GPU-hours where the plan requires them.
6. Preserve validation versus final-test separation. Do not tune on final-test results or claim human rank from a training win rate.

Report the verdict with units, run counts, spread or intervals, method, and limitations. A gap within measurement noise is inconclusive. A requested one-run ballpark stays labeled as one run and cannot justify promoting a model or choosing a winner.
