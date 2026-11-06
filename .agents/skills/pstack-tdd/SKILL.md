---
name: pstack-tdd
description: Create a failing-before and passing-after regression check for a kenggo bug when a cheap local test path exists, or when the user explicitly requests TDD.
---

# Practical regression testing

Identify the intended behavior and the smallest observable failure. Prefer an existing test target that exercises the caller's interface. Derive expected results from the contract, not the implementation.

Write and run the regression check before the production fix. Confirm it fails for the reported reason. Make the smallest correct fix, rerun the check, and run relevant neighboring checks.

Do not add a broad harness, brittle mocks, or unrelated fixtures just to satisfy TDD. Use a focused reproduction command or real runtime check when a new test is impractical, and record the limitation. Never weaken assertions to match a bug. Keep tests that catch broken behavior and survive internal refactoring.
