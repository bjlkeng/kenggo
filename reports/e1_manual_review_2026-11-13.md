# E1 owner review and game exclusions

13 November 2026 | `kenggo-bck`

The owner requested exclusion of the two recorded/referee winner-disagreement
games after the E1 manual review. Admission spot checks of a subset and inspection
of a few encoded examples were acceptable, and the stated capability limits were
accepted. The owner did not identify the inspected subset, so this is not a claim
that every admission or example was manually checked. CUDA readiness remains a
separate open task.

The [pinned owner decision](../manifests/e1_manual_review_v1.json) identifies the
two complete games by source URL, SGF hash and canonical hash:

| Historical case | Source record | Action |
| --- | --- | --- |
| 044 | `2025/3/2/lorbas72-chika.sgf` | Exclude all 256 examples |
| 083 | `2025/7/29/BasilDiug-yuumsuzuki.sgf` | Exclude all 223 examples |

Both were training games. Their original SGFs and source labels remain unchanged;
the decision does not establish whether the human result or referee is correct.
The full historical 108-ending comparison, including its 100/102 recorded-winner
agreement, is retained. Agreement on the post-selected remainder is not a new
referee evaluation.

## Current usable inputs

The [version-3 population manifest](../manifests/e1_population_v3.json) is the
current input reference. Of 241 originally admitted nonfinal source games, 239
remain usable after the two exclusions, including the 59 earlier validation
prefixes that still emit no adaptation examples. Training contains **106 games,
22,916 policy examples and 20,888 valid value targets**, including 97 real pass
actions. Across all roles there are 54,527 policy examples: 39,304 nonfinal and
15,223 sealed final examples. The remaining nonfinal data has 36,139 valid
source-value targets.

Only the two training memberships move to `excluded`. All other split files,
cohort boundaries, grouping/exposure registries and source records are unchanged.
Canonical records retain their historical admission evidence; the current split
manifest supplies the owner's additional exclusions. The two excluded memberships
have zero policy/value actions and cannot emit examples through the shared
example generator.

The derivation filters existing training shards instead of re-encoding the
corpus. All **22,916 retained training rows** were compared field by field against
the old arrays and metadata. Every nontraining shard, including the sealed final
arrays, was copied with matching hashes without decoding final targets. The
train-only normalizer was refitted to the filtered rows and reproduced exactly.
The split manifest, profile, example manifest and normalizer also reproduced byte
for byte in a second derivation. There was no source acquisition or neural
inference.

The first derivation used 2.71 process CPU seconds and 2.29 wall seconds; the repeat
used 2.75 and 2.31 respectively. Both used one worker and no HTTP/GPU work, within
the 120 CPU/180 wall second and 256 MiB output budgets. These are local run costs,
not a performance comparison. Summaries and verification pins are in the
population manifest; source/tests and independent review are recorded in Beads.

Use the [current data guide](../docs/e1_population.md) for reproduction and reader
commands. Version-2 artifacts remain available for historical reproduction and
must not be mistaken for the current training inputs. Later corpus expansions
must retain these source-ID/hash exclusions and the existing final reservation.

## Review UI

The updated [local UI exporter](../docs/e1_manual_review.md) uses the version-3
splits and examples. Its usable-source filter excludes both games; historical
ending views keep them visible with an owner-exclusion label. The checklist
shows the recorded subset checks and accepted limits. Earlier HTML exports and
their personal browser notes are preserved. The new export has its own evidence
fingerprint and therefore does not import notes from a different data bundle.

All earlier scope limits remain: no raw rank profile, personalization, timing
model, human-rank claim or GPU readiness is established by this review.
