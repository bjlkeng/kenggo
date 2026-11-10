# Shared Go rules adapter

`kenggo.go.Game` implements ordinary play under
[`kgs_japanese_certified_v1`](../plan/e1_game_and_data_contract.md). It uses pinned
SGFmill 1.1.1 for placement/capture mechanics. The adapter owns turn order, suicide,
simple ko, full history, actual passes, adjudication and administrative boundaries.
The [validation manifest](../manifests/e1_rules_validation_v1.json) records the
independent KataGo comparison and the limits of this implementation.

All 241 metadata-compatible cached audit games replayed legally, with identical
boards, turns, captures and ko across 52,900 actions in the pinned native reference.
The automatic certifier resolved **0 of their 108 two-pass endings**. This validates
ordinary replay; it does not establish usable automatic human-game adjudication,
KGS equivalence in exceptional positions, or baseline readiness. Bead `6yr.10`
owns the coverage blocker. The [eligibility policy](../reports/e1_kgs_eligibility_2026-11-10.md)
supplies conservative handle groups but admits no training games; `6yr.11` owns
the missing human/ordinary-game and training-use evidence.

## Caller interface

```python
from kenggo.go import Game, PASS, require_valid_batch
from kenggo.adjudication import adjudicate
from kenggo.rules_replay import dump_game, load_game, replay_sgf

game = Game.new(candidate_color="W")
game = game.play("B", 3 * 19 + 3)  # Opponent opening belongs to the same trajectory.
mask = game.legal_mask()          # 362 booleans: top-left row-major points, then pass.
branch = game.clone().play("W", 15 * 19 + 15)
restored = load_game(dump_game(branch))
boundary = game.boundary("W")     # Saved learner-to-move observation for bootstrap.
require_valid_batch([boundary])   # Must run before an optimizer update.
```

Create states through `Game.new`, `Game.from_fixture`, `replay_sgf` or `load_game`;
dataclass fields describe returned state, not an alternative input-validation API.
Every returned position, move and history collection is immutable. `play` returns
a new game; rejected actions leave the original unchanged. `recent_moves(0/8/32/64)`
is only a model view. Clones retain every original position and action, including
captures and the recurrence segment. Kernels are private scratch boards; no mutable
SGFmill object is stored in a state.

`Game.new()` without a candidate color is the uncapped replay/analysis mode. Supply
the fixed candidate/learner color for generated games. A legal placement that
repeats board and next player within the segment since the last actual pass requests
an administrative cap; it is not superko or a rule-level result. Equality compares
the actual tuples, not just hashes. The 800-action limit and recurrence request stop
at the next candidate-to-move boundary, allowing at most one reply and 801 actions.
Both modes use identical move legality and two-pass freezing. Search must preserve
the parent's mode, candidate color and history when it branches.

The match runner must enforce the contract's response/game wall limits and hard
experiment budget. `truncate(reason, candidate_color=...)` accepts only a saved
candidate-to-move position. If a deadline/fault prevents constructing that position,
use `fail(reason, responsible=...)`. A known responsible player loses in match
accounting; an infrastructure/unknown failure remains U. Both invalidate learning.
Scheduling, response timers, collection/update orchestration, score aggregation and
missing stress-root accounting belong to E2/E6, not this engine library.

Two actual passes produce `pending_adjudication` and an all-false legal mask.
`adjudicate(game)` returns a new resolved or unresolved game plus proof/cost evidence.
No cleanup move is appended. A score or explicit `resign(current_player)` has a
winner; resignation has no territory margin and adds no action. Automatic
resignation is absent. Calling `boundary(learner)` returns:

| Ending | Learning fields |
| --- | --- |
| Certified score or explicit resignation | Valid, learner reward ±1, terminated, no bootstrap |
| Playing/chunk boundary or administrative cap | Valid only at learner-to-move; reward 0, nonterminal, saved state for bootstrap; cap is truncated |
| Pending/unresolved adjudication or a fault | Invalid; reward, termination/truncation and bootstrap fields are absent |

`require_valid_batch` rejects a batch containing any invalid boundary. The later
collector must retain those trajectories and reject the whole batch before update;
it must not discard individual failures or resample scheduled games. Match results
are W/L/U; this contract has no scored draws. A configured candidate color cannot
be overridden by the `boundary` argument, including after resolution. Use
`outcome.value(color)` for arbitrary-color outcome queries.

## Scoring and pass proofs

The automatic `singleton_eyes_v1` certificate checks the entire 19×19 board:

1. Every empty component is one intersection.
2. All orthogonal neighbors of each empty point belong to one identical connected
   stone chain.
3. Every chain owns at least two distinct such points; no open ko remains.

An opponent cannot capture a chain while its other exclusive eye remains, and
playing inside an eye would be suicide. Every chain therefore lives and every empty
point is protected territory. This small subset excludes dead stones, seki, dame,
large potentially invadable regions and the empty board. A living boundary or
one-color flood fill by itself is never sufficient.

The separate `singleton_eye_dominance_v1` pass proof uses an always-pass opponent
strategy. That opponent preserves both eyes of every original chain; mover
placements can only consume the mover's own eyes and cannot capture or invade the
opponent. The mover cannot force a better territory/prisoner score than mutual
pass. This establishes an alternative-gain upper bound of zero for this geometry.

`settled_atari_bounds_v1` can prove a narrow premature-pass case. A legal saving move
must reach the proved settled subset, giving the mover a lower score bound by
passing thereafter. A legal opponent response to an initial pass must also reach
that subset, giving an upper bound on passing. A gap of at least two half-points
certifies premature passing. The witness records both actual branches and bounds.
The proof is unavailable after a prior actual pass, at a cap, or when either branch
is unsupported. All other cases are `unjudgeable`; neither an atari heuristic nor
an accepted neutral final map supplies a pass-quality proof. Coverage is not a
qualification result.

For independent neutral adjudication, pass `neutral=payload` and an independently
configured `authorized_hashes` catalog to `adjudicate`. The default catalog is
empty. The payload binds its complete whole-chain and empty-region map, referee,
provenance and budget to the full replay snapshot and rules ID. Its SHA-256 must
already be authorized outside the incoming record. **There is no production
referee/catalog.** Tests authorize only named synthetic fixture decisions. Hash
authorization establishes which decision a trusted operator accepted, not proof
that an arbitrary referee or map is correct. Referee independence and its budget
must be established before that operator authorizes a payload.

Maps partition original chains into alive/dead/seki and every resulting empty point
into Black territory, White territory or neutral. Partial/overlapping chains,
missing points, inconsistent regions, living-stone territory and seki-adjacent
territory are rejected. Dead chains disappear only from the scoring view. The exact
White-minus-Black score in half-points is
`2 * (T_white + C_black + D_black - T_black - C_white - D_white) + 13`;
`C` and `D` name the color removed. Living stones do not score. SGFmill's area scorer
is never used.

Automatic checks/judgments use at most 100,000 counted board/proof nodes, two
process-CPU seconds and ten wall seconds, with no GPU. They check the budget between
bounded board operations and publish counts/times. Exhaustion remains unresolved
or unjudgeable. A neutral referee records its own separately authorized budget;
local validation still uses the automatic limits. Invalid maps do not alter the
original state or source result.

## Source replay and persistence

`replay_sgf(bytes)` checks explicit game/board/rules/komi/clock scope and a single
unambiguous ordinary line, then applies every move through the same adapter.
Setup, handicap, altered player order, ambiguous variations, illegal moves and
ordinary continuation after two passes reject the whole game. Setup is available
only through the separately marked fixture constructor, which rejects positions
that SGFmill would repair. SGF's lower-edge coordinates are converted to the public
top-left convention. Both empty and legacy `tt` passes retain their raw encoding.

EOF adds no move or ending. Source `RE`, kind, winner and margin remain separate
from the engine outcome. Even a recorded resignation does not fabricate an engine
resign action. Source hashes remain attached. This replay API always reports
`training_ready=False`; source eligibility, archive conflicts, duplicate handling,
splits and per-head masks belong to the later canonical/example builders. Dynamic
clock inputs and duration labels remain masked by the version-1 data policy.

`dump_game` stores the initial origin, actual action history, candidate color and
ending. `load_game` replays the history, recomputes ko/captures/recurrence, verifies
ordinary origin, and independently rechecks resolved score evidence. Fixture
loading requires `allow_fixture=True`. Loading a neutral scored state requires the
external authorization catalog again; a serialized self-hash cannot authorize it.
Persisted outcome fields must have immutable scalar types; scored margins require
odd integer half-points, rejecting booleans and floats even when numerically equal.

## Reproduce validation

From the repository root with the [workspace cache exports](workspace.md):

```bash
uv run --locked python -m unittest discover -s tests -v
python3 tools/build_katago_reference.py --offline
KENGGO_KATAGO_REFERENCE="$PWD/.cache/katago-reference/katago-reference" \
  uv run --locked python -m unittest discover -s tests -p test_rules_reference.py -v
```

The ordinary suite explicitly skips native comparisons unless the environment
variable is set. The separate reference command must pass; a skip is not native
validation. A fresh machine can omit `--offline` once to fetch the pinned public
KataGo source archive (about 64 MiB). The builder verifies its exact SHA-256, extracts
only needed sources, and uses one `g++` compilation job. No CMake, model weights,
GPU, KGS login or KGS request is needed. Raw archives, binaries and build ledgers
stay under ignored `.cache/` and `runs/`.

The JSONL oracle supports independent `ordinary` and native `history` cases, explicit
boards/captures, optional legal masks, and cloned branches. The tests compare every
one of 362 action-mask entries in the ordinary fixtures, including rejected actions.
Native history deliberately shows its different 2/4/6-pass phases and third-state
cycle no-result. Neither those extra actions nor its score on an empty board can
become the adapter's actual actions or certified result.

When the original cached corpus is available, reproduce the bounded source check:

```bash
uv run --locked python tools/audit_rules_replay.py \
  --output runs/e1-rules-reference-repeat/source_replay.json
```

The tool pins the audit Parquet hash and reads only its 241 metadata-compatible
selected SGFs, verifies each original hash, compares all ordinary states, and records
failures and proof coverage. The other 759 candidates and unselected ZIP members are
not replayed. One measured validation run took 15.96 wall / 16.15 process-plus-child
CPU seconds, with zero HTTP/GPU use; these are run costs, not benchmark claims.
