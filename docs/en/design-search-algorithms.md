# Design: feasibility, and bisection, Delaunay search and the navigator on top of it

**Status: a design, not implemented yet.** It is written for whoever implements it: after this document and
[interfaces.md](interfaces.md) you should be able to start on the first phase. The "today" section describes
existing code; the rest is the design. The questions in it that were for the project lead were decided on
2026-11-15; the text follows the decisions, and section 10 lists each of them.

[中文版](../zh/design-search-algorithms.md)

Contents: [0 Conventions](#0-readers-and-conventions) · [1 Today](#1-bootlab-today) ·
[2 Feasibility](#2-feasibility-the-point-primitive) · [3 Search shell](#3-the-search-shell-shared-by-the-three-algorithms) ·
[4 Bisection](#4-bisection) · [5 Delaunay](#5-delaunay-search) · [6 Navigator](#6-navigator) ·
[7 OPE scan](#7-ope-scan-cutting-surfaces) ·
[8 Reference implementations](#8-reference-implementations-what-was-borrowed-and-why) ·
[9 Phases](#9-phased-implementation-plan) · [10 Decisions and risks](#10-decisions-and-risks) ·
[11 Contributing](#11-contributing)

## 0. Readers and conventions

**Read first.** [interfaces.md](interfaces.md) sections 1–12 (layers, the two interfaces, the Skydive loop,
checkpoints, records) and section 19 (the per-point scan). Python and a working knowledge of SDPB's
primal–dual problem and of the navigator function (arXiv:2104.09518) are enough; you do not need to have read
simpleboot or Hyperion: the behaviour used here is written out.

**Terms.**

| Term | Meaning |
|---|---|
| point | a map from coordinate names to decimal strings, in the order of `run.toml [coords].names` (interfaces 3.4) |
| allowed / excluded | the spectrum assumptions at this point are / are not consistent with crossing. The evidence for "excluded" is a dual functional |
| verdict | the outcome of one point: allowed, excluded, undecided, skipped, or not finished (section 2.3) |
| slot | a set of physical cores and memory on one node, computing one point at a time (section 2.9) |
| N | the value of the navigator function. Both of bootlab's navigator normalizations (Σ and GFF, interfaces 3.3) have N < 0 allowed |
| a solve | one `sdpb` call for one point (for the navigator, plus `approx_objective`) |
| reference program | the Hyperion-based driver the regression compares with (interfaces 16) |

**Project rules** (every one applies to the implementation; section 11 shows how):

1. Physics first. Nothing is finished before its numerical comparison passes.
2. Inputs are checked once, in `inputs.load`; later layers trust them and stop at the first error. No defensive
   checks inside, no automatic retries.
3. Abnormal endings end the run: iteration or time limits, stalls, failed programs end it with the reason
   written down; no hanging, never "converged".
4. Resources only from measured data (bench files, first-step measurements); where nothing was measured, say
   so and use the safe default (the whole node).
5. Physical cores only (`--hint=nomultithread`, `OMP_NUM_THREADS=1`).
6. Execution and inputs are decoupled: the image takes only non-physical data; models, spectra, objectives
   and search regions are all data.
7. Numbers are exact decimal strings; TOML floats are refused.
8. Learn ideas, do not copy code: the algorithms of simpleboot and Hyperion may be borrowed, the interfaces and
   names are bootlab's own; the strategies follow simpleboot's behaviour.
9. Auxiliary features stay simple. Speed-ups go only where measurements show a bottleneck, and never change
   numerical results.

Measurements quoted here come from bootlab's development records and from analyses of production runs of
simpleboot and Hyperion, written as general descriptions. Numbers marked "toy" come from small local models (a
grid linear program for a 2d single-correlator problem, random dense SDPs with n = 20); they show directions
only and do not carry over to real problems as they are.

---

## 1. bootlab today

### 1.1 Three layers, two fixed interfaces

| Layer | Does | Code |
|---|---|---|
| physics | the model (`bt-model/2` JSON), the spectrum (`spectrum.toml`), the objective (`objective.toml`), all data | `examples/*/` |
| translation | reads and checks the inputs; evaluates the spectrum exactly at each point; writes the generator's input; the outer loop (today only Skydive and the scan); records | `src/bootlab/*.py`, Python standard library |
| execution | the generator writes the PMP, `pmp2sdp` the SDP, `dynamical_sdp` or `sdpb` solves | programs in one Apptainer image, started through `srun` (or `mpirun` in local mode) |

The two fixed interfaces (interfaces 6 and 7). Every new algorithm attaches here and **does not change them**:

1. **The generator's input `request.json`**: the model file plus each point's coordinates and evaluated
   spectrum. `pmp.write_request(problem, model_file, points, out, blocks, cores, bits)` writes it,
   `pmp.generate(site, node, request, log)` runs the generator. `points` maps point names to points; one request
   may hold any number of points, and identical block tables are computed once. The objective is chosen by
   `objective.toml`'s `kind` (the model's `objectives` have variants such as `feasibility` and `navigator`), one
   for the whole run.
2. **The PMP**: SDPB 3's standard PMP JSON parts, one list per point, `<out>/<point>/pmp.nsv`;
   `pmp.pmp_lists(out, names)` returns the lists.

### 1.2 The Skydive outer loop

`skydive.Run` is the only complete algorithm today; the new ones take their skeleton from it.

- **A step** (`Run.step`): `Run.points` makes the centre `c` and k neighbours `plus_i` (x + ε eᵢ, exact decimals)
  from the current point x; the generator writes the PMP of all k+1 points at once; `Run.build_sdps` runs
  `pmp2sdp` in waves (at the first step the centre alone, measuring its memory; then `bench.pmp2sdp_at_once` at a
  time, with a watchdog cancelling a wave at 95 %); `execute.run_dynamical_sdp` solves; `execute.read_dive`
  reads a `Dive`.
- **Decisions** (`Run.loop`): `aborted` → end (`unfinished` or `failed`); `optimal` → `skydive.optimal_ending`
  checks whether the primal and dual objectives have the same sign, and only then decides `converged:
  allowed/excluded`, else `stopped: N undecided at optimal`; leaving the box, running out of steps, stopping on
  negative N each have their own ending. Nothing is retried.
- **Records**: one line per solve in `solves.jsonl` (`Run.record`), full-precision strings and no times of day,
  so two runs with the same layout compare byte for byte; `result.json` has the ending, the inputs' sha256 and the
  program versions (`Run.finish`); `bootlab summary` writes `summary.md`.
- **Checkpoints**: the solver writes `<run>/checkpoint/`; before each record, `execute.keep_generation` keeps that
  generation as `checkpoint/kept/<k>/` by hard links (with `layout.json`: nodes, node type, processes per node,
  precision); `execute.prune_kept` removes older ones once the record is written.
- **Resume** (`cli.cmd_resume`): `cli.place_checkpoint` takes the newest generation at or below the number of
  records and restores it when the layout is the same (`execute.restore_generation`), else starts cold;
  `skydive.resume_state` takes the point and BFGS state from the last record. A resume is started by hand,
  never automatically.
- **Default solver options**: `skydive.solver_defaults(nmax)`, the values the reference program passes at that
  nmax (pinned by unit tests).

### 1.3 Resources and layout

- **Sites** (interfaces 13): `sites/<name>/site.toml`, generated from measurements by `bootlab probe` +
  `bootlab site gen`; read by `execute.load_site`.
- **Bench** (interfaces 9): `sites/<site>/bench/<node type>.json`, written by `bootlab bench`.
  `bench.solver_procs(bench, cores, n, blocks, program)` takes, among the `solver` rows of the same SDPB version,
  the problem nearest in size (distance |ln N/Nᵢ| + |ln blocks/blocksᵢ|) and its fastest process count, among
  non-degenerate grids only (`bench.grid_ok`); with no data, all physical cores. The count is fixed for a run,
  because the binary checkpoint is stored per rank and cannot be read with another count.
- **pmp2sdp**: `bench.pmp2sdp_procs`, `bench.pmp2sdp_at_once`, `bench.sdp_build_row`, or the first-step
  measurement.
- **Nodes**: `layout.choose_nodes` decides how many nodes one solve uses from the bench's multi-node
  efficiency; `layout.resume_plan` keeps the checkpoint's layout on a resume. One solve uses one node type.
- **Local mode** (interfaces 18): without Slurm `execute.launch` uses the image's `mpirun` with a rankfile;
  concurrent programs get disjoint physical cores (`execute.local_argv`). Physics, records and resume are the
  same as on a cluster.

### 1.4 The existing per-point scan

`scan.Scan` (interfaces 19) is today the only "one `sdpb` per point" path, but only for built-in models
(`bootlab-builtin/1`, spinning external operators):

- Per point: `builtin.write_request` → `builtin.generate` → `execute.build_sdps` → `execute.run_sdpb` (options
  from the generator's `sdpb.args.json` with `[solver]` on top, `scan.solver_args`) → `execute.classify` → one
  line in `solves.jsonl`.
- A failure ends it; `resume` goes on from the first point without a line.
- One point at a time, whole node, cold starts, no checkpoints kept.
- The record's `N` is the **size of the SDP** (`execute.sdp_size`), not a navigator value. The new record format
  avoids this clash of names (section 2.5).
- The `feasibility` objective has **not been verified** on built-in models.

### 1.5 What the new algorithms reuse directly

| Existing | Use in the new algorithms | Change |
|---|---|---|
| `inputs.load`, `resolve_spectrum`, `spectrum_warnings` | read, check, spectrum at a point | checks for the new sections |
| `pmp.write_request`, `pmp.generate`, `pmp.pmp_lists` | one generation for a batch of points | none |
| `Run.points` | the navigator's gradient neighbours | becomes a module-level function that Skydive calls (no change of behaviour) |
| `Run.build_sdps`'s measurement and waves | every new algorithm's SDPs | becomes a shared function (no change of behaviour, checked by the regression) |
| `execute.run_sdpb` | one `sdpb` call | add `--initialCheckpointDir` (warm starts) |
| `execute.classify` | classifying solver endings | none; the verdict is a table of its own (section 2.3) |
| `skydive.optimal_ending` | the sign decision | used as is (without boundary) |
| the `execute.keep_generation` family | keeping each point's checkpoint | used as is, keyed by point id instead of record number |
| `bench.solver_procs`, `grid_ok` | processes per slot | plus a function choosing by throughput (section 2.9) |
| `execute.choose_scratch`, `with_binds`, `clear` | scratch | none |
| `cli.cmd_run`, `cmd_resume`, `write_summary` | dispatch by `run.toml` section | new branches |
| `regress.check`'s digit comparison (`regress.digits`) | comparisons with references | reused |

---

## 2. Feasibility: the point primitive

### 2.1 What deciding one point means

At a point x, with the spectrum assumptions given, crossing becomes SDPB's polynomial matrix program. SDPB's
"dual" problem looks for a functional y that is non-negative on every operator allowed to appear and equals 1
on the normalization; the "primal" problem looks for non-negative OPE coefficients (matrices) solving crossing.

- A dual feasible solution = a functional excluding this spectrum = **excluded**.
- A primal feasible solution = the truncated crossing equations have a solution = **allowed** (at this
  truncation Λ).

In exact arithmetic the two exclude each other. The actual solve is an interior-point method at finite
precision, so the verdict depends on why the solver stopped (section 2.3).

### 2.2 Two ways to decide

| Way | Objective | The solver stops when | Cost | Gives |
|---|---|---|---|---|
| **jump** `decide = "jump"` (default) | the model's `feasibility` variant: empty objective, the identity as normalization | the primal or the dual takes a step of length 1 (a "feasible jump") | usually less than solving to optimality (not measured in bootlab) | allowed / excluded |
| **sign** `decide = "sign"` | the model's `navigator` variant | the duality gap is below the threshold (optimal); then the signs of the primal and dual objectives are compared | a full solve | allowed / excluded / undecided, and N |

**Solver options for jump decisions** follow Hyperion's `jumpFindingParams(nmax)`: on top of the base options by
nmax, `--detectPrimalFeasibleJump` and `--detectDualFeasibleJump` on, `--findPrimalFeasible` and
`--findDualFeasible` off, the primal and dual error thresholds at 1e-200 (so the solver never stops first because
an error is small enough, only on a jump). The part by nmax (precision, initial scales, `maxComplementarity`,
checkpoint interval) has the same source as `skydive.solver_defaults`; write `point.jump_defaults(nmax)`, pinned by
a unit test, with `[solver]` on top. SDPB's documentation says a jump indicates that a feasible solution would be
found "if the precision were high enough", so the verdict depends on precision: **all points of a problem use one
precision**, and every record states it.

**Sign decisions** are a full navigator solve: N lies in [dual objective, primal objective], and only when the two
have the same sign is the point decided (the rule of `skydive.optimal_ending`). It costs more, but gives N, serves
as the navigator's "certification", and checks jump verdicts.

The model must have the matching objective variant. Note that `examples/ising`'s model has `opeAngle`: the
external OPE is (cos θ, sin θ), θ a coordinate, so a feasibility scan in the (Δσ, Δε) plane has θ fixed. To leave
the OPE ratio free, the model's `external` has no `opeAngle` (a j × j positive-semidefinite constraint): another
model file, not a switch of bootlab's.

### 2.3 From solver ending to verdict

`sdpb`'s `terminateReason` is recorded as written; the verdict follows this table, in one function
(`point.verdict(reason, decide, primal, dual)`); a reason not in the table stops the run (as `execute.classify`
does).

| `terminateReason` | jump | sign |
|---|---|---|
| `primal feasible jump detected`, `found primal feasible solution` | **allowed** | not expected: end, `failed: unexpected <reason>` |
| `dual feasible jump detected`, `found dual feasible solution` | **excluded** | same |
| `found primal-dual optimal solution` | **excluded** (as in Hyperion, D2 in section 10) | primal and dual of one sign: allowed or excluded; of opposite signs: **undecided** |
| `maxIterations exceeded` | not finished: `unfinished`, handled by the second table below | same |
| `maxRuntime exceeded`, `SIGTERM signal received` | the job's time ran out or it was stopped: `unfinished`, the run ends | same |
| `maxComplementarity exceeded`, `primal step too small`, `dual step too small` | stalled: `stalled`, handled by the second table below | same |
| a program exits non-zero or there is no `out.txt` | `failed: <program> exit <code>`, the run ends | same |

Where bootlab agrees with the reference implementations and where it does not:

- **Hyperion**'s default verdict is "not dual feasible means allowed", so `maxComplementarity` and `maxIterations`
  both count as **allowed**. bootlab does not do this: a solve that did not finish or stalled gave no evidence and is
  no verdict. `primal-dual optimal`, on the other hand, counts as dual feasible, that is excluded, as in Hyperion.
  The jump options set the error thresholds to 1e-200, so the solver normally stops at a jump; a point that ends
  optimal is most likely very close to the boundary, and a sign decision can check it (R1 in section 10).
- **simpleboot** records these "other" results but leaves them out of the decision, so bisection and Delaunay
  propose **the same point** again in the next round, until the point budget is spent. bootlab does one of two
  things with such a point: end the search, or mark it skipped and never propose it again; it never repeats it.

**Points that did not finish or stalled** (decided by the project lead, 2026-11-15):

| Case | Feasibility family (point lists, bisection, Delaunay, OPE scan) | Navigator |
|---|---|---|
| a cold-started point hits `maxIterations` or stalls | **skipped**: verdict `skipped`, `reason` gives the solver's reason; the point is never proposed again and the search goes on | the whole search ends |
| a warm-started point hits `maxIterations` or stalls | the whole search ends: the warm start itself may be the cause, and skipping would lose a point that could be computed; the warm-start distance limit prevents it (section 2.6) | the whole search ends |

`maxRuntime` and `SIGTERM` mean the job ran out of time or was stopped, not that the point is hard, so they are
never skipped: the run ends, and a resume recomputes the point from its own checkpoint. How each algorithm treats
skipped points is in its own section (sections 3.6, 4.2, 5.2, 7.3).

"Undecided" is not abnormal; it is a verdict, saying the point lies on the boundary to the solver's resolution.
Only sign decisions give it now. Each algorithm says what it does with it; by default each ends the search with the
reason (`stopped: undecided point <id>`), and the user tightens the threshold or raises the precision and resumes.

### 2.4 The point primitive `point.py`

The three algorithms and the existing scan get their points through it; it does not know which algorithm it
works for.

```python
@dataclass
class PointJob:            # a point to compute
    id: int                # unique in the run, numbered in the order points are proposed
    x: dict                # coordinate name -> decimal string
    purpose: str           # "scan" | "bisection" | "delaunay" | "navigator" | "line_search" | ...
    extra: dict            # the navigator's gradient neighbours etc. (section 6.2); empty for feasibility

@dataclass
class PointResult:
    id: int; x: list        # coordinate order
    verdict: str            # "allowed" | "excluded" | "undecided" | "skipped" | "aborted"
    reason: str             # the terminateReason as written, or "unfinished: ...", "stalled: ...", "failed: ..."
    primal: str; dual: str; gap: str
    navigator: str | None   # N for sign decisions (the primal objective, as in simpleboot); None for jumps
    gradient: list | None   # navigator only (section 6.2)
    iterations: int
    warm_from: int | None   # id of the point whose checkpoint started the solve; None for a cold start
    size: dict              # {"n": the SDP's N, "blocks": number of blocks}
    procs: int; slot: int; node: str
    precision: int
    seconds: dict           # pmp, sdp, solve, and approx (navigator)
```

**A round** (`point.solve_round(node, jobs)`) is a round on **one node**. It keeps today's rule that the three hard
stages have the whole node in turn:

1. All of this node's points of the round (for the navigator with their neighbours) go into **one** generator
   request: `pmp.write_request` + `pmp.generate`, on all physical cores of this node. Identical block tables are
   computed once; the block directory is on this node's scratch disk.
2. `pmp2sdp` in waves on this node (the shared function taken out of `Run.build_sdps`, memory measured at the first
   round).
3. One `sdpb` per slot of this node, all at once (section 2.9), `--initialCheckpointDir` pointing at the warm
   start's checkpoint (section 2.6).
4. Read `out.txt`, decide by section 2.3, keep the checkpoint and delete the point's large files (section 2.6),
   return the results.

Several nodes run their rounds independently, without waiting for each other (section 2.9). Within one node the
points take different times in `sdpb` (the iteration counts of jump decisions vary a lot), and a slot that
finishes early waits for this node's round. This is a deliberate simplification: each round records the slots'
idle time; if measurements show much idle time, slots get pipelines of their own (and first it must be confirmed
that two generator processes may write one block directory at the same time, which has not been checked, R3 in
section 10).

### 2.5 Records

In a run directory:

| File | Content |
|---|---|
| `queue.jsonl` | one line per proposed point: `id`, `x`, `purpose`, `round`, `extra`. Append only |
| `points.jsonl` | one line per finished point: every field of `PointResult`. Append only, no times of day |
| `search.json` | the algorithm's current state (section 3.3), rewritten atomically after each round (a temporary file, then a rename); a copy for people, never read by a resume |
| `result.json` | the ending (`status`, `reason`), each algorithm's results (sections 4–6), the inputs' sha256 (including the search section, section 3.5), program versions, `start`, `end` |
| `summary.md` | written by `bootlab summary`: the points, counts of verdicts, the results |
| `driver.log` | time-stamped decisions, every command line, the tail of each program's stderr |
| `points/<id>/` | the solver's small outputs (`out.txt`, `.cmd`, `.out`, `.err`); the navigator's gradient files |
| `checkpoint/<id>/` | checkpoints kept by section 2.6 |
| `files.jsonl` | the file management's records: each round's usage by kind, every deletion (section 2.6) |

New records call N `navigator` and the SDP's size `size.n`, so they cannot be mixed up with the scan records'
`N` (the size of the SDP). The existing scan records are not changed.

### 2.6 Warm starts, checkpoints and file management

These rules hold for every new algorithm; the OPE scan, with many more files, adds a few in section 7.4.

**Which checkpoint a warm start uses** (`[point] warm_start = "nearest"`, the default): among the kept checkpoints,
the one whose point is nearest the new point in **scaled** Euclidean distance (section 3.2's `[coords].scale`); at
equal distance the newer one. Beyond `warm_radius`, a cold start. simpleboot and Hyperion both start from the
nearest point; simpleboot does so however far it is, and the Hyperion-based reference driver recomputes cold after
400 iterations of a warm start. Production data show that a warm start from far away can be 3–6 times slower than a
cold one, hence a distance limit; but "over the limit, start again cold" is an automatic retry and is not done (D4
in section 10): a warm-started point over `maxIterations` ends the search with the reason, for the user to lower
`warm_radius` and resume.

**How deep a checkpoint to start from**: simpleboot's production runs saved an extra checkpoint when μ crossed a
threshold and warm-started from it; with thresholds of 1e-10 to 1e-15 a warm start took 0.20–0.38 of a cold start's
iterations, with 1e-8 0.64–0.88. Whether the image's `sdpb` has such an option is checked with `--help` in phase 0;
if not, warm starts are from the final checkpoint (the reference driver's choice).

**Layout**: the binary checkpoint is stored per rank, so it can be read only by a slot with the same number of
processes. All slots of a run have the same number of cores (section 2.9). Checkpoints are written in the run
directory (the shared disk), as for Skydive, so a slot on any node can use a checkpoint another node left.

**How many are kept** (D8 in section 10):

| Algorithm | Kept by default | Set by |
|---|---|---|
| navigator | the checkpoints of the last 2 evaluations | `[point] keep_checkpoints` |
| Skydive | 2 generations (interfaces 11) | `[skydive] keep_checkpoints`, default 2 |
| feasibility family (point lists, bisection, Delaunay, OPE scan) | max(20, ⌈1.5 × concurrency⌉), e.g. 45 for 30 at once | `[point] keep_checkpoints` |

The feasibility default grows with concurrency because the more points are computed at once, the more widely the
next points' warm starts are spread.

**Order of deletion**. The navigator's path only moves forward: it keeps the newest 2 and deletes the oldest. In
the feasibility family, after each record, when more are kept than the limit:

1. **Protected**: for every point proposed but without a result yet, the checkpoint nearest it is not deleted (it
   is, or is about to be, its warm start).
2. The others are deleted in order of their distance to the nearest of the last R proposed points, largest first,
   with R the concurrency; at equal distance, the one unused for the longest time as a warm start goes first (a new
   checkpoint counts as just used).
3. If the protected checkpoints alone exceed the limit, the limit is exceeded for the moment, with a line in
   `driver.log` and no error.

Why the kept checkpoints suffice: bisection brackets shrink, Delaunay's new points fall on the boundary band, the OPE
scan's θ closes in within one Δ; the next points are always near the latest ones. Checkpoints near those are kept,
far ones go first. And each pending point's start is protected, so it is never deleted before use.

**Deleted as soon as a point is done** (the large files are on the node's scratch disk, the directory
`execute.choose_scratch` chose):

| File | Deleted when |
|---|---|
| the PMP parts and lists of the point (and of its neighbours) | after its record; for the navigator after `approx_objective` |
| the SDPs | same |
| the solver's large outputs (the solution matrices) | same; `out.txt` and the other small files are copied to `points/<id>/` first |
| checkpoints | by the keeping and deletion rules above |
| block tables | kept on the node across rounds (later points reuse them); when the directory exceeds its share of the budget, cleared between rounds, and the generator recomputes them in the next round |

Exception: the navigator's end point, like Skydive's, has its PMP copied to `endpoint/` (for EFM).

**Disk budget** (`[files]`, optional):

```toml
[files]
budget_gb  = 500        # limit on the checkpoints in the run directory; left out: no limit, usage recorded only
blocks_gb  = 50         # limit on each node's block directory; left out: never cleared
```

- Before each round, an estimate: the kept checkpoints plus one new checkpoint for each point of the round (the
  largest size measured in the run). Over `budget_gb`: delete by the order above first (never protected ones); still
  over, end with `stopped: disk budget`, stating the usage of each kind.
- Scratch: before each round, the free space of the node's scratch directory; less than the largest round measured
  in the run: end with `stopped: scratch full`.

**Records**: `files.jsonl` has one line per round (number and bytes of checkpoints, bytes of each node's block
directory and scratch) and one line per deletion (round, kind, point id, bytes, why, e.g. "distance 3.2 to the
nearest recent point, unused for 5 rounds" or "budget"); `driver.log` gets a one-line summary per round.

### 2.7 Reproducibility

- The same point, the same SDP, the same number of processes, a cold start: `sdpb`'s iterations are identical bit
  for bit, and so are the verdict and the iteration count. This is what the acceptance comparisons rest on
  (section 9).
- The SDP itself does not depend on `pmp2sdp`'s process count (measured: identical bits); it does depend on
  `sdpb`'s, in the last digits only (rounding). So comparisons use the same `sdpb` process count, or compare only
  the verdict and the leading digits.
- A warm start changes the path of iterations. Points far from the boundary keep their verdict; points right at
  the boundary may not. So every record has `warm_from`, and acceptance is done with cold starts.
- With several slots, **which point finishes first** changes what Delaunay proposes next. Each point's verdict is
  unaffected, but the path of the search cannot be reproduced point by point. A one-slot run's path is
  deterministic.

### 2.8 Precision

- The generator, `pmp2sdp` and `sdpb` use the problem's precision (`Problem.solver_precision()`, or `[numerics]
  solver_precision`).
- Mixed precision (interfaces 10.1) is not used by the new algorithms at first: its rule is set by Skydive's gap,
  feasibility has no such quantity, and jump verdicts depend on precision. Before using it, compare the verdicts at
  both precisions on the same points (D7 in section 10).
- Coordinates are exact decimals; bisection midpoints and Delaunay candidates are computed exactly with
  `fractions.Fraction` and written as finite decimals (finite when the denominator divides a power of ten; otherwise
  rounded to `digits` significant digits, section 5.3).

### 2.9 Parallelism: slots on one or several nodes

One point's `sdpb` does not need a whole node: in the bench the time per iteration on all cores and on half of
them is often close (for example a small spinning problem on a 256-core node is fastest with 128 processes). Actual
use is often "10 points at once, 48 cores and 100 GB each". So the unit of parallelism is the **slot**: c physical
cores and m GB of memory on one node, computing one point at a time (`sdpb` with c processes). One job may hold
several nodes and many slots, with one driver and one set of records and resume. No job arrays (D1 in section 10).

**Input**:

```toml
[slots]
cores  = 48        # physical cores per slot, also sdpb's processes; or "bench"
mem_gb = 100       # memory per slot; or "bench"
count  = 10        # points computed at once, i.e. the number of slots; 1 for the navigator

[resources]        # as for Skydive (interfaces 4): nodes = "auto" or a list, prefer, max_nodes
prefer = ["typeA"]
```

- **`"bench"`**: the cores are the process count p with the largest throughput per node, ⌊cores/p⌋ / t(p) (t the
  seconds per iteration), among the bench's `solver` rows (same node type, same SDPB version, nearest size); the
  memory is that row's `max_memused_gb` times a margin of 1.15 (the `pmp2sdp` margin). Without such rows a slot is a
  whole node (all physical cores, the node's memory less the reserve), and `driver.log` says there was no data.
  Write it as `bench.slot_size(bench, cores, n, blocks, program)`.
- `bootlab bench` measures all cores and half of them today. For `"bench"` to have data to choose from, `benchrun`
  also measures a quarter of the cores and records each count's peak memory (phase 1 in section 9).
- A slot's cores do not change during a run, so every checkpoint can warm-start every slot (section 2.6).

**Slots per node, and how many nodes** (a new pure function `layout.slot_plan`, called by `bootlab submit`):

- Each node holds k = min(⌊physical cores / c⌋, ⌊(node memory − reserve) / m⌋), the reserve being the site's
  `driver_reserve_mb`; k = 0 is an error at submission.
- Example: a node with 256 physical cores and 376 GB, c = 48, m = 100: 5 by cores, ⌊(376 − 4)/100⌋ = 3 by memory, so
  3 per node; 10 slots need ⌈10/3⌉ = 4 nodes.
- The slots are spread evenly (3, 3, 2, 2, not 3, 3, 3, 1), so each node's generation and `pmp2sdp` load is about
  the same.
- All nodes of one type: in `prefer` order, the first type with enough nodes; idle nodes named with `-w`, else queued
  with `--constraint` (as `layout.choose_nodes`). More nodes than `max_nodes` is an error at submission, which says
  how many slots fit under that limit. `submit` prints the plan and its basis ("3 per node, limited by memory; 4
  nodes").
- Every slot lies within one node and there is no MPI between nodes, so slots need no InfiniBand and are not limited
  by the site's `[multinode] allowed` (which is about one solve across nodes); only `max_nodes` applies.

**How generation and SDPs reach each node**. Two ways:

| | node 0 generates the round, then copies | each node generates its own points |
|---|---|---|
| how | today's multi-node Skydive: the generator and `pmp2sdp` on node 0, `execute.distribute` copies the SDPs to the other nodes | the multi-node design's idea of "points split across nodes for generation": node i's generator and `pmp2sdp` handle only its points, the SDPs stay on its scratch disk |
| copies | SDPs every round, tens of GB a round for a large problem | no SDP copies; checkpoints are in the shared run directory, readable by all |
| generation | all points on node 0 | all nodes at once |
| block tables | computed once | each node computes its own (some repeated), one writer per directory |

The second way. It costs the repeated block tables shared between nodes, and gains no copies, parallel generation
and independent nodes. It requires a point's PMP to be the same whichever node generates it: expected for one node
type, and checked in phase 2's acceptance (one point generated on two nodes, parts byte-identical).

**How the driver schedules**: the driver runs on node 0 with one worker thread per node. Each thread loops: under a
lock, `ask` (`room` = this node's free slots; the `Strategy` sees the pending points of all nodes) → append to
`queue.jsonl` → `point.solve_round(this node, jobs)` → under the lock, append to `points.jsonl`. Programs start with
`srun --exact -w <this node>` (`execute.srun_argv`'s `hosts`): `sdpb` as a step with `--ntasks c --mem m`; the
generator and `pmp2sdp` run while this node's slots are all idle and may use the whole node. Nodes do not wait for
each other. This is the shape of Hyperion's multi-threaded `PointCloudSearch`.

**One node, and local mode**: with no more slots than fit on one node, one thread and one node. Local mode has one
machine, and slots are disjoint CPU slices (`execute.local_argv` already supports this); a slot's memory is not
bounded by a Slurm cgroup there, only their sum is checked against the available memory at the start.

---

## 3. The search shell shared by the three algorithms

### 3.1 Interface: ask and tell

Each algorithm is a class that does arithmetic only, touching no files and starting no programs:

```python
class Strategy:
    def ask(self, known: list[PointResult], pending: list[PointJob], room: int) -> list[PointJob] | Ending:
        """At most `room` new points; when there is none to propose, an ending (Ending("done", ...) etc.)."""
    def state(self) -> dict:
        """The current state, written to search.json (for people only)."""
```

The shell, `search.py`, does everything else:

```
load run.toml → restore from queue.jsonl and points.jsonl (section 3.3)
one thread per node, each looping:
  room = this node's free slots
  first the points interrupted last time, without a result (before calling ask)
  [lock] jobs = strategy.ask(known, pending, room); an ending: tell the other threads to finish, write result.json
  [lock] append to queue.jsonl
  results = point.solve_round(this node, jobs)
  [lock] append each to points.jsonl; a result that ends the run (section 2.3): tell the other threads to finish (section 3.4)
  check the file budget (section 2.6) and the time
```

This is the shape of Hyperion's `PointCloudSearch` (its next-point function gets a map "point → verdict or
pending", pending written `Nothing`; several threads share it) and of simpleboot's scanner shell (scheduled → calculating → calculated). The
differences: bootlab may ask for several points at once (`room`), and all state lives in two append-only files.

### 3.2 Coordinate scales

Every distance in the algorithms (warm starts, Delaunay, the navigator's initial Hessian) is taken in one scaled
frame:

```toml
[coords]
names = ["deltaSig", "deltaEps"]
scale = ["0.001", "0.01"]      # new: a length scale per coordinate; left out: the width of the search box
```

Distance = ‖(x − y)/scale‖. simpleboot is not consistent here (Delaunay scales by the initial points' extent,
BFGS uses raw coordinates, warm starts mix dimensions and OPE coordinates in one distance); Hyperion uses an affine
map that makes the allowed region roughly a unit sphere. bootlab starts with a diagonal scale; a rotation (for long,
slanted islands) can come later as a `[coords] frame` matrix.

### 3.3 State and resume

- **State is a function of the records.** `known` = `points.jsonl`, `pending` = the points of `queue.jsonl` without
  a result. Bisection and Delaunay compute their next points from these two lists every time and have no other
  state. The navigator's BFGS state (current point, Hessian, line-search bracket, phase) is also **replayed** from
  the finished points in the order they were proposed: the replay does only Fraction or fixed-precision `decimal`
  arithmetic, no solves, and matches points by **exact equality** of their decimal strings (simpleboot replays too,
  but looks coordinates up within 10^-170 and rereads its options each time).
- The replayed state is written to `search.json` for people. A unit test requires that the state replayed from
  records cut off anywhere equals the state of an uninterrupted run.
- **Resume**: `bootlab resume <run dir>` reads the records and computes the interrupted points (in `queue.jsonl`,
  not in `points.jsonl`) first, warm-started from the checkpoints they left (the solver writes one on `SIGTERM`).
  No manual draw-back as in simpleboot.

### 3.4 Endings and abnormal results

| Ending | When |
|---|---|
| `done: <the algorithm's statement>` | the algorithm finished normally: every bisection bracket below its threshold, Delaunay at its resolution, the navigator converged |
| `stopped: <reason>` | the algorithm's own stop: nothing left to propose, the box, an undecided point, a navigator guard, the disk budget |
| `unfinished: <reason>` | the point budget (`max_points`), not enough job time for another round, a point not finished or stalled that ends the run (section 2.3) |
| `failed: <reason>` | a program failed |

- **Skipped**: in the feasibility family a cold-started point that did not finish or stalled gets the verdict
  `skipped`, and the search goes on (section 2.3). The skipped point stays in `points.jsonl`, `ask` sees it, so it is
  never proposed again; `result.json` lists every skipped point with its reason.
- **A result that ends the run**: no new points; points still running in other slots (on any node) get SIGTERM (the
  solver writes its checkpoint; recorded as `unfinished: SIGTERM signal received`), then the run ends. A resume
  warm-starts those points from their own checkpoints.
- **Time**: each `sdpb` gets `--maxRuntime` = the job's time left less 15 minutes (as in Skydive). A node starts no
  new round when the time left is below the median of the run's recent rounds; when all nodes have stopped, the run
  ends with `unfinished: job time`.
- No automatic retries.

### 3.5 Frozen options

At the start `run.toml` is copied into the run directory (as for Skydive) and `result.json` records the sha256 of
the search section. A resume refuses any change to that section other than `max_points`, saying why: other options
would replay into another path; start a new run from this run's result instead.

### 3.6 The shared part of `run.toml`

A run has exactly one of `[skydive]`, `[scan]`, `[feasibility]`, `[bisection]`, `[delaunay]`, `[ope]`,
`[navigator]`; `inputs.load` checks it and `cli.cmd_run` dispatches on it. The new algorithms share three sections,
`[point]`, `[slots]` (section 2.9) and `[files]` (section 2.6):

```toml
[point]
decide           = "jump"      # "jump": the objective must be the model's feasibility variant; "sign": a navigator variant
warm_start       = "nearest"   # "nearest" | "cold"
warm_radius      = "1"         # scaled distance; farther: a cold start
# keep_checkpoints = 45        # left out: max(20, ⌈1.5 × concurrency⌉) for the feasibility family, 2 for the navigator

[slots]
cores = 48; mem_gb = 100; count = 10

[solver]                       # sdpb's options by name, over point.jump_defaults(nmax) or the navigator's defaults
```

`inputs.load` checks that `decide` matches `objective.toml`'s `kind` (jump decisions need a variant with an empty
objective, sign decisions a navigator variant), that `warm_radius` is positive, that `[slots]` has positive integers
or `"bench"` and `count` = 1 for the navigator, and that there is one algorithm section.

**`[feasibility]`** (a list of points: the per-point scan for bt-model/2 models):

```toml
[feasibility]
points = [["0.5181", "1.4126"], ["0.5182", "1.4126"]]      # one point a row
# or a grid:
# grid = { min = ["0.517", "1.40"], max = ["0.520", "1.42"], counts = [7, 5] }
```

Result: `result.json` lists the ids `allowed`, `excluded`, `undecided`, `skipped`. A skipped point does not affect
the others; the list is computed to the end.

---

## 4. Bisection

### 4.1 Idea

On a line x(t) = origin + t · direction, t_a is allowed and t_e excluded. Assuming the allowed part of the line is
an interval (monotone), take midpoints, halving the bracket each time, until it is shorter than the threshold.
Report the final pair [t_a, t_e]: the boundary lies between them.

### 4.2 Steps

1. Each line's initial bracket is given (`allowed`, `excluded`, two values of t); both ends are computed first: a
   verdict other than the one given ends the run with `stopped: line <i> end <t> is <verdict>, not <expected>`.
2. Each line keeps its (t, verdict) pairs sorted, pending points included. Find adjacent pairs, both finished, with
   different verdicts; for a bracket ≥ `threshold`, the candidate is its midpoint (exact, with Fraction).
3. With `room` slots: one point on every unfinished line first; with slots left over, **k-section** on the line
   with the longest bracket: split it into k equal parts and compute the k−1 interior points at once (k−1 ≤ free
   slots), shrinking it to 1/k in one round.
4. Verdicts along a line that contradict monotonicity (say allowed, excluded, allowed): end with `stopped: line
   <i> not monotone near t = <t>`, listing the points. No silent choice of one bracket.
5. An undecided point (sign decisions only): end with `stopped: undecided point`, naming its line and bracket. The
   bracket has then reached the solver's resolution, and the user may take it as the boundary.
6. **A skipped point** (section 2.3), handled the simplest way: **this line ends**, the other lines go on. The line
   reports the shortest adjacent pair of opposite verdicts decided before the skip (the skipped point takes no part),
   with status `ended: skipped point <id>` and the reason. No other midpoint is tried on this line: a skipped midpoint
   means the solver has trouble here, another point nearby would likely fare the same, and the sequence would no
   longer match the reference implementations.
7. Every line below `threshold` or ended: `done`; `result.json` says which lines ended on a skip.

### 4.3 Input

```toml
[bisection]
threshold = "1e-6"                         # absolute resolution in t
max_points = 200
[[bisection.line]]                         # one line; any number may be given
origin    = ["0.5181", "1.40"]             # one per coordinate
direction = ["0", "1"]                     # along deltaEps only; any direction works
allowed   = "0"                            # t
excluded  = "0.1"
[[bisection.line]]
origin    = ["0.5185", "1.40"]
direction = ["0", "1"]
allowed   = "0"
excluded  = "0.1"
```

A gap can be bisected too: spectrum values may refer to point parameters (interfaces 3.2), so a gap written as a
parameter (say `delta = "gapS"`) becomes a coordinate, and bisection runs along it. No special gap bisection is
needed.

### 4.4 Output

`result.json`'s `lines`: for each line its `bracket` (the two values of t, `allowed` and `excluded`, and their
points), `points` (the ids computed on it) and `status` (`done`, or `ended: skipped point <id>`).

### 4.5 Parallelism, resume, stopping

- Parallelism: several lines at once; k-section with free slots.
- Resume: lines and brackets come from the records, nothing else.
- Stopping: 4.2; also `max_points` and time (section 3.4).

### 4.6 Code

A new module `bisection.py` (a `Strategy`, arithmetic only), with `search.py` and `point.py`.

---

## 5. Delaunay search

### 5.1 Idea

Triangulate the known points (Delaunay). Simplices whose vertices include both allowed and excluded points
("mixed simplices") straddle the boundary; compute a new point in one of them, and the points gather around the
boundary. It generalizes bisection to higher dimensions and needs only verdicts, no N or gradient. Suited to 2–4
dimensions, an island of unknown shape, and the whole allowed region.

### 5.2 Steps

1. **Initial points**: a list or a grid. There must be allowed and excluded points, at least dimension + 2 in all;
   else the initial points are computed first, and if that is still not met the run ends with `stopped: need
   allowed and excluded points`.
2. **Triangulation**: the known points (pending ones included), scaled by `[coords].scale`, go to the image's
   `qdelaunay` (qhull, options `Qt i`, as in Hyperion), which returns each simplex's vertex numbers.
3. **Drop incomplete simplices**: simplices with a pending vertex are not used (Hyperion's rule). So concurrent slots
   do not propose points in the same place. simpleboot instead drops simplices whose circumsphere holds a pending
   point; the effect is similar and the rule more complicated. **A skipped point** (section 2.3) is a vertex that
   never finishes: it stays in the triangulation and the simplices containing it are never used, so no point is
   proposed around it again. The cost is that this small piece of the boundary band is not refined further;
   `result.json` lists the skipped points, and the user can check them with a separate point list.
4. **Choosing** (`choice`), among the mixed simplices:
   - `longest_mixed_edge` (default): the longest of all mixed edges (one end allowed, the other excluded), its
     midpoint. The midpoint is exact; with one slot this should agree point by point with Hyperion's
     `LongestMixedEdgeMidpoint`, which makes comparison easy.
   - `largest_change_distance`: the simplex whose shortest mixed edge is longest, the mean of its mixed edges'
     midpoints (Hyperion's `ChangeDistance`).
   - `largest_simplex`: the centroid of the mixed simplex of largest volume (simpleboot's rule).
5. For `room` points in one round, take the best `room` by the same score, a later one never in the simplex of an
   earlier one (or put each chosen point in as pending and triangulate again; `qdelaunay` is fast).
6. Candidates outside the box are dropped (the box is required).
7. **Stopping**:
   - the longest mixed edge (scaled) < `resolution`: `done: resolution reached`. Neither reference implementation
     has a resolution; they stop only when their point budget runs out. This is new.
   - no mixed simplex: `stopped: no mixed simplex` (all allowed or all excluded, or the boundary lies outside the
     known points).
   - an undecided point: it is neither allowed nor excluded; the search ends by section 2.3.
   - `max_points`, time.

### 5.3 Input

```toml
[delaunay]
box_min    = ["0.517", "1.39"]
box_max    = ["0.521", "1.43"]
grid       = { min = ["0.517", "1.39"], max = ["0.521", "1.43"], counts = [5, 5] }   # or points = [...]
choice     = "longest_mixed_edge"
resolution = "0.02"            # scaled length
max_points = 300
digits     = 30                # significant digits kept when a candidate (a centroid) is not a finite decimal
```

### 5.4 Output

`result.json`: the ids `allowed`, `excluded` and `skipped`; `mixed_simplices` (of the last triangulation, as vertex ids);
`longest_mixed_edge`. Also `points.csv` (coordinates and verdicts) for plotting.

### 5.5 Parallelism, resume, stopping

- Parallelism: up to `room` points a round, kept apart by steps 3 and 5 of 5.2. The path depends on the order of
  completion (section 2.7).
- Resume: only the records, no other state; `qdelaunay` gives the same triangulation for the same input.
- Dimensions: qhull works in 2–4; the number of simplices grows fast with dimension, so 5 and more are not
  recommended.

### 5.6 Code

A new module `delaunay.py`: scaling, writing qhull's input (count, dimension, coordinates), reading its output,
choosing. `qdelaunay` is a program in the image, started through `execute.launch` as one process (it is not MPI);
its input is in double precision and only fixes the topology; the chosen points are computed with Fraction in the
original coordinates.

---

## 6. Navigator

### 6.1 Idea

The navigator function N(x) is the optimum of an SDP: at x, crossing and the positivity of the spectrum
assumptions are written as "the least λ on the normalization M"; N < 0 is equivalent to allowed and N > 0 to
excluded, and outside the island N measures, continuously, how far away it is (arXiv:2104.09518). Its gradient
follows from the primal–dual solution by the envelope theorem, without another solve. So N can be driven by BFGS:

- **descend**: minimize N; N < 0 means the island has been entered;
- **extremize**: minimize or maximize one coordinate on {N ≤ 0};
- **trace the boundary**: walk along {N = 0} (section 6.8, research).

Each point is solved to the end: the "nested" scheme. Skydive takes Newton steps in the external parameters and the
interior-point variables together ("joint"), one `dynamical_sdp` call a step; see section 6.9.

### 6.2 Where N, the gradient and the Hessian come from

**N**: the objective is the model's `navigator` (Σ normalization, as in simpleboot) or `navigator_gff` variant. One
`sdpb` solves to `dualityGapThreshold`; N is the primal objective (simpleboot's convention) and the dual objective
is recorded too; N's sign is decided when the two have the same sign. The main point must end with `found
primal-dual optimal solution`, otherwise the search ends by section 2.3 (simpleboot does the same, which is why the
navigator's solver options may not turn on jump detection).

**Gradient**: from `approx_objective`, not from finite differences of full solves.

1. The generator writes, in one request, the PMP of x and of k neighbours x + ε eᵢ (as `Run.points` does; ε 1e-20
   by default and never below 1e-25, because the generator keys block tables by about 31 digits of Δ12,
   interfaces 6).
2. All are converted to SDPs; only x is solved. `sdpb` writes its full solution (x, y, X, Y).
3. For each neighbour one `approx_objective` call in linear mode (centre SDP, centre solution, neighbour SDP). Its
   output is the **first-order change d_objective, not divided by the step**, so ∂N/∂xᵢ = d_objectiveᵢ / ε. This was
   settled in bootlab's early trials (against full solves); its second-order mode outputs a second-order change
   that already contains the ½.
4. Known constraints: `pmp2sdp`'s process count when building the SDPs does not change `approx_objective`'s result
   (identical bits); `approx_objective`'s own process count changes the last digits (rounding), so a run fixes it.

simpleboot puts the neighbours at x − ε eᵢ and uses its own `sdpdd` tool. Hyperion runs a few "centering
iterations" before `approx_objective` (option `centeringIterations`).

**What centering iterations are.** An interior-point solver moves its solution along a curve called the central
path towards the optimum; `sdpb` stops when the gap is below the threshold, and the solution is then very close to
the optimum but usually not exactly on the central path: a little off it. `approx_objective` takes this solution
and makes a first-order estimate of "how much the objective changes at the neighbouring point"; the estimate is
derived assuming the solution is on the central path, and the further off it is, the less accurate the estimated
gradient. Centering iterations let `sdpb` take a few more steps at the same point that only pull the solution back
onto the central path, without shrinking the gap further (Hyperion's way: gap and error thresholds 0,
`stepLengthReduction` and `infeasibleCenteringParameter` 1, `maxIterations` the number wanted). They cost a few
iterations per evaluation. Hyperion does a fixed number at every point; bootlab chooses the number **adaptively**
from how far each point's solution is off the central path (D6 in section 10), and none when it is centred enough.

**How far off: R_error.** At the start of every iteration `sdpb` computes R_error = maxAbs(μI − XY) and writes it at
full precision to `out/iterations.json` (the image's SDPB 3.0 does); it is 0 when X and Y lie exactly on the central
path. bootlab uses the relative ρ = R_error / μ, dimensionless. Skydive's logs show this quantity falling by about
2–3 orders of magnitude per centering iteration (e.g. 1.7e-6 → 2.3e-9 → 1.4e-12; whether the R_err in
`dynamical_sdp`'s log has the same normalization is checked in phase 0).

**The error indicator: c · ρ, not the difference of the first- and second-order estimates.** Being off the central
path gives `approx_objective`'s estimate a systematic bias, written as a relative error ≈ c · ρ, with c a constant of
the problem set by the calibration (below). The difference between the first- and second-order estimates is not used:
both come from the same off-centre solution and carry the same bias, so their difference shows only the curvature
over the neighbour step ε (about 1e-20 relative for ε = 1e-20) and is blind to how far off centre the solution is.

**The threshold: how accurate this step needs to be.** The gradient serves to predict how much N falls in the next
step and to update BFGS. So its relative error must satisfy

  c · ρ ≤ η · min(|cos(g, p)|, ‖g − g_prev‖ / ‖g‖)

with η = `centering_accuracy` (default 1e-3: the error at most a thousandth of the expected decrease), p the previous
iteration's search direction and g_prev the previous point's gradient (both terms 1 at the first point). The first
term keeps the error of the predicted decrease g·p small, the second keeps the gradient difference that BFGS uses from
being swamped. The threshold changes from step to step: far from the solution the gradient is large and little
centering is needed; near convergence the gradient differences shrink and the threshold tightens.

**At each point.** After the main solve:

1. Estimate ρ₀. `iterations.json` records R_error at the **start** of each iteration, not after the main solve's last
   step; take the last record's R_error and scale it by μ to the end (ρ₀ ≈ R_last/μ_last, the gap from out.txt).
   The calibration checks this estimate.
2. c · ρ₀ already within the threshold: no centering.
3. Otherwise predict the number n = ⌈log₁₀(c·ρ₀ / threshold) / r⌉, with r the orders gained per centering iteration as
   calibrated (about 2–3 in Skydive's logs), and make one centering call with `maxIterations = n` from the main
   solve's checkpoint, writing the full solution.
4. If the ρ of that call's last record (taken before its last step, which only lowers it, so a conservative bound)
   meets the threshold, done; if not, another call by the same formula.
5. Total iterations reaching `max_centering` (default 6, not measured) without meeting it: the run ends, `unfinished:
   centering did not reach ρ ≤ <threshold> in <n> iterations at point <id>`, with every ρ.

Each point's record has `centering`: iterations, calls, ρ₀, the last ρ, the threshold.

**Calibration: once, at the start of each run.** The first `calibrate_points` points of a run (default 1) do more;
after them nothing is checked point by point:

1. Each of the k neighbours is solved fully, to a gap threshold of `calibration_gap` (default 1e-60). So strict
   because a neighbour's objective differs from the centre's only by about ε·|g| (ε = 1e-20), and the full solves'
   error (about their gap) must be far below it for the digits compared to mean anything. This gives the "true"
   changes Δᵢ.
2. From the main solve, one centering iteration at a time, until ρ stops falling or `max_centering`; at each level
   one `approx_objective`, giving estimates dᵢ and the relative error eⱼ = max |dᵢ − Δᵢ| / |Δᵢ|, with that level's ρⱼ.
3. Fit: c = max eⱼ / ρⱼ (the largest, to be safe); r = the median of the orders between neighbouring levels of ρ;
   and check that step 1's estimate of ρ₀ is within a factor of 10 of the ρ actually measured here.
4. Check the model: eⱼ must fall with ρⱼ. If ρ falls and the error does not (something else dominates it, such as
   full solves not accurate enough, or ε's truncation), the error model does not hold on this problem, and the run
   ends: `stopped: centering error model does not hold`, with each level's ρ and error, for the user to judge.
5. c, r and the levels go into `result.json`'s `centering_calibration` and into `driver.log`. A resume reads them
   from the records and does not calibrate again.

The calibration point costs k strict full solves and a few `approx_objective` calls, once per run.

**Hessian**: BFGS secant updates by default, no extra solves. `approx_objective`'s second-order mode gives an exact
Hessian, at the cost of the SDPs of k(k+1)/2 cross points. Toy measurements (6 parameters, a smooth boundary, mean
number of evaluated points to the endpoint):

| Curvature from | Points |
|---|---|
| BFGS only | 11.6 |
| exact rows and columns for parameters that enter only small blocks (OPE-like) at every point, BFGS for the rest | 8.6 |
| one exact Hessian at the start, then BFGS | 8.2 |
| both of the above | 6.7 |
| exact at every point | 5.2 |
| one exact Hessian at the start, then **no update** | 15.7 (the worst; do not do this) |

So BFGS first; `hessian = "exact_at_start"` is an option for phase 8.

**Finite differences of full solves** (`gradient = "full"`) are for tests only: every neighbour solved to the end,
gradient = (N(x+ε eᵢ) − N(x))/ε, to accept `approx_objective`'s gradient. The centre and the neighbours are solved to
`calibration_gap`, for the same reason as in the calibration.

### 6.3 Phases and states

```
start ──► descend ──(N<0 and mode = extremize)──► extremize ──► done
             │                                      │
             ├─(N<0, mode = minimize, stop_on_negative)──► done: allowed point
             ├─(|g| small while N>0)──► stopped: positive minimum
             └─(box, guards, limits)──► stopped / unfinished
```

- **descend** (simpleboot's MyBFGS with `FeasibilityQ`; Hyperion's `BFGS.run` with `stopOnNegative`): direction p
  = −B⁻¹g (reversed if not a descent direction), a line search (6.4), the BFGS update only when sᵀy > 0.
- **extremize** (simpleboot's ConstrBFGS; the paper's Algorithm 2): on the quadratic model m(s) = N + g·s + ½ sᵀBs,
  fix the target coordinate's step s_a, minimize over the others, and set m = 0, a quadratic in s_a; the smaller
  root for a minimum, the larger for a maximum: the step to the tip the model predicts. Without a real root, the
  model's minimum. This phase starts with descend's last B.
- The phase is part of the state and comes from the replay (section 3.3).

### 6.4 Line search and stopping (baseline: simpleboot's semantics)

By rule 8 the baseline follows simpleboot's behaviour and stays as the control; improvements are options, off by
default, until they are compared on the same problem.

**Descend's line search** (MyBFGS): α = 1 first; extrapolation at least doubles, and quadruples when the
extrapolation is unreliable; interpolation by cubic Hermite, clamped to [0.1, 0.9] of the bracket; accept |g·p| <
0.9 |g₀·p| with N < N₀, or N₀ > 0 and N < 0 (just entered the island: accept at once). α is bounded by the first
crossing of the box along p; at the box while still descending: `stopped: box`.

**Extremize's line search** (ConstrBFGS): α = 1 first (within the box); accept N < r·|N₀| (r = 2 by default); if
not, a two-point cubic interpolation, its minimum for N₀ > 0 and its smallest root for N₀ < 0, clamped to [0.1α,
0.9α]; no root in the bracket: `stopped: line search found no root`.

**Stopping**:

| Condition | Ending |
|---|---|
| \|g⊥\| ≤ `goal_g`, \|g_a\| > 10·max(`goal_g`, \|g⊥\|) and \|N\| < `goal_n` together (g⊥: the gradient without the target coordinate) | `done: endpoint` |
| step \|s\| < `goal_s` | `done: step below goal_s` |
| the target coordinate changed by less than `goal_x` in each of `stable_iterations` iterations (a new, relative criterion, see below) | `done: target coordinate stable` |
| the last two values of N differ by < 10 × `dualityGapThreshold` (simpleboot's first guard) | `stopped: N change below the solver's resolution` |
| the difference slope along the segment between two points disagrees with the directional derivatives at its ends (simpleboot's second guard, threshold 100) | `stopped: gradient inconsistent` |
| `max_iterations` (default 400, as simpleboot) or `max_points` | `unfinished: ...` |
| the main point not optimal, a program failed | section 3.4 |

Why the "target coordinate stable" criterion: in production data **none** of simpleboot's ten navigator runs
stopped by its convergence criterion (absolute thresholds of 1e-20); what stopped them was the first guard, at
points that had converged. Hyperion's endpoint search had dozens of points at the island's tip with |N| ≤ 1e-6, but
\|g⊥\| never came down (the tip is a kink, 6.5), and the points of four directions repeated about every 50 steps. On
a kink \|g⊥\| alone never stops, so a criterion on the target coordinate itself is needed. The first guard stays as a
named ending for the user to judge, not as convergence.

### 6.5 Measured conclusions: what to adopt and what has been ruled out

**To adopt** (from analyses of simpleboot and Hyperion production runs and from bootlab's own measurements):

- **The initial Hessian decides the first step.** simpleboot's default (the identity) makes the first step far too
  long; the line search shrinks at most 10 times per trial, every trial a full solve, and rejected trials cost tens
  of hours in production runs; runs with s·diag(box width) (s = 0.02) did not have this problem. Hyperion uses
  diag(\|g\|/0.2/box width), a first step of about 20 % of the box. bootlab defaults to Hyperion's rule (its units
  are right); `hessian_guess` overrides it, or it is carried over from an earlier run (or from a Skydive's last
  record) (D5 in section 10).
- **78–96 % of the wall time is the main solve**; the gradient is a small part. So what can be saved is the number
  of points and the iterations per point; warm-start depth and the distance limit matter most (2.6).
- **A line search that looks only at N crosses the boundary back and forth**: in one run 30 of 72 points were
  outside. The baseline keeps this behaviour as the control; the improvement (a merit function of the target's
  progress and N's violation) is a phase 8 option.
- **From outside, the descent can stay outside for a long time**, N falling slowly; a rule "slow decrease means no
  island" gives false alarms (one run entered the island after the rule fired). So "no island" is never decided
  automatically: `driver.log` carries a note, and the user can check elsewhere with a feasibility run. \|g\| small
  while N > 0 is a genuine local minimum: `stopped: positive minimum`.
- **Places in Hyperion's code to avoid**: the line search's zoom has no limit on its count; extending past the box
  ends the search silently; the bisecting line search's interpolation fraction is clamped to a constant 0.5 (plain
  bisection); the sign-change search doubles α without looking at the box. Every loop in bootlab has a limit, and
  leaving the box is a named ending.
- **The island's tip is a kink.** On a toy, a peninsula's tip is two smooth boundary branches meeting at about 16°;
  near it N is the larger of two smooth branch functions, and the gradient's direction jumps across the tip.
  Hyperion's measured gradients on the two sides of a tip were (+5, −0.5) and (−1700, +125). A single quadratic model
  (ConstrBFGS, Hyperion's BFGS, Skydive's BFGS) is the wrong model at a kink. What worked on toys: follow one branch
  along the boundary and, when the gradient's direction jumps, take the intersection of the two tangents (16–23
  points to the tip); or keep linear cuts from all points and take the next point from a small linear program (18
  points to 5e-6, but without curvature it did not reach 1e-2 in 150 points on a smooth 6-dimensional boundary).
  Measured only on toys; phase 8.

**Ruled out; do not recommend again:**

| Method | Finding |
|---|---|
| geodesic extrapolation X_k X_{k−1}⁻¹ X_k as a starting point | no better than an unchanged warm start (X⁻¹ magnifies the subspace's misalignment by 1/μ) |
| inferring a real solver's iteration counts from the grid linear-program toy | does not carry over: the grid makes the solution path non-smooth |
| reusing a dual functional at nearby points (certifying without a solve) | certifies less than 1 % of the distance moved |
| a barrier homotopy in parameter space, max v·p + ν log(−N) | 19–69 points, against 5–12 for model steps |
| a Gaussian-process surrogate for the boundary | accuracy stalls at 1e-3, worse at kinks |
| extrapolating the boundary's position across Λ | worse than reusing the lower Λ's point; what carries over is the start, the scales and the initial Hessian |
| intersecting two branches' linear models while still far from the tip | fails: outside the island the value belongs to the other branch, and the model misplaces the boundary |
| one exact Hessian at the start, then frozen | worse than BFGS |
| a log X extrapolated start with Ŷ = μX̂⁻¹ | worse than the unchanged start on the real solver (neighbouring records have different μ) |
| two-level μ extrapolation instead of descending | on the real solver saves 1–2 orders of descent, not the whole descent as on the toy |
| Skydive with a trust region, a lower `stepMinThreshold`, more climbing | the step length is set by the gap (measured step ∝ gap^0.46); none of these lengthens it |
| the solver's external corrector | deadlocked while saving its state on a larger problem; unusable |

### 6.6 Input

```toml
[navigator]
mode       = "extremize"                 # "minimize" | "extremize"
target     = { coordinate = "deltaEps", sense = "max" }    # extremize only
start      = ["0.5182", "1.4130"]
box_min    = ["0.5170", "1.3900"]
box_max    = ["0.5200", "1.4300"]
stop_on_negative = true                  # minimize: stop at the first point with N<0
gradient   = "approx_objective"          # "approx_objective" | "full" (tests only)
infinitesimal = "1e-20"                  # >= 1e-25
centering_accuracy = "1e-3"              # η: the gradient's relative error allowed (against the expected decrease); sets the centering threshold
max_centering = 6                        # limit on a point's total centering iterations; reaching it without enough ends the run
calibrate_points = 1                     # points calibrated at the start
calibration_gap = "1e-60"                # gap threshold of the calibration's full solves of the neighbours
hessian    = "bfgs"                      # "bfgs" | "exact_at_start" (phase 8)
# hessian_guess = ["1000000", "10000"]   # a diagonal, or a full k x k matrix; left out: diag(|g|/0.2/box width)
line_search = "simpleboot"               # "simpleboot" | "wolfe" (Hyperion's strong Wolfe, phase 8)
accept_ratio = "2"                       # extremize's line search accepts N < r |N0|
goal_g = "1e-20"; goal_n = "1e-20"; goal_s = "1e-20"
goal_x = "1e-10"; stable_iterations = 5
max_iterations = 400
max_points = 300

[point]
decide = "sign"                          # the navigator's main point is always solved to optimal; this sets the records' verdict
warm_start = "nearest"
warm_radius = "0.5"
# keep_checkpoints = 2                   # the navigator keeps the checkpoints of its last 2 evaluations by default

[slots]
cores = "bench"; mem_gb = "bench"; count = 1   # the navigator is sequential: one slot

[solver]
dualityGapThreshold = "1e-30"            # sets N's resolution and the first guard
```

`inputs.load` checks: `objective.toml`'s `kind` is a navigator variant; `start` is in the box;
`target.coordinate` is a coordinate name; `infinitesimal` ≥ 1e-25; every vector has one entry per coordinate;
`decide = "jump"` is incompatible with the navigator.

### 6.7 Output, parallelism, resume

- **Records**: one `points.jsonl` line per evaluation (`purpose` = `navigator` or `line_search`, with `navigator`
  and `gradient`); `search.json` holds the state: phase, current point, B, line-search bracket, iteration.
- **Result**: `result.json`'s `x`, `navigator`, `gradient`, `hessian` (the last B, usable by Skydive as it is),
  `inverse_hessian`, `iterations`, `phase`.
- **Parallelism**: an evaluation is the generation of k+1 points and one main solve, in one slot (`[slots] count =
  1`; its cores from the bench by default, the whole node without data). A point that does not finish or stalls ends
  the whole search (section 2.3), never skipped: the next BFGS step depends on it. Later, the line search's trial α can go to different slots at
  once (phase 8): a wrong trial then costs a slot, not a solve followed by a shorter step.
- **Resume**: the state is replayed from the records (3.3); an interrupted evaluation warm-starts from its own
  checkpoint.

### 6.8 Tracing the boundary (research)

On a 2d slice, starting from a boundary point, walk along {N = 0}. What worked on toys: estimate the curvature from
the tangents at the last two nodes and predict along a circular arc; at the predicted point compute N and ∇N and
take one Newton projection along the normal, x − N g/\|g\|²; when the projection is small, take the projected point
without another solve; a sudden change of tangent after the correction is a kink, and the node there is the
intersection of the two tangents; cubic Hermite between nodes. On a toy one side of a peninsula plus its tip took
29–82 evaluations, against 270–673 for the "attractor + repeller + tangential drift" flow.

Limits: in n dimensions the boundary has n−1 dimensions; above three, do not trace the whole surface, only 2d
projections, or give the 2n coordinate endpoints. The traced nodes are numerical zeros at a finite gap; to quote a
point as a bound in a paper, a point just outside must still be solved to the end and shown to have N > 0. Toy
evidence only; phase 8 starts with a free diagnostic on existing records before deciding.

### 6.9 Relation with Skydive

| | Navigator (this section) | Skydive (existing) |
|---|---|---|
| each point | solved to the end; N's sign can be decided | one `dynamical_sdp` call, a short step, not solved to the end |
| outer iteration | BFGS in the driver (Python, replayable) | BFGS inside the solver |
| progress | every evaluation is a cold or warm solve to the end | about 75 % of each step is fixed cost; step ∝ √gap |
| parallelism | line-search trials can run in parallel | sequential |
| suited to | points that each need a decided N; endpoints of any coordinate; certification | getting from a point into the island or to its boundary fast |

Shared code: the neighbours (`Run.points`, moved out), generation and `pmp2sdp` (one shared function), the bench,
checkpoint keeping. Hand-overs:

- **Skydive → navigator**: the Skydive's end point as `start`, the `hessian` of its last record (a full matrix) as
  `hessian_guess`.
- **Navigator → Skydive**: the navigator's end point as `[skydive] start`, its last B as `hessian_guess`. Today
  `hessian_guess` takes only a diagonal; it gets to take a full matrix as well (a small change; the Skydive regression
  unchanged).
- **Checkpoints**: whether `sdpb`'s checkpoint can start `dynamical_sdp` has not been checked (R5 in section 10);
  until it is, the solver starts cold at a hand-over.
- **Certification**: a Skydive's end point checked by one sign decision (`[feasibility] points = [end point]`,
  `decide = "sign"`). bootlab has checked a Skydive end point this way by hand before: bootlab and the reference
  program each built the SDP at the same point and solved it to the end, and N agreed to 162 digits.

---

## 7. OPE scan (cutting surfaces)

To be done later (phase 7 in section 9); the design is written down now. This algorithm piles up many files, so the
emphasis here is file management.

### 7.1 Idea

In mixed correlators some isolated operators (the external operators' own exchange, say) have an OPE coefficient
that is a j-component vector λ whose direction is unknown. bootlab's models write it in one of two ways (interfaces
3.1): with λ's direction as a coordinate (`opeAngle`'s θ, or `opeCoordinates`), so that λ is fixed at each point;
or without a direction, as a j × j positive-semidefinite constraint, which allows every PSD matrix, a weaker
condition than "some direction λ", and gives too large an allowed region.

The OPE scan sits between the two: the outer search runs over the dimensions (and other coordinates) only, and at
each outer point Δ the inner loop decides whether some λ makes (Δ, λ) allowed. The method is cutting surfaces
(arXiv:1912.03324; simpleboot's OPE scanner is this):

- Decide (Δ, λ). Allowed: Δ is allowed, the inner loop ends.
- Excluded: the functional y the solver found, acting on the external block's j × j matrix, gives a quadratic form
  Q; every direction λ where λᵀQλ has the sign it has at the current λ is excluded by this same functional. Each
  exclusion cuts away a piece of λ space.
- Choose the next λ in the region not yet cut away (its "centre"). Nothing left: Δ is excluded.

The outer search (a point list, bisection or Delaunay) sees only each Δ's final verdict.

### 7.2 The inner steps for one Δ

1. **Initial λ**: this Δ's last λ (on a resume); else the last λ of the nearest finished Δ (simpleboot's rule);
   else `[ope] start`.
2. **Decide** (a jump decision, section 2.2). When excluded, the solver must write y.
3. **The cut**: the external block is one PMP part; at fixed Δ it is j × j constant vectors M_ab, and Q_ab = y ·
   M_ab (with the component SDPB eliminates through the normalization). The driver computes it with Fraction from the
   part's JSON and y, with no other program. The sign convention is fixed by the first acceptance of phase 7: the
   current λ must be excluded by its own cut.
4. **The remaining region**: the intersection of all cuts {λ : λᵀQᵢλ of the sign opposite to exclusion} with `[ope]
   box`.
5. **The next λ**. Only j = 2 at first (`opeAngle`, one angle θ): each cut is a trigonometric quadratic in θ whose
   zeros can be solved exactly, the remaining region is a set of intervals, and the next θ is the midpoint of the
   longest one; no interval means nothing is left. So "nothing left" is an exact statement. For j ≥ 3 simpleboot
   searches for a feasible point by random sampling, and failing to find one does not prove the region empty, so it
   is no evidence of exclusion; j ≥ 3 comes later (D12 in section 10).
6. **Limit**: at most `max_cuts` cuts per Δ (simpleboot has no limit); at the limit this Δ is skipped (section 2.3),
   reason `ope: max cuts`.

Hyperion has an OPE search too (hyperion-bootstrap's `OPESearch`, built on bilinear forms of the functional). It was
not studied for this document; read it before implementing, and record what is borrowed and why as in section 8.

### 7.3 Outer search, skipping, parallelism

- `outer` chooses the outer search, `"points"`, `"bisection"` or `"delaunay"`, with its section written as usual over
  the outer coordinates only (no OPE coordinates). The outer search sees one verdict per Δ: allowed, excluded or
  skipped.
- **Skipping**: an inner decision of a cold-started point that does not finish or stalls skips this Δ (the outer
  search treats it as in sections 4 and 5); the same on a warm-started point ends the run (section 2.3).
- **Parallelism**: a slot works on one Δ's inner loop at a time (the inner loop is sequential); `[slots] count` is the
  number of Δ worked on at once.
- **Records**: `points.jsonl`, one line per decision (`purpose = "ope"`, `extra` with the Δ's point id and λ);
  `deltas.jsonl`, one line per finished Δ (verdict, number of cuts, last λ); `cuts/<id>.json`, all Q of that Δ (small,
  all kept: resume and checks need them).

### 7.4 File management

An OPE scan has hundreds of Δ, each with a few to tens of decisions, and each decision produces a PMP, an SDP, solver
output and a checkpoint. Unmanaged, they soon fill the disk. The rules, on top of section 2.6:

| File | Where | Kept until |
|---|---|---|
| PMP parts and lists | node scratch | the decision's SDP is built |
| SDP | node scratch | the decision is solved |
| the solution y | node scratch | Q is computed and written to `cuts/` |
| the solver's other large outputs | node scratch | the decision is solved; `out.txt` copied to `points/<id>/` |
| checkpoints within one Δ | run directory | only the latest decision's; the next λ starts from it and the previous one is deleted |
| a finished Δ's checkpoint | run directory | its last decision's is kept and enters section 2.6's keeping and deletion order (distances in outer coordinates) |
| block tables | node scratch | reused across Δ; cleared between rounds beyond `[files] blocks_gb` |
| `cuts/`, records | run directory | all kept |

**Making sure the kept checkpoints suffice for later warm starts**: each Δ being worked on has its current
checkpoint protected; whenever the outer search proposes a new Δ, the checkpoint of the nearest finished Δ is marked
protected at once, until the new Δ's first decision starts. The rest go by section 2.6's order: farthest from the
latest proposed Δ first, at equal distance the longest unused first.

**Disk budget**: in the run directory ≈ (`keep_checkpoints` + concurrency) × the size of one checkpoint; on each
node's scratch ≈ that node's slots × (one point's PMP + SDP + y) + the block directory. Once the first round has
measured each size, `driver.log` writes this estimate; beyond `[files] budget_gb` or the scratch's free space, section
2.6 applies. Every deletion goes to `files.jsonl`.

### 7.5 Input

```toml
[ope]
outer       = "delaunay"         # "points" | "bisection" | "delaunay"; the outer search's section as usual
coordinates = ["theta"]          # which coordinates are OPE coordinates (inner); the others are outer
start       = ["0.9"]
box_min     = ["0.5"]
box_max     = ["1.3"]
max_cuts    = 50                 # cuts per Δ; at the limit this Δ is skipped

[delaunay]                       # the outer search, over the outer coordinates only
box_min = ["0.517", "1.39"]; box_max = ["0.521", "1.43"]
grid = { min = ["0.517", "1.39"], max = ["0.521", "1.43"], counts = [5, 5] }
resolution = "0.02"; max_points = 300

[point]
decide = "jump"                  # the OPE scan uses jump decisions only

[slots]
cores = 48; mem_gb = 100; count = 10

[files]
budget_gb = 500; blocks_gb = 50
```

### 7.6 Code

A new module `ope.py`: the inner `Strategy` (one state per Δ, replayed from `points.jsonl` and `cuts/`), the cut
computation, the interval arithmetic for j = 2. The outer search reuses `bisection.py` and `delaunay.py`; the file
management functions live in `search.py`, shared with the feasibility family.

---

## 8. Reference implementations: what was borrowed and why

The sources of simpleboot (Mathematica) and of the Hyperion ecosystem (Haskell: hyperion-bootstrap, scalars-3d,
bootstrap-params, sdpb-haskell) were read. Ideas are learned, code is not copied; interfaces and names are
bootlab's.

| Idea | From | Taken? | Why |
|---|---|---|---|
| one "next point" shell for every strategy, pending points passed to the chooser to avoid duplicates | both | yes | the three algorithms differ only in choosing |
| results in three classes, "other" silently dropped | simpleboot | **no** | proposes the same point again and again |
| "not dual feasible" means allowed | Hyperion's default | **no** | `maxComplementarity` and time-outs would count as allowed |
| options for jump decisions | Hyperion `jumpFindingParams` | yes | ready, by nmax |
| `primal-dual optimal` as dual feasible (excluded) | Hyperion | yes | the project lead's decision (D2 in section 10) |
| a cold retry after a parse error or a warm start's maxIterations | simpleboot | **no** | rule 2: no automatic retries |
| a point that did not finish: skipped in the feasibility family, never proposed again | new | added | no repetition as in simpleboot, and one hard point does not stop a whole scan |
| warm start from the nearest point | both | yes, with a distance limit | far warm starts can be 3–6 times slower |
| recompute cold after 400 iterations of a warm start | the Hyperion-based reference driver | **no** | a retry; a distance limit and an ending instead |
| warm start from a mid checkpoint | simpleboot's production use | yes (if `sdpb` has it) | measured 0.2–0.4 of the iterations |
| bisection: adjacent pairs of opposite verdicts, the longest bracket, pending points in the sort | simpleboot `BisectionSampler`, Hyperion `pointMapToBracket` | yes | simple, concurrency comes free |
| bisection in any direction | neither has it natively | new | the line form covers coordinate directions too |
| k-section | new | added | uses free slots |
| stop when not monotone | new | added | no silent choice of a bracket |
| Delaunay drops simplices with pending vertices | Hyperion | yes | simpler than the circumsphere rule |
| choosing: midpoint of the longest mixed edge | Hyperion | default | exact rationals, comparable point by point |
| choosing: centroid of the largest simplex | simpleboot | option | keeps simpleboot's behaviour |
| scaling: affine map / initial points' extent | Hyperion / simpleboot | one `[coords].scale` | one definition of distance everywhere |
| a resolution for Delaunay | neither | new | otherwise only the point budget stops it |
| `qdelaunay Qt i` | Hyperion | yes | qhull is in the image |
| navigator state = a function of the history (replay) | simpleboot | yes, with exact matching and frozen options | resume is correct by construction; simpleboot matches within a tolerance and does not store options, so changing them changes the path |
| ConstrBFGS's ellipsoid-tip step, accepting N < r\|N₀\|, the [0.1, 0.9] clamp | simpleboot | the baseline | rule 8; the control |
| MyBFGS with FeasibilityQ, then ConstrBFGS, two phases | simpleboot (two runs by hand) | yes, as explicit phases | the phase change carries B |
| strong Wolfe line search, a Hessian kept and Newton steps by LU | Hyperion `BFGS.run` | option | compare with the baseline first |
| initial Hessian diag(\|g\|/0.2/box width) | Hyperion, the paper's Algorithm 1 | default | simpleboot's identity measured badly, and s·box width has the wrong units |
| absolute thresholds of 1e-20, a hard cap of 400 | simpleboot | kept, plus a relative criterion | production runs never reached the absolute thresholds |
| the two guards | simpleboot | yes, as named endings | what actually stopped runs, but not convergence |
| gradient from `approx_objective` | Hyperion | yes | in the image; the conversion is settled |
| gradient from `sdpdd`, neighbours at x − ε eᵢ | simpleboot | no | not in the image; neighbours at +ε as in Skydive |
| centering iterations before `approx_objective` | Hyperion (a fixed number) | yes, adaptive by R_error | none when centred enough; the number follows the accuracy the step needs, calibrated once at the start |
| N as y/(1−y) of the dual objective | Hyperion's GFF navigator | not needed | both of bootlab's normalizations give N directly |
| the cutting-surface OPE scan, the last θ carried from Δ to Δ | simpleboot | yes | section 7 |
| finding a feasible θ: minimization, random sampling, an external SDP relaxation | simpleboot | exact intervals for j = 2 first | failing to sample a point does not prove the region empty; for j = 2 it can be decided exactly |
| the last two checkpoints kept per θ, no limit on the inner loop | simpleboot | one per Δ, plus `max_cuts` | file management (7.4); abnormal cases need an ending |

A few more simpleboot behaviours found while reading its code, to keep in mind: a box written as `Automatic` does
not actually work (so bootlab's box is required); its explicit-Hessian tool generates the shifted SDPs at the
unshifted point (a suspected bug, not copied); its Skydive passes `totalIterationCount 0` at every call (bootlab
already counts cumulatively).

---

## 9. Phased implementation plan

No durations. After each phase the next one is planned from what actually happened. Every phase delivers code,
unit tests, `run.toml` examples, and the matching sections of interfaces in both languages.

**Test size**: unless stated otherwise, the Ising σ–ε system of `examples/ising` at Λ = 11 (the SDP has N = 104 and
36 blocks; one point takes minutes on one node), on one node. Phase 6 adds a larger problem (N≈2300) for the timing
and verdicts of a few points only, not a whole search.

**Reference results**: each comparison first runs simpleboot or Hyperion (the reference program) on the same
problem. Reference results **do not go into the public repository**; they stay with the maintainers, and the
conclusions of the acceptance (whether the verdicts agree, to how many digits) go into interfaces. Once there are
Ising tests, a small one may be put in `tests/reference/` as an example (D10 in section 10).

### Phase 0: checking the premises (reading, plus a few small jobs)

| Deliverable | Acceptance |
|---|---|
| versions and `--help` of the image's `sdpb`, `approx_objective` and `qdelaunay` in interfaces; confirm `--initialCheckpointDir`, the mid-checkpoint option, `approx_objective`'s linear and second-order options, the R-err in `out/iterations.json` (6.2) and whether its normalization is that of `dynamical_sdp`'s log | documented; `execute`'s command lines pinned by unit tests |
| the generator's `feasibility` objective: bootlab's PMP of one point against the reference program's | parts byte-identical (the standard of interfaces 6) |
| `point.jump_defaults(nmax)` | identical to Hyperion's `jumpFindingParams`, option by option (unit test) |

### Phase 1: the point primitive and point lists

| Deliverable | Acceptance |
|---|---|
| `point.py` (2.4), `search.py` (3), input checks of `[point]` and `[feasibility]`; `Run.build_sdps` and `Run.points` as shared functions | the Skydive regression `bootlab regress` passes as before (≥ 130 digits), `solves.jsonl` unchanged except for seconds |
| jump decisions: about 30 points across the Λ = 11 island's boundary (some very close to it) | every point's `terminateReason` and verdict the same as the reference program's (same solver options, cold starts, same process count); same iteration counts |
| sign decisions: the same points | same verdicts; primal and dual objectives agree with the reference program to ≥ 130 digits (the regression's standard) |
| slots on one node: `[slots]`, `bench.slot_size`, `layout.slot_plan` (one node), `benchrun` measuring a quarter of the cores and memory | the same points with 1 and with s slots: identical verdicts point by point; each round records the slots' idle time |
| skipping | a skip provoked by cold-started points with a very small `maxIterations`: recorded as `skipped`, never proposed again, the list computed to the end; the same on warm-started points ends the run |
| warm starts and file management (2.6) | points far from the boundary keep their verdicts; `warm_from` recorded; the ratio of warm to cold iterations written in interfaces; kept counts and deletion order as the unit tests expect, `files.jsonl`'s usage equal to the actual directory sizes |
| resume | SIGTERM in the middle of a round (`bootlab stop` in local mode) and SIGKILL between rounds: after the resume, the same result as an uninterrupted run (verdicts, set of points) |
| local mode | the same points give the same verdicts in local mode |

### Phase 2: slots on several nodes (right after phase 1's one-node slot acceptance)

| Deliverable | Acceptance |
|---|---|
| the multi-node part of `layout.slot_plan`, `bootlab submit` printing the plan; one worker thread per node, each node generating its own points (2.9) | unit tests: for given node types and `[slots]`, the slots per node, number of nodes and spread match section 2.9's example; beyond `max_nodes` an error stating how many slots fit |
| one point generated on two nodes of one type | PMP parts byte-identical |
| a set of points (about 30) on 2 nodes, several slots each | verdicts identical point by point to phase 1's one-node run; one set of records; each node's idle time written in interfaces |
| resume | SIGKILL of the whole job while slots on a node are computing; resumed on another set of nodes, the same result as an uninterrupted run |
| the user's real size: 10 slots of 48 cores and 100 GB | one run, recording the number of nodes, slots per node and time per round; timing only, no comparison |

### Phase 3: bisection (after phase 1; in parallel with phases 2, 4 and 5)

| Deliverable | Acceptance |
|---|---|
| `bisection.py`, `[bisection]` | one slot, two lines (one along Δε, one slanted): the sequence of brackets **identical point by point** to the reference implementation (simpleboot's `BisectionSampler` or Hyperion's `binarySearch`, same initial bracket and threshold): the midpoints are exact and the verdicts equal, so the sequences are equal |
| k-section | the final bracket contains the one-slot final bracket; points and rounds written in interfaces |
| the endings for non-monotone lines, undecided points and skipped midpoints (this line ends, the others go on) | unit tests (made-up verdict sequences) |

### Phase 4: Delaunay (after phase 1)

| Deliverable | Acceptance |
|---|---|
| `delaunay.py`, `[delaunay]` | 2d (a (Δσ, Δε) slice with θ fixed): one slot, `longest_mixed_edge`, the same initial points and scales: the first M points (M = 30) identical to Hyperion's `delaunaySearch` (`LongestMixedEdgeMidpoint`, one thread) |
| several slots | the region covered by the final mixed simplices consistent with one slot's (every verdict the same, the boundary bands within each other up to the resolution) |
| 3d | one run to the resolution, consistent with a Skydive's or navigator's end point on the same problem (the end point in the allowed region) |
| skipped vertices | unit test: no point is proposed in a simplex with a skipped vertex |

### Phase 5: navigator descent (the execution functions can be written after phase 0; runs need phase 1's shell)

| Deliverable | Acceptance |
|---|---|
| the `approx_objective` execution function and the gradient; adaptive centering and its calibration (6.2) | the calibration on Ising Λ = 11 gives c and r, with the error falling with ρ (the model holds). At 5 further points beyond the calibration, the neighbours solved fully to `calibration_gap`: at each point the calibrated bound c·ρ is not below the actual error (the estimated digits are no more than the digits actually matched), and the actual error meets the threshold. The mean centering iterations per point and the share of points with none written in interfaces. Agreement with simpleboot's gradient up to the O(ε) difference |
| centering's abnormal cases | unit tests: `max_centering` reached without enough, an error not falling with ρ; both endings named |
| `navigator.py`'s `minimize`, BFGS, MyBFGS line search, replay | against simpleboot's MyBFGS on the same problem, start, box and initial inverse Hessian: the same number of accepted points, the points of the first 10 iterations agreeing to ≥ 10 digits, the same ending |
| replay | unit test: the state replayed from records cut anywhere equals the uninterrupted one |

### Phase 6: navigator endpoints (after phase 5)

| Deliverable | Acceptance |
|---|---|
| `extremize`, ConstrBFGS's line search, the stopping rules and the two guards | against simpleboot's ConstrBFGS maximizing Δε (same problem, start and options): end points agree to `goal_x`, point counts within a few; every ending named |
| hand-overs | Skydive end point → navigator, navigator end point → Skydive (`hessian_guess` taking a full matrix; the Skydive regression unchanged) |
| a larger problem (N≈2300) | the times of 3 points (generation, `pmp2sdp`, main solve, `approx_objective`) in the bench; verdicts agree with sign decisions |

### Phase 7: OPE scan (later; needs the outer searches of phases 3 and 4)

| Deliverable | Acceptance |
|---|---|
| the cut (7.2 step 3) | at an excluded point: the current λ is excluded by its own cut; Q at the current λ consistent with the solver's dual objective (the sign convention written in interfaces) |
| `ope.py`, `[ope]`, j = 2 | Ising Λ = 11 with θ as the OPE coordinate and a point list as the outer search (about 20 Δ): every Δ's verdict the same as simpleboot's OPE scanner's (the θ sequences need not agree: simpleboot's choice is random) |
| file management (7.4) | one run with Delaunay outside: at no time more checkpoints in the run directory than the kept number plus the concurrency; the scratch peak within a factor of two of 7.4's estimate; every deletion in `files.jsonl` |

### Phase 8: research improvements (independent of each other; each starts with a free diagnostic)

| Item | Diagnostic first (no cluster) | Go on when |
|---|---|---|
| kink-aware endgame (tangent intersection or cutting planes) | from existing navigator records, pairs of neighbouring points on both sides of the boundary: the angle between normalized gradients against their distance | the angle does not go to zero as the distance does |
| curvature tiers (exact OPE rows, one exact Hessian at the start) | replay on existing records | fewer points on Ising Λ = 11 with an OPE coordinate added |
| parallel line search | none | after the slot acceptance |
| 2d boundary tracing (6.8) | none | after phase 6: trace a whole island of the Ising Λ = 11 2d navigator, compare with Delaunay |
| Hyperion's strong Wolfe | none | point counts against the baseline on phase 5's problem |

### Order and parallel work

```
phase 0 ──► phase 1 ──┬──► phase 2 (slots on several nodes)
     │                ├──► phase 3 (bisection) ──┐
     │                ├──► phase 4 (Delaunay)  ──┴──► phase 7 (OPE scan, later)
     └──► phase 5's execution functions and gradient ──(needs phase 1's shell)──► phase 5 ──► phase 6 ──► phase 8
```

Phases 2, 3, 4 and 5 do not depend on each other and can be done by different people at the same time; all depend
only on phase 1's `point.py` and `search.py`, so phase 1's interfaces must settle first. Phase 2 comes right after
the one-node slot acceptance.

---

## 10. Decisions and risks

### 10.1 Decided (by the project lead, 2026-11-15)

| Id | Question | Decision | Where |
|---|---|---|---|
| D1 | how to parallelize | slots in one job, no job arrays; slots may span several nodes (one node type), with one driver and one set of records and resume; `[slots]` gives each slot's cores and memory and the total concurrency, or `"bench"`; slots on several nodes come right after the one-node slot acceptance | section 2.9; phase 2 |
| D2 | `primal-dual optimal` under jump decisions | excluded, as in Hyperion | section 2.3 |
| D3 | a point unfinished or stalled | navigator: the whole search ends. Feasibility family: skipped, recorded as `skipped` with the reason, never proposed again; Delaunay treats it as a vertex that never finishes; bisection ends that line and the others go on | sections 2.3, 4.2, 5.2 |
| D4 | a warm start over its iteration limit | the run ends; prevented by warm-starting only from points not too far away | section 2.6 |
| D5 | the navigator's default initial Hessian | Hyperion's rule, diag(\|g\|/0.2/box width) | section 6.5 |
| D6 | centering iterations before `approx_objective` | adaptive: centering until ρ = R_error/μ meets a threshold set by the gradient accuracy this step needs, none when centred enough; calibrated once at the start of each run; a limit on the total, reaching it without enough ends the run | section 6.2 (with a plain explanation); phase 5 |
| D7 | mixed precision for the new algorithms | not at first | section 2.8 |
| D8 | checkpoints kept | the navigator and Skydive: the last 2 by default; the feasibility family: max(20, ⌈1.5 × concurrency⌉); all settable in `run.toml`; warm starts from the nearest; deletion by distance and last use | section 2.6 |
| D9 | the OPE scan (cutting surfaces) | later, but designed now in a section of its own and placed in the plan; the emphasis is file management, which the feasibility family uses too | section 7; phase 7 |
| D10 | reference results | not in the public repository; once there are Ising tests, a small one may be added as an example | section 9 |
| D11 | the new algorithms for built-in models (spinning externals) | later; unchanged | section 1.4 |
| D12 | how an OPE scan with j ≥ 3 (multi-component OPE directions) proves "nothing left" | j = 2 only at first; j ≥ 3 decided later | section 7.2 |
| D13 | whether `[files] budget_gb` has a default limit | no limit, usage recorded | section 2.6 |
| D14 | whether rounds within a node become per-slot pipelines | synchronous rounds first; decided later by the idle time measured in phases 1 and 2 (R2) | section 2.4 |

### 10.2 Risks

| Id | Risk | Countermeasure |
|---|---|---|
| R1 | jump verdicts depend on precision; points close to the boundary may differ between precisions; `primal-dual optimal` counted as excluded may be wrong very close to the boundary | one precision per run; phase 1 adds sign decisions on the points near the boundary |
| R2 | rounds synchronous within a node leave fast slots idle | idle time recorded each round; pipelines if it is large |
| R3 | whether two generator processes may write one block directory at once is unchecked | one generator process and one block directory per node; check before pipelining |
| R10 | several nodes writing checkpoints to the shared disk at once may exceed its bandwidth | phase 2 records the time to write checkpoints; decide if it is too slow |
| R11 | skipped points leave holes in Delaunay's boundary band and end bisection lines early | `result.json` lists the skipped points; the user checks them with a separate point list |
| R4 | warm starts make verdicts at the boundary path-dependent | `warm_from` recorded; acceptance with cold starts; `warm_start = "cold"` available |
| R5 | whether `sdpb`'s checkpoint can start `dynamical_sdp` is unchecked | cold starts at hand-overs; a separate test later |
| R6 | a single quadratic model does not converge at a kink | the "target coordinate stable" criterion stops it; kink handling in phase 8 |
| R7 | from outside with no island at all, the descent goes on without entering | `max_points`; `stopped: positive minimum`; a note telling the user to check with a feasibility run |
| R8 | Delaunay needs many points on a long, thin island | `[coords].scale` at the island's scale; rotation later |
| R9 | Delaunay's path with several slots is not reproducible point by point | acceptance with one slot; several slots compared by verdicts and region only |

---

## 11. Contributing

### 11.1 How to write the code

- **Python**: standard library only, at the version in `pyproject.toml`. Numbers are `fractions.Fraction` or
  fixed-precision `decimal`, and decimal strings in and out of files (`numbers.decimal_text`, `numbers.frac`).
- **Checks at the boundary only**: every check of a new section goes in `inputs.load` (or a function it calls),
  listing all errors at once before exiting (`SystemExit`), each saying which key, why, and how to fix it. Later
  code trusts its inputs.
- **Stop on errors**: a failed program raises `execute.StepFailed` and the search ends by section 3.4; never catch
  it and carry on, never retry. Points are skipped only by section 2.3's rule, never as a way around an error.
- **Algorithms do arithmetic only**: a `Strategy` touches no files and starts no programs, so unit tests can drive
  it with made-up verdicts and the replay can use it.
- **Resources from data only**: process counts, slots and concurrency come from the bench or a first-step
  measurement, and `driver.log` names the basis. Without data, the whole node.
- **Comments say why**: why a number has its value, on which measurement or which reference implementation's
  behaviour. In the public repository write general descriptions: no paths, host names or job numbers.
- **Names**: whole English words; record fields in lower case with underscores; an ending is one sentence `status:
  reason` (see `skydive.ending`).
- **Small, separate modules**: one file per algorithm; shared code in `point.py` and `search.py`; algorithms do not
  import each other.

### 11.2 How to write tests

- Unit tests go in `tests/test_<module>.py`, with `unittest`, no cluster: replace the execution layer with
  `unittest.mock` (see how `tests/test_skydive.py` drives the loop with made-up `Dive`s and pins command lines), and
  use the neutral sites in `tests/fixtures/sites/`.
- Each algorithm needs at least: a whole search driven by made-up verdicts (say "allowed when x₁ + x₂ < 1") checking
  the ending and the set of points; records cut off midway and resumed, with the same result; one test per error of
  the input checks.
- Tests pinning command lines (`sdpb`, `approx_objective`, `qdelaunay`) option by option against the reference
  implementation, written like `tests/test_skydive.py`'s pinning of the reference program's command line.
- **Numerical changes** need a regression or a comparison of the same strength (the acceptance of each phase in
  section 9), with the result in the commit message or in interfaces.

### 11.3 How to run

```bash
bin/bootlab selftest
```

```bash
bin/bootlab regress
```

```bash
bin/bootlab regress check <run dir>
```

`selftest` runs every unit test, no cluster needed; `regress` submits one whole-node job on a cluster that runs
three Skydive steps of Ising Λ = 11 and compares them with the reference program to ≥ 130 digits (in local mode,
`--wait` runs it in the foreground); `regress check` checks any run directory. Changes to shared code (generation,
`pmp2sdp`, execution functions) always need `regress`.

### 11.4 How to commit

- One change per commit; the message says what, why, and how it was checked (which unit tests, which regression,
  how many digits agreed).
- No run directories, checkpoints or SDPs in commits; reference results stay out of the public repository (D10 in
  section 10).
- Keep each file's line endings; look at `git diff --stat` before committing to catch whole-file line-ending
  changes.
- Change the documentation in both languages, with the same content.
- When an AI assistant writes code, give it this document, [interfaces.md](interfaces.md) and the modules to change,
  have it write the unit tests and the acceptance plan first and the code after, and check every numerical
  convention it states by a comparison. The approach of the [guide to writing models with AI](writing-models-with-ai.md)
  — give the complete format and conventions first, then check that the result matches item by item — applies here
  too.
