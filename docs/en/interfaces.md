# bootlab interfaces (v1)

This document describes every layer of bootlab, the data that passes between them, and every configuration
file, so that others can extend it. It is the public form of the project's specification. Where it says
"must", the code stops with an error otherwise.

[中文版](../zh/interfaces.md)

Contents: [1 Layers](#1-layers-and-the-two-fixed-interfaces) ·
[2 One step's data](#2-one-steps-data) · [3 Physics layer](#3-physics-layer) · [4 run.toml](#runtoml) ·
[5 Translation layer](#5-translation-layer-modules) · [6 Generator input](#6-interface-1-the-generators-input) ·
[7 PMP](#7-interface-2-the-pmp) · [8 Execution](#8-execution-layer-one-dive) ·
[9 Resources](#9-resources-from-measurements) · [10 Skydive loop](#skydive-loop) ·
[11 Checkpoints and resume](#11-checkpoints-kept-generations-resume) · [12 Records](#12-records) ·
[13 Installation and sites](#13-installation-bootlabtoml-and-sites) · [14 Multi-node](#multi-node-solves) ·
[15 Probe and site generation](#15-probe-and-site-generation) · [16 Regression](#16-regression) ·
[17 Extending bootlab](#17-extending-bootlab)

## Principles

- **Physics first.** Correct results matter most; engineering is kept to what is needed.
- **Check once, then trust.** Inputs are checked once, in `inputs.load`; later layers trust what they get and
  stop at the first error. No defensive checks inside, no automatic retries.
- **Correctness by comparison.** Every step agrees with a reference program to the digits two builds can agree
  to; the regression (section 16) is the guard.
- **Abnormal endings end the run** with the reason written down (no hanging, no retry, never "converged").
- **Resources from measured data**; physical cores only (`--hint=nomultithread`, `OMP_NUM_THREADS=1`).
- **Exact numbers.** Points, spectrum values and solver outputs are decimal strings at full precision; TOML
  floats are refused.

## 1. Layers and the two fixed interfaces

| Layer | Does | Language |
|---|---|---|
| physics | the model (`bt-model/2` JSON), the spectrum and the objective (TOML) | data |
| translation | reads and checks the inputs; evaluates the spectrum at each point; writes the generator's input; the Skydive loop (outer iterations); records | Python ≥ 3.11, standard library only |
| execution | the PMP generator writes the PMP; `pmp2sdp` writes the SDP; `dynamical_sdp` solves (inner iterations); bootlab reads the outputs | Haskell and C++ programs in one Apptainer image, started by Python through `srun` |

The two fixed interfaces, which v1 does not change:

1. **The generator's input** (`request.json`, section 6): the model file plus, for each point, its
   coordinates and its evaluated spectrum, in the form of the generator's parameter `model` section.
2. **The PMP** (section 7): SDPB 3's standard polynomial-matrix-program JSON parts, one list per point.

A new search algorithm, a new generator (for example for spinning operators) or EFM all attach at these two
points.

Every external program is started as an `srun` step inside the job's allocation (`execute.srun_argv`:
`srun --exact`, explicit `--ntasks`, `--cpus-per-task`, `--mem`, `--hint=nomultithread`,
`--cpu-bind=cores`, `--kill-on-bad-exit=1`; MPI programs through `rank_env.sh`; image programs through
`apptainer exec`).

## 2. One step's data

| # | Data | Producer | Where | Deleted |
|---|---|---|---|---|
| 1 | generator input `request.json` | `pmp.write_request` | `<scratch>/<run id>/step_<n>/request.json` | with the step |
| 2 | block tables (rational approximations of the blocks, one file per spin) | scalar_blocks, called inside the generator | `<scratch>/<run id>/blocks/`, shared across the run's steps | at the end of the run |
| 3 | PMP: JSON parts shared by the step's points, one list per point | the generator | `step_<n>/pmp/json/`, `step_<n>/pmp/<point>/pmp.nsv` | after the step; the end point's is copied to `endpoint/` |
| 4 | SDP (`block_info_*.json`, `block_data_*.bin`) | `pmp2sdp` | `step_<n>/sdp/<point>/` | after the step |
| 5 | solver output and checkpoint | `dynamical_sdp` | `step_<n>/out/`; the checkpoint in `<run>/checkpoint/` | out's small files go to `<run>/steps/<n>/`; the checkpoint is kept (section 11) |

Order: generator (1→2,3) → `k+1` `pmp2sdp` instances (4) → one `dynamical_sdp` (5) → one line in
`solves.jsonl`. `<scratch>` is node-local disk or the shared disk (section 13).

## 3. Physics layer

v1: d = 3, external scalars only.

### 3.1 The model: `bt-model/2` (JSON)

The crossing equations in the VBlock form of simpleboot's AutoEqu: for each channel and each equation a
j × j matrix of F/H (or S/T) terms. The generator reads it; its conventions are fixed (a change needs a new
format version).

```json
{
  "format": "bt-model/2",
  "name": "ising-oe",
  "dimension": 3,
  "externals": [{"name": "sig", "delta": "deltaSig"}, {"name": "eps", "delta": "deltaEps"}],
  "rows": ["odd", "all", "odd", "odd"],
  "channels": [
    {"name": "rep1+", "spins": "even", "basis": ["eps eps", "sig sig"], "matrices": [0, [[...]], ...]},
    ...
  ],
  "fixed": [
    {"name": "identity", "delta": "0", "spin": 0, "matrices": [...]},
    {"name": "external", "basis": ["sig sig eps", "eps eps eps"], "opeAngle": true, "matrices": [...]}
  ],
  "objectives": {
    "feasibility": {"constraints": ["external"], "objective": [], "normalization": [{"fixed": "identity"}]},
    "navigator":   {"constraints": ["external"], "objective": [{"fixed": "identity"}],
                    "sigma": {"shift": "1/5", "step": "3/10", "width": "6"}}
  }
}
```

| Key | Meaning |
|---|---|
| `externals` | the external scalars, each with the name of its dimension coordinate |
| `rows` | one entry per crossing equation: `odd` (odd derivatives only), `even`, or `all` (every derivative; for rows of mixed correlators rewritten with S/T) |
| `channels[]` | exchanged operators: `name` (e.g. `S+`, `T-`: representation and spin parity), `spins` (`even`, `odd`, `all`), `basis` (names of the OPE basis, length j), `matrices` (one j × j matrix per row; `0` = all zero; an entry is `0` or a list of terms) |
| term | `{"coeff": "<exact, sqrt allowed>", "F": [a,b,c,d]}`; also `H`, `S`, `T`. `coeff` defaults to 1 |
| `fixed[]` | the identity (`delta` 0, spin 0) and the external operators' own exchange (`external`; each term names `exchanged`) |
| OPE of `external` | none (a j × j positive-semidefinite constraint), `opeAngle: true` (j = 2; an extra coordinate θ, OPE vector (cos θ, sin θ)), or `opeVector` + `opeCoordinates` (components constants or coordinate names, exact rationals) |
| `objectives` | the objective variants `objective.toml` can choose: constraints, objective, normalization, and `sigma` for the Σ navigator |

Conventions (fixed):

1. `F[a,b,c,d]` is simpleboot's F^{ab,cd}_− (odd derivatives of A(z) − A(1−z)), `H` is F^{ab,cd}_+ (even
   derivatives of A(z) + A(1−z)), with A(z) = v^{(Δb+Δc)/2} g^{ab,cd}(u,v).
2. VBlock pairs `F[a,b,c,d]` with λ_{abO} λ_{cdO}; the upstream blocks with λ_{abO} λ_{dcO}. The generator
   multiplies every term by (−1)^ℓ of the exchanged spin. **Do not write this factor in the file.**
3. Odd-spin signs that the source equation carries explicitly are in the coefficients.
4. Rows of mixed correlators (where the prefactors u^q v^p differ between terms) must be written as one row
   with all derivatives: A(z) terms as `S`, A(1−z) terms as `T` (upstream's s- and t-channel blocks).
5. The navigator objective is the identity's contribution; the Σ normalization (simpleboot's
   `SigmaNavigatorNormalizer`) sums the trace of each channel's matrix over dimensions gap+shift, +step, …,
   gap+shift+width of every continuum; isolated operators are not counted. With coefficient +1, allowed points
   have N < 0.

The point's coordinates are, in this order: the externals' dimensions, then θ (`opeAngle`) or the
`opeCoordinates`, then any extra spectrum parameters (3.2).

The generator checks the file when it reads it (first step): names unique and defined, one j × j matrix per
row, F only in `odd` rows and H only in `even` rows, `exchanged` exactly in the external terms, dimension
expressions exact rationals.

### 3.2 The spectrum: `spectrum.toml`

```toml
format = "bootlab-spectrum/1"
default_above_unitarity = "0"     # every (channel, spin) not set below starts at unitarity + this

[[gap]]                           # exactly one of delta and above_unitarity
channel = "rep1+"
spins   = [0]                     # a list, one spin, "2-40", "2-40:2", or "all"
delta   = "3"

[[isolated]]                      # an operator outside the continua, OPE unknown (j x j PSD)
channel = "rep1+"
spin    = 2
delta   = "delta_tprime"          # may refer to point parameters
```

- Values are strings or integers: exact decimals, or `+ − × ÷` and parentheses of point parameters and
  decimals. They are evaluated with `fractions.Fraction` at each point and written as exact decimals; a value
  that is not a finite decimal is an error.
- A parameter that is not a model coordinate becomes an extra coordinate after the model's. The generator gets
  only the model coordinates and the evaluated spectrum.
- Spins: `"all"` = the channel's allowed spins in the run's spin list; a range takes the allowed spins in it
  (also beyond the spin list); a list must name allowed spins only; empty after expansion is an error.
- Errors (at load): unknown key, unknown channel, a spin the channel does not allow or negative, a (channel,
  spin) set twice, not exactly one of `delta`/`above_unitarity`. At a point: the same isolated operator twice.
- Warnings (at the start, in `driver.log`): negative above-unitarity values; a gap or isolated operator below
  unitarity (d = 3: 1/2 for spin 0, ℓ + 1 for ℓ ≥ 1); an isolated operator above its continuum; a spin not in
  the run's spin list.

### 3.3 The objective: `objective.toml`

```toml
format = "bootlab-objective/1"
kind   = "navigator"                                     # one of the model's objectives
# sigma = { shift = "1/5", step = "3/10", width = "6" }  # overrides the model's Sigma parameters
```

The model is copied into the run directory as `model.json` with `sigma` replaced key by key; the generator
reads that copy. Two navigator normalizations are in use: Σ (`navigator`, comparable with simpleboot) and
GFF (`navigator_gff` in `examples/ising-gff`: the GFF double-trace exchanges as normalization, as in
Hyperion's Ising navigator). Both give the same allowed/excluded decision and island boundary; N and the path
differ.

### 3.4 Points

A point maps coordinate names to decimal strings. Coordinates = model coordinates + extra spectrum parameters;
`run.toml [coords].names` must equal this list, and the gradient, Hessian and steps use its order.

<a id="runtoml"></a>
## 4. `run.toml`

```toml
format = "bootlab-run/1"
name   = "<run name>"                      # the run directory is <runs>/<name>-<yyyymmdd-hhmmss>

[inputs]                                   # relative to run.toml
model = "model.json"; spectrum = "spectrum.toml"; objective = "objective.toml"

[numerics]
lambda       = 11                          # odd, >= 3; nmax = (lambda + 1) / 2
block_preset = "simpleboot"                # "simpleboot" (by Lambda) or "hyperion" (by nmax; then give spins)
# spins = [0, 1, 2, ...]                   # the spin list (default: the preset's)
# block_precision = 1024                   # bits (default: the preset's)
# solver_precision = 785                   # pmp2sdp and solver (simpleboot: 785 for Lambda <= 19, else 1024)

[coords]
names = ["deltaSig", "deltaEps", "theta"]

[skydive]
start = [...]; box_min = [...]; box_max = [...]       # one string per coordinate
max_steps = 300
stop_on_negative = false
hessian_guess = []                         # diagonal of the initial Hessian; [] = the solver's own
find_boundary_direction = []               # [] = minimize N
find_boundary_when_allowed = true          # push along it only after N < 0
gap_upper_limit = "1e-12"                  # required
# gap_upper_limit_track = "1"              # c; "off" = a fixed limit
# gap_upper_limit_track_floor = "0.1"      # kappa
# gap_upper_limit_first_only = true        # only with track "off": true = 0 (no limit) after the first solve
# gap_threshold_tighten = true
# infinitesimal = "1e-20"                  # >= 1e-25
# block_timings = "/abs/path/checkpoint/block_timings"   # a new run: another run's block costs, no timing run

[solver]                                   # SDPB options by flag name, over skydive.solver_defaults(nmax)
[dynamical]                                # dynamical_sdp's own options, over skydive.DYNAMICAL_DEFAULTS

# [exec]
# site = "mysite"                          # else bootlab.toml's site

# [resources]                              # left out: one node
# nodes = "auto"                           # or ["host1", "host2"] of one type
# max_nodes = 3; min_cores = 256; mem_gb = 300; prefer = ["typeA"]
# allow_multinode = true                   # overrides a site's allowed = false
```

The solver defaults are those the reference program passes at this nmax (SDPB's options by nmax, plus
dynamical_sdp's defaults); `tests/test_skydive.py` pins them. `--maxRuntime` is the job's remaining time minus
15 minutes. `load` refuses `gap_upper_limit` ≤ the solver's `dualityGapThreshold` (the first solve would end
`optimal` at once), `infinitesimal` < 1e-25, `gap_upper_limit_track` outside (0, 3], and unknown keys in
`[resources]`.

## 5. Translation layer: modules

| Module | Role | Main entry points |
|---|---|---|
| `inputs.py` | reads and checks model, spectrum, objective, run.toml; spectrum at a point | `load(run_toml) -> Problem`, `resolve_spectrum(problem, point)`, `spectrum_warnings` |
| `numbers.py` | exact decimals and the expression language | `Expr(text).value(env)`, `decimal_text(Fraction)`, `Num` (a decimal string written as a JSON number) |
| `pmp.py` | the generator's input and run; PMP lists and labels | `write_request`, `generate`, `pmp_lists`, `write_labels` |
| `skydive.py` | the Skydive loop, records, endings; solver defaults | `Run.loop`, `Run.step`, `gap_threshold`, `tracked_gap_limit`, `optimal_ending`, `solver_defaults`, `start_state`, `resume_state` |
| `execute.py` | sites and installation; `srun` steps; `pmp2sdp` waves with the memory watchdog; `dynamical_sdp`; reading its output; scratch choice; copies between nodes; checkpoint generations | `load_site`, `srun_argv`, `build_sdps`, `run_dynamical_sdp`, `read_dive`, `classify`, `choose_scratch`, `distribute`, `keep_generation`, `restore_generation` |
| `layout.py` | nodes for a run from the site and bench; the allocation check | `choose_nodes`, `resume_plan`, `check_allocation`, `permitted` |
| `bench.py` | the bench files; solver processes; `pmp2sdp` processes and instances | `load_bench`, `solver_procs`, `grid_ok`, `pmp2sdp_procs`, `pmp2sdp_at_once` |
| `cli.py` | `run`, `resume`, `submit`, `summary`, `regress`, `probe`, `site` | `main` |
| `probe.py` | the cluster probe | `plan`, `node`, `pair`, `merge` subcommands |
| `sitegen.py` | a site from the probe; site diff | `generate`, `choose_scratch`, `multinode` |
| `receive.py` | the receiving end of a copy between nodes | `serve` |
| `regress.py` | the regression check | `check(run_dir)`, `input_dir` |
| `rank_env.sh` | per-rank network device, then `exec` the rest | see 14 |

## 6. Interface 1: the generator's input

One call per step handles all `k+1` points: `<generator> pmp <request.json>` (in the image:
`/opt/bootlab/bin/generator`, which sets the generator's libraries and its scalar_blocks script).

```json
{
  "model": {"name": "data", "file": "<run>/model.json", "lambda": 11, "blockPreset": "simpleboot"},
  "objective": "navigator",
  "points": [
    {"name": "c", "coords": [0.519, 1.4, 0.956],
     "spectrum": {"twistGap": 0, "gaps": [{"channel": "rep1+", "spins": [0], "gap": 3}], "isolated": []}},
    {"name": "plus_0", "coords": [0.51900000000000000001, 1.4, 0.956], "spectrum": {"...": "..."}}
  ],
  "out": "<step dir>/pmp",
  "blocks": "<run's block directory>",
  "cores": 128
}
```

- `model` has the field names of the generator's parameter `model` section (`blockPrecision`, `spinList`
  when given in `[numerics]`).
- `coords` are in model-coordinate order, JSON numbers written with all their decimal digits; `spectrum` is the
  point's evaluated spectrum (same convention).
- Points: `c` (centre) and `plus_0` … `plus_{k-1}` (coordinate i plus ε). The neighbours are computed in exact
  decimal arithmetic.
- The generator reads each coordinate to 512-bit floating point, builds each point's bound, computes the block
  tables (identical tables once for all points), and writes the parts with the upstream numerical code
  (Hyperion's `makeSDPBInput` path). Its own scheduling: worker processes on the node (a CPU pool of `cores`),
  each point's plan computed once on a worker, normalization parts split into about max(512, cores) pieces
  and summed in upstream's order, so the parts are byte-for-byte those of the upstream code.
- Block tables are keyed by about 31 decimal digits of Δ12, so `infinitesimal` must be ≥ 1e-25 (1e-40 would put
  the neighbour on the centre's table and lose the Δ12 derivative).

## 7. Interface 2: the PMP

```
<out>/json/*.json       standard SDPB 3 PMP JSON parts; shared parts written once for the step
<out>/<point>/pmp.nsv   the point's list of parts (NUL-separated absolute paths), given to pmp2sdp --input
<out>/workers/          the generator workers' logs
```

Part file names carry the constraint label (`continuum_<channel>_l<spin>_<u|d><gap>_mat_…`,
`isolated_…`) and a hash. Compare parts by label and content, not by file name. At convergence the centre's
parts are copied to `<run>/endpoint/pmp/parts/` with `pmp.nsv`, `precision` and `labels.json` (for EFM).

## 8. Execution layer: one dive

1. `k+1` `pmp2sdp` instances (`pmp2sdp --precision <p> --input <pmp.nsv> --output <sdp dir>`), in waves
   (section 9).
2. One `dynamical_sdp` on all physical cores of the node(s) (or the bench's choice), with `--sdpDir` the
   centre's SDP, `--newSdpDirs` the neighbours', `--checkpointDir <run>/checkpoint`, warm-started from the
   previous solve (cold at the first), plus the step's options (gap upper limit and threshold, the BFGS state:
   gradient, Δx, Hessian, total iterations, boundary direction).
3. Read `out.txt`, `skydiving_out.txt` and the vector files into a `Dive`:

```python
@dataclass
class Dive:
    verdict: str          # "stepped" | "optimal" | "aborted"
    reason: str           # the solver's terminateReason as written, or "failed: <program> exit <code>"
    primal: str; dual: str; gap: str
    navigator: str | None # N; None when optimal (the solver computes no N then)
    ext_step: list[str]   # the step Δx
    gradient: list[str]; hessian: list[list[str]]   # BFGS
    iterations: int       # cumulative, passed back at the next step
    alpha_p: str; alpha_d: str; beta: str; climbed: bool; find_minimum: bool
```

| Solver ending | verdict |
|---|---|
| `met criteria to update sdp input files` | `stepped` |
| `primal-dual optimal` (and the other normal endings) | `optimal` |
| `maxIterations`, `maxRuntime`, SIGTERM (exit 15) | `aborted`, `unfinished: <reason>` |
| `maxComplementarity`, primal/dual step too small | `aborted`, `stalled: <reason>` |
| a program exits non-zero or its output is missing | `aborted`, `failed: <program> exit <code>` |

An unknown `terminateReason` stops the run (`execute.classify`): add it there.

## 9. Resources from measurements

Each of the three stages (generator → `pmp2sdp` → solver) has the whole node in turn.

| Stage | Parallelism | How bootlab chooses |
|---|---|---|
| `dynamical_sdp` | MPI, once per step | from the bench rows of this node type on one node: the problem nearest in size (distance \|ln(N/Nᵢ)\| + \|ln(blocks/blocksᵢ)\|), its fastest process count, among counts with a non-degenerate grid; no rows: all physical cores. Fixed for the run (the warm-start chain needs it) |
| `pmp2sdp` | MPI, scales poorly | p = min(31, ⌊cores ÷ points⌋) processes per instance. The first step converts the centre alone and reads the srun step's cgroup (memory.current − file + shmem, every 0.2 s) for the peak P; then k = ⌊(node memory − driver_reserve) ÷ (1.15 P)⌋ instances at once. Each step's `--mem` = node memory ÷ instances. A **watchdog** cancels a wave whose step holds more than 95 % of its `--mem` (pmp2sdp does not fail but stalls when memory is short); the step fails. Without cgroup v2 the watchdog is off and the node's MemTotal − MemAvailable is used |
| block tables | scalar_blocks threads, inside the generator | k = min(tables, cores ÷ 4) tables at once, cores ÷ k threads each |

**Non-degenerate grid**: Elemental's default process grid takes rows = the first divisor of p from ⌊√p⌋ up;
p = 127 gives 127 × 1 (slow), 128 gives 16 × 8. Powers of two always qualify.

<a id="bench-files"></a>
**Bench files** `sites/<site>/bench/<node type>.json` (`format = "bootlab-bench/1"`): `node_type`,
`physical_cores`, `sockets`, and lists of rows with their source:

| List | Row fields (main) | Read by |
|---|---|---|
| `solver` | `problem`, `model`, `lambda`, `N`, `blocks`, `precision`, `program`, `nodes`, `procs_per_node`, `s_per_iter`, `max_memused_gb`, `transport` (several nodes; default `ib`), `source` | `bench.solver_procs`, `layout.choose_nodes` |
| `sdp_build`, `blocks`, `generator`, `copy`, `rdma`, `pmp_size`, `nfs` | measurements of each stage | for people; the code reads only `solver` |

A missing bench file means: all physical cores, one node, `pmp2sdp` measured at step 1.

<a id="skydive-loop"></a>
## 10. The Skydive loop

Semantics follow the reference program's Skydive loop (and simpleboot's driver), with the differences below.
Each step:

1. **Options.** Gap upper limit: step 1 (and a cold resume's first) `gap_upper_limit`; then with tracking
   (default c = 1) `min(gap_upper_limit, max(c·|N_prev|, κ·|N_1|, 1000·floor))`, κ =
   `gap_upper_limit_track_floor` (0.1), floor = 1e-30, N_1 = the run's first N; with `"off"`, `first_only`
   passes 0 (no limit) from step 2, else the setting. Stopping threshold: with `gap_threshold_tighten` (default)
   `max(1e-30, min(setting, 1e-3·|N_prev|))`, else the setting. The boundary direction, once on, stays on.
2. Generate the PMP of x and its neighbours, then dive.
3. Write one line to `solves.jsonl`.
4. Decide, in this order:
   - `aborted` → end: `unfinished` (with the solver's reason, `unfinished: …` or `stalled: …`) or `failed`;
     stay at x;
   - `optimal` → if primal and dual objectives have the same sign: `converged`, `allowed` (N < 0) or
     `excluded` (N > 0), copy `endpoint/`; opposite signs with the boundary direction on: `converged:
     boundary` (N reported 0, `n_bound` = the gap); otherwise `stopped: N undecided at optimal` (checkpoint
     kept; tighten the threshold and resume);
   - `stop_on_negative` and N < 0 → `stopped: negative navigator`, at x + Δx;
   - x + Δx outside the box → `stopped: left box`, at x;
   - steps used up → `unfinished: max steps`;
   - else x ← x + Δx, pass iterations, gradient, Δx, Hessian on; turn the boundary direction on if given and
     N < 0.

Why the gap upper limit tracks N: one step's length grows roughly like √gap; a limit far below |N| forces tiny
steps, one near |N| lets the point move (an Ising Λ = 19 run from a standard start converged in 84 steps
with the present rule, against 192 with an earlier version of it). Note: an `allowed` end point is a point with N < 0, not N's unique minimum; along flat
directions the end points of two runs can differ by far more than N does.

## 11. Checkpoints, kept generations, resume

- `dynamical_sdp`'s `--checkpointDir` is `<run>/checkpoint/` (all nodes share it). SDPB writes generation
  g+1 rank by rank, then `checkpoint_new.json`, then renames it to `checkpoint.json` atomically, and keeps two
  generations; any kill leaves the generation `checkpoint.json` names complete, but it may be a mid-call one.
- So after each call, before writing its record k, bootlab makes `checkpoint/kept/<k>/`: hard links to that
  generation's rank files, copies of `checkpoint.json` and `block_timings`, and `layout.json` (`nodes`,
  `node_type`, `ppn`, `generation`), built as `kept/<k>.new` and renamed. Older kept directories are removed
  after the record is written, so at most three generations exist at any time and two after each record.
- **Resume**: the largest `kept/<k>` with k ≤ the number of records; newer and unfinished ones are removed;
  in `checkpoint/` only the rank files of other generations are removed; the kept files are linked back. A
  different layout (nodes, node type, processes per node) means a cold start. The run directory's file
  system must make hard links: every run and resume tries one at the start and stops otherwise.

## 12. Records

All numbers are full-precision decimal strings.

**`solves.jsonl`**, one line per solve:

| Field | Meaning |
|---|---|
| `step`, `x` | step number and point (coordinate order of `[coords].names`) |
| `verdict`, `reason` | section 8 |
| `navigator`, `primal`, `dual`, `gap` | N and the objectives |
| `ext_step`, `gradient`, `hessian`, `iterations` | BFGS state passed on |
| `alpha_p`, `alpha_d`, `beta`, `climbed`, `find_minimum` | solver diagnostics |
| `gap_upper_limit`, `gap_threshold`, `boundary_on` | the step's options |
| `procs`, `node`, `nodes` | processes per node; node 0; all nodes (several nodes) |
| `pmp2sdp` | P, k, the measuring step, memory.peak, the node's memory rise |
| `seconds` | `pmp`, `sdp`, `dive`; with several nodes also `copy`; `checkpoint` |

`solves.jsonl` holds no times of day, so two runs with the same layout and block timings compare byte for
byte.

**`result.json`**: `status`, `reason`, `x`, `coords`, `navigator` (null at `max steps` and at `optimal`;
"0" with `n_bound` on a boundary), `primal`, `dual`, `gap`, `steps`, `inputs` (sha256 of `model.json`,
`spectrum.toml`, `objective.toml` and of `[numerics]`), `versions` (image path and sha256, the generator,
`pmp2sdp` and `dynamical_sdp` in it), `scratch` (`dir`, `shared`, `transfer`), `start`, `end` (ISO 8601 with
the writing machine's UTC offset).

**`summary.md`** (from `solves.jsonl` and `result.json`; `bootlab summary`), **`driver.log`** (time-stamped
lines: decisions with their basis, every command line, the tail of each program's stderr), **`steps/<n>/`**
(the solver's small outputs; `.cmd`, `.out`, `.err` of every program), **`endpoint/`** (section 7).

## 13. Installation, `bootlab.toml` and sites

**Install root** = the directory with `src/` and `sites/`. **`bootlab.toml`** (not in the repository):

| Key | Meaning |
|---|---|
| `site` | the site when `run.toml [exec]` names none |
| `image` | the image when the site has no `sif` |
| `runs` | the runs directory when the site has none (default `<root>/runs`; `$BOOTLAB_RUNS` overrides all) |
| `python` | the driver's Python when the site has none (default `<root>/python/bin/python3` when it exists) |

Relative paths are from the install root; `$VARIABLES` are expanded.

**A site** is `sites/<name>/site.toml` + `setup.sh` (sourced by every job; puts apptainer on `PATH`).
`sites/_template/site.toml` explains each key; summary:

| Key | Meaning |
|---|---|
| `sif`, `python`, `runs` | may be left out (then `bootlab.toml`) |
| `binds` | bound into the container (the runs must be under one; scratch and run dir are added) |
| `setup` | the setup script (`setup.sh`) |
| `generator`, `pmp2sdp`, `dynamical_sdp` | program names in the image |
| `scratch` | a path, or candidates in order (first that every node resolves to the same path and can write; `"shared"` = `<run>/work`, nothing copied between nodes) |
| `scratch_basis`, `shared_measured_at` | written by `site gen`: the measured basis; the directories the shared disk was measured in (the start warns when the runs are elsewhere) |
| `driver_reserve_mb` | memory kept back from the steps (default 4096) |
| `bench` | the bench directory (default `bench/` of the site) |
| `env_unset`, `[env]` | removed from / set in the driver's environment; every step inherits it |
| `[slurm]` | `partition`, `time`, `account`, `qos`, `extra` for `submit` (which adds `--exclusive --hint=nomultithread --mem=0`) |
| `[mpi]` | `transport` (a name: `ib`, `tcp-ipoib`, `tcp-1g`; bench rows of several nodes must match), `net_devices` (rules for `rank_env.sh`), `ib_exclude`, `plugin` (srun `--mpi`, default `pmix`) |
| `[node_types.<t>]` | `hosts` (fnmatch patterns or names), `physical_cores`, `mem_gb` |
| `[multinode]` | `allowed`, `basis`, `prefer`, `peers`, `max_nodes`, `min_efficiency`, `min_gain`, `transfer` (`srun` or `ssh`), `copy_network`, `copy_streams`; for ssh: `ipoib_prefix`, `ssh`, `remote_setup` |
| `[shared_network]` | `server`, `network`, `basis`: the runs file system's route as probed; a node that differs gets a warning |

The image's environment needs (in `[env]`): `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
`OMPI_MCA_pml=ucx`, `OMPI_MCA_btl=^openib`, `UCX_POSIX_USE_PROC_LINK=n`, `PMIX_MCA_gds=hash` (every rank runs in
its own Apptainer instance), plus `UCX_TLS` (`rc,self,sm` for InfiniBand, `tcp,self,sm` for TCP).

**`rank_env.sh`**: `srun` starts every rank through it, outside the container; it sets `UCX_NET_DEVICES` from
`BOOTLAB_NET_DEVICES` (the site's `net_devices`), first matching rule wins: `ib:auto` (this node's IB port:
ACTIVE, LinkUp, InfiniBand, the highest rate; at equal rates the port whose IPoIB interface reaches the shared
file system's server; not `ib_exclude`), `<host glob>=<device>`, or `subnet:<IPv4 prefix>` (the interface
with an address on that network). A host no rule matches stops the rank.

<a id="multi-node-solves"></a>
## 14. Multi-node solves

- One solve uses nodes of one type only (SDPB needs equal processes per node). The generator and `pmp2sdp`
  run on node 0; the step's SDPs are copied to the same path on the others, then `dynamical_sdp` runs across
  all of them. The checkpoint is in the shared run directory.
- **Choosing nodes** (`submit`, `nodes = "auto"`): for each type in `prefer`, the minimum count from memory and
  cores, then one more node while the bench's solver rows for the same model and Λ show
  t(1)/(k·t(k)) ≥ `min_efficiency` and t(k−1)/t(k) ≥ `min_gain`; idle peers are named with `-w`, else the
  next type; else queue with `--constraint`. `--resume` keeps the checkpoint's layout.
- **Copies**, `transfer = "srun"` (default): a light `srun --overlap` step starts `receive.py` on each target;
  node 0 sends a tar stream over TCP (`copy_streams` streams). The receiver binds only this node's address on
  the copy network, accepts only node 0's address, wants a one-time random token (passed in the environment)
  first, refuses absolute names and `..` and unpacks with tarfile's `data` filter, gives up after 120 s
  without the right connection or 600 s without data. File counts and bytes are checked after each copy
  (sha256 of every file with `BOOTLAB_COPY_SHA256=1`). `transfer = "ssh"` pipes tar through ssh instead.
- **TCP only**: with a few hundred ranks a node, MPI's all-to-all TCP connections failed at start-up on the
  test cluster (connect() refused or "Cannot assign requested address", then UCX "Destination is
  unreachable"; 2 × 256 ranks failed 6 of 7 times, 2 × 128 1 of 2, one rank a node never). The cause is not
  verified. So a site without InfiniBand is one node by default.

## 15. Probe and site generation

`bootlab probe plan` (login node) → `probe node` jobs (one per type) and optionally `probe pair` jobs →
`probe merge` → `bootlab site gen`.

| File | Content |
|---|---|
| `plan.json` | `format`, `time`, `host`, `user`, `sif`, `sif_sha256`, `shared`, `shared_where` (path, realpath, st_dev, fstype, source, mount point, mergerfs branch where visible), `apptainer_setup`, `slurm` (config), `srun_mpi`, `nodes` (sinfo), `types` (per type: `hosts`, `physical_cores`, `threads`, `mem_mb`, `features`, `partitions`, `probe_hosts`, `sbatch`) |
| `node-<host>-<job>.json` | `cpu`, `nics`, `ib` (ports at start and end, chosen), `shared` (mount, server, protocol, route, network, hard links, clock, `where`), `local` (candidates: writable, free, device, rotational), `throughput` (per candidate and shared: bootlab's file pattern and large files, 3 repeats), `apptainer`, `image` (UCX transports and devices), `cgroup`, `pythons`, `timezone`, `other_jobs_at_start` |
| `pair-<a>-<b>-<job>.json` | `light_srun`, `ib_ports`, `networks` (kind, prefix, both ends, speed), `perftest`, `startup` (per transport and repeat: exit, seconds, refused / cannot_assign / unreachable counts), `copies` (srun and ssh over each network), `shared` (concurrent writes, cold read) |
| `probe.json` | `plan`, `nodes` (host → records), `pairs`, `types` (with `probed`), `scratch_options` (host → option → seconds, parts, sigma, source) |

<a id="site-generation-rules"></a>
**`site gen` rules** (each value carries a comment naming its source):

- scratch: per probed node the options (write until close + read back of 2 GiB in bootlab's file pattern,
  plus the pair probe's copy or cold read for a type allowed on several nodes); the option present on every
  probed node with the least total wins; ties by name. Within 10 % or 2σ (σ from the repeats' median absolute
  deviation × 1.4826) of the best of the other kind, the basis says "close". A repeat over 3 × its median is
  recorded as a spike.
- MPI: InfiniBand rc with `ib:auto` when any probed node chose an IB port; else TCP on the first Ethernet
  subnet.
- multinode: allowed for a type whose pair probe passed every one of 3 rc start-ups and that has a writable
  local disk; else `allowed = false` with the basis; no pair probe: no `[multinode]`.
- copies: `srun`, over the IPoIB network the pair shares, else Ethernet.
- `[node_types]` from the plan (every type, probed or not); `[slurm]`, `bench`, `sif`, `python`, `runs`,
  `[ib] exclude`, `[multinode]` overrides from the hand file.
- `--mask ib,local-scratch` makes the decisions as if there were no IB or no local disk (the probe's facts
  unchanged): for testing the rules.

## 16. Regression

`bootlab regress [--site] [--node] [--time] [--dry]` writes `examples/ising` at Λ = 11 with three steps to
`<runs>/regress-input-<time>/` and submits one whole-node job running `regress run`; `regress check <run
dir>` compares any run directory. The reference is `tests/reference/ising-l11/bt-steps.json` (the reference
program's first three steps at full precision). Each number must agree to ≥ 130 significant digits
(computed with `decimal` at 500 digits; standard library only); discrete values must be equal. The run
directory gets `regress.json` (`passed`, `least`, per-step digits, problems).

Unit tests: `python/bin/python3 -m unittest discover -s tests` (no cluster needed; the site tests use neutral
fixtures in `tests/fixtures/sites/`). `tests/test_physics.py` checks the physics layer against recorded
outputs of the reference program's parameter checks (6 models, 14 cases).

## 17. Extending bootlab

- **A new algorithm** (Navigator, feasibility, …): a new module next to `skydive.py` that produces points and
  reads results; it reuses `pmp.write_request`/`pmp.generate` (interface 1), `execute.build_sdps` and a new
  execution function for its solver (`sdpb`, `approx_objective`), writes its own record lines, and gets a
  `cli.py` subcommand. Keep the rules: check inputs in `inputs.load`, end on abnormal results, choose resources
  from bench data, and compare with a reference before use.
- **A new generator** (spinning operators): any program that reads a request like section 6 and writes the PMP
  of section 7 plugs in through the site's `generator`.
- **A new site**: probe and generate (section 15), or copy `sites/_template`.
- **New measurements**: append rows to the bench files in the same form, with their source.
- Tests: add unit tests under `tests/`; numerical changes need a regression (section 16) or a comparison of the
  same strength.
