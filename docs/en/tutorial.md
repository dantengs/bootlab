# bootlab tutorial

This tutorial takes you from an empty directory to a finished Skydive run: install, probe the cluster,
generate a site, check the installation against the reference, write a `run.toml`, run, resume, and read the
results. Every step gives the command and what it prints.

Commands run on the cluster's **login node** unless the text says otherwise. Paths in the examples
(`/shared/...`, `node01`, `cpu128`) are placeholders; use your cluster's.

[中文版](../zh/tutorial.md)

## 0. What you need

- Slurm ≥ 21.08 and Apptainer ≥ 1.1 (see the README's [requirements](../../README.md#requirements)).
- The bootlab execution image, `bootlab-exec-<version>-r5.sif` or later, and its sha256. The image holds every
  program bootlab runs (the PMP generator, `scalar_blocks`, `pmp2sdp`, `dynamical_sdp`, Open MPI, UCX) and the
  driver's Python 3.12. How to get it: [Getting the image](../../README.md#getting-the-image).
- A directory on a file system that all compute nodes see (for the installation and the runs).

## 1. Install

```bash
cd /shared/$USER
git clone <this repository> bootlab
cd bootlab
module load apptainer            # whatever puts apptainer on PATH on your cluster
./install.sh --image /shared/images/bootlab-exec-r5.sif --sha256 <sha256 of the image>
```

`install.sh` needs only bash and apptainer. Options: `--site <name>` (the default site, written into
`bootlab.toml`), `--runs <dir>` (where run directories go; default `runs/` in the install root).

What it does and prints:

```
image /shared/images/bootlab-exec-r5.sif: sha256 checked
python: 2259 files from the image checked; 3.12.15
wrote /shared/<you>/bootlab/bootlab.toml
wrote /shared/<you>/bootlab/bin/bootlab
----------------------------------------------------------------------
Ran 18 tests in 0.2s

OK
installation checked: modules import, bin/bootlab starts, quick tests pass
......................................................................
----------------------------------------------------------------------
Ran 216 tests in 15.0s

OK
selftest: every unit test passed
```

1. checks the image's sha256 (when `--sha256` is given);
2. copies the image's `/opt/bootlab/hostpython` to `python/` and checks every file against its MANIFEST.
   `python/bin/python3` runs the image's Python 3.12 on the host with the image's own loader and libraries,
   without setting `LD_LIBRARY_PATH` or `PYTHONHOME`, so programs it starts (`srun`, `sbatch`) see a clean
   environment;
3. writes `bootlab.toml` (installation settings; see `bootlab.example.toml`) unless it exists;
4. writes `bin/bootlab`, the command you use from now on, unless it exists;
5. checks the installation quickly: every module imports, `bin/bootlab` starts, and two quick unit tests
   (`tests/test_install.py`, `tests/test_regress.py`) pass; a failure here stops it;
6. runs `bin/bootlab selftest`, every unit test. A failure there is only a warning (`install.sh: warning:
   bin/bootlab selftest failed`): the installation is in place, and most of those tests are not about it.

Apptainer may print `INFO:` lines of its own; they do not matter. If `apptainer` is not on `PATH`,
`install.sh` stops and says so. An existing `python/` stops it too: it never overwrites (move the old one away
to install anew). The number of tests grows with the code; what matters is `installation checked`, and for the
full tests the last `OK`. Run them again any time with `bin/bootlab selftest`.

From here on use `bin/bootlab <command>`; it works from any directory, also through a symbolic link.

## 2. Probe the cluster

bootlab does not assume anything about your cluster. A probe measures it, and a site configuration is
generated from the measurements.

### 2.1 Node probes (one per node type)

```bash
bin/bootlab probe plan --sif /shared/images/bootlab-exec-r5.sif --out probe --submit
```

On the login node, `probe plan` reads `sinfo`, `scontrol show config` and `srun --mpi=list`, groups the nodes
into types (by Slurm feature, else by sockets × cores, threads, memory and CPU model), and submits one small
job per type on an idle node: 8 cores, 16 GB, `--hint=nomultithread`, 15 minutes. It prints one line per type:

```
cpu128: 12 nodes, 128 physical cores, 515000 MB; node probe on node03
  Submitted batch job 81234
gpu96: 2 nodes, 96 physical cores, 1031000 MB; not enough idle nodes for a node probe
```

Useful options:

- `--shared <dir>`: the directory in which the shared file system is measured. Default: this installation's
  runs directory. **Probe where your runs will be**: one mount point can hide several disk arrays of very
  different speed, and the scratch choice rests on that number. At the start of a run, `driver.log` warns when
  the runs directory is not under a directory the probe measured.
- `--local <dir>` (repeatable): a node-local directory to try as scratch besides the usual candidates
  (`$TMPDIR`, `$SLURM_TMPDIR`, local block-device mounts, `/scratch`, `/local`, `/localscratch`, `/tmp`,
  `/var/tmp`). Use it
  for a scratch directory that is not a mount point of its own.
- `--types a,b`: only these node types.
- `--partition P`: only the nodes of Slurm partition P; the jobs go to it, and the generated site has only the
  node types probed there and `[slurm] partition = "P"`.
- `--node H`: a node probe on H (its type only), even when H is busy: the job waits for it.
- Without `--submit` it only prints the `sbatch` commands and writes `probe/plan.json`.

Each node job writes `probe/node-<host>-<job>.json` and `probe/slurm-probe-node-<job>.out`. It measures, among
other things: physical cores, memory, the InfiniBand ports and their state, network interfaces, every scratch
candidate (writable, free space, device) and the shared directory with bootlab's real file pattern (4 MiB
files, 8 writers, 2 GiB, three times), hard links in the runs directory, where `apptainer` comes from, the UCX
transports in the image, and cgroup v2.

**Node types that had no idle node.** Probe them later into a *separate* directory and copy their records over:

```bash
bin/bootlab probe plan --sif <image> --out probe-later --types gpu96 --submit
# when that job is done:
cp probe-later/node-*.json probe/
```

Do not run `--types` into `probe/` itself: `plan.json` is rewritten and would then list only those types, and a
site generated from it would lack the others.

### 2.2 Pair probe (only for one solve on several nodes)

Skip this if one node per solve is enough: without a pair probe the site is one node only.

```bash
bin/bootlab probe plan --sif <image> --out probe --what pair --sdp <small SDP directory> --submit
```

For each type with two idle nodes it submits one job on both nodes, whole and exclusive, up to one hour. It
measures: whether a light `srun` step reaches both nodes; `ucx_perftest` bandwidth and latency over each
InfiniBand port and each TCP network; a full-scale MPI start-up (every physical core a rank, the given SDP
solved for 2 iterations, three times) for InfiniBand rc, TCP over IPoIB and TCP over Ethernet; copying 2 GiB
between the nodes the way bootlab does (and by `ssh` where that works); and a cold read of the shared disk from
the other node. Results: `probe/pair-<host1>-<host2>-<job>.json`.

The `--sdp` argument is any small SDP directory at precision 785 (an Ising Λ = 11 SDP, about 24 MB, is enough).
One way to get one: run the example of section 5 to its end; the end point's PMP is kept in
`<run>/endpoint/pmp/`, and the exact `pmp2sdp` command bootlab used is in `<run>/steps/<n>/` (the `.cmd`
files), which you can repeat with `--output <dir>` in a small job.

### 2.3 Merge

```bash
bin/bootlab probe merge probe
```

prints `probe/probe.json`: the plan, all node and pair records, and each node's scratch options.

## 3. Generate the site

The probe cannot know your Slurm policy. Write it in a small "hand" file. The file may be empty
(`touch mysite.hand.toml`): then jobs go to the cluster's default partition with a 48-hour limit. Add what your
cluster needs:

```toml
# mysite.hand.toml
[slurm]
partition = "batch"
time = "48:00:00"
# account = "myproject"
# extra = ["--qos=normal"]

# Optional:
# bench = "../othersite/bench"        # measured resource data (section 9), relative to sites/<name>/
# [ib]
# exclude = ["node07=mlx5_0:1"]        # an InfiniBand port known to drop its link
# [multinode]
# transfer = "ssh"                     # overrides of the generated [multinode] keys
```

Then:

```bash
bin/bootlab site gen --probe probe/probe.json --hand mysite.hand.toml --out sites/mysite
```

It writes `sites/mysite/site.toml` and `sites/mysite/setup.sh` and prints the basis of every decision, for
example:

```
node probes: 81234, 81235; pair probes: 81240
node03: /tmp 0.69 s, /local 0.69 s, shared 4.97 s (node probe 81234, pair probe 81240)
...
scratch /local/$USER-bootlab, the fastest measured: /local 6.77 s; /tmp 7.73 s; shared 34.40 s (write until close + read of 2 GiB in bootlab's file pattern, summed over ...); shared measured in /shared/<you>/bootlab/runs (...)
IB ports the probe chose (ib:auto applies the same rule): node03=mlx5_1:1, node04=mlx5_0:1
sites/mysite/site.toml
```

The decisions (details in [interfaces](interfaces.md#site-generation-rules)):

| What | Rule |
|---|---|
| scratch | the fastest measured option, a node-local path or the shared disk, summed over the probed nodes; "close" is noted when within 10 % or 2σ |
| MPI transport | InfiniBand rc when the probe saw active IB ports (each node's port chosen at every start, `ib:auto`), else TCP on the Ethernet subnet |
| several nodes for one solve | only for a type whose pair probe passed all three IB rc start-ups and which has a writable local disk; otherwise one node (a run can override with `allow_multinode = true`) |
| copies between nodes | `srun` + TCP to bootlab's receiver, over IPoIB when the pair shares one |
| shared-disk route check | the NFS server and network every probed node used; `driver.log` warns at the start when a node's differs |

The file is plain TOML with a comment on every generated value saying where it came from; you can edit it. To
compare it with a hand-written site: `bin/bootlab site diff sites/mysite/site.toml <other site.toml> --probe
probe/probe.json`.

Make it the default site:

```bash
echo 'site = "mysite"' >> bootlab.toml
```

(Or name the site in each `run.toml`, `[exec] site = "mysite"`.)

## 4. Check the installation: `bootlab regress`

```bash
bin/bootlab regress                    # or: bin/bootlab regress --node node03 --time 01:00:00
```

It writes the regression inputs (the Ising example at Λ = 11, three steps) to
`runs/regress-input-<time>-<pid>/` and submits one whole-node job. It prints the inputs and the `sbatch` command:

```
# inputs /shared/<you>/bootlab/runs/regress-input-20261113-202539-41872
# the nodes named in run.toml: node03
sbatch --job-name=bootlab --exclusive --hint=nomultithread --mem=0 --time=01:00:00 ... --wrap ...
Submitted batch job 81250
```

On a 96-core node the job takes a few minutes. The end of its output, `runs/slurm-<job>.out`:

```
     step         x navigator  ext_step  gradient   hessian    primal      dual       gap   alpha_p      beta    reason iteration   climbed find_mini
        1       all     196.9     133.9     133.9       all     196.9     197.1     196.9     139.7       all         =         =         =         =
        2     140.2     139.7     133.9     133.8     133.6     139.7     139.9     139.7     139.0       all         =         =         =         =
        3     140.0     139.3     133.4     133.6     133.7     139.3     139.4     139.3     139.4       all         =         =         =         =
least digits: {"x": 140.0, "navigator": 139.3, "ext_step": 133.4, "gradient": 133.6, "hessian": 133.6}
PASS: 3 steps, every number to 130 digits or more
```

Each entry is the number of significant digits to which bootlab's value agrees with the reference
(`tests/reference/ising-l11/bt-steps.json`, three steps of the reference Hyperion-based program); `all` means
every digit, `=` a discrete value that is equal. The check passes when every number agrees to 130 digits or
more (the reference computes the neighbour points in 512-bit binary floating point, bootlab in exact decimal,
so the derivatives agree to about 134 digits, not to every digit). The run directory also holds
`regress.json`; a FAIL exits with code 1.

`bin/bootlab regress check <run dir>` repeats the comparison on an existing run directory, anywhere.

## 5. Write a run

A run is a directory with four input files. `examples/ising/` is the 3d Ising σ–ε system:

```
examples/ising/
  model.json        crossing equations (bt-model/2)
  spectrum.toml     gaps
  objective.toml    which objective of the model
  run.toml          the run: inputs, numerics, coordinates, Skydive options, solver options
```

`run.toml` (abridged; the full key list is in [interfaces](interfaces.md#runtoml)):

```toml
format = "bootlab-run/1"
name   = "ising-l11-skydive"

[inputs]
model     = "model.json"
spectrum  = "spectrum.toml"
objective = "objective.toml"

[numerics]
lambda       = 11              # derivative order, odd
block_preset = "simpleboot"    # block truncation by Lambda

[coords]
names = ["deltaSig", "deltaEps", "theta"]   # must equal the model's coordinates (+ extra spectrum parameters)

[skydive]
start     = ["0.519", "1.4", "0.956"]       # strings: exact decimals
max_steps = 300
box_min   = ["0.517", "1.2", "0.95"]
box_max   = ["0.530", "1.5", "0.99"]
gap_upper_limit = "0.1"
infinitesimal   = "1e-20"

[solver]          # SDPB options by flag name, over the defaults for this Lambda
[dynamical]       # dynamical_sdp's own options
# [exec] site = "mysite"                    # else bootlab.toml's site
# [resources] nodes = "auto"                # several nodes for one solve (section 8)
```

Rules worth knowing:

- Numbers are **strings** (`"0.519"`, `"1e-20"`) or integers. A TOML float such as `0.519` is an error: it has
  already lost the decimal digits you wrote.
- Spectrum values may use the point's parameters: `delta = "deltaSig + 2"`. A parameter that is not a model
  coordinate becomes an extra search coordinate, appended after the model's.
- `inputs.load` checks everything once and lists all errors together; then the run starts.

To check the inputs without submitting anything:

```bash
bin/bootlab submit --dry examples/ising/run.toml
```

It loads and checks the inputs, chooses the nodes and prints the `sbatch` command without running it.

## 6. Run

```bash
bin/bootlab submit examples/ising/run.toml
```

`submit` chooses the nodes (one by default; section 8 for several), prints the reasons and the `sbatch`
command, and submits. The job's output goes to `runs/slurm-<job>.out`; the run directory is
`runs/<name>-<yyyymmdd-hhmmss>/` (the time is the node's local time).

While it runs:

```bash
tail -f runs/ising-l11-skydive-*/driver.log
```

`driver.log` starts with the site, scratch, image (and its sha256), each node's MPI device and route to the
shared disk, the run's `run.toml` and the spectrum's warnings; then one block per step: the generator, the
`pmp2sdp` waves and the solver, with times.

This example (the reference program's own settings, `gap_upper_limit_track = "off"`) leaves its box after a few
steps and ends with `stopped: left box`. For a real search, keep the defaults of section 7 instead (the gap
upper limit then tracks N).

## 7. The ending and the records

A run always ends with one outcome in `result.json` (`status`, `reason`):

| status: reason | when |
|---|---|
| `converged: allowed` / `excluded` | the solver ended `optimal` with primal and dual objectives of the same sign: N < 0 (allowed) or N > 0 (excluded); the end point's PMP and solver output are copied to `endpoint/` |
| `converged: boundary` | with `find_boundary_direction`: optimal with opposite signs; N is reported as 0 and `n_bound` = the gap (|N| ≤ gap) |
| `stopped: N undecided at optimal` | optimal with opposite signs while minimizing N: tighten the solver's gap threshold and resume |
| `stopped: left box` | the next point `x + Δx` is outside `box_min`–`box_max` |
| `stopped: negative navigator` | `stop_on_negative = true` and N < 0 |
| `unfinished: max steps` / `unfinished: <solver reason>` | step limit, iteration or time limit, SIGTERM |
| status `unfinished`, reason `stalled: <solver reason>` | the solver stalled: `maxComplementarity`, primal or dual step too small |
| `failed: <program> exit <code>` | a program failed or its output was missing |

The run directory:

```
run.toml, model.json, spectrum.toml, objective.toml   copies of the inputs (a resume reads these)
driver.log      every command line, the tail of each program's stderr, the decisions and their basis
solves.jsonl    one line per solve: step, x, N, primal, dual, gap, Δx, gradient, Hessian, iterations, ...
result.json     the outcome, end point, inputs' and image's sha256, scratch, start and end time
summary.md      settings, a table of the steps (numbers cut to 8 digits), the outcome
steps/<n>/      the solver's small output files and every program's .cmd/.out/.err
checkpoint/     the solver's checkpoint and kept/<k>/ (deleted when converged)
endpoint/       when converged: pmp/ (the end point's PMP parts, labels.json, precision) and out/
```

`summary.md` is written at the end; `bin/bootlab summary <run dir>` rewrites it at any time from
`solves.jsonl` (for example while the run is still going). All numbers in `solves.jsonl` and `result.json` are
full-precision decimal strings.

The settings that matter most for a real search (defaults in brackets; details in
[interfaces](interfaces.md#skydive-loop)):

- `gap_upper_limit` (required; the production examples use `"1e-12"` or below): the first solve's
  duality-gap upper limit. It must be larger than the solver's `dualityGapThreshold`, or the first solve would
  end `optimal` at once; `load` refuses that.
- `gap_upper_limit_track` (`"1"`): from step 2 the limit follows `min(gap_upper_limit, max(c·|N_prev|,
  κ·|N_1|, 1000·floor))` with κ = `gap_upper_limit_track_floor` (`"0.1"`); `"off"` keeps the fixed limit.
- `gap_threshold_tighten` (`true`): the solver's stopping threshold becomes `max(1e-30, min(setting,
  1e-3·|N_prev|))`; set `false` for boundary-finding runs.
- `infinitesimal` (`"1e-20"`): the finite-difference step for the neighbour points; below `1e-25` is refused
  (the generator keys block tables by about 31 digits of Δ12).
- `find_boundary_direction`: a direction to push along once N < 0 (empty: minimize N).

## 8. Resume, and several nodes

**Resume** a run that ended `unfinished`, `stopped` or `failed`, or whose job was killed:

```bash
# optionally edit the run's own run.toml first, e.g. raise max_steps
bin/bootlab submit --resume runs/ising-l11-skydive-20261113-202539
```

The resume reads the run's current `run.toml`, restores x and the BFGS state from the last line of
`solves.jsonl`, and appends to the same directory. `driver.log` says which way it went:

- `resume: checkpoint generation <g> kept for record <k>, 1 x cpu128 x 128 processes`: warm start from the
  checkpoint kept for the last record (the normal case; also after a kill in the middle of writing a
  checkpoint);
- `resume: layout changed ...` or `resume: no checkpoint kept`: cold start, the solver starts from scratch.
  Its first solve uses `gap_upper_limit`, so keep that at `1e-12` or below.

`submit --resume` asks for the same layout as the checkpoint (the same nodes if idle, else other idle nodes of
that type).

**Several nodes for one solve** (InfiniBand only):

```toml
[resources]
nodes = "auto"                 # or a list of hosts of one type
# max_nodes = 2; mem_gb = 300; prefer = ["cpu128"]
# allow_multinode = true       # override a site's "one node" default (driver.log notes the risk)
```

With `"auto"`, `submit` adds a node only when the site's bench has measured solver times for this model and
Λ showing it pays: efficiency t(1)/(k·t(k)) ≥ 0.8 and gain t(k−1)/t(k) ≥ 1.15. Without such data it stays on
one node and says why.

## 9. Measured resources (bench)

`sites/<site>/bench/<node type>.json` holds measurements: solver seconds per iteration by processes and nodes,
`pmp2sdp` instances and memory, block-table threads, generator times. The run uses them to choose the solver's
processes per node (the measured fastest for the nearest problem size, on a non-degenerate process grid), and
the nodes for `nodes = "auto"`. Without a bench, the solver uses every physical core of one node, and the number
of `pmp2sdp` instances at once is measured in the run's first step (the peak memory of one instance). A
generated site has no bench of its own; point to measurements with `bench = ...` in the hand file. The format is
in [interfaces](interfaces.md#bench-files).

These defaults are safe and need no measurements. For better layouts, measure your own problem on the node
type you will run on:

```bash
bin/bootlab bench mymodel/run.toml [--node H] [--time 02:00:00]
```

One whole-node job: the SDP of the run's start point, then `sdpb` on all physical cores and on half of them,
12 iterations each; the steady seconds per iteration (from the 4th iteration on) become `solver` rows of
`sites/<site>/bench/<node type>.json`, with the machine's configuration as their source. Later runs of the
same model and Λ use the faster count. The table's times are whole seconds, so the measure is coarse when an
iteration takes under a second. Rows for several nodes are not measured by this command.

## 10. When something goes wrong

| Symptom | Where to look |
|---|---|
| `submit` prints `the inputs have errors:` | the list names each key and what is wrong; fix the input files |
| the run ends `failed: generator exit 1` | `steps/<n>/generator.err` (model-file errors are reported by the generator) |
| `failed: pmp2sdp ...` or a cancelled wave | `driver.log`: memory per instance, the watchdog's message |
| the solver stalls (`stalled: maxComplementarity`, step too small) | solver options in `[solver]` (initial matrix scales, `maxComplementarity`) |
| `warning: the runs directory ... is not in the directory the probe measured` | probe again with `--shared` in your runs directory, or ignore if speed does not matter |
| `warning:` about the shared file system's route on a node | an NFS mount fell back to a slower network; the run goes on |
| a resume starts cold | `driver.log` gives the reason (layout changed, no kept checkpoint) |

bootlab never retries on its own. Fix the cause and resume.
