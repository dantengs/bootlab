# bootlab

**bootlab** is a lightweight framework for the numerical conformal bootstrap on Slurm clusters.

This is **version 1, a deliberately small release: it has one search algorithm, Skydive** (the "skydiving"
algorithm of Liu, Simmons-Duffin, Su and van Rees, [arXiv:2307.13046](https://arxiv.org/abs/2307.13046)),
for 3d CFTs with external scalars, with SDPB's `dynamical_sdp` as the solver. Other algorithms (Navigator,
feasibility scans and bisection, Delaunay, OPE scans, spinning external operators) are planned; see
[Roadmap](#roadmap).

[中文说明](README.zh-CN.md)

## How it works

bootlab is three layers with two fixed interfaces between them:

```
model.json (bt-model/2) ─┐   translation layer (Python)               execution layer (programs in one Apptainer image)
spectrum.toml ───────────┼─► load ─► Skydive loop ─► generator input ═══► PMP generator ═══► PMP ─► pmp2sdp ─► dynamical_sdp
objective.toml ──────────┘              ▲            (interface 1)        (Haskell)       (interface 2)              │
run.toml ───────────────────────────────┘◄──────────────── one record per solve: stepped / optimal / aborted ◄───────┘
```

1. **Physics layer** (data): the model's crossing equations (`model.json`), the spectrum assumptions
   (`spectrum.toml`) and the objective (`objective.toml`). Each can be replaced on its own.
2. **Translation layer** (Python, standard library only): reads and checks the inputs once, evaluates the
   spectrum exactly at each point, writes the generator's input, runs the Skydive loop and keeps the records.
3. **Execution layer** (programs in the bootlab image): the PMP generator (Haskell, on Hyperion and
   scalars-3d) writes the polynomial matrix program, `pmp2sdp` turns it into an SDP on node-local disk, and
   `dynamical_sdp` solves it, warm-started from the previous step's checkpoint, and proposes the next step.

Each Skydive step evaluates the current point `x` and its `k` neighbours `x + ε eᵢ`, solves once, and either
moves to `x + Δx` or ends the run with a written outcome (`converged`, `stopped`, `unfinished`, `failed`).
Abnormal endings (iteration or time limits, stalls, crashes) end the run with the reason; nothing is retried
automatically. A run is resumed by hand from the last kept solver checkpoint.

## Main features

- **Inputs as data.** Crossing equations in a JSON model (simpleboot's VBlock conventions), spectrum and
  objective in TOML. Spectrum values may be exact expressions in the point's parameters (for example
  `"deltaSig + 2"`) and are evaluated with exact rational arithmetic; numbers are full-precision decimal
  strings everywhere, never binary floats.
- **Checked against a reference.** Every step of the regression case (Ising, Λ = 11) agrees with the
  reference Hyperion-based program to at least 130 significant digits; `bin/bootlab regress` repeats the
  check on your cluster.
- **Resources from measurements.** Processes per solve, `pmp2sdp` instances at once and the number of nodes
  for one solve are chosen from measured data (`sites/<site>/bench/`), or from a measurement made in the
  run's first step; physical cores only, never hyperthreads.
- **Portable.** A cluster probe (`bootlab probe`) measures the node types, InfiniBand ports, local and shared
  disks and the network, and `bootlab site gen` writes the site configuration from it. The host needs only
  Slurm and Apptainer: the driver's Python 3.12 is copied out of the image.
- **Restartable.** The solver's checkpoint lives in the run directory; for each record bootlab keeps that
  solve's checkpoint generation with hard links, so a job killed at any moment resumes from the last record.
- **Plain records.** One JSON line per solve (`solves.jsonl`), the outcome in `result.json`, every command
  and the tail of every program's errors in `driver.log`.

[docs/en/comparison.md](docs/en/comparison.md) compares bootlab with simpleboot and Hyperion.

## Requirements

- A Linux x86-64 cluster with **Slurm ≥ 21.08** (`srun --exact`, `--overlap`) and an `srun` MPI plugin that
  the image's Open MPI 4.1 can use (pmix by default).
- **Apptainer ≥ 1.1** on the login and compute nodes.
- The **bootlab execution image** (`bootlab-exec`, revision r5 or later). It is not yet distributed with this
  repository; see [Known limitations](#known-limitations).
- A file system shared by the nodes of a job, for the run directories. It must support hard links.
- Node-local scratch disk recommended (the probe measures whether it is faster than the shared disk).
- Optional: cgroup v2 as Slurm sets it up (for the `pmp2sdp` memory watchdog; without it the watchdog is off
  and `driver.log` says so).
- One solve on several nodes needs InfiniBand; with Ethernet TCP only, bootlab stays on one node by default
  ([interfaces](docs/en/interfaces.md#multi-node-solves)).

Or one Linux workstation without Slurm, with Apptainer and the image ([local mode](#a-workstation-without-slurm-local-mode)).

No Python is needed on the host.

## Quick start

```bash
git clone <this repository> bootlab && cd bootlab
module load apptainer                     # or however your cluster puts apptainer on PATH
./install.sh --image /path/to/bootlab-exec-r5.sif --sha256 <the image's sha256>

# what the probe cannot know (your Slurm partition, time limit); it may stay empty: then the cluster's default
# partition and a 48-hour limit are used
touch mysite.hand.toml

# probe the cluster (one small job per node type: 8 cores, 16 GB, 15 minutes), merge, generate the site
bin/bootlab probe plan --sif /path/to/bootlab-exec-r5.sif --out probe --submit
bin/bootlab probe merge probe              # after the probe jobs have finished
bin/bootlab site gen --probe probe/probe.json --hand mysite.hand.toml --out sites/mysite
echo 'site = "mysite"' >> bootlab.toml

bin/bootlab regress                        # Ising Lambda 11, three steps, against the reference: PASS / FAIL
bin/bootlab submit examples/ising/run.toml # a run
```

The [tutorial](docs/en/tutorial.md) goes through every step with its expected output.

### A workstation without Slurm (local mode)

bootlab also runs on one Linux machine without Slurm: the image's programs start with `apptainer exec`, the
MPI ones through the image's own `mpirun`, one rank a physical core (never a core's second hardware thread),
and the memory is what the machine has available when the run starts.

Prerequisites:
- Apptainer able to run without root: user namespaces enabled (most current distributions), or a setuid
  installation.
- `apptainer` on PATH. A workstation usually has no module system and needs nothing else; a cluster's
  `module load apptainer` is not part of local mode.
- No Slurm on the host: the generator asks for its node list through the image's own `scontrol` stand-in,
  which answers with this machine's name.
- The bootlab image on a local path.
- Only one machine: no solve on several machines.
- No Python needed on the host, as on a cluster.

```bash
./install.sh --image /path/to/bootlab-exec-r5.sif --sha256 <the image's sha256>
bin/bootlab probe local --sif /path/to/bootlab-exec-r5.sif --out probe   # here, a few minutes: cores, memory,
                                                                         # disks, mpirun on every core
touch ws.hand.toml
bin/bootlab site gen --probe probe/probe.json --hand ws.hand.toml --out sites/ws
echo 'site = "ws"' >> bootlab.toml

bin/bootlab regress --wait                 # here, in the foreground: PASS / FAIL
bin/bootlab submit examples/ising/run.toml # in the background (setsid nohup); the log under runs/
bin/bootlab stop runs/<run>                # the solver writes its checkpoint; later:
bin/bootlab submit --resume runs/<run>
```

How bootlab chooses:
- Without Slurm (no `SLURM_JOB_ID`, no `sbatch` on PATH) bootlab works locally. `launcher = "local"` or
  `"slurm"` in the site or in bootlab.toml says it outright. `driver.log` records which one and why.
- One local run at a time on a machine (a lock in the runs directory).
- The `pmp2sdp` memory watchdog reads each instance's processes (`smaps_rollup`: Pss_Anon + Pss_Shmem), or,
  with `memory_guard = "systemd"` (the probe sets it when it works), a systemd user scope with a memory limit.

`submit --systemd` starts the driver as a transient systemd user unit. Background runs and logging out:
- Some systems end every process of a session at logout (`KillUserProcesses=yes` in logind.conf), also those
  started with nohup.
- A user unit outlives the login only with `loginctl enable-linger`.

## Documentation

| | English | 中文 |
|---|---|---|
| User tutorial: install, probe, site, run, resume, results, regression | [docs/en/tutorial.md](docs/en/tutorial.md) | [docs/zh/tutorial.md](docs/zh/tutorial.md) |
| Interfaces for developers: every layer, file format and configuration key | [docs/en/interfaces.md](docs/en/interfaces.md) | [docs/zh/interfaces.md](docs/zh/interfaces.md) |
| bootlab compared with simpleboot and Hyperion | [docs/en/comparison.md](docs/en/comparison.md) | [docs/zh/comparison.md](docs/zh/comparison.md) |
| Writing a new model with an AI assistant | [docs/en/writing-models-with-ai.md](docs/en/writing-models-with-ai.md) | [docs/zh/writing-models-with-ai.md](docs/zh/writing-models-with-ai.md) |

## Roadmap

v1 is Skydive only, on purpose: the three layers were built and checked against a reference with one
algorithm before anything else was added. Each later item adds to the same skeleton without changing the two
fixed interfaces. The order is not fixed.

- **Navigator** (minimising the navigator function with `sdpb` and `approx_objective`).
- **Feasibility scans and bisection**, as Slurm job arrays of independent points.
- **Delaunay region search, OPE scans, extremal functional (EFM) spectrum extraction.**
- **The rest of `bootlab bench`**: it measures the solver's processes on one node now; block tables,
  `pmp2sdp` instances and several nodes are still to come.
- **Spinning external operators** (current and stress-tensor bootstrap), with a second PMP generator that
  writes the same PMP format.

## Known limitations

- The execution image is not public yet. Without it bootlab runs only its unit tests.
- Only d = 3 and external scalars (the generator's `bt-model/2` format and scalar_blocks).
- A converter from simpleboot's AutoEqu/VBlock files to `bt-model/2` exists in the authors' toolchain and is
  not part of this release; models are written by hand or with an AI assistant
  ([guide](docs/en/writing-models-with-ai.md)).
- Developed and tested on one cluster (four node types, InfiniBand, NFS over IPoIB). The probe and the
  generated sites are meant for other clusters but have not been run on another one yet.
- Local mode was accepted on a cluster's compute node with Slurm hidden, not on a machine that really has no
  Slurm. Two of its options are untested beyond their unit tests: `memory_guard = "systemd"` (the pmp2sdp
  watchdog through a systemd user scope) and `submit --systemd` (the driver as a systemd user unit).

## Acknowledgements

bootlab is a new code base and contains no source code from the projects below. Its design borrows their
ideas, and its execution image runs their programs under their own (MIT) licenses.

- **simpleboot** by Ning Su ([gitlab.com/bootstrapcollaboration/simpleboot](https://gitlab.com/bootstrapcollaboration/simpleboot)),
  a Mathematica framework for bootstrap calculations. bootlab follows simpleboot's Skydive driver semantics
  and option meanings, its VBlock conventions for crossing equations (the `bt-model/2` format), its Σ
  navigator normalization and its block-truncation presets.
- **Hyperion** by David Simmons-Duffin ([github.com/davidsd/hyperion](https://github.com/davidsd/hyperion)),
  a Haskell framework for concurrent computations on HPC clusters, and the bootstrap libraries built on it:
  **scalars-3d** ([gitlab.com/davidsd/scalars-3d](https://gitlab.com/davidsd/scalars-3d)),
  [hyperion-bootstrap](https://gitlab.com/davidsd/hyperion-bootstrap),
  [bootstrap-bounds](https://gitlab.com/davidsd/bootstrap-bounds) and
  [sdpb-haskell](https://gitlab.com/davidsd/sdpb-haskell). The PMP generator in the image is scalars-3d's
  code with a small entry point that writes the PMP of a batch of points; the numerical code is upstream's.
- **SDPB** by David Simmons-Duffin, Walter Landry and Vasiliy Dommes
  ([github.com/davidsd/sdpb](https://github.com/davidsd/sdpb)); `pmp2sdp` and `dynamical_sdp` are built from
  its `skydiving-master` branch.
- **scalar_blocks** by Walter Landry, David Simmons-Duffin and Petr Kravchuk
  ([gitlab.com/bootstrapcollaboration/scalar_blocks](https://gitlab.com/bootstrapcollaboration/scalar_blocks)),
  which computes the conformal blocks inside the generator.

If you use bootlab in published work, please cite the methods and programs it runs:

- D. Simmons-Duffin, *A Semidefinite Program Solver for the Conformal Bootstrap*, JHEP 06 (2015) 174,
  [arXiv:1502.02033](https://arxiv.org/abs/1502.02033).
- W. Landry and D. Simmons-Duffin, *Scaling the semidefinite program solver SDPB*,
  [arXiv:1909.09745](https://arxiv.org/abs/1909.09745).
- M. Reehorst, S. Rychkov, D. Simmons-Duffin, B. Sirois, N. Su and B. van Rees, *Navigator Function for the
  Conformal Bootstrap*, SciPost Phys. 11 (2021) 072, [arXiv:2104.09518](https://arxiv.org/abs/2104.09518).
- A. Liu, D. Simmons-Duffin, N. Su and B. C. van Rees, *Skydiving to Bootstrap Islands*, JHEP 11 (2025) 003,
  [arXiv:2307.13046](https://arxiv.org/abs/2307.13046).

## Contributors

- [dantengs](https://github.com/dantengs): design, physics, testing on the cluster
- [Claude Code](https://claude.com/claude-code) (Anthropic): implementation and documentation, working with
  dantengs

## License

MIT, see [LICENSE](LICENSE). Copyright (c) 2026 dantengs.

The programs in the execution image (SDPB, scalar_blocks, Hyperion and the libraries on it) keep their own
MIT licenses and copyright notices.
