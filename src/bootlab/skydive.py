"""The Skydive loop, its records and its endings (docs/en/interfaces.md sections 10 and 12).

Each step: the generator writes the PMP of x and of its neighbours x + eps e_i; pmp2sdp builds their SDPs
side by side; one dynamical_sdp call; one line in solves.jsonl. The decisions follow the reference program's (bt r3.8)
Skydive loop: the gap upper limit only for the first
solver call, the boundary direction from the first negative navigator value on (or from the start), and
the endings below. Nothing is retried.
"""
import hashlib
import json
import shutil
import time
from fractions import Fraction
from pathlib import Path

from . import bench, execute, inputs, pmp
from .numbers import decimal_text, frac

# The options of dynamical_sdp that bt r3.8 passes when the parameter file leaves them out: upstream's
# defaultDynamicalConfig (dynamical-sdp, Projects.SkydivingRun). run.toml [dynamical] overrides them.
DYNAMICAL_DEFAULTS = {
    "centeringRThreshold": "1e-20",
    "betaScanMin": "0.1",
    "betaScanMax": "1.01",
    "betaScanStep": "0.1",
    "stepMinThreshold": "0.1",
    "stepMaxThreshold": "0.6",
    "primalDualObjWeight": "0.2",
    "maxClimbingSteps": 1,
    "betaClimbing": "2",
    "navigatorWithLogDetX": False,
    "gradientWithLogDetX": True,
    "finiteDualityGapTarget": "0",
    "BFGSPartialUpdate": -1,
}


# SDPB's options as bt r3.8 passes them to dynamical_sdp (sdpb-haskell getParamArgs passes every one) when
# the parameter file leaves them out. bt's Skydive bound has Hyperion's optimizationParams at its nmax
# (bootstrap-params sdpbParamsNmax over sdpb-haskell defaultParams), with bt's Skydive solver defaults on
# top (SkydiveBT isingSkydiver: dualityGap 1e-6, errors 1e-12, feasible centering 0.3, for every model) and
# writeSolution y,x. The precision is the problem's (Problem.solver_precision, bt's solverPrecisionDefault);
# maxRuntime is set per step from the job's time left (docs/en/interfaces.md section 8), where bt passes 20 h at
# nmax <= 6, 4 h up to 14 and 8 h above. run.toml [solver] overrides any of them. Checked against the
# parameters dynamical_sdp printed in a reference run (ising-oe, Lambda 11, nmax 6).
SOLVER_COMMON = {
    "dualityGapThreshold": "1e-6",
    "primalErrorThreshold": "1e-12",
    "dualErrorThreshold": "1e-12",
    "feasibleCenteringParameter": "0.3",
    "infeasibleCenteringParameter": "0.3",
    "stepLengthReduction": "0.7",
    "maxIterations": 1000,
    "verbosity": 1,
    "procGranularity": 1,
    "writeSolution": "y,x",
}
# By nmax, upper bounds inclusive: checkpointInterval, initialMatrixScalePrimal = initialMatrixScaleDual,
# maxComplementarity. (sdpbParamsNmax's error and gap thresholds are all replaced by the Skydive defaults.)
SOLVER_BY_NMAX = [
    (6, 3600, "1e20", "1e100"),
    (10, 1800, "1e40", "1e100"),
    (14, 1800, "1e50", "1e130"),
    (18, 1800, "1e50", "1e160"),
    (20, 1800, "1e60", "1e180"),
    (None, 1800, "1e60", "1e200"),
]


# The stopping threshold tightens with |N| (docs/en/interfaces.md section 10): max(floor, min(setting, rho |N_prev|)).
GAP_RHO = Fraction(1, 1000)
GAP_FLOOR = "1e-30"


def gap_threshold(setting: str, n_prev, tighten: bool = True) -> str:
    """dualityGapThreshold for a step: the setting at the first solver call (n_prev None), else
    max(1e-30, min(setting, 1e-3 |N_prev|)). The setting and the floor are passed as written.
    [skydive].gap_threshold_tighten = false (tighten False) keeps the setting at every step: a find-boundary run
    drives N to 0, so 1e-3 |N_prev| stays about 1000 times below the gap and no call ends optimal."""
    if n_prev is None or not tighten:
        return str(setting)
    tight = GAP_RHO * abs(frac(n_prev))
    if tight >= frac(setting):
        return str(setting)
    if tight <= frac(GAP_FLOOR):
        return GAP_FLOOR
    return f"{float(tight):.6e}"


# The gap upper limit following N (docs/en/interfaces.md section 10): on unless [skydive].gap_upper_limit_track = "off"
# or gap_upper_limit_first_only = true (Ising Lambda 19 converged in 84 steps with it).
TRACK_FLOOR = 1000 * Fraction(GAP_FLOOR)
TRACK_KAPPA = "0.1"                # [skydive].gap_upper_limit_track_floor when left out


def tracked_gap_limit(setting: str, c, kappa, n_prev, n_first) -> str:
    """The gap upper limit of a step after the solver's first call: min(setting, max(c |N_prev|, kappa |N_1|,
    1000 floor)), N_1 the N of the run's step 1; the setting (as written) with no step behind, as at the first
    call. The kappa term keeps the limit at the problem's own scale where N crosses zero (with the
    limit at |N| alone the whole run shrank with N and approached zero from above)."""
    if n_prev is None:
        return str(setting)
    limit = max(frac(c) * abs(frac(n_prev)), frac(kappa) * abs(frac(n_first)), TRACK_FLOOR)
    if limit >= frac(setting):
        return str(setting)
    return f"{float(limit):.6e}"


def optimal_ending(primal: str, dual: str, boundary: bool = False) -> tuple:
    """(status, reason) of an optimal dive: N lies in [dual, primal], so its sign is decided when the two
    objectives have the same sign (docs/en/interfaces.md section 10, step 4). With opposite signs |primal|, |dual| and
    |N| are all below the gap (here the gap is |primal - dual|: in one run 5.806e-21 + 0.708e-21 =
    6.514e-21): a run finding the boundary (boundary on at this step) has its point on the boundary to the
    solver's resolution, converged; one minimizing N has not decided N's sign, stopped."""
    p, d = frac(primal), frac(dual)
    if p < 0 and d < 0:
        return "converged", "allowed"
    if p > 0 and d > 0:
        return "converged", "excluded"
    if boundary:
        return "converged", "boundary"
    return "stopped", "stopped: N undecided at optimal"


def solver_defaults(nmax: int) -> dict:
    ck, scale, comp = next((c, s, m) for top, c, s, m in SOLVER_BY_NMAX if top is None or nmax <= top)
    return {**SOLVER_COMMON, "checkpointInterval": ck, "initialMatrixScalePrimal": scale,
            "initialMatrixScaleDual": scale, "maxComplementarity": comp}


class Run:
    def __init__(self, problem, site, node, run_dir: Path, scratch: Path = None):
        """scratch: the run's directory as execute.choose_scratch chose it; left out, under the site's one
        scratch path (tests, and sites with one)."""
        self.p, self.site, self.node, self.dir = problem, site, node, run_dir
        self.opts = problem.run["skydive"]
        self.scratch = scratch or Path(site.scratch) / run_dir.name
        self.ck = run_dir / "checkpoint"   # the solver's checkpoint directory, on the runs file system (SPEC 7.3)
        self.blocks = self.scratch / "blocks"
        self.scratch_info = None    # result.json: where the run's large files were (cli)
        self.model_file = run_dir / "model.json"
        self.started = execute.now()
        self.sdp_mem = None         # pmp2sdp's measured memory and k (build_sdps); a resume sets it
        self.sif_sha256 = None      # the image's, worked out once at the start (cli.note_image)

    # ------------------------------------------------------------ records

    def log(self, msg: str):
        line = f"{execute.now()} {msg}"
        print(line, flush=True)
        with open(self.dir / "driver.log", "a") as f:
            f.write(line + "\n")

    def where(self) -> dict:
        """A record's node: node 0's name; with more than one node also all of them, node 0 first (procs is
        then the processes on each)."""
        return {"node": self.node.host, "nodes": self.node.hosts} if self.node.multi else {"node": self.node.host}

    def records(self) -> int:
        f = self.dir / "solves.jsonl"
        return sum(1 for l in f.read_text().splitlines() if l.strip()) if f.exists() else 0

    def record(self, line: dict):
        with open(self.dir / "solves.jsonl", "a") as f:
            f.write(json.dumps(line) + "\n")
        execute.prune_kept(self.ck, self.records())

    def inputs_sha(self) -> dict:
        out = {name: hashlib.sha256((self.dir / name).read_bytes()).hexdigest()
               for name in ("model.json", "spectrum.toml", "objective.toml")}
        out["numerics"] = hashlib.sha256(json.dumps(self.p.numerics, sort_keys=True).encode()).hexdigest()
        return out

    def finish(self, status: str, reason: str, x: dict, steps: int, navigator=None, dive=None, n_bound=None):
        """dive: the step whose N (or, at optimal, whose primal, dual and gap) the ending reports; n_bound: on
        the boundary, the bound |N| <= gap (navigator is then "0")."""
        result = {"status": status, "reason": reason, "x": [x[c] for c in self.p.coords], "coords": self.p.coords,
                  "navigator": navigator, "primal": dive.primal if dive else None, "dual": dive.dual if dive else None,
                  "gap": dive.gap if dive else None, "steps": steps, "inputs": self.inputs_sha(),
                  "versions": {"sif": self.site.sif, "sif_sha256": self.sif_sha256,
                               "generator": self.site.generator, "pmp2sdp": self.site.pmp2sdp,
                               "dynamical_sdp": self.site.dynamical_sdp},
                  "start": self.started, "end": execute.now()}
        if n_bound is not None:
            result["n_bound"] = n_bound
        if self.scratch_info:
            result["scratch"] = self.scratch_info
        (self.dir / "result.json").write_text(json.dumps(result, indent=1) + "\n")
        self.log(f"end: {ending(status, reason, n_bound)} at {result['x']} after {steps} steps")
        execute.clear(self.site, self.node, self.scratch)    # on every node; the checkpoint is in the run's

    # ------------------------------------------------------------ a step

    def points(self, x: dict) -> dict:
        eps = frac(self.opts.get("infinitesimal", inputs.INFINITESIMAL))
        pts = {"c": dict(x)}
        for i, c in enumerate(self.p.coords):
            shifted = dict(x)
            shifted[c] = decimal_text(frac(x[c]) + eps)
            pts[f"plus_{i}"] = shifted
        return pts

    def step(self, n: int, x: dict, state: dict, procs):
        """Step n (from 1). Returns (Dive, procs, seconds)."""
        sdir = self.scratch / f"step_{n}"
        logs = self.dir / "steps" / str(n)
        pts = self.points(x)
        seconds = {}
        t = time.time()
        req = pmp.write_request(self.p, self.model_file, pts, sdir / "pmp", self.blocks, self.node.cores)
        pmp.generate(self.site, self.node, req, logs / "generator")
        lists = pmp.pmp_lists(sdir / "pmp", pts)
        seconds["pmp"] = round(time.time() - t, 1)

        t = time.time()
        sdps = self.build_sdps(n, lists, sdir / "sdp", logs)
        seconds["sdp"] = round(time.time() - t, 1)
        if self.node.others:         # pmp2sdp ran on node 0; every node reads the SDPs from its own disk (a shared
                                     # scratch: from node 0's, nothing to copy)
            copy = execute.distribute(self.site, self.node, sdir / "sdp")
            seconds["copy"] = copy["seconds"]
            self.log(f"step {n}: SDPs to {', '.join(self.node.others)}: {copy['files']} files, "
                     f"{copy['bytes']} bytes, {copy['seconds']} s" +
                     (f"; sha256 of all {copy['sha256_same']} files the same on every node" if "sha256_same" in copy else ""))

        if procs is None:
            n_q, blocks = execute.sdp_size(sdps["c"])
            procs, why = bench.solver_procs(bench.load_bench(self.site.bench_dir, self.node.node_type),
                                            self.node.cores, n_q, blocks)
            self.log(f"solver processes {procs}: {why}")

        t = time.time()
        nmax = (self.p.numerics["lambda"] + 1) // 2
        solver = {"precision": self.p.solver_precision(), **solver_defaults(nmax),
                  "maxRuntime": execute.job_seconds_left() - 900, **self.p.run.get("solver", {}),
                  "dualityGapThreshold": state["gap_threshold"]}
        dynamical = {**DYNAMICAL_DEFAULTS, **self.p.run.get("dynamical", {}),
                     "externalParamInfinitestimal": self.opts.get("infinitesimal", inputs.INFINITESIMAL)}
        out = sdir / "out"
        code = execute.run_dynamical_sdp(self.site, self.node, procs, sdps["c"],
                                         [sdps[f"plus_{i}"] for i in range(len(self.p.coords))],
                                         out, self.ck, solver, dynamical, state, logs / "dynamical_sdp")
        seconds["dive"] = round(time.time() - t, 1)
        dive = execute.read_dive(out, code)
        for f in ("out.txt", "skydiving_out.txt"):
            if (out / f).exists():
                shutil.copy(out / f, logs / f)
        done = self.keep_generation(procs)
        if done:
            seconds["checkpoint"] = done["seconds"]
        return dive, procs, seconds, sdir, lists["c"]

    def keep_generation(self, procs):
        """The generation the call ended with, kept for its record (execute.keep_generation): a job killed
        later, while the solver saves again, resumes from it (docs/en/interfaces.md section 11)."""
        return execute.keep_generation(self.ck, self.records() + 1, {"nodes": self.node.all_hosts,
                                                                    "node_type": self.node.node_type, "ppn": procs})

    def build_sdps(self, n: int, lists: dict, sdp_root: Path, logs: Path) -> dict:
        """pmp2sdp for the step's PMPs, as many at a time as the node's memory holds (docs/en/interfaces.md section 9).

        The run's first step, or the first after a resume on another node type or with other processes,
        converts the central point's SDP alone and measures it: its held memory (execute.build_sdps) is the
        peak, and k = bench.pmp2sdp_at_once of it; the other points follow k at a time. Later steps convert k
        at a time without measuring. self.sdp_mem keeps the node type, the processes, the peak, k and the step
        it was measured at; solves.jsonl records it with each step and a resume takes it from there."""
        per = bench.pmp2sdp_procs(self.node.cores, len(lists))
        mem = self.node.mem_mb - self.site.driver_reserve_mb
        prec = self.p.solver_precision()
        m = self.sdp_mem
        if m and (m["node_type"], m["procs"]) == (self.node.node_type, per):
            at_once = min(m["k"], len(lists))
            self.log(f"step {n}: pmp2sdp {len(lists)} SDPs, {at_once} at a time with {per} processes each "
                     f"(k = {m['k']}: {m['peak_mb']} MB an instance, measured at step {m['measured_at_step']})")
            return execute.build_sdps(self.site, self.node, lists, sdp_root, prec, logs, at_once, per)[0]
        self.log(f"step {n}: pmp2sdp the central point's SDP alone with {per} processes, to measure its memory")
        sdps, (w,) = execute.build_sdps(self.site, self.node, {"c": lists["c"]}, sdp_root, prec, logs, 1, per)
        if w.get("watchdog") is False:
            self.log(f"step {n}: the steps' cgroups cannot be read here: no memory watchdog for pmp2sdp, and its "
                     f"memory is the node's MemTotal - MemAvailable rise")
        k = bench.pmp2sdp_at_once(mem, w["held_mb"])
        self.sdp_mem = {"node_type": self.node.node_type, "procs": per, "peak_mb": w["held_mb"], "k": k,
                        "measured_at_step": n, "cgroup_peak_mb": w["cgroup_peak_mb"], "node_mb": w["node_mb"]}
        self.log(f"step {n}: the SDP held {w['held_mb']} MB that --mem pressure cannot reclaim (memory.peak "
                 f"{w['cgroup_peak_mb']} MB, the node's MemTotal - MemAvailable +{w['node_mb']} MB); {mem} MB / "
                 f"({w['held_mb']} MB x {bench.SDP_MEMORY_MARGIN}) = k = {k}")
        if k < 1:
            raise execute.StepFailed(f"pmp2sdp: one instance holds {w['held_mb']} MB, more than {mem} MB over "
                                     f"the margin {bench.SDP_MEMORY_MARGIN}")
        rest = {name: nsv for name, nsv in lists.items() if name != "c"}
        if rest:
            at_once = min(k, len(rest))
            self.log(f"step {n}: pmp2sdp the other {len(rest)} SDPs, {at_once} at a time with {per} processes each")
            sdps.update(execute.build_sdps(self.site, self.node, rest, sdp_root, prec, logs, at_once, per)[0])
        return sdps

    # ------------------------------------------------------------ the loop

    def loop(self, x: dict, n: int, bfgs: dict, boundary_on: bool, procs):
        """x: the point of step n + 1; bfgs: total_iterations, prev_step, prev_gradient, prev_hessian,
        prev_navigator (the N of step n), and first_call when step n + 1 is the solver's first call (a new
        run, or a resume without a checkpoint)."""
        o = self.opts
        k = len(self.p.coords)
        threshold = {**solver_defaults((self.p.numerics["lambda"] + 1) // 2),
                     **self.p.run.get("solver", {})}["dualityGapThreshold"]
        box_min = [frac(v) for v in o["box_min"]]
        box_max = [frac(v) for v in o["box_max"]]
        direction = [str(v) for v in o.get("find_boundary_direction", [])]
        guess = o.get("hessian_guess", [])
        while True:
            if n >= o["max_steps"]:
                self.keep_checkpoint()
                return self.finish("unfinished", "unfinished: max steps", x, n)
            first = bfgs.get("first_call", False)
            if first:
                limit = str(o["gap_upper_limit"])
            elif inputs.track_c(o) is not None:
                limit = tracked_gap_limit(o["gap_upper_limit"], inputs.track_c(o),
                                          o.get("gap_upper_limit_track_floor", TRACK_KAPPA),
                                          bfgs.get("prev_navigator"), bfgs.get("first_navigator"))
            else:
                limit = "0" if o.get("gap_upper_limit_first_only", True) else str(o["gap_upper_limit"])
            state = {
                **bfgs,
                "gap_upper_limit": limit,
                "boundary_direction": direction if direction and (boundary_on or not o.get("find_boundary_when_allowed", True)) else None,
                "box_min": [str(v) for v in o["box_min"]], "box_max": [str(v) for v in o["box_max"]],
                "hessian_guess": [str(guess[i]) if i == j else "0" for i in range(k) for j in range(k)] if guess else None,
                "gap_threshold": gap_threshold(threshold, None if first else bfgs.get("prev_navigator"),
                                               o.get("gap_threshold_tighten", True)),
            }
            self.log(f"step {n + 1} at {[x[c] for c in self.p.coords]}: gap upper limit {state['gap_upper_limit']}, "
                     f"gap threshold {state['gap_threshold']}, boundary {'on' if state['boundary_direction'] else 'off'}")
            try:
                dive, procs, seconds, sdir, center_pmp = self.step(n + 1, x, state, procs)
            except execute.StepFailed as e:
                self.record({"step": n + 1, "x": [x[c] for c in self.p.coords], "verdict": "aborted",
                             "reason": f"failed: {e}", "procs": procs, **self.where(),
                             "pmp2sdp": self.sdp_mem})
                return self.finish("failed", f"failed: {e}", x, n)
            line = {"step": n + 1, "x": [x[c] for c in self.p.coords], "verdict": dive.verdict, "reason": dive.reason,
                    "navigator": dive.navigator, "primal": dive.primal, "dual": dive.dual, "gap": dive.gap,
                    "ext_step": dive.ext_step, "gradient": dive.gradient, "hessian": dive.hessian,
                    "iterations": dive.iterations, "alpha_p": dive.alpha_p, "alpha_d": dive.alpha_d, "beta": dive.beta,
                    "climbed": dive.climbed, "find_minimum": dive.find_minimum,
                    "gap_upper_limit": state["gap_upper_limit"], "gap_threshold": state["gap_threshold"],
                    "boundary_on": bool(state["boundary_direction"]),
                    "procs": procs, **self.where(), "seconds": seconds, "pmp2sdp": self.sdp_mem}
            self.record(line)
            if dive.verdict == "aborted":
                self.keep_checkpoint()
                return self.finish("unfinished", dive.reason, x, n)
            if dive.verdict == "optimal":
                status, reason = optimal_ending(dive.primal, dive.dual, bool(state["boundary_direction"]))
                if status == "converged":
                    self.keep_endpoint(center_pmp, sdir / "out")
                    # a converged run keeps its endpoint and no checkpoint (docs/en/interfaces.md section 12)
                    shutil.rmtree(self.dir / "checkpoint", ignore_errors=True)
                else:
                    self.keep_checkpoint()
                if reason == "boundary":        # N reported as 0, |N| <= the gap
                    return self.finish(status, reason, x, n + 1, "0", dive, n_bound=dive.gap)
                return self.finish(status, reason, x, n + 1, None, dive)
            x_next = {c: decimal_text(frac(x[c]) + frac(s)) for c, s in zip(self.p.coords, dive.ext_step)}
            negative = frac(dive.navigator) < 0
            if o.get("stop_on_negative", False) and negative:
                return self.finish("stopped", "stopped: negative navigator", x_next, n + 1, dive.navigator, dive)
            if not all(lo <= frac(x_next[c]) <= hi for c, lo, hi in zip(self.p.coords, box_min, box_max)):
                return self.finish("stopped", "stopped: left box", x, n + 1, dive.navigator, dive)
            boundary_on = boundary_on or bool(direction and negative)
            bfgs = {"total_iterations": dive.iterations, "prev_step": dive.ext_step,
                    "prev_gradient": dive.gradient, "prev_hessian": [h for row in dive.hessian for h in row],
                    "prev_navigator": dive.navigator, "first_navigator": bfgs.get("first_navigator", dive.navigator)}
            shutil.rmtree(sdir, ignore_errors=True)
            execute.on_hosts(self.site, self.node, f"rm -rf {sdir}", self.node.others)
            x, n = x_next, n + 1

    def keep_checkpoint(self):
        """The solver's checkpoint, for a resume to start hot (docs/en/interfaces.md section 11): the run's checkpoint/,
        with the generation of every call kept (keep_generation), so already there."""

    def keep_endpoint(self, center_nsv: Path, out_dir: Path):
        end = self.dir / "endpoint"
        parts_dir = end / "pmp" / "parts"
        parts_dir.mkdir(parents=True, exist_ok=True)
        kept = []
        for p in center_nsv.read_text().split("\0"):
            if p:
                shutil.copy(p, parts_dir / Path(p).name)
                kept.append(str(parts_dir / Path(p).name))
        (end / "pmp" / "pmp.nsv").write_text("".join(f"{p}\0" for p in kept))
        (end / "pmp" / "precision").write_text(f"{self.p.solver_precision()}\n")
        pmp.write_labels(end / "pmp")
        if self.node.others:          # x_<block>.txt are written where the blocks' ranks run
            execute.gather(self.site, self.node, out_dir)
        shutil.copytree(out_dir, end / "out", dirs_exist_ok=True)


def ending(status: str, reason: str, n_bound=None) -> str:
    """The ending in words: the reason alone when it already starts with the status ("unfinished: max steps");
    on the boundary with the bound on |N|."""
    if reason == "boundary":
        return f"{status}: on the boundary, |N| <= {n_bound}"
    return reason if reason.startswith(status) else f"{status}: {reason}"


def start_state(problem) -> dict:
    o = problem.run["skydive"]
    x = dict(zip(problem.coords, [str(v) for v in o["start"]]))
    return {"x": x, "n": 0, "bfgs": {"total_iterations": 0, "first_call": True}, "boundary_on": False, "procs": None}


def resume_state(problem, run_dir: Path, hot: bool) -> dict:
    """Where a run left off, from its last record: after a step, the next point with that step's BFGS
    state; after an aborted step, the same point with the state of the step before it. hot: the solver
    starts from the run's checkpoint/; without it the solver starts from scratch, so its first call is the
    first call and gets the gap upper limit even when that is for the first call only (docs/en/interfaces.md section 10)."""
    lines = [json.loads(l) for l in (run_dir / "solves.jsonl").read_text().splitlines() if l.strip()]
    last = lines[-1]
    stepped = [l for l in lines if l["verdict"] == "stepped"]
    o = problem.run["skydive"]
    direction = o.get("find_boundary_direction", [])
    # a failed step's line has no boundary_on (a resume after one once stopped at KeyError)
    boundary_on = any(l.get("boundary_on", False) for l in lines) or any(direction and frac(l["navigator"]) < 0 for l in stepped)
    if last["verdict"] == "stepped":
        x = {c: decimal_text(frac(v) + frac(s)) for c, v, s in zip(problem.coords, last["x"], last["ext_step"])}
        ref, n = last, last["step"]
    else:
        x = dict(zip(problem.coords, last["x"]))
        ref, n = (stepped[-1] if stepped else None), last["step"] - 1
    bfgs = {"total_iterations": 0, "first_call": not hot}
    if ref:
        bfgs.update({"total_iterations": ref["iterations"], "prev_step": ref["ext_step"],
                     "prev_gradient": ref["gradient"], "prev_hessian": [h for row in ref["hessian"] for h in row],
                     "prev_navigator": ref["navigator"],
                     "first_navigator": stepped[0]["navigator"]})    # N_1, for gap_upper_limit_track
    sdp_mem = next((l["pmp2sdp"] for l in reversed(lines) if l.get("pmp2sdp")), None)
    return {"x": x, "n": n, "bfgs": bfgs, "boundary_on": boundary_on, "procs": last.get("procs"),
            "pmp2sdp": sdp_mem}
