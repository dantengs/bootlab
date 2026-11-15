"""Built-in models: bounds whose physics is Haskell code in a generator, named by a model card (docs/en/interfaces.md, section 19).

The first such generator is the current bootstrap's: DSD's stress-tensors-3d with bootlab's `pmp` entry
(gen/current). A card names the generator and its bound ("JJJJ3d"), the bound's channels and objectives;
the spectrum is a bootlab-spectrum/1 file as for bt-model/2 models, the objective a bootlab-objective/1 file
with the card's objective kinds. Spectrum and objective values may refer to point parameters, worked out
exactly at each point. Everything is checked here, once.
"""
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from . import execute, inputs, pmp, skydive
from .numbers import Expr, Num, decimal_text, dumps, frac

CARD = "bootlab-builtin/1"
NUMERICS = ["lambda", "order", "num_poles", "max_spin", "block_precision"]
OBJECTIVE_KEYS = ["format", "kind", "lambda", "direction"]
DIRECTIONS = ["upper", "lower"]


def check_objective(objective, card) -> tuple:
    """The objective's errors and the parameter names its values use. Kinds: feasibility (lambda optional),
    ope_bound (lambda and direction); lambda has the card's length."""
    errors = inputs._unknown_keys("objective", objective, OBJECTIVE_KEYS)
    names = []
    kind = objective.get("kind")
    if objective.get("format") != "bootlab-objective/1":
        errors.append("objective: format must be bootlab-objective/1")
    if kind not in card["objectives"]:
        errors.append(f"objective: kind {kind!r} is not among this model's objectives {card['objectives']}")
        return errors, names
    lam = objective.get("lambda")
    if kind == "ope_bound" and lam is None:
        errors.append("objective: ope_bound needs lambda, the direction of the external OPE coefficients")
    if lam is not None:
        if not isinstance(lam, list) or len(lam) != card["lambda"]:
            errors.append(f"objective: lambda needs {card['lambda']} entries, not {lam!r}")
        else:
            for i, v in enumerate(lam):
                e = inputs._value("objective", f"lambda[{i}]", v, errors)
                if e is not None:
                    names.extend(n for n in e.names if n not in names)
    d = objective.get("direction")
    if kind == "ope_bound" and d not in DIRECTIONS:
        errors.append(f"objective: ope_bound needs direction {' or '.join(DIRECTIONS)}, not {d!r}")
    if kind == "feasibility" and d is not None:
        errors.append("objective: feasibility has no direction")
    return errors, names


def check_points(run, coords) -> list:
    """[scan] points: a list of points, each one value per coordinate, as exact strings or integers."""
    pts = run.get("scan", {}).get("points")
    if not isinstance(pts, list) or not pts:
        return ["run.toml [scan] points needs a list of points, each a list with one value per coordinate"]
    errors = []
    for i, p in enumerate(pts):
        if not isinstance(p, list) or len(p) != len(coords):
            errors.append(f"run.toml [scan] points[{i}] needs {len(coords)} values, one for each of {coords}, not {p!r}")
        elif not all(inputs._exact(v) and not isinstance(v, float) for v in p):
            errors.append(f"run.toml [scan] points[{i}]: {p!r} is not all exact numbers; write each as a string "
                          "with the digits you mean (a TOML float has already lost them)")
    return errors


def check_skydive(run, objective) -> list:
    """[skydive] of a built-in model: the solver minimizes the SDP's objective over the coordinates, so the
    objective must have one (ope_bound; a feasibility SDP has none). That N is the bound itself, not a navigator
    whose sign says allowed or excluded: no find_boundary_direction, no stop_on_negative. The solver's precision
    and options are the bound's (the generator's sdpb.args.json); with mixed_precision the bound is built at the
    call's precision."""
    o = run["skydive"]
    errors = []
    if objective.get("kind") != "ope_bound":
        errors.append(f"objective: a built-in model's Skydive minimizes the SDP's objective; kind {objective.get('kind')!r} "
                      "has none (use ope_bound)")
    if o.get("find_boundary_direction"):
        errors.append("run.toml [skydive] find_boundary_direction: with an ope_bound objective N is the bound, it has "
                      "no boundary at N = 0")
    if o.get("stop_on_negative"):
        errors.append("run.toml [skydive] stop_on_negative: with an ope_bound objective the sign of N is the bound's")
    t = run.get("solver", {}).get("dualityGapThreshold")
    if t is not None and "gap_upper_limit" in o and inputs._exact(t) and 0 < frac(o["gap_upper_limit"]) <= frac(t):
        errors.append(f"run.toml [skydive] gap_upper_limit {o['gap_upper_limit']} is not above [solver] "
                      f"dualityGapThreshold {t}: dynamical_sdp would stop as primal-dual optimal before taking a step")
    return errors


def solver_options(args_file: Path) -> dict:
    """The bound's sdpb options (the generator's sdpb.args.json, sdpb's command line) by flag name: a value, or
    True for a switch. A Skydive passes them to dynamical_sdp, as upstream's runSkydive passes the bound's
    solverParams (SDPB.getParamArgs)."""
    return solver_options_from(json.loads(args_file.read_text()))


def solver_options_from(argv: list) -> dict:
    out, i = {}, 0
    while i < len(argv):
        has_value = i + 1 < len(argv) and not argv[i + 1].startswith("--")
        out[argv[i][2:]] = argv[i + 1] if has_value else True
        i += 2 if has_value else 1
    return out


def load(run_toml: Path, run: dict, card: dict, spectrum: dict, objective: dict) -> inputs.Problem:
    errors = []
    if spectrum.get("format") != "bootlab-spectrum/1":
        errors.append("spectrum: format must be bootlab-spectrum/1")
    n = run["numerics"]
    missing = [k for k in NUMERICS if k not in n]
    extra = [k for k in n if k not in NUMERICS]
    if missing or extra:
        errors.append(f"numerics: a built-in model takes exactly {', '.join(NUMERICS)}"
                      + (f"; missing {', '.join(missing)}" if missing else "")
                      + (f"; unknown {', '.join(extra)}" if extra else ""))
        raise SystemExit("the inputs have errors:\n  " + "\n  ".join(errors))
    if n["lambda"] % 2 == 0 or n["lambda"] < 3:
        errors.append(f"numerics.lambda {n['lambda']} must be odd and at least 3")
    spins = list(range(0, n["max_spin"] + 1))
    channels = {c["name"]: c["spins"] for c in card["channels"]}
    errs, warnings, gap_spins, names = inputs.check_spectrum(spectrum, channels, spins)
    errors += errs
    errs, obj_names = check_objective(objective, card)
    errors += errs
    model_coords = list(card["coordinates"])
    coords = model_coords + [m for m in dict.fromkeys(names + obj_names) if m not in model_coords]
    given = run["coords"]["names"]
    if given != coords:
        errors.append(f"run.toml coords.names is {given}; this model, spectrum and objective give {coords}")
    errors += inputs.check_run_numbers(run, coords)
    if ("scan" in run) == ("skydive" in run):
        errors.append("run.toml: a built-in model has either [scan] (one solve per point) or [skydive], one of them")
    elif "scan" in run:
        errors += check_points(run, coords)
    else:
        errors += check_skydive(run, objective)
    if errors:
        raise SystemExit("the inputs have errors:\n  " + "\n  ".join(errors))
    return inputs.Problem(run_toml, run, card, spectrum, objective, model_coords, coords, spins, gap_spins, warnings)


def spectrum_at(problem, point: dict) -> dict:
    """bootlab's spectrum at a point, each gap with its resolved spins (the generator does not resolve them)."""
    s = inputs.resolve_spectrum(problem, point)
    for g, ls in zip(s["gaps"], problem.gap_spins):
        g["spins"] = ls
    return s


def objective_at(problem, point: dict) -> dict:
    env = {k: frac(v) for k, v in point.items()}
    o = problem.objective
    out = {"kind": o["kind"]}
    if "lambda" in o:
        out["lambda"] = [Num(decimal_text(Expr(v).value(env))) for v in o["lambda"]]
    if "direction" in o:
        out["direction"] = o["direction"]
    return out


def write_request(problem, points: dict, out: Path, blocks: Path, cores: int, bits: int = None) -> Path:
    """points: name -> {coordinate: decimal string}, in problem.coords. bits: a mixed precision call's precision,
    below block_precision: the bound is built at it (SimpleBoundParams' precision is both the blocks' and the
    solver's), so the generator works at the call's precision too."""
    n = problem.numerics
    prec = min(bits, n["block_precision"]) if bits else n["block_precision"]
    req = {
        "bound": problem.model["bound"],
        "numerics": {"nmax": (n["lambda"] + 1) // 2, "order": n["order"], "numPoles": n["num_poles"],
                     "maxSpin": n["max_spin"], "precision": prec},
        "points": [{"name": name,
                    "coords": [Num(p[c]) for c in problem.model_coords],
                    "spectrum": spectrum_at(problem, p),
                    "objective": objective_at(problem, p)}
                   for name, p in points.items()],
        "out": str(out),
        "blocks": str(blocks),
        "cores": cores,
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / "request.json"
    path.write_text(dumps(req) + "\n")
    return path


def generator_env(gen: dict, mem_mb: int, local_dir: Path) -> dict:
    """The generator's environment: the site's (its site scripts, libraries), the memory its scheduler may plan
    with (this step's --mem), and no node-local paths, so the scheduler keeps the block tables for the next
    points and the parts for pmp2sdp instead of deleting them (Hyperion's site configuration).
    Hyperion's data directory (HYP_DATA_DIR) is the site's, else one under local_dir: the pmp path writes nothing
    there (the bound's files go to the request's out), but it must be a writable place, which /opt in the image
    is not."""
    own = {k: str(v).replace("{root}", str(execute.REPO)) for k, v in gen.get("env", {}).items()}   # {root}: install root
    return {**os.environ, "HYP_DATA_DIR": str(local_dir / "hyp-data"), **own, "CUR_MEM_MB": str(mem_mb),
            "CUR_LOCAL_PATHS": "", "CUR_LOCAL_DIR": str(local_dir)}


def generate(site, gen: dict, node, request: Path, local_dir: Path, log: Path, stall_s: float = None) -> float:
    """Run the generator on the whole node: in the site's image when the site says container = true (a generator
    carried by the image, whose program sets its own libraries), else on the host. stall_s: as pmp.generate's.
    Returns its wall time."""
    t = time.time()
    mem = node.mem_mb - site.driver_reserve_mb
    cmd = execute.srun_argv(site, [gen["program"], "pmp", request], 1, node.cores, mem, mpi=False,
                            container=gen.get("container", False), hosts=node.solo())
    proc = execute.start(cmd, log, env=generator_env(gen, mem, local_dir))
    code = proc.wait() if stall_s is None else \
        execute.wait_watched(proc, pmp.watched(request, log), stall_s, "generator", log)
    if code != 0:
        raise execute.StepFailed(f"generator exit {code}")
    return time.time() - t


class Skydive(skydive.Run):
    """A built-in model's Skydive: bootlab's loop (skydive.Run, unchanged) with the card's generator for the PMPs,
    the pmp2sdp and dynamical_sdp it names (dynamical_sdp: the site's unless it names one), the bound's own solver
    options under run.toml [solver] (the centre's sdpb.args.json: upstream's Skydive passes the bound's
    solverParams to dynamical_sdp, Skydive/Bound.hs runSkydiveBound), and an ope_bound's ending at optimal: the
    bound's optimum over the coordinates (check_skydive).

    The generator writes every step into one directory of the run, pmp_dir, as upstream's Skydive keeps its json
    directory from step to step (runSkydiveBound deletes only the SDPs): the scheduler skips the parts that exist,
    so a step builds only the parts that depend on x (<JJJJ>: 2 of 64, 1.4 s instead of 20.5 s, the parts byte for
    byte the same). Each step takes its points' pmp.nsv from there. pmp_dir is in the scratch: a resume
    on another node or in another job starts without it and the generator builds everything once more."""

    def __init__(self, problem, site, node, run_dir: Path, scratch: Path = None):
        super().__init__(problem, site, node, run_dir, scratch)
        self.pmp_dir = self.scratch / "pmp"
        self.gen = site.generators[problem.model["generator"]]
        self.pmp2sdp = self.gen["pmp2sdp"]
        self.dynamical_sdp = self.gen.get("dynamical_sdp", site.dynamical_sdp)
        self.bound_solver = None    # the bound's options: the same at every point (make_pmps checks)
        self.ahead = None           # (step, x, bits, its PMPs or the generator's failure), made by threshold_setting

    def make_pmps(self, n: int, x: dict, bits: int):
        if self.ahead and self.ahead[:3] == (n, x, bits):
            made, self.ahead = self.ahead[3], None
            if isinstance(made, execute.StepFailed):
                raise made
            return made
        sdir = self.scratch / f"step_{n}"
        logs = self.dir / "steps" / str(n)
        pts = self.points(x)
        req = write_request(self.p, pts, self.pmp_dir, self.blocks, self.node.cores, bits)
        self.timed_generator(n, lambda stall: generate(self.site, self.gen, self.node, req, self.scratch / "local",
                                                       logs / "generator", stall_s=stall))
        lists = pmp.pmp_lists(self.pmp_dir, pts)
        args = self.pmp_dir / "c" / "sdpb.args.json"
        shutil.copy(args, logs / "sdpb.args.json")
        opts = solver_options(args)
        if int(opts["precision"]) != bits:
            raise execute.StepFailed(f"generator: the bound's solver precision {opts['precision']} is not "
                                     f"{bits}, the call's")
        same = {**opts, "precision": None} == {**(self.bound_solver or opts), "precision": None}
        if not same:            # the bound's options but the precision are the same at every point and step
            raise execute.StepFailed(f"generator: the bound's solver options at step {n} are not those of the "
                                     f"run's first point: {opts} against {self.bound_solver}")
        self.bound_solver = opts
        return sdir, logs, lists

    def solver_base(self) -> dict:
        return self.bound_solver

    def threshold_setting(self, n: int, x: dict, bits: int) -> str:
        """[solver] dualityGapThreshold, else the bound's, which only the generator gives: step n + 1's PMPs are
        made here, ahead, and its make_pmps returns them (a generator failure is raised there, inside the loop's
        handling of a failed step; the threshold returned then is never used)."""
        t = time.time()
        try:
            made = self.make_pmps(n + 1, x, bits)
        except execute.StepFailed as e:
            made = e
        self.ahead = (n + 1, dict(x), bits, made)
        self.pmp_ahead_s = time.time() - t        # in step n + 1's record (its seconds.pmp)
        if isinstance(made, execute.StepFailed):
            return skydive.GAP_FLOOR
        return {**self.bound_solver, **self.p.run.get("solver", {})}["dualityGapThreshold"]

    def optimal(self, dive, boundary: bool) -> tuple:
        return "converged", "optimum"

    def versions(self) -> dict:
        """The image's sha256, and the generator's: in the image (container = true), or its own sha256 on the host."""
        v = {**super().versions(), "generator": self.gen["program"]}
        if self.gen.get("container"):
            v["generator_in"] = "image"
        else:
            with open(self.gen["program"], "rb") as f:
                v["generator_sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
        return v
