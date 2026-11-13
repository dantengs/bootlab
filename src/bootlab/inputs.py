"""The physics layer: the model (bt-model/2 JSON), the spectrum and the objective (TOML), and run.toml.

Everything is checked once, here. A spectrum value may refer to point parameters; resolve_spectrum works
it out exactly at a point and gives bt's spectrum section (docs/en/interfaces.md section 3).
"""
import json
import tomllib
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from .numbers import Expr, Num, decimal_text, frac

# simpleboot's HYC block setting (bt Projects.BT.Model.hycPreset): Lambda -> (kept pole order, spins).
HYC = {
    5: (9, list(range(0, 6))),
    7: (9, list(range(0, 21)) + [49, 52]),
    9: (9, list(range(0, 21)) + [49, 52]),
    11: (12, list(range(0, 21)) + [49, 52]),
    13: (12, list(range(0, 27)) + [49, 52]),
    15: (14, list(range(0, 27)) + [49, 50]),
    17: (14, list(range(0, 27)) + [49, 50]),
    19: (14, list(range(0, 27)) + [49, 50]),
    21: (18, list(range(0, 31)) + [38, 49, 52]),
    23: (18, list(range(0, 31)) + [38, 49, 52]),
    25: (20, list(range(0, 31)) + [38, 49, 52]),
    27: (20, list(range(0, 32)) + [33, 34, 37, 38, 41, 42, 45, 46, 49, 50]),
    29: (20, list(range(0, 36)) + [38, 41, 42, 45, 46, 49, 50, 51, 52, 55, 56]),
    31: (25, list(range(0, 41)) + [47, 48, 51, 52, 55, 56, 59, 60]),
    35: (30, list(range(0, 45)) + [47, 48, 51, 52, 55, 56, 59, 60, 63, 64, 67, 68]),
    43: (40, list(range(0, 65)) + [67, 68, 71, 72, 75, 76, 79, 80, 83, 84, 87, 88]),
}


def unitarity(spin: int) -> Fraction:
    """The d = 3 unitarity bound."""
    return Fraction(1, 2) if spin == 0 else Fraction(spin + 1)


RULE = {"even": "even spins", "odd": "odd spins", "all": "all spins"}
SPECTRUM_KEYS = ["format", "default_above_unitarity", "gap", "isolated"]
GAP_KEYS = ["channel", "spins", "delta", "above_unitarity"]
ISOLATED_KEYS = ["channel", "spin", "delta"]
BAD_SPINS = 'spins must be a list, a spin, "all", "a-b" or "a-b:step", not '


def allows(rule: str, spin: int) -> bool:
    return spin >= 0 and (rule == "all" or (rule == "even") == (spin % 2 == 0))


def read_spins(spec):
    """bt's reading of spins (Projects.BT.Spectrum.Spins): a list, one spin, "all", "a-b" or "a-b:step" with
    0 <= a <= b and step > 0. A ValueError says what is wrong."""
    if isinstance(spec, int) and not isinstance(spec, bool):
        return ("list", [spec])
    if isinstance(spec, list) and all(isinstance(l, int) and not isinstance(l, bool) for l in spec):
        return ("list", list(spec))
    if spec == "all":
        return ("all",)
    if isinstance(spec, str):
        rng, colon, step = spec.partition(":")
        ends = [x.strip() for x in rng.split("-")]
        nums = ends + ([step.strip()] if colon else [])
        if len(ends) == 2 and all(x.isdigit() for x in nums):
            a, b = int(ends[0]), int(ends[1])
            s = int(step) if colon else 1
            if s > 0 and a <= b:
                return ("range", a, b, s)
    raise ValueError(BAD_SPINS + repr(spec))


def resolve_spins(spins, ch, rule, problem_spins):
    """The spins of channel ch an entry sets, as bt resolves them: "all" is the run's spins that the channel
    allows, a range the allowed spins in it (also beyond the run's spins); a list must name allowed spins
    only. A ValueError says what is wrong."""
    if spins[0] == "all":
        ls = [l for l in problem_spins if allows(rule, l)]
    elif spins[0] == "range":
        _, a, b, s = spins
        ls = [l for l in range(a, b + 1, s) if allows(rule, l)]
    else:
        ls = spins[1]
        bad = [l for l in ls if not allows(rule, l)]
        if bad:
            raise ValueError(f"{ch} has {RULE[rule]}, not spin {', '.join(map(str, bad))}")
    if not ls:
        raise ValueError(f"no spin of {ch} ({RULE[rule]}) is in it")
    return ls


def rule_of(ch, channels):
    """The spin rule of channel ch; a ValueError names the model's channels."""
    if ch not in channels:
        raise ValueError(f"there is no channel {ch!r}; the channels of this model are "
                         + ", ".join(f"{c} ({RULE[r]})" for c, r in channels.items()))
    return channels[ch]


def _value(where, key, v, errors):
    """A spectrum value as an expression. A TOML float has already lost the decimal written, so it is an error."""
    if isinstance(v, bool) or not isinstance(v, (str, int)):
        errors.append(f"{where}: write {key} as a string, such as \"0.518149\" or \"deltaSig + 2\", not {v!r}; "
                      "a TOML float is not the exact decimal written")
        return None
    try:
        return Expr(v)
    except ValueError as e:
        errors.append(f"{where}: {key}: {e}")
        return None


def _unknown_keys(where, entry, allowed):
    bad = [k for k in entry if k not in allowed]
    return [f"{where}: unknown key {', '.join(bad)}; the keys are {', '.join(allowed)}"] if bad else []


def check_spectrum(spectrum, channels, problem_spins):
    """bt's checks of a spectrum (Projects.BT.Spectrum.buildSpectrum) that do not depend on the values: the
    errors, the warnings, the resolved spins of each gap and the parameter names the values use. The checks
    that do are spectrum_warnings and resolve_spectrum's."""
    errors = _unknown_keys("spectrum", spectrum, SPECTRUM_KEYS)
    warnings, gap_spins, names = [], [], []

    def expr(where, key, v):
        e = _value(where, key, v, errors)
        if e is not None:
            names.extend(n for n in e.names if n not in names)

    expr("spectrum", "default_above_unitarity", spectrum.get("default_above_unitarity", "0"))
    set_by = {}
    for i, g in enumerate(spectrum.get("gap", [])):
        at = f"spectrum gap[{i}]"
        errors += _unknown_keys(at, g, GAP_KEYS)
        values = [k for k in ("delta", "above_unitarity") if k in g]
        if len(values) != 1:
            errors.append(f"{at}: the gap for {g.get('channel')} needs exactly one of delta and above_unitarity")
        else:
            expr(at, values[0], g[values[0]])
        ls = []
        try:
            spins = read_spins(g["spins"])
            ls = resolve_spins(spins, g["channel"], rule_of(g["channel"], channels), problem_spins)
        except ValueError as e:
            errors.append(f"{at}: {e}")
        gap_spins.append(ls)
        for l in ls:
            set_by.setdefault((g["channel"], l), []).append(i)
        outside = [l for l in ls if l not in problem_spins] if ls and spins[0] == "list" else []
        if outside:
            warnings.append(f"{at}: spin {', '.join(map(str, outside))} is not among the spins of this run, "
                            "so the gap does nothing there")
    for (ch, l), idx in sorted(set_by.items(), key=lambda kv: kv[1]):
        if len(idx) > 1:
            errors.append(f"spectrum: the gap of {ch} spin {l} is set twice, in " + " and ".join(f"gap[{i}]" for i in idx))
    for i, o in enumerate(spectrum.get("isolated", [])):
        at = f"spectrum isolated[{i}]"
        errors += _unknown_keys(at, o, ISOLATED_KEYS)
        expr(at, "delta", o["delta"])
        try:
            resolve_spins(read_spins([o["spin"]]), o["channel"], rule_of(o["channel"], channels), problem_spins)
        except ValueError as e:
            errors.append(f"{at}: {e}")
            continue
        if o["spin"] not in problem_spins:
            warnings.append(f"{at}: spin {o['spin']} is not among the spins of this run, so the operator is left out")
    return errors, warnings, gap_spins, names


SKYDIVE_VECTORS = ["start", "box_min", "box_max", "hessian_guess", "find_boundary_direction"]
SKYDIVE_NUMBERS = ["gap_upper_limit", "infinitesimal", "gap_upper_limit_track", "gap_upper_limit_track_floor"]
# gap_upper_limit_track = c (docs/en/interfaces.md section 10): untested with correct gradients above 3. On by default; "off" gives bt's limits, which every run compared with bt must write.
TRACK_MAX = 3
TRACK_C = "1"                      # [skydive].gap_upper_limit_track when left out
TRACK_OFF = "off"


def track_c(o: dict):
    """The c of the tracked gap upper limit for [skydive] o, or None when the limit does not track N: "off" written,
    or gap_upper_limit_first_only = true with no c written (first_only is its own choice of the limits after the
    first call)."""
    c = o.get("gap_upper_limit_track")
    if c == TRACK_OFF:
        return None
    if c is None:
        return None if o.get("gap_upper_limit_first_only") is True else TRACK_C
    return c
# docs/en/interfaces.md section 6: the generator keys and computes its block tables with about 31 digits of Delta12, so a
# neighbour x + eps e_i with eps below that shares the centre's table and the gradient loses d/dDelta12.
# The floor keeps 5 digits of margin.
INFINITESIMAL = "1e-20"            # bt's and upstream's
INFINITESIMAL_MIN = "1e-25"


def _exact(v) -> bool:
    """An integer, or a string numbers.frac reads exactly."""
    if isinstance(v, bool):
        return False
    if isinstance(v, int):
        return True
    try:
        frac(v)
        return True
    except (ValueError, ZeroDivisionError):
        return False


def check_run_numbers(run, coords) -> list:
    """Every number of run.toml that reaches a program keeps the digits written: [skydive]'s numbers are
    strings frac reads or integers, its vectors have one entry per coordinate (hessian_guess and
    find_boundary_direction when given), and [skydive], [solver] and [dynamical] have no TOML float, which
    has already lost the digits written. Booleans and other strings (writeSolution = "x,y") are left as they are."""
    errors = []
    for sec in ("skydive", "solver", "dynamical"):
        for k, v in run.get(sec, {}).items():
            if any(isinstance(x, float) for x in (v if isinstance(v, list) else [v])):
                errors.append(f"run.toml [{sec}] {k} = {v!r} has a TOML float, which has already lost the digits "
                              f"written; put each number in quotes, with all the digits as you wrote them in the file")
    o = run.get("skydive", {})
    for k in SKYDIVE_VECTORS:
        v = o.get(k)
        if v is None or (k in ("hessian_guess", "find_boundary_direction") and v == []):
            continue
        if not isinstance(v, list) or len(v) != len(coords):
            errors.append(f"run.toml [skydive] {k} needs {len(coords)} entries, one for each of {coords}, not {v!r}")
        elif not all(isinstance(x, float) or _exact(x) for x in v):
            errors.append(f"run.toml [skydive] {k}: {v!r} is not all numbers")
    for k in SKYDIVE_NUMBERS:
        if k in o and not (k == "gap_upper_limit_track" and o[k] == TRACK_OFF) \
                and not isinstance(o[k], float) and not _exact(o[k]):
            errors.append(f"run.toml [skydive] {k}: {o[k]!r} is not a number")
    c = o.get("gap_upper_limit_track")
    if isinstance(c, (int, str)) and c != TRACK_OFF and _exact(c) and not 0 < frac(c) <= TRACK_MAX:
        errors.append(f"run.toml [skydive] gap_upper_limit_track {c} is not in (0, {TRACK_MAX}]: with the gap upper "
                      f"limit much larger than |N| (about 10 |N|) the runs drifted, and above {TRACK_MAX} it is untested "
                      f"with correct gradients; \"{TRACK_OFF}\" gives bt's limits (docs/en/interfaces.md 10)")
    if c is not None and c != TRACK_OFF and o.get("gap_upper_limit_first_only") is True:
        errors.append("run.toml [skydive] gap_upper_limit_track and gap_upper_limit_first_only = true are exclusive: "
                      "track sets the limit of every call after the first, first_only passes 0 there; set "
                      "gap_upper_limit_first_only = false (docs/en/interfaces.md 10)")
    if "gap_upper_limit_track_floor" in o and track_c(o) is None:
        errors.append("run.toml [skydive] gap_upper_limit_track_floor is set while the gap upper limit does not track N "
                      f"(gap_upper_limit_track = \"{TRACK_OFF}\" or gap_upper_limit_first_only = true): it is the kappa "
                      "of the tracked limit and does nothing alone (docs/en/interfaces.md 10)")
    if not isinstance(o.get("gap_threshold_tighten", True), bool):
        errors.append(f"run.toml [skydive] gap_threshold_tighten = {o['gap_threshold_tighten']!r} is not true or false: "
                      "false keeps [solver] dualityGapThreshold at every step, true (the default) tightens it with N "
                      "(docs/en/interfaces.md 10)")
    if not isinstance(o.get("mixed_precision", False), bool):
        errors.append(f"run.toml [skydive] mixed_precision = {o['mixed_precision']!r} is not true or false: true runs "
                      "the solver and pmp2sdp at the lowest precision that keeps enough digits and ends the run at "
                      "the problem's own (precision.py); false (the default) keeps the problem's at every call")
    eps = o.get("infinitesimal")
    if isinstance(eps, (int, str)) and _exact(eps) and frac(eps) < frac(INFINITESIMAL_MIN):
        errors.append(f"run.toml [skydive] infinitesimal {eps} is below {INFINITESIMAL_MIN}: the generator keys and "
                      f"computes its block tables with about 31 digits of Delta12, so the neighbour points would share "
                      f"the centre's tables and the gradient would be wrong (docs/en/interfaces.md 6); use {INFINITESIMAL} "
                      f"(the default) or leave it out")
    return errors


def check_gap_limit(run) -> list:
    """A gap upper limit at or below the solver's dualityGapThreshold makes dynamical_sdp end its call as
    primal-dual optimal before it computes any step: the run
    would end converged at the point it started from."""
    from .skydive import solver_defaults   # here, not at the top: skydive imports this module through pmp
    o = run.get("skydive", {})
    if "gap_upper_limit" not in o:      # the physics layer's checks load run.toml files without [skydive]
        return []
    limit = frac(o["gap_upper_limit"])
    threshold = run.get("solver", {}).get("dualityGapThreshold",
                                          solver_defaults((run["numerics"]["lambda"] + 1) // 2)["dualityGapThreshold"])
    if 0 < limit <= frac(threshold):
        return [f"run.toml [skydive] gap_upper_limit {o['gap_upper_limit']} is not above the solver's dualityGapThreshold "
                f"{threshold}: dynamical_sdp would stop as primal-dual optimal before taking a step; set [solver] "
                f"dualityGapThreshold below it"]
    return []


@dataclass
class Problem:
    run_toml: Path
    run: dict
    model: dict            # bt-model/2, with the objective's Sigma override applied
    spectrum: dict
    objective: dict
    model_coords: list     # the model's own coordinates, in bt's order
    coords: list           # model_coords, then the parameters only the spectrum uses
    spins: list            # the run's spin list
    gap_spins: list        # for each spectrum gap, the spins it sets (bt's resolution of its spins)
    warnings: list = field(default_factory=list)   # the spectrum's warnings that do not depend on the point

    @property
    def numerics(self) -> dict:
        return self.run["numerics"]

    def solver_precision(self) -> int:
        n = self.numerics
        if "solver_precision" in n:
            return n["solver_precision"]
        if n.get("block_preset", "simpleboot") == "simpleboot":
            return 785 if n["lambda"] <= 19 else 1024
        return 768 if (n["lambda"] + 1) // 2 <= 6 else 1024


def model_coordinates(model: dict) -> list:
    """bt's point order: the external dimensions, the OPE coordinates, then theta if there is an OPE angle."""
    names = [e["delta"] for e in model["externals"]]
    names += [c for f in model["fixed"] for c in f.get("opeCoordinates", [])]
    if any(f.get("opeAngle") for f in model["fixed"]):
        names.append("theta")
    return names


def load(run_toml) -> Problem:
    run_toml = Path(run_toml).resolve()
    base = run_toml.parent
    with open(run_toml, "rb") as f:
        run = tomllib.load(f)
    inputs = run["inputs"]
    model = json.loads((base / inputs["model"]).read_text(encoding="utf-8"))
    with open(base / inputs["spectrum"], "rb") as f:
        spectrum = tomllib.load(f)
    with open(base / inputs["objective"], "rb") as f:
        objective = tomllib.load(f)

    errors, warnings = [], []
    if model.get("format") != "bt-model/2":
        errors.append(f"model: format is {model.get('format')!r}, not bt-model/2")
    if spectrum.get("format") != "bootlab-spectrum/1":
        errors.append("spectrum: format must be bootlab-spectrum/1")
    if objective.get("format") != "bootlab-objective/1":
        errors.append("objective: format must be bootlab-objective/1")
    kind = objective.get("kind")
    if kind not in model.get("objectives", {}):
        errors.append(f"objective: kind {kind!r} is not among the model's objectives {sorted(model.get('objectives', {}))}")
    elif "sigma" in objective:
        if "sigma" not in model["objectives"][kind]:
            errors.append(f"objective: the model's {kind} objective has no Sigma to override")
        else:
            for k, v in objective["sigma"].items():
                if isinstance(v, bool) or not isinstance(v, (str, int)):
                    errors.append(f"objective: write sigma.{k} as a string such as \"3/10\", not {v!r}; "
                                  "a TOML float is not the exact decimal written")
                model["objectives"][kind]["sigma"][k] = str(v)

    n = run["numerics"]
    lam = n["lambda"]
    if lam % 2 == 0 or lam < 3:
        errors.append(f"numerics.lambda {lam} must be odd and at least 3")
    if "spins" in n:
        spins = list(n["spins"])
    elif n.get("block_preset", "simpleboot") == "simpleboot" and lam in HYC:
        spins = HYC[lam][1]
    else:
        spins = []
        errors.append("numerics: give spins, or use block_preset simpleboot at a Lambda it has")

    channels = {c["name"]: c["spins"] for c in model["channels"]}
    model_coords = model_coordinates(model)
    errs, warns, gap_spins, names = check_spectrum(spectrum, channels, spins)
    errors += errs
    warnings += warns
    extra = [name for name in names if name not in model_coords]

    coords = model_coords + extra
    given = run["coords"]["names"]
    if given != coords:
        errors.append(f"run.toml coords.names is {given}; this model and spectrum give {coords}")
    errors += check_run_numbers(run, coords)
    bt = run.get("skydive", {}).get("block_timings")
    if bt is not None and not (isinstance(bt, str) and Path(bt).is_absolute() and Path(bt).is_file()):
        errors.append(f"run.toml [skydive] block_timings = {bt!r}: give the absolute path of an existing file (an "
                      "earlier run's checkpoint/block_timings); a new run then makes no timing run")
    # the solver's nodes (layout.choose_nodes): a list of nodes or "auto", and auto's limits; allow_multinode:
    # several nodes where the site's default is one (layout.permitted)
    res = run.get("resources", {})
    errors += _unknown_keys("run.toml [resources]", res,
                            ["nodes", "max_nodes", "min_cores", "mem_gb", "prefer", "allow_multinode"])
    if not isinstance(res.get("allow_multinode", False), bool):
        errors.append(f"run.toml [resources] allow_multinode = {res['allow_multinode']!r}: true or false")
    if not errors:
        errors += check_gap_limit(run)
    if errors:
        raise SystemExit("the inputs have errors:\n  " + "\n  ".join(errors))
    return Problem(run_toml, run, model, spectrum, objective, model_coords, coords, spins, gap_spins, warnings)


def _values(problem: Problem, point: dict):
    """The spectrum's values at a point, exact: default_above_unitarity, each gap's, each isolated delta."""
    env = {k: frac(v) for k, v in point.items()}
    s = problem.spectrum
    twist = Expr(s.get("default_above_unitarity", "0")).value(env)
    gaps = [Expr(g["delta"] if "delta" in g else g["above_unitarity"]).value(env) for g in s.get("gap", [])]
    isolated = [Expr(o["delta"]).value(env) for o in s.get("isolated", [])]
    return twist, gaps, isolated


def resolve_spectrum(problem: Problem, point: dict) -> dict:
    """bt's spectrum section at a point (coordinate name -> decimal string), with every value exact. Like bt,
    the same isolated operator twice is an error."""
    s = problem.spectrum
    twist, gaps, isolated = _values(problem, point)
    seen = set()
    for o, d in zip(s.get("isolated", []), isolated):
        if (o["channel"], o["spin"], d) in seen:
            raise SystemExit(f"spectrum isolated has {o['channel']} spin {o['spin']} at {decimal_text(d)} twice "
                             f"at the point {point}")
        seen.add((o["channel"], o["spin"], d))
    out = {"twistGap": Num(decimal_text(twist)), "gaps": [], "isolated": []}
    for g, x in zip(s.get("gap", []), gaps):
        key = "gap" if "delta" in g else "aboveUnitarity"
        out["gaps"].append({"channel": g["channel"], "spins": g["spins"], key: Num(decimal_text(x))})
    for o, d in zip(s.get("isolated", []), isolated):
        out["isolated"].append({"channel": o["channel"], "spin": o["spin"], "delta": Num(decimal_text(d))})
    return out


def spectrum_warnings(problem: Problem, point: dict) -> list:
    """bt's warnings that depend on the values (bt computes with the values as written): a negative
    default_above_unitarity or above_unitarity, a gap below the unitarity bound, an isolated operator below
    the bound or above where the continuum of its channel and spin starts."""
    s = problem.spectrum
    twist, gaps, isolated = _values(problem, point)
    out = ["spectrum default_above_unitarity is negative, below the unitarity bound"] if twist < 0 else []
    start = {}
    for i, (g, x, ls) in enumerate(zip(s.get("gap", []), gaps, problem.gap_spins)):
        for l in ls:
            start[(g["channel"], l)] = x if "delta" in g else unitarity(l) + x
        if "delta" not in g and x < 0:
            out.append(f"spectrum gap[{i}]: above_unitarity is negative, below the unitarity bound")
        if "delta" in g:
            out += [f"spectrum gap[{i}]: delta {decimal_text(x)} of {g['channel']} spin {l} is below the unitarity "
                    f"bound {decimal_text(unitarity(l))}" for l in ls if x < unitarity(l)]
    for i, (o, d) in enumerate(zip(s.get("isolated", []), isolated)):
        ch, l = o["channel"], o["spin"]
        u = unitarity(l)
        continuum = start.get((ch, l), u + twist)
        if d < u:
            out.append(f"spectrum isolated[{i}]: delta {decimal_text(d)} is below the unitarity bound {decimal_text(u)}")
        if d > continuum:
            out.append(f"spectrum isolated[{i}]: delta {decimal_text(d)} is above where the continuum of {ch} spin {l} "
                       f"starts, {decimal_text(continuum)}")
    return out
