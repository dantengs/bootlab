"""bootlab's regression (docs/en/interfaces.md section 16): examples/ising (ising-oe, Lambda = 11,
bt r3.8's Skydive settings), three steps, against bt r3.8's first three steps kept in the package
(tests/reference/ising-l11/bt-steps.json). Standard library only (decimal at 500 digits): the installation's
Python has no mpmath.

For each step and each number: the digits on which bootlab and bt agree, -log10(|a - b| / max(|a|, |b|)), "all"
when equal; every number must agree to MIN_DIGITS or more, and the reason, the iteration count, climbedQ and
findMinimumQ must be equal; every reference step must be there.
"""
import decimal
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REFERENCE = REPO / "tests" / "reference" / "ising-l11" / "bt-steps.json"
EXAMPLE = REPO / "examples" / "ising"
MIN_DIGITS = 130
STEPS = 3
NUMBERS = ["navigator", "primal", "dual", "gap", "alpha_p", "alpha_d", "beta"]
VECTORS = ["x", "ext_step", "gradient"]
EXACT = ["reason", "iterations", "climbed", "find_minimum"]
CTX = decimal.Context(prec=500)


def digits(a: str, b: str):
    a, b = CTX.create_decimal(a), CTX.create_decimal(b)
    if a == b:
        return "all"
    rel = CTX.divide(abs(CTX.subtract(a, b)), max(abs(a), abs(b)))
    return round(float(-rel.log10(CTX)), 1)


def least(ds):
    nums = [d for d in ds if d != "all"]
    return min(nums) if nums else "all"


def run_steps(run: Path) -> list:
    lines = [json.loads(l) for l in (Path(run) / "solves.jsonl").read_text().splitlines() if l.strip()]
    for l in lines:   # bootlab writes "unfinished: <solver's reason>" for a step the solver gave up on
        l["reason"] = l["reason"].split(": ", 1)[-1]
    return lines


def compare(ours: dict, ref: dict) -> dict:
    row = {"step": ours["step"]}
    for key in EXACT:
        row[key] = "same" if ours.get(key) == ref.get(key) else f"{ours.get(key)} != {ref.get(key)}"
    if ours["verdict"] == "aborted":      # nothing else is read from a step the solver gave up on
        return row
    for key in NUMBERS:
        row[key] = digits(ours[key], ref[key])
    for key in VECTORS:
        row[key] = least([digits(a, b) for a, b in zip(ours[key], ref[key])])
    row["hessian"] = least([digits(a, b) for ra, rb in zip(ours["hessian"], ref["hessian"]) for a, b in zip(ra, rb)])
    return row


def check(run: Path, reference: Path = REFERENCE, min_digits: float = MIN_DIGITS) -> tuple:
    """(passed, report lines, report dict) of a run against the reference."""
    ours = run_steps(run)
    ref = json.loads(Path(reference).read_text())["steps"]
    by_step = {r["step"]: r for r in ref}
    rows = [compare(o, by_step[o["step"]]) for o in ours if o["step"] in by_step]
    cols = ["step", "x", "navigator", "ext_step", "gradient", "hessian", "primal", "dual", "gap", "alpha_p", "beta",
            *EXACT]
    lines = [f"bootlab {len(ours)} steps ({run}), reference {len(ref)} steps ({reference}); compared {len(rows)}",
             " ".join(f"{c[:9]:>9}" for c in cols)]
    lines += [" ".join(f"{str(r.get(c, '')).replace('same', '=')[:9]:>9}" for c in cols) for r in rows]
    worst = {c: least([r[c] for r in rows if c in r]) for c in ["x", "navigator", "ext_step", "gradient", "hessian"]}
    lines.append("least digits: " + json.dumps(worst))
    bad = [f"step {r['step']} {k}: {v}" for r in rows for k, v in r.items() if k != "step"
           and (v != "same" if k in EXACT else v != "all" and v < min_digits)]
    if len(rows) < len(ref):
        bad.append(f"{len(rows)} steps compared, the reference has {len(ref)}")
    lines.append("FAIL\n" + "\n".join(bad) if bad else f"PASS: {len(rows)} steps, every number to {min_digits} digits or more")
    report = {"run": str(run), "reference": str(reference), "min_digits": min_digits, "ours_steps": len(ours),
              "ref_steps": len(ref), "least": worst, "steps": rows, "passed": not bad, "problems": bad}
    return not bad, lines, report


def input_dir(out: Path, site: str = None, nodes: list = None) -> Path:
    """examples/ising with STEPS steps (and the site and nodes when given) in a new directory out."""
    out.mkdir(parents=True)
    for f in ("model.json", "spectrum.toml", "objective.toml"):
        (out / f).write_bytes((EXAMPLE / f).read_bytes())
    text = (EXAMPLE / "run.toml").read_text()
    a, b = 'name   = "ising-l11-skydive"', "max_steps      = 300"
    assert text.count(a) == 1 and text.count(b) == 1, "examples/ising/run.toml changed: the regression's edits"
    text = text.replace(a, 'name   = "regress-ising-l11"').replace(b, f"max_steps      = {STEPS}")
    if site:
        text += f'\n[exec]\nsite = "{site}"\n'
    if nodes:
        text += "\n[resources]\nnodes = [" + ", ".join(json.dumps(n) for n in nodes) + "]\n"
    (out / "run.toml").write_text(text)
    return out / "run.toml"
