"""The conversion layer's side of PMP generation: the generator's input (docs/en/interfaces.md section 6), the
generator's run, and the PMP of each point.

The generator is bt's scalars-3d-exec with the `pmp` subcommand. It reads each point's coordinates as bt's
Skydive does (to BigFloat 512, then to the bound's key), builds the blocks and writes the PMP json parts of
all the points into one shared directory, and one NUL-separated list of parts per point,
<out>/<name>/pmp.nsv. It runs in bootlab's image (site.toml sif), as the other programs do.
"""
import json
import re
import time
from pathlib import Path

from . import execute
from .inputs import Problem, resolve_spectrum
from .numbers import Num, dumps

_LABEL = re.compile(r"^(continuum|isolated)_(.+)_l(\d+)_([ud])(.+?)_mat_")


def model_section(problem: Problem, model_file: Path, bits=None) -> dict:
    """bits: the precision of the call the PMP is for. Below the problem's (Problem.solver_precision, a mixed
    precision call) the generator works at it too, blocks and parts (blockPrecision; at most the problem's
    block_precision when it sets one); at the problem's precision the request is the one without bits."""
    n = problem.numerics
    out = {"name": "data", "file": str(model_file), "lambda": n["lambda"],
           "blockPreset": n.get("block_preset", "simpleboot")}
    if "block_precision" in n:
        out["blockPrecision"] = n["block_precision"]
    if bits is not None and bits < problem.solver_precision():
        out["blockPrecision"] = min(bits, n.get("block_precision", bits))
    if "spins" in n:
        out["spinList"] = n["spins"]
    return out


def write_request(problem: Problem, model_file: Path, points: dict, out: Path, blocks: Path, cores: int,
                  bits=None) -> Path:
    """points: name -> {coordinate: decimal string}, in problem.coords; bits: the call's precision (model_section)."""
    req = {
        "model": model_section(problem, model_file, bits),
        "objective": problem.objective["kind"],
        "points": [{"name": name,
                    "coords": [Num(p[c]) for c in problem.model_coords],
                    "spectrum": resolve_spectrum(problem, p)}
                   for name, p in points.items()],
        "out": str(out),
        "blocks": str(blocks),
        "cores": cores,
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / "request.json"
    path.write_text(dumps(req) + "\n")
    return path


def watched(request: Path, log: Path) -> list:
    """What a generator call changes while it works: its logs, its output directory, its block directory."""
    return [log.with_suffix(".out"), log.with_suffix(".err"), request.parent,
            Path(json.loads(request.read_text())["blocks"])]


def generate(site, node, request: Path, log: Path, stall_s: float = None) -> float:
    """Run the generator on the whole node (node 0 in a job of several nodes), in the image: site.generator is
    its path there, a script that sets the generator's libraries and its scalar_blocks script and starts it;
    its workers are processes of the same container. stall_s: the call ends as failed when none of its logs,
    output and block directories changes for that long (execute.wait_watched). Returns its wall time."""
    t = time.time()
    proc = execute.launch(site, node, [site.generator, "pmp", request], 1, node.cores,
                          node.mem_mb - site.driver_reserve_mb, False, log, hosts=node.solo())
    code = proc.wait() if stall_s is None else \
        execute.wait_watched(proc, watched(request, log), stall_s, "generator", log)
    if code != 0:
        raise execute.StepFailed(f"generator exit {code}")
    return time.time() - t


def pmp_lists(out: Path, names) -> dict:
    """name -> the point's pmp.nsv, checking that every part it lists exists."""
    lists = {}
    for name in names:
        nsv = out / name / "pmp.nsv"
        parts = [p for p in nsv.read_text().split("\0") if p]
        missing = [p for p in parts if not Path(p).exists()]
        if missing:
            raise execute.StepFailed(f"generator: {len(missing)} parts of {name} missing, e.g. {missing[0]}")
        lists[name] = nsv
    return lists


def labels(nsv: Path) -> list:
    """The constraint labels bt writes into part names (continuum_<channel>_l<spin>_<u|d><gap>_mat_...)."""
    out = []
    for p in nsv.read_text().split("\0"):
        m = _LABEL.match(Path(p).name)
        if m:
            kind, channel, spin, rel, delta = m.groups()
            out.append({"part": Path(p).name, "kind": kind, "channel": channel, "spin": int(spin),
                        "delta": delta, "relative_to_unitarity": rel == "u"})
    return out


def write_labels(pmp_dir: Path):
    """labels.json next to pmp_dir/pmp.nsv: the constraint of each part, for the EFM (docs/en/interfaces.md section 7)."""
    (pmp_dir / "labels.json").write_text(json.dumps(labels(pmp_dir / "pmp.nsv"), indent=1) + "\n")
