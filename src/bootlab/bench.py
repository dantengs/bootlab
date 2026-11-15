"""Cores and processes from measured data (docs/en/interfaces.md section 9, sites/<site>/bench/<node type>.json)."""
import json
import math
from pathlib import Path


def grid_ok(p: int) -> bool:
    """Elemental's default grid for p processes: the first divisor of p from floor(sqrt(p)) up is the height.
    A grid whose height is more than twice its width (127 -> 127x1, 254 -> 127x2) slows the solver down."""
    h = next(d for d in range(math.isqrt(p), p + 1) if p % d == 0)
    return h <= 2 * (p // h)


def load_bench(bench_dir: Path, node_type: str) -> dict:
    path = Path(bench_dir) / f"{node_type}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"solver": [], "sdp_build": [], "blocks": []}


def sdpb_version(program: str) -> str:
    """The SDPB version a program is built on, its name's last word: sdpb_3.1.0 and "sdpb 3.1.0" -> 3.1.0,
    dynamical_sdp_RV1B_3.0sky and "dynamical_sdp RV1B 3.0sky" -> 3.0sky (dynamical_sdp is SDPB 3.0sky's solver)."""
    return Path(program).name.replace("_", " ").split()[-1]


def solver_procs(bench: dict, cores: int, n: int, blocks: int, program: str = None):
    """The solver's processes on one node: the fastest measured count, on a grid that does not degenerate,
    for the measured problem closest to (N, blocks), among the rows of program's SDPB version when there are
    any (another version's problems once pulled Ising Lambda 11 from 256 processes to 128), else among all;
    all physical cores when nothing is measured. Returns (procs, reason)."""
    rows = [r for r in bench.get("solver", []) if r["nodes"] == 1 and grid_ok(r["procs_per_node"])
            and r["procs_per_node"] <= cores]
    if not rows:
        return cores, f"no solver measurement on this node type: all {cores} physical cores"
    version = program and sdpb_version(program)
    if version and any(sdpb_version(r["program"]) == version for r in rows):
        rows = [r for r in rows if sdpb_version(r["program"]) == version]

    def distance(r):
        return abs(math.log(n / r["N"])) + abs(math.log(blocks / r["blocks"]))

    nearest = min({r["problem"]: r for r in rows}.values(), key=distance)
    same = [r for r in rows if r["problem"] == nearest["problem"]]
    best = min(same, key=lambda r: r["s_per_iter"])
    measured = sorted(r["procs_per_node"] for r in same)
    note = (f"nearest measured problem {nearest['problem']}, {nearest['program']} (N={nearest['N']}, {nearest['blocks']} blocks; "
            f"this SDP N={n}, {blocks} blocks): fastest of {measured} is {best['procs_per_node']} "
            f"({best['s_per_iter']} s/iter; {best['source']})")
    if cores not in measured and grid_ok(cores):
        note += f"; {cores} was not measured there"
    return best["procs_per_node"], note


# What one pmp2sdp instance is taken to need: its measured peak times this. It covers repeats of the same
# measurement (a Lambda 23 problem at 16, 32 processes: within 1%), 8 instances at once against one
# (+6%: 28.6 GB an instance against 27.0 alone) and the points of a step (the same N,
# blocks and PMP size). Two repeats at 64 and 128 processes were 11-15% higher, cause
# unknown. A wave that needs more anyway is cancelled (execute.MEMORY_WATCH), it does not stall.
SDP_MEMORY_MARGIN = 1.15


def pmp2sdp_procs(cores: int, count: int) -> int:
    """Processes of each pmp2sdp instance for count SDPs: min(31, cores // count) (measured: one SDP is
    fastest at 31; many at once are fastest with many small instances). A run keeps them for all its steps,
    so the peak measured at its first step is the peak at the processes it uses."""
    return max(1, min(31, cores // min(count, cores)))


def sdp_build_row(bench: dict, model: str, lam: int, precision: int, procs: int):
    """The last sdp_build row that bootlab bench wrote (it has held_mb) for this model and Lambda, precision and
    pmp2sdp processes, or None."""
    rows = [r for r in bench.get("sdp_build", []) if "held_mb" in r and (r.get("model"), r.get("lambda"),
            r.get("precision"), r.get("procs")) == (model, lam, precision, procs)]
    return rows[-1] if rows else None


def pmp2sdp_at_once(mem_mb: int, peak_mb: float) -> int:
    """How many pmp2sdp instances the node holds at once: mem_mb (the node's memory less the driver's reserve)
    over one instance's measured peak times SDP_MEMORY_MARGIN, rounded down."""
    return int(mem_mb // (peak_mb * SDP_MEMORY_MARGIN))
