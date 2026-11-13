"""bootlab bench: one job on one node measures the solver on the user's own problem and writes the rows the
layout reads (docs/en/interfaces.md section 9). Fixed: the SDP of the run's start point (no neighbours), sdpb on
all the node's physical cores and on half of them (non-degenerate grids only), ITERATIONS iterations each; the
steady seconds per iteration from sdpb's iteration table. A row's source names the machine only."""
import datetime
import json
import re
from pathlib import Path

from . import bench, execute, pmp
from .probe import cpu

ITERATIONS = 12
SKIP = 3                  # the first iterations are not steady (the start, the first factorizations)
SDPB = "sdpb_3.0sky"      # in the image
# a row of sdpb's iteration table: its number, the seconds so far (whole), mu, P-obj (signed), ...
_ROW = re.compile(r"^\s*(\d+)\s+(\d+)\s+\S+\s+[+-]\S+\s", re.M)


def procs_to_try(cores: int) -> list:
    """All physical cores and half of them, the grids that do not degenerate (bench.grid_ok)."""
    return [p for p in dict.fromkeys((cores, cores // 2)) if p >= 1 and bench.grid_ok(p)]


def steady_seconds(text: str):
    """Seconds per iteration from an iteration table, after the first SKIP iterations: (t_last - t_first) /
    (n_last - n_first) over the rows from SKIP + 1 on. The table's seconds are whole, so the error is about
    1 s over the iterations counted. None with fewer than two such rows."""
    rows = [(int(n), int(t)) for n, t in _ROW.findall(text) if int(n) > SKIP]
    if len(rows) < 2 or rows[-1][0] == rows[0][0]:
        return None
    return (rows[-1][1] - rows[0][1]) / (rows[-1][0] - rows[0][0])


def machine() -> str:
    """The machine's configuration for a row's source: sockets, CPU model, physical cores, memory."""
    c = cpu()
    return (f"{c['sockets']} x {c['model']}, {c['physical_cores']} physical cores, "
            f"{(c['mem_total_mb'] or 0) // 1024} GB")


def measure(problem, site, node, work: Path, scratch: Path, log) -> list:
    """The solver rows of one problem on this node (see the module's docstring)."""
    model_file = work / "model.json"
    model_file.write_text(json.dumps(problem.model, indent=1) + "\n")
    x = dict(zip(problem.coords, problem.run["skydive"]["start"]))
    out = scratch / "pmp"
    req = pmp.write_request(problem, model_file, {"c": x}, out, scratch / "blocks", node.cores)
    log(f"generator: {pmp.generate(site, node, req, work / 'generator.log'):.1f} s")
    lists = pmp.pmp_lists(out, ["c"])
    precision = problem.solver_precision()
    sdps, _ = execute.build_sdps(site, node, lists, scratch / "sdp", precision, work, 1,
                                 bench.pmp2sdp_procs(node.cores, 1))
    n, blocks = execute.sdp_size(sdps["c"])
    log(f"SDP: N {n}, {blocks} blocks, precision {precision}")
    rows, where = [], machine()
    for p in procs_to_try(node.cores):
        d = scratch / f"sdpb-{p}"
        argv = [str(a) for a in [SDPB, "--sdpDir", sdps["c"], "--outDir", d / "out", "--checkpointDir", d / "ck",
                "--precision", precision, "--maxIterations", ITERATIONS, "--verbosity", "1"]]
        out_log = work / f"sdpb-{p}.out"
        code = execute.launch(site, node, argv, p, 1, node.mem_mb - site.driver_reserve_mb, True, out_log,
                              hosts=node.solo()).wait()
        s = steady_seconds(out_log.read_text(errors="replace"))
        log(f"sdpb {p} processes: exit {code}, " + (f"{s:.3f} s/iter" if s is not None else "no steady time"))
        if s is None:
            continue
        rows.append({"problem": f"{problem.model['name']} L={problem.numerics['lambda']} (bootlab bench)",
                     "model": problem.model["name"], "lambda": problem.numerics["lambda"], "N": n,
                     "blocks": blocks, "precision": precision, "program": "sdpb 3.0sky", "nodes": 1,
                     "procs_per_node": p, "s_per_iter": round(s, 3),
                     "source": f"bootlab bench {datetime.date.today().isoformat()}, {where}"})
    return rows


def append_rows(bench_dir: Path, node_type: str, cores: int, rows: list) -> Path:
    """The rows added to <bench_dir>/<node_type>.json (made when there is none); nothing else changes."""
    path = Path(bench_dir) / f"{node_type}.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else \
        {"format": "bootlab-bench/1", "node_type": node_type, "physical_cores": cores}
    data.setdefault("solver", []).extend(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
