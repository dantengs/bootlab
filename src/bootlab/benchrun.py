"""bootlab bench: one job on one node measures the generation and the solver on the user's own problem and writes
the rows the layout and the Skydive loop read (docs/en/interfaces.md section 9). Fixed: the run's start point
alone (no neighbours). At each precision a run calls at first (precisions): the generator on all physical cores,
its wall time and the node's memory rise (a `generator` row), and pmp2sdp with the processes a run gives each
instance (bench.pmp2sdp_procs for the run's points), its wall time and held memory as execute.build_sdps measures
it (an `sdp_build` row, which skydive's build_sdps takes for its first measurement). Then at full precision sdpb on
all the node's physical cores and on half of them (non-degenerate grids only), ITERATIONS iterations each; the
steady seconds per iteration from sdpb's iteration table (`solver` rows). A row's source names the date and the
machine only."""
import datetime
import json
import re
import threading
import time
from pathlib import Path

from . import bench, execute, pmp, precision
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


def precisions(problem) -> list:
    """The precisions a run of the problem calls at first: its own, and with [skydive] mixed_precision the
    ladder's choice at gap_upper_limit (the first call's, skydive's loop) before it."""
    full = problem.solver_precision()
    o = problem.run.get("skydive", {})
    low = precision.choose(full, o["gap_upper_limit"]) if o.get("mixed_precision") else full
    return list(dict.fromkeys([low, full]))


class NodeRise:
    """The largest rise of the node's MemTotal - MemAvailable while the block runs, MB, sampled every 2 s."""
    def __enter__(self):
        self.base, self.mb, self.on = execute.node_used_mb(), 0.0, True
        self.thread = threading.Thread(target=self.watch, daemon=True)
        self.thread.start()
        return self

    def watch(self):
        while self.on:
            self.mb = max(self.mb, execute.node_used_mb() - self.base)
            time.sleep(2)

    def __exit__(self, *exc):
        self.on = False
        self.thread.join()


def measure(problem, site, node, work: Path, scratch: Path, log) -> dict:
    """The rows of one problem on this node, {"generator": [...], "sdp_build": [...], "solver": [...]} (see the
    module's docstring)."""
    model_file = work / "model.json"
    model_file.write_text(json.dumps(problem.model, indent=1) + "\n")
    x = dict(zip(problem.coords, problem.run["skydive"]["start"]))
    name, lam = problem.model_name, problem.numerics["lambda"]
    what = {"problem": f"{name} L={lam} (bootlab bench)", "model": name, "lambda": lam}
    source = f"bootlab bench {datetime.date.today().isoformat()}, {machine()}"
    per = bench.pmp2sdp_procs(node.cores, len(problem.coords) + 1)
    rows = {"generator": [], "sdp_build": [], "solver": []}
    full = problem.solver_precision()
    for bits in precisions(problem):
        out = scratch / f"pmp-{bits}"
        req = pmp.write_request(problem, model_file, {"c": x}, out, scratch / f"blocks-{bits}", node.cores, bits)
        with NodeRise() as rise:
            t = pmp.generate(site, node, req, work / f"generator-{bits}")
        lists = pmp.pmp_lists(out, ["c"])
        rows["generator"].append({**what, "precision": bits, "points": 1, "cores": node.cores,
                                  "program": "scalars-3d-exec pmp", "wall": round(t, 1),
                                  "node_gb": round(rise.mb / 1024, 1), "source": source})
        t = time.time()
        sdps, (w,) = execute.build_sdps(site, node, lists, scratch / f"sdp-{bits}", bits, work / f"sdp-{bits}", 1, per)
        wall = round(time.time() - t, 1)
        n, blocks = execute.sdp_size(sdps["c"])
        rows["sdp_build"].append({**what, "N": n, "blocks": blocks, "precision": bits, "program": "pmp2sdp 3.0sky",
                                  "instances": 1, "procs": per, "sdps": 1, "wall": wall, "per_sdp": wall,
                                  "peak_gb_per_instance": round(w["held_mb"] / 1024, 2), "held_mb": w["held_mb"],
                                  "cgroup_peak_mb": w["cgroup_peak_mb"], "source": source})
        log(f"{bits} bits: generator {rows['generator'][-1]['wall']} s (node +{rows['generator'][-1]['node_gb']} GB); "
            f"pmp2sdp {per} processes {wall} s, held {w['held_mb']} MB; SDP N {n}, {blocks} blocks")
    sdp = scratch / f"sdp-{full}" / "c"
    n, blocks = execute.sdp_size(sdp)
    for p in procs_to_try(node.cores):
        d = scratch / f"sdpb-{p}"
        argv = [str(a) for a in [SDPB, "--sdpDir", sdp, "--outDir", d / "out", "--checkpointDir", d / "ck",
                "--precision", full, "--maxIterations", ITERATIONS, "--verbosity", "1"]]
        out_log = work / f"sdpb-{p}.out"
        code = execute.launch(site, node, argv, p, 1, node.mem_mb - site.driver_reserve_mb, True, out_log,
                              hosts=node.solo()).wait()
        s = steady_seconds(out_log.read_text(errors="replace"))
        log(f"sdpb {p} processes: exit {code}, " + (f"{s:.3f} s/iter" if s is not None else "no steady time"))
        if s is None:
            continue
        rows["solver"].append({**what, "N": n, "blocks": blocks, "precision": full, "program": "sdpb 3.0sky",
                               "nodes": 1, "procs_per_node": p, "s_per_iter": round(s, 3), "source": source})
    return rows


def append_rows(bench_dir: Path, node_type: str, cores: int, rows: dict) -> Path:
    """The rows (list name -> rows) added to <bench_dir>/<node_type>.json (made when there is none); nothing
    else changes."""
    path = Path(bench_dir) / f"{node_type}.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else \
        {"format": "bootlab-bench/1", "node_type": node_type, "physical_cores": cores}
    for k, v in rows.items():
        data.setdefault(k, []).extend(v)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
