"""A scan: one solve per point of a list, for the bounds of built-in models (docs/en/interfaces.md, section 19).

Each point: the generator writes its PMP (block tables are kept in the run's block directory, and the parts
in the run's one PMP directory, so later points reuse what they share with earlier ones: blocks, and the parts
that do not depend on the point, as upstream's persistent composite blocks and json parts are reused);
pmp2sdp builds its SDP; one sdpb call with the bound's own solver parameters (the generator's
sdpb.args.json, upstream's), run.toml [solver] on top; one line in solves.jsonl. A failure ends the scan;
`bootlab resume` goes on from the first point without a line.
"""
import hashlib
import json
import shutil
import time
from pathlib import Path

from . import bench, builtin, execute, pmp


def solver_args(args_file: Path, overrides: dict, max_runtime: int) -> list:
    """sdpb's options: the generator's (flags with their values, switches alone), maxRuntime from the job's
    time left, then run.toml [solver]: a value replaces the flag's, true adds a switch, false removes it."""
    flags = {k: None if v is True else v for k, v in builtin.solver_options(args_file).items()}
    flags["maxRuntime"] = str(max_runtime)
    for k, v in overrides.items():
        if v is False:
            flags.pop(k, None)
        else:
            flags[k] = None if v is True else str(v)
    out = []
    for k, v in flags.items():
        out += [f"--{k}"] + ([v] if v is not None else [])
    return out


class Scan:
    def __init__(self, problem, site, node, run_dir: Path, scratch: Path = None):
        """scratch: the run's directory as execute.choose_scratch chose it; left out, under the site's one
        scratch path."""
        self.p, self.site, self.node, self.dir = problem, site, node, run_dir
        self.gen = site.generators[problem.model["generator"]]
        self.scratch = scratch or Path(site.scratch) / run_dir.name
        self.blocks = self.scratch / "blocks"
        self.pmp_dir = self.scratch / "pmp"       # every point's parts, as builtin.Skydive's
        self.started = time.strftime("%F %T")

    def log(self, msg: str):
        line = f"{time.strftime('%F %T')} {msg}"
        print(line, flush=True)
        with open(self.dir / "driver.log", "a") as f:
            f.write(line + "\n")

    def record(self, line: dict):
        with open(self.dir / "solves.jsonl", "a") as f:
            f.write(json.dumps(line) + "\n")

    def done(self) -> int:
        """Points with a line in solves.jsonl."""
        path = self.dir / "solves.jsonl"
        return sum(1 for l in path.read_text().splitlines() if l.strip()) if path.exists() else 0

    def run(self):
        points = self.p.run["scan"]["points"]
        procs = None
        for i in range(self.done(), len(points)):
            x = dict(zip(self.p.coords, [str(v) for v in points[i]]))
            self.log(f"point {i} at {points[i]}")
            try:
                line, procs = self.point(i, x, procs)
            except execute.StepFailed as e:
                self.record({"point": i, "x": [x[c] for c in self.p.coords], "verdict": "aborted",
                             "reason": f"failed: {e}", "node": self.node.host})
                return self.finish("failed", f"failed: {e}")
            self.record(line)
            if line["verdict"] == "aborted":
                return self.finish("unfinished", line["reason"])
        return self.finish("done", f"{len(points)} points")

    def point(self, i: int, x: dict, procs):
        d = self.scratch / f"point_{i}"
        logs = self.dir / "points" / str(i)
        seconds = {}
        t = time.time()
        req = builtin.write_request(self.p, {"p": x}, self.pmp_dir, self.blocks, self.node.cores)
        builtin.generate(self.site, self.gen, self.node, req, self.scratch / "local", logs / "generator")
        lists = pmp.pmp_lists(self.pmp_dir, ["p"])
        for f in ("sdpb.args.json", "bound.json"):
            shutil.copy(self.pmp_dir / "p" / f, logs / f)
        seconds["pmp"] = round(time.time() - t, 1)

        args_file = self.pmp_dir / "p" / "sdpb.args.json"
        argv = json.loads(args_file.read_text())
        precision = argv[argv.index("--precision") + 1]
        t = time.time()
        sdps, _ = execute.build_sdps(self.site, self.node, lists, d / "sdp", precision, logs, 1,
                                     bench.pmp2sdp_procs(self.node.cores, 1), program=self.gen["pmp2sdp"])
        seconds["sdp"] = round(time.time() - t, 1)
        n_q, blocks = execute.sdp_size(sdps["p"])
        if procs is None:
            procs, why = bench.solver_procs(bench.load_bench(self.site.bench_dir, self.node.node_type),
                                            self.node.cores, n_q, blocks, self.gen["sdpb"])
            self.log(f"solver processes {procs}: {why}")

        t = time.time()
        args = solver_args(args_file, self.p.run.get("solver", {}), execute.job_seconds_left() - 900)
        out = d / "out"
        code = execute.run_sdpb(self.site, self.node, self.gen["sdpb"], procs, sdps["p"], out, d / "ck", args,
                                logs / "sdpb")
        seconds["solve"] = round(time.time() - t, 1)
        line = {"point": i, "x": [x[c] for c in self.p.coords], "N": n_q, "blocks": blocks, "procs": procs,
                "node": self.node.host, "seconds": seconds}
        if not (out / "out.txt").exists():
            line.update(verdict="aborted", reason=f"failed: sdpb exit {code}, no out.txt")
        else:
            shutil.copy(out / "out.txt", logs / "out.txt")
            o = execute.read_kv(out / "out.txt")
            verdict, category = execute.classify(o["terminateReason"])
            line.update(verdict=verdict, reason=f"{category}: {o['terminateReason']}" if category else o["terminateReason"],
                        primal=o.get("primalObjective"), dual=o.get("dualObjective"), gap=o.get("dualityGap"))
        shutil.rmtree(d, ignore_errors=True)
        return line, procs

    def finish(self, status: str, reason: str):
        result = {"status": status, "reason": reason, "points": self.done(), "inputs": {
                      name: hashlib.sha256((self.dir / name).read_bytes()).hexdigest()
                      for name in ("model.json", "spectrum.toml", "objective.toml")},
                  "versions": {**self.versions(), "pmp2sdp": self.gen["pmp2sdp"], "sdpb": self.gen["sdpb"]},
                  "start": self.started, "end": time.strftime("%F %T")}
        (self.dir / "result.json").write_text(json.dumps(result, indent=1) + "\n")
        self.log(f"end: {status}: {reason}")
        shutil.rmtree(self.scratch, ignore_errors=True)

    def versions(self) -> dict:
        """What fixes the programs: the image's sha256 (pmp2sdp and sdpb are in it; so is the generator when it
        runs there, container = true, its path then being the image's), and a host generator's own sha256."""
        def sha(p):
            with open(p, "rb") as f:
                return hashlib.file_digest(f, "sha256").hexdigest()
        v = {"generator": self.gen["program"], "sif": self.site.sif, "sif_sha256": sha(self.site.sif)}
        if self.gen.get("container"):
            v["generator_in"] = "image"
        else:
            v["generator_sha256"] = sha(self.gen["program"])
        return v


def summary_lines(run: dict, lines: list) -> list:
    coords = run["coords"]["names"]
    head = ["point", *coords, "reason", "primal", "dual", "gap", "N", "blocks", "pmp s", "sdp s", "solve s"]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for l in lines:
        sec = l.get("seconds", {})
        row = [l["point"], *l["x"], l["reason"], l.get("primal", ""), l.get("dual", ""), l.get("gap", ""),
               l.get("N", ""), l.get("blocks", ""), sec.get("pmp", ""), sec.get("sdp", ""), sec.get("solve", "")]
        out.append("| " + " | ".join(str(v) if v is not None else "" for v in row) + " |")
    return out
