"""bootlab run <run.toml> | resume <run dir> | submit [--resume] [--dry] <run.toml | run dir> | summary <run dir>
| regress [--node <host>] [--dry] | probe ... | site ....

run and resume work inside a Slurm allocation (sites/<site>/run.sbatch) of one node, or of several of one type
for the solver (run.toml [resources]); submit, on the login node, chooses the nodes and calls sbatch; summary
works anywhere.
"""
import argparse
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import time
import tomllib
from pathlib import Path

from . import bench, execute, inputs, layout, probe, regress, sitegen, skydive


def _new_run_dir(problem, site) -> Path:
    name = problem.run["name"]
    d = Path(site.runs) / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}"
    d.mkdir(parents=True)
    # The run keeps its own copy of the inputs; run.toml there points at them, so a resume reads the copies.
    (d / "model.json").write_text(json.dumps(problem.model, indent=1) + "\n")
    base = problem.run_toml.parent
    shutil.copy(base / problem.run["inputs"]["spectrum"], d / "spectrum.toml")
    shutil.copy(base / problem.run["inputs"]["objective"], d / "objective.toml")
    text = problem.run_toml.read_text()
    for key, new in (("model", "model.json"), ("spectrum", "spectrum.toml"), ("objective", "objective.toml")):
        text = re.sub(rf'^({key}\s*=\s*)"[^"]*"', rf'\g<1>"{new}"', text, count=1, flags=re.M)
    (d / "run.toml").write_text(text)
    return d


def note_image(r):
    """The image's sha256, once at the start of a run or resume: it fixes every program the run uses, the
    generator among them (result.json versions; the image's /opt/bootlab/MANIFEST lists them)."""
    with open(r.site.sif, "rb") as f:
        r.sif_sha256 = hashlib.file_digest(f, "sha256").hexdigest()
    r.log(f"image {r.site.sif} sha256 {r.sif_sha256}; generator {r.site.generator} in it")


def here(problem, site) -> execute.Node:
    """This job's nodes, checked against run.toml [resources] and the site before anything runs; the site's
    environment applied to the driver's first, so every step inherits it."""
    execute.apply_env(site)
    node = execute.Node.here(site.node_types)
    mn = layout.multinode_of({"multinode": site.multinode}) if site.multinode else None
    layout.check_allocation(node.all_hosts, problem.run.get("resources", {}), site.node_types, mn)
    return node


def multinode_note(problem, site) -> list:
    """driver.log: whether several nodes are allowed, when run.toml says allow_multinode (an override is
    written down with the probe's basis and the risk)."""
    res = problem.run.get("resources", {})
    if "allow_multinode" not in res:
        return []
    mn = layout.multinode_of({"multinode": site.multinode}) if site.multinode else None
    return [layout.permitted(res, mn)[1] or f"allow_multinode = {str(res['allow_multinode']).lower()}"]


def make_run(problem, site, node, run_dir: Path):
    """The Run with its scratch chosen (execute.choose_scratch) and those directories bound into the container;
    first the run directory's file system checked for hard links (execute.check_hard_links)."""
    execute.check_hard_links(run_dir)
    scratch, shared, notes = execute.choose_scratch(site, node, run_dir)
    node.shared = shared and node.multi
    site = execute.with_binds(site, [scratch, run_dir])     # run_dir: the solver's checkpoint/
    r = skydive.Run(problem, site, node, run_dir, scratch)
    r.scratch_info = {"dir": str(scratch), "shared": shared, "transfer":
                      "none" if node.shared else (site.transfer if node.multi else "none")}
    r.log(f"site {site.name} ({site.site_dir}); runs {site.runs}; driver python {site.python or 'none'}")
    for n in notes:
        r.log(n)
    if site.scratch_basis:
        r.log(site.scratch_basis)
    return r


def start_lines(problem, site, run_site, node) -> list:
    """driver.log at a run's or a resume's start: the multinode rule, each node's MPI device, each node's route to
    the runs file system, and how the copies reach each node."""
    return (multinode_note(problem, site) + execute.net_report(run_site, node) + execute.runs_network(run_site, node)
            + execute.copy_report(run_site, node) + execute.runs_measured_note(run_site))


def where(node) -> str:
    each = f"{node.node_type}, {node.cores} physical cores, {node.mem_mb} MB"
    if node.multi:
        return f"{len(node.hosts)} nodes {', '.join(node.hosts)} ({each} each; the driver on {node.host})"
    return f"{node.host} ({each})"


def cmd_run(args):
    problem = inputs.load(args.run_toml)
    site = execute.load_site(execute.site_name(problem.run.get("exec", {})))
    node = here(problem, site)
    run_dir = _new_run_dir(problem, site)
    problem = inputs.load(run_dir / "run.toml")
    r = make_run(problem, site, node, run_dir)
    r.log(f"run {run_dir.name} on {where(node)}")
    for line in start_lines(problem, site, r.site, node):
        r.log(line)
    given = problem.run["skydive"].get("block_timings")
    if given:   # the block costs of an earlier run: no timing run, that run's block mapping
        r.ck.mkdir(parents=True, exist_ok=True)
        shutil.copy(given, r.ck / "block_timings")
        r.log(f"block_timings from {given} (sha256 {hashlib.sha256(Path(given).read_bytes()).hexdigest()}): "
              f"the solver's first call makes no timing run")
    note_image(r)
    r.log("run.toml:\n" + (run_dir / "run.toml").read_text())
    s = skydive.start_state(problem)
    for w in problem.warnings + inputs.spectrum_warnings(problem, s["x"]):   # the values' warnings at the start
        r.log(f"warning: {w}")
    r.loop(s["x"], s["n"], s["bfgs"], s["boundary_on"], s["procs"])
    write_summary(run_dir)
    return run_dir


def cmd_resume(args):
    run_dir = Path(args.run_dir).resolve()
    problem = inputs.load(run_dir / "run.toml")
    site = execute.load_site(execute.site_name(problem.run.get("exec", {})))
    node = here(problem, site)
    r = make_run(problem, site, node, run_dir)
    hot = place_checkpoint(r, run_dir, last_record(run_dir).get("procs"))   # resume_state's procs
    s = skydive.resume_state(problem, run_dir, hot)
    if not hot:
        r.log("resume: cold start: its first solver call gets the gap upper limit (keep it at 1e-12 or below)")
        last = last_record(run_dir)
        if r.node.multi or len(last.get("nodes", [last["node"]])) > 1 or \
                layout.node_type_of(last["node"], site.node_types) != r.node.node_type:
            s["procs"] = None           # another layout: the processes from the bench again
    r.log(f"resume on {where(node)} at step {s['n'] + 1}; run.toml now:\n" + (run_dir / "run.toml").read_text())
    for line in start_lines(problem, site, r.site, node):
        r.log(line)
    note_image(r)
    (run_dir / "result.json").unlink(missing_ok=True)
    r.sdp_mem = s["pmp2sdp"]           # pmp2sdp's k, measured anew only on another node type (docs/en/interfaces.md 9)
    r.loop(s["x"], s["n"], s["bfgs"], s["boundary_on"], s["procs"])
    write_summary(run_dir)


def records(run_dir: Path) -> list:
    return [json.loads(l) for l in (run_dir / "solves.jsonl").read_text().splitlines() if l.strip()]


def last_record(run_dir: Path) -> dict:
    return records(run_dir)[-1]


def before_kept(ck: Path) -> bool:
    """A run's checkpoint/ from before kept/ (80f63f6 and earlier): then the run's copy of its last call's
    generation, saved after the call."""
    return (ck / "checkpoint.json").exists() and not (ck / "kept").exists()


def old_layout(run_dir: Path, site) -> dict:
    """The layout of such a checkpoint/: its layout.json, or for one saved before there was one, one node
    with the last record's node and processes."""
    f = run_dir / "checkpoint" / "layout.json"
    if f.exists():
        return {k: v for k, v in json.loads(f.read_text()).items() if k != "generation"}
    last = last_record(run_dir)
    return {"nodes": [last["node"]], "node_type": layout.node_type_of(last["node"], site.node_types), "ppn": last["procs"]}


def resume_layout(run_dir: Path, site):
    """The layout {nodes, node_type, ppn} of the generation a resume starts from, or None (none kept)."""
    ck = run_dir / "checkpoint"
    if before_kept(ck):
        return old_layout(run_dir, site)
    kept = execute.kept_for(ck, len(records(run_dir)))
    return json.loads((kept / "layout.json").read_text()) if kept else None


def place_checkpoint(r, run_dir: Path, ppn) -> bool:
    """The run's checkpoint/ set back to the generation kept for its last record that has one
    (execute.restore_generation) when this job has that generation's layout: nodes, node type and ppn, the
    solver processes a node the resume will use (the binary checkpoint holds each rank's local blocks); else
    emptied, and the solver starts from scratch. Returns hot; driver.log says why."""
    ck, n = r.ck, len(records(run_dir))
    if before_kept(ck):
        execute.keep_generation(ck, n, old_layout(run_dir, r.site))
        r.log(f"resume: checkpoint/ from before kept/: its generation {execute.generation(ck)} kept for record {n}")
    kept = execute.kept_for(ck, n)
    if kept is None:
        r.log("resume: no checkpoint kept")
        shutil.rmtree(ck, ignore_errors=True)
        return False
    then = json.loads((kept / "layout.json").read_text())
    shape = f"{len(then['nodes'])} x {then['node_type']} x {then['ppn']} processes"
    if (len(then["nodes"]), then["node_type"], then["ppn"]) != (len(r.node.all_hosts), r.node.node_type, ppn):
        r.log(f"resume: layout changed: the checkpoint's {shape}, now {len(r.node.all_hosts)} x {r.node.node_type} "
              f"x {ppn} processes; the binary checkpoint holds each rank's local blocks")
        shutil.rmtree(ck, ignore_errors=True)
        return False
    g = execute.restore_generation(ck, kept)
    r.log(f"resume: checkpoint generation {g} kept for record {kept.name}, {shape}")
    return True


def sbatch_command(site, plan, what: str, target: Path, time_limit: str = None) -> list:
    """sbatch for one run or resume from the site alone (no batch script of its own): whole nodes, physical
    cores, the site's [slurm] options (time_limit in place of its time), the plan's nodes; the job sources the
    site's setup and starts the driver with the site's python."""
    if not site.python:
        raise SystemExit(f"site {site.name}: no python (the driver needs Python >= 3.11): the site's python, python "
                         f"in bootlab.toml, or <install root>/python (execute.site_paths)")
    s = site.slurm
    cmd = ["sbatch", "--job-name=bootlab", "--exclusive", "--hint=nomultithread", "--mem=0",
           f"--time={time_limit or s.get('time', '48:00:00')}", f"--output={Path(site.runs) / 'slurm-%j.out'}"]
    cmd += [f"--{k}={s[k]}" for k in ("partition", "account", "qos") if s.get(k)] + list(s.get("extra", []))
    cmd += layout.sbatch_args(plan) or ["--nodes=1"]
    setup = f". {shlex.quote(str(site.site_dir / site.setup))}; " if site.setup else ""
    wrap = (f"set -eu; {setup}export PYTHONPATH={shlex.quote(str(execute.REPO / 'src'))}; "
            f"exec {shlex.quote(site.python)} -m bootlab.cli {what} {shlex.quote(str(target))}")
    return cmd + ["--wrap", wrap]


def cmd_submit(args, what: str = None, time_limit: str = None):
    """On the login node: choose the nodes (layout.choose_nodes, or for --resume the checkpoint's layout) and
    sbatch run.sbatch with them; --dry prints the command only. what: the job's bootlab command (run, resume)."""
    if args.resume:
        run_dir = Path(args.target).resolve()
        problem = inputs.load(run_dir / "run.toml")
    else:
        problem = inputs.load(args.target)
    site = execute.load_site(execute.site_name(problem.run.get("exec", {})))
    mn = layout.multinode_of({"multinode": site.multinode}) if site.multinode else \
        layout.Multinode(prefer=list(site.node_types), peers=[], max_nodes=1, min_efficiency=1.0, min_gain=1.0,
                         allowed=False, basis="the site has no [multinode]")
    out = subprocess.run(["sinfo", "-N", "-h", "-o", "%N %T"], capture_output=True, text=True, check=True).stdout
    snapshot = sorted({tuple(line.split()[:2]) for line in out.splitlines() if line.strip()})
    then = resume_layout(run_dir, site) if args.resume else None
    if then:
        plan = layout.resume_plan(then, site.node_types, mn, snapshot, problem.run.get("resources", {}))
    else:
        benches = {t: bench.load_bench(site.bench_dir, t) for t in site.node_types}
        plan = layout.choose_nodes(problem.run.get("resources", {}), site.node_types, mn, benches,
                                   layout.problem_key(problem), snapshot, site.transport)
    cmd = sbatch_command(site, plan, what or ("resume" if args.resume else "run"), Path(args.target).resolve(),
                         time_limit)
    for line in plan.reasons:
        print(f"# {line}")
    print(" ".join(cmd))
    if not args.dry:
        Path(site.runs).mkdir(parents=True, exist_ok=True)     # sbatch's --output goes there
        subprocess.run(cmd, check=True)


def cmd_regress(args):
    """bootlab regress: on the login node, the regression's inputs (examples/ising, three
    steps) in <runs>/regress-input-<time>/ and a job of one whole node that runs them and checks them against
    bt r3.8's steps (regress.check); in the job (run), the run and the check; check, a run directory anywhere.
    PASS or FAIL in the job's output and in <run dir>/regress.json; FAIL exits 1."""
    if args.action == "check" or args.action == "run":
        run_dir = Path(args.target) if args.action == "check" else cmd_run(argparse.Namespace(run_toml=args.target))
        ok, lines, report = regress.check(run_dir.resolve())
        print("\n".join(lines))
        (run_dir / "regress.json").write_text(json.dumps(report, indent=1) + "\n")
        raise SystemExit(0 if ok else 1)
    site = execute.load_site(execute.site_name({"site": args.site} if args.site else {}))
    run_toml = regress.input_dir(Path(site.runs) / f"regress-input-{time.strftime('%Y%m%d-%H%M%S')}", args.site,
                                 args.node and [args.node])
    print(f"# inputs {run_toml.parent}")
    cmd_submit(argparse.Namespace(target=str(run_toml), resume=False, dry=args.dry), what="regress run",
               time_limit=args.time)


def short(v, digits=8):
    """A number of the records cut to `digits` significant digits for summary.md (docs/en/interfaces.md section 12)."""
    if v is None:
        return ""
    s = str(v)
    return s if len(s) <= digits + 2 else f"{float(s):.{digits}g}"


def write_summary(run_dir: Path):
    run_dir = Path(run_dir)
    with open(run_dir / "run.toml", "rb") as f:
        run = tomllib.load(f)
    lines = [json.loads(l) for l in (run_dir / "solves.jsonl").read_text().splitlines() if l.strip()] \
        if (run_dir / "solves.jsonl").exists() else []
    result = json.loads((run_dir / "result.json").read_text()) if (run_dir / "result.json").exists() else None
    coords = run["coords"]["names"]
    out = [f"# {run_dir.name}", "",
           f"- model: {run['inputs']['model']}, Lambda = {run['numerics']['lambda']}",
           f"- start: {run['skydive']['start']}",
           f"- result: {skydive.ending(result['status'], result['reason'], result.get('n_bound')) if result else 'not finished'}"]
    if result and result.get("primal") is not None:
        out.append(f"- last solve: N {short(result['navigator']) or 'not computed (optimal): N in [dual, primal]'}, "
                   f"primal {short(result['primal'])}, dual {short(result['dual'])}, gap {short(result['gap'])}")
    out.append("")
    if result:
        out += ["| coordinate | end point |", "|---|---|"] + [f"| {c} | {v} |" for c, v in zip(coords, result["x"])] + [""]
    head = ["step", *coords, "N", "primal", "dual", "gap", "verdict", "iterations", "pmp s", "sdp s", "dive s"]
    out += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for l in lines:
        sec = l.get("seconds", {})
        n = None if l["verdict"] == "optimal" else l.get("navigator")   # records before c7be6be wrote 0 here
        row = [l["step"], *[short(v) for v in l["x"]], short(n), short(l.get("primal")), short(l.get("dual")),
               short(l.get("gap")), l["verdict"], l.get("iterations") or "",
               sec.get("pmp", ""), sec.get("sdp", ""), sec.get("dive", "")]
        out.append("| " + " | ".join(map(str, row)) + " |")
    (run_dir / "summary.md").write_text("\n".join(out) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bootlab")
    sub = ap.add_subparsers(required=True)
    p = sub.add_parser("run")
    p.add_argument("run_toml")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("resume")
    p.add_argument("run_dir")
    p.set_defaults(fn=cmd_resume)
    p = sub.add_parser("submit", help="choose the nodes and sbatch run.sbatch (on the login node)")
    p.add_argument("target", help="run.toml, or with --resume the run directory")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry", action="store_true", help="print the sbatch command only")
    p.set_defaults(fn=cmd_submit)
    p = sub.add_parser("summary")
    p.add_argument("run_dir")
    p.set_defaults(fn=lambda a: write_summary(a.run_dir))
    p = sub.add_parser("regress", help="Ising Lambda 11, three steps, against bt r3.8: bin/bootlab regress "
                                       "[submit] (login node) | run (in the job) | check <run dir>")
    p.add_argument("action", nargs="?", default="submit", choices=["submit", "run", "check"])
    p.add_argument("target", nargs="?", help="run: the run.toml; check: the run directory")
    p.add_argument("--site", help="the site (default: bootlab.toml's)")
    p.add_argument("--node", help="the node to run on (default: where Slurm puts the job)")
    p.add_argument("--time", default="01:00:00", help="the job's time limit")
    p.add_argument("--dry", action="store_true", help="print the sbatch command only")
    p.set_defaults(fn=cmd_regress)
    p = sub.add_parser("probe", help="the cluster probe: plan | node | merge (bootlab.probe)", add_help=False)
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p.set_defaults(fn=lambda a: probe.main(a.rest))
    p = sub.add_parser("site", help="a site from the probe: gen | diff (bootlab.sitegen)", add_help=False)
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p.set_defaults(fn=lambda a: sitegen.main(a.rest))
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
