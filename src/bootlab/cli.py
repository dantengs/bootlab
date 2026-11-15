"""bootlab run <run.toml> | resume <run dir> | submit [--resume] [--dry] [--systemd] <run.toml | run dir> | summary
<run dir> | stop <run dir> | bench <run.toml> [--node <host>] | regress [--node <host>] [--dry] [--wait] | probe ... |
site ....

A run is a Skydive, or a scan (one solve per point, one node) when run.toml has [scan] (built-in models,
docs/en/interfaces.md section 19).

Without Slurm (launcher local, execute.site_launcher) submit starts the driver in the background on this machine
and stop ends it (README: a workstation without Slurm).

run and resume work inside a Slurm allocation (sites/<site>/run.sbatch) of one node, or of several of one type
for the solver (run.toml [resources]); submit, on the login node, chooses the nodes and calls sbatch; summary
works anywhere.
"""
import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import tomllib
from pathlib import Path

from . import bench, benchrun, builtin, execute, inputs, layout, probe, regress, scan, sitegen, skydive


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
    if isinstance(r, builtin.Skydive):       # a built-in model's generator: in the image (container = true) or on the host
        r.log(f"image {r.site.sif} sha256 {r.sif_sha256}; generator {r.gen['program']} "
              f"{'in it' if r.gen.get('container') else 'on the host'}; pmp2sdp {r.pmp2sdp}, solver {r.dynamical_sdp}")
    else:
        r.log(f"image {r.site.sif} sha256 {r.sif_sha256}; generator {r.site.generator} in it")


def here(problem, site) -> execute.Node:
    """This job's nodes, checked against run.toml [resources] and the site before anything runs; the site's
    environment applied to the driver's first, so every step inherits it."""
    execute.apply_env(site)
    launcher, why = execute.site_launcher(site)
    node = execute.Node.here(site.node_types, launcher, site.mem_gb)
    node.launcher_why = why
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
    r = (builtin.Skydive if problem.builtin else skydive.Run)(problem, site, node, run_dir, scratch)
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
    local = [f"physical cores {node.cores} (cpus {','.join(map(str, node.cpus))}), memory {node.mem_mb} MB "
             f"available at the start" + (f" (at most the site's mem_gb {site.mem_gb})" if site.mem_gb else "")] \
        if node.launcher == "local" else []
    return ([f"launcher {node.launcher}: {node.launcher_why}"] + local
            + multinode_note(problem, site) + execute.net_report(run_site, node) + execute.runs_network(run_site, node)
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
    if node.launcher == "local":
        execute.local_lock(Path(site.runs), f"run {args.run_toml}")
    run_dir = _new_run_dir(problem, site)
    if node.launcher == "local":
        execute.write_driver_pid(run_dir)
    problem = inputs.load(run_dir / "run.toml")
    if "scan" in problem.run:
        return run_scan(problem, site, node, run_dir, "run")
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
    if node.launcher == "local":
        old = run_dir / "driver.pid"
        if old.exists():
            d = json.loads(old.read_text())
            if d.get("host") == node.host and d["pgid"] != os.getpgid(0) and execute.group_alive(d["pgid"]):
                raise SystemExit(f"{run_dir}: its driver (process group {d['pgid']}) still runs: bin/bootlab stop first")
        execute.local_lock(Path(site.runs), f"resume {run_dir}")
        execute.write_driver_pid(run_dir)
    if "scan" in problem.run:
        return run_scan(problem, site, node, run_dir, "resume")
    r = make_run(problem, site, node, run_dir)
    hot = place_checkpoint(r, run_dir, last_record(run_dir).get("procs"))   # resume_state's procs
    s = skydive.resume_state(problem, run_dir, hot)
    for w in s.get("warnings", []):
        r.log(f"warning: {w}")
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


def run_scan(problem, site, node, run_dir: Path, how: str):
    """A scan's run or resume: one node; its scratch chosen as a Skydive's (execute.choose_scratch)."""
    if node.multi:
        raise SystemExit(f"a scan runs on one node; this job has {len(node.hosts)}")
    scratch, _, notes = execute.choose_scratch(site, node, run_dir)
    site = execute.with_binds(site, [scratch, run_dir])
    s = scan.Scan(problem, site, node, run_dir, scratch)
    s.log(f"{how} {run_dir.name} on {where(node)}, from point {s.done()}")
    for line in [f"launcher {node.launcher}: {node.launcher_why}", f"scratch {scratch}", *notes]:
        s.log(line)
    s.log("run.toml:\n" + (run_dir / "run.toml").read_text())
    for w in problem.warnings:
        s.log(f"warning: {w}")
    (run_dir / "result.json").unlink(missing_ok=True)
    s.run()
    write_summary(run_dir)
    return run_dir


def records(run_dir: Path) -> list:
    return [json.loads(l) for l in (run_dir / "solves.jsonl").read_text().splitlines() if l.strip()]


def last_record(run_dir: Path) -> dict:
    return records(run_dir)[-1]


def resume_records(run_dir: Path) -> int:
    """The records whose kept generation a resume may load: all, but for a last record in state confirm (its call
    is done again from the generation before it, skydive.resume_state) the ones before it."""
    lines = records(run_dir)
    return len(lines) - (lines[-1].get("precision_state") == "confirm" if lines else 0)


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
    kept = execute.kept_for(ck, resume_records(run_dir))
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
    kept = execute.kept_for(ck, resume_records(run_dir))
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
    if getattr(args, "nodes", None):          # bootlab bench --node: that node, whatever run.toml says
        problem.run["resources"] = {"nodes": args.nodes}
    site = execute.load_site(execute.site_name(problem.run.get("exec", {})))
    launcher, why = execute.site_launcher(site)
    if launcher == "local":
        return local_submit(site, what or ("resume" if args.resume else "run"), Path(args.target).resolve(),
                            args.dry, getattr(args, "systemd", False), why)
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


def cmd_bench(args):
    """bootlab bench <run.toml> [--node <host>] [--time] [--dry] (login node): one job of one whole node runs
    `bench run`, which measures the generator, pmp2sdp and the solver on the run's problem (benchrun) and adds
    the rows to the site's bench file of that node type."""
    if len(args.words) > 2 or (len(args.words) == 2 and args.words[0] != "run"):
        raise SystemExit("bootlab bench <run.toml> [--node <host>] | bench run <run.toml>")
    args.action, args.target = ("run", args.words[1]) if len(args.words) == 2 else ("submit", args.words[0])
    if args.action == "submit":
        return cmd_submit(argparse.Namespace(target=args.target, resume=False, dry=args.dry, systemd=False,
                                             nodes=[args.node] if args.node else None),
                          what="bench run", time_limit=args.time)
    problem = inputs.load(args.target)
    site = execute.load_site(execute.site_name(problem.run.get("exec", {})))
    node = here(problem, site)
    if node.launcher == "local":
        execute.local_lock(Path(site.runs), f"bench {args.target}")
    work = Path(site.runs) / f"bench-{problem.run['name']}-{time.strftime('%Y%m%d-%H%M%S')}"
    work.mkdir(parents=True)
    scratch, _, notes = execute.choose_scratch(site, node, work)
    site = execute.with_binds(site, [scratch, work])

    def log(msg):
        line = f"{execute.now()} {msg}"
        print(line, flush=True)
        with open(work / "bench.log", "a") as f:
            f.write(line + "\n")
    for n in notes:
        log(n)
    log(f"bench of {problem.model_name} Lambda {problem.numerics['lambda']} on {node.host} "
        f"({node.node_type}, {node.cores} physical cores)")
    try:
        rows = benchrun.measure(problem, site, node, work, Path(scratch), log)
    finally:
        execute.clear(site, node, Path(scratch))
    if not rows["solver"]:
        raise SystemExit(f"bench: no solver row measured (see {work})")
    path = benchrun.append_rows(site.bench_dir, node.node_type, node.cores, rows)
    log(f"rows added to {path}: " + ", ".join(f"{len(v)} {k}" for k, v in rows.items()))


def cmd_regress(args):
    """bootlab regress: on the login node, the regression's inputs (examples/ising, three
    steps) in <runs>/regress-input-<time>-<pid>/ (two in the same second apart) and a job of one whole node that runs them and checks them against
    bt r3.8's steps (regress.check); in the job (run), the run and the check; check, a run directory anywhere.
    PASS or FAIL in the job's output and in <run dir>/regress.json; FAIL exits 1."""
    if args.action == "check" or args.action == "run":
        run_dir = Path(args.target) if args.action == "check" else cmd_run(argparse.Namespace(run_toml=args.target))
        ok, lines, report = regress.check(run_dir.resolve())
        print("\n".join(lines))
        (run_dir / "regress.json").write_text(json.dumps(report, indent=1) + "\n")
        raise SystemExit(0 if ok else 1)
    site = execute.load_site(execute.site_name({"site": args.site} if args.site else {}))
    run_toml = regress.input_dir(Path(site.runs) / f"regress-input-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}",
                                 args.site,
                                 args.node and [args.node])
    print(f"# inputs {run_toml.parent}")
    if args.wait:              # local mode: run it here and print PASS or FAIL
        if execute.site_launcher(site)[0] != "local":
            raise SystemExit("--wait is for local mode (on a cluster the job runs it)")
        return cmd_regress(argparse.Namespace(action="run", target=str(run_toml)))
    cmd_submit(argparse.Namespace(target=str(run_toml), resume=False, dry=args.dry), what="regress run",
               time_limit=args.time)


def cmd_selftest(args):
    """bootlab selftest: every unit test under the install root's tests/ with this Python (no cluster needed);
    exits 1 when one fails. install.sh runs it after its own checks and only warns on a failure."""
    import unittest
    os.chdir(execute.REPO)
    prog = unittest.main(module=None, exit=False, verbosity=1,
                         argv=["bootlab selftest", "discover", "-s", str(execute.REPO / "tests")])
    raise SystemExit(0 if prog.result.wasSuccessful() else 1)


BOOT = "import sys; sys.path.insert(0, sys.argv.pop(1)); from bootlab.cli import main; main(sys.argv[1:])"


def local_submit(site, what: str, target: Path, dry: bool, systemd: bool, why: str):
    """bootlab submit on a machine without Slurm: the driver started in the background in
    a session of its own (setsid; its process group is what bootlab stop reaches), output to
    <runs>/local-<what>-<time>.log; with systemd, as a transient user unit (it outlives the login only with
    loginctl enable-linger; some systems end a session's processes at logout, KillUserProcesses)."""
    runs = Path(site.runs)
    runs.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log = runs / f"local-{what.replace(' ', '-')}-{stamp}.log"
    cmd = [site.python or sys.executable, "-I", "-c", BOOT, str(execute.REPO / "src"), *what.split(), str(target)]
    if systemd:
        keep = {k: v for k, v in os.environ.items()
                if k in ("PATH", "HOME", "USER", "LANG", "MODULEPATH", "BOOTLAB_RUNS") or k.startswith("APPTAINER")}
        cmd = (["systemd-run", "--user", "--quiet", "--collect", "--unit", f"bootlab-{stamp}",
                "-p", f"WorkingDirectory={os.getcwd()}", "-p", f"StandardOutput=append:{log}",
                "-p", f"StandardError=append:{log}"] + [a for k, v in keep.items() for a in ("--setenv", f"{k}={v}")]
               + cmd)
    print(f"# launcher local: {why}")
    print(" ".join(shlex.quote(c) for c in cmd))
    if dry:
        return
    held = execute.lock_holder(runs)
    if held:
        raise SystemExit(f"another local run is going: {held}")
    if systemd:
        subprocess.run(cmd, check=True)
        print(f"started as the user unit bootlab-{stamp}; log {log}")
        return
    with open(log, "ab") as out:
        p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                             start_new_session=True)
    print(f"started: process group {p.pid}, log {log}; bin/bootlab stop <run dir> ends it")


def cmd_stop(args):
    """bootlab stop <run dir> (local mode): while the solver runs, SIGTERM to its ranks only, which write the
    checkpoint and end the call ("SIGTERM signal received"), so the driver records the step as unfinished and
    ends (mpirun given a SIGTERM would SIGKILL the ranks a few seconds later); before that, SIGTERM to the
    driver's process group (the step is done again from the last checkpoint on resume). Waits up to --wait
    seconds for the driver's group to end."""
    run_dir = Path(args.run_dir).resolve()
    d = json.loads((run_dir / "driver.pid").read_text())
    host = socket.gethostname().split(".")[0]
    if d["host"] != host:
        raise SystemExit(f"{run_dir}: its driver runs on {d['host']}, this is {host}")
    if not execute.group_alive(d["pgid"]):
        print(f"{run_dir}: no driver runs (process group {d['pgid']})")
        return
    problem = inputs.load(run_dir / "run.toml")
    site = execute.load_site(execute.site_name(problem.run.get("exec", {})))
    ranks = execute.solver_ranks(d["pid"], site.dynamical_sdp)
    if ranks:
        for p in ranks:
            os.kill(p, signal.SIGTERM)
        print(f"SIGTERM to the solver's {len(ranks)} ranks: they write the checkpoint and end the call")
    else:
        os.killpg(d["pgid"], signal.SIGTERM)
        print(f"SIGTERM to the driver's process group {d['pgid']} (no solver running)")
    t = time.time() + args.wait
    while time.time() < t and execute.group_alive(d["pgid"]):
        time.sleep(1)
    print(f"{run_dir}: " + ("still running after the wait" if execute.group_alive(d["pgid"]) else "stopped;"
                            " bin/bootlab submit --resume continues it"))


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
    if "scan" in run:
        out = [f"# {run_dir.name}", "", f"- model: {run['inputs']['model']}, Lambda = {run['numerics']['lambda']}",
               f"- result: {result['status'] + ': ' + result['reason'] if result else 'not finished'}", ""]
        short_lines = [{**l, **{k: short(l.get(k)) for k in ("primal", "dual", "gap")}} for l in lines]
        (run_dir / "summary.md").write_text("\n".join(out + scan.summary_lines(run, short_lines)) + "\n")
        return
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
    changes = [l for l in lines if l.get("precision_note")]
    if changes:      # mixed precision (precision.decide): every change of precision or state, in order
        out += ["", "Precision:", ""] + [
            f"- step {l['step']}: {l['precision']} bits, next {l['precision_next']} bits"
            + (f" ({l['precision_state']})" if "precision_state" in l else "") + f": {l['precision_note']}"
            for l in changes]
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
    p.add_argument("--systemd", action="store_true", help="local mode: start the driver as a transient user unit")
    p.set_defaults(fn=cmd_submit)
    p = sub.add_parser("stop", help="local mode: end a run (its solver writes the checkpoint)")
    p.add_argument("run_dir")
    p.add_argument("--wait", type=int, default=600, help="seconds to wait for the driver to end")
    p.set_defaults(fn=cmd_stop)
    p = sub.add_parser("summary")
    p.add_argument("run_dir")
    p.set_defaults(fn=lambda a: write_summary(a.run_dir))
    p = sub.add_parser("bench", help="measure the solver on a run's problem on one node: bin/bootlab bench "
                                     "<run.toml> [--node H] (login node) | run <run.toml> (in the job)")
    p.add_argument("words", nargs="+", metavar="[run] run.toml",
                   help="the run.toml whose problem (model, Lambda, start point) is measured; run: in the job")
    p.add_argument("--node", help="the node to measure (default: where Slurm puts the job); waits when busy")
    p.add_argument("--time", default="02:00:00", help="the job's time limit")
    p.add_argument("--dry", action="store_true", help="print the sbatch command only")
    p.set_defaults(fn=cmd_bench)
    p = sub.add_parser("regress", help="Ising Lambda 11, three steps, against bt r3.8: bin/bootlab regress "
                                       "[submit] (login node) | run (in the job) | check <run dir>")
    p.add_argument("action", nargs="?", default="submit", choices=["submit", "run", "check"])
    p.add_argument("target", nargs="?", help="run: the run.toml; check: the run directory")
    p.add_argument("--site", help="the site (default: bootlab.toml's)")
    p.add_argument("--node", help="the node to run on (default: where Slurm puts the job)")
    p.add_argument("--time", default="01:00:00", help="the job's time limit")
    p.add_argument("--dry", action="store_true", help="print the sbatch command only")
    p.add_argument("--wait", action="store_true", help="local mode: run here and wait for PASS or FAIL")
    p.set_defaults(fn=cmd_regress)
    p = sub.add_parser("selftest", help="every unit test of the installation (no cluster needed)")
    p.set_defaults(fn=cmd_selftest)
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
