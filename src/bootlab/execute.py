"""The execution layer: start the image's programs with srun inside the allocation and read their output.

Every program runs as one `srun --exact` step with its cores and memory given explicitly (docs/en/interfaces.md
section 8; concurrent steps need --mem). MPI programs go through bootlab's rank_env.sh (the
network device per node from the site's [mpi]). Nothing is retried: a program that fails ends the step as
aborted.
"""
import io
import json
import os
import re
import secrets
import shlex
import tarfile
import shutil
import socket
import subprocess
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path

from .layout import node_type_of

REPO = Path(__file__).resolve().parents[2]

STEPPED = {"met criteria to update sdp input files"}
OPTIMAL = {"found primal-dual optimal solution", "found primal feasible solution", "found dual feasible solution",
           "primal feasible jump detected", "dual feasible jump detected"}
UNFINISHED = {"maxIterations exceeded", "maxRuntime exceeded", "SIGTERM signal received"}
STALLED = {"maxComplementarity exceeded", "primal step too small", "dual step too small"}


RANK_ENV = Path(__file__).resolve().parent / "rank_env.sh"


def now() -> str:
    """The time bootlab writes (driver.log, result.json): ISO 8601 with this machine's UTC offset, e.g.
    2026-11-13T11:27:05+08:00 (nodes of one cluster can be in different time zones). probe.now is the same for the probe, which runs on Python 3.9."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass
class Site:
    """sites/<name>/site.toml (sites/_template/site.toml explains every key)."""
    name: str
    sif: str
    binds: list
    generator: str    # the generator's path in the image
    scratch: object   # a path, or candidates in order ("$VAR/..." expanded on the nodes, "shared": runs/<run>/work)
    runs: str
    pmp2sdp: str
    dynamical_sdp: str
    driver_reserve_mb: int
    bench_dir: Path
    site_dir: Path = None
    rank_env: str = str(RANK_ENV)                    # the wrapper every MPI rank starts through
    python: str = None                               # the driver's Python (>= 3.11) for bootlab submit (site_paths)
    setup: str = None                                # a shell file in the site directory: apptainer on PATH
    env: dict = field(default_factory=dict)          # the driver's environment, every step inherits it
    env_unset: list = field(default_factory=list)
    mpi: dict = field(default_factory=dict)          # transport (ib, tcp-ipoib, tcp-1g), net_devices (rules)
    slurm: dict = field(default_factory=dict)        # partition, account, time, extra: bootlab submit's sbatch
    node_types: dict = field(default_factory=dict)   # [node_types.<type>]: physical_cores, mem_gb, hosts
    multinode: dict = None                           # [multinode]: layout.Multinode's fields
    scratch_basis: str = None                        # why scratch is what it is (site gen: each way's seconds, probe jobs)
    shared_network: dict = None                      # the runs file system as the probe saw it: server, network, basis
    shared_measured_at: list = None                  # the directories the probe measured the shared file system in
    launcher: str = None                             # slurm | local; None: site_launcher decides
    mem_gb: int = None                               # local: the memory bootlab may use at most (else MemAvailable)
    memory_guard: str = None                         # local: rss (the default) | systemd (a scope per pmp2sdp instance)

    @property
    def transfer(self) -> str:
        """How the driver reaches the job's other nodes: srun (a light step), ssh (with copies over IPoIB)."""
        return (self.multinode or {}).get("transfer", "srun")

    @property
    def transport(self) -> str:
        return self.mpi.get("transport", "ib")


def install_config(root: Path = None) -> dict:
    """<install root>/bootlab.toml (bootlab.example.toml): this installation's own
    settings, so no path of a test cluster is in the code or the shipped files. Keys: site (the site when run.toml
    [exec] names none), image (the sif when the site gives none), runs, python. Relative paths are from the
    install root. No file: {}."""
    root = root or REPO
    f = root / "bootlab.toml"
    if not f.exists():
        return {}
    with open(f, "rb") as fh:
        c = tomllib.load(fh)
    for k in ("image", "runs", "python"):
        if k in c:
            c[k] = str(root / os.path.expandvars(c[k])) if not Path(os.path.expandvars(c[k])).is_absolute() \
                else os.path.expandvars(c[k])
    return c


def install_runs(root: Path = None) -> str:
    """The installation's runs directory before any site: $BOOTLAB_RUNS, bootlab.toml's runs, <install root>/runs
    (site_paths without the site's own runs). probe plan measures the shared file system there by default."""
    root = root or REPO
    return os.environ.get("BOOTLAB_RUNS") or install_config(root).get("runs") or str(root / "runs")


def site_name(run_exec: dict, root: Path = None) -> str:
    """run.toml [exec] site, else bootlab.toml's site."""
    name = run_exec.get("site") or install_config(root).get("site")
    if not name:
        raise SystemExit("no site: run.toml [exec] site, or site in <install root>/bootlab.toml")
    return name


def site_paths(s: dict, root: Path = None) -> dict:
    """The paths a site.toml may leave out, from the installation, in this order:
    sif: the site's, bootlab.toml's image; runs: $BOOTLAB_RUNS, the site's, bootlab.toml's, <install root>/runs;
    python (the driver's, for bootlab submit and the copies' receiver): the site's, bootlab.toml's,
    <install root>/python/bin/python3 when it exists (the Python copied out of the image), else None."""
    root = root or REPO
    c = install_config(root)
    s = dict(s)
    if "sif" not in s:
        if "image" not in c:
            raise SystemExit("no image: the site's sif, or image in <install root>/bootlab.toml")
        s["sif"] = c["image"]
    s["runs"] = os.environ.get("BOOTLAB_RUNS") or s.get("runs") or c.get("runs") or str(root / "runs")
    if "launcher" not in s and c.get("launcher"):
        s["launcher"] = c["launcher"]
    if not s.get("python"):
        own = root / "python" / "bin" / "python3"
        s["python"] = c.get("python") or (str(own) if own.exists() else None)
    return s


def load_site(name: str, root: Path = None) -> Site:
    root = root or REPO
    d = root / "sites" / name
    with open(d / "site.toml", "rb") as f:
        s = site_paths(tomllib.load(f), root)
    bench = d / s.pop("bench", "bench")
    m = s.get("multinode")
    if m:   # {site}: this site's directory
        m["ssh"] = [a.replace("{site}", str(d)) for a in m.get("ssh", [])]
        m["remote_setup"] = m.get("remote_setup", "").replace("{site}", str(d))
    return Site(name=name, bench_dir=bench.resolve(), site_dir=d, **s)


def apply_env(site: Site):
    """The site's environment into the driver's own, once at the start: every step inherits it (as bt's env.sh
    did for the batch script), and the rank wrapper finds the network device rules in BOOTLAB_NET_DEVICES."""
    for k in site.env_unset:
        os.environ.pop(k, None)
    os.environ.update({k: str(v) for k, v in site.env.items()})
    os.environ.update(net_env(site))


def net_env(site: Site) -> dict:
    """What rank_env.sh reads: the device rules, and for ib:auto the site's excluded ports and the address of the
    runs file system's server (its route picks between IB ports of equal rate)."""
    rules = site.mpi.get("net_devices") or []
    if not rules:
        return {}
    env = {"BOOTLAB_NET_DEVICES": " ".join(rules)}
    if "ib:auto" in rules:
        from .probe import mount_of, opt, parse_mountinfo
        m = mount_of(site.runs, parse_mountinfo(Path("/proc/self/mountinfo").read_text())) or {}
        addr = opt(m.get("options", ""), "addr") if m.get("fstype", "").startswith("nfs") else None
        env["BOOTLAB_IB_EXCLUDE"] = " ".join(site.mpi.get("ib_exclude", []))
        if addr:
            env["BOOTLAB_ROUTE_ADDR"] = addr
    return env


def net_report(site: Site, node: "Node") -> list:
    """driver.log lines: each node's network device for MPI and why (rank_env.sh's own choice, so what the ranks
    get). The variables go in the script: a remote shell over ssh does not have the driver's environment."""
    env = net_env(site)
    if not env:
        return []
    assign = " ".join(f"{k}={shlex.quote(v)}" for k, v in {**env, "BOOTLAB_NET_EXPLAIN": "1"}.items())
    out = on_hosts(site, node, f"{assign} bash {shlex.quote(site.rank_env)} true")
    return [f"MPI device {line}" for h in node.all_hosts for line in out[h].splitlines() if line.strip()]


# Each node: the runs file system's type and options, and for NFS the device its server is routed over and that
# device's type (/sys/class/net/<dev>/type: 1 Ethernet, 32 InfiniBand, i.e. IPoIB). The mount is the last line of
# /proc/self/mountinfo on the longest mount point above the path: an automounted NFS sits on its autofs entry at
# the same point (findmnt -T names the autofs one; so does /proc/self/mountstats).
RUNS_NET_SCRIPT = (
    'p=$(realpath -m {runs}); ls "$p" > /dev/null 2>&1; '
    'set -- $(awk -v p="$p" \'{{split($0, h, " - "); split(h[1], a, " "); split(h[2], b, " "); m = a[5]; '
    'if ((p == m || index(p, (m == "/" ? "" : m) "/") == 1) && length(m) >= n) '
    '{{n = length(m); f = b[1]; o = a[6] "," b[3]}}}} END {{print f, o}}\' "${{BOOTLAB_MOUNTINFO:-/proc/self/mountinfo}}"); '
    'f=${{1:-}}; o=,${{2:-}},; a=; pr=; d=; t=; '
    'case $o in *,addr=*) a=${{o##*,addr=}}; a=${{a%%,*}} ;; esac; '
    'case $o in *,proto=*) pr=${{o##*,proto=}}; pr=${{pr%%,*}} ;; esac; '
    'if [ -n "$a" ]; then set -- $(ip route get "$a" 2>/dev/null); '
    'while [ $# -gt 1 ] && [ "$1" != dev ]; do shift; done; d=${{2:-}}; t=$(cat /sys/class/net/$d/type 2>/dev/null); fi; '
    'echo "${{f:--}} ${{a:--}} ${{pr:--}} ${{d:--}} ${{t:--}}"')
NET_KINDS = {"1": "ethernet", "32": "ipoib"}


def runs_network_lines(site: Site, out: dict) -> list:
    """driver.log lines from RUNS_NET_SCRIPT's output (host -> "fstype addr proto dev type"): each node's way to
    the runs file system, and a warning where it is not what the probe saw (site.shared_network): only a
    warning. An autofs map with two server addresses falls back to the slower one when the fast network is down at
    mount time and stays there; whether a run then slows down or hangs is untested."""
    want = site.shared_network or {}
    lines = []
    for h, text in out.items():
        fstype, addr, proto, dev, typ = (text.split() + ["-"] * 5)[:5]
        if not fstype.startswith("nfs"):
            lines.append(f"runs file system on {h}: {fstype}")
            continue
        kind = "rdma" if proto == "rdma" else NET_KINDS.get(typ, f"type {typ}")
        lines.append(f"runs file system on {h}: {fstype} server {addr} over {dev} ({kind})")
        if want and (addr != want.get("server") or kind != want.get("network")):
            lines.append(f"warning: {h}: the runs file system's server is {addr} over {dev} ({kind}); the probe saw "
                         f"{want.get('server')} over {want.get('network')} ({want.get('basis', 'site')}). The mount may "
                         f"have fallen back to another network; whether a run then slows down or hangs is untested; "
                         f"the run goes on")
    return lines


def runs_network(site: Site, node: "Node") -> list:
    out = on_hosts(site, node, RUNS_NET_SCRIPT.format(runs=shlex.quote(site.runs)))
    return runs_network_lines(site, {h: out[h].strip() for h in node.all_hosts})


def runs_measured_note(site: Site) -> list:
    """driver.log at the start: a warning when the runs directory is not in a directory the probe measured the
    shared file system in (site gen's shared_measured_at). On one NFS mount two directories can write 2 GiB
    in 4-9 s and in 0.8 s (two disk arrays behind the server's export), so the scratch choice's shared
    numbers hold only where they were measured. Nothing for a site without the field (hand-written)."""
    if not site.shared_measured_at:
        return []
    runs = os.path.realpath(site.runs)
    if any(runs == m or runs.startswith(m.rstrip("/") + "/") for m in site.shared_measured_at):
        return []
    return [f"warning: the runs directory {runs} is not in the directory the probe measured the shared file system "
            f"in ({', '.join(site.shared_measured_at)}): the scratch choice's shared numbers may not hold for it "
            f"(another disk array behind the same mount); the run goes on"]


def copy_report(site: Site, node: "Node") -> list:
    """driver.log: how node 0's step files reach each other node (a run's log names the
    scratch, the copies and the MPI device of every node). One node: nothing; a shared scratch: no copies."""
    if not node.multi:
        return []
    if node.shared:
        return ["copies: none, the scratch is shared (every node sees node 0's files)"]
    k = site.multinode.get("copy_streams", 1)
    if site.transfer == "ssh":
        return [f"copies to {h}: transfer ssh, {k} tar streams to {ib_address(site, node, h)} (IPoIB, ipoib_prefix "
                f"{site.multinode.get('ipoib_prefix')})" for h in node.others]
    net = site.multinode.get("copy_network")
    return [f"copies to {h}: transfer srun, {k} tar streams to receive.py at {copy_address(site, node, h)} "
            + (f"(copy_network {net})" if net else "(the address of its name)") for h in node.others]


@dataclass
class Node:
    """The node the driver runs on (node 0). A job of several nodes (one type, layout.check_allocation) has
    them in hosts, node 0 first; the others have the same cores and memory. hosts None: a job of one node."""
    host: str
    node_type: str
    cores: int        # physical cores the job may use
    mem_mb: int       # the job's memory on this node
    hosts: list = None
    shared: bool = False   # the run's scratch is on a file system every node sees: nothing is copied between nodes
    launcher: str = "slurm"    # slurm | local (site_launcher)
    cpus: list = None          # local: the first hardware thread of each physical core the driver may use
    launcher_why: str = None   # why that launcher (driver.log)

    @property
    def multi(self) -> bool:
        return bool(self.hosts) and len(self.hosts) > 1

    @property
    def all_hosts(self) -> list:
        return self.hosts if self.multi else [self.host]

    def solo(self):
        """The hosts argument of a step on node 0 alone: in a job of several nodes srun without -w may put
        it on another node; None in a job of one node (the step's command as before)."""
        return [self.host] if self.multi else None

    def spread(self):
        """The hosts argument of a step on all the job's nodes: None in a job of one node."""
        return self.hosts if self.multi else None

    @property
    def others(self) -> list:
        """The nodes whose files the driver reaches over srun or ssh: none when the scratch is shared."""
        return [] if self.shared else [h for h in self.all_hosts if h != self.host]

    @staticmethod
    def here(node_types: dict = None, launcher: str = "slurm", mem_gb: int = None) -> "Node":
        if launcher == "local":    # one machine: its physical cores and the memory available now
            host = socket.gethostname().split(".")[0]
            cpus = physical_cpus(os.sched_getaffinity(0))
            return Node(host, node_type_of(host, node_types or {}), len(cpus), local_mem_mb(mem_gb),
                        launcher="local", cpus=cpus)
        host = socket.gethostname().split(".")[0]
        cpus = os.sched_getaffinity(0)
        cores = set()
        for c in cpus:
            topo = Path(f"/sys/devices/system/cpu/cpu{c}/topology")
            cores.add(((topo / "physical_package_id").read_text().strip(), (topo / "core_id").read_text().strip()))
        return Node(host, node_type_of(host, node_types or {}), len(cores), job_mem_mb(), job_hosts(host))


def scratch_candidates(value) -> list:
    return [value] if isinstance(value, str) else list(value)


def probe_dirs(site: "Site", node: "Node", candidates: list, hosts: list) -> dict:
    """host -> [(resolved path, writable, free GB)] for each candidate that is not "shared": the path with the
    node's environment ($TMPDIR, $USER, ...), whether a directory and a file can be made in it there, its free
    space. One remote call for all the nodes (an external boundary, checked once at the start)."""
    lines = []
    for c in candidates:
        if c != "shared":
            lines.append(f'p=$(eval echo "{c}"); t="$p/.bootlab-probe-$$"; '
                         'if mkdir -p "$t" 2>/dev/null && : > "$t/f" 2>/dev/null; then w=1; else w=0; fi; rm -rf "$t"; '
                         'g=$(df -Pk "$p" 2>/dev/null | awk "NR == 2 {printf \\"%.0f\\", \\$4 / 1048576}"); '
                         'echo "$p|$w|${g:-0}"')
    out = on_hosts(site, node, "\n".join(lines), hosts)
    return {h: [tuple(l.split("|")) for l in out[h].splitlines() if l] for h in hosts}


def choose_scratch(site: "Site", node: "Node", run_dir: Path):
    """The run's scratch directory from site.scratch (one path: as before, no probe); the generator's blocks go
    in it (blocks on /tmp or on NFS made no measurable difference). Candidates in order: the first
    that every node of the job resolves to the same path and can write; "shared" is <run dir>/work on the runs
    file system, which every node sees (nothing copied between nodes, Node.shared). Returns (scratch, shared,
    notes)."""
    cands = scratch_candidates(site.scratch)
    if len(cands) == 1 and cands[0] != "shared":
        scratch = Path(os.path.expandvars(cands[0])) / run_dir.name
        notes = [f"scratch {scratch} (the site's one path)"]
        shared = False
    else:
        found = probe_dirs(site, node, cands, node.all_hosts)
        notes, scratch, shared, i = [], None, False, 0
        for c in cands:
            if c == "shared":
                scratch, shared = run_dir / "work", True
                notes.append(f"scratch {scratch}: shared, on the runs file system every node sees")
                break
            got = [found[h][i] for h in node.all_hosts]
            i += 1
            paths = {p for p, _, _ in got}
            ok = len(paths) == 1 and all(w == "1" for _, w, _ in got)
            notes.append(f"candidate {c}: " + ", ".join(f"{h} {p} {'writable' if w == '1' else 'not writable'} "
                                                        f"{g} GB free" for h, (p, w, g) in zip(node.all_hosts, got)))
            if ok:
                scratch = Path(paths.pop()) / run_dir.name
                notes.append(f"scratch {scratch}")
                break
        if scratch is None:
            raise SystemExit("no scratch candidate is writable on every node:\n  " + "\n  ".join(notes))
    return scratch, shared, notes


def with_binds(site: "Site", dirs: list) -> "Site":
    """The site with the run's own directories bound into the container too (a candidate like $TMPDIR is known
    only on the node)."""
    import dataclasses
    have = [Path(b) for b in site.binds]
    extra = []
    for d in dirs:
        top = Path(d)
        if any(top == b or b in top.parents for b in have):
            continue
        while top.parent != top.parent.parent and not top.exists():
            top = top.parent
        if not any(top == b or b in top.parents for b in have + [Path(x) for x in extra]):
            extra.append(str(top))
    return dataclasses.replace(site, binds=site.binds + extra) if extra else site


def job_hosts(here: str):
    """The job's nodes, this one first (the batch script runs on the allocation's first node); None for a
    job of one node."""
    if int(os.environ.get("SLURM_JOB_NUM_NODES", "1")) == 1:
        return None
    out = subprocess.run(["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]],
                         capture_output=True, text=True, check=True).stdout.split()
    if here not in out:
        raise SystemExit(f"this node {here} is not among the job's nodes {out}")
    return [here] + [h for h in out if h != here]


def job_seconds_left() -> int:
    """Seconds until the Slurm job's time limit, from its EndTime."""
    out = subprocess.run(["scontrol", "show", "job", "-o", os.environ["SLURM_JOB_ID"]],
                         capture_output=True, text=True, check=True).stdout
    end = datetime.strptime(re.search(r"EndTime=(\S+)", out).group(1), "%Y-%m-%dT%H:%M:%S")
    return int((end - datetime.now()).total_seconds())


def job_mem_mb() -> int:
    """The job's memory on this node. --mem=0 (the whole node, as run.sbatch asks) leaves SLURM_MEM_PER_NODE
    unset; the memory is then in the job's AllocTRES (AllocTRES=cpu=256,mem=253089M,...), which is
    the sum over its nodes (one type, the same memory each)."""
    if "SLURM_MEM_PER_NODE" in os.environ:
        return int(os.environ["SLURM_MEM_PER_NODE"])
    out = subprocess.run(["scontrol", "show", "job", "-o", os.environ["SLURM_JOB_ID"]],
                         capture_output=True, text=True, check=True).stdout
    return alloc_mem_mb(out)


def alloc_mem_mb(scontrol_job: str) -> int:
    """A node's memory from `scontrol show job -o`: AllocTRES's mem over NumNodes."""
    amount, unit = re.search(r"AllocTRES=\S*?mem=(\d+(?:\.\d+)?)([MGT])", scontrol_job).groups()
    nodes = int(re.search(r"NumNodes=(\d+)", scontrol_job).group(1))
    return int(float(amount) * {"M": 1, "G": 1024, "T": 1024 * 1024}[unit]) // nodes


def srun_argv(site: Site, argv: list, ntasks: int, cpus: int, mem_mb: int, mpi: bool, container: bool = True,
              hosts: list = None) -> list:
    """One srun step. Programs of the image run in the container; bt's Hyperion executables run on the host.
    hosts: the step's nodes by name, ntasks and mem_mb then per node (Node.solo, Node.spread); None: one node,
    the step's command as in a job of one node."""
    where = ["--nodes=1"] if hosts is None else [f"--nodes={len(hosts)}", "-w", ",".join(hosts)]
    tasks = [f"--ntasks={ntasks}"] if not hosts or len(hosts) == 1 else \
        [f"--ntasks={ntasks * len(hosts)}", f"--ntasks-per-node={ntasks}"]
    cmd = ["srun", "--exact", *where, *tasks, f"--cpus-per-task={cpus}", f"--mem={mem_mb}M",
           "--immediate=10", "--hint=nomultithread", "--cpu-bind=cores", "--kill-on-bad-exit=1"]
    if mpi:
        cmd += [f"--mpi={getattr(site, 'mpi', {}).get('plugin', 'pmix')}", "bash", site.rank_env]
    if not container:
        return cmd + [str(a) for a in argv]
    cmd += ["apptainer", "exec"]
    for b in site.binds:
        cmd += ["--bind", b]
    return cmd + [site.sif] + [str(a) for a in argv]


def start(cmd: list, log: Path, env: dict = None) -> subprocess.Popen:
    """Start a step; its stdout and stderr go to log.out and log.err, the command line to log.cmd. env, if
    given, is the step's whole environment (srun passes it on)."""
    log.parent.mkdir(parents=True, exist_ok=True)
    log.with_suffix(".cmd").write_text(" ".join(map(str, cmd)) + "\n")
    out = open(log.with_suffix(".out"), "w")
    err = open(log.with_suffix(".err"), "w")
    return subprocess.Popen([str(c) for c in cmd], stdout=out, stderr=err, env=env)


def wait_all(procs: dict) -> dict:
    """Wait for named steps; their exit codes."""
    return {name: p.wait() for name, p in procs.items()}


# ---------------------------------------------------------------- output files

_KV = re.compile(r'^\s*([A-Za-z][A-Za-z ]*?)\s*=\s*"?(.*?)"?\s*;?\s*$')


def read_kv(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        m = _KV.match(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def read_matrix(path: Path):
    """Elemental's ASCII matrix: "rows cols", then the entries. Returns (rows, cols, entries as strings)."""
    tokens = path.read_text().split()
    rows, cols = int(tokens[0]), int(tokens[1])
    return rows, cols, tokens[2:2 + rows * cols]


def classify(reason: str) -> tuple:
    """(verdict, category) of a solver's terminateReason (docs/en/interfaces.md section 8)."""
    if reason in STEPPED:
        return "stepped", None
    if reason in OPTIMAL:
        return "optimal", None
    if reason in UNFINISHED:
        return "aborted", "unfinished"
    if reason in STALLED:
        return "aborted", "stalled"
    raise SystemExit(f"unknown terminateReason {reason!r}; add it to execute.py")


@dataclass
class Dive:
    verdict: str                      # stepped | optimal | aborted
    reason: str
    primal: str = None
    dual: str = None
    gap: str = None
    navigator: str = None
    ext_step: list = field(default_factory=list)
    gradient: list = field(default_factory=list)
    hessian: list = field(default_factory=list)   # rows, each a list of strings
    iterations: int = None
    alpha_p: str = None
    alpha_d: str = None
    beta: str = None
    climbed: bool = None
    find_minimum: bool = None
    seconds: dict = field(default_factory=dict)
    call_iterations: int = None       # aborted: the iterations this call made (call_iterations), None unknown


# a row of dynamical_sdp's iteration table: the call's iteration number, the seconds, mu, P-obj (signed), ...
_ITERATION_ROW = re.compile(r"^\s*(\d+)\s+(\d+)\s+\S+\s+[+-]\S+\s", re.M)


def call_iterations(stdout: Path):
    """The iterations a dynamical_sdp call made, from its own output (the iteration table's numbers start at 1
    in every call, a call resumed from a checkpoint too); None without a table."""
    try:
        rows = _ITERATION_ROW.findall(stdout.read_text(errors="replace"))
    except OSError:
        return None
    return max(int(r[0]) for r in rows) if rows else None


def read_dive(out_dir: Path, exit_code: int, stdout: Path = None) -> Dive:
    """The solver's results; for an aborted call also the iterations it made (call_iterations of its stdout)."""
    d = _read_dive(out_dir, exit_code)
    if d.verdict == "aborted" and stdout is not None:
        d.call_iterations = call_iterations(stdout)
    return d


def _read_dive(out_dir: Path, exit_code: int) -> Dive:
    out_txt = out_dir / "out.txt"
    if not out_txt.exists():
        return Dive("aborted", f"failed: dynamical_sdp exit {exit_code}, no out.txt")
    o = read_kv(out_txt)
    reason = o["terminateReason"]
    verdict, category = classify(reason)
    s = read_kv(out_dir / "skydiving_out.txt")
    # At optimal the solver does not compute N: skydiving_out.txt's NavigatorValue 0 is not N (docs/en/interfaces.md section 8).
    d = Dive(verdict, f"{category}: {reason}" if category else reason,
             primal=o.get("primalObjective"), dual=o.get("dualObjective"), gap=o.get("dualityGap"),
             navigator=None if verdict == "optimal" else s.get("NavigatorValue"), alpha_p=s.get("primalStepSize"), alpha_d=s.get("dualStepSize"),
             beta=s.get("beta"), climbed=s.get("climbedQ") not in (None, "0"),
             find_minimum=s.get("findMinimumQ") not in (None, "0"))
    if verdict == "aborted":
        return d
    d.ext_step = read_matrix(out_dir / "externalParamStep.txt")[2]
    d.gradient = read_matrix(out_dir / "gradient.txt")[2]
    _, _, h = read_matrix(out_dir / "hessBFGS.txt")
    k = len(d.ext_step)
    d.hessian = [h[i * k:(i + 1) * k] for i in range(k)]
    d.iterations = int((out_dir / "iterations.txt").read_text().split()[0])
    return d


# ---------------------------------------------------------------- the step's programs

# A step whose memory that --mem pressure cannot reclaim passes this share of its --mem is cancelled:
# pmp2sdp given too little memory stalls instead of failing, so a wave must fail at once.
MEMORY_WATCH = 0.95
SAMPLE_SECONDS = 0.2


def job_cgroup_of(line: str) -> Path:
    """The job's cgroup v2 directory from a /proc/<pid>/cgroup line of a process in one of its steps: the
    directory above the first step_* one. Some Slurm set-ups name it with no job id
    (0::/system.slice/slurmstepd.scope/<id>/step_batch/user/task_0), others job_<id>."""
    parts = Path(line.split("::", 1)[1].strip()).parts
    i = next(i for i, p in enumerate(parts) if p.startswith("step_"))
    return Path("/sys/fs/cgroup", *parts[1:i])


def job_cgroup() -> Path:
    """This job's cgroup directory, from the driver's own /proc/self/cgroup: the driver runs in the batch step,
    the srun steps are step_<n> beside it."""
    return job_cgroup_of(Path("/proc/self/cgroup").read_text())


def srun_steps(job: Path) -> set:
    """The cgroup directories of the job's srun steps (step_0, step_1, ...; not the batch or extern step)."""
    return {d for d in job.glob("step_*") if d.name[5:].isdigit()}


def step_memory(d: Path):
    """(held, peak, limit) of one step in MB, from its user/ cgroup: Slurm sets the step's --mem there as
    memory.max and memory.high, step_<n>/ itself and the task_* below are unlimited. Over
    memory.high the kernel throttles and reclaims instead of killing: how pmp2sdp stalls. held =
    memory.current less the page cache plus shmem (memory.stat's file counts shmem), what --mem pressure cannot
    reclaim; peak = memory.peak, page cache included (an SDP written stays in it); limit = memory.max, None
    when unlimited."""
    d = d / "user"
    stat = dict(line.split() for line in (d / "memory.stat").read_text().splitlines())
    current = int((d / "memory.current").read_text())
    peak = int((d / "memory.peak").read_text())
    limit = (d / "memory.max").read_text().strip()
    return ((current - int(stat["file"]) + int(stat["shmem"])) / 2 ** 20, peak / 2 ** 20,
            None if limit == "max" else int(limit) / 2 ** 20)


def node_used_mb() -> float:
    """MemTotal - MemAvailable of the node, MB: the measure of the bench's sdp_build peaks (multinode mn.py)."""
    m = {k: int(v.split()[0])
         for k, v in (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())}
    return (m["MemTotal"] - m["MemAvailable"]) / 1024


def build_sdps(site: Site, node: Node, pmps: dict, sdp_root: Path, precision: int, log_dir: Path, instances: int,
               procs: int):
    """pmp2sdp for each named PMP (its pmp.nsv), `instances` at a time with `procs` processes each, each step
    with --mem = (node memory - driver reserve) / instances.

    While a wave runs, the steps' cgroups are read every SAMPLE_SECONDS. Returns (name -> SDP directory,
    waves): for each wave its names, held_mb (the most one of its steps held that --mem pressure cannot
    reclaim), cgroup_peak_mb (the largest memory.peak of one step) and node_mb (the node's MemTotal -
    MemAvailable above its level before the wave). A step holding more than MEMORY_WATCH of its --mem is
    cancelled with its whole wave, and the step fails; so does a pmp2sdp that exits non-zero.

    Where the steps' cgroups cannot be read (cgroup v1, another layout: an external boundary) there is no
    watchdog: the wave's "watchdog" is false and held_mb is node_mb, the node's MemTotal - MemAvailable rise."""
    if node.launcher == "local":
        return build_sdps_local(site, node, pmps, sdp_root, precision, log_dir, instances, procs)
    mem = (node.mem_mb - site.driver_reserve_mb) // instances
    try:
        job = job_cgroup()
        job = job if job.is_dir() else None
    except (OSError, IndexError, StopIteration, ValueError):
        job = None
    names = list(pmps)
    sdps, failed, waves = {}, [], []
    for i in range(0, len(names), instances):
        batch = names[i:i + instances]
        before, base = (srun_steps(job) if job else set()), node_used_mb()
        running = {}
        for name in batch:
            sdps[name] = sdp_root / name
            argv = [site.pmp2sdp, "--precision", precision, "--input", pmps[name], "--output", sdps[name]]
            running[name] = start(srun_argv(site, argv, procs, 1, mem, mpi=True, hosts=node.solo()),
                                  log_dir / f"pmp2sdp-{name}")
        wave = {"names": batch, "held_mb": 0.0, "cgroup_peak_mb": 0.0, "node_mb": 0.0}
        over = None
        while over is None and any(p.poll() is None for p in running.values()):
            for d in (srun_steps(job) - before if job else ()):
                try:
                    held, peak, limit = step_memory(d)
                except (OSError, KeyError, ValueError):      # the step ending as it is read
                    continue
                wave["held_mb"] = max(wave["held_mb"], held)
                wave["cgroup_peak_mb"] = max(wave["cgroup_peak_mb"], peak)
                if limit and held > MEMORY_WATCH * limit:
                    over = f"{d.name} held {held:.0f} MB of its {limit:.0f} MB"
            wave["node_mb"] = max(wave["node_mb"], node_used_mb() - base)
            if over is None:
                time.sleep(SAMPLE_SECONDS)
        if over:
            for d in srun_steps(job) - before:
                subprocess.run(["scancel", f"{os.environ['SLURM_JOB_ID']}.{d.name[5:]}"])
            wait_all(running)
            raise StepFailed(f"pmp2sdp {', '.join(batch)}: {over}; the wave was cancelled (an instance needs "
                             f"more memory than its share)")
        failed += [f"{n} exit {c}" for n, c in wait_all(running).items() if c != 0]
        if job is None or wave["held_mb"] == 0:      # no cgroup read: the node's measure stands for the step's
            wave.update({"held_mb": wave["node_mb"], "watchdog": False})
        waves.append({**wave, **{k: round(wave[k]) for k in ("held_mb", "cgroup_peak_mb", "node_mb")}})
    if failed:
        raise StepFailed("pmp2sdp " + ", ".join(failed))
    return sdps, waves


class StepFailed(Exception):
    """A program of the step failed: the step ends as aborted, failed."""


def flag(v) -> str:
    """A run.toml value as a command-line value. Floats never get here: load rejects them."""
    if isinstance(v, bool):
        return "True" if v else "False"
    return str(v)


def run_dynamical_sdp(site: Site, node: Node, procs: int, center: Path, neighbors: list, out_dir: Path,
                      ck_dir: Path, solver: dict, dynamical: dict, state: dict, log: Path) -> int:
    """One dynamical_sdp call. solver: SDPB options by flag name; dynamical: dynamical_sdp's own options;
    state: total_iterations, prev_step, prev_gradient, prev_hessian (flattened), hessian_guess (flattened),
    gap_upper_limit, boundary_direction, box_min, box_max."""
    nsv = out_dir.parent / "new_sdp_dirs.nsv"
    nsv.write_text("".join(f"{p}\0" for p in neighbors))
    argv = [site.dynamical_sdp, "--sdpDir", center, "--newSdpDirs", nsv, "--outDir", out_dir,
            "--checkpointDir", ck_dir]
    for k, v in solver.items():
        argv += [f"--{k}", flag(v)]
    argv += ["--numExternalParams", len(state["box_min"])]
    for k, v in dynamical.items():
        argv += [f"--{k}", flag(v)]
    argv += ["--dualityGapUpperLimit", state["gap_upper_limit"], "--totalIterationCount", state["total_iterations"]]
    argv += ["--boundingBoxMax", *state["box_max"], "--boundingBoxMin", *state["box_min"]]
    if state.get("boundary_direction"):
        argv += ["--findBoundaryDirection", *state["boundary_direction"]]
    argv += ["--useExactHessian", "False"]
    if state.get("prev_step"):
        argv += ["--prevExternalStep", *state["prev_step"], "--prevGradientBFGS", *state["prev_gradient"],
                 "--prevHessianBFGS", *state["prev_hessian"]]
    elif state.get("hessian_guess"):
        argv += ["--prevHessianBFGS", *state["hessian_guess"]]
    mem = node.mem_mb - site.driver_reserve_mb
    if node.others:
        # Every node needs the neighbour list and the out directory in the scratch (the checkpoint directory
        # is the run's, which every node sees). One remote call: a step costs about a second with transfer srun.
        on_hosts(site, node, remote_tar(site, "-C", "/", "-xf", "-", before=f"mkdir -p {out_dir} && "),
                 node.others, small_tar([nsv]))
    return launch(site, node, argv, procs, 1, mem, True, log, hosts=node.spread()).wait()


# ---------------------------------------------------------------- local mode (no Slurm; README)
#
# One workstation, no Slurm: the image's programs start with `apptainer exec`, MPI ones through the image's
# own mpirun (all ranks children of it in one container), bound to the physical cores (the first hardware thread
# of each core; concurrent programs get disjoint cores). The memory is MemAvailable at the start (at most the
# site's mem_gb). The slurm functions above are unchanged: launch() chooses.

LAUNCHERS = ("slurm", "local")
LOCAL_UNSET = ("SLURM_", "PMIX_", "PMI_")   # mpirun in a Slurm job would start its ranks with srun


def site_launcher(site: Site) -> tuple:
    """(launcher, why): the site's or bootlab.toml's launcher when given; else slurm inside a Slurm job
    (SLURM_JOB_ID) or where sbatch is on PATH, local otherwise."""
    if site.launcher:
        if site.launcher not in LAUNCHERS:
            raise SystemExit(f"site {site.name}: launcher {site.launcher!r}: slurm or local")
        return site.launcher, f"the site (or bootlab.toml) says launcher = {site.launcher}"
    if os.environ.get("SLURM_JOB_ID"):
        return "slurm", f"SLURM_JOB_ID is set ({os.environ['SLURM_JOB_ID']}): inside a Slurm job"
    if shutil.which("sbatch"):
        return "slurm", f"sbatch is on PATH ({shutil.which('sbatch')})"
    return "local", "no SLURM_JOB_ID and no sbatch on PATH: a machine without Slurm"


def physical_cpus(cpus, sysdir: Path = Path("/sys/devices/system/cpu")) -> list:
    """The first hardware thread (the lowest cpu number) of each physical core among cpus: one rank or worker a
    core, never a core's second thread."""
    first = {}
    for c in sorted(cpus):
        topo = sysdir / f"cpu{c}" / "topology"
        key = ((topo / "physical_package_id").read_text().strip(), (topo / "core_id").read_text().strip())
        first.setdefault(key, c)
    return sorted(first.values())


def local_mem_mb(cap_gb: int = None, meminfo: Path = Path("/proc/meminfo")) -> int:
    """The memory a local run uses: MemAvailable now (a workstation runs other things), at most cap_gb."""
    m = {k: int(v.split()[0]) for k, v in (line.split(":", 1) for line in meminfo.read_text().splitlines())}
    mb = m["MemAvailable"] // 1024
    return min(mb, cap_gb * 1024) if cap_gb else mb


def local_env() -> dict:
    """The environment of a local program: the driver's without Slurm's and PMIx's variables, and mpirun's
    launcher isolated (Open MPI 4.1.1 in the image has it: every rank on this machine, never through srun)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(LOCAL_UNSET)}
    env["OMPI_MCA_plm"] = "isolated"
    return env


def write_rankfile(path: Path, cpus: list) -> Path:
    """Open MPI's rankfile: rank i on cpu i of the list (physical cpu numbers, rmaps_rank_file_physical; with
    --bind-to hwthread each rank on that hardware thread alone: the first of its core, no hyperthread). mpirun's
    --cpu-set bound every rank to the whole set instead, taskset around mpirun was not honoured (two programs
    landed on the same cores), and logical numbers put a rank on a core's second thread."""
    host = socket.gethostname().split(".")[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"rank {i}={host} slot={c}\n" for i, c in enumerate(cpus)))
    return path


def local_argv(site: Site, argv: list, cpus: list, mpi: bool, rankfile: Path = None) -> list:
    """A program of the image on these cpus: MPI through the image's mpirun with a rankfile (write_rankfile), a
    rank a core of the list; otherwise under the image's taskset (the host needs apptainer only: no host tool
    is assumed, a node had no tar)."""
    image = ["apptainer", "exec"]
    for b in site.binds:
        image += ["--bind", b]
    image += [site.sif]
    if mpi:
        return image + ["mpirun", "-np", str(len(cpus)), "--rankfile", str(rankfile),
                        "--mca", "rmaps_rank_file_physical", "1", "--bind-to", "hwthread"] + [str(a) for a in argv]
    return image + ["taskset", "-c", ",".join(map(str, cpus))] + [str(a) for a in argv]


def launch(site: Site, node: Node, argv: list, ntasks: int, cpus: int, mem_mb: int, mpi: bool, log: Path,
           container: bool = True, hosts: list = None, cores: list = None, wrap: list = None) -> subprocess.Popen:
    """Start one program of a step: srun (srun_argv, unchanged) or, local, on cores (default the first
    ntasks x cpus physical cores); wrap goes before a local command (systemd-run's scope)."""
    if node.launcher != "local":
        return start(srun_argv(site, argv, ntasks, cpus, mem_mb, mpi, container, hosts), log)
    if not container:
        raise SystemExit("local mode runs only the image's programs")
    use = cores if cores is not None else node.cpus[:ntasks * cpus]
    env = local_env()
    if not mpi:    # the generator: Hyperion asks scontrol (the image's stand-in) for $SLURM_JOB_NODELIST's hosts
        env["SLURM_JOB_NODELIST"] = node.host
    rankfile = write_rankfile(log.with_suffix(".rankfile"), use) if mpi else None
    return start((wrap or []) + local_argv(site, argv, use, mpi, rankfile), log, env=env)


def descendants(pid: int, proc: Path = Path("/proc")) -> list:
    """pid and every process below it (by the parent pids of /proc/<n>/stat)."""
    children = {}
    for d in proc.iterdir():
        if d.name.isdigit():
            try:
                stat = (d / "stat").read_text()
            except OSError:
                continue
            ppid = int(stat.rsplit(")", 1)[1].split()[1])
            children.setdefault(ppid, []).append(int(d.name))
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        out.append(p)
        todo += children.get(p, [])
    return out


def tree_memory_mb(pid: int, proc: Path = Path("/proc")) -> tuple:
    """(MB, measure) of a process tree that memory pressure cannot reclaim: the sum of smaps_rollup's
    Pss_Anon + Pss_Shmem (shared pages split between their processes, so the sum counts each once: MPI's sm
    transport is shared memory); without smaps_rollup the sum of RssAnon, which leaves the shared memory out.
    A process that has ended (a zombie: neither of the two) holds no memory and is passed."""
    total, measure = 0, "pss"
    for p in descendants(pid, proc):
        try:
            rollup = (proc / str(p) / "smaps_rollup").read_text()
            kb = dict((l.split(":")[0], int(l.split()[1])) for l in rollup.splitlines() if l.startswith("Pss_"))
            total += kb["Pss_Anon"] + kb["Pss_Shmem"]
        except (OSError, KeyError, IndexError, ValueError):
            try:
                status = (proc / str(p) / "status").read_text()
            except OSError:
                continue                                  # the process ended as it was read
            rss = [l.split()[1] for l in status.splitlines() if l.startswith("RssAnon:")]
            if rss:
                total += int(rss[0])
                measure = "rss_anon"
    return total / 1024, measure


def running(pid: int, proc: Path = Path("/proc")) -> bool:
    """The process exists and is not a zombie (ended, its parent has not waited for it yet)."""
    try:
        return (proc / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


def end_trees(pids: list, grace: float = 10):
    """SIGTERM every process of these trees, SIGKILL what is left after grace seconds."""
    import signal
    procs = {p for pid in pids for p in descendants(pid)}
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for p in procs:
            try:
                os.kill(p, sig)
            except ProcessLookupError:
                pass
        t = time.time() + grace
        while sig == signal.SIGTERM and time.time() < t and any(running(p) for p in procs):
            time.sleep(0.2)


def scope_cgroup(unit: str) -> Path:
    """The cgroup directory of a systemd user scope."""
    out = subprocess.run(["systemctl", "--user", "show", "-P", "ControlGroup", unit], capture_output=True,
                         text=True).stdout.strip()
    return Path("/sys/fs/cgroup" + out) if out else None


def scope_memory(d: Path):
    """(held, peak, limit) of a systemd scope in MB: as step_memory, the scope's own cgroup."""
    stat = dict(line.split() for line in (d / "memory.stat").read_text().splitlines())
    current = int((d / "memory.current").read_text())
    peak = int((d / "memory.peak").read_text()) if (d / "memory.peak").exists() else current
    limit = (d / "memory.max").read_text().strip()
    return ((current - int(stat["file"]) + int(stat["shmem"])) / 2 ** 20, peak / 2 ** 20,
            None if limit == "max" else int(limit) / 2 ** 20)


def build_sdps_local(site: Site, node: Node, pmps: dict, sdp_root: Path, precision: int, log_dir: Path,
                     instances: int, procs: int):
    """build_sdps on a machine without Slurm: the instances of a wave on disjoint physical cores; the
    watchdog reads each instance's process tree (tree_memory_mb) or, with memory_guard = systemd, its scope's
    cgroup; over MEMORY_WATCH of its share the wave's trees are ended and the step fails. The waves as
    build_sdps's, with "measure" (pss, rss_anon, cgroup) and "watchdog" true."""
    mem = (node.mem_mb - site.driver_reserve_mb) // instances
    scoped = site.memory_guard == "systemd"
    names = list(pmps)
    sdps, failed, waves = {}, [], []
    for i in range(0, len(names), instances):
        batch = names[i:i + instances]
        base = node_used_mb()
        running, units = {}, {}
        for j, name in enumerate(batch):
            sdps[name] = sdp_root / name
            argv = [site.pmp2sdp, "--precision", precision, "--input", pmps[name], "--output", sdps[name]]
            wrap = None
            if scoped:
                units[name] = f"bootlab-pmp2sdp-{os.getpid()}-{i + j}.scope"
                wrap = ["systemd-run", "--user", "--scope", "--quiet", "--unit", units[name],
                        "-p", f"MemoryMax={mem}M", "-p", f"MemoryHigh={mem}M"]
            running[name] = launch(site, node, argv, procs, 1, mem, True, log_dir / f"pmp2sdp-{name}",
                                   cores=node.cpus[j * procs:(j + 1) * procs], wrap=wrap)
        wave = {"names": batch, "held_mb": 0.0, "cgroup_peak_mb": 0.0, "node_mb": 0.0, "watchdog": True,
                "measure": "cgroup" if scoped else "pss"}
        over = None
        while over is None and any(p.poll() is None for p in running.values()):
            for name, p in running.items():
                if p.poll() is not None:
                    continue
                if scoped:
                    d = scope_cgroup(units[name])
                    try:
                        held, peak, _ = scope_memory(d)
                    except (OSError, KeyError, ValueError, TypeError):
                        continue
                else:
                    held, measure = tree_memory_mb(p.pid)
                    peak = held
                    if measure != "pss":
                        wave["measure"] = measure
                wave["held_mb"] = max(wave["held_mb"], held)
                wave["cgroup_peak_mb"] = max(wave["cgroup_peak_mb"], peak)
                if held > MEMORY_WATCH * mem:
                    over = f"{name} held {held:.0f} MB of its {mem} MB ({wave['measure']})"
            wave["node_mb"] = max(wave["node_mb"], node_used_mb() - base)
            if over is None:
                time.sleep(SAMPLE_SECONDS)
        if over:
            end_trees([p.pid for p in running.values()])
            wait_all(running)
            raise StepFailed(f"pmp2sdp {', '.join(batch)}: {over}; the wave was ended (an instance needs "
                             f"more memory than its share)")
        failed += [f"{n} exit {c}" for n, c in wait_all(running).items() if c != 0]
        waves.append({**wave, **{k: round(wave[k]) for k in ("held_mb", "cgroup_peak_mb", "node_mb")}})
    if failed:
        raise StepFailed("pmp2sdp " + ", ".join(failed))
    return sdps, waves


LOCK = None   # the local driver's lock file, open while it runs


def local_lock(runs: Path, what: str):
    """One local run on this machine at a time: an exclusive flock on
    <runs>/.local.lock, held while the driver lives; its text names what holds it."""
    import fcntl
    global LOCK
    runs.mkdir(parents=True, exist_ok=True)
    f = open(runs / ".local.lock", "a+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.seek(0)
        held = f.read().strip()
        f.close()
        raise SystemExit(f"another local run holds {runs / '.local.lock'}: {held}")
    f.seek(0)
    f.truncate()
    f.write(f"{what} pid {os.getpid()} since {now()}\n")
    f.flush()
    LOCK = f


def write_driver_pid(run_dir: Path):
    """<run dir>/driver.pid: the driver's process group (bootlab submit starts it with setsid), for bootlab stop
    and for a resume to see that no driver of the run still lives."""
    (run_dir / "driver.pid").write_text(json.dumps({"pid": os.getpid(), "pgid": os.getpgid(0),
                                                    "host": socket.gethostname().split(".")[0], "start": now()}) + "\n")
    if LOCK:
        LOCK.write(f"run directory {run_dir}\n")
        LOCK.flush()


def lock_holder(runs: Path):
    """What holds <runs>/.local.lock now, or None (bootlab submit looks before it starts a driver)."""
    import fcntl
    f = runs / ".local.lock"
    if not f.exists():
        return None
    with open(f, "a+") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fh.seek(0)
            return fh.read().strip() or "a local run"
        fcntl.flock(fh, fcntl.LOCK_UN)
    return None


def solver_ranks(pid: int, solver: str, proc: Path = Path("/proc")) -> list:
    """The processes below pid that run the solver itself (argv[0] its name; not mpirun, not apptainer)."""
    out = []
    for p in descendants(pid, proc):
        try:
            argv = (proc / str(p) / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if argv and os.path.basename(argv[0].decode(errors="replace")) == solver:
            out.append(p)
    return out


def group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def sdp_size(sdp: Path):
    """(N, blocks) of an SDP directory written by pmp2sdp 3.0: N from objectives.json's b, blocks from
    block_info_*.json (checked on a Lambda 11 SDP: b has 608 entries, its N)."""
    import json
    blocks = len(list(sdp.glob("block_info_*.json")))
    obj = json.loads((sdp / "objectives.json").read_text())
    return len(obj["b"]), blocks


# ---------------------------------------------------------------- the job's other nodes (plan 3, 4.3)
#
# The driver runs on node 0. Every node keeps its files under the same paths (the run's scratch directory).
# Work on another node goes through the site's multinode.transfer: "srun" (the default: a light step inside the
# job, --overlap so it never waits for the solver's cores; nothing outside Slurm needed) or "ssh" (to the node's
# name, the bulk copies over IPoIB where ssh into the job's nodes works). Bulk copies run the image's tar on both sides (some nodes have
# no tar). With a shared scratch (Node.shared) nothing goes to another node: every node sees node 0's
# files. Nothing is retried: a failure ends the step (StepFailed).

def ssh_argv(site: Site, host: str, address: str = None) -> list:
    home = os.path.expanduser("~")
    return [a.replace("~/", home + "/") for a in site.multinode["ssh"]] + ["-o", f"HostKeyAlias={host}",
                                                                           address or host]


def srun_light(hosts: list) -> list:
    """A light step on these nodes, one task each, beside whatever runs (--overlap); stdin to every task."""
    return ["srun", "--overlap", f"--nodes={len(hosts)}", f"--ntasks={len(hosts)}", "--ntasks-per-node=1",
            "-w", ",".join(hosts), "--cpu-bind=none", "--input=all"]


def remote_argv(site: Site, node: Node, host: str, command: str, bulk: bool = False) -> list:
    """argv that runs a shell command on another node of the job; bulk (a copy's data): over IPoIB with ssh."""
    if site.transfer == "ssh":
        return ssh_argv(site, host, ib_address(site, node, host) if bulk else None) + [command]
    return srun_light([host]) + ["bash", "-c", command]


def on_host(site: Site, node: Node, host: str, script: str, stdin: bytes = None) -> str:
    """Run a shell script on one of the job's nodes (node 0 locally); its stdout."""
    argv = ["bash", "-c", script] if host == node.host else remote_argv(site, node, host, script)
    r = subprocess.run(argv, input=stdin, capture_output=True)
    if r.returncode != 0:
        raise StepFailed(f"{host}: {script[:200]!r} exit {r.returncode}: {r.stderr.decode(errors='replace')[-800:]}")
    return r.stdout.decode()


def on_hosts(site: Site, node: Node, script: str, hosts: list = None, stdin: bytes = None) -> dict:
    """The script on every node of the job (or on hosts) at once; host -> stdout. With transfer srun the other
    nodes get it in one step (a step costs about a second), each line of its output tagged
    with the node."""
    hosts = node.all_hosts if hosts is None else hosts
    remote = [h for h in hosts if h != node.host]
    out = {}
    if remote and site.transfer == "srun":
        tagged = ('o=$(bash -c "$BOOTLAB_SCRIPT" 2>&1); r=$?; h=$(hostname -s); '
                  'printf "%s\\n" "$o" | sed "s|^|$h\\t|"; exit $r')
        r = subprocess.run(srun_light(remote) + ["bash", "-c", tagged], input=stdin, capture_output=True,
                           env={**os.environ, "BOOTLAB_SCRIPT": script})
        text = r.stdout.decode(errors="replace")
        if r.returncode != 0:
            raise StepFailed(f"{remote}: {script[:200]!r} exit {r.returncode}: {text[-800:]}"
                             f"{r.stderr.decode(errors='replace')[-400:]}")
        for h in remote:
            out[h] = "".join(line.split("\t", 1)[1] + "\n" for line in text.splitlines()
                             if line.startswith(h + "\t") and line.split("\t", 1)[1])
    elif remote:
        with ThreadPoolExecutor(len(remote)) as ex:
            futures = {h: ex.submit(on_host, site, node, h, script, stdin) for h in remote}
        out.update({h: f.result() for h, f in futures.items()})
    if node.host in hosts:
        out[node.host] = on_host(site, node, node.host, script, stdin)
    return out


def clear(site: Site, node: Node, d: Path):
    """d removed on every node of the job: node 0's from Python, the others' through the transfer (none with
    a shared scratch)."""
    on_hosts(site, node, f"rm -rf {d}", node.others)
    if Path(d).is_dir():
        shutil.rmtree(d)


def small_tar(files: list) -> bytes:
    """A tar of small files of node 0 under their paths from / (for tar -C / -xf - on another node)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as t:
        for f in files:
            t.add(str(f), arcname=str(f).lstrip("/"))
    return buf.getvalue()


_ADDRESSES = {}


def ib_address(site: Site, node: Node, host: str) -> str:
    """The host's IPoIB address (site multinode.ipoib_prefix), read once."""
    if host not in _ADDRESSES:
        prefix = site.multinode["ipoib_prefix"]
        out = on_host(site, node, host, "ip -4 -o addr")
        found = [w.split("/")[0] for line in out.splitlines() for w in line.split() if w.startswith(prefix)]
        if not found:
            raise StepFailed(f"{host}: no IPv4 address in {prefix}")
        _ADDRESSES[host] = found[0]
    return _ADDRESSES[host]


def tar_argv(site: Site, *args) -> list:
    """tar in the image, with the site's binds."""
    binds = [x for b in site.binds for x in ("--bind", b)]
    return ["apptainer", "exec", *binds, site.sif, "tar", *map(str, args)]


def remote_tar(site: Site, *args, before: str = "") -> str:
    """The same as a shell command on another node; before: shell text in front of tar in the same pipeline (a
    file list into tar -T -). Over ssh apptainer comes from multinode.remote_setup; an srun step has the
    driver's environment."""
    tar = " ".join(shlex.quote(a) for a in tar_argv(site, *args))
    setup = f"{site.multinode['remote_setup']} > /dev/null 2>&1; " if site.transfer == "ssh" else ""
    return f"{setup}{before}{tar}"


def stream(send_argv: list, recv_argv: list, what: str, stdin: bytes = None):
    """send's stdout into recv's stdin (a tar pipe between two nodes); StepFailed when either fails."""
    send = subprocess.Popen(send_argv, stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    recv = subprocess.Popen(recv_argv, stdin=send.stdout, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    send.stdout.close()
    if stdin is not None:
        send.stdin.write(stdin)
        send.stdin.close()
    return send, recv, what


# Copies with transfer srun: a light srun step on the target runs receive.py (the site's python), node 0 sends
# its tar over a direct TCP connection. Through Slurm's own stdin forwarding a copy ran at 39 MB/s, and sbcast
# crashed with a bus error twice (2.6 GB: 70 and 68 s; sbcast core dumped); a direct
# connection runs at the network's speed. receive.py says how the receiver keeps other users out.
RECEIVE = Path(__file__).resolve().parent / "receive.py"


def copy_address(site: Site, node: Node, host: str) -> str:
    """A node's address for copies: on multinode.copy_network (an IPv4 prefix, a fast network), or the
    address its name resolves to."""
    prefix = (site.multinode or {}).get("copy_network")
    if not prefix:
        return socket.gethostbyname(host)
    key = (host, prefix)
    if key not in _ADDRESSES:
        out = on_host(site, node, host, "ip -4 -o addr")
        found = [w.split("/")[0] for line in out.splitlines() for w in line.split() if w.startswith(prefix)]
        if not found:
            raise StepFailed(f"{host}: no IPv4 address in {prefix}")
        _ADDRESSES[key] = found[0]
    return _ADDRESSES[key]


def tcp_copy(site: Site, node: Node, host: str, send_argv: list, dest: Path) -> str:
    """send's output (a tar) unpacked into dest on host by receive.py, which a light srun step starts there:
    it listens on host's copy address only, takes node 0's address only and then this copy's one-time token
    (passed in the step's environment). Node 0 connects from its own copy address. The failure as text, or
    None."""
    token = secrets.token_hex(32)
    bind, peer = copy_address(site, node, host), copy_address(site, node, node.host)
    env = {**os.environ, "BOOTLAB_COPY_TOKEN": token, "BOOTLAB_COPY_BIND": bind, "BOOTLAB_COPY_PEER": peer}
    recv = subprocess.Popen(srun_light([host]) + [site.python, str(RECEIVE), str(dest)], env=env,
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    line = recv.stdout.readline().strip()
    if not line.isdigit():
        recv.kill()
        return f"{host}: no receiver ({line!r} {recv.stderr.read().decode(errors='replace')[-300:]})"
    send = subprocess.Popen(send_argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as c:
            c.bind((peer, 0))
            c.settimeout(60)
            c.connect((bind, int(line)))
            c.settimeout(None)
            c.sendall(token.encode())
            f = c.makefile("wb")
            shutil.copyfileobj(send.stdout, f, 1 << 20)
            f.close()                        # the buffer flushed, then end of stream for the receiver
            c.shutdown(socket.SHUT_WR)
    except OSError as e:
        send.kill()
        recv.kill()
        return f"{host}: {e}"
    if send.wait() != 0 or recv.wait() != 0:
        return (f"{host}: tar exit {send.returncode}/{recv.returncode} "
                f"{(send.stderr.read() + recv.stderr.read()).decode(errors='replace')[-300:]}")
    return None


def finish_streams(pipes: list) -> list:
    """Wait for stream()s; their failures as text."""
    bad = []
    for send, recv, what in pipes:
        err = recv.communicate()[1]
        if send.wait() != 0 or recv.returncode != 0:
            bad.append(f"{what}: tar exit {send.returncode}/{recv.returncode} "
                       f"{(send.stderr.read() + err).decode(errors='replace')[-300:]}")
    return bad


def tree_size(site: Site, node: Node, host: str, d: Path) -> tuple:
    """(files, bytes) under d on host: file sizes only (a directory's own size differs between copies)."""
    out = on_host(site, node, host, f"find {d} -type f -printf '%s\\n' | awk '{{s += $1}} END {{print NR, s + 0}}'")
    files, size = out.split()
    return int(files), int(size)


def distribute(site: Site, node: Node, src: Path) -> dict:
    """The directory src of node 0 to the same path on every other node of the job: the directory tree first
    in one remote call (tar streams making the same parent directories at once lost a file), then
    copy_streams tar streams to each node (tar | ssh over IPoIB, or tar | an srun step's stdin), all nodes at
    once, the files dealt to the streams largest first. Checked by the number of files and their bytes on each
    node. Returns {files, bytes, seconds}."""
    t = time.time()
    others = node.others
    files = sorted(((p.stat().st_size, str(p.relative_to(src))) for p in src.rglob("*") if p.is_file()), reverse=True)
    dirs = "\n".join(str(p.relative_to(src)) for p in src.rglob("*") if p.is_dir())
    want = (len(files), sum(s for s, _ in files))
    k = site.multinode.get("copy_streams", 1)
    lists = []
    for i in range(k):
        names = [n for _, n in files[i::k]]
        if names:
            lst = src.parent / f".{src.name}.copy{i}"
            lst.write_text("\n".join(names) + "\n")
            lists.append(lst)
    on_hosts(site, node, f"mkdir -p {src} && cd {src} && xargs -r -d '\\n' mkdir -p", others, dirs.encode())
    if not lists:
        bad = []
    elif site.transfer == "srun":
        with ThreadPoolExecutor(len(others) * len(lists)) as ex:
            bad = [b for b in ex.map(lambda hl: tcp_copy(site, node, hl[0], tar_argv(site, "-C", src, "-cf", "-", "-T", hl[1]), src),
                                     [(h, lst) for h in others for lst in lists]) if b]
    else:
        pipes = [stream(tar_argv(site, "-C", src, "-cf", "-", "-T", lst),
                        remote_argv(site, node, h, remote_tar(site, "-C", str(src), "-xf", "-"), bulk=True), h)
                 for h in others for lst in lists]
        bad = finish_streams(pipes)
    for lst in lists:
        lst.unlink()
    if bad:
        raise StepFailed(f"copy of {src}: " + "; ".join(bad))
    for h in others:
        have = tree_size(site, node, h, src)
        if have != want:
            raise StepFailed(f"copy of {src} to {h}: {have} (files, bytes), want {want}")
    done = {"files": want[0], "bytes": want[1], "seconds": round(time.time() - t, 1)}
    if os.environ.get("BOOTLAB_COPY_SHA256"):   # the acceptance's check, file by file (plan 4.3, item 1)
        sums = on_hosts(site, node, f"cd {src} && find . -type f -print0 | sort -z | xargs -0 sha256sum",
                        [node.host] + others)
        differ = [h for h in others if sums[h] != sums[node.host]]
        if differ:
            raise StepFailed(f"copy of {src}: sha256 differs on {differ}")
        done["sha256_same"] = len(sums[node.host].splitlines())
    return done


def gather(site: Site, node: Node, d: Path):
    """The files the other nodes wrote under d (the solution's x_<block>.txt of the blocks their ranks hold)
    into d on node 0, keeping node 0's own (nothing to do with a shared scratch)."""
    for h in node.others:
        bad = finish_streams([stream(remote_argv(site, node, h, remote_tar(site, "-C", str(d), "-cf", "-", "."), bulk=True),
                                     tar_argv(site, "-C", d, "-xf", "-", "--skip-old-files"), f"gather {d} from {h}")])
        if bad:
            raise StepFailed("; ".join(bad))


# ---------------------------------------------------------------- the solver's checkpoint (docs/en/interfaces.md 11)
#
# dynamical_sdp's checkpoint directory is the run's checkpoint/ on the runs file system, one for all nodes.
# A save of generation g + 1 (save_checkpoint.cxx, cb3fa24): each rank removes its file of the backup
# generation g - 1 and writes checkpoint_<g+1>_<rank>; rank 0 writes checkpoint_new.json; after a barrier
# rank 0 renames it to checkpoint.json. So checkpoint.json always names a generation whose files are all
# there. It saves every checkpointInterval seconds, on SIGTERM and at the end of a call, and keeps two
# generations, so while a call runs the generation of the last recorded call can go. kept/<k> keeps it for
# record k of solves.jsonl: hard links to its rank files (no file of a generation is written twice; a save
# removes the old generation's names, the links keep the data), copies of checkpoint.json and block_timings,
# and layout.json. A resume sets checkpoint/ back to the newest kept/<k> with k <= the records.

RANK_FILE = re.compile(r"checkpoint_\d+_\d+")


def generation(ck: Path) -> int:
    return json.loads((ck / "checkpoint.json").read_text())["current"]


def check_hard_links(runs: Path):
    """At the start: the runs file system makes hard links (kept/ is made of them, SPEC 7.3). Not: stop,
    saying why (no copy instead: each kept generation would then cost a whole copy)."""
    probe = runs / f".bootlab-link-probe-{os.getpid()}"
    try:
        probe.write_text("probe\n")
        os.link(probe, probe.with_name(probe.name + ".link"))
        ok = probe.stat().st_nlink == 2
    except OSError as e:
        ok, why = False, str(e)
    else:
        why = f"st_nlink {probe.stat().st_nlink}, want 2"
    finally:
        for p in (probe, probe.with_name(probe.name + ".link")):
            p.unlink(missing_ok=True)
    if not ok:
        raise SystemExit(f"the runs directory {runs} cannot make hard links ({why}): bootlab keeps the solver's "
                         f"checkpoint generations by hard links there (docs/en/interfaces.md section 11); put runs on a file "
                         f"system that does (sites/_template/site.toml)")


def wait_present(d: Path, names: list, what: str):
    """Files other nodes wrote into d on NFS can take a while to show here: look until all are there,
    up to a minute."""
    for _ in range(30):
        missing = [n for n in names if not (d / n).exists()]
        if not missing:
            return
        time.sleep(2)
    raise StepFailed(f"{what}: {len(missing)} of {len(names)} files not in {d} (e.g. {missing[0]})")


def rank_files(lay: dict, g: int) -> list:
    return [f"checkpoint_{g}_{r}" for r in range(len(lay["nodes"]) * lay["ppn"])]


def keep_generation(ck: Path, k: int, lay: dict):
    """After a solver call, before its record k: kept/<k> for the generation checkpoint.json names, made in
    kept/<k>.new and renamed; lay is the job's {nodes, node_type, ppn}, written with the generation as
    layout.json. Returns {generation, files, seconds}, or None when the solver has written no checkpoint."""
    if not (ck / "checkpoint.json").exists():
        return None
    t = time.time()
    g = generation(ck)
    names = rank_files(lay, g)
    wait_present(ck, names, "checkpoint")
    kept = ck / "kept"
    new = kept / f"{k}.new"
    shutil.rmtree(new, ignore_errors=True)
    new.mkdir(parents=True)
    for n in names:
        os.link(ck / n, new / n)
    shutil.copy(ck / "checkpoint.json", new / "checkpoint.json")
    shutil.copy(ck / "block_timings", new / "block_timings")
    (new / "layout.json").write_text(json.dumps({**lay, "generation": g}) + "\n")
    shutil.rmtree(kept / str(k), ignore_errors=True)
    new.rename(kept / str(k))
    return {"generation": g, "files": len(names) + 3, "seconds": round(time.time() - t, 1)}


def kept_for(ck: Path, records: int):
    """The newest kept/<k> with k <= records (a record without one, a failed step's, keeps the one before);
    None if there is none."""
    d = ck / "kept"
    ks = [int(p.name) for p in d.iterdir() if p.name.isdigit() and int(p.name) <= records] if d.exists() else []
    return d / str(max(ks)) if ks else None


def prune_kept(ck: Path, records: int):
    """After record `records` is written: the kept/<k> older than kept_for's removed, so one is left (each
    holds a whole generation once the solver has saved over it). Not when kept/<k> is made: a job killed
    between that and record k resumes with k - 1 records, and needs kept/<k - 1>; so two for that while."""
    last = kept_for(ck, records)
    if last:
        for p in (ck / "kept").iterdir():
            if p.name.isdigit() and int(p.name) < int(last.name):
                shutil.rmtree(p)


def restore_generation(ck: Path, kept: Path) -> int:
    """checkpoint/ set back to kept's generation before a resume: newer kept/<k> and unfinished kept/<k>.new
    removed; the rank files checkpoint_<g>_<rank> of every other generation removed (one written when the job
    was killed, newer or older ones), nothing else; its rank files linked back; its checkpoint.json and
    block_timings copied under a temporary name and renamed over. Returns the generation."""
    lay = json.loads((kept / "layout.json").read_text())
    g = lay["generation"]
    for p in (ck / "kept").iterdir():
        if p.name.endswith(".new") or (p.name.isdigit() and int(p.name) > int(kept.name)):
            shutil.rmtree(p)
    mine = set(rank_files(lay, g))
    for f in ck.iterdir():
        if RANK_FILE.fullmatch(f.name) and f.name not in mine:
            f.unlink()
    for n in mine:
        (ck / n).unlink(missing_ok=True)
        os.link(kept / n, ck / n)
    for n in ("block_timings", "checkpoint.json"):
        tmp = ck / f".{n}.restore"
        shutil.copy(kept / n, tmp)
        os.replace(tmp, ck / n)
    return g
