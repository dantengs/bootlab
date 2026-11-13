"""Cluster probe: the facts a site needs, measured on the
cluster instead of written by hand.

    bootlab probe plan --sif <image> --shared <dir> --out <dir> [--what node|pair] [--types a,b] [--submit]
                                                                               on the login node
    bootlab probe node --sif <image> --shared <dir> --out <dir> [--py-test <dir>]   inside a job on one node
    bootlab probe pair --sif <image> --shared <dir> --out <dir> --sdp <dir>    inside a job on two nodes of a type
    bootlab probe merge <out dir>                                              -> <out dir>/probe.json

plan reads sinfo, groups the nodes into types (one solve uses one type), picks one idle node of each type and
prints (or submits) one small job per type: never one srun across types (one that did hung). node
writes node-<host>-<job>.json: CPU and memory, IB ports, network interfaces, the local scratch candidates and the
shared directory (mount, the network it goes over, hard links, throughput in bootlab's file pattern), apptainer,
the image's UCX, cgroup v2, Python, the time zone. pair (two whole nodes of one type,
only when both are idle) writes pair-<host>-<host>-<job>.json: a light srun across them, ucx_perftest over each
network they share, bootlab's own copy (execute.distribute) over each of them and over ssh where ssh into the
job's nodes works, both nodes writing the shared directory at once and a cold read of the other's files, and
the full-scale MPI start-up (every physical core of both nodes, sdpb on a small SDP, 2 iterations) three times
over each transport, the connection errors counted. It writes only its own directories
(<candidate>/<user>-bootlab-probe-<job>) and removes what it wrote. Every time is ISO 8601 with its UTC offset.
Every step has --kill-on-bad-exit and a time limit that ends its process group.
"""
import argparse
import datetime
import getpass
import hashlib
import json
import multiprocessing
import os
import random
import re
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

FORMAT = "bootlab-probe/1"
# bootlab writes files of a few MB (blocks about 3.4 MB, SDP 5.8 MB, PMP parts 6 MB, checkpoint 5 MB
# a rank), several at once (pmp2sdp): its pattern is 512 files of 4 MiB from 8 writers. Large files are measured
# beside it for reference only (an NFS that dd finds faster than a local NVMe can write bootlab's files 7-10
# times slower).
PATTERNS = {"bootlab": (512, 4 << 20), "large": (8, 256 << 20)}
MAD_SCALE = 1.4826        # MAD x this estimates a normal spread's sigma
SPIKE = 3                 # a repeat over this times its median is a spike (seen: 8.16 s against 1.06)
WRITERS = 8
REPEATS = 3
MIN_FREE_GB = 20          # a local candidate with less is recorded, not measured
# the usual names; a site's own (a directory on a local disk that is not a mount point) by --local
LOCAL_NAMES = ["/scratch", "/local", "/localscratch", "/tmp", "/var/tmp"]
NETWORK_FS = {"nfs", "nfs4", "cifs", "smb3", "lustre", "gpfs", "beegfs", "ceph", "fuse.sshfs", "panfs", "wekafs"}
MEMORY_FS = {"tmpfs", "ramfs", "devtmpfs", "overlay", "squashfs", "proc", "sysfs"}

# pair: what every bootlab MPI step's environment has (sites/_template [env], each rank in its own apptainer
# instance), the start-up test's repeats and limits. Measured: a 2 x 256 start-up that fails ends after about
# 111 s, a good 2 x 128 run of the same small SDP took 51 s; STARTUP_LIMIT bounds one run, and a run starts only
# when the job has that much time left (nothing is cut short). PAIR_TIME: the other items take a few minutes
# (copies of 2 GiB three times over each network, ucx_perftest), plus 3 transports x 3 runs x STARTUP_LIMIT.
MPI_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "OMPI_MCA_pml": "ucx", "OMPI_MCA_btl": "^openib",
           "UCX_POSIX_USE_PROC_LINK": "n", "PMIX_MCA_gds": "hash", "UCX_LOG_FILE": "stderr",
           "OMPI_MCA_pml_ucx_num_disconnect": "1000000"}
MPI_UNSET = ["UCX_NET_DEVICES", "UCX_TLS", "OMPI_MCA_plm"]
STARTUP_REPEATS = 3
STARTUP_LIMIT = 300
PAIR_TIME = "01:00:00"
ERRORS = {"refused": "failed: Connection refused", "cannot_assign": "Cannot assign requested address",
          "unreachable": "Destination is unreachable"}


def now() -> str:
    """The time with this machine's UTC offset."""
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def run(cmd, timeout=60, shell=False) -> dict:
    """A command's exit, stdout and stderr; never raises (a probe item records its failure and goes on)."""
    try:
        p = subprocess.run(cmd, shell=shell, capture_output=True, text=True, timeout=timeout)
        return {"exit": p.returncode, "out": p.stdout, "err": p.stderr[-2000:]}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"exit": None, "out": "", "err": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------------------------- plan

SINFO_FORMAT = "%N|%P|%T|%c|%z|%m|%f"


def parse_sinfo(text: str) -> list:
    """sinfo -N -h -o SINFO_FORMAT -> one dict a node and partition."""
    out = []
    for line in text.splitlines():
        if not line.strip():
            continue
        host, part, state, cpus, szt, mem, feat = line.split("|")
        s, c, t = (int(x) for x in szt.split(":"))
        out.append({"host": host, "partition": part.rstrip("*"), "default_partition": part.endswith("*"),
                    "state": state, "cpus": int(cpus), "sockets": s, "cores": c, "threads": t,
                    "physical_cores": s * c, "mem_mb": int(mem),
                    "features": [] if feat in ("(null)", "") else feat.split(",")})
    return out


def node_types(nodes: list) -> dict:
    """Group the nodes by (features, physical cores, threads); a group's name is its first feature that no other
    group has, else the host name less its last -<number> part. Slurm's RealMemory differs a little between nodes
    of one type (seen: three nodes of a type at 253089-253104 MB, two of another at 500000 and 515111): the type's
    mem_mb is the least, what a job can have on any of them; each host's is kept. name -> {hosts, ...}."""
    groups = {}
    for n in nodes:
        key = (tuple(n["features"]), n["physical_cores"], n["threads"])
        g = groups.setdefault(key, {"hosts": [], "physical_cores": n["physical_cores"], "threads": n["threads"],
                                    "mem_mb": n["mem_mb"], "host_mem_mb": {}, "features": n["features"],
                                    "partitions": []})
        g["mem_mb"] = min(g["mem_mb"], n["mem_mb"])
        g["host_mem_mb"][n["host"]] = n["mem_mb"]
        if n["host"] not in g["hosts"]:
            g["hosts"].append(n["host"])
        if n["partition"] not in g["partitions"]:
            g["partitions"].append(n["partition"])
    out = {}
    for key, g in groups.items():
        others = {f for k in groups if k != key for f in k[0]}
        unique = [f for f in g["features"] if f not in others]
        name = unique[0] if unique else re.sub(r"-\d+$", "", sorted(g["hosts"])[0])
        while name in out:
            name += "x"
        g["hosts"].sort()
        out[name] = g
    return dict(sorted(out.items()))


def idle_hosts(nodes: list, hosts: list) -> list:
    """The hosts of a type whose every partition line says idle."""
    state = {}
    for n in nodes:
        if n["host"] in hosts:
            state.setdefault(n["host"], set()).add(n["state"])
    return [h for h in hosts if state.get(h) == {"idle"}]


def pick_idle(nodes: list, hosts: list):
    """The first idle host of a type, or None."""
    return next(iter(idle_hosts(nodes, hosts)), None)


def pair_job(hosts: list, cores: int, args, python: str) -> list:
    """The sbatch command of one pair probe: the two nodes whole (exclusive, every physical core a task for the
    start-up test), singleton with the node probes, PAIR_TIME."""
    src = Path(__file__).resolve().parents[1]
    cmd = (f"export PYTHONPATH={shlex.quote(str(src))}; exec {shlex.quote(python)} -m bootlab.probe pair "
           f"--sif {shlex.quote(args.sif)} --shared {shlex.quote(args.shared)} --out {shlex.quote(args.out)} "
           f"--sdp {shlex.quote(args.sdp)} --sdp-precision {args.sdp_precision} --setup {shlex.quote(args.setup)}")
    cmd += "".join(f" --local {shlex.quote(d)}" for d in args.local)
    return ["sbatch", "--job-name=bootlab-probe", "--dependency=singleton", "-w", ",".join(hosts), "--nodes=2",
            f"--ntasks-per-node={cores}", "--exclusive", "--mem=0", "--hint=nomultithread", f"--time={PAIR_TIME}",
            f"--output={args.out}/slurm-probe-pair-%j.out", "--wrap", cmd]


def node_job(host: str, args, python: str) -> list:
    """The sbatch command of one node probe: a few cores, a short limit, physical cores (--hint=nomultithread)."""
    src = Path(__file__).resolve().parents[1]
    cmd = (f"export PYTHONPATH={shlex.quote(str(src))}; exec {shlex.quote(python)} -m bootlab.probe node "
           f"--sif {shlex.quote(args.sif)} --shared {shlex.quote(args.shared)} --out {shlex.quote(args.out)} "
           f"--setup {shlex.quote(args.setup)}")
    if args.py_test:
        cmd += f" --py-test {shlex.quote(args.py_test)}"
    cmd += "".join(f" --local {shlex.quote(d)}" for d in args.local)
    # one node probe at a time (singleton: same name and user): probes that wrote to the shared directory at
    # the same moment and measured each other. 16G: the bootlab pattern's 2 GiB stays below the job's dirty limit
    # (a run has the whole node's memory, and its page cache takes the files).
    return ["sbatch", "--job-name=bootlab-probe", "--dependency=singleton", "-w", host, "--nodes=1", "--ntasks=1",
            f"--cpus-per-task={WRITERS}", "--mem=16G", "--hint=nomultithread", "--time=00:15:00",
            f"--output={args.out}/slurm-probe-node-%j.out", "--wrap", cmd]


def cmd_plan(args):
    from . import execute
    args.shared = args.shared or execute.install_runs()
    Path(args.shared).mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    args.setup = apptainer_setup()
    # --partition: that partition's nodes only, the jobs go to it; --node: that node, busy or not (the job waits)
    s = run(["sinfo", "-N", "-h", "-o", SINFO_FORMAT] + (["-p", args.partition] if args.partition else []))
    if s["exit"] != 0:
        raise SystemExit(f"sinfo failed: {s['err']}")
    nodes = parse_sinfo(s["out"])
    types = node_types(nodes)
    if args.node and args.what != "node":
        raise SystemExit("--node is for node probes")
    if args.node and not any(args.node in t["hosts"] for t in types.values()):
        raise SystemExit(f"--node {args.node} is not among the nodes sinfo lists"
                         + (f" in partition {args.partition}" if args.partition else ""))
    cfg = run(["scontrol", "show", "config"])
    conf = dict(re.findall(r"^(\w+)\s+=\s+(.*?)\s*$", cfg["out"], flags=re.M))
    mpi = run(["srun", "--mpi=list"])
    h = hashlib.sha256()
    with open(args.sif, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    plan = {"format": FORMAT, "what": "plan", "time": now(), "host": socket.gethostname().split(".")[0],
            "user": getpass.getuser(), "sif": args.sif, "sif_sha256": h.hexdigest(), "shared": args.shared,
            "shared_where": where(args.shared, parse_mountinfo(read("/proc/self/mountinfo", ""))),
            "apptainer_setup": args.setup,
            "slurm": {k: conf.get(k) for k in ("SLURM_VERSION", "MpiDefault", "SelectType", "SelectTypeParameters",
                                               "ProctrackType", "TaskPlugin", "JobAcctGatherType")},
            "srun_mpi": mpi["out"] + mpi["err"], "nodes": nodes, "types": {}}
    if args.partition:
        plan["partition"] = args.partition
    python = args.python or sys.executable
    plan["what"] = args.what
    if args.what == "pair" and not args.sdp:
        raise SystemExit("--what pair needs --sdp (a small SDP for the start-up test)")
    wanted = set(args.types.split(",")) if args.types else set(types)
    if args.node:
        wanted = {name for name, t in types.items() if args.node in t["hosts"]}
    for name, t in types.items():
        if name not in wanted:
            continue
        idle = [args.node] if args.node else idle_hosts(nodes, t["hosts"])
        if args.what == "node":
            hosts = idle[:1]
            job = node_job(hosts[0], args, python) if hosts else None
        else:
            hosts = idle[:2] if len(idle) >= 2 else []
            job = pair_job(hosts, t["physical_cores"], args, python) if hosts else None
        if job and args.partition:
            job = job[:1] + [f"--partition={args.partition}"] + job[1:]
        plan["types"][name] = {**t, "probe_hosts": hosts, "sbatch": job}
        print(f"{name}: {len(t['hosts'])} nodes, {t['physical_cores']} physical cores, {t['mem_mb']} MB; "
              + (f"{args.what} probe on {','.join(hosts)}" if job else f"not enough idle nodes for a {args.what} probe"))
        if job and args.submit:
            r = run(job)
            plan["types"][name]["submitted"] = (r["out"] + r["err"]).strip()
            print("  " + plan["types"][name]["submitted"])
        elif job:
            print("  " + shlex.join(job))
    (out / ("plan.json" if args.what == "node" else f"plan-pair-{time.strftime('%Y%m%d-%H%M%S')}.json")).write_text(
        json.dumps(plan, indent=1) + "\n")


# ---------------------------------------------------------------------------------------------- node

def read(path, default=None):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return default


def cpu() -> dict:
    """The whole node's topology (not this job's CPUs): physical cores are distinct (package, core) pairs."""
    cores, threads = set(), 0
    for d in Path("/sys/devices/system/cpu").glob("cpu[0-9]*"):
        t = d / "topology"
        if (d / "online").exists() and read(d / "online") == "0":
            continue
        if t.exists():
            cores.add((read(t / "physical_package_id"), read(t / "core_id")))
            threads += 1
    model = next((l.split(":", 1)[1].strip() for l in read("/proc/cpuinfo", "").splitlines()
                  if l.startswith("model name")), None)
    mem = next((int(l.split()[1]) // 1024 for l in read("/proc/meminfo", "").splitlines()
                if l.startswith("MemTotal:")), None)
    return {"model": model, "physical_cores": len(cores), "logical_cpus": threads,
            "sockets": len({p for p, _ in cores}), "numa_nodes": len(list(Path("/sys/devices/system/node").glob("node[0-9]*"))),
            "mem_total_mb": mem}


def ib_ports(root="/sys/class/infiniband") -> list:
    out = []
    for dev in sorted(Path(root).glob("*")):
        nets = sorted(p.name for p in (dev / "device" / "net").glob("*")) if (dev / "device" / "net").exists() else []
        for port in sorted((dev / "ports").glob("*"), key=lambda p: int(p.name)):
            c = port / "counters"
            out.append({"port": f"{dev.name}:{port.name}", "state": read(port / "state"),
                        "phys_state": read(port / "phys_state"), "rate": read(port / "rate"),
                        "link_layer": read(port / "link_layer"), "netdevs": nets,
                        "counters": {k: int(read(c / k, "-1")) for k in ("link_downed", "link_error_recovery",
                                                                         "symbol_error")}})
    return out


def rate_gbps(rate: str) -> float:
    m = re.match(r"([\d.]+)\s*Gb/sec", rate or "")
    return float(m[1]) if m else 0.0


def choose_ib(start: list, end: list, prefer=()):
    """An InfiniBand port that is ACTIVE and LinkUp at both readings and whose link_downed
    did not rise between them; the highest rate, then one whose IPoIB interface is in prefer (the network the
    shared file system goes over: a node can have two 100 Gb/s ports, the second on another subnet), then the fewest link_downed. None when there is none."""
    before = {p["port"]: p for p in start}
    ok = [p for p in end if p["port"] in before
          and all("ACTIVE" in (q["state"] or "") and "LinkUp" in (q["phys_state"] or "")
                  and q["link_layer"] == "InfiniBand" for q in (p, before[p["port"]]))
          and p["counters"]["link_downed"] == before[p["port"]]["counters"]["link_downed"]]
    if not ok:
        return None
    return max(ok, key=lambda p: (rate_gbps(p["rate"]), bool(set(p["netdevs"]) & set(prefer)),
                                  -p["counters"]["link_downed"]))["port"]


NET_TYPES = {1: "ethernet", 32: "infiniband", 772: "loopback"}


def nics() -> list:
    addrs = {}
    for line in run(["ip", "-4", "-o", "addr"])["out"].splitlines():
        f = line.split()
        if len(f) > 3:
            addrs.setdefault(f[1], []).append(f[3])
    out = []
    for d in sorted(Path("/sys/class/net").glob("*")):
        t = int(read(d / "type", "0"))
        speed = read(d / "speed")
        out.append({"name": d.name, "type": NET_TYPES.get(t, str(t)), "operstate": read(d / "operstate"),
                    "speed_mbps": int(speed) if speed and speed.lstrip("-").isdigit() and int(speed) > 0 else None,
                    "mtu": int(read(d / "mtu", "0")), "ipoib_mode": read(d / "mode") if t == 32 else None,
                    "ipv4": addrs.get(d.name, [])})
    return out


def parse_mountinfo(text: str) -> list:
    """/proc/self/mountinfo -> [{mountpoint, fstype, source, options}] (options: the mount's and the
    superblock's, joined)."""
    out = []
    for line in text.splitlines():
        a, b = line.split(" - ", 1)
        f, g = a.split(), b.split()
        mp = f[4].replace("\\040", " ")
        out.append({"mountpoint": mp, "fstype": g[0], "source": g[1],
                    "options": ",".join([f[5]] + g[2:3]), "dev": f[2]})
    return out


def mount_of(path: str, mounts: list) -> dict:
    p = os.path.realpath(path) if os.path.exists(path) else path
    best = None
    for m in mounts:
        mp = m["mountpoint"]
        if (p == mp or p.startswith(mp.rstrip("/") + "/")) and (best is None or len(mp) >= len(best["mountpoint"])):
            best = m
    return best


def opt(options: str, key: str):
    m = re.search(rf"(?:^|,){key}=([^,]*)", options)
    return m[1] if m else None


def nfs_xprt(mountpoint: str):
    """The transport line of a mount in /proc/self/mountstats (tcp, rdma), or None."""
    for b in re.split(r"^device ", read("/proc/self/mountstats", ""), flags=re.M):
        # an automount point has an autofs entry first, without xprt
        if b and f" mounted on {mountpoint} with fstype nfs" in b.splitlines()[0]:
            m = re.search(r"^\s*xprt:\s*(\S+)", b, flags=re.M)
            return m[1] if m else None
    return None


def route_dev(addr: str):
    r = run(["ip", "route", "get", addr])
    m = re.search(r"\bdev (\S+)", r["out"])
    return m[1] if m else None


def where(path: str, mounts: list) -> dict:
    """Where a directory is (one NFS mount can have several disk arrays behind it, for example a mergerfs
    pool on the server): its path and real path, st_dev, the mount's file system and source, and the mergerfs
    branch when this host sees one (user.mergerfs.basepath: on the server, not on an NFS client)."""
    real = os.path.realpath(path)
    st = os.stat(real)
    m = mount_of(real, mounts) or {}
    try:
        branch = os.getxattr(real, "user.mergerfs.basepath").decode()
    except OSError:
        branch = None
    return {"path": path, "realpath": real, "st_dev": f"{os.major(st.st_dev)}:{os.minor(st.st_dev)}",
            "fstype": m.get("fstype"), "source": m.get("source"), "mountpoint": m.get("mountpoint"),
            "mergerfs_branch": branch}


def shared_fs(path: str, mounts: list, interfaces: list) -> dict:
    m = mount_of(path, mounts) or {}
    out = {"path": path, **m, "where": where(path, mounts)}
    if m.get("fstype", "").startswith("nfs"):
        addr = opt(m["options"], "addr")
        dev = route_dev(addr) if addr else None
        kind = {n["name"]: n["type"] for n in interfaces}.get(dev)
        proto = opt(m["options"], "proto")
        out.update({"server_addr": addr, "proto": proto, "vers": opt(m["options"], "vers"),
                    "nconnect": opt(m["options"], "nconnect"), "xprt": nfs_xprt(m["mountpoint"]),
                    "route_dev": dev,
                    "network": "rdma" if proto == "rdma" else ("ipoib" if kind == "infiniband" else kind)})
    return out


def block_device(source: str):
    """/dev/nvme0n1p2 -> (nvme0n1, rotational); None for what is not a block device."""
    if not source.startswith("/dev/"):
        return None
    name = Path(os.path.realpath(source)).name
    sys_ = Path("/sys/class/block") / name
    if not sys_.exists():
        return None
    disk = Path(os.path.realpath(sys_)).parent.name if (sys_ / "partition").exists() else name
    return {"device": disk, "rotational": read(Path("/sys/block") / disk / "queue" / "rotational")}


def local_candidates(mounts: list, shared_dev: str, extra=()) -> list:
    """Directories that might be a node-local scratch: $TMPDIR, $SLURM_TMPDIR, the site's own (extra, probe
    --local), the usual names, and the mount
    points of local block-device file systems. Each: its mount, free space, whether this user can make a
    directory there (the probe's own, removed at the end)."""
    names = [os.environ.get("TMPDIR"), os.environ.get("SLURM_TMPDIR")] + list(extra) + LOCAL_NAMES
    names += [m["mountpoint"] for m in mounts if m["source"].startswith("/dev/") and m["fstype"] not in MEMORY_FS
              and m["mountpoint"] not in ("/", "/boot", "/boot/efi")]
    out, seen = [], set()
    for n in names:
        if not n or n in seen or not os.path.isdir(n):
            continue
        seen.add(n)
        m = mount_of(n, mounts) or {}
        try:
            st = os.statvfs(n)
            free = st.f_bavail * st.f_frsize / 1e9
        except OSError:
            free = None
        local = m.get("fstype") not in NETWORK_FS and m.get("fstype") not in MEMORY_FS and m.get("source", "").startswith("/dev/")
        out.append({"path": n, "fstype": m.get("fstype"), "source": m.get("source"), "mountpoint": m.get("mountpoint"),
                    "free_gb": round(free, 1) if free is not None else None, "local_block_device": local,
                    "same_device_as_shared": m.get("dev") == shared_dev,
                    "block": block_device(m.get("source", "")) if local else None,
                    "writable": os.access(n, os.W_OK | os.X_OK)})
    return out


def _write_part(job):
    d, names, size = job
    buf = os.urandom(min(size, 4 << 20))
    for name in names:
        with open(os.path.join(d, name), "wb") as f:
            left = size
            while left > 0:
                f.write(buf[:min(left, len(buf))])
                left -= len(buf)


def _read_part(job):
    d, names = job
    total = 0
    for name in names:
        with open(os.path.join(d, name), "rb", buffering=0) as f:
            while True:
                b = f.read(4 << 20)
                if not b:
                    break
                total += len(b)
    return total


def _drop(d, names):
    for name in names:
        fd = os.open(os.path.join(d, name), os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def spread(seconds: list, nbytes: int) -> dict:
    """The repeats of one measurement: raw seconds, median MB/s, and sigma in seconds as the larger of the
    repeats' standard deviation and the clock's resolution (three repeats can give an
    unreasonably small spread)."""
    res = time.get_clock_info("perf_counter").resolution
    sd = statistics.stdev(seconds) if len(seconds) > 1 else 0.0
    med = statistics.median(seconds)
    return {"seconds": [round(s, 4) for s in seconds], "bytes": nbytes, "median_mb_per_s": round(nbytes / med / 1e6, 1),
            "sigma_s": max(sd, res), "stdev_s": sd, "clock_resolution_s": res}


def throughput(base: str, tag: str) -> dict:
    """Write and read back (after dropping the files' pages) each pattern REPEATS times with WRITERS processes in
    base/<user>-bootlab-probe-<job>-<tag>; the directory is removed after. Two write times: until the last file
    is closed ("write", what bootlab's programs wait for: they do not fsync, so a local disk takes the files
    into the page cache while NFS commits each at close) and until syncfs returns ("write_sync", the device;
    for information). A local disk that syncs at 180 MB/s, no faster than an NFS, was still the faster
    scratch for a real step."""
    d = os.path.join(base, f"{getpass.getuser()}-bootlab-probe-{os.environ.get('SLURM_JOB_ID', 'x')}-{tag}")
    os.makedirs(d)
    out = {"dir": d}
    try:
        with multiprocessing.get_context("fork").Pool(WRITERS) as pool:
            for pat, (count, size) in PATTERNS.items():
                names = [f"f{i:04d}" for i in range(count)]
                parts = [names[i::WRITERS] for i in range(WRITERS)]
                w, ws, r = [], [], []
                for _ in range(REPEATS):
                    run(["sync", "-f", d], timeout=600)       # nothing of the last repeat left to write back
                    t0 = time.perf_counter()
                    pool.map(_write_part, [(d, p, size) for p in parts])
                    w.append(time.perf_counter() - t0)
                    sync = run(["sync", "-f", d], timeout=600)
                    ws.append(time.perf_counter() - t0)
                    _drop(d, names)
                    t0 = time.perf_counter()
                    got = sum(pool.map(_read_part, [(d, p) for p in parts]))
                    r.append(time.perf_counter() - t0)
                    if got != count * size or sync["exit"] != 0:
                        raise RuntimeError(f"{pat}: read {got} of {count * size} bytes, sync {sync}")
                    for n in names:
                        os.unlink(os.path.join(d, n))
                out[pat] = {"files": count, "file_bytes": size, "writers": WRITERS,
                            "write": spread(w, count * size), "write_sync": spread(ws, count * size),
                            "read": spread(r, count * size)}
    except Exception as e:      # the item records its failure; the rest of the probe goes on
        out["error"] = f"{type(e).__name__}: {e}"
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return out


def scratch_options(rec: dict, pair: dict = None) -> dict:
    """The fastest way the probe measured is the default: the
    seconds a step's data costs in bootlab's file pattern, write until close plus read (medians of the repeats),
    for each local device measured and for the shared directory. With a pair probe of the node's type (more than
    one node), local adds the copy to the other node (the fastest way measured there) and shared reads cold on the
    other node. path or "shared" -> {seconds, parts, sigma, source}: sigma from the repeats' MAD (x 1.4826,
    the measure that goes with the medians; repeats of 8.16, 1.00, 1.06 s give a standard
    deviation of 4.1 s), for site gen's closeness note only; parts["spikes"]: each repeat over SPIKE x its
    median."""
    def med(x):
        return statistics.median(x["seconds"])

    def mad(x):
        m = med(x)
        return max(MAD_SCALE * statistics.median(abs(s - m) for s in x["seconds"]), x.get("clock_resolution_s", 0))

    def spikes(name, x):
        m = med(x)
        return [f"{name} repeat {i + 1}: {s} s, {s / m:.1f} x the median" for i, s in enumerate(x["seconds"])
                if m > 0 and s > SPIKE * m]
    tp, out = rec["throughput"], {}
    copy = None
    if pair:
        done = [c for c in pair.get("copies", []) if c.get("seconds")]
        if done:
            best = min(done, key=lambda c: statistics.median(c["seconds"]))
            copy = (best["variant"], statistics.median(best["seconds"]))
    src = f"node probe {rec['job']}" + (f", pair probe {pair['job']}" if pair else "")
    for path, v in tp.items():
        if "bootlab" not in v:
            continue
        w, r = med(v["bootlab"]["write"]), med(v["bootlab"]["read"])
        parts = {"write": round(w, 3), "read": round(r, 3)}
        if path == "shared":
            cold = (pair or {}).get("shared", {}).get("cold_read_other_node", {}).get("read_s")
            if cold is not None:
                parts["read"] = round(cold, 3)
                parts["read_is"] = "cold on the other node"
        elif copy:
            parts["copy"] = round(copy[1], 3)
            parts["copy_is"] = copy[0]
        sigma = (mad(v["bootlab"]["write"]) ** 2 + (0 if parts.get("read_is") else mad(v["bootlab"]["read"]) ** 2)) ** 0.5
        found = spikes("write", v["bootlab"]["write"]) + ([] if parts.get("read_is") else spikes("read", v["bootlab"]["read"]))
        if found:
            parts["spikes"] = found
        out[path] = {"seconds": round(sum(x for x in parts.values() if isinstance(x, float)), 3), "parts": parts,
                     "sigma": sigma, "source": src}
    return out


def hardlink(base: str) -> dict:
    d = os.path.join(base, f"{getpass.getuser()}-bootlab-probe-{os.environ.get('SLURM_JOB_ID', 'x')}-link")
    os.makedirs(d)
    try:
        a, b = os.path.join(d, "a"), os.path.join(d, "b")
        Path(a).write_text("x")
        os.link(a, b)
        return {"ok": os.stat(a).st_ino == os.stat(b).st_ino}
    except OSError as e:
        return {"ok": False, "error": str(e)}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def server_clock(base: str) -> dict:
    """The shared file system's clock against this node's: a new file's mtime (set by the server for NFS) minus
    the middle of the local times around its creation."""
    p = os.path.join(base, f"{getpass.getuser()}-bootlab-probe-{os.environ.get('SLURM_JOB_ID', 'x')}-clock")
    t0 = time.time()
    Path(p).write_text("x")
    t1 = time.time()
    try:
        return {"offset_s": round(os.stat(p).st_mtime - (t0 + t1) / 2, 3), "uncertainty_s": round((t1 - t0) / 2, 3)}
    finally:
        os.unlink(p)


def apptainer_setup() -> str:
    """On the login node, in the shell the user works in: a shell line that puts this apptainer on PATH in a batch
    job. From a module when one named apptainer* is loaded (its MODULEPATH with it: a cluster's modules can be
    outside the default MODULEPATH), else its directory. SystemExit when there is no apptainer here."""
    path = shutil.which("apptainer")
    if not path:
        raise SystemExit("no apptainer on PATH here: make it available in this shell (module load ...) and run "
                         "probe plan again")
    mod = next((m for m in os.environ.get("LOADEDMODULES", "").split(":") if m.startswith("apptainer")), None)
    if mod:
        init = next((p for p in ("/etc/profile.d/modules.sh", "/usr/share/lmod/lmod/init/bash")
                     if os.path.exists(p)), None)
        return (f"source {init} >/dev/null 2>&1; " if init else "") + \
            f"export MODULEPATH={shlex.quote(os.environ.get('MODULEPATH', ''))}; module load {mod} >/dev/null 2>&1;"
    return f"export PATH={shlex.quote(str(Path(path).parent))}:$PATH;"


def find_apptainer(setup: str) -> dict:
    """How apptainer gets on PATH on this node: already there, a login shell, or the line probe plan made on the
    login node (setup). The first that works is the line for the site's setup.sh."""
    tries = [("PATH", ""), ("login shell", "source /etc/profile >/dev/null 2>&1;"), ("plan", setup or "")]
    for how, pre in tries:
        r = run(["bash", "-c", f"{pre} command -v apptainer && apptainer --version"])
        if r["exit"] == 0:
            path, ver = (r["out"].strip().splitlines() + [""])[:2]
            return {"how": how, "setup": pre, "path": path, "version": ver}
    return {"how": None, "tried": [t[1] for t in tries]}


def parse_ucx_devices(text: str) -> list:
    """ucx_info -d -> [(transport, device)]."""
    out, tl = [], None
    for line in text.splitlines():
        m = re.match(r"#\s+Transport:\s+(\S+)", line)
        if m:
            tl = m[1]
            continue
        m = re.match(r"#\s+Device:\s+(\S+)", line)
        if m and tl:
            out.append([tl, m[1]])
    return out


def image(sif: str, setup: str) -> dict:
    ex = f"{setup} apptainer exec {shlex.quote(sif)}"
    v = run(["bash", "-c", f"{ex} ucx_info -v"])
    d = run(["bash", "-c", f"{ex} ucx_info -d"], timeout=120)
    py = run(["bash", "-c", f"{ex} python3 --version"])
    return {"ucx_version": v["out"].strip().splitlines()[:2], "devices": parse_ucx_devices(d["out"]),
            "ucx_info_exit": d["exit"], "python": (py["out"] + py["err"]).strip()}


def in_job() -> bool:
    """srun outside an allocation makes a new job of its own (it did in a login-node dry run): the
    items that start a step run only inside the probe's job."""
    return "SLURM_JOB_ID" in os.environ


def cgroup() -> dict:
    ctl = read("/sys/fs/cgroup/cgroup.controllers")
    if not in_job():
        return {"v2_controllers": ctl.split() if ctl else None, "step_probe": "skipped: not in a job"}
    step = run(["srun", "--overlap", "--ntasks=1", "--cpus-per-task=1", "--mem=100M", "--kill-on-bad-exit=1",
                "bash", "-c", 'p=$(sed -n "s/^0:://p" /proc/self/cgroup); cat "/sys/fs/cgroup${p%/*}/memory.peak"'],
               timeout=120)
    return {"v2_controllers": ctl.split() if ctl else None, "self": read("/proc/self/cgroup"),
            "step_memory_peak_readable": step["exit"] == 0 and step["out"].strip().isdigit(),
            "step_probe": {"exit": step["exit"], "err": step["err"][-300:]}}


def pythons() -> list:
    out, seen = [], set()
    for n in ["python3", "python3.13", "python3.12", "python3.11", "python3.10", "python3.9"]:
        p = shutil.which(n)
        if p and os.path.realpath(p) not in seen:
            seen.add(os.path.realpath(p))
            v = run([p, "--version"])
            out.append({"path": p, "version": (v["out"] + v["err"]).strip()})
    return out


PY_TEST = r"""
import importlib, json, os, socket, subprocess, sys
mods = {}
for m in ("hashlib", "multiprocessing", "tarfile", "zlib", "lzma", "bz2", "ssl", "sqlite3", "ctypes", "decimal",
          "fractions", "statistics", "tomllib"):
    try:
        importlib.import_module(m)
        mods[m] = "ok"
    except Exception as e:
        mods[m] = f"{type(e).__name__}: {e}"
def sh(c):
    p = subprocess.run(c, capture_output=True, text=True, timeout=120)
    return {"cmd": c, "exit": p.returncode, "out": p.stdout[-300:], "err": p.stderr[-300:]}
host = socket.gethostname().split(".")[0]
child = subprocess.run([sys.executable, "-c", "import sys, os; print(sys.prefix, os.__file__)"], capture_output=True,
                       text=True, timeout=60)
print(json.dumps({"version": sys.version, "executable": sys.executable, "prefix": sys.prefix, "modules": mods,
    "stdlib": os.__file__, "child": {"exit": child.returncode, "out": child.stdout.strip(), "err": child.stderr[-300:]},
    "env_leak": [k for k in ("PYTHONHOME", "PYTHONPATH", "LD_LIBRARY_PATH") if k in os.environ],
    "srun": sh(["srun", "--overlap", "--ntasks=1", "--cpus-per-task=1", "--mem=100M", "--kill-on-bad-exit=1", "hostname"]),
    "scontrol": sh(["scontrol", "show", "hostnames", os.environ.get("SLURM_JOB_NODELIST", host)]),
    "sinfo": sh(["sinfo", "-h", "-n", host, "-o", "%N %T"])}))
"""


def py_test(d: str) -> dict:
    """A Python copied out of the image with the
    image's own loader and libraries, run on this host without LD_LIBRARY_PATH;
    its children (srun, scontrol, sinfo) are the host's and must work as from the host's shell."""
    if not in_job():
        return {"skipped": "not in a job (its srun would make a job of its own)"}
    d = Path(d)
    env = {k: v for k, v in os.environ.items() if k not in ("LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONHOME")}
    wrapper = d / "bin" / "python3"
    if wrapper.is_file() and wrapper.read_bytes()[:2] == b"#!":
        # r4's hostpython (r4-requirements.md 3.3): the wrapper runs the loader with --argv0, no PYTHONHOME
        cmd, how = [str(wrapper)], "wrapper"
    else:   # py_extract.sh's tree (r3's 3.9): the loader by hand, PYTHONHOME set
        loader = next(iter(sorted((d / "so").glob("ld-linux*.so*"))), None)
        py = next(iter(sorted((d / "bin").glob("python3.*"))), None)
        if not loader or not py:
            return {"error": f"no loader or python in {d}"}
        env["PYTHONHOME"] = str(d)
        cmd, how = [str(loader), "--library-path", str(d / "so"), str(py)], "loader"
    try:
        p = subprocess.run(cmd + ["-c", PY_TEST], capture_output=True, text=True, timeout=300, env=env)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"error": f"{type(e).__name__}: {e}"}
    out = {"how": how, "exit": p.returncode, "err": p.stderr[-2000:]}
    try:
        out.update(json.loads(p.stdout.strip().splitlines()[-1]))
    except (ValueError, IndexError):
        out["out"] = p.stdout[-2000:]
    return out


def timezone() -> dict:
    return {"name": time.strftime("%Z"), "offset": time.strftime("%z"),
            "localtime": os.path.realpath("/etc/localtime") if os.path.exists("/etc/localtime") else None}


def cmd_node(args):
    out = Path(args.out)
    host = socket.gethostname().split(".")[0]
    job = os.environ.get("SLURM_JOB_ID", "nojob")
    rec = {"format": FORMAT, "what": "node", "host": host, "job": job, "start": now(), "timezone": timezone(),
           "user": getpass.getuser()}
    others = run(["squeue", "-h", "-w", host, "-o", "%i %u %j %C"])
    rec["other_jobs_at_start"] = [l for l in others["out"].splitlines() if not l.startswith(job + " ")]
    ib0 = ib_ports()
    rec["cpu"] = cpu()
    rec["nics"] = nics()
    mounts = parse_mountinfo(read("/proc/self/mountinfo", ""))
    Path(args.shared).mkdir(parents=True, exist_ok=True)
    rec["shared"] = shared_fs(args.shared, mounts, rec["nics"])
    rec["shared"]["hardlink"] = hardlink(args.shared)
    rec["shared"]["clock"] = server_clock(args.shared)
    rec["local_extra"] = args.local
    rec["local"] = local_candidates(mounts, rec["shared"].get("dev"), args.local)
    rec["apptainer"] = find_apptainer(args.setup)
    rec["image"] = image(args.sif, rec["apptainer"].get("setup") or "") if rec["apptainer"]["how"] else None
    rec["cgroup"] = cgroup()
    rec["pythons"] = pythons()
    if args.py_test:
        rec["py_test"] = py_test(args.py_test)
    # throughput last: the longest item; one directory per distinct local device, then the shared directory
    rec["throughput"] = {}
    done = set()
    for c in rec["local"]:
        key = (c["block"] or {}).get("device")
        if not (c["local_block_device"] and c["writable"] and not c["same_device_as_shared"]) or key in done:
            continue
        if (c["free_gb"] or 0) < MIN_FREE_GB:
            c["not_measured"] = f"free {c['free_gb']} GB < {MIN_FREE_GB}"
            continue
        done.add(key)
        rec["throughput"][c["path"]] = throughput(c["path"], "local")
    rec["throughput"]["shared"] = throughput(args.shared, "shared")
    rec["ib"] = {"start": ib0, "end": ib_ports()}
    rec["ib"]["chosen"] = choose_ib(ib0, rec["ib"]["end"], [rec["shared"].get("route_dev")])
    rec["end"] = now()
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"node-{host}-{job}.json"
    f.write_text(json.dumps(rec, indent=1) + "\n")
    print(f)


# ---------------------------------------------------------------------------------------------- local

def allowed_cpus(text: str) -> set:
    """'3,99' or '0-3,8' (Cpus_allowed_list) -> {3, 99}."""
    out = set()
    for part in text.strip().split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


def core_threads(sysdir: Path = Path("/sys/devices/system/cpu")) -> dict:
    """cpu -> the set of hardware threads of its physical core (thread_siblings_list)."""
    out = {}
    for d in sysdir.glob("cpu[0-9]*"):
        f = d / "topology" / "thread_siblings_list"
        if f.exists():
            out[int(d.name[3:])] = allowed_cpus(f.read_text())
    return out


def binding_check(lines: list, cpus: list, threads: dict) -> dict:
    """Each rank's Cpus_allowed_list against cpus (the first thread of each physical core): every rank on one
    cpu of the list alone, no hyperthread, no two ranks on one cpu. threads (the cores' threads) only explain
    a failure: a rank on a whole core shows both of its threads."""
    want = {frozenset({c}) for c in cpus}
    got = [frozenset(allowed_cpus(l.split(":", 1)[1])) for l in lines if l.startswith("Cpus_allowed_list:")]
    bad = [sorted(g) for g in got if g not in want]
    return {"ranks": len(got), "on_a_core_of_the_list": not bad, "distinct_cores": len(set(got)) == len(got),
            "ok": len(got) == len(cpus) and not bad and len(set(got)) == len(got), "outside": bad[:4]}


def mpi_local(args, cpus: list, logs: Path) -> dict:
    """mpirun of the image on every physical core, REPEATS times: each rank's binding; then two halves of the
    cores at once (disjoint --cpu-set); with --sdp, sdpb on the small SDP with every core (2 iterations)."""
    from . import execute
    site = execute.Site(name="probe", sif=args.sif, binds=[str(logs.resolve())], generator="", scratch="", runs="",
                        pmp2sdp="", dynamical_sdp="", driver_reserve_mb=0, bench_dir=Path("."))
    env = {**execute.local_env(), **MPI_ENV, "UCX_TLS": "self,sm"}
    threads = core_threads()
    show = ["sh", "-c", "grep Cpus_allowed_list /proc/self/status"]
    out = {"cpus": cpus, "full": [], "halves": []}
    for rep in range(REPEATS):
        log = logs / f"mpi-full-r{rep}.log"
        rf = execute.write_rankfile(logs / f"mpi-full-r{rep}.rankfile", cpus)
        r = limited(execute.local_argv(site, show, cpus, True, rf), STARTUP_LIMIT, log, env)
        chk = binding_check(log.read_text(errors="replace").splitlines(), cpus, threads)
        out["full"].append({**r, **chk, "ok": chk["ok"] and r["exit"] == 0})
    half = len(cpus) // 2
    for rep in range(REPEATS):
        parts = [cpus[:half], cpus[half:2 * half]]
        rfs = [execute.write_rankfile(logs / f"mpi-half{k}-r{rep}.rankfile", p) for k, p in enumerate(parts)]
        procs = [subprocess.Popen(execute.local_argv(site, show, p, True, rf), stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, env=env, start_new_session=True) for p, rf in zip(parts, rfs)]
        texts = [p.communicate(timeout=STARTUP_LIMIT)[0].decode(errors="replace") for p in procs]
        for k, t in enumerate(texts):
            (logs / f"mpi-half{k}-r{rep}.log").write_text(t)
        checks = [binding_check(t.splitlines(), p, threads) for t, p in zip(texts, parts)]
        out["halves"].append({"exit": [p.returncode for p in procs], "checks": checks,
                              "ok": all(c["ok"] for c in checks) and all(p.returncode == 0 for p in procs)})
    if args.sdp:
        out["sdpb"] = []
        for rep in range(REPEATS):
            w = logs / f"sdpb-r{rep}"
            w.mkdir(parents=True)
            site.binds = [str(logs.resolve()), args.sdp, str(w.resolve())]
            rf = execute.write_rankfile(logs / f"sdpb-r{rep}.rankfile", cpus)
            cmd = execute.local_argv(site, ["sdpb_3.0sky", "--sdpDir", args.sdp, "--outDir", str(w / "out"),
                                            "--checkpointDir", str(w / "ck"), "--precision", str(args.sdp_precision),
                                            "--maxIterations", "2", "--verbosity", "1"], cpus, True, rf)
            log = logs / f"sdpb-r{rep}.log"
            r = limited(cmd, STARTUP_LIMIT, log, env)
            out["sdpb"].append({**r, **count_errors(log.read_text(errors="replace"))})
            shutil.rmtree(w, ignore_errors=True)
    out["ok"] = all(x["ok"] for x in out["full"] + out["halves"]) and \
        all(x["exit"] == 0 for x in out.get("sdpb", []))
    return out


def systemd_scope() -> dict:
    """Can bootlab put a pmp2sdp instance in a systemd user scope with a memory limit (memory_guard =
    systemd)? A scope with MemoryMax=64M whose own memory.max says so."""
    if not shutil.which("systemd-run"):
        return {"available": False, "why": "no systemd-run"}
    r = run(["systemd-run", "--user", "--scope", "--quiet", "-p", "MemoryMax=64M", "sh", "-c",
             "cat /sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)/memory.max"])
    ok = r["exit"] == 0 and r["out"].strip() == str(64 << 20)
    return {"available": ok, "why": "a scope's memory.max is set" if ok else
            f"exit {r['exit']}: {(r['out'] + r['err']).strip()[-300:]}"}


def cmd_local(args):
    """bootlab probe local: this machine, directly, no Slurm: the CPU and its physical
    cores, the memory, the runs directory and the local disks (bootlab's file pattern), apptainer and the image,
    mpirun of the image on every core (the bindings), whether a systemd user scope can bound the memory. Writes
    <out>/node-<host>-local.json and <out>/probe.json for site gen (plan what = local)."""
    from . import execute
    out = Path(args.out)
    logs = out / "local-logs"
    logs.mkdir(parents=True, exist_ok=True)
    args.shared = args.shared or execute.install_runs()
    Path(args.shared).mkdir(parents=True, exist_ok=True)
    host = socket.gethostname().split(".")[0]
    mounts = parse_mountinfo(read("/proc/self/mountinfo", ""))
    cpus = execute.physical_cpus(os.sched_getaffinity(0))
    mem = {k: int(v.split()[0]) for k, v in (l.split(":", 1) for l in read("/proc/meminfo", "").splitlines())}
    rec = {"format": FORMAT, "what": "node", "host": host, "job": "local", "start": now(), "timezone": timezone(),
           "user": getpass.getuser(), "cpu": cpu(), "physical_cpus": cpus,
           "memory_mb": {k: mem[k] // 1024 for k in ("MemTotal", "MemAvailable", "SwapTotal") if k in mem},
           "nics": nics()}
    rec["shared"] = shared_fs(args.shared, mounts, rec["nics"])
    rec["shared"]["hardlink"] = hardlink(args.shared)
    rec["local_extra"] = args.local
    rec["local"] = local_candidates(mounts, rec["shared"].get("dev"), args.local)
    setup = apptainer_setup()
    rec["apptainer"] = find_apptainer(setup)
    rec["image"] = image(args.sif, rec["apptainer"].get("setup") or "") if rec["apptainer"]["how"] else None
    rec["mpi"] = mpi_local(args, cpus, logs)
    rec["memory_guard"] = systemd_scope()
    rec["throughput"] = {}
    done = set()
    for c in rec["local"]:
        key = (c["block"] or {}).get("device")
        if not (c["local_block_device"] and c["writable"] and not c["same_device_as_shared"]) or key in done:
            continue
        if (c["free_gb"] or 0) < MIN_FREE_GB:
            c["not_measured"] = f"free {c['free_gb']} GB < {MIN_FREE_GB}"
            continue
        done.add(key)
        rec["throughput"][c["path"]] = throughput(c["path"], "local")
    rec["throughput"]["shared"] = throughput(args.shared, "shared")
    rec["ib"] = {"start": [], "end": [], "chosen": None}
    rec["end"] = now()
    (out / f"node-{host}-local.json").write_text(json.dumps(rec, indent=1) + "\n")
    plan = {"format": FORMAT, "what": "local", "time": rec["start"], "host": host, "user": rec["user"],
            "sif": args.sif, "shared": args.shared, "shared_where": where(args.shared, mounts),
            "apptainer_setup": setup}
    probe = {"format": FORMAT, "created": now(), "plan": plan, "nodes": {host: [rec]}, "pairs": [],
             "types": {"local": {"hosts": [host], "physical_cores": len(cpus),
                                 "mem_mb": rec["memory_mb"]["MemTotal"]}}}
    (out / "probe.json").write_text(json.dumps(probe, indent=1) + "\n")
    m = rec["mpi"]
    print(f"{host}: {len(cpus)} physical cores, {rec['memory_mb']} MB; mpirun on every core "
          f"{'ok' if m['ok'] else 'FAILED'}; systemd scope {'yes' if rec['memory_guard']['available'] else 'no'}")
    print(out / "probe.json")


# ---------------------------------------------------------------------------------------------- pair

def limited(cmd: list, seconds: int, log: Path, env: dict = None) -> dict:
    """cmd in a process group of its own, stdout and stderr to log; at the limit TERM to the group, a minute
    later KILL (an srun that lost ranks once waited 2.7 h). {exit, seconds, timed_out}."""
    t0 = time.perf_counter()
    with open(log, "wb") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
                             start_new_session=True)
        try:
            rc, out = p.wait(timeout=seconds), False
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            rc, out = p.returncode, True
    return {"exit": rc, "seconds": round(time.perf_counter() - t0, 1), "timed_out": out}


def light(hosts: list, argv: list, timeout: int = 120) -> dict:
    """A light step (one task a node, --overlap) with a start limit: an srun across a firewall hangs."""
    from . import execute
    return run(execute.srun_light(hosts) + ["--kill-on-bad-exit=1"] + argv, timeout=timeout)


def facts(server: str = None) -> dict:
    """This node's IB ports and interfaces, its physical cores, and the interface its own route to the shared file
    system's server goes over (pair asks each node: the route can differ by node, two nodes of one type went over
    different interfaces)."""
    ib = ib_ports()
    n = nics()
    return {"host": socket.gethostname().split(".")[0], "ib": ib, "nics": n, "cores": cpu()["physical_cores"],
            "route_dev": route_dev(server) if server else None}


def prefix_of(cidr: str) -> str:
    """a.b.c.d/24 -> 'a.b.c.' (whole octets of the prefix: rank_env.sh's subnet: rules)."""
    ip, plen = cidr.split("/")
    return ".".join(ip.split(".")[:max(1, int(plen) // 8)]) + "."


def shared_networks(a: dict, b: dict) -> list:
    """The IPv4 networks two nodes share, each [kind, prefix, (iface, address) on a, on b]: IPoIB and Ethernet,
    not loopback or link-local."""
    def nets(f):
        out = {}
        for n in f["nics"]:
            if n["type"] not in ("ethernet", "infiniband") or n["operstate"] not in ("up", "unknown"):
                continue
            for c in n["ipv4"]:
                if not c.startswith("169.254."):
                    out[prefix_of(c)] = (n["name"], c.split("/")[0], "ipoib" if n["type"] == "infiniband" else "ethernet",
                                         n["speed_mbps"])
        return out
    na, nb = nets(a), nets(b)
    return [{"kind": na[p][2], "prefix": p, "a": na[p][:2], "b": nb[p][:2], "speed_mbps": na[p][3]}
            for p in sorted(na) if p in nb]


def parse_perftest(text: str) -> dict:
    """ucx_perftest's Final line: latency (usec) and bandwidth (MB/s), overall."""
    for line in text.splitlines():
        f = line.split()
        if f and f[0] == "Final:" and len(f) >= 9:
            return {"latency_us": float(f[4]), "bandwidth_mb_per_s": float(f[6]), "iterations": int(f[1])}
    return {}


def perftest(sif: str, appt: str, hosts: list, env_a: dict, env_b: dict, server_addr: str, test: list, log: Path) -> dict:
    """ucx_perftest: a server on hosts[0], then the client on hosts[1], both in the image, an out-of-band port
    from the ephemeral range."""
    port = random.randint(40000, 60000)
    ex = f"{shlex.quote(appt)} exec {shlex.quote(sif)} ucx_perftest"
    srv = " ".join(f"{k}={shlex.quote(v)}" for k, v in env_a.items())
    cli = " ".join(f"{k}={shlex.quote(v)}" for k, v in env_b.items())
    from . import execute
    server = subprocess.Popen(execute.srun_light([hosts[0]]) + ["--kill-on-bad-exit=1", "bash", "-c",
                              f"env {srv} timeout 100 {ex} -p {port}"], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    time.sleep(5)
    c = light([hosts[1]], ["bash", "-c", f"env {cli} timeout 90 {ex} {server_addr} -p {port} " + " ".join(test)], timeout=120)
    try:
        s_out = server.communicate(timeout=30)[0].decode(errors="replace")
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGTERM)
        s_out = server.communicate()[0].decode(errors="replace")
    log.write_text(f"server:\n{s_out}\nclient {c['exit']}:\n{c['out']}\n{c['err']}")
    return {"exit": c["exit"], **parse_perftest(c["out"]), "err": c["err"][-300:] if c["exit"] else ""}


def count_errors(text: str) -> dict:
    return {k: text.count(v) for k, v in ERRORS.items()}


def startup(args, hosts: list, cores: int, transport: dict, rep: int, work: Path, logs: Path) -> dict:
    """The full-scale MPI start-up: every physical core of both nodes a rank of sdpb on the small SDP, 2
    iterations, the ranks through bootlab's rank_env.sh with this transport's UCX_TLS and device rules."""
    from . import execute
    env = {k: v for k, v in os.environ.items() if k not in MPI_UNSET}
    env.update(MPI_ENV)
    env.update({"UCX_TLS": transport["tls"], "BOOTLAB_NET_DEVICES": transport["devices"]})
    w = work / f"{transport['name']}-r{rep}"
    w.mkdir(parents=True)
    mem = execute.job_mem_mb() - 4096
    cmd = ["srun", f"--nodes={len(hosts)}", "-w", ",".join(hosts), f"--ntasks-per-node={cores}",
           f"--ntasks={cores * len(hosts)}", "--cpus-per-task=1", f"--mem={mem}M", "--hint=nomultithread",
           "--cpu-bind=cores", "--kill-on-bad-exit=1", "--mpi=pmix", "bash", str(execute.RANK_ENV),
           args.apptainer, "exec", "--bind", args.sdp, "--bind", str(w), args.sif,
           "sdpb_3.0sky", "--sdpDir", args.sdp, "--outDir", str(w / "out"), "--checkpointDir", str(w / "ck"),
           "--precision", str(args.sdp_precision), "--maxIterations", "2", "--verbosity", "1"]
    log = logs / f"startup-{transport['name']}-r{rep}.log"
    r = limited(cmd, STARTUP_LIMIT, log, env)
    text = log.read_text(errors="replace")
    shutil.rmtree(w, ignore_errors=True)
    return {**r, **count_errors(text), "log": str(log)}


def copies(args, hosts: list, nets: list, logs: Path) -> list:
    """bootlab's own copy of 2 GiB of 4 MiB files from node 0's local scratch to the other's (execute.distribute,
    8 streams, sha256 of every file checked) REPEATS times over each network with transfer srun (receive.py), and
    with transfer ssh over IPoIB when ssh into the other node works. Times only; the cost
    against a step is worked out with the bench."""
    from . import execute
    mounts = parse_mountinfo(read("/proc/self/mountinfo", ""))
    shared_dev = (mount_of(args.shared, mounts) or {}).get("dev")
    cands = [c for c in local_candidates(mounts, shared_dev, args.local) if c["local_block_device"] and c["writable"]
             and not c["same_device_as_shared"] and (c["free_gb"] or 0) >= MIN_FREE_GB]
    if not cands:
        return [{"error": "no local scratch candidate on node 0"}]
    base = max(cands, key=lambda c: c["free_gb"])["path"]
    src = Path(base) / f"{getpass.getuser()}-bootlab-probe-{os.environ.get('SLURM_JOB_ID', 'x')}-copy"
    src.mkdir()
    count, size = PATTERNS["bootlab"]
    names = [f"f{i:04d}" for i in range(count)]
    with multiprocessing.get_context("fork").Pool(WRITERS) as pool:
        pool.map(_write_part, [(str(src), names[i::WRITERS], size) for i in range(WRITERS)])
    node = execute.Node(hosts[0], "", 0, 0, hosts)
    base_site = dict(name="probe", sif=args.sif, binds=[str(src.parent)], generator="", scratch="", runs="",
                     pmp2sdp="", dynamical_sdp="", driver_reserve_mb=0, bench_dir=Path("."), python=sys.executable)
    variants = [(f"srun-{n['kind']}", {"transfer": "srun", "copy_network": n["prefix"], "copy_streams": 8})
                for n in nets]
    ipoib = next((n for n in nets if n["kind"] == "ipoib"), None)
    ssh_ok = run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", hosts[1], "true"], timeout=30)["exit"] == 0
    if ssh_ok and ipoib:
        variants.append(("ssh-ipoib", {"transfer": "ssh", "ipoib_prefix": ipoib["prefix"], "copy_streams": 8,
                                       "ssh": ["ssh", "-o", "BatchMode=yes"], "remote_setup": args.setup}))
    out = [{"src": str(src), "files": count, "bytes": count * size, "ssh_into_job_nodes": ssh_ok}]
    env0 = dict(os.environ)
    os.environ["BOOTLAB_COPY_SHA256"] = "1"
    try:
        for name, mn in variants:
            site = execute.Site(**base_site, multinode=mn)
            secs, err = [], None
            for _ in range(REPEATS):
                try:
                    execute.on_hosts(site, node, f"rm -rf {shlex.quote(str(src))}", hosts[1:])
                    d = execute.distribute(site, node, src)
                    secs.append(d["seconds"])
                except Exception as e:      # recorded; the next variant goes on
                    err = f"{type(e).__name__}: {str(e)[-400:]}"
                    break
            out.append({"variant": name, **mn, "ssh": None, "seconds": secs, "error": err,
                        **({"median_mb_per_s": round(count * size / statistics.median(secs) / 1e6, 1)} if secs else {})})
    finally:
        os.environ.clear()
        os.environ.update(env0)
        try:
            execute.on_hosts(execute.Site(**base_site, multinode={"transfer": "srun"}), node,
                             f"rm -rf {shlex.quote(str(src))}", hosts[1:])
        except Exception:
            pass
        shutil.rmtree(src, ignore_errors=True)
    return out


def cmd_io(args):
    """One node's part of pair's shared-directory items: write the bootlab pattern into dir (time until close and
    until syncfs), or read every file in dir after dropping its pages. JSON on stdout."""
    d = Path(args.dir)
    count, size = PATTERNS["bootlab"]
    names = [f"f{i:04d}" for i in range(count)]
    with multiprocessing.get_context("fork").Pool(WRITERS) as pool:
        if args.op == "write":
            d.mkdir(parents=True)
            t0 = time.perf_counter()
            pool.map(_write_part, [(str(d), names[i::WRITERS], size) for i in range(WRITERS)])
            t1 = time.perf_counter()
            run(["sync", "-f", str(d)], timeout=600)
            print(json.dumps({"close_s": round(t1 - t0, 4), "sync_s": round(time.perf_counter() - t0, 4), "bytes": count * size}))
        else:
            _drop(str(d), names)
            t0 = time.perf_counter()
            got = sum(pool.map(_read_part, [(str(d), names[i::WRITERS]) for i in range(WRITERS)]))
            print(json.dumps({"read_s": round(time.perf_counter() - t0, 4), "bytes": got}))


def shared_pair(args, hosts: list) -> dict:
    """Both nodes write the bootlab pattern into the shared directory at once (REPEATS times), then the second
    node reads what the first wrote (never read there: no page cache)."""
    py = shlex.quote(sys.executable)
    src = shlex.quote(str(Path(__file__).resolve().parents[1]))
    base = Path(args.shared) / f"{getpass.getuser()}-bootlab-probe-{os.environ.get('SLURM_JOB_ID', 'x')}-pair"
    out = {"together": []}
    try:
        for rep in range(REPEATS):
            d = base / f"r{rep}"
            r = light(hosts, ["bash", "-c", f"PYTHONPATH={src} {py} -m bootlab.probe io --op write "
                                            f"--dir {shlex.quote(str(d))}/$(hostname -s)"], timeout=900)
            out["together"].append({"exit": r["exit"], "per_node": [json.loads(l) for l in r["out"].splitlines()
                                                                    if l.startswith("{")], "err": r["err"][-300:]})
            if rep < REPEATS - 1:
                shutil.rmtree(d, ignore_errors=True)
        last = base / f"r{REPEATS - 1}" / hosts[0]
        r = light([hosts[1]], ["bash", "-c", f"PYTHONPATH={src} {py} -m bootlab.probe io --op read "
                                             f"--dir {shlex.quote(str(last))}"], timeout=600)
        out["cold_read_other_node"] = json.loads(r["out"].strip().splitlines()[-1]) if r["exit"] == 0 else {"err": r["err"][-300:]}
    finally:
        shutil.rmtree(base, ignore_errors=True)
    return out


def cmd_pair(args):
    from . import execute
    out = Path(args.out)
    hosts = execute.job_hosts(socket.gethostname().split(".")[0])
    if not hosts or len(hosts) != 2:
        raise SystemExit(f"pair: a job of two nodes, not {hosts}")
    job = os.environ.get("SLURM_JOB_ID", "nojob")
    logs = out / f"pair-{job}"
    logs.mkdir(parents=True)
    rec = {"format": FORMAT, "what": "pair", "hosts": hosts, "job": job, "start": now(), "timezone": timezone()}
    rec["apptainer"] = find_apptainer(args.setup)
    if not rec["apptainer"]["how"]:
        raise SystemExit("pair: no apptainer here")
    args.apptainer = rec["apptainer"]["path"]
    os.environ["PATH"] = f"{Path(args.apptainer).parent}:{os.environ['PATH']}"   # the copies' tar, the light steps
    f = out / f"pair-{hosts[0]}-{hosts[1]}-{job}.json"

    def save():
        rec["end"] = now()
        f.write_text(json.dumps(rec, indent=1) + "\n")

    r = light(hosts, ["hostname", "-s"])
    rec["light_srun"] = {"exit": r["exit"], "out": r["out"].split(), "err": r["err"][-300:]}
    if r["exit"] != 0:
        save()
        raise SystemExit(f"pair: a light srun across {hosts} failed: {r['err'][-300:]}")
    py, src = shlex.quote(sys.executable), shlex.quote(str(Path(__file__).resolve().parents[1]))
    mounts = parse_mountinfo(read("/proc/self/mountinfo", ""))
    server = shared_fs(args.shared, mounts, nics()).get("server_addr")
    r = light(hosts, ["bash", "-c", f"PYTHONPATH={src} {py} -c 'import json; from bootlab import probe; "
                                    f"print(json.dumps(probe.facts({json.dumps(server) if server else None})))'"])
    fs = {x["host"]: x for x in map(json.loads, r["out"].splitlines())}
    a, b = fs[hosts[0]], fs[hosts[1]]
    # each node's port by the rule rank_env.sh's ib:auto applies there (its own route breaks a tie)
    ports = {h: choose_ib(fs[h]["ib"], fs[h]["ib"], [fs[h]["route_dev"]]) for h in hosts}
    nets = shared_networks(a, b)
    rec.update({"cores": a["cores"], "ib_ports": ports, "networks": nets})
    save()
    # ucx_perftest: rc over the chosen IB ports, tcp over each shared network
    tests = {"bw": ["-t", "tag_bw", "-s", str(4 << 20), "-n", "200"], "lat": ["-t", "tag_lat", "-s", "8", "-n", "2000"]}
    oob = nets[0]["a"][1] if nets else hosts[0]
    rec["perftest"] = []
    if all(ports.values()):
        for t, argv in tests.items():
            rec["perftest"].append({"tls": "rc", "devices": ports, "test": t, **perftest(
                args.sif, args.apptainer, hosts, {"UCX_TLS": "rc,self", "UCX_NET_DEVICES": ports[hosts[0]]},
                {"UCX_TLS": "rc,self", "UCX_NET_DEVICES": ports[hosts[1]]}, oob, argv, logs / f"perftest-rc-{t}.log")})
    for n in nets:
        for t, argv in tests.items():
            rec["perftest"].append({"tls": "tcp", "network": n["kind"], "prefix": n["prefix"], "test": t, **perftest(
                args.sif, args.apptainer, hosts, {"UCX_TLS": "tcp,self", "UCX_NET_DEVICES": n["a"][0]},
                {"UCX_TLS": "tcp,self", "UCX_NET_DEVICES": n["b"][0]}, n["a"][1], argv,
                logs / f"perftest-tcp-{n['kind']}-{t}.log")})
    save()
    rec["copies"] = copies(args, hosts, nets, logs)
    save()
    rec["shared"] = shared_pair(args, hosts)
    save()
    # the full-scale start-up, last: the longest; a run starts only with STARTUP_LIMIT + 2 min left in the job
    transports = []
    if all(ports.values()):
        transports.append({"name": "rc", "tls": "rc,self,sm", "devices": " ".join(f"{h}={p}" for h, p in ports.items())})
    for n in nets:
        transports.append({"name": f"tcp-{n['kind']}", "tls": "tcp,self,sm", "devices": f"subnet:{n['prefix']}"})
    work = Path(args.shared) / f"{getpass.getuser()}-bootlab-probe-{job}-startup"
    rec["startup"] = []
    try:
        for rep in range(STARTUP_REPEATS):
            for t in transports:
                left = execute.job_seconds_left()
                if left < STARTUP_LIMIT + 120:
                    rec["startup"].append({**t, "rep": rep, "skipped": f"{left} s left in the job"})
                    continue
                rec["startup"].append({**t, "rep": rep, **startup(args, hosts, a["cores"], t, rep, work, logs)})
                save()
    finally:
        shutil.rmtree(work, ignore_errors=True)
    save()
    print(f)


def cmd_merge(args):
    d = Path(args.dir)
    plan = json.loads((d / "plan.json").read_text())
    nodes = {}
    for f in sorted(d.glob("node-*.json")):
        r = json.loads(f.read_text())
        nodes.setdefault(r["host"], []).append(r)
    pairs = [json.loads(f.read_text()) for f in sorted(d.glob("pair-*.json"))]
    type_of = {h: t for t, v in plan["types"].items() for h in v["hosts"]}
    pair_of = {type_of.get(p["hosts"][0]): p for p in pairs}
    probe = {"format": FORMAT, "created": now(), "plan": plan, "nodes": nodes, "pairs": pairs,
             "types": {name: {"hosts": t["hosts"], "physical_cores": t["physical_cores"], "mem_mb": t["mem_mb"],
                              "probed": [h for h in t["hosts"] if h in nodes]}
                       for name, t in plan["types"].items()},
             "scratch_options": {h: scratch_options(rs[-1], pair_of.get(type_of.get(h))) for h, rs in nodes.items()}}
    (d / "probe.json").write_text(json.dumps(probe, indent=1) + "\n")
    print(d / "probe.json")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bootlab probe")
    sub = ap.add_subparsers(required=True)
    p = sub.add_parser("plan", help="on the login node: node types, one idle node each, the node jobs")
    p.add_argument("--sif", required=True)
    p.add_argument("--shared", help="the directory the shared file system is measured in (default: the "
                   "installation's runs, execute.install_runs: the runs go there)")
    p.add_argument("--out", required=True)
    p.add_argument("--python", help="the Python the node jobs run (default: this one)")
    p.add_argument("--py-test", help="a Python copied out of the image to try on each node type")
    p.add_argument("--submit", action="store_true")
    p.add_argument("--what", choices=["node", "pair"], default="node")
    p.add_argument("--types", help="only these node types (comma separated)")
    p.add_argument("--partition", help="only this Slurm partition's nodes; the probe jobs go to it")
    p.add_argument("--node", help="a node probe on this node (its type only); the job waits when it is busy")
    p.add_argument("--sdp", help="pair: a small SDP for the start-up test")
    p.add_argument("--sdp-precision", type=int, default=785)
    p.add_argument("--local", action="append", default=[],
                   help="a directory to try as node-local scratch besides the usual ones (repeatable)")
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("node", help="inside a job: this node's facts")
    p.add_argument("--sif", required=True)
    p.add_argument("--shared", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--py-test")
    p.add_argument("--setup", default="", help="probe plan's shell line that puts apptainer on PATH")
    p.add_argument("--local", action="append", default=[],
                   help="a directory to try as node-local scratch besides the usual ones (repeatable)")
    p.set_defaults(fn=cmd_node)
    p = sub.add_parser("pair", help="inside a job on two nodes of a type: what is between them")
    p.add_argument("--sif", required=True)
    p.add_argument("--shared", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sdp", required=True)
    p.add_argument("--sdp-precision", type=int, default=785)
    p.add_argument("--setup", default="")
    p.add_argument("--local", action="append", default=[],
                   help="a directory to try as node-local scratch besides the usual ones (repeatable)")
    p.set_defaults(fn=cmd_pair)
    p = sub.add_parser("local", help="a machine without Slurm: probe it here (no job)")
    p.add_argument("--sif", required=True)
    p.add_argument("--shared", help="the runs directory (default: the installation's, execute.install_runs)")
    p.add_argument("--out", required=True)
    p.add_argument("--local", action="append", default=[], help="a scratch directory to try (repeatable)")
    p.add_argument("--sdp", help="a small SDP: sdpb with every core, besides the bindings")
    p.add_argument("--sdp-precision", type=int, default=785)
    p.set_defaults(fn=cmd_local)
    p = sub.add_parser("io", help="pair's part on one node (internal)")
    p.add_argument("--op", choices=["write", "read"], required=True)
    p.add_argument("--dir", required=True)
    p.set_defaults(fn=cmd_io)
    p = sub.add_parser("merge", help="plan.json, node-*.json and pair-*.json -> probe.json")
    p.add_argument("dir")
    p.set_defaults(fn=cmd_merge)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
