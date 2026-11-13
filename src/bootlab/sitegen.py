"""A site from the cluster probe (docs/en/interfaces.md section 15).

    bootlab site gen --probe <probe.json> --hand <site.hand.toml> --out <sites/name> [--mask ib,local-scratch]
    bootlab site diff <generated site.toml> <hand-written site.toml> --probe <probe.json>

gen writes site.toml (every value with the probe job or the hand file it comes from) and setup.sh. The hand file
holds only what the probe cannot know: name, sif, python, runs (each may be left out: the installation's
bootlab.toml has them, execute.site_paths), bench (a directory of measurements), [slurm],
IB ports to leave out ([ib] exclude: "<host>=<port>"), and [multinode] overrides. What the probe has not measured gets the conservative value: no pair probe of a type, no multi-node
for it; a host without a measured IB port, none in the table (listed). --mask makes gen act as if the cluster
had no IB or no local disk (for testing the rules); the probe's facts are not
changed. diff lists each key: the same, or both values.
"""
import argparse
import fnmatch
import json
import statistics
import tomllib
from pathlib import Path

from .probe import MPI_ENV, scratch_options

# The image's programs (not the cluster's)
IMAGE_PROGRAMS = {"generator": "/opt/bootlab/bin/generator", "pmp2sdp": "pmp2sdp_3.0sky",
                  "dynamical_sdp": "dynamical_sdp_RV1B_3.0sky"}
DRIVER_RESERVE_MB = 4096
# layout.py: a second node only for a gain the bench shows
MULTINODE_RULES = {"max_nodes": 3, "min_efficiency": 0.8, "min_gain": 1.15}
COPY_STREAMS = 8
STARTUP_PASS = 3          # every full-scale start-up of a transport passed


def node_records(probe: dict) -> list:
    """The last node probe of each host."""
    return [rs[-1] for rs in probe["nodes"].values()]


def masked(rec: dict, mask: set) -> dict:
    """A node record as if the cluster had no IB (ports and IPoIB gone) or no local disk (candidates gone)."""
    r = json.loads(json.dumps(rec))
    if "ib" in mask:
        r["ib"] = {"start": [], "end": [], "chosen": None}
        r["nics"] = [n for n in r["nics"] if n["type"] != "infiniband"]
        if r.get("shared", {}).get("network") in ("ipoib", "rdma"):
            r["shared"]["network"] = "ethernet (masked)"
    if "local-scratch" in mask:
        r["local"] = []
        r["throughput"] = {k: v for k, v in r["throughput"].items() if k == "shared"}
    return r


def choose_scratch(records: list, pair_of: dict = None) -> tuple:
    """(site scratch value, basis, report lines): the fastest
    measured way is the default. Each probed node's options (probe.scratch_options: write until close + read in
    bootlab's file pattern, with the copy or the cold read of a pair probe of its type); a local option counts for
    every candidate path on the device measured. The site has one scratch value: the option, a local path on every
    probed node or the shared directory, with the least seconds summed over the probed nodes."""
    pair_of = pair_of or {}
    per_node, lines = [], []
    for r in records:
        opts = scratch_options(r, pair_of.get(r["host"]))
        dev = {c["path"]: (c.get("block") or {}).get("device") for c in r["local"]}
        expanded = {"shared": opts["shared"]} if "shared" in opts else {}
        for c in r["local"]:
            if not (c["writable"] and c["local_block_device"] and c.get("free_gb")):
                continue
            got = [v for p, v in opts.items() if p != "shared" and dev.get(p) is not None and dev.get(p) == dev[c["path"]]]
            if got:
                expanded[c["path"]] = min(got, key=lambda v: v["seconds"])
        per_node.append(expanded)
        lines += [f"{r['host']}: {p} spike, {s}" for p, v in sorted(opts.items()) for s in v["parts"].get("spikes", [])]
        lines.append(f"{r['host']}: " + ", ".join(f"{p} {v['seconds']} s" for p, v in sorted(expanded.items(),
                                                                                         key=lambda kv: kv[1]["seconds"]))
                     + f" ({next(iter(expanded.values()))['source'] if expanded else 'nothing measured'})")
    common = set.intersection(*(set(o) for o in per_node)) if per_node else set()
    if not common:
        return "shared", "no option measured on every probed node: shared", lines
    total = {p: round(sum(o[p]["seconds"] for o in per_node), 3) for p in common}
    sigma = {p: sum(o[p].get("sigma", 0) ** 2 for o in per_node) ** 0.5 for p in common}
    free = {p: min((c["free_gb"] for r in records for c in r["local"] if c["path"] == p), default=0) for p in common}
    # equal seconds (paths on one device): the most free space
    best = min(total, key=lambda p: (total[p], p != "shared", -free[p], p))   # the name last: the same site every time
    value = "shared" if best == "shared" else f"{best}/$USER-bootlab"
    basis = (f"scratch {value}, the fastest measured: " + "; ".join(f"{p} {t} s" for p, t in sorted(total.items(),
             key=lambda kv: (kv[1], kv[0]))) + " (write until close + read of 2 GiB in bootlab's file pattern, summed over "
             + ", ".join(f"{r['host']} job {r['job']}" for r in records) + ")")
    # the fastest is chosen; within 10 % or 2 sigma of the best of the other kind (local
    # against shared) the basis says so (seen: 1.299 s local against 1.331 s shared on one node)
    rival = min((p for p in common if (p == "shared") != (best == "shared")), key=lambda p: (total[p], p), default=None)
    if rival is not None:
        gap = total[rival] - total[best]
        if gap < 0.10 * total[best] or gap < 2 * (sigma[best] ** 2 + sigma[rival] ** 2) ** 0.5:
            note = (f"close: {rival} {total[rival]} s is within 10 % or 2 sigma of {best} {total[best]} s, "
                    f"one probe can turn the choice")
            basis += "; " + note
            lines.append(note)
    return value, basis, lines


def ib_ports_seen(records: list, pairs: list) -> dict:
    """host -> the IB port the probe chose (node or pair probe): a check of rank_env.sh's ib:auto, which applies
    the same rule at every start (no port tables)."""
    ports = {r["host"]: r["ib"]["chosen"] for r in records if r["ib"]["chosen"]}
    for p in pairs:
        ports.update({h: v for h, v in p.get("ib_ports", {}).items() if v})
    return ports


def multinode(pairs: list, mask: set, local_types: set = None) -> tuple:
    """(allowed, refused): type -> hosts that may solve together, and type -> (hosts, why not) for the pair-probed
    types that are one node by default (several nodes only when the pair probe's rc start-ups all passed and the
    type has a writable node-local disk; TCP or no local disk: one node unless run.toml says
    allow_multinode). local_types: the types with a local disk (None: all)."""
    ok, no = {}, {}
    for p in pairs:
        t = p["type"] if "type" in p else p["hosts"][0]
        runs = [s for s in p.get("startup", []) if s.get("name") == "rc" and not s.get("skipped")]
        passed = sum(1 for s in runs if s["exit"] == 0)
        if "ib" in mask:
            why = f"no IB (masked): MPI over TCP (pair probe {p.get('job')})"
        elif len(runs) < STARTUP_PASS or passed < len(runs):
            why = f"IB rc start-up passed {passed} of {len(runs)}, {STARTUP_PASS} needed (pair probe {p.get('job')})"
        elif local_types is not None and t not in local_types:
            why = f"no writable node-local disk (pair probe {p.get('job')})"
        else:
            ok.setdefault(t, set()).update(p["hosts"])
            continue
        hosts, whys = no.get(t, (set(), []))
        no[t] = (hosts | set(p["hosts"]), whys + [why])
    return ok, {t: (h, "; ".join(w)) for t, (h, w) in no.items() if t not in ok}


def copy_network(pairs: list, mask: set):
    """The IPoIB network the pairs share (the copies' network), else Ethernet; None without a pair."""
    nets = [n for p in pairs for n in p.get("networks", []) if not ("ib" in mask and n["kind"] == "ipoib")]
    pick = next((n for n in nets if n["kind"] == "ipoib"), None) or next(iter(nets), None)
    return pick["prefix"] if pick else None


def toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, list):
        return "[" + ", ".join(toml_value(x) for x in v) + "]"
    return json.dumps(v)


def write_toml(d: dict, notes: dict) -> str:
    out = []
    flat = {k: v for k, v in d.items() if not isinstance(v, dict)}
    for k, v in flat.items():
        out.append(f"{k} = {toml_value(v)}" + (f"   # {notes[k]}" if k in notes else ""))
    for k, v in d.items():
        if isinstance(v, dict):
            subs = {kk: vv for kk, vv in v.items() if isinstance(vv, dict)}
            if len(subs) < len(v):
                out.append(f"\n[{k}]" + (f"   # {notes[k]}" if k in notes else ""))
                out += [f"{kk} = {toml_value(vv)}" for kk, vv in v.items() if not isinstance(vv, dict)]
            for kk, vv in subs.items():
                out.append(f"\n[{k}.{kk}]")
                out += [f"{a} = {toml_value(b)}" for a, b in vv.items()]
    return "\n".join(out) + "\n"


def generate(probe: dict, hand: dict, mask: set) -> tuple:
    """(site dict, notes, report lines)."""
    records = [masked(r, mask) for r in node_records(probe)]
    pairs = probe.get("pairs", [])
    types = probe["types"]
    for p in pairs:
        p["type"] = next((t for t, v in types.items() if p["hosts"][0] in v["hosts"]), None)
    jobs = ", ".join(sorted({str(r["job"]) for r in records}))
    report = [f"node probes: {jobs}; pair probes: {', '.join(str(p['job']) for p in pairs) or 'none'}"
              + (f"; masked: {', '.join(sorted(mask))}" if mask else "")]
    local_types = {t for t, v in types.items() for r in records if r["host"] in v["hosts"]
                   and any(c.get("writable") and c.get("local_block_device") for c in r["local"])}
    allowed, refused = multinode(pairs, mask, local_types)
    # the copy and the cold read count only where a type solves on several nodes by default (one node by
    # default copies nothing; IB masked: local scratch, one node)
    pair_of = {h: p for p in pairs for t in [p.get("type")] if t in allowed for h in types.get(t, {}).get("hosts", [])}
    report += [f"pair probe {p.get('job')} ({p.get('type')}) not in the scratch choice: one node by default"
               for p in pairs if p.get("type") not in allowed]
    scratch, basis, lines = choose_scratch(records, pair_of)
    measured = sorted({(r["shared"].get("where") or {}).get("realpath") or r["shared"]["path"] for r in records
                       if r["shared"].get("path")})
    pw = probe.get("plan", {}).get("shared_where") or {}
    if measured:              # where the shared numbers hold (one mount can hide two disk arrays)
        basis += (f"; shared measured in {', '.join(measured)}"
                  + (f" ({pw.get('fstype')} {pw.get('source')}" + (f", mergerfs branch {pw['mergerfs_branch']}"
                     if pw.get("mergerfs_branch") else "") + ")" if pw else ""))
    report += lines + [basis]
    shared_mp = records[0]["shared"].get("mountpoint")
    binds = [shared_mp] + ([scratch.split("/$USER")[0]] if scratch != "shared" else [])
    seen = {} if "ib" in mask else ib_ports_seen(records, pairs)
    ib = bool(seen)
    prefix = copy_network(pairs, mask)
    eth = next((n for n in records[0]["nics"] if n["type"] == "ethernet" and n["ipv4"]
                and not n["ipv4"][0].startswith("169.254.")), None)
    paths = {k: hand[k] for k in ("sif", "python", "runs") if k in hand}   # left out: from bootlab.toml
    site = {**paths, "binds": binds, "setup": "setup.sh",
            **IMAGE_PROGRAMS, "scratch": scratch, "scratch_basis": basis, "driver_reserve_mb": DRIVER_RESERVE_MB,
            "env_unset": ["OMPI_MCA_plm", "UCX_NET_DEVICES"],
            "slurm": hand.get("slurm", {}),
            "env": {**MPI_ENV, "UCX_TLS": "rc,self,sm" if ib else "tcp,self,sm"},
            "mpi": {"transport": "ib" if ib else f"tcp-{(eth or {}).get('speed_mbps', 0) // 1000}g",
                    "net_devices": ["ib:auto"] if ib else ([f"subnet:{'.'.join(eth['ipv4'][0].split('.')[:3])}."] if eth else [])},
            **({"mpi_ib_exclude": hand.get("ib", {}).get("exclude", [])} if ib else {}),
            "node_types": {t: {"hosts": v["hosts"], "physical_cores": v["physical_cores"], "mem_gb": v["mem_mb"] // 1024}
                           for t, v in types.items()}}
    if "mpi_ib_exclude" in site:
        site["mpi"]["ib_exclude"] = site.pop("mpi_ib_exclude")
    routes = {(r["shared"].get("server_addr"), r["shared"].get("network")) for r in records}
    if "ib" in mask:             # the masked network is not the one the nodes use: no check at the start
        pass
    elif len(routes) == 1 and None not in next(iter(routes)):   # the start's check (execute.runs_network)
        server, network = next(iter(routes))
        site["shared_network"] = {"server": server, "network": network, "basis": f"node probes {jobs}"}
    elif any(r["shared"].get("server_addr") for r in records):
        report.append("the shared file system's route differs between the probed nodes: "
                      + ", ".join(f"{r['host']} {r['shared'].get('server_addr')} {r['shared'].get('network')}"
                                  for r in records) + "; no shared_network, no check at the start")
    if measured:
        site["shared_measured_at"] = measured
    if "bench" in hand:          # measurements are not the probe's: a hand-named bench directory
        site["bench"] = hand["bench"]
    if allowed or refused:
        use = allowed or {t: h for t, (h, _) in refused.items()}
        mn = {"prefer": sorted(use, key=lambda t: -types[t]["physical_cores"]),
              "peers": sorted(h for hs in use.values() for h in hs), **MULTINODE_RULES,
              "copy_streams": COPY_STREAMS, "transfer": "srun"}
        if not allowed:          # one node unless run.toml [resources] allow_multinode = true
            mn.update({"allowed": False, "basis": "; ".join(f"{t}: {w}" for t, (_, w) in sorted(refused.items()))})
        tail = "left out: another type is allowed" if allowed else "run.toml allow_multinode = true overrides it"
        for t, (_, w) in sorted(refused.items()):
            report.append(f"{t}: one node by default: {w}; {tail}")
        if prefix:
            mn["copy_network"] = prefix
        mn.update(hand.get("multinode", {}))
        site["multinode"] = mn
    notes = {"scratch": "the fastest measured (scratch_basis)", "binds": "the shared directory's mount point and the local scratch's base",
             "env_unset": "rank_env.sh sets UCX_NET_DEVICES per node",
             "mpi": "rank_env.sh picks each node's IB port at every start; ib_exclude from the hand file",
             "sif": "hand", "python": "hand", "runs": "hand", "slurm": "hand",
             "shared_measured_at": "the start warns when the runs directory is not in one of them",
             "shared_network": "the runs file system's server and network on every probed node; the start warns when "
                               "a node's differs"}
    if "multinode" not in site:
        report.append("no pair probe: one node only")
    if seen:
        report.append("IB ports the probe chose (ib:auto applies the same rule): "
                      + ", ".join(f"{h}={p}" for h, p in sorted(seen.items())))
    return site, notes, report


def cmd_gen(args):
    probe = json.loads(Path(args.probe).read_text())
    hand = tomllib.loads(Path(args.hand).read_text())
    mask = set(filter(None, (args.mask or "").split(",")))
    site, notes, report = generate(probe, hand, mask)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    head = (f"# Generated by bootlab site gen from {args.probe} and {args.hand}"
            + (f", masked {sorted(mask)}" if mask else "") + ".\n# " + "\n# ".join(report) + "\n")
    (out / "site.toml").write_text(head + write_toml(site, notes))
    setup = probe["plan"].get("apptainer_setup", "")
    (out / "setup.sh").write_text("# apptainer on PATH (from the shell probe plan ran in)\n" + setup.replace("; ", "\n") + "\n")
    print("\n".join(report))
    print(out / "site.toml")


def expand_hosts(patterns: list, all_hosts: list) -> list:
    return sorted({h for p in patterns for h in all_hosts if fnmatch.fnmatch(h, p)} | {p for p in patterns if "*" not in p and "[" not in p})


def cmd_diff(args):
    gen = tomllib.loads(Path(args.generated).read_text())
    hand = tomllib.loads(Path(args.hand).read_text())
    probe = json.loads(Path(args.probe).read_text())
    hosts = [n["host"] for n in probe["plan"]["nodes"]]

    def flat(d, pre=""):
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out.update(flat(v, f"{pre}{k}."))
            else:
                out[f"{pre}{k}"] = v
        return out
    g, h = flat(gen), flat(hand)
    for k in sorted(set(g) | set(h)):
        a, b = g.get(k), h.get(k)
        if k.endswith(".hosts") and isinstance(a, list) and isinstance(b, list):
            a, b = expand_hosts(a, hosts), expand_hosts(b, hosts)
        if k in ("mpi.net_devices", "multinode.peers", "env_unset") and isinstance(a, list) and isinstance(b, list):
            a, b = sorted(a), sorted(b)
        print(f"{'same' if a == b else 'DIFF'}  {k}: " + (toml_value(a) if a == b else f"generated {toml_value(a)} | hand {toml_value(b)}"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bootlab site")
    sub = ap.add_subparsers(required=True)
    p = sub.add_parser("gen")
    p.add_argument("--probe", required=True)
    p.add_argument("--hand", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mask", help="ib, local-scratch (comma separated): as if the cluster had none")
    p.set_defaults(fn=cmd_gen)
    p = sub.add_parser("diff")
    p.add_argument("generated")
    p.add_argument("hand")
    p.add_argument("--probe", required=True)
    p.set_defaults(fn=cmd_diff)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
