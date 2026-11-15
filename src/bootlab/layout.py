"""Which nodes a run's solver uses (docs/en/interfaces.md section 14). Pure functions: the
inputs are run.toml's [resources], the site's [node_types] and [multinode], the bench files and a snapshot of
sinfo; `bootlab submit` turns the result into an sbatch command and the driver checks its allocation with
check_allocation before it starts.

One solve uses nodes of one type only: SDPB needs the same processes on every node and splits the blocks as
if every node were as fast (Block_Info::allocate_blocks, compute_block_grid_mapping; plan 2.1).
"""
import fnmatch
import math
from dataclasses import dataclass, field


def node_type_of(host: str, node_types: dict) -> str:
    """The site's node type whose hosts patterns (fnmatch) match the host; "default" when none does (a site that
    lists none, or a node it does not list: no bench data, all its physical cores)."""
    for t, v in node_types.items():
        if any(fnmatch.fnmatchcase(host, p) for p in v.get("hosts", [])):
            return t
    return "default"


@dataclass
class Multinode:
    prefer: list            # node types, the first that has enough idle nodes wins
    peers: list             # nodes that may solve together (the firewall is open between them)
    max_nodes: int
    min_efficiency: float   # one more node only if t(1) / (k t(k)) stays at least this
    min_gain: float         # ... and t(k) / t(k + 1) at least this
    copy_streams: int = 1   # copy streams to each node when the SDPs are placed
    transfer: str = "srun"  # how the driver reaches the other nodes: srun (light steps) or ssh
    copy_network: str = None   # srun: the IPv4 prefix of the network the copies use (else the host names)
    ipoib_prefix: str = None   # ssh: the IPoIB network's address prefix (the copies go over it)
    ssh: list = field(default_factory=list)   # ssh: the command for the copies, before the address
    remote_setup: str = ""     # ssh: shell lines that make apptainer available in a remote shell
    allowed: bool = True       # false: one node unless run.toml [resources] allow_multinode = true
    basis: str = None          # why not allowed (site gen: the pair probe's job)


def multinode_of(site_toml: dict) -> Multinode:
    return Multinode(**site_toml["multinode"])


RISK = ("one solve on several nodes over TCP: MPI's all-to-all connections failed at start-up with a few hundred "
        "ranks a node on the cluster bootlab was tested on (docs/en/interfaces.md 14)")


def permitted(resources: dict, mn) -> tuple:
    """(several nodes allowed, the reason for driver.log or the error). The site's [multinode] allowed = false
    (the probe's default) is overridden by run.toml [resources] allow_multinode = true."""
    flag = resources.get("allow_multinode", False)
    if not isinstance(flag, bool):
        raise SystemExit(f"[resources] allow_multinode = {flag!r}: true or false")
    if mn is None:
        return False, "the site has no [multinode] (no pair probe passed): one node" +             ("; allow_multinode cannot add one (no peers measured)" if flag else "")
    if mn.allowed:
        return True, "allow_multinode = true: the site allows several nodes anyway" if flag else None
    if flag:
        return True, (f"run.toml [resources] allow_multinode = true overrides the site's default of one node "
                      f"({mn.basis}); risk: {RISK}")
    return False, (f"the site's default is one node ({mn.basis}); run.toml [resources] allow_multinode = true "
                   f"overrides it")


def rank_node(rank: int, ppn: int) -> int:
    """The node (its place in the job's node list) of an MPI rank: srun's default block distribution."""
    return rank // ppn


def problem_key(problem) -> tuple:
    """What a bench solver row must name to stand for this run: the model's name (Problem.model_name: a built-in
    model's card its bound) and Lambda. A row without them (older rows) stands for no run."""
    return problem.model_name, problem.numerics["lambda"]


def solver_times(bench: dict, key: tuple, ppn: int, transport: str = "ib") -> dict:
    """nodes -> mean s/iter of the bench's sdpb rows for this problem at ppn processes a node. sdpb rows only:
    dynamical_sdp is the same solver, and its rows of a few iterations are read in whole seconds. Rows of more
    than one node count only with the site's MPI transport (their "transport", ib when left out: older rows
    are all IB); one node's rows go over shared memory whatever it is."""
    by = {}
    for r in bench.get("solver", []):
        if (r.get("model"), r.get("lambda")) == key and r["procs_per_node"] == ppn \
                and r["program"].startswith("sdpb") and (r["nodes"] == 1 or r.get("transport", "ib") == transport):
            by.setdefault(r["nodes"], []).append(r["s_per_iter"])
    return {k: sum(v) / len(v) for k, v in by.items()}


def solver_memory_gb(bench: dict, key: tuple):
    """The largest max_memused_gb of the problem's one-node rows, None when none is measured."""
    m = [r["max_memused_gb"] for r in bench.get("solver", [])
         if (r.get("model"), r.get("lambda")) == key and r["nodes"] == 1 and "max_memused_gb" in r]
    return max(m) if m else None


def nodes_wanted(times: dict, k_min: int, mn: Multinode) -> tuple:
    """(k, reason): from k_min up, one more node while the bench measured it and it pays: efficiency
    t(1) / (k t(k)) >= min_efficiency and gain t(k-1) / t(k) >= min_gain. No measurement on one node: k_min."""
    if 1 not in times:
        return k_min, f"no solver measurement of this problem on one node: {k_min} (the lower bound)"
    k, why = k_min, []
    while k + 1 <= mn.max_nodes and k + 1 in times and k in times:
        eff = times[1] / ((k + 1) * times[k + 1])
        gain = times[k] / times[k + 1]
        if eff < mn.min_efficiency or gain < mn.min_gain:
            why.append(f"not {k + 1}: S({k + 1}) = {times[1] / times[k + 1]:.2f}, efficiency {eff:.2f} "
                       f"(at least {mn.min_efficiency}), gain {gain:.2f} (at least {mn.min_gain})")
            break
        why.append(f"{k + 1}: S({k + 1}) = {times[1] / times[k + 1]:.2f}, efficiency {eff:.2f}, gain {gain:.2f}")
        k += 1
    measured = ", ".join(f"{n}: {t:.2f}" for n, t in sorted(times.items()))
    if not why:
        why.append(f"nothing measured above {k}")
    return k, f"s/iter by nodes {{{measured}}}; " + "; ".join(why)


@dataclass
class Plan:
    node_type: str
    nodes: int
    hosts: list             # the nodes to ask for; empty: wait for any `nodes` of node_type in peers
    exclude: list           # with hosts empty: the nodes of node_type that are not peers
    ppn: int                # physical cores a node: the job's tasks a node
    reasons: list


def choose_nodes(resources: dict, node_types: dict, mn: Multinode, benches: dict, key: tuple,
                 sinfo: list, transport: str = "ib") -> Plan:
    """resources: run.toml [resources] (empty: one node, where Slurm puts it, as before). node_types: name ->
    {physical_cores, mem_gb, hosts}. benches: node type -> its bench. sinfo: (host, state) of every node.
    transport: the site's MPI transport, which the bench rows of more than one node must have.
    Raises SystemExit when the request cannot be met as written."""
    wanted = resources.get("nodes", 1)
    if isinstance(wanted, int):
        if wanted != 1:
            raise SystemExit("[resources] nodes is a list of nodes, \"auto\", or left out (one node)")
        return Plan(None, 1, [], [], None, ["no [resources]: one node, where Slurm puts it"])
    if isinstance(wanted, list):
        types = sorted({node_type_of(h, node_types) for h in wanted})
        if len(types) != 1:
            raise SystemExit(f"[resources] nodes {wanted}: one solve uses one node type, these are {types}")
        if len(wanted) != len(set(wanted)):
            raise SystemExit(f"[resources] nodes {wanted}: a node twice")
        t = types[0]
        if t not in node_types:
            raise SystemExit(f"[resources] nodes: node type {t} is not in the site's node_types")
        if len(wanted) > 1:
            ok, why = permitted(resources, mn)
            if not ok:
                raise SystemExit(f"[resources] nodes {sorted(wanted)}: {why}")
            outside = [h for h in wanted if h not in mn.peers]
            if outside:
                raise SystemExit(f"[resources] nodes: {outside} not in the site's multinode peers")
        why = permitted(resources, mn)[1] if len(wanted) > 1 else None
        return Plan(t, len(wanted), sorted(wanted), [], node_types[t]["physical_cores"],
                    [f"the nodes named in run.toml: {', '.join(sorted(wanted))}"] + ([why] if why else []))
    if wanted != "auto":
        raise SystemExit(f"[resources] nodes = {wanted!r}: a list of nodes or \"auto\"")

    states = dict(sinfo)
    ok, why = permitted(resources, mn)
    reasons = [why] if why else []
    if not ok:
        resources = {**resources, "max_nodes": 1}
    fits = []
    for t in resources.get("prefer", mn.prefer):
        nt = node_types[t]
        bench = benches.get(t, {})
        mem = resources.get("mem_gb") or solver_memory_gb(bench, key)
        k_mem = math.ceil(mem / nt["mem_gb"]) if mem else 1
        k_core = math.ceil(resources.get("min_cores", 1) / nt["physical_cores"])
        k_min = max(k_mem, k_core)
        k, why = nodes_wanted(solver_times(bench, key, nt["physical_cores"], transport), k_min,
                              Multinode(**{**mn.__dict__, "max_nodes": resources.get("max_nodes", mn.max_nodes)}))
        pool = sorted(h for h in states if node_type_of(h, node_types) == t and (h in mn.peers or k == 1))
        idle = [h for h in pool if states[h] == "idle"]
        reasons.append(f"{t}: memory {f'{mem} GB' if mem else 'not measured'} -> {k_mem}, cores -> {k_core}; {why}; "
                       f"{len(idle)} of {len(pool)} idle")
        if k > resources.get("max_nodes", mn.max_nodes):
            continue
        if len(idle) >= k:
            return Plan(t, k, idle[:k], [], nt["physical_cores"], reasons + [f"{t}: {', '.join(idle[:k])}"])
        if len(pool) >= k:
            fits.append((t, k, pool))
    if not fits:
        raise SystemExit("no node type has enough nodes:\n  " + "\n  ".join(reasons))
    t, k, pool = fits[0]
    others = sorted(h for h in states if node_type_of(h, node_types) == t and h not in pool)
    return Plan(t, k, [], others, node_types[t]["physical_cores"],
                reasons + [f"none has enough idle nodes: wait for {k} of {t}"])


def resume_plan(then: dict, node_types: dict, mn: Multinode, sinfo: list, resources: dict = None) -> Plan:
    """The nodes for a resume of a checkpoint of layout then (nodes, node_type, ppn): the same layout, so the
    binary checkpoint loads (cli.place_checkpoint). Its own nodes when all are idle; else as many idle nodes of
    its type (peers when more than one); else wait."""
    t, k = then["node_type"], len(then["nodes"])
    if k > 1:
        ok, why = permitted(resources or {}, mn)
        if not ok:
            raise SystemExit(f"resume on the checkpoint's {k} nodes: {why}")
    states = dict(sinfo)
    ppn = node_types[t]["physical_cores"]
    if all(states.get(h) == "idle" for h in then["nodes"]):
        return Plan(t, k, sorted(then["nodes"]), [], ppn, [f"the checkpoint's own nodes, idle"])
    pool = sorted(h for h in states if node_type_of(h, node_types) == t and (h in mn.peers or k == 1))
    idle = [h for h in pool if states[h] == "idle"]
    if len(idle) >= k:
        return Plan(t, k, idle[:k], [], ppn, [f"the checkpoint's nodes are busy: {k} other idle {t}"])
    others = sorted(h for h in states if node_type_of(h, node_types) == t and h not in pool)
    return Plan(t, k, [], others, ppn, [f"wait for {k} of {t}"])


def sbatch_args(plan: Plan) -> list:
    """The sbatch options that give the plan's nodes (run.sbatch keeps --exclusive --hint=nomultithread
    --mem=0; its --nodes=1 --ntasks=128 are only defaults)."""
    if plan.nodes == 1 and not plan.hosts:
        return []
    args = [f"--nodes={plan.nodes}", f"--ntasks-per-node={plan.ppn}", f"--ntasks={plan.nodes * plan.ppn}"]
    if plan.hosts:
        return args + ["-w", ",".join(plan.hosts)]
    args += [f"--constraint={plan.node_type}"]
    return args + ([f"--exclude={','.join(plan.exclude)}"] if plan.exclude else [])


def check_allocation(hosts: list, resources: dict, node_types: dict, mn: Multinode):
    """The driver's check of what Slurm gave before it computes anything (fail fast): one node type that the
    site knows; with more than one node, all peers and at most max_nodes; the nodes run.toml names, if it
    names them. mn None: the site has no [multinode], one node only."""
    types = sorted({node_type_of(h, node_types) for h in hosts})
    if len(types) != 1:
        raise SystemExit(f"allocation {hosts}: more than one node type {types}; one solve uses one type")
    if len(hosts) > 1 and mn is None:
        raise SystemExit(f"allocation {hosts}: the site has no [multinode]")
    if len(hosts) > 1 and types[0] not in node_types:
        raise SystemExit(f"allocation {hosts}: node type {types[0]} is not in the site's node_types")
    if len(hosts) > 1:
        ok, why = permitted(resources, mn)
        if not ok:
            raise SystemExit(f"allocation {hosts}: {why}")
        outside = [h for h in hosts if h not in mn.peers]
        if outside:
            raise SystemExit(f"allocation {hosts}: {outside} not in the site's multinode peers")
        if len(hosts) > resources.get("max_nodes", mn.max_nodes):
            raise SystemExit(f"allocation {hosts}: more than max_nodes {resources.get('max_nodes', mn.max_nodes)}")
    named = resources.get("nodes", 1)
    if isinstance(named, list) and sorted(named) != sorted(hosts):
        raise SystemExit(f"allocation {hosts}: run.toml [resources] names {sorted(named)}")
    if named == 1 and len(hosts) != 1:
        raise SystemExit(f"allocation {hosts}: run.toml has no [resources], which means one node")
