"""layout.py: the nodes of a run (docs/en/interfaces.md section 14). No cluster needed."""
import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"   # tools/make_test_fixture.py
sys.path.insert(0, str(ROOT / "src"))

from bootlab import bench, layout  # noqa: E402

SITE = FIXTURES / "sites" / "lab"
with open(SITE / "site.toml", "rb") as f:
    SITE_TOML = tomllib.load(f)
NODE_TYPES = SITE_TOML["node_types"]
MN = layout.multinode_of(SITE_TOML)
BENCHES = {t: bench.load_bench(SITE / "bench", t) for t in ("amd2", "amd1")}
L23 = ("seven", 23)
L27 = ("seven", 27)
HOSTS = ["nodea-0", "nodea-1", "nodea-2", "nodeb-0", "nodeb-1",
         "nodeb-2", "nodeb-3", "nodeb-4", "nodec-0", "nodec-1"]


def sinfo(idle):
    return [(h, "idle" if h in idle else "allocated") for h in HOSTS]


ALL_IDLE = sinfo(HOSTS)


class Wanted(unittest.TestCase):
    """The rule reads the bench as measured: one more node while t(1) / (k t(k)) >= 0.8 and t(k-1) / t(k) >= 1.15."""

    def test_l23_amd2_one_node(self):
        # 20.7 -> 13.7 s/iter, S(2) = 1.51, efficiency 0.76
        times = layout.solver_times(BENCHES["amd2"], L23, 256)
        self.assertEqual(times, {1: 20.7, 2: 13.7})
        k, why = layout.nodes_wanted(times, 1, MN)
        self.assertEqual(k, 1)
        self.assertIn("not 2: S(2) = 1.51, efficiency 0.76", why)

    def test_l27_amd2_two_nodes(self):
        # 39.4, 39.1, 39.1 -> 23.9, S(2) = 1.64, efficiency 0.82
        times = layout.solver_times(BENCHES["amd2"], L27, 256)
        self.assertAlmostEqual(times[1], 39.2)
        k, why = layout.nodes_wanted(times, 1, MN)
        self.assertEqual(k, 2)
        self.assertIn("2: S(2) = 1.64, efficiency 0.82", why)

    def test_amd1_stops_at_two(self):
        # L23 at 128: 53.7, 31.04, 24.7 (3 nodes: efficiency 0.72); L27: 103.75, 59.03, 45.14 (0.77)
        for key in (L23, L27):
            k, _ = layout.nodes_wanted(layout.solver_times(BENCHES["amd1"], key, 128), 1, MN)
            self.assertEqual(k, 2)

    def test_unmeasured_problem_lower_bound(self):
        k, why = layout.nodes_wanted(layout.solver_times(BENCHES["amd2"], ("ising-oe", 11), 256), 1, MN)
        self.assertEqual(k, 1)
        self.assertIn("no solver measurement", why)
        self.assertEqual(layout.nodes_wanted({}, 2, MN)[0], 2)


class Choose(unittest.TestCase):
    def test_no_resources_is_one_node_as_before(self):
        p = layout.choose_nodes({}, NODE_TYPES, MN, BENCHES, L23, ALL_IDLE)
        self.assertEqual((p.nodes, p.hosts), (1, []))
        self.assertEqual(layout.sbatch_args(p), [])

    def test_auto_l23_one_amd2(self):
        p = layout.choose_nodes({"nodes": "auto"}, NODE_TYPES, MN, BENCHES, L23, ALL_IDLE)
        self.assertEqual((p.node_type, p.nodes, p.hosts), ("amd2", 1, ["nodeb-0"]))
        self.assertEqual(layout.sbatch_args(p), ["--nodes=1", "--ntasks-per-node=256", "--ntasks=256",
                                                 "-w", "nodeb-0"])

    def test_auto_l27_two_amd2(self):
        p = layout.choose_nodes({"nodes": "auto"}, NODE_TYPES, MN, BENCHES, L27,
                                sinfo(["nodeb-2", "nodeb-4", "nodea-0"]))
        self.assertEqual((p.node_type, p.hosts), ("amd2", ["nodeb-2", "nodeb-4"]))
        self.assertEqual(layout.sbatch_args(p), ["--nodes=2", "--ntasks-per-node=256", "--ntasks=512",
                                                 "-w", "nodeb-2,nodeb-4"])

    def test_auto_l27_amd2_short_takes_amd1(self):
        p = layout.choose_nodes({"nodes": "auto"}, NODE_TYPES, MN, BENCHES, L27,
                                sinfo(["nodeb-2", "nodea-0", "nodea-2"]))
        self.assertEqual((p.node_type, p.hosts), ("amd1", ["nodea-0", "nodea-2"]))

    def test_auto_none_idle_waits_for_the_first_type(self):
        p = layout.choose_nodes({"nodes": "auto"}, NODE_TYPES, MN, BENCHES, L27, sinfo([]))
        self.assertEqual((p.node_type, p.nodes, p.hosts), ("amd2", 2, []))
        self.assertEqual(layout.sbatch_args(p), ["--nodes=2", "--ntasks-per-node=256", "--ntasks=512",
                                                 "--constraint=amd2"])

    def test_auto_max_nodes_and_prefer(self):
        p = layout.choose_nodes({"nodes": "auto", "max_nodes": 1}, NODE_TYPES, MN, BENCHES, L27, ALL_IDLE)
        self.assertEqual((p.node_type, p.nodes), ("amd2", 1))
        p = layout.choose_nodes({"nodes": "auto", "prefer": ["amd1"]}, NODE_TYPES, MN, BENCHES, L23, ALL_IDLE)
        self.assertEqual((p.node_type, p.nodes), ("amd1", 2))

    def test_auto_memory_lower_bound(self):
        p = layout.choose_nodes({"nodes": "auto", "mem_gb": 600}, NODE_TYPES, MN, BENCHES, L23, ALL_IDLE)
        self.assertEqual(p.nodes, 2)          # 600 GB over 367 a node, though one node is faster per node
        with self.assertRaises(SystemExit):   # 2000 GB: 6 amd2, 9 amd1, over max_nodes
            layout.choose_nodes({"nodes": "auto", "mem_gb": 2000}, NODE_TYPES, MN, BENCHES, L23, ALL_IDLE)

    def test_named_nodes(self):
        p = layout.choose_nodes({"nodes": ["nodeb-4", "nodeb-2"]}, NODE_TYPES, MN, BENCHES, L23, [])
        self.assertEqual((p.node_type, p.hosts, p.ppn), ("amd2", ["nodeb-2", "nodeb-4"], 256))

    def test_named_nodes_refused(self):
        for nodes in (["nodeb-2", "nodea-0"],       # two types
                      ["nodeb-2", "nodeb-2"],       # twice
                      ["nodec-0", "nodec-1"]):    # -0 not a peer
            with self.assertRaises(SystemExit):
                layout.choose_nodes({"nodes": nodes}, NODE_TYPES, MN, BENCHES, L23, [])
        with self.assertRaises(SystemExit):
            layout.choose_nodes({"nodes": 2}, NODE_TYPES, MN, BENCHES, L23, [])


class Allocation(unittest.TestCase):
    def test_check(self):
        layout.check_allocation(["nodeb-2", "nodeb-4"], {"nodes": "auto"}, NODE_TYPES, MN)
        layout.check_allocation(["nodec-0"], {}, NODE_TYPES, MN)          # one node needs no peers
        layout.check_allocation(["h-1"], {}, {}, None)                            # nor a [multinode]
        for hosts, res in ((["nodeb-2", "nodea-0"], {"nodes": "auto"}),
                           (["nodeb-2", "nodeb-4"], {}),
                           (["nodeb-2", "nodeb-4"], {"nodes": ["nodeb-2", "nodeb-3"]}),
                           (["nodec-0", "nodec-1"], {"nodes": "auto"}),
                           (["nodeb-0", "nodeb-1", "nodeb-2", "nodeb-3"], {"nodes": "auto"})):
            with self.assertRaises(SystemExit):
                layout.check_allocation(hosts, res, NODE_TYPES, MN)

    def test_rank_node(self):
        self.assertEqual([layout.rank_node(r, 256) for r in (0, 255, 256, 511)], [0, 0, 1, 1])


class ResumePlan(unittest.TestCase):
    THEN = {"nodes": ["nodeb-2", "nodeb-4"], "node_type": "amd2", "ppn": 256}

    def test_own_other_wait(self):
        p = layout.resume_plan(self.THEN, NODE_TYPES, MN, ALL_IDLE)
        self.assertEqual(p.hosts, ["nodeb-2", "nodeb-4"])
        p = layout.resume_plan(self.THEN, NODE_TYPES, MN, sinfo(["nodeb-0", "nodeb-1", "nodeb-4"]))
        self.assertEqual(p.hosts, ["nodeb-0", "nodeb-1"])
        p = layout.resume_plan(self.THEN, NODE_TYPES, MN, sinfo(["nodeb-0"]))
        self.assertEqual(layout.sbatch_args(p), ["--nodes=2", "--ntasks-per-node=256", "--ntasks=512",
                                                 "--constraint=amd2"])


class AllowMultinode(unittest.TestCase):
    """A site whose probe left several nodes off by default ([multinode] allowed = false)
    and run.toml's override [resources] allow_multinode = true."""
    OFF = layout.Multinode(**{**MN.__dict__, "allowed": False, "basis": "amd2: IB rc start-up passed 2 of 3"})

    def test_auto_is_one_node_by_default(self):
        plan = layout.choose_nodes({"nodes": "auto"}, NODE_TYPES, self.OFF, BENCHES, L27, ALL_IDLE)
        self.assertEqual((plan.node_type, plan.nodes), ("amd2", 1))       # MN gives 2 (test_auto_l27_two_amd2)
        self.assertIn("default is one node (amd2: IB rc start-up passed 2 of 3)", plan.reasons[0])

    def test_override(self):
        res = {"nodes": "auto", "allow_multinode": True}
        plan = layout.choose_nodes(res, NODE_TYPES, self.OFF, BENCHES, L27, ALL_IDLE)
        self.assertEqual(plan.nodes, 2)
        self.assertIn("overrides the site's default of one node", plan.reasons[0])
        self.assertIn("docs/en/interfaces.md 14", plan.reasons[0])
        named = layout.choose_nodes({"nodes": ["nodeb-0", "nodeb-1"], "allow_multinode": True},
                                    NODE_TYPES, self.OFF, BENCHES, L27, ALL_IDLE)
        self.assertEqual(named.nodes, 2)
        layout.check_allocation(["nodeb-0", "nodeb-1"], res, NODE_TYPES, self.OFF)

    def test_refused_without_the_override(self):
        two = ["nodeb-0", "nodeb-1"]
        with self.assertRaisesRegex(SystemExit, "allow_multinode = true overrides it"):
            layout.choose_nodes({"nodes": two}, NODE_TYPES, self.OFF, BENCHES, L27, ALL_IDLE)
        with self.assertRaisesRegex(SystemExit, "default is one node"):
            layout.check_allocation(two, {"nodes": "auto"}, NODE_TYPES, self.OFF)
        then = {"nodes": two, "node_type": "amd2", "ppn": 256}
        with self.assertRaisesRegex(SystemExit, "checkpoint's 2 nodes"):
            layout.resume_plan(then, NODE_TYPES, self.OFF, ALL_IDLE)
        self.assertEqual(layout.resume_plan(then, NODE_TYPES, self.OFF, ALL_IDLE, {"allow_multinode": True}).nodes, 2)
        with self.assertRaisesRegex(SystemExit, "true or false"):
            layout.permitted({"allow_multinode": "tcp"}, self.OFF)

    def test_allowed_site_unchanged(self):
        self.assertEqual(layout.permitted({}, MN), (True, None))
        self.assertIn("anyway", layout.permitted({"allow_multinode": True}, MN)[1])
        self.assertFalse(layout.permitted({"allow_multinode": True}, None)[0])     # no [multinode]: nothing to allow


if __name__ == "__main__":
    unittest.main()
