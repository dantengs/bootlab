"""bootlab.sitegen's decisions on made-up probe records: python -m unittest tests.test_sitegen."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bootlab import sitegen  # noqa: E402


def tp(write, read, sigma=0.01):
    return {"bootlab": {"write": {"seconds": write, "sigma_s": sigma}, "read": {"seconds": read, "sigma_s": sigma}}}


def node(host, local_w, shared_w, port="mlx5_0:1", local=True):
    cands = [{"path": "/scratch", "writable": True, "local_block_device": True, "free_gb": 400.0,
              "block": {"device": "nvme0n1"}},
             {"path": "/tmp", "writable": True, "local_block_device": True, "free_gb": 60.0,
              "block": {"device": "nvme0n1"}}] if local else []
    t = {"shared": tp(shared_w, [0.5] * 3)}
    if local:
        t["/scratch"] = tp(local_w, [0.5] * 3)
    return {"host": host, "job": "1", "local": cands, "throughput": t, "ib": {"chosen": port},
            "nics": [], "shared": {"mountpoint": "/data"}}


class Decisions(unittest.TestCase):
    def test_fastest_wins(self):
        recs = [node("a", [0.1] * 3, [5.0] * 3), node("b", [0.1] * 3, [5.0] * 3)]
        value, basis, lines = sitegen.choose_scratch(recs)
        self.assertEqual(value, "/scratch/$USER-bootlab")        # 0.6 s against 5.5 s, the most free space's path
        self.assertIn("/scratch 1.2 s; /tmp 1.2 s", basis)        # summed over the two nodes; ties by name
        self.assertIn("shared 11.0 s", basis)
        recs = [node("a", [1.0] * 3, [0.9] * 3)]
        self.assertEqual(sitegen.choose_scratch(recs)[0], "shared")   # no preference either way: the faster
        recs = [node("a", [0.1] * 3, [5.0] * 3), node("b", None, [5.0] * 3, local=False)]
        self.assertEqual(sitegen.choose_scratch(recs)[0], "shared")   # no local path on every node

    def test_a_close_choice_is_noted(self):
        """The fastest is chosen; within 10 % or 2 sigma of the other kind it says so."""
        recs = [node("a", [0.5] * 3, [0.55] * 3)]                 # local 1.0 s, shared 1.05 s: 5 %
        value, basis, lines = sitegen.choose_scratch(recs)
        self.assertEqual(value, "/scratch/$USER-bootlab")
        self.assertIn("close: shared 1.05 s is within 10 % or 2 sigma of /scratch 1.0 s, one probe can turn the choice",
                      basis)
        self.assertTrue(any(l.startswith("close: ") for l in lines))
        recs = [node("a", [0.5] * 3, [0.65] * 3)]                 # 15 %: 1.0 against 1.15, sigma 0.014: not close
        self.assertNotIn("; close: ", sitegen.choose_scratch(recs)[1])
        self.assertNotIn("; close: ", sitegen.choose_scratch([node("a", [0.1] * 3, [5.0] * 3)])[1])

    def test_a_spike_is_kept_and_does_not_make_it_close(self):
        """Repeats of 8.16, 1.00, 1.06 s: sigma from the MAD, the spike in the parts and the
        report; the choice is the medians' as before."""
        rec = node("a", [0.5] * 3, [8.16, 1.0, 1.06])
        o = sitegen.scratch_options(rec)
        self.assertEqual(o["shared"]["seconds"], 1.56)                       # median 1.06 + read 0.5
        self.assertEqual(o["shared"]["parts"]["spikes"], ["write repeat 1: 8.16 s, 7.7 x the median"])
        self.assertAlmostEqual(o["shared"]["sigma"], 1.4826 * 0.06, places=6)  # MAD of the writes; reads all equal
        self.assertNotIn("spikes", o["/scratch"]["parts"])
        value, basis, lines = sitegen.choose_scratch([rec])
        self.assertEqual(value, "/scratch/$USER-bootlab")                    # 1.0 s against 1.56 s
        self.assertNotIn("; close: ", basis)                                 # 56 %, and 2 MAD-sigma is 0.18 s
        self.assertIn("a: shared spike, write repeat 1: 8.16 s, 7.7 x the median", lines)

    def test_the_measured_directory_is_written(self):
        recs = [node("a-1", [0.1] * 3, [5.0] * 3), node("a-2", [0.1] * 3, [5.0] * 3)]
        for r in recs:
            r["shared"].update({"path": "/data/inst/runs", "where": {"realpath": "/data/inst/runs"}})
        p = self.probe(recs)
        p["plan"]["shared_where"] = {"fstype": "nfs4", "source": "nfs-server:/data", "mergerfs_branch": None}
        site, _, _ = sitegen.generate(p, {}, set())
        self.assertEqual(site["shared_measured_at"], ["/data/inst/runs"])
        self.assertIn("; shared measured in /data/inst/runs (nfs4 nfs-server:/data)", site["scratch_basis"])

    def test_pair_adds_the_copy_and_reads_cold(self):
        rec = node("a-1", [0.1] * 3, [5.0] * 3)
        pair = {"job": "9", "copies": [{"variant": "srun-ipoib", "seconds": [10.0, 11.0, 12.0]},
                                       {"variant": "srun-ethernet", "seconds": [20.0] * 3}],
                "shared": {"cold_read_other_node": {"read_s": 0.3}}}
        opts = sitegen.scratch_options(rec, pair)
        self.assertEqual(opts["/scratch"]["seconds"], 11.6)       # 0.1 + 0.5 + the fastest copy's 11
        self.assertEqual(opts["shared"]["seconds"], 5.3)          # 5.0 + the cold read
        self.assertEqual(sitegen.choose_scratch([rec], {"a-1": pair})[0], "shared")

    def test_ib_auto_with_the_hand_excludes(self):
        recs = [node("a-1", [0.1] * 3, [5.0] * 3), node("a-2", [0.1] * 3, [5.0] * 3, port="mlx5_1:1")]
        probe = {"nodes": {r["host"]: [r] for r in recs}, "pairs": [], "plan": {"apptainer_setup": ""},
                 "types": {"a": {"hosts": ["a-1", "a-2", "a-3"], "physical_cores": 8, "mem_mb": 16000}}}
        hand = {"sif": "s", "python": "p", "runs": "r", "ib": {"exclude": ["a-2=mlx5_0:1"]}}
        site, _, report = sitegen.generate(probe, hand, set())
        self.assertEqual(site["mpi"], {"transport": "ib", "net_devices": ["ib:auto"], "ib_exclude": ["a-2=mlx5_0:1"]})
        self.assertIn("a-1=mlx5_0:1, a-2=mlx5_1:1", report[-1])        # the probe's choices, to check the rule
        site, _, _ = sitegen.generate(probe, hand, {"ib"})
        self.assertNotIn("ib_exclude", site["mpi"])
        self.assertEqual(site["env"]["UCX_TLS"], "tcp,self,sm")

    def test_multinode_needs_three_rc_startups(self):
        ok = {"hosts": ["a-1", "a-2"], "type": "a", "job": "9",
              "startup": [{"name": "rc", "exit": 0}] * 3 + [{"name": "tcp-ipoib", "exit": 16}] * 3}
        self.assertEqual(sitegen.multinode([ok], set()), ({"a": {"a-1", "a-2"}}, {}))
        bad = {**ok, "startup": [{"name": "rc", "exit": 0}] * 2 + [{"name": "rc", "exit": 16}]}
        self.assertEqual(sitegen.multinode([bad], set()),
                         ({}, {"a": ({"a-1", "a-2"}, "IB rc start-up passed 2 of 3, 3 needed (pair probe 9)")}))
        short = {**ok, "startup": [{"name": "rc", "exit": 0}] * 2 + [{"name": "rc", "skipped": "time"}]}
        self.assertEqual(sitegen.multinode([short], set())[0], {})                   # skipped is not passed
        self.assertIn("no IB (masked)", sitegen.multinode([ok], {"ib"})[1]["a"][1])
        self.assertIn("no writable node-local disk", sitegen.multinode([ok], set(), set())[1]["a"][1])

    def probe(self, recs, pairs=()):
        return {"nodes": {r["host"]: [r] for r in recs}, "pairs": list(pairs), "plan": {"apptainer_setup": ""},
                "types": {"a": {"hosts": ["a-1", "a-2"], "physical_cores": 8, "mem_mb": 16000}}}

    def test_refused_type_written_with_its_basis(self):
        recs = [node("a-1", [0.1] * 3, [5.0] * 3), node("a-2", [0.1] * 3, [5.0] * 3)]
        bad = {"hosts": ["a-1", "a-2"], "job": "9", "startup": [{"name": "rc", "exit": 16}] * 3, "networks": []}
        site, _, report = sitegen.generate(self.probe(recs, [bad]), {}, set())
        mn = site["multinode"]
        self.assertFalse(mn["allowed"])
        self.assertEqual(mn["basis"], "a: IB rc start-up passed 0 of 3, 3 needed (pair probe 9)")
        self.assertEqual(mn["peers"], ["a-1", "a-2"])
        self.assertTrue(any("allow_multinode = true overrides it" in line for line in report))
        good = {**bad, "startup": [{"name": "rc", "exit": 0}] * 3}
        self.assertNotIn("allowed", sitegen.generate(self.probe(recs, [good]), {}, set())[0]["multinode"])
        site, _, _ = sitegen.generate(self.probe(recs, [good]), {}, {"local-scratch"})
        self.assertIn("no writable node-local disk", site["multinode"]["basis"])   # no local disk: one node

    def test_masks_follow_the_decision_table(self):
        """The decision table on two nodes with IB, local disks and a good pair probe: no mask,
        IB masked, local disk masked, both."""
        recs = [node("a-1", [0.1] * 3, [5.0] * 3), node("a-2", [0.1] * 3, [5.0] * 3)]
        for r in recs:
            r["shared"].update({"server_addr": "nfs-server", "network": "ipoib"})
            r["nics"] = [{"name": "eth0", "type": "ethernet", "ipv4": ["10.20.0.7/24"], "speed_mbps": 1000},
                         {"name": "ib0", "type": "infiniband", "ipv4": ["10.0.0.7/24"], "speed_mbps": 100000}]
        pair = {"hosts": ["a-1", "a-2"], "job": "9", "startup": [{"name": "rc", "exit": 0}] * 3,
                "networks": [{"kind": "ipoib", "prefix": "10.0.0."}, {"kind": "ethernet", "prefix": "10.20.0."}],
                "copies": [{"variant": "srun-ipoib", "seconds": [1.5] * 3}, {"variant": "srun-ethernet", "seconds": [19.0] * 3}],
                "shared": {"cold_read_other_node": {"read_s": 1.0}}}

        def gen(mask):
            return sitegen.generate(self.probe(recs, [json.loads(json.dumps(pair))]), {}, mask)[0]
        site = gen(set())                                         # IB, rc 3/3, local disk: the first row
        self.assertEqual((site["env"]["UCX_TLS"], site["mpi"]["net_devices"]), ("rc,self,sm", ["ib:auto"]))
        self.assertEqual(site["scratch"], "/scratch/$USER-bootlab")       # 0.6 + 1.5 copy against 5.5 + 1.0 cold
        self.assertNotIn("allowed", site["multinode"])
        self.assertEqual((site["multinode"]["transfer"], site["multinode"]["copy_network"]), ("srun", "10.0.0."))
        site = gen({"ib"})                                        # no IB: TCP on Ethernet, local scratch, one node
        self.assertEqual(site["env"]["UCX_TLS"], "tcp,self,sm")
        self.assertEqual(site["mpi"], {"transport": "tcp-1g", "net_devices": ["subnet:10.20.0."]})
        self.assertEqual(site["scratch"], "/scratch/$USER-bootlab")       # one node by default: no copy counted
        self.assertFalse(site["multinode"]["allowed"])
        self.assertIn("no IB (masked)", site["multinode"]["basis"])
        self.assertEqual(site["multinode"]["copy_network"], "10.20.0.")
        site = gen({"local-scratch"})                             # no local disk: rc, shared scratch, one node
        self.assertEqual(site["env"]["UCX_TLS"], "rc,self,sm")
        self.assertEqual((site["scratch"], site["binds"]), ("shared", ["/data"]))
        self.assertFalse(site["multinode"]["allowed"])
        self.assertIn("no writable node-local disk", site["multinode"]["basis"])
        site = gen({"ib", "local-scratch"})                       # neither: TCP on Ethernet, shared, one node
        self.assertEqual((site["env"]["UCX_TLS"], site["mpi"]["transport"], site["scratch"]),
                         ("tcp,self,sm", "tcp-1g", "shared"))
        self.assertFalse(site["multinode"]["allowed"])

    def test_paths_from_the_installation_and_the_route(self):
        recs = [node("a-1", [0.1] * 3, [5.0] * 3), node("a-2", [0.1] * 3, [5.0] * 3)]
        for r in recs:
            r["shared"].update({"server_addr": "nfs-server", "network": "ipoib"})
        site, _, _ = sitegen.generate(self.probe(recs), {}, set())
        self.assertFalse({"sif", "python", "runs"} & set(site))                 # bootlab.toml has them
        self.assertEqual(site["shared_network"], {"server": "nfs-server", "network": "ipoib", "basis": "node probes 1"})
        self.assertNotIn("shared_network", sitegen.generate(self.probe(recs), {}, {"ib"})[0])
        recs[1]["shared"]["network"] = "ethernet"
        site, _, report = sitegen.generate(self.probe(recs), {}, set())
        self.assertNotIn("shared_network", site)
        self.assertTrue(any("differs between the probed nodes" in line for line in report))


if __name__ == "__main__":
    unittest.main()
