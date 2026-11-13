"""The probe's parsing and choices (bootlab.probe), with made-up nodes: python -m unittest tests.test_probe."""
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bootlab import probe  # noqa: E402

SINFO = """\
cpua-1|normal*|idle|512|2:128:2|376000|amd,cpua,mem384g
cpua-2|normal*|mixed|512|2:128:2|376000|amd,cpua,mem384g
cpua-2|long|idle|512|2:128:2|376000|amd,cpua,mem384g
cpub-1|normal*|idle|256|2:64:2|253000|amd,cpub,mem256g
cpub-2|normal*|idle|256|2:64:2|252985|amd,cpub,mem256g
gpu-07|normal*|drained|64|1:32:2|128000|(null)
"""


def port(name, rate="100 Gb/sec (4X EDR)", state="4: ACTIVE", phys="5: LinkUp", downed=0, netdevs=()):
    return {"port": name, "state": state, "phys_state": phys, "rate": rate, "link_layer": "InfiniBand",
            "netdevs": list(netdevs), "counters": {"link_downed": downed, "link_error_recovery": 0, "symbol_error": 0}}


class Plan(unittest.TestCase):
    def test_sinfo_and_types(self):
        nodes = probe.parse_sinfo(SINFO)
        self.assertEqual(nodes[0]["physical_cores"], 256)          # sockets x cores, not %c
        self.assertTrue(nodes[0]["default_partition"])
        types = probe.node_types(nodes)
        self.assertEqual(sorted(types), ["cpua", "cpub", "gpu"])   # a feature only that group has, else host prefix
        self.assertEqual(types["cpua"]["hosts"], ["cpua-1", "cpua-2"])
        self.assertEqual(types["cpua"]["partitions"], ["normal", "long"])
        self.assertEqual(types["cpub"]["physical_cores"], 128)
        self.assertEqual(types["cpub"]["hosts"], ["cpub-1", "cpub-2"])            # RealMemory differs a little
        self.assertEqual(types["cpub"]["mem_mb"], 252985)                         # the least

    def test_pick_idle(self):
        nodes = probe.parse_sinfo(SINFO)
        self.assertEqual(probe.pick_idle(nodes, ["cpua-2", "cpua-1"]), "cpua-1")   # cpua-2 is mixed in one partition
        self.assertIsNone(probe.pick_idle(nodes, ["gpu-07"]))


class Node(unittest.TestCase):
    def test_choose_ib(self):
        start = [port("mlx5_0:1", "25 Gb/sec (1X EDR)"), port("mlx5_1:1")]
        self.assertEqual(probe.choose_ib(start, start), "mlx5_1:1")                 # the faster
        flapped = [start[0], port("mlx5_1:1", downed=1)]
        self.assertEqual(probe.choose_ib(start, flapped), "mlx5_0:1")               # link_downed rose
        twin = [port("mlx5_0:1", netdevs=["ib0"]), port("mlx5_1:1", netdevs=["ib1"])]
        self.assertEqual(probe.choose_ib(twin, twin, ["ib1"]), "mlx5_1:1")          # equal rates: the shared FS's
        down = [port("mlx5_0:1", state="1: DOWN"), port("mlx5_1:1", phys="3: Disabled")]
        self.assertIsNone(probe.choose_ib(down, down))

    def test_mounts(self):
        text = ("22 1 259:2 / / rw,relatime - xfs /dev/nvme0n1p2 rw\n"
                "98 22 0:52 / /data/work rw,relatime - nfs4 nfs-server:/work "
                "rw,vers=4.1,rsize=1048576,proto=tcp,addr=nfs-server,clientaddr=10.0.0.9\n"
                "99 22 259:3 / /data/scratch rw - ext4 /dev/nvme0n1p3 rw\n")
        mounts = probe.parse_mountinfo(text)
        m = probe.mount_of("/data/work/u/runs", mounts)
        self.assertEqual((m["fstype"], probe.opt(m["options"], "addr"), probe.opt(m["options"], "proto")),
                         ("nfs4", "nfs-server", "tcp"))
        self.assertEqual(probe.mount_of("/data/scratchy", mounts)["mountpoint"], "/")   # not a prefix of a name
        self.assertEqual(probe.mount_of("/data/scratch/x", mounts)["source"], "/dev/nvme0n1p3")

    def test_ucx_devices(self):
        text = "#      Transport: tcp\n#         Device: eth0\n#      Transport: rc_mlx5\n#         Device: mlx5_1:1\n"
        self.assertEqual(probe.parse_ucx_devices(text), [["tcp", "eth0"], ["rc_mlx5", "mlx5_1:1"]])

    def test_spread(self):
        s = probe.spread([2.0, 2.0, 2.0], 4_000_000_000)
        self.assertEqual(s["median_mb_per_s"], 2000.0)
        self.assertEqual(s["stdev_s"], 0.0)
        self.assertEqual(s["sigma_s"], time.get_clock_info("perf_counter").resolution)   # never below the clock
        self.assertEqual(probe.spread([1.0, 2.0, 3.0], 1)["sigma_s"], 1.0)

    def test_now_has_offset(self):
        self.assertRegex(probe.now(), r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d$")

    @unittest.skipUnless(hasattr(os, "major"), "POSIX")
    def test_where(self):
        """Where a directory is: path, real path, st_dev, the mount; no mergerfs branch here."""
        with tempfile.TemporaryDirectory() as d:
            mounts = probe.parse_mountinfo("22 1 259:2 / / rw,relatime - xfs /dev/nvme0n1p2 rw\n")
            w = probe.where(d, mounts)
            self.assertEqual((w["path"], w["realpath"]), (d, os.path.realpath(d)))
            self.assertRegex(w["st_dev"], r"^\d+:\d+$")
            self.assertEqual((w["fstype"], w["source"], w["mountpoint"], w["mergerfs_branch"]),
                             ("xfs", "/dev/nvme0n1p2", "/", None))

    def test_local_extra(self):
        """--local: a site's scratch directory that is not a mount point (a directory of /) is a candidate, before
        the usual names, and plan hands it to the node jobs."""
        bare = {k: v for k, v in os.environ.items() if k not in ("TMPDIR", "SLURM_TMPDIR")}   # those come first
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, bare, clear=True):
            mounts = probe.parse_mountinfo("22 1 259:2 / / rw,relatime - xfs /dev/nvme0n1p2 rw\n")
            paths = [c["path"] for c in probe.local_candidates(mounts, "0:52", [d])]
            self.assertEqual(paths[0], d)
            self.assertNotIn(d, [c["path"] for c in probe.local_candidates(mounts, "0:52")])
        args = SimpleNamespace(sif="s.sif", shared="/data/work", out="/data/work/p", setup="", py_test=None,
                               local=["/data/scratch dir", "/x"])
        cmd = probe.node_job("cpua-1", args, "py")[-1]
        self.assertIn(" --local '/data/scratch dir' --local /x", cmd)


class Pair(unittest.TestCase):
    def test_networks(self):
        self.assertEqual(probe.prefix_of("10.1.2.3/24"), "10.1.2.")
        self.assertEqual(probe.prefix_of("10.1.2.3/16"), "10.1.")

        def f(host, ib, eth):
            return {"host": host, "nics": [
                {"name": "ib0", "type": "infiniband", "operstate": "up", "speed_mbps": 100000, "ipv4": [ib]},
                {"name": "eth0", "type": "ethernet", "operstate": "up", "speed_mbps": 1000, "ipv4": [eth]},
                {"name": "usb0", "type": "ethernet", "operstate": "unknown", "speed_mbps": None,
                 "ipv4": ["169.254.3.1/24"]},
                {"name": "lo", "type": "loopback", "operstate": "unknown", "speed_mbps": None, "ipv4": ["127.0.0.1/8"]}]}
        nets = probe.shared_networks(f("a", "10.100.0.1/24", "10.1.0.1/24"), f("b", "10.100.0.2/24", "10.2.0.2/24"))
        self.assertEqual([(n["kind"], n["prefix"], n["a"], n["b"]) for n in nets],
                         [("ipoib", "10.100.0.", ("ib0", "10.100.0.1"), ("ib0", "10.100.0.2"))])   # eth: other subnets

    def test_perftest_and_errors(self):
        text = ("+--------------+\n"
                "Final:                  2000     24.542    24.518    24.518        0.31       0.31       40786       40786\n")
        self.assertEqual(probe.parse_perftest(text), {"latency_us": 24.518, "bandwidth_mb_per_s": 0.31, "iterations": 2000})
        self.assertEqual(probe.parse_perftest("no final"), {})
        log = ("x connect(fd=3, dest_addr=1.2.3.4:5) failed: Connection refused\n"
               "y connect(fd=4) failed: Cannot assign requested address\n"
               "z ucp_ep_create(proc=1) failed: Destination is unreachable\n")
        self.assertEqual(probe.count_errors(log), {"refused": 1, "cannot_assign": 1, "unreachable": 1})

    @unittest.skipUnless(hasattr(__import__("os"), "killpg"), "POSIX process groups")
    def test_limited_ends_the_group(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "log"
            t0 = time.time()
            r = probe.limited(["bash", "-c", "sleep 30 & sleep 30"], 1, log)
            self.assertTrue(r["timed_out"])
            self.assertLess(time.time() - t0, 10)


if __name__ == "__main__":
    unittest.main()
