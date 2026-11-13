"""rank_env.sh's ib:auto on the IB ports of lab's nodes (the rule must pick, on each
known node, the port a hand-written table had): python -m unittest tests.test_rank_env. Fake sysfs trees,
the real rank_env.sh in bash. Measured: nodeb-3, nodec-1,
noded-0, nodea-1 (mlx5_0 down, link_downed 255), nodea-2 (both at 100 Gb/s, the
route over mlx5_0's ib2 decides; its mlx5_1 has link_downed 255). From the hand-written site
(not probed): nodeb-2 (mlx5_0 the flapping port, excluded by the site), nodea-0 (mlx5_0
25 Gb/s, mlx5_1 100 Gb/s)."""
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

RANK_ENV = Path(__file__).resolve().parents[1] / "src" / "bootlab" / "rank_env.sh"
UP = ("4: ACTIVE", "5: LinkUp")
DOWN = ("1: DOWN", "3: Disabled")
EDR, SDR, G25 = "100 Gb/sec (4X EDR)", "10 Gb/sec (4X SDR)", "25 Gb/sec (1X EDR)"

# host -> ([(device, state, rate, IPoIB netdev)], the netdev the shared file system's server is routed over)
NODES = {
    "nodeb-3": ([("mlx5_0", UP, EDR, "ib0"), ("mlx5_1", DOWN, SDR, "ib1")], "ib0"),
    "nodec-1": ([("mlx5_0", DOWN, SDR, "ib4"), ("mlx5_1", UP, EDR, "ib5")], "ib5"),
    "noded-0": ([("mlx5_0", UP, EDR, "ib6"), ("mlx5_1", UP, EDR, "ib7")], "ib6"),
    "nodeb-2": ([("mlx5_0", UP, EDR, "ib0"), ("mlx5_1", UP, EDR, "ib1")], "ib0"),
    "nodea-0": ([("mlx5_0", UP, G25, "ib0"), ("mlx5_1", UP, EDR, "ib1")], "ib1"),
    "nodea-1": ([("mlx5_0", DOWN, SDR, "ib2"), ("mlx5_1", UP, EDR, "ib3")], "ib3"),
    "nodea-2": ([("mlx5_0", UP, EDR, "ib2"), ("mlx5_1", UP, EDR, "ib3")], "ib2"),
}
# the hand-written lab table for these hosts
TABLE = {"nodeb-3": "mlx5_0:1", "nodec-1": "mlx5_1:1", "noded-0": "mlx5_0:1",
         "nodeb-2": "mlx5_1:1", "nodea-0": "mlx5_1:1", "nodea-1": "mlx5_1:1",
         "nodea-2": "mlx5_0:1"}
EXCLUDE = "nodeb-2=mlx5_0:1"      # sites/lab: the flapping second IB


def sysfs(root: Path, ports):
    for dev, (state, phys), rate, net in ports:
        p = root / dev / "ports" / "1"
        p.mkdir(parents=True)
        (p / "state").write_text(state + "\n")
        (p / "phys_state").write_text(phys + "\n")
        (p / "rate").write_text(rate + "\n")
        (p / "link_layer").write_text("InfiniBand\n")
        (root / dev / "device" / "net" / net).mkdir(parents=True)


@unittest.skipUnless(shutil.which("bash") and os.name == "posix", "bash on a POSIX host")
class IbAuto(unittest.TestCase):
    def pick(self, host, ports, route, exclude=""):
        with tempfile.TemporaryDirectory() as d:
            sysfs(Path(d), ports)
            env = {**os.environ, "BOOTLAB_NET_DEVICES": "ib:auto", "BOOTLAB_HOSTNAME": host, "BOOTLAB_SYSFS_IB": d,
                   "BOOTLAB_ROUTE_DEV": route, "BOOTLAB_IB_EXCLUDE": exclude, "BOOTLAB_NET_EXPLAIN": "1"}
            r = subprocess.run(["bash", str(RANK_ENV), "printenv", "UCX_NET_DEVICES"], env=env, capture_output=True,
                               text=True)
        return r

    def test_known_nodes_match_the_table(self):
        for host, (ports, route) in NODES.items():
            with self.subTest(host=host):
                r = self.pick(host, ports, route, EXCLUDE)
                self.assertEqual(r.returncode, 0, r.stderr)
                explain, device = r.stdout.strip().splitlines()
                self.assertEqual(device, TABLE[host])
                self.assertTrue(explain.startswith(f"{host} {TABLE[host]}:"), explain)

    def test_equal_rates_without_a_route_take_the_first(self):
        ports, _ = NODES["noded-0"]
        self.assertEqual(self.pick("x", ports, "").stdout.split()[-1], "mlx5_0:1")
        self.assertEqual(self.pick("x", ports, "ib7").stdout.split()[-1], "mlx5_1:1")   # the route decides

    def test_no_port_stops_the_rank(self):
        r = self.pick("x", [("mlx5_0", DOWN, SDR, "ib0")], "")
        self.assertEqual(r.returncode, 1)
        self.assertIn("no network device", r.stderr)


if __name__ == "__main__":
    unittest.main()
