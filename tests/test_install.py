"""An installation without cluster paths in the code, bootlab's times and the
start's check of the runs file system's network: python -m unittest tests.test_install."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bootlab import execute, inputs  # noqa: E402

SITE = """generator = "/opt/bootlab/bin/generator"
pmp2sdp = "p"
dynamical_sdp = "d"
driver_reserve_mb = 4096
binds = []
scratch = "shared"
"""


class Install(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "sites" / "x").mkdir(parents=True)
        (self.root / "sites" / "x" / "site.toml").write_text(SITE)
        self.env = mock.patch.dict(os.environ, {k: v for k, v in os.environ.items() if k != "BOOTLAB_RUNS"},
                                   clear=True)
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def config(self, text):
        (self.root / "bootlab.toml").write_text(text)

    def test_no_config_no_image(self):
        self.assertEqual(execute.install_config(self.root), {})
        with self.assertRaisesRegex(SystemExit, "no image"):
            execute.load_site("x", self.root)
        with self.assertRaisesRegex(SystemExit, "no site"):
            execute.site_name({}, self.root)

    def test_from_the_installation(self):
        self.config('site = "x"\nimage = "images/b.sif"\n')
        self.assertEqual(execute.site_name({}, self.root), "x")
        self.assertEqual(execute.site_name({"site": "y"}, self.root), "y")          # run.toml first
        s = execute.load_site("x", self.root)
        self.assertEqual(s.sif, str(self.root / "images" / "b.sif"))                # relative: from the root
        self.assertEqual(s.runs, str(self.root / "runs"))                           # the default
        self.assertIsNone(s.python)
        (self.root / "python" / "bin").mkdir(parents=True)
        (self.root / "python" / "bin" / "python3").write_text("")
        self.assertEqual(execute.load_site("x", self.root).python, str(self.root / "python" / "bin" / "python3"))

    @unittest.skipUnless(os.name == "posix", "absolute paths as on the cluster")
    def test_order(self):
        self.config('image = "/i.sif"\nruns = "/data/runs"\npython = "/opt/py/bin/python3"\n')
        self.assertEqual(execute.load_site("x", self.root).runs, "/data/runs")
        (self.root / "sites" / "x" / "site.toml").write_text(SITE + 'sif = "/s.sif"\nruns = "/site/runs"\n')
        s = execute.load_site("x", self.root)
        self.assertEqual((s.sif, s.runs, s.python), ("/s.sif", "/site/runs", "/opt/py/bin/python3"))
        os.environ["BOOTLAB_RUNS"] = "/elsewhere"
        self.assertEqual(execute.load_site("x", self.root).runs, "/elsewhere")     # the variable overrides all


class Resources(unittest.TestCase):
    def test_allow_multinode_in_run_toml(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for f in ("model.json", "spectrum.toml", "objective.toml", "run.toml"):
                (d / f).write_bytes((ROOT / "examples" / "ising" / f).read_bytes())
            base = (d / "run.toml").read_text()
            (d / "run.toml").write_text(base + "\n[resources]\nallow_multinode = true\n")
            self.assertTrue(inputs.load(d / "run.toml").run["resources"]["allow_multinode"])
            (d / "run.toml").write_text(base + '\n[resources]\nallow_multinode = "tcp"\n')
            with self.assertRaisesRegex(SystemExit, "true or false"):
                inputs.load(d / "run.toml")


class MeasuredAt(unittest.TestCase):
    """The shared numbers hold where the probe measured them."""

    def test_runs_in_the_measured_directory(self):
        site = execute.Site(name="t", sif="s", binds=[], generator="g", scratch="shared", runs="/data/inst/runs",
                            pmp2sdp="p", dynamical_sdp="d", driver_reserve_mb=0, bench_dir=Path("."))
        self.assertEqual(execute.runs_measured_note(site), [])                                  # hand-written: none
        at = lambda m: execute.runs_measured_note(execute.Site(**{**site.__dict__, "shared_measured_at": m}))
        self.assertEqual(at(["/data/inst/runs"]), [])
        self.assertEqual(at(["/data/inst"]), [])                                                # below it
        w = at(["/data/other/runs", "/data/inst/runsx"])                                        # not a prefix of a name
        self.assertEqual(len(w), 1)
        self.assertTrue(w[0].startswith("warning: the runs directory /data/inst/runs is not in the directory the probe "
                                        "measured the shared file system in (/data/other/runs, /data/inst/runsx)"))
        self.assertIn("another disk array", w[0])

    def test_install_runs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            env = {k: v for k, v in os.environ.items() if k != "BOOTLAB_RUNS"}
            with mock.patch.dict(os.environ, env, clear=True):
                self.assertEqual(execute.install_runs(root), str(root / "runs"))
                (root / "bootlab.toml").write_text('runs = "r2"\n')
                self.assertEqual(execute.install_runs(root), str(root / "r2"))
            with mock.patch.dict(os.environ, {"BOOTLAB_RUNS": "/elsewhere"}):
                self.assertEqual(execute.install_runs(root), "/elsewhere")


class Times(unittest.TestCase):
    def test_iso_with_offset(self):
        self.assertRegex(execute.now(), r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d$")


class RunsNetwork(unittest.TestCase):
    SITE = execute.Site(name="t", sif="s", binds=[], generator="g", scratch="shared", runs="/r", pmp2sdp="p",
                        dynamical_sdp="d", driver_reserve_mb=0, bench_dir=Path("."),
                        shared_network={"server": "nfs-ib", "network": "ipoib", "basis": "node probe 1"})

    def test_as_probed(self):
        lines = execute.runs_network_lines(self.SITE, {"a": "nfs4 nfs-ib tcp ib0 32"})
        self.assertEqual(lines, ["runs file system on a: nfs4 server nfs-ib over ib0 (ipoib)"])

    def test_fallback_warns(self):
        lines = execute.runs_network_lines(self.SITE, {"a": "nfs4 nfs-ib tcp ib0 32",
                                                       "b": "nfs4 nfs-eth tcp eno1 1"})
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[2].startswith("warning: b: the runs file system's server is nfs-eth over eno1 "
                                            "(ethernet); the probe saw nfs-ib over ipoib (node probe 1)"))
        self.assertIn("untested", lines[2])

    def test_not_nfs_and_no_expectation(self):
        self.assertEqual(execute.runs_network_lines(self.SITE, {"a": "lustre - - - -"}),
                         ["runs file system on a: lustre"])
        bare = execute.Site(**{**self.SITE.__dict__, "shared_network": None})
        self.assertEqual(len(execute.runs_network_lines(bare, {"b": "nfs4 nfs-eth tcp eno1 1"})), 1)

    @unittest.skipUnless(os.name == "posix" and Path("/proc/self/mountinfo").exists(), "Linux")
    def test_script_runs(self):
        out = subprocess.run(["bash", "-c", execute.RUNS_NET_SCRIPT.format(runs="/")], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(len(out.stdout.split()), 5, out.stdout)

    @unittest.skipUnless(os.name == "posix" and Path("/proc/self/mountinfo").exists(), "Linux")
    def test_script_takes_the_nfs_over_its_autofs(self):
        # /shared is an autofs entry with the NFS mounted on it, in that order in mountinfo
        info = ("25 1 0:21 / / rw,relatime - xfs /dev/mapper/root rw\n"
                "80 25 0:45 / /shared rw,relatime - autofs /etc/auto.nfs rw,fd=17\n"
                "90 80 0:53 / /shared rw,relatime - nfs4 nfs-ib:/shared "
                "rw,vers=4.1,proto=tcp,addr=nfs-ib\n"
                "91 25 0:54 / /sharedspace rw - nfs4 10.0.0.9:/w rw,addr=10.0.0.9\n")
        with tempfile.NamedTemporaryFile("w", suffix=".mountinfo", delete=False) as f:
            f.write(info)
        try:
            script = execute.RUNS_NET_SCRIPT.format(runs="/shared/user/bootlab/runs")
            out = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                                 env={**os.environ, "BOOTLAB_MOUNTINFO": f.name}).stdout.split()
        finally:
            os.unlink(f.name)
        self.assertEqual(out[:3], ["nfs4", "nfs-ib", "tcp"])


class SitesWithoutTreePaths(unittest.TestCase):
    """The repository's sites give no runs or python: those are the installation's (bootlab.toml, else
    <install root>/runs and <install root>/python), so a site never points one installation at another's tree
    (a test tree's regression once ran into the deployed tree's runs)."""

    def test_no_runs_or_python(self):
        import tomllib
        files = sorted((ROOT / "sites").glob("*/site*.toml"))
        self.assertTrue(files or not (ROOT / "sites").exists())     # a tree without sites/ has nothing to check
        for f in files:
            with open(f, "rb") as fh:
                d = tomllib.load(fh)
            self.assertFalse({"runs", "python"} & set(d), f"{f.relative_to(ROOT)} gives runs or python")

    def test_from_the_installation(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BOOTLAB_RUNS", None)
            root = Path(d)
            s = execute.site_paths({"sif": "/images/x.sif"}, root)
            self.assertEqual(s["runs"], str(root / "runs"))
            (root / "python" / "bin").mkdir(parents=True)
            (root / "python" / "bin" / "python3").touch()
            self.assertEqual(execute.site_paths({"sif": "/images/x.sif"}, root)["python"],
                             str(root / "python" / "bin" / "python3"))


if __name__ == "__main__":
    unittest.main()
