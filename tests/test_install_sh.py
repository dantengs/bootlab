"""install.sh with a stand-in apptainer (POSIX only): python -m unittest tests.test_install_sh. The stand-in runs
the command after the image on this host with /opt/bootlab/hostpython pointing at a made-up tree whose
bin/python3 is this Python; the install root is a temporary copy with one trivial test."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FAKE_APPTAINER = """#!/bin/bash
# apptainer exec [--bind X] <image> <command...>: the command here, the image's hostpython a local tree
[ "$1" = exec ] || exit 9
shift
while [ "${1#--}" != "$1" ]; do shift 2; done
shift
args=()
for a in "$@"; do args+=("${a//\\/opt\\/bootlab\\/hostpython/$FAKE_HOSTPYTHON}"); done
exec "${args[@]}"
"""


@unittest.skipUnless(os.name == "posix" and shutil.which("bash") and shutil.which("sha256sum"), "POSIX")
class InstallSh(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "bootlab"
        (self.root / "tests").mkdir(parents=True)
        shutil.copy(ROOT / "install.sh", self.root / "install.sh")
        (self.root / "tests" / "test_one.py").write_text("import unittest\nclass T(unittest.TestCase):\n"
                                                          "    def test(self):\n        pass\n")
        shutil.copytree(ROOT / "src" / "bootlab", self.root / "src" / "bootlab",
                        ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(ROOT / "tests" / "reference" / "ising-l11", self.root / "tests" / "reference" / "ising-l11")
        hp = self.tmp / "hostpython"
        (hp / "bin").mkdir(parents=True)
        # the stand-in for the image's wrapper keeps the environment it was started with (bin/bootlab's)
        (hp / "bin" / "python3").write_text(f"#!/bin/sh\nenv > \"$(dirname \"$0\")/../env-{{$$}}.txt\"\n"
                                            f"exec {sys.executable} \"$@\"\n".replace("{$$}", "$$"))
        (hp / "bin" / "python3").chmod(0o755)
        digest = hashlib.sha256((hp / "bin" / "python3").read_bytes()).hexdigest()
        (hp / "MANIFEST").write_text(f"{digest}  bin/python3  -\n")
        bin_ = self.tmp / "bin"
        bin_.mkdir()
        (bin_ / "apptainer").write_text(FAKE_APPTAINER)
        (bin_ / "apptainer").chmod(0o755)
        self.image = self.tmp / "r4.sif"
        self.image.write_bytes(b"image")
        self.env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "FAKE_HOSTPYTHON": str(hp)}

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def install(self, *args):
        return subprocess.run(["bash", str(self.root / "install.sh"), "--image", str(self.image), *args],
                              env=self.env, capture_output=True, text=True)

    def test_install(self):
        sha = hashlib.sha256(b"image").hexdigest()
        r = self.install("--sha256", sha, "--site", "mysite")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("1 files from the image checked", r.stdout)
        toml = (self.root / "bootlab.toml").read_text()
        self.assertIn('site = "mysite"', toml)
        self.assertIn(f'image = "{self.image}"', toml)
        self.assertIn("Ran 1 test", r.stderr)
        self.assertTrue(os.access(self.root / "bin" / "bootlab", os.X_OK))
        self.assertIn(f"wrote {self.root}/bin/bootlab", r.stdout)

    def test_bin_bootlab(self):
        """bin/bootlab from elsewhere (a link, another current directory), no PYTHONPATH: the regression's check
        runs, and the environment it gives the Python (the driver's, its children's) has no PYTHONPATH."""
        self.assertEqual(self.install().returncode, 0)
        run = self.tmp / "run"
        run.mkdir()
        steps = json.loads((ROOT / "tests" / "reference" / "ising-l11" / "bt-steps.json").read_text())["steps"]
        (run / "solves.jsonl").write_text("".join(json.dumps({**s, "verdict": "stepped"}) + "\n" for s in steps))
        link = self.tmp / "elsewhere" / "bootlab"
        link.parent.mkdir()
        link.symlink_to(self.root / "bin" / "bootlab")
        env = {k: v for k, v in self.env.items() if k != "PYTHONPATH"}
        for py in (self.root / "python").glob("env-*.txt"):     # the installation's own runs of the Python
            py.unlink()
        r = subprocess.run([str(link), "regress", "check", str(run)], env=env, cwd=str(self.tmp),
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("PASS: 3 steps", r.stdout)
        seen = list((self.root / "python").glob("env-*.txt"))
        self.assertEqual(len(seen), 1)
        self.assertNotIn("PYTHONPATH=", seen[0].read_text())
        r = subprocess.run([str(link), "regress", "check", str(run)], env={**env, "PYTHONPATH": "/nonexistent"},
                           cwd=str(self.tmp), capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)      # -I: a stray PYTHONPATH is not read

    def test_refusals(self):
        r = self.install("--sha256", "0" * 64)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("expected", r.stderr)
        self.assertEqual(self.install().returncode, 0)
        r = self.install()                                        # python/ is there now: nothing overwritten
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("exists", r.stderr)

    def test_manifest_mismatch(self):
        hp = Path(self.env["FAKE_HOSTPYTHON"])
        (hp / "MANIFEST").write_text(f"{'0' * 64}  bin/python3  -\n")
        r = self.install()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("differs from its MANIFEST", r.stderr)


if __name__ == "__main__":
    unittest.main()
