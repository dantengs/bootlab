"""The generator's input and environment (docs/en/interfaces.md sections 6-7), without a cluster:
python -m unittest discover -s tests (from the repository root)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"   # tools/make_test_fixture.py
sys.path.insert(0, str(ROOT / "src"))

from bootlab import execute, inputs, pmp  # noqa: E402

EXAMPLE = ROOT / "examples" / "ising" / "run.toml"


class Request(unittest.TestCase):
    def test_request(self):
        """bt's model section, coordinates as exact JSON numbers in the model's order, bt's spectrum section."""
        p = inputs.load(EXAMPLE)
        x = "0.5181489000000000000000000000000000000000000000001"
        pts = {"c": {"deltaSig": x, "deltaEps": "1.4", "theta": "0.956"}}
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            text = pmp.write_request(p, Path("/m/model.json"), pts, out, Path("/b"), 64).read_text()
        self.assertIn(f"{x},", text)            # every digit, as a number, not a string
        req = json.loads(text)
        self.assertEqual(req["model"], {"name": "data", "file": str(Path("/m/model.json")), "lambda": 11,
                                        "blockPreset": "simpleboot"})
        self.assertEqual(req["objective"], "navigator")
        self.assertEqual(req["cores"], 64)
        pt = req["points"][0]
        self.assertEqual(pt["name"], "c")
        self.assertEqual(len(pt["coords"]), 3)
        self.assertEqual(pt["spectrum"], {"twistGap": 0, "isolated": [],
                                          "gaps": [{"channel": "rep1+", "spins": [0], "gap": 3},
                                                   {"channel": "rep2+", "spins": [0], "gap": 3}]})

    def test_generator_in_the_image(self):
        """The generator runs in the image like the other programs: apptainer exec of site.toml's sif, the
        generator's path there, the driver's environment (no bt site scripts or libraries from the host)."""
        from unittest import mock
        site = execute.load_site("lab", FIXTURES)
        node = execute.Node("h-1", "amd1", 128, 253099)
        seen = {}

        class Done:
            def wait(self):
                return 0

        def start(cmd, log, env=None):
            seen["cmd"], seen["env"] = [str(c) for c in cmd], env
            return Done()

        with mock.patch.object(execute, "start", start):
            pmp.generate(site, node, Path("/r/request.json"), Path("/l/generator"))
        cmd = seen["cmd"]
        i = cmd.index("apptainer")
        self.assertEqual(cmd[i:i + 2], ["apptainer", "exec"])
        self.assertEqual(cmd[cmd.index(site.sif) + 1:], [site.generator, "pmp", str(Path("/r/request.json"))])
        self.assertEqual(site.generator, "/opt/bootlab/bin/generator")
        self.assertIn("--cpus-per-task=128", cmd)
        self.assertIsNone(seen["env"])
        self.assertNotIn("--mpi=pmix", cmd)

    def test_labels(self):
        with tempfile.TemporaryDirectory() as d:
            nsv = Path(d) / "pmp.nsv"
            nsv.write_text("/j/continuum_rep1+_l0_d3_mat_ab12.json\0/j/objective_x.json\0"
                           "/j/isolated_rep2+_l2_u0.5_mat_cd.json\0")
            got = pmp.labels(nsv)
        self.assertEqual([(x["kind"], x["channel"], x["spin"], x["delta"], x["relative_to_unitarity"]) for x in got],
                         [("continuum", "rep1+", 0, "3", False), ("isolated", "rep2+", 2, "0.5", True)])


if __name__ == "__main__":
    unittest.main()
