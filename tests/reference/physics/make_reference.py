"""Makes the bt references of tests/test_physics.py: python tests/reference/physics/make_reference.py

Each case directory has bootlab's inputs (run.toml, spectrum.toml, objective.toml), its points (points.json) and,
for each point k, bt-k.json: the same spectrum as a bt parameter file, written by hand in bt's own form (the
cases real-* are bt runs' own files). This script resolves the bootlab spectrum at each point, writes
bootlab-k.json (bt-k.json with its spectrum section replaced by bootlab's), and has the production bt (r3.8 on
a cluster's login node, no Slurm job) read both: BT_PARAMS=<file> scalars-3d-exec params <set>, which
loads the model, makes every check and prints the parameters or the errors. Each run is kept as
<file>.result.json (exit code, stdout, stderr). coords/<model>.json gives bt a point with no coordinates, and
bt's error names them in order.
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))

from bootlab import inputs  # noqa: E402
from bootlab.numbers import dumps  # noqa: E402

HOST = REMOTE = BT = None   # --host, --remote, --bt: the login node's ssh alias, the directory there, bt's install


def write_bootlab_files():
    for case in sorted((HERE / "cases").iterdir()):
        for old in case.glob("bootlab-*"):
            old.unlink()
        try:
            p = inputs.load(case / "run.toml")
        except SystemExit as e:
            print(f"{case.name}: bootlab rejects the inputs:\n{e}")
            continue
        points = json.loads((case / "points.json").read_text())
        for k, point in enumerate(points):
            try:
                spectrum = inputs.resolve_spectrum(p, point)
            except SystemExit as e:
                print(f"{case.name} point {k}: bootlab rejects the spectrum:\n{e}")
                continue
            text = (case / f"bt-{k}.json").read_text()
            (case / f"bootlab-{k}.json").write_text(replace_spectrum(text, dumps(spectrum, indent=1)), newline="\n")


def replace_spectrum(text, new):
    """The bt file with bootlab's spectrum in place of its own; every other character stays as written."""
    i = text.index('"spectrum"')
    start = text.index("{", i)
    depth = 0
    for end in range(start, len(text)):
        depth += {"{": 1, "}": -1}.get(text[end], 0)
        if depth == 0:
            break
    return text[:start] + new + text[end + 1:]


RUNNER = r"""
set -u
export LD_LIBRARY_PATH=%(bt)s/hyperion/lib
cd %(remote)s
for f in cases/*/bt-*.json cases/*/bootlab-*.json coords/*.json; do
  case $f in *.result.json) continue;; esac
  if grep -q '"bound"' "$f"; then set=navigator; else set=feasibility; fi
  BT_PARAMS=$PWD/$f %(bt)s/hyperion/bin/scalars-3d-exec params $set > "$f.stdout" 2> "$f.stderr"
  rc=$?
  python3 -c 'import json,sys; f=sys.argv[1]; print(json.dumps({"set": sys.argv[2], "exit": int(sys.argv[3]), "stdout": open(f+".stdout").read(), "stderr": open(f+".stderr").read()}, indent=1))' "$f" $set $rc > "${f%%.json}.result.json"
  rm -f "$f.stdout" "$f.stderr"
done
"""


def run_bt():
    for old in HERE.glob("*/**/*.result.json"):
        old.unlink()
    subprocess.run(["ssh", HOST, f"rm -rf {REMOTE}/cases {REMOTE}/coords"], check=True)
    subprocess.run(["scp", "-q", "-r", str(HERE / "cases"), str(HERE / "coords"), f"{HOST}:{REMOTE}/"], check=True)
    subprocess.run(["ssh", HOST, "bash -s"], input=(RUNNER % {"bt": BT, "remote": REMOTE}).encode(), check=True)
    tar = subprocess.run(["ssh", HOST, f"cd {REMOTE} && tar cf - cases/*/*.result.json coords/*.result.json"],
                         capture_output=True, check=True).stdout
    subprocess.run(["tar", "xf", "-", "-C", str(HERE)], input=tar, check=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-bt", action="store_true", help="only the bootlab-k.json files")
    ap.add_argument("--host")
    ap.add_argument("--remote")
    ap.add_argument("--bt")
    a = ap.parse_args()
    write_bootlab_files()
    if not a.no_bt:
        if not (a.host and a.remote and a.bt):
            raise SystemExit("bt needs --host, --remote and --bt")
        HOST, REMOTE, BT = a.host, a.remote, a.bt
        run_bt()
