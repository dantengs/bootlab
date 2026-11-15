#!/bin/bash
# bootlab's installation: needs bash and apptainer only, as the
# host may have no Python >= 3.11. Run in the install root (the directory with src/ and sites/),
# with apptainer on PATH (the cluster's module for it loaded first; install.sh stops when it finds none):
#   ./install.sh --image <sif> [--sha256 <hex>] [--site <name>] [--runs <dir>]
# 1. checks the image (its sha256 when given);
# 2. copies the image's /opt/bootlab/hostpython (r4 and later) to <root>/python and checks it against its MANIFEST
#    ("<sha256>  <path>  <rpm>" a line); <root>/python/bin/python3 then runs the image's Python 3.12 on the host
#    with the image's own loader and libraries, without LD_LIBRARY_PATH or PYTHONHOME;
# 3. writes <root>/bootlab.toml (image, and site and runs when given) unless it exists;
# 4. writes <root>/bin/bootlab, the command (that Python on <root>/src, no PYTHONPATH), unless it exists;
# 5. checks the installation quickly, a failure stopping it: every module of src/bootlab imports, bin/bootlab
#    starts, and two quick unit tests (tests/test_install.py, tests/test_regress.py) pass;
# 6. runs bin/bootlab selftest (every unit test): a failure there is a warning only, as most of those tests are
#    not about the installation (run it again any time with bin/bootlab selftest).
# Nothing is overwritten or removed: an existing <root>/python stops the installation.
set -euo pipefail
root=$(cd "$(dirname "$0")" && pwd)
image= sha= site= runs=
while [ $# -gt 0 ]; do
  case $1 in
    --image) image=$2; shift 2 ;;
    --sha256) sha=$2; shift 2 ;;
    --site) site=$2; shift 2 ;;
    --runs) runs=$2; shift 2 ;;
    *) echo "install.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done
die() { echo "install.sh: $*" >&2; exit 1; }
[ -n "$image" ] || die "--image <sif> is required"
image=$(cd "$(dirname "$image")" && pwd)/$(basename "$image")
[ -r "$image" ] || die "cannot read $image"
command -v apptainer > /dev/null || die "apptainer is not on PATH: load the cluster's module for it first (for example \`module load apptainer\`), then run install.sh again"
if [ -n "$sha" ]; then
  got=$(sha256sum "$image" | cut -c1-64)
  [ "$got" = "$sha" ] || die "$image: sha256 $got, expected $sha"
  echo "image $image: sha256 checked"
fi
py=$root/python
[ ! -e "$py" ] || die "$py exists: the installation leaves it as it is (move it away to install anew)"
apptainer exec "$image" test -f /opt/bootlab/hostpython/MANIFEST \
  || die "$image has no /opt/bootlab/hostpython (an image from r4 on carries the driver's Python)"
apptainer exec --bind "$root" "$image" cp -a /opt/bootlab/hostpython "$py"
(cd "$py" && awk '{print $1 "  " $2}' MANIFEST | sha256sum -c --quiet -) || die "$py differs from its MANIFEST"
echo "python: $(wc -l < "$py/MANIFEST") files from the image checked; $("$py/bin/python3" -c 'import sys; print(sys.version.split()[0])')"
"$py/bin/python3" -c 'import sys; sys.exit(sys.version_info < (3, 11))' || die "the image's Python is older than 3.11"
if [ -e "$root/bootlab.toml" ]; then
  echo "bootlab.toml exists: left as it is"
else
  {
    echo "# written by install.sh $(date -Iseconds) (bootlab.example.toml explains the keys)"
    [ -z "$site" ] || echo "site = \"$site\""
    echo "image = \"$image\""
    [ -z "$runs" ] || echo "runs = \"$runs\""
  } > "$root/bootlab.toml"
  echo "wrote $root/bootlab.toml"
fi
if [ -e "$root/bin/bootlab" ]; then
  echo "bin/bootlab exists: left as it is"
else
  mkdir -p "$root/bin"
  cat > "$root/bin/bootlab" <<'EOF'
#!/bin/sh
# bootlab's command, written by install.sh: bin/bootlab <run | resume | submit | summary | regress | selftest | probe | site> ...
# The install root from this file's own place (a link to it works too); its Python, isolated (-I: no PYTHONPATH
# read, no current directory on the path), with <root>/src put on the path in the process: nothing is added to
# the environment, so the children (srun, apptainer, the steps) see no PYTHONPATH.
r=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
exec "$r/python/bin/python3" -I -c 'import sys; sys.path.insert(0, sys.argv.pop(1)); from bootlab.cli import main; main(sys.argv[1:])' "$r/src" "$@"
EOF
  chmod +x "$root/bin/bootlab"
  echo "wrote $root/bin/bootlab"
fi
"$py/bin/python3" -I -c 'import importlib, pkgutil, sys; sys.path.insert(0, sys.argv[1]); import bootlab
for m in pkgutil.iter_modules(bootlab.__path__): importlib.import_module("bootlab." + m.name)' "$root/src" \
  || die "a module of $root/src/bootlab does not import with the image's Python"
"$root/bin/bootlab" --help > /dev/null || die "$root/bin/bootlab does not start"
(cd "$root/tests" && "$py/bin/python3" -m unittest -q test_install test_regress) \
  || die "the installation's quick tests failed (tests/test_install.py, tests/test_regress.py)"
echo "installation checked: modules import, bin/bootlab starts, quick tests pass"
if "$root/bin/bootlab" selftest; then
  echo "selftest: every unit test passed"
else
  echo "install.sh: warning: bin/bootlab selftest failed (above); the installation itself is checked and in place" >&2
fi
