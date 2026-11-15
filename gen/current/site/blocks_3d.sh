#!/bin/bash
# blocks_3d for the current generator (its HYP_SITE_DIR; the generator's scheduler calls <dir>/blocks_3d.sh): the
# build under $CUR_ROOT/blocks3d. Where the node has /usr/bin/time, -v logs each
# run's wall time, threads and peak memory to stderr, which Hyperion keeps in the worker log, as DSD's own site
# script does; not every node has it (two nodes of the development cluster have no time package).
export LD_LIBRARY_PATH=$CUR_ROOT/blocks3d/lib
b=$CUR_ROOT/blocks3d/bin/blocks_3d
[ -x /usr/bin/time ] && exec /usr/bin/time -v "$b" "$@"
exec "$b" "$@"
