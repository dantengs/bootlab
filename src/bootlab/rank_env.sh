#!/bin/bash
# bootlab's rank wrapper: srun starts every MPI rank through it, outside the container, then it execs the rest
# of the command line (apptainer exec ...). srun hands every node the same environment, but the network device
# can differ by node, so it is chosen here from BOOTLAB_NET_DEVICES (the site's [mpi] net_devices, set by the
# driver): rules separated by spaces, the first that matches this host wins:
#   ib:auto                       this node's IB port (no port tables): ACTIVE, LinkUp, link_layer InfiniBand, the highest rate; at equal rates
#                                 the port whose IPoIB interface the shared file system's server is reached over
#                                 (ip route get $BOOTLAB_ROUTE_ADDR); not the ports in BOOTLAB_IB_EXCLUDE
#                                 ("<host>=<port> ...", the site's [mpi] ib_exclude). Every rank of a node reads
#                                 the same sysfs, so they agree.
#   <host pattern>=<UCX device>   bash glob on `hostname -s`, e.g. node0[1-4]=mlx5_1:1
#   subnet:<IPv4 prefix>          the interface that has an address starting with it, e.g. subnet:10.20.0.
#                                 (for UCX_TLS=tcp: the network the ranks talk over)
# No BOOTLAB_NET_DEVICES: UCX_NET_DEVICES is left as it is (UCX's own choice). A host no rule matches stops the
# rank (fail fast: a wrong port is slow or hangs). BOOTLAB_NET_EXPLAIN=1: one line "<host> <device>: <why>" on
# stdout before the command (the driver's driver.log line for each node).
h=${BOOTLAB_HOSTNAME:-$(hostname -s)}
d=
why=
ib_auto() {
  local root=${BOOTLAB_SYSFS_IB:-/sys/class/infiniband} route=${BOOTLAB_ROUTE_DEV:-} p dev name rate pref
  local best_rate=-1 best_pref=-1
  if [ -z "$route" ] && [ -n "${BOOTLAB_ROUTE_ADDR:-}" ]; then
    route=$(ip route get "$BOOTLAB_ROUTE_ADDR" 2>/dev/null | sed -n 's/.* dev \([^ ]*\).*/\1/p')
  fi
  for p in "$root"/*/ports/*; do
    [ -r "$p/state" ] || continue
    dev=${p#"$root"/}; dev=${dev%%/*}; name=$dev:${p##*/}
    case " ${BOOTLAB_IB_EXCLUDE:-} " in *" $h=$name "*) why="$why $name excluded by the site;"; continue ;; esac
    if ! grep -q ACTIVE "$p/state" || ! grep -q LinkUp "$p/phys_state" || [ "$(cat "$p/link_layer")" != InfiniBand ]; then
      why="$why $name $(cut -d' ' -f2 "$p/state")/$(cut -d' ' -f2 "$p/phys_state");"
      continue
    fi
    rate=$(sed -n 's|^\([0-9]*\).*Gb/sec.*|\1|p' "$p/rate")
    rate=${rate:-0}
    pref=0
    [ -n "$route" ] && [ -e "$root/$dev/device/net/$route" ] && pref=1
    why="$why $name $rate Gb/s$([ $pref = 1 ] && echo ", the shared file system's network ($route)");"
    if [ "$rate" -gt "$best_rate" ] || { [ "$rate" -eq "$best_rate" ] && [ "$pref" -gt "$best_pref" ]; }; then
      d=$name best_rate=$rate best_pref=$pref
    fi
  done
}
if [ -n "${BOOTLAB_NET_DEVICES:-}" ]; then
  for rule in $BOOTLAB_NET_DEVICES; do
    case $rule in
      ib:auto) ib_auto ;;
      subnet:*) d=$(ip -4 -o addr | awk -v p="${rule#subnet:}" 'index($4, p) == 1 {print $2; exit}') ;;
      *=*) pattern=${rule%%=*}
           # shellcheck disable=SC2053
           [[ $h == $pattern ]] && d=${rule#*=} ;;
    esac
    [ -n "$d" ] && break
  done
  [ -z "${BOOTLAB_NET_EXPLAIN:-}" ] || echo "$h ${d:-none}:${why:- rule $BOOTLAB_NET_DEVICES}"
  [ -n "$d" ] || { echo "rank_env.sh: no network device for $h in BOOTLAB_NET_DEVICES ($BOOTLAB_NET_DEVICES)$why" >&2; exit 1; }
  export UCX_NET_DEVICES=$d
fi
exec "$@"
