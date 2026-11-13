"""The receiving side of a copy between the nodes of a job with transfer srun (execute.tcp_copy): a light srun
step on the target starts `python3 receive.py <dest>`, which listens, prints its port, takes one tar stream
from node 0 and unpacks it into dest.

The cluster is shared, so the receiver
  - listens only on BOOTLAB_COPY_BIND, the address of this node on the copy's network (never 0.0.0.0), on a
    port the system picks;
  - takes a connection only from BOOTLAB_COPY_PEER, node 0's address on that network; others are closed;
  - then wants BOOTLAB_COPY_TOKEN, a one-time random token of this copy (in the environment, which other users
    cannot read, unlike the command line), as the first bytes; a wrong one is closed;
  - refuses a member with an absolute name or ".." and unpacks with tarfile's "data" filter (no links that point
    out of dest, no devices); a member like that stops the copy (exit 1), and execute.distribute then finds the
    files and bytes short anyway;
  - gives up after ACCEPT_SECONDS without the right connection (exit 2), and after IDLE_SECONDS without data.
Python >= 3.11.4 (tarfile filters); the standard library only.
"""
import hmac
import os
import socket
import sys
import tarfile
import time

ACCEPT_SECONDS = 120
IDLE_SECONDS = 600


def recv_exactly(c: socket.socket, n: int) -> bytes:
    data = b""
    while len(data) < n:
        b = c.recv(n - len(data))
        if not b:
            break
        data += b
    return data


def serve(dest: str, bind: str, peer: str, token: str, announce=print, accept_seconds: float = ACCEPT_SECONDS,
          idle_seconds: float = IDLE_SECONDS, log=lambda m: print(m, file=sys.stderr, flush=True)) -> int:
    """Listen on (bind, any port), announce the port, unpack the one right stream into dest. 0 when unpacked,
    1 when the stream was refused by the filter or broke, 2 when no right connection came in time."""
    want = token.encode()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((bind, 0))
    s.listen(8)
    s.settimeout(1)
    announce(s.getsockname()[1])
    deadline = time.monotonic() + accept_seconds
    try:
        while time.monotonic() < deadline:
            try:
                c, (addr, _) = s.accept()
            except socket.timeout:
                continue
            with c:
                if addr != peer:
                    log(f"receive: refused a connection from {addr} (not {peer})")
                    continue
                c.settimeout(30)
                got = recv_exactly(c, len(want))
                if not hmac.compare_digest(got, want):
                    log(f"receive: refused a connection from {addr}: wrong token")
                    continue
                c.settimeout(idle_seconds)
                try:
                    with tarfile.open(fileobj=c.makefile("rb"), mode="r|", copybufsize=1 << 20) as t:
                        for m in t:
                            # refused outright: the data filter would strip a leading / and go on
                            if m.name.startswith("/") or ".." in m.name.split("/"):
                                raise tarfile.FilterError(f"member {m.name!r} is not under the directory")
                            t.extract(m, dest, filter="data")
                except (tarfile.TarError, OSError) as e:
                    log(f"receive: {type(e).__name__}: {e}")
                    return 1
                return 0
        log(f"receive: no connection from {peer} with the token in {accept_seconds} s")
        return 2
    finally:
        s.close()


def main():
    dest = sys.argv[1]
    env = os.environ
    sys.exit(serve(dest, env["BOOTLAB_COPY_BIND"], env["BOOTLAB_COPY_PEER"], env["BOOTLAB_COPY_TOKEN"],
                   announce=lambda port: print(port, flush=True)))


if __name__ == "__main__":
    main()
