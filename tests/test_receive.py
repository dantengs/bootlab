"""receive.py, the copy receiver with transfer srun: only node 0's address, only with the copy's token, only
into its directory. On the loopback interface; no cluster needed."""
import io
import socket
import sys
import tarfile
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bootlab import receive  # noqa: E402

TOKEN = "ab" * 32


def tar_of(members) -> bytes:
    """members: (name, bytes) files, or (name, None, target) symbolic links."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as t:
        for m in members:
            info = tarfile.TarInfo(m[0])
            if m[1] is None:
                info.type, info.linkname = tarfile.SYMTYPE, m[2]
                t.addfile(info)
            else:
                info.size = len(m[1])
                t.addfile(info, io.BytesIO(m[1]))
    return buf.getvalue()


class Receive(unittest.TestCase):
    def start(self, peer="127.0.0.1", accept_seconds=3):
        self.dest = Path(tempfile.mkdtemp()) / "dest"
        self.dest.mkdir()
        ports, self.result, self.logs = [], [], []
        th = threading.Thread(target=lambda: self.result.append(receive.serve(
            str(self.dest), "127.0.0.1", peer, TOKEN, announce=ports.append, accept_seconds=accept_seconds,
            log=self.logs.append)))
        th.start()
        while not ports:
            pass
        return th, ports[0]

    def send(self, port, token, data):
        """A refused connection may be reset under the sender (Windows), which is the point."""
        try:
            with socket.create_connection(("127.0.0.1", port)) as c:
                c.sendall(token.encode() + data)
                c.shutdown(socket.SHUT_WR)
                c.recv(1)
        except ConnectionError:
            pass

    def test_right_copy(self):
        th, port = self.start()
        self.send(port, TOKEN, tar_of([("a/b.bin", b"x" * 1000), ("c.txt", b"hello")]))
        th.join()
        self.assertEqual(self.result, [0])
        self.assertEqual((self.dest / "a" / "b.bin").read_bytes(), b"x" * 1000)
        self.assertEqual((self.dest / "c.txt").read_bytes(), b"hello")

    def test_wrong_token_refused(self):
        th, port = self.start(accept_seconds=2)
        self.send(port, "cd" * 32, tar_of([("evil.txt", b"x")]))
        th.join()
        self.assertEqual(self.result, [2])
        self.assertFalse(any(self.dest.iterdir()))
        self.assertIn("wrong token", self.logs[0])

    def test_wrong_token_then_right(self):
        th, port = self.start()
        self.send(port, "cd" * 32, tar_of([("evil.txt", b"x")]))
        self.send(port, TOKEN, tar_of([("good.txt", b"y")]))
        th.join()
        self.assertEqual(self.result, [0])
        self.assertEqual(sorted(p.name for p in self.dest.iterdir()), ["good.txt"])

    def test_other_address_refused(self):
        th, port = self.start(peer="127.0.0.2", accept_seconds=2)    # node 0 is 127.0.0.2, this comes from .1
        self.send(port, TOKEN, tar_of([("evil.txt", b"x")]))
        th.join()
        self.assertEqual(self.result, [2])
        self.assertFalse(any(self.dest.iterdir()))
        self.assertIn("not 127.0.0.2", self.logs[0])

    def test_paths_out_of_dest_refused(self):
        """.., an absolute name, a link out of the directory: the copy stops (1), nothing lands outside."""
        for members in ([("../escaped.txt", b"x")], [("a/../../escaped.txt", b"x")], [("/abs.txt", b"x")],
                        [("ln", None, "/etc"), ("ln/passwd", b"x")], [("ln", None, "../..")]):
            th, port = self.start()
            self.send(port, TOKEN, tar_of(members))
            th.join()
            self.assertEqual(self.result, [1], members)
            self.assertFalse((self.dest.parent / "escaped.txt").exists())
            self.assertFalse((self.dest / "abs.txt").exists())
            self.assertTrue(any("Error" in l for l in self.logs), self.logs)


if __name__ == "__main__":
    unittest.main()
