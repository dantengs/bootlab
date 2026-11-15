"""Mixed precision (bootlab.precision): the choice of precision, the health check, and the checkpoint at another
precision (Elemental's BigFloat records)."""
import json
import os
import struct
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from bootlab import precision

B = 2 ** 64
FIXTURE = Path(__file__).parent / "fixtures" / "checkpoint-384"
FIXTURE_785 = Path(__file__).parent / "fixtures" / "checkpoint-785"


def record(value: Fraction, n: int) -> bytes:
    """value as Elemental serializes a BigFloat of n limbs: int32 size (negative for negative values), int64
    exponent in limbs, n limbs least significant first (the first |size| meaningful; the rest garbage here)."""
    if value == 0:
        return struct.pack("<iq", 0, 0) + b"\xab" * 8 * n
    sign, v = (1 if value > 0 else -1), abs(value)
    exp = 0
    while v >= 1:
        v, exp = v / B, exp + 1
    while v < Fraction(1, B):
        v, exp = v * B, exp - 1
    d = []
    for _ in range(n - 1):            # n - 1 limbs: GMP's precision of n - 1 limbs, size up to n - 1 here
        v *= B
        d.append(int(v))
        v -= int(v)
    while d and d[-1] == 0:
        d.pop()
    return struct.pack("<iq", sign * len(d), exp) + b"".join(struct.pack("<Q", x) for x in reversed(d)) \
        + b"\xab" * 8 * (n - len(d))


def value(b: bytes, i: int) -> Fraction:
    s, exp = struct.unpack_from("<iq", b, i)
    a = abs(s)
    limbs = struct.unpack_from(f"<{a}Q", b, i + 12)
    v = sum(Fraction(x) * Fraction(B) ** (exp - a + k) for k, x in enumerate(limbs))
    return v if s >= 0 else -v


def rank_file(values: list, n: int, shape=(2, 3)) -> bytes:
    """Blocks of shape, values in order, as many blocks as it takes."""
    out, k = b"", 0
    h, w = shape
    while k < len(values):
        out += struct.pack("<qq", h, w) + b"".join(record(v, n) for v in values[k:k + h * w])
        k += h * w
    return out


VALUES = [Fraction(1, 3), Fraction(-22, 7), Fraction(0), Fraction(10) ** 40 / 3, -Fraction(1, 10 ** 30), Fraction(5)]


class Choice(unittest.TestCase):
    def test_ladder(self):
        """A loss of 95 digits: 384 bits never (20.6 digits at gap 1); 512 bits down to the gap floor 1e-30 (41
        digits); full when no rung below it keeps 30 digits."""
        self.assertEqual(precision.choose(785, "1"), 512)
        self.assertEqual(precision.choose(785, "0.0784"), 512)
        self.assertEqual(precision.choose(1024, "2.4e-11"), 512)
        self.assertEqual(precision.choose(1024, "1e-30"), 512)
        self.assertAlmostEqual(precision.digits_kept(512, "1e-30"), 41.1, places=1)
        self.assertEqual(precision.choose(1024, "1e-50"), 768)
        self.assertEqual(precision.choose(512, "1e-30"), 512)
        self.assertEqual(precision.choose(384, "1e-30"), 384)

    def test_low_stays(self):
        line = {"verdict": "stepped", "precision": 512, "gap": "1", "find_minimum": False, "climbed": False,
                "call_iterations": 2}
        self.assertEqual(precision.decide(line, [], 1024), {"precision_state": "low", "precision_next": 512})

    def test_raised_by_the_gap(self):
        line = {"verdict": "stepped", "precision": 384, "gap": "1e-25", "find_minimum": False, "climbed": False,
                "call_iterations": 2}
        d = precision.decide(line, [], 1024)
        self.assertEqual((d["precision_state"], d["precision_next"]), ("low", 512))
        self.assertIn("below 30", d["precision_note"])

    def test_formula_locks(self):
        """Full 768, gap 1e-60: 512 bits keep 23 digits and nothing between: locked."""
        line = {"verdict": "stepped", "precision": 512, "gap": "1e-60", "find_minimum": False, "climbed": False,
                "call_iterations": 2}
        d = precision.decide(line, [], 768)
        self.assertEqual((d["precision_state"], d["precision_next"]), ("locked", 768))
        self.assertIn("no lower precision does", d["precision_note"])


def call(bits, it=3, state=None, climbed=False, find_minimum=False, verdict="stepped", gap="1e-12", first=False):
    """A record of a call at `bits` that left precision_state `state` (None: a record from before dynamic
    precision, with precision_next only: full after a call at full, else bits)."""
    line = {"verdict": verdict, "precision": bits, "gap": gap, "climbed": climbed, "find_minimum": find_minimum,
            "call_iterations": it}
    if first:
        line["first_call"] = True
    if state:
        line["precision_state"] = state
        line["precision_next"] = 1024 if state in ("confirm", "high", "locked") else bits
    else:
        line["precision_next"] = bits
    return line


class Decide(unittest.TestCase):
    """The states of precision.decide at full 1024 bits, gap 1e-12 (512 bits keep 52 digits)."""
    LOW = [call(512, 180, "low", first=True), call(512, 3, "low"), call(512, 3, "low")]

    def test_confirm(self):
        d = precision.decide(call(512, 3, find_minimum=True), self.LOW, 1024)
        self.assertEqual((d["precision_state"], d["precision_next"]), ("confirm", 1024))
        self.assertIn("the same call again at full precision", d["precision_note"])

    def test_confirmed_the_paths(self):
        """The call again at full fails the check too: low again at the ladder's choice."""
        hist = self.LOW + [call(512, 3, "confirm", find_minimum=True)]
        d = precision.decide(call(1024, 3, find_minimum=True), hist, 1024)
        self.assertEqual((d["precision_state"], d["precision_next"]), ("low", 512))
        self.assertIn("confirmed at full precision (findMinimumQ changed", d["precision_note"])

    def test_confirmed_the_precisions(self):
        hist = self.LOW + [call(512, 3, "confirm", find_minimum=True)]
        d = precision.decide(call(1024, 3), hist, 1024)
        self.assertEqual((d["precision_state"], d["precision_next"]), ("high", 1024))
        self.assertIn("healthy at full precision where 512 bits were not", d["precision_note"])

    def test_aborted_confirmed(self):
        """Aborted at 512 and again at full (no gap): locked (the run ends there)."""
        hist = self.LOW + [call(512, 7, "confirm", verdict="aborted", gap=None)]
        d = precision.decide(call(1024, 7, verdict="aborted", gap=None), hist, 1024)
        self.assertEqual(d["precision_state"], "locked")

    # a call at 512 failing the check, the call again at full (healthy: high), one more healthy call at full
    UP = [call(512, 3, "confirm", find_minimum=True), call(1024, 3, "high")]
    HIGH = LOW + UP + [call(1024, 3, "high")]

    def test_back_after_three(self):
        """The confirming call and two more healthy calls at full: back to 512; after one more only: still high."""
        self.assertEqual(precision.decide(call(1024, 3), self.HIGH[:-1], 1024)["precision_state"], "high")
        d = precision.decide(call(1024, 3), self.HIGH, 1024)
        self.assertEqual((d["precision_state"], d["precision_next"]), ("low", 512))
        self.assertIn("back to 512 bits (return 1 of 2)", d["precision_note"])

    def test_climbing_neither_counts_nor_breaks(self):
        hist = self.LOW + self.UP + [call(1024, 30, "high", climbed=True)]
        self.assertEqual(precision.decide(call(1024, 3), hist, 1024)["precision_state"], "high")
        self.assertEqual(precision.decide(call(1024, 3), hist + [call(1024, 3, "high")], 1024)["precision_state"],
                         "low")

    def test_unhealthy_starts_again(self):
        """A call at full with 9 iterations (over max(4, 2 x 3)) breaks the run of healthy calls."""
        hist = self.HIGH + [call(1024, 9, "high")]
        for more, state in ((0, "high"), (1, "high"), (2, "low")):
            self.assertEqual(precision.decide(call(1024, 3), hist + [call(1024, 3, "high")] * more, 1024)
                             ["precision_state"], state)

    def test_settings(self):
        """After 1 healthy call, 20 digits above 30: 512 bits at gap 1e-12 (52 digits), 768 at 1e-30 (512: 41)."""
        o = {"precision_return_after": 1, "precision_return_margin": 20}
        hist = self.LOW + self.UP
        self.assertEqual(precision.decide(call(1024, 3), hist, 1024, o)["precision_next"], 512)
        self.assertEqual(precision.decide(call(1024, 3, gap="1e-30"), hist, 1024, o)["precision_next"], 768)

    def test_limit_locks(self):
        """Gone back down twice (the limit): locked at the next call at full."""
        twice = self.HIGH + [call(1024, 3, "low")] + [call(512, 3, "low")] * 5 + self.UP + \
            [call(1024, 3, "high"), call(1024, 3, "low")] + [call(512, 3, "low")] * 5
        self.assertEqual(len(precision.returns(twice, 1024)), 2)
        d = precision.decide(call(1024, 3), twice + self.UP, 1024)
        self.assertEqual(d["precision_state"], "locked")
        self.assertIn("the limit (2)", d["precision_note"])

    def test_recent_trigger_locks(self):
        """A failed check within 5 calls of going back down: locked at once, no confirming call; after 5, confirm."""
        back = self.HIGH + [call(1024, 3, "low")]
        d = precision.decide(call(512, 3, find_minimum=True), back + [call(512, 3, "low")], 1024)
        self.assertEqual(d["precision_state"], "locked")
        self.assertIn("calls after going back down", d["precision_note"])
        later = back + [call(512, 3, "low")] * 5
        self.assertEqual(precision.decide(call(512, 3, find_minimum=True), later, 1024)["precision_state"], "confirm")

    def test_optimal_locks(self):
        d = precision.decide({**call(512, 1), "verdict": "optimal", "gap": "1e-31"}, self.LOW, 1024)
        self.assertEqual((d["precision_state"], d["precision_next"]), ("locked", 1024))
        self.assertIn("optimal at 512 bits", d["precision_note"])
        self.assertEqual(precision.decide(call(1024, 3), self.LOW + [call(1024, 1, "locked")], 1024),
                         {"precision_state": "locked", "precision_next": 1024})

    def test_old_records(self):
        """Records from before dynamic precision (a mixed run: step 2 sent it to full, steps 3-6 at full):
        high from step 3 on; four healthy calls at full: back to 512."""
        old = [call(512, 370, first=True), {**call(512, 10), "precision_next": 1024}, call(1024, 8),
               call(1024, 3), call(1024, 3)]
        old[2]["precision_next"] = old[3]["precision_next"] = old[4]["precision_next"] = 1024
        self.assertEqual([precision.state_of(l, 1024) for l in old], ["low", "high", "high", "high", "high"])
        d = precision.decide(call(1024, 3, gap="1.9e-15"), old, 1024)
        self.assertEqual((d["precision_state"], d["precision_next"]), ("low", 512))


def stepped(it, climbed=False, find_minimum=False, first=False):
    line = {"verdict": "stepped", "precision": 384, "gap": "1e-12", "climbed": climbed,
            "find_minimum": find_minimum, "call_iterations": it}
    if first:
        line["first_call"] = True
    return line


class Health(unittest.TestCase):
    HISTORY = [stepped(180, first=True), stepped(2), stepped(3), stepped(9, climbed=True), stepped(2), stepped(3)]

    def test_iterations(self):
        """The median of the last non-climbing, not cold calls (2, 3, 2, 3: 2.5) times 2 is 5; climbing calls
        and the cold start do not count."""
        self.assertIsNone(precision.health(stepped(5), self.HISTORY))
        self.assertIn("6 iterations, more than 5", precision.health(stepped(6), self.HISTORY))
        self.assertIsNone(precision.health(stepped(9, climbed=True), self.HISTORY))

    def test_at_least_four(self):
        hist = [stepped(1), stepped(1), stepped(1)]
        self.assertIsNone(precision.health(stepped(4), hist))
        self.assertIsNotNone(precision.health(stepped(5), hist))

    def test_no_history(self):
        self.assertIsNone(precision.health(stepped(50), [stepped(180, first=True)]))

    def test_find_minimum_leaving(self):
        """False to True: the path left the allowed region."""
        self.assertIn("left the allowed region", precision.health(stepped(2, find_minimum=True), self.HISTORY))

    def test_find_minimum_entering(self):
        """True to False: entering the region from a start outside it (the second step of such a run), even after several steps
        outside."""
        outside = [stepped(180, first=True, find_minimum=True), stepped(3, find_minimum=True)]
        self.assertIsNone(precision.health(stepped(3), outside))

    def test_aborted(self):
        aborted = {"verdict": "aborted", "precision": 384, "gap": None, "call_iterations": 7}
        self.assertIn("aborted at 384 bits", precision.health(aborted, self.HISTORY))


class Checkpoint(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())

    def test_limbs(self):
        """Elemental's num_limbs: 785 bits are 15 limbs (132 bytes a number, as in a real checkpoint)."""
        self.assertEqual([precision.limbs(b) for b in (384, 512, 768, 785, 1024)], [8, 10, 14, 15, 18])

    def test_up(self):
        """8 -> 15 limbs (384 -> 785 bits): the same numbers."""
        src, up = self.d / "a", self.d / "b"
        src.write_bytes(rank_file(VALUES * 2, 8))
        precision.convert_rank_file(src, up, 8, 15)
        self.assertEqual(len(up.read_bytes()), 2 * 16 + 12 * (12 + 8 * 15))
        for k, v in enumerate(VALUES * 2):
            got = value(up.read_bytes(), 16 * (1 + k // 6) + k * (12 + 120))
            self.assertLessEqual(abs(got - v), abs(v) / Fraction(B) ** 6)
            self.assertEqual(got, value(src.read_bytes(), 16 * (1 + k // 6) + k * (12 + 64)))

    def test_down(self):
        """15 -> 8 limbs (785 -> 384 bits): each number cut to its 8 most significant limbs, within 2^-448 of it
        relative; back up at 15 limbs the same as cut."""
        src, down, up = self.d / "a", self.d / "b", self.d / "c"
        src.write_bytes(rank_file(VALUES, 15))
        precision.convert_rank_file(src, down, 15, 8)
        precision.convert_rank_file(down, up, 8, 15)
        for k, v in enumerate(VALUES):
            got = value(down.read_bytes(), 16 + k * (12 + 64))
            self.assertLessEqual(abs(got - v), abs(v) / Fraction(B) ** 7)
            self.assertEqual(got, value(up.read_bytes(), 16 + k * (12 + 120)))

    def test_real_checkpoint_down(self):
        """A rank file of a real 785-bit generation (tests/fixtures/checkpoint-785: Ising Lambda 11 at
        full precision, kept/4) to 512 bits and back (15 -> 10 -> 15 limbs): every number's exponent, sign and its
        min(|size|, 10) most significant limbs byte for byte, zeros after them, and the files read to their end."""
        a = FIXTURE_785.read_bytes()
        down, up = self.d / "down", self.d / "up"
        precision.convert_rank_file(FIXTURE_785, down, 15, 10)
        precision.convert_rank_file(down, up, 10, 15)
        b = up.read_bytes()
        self.assertEqual(len(a), len(b))
        i = numbers = cut = 0
        while i < len(a):
            self.assertEqual(a[i:i + 16], b[i:i + 16])
            h, w = struct.unpack_from("<qq", a, i)
            i += 16
            for _ in range(h * w):
                s = struct.unpack_from("<i", a, i)[0]
                n, m = abs(s), min(abs(s), 10)
                cut += n > 10
                self.assertEqual(struct.unpack_from("<i", b, i)[0], m if s >= 0 else -m)
                self.assertEqual(a[i + 4:i + 12], b[i + 4:i + 12])
                self.assertEqual(a[i + 12 + 8 * (n - m):i + 12 + 8 * n], b[i + 12:i + 12 + 8 * m])
                self.assertEqual(b[i + 12 + 8 * m:i + 12 + 120], bytes(8 * (15 - m)))
                i, numbers = i + 12 + 120, numbers + 1
        self.assertEqual(i, len(a))
        self.assertGreater(cut, numbers // 2)      # most numbers had more than 10 limbs: the cut is exercised

    def test_real_checkpoint(self):
        """A rank file of a real 384-bit generation (tests/fixtures/checkpoint-384: Ising Lambda 11 at
        gap 1e-20) raised to 1024 bits (8 -> 18 limbs): every number's size, exponent and |size| limbs byte for
        byte, zeros after them, and the file read to its end; back at 8 limbs it is the original's meaningful
        bytes."""
        src = FIXTURE
        up = self.d / "up"
        precision.convert_rank_file(src, up, 8, 18)
        a, b = src.read_bytes(), up.read_bytes()
        i = j = numbers = 0
        while i < len(a):
            self.assertEqual(a[i:i + 16], b[j:j + 16])
            h, w = struct.unpack_from("<qq", a, i)
            i, j = i + 16, j + 16
            for _ in range(h * w):
                n = abs(struct.unpack_from("<i", a, i)[0])
                self.assertLessEqual(n, 8)
                self.assertEqual(a[i:i + 12 + 8 * n], b[j:j + 12 + 8 * n])
                self.assertEqual(b[j + 12 + 8 * n:j + 12 + 8 * 18], bytes(8 * (18 - n)))
                i, j, numbers = i + 12 + 64, j + 12 + 144, numbers + 1
        self.assertEqual((i, j), (len(a), len(b)))
        self.assertEqual(numbers, 664)          # 6 blocks: 6 x 16 + 664 x 76 = 50560 bytes, the file's size

    def test_not_that_precision(self):
        src = self.d / "a"
        src.write_bytes(rank_file(VALUES, 15))
        with self.assertRaises(SystemExit):
            precision.convert_rank_file(src, self.d / "b", 14, 8)

    def test_generation(self):
        """Generation 3 at 384 bits written at 785 as generation 4, checkpoint.json naming 4 with backup 3; a
        kept/ hard link of generation 3 untouched; an old file named like generation 4 replaced, not written into."""
        ck = self.d / "ck"
        (ck / "kept" / "2").mkdir(parents=True)
        lay = {"nodes": ["h-1"], "ppn": 2}
        for r in range(2):
            (ck / f"checkpoint_3_{r}").write_bytes(rank_file(VALUES, 8))
            os.link(ck / f"checkpoint_3_{r}", ck / "kept" / "2" / f"checkpoint_3_{r}")
        os.link(ck / "checkpoint_3_0", ck / "checkpoint_4_0")      # a name that is a link of kept data
        (ck / "checkpoint.json").write_text(json.dumps({"current": 3, "backup": 2, "version": "v", "options": {}}))
        before = (ck / "kept" / "2" / "checkpoint_3_0").read_bytes()
        self.assertEqual(precision.convert_checkpoint(ck, lay, 384, 785), 4)
        meta = json.loads((ck / "checkpoint.json").read_text())
        self.assertEqual((meta["current"], meta["backup"], meta["version"]), (4, 3, "v"))
        self.assertEqual((ck / "kept" / "2" / "checkpoint_3_0").read_bytes(), before)
        self.assertEqual(len((ck / "checkpoint_4_1").read_bytes()), 16 + 6 * (12 + 120))


if __name__ == "__main__":
    unittest.main()
