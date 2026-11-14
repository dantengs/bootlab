"""Mixed precision for a Skydive run ([skydive] mixed_precision = true): the solver and pmp2sdp at the lowest precision of a ladder that keeps enough digits, raised to the
problem's own precision (Problem.solver_precision) when a health check fails and for the end of the run. Never
lowered again.

The digits kept: about dec(P) - LOSS - 0.6 log10(1/gap), dec(P) = 0.30103 P. The loss measured: 78 (Lambda 23)
and 83 (Lambda 27) digits with infinitesimal 1e-40; 68 with bootlab's 1e-20 (Ising
Lambda 11: 384 bits agreed with 785 and with bt r3.8 to 47.5 digits at gap 0.08); at gap 6.8e-21
the external steps at 384 and 512 bits agreed with 785 to 40.9 and 80.2 digits, N to 49 and 88: a loss of 62 by this
formula; on a larger problem (N about 2300, 1024 bits full; gap 1.9e-14) 512 bits agreed with 1024 to 54.6 digits, a loss of 91. LOSS is the
largest measured, 91, with some room: 95. The rule asks for MARGIN digits or more: 512 bits keep about 41 at a gap of
1e-30, 384 bits are never chosen.

The solver's binary checkpoint stores each number in a size fixed by the precision (Elemental's
BigFloat::Serialize: int32 size, int64 exponent, then num_limbs 64-bit limbs, least significant first, the first
|size| meaningful), and does not store the precision. dynamical_sdp has no way to read one written at another
precision; its one precision-free route is a text checkpoint (writeSolution x,y,X,Y read back with
initialCheckpointDir), which would have every call below full precision write X and Y as text, gathered from every
node, for the call that might follow at full. So before a call at a higher precision than its checkpoint's,
convert_checkpoint rewrites the checkpoint's generation at the higher precision as the next generation, the limbs
padded with zeros: the same numbers. Only up: a call is never below its checkpoint's precision.
"""
import json
import math
import os
import statistics
import struct
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import jump

LADDER = (384, 512, 768, 1024)
MARGIN = 30
LOSS = 95


def digits_kept(bits: int, gap) -> float:
    return 0.30103 * bits - LOSS - 0.6 * math.log10(1 / float(gap))


def choose(full: int, gap) -> int:
    """The lowest precision of the ladder (below full) that keeps MARGIN digits at this gap, else full."""
    for p in [p for p in LADDER if p < full]:
        if digits_kept(p, gap) >= MARGIN:
            return p
    return full


def health(line: dict, history: list):
    """Why the call of record `line` (after the records `history`) asks for full precision, or None: it was
    aborted; findMinimumQ is True after a stepped record where it was False (the path left the allowed region: from
    False to True; the other way, entering the region, is a Skydive run's usual path); or a call that
    neither climbed, started cold nor
    started a large step ahead (jump.after_jump) made more than max(4, 2 x the median of the last 5 such stepped
    calls) iterations (no check without such calls). The navigator's sign is not checked (it changes near the
    boundary), climbing calls are not, nor the recovery after a large step (its own budget)."""
    if line["verdict"] == "aborted":
        return f"aborted at {line['precision']} bits"
    stepped = [l for l in history if l["verdict"] == "stepped"]
    if stepped and not stepped[-1].get("find_minimum") and line["find_minimum"]:
        return "findMinimumQ changed to True (left the allowed region)"
    usual = [l["call_iterations"] for k, l in enumerate(history) if l["verdict"] == "stepped" and not l.get("climbed")
             and not l.get("first_call") and not jump.after_jump(history, k) and l.get("call_iterations") is not None][-5:]
    it = line.get("call_iterations")
    if usual and it is not None and not line["climbed"] and not line.get("first_call")             and not jump.after_jump(history, len(history)):
        limit = max(4, 2 * statistics.median(usual))
        if it > limit:
            return f"{it} iterations, more than {limit:g} (2 x the median of the last {len(usual)} such calls)"
    return None


def next_precision(line: dict, history: list, full: int) -> tuple:
    """(precision of the next call, note) after the call of record `line`: full from full; full after a failed
    health check, or after an optimal call below full (the run's end is decided at full precision); else the
    ladder's choice at this gap, never below this call's."""
    bits = line["precision"]
    if bits >= full:
        return full, None
    why = health(line, history)
    if why:
        return full, f"{why}: full precision from the next call on"
    if line["verdict"] == "optimal":
        return full, f"optimal at {bits} bits: solved again at full precision"
    want = choose(full, line["gap"])
    if want > bits:
        return want, f"gap {line['gap']}: {bits} bits keep {digits_kept(bits, line['gap']):.0f} digits, below {MARGIN}"
    return bits, None


# ---------------------------------------------------------------- the checkpoint at another precision

def limbs(bits: int) -> int:
    """Elemental's gmp::num_limbs for a precision (SetPrecision, as GMP's mpf_set_prec)."""
    return (max(bits, 53) + 127) // 64 + 1


def convert_rank_file(src: Path, dst: Path, n_from: int, n_to: int):
    """One rank's checkpoint file at n_to >= n_from limbs a number: blocks of (int64 height, int64 width, height x
    width numbers); a number keeps its size, exponent and its |size| limbs, zeros after them (where the solver leaves
    whatever was there). The file must end at a block's end and no number may have more than n_from limbs: else it
    is not of that precision."""
    if n_to < n_from:
        raise SystemExit(f"{src}: a checkpoint is only raised in precision, not from {n_from} to {n_to} limbs")
    b = src.read_bytes()
    rec = 12 + 8 * n_from
    out = bytearray()
    zeros = bytes(8 * n_to)
    i = 0
    while i < len(b):
        h, w = struct.unpack_from("<qq", b, i)
        out += b[i:i + 16]
        i += 16
        end = i + h * w * rec
        if end > len(b):
            raise SystemExit(f"{src}: a block of {h} x {w} numbers runs past the file's end at {n_from} limbs a "
                             f"number: not a checkpoint of that precision")
        while i < end:
            s = struct.unpack_from("<i", b, i)[0]
            a = abs(s)
            if a > n_from:
                raise SystemExit(f"{src}: a number with {a} limbs in a checkpoint of {n_from}")
            out += b[i:i + 12 + 8 * a] + zeros[:8 * (n_to - a)]
            i += rec
    dst.unlink(missing_ok=True)       # a new file: the old name may be a hard link of kept/ (execute.keep_generation)
    dst.write_bytes(out)


def _convert(args):
    convert_rank_file(*args)


def convert_checkpoint(ck: Path, lay: dict, bits_from: int, bits_to: int) -> int:
    """The generation g that checkpoint/checkpoint.json names, at bits_from, written at bits_to as generation g + 1
    (one file a rank, lay's nodes x ppn ranks), and checkpoint.json set to it with backup g (the solver removes g's
    names at its first save; kept/ keeps g's data). Returns g + 1."""
    from .execute import rank_files
    meta = json.loads((ck / "checkpoint.json").read_text())
    g = meta["current"]
    jobs = [(ck / a, ck / b, limbs(bits_from), limbs(bits_to)) for a, b in zip(rank_files(lay, g), rank_files(lay, g + 1))]
    with ProcessPoolExecutor(max_workers=min(len(jobs), os.cpu_count() or 1, 32)) as pool:
        list(pool.map(_convert, jobs))
    meta.update({"current": g + 1, "backup": g})
    tmp = ck / ".checkpoint.json.precision"
    tmp.write_text(json.dumps(meta, indent=4) + "\n")
    os.replace(tmp, ck / "checkpoint.json")
    return g + 1
