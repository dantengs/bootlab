"""Mixed precision for a Skydive run ([skydive] mixed_precision = true): the solver and pmp2sdp at the lowest precision of a ladder that keeps enough digits, raised to the
problem's own precision (Problem.solver_precision) when a health check fails and for the end of the run, and lowered
again under conditions (decide: a failed check is first confirmed at full precision; after some healthy calls at full
a run goes back down, a limited number of times, unless it can no longer).

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
node, for the call that might follow at full. So before a call at another precision than its checkpoint's,
convert_checkpoint rewrites the checkpoint's generation at the call's precision as the next generation: up, the limbs
padded with zeros (the same numbers); down, each number cut to its most significant limbs (GMP's mpf truncates the
same way; the error, 2^-64 (limbs - 1) relative, is 2^-576 at 512 bits, far below the 41 or so digits a call there
keeps).
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
# Going back down from full precision ([skydive] precision_return_after, precision_return_limit,
# precision_return_margin): starting values, not measured.
RETURN_AFTER = 3       # healthy calls at full precision in a row (climbing calls neither count nor break the run)
RETURN_LIMIT = 2       # returns a run may make
RETURN_MARGIN = 10     # digits above MARGIN a return asks for at the gap of the last call
RECENT = 5             # a failed check within this many calls after a return locks at full at once


def digits_kept(bits: int, gap) -> float:
    return 0.30103 * bits - LOSS - 0.6 * math.log10(1 / float(gap))


def choose(full: int, gap, margin: float = MARGIN) -> int:
    """The lowest precision of the ladder (below full) that keeps `margin` digits at this gap, else full."""
    for p in [p for p in LADDER if p < full]:
        if digits_kept(p, gap) >= margin:
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


# ---------------------------------------------------------------- going up and down
#
# A record's precision_state is the state the next call is made in: low (below full, the ladder's choice), confirm
# (the call just made below full failed the health check: the same call again at full precision, from the generation
# before it), high (full precision, which may return), locked (full precision to the end). Everything is worked out
# from the records, so a resume needs nothing else.

def state_of(rec: dict, full: int) -> str:
    """The state a record left for the next call; a record from before dynamic precision: low when its
    precision_next is below full, else high."""
    if "precision_state" in rec:
        return rec["precision_state"]
    return "low" if rec.get("precision_next", full) < full else "high"


def live(history: list) -> list:
    """The records the path is made of: a record in state confirm is done again by the next."""
    return [l for l in history if l.get("precision_state") != "confirm"]


def returns(history: list, full: int) -> list:
    """The indices in history of the records after which the run went back down from full precision."""
    return [k for k in range(1, len(history))
            if state_of(history[k], full) == "low" and state_of(history[k - 1], full) == "high"]


def healthy_in_a_row(calls: list, full: int) -> int:
    """How many calls made at full precision in states confirm or high, ending with the last of `calls`, passed
    the health check in a row; climbing calls neither count nor break the run."""
    count = 0
    for j, rec in enumerate(calls):
        if not j or state_of(calls[j - 1], full) not in ("confirm", "high"):
            count = 0
        elif not rec.get("climbed"):
            count = count + 1 if health(rec, live(calls[:j])) is None else 0
    return count


def decide(line: dict, history: list, full: int, opts: dict = None) -> dict:
    """precision_state, precision_next and (when either changes) precision_note after the call of record `line`;
    history: the run's records before it; opts: [skydive] (precision_return_after, _limit, _margin).
    - low: a failed health check asks for confirm, or for locked within RECENT calls of a return; an optimal call is
      solved again at full precision (locked: the end); the ladder's choice at this gap otherwise, locked when that
      is full.
    - confirm (this call was the one before, again at full): it fails the check too: the path's own behaviour, low
      again at the ladder's choice; it passes: high.
    - high: locked when the run has returned precision_return_limit times or no lower precision keeps MARGIN +
      precision_return_margin digits at this gap; low at that precision after precision_return_after healthy calls
      in a row.
    - locked stays."""
    o = opts or {}
    after = o.get("precision_return_after", RETURN_AFTER)
    limit = o.get("precision_return_limit", RETURN_LIMIT)
    extra = o.get("precision_return_margin", RETURN_MARGIN)
    bits, gap = line["precision"], line.get("gap")
    state = state_of(history[-1], full) if history else ("low" if bits < full else "high")

    def out(s, p, note=None):
        return {"precision_state": s, "precision_next": p, **({"precision_note": note} if note else {})}

    if state == "locked":
        return out("locked", full)
    path = live(history)
    if state == "low":
        if line["verdict"] == "optimal" and bits < full:
            return out("locked", full, f"optimal at {bits} bits: solved again at full precision, full to the end")
        why = health(line, path)
        if why:
            back = returns(history, full)
            if back and len(history) - back[-1] <= RECENT:
                return out("locked", full, f"{why}, {len(history) - back[-1]} calls after going back down to {bits} "
                                           f"bits: locked at full precision")
            return out("confirm", full, f"{why}: the same call again at full precision, from the generation before it")
        want = choose(full, gap)
        if want >= full:
            return out("locked", full, f"gap {gap}: {bits} bits keep {digits_kept(bits, gap):.0f} digits, below "
                                       f"{MARGIN}, and no lower precision does: locked at full precision")
        if want > bits:
            return out("low", want, f"gap {gap}: {bits} bits keep {digits_kept(bits, gap):.0f} digits, below {MARGIN}")
        return out("low", bits)
    if state == "confirm":
        low = history[-1]
        why = health(line, path)
        if not why:
            return out("high", full, f"healthy at full precision where {low['precision']} bits were not: the "
                                     f"precision's; full until {after} healthy calls in a row")
        if gap is None:
            return out("locked", full, f"confirmed at full precision ({why}): the path's own, not the precision's")
        want = choose(full, gap)
        if want >= full:
            return out("locked", full, f"confirmed at full precision ({why}): the path's own, not the precision's; "
                                       f"no lower precision keeps {MARGIN} digits at gap {gap}: locked at full")
        return out("low", want, f"confirmed at full precision ({why}): the path's own, not the precision's: {want} bits "
                                f"again")
    # high
    back = returns(history, full)
    if len(back) >= limit:
        return out("locked", full, f"gone back down {len(back)} times, the limit ({limit}): locked at full precision")
    if gap is None:
        return out("high", full)
    to = choose(full, gap, MARGIN + extra)
    if to >= full:
        return out("locked", full, f"gap {gap}: no lower precision keeps {MARGIN} + {extra} digits: locked at full "
                                   f"precision")
    count = healthy_in_a_row(history + [line], full)
    if count >= after:
        return out("low", to, f"{count} healthy calls at full precision in a row; gap {gap}: {to} bits keep "
                              f"{digits_kept(to, gap):.0f} digits, at least {MARGIN} + {extra}: back to {to} bits "
                              f"(return {len(back) + 1} of {limit})")
    return out("high", full)


# ---------------------------------------------------------------- the checkpoint at another precision

def limbs(bits: int) -> int:
    """Elemental's gmp::num_limbs for a precision (SetPrecision, as GMP's mpf_set_prec)."""
    return (max(bits, 53) + 127) // 64 + 1


def convert_rank_file(src: Path, dst: Path, n_from: int, n_to: int):
    """One rank's checkpoint file at n_to limbs a number: blocks of (int64 height, int64 width, height x width
    numbers); a number keeps its exponent and its |size| limbs, zeros after them (where the solver leaves whatever
    was there); with more than n_to limbs (n_to < n_from) only its n_to most significant, its size n_to with its
    sign (truncated, as GMP's mpf). The file must end at a block's end and no number may have more than n_from
    limbs: else it is not of that precision."""
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
            if a > n_to:
                out += struct.pack("<i", n_to if s > 0 else -n_to) + b[i + 4:i + 12] \
                    + b[i + 12 + 8 * (a - n_to):i + 12 + 8 * a]
            else:
                out += b[i:i + 12 + 8 * a] + zeros[:8 * (n_to - a)]
            i += rec
    dst.unlink(missing_ok=True)       # a new file: the old name may be a hard link of kept/ (execute.keep_generation)
    dst.write_bytes(out)


def _convert(args):
    convert_rank_file(*args)


def convert_checkpoint(ck: Path, lay: dict, bits_from: int, bits_to: int) -> int:
    """The generation g that checkpoint/checkpoint.json names, at bits_from, written at bits_to (higher or lower) as
    generation g + 1
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
