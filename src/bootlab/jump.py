"""Dynamic large steps for a Skydive run ([skydive] large_steps = true): the next point x + m dx instead of x + dx,
m >= 1 chosen from the records (rules taken unchanged from earlier research runs).

An external step's length is set by the gap (about sqrt(gap)), not by how far the boundary is; on a straight stretch
of the path the solver's full step dx / alpha points the same way call after call, and a point m steps ahead
recovers in about 1.2 more iterations a step (measured), against a fixed cost of a whole call. So:

  before: c = cos(full step, the previous stepped call's full step), full step = dx / alpha_d;
     m = 1 when cooling down, c <= C1 or the call climbed; 2 when c <= C2; else floor(F / alpha_d) within [1, M];
  after a call that started m_prev > 1 steps ahead:
     findMinimumQ false to true (left the allowed region) -> the run stops; true to false (entered it, the usual
       path of a run started outside) -> m = 1 for COOL calls, as the health check of mixed precision;
     N changed sign -> no more large steps in this run; with [skydive] large_steps_sign_cooldown = k (not
       tested yet), m = 1 for the next k calls only;
     more than 3 + 1.5 (m_prev - 1) iterations, or c < COS_ACCEPT -> the point kept, m = 1 for COOL calls.

Measured on a larger problem (N about 3000) at 1024 bits, 13 h: 1.84 times the progress of the same run without;
at 512 bits, 13 h: 1.32 times, N changing sign once at record 27 ended its large steps there. Every point is
still a point the solver converged at: a bad large step costs time, not correctness. M = 8 as in those runs (a
replay with up to 20 steps at once left the allowed region).
"""
import math

F = 0.5
M = 8
C1 = 0.99
C2 = 0.999
COS_ACCEPT = 0.98
COOL = 2
FOREVER = 10 ** 9


def full_step(line: dict) -> list:
    a = float(line["alpha_d"])
    return [float(s) / a for s in line["ext_step"]]


def cosine(u: list, v: list) -> float:
    return sum(a * b for a, b in zip(u, v)) / math.sqrt(sum(a * a for a in u) * sum(b * b for b in v))


def decide(line: dict, history: list, sign_cooldown: int = FOREVER) -> dict:
    """The large step after the stepped call of record `line` (with call_iterations), the records `history` before
    it: {jump: m, jump_cooldown, jump_note, and jump_stop (why the run stops) when it does}. sign_cooldown: the calls
    at m = 1 after N changed sign following a large step (FOREVER: the research rule, no more large steps)."""
    stepped = [l for l in history if l["verdict"] == "stepped"]
    if not stepped:
        return {"jump": 1, "jump_cooldown": 0, "jump_note": "first step"}
    prev = stepped[-1]
    c = cosine(full_step(line), full_step(prev))
    cool = max(prev.get("jump_cooldown", 0) - 1, 0)
    note = f"cos {c:.8f}"
    out = {}
    m_prev = prev.get("jump", 1)
    if m_prev > 1:
        it = line.get("call_iterations")
        if line["find_minimum"] and not prev["find_minimum"]:
            out["jump_stop"] = f"findMinimumQ changed to true after a step {m_prev} times ahead (left the allowed region)"
        elif prev["find_minimum"] and not line["find_minimum"]:
            # entering the allowed region, the usual path of a run started outside; N changes sign
            # with it, which is that, not a step past the boundary
            cool = COOL
            note += f"; entered the allowed region (findMinimumQ true to false) after a step {m_prev} times ahead: " \
                    f"m = 1 for {COOL} calls"
        elif (float(line["navigator"]) < 0) != (float(prev["navigator"]) < 0):
            cool = sign_cooldown
            note += f"; N changed sign after a step {m_prev} times ahead: " + (
                "no more large steps" if sign_cooldown == FOREVER else f"m = 1 for {sign_cooldown} calls")
        elif it is not None and it > 3 + 1.5 * (m_prev - 1):
            cool = COOL
            note += f"; {it} iterations after a step {m_prev} times ahead, over {3 + 1.5 * (m_prev - 1):g}"
        elif c < COS_ACCEPT:
            cool = COOL
            note += f"; the direction changed after a step {m_prev} times ahead"
    if cool > 0 or c <= C1 or line["climbed"]:
        m = 1
    elif c <= C2:
        m = 2
    else:
        m = min(max(math.floor(F / float(line["alpha_d"])), 1), M)
    return {"jump": m, "jump_cooldown": cool, "jump_note": note, **out}


def after_jump(history: list, k: int) -> bool:
    """Record k of history (all records) started more than one step ahead: the stepped record before it had jump > 1."""
    before = [l for l in history[:k] if l["verdict"] == "stepped"]
    return bool(before) and before[-1].get("jump", 1) > 1
