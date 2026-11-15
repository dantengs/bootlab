"""Dynamic large steps (bootlab.jump): the rules. The loop with them: test_skydive.Loop."""
import unittest

from bootlab import jump


def rec(step=["1e-4", "0", "0"], alpha="0.05", it=3, climbed=False, fm=False, n="-1e-3", **kw):
    return {"verdict": "stepped", "ext_step": step, "alpha_d": alpha, "call_iterations": it, "climbed": climbed,
            "find_minimum": fm, "navigator": n, **kw}


class Rules(unittest.TestCase):
    def test_first(self):
        self.assertEqual(jump.decide(rec(), [])["jump"], 1)

    def test_straight(self):
        """cos 1: floor(0.5 / alpha) steps, at most 8, at least 1."""
        self.assertEqual(jump.decide(rec(alpha="0.1"), [rec()])["jump"], 5)
        self.assertEqual(jump.decide(rec(alpha="0.01"), [rec()])["jump"], 8)
        self.assertEqual(jump.decide(rec(alpha="0.9"), [rec()])["jump"], 1)

    def test_turning(self):
        """cos 0.995: 2 steps; cos 0.95: 1; a climbing call: 1. The full step is dx / alpha."""
        bent = ["1e-4", str(1e-4 * 0.1005), "0"]           # cos about 0.995
        self.assertEqual(jump.decide(rec(step=bent), [rec()])["jump"], 2)
        self.assertEqual(jump.decide(rec(step=["1e-4", "3.3e-5", "0"]), [rec()])["jump"], 1)
        self.assertEqual(jump.decide(rec(climbed=True), [rec()])["jump"], 1)
        self.assertEqual(jump.decide(rec(step=["2e-4", "0", "0"], alpha="0.1"), [rec()])["jump"], 5)

    def test_after_a_large_step(self):
        before = [rec(jump=4)]
        stop = jump.decide(rec(fm=True), before)
        self.assertIn("findMinimumQ changed", stop["jump_stop"])
        sign = jump.decide(rec(n="2e-17"), before)
        self.assertEqual((sign["jump"], sign["jump_cooldown"]), (1, jump.FOREVER))
        over = jump.decide(rec(it=8), before)                # 3 + 1.5 x 3 = 7.5
        self.assertEqual((over["jump"], over["jump_cooldown"]), (1, 2))
        self.assertEqual(jump.decide(rec(it=7), before)["jump"], 8)
        turned = jump.decide(rec(step=["1e-4", "2.1e-5", "0"]), before)   # cos 0.979
        self.assertEqual((turned["jump"], turned["jump_cooldown"]), (1, 2))

    def test_leaving_stops(self):
        """findMinimumQ false to true after a large step: the path left the allowed region, the run stops."""
        stop = jump.decide(rec(fm=True, n="2e-17"), [rec(jump=4)])
        self.assertIn("findMinimumQ changed to true after a step 4 times ahead (left the allowed region)",
                      stop["jump_stop"])

    def test_entering_cools_down(self):
        """findMinimumQ true to false after a large step, N changing sign with it: the usual way
        into the allowed region; no stop, and not the sign rule (no large steps for ever): m = 1 for 2 calls, then
        large steps again."""
        d = jump.decide(rec(fm=False, n="-6.5e-19"), [rec(jump=2, fm=True, n="1.1e-18")])
        self.assertNotIn("jump_stop", d)
        self.assertEqual((d["jump"], d["jump_cooldown"]), (1, 2))
        self.assertIn("entered the allowed region", d["jump_note"])
        hist = [rec(jump=2, fm=True, n="1.1e-18"), rec(n="-6.5e-19", **d)]
        for left in (1, 0):
            e = jump.decide(rec(n="-6.5e-19"), hist)
            self.assertEqual(e["jump_cooldown"], left)
            hist.append(rec(n="-6.5e-19", **e))
        self.assertEqual([l["jump"] for l in hist[1:]], [1, 1, 8])

    def test_sign_cooldown(self):
        """large_steps_sign_cooldown = 3: N changing sign after a large step cools down 3 calls, then jumps again."""
        sign = jump.decide(rec(n="2e-17"), [rec(jump=4)], sign_cooldown=3)
        self.assertEqual((sign["jump"], sign["jump_cooldown"]), (1, 3))
        self.assertIn("m = 1 for 3 calls", sign["jump_note"])
        hist = [rec(jump=4), rec(n="2e-17", **sign)]
        for left in (2, 1, 0):
            d = jump.decide(rec(n="2e-17"), hist, sign_cooldown=3)
            self.assertEqual(d["jump_cooldown"], left)
            hist.append(rec(n="2e-17", **d))
        self.assertEqual([l["jump"] for l in hist[2:]], [1, 1, 8])

    def test_cooldown(self):
        self.assertEqual(jump.decide(rec(), [rec(jump=1, jump_cooldown=2)])["jump"], 1)
        self.assertEqual(jump.decide(rec(), [rec(jump=1, jump_cooldown=1)])["jump"], 8)
        forever = jump.decide(rec(), [rec(jump=1, jump_cooldown=jump.FOREVER)])
        self.assertEqual((forever["jump"], forever["jump_cooldown"]), (1, jump.FOREVER - 1))

    def test_after_jump(self):
        hist = [rec(jump=1), {"verdict": "aborted"}, rec(jump=3), {"verdict": "aborted"}]
        self.assertEqual([jump.after_jump(hist, k) for k in range(5)], [False, False, False, True, True])


if __name__ == "__main__":
    unittest.main()
