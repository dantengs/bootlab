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
