"""Exact numbers: decimal strings in, exact decimal strings out, and small rational expressions.

Every number that reaches a program is written as the exact decimal it stands for. bt reads JSON numbers
as exact decimals (Projects.ParamsFile.Dec), so a JSON number must be written as its digits, never as a
Python float: `dumps` writes a `Num` as a bare JSON number.
"""
import json
import re
from fractions import Fraction


def frac(text) -> Fraction:
    """An exact number from a decimal string such as "0.518149", "1e-40" or "3/10"."""
    return Fraction(str(text).strip())


def decimal_text(q: Fraction) -> str:
    """q as an exact decimal; a q with no finite decimal expansion is an error."""
    d = q.denominator
    twos = fives = 0
    while d % 2 == 0:
        d //= 2
        twos += 1
    while d % 5 == 0:
        d //= 5
        fives += 1
    if d != 1:
        raise ValueError(f"{q} has no finite decimal expansion; write it as a finite decimal")
    k = max(twos, fives)
    n = q.numerator * 10 ** k // q.denominator
    digits = str(abs(n)).rjust(k + 1, "0")
    if k:
        digits = digits[:-k] + "." + digits[-k:]
    return ("-" if n < 0 else "") + digits


class Num(str):
    """A decimal string to be written as a JSON number."""


def dumps(obj, indent=1) -> str:
    """json.dumps that writes each Num as a bare number with all its digits."""
    raw = {}

    def mark(x):
        if isinstance(x, Num):
            key = f"\u0000num{len(raw)}\u0000"
            raw[key] = str(x)
            return key
        if isinstance(x, dict):
            return {k: mark(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [mark(v) for v in x]
        return x

    text = json.dumps(mark(obj), indent=indent, ensure_ascii=False)
    for key, digits in raw.items():
        text = text.replace(json.dumps(key), digits)
    return text


# ---------------------------------------------------------------- expressions

_TOKEN = re.compile(r"\s*(?:(?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)|(?P<name>[A-Za-z_][A-Za-z0-9_]*)|(?P<op>[-+*/()]))")


class Expr:
    """A rational expression of numbers and parameter names with + - * / and brackets."""

    def __init__(self, text):
        self.text = str(text)
        self.names = []
        self._tokens = self._tokenize(self.text)
        self._pos = 0
        self._tree = self._sum()
        if self._pos != len(self._tokens):
            raise ValueError(f"cannot read {self.text!r}")
        del self._tokens

    @staticmethod
    def _tokenize(text):
        tokens, i = [], 0
        while i < len(text):
            if text[i:].strip() == "":
                break
            m = _TOKEN.match(text, i)
            if not m:
                raise ValueError(f"cannot read {text!r} at {text[i:]!r}")
            tokens.append((m.lastgroup, m.group(m.lastgroup)))
            i = m.end()
        return tokens

    def _peek(self):
        return self._tokens[self._pos] if self._pos < len(self._tokens) else (None, None)

    def _take(self):
        t = self._peek()
        self._pos += 1
        return t

    def _sum(self):
        node = self._product()
        while self._peek() in (("op", "+"), ("op", "-")):
            op = self._take()[1]
            node = (op, node, self._product())
        return node

    def _product(self):
        node = self._unary()
        while self._peek() in (("op", "*"), ("op", "/")):
            op = self._take()[1]
            node = (op, node, self._unary())
        return node

    def _unary(self):
        if self._peek() == ("op", "-"):
            self._take()
            return ("neg", self._unary())
        if self._peek() == ("op", "+"):
            self._take()
            return self._unary()
        kind, value = self._take()
        if kind == "num":
            return ("num", Fraction(value))
        if kind == "name":
            if value not in self.names:
                self.names.append(value)
            return ("name", value)
        if (kind, value) == ("op", "("):
            node = self._sum()
            if self._take() != ("op", ")"):
                raise ValueError(f"unbalanced brackets in {self.text!r}")
            return node
        raise ValueError(f"cannot read {self.text!r}")

    def value(self, env) -> Fraction:
        def ev(n):
            tag = n[0]
            if tag == "num":
                return n[1]
            if tag == "name":
                return env[n[1]]
            if tag == "neg":
                return -ev(n[1])
            a, b = ev(n[1]), ev(n[2])
            return {"+": a + b, "-": a - b, "*": a * b, "/": a / b}[tag]
        return ev(self._tree)
