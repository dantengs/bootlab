# Writing a new model with an AI assistant

A bootlab "model" is three data files (model.json, spectrum.toml, objective.toml); a run adds run.toml. An AI
assistant (Claude, or any capable model) can write them from the crossing equations of a paper or from
simpleboot's AutoEqu output, if it is given the exact format and conventions. This guide gives those, a
complete worked example (the 3d Ising σ–ε system), a prompt template, and the checks that tell you whether the
result is right. Appendix A reports a test of the guide itself: an assistant that saw only this guide (without
section 3) translated simpleboot's Ising file into a model identical to the accepted one, PMP byte for byte.

**Read section 5 before trusting any model.** A convention error produces no error message, only wrong
physics.

[中文版](../zh/writing-models-with-ai.md)

## 1. What you need to produce

| File | Content | Format reference |
|---|---|---|
| `model.json` | crossing equations, channels, identity and external-operator terms, objectives | [interfaces 3.1](interfaces.md#31-the-model-bt-model2-json) |
| `spectrum.toml` | the gaps and isolated operators | [interfaces 3.2](interfaces.md#32-the-spectrum-spectrumtoml) |
| `objective.toml` | which objective of the model | [interfaces 3.3](interfaces.md#33-the-objective-objectivetoml) |
| `run.toml` | Λ, coordinates, start point, box, Skydive and solver options | [interfaces 4](interfaces.md#runtoml) |

Limits of v1: d = 3; every external operator is a scalar; blocks come from scalar_blocks.

## 2. The format and its conventions, compactly

**Top level**: `format` = `"bt-model/2"`, `name`, `dimension` = 3, `externals`, `rows`, `channels`,
`fixed`, `objectives`.

- `externals`: `[{"name": "sig", "delta": "deltaSig"}, ...]`. The `delta` string is the coordinate name.
- `rows`: one entry per crossing equation, `"odd"`, `"even"` or `"all"`. The order of the rows is free.
- `channels`: one per exchanged family: `name`, `spins` (`"even"`, `"odd"`, `"all"`), `basis` (j names, the
  OPE basis), `matrices` (exactly one entry per row; each a j × j matrix, or `0` for all zero; each matrix
  element `0` or a list of terms).
  - `name`: the representation and the spin parity, e.g. `"rep1+"`, `"[2,0]+"`, `"T-"`. It is used in the PMP's
    part file names, so keep to letters, digits, `[ ] , + - _`. From AutoEqu, `op[op, rep[k], 1, ±1]` is
    `"repk±"` (see the table below).
  - `basis`: free labels (`"sig eps"`, `"v v"`); only their number j and order matter. The order is the order
    of the OPE coefficients in the matrices.
  - The nesting of a 1 × 1 matrix with one term: `[[ [ {"F": ["v","v","v","v"]} ] ]]` (matrix → line →
    element → list of terms).
- A term: `{"coeff": "<exact>", "F": ["a","b","c","d"]}`. Kinds: `F`, `H`, `S`, `T`. `coeff` defaults to
  `"1"`; it may contain `sqrt(...)`, e.g. `"2*sqrt(2/7)"`. An off-diagonal element of a symmetric j × j matrix
  carries half the cross term: the cross term 2 λ₁λ₂ X is written as X/2 in elements (1,2) and (2,1).
- `fixed`: the identity, `{"name": "identity", "delta": "0", "spin": 0, "matrices": [...]}` (1 × 1 matrices),
  and, when the external operators themselves appear in the OPE, `{"name": "external", "basis": [...],
  "matrices": [...]}` plus its OPE key. Every term of `external` has `"exchanged": "<external>"`, the external
  operator exchanged in that term. Its OPE: nothing (a PSD constraint), `"opeAngle": true` (only j = 2: the
  coordinate `theta` is appended and the OPE vector is (cos θ, sin θ) in basis order), or `"opeVector": [...]`
  with `"opeCoordinates": [...]`. Leave `external` out when no external operator is exchanged.
- `objectives`: these two; `constraints` lists `"external"` when there is such an entry, else it is `[]`:

```json
"objectives": {
  "feasibility": {"constraints": ["external"], "objective": [], "normalization": [{"fixed": "identity"}]},
  "navigator":   {"constraints": ["external"], "objective": [{"fixed": "identity"}],
                  "sigma": {"shift": "1/5", "step": "3/10", "width": "6"}}
}
```

**From simpleboot's AutoEqu output** (Mathematica InputForm), key by key:

| AutoEqu | bt-model/2 |
|---|---|
| `"VBlock$Deriv" -> {{"odd"}, {"even"}, …}` | the VBlock rows' parities, before mixed rows are merged (convention 4) |
| `"VBlock" -> {op[op, rep[k], 1, s] -> {row 1, row 2, …}}` | one channel per key: name `"repk+"` for s = 1 (even spins), `"repk-"` for s = −1 (odd spins); each row a j × j matrix (`{x}` for j = 1, `{{matrix}}` for j ≥ 2) |
| `"VBlock$OPEBasis"` | the channel's `basis` (its length and order) |
| `F[a,b,c,d]`, `H[a,b,c,d]`, a number times one of them | a term with that `coeff` |
| `"VBlock$Identity" -> {…}` with `Fp[a,b,c,d,0]`, `Hp[…,0]` | the `identity` entry (the 5th argument 0 is the identity) |
| `"VBlock$Single" -> {…}` with `Fp[a,b,c,d,x]`, `Hp[…,x]` | the `external` entry; the 5th argument x is `exchanged` |
| `"VBlock$Single$OPEBasis"` | the `external` entry's `basis` |
| `"VBlock$External" -> {"sig", "eps"}` | `externals`, in that order |

**Conventions** (fixed by the generator):

1. `F[a,b,c,d]` = simpleboot's F^{ab,cd}_−, the odd z-derivatives of A(z) − A(1−z); `H[a,b,c,d]` = F^{ab,cd}_+,
   the even derivatives of A(z) + A(1−z); A(z) = v^{(Δb+Δc)/2} g^{ab,cd}(u, v). So **F terms only in `odd`
   rows, H terms only in `even` rows, S and T terms only in `all` rows.**
2. **The odd-spin signs.** This file pairs a term [a,b,c,d] with λ_{abO} λ_{cdO} (simpleboot's VBlock pairing).
   Upstream's blocks pair λ_{abO} λ_{dcO} = (−1)^ℓ λ_{abO} λ_{cdO}; the generator converts by multiplying every
   term (F, H, S and T) by (−1)^ℓ of the exchanged spin. So:
   - from AutoEqu output: copy every coefficient as it is, including the different signs of a representation's
     `+` and `-` channels; never add or remove a (−1)^ℓ;
   - from a paper written with positive blocks in the λ_{ab}λ_{dc} pairing (most papers since SDPB; for
     example the O(N) vector equations of Kos, Poland and Simmons-Duffin, where the antisymmetric channel
     is (−F, F, −H)): the paper's odd-spin signs contain the ordering sign, so multiply the coefficients of
     every odd-spin channel by −1 (that channel becomes (F, −F, H));
   - a test of which pairing a source uses: in this file's pairing a correlator ⟨a b a b⟩ has +λ² for every
     spin and ⟨a b b a⟩ has (−1)^ℓ λ².
3. A sign that belongs to the equation itself (a tensor-structure coefficient, the relative sign of two terms
   of a crossing relation) is part of `coeff` like any other coefficient.
4. **Mixed rows.** Write upstream's blocks as S^{abcd}(z) = u^{−(Δa+Δb)/2} g^{ab,cd}(u, v) and
   T^{abcd}(z) = S^{abcd}(1−z). Then A^{abcd}(z) = u^q v^p S^{abcd}(z) and A^{abcd}(1−z) = u^p v^q T^{abcd}(z),
   with q = (Δa+Δb)/2 and p = (Δb+Δc)/2.
   - When every term of a row has the same q and p (⟨vvvv⟩, ⟨σεσε⟩), the factor drops: F and H are upstream's
     odd and even s-channel derivatives, and the row stays `odd` (F) or `even` (H).
   - Otherwise VBlock gives the relation as an F row and an H row over the same terms, coefficients f_i and
     h_i. They are the odd and even parts of one relation Σ a_i A_i(z) = Σ b_i A_i(1−z), a_i = (f_i + h_i)/2,
     b_i = (f_i − h_i)/2. Write it as one `all` row: an `S` term with coefficient a_i for every a_i ≠ 0 and a
     `T` term with coefficient −b_i for every b_i ≠ 0; the row replaces the pair (anywhere in the row list).
   - A term's A is the same function in the orders abcd and badc (Euler's identity), but not its upstream label
     or its factor: badc has the same q and p' = (Δa+Δd)/2. Choose for each term the order whose factor
     (u^q v^p for an S term, u^p v^q for a T term) is the row's common factor; then the factor drops.
   - Example (Ising, the row of ⟨σσεε⟩): S[σσεε] has u^{Δσ} v^{(Δσ+Δε)/2}. The VBlock term [σεεσ] in the T
     part has, as written, u^{Δε} v^{(Δσ+Δε)/2}, as [εσσε] u^{Δσ} v^{(Δσ+Δε)/2}: so it is written T[eps,sig,sig,eps].
5. **The identity** goes into the rows where its coefficient is not zero: the rows of correlators ⟨aabb⟩, and,
   with a global symmetry, only those projections where the identity's tensor structure survives. In an `all`
   row it is an `S` term. `delta` 0, spin 0.
6. Coordinates are ordered: external dimensions in `externals` order, then θ or the `opeCoordinates`, then
   extra spectrum parameters (point parameters the spectrum refers to that are not model coordinates).
7. Exact numbers only: coefficients as strings (`"1/2"`, `"sqrt(5/7)"`); dimensions (`delta`) exact
   rationals (no `sqrt`). In the TOML files every number is a string or an integer, never a float.

**spectrum.toml and objective.toml** (details in [interfaces 3.2–3.3](interfaces.md#32-the-spectrum-spectrumtoml)):

```toml
format = "bootlab-spectrum/1"
default_above_unitarity = "0"   # every (channel, spin) not listed: from the unitarity bound + this
[[gap]]                         # the whole continuum of these (channel, spin) starts here
channel = "rep1+"
spins   = [0]                   # a list, a spin, "2-40", "2-40:2" or "all"
delta   = "3"                   # or above_unitarity = "..."
[[isolated]]                    # an operator below its continuum, OPE unknown
channel = "rep1+"
spin    = 2
delta   = "3"
```

A gap bounds the whole continuum of that (channel, spin); an operator below it (for example a known lower
scalar) must be an `[[isolated]]` entry or one of the fixed terms. `objective.toml` is
`format = "bootlab-objective/1"` and `kind = "navigator"` (or another name from the model's `objectives`).

## 3. Worked example: the 3d Ising σ–ε system

The complete files are in `examples/ising/`. The system is ⟨σσσσ⟩, ⟨σσεε⟩, ⟨εεεε⟩ (σ Z2-odd, ε Z2-even), five
VBlock equations that become four rows.

**Externals and rows.**

```json
"externals": [{"name": "sig", "delta": "deltaSig"}, {"name": "eps", "delta": "deltaEps"}],
"rows": ["odd", "all", "odd", "odd"]
```

| Row | Correlator | Kind |
|---|---|---|
| 0 | ⟨σεσε⟩ | `odd`, F terms |
| 1 | ⟨σσεε⟩ = ⟨εσσε⟩ (mixed) | `all`, S and T terms |
| 2 | ⟨σσσσ⟩ | `odd` |
| 3 | ⟨εεεε⟩ | `odd` |

**Channels.** Z2-odd operators (in σ × ε) with odd and even spins, j = 1; Z2-even operators with even spins
(in σ × σ and ε × ε), j = 2:

```json
{"name": "rep2-", "spins": "odd", "basis": ["sig eps"],
 "matrices": [ [[ [{"F": ["sig","eps","sig","eps"]}] ]],
               [[ [{"coeff": "1", "T": ["eps","sig","sig","eps"]}] ]],
               0, 0 ]},
{"name": "rep2+", "spins": "even", "basis": ["sig eps"],
 "matrices": [ [[ [{"F": ["sig","eps","sig","eps"]}] ]],
               [[ [{"coeff": "-1", "T": ["eps","sig","sig","eps"]}] ]],
               0, 0 ]},
{"name": "rep1+", "spins": "even", "basis": ["eps eps", "sig sig"],
 "matrices": [ 0,
               [[ 0, [{"coeff": "1/2", "S": ["sig","sig","eps","eps"]}] ],
                [ [{"coeff": "1/2", "S": ["sig","sig","eps","eps"]}], 0 ]],
               [[ 0, 0 ], [ 0, [{"F": ["sig","sig","sig","sig"]}] ]],
               [[ [{"F": ["eps","eps","eps","eps"]}], 0 ], [ 0, 0 ]] ]}
```

Read it as: in row 1, a Z2-even operator O contributes (λ_{εεO}, λ_{σσO}) · M · (λ_{εεO}, λ_{σσO})ᵀ with the
off-diagonal ½ S[σσεε] (twice ½ = one g_S(σσεε)), and a Z2-odd operator contributes ∓ g_T(εσσε), with the sign
of the source equation's (−1)^ℓ written into `coeff` of `rep2+` and `rep2-`.

**Identity and external terms.**

```json
{"name": "identity", "delta": "0", "spin": 0,
 "matrices": [ 0,
               [[ [{"coeff": "1", "S": ["sig","sig","eps","eps"]}] ]],
               [[ [{"F": ["sig","sig","sig","sig"]}] ]],
               [[ [{"F": ["eps","eps","eps","eps"]}] ]] ]},
{"name": "external", "basis": ["sig sig eps", "eps eps eps"], "opeAngle": true,
 "matrices": [ [[ [{"F": ["sig","eps","sig","eps"], "exchanged": "sig"}], 0 ], [ 0, 0 ]],
               [[ [{"coeff": "-1", "T": ["eps","sig","sig","eps"], "exchanged": "sig"}],
                  [{"coeff": "1/2", "S": ["sig","sig","eps","eps"], "exchanged": "eps"}] ],
                [ [{"coeff": "1/2", "S": ["sig","sig","eps","eps"], "exchanged": "eps"}], 0 ]],
               [[ [{"F": ["sig","sig","sig","sig"], "exchanged": "eps"}], 0 ], [ 0, 0 ]],
               [[ 0, 0 ], [ 0, [{"F": ["eps","eps","eps","eps"], "exchanged": "eps"}] ]] ]}
```

The external operators σ and ε themselves are exchanged with OPE vector (λ_{σσε}, λ_{εεε}) = (cos θ, sin θ),
hence the third coordinate `theta`. The coordinates are `["deltaSig", "deltaEps", "theta"]`.

**Spectrum** (`examples/ising/spectrum.toml`):

```toml
format = "bootlab-spectrum/1"
default_above_unitarity = "0"

[[gap]]
channel = "rep1+"        # the next Z2-even scalar after epsilon
spins   = [0]
delta   = "3"

[[gap]]
channel = "rep2+"        # the next Z2-odd scalar after sigma
spins   = [0]
delta   = "3"
```

**Objective**: `format = "bootlab-objective/1"`, `kind = "navigator"`.

## 4. A prompt template

Give the assistant this, filled in, together with the sections 2 and 3 above (or this whole file), and the
source of the equations (a paper's crossing equations, or simpleboot AutoEqu output).

````text
You are writing input files for bootlab, a conformal bootstrap framework. Produce model.json in the format
"bt-model/2", plus spectrum.toml and objective.toml, for the system below. Follow the format and conventions
I attach exactly; they are fixed by the program and cannot be changed per model.

System: <e.g. O(N) vector v and singlet s in 3d, correlators <vvvv>, <vvss>, <ssss>, N = 3>
External scalars and their dimension coordinates: <e.g. v: deltaV, s: deltaS>
Source of the crossing equations: <paste the equations, or the AutoEqu/VBlock output>
  If a paper: <its block normalization and OPE-coefficient pairing, if known>
OPE of the external operators: <none | opeAngle (two coefficients) | opeVector with coordinates ...>
Channel names: <e.g. "repk+/repk-" from AutoEqu's op[op, rep[k], 1, +-1], or your own labels>
Spectrum assumptions: <e.g. next singlet scalar above 3, next vector scalar above 3, all else unitarity>

Rules (repeat of the conventions):
1. F terms only in "odd" rows, H terms only in "even" rows, S and T terms only in "all" rows.
2. Odd-spin signs: this file pairs a term [a,b,c,d] with lambda_ab lambda_cd. From AutoEqu output copy every
   coefficient as it is and never add or remove a (-1)^l. From a paper with positive blocks in the
   lambda_ab lambda_dc pairing, multiply the coefficients of every odd-spin channel by -1. Say which case
   you are in and why (use the <abab> / <abba> test of convention 2).
3. A row whose terms have different factors u^q v^p (mixed correlators) must be one "all" row with S and T
   terms (a = (f+h)/2 as S, -b = -(f-h)/2 as T, each term in the order abcd or badc whose factor is the row's).
   Show the derivation, with the factor of every term, for every such row.
4. Every channel has exactly one matrix per row, j x j with j = len(basis); 0 for an all-zero matrix or
   element. Off-diagonal elements carry half the cross term.
5. Every term of the "external" entry has "exchanged"; leave "external" out if no external operator is
   exchanged, and then write "constraints": [] in the objectives.
6. Coefficients are exact strings ("1/2", "sqrt(2/7)"). No floats anywhere. In TOML, numbers are strings.
7. Objectives: "feasibility" and "navigator" as in the attached example.

Then give, in this order:
(a) model.json, spectrum.toml, objective.toml;
(b) a table: for each row, which correlator and derivative parity it is, and which source equation(s);
(c) for each channel, which representation and spin parity, and its OPE basis;
(d) the coordinate order;
(e) every place where you were unsure, and what you assumed;
(f) the output of the attached structural check (section 5.1) on your model.json, if you can run Python.
Do not claim the model is correct: say which of the checks in the attached section 5 the user should run.
````

## 5. How to check the result

Work through these in order; the first two are cheap and catch most mistakes, the last two are the ones that
make a model trustworthy.

**5.1 Mechanical checks (local, seconds).** Save this as `check_model.py` and run `python3 check_model.py model.json`:

```python
import json
import sys


def no_float(x):
    raise SystemExit(f"a JSON float {x}: write every number as a string or an integer")


m = json.load(open(sys.argv[1]), parse_float=no_float)
problems = []


def bad(*what):
    problems.append(" ".join(map(str, what)))


rows = m.get("rows", [])
if m.get("format") != "bt-model/2":
    bad("format is not bt-model/2")
if any(r not in ("odd", "even", "all") for r in rows):
    bad("rows must be odd, even or all:", rows)
externals = [e["name"] for e in m.get("externals", [])]
entries = m.get("channels", []) + m.get("fixed", [])
names = [e["name"] for e in entries]
if len(set(names)) != len(names):
    bad("names are not unique:", names)
for e in entries:
    name = e["name"]
    is_external = name == "external"
    j = len(e.get("basis", [None]))
    if is_external and e.get("opeAngle") and j != 2:
        bad(name, "opeAngle needs a basis of 2")
    if len(e["matrices"]) != len(rows):
        bad(name, f"has {len(e['matrices'])} matrices for {len(rows)} rows")
        continue
    for r, (par, mat) in enumerate(zip(rows, e["matrices"])):
        if mat == 0:
            continue
        if not (isinstance(mat, list) and len(mat) == j and all(isinstance(l, list) and len(l) == j for l in mat)):
            bad(name, f"row {r}: not a {j} x {j} matrix")
            continue
        for line in mat:
            for el in line:
                if el == 0:
                    continue
                if not isinstance(el, list):
                    bad(name, f"row {r}: an element is {el!r}, not 0 or a list of terms")
                    continue
                for t in el:
                    kinds = [k for k in ("F", "H", "S", "T") if k in t]
                    if len(kinds) != 1:
                        bad(name, f"row {r}: a term needs exactly one of F, H, S, T: {t}")
                        continue
                    kind = kinds[0]
                    if (kind, par) not in (("F", "odd"), ("H", "even"), ("S", "all"), ("T", "all")):
                        bad(name, f"row {r} ({par}): a {kind} term")
                    if any(o not in externals for o in t[kind]) or len(t[kind]) != 4:
                        bad(name, f"row {r}: {kind}{t[kind]} must name four externals")
                    if ("exchanged" in t) != is_external:
                        bad(name, f"row {r}: exchanged belongs in the external entry's terms, and in every one of them")
                    if "exchanged" in t and t["exchanged"] not in externals:
                        bad(name, f"row {r}: exchanged {t['exchanged']} is not an external")
                    if not isinstance(t.get("coeff", "1"), str):
                        bad(name, f"row {r}: coeff {t.get('coeff')!r} must be a string")
for kind, o in m.get("objectives", {}).items():
    for c in o.get("constraints", []):
        if c not in names:
            bad(f"objective {kind}: constraint {c} is not a channel or fixed entry")
    for part in ("objective", "normalization"):
        for x in o.get(part, []):
            if "fixed" in x and x["fixed"] not in names:
                bad(f"objective {kind}: {part} names {x['fixed']}, not defined")
for p in problems:
    print("PROBLEM:", p)
print(("structure OK: " if not problems else f"{len(problems)} problems: ")
      + f"{len(rows)} rows, {len(m.get('channels', []))} channels")
sys.exit(1 if problems else 0)
```

It passes on every accepted model in the repository and stops at floats, terms in the wrong kind of row,
misspelled operator names, a missing or misplaced `exchanged`, undefined names in the objectives and
wrongly shaped matrices. It checks the structure only, not the physics.

**5.2 bootlab's own input check (login node, seconds).**

```bash
bin/bootlab submit --dry mymodel/run.toml
```

`inputs.load` checks the spectrum against the model's channels and spins, the coordinates against
`[coords].names`, the objective's `kind` against the model's objectives, and every number; it lists all errors
at once. It does not check the matrices: the generator does that when it reads the model, in the run's first
step. So, next:

**5.3 One cheap step on the cluster.** Set `max_steps = 1` and a low Λ (11, or the lowest your system makes
sense at), and submit. A model-file error shows as `failed: generator exit 1`, with the generator's message in
`<run>/steps/1/generator.err`. A clean first step means the file is well formed, not that it is right.

**5.4 Reduce to a known model.** If the system contains a known one (a single correlator, the Ising
σ–ε system, O(N) without the extra operator), make a reduced copy of the model and compare its results with
the known model's at the same points, Λ and solver settings: the same navigator value N, to the digits the
solver gives.
`examples/ising` is a known model.

**5.5 Compare with a known result.** Take a published island or bound, or a simpleboot run of the same system:
choose points clearly inside and clearly outside, at the same truncation (block preset, Λ) and solver
settings, and check that N < 0 inside and N > 0 outside (run one solve per point with a small `max_steps`, or a
short Skydive). Points right on the boundary are not evidence either way. For the Σ navigator only the sign
is comparable with simpleboot, not the size of N.

**Typical mistakes** these checks catch: a (−1)^ℓ added to or removed from AutoEqu's coefficients, or a
positive-block paper's odd-spin signs copied without the −1 of convention 2 (odd-spin channels flip),
an F term in an even row, a mixed row left as F/H, the identity in a row whose correlator has none, a missing
`exchanged`, a coordinate order different from `[coords].names`, a float in the TOML.

## 6. Writing run.toml

Start from `examples/ising/run.toml` and change: `name`, `[numerics] lambda`, `[coords] names` (the order of
convention 6), `[skydive] start`, `box_min`, `box_max` (strings), `gap_upper_limit` (`"1e-12"` or below for production;
the Ising example uses the reference program's `"0.1"` with tracking off), and `hessian_guess` (or `[]`).
Remove `gap_upper_limit_track = "off"` and `gap_upper_limit_first_only` to get the default tracking. Solver
options that larger systems need (for example the initial matrix scales and `maxComplementarity` for some O(N)
systems) go into `[solver]`.

## Appendix A. A test of this guide

**What was done** (2026-11-13). An AI assistant (a Claude subagent) was given only this guide, with section 3
and the sentence of convention 4 that names the Ising answer removed, and simpleboot's public tutorial file
`AutoEqu_Ising_OE.txt`. It had no access to the repository or the web and followed section 4's template.

**Result.**

- The structural check passed.
- Term by term the model was the accepted `ising-oe`. That covers:
  - the four rows in the same order (odd, all, odd, odd);
  - the merged ⟨σσεε⟩ row: S[σσεε] with ½ in the off-diagonal elements of the Z2-even channel, T[εσσε] with
    −1 and +1 in the Z2-odd even- and odd-spin channels, the identity as S with 1;
  - the external terms with `opeAngle`.

  The only difference was the channel names, which the assistant chose itself (E+, O+, O−; the accepted
  model has rep1+, rep2+, rep2-). The guide had no naming rule.
- With the channels renamed, the generator wrote the PMP of one point (the Ising example's start, Λ = 11)
  for both models. All 38 parts were the same, byte for byte (file names and sha256).

**What the test showed, and what changed in this guide.**

| Problem found | Change |
|---|---|
| no rule for channel names from AutoEqu | the AutoEqu table and the naming rule in section 2 |
| convention 4 spoke of "one prefactor" without defining it; as written, the orders abcd and badc gave the same function | convention 4 gives each term's factor u^q v^p and works the Ising row through |
| conventions 2 and 3 could be read against each other (is AutoEqu's ± an "explicit" odd-spin sign?) | convention 2 rewritten: copy AutoEqu as it is; the −1 for papers with positive blocks; the ⟨abab⟩ / ⟨abba⟩ test |
| spectrum.toml and objective.toml only by link | their format in section 2 |
| unstated: the `opeAngle` coordinate and its convention, the identity in an `all` row, where a merged row goes, objectives without `external`, what basis names are, the halved off-diagonal elements | stated in section 2 |
| the structural check did less than the prompt said | section 5.1 replaced by a stricter check (floats, term kinds by row, names, `exchanged`, objectives); it passes on every accepted model |

**To repeat it.** Give an assistant this guide without section 3, a simpleboot AutoEqu file of a system with
an accepted model, and the template of section 4. Compare its `model.json` with the accepted one term by term,
then compare the PMP of one point (one small job: the generator on one request for each model).
