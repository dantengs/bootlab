# 用 AI 助手写新模型

bootlab 的"模型"就是三个数据文件（model.json、spectrum.toml、objective.toml），一次运行另加 run.toml。只要给 AI 助手（Claude 或其他
能力足够的模型）提供准确的格式和约定，它就能根据论文里的交叉方程或 simpleboot 的 AutoEqu 输出写出这些文件。本指南给出格式和约定、一个完整的
例子（三维 Ising σ–ε 体系）、一个提示词模板，以及判断结果对不对的检查办法。附录 A 记录了对本指南本身的一次实测：一个只看过本指南（去掉第 3 节）
的助手，把 simpleboot 的 Ising 文件翻译成了与已验收模型完全相同的模型，PMP 逐字节相同。

**在信任任何模型之前，先读第 5 节。** 约定写错不会有任何报错，只会给出错误的物理结果。

[English](../en/writing-models-with-ai.md)

## 1. 要写出什么

| 文件 | 内容 | 格式说明 |
|---|---|---|
| `model.json` | 交叉方程、channel、恒等算符和外部算符的项、目标函数 | [接口文档 3.1](interfaces.md#31-模型bt-model2json) |
| `spectrum.toml` | 间隙和孤立算符 | [接口文档 3.2](interfaces.md#32-谱spectrumtoml) |
| `objective.toml` | 用模型里的哪个目标函数 | [接口文档 3.3](interfaces.md#33-目标函数objectivetoml) |
| `run.toml` | Λ、坐标、起点、盒子、Skydive 和求解器选项 | [接口文档 4](interfaces.md#runtoml) |

v1 的限制：d = 3；所有外部算符都是标量；block 由 scalar_blocks 计算。

## 2. 格式与约定（精简版）

**顶层**：`format` = `"bt-model/2"`、`name`、`dimension` = 3、`externals`、`rows`、`channels`、`fixed`、`objectives`。

- `externals`：`[{"name": "sig", "delta": "deltaSig"}, ...]`。`delta` 这个字符串就是坐标名。
- `rows`：每条交叉方程一项，`"odd"`、`"even"` 或 `"all"`。行的先后顺序随意。
- `channels`：每一族交换算符一个：`name`、`spins`（`"even"`、`"odd"`、`"all"`）、`basis`（j 个名字，即 OPE 基）、`matrices`（每行恰好一项；
  每项是一个 j × j 矩阵，或全零时写 `0`；矩阵元是 `0` 或若干项的列表）。
  - `name`：表示和自旋奇偶，如 `"rep1+"`、`"[2,0]+"`、`"T-"`。它会出现在 PMP 部件的文件名里，所以只用字母、数字和 `[ ] , + - _`。从 AutoEqu 来时，
    `op[op, rep[k], 1, ±1]` 就写成 `"repk±"`（见下表）。
  - `basis`：随意的标签（`"sig eps"`、`"v v"`），只有个数 j 和顺序有意义；顺序就是矩阵里 OPE 系数的顺序。
  - 1 × 1 矩阵、一项时的嵌套：`[[ [ {"F": ["v","v","v","v"]} ] ]]`（矩阵 → 行 → 矩阵元 → 项的列表）。
- 一项：`{"coeff": "<精确值>", "F": ["a","b","c","d"]}`。种类：`F`、`H`、`S`、`T`。`coeff` 默认 `"1"`，可以含 `sqrt(...)`，如
  `"2*sqrt(2/7)"`。对称的 j × j 矩阵，非对角元各写交叉项的一半：交叉项 2 λ₁λ₂ X 在 (1,2) 和 (2,1) 两个元里各写 X/2。
- `fixed`：恒等算符 `{"name": "identity", "delta": "0", "spin": 0, "matrices": [...]}`（1 × 1 矩阵）；外部算符本身出现在 OPE 里时，还有
  `{"name": "external", "basis": [...], "matrices": [...]}` 加它的 OPE 键。`external` 的每一项都写 `"exchanged": "<外部算符>"`，即这一项里交换
  的是哪个外部算符。它的 OPE：不写（半正定约束）、`"opeAngle": true`（只能 j = 2：坐标 `theta` 接在最后，OPE 向量按 basis 的顺序为
  (cos θ, sin θ)），或 `"opeVector": [...]` 加 `"opeCoordinates": [...]`。没有外部算符被交换时，不写 `external`。
- `objectives`：下面两种；有 `external` 时 `constraints` 写 `["external"]`，没有就写 `[]`：

```json
"objectives": {
  "feasibility": {"constraints": ["external"], "objective": [], "normalization": [{"fixed": "identity"}]},
  "navigator":   {"constraints": ["external"], "objective": [{"fixed": "identity"}],
                  "sigma": {"shift": "1/5", "step": "3/10", "width": "6"}}
}
```

**从 simpleboot 的 AutoEqu 输出**（Mathematica InputForm）逐键对应：

| AutoEqu | bt-model/2 |
|---|---|
| `"VBlock$Deriv" -> {{"odd"}, {"even"}, …}` | VBlock 各行的奇偶，合并混合行之前（约定 4） |
| `"VBlock" -> {op[op, rep[k], 1, s] -> {第 1 行, 第 2 行, …}}` | 每个键一个 channel：s = 1（偶自旋）名为 `"repk+"`，s = −1（奇自旋）名为 `"repk-"`；每行一个 j × j 矩阵（j = 1 时是 `{x}`，j ≥ 2 时是 `{{矩阵}}`） |
| `"VBlock$OPEBasis"` | channel 的 `basis`（个数和顺序） |
| `F[a,b,c,d]`、`H[a,b,c,d]`，以及乘了某个数的它们 | 带这个 `coeff` 的一项 |
| `"VBlock$Identity" -> {…}`，其中 `Fp[a,b,c,d,0]`、`Hp[…,0]` | `identity`（第 5 个参数 0 表示恒等算符） |
| `"VBlock$Single" -> {…}`，其中 `Fp[a,b,c,d,x]`、`Hp[…,x]` | `external`；第 5 个参数 x 就是 `exchanged` |
| `"VBlock$Single$OPEBasis"` | `external` 的 `basis` |
| `"VBlock$External" -> {"sig", "eps"}` | `externals`，按这个顺序 |

**约定**（由生成器固定）：

1. `F[a,b,c,d]` = simpleboot 的 F^{ab,cd}_−，即 A(z) − A(1−z) 的奇数阶 z 导数；`H[a,b,c,d]` = F^{ab,cd}_+，即 A(z) + A(1−z) 的偶数阶导数；
   A(z) = v^{(Δb+Δc)/2} g^{ab,cd}(u, v)。所以 **F 项只能在 `odd` 行，H 项只能在 `even` 行，S 和 T 项只能在 `all` 行。**
2. **奇自旋的符号。** 本文件里一项 [a,b,c,d] 配 λ_{abO} λ_{cdO}（simpleboot 的 VBlock 配法）。上游的 block 配 λ_{abO} λ_{dcO} =
   (−1)^ℓ λ_{abO} λ_{cdO}；生成器给每一项（F、H、S、T）乘上交换算符自旋的 (−1)^ℓ 来换算。所以：
   - 从 AutoEqu 输出来：每个系数照原样抄，包括同一表示的 `+`、`-` 两个 channel 符号不同的地方；绝不加上或去掉 (−1)^ℓ；
   - 从论文来，而论文用正的 block、按 λ_{ab}λ_{dc} 配对（SDPB 以来的大多数论文；例如 Kos、Poland、Simmons-Duffin 的 O(N) 矢量方程，其中
     反对称 channel 是 (−F, F, −H)）：论文里的奇自旋符号包含了顺序带来的符号，所以把每个奇自旋 channel 的系数都乘以 −1（那个 channel 就成了
     (F, −F, H)）；
   - 判断来源用哪种配法的办法：按本文件的配法，关联函数 ⟨a b a b⟩ 对所有自旋都是 +λ²，⟨a b b a⟩ 是 (−1)^ℓ λ²。
3. 属于方程本身的符号（张量结构的系数、交叉关系里两项的相对符号）和其他系数一样写进 `coeff`。
4. **混合行。** 把上游的 block 写成 S^{abcd}(z) = u^{−(Δa+Δb)/2} g^{ab,cd}(u, v)、T^{abcd}(z) = S^{abcd}(1−z)。于是
   A^{abcd}(z) = u^q v^p S^{abcd}(z)，A^{abcd}(1−z) = u^p v^q T^{abcd}(z)，其中 q = (Δa+Δb)/2，p = (Δb+Δc)/2。
   - 一行里各项的 q、p 都相同时（⟨vvvv⟩、⟨σεσε⟩），因子可以整行约掉：F、H 就是上游 s channel 的奇、偶阶导数，这一行保持 `odd`（F）或 `even`（H）。
   - 否则 VBlock 把这个关系写成同一组项上的一条 F 行和一条 H 行，系数 f_i、h_i。它们是同一个关系 Σ a_i A_i(z) = Σ b_i A_i(1−z) 的奇、偶部分，
     a_i = (f_i + h_i)/2，b_i = (f_i − h_i)/2。写成一行 `all`：每个 a_i ≠ 0 写一个系数为 a_i 的 `S` 项，每个 b_i ≠ 0 写一个系数为 −b_i 的 `T` 项；
     这一行替换那两行（放在行列表的哪里都行）。
   - 一项的 A 在 abcd 和 badc 两种顺序下是同一个函数（Euler 恒等式），但它在上游的标签和因子不同：badc 的 q 不变，p' = (Δa+Δd)/2。每一项选因子
     （S 项是 u^q v^p，T 项是 u^p v^q）等于这一行公共因子的那种顺序；这样因子就约掉了。
   - 例子（Ising，⟨σσεε⟩ 那一行）：S[σσεε] 的因子是 u^{Δσ} v^{(Δσ+Δε)/2}。T 部分里 VBlock 的项 [σεεσ] 照原样写因子是 u^{Δε} v^{(Δσ+Δε)/2}，写成
     [εσσε] 是 u^{Δσ} v^{(Δσ+Δε)/2}：所以写作 T[eps,sig,sig,eps]。
5. **恒等算符**写在它的系数不为零的行里：⟨aabb⟩ 型关联函数的行；有整体对称性时，只写在恒等算符的张量结构投影后不为零的那些行。在 `all` 行里它是
   一个 `S` 项。`delta` 为 0、自旋为 0。
6. 坐标的顺序：先按 `externals` 的顺序写外部算符的维度，然后是 θ 或 `opeCoordinates`，最后是谱里额外的参数（谱引用了、但不是模型坐标的点参数）。
7. 只用精确数：系数写成字符串（`"1/2"`、`"sqrt(5/7)"`）；维度（`delta`）是精确有理数（不能有 `sqrt`）。TOML 文件里的每个数都是字符串或整数，
   不能是浮点数。

**spectrum.toml 和 objective.toml**（细节见[接口文档 3.2–3.3](interfaces.md#32-谱spectrumtoml)）：

```toml
format = "bootlab-spectrum/1"
default_above_unitarity = "0"   # 没列出的 (channel, 自旋)：从幺正下限 + 这个值开始
[[gap]]                         # 这些 (channel, 自旋) 的整个连续谱从这里开始
channel = "rep1+"
spins   = [0]                   # 列表、单个自旋、"2-40"、"2-40:2" 或 "all"
delta   = "3"                   # 或者 above_unitarity = "..."
[[isolated]]                    # 低于它的连续谱的算符，OPE 未知
channel = "rep1+"
spin    = 2
delta   = "3"
```

gap 限定的是这个 (channel, 自旋) 的整个连续谱；比它低的算符（例如一个已知的更低的标量）必须写成 `[[isolated]]` 或者是固定项之一。
`objective.toml` 写 `format = "bootlab-objective/1"` 和 `kind = "navigator"`（或模型 `objectives` 里的另一个名字）。

## 3. 完整例子：三维 Ising σ–ε 体系

完整的文件在 `examples/ising/`。体系是 ⟨σσσσ⟩、⟨σσεε⟩、⟨εεεε⟩（σ 为 Z2 奇，ε 为 Z2 偶），VBlock 的五条方程重组成四行。

**外部算符和行。**

```json
"externals": [{"name": "sig", "delta": "deltaSig"}, {"name": "eps", "delta": "deltaEps"}],
"rows": ["odd", "all", "odd", "odd"]
```

| 行 | 关联函数 | 种类 |
|---|---|---|
| 0 | ⟨σεσε⟩ | `odd`，F 项 |
| 1 | ⟨σσεε⟩ = ⟨εσσε⟩（混合） | `all`，S 和 T 项 |
| 2 | ⟨σσσσ⟩ | `odd` |
| 3 | ⟨εεεε⟩ | `odd` |

**channel。** Z2 奇的算符（在 σ × ε 里）有奇、偶自旋，j = 1；Z2 偶的算符只有偶自旋（在 σ × σ 和 ε × ε 里），j = 2：

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

读法：第 1 行里，一个 Z2 偶的算符 O 的贡献是 (λ_{εεO}, λ_{σσO}) · M · (λ_{εεO}, λ_{σσO})ᵀ，非对角元是 ½ S[σσεε]（两个 ½ 合起来是一个 g_S(σσεε)）；
一个 Z2 奇的算符贡献 ∓ g_T(εσσε)，源方程里 (−1)^ℓ 的符号写进了 `rep2+` 和 `rep2-` 的 `coeff`。

**恒等算符和外部算符项。**

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

外部算符 σ、ε 本身作为交换算符出现，OPE 向量 (λ_{σσε}, λ_{εεε}) = (cos θ, sin θ)，所以有第三个坐标 `theta`。坐标是
`["deltaSig", "deltaEps", "theta"]`。

**谱**（`examples/ising/spectrum.toml`）：

```toml
format = "bootlab-spectrum/1"
default_above_unitarity = "0"

[[gap]]
channel = "rep1+"        # ε 之后的下一个 Z2 偶标量
spins   = [0]
delta   = "3"

[[gap]]
channel = "rep2+"        # σ 之后的下一个 Z2 奇标量
spins   = [0]
delta   = "3"
```

**目标函数**：`format = "bootlab-objective/1"`，`kind = "navigator"`。

## 4. 提示词模板

把下面的模板填好，连同上面第 2、3 节（或者整份本文件）和方程的来源（论文里的交叉方程，或 simpleboot 的 AutoEqu 输出）一起交给助手。
模板用英文写，因为格式说明和代码里的名字都是英文；也可以用中文。

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

## 5. 怎样检查结果

按顺序做。前两项很便宜，能查出大部分错误；最后两项才让一个模型值得信任。

**5.1 机械检查（本地，几秒钟）**。把下面这段存为 `check_model.py`，运行 `python3 check_model.py model.json`：

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

它对仓库里所有已验收的模型都通过；能拦下浮点数、放错种类的行里的项、拼错的算符名、漏写或错放的 `exchanged`、目标函数里
未定义的名字，以及形状不对的矩阵。它只查结构，不查物理。

**5.2 bootlab 自己的输入检查（登录节点，几秒钟）。**

```bash
bin/bootlab submit --dry mymodel/run.toml
```

`inputs.load` 按模型的 channel 和自旋检查谱，按 `[coords].names` 检查坐标，按模型的 objectives 检查目标函数的 `kind`，并检查每一个数；所有错误
一次列出。它不检查矩阵：那是生成器在运行第一步读模型时做的。所以接下来：

**5.3 在集群上跑便宜的一步**。设 `max_steps = 1`，取低的 Λ（11，或者你的体系有意义的最低值），提交。模型文件有错时结局是
`failed: generator exit 1`，生成器的说明在 `<运行目录>/steps/1/generator.err`。第一步干净地跑完，只说明文件格式没问题，不说明它是对的。

**5.4 退化到已知模型**。如果体系里包含一个已知模型（单个关联函数、Ising σ–ε 体系、去掉额外算符的 O(N)），做一份退化的模型副本，在相同的点、
相同的 Λ 和求解器设置下与已知模型比较：navigator 值 N 应当相同，到求解器能给出的位数。`examples/ising` 就是一个已知模型。

**5.5 与已知结果对照**。取一个已发表的岛或边界，或者同一体系的一次 simpleboot 运行：在同样的截断（block 预设、Λ）和求解器设置下，选明显在
里面和明显在外面的点，检查里面 N < 0、外面 N > 0（每个点做一次求解，用很小的 `max_steps`，或者跑一段短的 Skydive）。正好在边界上的点不能
作为任何一方的证据。用 Σ navigator 时，与 simpleboot 只能比较符号，不能比较 N 的大小。

**这些检查能发现的常见错误**：给 AutoEqu 的系数加上或去掉了 (−1)^ℓ，或者照抄了用正 block 的论文的奇自旋符号而没乘约定 2 的 −1（奇自旋 channel 反号）、F 项出现在偶数行、混合行仍写成 F/H、关联函数里
没有恒等算符的行却写了恒等算符、漏了 `exchanged`、坐标顺序与 `[coords].names` 不同、TOML 里有浮点数。

## 6. 写 run.toml

从 `examples/ising/run.toml` 改起：`name`、`[numerics] lambda`、`[coords] names`（按约定 6 的顺序）、`[skydive] start`、`box_min`、`box_max`
（字符串）、`gap_upper_limit`（生产用 `"1e-12"` 或更小；Ising 例子用的是参考程序的 `"0.1"` 并关了跟踪）、`hessian_guess`（或 `[]`）。删掉
`gap_upper_limit_track = "off"` 和 `gap_upper_limit_first_only`，就得到默认的跟踪。较大的体系需要的求解器选项（例如某些 O(N) 体系的初始矩阵
缩放和 `maxComplementarity`）写进 `[solver]`。

## 附录 A. 对本指南的一次实测

**做了什么**（2026-11-13）。给一个 AI 助手（Claude 的子代理）的只有本指南（去掉第 3 节，以及约定 4 里点出 Ising 答案的那一句）和 simpleboot
公开教程里的 `AutoEqu_Ising_OE.txt`；它不能看仓库，也不能上网，按第 4 节的模板去做。

**结果。**

- 结构检查通过。
- 逐项对比，模型与已验收的 `ising-oe` 完全相同，包括：
  - 四行，顺序也相同（odd、all、odd、odd）；
  - 合并后的 ⟨σσεε⟩ 那一行：Z2 偶 channel 的非对角元里是 ½ S[σσεε]，Z2 奇的偶、奇自旋 channel 里分别是 −1、+1 的 T[εσσε]，恒等算符是系数 1
    的 S；
  - 带 `opeAngle` 的外部算符项。

  唯一的差别是 channel 名：助手自己起了 E+、O+、O−，已验收模型用的是 rep1+、rep2+、rep2-。指南当时没有命名规则。
- 把 channel 改名之后，让生成器对同一个点（Ising 例子的起点，Λ = 11）各写一份 PMP：38 个部件全部逐字节相同（文件名和 sha256 都一样）。

**实测暴露的问题，以及本指南因此做的修改。**

| 发现的问题 | 修改 |
|---|---|
| 没有从 AutoEqu 取 channel 名的规则 | 第 2 节加了 AutoEqu 对应表和命名规则 |
| 约定 4 说"同一个前置因子"却没有定义；照原来的写法，abcd 和 badc 两种顺序得到的是同一个函数 | 约定 4 写出每一项的因子 u^q v^p，并把 Ising 那一行完整算了一遍 |
| 约定 2 和 3 可以读成互相矛盾（AutoEqu 里的 ± 算不算"显式的"奇自旋符号？） | 重写约定 2：AutoEqu 照原样抄；用正 block 的论文乘 −1；⟨abab⟩ / ⟨abba⟩ 的判断办法 |
| spectrum.toml 和 objective.toml 只给了链接 | 第 2 节写出了它们的格式 |
| 没有写明：`opeAngle` 的坐标名和约定、`all` 行里的恒等算符、合并出的行放在哪里、没有 `external` 时的目标函数、basis 名字的含义、非对角元写一半 | 都写进了第 2 节 |
| 结构检查做的事比提示词里说的少 | 第 5.1 节换成更严的检查（浮点数、各种行允许的项、名字、`exchanged`、目标函数）；它对所有已验收模型都通过 |

**怎样重做。** 给助手本指南（去掉第 3 节）、一个已有验收模型的体系的 simpleboot AutoEqu 文件，以及第 4 节的模板。拿它写出的 `model.json`
与已验收的那份逐项比较，再比较一个点的 PMP（一个小作业：生成器对每个模型各处理一份请求）。
