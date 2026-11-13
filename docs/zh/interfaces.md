# bootlab 接口文档（v1）

本文写明 bootlab 的每一层、层与层之间传递的数据，以及每一个配置文件，供别人在此基础上继续开发。它是项目规格的公开版本。凡写"必须"的地方，
不满足时程序报错停止。

[English](../en/interfaces.md)

目录：[1 分层](#1-分层与两个固定接口) · [2 一步的数据](#2-一步的数据) · [3 物理层](#3-物理层) · [4 run.toml](#runtoml) ·
[5 转换层](#5-转换层各模块) · [6 生成器输入](#6-接口一生成器输入) · [7 PMP](#7-接口二pmp) · [8 执行](#8-执行层一次-dive) ·
[9 资源](#9-资源由实测决定) · [10 Skydive 循环](#skydive-loop) · [11 检查点与续算](#11-检查点留下的一代续算) ·
[12 记录](#12-记录) · [13 安装与站点](#13-安装bootlabtoml-与站点) · [14 多节点](#multi-node-solves) ·
[15 探测与站点生成](#15-探测与站点生成) · [16 回归](#16-回归) · [17 扩展 bootlab](#17-扩展-bootlab) · [18 单机模式](#local-mode)

## 原则

- **物理结果第一**：结果正确最重要，工程上做到够用即可。
- **只校验一次，之后信任**：输入只在 `inputs.load` 里检查一次；之后各层相信收到的数据，遇到第一个错误就停。内部不做防御性检查，不自动重试。
- **正确性靠对照**：每一步都要与参考程序一致到两次构建之间能达到的位数；回归（第 16 节）是这道关。
- **异常就结束运行**，并写下原因（不挂起、不重试，也绝不记为"收敛"）。
- **资源由实测数据决定**；只用物理核（`--hint=nomultithread`、`OMP_NUM_THREADS=1`）。
- **数值精确**：点、谱的值、求解器的输出都是全精度十进制字符串；TOML 浮点数一律拒收。

## 1. 分层与两个固定接口

| 层 | 做什么 | 语言 |
|---|---|---|
| 物理层 | 模型（`bt-model/2` JSON）、谱和目标函数（TOML） | 数据 |
| 转换层 | 读入并检查输入；在每个点上求谱；写生成器输入；Skydive 循环（外层迭代）；记录 | Python ≥ 3.11，只用标准库 |
| 执行层 | PMP 生成器写 PMP；`pmp2sdp` 写 SDP；`dynamical_sdp` 求解（内层迭代）；bootlab 读输出 | 同一个 Apptainer 镜像里的 Haskell 和 C++ 程序，由 Python 经 `srun` 启动 |

两个固定接口，v1 不改：

1. **生成器输入**（`request.json`，第 6 节）：模型文件，加上每个点的坐标和算好的谱，格式是生成器参数文件里 `model` 一节的格式。
2. **PMP**（第 7 节）：SDPB 3 标准的多项式矩阵规划 JSON 部件，每个点一份清单。

新的搜索算法、新的生成器（例如带自旋的算符）、EFM，都接在这两个接口上。

每个外部程序都作为作业分配内的一个 `srun` 步启动（`execute.srun_argv`：`srun --exact`，明确给出 `--ntasks`、`--cpus-per-task`、`--mem`、
`--hint=nomultithread`、`--cpu-bind=cores`、`--kill-on-bad-exit=1`；MPI 程序经 `rank_env.sh`；镜像里的程序经 `apptainer exec`）。

## 2. 一步的数据

| # | 数据 | 由谁产生 | 放在哪里 | 何时删除 |
|---|---|---|---|---|
| 1 | 生成器输入 `request.json` | `pmp.write_request` | `<scratch>/<运行 id>/step_<n>/request.json` | 随这一步删除 |
| 2 | block 表（block 的有理逼近，每个自旋一个文件） | scalar_blocks，由生成器内部调用 | `<scratch>/<运行 id>/blocks/`，整个运行的各步共用 | 运行结束时 |
| 3 | PMP：本步各点共用的 JSON 部件，每个点一份清单 | 生成器 | `step_<n>/pmp/json/`、`step_<n>/pmp/<点名>/pmp.nsv` | 这一步做完后；终点的拷到 `endpoint/` |
| 4 | SDP（`block_info_*.json`、`block_data_*.bin`） | `pmp2sdp` | `step_<n>/sdp/<点名>/` | 这一步做完后 |
| 5 | 求解输出与检查点 | `dynamical_sdp` | `step_<n>/out/`；检查点在 `<运行目录>/checkpoint/` | out 的小文件拷到 `<运行目录>/steps/<n>/`；检查点留着（第 11 节） |

顺序：生成器（1→2、3）→ `k+1` 个 `pmp2sdp` 实例（4）→ 一个 `dynamical_sdp`（5）→ 往 `solves.jsonl` 写一行。`<scratch>` 是节点本地盘
或共享盘（第 13 节）。

## 3. 物理层

v1：d = 3，外部算符只能是标量。

### 3.1 模型：`bt-model/2`（JSON）

交叉方程按 simpleboot AutoEqu 的 VBlock 写法：每个 channel 对每条方程给一个 j × j 的 F/H（或 S/T）项矩阵。由生成器读入；约定是固定的
（要改就新开格式版本）。

```json
{
  "format": "bt-model/2",
  "name": "ising-oe",
  "dimension": 3,
  "externals": [{"name": "sig", "delta": "deltaSig"}, {"name": "eps", "delta": "deltaEps"}],
  "rows": ["odd", "all", "odd", "odd"],
  "channels": [
    {"name": "rep1+", "spins": "even", "basis": ["eps eps", "sig sig"], "matrices": [0, [[...]], ...]},
    ...
  ],
  "fixed": [
    {"name": "identity", "delta": "0", "spin": 0, "matrices": [...]},
    {"name": "external", "basis": ["sig sig eps", "eps eps eps"], "opeAngle": true, "matrices": [...]}
  ],
  "objectives": {
    "feasibility": {"constraints": ["external"], "objective": [], "normalization": [{"fixed": "identity"}]},
    "navigator":   {"constraints": ["external"], "objective": [{"fixed": "identity"}],
                    "sigma": {"shift": "1/5", "step": "3/10", "width": "6"}}
  }
}
```

| 键 | 含义 |
|---|---|
| `externals` | 外部标量，每个带它的维度坐标名 |
| `rows` | 每条交叉方程一项：`odd`（只取奇数阶导数）、`even`，或 `all`（取全部导数；混合关联函数用 S/T 重写的行） |
| `channels[]` | 交换的算符：`name`（如 `S+`、`T-`：表示和自旋奇偶）、`spins`（`even`、`odd`、`all`）、`basis`（OPE 基的名字，长度 j）、`matrices`（每行一个 j × j 矩阵；`0` 为全零；矩阵元是 `0` 或若干项的列表） |
| 项 | `{"coeff": "<精确值，可含 sqrt>", "F": [a,b,c,d]}`；还有 `H`、`S`、`T`。`coeff` 默认 1 |
| `fixed[]` | 恒等算符（`delta` 0、自旋 0），以及外部算符自身的交换（`external`；每一项写 `exchanged`） |
| `external` 的 OPE | 不给（j × j 半正定约束）、`opeAngle: true`（j = 2；多一个坐标 θ，OPE 向量为 (cos θ, sin θ)），或 `opeVector` 加 `opeCoordinates`（分量为常数或坐标名，精确有理数） |
| `objectives` | `objective.toml` 可选的目标函数：约束、目标、归一化，以及 Σ navigator 的 `sigma` |

约定（固定不变）：

1. `F[a,b,c,d]` 是 simpleboot 的 F^{ab,cd}_−（A(z) − A(1−z) 的奇数阶导数），`H` 是 F^{ab,cd}_+（A(z) + A(1−z) 的偶数阶导数），
   A(z) = v^{(Δb+Δc)/2} g^{ab,cd}(u,v)。
2. VBlock 的 `F[a,b,c,d]` 配 λ_{abO} λ_{cdO}；上游的 block 配 λ_{abO} λ_{dcO}。生成器给每一项乘上交换算符自旋的 (−1)^ℓ。
   **文件里不要再写这个因子。**
3. 源方程里显式带有的奇自旋符号写在系数里。
4. 混合关联函数的行（各项的前置因子 u^q v^p 不同）必须写成一行取全部导数的方程：A(z) 项写成 `S`，A(1−z) 项写成 `T`（上游的 s、t
   channel block）。
5. Navigator 的目标取恒等算符的贡献；Σ 归一化（simpleboot 的 `SigmaNavigatorNormalizer`）对每个连续谱在 gap+shift、+step、……、
   gap+shift+width 处取各 channel 矩阵的迹相加，孤立算符不计入。系数为 +1 时，允许点的 N < 0。

点的坐标依次是：外部算符的维度，然后是 θ（`opeAngle`）或 `opeCoordinates`，最后是谱里额外引用的参数（3.2）。

生成器读模型时（第一步）检查：名字不重复且都有定义；每行一个 j × j 矩阵；F 只在 `odd` 行、H 只在 `even` 行；`exchanged` 恰好出现在外部
算符项里；维度表达式是精确有理数。

### 3.2 谱：`spectrum.toml`

```toml
format = "bootlab-spectrum/1"
default_above_unitarity = "0"     # 下面没写到的 (channel, 自旋) 从幺正下限 + 这个值开始

[[gap]]                           # delta 与 above_unitarity 恰好写一个
channel = "rep1+"
spins   = [0]                     # 列表、单个自旋、"2-40"、"2-40:2" 或 "all"
delta   = "3"

[[isolated]]                      # 连续谱之外的算符，OPE 未知（j x j 半正定）
channel = "rep1+"
spin    = 2
delta   = "delta_tprime"          # 可以引用点参数
```

- 值写成字符串或整数：精确的十进制数，或者由点参数和十进制数做 `+ − × ÷`、加括号构成的表达式。在每个点上用 `fractions.Fraction`
  求值，写成精确十进制数；结果不是有限小数就报错。
- 引用的参数如果不是模型坐标，就成为额外坐标，接在模型坐标后面。生成器只收到模型坐标和算好的谱。
- 自旋：`"all"` = 本次运行自旋表里这个 channel 允许的自旋；区间取其中 channel 允许的自旋（也可以超出自旋表）；列表只能写允许的自旋；
  展开后为空是错误。
- 错误（读入时）：未知的键、不存在的 channel、channel 不允许的或为负的自旋、同一 (channel, 自旋) 设了两次、`delta`/`above_unitarity`
  不是恰好一个。在某个点上：同一个孤立算符写了两次。
- 警告（开跑时写进 `driver.log`）：高于幺正下限的量为负；gap 或孤立算符低于幺正下限（d = 3：自旋 0 是 1/2，ℓ ≥ 1 是 ℓ + 1）；孤立算符
  高于它所在的连续谱；自旋不在本次运行的自旋表里。

### 3.3 目标函数：`objective.toml`

```toml
format = "bootlab-objective/1"
kind   = "navigator"                                     # 模型 objectives 里的一种
# sigma = { shift = "1/5", step = "3/10", width = "6" }  # 覆盖模型里的 Σ 参数
```

模型被拷到运行目录作为 `model.json`，`sigma` 在拷贝里逐键替换；生成器读的是这份拷贝。在用的 navigator 归一化有两种：Σ（`navigator`，
与 simpleboot 可比）和 GFF（`examples/ising-gff` 的 `navigator_gff`：用 GFF 双迹算符的交换做归一化，同 Hyperion 的 Ising navigator）。
两者给出相同的允许/排除判定和岛的边界，N 的数值和路径不同。

### 3.4 点

点是坐标名到十进制字符串的映射。坐标 = 模型坐标 + 谱里额外的参数；`run.toml [coords].names` 必须等于这个列表，梯度、Hessian、步长都按
这个顺序。

<a id="runtoml"></a>
## 4. `run.toml`

```toml
format = "bootlab-run/1"
name   = "<运行名>"                         # 运行目录是 <runs>/<name>-<年月日-时分秒>

[inputs]                                   # 相对于 run.toml
model = "model.json"; spectrum = "spectrum.toml"; objective = "objective.toml"

[numerics]
lambda       = 11                          # 奇数，>= 3；nmax = (lambda + 1) / 2
block_preset = "simpleboot"                # "simpleboot"（按 Λ）或 "hyperion"（按 nmax；这时要给 spins）
# spins = [0, 1, 2, ...]                   # 自旋表（默认取预设）
# block_precision = 1024                   # 二进制位（默认取预设）
# solver_precision = 785                   # pmp2sdp 和求解器（simpleboot：Λ <= 19 用 785，否则 1024）

[coords]
names = ["deltaSig", "deltaEps", "theta"]

[skydive]
start = [...]; box_min = [...]; box_max = [...]       # 每个坐标一个字符串
max_steps = 300
stop_on_negative = false
hessian_guess = []                         # 初始 Hessian 的对角元；[] = 求解器自取
find_boundary_direction = []               # [] = 极小化 N
find_boundary_when_allowed = true          # N < 0 之后才沿方向推
gap_upper_limit = "1e-12"                  # 必须写
# gap_upper_limit_track = "1"              # c；"off" = 固定上限
# gap_upper_limit_track_floor = "0.1"      # κ
# gap_upper_limit_first_only = true        # 只在 track 为 "off" 时起作用：true = 第一次之后传 0（不限）
# gap_threshold_tighten = true
# infinitesimal = "1e-20"                  # >= 1e-25
# block_timings = "/绝对路径/checkpoint/block_timings"   # 新运行：用另一次运行的块耗时，不做 timing run
# mixed_precision = false                  # true：见 10.1 节

[solver]                                   # SDPB 选项，键是参数名，覆盖 skydive.solver_defaults(nmax)
[dynamical]                                # dynamical_sdp 自己的选项，覆盖 skydive.DYNAMICAL_DEFAULTS

# [exec]
# site = "mysite"                          # 不写就用 bootlab.toml 的 site

# [resources]                              # 不写：单节点
# nodes = "auto"                           # 或同一型号的 ["host1", "host2"]
# max_nodes = 3; min_cores = 256; mem_gb = 300; prefer = ["typeA"]
# allow_multinode = true                   # 覆盖站点的 allowed = false
```

求解器的默认值就是参考程序在这个 nmax 下传的那一份（SDPB 按 nmax 分档的选项，加上 dynamical_sdp 的默认值），`tests/test_skydive.py`
逐项固定。`--maxRuntime` 是作业剩余时间减 15 分钟。`load` 拒绝：`gap_upper_limit` ≤ 求解器的 `dualityGapThreshold`（第一次求解会直接以
`optimal` 结束），`infinitesimal` < 1e-25，`gap_upper_limit_track` 不在 (0, 3] 内，`[resources]` 里的未知键。

## 5. 转换层：各模块

| 模块 | 作用 | 主要入口 |
|---|---|---|
| `inputs.py` | 读入并检查模型、谱、目标函数、run.toml；在点上求谱 | `load(run_toml) -> Problem`、`resolve_spectrum(problem, point)`、`spectrum_warnings` |
| `numbers.py` | 精确十进制数和表达式语言 | `Expr(text).value(env)`、`decimal_text(Fraction)`、`Num`（写成 JSON 数字的十进制字符串） |
| `pmp.py` | 生成器输入及其运行；PMP 清单和标签 | `write_request`、`generate`、`pmp_lists`、`write_labels` |
| `skydive.py` | Skydive 循环、记录、结局；求解器默认值 | `Run.loop`、`Run.step`、`gap_threshold`、`tracked_gap_limit`、`optimal_ending`、`solver_defaults`、`start_state`、`resume_state` |
| `execute.py` | 站点和安装；`srun` 步；带内存看门狗的 `pmp2sdp` 波次；`dynamical_sdp`；读输出；选 scratch；节点间拷贝；检查点的代 | `load_site`、`srun_argv`、`build_sdps`、`run_dynamical_sdp`、`read_dive`、`classify`、`choose_scratch`、`distribute`、`keep_generation`、`restore_generation` |
| `layout.py` | 按站点和 bench 给运行选节点；核对分配 | `choose_nodes`、`resume_plan`、`check_allocation`、`permitted` |
| `bench.py` | bench 文件；求解器进程数；`pmp2sdp` 的进程数和实例数 | `load_bench`、`solver_procs`、`grid_ok`、`pmp2sdp_procs`、`pmp2sdp_at_once` |
| `cli.py` | `run`、`resume`、`submit`、`summary`、`regress`、`probe`、`site` | `main` |
| `probe.py` | 集群探测 | 子命令 `plan`、`node`、`pair`、`merge` |
| `sitegen.py` | 由探测生成站点；站点对比 | `generate`、`choose_scratch`、`multinode` |
| `receive.py` | 节点间拷贝的接收端 | `serve` |
| `regress.py` | 回归检查 | `check(run_dir)`、`input_dir` |
| `rank_env.sh` | 给每个 rank 设网络设备，然后 `exec` 后面的命令 | 见第 14 节 |

## 6. 接口一：生成器输入

每一步调用一次，处理全部 `k+1` 个点：`<generator> pmp <request.json>`（镜像里是 `/opt/bootlab/bin/generator`，它设好生成器的库和它调用的
scalar_blocks 脚本）。

```json
{
  "model": {"name": "data", "file": "<运行目录>/model.json", "lambda": 11, "blockPreset": "simpleboot"},
  "objective": "navigator",
  "points": [
    {"name": "c", "coords": [0.519, 1.4, 0.956],
     "spectrum": {"twistGap": 0, "gaps": [{"channel": "rep1+", "spins": [0], "gap": 3}], "isolated": []}},
    {"name": "plus_0", "coords": [0.51900000000000000001, 1.4, 0.956], "spectrum": {"...": "..."}}
  ],
  "out": "<本步目录>/pmp",
  "blocks": "<本次运行的 block 目录>",
  "cores": 128
}
```

- `model` 的字段名就是生成器参数文件 `model` 一节的字段名（`[numerics]` 写了时还有 `blockPrecision`、`spinList`）。
- `coords` 按模型坐标的顺序，是 JSON 数字，照十进制写出全部位数；`spectrum` 是这个点算好的谱（同样的写法）。
- 点：`c`（中心）和 `plus_0` … `plus_{k-1}`（第 i 个坐标加 ε）。邻点用精确十进制算术算出。
- 生成器把每个坐标读成 512 位浮点数，为每个点建 bound，计算 block 表（各点相同的表只算一次），用上游的数值代码（Hyperion 的
  `makeSDPBInput` 那一路）写出部件。调度是它自己的：在本节点起 worker 进程（CPU 池为 `cores` 个），每个点的计划在 worker 上只算一次，
  归一化部件拆成约 max(512, 核数) 块、按上游原来的顺序相加，所以部件与上游代码写出的逐字节相同。
- block 表的键只用 Δ12 的约 31 位小数，所以 `infinitesimal` 必须 ≥ 1e-25（1e-40 会让邻点和中心点共用一张表，丢掉对 Δ12 的导数）。

## 7. 接口二：PMP

```
<out>/json/*.json       标准 SDPB 3 PMP JSON 部件；本步共用的部件只写一次
<out>/<点名>/pmp.nsv    这个点的部件清单（NUL 分隔的绝对路径），作为 pmp2sdp --input
<out>/workers/          生成器 worker 的日志
```

部件文件名带约束标签（`continuum_<channel>_l<自旋>_<u|d><gap>_mat_…`、`isolated_…`）和一个哈希。比较部件要按标签和内容，不按文件名。
收敛时中心点的部件拷到 `<运行目录>/endpoint/pmp/parts/`，另有 `pmp.nsv`、`precision` 和 `labels.json`（供 EFM 用）。

## 8. 执行层：一次 dive

1. `k+1` 个 `pmp2sdp` 实例（`pmp2sdp --precision <p> --input <pmp.nsv> --output <sdp 目录>`），分波次运行（第 9 节）。
2. 一个 `dynamical_sdp`，用节点（或多台节点）的全部物理核（或 bench 选出的进程数）；参数是 `--sdpDir` 中心 SDP、`--newSdpDirs` 邻点的 SDP、
   `--checkpointDir <运行目录>/checkpoint`，从上一次求解热启动（第一次冷启动），再加上本步的选项（间隙上限和阈值、BFGS 状态：梯度、Δx、
   Hessian、累计迭代数、找边界方向）。
3. 把 `out.txt`、`skydiving_out.txt` 和向量文件读成一个 `Dive`：

```python
@dataclass
class Dive:
    verdict: str          # "stepped" | "optimal" | "aborted"
    reason: str           # 求解器原样的 terminateReason，或 "failed: <程序> exit <码>"
    primal: str; dual: str; gap: str
    navigator: str | None # N；optimal 时为 None（求解器这时不算 N）
    ext_step: list[str]   # 步长 Δx
    gradient: list[str]; hessian: list[list[str]]   # BFGS
    iterations: int       # 累计迭代数，下一步原样传回
    alpha_p: str; alpha_d: str; beta: str; climbed: bool; find_minimum: bool
```

| 求解器结束情况 | verdict |
|---|---|
| `met criteria to update sdp input files` | `stepped` |
| `primal-dual optimal`（及其他正常结束） | `optimal` |
| `maxIterations`、`maxRuntime`、SIGTERM（退出码 15） | `aborted`，`unfinished: <原因>` |
| `maxComplementarity`、原始/对偶步长太小 | `aborted`，`stalled: <原因>` |
| 某个程序非 0 退出或输出缺失 | `aborted`，`failed: <程序> exit <码>` |

遇到未知的 `terminateReason`，运行停止（`execute.classify`）：把它加进去。

## 9. 资源由实测决定

三段（生成器 → `pmp2sdp` → 求解器）依次独占整个节点。

| 环节 | 并行方式 | bootlab 怎么选 |
|---|---|---|
| `dynamical_sdp` | MPI，每步一次 | 这个型号单节点的 bench 记录里，取规模最接近的问题（距离 \|ln(N/Nᵢ)\| + \|ln(块数/块数ᵢ)\|）最快的进程数，只在网格不退化的进程数里选；没有记录就用全部物理核。整个运行不变（热启动链要求） |
| `pmp2sdp` | MPI，扩展性差 | 每个实例 p = min(31, ⌊核数 ÷ 点数⌋) 个进程。第一步只转中心点，从这个 srun 步的 cgroup 每 0.2 s 读一次 memory.current − file + shmem，取峰值 P；之后 k = ⌊(节点内存 − driver_reserve) ÷ (1.15 P)⌋ 个实例同时跑。每个步的 `--mem` = 节点内存 ÷ 实例数。**看门狗**：一波里任何一个步占用超过它 `--mem` 的 95%，就撤掉整波（内存不够时 pmp2sdp 不报错而是停住），这一步以失败结束。没有 cgroup v2 时看门狗关掉，内存改用节点 MemTotal − MemAvailable |
| block 表 | 生成器内部，scalar_blocks 多线程 | k = min(表数, 核数 ÷ 4) 张表同时算，每张 核数 ÷ k 个线程 |

**网格不退化**：Elemental 的默认进程网格从 ⌊√p⌋ 往上找第一个能整除 p 的数作为行数；p = 127 得到 127 × 1（慢），128 得到 16 × 8。
2 的幂都满足。

<a id="bench-files"></a>
**bench 文件** `sites/<站点>/bench/<节点型号>.json`（`format = "bootlab-bench/1"`）：`node_type`、`physical_cores`、`sockets`，以及若干
记录列表，每条记录写明来源：

| 列表 | 记录的主要字段 | 谁读 |
|---|---|---|
| `solver` | `problem`、`model`、`lambda`、`N`、`blocks`、`precision`、`program`、`nodes`、`procs_per_node`、`s_per_iter`、`max_memused_gb`、`transport`（多节点；默认 `ib`）、`source` | `bench.solver_procs`、`layout.choose_nodes` |
| `sdp_build`、`blocks`、`generator`、`copy`、`rdma`、`pmp_size`、`nfs` | 各环节的实测 | 给人看；代码只读 `solver` |

没有 bench 文件就意味着：全部物理核、单节点、第一步实测 `pmp2sdp`。

**`bootlab bench <run.toml> [--node H] [--time] [--dry]`**（`benchrun.py`）：经 `submit` 交一个整机作业（作业里运行 `bench run <run.toml>`；
单机模式下在后台运行）。生成器只写运行起点这一个点的 PMP，一个 `pmp2sdp` 转成 SDP，`sdpb_3.0sky` 在全部物理核和一半物理核上各跑
`--maxIterations 12`（只取过得了 `grid_ok` 的进程数）。每次迭代的秒数 = 迭代表的 (t₁₂ − t₄) / 8（整秒）。每个进程数写一行 `solver`（`model`、
`lambda`、`N`、`blocks`、`precision`、`program`、`nodes` = 1、`procs_per_node`、`s_per_iter`、`source` = 日期和机器的插槽数、CPU 型号、物理核数、
内存），追加到 `<站点的 bench>/<节点型号>.json`。

<a id="skydive-loop"></a>
## 10. Skydive 循环

语义跟参考程序的 Skydive 循环（也就是 simpleboot 的驱动），区别见下。每一步：

1. **定选项**。间隙上限：第 1 步（以及冷续算的第一步）用 `gap_upper_limit`；之后跟踪时（默认 c = 1）取
   `min(gap_upper_limit, max(c·|N_prev|, κ·|N_1|, 1000·floor))`，κ = `gap_upper_limit_track_floor`（0.1），floor = 1e-30，N_1 是这次运行
   第一步的 N；写 `"off"` 时，`first_only` 从第 2 步起传 0（不限），否则每步传设定值。停止阈值：`gap_threshold_tighten`（默认）时取
   `max(1e-30, min(设定值, 1e-3·|N_prev|))`，否则取设定值。找边界方向一旦打开就一直开着。
2. 生成 x 及其邻点的 PMP，然后 dive。
3. 往 `solves.jsonl` 写一行。
4. 依次判断：
   - `aborted` → 结束：`unfinished`（带求解器的原因，`unfinished: …` 或 `stalled: …`）或 `failed`；停在 x；
   - `optimal` → 原始和对偶目标值同号：`converged`，`allowed`（N < 0）或 `excluded`（N > 0），拷 `endpoint/`；异号且找边界方向开着：
     `converged: boundary`（N 报 0，`n_bound` = 间隙）；否则 `stopped: N undecided at optimal`（保留检查点；收紧阈值后续算）；
   - `stop_on_negative` 且 N < 0 → `stopped: negative navigator`，停在 x + Δx；
   - x + Δx 出了盒子 → `stopped: left box`，停在 x；
   - 步数用完 → `unfinished: max steps`；
   - 否则 x ← x + Δx，把迭代数、梯度、Δx、Hessian 传给下一步；给了找边界方向且 N < 0 时打开它。

间隙上限为什么跟着 N 走：单步的长度大约随 √间隙 变化；上限远小于 |N| 时只能走小步，与 |N| 相当时点才走得动（Ising Λ = 19 从标准起点，
用现在的规则 84 步收敛，用它的早先版本要 192 步）。注意：`allowed` 的终点是一个 N < 0 的点，不是 N 唯一的极小点；在平坦的方向上，
两次运行终点的差别可能远大于 N 的差别。

### 10.1 混合精度（`mixed_precision = true`）

默认关；`bin/bootlab regress` 用全精度。求解器和 `pmp2sdp` 用 {384, 512, 768, 1024} 中低于问题 `solver_precision` 的、满足
`0.30103·P − 85 − 0.6·log10(1/间隙) ≥ 30` 位的最小精度 P（间隙取上一条记录的，开始时取 `gap_upper_limit`），都不满足就用全精度。
只升不降。依据（实测）：`infinitesimal` 为 1e-40 时 Λ = 23、27 分别丢 78、83 位，那里 384–512 位时整步快 2.3–2.6 倍；
1e-20 时 Ising Λ = 11 丢 68 位（384 位与 785 位、与参考程序都一致到 47.5 位）；间隙 6.8e-21 时 384、512 位的步长与
785 位分别一致到 40.9、80.2 位（按公式折算丢 62 位）。按这个规则，
间隙从约 1 到 1e-30 都是 512 位。

- **健康检查**（低于全精度时；触发后从下一次调用起用全精度）：调用中止；`findMinimumQ` 和上一次走步的调用不同；既没爬升也不是冷启动的
  调用，迭代数超过 max(4, 最近 5 次这类调用迭代数中位数的 2 倍)。不查 N 的符号。
- **结束**：低于全精度的调用以 `optimal` 结束时不下结论；在同一点、用同一 BFGS 状态再用全精度解一次（低精度那次的日志移到
  `steps/<n>-<P>bits/`），由这一次决定。终点和 EFM 都是全精度。
- **检查点**：二进制检查点里每个数的长度由精度决定，文件里不存精度，`dynamical_sdp` 读不了别的精度写的检查点（它的文本检查点要求
  每次低于全精度的调用都把 X、Y 写成文本）。调用的精度不低于它所用的留下的一代（`layout.json` 的 `precision`；没有就是问题的精度）；
  要升高时，驱动先把这一代按新精度改写成下一代，limb 补零（`precision.convert_checkpoint`）。所以检查点已是全精度的运行再打开混合精度，
  仍是全精度。续算从最后一条记录的 `precision_next` 接着。
- 生成器不变（PMP 是十进制文本）；精度升高时重新测一次 `pmp2sdp` 的内存。

## 11. 检查点、留下的一代、续算

- `dynamical_sdp` 的 `--checkpointDir` 就是 `<运行目录>/checkpoint/`（各节点共用）。SDPB 写第 g+1 代时先由各 rank 写自己的文件，然后写
  `checkpoint_new.json`，再原子地改名为 `checkpoint.json`，只留两代；所以任何时刻被杀，`checkpoint.json` 指的那一代都完整，但它可能是
  一次调用中途的那一代。
- 所以每次调用结束后、写它的第 k 条记录之前，bootlab 建 `checkpoint/kept/<k>/`：这一代各 rank 文件的硬链接，`checkpoint.json` 和
  `block_timings` 的拷贝，以及 `layout.json`（`nodes`、`node_type`、`ppn`、`precision`、`generation`）；先在 `kept/<k>.new` 里建好再改名。记录写完后删掉
  更早的 kept，所以任何时刻至多三代，每条记录写完后两代。
- **续算**：取 k ≤ 记录数的最大 `kept/<k>`；更新的和没建完的删掉；`checkpoint/` 里只删别的代的 rank 文件；把留下的文件链回来。布局
  （节点数、型号、每节点进程数）变了就冷启动。运行目录所在的文件系统必须能建硬链接：每次 run、resume 开始时试建一个，不行就停。

## 12. 记录

所有数值都是全精度十进制字符串。

**`solves.jsonl`**，每次求解一行：

| 字段 | 含义 |
|---|---|
| `step`、`x` | 步号和点（按 `[coords].names` 的顺序） |
| `verdict`、`reason` | 第 8 节 |
| `navigator`、`primal`、`dual`、`gap` | N 和目标值 |
| `ext_step`、`gradient`、`hessian`、`iterations` | 传给下一步的 BFGS 状态 |
| `alpha_p`、`alpha_d`、`beta`、`climbed`、`find_minimum` | 求解器的诊断量 |
| `gap_upper_limit`、`gap_threshold`、`boundary_on` | 本步的选项 |
| `procs`、`node`、`nodes` | 每节点进程数；节点 0；全部节点（多节点时） |
| `pmp2sdp` | P、k、测量的那一步、memory.peak、节点内存的增量 |
| `seconds` | `pmp`、`sdp`、`dive`；多节点时还有 `copy`；`checkpoint` |
| `precision` | 这次调用 `pmp2sdp` 和求解器的位数 |
| `precision_next`、`precision_note`、`call_iterations`、`first_call` | 只在混合精度时有（10.1）：下一次的位数、变化的原因、这次的迭代数、冷启动 |

`solves.jsonl` 不记时刻，所以同一布局、同一份块耗时的两次运行可以逐字节比较。

**`result.json`**：`status`、`reason`、`x`、`coords`、`navigator`（`max steps` 和 `optimal` 时为 null；在边界上为 "0" 并带 `n_bound`）、
`primal`、`dual`、`gap`、`steps`、`inputs`（`model.json`、`spectrum.toml`、`objective.toml` 以及 `[numerics]` 的 sha256）、`versions`
（镜像路径和 sha256，镜像里的生成器、`pmp2sdp`、`dynamical_sdp`）、`scratch`（`dir`、`shared`、`transfer`）、`start`、`end`（ISO 8601，
带写入时那台机器的 UTC 偏移）。

**`summary.md`**（由 `solves.jsonl` 和 `result.json` 生成；`bootlab summary`），**`driver.log`**（带时刻的行：每项决定及其依据、每条命令行、
每个程序 stderr 的末尾），**`steps/<n>/`**（求解器的小输出；每个程序的 `.cmd`、`.out`、`.err`），**`endpoint/`**（第 7 节）。

## 13. 安装、`bootlab.toml` 与站点

**安装根目录** = 含 `src/` 和 `sites/` 的目录。**`bootlab.toml`**（不进仓库）：

| 键 | 含义 |
|---|---|
| `site` | `run.toml [exec]` 没写站点时用的站点 |
| `image` | 站点没写 `sif` 时用的镜像 |
| `runs` | 站点没写时的运行目录（默认 `<根>/runs`；环境变量 `$BOOTLAB_RUNS` 优先于一切） |
| `python` | 站点没写时驱动用的 Python（默认 `<根>/python/bin/python3`，存在时） |

相对路径相对于安装根目录；`$变量` 会展开。

**站点**就是 `sites/<名字>/site.toml` 加 `setup.sh`（每个作业都 source 它；让 apptainer 进 `PATH`）。`sites/_template/site.toml` 逐项说明；
摘要如下：

| 键 | 含义 |
|---|---|
| `sif`、`python`、`runs` | 可以不写（这时取 `bootlab.toml`） |
| `binds` | 绑进容器的目录（运行目录必须在其中某个之下；scratch 和运行目录会自动加上） |
| `setup` | 环境脚本（`setup.sh`） |
| `generator`、`pmp2sdp`、`dynamical_sdp` | 镜像里的程序名 |
| `scratch` | 一个路径，或依次的候选（取第一个各节点路径相同且可写的；`"shared"` = `<运行目录>/work`，节点之间不拷贝） |
| `scratch_basis`、`shared_measured_at` | 由 `site gen` 写：实测的依据；量过共享盘的目录（运行目录不在其中时开跑警告） |
| `driver_reserve_mb` | 给驱动和页缓存留下、不分给各步的内存（默认 4096） |
| `bench` | bench 目录（默认是站点下的 `bench/`） |
| `env_unset`、`[env]` | 从驱动的环境里删去 / 设置；每一步都继承它 |
| `[slurm]` | `submit` 用的 `partition`、`time`、`account`、`qos`、`extra`（它另加 `--exclusive --hint=nomultithread --mem=0`） |
| `[mpi]` | `transport`（一个名字：`ib`、`tcp-ipoib`、`tcp-1g`；多节点的 bench 记录要与它一致）、`net_devices`（`rank_env.sh` 的规则）、`ib_exclude`、`plugin`（srun 的 `--mpi`，默认 `pmix`） |
| `[node_types.<t>]` | `hosts`（fnmatch 通配或主机名）、`physical_cores`、`mem_gb` |
| `[multinode]` | `allowed`、`basis`、`prefer`、`peers`、`max_nodes`、`min_efficiency`、`min_gain`、`transfer`（`srun` 或 `ssh`）、`copy_network`、`copy_streams`；ssh 时还有 `ipoib_prefix`、`ssh`、`remote_setup` |
| `[shared_network]` | `server`、`network`、`basis`：探测时运行目录文件系统的走向；某台节点不同就警告 |

镜像需要的环境（写在 `[env]` 里）：`OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、`OMPI_MCA_pml=ucx`、`OMPI_MCA_btl=^openib`、
`UCX_POSIX_USE_PROC_LINK=n`、`PMIX_MCA_gds=hash`（每个 rank 在自己的 Apptainer 实例里运行），再加 `UCX_TLS`（InfiniBand 用 `rc,self,sm`，
TCP 用 `tcp,self,sm`）。

**`rank_env.sh`**：`srun` 经它启动每个 rank，在容器之外；它按 `BOOTLAB_NET_DEVICES`（站点的 `net_devices`）设 `UCX_NET_DEVICES`，第一条匹配的
规则生效：`ib:auto`（本节点的 IB 口：ACTIVE、LinkUp、InfiniBand、速率最高；速率相同时取其 IPoIB 网卡能通到共享文件系统服务器的那个口；
不选 `ib_exclude` 里的）、`<主机通配>=<设备>`，或 `subnet:<IPv4 前缀>`（在该网段上有地址的网卡）。没有规则匹配的主机，rank 直接退出。

<a id="multi-node-solves"></a>
## 14. 多节点求解

- 一个求解只用同一型号的节点（SDPB 要求每节点进程数相同）。生成器和 `pmp2sdp` 在节点 0 上运行；本步的 SDP 拷到其他节点的同一路径，
  然后 `dynamical_sdp` 跨全部节点运行。检查点在共享的运行目录里。
- **选节点**（`submit`，`nodes = "auto"`）：按 `prefer` 依次看各型号，先按内存和核数定台数下限，再看 bench 里同一模型、同一 Λ 的求解记录，
  只要 t(1)/(k·t(k)) ≥ `min_efficiency` 且 t(k−1)/t(k) ≥ `min_gain` 就加一台；空闲的 peers 够就用 `-w` 指定，否则看下一个型号；都不够就用
  `--constraint` 排队。`--resume` 保持检查点的布局。
- **拷贝**，`transfer = "srun"`（默认）：在每台目标节点上用一个轻量 `srun --overlap` 步启动 `receive.py`；节点 0 经 TCP 发 tar 流（`copy_streams`
  个流）。接收端只绑本节点在拷贝网络上的地址，只接受节点 0 的地址，要先收到一次性随机令牌（经环境变量传递），拒绝绝对路径和 `..`，用 tarfile
  的 `data` 过滤解包，120 s 等不到正确的连接或 600 s 没有数据就放弃。每次拷贝后核对文件数和字节数（`BOOTLAB_COPY_SHA256=1` 时逐个文件核对
  sha256）。`transfer = "ssh"` 则改为经 ssh 管道传 tar。
- **只有 TCP 时**：每台几百个 rank 时，MPI 全对全建立 TCP 连接在测试集群上会在启动阶段失败（connect() 被拒或 "Cannot assign requested address"，
  随后 UCX 报 "Destination is unreachable"；2 × 256 个 rank 七次失败六次，2 × 128 两次失败一次，每台 1 个 rank 从不失败）。原因未查明。所以
  没有 InfiniBand 的站点默认单节点。

## 15. 探测与站点生成

`bootlab probe plan`（登录节点）→ `probe node` 作业（每型号一个），可选 `probe pair` 作业 → `probe merge` → `bootlab site gen`。`probe plan --partition P` 只读分区 P（`sinfo -p`），作业交到 P，并在 plan.json 里记下 `partition`；之后
`site gen` 只放探测过的型号，写 `[slurm] partition = P`（手写文件的 `[slurm]` 优先）。`probe plan --node H` 在 H 上探测，不管它忙不忙。

| 文件 | 内容 |
|---|---|
| `plan.json` | `format`、`time`、`host`、`user`、`sif`、`sif_sha256`、`shared`、`shared_where`（路径、真实路径、st_dev、文件系统类型、来源、挂载点、看得到时的 mergerfs 分支）、`apptainer_setup`、`slurm`（配置）、`srun_mpi`、`nodes`（sinfo）、`types`（每型号：`hosts`、`physical_cores`、`threads`、`mem_mb`、`features`、`partitions`、`probe_hosts`、`sbatch`） |
| `node-<主机>-<作业号>.json` | `cpu`、`nics`、`ib`（开始和结束时的端口、选中的口）、`shared`（挂载、服务器、协议、路由、网络、硬链接、时钟、`where`）、`local`（候选：能否写、剩余空间、设备、是否旋转盘）、`throughput`（每个候选和共享盘：bootlab 文件形态和大文件，三遍）、`apptainer`、`image`（UCX 传输层和设备）、`cgroup`、`pythons`、`timezone`、`other_jobs_at_start` |
| `pair-<a>-<b>-<作业号>.json` | `light_srun`、`ib_ports`、`networks`（类型、网段前缀、两端、速率）、`perftest`、`startup`（每种传输层每一遍：退出码、秒数、refused / cannot_assign / unreachable 计数）、`copies`（每张网上的 srun 和 ssh 拷贝）、`shared`（同时写、冷读） |
| `probe.json` | `plan`、`nodes`（主机 → 记录）、`pairs`、`types`（含 `probed`）、`scratch_options`（主机 → 方案 → 秒数、分项、sigma、来源） |

<a id="site-generation-rules"></a>
**`site gen` 的规则**（每个值都带注释写明来源）：

- scratch：每台探测过的节点上各方案的用时（按 bootlab 文件形态写到关闭再读回 2 GiB；对允许多节点的型号再加上两节点探测的拷贝或冷读），
  取所有探测节点上都有、合计最少的方案；相同时按名字排。与另一类最好的相差不到 10 % 或 2σ（σ 取各遍的中位绝对偏差 × 1.4826）时，依据里
  注明"close"。某一遍超过中位数 3 倍记为尖峰。
- MPI：有任何探测节点选出了 IB 口，就用 InfiniBand rc 和 `ib:auto`；否则用第一个以太网网段上的 TCP。
- 多节点：只给两节点探测 3 次 rc 启动全部通过、且有可写本地盘的型号；否则写 `allowed = false` 并给出依据；没有两节点探测就不写 `[multinode]`。
- 拷贝：`srun`，走两台共有的 IPoIB 网，否则走以太网。
- `[node_types]` 取自 plan（所有型号，探测过与否都列）；`[slurm]`、`bench`、`sif`、`python`、`runs`、`[ib] exclude`、`[multinode]` 的覆盖取自
  手写文件。
- `--mask ib,local-scratch` 让决定按"没有 IB"或"没有本地盘"来做（探测到的事实不变）：用来测试规则。

## 16. 回归

`bootlab regress [--site] [--node] [--time] [--dry]` 把 `examples/ising`（Λ = 11、三步）写到 `<runs>/regress-input-<时间>/`，交一个整机作业运行
`regress run`；`regress check <运行目录>` 可以比较任何一个运行目录。参考值是 `tests/reference/ising-l11/bt-steps.json`（参考程序前三步的全精度值）。
每个数要一致 ≥ 130 位有效数字（用 `decimal` 以 500 位精度计算，只用标准库）；离散量必须相等。运行目录里写 `regress.json`（`passed`、`least`、
每步的位数、问题）。

单元测试：`python/bin/python3 -m unittest discover -s tests`（不需要集群；站点相关的测试用 `tests/fixtures/sites/` 里的中性夹具）。
`tests/test_physics.py` 拿参考程序参数检查的录下输出（6 个模型、14 组用例）核对物理层。

## 17. 扩展 bootlab

- **新算法**（Navigator、可行性……）：在 `skydive.py` 旁边加一个模块，由它产生点、读取结果；复用 `pmp.write_request`/`pmp.generate`（接口一）、
  `execute.build_sdps`，再为它的求解器（`sdpb`、`approx_objective`）写一个执行函数；写自己的记录行；在 `cli.py` 里加子命令。规矩照旧：输入在
  `inputs.load` 里检查，遇到异常结果就结束，资源按 bench 数据选，投入使用前与参考结果对照。
- **新生成器**（带自旋的算符）：任何能读第 6 节那样的请求、写出第 7 节那样 PMP 的程序，都可以经站点的 `generator` 接进来。
- **新站点**：探测后生成（第 15 节），或者复制 `sites/_template`。
- **新的实测**：按同样的格式往 bench 文件里追加记录，写明来源。
- 测试：单元测试加在 `tests/` 下；改动数值的地方要跑回归（第 16 节）或同样强度的对照。

<a id="local-mode"></a>
## 18. 单机模式（没有 Slurm 的一台机器）

bootlab 可以在一台没有 Slurm 的 Linux 机器（工作站）上运行。物理层、转换层、Skydive 循环、检查点和留下的代、续算、记录都与集群上相同；不同的只有
启动方式、资源和内存看门狗。命令见 README 的[单机模式一节](../../README.zh-CN.md)。

**用哪种启动方式**（`execute.site_launcher`；`driver.log` 写明选了哪种、依据是什么）：

1. 站点的 `launcher`（`"slurm"` 或 `"local"`），没写就看 bootlab.toml；
2. 都没写时，设了 `SLURM_JOB_ID`（在作业里）或 PATH 上有 `sbatch`，就是 `slurm`；
3. 否则是 `local`。

**启动程序**（`execute.local_argv`、`launch`）：

- 镜像里的每个程序都是 `apptainer exec <binds> <sif> …`；主机上只需要 apptainer；
- MPI 程序经镜像自带的 `mpirun` 加 rankfile 启动：第 i 个 rank 绑列表里的第 i 个 cpu，用物理编号，`--bind-to hwthread`。列表是每个物理核的第一个
  硬件线程（`execute.physical_cpus`），所以一个 rank 不会和别的 rank 共用一个核，也不会用到一个核的第二个线程；
- 同时运行的几个程序（一波里的几个 `pmp2sdp` 实例）各分列表里互不相交的一段，相当于集群上的 `srun --exact` 步；
- 启动器给每个程序设 `OMPI_MCA_plm=isolated`（不会经 srun 起 rank），并去掉 `SLURM_*`、`PMIX_*`、`PMI_*` 变量；`UCX_TLS=self,sm`（只走共享内存）
  来自 `site gen` 写进单机站点的 `[env]`；
- 生成器不是 MPI 程序：它在镜像的 `taskset` 下用整张列表运行，`SLURM_JOB_NODELIST` 设为本机（这是它唯一得到的 Slurm 变量），节点列表由镜像里的
  `scontrol` 替身给出；
- 各程序与驱动在同一个进程组里（`submit` 用 setsid 起驱动，自成一组）；任何一个 rank 失败，它所在的整个 `mpirun` 作业结束。

**资源**：本机的物理核（按进程的亲和性），以及开跑时的可用内存（MemAvailable，至多站点的 `mem_gb`），再减去 `driver_reserve_mb`。除非 `[solver]`
自己写了，否则不传 `--maxRuntime`；运行按它自己的规则结束，或者由 `bootlab stop` 停下。

**没有 Slurm cgroup 时的 pmp2sdp 看门狗**（`execute.build_sdps_local`；第 9 节的规则不变：第一步测一个实例，k = ⌊内存 ÷ (1.15 P)⌋，某个实例超过
份额的 95% 就撤掉整波）：

| `memory_guard` | 读什么 | solves.jsonl 里的 `measure` |
|---|---|---|
| `rss`（默认） | 实例进程树上 `smaps_rollup` 的 Pss_Anon + Pss_Shmem 之和（共享页只计一次；没有 smaps_rollup 时用 RssAnon 之和，不含共享内存） | `pss`（或 `rss_anon`） |
| `systemd`（probe local 测到可用时设置） | 实例运行在带 `MemoryMax`/`MemoryHigh` 的 `systemd-run --user --scope` 里，像集群上一样读这个 scope 的 cgroup | `cgroup` |

被撤的一波先收到 SIGTERM，再收到 SIGKILL；这一步以失败结束。

**命令**（名字与集群上相同）：

| 命令 | 单机时的含义 |
|---|---|
| `bin/bootlab probe local --sif <镜像> --out probe [--shared <目录>] [--local <目录>] [--sdp <小 SDP>]` | 直接在本机测，几分钟：CPU 和物理核、内存、按 bootlab 文件形态测各盘、运行目录、apptainer 和镜像、全核 `mpirun` 以及两半同时（记下每个 rank 的绑定；给了 `--sdp` 时再用它跑 `sdpb`）、systemd 用户 scope 能否限内存；写出 `probe/node-<主机>-local.json` 和 `probe/probe.json` |
| `bin/bootlab site gen --probe probe/probe.json --hand <文件> --out sites/<名字>` | 对单机探测：`launcher = "local"`，本机是唯一的节点型号，没有 `[multinode]`，没有 IB，`[mpi] transport = "local"`，`UCX_TLS = "self,sm"`，最快的 scratch，`memory_guard`；手写文件里可以给 `mem_gb` 和 `bench` |
| `bin/bootlab submit <run.toml>` / `submit --resume <运行目录>` | 在后台启动驱动（`setsid nohup`；加 `--systemd` 时作为临时的用户 unit），日志放在 `<runs>/` 下；`<运行目录>/driver.pid` 记它的进程组 |
| `bin/bootlab stop <运行目录> [--wait <秒>]` | 求解器在跑时，只给求解器的 rank 发 SIGTERM（不经 `mpirun`，否则它几秒后会补发 SIGKILL）：rank 写好检查点，这次调用以 `unfinished: SIGTERM signal received` 结束，驱动记下后正常结束；不在求解时，对驱动的进程组发 SIGTERM，续算时这一步从最后保留的那一代重做。`--wait`（默认 600 秒）等驱动的进程组结束。旧驱动的进程组还活着时，`submit --resume` 会拒绝 |
| `bin/bootlab regress --wait` | 在本机前台跑回归，打印 PASS 或 FAIL（只在单机模式下可用；集群上由作业去跑，`--wait` 会报错退出） |

一台机器同时只跑一个单机运行：驱动持有 `<runs>/.local.lock`，第二次 `submit` 会说明是哪个运行占着。

**被中止的调用的迭代数**（两种启动方式都适用）：被中止的调用（SIGTERM、maxRuntime、maxIterations）的记录里，累计迭代数包括这次调用自己做的迭代
（从求解器的迭代表读出），所以中途停下再续算的运行，与没停过的运行计数相同。求解器输出里没有迭代表时，记录写 `iterations_unknown` 和原因；续算从
最后一个已知的计数接着走，并写明这一点；`regress check` 会点出这样的调用。集群上 `scancel` 时驱动也被杀，不留 aborted 记录，续算从上一条记录保留的
那一代整步重算，计数规则与以前相同；这条规则之前写下的 aborted 记录（没有迭代数，也没有 `iterations_unknown`）在续算时照旧跳过，不报警。

`memory_guard = "systemd"` 时，某次采样读不到 scope 的 cgroup 就跳过这一次，不会退回 Pss。

**还没测过的**：`memory_guard = "systemd"` 和 `submit --systemd`（只有单测覆盖命令的写法），以及真正没有 Slurm 的机器（验收是在一台计算节点上把
Slurm 藏起来模拟的）。见 README 的已知限制。
