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
| 2 | block 表（block 的有理逼近，每个自旋一个文件） | scalar_blocks，由生成器内部调用 | `<scratch>/<运行 id>/blocks/p<位数>/`（每种精度一个目录），整个运行的各步共用 | 每次生成器调用之后（Skydive，`skydive.Run.prune_blocks`）：删掉最近两次调用都没建的，Δ12 = Δ34 = 0 的（每个点都一样）除外，driver.log 记留下和删掉的大小；其余运行结束时删 |
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
# precision_return_after = 3               # 10.1：降回前要连续几次健康的全精度调用（未测的初值）
# precision_return_limit = 2               # 10.1：一个运行最多降回几次（未测的初值）
# precision_return_margin = 10             # 10.1：降回时在 30 位之外再要求的位数（未测的初值）
# large_steps = false                      # true：见 10.2 节
# keep_checkpoints = 2                     # 每条记录后留下的检查点代数（第 11 节）
# large_steps_sign_cooldown = 3            # 不写：N 变号后不再走大步（10.2 节）

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
| `bench.py` | bench 文件；求解器进程数；`pmp2sdp` 的进程数和实例数 | `load_bench`、`solver_procs`、`grid_ok`、`pmp2sdp_procs`、`pmp2sdp_at_once`、`sdp_build_row` |
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

在这之前，本步的 PMP 由一次生成器调用写出，有看门狗：这次调用的日志、输出目录和块目录都超过一定时间没有变化，就结束调用，运行以
`unfinished: generator stalled after N s without output …` 结束，写出时限的来源和 `generator.err` 的最后几行（曾有生成器在部件
阶段无声挂住）；检查点完好，续算从这一步重新开始，和 maxRuntime 之后一样。时限是本运行第一次生成器调用用时的 5 倍，不少于 10 分钟；
还没有计时时为 1 小时，driver.log 写明所用的值。

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
| `pmp2sdp` | MPI，扩展性差 | 每个实例 p = min(31, ⌊核数 ÷ 点数⌋) 个进程。第一步（bench 里没有 `bootlab bench` 写的对应 `sdp_build` 行给出 P 时）只转中心点，从这个 srun 步的 cgroup 每 0.2 s 读一次 memory.current − file + shmem，取峰值 P；之后 k = ⌊(节点内存 − driver_reserve) ÷ (1.15 P)⌋ 个实例同时跑。每个步的 `--mem` = 节点内存 ÷ 实例数。**看门狗**：一波里任何一个步占用超过它 `--mem` 的 95%，就撤掉整波（内存不够时 pmp2sdp 不报错而是停住），这一步以失败结束。没有 cgroup v2 时看门狗关掉，内存改用节点 MemTotal − MemAvailable |
| block 表 | 生成器内部，scalar_blocks 多线程 | k = min(表数, 核数 ÷ 4) 张表同时算，每张 核数 ÷ k 个线程 |

**网格不退化**：Elemental 的默认进程网格从 ⌊√p⌋ 往上找第一个能整除 p 的数作为行数；p = 127 得到 127 × 1（慢），128 得到 16 × 8。
2 的幂都满足。

<a id="bench-files"></a>
**bench 文件** `sites/<站点>/bench/<节点型号>.json`（`format = "bootlab-bench/1"`）：`node_type`、`physical_cores`、`sockets`，以及若干
记录列表，每条记录写明来源：

| 列表 | 记录的主要字段 | 谁读 |
|---|---|---|
| `solver` | `problem`、`model`、`lambda`、`N`、`blocks`、`precision`、`program`、`nodes`、`procs_per_node`、`s_per_iter`、`max_memused_gb`、`transport`（多节点；默认 `ib`）、`source` | `bench.solver_procs`、`layout.choose_nodes` |
| `sdp_build` | `problem`、`model`、`lambda`、`N`、`blocks`、`precision`、`program`、`instances`、`procs`、`sdps`、`wall`、`per_sdp`、`peak_gb_per_instance`、`held_mb`（`bootlab bench` 写的行才有）、`cgroup_peak_mb`、`source` | `bench.sdp_build_row`（只认有 `held_mb` 的行）：Skydive 循环里 `pmp2sdp` 的内存（第 10 节） |
| `generator` | `problem`、`model`、`lambda`、`precision`、`points`、`cores`、`program`、`wall`、`node_gb`、`source`（`bootlab bench` 写的行；较早的行是整步的实测） | 给人看 |
| `blocks`、`copy`、`rdma`、`pmp_size`、`nfs` | 各环节的实测 | 给人看 |

没有 bench 文件就意味着：全部物理核、单节点、第一步实测 `pmp2sdp`。某一步在本节点型号的 bench 文件里找到 `bootlab bench` 写的、模型、Λ、
精度和 `pmp2sdp` 进程数都相同的 `sdp_build` 行时，就按这一行的 `held_mb` 定 k，不再先单独转中心点的 SDP 来测。

**`bootlab bench <run.toml> [--node H] [--time] [--dry]`**（`benchrun.py`）：经 `submit` 交一个整机作业（作业里运行 `bench run <run.toml>`；
单机模式下在后台运行）。在运行开始时会用的每个精度下（问题自己的；开了 `mixed_precision` 时再加梯级在 `gap_upper_limit` 处的选择，见 10.1），
生成器在全部物理核上只写运行起点这一个点的 PMP（写一行 `generator`：用时，`node_gb` 是节点 MemTotal − MemAvailable 的升幅），一个 `pmp2sdp`
按运行给每个实例的进程数（按运行的点数取 `pmp2sdp_procs`）转成 SDP（写一行 `sdp_build`：用时，以及与运行中一步相同口径的占用内存）。然后
`sdpb_3.0sky` 以全精度在全部物理核和一半物理核上各跑 `--maxIterations 12`（只取过得了 `grid_ok` 的进程数）。每次迭代的秒数 = 迭代表的 (t₁₂ − t₄) / 8（整秒）。每个进程数写一行 `solver`（`model`、
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
`0.30103·P − 95 − 0.6·log10(1/间隙) ≥ 30` 位的最小精度 P（间隙取上一条记录的，开始时取 `gap_upper_limit`），都不满足就用全精度。
按下面的规则升到全精度，也会再降回来。依据（实测）：`infinitesimal` 为 1e-40 时 Λ = 23、27 分别丢 78、83 位，那里 384–512 位时整步快 2.3–2.6 倍；
1e-20 时 Ising Λ = 11 丢 68 位（384 位与 785 位、与参考程序都一致到 47.5 位）；间隙 6.8e-21 时 384、512 位的步长与
785 位分别一致到 40.9、80.2 位（按公式折算丢 62 位）；一个较大的问题（N ≈ 2300、约 220 块，全精度 1024 位）在间隙 1.9e-14 时
512 位与 1024 位一致到 54.6 位（丢 91 位）。损失常数取实测最大的 91 再留余量，定为 95。按这个规则，间隙从 1 到 1e-30 都是 512 位（1e-30 时剩 41 位），
不会选 384 位。

省下的是求解器迭代的时间，所以收益取决于迭代在整步里的占比。Λ = 23、27、全精度 1024 位时迭代占一步的大头（上面的 2.3–2.6 倍）。
Ising Λ ≤ 19（全精度 785 位，每步 3 次迭代）只占约五分之一：Λ = 19 的第 2–40 步，迭代 785 位 546 s、512 位 344 s，生成器（559 s）、
`pmp2sdp`（342 s）、求解器启动（688 s）、载入和保存（420 s）都不变；这点收益被两台节点之间的差别抵消，看不出来。
在 bootlab 上，较大的问题（N ≈ 2300、约 220 块、全精度 1024 位）第 2–6 步，512 位整步快 1.27 倍：其中迭代快约 1.6 倍，
生成器加 `pmp2sdp` 每步约 220 s，不随精度变。所以在 bootlab 里整步的收益小于上面研究里的 2.3–2.6 倍（那里生成也降了精度）。

- **健康检查**：调用中止；上一次走步的调用 `findMinimumQ` 为假、这次为真（路径离开了允许区；
  反过来由真变假是从区外开始的运行进入允许区，属正常路径）；既没爬升、不是冷启动、
  也不是大步之后（10.2）的调用，迭代数超过 max(4, 最近 5 次这类调用迭代数中位数的 2 倍)。不查 N 的符号。
- **状态**（`precision.decide`）：每条记录的 `precision_state` 是下一次调用所处的状态，`precision_next` 是那次的精度；
  状态或精度变了，`precision_note` 写原因；`summary.md` 在"Precision"下列出每次变化。状态全由记录推出，续算不需要别的信息；
  没有这些状态之前的记录，`precision_next` 低于全精度算 low，否则算 high。
  - **low**：调用没通过健康检查，进入 **confirm**；如果离上次降回不到 5 次调用，直接 **locked**。
    当前间隙下全精度以下没有一档能留住 30 位时，**locked**。
  - **confirm**：同一次调用用全精度原样重做：同一点、同一 BFGS 状态，从上一条记录留下的那一代检查点开始（没有就冷启动，与原来那次一样）。
    低精度那次的记录保留，日志移到 `steps/<n>-<P>bits/`；低精度调用中止也照此确认。全精度那次同样没通过检查，说明是路径本身的行为、
    与精度无关：回到 **low**，取梯级的选择，从全精度这次的步和检查点接着算（中止的调用照常结束运行，任何精度都一样）。
    通过了：**high**。
  - **high**：全精度。已降回 `precision_return_limit` 次（2），或当前间隙下全精度以下没有一档能留住 30 + `precision_return_margin`（10）位，
    就 **locked**；连续 `precision_return_after`（3）次全精度调用通过检查（确认的那次也算；爬升调用不计数、也不打断）后，降到能留住这么多位的
    最低一档，回到 **low**。这三个默认值是未测的初值。
  - **locked**：全精度到底。
- **结束**：低于全精度的调用以 `optimal` 结束时不下结论；在同一点、用同一 BFGS 状态再用全精度解一次（低精度那次的日志移到
  `steps/<n>-<P>bits/`），由这一次决定（**locked**）。终点和 EFM 都是全精度。
- **检查点**：二进制检查点里每个数的长度由精度决定，文件里不存精度，`dynamical_sdp` 读不了别的精度写的检查点（它的文本检查点要求
  每次低于全精度的调用都把 X、Y 写成文本）。调用的精度与它所用的那一代（`layout.json` 的 `precision`；没有就是问题的精度）不同时，
  驱动先把这一代按调用的精度改写成下一代（`precision.convert_checkpoint`）：升高时 limb 补零，数值不变；降低时每个数只留最高的那些 limb，
  与 GMP 浮点数自身的截断方式相同（512 位时相对误差 2^-576，远小于调用能留住的位数）。检查点已是全精度的运行再打开混合精度，
  从梯级的选择开始。续算从最后一条记录的 `precision_next` 接着；最后一条记录处于 confirm 时，从它前一代检查点重做那次调用。
- 生成器也用这次调用的精度（请求里的 `blockPrecision`，见 `pmp.model_section`：块表和它的算术都用；问题设了 `block_precision` 时不超过它）；
  全精度时请求与不开混合精度时相同，PMP 是同一个文件。各精度的块表放在各自的目录 `blocks/p<位数>` 下：块表文件名里没有精度，而生成器只在文件不存在时才建表
  （修正之前，同一点先 512 位、再全精度时，全精度那次读的是 512 位的表）。精度升高时重新测一次 `pmp2sdp`
  的内存（或取 bench 里的行，见第 9 节）。

### 10.2 动态大步（`large_steps = true`）

默认关。下一点取 x + m·Δx 而不是 x + Δx，m 在每次走步的调用之后，由这条记录和上一条走步记录决定（`jump.py`；规则沿用此前研究中实测过的做法，未改动）。外部步的长度由间隙决定，与离边界多远无关；在路径的直段上，
多走 m 步每步只多花约 1.2 次迭代，而一次调用的固定开销是整次调用。

- c = cos(Δx/α, 上一次的 Δx/α)，α = `alpha_d`。冷却中、c ≤ 0.99 或这次爬升了，m = 1；c ≤ 0.999 时 m = 2；否则 m = ⌊0.5/α⌋，
  限在 [1, 8]；x + m·Δx 出了盒子就取 1。
- 一次 m > 1 的大步之后的那次调用：`findMinimumQ` 由假变真（路径离开了允许区）→ `stopped: findMinimumQ changed to true after a step m
  times ahead (left the allowed region)`；由真变假（进入允许区，是从区外开始的运行的正常路径；N 同时变号）→ 接下来 2 次调用 m = 1；
  除此之外 N 变号 → 这次
  运行不再走大步（设了 `large_steps_sign_cooldown = k` 时只让接下来 k 次调用 m = 1）；迭代超过 3 + 1.5(m − 1)，或 c < 0.98 → 接下来 2 次调用 m = 1。
- 传给 BFGS 的步长是 m·Δx；记录里 `ext_step` 仍是求解器给的 Δx，另记 `jump` = m；续算从 x + m·Δx 接着。
- 与混合精度同时开：大步之后的那次调用按大步自己的迭代预算，不进健康检查。
- 实测（一个较大的问题，N ≈ 3000、约 270 块；同一起点，各 13 h）：1024 位时推进是不跳的 1.84 倍。每个点仍是求解器收敛到的点：大步走坏了只费
  时间，不影响结果。
- 在 bootlab 上，10.1 里那个较大的问题（N ≈ 2300；512 位、5 步）：大步对不跳每小时推进 3.45 倍，对全精度不跳 4.3 倍；m 依次为 1、3、6、4、3，
  没有触发冷却。步数少，只作参考。
- **已知限制**：N 一变号，这次运行就不再走大步，哪怕只是短暂变号。512 位时（同一问题、同一起点）13 h 只快 1.32 倍：第 27 条 N
  短暂变号，下一条又变回来，大步在第 4.5 h 就停了。改成冷却若干次，要等更长的实跑试过再定。

## 11. 检查点、留下的一代、续算

- `dynamical_sdp` 的 `--checkpointDir` 就是 `<运行目录>/checkpoint/`（各节点共用）。SDPB 写第 g+1 代时先由各 rank 写自己的文件，然后写
  `checkpoint_new.json`，再原子地改名为 `checkpoint.json`，只留两代；所以任何时刻被杀，`checkpoint.json` 指的那一代都完整，但它可能是
  一次调用中途的那一代。
- 所以每次调用结束后、写它的第 k 条记录之前，bootlab 建 `checkpoint/kept/<k>/`：这一代各 rank 文件的硬链接，`checkpoint.json` 和
  `block_timings` 的拷贝，以及 `layout.json`（`nodes`、`node_type`、`ppn`、`precision`、`generation`）；先在 `kept/<k>.new` 里建好再改名。记录写完后，
  k ≤ 记录数的 kept 只留最新的 `[skydive] keep_checkpoints` 个（不写是 2，至少 1），更早的删掉；续算用最新的一个，其余是留着手动回退用的。
  所以任何时刻至多 keep + 2 代（另加求解器自己的当前代和备份代），每条记录写完后 min(记录数, keep) + 1 代。混合精度只用到最新的一个：
  optimal 之后的全精度重解和升精度，都是把最后一条记录留下的那一代转换成新文件（10.1）。
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
| `jump`、`jump_cooldown`、`jump_note`、`jump_stop` | 只在动态大步时有（10.2）：m、还要冷却的调用数、原因、运行停下的原因；也有 `call_iterations`、`first_call` |

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
`site gen` 只放探测过的型号，写 `[slurm] partition = P`（手写文件的 `[slurm]` 优先），旁注写明来源（`partition from the probe plan` 或 `hand`）。`probe plan --node H` 在 H 上探测，不管它忙不忙。

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

`bootlab regress [--site] [--node] [--time] [--dry]` 把 `examples/ising`（Λ = 11、三步）写到 `<runs>/regress-input-<时间>-<进程号>/`（同一秒交两次也不重名），交一个整机作业运行
`regress run`；`regress check <运行目录>` 可以比较任何一个运行目录。参考值是 `tests/reference/ising-l11/bt-steps.json`（参考程序前三步的全精度值）。
每个数要一致 ≥ 130 位有效数字（用 `decimal` 以 500 位精度计算，只用标准库）；离散量必须相等。运行目录里写 `regress.json`（`passed`、`least`、
每步的位数、问题）。

单元测试：`bin/bootlab selftest`（全部单测，等同 `python/bin/python3 -m unittest discover -s tests`；`install.sh` 会跑它，
失败只告警，安装闸门本身只查各模块能导入、`bin/bootlab` 能启动和两个快的单测 `tests/test_install.py`、`tests/test_regress.py`；不需要集群；站点相关的测试用 `tests/fixtures/sites/` 里的中性夹具）。
`tests/test_physics.py` 拿参考程序参数检查的录下输出（6 个模型、14 组用例）核对物理层。

## 17. 扩展 bootlab

- **新算法**（Navigator、可行性……）：在 `skydive.py` 旁边加一个模块，由它产生点、读取结果；复用 `pmp.write_request`/`pmp.generate`（接口一）、
  `execute.build_sdps`，再为它的求解器（`sdpb`、`approx_objective`）写一个执行函数；写自己的记录行；在 `cli.py` 里加子命令。规矩照旧：输入在
  `inputs.load` 里检查，遇到异常结果就结束，资源按 bench 数据选，投入使用前与参考结果对照。
- **新生成器**（带自旋的算符）：任何能读第 6 节那样的请求、写出第 7 节那样 PMP 的程序，都可以经站点的 `generator` 接进来；自带模型和 PMP
  格式的生成器按第 19 节放在 `[generators.<名>]` 下。
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

<a id="current-bootstrap"></a>
## 19. 带自旋的算符：内置模型、逐点扫描与 Skydive（current bootstrap）

DSD 处理带自旋外部算符的 Haskell 代码（stress-tensors-3d，`jt-z2-charge-conjugation-invariant` 分支 478a3fd，基于
hyp-1.3.36）作为第二个 PMP 生成器接进来。物理（三点结构、守恒、Ward 恒等式、交叉方程）原样留在上游的 Haskell 里；可替换的部分仍是数据。

**内置模型。** 这类模型用模型卡 `bootlab-builtin/1`（JSON）代替 `bt-model/2`：`generator`（站点 `[generators.<名>]` 下的名字）、
`bound`（生成器登记的 Bound，例如 `JJJJ3d`）、`coordinates`（Bound 自带的坐标，⟨JJJJ⟩ 没有）、`channels`（谱可以用的 channel 及其自旋规则）、
`objectives`（允许的目标种类）、`lambda`（外部 OPE 方向的维数）。加一个上游 Bound = 在生成器的登记表
（生成器源码的 `StressTensors3d.Bootlab.Pmp` 模块）里加一行，再写一张卡。例子：`examples/jjjj/`。

- 谱：与 3.2 节相同的 `bootlab-spectrum/1`。⟨JJJJ⟩ 的 channel 是宇称扇区 `P+`、`P-`；`delta` 对应上游的 `Fixed`，`above_unitarity` 对应
  `RelativeUnitarity`。
- 目标函数：`kind = "feasibility"`（`lambda` 可选）或 `"ope_bound"`（`lambda` 和 `direction = "upper" | "lower"`）。`lambda` 可以引用点参数，
  例如 `["1", "x"]`；引用到的参数成为坐标。
- `run.toml`：`[numerics]` 恰好是上游 `SimpleBoundParams` 的五项：`lambda`（nmax = (Λ+1)/2）、`order`、`num_poles`、`max_spin`
  （自旋 0…max_spin）、`block_precision`。`[scan] points` 是点的列表，每个点按坐标给精确值（字符串，不收浮点数）。`[solver]` 按 flag 名覆盖
  求解器参数（给值就替换，`true` 加开关，`false` 去掉）。内置模型有 `[scan]` 或 `[skydive]`，二者取一；`bt-model/2` 模型带 `[scan]` 是错误。

**接口一′：生成器输入。** `stress-tensors-3d-exec pmp <request.json>`：Bound 名、截断参数、每个点的谱（自旋已展开）和目标函数（精确的 JSON
数字）、输出目录、block 目录、核数。每个点的 Bound 由上游程序自己的构造函数建（⟨JJJJ⟩ 是 `JJJJ2025.jjjj3dBound`），所以键、精度、求解器参数都与
上游相同。所有点一次交给上游的 `buildBounds`，任务图停在 PMP 部件。每个点写 `pmp.nsv`、`sdpb.args.json`（上游会传给 sdpb 的参数）和
`bound.json`（键）。生成器在宿主机上运行，环境取 `[generators.<名>.env]`；bootlab 把它调度器的内存设为本步的 `--mem`，并声明没有节点本地路径
（上游会删掉节点本地路径上的所有输出，最终的部件也在内）。

**接口二：PMP 是 SDPB 3.1 格式。** 生成器写 SDPB 3.1 的 JSON（`prefactor`；Bound 设了 `num_poles` 时还有 `reducedPrefactor`，即 3.1 的降采样，
改写成 3.0 就是另一个 SDP）。所以不做转换：站点的 `[generators.<名>]` 写明它用的 `pmp2sdp` 和 `sdpb`（3.1.0），这两个程序必须在站点的
镜像里（bootlab-exec 从 r6 起同时带 3.0sky 和 3.1.0）。标量链不变（3.0sky）。

**逐点扫描（scan）。** 每个点求解一次：生成器 → pmp2sdp → sdpb → `solves.jsonl` 一行（点、primal、dual、gap、N、块数、进程数、用时）。
block 目录在整个运行内共用。求解器参数取生成器写的 `sdpb.args.json`，`maxRuntime` 改为作业剩余时间减 15 分钟，再叠 `[solver]`。失败直接结束；
`resume` 从第一个没有记录的点接着算。命令不变（`run`、`resume`、`summary`）；scan 只在一个节点上跑，scratch 的选法与 Skydive 相同；求解器进程数
按第 9 节从 bench 选（没有实测就用全部物理核）。

**Skydive。** 用的是第 10–12 节的同一个循环（记录、检查点、续算、结局都不变；代码是 `builtin.Skydive`，继承 `skydive.Run` 并覆盖几个钩子），只换四处。
- PMP：中心点和邻点由 `builtin.write_request`、`builtin.generate` 一次建好，SDP 用生成器指定的 `pmp2sdp`。生成器的输出放在运行级目录 `<scratch>/pmp`，步与步之间（scan 是点与点之间）保留，调度器跳过已有的部件，每一步只建与点有关的部件；换节点或换作业续算时从头建。
- 求解器：站点的 `dynamical_sdp`；`[generators.<名>]` 可写 `dynamical_sdp` 换掉。3.0sky 能读 pmp2sdp 3.1.0 建的 SDP。
- 求解器参数：以 Bound 自己的（中心点的 `sdpb.args.json`）为底，之上是 `maxRuntime`、`[solver]`、本步的停止阈值。阈值的设定值是 Bound 的，只有生成器给得出，所以下一步的 PMP 先建好、读出参数，再定选项。精度是 Bound 的，即 `block_precision`；每一步都核对生成器写的精度等于调用的精度、其余参数与第一步相同，不一致就以失败结束。
- 目标函数：dynamical_sdp 在坐标上把 SDP 的目标值 N 取到极小，所以只收 `ope_bound`。上界方向时 dual 等于界取负，Skydive 走到界最大的点。N 的符号不表示允许或排除：`find_boundary_direction`、`stop_on_negative` 是错误，以 optimal 结束记为 `converged: optimum`。

`large_steps` 照用；`mixed_precision` 时生成器也在调用精度上建 Bound（请求的 `precision` 同时决定 block 与求解精度）。坐标是模型卡的 `coordinates`（写进请求的 `coords`），再加谱和目标函数用到的点参数；⟨JJJJ⟩ 只有点参数，例子 `examples/jjjj/skydive.toml` 走 x = 12γ。以后有带外部维数的 Bound，坐标写进卡里，接口不变。

**已验证**（与上游自己对同一 Bound、同样几个点的运行对照）：d = 3 的 ⟨JJJJ⟩，Λ = 11（nmax 6、order 60、20 个极点、自旋 0–40、768 位），
`ope_bound` 上界、`lambda = ["1", "x"]`，一个 `P+` 自旋 2 的 gap 3.5，12γ = 0、−1、+1 三个点，单节点。63 个 PMP 部件与上游逐字节相同；
pmp2sdp 3.1.0 建的 SDP 126 个文件里 124 个逐字节相同，另两个（`control.json`、`pmp_info.json`）只差路径；三个点的 dual 一致至少 172.9 位、
primal 至少 186.6 位（间隙约 9e-31），N = 116、61 个块两边相同（每次验收都如此，最近一次在 bootlab-exec r6 上）。

**已验证（Skydive，r7）**：⟨JJJJ⟩ Λ = 11，坐标 x = 12γ，上界。dynamical_sdp 3.0sky 读 pmp2sdp 3.1.0 的 SDP，解到最优时与 sdpb 3.1.0 一致 31–32 位（两边间隙约 3e-31）；Skydive 的 dN/dx 与 sdpb 3.1.0 在 x ± 1e-10 的中心差分一致 19.5–19.6 位、符号一致；从 x = −0.11 起 60 步以 `converged: optimum` 结束（中间续算一次），终点两侧 ± 1e-6、± 1e-10 的 B 都更小，差分斜率约为 0，差分曲率与 BFGS 的 Hessian 一致 6 位；按上游源码独立复现的循环前 5 步与 bootlab 一致 ≥ 134 位；`large_steps` 从 x = 0.5 起 30 步到达极大点附近（不开大步 30 步只到 0.34）；`mixed_precision` 12 步 512 位，与全精度一致 78–86 位，高于 10.1 节公式的 55–57 位。

**已知行为**：离极大点远、N 是凹的一段，BFGS 不更新 Hessian，外部步只有 Newton 步的 0.5–0.8%，间隙每步只降约 1.5%：⟨JJJJ⟩ 从 x = 0.5 起不开大步，60 步只走到 0.22。开 `large_steps` 能绕过去，30 步就到了极大点附近，所以从远处起跑建议打开（全局默认仍是关，`examples/jjjj/skydive.toml` 打开了）。到了极大点附近，α ≈ 0.78，间隙每步降到约 0.3 倍，几十步就收敛。

**未验证**：`feasibility` 目标；孤立算符；`P-` 扇区的 gap；其他 Λ；其他上游 Bound（都还没登记：TTTT3d 要先给 `[numerics]` 加半整数自旋表）；
生成器的单机模式（目前只用 srun 启动）；带外部维数的 Bound 的 Skydive（还没有登记这样的 Bound）；内置模型的 Navigator；内置模型 Skydive 的
多节点。

**求解器进程数**（在 12γ = 0 实测，取物理核的 1/4、1/2 和全部，各解一次；见站点的 bench 文件）：scan 选最快的。在开发用的集群上，
最大的节点（256 物理核）是用一半的核最快，较小的节点（96、128 核）是用全部核最快。只测了 Λ = 11 的 ⟨JJJJ⟩；bench 先在求解程序同一 SDPB 版本（3.0sky、3.1.0）的记录里按 N 和块数选最接近的已测问题，该版本没有记录才看全部。
