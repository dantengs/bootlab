# 设计稿：可行性判定，以及在它之上的二分、Delaunay 与 navigator

**状态：设计稿，尚未实现。** 本文写给接手实现的人：读完本文和 [interfaces.md](interfaces.md)，应当能动手写第一阶段。文中“现状”一节描述的是已有代码；其余各节是设计。设计里要项目负责人定的事，已在 2026-11-15 定下，正文按决定写，第 10 节列出每一项的决定。

[English](../en/design-search-algorithms.md)

目录：[0 约定](#0-读者与约定) · [1 现状](#1-bootlab-现状) · [2 可行性判定](#2-可行性判定一个点的判定原语) ·
[3 搜索外壳](#3-三种算法共用的搜索外壳) · [4 二分](#4-二分) · [5 Delaunay](#5-delaunay-扫描) ·
[6 navigator](#6-navigator) · [7 OPE 扫描](#7-ope-扫描切割面法) ·
[8 参考实现与取舍](#8-参考实现借鉴了什么为什么取舍) · [9 分阶段计划](#9-分阶段实施计划) ·
[10 决定与风险](#10-决定和风险) · [11 贡献指南](#11-贡献指南)

## 0. 读者与约定

**要先读的。** [interfaces.md](interfaces.md) 第 1–12 节（三层、两个接口、Skydive 循环、检查点、记录），第 19 节（逐点 scan）。会写 Python、知道 SDPB 的原始–对偶问题和 navigator 函数（arXiv:2104.09518）就够了，不需要读过 simpleboot 或 Hyperion 的源码：本文把用到的行为都写出来了。

**术语。**

| 词 | 含义 |
|---|---|
| 点 | 坐标名到十进制字符串的映射，坐标次序是 `run.toml [coords].names`（interfaces 3.4） |
| 允许 / 排除 | 这个点上的谱假设与交叉方程相容 / 不相容。排除的证据是一个对偶泛函 |
| 判定 | 一个点的结果：允许、排除、未定、跳过，或者没算完（第 2.3 节） |
| 槽位 | 一台节点上的一组物理核和内存，一次算一个点（第 2.9 节） |
| N | navigator 函数的值。bootlab 的两种 navigator 归一化（Σ 与 GFF，interfaces 3.3）都是 N < 0 为允许 |
| 一次求解 | 一个点的一次 `sdpb` 调用（navigator 还要加上 `approx_objective`） |
| 参考程序 | 回归对照用的、基于 Hyperion 的驱动程序（interfaces 16） |

**项目规则**（实现时每一条都要遵守，第 11 节有具体写法）：

1. 物理结果第一。数值对照通过以前不算完成。
2. 输入只在 `inputs.load` 检查一次；之后各层信任输入，出错直接停。不做防御性检查，不自动重试。
3. 异常直接结束：迭代上限、时间上限、停滞、程序失败，都结束运行，写下原因；不挂起，不当作收敛。
4. 资源配置只从实测数据定（bench 文件、首步实测）；没有测过就说没测过，用保守的默认（整节点）。
5. 只用物理核（`--hint=nomultithread`，`OMP_NUM_THREADS=1`）。
6. 执行层与输入层脱耦：镜像只接受非物理的数据；模型、谱、目标函数、搜索区域都写成数据。
7. 数都是精确的十进制字符串，TOML 浮点数一律拒绝。
8. 只学思路，不抄代码：simpleboot 和 Hyperion 的算法思路可以借鉴，接口和命名按 bootlab 自己的设计；算法策略以 simpleboot 的行为为参照。
9. 辅助功能写简单。效率优化只加在实测出来的瓶颈上，而且不能改变数值结果。

本文引用的实测都来自 bootlab 的开发记录和对 simpleboot、Hyperion 生产运行的分析，写成一般性描述。标“玩具”的数字来自本机的小型模型（二维单关联函数的格点线性规划、n=20 的随机稠密 SDP），只能说明方向，不能直接搬到真实问题。

---

## 1. bootlab 现状

### 1.1 三层，两个固定接口

| 层 | 做什么 | 代码 |
|---|---|---|
| 物理层 | 模型（`bt-model/2` JSON）、谱假设（`spectrum.toml`）、目标函数（`objective.toml`），都是数据 | `examples/*/` |
| 转换层 | 读入并检查输入；在每个点把谱算成精确数；写生成器的输入；外层循环（现在只有 Skydive 和 scan）；记录 | `src/bootlab/*.py`，Python 标准库 |
| 执行层 | 生成器写 PMP，`pmp2sdp` 转 SDP，`dynamical_sdp` 或 `sdpb` 求解 | 一个 Apptainer 镜像里的程序，由 `srun`（或本机模式的 `mpirun`）启动 |

两个固定接口（interfaces 6、7），新算法都接在这里，**不改它们**：

1. **生成器输入 `request.json`**：模型文件加上每个点的坐标和算好的谱。`pmp.write_request(problem, model_file, points, out, blocks, cores, bits)` 写它，`pmp.generate(site, node, request, log)` 调用生成器。`points` 是“点名 → 点”的字典，一次请求可以有任意多个点，相同的 block 表只算一次。目标函数由 `objective.toml` 的 `kind` 选（模型的 `objectives` 里有 `feasibility`、`navigator` 等变体），整个运行用同一个。
2. **PMP**：SDPB 3 标准的 PMP JSON 部件，每个点一个清单 `<out>/<点名>/pmp.nsv`。`pmp.pmp_lists(out, names)` 取清单。

### 1.2 Skydive 的外层循环

`skydive.Run` 是现在唯一的完整算法，新算法的骨架照它来。

- **一步**（`Run.step`）：`Run.points` 由当前点 x 造出中心 `c` 和 k 个邻点 `plus_i`（x + ε eᵢ，精确十进制）；生成器一次写出这 k+1 个点的 PMP；`Run.build_sdps` 分批跑 `pmp2sdp`（第一步单独转中心点、测内存，之后按 `bench.pmp2sdp_at_once` 并发，看门狗在 95 % 时取消整批）；`execute.run_dynamical_sdp` 求解；`execute.read_dive` 读出 `Dive`。
- **判断**（`Run.loop`）：`aborted` → 结束（`unfinished` 或 `failed`）；`optimal` → `skydive.optimal_ending` 看原始、对偶目标是否同号，同号才判 `converged: allowed/excluded`，否则 `stopped: N undecided at optimal`；出盒子、步数用完、N<0 即停等各有自己的结局。没有任何重试。
- **记录**：每次求解在 `solves.jsonl` 写一行（`Run.record`），全是全精度字符串，不带钟点，所以两个相同布局的运行逐字节可比；`result.json` 写结局、输入的 sha256、程序版本（`Run.finish`）；`bootlab summary` 生成 `summary.md`。
- **检查点**：求解器直接写 `<run>/checkpoint/`；每条记录之前，`execute.keep_generation` 用硬链接把这一代保留为 `checkpoint/kept/<k>/`（带 `layout.json`：节点、机型、每节点进程数、精度）；`execute.prune_kept` 在记录写完后删旧的。
- **续算**（`cli.cmd_resume`）：`cli.place_checkpoint` 取不超过记录数的最新一代，布局相同就恢复（`execute.restore_generation`），否则冷启动；`skydive.resume_state` 从最后一条记录恢复点和 BFGS 状态。续算要手动发起，不自动重来。
- **默认求解器参数**：`skydive.solver_defaults(nmax)`，与参考程序在同一 nmax 下传的值一致（单测固定）。

### 1.3 资源与布局

- **站点**（interfaces 13）：`sites/<name>/site.toml`，由 `bootlab probe` + `bootlab site gen` 从实测生成；`execute.load_site` 读入。
- **bench**（interfaces 9）：`sites/<site>/bench/<机型>.json`，`bootlab bench` 写入。`bench.solver_procs(bench, cores, n, blocks, program)` 在同一 SDPB 版本的 `solver` 行里找规模最接近的问题（距离 |ln N/Nᵢ| + |ln 块数/块数ᵢ|），取最快的进程数，只取网格不退化的（`bench.grid_ok`）；没有数据就用全部物理核。进程数在一次运行里固定，因为二进制检查点按秩存，换进程数就读不了。
- **pmp2sdp**：`bench.pmp2sdp_procs`、`bench.pmp2sdp_at_once`、`bench.sdp_build_row`；或者首步实测。
- **节点**：`layout.choose_nodes` 按 bench 的多节点效率决定一个求解用几台；`layout.resume_plan` 续算时保持检查点的布局。一个求解只用同一机型。
- **本机模式**（interfaces 18）：没有 Slurm 时 `execute.launch` 用镜像里的 `mpirun` 和 rankfile；并发的程序拿到互不重叠的物理核（`execute.local_argv`）。物理、记录、续算与集群上相同。

### 1.4 已有的逐点 scan

`scan.Scan`（interfaces 19）是现在唯一“一个点一次 `sdpb`”的路径，但只给内置模型（`bootlab-builtin/1`，自旋外算符）用：

- 每个点：`builtin.write_request` → `builtin.generate` → `execute.build_sdps` → `execute.run_sdpb`（参数取生成器写的 `sdpb.args.json`，再叠 `[solver]`，`scan.solver_args`）→ `execute.classify` → 一行 `solves.jsonl`。
- 失败就结束；`resume` 从第一个没有记录的点接着算。
- 一次一个点，整节点，冷启动，不保留检查点。
- 记录里的 `N` 是 **SDP 的规模**（`execute.sdp_size`），不是 navigator 值。新记录格式要避开这个名字冲突（第 2.5 节）。
- `feasibility` 目标在内置模型上**没有验证过**。

### 1.5 新算法能直接复用的

| 已有 | 新算法里的用途 | 要不要改 |
|---|---|---|
| `inputs.load`、`resolve_spectrum`、`spectrum_warnings` | 读入、检查、在点上算谱 | 加新节的检查 |
| `pmp.write_request`、`pmp.generate`、`pmp.pmp_lists` | 一批点一次生成 | 不改 |
| `Run.points` | navigator 的梯度邻点 | 移成模块级函数，Skydive 调用它（行为不变） |
| `Run.build_sdps` 的实测与分批 | 所有新算法转 SDP | 移成共用函数（行为不变，回归验证） |
| `execute.run_sdpb` | 一次 `sdpb` | 加 `--initialCheckpointDir`（热启动） |
| `execute.classify` | 终止原因分类 | 不改；判定另写一张表（第 2.3 节） |
| `skydive.optimal_ending` | 符号判定 | 直接用（不传 boundary） |
| `execute.keep_generation` 一族 | 保留每个点的检查点 | 照样用，以点号代替记录号 |
| `bench.solver_procs`、`grid_ok` | 每个槽位的进程数 | 加一个按吞吐量选的函数（第 2.9 节） |
| `execute.choose_scratch`、`with_binds`、`clear` | 暂存目录 | 不改 |
| `cli.cmd_run`、`cmd_resume`、`write_summary` | 按 `run.toml` 的节分派 | 加分支 |
| `regress.check` 的位数比较（`regress.digits`） | 对照参考程序 | 复用 |

---

## 2. 可行性判定：一个点的判定原语

### 2.1 一个点的判定是什么

在点 x 上，谱假设给定后，交叉方程写成 SDPB 的多项式矩阵规划。SDPB 的“对偶”问题求一个泛函 y，使它作用在每个允许出现的算符上都非负、作用在归一化上为 1；“原始”问题求一组非负的 OPE 系数（矩阵）满足交叉方程。

- 找到对偶可行解 = 存在排除这个谱的泛函 = **排除**。
- 找到原始可行解 = 截断后的交叉方程有解 = **允许**（在这个截断 Λ 下）。

这两种结论在精确算术下互斥。实际求解是有限精度的内点法，所以判定要看求解器以什么理由停下（第 2.3 节）。

### 2.2 两种判定方式

| 方式 | 目标函数 | 求解器在什么时候停 | 代价 | 给出什么 |
|---|---|---|---|---|
| **跳跃判定** `decide = "jump"`（默认） | 模型的 `feasibility` 变体：目标为空，归一化为单位算符 | 原始或对偶走出一个步长为 1 的步（“feasible jump”） | 通常比解到最优少（bootlab 上未测） | 允许/排除 |
| **符号判定** `decide = "sign"` | 模型的 `navigator` 变体 | 对偶间隙小于阈值（optimal），然后看原始、对偶目标是否同号 | 解到底 | 允许/排除/未定，以及 N |

**跳跃判定的求解器参数**照 Hyperion 的 `jumpFindingParams(nmax)`：在按 nmax 分档的基本参数上，`--detectPrimalFeasibleJump`、`--detectDualFeasibleJump` 打开，`--findPrimalFeasible`、`--findDualFeasible` 关掉，原始、对偶误差阈值设成 1e-200（这样求解器不会因为误差够小而先停，只会在跳跃时停）。按 nmax 分档的那部分（精度、初始缩放、`maxComplementarity`、检查点间隔）与 `skydive.solver_defaults` 同源，写一个 `point.jump_defaults(nmax)`，单测固定，`[solver]` 可以覆盖。SDPB 的说明里讲，跳跃说明“精度足够时会找到可行解”，所以判定与精度有关：**同一个问题的所有点用同一精度**，精度写进每条记录。

**符号判定**就是 navigator 的一次完整求解：N 落在 [对偶目标, 原始目标] 里，两者同号才判定（`skydive.optimal_ending` 的规则）。它贵，但给出 N，可以当作 navigator 的“认证”，也用来复核跳跃判定。

模型要有对应的目标变体。注意 `examples/ising` 的模型带 `opeAngle`：外部 OPE 是 (cos θ, sin θ)，θ 是坐标，所以在 (Δσ, Δε) 平面上做可行性扫描时 θ 是固定的。要让 OPE 比值不定，模型里 `external` 不写 `opeAngle`（j×j 半正定约束），这是另一个模型文件，不是 bootlab 的开关。

### 2.3 终止原因到判定

`sdpb` 的 `terminateReason` 原样记下；判定按下表，写在一个函数里（`point.verdict(reason, decide, primal, dual)`），表外的原因直接让运行停下（同 `execute.classify`）。

| `terminateReason` | 跳跃判定 | 符号判定 |
|---|---|---|
| `primal feasible jump detected`、`found primal feasible solution` | **允许** | 不应出现：结束，`failed: unexpected <reason>` |
| `dual feasible jump detected`、`found dual feasible solution` | **排除** | 同上 |
| `found primal-dual optimal solution` | **排除**（与 Hyperion 相同，第 10 节 D2） | 原始、对偶同号：允许或排除；异号：**未定** |
| `maxIterations exceeded` | 没算完：`unfinished`，按下面的第二张表处理 | 同左 |
| `maxRuntime exceeded`、`SIGTERM signal received` | 作业到时或被停：`unfinished`，结束运行 | 同左 |
| `maxComplementarity exceeded`、`primal step too small`、`dual step too small` | 停滞：`stalled`，按下面的第二张表处理 | 同左 |
| 程序非零退出或没有 `out.txt` | `failed: <程序> exit <码>`，结束运行 | 同左 |

和参考实现的异同：

- **Hyperion** 的默认判定是“不是对偶可行就算允许”，所以 `maxComplementarity`、`maxIterations` 都会被当作**允许**。bootlab 不这样做：没算完、停滞都没有给出证据，不算判定。`primal-dual optimal` 则与 Hyperion 一样算作对偶可行，即排除。跳跃判定的参数把误差阈值设成 1e-200，正常情况下求解器在跳跃时就停了，走到 optimal 的多半是离边界很近的点；需要的话可以用符号判定复核（第 10 节 R1）。
- **simpleboot** 把这些“其他”结果记下来但不参与判定，于是二分和 Delaunay 下一轮会原地提出**同一个点**，一直算到点数用完。bootlab 对这样的点只有两种处理：结束搜索，或者记为跳过、永不再提，不会原地重复。

**没算完或停滞的点怎么处理**（项目负责人 2026-11-15 定）：

| 情形 | 可行性类（点表、二分、Delaunay、OPE 扫描） | navigator |
|---|---|---|
| 冷启动的点 `maxIterations` 或停滞 | **跳过**：判定记为 `skipped`，`reason` 写求解器的原因；这个点永不再提，搜索继续 | 结束整个搜索 |
| 热启动的点 `maxIterations` 或停滞 | 结束整个搜索：原因可能是热启动本身，跳过会丢掉一个本来能算的点；靠热启动的距离门槛预防（第 2.6 节） | 结束整个搜索 |

`maxRuntime`、`SIGTERM` 说明作业到时或被停，不是这个点的问题，所以不跳过：运行结束，续算时这个点用自己的检查点重算。各算法怎样对待跳过的点，写在各自的一节（第 3.6、4.2、5.2、7.3 节）。

“未定”不是异常，是一种判定：它说明这个点在求解器的分辨率内贴着边界。现在只有符号判定会给出未定。各算法怎么用它，各节写明；默认都是结束搜索、写明原因（`stopped: undecided point <id>`），由用户收紧阈值或改精度后续算。

### 2.4 点判定原语 `point.py`

三种算法和已有的 scan 都通过它求点，它不知道自己在为哪个算法工作。

```python
@dataclass
class PointJob:            # 一个要算的点
    id: int                # 运行内唯一，按提出的先后编号
    x: dict                # 坐标名 -> 十进制字符串
    purpose: str           # "scan" | "bisection" | "delaunay" | "navigator" | "line_search" | ...
    extra: dict            # navigator 的梯度邻点等（第 6.2 节）；可行性判定为空

@dataclass
class PointResult:
    id: int; x: list        # 坐标次序
    verdict: str            # "allowed" | "excluded" | "undecided" | "skipped" | "aborted"
    reason: str             # terminateReason 原文，或 "unfinished: ..."、"stalled: ..."、"failed: ..."
    primal: str; dual: str; gap: str
    navigator: str | None   # 符号判定时的 N（取原始目标，与 simpleboot 一致）；跳跃判定为 None
    gradient: list | None   # navigator 才有（第 6.2 节）
    iterations: int
    warm_from: int | None   # 热启动所用点的 id；冷启动为 None
    size: dict              # {"n": SDP 的 N, "blocks": 块数}
    procs: int; slot: int; node: str
    precision: int
    seconds: dict           # pmp、sdp、solve，以及 approx（navigator）
```

**一轮**（`point.solve_round(node, jobs)`）是**一台节点**上的一轮。照现有规则“三段硬计算轮流占满整个节点”，一轮里：

1. 把这台节点这一轮的所有点（navigator 还有它们的邻点）放进**一个**生成器请求：`pmp.write_request` + `pmp.generate`，在这台节点上用它的全部物理核。相同的 block 表只算一次，block 目录在这台节点的暂存盘上。
2. `pmp2sdp` 在这台节点上分批（共用 `Run.build_sdps` 移出来的函数，第一轮实测内存）。
3. 这台节点的每个槽位一个 `sdpb`，同时跑（第 2.9 节），`--initialCheckpointDir` 指向热启动点的检查点（第 2.6 节）。
4. 读 `out.txt`，按第 2.3 节判定；保留检查点、删掉这个点的大文件（第 2.6 节）；返回结果。

多台节点各自一轮一轮地跑，互不等待（第 2.9 节）。同一台节点内，各点的 `sdpb` 用时不同（跳跃判定的迭代数差别很大），先算完的槽位要等这台节点的这一轮结束。这是有意的简化：每轮记下槽位的空等时间；如果实测空等占得多，再改成槽位各自流水（那时要先确认两个生成器进程同时写一个 block 目录是安全的，现在没有验证，第 10 节 R3）。

### 2.5 记录格式

一个运行目录里：

| 文件 | 内容 |
|---|---|
| `queue.jsonl` | 每提出一个点写一行：`id`、`x`、`purpose`、`round`、`extra`。只追加 |
| `points.jsonl` | 每算完一个点写一行：`PointResult` 的全部字段。只追加，不带钟点 |
| `search.json` | 算法的当前状态（第 3.3 节），每轮结束原子地重写（写临时文件再改名）；只是给人看的副本，续算不读它 |
| `result.json` | 结局（`status`、`reason`）、各算法的结果（第 4–6 节）、输入的 sha256（含搜索节，第 3.5 节）、程序版本、`start`、`end` |
| `summary.md` | `bootlab summary` 生成：点表、判定统计、结果 |
| `driver.log` | 带时间的决策日志，每条命令行，每个程序 stderr 的尾部 |
| `points/<id>/` | 求解器的小输出（`out.txt`、`.cmd`、`.out`、`.err`）；navigator 的梯度文件 |
| `checkpoint/<id>/` | 第 2.6 节保留的检查点 |
| `files.jsonl` | 文件管理的记录：每轮各类文件的占用，每次删除（第 2.6 节） |

新记录用 `navigator` 表示 N、`size.n` 表示 SDP 规模，避免和 scan 记录里的 `N`（SDP 规模）混淆。已有的 scan 记录格式不改。

### 2.6 热启动、检查点与文件管理

这一节的规则用于所有新算法；OPE 扫描文件多，第 7.4 节在此之上再加几条。

**热启动选哪一个检查点**（`[point] warm_start = "nearest"`，默认）：在保留着的检查点里，取对应的点离新点**缩放后**欧氏距离（第 3.2 节的 `[coords].scale`）最近的一个；一样近的取较新的。距离超过 `warm_radius` 就冷启动。simpleboot 和 Hyperion 都从最近点热启动，区别在于：simpleboot 不管多远都用，Hyperion 的参考驱动在热启动超过 400 次迭代后改为冷启动重算。生产数据显示，从远处热启动可以比冷启动慢 3–6 倍，所以要有距离门槛；但“超了就冷启动重来”是一次自动重试，不做（第 10 节 D4）：热启动的点超过 `maxIterations` 就结束搜索、记下原因，由用户调小 `warm_radius` 后续算。

**从多深的检查点热启动**：simpleboot 的生产运行在 μ 降到某个阈值时另存一份中间检查点，从它热启动；阈值 1e-10 到 1e-15 时，热启动的迭代数是冷启动的 0.20–0.38 倍，阈值 1e-8 时是 0.64–0.88 倍。镜像里的 `sdpb` 有没有这个选项，要在第 0 阶段用 `--help` 确认；没有的话就从最终检查点热启动（参考驱动的做法）。

**布局**：二进制检查点按秩存，所以只能在同样进程数的槽位上读。一个运行的所有槽位用同样的核数（第 2.9 节）。检查点写在运行目录（共享盘）里，同 Skydive，所以任何一台节点上的槽位都能用别的节点留下的检查点。

**保留多少**（第 10 节 D8）：

| 算法 | 默认保留 | 怎样改 |
|---|---|---|
| navigator | 最近 2 个求值点的检查点 | `[point] keep_checkpoints` |
| Skydive | 2 代（interfaces 11） | `[skydive] keep_checkpoints`，默认 2 |
| 可行性类（点表、二分、Delaunay、OPE 扫描） | max(20, ⌈1.5 × 并发数⌉)，例如并发 30 个保留 45 个 | `[point] keep_checkpoints` |

可行性类的默认随并发数放大，因为同时在算的点越多，下一批点的热启动起点就分布得越散。

**删除顺序**。navigator 的路径一直往前走，只留最新的 2 个，删最旧的。可行性类在每条记录写完、保留数超过上限时：

1. **保护**：每个已提出、还没有结果的点，离它最近的那个检查点不删（它可能正在或即将用作热启动起点）。
2. 其余的，按“离最近提出的 R 个点的最近距离”从大到小删，R 是并发数；距离相同的，先删最久没有被用作热启动起点的（新建的检查点算作刚用过）。
3. 受保护的检查点本身就超过上限时，暂时超出，`driver.log` 记一句，不报错。

为什么这样留下的检查点够用：二分的区间越缩越小，Delaunay 的新点落在边界带上，OPE 扫描的 θ 在一个 Δ 内收拢，下一批点总在最近提出的点附近；离它们近的检查点留着，远的先删。每个待算的点的起点又受保护，不会在用之前被删掉。

**每个点算完立即删的**（大文件都在节点的暂存盘上，`execute.choose_scratch` 选的目录）：

| 文件 | 什么时候删 |
|---|---|
| 这个点（和它的邻点）的 PMP 部件和清单 | 记录写完之后；navigator 在 `approx_objective` 之后 |
| SDP | 同上 |
| 求解器输出里的大文件（解的矩阵） | 同上；`out.txt` 等小文件先拷进 `points/<id>/` |
| 检查点 | 按上面的保留和删除顺序 |
| block 表 | 同一台节点上跨轮保留（后面的点会复用）；目录超过它的预算份额时，在两轮之间整个清空，生成器下一轮重算 |

例外：navigator 的终点同 Skydive，把 PMP 拷进 `endpoint/`（给 EFM 用）。

**磁盘预算**（`[files]`，可选）：

```toml
[files]
budget_gb  = 500        # 运行目录里检查点的总量上限；不写：不设上限，只记录占用
blocks_gb  = 50         # 每台节点上 block 目录的上限；不写：不清理
```

- 每轮开始前估算：保留着的检查点，加上这一轮每个点各一个新检查点（大小取本运行测到的最大值）。超过 `budget_gb` 就先按删除顺序删（受保护的不删）；删完还超，就结束为 `stopped: disk budget`，写明各类占用。
- 暂存盘：每轮开始前看这台节点暂存目录的剩余空间，小于本运行测到的最大一轮的用量就结束为 `stopped: scratch full`。

**记录**：`files.jsonl` 每轮一行（检查点个数和字节、各节点 block 目录和暂存的字节），每次删除一行（轮次、类别、点号、字节、原因，例如“离最近的点 3.2，已 5 轮没用”或“预算”）；`driver.log` 每轮一句汇总。

### 2.7 可重复性

- 同一个点、同一 SDP、同样的进程数、冷启动：`sdpb` 的迭代逐位相同，判定和迭代数也相同。这是验收对照的基础（第 9 节）。
- SDP 本身与 `pmp2sdp` 的进程数无关（已实测逐位相同）；与 `sdpb` 的进程数有关，只差末位舍入。所以对照时 `sdpb` 的进程数要相同，或者只比判定和前若干位。
- 热启动会改变迭代路径，离边界远的点判定不受影响，贴着边界的点可能不同。所以每条记录写 `warm_from`，验收在冷启动下做。
- 多槽位并发时，**哪个点先算完**会影响 Delaunay 下一轮提出什么点。这不影响每个点的判定，但搜索路径不能逐点复现。单槽位运行的路径是确定的。

### 2.8 精度

- 生成器、`pmp2sdp`、`sdpb` 都用问题的精度（`Problem.solver_precision()`，或 `[numerics] solver_precision`）。
- 混合精度（interfaces 10.1）先不用于新算法：它的规则是按 Skydive 的间隙定的，可行性判定没有间隙这个量，跳跃判定又与精度有关。要用，先在同一组点上比较两种精度的判定（第 10 节 D7）。
- 点的坐标是精确十进制；二分的中点、Delaunay 的候选点用 `fractions.Fraction` 精确算，再写成有限小数（分母是 2 的幂或 10 的幂才是有限小数；不是的话按 `digits` 位有效数字舍入，见第 5.3 节）。

### 2.9 并行：槽位，一台或多台节点

一个点的 `sdpb` 不需要整个节点：bench 里同一问题在全部核和一半核上的每次迭代用时常常相差不大（例如一个较小的自旋问题在 256 核的节点上，128 进程最快）。实际的用法也常是“同时算 10 个点，每个点 48 核、100 GB”。所以并行的单位是**槽位**：一台节点上的 c 个物理核和 m GB 内存，一次算一个点（`sdpb` 用 c 个进程）。一个作业里可以有多台节点、多个槽位，只有一个驱动、一套记录和续算。不用作业数组（第 10 节 D1）。

**输入**：

```toml
[slots]
cores  = 48        # 每个槽位的物理核数，也是 sdpb 的进程数；或 "bench"
mem_gb = 100       # 每个槽位的内存；或 "bench"
count  = 10        # 同时算的点数，即槽位总数；navigator 固定为 1

[resources]        # 同 Skydive（interfaces 4）：nodes = "auto" 或节点列表、prefer、max_nodes
prefer = ["typeA"]
```

- **`"bench"`**：核数取 bench 的 `solver` 行（同机型、同 SDPB 版本、规模最近）里每节点吞吐量 ⌊核数/p⌋ / t(p) 最大的进程数 p，t 是每次迭代秒数；内存取那一行的 `max_memused_gb` 乘余量 1.15（同 `pmp2sdp` 的余量）。没有这样的行，槽位就是一整台节点（全部物理核、节点内存减预留），`driver.log` 写明没有数据。写成 `bench.slot_size(bench, cores, n, blocks, program)`。
- `bootlab bench` 现在只测全部核和一半核。要让 `"bench"` 有数据可选，`benchrun` 加测四分之一核，并记每种进程数的内存峰值（第 9 节第 1 阶段）。
- 槽位的核数在一个运行里不变，这样所有检查点都能互相热启动（第 2.6 节）。

**每台节点放几个槽位，要几台节点**（新的纯函数 `layout.slot_plan`，`bootlab submit` 调用）：

- 每台放 k = min(⌊物理核数 / c⌋, ⌊(节点内存 − 预留) / m⌋) 个，预留是站点的 `driver_reserve_mb`；k = 0 就在提交时报错。
- 例：256 个物理核、376 GB 的节点，c = 48、m = 100：按核能放 5 个，按内存 ⌊(376 − 4)/100⌋ = 3 个，所以每台 3 个；10 个槽位要 ⌈10/3⌉ = 4 台。
- 槽位平均分到各台（3、3、2、2，而不是 3、3、3、1），让每台的生成和 `pmp2sdp` 负担接近。
- 所有节点同一机型：按 `prefer` 的次序，取第一个能凑够台数的机型；有空闲节点就用 `-w` 点名，否则带 `--constraint` 排队（同 `layout.choose_nodes`）。台数超过 `max_nodes` 就在提交时报错，并说明这个上限下最多能并发几个。`submit` 打印这份计划和依据（“每台 3 个，受内存限制；4 台”）。
- 槽位都在一台节点之内，节点之间没有 MPI 通信，所以不需要 InfiniBand，也不受站点 `[multinode] allowed` 的限制（那一项管的是一个求解跨节点）；只受 `max_nodes` 限制。

**生成和 SDP 怎样到各台节点**。两种做法：

| | 节点 0 生成整轮，再分发 | 每台生成自己的点 |
|---|---|---|
| 做法 | 现在多节点 Skydive 的做法：生成器和 `pmp2sdp` 在节点 0，`execute.distribute` 把 SDP 拷到其他节点 | 多节点设计里“按点分节点生成”的想法：节点 i 的生成器和 `pmp2sdp` 只处理分到这台的点，SDP 留在这台的暂存盘 |
| 拷贝 | 每轮拷 SDP，大问题一轮几十 GB | 不拷 SDP；检查点在共享的运行目录里，谁都能读 |
| 生成 | 全部点挤在节点 0 上 | 各台同时生成 |
| block 表 | 只算一次 | 每台各算各的（重复算一部分），各台目录只有一个写者 |

用第二种。代价是各台重复算共享的 block 表，好处是没有拷贝、生成并行、各台互不依赖。它要求同一个点在哪台节点上生成都得到同样的 PMP：同一机型下应当成立，第 2 阶段验收时核对（同一个点在两台节点上生成，部件逐字节相同）。

**驱动怎样调度**：驱动在节点 0 上，每台节点一个工作线程。每个线程循环：在锁里调用 `ask`（`room` = 这台节点的空槽位数，`Strategy` 看得到所有节点的待定点）→ 写 `queue.jsonl` → `point.solve_round(这台节点, jobs)` → 在锁里写 `points.jsonl`。程序用 `srun --exact -w <这台节点>` 启动（`execute.srun_argv` 的 `hosts`）：`sdpb` 是 `--ntasks c --mem m` 的 step；生成器和 `pmp2sdp` 在这台节点的槽位都空着的时候跑，可以用整台节点。各台节点互不等待。这和 Hyperion `PointCloudSearch` 的多线程是同一个形状。

**单节点和本机模式**：槽位数不超过一台能放的，就是一个线程、一台节点。本机模式只有一台机器，槽位是互不重叠的 CPU 切片（`execute.local_argv` 已经支持）；槽位内存没有 Slurm 的 cgroup 约束，只在开始时检查总和不超过可用内存。

---

## 3. 三种算法共用的搜索外壳

### 3.1 接口：提出与告知

每个算法是一个只做算术的类，不碰文件、不启动程序：

```python
class Strategy:
    def ask(self, known: list[PointResult], pending: list[PointJob], room: int) -> list[PointJob] | Ending:
        """最多提出 room 个新点；没有可提的点时返回结局（Ending("done", ...) 等）。"""
    def state(self) -> dict:
        """当前状态，写进 search.json（只给人看）。"""
```

外壳 `search.py` 负责其余的一切：

```
载入 run.toml → 从 queue.jsonl、points.jsonl 恢复（第 3.3 节）
每台节点一个线程，各自循环：
  room = 这台节点的空槽位数
  先把上次被打断、没有结果的点重新排进来（优先，不调用 ask）
  [锁] jobs = strategy.ask(known, pending, room)；是结局就通知其他线程收尾，写 result.json
  [锁] 写 queue.jsonl
  results = point.solve_round(这台节点, jobs)
  [锁] 逐个写 points.jsonl；要结束运行的结果（第 2.3 节）就通知其他线程收尾（第 3.4 节）
  检查文件预算（第 2.6 节）、时间
```

这个形状同 Hyperion 的 `PointCloudSearch`（下一点函数拿到“点 → 判定或待定”的映射，待定点的写法是 `Nothing`；多个线程共用它），也同 simpleboot 的扫描器外壳（scheduled → calculating → calculated）。不同的是：bootlab 一次可以要多个点（`room`），所有状态都在两个只追加的文件里。

### 3.2 坐标缩放

各算法里的距离（热启动、Delaunay、navigator 的初始 Hessian）都在同一个缩放后的坐标里算：

```toml
[coords]
names = ["deltaSig", "deltaEps"]
scale = ["0.001", "0.01"]      # 新：每个坐标的长度尺度；不写时取搜索盒子的宽度
```

距离 = ‖(x − y)/scale‖。simpleboot 各处不一致（Delaunay 用初始点的极差，BFGS 用原始坐标，热启动把维数和 OPE 坐标混着算距离），Hyperion 用一个仿射变换把允许区变成大约单位球。bootlab 先只做对角缩放；要旋转（细长、斜着的岛）以后再加一个 `[coords] frame` 矩阵。

### 3.3 状态与续算

- **状态是记录的函数。** `known` = `points.jsonl`，`pending` = `queue.jsonl` 中没有结果的。二分和 Delaunay 每次都从这两个列表重新算下一点，本来就没有别的状态。navigator 的 BFGS 状态（当前点、Hessian、线搜索区间、阶段）也由已算的点按提出顺序**重放**出来：重放只做 Fraction 或固定精度 `decimal` 的算术，不求解，按点的十进制字符串**精确相等**来匹配（simpleboot 也是重放，但按 10^-170 的容差查坐标，选项每次重读）。
- 重放的结果写进 `search.json` 给人看。单测要求：任意截断的记录重放出来的状态，与不中断地跑出来的状态相同。
- **续算**：`bootlab resume <run dir>` 读记录，把被打断的点（`queue.jsonl` 里有、`points.jsonl` 里没有）先算；它们自己留下的检查点（`SIGTERM` 时求解器会写）用来热启动。不需要 simpleboot 那样手工 Drawback。

### 3.4 结局与异常

| 结局 | 什么时候 |
|---|---|
| `done: <算法的说明>` | 算法正常完成：二分区间都小于阈值、Delaunay 达到分辨率、navigator 收敛 |
| `stopped: <原因>` | 算法自己的停止条件：没有可提的点、出盒子、遇到未定点、navigator 的保护触发、磁盘预算 |
| `unfinished: <原因>` | 点数用完（`max_points`）、作业时间不够再算一轮、要结束运行的没算完或停滞（第 2.3 节） |
| `failed: <原因>` | 程序失败 |

- **跳过**：可行性类里冷启动的点没算完或停滞，判定记为 `skipped`，搜索继续（第 2.3 节）。跳过的点留在 `points.jsonl` 里，`ask` 看得到它，所以不会再提它；`result.json` 列出所有跳过的点和原因。
- **结束运行的结果**：不再提新点；其他槽位（各台节点）上还在算的点收到 SIGTERM（求解器写检查点，记录为 `unfinished: SIGTERM signal received`），然后结束。续算时这些点用自己的检查点热启动。
- **时间**：每次 `sdpb` 的 `--maxRuntime` 是作业剩余时间减 15 分钟（同 Skydive）。一台节点开新的一轮前，如果剩余时间小于本运行最近几轮实测用时的中位数，这台就不开新的一轮；所有节点都停下后，结束为 `unfinished: job time`。
- 没有自动重试。

### 3.5 选项冻结

运行开始时把 `run.toml` 复制进运行目录（同 Skydive），`result.json` 记下搜索那一节的 sha256。续算时这一节除了 `max_points` 以外有任何改动就拒绝，并说明：改选项会改变重放出来的路径，应该从这个运行的结果另开一个运行。

### 3.6 `run.toml` 的共用部分

一个运行只能有 `[skydive]`、`[scan]`、`[feasibility]`、`[bisection]`、`[delaunay]`、`[ope]`、`[navigator]` 中的一节；`inputs.load` 检查，`cli.cmd_run` 按它分派。新算法共用 `[point]`、`[slots]`（第 2.9 节）、`[files]`（第 2.6 节）三节：

```toml
[point]
decide           = "jump"      # "jump"：目标函数要是模型的 feasibility 变体；"sign"：navigator 变体
warm_start       = "nearest"   # "nearest" | "cold"
warm_radius      = "1"         # 缩放后的距离，超过就冷启动
# keep_checkpoints = 45        # 不写：可行性类 max(20, ⌈1.5 × 并发数⌉)，navigator 2

[slots]
cores = 48; mem_gb = 100; count = 10

[solver]                       # sdpb 的选项，按名字覆盖 point.jump_defaults(nmax) 或 navigator 的默认
```

`inputs.load` 要检查：`decide` 与 `objective.toml` 的 `kind` 相符（跳跃判定要目标为空的变体，符号判定要 navigator 变体）；`warm_radius` 是正数；`[slots]` 的数是正整数或 `"bench"`，navigator 的 `count` 是 1；只有一个算法节。

**`[feasibility]`**（点表，即 bt-model/2 模型上的逐点 scan）：

```toml
[feasibility]
points = [["0.5181", "1.4126"], ["0.5182", "1.4126"]]      # 每个点一行
# 或者一个格子：
# grid = { min = ["0.517", "1.40"], max = ["0.520", "1.42"], counts = [7, 5] }
```

结果：`result.json` 的 `allowed`、`excluded`、`undecided`、`skipped` 四个点号列表。跳过的点不影响其他点，点表照常算完。

---

## 4. 二分

### 4.1 原理

在一条线 x(t) = origin + t · direction 上，已知 t_a 允许、t_e 排除。假设线上允许区是一个区间（单调），不断取中点，区间每次减半，直到长度小于阈值。最后报告 [t_a, t_e] 这一对，边界在它们之间。

### 4.2 步骤

1. 每条线的初始区间由用户给（`allowed`、`excluded` 两个 t），先把两端各算一次确认：判定和给的不符就结束为 `stopped: line <i> end <t> is <verdict>, not <expected>`。
2. 每条线维护排好序的 (t, 判定)；待定点也放在序列里。找相邻、都已算完、判定不同的一对；区间长度 ≥ `threshold` 的，候选点取中点（Fraction 精确取中）。
3. `room` 个槽位：先在每条还没完成的线上各提一个点；还有空的话，在区间最长的线上做 **k 分**——把同一个区间等分成 k 段，同时算 k−1 个内点（k−1 ≤ 空槽数），一轮把区间缩到 1/k。
4. 一条线上判定的次序与单调假设不符（例如允许、排除、允许）：结束为 `stopped: line <i> not monotone near t = <t>`，并列出这几个点。不悄悄挑一个区间接着做。
5. 未定点（符号判定才有）：结束为 `stopped: undecided point`，列出它所在的线和区间。这时区间已经到了求解器的分辨率，用户可以看情况认为它就是边界。
6. **跳过的点**（第 2.3 节）：用最简单的处理——**这条线结束**，其他线照常继续。这条线报告的区间是跳过之前已有判定的、相邻异号的最短一对（跳过的点不参与），状态写 `ended: skipped point <id>`，连同跳过的原因。不在这条线上另找别的中点：中点被跳过说明这附近求解器有困难，换一个点多半也一样，而且会让二分的序列与参考实现对不上。
7. 所有线都小于 `threshold` 或已经结束：`done`；`result.json` 列出哪些线是因跳过而结束的。

### 4.3 输入

```toml
[bisection]
threshold = "1e-6"                         # t 的绝对分辨率
max_points = 200
[[bisection.line]]                         # 一条线；可以写多条
origin    = ["0.5181", "1.40"]             # 每个坐标一个
direction = ["0", "1"]                     # 只沿 deltaEps；任意方向都行
allowed   = "0"                            # t
excluded  = "0.1"
[[bisection.line]]
origin    = ["0.5185", "1.40"]
direction = ["0", "1"]
allowed   = "0"
excluded  = "0.1"
```

谱里的 gap 也可以二分：谱的值可以引用点参数（interfaces 3.2），把 gap 写成一个参数（例如 `delta = "gapS"`），它就成为一个坐标，沿它二分即可。不需要专门的 gap 二分。

### 4.4 输出

`result.json` 的 `lines`：每条线的 `bracket`（`allowed`、`excluded` 两个 t，以及对应的点）、`points`（这条线上算过的点号）、`status`（`done`，或 `ended: skipped point <id>`）。

### 4.5 并行、续算、停止

- 并行：多条线同时；空槽多时 k 分。
- 续算：线和区间全部由记录重算，没有别的状态。
- 停止：见 4.2；另有 `max_points`、时间（第 3.4 节）。

### 4.6 代码

新模块 `bisection.py`（`Strategy` 的一个实现，只有算术），共用 `search.py`、`point.py`。

---

## 5. Delaunay 扫描

### 5.1 原理

在已知点上做 Delaunay 剖分。顶点里既有允许点、又有排除点的单形（“混合单形”）跨在边界上；在其中选一个新点去算，边界附近就逐渐加密。这是二分在高维的推广，不需要 N 或梯度，只要判定。适合 2–4 维、岛的形状未知、想要整个允许区的情形。

### 5.2 步骤

1. **初始点**：用户给一张点表或一个格子。允许点和排除点都要有，总数 ≥ 维数 + 2，否则先只算初始点，算完仍不满足就结束为 `stopped: need allowed and excluded points`。
2. **剖分**：把已知点（包括待定点）按 `[coords].scale` 缩放后交给镜像里的 `qdelaunay`（qhull，选项 `Qt i`，同 Hyperion），读回单形的顶点号。
3. **去掉不完整的单形**：有待定顶点的单形不用（Hyperion 的做法）。这样并发的槽位不会在同一块地方重复提点。simpleboot 的做法是去掉外接球含待定点的单形，效果相近，更复杂。**跳过的点**（第 2.3 节）当作永远不会算完的顶点：它照样放进剖分，含它的单形永远不用，所以不会在它周围再提点。代价是那一小块边界带不再加密；`result.json` 列出跳过的点，用户看了可以另开一个点表单独复查。
4. **选点**（`choice`），在混合单形里：
   - `longest_mixed_edge`（默认）：所有混合边（一端允许、一端排除）里最长的一条，取它的中点。中点是精确的；这一条在单槽位时和 Hyperion 的 `LongestMixedEdgeMidpoint` 应当逐点相同，便于对照。
   - `largest_change_distance`：混合边最短长度最大的单形，取它各混合边中点的平均（Hyperion 的 `ChangeDistance`）。
   - `largest_simplex`：体积最大的混合单形的重心（simpleboot 的做法）。
5. 一轮要 `room` 个点时，按同一个评分取前 `room` 个，而且后选的点不能落在先选的点所在的单形里（先选的点已经当作待定点放进去重新剖分也可以，代价是多调用几次 `qdelaunay`，它很快）。
6. 盒子之外的候选点不要（盒子必填）。
7. **停止**：
   - 最长的混合边（缩放后）< `resolution`：`done: resolution reached`。两个参考实现都没有分辨率阈值，只靠点数用完停；这是新加的。
   - 没有混合单形：`stopped: no mixed simplex`（全允许或全排除，或者边界在已知点之外）。
   - 未定点：它不能当允许也不能当排除，按第 2.3 节结束搜索。
   - `max_points`、时间。

### 5.3 输入

```toml
[delaunay]
box_min    = ["0.517", "1.39"]
box_max    = ["0.521", "1.43"]
grid       = { min = ["0.517", "1.39"], max = ["0.521", "1.43"], counts = [5, 5] }   # 或 points = [...]
choice     = "longest_mixed_edge"
resolution = "0.02"            # 缩放后的长度
max_points = 300
digits     = 30                # 候选点不是有限小数时（重心）保留的有效位数
```

### 5.4 输出

`result.json`：`allowed`、`excluded`、`skipped` 的点号；`mixed_simplices`（最后一次剖分的混合单形，顶点号）；`longest_mixed_edge`。另写 `points.csv`（坐标与判定），供画图。

### 5.5 并行、续算、停止

- 并行：每轮最多 `room` 个点，靠第 5.2 节第 3、5 条避免重复。路径与完成次序有关（第 2.7 节）。
- 续算：只看记录，没有别的状态；`qdelaunay` 对同样的输入给同样的剖分。
- 维数：qhull 在 2–4 维都能用，单形数随维数增长很快，5 维以上不建议。

### 5.6 代码

新模块 `delaunay.py`：缩放、写 qhull 的输入（点数、维数、坐标）、读输出、选点。`qdelaunay` 是镜像里的程序，经 `execute.launch` 以单进程启动（它不是 MPI 程序）；它的输入是双精度坐标，只用来定拓扑，选出的点仍用 Fraction 在原始坐标里算。

---

## 6. navigator

### 6.1 原理

navigator 函数 N(x) 是一个 SDP 的最优值：在点 x 上，把交叉方程和谱假设的正性条件写成“在归一化 M 上的最小 λ”，N < 0 等价于允许，N > 0 等价于排除，而且 N 在岛外也连续地给出“离岛多远”（arXiv:2104.09518）。N 的梯度由包络定理直接从原始–对偶解得到，不需要额外求解。于是可以对 N 做 BFGS：

- **下降**：极小化 N，遇到 N < 0 就说明进了岛；
- **求端点（追极值）**：在 {N ≤ 0} 上极小化或极大化某个坐标；
- **追边界**：沿 {N = 0} 走（第 6.8 节，研究阶段）。

每个点都把 SDP 解到底，这是“嵌套”的做法。Skydive 把外参数和内点法的变量放在一起做牛顿步（“联立”），每步一次 `dynamical_sdp`，见第 6.9 节。

### 6.2 N、梯度和 Hessian 从哪里来

**N**：目标函数用模型的 `navigator`（Σ 归一化，与 simpleboot 相同）或 `navigator_gff` 变体。一次 `sdpb` 解到 `dualityGapThreshold`，N 取原始目标（simpleboot 的约定），对偶目标也记下；两者同号时 N 的符号是确定的。主点必须以 `found primal-dual optimal solution` 结束，否则按第 2.3 节结束搜索（simpleboot 也是这样，所以 navigator 的选项里不能开跳跃判定）。

**梯度**：用 `approx_objective`，不用有限差分的完整求解。

1. 生成器在同一个请求里写出 x 和 k 个邻点 x + ε eᵢ 的 PMP（`Run.points` 的做法，ε 默认 1e-20；ε 不能小于 1e-25，因为生成器按约 31 位 Δ12 给 block 表做键，见 interfaces 6）。
2. 全部转成 SDP；只求解 x。`sdpb` 要写出完整的解（x、y、X、Y）。
3. 对每个邻点调一次 `approx_objective` 的线性模式（中心 SDP、中心的解、邻点 SDP）。它输出目标的**一阶变化量 d_objective，没有除以步长**，所以 ∂N/∂xᵢ = d_objectiveᵢ / ε。这个换算已经在 bootlab 的早期试算里定下来（与完整求解对照）；它的二阶模式输出的二阶变化量已经含 ½。
4. 已知的约束：建 SDP 时 `pmp2sdp` 的进程数不影响 `approx_objective` 的结果（逐位相同）；`approx_objective` 自己的进程数不同时末尾若干位不同（舍入），所以一个运行里固定它的进程数。

simpleboot 的做法是在 x − ε eᵢ 生成邻点、用它自己的 `sdpdd` 工具。Hyperion 在 `approx_objective` 前先做几次“居中迭代”（选项 `centeringIterations`）。

**居中迭代是什么。** 内点法求解时，解沿着一条叫“中心路径”的曲线走向最优解；`sdpb` 在间隙小于阈值时停下，这时的解离最优解已经很近，但通常不正好落在中心路径上，而是偏一点。`approx_objective` 拿这个解做一阶近似，估计“邻点上的目标比这里变了多少”；推导这个近似时假定解在中心路径上，解偏得越多，估出的梯度就越不准。居中迭代就是在同一个点上让 `sdpb` 再走几步：只把解拉回中心路径，不再缩小间隙（Hyperion 的做法：间隙和误差阈值设成 0，`stepLengthReduction` 和 `infeasibleCenteringParameter` 设成 1，`maxIterations` 设成要做的次数）。代价是每个求值点多几次迭代。Hyperion 每个点做固定的次数；bootlab 按每个点偏离中心路径的程度**自适应**地定次数（第 10 节 D6），已经够居中时就是 0 次。

**偏离多少：R_error。** `sdpb` 在每次迭代开始时算 R_error = maxAbs(μI − XY)，以全精度写进 `out/iterations.json`（镜像里的 SDPB 3.0 就有）；X、Y 正好在中心路径上时它是 0。bootlab 用相对量 ρ = R_error / μ，量纲为 1。Skydive 的日志显示，每做一次居中，这个量大约降 2–3 个量级（例如 1.7e-6 → 2.3e-9 → 1.4e-12；`dynamical_sdp` 日志里的 R_err 是不是同一个归一化，第 0 阶段核对）。

**误差指标：用 c · ρ，不用一阶与二阶估计之差。** 解偏离中心路径带来的是 `approx_objective` 估计的一个系统偏差，可以写成相对误差 ≈ c · ρ，c 是这个问题的常数，由校准定（见下）。不用“一阶估计与二阶估计之差”，因为两者是从同一个偏离中心的解算出来的，带着同样的偏差，它们的差只反映邻点步长 ε 上的曲率（ε = 1e-20 时相对大小约为 1e-20），看不到偏离中心的程度。

**门槛：这一步需要多准。** 梯度的用处是预测下一步 N 降多少，以及更新 BFGS。所以要求梯度的相对误差满足

  c · ρ ≤ η · min(|cos(g, p)|, ‖g − g_prev‖ / ‖g‖)

η = `centering_accuracy`（默认 1e-3，即误差至多是预期下降量的千分之一），p 是上一次迭代的搜索方向，g_prev 是上一个点的梯度（第一个点取 1）。第一项保证预测下降量 g·p 的误差小，第二项保证 BFGS 更新所用的梯度差不被误差淹没。门槛随步而变：离解远时梯度大，几乎不用居中；接近收敛时梯度差变小，门槛变严。

**每个点怎样做。** 主点解完以后：

1. 估计 ρ₀。`iterations.json` 记的是每次迭代**开始时**的 R_error，主点最后一步之后的值没有记；取最后一条记录的 R_error，按 μ 的比例换算到结束时（ρ₀ ≈ R_last/μ_last，间隙按 out.txt 的值）。这个估计是否成立，在校准里核对。
2. c · ρ₀ 已经满足门槛：0 次。
3. 否则预测要做的次数 n = ⌈log₁₀(c·ρ₀ / 门槛) / r⌉，r 是校准测到的每次居中降的量级数（Skydive 的日志约为 2–3），从主点的检查点做一次 `maxIterations = n` 的居中调用，写出完整的解。
4. 这次调用里最后一条记录的 ρ（它是最后一步之前的值，最后一步只会让它更小，所以是偏保守的上界）满足门槛就结束；不满足，按同样的公式再做一次调用。
5. 累计次数到 `max_centering`（默认 6，未测）还不满足：结束运行，`unfinished: centering did not reach ρ ≤ <门槛> in <n> iterations at point <id>`，写明每次的 ρ。

每个点的记录写 `centering`：次数、调用数、ρ₀、最后的 ρ、门槛。

**校准：每个运行开头一次。** 运行的前 `calibrate_points` 个点（默认 1）多做以下几件事，之后不再逐点核对：

1. 对 k 个邻点各做一次完整求解，间隙阈值用 `calibration_gap`（默认 1e-60）。要这么严，是因为邻点与中心的目标差只有 ε·|g| 量级（ε = 1e-20），完整求解的误差（约等于间隙）必须远小于它，比出来的位数才有意义。得到“真的”目标变化 Δᵢ。
2. 从主点出发一次做一次居中，直到 ρ 不再下降或到 `max_centering`；每一层都算一次 `approx_objective`，得到估计 dᵢ 和相对误差 eⱼ = max |dᵢ − Δᵢ| / |Δᵢ|，同时记下这一层的 ρⱼ。
3. 拟合：c = max eⱼ / ρⱼ（取最大值，偏保守）；r = 相邻两层 ρ 的量级差的中位数；同时核对第 1 步的 ρ₀ 估计与这次实际测到的 ρ 相差不超过 10 倍。
4. 核对模型：eⱼ 要随 ρⱼ 一起下降。如果 ρ 降了而误差不降（误差被别的东西主导，例如完整求解不够准，或者 ε 的截断误差），说明这个误差模型在这个问题上不成立，结束运行：`stopped: centering error model does not hold`，写出各层的 ρ 和误差，由用户判断。
5. c、r 和各层数据写进 `result.json` 的 `centering_calibration` 和 `driver.log`。续算时从记录里读，不重做。

校准点多花的是 k 次严格的完整求解和几次 `approx_objective`，只在运行开头一次。

**Hessian**：默认 BFGS 割线更新，不额外求解。`approx_objective` 的二阶模式可以给精确的 Hessian，但要多生成 k(k+1)/2 个交叉点的 SDP。玩具实测（6 个参数、光滑边界，到端点的平均求值点数）：

| 曲率来源 | 点数 |
|---|---|
| 只用 BFGS | 11.6 |
| 只进小块的参数（OPE 类）的行列每点精确，其余 BFGS | 8.6 |
| 起点算一次精确 Hessian，之后 BFGS | 8.2 |
| 上两条合用 | 6.7 |
| 每点精确 | 5.2 |
| 起点算一次精确 Hessian，之后**不更新** | 15.7（最差，不要这样用） |

所以先只做 BFGS；`hessian = "exact_at_start"` 作为第 8 阶段的选项。

**有限差分的完整求解**（`gradient = "full"`）只给测试用：每个邻点完整求解，梯度 = (N(x+ε eᵢ) − N(x))/ε，用来验收 `approx_objective` 的梯度。中心和邻点都要解到 `calibration_gap`，理由同校准。

### 6.3 阶段与状态机

```
start ──► descend ──(N<0 且 mode = extremize)──► extremize ──► done
             │                                      │
             ├─(N<0 且 mode = minimize 且 stop_on_negative)──► done: allowed point
             ├─(|g| 小而 N>0)──► stopped: positive minimum
             └─(盒子、保护、上限)──► stopped / unfinished
```

- **descend**（simpleboot 的 MyBFGS 加 `FeasibilityQ`；Hyperion 的 `BFGS.run` 加 `stopOnNegative`）：方向 p = −B⁻¹g（不是下降方向就取反），线搜索（第 6.4 节），BFGS 更新只在 sᵀy > 0 时做。
- **extremize**（simpleboot 的 ConstrBFGS；论文的 Algorithm 2）：在二次模型 m(s) = N + g·s + ½ sᵀBs 上，固定目标坐标 a 的步长 s_a，其余坐标取极小，代回 m = 0 得到 s_a 的二次方程；求极小取小根、求极大取大根，这就是“走向模型预测的岛尖”。没有实根时取模型的极小点。进入这个阶段时带着 descend 最后的 B。
- 阶段的切换写进状态，由重放得到（第 3.3 节）。

### 6.4 线搜索与停止（基线：simpleboot 的语义）

按规则 8，基线照 simpleboot 的行为写，并作为对照组保留；改进都是选项，默认关，等在同一问题上对照过再改默认。

**descend 的线搜索**（MyBFGS）：先试 α = 1；外推时至少放大 2 倍，外推不可靠时取 4 倍；内插用三次 Hermite，截在当前区间的 [0.1, 0.9]；接受 |g·p| < 0.9 |g₀·p| 且 N < N₀，或者 N₀ > 0 而 N < 0（刚进岛，立刻接受）。α 的上限是沿 p 第一次碰到盒子的位置；碰到盒子且还在下降：结束为 `stopped: box`。

**extremize 的线搜索**（ConstrBFGS）：先试 α = 1（不超过盒子）；接受 N < r·|N₀|（r 默认 2）；不接受时用两点三次插值，N₀ > 0 取插值的极小点，N₀ < 0 取最小根，截在 [0.1α, 0.9α]；区间里没有根：`stopped: line search found no root`。

**停止**：

| 条件 | 结局 |
|---|---|
| \|g⊥\| ≤ `goal_g`、\|g_a\| > 10·max(`goal_g`, \|g⊥\|)、\|N\| < `goal_n` 同时成立（g⊥ 是去掉目标坐标的梯度） | `done: endpoint` |
| 步长 \|s\| < `goal_s` | `done: step below goal_s` |
| 目标坐标连续 `stable_iterations` 次迭代变化都小于 `goal_x`（新加的相对判据，见下） | `done: target coordinate stable` |
| 最近两次求值的 N 之差 < 10 × `dualityGapThreshold`（simpleboot 的保护一） | `stopped: N change below the solver's resolution` |
| 沿两点连线的差分斜率与两端的方向导数不一致（simpleboot 的保护二，阈值 100） | `stopped: gradient inconsistent` |
| `max_iterations`（默认 400，同 simpleboot）或 `max_points` | `unfinished: ...` |
| 主点不是 optimal、程序失败 | 按第 3.4 节 |

为什么要新加“目标坐标稳定”这一条：生产数据里，simpleboot 的十个 navigator 运行**没有一个**是按收敛判据（绝对阈值 1e-20）停下的；真正起作用的是保护一，它停下时点已经收敛了。Hyperion 的端点搜索在岛尖上满足了 |N| ≤ 1e-6 的点有几十个，但 \|g⊥\| 始终降不下来（岛尖是折角，见第 6.5 节），四个方向的点大约每 50 步重复一次。单看 \|g⊥\| 在折角上永远停不了，所以要有一个看目标坐标本身的判据。保护一保留为一个有名字的结局，让用户看了再判断，不当作收敛。

### 6.5 实测结论：要采纳的和已判无效的

**要采纳的**（来自对 simpleboot、Hyperion 生产运行的分析，以及 bootlab 自己的实测）：

- **初始 Hessian 决定第一步。** simpleboot 的默认（单位矩阵）让第一步冲得太远，线搜索每次最多缩小 10 倍、每次试探都是一次完整求解，被拒的试探点在生产运行里耗掉几十个小时；用 s·diag(盒宽)（s = 0.02）的运行没有这个问题。Hyperion 用 diag(\|g\|/0.2/盒宽)，第一步约为盒宽的 20 %。bootlab 默认用 Hyperion 的规则（量纲是对的），可以写 `hessian_guess` 覆盖，或者从上一个运行（或 Skydive 的最后一条记录）带过来（第 10 节 D5）。
- **墙钟的 78–96 % 是主点求解**，梯度只占很少。所以要省的是点数和每点的迭代数，热启动的深度和距离门槛最重要（第 2.6 节）。
- **只看 N 的线搜索会贴着边界来回穿越**：一个运行 72 个点里 30 个在岛外。基线保留这个行为作为对照；改进（同时看目标坐标进展和 N 的约束违背的评价函数）是第 8 阶段的选项。
- **从岛外出发可能长时间停在岛外**，N 下降很慢；“下降变慢就说明没有岛”的规则会误报（一个运行在规则触发以后又进了岛）。所以不自动判“无岛”：`driver.log` 写提醒，用户可以另开一个可行性运行在别处检查。\|g\| 小而 N > 0 是真正的局部极小，结束为 `stopped: positive minimum`。
- **Hyperion 代码里的几处，要避开**：线搜索的 zoom 没有次数上限；扩张越过盒子就悄悄结束；二分线搜索的插值比例被钳成常数 0.5（等于纯二分）；找变号时 α 翻倍而不看盒子。bootlab 的每个循环都有上限，越界是有名字的结局。
- **岛尖是折角。** 玩具上，半岛的尖由两条光滑边界分支以约 16° 相交而成；N 在尖附近等于两个光滑分支函数中较大的那个，梯度方向在尖上跳变。Hyperion 的实测里，尖的两侧梯度是 (+5, −0.5) 和 (−1700, +125)。单个二次模型（ConstrBFGS、Hyperion 的 BFGS、Skydive 的 BFGS）在折角上是错的模型。玩具上可行的做法：沿一条分支贴着边界走，梯度方向突变时取两侧切线的交点（16–23 个点到尖）；或者保留所有点的线性割、用一个小线性规划取下一个点（18 个点到 5e-6，但它不含曲率，在 6 维光滑边界上 150 个点到不了 1e-2）。这些都只在玩具上测过，放在第 8 阶段。

**已判无效、不要再推荐的**：

| 办法 | 结论 |
|---|---|
| 锥上测地线外推 X_k X_{k−1}⁻¹ X_k 作起点 | 无效，与原样热启动相同（X⁻¹ 把子空间的错位放大 1/μ 倍） |
| 用格点线性规划玩具的迭代数推断真实求解器 | 不可外推：格点让解路径不光滑 |
| 对偶泛函复用到邻近的点（不求解直接认证） | 只能认证到离开距离的 1 % 以下 |
| 参数空间的势垒同伦 max v·p + ν log(−N) | 19–69 个点，模型步只要 5–12 个 |
| 高斯过程代理模型找边界 | 精度停在 1e-3，折角处更差 |
| 跨 Λ 外推边界的位置 | 比直接沿用低 Λ 的点更差；能带过来的只有起点、尺度和初始 Hessian |
| 离尖还远时用两条分支的线性模型直接求交点 | 失败：岛外的值属于另一个分支，模型把边界放错位置 |
| 起点算一次精确 Hessian 后冻结 | 比 BFGS 还差 |
| log X 外推起点，Ŷ = μX̂⁻¹ | 真实求解器上比原样起点更差（相邻记录的 μ 不同） |
| 两层 μ 外推代替下降 | 真实求解器上只省 1–2 个量级的下降，不是玩具上的整段 |
| Skydive 加信赖域、调低 `stepMinThreshold`、多爬升 | 每步能走多远由间隙决定（实测步长 ∝ gap^0.46），这些都不让它走得更远 |
| 求解器的外部校正器 | 在较大问题上保存状态时死锁，不可用 |

### 6.6 输入

```toml
[navigator]
mode       = "extremize"                 # "minimize" | "extremize"
target     = { coordinate = "deltaEps", sense = "max" }    # extremize 才要
start      = ["0.5182", "1.4130"]
box_min    = ["0.5170", "1.3900"]
box_max    = ["0.5200", "1.4300"]
stop_on_negative = true                  # minimize：第一个 N<0 的点就停
gradient   = "approx_objective"          # "approx_objective" | "full"（只给测试）
infinitesimal = "1e-20"                  # >= 1e-25
centering_accuracy = "1e-3"              # η：梯度的相对误差上限（相对预期下降量），定居中次数的门槛
max_centering = 6                        # 一个点累计居中次数的上限，到了仍不够就结束运行
calibrate_points = 1                     # 开头校准几个点
calibration_gap = "1e-60"                # 校准时邻点完整求解的间隙阈值
hessian    = "bfgs"                      # "bfgs" | "exact_at_start"（第 8 阶段）
# hessian_guess = ["1000000", "10000"]   # 对角；或 k×k 的完整矩阵；不写：diag(|g|/0.2/盒宽)
line_search = "simpleboot"               # "simpleboot" | "wolfe"（Hyperion 的强 Wolfe，第 8 阶段）
accept_ratio = "2"                       # extremize 线搜索接受 N < r |N0|
goal_g = "1e-20"; goal_n = "1e-20"; goal_s = "1e-20"
goal_x = "1e-10"; stable_iterations = 5
max_iterations = 400
max_points = 300

[point]
decide = "sign"                          # navigator 的主点总是解到 optimal；这里只决定记录里的判定
warm_start = "nearest"
warm_radius = "0.5"
# keep_checkpoints = 2                   # navigator 默认保留最近 2 个求值点的检查点

[slots]
cores = "bench"; mem_gb = "bench"; count = 1   # navigator 是顺序的，一个槽位

[solver]
dualityGapThreshold = "1e-30"            # 决定 N 的分辨率和保护一
```

`inputs.load` 检查：`objective.toml` 的 `kind` 是 navigator 变体；`start` 在盒子里；`target.coordinate` 是坐标名；`infinitesimal` ≥ 1e-25；向量长度都等于坐标数；`decide = "jump"` 与 navigator 不相容。

### 6.7 输出、并行、续算

- **记录**：每次求值一行 `points.jsonl`（`purpose` = `navigator` 或 `line_search`，带 `navigator`、`gradient`）；`search.json` 的状态：阶段、当前点、B、线搜索区间、迭代数。
- **结果**：`result.json` 的 `x`、`navigator`、`gradient`、`hessian`（最后的 B，Skydive 可以直接用）、`inverse_hessian`、`iterations`、`phase`。
- **并行**：一次求值是 k+1 个点的生成和一次主点求解，一个槽位（`[slots] count = 1`；核数默认由 bench 定，没有数据就是整台节点）。一个点没算完或停滞，整个搜索结束（第 2.3 节），不跳过：BFGS 的下一步依赖这个点。以后可以把线搜索的几个 α 同时放进不同槽位（第 8 阶段）：试错的代价从“白算一次再缩步”变成“白用一个槽位”。
- **续算**：重放记录得到状态（第 3.3 节）；被打断的求值用它自己的检查点热启动。

### 6.8 追边界（研究阶段）

二维截面上，从一个边界点出发沿 {N = 0} 走。玩具上可行的组合：用最近两个节点的切向估曲率，沿圆弧预测；在预测点求 N、∇N，沿法向做一步牛顿投影 x − N g/\|g\|²；投影量很小时不再为校正求解，直接取投影点；校正后切向突变就当作折角，取两侧切线的交点；节点之间用三次 Hermite 重建。玩具上走完半岛的一侧加尖端用了 29–82 个求值点，“吸引子 + 排斥子 + 切向漂移”的流方法要 270–673 个。

限制：n 维里边界是 n−1 维的，三维以上不追整张面，只追二维投影或给出 2n 个坐标端点。追踪给出的节点是有限间隙下的数值零点，要作为论文里的界，每个要用的点仍要在外侧取一个点解到底并确认 N > 0。这一段只有玩具证据，第 8 阶段先用已有记录做不花钱的诊断，再决定做不做。

### 6.9 与 Skydive 的关系

| | navigator（本节） | Skydive（已有） |
|---|---|---|
| 每个点 | 解到底，N 的符号可以确定 | 一次 `dynamical_sdp` 调用，只走一小步，不解到底 |
| 外层 | BFGS 在驱动里（Python，可重放） | BFGS 在求解器里 |
| 进度 | 每个求值点都要冷启动或热启动到底 | 每步固定开销约占 75 %，步长 ∝ √gap |
| 并行 | 线搜索的试探可以并行 | 顺序 |
| 适合 | 需要每个点都有确定的 N；任意坐标的端点；认证 | 从一个点快速走到岛内或边界 |

共用的代码：邻点（`Run.points` 移出来）、生成和 `pmp2sdp`（同一个共用函数）、bench、检查点保留。互相交接：

- **Skydive → navigator**：Skydive 的终点作 `start`，它最后一条记录的 `hessian`（完整矩阵）作 `hessian_guess`。
- **navigator → Skydive**：navigator 的终点作 `[skydive] start`，最后的 B 作 `hessian_guess`。现在 `hessian_guess` 只收对角元，要扩成也收完整矩阵（小改动，Skydive 回归不变）。
- **检查点**：`sdpb` 的检查点能不能直接给 `dynamical_sdp` 用，没有验证过（第 10 节 R5）；在验证之前交接时求解器冷启动。
- **认证**：Skydive 的终点用一次符号判定（`[feasibility] points = [终点]`，`decide = "sign"`）复核。bootlab 以前手工这样复核过一个 Skydive 终点：bootlab 与参考程序在同一点各自建 SDP、解到底，N 一致 162 位。

---

## 7. OPE 扫描（切割面法）

以后再做（第 9 节第 7 阶段），现在先把设计写下来。这个算法会累积大量文件，所以本节的重点是文件管理。

### 7.1 原理

混合关联函数里，有些孤立算符（例如外算符自己的交换）的 OPE 系数是一个 j 维向量 λ，只知道它的方向未知。bootlab 的模型有两种写法（interfaces 3.1）：把 λ 的方向当坐标（`opeAngle` 的 θ，或 `opeCoordinates`），这时每个点的 λ 是固定的；或者不写方向，写成 j×j 半正定约束，这样允许的是所有半正定矩阵，比“存在某个方向 λ”宽，得到的允许区偏大。

OPE 扫描在两者之间：外层只在维数（以及其他坐标）上搜索，每个外层点 Δ 内部判断“是否存在一个 λ 使 (Δ, λ) 允许”。做法是切割面法（arXiv:1912.03324；simpleboot 的 OPE 扫描器就是它）：

- 在 (Δ, λ) 上做一次判定。允许：这个 Δ 允许，内层结束。
- 排除：求解器给出的泛函 y 作用在外部那一块的 j×j 矩阵上，得到一个二次型 Q；凡是 λᵀQλ 与当前 λ 同号的方向，都被这同一个泛函排除了。每排除一次就切掉 λ 空间的一块。
- 在没被切掉的区域里选下一个 λ（取区域的“中心”）。区域切空了：这个 Δ 排除。

外层（点表、二分或 Delaunay）只看每个 Δ 的最后判定。

### 7.2 每个 Δ 的内层步骤

1. **初始 λ**：这个 Δ 上次的 λ（续算时）；否则离它最近的、已经做完的 Δ 的最后 λ（simpleboot 的做法）；再没有就用 `[ope] start`。
2. **判定**（跳跃判定，第 2.2 节）。排除时求解器要写出 y。
3. **算切割**：外部那一块在 PMP 里是一个部件，Δ 固定时它是 j×j 个常数向量 M_ab；Q_ab = y · M_ab（连同 SDPB 用归一化消去的那一个分量）。驱动用 Fraction 读部件的 JSON 和 y 来算，不调用别的程序。符号约定在第 7 阶段的第一个验收里定：当前 λ 必须恰好被它自己的切割排除。
4. **剩下的区域**：所有切割 {λ : λᵀQᵢλ 与排除的符号相反} 与 `[ope] box` 的交。
5. **选下一个 λ**。先只支持 j = 2（`opeAngle`，一个角度 θ）：每个切割是 θ 的三角二次式，零点可以精确解出，剩下的区域是若干个区间，取最长区间的中点；没有区间就是切空了。这样“切空”是精确的判断。j ≥ 3 时 simpleboot 用随机采样找可行点，采样找不到不等于区域为空，不能作为排除的证据，所以 j ≥ 3 以后再做（第 10 节 D12）。
6. **上限**：每个 Δ 最多 `max_cuts` 次切割（simpleboot 没有上限）；到了上限，这个 Δ 按跳过处理（第 2.3 节），原因写 `ope: max cuts`。

Hyperion 也有 OPE 搜索（hyperion-bootstrap 的 `OPESearch`，用泛函的双线性型）。本文没有细读它，实现前要读，按第 8 节的方式写明借鉴和取舍。

### 7.3 外层、跳过、并行

- 外层用 `outer` 选：`"points"`、`"bisection"`、`"delaunay"`，照常写同名的那一节，坐标只是外层坐标（不含 OPE 坐标）。外层看到的每个 Δ 只有一个判定：允许、排除或跳过。
- **跳过**：内层某次判定是冷启动的点没算完或停滞，这个 Δ 跳过（外层按第 4、5 节对待跳过的点）；热启动的点出这种情况，结束运行（第 2.3 节）。
- **并行**：一个槽位一次做一个 Δ 的内层（内层是顺序的），`[slots] count` 就是同时在做的 Δ 数。
- **记录**：`points.jsonl` 每次判定一行（`purpose = "ope"`，`extra` 有 `delta` 的点号和 λ）；`deltas.jsonl` 每个做完的 Δ 一行（判定、切割数、最后的 λ）；`cuts/<点号>.json` 存这个 Δ 的所有 Q（很小，全部保留，续算和复查都要用）。

### 7.4 文件管理

一个 OPE 扫描有几百个 Δ，每个 Δ 有几次到几十次判定，每次判定都会生成 PMP、SDP、求解器输出和检查点。不管理的话很快占满磁盘。规则（在第 2.6 节之上）：

| 文件 | 在哪 | 保留到什么时候 |
|---|---|---|
| PMP 部件和清单 | 节点暂存 | 这次判定的 SDP 建好就删 |
| SDP | 节点暂存 | 这次判定求解完就删 |
| 解 y | 节点暂存 | Q 算出、写进 `cuts/` 就删 |
| 求解器的其他大输出 | 节点暂存 | 求解完就删；`out.txt` 拷进 `points/<id>/` |
| 一个 Δ 内的检查点 | 运行目录 | 只留最新一次判定的；下一个 λ 从它热启动，上一个随即删掉 |
| 做完的 Δ 的检查点 | 运行目录 | 留它最后一次判定的，进入第 2.6 节的保留和删除顺序（距离按外层坐标算） |
| block 表 | 节点暂存 | 跨 Δ 复用；超过 `[files] blocks_gb` 时在两轮之间清空 |
| `cuts/`、记录 | 运行目录 | 全部保留 |

**怎样保证留下的检查点够后续热启动**：正在做的每个 Δ，它当前的检查点受保护；外层每提出一个新 Δ，就立即把离它最近的已做完 Δ 的检查点标为受保护，直到新 Δ 的第一次判定开始。其余的按第 2.6 节的删除顺序淘汰：先删离最近提出的 Δ 最远的，距离相同先删最久没用的。

**磁盘预算**：运行目录里的量 ≈ (`keep_checkpoints` + 并发数) × 一个检查点的大小；每台节点暂存里的量 ≈ 这台的槽位数 × (一个点的 PMP + SDP + y) + block 目录。第一轮测出每类的大小后，`driver.log` 写出这个估计；超过 `[files] budget_gb` 或暂存剩余空间时按第 2.6 节处理。每次删除都记进 `files.jsonl`。

### 7.5 输入

```toml
[ope]
outer       = "delaunay"         # "points" | "bisection" | "delaunay"；外层的节照常写
coordinates = ["theta"]          # 哪些坐标是 OPE 坐标（内层），其余是外层坐标
start       = ["0.9"]
box_min     = ["0.5"]
box_max     = ["1.3"]
max_cuts    = 50                 # 每个 Δ 的切割上限，到了这个 Δ 跳过

[delaunay]                       # 外层，坐标只有外层的
box_min = ["0.517", "1.39"]; box_max = ["0.521", "1.43"]
grid = { min = ["0.517", "1.39"], max = ["0.521", "1.43"], counts = [5, 5] }
resolution = "0.02"; max_points = 300

[point]
decide = "jump"                  # OPE 扫描只用跳跃判定

[slots]
cores = 48; mem_gb = 100; count = 10

[files]
budget_gb = 500; blocks_gb = 50
```

### 7.6 代码

新模块 `ope.py`：内层的 `Strategy`（一个 Δ 一个状态，状态由 `points.jsonl` 和 `cuts/` 重放），切割的计算，j = 2 的区间运算。外层复用 `bisection.py`、`delaunay.py`；文件管理的函数放在 `search.py`，与可行性类共用。

---

## 8. 参考实现：借鉴了什么，为什么取舍

simpleboot（Mathematica）和 Hyperion 生态（Haskell：hyperion-bootstrap、scalars-3d、bootstrap-params、sdpb-haskell）都读过源码。只学思路，不抄代码；接口和命名是 bootlab 自己的。

| 思路 | 来自 | 取舍 | 理由 |
|---|---|---|---|
| 所有策略共用一个“下一点”外壳，待定点传给选点函数避免重复 | 两者都有 | 取 | 三种算法的差别只在选点 |
| 结果只分三类，“其他”静默丢弃 | simpleboot | **舍** | 会原地重复提同一个点 |
| “不是对偶可行”即允许 | Hyperion 默认 | **舍** | `maxComplementarity`、超时都会被当作允许 |
| 跳跃判定的参数 | Hyperion `jumpFindingParams` | 取 | 现成的、按 nmax 分档的 |
| `primal-dual optimal` 算作对偶可行（排除） | Hyperion | 取 | 项目负责人定（第 10 节 D2） |
| 解析失败、热启动的 maxIterations 冷重试一次 | simpleboot | **舍** | 规则 2：不自动重试 |
| 没算完的点：可行性类跳过、永不再提 | 新 | 加 | 不像 simpleboot 那样原地重复，也不因一个难算的点停掉整个扫描 |
| 热启动取最近点 | 两者 | 取，加距离门槛 | 远处热启动可以慢 3–6 倍 |
| 热启动超 400 次迭代改冷启动 | Hyperion 参考驱动 | **舍** | 这是一次重试；改为距离门槛加结束 |
| 中间检查点热启动 | simpleboot 生产用法 | 取（若 `sdpb` 支持） | 实测迭代数 0.2–0.4 倍 |
| 二分：相邻异号对，取最长区间，待定点混在排序里 | simpleboot `BisectionSampler`、Hyperion `pointMapToBracket` | 取 | 简单，天然支持并发 |
| 任意方向的二分 | 两者都没有原生的 | 新加 | 一条线的写法同时覆盖坐标方向 |
| k 分 | 新 | 加 | 用空槽位 |
| 不单调时停 | 新 | 加 | 不悄悄选一个区间 |
| Delaunay 剔除有待定顶点的单形 | Hyperion | 取 | 比外接球判据简单 |
| 选点：最长混合边中点 | Hyperion | 取为默认 | 精确有理，可以逐点对照 |
| 选点：最大单形的重心 | simpleboot | 选项 | 保留 simpleboot 的行为 |
| 缩放：仿射变换 / 初始点极差 | Hyperion / simpleboot | 统一成 `[coords].scale` | 所有距离一个定义 |
| Delaunay 的分辨率阈值 | 两者都没有 | 新加 | 否则只能靠点数用完停 |
| `qdelaunay Qt i` | Hyperion | 取 | 镜像里已有 qhull |
| navigator 状态 = 历史的函数（重放） | simpleboot | 取，改成精确匹配、选项冻结 | 续算天然正确；simpleboot 按容差匹配、选项不存，改选项就改路径 |
| ConstrBFGS 的椭球尖步、接受 N < r\|N₀\|、[0.1, 0.9] 截断 | simpleboot | 取为基线 | 规则 8；作为对照组 |
| MyBFGS 加 FeasibilityQ，再 ConstrBFGS，两阶段 | simpleboot（手工两次运行） | 取，写成显式阶段 | 换阶段带着 B |
| 强 Wolfe 线搜索，维护 Hessian 用 LU 解牛顿步 | Hyperion `BFGS.run` | 选项 | 先对照基线 |
| 初始 Hessian diag(\|g\|/0.2/盒宽) | Hyperion、论文 Algorithm 1 | 取为默认 | simpleboot 的单位矩阵实测很差，s·盒宽量纲不对 |
| 绝对阈值 1e-20、硬上限 400 | simpleboot | 保留，加相对判据 | 生产里从没达到绝对阈值 |
| 两道保护 | simpleboot | 取，作为有名字的结局 | 实际起作用的收敛判据，但不当作收敛 |
| 梯度用 `approx_objective` | Hyperion | 取 | 镜像里有；换算已定 |
| 梯度用 `sdpdd`，邻点在 x − ε eᵢ | simpleboot | 舍 | 镜像里没有这个工具；邻点方向与 Skydive 统一为 +ε |
| `approx_objective` 前的居中迭代 | Hyperion（固定次数） | 取，改为按 R_error 自适应 | 够居中就不做；次数由这一步需要的精度定，开头校准一次 |
| N 取 y/(1−y)（对偶目标） | Hyperion 的 GFF navigator | 不需要 | bootlab 的两种归一化都直接给 N |
| 切割面法的 OPE 扫描，Δ 之间沿用最后的 θ | simpleboot | 取 | 第 7 节 |
| 找可行 θ：极小化、随机采样、外部的 SDP 松弛 | simpleboot | 先只做 j = 2 的精确区间 | 采样找不到不等于区域为空；j = 2 可以精确判断 |
| 每个 θ 保留最近两个检查点，内层没有上限 | simpleboot | 改为每个 Δ 留一个，加 `max_cuts` | 文件管理（第 7.4 节）；异常要有结局 |

simpleboot 的另外几处行为，读代码时发现、实现时要注意：盒子写 `Automatic` 实际不能用（所以 bootlab 的盒子必填）；显式 Hessian 工具生成移位 SDP 时传了未移位的点（疑似 bug，bootlab 不照搬）；Skydive 每次传 `totalIterationCount 0`（bootlab 已改为累计）。

---

## 9. 分阶段实施计划

不写工期。每一阶段做完，按实际情况定下一阶段。每阶段的交付都包括：代码、单测、`run.toml` 示例、interfaces 中英文相应章节的更新。

**测试规模**：除非另说，都用 `examples/ising` 的 Ising σ–ε 系统，Λ = 11（SDP 的 N = 104，36 块，一个点在一个节点上几分钟），一个节点。第 6 阶段加一个较大的问题（N≈2300）只测几个点的用时和判定，不跑完整搜索。

**参考结果**：对照要先在同一问题上跑 simpleboot 或 Hyperion（参考程序）。参考结果**不放进公开仓库**，留在维护者那里，验收结论（判定是否相同、一致几位）写进 interfaces。以后有了 Ising 的测试，可以挑一个小的放进 `tests/reference/` 作为例子（第 10 节 D10）。

### 第 0 阶段：前提核实（只读，加少量小作业）

| 交付 | 验收 |
|---|---|
| 镜像里的 `sdpb`、`approx_objective`、`qdelaunay` 的版本和 `--help` 记进 interfaces；确认 `--initialCheckpointDir`、中间检查点选项、`approx_objective` 线性和二阶模式的参数、`out/iterations.json` 里的 R-err（第 6.2 节）及其与 `dynamical_sdp` 日志的归一化是否相同 | 文档化；`execute` 的命令行有单测固定 |
| 生成器的 `feasibility` 目标：同一个点，bootlab 的 PMP 与参考程序的 PMP | 部件逐字节相同（同 interfaces 6 的标准） |
| `point.jump_defaults(nmax)` | 与 Hyperion `jumpFindingParams` 逐项相同（单测） |

### 第 1 阶段：点判定原语与点表

| 交付 | 验收 |
|---|---|
| `point.py`（第 2.4 节）、`search.py`（第 3 节）、`[point]`、`[feasibility]` 的输入检查；`Run.build_sdps`、`Run.points` 移成共用函数 | Skydive 回归 `bootlab regress` 照常 PASS（≥ 130 位），`solves.jsonl` 除秒数外不变 |
| 跳跃判定：约 30 个点，横跨 Λ = 11 岛的边界（含离边界很近的点） | 每个点的 `terminateReason` 和判定与参考程序（同样的求解器参数、冷启动、同样进程数）相同；迭代数相同 |
| 符号判定：同一组点 | 判定相同；原始、对偶目标与参考程序一致 ≥ 130 位（同回归标准） |
| 单节点槽位：`[slots]`、`bench.slot_size`、`layout.slot_plan`（一台），`benchrun` 加测四分之一核和内存 | 同一组点用 1 个和 s 个槽位，判定逐点相同；每轮记录槽位的空等时间 |
| 跳过 | 用 `maxIterations` 很小的冷启动点造出跳过：记为 `skipped`，不再提，点表照常算完；热启动的点同样设置时运行结束 |
| 热启动与文件管理（第 2.6 节） | 热启动下离边界远的点判定不变；记录 `warm_from`；迭代数与冷启动之比写进 interfaces；保留数和删除顺序与单测预期一致，`files.jsonl` 的占用与实际目录大小一致 |
| 续算 | 在一轮中间 SIGTERM（本机模式用 `bootlab stop`）、在两轮之间 SIGKILL，续算后结果与不中断的运行相同（判定、点集） |
| 本机模式 | 同一组点在本机模式下判定相同 |

### 第 2 阶段：多节点槽位（第 1 阶段的单节点槽位验收之后）

| 交付 | 验收 |
|---|---|
| `layout.slot_plan` 的多节点部分、`bootlab submit` 打印计划；每台节点一个工作线程，各台生成自己的点（第 2.9 节） | 单测：给定机型和 `[slots]`，每台个数、台数、分配与第 2.9 节的例子相同；超过 `max_nodes` 时报错并给出能并发的上限 |
| 同一个点在两台同机型节点上生成 | PMP 部件逐字节相同 |
| 一组点（约 30 个）用 2 台节点、每台若干槽位 | 判定与第 1 阶段单节点的逐点相同；记录只有一份；每台的空等时间写进 interfaces |
| 续算 | 一台节点的槽位正在算时 SIGKILL 整个作业，续算到别的节点组合上，结果与不中断的相同 |
| 用户的实际规模：10 个槽位，每个 48 核、100 GB | 一次运行，记下台数、每台槽位数、每轮用时；只测用时，不对照 |

### 第 3 阶段：二分（第 1 阶段之后，可以和第 2、4、5 阶段并行）

| 交付 | 验收 |
|---|---|
| `bisection.py`、`[bisection]` | 单槽位：两条线（一条沿 Δε，一条斜线），区间序列与参考实现（simpleboot 的 `BisectionSampler` 或 Hyperion 的 `binarySearch`，同样的初始区间和阈值）**逐点相同**：中点是精确的，判定相同，序列就相同 |
| k 分 | 最终区间包含单槽位的最终区间；点数与轮数写进 interfaces |
| 不单调、未定点、中点被跳过（这条线结束、其他线继续）的结局 | 单测（假的判定序列） |

### 第 4 阶段：Delaunay（第 1 阶段之后）

| 交付 | 验收 |
|---|---|
| `delaunay.py`、`[delaunay]` | 二维（θ 固定的 (Δσ, Δε) 截面）：单槽位、`longest_mixed_edge`、同样的初始点和缩放，前 M 个点（M 取 30）与 Hyperion 的 `delaunaySearch`（`LongestMixedEdgeMidpoint`，单线程）逐点相同 |
| 多槽位 | 最后的混合单形覆盖的区域与单槽位的相容（每个点的判定都相同，边界带互相包含在分辨率以内） |
| 三维 | 一次运行到分辨率，与同问题上 Skydive 或 navigator 终点相容（终点在允许区里） |
| 跳过的顶点 | 单测：含跳过顶点的单形不再提点 |

### 第 5 阶段：navigator 下降（第 0 阶段之后就能开始写；运行要第 1 阶段的外壳）

| 交付 | 验收 |
|---|---|
| `approx_objective` 的执行函数和梯度；自适应居中与校准（第 6.2 节） | 校准在 Ising Λ = 11 上得出 c、r，误差随 ρ 下降（模型成立）。另取校准之外的 5 个点，邻点完整求解到 `calibration_gap`：每个点上校准模型给出的误差上界 c·ρ 不小于实际误差（即估计的位数不多于实际对上的位数），而且实际误差满足门槛。记录每点平均居中次数和 0 次的比例，写进 interfaces。与 simpleboot 的梯度一致到 O(ε) 的差别以内 |
| 居中的异常 | 单测：到 `max_centering` 仍不够、误差不随 ρ 下降，两种结局都有名字 |
| `navigator.py` 的 `minimize`、BFGS、MyBFGS 线搜索、重放 | 与 simpleboot 的 MyBFGS 在同一问题、同一起点、同一盒子和初始逆 Hessian 下：被接受的点数相同，前 10 次迭代的点一致 ≥ 10 位，结局相同 |
| 重放 | 单测：任意截断的记录重放出的状态与不中断的相同 |

### 第 6 阶段：navigator 求端点（第 5 阶段之后）

| 交付 | 验收 |
|---|---|
| `extremize`、ConstrBFGS 线搜索、停止条件和两道保护 | 与 simpleboot 的 ConstrBFGS 求 Δε 最大（同一问题、同起点、同参数）：终点一致到 `goal_x`，点数相差不超过几个；结局有名字 |
| 交接 | Skydive 终点 → navigator，navigator 终点 → Skydive（`hessian_guess` 收完整矩阵，Skydive 回归不变） |
| 较大的问题（N≈2300） | 3 个点的用时（生成、`pmp2sdp`、主点、`approx_objective`）写进 bench；判定与符号判定一致 |

### 第 7 阶段：OPE 扫描（以后做；要第 3、4 阶段的外层）

| 交付 | 验收 |
|---|---|
| 切割的计算（第 7.2 节第 3 步） | 一个排除点上：当前 λ 被它自己的切割排除；Q 作用在当前 λ 上的值与求解器的对偶目标相容（符号约定写进 interfaces） |
| `ope.py`、`[ope]`，j = 2 | Ising Λ = 11，θ 为 OPE 坐标，外层是点表（约 20 个 Δ）：每个 Δ 的判定与 simpleboot 的 OPE 扫描器相同（θ 的序列不要求相同：simpleboot 的选点有随机性） |
| 文件管理（第 7.4 节） | 一次外层 Delaunay 运行：任何时刻运行目录的检查点数不超过保留数加并发数；暂存盘的峰值与第 7.4 节的估计相差在一倍以内；每次删除都在 `files.jsonl` 里 |

### 第 8 阶段：研究性的改进（各自独立，每一项先做不花钱的诊断）

| 项 | 先做的诊断（不用集群） | 做下去的条件 |
|---|---|---|
| 折角感知的端点收尾（切线交点或割平面） | 从已有的 navigator 记录里取边界两侧相邻的点对，算归一化梯度夹角随间距的变化 | 间距趋于 0 时夹角不趋于 0 |
| 曲率分级（OPE 行列精确、起点一次精确） | 在已有记录上回放 | 在 Ising Λ=11 加一个 OPE 坐标上点数减少 |
| 并行线搜索 | 无 | 槽位验收之后 |
| 二维追边界（第 6.8 节） | 无 | 第 6 阶段之后，Ising Λ=11 二维 navigator 追完整个岛，与 Delaunay 结果比较 |
| Hyperion 的强 Wolfe | 无 | 在第 5 阶段的问题上与基线比点数 |

### 先后与并行

```
第 0 阶段 ──► 第 1 阶段 ──┬──► 第 2 阶段（多节点槽位）
      │                  ├──► 第 3 阶段（二分）    ──┐
      │                  ├──► 第 4 阶段（Delaunay）──┴──► 第 7 阶段（OPE 扫描，以后）
      └──► 第 5 阶段的执行函数与梯度 ──(要第 1 阶段的外壳)──► 第 5 阶段 ──► 第 6 阶段 ──► 第 8 阶段
```

第 2、3、4、5 阶段互不依赖，可以由不同的人同时做；它们都只依赖第 1 阶段的 `point.py` 和 `search.py`，所以第 1 阶段的接口要先定稳。第 2 阶段排在单节点槽位验收之后紧接着做。

---

## 10. 决定和风险

### 10.1 已定的事（项目负责人，2026-11-15）

| 编号 | 问题 | 决定 | 写在 |
|---|---|---|---|
| D1 | 并行方式 | 一个作业里的槽位，不用作业数组；槽位可以跨多台节点（同一机型），一个驱动、一套记录和续算；`[slots]` 可以写每个槽位的核数、内存和总并发数，也可以写 `"bench"`；多节点槽位紧接在单节点槽位验收之后做 | 第 2.9 节；第 2 阶段 |
| D2 | 跳跃判定下的 `primal-dual optimal` | 算排除，与 Hyperion 相同 | 第 2.3 节 |
| D3 | 一个点没算完或停滞 | navigator：结束整个搜索。可行性类：跳过，记为 `skipped` 并写原因，不再提这个点；Delaunay 把它当作永不完成的顶点；二分的这条线结束、其他线继续 | 第 2.3、4.2、5.2 节 |
| D4 | 热启动超迭代 | 直接结束；靠“离已算点不太远才热启动”预防 | 第 2.6 节 |
| D5 | navigator 初始 Hessian 的默认 | Hyperion 的规则 diag(\|g\|/0.2/盒宽) | 第 6.5 节 |
| D6 | `approx_objective` 前的居中次数 | 自适应：按 ρ = R_error/μ 的门槛做，门槛由这一步需要的梯度精度定，够居中就是 0 次；每个运行开头校准一次；累计次数设上限，到了仍不够就结束运行 | 第 6.2 节（含通俗的说明）；第 5 阶段 |
| D7 | 新算法用不用混合精度 | 先不用 | 第 2.8 节 |
| D8 | 检查点保留 | navigator、Skydive 默认最近 2 个；可行性类默认 max(20, ⌈1.5 × 并发数⌉)；都可以在 `run.toml` 改；热启动取最近的；删除顺序按距离和最近使用 | 第 2.6 节 |
| D9 | OPE 扫描（切割面法） | 以后做，现在写进设计，单独一节，排进计划；重点是文件管理，这套文件管理也用于可行性类 | 第 7 节；第 7 阶段 |
| D10 | 参考结果 | 不放进公开仓库；以后有 Ising 的测试，可以放一个小的作为例子 | 第 9 节 |
| D11 | 内置模型（自旋外算符）上的新算法 | 以后再做，保持原样 | 第 1.4 节 |
| D12 | OPE 扫描 j ≥ 3（多分量的 OPE 方向）怎样证明“切空了” | 先只做 j = 2，j ≥ 3 以后再定 | 第 7.2 节 |
| D13 | `[files] budget_gb` 不写时要不要有默认上限 | 不设上限，只记录占用 | 第 2.6 节 |
| D14 | 单节点内的整轮同步要不要改成槽位各自流水 | 先整轮同步；按第 1、2 阶段测到的空等时间再定（R2） | 第 2.4 节 |

### 10.2 风险

| 编号 | 风险 | 对策 |
|---|---|---|
| R1 | 跳跃判定与精度有关，贴着边界的点在两种精度下可能不同；`primal-dual optimal` 算排除，离边界很近时可能不对 | 一个运行一个精度；第 1 阶段在边界附近的点上同时做符号判定 |
| R2 | 一台节点内整轮同步，快的槽位空等 | 每轮记空等时间；占得多再改流水 |
| R3 | 两个生成器进程同时写一个 block 目录是否安全，未验证 | 每台节点只有一个生成器进程、自己的 block 目录；改流水之前先验证 |
| R10 | 多台节点同时把检查点写到共享盘，盘的带宽不够 | 第 2 阶段记录检查点写入的用时；不够再定 |
| R11 | 跳过的点让 Delaunay 的边界带留下空洞、让二分的线提早结束 | `result.json` 列出跳过的点；用户另开点表复查 |
| R4 | 热启动让贴着边界的判定依赖于路径 | 记 `warm_from`；验收在冷启动下做；可以选 `warm_start = "cold"` |
| R5 | `sdpb` 的检查点能否给 `dynamical_sdp` 用，未验证 | 交接时冷启动，以后单独测 |
| R6 | navigator 的单个二次模型在折角上不收敛 | “目标坐标稳定”判据保证能停；第 8 阶段做折角处理 |
| R7 | 从岛外出发、实际没有岛时，navigator 一直下降不进岛 | 有 `max_points`；`stopped: positive minimum`；提醒用户用可行性运行检查 |
| R8 | Delaunay 在细长的岛上需要很多点 | `[coords].scale` 按岛的尺度给；以后加旋转 |
| R9 | 多槽位时 Delaunay 的路径不可逐点复现 | 验收在单槽位下做；多槽位只比判定和区域 |

---

## 11. 贡献指南

### 11.1 代码怎么写

- **Python**：只用标准库，版本按 `pyproject.toml`。数用 `fractions.Fraction` 或固定精度的 `decimal`，进出文件都是十进制字符串（`numbers.decimal_text`、`numbers.frac`）。
- **检查只在边界**：新节的所有检查写在 `inputs.load`（或它调用的函数），一次列出全部错误再退出（`SystemExit`），错误信息说清哪个键、为什么、怎么改。之后的代码信任输入，不再检查。
- **出错就停**：程序失败抛 `execute.StepFailed`，搜索按第 3.4 节结束；不要 `try` 掉再继续，不要重试。点只按第 2.3 节的规则跳过，不能拿跳过来绕开错误。
- **算法只做算术**：`Strategy` 不碰文件、不启动程序，这样它能在单测里用假的判定驱动，也能重放。
- **资源只从数据来**：进程数、槽位、并发数都从 bench 或首步实测来；`driver.log` 写明依据。没有数据就用整节点。
- **注释写为什么**：一个数为什么是这个值，依据是什么实测或哪个参考实现的行为。公开仓库里写一般性描述，不写路径、主机名、作业号。
- **命名**：完整的英文词；记录字段用小写加下划线；结局是 `状态: 原因` 的一句话（看 `skydive.ending`）。
- **小而独立的模块**：每个算法一个文件；共用的东西放 `point.py`、`search.py`，不要在算法之间互相导入。

### 11.2 测试怎么写

- 单测放在 `tests/test_<模块>.py`，用 `unittest`，不需要集群：执行层用 `unittest.mock` 换掉（看 `tests/test_skydive.py` 怎么用假的 `Dive` 驱动循环、怎么固定命令行），站点用 `tests/fixtures/sites/` 里的中性示例。
- 每个算法至少要有：用假判定（例如“x₁ + x₂ < 1 为允许”）跑完整个搜索、检查结局和点集的测试；中途截断记录再续算、结果相同的测试；输入检查的每一条错误都有一个测试。
- 命令行（`sdpb`、`approx_objective`、`qdelaunay`）与参考实现逐项对照的测试，照 `tests/test_skydive.py` 固定参考程序命令行的写法。
- **数值改动**要有一次回归或同样强度的对照（第 9 节各阶段的验收），结果写进提交说明或 interfaces。

### 11.3 怎样跑

```bash
bin/bootlab selftest
```

```bash
bin/bootlab regress
```

```bash
bin/bootlab regress check <run dir>
```

`selftest` 跑全部单测，不需要集群；`regress` 在集群上提交一个整节点作业，跑 Ising Λ = 11 的三步 Skydive，与参考程序比 ≥ 130 位（本机模式加 `--wait` 在前台跑）；`regress check` 检查任意一个运行目录。改了共用代码（生成、`pmp2sdp`、执行函数）一定要跑 `regress`。

### 11.4 怎样提交

- 一个提交只做一件事；提交说明写做了什么、为什么、怎么验证的（哪些单测、哪次回归、对照了几位）。
- 不提交运行目录、检查点、SDP；参考结果不进公开仓库（第 10 节 D10）。
- 保持文件原有的换行；提交前看一眼 `git diff --stat`，确认没有整文件的换行变化。
- 文档中英文同时改，内容一致。
- 用 AI 助手写代码时，把本文、[interfaces.md](interfaces.md) 和要改的模块一起给它，先让它写单测和验收计划，再写代码；它给出的每个数值约定都要用对照验证。[用 AI 写模型的指南](writing-models-with-ai.md)里“先给出完整的格式和约定，再看结果是否能逐项对上”的做法同样适用。
