# bootlab 与 simpleboot、Hyperion 的比较

bootlab 借鉴了两个成熟的框架：**simpleboot**（Ning Su；Mathematica）和 **Hyperion** 生态（David Simmons-Duffin；Haskell：Hyperion、
hyperion-bootstrap、bootstrap-bounds、scalars-3d、stress-tensors-3d）。本页说明 bootlab 在哪里沿用了它们、在哪里不同，以及它缺什么。
关于另外两者的说法，来自 bootlab 设计时阅读它们源代码的结果；关于 bootlab 的说法，来自它的规格和实测。这里没有三者之间的性能评测。

[English](../en/comparison.md)

## 一张表

| | simpleboot | Hyperion 生态 | bootlab v1 |
|---|---|---|---|
| 驱动的语言 | Mathematica | Haskell（经 Slurm 起 master 和 worker，distributed-process） | Python，只用标准库，每次运行一个 Slurm 作业 |
| 搜索算法 | 可行性、二分、Delaunay、OPE 扫描、Navigator、Skydive、EFM | Navigator 加 BFGS（hyperion-bootstrap）、Skydive（stress-tensors-3d）等 | **只有 Skydive** |
| 交叉方程 | Mathematica 里的 AutoEqu / VBlock | 每个模型一段 Haskell 代码（`ToSDP` 实例） | JSON 数据（`bt-model/2`，沿用 simpleboot 的 VBlock 约定） |
| 谱假设 | Mathematica 设置的一部分 | bound 的 Haskell 键的一部分 | TOML 数据；值可以是点参数的精确表达式 |
| 外部算符 | 标量（其他情形本次比较没有研究） | 标量、应力张量、守恒流（blocks_3d） | 只有 d = 3 的标量 |
| 运行环境 | 笔记本（Docker）或集群（Apptainer） | 集群，每台机器单独配置编译（`hyperion-config`） | Slurm 集群，一个预先构建的镜像，站点由探测生成 |
| 资源配置 | 用户配置 | 程序配置 | 由实测数据决定（bench 文件、第一步实测的内存） |

## bootlab 沿用了什么

- **Skydive 的语义**。循环（中心点和 `k` 个邻点、每步调用一次 `dynamical_sdp`、BFGS 状态逐步传递、找边界）照基于 Hyperion 的参考程序的
  Skydive 循环，而后者又照 simpleboot 的驱动。选项名一一对应。
- **交叉方程的约定**。模型格式用 simpleboot 的 VBlock 写法（每个 channel 对每条方程给一个 j × j 的 F/H 项矩阵），所以 simpleboot 的方程可以
  逐项翻译。与上游 block 的两处约定差异（λ 的顺序及其 (−1)^ℓ；混合关联函数的前置因子）在生成器里统一处理，已在 Ising 体系和一个更大的混合体系上与
  simpleboot 对照过。
- **Σ navigator 归一化**就是 simpleboot 的 `SigmaNavigatorNormalizer`（符号相同：N < 0 为允许）。另有仿照 Hyperion Ising navigator 的 GFF
  归一化；在三个 Ising 点上，它与参考程序内置的 GFF navigator 一致 29–32 位。
- **block 截断预设**：simpleboot 的（按 Λ）和 Hyperion 的（按 nmax）。
- **PMP 生成的数值代码是 Hyperion 的**。镜像里的生成器就是 scalars-3d 加一个小入口；中心点的 PMP 部件与参考程序写出的逐字节相同。

## bootlab 有什么不同

**输入是数据，只检查一次**。三个物理文件都是数据；`inputs.load` 在提交任何东西之前检查一次，列出全部错误（未知的键、错误的自旋、设了两次的
gap、已经丢了位数的 TOML 浮点数）。谱的值是点参数的精确表达式（不是模型坐标的参数成为额外的搜索坐标），用有理数精确求值。在 Hyperion 里，
新模型是 Haskell 代码；在 simpleboot 里是 Mathematica 代码。

**数值保持精确**。点、邻点（`x + ε eᵢ`）和谱的值都是精确十进制数；记录以字符串保存每一位。回归要求每一步都与参考程序一致 ≥ 130 位
（剩下的差别来自参考程序用 512 位二进制浮点数算邻点）。

**结局写得明白**。每次运行都以一个写下的结局结束：`converged`（allowed、excluded 或 boundary，先核对原始和对偶目标值同号）、`stopped`
（出盒子、optimal 时 N 正负未定、N 为负）、`unfinished`（步数、迭代数或时间上限；求解器停滞）或 `failed`（某个程序失败）。simpleboot 的
Skydive 驱动遇到"update"以外的任何求解器结束情况都停下，不加区分，也不检查盒子。bootlab 里没有任何自动重试。

**与 simpleboot Skydive 驱动不同的默认值**（每一项都可以在 `run.toml` 里改回去）：

| | simpleboot 驱动 | bootlab |
|---|---|---|
| 有限差分步长 ε | 1e-40 | 1e-20；小于 1e-25 被拒，因为生成器给 block 表的键只用 Δ12 的约 31 位（simpleboot 给 scalar_blocks 传 Δ12 用 50 位，所以 1e-40 在那里没问题） |
| 间隙上限 | `Automatic[1]`：只限第一次调用 | 第一次调用用设定值，之后跟着 N 走：`min(上限, max(c·|N_prev|, κ·|N_1|, 1000·floor))`；写 `"off"` 得到固定或只限第一次的行为 |
| 停止阈值 | 求解器选项，固定 | 随 N 收紧：`max(1e-30, min(设定值, 1e-3·|N_prev|))`；可以关掉 |
| 传给求解器的迭代数 | 每次都传 0 | 累计 |
| 盒子 | 不检查 | x + Δx 出盒子就 `stopped: left box` |
| 找边界方向的门控 | 某一步出现 Dobj < 0 之后 | 某一步 N < 0 之后（`find_boundary_when_allowed`），或从第一步起 |

**任何时刻被杀都能续算**。求解器的检查点直接写在运行目录里，每写一条记录，bootlab 用硬链接留住那次求解的一代（`checkpoint/kept/<k>`）。
手动续算时回到最后一条记录的那一代并热启动；这一点在写检查点的半途发 SIGKILL 测试过。在 simpleboot 里，被杀的点要先 Drawback，再提交同一个
task。

**资源由实测决定**。求解器的进程数、同时跑几个 `pmp2sdp`、一个求解用几台节点，都按实测数据选；`pmp2sdp` 的内存在第一步实测，看门狗在内存
耗尽之前撤掉整波（否则 `pmp2sdp` 会停住而不是报错）。只用物理核。

**生成器的调度是 bootlab 自己的**。数值代码是上游的；只在实测显示有等待的地方改了外面的调度，每次改动后部件都逐字节不变：

| 改动 | 实测（一步，128 核） |
|---|---|
| Σ 归一化拆成单项、在全部核上并行、按上游原来的顺序相加（bootlab 的第一版生成器原样用上游的调度） | Ising Λ = 19：701.7 s → 66.1 s |
| 每个点的计划在 worker 上并行地只算一次；归一化部件先派发 | 一个 7 坐标的模型，Λ = 23，8 个点：1310.7 s → 670.6 s |

**靠探测移植**。`bootlab probe` 测出节点型号、InfiniBand 端口、scratch 盘和网络；`bootlab site gen` 据此写出站点，并写明每项选择的依据。
主机上只要 Slurm 和 Apptainer。Hyperion 程序要按每台机器配置并编译；simpleboot 提供 Docker 和 Apptainer 的工作区。

## bootlab（还）没有的

- Skydive 以外的任何算法：没有 Navigator、可行性扫描、二分、Delaunay、OPE 扫描、EFM。
- simpleboot Skydive 驱动的额外功能（GCP 扫描与贴附、爬升后的 β 覆盖、从以前的 `SDPBReturn` 起步）。
- 带自旋的外部算符，以及其他维数。
- 交互使用：没有 notebook 前端，没有笔记本模式；每次运行都是一个 Slurm 作业。
- 本次发布里没有从 simpleboot 文件转换的工具，也没有公开的执行层镜像。
- 只在一个集群上测试过。

simpleboot 和 Hyperion 生态是成熟的研究工具，已经用于大量发表的结果；bootlab v1 是一个范围很窄的新代码。在它的 Skydive 和记录方式适合你的
工作时使用它；有疑问时，与成熟的工具对照。
