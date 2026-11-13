# bootlab

**bootlab** 是在 Slurm 集群上做数值共形自举（conformal bootstrap）的轻量框架。

当前是**第一版，有意只做一种搜索算法：Skydive**（Liu、Simmons-Duffin、Su、van Rees 提出的 skydiving 算法，
[arXiv:2307.13046](https://arxiv.org/abs/2307.13046)），适用于外部算符都是标量的三维 CFT，求解器是 SDPB 的
`dynamical_sdp`。Navigator、可行性扫描与二分、Delaunay、OPE 扫描、带自旋的外部算符等以后再加，见[路线图](#路线图)。

[English](README.md)

## 工作方式

bootlab 分三层，层与层之间有两个固定接口：

```
model.json（bt-model/2）─┐   转换层（Python）                        执行层（同一个 Apptainer 镜像里的程序）
spectrum.toml ───────────┼─► 读入 ─► Skydive 循环 ─► 生成器输入 ═══► PMP 生成器 ═══► PMP ─► pmp2sdp ─► dynamical_sdp
objective.toml ──────────┘              ▲             （接口一）       （Haskell）   （接口二）                  │
run.toml ───────────────────────────────┘◄──────────── 每次求解一条记录：stepped / optimal / aborted ◄────────────┘
```

1. **物理层**（数据）：模型的交叉方程（`model.json`）、谱假设（`spectrum.toml`）、目标函数（`objective.toml`）。三者可以各自单独替换。
2. **转换层**（Python，只用标准库）：读入并校验一次；在每个点上精确地算出谱、写生成器输入；执行 Skydive 循环并记录。
3. **执行层**（bootlab 镜像里的程序）：PMP 生成器（Haskell，基于 Hyperion 和 scalars-3d）写出多项式矩阵规划，
   `pmp2sdp` 在节点本地盘上把它转成 SDP，`dynamical_sdp` 从上一步的检查点热启动求解，并给出下一步。

Skydive 每一步算当前点 `x` 及其 `k` 个邻点 `x + ε eᵢ`，求解一次，然后走到 `x + Δx`，或者结束运行并写下结局
（`converged`、`stopped`、`unfinished`、`failed`）。迭代或时间上限、停滞、崩溃这类异常情况都直接结束运行并记下原因，
不自动重试。续算由人手动发起，从最后留下的求解器检查点接着算。

## 主要特性

- **输入就是数据**：交叉方程写成 JSON 模型（沿用 simpleboot 的 VBlock 约定），谱和目标函数写成 TOML。谱里的值可以是点参数的
  精确表达式（如 `"deltaSig + 2"`），用有理数精确求值；所有数值都以全精度十进制字符串保存，不经二进制浮点数。
- **与参考程序对照过**：回归算例（Ising，Λ = 11）每一步的每个数与基于 Hyperion 的参考程序一致至少 130 位有效数字；
  `bin/bootlab regress` 在你的集群上重做这项对照。
- **资源配置由实测决定**：每次求解的进程数、同时跑几个 `pmp2sdp`、一个求解用几台节点，都按实测数据（`sites/<站点>/bench/`）
  或运行第一步的实测来定；只用物理核，不用超线程。
- **可移植**：集群探测（`bootlab probe`）测出节点型号、InfiniBand 端口、本地盘与共享盘、网络，`bootlab site gen`
  据此写出站点配置。主机上只要 Slurm 和 Apptainer，驱动用的 Python 3.12 从镜像里拷出。
- **可续算**：求解器的检查点就在运行目录里；每写一条记录，bootlab 用硬链接留住那次求解的检查点，作业在任何时刻被杀，
  都能从最后一条记录接着算。
- **记录直白**：每次求解一行 JSON（`solves.jsonl`），结局在 `result.json`，每条命令和每个程序错误输出的末尾在 `driver.log`。

与 simpleboot、Hyperion 的比较见 [docs/zh/comparison.md](docs/zh/comparison.md)。

## 环境要求

- Linux x86-64 集群，**Slurm ≥ 21.08**（要用 `srun --exact`、`--overlap`），`srun` 的 MPI 插件能与镜像里的 Open MPI 4.1
  配合（默认 pmix）。
- 登录节点和计算节点上有 **Apptainer ≥ 1.1**。
- **bootlab 执行层镜像**（`bootlab-exec`，r5 或更新的版本）。镜像暂不随本仓库发布，见[已知限制](#已知限制)。
- 一个作业的各节点都能看到的共享文件系统，放运行目录；必须支持硬链接。
- 建议有节点本地 scratch 盘（探测会实测它是否比共享盘快）。
- 可选：Slurm 配好的 cgroup v2（`pmp2sdp` 的内存看门狗要用；没有就关掉看门狗，并在 `driver.log` 里写明）。
- 一个求解跨多台节点需要 InfiniBand；只有以太网 TCP 时默认只用一台（见[接口文档](docs/zh/interfaces.md#multi-node-solves)）。

主机上不需要 Python。

## 快速开始

```bash
git clone <本仓库> bootlab && cd bootlab
module load apptainer                     # 或者用你的集群自己的办法让 apptainer 进 PATH
./install.sh --image /path/to/bootlab-exec-r5.sif --sha256 <镜像的 sha256>

# 探测拿不到的（你的 Slurm 分区、时限）；可以是空文件：这时用集群的默认分区和 48 小时时限
touch mysite.hand.toml

# 探测集群（每种节点型号一个小作业：8 核、16 GB、15 分钟），合并，生成站点
bin/bootlab probe plan --sif /path/to/bootlab-exec-r5.sif --out probe --submit
bin/bootlab probe merge probe              # 等探测作业都结束后
bin/bootlab site gen --probe probe/probe.json --hand mysite.hand.toml --out sites/mysite
echo 'site = "mysite"' >> bootlab.toml

bin/bootlab regress                        # Ising Λ=11 三步，与参考值对照：PASS / FAIL
bin/bootlab submit examples/ising/run.toml # 正式运行
```

[用户教程](docs/zh/tutorial.md)逐步讲解，并给出每一步的期望输出。

## 文档

| | English | 中文 |
|---|---|---|
| 用户教程：安装、探测、站点、运行、续算、读结果、回归 | [docs/en/tutorial.md](docs/en/tutorial.md) | [docs/zh/tutorial.md](docs/zh/tutorial.md) |
| 开发者接口：每一层、每种文件格式、每个配置项 | [docs/en/interfaces.md](docs/en/interfaces.md) | [docs/zh/interfaces.md](docs/zh/interfaces.md) |
| 与 simpleboot、Hyperion 的比较 | [docs/en/comparison.md](docs/en/comparison.md) | [docs/zh/comparison.md](docs/zh/comparison.md) |
| 用 AI 助手写新模型 | [docs/en/writing-models-with-ai.md](docs/en/writing-models-with-ai.md) | [docs/zh/writing-models-with-ai.md](docs/zh/writing-models-with-ai.md) |

## 路线图

第一版只做 Skydive 是有意的：先用一种算法把三层搭起来、与参考程序逐步对照通过，再加别的。以后每一项都只往同一个骨架上加东西，
不改两个固定接口。先后顺序不定。

- **Navigator**（用 `sdpb` 和 `approx_objective` 极小化 navigator 函数）。
- **可行性扫描与二分**：互相独立的点用 Slurm 作业数组。
- **Delaunay 区域搜索、OPE 扫描、极值泛函（EFM）谱提取。**
- **`bootlab bench`**：一条命令在一个真实的点上实测各项资源配置（block 表、`pmp2sdp`、求解器进程数），写进站点的 bench 文件。
- **带自旋的外部算符**（守恒流、应力张量的自举）：另写一个 PMP 生成器，输出同一种 PMP 格式。

## 已知限制

- 执行层镜像还没公开。没有镜像时，bootlab 只能跑单元测试。
- 只支持 d = 3、外部算符全是标量（生成器的 `bt-model/2` 格式和 scalar_blocks 的限制）。
- 把 simpleboot 的 AutoEqu/VBlock 文件转成 `bt-model/2` 的转换器在作者的工具链里，不在这次发布里；模型要手写或者借助 AI
  助手来写（[指南](docs/zh/writing-models-with-ai.md)）。
- 只在一个集群上开发和测试过（四种节点型号、InfiniBand、经 IPoIB 的 NFS）。探测和生成站点就是为别的集群准备的，但还没在别的
  集群上跑过。

## 致谢

bootlab 是新写的代码，不含下列项目的任何源代码。它的设计借鉴了这些项目的思路，它的执行层镜像按各自的（MIT）许可证运行它们的程序。

- **simpleboot**，作者 Ning Su（[gitlab.com/bootstrapcollaboration/simpleboot](https://gitlab.com/bootstrapcollaboration/simpleboot)），
  做自举计算的 Mathematica 框架。bootlab 沿用了 simpleboot 的 Skydive 驱动语义与选项含义、交叉方程的 VBlock 约定（即 `bt-model/2`
  格式）、Σ navigator 归一化，以及 block 截断的预设。
- **Hyperion**，作者 David Simmons-Duffin（[github.com/davidsd/hyperion](https://github.com/davidsd/hyperion)），在 HPC 集群上做并发
  计算的 Haskell 框架；以及建在它上面的自举库：**scalars-3d**（[gitlab.com/davidsd/scalars-3d](https://gitlab.com/davidsd/scalars-3d)）、
  [hyperion-bootstrap](https://gitlab.com/davidsd/hyperion-bootstrap)、[bootstrap-bounds](https://gitlab.com/davidsd/bootstrap-bounds)、
  [sdpb-haskell](https://gitlab.com/davidsd/sdpb-haskell)。镜像里的 PMP 生成器就是 scalars-3d 的代码，只加了一个小入口，用来一次写出
  一批点的 PMP；数值部分是上游原样的代码。
- **SDPB**，作者 David Simmons-Duffin、Walter Landry、Vasiliy Dommes（[github.com/davidsd/sdpb](https://github.com/davidsd/sdpb)）；
  `pmp2sdp` 和 `dynamical_sdp` 由它的 `skydiving-master` 分支编译。
- **scalar_blocks**，作者 Walter Landry、David Simmons-Duffin、Petr Kravchuk
  （[gitlab.com/bootstrapcollaboration/scalar_blocks](https://gitlab.com/bootstrapcollaboration/scalar_blocks)），在生成器内部计算共形 block。

在发表的工作中用到 bootlab 时，请引用它所运行的方法和程序：

- D. Simmons-Duffin, *A Semidefinite Program Solver for the Conformal Bootstrap*, JHEP 06 (2015) 174,
  [arXiv:1502.02033](https://arxiv.org/abs/1502.02033).
- W. Landry and D. Simmons-Duffin, *Scaling the semidefinite program solver SDPB*,
  [arXiv:1909.09745](https://arxiv.org/abs/1909.09745).
- M. Reehorst, S. Rychkov, D. Simmons-Duffin, B. Sirois, N. Su and B. van Rees, *Navigator Function for the
  Conformal Bootstrap*, SciPost Phys. 11 (2021) 072, [arXiv:2104.09518](https://arxiv.org/abs/2104.09518).
- A. Liu, D. Simmons-Duffin, N. Su and B. C. van Rees, *Skydiving to Bootstrap Islands*, JHEP 11 (2025) 003,
  [arXiv:2307.13046](https://arxiv.org/abs/2307.13046).

## 贡献者

- [dantengs](https://github.com/dantengs)：设计、物理、集群测试
- [Claude Code](https://claude.com/claude-code)（Anthropic）：与 dantengs 合作完成实现和文档

## 许可证

MIT，见 [LICENSE](LICENSE)。Copyright (c) 2026 dantengs。

执行层镜像里的程序（SDPB、scalar_blocks、Hyperion 及建在它上面的库）保留各自的 MIT 许可证和版权声明。
