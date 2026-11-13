# bootlab 用户教程

本教程从一个空目录开始，一直走到一次完整的 Skydive 运行：安装、探测集群、生成站点、对照参考值检查安装、写 `run.toml`、运行、
续算、读结果。每一步都给出命令和它的输出。

除非特别说明，命令都在集群的**登录节点**上运行。例子里的路径和名字（`/shared/...`、`node01`、`cpu128`）只是占位，请换成你的集群的。

[English](../en/tutorial.md)

## 0. 需要准备什么

- Slurm ≥ 21.08、Apptainer ≥ 1.1（见 README 的[环境要求](../../README.zh-CN.md#环境要求)）。
- bootlab 执行层镜像 `bootlab-exec-<版本>-r5.sif`（或更新的版本）及其 sha256。bootlab 运行的所有程序（PMP 生成器、`scalar_blocks`、
  `pmp2sdp`、`dynamical_sdp`、Open MPI、UCX）和驱动用的 Python 3.12 都在镜像里。镜像暂不随仓库发布。
- 一个所有计算节点都能看到的共享文件系统上的目录（放安装目录和运行目录）。

## 1. 安装

```bash
cd /shared/$USER
git clone <本仓库> bootlab
cd bootlab
module load apptainer            # 用你的集群自己的办法让 apptainer 进 PATH
./install.sh --image /shared/images/bootlab-exec-r5.sif --sha256 <镜像的 sha256>
```

`install.sh` 只需要 bash 和 apptainer。可选参数：`--site <名字>`（默认站点，写进 `bootlab.toml`），`--runs <目录>`
（运行目录放在哪里，默认是安装根目录下的 `runs/`）。

它做的事和输出：

```
image /shared/images/bootlab-exec-r5.sif: sha256 checked
python: 2259 files from the image checked; 3.12.15
wrote /shared/<you>/bootlab/bootlab.toml
wrote /shared/<you>/bootlab/bin/bootlab
......................................................................
----------------------------------------------------------------------
Ran 159 tests in 6.0s

OK
```

1. 核对镜像的 sha256（给了 `--sha256` 时）；
2. 把镜像里的 `/opt/bootlab/hostpython` 拷到 `python/`，按它的 MANIFEST 逐个核对文件。`python/bin/python3` 用镜像自带的加载器和库在
   主机上运行镜像里的 Python 3.12，不设 `LD_LIBRARY_PATH`、`PYTHONHOME`，所以它启动的程序（`srun`、`sbatch`）看到的环境是干净的；
3. 写 `bootlab.toml`（安装配置，键的说明见 `bootlab.example.toml`），已存在就不动；
4. 写 `bin/bootlab`，以后都用这个命令，已存在就不动；
5. 用这个 Python 跑单元测试。

Apptainer 自己可能打印几行 `INFO:`，无关紧要。`apptainer` 不在 `PATH` 上时，`install.sh` 会说明原因并停下。已有 `python/` 时它也停下：
它从不覆盖（要重新安装，先把旧的挪开）。测试的个数会随代码增加，要看的是最后的 `OK`。

此后一律用 `bin/bootlab <命令>`；在任何目录下都能用，从别处建符号链接指向它也行。

## 2. 探测集群

bootlab 对你的集群不做任何假设：先由探测实测，再根据实测结果生成站点配置。

### 2.1 单节点探测（每种节点型号一个）

```bash
bin/bootlab probe plan --sif /shared/images/bootlab-exec-r5.sif --out probe --submit
```

`probe plan` 在登录节点上读 `sinfo`、`scontrol show config` 和 `srun --mpi=list`，把节点分成型号（按 Slurm feature；没有 feature 时
按插槽 × 核数、线程数、内存和 CPU 型号），然后给每种型号在一台空闲节点上交一个小作业：8 核、16 GB、`--hint=nomultithread`、15 分钟。
每种型号打印一行：

```
cpu128: 12 nodes, 128 physical cores, 515000 MB; node probe on node03
  Submitted batch job 81234
gpu96: 2 nodes, 96 physical cores, 1031000 MB; not enough idle nodes for a node probe
```

常用选项：

- `--shared <目录>`：在哪个目录里测共享文件系统。默认是本安装的运行目录。**在你将来放运行目录的地方探测**：同一个挂载点背后可能是
  速度相差很大的几组盘阵，而 scratch 的选择就取决于这个数。开跑时，如果运行目录不在探测量过的目录之下，`driver.log` 会写一条警告。
- `--local <目录>`（可重复）：在常规候选（`$TMPDIR`、`$SLURM_TMPDIR`、本地块设备的挂载点、`/scratch`、`/local`、`/localscratch`、`/tmp`、
  `/var/tmp`）之外，再试一个节点本地目录做 scratch。本地 scratch 目录本身不是挂载点时要用它。
- `--types a,b`：只测这几种型号。
- 不加 `--submit`：只打印 `sbatch` 命令、写 `probe/plan.json`。

每个节点作业写出 `probe/node-<主机>-<作业号>.json` 和 `probe/slurm-probe-node-<作业号>.out`。测的项目包括：物理核、内存、InfiniBand
端口及其状态、网卡、每个 scratch 候选（能否写、剩余空间、所在设备）、按 bootlab 真实文件形态（4 MiB 的文件、8 个写进程、共 2 GiB，
做三遍）测共享目录、运行目录能否建硬链接、`apptainer` 从哪里来、镜像里 UCX 的传输层、cgroup v2。

**当时没有空闲节点的型号**：之后探测到一个**另外的**目录，再把记录拷过来：

```bash
bin/bootlab probe plan --sif <镜像> --out probe-later --types gpu96 --submit
# 这个作业结束后：
cp probe-later/node-*.json probe/
```

不要把 `--types` 的探测直接写进 `probe/`：`plan.json` 会被重写，只剩这几种型号，由它生成的站点就会缺少别的型号。

### 2.2 两节点探测（只在一个求解要跨多台节点时需要）

一个求解用一台节点就够时跳过这一节：没有两节点探测，站点就只用单节点。

```bash
bin/bootlab probe plan --sif <镜像> --out probe --what pair --sdp <一个小 SDP 目录> --submit
```

每种有两台空闲节点的型号交一个作业，两台整机独占，最多一小时。它测：轻量 `srun` 步能否到达两台；每个 InfiniBand 口、每张 TCP 网卡上
`ucx_perftest` 的带宽和延迟；全规模 MPI 启动（每个物理核一个 rank，把给定的 SDP 解 2 次迭代，做三遍），传输层分别是 InfiniBand rc、
IPoIB 上的 TCP、以太网上的 TCP；按 bootlab 的方式在两台之间拷 2 GiB（能 `ssh` 进作业节点时也用 `ssh` 拷一次）；另一台从共享盘冷读。
结果在 `probe/pair-<主机1>-<主机2>-<作业号>.json`。

`--sdp` 给一个精度 785 的小 SDP 目录即可（一个 Ising Λ = 11 的 SDP，约 24 MB，就够）。一种得到它的办法：把第 5 节的例子跑到结束，
终点的 PMP 留在 `<运行目录>/endpoint/pmp/`，bootlab 用过的完整 `pmp2sdp` 命令在 `<运行目录>/steps/<n>/` 的 `.cmd` 文件里，在一个小作业
里照样运行一次，把 `--output` 换成你的目录。

### 2.3 合并

```bash
bin/bootlab probe merge probe
```

打印 `probe/probe.json`：其中有 plan、全部节点和两节点记录，以及每台节点的 scratch 选项。

## 3. 生成站点

探测拿不到你的 Slurm 政策，写在一个小的手写文件里。这个文件可以是空的（`touch mysite.hand.toml`）：这时作业交到集群的默认分区，
时限 48 小时。你的集群需要什么就加什么：

```toml
# mysite.hand.toml
[slurm]
partition = "batch"
time = "48:00:00"
# account = "myproject"
# extra = ["--qos=normal"]

# 可选：
# bench = "../othersite/bench"        # 实测的资源数据（第 9 节），相对于 sites/<名字>/
# [ib]
# exclude = ["node07=mlx5_0:1"]        # 已知会掉线的 InfiniBand 口
# [multinode]
# transfer = "ssh"                     # 覆盖生成的 [multinode] 里的键
```

然后：

```bash
bin/bootlab site gen --probe probe/probe.json --hand mysite.hand.toml --out sites/mysite
```

它写出 `sites/mysite/site.toml` 和 `sites/mysite/setup.sh`，并打印每项决定的依据，例如：

```
node probes: 81234, 81235; pair probes: 81240
node03: /tmp 0.69 s, /local 0.69 s, shared 4.97 s (node probe 81234, pair probe 81240)
...
scratch /local/$USER-bootlab, the fastest measured: /local 6.77 s; /tmp 7.73 s; shared 34.40 s (write until close + read of 2 GiB in bootlab's file pattern, summed over ...); shared measured in /shared/<you>/bootlab/runs (...)
IB ports the probe chose (ib:auto applies the same rule): node03=mlx5_1:1, node04=mlx5_0:1
sites/mysite/site.toml
```

各项决定（细节见[接口文档](interfaces.md#site-generation-rules)）：

| 项目 | 规则 |
|---|---|
| scratch | 实测最快的方案（节点本地路径或共享盘），按探测过的节点求和；与另一类最好的相差不到 10 % 或 2σ 时注明"接近" |
| MPI 传输层 | 探测看到 ACTIVE 的 IB 口就用 InfiniBand rc（每次开跑时各节点自己选口，`ib:auto`），否则用以太网网段上的 TCP |
| 一个求解跨多台节点 | 只给两节点探测里 IB rc 全规模启动三次全过、且有可写本地盘的型号；其余默认单节点（运行可以用 `allow_multinode = true` 覆盖） |
| 节点间拷贝 | `srun` + TCP 发给 bootlab 的接收端；两台共有 IPoIB 网时走 IPoIB |
| 共享盘走向检查 | 记下各探测节点到 NFS 服务器的地址和网络；开跑时某台不同就在 `driver.log` 警告 |

生成的文件是普通 TOML，每个值后面有注释写明它的来源，可以手动改。要和手写的站点比较：
`bin/bootlab site diff sites/mysite/site.toml <另一个 site.toml> --probe probe/probe.json`。

设为默认站点：

```bash
echo 'site = "mysite"' >> bootlab.toml
```

（也可以在每个 `run.toml` 里写 `[exec] site = "mysite"`。）

## 4. 检查安装：`bootlab regress`

```bash
bin/bootlab regress                    # 或：bin/bootlab regress --node node03 --time 01:00:00
```

它把回归输入（Ising 例子，Λ = 11，三步）写到 `runs/regress-input-<时间>/`，交一个整机作业，打印输入目录和 `sbatch` 命令：

```
# inputs /shared/<you>/bootlab/runs/regress-input-20261113-202539
# the nodes named in run.toml: node03
sbatch --job-name=bootlab --exclusive --hint=nomultithread --mem=0 --time=01:00:00 ... --wrap ...
Submitted batch job 81250
```

在 96 核的节点上作业只要几分钟。作业输出 `runs/slurm-<作业号>.out` 的结尾：

```
     step         x navigator  ext_step  gradient   hessian    primal      dual       gap   alpha_p      beta    reason iteration   climbed find_mini
        1       all     196.9     133.9     133.9       all     196.9     197.1     196.9     139.7       all         =         =         =         =
        2     140.2     139.7     133.9     133.8     133.6     139.7     139.9     139.7     139.0       all         =         =         =         =
        3     140.0     139.3     133.4     133.6     133.7     139.3     139.4     139.3     139.4       all         =         =         =         =
least digits: {"x": 140.0, "navigator": 139.3, "ext_step": 133.4, "gradient": 133.6, "hessian": 133.6}
PASS: 3 steps, every number to 130 digits or more
```

每个数是 bootlab 的值与参考值（`tests/reference/ising-l11/bt-steps.json`，即基于 Hyperion 的参考程序的三步）一致的有效数字位数；
`all` 是每一位都相同，`=` 是离散量相等。每个数都一致 130 位以上才算通过（参考程序用 512 位二进制浮点数算邻点，bootlab 用精确十进制，
所以导数只一致到约 134 位，不是每一位）。运行目录里另有 `regress.json`；FAIL 时退出码为 1。

`bin/bootlab regress check <运行目录>` 在任何地方对一个已有的运行目录重做这项比较。

## 5. 写一次运行

一次运行就是一个目录里的四个输入文件。`examples/ising/` 是三维 Ising 的 σ–ε 体系：

```
examples/ising/
  model.json        交叉方程（bt-model/2）
  spectrum.toml     间隙
  objective.toml    用模型里的哪个目标函数
  run.toml          这次运行：输入、数值设置、坐标、Skydive 选项、求解器选项
```

`run.toml`（节选；完整的键见[接口文档](interfaces.md#runtoml)）：

```toml
format = "bootlab-run/1"
name   = "ising-l11-skydive"

[inputs]
model     = "model.json"
spectrum  = "spectrum.toml"
objective = "objective.toml"

[numerics]
lambda       = 11              # 导数阶数，奇数
block_preset = "simpleboot"    # 按 Λ 取 block 截断

[coords]
names = ["deltaSig", "deltaEps", "theta"]   # 必须等于模型的坐标（再加谱里额外引用的参数）

[skydive]
start     = ["0.519", "1.4", "0.956"]       # 字符串：精确的十进制数
max_steps = 300
box_min   = ["0.517", "1.2", "0.95"]
box_max   = ["0.530", "1.5", "0.99"]
gap_upper_limit = "0.1"
infinitesimal   = "1e-20"

[solver]          # SDPB 的选项，键就是命令行参数名，覆盖本 Λ 的默认值
[dynamical]       # dynamical_sdp 自己的选项
# [exec] site = "mysite"                    # 不写就用 bootlab.toml 的 site
# [resources] nodes = "auto"                # 一个求解跨多台节点（第 8 节）
```

值得记住的规则：

- 数值写成**字符串**（`"0.519"`、`"1e-20"`）或整数。TOML 浮点数（如 `0.519`）是错误：它已经丢了你写下的十进制位。
- 谱里的值可以引用点参数：`delta = "deltaSig + 2"`。引用的参数如果不是模型坐标，就成为额外的搜索坐标，接在模型坐标后面。
- `inputs.load` 把所有内容检查一遍，错误一次全部列出；没有错误才开跑。

不交作业、只检查输入：

```bash
bin/bootlab submit --dry examples/ising/run.toml
```

它读入并检查输入、选节点，打印 `sbatch` 命令但不执行。

## 6. 运行

```bash
bin/bootlab submit examples/ising/run.toml
```

`submit` 选节点（默认一台；多台见第 8 节），打印理由和 `sbatch` 命令，然后提交。作业输出在 `runs/slurm-<作业号>.out`；运行目录是
`runs/<名字>-<年月日-时分秒>/`（时间是节点的本地时间）。

运行中：

```bash
tail -f runs/ising-l11-skydive-*/driver.log
```

`driver.log` 开头写站点、scratch、镜像（及其 sha256）、每台节点的 MPI 设备和到共享盘的路由、这次的 `run.toml`、谱的警告；之后每一步
一段：生成器、`pmp2sdp` 的各波、求解器，以及各自的用时。

这个例子用的是参考程序自己的设置（`gap_upper_limit_track = "off"`），几步之后就会走出盒子，以 `stopped: left box` 结束。真正做搜索时，
用第 7 节的默认设置（间隙上限随 N 变化）。

## 7. 结局和记录

每次运行都以 `result.json` 里的一个结局结束（`status`、`reason`）：

| status: reason | 什么时候 |
|---|---|
| `converged: allowed` / `excluded` | 求解器以 `optimal` 结束，原始和对偶目标值同号：N < 0（允许）或 N > 0（排除）；终点的 PMP 和求解输出拷到 `endpoint/` |
| `converged: boundary` | 设了 `find_boundary_direction`：optimal 时异号；N 报 0，`n_bound` = 间隙（|N| ≤ 间隙） |
| `stopped: N undecided at optimal` | 极小化 N 时 optimal 却异号：收紧求解器的间隙阈值后续算 |
| `stopped: left box` | 下一个点 `x + Δx` 在 `box_min`–`box_max` 之外 |
| `stopped: negative navigator` | `stop_on_negative = true` 且 N < 0 |
| `unfinished: max steps` / `unfinished: <求解器原因>` | 步数、迭代数或时间用完，SIGTERM |
| status 为 `unfinished`，reason 为 `stalled: <求解器原因>` | 求解器停滞：`maxComplementarity`、原始或对偶步长太小 |
| `failed: <程序> exit <退出码>` | 某个程序失败，或者它的输出缺失 |

运行目录：

```
run.toml、model.json、spectrum.toml、objective.toml   输入的副本（续算读的是这些）
driver.log      每条命令行、每个程序 stderr 的末尾、每项决定及其依据
solves.jsonl    每次求解一行：step、x、N、primal、dual、gap、Δx、梯度、Hessian、迭代数……
result.json     结局、终点、输入和镜像的 sha256、scratch、开始和结束时间
summary.md      设置摘要、逐步表格（数截到 8 位）、结局
steps/<n>/      求解器的小输出文件，以及每个程序的 .cmd/.out/.err
checkpoint/     求解器的检查点和 kept/<k>/（converged 时删除）
endpoint/       converged 时：pmp/（终点的 PMP 部件、labels.json、precision）和 out/
```

`summary.md` 在结束时写出；`bin/bootlab summary <运行目录>` 随时可以从 `solves.jsonl` 重新生成它（例如运行还没结束时）。
`solves.jsonl` 和 `result.json` 里的所有数值都是全精度的十进制字符串。

做真实搜索时最要紧的几项设置（括号里是默认值；细节见[接口文档](interfaces.md#skydive-loop)）：

- `gap_upper_limit`（必须写；生产用的例子取 `"1e-12"` 或更小）：第一次求解的对偶间隙上限。它必须大于求解器的
  `dualityGapThreshold`，否则第一次求解就直接以 `optimal` 结束；`load` 会拒绝这种设置。
- `gap_upper_limit_track`（`"1"`）：从第 2 步起上限取 `min(gap_upper_limit, max(c·|N_prev|, κ·|N_1|, 1000·floor))`，κ =
  `gap_upper_limit_track_floor`（`"0.1"`）；写 `"off"` 就是固定上限。
- `gap_threshold_tighten`（`true`）：求解器的停止阈值变为 `max(1e-30, min(设定值, 1e-3·|N_prev|))`；找边界的运行要写 `false`。
- `infinitesimal`（`"1e-20"`）：邻点的有限差分步长；小于 `1e-25` 会被拒绝（生成器给 block 表的键只用 Δ12 的约 31 位）。
- `find_boundary_direction`：N < 0 之后沿这个方向推（空 = 极小化 N）。

## 8. 续算，以及多节点

运行以 `unfinished`、`stopped`、`failed` 结束，或者作业被杀了，都可以**续算**：

```bash
# 可以先改运行目录里自己的 run.toml，例如放大 max_steps
bin/bootlab submit --resume runs/ising-l11-skydive-20261113-202539
```

续算读运行目录里当前的 `run.toml`，从 `solves.jsonl` 的最后一行恢复 x 和 BFGS 状态，接着在同一个目录里追加记录。`driver.log` 写明走的是
哪条路：

- `resume: checkpoint generation <g> kept for record <k>, 1 x cpu128 x 128 processes`：从为最后一条记录留下的检查点热启动（正常情况；
  在写检查点的半途被杀也是这样）；
- `resume: layout changed ...` 或 `resume: no checkpoint kept`：冷启动，求解器从头开始。它的第一次求解用 `gap_upper_limit`，所以这个值要
  ≤ `1e-12`。

`submit --resume` 按检查点的布局要节点（原来的节点空闲就用原来的，否则用同型号的其他空闲节点）。

**一个求解跨多台节点**（只限 InfiniBand）：

```toml
[resources]
nodes = "auto"                 # 或者同一型号的主机列表
# max_nodes = 2; mem_gb = 300; prefer = ["cpu128"]
# allow_multinode = true       # 覆盖站点"默认单节点"的设定（driver.log 会写明风险）
```

写 `"auto"` 时，只有站点的 bench 里有这个模型、这个 Λ 的求解实测，而且多加一台划算（效率 t(1)/(k·t(k)) ≥ 0.8，加速 t(k−1)/t(k) ≥ 1.15），
`submit` 才多加一台节点。没有这样的数据就用一台，并写明原因。

## 9. 实测的资源数据（bench）

`sites/<站点>/bench/<节点型号>.json` 存实测数据：不同进程数、节点数下求解器每次迭代的秒数，`pmp2sdp` 的实例数和内存，block 表的线程数，
生成器的用时。运行据此选求解器每节点的进程数（问题规模最接近的记录里最快的进程数，且进程网格不退化），以及 `nodes = "auto"` 时的
节点数。没有 bench 时，求解器用一台节点的全部物理核，同时跑几个 `pmp2sdp` 由运行第一步实测（一个实例的内存峰值）决定。生成的站点没有
自己的 bench，要在手写文件里用 `bench = ...` 指向已有的实测数据。格式见[接口文档](interfaces.md#bench-files)。

这些默认值是稳妥的，不需要任何实测。想要更好的布局，就实测你自己的问题：在你的 Λ 下用你的模型跑一步（`max_steps = 1`），每节点试几种
进程数（例如 64、128 和全部物理核；每次迭代的秒数 = `solves.jsonl` 里这一步的 `seconds.dive` ÷ `iterations`），有 InfiniBand 时再在两台节点上跑一次，
然后把结果按 `model`、`lambda`、`N`、`blocks`、`nodes`、`procs_per_node`、`s_per_iter` 写进 `sites/<站点>/bench/<节点型号>.json`。
自动完成这件事的 `bootlab bench` 命令在路线图里。

## 10. 出了问题怎么办

| 现象 | 去哪里看 |
|---|---|
| `submit` 打印 `the inputs have errors:` | 列表逐项写明哪个键、错在哪里；改输入文件 |
| 运行以 `failed: generator exit 1` 结束 | `steps/<n>/generator.err`（模型文件的错误由生成器报告） |
| `failed: pmp2sdp ...` 或某一波被撤 | `driver.log`：每个实例的内存、看门狗的说明 |
| 求解器停滞（`stalled: maxComplementarity`、步长太小） | `[solver]` 里的求解器选项（初始矩阵缩放、`maxComplementarity`） |
| `warning: the runs directory ... is not in the directory the probe measured` | 用 `--shared` 在你的运行目录里重新探测；不在乎速度时可以不管 |
| 某台节点到共享盘的路由有 `warning:` | NFS 挂载退到了较慢的网络；运行照常进行 |
| 续算变成了冷启动 | `driver.log` 写明原因（布局变了、没有留下检查点） |

bootlab 从不自己重试。查清原因、改好之后续算。
