# 四轮足速度指令 + 机械臂末端位姿：GPU 4096 环境完整训练流程

目标：训练**一个**全身策略（22 路输出：12 腿位置 + 4 轮速度 + 6 臂位置），同时接收

- 底盘速度指令：机体系 `vx, vy(=0), wz`
- 末端（TCP）目标位姿：`x, y, z, qw, qx, qy, qz`

完整流程：**物理筛选的参考轨迹 → 世界模型/BC 预训练 → 4096 环境 MuJoCo Warp 在线采集 + 世界模型 BPTT → CPU MuJoCo 独立评估 → 导出**。

一键脚本：

```bash
bash scripts/wf_cmd_pipeline.sh          # 阶段 0..4 全部
bash scripts/wf_cmd_pipeline.sh 1 4      # 重新生成参考并开始一次新训练
```

包含阶段 1 时，每次运行自动创建一对带相同时间后缀的新目录，例如：

```text
outputs/wf_cmd_gpu_20261114_203000_123456/
outputs/wf_cmd_refs_20261114_203000_123456/library/
```

时间戳使用上海时区，精确到微秒；目录重名时自动增加编号。
训练目录中保存本次专用的 `pipeline_config.json`，其中 `output` 和
`demonstrations.directory` 已同步到新路径。原始配置保持不变，所有后续阶段都使用这份专用配置。
`REFS` 环境变量可指定参考目录前缀，配置中的 `output` 指定训练目录前缀。

继续某次运行时，从终端输出中找到其 `Pipeline config`，指定这份配置并从阶段 2 开始：

```bash
CONFIG=outputs/wf_cmd_gpu_20261114_203000_123456/pipeline_config.json \
  bash scripts/wf_cmd_pipeline.sh 2 4
```

阶段 2～4 会使用指定配置中的已有目录；存在 `ckpt_latest.pt` 时自动续训。
仅执行阶段 0 也不会分配新目录。手动运行下文的单模块示例时，仍可自行指定固定目录。

---

## 0. 这次改了什么（相对原仓库）

| 文件 | 改动 |
|---|---|
| `src/supermimic/wf/robot.py` | 读取 `robot.json` 的 `actuator_parameters`（腿/臂 PD、轮 kd、臂力矩上限 30 Nm）；轮半径改为从 URDF 圆柱读取（0.087 m，原代码写死 0.093）。修正后 10 s 静止保持：高度 0.4171 m、倾角 0.0790 rad，与 `wf-arm-tbot/leg_gain_update_validation.json` 的记录一致 |
| `src/supermimic/wf/simulator_warp.py`（新） | MuJoCo Warp 批量仿真后端，接口与 `BatchedSimulator` 相同：每个物理子步重新计算 PD 力矩、每个环境单独设置摩擦、接触/滑移/安全指标取自真实求解器结果，支持 CUDA Graph |
| `src/supermimic/wf/command_reference.py`（新） | 生成"速度指令 + 末端位姿"参考轨迹，并在真实物理中筛选；多进程并行 |
| `src/supermimic/wf/train.py`、`config.py` | `sim.backend="warp"`；预先 settle 好的重置状态库（`reset_bank_batches`）；`checkpoint_replay=false`，不把几百 MB 的回放数据写进 checkpoint |
| `src/supermimic/wf/simulator.py` | 增加 `set_friction` |
| `configs/examples/wf_cmd_gpu4096.json`（新） | 正式训练配置 |
| `configs/examples/wf_cmd_smoke.json`（新） | 16 个环境的冒烟配置 |
| `scripts/wf_cmd_pipeline.sh`（新） | 服务器上的一键流程 |
| `tests/test_wf_command_gpu.py`（新） | 配置、执行器参数、Warp/CPU 一致性、参考生成 |
| `pyproject.toml` | 新增 `gpu` extra，以及 `sm-wf-command-refs`、`sm-wf-warp-check` 两个入口 |
| `src/supermimic/wf/goal_trajectory.py`（新，第二轮） | 按 level 生成末端目标轨迹库；训练和评估共用的 Ta/Tb 调度器 |
| `evaluate.py`、`train.py`、`guard.py`、`config.py`（第二轮） | 新增 `trajectory` 配置组和跟踪成功率指标；`curriculum.metric`；守卫新增 `success_metric`、`success_tolerance`、`rejection_lr_factor` |
| `scripts/prepare_wf_run.py`、`wf_cmd_pipeline.sh`（第二轮） | 阶段 1 同时生成轨迹库，放在本次运行的参考目录下 |
| `src/supermimic/wf/dagger.py`（新，第三轮） | DAgger：策略与教师混合执行，离线教师在策略走到的状态上给出纠错标签，加入 BC |
| `tests/test_wf_dagger.py`（新，第三轮） | DAgger 配置、标签聚合、保存与加载，以及 β=1 时教师复现 clip 动作 |
| `tests/test_wf_trajectory.py`（新，第二轮） | 调度节奏、`mode=0` 和 `mode=1` 的坐标系、选最近候选、守卫容差、生成器 |

**轨迹库使用的配置（`configs/examples/wf_cmd_gpu4096.json`）**：`trajectory.library` 由 `prepare_wf_run.py` 自动改写到 `<参考目录>/trajectories`。`TRAJ_PER_LEVEL` 环境变量控制每个 level 生成多少条轨迹（默认 256）。

**注意**：修正执行器参数后 physics hash 变了。以前用旧代码训练出的 checkpoint 不能再用新模型续训或评估。

---

## 1. 指令格式（11 维）

`[x, y, z, qw, qx, qy, qz, vx, vy, wz, mode]`

- `mode=1`：边走边操作。底盘跟踪 `(vx, 0, wz)`，TCP 目标随机身一起移动（`goal_follow_frame="body"`）。
- `mode=0`：静止到达。TCP 目标固定在世界系，不检查底盘速度。

---

## 2. 服务器环境

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
# 先按服务器驱动版本安装对应的 CUDA 版 PyTorch，然后：
pip install -e '.[dev,gpu]'
export WFARM_ASSET_DIR=$PWD/wf-arm-tbot     # 或你放资产的路径
```

阶段 0（`bash scripts/wf_cmd_pipeline.sh 0 0`）依次检查：

1. 单元测试和资产检查
2. CUDA 可用
3. `python -m supermimic.wf.simulator_warp --device cuda:0 --num-envs 4096`：先在 GPU 上和 CPU MuJoCo 做一致性对比（不一致会直接退出），再测吞吐 `env_steps_per_s`，同时报告 CUDA Graph 是否生效、接触缓冲区用了多少

本地 CPU 上的实测一致性（Warp 跑在 CPU 设备上）：50 步噪声动作后状态最大误差 4e-6；不同摩擦下行驶的速度、滑移、轮子法向力都一致。**GPU 上没有实测过**，所以阶段 0 的一致性检查不能跳过。

---

## 3. 阶段 1：生成参考轨迹（CPU 多进程）

```bash
python -m supermimic.wf.command_reference --config configs/examples/wf_cmd_gpu4096.json \
    --output outputs/wf_cmd_refs --count 512 --workers 32
```

每条请求的生成方式：

1. 随机采样一组指令（vx、wz、mode）和机械臂目标构型，目标构型要经过碰撞检查。
2. 由离线教师执行。这个教师只在生成参考时使用，部署时不存在：
   - 轮子：根据轮轴几何算出差速映射，再用 PI 闭环跟踪实测的机体速度。
   - 腿：保持站立姿态，叠加俯仰/横滚 PI 稳定器。稳定器的灵敏度在启动时实测辨识，不依赖关节方向约定。
   - 机械臂：先沿无碰撞的关节通道运动，再用阻尼雅可比修正去跟踪随机身移动的目标。
3. 在摩擦 1.0 下跑一遍名义执行，再加关节噪声、摩擦取 0.85–1.15 独立重放一遍。**两遍都必须通过** `screen_reference`：最后 50 步同时满足末端位姿、末端速度、底盘速度误差要求，不失败，轮子保持接触，滑移不超限。最终只保存重放那一遍。

输出：

- `outputs/wf_cmd_refs/library/`：clip 文件和 `manifest.json`，训练读的就是这个目录
- `attempts.jsonl`：每次尝试的结果和失败原因
- `summary.json`

本地实测：通过率约 92%，40 条参考约 8 秒（11 个进程）。

---

## 3b. 阶段 1：末端目标轨迹库（Ta/Tb 调度）

配置里设置了 `trajectory.library` 时，阶段 1 会在参考目录下额外生成 `trajectories/`：

```bash
python -m supermimic.wf.goal_trajectory --config configs/examples/wf_cmd_gpu4096.json \
    --output outputs/wf_cmd_refs/trajectories --per-level 256 --workers 32
```

**生成方式**：每个 level 在关节空间采样一串经过碰撞检查的航点，用三次样条连成平滑轨迹，并按速度上限做时间拉伸。轨迹以控制周期采样，每个采样点都做碰撞检查，再用 FK 转成跟随坐标系（`goal_follow_frame`）下的末端位姿序列。每条轨迹默认 16 秒（800 个采样点）。

| level | 关节幅度系数 | 关节速度上限 | 末端线速度上限 | 末端角速度上限 | 实测末端活动范围（中位数） |
|---|---|---|---|---|---|
| 0 | 0.15 | 0.25 rad/s | 0.04 m/s | 0.20 rad/s | 0.07 m |
| 1 | 0.35 | 0.40 | 0.08 | 0.35 | 0.17 m |
| 2 | 0.60 | 0.60 | 0.12 | 0.50 | 0.24 m |
| 3 | 1.00 | 0.80 | 0.16 | 0.70 | 0.49 m |

**训练中的调度**（`trajectory` 配置组）：

- 每回合以 `fraction=0.8` 的概率启用轨迹，其余回合沿用单点静态目标。
- 每 `segment_steps=250` 步（Ta = 5 s）从库中取一段新窗口。轨迹从"当前课程 level"中抽取，有 30% 的概率改从更低的 level 中抽取。抽 16 个候选，选起点离当前末端最近的一条，避免目标跳变。
- 每 `update_steps=5` 步（Tb = 0.1 s）把窗口当前时刻的位姿写进 command 的前 7 维。`mode=1` 时相对当前机身坐标系，`mode=0` 时相对这一段开始时的坐标系（相当于世界系下运动的目标）。
- Actor 的输入格式不变，仍是 11 维；部署时不需要轨迹库。

**评估指标**：`tracking_success_rate` 是去掉前 `tracking_grace_steps=50` 步之后，满足以下条件的帧占整个评估时长的比例：末端位置 ≤ 3 cm、姿态 ≤ 0.15 rad，`mode=1` 时底盘速度误差也在容差内。失败之后没执行的帧算作未命中。另外还输出 `tracking_rms_position_error_m` 和 `tracking_rms_orientation_error_rad`。课程升级（阈值 0.6）和守卫都用这个指标。`evaluate` 默认使用 checkpoint 配置里的轨迹库；加 `--static-goals` 可以回到单点目标评估。

**限制**：策略只能看到当前目标点，看不到目标速度，想象展开时目标也保持静止，所以跟踪运动目标会有滞后。因此轨迹速度设得较低。之后可以考虑把目标速度加进观测做前馈。

---

## 3c. 阶段 2 开头：DAgger 纠错标注（BC 预训练内部）

纯 BC 只学到了教师自己走过的状态。在教师的状态上动作误差约 0.001，但在闭环中误差会逐步累积，底盘慢慢开走，末端偏离目标。DAgger 让策略自己去跑，再由同一个离线教师（`command_reference.CommandTeacher`）在策略实际走到的状态上给出"此时应该怎么做"的标签，这些标签加入 BC。

配置（`demonstrations` 组，`wf_cmd_gpu4096.json`）：

| 项 | 值 | 说明 |
|---|---|---|
| `pretrain_updates` | 6000 | BC 总更新次数（原来是 4000） |
| `dagger_rounds` / `dagger_interval` | 4 / 1000 | 在第 1000、2000、3000、4000 次 BC 更新时各采集一轮；最后一轮之后还有 2000 次更新在全部标签上训练 |
| `dagger_episodes` | 64 | 每轮从 64 条训练 clip 的起点出发，跑完整一个回合（600 步），CPU MuJoCo，摩擦随机取 0.85–1.15 |
| `dagger_beta` / `dagger_beta_decay` | 0.5 / 0.5 | 第 r 轮每一步以 β=0.5·0.5^r 的概率执行教师动作，否则执行策略动作（0.5、0.25、0.125、0.0625） |
| `query_fraction` | 0.5 | 每个 BC batch 中 DAgger 标签所占的比例，其余来自 clip |

实现要点（`src/supermimic/wf/dagger.py`）：

- 每个仿真世界配一个教师实例。教师从 clip 的起始状态和目标出发，PI 积分器、关节通道的进度都按实际走过的状态更新，所以它在偏离后的状态上给出的就是纠正动作。
- 标签和 clip 的 BC 目标含义相同（可执行的 [-1, 1] 指令），使用同样的归一化尺度。
- 只使用训练 clip。验证 clip 仍然留作 BC 物理选择用，不会混进训练。
- 标签只进入 BC，不进入世界模型的回放数据。教师只在离线阶段使用，部署时不存在。
- 每一轮会往 `reference_pretrain.jsonl` 写一行 `dagger_round` 记录，包括标签数、策略执行动作的比例、失败率，以及策略与教师在腿、轮、臂三组上的平均动作差。标签保存在 `dagger_queries.npz`，checkpoint 里记录它的 sha256；续训时会重新加载并校验。
- 每轮耗时：64 个回合在 CPU 上串行仿真，大约 1–3 分钟。

**必须同时设 `policy.ignore_previous_action=true`**。教师的输出有速率限制：每一步是 `上一步指令 + clip(期望值 − 上一步指令, ±限幅)`，限幅为腿 0.004、轮 0.01、臂 0.008。所以"照抄上一步指令"几乎就是 BC 的最优解，离线损失可以降到 3e-4。但在闭环里，每一步的小误差都会像随机游走一样累积，最后机身倾倒。DAgger 的标签形式相同，单独使用解决不了这个问题。打开这个开关后，Actor 看不到上一步指令（Critic 仍然能看到，用于计算动作变化率奖励），只能根据状态和目标输出动作。

本地验证：60 条 clip，48 条训练、12 条验证；4000 次 BC 更新，3 轮 DAgger，每轮 32 个回合；按物理回放挑选版本。

| 方案 | 验证集失败率 | 验证集末端误差 p50 / p95 | 训练集失败率 | 训练集末端误差 p50 | 机身位移 |
|---|---|---|---|---|---|
| BC | 0.92 | 0.84 / 1.23 m | 0.73 | 0.75 m | 0.6 m |
| BC + DAgger | 1.00 | 0.66 / 0.78 m | 1.00 | 0.46 m | 0.7 m |
| BC + 不看上一步指令 | 0 | 0.08 / 0.26 m | 0 | 0.07 m | **0.035 m**（几乎不动） |
| **BC + DAgger + 不看上一步指令（当前配置）** | **0** | **0.077 / 0.187 m** | **0** | **0.072 m** | 1.06 m（按指令行驶） |

"BC + 不看上一步指令"之所以末端误差小，是因为机身几乎不动，忽略了底盘速度指令；加上 DAgger 之后，机身按指令行驶，末端误差同样很小。这次验证的数据量很小，末端误差仍远大于 3 cm 的成功阈值，所以这只是一个更好的在线训练起点，不代表最终性能。

另外，没有开启物理挑选时（`physical_selection_interval=0`），会按离线验证损失挑版本，经常挑中第一轮 DAgger 之前的那一版，等于 DAgger 没起作用。正式配置使用 `physical_selection_interval=500`，不受这个问题影响。

---

## 4. 阶段 2：训练

### 针对 2026-11-14 运行结果的修正

| 现象（`outputs/wf_cmd_gpu*`） | 原因 | 修改 |
|---|---|---|
| `…203738`：第 1000/2000/3000 步的验证结果完全相同；守卫 5 次中 5 次拒绝 | 守卫只比较回报。失败率从 25% 降到 0 的候选，因为回报低了 0.2 也被拒绝；每次拒绝学习率减半，降到 0.05 倍 | `prefer_safety`（失败率降低就接受），`prefer_success` 加 `success_tolerance=0.03`，`score_tolerance=0.1`，`rejection_lr_factor=0.7`，`min_lr_scale=0.25`；每 250 次迭代检查一次，每次 32 个回合 |
| BC 后的基线失败率为 94% / 25% / 75%，三次运行差别很大 | 按离线动作损失挑选的 BC 结果在闭环中不稳定 | `demonstrations.physical_selection_interval=500`：在留出 clip 上做真实物理回放，按回放结果挑选 |
| level 1 姿态误差 0.2–0.26 rad，阈值 0.15 | 姿态奖励在 0.2 rad 附近几乎没有梯度 | `orientation_scales=[0.5, 0.15]`，`position_scales` 增加 0.02 m 一档 |
| `level2` 视频中 vx 误差 0.12、wz 误差 0.19 | 视频用的指令 `[0.3, 0, 0.3]` 中，wz 是训练上限 0.06 的 5 倍；vx=0.3 在第 3000 次迭代时还没进入速度课程（第 6000 次之前上限是 0.1） | 属于超出训练分布，不是 bug |

### 针对 2026-11-15 运行（`…010048_292613`，4096 环境）的修正

这次运行在第 6060 次迭代时进程崩溃（`double free or corruption`），当时的 level 是 1。耗时分布如下：

| 阶段 | 迭代 | 耗时 | 现象 |
|---|---|---|---|
| Actor 未启动 | 0–560 | 约 4 分钟 | 0.43 s/迭代 |
| 候选全部被拒 | 560–3000 | 约 64 分钟 | 守卫连续 10 次拒绝，Actor 每次都回滚到 BC 初值，accepted_actor_updates=0 |
| 正常学习 | 3000–5000 | 约 80 分钟 | level 0 跟踪成功率从 0.03 升到 0.99 |
| level 1 | 5000–6060 | 约 46 分钟 | 第 6000 次迭代跟踪成功率已达 0.99；按每 1000 次迭代评估一次、连续 2 次达标才升级的规则，要到第 7000 次才能升级，但进程在第 6060 次崩溃 |

| 原因 | 证据 | 修改 |
|---|---|---|
| Actor 更新慢，且随时域线性变慢 | 每次迭代从 0.43 s 涨到 2.5 s（时域 8→32）。一次 FK 约 3400 个小算子，世界模型一步只有约 245 个；GPU 上的耗时主要是 kernel 启动开销 | `compile_fk=true`（torch.compile 编译 FK，编译失败会自动回退到 eager）。本地 CPU 上 H=8 的展开加反传从 0.51 s 降到 0.25 s，数值差 1.5e-7；GPU 上收益应更大 |
| 被拒的候选仍在采集 | 候选崩坏后要等到下一次守卫（250 次迭代后）才回滚；在线失败率 1–4.7%/环境步，回放池被失败数据填满，世界模型误差维持在 0.005–0.012 | 在线熔断：`guard.online_failure_rate=0.01`、`online_failure_factor=3`、`online_failure_window=20`。20 次迭代窗口内的失败率超过 max(1%, 3×已接受策略的失败率) 时立即回滚，并按拒绝处理（学习率 ×0.7），记录写入 `guard_XXXXXXX_online.json` |
| 探索噪声过大 | 所有关节共用 `exploration_std=0.12`，对机械臂相当于每步约 0.3 rad 的白噪声。在线数据中 TCP 平均速度 1.3–2.3 m/s；还没开始更新 Actor 时，BC 策略在噪声下就有 0.15–0.75%/环境步的失败率，而无噪声的守卫评估失败率为 0 | `sim.exploration_group_std=[0.12, 0.06, 0.04]`（腿、轮、臂），臂约 0.08 rad，轮约 1.5 rad/s |
| 被拒的更新也会推高时域 | `horizon_progress=attempted`：第 3000 次迭代时一次更新都没被接受，时域却已涨到 23 | `policy.horizon_progress="accepted"` |
| 升级节奏慢 | 评估间隔 1000、要求连续 2 次达标，每个 level 至少 2000 次迭代 | `curriculum.eval_interval=500` |
| BC 初值在闭环中没用 | 教师强制下动作误差约 0.001，闭环中输入逐步漂移，底盘缓慢开走；留出 clip 的 held success 一直为 0，BC 后 TCP 误差 0.3 m，比随机初值（0.09 m）还差 | DAgger 纠错标注 + `policy.ignore_previous_action=true`，见 §3c |
| 原生崩溃 | 第 6060 次迭代 `double free or corruption (out)`，没有 Python 栈信息 | 训练开启 `faulthandler`。流水线在训练崩溃后自动从 `ckpt_latest.pt` 续训，最多 `TRAIN_RETRIES=3` 次（Ctrl+C 不重试）。下次崩溃时 `train.log` 里会有 Python 栈，请提供 |

新增的配置项改变了 `sim`、`policy`、`guard`、`curriculum` 配置组，所以旧 checkpoint 不能直接 `--resume`，需要从阶段 1 重新开始。

`wf_cmd_gpu`（第 3000 次迭代，level 0）本身已经能用：静态目标 94% 成功，底盘速度误差约 0.002 m/s，末端误差约 1 cm。下次分析时请一并提供 `metrics.jsonl`、`train.log` 和 `reference_pretrain.jsonl`，这次的记录里没有这几个文件。

```bash
python -m supermimic.wf.train --config configs/examples/wf_cmd_gpu4096.json
# 中断后续训：
python -m supermimic.wf.train --config configs/examples/wf_cmd_gpu4096.json --resume outputs/wf_cmd_gpu/ckpt_latest.pt
```

训练内部的顺序：

1. 建重置状态库：4 × 4096 个带噪声的复位，在真实物理中 settle 100 步，只保留安全的状态。之后结束的回合直接从库里取一个状态重启，不用每次复位都等 100 步 settle。
2. 在参考数据上预训练世界模型（2000 次）。
3. BC 预训练（6000 次，其中穿插 4 轮 DAgger 纠错标注，见 §3c），每 500 次在留出 clip 上做一次物理回放，挑选最好的一版，保存为 `ckpt_reference.pt`。
4. 在线阶段，每次迭代：4096 个环境各走 1 步（GPU 物理），更新 4 个 batch 的世界模型（每个 batch 4096 个窗口），再做 1 次 Actor BPTT（batch 2048，想象展开的时域从 8 逐步增加到 32）。参考轨迹的作用：BC 项、想象状态跟踪项（在 6000 次更新内衰减）、参考起点、目标库。
5. 每 250 次迭代做一次物理守卫：在 CPU MuJoCo 上跑 32 个任务回合 + 8 条参考验证 clip，策略变差就回滚。两次守卫之间，如果在线采集的失败率突然升高，会立即回滚（在线熔断）。每 500 次迭代做一次课程评估（32 个回合），连续 2 次跟踪成功率 ≥ 0.6 就升一级。

关键配置（`configs/examples/wf_cmd_gpu4096.json`）：

| 项 | 值 | 说明 |
|---|---|---|
| `sim.num_envs` / `backend` | 4096 / `warp` | `warp_device` 为 null 时跟随 `device` |
| `sim.warp_nconmax` / `warp_njmax` | 64 / 320 | 每个环境的接触/约束缓冲。站立时实测约 6.4 个接触/环境 |
| `replay_steps` | 256 | 回放容量 = 4096×256，约 0.7 GB 显存 |
| `checkpoint_replay` | false | checkpoint 不含回放数据，只有网络和优化器状态；续训后回放从当前世界状态重新填充，重新测出世界模型误差之前 Actor 不会更新 |
| `reward.base_command_ranges` | `[0.3, 0, 0.06]` | 见下文"转向限制" |
| `reward.success_base_angular` | 0.03 | 偏航成功阈值，原来是 0.06 |
| `base_speed_curriculum` | 0→0.1、6000→0.2、15000→0.3 m/s | |
| `iterations` | 30000 | 约 1.2 亿环境步。这是起点，不保证收敛 |

要看的输出（`outputs/wf_cmd_gpu/`）：

- `metrics.jsonl`：`curriculum_level`、`accepted_actor_updates`、`wm_validation_error_k`、`reference_validation_error_k`、`sim_*`、`reward_base_velocity`
- `validation_*.json`：课程评估
- `guard_*_{baseline,candidate}.json`：守卫的评估结果和是否接受
- `reference_pretrain.jsonl`：BC 训练/验证损失
- `reset_bank.json`：状态库里有多少可用状态
- `actor_XXXXXXX.pt`：周期性保存的 Actor 快照

---

## 5. 阶段 3/4：评估与导出

评估都在 CPU MuJoCo 上进行，与训练用的 Warp 后端相互独立：

```bash
python -m supermimic.wf.evaluate --checkpoint outputs/wf_cmd_gpu/ckpt_latest.pt --level 2 --episodes 64
python -m supermimic.wf.evaluate --checkpoint ... --reference-library outputs/wf_cmd_refs/library --reference-split validation
python -m supermimic.wf.evaluate --checkpoint ... --goals-json my_goals.json   # 每行 11 维
python -m supermimic.wf.evaluate --checkpoint ... --level 2 --ablation all
python -m supermimic.wf.export   --checkpoint ... --output outputs/wf_cmd_gpu/policy.pt
```

导出格式为 `wf_command_unified_torchscript_v2`，输入 `world_goal` 为 `[N,11]`，元数据写在 `policy.pt.json`。

### 录制策略视频

`supermimic.wf.play` 复用 `evaluate_policy` 的物理评估过程，在每个物理控制步之后
观察场景并渲染，不启动训练或世界模型。默认无窗口，CPU 运行策略与 MuJoCo，EGL 渲染。

```bash
# 在运行 WF 训练的 Python 环境中安装录像依赖：
python -m pip install 'imageio>=2.30,<3' 'imageio-ffmpeg>=0.5,<1'

# 目录自动选择 ckpt_latest.pt；也可以直接指定 actor_0003000.pt 等文件：
python -m supermimic.wf.play --checkpoint outputs/wf_cmd_gpu --episodes 3 --level 0

# 固定底座前进 0.1 m/s，同时追踪随机采样的末端目标：
python -m supermimic.wf.play --checkpoint outputs/wf_cmd_gpu \
  --episodes 1 --command 0.1 0 0 --video outputs/wf_cmd_gpu/videos/forward.mp4
```

默认输出为 `CHECKPOINT_DIR/videos/preview_iterXXXXXXX_levelN.mp4`，旁边保存同名 JSON，
包含 checkpoint 迭代数、每个回合的目标、失败判定、保持成功和最终误差。
每个回合默认录制训练配置中的 600 个控制步（12 秒），结束后停留 1 秒；失败时提前结束，
保留真实失败画面。30 FPS 按物理时间采样，录像速度不依赖机器的渲染速度。

绿色球为目标位置，橙色球为实际 TCP，RGB 箭头表示目标姿态。画面文字显示指令、
末端误差、成功/失败以及仿真时间；世界坐标网格方便观察底座运动。
默认跟随相机距离为 2.8 米，可以用 `--camera-distance` 调整。
常用参数还有 `--steps 600`、`--seed 20000`、`--fps 30`、`--width 1280 --height 720`。
不指定 `--level` 时使用完整 checkpoint 保存的课程等级，Actor 快照则默认等级 0。
`--command VX VY WZ` 的单位为 m/s、m/s、rad/s，需使用支持底座指令的 checkpoint。

脚本默认设置 `MUJOCO_GL=egl`。若机器使用其他已配置的渲染后端，可在启动进程前设置
`MUJOCO_GL=osmesa` 或 `MUJOCO_GL=glfw`；策略推理的 `--device` 与渲染后端相互独立，物理始终在 CPU 上运行。

---

## 6. 已知限制（务必先看）

1. **转向能力很弱，这是执行器参数决定的。** `robot.json` 里 `wheel_kd=0.5`、轮力矩上限 12 Nm。差速转向时四轮能产生的偏航力矩（约 116 Nm）小于 μ=1 时的侧向刮擦阻力（约 160 Nm）。实测：原地转向时轮子被卡住；差速再加大机身会侧翻（tilt 0.85–1.7 rad）；带臂载边走边转，可靠的偏航角速度只有约 0.04–0.06 rad/s。所以 `wz` 范围设为 ±0.06，偏航阈值收紧到 0.03。想要更大的转向能力，需要先修改轮子的执行器参数（例如更大的 `wheel_kd` 或力矩上限），然后重新生成参考、重新训练。这个参数要由你来定。
2. **GPU 路径没有在 GPU 上实测过。** 本地没有 GPU，验证的是 Warp 在 CPU 设备上的结果，以及完整流程（16 个环境）。CUDA 上的流同步和 CUDA Graph 捕获，靠阶段 0 的一致性检查把关。如果 graph 捕获失败，会打印警告并自动改为 eager 执行，结果不变，只是更慢。可以用 `sim.warp_cuda_graph=false` 强制关闭。
3. **吞吐、显存和总训练时长都是估计值。** 先看阶段 0 输出的 `env_steps_per_s`，再看 `metrics.jsonl` 里的 `elapsed_s`，据此调整 `iterations` 和 batch 大小。
4. 原仓库的轮速约定 `[-1,1,-1,1]·v`（用在 `reference.py`、`feedback_reference.py`、`goal_reference.py`）对这台机器人是反的：给正速度时实际在转向或倒退。新的生成器改用由几何推导出的映射，旧的生成器没有改。
5. `.gitignore` 里私有资产的排除规则仍是注释状态。用 git 推送到服务器之前，确认你确实打算让 `wf-arm-tbot/` 进入仓库。

---

## 7. 本地快速检查（可选）

```bash
python -m supermimic.wf.command_reference --config configs/examples/wf_cmd_smoke.json \
    --output outputs/wf_cmd_refs_smoke --count 10 --workers 8
python -m supermimic.wf.train --config configs/examples/wf_cmd_smoke.json --device cpu   # Warp 跑在 CPU 上，约 3 分钟
WFARM_ASSET_DIR=$PWD/wf-arm-tbot python -m pytest -q tests/
```
