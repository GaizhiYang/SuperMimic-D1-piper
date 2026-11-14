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

## 4. 阶段 2：训练

```bash
python -m supermimic.wf.train --config configs/examples/wf_cmd_gpu4096.json
# 中断后续训：
python -m supermimic.wf.train --config configs/examples/wf_cmd_gpu4096.json --resume outputs/wf_cmd_gpu/ckpt_latest.pt
```

训练内部的顺序：

1. 建重置状态库：4 × 4096 个带噪声的复位，在真实物理中 settle 100 步，只保留安全的状态。之后结束的回合直接从库里取一个状态重启，不用每次复位都等 100 步 settle。
2. 在参考数据上预训练世界模型（2000 次）。
3. BC 预训练（4000 次），按留出 clip 上的损失挑最好的一版，保存为 `ckpt_reference.pt`。
4. 在线阶段，每次迭代：4096 个环境各走 1 步（GPU 物理），更新 4 个 batch 的世界模型（每个 batch 4096 个窗口），再做 1 次 Actor BPTT（batch 2048，想象展开的时域从 8 逐步增加到 32）。参考轨迹的作用：BC 项、想象状态跟踪项（在 6000 次更新内衰减）、参考起点、目标库。
5. 每 500 次迭代做一次物理守卫：在 CPU MuJoCo 上跑 16 个任务回合 + 8 条参考验证 clip，策略变差就回滚。每 1000 次迭代做一次课程评估（32 个回合），据此决定是否把目标半径升级。

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
