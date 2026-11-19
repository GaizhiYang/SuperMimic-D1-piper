# SuperMimic-D1-Piper

**简体中文** · [English](README.md) · [许可证](LICENSE) · [来源与致谢](NOTICE.md)

利用可微世界模型学习轮腿式移动操作。
本项目将 SuperMimic 的学习方法应用于 D1 平台：机器人包含四条腿、四个轮子和一条六轴
Piper 机械臂。统一策略控制全部 22 个执行器，跟踪末端位姿，并在启用时跟踪底盘速度指令。

主要流程涵盖参考生成、连续 TCP 轨迹指令、CPU 或 GPU 仿真、世界模型训练、策略优化、
物理评估和视频录制。仓库还保留固定底盘机械臂实验和继承的 G1 实现，作为独立研究分支。
本项目是独立的研究适配，不声称复现原论文的基准结果或完成实机验证。

## 项目包含什么

| 组件 | 实现 |
|---|---|
| 全身控制 | 一个 Actor 输出 12 路腿部位置指令、4 路轮速指令和 6 路机械臂位置指令 |
| 可微动力学 | 学习仿真状态转移，再通过短时域想象展开，以 BPTT 优化 Actor |
| 参考生成 | 电机驱动的 MuJoCo 执行、物理筛选及可选的模仿辅助 |
| 移动 TCP 指令 | 对关节样条做碰撞检查，再转换为末端位置与姿态轨迹 |
| 仿真后端 | 原生 CPU MuJoCo，以及用于 GPU 训练的 MuJoCo Warp 批量采集 |
| 评估 | 位姿到达、轨迹跟踪、物理失败、摩擦变化和执行器消融 |
| 录像 | 完整连续轨迹、机器人原始可视网格、明亮的白灰地面，以及目标和实际 TCP 轨迹 |
| 导出 | 包含 Actor 和观测变换的 TorchScript 推理封装 |

WF 示例配置的控制周期为 20 ms（50 Hz），物理步长为 5 ms。位置指令使用配置中的
执行器缩放系数，并非关节力矩。四元数输入采用 **wxyz** 顺序。

## 学习流程

1. 生成参考运动，并在 MuJoCo 中执行对应的电机指令。通过筛选的状态/动作转移与
   TCP 目标路径分别保存。
2. 在物理仿真中运行当前策略，采集更多状态转移，再用回放窗口拟合可微动力学模型。
3. 更新策略时冻结世界模型参数，通过想象展开中的任务奖励反向传播梯度。
   可选的行为克隆和参考跟踪损失为 Actor 提供辅助。
4. 在 MuJoCo 中评估 Actor。课程与守卫检查通过真实物理执行判断进展及候选更新的效果。

教师能执行某条参考运动，不代表学到的 Actor 已经能完成该任务。TCP 轨迹库提供目标指令
及名义上的运动学可行性，其中的路径并非策略成功执行的记录。Actor 推理不需要学到的
世界模型或离线教师。

## 安装

使用 Linux 和 Python **3.11 或 3.12**。录像和 GPU 流程已在 MuJoCo **3.16**、
PyTorch **2.7**、Python **3.11** 环境中检查。建议使用独立环境：

```bash
git clone https://github.com/GaizhiYang/SuperMimic-D1-piper.git
cd SuperMimic-D1-piper
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,media]' 'mujoco~=3.16.0'
```

使用 NVIDIA GPU 训练时，先安装与驱动匹配、支持 CUDA 的 PyTorch，再安装 GPU 后端：

```bash
python -m pip install -e '.[dev,media,gpu]'
```

| 可选依赖组 | 用途 |
|---|---|
| `dev` | pytest |
| `media` | MP4 编码、图像标注和绘图 |
| `gpu` | MuJoCo 3.16、MuJoCo Warp 和 Warp |
| `teacher` | 其他全身控制实验使用的可选 OSQP/quadprog 教师 |
| `g1` | 继承的独立 G1 实现；D1/Piper 训练无需安装 |

主要的速度指令参考生成器和 TCP 轨迹生成器不需要 `teacher` 依赖组。
即使训练采集使用 Warp，评估和录像仍使用原生 CPU MuJoCo。无窗口录像默认采用 EGL，
运行机器需要可用的 OpenGL/EGL 环境。

## 机器人模型

当前源码包含 `wf-arm-tbot/`，其中提供 URDF、其引用的碰撞与可视网格、机器人配置及
TCP 定义。资产解析器会自动找到该目录。也可以显式指定它，或使用外部副本：

```bash
export WFARM_ASSET_DIR="$PWD/wf-arm-tbot"
python tools/check_assets.py
```

预期目录结构：

```text
wf-arm-tbot/
  urdf/wf-arm-tbot.urdf
  meshes/
  config/robot.json
  config/robot_calibration.json
  config/operating_limits.json     # 固定底盘机械臂适配器也会使用
```

显式传入的 `--asset-dir` 或配置中的 `asset_dir` 优先于环境变量。评估时会核对模型与
checkpoint 的物理指纹，因此应使用与训练相同的资产和执行器配置。
资产是否可用与是否允许再分发是两个问题，详见[许可与致谢](#许可与致谢)。

## 快速开始

激活环境后，在仓库根目录运行以下命令。

### 检查代码和 CPU 训练流程

```bash
# 无需加载机器人模型的核心测试。
python -m pytest -q -m 'not private_assets' tests/test_release_core.py

# 小规模集成运行：4 个环境，64 次迭代。
python -m supermimic.wf.train --smoke --device cpu --output outputs/wf_smoke
```

冒烟运行用于检查训练流程，不验证策略是否收敛。进行更长的 CPU 实验时，可使用或复制
`configs/examples/wf_train.json`，并设置新的输出目录：

```bash
python -m supermimic.wf.train --config configs/examples/wf_train.json
```

### 训练底盘速度指令与移动 TCP 目标

GPU 流水线包含以下阶段：

| 阶段 | 工作 |
|---|---|
| `0` | 检查依赖、资产、CUDA，以及 Warp/CPU 一致性 |
| `1` | 生成经过物理筛选的参考数据和 TCP 轨迹库 |
| `2` | 参考预训练、在线采集、动力学拟合，以及 Actor/Critic 更新 |
| `3` | 在 MuJoCo 中评估各课程等级、参考验证集、摩擦变化和消融设置 |
| `4` | 导出 TorchScript |

小规模 GPU 集成运行：

```bash
CONFIG=configs/examples/wf_cmd_smoke.json \
REF_COUNT=10 TRAJ_PER_LEVEL=4 WORKERS=4 \
bash scripts/wf_cmd_pipeline.sh 0 4
```

使用完整预设：

```bash
CUDA_VISIBLE_DEVICES=0 WORKERS=8 bash scripts/wf_cmd_pipeline.sh 0 4
```

默认配置文件名为 `wf_cmd_gpu4096.json`，但当前实际设置为 **8192 个环境、30000 次迭代**。
开始完整训练前，请根据硬件调整环境数、批量大小和参考生成进程数。
小规模速度指令预设同样需要 CUDA GPU；CPU 检查请使用 `train --smoke`。

运行范围包含阶段 `1` 时，会在 `outputs/` 下创建时间戳一致的参考目录和训练目录。
脚本会打印路径，并在训练目录保存 `pipeline_config.json`。继续同一次运行时使用这份配置：

```bash
# 将此值替换为流水线打印的训练目录。
RUN_DIR=outputs/wf_cmd_gpu_YYYYMMDD_HHMMSS_SUFFIX
CONFIG="$RUN_DIR/pipeline_config.json" bash scripts/wf_cmd_pipeline.sh 2 3
```

### 评估 checkpoint

评估和导出模块需要传入 checkpoint **文件**：

```bash
python -m supermimic.wf.evaluate \
  --checkpoint "$RUN_DIR/ckpt_latest.pt" --level 3 --episodes 64 \
  --output "$RUN_DIR/eval_level3.json"

python -m supermimic.wf.evaluate \
  --checkpoint "$RUN_DIR/ckpt_latest.pt" --static-goals --level 3 \
  --output "$RUN_DIR/eval_static.json"
```

若 checkpoint 配置了 TCP 轨迹库，评估默认使用该库。`--static-goals` 则选择单点到达目标。
其他选项包括 `--ablation all`、`--friction`，以及配合 `--reference-split validation`
使用的 `--reference-library`。移动训练目录时，请确保引用的轨迹库仍可访问。

应同时报告轨迹跟踪成功率、位置/姿态 RMS 误差和物理失败率。跟踪统计将失败后未执行的
步数计为未命中。较低的动力学模型误差、教师成功或良好的录像观感，都不能单独证明策略
表现。评估泛化能力应使用独立的留出请求。

### 录制连续轨迹

对于带速度指令、且已生成 TCP 轨迹库的 checkpoint：

```bash
# 录制静止、前进和缓慢转向三个场景。
bash scripts/wf_record_trajectories.sh "$RUN_DIR"

# 设置底盘前向/侧向速度及偏航角速度：m/s、m/s、rad/s。
bash scripts/wf_record_trajectories.sh "$RUN_DIR" \
  --scenario turn --command 0.2 0 0.03

# 放慢 TCP 路径，并选取三条不同的大范围路径。
bash scripts/wf_record_trajectories.sh "$RUN_DIR" \
  --scenario stationary --speed 0.4 --clips 3
```

录像程序按空间范围选择完整路径，先平滑接近起点，再逐控制步更新目标。
单段录像中不会随机切换轨迹窗口。默认使用 level 3、0.75 倍路径速度、1280×720 分辨率和
30 FPS。视频按仿真时间播放，不受渲染速度影响。

`--command VX VY WZ` 覆盖所选场景的底盘指令；`--speed` 单独控制 TCP 轨迹播放速度。
默认的 `--frame follow` 按配置的底座跟随坐标系搬移目标；
`--scenario stationary --frame world` 将路径固定在初始世界位姿，要求底盘速度指令为零。

默认显示原始 URDF 可视网格及材质，地面为明亮的白色/浅灰色网格。
使用 `--appearance collision` 可查看碰撞几何。渲染使用独立模型，不改变策略运行的物理模型。
青色表示完整目标路径，绿色表示已下发的目标轨迹，橙色表示实际 TCP 轨迹。

每段录像在 `"$RUN_DIR/videos/trajectories_<timestamp>/"` 下保存 MP4、JSON 指标、
控制频率的 NPZ 记录和 PNG 预览。这些预览复用训练轨迹库，不衡量未见轨迹上的表现。
单点目标录像请查看 `python -m supermimic.wf.play --help`。
更多选项见[录像指南](docs/WF_COMMAND_TRAINING_CN.md#录制连续末端轨迹视频)。

### 导出推理模型

```bash
python -m supermimic.wf.export \
  --checkpoint "$RUN_DIR/ckpt_latest.pt" --output "$RUN_DIR/policy_inference.pt"
```

对于带速度指令的 WF 策略，封装接收规范的 57 维状态、22 维上一步实际执行动作，以及
11 维世界系目标：TCP 位置、wxyz 四元数、底盘 vx/vy/wz 和指令模式标志。
相邻的导出 JSON 描述实际接口、关节顺序、执行器缩放系数和物理指纹。
调用方负责维护实际执行动作的历史，并完成目标坐标系搬移。导出模型不包含仿真器或硬件 I/O。
导出程序拒绝覆盖已有的模型或元数据文件。完整流水线已生成 `policy.pt`；续训后再次
导出时，请选择新的文件名。

## 其他实验与源码结构

```text
src/supermimic/wf/        全身模型、仿真、训练、评估和录像
src/supermimic/mimicagent/ 任务/计划/程序形式的参考生成前端
src/supermimic/world_model/, imagination/, kinematics/
                         共用的学习及可微运动学工具
frameworks/fixed_base_arm/ 六轴基线和结构化控制器变体
configs/examples/        CPU 和 GPU 实验预设
scripts/                 流水线和录像封装
tests/                   核心、物理、轨迹和渲染检查
wf-arm-tbot/             机器人描述、网格和模型配置
outputs/                 生成的运行结果；Git 忽略此目录
docs/                    详细流程与算法说明
```

固定底盘机械臂基线有独立入口：

```bash
python tools/check_assets.py --fixed-arm
PYTHONPATH=src:frameworks/fixed_base_arm python -m arm_smc.train \
  --config configs/examples/fixed_arm_smoke.json
```

固定机械臂的结构化变体结合学习组件、规划和解析雅可比反馈，与统一全身 Actor 分开实现。
详见[固定机械臂概览](frameworks/fixed_base_arm/README.md)、
[算法说明](docs/PROJECT_ANALYSIS_CN.md)和[脚本索引](scripts/README.md)。
继承的 G1 代码需要独立的环境和数据。

显式运行依赖机器人模型的 WF 测试：

```bash
WFARM_ASSET_DIR="$PWD/wf-arm-tbot" python -m pytest -q tests
```

## 许可与致谢

项目代码采用 [MIT 许可证](LICENSE)，并保留上游版权声明。
本项目在 Yuxi Liu / CuboiLeo 的独立实现
[SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced)
基础上扩展，借鉴 SuperMimic 的学习方法和 MimicAgent 的参考生成方法。
作者署名和研究参考文献见 [NOTICE](NOTICE.md)。

代码许可证不额外授予第三方机器人资源的使用权。随仓库提供的 D1 描述在
[package.xml](wf-arm-tbot/d1_description/package.xml) 中仍有未填写的许可证声明；
请向来源方确认模型和网格的再分发条款。请保留各资源的独立声明，包括 articulation
适配器中的 Apache-2.0 声明。

反馈问题请使用 [Issues](https://github.com/GaizhiYang/SuperMimic-D1-piper/issues)，
并附上代码版本、软件版本、配置及简短复现步骤。生成的参考、日志、checkpoint 和视频
应放在 `outputs/` 下，该目录不纳入 Git 跟踪。
