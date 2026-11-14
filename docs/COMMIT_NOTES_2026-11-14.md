# 提交说明：策略录像与按时间隔离的训练流水线

日期：2026-11-14（Asia/Shanghai）

本次提交为 WF 轮腿臂策略增加 MP4 录像入口，并让完整训练流水线自动创建带时间后缀的新输出目录，解决重复运行时参考库目录冲突的问题。

## 1. 训练与参考库输出目录自动隔离

涉及文件：`scripts/wf_cmd_pipeline.sh`、`scripts/prepare_wf_run.py`。

- 包含阶段 1 的运行会给训练目录和参考库目录追加同一个上海时间后缀，格式为 `YYYYMMDD_HHMMSS_ffffff`。
- 使用独占目录创建处理重名；即使时间戳相同或参考目录已存在，也会通过追加编号分配新目录。
- 在新训练目录保存 `pipeline_config.json`，同步更新 `output` 和 `demonstrations.directory`，供后续训练、评估和导出使用。
- 原始配置文件保持不变。新运行清空配置中的 `resume`，保留显式设置的 `warm_start`。
- 从阶段 2 开始运行时，复用 `CONFIG` 指定的配置与已有目录，存在 `ckpt_latest.pt` 时继续训练。
- 增加阶段范围检查，并在启动时打印配置、训练输出和参考库路径。

新训练的命令保持不变：

```bash
bash scripts/wf_cmd_pipeline.sh 1 4
```

示例输出目录：

```text
outputs/wf_cmd_gpu_20261114_203000_123456/
outputs/wf_cmd_refs_20261114_203000_123456/library/
```

继续该次运行：

```bash
CONFIG=outputs/wf_cmd_gpu_20261114_203000_123456/pipeline_config.json \
  bash scripts/wf_cmd_pipeline.sh 2 4
```

## 2. 新增策略录像入口

涉及文件：`src/supermimic/wf/play.py`、`src/supermimic/wf/evaluate.py`。

- 支持传入训练目录或单个 checkpoint，录制真实 MuJoCo 物理执行过程。
- 复用现有 `evaluate_policy`，增加只读的逐步场景回调，在失败环境重置前捕获真实终止状态。
- 默认 CPU 策略推理与物理、EGL 无窗口渲染，输出 H.264 MP4 和同名评估 JSON。
- 支持课程等级、回合数、随机种子、控制步数、帧率、分辨率、相机距离及底座速度指令。
- 画面显示目标与 TCP 标记、目标姿态、世界网格、误差、指令和成功/失败状态。
- 按物理时间采样帧，回合结束停留一秒，并保留两次计划帧之间发生的失败画面。

使用示例：

```bash
python -m pip install 'imageio>=2.30,<3' 'imageio-ffmpeg>=0.5,<1'
python -m supermimic.wf.play --checkpoint outputs/wf_cmd_gpu --episodes 3 --level 0
python -m supermimic.wf.play --checkpoint outputs/wf_cmd_gpu \
  --episodes 1 --command 0.1 0 0 --video outputs/wf_cmd_gpu/videos/forward.mp4
```

## 3. 文档与回归测试

- 更新 `README_CN.md`、`scripts/README.md` 和 `docs/WF_COMMAND_TRAINING_CN.md`，补充录像、时间戳目录与指定运行续训的用法。
- 新增 `tests/test_wf_pipeline.py`，覆盖目录碰撞、原配置保留、重复运行以及训练/评估/导出路径一致性；集成测试使用替身任务，不启动实际训练。
- 新增 `tests/test_wf_play.py`，覆盖帧率调度、录像不改变评估过程、真实终止状态捕获和移动指令下的目标跟随及程序上下文。

## 4. 验证

在 `supermimic-wfarm` Python 环境中执行：

```bash
WFARM_ASSET_DIR="$PWD/wf-arm-tbot" PYTHONPATH=src python -m pytest -q tests
bash -n scripts/wf_cmd_pipeline.sh
git diff --check
```

结果：29 项测试全部通过，Shell 语法与差异检查通过。

另已使用第 3000 步 checkpoint 录制并验证 1280×720、30 FPS 的 MP4：三个目标追踪回合共 39 秒，前进指令回合 13 秒。

1.训练效果很差，具体看/home/hehui/SuperMimic-Agent-WFArm/outputs/wf_cmd_gpu这个记录，你帮我分析并优化代码。
2.另外，我觉得对于机械臂末端目标位姿指令一次只采样一个位姿，不够优雅，而且间隔太长。我希望采样一段末端位姿轨迹，实现上是先生成各个课程level下的末端目标轨迹库，每条轨迹实质就是一串末端位姿序列，训练时候就每间隔Ta随机在轨迹库中采样得到一段轨迹，然后每隔一小段时间Tb把这条轨迹的目标末端位姿更新到机器人观测中的command项。你觉得这个ok吗？ok的话帮我实现。