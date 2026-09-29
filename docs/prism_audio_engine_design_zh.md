# Prism Resonance 次世代音频引擎设计方案

> 面向 Prism（Bevy fork）的数据导向、RT 安全、图编译式高性能音频引擎设计。
> 借鉴 UE5（MetaSounds / Submix / Quartz / Audio Modulation / HDR-Attenuation）、Unity（AudioMixer / DSP Graph / Spatializer & Ambisonic Decoder SDK / Snapshots）、Godot（AudioServer Bus/Effect/Stream / 频谱分析 / 麦克风捕获 / 程序化 Generator）、Wwise（Event/Container/State/Switch/RTPC / Interactive Music / HDR / Occlusion·Obstruction / Aux Sends / Profiler / SoundBank）、FMOD Studio（Event/Parameter/Snapshot / Bank 流式 / Transceiver）、Steam Audio（遮挡·衍射·透射·反射·烘焙·HRTF·Ambisonics）、Web Audio API（AudioNode 图）、微软 Windows Sonic / 索尼 Tempest 3D / 杜比 Atmos / Meta XR Audio（平台空间后端），取长补短。
> **空间与内容一等公民**：几何驱动空间化与 Event 驱动内容模型同为一等公民；程序化合成（Patch）、调制（Modulation）与母带合规（LUFS/True-Peak/HDR）三者贯通，可达顶级次世代 AAA 质量。
> 本文档为设计规格与落地实现的权威规范；采用纯经典 DSP 路线，不含任何 AI/ML 内容，不含任何 UE/Unity/Godot/Wwise/FMOD 源码或衍生代码。

- 版本: v0.4（基础层已落地编码：`pkg/prism_audio_core`；本版新增：程序化内容图 Patch、调制系统、多普勒/锥形/Spread/Focus/多位置、遮挡与障碍区分、Aux 发送与环境、HDR 音频、Bank/流式/内存、输入捕获、平台空间后端、虚拟语音行为、Profiler 与可视化调试；扩展了参考映射与路线图）
- 范围: 一步到位（统一渲染图 / 采样精确调度 / 程序化 Patch / 调制 / 几何声学 / Event+RTPC 编排 / 交互音乐 / LUFS+HDR 母带 / 平台空间输出）
- 适用引擎: Prism / Bevy ECS 生态
- 关键依赖: bevy_ecs（并行 ECS）、bevy_math（glam SIMD + `ops` 确定性标量数学）、bevy_tasks（资产解码/烘焙任务）、bevy_asset（音频资产与 Bank）、bevy_transform（听者/声源位姿）、bevy_a11y（无障碍）、cpal/AudioWorklet（设备后端，前端 crate）
- 复用的现有设施: bevy_asset 加载与热重载、bevy_tasks 异步解码/烘焙、bevy_math::ops（跨平台确定性 sin/cos/exp/log）、bevy_transform 空间层级、bevy_diagnostic 诊断面板、prism_physics 射线/几何查询（遮挡与反射复用）、prism_material_pipeline 表面属性协同
- 关联文档: `prism_physics_design_zh.md`（几何/射线查询原语，供遮挡与反射复用）、`prism_material_pipeline_design_zh.md`（声学材质与视觉材质的表面属性协同）

---

## 目录
1. 设计哲学与目标
2. 业界参考与采纳映射
3. 分层架构
4. 概念模型：Graph / Node / Bus / Voice / Patch
5. 统一实时渲染图
6. 数据布局与缓冲池
7. 参数系统与采样精确自动化
8. Transport 与采样精确调度器（多时钟）
9. 节点库（DSP 原语矩阵）
10. 声源与合成
11. 程序化内容图（Patch / MetaSounds 式可编译子图）
12. 音频调制系统（Modulation：控制总线 / LFO / 包络 / 曲线）
13. 母带与动态处理（LUFS / True-Peak / Sidechain / HDR）
14. 空间音频：几何声学传播（遮挡 / 障碍 / 透射 / 衍射 / 反射）
15. 距离与方向塑形（衰减曲线 / 锥形 / Spread / Focus / 多普勒 / 多位置）
16. HRTF / Ambisonics / 对象音频 / 平台空间后端
17. 环境与辅助发送（Aux Sends / Reverb Zones / Rooms & Portals）
18. 内容模型：Event / Container / State / Switch / RTPC
19. 交互音乐系统
20. 资产、Bank 与流式媒体
21. 无锁 ECS 集成层与线程/内存模型
22. 设备后端、离线渲染与输入捕获
23. 无障碍（Accessibility）
24. 确定性与网络
25. 语音管理与虚拟化
26. 剖析、遥测与可视化调试
27. 性能预算与验收
28. 质量与可信度基础设施
29. Crate 拆分与落地形态
30. 路线图
31. 关键扩展点清单
32. 开放问题
33. 术语表

---

## 1. 设计哲学与目标

对标商用旗舰音频中间件与引擎，确立八条铁律：

- **单一统一渲染图**：全引擎所有声源/效果/总线/空间化器都是同一张有向无环图（DAG）中的 `AudioNode`，而不是"每个声源一个独立 Sink"。融合 Web Audio 的 `AudioNode` 图、UE Submix 树、Godot Bus 链三者优点：一次离线编译成确定性、零分配的处理计划。
- **实时铁律（RT-safety）**：音频回调线程上的一切（`AudioGraph::process` 可达代码）**零分配、无锁、不 panic、不阻塞**。所有状态（滤波器记忆、延迟线、平滑参数）构造期预分配。图变更与编译只在音频线程之外发生。
- **内容与代码解耦（Event 驱动）**：游戏逻辑只触发 `Event`（"脚步"、"爆炸"），永不直接播放文件、永不硬编码总线/音量。声音设计师在数据侧决定随机化、容器、状态与实时参数（RTPC）——这是 Wwise/FMOD 的核心生产力，也是与"直接 `play(sound.wav)`" 玩具引擎的关键差异。
- **内容即图（程序化 Patch）**：单个声音本身也可以是一张可编译的 DSP 子图（Patch，对齐 UE5 MetaSounds）——用振荡器/包络/滤波/采样器程序化合成，而不仅是回放固定波形。Patch 编译后作为一个 `AudioNode` 嵌入运行时图，采样精确、零分配。
- **几何驱动空间化**：空间音频不是"距离衰减 + 声像"，而是遮挡→障碍→透射→衍射、实时/烘焙反射、Rooms & Portals、HRTF/Ambisonics 的完整链路（Steam Audio 级），且传播后端可插拔。
- **采样精确 + 多时钟**：一次性声、无缝循环、节拍量化的音乐过渡、stinger，全部对齐到样本，由样本计数驱动的 `Transport` 与并发命名时钟（对齐 UE Quartz），而非帧率抖动的游戏时钟。
- **确定性可回放**：可注入种子 RNG；相同输入产出相同样本。联机时音频作为本地表现层由确定性事件触发，天然可做 golden/parity 测试。
- **可扩展 + 可剖析**：Node / Patch / PropagationBackend / Panner / Modulator / SourceDecoder / DeviceBackend 均为 trait 插件点，第三方无需改内核即可注册；全链路可经遥测环导出到 Profiler（对齐 Wwise Profiler）。

设计取舍总表：

| 维度 | Resonance 选择 | 理由 |
|---|---|---|
| 图模型 | 单一统一 DAG（编译后处理） | 融合 Web Audio/Submix/Bus，零解释、确定性 |
| 内容模型 | Event 驱动 + 可编译 Patch | 生产力（Wwise/FMOD）+ 程序化合成（MetaSounds） |
| 处理粒度 | 定长块（block）planar 缓冲 | 对 per-channel DSP 与 SIMD 友好 |
| RT 分配 | 编译期预分配、热路径零分配 | 音频线程无 GC/malloc 抖动 |
| 内部格式 | f32 planar | 144dB 动态范围、总线可超 0dBFS 不裁切 |
| 数学 | bevy_math::ops（libm） | 跨平台位一致，可回放 |
| 参数 | 逐样本 Smoothed + 调制栈 | 无 zipper noise，可叠加 LFO/包络 |
| 空间后端 | 可插拔（几何/平台 SDK） | 兼容 Windows Sonic/Tempest/Atmos/XR |
| 调度 | 样本计数 Transport + 命名时钟 | 采样精确、多并发节拍网格 |

---

## 2. 业界参考与采纳映射

| 引擎/标准 | 我们采纳的核心思想 | 落地位置 |
|---|---|---|
| **Web Audio API** | `AudioNode` 有向图、参数自动化、离线渲染上下文 | §5 图 / §7 参数 / §22 离线 |
| **UE5 MetaSounds** | 采样精确程序化 DSP 图作为"内容"，可编译成节点 | §11 Patch |
| **UE5 Submix / Source Effect Chain** | 总线树 + 声源效果链 + 发送 | §5 图 / §9 节点 / §17 发送 |
| **UE5 Quartz** | 样本精确、量化的音乐/事件时钟，多并发时钟 | §8 调度器 |
| **UE5 Audio Modulation** | 调制控制总线、LFO/包络/曲线、参数目标、叠加规则 | §12 调制 |
| **UE5 HDR-Attenuation / Wwise HDR** | 动态响度窗口，突出前景声、压抑背景声 | §13 HDR |
| **Unity AudioMixer / Snapshots** | 总线组 + 快照插值 + sidechain ducking | §13 / §18 States |
| **Unity Spatializer & Ambisonic Decoder SDK** | 可插拔空间化器 / Ambisonic 解码接口 | §16 Panner |
| **Unity DSP Graph** | 数据导向、无 GC 的 DSP 图执行 | §5 图 / §6 缓冲 |
| **Godot AudioServer（Bus/Effect/Stream）** | 总线链、效果实例、Stream 抽象 | §5 / §9 / §10 |
| **Godot SpectrumAnalyzer / AudioEffectCapture / Generator** | 频谱分析、捕获总线、程序化推流 | §26 剖析 / §22 捕获 / §10 |
| **Godot AudioStreamPlayer3D + Microphone** | 3D 声源属性 / 麦克风输入 | §15 / §22 |
| **Wwise Event/Container/State/Switch/RTPC** | 数据驱动内容与实时参数控制 | §18 内容 |
| **Wwise Interactive Music** | 段/播放列表/量化过渡/stinger/垂直分层 | §19 音乐 |
| **Wwise Occlusion vs Obstruction** | 遮挡（直达+混响）与障碍（仅直达）区分 | §14 空间 |
| **Wwise Aux Sends / Game-Defined Aux** | 环境混响发送由游戏体积驱动 | §17 发送 |
| **Wwise Virtual Voices / Playback Limit** | 虚拟语音行为、优先级、实例上限 | §25 语音 |
| **Wwise Profiler / Meters** | 实时捕获、语音监视、计量、事件时间线 | §26 剖析 |
| **Wwise SoundBank / FMOD Bank** | 资产打包、内存/流式媒体、预取 | §20 资产 |
| **FMOD Studio Parameter/Snapshot/Transceiver** | 全局/本地参数、快照、无线发送 | §12 / §17 / §18 |
| **Steam Audio** | 遮挡·衍射·透射·反射（实时+烘焙）·HRTF·Ambisonics·探针 | §14 / §16 |
| **ITU-R BS.1770 / EBU R128** | LUFS 响度测量、门限、true-peak | §13 母带 |
| **RBJ Audio EQ Cookbook** | biquad 系数公式（已实现于 `nodes/biquad.rs`） | §9 节点 |
| **AmbiX / ACN-SN3D 约定** | Ambisonics 通道序与归一化标准 | §16 Ambisonics |
| **平台空间音频（Windows Sonic / Tempest 3D / Atmos / Meta XR）** | 对象音频床、平台原生双耳/多声道解码 | §16 平台后端 |

**次世代差异化**（相对单一中间件的组合优势）：
- 空间传播复用 `prism_physics` 的射线/几何查询做遮挡与反射，**声学与物理共享同一场景表示**，避免重复维护碰撞体。
- 声学材质与视觉材质在资产层协同（`prism_material_pipeline`），一个表面同时携带吸收/散射/透射系数与视觉 BRDF。
- 程序化 Patch（MetaSounds 级）+ Event 驱动内容（Wwise 级）+ 几何声学（Steam Audio 级）**三线合一**，而非只取其一。
- 全链路 `bevy_math::ops` 确定性数学 + 种子 RNG，使音频可做 **golden 逐样本对拍测试**，与仓库"不造假 parity"的工程 ethos 一致。
- 无锁命令环 + ECS 原生，声源即 `Entity`，天然融入 Prism 的 `Transform` 层级与并行调度。

---

## 3. 分层架构

自底向上四层，每层一个 crate，下层不依赖上层：

```
┌─────────────────────────────────────────────────────────────┐
│  L4  bevy_audio 前端（ECS 组件/系统、AudioPlayer 兼容 API）   │  → crates/bevy_audio（改接命令通道）
├─────────────────────────────────────────────────────────────┤
│  L3  prism_audio_authoring（Event/Container/State/RTPC/音乐/  │  → pkg/prism_audio_authoring
│      Patch 编译/Modulation/Bank）                             │
│      prism_audio_spatial（几何传播/HRTF/Ambisonics/panner）   │  → pkg/prism_audio_spatial
│      prism_audio_device（cpal/worklet/离线 FileSink/捕获）    │  → pkg/prism_audio_device
├─────────────────────────────────────────────────────────────┤
│  L2  prism_audio_core::nodes（gain/biquad/pan/mix/…效果/动态）│  → pkg/prism_audio_core（已落地）
├─────────────────────────────────────────────────────────────┤
│  L1  prism_audio_core（math/buffer/param/time/graph）         │  → pkg/prism_audio_core（已落地）
└─────────────────────────────────────────────────────────────┘
```

- **L1 内核（已实现）**：`math`（Sample/dB/denormal/等功率声像）、`buffer`（planar `AudioBuffer` + `ChannelLayout`）、`param`（`Smoothed`/`Ramp`）、`time`（`Transport`/`TimeSignature`）、`graph`（`AudioNode`/`AudioGraph`/编译/块渲染）。
- **L2 节点库（进行中）**：全部实现 `AudioNode` trait 的具体处理单元。首发 `GainNode`/`BiquadNode`/`StereoPanNode`/`SumNode`，规划扩展见 §9。
- **L3 子系统**：空间、编排/事件（含 Patch 编译与调制）、设备/捕获。互相独立、写集不相交，适合并行开发。
- **L4 前端**：`bevy_audio` 保留现有 `AudioPlayer`/`PlaybackSettings`/`Volume` API 兼容，内部改接无锁命令通道。

---

## 4. 概念模型：Graph / Node / Bus / Voice / Patch

- **AudioNode**（trait）：单个处理单元。契约——`process(&mut self, ctx, io)` 必须 RT 安全（不分配/锁/阻塞/panic），内部状态构造期预分配；`reset()` 清零；`latency_frames()` 报告延迟以供延迟补偿。
- **AudioGraph**：`AudioNode` 与其连接的容器。生命周期：`add_node`/`connect`（可分配，线程外）→ `compile`（Kahn 拓扑排序 + 预分配全部中间缓冲）→ `process`（音频线程，零分配）。
- **Port / 连接**：节点有若干输入/输出 port，每个 port 有 `ChannelLayout`。同一输入 port 的多条入边自动求和；一条输出可扇出多个输入。`connect_with_gain` 提供 send 风格的增益连接。层不匹配在连接期即被拒绝。
- **Bus（总线）**：约定意义上的"汇聚节点"（如 `SumNode` 或带效果链的子图），对应 Godot Bus / UE Submix。总线本身也是图中的节点，无特殊类型。
- **Voice（语音）**：一个正在发声的声源实例（一次 Event 触发可产生多个）。语音由语音池（§25）管理，虚拟化/优先级/限量，超限时按响度与优先级淘汰。
- **Patch（内容子图）**：由内容侧编排的一张小 DSP 图（振荡器/采样器/包络/滤波/数学节点），离线编译成单个 `AudioNode`（`PatchNode`）后嵌入运行时图（§11）。这是"声音即程序"的载体。

`master`：图指定某个输出 port 为母带输出，`process` 把它拷入调用者缓冲。

---

## 5. 统一实时渲染图

**核心差异化**。已落地于 `pkg/prism_audio_core/src/graph.rs`：

- **编译**：`compile()` 用 Kahn 算法做节点级拓扑排序；检测到环返回 `GraphError::Cycle`。为每个 port 预分配 `max_block` 容量的 `AudioBuffer`；对所有节点调用 `reset()`。
- **块渲染**：`process(frames, playhead, master_out)` 按拓扑序遍历——先清空本节点输入 port，再把上游输出按边增益累加进来，调用节点 `process`，最后把 master port 拷给调用者。全程零分配。
- **ProcessIo**：向节点暴露 `input(port)`/`output(port)`/`io(ip,op)`（效果原地变换的常见形态）。也可 `ProcessIo::new` 脱离图独立驱动一个节点（用于池化语音或单元测试）。
- **错误模型**：`GraphError::{UnknownNode, PortOutOfRange, LayoutMismatch, Cycle, NoMaster}`，连接/编译期充分校验，运行期不再校验（RT 铁律）。

图变更策略（线程外）：采用**三缓冲/epoch 图交换**——在任务线程构建/编译新图，通过命令环把"就绪的编译图"原子交给音频线程；音频线程在块边界切换指针，旧图经 epoch 回收队列在无引用后于任务线程 drop（避免在 RT 线程 drop 分配）。详见 §21。

---

## 6. 数据布局与缓冲池

已落地于 `buffer.rs`：

- **planar 存储**：`data[ch * capacity_frames + frame]`，每通道连续，利于 per-channel DSP 与自动向量化。
- **固定容量**：通道数与最大帧数构造期固定，RT 线程永不重分配；`active_frames` 可缩小用于流末尾的部分块。
- **通道布局**：`ChannelLayout::{Mono, Stereo, Quad, Surround5_1, Surround7_1, AmbisonicFoa}`（`#[non_exhaustive]`，可扩展 7.1.4 等 Atmos 床与 HOA 阶）。
- **混音原语**：`add_scaled`（图求和的基石）、`copy_from`、`channel_pair_mut`（无借用检查器摩擦的立体声处理）、`clear`。

规划：块内存来自图编译期分配的 arena；跨块的延迟线/卷积 tail 由各节点自持（构造期分配）。上/下混只在显式转换节点发生。

---

## 7. 参数系统与采样精确自动化

已落地于 `param.rs`：

- **Smoothed**：`Copy`、无堆数据，可直接内嵌进 RT 节点。`set_target(value, ramp)` 设置目标；`next_sample()` 在最内层 DSP 循环逐样本推进。
- **Ramp**：`Immediate`（瞬跳，仅用于非逐样本量如模式切换）、`Linear{samples}`（线性）、`Exponential{tau_samples}`（一极点，带吸附阈值确保收敛）。`Ramp::linear_seconds(s, sr)` 由秒构造。
- **无 zipper noise**：所有可听参数（增益/截止/声像）走 `Smoothed`，避免块边界阶跃爆音。
- **自动化事件**：命令环可携带带样本偏移的参数事件（"在本块第 N 帧把增益设为 X"），实现块内采样精确的参数变化（对齐 Web Audio `setValueAtTime`）。

调制叠加见 §12——最终参数值 = 基值（Smoothed）经调制栈（LFO/包络/控制总线）按叠加规则合成。

---

## 8. Transport 与采样精确调度器（多时钟）

已落地 `time.rs`（`Transport`/`TimeSignature`），规划调度器：

- **Transport**：样本计数驱动的播放头，`samples_per_beat`/`samples_per_bar`/`next_bar_boundary`/`seconds_to_samples`/`set_tempo_bpm`。这是音乐与量化事件的时间基准，不随帧率抖动。
- **命名时钟（Quartz 式）**：允许多个并发命名时钟（如"音乐时钟 128 BPM"与"环境脉冲时钟"），各自维护拍/小节网格，事件可量化到指定时钟的边界。
- **采样精确调度器**：维护按触发样本排序的事件堆；每块开始把落入本块的事件按帧偏移插入，语音/参数在精确帧启停。跨块事件保留到后续块。
- **量化过渡**：过渡对齐拍/小节/段边界（用 `next_bar_boundary`），保证无缝（§19）。
- **前瞻窗口**：调度器提前一个块预调度，配合设备缓冲吸收抖动。

---

## 9. 节点库（DSP 原语矩阵）

已实现（`pkg/prism_audio_core/src/nodes/`）：`GainNode`、`BiquadNode`（RBJ 7 型）、`StereoPanNode`（等功率）、`SumNode`（N 输入求和）。

规划扩展（每个：完整实现、构造期预分配、带 impulse/golden 稳定性测试）：

- **effects**：`ParametricEqNode`（biquad 级联）、`DelayNode`（分数延迟 + 反馈 + 湿干）、`WaveshaperNode`（过采样防混叠）、`ChorusNode`/`FlangerNode`/`PhaserNode`（调制延迟）、`FilterSweepNode`。
- **dynamics**：`CompressorNode`（软/硬拐点、前瞻）、`LimiterNode`（前瞻 true-peak）、`ExpanderGateNode`、`DuckingNode`（sidechain 输入）、`MultibandCompressorNode`。
- **reverb**：`FdnReverbNode`（反馈延迟网络，色散扩散）、`ConvolverNode`（分块 FFT 卷积，实测 IR）、`AlgorithmicRoomNode`（早反射 + 尾混）。
- **spatial**（L3 crate）：`VbapPannerNode`、`AttenuationNode`（距离衰减 + 空气吸收 + 锥形）、`HrtfNode`、`AmbisonicEncodeNode`/`AmbisonicDecodeNode`、`DopplerNode`。
- **routing**：`VcaNode`（增益控制总线）、`SendReturnNode`、`ChannelConverterNode`（上/下混）、`TransceiverNode`（无线发送，对齐 FMOD Transceiver）。
- **modulation**（见 §12）：`LfoNode`、`EnvelopeFollowerNode`、`ControlBusNode`。
- **analysis**（见 §26）：`MeterNode`（峰值/RMS/LUFS）、`SpectrumNode`（FFT）、`CaptureNode`（回读缓冲）。
- **sources**：见 §10。

---

## 10. 声源与合成

规划（L2/L3）：

- **SamplePlayerNode**：解码后 PCM 的播放头，支持循环点、分数重采样（线性/Catmull-Rom）、变速播放、start/stop 采样精确。解码在 `bevy_tasks` 任务线程，RT 线程只读环形/预载缓冲。
- **StreamingSource**：长音频流式，双缓冲预取，欠载保护（输出静音而非阻塞）。见 §20 流式媒体。
- **OscillatorNode / WavetableNode**：程序化合成（正弦/锯齿/方波/自定义波表），带 PolyBLEP 防混叠。
- **NoiseNode**：白/粉/棕噪声（种子确定性），程序化音效基石。
- **GeneratorSource**：外部程序按块推流（对齐 Godot `AudioStreamGenerator`），供 gameplay 生成的 PCM。
- **SilenceNode**：确定性静音源（占位/测试）。

资产：经 `bevy_asset` 加载 wav/ogg/flac；解码格式插件化（`SourceDecoder` trait）。程序化声音优先走 Patch（§11）而非固定波形。

---

## 11. 程序化内容图（Patch / MetaSounds 式可编译子图）

规划 crate `prism_audio_authoring`（对齐 UE5 MetaSounds，"声音即程序"）：

- **Patch 定义**：一张小型 DSP 图，节点为振荡器/采样器/包络/滤波/数学/逻辑原语，输入为 Patch 参数（频率/触发/衰减…），输出为若干音频通道。定义为数据资产（可编辑、可热重载）。
- **编译为节点**：Patch 经与运行时图相同的编译器（Kahn 拓扑 + 预分配）离线编译成一个 `PatchNode`，实现 `AudioNode`。运行时图看到的只是一个普通节点，采样精确、零分配。
- **输入/触发**：Patch 暴露命名输入（对齐 RTPC/参数）与触发端口（如 note-on）。触发经 §8 调度器采样精确注入。
- **确定性合成**：所有振荡/随机走 `bevy_math::ops` 与种子 RNG，Patch 输出逐样本可对拍。
- **复用与嵌套**：Patch 可作为节点被更大的 Patch/图引用（有界嵌套深度，编译期展开）。

价值：脚步、UI、武器、程序化环境声可完全由 Patch 合成，减少波形资产、天然随机化、内存友好——这是 MetaSounds 相对纯采样回放引擎的代际优势。

---

## 12. 音频调制系统（Modulation：控制总线 / LFO / 包络 / 曲线）

规划（对齐 UE5 Audio Modulation + FMOD 调制器）：

- **调制控制总线（Control Bus）**：一条命名的标量控制信号（如"紧张度"、"水下程度"），可被多个参数订阅。控制总线本身可被 RTPC、LFO、包络、其他总线驱动。
- **调制器（Modulator，trait）**：
  - `LfoModulator`（正弦/三角/方波/S&H，速率可同步到 §8 时钟）
  - `EnvelopeFollowerModulator`（跟随某总线电平，做自适应闪避/泵感）
  - `AdsrModulator`（触发型包络）
  - `CurveModulator`（RTPC 值经曲线映射）
- **参数目标与叠加规则**：一个参数（如某总线增益）可被多个调制器叠加，叠加规则可选 `Mix`（求和）/`Multiply`/`Max`/`Min`（对齐 UE Modulation Mixing）。最终值再进 §7 `Smoothed` 平滑。
- **RT 求值**：调制图在音频线程按块（或按控制率）求值，写入各节点的参数目标；求值零分配、拓扑有序、无环（编译期校验）。
- **与 RTPC 的关系**：RTPC 是"游戏量→参数"的直接映射；调制系统是"参数间与时变信号"的组合层。二者可级联（RTPC → 控制总线 → 多目标）。

---

## 13. 母带与动态处理（LUFS / True-Peak / Sidechain / HDR）

规划（对齐 ITU-R BS.1770 / EBU R128 + Wwise/UE HDR）：

- **响度测量**：K-weighting 预滤波 + 门限积分，输出 Integrated/Short-term/Momentary LUFS 与 Loudness Range。
- **响度归一化**：目标 LUFS（如 -16 游戏、-23 广播）自动增益（可按资产/总线归一）。
- **True-Peak Limiter**：4× 过采样峰值检测 + 前瞻，防止 inter-sample peak 削波，母带最后一级。
- **Sidechain Ducking**：`DuckingNode` 以对白/音乐总线为 sidechain 键，压低环境总线（对齐 Unity AudioMixer ducking）。
- **HDR 音频窗口**（对齐 Wwise HDR / UE HDR-Attenuation）：以场景内最响声源为参考，动态调整"响度窗口"，突出前景声（如近处枪声）、压抑背景声（远处环境），在有限扬声器动态范围内表达巨大声压差。窗口用 §7 平滑避免抽吸感。
- **Snapshot（快照）**：混音状态快照与插值切换（战斗/探索/过场），对齐 Unity Snapshot / Wwise States / FMOD Snapshot。快照由 §18 States 驱动。

---

## 14. 空间音频：几何声学传播（遮挡 / 障碍 / 透射 / 衍射 / 反射）

规划 crate `prism_audio_spatial`（Steam Audio 级，可插拔 `PropagationBackend`）：

- **遮挡（Occlusion）vs 障碍（Obstruction）**（对齐 Wwise 语义区分）：
  - **障碍（Obstruction）**：仅**直达路径**被挡（听者与声源在同一混响空间，但中间有物体），只衰减/低通直达声，混响仍完整。
  - **遮挡（Occlusion）**：直达**与**混响路径均被挡（声源在另一空间），直达与湿声一起衰减。
  - 二者由 `prism_physics` 射线 + 房间归属判定区分，分别驱动直达增益/低通与 aux 发送量。
- **透射（Transmission）**：穿过材质的频变衰减，材质携带透射损失曲线。
- **衍射（Diffraction）**：绕过边缘的路径，路径查询给出衍射角与附加衰减。
- **反射**：实时（少量镜像源/光线追踪反射）+ 烘焙（离线预计算响应，运行时插值，探针网格）。
- **声学材质**：表面携带吸收/散射/透射系数，与 `prism_material_pipeline` 视觉材质在资产层协同。

后端可插拔：默认几何后端；可注册更高精度（波动/BEM）或第三方后端。遮挡/障碍查询复用物理射线，避免重复维护碰撞体。

---

## 15. 距离与方向塑形（衰减曲线 / 锥形 / Spread / Focus / 多普勒 / 多位置）

规划（对齐 Wwise/FMOD/Unity 3D 声源属性）：

- **距离衰减曲线**：可配置形状（线性/对数/自定义曲线 + 最小/最大距离），驱动增益、低通（空气吸收）、混响发送量、Spread 等多条曲线（对齐 Wwise Attenuation ShareSets）。
- **锥形衰减（Cone）**：声源朝向 + 内/外锥角 + 外锥增益与低通，模拟指向性声源（喇叭/人声）。
- **Spread（扩散）**：随距离控制声像宽度——远处点声源收窄，近处可环绕，避免"点声源贴脸"失真。
- **Focus（聚焦）**：控制能量集中程度，与 Spread 配合塑造宽/窄声像。
- **多普勒（Doppler）**：由听者/声源相对径向速度计算频移，`DopplerNode` 用分数延迟线实现连续变调（无爆音），可配置多普勒强度系数。速度取自 `Transform` 帧间差分或显式速度组件。
- **多位置声源（Multi-Position）**：一个逻辑声源映射到多个空间位置（对齐 Wwise Multi-Position），用于大型/分布式声源（河流、人群、机器），按"最近/加权/全部"模式合成空间参数。

上述所有塑形量经 §7 `Smoothed` 平滑，随听者/声源运动逐块更新。

---

## 16. HRTF / Ambisonics / 对象音频 / 平台空间后端

规划：

- **HRTF 双耳渲染**：分块卷积 HRIR（按方位/仰角插值），近场效应与 ITD/ILD，可加载自定义 HRTF 数据集。
- **Ambisonics**：FOA/HOA 场景总线，声源编码进 Ambisonic 域，最终按输出布局解码（双耳/多声道）。约定采用 **AmbiX（ACN 通道序 + SN3D 归一化）**，与主流工具链兼容。已在 `ChannelLayout::AmbisonicFoa` 预留，HOA 阶数可扩展。
- **对象音频 / Atmos**：对象元数据（位置/大小）输出到支持的床（7.1.4）或下混到扬声器/耳机。
- **平台空间后端**（`Panner`/输出适配可插拔）：耳机（内建 HRTF）、立体声、5.1/7.1；并可桥接平台原生空间 API——**Windows Sonic / Spatial Sound**、**索尼 Tempest 3D**、**杜比 Atmos**、**Meta XR Audio**——由 `prism_audio_device` 侦测并选择解码路径。
- **输出适配**：自动按设备与用户偏好选择 HRTF / 多声道 / 对象床路径。

Panner 可插拔（`Panner` trait），对齐 Unity Spatializer / Ambisonic Decoder SDK。

---

## 17. 环境与辅助发送（Aux Sends / Reverb Zones / Rooms & Portals）

规划（对齐 Wwise Aux Sends / UE Submix Sends / FMOD Snapshot 区域）：

- **辅助发送（Aux Send）**：声源除干路外，可按可变增益发送到一个或多个混响/效果返回总线。发送量随距离曲线（§15）与遮挡/障碍状态（§14）动态调整。
- **游戏定义发送（Game-Defined Aux）**：由玩家所处的**混响体积（Reverb Zone）**自动决定发往哪个环境混响与发送量（进洞穴→洞穴混响，出洞→户外），过渡用 §7 平滑。
- **Rooms & Portals**：房间体积 + 门户连接。声音在房间间经门户传播与滤波；门户开合、朝向影响传播增益与方向（与 §14 遮挡/障碍联动）。
- **返回总线**：混响/延迟返回作为普通节点存在于统一图（§5），可再被母带链（§13）处理。

---

## 18. 内容模型：Event / Container / State / Switch / RTPC

规划 crate `prism_audio_authoring`（Wwise/FMOD 生产力核心）：

- **Event**：游戏触发的最小单位（"footstep"、"explosion"）。Event 携带动作（play/stop/set-param/set-switch/set-state）。游戏代码只发 Event。
- **Container**：
  - `Random`（随机选一，带避免重复窗口）
  - `Sequence`（顺序播放）
  - `Blend`（按参数交叉淡化多层，如引擎转速）
  - `Switch`（按 Switch 状态选分支）
  - `Scatter`（空间散布，如群鸟）
- **States**（全局）：游戏状态（战斗/潜行）驱动混音快照（§13）与 Event 变体。
- **Switches**（每对象）：如"地表材质"决定脚步音色。
- **RTPC（实时参数控制）**：连续游戏量（速度/血量/紧张度）经映射曲线驱动任意参数（音量/滤波/音高），可级联到调制控制总线（§12）。

编排层把这些解析成对 L1 图、§11 Patch 与 §8 调度器的命令，经 §21 命令环下发。

---

## 19. 交互音乐系统

规划（对齐 Wwise Interactive Music / FMOD Transition）：

- **Segment / 播放列表**：音乐段带入点/出点/前后余量；播放列表定义段的顺序/循环/随机。
- **量化过渡**：过渡在拍/小节/段边界发生（用 §8 `next_bar_boundary` 与命名时钟），采样精确无缝。可配过渡段（transition segment）桥接。
- **Stinger**：叠加短乐句（如命中提示），精确对齐到量化点。
- **垂直分层**：多轨按 State/RTPC 增减层（战斗强度）。
- **水平重排**：按玩法分支切换段。

---

## 20. 资产、Bank 与流式媒体

规划（对齐 Wwise SoundBank / FMOD Bank）：

- **Bank 打包**：把一组 Event/Container/Patch/媒体打包为可加载/卸载单元，按关卡或情景加载，控制内存占用。经 `bevy_asset` 加载与热重载。
- **内存 vs 流式媒体**：短音效常驻内存池；长音乐/环境走**流式**（磁盘→解码任务→环形预取缓冲→RT 只读），欠载输出静音不阻塞。
- **预取（Prefetch）**：流式声的首段常驻内存，保证零延迟起播，其余边播边取。
- **内存池**：解码缓冲、语音状态、延迟线来自构造期分配的池，RT 线程零 malloc。Bank 卸载在任务线程回收（epoch，§21）。
- **解码任务**：`bevy_tasks` 后台解码，格式插件化（`SourceDecoder` trait：wav/ogg/flac/自定义）。

---

## 21. 无锁 ECS 集成层与线程/内存模型

规划：

- **线程模型**：
  - **游戏/ECS 线程**：产生 Event 与命令，写入命令环；从遥测环读回状态。
  - **音频回调线程（RT）**：唯一执行 `AudioGraph::process` 的线程，零分配/锁/panic；每块吸收命令、渲染、写遥测。
  - **任务线程（bevy_tasks）**：解码、Bank 加载、图/Patch 编译、烘焙；产出的编译图/缓冲经命令环交付。
- **命令环（Command Ring）**：ECS/gameplay → 音频线程的 MPSC 无锁环。命令包括 play/stop/set-param/set-transform/set-switch/set-state/swap-graph。RT 线程每块开始批量吸收，带样本偏移的命令交 §8 调度器。
- **遥测环（Telemetry Ring）**：音频线程 → ECS 的 SPSC 环，回传峰值/RMS/LUFS/语音数/事件完成回调/Profiler 帧，用于 UI 表头、gameplay 反馈与 §26 剖析。
- **图/资源交换与回收**：编译好的新图/新缓冲经命令环原子交付，块边界切换指针；旧对象进 **epoch 回收队列**，确认无 RT 引用后在任务线程 drop（RT 线程从不 drop 分配）。
- **零锁**：主线程与音频线程只经环通信，无互斥锁，无优先级反转。

前端 `bevy_audio` 保留 `AudioPlayer`/`PlaybackSettings`/`Volume` API 兼容，内部翻译为命令。

---

## 22. 设备后端、离线渲染与输入捕获

规划 crate `prism_audio_device`：

- **cpal 后端**：桌面/移动原生输出，处理设备采样率/块大小协商与重采样；侦测平台空间能力（§16）。
- **AudioWorklet 后端**：Web 平台（wasm），在 worklet 线程跑图。
- **FileSink（离线）**：以任意块大小离线渲染到 wav，用于 golden 测试与过场预渲染。**离线路径与实时路径共用同一图**，保证一致性。
- **输入捕获（麦克风 / 回读）**（对齐 Godot Microphone / AudioEffectCapture）：设备输入作为源节点进图；`CaptureNode` 从任意总线回读到环形缓冲，供录制、语音、频谱 UI（§26）。
- **欠载保护**：设备回调欠载时输出静音并计数，不阻塞。

---

## 23. 无障碍（Accessibility）

规划（对齐 `bevy_a11y`）：

- **字幕/描述**：Event 可携带字幕元数据，触发时经遥测环上报给 UI 层。
- **单声道下混**：单耳听力用户一键下混。
- **语音优先/闪避增强**：对白优先级最高，可强化 ducking（§13）保证可懂度。
- **视觉声音提示**：关键音效可触发视觉指示（方向/类型）。
- **动态范围压缩档**：夜间/听障档启用更强压缩（§13 HDR 窗口收窄）。

---

## 24. 确定性与网络

- **种子 RNG**：Container 随机、Scatter 散布、Patch 随机源均用可注入种子 RNG，回放/测试可复现。
- **确定性数学**：全链路 `bevy_math::ops`（libm），跨平台位一致。
- **网络模型**：音频是本地表现层，由确定性游戏事件触发，不同步音频样本；避免网络抖动进入 RT 路径。
- **固定块**：离线与测试用固定块大小，产出逐样本可对拍的 golden。

---

## 25. 语音管理与虚拟化

规划（对齐 Wwise Virtual Voices / Playback Limit）：

- **语音池**：预分配固定数量语音，避免 RT 分配。
- **优先级 + 响度淘汰**：超限时按 (优先级, 估计响度, 距离) 淘汰最不重要者。
- **虚拟语音行为**（对齐 Wwise Virtual Voice Behavior）：被淘汰/超阈值的语音可配置行为——
  - `ContinueVirtual`（继续推进播放头不出声，资源空出可复活，保证循环声位置连续）
  - `Kill`（直接停止释放）
  - `RestartFromBeginning`（复活时从头播）
  - `PlayFromElapsedTime`（复活时从应到达位置播）
- **限量（Playback Limit）**：同类声源实例上限（如最多 8 个同时脚步），超限按策略拒绝或替换最旧/最弱者，对齐 Wwise Playback Limit。
- **进入/离开阈值**：以估计响度的进出阈值（带滞回）决定何时虚拟化，避免边界抖动。

---

## 26. 剖析、遥测与可视化调试

规划（对齐 Wwise Profiler / Godot SpectrumAnalyzer / Unity Audio Profiler）：

- **实时捕获**：遥测环（§21）导出每块的语音清单、总线电平、CPU 占用、事件时间线，供 UI/Profiler 面板回放（可录制会话）。
- **计量（Meters）**：`MeterNode` 在任意总线插入，回读峰值/RMS/LUFS/相位/相关度。
- **频谱分析**：`SpectrumNode`（FFT）输出频带能量，供 gameplay 反应（音乐可视化、节拍触发）与调试。
- **语音监视**：列出活动/虚拟语音及其优先级/响度/衰减状态，定位"为什么这个声音不响"。
- **图检视**：导出当前编译图（节点/连接/延迟）为可视化，配合 `bevy_diagnostic` 面板。
- **golden 差异**：离线渲染与参考波形的逐样本 diff 可视化，回归定位。

---

## 27. 性能预算与验收

| 指标 | 目标 |
|---|---|
| RT 回调分配 | 0（热路径零 malloc/lock） |
| 512 语音 @48k | 单核 < 30% 预算（桌面档） |
| 图/Patch 编译 | 线程外，< 数 ms（百节点级） |
| 延迟 | 设备块 + 前瞻，可配置（默认 ~10ms 桌面） |
| LUFS 归一误差 | < 0.5 LU |
| True-peak | ≤ -1 dBTP（母带后） |
| 多普勒/衰减更新 | 逐块平滑，无 zipper/爆音 |
| golden 对拍 | 逐样本 ULP 级一致（固定种子/块） |

---

## 28. 质量与可信度基础设施

- **单元测试**：每节点带 impulse/稳定性/golden 测试（已有：graph 求和/send/环拒绝/层校验、biquad 频响/稳定、pan 功率守恒、param 收敛，共 30 项通过）。
- **no_std + std 双构建**：内核与节点在 `--no-default-features` 下亦编译（libm 后端），保证嵌入式/wasm 可移植。
- **clippy 零告警**：遵守工作区严格 lints（`missing_docs`、`disallowed-methods` 确定性数学、`allow_attributes_without_reason` 等）。
- **离线 golden 渲染**：FileSink 渲染参考波形，回归对拍。
- **provenance**：无 UE/Unity/Godot/Wwise/FMOD 源码或衍生代码，所有 DSP 出自公开标准知识（RBJ cookbook、BS.1770、等功率声像、AmbiX/ACN-SN3D、FDN 等）。

---

## 29. Crate 拆分与落地形态

| Crate | 层 | 内容 | 状态 |
|---|---|---|---|
| `pkg/prism_audio_core` | L1+L2 | math/buffer/param/time/graph + nodes | ✅ 基础层已落地 |
| `pkg/prism_audio_spatial` | L3 | 几何传播/HRTF/Ambisonics/panner/多普勒/平台后端桥 | 规划 |
| `pkg/prism_audio_authoring` | L3 | Event/Container/State/Switch/RTPC/Patch 编译/Modulation/交互音乐/Bank | 规划 |
| `pkg/prism_audio_device` | L3 | cpal/worklet/FileSink/输入捕获 | 规划 |
| `crates/bevy_audio` | L4 | ECS 前端（改接命令通道，保留兼容 API） | 规划改造 |

**并行开发拆分**（写集不相交，可 fan-out 给并行 agent）：
- effects（biquad 级联/delay/waveshaper/调制延迟）
- dynamics（compressor/limiter/gate/ducking/multiband）
- reverb（FDN/convolver/早反射）
- spatial（panner/attenuation/cone/spread/doppler/HRTF/ambisonics）
- sources（sample player/oscillator/noise/streaming/generator）
- routing（bus/send/VCA/converter/transceiver）
- patch（内容子图编译器 + 合成原语）
- modulation（LFO/包络/控制总线/叠加）
- scheduler（采样精确调度器 + 命名时钟）
- 无锁环（command/telemetry ring、voice pool、epoch 回收）
- 剖析（meter/spectrum/capture + 面板）

---

## 30. 路线图

- **M0 内核（已完成）**：math/buffer/param/time/graph + 首发 4 节点，30 测试，双构建，零告警，已 commit。
- **M1 效果与动态**：EQ/delay/waveshaper/调制延迟 + compressor/limiter/gate/ducking + reverb（FDN/convolver）。
- **M2 声源与调度**：sample player/oscillator/noise/streaming/generator + 采样精确调度器 + 命名时钟 + 语音池与虚拟语音行为。
- **M3 ECS 桥与设备**：命令/遥测环 + epoch 回收 + cpal/worklet/FileSink/输入捕获 + `bevy_audio` 前端改造（兼容 API）。
- **M4 空间**：遮挡/障碍/透射/衍射/反射 + 距离塑形（衰减/锥形/spread/focus/doppler/多位置）+ HRTF/Ambisonics + Rooms&Portals + Aux 发送 + 平台空间后端桥。
- **M5 编排、Patch 与音乐**：Event/Container/State/Switch/RTPC + Patch 编译器与合成原语 + Modulation（LFO/包络/控制总线）+ 交互音乐（段/过渡/stinger）+ Bank/流式。
- **M6 母带、合规与剖析**：LUFS 归一 + true-peak limiter + HDR 窗口 + snapshot + 无障碍 + Profiler/频谱/计量面板。

---

## 31. 关键扩展点清单

| 扩展点 | trait | 用途 |
|---|---|---|
| 处理单元 | `AudioNode` | 任意 DSP/总线/空间化 |
| 内容子图 | `PatchNode`（编译产物） | 程序化合成声音 |
| 传播后端 | `PropagationBackend` | 几何/波动/第三方空间化 |
| 声像/空间化 | `Panner` | VBAP/HRTF/Ambisonic/平台 SDK |
| 调制器 | `Modulator` | LFO/包络/曲线/自定义调制 |
| 解码器 | `SourceDecoder` | wav/ogg/flac/自定义 |
| 设备后端 | `DeviceBackend` | cpal/worklet/离线/捕获 |
| 参数源 | 控制总线/RTPC 映射 | 参数联动与调制 |

---

## 32. 开放问题

- 图/资源交换的旧对象回收策略：延迟队列 vs 引用计数 vs epoch（当前倾向 epoch）。
- HOA 阶数与 CPU 预算的默认档位。
- 卷积混响的分块 FFT 大小与延迟/CPU 折中。
- 移动端功耗档：降采样率/降语音数的自适应策略。
- 声学材质与视觉材质资产的字段合并范围。
- Patch 嵌套深度上限与编译展开的内存上界。
- 平台空间后端的能力探测与优雅降级策略。

---

## 33. 术语表

- **RT-safe**：实时安全，指音频回调线程可执行（零分配/锁/阻塞/panic）。
- **block / 块**：一次处理的定长样本帧数。
- **planar**：每通道连续存储（对比 interleaved 交织）。
- **Patch**：可编译成单个节点的内容侧程序化 DSP 子图（MetaSounds 式）。
- **zipper noise**：参数突变导致的爆音。
- **denormal**：接近零的次正规浮点，某些 CPU 上运算极慢，需 flush。
- **LUFS**：响度单位（BS.1770），感知响度测量。
- **True-Peak**：过采样后的采样间峰值。
- **HDR 音频**：动态响度窗口，突出前景声、压抑背景声。
- **RTPC**：实时参数控制（游戏量→音频参数）。
- **Modulation**：参数间与时变信号（LFO/包络/控制总线）的组合调制层。
- **Occlusion / Obstruction**：遮挡（直达+混响均挡）/ 障碍（仅直达挡）。
- **Aux Send**：向混响/效果返回总线的辅助发送。
- **HRTF**：头相关传输函数，双耳空间化基础。
- **Ambisonics / AmbiX**：全景声场表示（W/X/Y/Z…），AmbiX = ACN 通道序 + SN3D 归一化。
- **Doppler**：相对运动导致的频移。
- **FDN**：反馈延迟网络（混响结构）。
- **VBAP**：基于矢量基的幅度声像（多声道定位）。
- **Virtual Voice**：被淘汰但可复活的语音（继续推进/杀死/重启/按时长续播）。
- **Bank / SoundBank**：可加载/卸载的资产打包单元。
- **epoch 回收**：确认无 RT 引用后在任务线程释放旧对象的无锁回收策略。
