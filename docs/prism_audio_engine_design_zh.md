# Prism Resonance 次世代音频引擎设计方案

> 面向 Prism（Bevy fork）的数据导向、RT 安全、图编译式高性能音频引擎设计。
> 借鉴 UE5（MetaSounds / Submix / Quartz / Audio Modulation / HDR-Attenuation）、Unity（AudioMixer / DSP Graph / Spatializer & Ambisonic Decoder SDK / Snapshots）、Godot（AudioServer Bus/Effect/Stream / 频谱分析 / 麦克风捕获 / 程序化 Generator）、Wwise（Event/Container/State/Switch/RTPC / Interactive Music / HDR / Occlusion·Obstruction / Aux Sends / Profiler / SoundBank）、FMOD Studio（Event/Parameter/Snapshot / Bank 流式 / Transceiver）、Steam Audio（遮挡·衍射·透射·反射·烘焙·HRTF·Ambisonics）、Web Audio API（AudioNode 图）、微软 Windows Sonic / 索尼 Tempest 3D / 杜比 Atmos / Meta XR Audio（平台空间后端），取长补短。
> **空间与内容一等公民**：几何驱动空间化与 Event 驱动内容模型同为一等公民；程序化合成（Patch）、调制（Modulation）与母带合规（LUFS/True-Peak/HDR）三者贯通，可达顶级次世代 AAA 质量。
> 本文档为设计规格与落地实现的权威规范；采用纯经典 DSP 路线，不含任何 AI/ML 内容，不含任何 UE/Unity/Godot/Wwise/FMOD 源码或衍生代码。

- 版本: v0.7（基础层已落地编码：`pkg/prism_audio_core`，M1 效果族已落地首个节点 `ParametricEqNode`（级联复用 `Biquad`，35 测试绿/clippy 零告警/no_std 双构建通过）；本版聚焦「逐引擎深读」精修方案：扩展 §2 为业界参考+采纳映射+逐引擎深读（UE5/Unity/Godot 借鉴·覆盖·超越三段式），并补齐各引擎较新特性——UE5 MetaSound Builder API 运行时构图（§11）、Audio Gameplay Volumes（§17）、Audio Insights 检视（§26）；Godot AudioStreamInteractive 片段式交互流与过渡类型（§19）、AudioStreamPolyphonic 多voice复用（§25）、延迟补偿播放头查询（§8）；Unity Audio Random Container 原生随机容器（§18）、Timeline 音频轨（§19）；相应增补 §41 扩展点 `PatchBuilder` 与 §42 开放问题。v0.6 前序：新增内容生产与跨模态层：对白与本地化（程序化对白/语言 Bank/字幕同步/viseme 口型）、触感与跨模态输出（音频同源触感/触感总线/DualSense·双马达后端）、程序化环境音景（Soundscape 调色板与散布）、实时授权与远程工具 API（WAAPI 式远程遥测+白名单写命令/live tuning/授权热重载），并为 §16 增补头部追踪双耳、§28 增补属性/模糊测试。v0.5 前序：编译图执行模型（拓扑计划/缓冲活跃度分配/就地别名/PDC）、并行 DSP 图调度（Job 化/确定性并行/岛屿划分）、GPU 加速几何声学（共享渲染器 BVH 的声线与路径追踪）、性能自适应治理与音频 LOD、心理声学虚拟化与声源聚类、时间伸缩变调与重采样质量分级；扩展了参考映射、扩展点、路线图与术语表。v0.4 前序：程序化内容图 Patch、调制系统、多普勒/锥形/Spread/Focus/多位置、遮挡与障碍区分、Aux 发送与环境、HDR 音频、Bank/流式/内存、输入捕获、平台空间后端、虚拟语音行为、Profiler 与可视化调试）
- 范围: 一步到位（统一渲染图 / 采样精确调度 / 程序化 Patch / 调制 / 几何声学 / Event+RTPC 编排 / 交互音乐 / LUFS+HDR 母带 / 平台空间输出）
- 适用引擎: Prism / Bevy ECS 生态
- 关键依赖: bevy_ecs（并行 ECS）、bevy_math（glam SIMD + `ops` 确定性标量数学）、bevy_tasks（资产解码/烘焙任务）、bevy_asset（音频资产与 Bank）、bevy_transform（听者/声源位姿）、bevy_a11y（无障碍）、cpal/AudioWorklet（设备后端，前端 crate）
- 复用的现有设施: bevy_asset 加载与热重载、bevy_tasks 异步解码/烘焙、bevy_math::ops（跨平台确定性 sin/cos/exp/log）、bevy_transform 空间层级、bevy_diagnostic 诊断面板、prism_physics 射线/几何查询（遮挡与反射复用）、prism_material_pipeline 表面属性协同
- 关联文档: `prism_physics_design_zh.md`（几何/射线查询原语，供遮挡与反射复用）、`prism_material_pipeline_design_zh.md`（声学材质与视觉材质的表面属性协同）

---

## 目录
1. 设计哲学与目标
2. 业界参考、采纳映射与逐引擎深读（UE5 / Unity / Godot）
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
29. 编译图执行模型（拓扑计划 / 缓冲活跃度分配 / 就地与别名优化）
30. 并行 DSP 图调度（Job 化渲染 / 确定性并行 / 岛屿划分）
31. GPU 加速几何声学（共享渲染器 BVH 的光线与路径追踪）
32. 性能自适应治理与音频 LOD（CPU 预算驱动质量缩放）
33. 心理声学虚拟化与声源聚类（掩蔽感知剔除 / 对象床限制）
34. 时间伸缩与变调 / 重采样质量分级（与多普勒解耦）
35. 对白与本地化（Dialogue / Localization / 字幕 / 口型）
36. 触感与跨模态输出（Haptics / Motion / 手柄反馈）
37. 程序化环境音景（Soundscape / 程序化 Ambience）
38. 实时授权与远程工具 API（Live Authoring / WAAPI 式 / 热调）
39. Crate 拆分与落地形态
40. 路线图
41. 关键扩展点清单
42. 开放问题
43. 术语表

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
| 执行模型 | 编译期 ExecPlan + 缓冲活跃度分配 | 零解释、内存最优、可逐字节对拍 |
| 图调度 | 编译期岛屿划分 + Job 化并行 | 高语音密度可扩展，确定性不变 |
| 声学加速 | 复用渲染器 GPU BVH 做声线/路径 | 声学与视觉同一几何真相，异步无 RT 阻塞 |
| 质量伸缩 | CPU 预算闭环 + 音频 LOD | 稳帧不爆音，感知优先分配算力 |

---

## 2. 业界参考、采纳映射与逐引擎深读（UE5 / Unity / Godot）

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
| **现代渲染图（render graph）transient 资源别名** | 编译期缓冲活跃度分析 + 图着色复用中间缓冲 | §29 执行模型 |
| **Unity DSP Graph（数据导向并行执行）** | 岛屿划分 + Job 化确定性并行渲染 | §30 图调度 |
| **Steam Audio 光线/路径 + Prism GPU BVH** | 复用渲染器加速结构做 GPU 声学传播 | §31 GPU 声学 |
| **游戏引擎 LOD / Wwise 语音上限（闭环化）** | CPU 预算驱动质量档与音频 LOD 自适应 | §32 治理器 |
| **心理声学掩蔽 / Atmos 对象床上限** | 掩蔽感知虚拟化 + 邻近声源聚类归并 | §33 感知层 |
| **相位声码器 / WSOLA / 多相 sinc** | 变调不变速 / 变速不变调 / 重采样质量分级 | §34 时间伸缩 |

| **FMOD Programmer Instrument / Wwise External Sources·Dialogue Event** | 运行时程序化选择对白媒体、语言变体、决策树命中 | §35 对白 |
| **本地化字幕轨 / viseme 口型时间线** | 字幕同步与口型/表情驱动数据（不占 RT） | §35 对白 |
| **Wwise Motion / PS5 DualSense·Tempest Haptics** | 音频同源触感、触感总线、宽频/双马达可插拔后端 | §36 触感 |
| **UE5 Soundscape** | 环境状态调色板 + 程序化 one-shot 散布，低重复环境床 | §37 音景 |
| **Wwise Authoring API (WAAPI) / FMOD Live Update** | 远程只读遥测 + 白名单写命令 + 实时调参与授权热重载 | §38 授权 |
| **UE5 MetaSound Builder API（运行时构图）** | 运行时以代码/数据增量拼装并热切换 Patch，而非仅离线烘焙 | §11 Patch |
| **UE5 Audio Gameplay Volumes（5.3+）** | 体积驱动的室内外/混响/衰减覆盖与门户连通，取代旧混响体 | §17 发送 |
| **UE5 Audio Insights（5.4+）** | 引擎内实时声源/总线/参数/虚拟化检视面板 | §26 剖析 |
| **Godot 4.x AudioStreamInteractive / Playlist / Synchronized** | 片段式交互音乐（immediate/next-beat/next-bar/marker 过渡）与多流同步 | §19 音乐 |
| **Godot 4.x AudioStreamPolyphonic** | 单播放器动态多 voice 复用（脚步/连发免手动建声源） | §25 语音 |
| **Godot 延迟补偿播放头查询** | playback_position + time_since_last_mix − output_latency 对齐画面 | §8 调度器 |
| **Unity Audio Random Container（2023.2+）** | 引擎原生随机容器（音高/音量/顺序/避免重复） | §18 内容 |
| **Unity Timeline 音频轨** | 时间线上采样精确编排音频剪辑/事件 | §19 音乐 |

**次世代差异化**（相对单一中间件的组合优势）：
- 空间传播复用 `prism_physics` 的射线/几何查询做遮挡与反射，**声学与物理共享同一场景表示**，避免重复维护碰撞体。
- 声学材质与视觉材质在资产层协同（`prism_material_pipeline`），一个表面同时携带吸收/散射/透射系数与视觉 BRDF。
- 程序化 Patch（MetaSounds 级）+ Event 驱动内容（Wwise 级）+ 几何声学（Steam Audio 级）**三线合一**，而非只取其一。
- 全链路 `bevy_math::ops` 确定性数学 + 种子 RNG，使音频可做 **golden 逐样本对拍测试**，与仓库"不造假 parity"的工程 ethos 一致。
- 无锁命令环 + ECS 原生，声源即 `Entity`，天然融入 Prism 的 `Transform` 层级与并行调度。
- **声学复用渲染器 GPU 加速结构**（§31）：作为"渲染器 + 音频引擎"合体，声学传播直接跑在渲染管线的 BVH/meshlet 上，独立中间件（各自维护声学场景）无法做到，这是最强次世代差异。
- **编译图执行 + 确定性并行 + 质量治理**（§29/§30/§32）：把图当作可编译、可并行、可逐字节对拍的确定性计划，并在 CPU 预算内闭环缩放质量——兼得高密度、稳帧与可回放。

**逐引擎深读（借鉴 / 我们覆盖在哪 / 我们如何超越）**

*UE Series（UE4 SoundCue → UE5 MetaSounds 时代）*
- **借鉴**：MetaSounds 把“单个声音”变成采样精确、可编译的程序化 DSP 图；Submix 树 + Source/Submix Effect Chain 做分层路由；Quartz 提供样本精确、多并发的音乐/事件时钟；Audio Modulation 用控制总线做参数联动；HDR-Attenuation 做动态响度窗口；较新版还有 **MetaSound Builder API**（运行时构图）、**Audio Gameplay Volumes**（体积驱动室内外/混响/门户，5.3+）、**Audio Insights**（引擎内检视，5.4+）、Convolution Reverb Submix。
- **覆盖**：Patch=§11、Submix/效果链=§5/§9/§17、Quartz=§8、Modulation=§12、HDR=§13；本版补齐 Builder API=§11、AGV=§17、Audio Insights=§26。
- **超越**：MetaSounds 的几何空间化仍依赖外部插件（Steam Audio），我们把几何声学（§14）与 **GPU BVH 声学**（§31）内建并与渲染器共享同一几何真相；全链路 `bevy_math::ops` 确定性数学使 Patch 与母带可做 **golden 逐样本对拍**（UE 不保证跨平台位一致）。

*Unity（AudioMixer / DSP Graph / DOTS Audio 时代）*
- **借鉴**：AudioMixer 组 + Snapshot 插值 + 暴露参数 + sidechain ducking；Native Audio Plugin SDK 的可插拔 **Spatializer / Ambisonic Decoder** 接口；DSP Graph（DOTS Audio）的数据导向、无 GC 并行执行；较新的 **Audio Random Container**（2023.2+，引擎原生随机容器）与 Timeline 音频轨。
- **覆盖**：Mixer/Snapshot/ducking=§13/§18、可插拔 Panner/Ambisonic=§16、DSP Graph 数据导向=§5/§6 与岛屿并行=§30；本版补齐 Random Container=§18、Timeline 轨=§19。
- **超越**：Unity 的 Snapshot 是控制率插值、AudioMixer 图为运行时解释；我们是**编译期 ExecPlan + 缓冲活跃度分配**（§29）零解释执行，参数逐样本 `Smoothed`（§7）无 zipper，且 §30 岛屿并行是**确定性可对拍**的（DOTS Audio 不保证跨平台样本一致）。

*Godot（AudioServer：Bus / Effect / Stream 时代）*
- **借鉴**：AudioServer 的 Bus 链 + 效果实例 + Stream 抽象；SpectrumAnalyzer / AudioEffectCapture / AudioStreamGenerator（频谱/捕获/程序化推流）；Area 驱动的混响总线覆盖；麦克风捕获；较新的 **AudioStreamInteractive**（片段图 + 过渡类型 immediate/next-beat/next-bar/marker）、**AudioStreamPolyphonic**（单播放器多 voice）、**AudioStreamSynchronized/Playlist**，以及**延迟补偿播放头查询**。
- **覆盖**：Bus/Effect/Stream=§5/§9/§10、频谱/捕获/Generator=§26/§22/§10、Area 混响=§17、麦克风=§22；本版补齐交互流=§19、Polyphonic=§25、延迟补偿播放头=§8。
- **超越**：Godot 的空间化仅距离衰减 + 简单混响，无遮挡/衍射/HRTF/Ambisonics 完整链路；我们提供 Steam Audio 级几何传播（§14/§16）；Godot 混音为运行时逐总线处理，我们是编译图 + Job 化并行（§29/§30）+ CPU 预算治理（§32），密度与稳帧维度代差领先。

*专业中间件参照（Wwise / FMOD / Steam Audio）*：内容生产力（Event/Container/State/Switch/RTPC=§18、交互音乐=§19、Bank/流式=§20、Profiler=§26、WAAPI/Live Update=§38）与几何声学（§14/§16）已系统性采纳；差异化在于把中间件的“内容生产 + 几何声学”与引擎内建的“渲染器共享 GPU 声学 + 编译图确定性并行 + 可回放对拍”合一，而非以外挂中间件形式并存。

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
- **延迟补偿播放头查询（对齐 Godot）**：向 gameplay 暴露“画面对齐”的播放位置 `pos = raw_playhead + time_since_last_mix − output_latency`，供画面/字幕/节奏玩法精确同步，而非直接用抖动的游戏帧时钟读原始播放头。

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
- **运行时增量构图（Builder API 式，对齐 UE5 MetaSound Builder API）**：除离线烘焙外，提供 `PatchBuilder` 在任务线程以代码/数据增量拼装或改写 Patch，编译成新 `PatchNode` 后经无锁命令环与 §21 epoch 原子热切换（旧节点确认无 RT 引用后延迟回收），实现运行时程序化音色演化——超越“只能离线固化内容”的传统管线，同时不破坏 RT 铁律（构图/编译永不在音频线程）。

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
- **头部追踪双耳（Head-tracked Binaural）**（对齐 Meta XR Audio / Steam Audio 头追）：XR/VR 下以低延迟头追姿态旋转 Ambisonic 场或重选 HRIR 方位，头动到声像更新走短前瞻路径，避免"声像黏在头上"；姿态更新经命令环下发，RT 侧插值平滑。
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
- **片段式交互流（对齐 Godot AudioStreamInteractive）**：以“剪辑图”描述交互音乐——节点为剪辑，边为带触发条件与**过渡类型**（immediate / next-beat / next-bar / segment-end / marker）的转移，采样精确切换，可挂过渡剪辑与淡入淡出；与上面的 Segment/播放列表模型互补，覆盖从“轻量剪辑跳转”到“专业段/垂直分层”的完整谱系。
- **时间线编排（对齐 Unity Timeline 音频轨）**：离线在时间线上按样本位置编排剪辑/stinger/事件，经 §8 调度器采样精确回放，用于过场与脚本化演出。

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
- **多voice复用声源（对齐 Godot AudioStreamPolyphonic）**：单个逻辑声源可动态复用池内多条 voice 播放重叠实例（脚步/连发/碰撞），免去为每次触发手动新建声源；复用与上限受同一语音池与 Playback Limit 约束。

---

## 26. 剖析、遥测与可视化调试

规划（对齐 Wwise Profiler / Godot SpectrumAnalyzer / Unity Audio Profiler）：

- **实时捕获**：遥测环（§21）导出每块的语音清单、总线电平、CPU 占用、事件时间线，供 UI/Profiler 面板回放（可录制会话）。
- **计量（Meters）**：`MeterNode` 在任意总线插入，回读峰值/RMS/LUFS/相位/相关度。
- **频谱分析**：`SpectrumNode`（FFT）输出频带能量，供 gameplay 反应（音乐可视化、节拍触发）与调试。
- **语音监视**：列出活动/虚拟语音及其优先级/响度/衰减状态，定位"为什么这个声音不响"。
- **图检视**：导出当前编译图（节点/连接/延迟）为可视化，配合 `bevy_diagnostic` 面板。
- **golden 差异**：离线渲染与参考波形的逐样本 diff 可视化，回归定位。
- **引擎内实时检视（对齐 UE5 Audio Insights）**：无需外部工具即可在编辑器/运行时面板检视活动声源、总线电平、参数/RTPC 当前值、虚拟化状态与每块 CPU 占用，数据源自遥测环（§21），只读、不占 RT。

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

- **单元测试**：每节点带 impulse/稳定性/golden 测试（已有：graph 求和/send/环拒绝/层校验、biquad 频响/稳定、pan 功率守恒、param 收敛，共 31 项通过：27 单测 + 4 doctest）。
- **no_std + std 双构建**：内核与节点在 `--no-default-features` 下亦编译（libm 后端），保证嵌入式/wasm 可移植。
- **clippy 零告警**：遵守工作区严格 lints（`missing_docs`、`disallowed-methods` 确定性数学、`allow_attributes_without_reason` 等）。
- **离线 golden 渲染**：FileSink 渲染参考波形，回归对拍。
- **provenance**：无 UE/Unity/Godot/Wwise/FMOD 源码或衍生代码，所有 DSP 出自公开标准知识（RBJ cookbook、BS.1770、等功率声像、AmbiX/ACN-SN3D、FDN 等）。
- **属性/模糊测试**：对节点输入（NaN/inf/denormal、极端块长、参数边界）做属性与模糊测试，验证 RT 契约（不 panic、输出有界、denormal 被 flush），发布构建保持 panic-free（错误经 `Result` 上抛而非 unwrap）。

---

## 29. 编译图执行模型（拓扑计划 / 缓冲活跃度分配 / 就地与别名优化）

图在音频线程之外编译成一份**确定性执行计划（ExecPlan）**——借鉴 Web Audio 的图模型与现代渲染图（render graph）的"编译期资源分配"思想，把"图是什么"与"如何跑"彻底分离：

- **拓扑排序计划**：编译期做 DAG 拓扑排序，产出定序的 `Vec<PlanStep>`（每步 = 节点索引 + 输入缓冲槽 + 输出缓冲槽 + 增益/求和指令）。RT 线程只顺序执行这份扁平计划，无递归、无 `HashMap` 查找、无虚分发以外的间接。
- **缓冲活跃度分配（Buffer Liveness Allocation）**：把中间缓冲当作"寄存器"，用**活跃区间分析 + 图着色**为每条边分配缓冲槽，生命周期不重叠的边复用同一物理缓冲（类似编译器寄存器分配 / 渲染图的 transient 资源别名）。相较"每条边一块缓冲"，峰值内存与 cache 足迹显著下降；分配在编译期完成，RT 期零分配。
- **就地处理（In-place）与别名优化**：当某节点的某输出端仅被单一消费者读取、且节点声明 `can_process_in_place()`，编译器让其输入/输出复用同一缓冲槽，省去一次 copy。别名安全性在编译期校验（禁止把仍被其他步读取的缓冲就地覆写）。
- **求和与发送内联**：同一输入端的多入边求和、以及 Aux 发送的加权累加，被编译成计划内的 `AddScaled` 指令序列（复用 `AudioBuffer::add_scaled`），而非运行时遍历边表。
- **延迟对齐（PDC）**：编译期沿计划累计各节点 `latency_frames()`，对并行路径插入整数样本延迟补偿（Plugin Delay Compensation），保证多路汇合相位对齐——这是母带链与并行发送正确性的前提。
- **热交换**：新计划在任务线程编译完成后经命令环交付，RT 在块边界原子切换计划指针；旧计划连同其缓冲池进 epoch 回收（§21 无锁模型的回收策略），RT 线程从不 drop。

`ExecPlan` 是纯数据（`Copy`/`Send` 的索引与指令），可序列化用于离线 golden 对拍：相同图编译出**逐字节一致**的计划，是确定性验收（§28）的基石。

---

## 30. 并行 DSP 图调度（Job 化渲染 / 确定性并行 / 岛屿划分）

单线程 RT 图在语音密集（数百并发语音 + 多总线 + 卷积混响）场景会成为瓶颈。借鉴 **Unity DSP Graph 的数据导向无 GC 执行**并结合 Prism 已有的 `bevy_tasks` 工作窃取，Resonance 支持**块内并行**的图执行，同时保持确定性：

- **岛屿划分（Island Partitioning）**：编译期把 DAG 按连通性与拓扑层级切成可并行的"岛屿/层"。同一拓扑层、无数据依赖的节点可并发执行；跨层用屏障（barrier）同步。
- **Job 化渲染**：每个可并行节点或子链封装为一个 job，提交到**固定规模的音频 worker 池**（独立于 gameplay 任务池，绑定优先级，避免与渲染/物理抢占）。worker 数按设备核数与 CPU 预算（§32）配置，可降到 1 = 纯串行回退。
- **确定性并行**：并行**不改变数值结果**——求和顺序在编译期固定（计划里的 `AddScaled` 定序），浮点累加按固定次序执行；job 只并行"互不依赖"的节点，不并行"同一求和的多个加数"。因此并行/串行两条路径产出**逐样本一致**，可交叉对拍验证。
- **RT 安全的 job 系统**：worker 不分配、不加锁竞争（每 worker 私有 scratch 缓冲，来自构造期预分配的池），完成计数用原子；主 RT 线程作为协调者提交并等待层屏障，欠时（overrun）时记录并降级（§32）。
- **NUMA/亲和**：缓冲槽分配（§29）尽量让同岛屿的读写落在同一 worker 的私有缓冲，减少跨核共享脏行。
- **回退保证**：若平台不支持多线程音频（wasm 单线程 worklet），编译器输出串行计划，行为等价。

并行是**编译期决策 + RT 期执行**，与 §29 的 ExecPlan 同源：并行计划只是给每个 PlanStep 附加"岛屿 id + 层号"。

---

## 31. GPU 加速几何声学（共享渲染器 BVH 的光线与路径追踪）

这是 Resonance 相对独立音频中间件（Wwise/FMOD/Steam Audio 各自维护声学场景）的**核心次世代差异**：Prism 既是渲染器又是音频引擎，声学传播可**直接复用渲染器的 GPU 场景表示与加速结构**，而非重建一套声学几何。

- **共享加速结构**：遮挡/衍射/反射查询复用 `prism_physics` 与渲染管线已有的 **BVH/meshlet 场景**（同一 `pkg/prism_render_*` 世界表示），声学与视觉/物理**共享同一几何真相**，杜绝"碰撞体与声学体不一致"的经典 bug。
- **GPU 声线/路径追踪（可插拔 `PropagationBackend` 的 GPU 实现）**：
  - **反射**：在 GPU 上从声源发射声线，蒙特卡洛追踪镜面 + 漫散射反射，累积能量到听者，估计**早反射 + 后期混响衰减（RT60/EDC）**，输出给 §14 的反射/混响参数与 FDN/卷积混响的房间响应。
  - **遮挡/障碍**：批量射线（声源→听者、声源→探针）在 GPU 上并行求交，返回直达可见性与穿越材质，驱动 §14 的直达增益/低通与 aux 发送。
  - **衍射**：沿几何边缘的最短路径（GPU 上的路径查询/边缘图），给出绕射角与附加衰减。
- **异步、无 RT 阻塞**：声学 GPU 作业在**渲染帧率节奏**（非音频块节奏）异步派发，结果（衰减/低通/发送量/房间响应系数）经遥测/命令环喂给音频线程，RT 侧只做**平滑插值**（§7）——GPU 延迟对听感透明。
- **烘焙与实时混合**：静态几何离线烘焙探针网格（GPU 加速预计算），运行时对动态遮挡做**少量实时声线**修正，二者插值。烘焙数据经 `bevy_asset` 加载。
- **声学材质协同**：表面从 `prism_material_pipeline` 同一资产读取吸收/散射/透射系数（与视觉 BRDF 并存于一个材质），GPU 追踪时按频带加权。
- **优雅降级**：无合适 GPU 或预算不足时，退回 `prism_physics` CPU 射线后端（§14 默认几何后端），接口不变。

安全边界：GPU 声学**只产出控制参数**（标量/低维系数），不产出音频样本流回 RT，避免 GPU→音频的实时同步风险；样本级 DSP（卷积、FDN、双耳）始终在 CPU 音频线程确定性执行。

---

## 32. 性能自适应治理与音频 LOD（CPU 预算驱动质量缩放）

对标主机/移动"稳帧"诉求，Resonance 内建**运行时质量治理器（QualityGovernor）**，在 CPU 预算内动态缩放质量，避免爆音/欠载——借鉴 Wwise 的语音上限/虚拟语音与游戏引擎的 LOD 思想，但做成**闭环自适应**：

- **CPU 预算闭环**：遥测环（§21）回传每块的渲染耗时占预算比。治理器据此在**块边界**（非 RT 热路径内决策）升/降质量档，滞回（hysteresis）避免抖动。
- **音频 LOD 维度**（按声源优先级 + 距离 + 感知重要度分级）：
  - **过采样档**：波形整形/非线性效果（§9 waveshaper）的抗混叠过采样从 4x→2x→1x 随预算下调。
  - **混响质量**：卷积分块 FFT 尺寸、FDN 延迟线数、早反射条数分级；远处/次要声源用更廉价的混响或共享混响返回。
  - **空间化档**：近处/重要声源用 HRTF 双耳卷积，远处降为廉价 VBAP/立体声声像；HOA 阶数按预算降阶。
  - **调制/自动化速率**：次要声源的调制与自动化从逐样本降为逐块控制率。
  - **语音数与虚拟化**：超预算时提高虚拟化阈值（§25），把更多低感知贡献语音转虚拟。
- **优先级模型**：每语音的有效重要度 = 显式优先级 × 距离衰减 × 感知响度 ×（是否被掩蔽，§33）。治理器优先降级低重要度语音。
- **平台档位**：移动端可叠加全局降采样率/降语音上限的功耗档（§32 呼应移动功耗开放问题）。
- **可观测**：当前档位、降级原因、各维度节省经 Profiler（§26）可视化，便于声音设计师与工程师调优。

治理器**只调参数、不改图拓扑**（拓扑热交换代价高、留给显式重编译），保证平滑无爆音。

---

## 33. 心理声学虚拟化与声源聚类（掩蔽感知剔除 / 对象床限制）

在"数百声源、有限扬声器/对象床/CPU"的次世代场景下，纯"最响的 N 个"淘汰会漏掉感知细节。Resonance 引入**心理声学感知层**（纯经典信号处理，无 ML），把有限资源花在**听得见**的声音上：

- **掩蔽感知虚拟化（Masking-aware Virtualization）**：在响度基础上估计**频域掩蔽**——用简化的临界频带（Bark/ERB 近似）能量比较，判断某语音是否被更响的邻近语音在其频带内掩蔽。被掩蔽且低贡献的语音**优先转虚拟**（§25：继续推进播放位置但不渲染），资源释放给可闻语音。掩蔽阈值随治理器（§32）预算收紧/放宽。
- **HDR 联动**：与 §13 HDR 窗口共享响度估计——落在动态窗口下沿之外的语音直接虚拟化。
- **声源聚类（Source Clustering）**：借鉴对象音频床（Atmos/平台后端）的对象数上限与 Wwise 多位置思想，把空间上邻近、音色相近的多个语音**动态聚类**为少量"代表性虚拟源"：
  - 聚类质心 = 成员按响度加权的空间位置；聚类信号 = 成员下混（能量守恒）。
  - 用于"人群/雨/弹雨/森林"等海量点声源，把 N 个空间化开销降为 K 个（K ≪ N），听感上保持空间分布。
  - 聚类在**块边界**重算（增量维护），成员进出用 §7 平滑淡入淡出避免爆音。
- **对象床预算**：输出到平台对象床（§16）时，若对象数超平台上限，用同一聚类机制归并到床容量内，其余下混到环境声道。
- **确定性**：掩蔽与聚类判定基于确定性能量/位置输入，可回放、可 golden 对拍（判定日志经遥测导出）。

感知层**只决定"谁渲染/如何归并"**，被判定的语音状态推进不变，恢复可闻时无缝复活（§25 虚拟语音行为）。

---

## 34. 时间伸缩与变调 / 重采样质量分级（与多普勒解耦）

音高、时值、播放速率与多普勒是四个**可独立控制**的量；朴素引擎把它们耦合在"改变采样读取步进"上，导致变速必变调、多普勒有爆音。Resonance 提供解耦的高质量原语：

- **重采样质量分级（`Resampler` trait）**：
  - 设备/资产采样率转换与任意比率播放走可插拔重采样器，质量分级：**线性**（LOD 低档/远处）→ **多相 FIR / windowed-sinc**（默认）→ **高阶 sinc**（母带/近场）。
  - 系数构造期预计算，RT 期零分配；比率随多普勒/变速连续变化时相位连续（无爆音）。
- **变调不变速 / 变速不变调（`TimeStretcher` trait）**：
  - 提供 **WSOLA/SOLA**（低开销、语音/音效友好）与**相位声码器**（音乐、大幅拉伸更平滑）两档实现，供交互音乐（§19）做节拍匹配、音效做音高随机化而不改时长。
  - 与 §11 Patch 的采样器原语集成：采样器可指定"保持音高的变速"或"保持时长的变调"。
- **多普勒解耦**：§15 的 `DopplerNode` 用**分数延迟线**（连续变长）实现物理多普勒频移，与"内容侧变调/变速"分开——多普勒是传播效应，变调是创作意图，二者可叠加而互不污染。
- **确定性**：所有重采样/伸缩系数与相位状态由 `bevy_math::ops`/`libm` 确定性计算，构造期预分配，逐样本可回放。

三条原语（重采样 / 时间伸缩 / 多普勒延迟线）共享"分数延迟 + 插值"内核，代码复用且各有独立质量档，随治理器（§32）自适应。

---

## 35. 对白与本地化（Dialogue / Localization / 字幕 / 口型）

规划 crate `prism_audio_authoring`（对齐 FMOD Programmer Instrument / Wwise External Sources + Dialogue Event / 通用本地化管线）：

- **程序化对白（运行时选择媒体）**：对白不预烘进逻辑，而由**对白解析器**在运行时按键值（角色 / 情绪 / 语言 / 变体 id）选择媒体后注入语音（对齐 FMOD Programmer Sound 与 Wwise External Sources）。游戏只发"说这句台词"的语义 Event，解析→取流→播放全在链路下游完成。
- **对白决策树（Dialogue Event）**：Wwise 式——按一组 State/Switch（谁在说、在哪、什么心情）沿决策树命中具体台词或随机变体，支持"通用回退"路径，避免缺变体时静默。
- **语言 Bank 热切换**：本地化媒体按语言分包（`voice_en`/`voice_zh`/…），切语言只换语音 Bank，逻辑与时间线不变；未加载语言回退默认并遥测告警。
- **字幕/说明文字同步（Caption Sync）**：媒体携带时间码字幕轨（或外部字幕资产），播放头驱动字幕事件经遥测环（§21）上抛给 UI，支持逐句/逐词高亮与无障碍全字幕（联动 §23）。
- **口型/表情驱动（Viseme / Lip-sync）**：viseme/音素时间线与能量包络随资产附带或离线分析导出，运行时经遥测环喂给动画系统，**不占 RT 预算**；无预烘数据时用 §26 `SpectrumNode`/包络跟随做粗略张口降级。
- **对白优先级与闪避**：对白总线作为 §13 HDR/ducking 的 sidechain 键，说话自动压低音乐/环境；对白语音高优先级，§25 虚拟化最后淘汰。
- **RT 边界**：解析/查表/取流在任务线程完成，RT 只播已就绪语音；缺失媒体输出静音并告警，不阻塞、不 panic。

---

## 36. 触感与跨模态输出（Haptics / Motion / 手柄反馈）

规划（归入 `prism_audio_device` 输出适配层，对齐 Wwise Motion / PS5 DualSense·Tempest Haptics / 通用手柄 rumble）：

- **音频同源触感**：把音频信号（或其低频/包络）转成触感波形——声音与触感**同源**，天然逐样本同步（对齐 Wwise Motion 把声音渲染到"运动设备"），免去另做一套振动曲线。
- **触感总线（Haptic Bus）**：统一图（§5）里一条并行输出总线，声源可像 Aux 发送（§17）一样按增益发往触感总线；总线走独立带通/整流/包络链后交 `HapticBackend`。
- **`HapticBackend`（可插拔 trait）**：
  - **宽频高保真**（DualSense / Tempest 式）：直接吃触感波形（低采样率重采样），表达细腻纹理。
  - **双马达 rumble**（通用手柄）：信号分低/高频包络分别驱动左右（低/高频）马达。
  - **无设备**：静默回退。
- **跨模态对齐**：触感与声音共享 §8 播放头与 §29 PDC，保证"看到—听到—摸到"同一样本时刻对齐；设备固有延迟由后端上报并补偿。
- **空间触感**：按声源方向/距离（§15）加权左右强度，做方向性冲击（左侧爆炸→左马达更强）。
- **预算与降级**：触感受 §32 治理器预算约束，低档旁路；触感生成**纯旁路**，不回灌音频路径。

---

## 37. 程序化环境音景（Soundscape / 程序化 Ambience）

规划（对齐 UE5 Soundscape / 通用程序化环境系统）：

- **音景状态（Soundscape State）**：由环境上下文（生物群系 / 天气 / 时段 / 室内外，来自 gameplay 与 §17 Reverb Zone）激活一组**调色板（Palette）**。
- **调色板 / 元素（Palette / Color Point）**：每个元素定义一个环境声（鸟鸣/风/滴水/远处交通）及其**播放规则**——触发概率、间隔分布、随机音高/增益、空间散布半径、并发上限、昼夜权重。
- **程序化调度**：调度器复用 §8 时钟 + 种子 RNG，在听者周围**程序化散布 one-shot**，形成永不循环、低重复感的环境床，替代"一段环境 loop 干听"。
- **几何/遮挡联动**：散布点经 §14 遮挡/障碍与 §17 房间归属过滤（室内不放室外鸟鸣），发送量随 §15 距离曲线。
- **确定性**：调度用种子 RNG（§24），同种子 + 同状态序列可复现，便于 golden 对拍与网络一致。
- **预算内自适应**：并发环境元素数受 §32/§33 治理（聚类/掩蔽剔除），远处密集元素聚合为床。

---

## 38. 实时授权与远程工具 API（Live Authoring / WAAPI 式 / 热调）

规划（对齐 Wwise Authoring API (WAAPI) / FMOD Studio Live Update / 通用远程调试）：

- **远程工具通道（`AuthoringTransport`）**：编辑器/外部工具经本地 socket（或进程内通道）连到运行引擎，**只读遥测**（语音清单/总线电平/CPU/事件时间线，来自 §26 遥测环）+ **白名单写命令**（改参数/触发 Event/切 State/换 snapshot），命令经 §21 命令环下发，绝不直接触碰 RT 内存。
- **实时调参（Live Tuning）**：运行时调 RTPC/总线增益/衰减曲线/混响参数并**即时听到**（对齐 FMOD Live Update），满意后回写授权资产；改动经 §7 平滑，无爆音。
- **授权数据热重载**：Event/Container/Bank/Patch 经 `bevy_asset` 热重载——任务线程重编译受影响子图/Patch 成新 `ExecPlan`（§29），RT 块边界原子热交换（§30），无需重启。
- **远程 Profiler 连接**：Profiler 面板（§26）可连本地或远端设备（主机/移动真机）会话，录制/回放；能力探测决定带宽与采样率。
- **安全边界**：远程通道鉴权 + 命令白名单，默认仅开发构建启用，发布构建整体编译剔除，收敛攻击面。
- **契约**：远程写与游戏代码走同一命令环，故**远程改动与代码改动语义一致、可确定性回放**。

---

## 39. Crate 拆分与落地形态

| Crate | 层 | 内容 | 状态 |
|---|---|---|---|
| `pkg/prism_audio_core` | L1+L2 | math/buffer/param/time/graph + nodes | ✅ 基础层已落地 |
| `pkg/prism_audio_spatial` | L3 | 几何传播/HRTF/Ambisonics/panner/多普勒/平台后端桥 | 规划 |
| `pkg/prism_audio_authoring` | L3 | Event/Container/State/Switch/RTPC/Patch 编译/Modulation/交互音乐/Bank/对白与本地化/音景 | 规划 |
| `pkg/prism_audio_device` | L3 | cpal/worklet/FileSink/输入捕获/触感后端/远程授权通道 | 规划 |
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
- dialogue（对白解析/本地化/语言 Bank/字幕/viseme）
- haptics（触感总线/`HapticBackend`/双马达/宽频）
- soundscape（程序化音景调色板与散布调度）
- tooling（远程授权 API/live update/远程 profiler）

---

## 40. 路线图

- **M0 内核（已完成）**：math/buffer/param/time/graph + 首发 4 节点，31 测试（27 单测 + 4 doctest），双构建，零告警，已 commit。
- **M1 效果与动态**：EQ/delay/waveshaper/调制延迟 + compressor/limiter/gate/ducking + reverb（FDN/convolver）。
- **M2 声源与调度**：sample player/oscillator/noise/streaming/generator + 采样精确调度器 + 命名时钟 + 语音池与虚拟语音行为。
- **M3 ECS 桥与设备**：命令/遥测环 + epoch 回收 + cpal/worklet/FileSink/输入捕获 + `bevy_audio` 前端改造（兼容 API）。
- **M4 空间**：遮挡/障碍/透射/衍射/反射 + 距离塑形（衰减/锥形/spread/focus/doppler/多位置）+ HRTF/Ambisonics + Rooms&Portals + Aux 发送 + 平台空间后端桥。
- **M5 编排、Patch 与音乐**：Event/Container/State/Switch/RTPC + Patch 编译器与合成原语 + Modulation（LFO/包络/控制总线）+ 交互音乐（段/过渡/stinger）+ Bank/流式 + 对白与本地化解析（§35）+ 程序化音景（§37）。
- **M6 母带、合规、剖析与工具**：LUFS 归一 + true-peak limiter + HDR 窗口 + snapshot + 无障碍 + Profiler/频谱/计量面板 + 触感与跨模态输出（§36）+ 实时授权与远程工具 API（§38）。
- **M7 次世代执行与声学**（横切增强，随 M1-M6 演进落地）：编译图 ExecPlan + 缓冲活跃度分配 + PDC（§29）；岛屿划分与 Job 化确定性并行调度（§30）；复用渲染器 GPU BVH 的声学声线/路径后端与烘焙（§31）；CPU 预算闭环治理器与音频 LOD（§32）；掩蔽感知虚拟化与声源聚类（§33）；`Resampler`/`TimeStretcher` 质量分级（§34）。每项均带确定性/golden 对拍验收。

---

## 41. 关键扩展点清单

| 扩展点 | trait | 用途 |
|---|---|---|
| 处理单元 | `AudioNode` | 任意 DSP/总线/空间化 |
| 内容子图 | `PatchNode`（编译产物） | 程序化合成声音 |
| 运行时构图 | `PatchBuilder` | 任务线程增量拼装/改写 Patch 后热切换 |
| 传播后端 | `PropagationBackend` | 几何/波动/第三方空间化 |
| 声像/空间化 | `Panner` | VBAP/HRTF/Ambisonic/平台 SDK |
| 调制器 | `Modulator` | LFO/包络/曲线/自定义调制 |
| 解码器 | `SourceDecoder` | wav/ogg/flac/自定义 |
| 设备后端 | `DeviceBackend` | cpal/worklet/离线/捕获 |
| 参数源 | 控制总线/RTPC 映射 | 参数联动与调制 |
| 图调度器 | `GraphScheduler` | 串行/Job 化并行图执行策略 |
| 重采样器 | `Resampler` | 线性/多相 sinc/高阶 sinc 质量分级 |
| 时间伸缩 | `TimeStretcher` | WSOLA/相位声码器（变调不变速） |
| 质量治理 | `QualityGovernor` | CPU 预算闭环的音频 LOD 策略 |
| 对白解析 | `DialogueResolver` | 运行时按键值/语言/决策树选媒体 |
| 触感后端 | `HapticBackend` | 宽频（DualSense）/双马达/运动设备 |
| 音景调色板 | `SoundscapePalette` | 程序化环境散布规则与调度 |
| 授权通道 | `AuthoringTransport` | 远程只读遥测 + 白名单写命令 |

---

## 42. 开放问题

- 图/资源交换的旧对象回收策略：延迟队列 vs 引用计数 vs epoch（当前倾向 epoch）。
- HOA 阶数与 CPU 预算的默认档位。
- 卷积混响的分块 FFT 大小与延迟/CPU 折中。
- 移动端功耗档：降采样率/降语音数的自适应策略。
- 声学材质与视觉材质资产的字段合并范围。
- Patch 嵌套深度上限与编译展开的内存上界。
- 平台空间后端的能力探测与优雅降级策略。
- 触感设备能力差异（宽频 vs 双马达）的统一波形抽象与降级映射。
- 对白媒体运行时选择的查表命中与流式预取延迟预算。
- 远程授权通道在发布构建的启用/鉴权策略与命令白名单粒度。
- 程序化音景元素密度与 §33 聚类阈值的默认档位。
- 运行时 `PatchBuilder` 增量构图的编译预算与热切换频率上限（防任务线程构图风暴与 epoch 回收积压）。
- 片段式交互流与段/播放列表两种交互音乐模型的统一数据表示与作者取舍。
- 延迟补偿播放头查询在不同设备后端（cpal/worklet）下 output_latency 的可得性与估计精度。

---

## 43. 术语表

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
- **ExecPlan / 执行计划**：编译期从图产出的扁平、定序、零分配的处理指令序列，RT 顺序执行。
- **缓冲活跃度分配**：用活跃区间分析 + 图着色为图的中间缓冲复用物理槽位（类比寄存器分配）。
- **就地处理（In-place）**：节点输入输出复用同一缓冲以省 copy，别名安全由编译期校验。
- **PDC（Plugin Delay Compensation）**：对并行路径插入整数样本延迟以对齐相位。
- **岛屿（Island）**：图中可并发执行的、无相互数据依赖的节点子集。
- **音频 LOD**：按重要度对声源缩放过采样/混响/空间化/控制率的分级质量策略。
- **QualityGovernor**：按 CPU 预算闭环调整音频 LOD 与语音数的运行时治理器。
- **掩蔽（Masking）**：频域上响声掩盖弱声的心理声学现象，用于感知剔除。
- **声源聚类（Source Clustering）**：把邻近相近的多个声源动态归并为少量代表性虚拟源。
- **WSOLA / 相位声码器**：变速不变调 / 变调不变速的时域/频域时间伸缩算法。
- **多相 FIR（Polyphase）**：高效任意比率重采样的滤波器组结构。
- **Programmer Sound / 程序化对白**：运行时按键值选择媒体注入语音的对白机制（FMOD/Wwise 式）。
- **Viseme / 口型**：与音素对应的口型帧，用于唇形/表情动画驱动。
- **Caption Sync / 字幕同步**：由播放头驱动、随媒体时间码上抛的字幕事件。
- **Haptics / 触感**：与音频同源生成的振动/力反馈输出（手柄/运动设备）。
- **Soundscape / 音景**：由环境状态驱动、程序化散布 one-shot 形成的低重复环境声床。
- **WAAPI / 授权 API**：远程连接运行引擎做只读遥测与白名单写命令的工具协议范式。
- **Builder API / 运行时构图**：在任务线程增量拼装或改写内容 DSP 图（Patch），编译后原子热切换到运行时图（UE5 MetaSound Builder API 式）。
- **Audio Gameplay Volume**：体积驱动的室内外/混响/衰减覆盖与门户连通模型（UE5 5.3+，取代旧混响体）。
- **片段式交互流（Interactive Stream）**：以剪辑为节点、以带过渡类型（next-beat/next-bar/marker 等）的转移为边的交互音乐图（Godot AudioStreamInteractive 式）。
- **延迟补偿播放头**：向 gameplay 暴露的、扣除输出延迟并加回自上次混音以来时间的“画面对齐”播放位置。
- **Polyphonic 声源**：单逻辑声源动态复用池内多条 voice 播放重叠实例（Godot AudioStreamPolyphonic 式）。
