# Prism Animation 次世代动画引擎设计方案

> 面向 Prism（Bevy fork）的数据导向、统一求值管线、声明式与程序化融合的 AAA 级动画引擎设计。
> 借鉴 UE5（AnimGraph / Control Rig / AnimNext / Pose Search / Motion Warping）、Unity（Mecanim / Playables / DOTS Animation + Burst）、Godot（AnimationTree）、ozz-animation（高性能运行时标杆）、ACL（Animation Compression Library）、Gears/Inertialization（惯性化混合）取长补短。
> 本文档为设计规格，纯经典数值动画路线，不含任何 AI/ML 推理内容（Motion Matching 为经典最近邻检索，非神经网络）。

- 版本: v0.1（设计阶段，未进入编码）
- 适用引擎: Prism / Bevy ECS 生态
- 关键依赖: bevy_ecs（并行 ECS）、bevy_math（glam SIMD）、bevy_tasks（任务系统）、bevy_asset（资产/热重载）、bevy_transform、bevy_render（蒙皮）

---

## 目录
1. 设计哲学
2. 分层架构
3. 数据模型（L0）
4. 求值内核（L1）：两相 + 指令流 + SoA + SIMD
5. 过渡系统：惯性化混合
6. 图与节点（L2）：声明式 base 段
7. 程序化修正（L2）：Control-Rig 式 post 段
8. 同步、相位与根运动（脚滑根治）
9. Motion Matching（独立可选）
10. 高层 API 与参数黑板（L3）
11. 重定向 Retargeting
12. 效果与画面质量专题
13. 性能工程：预算 / URO / 内存 / 并行
14. 网络确定性与回放
15. 调试与可观测性
16. 授权与工具链（L4）
17. Bevy 集成层 API
18. 质量与可信度基础设施
19. 能力全景（分级）
20. Crate 拆分与落地形态
21. 兼容与迁移
22. 路线图
23. 风险与取舍
24. 关键扩展点清单
25. 术语表

---

## 1. 设计哲学

对标商用旗舰引擎，取长补短，确立五条铁律：

- **数据导向 (DOD)**：骨架共享只读，姿势以 SoA（Structure-of-Arrays）连续存放，天然对齐 Bevy 并行调度与 glam SIMD。借鉴 ozz-animation 的 SoA + Job 管线与 Unity DOTS Animation。
- **统一求值管线**：声明式混合（AnimGraph 式）与程序化修正（Control Rig 式）不是两个并行引擎，而是**同一条指令流里的 base 段 + post 段**。这是相对"两套系统各自为政"的结构性差异化，也是环境贴合（脚贴地、精准落点、看向目标）的前提。
- **两相分离**：Update 相（游戏线程，读游戏状态、算权重与计划）与 Evaluate 相（工作线程，纯数学求值）彻底解耦。Evaluate 相为纯函数 → 天然并行、可确定性、可回放。借鉴 UE 的 Fast Path / Thread-safe Update。
- **效果即默认**：惯性化过渡、相位同步、脚部贴地等"看起来对"的技术作为默认路径而非高级选项；新手零配置即获得 AAA 观感。
- **易用性即 API**：参数黑板解耦逻辑与动画，三层 API 从"一行播放"到"节点级下钻"逐层开放。

### 设计取舍总表

| 维度 | Prism 选择 | 理由 |
|---|---|---|
| 姿势存储 | SoA + PoseArena 池 | 并行/SIMD 天然对齐，热路径零堆分配 |
| 求值模型 | 图编译为线性指令流 | 消除树遍历与虚表分派，cache 友好 |
| 并行 | 两相分离 + bevy_tasks | 跨角色并行、Evaluate 纯函数、确定性可控 |
| 过渡 | 惯性化为默认，crossfade 兜底 | 过渡期只求值一个动画，位置/速度连续、无穿插 |
| 压缩 | ACL 式量化 + 常量折叠 | 内存降一个数量级，带宽/cache 优化 |
| 采样 | 关键帧游标缓存 | 单调时间下摊销 O(1) 采样 |
| 声明式/程序化 | 单管线 base+post 段 | 融合而非并列，环境贴合能力的结构前提 |
| 同步 | Sync Group + Sync Marker | 根治走/跑混合脚滑 |
| 联机 | 两相纯函数 + 定点时间步 | 回滚网络/回放可选 |
| API | 默认零配置，分层下钻 | 对齐 Bevy modular/易用 |
| 重特性 | Motion Matching 独立 crate | 非默认，控制内存与构建体积 |

---

## 2. 分层架构

```
L4 授权层   状态机数据·glTF导入·热重载·Reflect Inspector·Rewind调试器·Anim Insights
L3 高层API  Animator(控制器) · AnimationParams(参数黑板) · 一行式兼容门面
L2 图节点   [base] StateMachine · BlendSpace(1D/2D) · Layer(BoneMask) · Additive · Mirror · SyncGroup
            [post] TwoBoneIK · FABRIK · LookAt · FootPlacement · RootMotion · MotionWarp · SpringBone
L1 求值内核  两相(Update/Evaluate) · 指令流编译 · SoA PoseArena · SIMD(4-wide)
            · Inertialization 过渡 · 并行调度 · URO/预算 · 零权重剪枝
L0 数据基元  Skeleton(共享只读,SoA) · CompressedClip(ACL式) · SamplingContext(游标)
            · BoneMask · RefPose · RetargetMap · SyncMarker
```

### 每帧核心数据流

```
游戏逻辑 → 写 AnimationParams(黑板)
  │
  ├─ Update 相 [游戏线程]
  │    状态机转移 · 时间推进 · 混合空间权重 · 相位同步 · 零权重剪枝
  │    产出纯数据 EvalPlan（指令流 + 采样输入），本帧不再触碰 ECS
  │
  └─ Evaluate 相 [ComputeTaskPool 并行]
       Sample(游标缓存+解压) → Blend/Masked/Additive(SIMD)
         → Inertialize(惯性化叠加) → Post 修正(IK/LookAt/...)
           → 最终 LocalPose → Local→Model(SIMD) → 蒙皮矩阵 → 渲染/GPU skinning
```

---

## 3. 数据模型（L0）

```rust
// 共享只读 asset：一个骨架供多角色复用
struct Skeleton {
    joint_names: Vec<Name>,
    parents: Vec<i16>,                 // 拓扑序，保证 parent 索引 < child
    bind_pose_local: SoaTransforms,    // SoA，4 关节一组打包，直接喂 SIMD
    joint_count: u16,
    mirror: Option<Vec<u16>>,          // 左右镜像映射
    retarget: Option<RetargetMap>,     // 通用骨架(humanoid)映射
}

// 编辑期表示：可读写、可被导入与压缩器消费
struct AnimationClip {
    curves: AnimationCurves,
    events: Vec<(f32, AnimationEvent)>,
    duration: f32,
    sync_markers: Vec<SyncMarker>,     // 相位标记（左脚落地/右脚落地）
}

// 运行期表示：构建期从 AnimationClip 烘焙，量化压缩
struct CompressedClip {
    tracks: CompressedTracks,          // 量化 + 常量轨折叠 + 分段
    joint_bindings: Vec<u16>,          // 轨道 → 骨架关节索引
    duration: f32,
    sync_markers: Vec<SyncMarker>,
}

// 每"播放实例"持有：关键帧游标 + 解压缓存 → 摊销 O(1) 采样
struct SamplingContext {
    cursors: Vec<u32>,                 // 每轨上次命中的关键帧
    cache: DecompressCache,
}

struct BoneMask { weights: Vec<f32> }  // per 关节 0..1，分层混合用
struct RefPose(SoaTransforms);         // 加法动画的参考姿势
struct SyncMarker { name: SmolStr, phase: f32 } // 归一化相位 0..1
struct RetargetMap { src_to_dst: Vec<i16>, bind_delta: SoaTransforms }
```

**设计要点**
- **骨架/动画解耦（ozz）**：clip 按关节索引绑定，一份动画可复用于任意兼容骨架，是重定向的结构基础。
- **压缩（ACL/UE）**：量化 + 范围压缩 + 静止轨折叠为常量；AAA 项目动画数据以 GB 计，压缩是刚需。编辑期/运行期表示分离，构建期一次性烘焙。
- **游标缓存（ozz SamplingJob::Context）**：动画时间通常单调推进，游标让采样从上次位置继续，摊销 O(1)，避免每帧 O(log n) 二分。

---

## 4. 求值内核（L1）：两相 + 指令流 + SoA + SIMD

### 4.1 两相模型（UE Fast Path）

```rust
// Update 相在游戏线程产出的纯数据计划；Evaluate 相不再触碰 ECS/游戏状态
struct EvalPlan {
    instrs: Vec<EvalInstr>,     // 编译后的线性指令流
    inputs: Vec<ClipInput>,     // clip 句柄 + 采样时间 + context 索引
    snapshots: Vec<SnapshotId>, // 惯性化快照引用
}

enum EvalInstr {
    Sample      { input: u32, out: PoseSlot },
    Blend       { a: PoseSlot, b: PoseSlot, w: f32, out: PoseSlot },
    Additive    { base: PoseSlot, add: PoseSlot, w: f32, out: PoseSlot },
    MaskedBlend { a: PoseSlot, b: PoseSlot, mask: MaskId, out: PoseSlot },
    Inertialize { target: PoseSlot, snapshot: SnapshotId, t: f32, out: PoseSlot },
    Post(PostOp), // IK / LookAt / FootPlacement / ...，均带 alpha
}
```

- **Update 相（游戏线程）**：跑状态机转移、推进时间、算混合空间与分层权重、相位同步、零权重剪枝，产出 `EvalPlan`。可自由读游戏状态。
- **Evaluate 相（并行）**：纯函数执行 `EvalPlan`，不碰 ECS/游戏状态 → 可跨角色并行、可确定性化、可回放。

### 4.2 内存与并行

- **PoseArena**：每帧从池分配 pose slot，SoA 连续；跨帧复用，热路径零堆分配。
- **并行粒度**：跨角色一角色一任务（`ComputeTaskPool`）；单角色内采样按轨道分块。
- **零权重剪枝**：`w < ε` 的子树在编译期标记、运行期 early-out，不浪费采样。

### 4.3 SIMD（借鉴 ozz Job 管线）

- SoA 下 4-wide 批量 `slerp/lerp`（四元数混合最近邻取号避免长弧）。
- Local→Model 阶段批量矩阵拼接（对应 ozz `LocalToModelJob`），直接产出蒙皮矩阵。

---

## 5. 过渡系统：惯性化混合（次时代默认）

传统 crossfade 的两大问题：过渡期需**同时求值两套动画**（成本翻倍），且混合中间姿势易穿插/失真。

**惯性化（Inertialization，Gears of War 提出，UE5 内置，现代标配）**：

```rust
struct InertializationState {
    offsets: SoaTransforms,  // 转移瞬间：旧 pose - 新 pose 的每关节差值
    velocities: SoaVel,      // 差值的一阶导（角速度/线速度），保证速度连续
    duration: f32,
    elapsed: f32,
}
```

- 转移瞬间，记录"旧 pose → 新 pose"的差值与一阶导。
- 之后**只求值新动画**，用五次多项式把差值平滑衰减到 0。
- 优点：过渡期**只跑一个动画**（省约一半求值）、位置与速度均连续、无穿插、对任意过渡组合通用。
- `BlendKind::Inertialize` 为状态机转移默认；`CrossFade` 作为需要两路同时呈现时的兜底。

---

## 6. 图与节点（L2）：声明式 base 段

### 6.1 节点分类与约束

- **base 段**（产出/混合基础 pose，无外部依赖、高度并行）：
  `Clip · StateMachine · BlendSpace · Layer(BoneMask) · Additive · Mirror · SyncGroup`
- **post 段**（在 base pose 之后修正，带 alpha，可读空间/物理）：见第 7 节。
- **约束**（编译期校验并告警）：
  1. post 段严格在 base 段之后执行；
  2. 每个 post 节点必须带 `alpha` 与淡入淡出；
  3. 分层遮罩关节集与 post 接管关节集不得冲突。

### 6.2 可扩展节点 trait

```rust
trait AnimNode: Reflect {
    fn update(&mut self, ctx: &mut UpdateCtx);                 // 游戏线程：算权重/时间/转移
    fn compile(&self, out: &mut EvalPlanBuilder) -> PoseSlot;  // 产出指令，返回输出槽
    fn phase(&self) -> NodePhase;                              // Base | Post
}
```

### 6.3 状态机（Mecanim / UE 式）

```rust
struct StateMachine {
    states: Vec<State>,            // 每个 State 内嵌子图
    transitions: Vec<Transition>,
    any_state: Vec<Transition>,    // Unity AnyState：任意态可触发
}
struct Transition {
    from: StateId, to: StateId,
    cond: Condition,               // float > / <、bool、trigger、归一化时间
    blend: BlendKind,              // Inertialize(默认) | CrossFade
    duration: f32, curve: EaseCurve,
    interruptible: bool,           // 可打断
}
```

### 6.4 混合空间（Unity Blend Tree / UE BlendSpace）

- **1D**：按参数在相邻样本间线性插值。
- **2D**：Delaunay 三角剖分 + 重心坐标权重（如 speed × direction 的 8 向移动）。
- 与 **SyncGroup** 联动：所有样本按归一化相位对齐重采样，避免脚滑。

### 6.5 分层、加法、镜像

- **Layer + BoneMask**：override / additive 两种叠加；per 关节遮罩（上半身挥手叠加下半身跑步）。对标 Unity Layers / UE layered blend per bone。
- **Additive**：`clip - RefPose` 叠加到基础 pose（呼吸、瞄准偏移、受击抖动）。
- **Mirror**：用骨架左右映射表，一份动画镜像复用，省一半资源。

---

## 7. 程序化修正（L2）：Control-Rig 式 post 段

承接"声明式与程序化融合"的核心结论：程序化不是第二套引擎，而是同一管线中、base pose 之后的修正段。全部带 alpha 与淡入淡出。

| 节点 | 作用 | 借鉴 |
|---|---|---|
| TwoBoneIK | 手/脚两骨 IK，解析解 | UE Two-Bone IK |
| FABRIK | 多骨链 IK，迭代 | UE/通用 |
| LookAt | 头/眼看向目标，带角度约束 | UE Look-At |
| FootPlacement | 脚贴斜坡/台阶，射线探地 + IK + 盆骨下压 | UE Foot IK |
| RootMotion | 从根骨提取位移/旋转，驱动 CharacterController | UE/Unity Root Motion |
| MotionWarp | 按目标弯曲根运动（精准落点/攀爬/交互对齐） | UE Motion Warping |
| SpringBone | 二次运动（头发/饰品/软组织）弹簧阻尼 | 通用 Secondary Motion |

**数据契约**：post 段共享同一 `Skeleton` + pose buffer；需要空间/物理查询的节点（FootPlacement、LookAt、MotionWarp）在 Update 相收集查询结果（地面高度、目标位置），Evaluate 相只做纯数学，保持并行与确定性。

---

## 8. 同步、相位与根运动（脚滑根治）

脚滑是 locomotion"看起来廉价"的头号元凶，专列一节。

- **Sync Group + Sync Marker**：给循环动画打相位标记（左脚落地 phase=0.0、右脚落地 phase=0.5）。同组内所有动画按归一化相位对齐播放，由主导动画（当前权重最高）驱动相位，混合时按相位重采样。走(1.2s) 与 跑(0.8s) 混合时脚步节拍一致，脚不滑。
- **根运动一致性**：root motion 与相位联动，位移速度与动画步频匹配；`MotionWarp` 在需要精准落点时弯曲轨迹但保持脚步相位。
- **Stride Warping（可选）**：按实际移动速度微调步幅，进一步消除残余滑动。

---

## 9. Motion Matching（独立可选 crate）

- 对标 UE Pose Search / Ubisoft For Honor 的 Motion Matching。
- 构建期从动作库提取**特征向量**：当前 pose 特征（关键关节位置/速度）+ 未来轨迹特征（朝向/位置）。
- 运行期按查询特征做**最近邻检索（KNN）**，选最匹配帧并用惯性化平滑切入；可大幅减少手工状态机。
- **纯经典最近邻，无神经网络**；因数据/内存较大，置于独立 `bevy_motion_matching` crate，非默认构建。

---

## 10. 高层 API 与参数黑板（L3）

```rust
// 逻辑侧：只写参数，彻底解耦（Mecanim / UE 变量模式）
params.set_float("speed", velocity.length());
params.set_bool("grounded", true);
params.trigger("attack");

// 装配：一处配置 base + 分层 + post
Animator::new(skeleton)
    .state_machine(locomotion_asset)             // base
    .layer("upper_body", upper_mask, Additive)   // 分层叠加
    .post(FootPlacement::default())              // post：脚贴地
    .post(LookAt::new(head_joint).target(gaze).alpha(0.7));
```

- **AnimationParams 黑板**：`float / bool / int / trigger`，游戏逻辑只写、图只读。
- **三层 API**：`AnimationPlayer::play(clip)`（一行式，向后兼容）→ `Animator` + 状态机资产（常用）→ 节点/指令流（高手下钻）。
- **向后兼容**：现有 `AnimationPlayer / AnimationGraph` 保留，新状态机内部 lower 到指令流。

---

## 11. 重定向 Retargeting

- 基于关节名映射 + bind pose 差值，把源骨架动画映射到目标骨架。
- 支持 humanoid 通用骨架（类 Unity Avatar）+ 逐关节手动映射覆盖。
- 作用于采样后的 local pose，与压缩/求值解耦；可配比例缩放处理不同体型。

---

## 12. 效果与画面质量专题

AAA 的"效果"大多来自以下默认开启的质量技术：

| 现象/目标 | 技术 | 默认 |
|---|---|---|
| 过渡跳变/脚穿插 | 惯性化混合（位置+速度连续） | ✅ |
| 走/跑混合脚滑 | Sync Group + Sync Marker + Stride Warp | ✅ |
| 脚悬空/穿地 | FootPlacement（射线探地 + IK + 盆骨下压） | 可选默认 |
| 落点/攀爬不对齐 | MotionWarp | 按需 |
| 看向/瞄准僵硬 | LookAt + 加法瞄准偏移 | 按需 |
| LOD 切换跳变 | URO + pose 插值补帧（不 popping） | ✅ |
| 根运动漂移 | 相位联动根运动 + 累积校正 | ✅ |
| 饰品/软组织呆板 | SpringBone 二次运动 | 按需 |
| 对称动作重复做 | Mirror 镜像复用 | ✅ |

质量底线：过渡"连续"（C1 连续，位置与速度均平滑）、locomotion"不滑"、LOD"不跳"。

---

## 13. 性能工程：预算 / URO / 内存 / 并行

- **URO（Update Rate Optimization，UE）**：按屏幕尺寸/距离/可见性降频求值，中间帧用 pose 插值补帧，杜绝 LOD popping。
- **预算调度（Anim Budget）**：每帧给动画一个时间预算，超预算时低优先级角色自动降级（降频 / 跳过 post / 降混合层数）。
- **零权重剪枝**：权重 < ε 的子树直接 early-out。
- **压缩 + 游标 + SoA**：降内存与带宽、摊销采样、SIMD 批处理。
- **并行**：跨角色一任务，单角色内分块；Evaluate 纯函数无锁。
- **GPU skinning 衔接**：Local→Model 产出蒙皮矩阵后交 `bevy_render`，CPU 侧只算骨骼变换。
- **规模目标**：千级可见角色在帧预算内（具体数值待 P1/P7 基准标定）。

---

## 14. 网络确定性与回放

- **确定性路径**：Evaluate 相纯函数、固定 SIMD 代码路径、定点时间步可选 → 同输入多端 bit 级一致，服务回滚网络 / 录像回放。与 f32 高性能路径编译期切换，零运行时代价。
- **快照**：参数黑板 + 状态机当前态 + 相位可序列化，支持回滚重放。

---

## 15. 调试与可观测性

- **Rewind 调试器**（借鉴 UE Rewind Debugger / Animation Insights）：可选录制每帧 `EvalPlan`、关键权重、状态机当前态与转移、相位，供时间轴回放定位"为什么这帧姿势不对"。
- **可视化**：骨架/遮罩/IK 目标/根运动轨迹 gizmo 叠加。
- **统计面板**：每角色求值耗时、活动节点数、压缩命中率、游标命中率、预算降级计数。

---

## 16. 授权与工具链（L4）

- **资产格式**：状态机 / 图 / 混合空间存为带 `version` 字段的 RON（现有 RON 基础），预留扩展位，避免破坏性迁移。
- **热重载**：改图/状态机即时生效（bevy_asset 热重载）。
- **glTF 导入**：自动构建 `Skeleton + CompressedClip + SyncMarker`（扩展现有 `gltf_curves`）。
- **Reflect 驱动 Inspector**：所有节点参数可反射，为未来节点式可视编辑器（类 UE AnimBP）预留数据结构。

---

## 17. Bevy 集成层 API

```rust
app.add_plugins(AnimationPlugin::default());

// 调度阶段
// PostUpdate:  AnimationSet::Update   (游戏线程, 产出 EvalPlan)
//            → AnimationSet::Evaluate (并行, 执行指令流, 写 LocalPose)
//            → AnimationSet::Skinning (Local→Model, 蒙皮矩阵)
```

- 组件：`Animator`、`AnimationParams`、`SkeletonHandle`、`SamplingContext`。
- 资产：`Skeleton`、`CompressedClip`、`StateMachineAsset`、`BlendSpaceAsset`。
- 系统集顺序保证 Update→Evaluate→Skinning 的两相边界与并行安全。

---

## 18. 质量与可信度基础设施

- **正确性单测**：压缩前后采样误差阈值、混合权重归一、sync 相位对齐、惯性化速度连续性。
- **黄金姿势回归**：对关键帧渲染 pose 做快照比对，防回归。
- **基准（benches）**：单角色多层混合、千角色并行、压缩内存占用、游标命中率、惯性化开销；每个 Phase 立基线防性能回退。
- **确定性测试**：同输入多次/多平台求值 bit 级一致。
- **Fuzz**：异常骨架/空 clip/极端参数不 panic。

---

## 19. 能力全景（分级）

| 级别 | 能力 |
|---|---|
| L0 基础 | Clip 采样、曲线、morph、事件/通知、一行式播放 |
| L1 混合 | 权重图、Additive、分层遮罩、Mirror |
| L2 控制 | 状态机、混合空间 1D/2D、参数黑板、惯性化过渡 |
| L3 质量 | Sync Group/Marker、Stride Warp、根运动、URO/预算 |
| L4 程序化 | Two-Bone/FABRIK IK、LookAt、FootPlacement、MotionWarp、SpringBone |
| L5 高级 | 重定向、Rewind 调试、网络确定性 |
| L6 前沿 | Motion Matching（独立 crate） |

---

## 20. Crate 拆分与落地形态

```
bevy_animation            # 主 crate（保名，兼容 re-export）
├── skeleton.rs           # Skeleton / RetargetMap / BoneMask / RefPose
├── clip/                 # AnimationClip + CompressedClip + 压缩器
├── pose.rs               # PoseBuffer(SoA) / PoseArena
├── sampling.rs           # SamplingContext(游标) + 采样
├── eval/                 # 指令流编译 + Evaluate 内核 + SIMD
├── graph/                # 节点 trait + 内置 base/post 节点 + 图资产
├── state_machine.rs
├── blend_space.rs
├── transition.rs         # 惯性化 + crossfade
├── sync.rs               # Sync Group / Marker / Stride Warp
├── params.rs             # 参数黑板
├── animator.rs           # L3 门面
├── retarget.rs
├── lod.rs                # URO / 预算
└── debug.rs              # Rewind 调试数据

bevy_motion_matching      # 独立可选 crate（P8）
bevy_animation_editor     # 编辑器消费数据/反射（可后置）
```

---

## 21. 兼容与迁移

- 新系统与旧 `AnimationPlayer` 并存；`Animator` 为新增门面，不删旧类型。
- Skeleton 显式化改动绑定模型（现为分散 UUID 绑定）→ 提供兼容层桥接旧绑定。
- glTF 导入升级为自动构建新资产，旧路径仍产出 `AnimationClip`。
- 资产 `version` 字段保证序列化前向兼容。
- 大改动先发 RFC / Discussion，分 Phase 合入，对齐上游节奏。

---

## 22. 路线图

| Phase | 内容 | 交付物 | 验收 |
|---|---|---|---|
| P0 | Skeleton(SoA)/RefPose/BoneMask/PoseArena | 数据层 + 单测 | 采样/Local→Model 正确 |
| P1 | 两相内核 + 指令流 + SIMD + SamplingContext | 内核 + benches | 性能基线达标、并行安全 |
| P2 | CompressedClip(ACL式) + 构建烘焙 | 压缩器 | 内存↓一个量级、误差 < 阈值 |
| P3 | 状态机 + 混合空间 + 参数黑板 + 惯性化 | 可用控制器 | idle/walk/run 切换连续无跳变 |
| P4 | SyncGroup + Additive + Mirror + 分层遮罩 | locomotion 套件 | 走/跑混合脚不滑 |
| P5 | post 段：IK/LookAt/FootIK/RootMotion/MotionWarp/SpringBone | 程序化套件 | 脚贴斜坡、精准落点、看向目标 |
| P6 | Animator 门面 + 编辑器数据 + 热重载 | 高层 API | 一处装配、改图即生效 |
| P7 | URO + 预算调度 + Rewind 调试 | 规模化 + 可观测 | 千角色帧预算内、无 LOD popping |
| P8（可选） | Motion Matching | 独立 crate | 无状态机 locomotion demo |

---

## 23. 风险与取舍

| 风险 | 缓解 |
|---|---|
| Skeleton 显式化改动绑定模型 | 兼容层过渡，旧 UUID 绑定桥接 |
| 压缩带来精度损失 | 可配误差阈值，关键动画可关压缩 |
| 惯性化实现复杂度 | 独立可测模块，crossfade 兜底，黄金姿势回归 |
| 两相边界误用（Evaluate 碰游戏状态） | 系统集约束 + API 隔离 + 编译期类型隔离 |
| 遮罩与 post 接管骨骼冲突 | 编译期校验告警 |
| Motion Matching 内存/构建体积 | 独立可选 crate，非默认 |
| 与上游 Bevy 节奏冲突 | 先 RFC，分 Phase，保持兼容 re-export |

---

## 24. 关键扩展点清单

- `AnimNode` trait：自定义 base/post 节点。
- `BlendKind`：自定义过渡策略（除惯性化/crossfade 外）。
- `PostOp`：自定义程序化修正（如自研 IK/物理耦合）。
- `Condition`：自定义状态机转移条件。
- `CompressedTracks`：可替换压缩算法后端。
- 查询注入：FootPlacement/LookAt/MotionWarp 的空间/物理查询源可替换（接 Prism Physics）。
- `bevy_motion_matching`：独立前沿模块挂载点。

---

## 25. 术语表

- **SoA**：Structure-of-Arrays，分量分离的连续数组布局，利于 SIMD/并行。
- **Pose / LocalPose / ModelPose**：姿势；局部空间（相对父关节）/模型空间（相对根）骨骼变换。
- **两相（Update/Evaluate）**：游戏线程算计划 + 工作线程纯数学求值的分离模型。
- **惯性化（Inertialization）**：记录过渡瞬间 pose 差值与速度并平滑衰减的过渡技术，过渡期只求值一个动画。
- **Sync Group / Sync Marker**：按归一化相位对齐多个循环动画，根治脚滑。
- **Additive / RefPose**：加法动画 = clip − 参考姿势，叠加到基础 pose。
- **Retarget**：把源骨架动画映射到目标骨架。
- **URO**：Update Rate Optimization，按距离/可见性降频求值 + 插值补帧。
- **Motion Matching**：按 pose+轨迹特征做最近邻检索选帧的经典技术（无神经网络）。
- **ACL**：Animation Compression Library，业界动画压缩方案参考。
- **ozz-animation**：开源高性能动画运行时，SoA + Job 管线的参考实现。
