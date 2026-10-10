# Prism World 顶级次世代 AAA 级世界系统设计方案

> 面向 Prism（脱离 Bevy 后的独立引擎）的 **世界系统** 设计：**空间分区索引 + i64 定点坐标 + 流送驻留 VM + 三档常驻与按需物化 + HLOD 远景代理 + 事务日志持久化（存档/多人/时间轴）+ 程序化生成与海量散布 + 侵蚀/破坏/生态/水文仿真 + 全局 SDF 世界场 + 行星尺度一体化**。它是旧 `bevy_scene` + 外挂关卡流送的自研替代，服务无缝大世界 / 行星尺度 / 可破坏 / 多人协作。
> **架构翻转（相对本文 v2 旧稿）**：不再把 `WorldDB` 立为与 scene 并列的「平行真相源」。`prism_scene` 是**唯一作者态逻辑真相**；世界系统降级为**架在 scene 之上的「空间组织 + 流送 + 常驻残留 + 持久化」基座**。世界系统的持久数据只有两类：**可由种子/规则确定性再生的派生缓存**（空间索引、HLOD 代理、SDF、导航、散布 proxy 列、烘焙 GI）与**override delta 事务日志**（后者即 `prism_scene` §15 的持久化语义）。没有第二份真相。
> 借形态不抄码。借鉴：
> - **空间分区与流送粒度**：Unreal（World Partition 隐式网格 / Data Layers / HLOD / OFPA 每对象文件化 / Level Instance）
> - **行星尺度与定点坐标**：Star Citizen / Outerra / MSFS（cube-sphere 四叉树 / 64 位定点消抖 / 分区消除浮点误差）
> - **GPU 驱动流送与程序化**：Decima（Horizon）、RDR2/RAGE（tile 流送 / GPU 植被 / 无缝 LOD / 远景烘焙 / 路网河流样条）
> - **程序化生成**：Houdini / UE PCG / HDA（节点图 + 点云散布 + 确定性烘焙 + 增量重算）
> - **可破坏与体素**：No Man's Sky / Teardown / Deep Rock / Chaos Geometry Collection（体素/SDF 布尔、Voronoi 预断裂、碎片持久化）
> - **协作与存档**：Git / CRDT / 事件溯源（append-only 事务日志 / 冲突合并 / 确定性回放 / 时间轴）
> - **数据导向存储与流式 IO**：列存 SoA（Arrow/Parquet 形态，列块 mmap 零解析、内容寻址去重）、DirectStorage / RTX IO / io_uring（GPU 直传解压、绕 CPU 拷贝）
> - **全局距离场**：UE Global Distance Field / Lumen SDF（稀疏 brick + mip 环，一场多用）
> 本文为纯经典数据结构 + 空间算法 + 流送调度路线（Morton/Hilbert、四叉/八叉树、CDLOD/clipmap、Transvoxel/Dual Contouring/SDF、domain warp 噪声、水力/热力侵蚀 PDE、泊松盘/蓝噪声、D8 流向、并查集、CRDT），**不含任何 AI/ML 内容**，不含任何 Unreal/Unity/其他引擎源码或衍生代码。

- 版本： v3.2（在 v3.1 基础上重构：网络框架整体剥离至独立文档 `prism_network`——原 §12「网络复制 / 权威 / AOI」正文改为「网络接缝」，只定义 world 向网络暴露的数据与方向；原 §25.3「服务器网格分片」移交 `prism_network`，§25 其余子节顺延重编号；与仓库「一子系统一文档」惯例对齐。去 Bevy、多 crate、架在 `prism_scene` 契约之上。设计阶段，未进入编码，无旧 API 需保留）
- 适用引擎： Prism（后 Bevy 时代，独立运行时）
- 关键依赖： `prism_scene`（作者内容单元 / spawn·despawn·materialize API / override delta / PersistentId，见 `prism_scene_design_zh.md`）、`prism_asset`（`StableGuid`/软引用/依赖闭包/单文件包/mmap/内容寻址）、`prism_ecs`（物化边界的批量预留/列写入）、`prism_transform`（层级）、`prism_tasks`（并行流送/烘焙）、`prism_math`（定点/空间编码/噪声）、`prism_diagnostic`（计数器/trace）；向上接 `prism_render`/`prism_physics`/`prism_gi`（经 GPU 常驻实例流 + ECS 物化两个边界，均已去 Bevy）
- 层级定位： 基座层 L6；下接 `prism_scene`/`prism_asset`/`prism_ecs`，上驱动渲染/物理/GI；与 `prism_terrain_world_design_zh.md`（地形多表示细节）互为上下位
- 明确约束： 核心 `no_std + alloc`；`std`/`async_io`/`gpu_stream`/`persist`/`pcg`/`sim`/`sdf`/`planet`/`vgeo`/`vtex`/`bake_farm` 为 feature（网络相关 feature `network`/`server_mesh` 移至 `prism_network`）；工作区 `forbid(unsafe_code)`；**不依赖任何 `bevy_*` crate**；GPU/IO 吞吐/显存/带宽均为 design target（本机无 GPU，非实测，如实标注），可验证部分限 CPU 纯函数
- 架构立场： **scene 是单一逻辑真相，世界系统是其上的流送/常驻/持久化基座，不另立平行真相源**；世界持久数据 = 确定性可再生的派生缓存 + override delta 事务日志

---

## 目录

1. 设计哲学与架构翻转
2. 参考产品取舍
3. 档位化（capability / quality tier / feature）
4. 多 crate 分层架构
5. 空间模型与坐标（sector / Morton-Hilbert / i64 定点 / cube-sphere）
6. cell ⇄ scene 映射（世界是 scene 的空间组织层，非内容真相）
7. prism_world_stream：流送 VM 与残留引擎
8. 异步 IO 与 GPU 直传
9. prism_world_residency：常驻档位与按需物化
10. 簇 LOD 统一模型与 HLOD 远景代理
11. prism_world_persist：事务日志 / CRDT / 时间轴 / 存档
12. 网络接缝（复制 / 权威 / AOI 归 `prism_network`）
13. 地形多表示层
14. 程序化生成 PCG
15. 海量散布与植被
16. 仿真：侵蚀 / 破坏 / 生态 / 水文
17. 全局 SDF 世界场
18. 四维世界：时间轴与回放
19. 行星尺度一体化
20. 渲染 / 物理 / GI 接入边界（去 Bevy）
21. 确定性与可复现
22. 性能工程
23. 易用性与作者工作流
24. crate 分层与模块布局
25. 次世代高级功能增补（虚拟几何 / 遮挡剔除 / 虚拟纹理 / 烘焙农场 / 自适应预算）
26. 契约、不变量与版本化（与 scene 接缝汇总）
27. 路线图、基准即规格、诚实边界与一句话总结

---

## 1. 设计哲学与架构翻转

世界系统要回答一个问题：**一个比 ECS 大几个数量级、作者摆不完、内存装不下的世界，如何被组织、流送、驻留、持久化、协作，而又只有一份真相？** 旧 v2 稿的答案是「立一个列式空间数据库 WorldDB 作真相源，ECS 实体只是临时视图」。但 `prism_scene` 定稿后，真相归属变了：**作者编辑的一切内容都是 scene（实体子树模板 + 嵌套 + override），scene 才是逻辑真相**。若世界系统再立一个 WorldDB 平行真相，就会出现「同一对象在 scene 和 WorldDB 各存一份、两边同步」的经典双真相之痛。

因此本次重构的根本翻转：**世界系统不拥有内容真相，只拥有「空间组织 + 流送调度 + 常驻残留 + 持久化 delta」**。

| 维度 | v2 旧稿（WorldDB-as-truth） | v3 本稿（scene-as-truth 基座） |
|---|---|---|
| 内容真相 | WorldDB 列式数据库 | `prism_scene`（cell = 顶层 scene，内容即嵌套 scene 实例 + scatter 节点） |
| 世界系统角色 | 真相源 + 流送 + 物化 | 纯基座：空间索引 + 流送 VM + 常驻档位 + 持久 delta |
| 持久可写状态 | WorldDB overlay | **仅 override delta 事务日志**（= scene §15 语义） |
| 其余世界数据 | DB 派生列 | **确定性可再生派生缓存**（索引/HLOD/SDF/nav/散布 proxy/烘焙 GI） |
| 对象双向 | DB record ⇄ 物化 entity | scene 的 `spawn/despawn/materialize`（§6、scene §12.2） |
| 渲染/物理/GI | 复用 Bevy 地基 | **全去 Bevy**，经 GPU 实例流 + ECS 物化喂 `prism_render`/`physics`/`gi`（§20） |

一句话定位：`prism_world` 是把 `prism_scene` 的作者内容，按空间**铺到行星尺度、按预算流送驻留、按档位物化、把运行时改动以 override delta 持久化并多人协作**的基座——它不发明内容，只调度内容。

设计信条（承接 Prism 范式）：

- **单一真相，派生可弃**：唯一权威可变状态是 override delta 事务日志；一切空间索引/HLOD/SDF/nav/散布 proxy/烘焙 GI 都是可由「scene 内容 + 种子 + 规则 + schema/图版本」确定性再生的缓存，丢了能重算，坏了能重建。
- **世界大小 ≠ ECS 大小**：绝大多数内容走 GPU 常驻实例流或 proxy 流，永不进 ECS；只有进入交互半径且需被玩法操作的对象才物化（scene Tier A）。
- **成本正比于变化**：相机静止、世界未编辑时，流送写入与持久化写入趋近零（§22、§27）。
- **禁 pop 硬切换**：几何 morph / 连续屏幕误差 LOD / 密度淡入 / HLOD dither，无突变。
- **GPU-first 零回读**：评估/剔除/LOD/密度在 GPU，CPU 只提交增量与策略；IO 异步、GPU 直传。
- **确定性**：固定 seed + sector id + schema/图版本 ⇒ PCG 可复现增量可缓存；事务日志可确定性回放（存档/多人/时间轴/回放调试）。

---

## 2. 参考产品取舍

借形态不抄码，只在**架构/算法层**借鉴，不复制任何源码。

| 产品 | 借鉴点（架构/算法层） | Prism 落法（本文） |
|---|---|---|
| **UE World Partition / Data Layers / HLOD / OFPA / Level Instance** | 隐式空间网格流送、数据层开关、分层代理、每对象协作粒度、关卡实例复用 | §5 空间索引、§6 cell=scene、§9 档位、§10 HLOD、§11 每对象 delta（OFPA 形态） |
| **Star Citizen / Outerra / MSFS** | cube-sphere 四叉树、64 位定点消抖、分区消除浮点误差、超大尺度 LOD | §5 i64 定点 + sector-local、§19 行星尺度 |
| **Decima / RDR2(RAGE)** | GPU-driven 植被、tile 流送、无缝 LOD/imposter、远景烘焙、路网河流样条 | §7 流送 VM、§15 散布、§10 代理、§14 PCG 样条 |
| **Houdini / UE PCG / HDA** | 节点图程序化生成、点云散布、确定性烘焙 + 增量重算 | §14 PCG 图→IR→GPU 求值 |
| **No Man's Sky / Teardown / Deep Rock / Chaos** | 体素/SDF 可破坏、Voronoi 预断裂、碎片持久化、运行时挖掘 | §16 破坏、§13 体素桶、§17 SDF |
| **Git / CRDT / 事件溯源** | append-only 事务日志、冲突合并、确定性回放、时间轴 | §11 持久化、§18 时间轴 |
| **列存 SoA（Arrow/Parquet）** | 列式块 mmap 零解析、只拉需要的列、内容寻址去重 | §5 sector 列、§8 IO、§22 内存 |
| **DirectStorage / RTX IO / io_uring** | GPU 直传解压、绕 CPU 拷贝、硬件解压 | §8 异步 IO |
| **UE Global Distance Field / Lumen SDF** | 稀疏体素 brick + mip 环，一场多用（阴影/AO/远场 GI/碰撞/EQS/nav） | §17 全局 SDF |
| **CDLOD / clipmap / Transvoxel / Dual Contouring** | 连续误差地形 LOD、体素等值面提取 | §13 地形多表示 |

**综合**：以 **UE World Partition 的空间流送粒度** 为骨架，**定点坐标 + cube-sphere** 为尺度无抖，**列存 SoA + GPU 直传** 为流送性能，**事务日志 + CRDT** 为协作/存档/时间轴，**PCG 图 + 全局 SDF** 为程序化与一场多用——但真相始终是 `prism_scene`，世界系统只是其空间调度基座。全部走 feature/档位门控，默认只付「空间索引 + 流送 + 整树卸载」最小成本。

---

## 3. 档位化（capability / quality tier / feature）

三重门控：**capability（平台/构建能力）× quality tier（运行时质量档）× feature flag（编译期裁剪）**。任何高级路径都可降级或编译期移除，核心路径永远可用。

| 能力 | feature | 默认 | 降级形态 |
|---|---|---|---|
| 空间索引 + 流送 VM + 整 cell 加载/卸载 | 恒开（`no_std+alloc` 核） | 开 | 不可降级，最小内核 |
| 异步流送 | `async_io` | 开（std） | 关则同步阻塞加载 |
| GPU 直传解压 | `gpu_stream`（需平台能力） | 开 | 关则 CPU 异步解压回退 |
| 三档常驻 + 按需物化 | 恒开（驱动 scene residency） | 开 | 关则所有 cell 整体物化（小世界可行） |
| HLOD 远景代理 | `hlod` | 开 | 关则远景直接剔除（画面退化不崩溃） |
| 持久化 / 存档 / 时间轴 | `persist` | 关 | 关则无 override delta 落盘、无回放 |
| 程序化生成 | `pcg` | 关 | 关则只走作者手摆 + 预烘焙资产 |
| 仿真（侵蚀/破坏/生态/水文） | `sim` | 关 | 关则世界静态（破坏退化为预断裂切换） |
| 全局 SDF 世界场 | `sdf` | 关 | 关则各系统回退各自代理几何 |
| 行星尺度 cube-sphere | `planet` | 关 | 关则平面 sector 网格（常规关卡尺度） |

quality tier（运行时）示例：流送半径、显存预算、每帧物化上限、proxy 密度、SDF 环分辨率、sim tick 频率随 tier 调。capability 示例：无平台 `gpu_stream` 能力时走 CPU 解压；无 `planet` 时走平面网格。

---

## 4. 多 crate 分层架构

世界系统拆成三个职责正交的 crate，均架在 `prism_scene` 契约（§12.2）之上，彼此只经稳定数据结构与事件通信，不互相反向依赖。拆分动机：**流送（何时拉数据）、常驻（内容以何种重量存在）、持久化（改动如何落盘与协作）** 是三条独立演化、独立 feature 门控、独立测试的轴，捆在一个 crate 里会让存档逻辑拖累流送热路径（网络等横切消费层另立 `prism_network`，不进 world 单机构建）。

```text
                         上层消费者（去 Bevy）
        prism_render        prism_physics        prism_gi / prism_nav
            ▲                     ▲                      ▲
            │ GPU 常驻实例流         │ ECS 物化实体            │ SDF 场 / proxy 列
            │ （Tier C）            │ （Tier A/B）            │
┌───────────┴─────────────────────┴──────────────────────┴────────────┐
│                        prism_world（世界系统基座）                      │
│                                                                      │
│  ┌────────────────┐   ┌──────────────────┐   ┌───────────────────┐  │
│  │ prism_world_   │   │ prism_world_      │   │ prism_world_      │  │
│  │   stream       │──▶│   residency       │──▶│   persist         │  │
│  │ 空间索引/流送VM  │   │ 三档常驻/按需物化    │   │ 事务日志/CRDT/存档  │  │
│  │ /GPU 直传触发   │   │ /proxy⇄ECS 升降级 │   │ /时间轴/快照回放    │  │
│  └───────┬────────┘   └────────┬─────────┘   └─────────┬─────────┘  │
└──────────┼─────────────────────┼───────────────────────┼───────────┘
           │ spawn_scene         │ materialize/           │ PersistentId
           │ despawn_scene       │ dematerialize          │ ⇄ OverrideDelta
           ▼                     ▼                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│   prism_scene（唯一作者态逻辑真相：Template / Instance / Override）        │
│   §12.2 契约：spawn·despawn·materialize·PersistentId⇄delta·reload       │
└──────────────────────────────────────────────────────────────────────┘
           │                     │                         │
           ▼                     ▼                         ▼
     prism_asset           prism_ecs                 prism_transform
   （StableGuid/软引用/      （物化边界的批量          （层级/世界变换）
     依赖闭包/mmap）          预留 + 列写入）          + prism_math（定点/编码/噪声）
```

**三 crate 职责边界**：

| crate | 拥有什么 | 不拥有什么 | 关键 feature |
|---|---|---|---|
| `prism_world_stream` | 空间索引（sector 树）、流送 VM（需求集 → 预算 → 优先级 → 预取 → 驱逐）、异步 IO 与 GPU 直传触发、cell manifest 解析 | 不拥有内容（调 scene spawn）、不拥有持久状态 | `async_io`/`gpu_stream`/`planet` |
| `prism_world_residency` | 三档常驻状态机、proxy ⇄ ECS 升降级调度、HLOD 簇驱动、物化预算与迟滞 | 不拥有真相（proxy delta 读写转交 persist）、不绘制（只发信号给 render） | `hlod`/`sim`/`sdf` |
| `prism_world_persist` | override delta 事务日志、CRDT 空间合并、快照/时间轴、存档容器 | 不拥有空间调度、不定义内容结构（结构归 scene）、不实现网络（归 `prism_network`） | `persist` |

**与 scene / ecs / asset 的接缝**：

- 与 `prism_scene`：唯一经 scene §12.2 的六条契约调用；世界系统**从不**绕过 scene 直接写实体，保证单一真相。
- 与 `prism_ecs`：物化时 `prism_world_residency` 向 ECS 申请 **批量 archetype 预留 + 列式写入**（接 scene §8 快路径），避免逐实体 spawn 的 archetype 抖动。
- 与 `prism_asset`：所有 cell manifest / scene template / 代理资产 / 派生缓存都是 `StableGuid` 寻址的内容包，经软引用 + 依赖闭包 + mmap 零解析加载（§8）。
- 与 `prism_tasks`：流送 IO、PCG 烘焙、SDF/nav 重算全提交到 `prism_tasks` 的并行作业系统，带优先级与可取消。

**依赖方向严格单向**：stream → residency → persist（数据与触发），persist 的产出（override delta）回流 residency 的物化（读存档施加改动），但只经稳定接口不形成编译环；三者都只向下依赖 scene/ecs/asset，不被它们依赖。

---

## 5. 空间模型与坐标

### 5.1 sector 分区与隐式网格

借 UE World Partition 的**隐式空间网格**形态：世界被切成固定边长的 sector（如 256 m 立方，可按项目配），sector 不是作者摆放的对象，而是由对象的空间坐标**隐式归属**——作者只管摆内容，系统按坐标自动算它落在哪个 sector。sector 构成流送、持久化、空间查询的最小寻址单元。

- **多级 sector 树**：sector 之上有 region（如 16×16 sector），region 之上有 tile；平面世界用四叉树（水平分区足够），行星/体素世界用八叉树（§19 的 cube-sphere 面内四叉 + 高度分层）。
- **sector id**：由 `(level, x, y, z)` 经空间编码（见 §5.3）压成一个 `u64`，稳定、可复现、可跨机一致，作为 override delta、派生缓存、网络订阅、PCG 种子的空间锚点。

### 5.2 i64 定点坐标 + sector-local f32（消灭 rebasing）

浮点坐标在行星尺度会抖动（远离原点时 f32 精度崩塌）。传统做法是「origin rebasing」（周期性把世界平移回原点），但 rebasing 要全局改坐标、易出 bug、与多人/确定性冲突。本设计**从根上消除 rebasing**：

```text
全局坐标 = i64 定点（每轴），单位可配（默认 1 单位 = 1 mm）
         ⇒ ±2^63 mm ≈ ±9.2e9 km，覆盖整个行星系尺度，零精度衰减
sector-local 坐标 = f32，相对所属 sector 原点
         ⇒ 256 m sector 内 f32 精度 ≈ 0.015 mm，渲染/物理足够
```

- **全局用 i64**：对象的权威位置是 `GlobalPos { x: i64, y: i64, z: i64 }`，定点运算无累积误差，加减精确，跨机位一致（确定性）。
- **局部用 f32**：进入某 sector 的内容，渲染/物理用「相对该 sector 原点」的 f32 局部坐标；GPU/物理引擎永远在小数值域工作，永不接触大坐标。
- **相机相对渲染**：绘制时统一换算到「相机所在 sector 为临时原点」的 camera-relative f32，远物体用 double→float 的分段精度，无抖动。
- **跨 sector 运动**：对象越界只改 `GlobalPos` 的 i64 并重算 local f32，是一次 O(1) 定点加减，无全局 rebasing，无其他对象受影响。

定点编解码、sector 归属、local 换算都是 `prism_math` 的纯函数，**可单元测试、可确定性验证**（这是本文少数能在无 GPU 本机实测的部分）。

### 5.3 Morton / Hilbert 空间近邻排序

sector id 与 sector 内对象的存储顺序，按空间填充曲线排序，使**空间近邻 ⇒ 存储近邻 ⇒ IO/缓存近邻**：

- **Morton（Z-order）**：位交织编码，编解码极廉（纯位运算），作为 sector id 的默认编码与八叉树遍历序。
- **Hilbert**：局部性优于 Morton（无 Z 跳变），用于磁盘上 cell 块的落盘顺序与预取窗口排序，让「相机半径内的 cell」在磁盘上也物理相邻，一次大块顺序读拉完。
- **列式块内排序**：sector 内的 Tier C proxy 按 Hilbert 排序成列（SoA），GPU 流送与视锥剔除按连续区间批处理（§7、§15）。

### 5.4 cube-sphere（行星尺度，`planet` 档位）

行星不是平面：用 cube-sphere（立方体六面投影到球面）替代经纬网格，避免两极奇异与赤道拉伸。每面一棵四叉树做 CDLOD 地形细分（§13、§19），面内坐标仍走 §5.2 的 i64 定点 + local f32。关闭 `planet` 时退化为单一平面 sector 网格（常规关卡尺度），零额外成本。

---

## 6. cell ⇄ scene 映射（世界是 scene 的空间组织层，非内容真相）

这是架构翻转（§1）的落地核心：**世界不存内容，世界把 scene 按空间铺开**。

### 6.1 cell = 顶层 scene

- 一个 **cell** 是「某 sector（或 sector 子块）内全部作者内容」的容器，其内容形态**就是一棵顶层 scene**（scene §9.3「level 即顶层 scene」）：根下挂若干**嵌套 scene 实例节点**（建筑、道具组、关卡区块）与**scatter 节点**（海量散布的规则声明，§14/§15）。
- cell 没有自己的内容结构定义，它复用 scene 的 Template / Instance / Override 三层（scene §5）。世界系统持有的只是「这个 sector 对应哪个 scene 的 `SoftHandle` + 其空间范围 + 流送元数据」，即 **cell manifest**。

### 6.2 cell manifest（流送元数据，非内容）

每个 sector 落盘一份轻量 manifest（内容寻址、mmap 零解析），供流送 VM 在**不加载内容**的前提下做调度决策：

```text
CellManifest {
  sector_id:        u64,                    // §5.3 空间编码
  bounds:           Aabb<i64>,              // 定点包围盒(§5.2)
  scene:            SoftHandle<SceneTemplate>, // 该 cell 的顶层 scene
  residency_hint:   ResidencyPlan 摘要,      // Tier A/B/C 占比(scene §11)
  data_layers:      BitSet,                 // 数据层掩码(§6.4)
  proxy_summary:    { count, 代理簇列表, HLOD 阈值 }, // §10 远景不加载也能画
  deps_closure_size: u64,                   // 依赖闭包字节数(预算用)
  derived:          { 空间索引/SDF/nav/烘焙GI 的 StableGuid 列 }, // §6.3
  content_hash:     Hash,                    // 确定性校验/增量判脏
}
```

manifest 极小（KB 级），常驻内存构成全世界的「空间骨架」；真正的内容（scene template + 依赖闭包）只在 cell 进范围时才按 `scene` 句柄加载。

### 6.3 两类持久数据（除此之外世界不存任何真相）

| 类别 | 内容 | 可变性 | 丢失后果 |
|---|---|---|---|
| **override delta 事务日志** | 运行时/编辑时对 scene 内容的改动（移动、删除、属性改、破坏碎片、物化对象状态） | **唯一权威可写状态** | 丢失即丢档，须备份/复制（§11、§12） |
| **确定性可再生派生缓存** | 空间索引、HLOD 代理、全局 SDF、导航网格、散布 proxy 列、烘焙 GI、PCG 产物 | 只读缓存，可重算 | 丢失只需按「scene + 种子 + 规则 + schema/图版本」重烘，不丢数据 |

override delta 的语义**完全等同 scene §15 的持久化语义**——世界系统的 persist crate 只是把 scene 产出的 `PersistentId → OverrideDelta` 按 sector 聚合、落盘、合并、复制，**不发明新的持久格式**。

### 6.4 数据层（Data Layers，借 UE）

正交于空间的内容分组：同一 sector 内的内容可按 **数据层**（如 `昼/夜`、`战前/战后`、`DLC_雪原`、`多人实例A`）打标，运行时按掩码开关整层的加载/可见，无需复制 cell。数据层掩码进 cell manifest，流送 VM 按当前激活掩码过滤要加载的子场景节点。数据层本身也是 scene 节点上的 tag，不是平行真相。

---

## 7. prism_world_stream：流送 VM 与残留引擎

流送子系统是一台**数据流虚拟机**：输入「需求集」，经预算与优先级求解，输出「本帧要加载/卸载/升降级什么」的指令流，驱动 scene 的 spawn/despawn（scene §12.2）。核心是「残留引擎」——决定每一刻哪些数据该留在内存/显存、哪些该走。

### 7.1 需求集（demand set）

每帧由若干**需求源**生成「空间中哪些区域、需要哪些数据层、到什么细节」的需求，叠加成需求场：

```text
Demand = ⋃ 需求源i {
  center:   GlobalPos,        // 相机/玩家/传送目标/服务器 AOI 中心
  radii:    [r_physics, r_interact, r_visible, r_proxy, r_prefetch], // 多层半径
  layers:   BitSet,           // 需要的数据层(§6.4)
  tier_cap: 各档上限,          // 该源允许的 Tier A/B/C 配额
  priority: f32,              // 源优先级(主相机 > 次相机 > 预测)
}
```

- **多层半径**：物理半径（最近，需 Tier A 活实体）< 交互半径 < 可见半径（Tier B 实例化）< proxy 半径（Tier C GPU 实例）< 预取半径（只拉 manifest 与依赖闭包，不实例化）。层层外扩，细节层层递减。
- **多需求源**：多相机（分屏/镜子/传送门预览）、多玩家（多人）、服务器 AOI 兴趣中心（§12）各贡献一个需求源，取并集，去重取最高细节。

### 7.2 残留引擎：预算 × 优先级 × 迟滞

需求集只说「想要什么」，残留引擎在**预算**约束下决定「实际做什么」：

```text
每帧:
  1. 空间查询: 用 sector 树(§5)求需求场覆盖的 cell 集,按细节层分桶
  2. diff: 对比当前常驻集,得 {to_load, to_unload, to_upgrade, to_downgrade}
  3. 优先级排序: 交互 > 视野中心 > 视野边缘 > 预取;近 > 远;高层半径 > 低层
  4. 预算裁剪: 本帧 IO 带宽/解压/显存/物化实体数各有预算(§22),从高优到低优填满为止
  5. 迟滞: to_unload/to_downgrade 加滞后带(hysteresis),防临界距离抖动
  6. 发指令: load→scene.spawn, unload→scene.despawn, upgrade→materialize, downgrade→dematerialize
```

- **预算驱动而非距离驱动**：不是「到了距离就加载」，而是「优先级 + 预算决定本帧配额」，超预算的低优需求顺延到后续帧，平滑降级而非卡顿或爆显存。
- **迟滞双阈值**：加载阈值 < 卸载阈值，中间是滞后带；对象在临界距离来回时不会反复 load/unload（借 §23 的滞后设计）。
- **平滑降级**：预算吃紧时，优先降 proxy 密度、降 HLOD 层级、延后预取，最后才牺牲可见半径；永不硬 pop（§1 信条）。

### 7.3 预测式预取

借 UE World Partition 的预加载边带：残留引擎用相机速度/朝向外推「未来 0.5–2 s 的需求中心」，对即将进范围的 cell 提前 `load_soft` 走依赖闭包就绪（scene §12.3），但**延后实例化到真正进范围**。预取与实例化解耦，掩盖 IO 延迟。预取半径随速度动态扩张（高速移动时提前更多）。

### 7.4 驱逐与回收

出范围内容按 scene §12.2 的 `despawn_scene` 整树回收，走延迟队列与加载错峰（scene §12.3 无缝卸载）；其依赖资产经 `prism_asset` 引用计数 + 释放队列回收。被改过的内容在卸载前先把 override delta 回写 persist（§11），再退回 proxy 或彻底卸载——**卸载不丢改动**。

### 7.5 流送 VM 指令流（可记录可回放）

残留引擎每帧产出的指令流是结构化事件（`Load/Unload/Upgrade/Downgrade/Prefetch`），可录制用于：确定性回放调试（§18）、性能剖析（哪帧加载了什么）、以及服务器把权威流送决策下发客户端（§12）。VM 自身无状态副作用，便于测试与复现。

---

## 8. 异步 IO 与 GPU 直传

流送性能的下限由 IO 决定。目标：**磁盘 → 显存零 CPU 回读、零多余拷贝、硬件解压**（借 DirectStorage / RTX IO / io_uring 形态，`async_io` + `gpu_stream` 档位）。

### 8.1 分层 IO 路径

```text
Tier C proxy / 几何 / 纹理(GPU 直传路径, gpu_stream):
  磁盘 → DMA → GPU 显存 → GPU 解压(硬件/compute) → GPU 常驻实例表
  （CPU 只提交 IO 请求与解压 dispatch,不碰数据,零回读）

scene template / manifest / override delta(CPU 路径, async_io):
  磁盘 → mmap 零解析(§8.2) → 直接作为结构体视图,无反序列化
  （schema 稳定字段 ID,内容寻址去重,见 scene §10.2）
```

- **GPU 直传**：大块、不可变、GPU-only 的数据（网格/纹理/proxy 实例列/SDF brick）走直传，绕开 CPU 内存与 CPU 解压，省带宽省延迟。
- **mmap 零解析**：结构化小数据（manifest/template/delta）用 `prism_asset` 的内容寻址包 + 稳定 schema，mmap 后按偏移直接当结构体读，无 `deserialize` 开销。

### 8.2 回退与可用性

- 无 `gpu_stream` 平台能力时，自动回退 **CPU 异步解压**（`prism_tasks` 工作线程解压后再上传），功能不变、吞吐降低，如实标注为 design target。
- 无 `async_io`（如极简嵌入构建）时退化为同步阻塞加载，仅适合小世界。
- 所有 IO 作业带优先级与可取消：高优 cell 的 IO 可抢占低优预取 IO；需求消失的预取 IO 立即取消，不浪费带宽。

### 8.3 IO 预算与背压

IO 子系统有显式带宽预算（design target，如 PCIe 4.0 NVMe ≈ 7 GB/s、直传解压后等效更高），残留引擎（§7）按此预算裁剪本帧 IO 指令；队列积压时产生背压，残留引擎自动降预取、降细节，保证热路径（交互半径）IO 优先完成。

> IO/带宽/显存数字均为 design target，本机无 GPU 与高速存储实测环境，如实标注；可验证部分限于 CPU 侧的 mmap 零解析、调度与预算裁剪逻辑的纯函数单测。

---

## 9. prism_world_residency：常驻档位与按需物化

residency crate 把 scene §11 的三档常驻落为一台状态机，并驱动 proxy ⇄ ECS 的升降级。它是「世界有多重」的调度者：同样一片内容，可以是一行 GPU 实例（最轻），也可以是一棵活 ECS 子树（最重）。

### 9.1 三档常驻状态机（对齐 scene §11.1）

| 档位 | 存在形态 | 进 ECS | 驱动来源 | 典型内容 |
|---|---|---|---|---|
| **Tier A interactive** | 活 ECS 实体子树 | 是 | 进物理/交互半径，或被物化穿透（§10） | 可交互对象、角色、任务点、动态物件 |
| **Tier B streamed** | 进范围整块实例化、出范围整树卸载 | 进范围时是 | 进/出可见半径 | 室内、建筑内部、关卡区块 |
| **Tier C proxy** | SoA proxy 列 → GPU 常驻实例 + 空间记录 | 否（除非物化） | 进/出 proxy 半径 | 海量静态散布（树/石/草/路灯） |

状态机的合法迁移：`C ⇄ B ⇄ A` 与 `C ⇄ A`（物化穿透）。每条迁移都映射到 scene §12.2 的一个契约调用，迁移带迟滞与预算（§7.2）。

### 9.2 按需物化（proxy ⇄ ECS，接 scene §11.2）

```text
materialize(proxy):   // C/B → A,被靠近或交互
  1. 查 proxy 的 SceneTemplate + 变换 + 持久 override(若曾改,从 persist §11 读)
  2. 调 scene.materialize → 走 scene §8 快路径实例化活子树
  3. 按 PersistentId 施加存档 override delta
  4. 从代理绘制集排除该单体(§10 物化穿透,防双绘)

dematerialize(instance):  // A → C/B,退离
  1. 把实例当前 override delta 序列化回 persist(§11),不丢改动
  2. 调 scene.despawn 整树回收
  3. 退回 GPU 常驻 proxy 表示,重新纳入代理绘制集
```

全程是 scene 的 spawn/despawn + override 读写，**无第二套序列化、无平行真相**。

### 9.3 物化预算、迟滞与分帧

- **物化预算**：每帧物化/反物化的实体数、archetype 列写入量有上限（§22），超出顺延；大子树实例化按 archetype 组分帧切片（scene §12.3），整树就绪才对外可见。
- **迟滞**：物化/反物化双阈值 + 最短驻留时间，防玩家在临界距离抖动导致反复升降级（§23）。
- **批量 ECS 预留**：物化走 `prism_ecs` 的批量 archetype 预留 + 列式写入（接 scene §8），避免逐实体 spawn 的结构抖动。

### 9.4 常驻与数据层/网络的交互

- 数据层（§6.4）关闭的内容即使在半径内也不常驻；重新激活数据层触发该层内容的流送进场。
- 多人下，服务器可强制某些对象常驻 Tier A（权威仿真对象），客户端按 AOI（§12）决定本地常驻档，权威状态经复制覆盖本地（见 `prism_network`）。

---

## 10. 簇 LOD 统一模型与 HLOD 远景代理

远景不能逐实例绘制。Prism 只有**一套簇（cluster）模型**，按尺度分两级、共用同一套不变量：**对象内部的三角簇**（虚拟几何，§25.1，`vgeo` 档位，达成单网格像素级细节）与**跨对象的聚合簇**（HLOD 远景代理，本节，把整片远景聚成代理）。本节在 §10.1 把两级共用的不变量**定义一次**，再落地跨对象聚合尺度；对象内部的 meshlet / 簇 DAG / GPU 软光栅细节见 §25.1，不在此重复，二者正交叠加而非两套平行系统。

residency crate 驱动 scene §11.4 的「簇声明 + 切换契约」，代理资产由渲染侧离线烘焙，本 crate 只持引用、维护成员归属、按阈值发切换信号（职责单向，借 UE HLOD / Decima tile 代理形态）。

### 10.1 簇模型不变量（两级尺度共用，只在此定义）

- **确定性成簇**：成簇是 `seed + content_hash` 的确定性烘焙，簇 id 可复现、跨机一致，供网络（`prism_network`）/ 存档按簇寻址（scene §11.4、§21）。
- **屏幕空间误差驱动 LOD**：按屏幕像素误差选细节档（HLOD 选代理级、虚拟几何选簇 DAG 切面），**成本正比于可感知细节**（§1）。
- **禁 pop**：切换带阈值滞后 + dither / 几何 morph 过渡（§1 信条、§13 CDLOD 同机制），无突变。
- **派生缓存**：簇及其代理 / 切面都是可重烘的派生缓存（§6.3），不入存档、永不作真相被编辑；编辑只发生在物化后的 Tier A 实例，改动经 override delta 标脏该簇 → 异步重烘（§16）。

### 10.2 簇声明与分层代理

- **成簇**：烘焙期把一片 Tier C proxy 按固定空间网格 + 稳定排序聚成跨对象聚合簇（成簇不变量见 §10.1，scene §11.4）。
- **分层**：每簇多级代理（近 → 中 → 远）对应 HLOD 层级，每级一个代理资产（合并网格 / impostor 公告板 / 体素化壳），按屏幕空间误差阈值切换。
- **声明内容**：`{ 代理资产 SoftHandle, 各级切换阈值, 成员 PersistentId 列, 簇包围盒 }`，进 cell manifest 的 `proxy_summary`，使**远景不加载 cell 内容也能画出代理**（manifest 常驻，§6.2）。

### 10.3 连续切换（禁 pop）

越过屏幕空间误差阈值发 `HLODSwitch{ cluster, from_level, to_level }`，渲染侧换绘制表示。为禁 pop：阈值带滞后 + dither/淡入过渡（§1 信条），单体 proxy 的 LOD 用连续屏幕误差 + 几何 morph（§13 CDLOD 同机制），玩家无感。

### 10.4 物化穿透

玩家进簇交互半径，被触碰的 proxy 单体按 §9.2 物化为 Tier A 实例，其余仍走代理；物化单体从代理绘制集按 `PersistentId` 剔除（渲染侧执行），避免双重绘制。退离后反物化、重归代理集。簇的其余成员完全不受影响，物化是**单体粒度**的。

### 10.5 远景只读、近景真相

跨对象聚合簇的代理遵循 §10.1 的派生缓存不变量：丢失可按簇成员的 scene 内容重烘，永不作真相被编辑，编辑经物化后的 Tier A 实例回写 override delta → 标脏该簇 → 异步重烘代理（§16）。

---

## 11. prism_world_persist：事务日志 / CRDT / 时间轴 / 存档

persist crate 是世界唯一权威可写状态的管理者（§6.3）。它不定义内容结构（结构归 scene），只把 scene 产出的 `PersistentId → OverrideDelta`（scene §15）按 sector 聚合、落盘、合并、复制、回放。核心数据结构是 **append-only 事务日志**（借 Git / 事件溯源 / CRDT 形态，`persist` 档位）。

### 11.1 存档即 override delta 事务日志

- 存档不是「世界快照」，而是「从初始作者态到当前的 override delta 序列」（scene §15.1）。初始态由 scene 内容 + 种子 + 规则确定性再生（§6.3），存档只记差异，**体积正比于玩家改动量而非世界大小**（§1「成本正比于变化」）。
- 事务是**不可变追加**：每条 `Txn { seq, timestamp, sector_id, PersistentId, OverrideDelta, author }`，只追加不改写，天然支持回滚（截断日志）、审计（谁在何时改了什么）、增量存档（只传新事务）。
- 按 sector 分片：日志按 `sector_id` 分段落盘（OFPA 每对象文件化形态——改一个对象只写一个小文件/一段，不重写整存档），冷热分离，并发写不同 sector 无锁冲突。

### 11.2 PersistentId 作为跨会话锚点

override delta 挂在 scene 的 `PersistentId`（确定性派生，scene §6.2）上，而非易变的运行时实体 id。重新加载世界时，物化（§9.2）按 `PersistentId` 定位对象并施加其 delta。即使 schema 迁移、对象重新实例化，锚点稳定，存档不失效（scene §15.4 版本迁移）。

### 11.3 CRDT 空间合并（多人/多端编辑）

多人协作或多端离线编辑时，同一 sector 可能被并发改动。用 **CRDT**（无冲突复制数据类型）做确定性合并，无需中央锁：

| 改动类型 | CRDT 策略 |
|---|---|
| 属性覆盖（位置/材质参数） | last-writer-wins + 向量钟（Lamport）定序；或字段级 LWW-register |
| 增删对象（破坏碎片/放置物） | add/remove set（OR-Set 形态），加后删安全，重复加幂等 |
| 计数/累积（资源量/损伤值） | PN-Counter / grow-only 累积 |
| 结构性冲突（同对象删 vs 改） | 策略可配：删优先 / 保留并标冲突待人工 |

合并是**确定性**的（相同事务集 ⇒ 相同结果，与到达顺序无关），这是多人与离线同步一致性的根基。合并发生在事务日志层，scene 内容只消费合并后的 delta。

### 11.4 时间轴快照与回放（接 §18）

- **快照（snapshot）**：周期性对事务日志做物化快照（某 seq 处的完整 override 集），加速加载（不必从头重放全部事务）。快照是优化缓存，可丢弃重建。
- **反向重放**：每条 override delta 记录逆操作（或由前一快照 + 正向重放求得），支持时间倒流、撤销/重做、回放调试（§18）。
- 时间轴是四维世界的基础：事务日志 = 事件流，快照 = 时间锚点，反向 = 回溯（§18 详述）。

### 11.5 存档容器与完整性

存档是 `prism_asset` 的内容寻址容器：事务日志分片 + 快照 + 元数据（schema/图版本、种子、世界 content_hash）。加载时校验 content_hash 与版本，不匹配走迁移（scene §15.4）。派生缓存（§6.3）**不入存档**——存档只存不可再生的 delta，派生物按需重烘，使存档最小、最鲁棒。

---

## 12. 网络接缝（复制 / 权威 / AOI 归 `prism_network`）

多人网络**不是世界系统的职责**，它是架在 scene / persist / world 之上的横切消费层，独立成 crate 与文档（`prism_network`，另文），与仓库「一子系统一文档」的惯例一致。world 在此只定义**接缝**：把网络框架所需的空间与数据原料暴露出去，不实现复制、权威、预测、分片。

### 12.1 world 向 `prism_network` 暴露什么

| 原料 | 来源 | 网络侧用途 |
|---|---|---|
| sector 树 / 空间索引查询 | §5 | AOI 空间裁剪、服务器分片按子树划分 |
| HLOD 簇 id（确定性、可复现） | §10、scene §11.4 | 按簇/实例分级订阅（近处逐实例、远处簇级聚合） |
| 需求源（demand source） | §7.1 | AOI 兴趣中心复用需求源结构，可反向下发权威流送决策（§7.5） |
| override delta 事务接口 | §11（persist） | 复制单元 = delta，与存档同一套序列化与确定性合并 |
| 确定性契约 | §21 | 客户端预测 / 和解 / 多端最终一致的前提 |

### 12.2 边界与不变量

- **world 不碰网络协议**：谁是权威、怎么订阅、怎么预测和解、跨服务器怎么移交权威，全在 `prism_network`；world 只回答「空间里有什么、簇怎么分、delta 怎么取」。
- **一套 delta 两个消费者**：存档（§11 事务日志）与网络（`prism_network`）消费同一份 override delta 语义，world/persist 不为网络另立真相。
- **零侵入热路径**：`prism_network` 只依赖 persist 的事务接口与 world 的空间查询接口，不侵入 stream/residency 热路径；无网络的单机构建里，world 不含任何网络代码。
- **服务器网格分片**（原 server_mesh）同属 `prism_network`：它只是「单权威 → N 服务器」的规模档位，消费 world 的 sector 树做分片划分、消费 persist 的 delta 做权威移交，不是 world 的高级功能。

> 本节只描述接缝的数据与方向，不含吞吐/延迟指标；网络的 design target 与一致性证明见 `prism_network`。

---

## 13. 地形多表示层

地形是世界的底板，需多种表示按场景取舍（接 `prism_terrain_world_design_zh.md` 的细节，本文定其在世界系统中的接缝与档位）。地形既是**内容**（作者可编辑的高度/体素），其几何又是**派生缓存**（LOD 网格可重算，§6.3）。

### 13.1 表示矩阵

| 表示 | 算法形态 | 适用 | 可破坏 | 档位 |
|---|---|---|---|---|
| **高度场 heightmap** | CDLOD / clipmap 连续误差 LOD | 常规地貌、丘陵平原 | 否（形变有限） | 恒开 |
| **虚拟高度图 VHM / 虚拟地形** | 页式虚拟纹理 + 按需加载高度页 | 超大高度场、行星 | 否 | `planet` |
| **体素 voxel** | Transvoxel / Dual Contouring 等值面提取 | 洞穴、悬垂、可挖掘 | 是 | `sim`/`sdf` |
| **SDF 地形** | 稀疏 brick SDF（接 §17） | 布尔编辑、平滑破坏 | 是 | `sdf` |
| **样条扫掠** | 路网/河流/护坡沿样条变形地形 | 道路、河床、人工地貌 | 作者态 | `pcg` |

### 13.2 CDLOD 连续细分（禁 pop）

高度场用 CDLOD（连续距离 LOD）：相邻 LOD 间顶点 morph 过渡（屏幕空间误差驱动的权重插值），裙边/缝合消 T-junction，无 LOD 突变。与行星 cube-sphere（§19）结合为「CDLOD-on-sphere」。地形块按 §5 sector 流送，远处块走 clipmap/VHM 页。

### 13.3 体素与等值面

可破坏/有悬垂的地形走体素：Transvoxel（跨 LOD 无缝等值面）或 Dual Contouring（保锐边）从体素场提取网格；布尔编辑（挖掘/堆积）改体素场 → 标脏 brick → 增量重提取受影响网格。体素场的**改动是 override delta**（§6.3、§16），网格是派生缓存。

### 13.4 地形与 scene / 流送的接缝

地形块作为 scene 的特殊节点（或 Tier C 几何），由流送 VM（§7）按 sector 加载、按 LOD 细分；地形编辑经 override delta 持久化（§11）；地形几何喂 §20 的 GPU 实例流 / 物理碰撞 / §17 全局 SDF。

---

## 14. 程序化生成 PCG

PCG（`pcg` 档位）让作者用**规则**而非手摆描述海量内容，确定性烘焙为可流送的 proxy/几何（借 Houdini HDA / UE PCG 节点图形态，纯程序化算法，无 AI/ML）。

### 14.1 节点图 → IR → GPU 求值

```text
作者: PCG 节点图(采样/散布/过滤/变换/布尔/样条/噪声节点)
  ↓ 编译
IR: 确定性数据流图(稳定节点 id + 版本)
  ↓ 求值(GPU compute 为主,CPU 回退)
产物: 点云/实例变换列/几何/体素 → 烘焙为 Tier C proxy 列 或 scene scatter 节点
```

- **图在 scene 模板声明**：PCG 规则作为 scene 的 scatter 节点（§6.1）声明在作者态，烘焙期求值为 proxy 列（§15），运行时流送 proxy 而非重算图。
- **确定性求值**：固定 `seed + sector_id + 图版本`（§21）⇒ 同输入同输出，可缓存、可增量、跨机一致。

### 14.2 确定性增量重算

改图的一个节点，只重算其下游受影响的 sector 分块（脏传播沿 IR 依赖边），其余 proxy 复用缓存。改动范围由节点的空间影响域界定，使大世界 PCG 编辑**成本正比于改动**（§1）。

### 14.3 典型程序化内容

| 内容 | 算法形态 |
|---|---|
| 植被/岩石散布 | 泊松盘/蓝噪声采样 + 坡度/高度/数据层过滤（§15） |
| 城市/建筑 | L-system / wave-function-collapse 式约束拼块（纯规则，非 ML） + 模块化拼接 |
| 路网/河流 | tensor field 引导样条 + 沿样条扫掠地形（§13.5）与放置 |
| 散布变化 | domain-warp 噪声（`prism_math`）驱动密度/种类/朝向 |

### 14.4 与流送/持久化的关系

PCG 产物是派生缓存（§6.3），不入存档；作者对 PCG 产物的手动改动（挪一棵树、删一块）才是 override delta（§11），重烘时保全这些 delta（按 PersistentId 叠加在重算结果上，scene §13.2 保全策略形态）。

---

## 15. 海量散布与植被

散布是「Tier C proxy 流」的主力场景：百万级树/草/石不进 ECS，以 GPU 常驻实例 + 空间记录存在（§9），由 §14 PCG 或作者刷子产出。

### 15.1 GPU-driven 实例流

```text
proxy 列(SoA, Hilbert 排序, §5.3)
  → GPU 常驻实例缓冲(位置/朝向/缩放/种类/LOD/数据层)
  → GPU 视锥+遮挡剔除 + 屏幕误差 LOD 选择(零 CPU 回读)
  → 间接绘制(indirect draw)喂 prism_render(§20)
```

CPU 只按 sector 流送 proxy 列的加载/卸载（§7），实例的剔除/LOD/密度全在 GPU，CPU 不逐实例处理（§1「GPU-first 零回读」）。

### 15.2 采样与分布

- **泊松盘 / 蓝噪声**：自然均匀无规则感的分布，GPU 可并行生成（tile 化蓝噪声），确定性（seed + sector）。
- **规则过滤**：按坡度、高度、曲率、数据层、其他散布层（如树下不长树）过滤采样点。
- **密度场**：domain-warp 噪声 + 作者刷子（刷密度/种类）叠加成密度场，驱动采样。

### 15.3 风场与交互场

- **风场**：全局 + 局部风场驱动植被 GPU 顶点动画（无需逐实例 CPU 更新），确定性噪声风 + 阵风事件。
- **交互场**：角色/载具经过压弯草木——写一张局部交互纹理（GPU），实例顶点采样该场形变，交互后弹回；交互场是瞬时的，不入持久化。
- **破坏/采集**：砍树/割草是 override delta（§11，删除某 PersistentId 的 proxy），重进范围时存档的删除 delta 使其不再出现。

### 15.4 LOD 与远景

散布实例多级 LOD（全网格 → 简化 → 公告板 impostor → 并入 HLOD 簇代理，§10），屏幕误差连续切换 + dither 淡入，禁 pop。极远并入簇代理，彻底不逐实例。

---

## 16. 仿真：侵蚀 / 破坏 / 生态 / 水文

仿真（`sim` 档位）让世界随时间演化。统一原则：**仿真改动 → override delta（§11）→ 标脏受影响派生缓存（HLOD/SDF/nav/网格）→ 异步重烘**；仿真不另立真相，其输出要么是 delta（持久改动）要么是派生缓存（可重算场）。全部为经典数值方法，无 AI/ML。

### 16.1 侵蚀（地形演化）

- **水力侵蚀**：浅水方程 / 管道模型（pipe model）在高度场上解水流 + 泥沙输运 PDE，GPU stencil 迭代；产出改高度场（§13）→ delta + 重烘网格。
- **热力侵蚀**：坡度超静止角则塌落（talus），迭代平滑陡坡。
- 可离线烘焙（作者态生成自然地貌）或运行时慢速演化（`sim` tick 频率随 tier，§3）。

### 16.2 破坏（可破坏世界）

- **预断裂**：烘焙期按 Voronoi 把可破坏对象预切为碎片簇（borrow Chaos Geometry Collection 形态），运行时受力超阈值才「激活」碎片，避免实时切割开销。
- **连通性**：碎片间连接用并查集（union-find）维护 island；断裂事件合并/分裂 island，脱离支撑的 island 转为动态刚体（交 §20 物理）。
- **持久化**：破坏结果（哪些碎片脱离、位置）是 override delta（§11），存档/多人复制；大规模破坏按簇聚合复制（存档 §11、网络复制见 `prism_network`）。

### 16.3 生态（植被/生物分布演替）

- **元胞自动机 / 反应扩散**：植被扩散、火蔓延、种群消长用 CA 或 reaction-diffusion 在空间格点上迭代（纯规则动力学，非 ML）。
- 演替结果改散布密度场（§15）→ 增量重散布；长时间尺度低频 tick。

### 16.4 水文（流域/河网）

- **D8 流向 + flow accumulation**：从高度场算每格水流去向与汇流量，提取河网/流域，驱动河流样条（§14.3）与侵蚀（§16.1）。
- 水体与 `prism_water_engine_design_zh.md` 接缝：世界系统提供流向/水位场，水引擎做表面模拟与渲染。

### 16.5 仿真调度与预算

仿真全走 `prism_tasks` 异步作业，按 tier 定 tick 频率与空间范围（只仿真玩家附近 + 低频全局）；超预算降频或缩范围。仿真的脏传播（§14.2）保证只重烘受影响区域。关闭 `sim` 时世界静态，破坏退化为预断裂切换（无实时解算）。

---

## 17. 全局 SDF 世界场

全局距离场（`sdf` 档位）是「一场多用」的派生缓存（借 UE Global Distance Field / Lumen SDF 形态）：把世界几何体素化为稀疏 SDF，供多个系统共享查询，省去各自维护近似几何。

### 17.1 结构：稀疏 brick + mip 环

```text
以相机为中心的多层 clipmap 环(mip 0 最近最细 → mip N 最远最粗)
每层: 稀疏体素 brick 池(只分配非空 brick) + 间接寻址表
对象: 各自烘焙局部 SDF,合成进全局场(距离取 min)
更新: 相机移动→环滚动(toroidal),只重算新进入的 brick;对象改动→标脏其 brick(§16)
```

### 17.2 一场多用

| 消费者 | 用途 |
|---|---|
| `prism_gi`（Lumen 形态） | 软阴影、AO、远场间接光追踪的几何代理 |
| `prism_render` | 体积雾/粒子的世界碰撞、SDF 软阴影 |
| `prism_physics` | 粒子/碎屑的廉价碰撞查询（距离 + 梯度） |
| gameplay / nav | EQS 式环境查询、导航预筛（可通行性粗判） |
| §15 散布 | 避让查询（不在实体内生成） |

一次烘焙，多方查询，避免每系统各维护一套近似几何，省显存省带宽（design target）。

### 17.3 更新与成本

环滚动只重算边缘新 brick（成本正比于相机移动，§1）；对象改动/破坏（§16）只标脏局部 brick 增量重烘。稀疏分配使空旷区域零成本。关闭 `sdf` 时各消费者回退各自代理几何（GI 用网格、物理用碰撞体），功能降级不崩溃。SDF 是纯派生缓存，丢失可重烘，不入存档。

---

## 18. 四维世界：时间轴与回放

把时间作为世界的第四维（基于 §11 事务日志的事件溯源）。世界的历史不是被覆盖的，而是一条可回溯的事务流。

### 18.1 事件溯源模型

- 世界状态 = 初始确定性态（§6.3）+ 事务日志按序重放（§11.1）。任一时刻 `t` 的世界 = 重放到 `seq(t)` 的 override 集。
- **快照 + 增量重放**：从最近快照（§11.4）正向重放到目标 seq，而非从头，使任意时刻跳转为 O(快照间隔)。

### 18.2 时间倒流 / 回溯

- 每条 delta 有逆操作（§11.4），反向重放实现时间倒流（gameplay 机制：倒带、复活点）、编辑撤销/重做（作者态）。
- 回溯是确定性的（§21）：相同事务集重放得相同世界，支持「回到 5 分钟前」精确复现。

### 18.3 回放调试

- 录制流送 VM 指令流（§7.5）+ 事务日志 + 输入 → 可脱机确定性重放整个会话，定位「某帧为何加载/物化/卡顿」。
- bug 复现：附上事务日志 + seed 即可他机复现，无需「正好那样操作」。

### 18.4 成本与档位

时间轴能力随 `persist` 档位；全量保留历史成本高，可配「保留窗口 + 定期压缩旧事务到快照后丢弃逆操作」（牺牲深度回溯换体积）。gameplay 短时倒带只需小窗口环形缓冲。

---

## 19. 行星尺度一体化（`planet` 档位）

把前述所有能力统一到行星尺度：从轨道俯瞰到地表行走无缝，无加载屏、无精度崩塌（借 Star Citizen / Outerra / MSFS 形态）。

### 19.1 cube-sphere 六面四叉树

- 行星 = 立方体六面投影到球面（§5.4），每面一棵四叉树；四叉树节点即 sector（§5.1），复用全套流送/常驻/持久化。
- **CDLOD-on-sphere**：地形细分（§13.2）沿球面四叉树做连续误差 LOD，裙边消缝，极点无奇异。

### 19.2 i64 定点消抖

全程 §5.2 的 i64 全局定点 + sector-local f32，从轨道（±千万公里）到地表（毫米级）同一坐标系，无 rebasing、无精度衰减、确定性一致（多人/存档前提）。

### 19.3 尺度连续性

- **LOD 连续**：轨道看整星（最粗 LOD + 云/大气）→ 下降逐级细分（CDLOD + VHM 高度页流入）→ 地表全细节（Tier C 散布 + Tier B 建筑 + Tier A 交互），全程屏幕误差驱动，禁 pop（§1）。
- **大气/云/海洋**作为行星级 Tier C 内容与体积场（接 `prism_volumetric_engine_design_zh.md` / `prism_water_engine_design_zh.md`），随 LOD 环流送。
- **多行星/星系**：每行星一个 cube-sphere，星系用更高层 sector 树组织，行星间虚空走极粗 LOD + 定点坐标跨越。

### 19.4 降级

关闭 `planet` 退化为单一平面 sector 网格（常规关卡/开放世界尺度），cube-sphere 与球面 CDLOD 代码编译期移除，零成本。平面与行星共享除「球面投影 + 面间接缝」外的全部流送/常驻/持久化逻辑。

---

## 20. 渲染 / 物理 / GI 接入边界（去 Bevy）

世界系统不复用任何 `bevy_*` 地基（§1 翻转）。它经**两个边界**把内容喂给上层引擎，边界之上世界系统不碰 GPU 资源、不建物理世界、不管光照——只送数据与信号。

### 20.1 两个边界

```text
边界一: GPU 常驻实例流(Tier C,§9/§15)
  世界系统 → proxy 实例 SoA 缓冲 + 代理簇声明(§10)
  prism_render 消费: GPU 剔除/LOD/间接绘制;世界系统不发绘制命令

边界二: ECS 物化实体(Tier A/B,§9.2)
  世界系统 → 调 scene.materialize → 活 ECS 子树(含 transform/mesh/collider 组件)
  prism_render/physics/gi 按既有 ECS 查询消费;世界系统不建渲染/物理对象
```

### 20.2 各上层接缝

| 上层 | 世界系统提供 | 边界 |
|---|---|---|
| `prism_render` | proxy 实例流 + HLOD 簇切换信号 + 物化实体的渲染组件 | 一 + 二 |
| `prism_physics` | 物化实体的碰撞体 + 地形碰撞几何 + 破坏 island 转刚体（§16.2） | 二 + SDF 查询（§17） |
| `prism_gi` | 全局 SDF 场（§17）+ 烘焙 GI proxy + 场景几何代理 | SDF 场 + 一 |
| `prism_nav` | SDF 可通行预筛 + 导航网格派生缓存（§6.3） | SDF 场 |

### 20.3 职责单向、无回依赖

世界系统向上单向推送（数据 + 事件），上层不反向驱动世界系统的真相；上层所需的「何时有什么内容」由流送/常驻决策（§7/§9）决定。渲染侧烘焙的代理资产（§10）、GI 烘焙结果作为派生缓存回存，但经 `prism_asset` 内容包，不形成编译环。去 Bevy 后，边界契约是纯数据结构 + 事件，不绑定任何第三方 ECS/渲染框架。

---

## 21. 确定性与可复现

确定性是存档一致（§11）、多人和解（§12）、PCG 缓存（§14）、时间轴回放（§18）、增量重烘（§16）的共同根基。

### 21.1 确定性的三要素

```text
确定性输出 = f(seed, sector_id, schema/图版本)
  seed:            世界种子(全局) + 派生子种子(sector/图节点)
  sector_id:       空间锚点(§5.3,稳定可复现)
  schema/图版本:    scene schema 版本 + PCG 图版本 + 算法版本
```

同三要素 ⇒ 同 PCG 产物、同散布、同成簇、同 SDF 烘焙，跨机一致、可缓存、可增量。

### 21.2 确定性的实现约束

- **定点优先**：空间坐标用 i64 定点（§5.2），避免浮点跨平台差异；必须用浮点处用确定性约定（固定运算序、禁 fast-math、必要处用软浮点）。
- **稳定排序**：所有空间聚合（成簇、合并、散布采样序）用稳定排序（Hilbert/Morton + 稳定 tiebreak），不依赖哈希遍历序。
- **版本化**：schema/图/算法版本进 content_hash（§6.2），版本变 ⇒ 缓存失效重烘 ⇒ 不会用旧算法产物冒充新结果。

### 21.3 可复现的收益

存档只需 seed + delta（§11.5）；多人只需复制 delta（§12）；bug 可凭日志他机复现（§18.3）；派生缓存可丢可重建（§6.3）。确定性让「成本正比于变化」（§1）成为可能——未变的部分永远算出同样结果，无需重传重存。

---

## 22. 性能工程

世界系统的性能目标：**稳定帧时、成本正比于变化、显存/带宽有预算上限、热路径零回读**。以下数字除 CPU 纯函数外均为 design target（本机无 GPU 与高速存储实测环境，如实标注）。

### 22.1 预算矩阵（随 quality tier 调，§3）

| 预算项 | 约束形态 | 超限行为 |
|---|---|---|
| IO 带宽 | 每帧 MB（design target，接 NVMe 直传能力，§8.3） | 降预取 → 降细节 → 背压 |
| GPU 解压 | 每帧 dispatch 数 / 显存暂存 | CPU 异步解压回退（§8.2） |
| 显存驻留 | proxy/纹理/SDF/几何各占配额 | 驱逐最低优先远景（§7.4） |
| 物化实体 | 每帧 spawn/despawn 实体数 + 列写入量 | 分帧切片顺延（§9.3） |
| proxy 密度 | 每 sector 实例上限 | 降密度淡出（§15） |
| sim tick | 频率 × 空间范围 | 降频 / 缩范围（§16.5） |

### 22.2 核心性能手段

- **预取掩延迟**（§7.3）：IO 延迟藏在相机移动前方，进范围时数据已就绪。
- **缓存可弃可重算**（§6.3）：内存紧张时丢派生缓存，不丢真相，重算而非重存。
- **零回读 GPU-first**（§15.1）：剔除/LOD/密度在 GPU，CPU 无逐实例回读。
- **mmap 零解析**（§8.1）：小结构数据直接当内存视图，无反序列化峰值。
- **列式 SoA + 空间排序**（§5.3）：IO/缓存/SIMD 友好，批处理连续区间。
- **成本正比于变化**：静止场景流送写入与持久写入趋近零（§1、§21）。
- **分帧切片与迟滞**（§9.3、§23）：大物化跨帧，临界距离不抖动，无帧尖峰。

### 22.3 内存与显存布局

- 内存分池：manifest 常驻池（KB×cell）、热内容池（进范围 scene）、延迟回收池（出范围待释放）、暂存池（IO 解压）。
- 显存分配：proxy 实例环、纹理流送池、SDF brick 池、几何池各有配额与 LRU 驱逐，远景先退。
- 内容寻址去重（scene §8.5）：相同资产跨 cell 共享一份，省内存。

> 可实测部分：定点坐标换算、sector 空间编码、需求集求解与预算裁剪、CRDT 合并、脏传播、稳定排序等 CPU 纯函数可单元测 + 基准（§27）。GPU/IO 吞吐、显存占用、帧时为 design target。

---

## 23. 易用性与作者工作流

易用性是世界系统的第三根支柱（与性能、效果并列）。作者面对的是「比屏幕大无数倍的世界」，工具必须让编辑**所见即所得、局部、可逆、无感流送**。

### 23.1 隐式分区，作者无感

作者只摆内容（放对象、刷散布、画样条），**不手动切 cell、不管流送**——sector 归属由坐标隐式决定（§5.1），cell = 作者编辑的那棵 scene（§6.1）。保存时系统按空间自动分片落盘（OFPA 形态，§11.1），改一个对象只写一个小文件，协作无大锁冲突。

### 23.2 所见即所得与数据层

- 编辑器实时流送预览：移动视口即触发真实流送/物化，作者看到的就是运行时所见。
- 数据层开关（§6.4）在编辑器一键切换（昼夜/战前战后/DLC），无需复制世界，所见即所得地编辑各层。
- HLOD/散布/PCG 产物实时预览，改规则即增量重烘可见（§14.2）。

### 23.3 撤销 / 重做 / 时间轴

编辑即追加 override delta（§11），撤销 = 反向重放（§18.2），重做 = 正向重放，天然无限撤销栈且与存档/多人同一套机制。作者可「回到任意编辑时刻」（时间轴，§18）。

### 23.4 热重载

改 scene 基体（prefab）→ 所有实例实时更新，各实例手改字段纹丝不动（scene §13 保全传播）；改 PCG 图 → 下游 sector 增量重烘；改派生算法版本 → 后台重烘缓存，前台无感。live 迭代不重启。

### 23.5 一键打包与渐进披露

- **一键打包**：从作者态世界烘焙出运行态（manifest + scene 包 + 派生缓存 + 空间索引），依赖闭包自动收集（`prism_asset`），缺失即诊断（scene §19.6）。
- **渐进披露**：默认只需摆内容；进阶才碰数据层、PCG 图、HLOD 阈值、流送预算；专家才调 tier/feature。prelude 暴露常用 API，高级能力按需引入（scene §19.4）。
- **错误即诊断**：流送失败、依赖缺失、版本不匹配、预算超限都给可定位的结构化诊断（接 `prism_diagnostic_design_zh.md`），而非静默降级或崩溃。

---

## 24. crate 分层与模块布局 + feature 矩阵

### 24.1 crate 拆分（接 §4）

```text
prism_world            // 门面:重导出三子 crate + prelude + 顶层编排
├── prism_world_stream  // 空间索引 + 流送 VM + IO/GPU 直传触发 + cell manifest
├── prism_world_residency // 三档常驻状态机 + proxy⇄ECS 升降级 + HLOD 驱动
└── prism_world_persist // 事务日志 + CRDT + 快照/时间轴 + 存档（网络经 prism_network 消费 delta）
```

依赖：三子 crate 向下依赖 `prism_scene`/`prism_ecs`/`prism_asset`/`prism_transform`/`prism_math`/`prism_tasks`/`prism_diagnostic`；`stream → residency → persist` 单向（§4）；门面 crate 聚合。

### 24.2 模块布局（示意）

```text
prism_world_stream/
  spatial/     sector 树、Morton/Hilbert、定点坐标(§5)
  manifest/    cell manifest 解析、数据层(§6)
  vm/          需求集、残留引擎、预算、优先级、预取、驱逐(§7)
  io/          异步 IO、GPU 直传、mmap、背压(§8)
prism_world_residency/
  tier/        三档状态机(§9)
  materialize/ proxy⇄ECS 升降级、预算、迟滞(§9)
  hlod/        簇声明、分层代理、切换、物化穿透(§10)
  terrain/     地形多表示接缝(§13)
  scatter/     海量散布 proxy 流(§15)
  sdf/         全局 SDF 场(§17)
  sim/         侵蚀/破坏/生态/水文(§16)
  pcg/         节点图→IR→求值(§14)
prism_world_persist/
  txn/         事务日志、按 sector 分片(§11)
  crdt/        空间合并(§11.3)
  timeline/    快照、反向重放(§11.4、§18)
  save/        存档容器、迁移(§11.5)
```

### 24.3 feature 矩阵

| feature | 默认 | 作用 | 关闭影响 |
|---|---|---|---|
| `std` | 开 | 标准库（文件/线程） | `no_std + alloc` 核仍可用（空间索引/VM 逻辑） |
| `async_io` | 开 | 异步流送 | 同步阻塞加载（小世界） |
| `gpu_stream` | 开* | GPU 直传解压 | CPU 异步解压回退 |
| `hlod` | 开 | 远景代理 | 远景剔除（退化不崩） |
| `persist` | 关 | 事务日志/存档/时间轴 | 无落盘无回放 |
| `pcg` | 关 | 程序化生成 | 手摆 + 预烘资产 |
| `sim` | 关 | 侵蚀/破坏/生态/水文 | 世界静态，破坏退预断裂切换 |
| `sdf` | 关 | 全局 SDF 场 | 各消费者回退代理几何 |
| `planet` | 关 | cube-sphere 行星 | 平面 sector 网格 |
| `vgeo` | 关 | 虚拟几何簇 LOD 流送（§25.1） | 回退离散 LOD 链 |
| `vtex` | 关 | 流送感知虚拟纹理（§25.3） | 常规纹理 mip 流送 |
| `bake_farm` | 关 | 分布式离线烘焙农场（§25.4） | 本机烘焙派生缓存 |

*`gpu_stream` 需平台 capability，无能力时运行时回退（§3）。核心路径（空间索引 + 流送 + 整 cell 加载/卸载）恒开，任何高级 feature 可编译期移除。

---

## 25. 次世代高级功能增补

在流送/常驻/持久化基座（§4–§24）之上，增补一批次世代大世界必备的高级能力。全部为经典数据结构 + GPU 算法形态，**无 AI/ML**、**不依赖 Bevy**、不复制任何引擎源码；均走 feature/档位门控，默认不付费，启用后仍服从「单一真相 + 可弃派生缓存 + override delta」三原则（§6.3）。

### 25.1 虚拟几何：簇化 LOD 流送（`vgeo` 档位，借 Nanite 形态）

逐网格 LOD 在 AAA 密度下仍嫌粗。虚拟几何是统一簇模型（§10）的**对象内部尺度**落地：把单个网格预切为**三角簇（cluster）层级**，以簇为流送与 LOD 单元，达成「屏幕像素级细节、显存只装可见簇」。两级共用的不变量（确定性成簇、屏幕空间误差驱动、禁 pop、派生缓存）在 §10.1 已集中定义一次，本节只讲对象内部尺度的专属机制，不重复通则：

```text
烘焙期: 网格 → meshlet 簇(≈128 三角/簇) → 构建簇 DAG(LOD 层级,父簇=子簇简化合并)
         每簇记: 包围球 / 法锥 / 简化误差 / 父子边界锁(消 LOD 裂缝)
运行期(GPU-driven, 零 CPU 回读):
  persistent-threads 遍历簇 DAG → 按屏幕空间误差选 LOD 切面(cut)
  → 两段遮挡剔除(§25.2) → 可见簇软光栅/硬光栅 → 喂 prism_render(§20 边界一)
流送: 簇按需从磁盘拉(GPU 直传,§8),显存驻留 = 可见切面附近簇,远处簇驱逐
```

- **在统一簇模型中的位置**：虚拟几何落地「对象内部」尺度的连续细节（§13 地形、§15 散布大件、建筑）；跨对象聚合由 HLOD 远景代理（§10）承担。二者是同一套簇模型（§10.1 不变量共用）的两级尺度、正交叠加——远景先 HLOD 代理，进范围后代理内对象走虚拟几何簇流送——而非两套平行系统。
- **成簇的网格特化**：§10.1 确定性成簇在对象内部尺度表现为——簇切分与簇 DAG 构建按 `mesh content_hash` 寻址烘焙（§21），同一网格跨机得到同一 DAG；派生缓存通则（可重烘、不入存档）见 §10.1，此处不重复。
- **边界锁消缝**：相邻簇在不同 LOD 切面时，用「组内锁定边」保证共享边三角一致，无 T-junction、无裂缝（借 Nanite group-boundary 形态）。
- 降级：无 `vgeo` 能力回退传统离散 LOD 链（§15.4），画面退化不崩溃。

### 25.2 两段式遮挡剔除与 HZB（GPU 可见性）

海量实例/簇必须在 GPU 剔除不可见者（§15.1 零回读）。用「上一帧深度 + 两段式」把假阴性降到最低：

```text
Phase 1: 用上一帧 HZB(层级 Z 缓冲)剔除 → 画「上一帧可见」的对象 → 建本帧 HZB
Phase 2: 用本帧 HZB 复测「Phase 1 被剔除」的对象 → 补画本帧新可见者(消除 disocclusion 假阴性)
```

- 剔除粒度：cell → HLOD 簇 → 实例 → 虚拟几何簇，层层递进，每层各自 HZB 测试，只对通过者下探。
- 全程 GPU indirect，CPU 只提交 dispatch 与读回「本帧绘制了多少」的统计（供诊断 §23/§27，不回读几何）。
- 与流送联动：被持续遮挡的 cell 可降级常驻（§9），遮挡信息作为残留引擎（§7.2）的一个优先级输入。

### 25.3 流送感知虚拟纹理（`vtex` 档位，feedback 驱动）

纹理总量远超显存。虚拟纹理把纹理切成页（page），只驻留「本帧着色实际采样到」的页：

```text
GPU 着色采样 → 写 feedback 缓冲(需要哪些页/哪个 mip)
  → CPU/GPU 聚合 feedback → 缺页请求入流送队列(复用 §8 异步 IO + GPU 直传)
  → 页上传到物理页池 → 更新间接表(indirection) → 下帧命中
```

- 一套页流送同时服务：反照率/法线/粗糙度纹理栈、地形 splat（§13）、烘焙 GI 光照图、虚拟阴影图（页式阴影）。
- 与 §8 IO 预算、§22 显存预算共管：缺页风暴时降 mip（临时模糊）而非卡顿，平滑降级（§1 禁 pop 精神延伸到纹理）。
- 虚拟纹理页是派生缓存（§6.3），不入存档。

### 25.4 分布式离线烘焙农场（`bake_farm` 档位）

派生缓存（空间索引/HLOD/SDF/nav/PCG/VT/虚拟几何簇，§6.3）在大世界需海量烘焙算力。烘焙农场把烘焙分布到多机，确定性 + 内容寻址去重：

- **任务分片**：按 sector 分烘焙任务，无依赖的分片并行；有依赖的（如 SDF 依赖几何）按 DAG 定序。
- **确定性缓存复用**：每个烘焙产物按 `f(输入 content_hash, seed, 算法版本)`（§21）寻址，输入未变则命中缓存不重烘——**增量烘焙成本正比于改动**（§1）。
- **去重**：相同输入的产物全农场单份（内容寻址，接 scene §8.5），省存储省带宽。
- **本地回退**：无农场时本机 `prism_tasks` 串行/并行烘焙（§16.5），功能一致、耗时更长（design target）。

### 25.5 世界合成图层与数据层组合（深化 §6.4）

数据层（§6.4）升级为可组合的**世界图层代数**：

```text
激活掩码 = Σ 图层(昼/夜 × 战前/战后 × DLC × 多人实例 × 难度)
内容可见/物化 = 其所属图层集 ⊆ 当前激活掩码
```

- **图层叠加**：同一 sector 多图层叠加（如「战后 + 雪原 DLC」同时激活），内容按图层交并集流送，无需复制 cell。
- **图层级 override**：override delta（§11）可挂在「图层维度」上（某图层专属改动），切图层即切 delta 集，支持「同一世界的多个剧情/难度分支」共存一份存档。
- **编辑器图层可视化**：一键切图层所见即所得（§23.2），图层是 scene 节点 tag，不是平行真相。

### 25.6 流送可视化与世界剖析器（`editor`/诊断）

大世界调优靠「看得见」。提供运行时可视化叠层（接 `prism_diagnostic_design_zh.md`）：

| 叠层 | 显示 |
|---|---|
| 常驻热图 | 每 cell 的 Tier A/B/C 占比与显存/内存占用热图 |
| 流送事件流 | 本帧 load/unload/upgrade/downgrade/prefetch 指令（§7.5）时间线 |
| 预算仪表 | IO 带宽 / GPU 解压 / 显存 / 物化实体各预算的实时占用与超限告警（§22） |
| 剔除可见性 | HZB 剔除前后对象数、虚拟几何簇切面统计（§25.1/§25.2） |
| 持久化差分 | 当前 sector 的 override delta 条目数、存档增长曲线（§11） |
| 确定性校验 | 派生缓存 content_hash 比对、跨机对拍差异定位（§21） |

剖析数据来自 VM 指令流录制（§7.5），可脱机回放分析（§27 测试复现）。

### 25.7 自适应质量与动态预算调速（QoS governor）

帧时是硬约束。调速器按实测帧时动态缩放质量旋钮，守住目标帧率：

```text
每帧: 测 CPU/GPU 帧时 → 与目标(如 16.6 ms)比
  超时: 降旋钮(流送半径↓ → proxy 密度↓ → HLOD 层级↓ → VT mip↓ → sim 频率↓ → 虚拟几何误差↑)
  富余: 升旋钮(反向),带滞后与速率限制,防质量抖动
```

- **分级 QoS**：交互半径内质量优先保，远景先牺牲（§7.2 平滑降级的全局化）。
- **平台自适应**：同一世界在高/中/低端设备经 quality tier（§3）+ 调速器自动适配，无需多份内容。
- 旋钮调整带滞后与最短保持时间（§23），避免画质呼吸式抖动。

### 25.8 无限程序世界与确定性分块（`pcg` + `planet` 协同）

结合确定性 PCG（§14）与 i64 定点（§5.2），世界可「无限」而存储有限：

- **按需生成、从不全存**：未访问的 sector 不存在于磁盘，首次进入由 `f(seed, sector_id, 图版本)`（§21）确定性生成为 proxy/几何（派生缓存），离开可丢弃，重进重算得相同结果。
- **仅存改动**：玩家对程序世界的改动才落 override delta（§11）叠加在重算结果上（§14.4 保全），存档体积正比于改动而非世界尺寸。
- **无缝分块**：相邻程序分块在边界处用共享种子 + 确定性缝合（地形高度、河网、路网跨块连续，§13/§14.3），无接缝突变。

### 25.9 大规模持久化破坏与形变的世界级合并

把 §16.2 的破坏/§13.3 的体素形变提升到世界级持久与协作：

- **按簇聚合的破坏 delta**：大面积破坏（炸塌一片建筑）不逐碎片存，而按 HLOD 簇/sector 聚合为「结构状态 delta」（哪些 island 脱离、聚合变换），存档/多人按簇复制（存档 §11、网络复制见 `prism_network`），带宽/体积正比于可感知变化。
- **形变场的事务化**：体素/SDF 的布尔编辑序列作为 override delta 事务（§11.1），可确定性重放（§18 时间轴：倒带破坏、回放攻城）、可 CRDT 合并（多人同时挖掘，§11.3）。
- **派生几何重烘**：破坏/形变标脏受影响 brick/簇 → 异步重提取网格（§13.3）、重烘 SDF（§17）、重算 nav——全是派生缓存更新，真相只有那串 delta。

---

## 26. 契约、不变量与版本化

### 26.1 与 scene 接缝契约汇总（唯一真相边界）

| 场景 | 世界系统调用 | scene 提供 | 对应节 |
|---|---|---|---|
| cell/Tier B 进范围 | — | `spawn_scene(handle, transform) → id` | §7、§9 |
| cell/Tier B 出范围 | — | `despawn_scene(id)` | §7、§9 |
| proxy 升级 | — | `materialize(proxy) → id` | §9.2、§10.4 |
| proxy 降级 | — | `dematerialize(id)` → 回收 delta | §9.2 |
| 存档/多人 | — | 读写 `PersistentId → OverrideDelta` | §11、§12 |
| 热重载 | — | `SceneReloaded{ instance }` 事件 | §23.4 |

世界系统**从不**绕过这六条直接写 ECS 实体或自建内容序列化——这是单一真相的硬不变量。

### 26.2 核心不变量（debug 档位断言）

- **单一真相**：任何持久可写状态都是 override delta（§6.3）；派生缓存可删除后由 `f(seed, sector_id, 版本)` 重算得位级相同结果（§21）。
- **坐标无抖**：全局坐标恒为 i64 定点，f32 只存在于 sector-local 域，越界只改 i64 做 O(1) 换算，无全局 rebasing（§5.2）。
- **成本正比于变化**：静止、未编辑场景的流送写入与持久写入计数趋近零（§1、§22）。
- **禁 pop**：所有 LOD/HLOD/密度切换带连续过渡（morph/dither/淡入），无硬切换（§1）。
- **档位可降**：任何高级 feature 关闭或 capability 缺失时，核心路径仍完整可用（§3、§24.3）。
- **依赖单向**：`stream → residency → persist` 无编译环，三者不被 scene/ecs/asset 反向依赖（§4）。

### 26.3 版本化与迁移

schema 版本（scene）、PCG 图版本、算法版本、存档容器版本各自独立演进，均进 content_hash（§6.2）。版本变 ⇒ 派生缓存失效重烘（无数据损失）；存档版本变 ⇒ 走 scene §15.4 的 delta 迁移（按 PersistentId 重定位，§11.2）。向前/向后兼容策略：未知字段保留透传，未知事务类型按策略跳过或拒绝（§11.3 冲突策略）。

---

## 27. 路线图 / 基准即规格 / 诚实边界

### 27.1 路线图（里程碑）

| 里程碑 | 内容 | 验收 |
|---|---|---|
| **M0 空间核** | sector 树 + Morton/Hilbert + i64 定点 + 换算（`no_std` 纯函数） | 单元测 + 基准达标（§26.2） |
| **M1 流送 VM** | 需求集 + 残留引擎 + 预算 + 优先级 + 迟滞；接 scene spawn/despawn | 单机整 cell 加载/卸载无 pop、无尖峰 |
| **M2 常驻 + 物化** | 三档状态机 + proxy⇄ECS 升降级 + 分帧切片 | 近交互物化、远景 proxy，预算内稳定帧时 |
| **M3 异步 IO** | `async_io` + mmap 零解析 + 优先级/取消/背压 | 预取掩延迟，无 IO 卡顿 |
| **M4 HLOD + 散布** | 簇声明 + 分层代理 + GPU proxy 流 + 连续切换 | 海量散布远景代理，禁 pop |
| **M5 持久化** | `persist` 事务日志 + 快照 + 存档 + 撤销/重做 | 改动落盘、重载复原、时间轴回溯 |
| **M6 PCG + 地形** | `pcg` 图→IR→求值 + 地形多表示 + 增量重烘 | 规则生成、改图增量重烘 |
| **M7 SDF + 仿真** | `sdf` 全局场 + `sim` 侵蚀/破坏/生态/水文 | 一场多用查询、破坏持久化 |
| **M8 行星 + 多人** | `planet` cube-sphere + CRDT 空间合并（§11.3）；多人经 `prism_network`（另文，消费 persist delta + world 空间查询） | 轨道到地表无缝、多端一致 |

### 27.2 基准即规格（CPU 可实测门槛）

以下为 CPU 纯函数的量化门槛（本机可测，GPU/IO 项见下方 design target）：

| 指标 | 门槛（design，待实测校准） |
|---|---|
| sector 空间编码/解码（Morton/Hilbert） | < 10 ns/次，纯位运算 |
| i64 定点 ↔ sector-local f32 换算 | < 5 ns/次，无堆分配 |
| 需求集求解 + 预算裁剪（万级 cell） | < 0.5 ms/帧（单核） |
| CRDT 合并（千事务） | 确定性，< 1 ms，顺序无关结果一致 |
| 脏传播（PCG/SDF 标脏） | 正比于受影响 sector 数，非全量 |
| 稳定排序（成簇/散布，百万点） | 确定性，跨机位级一致 |

### 27.3 诚实边界与风险

- **无 GPU/高速存储实测**：GPU 直传吞吐、显存占用、proxy 绘制帧时、SDF 烘焙耗时、IO 带宽全为 design target，标注清楚，待目标硬件校准（§8.3、§22）。
- **行星尺度未端到端验证**：cube-sphere + i64 定点在逻辑上消抖，但「轨道到地表全链路无缝 + 多人确定性」需真实工程迭代。
- **仿真规模权衡**：侵蚀/破坏/生态的实时规模受预算约束，大世界多为离线烘焙 + 局部实时（§16.5）。
- **CRDT 结构性冲突**：属性/增删可自动合并，深层结构冲突（同对象被删又被改）需策略或人工介入（§11.3）。
- **去 Bevy 工程量**：渲染/物理/GI 两边界（§20）要求上层引擎均已去 Bevy 且契约稳定，是全局重构的一环。

### 27.4 一句话总结

`prism_world` 不发明世界的内容，它只回答「`prism_scene` 的内容如何被铺到行星尺度、按预算流送驻留、按档位物化、把改动以 override delta 持久化（多人协作经 `prism_network` 消费同一份 delta）」——**单一真相（scene）+ 可弃派生缓存 + override delta 事务日志**，定点坐标消抖、GPU-first 零回读、成本正比于变化、全程禁 pop、去 Bevy、多 crate、确定性可复现，并以 §25 的虚拟几何 / 遮挡剔除 / 虚拟纹理 / 烘焙农场 / 自适应预算把细节、规模推到次世代上限——性能、效果、易用三轴皆为顶级次世代 AAA 级设计目标。
