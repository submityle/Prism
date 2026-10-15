# Prism Asset 顶级次世代 AAA 级资产内核设计方案

> 面向 Prism（脱离 Bevy 后的独立引擎）的 **资产身份 + 句柄/引用计数 + 代际存储 + 依赖图 + 加载器 + 虚拟文件系统 + 事件驱动加载调度 + 按需流送 + 热重载 + 离线烘焙/打包** 内核设计。它是 `bevy_asset` 的自研替代，是渲染（网格/纹理/材质/着色器）、音频、场景/预制体、动画、UI 的统一内容基座；同步 loader/AssetServer 仍是确定性基线，`prism_asset` 已有有限锁外异步完整读/解码/依赖装配，并支持初次异步装配的 labeled 输出原子提交；完整异步 EDL 仍是目标规格。
> 借形态不抄码。借鉴：
> - **Rust 资产人体工学**：Bevy asset（`Handle`/`AssetServer`/`AssetLoader`/`Assets<A>`/`AssetEvent`/标签子资产/`AssetProcessor` 烘焙/热重载）
> - **容器与事件驱动加载**：Unreal IoStore（`FIoChunkId` 容器化分块 I/O）+ Zen/EDL（Event-Driven Loader 异步装配）+ Asset Registry（不加载即可查元数据）+ `.pak`/cook 管线
> - **引用寻址与异步句柄**：Unity Addressables（内容目录 + 异步 `AsyncOperationHandle` + 引用计数释放）、GUID+fileID 的稳定寻址与 `.meta` 导入设置
> - **零拷贝/GPU 直通流送**：DirectStorage + RTX IO + **GDeflate** GPU 解压（对接本仓 `prism_gdeflate_gpu`）、mip/LOD 预算驱动驻留（DOOM Eternal / Decima 形态）
> - **调度/并发**：对接本仓 `prism_tasks`（work-stealing + 优先级 + 背压）承载加载作业图
> 本文为纯经典资产系统/内容管线路线，**不含任何 AI/ML 内容**，不含任何 Unreal/Unity 源码或衍生代码。

- 版本：v0.2（开发中，允许破坏性调整；状态以本文“实施状态”和代码/测试为准）
- 变更记录：本轮补充锁外异步 streaming executor、请求取消竞态门禁、异步 range 短读拒绝、容器单请求 range 预算门禁、异步 streaming gate 终态清理、构建验证字节预算、RenderAsset 延迟准备保留、`AssetServer` 事件/快照到 RenderAsset 的窄桥接、PrepareContext 事务 API 的根级/prelude 导出、CPU fixed-page residency、事务化 GPU fixed-page pool，以及有限范围的异步完整读/解码/依赖装配和独立 I/O/CPU 池切片；初次异步装配现会原子 staging labeled 输出及其声明的普通依赖，避免 payload 跨 await 借用；本轮另交付了已 resident 根资产的窄异步 reload、epoch supersede、source revision 门禁、失败不污染旧 revision，以及已 resident labeled outputs 的有限异步 staging 和回归测试；fixed-page key 现在绑定 source revision，范围检查覆盖末端溢出，失败/取消元数据按预算上限回收；新增 GPU fixed-page replacement 的调用方 completion proof、retiring/reclaim 状态和 CPU residency bridge 的显式 deferred API；新增 RenderApp fixed-page streaming 的 CPU read → owned extract → GPU upload 生产接线、按 revision 的稳定 slot 替换与迟到旧 revision 拒绝、CPU 驱逐后的 deferred release、驱逐前保存 slot/revision 元数据和每帧诊断；继续将 loader/AssetServer 的同步现实、RangeSource 的稳定内容前提、raw 容器能力边界和完整 EDL 目标规格显式分离。
- 本文定位：运行时资产身份、存储、加载、VFS、热重载、流送与消费契约的**目标规格**；已交付能力与目标设计分开记录，不以设计段落冒充实现。
- 当前基线：核心身份/存储/依赖核、同步 `AssetServer`/VFS/loader、bounded scheduler facade、锁外异步 range executor、CPU fixed-page residency、watch/reload 切片和 raw 容器消费切片已存在；渲染侧已有固定页 GPU pool、CPU residency bridge 和由调用方提供 completion proof 的 deferred replacement/reclaim，并新增一个真实的 RenderApp fixed-page 垂直接线（请求 → `read_range` → owned extract → render prepare 上传 → completion-safe reclaim）。它仍不包含 EDL 并行装配、完整流送/GPU 解码、GPU feedback/page-table、按 GPU 反馈驱动的 page pool 淘汰、通用发行容器能力和全资产 ECS/渲染闭环。
- 兼容策略：开发期不承诺旧 API/格式兼容；一旦进入发行，`StableGuid`、容器 ToC、Runtime Manifest 和资产 ABI 必须通过显式 schema 版本与迁移窗口演进。
- 适用引擎： Prism（后 Bevy 时代，独立运行时）
- 关键依赖： `prism_utils`（稳定哈希/句柄位打包/小容器）、`prism_tasks`（异步作业图/优先级/背压）、`prism_reflect`（导入设置/元数据序列化）、`prism_diagnostic`（计数器/trace）、可选 `prism_gdeflate_gpu`（GPU 解压）
- 层级定位： 内容层 L5；上接场景/渲染/音频/动画/UI，下接 `prism_tasks`/`prism_vfs`/平台 I/O
- 明确约束：身份/存储/依赖/句柄核 `no_std + alloc`；当前 `prism_asset` 实际 feature 仅提供 `std`、`ecs`、`file_watcher`；锁外异步 range executor 属于 `std` 下已实现的窄切片，完整异步 EDL、process/pack、GPU streaming 和 retain-cache 等仍属于目标能力分层，不是当前已发布 feature；工作区 `forbid(unsafe_code)`；**不依赖任何 `bevy_*` crate**
- 烘焙/打包归属： 离线烘焙、缓存键、DDC/CAS、Revision、打包与 Patch 的唯一持有者是 `Prism Asset Build`（`docs/prism_asset_build_design_zh.md` §2.7/§4.6/§10/§14）；本 crate 只提供运行时消费契约（§13）与进程内重烘适配。派生数据分类与登记见 `docs/prism_derived_data_contract_zh.md`

### 实施状态（2026-10-16）

以下状态只记录仓库中已存在、可运行并有测试覆盖的能力，不把设计文字当作实现。容器安全边界同时以实现的读路径为准：挂载时校验的是当时读取到的完整字节；range source 在 reader 生命周期内必须保持不可变；整 payload `read_all` 再校验 entry digest，局部 `read_range` 不提供密码学子范围认证（详见 §10.5）。

本轮异步调度切片补充了直接 `path#label` 请求，以及 reload 候选对新普通依赖和 sibling 新 label 的递归读取/解码/原子提交：调度器复用源资产的锁外读取、解码、依赖等待和原子 labeled 提交，成功返回精确子资产句柄；源资产已驻留时走同一解析路径，未知 label 终态失败但不污染已成功的源 revision。该能力不等同于完整 EDL，range-backed assembly、可抢占取消和 GPU 上传仍按下文状态单独验收。

- **已完成**：`prism_asset` M0/M1 核（稳定身份、句柄、代际存储、依赖图、加载状态、错误注册、loader 选择与同步 `AssetServer`/VFS）；`prism_asset_container` 的 v1 原始 payload 容器与 v2 独立 Chunk Table 容器读写、严格目录/Chunk Table/哈希/对齐/填充校验、Raw 与确定性 raw-DEFLATE、按 payload 范围读取与优先级 overlay；ChunkPlanner/Writer 的起始对齐约束、Runtime Manifest 的 manifest/chunk priority 传递、同名容器 remount 替换旧 mount，以及基于 `RangeSource` 的文件按范围读取（含 positional I/O、并发读取、短读/截断/篡改门禁）均已有测试覆盖。
- **部分完成**：M2 已有同步装配之上的 bounded scheduler facade，以及 `AsyncStreamIoExecutor`；后者使用现有 `prism_tasks::TaskPool`，在 controller 锁外真实调用 `AssetSources::read_range`，覆盖提交、预算拒绝、失败释放、读前取消和读中协作取消。当前还交付了有限的 M2.1 锁外完整读/解码/依赖等待/提交切片：不同路径可并行进入 source read，I/O 与 CPU 使用独立池，父资产只在依赖成功后发布，失败和环会终态失败；`max_in_flight` 现在限制并发 assembly 执行槽位，父资产等待依赖时释放槽位，依赖按逐项推进进入有界优先级队列后再恢复父装配，根请求仍在容量满时快速失败，队列溢出会显式终态失败而不是无界累积。初次异步 load 现在会为主资产的 labeled 输出建立稳定身份、校验类型、解析其声明的普通依赖，并把主资产、label graph 和 retained closure 一次性提交；重复标签、类型不一致、依赖失败和环不会发布部分 revision。异步 reload 目前可递归读取并暂存新普通依赖、label 对应源和 sibling 新 label 的依赖，所有候选在一次 revision-local commit 中发布；重复 label、类型变化、依赖失败、cycle、source overlay 变化、旧 epoch supersede 和超出候选上限都会保留旧值、旧图和旧 retained closure。range-backed assembly、可抢占取消和 GPU 上传闭环仍未完成。直接请求 `path#label` 已复用源资产的异步装配并返回精确 labeled handle，未知 label 会终态失败且不污染已成功的源 revision。
- **部分完成**：Asset Build 已有确定性串行 `BuildGraph`/`Processor`、ArtifactKey、BLAKE3 CAS、持久化本地 DDC 描述符索引、本地 manifest revision、带 `TransactionId` 的 append-only publication journal、启动恢复和显式 reconcile；`BuildExecutor` 已把图执行、DDC/CAS 校验和、缓存损坏重建及事务发布接成单机核心闭环，并提供受限的 ready-wave 并行 DDC/CAS 处理，按稳定 JobId 顺序应用缓存、Manifest staging 和发布结果；RuntimePackage 现在以来源 Build Manifest hash 绑定到同一 Revision，且 publication 会拒绝缺失 `AssetId -> StableGuid` runtime route 的包；RevisionStore 还提供了基于已提交 Manifest roots、带 grace period 且遇到 pending transaction 自动延期的本地 CAS 可达性 sweep；本地 DDC 另提供显式 `DdcCapacity`/`evict_to_capacity` 容量回收，但不是完整分层策略；PNG→canonical RGBA8 artifact→BuildExecutor→CAS/Manifest 的真实纹理切片、可复用的 Recipe/Importer 主链和调用同一核心的 `prism_asset_build_cli`（`plan/import/build/inspect`）已落地，CLI 另有受限 `package`/`validate` 发行命令，并可通过 `package --signing-key-file` 生成、`validate --verification-key-file` 验证 Ed25519 `release.sig`，但只支持单 PNG 垂直切片，不能视为通用 Cook；仍缺远程 CAS、runtime texture loader 的完整发行接线、真实跨进程测试、非 Unix 协调和完整磁盘故障恢复；底层 BuildExecutor 并行路径已有串并行等价、限流、缓存命中和失败不发布测试，但尚无大规模性能证据。
- **部分完成**：M5 容器已有 v1 raw/per-payload Deflate reader/writer，以及 v2 独立 Chunk Table reader/writer；v2 的每个 Chunk 独立使用 Raw 或 raw-DEFLATE，目录记录 `chunk_id`/`raw_offset`/原始长度和 hash，Chunk Table 记录物理 offset、stored/raw 长度、codec 和 Chunk hash，读写均执行严格 ToC/Chunk Table/范围/哈希/对齐/零填充校验；Raw Chunk 支持物理范围合并读取，Deflate Chunk 支持完整 Chunk 解码后切片。Runtime Manifest v3、overlay、`ManifestAssetReader`、ChunkPlanner 驱动的物理布局和同名换包拒绝已有测试/接线。容器版本闭集为 `{1, 2}`；block-level 压缩索引、加密、签名信任根、Provenance、GPU 解压、远程 CDN/DirectStorage 调度和完整发行协议仍未实现。当前 runtime 新增受 `RangeBatchPolicy` 限制的确定性批量 range 调度：同一容器内的重叠/近邻请求会合并为有界物理读，单请求不得超过物理读上限，`ManifestReader` 会在 manifest 身份校验后按容器分组并保持调用方结果顺序；无序、重复、重叠、v2 `raw_offset`、溢出、源边界和源失败均有回归测试。它仍不等价于完整 EDL；range-backed 装配、GPU feedback/page-table 和通用 Cook/发行闭环仍未接入。
- **部分完成**：M3 已有文件 watcher/去抖事件模型、`Arc<AssetServer>::watch` 托管后台消费入口、可由调用方驱动的 `AssetServer::apply_change_batch` 接线、依赖闭包重装配计划、失败依赖门禁及 labeled 子资产候选图校验；source reload 现在会传播到 labeled 输出的消费者，且有 promotion journal 与 tombstone cycle collector 回归，避免 source→label→source 形成永久强引用环。成功 reload 移除旧 label、旧 label 外部 handle 的失败/身份语义、provisional label 的临时 `interned`/catalog 回滚，以及候选失败后 foreign label promotion 的回滚均已有回归；完整 labeled 生命周期和更大规模 staging 交错验证仍未闭环。锁外异步 reload 现在可递归加载新普通依赖、label 对应源和 sibling 新 label 的依赖闭包，并在一次 revision-local commit 中发布；保留旧值/旧图/旧 retained closure，重复 label、类型变化、cycle、source overlay 变化、旧 epoch supersede、读取/解码失败和候选上限超出均拒绝发布，移除 label 转为失败。完整 EDL、range-backed assembly、可抢占取消或 GPU 上传闭环仍未完成。M4 的**流送预算决策层**已提供 CPU/GPU/staging 预算、优先级滞回、取消/失败/驱逐状态机和确定性驱逐候选排序；新增 `PageResidency` 以固定页大小真实调用 `AssetSources::read_range`，`PageKey` 绑定 source revision，页范围在请求阶段同时校验偏移和末端溢出，终态失败/取消元数据按显式上限保留并确定性回收，提供 CPU 页预算、in-flight 背压、失败/取消释放和确定性 eviction。同步 `StreamIoExecutor` 与新增 `AsyncStreamIoExecutor` 仍会真实调用 `AssetSources::read_range`，后者在 controller 锁外执行并验证取消/失败预算释放，且在成功、取消、失败等终态清理请求 gate，避免完成请求在 gate 表中无界累积；这些仍不是解码或 GPU 上传闭环。压缩/加密容器、DirectStorage 形态队列和完整 GPU 解压仍未实现；`prism_gdeflate_gpu` 现在提供真实 adapter/device probe、显式失败原因、实际 de-interleave 与 stored-block GPU 快速路径，仍由 CPU 处理 Huffman/完整 GDeflate；M5/M6 的 BakeService、ECS/渲染/编辑器集成、发行闭包/patch/DLC、签名与 Provenance 仍未完成。
- **部分完成**：渲染侧通用 `RenderAsset` 阶段已经真实执行 `Assets<T> -> AssetEvent -> ExtractedAssets -> prepare -> RenderAssets<T>`，并覆盖 Mesh/Image 的 GPU buffer/texture 上传；本轮修复了 `RetryNextUpdate` 丢失 pending source 的问题：最新副本会保留到下一次 prepare，removal 优先于 retry，重复 extraction 按 id 合并，且无需新的 source event 即可重试成功。现在还交付了窄桥接：`AssetServer` 在同一把锁内原子 drain typed arena 事件并复制 Added/Modified source，Render extract 复用原始代际 `AssetId`，Removed/Unused/Failed/LoadedWithDependencies 不复制 source；无 server resource 时仍走 `Assets<T>` 兼容路径，并有真实 null-RHI prepare/release 回归。新增 `PrepareContext` 提供可回滚的 buffer 分配预算、协作取消检查、Retry 失败恢复和显式替换卸载；本轮将 `GpuPrepareBudget`、`PrepareCancellation`、`PrepareContext`、`PrepareContextError` 和 `prepare_extracted_assets_with_context` 正式暴露到 `prism_render_app` 根 API 与 prelude，供实际渲染集成调用。另新增固定页 streaming 的真实 RenderApp 接线：主世界有界 `read_range`、owned payload extract、render prepare 的真实 buffer upload、completion-safe replacement/reclaim，以及按 revision 的提取去重、驱逐追踪和每帧诊断；测试覆盖 Null backend 的一次 `App::update` 全链路。它仍不是完整纹理/解码/GPU feedback/page-table/DirectStorage 流送闭环。依赖 ready 触发的生产级 GPU feed、完整纹理资源回滚、上传队列时间预算和端到端全资产 residency 仍未接入。

因此本文下方路线图仍是目标规格；任何“已支持”表述均以上述实现状态和对应测试为准。

Asset Build 与本运行时资产内核的状态分开验收：构建/缓存/Revision/打包的实现矩阵及测试证据见 [`prism_asset_build_design_zh.md`](prism_asset_build_design_zh.md) 的“可执行验收矩阵”。该矩阵不替代本节运行时 EDL、热重载、流送和容器消费的独立状态说明。

---

## 目录

1. 设计哲学与目标
2. 参考产品取舍
3. 档位化（capability / quality tier）
4. 分层架构
5. 核心模型：身份（AssetId / StableGuid / AssetPath / AssetKey）
6. 句柄与生命周期（strong / weak / soft / 引用计数 / 延迟回收）
7. 代际存储 `Assets<A>` 与类型擦除注册
8. 依赖图与递归加载状态
9. 加载管线：AssetServer / AssetLoader / 事件驱动装配
10. 虚拟文件系统与数据源（VFS / Reader / 容器）
11. 按需流送与 GPU 直通解压
12. 热重载与依赖失效传播
13. 离线烘焙 / 导入 / 打包管线
14. 事件、变更与世界集成
15. 可观测性与诊断
16. 确定性保证
17. 性能工程
18. 易用性与迁移策略
19. crate 分层与模块布局
20. 契约、不变量与版本化
21. 路线图（M0–M6）与基准即规格
22. 诚实边界与风险
23. AAA 高级功能增补
24. 基准即规格：量化目标与验收门槛
25. 测试与验证策略
26. API 人体工学深化（派生宏 / 系统参数 / prelude）

---

## 1. 设计哲学与目标

资产系统是「磁盘上的字节」接到「运行时的 GPU 资源与 ECS 组件」之间的唯一桥梁。渲染要网格/纹理/材质/着色器，音频要波形，场景要预制体，动画要骨骼/剪辑——它们都不应各自写一套「找文件→解码→建资源→引用计数→热重载」。`prism_asset` 把这件事统一成**一套身份、一套句柄、一套异步装配协议**：上层只持 `Handle<A>`，不关心它在磁盘、在网络、在 pak、还是正从 SSD 直灌显存。

**一句话定位**：`prism_asset` 是 Prism 的「内容真相层」——稳定的资产身份 + 引用计数句柄 + 代际存储 + 依赖感知的异步加载/流送/热重载/烘焙，一次 `load()`，处处可用（渲染驻留、场景实例化、脚本引用、编辑器热改）。

四条总目标（按权重）：

1. **性能**：加载走 `prism_tasks` 作业图并行解码，依赖按拓扑装配（EDL 形态，无「加载完一个再排下一个」的串行栅栏）；容器化分块 I/O 把「上千个小文件 open/read」压成「少数 pak 的大顺序读」；流送按预算驱动 mip/LOD 驻留，可选 GPU 直通解压绕过 CPU 拷贝；身份为 `Copy` 值类型、句柄引用计数 lock-free。
2. **效果（能力）**：子资产标签、软/硬引用、依赖闭包就绪事件、热重载失效传播、内容哈希增量烘焙、平台相关压缩、单文件资产包内嵌导入设置、容器打包——支撑编辑器迭代、运行时流送、发行打包等 AAA 刚需。
3. **易用**：`server.load::<Mesh>("models/hero.gltf#Mesh0")` 一行拿句柄；`assets.get(id)` 取值；`get_mut` 自动发变更事件；loader 实现一个 trait；软引用显式区分「现在就要」与「将来可能要」。
4. **可移植 + 档位化**：身份/存储/依赖/句柄核 `no_std + alloc`，可跑在服务器/工具/受限平台；异步 I/O、热重载、烘焙、打包、GPU 流送按 feature 裁剪，移动端可只带「只读 pak + 最小流送」。

非目标：不做具体格式解码（glTF/FBX/PNG/KTX2/WAV 等归 `prism_asset_import`，本 crate 只定义 loader 协议与装配）；不做场景图语义（归 `prism_scene`）；不替代操作系统文件系统（经 VFS 薄封装隔离）；不保证流送命中率在任意访问模式下最优（预算与预取是启发式）。

---

## 2. 参考产品取舍

| 产品 | 吸收 | 规避 |
|---|---|---|
| Bevy asset | `Handle`/`AssetServer`/`AssetLoader`/`Assets<A>`/`AssetEvent`、标签子资产、`AssetProcessor` 烘焙、热重载 | 每类型一个 `Assets<A>` 资源的 ECS 耦合、字符串路径散列开销、处理器确定性弱点 |
| Unreal IoStore / Zen / EDL | **单文件自包含资产包**（身份/设置/载荷同文件，§13.1）、容器化分块 I/O（`FIoChunkId`）、事件驱动异步装配、Asset Registry 免加载查元数据、cook 管线 | UObject/GC 耦合、UObject 序列化反射格式包袱、全局加载锁历史债 |
| Unreal `.pak` / bulk data | 打包容器、按需 bulk 流送、挂载优先级 | 平台专有压缩器绑定 |
| Unity Addressables | 内容目录（catalog）、异步句柄、引用计数释放、远程/本地内容分发 | C#/GC、运行时反射装配 |
| Unity GUID + `.meta` | 稳定 GUID 寻址（路径可改身份不变）、导入设置随资产走 | 每文件旁车 `.meta` 的目录污染与移动脱钩（改采 UE 单文件内嵌，§13.1）、编辑器数据库中心化、import 不确定性 |
| DirectStorage + RTX IO + GDeflate | NVMe 批量队列、GPU 解压直灌显存、绕过 CPU 中转 | 厂商/平台绑定（经 feature + `prism_gdeflate_gpu` 隔离） |
| DOOM Eternal / Decima 流送 | 预算驱动 mip/LOD 驻留、距离/可见性预取、无卡顿换入换出 | 引擎专有数据布局 |

综合：**Bevy 人体工学（Handle/Server/Loader/Assets）+ Unreal 容器化分块 I/O 与事件驱动装配（EDL）+ Unity Addressables 引用寻址与异步句柄 + UE 单文件资产包（身份/导入设置内嵌、无旁车） + DirectStorage/GDeflate GPU 直通流送** 五支柱，叠加 **内容哈希增量烘焙 + 平台档位压缩 + 依赖失效热重载** 的 AAA 能力层，全部走 feature/档位门控，默认只付「只读加载 + 引用计数」的最小成本。

---

## 3. 档位化（capability / quality tier）

资产系统必须「同一套代码，从单元测试跑到 AAA 发行包」。用 feature 把能力切成互相独立的档位，默认档只付最小成本：

| 档位 | feature | 内容 | 典型场景 |
|---|---|---|---|
| **核（Core）** | 默认 `no_std+alloc` | 身份 / 句柄 / 代际存储 / 依赖图 / 加载状态 / 事件 | 服务器、工具、确定性测试 |
| **标准 I/O** | `std` + `async_io` | VFS 文件读、`AssetServer`、异步加载作业图、loader 注册 | 开发期桌面运行 |
| **热重载** | `hot_reload` | 文件系统 watch + 依赖失效传播 + 重装配 | 编辑器 / 快速迭代 |
| **处理** | `process` | 资产包读取契约（导入设置内嵌）+ 进程内重烘适配（离线烘焙/缓存键/DDC 见 `prism_asset_build_design_zh.md`） | 编辑器预览 / 运行时局部重烘 |
| **打包** | `pack` | 容器化分块 I/O（pak/IoStore 形态）+ 压缩 + 挂载优先级 | 发行包 |
| **GPU 流送** | `gpu_stream`（+ `prism_gdeflate_gpu`） | DirectStorage 形态批量队列 + GPU 解压直灌 | 主机/高端 PC |

**运行时质量档**（与 feature 正交，运行期可调）：`StreamingBudget`（显存/内存上限、每帧 I/O 带宽、最大并发解码数、预取半径），分 `Minimal / Balanced / Ultra` 预设，移动端收紧、主机放宽。档位切换只改预算与调度参数，不改身份/句柄语义。

设计铁律：**上层代码对档位无感**。无论资产来自散文件、pak、还是网络，无论是否 GPU 直通，`server.load()` 与 `assets.get()` 的签名与语义不变；档位只影响「字节怎么到」，不影响「身份是谁、句柄怎么活」。

---

## 4. 分层架构

```
                      ┌───────────────────────────────────────────┐
消费方（渲染/音频/场景/动画/UI/脚本/编辑器）
                      └───────────────┬───────────────────────────┘
                                      │ Handle<A> / AssetServer / Assets<A>
┌──────────────────────────────────────────────────────────────────┐
│ L5  装配与生命周期                                                  │
│   AssetServer（path↔id interning、load 分发、就绪聚合）             │
│   Assets<A>（代际存储 + 变更事件 + remove_unused）                  │
│   DependencyGraph（拓扑装配 + 环检测 + 失效传播）                   │
├──────────────────────────────────────────────────────────────────┤
│ L4  调度                                                            │
│   LoadScheduler（EDL 事件驱动作业图，跑在 prism_tasks 上）          │
│   StreamingManager（预算驱动 mip/LOD 驻留 + 预取）                  │
├──────────────────────────────────────────────────────────────────┤
│ L3  装配协议                                                        │
│   AssetLoader trait（bytes + 上下文 → 资产 + 依赖声明）             │
│   AssetSaver / Processor（离线烘焙，feature=process）               │
├──────────────────────────────────────────────────────────────────┤
│ L2  I/O 抽象                                                        │
│   AssetSource / AssetReader（VFS：fs / memory / embedded / pak）    │
│   ContainerIo（分块容器，feature=pack）/ GpuTransfer（gpu_stream）  │
├──────────────────────────────────────────────────────────────────┤
│ L1  身份与地基                                                      │
│   AssetId / StableGuid / AssetPath / Handle / HandleProvider        │
│   prism_utils（哈希/位打包）· prism_tasks（作业）· prism_reflect     │
└──────────────────────────────────────────────────────────────────┘
```

规则：依赖严格向下；L1 身份核不知道 I/O、调度、容器存在（可在 `no_std` 下独立编译与测试）。每一层只经上一层的 trait 对象/句柄交互，容器/GPU/网络都是 L2 的可替换后端。这是「身份与装配解耦」的关键——重铸 I/O 后端不动身份语义，重铸调度器不动 loader 协议。

---

## 5. 核心模型：身份

重构要点（不保留旧 API）：旧 M0 的 `AssetId` 只有「代际槽位 id」一个维度，无法表达「跨构建/跨网络稳定」的身份，也无法在资产尚未加载时引用。新模型把身份拆成**三层**，各司其职：

### 5.1 三层身份

```rust
/// 运行时槽位身份：代际 (index, generation)，Copy、紧凑、仅进程内有效。
/// 用于 Assets<A> 的 O(1) 寻址；槽位回收后 generation+1，旧 id 自动失效。
pub struct AssetIndex { index: u32, generation: u32 }

/// 跨构建/跨网络稳定身份：128-bit 内容/路径派生 GUID。
/// 存档、网络复制、cook 清单只认 StableGuid，不认 AssetIndex。
pub struct StableGuid(u128);

/// 类型化运行时 id：AssetIndex + 类型标签（零成本 PhantomData<fn()->A>）。
pub struct AssetId<A: ?Sized> { index: AssetIndex, marker: /* ... */ }

/// 类型擦除运行时 id：依赖图、untyped 句柄、事件广播用。
pub struct UntypedAssetId { index: AssetIndex, type_id: AssetTypeId }
```

关键改动：`UntypedAssetId` **新增 `AssetTypeId`**（类型擦除存储的类型标签），解决旧设计里「untyped id typed::<错类型>() 静默命中错 arena」的隐患——重铸后类型不符在类型擦除存储层直接 miss 且可诊断。

### 5.2 稳定 GUID 与路径

- **`StableGuid`**：资产的跨构建身份。源资产由「规范化路径 + 子标签」派生（路径可改需经重映射表，不静默变更，参照 Unity GUID 的「路径变身份不变」但以内容清单承载，避免编辑器数据库中心化）；烘焙产物由**内容哈希**派生（内容相同则 guid 相同 → 增量烘焙/去重的基础）。哈希算法与规范化规则一经发行即冻结（见 §19 契约）。
- **`AssetPath`**（保留并增强）：`path` + 可选 `#label` 子资产 + 可选 `source://` 前缀（多数据源，如 `remote://cdn/…`、`embedded://…`）。新增 `AssetPath<'a>` 的借用形态与 `OwnedAssetPath`，热路径查表零分配（借用），存储时才 owned——规避旧设计「每次 load 都 `String` 分配」。

### 5.3 身份关系

```
AssetPath ──(interning, AssetServer)──▶ StableGuid ──(load 完成)──▶ AssetIndex ──▶ AssetId<A>
   "models/hero.gltf#Mesh0"               (稳定)                    (进程内槽位)      (类型化)
```

- `AssetServer` 维护 `AssetPath ↔ StableGuid ↔ AssetIndex` 三向 interning 表：同一路径多次 `load` 返回同一 `AssetIndex`（去重），不同构建同一资产 `StableGuid` 一致（存档可移植）。
- `AssetId<A>` 仍是 `Copy`/`Send`/`Sync`、可塞进 ECS 组件、可跨 render/main world；生命周期由 `Handle` 引用计数承担，不是 GC。

---

## 6. 句柄与生命周期

句柄回答「这份资产现在是否该活着」。重构后区分三种引用强度，并把「释放」从「立即删」改成「延迟回收」，避免跨线程/跨帧的释放竞态。

### 6.1 三种引用强度

```rust
/// 强句柄：持有即保活。最后一个强句柄 drop → 资产进入可回收集合。
pub struct Handle<A: ?Sized> { /* Arc<HandleInner> + 类型标签 */ }

/// 弱句柄：不保活，仅观察。upgrade() 仅在仍有强句柄时成功。
pub struct WeakHandle<A: ?Sized> { /* Weak<HandleInner> */ }

/// 软引用（新增）：不保活、不触发加载，只记稳定身份 + 期望类型。
/// 场景/预制体里「将来可能要」的引用，显式 resolve() 才变 Handle。
pub struct SoftHandle<A: ?Sized> { guid: StableGuid, /* 类型标签 */ }
```

- **强 / 弱**沿用 `Arc`/`Weak`，lock-free、`no_std+alloc`。
- **软引用**是对标 Unreal `FSoftObjectPtr` / Unity Addressables `AssetReference` 的关键新增：场景里成千上万个「引用但未加载」的资产只存 `StableGuid`（16 字节 + 类型标签），不占存储槽、不触发 I/O，直到显式 `server.load_soft(&soft)` 才装入。这是「打开大世界场景不卡死」的前提。
- `UntypedHandle` 保留，承载类型擦除保活（资产热重载、通用缓存）。

### 6.2 延迟回收（deferred drop）

旧 `remove_unused` 在调用点同步删除并发事件。问题：渲染世界可能正持有该资产的 GPU 句柄，主世界一删就悬垂。重构为**两阶段回收**：

1. 最后一个强句柄 drop → `HandleInner` 的 `Drop` 把 `AssetIndex` 推入无锁 `ReleaseQueue`（MPSC，`prism_utils` 提供），**不立即动存储**。
2. 每帧 `Assets::<A>::collect_releases()` 在确定的回收点排空队列：此时再校验「确实无强引用 + 无 pending 加载 + 下游依赖已松手」，才真正 take 槽位、bump generation、发 `Removed` 事件。可配置「宽限帧数」让跨世界消费方有时间松手。

好处：释放时机**确定且集中**（便于与渲染帧同步、便于诊断），槽位回收与 GPU 资源回收可对齐同一回收点。

### 6.3 加载态与句柄解耦

句柄一经 `load` 即刻返回（不阻塞），此刻资产可能还在加载。`assets.get(id)` 在就绪前返回 `None`；要等就绪用事件（§8/§14）或 `server.load_state(id)` 轮询。这是异步加载的核心人体工学：**句柄立即有，数据稍后到**，对标 Unity Addressables 的 `AsyncOperationHandle` 但不强制 await。

---

### 6.4 按需加载与自动卸载

资产生命周期的两端——**何时装入、何时卸出**——都应「默认按需、无需手动」。`prism_asset` 把两者统一成**句柄引用计数 + 流送预算 + 可选保留缓存**三层策略，上层不写一行「卸载」代码也不漏内存。

**① 按需加载（lazy / on-demand）——四个粒度，逐级更懒：**

| 粒度 | 机制 | 何时触发 I/O | 典型场景 |
|---|---|---|---|
| 显式懒加载 | `server.load(path)` → 句柄立即有、数据后到（§6.3） | 调用即排加载作业 | 代码主动要的资产 |
| 软引用按需 | `SoftHandle` 零 I/O 持有，`load_soft()` 才装（§6.1） | 真要用时才触发 | 大世界/预制体里的海量「可能要」引用 |
| 预算分批 | 软引用实例化按 `每帧 N 个` 分批点亮（§23.2） | 分帧摊销 | 打开大场景不一次性爆 I/O/内存 |
| 层级流送 | 纹理 mip / 网格 LOD / 虚拟页按 `read_range` 增量拉取（§11） | 预算 + 可见性驱动换入 | 高清资产只驻留当前需要的层级 |

优先级贯穿全链：`LoadPriority`（`Immediate / High / Streaming / Prefetch`）映射到 `prism_tasks` 优先级——相机前方插队、预取让路（§9）。首帧只等 `Immediate` 闭包就绪即可玩，其余后台渐进流入（§23.8）。

**② 自动卸载（automatic unload）——两条正交机制：**

- **引用计数回收（整资产）**：最后一个强句柄 `drop` → 入 `ReleaseQueue` → 下个回收点 `collect_releases` 校验「真无强引用 + 无在途加载 + 下游松手」后 take 槽位、发 `Unused`→`Removed`（§6.2/§14）。**无 GC、确定性、无需手动 free**；弱句柄/软引用不保活，自然不阻止卸载。
- **流送驱逐（层级降级）**：`StreamingManager` 按 `StreamingBudget`（显存/带宽/并发上限）给每个可流送资产的目标层级打分，超预算时**驱逐低分层级**（高清 mip/LOD 卸出，保留低清兜底），句柄与 `AssetIndex` 不变——这是「降级驻留」而非「整资产删除」，相机转回来再换入（§11.1）。

**③ 可选保留缓存（retention / 防抖动）**：纯引用计数在「频繁 load→drop→reload 同一资产」时会抖动（刚卸就要）。`feature = "retain_cache"` 提供可配置保留策略，介于 `Unused` 与 `Removed` 之间给一个挽留窗口：

```rust
pub enum RetentionPolicy {
    Immediate,                 // 无强引用立即可回收（默认，最省内存）
    KeepAlive { frames: u32 }, // 宽限 N 帧，吸收瞬时抖动
    Lru { budget_bytes: u64 }, // 字节预算 LRU，热资产常驻、冷资产按最久未用淘汰
    Pinned,                    // 显式钉住，不自动卸载（启动必备/UI 常驻）
}
```

`Unused` 事件正是留给保留缓存的「挽留点」：缓存层可在此决定续命或放行。`pin`/`unpin` 提供显式覆盖（如 boot 资产、字体、兜底资产常钉住）。所有策略都是预算/档位门控——移动端用 `Immediate` 最省内存，PC/主机可配 `Lru` 吸收抖动。

> 一句话：**强句柄决定「整资产是否存活」，流送预算决定「存活资产驻留到哪一层级」，保留缓存决定「失去最后引用后挽留多久」**——三者皆自动、皆预算驱动，上层只管持句柄与声明优先级。

---

## 7. 代际存储 `Assets<A>` 与类型擦除注册

### 7.1 代际稠密竞技场（重铸）

沿用「代际槽位 + 空闲链 + 稠密值」，但重构三处：

- **值与元数据分离**：槽位存 `generation + State`，`State ∈ {Empty, Pending, Loaded(A), Failed}`。旧设计只有 `Option<Entry>`，无法表达「槽位已分配、句柄已发、数据未到」——异步加载必需。
- **弱引用探活改为队列驱动**：旧 `remove_unused` 每次 O(n) 扫描所有槽位探 `strong_count==0`。重构为 §6.2 的 `ReleaseQueue`，回收 O(待回收数) 而非 O(总资产数)——大场景几万资产时这是帧时间差异。
- **可选 `handle_provider` 外置**：id/句柄的铸造从 `Assets<A>` 抽到 `AssetServer` 的 `HandleProvider`，使「先发句柄（path→id）、后填数据（load 完成）」成为可能；`Assets<A>` 退化为纯存储，不再自己 mint id。

```rust
pub struct Assets<A> {
    slots: Vec<Slot<A>>,          // generation + State<A>
    free: Vec<u32>,               // 空闲槽位，LIFO 复用
    len: usize,
    events: Vec<AssetEvent<A>>,   // 每帧 drain
}
enum State<A> { Empty, Pending, Loaded(A), Failed(AssetErrorId) }
```

API（重构后，不保留旧 `insert` 自铸 id 的形态）：

```rust
impl<A: Asset> Assets<A> {
    fn reserve(&mut self, index: AssetIndex);           // 预留 Pending 槽位，由 server 调
    fn fulfill(&mut self, index: AssetIndex, value: A); // 填入数据 → Added/Modified
    fn fail(&mut self, index: AssetIndex, err: AssetErrorId);
    fn get(&self, id: AssetId<A>) -> Option<&A>;        // Loaded 才 Some
    fn get_mut(&mut self, id: AssetId<A>) -> Option<&mut A>; // 发 Modified
    fn remove(&mut self, id: AssetId<A>) -> Option<A>;  // 发 Removed
    fn collect_releases(&mut self) -> usize;            // 排空 ReleaseQueue
    fn iter(&self) -> impl Iterator<Item = (AssetId<A>, &A)>;
    fn drain_events(&mut self) -> Vec<AssetEvent<A>>;
}
```

### 7.2 类型擦除注册 `AssetTypes`

依赖图、热重载、通用诊断需要「不知道具体类型也能操作资产」。新增 `AssetTypes` 注册表：`AssetTypeId → dyn AssetStorage`，提供类型擦除的 `remove / contains / collect_releases`。每个 `Asset` 类型注册时登记一次（`register_asset::<A>()`），`AssetTypeId` 由 `prism_reflect` 的 `StableTypeId` 派生，保证跨构建一致（存档/依赖图可移植）。

### 7.3 `Asset` trait

```rust
pub trait Asset: Send + Sync + 'static {
    /// 稳定类型标识（来自 prism_reflect），用于 UntypedAssetId / 容器清单。
    fn asset_type() -> AssetTypeId;
    /// 可选：声明本资产值内嵌的依赖句柄，供依赖图自动登记。
    fn visit_dependencies(&self, visit: &mut dyn FnMut(UntypedAssetId)) {}
}
```

`visit_dependencies` 让材质「引用的纹理」、场景「引用的网格」自动进依赖图，无需 loader 手工维护——对标 Bevy 的 `VisitAssetDependencies`，但走 `prism_reflect` 可自动派生（`#[derive(Asset)]` 扫描字段里的 `Handle<_>`/`SoftHandle<_>`）。

---

### 7.4 内存架构与分配策略

AAA 资产系统的瓶颈常不在 CPU 而在**内存布局与分配抖动**。借鉴 UE 的 inline/bulk 数据分离与 Decima 的分类预算，`prism_asset` 把资产内存分三类，各有独立分配器与预算：

| 类别 | 内容 | 分配器 | 生命周期 | 预算维度 |
|---|---|---|---|---|
| **元数据（inline）** | 身份、句柄头、`State`、依赖边、ToC | 代际竞技场 + 稳定索引池 | 常驻（资产活着就在） | 条目数，极小 |
| **资产值（asset body）** | 解码后的 CPU 侧结构（网格索引、音频 PCM 头） | 分类池分配器（size-class） | 随强句柄 | `cpu_asset_bytes` |
| **批量数据（bulk / GPU）** | mip/LOD/page 原始或压缩块、顶点/索引 blob | 环形暂存 + GPU 堆（归渲染侧建对象） | 随流送层级 | `vram_bytes` / `staging_bytes` |

- **元数据与 body 分离**：槽位只存小而定长的头（§7.1 `State`），大 body 存池中由 `State::Loaded` 持句柄指向——竞技场扩容时搬动的只是定长头，不搬 body，`&A` 稳定性由池保证（池内对象地址不随竞技场 realloc 变化）。
- **分类池（size-class pool）**：按常见资产 body 大小分级（如 ≤256B / ≤4K / ≤64K / 大对象直分配），同级复用空闲块，消解频繁 load/drop 的分配抖动与碎片——对标游戏引擎惯用的 TLSF/size-class 思路，但以 `prism_utils` 的池原语实现，`no_std+alloc` 下可用。
- **GPU 暂存环（staging ring）**：`gpu_stream` 下，解压目标是一段固定大小的环形暂存显存；批量请求轮转复用，避免每次换入都 `create_buffer`。满则背压（§9.4），不无限扩张。
- **预算中心（MemoryBudget）**：`cpu_asset_bytes` / `vram_bytes` / `staging_bytes` / `meta_bytes` 四类硬上限，由档位（§3）设定（移动端紧、主机宽）。流送驱逐（§11）与保留缓存（§6.4）都读同一预算中心打分，保证「总量可控、超限必驱逐」。
- **零拷贝路径**：容器内未压缩或 GPU 直解的 blob，`read_range` 直灌暂存/显存，CPU 侧不保留副本；仅需 CPU 访问的 body（碰撞网格、音频解码缓冲）才进分类池。

**反碎片**：长时运行（如 MMO、开放世界）靠分类池 + 固定暂存环把碎片锁死在池内；大对象单独 mmap/分配、释放即还 OS。诊断暴露每类池的占用/空洞率（§15），超阈值告警。

---

## 8. 依赖图与递归加载状态

### 8.1 图结构（重铸自 M0）

保留「`asset → dependency` 有向边 + Kahn 确定性拓扑 + 环检测」，但重构为**增量 + 失效感知**：

- 存储仍用有序容器（`BTreeMap/BTreeSet`）保证**确定性**（双跑拓扑序一致，是回归测试与网络一致的基石）。
- 新增**增量就绪计数**：不每次全图跑 Kahn，而是维护每节点 `unmet_deps` 计数；某依赖 `Loaded` 事件到达时只递减其 dependents 的计数，归零即「依赖闭包就绪」→ 发 `LoadedWithDependencies`。全量拓扑序仅在批量加载/环诊断时跑。
- 新增**失效传播**（热重载用）：源资产变更 → 其所有传递 dependents 标记 `stale`，驱动重装配（§12）。

### 8.2 递归加载状态

保留 `LoadState`（单资产自身）与 `RecursiveDependencyLoadState`（闭包聚合，worst-wins 折叠：`Failed > Loading > NotLoaded > Loaded`）。重构点：

- 聚合**增量维护**而非每查询重算闭包：每节点缓存 `rec_state`，依赖状态变化沿 dependents 方向增量上推，O(受影响子图) 而非 O(全闭包)。
- 新增 `AssetErrorId`：`Failed` 不再内嵌 `String`（每失败一次分配），而是指向集中 `ErrorRegistry` 的 id，`LoadState` 保持 `Copy`、失败信息集中可查、便于诊断面板聚合「同一文件缺失导致 N 个资产失败」。

### 8.3 环检测语义

拓扑失败时返回 `DependencyError::Cycle { participants }`（环及其下游，排序确定）。重构新增：环检测在**装配期**即触发（loader 声明依赖时增量检测回边），不必等全量拓扑——尽早报「材质 A 依赖 B 依赖 A」而非加载到死锁才发现。

---

## 9. 加载管线：AssetServer / AssetLoader / 事件驱动装配

> **实现契约（必须与目标规格分开阅读）**：当前 `AssetLoader::load(&[u8], &mut LoadContext)` 是同步、纯 CPU 解码函数；`AssetServer` 的公开同步路径仍在一次 mutex 临界区内完成读取、解码和依赖登记。`LoadScheduler` 已增加有限的锁外 `read -> decode -> dependency wait -> commit` 装配切片，并把阻塞 source read 与 CPU decode 放入独立池；它还提供一个受限异步 reload 切片：已 resident 的根资产在独立 reload flight 中读取/解码，候选阶段会在锁外递归读取并解码 root、普通依赖、label 对应源和 label 声明的依赖，统一以 revision-local candidate 原子提交；按路径 epoch 只允许最新结果提交，source overlay 变化、解码失败、类型不匹配、cycle 和重复 label 均不会污染旧 revision。该切片仍不是完整 EDL，也不包含 range-backed assembly、可抢占取消或 GPU 上传闭环；候选闭包受固定数量上限约束，优先级不能抢占已经开始的同步 loader，取消也只取消等待者而不强杀正在执行的 loader。下文 §9.3–§9.4 的依赖增量就绪、可抢占阶段和真正 EDL 仍是待实现目标，不能作为当前性能或并发保证。

### 9.1 AssetLoader 协议

```rust
pub trait AssetLoader: Send + Sync + 'static {
    type Asset: Asset;
    type Settings: Default + Send + Sync;      // 来自资产包内嵌设置 / 代码
    type Error: Into<AssetErrorId>;

    /// 目标规格：异步装配；当前实现请见 `prism_asset::loader_exec::AssetLoader`，其为同步字节解码。
    async fn load(
        &self,
        reader: &mut dyn AssetReader,          // 流式读，支持部分/分块
        settings: &Self::Settings,
        ctx: &mut LoadContext<'_>,             // 发子资产、声明依赖、再 load
    ) -> Result<Self::Asset, Self::Error>;

    /// 认领的扩展名（注册期登记，path→loader 分发）。
    fn extensions(&self) -> &[&str];
}
```

- **当前实现映射**：`AssetLoader` 不携带 `Settings` associated type，也不接受 `AssetReader`；它接收已读取的 `&[u8]`，以 `LoadContext` 声明依赖和 labeled 子资产。上方 `async fn` 是未来 EDL ABI 草案，不可直接作为当前实现示例。
- `LoadContext` 承载**子资产标签**（`ctx.labeled("Mesh0", mesh)` → `path#Mesh0` 自动获 id/句柄）、**依赖声明**（`ctx.load::<Image>("tex.png")` 返回句柄并自动连依赖边）、**依赖直嵌**（`ctx.add_dependency(guid)`）。对标 Bevy `LoadContext`，但 `load` 是真 `async fn`（跑在 `prism_tasks`），不借 Bevy 的 future 包装。
- `Settings` 来自资产包内嵌导入设置（§13.1）或代码传入，经 `prism_reflect` 反序列化——同一 loader 对不同资产可有不同导入参数（纹理 sRGB/线性、压缩格式）。

**自定义后缀是一等公民。** `extensions()` 可认领**任意**后缀——没有内置白名单，`.lvl`/`.mymesh`/`.vox`/`.mygame` 注册即生效。后缀只是「选哪个 loader」的默认线索，不是资产身份的一部分（身份是 `StableGuid`，§5.2），所以能灵活映射：

- **一 loader 多后缀**：同一 `AssetLoader` 可认领多个后缀（`["png","jpg","jpeg","tga"]` 都进图像 loader）；**逻辑类型与后缀解耦**——同一 `Asset` 类型可由多种后缀喂入。
- **复合后缀最长匹配**：按最长后缀优先匹配，`.tar.zst`、`.gltf`、`.ktx2` 这类多段/长后缀不会被 `.zst` 之类短后缀抢走。
- **大小写归一**：后缀匹配统一小写规范化，`.PNG` 与 `.png` 等价。
- **同后缀多候选的消解**：若多个 loader 认领同后缀，按「请求类型优先」消歧——`load::<Mesh>(path)` 带显式类型，只选产出 `Mesh` 的 loader；`load_untyped` 无类型时按注册优先级（后注册可显式覆盖）取首个认领者，冲突在注册期即可诊断告警。
- **显式覆盖后缀推断**：`load::<A>()` 的类型参数优先于后缀；`load_with_settings` 可在设置里**显式指名 loader**，用于「后缀撒谎」或无后缀的情形。
- **后缀别名与内容嗅探兜底**：支持别名重映射（`jpeg→jpg`）；对无后缀/后缀不可信的字节流，可选 magic-bytes 内容嗅探兜底选 loader，而非只认文件名。
- **资产包后缀可配**：单文件资产包默认 `.prism`（§13.1），项目可改为自定义后缀（如 `.myasset`），注册期声明即可。

### 9.2 AssetServer

```rust
impl AssetServer {
    fn register_loader<L: AssetLoader>(&self, loader: L);
    fn load<A: Asset>(&self, path: impl Into<AssetPath>) -> Handle<A>;        // 立即返句柄
    fn load_with_settings<A, S>(&self, path, settings: S) -> Handle<A>;
    fn load_soft<A: Asset>(&self, soft: &SoftHandle<A>) -> Handle<A>;         // 软→强，触发加载
    fn load_untyped(&self, path) -> UntypedHandle;                            // 扩展名推类型
    fn load_state(&self, id: UntypedAssetId) -> LoadState;
    fn recursive_load_state(&self, id: UntypedAssetId) -> RecursiveDependencyLoadState;
    fn reload(&self, path: &AssetPath);                                       // 手动触发重装配
}
```

- `load` **幂等去重**：同 `AssetPath` 并发多次 `load` 只排一次作业，共享 `AssetIndex` 与句柄分配（经 interning 表 + in-flight 表）。
- 句柄立即返回，作业入 `LoadScheduler`。

### 9.3 事件驱动装配（EDL 形态）

对标 Unreal Event-Driven Loader：加载不是「读完整文件→解码→完成」的串行栅栏，而是一张**作业依赖图**跑在 `prism_tasks` 上：

```
load(scene) ─▶ read bytes ─▶ decode scene ─┬─▶ load(mesh)  ─▶ read ─▶ decode ─┐
                                            ├─▶ load(mat)   ─▶ read ─▶ decode ─┤
                                            └─▶ load(tex)   ─▶ read ─▶ decode ─┘
                                                                                ▼
                                              scene「依赖闭包就绪」事件 ◀── 全部 Loaded 聚合
```

- 每个 `read` / `decode` 是一个 `prism_tasks` job，I/O 密集走 I/O 线程池、CPU 密集（解码/解压）走 work-stealing 池，互不阻塞。
- 依赖边由 §8.1 增量就绪计数驱动：子资产 `Loaded` 即递减父计数，归零发 `LoadedWithDependencies`。无全局加载锁、无「等整条链」。
- **优先级**：`load` 可带 `LoadPriority`（`Immediate / High / Streaming / Prefetch`），映射到 `prism_tasks` 优先级；相机前方资产插队，预取资产让路。
- **背压**：并发解码数、I/O 带宽受 `StreamingBudget` 限制，超限的作业排队而非一拥而上打爆内存/带宽（借 `prism_tasks` 背压）。

---

### 9.4 并发调度内核（LoadScheduler 内部）

EDL 作业图（§9.3）的正确性与吞吐全系于调度内核。这是最易出竞态的地方（§22 风险 2），因此把状态机、去重、取消、背压显式钉死。

**作业状态机**（每个 `(AssetIndex, 目标层级)` 一个 `LoadJob`）：

```
Queued ─▶ Reading ─▶ Decoding ─▶ ResolvingDeps ─▶ Finalizing ─▶ Loaded
   │          │          │             │              │
   └──────────┴──────────┴─────────────┴──────────────┴─▶ Failed(AssetErrorId)
   └─▶ Cancelled（无强句柄且无 pending 依赖者时提前短路）
```

- **幂等去重（in-flight 表）**：`AssetIndex → Arc<LoadJob>` 的无锁表（`prism_utils` 分片 `DashMap` 形态）。并发 `load` 同一资产：首个插入者建作业，后续者 `clone` 同一 `Arc<LoadJob>` 并挂上**完成通知**（`SharedFuture`），只排一次 I/O/解码。句柄分配经同一 interning，返回同一 `AssetIndex`。
- **I/O 池与 CPU 池分离**：`read`/`read_range` 投到有界 **I/O 线程池**（阻塞 syscall / DirectStorage 队列），`decode`/解压投到 `prism_tasks` **work-stealing 池**。两池互不阻塞——I/O 等待不占 CPU 核，CPU 解码不被慢盘拖死。
- **优先级与抢占**：`LoadPriority`（Immediate/High/Streaming/Prefetch）映射到 `prism_tasks` 多级队列；运行中作业的层级可**提升**（相机转向使 Prefetch 变 Immediate）——提升只改队列权重，不重启已完成阶段。
- **优先级反转防护**：若高优作业依赖低优作业的产物，依赖边触发**优先级继承**（被依赖者临时升到依赖者的优先级），避免「首帧必需卡在后台预取后面」。
- **取消与短路**：作业在每个阶段边界检查「是否仍有强句柄或 pending 依赖者」；都没有则转 `Cancelled`，不进入下一昂贵阶段（已读的字节丢弃或入短期缓存）。热重载 reload 对同资产的新作业会**取代**在途旧作业（代标记 epoch，旧作业完成时发现 epoch 失配即丢弃结果）。
- **背压令牌**：并发解码数、在途 I/O 字节、暂存环占用各有令牌桶；无令牌的作业停在 `Queued` 而非占资源，防「几千个 load 同时打爆内存/带宽」。令牌数由档位/预算设定。
- **失败与半装配回滚**：`ResolvingDeps` 阶段某依赖 `Failed` → 本作业按策略 `Failed`（严格）或以兜底继续（§23.3，宽松）；已分配的子资产槽位在回滚路径统一释放，不泄漏半装配状态。

**正确性验证**：该内核以 `loom`（§25）穷举关键交错——并发 load 去重、乱序完成、取消与完成竞争、epoch 失配丢弃——作为合入门槛。

---

## 10. 虚拟文件系统与数据源（VFS / Reader / 容器）

### 10.1 数据源抽象

```rust
pub trait AssetSource: Send + Sync {
    /// 打开一个流式读取器（可 seek、可分块读 bulk 区段）。
    async fn read(&self, path: &str) -> Result<Box<dyn AssetReader>, IoError>;
    /// 读资产包内嵌的导入设置段（§13.1，无旁车）。
    async fn read_meta(&self, path: &str) -> Result<Option<Vec<u8>>, IoError>;
    /// 枚举（热重载/烘焙扫描用）。
    async fn list(&self, dir: &str) -> Result<Vec<String>, IoError>;
    /// 可选：订阅变更（热重载）。
    fn watch(&self, on_change: ChangeSink) -> Option<WatchHandle>;
}

pub trait AssetReader: Send {
    async fn read_to_end(&mut self, buf: &mut Vec<u8>) -> Result<usize, IoError>;
    async fn read_range(&mut self, offset: u64, len: usize, buf: &mut [u8]) -> Result<(), IoError>;
    fn len(&self) -> Option<u64>;
}
```

- 多数据源经 `source://` 前缀路由：`"fs://assets/…"`（默认）、`"mem://…"`（测试/内嵌）、`"pak://game.pak#…"`（打包容器）、`"remote://cdn/…"`（CDN，Addressables 形态）。
- `read_range` 是**流送与 bulk data 的基础**：纹理只读需要的 mip 区段、网格只读需要的 LOD，不整文件灌内存。

### 10.2 内置后端

- **`FsSource`**（`std`）：OS 文件系统，开发期默认；`watch` 接平台 FS 事件（§12）。
- **`MemSource`**：内存字典，单测与 `embedded://` 内嵌资产（编译进二进制的 include_bytes）。
- **`ContainerSource`**（`pack`）：容器化分块 I/O，见 §10.3。
- **`RemoteSource`**（`async_io`）：HTTP(S) 拉取 + 本地缓存，远程内容分发。

### 10.3 容器化分块 I/O（pak / IoStore 形态）

发行包不该是「几万个小文件」（open/stat/read syscall 爆炸 + 碎片）。对标 Unreal IoStore / `.pak`：

- **容器 = 清单 + 大 blob**：`ContainerToc`（`ChunkId → (offset, len, compression, hash)`）+ 连续数据区。`ChunkId` 由 `StableGuid`（+ 子资产 + mip 层级）派生。
- **一次 open、顺序大读**：加载 N 个相关资产 = 读容器的一段连续区，而非 N 次 open。清单常驻内存，数据按需 `read_range`。
- **挂载优先级**：多容器可叠挂（基础包 + patch 包 + DLC），同 `ChunkId` 高优先级容器覆盖低——对标 `.pak` 的 mount 顺序，支撑热更新/补丁。
- **压缩分块**：每 chunk 独立压缩（GDeflate/zstd/平台专有），独立可解——支撑随机访问与 GPU 直通解压（§11）。

---

### 10.4 挂载点、多根、嵌套与自定义目录

资产的「物理从哪来」与「逻辑叫什么」必须解耦。`prism_asset` 用**挂载表（MountTable）**统一表达自定义目录、多根搜索、嵌套路径与运行期增删——文件目录、内存、容器、远程源一视同仁，对标 UE 的 pak mount / Unity 的多 provider。

```rust
/// 一条挂载：把某个数据源挂到一个逻辑前缀上，带搜索优先级与可写标记。
pub struct Mount {
    scheme: &'static str,       // 逻辑前缀：如 "assets" → "assets://levels/a.scene"
    source: Arc<dyn AssetSource>, // 任意后端：Fs / Mem / Container / Remote / 自定义
    priority: i32,              // 高优先级先命中（叠挂覆盖）
    writable: bool,             // 可写（编辑器保存 / 导入产物落盘）
}

pub struct MountTable { mounts: Vec<Mount> /* 按 (scheme, priority desc) 索引 */ }
impl MountTable {
    pub fn mount(&mut self, scheme: &'static str, source: Arc<dyn AssetSource>, priority: i32);
    pub fn unmount(&mut self, scheme: &'static str, source_id: SourceId); // 运行期卸载
    /// 解析：按前缀选候选挂载，按优先级降序逐个探测存在性，首个命中者胜。
    pub fn resolve(&self, path: &AssetPath) -> Option<ResolvedMount>;
}
```

**① 自定义目录（任意根）**：`FsSource::new(root)` 可指向任意绝对/相对目录；`mount("assets", FsSource::new("D:/MyGame/content"))` 即把该目录挂为逻辑 `assets://`。逻辑前缀稳定，物理根可换——换盘/换机不改任何引用。自定义后端只需实现 `AssetSource` 即可挂载（如数据库源、加密源、网络源）。

**② 多目录 / 多根 / 叠挂覆盖**：同一个逻辑前缀可挂**多个**源，按 `priority` 组成**搜索链**：

```
mount("assets", base_fs,  priority=0)    // 基础内容目录
mount("assets", dlc_pak,  priority=10)   // DLC 容器
mount("assets", mod_fs,   priority=20)   // 玩家 MOD 目录（最高优先）
mount("assets", patch_pak,priority=30)   // 热补丁
```

`resolve("assets://levels/arena.scene")` 从高到低探测，命中即返回——**MOD/补丁覆盖基础资产而不改原文件**（override-in-place）。这把容器层的「mount 优先级覆盖」（§10.3）推广到所有源：文件目录、容器、远程混挂。可写挂载（`writable=true`）用于编辑器保存与导入产物落盘，默认指向最高优先级的可写根。

**③ 嵌套目录**：路径分层是一等公民——`AssetPath` 的 `path` 段即 `a/b/c/x.ext`，深度不限；`AssetSource::list(dir)` 递归枚举子树（热重载扫描、烘焙闭包收集、编辑器资产浏览器都用它）。子资产标签 `#label` 正交于目录层级，`"a/b/hero.gltf#Mesh0/LOD1"` 这类「嵌套路径 + 嵌套子资产」同时成立。

**④ 运行期热挂载 + 多根热重载**：`mount`/`unmount` 可在运行期调用（挂载 DLC、玩家进 MOD 目录、CDN 上线），触发**内容目录增量重建**与依赖失效重解析——新挂载的高优先级源会让已加载的被覆盖资产标 `stale` 并按 §12 重装配。每个 `watchable` 挂载各自订阅变更：开发期 MOD 目录与基础目录可**同时**热重载，`watch` 事件带回挂载来源，失效传播据解析后的实际命中源判定——避免「改了被覆盖的底层文件却触发重载」的误报。

> 档位说明：挂载表是 `core` 能力（纯值表，`no_std` 可用）；`FsSource.watch` 属 `hot_reload`+`std`，`ContainerSource` 属 `pack`，`RemoteSource` 属 `async_io`。移动/发行档位常只挂「只读容器 + 远程」而不带 FS watch，开发档位挂「多个 FS 根 + watch」。

---

### 10.5 容器二进制格式与 I/O 调度

容器不是「zip 换皮」。要吃满 NVMe 顺序带宽并对接 DirectStorage，格式与 I/O 调度须按硬件对齐设计（借鉴 UE IoStore 的分离 ToC/数据与固定块、DirectStorage 的扇区对齐）。

**布局（分离 ToC + 数据区 + 可选独立索引文件）：**

```
┌─ Header ─────────────────────────────────────────────┐
│ magic / format_version / platform / flags             │
│ toc_offset,toc_len / chunk_count / compression_default │
├─ ToC（可独立成 .utoc 文件，常驻内存）────────────────┤
│ ChunkEntry[chunk_count]（按 ChunkId 排序，二分查找）  │
│   ChunkId(16B=StableGuid 派生) → {                     │
│     data_offset(u64, 4KB 对齐),                        │
│     comp_size(u32), raw_size(u32),                     │
│     block_count(u16), compression(u8), flags(u8),      │
│     content_hash(u64 截断，校验/去重)                  │
│   }                                                    │
│ CompressionBlock[]（每 chunk 的块表：块偏移+块压缩长） │
├─ 数据区（.ucas，大顺序 blob）────────────────────────┤
│ [chunk0 blocks][chunk1 blocks]... 每块 ≤64KB 独立可解  │
└───────────────────────────────────────────────────────┘
```

- **4KB 扇区对齐**：每 chunk `data_offset` 对齐到 4KB（DirectStorage/NVMe 友好，免跨扇区读放大）；块级 ≤64KB 粒度支撑随机访问与 GPU 批量解压（§11.2）。
- **ToC 常驻、数据按需**：ToC 可单独成文件先行加载（或 mmap），运行时只认 ChunkId 二分定位，再 `read_range` 拉数据区区段——「海量小文件」压成「一个大文件的若干段读」。
- **压缩块独立可解**：每块独立压缩（GDeflate/zstd/平台专有）+ 独立 `content_hash`，支撑随机 mip/page 访问与并行/GPU 解压；块表让 `read_range(mip_offset,len)` 精确定位到块集合。
- **挂载叠加**：多容器按优先级构成合并 ToC（§10.3/§10.4），同 ChunkId 高优先级覆盖——补丁/DLC 只追加容器 + 新 ToC。

> **当前实现边界**：上图仍是目标协议的扩展形态，包含当前 v2 尚未覆盖的 block table、GPU 直解和远程交付字段。仓库当前同时支持 v1 与 v2：v1 是逐 payload 的 Raw/raw-DEFLATE；v2 是独立 Chunk Table，支持每 Chunk Raw/raw-DEFLATE、Chunk hash、entry 的 `chunk_id`/`raw_offset`、严格目录/表/范围/哈希/对齐/零填充校验，以及 Raw Chunk 的物理范围合并读取。v2 Deflate Chunk 当前按完整 Chunk 解码后再切片，因此不是 block-level 压缩索引，也不提供 GPU 直通。未知版本/codec 返回结构化错误；加密、认证签名、Merkle/防回滚、完整依赖/流送表、远程 CDN/DirectStorage 发行协议仍未实现。

**I/O 调度器（吃满带宽、压读放大）：**

- **请求合并（coalescing）**：同容器相邻/重叠的 `read_range` 合并成一次大读，摊薄 syscall/队列开销；间隙小于阈值时「宁可多读一点」也不拆成两次（减少 IOP）。
- **按偏移重排（elevator）**：批内请求按 `data_offset` 排序顺序发射，逼近顺序读带宽，避免随机寻道（SSD 上降低队列深度压力）。
- **深队列异步**：DirectStorage 形态下维持高队列深度（成百请求在飞），完成走 fence/回调驱动 §9.4 作业图，不逐个同步等待。
- **读放大预算**：诊断暴露「实际读字节 / 有效使用字节」比，打包分组（§13.3）以最小化该比为目标（相关资产同容器、同簇）。

---

## 11. 按需流送与 GPU 直通解压

### 11.1 流送管理器

大世界/高清资产装不下显存，必须按需换入换出。`StreamingManager` 对标 DOOM Eternal / Decima 的预算驱动流送：

> **当前实现边界**：`prism_asset` 已实现预算控制器、同步 `StreamIoExecutor` 和使用现有 `TaskPool` 的锁外 `AsyncStreamIoExecutor`，两者都会真实调用 `AssetSources::read_range`；异步 executor 会在成功、取消、失败等请求终态清理 gate，并把重复执行/重复取消映射为明确错误，但只负责有界范围读取与状态提交，仍不包含解码器、GPU 资源上传或 DirectStorage 形态批量队列。完整预取、GPU 反馈闭环和按 Chunk 流送调度仍未实现。容器层已有文件 `RangeSource`、确定性 range batching、raw-DEFLATE per-payload 解码和按范围读取，但压缩 payload 仍需完整解码，不能替代完整流送系统。`prism_gdeflate_gpu` 已有真实 GPU de-interleave/stored-block 子集和 CPU Huffman fallback，不能宣称已完成完整 GPU 直通解压。

- **可流送资产声明分级**：纹理按 mip、网格按 LOD、虚拟纹理按 page。资产值内只常驻「低清兜底 + 元数据」，高清层级按需 `read_range` 流入。
- **预算驱动驻留**：`StreamingBudget`（显存上限、每帧 I/O 带宽、并发解码）。管理器维护「每个可流送资产的目标层级」，由**请求器**（相机距离、屏幕占比、可见性、显式 pin）打分，预算内尽量满足高分请求，超预算则驱逐低分层级。
- **滞回与无卡顿换入**：目标层级变化加滞回（避免临界距离抖动反复换入换出）；换入在后台解码完成后**原子替换** GPU 句柄，不阻塞渲染帧（double-buffer 层级）。
- **预取**：相机运动矢量/世界分区预测即将可见的资产，以 `Prefetch` 优先级预载——命中则无感，未命中让路于 `Immediate`。

```rust
pub struct StreamingRequest { id: UntypedAssetId, desired: StreamLevel, score: f32, source: RequesterId }
impl StreamingManager {
    fn request(&mut self, req: StreamingRequest);
    fn tick(&mut self, budget: &StreamingBudget, io: &mut LoadScheduler); // 每帧：打分→排序→预算内换入/驱逐
    fn resident_level(&self, id: UntypedAssetId) -> StreamLevel;
}
```

### 11.2 GPU 直通解压（DirectStorage / RTX IO / GDeflate 形态）

`feature = "gpu_stream"`，对接本仓 `prism_gdeflate_gpu`。传统路径「NVMe→CPU 读→CPU 解压→上传显存」有两次拷贝 + CPU 解压瓶颈；直通路径：

```
NVMe ──(批量队列, DirectStorage 形态)──▶ 显存暂存 ──(GPU 计算着色器, GDeflate)──▶ 解压至目标纹理/缓冲
```

- **批量队列**：成百上千小请求（mip/page）打包提交，摊薄每请求开销；完成走 fence/回调驱动 §9.3 作业图。
- **GPU 解压（目标规格，尚未完整实现）**：目标是压缩 blob 直灌显存暂存区并由计算着色器解压。当前 `prism_gdeflate_gpu` 的通用路径只 GPU 去交错、CPU 执行 Huffman inflate；另有 stored-block 专用 GPU fast path，仍由 CPU 解析 raw-DEFLATE block 边界，且 fixed/dynamic Huffman 块拒绝进入该路径。
- **优雅降级**：平台无 DirectStorage/无 GPU 解压则回退 CPU 路径（`async_io` + CPU zstd/gdeflate），上层无感——档位铁律（§3）。

### 11.3 与渲染驻留的边界

`prism_asset` 负责「字节到达 + 层级决策 + 解压」，**GPU 资源对象的创建/销毁**归渲染侧（`prism_render_*`）：流送管理器发「层级 N 的字节已就绪」事件，渲染侧据此建/换 `wgpu` 纹理视图。两者经句柄 + 事件解耦，`prism_asset` 不依赖任何渲染 crate（保持 L5→L4 单向）。

**2026-10-15 渲染侧窄切片已交付**：`prism_render_app::render_asset` 新增
`FixedPagePool`，以 `FixedPagePoolDescriptor` 创建单一固定页 RHI buffer；资产或
streaming producer 只需提交 typed `FixedPageUpload`，不需要依赖 `AssetId` 或
`AssetServer`。`PageUploadTransaction`、`PageReplaceTransaction` 和
这些事务覆盖分配、真实 `RenderQueue::write_buffer` 上传、原子替换和释放；新增
`commit_deferred(GpuCompletionValue)` 与 `reclaim_completed`，旧 allocation
在 `Retiring` 状态中保持不可复用，直到调用方提供的 GPU completion value 到达。
`FixedPageResidencyBridge` 另提供 `upload_page_deferred`/`reclaim_completed`，不会自行
猜测 fence 或 frame。backing buffer 通过现有 `PrepareContext`/`GpuPrepareBudget` 创建，
失败会回到上下文的资源回滚路径，所属 `RenderAsset::unload` 应调用
`FixedPagePool::destroy`。兼容性的立即替换 API 仍存在，但调用方必须证明没有在途 GPU
消费者；生产路径应使用 deferred API。该切片仍不是 asset 层 GPU streaming、
DirectStorage、GDeflate 完整 GPU 解压、GPU feedback/page-table、按反馈驱动的 page pool 淘汰或全资产 ECS/渲染闭环。固定页 streaming 的当前生产接线见 `page_streaming.rs` 与 `page_streaming_stage.rs`：它只接受显式 page request，不伪造 GPU feedback，也不把单页垂直切片描述成完整 M4。
Null RHI 回归和 wgpu readback 测试均已接入；无 adapter 时测试显式跳过，设置
`PRISM_REQUIRE_GPU=1` 则明确失败，不把 Null RHI 当作硬件可见性证明。

---

### 11.4 流送评分、GPU 反馈与驻留状态机

无卡顿流送（§22 风险 3）的核心是**打分、反馈、滞回**三件事定量化，而非拍脑袋。

**① 请求评分公式**（每可流送资产每帧算目标层级分，高分优先占预算）：

```
score = w_vis · screen_coverage        // 屏幕占比（像素覆盖估计）
      + w_dist · 1/(1+distance²)        // 相机距离衰减
      + w_dir  · facing_bonus           // 在视锥/运动前方加成
      + w_pin  · pinned                 // 显式钉住（UI/首帧）置顶
      - w_cost · bytes_to_promote        // 换入代价（大层级惩罚，防抖动换巨块）
```

权重 `w_*` 为可调参数（档位/项目级），由基准（§24）校准。得分映射到**目标 mip/LOD/page 层级**；预算内从高分到低分满足，超预算驱逐最低分已驻留层级。

**② GPU 反馈驱动（虚拟纹理/Nanite 形态）**：虚拟纹理/虚拟几何的真实需求由 GPU 在渲染时写**反馈缓冲**（本帧实际采样到哪些 page/cluster）。`prism_asset` 读回反馈（延迟 1–2 帧）转成 `StreamingRequest`，只拉真正被采样的页——比纯 CPU 距离估计精确得多，是「只流送看得见的」关键。反馈读回与请求生成归 §11.1，页表/物理页管理归渲染侧（§11.3 边界）。

**③ 驻留状态机**（每可流送资产）：

```
NotResident ─▶ Promoting(目标层级,后台解码中) ─▶ Resident(level=k)
     ▲                                                │
     │                                   ┌────────────┤
 Evicted ◀── Demoting(释放高层级) ◀───────┘   target<k（预算压力/离远）
```

- **原子换入（double-buffer）**：`Promoting` 在后台把新层级解码进**影子资源**，完成后一帧内**原子替换**句柄指向，渲染永远看到完整的旧层级或完整的新层级，无半更新撕裂。
- **滞回防抖**：目标层级用双阈值（promote 阈值 > demote 阈值）+ 最小驻留帧数，临界距离来回走不触发反复换入换出。
- **驱逐宽限**：`Demoting` 不立即还显存，留一个短窗口（相机常常马上转回来），窗口内复用则秒恢复——与 §6.4 保留缓存同源思想，层级粒度版。

**④ 无卡顿保证**：换入解码/上传全在后台池（§9.4），渲染帧只做「指针原子 swap」；每帧换入字节受 `staging_bytes`/带宽令牌限额，摊到多帧，杜绝单帧上传尖峰。以帧时间基准（§24）守护「流送开启 vs 关闭的帧时间差 < 阈值」。

---

## 12. 热重载与依赖失效传播

`feature = "hot_reload"`，编辑器/迭代核心。

> **当前实现边界**：已有 watcher 事件/去抖模型、`Arc<AssetServer>::watch` 托管后台消费入口和由调用方驱动的批次应用入口；锁外异步 reload 已有限支持 resident 根资产的 revision-local staging，并可递归纳入新普通依赖、label 对应源及 label 声明的依赖，所有候选在一次 commit 中发布。重复 label、类型变化、cycle、source overlay 变化、读取/解码失败、旧 epoch 和超出候选上限均拒绝发布并保留旧 revision；更大规模交错验证仍受边界约束。同步候选 staging 的 promotion/label 生命周期回归已覆盖。Unix `FsSource` 已使用目录 fd 相对打开与 `O_NOFOLLOW` 消除 metadata→open 的 TOCTOU 窗口；Windows reparse-point 防护及非 Unix 等价安全打开语义仍待实现。

- **变更侦测**：`AssetSource::watch` 接平台 FS 事件（inotify/FSEvents/ReadDirectoryChangesW，经 `prism_platform` 隔离），去抖后产出「路径 X 变更」。
- **失效传播**：路径 → `StableGuid` → `AssetIndex`，经 §8.1 依赖图**反向遍历** dependents，把该资产及其所有传递依赖者标 `stale`。例：`tex.png` 改 → 用它的 `material` stale → 用该材质的 `scene` 的受影响部分 stale。
- **重装配**：stale 资产重走 §9.3 装配，完成后**原地替换**存储槽的值（同 `AssetIndex`、句柄不变），发 `Modified` 事件。消费方收 `Modified` 自行刷新（渲染重建 GPU 资源、UI 重绘）。句柄身份不变是关键——热改纹理不该让所有持有者的句柄失效。
- **原子性与一致性**：重装配在后台完成后一次性 swap，不出现「半更新」中间态；失败则保留旧值 + 发诊断（§15），不让坏文件搞崩运行时。
- **依赖结构变更**：若重装配后依赖集变化（材质新增一张纹理），依赖图增量更新边并触发新依赖加载。

---

## 13. 资产包与制品消费契约

> **归属声明**：离线烘焙、缓存键、DDC/CAS、Revision、打包与 Patch 的唯一持有者是 `Prism Asset Build`（`docs/prism_asset_build_design_zh.md` §2.7/§4.6/§10/§14）。本节只定义 `prism_asset` 侧的**消费契约**：读取编辑态资产包（§13.1）、适配进程内重烘（§13.2）、运行时容器与挂载（§13.3）；缓存键与 DDC 分层已收敛（§13.4）。

`feature = "process"`（资产包与制品消费 + 进程内重烘适配）+ `pack`（运行时容器挂载）。「源资产 → 平台最优的运行时资产」的**生产**在 Asset Build，本 crate 负责**消费**与**挂载**，对标 Unreal cook 之后的运行时 IoStore 侧。

### 13.1 单文件资产包（UE 形态：一个资产一个文件，无旁车）

采用 **UE 的「单文件资产包」模型**：编辑器里每个资产就是**一个自包含文件**（`.prism` 资产包），导入设置、身份、源引用、可编辑载荷全部**内嵌在同一个文件里**——没有 `.meta` 旁车，没有散落的隐藏清单。目录里「一个文件 = 一个资产」，所见即所有。这是对标 Unreal 的 `.uasset`/`.umap`：把原始源文件（psd/fbx/wav）**导入**成引擎自己的资产包后，包就是资产的唯一真相。

**资产包内嵌布局（非规范摘录；权威定义见 `docs/prism_asset_build_design_zh.md` §5.2，两处若有冲突以该文为准）：**

```
┌─ Header ──────────────────────────────────────────────┐
│ magic / schema_version / StableGuid(128bit，身份内嵌)   │
├─ ImportSettings 段（prism_reflect 序列化）─────────────┤
│   处理器选择 + 参数：压缩格式(BCn/ASTC/ETC2)、mip、     │
│   网格优化/量化、音频码率…                              │
├─ SourceRecord 段（可重导入）──────────────────────────┤
│   源格式 + 源内容哈希 + 源定位（外部路径 或 内嵌源字节）│
├─ Dependencies 段 ─────────────────────────────────────┤
│   引用的其它资产 StableGuid 列表（§8 依赖图据此建边）   │
├─ Payload 段（可编辑表示）─────────────────────────────┤
│   导入后的编辑态数据（网格/图像/曲线…）                 │
└───────────────────────────────────────────────────────┘
```

**① 身份随文件走——移动/重命名零成本。** `StableGuid`（§5.2）直接写在包头里，**资产的身份就在文件内部**，不靠路径、不靠旁车。把 `.prism` 文件移到任何目录、改任何名字，所有对它的引用都不断——根除了 Unity「移动文件必须连 `.meta` 一起移、否则 GUID 丢」的脆弱点。GUID→路径索引仅作**派生加速缓存**（扫描包头即可随时重建），不是真相来源，因此它损坏/过期也不会丢身份。

**② 设置与数据不可能脱钩。** 导入设置、源引用、依赖、载荷同在一个文件里原子读写——不存在「meta 与资产分离、各自被改/被移造成 desync」的问题；也不会产生孤儿 `.meta`。一次保存即一致快照。

**③ 导入 = 外部源 → 资产包；可重导入。** 美术的工作源（psd/blend/高模）可留在内容树之外；`SourceRecord` 记源定位 + 内容哈希，源变了即提示/自动重导入；小源也可**内嵌进包**做到完全自包含、脱离原始源仍可打开。约定优于配置仍成立：导入时按项目/文件夹类型默认填 `ImportSettings`，之后随包走、按需在编辑器里逐资产改。

**④ 编辑态与运行态分离（和 UE 一致）。** `.prism` 资产包是**编辑/作者态**的单位，**不随游戏发行**；打包（§13.3）把资产包闭包 cook 成运行时容器 `.ucas`（§10.3/§10.5，ToC + 压缩分块，无逐文件开销）。即「编辑期一资产一文件、运行期合并进容器」——两头各取所需。

> 诚实权衡：单文件是**二进制自包含**，不可逐行 diff，合并冲突按**整资产粒度**解决（UE 同款局限）。缓解：① 资产级独占签出/锁（内容索引记锁位），团队按资产而非按行协作；② 序列化**确定性**（纪律见 `prism_asset_build_design_zh.md` §18.1：字段定序、无时间戳/无不定并行序）——同编辑产同字节，减少伪冲突；③ 可选把纯参数型小资产导出为文本旁视图供 review，真相仍以包为准。相比旁车/中心清单，单文件换来的是**目录干净 + 身份稳健 + 读写原子**，这正是你要的 UE 手感。

### 13.2 进程内重烘适配（`BakeService`）

离线烘焙的处理器、缓存键与 DDC 归 `Prism Asset Build`（其 `Processor` trait 见该文 §6）。本 crate 只保留**进程内调度适配**，供编辑器实时预览与运行时 dirty 重烘调用同一个 Bake Kernel（`DerivationClass::RuntimeDerived`）：

- **同一内核**：适配层不做任何格式解析或算法实现，只负责把场景/资产 IR 喂给 `prism_bake_*` 内核（Asset Build §6.6），并把产物装入本地 runtime cache。
- **不写 DDC/CAS**：`RuntimeDerived` 产物不进内容寻址存储、不进 Manifest；仅可持久化为本地可弃缓存。
- **失效驱动**：编辑器内重烘的失效由场景 Revision / 资产 `Modified` / `Unused` 等事件驱动（§14.1、§12），不复制 Asset Build 的 `ArtifactKey`。
- **确定性**：仍受 `prism_asset_build_design_zh.md` §18.1 纪律约束；同一输入下进程内重烘与离线烘焙必须产出一致（声明为 `Deterministic-not-byte` 者按其容差，见 `docs/prism_derived_data_contract_zh.md`）。
- **归属边界**：`BakeService` 不持有编译期插件、不写发布目录、不改 Registry；它只决定“由谁调度、产物放哪”。

### 13.3 运行时容器与挂载

**生产侧（发行闭包扫描、chunk 规划、容器与 patch 产出）归 `Prism Asset Build`**（`docs/prism_asset_build_design_zh.md` §14）；本 crate 的 `pack` 档只负责**运行时侧**读取同一容器格式：

- 发行闭包扫描、分组与 chunk 规划：Asset Build §14.1–§14.4。
- 容器格式（ToC + 压缩分块 blob）与 patch/DLC：Asset Build §14.3/§14.5。
- 本 crate 侧：按内容目录（`StableGuid → ChunkId → 容器`）定位、挂载优先级（§10.4）、按段异步读取（§10.3/§10.5）、解压与 GPU 直通（§11.2）。

> **当前实现边界**：当前运行时消费切片覆盖 raw/per-payload Deflate 容器、ToC/范围/哈希校验、文件 `RangeSource`、Manifest 路由、overlay 和显式 Patch tombstone（高优先级删除标记可隐藏底层 GUID，卸载后恢复）。`install_runtime_package` 现在提供本地原子安装事务：先在 overlay 克隆快照中挂载容器/tombstone，完成 Manifest route、优先级、容器 GUID 与 immutable identity 全量绑定校验后，才一次性替换 live overlay；任一失败都保留旧快照。构建侧已有可验证的 Patch vertical slice：`prism_asset_build_core::package_runtime_patch` 只发新增/变更或 mapping 变化的 payload，并为变更 route 携带无 payload 的依赖上下文；`RuntimePatchPackage::apply_to_manifest` 校验 descriptor、base/patch/target manifest hash 和路由/tombstone/依赖元数据一致性；`prism_asset_build_cli patch` 已能对本地已发布 package 生成确定性 Patch 目录。`prism_asset_build_core::release` 和 CLI 还提供 caller-owned Ed25519 release descriptor 签名/验签（`release.sig`），但没有密钥托管、跨机器信任根或完整发行服务。上述能力尚未接入运行时下载、OverlaySet 的自动安装发现、远程发行闭环或 package-level 回滚。BLAKE3 是未认证完整性摘要，不是防篡改签名。发行 publication 边界已额外严格证明 Build Manifest 的每个 `AssetId` 都有唯一的 `StableGuid` runtime route，防止发布内部自洽但缺少运行时资产的包。压缩块表、加密、密钥/nonce/tag、容器/Merkle 签名、anti-rollback、远程 CDN 按段读取及 package-level rollback 均未完成。

---

### 13.4 派生数据缓存（DDC）——已收敛至 Asset Build

本 crate **不再定义**缓存键与 DDC 分层。此前本节自建的三级 DDC（本地内存 → 本地磁盘 → 共享远程）与 §13.2 的缓存键定义，与 `Prism Asset Build` §10 的 `ArtifactKey → DDC → CAS` 及 L0–L3 分层缓存重复，构成两套互不相认的缓存协议，是本轮收敛的对象：

- **键与分层**：唯一真相为 `docs/prism_asset_build_design_zh.md` §10（CAS/DDC 职责分离、分层缓存、负缓存、GC、性能设计）。
- **确定性纪律**：唯一真相为 `docs/prism_asset_build_design_zh.md` §18.1；本 crate 若实现任何处理器逻辑，必须服从同一纪律，并以同一双跑对拍验收。
- **运行时只消费**：本 crate 通过 Asset Build 产出的 Runtime Manifest 解析制品（`prism_asset_build_design_zh.md` §15.1），不参与键计算。
- **保留的本地缓存**：`retain_cache` 档的驻留/内存缓存（§7、§11.4）与 DDC 不是一回事——前者是运行时驻留策略，不跨机构共享，随进程生命周期回收，不属于本节收敛范围。

> 收益不变、契约唯一：内容寻址 + 确定性 = 增量 + 可共享 + 可去重 + 可回归。这四者由 Asset Build 的键协议一次给出，而不是两处各写一半。

---

## 14. 事件、变更与世界集成

### 14.1 事件模型（重铸）

```rust
pub enum AssetEvent<A: ?Sized> {
    Added { id: AssetId<A> },                    // 数据首次就绪
    Modified { id: AssetId<A> },                 // get_mut / 热重载重装配
    Removed { id: AssetId<A> },                  // 回收点真正删除
    Failed { id: AssetId<A>, error: AssetErrorId }, // 新增：加载失败可观测
    LoadedWithDependencies { id: AssetId<A> },   // 依赖闭包就绪
    Unused { id: AssetId<A> },                   // 新增：进入可回收但尚未删（给缓存层挽留的机会）
}
```

事件仍 `Copy`（只含 `Copy` 的 id + error id）、按类型分队列、每帧 `drain_events`。新增 `Failed`（诊断/占位资源替换）与 `Unused`（缓存层可在真正回收前「再次 load 挽留」，对标 Addressables 的释放回调时机）。

### 14.2 与 ECS / 渲染 / 场景集成

- **ECS**：`Assets<A>` 作为资源（经 `prism_ecs`），`AssetEvent<A>` 经事件通道给系统消费。资产加载/热重载不进 ECS 调度热路径——系统只读句柄、按事件响应。
- **渲染**：渲染提取阶段读 `Assets<Mesh>`/`Assets<Image>`，`Modified`/流送层级事件触发 GPU 资源重建。`prism_asset` 不反向依赖渲染（§11.3）。
- **场景/预制体**（`prism_scene`）：场景存 `SoftHandle`（`StableGuid`），实例化时批量 `load_soft` → 依赖闭包就绪事件驱动「场景可见」。
- **脚本/编辑器**：经 `prism_reflect` 把句柄/软引用暴露为可检视/可改的属性；编辑器拖拽改引用 = 改 `SoftHandle` 的 `StableGuid`。

---

## 15. 可观测性与诊断

- **计数器**（`prism_diagnostic`）：在途加载数、各类型资产驻留数/字节、每帧 I/O 带宽、解码队列深度、流送命中/驱逐率、缓存命中率、失败数。
- **错误注册表**：`AssetErrorId → {path, source, reason, 受影响 dependents}`，诊断面板按根因聚合（「hero.gltf 缺失导致 12 个资产失败」）。
- **加载时间线 trace**：每资产的 read/decode/依赖等待耗时打点，chrome-trace 导出，定位加载长尾。
- **确定性校验钩子**：烘焙双跑产物哈希对拍、拓扑序双跑对拍（CI 守护 §16）。
- **断言式不变量**（debug）：无悬垂句柄、无「Loaded 但 strong_count==0 还没进回收队列」、依赖图无自环。

---

## 16. 确定性保证

- **拓扑序确定**：有序容器 + 最小就绪 id 选择，双跑一致（网络/回归基石）。
- **烘焙确定**：同输入逐字节同产物——纪律与实现归 `prism_asset_build_design_zh.md` §18.1；本 crate 不定义缓存键（§13）。
- **身份确定**：`StableGuid` 由规范化路径/内容哈希派生，跨构建/平台一致。
- **回收点确定**：延迟回收集中在 `collect_releases`，不受 drop 时序/线程调度影响最终可见状态。

这些共同保证「同样的输入 → 同样的加载顺序、同样的产物、同样的身份」，是存档可移植、网络可复制、回归可对拍的前提。

---

## 17. 性能工程

1. **身份 `Copy` 且紧凑**：`AssetIndex` 8 字节、`AssetId<A>` 零额外成本（类型标签是函数指针 `PhantomData`），塞进 ECS 组件/热路径不分配。
2. **路径查表零分配**：`AssetPath<'a>` 借用形态走 interning 查表，命中不分配 `String`；`StableGuid` 为查表键（16 字节哈希比较）而非字符串比较。
3. **加载并行**：read（I/O 池）与 decode（work-stealing 池）分离，依赖按 EDL 装配，无串行栅栏、无全局加载锁。
4. **容器化 I/O**：少数大顺序读替代海量小文件 open/read；清单常驻、数据 `read_range` 按需。
5. **GPU 直通**：绕过 CPU 解压与两次拷贝，NVMe→显存直灌（`gpu_stream`）。
6. **回收 O(待回收)**：`ReleaseQueue` 替代全槽位扫描；增量就绪计数替代全闭包重算；增量递归状态替代全图折叠。
7. **流送预算**：显存/带宽/并发硬上限 + 滞回，防「一拥而上打爆」与「临界抖动」。
8. **增量烘焙**：内容哈希缓存，改一个资产只烘一个。
9. **档位剪裁**：移动端只带 `core + 只读 pak + 最小流送`，不付异步/热重载/烘焙成本。

**基准即规格**：冷/热加载吞吐（资产数/秒、MB/秒）、首帧可玩前加载墙时长、流送换入延迟与命中率、依赖闭包就绪延迟、热重载往返时延、烘焙全量/增量耗时、回收/事件开销随资产规模的曲线。

---

## 18. 易用性与迁移策略

### 18.1 典型用法

```rust
// 加载（立即得句柄，数据稍后到）
let mesh: Handle<Mesh> = server.load("models/hero.gltf#Mesh0");
let scene: Handle<Scene> = server.load("levels/arena.scene");

// 取值（就绪前 None）
if let Some(mesh) = meshes.get(mesh.id()) { /* 用 */ }

// 等依赖闭包就绪（事件驱动，不轮询）
for ev in scene_events.drain() {
    if let AssetEvent::LoadedWithDependencies { id } = ev { spawn_scene(id); }
}

// 软引用（场景里「将来可能要」，不触发加载）
let soft: SoftHandle<Material> = SoftHandle::from_guid(guid);
let strong = server.load_soft(&soft);   // 真要用了才加载

// 自定义 loader
impl AssetLoader for GltfLoader {
    type Asset = Scene; type Settings = GltfSettings; type Error = GltfError;
    async fn load(&self, r, s, ctx) -> Result<Scene, GltfError> {
        let mesh = ctx.labeled("Mesh0", decode_mesh(r)?);   // 子资产
        let tex  = ctx.load::<Image>("hero_albedo.png");    // 依赖，自动连边
        Ok(Scene { mesh, material: Material { albedo: tex } })
    }
    fn extensions(&self) -> &[&str] { &["gltf", "glb"] }
}
```

### 18.2 从 M0 / bevy_asset 迁移（不保留旧 API）

| 旧（M0 / bevy_asset 形态） | 新 | 说明 |
|---|---|---|
| `assets.insert(v)` 自铸 id | `server.load(path)` 或 `assets.reserve`+`fulfill` | id 由 server interning 统一铸造，支持「先句柄后数据」 |
| `UntypedAssetId{index}` | `UntypedAssetId{index, type_id}` | 类型擦除带类型标签，防错 arena 命中 |
| `LoadState::Failed(String)` | `Failed(AssetErrorId)` | 失败信息集中、`Copy` 保持 |
| `remove_unused()` 全扫描 | `collect_releases()` 队列驱动 | O(待回收) + 延迟回收宽限 |
| `AssetPath{String,Option<String>}` | `AssetPath<'a>` 借用 + `source://` + guid interning | 热路径零分配、多数据源、稳定身份 |
| 无软引用 | `SoftHandle<A>` | 大场景「引用不加载」 |
| 无流送/容器/烘焙 | §10–§13 | 全新能力层 |

迁移基调：旧 API 直接废弃；提供一份 `migration_zh` 对照与 codemod 要点（load 调用点替换、insert→reserve/fulfill、Failed 构造改走 ErrorRegistry）。因 M0 仅内核且下游尚未大规模接线，重铸窗口成本最低——**现在重构代价最小**。

---

## 19. crate 分层与模块布局

```
pkg/prism_asset/                 # 身份/存储/依赖/句柄/事件核（no_std+alloc 默认）
  src/
    id.rs                        # AssetIndex / AssetId / UntypedAssetId / AssetTypeId
    guid.rs                      # StableGuid（路径/内容派生，规范化+冻结哈希）
    path.rs                      # AssetPath<'a> + source:// + #label
    handle.rs                    # Handle / WeakHandle / SoftHandle / UntypedHandle / ReleaseQueue
    storage.rs                   # Assets<A>（reserve/fulfill/fail/collect_releases）
    types.rs                     # AssetTypes 类型擦除注册 + Asset trait
    dependency.rs                # DependencyGraph（增量就绪 + 失效传播 + 环检测）
    load_state.rs                # LoadState / RecursiveDependencyLoadState / ErrorRegistry
    event.rs                     # AssetEvent（含 Failed/Unused）
  features:
    std / async_io               # AssetServer / LoadScheduler / AssetReader（跑在 prism_tasks）
    ecs                          # 把 `Assets<A>` 实现为 `prism_ecs::Resource`（§14.2：资产作为资源住进世界）；
                                 # 默认关闭，核心保持零依赖，只有拥有世界的 crate 打开它
    hot_reload                   # watch + 失效传播 + 重装配
    process                      # AssetProcessor + 内容哈希缓存 + 单文件资产包
    pack                         # ContainerIo（ToC + 压缩分块）+ 挂载优先级
    gpu_stream                   # DirectStorage 形态队列 + GDeflate（prism_gdeflate_gpu）
    retain_cache                 # 保留缓存：KeepAlive/LRU/Pinned 卸载策略（防 load/drop 抖动）

pkg/prism_asset_import/          # 具体格式 loader：glTF/FBX/USD、PNG/KTX2/basis、WAV/OGG、字体
# 资产**数据**类型（运行时的 `Assets<Mesh>` / `Assets<Image>`，§14.2）：与底座同纪律——
# 只借 RHI 的词表（`VertexFormat`/`TextureFormat` 等），不持 GPU 句柄、不依赖 ECS、不链接后端。
pkg/prism_mesh/                  # Mesh 资产：具名顶点属性 + 索引 + 拓扑 → 派生布局/顶点字节/包围体
pkg/prism_image/                 # Image 资产：extent + format + texel 数据 + 自带 sampler（尺寸/格式校验）
# 平台烘焙不在本 crate 树内：算法内核为 pkg/prism_bake_*，DAG/键/DDC/CAS/打包归 pkg/prism_asset_build_*
# （原 pkg/prism_asset_bake 命名废弃；见 prism_asset_build_design_zh.md §2.7/§6.6）
pkg/prism_vfs/                   # pak/容器打包工具 + 远程/CDN 源（可被 prism_asset 的 pack 复用）
```

依赖：`prism_utils`（哈希/位打包/MPSC 队列）、`prism_tasks`（作业图）、`prism_reflect`（settings/meta）、`prism_diagnostic`（计数/trace），可选 `prism_gdeflate_gpu`、`prism_platform`（FS watch）。**不碰任何 `bevy_*`。** 格式解码与平台烘焙拆到独立 crate，保持内核纯净可测、可 `no_std`。

---

## 20. 契约、不变量与版本化

- **AssetIndex 代际**：槽位回收 generation+1，旧 id 永不别名新占用；仅进程内有效，不入存档/网络。
- **StableGuid 稳定**：同资产跨构建/平台一致；哈希算法 + 路径规范化规则**一经发行即冻结**，变更须经重映射表（否则旧存档/旧容器目录全失效）。
- **句柄身份不变**：热重载重装配**复用同 AssetIndex**，持有者句柄不失效；只发 `Modified`。
- **回收单点**：真正删除只发生在 `collect_releases`；`Unused` 先行、`Removed` 后至，给缓存层挽留窗口。
- **拓扑/烘焙确定**：双跑一致；处理器版本号单调递增，逻辑变更必 bump。
- **类型安全**：`UntypedAssetId` 带 `AssetTypeId`，错类型 `typed::<B>()` 在存储层 miss 且可诊断，不静默命中。
- **版本化契约**：`StableGuid` 派生规则、容器 ToC 格式、内容目录格式、资产包 schema、`AssetEvent` 变体——均为跨版本兼容面，变更走 schema 版本 + 迁移（复用 `prism_reflect` §24 schema 迁移）。

---

## 21. 路线图（M0–M6）与验收方法

> 下列里程碑是交付顺序与验收目标，不是当前状态标签。当前状态只认本文 §实施状态和代码/测试证据。

- **M0 身份/存储/依赖核（已交付）**：`AssetIndex`/`AssetId`/`UntypedAssetId`、`AssetPath`、`Handle`/`WeakHandle`/`UntypedHandle`、代际 `Assets<A>` + `AssetEvent` + `remove_unused`、`DependencyGraph`（Kahn + 环检测）、`LoadState`。`no_std+alloc`、无 `unsafe`、单测绿。
- **M1 内核重铸（已交付）**：三层身份（+`StableGuid`/`AssetTypeId`）、`SoftHandle`、类型擦除注册、`Asset` trait + `visit_dependencies`、`reserve/fulfill/fail` + 延迟回收、依赖图增量就绪/失效、`ErrorRegistry`、确定性 loader 选择。旧 API 已按开发期策略重铸。
- **M2 加载管线（部分交付）**：`AssetServer`、`AssetLoader`/`LoadContext`、VFS、同步依赖装配、bounded scheduler facade 和锁外异步 range executor 已有；M2.1 已交付独立 I/O/CPU 池上的有限完整读/解码/依赖等待/提交切片；依赖增量驱动的 EDL 作业图、range-backed 装配和真正可抢占阶段仍待交付。
- **M3 热重载（部分交付）**：watch/去抖、失效反向传播、reload plan、失败保留 last-known-good 和 `Modified` 已有；锁外异步重装配现在覆盖新普通依赖、label 对应源和 sibling 新 label 的候选闭包，并以一次 revision-local commit 发布；完整 labeled revision 生命周期、可抢占阶段和跨平台安全打开语义仍待交付。
- **M4 流送（部分交付）**：预算/滞回/取消/失败/驱逐状态机、同步与锁外异步 range executor，以及一个固定页请求到 RenderApp GPU 上传/延迟回收的窄垂直切片已交付；`StreamingManager` 预取、GPU feedback/page-table、按反馈驱动的 page pool、块级解码、DirectStorage 形态队列和完整 GDeflate 解压仍待交付。
- **M5 资产包消费与容器挂载（部分交付）**：v1 raw/Deflate 与 v2 独立 Chunk Table 容器、Manifest/overlay、范围读取和 ChunkPlanner 驱动的物理布局已有；v2 已覆盖 Raw/Deflate、Chunk hash、entry raw offset、确定性对齐与严格损坏拒绝。完整 block-level Chunk wire contract、压缩块索引/加密/签名、GPU 直解、Cook/发行闭环仍待交付。
- **M6 集成（目标）**：ECS、渲染驻留、场景/编辑器和 Asset Build 制品的端到端接线。

**基准即规格**：每个已交付切片先以功能/一致性/故障门禁验收；性能数字只有在固定数据集、硬件、构建模式、预热/冷启动和统计方法写入 benchmark 后才升级为 CI 阻断门槛。

---

## 22. 诚实边界与风险

- **现状**：M0/M1 核与同步 AssetServer/VFS 已落地并有单测；构建核心有 CAS/ArtifactKey/Revision、支持 ready-wave 并行 DDC/CAS 的 BuildExecutor、本地 DDC 描述符索引和独立质量门禁，并有同进程双跑确定性门禁；容器 v1/v2 有 Runtime Manifest 路由读取、`AssetReader` 适配、overlay、确定性有界 range batching，且 ChunkPlanner 已接入 v1/v2 物理布局，v2 已由 reader/writer 实际生成与消费；有界异步调度 facade 已覆盖容量、去重、取消和通知竞态，AssetServer 已提供 watcher 批次应用入口和托管后台消费入口，streaming 另有真实锁外异步 range executor。M2.1 已交付有限范围的锁外完整 `read -> decode -> dependency wait -> commit`：不同路径的 source read 可并行，I/O 与 CPU 使用独立池，父资产依赖成功后才提交，依赖失败/环会终态失败且不发布父 revision；另有 resident 根资产的有限 async reload/epoch 与 revision-local 依赖闭包 staging，允许候选递归纳入新普通依赖、label 对应源和 label 声明的依赖，并在单次 commit 中原子发布，但仍不包含 range-backed assembly、真正可抢占取消或完整 EDL。M3 的自动 watcher 消费切片已落地；旧 revision labeled 生命周期、provisional rollback 和 foreign-label promotion rollback 已有回归，完整异步 labeled 生命周期的交错负载和规模化验证仍未完成；M4–M6（流送/GPU 完整解压、BakeService、ECS/渲染/编辑器集成和发行生产链）仍未完成。本文是实施规格而非对未交付能力的宣称。
- **高风险项**：
  1. **StableGuid 冻结（M1/M5）**：派生规则一旦发行即不可变，变则旧存档/容器目录失效。须早定早冻、哈希算法与规范化规则定死并测「跨构建一致」。
  2. **异步装配正确性（M2）**：EDL 作业图 + 依赖增量就绪的竞态（并发 load 去重、依赖边并发更新、失败半装配回滚）是最易出错处，须压测并发加载 + 乱序完成。
  3. **流送无卡顿（M4）**：换入换出的原子替换、滞回参数、预算打分若调不好，表现为纹理突变/卡顿/显存抖动——须真实场景调参并以帧时间基准守护。
  4. **GPU 直通可移植（M4）**：DirectStorage/GDeflate 平台差异大，CPU 回退路径必须始终可用且经测，不能让「高端路径」成为唯一路径。
  5. **烘焙确定性（M5）**：处理器非确定（浮点/并行序/时间戳混入）会毁掉增量缓存与 CI 共享；须纪律性地禁非确定输入并双跑对拍。
  6. **热重载依赖结构变更（M3）**：重装配后依赖集变化的增量图更新 + 环重检测，边界多，须覆盖「新增/删除依赖、变环」用例。
- **与既有文档关系**：本 crate 是渲染（网格/纹理/材质/着色器驻留）、音频、`prism_scene`、动画、UI 的内容基座；软引用/依赖闭包就绪/热重载 `Modified` 的契约须与这些消费方一致。身份/存储核已对齐 `prism_engine_component_gap_zh.md` §3.3 的 `prism_asset` 定位（句柄/异步加载/依赖图/热重载/引用计数/生命周期）；导入器按该文拆 `prism_asset_import`，**离线烘焙不再拆为 `prism_asset_bake`**——算法内核为 `prism_bake_*`，DAG/键/DDC/CAS/打包归 `Prism Asset Build`（`prism_asset_build_design_zh.md` §2.7/§4.6），本 crate 只消费其制品。
- 所有 Prism crate 不含任何 Unreal Engine / Unity 源码或衍生代码；仅借鉴公开架构形态与经典数值。本文无任何 AI/ML 内容。

---

## 23. AAA 高级功能增补

本章补齐顶级资产系统常被忽视、却在真实 AAA 项目里缺一不可的能力。均 feature/档位门控，默认不付成本。

### 23.1 子资产与标签图
一个源文件（glTF）产出网格/材质/动画/纹理多个子资产，各有独立 `StableGuid`（`parentGuid + label` 派生）与独立生命周期：引用 `scene#Mesh0` 不必保活整个 glTF 的全部子资产。标签解析在装配期一次建图。

### 23.2 软引用与引用闭包预算
场景可含百万级软引用而零 I/O 成本（仅 16 字节 guid）。实例化时按**预算分批** `load_soft`（每帧 N 个），配合依赖闭包就绪事件渐进点亮，规避「打开大世界一次性加载爆内存」。

### 23.3 占位 / 兜底资产（placeholder）
加载中/失败时 `get` 可返回**类型化兜底**（粉色纹理、单位立方体、静音波形）而非 `None`，渲染不中断；`Failed` 事件驱动可见诊断。兜底由类型注册时登记，对标各引擎的「missing asset」占位。

### 23.4 内容目录与远程分发（Addressables 形态）
`StableGuid → ChunkId → 容器（本地/远程）` 的内容目录，支撑：本地包 + 远程 CDN 混合、按需下载 + 本地缓存、版本化目录热更。运行时只认 guid，不认物理位置——内容可重定位。

### 23.5 patch / DLC 叠挂
多容器按优先级叠挂，同 `ChunkId` 高优先级覆盖：发补丁只发增量容器 + 新目录，不重发全量；DLC 挂载即扩展内容闭包。

### 23.6 虚拟纹理 / 虚拟几何页流送对接
超大纹理/网格以 page/cluster 为流送粒度（对接 `prism_virtual_texture_gpu` / `prism_virtual_geometry_gpu`），`read_range` 按 page 拉取，GPU 直通解压直灌页缓存——`prism_asset` 提供「按区段异步取 + 解压」，页表/反馈归渲染侧。

### 23.7 不可信容器/网络字节的安全边界
容器 ToC、远程字节来自不可信源：读取前校验 chunk 长度/偏移在界内、声明长度与实际一致、内容哈希校验（防篡改/损坏）；解压前校验解压后尺寸上限（防「解压炸弹」）。失败走 `AssetErrorId` 返回，不 panic、不 OOM（复用 `prism_reflect` §24.8 不可信反序列化边界的同款防御）。

### 23.8 加载预算与首帧墙优化
区分 `Immediate`（首帧可玩必需）与 `Streaming/Prefetch`（可后补），首帧只等 `Immediate` 闭包就绪即进入可玩，其余后台渐进流入——AAA「秒进游戏」的关键。配合 §23.2 分批与 §11 预取，隐藏加载长尾。
---

## 24. 基准目标（尚非已验证门槛）

下列数值是**待测目标**，不是当前性能承诺或 CI 阻断条件。只有在仓库提供可复现 benchmark、固定资产集、固定硬件/OS/编译配置、预热与采样方法、统计分位和可接受方差后，才可针对对应实现切片设为验收门槛；未实现的里程碑不适用。NVMe + 中端独显仅是未来桌面基线候选，移动/主机需独立校准。

| 维度 | 指标 | 目标门槛 | 关联里程碑 |
|---|---|---|---|
| 身份/句柄 | `load()` 返回句柄（已 interned） | < 1 µs，零堆分配 | M1 |
| 存储 | `get(id)` 命中 | O(1),< 50 ns | M1 |
| 回收 | `collect_releases` 每万待回收 | < 1 ms | M1 |
| 加载吞吐 | 批量小资产（容器内） | ≥ 50k 资产/秒；≥ 2 GB/s 解码 | M2 |
| 加载墙 | 首帧 `Immediate` 闭包就绪 | 典型关卡 < 2 s | M2/§23.8 |
| 依赖就绪 | `LoadedWithDependencies` 延迟 | 深度 N 链 ≈ max(路径) 而非 Σ(全部) | M2 |
| 热重载 | 改纹理→`Modified` 往返 | < 200 ms（去抖后） | M3 |
| 流送换入 | mip/LOD 从请求到驻留 | p99 < 1 帧预算的后台时间 | M4 |
| 流送无卡顿 | 开/关流送的帧时间差 | < 0.5 ms（主线程） | M4 |
| GPU 直通 | vs CPU 解压路径带宽 | ≥ 2× 且 CPU 占用显著下降 | M4 |
| 增量烘焙 | 改 1 资产重烘 | 只烘该资产闭包，DDC 命中其余 | M5 |
| DDC 命中 | 团队/CI 共享命中率 | 常态 > 90% | M5 |
| 读放大 | 实际读/有效用 | 发行包 < 1.1× | M5 |

**CI 转门槛条件**：先提交固定 benchmark 数据集与运行脚本，记录硬件/OS/编译/温控与采样环境，报告 p50/p95/p99 及重复运行方差；至少建立可信基线后再设置告警/阻断阈值。当前表格本身不代表已有 benchmark 或 CI 门禁。

---

## 25. 测试与验证策略

### 25.0 当前实现的最小验收门槛

- 同一 `AssetPath` 的并发 load 必须只产生一次实际读取/解码；等待者取消不得伪造成功或中断仍被其他消费者使用的工作。
- loader 的 malformed、unsupported、dependency failure 和 decode limit 必须保留可诊断错误；loader 不得把未验证的长度/递归/分配请求直接交给下游。
- `RangeSource` 在 reader 生命周期内必须提供稳定内容；完整 `read_all` 必须复验 entry digest，局部 range 在 Merkle/chunk hash 协议实现前不得宣称具有独立密码学认证。
- reload 的新 epoch 只能发布最新结果；旧结果不得覆盖新 revision，失败 staging 不得污染当前可消费 revision。
- 每个“已完成”条目至少绑定一个源码路径、一个回归测试和一条可复现命令；目标规格不计入已完成清单。

资产系统的 bug 多为**并发竞态**与**不可信输入**，普通单测抓不住。分层验证：

- **单元/属性测试**：身份往返（`path↔guid↔index`）、代际失效、依赖图拓扑/环检测、递归状态折叠。用属性测试（`proptest` 形态）对「随机依赖图 → 拓扑序合法 + 环必被检出」做大样本覆盖。
- **并发交错（loom）**:§9.4 调度内核的关键交错——并发 `load` 去重、乱序完成、取消/完成竞争、`ReleaseQueue` MPSC、epoch 失配丢弃——用 `loom` 穷举小规模交错，把「只在特定线程时序出现」的竞态挡在合入前。
- **确定性双跑**：烘焙对关键资产跑两次比对逐字节——纪律与流程唯一真相为 `prism_asset_build_design_zh.md` §18.1，逐项登记见 `docs/prism_derived_data_contract_zh.md`；拓扑序双跑对拍（§16）。任何不一致即判非确定失败。
- **模糊测试（fuzz）**：对容器 ToC 解析、`read_range` 边界、解压入口喂**畸形/截断/恶意**字节（§23.7），断言「只返 `AssetErrorId`，绝不 panic/OOM/越界/解压炸弹」。CI 常驻 fuzz 语料库 + 崩溃回归。
- **耐久/浸泡（soak）**：长时间随机 load/drop/stream in-out，监控内存不泄漏、碎片不膨胀（§7.4）、句柄计数守恒、无悬垂。模拟开放世界来回穿梭的流送抖动。
- **故障注入**:I/O 失败/超时、依赖缺失、半截字节、磁盘满、远程 404，断言失败路径走兜底（§23.3）+ 诊断（§15），不崩运行时。
- **跨平台一致**：同输入在不同 OS/架构烘焙 → 相同 `StableGuid` 与产物哈希，守护身份与烘焙的平台无关性（§16）。
- **基准回归**:§24 的量化门槛入 CI，与功能测试同门。

> 门槛纪律：新增并发路径必附 `loom` 用例；新增解析/解压入口必附 fuzz 目标；新增处理器必附双跑对拍。无此三者不予合入。

---

## 26. API 人体工学深化（派生宏 / 系统参数 / prelude）

易用性是 AAA 协作规模的隐形生产力。目标：**常见事一行、复杂事可达、错误事编译期挡**。

- **`#[derive(Asset)]`**：自动实现 `asset_type()`（取 `prism_reflect` `StableTypeId`）与 `visit_dependencies`（扫描字段里的 `Handle<_>`/`SoftHandle<_>`，含 `Vec`/`Option`/嵌套）。开发者只标注，不手写依赖登记——消除「忘了登记依赖导致提前回收」的经典 bug。

```rust
#[derive(Asset)]
struct Material {
    albedo: Handle<Image>,          // 自动进依赖图
    normal: Option<Handle<Image>>,  // 自动（跳过 None）
    layers: Vec<SoftHandle<Image>>, // 软引用：登记但不保活
    tint: [f32; 4],                 // 非句柄字段跳过
}
```

- **类型化系统参数**：上层经注入式参数访问，不手持全局——
  ```rust
  fn setup(server: Res<AssetServer>, mut meshes: ResMut<Assets<Mesh>>) {
      let h = server.load("hero.gltf#Mesh0");       // 立即得句柄
  }
  fn react(mut ev: EventReader<AssetEvent<Scene>>) { /* 只收 Scene 事件 */ }
  ```
- **就绪即回调（ergonomic await）**：除事件轮询，提供 `server.when_loaded(handle, |asset| { ... })` 与 `handle.loaded().await`（在 `prism_tasks` 上），按喜好选事件/回调/await，底层同一通知。
- **类型安全防呆**：`AssetId<A>` 带类型标签，`UntypedAssetId` 带 `AssetTypeId`——错类型 `typed::<B>()` 返 `None` 且可诊断（§5.1），不静默命中错 arena。软引用 `resolve` 保留期望类型，类型不符立即可报。
- **`prism_asset::prelude`**：一处导出 `Handle/WeakHandle/SoftHandle/UntypedHandle`、`AssetServer`、`Assets`、`AssetEvent`、`AssetLoader`、`LoadContext`、`AssetPath`、`LoadState`、派生宏，`use prism_asset::prelude::*;` 即可。
- **清晰错误**：`AssetErrorId → {path, source, reason, dependents}`，面板按根因聚合（§15）；错误信息带「从哪 load 的、缺哪个依赖、影响谁」，不是裸 `None`。
- **编辑器/脚本暴露**：经 `prism_reflect` 把句柄/加载态/依赖关系暴露给编辑器与脚本层，资产浏览器、引用查看、热改即见，无需各上层重写一套。

> 人体工学铁律：默认路径零样板（load 一行、依赖自动、就绪有事件/回调/await 三选一）；危险操作编译期或类型层拦截；出错时诊断可读可聚合。

---
