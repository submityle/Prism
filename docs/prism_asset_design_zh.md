# Prism Asset 顶级次世代 AAA 级资产内核设计方案

> 面向 Prism（脱离 Bevy 后的独立引擎）的 **资产身份 + 句柄/引用计数 + 代际存储 + 依赖图 + 异步加载器 + 虚拟文件系统 + 事件驱动加载调度 + 按需流送 + 热重载 + 离线烘焙/打包** 内核设计。它是 `bevy_asset` 的自研替代，是渲染（网格/纹理/材质/着色器）、音频、场景/预制体、动画、UI 的统一内容基座。
> 借形态不抄码。借鉴：
> - **Rust 资产人体工学**：Bevy asset（`Handle`/`AssetServer`/`AssetLoader`/`Assets<A>`/`AssetEvent`/标签子资产/`AssetProcessor` 烘焙/热重载）
> - **容器与事件驱动加载**：Unreal IoStore（`FIoChunkId` 容器化分块 I/O）+ Zen/EDL（Event-Driven Loader 异步装配）+ Asset Registry（不加载即可查元数据）+ `.pak`/cook 管线
> - **引用寻址与异步句柄**：Unity Addressables（内容目录 + 异步 `AsyncOperationHandle` + 引用计数释放）、GUID+fileID 的稳定寻址与 `.meta` 导入设置
> - **零拷贝/GPU 直通流送**：DirectStorage + RTX IO + **GDeflate** GPU 解压（对接本仓 `prism_gdeflate_gpu`）、mip/LOD 预算驱动驻留（DOOM Eternal / Decima 形态）
> - **调度/并发**：对接本仓 `prism_tasks`（work-stealing + 优先级 + 背压）承载加载作业图
> 本文为纯经典资产系统/内容管线路线，**不含任何 AI/ML 内容**，不含任何 Unreal/Unity 源码或衍生代码。

- 版本： v0.1（现状：M0 身份/存储/依赖核已落地并单测——`AssetIndex`/`AssetId`/`UntypedAssetId`、`AssetPath`、`Handle`/`WeakHandle`/`UntypedHandle`、代际 `Assets<A>` + `AssetEvent` + `remove_unused`、`DependencyGraph`（Kahn 拓扑 + 环检测）、`LoadState`/`RecursiveDependencyLoadState`，`no_std + alloc`、无 `unsafe`。本文为**全量重构蓝图**：不保留旧 API，身份/句柄/存储/依赖核在 M1 起按本文重铸，M1–M6 新增加载器/VFS/调度/流送/热重载/烘焙打包）
- 适用引擎： Prism（后 Bevy 时代，独立运行时）
- 关键依赖： `prism_utils`（稳定哈希/句柄位打包/小容器）、`prism_tasks`（异步作业图/优先级/背压）、`prism_reflect`（导入设置/元数据序列化）、`prism_diagnostic`（计数器/trace）、可选 `prism_gdeflate_gpu`（GPU 解压）
- 层级定位： 内容层 L5；上接场景/渲染/音频/动画/UI，下接 `prism_tasks`/`prism_vfs`/平台 I/O
- 明确约束： 身份/存储/依赖/句柄核 `no_std + alloc`；`std`/`async_io`/`hot_reload`/`process`/`pack`/`gpu_stream`/`retain_cache` 为 feature；工作区 `forbid(unsafe_code)`；**不依赖任何 `bevy_*` crate**

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
| **处理** | `process` | 导入设置（单文件资产包内嵌）+ 内容哈希缓存 + 离线烘焙 | 内容流水线 / CI |
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
    fn reserve(&mut self, index: AssetIndex);           // 占位（Pending），由 server 调
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

### 9.1 AssetLoader 协议

```rust
pub trait AssetLoader: Send + Sync + 'static {
    type Asset: Asset;
    type Settings: Default + Send + Sync;      // 来自资产包内嵌设置 / 代码
    type Error: Into<AssetErrorId>;

    /// 异步装配：读字节 + 解码 + 声明依赖。不阻塞调度线程。
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

**I/O 调度器（吃满带宽、压读放大）：**

- **请求合并（coalescing）**：同容器相邻/重叠的 `read_range` 合并成一次大读，摊薄 syscall/队列开销；间隙小于阈值时「宁可多读一点」也不拆成两次（减少 IOP）。
- **按偏移重排（elevator）**：批内请求按 `data_offset` 排序顺序发射，逼近顺序读带宽，避免随机寻道（SSD 上降低队列深度压力）。
- **深队列异步**：DirectStorage 形态下维持高队列深度（成百请求在飞），完成走 fence/回调驱动 §9.4 作业图，不逐个同步等待。
- **读放大预算**：诊断暴露「实际读字节 / 有效使用字节」比，打包分组（§13.3）以最小化该比为目标（相关资产同容器、同簇）。

---

## 11. 按需流送与 GPU 直通解压

### 11.1 流送管理器

大世界/高清资产装不下显存，必须按需换入换出。`StreamingManager` 对标 DOOM Eternal / Decima 的预算驱动流送：

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
- **GPU 解压**：压缩 blob 直灌显存暂存区，`prism_gdeflate_gpu` 的计算着色器就地解压，CPU 不碰资产字节。
- **优雅降级**：平台无 DirectStorage/无 GPU 解压则回退 CPU 路径（`async_io` + CPU zstd/gdeflate），上层无感——档位铁律（§3）。

### 11.3 与渲染驻留的边界

`prism_asset` 负责「字节到达 + 层级决策 + 解压」，**GPU 资源对象的创建/销毁**归渲染侧（`prism_render_*`）：流送管理器发「层级 N 的字节已就绪」事件，渲染侧据此建/换 `wgpu` 纹理视图。两者经句柄 + 事件解耦，`prism_asset` 不依赖任何渲染 crate（保持 L5→L4 单向）。

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

- **变更侦测**：`AssetSource::watch` 接平台 FS 事件（inotify/FSEvents/ReadDirectoryChangesW，经 `prism_platform` 隔离），去抖后产出「路径 X 变更」。
- **失效传播**：路径 → `StableGuid` → `AssetIndex`，经 §8.1 依赖图**反向遍历** dependents，把该资产及其所有传递依赖者标 `stale`。例：`tex.png` 改 → 用它的 `material` stale → 用该材质的 `scene` 的受影响部分 stale。
- **重装配**：stale 资产重走 §9.3 装配，完成后**原地替换**存储槽的值（同 `AssetIndex`、句柄不变），发 `Modified` 事件。消费方收 `Modified` 自行刷新（渲染重建 GPU 资源、UI 重绘）。句柄身份不变是关键——热改纹理不该让所有持有者的句柄失效。
- **原子性与一致性**：重装配在后台完成后一次性 swap，不出现「半更新」中间态；失败则保留旧值 + 发诊断（§15），不让坏文件搞崩运行时。
- **依赖结构变更**：若重装配后依赖集变化（材质新增一张纹理），依赖图增量更新边并触发新依赖加载。

---

## 13. 离线烘焙 / 导入 / 打包管线

`feature = "process"`（烘焙）+ `pack`（打包）。把「源资产」变成「平台最优的运行时资产」，对标 Unreal cook + Unity import/Addressables build。

### 13.1 单文件资产包（UE 形态：一个资产一个文件，无旁车）

采用 **UE 的「单文件资产包」模型**：编辑器里每个资产就是**一个自包含文件**（`.prism` 资产包），导入设置、身份、源引用、可编辑载荷全部**内嵌在同一个文件里**——没有 `.meta` 旁车，没有散落的隐藏清单。目录里「一个文件 = 一个资产」，所见即所有。这是对标 Unreal 的 `.uasset`/`.umap`：把原始源文件（psd/fbx/wav）**导入**成引擎自己的资产包后，包就是资产的唯一真相。

**资产包的内嵌布局（单文件自描述）：**

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

> 诚实权衡：单文件是**二进制自包含**，不可逐行 diff，合并冲突按**整资产粒度**解决（UE 同款局限）。缓解：① 资产级独占签出/锁（内容索引记锁位），团队按资产而非按行协作；② 序列化**确定性**（§13.4：字段定序、无时间戳/无不定并行序）——同编辑产同字节，减少伪冲突；③ 可选把纯参数型小资产导出为文本旁视图供 review，真相仍以包为准。相比旁车/中心清单，单文件换来的是**目录干净 + 身份稳健 + 读写原子**，这正是你要的 UE 手感。

### 13.2 处理器与内容哈希缓存

```rust
pub trait AssetProcessor: Send + Sync {
    type Settings: Default;
    async fn process(&self, input: &mut dyn AssetReader, settings: &Self::Settings,
                     out: &mut ProcessOutput) -> Result<(), ProcessError>;
}
```

- **增量**：缓存键 = `hash(源内容) + hash(settings) + 处理器版本`。键命中则跳过（直接复用上次产物）——改一个资产不必重烘全库。对标 Unreal DDC（Derived Data Cache）/ Bazel 内容寻址，但以 `StableGuid`+内容哈希自建，不绑外部系统。
- **确定性**：同输入同设置同版本 → 逐字节相同产物（便于 CI 缓存共享、便于「双跑一致」回归）。处理器须声明版本号，逻辑变更即 bump，自动失效旧缓存。
- **产物身份**：烘焙产物 `StableGuid` 由内容哈希派生 → 相同产物自动去重（两材质引用同一处理后纹理，容器里只存一份）。

### 13.3 打包

- 扫描「发行所需资产闭包」（从入口场景/清单经依赖图求传递闭包，剔除仅编辑期资产）。
- 按**分组/chunk 策略**（对标 Addressables groups）切成多个容器：基础包、按关卡/区域、按 DLC。
- 产出 §10.3 容器（ToC + 压缩分块 blob）+ 内容目录（`StableGuid → ChunkId → 容器`），运行时据目录定位。
- 支持 patch 包（增量容器 + 高优先级挂载覆盖），不重发全量。

---

### 13.4 派生数据缓存（DDC）与确定性纪律

增量烘焙（§13.2）要在**团队与 CI 间共享**才有 AAA 价值——一人烘过，全队命中。借鉴 UE DDC2 / Bazel 远程缓存的内容寻址思想：

- **三级缓存**：本地内存 → 本地磁盘（`~/.prism/ddc`）→ 共享远程（CI/团队 CDN，只读或读写）。缓存键 = `blake3(源内容 ⊕ settings 规范化 ⊕ 处理器版本 ⊕ 目标平台)`，键即内容地址，产物按键寻址存取。
- **拉取而非重算**：命中远程即下载产物，miss 才本地烘焙并回写——改一个资产、换一个平台，大多数产物直接命中，CI 烘焙从「全量数小时」降到「增量数分钟」。
- **确定性纪律（硬约束）**：处理器 `process` 必须是纯函数——
  - 禁时间戳/随机数/未排序并行序/绝对路径/浮点非确定（指定舍入、禁 FMA 差异）混入产物；
  - 需随机则用**内容派生种子**（`hash(input)`），保证同输入同输出；
  - 并行处理结果须**排序归并**后写出，消除线程完成序影响。
- **版本门控**：每处理器声明 `PROCESSOR_VERSION`，逻辑变更必 bump → 自动失效旧键，不会误用过期产物。
- **双跑对拍**：CI 对关键资产**跑两次比对逐字节**(§25)，任何不一致即判「处理器非确定」失败——把确定性退化挡在合入前。
- **产物去重**：产物身份由内容哈希派生（§5.2），两资产烘出相同产物自动共享一份（容器只存一份，ChunkId 相同）。

> 收益：确定性 + 内容寻址 = 增量 + 可共享 + 可去重 + 可回归，四者同根。这是 AAA 内容管线「几万资产、几十人协作、每日构建」能转起来的地基。

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
- **烘焙确定**：同输入逐字节同产物（§13.2），CI 可共享缓存。
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
    hot_reload                   # watch + 失效传播 + 重装配
    process                      # AssetProcessor + 内容哈希缓存 + 单文件资产包
    pack                         # ContainerIo（ToC + 压缩分块）+ 挂载优先级
    gpu_stream                   # DirectStorage 形态队列 + GDeflate（prism_gdeflate_gpu）
    retain_cache                 # 保留缓存：KeepAlive/LRU/Pinned 卸载策略（防 load/drop 抖动）

pkg/prism_asset_import/          # 具体格式 loader：glTF/FBX/USD、PNG/KTX2/basis、WAV/OGG、字体
pkg/prism_asset_bake/            # 平台烘焙：BCn/ASTC 压缩、mesh 优化/量化、虚拟几何/纹理预处理
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

## 21. 路线图（M0–M6）与基准即规格

- **M0 身份/存储/依赖核（已落地）**：`AssetIndex`/`AssetId`/`UntypedAssetId`、`AssetPath`、`Handle`/`WeakHandle`/`UntypedHandle`、代际 `Assets<A>` + `AssetEvent` + `remove_unused`、`DependencyGraph`（Kahn + 环检测）、`LoadState`。`no_std+alloc`、无 `unsafe`、单测绿。
- **M1 内核重铸（本文核心改动）**：三层身份（+`StableGuid`/`AssetTypeId`）、`SoftHandle`、`AssetTypes` 类型擦除注册、`Asset` trait + `visit_dependencies`、`reserve/fulfill/fail` + 延迟回收 `ReleaseQueue`、依赖图增量就绪 + 失效接口、`AssetErrorId`/`ErrorRegistry`、`AssetPath<'a>` 借用 + `source://`。**废弃旧 insert/remove_unused/Failed(String)。**
- **M2 加载管线（`std`+`async_io`）**：`AssetServer`（path↔guid↔index interning + load 去重）、`AssetLoader` trait + `LoadContext`（子资产/依赖）、`LoadScheduler` EDL 作业图（跑 `prism_tasks`）、`AssetSource`/`AssetReader`（`FsSource`/`MemSource`）、优先级 + 背压。
- **M3 热重载（`hot_reload`）**：`watch` + 去抖、失效反向传播、后台重装配 + 原子 swap + `Modified`、依赖结构变更处理。
- **M4 流送（`gpu_stream` 可选）**：`StreamingManager` 预算驱动 mip/LOD 驻留 + 请求打分 + 滞回 + 预取；`read_range` bulk 流送；DirectStorage 形态队列 + GDeflate GPU 解压（`prism_gdeflate_gpu`）+ CPU 回退。
- **M5 烘焙/打包（`process`+`pack`）**：单文件资产包（内嵌导入设置）、`AssetProcessor` + 内容哈希缓存（增量+确定）、`ContainerIo`（ToC+压缩分块）+ 挂载优先级 + 内容目录、依赖闭包打包 + 分组 + patch 包。
- **M6 集成**：ECS 资源/事件接线、渲染驻留事件桥、`prism_scene` 软引用实例化、编辑器/脚本经 `prism_reflect` 暴露、`prism_asset_import`/`prism_asset_bake` 对接。

**基准即规格**：每里程碑以基准验收——M2 加载吞吐与加载墙、M3 热重载往返时延、M4 流送换入延迟/命中率/无卡顿、M5 增量烘焙耗时与打包体积/读放大。核心价值在 **M2（异步装配）+ M4（流送）+ M5（烘焙打包）**。

---

## 22. 诚实边界与风险

- **现状**：仅 M0 内核落地（身份/存储/依赖/事件/加载态，单测绿、`no_std`、无 `unsafe`）。M1–M6 为本文蓝图，**尚未实现**；本文是重构方向而非既成事实。
- **高风险项**：
  1. **StableGuid 冻结（M1/M5）**：派生规则一旦发行即不可变，变则旧存档/容器目录失效。须早定早冻、哈希算法与规范化规则定死并测「跨构建一致」。
  2. **异步装配正确性（M2）**：EDL 作业图 + 依赖增量就绪的竞态（并发 load 去重、依赖边并发更新、失败半装配回滚）是最易出错处，须压测并发加载 + 乱序完成。
  3. **流送无卡顿（M4）**：换入换出的原子替换、滞回参数、预算打分若调不好，表现为纹理突变/卡顿/显存抖动——须真实场景调参并以帧时间基准守护。
  4. **GPU 直通可移植（M4）**：DirectStorage/GDeflate 平台差异大，CPU 回退路径必须始终可用且经测，不能让「高端路径」成为唯一路径。
  5. **烘焙确定性（M5）**：处理器非确定（浮点/并行序/时间戳混入）会毁掉增量缓存与 CI 共享；须纪律性地禁非确定输入并双跑对拍。
  6. **热重载依赖结构变更（M3）**：重装配后依赖集变化的增量图更新 + 环重检测，边界多，须覆盖「新增/删除依赖、变环」用例。
- **与既有文档关系**：本 crate 是渲染（网格/纹理/材质/着色器驻留）、音频、`prism_scene`、动画、UI 的内容基座；软引用/依赖闭包就绪/热重载 `Modified` 的契约须与这些消费方一致。身份/存储核已对齐 `prism_engine_component_gap_zh.md` §3.3 的 `prism_asset` 定位（句柄/异步加载/依赖图/热重载/引用计数/生命周期），导入/烘焙按该文拆 `prism_asset_import`/`prism_asset_bake`。
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

## 24. 基准即规格：量化目标与验收门槛

「基准即规格」要有数才算规格。下列目标为**验收门槛**（参考 NVMe SSD + 中端独显 PC 基线；移动/主机档位单独定标），每里程碑由基准守护，回归超阈值即红。数值随硬件基线校准，此处给工程量级而非终值。

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

**基准即 CI 门槛**：关键路径基准入 CI，设阈值与方差带；超阈值阻断合入。基准场景固定（可复现数据集），与确定性双跑（§25）同一套 CI。

---

## 25. 测试与验证策略

资产系统的 bug 多为**并发竞态**与**不可信输入**，普通单测抓不住。分层验证：

- **单元/属性测试**：身份往返（`path↔guid↔index`）、代际失效、依赖图拓扑/环检测、递归状态折叠。用属性测试（`proptest` 形态）对「随机依赖图 → 拓扑序合法 + 环必被检出」做大样本覆盖。
- **并发交错（loom）**:§9.4 调度内核的关键交错——并发 `load` 去重、乱序完成、取消/完成竞争、`ReleaseQueue` MPSC、epoch 失配丢弃——用 `loom` 穷举小规模交错，把「只在特定线程时序出现」的竞态挡在合入前。
- **确定性双跑**：烘焙对关键资产跑两次比对逐字节（§13.4）；拓扑序双跑对拍（§16）。任何不一致即判非确定失败。
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
