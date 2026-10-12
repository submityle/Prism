# Prism Derived Data Contract：派生数据契约登记表

> 本文是 Prism 全引擎**派生数据**（烘焙产物、缓存、代理、预计算场）的唯一登记表。它的存在是为了消除一个结构性缺陷：`prism_asset_build_design_zh.md` 与 `prism_asset_design_zh.md` 各自定义过一套缓存键与 DDC，而 `prism_world_system_design_zh.md`、`prism_scene_design_zh.md`、`prism_terrain_world_design_zh.md`、`prism_gi_lumen_design_zh.md`、`prism_material_pipeline_refactor_zh.md` 又各自把派生数据写成“烘焙期”，彼此之间没有统一的键与归属契约。

- 文档状态：架构提案（Draft，允许破坏性调整）
- 规范来源：`docs/prism_asset_build_design_zh.md` §2.7（离线制品 vs 进程内派生）、§4.6（`DerivationClass`）、§6.6（Bake Kernel 与两种调度）、§12.8（派生场景数据 Recipe）
- 关联文档：`docs/prism_asset_design_zh.md`（运行时消费契约）、`docs/prism_world_system_design_zh.md`、`docs/prism_scene_design_zh.md`、`docs/prism_terrain_world_design_zh.md`、`docs/prism_gi_lumen_design_zh.md`、`docs/prism_material_pipeline_refactor_zh.md`、`docs/prism_animation_engine_design_zh.md`、`docs/prism_audio_engine_design_zh.md`、`docs/prism_hair_engine_design_zh.md`、`docs/prism_cloth_engine_design_zh.md`、`docs/prism_volumetric_engine_design_zh.md`、`docs/prism_game_creator_design_zh.md`

---

## 1. 契约字段

每一份派生数据必须填写以下五元组，外加实现归属：

| 字段 | 含义 | 约束 |
|---|---|---|
| **输入集合** | 该派生实际读取的全部输入 | 未声明的输入访问视为可重复性违规（Asset Build §6.1） |
| **键材料** | 参与 `ArtifactKey` 的组成部分 | 必须内容可派生；禁时间戳/随机数/未排序并行序/绝对路径/机器状态（§18.1） |
| **失效触发** | 什么变化会使其失效 | 源变 / 设置变 / 内核版本 / Profile / Schema / 上游 Artifact 变（§8.3） |
| **DerivationClass** | 调度与持久化等级 | §4.6 四值，判定顺序=纯函数性→发行/共享价值→时延容忍（§2.7） |
| **确定性等级** | 输出可复现的强度 | `Byte Reproducible` / `Deterministic-not-byte` / `Best-effort` |
| **实现归属** | 持有内核与调度权的 crate | 算法内核为 `prism_bake_*`；离线调度归 `prism_asset_build_*`（§6.6） |

### 1.1 确定性等级

- **Byte Reproducible**：同输入、同设置、同内核版本、同 Profile → 逐字节相同。共享发布 DDC 与 Patch 差分的前提。
- **Deterministic-not-byte**：同输入下算法确定，但允许声明的差异（GPU 非位精确、并行归约顺序已排序但仍非位等）。必须写明差异与容差；不得进共享发布 DDC，或必须经后续确定性规范化节点消除差异（Asset Build §18.3）。
- **Best-effort**：可弃缓存，不保证跨机/跨会话一致。仅允许 `RuntimeDerived` / `DriverLocal`。

### 1.2 与 `DerivationClass` 的对应

| DerivationClass | 进 DDC | 进 Manifest/发行包 | 确定性下限 |
|---|---:|---:|---|
| `OfflineShipped` | 共享 | 是 | `Byte Reproducible`（特殊项可 `Deterministic-not-byte` 但需显式豁免） |
| `OfflineEditorOnly` | 本地/共享 | 否 | `Deterministic-not-byte` 起 |
| `RuntimeDerived` | 否 | 否 | `Deterministic-not-byte` 起 |
| `DriverLocal` | 否 | 否 | `Best-effort` |

---

## 2. 登记表

### 2.1 源资产派生（A 组）

| # | 派生数据 | 键材料（除源内容外） | 失效触发 | 类 | 确定性 | 实现归属 |
|---|---|---|---|---|---|---|
| A1 | 纹理 mip / 平台压缩（BCn/ASTC/ETC2） | 源哈希 + ImportSettings + 平台格式 + Quality Profile + 内核版本 | 源/设置/内核/Profile | `OfflineShipped` | Byte Reproducible | `prism_bake_texture` |
| A2 | 静态/蒙皮网格、meshlet、虚拟几何页 | 源哈希 + 导入/优化设置 + 平台 + 内核版本 | 源/设置/内核/Profile | `OfflineShipped` | Byte Reproducible | `prism_bake_mesh` |
| A3 | 材质层栈烘焙 → OpenPBR über 参数块 | 作者图/层栈哈希 + über Schema + 内核版本 | 图/Schema/内核 | `OfflineShipped` | Byte Reproducible | `prism_render_material::bake` |
| A4 | 打包期瓣单态化（静态 WESL + 每材质特化） | 材质块哈希 + PCM 清单 + WESL 源哈希 + 内核版本 | 材质/PCM/WESL/内核 | `OfflineShipped` | Deterministic-not-byte（GPU 编译路径需声明） | `prism_bake_shader` |
| A5 | 骨骼动画压缩（量化/范围压缩/常量轨折叠） | 源哈希 + 误差阈值 + 帧率策略 + 内核版本 | 源/设置/内核 | `OfflineShipped` | Byte Reproducible | `prism_bake_animation` |
| A6 | 音频条件化（响度 BS.1770 / 无缝循环点 / 重采样 / 编码 / 预滚） | 源哈希 + 条件化参数 + 目标码率/格式 + 内核版本 | 源/设置/内核 | `OfflineShipped` | Byte Reproducible | `prism_bake_audio`、`prism_audio_wave::bake` |
| A7 | 音频传播 / ARD 探针参数场 | 探针几何哈希 + 声学材质 + 求解参数 + 内核版本 | 几何/材质/参数/内核 | `OfflineShipped` | Deterministic-not-byte（GPU 求解需声明） | `prism_audio_wave::bake` |
| A8 | OMM opacity micromap | alpha 掩码哈希 + 细分等级 + 格式 + 内核版本 | 掩码/设置/内核 | `OfflineShipped` | Byte Reproducible | `prism_micromap_gpu::bake` |
| A9 | 字形 SDF 图集（发行字体） | 字体文件哈希 + `bake_em`/`padding`/`px_range` + 内核版本 | 字体/设置/内核 | `OfflineShipped` | Byte Reproducible | `prism_ui_font::sdf` |
| A10 | Scene/Prefab → 实例化计划、World Partition Cell | scene 内容哈希 + 预合成开关 + Schema + 内核版本 | 内容/设置/Schema/内核 | `OfflineShipped` | Byte Reproducible | `prism_scene` + Asset Build §12.6 |
| A11 | 毛发 Groom→card LOD 烘焙（density/flow/depth 图集） | groom 哈希 + LOD 档 + atlas 布局策略 + 内核版本 | 源/档位/内核 | `OfflineShipped` | Byte Reproducible | Asset Build §12.7 + `hair/card_bake.rs` |
| A12 | 布料高模→模拟网格映射、约束、碰撞代理 | 源哈希 + 布料设置 + 平台 Solver Profile | 源/设置/Profile | `OfflineShipped` | Deterministic-not-byte（求解器平台差异按 Profile 声明） | Asset Build §12.7 |
| A13 | 玩法数据烘焙（掉落表 / 技能树 / 任务图 → 紧凑运行时结构） | 作者数据哈希 + Schema + 内核版本 | 数据/Schema/内核 | `OfflineShipped` | Byte Reproducible | Asset Build 通用 Recipe（§7/§12）+ `prism_game_creator_design_zh.md` |

### 2.2 场景与世界派生（B 组）

| # | 派生数据 | 键材料（除源内容外） | 失效触发 | 类 | 确定性 | 实现归属 |
|---|---|---|---|---|---|---|
| B1 | 烘焙 GI / 光照缓存（可选，主路径为实时无烘焙） | scene 内容 + 种子 + 光照规则 + Schema | 内容/规则/Schema | `OfflineShipped` | Deterministic-not-byte | Asset Build §12.8 + GI 子系统（`prism_gi_lumen_design_zh.md`） |
| B2 | 导航网格分块 | scene 内容 + 代理半径/高度 + 区域代价 + Schema | 内容/参数/Schema | `OfflineShipped` | Byte Reproducible | Asset Build §12.8 |
| B3 | 全局/局部 SDF | scene 内容 + 体素尺寸 + 带宽 + Schema | 内容/参数/Schema | `OfflineShipped` | Byte Reproducible | Asset Build §12.8 |
| B4 | HLOD 代理（合并网格 / impostor / 体素壳） | scene 内容 + 成簇规则 + 种子 + 屏幕误差 | 内容/规则/种子 | `OfflineShipped` | Deterministic-not-byte（GPU 合并需声明） | Asset Build §12.8 + `prism_scene` §11.4 |
| B5 | PCG 散布 proxy 列 | scene 内容 + 节点图 + 种子 + 规则版本 | 内容/图/种子 | `OfflineShipped` | Byte Reproducible | Asset Build §12.8 |
| B6 | 地形侵蚀 / 远景底图 / RVT 烘底 | 高度/掩码 + 迭代参数 + 种子 + Schema | 地形/参数/种子 | `OfflineShipped` | Deterministic-not-byte | Asset Build §12.8 + 地形子系统（`prism_terrain_world_design_zh.md`，现命名 `bevy_terrain*`） |
| B7 | 物理离线轨迹缓存（软体/破碎） | 初始状态哈希 + 积分参数 + 帧数 + 内核版本 | 状态/参数/内核 | `OfflineShipped` | Deterministic-not-byte（定点量化后趋近位等） | `prism_physics_core::cache::bake` |
| B8 | 静态变换烘焙（世界矩阵 / 合并网格） | scene 内容 + `TransformStatic` 标记 + Schema | 内容/Schema | `OfflineEditorOnly` | Byte Reproducible | `prism_transform` §24.1 |
| B9 | 预断裂碎片簇（Voronoi 预切，可破坏对象） | scene 内容 + 切割规则 + 种子 + Schema | 内容/规则/种子 | `OfflineShipped` | Deterministic-not-byte | `prism_world_system_design_zh.md` §16 + Asset Build（同 §12.8 DAG） |

### 2.3 进程内派生（C 组）

| # | 派生数据 | 触发 | 类 | 确定性 | 实现归属 |
|---|---|---|---|---|---|
| C1 | dirty-cell 导航网格重烘（破坏/编辑后） | 脏区集合 + 场景 Revision | `RuntimeDerived` | Deterministic-not-byte | `BakeService` + B2 同一内核 |
| C2 | 运行时 SDF 场更新（动态物体） | 动态物体集合 + 帧 tick | `RuntimeDerived` | Deterministic-not-byte | `BakeService` + B3 同一内核 |
| C3 | 运行时虚拟纹理页（RVT 写入） | 视点/地形改动 + 页映射 | `RuntimeDerived` | Best-effort | 渲染侧 |
| C4 | 体积密度缓存（远景性能档） | 体素砖 + 质量档 | `RuntimeDerived` | Deterministic-not-byte | `prism_render_architecture` / `prism_render_scene` 体积模块 |
| C5 | 粒子曲线/渐变 LUT | 作者曲线哈希（可提升为 A 组随包发行） | `RuntimeDerived` | Byte Reproducible | `prism_render_architecture::particle::curves` |
| C6 | 毛发 deep opacity / deep shadow map（光源视角） | 光源方向 + 密度场 + 帧 tick | `RuntimeDerived` | Best-effort | `prism_render_architecture::particle::deep_opacity_bake` |
| C7 | 编辑器 HLOD 预览代理 | 编辑器视点 + 选中集 | `OfflineEditorOnly` | Best-effort | 编辑器框架 §D7 |
| C8 | 波形缩略图 / 编辑器分析数据 | 音频源哈希 + 分析参数 | `OfflineEditorOnly` | Byte Reproducible | Asset Build §12.5 |
| C9 | 字形 SDF 随用随烘（回退字体 / 非预置字符集） | 字体哈希 + 字符集 | `RuntimeDerived` | Deterministic-not-byte | `prism_ui_font::sdf` + A9 同一内核 |

### 2.4 驱动本地（D 组）

| # | 派生数据 | 键材料 | 类 | 确定性 | 实现归属 |
|---|---|---|---|---|---|
| D1 | PSO / Pipeline Cache | 材质特化类 + 固定功能状态 + 驱动版本 + GPU | `DriverLocal` | Best-effort | `prism_render_app::pipeline_cache` |
| D2 | 驱动着色器二进制缓存（DXIL/Metal/SPIR-V） | 着色器源哈希 + 驱动版本 + GPU | `DriverLocal` | Best-effort | 渲染驱动侧 |

### 2.5 不参与（E 组）

| # | 数据 | 说明 |
|---|---|---|
| E1 | Lumen 式实时 GI 主路径 | 明确“实时无烘焙”（`prism_gi_lumen_design_zh.md` §0）；不产生派生数据。B1 仅为可选离线缓存，二者不冲突 |
| E2 | 每帧程序化求值（体积极小声、PCG 运行时求值、水面/天气 sim） | 属于运行时求值，不是派生数据；若被持久化则必须按 C 组登记 |

---

## 3. 硬约束

1. **禁止第二实现**：同一份派生数据的 `OfflineShipped` 与 `RuntimeDerived` 路径必须指向同一 Bake Kernel（§6.6）。表中“实现归属”列的重复引用即为该约束的体现。
2. **禁止键材料泄入非确定项**：`Best-effort` 项不得进共享 DDC；`DriverLocal` 项永不进 Manifest。
3. **未登记项即缺陷**：任何文档中出现的“烘焙期/预计算/离线生成”字样，若未在本表登记，视为契约缺口，须补登或改为运行时求值。
4. **确定性升级路径**：`Deterministic-not-byte` 项若要进共享发布 DDC，必须先有后续确定性规范化节点（Asset Build §18.3），并在本表标注该节点。
5. **变更纪律**：新增/改名派生数据、改变 `DerivationClass`、或调整键材料，必须在同一改动中更新本表与对应子系统文档。

---

## 4. 验收

- 静态：本表每一项在 `docs/prism_asset_build_design_zh.md` §12.8 或对应子系统文档中有反向链接。
- 动态：对每一个 `OfflineShipped` 项，CI 双跑产出逐字节一致（`prism_asset_build_design_zh.md` §18.1/§18.3/§23）；对每一个 `RuntimeDerived` 项，与同内核的离线路径做同输入一致性对拍，差异落在声明容差内。
