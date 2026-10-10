# Prism 渲染引擎 — 次世代 AAA 级世界与地形系统完整设计（v1 / wgpu + WESL）

> 状态：架构提案（Draft，允许破坏性重构）
> 定位：AAA / 次世代 **世界与地形系统**（大世界分区 + 多表示地形 + 程序化内容生成 PCG + 海量植被/散布 + 流送 + 侵蚀/地貌 sim + 碰撞），与 PBR/NPR/自定义/混合四前端正交协同，四前端同享共享高级基底（虚拟几何 vis-buffer / Lumen 式混合 GI / ReSTIR DI-GI / VSM / froxel 体积 / RVT 运行时虚拟纹理 / RT 参考 / 时序上采样 / 大气散射服务）。
> Shader：WESL（naga → WGSL/SPIR-V/Metal/DXIL）；运行时：wgpu（Metal/VK/DX12/WebGPU）
> 关联文档：`prism_rendering_architecture_zh.md`（总，GPU Scene / 虚拟几何 / 虚拟资源 / 虚拟阴影 / Render Graph）、`prism_material_pipeline_design_zh.md`（§6.2 子系统注册表与档位判据）、`prism_gi_lumen_design_zh.md`（L0 场景表示接入）、`prism_water_engine_design_zh.md`（河流/湖海/岸线耦合）、`prism_volumetric_engine_design_zh.md`（雾/云影/天气耦合）、`prism_particle_engine_design_zh.md`（散布飞沫/尘/落叶委托）、`prism_physics_design_zh.md`（地形碰撞 / heightfield / 体素破坏）、`prism_gameplay_design_zh.md`（导航 / 任务 / 流送触发）
> 沙盒说明：本机屏蔽 Metal（无 GPU），文档内帧预算/带宽/显存/耗时均为**设计目标（design target），非实测**，如实标注；可验证部分限于 CPU 可计算纯函数（分区索引、LOD 误差、四叉树切分、噪声/侵蚀核）。
> 数值路线：纯经典数值（高度场/CDLOD/clipmap/四叉树、Transvoxel/Dual Contouring/SDF、Perlin-Worley/domain warp 噪声、水力/热力侵蚀 PDE、泊松盘/蓝噪声散布），不含任何 AI/ML。

---

## 0. 定位：这是什么级别的世界/地形系统

是**完整的世界与地形系统**（大世界空间管理 + 多表示地形几何 + 程序化生成 + 海量实例散布 + 流送驻留 + 地貌仿真 + 碰撞导航），不是"一张高度图贴个 splat 材质"的地形贴皮。对标业界成熟产品，仅在**算法层**借鉴，不复制其代码：

| 产品 | 借鉴点（算法层） |
|---|---|
| **UE5 World Partition / Data Layers / HLOD / OFPA / Level Instance** | 隐式网格分区、按单元流送、数据层开关、分层 HLOD 代理、每 Actor 单文件协作、关卡实例复用 |
| **UE5 Landscape + Nanite Landscape + Virtual Heightfield Mesh (VHM)** | 多组件地形、Landscape layer weight blend、Nanite 化地形连续细分、虚拟高度场网格 GPU 评估 |
| **UE5 Runtime Virtual Texture (RVT)** | 地形/贴花/样条写入 RVT，着色与散布统一采样，远景材质烘焙缓存 |
| **UE5 PCG Framework / Houdini + HDA** | 节点图程序化生成、点云散布、密度/规则驱动、与关卡确定性烘焙、运行时局部重算 |
| **Decima（Horizon Zero Dawn / Forbidden West）** | GPU-driven 程序化植被、tile 流送、生物群系规则、placement density map、海量植物 |
| **RDR2 / RAGE** | 大世界无缝流送、LOD/imposter 过渡、地表细节纹理混合、远景地形烘焙、路网/河流样条 |
| **CryEngine / Far Cry** | 体素地形编辑、层混合、程序化植被笔刷、clipmap 远景、地形 occlusion |
| **Microsoft Flight Simulator / Star Citizen / Outerra** | 行星级球面四叉树（CDLOD on sphere）、相机相对坐标 / 浮点原点重定基、超大尺度 LOD |
| **No Man's Sky / Minecraft / Deep Rock / Teardown** | 程序化生成、体素/SDF 可破坏地形、Transvoxel/Dual Contouring 表面重建、运行时挖掘 |
| **Geometry Clipmaps（Losasso-Hoppe）/ CDLOD（Strugar）/ Chunked LOD** | 嵌套网格远景、连续细节 LOD 无裂缝 morph、四叉树分块误差度量 |
| **GPU 侵蚀（Mei / Št'ava / Jákó-Tóth）** | 水力/热力侵蚀 PDE、泥沙输运、沟壑/冲积、离线烘焙 + 运行时局部 |
| **Ghost of Tsushima / 塞尔达 / 原神（NPR 侧）** | 风格化地表 ramp、风场草海、艺术化等高、卡通远山、程序化落叶/花海 |

**判据（承接材质 §6.2）**：一等子系统必须拥有**自己的几何 + 自己的 sim + 特殊渲染/服务**。世界与地形三者全占：
- **几何**：高度场 clipmap/CDLOD 连续网格、虚拟高度场网格（VHM）、Nanite 式虚拟几何地形块、体素/SDF 等值面网格、样条扫掠几何（道路/河床/墙）；
- **sim**：水力/热力侵蚀与泥沙输运、程序化生成 PCG 求值、流送驻留调度、植被风场与交互压弯、体素破坏布尔运算；
- **特殊渲染/服务**：RVT 层混合、三平面/立体投影、虚拟纹理 megatexture 流送、远景 HLOD/imposter、地形 occluder（写 HZB）、与 GI/阴影/水/体积的场景表示接入。

因此世界与地形是**一等子系统**。**关键边界**：
- **大气散射/雾/云是共享服务与各自子系统**（见 volumetric/water）；地形**消费**其 LUT 并写回云影/地形遮挡，不重写。
- **散布出来的飞沫/尘土/落叶粒子委托 Ember 粒子子系统**，本系统只产出 placement 实例流与密度场，不自建粒子池。
- **碰撞/刚体/车辆对接 physics_core**，本系统提供 heightfield/体素 collider 几何与法线场，不重写求解器。
- **NPR 不与地形同级**——地形是几何/服务子系统，NPR 是 illumination（风格轴）；地表既可 PBR 也可 NPR，正交，且 NPR 地表同享全部共享高级基底。

---

## 1. 设计原则与约束（Prism 范式）

承接毛发/布料/粒子/水/体积同构范式：**共享基底 + PBR/NPR 分叉响应**。

```
共享基底(PBR/NPR/自定义/混合 都吃):
  世界分区(隐式网格 + 四叉树/八叉树 + 数据层 + HLOD 代理)
  + 地形几何(高度场 CDLOD/clipmap + VHM + Nanite式虚拟地形块 + 体素/SDF 等值面, 连续LOD 禁硬切换)
  + 虚拟纹理(接入渲染架构 §8 虚拟资源系统: 几何页/纹理页/RVT页 共用预算与反馈)
  + 程序化生成 PCG(确定性图求值 → 密度场/点云 → GPU-driven 散布)
  + 流送(接入虚拟资源系统的页式预算/反馈/预取/驱逐框架)
  + 共享高级基底: 虚拟几何 vis-buffer / Lumen式混合GI / ReSTIR DI-GI / VSM / froxel体积 / RVT / 路径追踪参考 / 时序上采样
  + 碰撞/导航(委托 physics_core heightfield/体素 collider + 导航网格烘焙)
  + motion vector(共享时序) + occluder(写共享 HZB) + 大气/水/云影耦合(共享服务)

分叉响应(只在"光照响应"分家, 见 §15):
  PBR:  物理层混合BRDF(金属度/粗糙度) + 三平面世界空间 + POM/位移 + 距离detail + 积雪/湿润/泥浆遮罩 + RVT烘焙
  NPR:  ramp 量化地色 + 卡通等高带 + 手绘岩纹/草笔触 + 风格化远山 + 水墨扩散雪线 + 艺术化AO
  自定义/混合: 同一地形多材质层, 逐生物群系/逐坡度/逐高度混合 PBR↔NPR

fallback:  近景 Nanite虚拟地形/体素重建; 中景 CDLOD高度场; 远景 HLOD代理网格 + imposter + RVT烘焙底图; 极远 行星曲率低模。
```

硬约束：
- **GPU-first 零回读**：地形评估、LOD 选择、散布实例剔除、密度采样尽量在 GPU 完成；CPU 只提交分区增量、流送决策、PCG 图参数。
- **成本与变化成正比**：相机静止、世界未编辑时，几何/纹理/散布缓存写入趋近于零（承接渲染架构成功定义）。
- **禁 pop 硬切换**：所有 LOD 过渡用几何 morph（CDLOD）或虚拟几何连续误差，纹理用 trilinear/anisotropic + mip 过渡，散布用密度淡入淡出，HLOD 用 dither/stochastic 过渡。
- **大世界浮点精度**：世界坐标用双精度/分段原点（camera-relative rendering + origin rebasing），GPU 侧用相机相对单精度，禁止在远离原点处出现抖动。
- **三重门控**：capability（硬件特性）× quality tier（画质档）× feature flag（实验开关），任意高级路径都可降级到 fallback。

---

## 2. 现状盘点（可复用地基）

静态阅读当前仓库 `crates/`，可直接或改造复用：

- `bevy_render`：Render Graph、extract/prepare/queue、GPU buffer、HZB 基础 → 作为地形 Pass 与 GPU-driven 剔除宿主。
- `bevy_pbr/meshlet`：Nanite 式虚拟几何原型 → **Nanite 化地形块**与 HLOD 代理直接复用其 cluster/误差 LOD/GPU 选择路径。
- `bevy_pbr/{decal, parallax, cluster, atmosphere, prepass, ssao, ssr, environment_map, light_probe, lightmap}`：贴花、视差/POM、聚簇光、大气、prepass、AO、反射、探针 → 地形着色与 RVT 写入复用。
- `bevy_mesh`：mesh/vertex/index/morph → 高度场网格与 morph LOD、体素等值面网格生成目标格式。
- `bevy_scene` + `bevy_world_serialization`（`dynamic_world*`、`world_asset*`、`world_filter`）：**World Partition 的协作与序列化地基** → 分区单元、OFPA 式单 Actor 资产、数据层过滤直接扩展。
- `bevy_camera` + `bevy_camera_controller`：相机与视锥 → LOD 误差度量、分区激活半径、origin rebasing 挂点。
- `bevy_asset`：异步加载 → 流送页加载后端。
- `bevy_math` / `bevy_transform`：四叉树/八叉树数学、相机相对变换。
- `bevy_solari`：RT 参考 → 地形 RT 碰撞/焦散/离线对拍可选接入。

缺口（需新增）：世界分区运行时、地形多表示几何管线、PCG 图编译器与 GPU 求值、海量散布 GPU 实例系统、虚拟高度场/虚拟纹理地形特化、侵蚀 sim、体素破坏、地形编辑器。

---

## 3. 总体架构（分层）

```
L0 世界空间层 World Space
    隐式分区网格(2D cell / 3D 行星四叉面) + 四叉树/八叉树 + 数据层 Data Layers
    + 坐标系统(double / camera-relative / origin rebasing) + 分区激活策略
         │
L1 地形表示层 Terrain Representation (多桶, 见 §5)
    高度场(CDLOD/clipmap) · 虚拟高度场网格(VHM) · Nanite式虚拟地形 · 体素/SDF · 样条扫掠
         │
L2 程序化生成层 PCG
    节点图 → IR → GPU 求值 → 密度场/点云/规则 → 生物群系分配
         │
L3 散布与植被层 Scatter/Foliage
    GPU-driven 实例生成/剔除/LOD → 风场/交互 → imposter/billboard → 委托粒子
         │
L4 流送与驻留层 Streaming
    页式预算/反馈/预取/驱逐(接入渲染架构 §8 虚拟资源系统) + HLOD 代理调度
         │
L5 仿真层 Sim
    水力/热力侵蚀 + 泥沙输运(离线烘焙 + 运行时局部) + 体素布尔破坏 + 植被风场
         │
L6 渲染接入层 Render Integration
    vis-buffer 虚拟几何 · RVT 层混合 · VSM 阴影 · Lumen GI · froxel 雾 · occluder HZB · PBR/NPR 分叉
         │
L7 碰撞/导航/物理 (委托 physics_core)
    heightfield/体素 collider + 法线场 + 导航网格烘焙 + 流送同步
```

数据流原则：**同一个世界真相源**（World Partition）被所有层消费；地形几何、散布、碰撞、GI/阴影场景表示都从同一分区/LOD 决策派生，禁止各层重复维护完整世界镜像（承接渲染架构"统一场景真相源"）。

---

## 4. L0 世界空间层：大世界分区（World Partition 对标）

### 4.1 隐式分区网格
- 平面世界：2D 规则网格 `cell = floor(pos.xz / cell_size)`，默认 cell 可配（如 128m/256m），分区**隐式**（不手动切 sublevel），Actor 按包围盒落入 cell。
- 行星世界：球面立方体六面各一棵**四叉树**（quadtree cube-sphere），cell 为四叉树叶节点，支持 CDLOD-on-sphere。
- 垂直世界（洞穴/体素）：八叉树扩展，cell 为 3D 体素块（brick）。

### 4.2 数据层 Data Layers
- 运行时层（白天/夜晚/损毁态/剧情阶段）与编辑层（美术分工）正交。
- 层开关驱动分区内 Actor 的激活/卸载；与 gameplay 任务状态耦合（见 gameplay 文档）。
- 继承 `bevy_world_serialization/world_filter` 做层过滤与条件 spawn。

### 4.3 坐标与浮点精度
- 世界坐标：逻辑上 `f64` 或 `i64`+局部 `f32`（分段原点）。
- 渲染/物理：**camera-relative**，GPU 侧传相机相对 `f32`；相机越过阈值触发 **origin rebasing**（整世界平移，重算相对变换），对用户透明。
- 大坐标纹理/噪声：用 cell-local UV + cell id，避免远原点精度塌陷。

### 4.4 分区激活与 HLOD
- 激活策略：按相机的多层半径（物理半径 / 可见半径 / HLOD 半径）决定 cell 的状态机：`Unloaded → Streaming → Loaded(active) → HLOD-proxy → Unloaded`。
- **HLOD（分层细节代理）**：离线烘焙每级 cell 群组的合并代理网格（可 Nanite 化）+ 合并材质（RVT 底图 + imposter），远景只渲染代理，近景切回真实 Actor，过渡用 dither/stochastic。
- OFPA 式协作：每 Actor 单文件（继承 `dynamic_world`），减少多人编辑冲突；分区元数据与 Actor 数据分离。

### 4.5 关卡实例 Level Instance
- 可复用预制世界块（村庄/营地）作为实例引用，支持嵌套与覆盖；实例内 PCG 结果可缓存复用。

---

## 5. L1 地形表示层（多桶，连续 LOD 禁 pop）

**判据**：没有单一表示能同时满足"行星尺度 + 近景微表面 + 可破坏 + 悬垂洞穴"。按场景分桶，统一 LOD 语义与材质接口。

| 桶 | 表示 | 适用 | LOD 机制 | 碰撞 |
|---|---|---|---|---|
| **A. 高度场 CDLOD/clipmap** | 2.5D heightmap + 嵌套网格 | 开阔地貌、远景、行星面 | CDLOD morph / clipmap 环 | heightfield collider |
| **B. 虚拟高度场网格 VHM** | heightmap → GPU 评估虚拟网格 | 需与贴花/样条/RVT 深度融合的地表 | 虚拟几何连续误差 | 从 RVT 高度采样 |
| **C. Nanite 式虚拟地形** | 预构 cluster + 连续误差 | 极致近景密度、雕刻细节 | 复用 meshlet 误差 LOD | 代理凸包/低模 |
| **D. 体素/SDF** | 稀疏体素八叉树 + 等值面 | 可破坏、悬垂、洞穴、矿道 | 多分辨率 brick + Transvoxel | 体素/SDF collider |
| **E. 样条扫掠** | 样条 + 剖面扫掠 | 道路、河床、堤墙、梯田 | 沿样条分段 LOD | 扫掠 mesh collider |

### 5.1 桶 A：高度场 CDLOD / Geometry Clipmap
- CDLOD（Strugar）：四叉树分块，每块固定网格，顶点在 LOD 边界按相机距离 **morph** 到父块位置，GPU 顶点着色器完成，天然无裂缝。
- Geometry Clipmap（Losasso-Hoppe）：以相机为中心的嵌套方形环，远环更粗，环间用过渡区 blend；适合超大/行星远景。
- 高度采样：虚拟高度场纹理（见 §7），顶点 shader 采样 + 法线由高度差分或法线贴图。

### 5.2 桶 B：虚拟高度场网格 VHM
- heightmap + layer 写入 **RVT**，运行时在 GPU 用 RVT 高度/法线评估生成虚拟网格；贴花、样条、笔刷、PCG 的改动写回 RVT，地表、散布、碰撞统一采样同一真相源。

### 5.3 桶 C：Nanite 式虚拟地形
- 把地形块按 §6（渲染架构虚拟几何）离线构 cluster + DAG 连续误差，复用 meshlet 的 GPU 选择 / 软硬光栅；适合"地形当作普通虚拟几何"的极致密度路径。

### 5.4 桶 D：体素 / SDF（可破坏）
- 稀疏体素八叉树（SVO）/ brick grid 存密度或 SDF；**Transvoxel**（Lengyel）或 **Dual Contouring** 做跨 LOD 无裂缝等值面；运行时布尔（挖掘/爆破）只重算受影响 brick。
- 与粒子/物理耦合：破碎产出碎块委托粒子/刚体；密度场改动触发局部碰撞与导航重烘焙。

### 5.5 桶 E：样条扫掠
- 道路/河流/墙：样条 + 剖面，扫掠生成网格并**压平/混入**底层地形（写 RVT 高度/层权重），避免悬浮/穿插；河流与 water 子系统共享样条真相源。

### 5.6 跨桶接缝
- 桶与桶边界（如体素挖洞接高度场）用统一 SDF/高度真相源对齐 + skirt 裙边消隙；材质在接缝用层权重 blend。

---

## 6. 连续 LOD 与裂缝消除（禁 pop 专章）

- 几何：CDLOD morph（桶 A/B）、虚拟几何连续误差（桶 C）、Transvoxel 跨分辨率过渡单元（桶 D）、样条分段 geomorph（桶 E）。
- 纹理：trilinear + anisotropic + RVT mip 过渡；virtual texture 页按屏幕误差请求（见 §7）。
- 散布：按屏占/距离做密度淡入淡出 + per-instance alpha dither，禁整块突现。
- HLOD：真实 Actor ↔ 代理用时序 stochastic/dither，配合时序上采样 TAA 消闪烁。
- 误差度量：屏幕空间几何误差（像素）为统一货币，跨桶可比；LOD 选择全在 GPU，CPU 只给预算。

---

## 7. 虚拟纹理与地形材质（RVT + Virtual Texture）

### 7.1 运行时虚拟纹理 RVT
- 地形层权重、高度、法线、贴花、样条、笔刷、PCG 结果都写入 **RVT**（分页纹理）；地表着色与散布密度统一采样 RVT。
- 接入渲染架构 §8 **虚拟资源系统**：RVT 页与几何页、纹理页、阴影页**共用**预算 / 反馈 / 优先级 / 驻留 / 驱逐框架，不另造一套。
- 远景材质烘焙：把高成本层混合离线/异步烘焙到 RVT 低 mip，远景直接采样，省掉逐像素层混合。

### 7.2 地表材质层混合
- Layer blend：权重图（RVT）× 每层材质（albedo/normal/rough/height/AO）；高度感知混合（height-blend，按层高度图做锐利过渡，非线性 lerp）。
- 三平面 / 立体投影：陡坡用世界空间三平面避免拉伸；按坡度在 UV 投影与三平面间 blend。
- 距离 detail：近景叠高频 detail normal/albedo，远景 fade，消 tiling。
- POM / 位移：复用 `bevy_pbr/parallax`，近景 POM，极近用虚拟几何真位移（桶 B/C）。
- 遮罩系统：积雪（按法线.y + 高度 + 噪声）、湿润（与 water/weather 耦合，见 §13）、泥浆、沙积、落叶，统一遮罩层叠加。

### 7.3 虚拟纹理 megatexture（可选高配）
- 唯一化 UV 的超大纹理（id-tech 式）流送，适合手绘唯一地貌；与 RVT 共享分页/反馈。

---

## 8. L2 程序化生成层（PCG）

### 8.1 节点图 → IR → GPU 求值
- 美术用节点图（采样噪声/高度/坡度/曲率/生物群系 → 过滤规则 → 散布点/密度场/样条）；编译为 IR，codegen 到 **WESL compute**，GPU 批量求值（承接 Ember 粒子的图→IR→WESL codegen 范式）。
- 确定性：固定 seed + cell id + 图版本号 ⇒ 可复现、可缓存、可增量；编辑单元只重算受影响 cell。

### 8.2 密度场与规则
- 密度/概率场由高度、坡度、曲率、河网距离、生物群系、遮罩、手绘笔刷叠加；规则做 min/max 坡度、避让（道路/水体/建筑 footprint）、聚集/排斥。
- 生物群系 biome：按气候参数（高度/湿度/温度/纬度）分配，驱动地表层权重 + 散布资产集 + 颜色 ramp。

### 8.3 运行时 vs 烘焙
- 烘焙：离线把 PCG 结果固化为 cell 资产（点云/实例流/RVT 层），运行时直接流送，省求值。
- 运行时局部：编辑/破坏/数据层切换时只对脏 cell 重算，结果增量写回；相机静止时零求值。

### 8.4 散布点生成
- 泊松盘 / 蓝噪声分布（继承 `bevy_pbr/bluenoise`）保证均匀无规律；分层散布（大树→灌木→草→碎石）带避让。

---

## 9. L3 散布与植被层（海量实例）

### 9.1 GPU-driven 实例
- PCG 产出 per-cell 实例流（transform + 资产 id + 变体 + biome 色），GPU **生成 → 视锥/HZB 遮挡剔除 → LOD 选择 → indirect draw**，CPU 只提交脏 cell 增量（承接渲染架构 GPU-driven 可见性，百万实例移动千个 ≈ O(千)）。
- 实例进 GPU Scene，复用虚拟几何/VSM/GI/RVT，不另造实例渲染路径。

### 9.2 植被 LOD 与 imposter
- 近景真网格（可 Nanite）→ 中景减面 LOD → 远景 **octahedral imposter / billboard**（预渲染多视角图集）→ 极远并入 HLOD 代理 / RVT 底图。
- 过渡全用 dither + 时序，禁 pop。

### 9.3 风场与交互
- 全局风场（方向/强度/阵风噪声，与 weather 耦合）→ 顶点动画（层级摆动：主干/枝/叶，Ghost of Tsushima 式）；
- 交互压弯：角色/载具/爆炸写**交互位移场**（splat 到纹理），植被顶点读场做压弯回弹，草海踩踏（塞尔达式）。
- 落叶/花粉/尘：委托 Ember 粒子子系统（本层只给发射密度与风场），不自建粒子池。

### 9.4 草海特化
- 极密草：GPU 程序化草叶（曲面片/点生成），tile 实例 + 距离密度衰减 + per-blade 风相位；远景转地表草层纹理。

---

## 10. L4 流送与驻留层

- 全部接入渲染架构 §8 **虚拟资源系统**的统一框架：预算（显存/带宽/CPU）、反馈（屏幕误差/可见性请求）、优先级、异步加载、驱逐。
- 流送粒度：cell（Actor/HLOD）、几何页（虚拟地形）、纹理页（RVT/VT）、体素 brick、散布实例块。
- 预取：按相机速度/朝向预测，预加载前方 cell；高速移动（载具/飞行）用更大半径 + 更激进 LOD。
- 预算保护：超预算时平滑降级（降 LOD、推迟散布、用 HLOD 代理替真 Actor、停侵蚀 sim），不崩溃、不无限分配（承接渲染架构稳定性）。
- 零回读：加载后端异步 `bevy_asset`，GPU 侧 upload，CPU 不等 GPU。

---

## 11. L5 仿真层：侵蚀与地貌

### 11.1 水力 / 热力侵蚀
- 水力侵蚀（Mei / 虚拟管道 pipe model 或粒子法 Št'ava）：降雨 → 水流 → 溶蚀/沉积 → 泥沙输运 → 沟壑/冲积扇；GPU compute 在高度场上迭代。
- 热力侵蚀：超过休止角的坡度坍塌堆积，软化尖脊。
- 用途：离线烘焙把程序化地形变自然（沟壑/河网/冲积），结果写回高度/层权重/RVT。

### 11.2 运行时局部地貌
- 可选运行时局部侵蚀（暴雨场景、动态河流改道），只在相机近邻 cell 低频迭代，预算受限。

### 11.3 体素破坏 sim
- 挖掘/爆破：对 SDF/体素做布尔，受影响 brick 重算等值面 + 碰撞 + 导航；碎块给物理/粒子；支持持久化（存改动 delta 到 cell）。

### 11.4 植被/生态演化（可选）
- 低频 biome 演化（火烧迹地、季节、践踏恢复），改散布密度场；纯规则/PDE，非 AI。

---

## 12. L6 渲染接入层（复用，不重写）

- **虚拟几何**：地形桶 C/散布近景走 vis-buffer（复用 meshlet），地形写 occluder 进 HZB 供全场景剔除。
- **虚拟阴影 VSM**：地形/植被作为投射者与接收者接 VSM；大地形用远 cascade + VSM 页缓存，静止零重渲。
- **Lumen 式 GI**：地形进 GI 的 L0 场景表示（见 GI 文档 §4），地表作为大面积 bounce 源；远景用 RVT 底图喂辐照缓存。
- **froxel 体积雾**：地形写地形遮挡给体积，谷雾/地表雾与 volumetric 子系统共享 froxel。
- **大气/aerial perspective**：远山地形消费大气 LUT 做空气透视，统一天色。
- **RVT 作为中间真相源**：地表层混合烘焙进 RVT，着色、散布、GI、碰撞统一采样。
- **motion vector**：地形 morph/体素更新/植被摆动写 MV 给 TAA/时序上采样，消 LOD 过渡闪烁。

---

## 13. 跨子系统耦合（水 / 天气 / 时间）

- **水体耦合**：河流/湖海样条与 water 子系统共享真相源；岸线 waterline 掩膜、浅水混合、湿润岸线由 water §9e 驱动地表遮罩；地形高度/法线供水体深度与浅水着色。
- **天气耦合**：降雨 → 湿润遮罩 + 积水（低洼由曲率检测）+ 泥浆；降雪 → 积雪遮罩按法线/高度/风积累 + 融雪；风场 → 草木摆动 + 沙尘。
- **时间/季节**：昼夜驱动地表色温/积雪消融/露水；季节改 biome 色 ramp 与散布密度（落叶/花期）。
- **云影**：volumetric 云投影云影到地形（共享服务），地形写回地形遮挡给大气/体积。

---

## 14. L7 碰撞 / 导航 / 物理（委托 physics_core）

- heightfield collider（桶 A/B）：直接从高度真相源构建，流送同步，LOD 与渲染解耦（物理用稳定中 LOD 防抖）。
- 体素/SDF collider（桶 D）：从 brick SDF 生成，破坏后局部重建。
- 样条/网格 collider（桶 E）：道路/墙扫掠凸分解。
- 导航网格：按 cell 烘焙，流送时拼接；破坏/编辑触发脏 cell 局部重烘焙（见 gameplay 导航）。
- 载具/角色对接 physics_core，不重写求解器；地形提供法线/材质（摩擦/声音/粒子）查询。

---

## 15. PBR / NPR / 自定义 / 混合 分叉响应（只在光照响应分家）

共享到"几何 + RVT 层数据 + 遮罩 + 光照输入"为止，仅在**着色响应**分叉：

| 轴 | PBR | NPR |
|---|---|---|
| 地表色 | 物理 albedo + 层混合 BRDF | ramp 量化地色 / biome 色块 |
| 高程表达 | 真实法线 + POM + 位移 | 卡通等高带 / 手绘岩层线 |
| 光照 | 金属度-粗糙度 + GI + VSM | ramp 受光 + 卡通阴影块 |
| 远山 | 空气透视物理 | 风格化平涂远山 / 水墨渐隐 |
| 雪/湿 | 物理遮罩 BRDF 切换 | 水墨扩散雪线 / 手绘湿痕 |
| 草木 | 透射 SSS + 物理高光 | 卡通层描边 + 块状高光 |

- 自定义/混合：同一地形多材质层，逐 biome / 逐坡度 / 逐高度在 PBR↔NPR 间混合；NPR 地表**同享**虚拟几何/GI/VSM/RVT 全部共享高级基底（NPR 是风格轴，不降级基底）。

---

## 16. 跨平台分档矩阵（Metal 优先，design target）

| 档 | 平台参考 | 地形桶 | 虚拟纹理 | 散布 | 侵蚀 | GI/阴影 |
|---|---|---|---|---|---|---|
| **高端桌面** | 独显 | C(Nanite)+D(体素)+B | RVT + VT megatexture | 百万级 GPU-driven + Nanite 植被 | 烘焙 + 运行时局部 | Lumen + VSM + RT 增强 |
| **桌面兼容** | 中端 | B(VHM)+A(CDLOD) | RVT | 十万级 + imposter | 烘焙 | Lumen 软件 + VSM |
| **移动** | iOS/Android | A(CDLOD) | RVT 低页 / 预烘焙底图 | 万级 + billboard | 全烘焙 | 预计算 GI + 常规阴影 |
| **Web** | WebGPU | A(CDLOD) | 预烘焙底图 | 千级 | 全烘焙 | 轻量 GI |

- 同语义不同路径：高低档地形**语义一致**（同高度/层数据），只是 LOD 深度、虚拟化程度、散布密度、sim 质量不同；禁按 OS 硬编码，按 capability + 性能基准选档。

---

## 17. 性能预算与效果验收（design target，非实测）

效果验收（画质门槛）：
- 近景地表像素级 detail（POM/位移），无可见 tiling；陡坡无纹理拉伸。
- LOD 过渡全程无肉眼可见 pop（几何/纹理/散布/HLOD）。
- 行星/大世界远景无浮点抖动，远山空气透视与天色一致。
- 海量植被风场自然，交互压弯/踩踏即时响应。
- 可破坏地形挖掘/爆破接缝无裂缝，碰撞/导航即时跟随。

性能预算（高端桌面参考，design target）：
- 相机静止 + 世界未编辑：地形几何/RVT/散布缓存写入 ≈ 0。
- 百万散布实例移动千个：CPU 提取/上传 ≈ O(千)。
- 视锥外/遮挡/低贡献地形与散布不产生着色成本（GPU 剔除）。
- 超预算平滑降级（降 LOD/HLOD 替真 Actor/停 sim），帧时不尖刺、不 OOM。
- origin rebasing 单帧完成，无可见跳变。

---

## 18. 可测性（CPU 可验证，沙盒内）

无 GPU 下仍可单测的纯函数：
- 分区索引：`pos → cell`、cell 邻接、激活半径集合、四叉树/八叉树切分与叶节点枚举。
- LOD：CDLOD morph 系数、屏幕空间误差度量、clipmap 环归属、Transvoxel 过渡单元查表。
- 噪声/PCG：固定 seed 的噪声/domain warp/泊松盘确定性复现、图 IR 求值的参考实现。
- 侵蚀核：单步水力/热力迭代的质量守恒（泥沙输入=输出+沉积）、休止角约束。
- 坐标：origin rebasing 前后相对坐标一致性、double↔camera-relative 往返误差界。
- 流送：预算分配、优先级排序、驱逐策略的确定性决策。

---

## 19. Crate 拆分与落地形态

```
bevy_world            世界分区运行时 / 数据层 / 坐标(double+rebasing) / HLOD 调度 / Level Instance
                      (扩展 bevy_scene + bevy_world_serialization)
bevy_terrain          地形多表示(高度场/VHM/Nanite地形/体素/样条) / LOD / 碰撞几何导出
bevy_terrain_render   extract/prepare/queue / RVT / 层混合 / 三平面 / 渲染图节点 / WESL / occluder
bevy_pcg              PCG 节点图 / IR / WESL codegen / GPU 求值 / 散布点生成 / biome
bevy_pcg_scatter      GPU-driven 海量实例 / 剔除 / LOD / imposter / 风场 / 交互场
bevy_terrain_sim      水力/热力侵蚀 / 泥沙输运 / 体素布尔破坏 / 生态演化
bevy_terrain_editor   (可选) 高度/层/样条笔刷 / PCG 图编辑 / 烘焙 / 预览 HUD
```
依赖方向：`editor → {world, terrain, pcg, pcg_scatter, terrain_sim} → terrain_render → bevy_render/bevy_pbr(meshlet)/bevy_light/bevy_solari/...`。接入虚拟资源系统（渲染架构 §8）、GPU Scene（§5）、虚拟几何（§6）、VSM（§9）、GI（§10）。可作为 optional feature 接入 `bevy_internal`。新代码落在 `pkg/` 下对应 crate。

---

## 20. 分阶段实施路线（按收益/风险排序）

- **M0 分区地基**：隐式网格 cell + 数据层 + camera-relative + origin rebasing；扩展 `world_serialization`。可合并切片：静态 cell 加载/卸载 + Actor 过滤。
- **M1 高度场地形（桶 A）**：CDLOD morph + 高度/法线采样 + 基础层混合，禁 pop。
- **M2 RVT 接入**：层权重/高度写 RVT，接虚拟资源系统；远景烘焙底图。
- **M3 GPU-driven 散布**：PCG 点云 → GPU 实例 → 剔除/LOD/imposter；接 GPU Scene。
- **M4 流送与 HLOD**：页式预算/预取/驱逐 + HLOD 代理烘焙与过渡。
- **M5 VHM + Nanite 地形（桶 B/C）**：虚拟高度场网格 + Nanite 化地形块（复用 meshlet）。
- **M6 PCG 图编译器**：节点图 → IR → WESL GPU 求值 + 确定性烘焙 + 增量重算。
- **M7 风场/交互植被 + 草海**：层级摆动 + 交互位移场 + GPU 草。
- **M8 体素/SDF 可破坏（桶 D）**：SVO + Transvoxel + 布尔破坏 + 局部碰撞/导航。
- **M9 侵蚀 sim**：GPU 水力/热力离线烘焙 + 运行时局部。
- **M10 渲染接入深化**：VSM/Lumen/froxel/大气/水/天气全耦合 + NPR 分叉。
- **M11 行星尺度（可选）**：cube-sphere 四叉树 + CDLOD-on-sphere + 行星级 LOD。

每阶段独立收益、独立回退路径。

---

## 21. 风险与降级矩阵

| 风险 | 降级 / 对策 |
|---|---|
| 虚拟地形/体素实现复杂 | 先桶 A(CDLOD) 保底，B/C/D 作为逐步增强，capability 门控 |
| RVT 页抖动/带宽超预算 | 收紧屏幕误差阈值 + 远景烘焙底图 + 预算优先级降级 |
| 海量散布 GPU 压力 | 密度衰减 + imposter 前移 + HLOD 替代 + 预算截断 |
| 侵蚀 sim 不收敛/爆量 | 固定迭代上限 + 质量守恒断言 + 仅离线烘焙(关运行时) |
| 大世界浮点抖动 | origin rebasing + cell-local UV + double 逻辑坐标，CPU 单测守护 |
| 体素破坏持久化膨胀 | 存 delta + 压缩 + 远 cell 固化回基线 |
| 跨桶接缝裂缝 | 统一真相源对齐 + skirt 裙边 + 层权重 blend，回归测试 |
| 平台能力不足 | 三重门控自动降档，语义一致、质量可解释 |

---

## 22. 验收门槛（毕业标准）

- 高端档正式发布时达到 UE5.x 同级地貌画质（近景 detail、远景无抖、LOD 无 pop、海量植被、空气透视）。
- 全桶 LOD 过渡无可见 pop，跨桶接缝无裂缝。
- 相机静止零缓存写入；百万实例移动千个 ≈ O(千)。
- 超预算平滑降级，无崩溃/OOM/帧尖刺。
- CPU 纯函数（分区/LOD/噪声/侵蚀核/坐标）全部单测通过，确定性可复现。
- PBR 与 NPR 地表同享全部共享高级基底，互不降级。

---

## 23. 一句话总结

**一个世界真相源（World Partition）驱动多表示地形、程序化生成、海量散布、流送、侵蚀与碰撞，全部 GPU-first、连续 LOD 禁 pop，并复用 Prism 既有的虚拟几何 / 虚拟资源 / RVT / VSM / Lumen GI / froxel 共享基底，PBR 与 NPR 只在光照响应分叉——这就是 Prism 的次世代 AAA 级世界与地形系统。**
