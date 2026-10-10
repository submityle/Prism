# Prism × Bevy 全量重构方案：用 pkg AAA 实现替换 crates 全部子系统

> 状态：重构方案（Draft，仅文档，不含代码改动）
> 面向版本：Prism（Bevy `0.20-dev` fork）
> 目标读者：渲染、物理、音频、UI、平台与 LLM Agent 工程人员
> 本文依据：对当前仓库 `crates/`（62 个）、`pkg/`（47 个）的静态阅读与已落地的 slice 1/2 验证
> 最后更新：2026-10-02

## 1. 摘要

仓库同时存在两套实现：

- `crates/`：62 个原版 Bevy crate（`bevy_*`），上游 Bevy 的 fork，合计约 50 万行。
- `pkg/`：47 个 Bevy 无关的 `prism_*` crate，面向次时代顶级 AAA，基于 `glam`，不依赖 Bevy 渲染栈。

**重构目标：把 `crates/` 下全部 62 个 crate 逐一处置，凡 `pkg/` 已覆盖的能力一律以 pkg 为唯一真相源，crate 侧重复实现退役；pkg 未覆盖的 crate 保留为平台/机制/资产外壳，并标注后续 pkg 化路径。** 不在 `crates/` 内再写第二套 AAA。

四条硬约束：

1. **pkg 优先**：有 pkg 就用 pkg，弃用 crate 侧重复实现，不二次重写。
2. **机制稳定、桥接下沉、禁止依赖环**：`prism_*` 的 ABI 永不依赖 Bevy；Bevy 类型只在桥接层单向下降为 Prism 记录（见 §6）。
3. **LLM Agent 易维护**：单一桥接缝、冻结 ABI、切片式推进、每片可独立编译/测试/回退。
4. **性能与效果**：GPU 驱动、增量更新、成本正比于变更量；高端档对标 UE 5.x 画质门槛。

## 2. 现状

- `crates/*` 无一依赖 `pkg/prism_*`（已核实 `rg prism_ crates/*/Cargo.toml` 为空）。
- `pkg/prism_*` 刻意不依赖 Bevy 渲染栈。
- 已接 `bevy` 桥接的 pkg 仅 2 个：`prism_render_material`、`prism_render_visibility`（均含 `src/bevy_bridge.rs`）。
- 下游集成 crate `pkg/prism_bevy` 已建立，承载可见性插件（slice 2）。

## 3. 处置分类定义

| 代号 | 含义 | 对 crate 的最终形态 |
| --- | --- | --- |
| **R** | 弃用重构 → pkg | 实现退役；仅保留 ECS 组件/事件/资产外壳供桥接；逻辑由 pkg + 下游集成 crate 承载 |
| **R-部分** | 部分弃用 | 机制/数据部分迁 pkg，平台/管线外壳保留 |
| **M** | 保留·机制底座 | 永久保留；作为 ECS/平台机制与桥接宿主 |
| **P** | 保留·平台 I/O | 永久保留；原始设备/窗口事件源 |
| **A** | 保留·资产/管线外壳 | 保留；内部计算逐步 pkg 化，资产生命周期留在 crate |
| **Q** | 保留·质量/效果外壳 | pkg 暂无覆盖，保留并标注 pkg 化候选 |

## 4. 全部 62 个 crate 处置总表

> “目标 pkg”为空表示保留类（M/P/A/Q）。表按处置类型分组。

### 4.1 R / R-部分：弃用重构 → pkg（17 个）

| crate | 行数 | 处置 | 目标 pkg（真相源） | 重构要点 |
| --- | ---: | --- | --- | --- |
| bevy_camera | 5152 | R-部分 | prism_render_visibility | 视锥/可见性→pkg；相机投影/变换留 crate |
| bevy_material | 1289 | R | prism_render_material | 材质定义/参数下降为 `MaterialRecord` |
| bevy_pbr | 40936 | R | prism_render_shading + prism_render_material + prism_render_architecture | 着色模型/材质/GPU 场景全迁 pkg，最大切片，须再拆 |
| bevy_render | 35303 | R-部分 | prism_render_architecture + prism_render_scene | GPU Scene/可见性机制迁 pkg；设备/RHI 外壳保留 |
| bevy_extract | 1878 | R | prism_render_scene | 提取逻辑换成 pkg 增量事务 |
| bevy_light | 5469 | R | prism_render_shading | 光源记录/光照计算迁 pkg |
| bevy_core_pipeline | 7851 | R-部分 | prism_render_architecture | Pass 调度经 pkg Render Graph；管线外壳保留 |
| bevy_audio | 1893 | R | prism_audio_core + prism_audio_device | DSP 图/设备 I/O 迁 pkg |
| bevy_ui | 16179 | R | prism_ui_component/layout/style/text/theme | UI 核心迁 pkg，须拆多子切片 |
| bevy_ui_render | 7207 | R | prism_ui_render_backend | UI 渲染后端迁 pkg |
| bevy_ui_widgets | 6709 | R | prism_ui_component | 组件库迁 pkg |
| bevy_feathers | 11859 | R | prism_ui_theme + prism_ui_component | 主题/控件迁 pkg |
| bevy_text | 6278 | R | prism_ui_text | 文本排版/成形迁 pkg |
| bevy_a11y | 283 | R | prism_ui_a11y | 可访问性迁 pkg |
| bevy_input_focus | 3320 | R | prism_ui_input | 焦点导航迁 pkg |
| bevy_picking | 5137 | R-部分 | prism_ui_input | 命中/拾取迁 pkg；射线来源留平台 |
| bevy_dev_tools | 5096 | R-部分 | prism_ui_devtools + prism_ui_inspector | 检视/快照/热重载迁 pkg |

### 4.2 M：保留·机制底座（19 个）

| crate | 行数 | 说明 |
| --- | ---: | --- |
| bevy_ecs | 119849 | ECS 内核，整个引擎机制地基，永久保留 |
| bevy_app | 6328 | 应用/插件/调度，桥接与集成装配宿主 |
| bevy_math | 7196 | glam 基座，crate 与 pkg 共用，保留 |
| bevy_transform | 2744 | 变换组件/传播，桥接输入源 |
| bevy_reflect | 36267 | 反射，序列化/编辑器依赖，保留 |
| bevy_time | 3456 | 时间/计时 |
| bevy_tasks | 3492 | 任务/并行执行器 |
| bevy_platform | 5205 | 平台抽象 |
| bevy_state | 4493 | 应用状态机 |
| bevy_curve | 8559 | 曲线/插值工具 |
| bevy_settings | 1395 | 配置 |
| bevy_world_serialization | 4501 | World serde |
| bevy_diagnostic | 1277 | 诊断/帧统计 |
| bevy_log | 599 | 日志 |
| bevy_derive | 413 | 派生宏 |
| bevy_macro_utils | 864 | 宏工具 |
| bevy_encase_derive | 38 | GPU 缓冲派生 |
| bevy_ptr | 1624 | 指针工具 |
| bevy_utils | 1214 | 通用工具 |

> `bevy_internal`(427)、`bevy_dylib`(62)、`bevy_android`(10) 为伞 crate/链接/平台外壳，归 M，随插件图调整自动跟随。

### 4.3 P：保留·平台 I/O（5 个）

| crate | 行数 | 说明 |
| --- | ---: | --- |
| bevy_winit | 4928 | 窗口/事件循环后端 |
| bevy_window | 3172 | 窗口抽象 |
| bevy_input | 6909 | 原始输入事件源（焦点/拾取逻辑迁 pkg，事件源保留） |
| bevy_gilrs | 503 | 手柄输入 |
| bevy_clipboard | 427 | 剪贴板 |

### 4.4 A：保留·资产/管线外壳（6 个）

| crate | 行数 | 说明 | pkg 化方向 |
| --- | ---: | --- | --- |
| bevy_asset | 24301 | 资产系统/生命周期，保留 | 计算部分可供 pkg 消费 |
| bevy_mesh | 11927 | 网格资产 | 喂给 prism_virtual_geometry_gpu / architecture |
| bevy_image | 8106 | 图像/纹理资产 | 喂给 pkg 材质/纹理驻留 |
| bevy_shader | 1184 | Shader 资产 | 喂给 pkg Render Graph |
| bevy_gltf | 5609 | glTF 加载器 | 导入后下降为 pkg 场景 |
| bevy_scene | 6448 | ECS 场景 serde | 保留，与 prism_render_scene 概念不同 |

### 4.5 Q：保留·质量/效果外壳（12 个）

| crate | 行数 | 说明 | pkg 化候选 |
| --- | ---: | --- | --- |
| bevy_color | 9644 | 色彩科学 | 暂无 pkg |
| bevy_shape | 11587 | 形状/网格基元（桥接测试在用） | 暂无 pkg |
| bevy_animation | 5482 | 骨骼/属性动画 | 待 pkg 动画 crate（设计文档已有） |
| bevy_anti_alias | 2903 | 抗锯齿 | 待并入 pkg 后处理 |
| bevy_post_process | 4252 | 后处理 | 待并入 pkg Render Graph |
| bevy_solari | 4977 | 实时光追 GI | 待并入 prism_render_architecture RT 路径 |
| bevy_sprite | 2051 | 2D 精灵 | 暂无 pkg（2D 域） |
| bevy_sprite_render | 7223 | 2D 渲染 | 暂无 pkg |
| bevy_gizmos | 8730 | 调试绘制 | 暂无 pkg |
| bevy_gizmos_render | 2031 | 调试绘制渲染 | 暂无 pkg |
| bevy_camera_controller | 3619 | 相机控制器（gameplay helper） | 暂无 pkg |
| bevy_remote | 5547 | 远程协议(BRP) | 可与 prism_ui_devtools 协同 |

### 4.6 汇总

- R / R-部分（弃用重构）：**17**
- M（机制底座）：**19**（含 internal/dylib/android）
- P（平台 I/O）：**5**
- A（资产/管线外壳）：**6**
- Q（质量/效果外壳）：**12**
- 合计：**59 行 + internal/dylib/android 3 = 62**，全部 crate 均已指派处置。

## 5. pkg 新增能力（无 crate 对应，经下游集成 crate 附加）

Bevy 原本缺失、由 pkg 带来的全新 AAA 能力，不涉及“重构某 crate”，而是经下游集成 crate 装配：

- **物理**：`prism_physics_core` / `prism_physics_geometry` / `prism_physics_gpu`（Bevy 无内建物理）。
- **虚拟几何**：`prism_virtual_geometry_gpu`（类 Nanite）。
- **体积 / 头发**：`prism_volumetric_gpu` / `prism_hair_gpu`。
- **空间音频**：`prism_audio_hrtf` / `prism_audio_spatial` / `prism_audio_rt`。
- **UI 增强**：`prism_ui_reactive`/`store`/`timetravel`/`router`/`scheduler`/`async`/`i18n`/`form`/`virtual`/`sdui`/`workbench`/`motion`/`anim`/`overlay`/`snapshot`/`hotreload`/`scoped`/`tree` 等（超出 bevy_ui 的能力）。

## 6. 集成架构：单一桥接缝，禁止依赖环

关键约束：**不能让 `crates/bevy_*` 直接依赖 `pkg/prism_*`**。因为 pkg 带可选 `bevy` feature（桥接层会反向依赖 `bevy_*`），若 crate 再依赖 pkg 就形成环。正确形态：

```
   crates/bevy_*  (ECS 组件/事件/资产/平台外壳；实现逻辑逐步掏空)
          │  Bevy 组件/资源 (Aabb, GlobalTransform, Frustum, StandardMaterial…)
          ▼
   pkg/<prism_x>/src/bevy_bridge.rs   ← 唯一“下降”层 (feature = "bevy")，单向 Bevy→Prism
          │  Prism 冻结 ABI 记录 (InstanceRecord, GpuViewRecord, MaterialRecord…)
          ▼
   pkg/prism_*  (引擎中立 AAA 实现 / 真相源)
          ▲
          │  Plugin / System / Resource 装配
   下游集成 crate：prism_bevy(渲染) · prism_bevy_ui · prism_bevy_audio · prism_bevy_physics
          ▲
          │  默认插件图接入 / 摘除
   crates/bevy_internal (伞 crate)
```

因此“重构一个 R 类 crate”= 三步：
1. 真相源下沉：能力实现搬到对应 pkg（多数已存在）。
2. crate 退化为外壳：仅保留 ECS 组件/事件/资产定义；原实现从默认插件图摘除。
3. 下游装配：集成 crate 每帧把组件下降为 pkg 记录、调用 pkg、回写结果组件。

## 7. 分阶段切片执行队列

> 每片 = 可独立 `cargo build/test -p <crate>` + 可回退的最小集成。

| 切片 | 范围 | R 类覆盖 | 状态 |
| --- | --- | --- | --- |
| S1 | `prism_render_visibility` 接 `bevy` 桥接 | — | ✅ 完成（98 测试） |
| S2 | `prism_bevy` 可见性插件：实体镜像 + `cull_view` + 回写 | bevy_camera 可见性 | ✅ 完成（3 集成测试，零警告） |
| S3 | 材质自动注册 `StandardMaterial`→`MaterialRecord` | bevy_material | 计划 |
| S4 | 场景提取统一 → `prism_render_scene` | bevy_extract, bevy_render(部分) | 计划 |
| S5 | 着色/光照 → `prism_render_shading` | bevy_pbr, bevy_light | 计划（最大，须再拆） |
| S6 | 虚拟几何 `prism_virtual_geometry_gpu`（需 GPU） | — | 计划 |
| S7 | 体积 / 头发（需 GPU） | — | 计划 |
| S8 | Render Graph/Pass 统一 → architecture | bevy_core_pipeline(部分), bevy_render(部分) | 计划 |
| S9 | 音频 → `prism_audio_*` | bevy_audio | 计划 |
| S10 | 物理插件（纯新增） | — | 计划 |
| S11 | UI 核心 → `prism_ui_*`（文本/布局/样式/主题） | bevy_text, bevy_ui(部分) | 计划 |
| S12 | UI 控件/后端 → `prism_ui_*` | bevy_ui_widgets, bevy_ui_render, bevy_feathers | 计划 |
| S13 | UI 输入/焦点/拾取/可访问性 | bevy_input_focus, bevy_picking, bevy_a11y | 计划 |
| S14 | 开发工具/检视 → `prism_ui_devtools` | bevy_dev_tools | 计划 |

> 优先级：**渲染(S3–S8) → 音频/物理(S9–S10) → UI(S11–S14)**。渲染侧 pkg 最成熟、集成缺口最清晰；UI 面最大、风险最高，置后分子切片。

## 8. 每个 R 类 crate 的重构手法（统一模板）

对每个 R 类 crate，Agent 执行同一套可复制步骤：

1. **接桥**：为目标 pkg 增加 `bevy` feature + `src/bevy_bridge.rs`（无分配、无副作用，单向下降）。参照 `prism_render_material` / `prism_render_visibility`。
2. **装配**：在对应下游集成 crate 内加 `Plugin` + `SystemSet` + 资源 + 同步/执行系统。
3. **掏空**：把 crate 原实现从默认插件图摘除（优先摘除而非物理删除，降低上游 rebase 冲突），仅保留组件/事件/资产定义。
4. **测试**：补 CPU 集成测试，固化行为（命中/剔除/翻转/销毁等）作为完成判据。
5. **验证**：`cargo build/test -p <crate>`、`--features bevy`、`cargo clippy -p <crate> --tests` 零警告。

## 9. LLM Agent 可维护性设计

1. **单一桥接缝**：所有 Bevy↔Prism 转换只在各 pkg 的 `bevy_bridge.rs`，Agent 改集成只读写一处。
2. **冻结 ABI**：`InstanceRecord`/`GpuViewRecord`/`MaterialRecord`/`SceneBounds` 等为契约，变更需显式版本化。
3. **切片纪律**：一次一片，可独立编译/测试/回退，失败只回退单片。
4. **处置表驱动**：§4 的 62 行总表是机器可读决策来源——“有 pkg 就弃 crate”，Agent 按表推进不自行发明实现。
5. **测试即规格**：行为以集成测试固化，红绿作为完成判据。
6. **零警告门槛**：新集成代码 `clippy --tests` 零警告（含 `missing_docs`、`std_instead_of_alloc`）。
7. **已知 ABI 不变量集中记录**：如句柄 index 0 保留为 null、`Create` 要求 generation 严格递增（slice 2 踩过），写入分配器约定供复用。

## 10. 性能与效果目标

- **成本随变更量**：每帧只提交增量事务（Create/SetTransform/SetBounds/Destroy），正比于变化实体数而非场景总量。
- **GPU 驱动**：可见性、LOD、遮挡、虚拟几何选择尽量在 GPU 完成，CPU 只提交策略参数与增量。
- **画质门槛**：高端桌面档正式发布须达 UE 5.x 同质量等级（见 `prism_rendering_architecture_zh.md`）；中间版本须标注缺失项。
- **可伸缩**：capability / quality tier / feature flag 三重门控选择渲染档位。

## 11. 验证、回退与风险

- **逐片验证**：`cargo build -p <crate>` → `cargo test -p <crate>`；桥接改动追加 `--features bevy`。
- **回归基线**：`prism_render_visibility --features bevy` 维持 98 测试；`prism_bevy` 集成测试全绿。
- **回退路径**：任一 R 类切换后回归，可从默认插件图摘除 pkg 插件、恢复对应 crate 实现，互不影响其他切片。
- **GPU 切片**：虚拟几何/体积/头发/光追需真实设备，CPU 测试工具无法断言，单列设备内验证阶段。
- **主要风险**：
  - **bevy_pbr(41k)/bevy_render(35k)/bevy_ui(16k) 体量巨大**，须再拆多个子切片，严禁一次性大改。
  - **资产与管线耦合**：`bevy_asset`/`bevy_render` 管线短期保留，pkg 化须谨慎处理资产生命周期与 GPU 资源驻留。
  - **依赖环**：严守 §6，禁止 `bevy_*` 直接依赖 `pkg/*`。
  - **上游同步**：crate 为 Bevy fork，弃用优先“摘除于插件图”而非物理删除。

## 12. 当前进度

- ✅ S1：`prism_render_visibility` 接桥，98 测试通过。
- ✅ S2：`prism_bevy` 可见性插件，3 集成测试 + doctest 全绿，零 clippy 警告；修复句柄分配不变量 bug（index 0 保留、generation 须递增）。
- ⏭️ 下一步（待批准后动代码）：S3 材质自动注册（`StandardMaterial`→`MaterialRecord`）。

> 本文仅为文档；代码改动按 §7 切片队列在获批后逐片推进。
