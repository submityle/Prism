# Prism 渲染生产路径边界审计与删除清单

> **状态校正（2026-10-23）**：本文主体是 2026-10-21 的历史快照，以下依赖计数与部分实现描述需按本注更新。`pkg/prism_render_scene/Cargo.toml` 已将旧 Bevy 依赖改为 optional，并集中在显式 `legacy-bevy` feature（`default = []`）；同时移除了不存在的 `prism_render_material/bevy` feature 请求。当前工作树 `cargo check --offline --locked -p prism_render_scene` 通过，证明默认 feature 的依赖闭包可编译。**这不证明旧 Bevy adapter 已迁移，也不证明 `legacy-bevy` 可编译或 GPU/整帧验收通过。** 此外，当前 `DrawDecision` 已只有 `Draw`/`Skip`，未就绪管线走 skip，不再有文中所述 PSO placeholder 选择；Spine 白色 stand-in 仍作为显式 legacy constructor 存在，但 native production plugin 使用 `MeshPipelines::surface_encode`。原始迁移目标和真实 fallback 风险仍须分别按当前源码复核，不能把本注视为全渲染系统完成声明。

审计日期：2026-10-21（Asia/Shanghai）
审计目标：审计渲染生产路径中的 `bevy_*` 依赖、`TODO`、`placeholder`、`stand-in` 与静默 fallback，并给出可执行的删除顺序。
本轮约束：只修改审计报告；不修改核心实现、不删除源码、不调整 manifest。

## 1. 结论

当前不能声称 Bevy 已从 Prism 渲染生产路径移除。

最关键的事实有三项：

1. workspace 默认成员仍是根包 `bevy`，默认 feature 仍启用 `2d`、`3d`、`ui`、`audio`。
2. `pkg/prism_render_scene` 仍直接声明 17 个正常 `bevy_*` 依赖，另有 1 个开发依赖；其 plugin、material、opaque、visibility、shading、GPU RHI 仍由 Bevy RenderApp/RenderWorld 驱动。
3. native `prism_render_app` 已经创建 `MeshPipelines::forward_lit`，但仍存在可执行的 Spine 白色管线实现、PSO placeholder 选择和材料 fallback 语义。这些属于当前 P0，不在本轮修改范围内。

因此，本轮的“删除清单”是迁移顺序和删除门，不是立即删除文件的授权。没有任何源码命中可以在当前证据下安全直接删除而不改变生产行为。

## 2. 可复现基线

本次审计使用当前 checkout，命令如下：

```sh
cargo metadata --manifest-path Cargo.toml --format-version 1 --no-deps --offline
cargo tree -p prism_render_scene -e normal --depth 2 --offline --no-dedupe
tools/check_prism_bevy_boundary.sh --report
tools/check_prism_bevy_boundary.sh --strict
rg -n --glob '*.rs' --glob '*.wgsl' --glob '*.wesl' \
  'bevy_[A-Za-z0-9_]+|TODO|FIXME|placeholder|stand[- ]?in|fallback' \
  pkg/prism_render_app pkg/prism_render_architecture \
  pkg/prism_render_scene pkg/prism_render_shading
```

当前基线输出：

| 检查项 | 结果 | 判定 |
|---|---:|---|
| workspace 成员 | 259 | Bevy 与 Prism 并存 |
| workspace 默认成员 | `bevy` | 默认生产入口仍是旧宿主 |
| workspace 内 `bevy*` 包 | 79 | 迁移尚未完成 |
| `prism_render_scene` 直接 Bevy 正常依赖 | 17 | 真实生产编译依赖 |
| `prism_render_scene` Bevy 开发依赖 | 1 (`bevy_tasks`) | 测试/工具边界仍未清理 |
| `prism_render_app` 直接 Bevy 依赖 | 0 | native app manifest 已脱 Bevy |
| `TODO`/`FIXME`（四个渲染包实现文件） | 未发现 | 没有可直接关闭的 TODO 项 |
| Bevy 边界 `--strict` | 失败 | 失败是预期的迁移未完成证据 |

## 3. Bevy 生产依赖删除清单

### 3.1 P0：必须完成后才能删除

| 删除对象 | 当前证据 | 删除前置条件 | 建议切片 |
|---|---|---|---|
| `pkg/prism_render_scene` 的 `bevy_app` | `src/plugin.rs` 使用 `App`、`Plugin`、`RenderApp`、`ExtractSchedule` | native plugin/sub-app/lifecycle 覆盖所有场景系统，并有集成测试证明启动、抽取、渲染、恢复 | `scene-host-native` |
| `bevy_ecs` | `src/extract`、`src/opaque`、`src/shading`、`src/material` 以 Bevy World/SystemParam/Resource 为生产入口 | scene/material/visibility/shading 的生产数据全部改为 `prism_ecs` snapshot/command/render intent | `scene-data-owner` |
| `bevy_render` | `src/gpu/rhi.rs` 的 `SceneRhi::from_bevy`，以及大量 Buffer/Texture/BindGroup/RenderQueue | 设备、资源、队列、提交完成、generation/recovery 全部由 `prism_render_runtime`/`prism_render_driver` 持有 | `native-rhi-owner` |
| `bevy_core_pipeline`、`bevy_pbr` | opaque phase、Core3d、MeshPipeline、PBR material 仍被使用 | native frame graph、material ABI、opaque/transparent/shadow 真实像素链通过 GPU 验收 | `native-pass-owner` |
| `bevy_asset`、`bevy_image`、`bevy_mesh` | `standard_adapter`、texture heap、mesh consumer 使用 Bevy AssetId/Image/Mesh | `prism_asset`、`prism_image`、`prism_mesh` 完成生命周期、revision、streaming、upload 和 draw binding | `asset-owner-native` |
| `bevy_camera`、`bevy_transform` | view/camera/transform 抽取依赖 Bevy 类型 | native view、jitter、frustum、temporal history 与 transform propagation 有等价验收 | `native-view-transform` |
| `bevy_material`、`bevy_shader` | StandardMaterial、embedded asset、ShaderCache 仍在生产路径 | Prism material IR、shader registry、permutation cache 完整接管，并删除 embedded Bevy API | `material-shader-owner` |
| `bevy_platform`、`bevy_log`、`bevy_derive` | scene crate 仍直接声明并使用 | 替换为 `prism_platform`/`tracing`/native derive；生产 manifest 和源码均无引用 | `native-support-closure` |
| `bevy_tasks` | `prism_render_scene` dev-dependencies | Bevy bridge 测试迁移或删除，native tests 不再需要该 crate | `scene-test-exit` |

### 3.2 迁移后的最终删除顺序

1. 先把 shader asset owner、material/mesh/image ABI 和 GPU 资源 owner 移到 Prism 包；保留 `prism_render_scene` 作为过渡验证包。
2. 将 extract、visibility、opaque、shading、water、cloth、raytrace 的调度从 Bevy RenderApp 迁移到 native render schedule/frame graph。
3. 将 `SceneRhi::from_bevy`、Bevy queue completion 和双设备语义全部删除，确认 recovery 与 retirement 测试只走 native runtime。
4. 将 `prism_render_scene` 的直接 Bevy 依赖降为零，再删除该 crate 的生产 plugin/模块；仅保留有明确价值的算法、ABI、shader fixture，并由新 owner 发布。
5. 修改 workspace 默认成员，移除根包 `bevy` 的默认生产选择；随后再处理上游 `crates/bevy_*` 的 workspace/CI 归属。
6. 运行 `tools/check_prism_bevy_boundary.sh --strict`，并以 `cargo tree` 证明 native production closure 不再包含任何 `bevy_*`。

## 4. placeholder / stand-in / fallback 清单

### 4.1 当前 P0，不能在本轮删除

| 命中 | 位置 | 风险 | 处置 |
|---|---|---|---|
| Spine 白色 stand-in pipeline | `pkg/prism_render_app/src/mesh_pipelines.rs:639-666` | 真实生产 draw 可能输出固定白色，不能作为 AAA 光照像素链 | 等 forward-lit 的 frame/view/light/material bind group 和真实 HDR composite 完成后删除整个 Spine provider |
| Spine 构造入口 | `pkg/prism_render_app/src/plugin.rs:307` 当前已指向 `forward_lit`；历史 Spine 构造仍由实现保留 | 保留实现会允许错误回退或误用 | 在 native forward-lit 物理 GPU 验收后删除 Spine 类型、构造函数、测试和 recovery recipe |
| PSO `DrawDecision::Placeholder` | `pkg/prism_render_architecture/src/shader_package/pso_cache/warm_runtime.rs:80-88,234-255`；调用接口见 `pkg/prism_render_app/src/pipeline_cache.rs:1116-1129` | cold pipeline 时可能用 stand-in draw，违背生产路径无占位要求 | 生产调用固定 `allow_placeholder = false`；待 warm/recovery/diagnostic contract 完成后删除 Placeholder 变体，而不是静默跳过真实 draw |
| `fallback_material_header` | `pkg/prism_render_scene/src/shading/runtime.rs:570-573` 及 native/shading 测试 | 缺失材料时可能进入默认材质，掩盖 asset/material binding 错误 | 将缺失材料变成带诊断的显式 rejection；测试 fixture 可保留，生产 fallback 必须删除 |
| `FALLBACK_MATERIAL_HANDLE` | `pkg/prism_render_scene/src/extract/systems.rs:195` | Bevy extract 缺材料时静默替换 | 在 material owner native 化后改为 revision/asset error，禁止无诊断替换 |
| `SceneRhi::from_bevy` | `pkg/prism_render_scene/src/gpu/rhi.rs:53,211,528` | 双 RHI owner，恢复与提交边界不清晰 | native RHI 接管后删除，不做兼容 wrapper |

### 4.2 本轮确认不是生产占位，不应误删

以下命中是实现语义或测试辅助，不等于生产 stand-in：

- `gpu_scene`/buffer allocator 中为保证非零容量而保留的零值槽位：属于 ABI padding/生命周期占位，必须由计数或 generation 保证永不读取；不能按词面删除。
- water/cloth bind group 中空数组的单元素 zeroed buffer：属于 wgpu 非零 buffer 约束；拥有 pass 的 count 为零时不得读取。应保留契约测试，不应当作光照 fallback。
- BVH 建造过程中的 reserved node placeholder：属于构建期间的内部哨兵，发布前会被 patch；不在渲染生产 fallback 路径。
- shader 数学中的 `normalize_or`、有限值替代和边界零处理：是数值稳定性策略，不是材质/光照结果的静默替代。
- `stand-in` 出现在 Beer-Lambert、幂函数、解析边界或测试 golden 的注释中：是算法近似或 CPU/GPU parity 说明，不能仅因词面删除。
- `pkg/prism_render_shading/src/gi/surface_cache/gpu/tests.rs` 中“stand in for a real dispatch”：明确是测试限制说明，不是生产调用。

这些项目本轮视为已“安全处理”：已从删除清单中分类排除，避免误删 ABI 哨兵、数值 guard 和测试 oracle。没有修改它们的实现。

## 5. TODO 与静默 fallback 审计

在四个渲染包的 Rust/WGSL/WESL 实现文件中，未发现 `TODO` 或 `FIXME` 命中；因此没有可安全通过改文档“关闭”的 TODO。根 `Cargo.toml` 的 feature 注释仍有一个上游 Bevy 的 TODO（`Cargo.toml:384`），它属于根包上游功能命名，不是 Prism 渲染生产实现，本轮不改。

不能把所有 `unwrap_or` 视为静默 fallback。安全判据是：

- 解析失败/资源缺失是否转成明确错误、panic 或 rejection；
- fallback 是否影响最终颜色、深度、可见性或光照能量；
- 是否有计数、generation、有效标志或显式诊断证明 fallback 不会被读取。

按此判据，本轮真正需要 P0 处理的 fallback 是材料缺失、Spine 白色管线和允许 Placeholder PSO；普通容量边界、数值稳定和测试 oracle 不应列入删除。

## 6. 删除验收门

删除任何 Bevy 生产依赖或 fallback 前，必须同时具备：

1. `cargo metadata`：native production package 的正常依赖闭包不含 `bevy_*`。
2. `cargo tree -e normal`：无通过 `bevy_core_pipeline`、`bevy_pbr` 等传递引入的 Bevy 图。
3. 源码扫描：生产目录无 `use bevy_*`、`bevy::`、`from_bevy` 和旧 RenderApp/RenderWorld 入口。
4. native GPU required test：material、light、normal/position、camera/resize、depth occlusion、HDR > 1、display transform、device recovery 都改变并验证最终 target 像素。
5. 失败可见：资源缺失、pipeline 未就绪、设备丢失不能静默替换成白色、默认材质或黑色结果；必须返回 rejection/error 并进入诊断计数。
6. `tools/check_prism_bevy_boundary.sh --strict` 通过，且删除门结果被记录在同一变更的审计报告中。

## 7. 本轮变更与边界

- 新增本审计报告；未修改核心实现、manifest、脚本或测试。
- 未删除任何源码命中，因为当前 P0 仍缺少真实 forward-lit bind/最终 HDR target 闭环，直接删除会把未完成状态伪装成完成状态。
- 已安全排除 allocator 哨兵、空数组 GPU padding、数值稳定 guard、BVH 构建哨兵和测试-only stand-in 的误报。
- 下一步实现应优先完成 P0 表中的 native frame/light/material/view binding、真实 HDR composite、required GPU readback，然后按本报告顺序删除 Spine、生产 fallback 和 Bevy scene host。
