# Prism Native Lighting M1 实施状态与验收记录

> 更新日期：2026-10-21
>
> 状态：M1 surface/view 接入切片进行中；不是 AAA 光照/GI 完成声明。
>
> 设计权威：[v20 AAA 设计](prism_lighting_gi_design_v20_aaa_zh.md)；验证口径：[Native Lighting 验收基线](prism_native_lighting_validation_acceptance_zh.md)。

## 1. 本切片的目标

把后续真实 deferred lighting 所需的两项基础合同落到原生 Prism render app：

1. opaque surface pipeline 只编码 surface，不在 raster provider 内评价灯光；
2. 每个真实 native camera 提供有效的当前/上一帧 clip-from-world 与 viewport，历史无效时显式标记，不合成默认相机。

此切片不包含逐像素 direct lighting、shadow、GI、reflection、HDR output，也不表示 queue 已形成可执行的 per-view draw。

## 2. 当前实现证据

| 项目 | 当前证据 | 状态 |
| --- | --- | --- |
| Surface provider | `pkg/prism_render_app/src/forward_lit.rs` 提供 `SurfaceEncodeProvider`、128B `SurfaceFrame`、32B `SurfaceMaterial` 和实际 group-0 layout；frame/material 为动态 uniform | 已实现，待构建验证 |
| Pipeline/recovery | `MeshPipelines::surface_encode` 作为显式生产构造器；设备恢复时重建 provider layout | 已实现，待构建验证 |
| Native camera extraction | `surface_views.rs` 要求 `PrismCamera + PrismFrustum + GlobalTransform`，验证矩阵、frustum、viewport；拒绝原因结构化 | 已实现；内嵌测试存在，尚未运行 |
| Camera history | 仅连续的同 world/entity incarnation、viewport、epoch 且无 camera cut 才继承上一帧；否则 previous=current 且 history invalid | 已实现；测试存在，尚未运行 |
| Plugin extraction | `surface_views.rs` 已声明为 app 模块、公开导出，并注册到 native render extraction 链 | 已接入，待编译验证 |
| Per-view visibility | 当前 `VisibilityWorkList` 仍是无 view identity 的 `GpuRenderWorkItem` 集合 | 未实现；不能把多相机 work 合并后绑定任意相机 |
| 实际 draw bindings | 当前 `queue_stage.rs` 仍将空 bind-group 列表入队；frame/material arena、对齐动态 offsets、buffer 生命周期与真实 bind group 尚未接通 | 未实现；当前 surface pipeline 不能据此宣称可绘制 |
| GBuffer profile | 当前 provider/target 实际为六附件：XYZ 拆分 world position、normal、albedo、surface ID | 与 v20 四 MRT 目标不符；缺 roughness/metallic/emission packing 与独立 motion |
| 真实 surface/material 输入 | LightingSceneInputs 的默认值和 with_stats 构造器现为空 surface 集，新增回归测试；逐像素材质采样未接通 | 默认输入已修正，draw data 链未完成 |
| GPU raster/readback | 尚无本切片的真实场景→相机→opaque raster→GBuffer readback 证据 | 未验证 |

## 3. 不变量与禁止的捷径

- `PrismCamera.clip_from_world` 已是 clip-from-world；不得再次乘相机 transform。`GlobalTransform` 用于校验相机实体具备有效 native pose。
- 无有效 view 时拒绝或跳过该 view，并留下诊断；不得 identity fallback。
- 队列项必须保留 view identity/generation。禁止把不同 view 的工作列表 union 后统一绑定一个 camera。
- provider 的实际 bind-group layout 是资源创建依据；设备恢复后旧 layout/bind group 不可复用。
- dynamic uniform offset 必须符合设备最小对齐；每个 draw 的绑定范围、buffer bounds 和在途帧退休都必须验证。
- 空 bind groups、默认材质/灯光、shader 能编译、pipeline 能创建，都不能作为真实 surface draw 或照明完成证据。

## 4. 下一阶段完成门

### M1a：可提交的真实 surface draw

- 把可见工作改为携带 view identity 的 per-view work；视图 generation 与 target generation 纳入键。
- 建立按帧/代际管理的 frame/material uniform arena，验证动态 offset 对齐、最大 binding range、容量溢出策略和 deferred retirement。
- queue 阶段按真实 camera、material、geometry 解析 surface bindings；任一必需输入缺失时 fail closed 并产出结构化诊断。
- 将 default test fixture 移出生产插件初始化，但保留 CPU/GPU oracle 测试 fixture。
- 扩展测试覆盖双相机、resize、camera cut、entity/component 重建、buffer 扩容与设备恢复。

### M1b：像素正确性门

- 真实输入产生不透明几何；验证变换、法线、深度和当前帧 surface metadata。
- GPU raster 后 readback attachments，与确定性 CPU/reference oracle 对拍。
- 单独证明 viewport 裁剪、多 view target 隔离、无效 surface、透明分流和 resize/recovery。

### 后续照明阶段

本记录的 M1a/M1b 只是 v20 M1 内部的 surface/view 子步骤，不替代 v20 M1 所要求的紧凑逐像素材质、motion、dynamic binding 与 composite 闭环。随后按 v20 M2 接 direct GGX、传统 shadow 和 HDR 回读，再逐项完成间接光、reflection、temporal、HDR 输出、自动预算与规模化 provider。每项绑定需求、合同、测试、GPU artifact 和退出门。

## 5. 本轮验证记录

| 检查 | 结果 | 说明 |
| --- | --- | --- |
| Surface provider / pipeline 定向 rustfmt | `PASS`（worker 报告） | 仅格式检查，不等于编译 |
| Surface view 模块定向 rustfmt | `PASS`（worker 报告） | 21 项内嵌测试尚未执行 |
| 定向 cargo test | `BLOCKED`（worker 报告） | 当时链接 `parking_lot_core` build script 因磁盘空间不足失败；未清理共享 target |
| `cargo check -p prism_render_app --lib` | `BLOCKED` | target 清理后并发 workspace 构建占满磁盘；Cargo 在建 fingerprint 时返回 `No space left on device`。未重复清理共享 target |
| 真实物理 GPU raster/readback | `NOT RUN` | 没有 GPU 像素证据 |
| 性能/显存 P50/P95/P99 | `NOT RUN` | 没有 benchmark 证据 |

## 6. 文件责任边界

- `pkg/prism_render_app/src/forward_lit.rs`：surface-only provider 与 shader/ABI。
- `pkg/prism_render_app/src/mesh_pipelines.rs`：provider selection 与恢复 recipe。
- `pkg/prism_render_app/src/surface_views.rs`：真实相机提取、history 和拒绝诊断。
- `pkg/prism_render_app/src/lib.rs`、`plugin.rs`：后续需要完成模块声明/导出及原生 render extraction 注册。
- `pkg/prism_render_app/src/queue_stage.rs`、`scene.rs`：下一阶段 per-view identity 与 binding 闭环的主要缺口。
