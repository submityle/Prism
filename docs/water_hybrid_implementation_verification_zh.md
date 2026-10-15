# Water Hybrid 行为契约实现验证

## 范围

本次接线覆盖水体 `MaskBlend` 计划与 raster shader：CPU planner、Hybrid WESL 入口、八项 golden 契约，以及 WESL 编译和能力分类后的真实 GPU gate。响应程序只有 `PBR` / `NPR` / `Custom` 三类；`Hybrid` 是 `MaskBlend` 组合计划，不是第四个响应程序。为兼容已发布的 shader package 身份，四个 raster 入口当前均保留：

- `water_surface_fs_pbr`
- `water_surface_fs_npr`
- `water_surface_fs_custom`
- `water_surface_fs_hybrid`

## 统一权重语义

CPU 与 WESL 都使用同一规则：

1. 将岸边信号和浅水信号分别限制到 `[0, 1]`。
2. `npr_weight = max(shore_signal, shallow_signal)`。
3. `pbr_weight = 1 - npr_weight`。
4. `overlay_opacity` 单独限制到 `[0, 1]`，在区域混合之后再应用 stylized overlay。

CPU 的物理单位输入由岸边距离和水深换算为归一化信号；raster ABI 保持 144 字节不变，WESL 使用已有 `style_params.w` 作为 Hybrid 的归一化浅水/overlay 控制量，并把 foam 作为运行时岸边信号。这样 GPU 入口不再使用独立的 `foam + bias` 权重公式，overlay opacity 也会实际影响最终 Hybrid 颜色。

## Golden 覆盖

`pkg/prism_render_scene/tests/water_hybrid_golden.rs` 覆盖：

- 深度混合
- 岸边混合
- 泡沫
- 焦散
- 反射层级
- 透明与 overlay 消费
- motion vector
- reactive mask

测试同时检查 CPU 数值、WESL 接线、四个入口和 raster WESL 编译。真实 GPU gate 还检查设备的 `max_bind_groups`：水面 raster 当前需要 10 个 bind groups；本机真实适配器可用但只提供 8 个，因此普通运行明确输出 `SKIP`，并未冒充 GPU 通过。设置 `PRISM_REQUIRE_GPU=1` 时该能力不足会升级为测试失败，避免 CI 把不兼容设备误报为验证成功。

真实验证结果（2026-10-15，沙盒外）：CPU/WESL golden **14 项通过**；真实适配器为 Apple M2 Metal，设备请求成功，但因 `max_bind_groups=8 < 10`，严格 GPU gate **1 项失败**，没有把能力不足记为通过。此前设备级校验还发现水面 shader 写入了不存在的 `SurfaceSample.geometry_opacity` 字段，已删除该错误写入并重新通过 WESL 编译门禁。
