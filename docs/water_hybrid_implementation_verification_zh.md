# Water Hybrid 实现验证事实

本文件记录当前水体 Hybrid/MaskBlend 路径已经接通并验证的实现。水体 fragment 同时保留 legacy color/motion MRT，并向 resident response composition 写入 typed response planes；composition 在同一帧、同一 command stream 中消费这些写入。

## 已接通的实现

- CPU Hybrid planner、PBR/NPR plan 组合、权重 clamp/partition、overlay 和 reactive mask 已有 golden 覆盖。
- \`water_surface_raster.wesl\` 保留四个稳定 fragment 入口：\`pbr\`、\`npr\`、\`custom\`、\`hybrid\`。
- 四个入口都写入 \`response_left\`、\`response_right\`、\`response_controls\` 和 \`water_coverage\`；每个 fragment 仍同时返回 location 0 color 与 location 1 motion。
- 水体 draw 在录制前清零 \`water_coverage\`，所有水体 draw 完成后才发布当前 \`frame_index\`；没有有效 draw 时不会发布伪 token。
- resident composition 的 radiance shader 读取 coverage sidecar，按 controls 混合 response，并由 HDR bridge 写回共享 HDR scene color。
- frame graph 明确声明 \`shading_resolve -> water_surface_raster -> response_composition -> post-process\`，并为四个 response 资源声明读写访问。
- 水体 raster 由 10 个 bind group 压缩为 7 个：GTAO、SSGI、world-space GI 与 caustics 合并到 \`@group(6)\`，满足 Apple Metal 的 8 组能力上限。

## Hybrid response 的 GPU 编码

PBR、NPR 和 custom 入口把同一结果写入左右 arm，并使用零 mask，因此 response composition 对这些入口是恒等操作。

Hybrid 入口先计算 PBR/NPR 区域混合，再计算 reactive overlay。为使 response composition 的 HDR 结果与 legacy fragment color 使用同一数学规则，producer 写入：

1. \`left = mix(pbr, npr, npr_weight)\`；
2. \`right = npr\`；
3. \`controls.mask = overlay_factor\`，\`channels = 1\`（当前 \`ResponseOutput\` 的 direct RGB 通道）；
4. \`water_coverage = 1\`。

因此 compositor 执行的 \`mix(left, right, overlay_factor)\` 与 legacy fragment 的最终 Hybrid 颜色一致；legacy MRT 仍保留，便于兼容现有 motion 和目标附件 ABI。

## 帧生命周期与防陈旧

每个 view 的流程固定为：

\`\`\`text
clear water_coverage
    -> water raster writes left/right/controls/coverage
    -> publish(frame_index)
    -> response_composition consumes the token once
    -> HDR bridge exports the composed response
\`\`\`

\`publish\` 只发布 GPU command order 中已经记录的写入，不表示 CPU 等待 GPU 完成。\`record\` 只接受当前 frame token；缺失、过期或已经消费的 token 返回 \`Ok(false)\`，不会重复记录 composition work。target、pipeline、extent、buffer/storage limit、dispatch limit 和 workgroup shape 均在录制前校验。

coverage 每帧在水体 draw 的最前面清零，即使随后因为 MSAA、pipeline、light group 或 body 状态而提前返回，也不会把上一帧水体像素带入本帧 composition。

## 验证覆盖

\`pkg/prism_render_scene/tests/water_hybrid_golden.rs\` 和相关 unit tests 覆盖：

- CPU depth/shore mixing、foam、caustics、reflection tier、透明度、overlay、motion、reactive mask；
- clamp/partition 守恒、固定 workload digest 和 CPU/WESL reference formula；
- WESL raster compile、四个入口保留、shader cache cold/warm；
- response plane stride、coverage allocation、coverage graph read/write、publication token 单次消费；
- producer → composition 的 graph 顺序、resource barrier 和 bind-group 数量不超过 8；
- 真实设备上的 water raster shader module 创建门禁。

这些测试分别验证 CPU 数学、源级 ABI、frame-graph 声明、资源生命周期和设备能力；GPU module gate 不冒充逐像素图像对拍。

## Resident response ABI 与性能

| Plane | stride | 用途 |
|---|---:|---|
| \`response_left\` | 64 B/pixel | shading resolve 或 producer 的左 arm |
| \`response_right\` | 64 B/pixel | producer 的右 arm |
| \`response_controls\` | 16 B/pixel | mask、channel、alpha merge rule |
| \`water_coverage\` | 4 B/pixel | 当前帧水体覆盖标记 |
| \`response_auxiliary\` | 64 B/job | 可选 typed auxiliary job |

基础 response allocation 为 \`148 B/pixel\`，不含 auxiliary、HDR、visibility 和水体自身资源。left/right 在 1920×1080 下为 132,710,400 B；加上 controls 与 coverage 后为 306,892,800 B。radiance kernel 使用 \`ceil(pixel_count / 64)\` 个 workgroup，HDR bridge 使用 \`ceil(width / 8) × ceil(height / 8)\`。

water raster 的实际 pipeline layout 为 7 个 bind group。7 组布局不依赖可选特性是否开启：关闭特性时仍绑定 format-correct fallback，配置 bit 决定 shader 是否采样，从而避免每个 feature 重新创建 pipeline layout。

## GPU 门禁结果

- 默认模式：adapter 不可用、device request 失败或 \`max_bind_groups < 7\` 时分类为 \`SKIP\`。
- 严格模式 \`PRISM_REQUIRE_GPU=1\`：同样的能力不足分类为 \`FAIL\`。
- 只有真实 adapter/device 满足 7 组能力、device request 成功且 water raster shader module 创建成功，才输出 \`PASS\`。
- 本轮真实验证设备为 Apple M2 Metal，\`max_bind_groups = 8\`；严格 GPU 门禁已通过，结果为 \`PASS\`。

该 PASS 证明真实设备和 shader module ABI 可用；CPU/WESL、frame graph、publication 和 response resource 测试分别证明其余闭环，不把单一门禁扩大解释为图像质量基准。

## 结论

Water Hybrid 已完成从水体 raster producer、coverage 生命周期、publication token、frame graph 顺序到 resident composition/HDR 输出的接线。四个 legacy 入口和 color/motion MRT 作为兼容 ABI 保留，但不再代表水体只能走 legacy producer；MaskBlend 已是 active response producer，Fabric 与正式 frame graph 均记录了真实的四资源写入/消费关系。
