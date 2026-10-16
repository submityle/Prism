# Water Render Fabric 实现事实

本文件记录当前代码已经具备的水体 Fabric、response producer、resident composition、调度、性能预算和 GPU 能力验证。

## 状态摘要

- \`WaterFabricPlan\` 由 \`WaterShadingPlan\` 构造，并按稳定 body 顺序进入 render-world \`WaterGpuBody\`。
- 四个常驻 pass 具有稳定身份：simulation、surface raster、MaskBlend producer、Composition。
- MaskBlend 已从兼容占位声明升级为 active graphics producer；它写入 left/right/controls/coverage，legacy color/motion 仍由同一 fragment 保留。
- resident CompositionPass 在 shading resolve 后、water raster producer 后、SSR color mips 和 post-process 前执行。
- water raster pipeline 的 group 数已由 10 压缩为 7；GTAO、SSGI、world-space GI、caustics 共用 \`@group(6)\`。

## Fabric pass 与资源访问

| Pass | 队列 | 主要访问 | resident | compatibility-only |
|---|---|---|---:|---:|
| WaterSimulation | Compute | surface/foam 写入，上一帧 spray 读取 | 是 | 否 |
| WaterSurfaceRaster | Graphics | surface/foam/spray 读取，scene color 写入，water depth 读取 | 是 | 否 |
| WaterMaskBlend | Graphics | surface/foam/spray 读取，response left/right/controls/coverage 写入 | 是 | 否 |
| WaterComposition | Compute | left 写入，right/controls/coverage 读取，auxiliary 与 scene color 写入 | 是 | 否 |

\`WaterFabricPlan::is_link_complete()\` 校验服务闭包、四个 resident pass、队列、访问集合和依赖顺序。MaskBlend 现在直接使用 \`water_surface_fs_mask_blend\` resident entry；color/motion MRT 与 typed response planes 在同一生产路径中保持稳定 ABI。

MaskBlend 的 producer 访问集合明确包含：

- 水体 surface、foam、上一帧 spray 的读取；
- \`response_left\`、\`response_right\`、\`response_controls\`、\`water_coverage\` 的 storage write。

Composition 的访问集合明确包含：

- \`response_left\` 的 storage write；
- \`response_right\`、\`response_controls\`、\`water_coverage\` 的 storage read；
- auxiliary 与 HDR scene color 的 storage write。

## 实际帧图顺序

\`\`\`text
dispatch_shading_resolve
        -> water_surface_raster
        -> response_composition_pass
        -> ssr_color_mips_pass
        -> MainPass / post-process consumers
\`\`\`

render graph 为 response left、coverage 等资源生成 producer→consumer barrier。水体 draw 在 command encoder 中先清 coverage，再记录 graphics raster；draw 完成后发布当前帧 token，composition pass 只在同一 frame index 存在新 token 时记录 compute。这样既保持 GPU command order，又避免跨帧重复消费。

## Response ABI

| Resource | 类型 | stride | 生命周期 |
|---|---|---:|---|
| \`response_left\` | typed storage buffer | 64 B/pixel | imported per view |
| \`response_right\` | typed storage buffer | 64 B/pixel | imported per view |
| \`response_controls\` | typed storage buffer | 16 B/pixel | imported per view |
| \`water_coverage\` | \`u32\` storage buffer | 4 B/pixel | imported per view |
| \`response_auxiliary\` | typed storage buffer | 64 B/job | imported when jobs exist |

所有 buffer 使用 \`STORAGE | COPY_DST | COPY_SRC\`。这同时支持 GPU producer、显式上传和诊断 readback。allocation 前校验 extent、pixel count、buffer/storage limit、dispatch limit 和 workgroup shape；record 阶段不分配资源、不创建 bind group、不 mapping，也不执行 CPU response evaluator。

## 水体 shader 与 legacy ABI

\`water_surface_raster.wesl\` 的四个入口都调用 \`write_response_planes\`。PBR/NPR/custom 使用 identity arms；Hybrid 将 PBR/NPR region blend 编入 left，再将 stylized overlay 编入 right/mask，使 response HDR export 与 legacy fragment color 使用同一输出规则。

同时保留：

- location 0 的水体 color attachment；
- location 1 的 \`Rg16Float\` motion attachment；
- 四个稳定 entry-point 名称。

因此已有材质和 MRT ABI 不需要迁移，新的 response consumer 也可以直接读取 typed planes。

## 性能与扩展性

- 7 个固定 bind group layout 覆盖全部 lighting feature，feature 开关只改变配置 bit 和 fallback 资源，不造成 layout 分裂。
- response producer 只写 raster 覆盖的像素；coverage 每帧清零后作为 ownership sidecar，避免每帧清空两个 64 B/pixel radiance plane。
- resident composition pipeline 与 per-view allocation 分离：pipeline/device capability 只初始化一次，view extent 变化时只重建该 view 的 target/allocation。
- \`CompositionPlane\`、Fabric \`ResourceKey\` 和 frame graph resource name 使用同一稳定命名，减少跨模块字符串/枚举漂移。
- 新水体响应可复用同一 \`left/right/controls/coverage\` contract，不需要复制一套 compositor 或新增 bind-group family。

基础 response allocation 为 \`148 B/pixel\`；1920×1080 为 306,892,800 B，3840×2160 为 1,227,571,200 B，不含 auxiliary、HDR、visibility 和水体自身资源。

## 验证结果

- Fabric tests：验证服务集合、pass 唯一性、active MaskBlend、四资源写入、coverage read、依赖顺序和 history frame lag。
- Frame-graph tests：验证 resolve→water→composition 的执行顺序、left/coverage barrier 和 imported resource lifetime。
- Composition tests：验证 typed strides、allocation/admission、target validation、coverage clear、publication token 单次消费。
- WESL tests：验证四个 raster entry 编译，并验证 SSGI、world-space GI、caustics 等 merged group 6 binding 有真实消费点。
- GPU gate：真实 Apple M2 Metal adapter/device，\`max_bind_groups = 8\`，实际 water raster layout 为 7，严格 \`PRISM_REQUIRE_GPU=1\` 为 \`PASS\`。

GPU gate 的 PASS 只证明真实设备能创建并接受当前 shader module；数学 golden、frame graph 和资源生命周期由各自测试单独证明。

### 2026-10-16 收口边界

- `water::frame_golden::whole_frame_plan_matches_the_golden_digest` 已按当前
  canonical plan 重新固定为 `0x2d726174148032c7`，并以独立测试通过；这只证明
  CPU frame-plan 的确定性，不替代 GPU 输出证据。
- 新增的 `water_hybrid_gpu` integration target 直接编译生产 WESL，创建三种
  raster pipeline，执行 render-FX、resident composition 与 HDR export，并对
  storage/readback 结果做断言。它不是 CPU 重写 shader 的测试。
- 当前开发机本轮未拿到可用硬件 adapter，且首次完整编译曾因 `No space left on
  device` 中止；因此 `water_hybrid_gpu` 尚无本轮真实 dispatch/readback 通过证据。
  只能把既有跨沙盒 Apple M2/Metal 的 shader-module gate 记为历史证据，不能将其
  记作本 integration target 的输出通过。

## 结论

水体 Fabric 已拥有可执行的 producer→composition 闭环：声明层、GPU buffer、shader storage write、coverage 清零、publication token、frame graph barrier 和 HDR consumer 相互一致。legacy entry/MRT 是兼容输出，不是未实现的替代路径；新增响应功能应继续通过现有 typed plane 与 Fabric resource contract 接入。
