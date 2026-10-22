# GPU 行星高度场 placement

`prism_world_gpu::GpuPlanetPlacement` 为 `PlanetPlacementConfig`、
`PlanetPlacementRegion` 和完整 `HeightfieldTile` 提供固定容量 GPU placement。请求先检查候选
容量、heightfield 快照、dispatch、设备 buffer 限额与工作预算；失败不上传新请求、不覆盖旧请求。

生产 shader 采用 U256 整数 cube-sphere 投影和 radial-adaptive Q37..Q84 精度。当前 ABI 为
116-byte 候选 record：稳定 identity 前六个 word，后续 xyz 使用 signed-i128 frame-local
毫米；不把行星坐标转换为绝对 `f32`。每个 height sample 为 16-byte signed-i128 limbs。
compact source-slot 与 `[vertex_count, accepted_count, 0, 0]` indirect 参数可直接留在 GPU，
正常 `encode` 不提交、不等待、不读回；`encode_readback` 和候选诊断 readback 仅供测试/观测。

`GpuPlanetPlacement::registration_snapshot()` 暴露 6 个 buffer、候选与 finalize stage、generation、
candidate count 及 `output_origin_mm`，`snapshot.encode()` 记录 indirect/status reset 和非空请求的两个
dispatch。完整键删除 producer 的 `registration_snapshot()` 另暴露上游 placement snapshot、删除
params/keys、过滤 compact/indirect 与非空删除 stage；empty request 仍要求 consumer reset 删除后的
indirect count。两个 snapshot 只 clone 同设备 native handles，本身不提交工作或持有完成期资源 lease。

截至 2026-10-23，`placement_rhi` feature 的 RHI bridge 仍只接 planar heightfield placement/deletion；
尚无 planetary RenderGraph lowering、planetary RHI 注册/submit/completion retirement、signed-i128
frame-local record draw decoder 或产品 draw consumer。不得把 planar 64-byte record draw binding
复用于行星 ABI，也不得将 producer snapshot 视为 RHI 接线完成。跨 sector 生命周期管理仍未闭环。

2026-10-23 当前工作树验证记录：Apple M2 / Metal / IntegratedGpu 的 planetary placement
测试 17/17、U256/reference 实卡测试 14/14、完整键 deletion 实卡测试 1/1 通过；离线
`cargo test -p prism_world_gpu --features gpu --lib --offline` 为 139/139，
`cargo test -p prism_world_runtime --lib --offline` 为 183/183。实卡测试由获准宿主执行；
普通 sandbox 没有 Metal adapter。行星 placement release 基准经过有效-limb 剪枝后，62,500
候选三轮 submit+wait P50 为 24.343、24.436、25.418 ms；较同工作树优化前约 29.058 ms
有所下降，仍超过 16.7 ms。该数据是 Apple M2 fixture 的 host submit+wait，不是 GPU timestamp，
也不代表产品帧时或跨厂商性能。详见
`prism_world_system_implementation_zh.md` 的 2026-10-22/23 记录。

同日 `cargo check -p prism_world_gpu --features placement_rhi --offline` 编译成功；获准 Metal
宿主运行 `cargo test -p prism_world_gpu --features placement_rhi --test placement_rhi --offline
-- --nocapture --test-threads=1` 为 14/14 通过，覆盖的是 planar RHI bridge，不是 planetary
consumer。当前 feature check 对 planetary deletion `device_identity` getter 有一条 dead-code
warning；consumer 集成后应复验无 warning。

可重复的离线命令：

```text
cargo test -p prism_world_gpu --features gpu --lib --offline
cargo test -p prism_world_gpu --features terrain_scatter --test planet_placement --offline
```

严格 Metal 实卡命令（必须在能取得 Apple Metal adapter 的宿主运行）：

```bash
WGPU_BACKEND=metal PRISM_REQUIRE_GPU=1 \
cargo test -p prism_world_gpu --features terrain_scatter \
  --test planet_placement_gpu --offline -- --nocapture --test-threads=1
WGPU_BACKEND=metal PRISM_REQUIRE_GPU=1 \
cargo test -p prism_world_gpu --features terrain_scatter \
  --test planet_placement_u256_gpu --offline -- --nocapture --test-threads=1
WGPU_BACKEND=metal PRISM_REQUIRE_GPU=1 \
cargo test -p prism_world_gpu --features terrain_scatter \
  --test planet_placement_deletion --offline -- --nocapture --test-threads=1
```

仍未闭环的范围：行星 placement/deletion 的 RHI 注册与 retirement、RenderGraph hazard/lowering、
signed-i128 record 的生产绘制 consumer、`prism_render_scene` mesh/material/visibility 接入、
跨 sector 连续性、混合 LOD 行星 mesh stitching、真实场景画质/帧时，以及多厂商硬件验收。
