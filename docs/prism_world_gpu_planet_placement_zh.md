# GPU 行星高度场 placement

`prism_world_gpu::GpuPlanetPlacement` 是 `PlanetPlacementConfig`、`PlanetPlacementRegion`
和完整 `HeightfieldTile` 的固定容量 GPU placement 切片。请求先做候选预算、候选容量、
完整 heightfield 快照、dispatch 和设备 buffer 限额检查；检查失败不会上传新请求，也不会
覆盖旧请求。

GPU 资源包含 192 字节 uniform、80 字节候选记录、每样本 16 字节 signed-i128 limbs 高度、
compact source-slot、8 字节 status 和 `[vertex_count, accepted_count, 0, 0]` indirect 参数。
WGSL 使用 cube-face cell-center 投影、四角平均高度、确定性 hash 阈值和原子 compact 计数。
`encode` 只记录候选 compute 与单 workgroup finalize 两个 compute
pass，不提交、不等待、不读回，必须由调用者在提交 encoder 后再写下一请求。
如需设备验证，`encode_readback` 会把 status 与 indirect 参数复制到调用者提供的至少 24
字节 readback buffer；生产路径仍可直接消费 indirect buffer，不强制 readback。

当前设备 ABI 不启用 shader-f64 或 shader-i64。渲染位置和投影因此在 shader 中以 `f32` 近似；
高度、半径和 origin 先以 signed-i128 limbs 上传，再在 shader 中转换为 `f32`。完整的 i64
CPU 配置、tile、region 和样本仍保存在 `GpuPlanetPlacementCacheKey` 中。该切片
提供 GPU placement 资源和 indirect 输出，不宣称与 CPU `place_on_planet_heightfield` 的
i64 毫米坐标逐位相等，也不包含 scene/ECS、renderer graph 或跨 LOD seam。GPU 候选记录现在
提供独立的 diagnostic readback；真实设备回归逐候选核对 tile/cell identity、CPU 导出的稳定
rank、四角高度 numerator、threshold accepted 和 invalid 标志。这是候选语义 parity，不是
`f32` 几何位置的 i64 毫米逐位 parity。`planet_placement_deletion` 已在 Apple M2 / Metal /
IntegratedGpu 上对包含该 placement producer 的真实 compute pipeline 完成 dispatch/readback
回归。

离线无设备验证：

```text
cargo test --offline -p prism_world_gpu --features terrain_scatter --test planet_placement
cargo test --offline -p prism_world_gpu --features terrain_scatter --lib
```

严格真实设备回归：

```bash
CARGO_TARGET_DIR=/private/tmp/prism-world-planet-placement-gpu-20261020 \
WGPU_BACKEND=metal PRISM_REQUIRE_GPU=1 \
cargo test --offline -p prism_world_gpu --features terrain_scatter \
  --test planet_placement_gpu -- --nocapture --test-threads=1
```

2026-10-20 本次隔离运行已完成 placement GPU 测试编译；严格 Metal 命令实际因运行环境未
返回可用 adapter 而未执行 shader，输出为 `metal found no adapters`，因此不能记为真实设备
通过。测试仍明确拒绝无硬件 adapter，不把 CPU/noop/virtual adapter 算作通过。离线 WGSL/CPU
验证已通过，真实 Apple M2 / Metal / IntegratedGpu 的该 placement producer dispatch/readback
与候选语义 parity 仍待在有 adapter 的环境重跑；它也不证明与 CPU i64 几何逐位相等。
