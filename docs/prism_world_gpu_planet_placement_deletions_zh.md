# 行星 GPU Placement 完整键删除切片

本切片把 `GpuPlanetPlacement` 的固定容量行星候选流连接到独立的 GPU 完整键删除 pass。删除身份不是 compact slot，而是完整的 `PlanetPlacementKey`：

`placement_version、seed、version、salt、tile(face/level/x/y)、cell(u/v)`。

GPU ABI 为 12 个 `u32`、48 bytes。host 端先验证上游 placement、键容量、输入槽预算、二分比较预算、scratch 和 wgpu binding/dispatch 限制；验证失败不会上传新 uniform、键或替换旧 request。键集合沿用 CPU 的排序、去重和完整依赖语义。

删除 shader 只读取上游 records、compact 和 indirect；它通过完整键二分查找、64-lane workgroup prefix scan 生成自己的 compact 前缀，并以原子加法累计 indirect instance count。上游 records、height samples、status 和属性不被改写。

## 验证

设备无关的 Naga、ABI、键打包、预算和容量测试：

```bash
CARGO_TARGET_DIR=/private/tmp/prism-world-planet-deletion-test \
cargo test --offline -p prism_world_gpu --features terrain_scatter --lib
```

当前已验证：128 tests passed。

严格真实设备回归（不把 CPU/noop/virtual adapter 当作 GPU）：

```bash
CARGO_TARGET_DIR=/private/tmp/prism-world-planet-deletion-gpu \
WGPU_BACKEND=metal \
cargo test --offline -p prism_world_gpu --features terrain_scatter \
  --test planet_placement_deletion -- --nocapture --test-threads=1
```

该测试在没有可用硬件 adapter 时明确失败而不是静默跳过。2026-10-20 已在 Apple M2 / Metal / IntegratedGpu 上通过 1 项：真实创建 placement 与 deletion compute pipeline，提交 dispatch，读回 indirect/compact/records，并确认删除一个完整键后仍保留其余三个 cell。受限沙盒内的同一命令因无 Metal adapter 失败，不计作通过；本条证据来自宿主机跨沙盒运行。

## 边界

- 该切片只完成 planetary placement 的 GPU deletion producer，不声称已经接入产品 scene/ECS decoder、renderer graph、physics、GI 或网络 session。
- GPU 行星 placement 本身的 `f32` 几何仍不是 CPU `i64` position parity；本切片只保证完整键删除身份和 compact/indirect 过滤语义。
- 未进行多厂商 adapter、跨 tile 法线、LOD seam、长期 VRAM 压力或产品内容级 P95/P99 性能验收。
