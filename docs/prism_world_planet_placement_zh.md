# 行星 Cube-Sphere Heightfield Placement

## 交付边界

本切片实现 `prism_world_terrain` 的 engine-neutral 行星贴地 placement CPU contract：

- 使用 `TileKey`/`CubeFace` 作为稳定空间身份，在 face-local cell 中确定性生成候选 key。
- 以 cube-sphere 面中心方向投影到球面，使用高度 tile 四角的中心双线性高度，输出绝对 i64 毫米坐标和外向单位法线。
- 采用半开 cell ownership；相邻 tile 不重复拥有边界 cell，跨 cube face 继续复用同一面向规则。
- 在枚举和分配前检查候选数量、输出数量、容量字节、半径和坐标范围；错误不返回部分结果。
- `PlanetPlacementKey` 包含算法版本、seed、recipe version、salt、tile 和 cell，不使用 hash 或 dense slot 作为身份。

入口：

```rust
use prism_world_terrain::{
    place_on_planet_heightfield, PlanetPlacementBudget, PlanetPlacementConfig,
    PlanetPlacementRegion,
};
```

## 验证

```bash
rustfmt --edition 2024 --check \
  pkg/prism_world_terrain/src/planet_placement.rs \
  pkg/prism_world_terrain/src/lod_seam.rs \
  pkg/prism_world_terrain/src/lib.rs

CARGO_TARGET_DIR=/private/tmp/prism-world-planet-placement-20261016 \
cargo test --offline -p prism_world_terrain --no-default-features --lib --no-fail-fast

CARGO_TARGET_DIR=/private/tmp/prism-world-planet-placement-clippy-20261016 \
cargo clippy --offline -p prism_world_terrain --no-default-features \
  --all-targets -- -D warnings
```

结果：terrain no-default library 测试 `91 passed`；strict Clippy 通过。

覆盖包括六面 outward placement、相邻 tile 的确定性和不重复 ownership、预算拒绝、负径向距离 fail-closed、i64 毫米输出和重复调用确定性。

## 未包含边界

- 没有 GPU dispatch、GPU indirect draw、renderer graph 或 ECS decoder 接线。
- 没有行星跨 tile 高度连续性修复、跨 LOD seam stitch 或跨 tile 法线平滑；这些仍由 mesh/seam 消费者负责。
- 没有风、交互、数据层、持久化删除、网络复制或编辑器 authoring 接线。
- 没有跨硬件 GPU 性能、产品帧时、功耗或 UE 对比结论。
