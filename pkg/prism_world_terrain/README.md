# prism_world_terrain

Engine-neutral, `no_std + alloc` terrain and planetary primitives for Prism.

## Scope

- Cube-sphere mapping uses six fixed orientations and deterministic dominant-axis tie breaking.
- `TileKey` is a face-local quadtree key with stable child order and explicit cross-face edge transforms.
- LOD uses projected geometric error in pixels, a clamped continuous level, and smooth morph alpha.
- Integer and Q32.32 positions subtract the origin before converting to `f32`; the result carries a conservative absolute error bound.
- Height sampling is requested with quantized tile-local coordinates and returns integer millimetres, making the interface recordable and replayable.
- Voxel storage is a validated little-endian, row-major byte contract with X as the fastest-changing axis and power-of-two streaming bricks.

## Performance layout

The hot-path keys and geometry values are small `Copy` structs with no heap allocation. Tile traversal uses integer coordinates; only face-to-sphere conversion uses scalar `f32` normalization. Cross-face adjacency is a fixed table lookup plus one optional integer index reversal, so it does not require a floating-point direction round trip. Voxel data is contiguous, two bytes per component, X-major, and suitable for direct upload or brick-local staging. `alloc` is used only by the optional `VoxelBuffer` owner; the geometry, LOD, coordinate, and sampling contracts do not allocate.

The crate intentionally stops at data and algorithm contracts. Engine scheduling, streaming policy, render submission, collision, persistence, and GPU formats remain external adapters.

## Validation

From the repository root:

```text
cargo fmt --manifest-path pkg/prism_world_terrain/Cargo.toml -- --check
cargo test --manifest-path pkg/prism_world_terrain/Cargo.toml --no-default-features
cargo clippy --manifest-path pkg/prism_world_terrain/Cargo.toml --no-default-features -- -D warnings
```
