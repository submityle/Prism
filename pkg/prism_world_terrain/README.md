# prism_world_terrain

Engine-neutral, `no_std + alloc` terrain and planetary primitives for Prism.

## Scope

- Cube-sphere mapping uses six fixed orientations and deterministic dominant-axis tie breaking.
- `TileKey` is a face-local quadtree key with stable child order and explicit cross-face edge transforms.
- Child expansion is fallible at `MAX_TILE_LEVEL`; callers cannot accidentally create an invalid level-31 tile.
- LOD uses projected geometric error in pixels, a clamped continuous level, and smooth morph alpha.
- Non-finite intermediate LOD values and fixed-point height multiplication overflow are rejected or saturated deterministically.
- Integer and Q32.32 positions subtract the origin before converting to `f32`; the result carries a conservative absolute error bound.
- Height sampling is requested with quantized tile-local coordinates and returns integer millimetres, making the interface recordable and replayable.
- Voxel storage is a validated little-endian, row-major byte contract with X as the fastest-changing axis and power-of-two streaming bricks.
- `TerrainStreamAdapter` projects each valid cube-sphere `TileKey` into a stable
  `prism_world_stream::CellManifest`, including an exact tile identity sector,
  conservative signed-millimetre planetary bounds, content metadata, and a
  deterministic derived-payload hash.

## Performance layout

The hot-path keys and geometry values are small `Copy` structs with no heap allocation. Tile traversal uses integer coordinates; only face-to-sphere conversion uses scalar `f32` normalization. Cross-face adjacency is a fixed table lookup plus one optional integer index reversal, so it does not require a floating-point direction round trip. Voxel data is contiguous, two bytes per component, X-major, and suitable for direct upload or brick-local staging. `alloc` is used only by the optional `VoxelBuffer` owner; the geometry, LOD, coordinate, and sampling contracts do not allocate.

The crate intentionally stops at data and algorithm contracts. Engine scheduling,
render submission, collision, persistence, and GPU formats remain external
adapters. The terrain-to-stream adapter is the narrow exception: it only emits
validated stream manifests and does not perform IO, decompression, residency,
or scene mutation.

## Terrain-to-stream adapter

`TerrainStreamConfig::new` validates a positive planet radius and non-zero
per-tile IO estimate. `TerrainStreamAdapter::manifest_for` then produces a
manifest that can be passed directly to `StreamScheduler::plan`. The stream
sector uses `(face, tile_x, tile_y)` at the tile's quadtree level, preserving a
one-to-one identity without pretending that the stream scheduler's signed grid
is a cube-sphere adjacency metric. Planetary spatial extent is carried by the
manifest AABB generated from the tile's cube-sphere patch.

The bounds calculation evaluates the rectangle's corner and axis-zero
extrema, encloses them in integer millimetres, and adds a radius-scaled margin
for the terrain crate's `f32` direction normalization. Batch conversion keeps
input order and stops with a typed error rather than emitting a partial
manifest set.

## Validation

From the repository root:

```text
cargo fmt --manifest-path pkg/prism_world_terrain/Cargo.toml -- --check
cargo test --manifest-path pkg/prism_world_terrain/Cargo.toml --no-default-features
cargo clippy --manifest-path pkg/prism_world_terrain/Cargo.toml --no-default-features -- -D warnings
```
