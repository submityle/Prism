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
- Representation selection covers heightfield, voxel, and signed-distance data with validated dimensions, deterministic query-driven selection, and explicit conversion metadata.
- Incremental rebake planning merges tile-local dirty regions deterministically and admits work through task, abstract-work, and output-byte budgets without allocating in the geometry hot path.
- Heightfield baking supports bounded full builds and atomic local resampling;
  its indexed planar mesh updates only the dirty vertices and normal halo.
- `PlanarHeightfieldSurface` queries a shared baked snapshot with exact integer
  triangle interpolation and upward geometric normals, without a mesh or float
  world coordinates. Its explicit embedding has half-open XZ ownership.
- Optional `scatter` combines source scatter and author density/species with
  exact height, signed vertical-clearance and rational-slope thinning. It only
  removes points, preserving source keys, minimum distance and XYZ partition
  invariance; this is not ground projection or guaranteed surface density.
- Planet mesh baking projects those heights using f64 tile coordinates and an
  integer-millimetre origin, checks local f32 position error against a caller
  budget, and preserves level-30 tile detail. The optional `mesh` feature emits
  validated native Prism meshes for both planar and planet geometry.
- Bounded SDF volume baking stores signed Q16.16 samples in an X-fastest packed grid, stages local resamples before publication, and exposes an exact Q8.8 voxel adapter without pretending to reconstruct missing metric data.
- `TerrainStreamAdapter` projects each valid cube-sphere `TileKey` into a stable
  `prism_world_stream::CellManifest`, including an exact tile identity sector,
  conservative signed-millimetre planetary bounds, content metadata, and a
  deterministic derived-payload hash.

## Performance layout

The hot-path keys and geometry values are small `Copy` structs with no heap allocation. Tile traversal uses integer coordinates. The direction contract uses scalar `f32` normalization; planet mesh projection keeps tile coordinates and normalization in `f64` until origin-relative metre encoding. Cross-face adjacency is a fixed table lookup plus one optional integer index reversal, so it does not require a floating-point direction round trip. Voxel data is contiguous, two bytes per component, X-major, and suitable for direct upload or brick-local staging. Buffer owners, baking, and planning use `alloc`; the basic key, LOD, coordinate, and sampling contracts do not allocate. Heightfield baking and mesh construction deliberately allocate bounded staging/output buffers and are not zero-allocation hot paths.

The default feature set remains `no_std + alloc`. The optional `mesh` feature is a
host/render-edge adapter and currently enables the standard-library-backed
`prism_mesh` and `prism_render_driver` dependencies; it is not a no-std feature
combination. Cache-lineage tokens on baked heightfields prevent a change from
one independent cache from being applied to another cache with the same tile,
specification, and revision.

The optional `scatter` feature remains `no_std + alloc` and only adds the
engine-neutral residency kernel. Surface construction/queries allocate nothing;
scatter reuses the source's two point/attribute Vec allocations for stable
in-place thinning. Terrain work is admitted as four logical sample reads per
primary candidate before enumeration. The complete cache key shares the actual
immutable terrain snapshot and structurally compares all samples; tile/revision
alone is not identity. Snapshot residency and structural comparison costs remain
caller concerns. Exact contracts and benchmark methodology are documented in
`docs/prism_world_heightfield_scatter_zh.md` at the repository root.

The crate owns data contracts and executable geometry/data kernels. Engine scheduling,
render submission, collision, persistence, and GPU formats remain external
adapters. The terrain-to-stream adapter is the narrow exception: it only emits
validated stream manifests and does not perform IO, decompression, residency,
or scene mutation.

## Representation and rebake planning

`TerrainRepresentationSelector` chooses a representation from explicit query
requirements: signed-distance data for collision or distance queries, voxel data
for volumetric queries, and heightfields for surface-only queries. The selector
is deterministic and does not inspect runtime device state, so the same input
produces the same representation during replay and offline baking.

`RebakePlanner` accepts closed Q16 tile-local dirty regions, merges overlapping
or edge-adjacent regions in stable tile/coordinate order, estimates target
representation work and output bytes with checked arithmetic, and admits only
the tasks that fit the caller's budget. Deferred regions remain explicit in the
returned plan and can be submitted again with a later budget without changing
the ordering.

`SdfGrid` applies the same admission discipline to volumetric data: dimensions,
sample calls, full-grid bytes, and local-update scratch bytes are checked before
sampling. A local update reads only its half-open XYZ region into a temporary
buffer, then copies it into the packed grid and increments the revision; a
source failure leaves both the bytes and revision unchanged. `VoxelSdfSource`
adapts an existing signed-distance Q8.8 voxel field by an exact 256x fractional
expansion. It rejects density/unsigned formats and does not infer a physical
distance transform from voxel values.

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
cargo test --manifest-path pkg/prism_world_terrain/Cargo.toml --features scatter
cargo clippy --manifest-path pkg/prism_world_terrain/Cargo.toml --features scatter --all-targets -- -D warnings
cargo check --manifest-path pkg/prism_world_terrain/Cargo.toml --features mesh
cargo test --manifest-path pkg/prism_world_terrain/Cargo.toml --features mesh
cargo clippy --manifest-path pkg/prism_world_terrain/Cargo.toml --no-default-features -- -D warnings
```

The volumetric baker is covered by the same test command, including singleton
axes, extreme signed encodings, exact budget admission, overflow rejection,
atomic local resampling, revision overflow, source failures, and voxel-format
validation.

Planet meshes use 32 bytes per vertex plus 24 per quad, with checked budgets and
fallible reservations. Six-face outward winding, unit normals, same-resolution
tile boundaries, a level-30 Earth-radius precision case, and native mesh
validation are tested. The checked error covers projection arithmetic and f32
encoding, not height quantization or surface approximation. Boundary normals are
one-sided; mixed-LOD stitching and cross-tile normal continuity are separate
operations, not guarantees of this baker.

The release benchmark `cargo bench -p prism_world_terrain --bench heightfield_bake`
reports nearest-rank P50/P95/P99 for full/local heightfield and SDF baking,
planar mesh builds/updates, and full planet mesh projection. Logical payload
sizes are printed separately from timings; they are not allocator/RSS peaks.

`cargo bench -p prism_world_terrain --features scatter --bench heightfield_scatter`
measures 100k exact queries (including result checks) and full 10k-candidate
source/author-field/terrain thinning separately. It checks zero query allocations,
exactly two full-result allocations, stable buffer capacities and every final
point/attribute pair. Eight warmups and 64 samples yield nearest-rank
P50/P95/P99 host wall time, not product frame time or GPU measurements.
