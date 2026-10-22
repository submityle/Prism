# `prism_world_residency`

This crate owns deterministic, engine-neutral residency primitives. Its sparse
SDF slice is CPU-only: `SparseSdfBrickGrid::export_allocated_bricks` produces a
bounded packed staging payload for a future GPU adapter, but it does not acquire
an adapter, allocate a device buffer, dispatch a shader, or claim GPU sampling
parity.

## HLOD proxy bake contract

`bake_hlod` produces a complete CPU-side HLOD bake artifact, not just a
cluster declaration. Every non-empty representation contains a deterministic
axis-aligned proxy box with eight finite vertices, thirty-six triangle-list
indices, source-derived bounds, and a stable geometry hash. It also contains an
engine-neutral material key, finite base-color/roughness/metallic parameters,
and a stable material and artifact hash. Input records are canonically sorted,
so input iteration order does not affect any declaration, payload, or hash.

The existing HLOD byte budget covers the complete estimated output: declaration
metadata plus all proxy geometry/material payloads. Count, finite-value, radius,
size, and generated-geometry validation happens before a successful output is
returned; failures return an error rather than a partial bake. Empty input is a
valid, stable empty artifact with zero output bytes and no proxy payloads.

This is an engine-neutral baked artifact boundary. The crate does not submit
geometry to a renderer, create GPU buffers, choose a material backend, stream
assets, or materialize ECS/scene entities. Those consumers must validate their
own upload, residency, and rendering contracts against the returned payload.

## SDF export contract

`export_allocated_bricks` validates both limits in `SdfExportBudget` before
allocating output. The metadata order is the grid's `BrickCoord` order; each
record points into one contiguous `i16` sample array using the existing
z-major/y-next/x-fastest brick layout. Export overflow and budget violations
return `SdfError` and never return a partial payload.

The packed result is an adapter boundary, not a GPU result. An adapter may
upload `SdfBrickExport::bricks` and `SdfBrickExport::samples` and validate its
own device limits, binding layout, shader, and readback parity separately.

This crate has no adapter test. Therefore an external GPU sampling test must
report `SKIPPED` when no usable adapter is available by default. With
`PRISM_REQUIRE_GPU=1`, the same missing adapter/device or unsupported required
limits must be a strict failure. A CPU export test passing is not a GPU pass,
and a skipped adapter test must not be counted as parity evidence.

## Verification

```text
cargo fmt --check --package prism_world_residency
cargo test --offline -p prism_world_residency --no-default-features
cargo clippy --offline -p prism_world_residency --all-targets --no-default-features -- -D warnings
```

These commands verify the CPU crate only. They do not run GPU sampling because
this package intentionally does not own a GPU adapter or backend.
