# `prism_world_residency`

This crate owns deterministic, engine-neutral residency primitives. Its sparse
SDF slice is CPU-only: `SparseSdfBrickGrid::export_allocated_bricks` produces a
bounded packed staging payload for a future GPU adapter, but it does not acquire
an adapter, allocate a device buffer, dispatch a shader, or claim GPU sampling
parity.

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
