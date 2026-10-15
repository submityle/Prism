# Prism World GPU Contract

This standalone crate defines the renderer-facing contract for Tier-C world
proxy visibility. It contains no ECS, scene, asset, or engine dependency.

The CPU contract owns:

- `ProxyInstance` and `Cluster` identity/bounds records;
- `VisibilityInput` frustum and camera constants;
- sphere/frustum and range rejection;
- screen-space geometric error and four-level LOD selection;
- 1/256-pixel error quantisation;
- deterministic structure-of-arrays packing with explicit little-endian words.

The optional `gpu` feature compiles `src/shaders/proxy_visibility.wgsl` with
`wgpu` 30. The kernel reads position/error/metadata SoA columns, executes the
same visibility and LOD policy, writes four `u32` words and one unquantized `f32`
screen-space error per proxy (20-byte stride), and the host
copies those records into a `MAP_READ` staging buffer. The parity test compares
every canonical word and the unquantized error against the CPU golden.
`ClusterSoa` preserves geometric error as float bits in its fourth span word;
its bounds column contains centre and radius.

`ProxyRegistry` adds bounded sector ownership, stable generational handles,
swap-remove dense compaction, and dirty ranges without enabling a GPU backend.
Proxy and sector counts have independent budgets, including empty sectors.
`GpuSectorProxies` couples that registry to fixed `GpuResidentVisibility`
buffers and pre-reserved SoA scratch. Uploads transpose only dirty ranges;
unchanged upload is O(1), while changed upload scans the active N rows and
transposes K dirty rows in O(N + K). Rendering rejects unuploaded edits or a
replaced registry. Production upload/encode never submit, wait, or read back.

`prism_world_runtime/proxy_host` decorates a caller-owned scene host and decoder
with this registry's lifecycle, without enabling wgpu. Load failures compensate
only derived proxy insertion, not partial authoritative-host changes. See
`docs/prism_world_sector_proxies_zh.md` for budgets, call ordering, failure
semantics, real file/coordinator-to-indirect-pixel evidence, and CPU benchmarks.

## Verification

CPU-only contract and packing tests:

```text
cargo test --manifest-path pkg/prism_world_gpu/Cargo.toml
cargo clippy --manifest-path pkg/prism_world_gpu/Cargo.toml --all-targets -- -D warnings
```

WGSL parse/validation and real backend tests:

```text
cargo test --manifest-path pkg/prism_world_gpu/Cargo.toml --features gpu --test wgsl_compile
cargo test --manifest-path pkg/prism_world_gpu/Cargo.toml --features gpu --test gpu_parity -- --nocapture
cargo test --manifest-path pkg/prism_world_gpu/Cargo.toml --features gpu --test resident_visibility -- --nocapture
```

The parity test is a strict real-device gate: missing adapter/device acquisition
always fails, independent of environment variables. With an adapter, the test
logs one stable telemetry record containing input proxy count, upload bytes,
readback bytes, dispatch workgroups, fixed-unit wall-clock dispatch/readback
timings, and backend/device/driver evidence. `GpuVisibility::evaluate_with_telemetry`
returns the same telemetry alongside the canonical output records.

The pipeline requires no optional device feature. The test records the actual
adapter metadata and uses the device's native WGSL validation path; a passing
GPU result therefore proves shader compilation, pipeline creation, dispatch,
and readback on that adapter. Upload bytes cover the parameter block and the
three proxy SoA columns (48 bytes per proxy plus 128 parameter bytes); readback
bytes cover the 20-byte output record per proxy. `WGPU_BACKEND` is honored and
checked by the real-device test, which rejects CPU adapters. Tests cover all LOD
thresholds, error saturation, a 129-proxy multi-workgroup tail, and empty inputs.
Timings are host wall-clock measurements around queue completion and
staging-buffer map/readback, not GPU timestamp-query measurements.
