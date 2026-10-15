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
same visibility and LOD policy, writes four `u32` words per proxy, and the host
copies those records into a `MAP_READ` staging buffer. The parity test compares
every readback word against the CPU golden.

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
```

Without `PRISM_REQUIRE_GPU`, the parity test returns successfully when no native
adapter is available. When `PRISM_REQUIRE_GPU` exists (including `0`), missing
adapter/device acquisition is a test failure. With an adapter, the test logs
backend, device type, vendor, device, driver, and driver-info evidence, creates
the compute pipeline, dispatches the kernel, waits for the queue, maps the
readback buffer, and compares all four words for every proxy.

The pipeline requires no optional device feature. The test records the actual
adapter metadata and uses the device's native WGSL validation path; a passing
GPU result therefore proves shader compilation, pipeline creation, dispatch,
and readback on that adapter. Throughput, frame-time, and memory figures remain
benchmark measurements rather than claims made by this contract test.
