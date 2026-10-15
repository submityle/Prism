# prism_world_runtime

`prism_world_runtime` is the engine-neutral coordination slice between the
world stream policy and cell IO. It accepts demand sources, manifests, and
the host's loaded-state snapshot, then submits admitted cells through a
bounded `IoRuntime`. Completed bytes are checked against an explicit manifest
header before they cross the `SceneHost` boundary.

The crate does not parse scenes, create ECS entities, or own renderer state.
`CellProvider` supplies opaque cell bytes and `SceneHost` decides how those
bytes become scene state. `ChunkProvider` is the low-level provider seam used
by `IoRuntime`; a provider may implement it directly or use the blanket
implementation from `CellProvider`.

The header format is fixed and little-endian:

```text
bytes 0..4    magic: PRWC
bytes 4..20   scene_key: u128
bytes 20..28  content_version: u64
bytes 28..44  derived_hash: u128
bytes 44..    opaque scene payload
```

The payload is intentionally opaque. A host may reject it, retain it, decode
it, or forward it to another subsystem.

The coordinator's `no_std` poll layer is independent from an OS executor:
`CellProvider` is the provider boundary and `SceneHost` is the scene boundary.
The separate `prism_world_io::IoScheduler` remains the chunk-level async/file
provider implementation; an application can adapt its completed read events
into `CellProvider` without making world policy depend on a thread runtime or
filesystem.

## Features

- `std` is enabled by default and only enables the `prism_world_io` and
  `prism_world_persist` standard-library features.
- `--no-default-features` builds the coordinator, bounded IO queue, header
  validation, and replayable trace without `std`.

## Verification

```text
cargo fmt --manifest-path pkg/prism_world_runtime/Cargo.toml -- --check
cargo test --manifest-path pkg/prism_world_runtime/Cargo.toml
cargo clippy --manifest-path pkg/prism_world_runtime/Cargo.toml --all-targets --all-features -- -D warnings
cargo check --manifest-path pkg/prism_world_runtime/Cargo.toml --no-default-features
```
