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
Under `std`, `IoSchedulerAdapter` is the concrete bridge to
`prism_world_io::IoScheduler`, `ReadProvider`, and `ChunkedRead`: it submits
`CellReadRequest` values, exposes chunk/completion events, forwards cancellation,
backpressure, and provider errors, and can serve as the coordinator's
`ChunkProvider`. The adapter's direct `poll` method preserves the asynchronous
boundary; its `ChunkProvider` implementation is a blocking compatibility path
for world coordination. Provider and validation errors (including cancellation)
are `IoAdapterEvent::Failed { request_id, request, error }`; `poll` returns `Err`
only for an internal adapter failure. Concurrent blocking readers reserve their
own events and share scheduler wake notifications. Unrelated events remain
available to public polling, and terminal events retain queue capacity until
consumed, so a blocked consumer cannot bypass backpressure.

## File-backed cells

`FileCellProvider` reuses `prism_world_io::FileProvider` and the adapter without
depending on `prism_asset`. Register each complete manifest with its byte offset
in a file/packed archive and, optionally, the FNV-1a checksum of the entire cell
(header plus payload). The coordinator's manifest-only requests resolve through
these registrations; missing or stale registrations fail before file IO.

```rust
use prism_world_runtime::{CellReadRequest, FileCellProvider};
use prism_world_stream::CellManifest;

fn file_cells(path: &str, manifest: CellManifest, offset: u64, checksum: u64)
    -> FileCellProvider
{
    let mut provider = FileCellProvider::with_chunk_size(path, 2, 8, 4096)
        .expect("nonzero scheduler limits");
    provider.register(CellReadRequest::new(manifest, 0)
        .with_byte_offset(offset)
        .with_expected_checksum(checksum))
        .expect("non-overflowing file range");
    provider
}
```

Pass this provider to `WorldFrameCoordinator::new`; only validated payload bytes
cross `SceneHost::load_cell`. For explicit chunk polling, use `resolve` and submit
the returned request to `scheduler()`. Re-register changed manifests explicitly.
The existing file provider opens one file per operation and performs bounded
**blocking** filesystem reads during polling; this is not worker-thread async IO.
Capacity bounds outstanding requests, not total payload bytes; callers must also
bound manifest sizes/stream budgets. Chunk events and collected completion bytes
are both retained when another reader drives a request ahead of its consumer.

## Features

- `std` is enabled by default and only enables the `prism_world_io` and
  `prism_world_persist` standard-library features.
- `--no-default-features` builds the coordinator, bounded IO queue, header
  validation, and replayable trace without `std`.

## Verification

```text
cargo fmt --manifest-path pkg/prism_world_runtime/Cargo.toml -- --check
cargo test --manifest-path pkg/prism_world_runtime/Cargo.toml
cargo test --manifest-path pkg/prism_world_runtime/Cargo.toml --no-default-features
cargo check --manifest-path pkg/prism_world_runtime/Cargo.toml --all-targets --all-features
cargo clippy --manifest-path pkg/prism_world_runtime/Cargo.toml --all-targets --all-features -- -D warnings
cargo clippy --manifest-path pkg/prism_world_runtime/Cargo.toml --all-targets --no-default-features -- -D warnings
cargo check --manifest-path pkg/prism_world_runtime/Cargo.toml --no-default-features
```

Tests create real temporary files and cover packed ranges with split headers,
exact payload delivery to `SceneHost`, EOF/short reads, checksums, open errors,
invalid headers, stale registrations, request-associated cancellation/errors,
buffered-event backpressure, and simultaneous pending readers sharing one waker.
