# prism_world_io

`prism_world_io` is an engine-neutral, dependency-free IO/provider crate for Prism world data. The core request, chunk, checksum, provider, and pollable-operation contracts compile without `std`. The default `std` feature adds a bounded priority scheduler and a deterministic file provider.

## Design

- `ReadProvider` creates a `ChunkedRead` operation without referring to an engine, ECS, renderer, or runtime.
- `IoScheduler` orders queued work by descending priority and preserves FIFO order for equal priorities.
- `max_in_flight` limits active provider operations; `capacity` limits all outstanding requests and returns `IoError::Backpressure` instead of growing without bound.
- Cancellation is explicit and observable: the next scheduler poll returns `IoError::Cancelled` and releases the request slot.
- Each chunk is checked for contiguous offsets and request-length bounds. Completion checks both exact byte count and an optional FNV-1a 64-bit checksum.
- `FileProvider` is a pollable `std` implementation. Each poll performs at most one bounded file read, so callers can integrate it with an executor or a manual poll loop without adding an async runtime dependency.

## Features

- Default: `std`, scheduler, and file provider.
- `--no-default-features`: core contracts and checksum remain available for `no_std` targets.
- No external dependencies.
- `unsafe` is forbidden at crate level and in the manifest lint configuration.

## Verification

Run these commands from the repository root:

```text
cargo fmt --manifest-path pkg/prism_world_io/Cargo.toml -- --check
cargo test --manifest-path pkg/prism_world_io/Cargo.toml
cargo clippy --manifest-path pkg/prism_world_io/Cargo.toml --all-targets --all-features -- -D warnings
cargo check --manifest-path pkg/prism_world_io/Cargo.toml --no-default-features
```

The crate is intentionally a standalone workspace so these commands do not require changes to Prism's root `Cargo.toml`.
