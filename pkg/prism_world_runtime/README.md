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
use prism_world_runtime::{CellReadRequest, FileCellProvider, IoByteLimits};
use prism_world_stream::CellManifest;

fn file_cells(path: &str, manifest: CellManifest, offset: u64, checksum: u64)
    -> FileCellProvider
{
    let mut provider = FileCellProvider::with_byte_limits(path, 2, 8, 4096,
        IoByteLimits {
            max_request_bytes: 16 * 1024 * 1024,
            max_total_bytes: 64 * 1024 * 1024,
        })
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
The request-count capacity and byte admission limits are separate controls.

## Byte admission and memory scope

`IoSchedulerAdapter::with_byte_limits` and `FileCellProvider::with_byte_limits`
accept `IoByteLimits { max_request_bytes, max_total_bytes }`. Both limits use
the full declared `manifest.io_bytes`, including the 44-byte header, not just
the scene payload. `byte_limits()` returns the original limits, and
`admitted_bytes()` reports current declared-byte admission. Source registration
alone does not reserve bytes. The legacy `new` and `with_chunk_size` signatures
remain available without configured byte limits (`byte_limits() == None`).

Submission checks request-count capacity, the single-request byte limit, checked
`u64` addition, and the cumulative byte limit before assigning a scheduler ID or
starting IO. Byte rejection is an `IoError::Provider` diagnostic with the
requested/limit values (or the overflowing operands); request-count rejection
is `IoError::Backpressure`. Rejections consume neither quota nor IDs. Limits are
never silently clamped: zero permits only empty declarations, and a single limit
larger than the total is preserved. Even legacy constructors reject arithmetic
overflow. Scheduler range-validation rejection also consumes no admission.

Queued requests, active operations, and **unconsumed terminal results** all hold
their full declared bytes and request slot. EOF, checksum/provider failure,
cancellation, and completion do not release quota until the corresponding
`Failed`/`Completed` event is consumed. Consuming a `Chunk` does not release it.
A blocking read consumes its own terminal result before returning.

Aggregation lazily uses `try_reserve_exact` for the full declaration on the first
chunk, avoiding geometric growth and repeated aggregate copying; further appends
also use fallible reservation. Capacity/allocator errors cancel the underlying
operation, wake the poller, and wait for scheduler removal/drop before delivering
`Failed { request_id, request, error }` with the original allocation diagnostic.
They do not turn into a second cancellation event. This recovery is limited to
aggregate reservation, not arbitrary allocator failure throughout the process.
Provider cancellation IDs are normalized at both start and poll boundaries;
scheduler cancellations use their explicit request ID, not the latest provider
callback's ID. Allocation cleanup preserves this identity and replaces only the
terminal error cause with the original allocation diagnostic.

For a configured cumulative limit `B`, the sum of admitted declarations is at
most `B`. Aggregate reservation targets sum to at most `B`, but this is not an
exact allocator-capacity or RSS ceiling. In particular:

- Ready `Chunk` events retain independent byte copies while the aggregate is
  built. `Completed` takes ownership of the aggregate without another clone,
  but may coexist with all those chunks. Retained logical chunk lengths plus
  aggregate/completed byte lengths can approach **2B**, not B.
- Actual `Vec` capacities, allocator rounding/metadata, provider-owned buffers
  and operations, temporary chunks, and scheduler/event/map/waker storage are
  not bounded by this byte counter. Empty chunks can add event metadata without
  adding payload length. Registration storage is also outside admission.
- Once returned to the caller, chunk/completion buffers and any `SceneHost`
  copies are outside admission lifetime; callers must manage their retention.

Thus byte admission bounds declared work and logical retained payload, not all
allocation overhead. The provider and consumer still need their own memory and
retention policies; an arbitrary provider can allocate excess backing capacity.

## Features

- `std` is enabled by default and only enables the `prism_world_io` and
  `prism_world_persist` standard-library features.
- `--no-default-features` builds the coordinator, bounded IO queue, header
  validation, and replayable trace without `std`.

## Persistent scatter deletions

The runtime also bridges regenerable CPU scatter to scene-owned override fields,
without depending on an ECS or renderer. `ScatterDeletionBinding { object, field }`
identifies the author scatter node and the caller's schema field. A candidate is
identified by its complete `ScatterCandidateKey`, never a dense index or a hash
truncated to `PersistentId`. `ScatterDeletionSet` is provided by
`prism_world_residency`; its budgets count actual Rust key Vec capacity, while
`ScatterDeletionCodecBudget` separately limits canonical encoded bytes.

Read a binding from `Timeline::state()`, edit its immutable set with
`with_deleted(key, true, budget.deletions)`, and pass
`binding.delta(&updated, budget.max_encoded_bytes)` to `Timeline::commit` with
`binding.object`. The timeline captures the real prior value for undo. Read the
resulting snapshot and call `filter_source` or `filter_field` on a regenerated
buffer. The returned wrapper retains the complete snapshot, including node and
field; compare it in addition to the upstream source cache key. Filtering is
stable, in-place, and does not move/refill candidates or shrink capacities.

The complete executable example creates its own temporary journal, persists a
full-key deletion, closes/reopens/replays, regenerates points, and checks durable
undo/redo. It cleans up only the temporary directory it created:

```text
cargo run --offline -p prism_world_runtime --example scatter_deletions
cargo bench --offline -p prism_world_runtime --bench scatter_deletions
```

The example stages a cloned timeline, appends the transaction durably, then
publishes the new timeline. This demonstrates a failure-safe publication policy,
not an efficient incremental transaction implementation. `Timeline::commit`
does not write a `FileJournal`; the two operations are not automatically atomic.
An append error may poison the journal: stop publishing and reopen/recover before
deciding how to retry. History/inverse fields contain full snapshots and their
cumulative storage is outside the single-field codec budget.
The example retains its in-process undo stack while reopening the journal;
replay restores `WorldState`, not an editable undo history after restart.

The version-1 `PSDL` encoding uses a 16-byte header and 52 bytes per full key,
strict numeric structural order, and checked length/count/capacity limits.
Malformed fields return an error, never silently clear deletions. Empty deltas
remove the field; standalone empty encoding still needs a 16-byte header.
The field uses existing whole-field LWW, **not per-key concurrent union**: racing
snapshots can lose another writer's deletion. Recipe identity changes require
explicit migration. GPU deletion, heightfield wrappers, and product integration
are separate work. Full contracts and measured CPU percentiles are recorded in
`docs/prism_world_scatter_deletions_zh.md`.

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
Byte-budget regressions cover exact/zero/independent limits, sum/range overflow,
rejection without ID allocation, unconsumed success/error/cancellation quota,
fallible aggregate allocation with counted operation cancellation/drop, blocking
reader wake-up, concurrent byte admission/release, and header-inclusive bounded
file-to-`SceneHost` loading.
