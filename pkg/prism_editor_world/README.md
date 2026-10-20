# prism_editor_world

`prism_editor_world` is the engine-neutral executable data layer for large-world editing. It owns deterministic cell metadata and residency transitions, hierarchical data-layer visibility/editability policy, cell edit locks, opaque payload residency, OFPA actor-file persistence, and HLOD proxy visibility planning.

`CellIoProvider` is the storage boundary: providers return owned `Future`s for load/save operations, while `WorldIoDriver` enforces request-count and reserved-byte budgets and exposes explicit cancellation. `FileCellIoProvider` implements actual versioned disk persistence of opaque cell payloads. It does not interpret scene components or actor files. The driver validates payload size but cannot police a custom adapter's internal allocations or storage side effects.

`EditorWorldData::begin_residency(demand, now)` starts tokenized transitions. Submit each load through `EditorWorldData::submit_load`: admission/startup failure restores `Unloaded`; resubmitting an already accepted load does not invalidate it. Drain terminal events and apply them through the matching load/save completion method. Cancelled/failed loads restore `Unloaded`; a failed unload retains bytes in `Loaded`. Clean eviction needs no storage operation. Before dropping a driver, cancel/drain and reconcile all its requests; dropping it alone does not mutate the separate world model.

Each data layer may name one parent. A cell assigned to a child layer is visible and editable only when the child and every ancestor permit that policy; unknown parents and parent cycles are rejected before model mutation. Visibility and editability are independent of each other and of residency. Payload edits also require the active edit scope and a live `CellLockToken`. Both dirty cells and live-locked cells are protected against eviction. Lease time is a caller-supplied monotonic clock with exclusive expiry.

A save completion clears dirty state only if its request matches the submitted snapshot, the authorizing lock remains valid, the per-cell revision is unchanged, the bytes still match, and no dirty actor writes remain. Byte-identical edits still advance revisions. OFPA plans are sorted by stable actor identity; planning does not perform I/O or consume dirty state. The world facade requires a real actor save receipt before acknowledging each actor, then a separate cell save completion. Standalone `OfpaCatalog::commit` remains an external metadata acknowledgement only.

HLOD declarations use explicit half-open Chebyshev distance bands in cell units. The stateful planner suppresses a proxy whenever any source cell is resident, commits its preview state when `plan` is called, and emits deterministic show/hide differences with hides before shows per cluster. It does not bake or render proxy assets. Cell payload, actor and HLOD capacities and I/O admission budgets are explicit; the crate has no renderer/device dependency. `model()` exposes immutable planning/diagnostic state; use the facade for lock-protected execution.

## Desktop storage integration

Create/select an existing, application-owned directory **per world**, then construct:

```rust,ignore
let provider = FileCellIoProvider::new(
    world_directory,
    FileCellIoLimits::new(8 * 1024 * 1024, 4)?,
)?;
let budget = IoBudget::new(4, 32 * 1024 * 1024).expect("nonzero constant budget");
let mut driver = WorldIoDriver::new(provider, budget);
```

Only construction resolves the root synchronously. Submission and polling do no filesystem I/O; a bounded worker per accepted operation performs it and wakes the caller's waker. Provider clones share operation capacity, including cancelled workers still cleaning up and completed futures not yet dropped. Independent instances/processes coordinate through OS advisory file locks. `Busy` and `Capacity` are retryable, not reasons to discard editor data. The desktop loop must schedule another poll from the registered waker (or its existing frame loop).

Load: `begin_residency` supplies a load token; call `world.submit_load(&mut driver, token)`, then `world.apply_load_completion(completion)`. Save: `world.prepare_save(cell, &lease, now)` captures a `PendingCellSave`; submit its `request().clone()` and `payload().to_vec()` through `driver.submit_save`, then use `world.apply_save_completion(&pending, completion, now)`. Keep the pending snapshot until terminal reconciliation. Initially authored cells can use `complete_load(token, initial_bytes)` before lock-protected edits and saving. Missing files are errors, never implicit empty worlds.

`load_snapshot` adds `CellSnapshot { bytes, revision }`; `WorldIoDriver` uses it automatically. Existing providers implementing only `load_cell` retain a default unversioned adapter. `IoCompletion::Loaded` now includes `revision: Option<u64>`; exact field matches/construction must account for it. Successful filesystem loads restore the durable revision into the model, so a restarted editor does not restart its counter at zero. `prepare_save` adds the last observed durable base via `CellSaveRequest::with_base_revision`. Under the disk lock, save rejects a mismatched base, lower revisions, and equal revisions with different bytes. Exact same revision/bytes is an idempotent retry and re-syncs the directory. Low-level requests without a base enforce monotonic revisions but cannot identify a divergent editor that has made more edits; use the facade for optimistic concurrency. A changed-on-disk conflict needs an explicit reload/merge decision; there is no automatic force overwrite.

Successful old snapshot completions update the observed disk base even when a later edit or expired lease prevents clearing dirty state. `Saved` alone is never authority to mark the editor clean: apply the facade's revision, payload, OFPA and lease checks. Disk locks and authoring leases are separate: workers do not receive or extend editor leases.

## Real OFPA actor storage

`FileActorIoProvider` uses the same envelope codec, cancellation fence, directory checks, OS locks and durable replacement implementation as cells. Actor files have distinct magic `PEWACTR` followed by NUL, the same version 1 header below, and the actor GUID in the identity field. Their catalog-compatible numeric paths are `cells/x_y_z/actors/<decimal-u128-guid>`, with persistent `.lock` and recoverable `.pending` sidecars. Nil GUIDs are rejected. All signed i32 coordinates, including extrema, are valid; decoded coordinates and GUID must exactly match the requested identity. Payload bytes have no component, gameplay or engine schema.

Construct the provider with `ActorIoLimits::new(max_payload_bytes, max_requests, max_reserved_bytes)`. Clones share both admission counters; independent instances have independent budgets but share OS locks. Loads reserve their requested cap before worker admission; saves reserve their immutable `Arc<[u8]>` length without copying. Reservations survive dropped futures while workers clean up, and completed results waiting in unpolled futures. Results already returned to the caller and retained pending snapshots/receipts are caller-owned memory outside this budget. Header, fixed 64 KiB worker scratch, thread stacks and metadata are additional overhead. The world bounds the sum of resident actor payload bytes in each cell by `EditorWorldLimits::max_cell_bytes()`, independently of its existing cell-payload bound. Thus resident payload storage can use twice that limit per cell, plus metadata and externally retained snapshots.

Use `replace_actor_payload` under the live cell lease to advance both actor and cell revisions. `prepare_actor_save(actor, &lease, now, &provider)` captures immutable bytes, actor and cell revisions, residency generation, canonical root and lease. Await `provider.save_actor(pending.request().clone(), pending.payload(), timeout)` and pass its receipt to `world.commit_ofpa(&pending, &receipt, now)`. The receipt must match the exact request, bytes and root; stale actor/cell edits, eviction/reload, changed policy or expired leases prevent acknowledgement. A matched durable receipt still advances the observed actor disk base when it cannot clear dirty state, allowing the next explicit retry. Actor success leaves the cell dirty; save the cell separately after all dirty actors are acknowledged. The former world metadata-only `commit_ofpa(operation, lease, now)` API is replaced by this receipt API.

For existing files, register the actor in the project-supplied catalog, then call `prepare_actor_load`, await `load_actor(identity, max_bytes, timeout)`, and reconcile with `apply_actor_load`. Dirty actors cannot be loaded over. Stale cell/actor revisions, residency, root or lease reject the snapshot without mutation. A successful load restores the durable revision/base and opaque bytes. Unloading the cell drops resident actor bytes. This provider does not discover a catalog or infer actor membership from payloads.

Every actor save explicitly chooses `ActorRevisionBase::Missing` (create-only) or `Revision(base)` (compare-and-save). Under the OS lock, divergent or older snapshots and mismatched bases fail. An already present identical revision and exact bytes is an idempotent retry even if the original base is now old; it re-syncs the directory before producing a receipt. Corrupt finals are preserved and block writes. Revisions are caller-authored counters, not timestamps or globally ordered transaction IDs.

Timeouts are nonzero `Duration`s converted to monotonic deadlines at submission. Checks run before work, between bounded chunks and before the publication fence. `TimedOut` leaves the previous final untouched. Blocking kernel calls are not interruptible; a rename inside the commit fence finishes with its directory sync even if the deadline passes. Drop cancels with the same publication fence as cells; wait for both `outstanding_operations()` and `reserved_bytes()` to reach zero before removing the root.

Atomicity is per actor or per cell file. An actor commit followed by a cell commit can crash between files and leave mixed revisions. Cross-cell/OFPA coordinated transactions, a shared journal, transaction replay and catalog discovery remain required integration work. Sequential saves and metadata plans provide no such guarantee. See `docs/prism_editor_world_ofpa_zh.md` for the exact boundary and verification evidence.

## Layout and stable version 1 envelope

The provider reuses `prism_scene::FileCellStore::cell_path` under `<world>/cells`, the scene envelope size cap, and its write/fsync/rename/directory-fsync durability contract. It does not call the scene's private PSCL codec: that codec serializes ECS entity/member/relation semantics, whereas editor payloads are application-defined. Editor envelopes are deliberately distinguishable and are **not** PSCL archives. Keep runtime PSCL storage in a different root, even when an editor payload itself contains scene bytes. No new dependency or Bevy crate is introduced.

All integers are little endian. No native struct layout, timestamps, compression, or platform-size fields enter the format. The complete file is exactly a 64-byte header followed by the payload:

| Offset | Bytes | Field |
| --- | --- | --- |
| 0 | 8 | ASCII `PEWCELL` followed by NUL |
| 8 | 4 | Version `1` as u32 |
| 12 | 12 | Signed i32 coordinates x, y, z |
| 24 | 16 | Stable cell GUID as u128 |
| 40 | 8 | Authoring revision as u64 |
| 48 | 8 | Payload byte length as u64 |
| 56 | 8 | FNV-1a-64 of header bytes 0..56, then payload |
| 64 | declared length | Opaque application bytes, including empty payloads |

FNV uses offset `0xcbf29ce484222325` and wrapping prime `0x100000001b3`. This detects accidental corruption, not malicious tampering/authentication. Unknown versions, coordinate/GUID mismatch, truncated/trailing bytes, length overflow and checksum mismatch fail closed. Stat and the fixed header are checked before payload allocation; allocation is fallible and bounded by the smaller of request/provider limits. Streaming reads also check exact EOF and checksum. Comparing an existing save uses a fixed 64 KiB buffer, not a second payload allocation. Maximum configurable payload is `prism_scene::partition::MAX_CELL_BYTES - 64` (64 MiB envelope cap). Align `EditorWorldLimits` and driver budgets with this bound; its generic 64 MiB payload default is larger by the envelope overhead.

Only numeric coordinates generate paths (`cells/x_y_z.prismcell`); arbitrary request paths are never joined. Identity/path inconsistency, `..` in the configured root, existing symlinks below the canonical root, directories used as files, and special files are rejected. The explicitly chosen root may resolve through a symlink once at construction. The root, its ancestors and contents must be application-owned: portable `std` metadata checks are **not** a capability sandbox against hostile concurrent directory/inode substitution, and non-cooperating writers can bypass advisory locks. A read may create a small lock sidecar, so this is a read-write workspace provider, not a read-only archive browser.

## Commit, cancellation and recovery

Each cell has a persistent `.prismcell.lock` sidecar. Its inode is never unlinked/truncated; exclusive writers and shared readers use nonblocking OS locks. Process death releases locks without stale lock-file deletion. The writer holds the lock across validation and commit. It creates its same-directory `.prismcell.pending` stage with `create_new`, writes/checks cancellation in bounded chunks, fsyncs the file, atomically renames it over the final file, then fsyncs the directory. Creation of `cells` is also persisted by syncing the world directory. Existing unsupported directory-sync/locking behavior fails explicitly; this implementation is validated on the current macOS filesystem, not every OS/network filesystem.

Before rename, any error/cancellation leaves the old final untouched; RAII removes the owned stage. A crash can leave a stage; only the final filename is ever loaded, and a later non-idempotent save reclaims that exact regular-file stage under the cell lock. It never scans/deletes unrelated temporary files. If the old final is corrupt, save refuses to replace it; preserve it for diagnosis, restore a known-good file externally, or explicitly choose a fresh world root. This provider is not an automatic backup/repair system.

`CommitUncertain` means rename succeeded but directory fsync failed: old or new complete data may survive power loss. Keep the dirty payload and retry the **same snapshot** to settle durability. Do not acknowledge, evict, or silently overwrite newer disk data. Tests inject both pre-rename and post-rename failures on actual temporary files; they do not simulate physical power loss.

Dropping a future (including `driver.cancel`) marks its worker cancelled and fences rename. No later publication can happen after cancellation returns. A commit already inside that fence finishes before cancellation returns; cancellation therefore cannot promise the disk was unchanged. The drop may briefly wait for that final rename/directory fsync; it does not join the worker, which can finish stage cleanup afterwards. `outstanding_operations()` exposes that cleanup lifetime. Keep/retry the original snapshot or reload to reconcile an already-published cancelled save before choosing a new base for subsequent edits. Failed/cancelled completions never clear dirty state. On shutdown cancel/drain/reconcile before dropping the independent model, and keep the workspace directory alive until provider operations settle.

Integration boundary: durability/atomicity is per actor/cell file, not a cross-cell/OFPA transaction journal. This work supplies no network provider, backup history, world UI, viewport, runtime instantiation or renderer integration. HLOD plans do not load proxies or acknowledge renderer success. This storage slice does not complete the whole editor system.

Verification from the repository root:

```sh
export CARGO_TARGET_DIR="$(mktemp -d /private/tmp/prism-editor-world-XXXXXX)"
export CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0
export CARGO_BUILD_JOBS=2
cargo fmt --manifest-path pkg/prism_editor_world/Cargo.toml --package prism_editor_world -- --check
cargo clippy --locked --offline --manifest-path pkg/prism_editor_world/Cargo.toml --all-targets --no-deps -- -D warnings
cargo test --locked --offline --manifest-path pkg/prism_editor_world/Cargo.toml --all-targets
cargo test --locked --offline --manifest-path pkg/prism_editor_world/Cargo.toml --doc
cargo clean --target-dir "$CARGO_TARGET_DIR"
```

`--no-deps` keeps strict lint enforcement scoped to this crate, including its test crates. Remove it for a strict dependency audit. Use a unique target while other checkout verification runs are active; never clean another task's target directory. The filesystem acceptance tests use private real temporary directories, reopen providers, binary/empty/negative/extreme-coordinate roundtrips, malformed envelopes, sparse oversized files, symlink refusal, revision/base conflicts, concurrent providers and actual subprocess lock/death recovery. Private test hooks exercise cancellation after staging and recoverable pre/post-rename failures. Existing bounded in-memory tests and hierarchy, OFPA and HLOD regressions remain in place.
