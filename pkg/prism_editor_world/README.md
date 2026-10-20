# prism_editor_world

`prism_editor_world` is the engine-neutral executable data layer for large-world editing. It owns deterministic cell metadata and residency transitions, hierarchical data-layer visibility/editability policy, cell edit locks, opaque payload residency, OFPA actor-file planning, and HLOD proxy visibility planning.

The crate does not define or parse a cell/actor file format. `CellIoProvider` is the storage boundary: providers return owned `Future`s for load/save operations, while `WorldIoDriver` enforces request-count and reserved-byte budgets and exposes explicit cancellation. Startup must be nonblocking; providers must honor read limits before allocation, cancel work when their future is dropped, and publish saves atomically before reporting success. The driver validates payload size but cannot police an adapter's internal allocations or storage side effects.

`EditorWorldData::begin_residency(demand, now)` starts tokenized transitions. Submit each load through `EditorWorldData::submit_load`: admission/startup failure restores `Unloaded`; resubmitting an already accepted load does not invalidate it. Drain terminal events and apply them through the matching load/save completion method. Cancelled/failed loads restore `Unloaded`; a failed unload retains bytes in `Loaded`. Clean eviction needs no storage operation. Before dropping a driver, cancel/drain and reconcile all its requests; dropping it alone does not mutate the separate world model.

Each data layer may name one parent. A cell assigned to a child layer is visible and editable only when the child and every ancestor permit that policy; unknown parents and parent cycles are rejected before model mutation. Visibility and editability are independent of each other and of residency. Payload edits also require the active edit scope and a live `CellLockToken`. Both dirty cells and live-locked cells are protected against eviction. Lease time is a caller-supplied monotonic clock with exclusive expiry.

A save completion clears dirty state only if its request matches the submitted snapshot, the authorizing lock remains valid, the per-cell revision is unchanged, the bytes still match, and no dirty actor writes remain. Byte-identical edits still advance revisions. OFPA plans are sorted by stable actor identity and carry opaque relative path keys and actor revisions; failed writes do not consume plan state. Commit each successful actor write under its current cell lock, then persist/acknowledge the cell snapshot. The provider remains responsible for the actual actor write format; OFPA is planning, not an actor codec.

HLOD declarations use explicit half-open Chebyshev distance bands in cell units. The stateful planner suppresses a proxy whenever any source cell is resident, commits its preview state when `plan` is called, and emits deterministic show/hide differences with hides before shows per cluster. It does not bake or render proxy assets. Cell payload, actor and HLOD capacities and I/O admission budgets are explicit; the crate has no renderer/device dependency. `model()` exposes immutable planning/diagnostic state; use the facade for lock-protected execution.

Integration boundary: this crate ships no production disk/network provider, cell/actor codec, cross-file transaction journal, or crash-recovery storage backend. Recovery here restores in-process lifecycle state; durability and multi-file consistency belong to the provider/application. The in-memory provider is test-only. HLOD plans do not load proxies or acknowledge renderer success, and OFPA plans do not execute actor writes. No editor UI, viewport binding, runtime entity instantiation, or render integration is included.

Verification from the repository root:

```sh
export CARGO_TARGET_DIR="$(mktemp -d /private/tmp/prism-editor-world-XXXXXX)"
export CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0
cargo fmt --manifest-path pkg/prism_editor_world/Cargo.toml --package prism_editor_world -- --check
cargo clippy --manifest-path pkg/prism_editor_world/Cargo.toml --all-targets --no-deps -- -D warnings
cargo test --manifest-path pkg/prism_editor_world/Cargo.toml --all-targets
cargo test --manifest-path pkg/prism_editor_world/Cargo.toml --doc
```

`--no-deps` keeps strict lint enforcement scoped to this crate, including its test crates. Remove it for a strict dependency audit. Use a unique target while other checkout verification runs are active; never clean another task's target directory. Tests include a bounded in-memory provider exercising `Pending`/wake/load/save/reload without choosing a disk format, plus failure, cancellation, stale completion, lock expiry, hierarchy, OFPA and HLOD regressions.
