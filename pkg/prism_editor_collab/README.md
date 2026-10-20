# prism_editor_collab

`prism_editor_collab` is the optional, transport-neutral collaboration data
layer for Prism editor clients. It is intentionally independent of gameplay,
rendering, ECS, UI, networking, and engine-specific crates.

## Consistency model

- `HlcClock` is caller-driven: physical time is an explicit integer, so tests
  and replay do not read the system clock.
- `CommandEnvelope` carries actor/sequence identity, an HLC timestamp, sorted
  causal command IDs, affected domains, and either a built-in field operation
  or opaque adapter bytes.
- `Replica` deduplicates exact command IDs, rejects conflicting duplicates,
  buffers commands whose causal predecessors are absent, and drains ready work
  in deterministic timestamp/actor/sequence order.
- The document is a field-level LWW CRDT. A delete is a tombstone, so an older
  write cannot resurrect a field. Map keys and list indices can be represented
  by `FieldPath` segments without imposing a schema.
- History is bounded. Once entries are dropped, incremental replay before the
  retention floor returns an explicit error; a snapshot plus retained entries
  remains the recovery path.

## Presence and soft locks

`PresenceRegistry` and `SoftLockRegistry` use explicit caller-supplied time and
exclusive TTLs. Leases contain epochs/tokens, stale heartbeats and releases are
rejected, and expiry is deterministic. Soft locks are advisory coordination
state, not an authorization or conflict-resolution boundary; the field CRDT
remains authoritative for document convergence.

## Validation

The crate is a standalone workspace so it can be tested without changing the
repository root manifest:

```text
cargo fmt --manifest-path pkg/prism_editor_collab/Cargo.toml -- --check
cargo clippy --manifest-path pkg/prism_editor_collab/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path pkg/prism_editor_collab/Cargo.toml --all-targets
```
