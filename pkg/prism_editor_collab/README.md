# prism_editor_collab

`prism_editor_collab` is the collaboration layer for Prism editor clients. It
is intentionally independent of gameplay, rendering, ECS, UI, and
engine-specific crates. It contains both the deterministic CRDT core and a
standard-library TCP transport; no Bevy crate is required.

## TCP collaboration

`TcpCollaborationServer` is the authoritative admission point and
`TcpCollaborationClient` is a synchronous client suitable for an editor event
loop. Frames use a versioned, four-byte big-endian length prefix. The server
assigns one monotonically increasing admission order to every new command and
broadcasts that order to all connected clients.

- `ServerConfig::capabilities` maps an actor to allowed domains, operation
  kinds, and payload size. Unknown actors cannot complete the handshake.
- The server validates command limits, capabilities, causal metadata, and the
  authoritative `Replica` before appending a command to its history.
- Server history is bounded by `TransportLimits::max_history_commands`. When
  full, new commands are rejected explicitly; accepted history is never
  silently evicted.
- Each connection has a bounded outbound queue. A slow connection is removed
  from live broadcast when its queue is full. Its client reconnects with the
  last contiguous server order and receives the retained history delta.
- Client submissions remain in a bounded pending queue until the authoritative
  command is observed. Disconnects preserve that queue. A rejected command is
  removed only after an explicit rejection event; transport failures retain it
  for retry.
- Causal predecessors are never dropped to make room. If the configured
  replica or transport bound cannot retain a command, the operation fails
  explicitly and the caller must recover with a snapshot or a larger bound.

The actor identifier is the capability identity supplied by the embedding
application. Applications that expose the listener beyond a trusted process
boundary must authenticate that identity before constructing the server
session.

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
- Command metadata is bounded independently by `ReplicaLimits`: `known`,
  `pending`, and `applied` entries are rejected at their configured limits
  rather than silently evicted, because eviction would weaken causal ordering
  and duplicate detection. A caller can compact by bootstrapping a fresh
  replica from a snapshot.

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

The integration suite includes real loopback TCP coverage for admission order,
causal delivery, capability rejection, unknown-actor handshake rejection,
bounded pending queues, disconnect/reconnect, and history replay.
