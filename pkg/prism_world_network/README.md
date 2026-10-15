# prism_world_network

`prism_world_network` is the M8 world-to-network seam. It is a `no_std +
alloc` crate with no socket, serializer, ECS, scene, or filesystem dependency.
It includes a small reliable session seam and a deterministic in-memory
transport for host integration tests.

## Single-source-of-truth boundary

- `prism_world_stream` owns spatial identity and world-space bounds. This crate
  reuses its `SectorId`, `GlobalPos`, and `AabbI64` types rather than defining
  network copies.
- `prism_world_persist` owns durable object identity, field deltas, transaction
  ordering, and field-level LWW arbitration. This crate reuses its
  `PersistentId`, `Transaction`, and CRDT types rather than creating a second
  persistence model.
- This crate owns the boundary policy: inclusive AOI selection, deterministic
  delta envelopes, publish permissions, sector handoff records, an
  object-indexed merge facade, and the transport-independent session contract.
- Merge duplicate detection is indexed by `TransactionId`, so replaying a
  large batch does not scan every object bucket linearly.

The world remains authoritative for spatial placement and persistence remains
authoritative for object state. An envelope carries a validated persistence
transaction plus its world sector; it does not become a second source of scene
truth. A handoff changes the placement/authority ledger only after the current
authority accepts it. A merge state delegates field conflict resolution to
`prism_world_persist::CrdtState` and indexes the resulting transactions by
`PersistentId`.

AOI bounds use inclusive integer coordinates on all three axes. Therefore both
the minimum and maximum corners are inside the subscription.

## Session and transport seam

`Session` is an authenticated, bounded, ordered delivery state machine. The
host supplies a non-blocking `Transport` implementation; no network library is
required. A host must authenticate the opaque `AuthToken` before data can be
queued or accepted. The token type is deliberately only a comparison seam, so
cryptographic verification remains the responsibility of the host control
plane.

- `SessionConfig::outbound_capacity` bounds unacknowledged data packets;
  `inbound_capacity` bounds retained out-of-order packets.
- `Session::send` assigns sequence numbers starting at one and validates
  outgoing `DeltaEnvelope` and `AoiSubscription` against `AuthorityPolicy` and
  the local actor.
- `Session::flush` emits cumulative acknowledgements and data packets without
  blocking. Transport backpressure does not consume a retry attempt.
- `retry_after_ticks` and `max_attempts` bound retransmission. Exhaustion
  moves the session to `SessionState::Failed`.
- `Session::receive` buffers out-of-order packets, delivers contiguous packets
  in sequence order, cumulatively acknowledges them, and suppresses duplicates.
- `MemoryTransport::pair` is a bounded no-std adapter used by integration
  tests and deterministic host harnesses; production hosts can implement
  `Transport` over their own authenticated channel.

The transport packet carries a `SessionId`, while token verification is a
session-establishment contract rather than a wire encryption scheme. Hosts
that need confidentiality or tamper resistance must bind this seam to an
authenticated/encrypted transport.
