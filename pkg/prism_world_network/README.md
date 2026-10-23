# prism_world_network

`prism_world_network` is the M8 world-to-network seam. It is a `no_std +
alloc` crate with no socket, serializer, ECS, scene, or filesystem dependency.
It includes a small reliable session seam, a deterministic in-memory transport
for host integration tests, and the bounded M7 server-mesh authority-handoff
contract.

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
- The session integration regression encodes a canonical `AppendOnlyLog` with
  the persistence wire codec, decodes it, sends its transaction as a sector
  `DeltaEnvelope`, validates authority at both session edges, merges the
  delivered transaction through the persistence CRDT facade, and completes the
  cumulative-ACK lifecycle. This is codec/session/merge integration coverage;
  it is not a disk-journal, socket, or encrypted transport test.
- `encode_packet`/`decode_packet` provide the stable `PWN1` little-endian byte
  frame for host transports. Delta payloads reuse the versioned persist
  transaction codec and retain the full `SectorId`; subscriptions retain signed
  inclusive AOI bounds. Header version, packet sequence, message kind, payload
  length, and all integer fields are checked before publication, with a fixed
  16 MiB payload cap. ACK frames are represented explicitly. This is a
  serialization boundary, not encryption, key exchange, socket I/O, or packet
  retry; the `Session` state machine remains responsible for the latter.
- `ServerMesh` owns bounded partition-authority metadata only. It fences old
  authorities with `(authority, epoch, lease)` matching, reserves target
  capacity during prepare, and rechecks the source fence during commit.
- `HandoffTransactionId` is retained for the configured transaction bound, so
  exact command replay stays idempotent for the lifetime of that mesh state;
  conflicting reuse of one id is rejected.

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
  `inbound_capacity` bounds retained out-of-order packets. The receiver also
  rejects future sequences farther than this distance from its next expected
  sequence before they consume reorder storage, preventing a forged far-future
  packet from pinning the receive cursor.
- `Session::send` assigns sequence numbers starting at one and validates
  outgoing `DeltaEnvelope` and `AoiSubscription` against `AuthorityPolicy` and
  the local actor.
- `Session::flush` emits cumulative acknowledgements and data packets without
  blocking. Transport backpressure does not consume a retry attempt.
- `retry_after_ticks` and `max_attempts` bound retransmission. Exhaustion
  moves the session to `SessionState::Failed`.
- `Session::receive` buffers out-of-order packets, delivers contiguous packets
  in sequence order, cumulatively acknowledges them, and suppresses duplicates.
  Reorder state is bounded by both packet count and sequence distance.
- `MemoryTransport::pair` is a bounded no-std adapter used by integration
  tests and deterministic host harnesses; production hosts can implement
  `Transport` over their own authenticated channel.

The transport packet carries a `SessionId`, while token verification is a
session-establishment contract rather than a wire encryption scheme. Hosts
that need confidentiality or tamper resistance must bind this seam to an
authenticated/encrypted transport.

## Server mesh and authority handoff seam

`ServerMesh` maps the existing `prism_world_stream::SectorId` partition identity
to an `AuthorityLease`. A lease has an opaque lease id, a monotonically
increasing `AuthorityEpoch`, and an exclusive logical expiry tick. The mesh
does not choose a clock or a server-discovery policy; the host supplies both
the tick and the authority identities.

Handoffs use a bounded prepare/commit/abort transaction:

- `prepare_handoff` validates the current source authority, requires the target
  epoch to be exactly the next epoch, and reserves the target authority's
  partition slot and the configured in-flight window.
- `commit_handoff` revalidates the source fence before changing the partition
  record. A command prepared by an authority that has since lost the partition
  is rejected, even if its transaction was previously admitted.
- Exact command and commit replays are idempotent. A transaction id reused with
  different command contents is rejected. `prepare_batch` sorts commands before
  admission, so arrival order does not select a different winner.
- `max_transactions` is a bounded idempotency history. It deliberately does
  not evict old receipts; once full, new transaction ids are rejected while
  exact replays remain safe.

This is a control-plane contract, not a deployment implementation. It does not
provide cross-machine sockets, QUIC/TCP framing, encryption, authentication,
failure detection, leader election, clock synchronization, durable journal or
provider recovery, state transfer, object streaming, or network partitions/
reconciliation. A host must bind the values to its own authenticated channel,
durable provider, failure detector, and world-state transfer protocol.
