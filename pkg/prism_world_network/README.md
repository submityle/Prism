# prism_world_network

`prism_world_network` is the M8 world-to-network seam. It is a `no_std +
alloc` policy crate with no transport, socket, serializer, ECS, scene, or
filesystem implementation.

## Single-source-of-truth boundary

- `prism_world_stream` owns spatial identity and world-space bounds. This crate
  reuses its `SectorId`, `GlobalPos`, and `AabbI64` types rather than defining
  network copies.
- `prism_world_persist` owns durable object identity, field deltas, transaction
  ordering, and field-level LWW arbitration. This crate reuses its
  `PersistentId`, `Transaction`, and CRDT types rather than creating a second
  persistence model.
- This crate owns only the boundary policy: inclusive AOI selection,
  deterministic delta envelopes, publish permissions, sector handoff records,
  and an object-indexed merge facade.
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
the minimum and maximum corners are inside the subscription. Transport code
may use these contracts to decide what to send, but delivery, retries,
serialization, authentication, and connection lifecycle stay outside this
crate.
