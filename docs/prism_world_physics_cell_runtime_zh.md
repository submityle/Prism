# World Physics Cell Runtime

`prism_world_physics_cell_runtime` stages terrain-derived heightfield residency around the `SceneHost` cell boundary. It decodes and validates a complete cell snapshot, checks sector identity and tile ownership, stages the aggregate physics residency, and only publishes the staged residency after the downstream host accepts the cell transition.

## Same-cell reload revision contract

For every tile retained across a reload of the same resident sector, the incoming terrain revision must be strictly greater than the committed physics snapshot revision. Equal and lower revisions return `HeightfieldBridgeError::StaleRevision` before calling the downstream host. The old snapshot remains value-identical, including its revision, embedding, converted physics heightfield, and upload-byte accounting.

The revision floor is read from committed residency before a staged copy evicts any old tiles. Only after identity, ownership, and monotonicity validation succeeds does the adapter reserve the replacement tile index, clone the budget-bounded residency, evict the old cell's tiles from that temporary copy, and publish replacement tiles. Any validation, allocation, bridge, or host failure leaves committed residency and ownership indexes unchanged. Tiles removed from the replacement cell are evicted after a successful host load; tiles newly introduced by the cell follow normal ownership checks.

## Regression coverage

The unit regression loads tile revision 10, attempts same-tile reloads at revisions 9 and 10, and verifies both fail before a second host load while the complete old `PhysicsHeightfieldSnapshot` remains equal to its saved value. A strictly newer revision remains the supported replacement path and is covered by the codec-driven reload tests.

Run the focused crate checks from the repository root:

```sh
cargo test -p prism_world_physics_cell_runtime --lib --offline
cargo fmt --check -p prism_world_physics_cell_runtime
```
