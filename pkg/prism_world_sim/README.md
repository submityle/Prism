# prism_world_sim

Deterministic, budgeted, `no_std + alloc` world-simulation primitives.

The crate keeps authoritative state in sparse integer fields and emits stable
voxel, fragment, flow, and ecology deltas. Hydrology, erosion, and ecology
run in a fixed phase order over `BTreeMap` iteration, so equal inputs and
budgets produce equal reports. A tick cannot wrap: callers receive an explicit
`SimulationError::TickOverflow` instead.

## D8 hydrology analysis

`HydrologyField::analyze_d8` is a read-only, bounded analysis API for sparse
heightfield cells. It selects a strictly lower X/Z neighbor, uses canonical
`CellCoord` ordering to break equal-height ties, and returns the downstream
direction, flow accumulation, terminal basin, and thresholded river cells in
stable coordinate order. An optional `Region` clips both sources and
downstream candidates, so an analysis cannot silently route outside its
requested area. The caller supplies both a cell limit and a fixed eight-probe-
per-cell neighbor budget; checked `u64` accumulation never returns partial
results after overflow or budget failure.

This is a hydrology topology/analysis primitive. It does not claim a GPU
stencil solver, water-surface simulation, erosion feedback, river spline
authoring, or renderer/water-engine integration.

`encode_delta` validates canonical records before writing them. Decoding has a
16 MiB / one-million-record default budget and
`decode_delta_with_limits` allows a stricter caller-owned budget for untrusted
payloads. The wire format is little-endian and rejects truncation, trailing
bytes, invalid records, and count/size overflow.

## Optional persistence bridge

Enable `persist` to connect a `SimulationDelta` to
`prism_world_persist::Transaction` without adding a dependency to the default
core:

```text
cargo test --manifest-path pkg/prism_world_sim/Cargo.toml --features persist
```

`simulation_delta_to_transaction` copies the caller's stable transaction,
object, actor, logical-clock, and sequence inputs. It stores the canonical
simulation codec bytes as one opaque `OverrideValue::Bytes` field under the
stable `SIMULATION_DELTA_FIELD`; the inverse uses `OverrideValue::Delete` on
that same field. `transaction_to_simulation_delta` checks the exact field
shape and value types before decoding, so unrelated persistence transactions
cannot be mistaken for simulation deltas.

## Deterministic replay

`WorldSim::apply_delta` applies one previously emitted `SimulationDelta` to the
next tick without executing hydrology, erosion, or ecology again. It validates
the deterministic flow-routing and erosion equations as well as voxel,
fragment, and ecology records first, rejects out-of-order or divergent state,
and only then publishes the changes. Flow adjustments use a bounded map for
touched cells; the full sparse world is not cloned. A failed application
leaves the receiver unchanged, which makes decoded journal or network deltas
safe to retry after the caller restores the expected state.

The replay contract is covered by the crate unit tests and can be exercised
with:

```text
cargo test --manifest-path pkg/prism_world_sim/Cargo.toml --no-default-features
cargo test --manifest-path pkg/prism_world_sim/Cargo.toml --features persist
```

## Validation

From the repository root:

```text
cargo fmt --manifest-path pkg/prism_world_sim/Cargo.toml -- --check
cargo test --manifest-path pkg/prism_world_sim/Cargo.toml --no-default-features
cargo clippy --manifest-path pkg/prism_world_sim/Cargo.toml --no-default-features -- -D warnings
cargo check --manifest-path pkg/prism_world_sim/Cargo.toml --no-default-features
```
