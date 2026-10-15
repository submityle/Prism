# prism_world_sim

Deterministic, budgeted, `no_std + alloc` world-simulation primitives.

The crate keeps authoritative state in sparse integer fields and emits stable
voxel, fragment, flow, and ecology deltas. Hydrology, erosion, and ecology
run in a fixed phase order over `BTreeMap` iteration, so equal inputs and
budgets produce equal reports. A tick cannot wrap: callers receive an explicit
`SimulationError::TickOverflow` instead.

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

## Validation

From the repository root:

```text
cargo fmt --manifest-path pkg/prism_world_sim/Cargo.toml -- --check
cargo test --manifest-path pkg/prism_world_sim/Cargo.toml --no-default-features
cargo clippy --manifest-path pkg/prism_world_sim/Cargo.toml --no-default-features -- -D warnings
cargo check --manifest-path pkg/prism_world_sim/Cargo.toml --no-default-features
```
