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

## Validation

From the repository root:

```text
cargo fmt --manifest-path pkg/prism_world_sim/Cargo.toml -- --check
cargo test --manifest-path pkg/prism_world_sim/Cargo.toml --no-default-features
cargo clippy --manifest-path pkg/prism_world_sim/Cargo.toml --no-default-features -- -D warnings
cargo check --manifest-path pkg/prism_world_sim/Cargo.toml --no-default-features
```
