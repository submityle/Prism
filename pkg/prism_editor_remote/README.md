# prism_editor_remote

`prism_editor_remote` is a transport-neutral synchronization boundary between
an editor and a runtime. It defines stable request/response envelopes,
endpoint capabilities, typed values, snapshots, incremental component/property
changes, revision checks, conflict errors, an in-memory reference endpoint, a
deterministic binary codec, and a bounded JSON-RPC 2.0 adapter.

The JSON-RPC adapter is runtime-neutral and uses BRP-style method names:
`prism/initialize`, `prism/describe`, `prism/world/readSnapshot`,
`prism/world/readChanges`, `prism/world/applyChanges`,
`prism/world/subscribe`, and `prism/world/unsubscribe`. It implements strict
JSON-RPC 2.0 request/response/notification shapes, deterministic encoding,
bounded parsing, capability negotiation, request correlation, and contiguous
incremental notifications. `publish_changes` accepts a target explicitly
because `ChangeSet` remains reusable across transport-neutral stores.

The crate includes a synchronous standard-library TCP transport. Each payload
is prefixed by a four-byte big-endian length and is rejected before allocation
when it exceeds `TransportLimits::max_frame_bytes`. `TcpJsonRpcListener` and
`connect_tcp` attach `JsonRpcServer` and `JsonRpcClient` to that framing layer;
read/write/connect timeouts are configurable, clean EOF is distinct from a
truncated frame, and `JsonRpcServerSession::publish_changes` writes subscribed
notifications immediately. `LengthPrefixedStream` also works with any
`Read`/`Write` pair, which keeps framing tests deterministic and transport
independent.

The crate does not depend on a renderer, ECS, engine, or third-party socket
runtime. The wire path is:

```text
TCP stream -> length prefix -> JSON-RPC payload -> JsonRpcServer -> Endpoint
TCP stream <- length prefix <- JSON-RPC response/notification
```

The binary codec is deliberately JSON-free. It uses little-endian integers,
length-prefixed strings/byte arrays, sorted map keys, and explicit enum tags.
`CodecLimits` bounds frame size, collection size, string size, byte-array size,
and value nesting depth before allocating untrusted data.

`PieSession` starts from a cloned editor snapshot. Runtime changes are applied
only to the private runtime copy and recorded in deterministic order. Stopping
with `Rollback` discards the commit set; stopping with `Commit` returns an
explicit, revision-checked `ChangeSet`. The session never mutates its edit
baseline implicitly, and `PieStopResult::apply_commit` is the only explicit
editor mutation path.

Run the crate tests with:

```text
cargo test --manifest-path pkg/prism_editor_remote/Cargo.toml
```
