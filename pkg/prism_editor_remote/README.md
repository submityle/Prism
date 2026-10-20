# prism_editor_remote

`prism_editor_remote` is a transport-neutral synchronization boundary between
an editor and a runtime. It defines stable request/response envelopes,
endpoint capabilities, typed values, snapshots, incremental component/property
changes, revision checks, conflict errors, an in-memory reference endpoint, and
a deterministic binary codec.

The crate does not implement BRP, JSON-RPC, sockets, HTTP, or a particular
runtime. A future adapter can map a transport frame to
`RequestEnvelope`/`ResponseEnvelope`, forward it to an `Endpoint`, and map the
response back. This keeps transport policy and runtime ownership outside the
protocol core.

The binary codec is deliberately JSON-free. It uses little-endian integers,
length-prefixed strings/byte arrays, sorted map keys, and explicit enum tags.
`CodecLimits` bounds frame size, collection size, string size, byte-array size,
and value nesting depth before allocating untrusted data.

Run the crate tests with:

```text
cargo test --manifest-path pkg/prism_editor_remote/Cargo.toml
```
