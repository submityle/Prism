# prism_editor_viewport

`prism_editor_viewport` owns the deterministic editor-side contracts for
multiple viewports, picking, and command-oriented gizmo drags. It also ships a
renderer-independent but executable wgpu ID-buffer backend in
`src/gpu_picking.rs`.

The backend creates an `Rgba32Uint` render target with `RENDER_ATTACHMENT | COPY_SRC`
usage, exposes a frame-scoped render pass for the renderer's object-ID pipeline,
and copies one requested pixel into a 256-byte aligned staging buffer. The
`submit_and_read_sync` method intentionally uses a blocking `Device::poll` and
map callback: this is a bounded, explicit policy for editor pointer picks, not
a bulk per-frame readback path. The returned `PickResult` retains the request's
viewport, generation, and correlation ID so the existing coordinator can reject
stale results.

Object IDs are encoded as four little-endian `u32` channels. All zero means a
miss; an entity is written as `entity + 1`, which makes the maximum `u128`
entity value an explicit capacity error instead of silently aliasing a miss.

The GPU integration test is opt-in with `--features gpu-tests`; it draws a
triangle into the real integer target, submits the copy, waits for mapping, and
verifies the decoded entity. The default unit tests cover encoding and all
boundary/correlation checks without requiring an adapter.

The model rejects stale pick results, keeps viewport identities monotonic, and
coalesces a drag into one final transform delta so an editor command stack can
undo the interaction as a unit.
