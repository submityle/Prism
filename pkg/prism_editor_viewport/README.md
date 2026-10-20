# prism_editor_viewport

`prism_editor_viewport` is the engine-neutral viewport boundary for Prism. It
contains no Bevy dependency and owns the contracts needed by a renderer/input
adapter:

- `SceneIdPass` keeps a persistent mapping from `(entity, mesh, subobject)` to
  a compact `Rgba32Uint` ID value. Scene snapshots and incremental patches produce
  deterministic added/updated/removed deltas; tokens are never reused during
  the encoder lifetime.
- `GpuPickingBackend` owns a real integer ID target and the 256-byte-aligned
  copy buffer required by wgpu. `submit_readback` plus `poll_readback` provides
  non-blocking correlation. An `InFlightPickReadback` owns its device/buffer and
  can be polled independently after its target closes; completion is delivered
  at most once. `submit_and_read_sync` is an explicit blocking policy.
- `RenderTargetRegistry` owns one target per viewport. Open, resize, generation
  update, and close are explicit. A stable target handle survives resize while
  `target_generation` invalidates readbacks recorded against the old texture.
- `PickCoordinator` and `BoxSelectionCoordinator` bind request IDs to viewport,
  independent viewport/camera and scene generations, and target generation.
  Old, cross-viewport, over-limit, or
  closed-target results are rejected before they can update selection. They also
  carry the allocation handle, so closing and reopening a viewport cannot make
  a late result look current merely because generation numbers coincide. GPU
  readback yields `EncodedPickId`; the coordinators resolve it through the
  current `SceneIdPass` into full `SceneObjectId` values.
- `SceneIdPassSink` is the formal renderer boundary. It receives deterministic
  added/updated/removed entries, including the exact encoded value to write;
  pipeline construction and draw recording remain renderer-owned.
- `ViewportInputFocus`, `GizmoSession`, `GizmoDrawCommand`, and `PixelRect` are
  engine-neutral interaction contracts. Platform events, camera math, and
  actual gizmo drawing remain outside this crate.

The scene delta is sorted by stable `SceneObjectId`, so adapters can apply
changes without a full scene upload. Rejected snapshots/patches leave all state
unchanged. Identical submissions do not advance the scene generation.
`draw_entries()` filters visibility and sorts by renderer sort key and identity.
The ID encoder preserves complete 128-bit entity, 128-bit mesh, and 64-bit
subobject identities in CPU tables while the GPU stores a 64-bit monotonic token
in the low two `u32` channels; all four channels are cleared to zero for a miss.
Tokens remain stable for the encoder lifetime, including remove/readd, not across
independently created encoders. Retain one `SceneIdPass` per logical scene;
replace it only after draining/cancelling its requests. Its historical identity
table grows intentionally to avoid token reuse.

## Adapter workflow

1. Submit a full `SceneIdPass::submit` snapshot or `apply_patch` operations. Apply
   the returned commit to a renderer's retained draw list with `record_commit`
   and a `SceneIdPassSink`. Commits must be applied in order; sink errors can be
   partial, so the renderer owns retry/rebuild policy.
2. Open each viewport's `GpuPickingBackend` through `RenderTargetRegistry::open`.
   Call `set_viewport_generation` after camera/settings changes and `resize`
   after extent changes. Resize preserves the handle; close/reopen does not.
3. Submit picks with `picks.submit(&viewport, target.state(), &scene, pixel)`.
   Use `target.begin_frame(&scene)`, draw each entry's `encoded_id().channels()`
   as an unblended integer fragment output, then `encode_readback` and
   `submit_readback`. Poll without blocking the editor frame.
4. Resolve with `picks.resolve(&viewport, targets.state(viewport.id()), &scene,
   result)`. Always pass current registry state, not a saved snapshot. On close,
   release input capture and cancel pending pick/box correlations. Outstanding
   GPU readbacks may finish, but closed/stale results cannot update selection.

Single-pick pending capacity is 1024; box pending capacity is 64, with an explicit
per-result hit limit supplied to `BoxSelectionCoordinator::new`. Requests outside
the current extent are rejected before allocating correlation state.

The box-selection API is a tested request/result contract, **not** a rectangular
GPU copy/reduction implementation. Its adapter must collect `EncodedPickId`s for
the requested inclusive-exclusive `PixelRect`; resolution validates generations
and IDs, then deduplicates full scene objects in stable order.

Actual scene traversal, depth/occlusion policy, geometry/transform binding,
camera matrices, platform focus routing, gizmo hit testing/rendering and applying
edits/undo remain host renderer/editor responsibilities. This crate supplies
neutral draw/submission, lifetime, focus/capture and gizmo interaction contracts;
it does not claim those application integrations are wired.

## Validation

Default tests cover identity stability, deterministic deltas, patch conflicts,
selection bounds, stale-result rejection, snapping, and request correlation.
Run the real adapter tests on a machine with a supported GPU backend:

```text
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 \
  cargo test -p prism_editor_viewport
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 \
  PRISM_REQUIRE_GPU=1 cargo test -p prism_editor_viewport --features gpu-tests -- --nocapture
cargo fmt -p prism_editor_viewport -- --check
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 \
  cargo clippy -p prism_editor_viewport --all-targets --all-features -- -D warnings
```

GPU tests print adapter name/type/backend followed by `EXERCISED` markers after
real integer draw/miss/readback and multi-viewport resize/close/reopen assertions.
`PRISM_REQUIRE_GPU=1` fails explicitly if adapter or device creation fails. When
unset, unavailable hardware prints `SKIP` with the capability error; such a run
does not prove GPU coverage. The `gpu-tests` feature must be enabled in either
case. Use an isolated `CARGO_TARGET_DIR` when validating alongside other builds.
