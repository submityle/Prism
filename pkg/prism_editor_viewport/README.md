# prism_editor_viewport

`prism_editor_viewport` is the engine-neutral viewport boundary for Prism. It
contains no Bevy dependency and owns the contracts needed by a renderer/input
adapter:

- `SceneIdPass` keeps a persistent mapping from `(entity, mesh, subobject)` to
  a compact `Rgba32Uint` ID value. Scene snapshots and incremental patches produce
  deterministic added/updated/removed deltas; tokens are never reused during
  the encoder lifetime.
- `GpuPickingBackend` owns a sampleable `Rgba32Uint` ID target, matching
  `Depth32Float` attachment, and cached compute/readback resources for bounded
  rectangle selection.
  `submit_readback` plus `poll_readback` provides
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
- `SceneIdGeometryRenderer` implements that sink with real retained indexed
  triangle geometry, per-instance model transforms, a host-supplied camera,
  and forward/reverse depth-tested ID drawing on the host's shared wgpu device.
  The ID texture can be colorized entirely on GPU for object-ID debug preview;
  this is not an engine material/color renderer.
- `ViewportInputFocus`, `ScreenGizmoLayout`, `GizmoInteraction`, `GizmoSession`,
  `GizmoDrawCommand`, and `PixelRect` are engine-neutral interaction contracts.
  Screen gizmo hit testing is deterministic and renderer-independent; platform
  events, camera math, transform-delta mapping, and actual gizmo drawing remain
  outside this crate.

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

## Screen-space gizmo interaction

`ScreenGizmoLayout` represents each axis as a screen-pixel segment from one
origin to one endpoint. Construct it with a finite, non-negative tolerance no
larger than `MAX_SCREEN_GIZMO_TOLERANCE_PX` (64 pixels). Layout coordinates,
axis endpoints, tolerance, and pointer coordinates must be finite; invalid
layouts return `ScreenGizmoLayoutError`, while a non-finite pointer is always a
miss. A zero-length axis is rejected.

`hit_test` uses point-to-segment distance with the bounded tolerance. Overlapping
axes are resolved by the fixed priority X, then Y, then Z, so the result does
not depend on a caller's collection order or tiny distance differences. A hit
can start `GizmoInteraction`, which stores the selected axis and delegates
`update`, `finish`, and `cancel` to the existing `GizmoSession` exactly once
according to its active-state semantics:

```rust
let layout = ScreenGizmoLayout::new(
    ScreenPoint::new(320.0, 240.0),
    [
        ScreenPoint::new(400.0, 240.0),
        ScreenPoint::new(320.0, 320.0),
        ScreenPoint::new(280.0, 240.0),
    ],
    6.0,
)?;
let mut interaction = GizmoInteraction::begin_from_pointer(
    layout,
    pointer,
    GizmoMode::Translate,
    SnapSettings::default(),
)?;
let preview = interaction.update(axis_aware_delta);
let committed = interaction.finish();
```

The crate deliberately does not infer camera projection or convert pixel
movement into world units. The host supplies an axis-aware `TransformDelta`;
`GizmoInteraction::draw_commands()` exposes the selected axis for a renderer or
UI adapter.

Single-pick pending capacity is 1024; box pending capacity is 64, with an explicit
per-result hit limit supplied to `BoxSelectionCoordinator::new`. Requests outside
the current extent are rejected before allocating correlation state.

## Shared GPU indexed geometry

Use the desktop's existing wgpu 30 `Device`, `Adapter`, and `Queue`; no window,
surface, second device, or ECS is created by this crate. The following excerpts
use host-owned values (`viewport`, `targets`, `scene`, and mesh/camera data).

```rust
let backend = GpuPickingBackend::new(
    &device, &adapter, &queue, GpuPickingConfig::new(width, height),
)?;
targets.open(viewport.id(), viewport.generation(), backend)?;
let mut renderer = SceneIdGeometryRenderer::new(&device);

// positions: &[[f32; 3]], indices: &[u32], triangle list, object space.
renderer.upload_geometry(&mut scene, draw_key, positions, indices)?;
renderer.set_transform(&mut scene, instance_key, model_matrix)?;
let commit = scene.submit(items)?;
scene.record_commit(&commit, &mut renderer)?;
```

Each `SceneIdPassItem` associates a full `SceneObjectId` with `draw_key` and
`instance_key`. Keys may be shared between objects/subobjects. Upload resources
before creating requests: every successful mesh/transform upload or replacement
advances `scene.generation()` even when the key is unchanged. Removal does the
same when a resource exists. Invalid inputs leave the scene/resources unchanged.
Tokens do not change. Custom render adapters must call `invalidate_content()`
when geometry or transforms change without a scene-item delta.

Retain one renderer per logical scene/device. Apply scene commits in order with
`record_commit`, or use `renderer.synchronize(&scene)` for initial attach/recovery.
`render` rejects an unsynchronized draw list, missing visible mesh/transform,
stale scene frame, mismatched camera target, or non-finite matrix before recording
a pass. Geometry is a position-only `Float32x3` vertex stream and `Uint32` index
stream; indices and finite positions are validated at upload.

```rust
let target = targets.get(viewport.id()).expect("open viewport target");
let camera = SceneIdCamera {
    target: target.state(),
    view_projection,
    depth: SceneIdDepth::Forward,
};
let frame = target.begin_frame(&scene);
renderer.render(&mut encoder, &frame, &scene, &camera)?;
```

`SceneIdMatrix = [[f32; 4]; 4]` is column-major; WGSL computes
`view_projection * model * position`. The host supplies a WebGPU `[0, 1]` clip
depth projection with the current target aspect ratio. `Forward` clears depth to
1 and tests strict `Less`; `Reverse` clears to 0 and tests strict `Greater`.
Both write depth, use single-sample opaque triangles and no face culling. Equal
depth keeps the first draw in deterministic scene sort order. Each render clears
the entire ID target to zero/miss and the matching depth target to far.

Camera matrix or depth-convention changes require advancing the viewport/camera
generation and calling `targets.set_viewport_generation` before constructing the
new camera/request. Resize through the registry and reconstruct `SceneIdCamera`
from its new target state. These host-supplied matrices are not inferred from
`CameraState` by this crate.

## GPU-only object-ID debug preview

Public `target.backend().target_view() -> &wgpu::TextureView` exposes the actual
ID target with `RENDER_ATTACHMENT | COPY_SRC | TEXTURE_BINDING` usage. After
`renderer.render` ends its ID/depth pass, bind that view in a separate surface
render pass. Its layout entry must be unsigned, not float/filterable:

```rust
let entry = wgpu::BindGroupLayoutEntry {
    binding: 0,
    visibility: wgpu::ShaderStages::FRAGMENT,
    ty: wgpu::BindingType::Texture {
        sample_type: wgpu::TextureSampleType::Uint,
        view_dimension: wgpu::TextureViewDimension::D2,
        multisampled: false,
    },
    count: None,
};
let binding = wgpu::BindGroupEntry {
    binding: 0,
    resource: wgpu::BindingResource::TextureView(target.backend().target_view()),
};
```

A minimal fragment stage for a full-size, 1:1 target/surface preview is below.
It consumes geometry already rasterized by the indexed pass, not a synthetic
fullscreen ID fixture. The fullscreen presentation triangle is host-owned.

```wgsl
@group(0) @binding(0) var ids: texture_2d<u32>;

@fragment fn fs(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    let size = vec2<i32>(textureDimensions(ids));
    let pixel = clamp(vec2<i32>(p.xy), vec2<i32>(0), size - vec2<i32>(1));
    let id = textureLoad(ids, pixel, 0);
    if all(id == vec4<u32>(0u)) {
        return vec4<f32>(0.025, 0.025, 0.035, 1.0);
    }
    let hash = id.x * 1664525u + id.y * 1013904223u;
    let rgb = vec3<u32>(hash, hash >> 8u, hash >> 16u) & vec3<u32>(255u);
    return vec4<f32>(0.2 + 0.8 * vec3<f32>(rgb) / 255.0, 1.0);
}
```

No sampler or CPU image readback is needed. Use the actual surface format for
the host's color pipeline. If the viewport is offset/scaled inside a larger
surface, map surface positions/UVs to ID texels explicitly; the snippet assumes
equal extents and origin. Never attach and sample the ID texture in the same
pass. Rebuild the host bind group whenever `(handle, target_generation)` changes
(resize, close/reopen); `target_view()` after resize refers to the new texture.
`depth_view()` also exposes the matching `Depth32Float` attachment, not a
sampleable depth texture. There is no basic entity color/MRT/material pass in
this renderer; the supported presentation is explicitly object-ID debug color.

## Point readback and GPU rectangle reduction

Point readback uses `frame.encode_readback`. Rectangle selection is an actual
requested-region GPU compute scan and compact reduction, followed by bounded
CPU extraction of the mapped result table:

```rust
let request = boxes.submit(&viewport, target.state(), &scene, rectangle)?;
let pending = frame.encode_box_readback(
    &mut encoder, request, GpuBoxSelectionLimits::default(),
)?;
let mut flight = target.backend().submit_box_readback(encoder, pending);
if let Some(result) = flight.poll()? {
    let selection = boxes.resolve(
        &viewport, targets.state(viewport.id()), &scene, result,
    ); // SelectionOutcome: applied selection or an explicit rejection.
}
```

`PixelRect` is inclusive-exclusive and must be non-empty. The request is rejected
before GPU allocation when it is outside the current target. The rectangle path
is a real GPU reduction: one compute invocation reads one pixel from the actual
`Rgba32Uint` target, ignores the all-zero miss, and inserts the four-channel ID
into a bounded linear-probing hash table guarded by GPU atomics. The GPU also
publishes the unique-ID count and an overflow flag. The mapped staging buffer
contains only the compact hash table and those two counters; the host extracts
non-empty slots but does not scan rectangle pixels or perform ID deduplication.
This selects visible fragments, not objects hidden behind other geometry; it is
not a through-selection algorithm.

`GpuBoxSelectionLimits` bounds scanned pixels, compact reduction staging bytes,
and unique IDs. Defaults are 1,048,576 pixels, 32 MiB, and 4096 unique IDs. The
GPU table is sized to at least twice `max_ids` and rounded to a power of two;
the resulting storage and dispatch sizes are checked against device limits.
Set `max_ids` no higher than `BoxSelectionCoordinator::new(max_hits)`'s bound.
Exceeding a limit or GPU table capacity returns an error, never a silently
truncated selection. The host must also bound the number of in-flight staging
buffers and cancel coordinator requests on encoding or mapping errors; the
coordinator's pending bound does not own those buffers.
Dropping a flight cancels mapping/releases its buffer, but does not cancel the
coordinator request. Both point and box `poll()` deliver completion/error at
most once and can finish after their original target closes; always resolve
against current viewport, target and scene state.

For shared-queue batching, encode ID geometry, presentation, and any point/box
readback operations into the host encoder; call `queue.submit([encoder.finish()])`, then
`target.backend().start_readback(pending_pick)` and/or
`target.backend().start_box_readback(pending_box)`. These methods only start
mapping: they must be called after submitting the matching operations. The
  `submit_*` convenience methods instead submit the encoder themselves; do not
  double-submit. Blocking `submit_and_read_sync` / `submit_box_and_read_sync`
  are explicit tool/test policies, not a per-frame presentation path.

Actual scene traversal/extraction, camera math, surface creation/compositing,
platform events/focus, gizmo rendering, pixel-to-transform mapping, edits/undo,
and the engine's real material/render adapter remain host responsibilities.
These GPU and interaction primitives do not mean those application integrations
are wired or that a complete editor is implemented.

## Validation

Default tests cover identity stability, deterministic deltas, patch conflicts,
selection bounds, stale-result rejection, snapping, and request correlation.
The required GPU tests cover indexed geometry, nonzero-origin rectangles, misses,
GPU deduplication, overflow rejection, shared point/box submission, and all
viewport/scene/target generation invalidation paths. Run the real adapter tests
on a machine with a supported GPU backend. Allocate
an isolated target and clean only that exact directory when finished:

```text
viewport_target=$(mktemp -d /private/tmp/prism-editor-viewport-target.XXXXXX)
export CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0
export CARGO_BUILD_JOBS=2
cargo test --locked --offline -p prism_editor_viewport --target-dir "$viewport_target"
PRISM_REQUIRE_GPU=1 cargo test --locked --offline -p prism_editor_viewport \
  --features gpu-tests --target-dir "$viewport_target" -- --nocapture
cargo fmt -p prism_editor_viewport -- --check
cargo clippy --locked --offline -p prism_editor_viewport --all-targets \
  --all-features --target-dir "$viewport_target" -- -D warnings
cargo clean --target-dir "$viewport_target"
```

GPU tests print adapter name/type/backend followed by `EXERCISED` markers after
real indexed overlapping geometry/forward+reverse depth/camera/model/mesh
mutation/miss/readback, multi-viewport resize/close/reopen, bounded rectangular
copies, and unsigned ID texture sampling into a GPU debug-color pass. That last
test uses a color readback only to assert output, not as a presentation path.
`PRISM_REQUIRE_GPU=1` fails explicitly if adapter or device creation fails. When
unset, unavailable hardware prints `SKIP` with the capability error; such a run
does not prove GPU coverage. The `gpu-tests` feature must be enabled in either
case. Use an isolated `CARGO_TARGET_DIR` when validating alongside other builds.
