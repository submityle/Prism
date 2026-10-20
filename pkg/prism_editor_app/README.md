# prism_editor_app

`prism_editor_app` assembles the editor data layers into a deterministic
application harness. Headless mode supports CI, golden snapshots, and
automation without opening a window or linking game logic. With the `desktop`
feature, the binary also exposes a native winit/wgpu smoke entry point and the
library exposes the Loom presenter plus the engine-neutral viewport seam.

The harness owns frame advancement, snapshot hashing, and resource lifecycle:

- Startup acquires windows, then the GPU context, then per-window surfaces in
  stable window-identity order. Partial startup and frame failures release
  acquired resources and expose the error through `last_error`.
- `tick` polls a bounded event batch, processes resize/suspend/resume/close,
  and presents active surfaces. A suspended window does not pause other active
  windows; zero-sized drawables suspend until a non-zero resize or resume.
- `stop` and `Drop` release surfaces in reverse creation order, then the GPU,
  then windows. Cleanup continues after a backend error; the first error is
  returned. Backends must make their own stop operation idempotent and clean up
  partial starts.
- `recover` rebuilds host resources after failure while retaining documents,
  selection, and other editor models. Replacing adapters and configuring
  windows are restricted to inactive lifecycle states.
- Headless runs have a frame limit, preserve a hook-requested exit reason, and
  report the actual number of frames advanced. Window count and event work
  have explicit budgets; a backend that violates the polling limit fails
  visibly rather than creating unbounded per-frame work.
- `build_prism_app` and `build_prism_app_with_backends` install the application
  resource and update system. Completion and failure propagate to
  `prism_app::AppExitRequest`.

Document persistence is opt-in through `DesktopDocumentSession` and
`run_desktop_with_document`; the session preserves dirty/draft state and
fences stale save/load receipts. Runtime transport and domain-specific modules
remain explicit adapters. Surface lifecycle tests use recording backends; the
native smoke entry point is the path for actual OS window/GPU validation.

For document-driven native scenes, `DesktopScene::new_with_document_adapter`
accepts a host-owned `SceneDocumentAdapter`. The adapter converts the validated
`EditorController` candidate into indexed geometry, full selection identities,
and a camera without embedding project gameplay types. Scene candidates are
staged before UI publication and committed only after the document install
succeeds; an install failure discards the candidate and keeps the active scene.

Viewport clicks use asynchronous integer-ID readback. A native drag inside the
viewport becomes a bounded rectangle copy and resolves all visible identities
through one controller selection commit; stale target, scene, or controller
revisions are rejected without mutating selection.

Host gizmo/runtime adapters can update a single `DesktopSceneObject` through
`DesktopScene::update_object_model`. A live device replaces only that instance
transform bind group and advances the scene fence; unrelated mesh buffers are
retained. Before device initialization or after suspension, the validated CPU
transform is deferred to the next upload. Pending picks are cancelled so a
transform edit cannot resolve against old pixels.

Native transform gestures are opt-in through
`DesktopScene::set_transform_adapter(SceneTransformAdapter)`. The host explicitly
maps reflected roots to mesh matrices and cumulative `TransformDelta` values to
scalar `ControllerFieldEdit` paths. The app never assumes field names, gameplay
types, hierarchy inheritance, or a project coordinate convention. Methods must
be pure and deterministic; unsupported or locked selections must return an error
for the entire gesture. Every selected identity needs a mesh binding; multiple
meshes may share one model. The default interaction applies a delta about each
object's own pivot, using the primary object's screen axes. Hosts own parent/child
policy and world/local conversion through the adapter.

W/E/R select translation/rotation/scale. Pressing an axis captures input before
point/box selection, movement previews the selected meshes, and release commits
one real multi-object command. Escape, focus loss, resize, suspend and device loss
restore all drag-start matrices. Axis geometry is shared between hit testing and
the retained GPU overlay; logical input and physical output use the viewport's
scale. Translation uses calibrated world units per pixel, rotation one degree
per pixel, and scale one percent per pixel. Default snapping is 0.1 world units,
15 degrees and 10 percent; hosts can configure it with `set_gizmo_tool`. The three
tools currently share line handles, not rotation rings or plane handles.

The default `GizmoFrame::project` projects world axes to 72 logical pixels and
hides the handles outside/behind the camera or when an axis projects to zero
length. A host can override `SceneTransformAdapter::frame` for another camera or
degenerate-axis policy. The runnable example uses an oblique camera so X/Y/Z
are all visible.

Each move prepares isolated model candidates and validates all resulting mesh
matrices before publishing a preview. An adapter, field-validation, capacity or
matrix failure cancels the complete gesture and restores the original preview.
The controller, model revision, dirty flag and history remain unchanged until
release; returning to the original transform creates no command. Undo/Redo exchange
the complete validated batch of scalar fields, independent of selection,
preserving the concrete Rust model roots throughout publication and history.
`ControllerLimits::max_batch_fields` bounds each batch, and existing model and
history limits also apply. Duplicate field targets are rejected.

An active gesture fences selection, other model commands, document encoding,
save/load submission and native close. Native press also fences pending document
I/O and Inspector drafts. External hosts must cancel with `gesture_cancel` (or
`gizmo_cancel` with the same controller) before suspension, document replacement
or dropping a scene, and must serialize document installation with input. Direct
scene/camera/transform replacement is rejected during capture. The native host
performs these cancellation and close checks even without a document session.

After release, Undo/Redo or external controller commands, the next encode reads
only changed identities from a bounded revision journal and updates their
instance transforms. Idle frames do no transform uploads. Selection-to-mesh and
mesh-ID indices avoid full scene scans; a full transform resync is needed after
load/rebinding or when a renderer falls behind the journal. Mesh buffers remain
retained. Full material rendering and inferred project transform extraction are
outside this example.

`examples/native_gizmo/support.rs` is a concrete host schema mapping a reflected
matrix to 16 explicit scalar paths. `tests/gizmo.rs` exercises real offscreen
axis pixels, geometry, picking, multiselect commands, Undo/Redo, cancellation,
failure rollback and codec reload; the transform persistence test in
`tests/document.rs` uses the production filesystem provider. Required-GPU runs
fail if hardware is unavailable. The native smoke proves window presentation,
not automated OS mouse/key delivery.

The native content dock mounts the real bound Loom browser through
`EditorDesktopUi::mount_content_browser(ContentCatalog)`. Both the CLI desktop
entry point and the Gizmo example mount an initially empty catalog. Hosts publish
real source/Git results through `content_catalog_mut()` on the UI thread, then wake
the native loop. Its refresh checks catalog generation independently of document
revision; unchanged frames keep the retained draw list and GPU uploads. Browser
queries and asset selection are available through `content_browser()` and remain
independent of document object selection, dirty state and Undo/Redo history.

The dock uses actual layout rectangles for row/button hits and four text inputs,
including IME composition, Enter/Tab commits and Escape cancellation. Nested
favorite/pin controls update the real catalog. A catalog publication invalidates
old browser clicks and text drafts; an Inspector draft survives that publication.
While browser text owns focus, Undo/Redo does not alter document history. All text
focus conservatively fences close/reload and Gizmo capture until committed or
cancelled. Resize adjusts the bounded page size; navigation buttons reach rows
outside the current page. Input and drawing share the dock clip bounds.

Mounting supplies no source scan, Git worker, thumbnail or pipeline by itself;
those services publish into the application-owned catalog. No sample assets are
inserted into the native catalog. The offscreen browser integration exercises
real GPU text changes after publication and verifies idle frames add no uploads.

Verification from the repository root:

```text
cargo test -p prism_editor_app --all-targets
cargo clippy -p prism_editor_app --all-targets --no-deps -- -D warnings
cargo run -p prism_editor_app -- --frames 4
cargo run -p prism_editor_app --features desktop -- --desktop --frames 4
PRISM_REQUIRE_GPU=1 cargo test -p prism_editor_app --features desktop --test gizmo -- --nocapture
cargo run -p prism_editor_app --features desktop --example native_gizmo -- --frames 3
```

2026-10-20 verification: required-GPU `--all-features --all-targets` passed
85 tests on Apple M2/Metal, including three Gizmo offscreen integration tests and
one native content dock GPU integration test; default-feature `--all-targets`
passed 62 tests. Scoped all-feature/all-target
Clippy with `--no-deps -- -D warnings` passed. Native smoke initialized an
Apple M2/Metal device but presented zero frames and reached its 15-second bound.
The desktop automation service confirmed the Mac was locked and could not
unlock it; this was confirmed again after the content dock integration. Native
presentation and OS input validation therefore remain
unverified; unlock the desktop before repeating the bounded smoke command.
