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
fences stale save/load receipts. Native document replacement is rejected while
the controller is dirty or an Inspector draft is active, so loading cannot
silently overwrite unsaved edits. When neither an active panel file binding nor
a scene document session exists, native Save/Open reports the missing storage
configuration instead of silently doing nothing. Runtime transport and
domain-specific modules remain explicit adapters. Surface lifecycle tests use
recording backends; the native smoke entry point is the path for actual OS
window/GPU validation.

For document-driven native scenes, `DesktopScene::new_with_document_adapter`
accepts a host-owned `SceneDocumentAdapter`. The adapter converts the validated
`EditorController` candidate into indexed geometry, full selection identities,
and a camera without embedding project gameplay types. Scene candidates are
staged from the initial controller before any native window/GPU publication.
Startup rejection returns the still-owned UI and renderer without opening the
event loop. Later load candidates are staged before UI publication and committed
only after the document install succeeds; either scene preparation or document
installation failure discards the candidate and keeps the active scene.

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
cancel panel capture, viewport gestures and pending box selection; transform
gestures also restore all drag-start matrices. Axis geometry is shared between
hit testing and the retained GPU overlay; logical input and physical output use
the viewport's scale. Translation uses calibrated world units per pixel,
rotation one degree per pixel, and scale one percent per pixel. Default snapping
is 0.1 world units, 15 degrees and 10 percent; hosts can configure it with
`set_gizmo_tool`. The three tools currently share line handles, not rotation
rings or plane handles.

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
save/load submission, active-panel file commands and native close. Failed scene
file requests do not leave the native event loop in pending-I/O polling. Native
renderer, scene-document and UI/background work retain independent pending
state: a terminal result clears only its own source, while suspend or device loss
stops GPU polling without forgetting CPU document I/O. Each native-loop pass
recomputes background pending state from the owning services, so completed work
cannot leave a stale 4 ms poll loop behind or mask another active source. Native
press also fences pending document I/O and Inspector drafts. External hosts must
cancel with `gesture_cancel` (or
`gizmo_cancel` with the same controller) before suspension, document replacement
or dropping a scene, and must serialize document installation with input. Direct
scene/camera/transform replacement is rejected during capture. The native host
performs these cancellation and close checks even without a document session:
a dirty controller cannot be closed merely because no storage provider was
configured. A completed load also rechecks transient drafts/interactions before
replacing the retained UI, so input started after submission is never discarded.

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
fail if hardware is unavailable. The latest native smoke reached Metal
initialization on a locked desktop but presented zero frames and received zero
native input events; window presentation and OS input acceptance remain unverified.

The native content dock mounts the real bound Loom browser through
`EditorDesktopUi::mount_content_browser(ContentCatalog)`. Both the CLI desktop
entry point and the Gizmo example mount an initially empty catalog. Hosts publish
real source/Git results through `content_catalog_mut()` on the UI thread, then wake
the native loop. Its refresh checks catalog generation independently of document
revision; unchanged frames keep the retained draw list and GPU uploads. Browser
queries and asset selection are available through `content_browser()` and remain
independent of document object selection, dirty state and Undo/Redo history.

The dock uses actual layout rectangles for row/button hits and browser text inputs,
including IME composition, Enter/Tab commits and Escape cancellation. Nested
favorite/pin controls update the real catalog. A catalog publication invalidates
old browser clicks and text drafts; an Inspector draft survives that publication.
While browser text owns focus, Undo/Redo does not alter document history. All text
focus conservatively fences close/reload and Gizmo capture until committed or
cancelled. Resize adjusts the bounded page size; navigation buttons reach rows
outside the current page. Input and drawing share the dock clip bounds.

Mounting the dock supplies an application-owned catalog. Source producers remain
explicit host bindings; the app does not infer project files or revisions.
The offscreen browser integration exercises real GPU text changes after
publication and verifies idle frames add no uploads.

Native timeline/world/profiler panels are explicit host contributions through
`mount_native_panel`. `DesktopTimelinePanel` retains the real authoring editor
and transport controller and delivers actual samples/events to an atomic host
consumer. World and profiler providers poll real I/O/capture results without
blocking; pending work, unsaved data and failures fence document operations.
The dock retains real Loom layout, semantic bindings and independent scroll
positions. Hidden playback continues from monotonic time without rebuilding the
browser, while paused transport permits native idle sleep.

`DesktopGraphPanel`, `DesktopPrefabPanel` and `DesktopFrameGraphPanel` mount
the host's actual graph, Prefab store and immutable capture reports. Graph
pointer input shares retained canvas geometry with painting and commits one
inverse command at release; Prefab review/confirm uses its actual controller
and fences unresolved proposals. Active authoring Undo/Redo stays in its panel.
Graph and timeline can bind an explicit absolute file/revision with
`mount_storage`: asynchronous Save/Open retains real requests, durable receipts
and isolated load candidates. Adoption needs exact load-start/current model
evidence; native adoption is disabled while dirty. Timeline save baselines
include frequency: external persistence acknowledges with
`acknowledge_saved(&token, ticks_per_second)`. Capture declaration navigation
does not by itself prove GPU execution or preview image presentation.

The native host uses one lifecycle cancellation boundary for panel pointer
captures, viewport transform gestures and pending viewport box selection. Focus
loss, resize, suspension, device loss and final shutdown therefore cannot retain
a preview whose originating native pointer sequence no longer exists.

Timeline key/tangent pointer input uses actual retained lane/curve origins,
pauses playback and paints bounded previews. Release validates its final position
before publishing one undoable transaction. Escape, focus loss, resize, suspend
and host document replacement cancel; invalid release cannot commit an earlier
valid preview. See `docs/prism_editor_native_panels_zh.md` for provider ownership,
save acknowledgement, contribution limits and acceptance evidence.

Thumbnail hosts install a shared bounded decoder and a pure
`ContentThumbnailSource` through `mount_content_thumbnails`. A source key contains
the asset, absolute path, authoritative source revision and requested size.
`invalidate_content_thumbnails` revokes old exposed images even when catalog
metadata is unchanged; the host must wake the native loop. Completion polling
builds the real keyed image nodes, and the app maps their reconciled layout to
the same dock clip used for input and text drawing. Native image ancestors are
fully opaque. Grid paging and dock wheel scrolling keep long controls/cards
reachable; resizing preserves the host's preferred column count.

`DesktopContentImages` uses the viewport's `ImageRenderer` on the native GPU.
It uploads only new visible source keys (or replaced CPU publications), releases
removed keys before allocating replacements, converts coordinates to physical
pixels, and encodes into the acquired surface texture after Loom. It introduces
no intermediate attachment or production readback. Unchanged frames retain
textures and geometry. Suspension recreates GPU residency from retained CPU
page pixels. Image preparation errors clear old preparation/residency.

`mount_content_git(provider, absolute_worktree_root)` mounts the real asynchronous
Git provider and a native refresh button/status. One active request and one latest
intent share the provider's capacity. Refresh, cancel and catalog/root replacement
revoke the old epoch; cancelled ready/running requests are drained before the
next starts. Publication requires the exact launch catalog token and current
binding, preserves previous metadata on failure, and never edits document history.
Shared-capacity Busy is visible and retried on wakes. The public status retains
the provider's exact bounded diagnostics independently of its display preview.

`mount_content_validation` installs trusted bounded host rules and policy.
Validate shows real receipt counts. `DesktopDocumentSession::save` reruns Save
validation, authorizes the exact catalog/rule/operation/policy snapshots and holds
the immutable authorization while preparing actual filesystem work. Errors,
incomplete receipts and denied warnings prevent submission; success does not
clear dirty state until the existing persistence acknowledgement. Hosts can
borrow `validated_content(Commit/Cook)` while preparing their own operations;
the app does not yet provide a Git commit or cooking product workflow. Later
asynchronous publication still needs its destination/source revision fence.

At explicit project initialization, `mount_content_collections(CollectionStore)`
loads Newton's bounded atomic collection file into a controller candidate before
publication. Startup load is synchronous and must run outside rendering; failures
preserve the prior controller/binding and never trigger automatic overwrite.
Accepted collection mutations schedule one background writer plus one latest
snapshot. UI builds perform no filesystem work. Pending or failed writes fence
close, document replacement and project/catalog replacement, preserving local
metadata without changing the document dirty flag/history. A failed write exposes
Retry collections; corrupt/unsupported files require explicit host recovery first.
A successful rename with a directory durability warning is reported as published.
Project replacement waits for the previous writer rather than discarding it.

The concrete CLI example accepts `--png <path> --revision <u128>` for one explicit
PNG binding, and `--project <existing-root>` for collection storage and Git status.
No sample source is inserted unless requested. `DesktopReport.content_image_frames`
counts actual presented native frames containing decoded resident images.

`tests/content_images.rs` exercises actual PNG decoding, keyed grid layout and
GPU pixels, replacement without a catalog-generation change, DPI, partial clipping,
idle upload counts, renderer reconstruction on the same device and image removal.
It does not simulate a physical device loss. Git tests use real temporary
repositories/processes; collection tests use actual metadata files; the native
document gate test rejects I/O then saves and reloads the real reflected model.

Verification from the repository root:

```text
cargo test -p prism_editor_app --all-targets
cargo clippy -p prism_editor_app --all-targets --no-deps -- -D warnings
cargo run -p prism_editor_app -- --frames 4
cargo run -p prism_editor_app --features desktop -- --desktop --frames 4
PRISM_REQUIRE_GPU=1 cargo test -p prism_editor_app --features desktop --test gizmo -- --nocapture
cargo run -p prism_editor_app --features desktop --example native_gizmo -- --frames 3
```

2026-10-21 verification: required-GPU `--all-features --all-targets` passed
94 tests on Apple M2/Metal: 55 library tests, 2 content-image tests, 15 controller
tests, 18 document tests and 4 Gizmo/dock tests. These include actual PNG pixel
readback, three Gizmo GPU integrations, the dock GPU integration, real Git
repositories, collection click/text/Enter followed by file persistence, failure
recovery, and rejected then successful validated save/reload. Default-feature
`--all-targets` passed 68 tests. Scoped all-feature/all-target Clippy with
`--no-deps -- -D warnings` passed; normal/build/dev dependency inspection found
no Bevy packages. No workspace build was run.

The bounded native smoke with an explicit PNG decoded its source and initialized
Apple M2/Metal, but returned `TimeLimit` after 15 seconds: 0 presented frames,
0 content-image frames and 0 native input events. The desktop automation service
again confirmed that the Mac was locked and automatic unlock failed. Native
presentation and OS input therefore remain unverified; unlock the desktop before
repeating the bounded smoke with `--png <absolute-path> --revision <u128>`.
Offscreen GPU pixel evidence does not establish native presentation.

Native authoring panels are host-owned contributions: `DesktopGraphPanel`,
`DesktopTimelinePanel`, `DesktopPrefabPanel`, `DesktopWorldPanel`,
`DesktopProfilerPanel`, `DesktopFrameGraphPanel` and `DesktopSequencePanel`.
See `docs/prism_editor_native_panels_zh.md` for explicit ownership, providers,
real semantic/pointer routing, exact dirty tokens and asynchronous file fences.

`DesktopSequencePanel::new(SequenceProjectEditor, SequencePanelLimits)` mounts
the real complete-project controller and keyed Loom view. The host supplies
documents and limits; the app creates no sample project. Numeric clip drafts
require Apply or explicit Discard before navigation/history/file replacement.
Invalid input is retained by the actual controller and fences save/close.
Active-panel native Undo/Redo uses real project inverse history. Evaluate
reports actual nested kernel samples/events without continuous transport or
runtime mutation.

Sequence `mount_storage` accepts an absolute host path,
`DesktopSequenceStorageLimits`, the observed `TimelineFileRevision` and timeout.
Save/Open use the actual complete-project worker, retaining requests until
terminal results. Loads remain isolated candidates until explicit adoption
validates both the load-start and current project tokens; drafts/errors reject
replacement. Old durable receipts cannot confirm newer authoring or clock edits.
The eight native Sequence regressions exercise retained hit geometry and scrolling,
text/Enter, actual inverse commands, nested diagnostics and real files, including
temporary and persistent actual file locks. Graph/Timeline/Sequence file routes
retry structured pre-publication Busy errors at most eight times within the
original deadline. They retain the original encoded snapshot and observed disk
revision, use monotonic wakeups without sleeping on the UI thread, and require
explicit retry after exhaustion. Other failures remain visible and fenced.
The default all-targets suite passed 100 tests (69 unit, 15 controller, 16 document).
These results do not establish Sequence runtime execution or native OS input
acceptance. After disk recovery the desktop strict Clippy attempt (session 96266)
was blocked before checking app by in-progress UI World compilation errors:
a const wrapper calling non-const transaction request and unmatched new project
actions. The planned repeated concurrency run did not execute because the shared
test binary was no longer present; no replacement test build was started.
