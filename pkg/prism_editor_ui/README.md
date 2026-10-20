# prism_editor_ui

`prism_editor_ui` is the concrete Loom assembly layer for Prism Studio. It
consumes the editor kernel and panel models and emits renderer-independent
`prism_ui::Element` trees. A backend can reconcile these trees through the
normal Loom runtime without this crate knowing whether the result is drawn by
a GPU, a native host, or a recording backend.

The crate provides builders for:

- shell chrome with menu, toolbar, and status contributions;
- a scored command palette backed by `prism_editor_shell::CommandIndex`;
- bounded, keyed Outliner and Console panel trees;
- a paged content folder/list or thumbnail grid with search, kind/tag/VCS
  filters, favorites, pins, bounded multiselection and named collections;
- bounded dependency/dependent navigation and paged validation receipts with
  explicit run and locate actions;
- reflection projection rows with field paths and editability hints;
- dock workspace chrome backed by `prism_editor_dock` contributions;
- a renderer-neutral viewport host with camera/settings readouts;
- a fixed-order editor workspace composition.

Dynamic rows always carry semantic string keys. The visible-window builders
cap materialized rows at `MAX_VISIBLE_ROWS`, so a large model does not turn a
single render into an unbounded element allocation. Filtering and empty states
are represented directly in the tree and are therefore observable in headless
Loom tests.

The crate also exposes an engine-neutral action contract. The *_bound builders
return a BoundElement, which pairs the Element tree with bounded,
deterministically ordered ActionBindings:

- Outliner rows produce SelectOutliner; disclosure targets produce
  ToggleOutlinerExpansion.
- Editable Inspector leaves produce EditInspectorField with a canonical
  FieldPath and a bounded ActionValue; read-only fields have no edit binding.
- Enabled command-palette, menu, and toolbar entries produce ExecuteCommand.
- ActionBindings::bind_undo and bind_redo let the application layer inject
  Undo/Redo targets without making the UI own a command stack.

The backend sends a semantic target key plus a UiEvent to
BoundElement::dispatch; the result is a deterministic EditorAction for the
application or editor kernel to execute. No callback, window handle, GPU
object, ECS world, or engine type crosses this boundary.

Dynamic inputs are bounded: visible rows, Inspector depth/node count, action
target/command/path/text byte lengths, and total bindings all have explicit
limits. Duplicate bindings and trigger mismatches are rejected before
dispatch. The plain builders remain available when an application only needs
the view tree; bound builders are the opt-in event contract.

This crate does not add a renderer, input device dispatcher, reflection
mutation policy, or engine runtime adapter. Those policies remain owned by
the application and existing Prism crates; this crate only assembles current
state and translates semantic UI events into actions.

Content browsing uses a separate `ContentBrowserAction` contract and does not
extend `EditorAction`. `ContentBrowserController::build(&mut catalog)` returns
`BoundContentBrowser`; mount or reconcile `view.element()` in Loom and use
`view.bindings()` for semantic hit targets. Pass the displayed view, target key
and `UiEvent` to `controller.handle_event(&mut catalog, &view, key, &event)`,
then rebuild and reconcile. Search/kind/tag/VCS inputs accept text `Commit`
events, asset rows accept `Select`, and buttons accept `Activate`. Disabled
controls have no binding. `dispatch` returns an owned action without writing;
`apply` supports explicit host commands. `handle_event` additionally rejects
views from an earlier controller revision or catalog generation.

The controller owns query, bounded cursor history, selection and collections. It
borrows the host's catalog for `tree_page` queries and favorite/pinned updates,
preserving all other record fields. Each page contains at most
`min(page_size, MAX_VISIBLE_ROWS, catalog.limits().max_page_size)` rows; the
default request is 64. Selection is capped at 4096 assets and previous-page
history at 64 cursors. `Extend` replaces selection with the current page's
inclusive anchor range; an anchor on another page falls back to the target.
Selection survives navigation/filtering but deleted IDs are pruned on refresh.
Catalog generation changes restart paging and a removed open folder falls
back to root. Home also restarts the current folder traversal at root while
retaining filters. Search normalization and filtering are the catalog's own
semantics (ASCII case folding); kinds match exactly.

`ContentLayout::Grid { columns }` uses real fixed-width card/row layout with
1..=8 columns. Collection creation, rename, deletion, membership changes and
navigation use explicit `ContentBrowserAction` values and real keyed controls.
At most 64 collections with 4096 stable asset IDs each are retained. Collection
queries span folders, apply the same filters, sort by path then ID, and page
deterministically. Deleted IDs remain members but have no visible row. Collection
persistence/restoration is outside this controller's current API.

`ContentThumbnails` shares the host's `ThumbnailService`. After building a page,
supply authoritative visible `ThumbnailKey` values through `sync_visible`, tick
`poll` while `needs_poll`, then use `build_with_thumbnails`. No caller file I/O
occurs. Ready results alone produce `prism-editor-content-image` custom layout
nodes and `BoundContentBrowser::images()` entries pairing exact `visual_key`,
request identity and `Arc<Thumbnail>` RGBA8 pixels. Project each key's actual
Loom layout to physical rect/clip/opacity and upload/prepare/encode using the
host's image renderer. Custom nodes alone do not draw images. Missing, queued
and failed requests never create fake images. Page-owned decoded pixels are
capped at 32 MiB; off-page requests/results are dropped at synchronization.

The native app owns input routing, revision/source resolution, redraw scheduling,
layout projection and GPU handles. Rebuild and resynchronize keys after catalog
publication; generation mismatch hides old visuals. Project close must clear
page data, shared service and host textures; service clear alone cannot revoke
already cloned Arcs. This UI owns no filesystem/Git scanner, GPU resource,
persistence service or window. See `docs/prism_editor_content_grid_ui_zh.md` for
the additive API, resource ownership and actual PNG/Loom/GPU pixel regression.

`ContentDiagnosticsController` provides a separate E7 panel without extending
`EditorAction`. Call `apply(&catalog, &registry, ContentDiagnosticsAction::Focus(
Some(asset_id)))` when host selection changes, then `build(&catalog, &registry)`
to obtain `BoundContentDiagnostics`. Mount/reconcile `element()` and forward
its `bindings()` through `handle_event(&catalog, &registry, &view, key, &event)`.
All bound controls accept `Activate`; disabled rows have no binding. After a
dependency or finding locate action, `selected_asset()` is the host's navigation
result. The host remains responsible for synchronizing other panels and opening
asset editors. Views belong to their originating controller; rebuild after each
accepted action or catalog/registry publication.

Outgoing/incoming lists use the catalog's bounded transitive closures, sort by
stable asset ID, and materialize at most `MAX_VISIBLE_ROWS` per list (64 by
default). Traversal failures are visible; missing targets are labeled and cannot
be opened. Back history holds at most 64 asset identities. A catalog generation
change resets dependency paging; a removed focus remains visibly missing.
Only explicit `RunValidation(Save/Commit/Cook)` invokes the real registry. One
immutable `ValidationReceipt` is retained with its error/warning counts,
completeness, truncation and freshness. Stale reports remain readable with locate
disabled. `receipt()` exposes the actual result; the application must still use
its authorization API immediately before preparing an operation. No operation,
filesystem worker, graph canvas, renderer or GPU resource is owned by this UI.
See `docs/prism_editor_content_diagnostics_ui_zh.md` for the additive native API
boundary and RecordingBackend verification.
