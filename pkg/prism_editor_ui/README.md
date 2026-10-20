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
