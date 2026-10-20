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

This crate does not add a renderer, input dispatcher, reflection mutation
policy, or engine runtime adapter. Those contracts remain owned by the existing
Prism crates; this crate only assembles their current state into Elements.
