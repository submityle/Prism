# prism_editor_dock

`prism_editor_dock` is the editor-layer adapter between
`prism_editor_core` panel contributions and the pure-data `prism_ui_dock`
model.

It provides:

- deterministic mapping from core `PanelDescriptor` values to dock
  `PanelDescriptor` values, including singleton versus multi-instance policy;
- stable panel-instance lifecycle events for create, close, and restore;
- workspace and perspective management with panel-reference validation;
- primary -> last-good -> default snapshot recovery;
- a byte-only persistence boundary with separate workspace and instance
  streams.

The crate does not render widgets, perform filesystem I/O, autosave, or own a
runtime/engine connection. An outer host can pass the two streams to its file
store and use `prism_ui_dock::WorkspaceFileStore` when filesystem persistence
is desired. No `bevy_*` crate is used.

Run the isolated crate tests with:

```text
cargo test --manifest-path pkg/prism_editor_dock/Cargo.toml
```
