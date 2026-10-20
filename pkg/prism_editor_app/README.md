# prism_editor_app

`prism_editor_app` assembles the editor data layers into a deterministic,
headless application harness. It is suitable for CI, golden snapshots, and
automation without opening a window or linking game logic.

The harness owns frame advancement and snapshot hashing. Loom rendering,
runtime transport, and domain-specific modules remain explicit adapters rather
than hidden global state.
