# prism_editor_shell

`prism_editor_shell` provides the renderer-independent data model for the
Loom Studio shell:

- `CommandIndex` stores disposable command contributions and performs
  deterministic token search for a command panel.
- `KeySequence` parses normalized keyboard strokes, while `ContextExpr` parses
  boolean `when` expressions over context keys.
- `Keymap` rejects bindings whose sequence and context overlap, and resolves
  active bindings by priority and command ID.
- `ShellContributions` aggregates menu, toolbar, and status-bar contributions
  with context filtering and stable ordering.

Every registry returns a registration token. Unregistering a stale token is a
no-op, so replacing a contribution cannot accidentally remove its replacement.
The crate contains no widgets, rendering code, event loop, or `bevy_*`
dependency. A Loom UI adapter can consume the returned models without changing
the shell semantics.

Run the standalone tests with:

```text
cargo test --manifest-path pkg/prism_editor_shell/Cargo.toml
cargo clippy --manifest-path pkg/prism_editor_shell/Cargo.toml --all-targets -- -D warnings
```
