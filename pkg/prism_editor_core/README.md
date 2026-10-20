# prism_editor_core

`prism_editor_core` is the renderer-, UI-, runtime-, and engine-independent
editor kernel. It owns the reusable state transitions that must remain stable
across a desktop host, headless automation, and remote adapters.

## Core contracts

- `DocumentState` manages ordered base, override, and runtime layers together
  with active-layer selection, revision tracking, and dirty state.
- `Command`, `Transaction`, and `CommandStack` provide reversible edits,
  failure rollback, merge hooks, and deterministic undo/redo history.
- `Selection` stores ordered, duplicate-free entity, asset, and sub-object
  identities with an explicit primary item.
- `ServiceRegistry` stores type-keyed `Send + Sync` services, while
  `ContributionRegistry` manages disposable command and panel metadata.
- All collections use deterministic ordering where their contents are exposed
  or used for identity-sensitive editor decisions.

The crate contains no UI, renderer, filesystem, network, ECS, or engine
runtime policy. Those concerns belong to adapters above this kernel, and the
kernel has no `bevy_*` dependencies.

## Verification

```text
cargo fmt --manifest-path pkg/prism_editor_core/Cargo.toml -- --check
cargo clippy -p prism_editor_core --all-targets -- -D warnings
cargo test -p prism_editor_core --all-targets
```
