# Prism Editor Domains

`prism_editor_domains` is the generic assembly boundary for editor domains. It
describes what a domain provides without pretending to implement a concrete VFX,
animation, rendering, graph-evaluation, or timeline algorithm.

## Contract

- `DomainId` and the typed contribution/profile IDs use a canonical lower-case
  ASCII form, making persistence and diagnostics stable across processes.
- `DomainVersion`, `CapabilityDescriptor`, and `VersionConstraint` express
  domain and capability compatibility without depending on a package manager.
- `EditorContribution` covers panel, command, graph-node, timeline, and
  validator descriptors. They are metadata contracts; adapters own behavior.
- `DomainRegistry` validates missing dependencies, version/capability mismatch,
  cross-domain contribution collisions, profile conflicts, and dependency
  cycles. It produces sorted diagnostics and deterministic start/stop orders.
- `DomainRuntime` applies dependency-first startup and dependent-first shutdown
  as a small, observable state machine. It performs no I/O and invokes no
  engine-specific callbacks.
- `DomainProfile` is optional. A descriptor can bundle one, or a profile can be
  registered separately after its owning domain is present. Conflicting profile
  declarations are rejected or diagnosed deterministically.

## Integration boundary

An editor shell, a renderer, an ECS/runtime adapter, or a project-specific
domain can consume these contracts. Those integrations should provide actual
panel factories, command handlers, node semantics, timeline evaluation, and
validator execution in their own crates. Keeping those concerns outside this
crate prevents the reusable editor base from hard-coding project gameplay or
engine policy.

## Verification

The crate is intentionally standalone so it can be checked without changing
the repository root workspace manifest:

```text
cargo fmt --manifest-path pkg/prism_editor_domains/Cargo.toml -- --check
cargo clippy --manifest-path pkg/prism_editor_domains/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path pkg/prism_editor_domains/Cargo.toml --all-targets
```
