# prism_editor_profiler

`prism_editor_profiler` is a pure data layer for editor analysis. It models
bounded frame/span captures, render-frame graphs, memory budgets, and
structured visual-log events without opening Tracy/GPU connections or
depending on an engine runtime.

The crate deliberately uses integer time and deterministic ordered
collections. Inputs are validated against `ProfilerLimits`; names, graph
sizes, span depth, samples, events, and projected output cannot grow without
an explicit bound. Stable tie-breakers are part of the public behavior, so a
remote capture and a local capture produce the same ordering when their data
matches.

## Data model

- `frame` validates frame-local span ranges, builds a stable span tree, and
  stores a bounded ordered frame timeline.
- `frame_graph` validates pass/node dependencies and computes a deterministic
  topological order and weighted critical path.
- `memory` records category samples and turns warning/over-budget thresholds
  into deterministic alerts.
- `visual_log` stores structured primitives and projects intersecting events
  into stable tracks for a timeline panel.

Adapters for Tracy, GPU timestamps, `prism_diagnostic`, or a UI belong outside
this crate. This package only accepts already-normalized data.

## Verification

```text
cargo test --manifest-path pkg/prism_editor_profiler/Cargo.toml
cargo clippy --manifest-path pkg/prism_editor_profiler/Cargo.toml --all-targets -- -D warnings
```
