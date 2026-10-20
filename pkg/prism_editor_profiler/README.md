# prism_editor_profiler

`prism_editor_profiler` provides bounded editor analysis models and real capture
producers without Bevy. The default CPU producer uses `std::time::Instant`.
Optional `gpu` uses the application's wgpu 30 device/queue for hardware
timestamp queries, resolve and asynchronous readback. It rejects devices
without enabled timestamp capabilities.

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

`capture::CpuProducer` and `gpu::GpuProducer` return the existing `Frame` type
for insertion into `FrameStore`. Live frames, span count/depth/text and GPU
readbacks have explicit bounds. Opaque capture generations fence stale handles.
Cancelled submitted GPU slots remain occupied until their callbacks drain;
repeated cancellation cannot grow an unbounded callback backlog. GPU frames
use a separate clock domain from CPU frames.

Tracy, diagnostic-counter and UI adapters remain external. The app can encode
actual passes through `GpuProducer::begin_render_pass` / `begin_compute_pass`, then call
`submit_frame` and nonblocking `poll`. Capture integration and limitations are
documented in `docs/prism_editor_profiler_capture_zh.md` at the repository root.

## Verification

```text
cargo test --manifest-path pkg/prism_editor_profiler/Cargo.toml
cargo clippy --manifest-path pkg/prism_editor_profiler/Cargo.toml --all-targets --all-features -- -D warnings
PRISM_REQUIRE_GPU=1 cargo test -p prism_editor_profiler --features gpu-tests --test gpu_capture -- --nocapture
```

`gpu-tests` always requires a real native GPU with the necessary timestamp
features; adapter/capability absence fails instead of silently skipping. It
checks real compute results, a render draw, nonzero query timings, cancellation,
slot reuse, empty-pass generation isolation, and explicit refusal on a device
without enabled timestamp features. GPU spans are flat pass intervals; frame
duration is their measured envelope. Each generation gets a fresh query set
because Metal may omit samples for empty passes without clearing old queries.
