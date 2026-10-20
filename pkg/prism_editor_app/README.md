# prism_editor_app

`prism_editor_app` assembles the editor data layers into a deterministic
application harness. Headless mode supports CI, golden snapshots, and
automation without opening a window or linking game logic. Windowed mode
requires explicit `WindowBackend` and `GpuSurfaceLifecycle` adapters; this
crate does not ship an operating-system window or renderer implementation.

The harness owns frame advancement, snapshot hashing, and resource lifecycle:

- Startup acquires windows, then the GPU context, then per-window surfaces in
  stable window-identity order. Partial startup and frame failures release
  acquired resources and expose the error through `last_error`.
- `tick` polls a bounded event batch, processes resize/suspend/resume/close,
  and presents active surfaces. A suspended window does not pause other active
  windows; zero-sized drawables suspend until a non-zero resize or resume.
- `stop` and `Drop` release surfaces in reverse creation order, then the GPU,
  then windows. Cleanup continues after a backend error; the first error is
  returned. Backends must make their own stop operation idempotent and clean up
  partial starts.
- `recover` rebuilds host resources after failure while retaining documents,
  selection, and other editor models. Replacing adapters and configuring
  windows are restricted to inactive lifecycle states.
- Headless runs have a frame limit, preserve a hook-requested exit reason, and
  report the actual number of frames advanced. Window count and event work
  have explicit budgets; a backend that violates the polling limit fails
  visibly rather than creating unbounded per-frame work.
- `build_prism_app` and `build_prism_app_with_backends` install the application
  resource and update system. Completion and failure propagate to
  `prism_app::AppExitRequest`.

Loom pixel rendering, device input translation, runtime transport, and
domain-specific modules remain explicit adapters. Surface lifecycle tests use
recording backends; they do not claim native window or GPU device validation.

Verification from the repository root:

```text
cargo test -p prism_editor_app --all-targets
cargo clippy -p prism_editor_app --all-targets --no-deps -- -D warnings
cargo run -p prism_editor_app -- --frames 4
```
