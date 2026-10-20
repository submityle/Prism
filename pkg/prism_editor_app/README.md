# prism_editor_app

`prism_editor_app` assembles the editor data layers into a deterministic
application harness. Headless mode supports CI, golden snapshots, and
automation without opening a window or linking game logic. With the `desktop`
feature, the binary also exposes a native winit/wgpu smoke entry point and the
library exposes the Loom presenter plus the engine-neutral viewport seam.

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

Document persistence is opt-in through `DesktopDocumentSession` and
`run_desktop_with_document`; the session preserves dirty/draft state and
fences stale save/load receipts. Runtime transport and domain-specific modules
remain explicit adapters. Surface lifecycle tests use recording backends; the
native smoke entry point is the path for actual OS window/GPU validation.

For document-driven native scenes, `DesktopScene::new_with_document_adapter`
accepts a host-owned `SceneDocumentAdapter`. The adapter converts the validated
`EditorController` candidate into indexed geometry, full selection identities,
and a camera without embedding project gameplay types. Scene candidates are
staged before UI publication and committed only after the document install
succeeds; an install failure discards the candidate and keeps the active scene.

Verification from the repository root:

```text
cargo test -p prism_editor_app --all-targets
cargo clippy -p prism_editor_app --all-targets --no-deps -- -D warnings
cargo run -p prism_editor_app -- --frames 4
cargo run -p prism_editor_app --features desktop -- --desktop --frames 4
```
