# Prism Render Scene

This package integrates the retained Prism GPU Scene with Bevy's render app without modifying Bevy internals. Applications opt in by adding `PrismGpuScenePlugin`.

`PrismGpuSceneOpaquePlugin` is the first real raster consumer. Add it after Bevy PBR and the GPU Scene plugin to render opted-in opaque meshes directly from stable scene index/generation tables. It stays explicit while material parity matures; the base scene plugin never silently replaces Bevy PBR.

`GpuSceneMode::Compare` keeps the consumer enabled and exposes parity/queue counters through `GpuSceneDiagnostics`. It does not draw both paths into the same color target; visual A/B tooling should render separate views or captures so double depth/color writes cannot corrupt the comparison.

Set the render-world `GpuSceneDebugView` resource to `InstanceId`, `GeometryId`, `MaterialId`, or `Motion` to inspect the retained tables directly.
Set `GpuSceneOpaqueEnabled(false)` for a runtime fallback to the legacy Bevy PBR opaque path without disabling GPU Scene extraction for other consumers.

Unified visibility builds deterministic per-view draw bins and bounded indexed/non-indexed indirect command streams. `GpuSceneOpaqueIndirectEnabled` is deliberately disabled by default: enable it only for runtime parity testing on devices that report `INDIRECT_FIRST_INSTANCE`. Otherwise the opaque queue keeps its per-visible-item direct path. Asynchronous diagnostics compare GPU work, command classes, and per-bin command counts without stalling the current frame.

The HZB integration reuses Bevy's public persistent `ViewDepthPyramid`: before early depth it contains previous-frame data, and after early downsampling it contains current-frame data. Prism tracks epoch/mip validity and schedules both seams without cloning texture views as fake snapshots. `hzb_occlusion` remains disabled by default until the actual GPU sample/late-compaction kernels and runtime parity gate are complete.

The package owns ECS integration, sparse GPU buffers, submission completion tracking, diagnostics, and the stable shader ABI. Unreal Engine-derived code is not stored here.

Opt a mesh into synchronization and the GPU Scene with:

```rust,ignore
commands.spawn((
    Mesh3d(mesh),
    PrismGpuScenePlugin::entity(PrismGpuSceneEntity::default()),
));
```
