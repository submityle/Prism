# Prism Render Scene

This package currently contains two explicit integration boundaries:

* NativeRenderScene is the Prism-native retained-scene core. It has no Bevy API
  in its implementation and is installed into prism_app with
  NativeRenderScenePlugin.
* PrismGpuScenePlugin is the legacy Bevy render-world adapter. It remains a
  separate migration boundary until its extraction, GPU resource, and render
  scheduling paths are rewritten against prism_render_app and the Prism RHI.

The native vertical slice is:

    prism_ecs::Entity
      -> NativeRenderScene entity/SceneHandle mapping
      -> SceneTransaction / CpuRenderScene
      -> dirty UploadPlan
      -> architecture GPU table rows and GpuSceneSnapshot

Native scene handles are generational. Removing an entity retires its slot at
the caller-provided GPU completion value; reclaim_completed is the only path
that permits reuse. Transform updates within one frame preserve the first
pre-frame value as previous_transform.

The package manifest still has direct Bevy dependencies because the legacy
adapter modules are not yet migrated. The remaining direct dependencies are
listed in docs/prism_render_scene_native_migration_zh.md; no feature switch or
comment claims that the whole package is already Bevy-free.

This package integrates the retained Prism GPU Scene with Bevy's render app without modifying Bevy internals. Applications opt in by adding `PrismGpuScenePlugin`.

`PrismGpuSceneOpaquePlugin` is the first real raster consumer. Add it after Bevy PBR and the GPU Scene plugin to render opted-in opaque meshes directly from stable scene index/generation tables. It stays explicit while material parity matures; the base scene plugin never silently replaces Bevy PBR.

`GpuSceneMode::Compare` keeps the consumer enabled and exposes parity/queue counters through `GpuSceneDiagnostics`. It does not draw both paths into the same color target; visual A/B tooling should render separate views or captures so double depth/color writes cannot corrupt the comparison.

Set the render-world `GpuSceneDebugView` resource to `InstanceId`, `GeometryId`, `MaterialId`, or `Motion` to inspect the retained tables directly.
Set `GpuSceneOpaqueEnabled(false)` for a runtime fallback to the legacy Bevy PBR opaque path without disabling GPU Scene extraction for other consumers.

Unified visibility builds deterministic per-view draw bins and bounded indexed/non-indexed indirect command streams. `GpuSceneOpaqueIndirectEnabled` is deliberately disabled by default: enable it only for runtime parity testing on devices that report `INDIRECT_FIRST_INSTANCE`. Otherwise the opaque queue keeps its per-visible-item direct path. Asynchronous diagnostics compare GPU work, command classes, and per-bin command counts without stalling the current frame.

The HZB integration reuses Bevy's public persistent `ViewDepthPyramid`: before early depth it contains previous-frame data, and after early downsampling it contains current-frame data. Prism tracks epoch/mip validity, projects current/previous GPU Scene bounds into per-view candidates, dispatches both reverse-Z classifiers, and feeds their stage stream into visibility compaction. Early and late command generation have independent atomic headers and indexed/non-indexed streams; the late kernel emits only `LATE_RETEST | LATE_VISIBLE` candidates and resolves complete commands through the Geometry ABI. The opaque indirect consumer submits both disjoint streams only when visibility, HZB, and `INDIRECT_FIRST_INSTANCE` capability gates all pass, while asynchronous readback validates late counters, bin capacities, command classes, and overflow. `PrismVisibilityDiagnostics::hzb_graduation_ready` becomes true only after an enabled runtime parity frame has no mismatch, overflow, or readback failure. `hzb_occlusion` remains disabled by default until image parity and performance pass on real GPUs.

The package owns ECS integration, sparse GPU buffers, submission completion tracking, diagnostics, and the stable shader ABI. Unreal Engine-derived code is not stored here.

Opt a mesh into synchronization and the GPU Scene with:

```rust,ignore
commands.spawn((
    Mesh3d(mesh),
    PrismGpuScenePlugin::entity(PrismGpuSceneEntity::default()),
));
```
