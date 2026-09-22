# Prism Render Scene

This package integrates the retained Prism GPU Scene with Bevy's render app without modifying Bevy internals. Applications opt in by adding `PrismGpuScenePlugin`.

`PrismGpuSceneOpaquePlugin` is the first real raster consumer. Add it after Bevy PBR and the GPU Scene plugin to render opted-in opaque meshes directly from stable scene index/generation tables. It stays explicit while material parity matures; the base scene plugin never silently replaces Bevy PBR.

The package owns ECS integration, sparse GPU buffers, submission completion tracking, diagnostics, and the stable shader ABI. Unreal Engine-derived code is not stored here.

Opt a mesh into synchronization and the GPU Scene with:

```rust,ignore
commands.spawn((
    Mesh3d(mesh),
    PrismGpuScenePlugin::entity(PrismGpuSceneEntity::default()),
));
```
