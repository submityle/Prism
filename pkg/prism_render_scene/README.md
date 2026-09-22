# Prism Render Scene

This package integrates the retained Prism GPU Scene with Bevy's render app without modifying Bevy internals. Applications opt in by adding `PrismGpuScenePlugin`.

The package owns ECS integration, sparse GPU buffers, submission completion tracking, diagnostics, and the stable shader ABI. Unreal Engine-derived code is not stored here.

Opt a mesh into synchronization and the GPU Scene with:

```rust,ignore
commands.spawn((
    Mesh3d(mesh),
    PrismGpuScenePlugin::entity(PrismGpuSceneEntity::default()),
));
```
