use bevy_ecs::{prelude::*, system::SystemParam};
use bevy_render::render_resource::{BindGroup, BindGroupLayout, Buffer};
use prism_render_architecture::gpu_scene::{GpuSceneSnapshot, SceneHandle};

use crate::{
    buffers::{GpuSceneBindGroup, GpuSceneBuffers},
    extract::ExtractedSceneInstance,
    scene::RenderGpuScene,
};

/// Read-only ECS system parameter for raster, shadow, meshlet, and GI clients.
#[derive(SystemParam)]
pub struct GpuSceneReader<'w, 's> {
    scene: Res<'w, RenderGpuScene>,
    buffers: Res<'w, GpuSceneBuffers>,
    bindings: Res<'w, GpuSceneBindGroup>,
    instances: Query<'w, 's, &'static ExtractedSceneInstance>,
}

impl GpuSceneReader<'_, '_> {
    pub fn snapshot(&self) -> GpuSceneSnapshot {
        self.scene.snapshot()
    }

    pub fn handle(&self, render_entity: Entity) -> Option<SceneHandle> {
        self.scene
            .handle_from_component(self.instances.get(render_entity).ok()?)
    }

    pub fn layout(&self) -> &BindGroupLayout {
        &self.bindings.layout
    }

    pub fn bind_group(&self) -> Option<&BindGroup> {
        self.bindings.bind_group.as_ref()
    }

    pub fn buffers(&self) -> Option<GpuSceneBufferBindings<'_>> {
        Some(GpuSceneBufferBindings {
            instances: self.buffers.instances()?,
            current_transforms: self.buffers.current_transforms()?,
            previous_transforms: self.buffers.previous_transforms()?,
            bounds: self.buffers.bounds()?,
        })
    }
}

/// Raw buffers for clients that need a custom bind-group layout.
pub struct GpuSceneBufferBindings<'a> {
    pub instances: &'a Buffer,
    pub current_transforms: &'a Buffer,
    pub previous_transforms: &'a Buffer,
    pub bounds: &'a Buffer,
}
