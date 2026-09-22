use bevy_asset::AssetId;
use bevy_ecs::{entity::Entity, resource::Resource};
use bevy_mesh::Mesh;
use prism_render_architecture::gpu_scene::{
    CpuRenderScene, GeometryHandle, GpuSceneSnapshot, SceneApplyReport, SceneCapacityError,
    SceneHandle, SceneHandleAllocator, SceneHandleError, SceneTransaction,
};
use std::collections::HashMap;

use crate::{buffers::GpuSceneBuffers, completion::GpuCompletionTracker};

const DEFAULT_MAX_SCENE_SLOTS: u32 = 1 << 24;

/// Owns stable handles, the CPU mirror, and the last published snapshot.
#[derive(Resource)]
pub struct RenderGpuScene {
    allocator: SceneHandleAllocator,
    mirror: CpuRenderScene,
    snapshot: GpuSceneSnapshot,
    buffer_version: u32,
    entities: HashMap<Entity, SceneHandle>,
    geometry: HashMap<AssetId<Mesh>, GeometryHandle>,
    next_geometry_index: u32,
}

impl Default for RenderGpuScene {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_SCENE_SLOTS)
    }
}

impl RenderGpuScene {
    pub fn new(max_slots: u32) -> Self {
        Self {
            allocator: SceneHandleAllocator::new(max_slots),
            mirror: CpuRenderScene::default(),
            snapshot: GpuSceneSnapshot::default(),
            buffer_version: 1,
            entities: HashMap::new(),
            geometry: HashMap::new(),
            next_geometry_index: 1,
        }
    }

    pub fn allocate(&mut self) -> Result<SceneHandle, SceneCapacityError> {
        self.allocator.allocate()
    }

    pub fn apply_transaction(
        &mut self,
        buffers: &mut GpuSceneBuffers,
        transaction: &SceneTransaction,
    ) -> SceneApplyReport {
        let report = self.mirror.apply(transaction);
        if report.errors.is_empty() {
            buffers.apply_dirty_slots(&self.mirror, &report.dirty_slots);
            self.snapshot = GpuSceneSnapshot {
                frame_epoch: transaction.frame_epoch,
                scene_epoch: self.mirror.scene_epoch(),
                instance_count: self.mirror.live_count(),
                buffer_version: self.buffer_version,
            };
        }
        report
    }

    pub fn apply_entity_transaction(
        &mut self,
        buffers: &mut GpuSceneBuffers,
        transaction: &SceneTransaction,
    ) -> SceneApplyReport {
        self.apply_transaction(buffers, transaction)
    }

    pub fn bind_entity(&mut self, entity: Entity, handle: SceneHandle) {
        self.entities.insert(entity, handle);
    }

    pub fn remove_entity(&mut self, entity: Entity) -> Option<SceneHandle> {
        self.entities.remove(&entity)
    }

    pub fn handle_for_entity(&self, entity: Entity) -> Option<SceneHandle> {
        self.entities.get(&entity).copied()
    }

    pub fn geometry_for_mesh(&mut self, mesh: AssetId<Mesh>) -> GeometryHandle {
        if let Some(handle) = self.geometry.get(&mesh) {
            return *handle;
        }
        let handle = GeometryHandle {
            index: self.next_geometry_index,
            generation: 1,
        };
        self.next_geometry_index = self
            .next_geometry_index
            .checked_add(1)
            .expect("GPU Scene geometry handle space exhausted");
        self.geometry.insert(mesh, handle);
        handle
    }

    pub fn retire(
        &mut self,
        handle: SceneHandle,
        completion: &GpuCompletionTracker,
    ) -> Result<(), SceneHandleError> {
        self.allocator
            .retire(handle, completion.next_submission_value())
    }

    pub fn snapshot(&self) -> GpuSceneSnapshot {
        self.snapshot
    }

    pub fn mirror(&self) -> &CpuRenderScene {
        &self.mirror
    }

    /// Rebuilds all GPU tables from the authoritative CPU mirror after device
    /// loss or backend recreation.
    pub fn rebuild_gpu_buffers(&mut self, buffers: &mut GpuSceneBuffers) {
        buffers.rebuild_from_mirror(&self.mirror);
        self.buffer_version = self.buffer_version.wrapping_add(1).max(1);
        self.snapshot.buffer_version = self.buffer_version;
        self.snapshot.scene_epoch = self.mirror.scene_epoch();
        self.snapshot.instance_count = self.mirror.live_count();
    }

    pub(crate) fn reclaim_completed(
        &mut self,
        completed: prism_render_architecture::gpu_scene::GpuCompletionValue,
    ) -> u32 {
        self.allocator.reclaim_completed(completed)
    }
}
