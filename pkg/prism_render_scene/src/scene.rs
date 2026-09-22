use bevy_ecs::{entity::Entity, resource::Resource};
use prism_render_architecture::gpu_scene::{
    CpuRenderScene, GpuSceneSnapshot, SceneApplyReport, SceneCapacityError, SceneHandle,
    SceneHandleAllocator, SceneHandleError, SceneTransaction,
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

    pub(crate) fn reclaim_completed(
        &mut self,
        completed: prism_render_architecture::gpu_scene::GpuCompletionValue,
    ) -> u32 {
        self.allocator.reclaim_completed(completed)
    }
}
