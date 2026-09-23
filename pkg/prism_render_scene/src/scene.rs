use bevy_asset::AssetId;
use bevy_ecs::{entity::Entity, resource::Resource};
use bevy_mesh::Mesh;
use prism_render_architecture::gpu_scene::{
    CpuRenderScene, GeometryHandle, GpuSceneSnapshot, SceneApplyReport, SceneCapacityError,
    SceneHandle, SceneHandleAllocator, SceneHandleError, SceneTransaction,
};
use std::collections::HashMap;

use crate::{
    buffers::GpuSceneBuffers, completion::GpuCompletionTracker, extract::ExtractedSceneInstance,
};

const DEFAULT_MAX_SCENE_SLOTS: u32 = 1 << 24;

/// Owns stable handles, the CPU mirror, and the last published snapshot.
#[derive(Resource)]
pub struct RenderGpuScene {
    allocator: SceneHandleAllocator,
    mirror: CpuRenderScene,
    snapshot: GpuSceneSnapshot,
    buffer_version: u32,
    entities: HashMap<Entity, SceneHandle>,
    main_entities: HashMap<Entity, bevy_render::sync_world::MainEntity>,
    geometry: HashMap<AssetId<Mesh>, GeometryHandle>,
    next_geometry_index: u32,
    material_generations: HashMap<u32, (u32, bool)>,
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
            main_entities: HashMap::new(),
            geometry: HashMap::new(),
            next_geometry_index: 1,
            material_generations: HashMap::new(),
        }
    }

    pub fn allocate(&mut self) -> Result<SceneHandle, SceneCapacityError> {
        self.allocator.allocate()
    }

    pub fn cancel_allocation(&mut self, handle: SceneHandle) -> Result<(), SceneHandleError> {
        self.allocator.cancel_allocation(handle)
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

    pub fn bind_entity(
        &mut self,
        entity: Entity,
        main_entity: bevy_render::sync_world::MainEntity,
        handle: SceneHandle,
    ) {
        self.entities.insert(entity, handle);
        self.main_entities.insert(entity, main_entity);
    }

    pub fn remove_entity(&mut self, entity: Entity) -> Option<SceneHandle> {
        self.main_entities.remove(&entity);
        self.entities.remove(&entity)
    }

    pub fn handle_for_entity(&self, entity: Entity) -> Option<SceneHandle> {
        self.entities.get(&entity).copied()
    }

    pub fn entity_binding_for_handle(
        &self,
        handle: SceneHandle,
    ) -> Option<(Entity, bevy_render::sync_world::MainEntity)> {
        self.entities.iter().find_map(|(entity, candidate)| {
            (*candidate == handle)
                .then(|| {
                    self.main_entities
                        .get(entity)
                        .copied()
                        .map(|main| (*entity, main))
                })
                .flatten()
        })
    }

    pub fn handle_from_component(&self, extracted: &ExtractedSceneInstance) -> Option<SceneHandle> {
        extracted
            .handle
            .filter(|handle| self.mirror.get(*handle).is_some())
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

    pub fn retire_geometry(&mut self, mesh: AssetId<Mesh>) -> Option<GeometryHandle> {
        self.geometry.remove(&mesh)
    }

    pub fn geometry_handle(&self, mesh: AssetId<Mesh>) -> Option<GeometryHandle> {
        self.geometry.get(&mesh).copied()
    }

    /// Registers an externally-owned material row and rejects stale reuse.
    pub fn register_material(
        &mut self,
        handle: prism_render_architecture::gpu_scene::SceneMaterialHandle,
    ) -> bool {
        if !handle.is_valid() || handle.index == 0 {
            return false;
        }
        match self.material_generations.get(&handle.index) {
            Some((generation, _)) if *generation >= handle.generation => false,
            _ => {
                self.material_generations
                    .insert(handle.index, (handle.generation, true));
                true
            }
        }
    }

    pub fn material_is_current(
        &self,
        handle: prism_render_architecture::gpu_scene::SceneMaterialHandle,
    ) -> bool {
        self.material_generations.get(&handle.index) == Some(&(handle.generation, true))
    }

    pub fn retire_material(
        &mut self,
        handle: prism_render_architecture::gpu_scene::SceneMaterialHandle,
    ) -> bool {
        if !self.material_is_current(handle) {
            return false;
        }
        self.material_generations
            .insert(handle.index, (handle.generation, false));
        true
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

    pub fn buffer_version(&self) -> u32 {
        self.buffer_version
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
