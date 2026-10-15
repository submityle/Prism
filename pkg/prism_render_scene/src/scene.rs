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
    geometry_assets: HashMap<GeometryHandle, AssetId<Mesh>>,
    geometry_lods: HashMap<GeometryHandle, Vec<(AssetId<Mesh>, f32)>>,
    geometry_generations: HashMap<AssetId<Mesh>, u32>,
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
            geometry_assets: HashMap::new(),
            geometry_lods: HashMap::new(),
            geometry_generations: HashMap::new(),
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
        let generation = self.geometry_generations.get(&mesh).copied().unwrap_or(1);
        let handle = GeometryHandle {
            index: self.next_geometry_index,
            generation,
        };
        self.next_geometry_index = self
            .next_geometry_index
            .checked_add(1)
            .expect("GPU Scene geometry handle space exhausted");
        self.geometry.insert(mesh, handle);
        self.geometry_assets.insert(handle, mesh);
        handle
    }

    /// Binds the base asset without overwriting an existing identity.
    ///
    /// A geometry handle and a base asset are both stable identities. Reusing
    /// either side for a different owner would leave already extracted
    /// instances resolving to the wrong physical mesh, so conflicts are
    /// rejected and the previous mapping is preserved.
    pub fn bind_geometry_mesh(&mut self, handle: GeometryHandle, mesh: AssetId<Mesh>) -> bool {
        if self
            .geometry_assets
            .get(&handle)
            .is_some_and(|mapped| *mapped != mesh)
            || self
                .geometry
                .get(&mesh)
                .is_some_and(|mapped| *mapped != handle)
        {
            return false;
        }
        self.geometry_assets.insert(handle, mesh);
        self.geometry.insert(mesh, handle);
        true
    }

    pub fn bind_geometry_lod_meshes(
        &mut self,
        handle: GeometryHandle,
        meshes: &[(AssetId<Mesh>, f32)],
    ) -> bool {
        let base = self.geometry_assets.get(&handle).copied();
        if base.is_none()
            || meshes.iter().any(|(asset, _)| {
                self.geometry
                    .get(asset)
                    .is_some_and(|mapped| *mapped != handle)
                    || self.geometry_lods.iter().any(|(owner, lods)| {
                        *owner != handle && lods.iter().any(|(mapped, _)| mapped == asset)
                    })
            })
        {
            return false;
        }
        let mut seen = std::collections::HashSet::new();
        self.geometry_lods.insert(
            handle,
            meshes
                .iter()
                .copied()
                .filter(|(asset, error)| {
                    base != Some(*asset) && error.is_finite() && seen.insert(*asset)
                })
                .collect(),
        );
        true
    }

    pub(crate) fn retire_geometry_lod(&mut self, mesh: AssetId<Mesh>) -> bool {
        let mut changed = false;
        for lods in self.geometry_lods.values_mut() {
            let before = lods.len();
            lods.retain(|(asset, _)| *asset != mesh);
            changed |= before != lods.len();
        }
        changed
    }

    pub fn retire_geometry(&mut self, mesh: AssetId<Mesh>) -> Option<GeometryHandle> {
        let retired = self.geometry.remove(&mesh)?;
        self.geometry_assets.remove(&retired);
        self.geometry_lods.remove(&retired);
        self.geometry_generations
            .insert(mesh, retired.generation.saturating_add(1));
        Some(retired)
    }

    pub fn geometry_handle(&self, mesh: AssetId<Mesh>) -> Option<GeometryHandle> {
        self.geometry.get(&mesh).copied()
    }

    pub(crate) fn geometry_assets(
        &self,
    ) -> impl Iterator<Item = (AssetId<Mesh>, GeometryHandle)> + '_ {
        self.geometry_assets
            .iter()
            .map(|(handle, asset)| (*asset, *handle))
    }

    pub(crate) fn geometry_lod_assets(
        &self,
        handle: GeometryHandle,
    ) -> impl Iterator<Item = (u32, AssetId<Mesh>, f32)> + '_ {
        self.geometry_lods
            .get(&handle)
            .into_iter()
            .flat_map(|lods| lods.iter().enumerate())
            .filter_map(|(index, (asset, screen_error))| {
                Some((
                    u32::try_from(index).ok()?.saturating_add(1),
                    *asset,
                    *screen_error,
                ))
            })
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

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_asset::uuid::Uuid;

    fn mesh(id: u128) -> AssetId<Mesh> {
        AssetId::Uuid {
            uuid: Uuid::from_u128(id),
        }
    }

    #[test]
    fn geometry_binding_rejects_handle_and_asset_aliasing() {
        let mut scene = RenderGpuScene::new(8);
        let first = GeometryHandle::new(1, 1);
        let second = GeometryHandle::new(2, 1);
        let base = mesh(1);

        assert!(scene.bind_geometry_mesh(first, base));
        assert!(!scene.bind_geometry_mesh(second, base));
        assert!(!scene.bind_geometry_mesh(first, mesh(2)));
        assert_eq!(scene.geometry_handle(base), Some(first));
    }

    #[test]
    fn lod_binding_rejects_shared_assets_and_preserves_owner() {
        let mut scene = RenderGpuScene::new(8);
        let first = GeometryHandle::new(1, 1);
        let second = GeometryHandle::new(2, 1);
        let shared_lod = mesh(3);

        assert!(scene.bind_geometry_mesh(first, mesh(1)));
        assert!(scene.bind_geometry_mesh(second, mesh(2)));
        assert!(scene.bind_geometry_lod_meshes(first, &[(shared_lod, 1.0)]));
        assert!(!scene.bind_geometry_lod_meshes(second, &[(shared_lod, 1.0)]));
        assert_eq!(scene.geometry_lod_assets(first).count(), 1);
        assert_eq!(scene.geometry_lod_assets(second).count(), 0);
    }
}
