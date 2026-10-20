use std::collections::BTreeMap;

use prism_ecs::{entity::Entity, resource::Resource};
use prism_render_architecture::gpu_scene::{
    CpuRenderScene, GeometryHandle, SceneApplyReport, SceneHandle, SceneMaterialHandle,
    SceneTransaction,
};
use prism_render_material::MaterialRecord;
use prism_render_visibility::{
    GeometryLodChain, GpuRenderWorkItem, GpuViewRecord, ViewHandle, VisibilityDiagnostics,
};
use prism_transform::ecs::TransformEcsError;

/// Most recent cull result for one camera. Work stays associated with its view.
pub struct PrismViewResult {
    /// Current parameters and previous clip matrix supplied to `cull_view`.
    pub view: GpuViewRecord,
    /// History epoch used for this cull, or None for a newly created camera.
    pub previous_history_epoch: Option<u64>,
    /// Emitted work, also used to write the per-entity union boolean.
    pub work: Vec<GpuRenderWorkItem>,
    /// Rejections, missing resources and capacity overflow from the core culler.
    pub diagnostics: VisibilityDiagnostics,
}

/// Native entity mirror and culling registries. No GPU resources are required.
/// Scene mutation is private so entity mappings and generational slots agree.
pub struct PrismRenderScene {
    pub(super) scene: CpuRenderScene,
    pub(super) geometry: BTreeMap<GeometryHandle, GeometryLodChain>,
    pub(super) materials: BTreeMap<SceneMaterialHandle, MaterialRecord>,
    pub(super) entities: BTreeMap<Entity, SceneHandle>,
    pub(super) owners: BTreeMap<SceneHandle, Entity>,
    pub(super) views: BTreeMap<Entity, PrismViewResult>,
    pub(super) previous_lods: BTreeMap<(ViewHandle, SceneHandle), u16>,
    pub(super) free: Vec<SceneHandle>,
    pub(super) next_index: u32,
    pub(super) next_view: u32,
    pub(super) frame: u64,
    geometry_epoch: u64,
    pub(super) capacity: u32,
    pub(super) transaction: SceneTransaction,
    pub(super) report: SceneApplyReport,
    pub(super) transform_error: Option<TransformEcsError>,
}
impl Resource for PrismRenderScene {}
impl Default for PrismRenderScene {
    fn default() -> Self {
        Self {
            scene: CpuRenderScene::default(),
            geometry: BTreeMap::new(),
            materials: BTreeMap::new(),
            entities: BTreeMap::new(),
            owners: BTreeMap::new(),
            views: BTreeMap::new(),
            previous_lods: BTreeMap::new(),
            free: Vec::new(),
            next_index: 1,
            next_view: 0,
            frame: 0,
            geometry_epoch: 0,
            capacity: 1 << 20,
            transaction: SceneTransaction::default(),
            report: SceneApplyReport::default(),
            transform_error: None,
        }
    }
}
impl PrismRenderScene {
    /// Register or replace a LOD chain. Replacements invalidate its LOD history.
    pub fn insert_geometry(&mut self, chain: GeometryLodChain) {
        let geometry = chain.geometry;
        let changed = self
            .geometry
            .get(&geometry)
            .is_none_or(|current| current != &chain);
        self.invalidate_geometry(geometry);
        if changed {
            self.geometry.insert(geometry, chain);
            self.bump_geometry_epoch();
        }
    }
    /// Unregister geometry; subsequent culls report missing geometry.
    pub fn remove_geometry(&mut self, handle: GeometryHandle) -> Option<GeometryLodChain> {
        self.invalidate_geometry(handle);
        let removed = self.geometry.remove(&handle);
        if removed.is_some() {
            self.bump_geometry_epoch();
        }
        removed
    }
    /// Register or replace a material, consumed afresh by every cull.
    pub fn insert_material(&mut self, record: MaterialRecord) {
        self.materials.insert(record.handle, record);
    }
    /// Unregister material; subsequent culls report missing material.
    pub fn remove_material(&mut self, handle: SceneMaterialHandle) -> Option<MaterialRecord> {
        self.materials.remove(&handle)
    }
    /// Maximum emitted work items per view, including zero for diagnostic use.
    pub fn set_view_capacity(&mut self, capacity: u32) {
        self.capacity = capacity;
    }
    /// Read-only CPU scene for upload, capture or inspection.
    pub fn scene(&self) -> &CpuRenderScene {
        &self.scene
    }
    /// Monotonically changing geometry-table epoch; zero is reserved for the initial state.
    pub fn geometry_epoch(&self) -> u64 {
        self.geometry_epoch
    }
    /// Iterate the currently registered geometry records without exposing mutation.
    pub fn geometry_records(&self) -> impl Iterator<Item = &GeometryLodChain> {
        self.geometry.values()
    }
    /// Number of complete renderable entities mirrored this frame.
    pub fn instance_count(&self) -> usize {
        self.entities.len()
    }
    /// Number of complete camera entities culled this frame.
    pub fn view_count(&self) -> usize {
        self.views.len()
    }
    /// Stable scene identity until an entity loses a prerequisite or despawns.
    pub fn handle_for(&self, entity: Entity) -> Option<SceneHandle> {
        self.entities.get(&entity).copied()
    }
    /// Resolve a live scene identity, including its generation.
    pub fn entity_for(&self, handle: SceneHandle) -> Option<Entity> {
        self.owners.get(&handle).copied()
    }
    /// Current camera result; absent cameras retain neither work nor history.
    pub fn view_result(&self, entity: Entity) -> Option<&PrismViewResult> {
        self.views.get(&entity)
    }
    /// Iterate live camera results in generational entity order.
    pub fn view_results(&self) -> impl Iterator<Item = (Entity, &PrismViewResult)> {
        self.views.iter().map(|(&entity, result)| (entity, result))
    }
    /// LOD history written from the most recent emitted work.
    pub fn previous_lod(&self, view: ViewHandle, scene: SceneHandle) -> Option<u16> {
        self.previous_lods.get(&(view, scene)).copied()
    }
    /// Number of retained view/instance LOD pairs.
    pub fn history_count(&self) -> usize {
        self.previous_lods.len()
    }
    /// Last synchronization transaction, empty on an unchanged static frame.
    pub fn last_transaction(&self) -> &SceneTransaction {
        &self.transaction
    }
    /// Scene apply report. Internal invariant failures are never discarded.
    pub fn last_apply_report(&self) -> &SceneApplyReport {
        &self.report
    }
    /// A malformed transform hierarchy clears visibility until propagation recovers.
    pub fn last_transform_error(&self) -> Option<&TransformEcsError> {
        self.transform_error.as_ref()
    }

    fn invalidate_geometry(&mut self, geometry: GeometryHandle) {
        let scene = &self.scene;
        self.previous_lods.retain(|&(_, handle), _| {
            scene
                .get(handle)
                .is_some_and(|record| record.geometry != geometry)
        });
    }
    fn bump_geometry_epoch(&mut self) {
        self.geometry_epoch = self.geometry_epoch.wrapping_add(1);
        if self.geometry_epoch == 0 {
            self.geometry_epoch = 1;
        }
    }
    pub(super) fn forget_instance(&mut self, handle: SceneHandle) {
        self.previous_lods
            .retain(|&(_, instance), _| instance != handle);
    }
    pub(super) fn allocate(&mut self) -> SceneHandle {
        if let Some(handle) = self.free.pop() {
            return handle;
        }
        let index = self.next_index;
        self.next_index = index
            .checked_add(1)
            .expect("visibility scene index exhausted");
        SceneHandle::new(index, 1)
    }
    pub(super) fn retire(&mut self, handle: SceneHandle) {
        if let Some(generation) = handle.generation.checked_add(1) {
            self.free.push(SceneHandle::new(handle.index, generation));
        }
    }
    pub(super) fn allocate_view(&mut self) -> ViewHandle {
        let index = self.next_view;
        self.next_view = index
            .checked_add(1)
            .expect("visibility view index exhausted");
        ViewHandle::new(index, 1)
    }
}
