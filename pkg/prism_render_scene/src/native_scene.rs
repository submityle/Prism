//! Prism-native retained scene core.
//!
//! This module is independent from the legacy Bevy adapter. It maps the
//! authoritative Prism ECS entity to a generational scene handle and drives
//! the architecture-layer allocator, CPU mirror, upload planner, and ABI rows
//! as one atomic path.

use std::collections::{BTreeMap, BTreeSet};

use prism_ecs::{entity::Entity, resource::Resource};
use prism_render_architecture::gpu_scene::{
    bounds_row, current_transform_row, instance_row, previous_transform_row, CpuRenderScene,
    DirtySceneSlot, GpuCompletionValue, GpuSceneBounds, GpuSceneInstance, GpuSceneSnapshot,
    GpuSceneTransform, InstanceRecord, SceneApplyError, SceneApplyReport, SceneBounds,
    SceneCapacityError, SceneFieldMask, SceneHandle, SceneHandleAllocator, SceneHandleError,
    SceneMaterialHandle, SceneOperation, SceneTransactionBuilder, SceneTransform, UploadBudget,
    UploadPlan, UploadPlanner,
};

/// Maximum scene slots used by the default native plugin.
pub const DEFAULT_NATIVE_SCENE_CAPACITY: u32 = 1_048_576;

/// Failure returned by the entity-facing native scene API.
#[derive(Debug, thiserror::Error)]
pub enum NativeSceneError {
    #[error("entity {0:?} is already registered in the native render scene")]
    DuplicateEntity(Entity),
    #[error("entity {0:?} is not registered in the native render scene")]
    UnknownEntity(Entity),
    #[error(transparent)]
    Capacity(#[from] SceneCapacityError),
    #[error(transparent)]
    Handle(#[from] SceneHandleError),
    #[error(transparent)]
    Apply(#[from] SceneApplyError),
}

/// Result of a batch insertion, preserving input order.
#[derive(Debug)]
pub struct NativeSceneInsertBatch {
    pub handles: Vec<SceneHandle>,
    pub report: SceneApplyReport,
}

/// Prism-native retained render scene resource.
pub struct NativeRenderScene {
    allocator: SceneHandleAllocator,
    entity_handles: BTreeMap<Entity, SceneHandle>,
    cpu: CpuRenderScene,
    upload_planner: UploadPlanner,
    // Keyed by handle so repeated updates coalesce in O(log n) instead of
    // linearly scanning every dirty slot. BTreeMap also preserves the
    // deterministic handle order expected by upload planning and captures.
    dirty_slots: BTreeMap<SceneHandle, SceneFieldMask>,
    frame_epoch: u64,
    next_sequence: u64,
    buffer_version: u32,
}

impl Resource for NativeRenderScene {}

impl NativeRenderScene {
    /// Create an empty native scene with the default upload budget.
    pub fn new(max_slots: u32) -> Self {
        Self::with_budget(max_slots, UploadBudget::default())
    }

    /// Create an empty native scene with an explicit upload budget.
    pub fn with_budget(max_slots: u32, budget: UploadBudget) -> Self {
        Self {
            allocator: SceneHandleAllocator::new(max_slots),
            entity_handles: BTreeMap::new(),
            cpu: CpuRenderScene::default(),
            upload_planner: UploadPlanner::new(budget),
            dirty_slots: BTreeMap::new(),
            frame_epoch: 0,
            next_sequence: 0,
            buffer_version: 0,
        }
    }

    /// Set the authoritative frame epoch used by transform history.
    pub fn begin_frame(&mut self, frame_epoch: u64) {
        self.frame_epoch = frame_epoch;
    }

    pub fn frame_epoch(&self) -> u64 {
        self.frame_epoch
    }

    /// Insert one ECS entity and publish its initial instance record.
    pub fn insert(
        &mut self,
        entity: Entity,
        record: InstanceRecord,
    ) -> Result<SceneHandle, NativeSceneError> {
        if self.entity_handles.contains_key(&entity) {
            return Err(NativeSceneError::DuplicateEntity(entity));
        }
        let handle = self.allocator.allocate()?;
        if let Err(error) = self.apply_operations([SceneOperation::Create { handle, record }]) {
            let _ = self.allocator.cancel_allocation(handle);
            return Err(error);
        }
        self.entity_handles.insert(entity, handle);
        Ok(handle)
    }

    /// Insert several entities atomically. Handles preserve input order.
    pub fn insert_batch<I>(
        &mut self,
        entries: I,
    ) -> Result<NativeSceneInsertBatch, NativeSceneError>
    where
        I: IntoIterator<Item = (Entity, InstanceRecord)>,
    {
        let entries: Vec<_> = entries.into_iter().collect();
        let mut entities = BTreeSet::new();
        for (entity, _) in &entries {
            if self.entity_handles.contains_key(entity) || !entities.insert(*entity) {
                return Err(NativeSceneError::DuplicateEntity(*entity));
            }
        }
        let handles = self.allocator.allocate_batch(entries.len())?;
        let operations =
            entries
                .iter()
                .zip(handles.iter().copied())
                .map(|((_, record), handle)| SceneOperation::Create {
                    handle,
                    record: *record,
                });
        let report = match self.apply_operations(operations) {
            Ok(report) => report,
            Err(error) => {
                for handle in handles.iter().copied() {
                    let _ = self.allocator.cancel_allocation(handle);
                }
                return Err(error);
            }
        };
        for ((entity, _), handle) in entries.into_iter().zip(handles.iter().copied()) {
            self.entity_handles.insert(entity, handle);
        }
        Ok(NativeSceneInsertBatch { handles, report })
    }

    pub fn handle(&self, entity: Entity) -> Option<SceneHandle> {
        self.entity_handles.get(&entity).copied()
    }

    pub fn set_transform(
        &mut self,
        entity: Entity,
        current: SceneTransform,
    ) -> Result<(), NativeSceneError> {
        let handle = self.require_handle(entity)?;
        self.apply_operations([SceneOperation::SetTransform { handle, current }])
            .map(|_| ())
    }

    pub fn set_bounds(
        &mut self,
        entity: Entity,
        bounds: SceneBounds,
    ) -> Result<(), NativeSceneError> {
        let handle = self.require_handle(entity)?;
        self.apply_operations([SceneOperation::SetBounds { handle, bounds }])
            .map(|_| ())
    }

    pub fn set_geometry(
        &mut self,
        entity: Entity,
        geometry: prism_render_architecture::gpu_scene::GeometryHandle,
    ) -> Result<(), NativeSceneError> {
        let handle = self.require_handle(entity)?;
        self.apply_operations([SceneOperation::SetGeometry { handle, geometry }])
            .map(|_| ())
    }

    pub fn set_material(
        &mut self,
        entity: Entity,
        material: SceneMaterialHandle,
    ) -> Result<(), NativeSceneError> {
        let handle = self.require_handle(entity)?;
        self.apply_operations([SceneOperation::SetMaterial { handle, material }])
            .map(|_| ())
    }

    pub fn set_flags(
        &mut self,
        entity: Entity,
        mask: u32,
        value: u32,
    ) -> Result<(), NativeSceneError> {
        let handle = self.require_handle(entity)?;
        self.apply_operations([SceneOperation::SetFlags {
            handle,
            mask,
            value,
        }])
        .map(|_| ())
    }

    pub fn set_render_layers(
        &mut self,
        entity: Entity,
        render_layers: u32,
    ) -> Result<(), NativeSceneError> {
        let handle = self.require_handle(entity)?;
        self.apply_operations([SceneOperation::SetRenderLayers {
            handle,
            render_layers,
        }])
        .map(|_| ())
    }

    /// Remove an entity and delay handle reuse until the supplied completion.
    pub fn remove(
        &mut self,
        entity: Entity,
        completion: GpuCompletionValue,
    ) -> Result<(), NativeSceneError> {
        let handle = self.require_handle(entity)?;
        if !self.allocator.validate(handle) {
            return Err(NativeSceneError::Handle(SceneHandleError::NotLive(handle)));
        }
        self.apply_operations([SceneOperation::Destroy { handle }])?;
        self.allocator.retire(handle, completion)?;
        self.entity_handles.remove(&entity);
        Ok(())
    }

    pub fn reclaim_completed(&mut self, completed: GpuCompletionValue) -> u32 {
        self.allocator.reclaim_completed(completed)
    }

    /// Compute the upload strategy for dirty rows not yet acknowledged.
    pub fn upload_plan(&self) -> UploadPlan {
        let dirty_slots: Vec<_> = self
            .dirty_slots
            .iter()
            .map(|(handle, fields)| DirtySceneSlot {
                handle: *handle,
                fields: *fields,
            })
            .collect();
        self.upload_planner
            .plan(self.allocator.stats().allocated_slots, &dirty_slots)
    }

    /// Mark the current dirty set as consumed by the backend.
    pub fn acknowledge_upload(&mut self) {
        self.dirty_slots.clear();
    }

    pub fn instance_row(&self, index: u32) -> Option<GpuSceneInstance> {
        instance_row(&self.cpu, index)
    }

    pub fn current_transform_row(&self, index: u32) -> Option<GpuSceneTransform> {
        current_transform_row(&self.cpu, index)
    }

    pub fn previous_transform_row(&self, index: u32) -> Option<GpuSceneTransform> {
        previous_transform_row(&self.cpu, index)
    }

    pub fn bounds_row(&self, index: u32) -> Option<GpuSceneBounds> {
        bounds_row(&self.cpu, index)
    }

    pub fn snapshot(&self) -> GpuSceneSnapshot {
        GpuSceneSnapshot {
            frame_epoch: self.frame_epoch,
            scene_epoch: self.cpu.scene_epoch(),
            instance_count: self.cpu.live_count(),
            buffer_version: self.buffer_version,
        }
    }

    pub fn len(&self) -> usize {
        self.entity_handles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entity_handles.is_empty()
    }

    fn require_handle(&self, entity: Entity) -> Result<SceneHandle, NativeSceneError> {
        self.handle(entity)
            .ok_or(NativeSceneError::UnknownEntity(entity))
    }

    fn apply_operations<I>(&mut self, operations: I) -> Result<SceneApplyReport, NativeSceneError>
    where
        I: IntoIterator<Item = SceneOperation>,
    {
        let mut builder =
            SceneTransactionBuilder::for_producer(self.frame_epoch, self.next_sequence, 0);
        self.next_sequence = self.next_sequence.wrapping_add(1);
        for operation in operations {
            builder.push(operation);
        }
        let report = self.cpu.apply(&builder.finish());
        if let Some(error) = report.errors.first().copied() {
            return Err(NativeSceneError::Apply(error));
        }
        if report.created != 0 || report.destroyed != 0 || report.updated != 0 {
            self.buffer_version = self.buffer_version.wrapping_add(1);
            merge_dirty_slots(&mut self.dirty_slots, report.dirty_slots.iter().copied());
        }
        Ok(report)
    }
}

fn merge_dirty_slots(
    destination: &mut BTreeMap<SceneHandle, SceneFieldMask>,
    incoming: impl IntoIterator<Item = DirtySceneSlot>,
) {
    for dirty in incoming {
        destination
            .entry(dirty.handle)
            .and_modify(|fields| fields.insert(dirty.fields))
            .or_insert(dirty.fields);
    }
}

/// Native Prism application plugin that registers NativeRenderScene.
pub struct NativeRenderScenePlugin {
    max_slots: u32,
}

impl NativeRenderScenePlugin {
    pub const fn with_capacity(max_slots: u32) -> Self {
        Self { max_slots }
    }
}

impl Default for NativeRenderScenePlugin {
    fn default() -> Self {
        Self::with_capacity(DEFAULT_NATIVE_SCENE_CAPACITY)
    }
}

impl prism_app::Plugin for NativeRenderScenePlugin {
    fn build(&self, app: &mut prism_app::App) {
        app.insert_resource(NativeRenderScene::new(self.max_slots));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::gpu_scene::{SceneBounds, UploadStrategy};

    fn entity(index: u64) -> Entity {
        Entity::from_bits((1_u64 << 32) | index).expect("valid test entity")
    }

    fn transform(x: f32) -> SceneTransform {
        SceneTransform {
            rows: [
                [1.0, 0.0, 0.0, x],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
        }
    }

    fn record() -> InstanceRecord {
        InstanceRecord {
            bounds: SceneBounds {
                center: [0.0, 0.0, 0.0],
                radius: 1.0,
                half_extents: [1.0, 1.0, 1.0],
                ..SceneBounds::default()
            },
            ..InstanceRecord::default()
        }
    }

    #[test]
    fn native_input_reaches_rows_plan_and_snapshot() {
        let mut scene = NativeRenderScene::new(8);
        scene.begin_frame(7);
        let handle = scene.insert(entity(1), record()).expect("insert succeeds");
        let plan = scene.upload_plan();
        assert_eq!(plan.instances.strategy, UploadStrategy::FullRewrite);
        assert_eq!(plan.instances.indices, vec![handle.index]);
        assert_eq!(scene.snapshot().instance_count, 1);
        assert_eq!(scene.instance_row(handle.index).expect("row").active, 1);
        assert_eq!(
            scene.bounds_row(handle.index).expect("row").center_radius[3],
            1.0
        );
    }

    #[test]
    fn transform_history_is_frame_stable() {
        let mut scene = NativeRenderScene::new(8);
        let handle = scene.insert(entity(2), record()).expect("insert succeeds");
        scene
            .set_transform(entity(2), transform(2.0))
            .expect("update");
        scene
            .set_transform(entity(2), transform(3.0))
            .expect("update");
        assert_eq!(
            scene
                .previous_transform_row(handle.index)
                .expect("row")
                .rows[0][3],
            0.0
        );
        scene.begin_frame(1);
        scene
            .set_transform(entity(2), transform(4.0))
            .expect("update");
        assert_eq!(
            scene
                .previous_transform_row(handle.index)
                .expect("row")
                .rows[0][3],
            3.0
        );
        assert_eq!(
            scene.current_transform_row(handle.index).expect("row").rows[0][3],
            4.0
        );
    }

    #[test]
    fn repeated_updates_coalesce_deterministically() {
        let mut scene = NativeRenderScene::new(8);
        let handle = scene.insert(entity(5), record()).expect("insert succeeds");
        scene
            .set_transform(entity(5), transform(1.0))
            .expect("transform update");
        scene
            .set_bounds(
                entity(5),
                SceneBounds {
                    radius: 2.0,
                    ..SceneBounds::default()
                },
            )
            .expect("bounds update");

        let plan = scene.upload_plan();
        assert_eq!(plan.current_transforms.indices, vec![handle.index]);
        assert_eq!(plan.bounds.indices, vec![handle.index]);
        assert_eq!(plan.instances.indices, vec![handle.index]);
    }

    #[test]
    fn removal_delays_reuse_and_rejects_stale_entity_access() {
        let mut scene = NativeRenderScene::new(2);
        let first = scene.insert(entity(3), record()).expect("insert succeeds");
        scene
            .remove(entity(3), GpuCompletionValue(9))
            .expect("remove succeeds");
        assert_eq!(scene.handle(entity(3)), None);
        assert!(scene.insert(entity(4), record()).is_err());
        assert_eq!(scene.reclaim_completed(GpuCompletionValue(8)), 0);
        assert_eq!(scene.reclaim_completed(GpuCompletionValue(9)), 1);
        let reused = scene.insert(entity(4), record()).expect("slot reused");
        assert_eq!(reused.index, first.index);
        assert_ne!(reused.generation, first.generation);
    }

    #[test]
    fn app_plugin_registers_native_resource() {
        let mut app = prism_app::App::new();
        app.add_plugins(NativeRenderScenePlugin::with_capacity(4));
        assert_eq!(app.world().resource::<NativeRenderScene>().len(), 0);
    }
}
