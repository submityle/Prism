//! GPU residency for the native retained scene.
//!
//! The main world owns PrismRenderScene and its authoritative CPU mirror. This
//! module is the one-way render projection: it extracts only the rows selected
//! by the architecture upload plan, keeps four stable-index storage buffers,
//! and applies sparse/range/full uploads in the render world's prepare stage.
//! No render-world resource is used as a source of truth for scene state.

use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

use prism_ecs::prelude::{Res, ResMut, Resource};
use prism_ecs::world::World;
use prism_render_architecture::gpu_scene::{
    bounds_row, current_transform_row, instance_row, previous_transform_row, GpuSceneBounds,
    GpuSceneInstance, GpuSceneTransform, SparseBufferPlanError, SparseBufferUploadReport,
    SparseBufferVec, UploadBudget, UploadPlan, UploadPlanner,
};
use prism_render_driver::{BufferId, BufferUsages, FrameSync, Instance, RenderDevice};
use prism_render_runtime::RenderRuntime;

use crate::runtime_resource::RenderRuntimeResource;
use crate::visibility::PrismRenderScene;

/// One row copied across the world boundary for a stable scene index.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NativeSceneGpuRow {
    /// The stable scene instance data.
    pub instance: GpuSceneInstance,
    /// The current transform used by culling and shading.
    pub current_transform: GpuSceneTransform,
    /// The previous transform used by temporal reprojection.
    pub previous_transform: GpuSceneTransform,
    /// The world-space bounds used by visibility.
    pub bounds: GpuSceneBounds,
}

/// The extracted, device-independent native scene projection.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExtractedNativeScene {
    /// CPU slot high-water mark, including trailing tombstones.
    pub capacity: u32,
    /// Scene epoch associated with the plan.
    pub scene_epoch: u64,
    /// Per-table upload decisions made from the authoritative dirty report.
    pub plan: UploadPlan,
    /// Rows needed by the plan, or the complete slot image on first upload or
    /// after device replacement.
    pub rows: Vec<(u32, NativeSceneGpuRow)>,
    /// Whether rows is a complete slot image rather than a dirty projection.
    pub full_snapshot: bool,
}

impl Resource for ExtractedNativeScene {}

/// The result of preparing all four native scene tables for one frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NativeSceneGpuUploadReport {
    /// Scene epoch that was prepared.
    pub scene_epoch: u64,
    /// Slot high-water mark represented by the buffers.
    pub capacity: u32,
    /// Device generation to which the buffers belong.
    pub device_generation: u64,
    /// Instance-table upload.
    pub instances: SparseBufferUploadReport,
    /// Current-transform upload.
    pub current_transforms: SparseBufferUploadReport,
    /// Previous-transform upload.
    pub previous_transforms: SparseBufferUploadReport,
    /// Bounds upload.
    pub bounds: SparseBufferUploadReport,
}

/// A native scene upload can fail only when the extracted architecture plan
/// does not match the table ABI or its stable-index allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeSceneGpuUploadError {
    /// The render device changed before a full CPU snapshot was extracted.
    DeviceGenerationRequiresFullSnapshot { device_generation: u64 },
    /// The instance table rejected its plan.
    Instances(SparseBufferPlanError),
    /// The current-transform table rejected its plan.
    CurrentTransforms(SparseBufferPlanError),
    /// The previous-transform table rejected its plan.
    PreviousTransforms(SparseBufferPlanError),
    /// The bounds table rejected its plan.
    Bounds(SparseBufferPlanError),
}

impl fmt::Display for NativeSceneGpuUploadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeviceGenerationRequiresFullSnapshot { device_generation } => write!(
                formatter,
                "native scene device generation {device_generation} requires a full CPU snapshot"
            ),
            Self::Instances(error) => write!(formatter, "native scene instances: {error}"),
            Self::CurrentTransforms(error) => {
                write!(formatter, "native scene current transforms: {error}")
            }
            Self::PreviousTransforms(error) => {
                write!(formatter, "native scene previous transforms: {error}")
            }
            Self::Bounds(error) => write!(formatter, "native scene bounds: {error}"),
        }
    }
}

impl std::error::Error for NativeSceneGpuUploadError {}

/// The render-world owner of the four native scene storage buffers.
pub struct NativeSceneGpuState {
    instances: SparseBufferVec<GpuSceneInstance>,
    current_transforms: SparseBufferVec<GpuSceneTransform>,
    previous_transforms: SparseBufferVec<GpuSceneTransform>,
    bounds: SparseBufferVec<GpuSceneBounds>,
    device_generation: Option<u64>,
    initialized: bool,
    last_report: NativeSceneGpuUploadReport,
}

impl Resource for NativeSceneGpuState {}

impl Default for NativeSceneGpuState {
    fn default() -> Self {
        Self::new()
    }
}

impl NativeSceneGpuState {
    /// Creates an empty native scene GPU owner. Buffers are allocated lazily
    /// during the first prepare pass, when the scene high-water mark is known.
    #[must_use]
    pub fn new() -> Self {
        Self {
            instances: SparseBufferVec::new(
                BufferUsages::STORAGE,
                Arc::from("prism.native_scene.instances"),
            ),
            current_transforms: SparseBufferVec::new(
                BufferUsages::STORAGE,
                Arc::from("prism.native_scene.current_transforms"),
            ),
            previous_transforms: SparseBufferVec::new(
                BufferUsages::STORAGE,
                Arc::from("prism.native_scene.previous_transforms"),
            ),
            bounds: SparseBufferVec::new(
                BufferUsages::STORAGE,
                Arc::from("prism.native_scene.bounds"),
            ),
            device_generation: None,
            initialized: false,
            last_report: NativeSceneGpuUploadReport::default(),
        }
    }

    /// Returns whether extraction must include every CPU scene slot.
    #[must_use]
    pub fn needs_full_snapshot(&self, device_generation: u64) -> bool {
        !self.initialized || self.device_generation != Some(device_generation)
    }

    /// The report from the most recent successful prepare.
    #[must_use]
    pub fn last_report(&self) -> NativeSceneGpuUploadReport {
        self.last_report
    }

    /// The stable-index instance table, if the first prepare has run.
    #[must_use]
    pub fn instance_buffer(&self) -> Option<BufferId> {
        self.instances.buffer()
    }

    /// The current-transform table, if the first prepare has run.
    #[must_use]
    pub fn current_transform_buffer(&self) -> Option<BufferId> {
        self.current_transforms.buffer()
    }

    /// The previous-transform table, if the first prepare has run.
    #[must_use]
    pub fn previous_transform_buffer(&self) -> Option<BufferId> {
        self.previous_transforms.buffer()
    }

    /// The bounds table, if the first prepare has run.
    #[must_use]
    pub fn bounds_buffer(&self) -> Option<BufferId> {
        self.bounds.buffer()
    }

    /// Uploads one extracted projection to the current render device.
    pub fn prepare<I>(
        &mut self,
        runtime: &RenderRuntime<I>,
        extracted: &ExtractedNativeScene,
    ) -> Result<NativeSceneGpuUploadReport, NativeSceneGpuUploadError>
    where
        I: Instance + 'static,
    {
        let device_generation = runtime.device_generation();
        let generation_changed = self.device_generation != Some(device_generation);
        if generation_changed {
            if !extracted.full_snapshot {
                return Err(
                    NativeSceneGpuUploadError::DeviceGenerationRequiresFullSnapshot {
                        device_generation,
                    },
                );
            }
            self.reset_for_device(device_generation);
        }

        self.instances.ensure_capacity(extracted.capacity);
        self.current_transforms.ensure_capacity(extracted.capacity);
        self.previous_transforms.ensure_capacity(extracted.capacity);
        self.bounds.ensure_capacity(extracted.capacity);

        for (index, row) in &extracted.rows {
            self.instances.grow_and_set(*index, row.instance);
            self.current_transforms
                .grow_and_set(*index, row.current_transform);
            self.previous_transforms
                .grow_and_set(*index, row.previous_transform);
            self.bounds.grow_and_set(*index, row.bounds);
        }

        let instances = self
            .instances
            .write_buffer_with_plan(runtime.device(), runtime.queue(), &extracted.plan.instances)
            .map_err(NativeSceneGpuUploadError::Instances)?;
        let current_transforms = self
            .current_transforms
            .write_buffer_with_plan(
                runtime.device(),
                runtime.queue(),
                &extracted.plan.current_transforms,
            )
            .map_err(NativeSceneGpuUploadError::CurrentTransforms)?;
        let previous_transforms = self
            .previous_transforms
            .write_buffer_with_plan(
                runtime.device(),
                runtime.queue(),
                &extracted.plan.previous_transforms,
            )
            .map_err(NativeSceneGpuUploadError::PreviousTransforms)?;
        let bounds = self
            .bounds
            .write_buffer_with_plan(runtime.device(), runtime.queue(), &extracted.plan.bounds)
            .map_err(NativeSceneGpuUploadError::Bounds)?;

        self.initialized = true;
        let report = NativeSceneGpuUploadReport {
            scene_epoch: extracted.scene_epoch,
            capacity: extracted.capacity,
            device_generation,
            instances,
            current_transforms,
            previous_transforms,
            bounds,
        };
        self.last_report = report;
        Ok(report)
    }

    /// Destroys all buffers while the owning device is known to be idle.
    pub fn destroy<D: RenderDevice>(&mut self, device: &D) {
        for buffer in [
            self.instances.buffer(),
            self.current_transforms.buffer(),
            self.previous_transforms.buffer(),
            self.bounds.buffer(),
        ]
        .into_iter()
        .flatten()
        {
            device.destroy_buffer(buffer);
        }
        *self = Self::new();
    }

    fn reset_for_device(&mut self, device_generation: u64) {
        *self = Self::new();
        self.device_generation = Some(device_generation);
    }
}

/// Extracts the native scene after main-world visibility synchronization.
pub fn extract_native_scene<I>(main: &mut World, secondary: &mut World)
where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    let device_generation = secondary
        .get_resource::<RenderRuntimeResource<I>>()
        .map_or(0, |runtime| runtime.runtime().device_generation());
    let full_snapshot = secondary
        .get_resource::<NativeSceneGpuState>()
        .map_or(true, |state| state.needs_full_snapshot(device_generation));
    let Some(scene) = main.get_resource::<PrismRenderScene>() else {
        secondary.insert_resource(ExtractedNativeScene {
            full_snapshot,
            ..ExtractedNativeScene::default()
        });
        return;
    };

    let cpu = scene.scene();
    let planner = UploadPlanner::new(UploadBudget::default());
    let plan = planner.plan(
        cpu.capacity() as u32,
        &scene.last_apply_report().dirty_slots,
    );
    let indices = planned_indices(&plan, full_snapshot, cpu.capacity() as u32);
    let rows = indices
        .into_iter()
        .filter_map(|index| {
            let row = NativeSceneGpuRow {
                instance: instance_row(cpu, index)?,
                current_transform: current_transform_row(cpu, index)?,
                previous_transform: previous_transform_row(cpu, index)?,
                bounds: bounds_row(cpu, index)?,
            };
            Some((index, row))
        })
        .collect();
    secondary.insert_resource(ExtractedNativeScene {
        capacity: cpu.capacity() as u32,
        scene_epoch: cpu.scene_epoch(),
        plan,
        rows,
        full_snapshot,
    });
}

/// Prepare system installed in the render world's PrepareResources stage.
pub fn prepare_native_scene_gpu<I>(
    mut state: ResMut<NativeSceneGpuState>,
    runtime: Res<RenderRuntimeResource<I>>,
    extracted: Res<ExtractedNativeScene>,
) where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    state
        .prepare(runtime.runtime(), &extracted)
        .unwrap_or_else(|error| panic!("native scene GPU preparation failed: {error}"));
}

fn planned_indices(plan: &UploadPlan, full_snapshot: bool, capacity: u32) -> BTreeSet<u32> {
    if full_snapshot {
        return (0..capacity).collect();
    }
    let mut indices = BTreeSet::new();
    for table in [
        &plan.instances,
        &plan.current_transforms,
        &plan.previous_transforms,
        &plan.bounds,
    ] {
        indices.extend(table.indices.iter().copied());
        for range in &table.ranges {
            indices.extend(range.clone());
        }
    }
    indices
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::gpu_scene::{
        CpuRenderScene, GeometryHandle, GpuSceneBounds, GpuSceneTransform, InstanceRecord,
        SceneApplyReport, SceneBounds, SceneFieldMask, SceneHandle, SceneMaterialHandle,
        SceneOperation, SceneTransaction, SceneTransform, UploadStrategy,
    };
    use prism_render_driver::Extent3d;
    use prism_render_driver_null::NullInstance;
    use prism_render_runtime::RenderRuntime;

    fn runtime() -> RenderRuntime<NullInstance> {
        prism_render_runtime::RenderRuntime::headless(NullInstance::new(), Extent3d::new_2d(1, 1))
            .expect("null runtime")
    }

    fn scene_with_one_row() -> (CpuRenderScene, SceneApplyReport) {
        let mut scene = CpuRenderScene::default();
        let transaction = SceneTransaction {
            frame_epoch: 1,
            sequence: 1,
            producer: 0,
            operations: vec![SceneOperation::Create {
                handle: SceneHandle::new(1, 1),
                record: InstanceRecord {
                    current_transform: SceneTransform::IDENTITY,
                    previous_transform: SceneTransform::IDENTITY,
                    bounds: SceneBounds {
                        center: [1.0, 2.0, 3.0],
                        radius: 4.0,
                        half_extents: [1.0, 1.0, 1.0],
                        _padding: 0.0,
                    },
                    geometry: GeometryHandle::new(2, 1),
                    material: SceneMaterialHandle::new(3, 1),
                    render_layers: 7,
                    flags: 9,
                },
            }],
        };
        let report = scene.apply(&transaction);
        (scene, report)
    }

    fn scene_with_rows(count: u32) -> (CpuRenderScene, SceneApplyReport) {
        let mut scene = CpuRenderScene::default();
        let operations = (1..=count)
            .map(|index| SceneOperation::Create {
                handle: SceneHandle::new(index, 1),
                record: InstanceRecord {
                    current_transform: SceneTransform::IDENTITY,
                    previous_transform: SceneTransform::IDENTITY,
                    bounds: SceneBounds {
                        center: [index as f32, 0.0, 0.0],
                        radius: 1.0,
                        half_extents: [1.0, 1.0, 1.0],
                        _padding: 0.0,
                    },
                    geometry: GeometryHandle::new(index, 1),
                    material: SceneMaterialHandle::new(index, 1),
                    render_layers: 1,
                    flags: index,
                },
            })
            .collect();
        let report = scene.apply(&SceneTransaction {
            frame_epoch: 1,
            sequence: 1,
            producer: 0,
            operations,
        });
        (scene, report)
    }

    fn extracted_for(
        cpu: &CpuRenderScene,
        apply_report: &SceneApplyReport,
        full_snapshot: bool,
    ) -> ExtractedNativeScene {
        let capacity = cpu.capacity() as u32;
        let plan =
            UploadPlanner::new(UploadBudget::default()).plan(capacity, &apply_report.dirty_slots);
        let rows = planned_indices(&plan, full_snapshot, capacity)
            .into_iter()
            .filter_map(|index| {
                Some((
                    index,
                    NativeSceneGpuRow {
                        instance: instance_row(cpu, index)?,
                        current_transform: current_transform_row(cpu, index)?,
                        previous_transform: previous_transform_row(cpu, index)?,
                        bounds: bounds_row(cpu, index)?,
                    },
                ))
            })
            .collect();
        ExtractedNativeScene {
            capacity,
            scene_epoch: cpu.scene_epoch(),
            plan,
            rows,
            full_snapshot,
        }
    }

    #[test]
    fn first_prepare_writes_all_four_native_tables() {
        let runtime = runtime();
        let (cpu, apply_report) = scene_with_one_row();
        let extracted = extracted_for(&cpu, &apply_report, true);
        let mut state = NativeSceneGpuState::new();
        let report = state.prepare(&runtime, &extracted).expect("native upload");
        assert_eq!(report.device_generation, runtime.device_generation());
        assert!(report.instances.recreated);
        assert!(report.current_transforms.recreated);
        assert!(report.previous_transforms.recreated);
        assert!(report.bounds.recreated);
        assert!(state.instance_buffer().is_some());
        assert_eq!(state.last_report().scene_epoch, 1);
    }

    #[test]
    fn incremental_sparse_upload_only_writes_the_dirty_table() {
        let runtime = runtime();
        let (mut cpu, initial) = scene_with_rows(100);
        let mut state = NativeSceneGpuState::new();
        state
            .prepare(&runtime, &extracted_for(&cpu, &initial, true))
            .expect("initial native upload");

        let mut current = SceneTransform::IDENTITY;
        current.rows[0][3] = 42.0;
        let update = cpu.apply(&SceneTransaction {
            frame_epoch: 2,
            sequence: 2,
            producer: 0,
            operations: vec![SceneOperation::SetTransform {
                handle: SceneHandle::new(50, 1),
                current,
            }],
        });
        let report = state
            .prepare(&runtime, &extracted_for(&cpu, &update, false))
            .expect("sparse native upload");

        assert_eq!(report.instances.strategy, UploadStrategy::None);
        assert_eq!(
            report.current_transforms.strategy,
            UploadStrategy::SparseScatter
        );
        assert_eq!(
            report.previous_transforms.strategy,
            UploadStrategy::SparseScatter
        );
        assert_eq!(report.bounds.strategy, UploadStrategy::None);
        assert_eq!(report.current_transforms.regions, 1);
        assert_eq!(report.previous_transforms.regions, 1);
        assert!(report.current_transforms.bytes_written > 0);
        assert!(report.previous_transforms.bytes_written > 0);
    }

    #[test]
    fn incremental_contiguous_upload_coalesces_a_range() {
        let runtime = runtime();
        let (mut cpu, initial) = scene_with_rows(100);
        let mut state = NativeSceneGpuState::new();
        state
            .prepare(&runtime, &extracted_for(&cpu, &initial, true))
            .expect("initial native upload");

        let update = cpu.apply(&SceneTransaction {
            frame_epoch: 2,
            sequence: 2,
            producer: 0,
            operations: (10..20)
                .map(|index| SceneOperation::SetBounds {
                    handle: SceneHandle::new(index, 1),
                    bounds: SceneBounds {
                        center: [index as f32, 3.0, 0.0],
                        radius: 2.0,
                        half_extents: [2.0, 2.0, 2.0],
                        _padding: 0.0,
                    },
                })
                .collect(),
        });
        let report = state
            .prepare(&runtime, &extracted_for(&cpu, &update, false))
            .expect("range native upload");

        assert_eq!(report.instances.strategy, UploadStrategy::None);
        assert_eq!(report.current_transforms.strategy, UploadStrategy::None);
        assert_eq!(report.previous_transforms.strategy, UploadStrategy::None);
        assert_eq!(report.bounds.strategy, UploadStrategy::ContiguousRanges);
        assert_eq!(report.bounds.regions, 1);
        assert_eq!(
            report.bounds.bytes_written,
            10 * core::mem::size_of::<GpuSceneBounds>() as u64
        );
    }

    #[test]
    fn destroy_uploads_zero_tombstones_to_all_four_tables() {
        let runtime = runtime();
        let (mut cpu, initial) = scene_with_one_row();
        let mut state = NativeSceneGpuState::new();
        state
            .prepare(&runtime, &extracted_for(&cpu, &initial, true))
            .expect("initial native upload");

        let destroy = cpu.apply(&SceneTransaction {
            frame_epoch: 2,
            sequence: 2,
            producer: 0,
            operations: vec![SceneOperation::Destroy {
                handle: SceneHandle::new(1, 1),
            }],
        });
        assert_eq!(destroy.dirty_slots[0].fields, SceneFieldMask::ALL);
        let report = state
            .prepare(&runtime, &extracted_for(&cpu, &destroy, false))
            .expect("tombstone native upload");

        assert_ne!(report.instances.strategy, UploadStrategy::None);
        assert_ne!(report.current_transforms.strategy, UploadStrategy::None);
        assert_ne!(report.previous_transforms.strategy, UploadStrategy::None);
        assert_ne!(report.bounds.strategy, UploadStrategy::None);
        assert_eq!(state.instances.get(1).unwrap().active, 0);
        assert_eq!(
            *state.current_transforms.get(1).unwrap(),
            GpuSceneTransform::default()
        );
        assert_eq!(
            *state.previous_transforms.get(1).unwrap(),
            GpuSceneTransform::default()
        );
        assert_eq!(*state.bounds.get(1).unwrap(), GpuSceneBounds::default());
    }

    #[test]
    fn device_recovery_rejects_delta_and_rebuilds_from_full_snapshot() {
        let mut runtime = runtime();
        let (cpu, initial) = scene_with_one_row();
        let full = extracted_for(&cpu, &initial, true);
        let mut state = NativeSceneGpuState::new();
        state
            .prepare(&runtime, &full)
            .expect("initial native upload");

        let recovery = runtime.recover_device().expect("null recovery");
        assert_eq!(recovery.generation, 1);
        let delta = ExtractedNativeScene {
            full_snapshot: false,
            ..full.clone()
        };
        assert!(matches!(
            state.prepare(&runtime, &delta),
            Err(
                NativeSceneGpuUploadError::DeviceGenerationRequiresFullSnapshot {
                    device_generation: 1
                }
            )
        ));

        let rebuilt = state
            .prepare(&runtime, &full)
            .expect("full native recovery upload");
        assert_eq!(rebuilt.device_generation, 1);
        assert!(rebuilt.instances.recreated);
        assert!(rebuilt.current_transforms.recreated);
        assert!(rebuilt.previous_transforms.recreated);
        assert!(rebuilt.bounds.recreated);
        assert!(!state.needs_full_snapshot(1));
    }

    #[test]
    fn extraction_without_a_prism_render_scene_is_an_empty_snapshot() {
        let mut main = World::new();
        let mut secondary = World::new();
        extract_native_scene::<NullInstance>(&mut main, &mut secondary);
        let extracted = secondary
            .get_resource::<ExtractedNativeScene>()
            .expect("empty extraction resource");
        assert_eq!(extracted.capacity, 0);
        assert_eq!(extracted.scene_epoch, 0);
        assert!(extracted.rows.is_empty());
        assert!(extracted.full_snapshot);
    }

    #[test]
    fn destroy_releases_all_four_native_scene_buffers() {
        let runtime = runtime();
        let (cpu, initial) = scene_with_one_row();
        let mut state = NativeSceneGpuState::new();
        state
            .prepare(&runtime, &extracted_for(&cpu, &initial, true))
            .expect("native upload");
        assert_eq!(runtime.device().live_resource_count(), 4);

        state.destroy(runtime.device());

        assert_eq!(runtime.device().live_resource_count(), 0);
        assert!(state.instance_buffer().is_none());
        assert!(state.current_transform_buffer().is_none());
        assert!(state.previous_transform_buffer().is_none());
        assert!(state.bounds_buffer().is_none());
    }
}
