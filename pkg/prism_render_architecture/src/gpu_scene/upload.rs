use std::collections::BTreeSet;

use super::{CpuRenderScene, DirtySceneSlot, InstanceRecord, SceneFieldMask, SceneTransform};

/// Hot instance row consumed by visibility and classification shaders.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GpuSceneInstance {
    pub generation: u32,
    pub flags: u32,
    pub geometry_index: u32,
    pub geometry_generation: u32,
    pub material_index: u32,
    pub material_generation: u32,
    pub active: u32,
    pub render_layers: u32,
}

impl GpuSceneInstance {
    pub fn from_record(generation: u32, record: Option<&InstanceRecord>) -> Self {
        let Some(record) = record else {
            return Self {
                generation,
                ..Self::default()
            };
        };
        Self {
            generation,
            flags: record.flags,
            geometry_index: record.geometry.index,
            geometry_generation: record.geometry.generation,
            material_index: record.material.index,
            material_generation: record.material.generation,
            render_layers: record.render_layers,
            active: 1,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GpuSceneTransform {
    pub rows: [[f32; 4]; 3],
}

impl From<SceneTransform> for GpuSceneTransform {
    fn from(value: SceneTransform) -> Self {
        Self { rows: value.rows }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GpuSceneBounds {
    pub center_radius: [f32; 4],
    pub half_extents: [f32; 4],
}

impl GpuSceneBounds {
    pub fn from_record(record: Option<&InstanceRecord>) -> Self {
        let Some(record) = record else {
            return Self::default();
        };
        Self {
            center_radius: [
                record.bounds.center[0],
                record.bounds.center[1],
                record.bounds.center[2],
                record.bounds.radius,
            ],
            half_extents: [
                record.bounds.half_extents[0],
                record.bounds.half_extents[1],
                record.bounds.half_extents[2],
                0.0,
            ],
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum UploadStrategy {
    #[default]
    None,
    SparseScatter,
    ContiguousRanges,
    FullRewrite,
    Deferred,
}

#[derive(Clone, Copy, Debug)]
pub struct UploadBudget {
    pub max_bytes_per_frame: u64,
    pub max_sparse_elements: u32,
    pub max_copy_regions: u32,
    pub sparse_fraction_threshold: f32,
}

impl Default for UploadBudget {
    fn default() -> Self {
        Self {
            max_bytes_per_frame: 64 * 1024 * 1024,
            max_sparse_elements: 65_536,
            max_copy_regions: 1_024,
            sparse_fraction_threshold: 0.15,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TableUploadPlan {
    pub strategy: UploadStrategy,
    pub element_size: u32,
    pub indices: Vec<u32>,
    pub ranges: Vec<core::ops::Range<u32>>,
    pub estimated_bytes: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct UploadPlan {
    pub instances: TableUploadPlan,
    pub current_transforms: TableUploadPlan,
    pub previous_transforms: TableUploadPlan,
    pub bounds: TableUploadPlan,
    pub estimated_bytes: u64,
    pub deferred_slots: Vec<u32>,
}

pub struct UploadPlanner {
    budget: UploadBudget,
}

impl UploadPlanner {
    pub fn new(budget: UploadBudget) -> Self {
        assert!(
            (0.0..=1.0).contains(&budget.sparse_fraction_threshold),
            "sparse fraction threshold must be in 0..=1"
        );
        Self { budget }
    }

    pub fn plan(&self, scene_capacity: u32, dirty_slots: &[DirtySceneSlot]) -> UploadPlan {
        let mut instances = BTreeSet::new();
        let mut current = BTreeSet::new();
        let mut previous = BTreeSet::new();
        let mut bounds = BTreeSet::new();
        for dirty in dirty_slots {
            if dirty.fields.contains(SceneFieldMask::INSTANCE)
                || dirty.fields.contains(SceneFieldMask::GENERATION)
            {
                instances.insert(dirty.handle.index);
            }
            if dirty.fields.contains(SceneFieldMask::CURRENT_TRANSFORM) {
                current.insert(dirty.handle.index);
            }
            if dirty.fields.contains(SceneFieldMask::PREVIOUS_TRANSFORM) {
                previous.insert(dirty.handle.index);
            }
            if dirty.fields.contains(SceneFieldMask::BOUNDS) {
                bounds.insert(dirty.handle.index);
            }
        }

        let mut plan = UploadPlan {
            instances: self.plan_table(scene_capacity, instances, 32),
            current_transforms: self.plan_table(scene_capacity, current, 48),
            previous_transforms: self.plan_table(scene_capacity, previous, 48),
            bounds: self.plan_table(scene_capacity, bounds, 32),
            ..UploadPlan::default()
        };
        plan.estimated_bytes = plan.instances.estimated_bytes
            + plan.current_transforms.estimated_bytes
            + plan.previous_transforms.estimated_bytes
            + plan.bounds.estimated_bytes;

        if plan.estimated_bytes > self.budget.max_bytes_per_frame {
            // Structural and identity updates are atomic and cannot be deferred.
            // The implementation may defer lower-priority custom tables later;
            // the core scene tables deliberately report the over-budget plan.
            plan.deferred_slots = Vec::new();
        }
        plan
    }

    fn plan_table(
        &self,
        capacity: u32,
        indices: BTreeSet<u32>,
        element_size: u32,
    ) -> TableUploadPlan {
        if indices.is_empty() || capacity == 0 {
            return TableUploadPlan {
                element_size,
                ..TableUploadPlan::default()
            };
        }
        let indices: Vec<_> = indices.into_iter().collect();
        let ranges = contiguous_ranges(&indices);
        let dirty_fraction = indices.len() as f32 / capacity as f32;
        let (strategy, estimated_bytes) = if dirty_fraction > self.budget.sparse_fraction_threshold
        {
            (
                UploadStrategy::FullRewrite,
                capacity as u64 * element_size as u64,
            )
        } else if ranges.len() as u32 <= self.budget.max_copy_regions
            && ranges.len() * 2 < indices.len()
        {
            (
                UploadStrategy::ContiguousRanges,
                indices.len() as u64 * element_size as u64,
            )
        } else if indices.len() as u32 <= self.budget.max_sparse_elements {
            (
                UploadStrategy::SparseScatter,
                indices.len() as u64 * (element_size as u64 + 4),
            )
        } else {
            (
                UploadStrategy::FullRewrite,
                capacity as u64 * element_size as u64,
            )
        };
        TableUploadPlan {
            strategy,
            element_size,
            indices,
            ranges,
            estimated_bytes,
        }
    }
}

pub fn instance_row(scene: &CpuRenderScene, index: u32) -> Option<GpuSceneInstance> {
    let (generation, record) = scene.record_at(index)?;
    Some(GpuSceneInstance::from_record(generation, record))
}

pub fn current_transform_row(scene: &CpuRenderScene, index: u32) -> Option<GpuSceneTransform> {
    let (_, record) = scene.record_at(index)?;
    Some(record.map_or_else(GpuSceneTransform::default, |record| {
        record.current_transform.into()
    }))
}

pub fn previous_transform_row(scene: &CpuRenderScene, index: u32) -> Option<GpuSceneTransform> {
    let (_, record) = scene.record_at(index)?;
    Some(record.map_or_else(GpuSceneTransform::default, |record| {
        record.previous_transform.into()
    }))
}

pub fn bounds_row(scene: &CpuRenderScene, index: u32) -> Option<GpuSceneBounds> {
    let (_, record) = scene.record_at(index)?;
    Some(GpuSceneBounds::from_record(record))
}

fn contiguous_ranges(indices: &[u32]) -> Vec<core::ops::Range<u32>> {
    let Some(&first) = indices.first() else {
        return Vec::new();
    };
    let mut ranges = Vec::new();
    let mut start = first;
    let mut previous = first;
    for &index in &indices[1..] {
        if index != previous + 1 {
            ranges.push(start..previous + 1);
            start = index;
        }
        previous = index;
    }
    ranges.push(start..previous + 1);
    ranges
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_scene::{SceneHandle, SceneMaterialHandle};

    fn dirty(index: u32, fields: SceneFieldMask) -> DirtySceneSlot {
        DirtySceneSlot {
            handle: SceneHandle {
                index,
                generation: 1,
            },
            fields,
        }
    }

    #[test]
    fn gpu_layouts_are_word_aligned_and_stable() {
        assert_eq!(size_of::<GpuSceneInstance>(), 32);
        assert_eq!(size_of::<GpuSceneTransform>(), 48);
        assert_eq!(size_of::<GpuSceneBounds>(), 32);
        assert_eq!(align_of::<GpuSceneInstance>(), 4);
    }

    #[test]
    fn sparse_and_contiguous_strategies_are_selected() {
        let planner = UploadPlanner::new(UploadBudget::default());
        let sparse = planner.plan(
            1_000,
            &[
                dirty(1, SceneFieldMask::INSTANCE),
                dirty(500, SceneFieldMask::INSTANCE),
            ],
        );
        assert_eq!(sparse.instances.strategy, UploadStrategy::SparseScatter);

        let contiguous: Vec<_> = (10..30)
            .map(|index| dirty(index, SceneFieldMask::CURRENT_TRANSFORM))
            .collect();
        let contiguous = planner.plan(1_000, &contiguous);
        assert_eq!(
            contiguous.current_transforms.strategy,
            UploadStrategy::ContiguousRanges
        );
        assert_eq!(contiguous.current_transforms.ranges, vec![10..30]);
    }

    #[test]
    fn high_dirty_fraction_uses_full_rewrite() {
        let planner = UploadPlanner::new(UploadBudget::default());
        let dirty: Vec<_> = (1..30)
            .map(|index| dirty(index, SceneFieldMask::BOUNDS))
            .collect();
        let plan = planner.plan(32, &dirty);
        assert_eq!(plan.bounds.strategy, UploadStrategy::FullRewrite);
        assert_eq!(plan.bounds.estimated_bytes, 32 * 32);
    }

    #[test]
    fn inactive_instance_retains_generation() {
        let row = GpuSceneInstance::from_record(8, None);
        assert_eq!(row.generation, 8);
        assert_eq!(row.active, 0);
        let record = InstanceRecord {
            material: SceneMaterialHandle {
                index: 3,
                generation: 2,
            },
            ..InstanceRecord::default()
        };
        let row = GpuSceneInstance::from_record(8, Some(&record));
        assert_eq!(row.material_index, 3);
        assert_eq!(row.material_generation, 2);
        assert_eq!(row.active, 1);
    }
}
