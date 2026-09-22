//! Stable, incrementally updated GPU scene.

mod allocator;
mod mirror;
mod transaction;
mod upload;

pub use allocator::{
    GpuCompletionValue, SceneCapacityError, SceneHandleAllocator, SceneHandleError,
    SceneHandleStats,
};
pub use mirror::{
    CpuRenderScene, DirtySceneSlot, SceneApplyError, SceneApplyReport, SceneFieldMask,
};
pub use transaction::{
    merge_transactions, MergeIssue, MergeIssueKind, MergeReport, SceneOperation, SceneTransaction,
    SceneTransactionBuilder,
};
pub use upload::{
    bounds_row, current_transform_row, instance_row, previous_transform_row, GpuSceneBounds,
    GpuSceneInstance, GpuSceneTransform, TableUploadPlan, UploadBudget, UploadPlan, UploadPlanner,
    UploadStrategy,
};

use crate::abi::GenerationalHandle;

pub type SceneHandle = GenerationalHandle;

/// A stable reference to geometry stored by the render backend.
pub type GeometryHandle = GenerationalHandle;

/// A stable reference to material data stored by the render backend.
pub type SceneMaterialHandle = GenerationalHandle;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SceneBounds {
    pub center: [f32; 3],
    pub radius: f32,
    pub half_extents: [f32; 3],
    pub _padding: f32,
}

/// A row-major affine 3×4 transform used by the architecture contract.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SceneTransform {
    pub rows: [[f32; 4]; 3],
}

impl SceneTransform {
    pub const IDENTITY: Self = Self {
        rows: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
    };
}

impl Default for SceneTransform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// Complete CPU representation needed to publish an instance to the GPU scene.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct InstanceRecord {
    pub current_transform: SceneTransform,
    pub previous_transform: SceneTransform,
    pub bounds: SceneBounds,
    pub geometry: GeometryHandle,
    pub material: SceneMaterialHandle,
    pub render_layers: u32,
    pub flags: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SceneChangeKind {
    Create,
    Destroy,
    Transform,
    Bounds,
    Geometry,
    Material,
    Flags,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SceneChange {
    pub handle: SceneHandle,
    pub kind: SceneChangeKind,
    pub sequence: u64,
}

#[derive(Default)]
pub struct SceneChangeJournal {
    changes: Vec<SceneChange>,
}

impl SceneChangeJournal {
    pub fn push(&mut self, change: SceneChange) {
        self.changes.push(change);
    }

    pub fn changes(&self) -> &[SceneChange] {
        &self.changes
    }

    pub fn clear(&mut self) {
        self.changes.clear();
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GpuSceneSnapshot {
    pub frame_epoch: u64,
    pub scene_epoch: u64,
    pub instance_count: u32,
    pub buffer_version: u32,
}
