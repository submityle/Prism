use bevy_ecs::resource::Resource;
use prism_render_architecture::gpu_scene::{UploadBudget, UploadStrategy};

/// Runtime policy for the atomic core GPU Scene upload.
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct GpuSceneUploadSettings {
    pub budget: UploadBudget,
}

/// Per-frame CPU-side GPU Scene diagnostics.
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct GpuSceneDiagnostics {
    pub active_instances: u32,
    pub created: u32,
    pub destroyed: u32,
    pub updated_fields: u32,
    pub transaction_errors: u32,
    pub allocation_failures: u32,
    pub stale_handles: u32,
    pub reclaimed_handles: u32,
    pub dirty_slots: u32,
    pub uploaded_bytes: u64,
    pub upload_budget_exceeded: bool,
    pub instance_upload: UploadStrategy,
    pub current_transform_upload: UploadStrategy,
    pub previous_transform_upload: UploadStrategy,
    pub bounds_upload: UploadStrategy,
    pub buffer_version: u32,
    pub buffer_rebuilds: u32,
    pub opaque_visible: u32,
    pub opaque_queued: u32,
    pub opaque_skipped: u32,
    pub scene_epoch: u64,
}
