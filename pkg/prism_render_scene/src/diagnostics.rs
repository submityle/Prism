use bevy_ecs::resource::Resource;

/// Per-frame CPU-side GPU Scene diagnostics.
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct GpuSceneDiagnostics {
    pub active_instances: u32,
    pub created: u32,
    pub destroyed: u32,
    pub updated_fields: u32,
    pub transaction_errors: u32,
    pub scene_epoch: u64,
}
