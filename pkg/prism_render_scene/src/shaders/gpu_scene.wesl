//! Stable GPU Scene structures shared by render consumers.

struct GpuSceneInstance {
    generation: u32,
    flags: u32,
    geometry_index: u32,
    geometry_generation: u32,
    material_index: u32,
    material_generation: u32,
    render_layers: u32,
    is_active: u32,
}

struct GpuSceneTransform {
    world_from_local: mat3x4<f32>,
}

struct GpuSceneBounds {
    center_radius: vec4<f32>,
    half_extents: vec4<f32>,
}
