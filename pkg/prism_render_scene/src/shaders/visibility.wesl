//! Unified visibility ABI. The first implementation publishes the same work
//! through the deterministic CPU reference; GPU compute stages consume these
//! layouts without changing downstream clients.

struct PrismVisibilityView {
    clip_from_world: mat4x4<f32>,
    previous_clip_from_world: mat4x4<f32>,
    frustum_planes: array<vec4<f32>, 6>,
    world_position_lod_scale: vec4<f32>,
    viewport: vec4<u32>,
    handle_index: u32,
    handle_generation: u32,
    layer_mask: u32,
    flags: u32,
    history_epoch_low: u32,
    history_epoch_high: u32,
    padding: vec2<u32>,
}

struct PrismVisibilityWorkItem {
    scene_index: u32,
    scene_generation: u32,
    geometry_index: u32,
    geometry_generation: u32,
    material_index: u32,
    material_generation: u32,
    lod_or_cluster: u32,
    pass_mask: u32,
    sort_key_low: u32,
    sort_key_high: u32,
    padding: vec2<u32>,
}

struct PrismVisibilityRange {
    view_index: u32,
    view_generation: u32,
    start: u32,
    count: u32,
}
