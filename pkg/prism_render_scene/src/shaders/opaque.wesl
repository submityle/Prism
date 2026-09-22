//! Minimal opaque consumer proving stable GPU Scene addressing.

import bevy_pbr::render::{
    mesh_functions,
    view_transformations::position_world_to_clip
};
import bevy_render::maths::affine3_to_square;

struct GpuSceneInstance {
    generation: u32, flags: u32,
    geometry_index: u32, geometry_generation: u32,
    material_index: u32, material_generation: u32,
    render_layers: u32, active: u32,
}
struct GpuSceneTransform { world_from_local: mat3x4<f32> }
struct GpuSceneBounds { center_radius: vec4<f32>, half_extents: vec4<f32> }

@group(2) @binding(0) var<storage, read> scene_instances: array<GpuSceneInstance>;
@group(2) @binding(1) var<storage, read> scene_current_transforms: array<GpuSceneTransform>;
@group(2) @binding(2) var<storage, read> scene_previous_transforms: array<GpuSceneTransform>;
@group(2) @binding(3) var<storage, read> scene_bounds: array<GpuSceneBounds>;

struct SceneAddress { index: u32, generation: u32 }
var<immediate> scene_address: SceneAddress;

struct Vertex { @location(0) position: vec3<f32> };

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) @interpolate(flat) valid: u32,
};

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    let instance = scene_instances[scene_address.index];
    let valid = instance.active != 0u && instance.generation == scene_address.generation;
    let position = vertex.position;
    let world_from_local = affine3_to_square(
        scene_current_transforms[scene_address.index].world_from_local
    );
    var out: VertexOutput;
    out.clip_position = select(
        vec4<f32>(2.0, 2.0, 2.0, 1.0),
        position_world_to_clip((world_from_local * vec4(position, 1.0)).xyz),
        valid,
    );
    out.valid = u32(valid);
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    if in.valid == 0u { discard; }
    return vec4<f32>(0.8, 0.8, 0.8, 1.0);
}
