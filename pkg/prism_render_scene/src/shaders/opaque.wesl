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

struct Vertex {
    @if(VERTEX_POSITIONS_COMPRESSED)
    @location(0) compressed_position: vec4<f32>,
    @else
    @location(0) position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) @interpolate(flat) valid: u32,
    @location(1) @interpolate(flat) debug_color: vec3<f32>,
};

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    let instance = scene_instances[scene_address.index];
    let valid = instance.active != 0u && instance.generation == scene_address.generation;
    @if(VERTEX_POSITIONS_COMPRESSED) {
        let bounds = scene_bounds[scene_address.index];
        let position = bevy_render::utils::decompress_vertex_position(
            vertex.compressed_position,
            bounds.center_radius.xyz,
            bounds.half_extents.xyz,
        );
    } @else {
        let position = vertex.position;
    }
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
    let id_color = vec3<f32>(
        f32((scene_address.index * 97u) & 255u),
        f32((scene_address.index * 57u) & 255u),
        f32((scene_address.index * 17u) & 255u),
    ) / 255.0;
    @if(PRISM_DEBUG_VIEW == 1) out.debug_color = id_color;
    @else if(PRISM_DEBUG_VIEW == 2) out.debug_color = vec3<f32>(f32(instance.geometry_index & 255u) / 255.0, 0.2, 0.8);
    @else if(PRISM_DEBUG_VIEW == 3) out.debug_color = vec3<f32>(0.8, f32(instance.material_index & 255u) / 255.0, 0.2);
    @else if(PRISM_DEBUG_VIEW == 4) {
        let previous = affine3_to_square(scene_previous_transforms[scene_address.index].world_from_local);
        out.debug_color = abs((world_from_local[3] - previous[3]).xyz);
    } @else out.debug_color = vec3<f32>(0.8);
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    if in.valid == 0u { discard; }
    return vec4<f32>(in.debug_color, 1.0);
}
