//! Standard-mesh visibility raster output. Virtual geometry must emit the
//! same IDs and barycentric convention.

import bevy_pbr::render::{
    mesh_functions,
    view_transformations::position_world_to_clip
};
import bevy_render::maths::affine3_to_square;

struct GpuSceneInstance {
    generation: u32, flags: u32,
    geometry_index: u32, geometry_generation: u32,
    material_index: u32, material_generation: u32,
    render_layers: u32, is_active: u32,
}
struct GpuSceneTransform { world_from_local: mat3x4<f32> }
struct GpuSceneBounds { center_radius: vec4<f32>, half_extents: vec4<f32> }

@group(2) @binding(0) var<storage, read> scene_instances: array<GpuSceneInstance>;
@group(2) @binding(1) var<storage, read> scene_current_transforms: array<GpuSceneTransform>;
@group(2) @binding(2) var<storage, read> scene_previous_transforms: array<GpuSceneTransform>;
@group(2) @binding(3) var<storage, read> scene_bounds: array<GpuSceneBounds>;

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @builtin(vertex_index) vertex_index: u32,
    @if(VERTEX_POSITIONS_COMPRESSED)
    @location(0) compressed_position: vec4<f32>,
    @else
    @location(0) position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) @interpolate(flat) scene_index: u32,
    @location(1) @interpolate(flat) scene_generation: u32,
    @location(2) @interpolate(flat) geometry_lod_or_cluster: u32,
    @location(3) @interpolate(flat) material_index: u32,
    @location(4) @interpolate(flat) material_generation: u32,
};

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    let scene_in_bounds = vertex.instance_index < arrayLength(&scene_instances);
    let scene_index = select(0u, vertex.instance_index, scene_in_bounds);
    let instance = scene_instances[scene_index];
    let valid = scene_in_bounds && instance.is_active != 0u && instance.generation != 0u;
    @if(VERTEX_POSITIONS_COMPRESSED) {
        let bounds = scene_bounds[scene_index];
        let local_position = bevy_render::utils::decompress_vertex_position(
            vertex.compressed_position,
            bounds.center_radius.xyz,
            bounds.half_extents.xyz,
        );
    } @else {
        let local_position = vertex.position;
    }
    let world_from_local = affine3_to_square(scene_current_transforms[scene_index].world_from_local);
    let world_position = (world_from_local * vec4(local_position, 1.0)).xyz;
    var out: VertexOutput;
    out.clip_position = select(
        vec4<f32>(2.0, 2.0, 2.0, 1.0),
        position_world_to_clip(world_position),
        valid,
    );
    out.scene_index = select(0xffffffffu, scene_index, valid);
    out.scene_generation = select(0u, instance.generation, valid);
    out.geometry_lod_or_cluster = 0u;
    out.material_index = select(0xffffffffu, instance.material_index, valid);
    out.material_generation = select(0u, instance.material_generation, valid);
    return out;
}

struct FragmentOutput {
    @location(0) ids: vec4<u32>,
    @location(1) metadata: vec4<u32>,
    @location(2) barycentrics: vec2<f32>,
};

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(primitive_index) primitive_index: u32,
    @builtin(barycentric) barycentrics: vec3<f32>,
) -> FragmentOutput {
    var out: FragmentOutput;
    out.ids = vec4<u32>(
        in.scene_index,
        in.scene_generation,
        primitive_index,
        in.geometry_lod_or_cluster,
    );
    out.metadata = vec4<u32>(
        in.material_index,
        in.material_generation,
        0xffu,
        0u,
    );
    out.barycentrics = clamp(barycentrics.xy, vec2(0.0), vec2(1.0));
    return out;
}
