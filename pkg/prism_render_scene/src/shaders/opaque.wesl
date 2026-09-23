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
struct MaterialHeader {
    generation: u32, revision: u32, shading_model: u32, render_class: u32,
    feature_flags: u32, closure_mask: u32, parameter_offset: u32, parameter_size: u32,
    texture_offset: u32, texture_count: u32, sampler_offset: u32, sampler_count: u32,
    custom_program: u32, active: u32, material_epoch_low: u32, material_epoch_high: u32,
}
struct SurfaceParameters {
    base_color: vec4<f32>, emissive: vec4<f32>,
    metallic: f32, perceptual_roughness: f32, reflectance: f32, ambient_occlusion: f32,
    normal_scale: f32, alpha_cutoff: f32, transmission: f32, thickness: f32,
    clearcoat: f32, clearcoat_roughness: f32, anisotropy: f32, anisotropy_rotation: f32,
    sheen: f32, subsurface: f32, index_of_refraction: f32, dispersion: f32,
}
struct MaterialTexture { index: u32, generation: u32, semantic: u32, sampler_index: u32 }

@group(2) @binding(0) var<storage, read> scene_instances: array<GpuSceneInstance>;
@group(2) @binding(1) var<storage, read> scene_current_transforms: array<GpuSceneTransform>;
@group(2) @binding(2) var<storage, read> scene_previous_transforms: array<GpuSceneTransform>;
@group(2) @binding(3) var<storage, read> scene_bounds: array<GpuSceneBounds>;
@group(3) @binding(0) var<storage, read> material_headers: array<MaterialHeader>;
@group(3) @binding(1) var<storage, read> material_parameters: array<SurfaceParameters>;
@group(3) @binding(2) var<storage, read> material_textures: array<MaterialTexture>;

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
    @location(2) @interpolate(flat) material_index: u32,
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
    let material_header = material_headers[instance.material_index];
    out.material_index = select(
        0u,
        instance.material_index,
        material_header.active != 0u && material_header.generation == instance.material_generation,
    );
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
    } @else out.debug_color = material_parameters[out.material_index].base_color.rgb;
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    if in.valid == 0u { discard; }
    return vec4<f32>(in.debug_color, 1.0);
}
