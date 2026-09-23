//! Minimal opaque consumer proving stable GPU Scene addressing.

import bevy_pbr::render::{
    mesh_functions,
    mesh_view_bindings::view,
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
struct MaterialHeader {
    generation: u32, revision: u32, shading_model: u32, render_class: u32,
    feature_flags: u32, closure_mask: u32, parameter_offset: u32, parameter_size: u32,
    texture_offset: u32, texture_count: u32, sampler_offset: u32, sampler_count: u32,
    custom_program: u32, is_active: u32, material_epoch_low: u32, material_epoch_high: u32,
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

fn inverse_transpose_3x3(value: mat3x3<f32>) -> mat3x3<f32> {
    let x = cross(value[1], value[2]);
    let y = cross(value[2], value[0]);
    let z = cross(value[0], value[1]);
    let determinant = dot(value[2], z);
    if abs(determinant) <= 1.0e-8 {
        return mat3x3<f32>(1.0);
    }
    return mat3x3<f32>(x, y, z) / determinant;
}

struct Vertex {
    @if(VERTEX_POSITIONS_COMPRESSED)
    @location(0) compressed_position: vec4<f32>,
    @else
    @location(0) position: vec3<f32>,
    @if(VERTEX_NORMALS)
    @location(1) normal: vec3<f32>,
    @if(VERTEX_UVS)
    @location(2) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) @interpolate(flat) valid: u32,
    @location(1) @interpolate(flat) debug_color: vec3<f32>,
    @location(2) @interpolate(flat) material_index: u32,
    @location(3) world_position: vec3<f32>,
    @location(4) world_normal: vec3<f32>,
    @location(5) uv: vec2<f32>,
};

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    let instance = scene_instances[scene_address.index];
    let valid = instance.is_active != 0u && instance.generation == scene_address.generation;
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
    let world_position = (world_from_local * vec4(position, 1.0)).xyz;
    out.clip_position = select(
        vec4<f32>(2.0, 2.0, 2.0, 1.0),
        position_world_to_clip(world_position),
        valid,
    );
    out.world_position = world_position;
    @if(VERTEX_NORMALS) {
        let linear_world_from_local = mat3x3<f32>(
            world_from_local[0].xyz,
            world_from_local[1].xyz,
            world_from_local[2].xyz,
        );
        let transformed_normal = inverse_transpose_3x3(linear_world_from_local) * vertex.normal;
        out.world_normal = select(
            vec3<f32>(0.0, 1.0, 0.0),
            normalize(transformed_normal),
            dot(transformed_normal, transformed_normal) > 1.0e-12,
        );
    } @else {
        out.world_normal = vec3<f32>(0.0, 1.0, 0.0);
    }
    @if(VERTEX_UVS) { out.uv = vertex.uv; }
    @else { out.uv = vec2<f32>(0.0); }
    out.valid = u32(valid);
    let material_in_bounds = instance.material_index < arrayLength(&material_headers);
    let requested_material_index = select(0u, instance.material_index, material_in_bounds);
    let material_header = material_headers[requested_material_index];
    out.material_index = select(
        0u,
        requested_material_index,
        material_in_bounds
            && material_header.is_active != 0u
            && material_header.generation == instance.material_generation
            && material_header.parameter_offset < arrayLength(&material_parameters),
    );
    let id_color = vec3<f32>(
        f32((scene_address.index * 97u) & 255u),
        f32((scene_address.index * 57u) & 255u),
        f32((scene_address.index * 17u) & 255u),
    ) / 255.0;
    @if(PRISM_DEBUG_VIEW_1) { out.debug_color = id_color; }
    @else if(PRISM_DEBUG_VIEW_2) { out.debug_color = vec3<f32>(f32(instance.geometry_index & 255u) / 255.0, 0.2, 0.8); }
    @else if(PRISM_DEBUG_VIEW_3) { out.debug_color = vec3<f32>(0.8, f32(instance.material_index & 255u) / 255.0, 0.2); }
    @else if(PRISM_DEBUG_VIEW_4) {
        let previous = affine3_to_square(scene_previous_transforms[scene_address.index].world_from_local);
        out.debug_color = abs((world_from_local[3] - previous[3]).xyz);
    } @else {
        let header = material_headers[out.material_index];
        out.debug_color = material_parameters[header.parameter_offset].base_color.rgb;
    }
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    if in.valid == 0u { discard; }
    @if(PRISM_DEBUG_VIEW_1 || PRISM_DEBUG_VIEW_2 || PRISM_DEBUG_VIEW_3 || PRISM_DEBUG_VIEW_4) {
        return vec4<f32>(in.debug_color, 1.0);
    }
    let surface = material_parameters[material_headers[in.material_index].parameter_offset];
    let N = normalize(in.world_normal);
    let L = normalize(vec3<f32>(0.4, 0.8, 0.3));
    let to_view = view.world_position - in.world_position;
    let V = select(N, normalize(to_view), dot(to_view, to_view) > 1.0e-12);
    let H = normalize(L + V);
    let NdotL = max(dot(N, L), 0.0);
    let NdotH = max(dot(N, H), 0.0);
    let roughness = clamp(surface.perceptual_roughness, 0.045, 1.0);
    let diffuse = surface.base_color.rgb * (1.0 - surface.metallic) * NdotL;
    let f0 = mix(vec3<f32>(0.04 * surface.reflectance), surface.base_color.rgb, surface.metallic);
    let specular = f0 * pow(NdotH, max(2.0 / (roughness * roughness) - 2.0, 1.0)) * NdotL;
    let ambient = surface.base_color.rgb * 0.03 * surface.ambient_occlusion;
    return vec4<f32>(ambient + diffuse + specular + surface.emissive.rgb, surface.base_color.a);
}
