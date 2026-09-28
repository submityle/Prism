//! Transparent forward pass feeding the weighted-blended OIT (WBOIT) targets.
//!
//! Transparent geometry cannot live in the visibility buffer (one opaque
//! surface per pixel), so it is drawn here in a separate forward pass that
//! accumulates each fragment into the two MRT targets prepared by
//! `shading::transparent::targets` with McGuire & Bavoil blending (JCGT 2013):
//!
//! * `@location(0)` -> accumulation target (`Rgba16Float`, hardware blend
//!   `One`/`One` add): weighted premultiplied colour in `rgb`, summed weighted
//!   alpha in `a`;
//! * `@location(1)` -> revealage target (`R16Float`, hardware blend
//!   `Zero`/`Src`): the running product of `1 - alpha`.
//!
//! The per-fragment arithmetic is `oit_accumulate`, imported from the GPU twin
//! `shaders/oit.wesl` which is kept bit-aligned with the CPU golden
//! `prism_render_shading::oit`. The vertex stage mirrors `opaque.wesl` (GPU
//! Scene instance addressing + optional compressed positions), and additionally
//! derives the positive view-space depth the OIT weight needs.
//!
//! Shading here is the honest unlit baseline (base colour + emissive), matching
//! the opaque forward convention: the real lit resolve chain only owns opaque
//! surfaces via the visibility buffer, so inventing a light here would mislead.
//! Lit transparency (clustered light consumption in this pass) is a documented
//! follow-up, not silently faked.

import bevy_pbr::render::{
    mesh_view_bindings::view,
    view_transformations::{position_world_to_clip, position_world_to_view}
};
import bevy_render::maths::affine3_to_square;
import prism_render_scene::shaders::oit::{OitTargets, oit_accumulate};

struct GpuSceneInstance {
    generation: u32, flags: u32,
    geometry_index: u32, geometry_generation: u32,
    material_index: u32, material_generation: u32,
    render_layers: u32, is_active: u32,
}
struct GpuSceneTransform { world_from_local: mat3x4<f32> }
struct GpuSceneBounds { center_radius: vec4<f32>, half_extents: vec4<f32> }
struct MaterialHeader {
    generation: u32, revision: u32, illumination: u32, render_class: u32,
    feature_flags: u32, closure_mask: u32, parameter_offset: u32, parameter_size: u32,
    texture_offset: u32, texture_count: u32, sampler_offset: u32, sampler_count: u32,
    custom_program: u32, is_active: u32, material_epoch_low: u32, material_epoch_high: u32,
    closure_graph_offset: u32, specialization_low: u32, specialization_high: u32, reserved: u32,
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

struct Vertex {
    @builtin(instance_index) instance_index: u32,
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
    @location(1) @interpolate(flat) material_index: u32,
    @location(2) world_position: vec3<f32>,
    // Positive view-space distance from the camera, consumed by the OIT weight.
    @location(3) view_depth: f32,
};

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    let scene_index = vertex.instance_index;
    let scene_in_bounds = scene_index < arrayLength(&scene_instances);
    let safe_scene_index = select(0u, scene_index, scene_in_bounds);
    let instance = scene_instances[safe_scene_index];
    let valid = scene_in_bounds && instance.is_active != 0u && instance.generation != 0u;
    @if(VERTEX_POSITIONS_COMPRESSED) {
        let bounds = scene_bounds[safe_scene_index];
        let position = bevy_render::utils::decompress_vertex_position(
            vertex.compressed_position,
            bounds.center_radius.xyz,
            bounds.half_extents.xyz,
        );
    } @else {
        let position = vertex.position;
    }
    let world_from_local = affine3_to_square(
        scene_current_transforms[safe_scene_index].world_from_local
    );
    var out: VertexOutput;
    let world_position = (world_from_local * vec4(position, 1.0)).xyz;
    out.clip_position = select(
        vec4<f32>(2.0, 2.0, 2.0, 1.0),
        position_world_to_clip(world_position),
        valid,
    );
    out.world_position = world_position;
    // View space looks down -z, so negating the view-space z gives the positive
    // linear distance from the camera the McGuire/Bavoil weight expects.
    out.view_depth = max(-position_world_to_view(world_position).z, 0.0);
    out.valid = u32(valid);
    let material_in_bounds = instance.material_index < arrayLength(&material_headers);
    out.material_index = select(0u, instance.material_index, material_in_bounds);
    return out;
}

// MRT contribution the transparent forward pass writes. The hardware blend then
// combines it with the accumulation (additive) and revealage (multiplicative)
// targets, exactly reproducing OitAccumulation::accumulate on the CPU golden.
struct FragmentOutput {
    // location(0): weighted premultiplied colour (rgb) + weighted alpha (a),
    // blended One/One (additive) into the Rgba16Float accumulation target.
    @location(0) accum: vec4<f32>,
    // location(1): the (1 - alpha) factor, blended Zero/Src (multiplicative)
    // into the R16Float revealage target so it becomes the running product.
    @location(1) revealage: f32,
}

@fragment
fn fragment(in: VertexOutput) -> FragmentOutput {
    if in.valid == 0u { discard; }
    // Honest unlit baseline: base colour + emissive, straight (non-premultiplied)
    // linear RGB. The alpha drives coverage. oit_accumulate premultiplies and
    // applies the depth weight, matching the golden bit-for-bit.
    let surface = material_parameters[material_headers[in.material_index].parameter_offset];
    let color = surface.base_color.rgb + surface.emissive.rgb;
    let alpha = clamp(surface.base_color.a, 0.0, 1.0);
    let targets: OitTargets = oit_accumulate(color, alpha, in.view_depth);
    var out: FragmentOutput;
    out.accum = targets.accum;
    out.revealage = targets.revealage;
    return out;
}
