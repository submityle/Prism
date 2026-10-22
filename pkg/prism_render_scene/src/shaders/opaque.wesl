//! Minimal opaque consumer proving stable GPU Scene addressing.

import bevy_pbr::render::{
    mesh_functions,
    mesh_view_bindings::view,
    view_transformations::position_world_to_clip
};
import bevy_render::maths::affine3_to_square;
import prism_render_material::shaders::material::PrismSurfaceParameters;
import prism_render_scene::shaders::dynamic_mask::{
    dyn_dissolve_coverage,
};
import prism_render_material::shaders::material_unpack::{
    prism_unpack_surface, prism_packed_word_count, prism_neutral_surface
};

struct GpuSceneInstance {
    generation: u32, flags: u32,
    geometry_index: u32, geometry_generation: u32,
    material_index: u32, material_generation: u32,
    render_layers: u32, is_active: u32,
}
struct GpuSceneTransform { world_from_local: mat3x4<f32> }
struct GpuSceneBounds { center_radius: vec4<f32>, half_extents: vec4<f32> }
struct MaterialHeader {
    generation: u32, revision: u32, render_class: u32, render_domain: u32,
    feature_flags: u32, closure_mask: u32, parameter_offset: u32, parameter_size: u32,
    texture_offset: u32, texture_count: u32, sampler_offset: u32, sampler_count: u32,
    is_active: u32, material_epoch_low: u32, material_epoch_high: u32,
    lobe_mask: u32,
    // PCM extension digest (§5.3): used/required, each split low/high.
    extensions_used_low: u32, extensions_used_high: u32,
    extensions_required_low: u32, extensions_required_high: u32,
    // Runtime dynamic channels (§17.1): wetness / snow / damage / dissolve,
    // each an `f32` bitcast into the word (`0.0` = exact identity).
    dynamic_wetness: u32, dynamic_snow: u32,
    dynamic_damage: u32, dynamic_dissolve: u32,
    // ABI v8 execution binding; all five words use 0xffffffff when unbound.
    execution_plan_index: u32, execution_program_index: u32,
    execution_package_index: u32, execution_permutation_low: u32,
    execution_permutation_high: u32,
}
struct MaterialTexture {
    index: u32,
    generation: u32,
    semantic: u32,
    sampler_index: u32,
    virtual_texture_id: u32,
    virtual_texture_width: u32,
    virtual_texture_height: u32,
    virtual_texture_page_size: u32,
}
struct OpaqueDebugMetadata {
    abi_version: u32, view_index: u32, view_generation: u32, scene_index: u32,
    scene_generation: u32, lod_level: u32, cluster_index: u32,
    cluster_generation: u32, cluster_valid: u32, flags: u32,
    reserved_0: u32, reserved_1: u32,
}

@group(2) @binding(0) var<storage, read> scene_instances: array<GpuSceneInstance>;
@group(2) @binding(1) var<storage, read> scene_current_transforms: array<GpuSceneTransform>;
@group(2) @binding(2) var<storage, read> scene_previous_transforms: array<GpuSceneTransform>;
@group(2) @binding(3) var<storage, read> scene_bounds: array<GpuSceneBounds>;
@group(3) @binding(0) var<storage, read> material_headers: array<MaterialHeader>;
@group(3) @binding(1) var<storage, read> material_parameters: array<u32>;
@group(3) @binding(2) var<storage, read> material_textures: array<MaterialTexture>;
@group(4) @binding(0) var<storage, read> opaque_debug_metadata: array<OpaqueDebugMetadata>;

// Gather a material's variable-length packed block out of the shared u32 heap and
// expand it to a full surface. A declared range that overruns the bound heap
// decodes to the neutral surface, so callers never index out of bounds.
fn load_surface_parameters(word_offset: u32, lobe_mask: u32) -> PrismSurfaceParameters {
    let n = min(prism_packed_word_count(lobe_mask), 40u);
    let heap_words = arrayLength(&material_parameters);
    if word_offset > heap_words || n > heap_words - word_offset {
        return prism_neutral_surface();
    }
    var words: array<u32, 40>;
    for (var i = 0u; i < n; i = i + 1u) {
        words[i] = material_parameters[word_offset + i];
    }
    return prism_unpack_surface(words, lobe_mask);
}

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
    @location(1) @interpolate(flat) debug_color: vec3<f32>,
    @location(2) @interpolate(flat) material_index: u32,
    @location(3) world_position: vec3<f32>,
    @location(4) world_normal: vec3<f32>,
    @location(5) uv: vec2<f32>,
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
    @if(VERTEX_NORMALS) {
        let linear_world_from_local = mat3x3<f32>(
            world_from_local[0].xyz,
            world_from_local[1].xyz,
            world_from_local[2].xyz,
        );
        let transformed_normal = inverse_transpose_3x3(linear_world_from_local) * vertex.normal;
        let normal_length_squared = dot(transformed_normal, transformed_normal);
        out.world_normal = select(
            vec3<f32>(0.0),
            transformed_normal * inverseSqrt(max(normal_length_squared, 1.0e-12)),
            normal_length_squared > 1.0e-12,
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
            && material_header.generation == instance.material_generation,
    );
    var debug_metadata = OpaqueDebugMetadata(0u, 0u, 0u, 0u, 0u, 0xffffffffu,
        0xffffffffu, 0u, 0u, 0u, 0u, 0u);
    var debug_metadata_valid = false;
    @if(PRISM_DEBUG_VIEW_6 || PRISM_DEBUG_VIEW_7 || PRISM_DEBUG_VIEW_8 || PRISM_DEBUG_VIEW_9 || PRISM_DEBUG_VIEW_10) {
        let debug_metadata_in_bounds = scene_index < arrayLength(&opaque_debug_metadata);
        let debug_metadata_index = select(0u, scene_index, debug_metadata_in_bounds);
        debug_metadata = opaque_debug_metadata[debug_metadata_index];
        debug_metadata_valid = debug_metadata_in_bounds
            && debug_metadata.abi_version == 1u
            && debug_metadata.view_generation != 0u
            && debug_metadata.scene_index == scene_index
            && debug_metadata.scene_generation == instance.generation;
    }
    let id_color = vec3<f32>(
        f32((safe_scene_index * 97u) & 255u),
        f32((safe_scene_index * 57u) & 255u),
        f32((safe_scene_index * 17u) & 255u),
    ) / 255.0;
    @if(PRISM_DEBUG_VIEW_1) { out.debug_color = id_color; }
    @else if(PRISM_DEBUG_VIEW_2) { out.debug_color = vec3<f32>(f32(instance.geometry_index & 255u) / 255.0, 0.2, 0.8); }
    @else if(PRISM_DEBUG_VIEW_3) { out.debug_color = vec3<f32>(0.8, f32(instance.material_index & 255u) / 255.0, 0.2); }
    @else if(PRISM_DEBUG_VIEW_4) {
        let previous = affine3_to_square(scene_previous_transforms[safe_scene_index].world_from_local);
        out.debug_color = abs((world_from_local[3] - previous[3]).xyz);
    } @else if(PRISM_DEBUG_VIEW_5) {
        // World-normal visualization does not depend on material parameters.
        out.debug_color = vec3<f32>(0.0);
    } @else if(PRISM_DEBUG_VIEW_6) {
        let lod = debug_metadata.lod_level & 255u;
        out.debug_color = select(
            vec3<f32>(1.0, 0.0, 1.0),
            vec3<f32>(f32(lod * 97u & 255u) / 255.0, f32(lod * 57u & 255u) / 255.0, f32(lod * 17u & 255u) / 255.0),
            debug_metadata_valid && debug_metadata.lod_level != 0xffffffffu,
        );
    } @else if(PRISM_DEBUG_VIEW_7) {
        out.debug_color = select(vec3<f32>(1.0, 0.0, 1.0), id_color, debug_metadata_valid);
    } @else if(PRISM_DEBUG_VIEW_8) {
        // Opaque mesh work currently carries geometry/LOD identity, not the
        // virtual-cluster identity. Never substitute geometry identity here.
        out.debug_color = select(vec3<f32>(1.0, 0.0, 1.0), vec3<f32>(0.0),
            debug_metadata_valid && debug_metadata.cluster_valid == 1u
                && debug_metadata.cluster_index != 0xffffffffu
                && debug_metadata.cluster_generation != 0u);
        if debug_metadata_valid && debug_metadata.cluster_valid == 1u
            && debug_metadata.cluster_index != 0xffffffffu
            && debug_metadata.cluster_generation != 0u {
            let cluster = debug_metadata.cluster_index;
            out.debug_color = vec3<f32>(f32((cluster * 97u) & 255u),
                f32((cluster * 57u) & 255u), f32((cluster * 17u) & 255u)) / 255.0;
        }
    } @else if(PRISM_DEBUG_VIEW_9) {
        let requested_cut = debug_metadata.reserved_0;
        let cut = requested_cut & 255u;
        let base = vec3<f32>(f32((cut * 97u) & 255u),
            f32((cut * 57u) & 255u), f32((cut * 17u) & 255u)) / 255.0;
        out.debug_color = select(vec3<f32>(1.0, 0.0, 1.0), base, debug_metadata_valid);
        if debug_metadata_valid && (debug_metadata.flags & 4u) != 0u {
            out.debug_color = mix(base, vec3<f32>(1.0, 0.8, 0.0), 0.65);
        }
    } @else if(PRISM_DEBUG_VIEW_10) {
        out.debug_color = select(vec3<f32>(1.0, 0.0, 1.0),
            select(vec3<f32>(1.0, 0.1, 0.05), vec3<f32>(0.1, 0.9, 0.2),
                (debug_metadata.flags & 2u) != 0u), debug_metadata_valid);
    } @else {
        let header = material_headers[out.material_index];
        out.debug_color = load_surface_parameters(header.parameter_offset, header.lobe_mask).base_color.rgb;
    }
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    if in.valid == 0u { discard; }
    @if(PRISM_DEBUG_VIEW_5) {
        @if(VERTEX_NORMALS) {
            let normal_length_squared = dot(in.world_normal, in.world_normal);
            let normal = in.world_normal * inverseSqrt(max(normal_length_squared, 1.0e-12));
            let encoded = normal * 0.5 + vec3<f32>(0.5);
            return select(
                vec4<f32>(1.0, 0.0, 1.0, 1.0),
                vec4<f32>(encoded, 1.0),
                normal_length_squared > 1.0e-12,
            );
        } @else {
            // Missing mesh normals are data absence, not a synthetic up-vector.
            return vec4<f32>(1.0, 0.0, 1.0, 1.0);
        }
    }
    @if(PRISM_DEBUG_VIEW_1 || PRISM_DEBUG_VIEW_2 || PRISM_DEBUG_VIEW_3 || PRISM_DEBUG_VIEW_4 || PRISM_DEBUG_VIEW_6 || PRISM_DEBUG_VIEW_7 || PRISM_DEBUG_VIEW_8 || PRISM_DEBUG_VIEW_9 || PRISM_DEBUG_VIEW_10) {
        return vec4<f32>(in.debug_color, 1.0);
    }
    // Honest fallback shading. The real lighting lives in the GPU shading resolve
    // chain (visibility_raster -> material_classification -> shading_resolve ->
    // composite); `composite.wesl` overwrites every *covered* view-target pixel
    // with the resolved linear HDR radiance after this main pass has run. This
    // forward path only remains authoritative when the visibility buffer is
    // disabled (feature off, or MSAA != 1). It must therefore NOT invent lighting
    // with a hard-coded light direction -- doing so would fight the resolve chain
    // and mislead debugging. Emit unlit base color + emissive only.
    let material_header = material_headers[in.material_index];
    let surface = load_surface_parameters(material_header.parameter_offset, material_header.lobe_mask);
    // `OpenPBR` `geometry_opacity` scales the coverage the blend/mask path reads,
    // and the runtime dissolve channel (§17.1) cuts that coverage against the
    // procedural mask — the same mask the shading resolve samples to light its
    // rim, so the visible edge and the coverage edge agree.
    let dissolve = bitcast<f32>(material_header.dynamic_dissolve);
    let coverage = dyn_dissolve_coverage(dissolve, in.uv);
    return vec4<f32>(
        surface.base_color.rgb + surface.emissive.rgb,
        surface.base_color.a * surface.geometry_opacity * coverage,
    );
}
