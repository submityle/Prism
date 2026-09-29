//! Minimal opaque consumer proving stable GPU Scene addressing.

import bevy_pbr::render::{
    mesh_functions,
    mesh_view_bindings::view,
    view_transformations::position_world_to_clip
};
import bevy_render::maths::affine3_to_square;
import prism_render_scene::shaders::material::PrismSurfaceParameters;
import prism_render_scene::shaders::material_unpack::{
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
    generation: u32, revision: u32, illumination: u32, render_class: u32,
    feature_flags: u32, closure_mask: u32, parameter_offset: u32, parameter_size: u32,
    texture_offset: u32, texture_count: u32, sampler_offset: u32, sampler_count: u32,
    custom_program: u32, is_active: u32, material_epoch_low: u32, material_epoch_high: u32,
    closure_graph_offset: u32, specialization_low: u32, specialization_high: u32, lobe_mask: u32,
}
struct MaterialTexture { index: u32, generation: u32, semantic: u32, sampler_index: u32 }

@group(2) @binding(0) var<storage, read> scene_instances: array<GpuSceneInstance>;
@group(2) @binding(1) var<storage, read> scene_current_transforms: array<GpuSceneTransform>;
@group(2) @binding(2) var<storage, read> scene_previous_transforms: array<GpuSceneTransform>;
@group(2) @binding(3) var<storage, read> scene_bounds: array<GpuSceneBounds>;
@group(3) @binding(0) var<storage, read> material_headers: array<MaterialHeader>;
@group(3) @binding(1) var<storage, read> material_parameters: array<u32>;
@group(3) @binding(2) var<storage, read> material_textures: array<MaterialTexture>;

// Gather a material's variable-length packed block out of the shared u32 heap and
// expand it to a full surface. A declared range that overruns the bound heap
// decodes to the neutral surface, so callers never index out of bounds.
fn load_surface_parameters(word_offset: u32, lobe_mask: u32) -> PrismSurfaceParameters {
    let n = min(prism_packed_word_count(lobe_mask), 36u);
    if word_offset + n > arrayLength(&material_parameters) {
        return prism_neutral_surface();
    }
    var words: array<u32, 36>;
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
            && material_header.generation == instance.material_generation,
    );
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
    } @else {
        let header = material_headers[out.material_index];
        out.debug_color = load_surface_parameters(header.parameter_offset, header.lobe_mask).base_color.rgb;
    }
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    if in.valid == 0u { discard; }
    @if(PRISM_DEBUG_VIEW_1 || PRISM_DEBUG_VIEW_2 || PRISM_DEBUG_VIEW_3 || PRISM_DEBUG_VIEW_4) {
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
    return vec4<f32>(surface.base_color.rgb + surface.emissive.rgb, surface.base_color.a);
}
