//! Standard-mesh visibility raster output. Virtual geometry must emit the
//! same IDs and barycentric convention.

import bevy_pbr::render::{
    mesh_functions,
    view_transformations::position_world_to_clip
};
import bevy_render::maths::affine3_to_square;

// The bindless material variant uses binding arrays for the optional region
// mask. WESL requires feature enables after imports and before declarations;
// enabling it for the non-bindless specialization is harmless because its
// declarations are compiled out.
enable wgpu_binding_array;

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
    specialization_low: u32, specialization_high: u32,
    lobe_mask: u32,
    extensions_used_low: u32, extensions_used_high: u32,
    extensions_required_low: u32, extensions_required_high: u32,
    dynamic_wetness: u32, dynamic_snow: u32,
    dynamic_damage: u32, dynamic_dissolve: u32,
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

const FEATURE_REGION_ILLUMINATION: u32 = 1u << 7u;
const SEMANTIC_ILLUMINATION_MASK: u32 = 10u;
const REGION_ILLUMINATION_SENTINEL: u32 = 0xffffffffu;
const REGION_ILLUMINATION_LIT: u32 = 0u;
const REGION_ILLUMINATION_NPR: u32 = 1u;

@group(2) @binding(0) var<storage, read> scene_instances: array<GpuSceneInstance>;
@group(2) @binding(1) var<storage, read> scene_current_transforms: array<GpuSceneTransform>;
@group(2) @binding(2) var<storage, read> scene_previous_transforms: array<GpuSceneTransform>;
@group(2) @binding(3) var<storage, read> scene_bounds: array<GpuSceneBounds>;
@group(3) @binding(0) var<storage, read> material_headers: array<MaterialHeader>;
@group(3) @binding(2) var<storage, read> material_textures: array<MaterialTexture>;
@if(MATERIAL_BINDLESS) @group(3) @binding(3) var material_texture_heap: binding_array<texture_2d<f32>>;
@if(MATERIAL_BINDLESS) @group(3) @binding(4) var material_sampler_heap: binding_array<sampler>;
@if(MATERIAL_BINDLESS) @group(3) @binding(9) var<storage, read> material_texture_generation_table: array<u32>;

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @builtin(vertex_index) vertex_index: u32,
    @if(VERTEX_POSITIONS_COMPRESSED)
    @location(0) compressed_position: vec4<f32>,
    @else
    @location(0) position: vec3<f32>,
    @if(VERTEX_UVS)
    @location(2) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) @interpolate(flat) scene_index: u32,
    @location(1) @interpolate(flat) scene_generation: u32,
    @location(2) @interpolate(flat) geometry_lod_or_cluster: u32,
    @location(3) @interpolate(flat) material_index: u32,
    @location(4) @interpolate(flat) material_generation: u32,
    @if(VERTEX_UVS)
    @location(5) uv: vec2<f32>,
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
    @if(VERTEX_UVS) {
        out.uv = vertex.uv;
    } @else {
        out.uv = vec2<f32>(0.0);
    }
    return out;
}

struct FragmentOutput {
    @location(0) ids: vec4<u32>,
    @location(1) metadata: vec4<u32>,
};

fn region_illumination_selector(material_index: u32, uv: vec2<f32>) -> u32 {
    var selector = REGION_ILLUMINATION_SENTINEL;
    @if(MATERIAL_BINDLESS && REGION_ILLUMINATION) {
        if material_index != 0xffffffffu && material_index < arrayLength(&material_headers) {
            let header = material_headers[material_index];
            if header.is_active != 0u
                && (header.feature_flags & FEATURE_REGION_ILLUMINATION) != 0u {
                let physical = arrayLength(&material_textures);
                // Avoid `offset + count` overflow on malformed headers. A
                // missing/out-of-range row leaves the sentinel intact, so
                // classification falls back to the material-wide mode.
                for (var i = header.texture_offset;
                    i < physical && i - header.texture_offset < header.texture_count;
                    i = i + 1u) {
                    let record = material_textures[i];
                    if record.semantic == SEMANTIC_ILLUMINATION_MASK {
                        @if(VERTEX_UVS) {
                            // The arrays are densely populated, but malformed
                            // or stale material rows must never index them out
                            // of bounds. Array lengths are the exact per-device
                            // bindless capacity from MaterialTextureArrays.
                            if record.generation != 0u
                                && record.index < arrayLength(&material_texture_generation_table)
                                && record.sampler_index < arrayLength(&material_texture_generation_table) {
                                if material_texture_generation_table[record.index] != record.generation {
                                    break;
                                }
                                // A level-zero, linearly filtered sample is
                                // intentional: this categorical mask has no
                                // derivatives in visibility, and filtering
                                // gives stable sub-texel region boundaries.
                                // The producer's sampler controls address mode.
                                let mask = textureSampleLevel(
                                    material_texture_heap[record.index],
                                    material_sampler_heap[record.sampler_index],
                                    uv,
                                    0.0,
                                ).r;
                                selector = select(
                                    REGION_ILLUMINATION_LIT,
                                    REGION_ILLUMINATION_NPR,
                                    mask >= 0.5,
                                );
                            }
                        }
                        break;
                    }
                }
            }
        }
    }
    return selector;
}

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
    var selector = REGION_ILLUMINATION_SENTINEL;
    @if(MATERIAL_BINDLESS && REGION_ILLUMINATION && VERTEX_UVS) {
        selector = region_illumination_selector(in.material_index, in.uv);
    }
    out.metadata = vec4<u32>(
        in.material_index,
        in.material_generation,
        selector,
        pack2x16unorm(clamp(barycentrics.xy, vec2(0.0), vec2(1.0))),
    );
    return out;
}
