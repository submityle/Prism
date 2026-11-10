import prism_render_material::shaders::material::{PrismMaterialHeader, prism_material_is_valid};
import prism_render_material::shaders::material_unpack::{PRISM_LOBE_ALL, PRISM_SURFACE_MAX_WORDS, prism_packed_word_count, prism_unpack_surface};
import prism_render_material::shaders::material::{PrismMaterialTexture};
import prism_render_material::shaders::material_sample::{
    SampledMaterial,
    fold_material_texel,
    sample_texture_slot,
    sampled_material_defaults,
};
import prism_render_material::shaders::material_virtual_texture::MaterialVirtualTextureConfig;

struct SurfaceFrame {
    clip_from_world: mat4x4<f32>,
    previous_clip_from_world: mat4x4<f32>,
    exposure_stops: f32,
    camera_history_valid: u32,
    _padding: vec2<f32>,
};

struct SurfaceMaterial {
    index: u32,
    generation: u32,
    revision: u32,
    execution_plan_index: u32,
};

@group(0) @binding(0) var<uniform> frame: SurfaceFrame;
@group(0) @binding(1) var<uniform> material: SurfaceMaterial;
@group(0) @binding(2) var<storage, read> material_headers: array<PrismMaterialHeader>;
@group(0) @binding(3) var<storage, read> material_parameters: array<u32>;
@group(0) @binding(4) var<storage, read> material_textures: array<PrismMaterialTexture>;

@group(1) @binding(5) var<storage, read> material_vt_page_table: array<u32>;
@group(1) @binding(6) var<storage, read> material_vt_config: MaterialVirtualTextureConfig;
@group(1) @binding(7) var material_vt_atlas: texture_2d_array<f32>;
@group(1) @binding(8) var material_vt_sampler: sampler;

struct VertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv0: vec2<f32>,
};

struct SurfaceSample {
    position_x: f32,
    position_y: f32,
    position_z: f32,
    normal: vec4<f32>,
    albedo: vec4<f32>,
    surface_id: u32,
    surface_data: vec4<f32>,
    sampled_material_a: vec4<f32>,
    sampled_material_b: vec4<f32>,
    sampled_material_c: u32,
};

struct GeometryGBufferOut {
    @location(0) position_x: f32,
    @location(1) position_y: f32,
    @location(2) position_z: f32,
    @location(3) normal: vec4<f32>,
    @location(4) albedo: vec4<f32>,
};

struct MaterialGBufferOut {
    @location(0) surface_id: u32,
    @location(1) surface_data: vec4<f32>,
    @location(2) sampled_material_a: vec4<f32>,
    @location(3) sampled_material_b: vec4<f32>,
    @location(4) sampled_material_c: u32,
};

struct SurfaceDataFrame {
    valid: bool,
    tangent_angle: f32,
    footprint: f32,
};

// WGSL has no portable isNan/isInf built-ins.  The self-subtraction check
// rejects both NaN and infinities while remaining valid for every backend.
fn surface_is_finite(value: f32) -> bool {
    return (value - value) == 0.0;
}

fn surface_vec2_is_finite(value: vec2<f32>) -> bool {
    return surface_is_finite(value.x) && surface_is_finite(value.y);
}

fn surface_vec3_is_finite(value: vec3<f32>) -> bool {
    return surface_is_finite(value.x) && surface_is_finite(value.y) && surface_is_finite(value.z);
}

fn normalize_or(value: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let length_squared = dot(value, value);
    if (length_squared > 1.0e-12) {
        return value * inverseSqrt(length_squared);
    }
    return fallback;
}

fn oct_encode_normal(normal: vec3<f32>) -> vec2<f32> {
    let n = normalize_or(normal, vec3<f32>(0.0, 1.0, 0.0));
    let inv_l1 = 1.0 / max(abs(n.x) + abs(n.y) + abs(n.z), 1.0e-8);
    var p = n.xy * inv_l1;
    if (n.z < 0.0) {
        p = (1.0 - abs(p.yx)) * vec2<f32>(select(-1.0, 1.0, p.x >= 0.0), select(-1.0, 1.0, p.y >= 0.0));
    }
    return p * 0.5 + vec2<f32>(0.5);
}

fn pack_material_c(oct: vec2<f32>, roughness: f32, valid: bool) -> u32 {
    let x = u32(clamp(oct.x, 0.0, 1.0) * 255.0 + 0.5);
    let y = u32(clamp(oct.y, 0.0, 1.0) * 255.0 + 0.5);
    let z = u32(clamp(roughness, 0.0, 1.0) * 255.0 + 0.5);
    let w = select(0u, 255u, valid);
    return x | (y << 8u) | (z << 16u) | (w << 24u);
}

fn transform_normal(row0: vec4<f32>, row1: vec4<f32>, row2: vec4<f32>, normal: vec3<f32>) -> vec3<f32> {
    let cofactor0 = cross(row1.xyz, row2.xyz);
    let cofactor1 = cross(row2.xyz, row0.xyz);
    let cofactor2 = cross(row0.xyz, row1.xyz);
    let determinant = dot(row0.xyz, cofactor0);
    if (abs(determinant) > 1.0e-8) {
        return normalize_or(vec3<f32>(dot(cofactor0, normal), dot(cofactor1, normal), dot(cofactor2, normal)) / determinant, vec3<f32>(0.0, 1.0, 0.0));
    }
    return normalize_or(normal, vec3<f32>(0.0, 1.0, 0.0));
}

fn surface_texture_lod(row: PrismMaterialTexture, footprint: f32) -> f32 {
    let width = max(row.virtual_texture_width, 1u);
    let height = max(row.virtual_texture_height, 1u);
    let extent = max(footprint * f32(max(width, height)), 1.0);
    if (!surface_is_finite(extent) || extent <= 1.0) {
        return 0.0;
    }
    return log2(extent);
}

fn sample_surface_material(
    header: PrismMaterialHeader,
    surface: prism_render_material::shaders::material::PrismSurfaceParameters,
    uv: vec2<f32>,
    footprint: f32,
) -> SampledMaterial {
    var sampled = sampled_material_defaults(surface);
    let rows_len = arrayLength(&material_textures);
    if (header.texture_offset > rows_len || header.texture_count > rows_len - header.texture_offset) {
        return sampled;
    }
    for (var i = 0u; i < header.texture_count; i += 1u) {
        let row = material_textures[header.texture_offset + i];
        let texel = sample_texture_slot(row, uv, surface_texture_lod(row, footprint));
        sampled = fold_material_texel(row, texel, surface.normal_scale, sampled);
    }
    return sampled;
}

fn surface_basis_tangent(normal_input: vec3<f32>, angle: f32) -> vec3<f32> {
    let normal = normalize_or(normal_input, vec3<f32>(0.0, 1.0, 0.0));
    let reference = select(
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(1.0, 0.0, 0.0),
        abs(normal.y) > 0.9,
    );
    let basis_tangent = normalize_or(cross(reference, normal), vec3<f32>(1.0, 0.0, 0.0));
    let basis_bitangent = normalize_or(cross(normal, basis_tangent), vec3<f32>(0.0, 0.0, 1.0));
    return normalize_or(basis_tangent * cos(angle) + basis_bitangent * sin(angle), basis_tangent);
}

fn vertex_output(world: vec4<f32>, normal: vec3<f32>, uv0: vec2<f32>) -> VertexOut {
    var output: VertexOut;
    output.clip_position = frame.clip_from_world * world;
    output.world_position = world.xyz;
    output.world_normal = normalize_or(normal, vec3<f32>(0.0, 1.0, 0.0));
    output.uv0 = uv0;
    return output;
}

fn surface_data_frame(input: VertexOut) -> SurfaceDataFrame {
    var result = SurfaceDataFrame(false, 0.0, 0.0);
    let position_dx = dpdx(input.world_position);
    let position_dy = dpdy(input.world_position);
    let uv_dx = dpdx(input.uv0);
    let uv_dy = dpdy(input.uv0);
    let determinant = uv_dx.x * uv_dy.y - uv_dx.y * uv_dy.x;
    let footprint = max(length(uv_dx), length(uv_dy));
    if (!surface_vec2_is_finite(input.uv0)
        || !surface_vec3_is_finite(position_dx)
        || !surface_vec3_is_finite(position_dy)
        || !surface_vec2_is_finite(uv_dx)
        || !surface_vec2_is_finite(uv_dy)
        || !surface_is_finite(determinant)
        || abs(determinant) <= 1.0e-8
        || !surface_is_finite(footprint) || footprint <= 0.0) {
        return result;
    }
    let normal = normalize_or(input.world_normal, vec3<f32>(0.0, 1.0, 0.0));
    let tangent_raw = (position_dx * uv_dy.y - position_dy * uv_dx.y) / determinant;
    let tangent_projected = tangent_raw - normal * dot(normal, tangent_raw);
    let tangent_length_squared = dot(tangent_projected, tangent_projected);
    if (!surface_is_finite(tangent_length_squared)
        || tangent_length_squared <= 1.0e-12) { return result; }
    let tangent = tangent_projected * inverseSqrt(tangent_length_squared);
    let reference = select(
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(1.0, 0.0, 0.0),
        abs(normal.y) > 0.9,
    );
    let basis_tangent = normalize_or(cross(reference, normal), vec3<f32>(1.0, 0.0, 0.0));
    let basis_bitangent = normalize_or(cross(normal, basis_tangent), vec3<f32>(0.0, 0.0, 1.0));
    let angle = atan2(dot(tangent, basis_bitangent), dot(tangent, basis_tangent));
    if (!surface_is_finite(angle)) { return result; }
    result.valid = true;
    result.tangent_angle = angle;
    result.footprint = footprint;
    return result;
}

@vertex
fn vs(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv0: vec2<f32>,
) -> VertexOut {
    return vertex_output(vec4<f32>(position, 1.0), normal, uv0);
}

@vertex
fn vs_transform(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(13) row0: vec4<f32>,
    @location(14) row1: vec4<f32>,
    @location(15) row2: vec4<f32>,
) -> VertexOut {
    let local = vec4<f32>(position, 1.0);
    let world = vec4<f32>(dot(row0, local), dot(row1, local), dot(row2, local), 1.0);
    return vertex_output(world, transform_normal(row0, row1, row2, normal), uv0);
}

fn evaluate_surface(input: VertexOut) -> SurfaceSample {
    // Reject stale handles and malformed ranges before any indexed heap read.
    // Subtraction-based bounds checks cannot overflow at u32::MAX.
    if (material.index >= arrayLength(&material_headers)) { discard; }
    let header = material_headers[material.index];
    if (!prism_material_is_valid(header, material.generation)
        || header.revision != material.revision
        || header.execution_plan_index != material.execution_plan_index
        || (header.lobe_mask & ~PRISM_LOBE_ALL) != 0u) { discard; }
    let count = prism_packed_word_count(header.lobe_mask);
    let heap_words = arrayLength(&material_parameters);
    if (header.parameter_size != count * 4u || header.parameter_offset > heap_words) { discard; }
    if (count > heap_words - header.parameter_offset) { discard; }
    var words: array<u32, PRISM_SURFACE_MAX_WORDS>;
    for (var i = 0u; i < count; i += 1u) {
        words[i] = material_parameters[header.parameter_offset + i];
    }
    let surface = prism_unpack_surface(words, header.lobe_mask);
    let sampled = sample_surface_material(header, surface, input.uv0, surface_data_frame(input).footprint);
    var output: SurfaceSample;
    output.position_x = input.world_position.x;
    output.position_y = input.world_position.y;
    output.position_z = input.world_position.z;
    // Normal alpha is the opaque-coverage bit consumed by composite. Temporal
    // reactivity is carried by the dedicated motion/reactive attachment.
    let surface_data = surface_data_frame(input);
    if (!surface_data.valid) { discard; }
    let geometry_normal = normalize_or(input.world_normal, vec3<f32>(0.0, 1.0, 0.0));
    let tangent = surface_basis_tangent(geometry_normal, surface_data.tangent_angle);
    let bitangent = normalize_or(cross(geometry_normal, tangent), vec3<f32>(0.0, 0.0, 1.0));
    let sampled_normal = normalize_or(
        tangent * sampled.normal_tangent.x
            + bitangent * sampled.normal_tangent.y
            + geometry_normal * sampled.normal_tangent.z,
        geometry_normal,
    );
    let sampled_clearcoat_normal = normalize_or(
        tangent * sampled.clearcoat_normal_tangent.x
            + bitangent * sampled.clearcoat_normal_tangent.y
            + geometry_normal * sampled.clearcoat_normal_tangent.z,
        geometry_normal,
    );
    var output_normal = geometry_normal;
    if (sampled.has_normal_map) {
        output_normal = sampled_normal;
    }
    output.normal = vec4<f32>(output_normal, 1.0);
    output.albedo = vec4<f32>(sampled.base_color, sampled.base_alpha * surface.geometry_opacity);
    output.surface_id = material.index;
    output.surface_data = vec4<f32>(
        input.uv0,
        surface_data.tangent_angle,
        surface_data.footprint,
    );
    output.sampled_material_a = vec4<f32>(
        sampled.emissive,
        sampled.metallic,
    );
    output.sampled_material_b = vec4<f32>(
        sampled.perceptual_roughness,
        sampled.occlusion,
        sampled.clearcoat,
        sampled.offline_normal_variance,
    );
    let clearcoat_oct = oct_encode_normal(sampled_clearcoat_normal);
    output.sampled_material_c = pack_material_c(
        clearcoat_oct,
        sampled.clearcoat_roughness,
        surface_data.valid,
    );
    return output;
}

@fragment
fn fs_geometry(input: VertexOut) -> GeometryGBufferOut {
    let sample = evaluate_surface(input);
    var output: GeometryGBufferOut;
    output.position_x = sample.position_x;
    output.position_y = sample.position_y;
    output.position_z = sample.position_z;
    output.normal = sample.normal;
    output.albedo = sample.albedo;
    return output;
}

@fragment
fn fs_material(input: VertexOut) -> MaterialGBufferOut {
    let sample = evaluate_surface(input);
    var output: MaterialGBufferOut;
    output.surface_id = sample.surface_id;
    output.surface_data = sample.surface_data;
    output.sampled_material_a = sample.sampled_material_a;
    output.sampled_material_b = sample.sampled_material_b;
    output.sampled_material_c = sample.sampled_material_c;
    return output;
}
