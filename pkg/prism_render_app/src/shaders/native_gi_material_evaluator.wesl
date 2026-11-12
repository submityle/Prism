import prism_render_material::shaders::material::{
    PrismMaterialHeader,
    PrismMaterialTexture,
    PrismSurfaceParameters,
};
import prism_render_material::shaders::material_unpack::{
    PRISM_LOBE_ALL,
    PRISM_SURFACE_MAX_WORDS,
    prism_packed_word_count,
    prism_unpack_surface,
};
import prism_render_material::shaders::material_sample::{
    SampledMaterial,
    fold_material_texel,
    material_texture_record_is_valid,
    material_texture_generation_table_matches,
    sample_texture_slot,
    sampled_material_defaults,
};
import prism_render_app::shaders::brdf::{
    DirectLightSample,
    ShadingFrame,
    SurfaceSample,
    principled_direct_channels_with,
    principled_pre,
    stylized_direct,
};
import prism_render_app::shaders::lighting::StylizedParams;
import prism_render_material::shaders::material_virtual_texture::{
    MaterialVirtualTextureConfig,
    material_vt_atlas_uv,
    material_vt_empty_resolution,
    material_vt_geometry_is_valid,
    material_vt_page_request,
    material_vt_resolve,
    material_vt_sample_resolved,
    material_vt_slot_is_valid,
};

const INVALID: u32 = 0xffffffffu;
const HIT_WORDS: u32 = 32u;
const HIT_ATTRIBUTE_WORDS: u32 = 40u;
const MATERIAL_EVALUATOR_ABI_VERSION: u32 = 1u;
const RESPONSE_PROGRAM_KIND_UNLIT: u32 = 2u;
const RESPONSE_PROGRAM_KIND_NPR: u32 = 1u;
const RESPONSE_PROGRAM_KIND_LIT: u32 = 0u;
const RESPONSE_PROGRAM_KIND_CUSTOM: u32 = 3u;
const BUILTIN_UNLIT_PROGRAM_ID: u32 = 2u;
const BUILTIN_NPR_PROGRAM_ID: u32 = 1u;
const BUILTIN_LIT_PROGRAM_ID: u32 = 0u;
const TEXTURE_SEMANTIC_BASE_COLOR: u32 = 0u;
const TEXTURE_SEMANTIC_EMISSIVE: u32 = 1u;
const TEXTURE_SEMANTIC_METALLIC_ROUGHNESS: u32 = 2u;
const TEXTURE_SEMANTIC_NORMAL: u32 = 3u;
const TEXTURE_SEMANTIC_OCCLUSION: u32 = 4u;
const TEXTURE_SEMANTIC_CLEAR_COAT: u32 = 5u;
const TEXTURE_SEMANTIC_CLEAR_COAT_ROUGHNESS: u32 = 6u;
const TEXTURE_SEMANTIC_CLEAR_COAT_NORMAL: u32 = 7u;
const TEXTURE_SEMANTIC_NORMAL_VARIANCE: u32 = 11u;
const NO_VIRTUAL_TEXTURE: u32 = 0xffffffffu;
const PORTABLE_TEXTURE_ATLAS: u32 = 0xfffffffeu;
const EXECUTION_FAMILY_FUSED_BUILTIN: u32 = 0u;
const EXECUTION_FAMILY_SPLIT_RESPONSE: u32 = 1u;
const EXECUTION_FAMILY_COMPOSITION: u32 = 2u;
const MASK_KIND_DYNAMIC_CHANNEL: u32 = 5u;
const BLEND_DOMAIN_FULL_RADIANCE: u32 = 0u;
const BLEND_DOMAIN_DIRECT_ONLY: u32 = 1u;
const BLEND_DOMAIN_CUSTOM_CHANNELS: u32 = 2u;
const RESPONSE_CHANNEL_DIRECT: u32 = 1u;
const RESPONSE_CHANNEL_INDIRECT_DIFFUSE: u32 = 2u;
const RESPONSE_CHANNEL_REFLECTION: u32 = 4u;
const RESPONSE_CHANNEL_TRANSMISSION: u32 = 8u;
const RESPONSE_CHANNEL_EMISSION: u32 = 16u;
const RESPONSE_CHANNEL_RADIANCE: u32 = RESPONSE_CHANNEL_DIRECT
    | RESPONSE_CHANNEL_INDIRECT_DIFFUSE
    | RESPONSE_CHANNEL_REFLECTION
    | RESPONSE_CHANNEL_TRANSMISSION
    | RESPONSE_CHANNEL_EMISSION;
const PRISM_LOBE_EMISSION: u32 = 1u;
const PRISM_LOBE_CLEARCOAT: u32 = 2u;
const PRISM_LOBE_ANISOTROPY: u32 = 4u;
const PRISM_LOBE_SHEEN: u32 = 8u;
const PRISM_LOBE_SUBSURFACE: u32 = 16u;
const PRISM_LOBE_TRANSMISSION: u32 = 32u;
const PRISM_LOBE_THIN_FILM: u32 = 64u;
const PRISM_LOBE_NPR_FACE: u32 = 65536u;
const PRISM_LOBE_GLINT: u32 = 131072u;
const PRISM_LOBE_MEASURED_BRDF: u32 = 262144u;

fn finite(value: f32) -> bool {
    return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

struct NativeGiMaterialEvaluatorConfig {
    abi_version: u32,
    material_epoch_low: u32,
    material_epoch_high: u32,
    execution_table_version: u32,
    vt_generation_low: u32,
    vt_generation_high: u32,
    material_header_count: u32,
    parameter_word_count: u32,
    texture_row_count: u32,
    execution_plan_count: u32,
    execution_program_count: u32,
    texture_slot_count: u32,
    vt_page_table_word_count: u32,
    flags: u32,
    padding: vec2<u32>,
};

struct ExecutionPlanRow {
    left_program: u32,
    right_program: u32,
    execution_family: u32,
    blended_channels: u32,
    mask_kind: u32,
    mask_index: u32,
    blend_domain: u32,
    reserved: u32,
};

struct ExecutionProgramRow {
    response_kind: u32,
    package_index: u32,
    permutation_count_low: u32,
    permutation_count_high: u32,
    program_id: u32,
};

struct NativeGiMaterialEvaluationRecord {
    response_rgb: vec3<f32>,
    valid: u32,
    instance_id: u32,
    geometry_id: u32,
    material_index: u32,
    material_generation: u32,
    material_revision: u32,
    material_epoch_low: u32,
    material_epoch_high: u32,
    execution_plan_index: u32,
    execution_program_index: u32,
    execution_package_index: u32,
    execution_permutation_low: u32,
    execution_permutation_high: u32,
    response_kind: u32,
    execution_family: u32,
    blended_channels: u32,
    response_program_id: u32,
};

// This is the same function-link ABI used by registered response packages.
// Keeping it in the fixed evaluator lets the runtime append only the already
// validated function bodies; the adapter below still owns all resource access.
struct RegisteredSurface {
    base_color: vec3<f32>, base_weight: f32, diffuse_roughness: f32,
    metallic: f32, perceptual_roughness: f32, specular_weight: f32,
    specular_color: vec3<f32>, specular_ior: f32, transmission: f32,
    index_of_refraction: f32, dispersion: f32, thin_walled: f32,
    transmission_color: vec3<f32>, transmission_scatter: vec3<f32>,
    transmission_scatter_anisotropy: f32, transmission_dispersion_abbe_number: f32,
    coat_darkening: f32, scatter_anisotropy: f32, ambient_occlusion: f32,
    emissive: vec3<f32>, clearcoat: f32, clearcoat_roughness: f32,
    coat_ior: f32, coat_color: vec3<f32>, coat_normal: vec3<f32>,
    coat_roughness_anisotropy: f32, coat_tangent: vec3<f32>, sheen: f32,
    fuzz_roughness: f32, fuzz_color: vec3<f32>, subsurface: f32,
    subsurface_color: vec3<f32>, subsurface_radius: f32,
    subsurface_radius_scale: vec3<f32>, thickness: f32, anisotropy: f32,
    anisotropy_rotation: f32, normal_variance: f32, iridescence: f32,
    iridescence_thickness_nm: f32, iridescence_ior: f32, glint_weight: f32,
    glint_roughness: f32, glint_density: f32, glint_bin_half_width: f32,
    glint_state: f32, glint_cell: vec2<f32>, glint_footprint_area: f32,
};
struct RegisteredFrame { normal: vec3<f32>, view: vec3<f32>, tangent: vec3<f32>, bitangent: vec3<f32> };
struct RegisteredGeometry {
    position: vec3<f32>, normal: vec3<f32>, tangent: vec3<f32>, bitangent: vec3<f32>,
    uv: vec2<f32>, flags: u32,
};
struct RegisteredParameters {
    base_color: vec4<f32>, emission_color: vec4<f32>, emission_luminance: f32,
    base_metalness: f32, specular_roughness: f32, specular_weight: f32,
    specular_color: vec4<f32>, specular_ior: f32, thin_walled: f32,
    geometry_opacity: f32, ambient_occlusion: f32, normal_scale: f32,
    alpha_cutoff: f32, base_weight: f32, base_diffuse_roughness: f32,
    transmission_weight: f32, thickness: f32, coat_weight: f32,
    coat_roughness: f32, coat_roughness_anisotropy: f32, coat_ior: f32,
    coat_color: vec4<f32>, anisotropy: f32, anisotropy_rotation: f32,
    fuzz_weight: f32, fuzz_roughness: f32, fuzz_color: vec4<f32>,
    subsurface_weight: f32, subsurface_color: vec4<f32>, subsurface_radius: f32,
    subsurface_radius_scale: vec4<f32>, index_of_refraction: f32,
    transmission_dispersion_scale: f32, transmission_color: vec4<f32>,
    transmission_scatter: vec4<f32>, thin_film_weight: f32,
    thin_film_thickness: f32, thin_film_ior: f32,
    transmission_scatter_anisotropy: f32, transmission_dispersion_abbe_number: f32,
    coat_darkening: f32, subsurface_scatter_anisotropy: f32,
    npr_face_softness: f32, npr_stylized_ramp_softness: f32,
    npr_stylized_specular_scale: f32, npr_stylized_rim_scale: f32,
    glint_weight: f32, glint_roughness: f32, glint_density: f32,
    glint_bin_half_width: f32, measured_weight: f32, measured_phi_d: f32,
    measured_scale: f32,
};
struct RegisteredHeader {
    generation: u32, revision: u32, render_class: u32, render_domain: u32,
    feature_flags: u32, closure_mask: u32, parameter_offset: u32, parameter_size: u32,
    texture_offset: u32, texture_count: u32, sampler_offset: u32, sampler_count: u32,
    is_active: u32, material_epoch_low: u32, material_epoch_high: u32, lobe_mask: u32,
    extensions_used_low: u32, extensions_used_high: u32,
    extensions_required_low: u32, extensions_required_high: u32,
    dynamic_wetness: u32, dynamic_snow: u32, dynamic_damage: u32, dynamic_dissolve: u32,
    execution_plan_index: u32, execution_program_index: u32, execution_package_index: u32,
    execution_permutation_low: u32, execution_permutation_high: u32,
};
struct RegisteredStylizedParams {
    bands: u32, wrap: f32, ramp_softness: f32, shadow_threshold: f32,
    shadow_softness: f32, specular_intensity: f32, specular_threshold: f32,
    specular_softness: f32, specular_color: vec3<f32>, rim_intensity: f32,
    rim_color: vec3<f32>, rim_power: f32,
};
struct RegisteredEnvironment {
    ambient: vec3<f32>, directional_count: u32, sh: array<vec4<f32>, 9>,
    punctual_count: u32, reserved0: u32, exposure: f32, flags: u32,
    stylized: RegisteredStylizedParams,
};
struct RegisteredLight { direction: vec3<f32>, illuminance: vec3<f32>, visibility: f32, valid: f32 };
struct RegisteredArea {
    pos_x: f32, pos_y: f32, pos_z: f32, range: f32,
    color_x: f32, color_y: f32, color_z: f32, intensity: f32,
    axis_u_x: f32, axis_u_y: f32, axis_u_z: f32, half_width: f32,
    axis_v_x: f32, axis_v_y: f32, axis_v_z: f32, half_height: f32,
    shape: u32, two_sided: u32, radius: f32, pad0: u32,
};
struct RegisteredResponse {
    direct: vec3<f32>, indirect_diffuse: vec3<f32>, reflection: vec3<f32>,
    transmission: vec3<f32>, emission: vec3<f32>, alpha: f32,
};
struct RegisteredSurfaceData { sample: RegisteredSurface, frame: RegisteredFrame, geometry: RegisteredGeometry, parameters: RegisteredParameters };
struct RegisteredLightingContext {
    environment: RegisteredEnvironment, frame: RegisteredFrame, world_position: vec3<f32>,
    directional_count: u32, punctual_count: u32, area_count: u32, clustered: u32,
    cluster_offset: u32, indirect_total: vec3<f32>, diffuse_irradiance: vec3<f32>,
    environment_radiance: vec3<f32>, specular_weight: vec3<f32>,
};
struct RegisteredView {
    position: vec3<f32>, pixel: vec2<u32>, size: vec2<u32>, frame_index: u32,
    clip_from_world: mat4x4<f32>, previous_clip_from_world: mat4x4<f32>,
};
struct RegisteredCustomData { header: RegisteredHeader, permutation_low: u32, permutation_high: u32 };

fn prism_native_registered_evaluate(
    surface: RegisteredSurfaceData,
    lighting: RegisteredLightingContext,
    view: RegisteredView,
    custom: RegisteredCustomData,
    program_id: u32,
    permutation_low: u32,
    permutation_high: u32,
) -> RegisteredResponse {
    // Replaced by the exact validated link section when a registered program
    // is present. The static body is deliberately invalid, never a fallback.
    var invalid: RegisteredResponse;
    invalid.direct = vec3<f32>(bitcast<f32>(0x7fc00000u), 0.0, 0.0);
    invalid.indirect_diffuse = vec3<f32>(0.0);
    invalid.reflection = vec3<f32>(0.0);
    invalid.transmission = vec3<f32>(0.0);
    invalid.emission = vec3<f32>(0.0);
    invalid.alpha = bitcast<f32>(0x7fc00000u);
    return invalid;
}

@group(0) @binding(0) var<storage, read> trace_hits: array<u32>;
@group(0) @binding(1) var<uniform> evaluator_config: NativeGiMaterialEvaluatorConfig;
@group(0) @binding(2) var<storage, read_write> evaluations: array<NativeGiMaterialEvaluationRecord>;

@group(1) @binding(0) var<storage, read> material_headers: array<PrismMaterialHeader>;
@group(1) @binding(1) var<storage, read> material_parameters: array<u32>;
@group(1) @binding(2) var<storage, read> material_texture_rows: array<PrismMaterialTexture>;
// Plans and programs are packed consecutively into one storage buffer. This
// keeps the evaluator below Metal's combined storage-buffer stage limit while
// preserving both validated tables as real GPU data.
@group(2) @binding(0) var<storage, read> execution_table: array<u32>;
// Binding 5 is owned by the imported VT resolver. Its runtime array is
// clipped against the published entry_count inside material_vt_exact_slot.
@if(!MATERIAL_BINDLESS) @group(1) @binding(6) var<storage, read> material_vt_config: MaterialVirtualTextureConfig;
@if(!MATERIAL_BINDLESS) @group(1) @binding(7) var material_vt_atlas: texture_2d_array<f32>;
@if(!MATERIAL_BINDLESS) @group(1) @binding(8) var material_vt_sampler: sampler;

fn invalid_evaluation() -> NativeGiMaterialEvaluationRecord {
    return NativeGiMaterialEvaluationRecord(
        vec3<f32>(0.0),
        0u,
        INVALID,
        INVALID,
        INVALID,
        INVALID,
        INVALID,
        INVALID,
        INVALID,
        INVALID,
        INVALID,
        INVALID,
        INVALID,
        INVALID,
        INVALID,
        INVALID,
        INVALID,
        INVALID,
    );
}

fn execution_plan_at(index: u32) -> ExecutionPlanRow {
    let base = index * 8u;
    return ExecutionPlanRow(
        execution_table[base], execution_table[base + 1u],
        execution_table[base + 2u], execution_table[base + 3u],
        execution_table[base + 4u], execution_table[base + 5u],
        execution_table[base + 6u], execution_table[base + 7u],
    );
}

fn execution_program_at(index: u32) -> ExecutionProgramRow {
    let base = evaluator_config.execution_plan_count * 8u + index * 5u;
    return ExecutionProgramRow(
        execution_table[base], execution_table[base + 1u],
        execution_table[base + 2u], execution_table[base + 3u],
        execution_table[base + 4u],
    );
}

fn native_execution_table_is_valid() -> bool {
    let table_words = arrayLength(&execution_table);
    let plan_count = evaluator_config.execution_plan_count;
    let program_count = evaluator_config.execution_program_count;
    if plan_count > table_words / 8u {
        return false;
    }
    let plan_words = plan_count * 8u;
    let remaining_words = table_words - plan_words;
    return program_count <= remaining_words / 5u;
}

struct TextureSample {
    valid: bool,
    value: vec4<f32>,
}

struct NativeResponse {
    direct: vec3<f32>,
    indirect_diffuse: vec3<f32>,
    reflection: vec3<f32>,
    transmission: vec3<f32>,
    emission: vec3<f32>,
    alpha: f32,
    valid: bool,
}

fn invalid_native_response() -> NativeResponse {
    return NativeResponse(
        vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0),
        vec3<f32>(0.0), vec3<f32>(0.0), 0.0, false,
    );
}

@if(!MATERIAL_BINDLESS) fn sample_vt_row(row: PrismMaterialTexture, uv: vec2<f32>) -> TextureSample {
    var out: TextureSample;
    out.valid = false;
    out.value = vec4<f32>(0.0);
    if row.generation == 0u
        || evaluator_config.vt_page_table_word_count == 0u
        || evaluator_config.vt_page_table_word_count > material_vt_config.entry_count * 4u
        || !(evaluator_config.vt_generation_low != 0u
            || evaluator_config.vt_generation_high != 0u)
        || material_vt_config.enabled == 0u
        || !material_vt_geometry_is_valid(material_vt_config)
        || row.virtual_texture_width == 0u
        || row.virtual_texture_height == 0u
        || row.virtual_texture_page_size != material_vt_config.page_size {
        return out;
    }

    var resolution = material_vt_empty_resolution(true);
    if row.virtual_texture_id == PORTABLE_TEXTURE_ATLAS {
        if !material_vt_slot_is_valid(row.index, material_vt_config)
            || row.virtual_texture_width > row.virtual_texture_page_size
            || row.virtual_texture_height > row.virtual_texture_page_size {
            return out;
        }
        let slots_per_layer = material_vt_config.tiles_per_row
            * material_vt_config.tiles_per_row;
        let atlas_dimensions = textureDimensions(material_vt_atlas);
        if atlas_dimensions.x != material_vt_config.tile_extent_px * material_vt_config.tiles_per_row
            || atlas_dimensions.y != material_vt_config.tile_extent_px * material_vt_config.tiles_per_row
            || textureNumLayers(material_vt_atlas) < material_vt_config.atlas_layers {
            return out;
        }
        let source_uv = clamp(
            uv * vec2<f32>(
                f32(row.virtual_texture_width) / f32(row.virtual_texture_page_size),
                f32(row.virtual_texture_height) / f32(row.virtual_texture_page_size),
            ),
            vec2<f32>(0.0),
            vec2<f32>(1.0),
        );
        resolution.hit = true;
        resolution.slot = row.index;
        resolution.atlas_layer = row.index / max(slots_per_layer, 1u);
        resolution.page_local_uv = source_uv;
        resolution.atlas_uv = material_vt_atlas_uv(
            source_uv,
            row.index,
            material_vt_config.tile_extent_px,
            material_vt_config.tiles_per_row,
            material_vt_config.page_size,
            material_vt_config.gutter_px,
        );
    } else if row.virtual_texture_id != NO_VIRTUAL_TEXTURE {
        let request = material_vt_page_request(
            row.virtual_texture_id,
            row.virtual_texture_width,
            row.virtual_texture_height,
            row.virtual_texture_page_size,
            uv,
            0.0,
        );
        if !request.valid {
            return out;
        }
        resolution = material_vt_resolve(
            material_vt_config,
            row.virtual_texture_id,
            0u,
            request.mip,
            request.page_x,
            request.page_y,
            request.local_uv,
        );
    } else {
        return out;
    }
    if !resolution.hit || !material_vt_slot_is_valid(resolution.slot, material_vt_config) {
        return out;
    }
    let atlas_dimensions = textureDimensions(material_vt_atlas);
    if resolution.atlas_layer >= textureNumLayers(material_vt_atlas)
        || atlas_dimensions.x == 0u
        || atlas_dimensions.y == 0u {
        return out;
    }
    out.value = material_vt_sample_resolved(material_vt_atlas, material_vt_sampler, resolution);
    out.valid = finite(out.value.x)
        && finite(out.value.y)
        && finite(out.value.z)
        && finite(out.value.w);
    return out;
}

@if(!MATERIAL_BINDLESS) fn sample_texture_row(row: PrismMaterialTexture, uv: vec2<f32>) -> TextureSample {
    return sample_vt_row(row, uv);
}

@if(MATERIAL_BINDLESS) fn sample_texture_row(row: PrismMaterialTexture, uv: vec2<f32>) -> TextureSample {
    var out: TextureSample;
    out.valid = native_texture_row_binding_is_valid(row);
    out.value = vec4<f32>(0.0);
    if (!out.valid) { return out; }
    // The shared bindless sampler deliberately falls back to the authored
    // image when a VT request misses. That policy is valid for the main
    // surface resolve, but it would hide a stale/mismatched Native GI row.
    // Native GI therefore admits only rows whose source contract is direct;
    // a VT row is evaluated by the strict non-bindless path below instead of
    // silently changing semantic sources.
    if row.virtual_texture_id != NO_VIRTUAL_TEXTURE { return out; }
    out.value = sample_texture_slot(row, uv, 0.0);
    out.valid = finite(out.value.x)
        && finite(out.value.y)
        && finite(out.value.z)
        && finite(out.value.w);
    return out;
}

fn native_texture_binding_contract_is_valid() -> bool {
    @if(MATERIAL_BINDLESS) {
        // Require the host to publish a dense generation table covering the
        // advertised heap before any row can be consumed.
        if !material_texture_generation_table_matches(evaluator_config.texture_slot_count) {
            return false;
        }
    }
    @if(!MATERIAL_BINDLESS) {
        // Non-bindless devices have no authored-image arrays. Every authored
        // row must therefore resolve through the published VT resources.
        if evaluator_config.texture_slot_count != 0u {
            return false;
        }
    }
    return true;
}

fn native_texture_row_binding_is_valid(row: PrismMaterialTexture) -> bool {
    if row.generation == 0u || row.index == INVALID || row.sampler_index == INVALID {
        return false;
    }
    @if(MATERIAL_BINDLESS) {
        if row.index >= evaluator_config.texture_slot_count
            || row.sampler_index >= evaluator_config.texture_slot_count
            || !material_texture_record_is_valid(row) {
            return false;
        }
    }
    @if(!MATERIAL_BINDLESS) {
        // The portable evaluator has no authored-image heap. A non-VT row or
        // incomplete VT metadata would otherwise be folded through an
        // identity semantic fallback by the shared sampler.
        if row.virtual_texture_id == NO_VIRTUAL_TEXTURE
            || row.virtual_texture_width == 0u
            || row.virtual_texture_height == 0u
            || row.virtual_texture_page_size == 0u {
            return false;
        }
    }
    return true;
}

fn native_registered_light_count(lighting: RegisteredLightingContext) -> u32 {
    return lighting.directional_count + lighting.punctual_count;
}

fn registered_light(lighting: RegisteredLightingContext, index: u32) -> RegisteredLight {
    var invalid: RegisteredLight;
    invalid.direction = vec3<f32>(0.0, 1.0, 0.0);
    invalid.illuminance = vec3<f32>(0.0);
    invalid.visibility = 0.0;
    invalid.valid = 0.0;
    if index >= native_registered_light_count(lighting) { return invalid; }
    return RegisteredLight(
        native_normalize_or(lighting.frame.normal, vec3<f32>(0.0, 1.0, 0.0)),
        vec3<f32>(1.0),
        1.0,
        1.0,
    );
}

fn registered_area(lighting: RegisteredLightingContext, index: u32) -> RegisteredArea {
    var invalid: RegisteredArea;
    invalid.shape = 0xffffffffu;
    invalid.two_sided = 0u;
    invalid.pad0 = 0u;
    return invalid;
}

fn registered_custom_word(custom: RegisteredCustomData, index: u32) -> u32 {
    if index >= custom.header.parameter_size / 4u
        || custom.header.parameter_offset >= arrayLength(&material_parameters)
        || index >= arrayLength(&material_parameters) - custom.header.parameter_offset {
        return 0u;
    }
    return material_parameters[custom.header.parameter_offset + index];
}

fn native_registered_surface(
    surface: PrismSurfaceParameters,
    sampled: SampledMaterial,
    frame: ShadingFrame,
) -> RegisteredSurface {
    return RegisteredSurface(
        surface.base_color.xyz, surface.base_weight, surface.base_diffuse_roughness,
        surface.base_metalness, surface.specular_roughness, surface.specular_weight,
        surface.specular_color.xyz, surface.specular_ior, surface.transmission_weight,
        surface.index_of_refraction, surface.transmission_dispersion_scale, surface.thin_walled,
        surface.transmission_color.xyz, surface.transmission_scatter.xyz,
        surface.transmission_scatter_anisotropy, surface.transmission_dispersion_abbe_number,
        surface.coat_darkening, surface.subsurface_scatter_anisotropy, surface.ambient_occlusion,
        surface.emission_color.xyz * surface.emission_luminance, surface.coat_weight,
        surface.coat_roughness, surface.coat_ior, surface.coat_color.xyz,
        select(
            vec3<f32>(0.0),
            native_apply_tangent_normal(sampled.clearcoat_normal_tangent, frame, frame.normal),
            sampled.has_clearcoat_normal,
        ),
        surface.coat_roughness_anisotropy, frame.tangent,
        surface.fuzz_weight, surface.fuzz_roughness, surface.fuzz_color.xyz,
        surface.subsurface_weight, surface.subsurface_color.xyz, surface.subsurface_radius,
        surface.subsurface_radius_scale.xyz, surface.thickness, surface.anisotropy,
        surface.anisotropy_rotation, sampled.offline_normal_variance, surface.thin_film_weight,
        surface.thin_film_thickness * 1000.0, surface.thin_film_ior, surface.glint_weight,
        surface.glint_roughness, surface.glint_density, surface.glint_bin_half_width,
        -1.0, vec2<f32>(0.0), 0.0,
    );
}

fn native_folded_surface(
    surface: PrismSurfaceParameters,
    sampled: SampledMaterial,
) -> PrismSurfaceParameters {
    var folded = surface;
    folded.base_color = vec4<f32>(sampled.base_color, sampled.base_alpha);
    folded.emission_color = vec4<f32>(sampled.emissive, surface.emission_color.w);
    folded.base_metalness = sampled.metallic;
    folded.specular_roughness = sampled.perceptual_roughness;
    folded.ambient_occlusion = sampled.occlusion;
    folded.coat_weight = sampled.clearcoat;
    folded.coat_roughness = sampled.clearcoat_roughness;
    return folded;
}

fn native_registered_parameters(surface: PrismSurfaceParameters) -> RegisteredParameters {
    return RegisteredParameters(
        surface.base_color, surface.emission_color, surface.emission_luminance,
        surface.base_metalness, surface.specular_roughness, surface.specular_weight,
        surface.specular_color, surface.specular_ior, surface.thin_walled,
        surface.geometry_opacity, surface.ambient_occlusion, surface.normal_scale,
        surface.alpha_cutoff, surface.base_weight, surface.base_diffuse_roughness,
        surface.transmission_weight, surface.thickness, surface.coat_weight,
        surface.coat_roughness, surface.coat_roughness_anisotropy, surface.coat_ior,
        surface.coat_color, surface.anisotropy, surface.anisotropy_rotation,
        surface.fuzz_weight, surface.fuzz_roughness, surface.fuzz_color,
        surface.subsurface_weight, surface.subsurface_color, surface.subsurface_radius,
        surface.subsurface_radius_scale, surface.index_of_refraction,
        surface.transmission_dispersion_scale, surface.transmission_color,
        surface.transmission_scatter, surface.thin_film_weight, surface.thin_film_thickness,
        surface.thin_film_ior, surface.transmission_scatter_anisotropy,
        surface.transmission_dispersion_abbe_number, surface.coat_darkening,
        surface.subsurface_scatter_anisotropy, surface.npr_face_softness,
        surface.npr_stylized_ramp_softness, surface.npr_stylized_specular_scale,
        surface.npr_stylized_rim_scale, surface.glint_weight, surface.glint_roughness,
        surface.glint_density, surface.glint_bin_half_width, surface.measured_weight,
        surface.measured_phi_d, surface.measured_scale,
    );
}

fn native_registered_header(header: PrismMaterialHeader) -> RegisteredHeader {
    return RegisteredHeader(
        header.generation, header.revision, header.render_class, header.render_domain,
        header.feature_flags, header.closure_mask, header.parameter_offset, header.parameter_size,
        header.texture_offset, header.texture_count, header.sampler_offset, header.sampler_count,
        header.is_active, header.material_epoch_low, header.material_epoch_high, header.lobe_mask,
        header.extensions_used_low, header.extensions_used_high,
        header.extensions_required_low, header.extensions_required_high,
        header.dynamic_wetness, header.dynamic_snow, header.dynamic_damage, header.dynamic_dissolve,
        header.execution_plan_index, header.execution_program_index,
        header.execution_package_index, header.execution_permutation_low,
        header.execution_permutation_high,
    );
}

fn native_registered_response(
    surface: PrismSurfaceParameters,
    sampled: SampledMaterial,
    frame_input: ShadingFrame,
    header: PrismMaterialHeader,
    uv: vec2<f32>,
    program_id: u32,
    permutation_low: u32,
    permutation_high: u32,
) -> NativeResponse {
    var out = invalid_native_response();
    let folded = native_folded_surface(surface, sampled);
    let frame = RegisteredFrame(
        frame_input.normal, frame_input.view, frame_input.tangent, frame_input.bitangent,
    );
    let geometry = RegisteredGeometry(
        vec3<f32>(0.0), frame_input.normal, frame_input.tangent,
        frame_input.bitangent, uv, 0u,
    );
    let environment = RegisteredEnvironment(
        vec3<f32>(0.0), 1u,
        array<vec4<f32>, 9>(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0)),
        0u, 0u, 1.0, 0u,
        RegisteredStylizedParams(1u, 0.0, 0.0, 0.5, 0.1, 0.0, 0.5, 0.1,
            vec3<f32>(1.0), 0.0, vec3<f32>(1.0), 1.0),
    );
    let lighting = RegisteredLightingContext(
        environment, frame, vec3<f32>(0.0), 1u, 0u, 0u, 0u, 0u,
        vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(1.0),
    );
    let view = RegisteredView(
        frame_input.view, vec2<u32>(0u), vec2<u32>(0u), 0u,
        mat4x4<f32>(vec4<f32>(1.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 1.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 1.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 1.0)),
        mat4x4<f32>(vec4<f32>(1.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 1.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 1.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 1.0)),
    );
    let custom = RegisteredCustomData(native_registered_header(header), permutation_low, permutation_high);
    let response = prism_native_registered_evaluate(
        RegisteredSurfaceData(
            native_registered_surface(folded, sampled, frame_input), frame, geometry,
            native_registered_parameters(folded),
        ),
        lighting, view, custom, program_id, permutation_low, permutation_high,
    );
    let value = response.direct + response.indirect_diffuse + response.reflection
        + response.transmission + response.emission;
    out.direct = response.direct;
    out.indirect_diffuse = response.indirect_diffuse;
    out.reflection = response.reflection;
    out.transmission = response.transmission;
    out.emission = response.emission;
    out.alpha = response.alpha;
    out.valid = finite(value.x) && finite(value.y) && finite(value.z)
        && finite(response.alpha) && response.alpha >= 0.0 && response.alpha <= 1.0
        && all(value >= vec3<f32>(0.0));
    return out;
}

fn native_normalize_or(value: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let length_squared = dot(value, value);
    if finite(length_squared) && length_squared > 1.0e-12 {
        return value * inverseSqrt(length_squared);
    }
    return fallback;
}

fn finite3(value: vec3<f32>) -> bool {
    return finite(value.x) && finite(value.y) && finite(value.z);
}

fn native_texture_semantic_is_supported(semantic: u32) -> bool {
    return semantic == TEXTURE_SEMANTIC_BASE_COLOR
        || semantic == TEXTURE_SEMANTIC_EMISSIVE
        || semantic == TEXTURE_SEMANTIC_METALLIC_ROUGHNESS
        || semantic == TEXTURE_SEMANTIC_NORMAL
        || semantic == TEXTURE_SEMANTIC_OCCLUSION
        || semantic == TEXTURE_SEMANTIC_CLEAR_COAT
        || semantic == TEXTURE_SEMANTIC_CLEAR_COAT_ROUGHNESS
        || semantic == TEXTURE_SEMANTIC_CLEAR_COAT_NORMAL
        || semantic == TEXTURE_SEMANTIC_NORMAL_VARIANCE;
}

fn native_tangent_frame(
    normal: vec3<f32>,
    tangent_input: vec4<f32>,
) -> ShadingFrame {
    let n = native_normalize_or(normal, vec3<f32>(0.0, 1.0, 0.0));
    let projected = tangent_input.xyz - n * dot(n, tangent_input.xyz);
    let t = native_normalize_or(projected, vec3<f32>(1.0, 0.0, 0.0));
    let handedness = select(1.0, -1.0, tangent_input.w < 0.0);
    let b = native_normalize_or(cross(n, t) * handedness, vec3<f32>(0.0, 0.0, 1.0));
    return ShadingFrame(n, vec3<f32>(0.0), t, b);
}

fn native_apply_tangent_normal(
    tangent_normal: vec3<f32>,
    frame: ShadingFrame,
    fallback: vec3<f32>,
) -> vec3<f32> {
    return native_normalize_or(
        frame.tangent * tangent_normal.x
            + frame.bitangent * tangent_normal.y
            + frame.normal * tangent_normal.z,
        fallback,
    );
}

fn native_surface_sample(
    parameters: PrismSurfaceParameters,
    sampled: SampledMaterial,
    frame: ShadingFrame,
) -> SurfaceSample {
    var surface: SurfaceSample;
    surface.base_color = sampled.base_color;
    surface.base_weight = parameters.base_weight;
    surface.diffuse_roughness = parameters.base_diffuse_roughness;
    surface.metallic = sampled.metallic;
    surface.perceptual_roughness = sampled.perceptual_roughness;
    surface.specular_weight = parameters.specular_weight;
    surface.specular_color = parameters.specular_color.xyz;
    surface.specular_ior = parameters.specular_ior;
    surface.transmission = parameters.transmission_weight;
    surface.index_of_refraction = parameters.index_of_refraction;
    surface.dispersion = parameters.transmission_dispersion_scale;
    surface.thin_walled = parameters.thin_walled;
    surface.transmission_color = parameters.transmission_color.xyz;
    surface.transmission_scatter = parameters.transmission_scatter.xyz;
    surface.transmission_scatter_anisotropy = parameters.transmission_scatter_anisotropy;
    surface.transmission_dispersion_abbe_number = parameters.transmission_dispersion_abbe_number;
    surface.coat_darkening = parameters.coat_darkening;
    surface.scatter_anisotropy = parameters.subsurface_scatter_anisotropy;
    surface.ambient_occlusion = sampled.occlusion;
    surface.emissive = sampled.emissive * parameters.emission_luminance;
    surface.clearcoat = sampled.clearcoat;
    surface.clearcoat_roughness = sampled.clearcoat_roughness;
    surface.coat_ior = parameters.coat_ior;
    surface.coat_color = parameters.coat_color.xyz;
    surface.coat_normal = select(
        vec3<f32>(0.0),
        native_apply_tangent_normal(sampled.clearcoat_normal_tangent, frame, frame.normal),
        sampled.has_clearcoat_normal,
    );
    surface.coat_roughness_anisotropy = parameters.coat_roughness_anisotropy;
    surface.coat_tangent = vec3<f32>(0.0);
    surface.sheen = parameters.fuzz_weight;
    surface.fuzz_roughness = parameters.fuzz_roughness;
    surface.fuzz_color = parameters.fuzz_color.xyz;
    surface.subsurface = parameters.subsurface_weight;
    surface.subsurface_color = parameters.subsurface_color.xyz;
    surface.subsurface_radius = parameters.subsurface_radius;
    surface.subsurface_radius_scale = parameters.subsurface_radius_scale.xyz;
    surface.thickness = parameters.thickness;
    surface.anisotropy = parameters.anisotropy;
    surface.anisotropy_rotation = parameters.anisotropy_rotation;
    surface.normal_variance = sampled.offline_normal_variance;
    surface.iridescence = parameters.thin_film_weight;
    surface.iridescence_thickness_nm = parameters.thin_film_thickness * 1000.0;
    surface.iridescence_ior = parameters.thin_film_ior;
    surface.glint_weight = parameters.glint_weight;
    surface.glint_roughness = parameters.glint_roughness;
    surface.glint_density = parameters.glint_density;
    surface.glint_bin_half_width = parameters.glint_bin_half_width;
    surface.glint_state = -1.0;
    surface.glint_cell = vec2<f32>(0.0);
    surface.glint_footprint_area = 0.0;
    return surface;
}

fn native_lobe_mask_is_supported(response_kind: u32, lobe_mask: u32) -> bool {
    var supported = 0u;
    if response_kind == RESPONSE_PROGRAM_KIND_UNLIT {
        supported = PRISM_LOBE_EMISSION;
    } else if response_kind == RESPONSE_PROGRAM_KIND_NPR {
        supported = PRISM_LOBE_EMISSION | PRISM_LOBE_NPR_FACE;
    } else if response_kind == RESPONSE_PROGRAM_KIND_LIT {
        supported = PRISM_LOBE_EMISSION
            | PRISM_LOBE_CLEARCOAT
            | PRISM_LOBE_ANISOTROPY
            | PRISM_LOBE_TRANSMISSION
            | PRISM_LOBE_THIN_FILM
            | PRISM_LOBE_GLINT;
    } else if response_kind == RESPONSE_PROGRAM_KIND_CUSTOM {
        supported = PRISM_LOBE_ALL;
    }
    return (lobe_mask & ~supported) == 0u;
}

fn native_response_channels(response_kind: u32) -> u32 {
    if response_kind == RESPONSE_PROGRAM_KIND_LIT {
        return RESPONSE_CHANNEL_RADIANCE;
    }
    if response_kind == RESPONSE_PROGRAM_KIND_NPR
        || response_kind == RESPONSE_PROGRAM_KIND_UNLIT {
        return RESPONSE_CHANNEL_DIRECT | RESPONSE_CHANNEL_EMISSION;
    }
    if response_kind == RESPONSE_PROGRAM_KIND_CUSTOM {
        return RESPONSE_CHANNEL_RADIANCE;
    }
    return 0u;
}

fn native_channels_are_supported(response_kind: u32, channels: u32) -> bool {
    return channels != 0u
        && (channels & ~RESPONSE_CHANNEL_RADIANCE) == 0u
        && (channels & ~native_response_channels(response_kind)) == 0u;
}

fn native_builtin_response(
    parameters: PrismSurfaceParameters,
    surface: SurfaceSample,
    alpha: f32,
    frame: ShadingFrame,
    response_kind: u32,
) -> NativeResponse {
    var out = invalid_native_response();
    if response_kind == RESPONSE_PROGRAM_KIND_UNLIT {
        out.direct = surface.base_color;
        out.emission = surface.emissive;
        out.alpha = alpha;
        out.valid = finite3(surface.base_color + surface.emissive)
            && out.alpha >= 0.0 && out.alpha <= 1.0;
        return out;
    }
    if response_kind == RESPONSE_PROGRAM_KIND_NPR {
        var stylized = StylizedParams(
            4u, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.05,
            vec3<f32>(1.0), 0.0, vec3<f32>(1.0), 4.0,
        );
        stylized.ramp_softness = stylized.ramp_softness
            * clamp(parameters.npr_stylized_ramp_softness, 0.0, 1.0);
        stylized.specular_intensity = stylized.specular_intensity
            * max(parameters.npr_stylized_specular_scale, 0.0);
        stylized.rim_intensity = stylized.rim_intensity
            * max(parameters.npr_stylized_rim_scale, 0.0);
        var direct_surface = surface;
        direct_surface.emissive = vec3<f32>(0.0);
        let light = DirectLightSample(frame.normal, vec3<f32>(1.0), 1.0);
        out.direct = stylized_direct(direct_surface, frame, light, stylized);
        out.emission = surface.emissive;
        out.alpha = alpha;
        let response = out.direct + out.emission;
        out.valid = finite3(response) && all(response >= vec3<f32>(0.0))
            && finite(out.alpha) && out.alpha >= 0.0 && out.alpha <= 1.0;
        return out;
    }
    if response_kind != RESPONSE_PROGRAM_KIND_LIT {
        return out;
    }
    // The evaluator ABI has no incident-light buffer. Its canonical reference
    // convention is one unit-white light aligned to the sampled hit normal.
    let light = DirectLightSample(frame.normal, vec3<f32>(1.0), 1.0);
    let channels = principled_direct_channels_with(
        surface,
        frame,
        light,
        principled_pre(surface, frame),
    );
    out.direct = channels.direct;
    out.transmission = channels.transmission;
    out.emission = surface.emissive;
    out.alpha = alpha;
    let response = out.direct + out.transmission + out.emission;
    out.valid = finite3(response) && all(response >= vec3<f32>(0.0))
        && finite(out.alpha) && out.alpha >= 0.0 && out.alpha <= 1.0;
    return out;
}

fn native_mask_value(header: PrismMaterialHeader, index: u32) -> TextureSample {
    var out: TextureSample;
    out.valid = index <= 3u;
    out.value = vec4<f32>(0.0);
    if !out.valid { return out; }
    let raw = select(
        select(select(header.dynamic_wetness, header.dynamic_snow, index == 1u), header.dynamic_damage, index == 2u),
        header.dynamic_dissolve, index == 3u,
    );
    let value = bitcast<f32>(raw);
    out.valid = finite(value) && value >= 0.0 && value <= 1.0;
    out.value = vec4<f32>(clamp(value, 0.0, 1.0));
    return out;
}

fn native_execution_program_is_valid(
    program: ExecutionProgramRow,
    package_index: u32,
    permutation_low: u32,
    permutation_high: u32,
) -> bool {
    if program.response_kind > RESPONSE_PROGRAM_KIND_CUSTOM
        || (program.response_kind == RESPONSE_PROGRAM_KIND_LIT
            && program.program_id != BUILTIN_LIT_PROGRAM_ID)
        || (program.response_kind == RESPONSE_PROGRAM_KIND_UNLIT
            && program.program_id != BUILTIN_UNLIT_PROGRAM_ID)
        || program.package_index != package_index
        || permutation_high > program.permutation_count_high
        || (permutation_high == program.permutation_count_high
            && permutation_low >= program.permutation_count_low) {
        return false;
    }
    return true;
}

fn native_execution_arm_is_valid(
    program: ExecutionProgramRow,
    permutation_low: u32,
    permutation_high: u32,
) -> bool {
    if program.package_index == INVALID {
        return false;
    }
    return native_execution_program_is_valid(
        program, program.package_index, permutation_low, permutation_high,
    );
}

fn native_arm_response(
    parameters: PrismSurfaceParameters,
    sampled: SampledMaterial,
    sampled_surface: SurfaceSample,
    header: PrismMaterialHeader,
    program: ExecutionProgramRow,
    frame: ShadingFrame,
    uv: vec2<f32>,
    alpha: f32,
    permutation_low: u32,
    permutation_high: u32,
) -> NativeResponse {
    if program.response_kind == RESPONSE_PROGRAM_KIND_CUSTOM {
        return native_registered_response(
            parameters, sampled, frame, header, uv, program.program_id,
            permutation_low, permutation_high,
        );
    }
    if program.response_kind == RESPONSE_PROGRAM_KIND_LIT && program.program_id != BUILTIN_LIT_PROGRAM_ID {
        return invalid_native_response();
    }
    if program.response_kind == RESPONSE_PROGRAM_KIND_NPR && program.program_id != BUILTIN_NPR_PROGRAM_ID {
        return invalid_native_response();
    }
    if program.response_kind == RESPONSE_PROGRAM_KIND_UNLIT && program.program_id != BUILTIN_UNLIT_PROGRAM_ID {
        return invalid_native_response();
    }
    return native_builtin_response(
        parameters, sampled_surface, alpha, frame, program.response_kind,
    );
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ray_index = global_id.x;
    if (ray_index >= arrayLength(&evaluations)) { return; }
    var result = invalid_evaluation();

    let hit_base = ray_index * HIT_WORDS;
    if (ray_index >= arrayLength(&trace_hits) / HIT_WORDS) {
        evaluations[ray_index] = result;
        return;
    }
    if (trace_hits[hit_base] == 0u) {
        evaluations[ray_index] = result;
        return;
    }
    let triangle_index = trace_hits[hit_base + 5u];
    if (trace_hits[hit_base] != 1u || triangle_index == INVALID) {
        evaluations[ray_index] = result;
        return;
    }

    if (evaluator_config.abi_version != MATERIAL_EVALUATOR_ABI_VERSION
        || evaluator_config.execution_table_version == 0u
        || evaluator_config.material_header_count == 0u
        || evaluator_config.parameter_word_count == 0u
        || evaluator_config.execution_plan_count == 0u
        || evaluator_config.execution_program_count == 0u
        || evaluator_config.material_header_count > arrayLength(&material_headers)
        || evaluator_config.parameter_word_count > arrayLength(&material_parameters)
        || evaluator_config.texture_row_count > arrayLength(&material_texture_rows)
        || (evaluator_config.vt_page_table_word_count != 0u)
            != (evaluator_config.vt_generation_low != 0u
                || evaluator_config.vt_generation_high != 0u)
        || (evaluator_config.vt_page_table_word_count != 0u
            && (evaluator_config.vt_page_table_word_count < 4u
                || (evaluator_config.vt_page_table_word_count & 3u) != 0u))
        || evaluator_config.flags != 0u
        || any(evaluator_config.padding != vec2<u32>(0u))) {
        evaluations[ray_index] = result;
        return;
    }
    if !native_texture_binding_contract_is_valid() {
        evaluations[ray_index] = result;
        return;
    }
    if !native_execution_table_is_valid() {
        evaluations[ray_index] = result;
        return;
    }

    let instance_id = trace_hits[hit_base + 6u];
    let geometry_id = trace_hits[hit_base + 7u];
    let material_index = trace_hits[hit_base + 8u];
    let material_generation = trace_hits[hit_base + 9u];
    let material_revision = trace_hits[hit_base + 10u];
    let material_epoch_low = trace_hits[hit_base + 11u];
    let material_epoch_high = trace_hits[hit_base + 12u];
    let plan_index = trace_hits[hit_base + 13u];
    let program_index = trace_hits[hit_base + 14u];
    let package_index = trace_hits[hit_base + 15u];
    let permutation_low = trace_hits[hit_base + 16u];
    let permutation_high = trace_hits[hit_base + 17u];
    if (instance_id == INVALID || geometry_id == INVALID || material_index == INVALID
        || material_generation == INVALID
        || material_revision == INVALID
        || (material_epoch_low == INVALID && material_epoch_high == INVALID)
        || plan_index == INVALID
        || program_index == INVALID
        || package_index == INVALID
        || permutation_low == INVALID
        || permutation_high == INVALID
        || material_index >= evaluator_config.material_header_count
        || material_index >= arrayLength(&material_headers)
        || plan_index >= evaluator_config.execution_plan_count
        || program_index >= evaluator_config.execution_program_count
        || plan_index >= evaluator_config.execution_plan_count) {
        evaluations[ray_index] = result;
        return;
    }

    let header = material_headers[material_index];
    if (header.is_active == 0u
        || header.generation != material_generation
        || header.revision != material_revision
        || header.material_epoch_low != material_epoch_low
        || header.material_epoch_high != material_epoch_high
        || material_epoch_low != evaluator_config.material_epoch_low
        || material_epoch_high != evaluator_config.material_epoch_high
        || header.execution_plan_index != plan_index
        || header.execution_program_index != program_index
        || header.execution_package_index != package_index
        || header.execution_permutation_low != permutation_low
        || header.execution_permutation_high != permutation_high
        || (header.lobe_mask & ~PRISM_LOBE_ALL) != 0u
        || header.parameter_size == 0u
        || (header.parameter_size & 3u) != 0u
        || header.texture_offset > evaluator_config.texture_row_count
        || header.texture_count > evaluator_config.texture_row_count - header.texture_offset
        || header.texture_offset + header.texture_count > arrayLength(&material_texture_rows)
        || header.sampler_offset != header.texture_offset
        || header.sampler_count != header.texture_count
        || header.sampler_offset > evaluator_config.texture_row_count
        || header.sampler_count > evaluator_config.texture_row_count - header.sampler_offset
        || header.sampler_offset + header.sampler_count > arrayLength(&material_texture_rows)) {
        evaluations[ray_index] = result;
        return;
    }

    let plan = execution_plan_at(plan_index);
    if (plan.reserved != 0u
        || plan.execution_family > EXECUTION_FAMILY_COMPOSITION
        || plan.left_program >= evaluator_config.execution_program_count
        || (plan.right_program != INVALID && (plan.right_program >= evaluator_config.execution_program_count)
        )
        || (plan.right_program == INVALID && (plan.execution_family == EXECUTION_FAMILY_COMPOSITION
            || plan.blended_channels != 0u || plan.mask_kind != 0u
            || plan.mask_index != 0u || plan.blend_domain != 0u))
        ) {
        evaluations[ray_index] = result;
        return;
    }
    if plan.left_program != program_index && plan.right_program != program_index {
        evaluations[ray_index] = result;
        return;
    }

    let program = execution_program_at(program_index);
    if !native_execution_program_is_valid(program, package_index, permutation_low, permutation_high)
        || !native_lobe_mask_is_supported(program.response_kind, header.lobe_mask) {
        evaluations[ray_index] = result;
        return;
    }
    let left_program = execution_program_at(plan.left_program);
    if !native_execution_arm_is_valid(left_program, permutation_low, permutation_high)
        || !native_lobe_mask_is_supported(left_program.response_kind, header.lobe_mask) {
        evaluations[ray_index] = result;
        return;
    }
    if plan.right_program == INVALID {
        if plan.execution_family == EXECUTION_FAMILY_COMPOSITION
            || plan.blended_channels != 0u
            || plan.mask_kind != 0u
            || plan.mask_index != 0u
            || plan.blend_domain != 0u
            || (left_program.response_kind == RESPONSE_PROGRAM_KIND_LIT
                && plan.execution_family != EXECUTION_FAMILY_FUSED_BUILTIN)
            || ((left_program.response_kind == RESPONSE_PROGRAM_KIND_NPR
                || left_program.response_kind == RESPONSE_PROGRAM_KIND_UNLIT
                || left_program.response_kind == RESPONSE_PROGRAM_KIND_CUSTOM)
                && plan.execution_family != EXECUTION_FAMILY_SPLIT_RESPONSE) {
            evaluations[ray_index] = result;
            return;
        }
    } else {
        let right_program = execution_program_at(plan.right_program);
        if !native_execution_arm_is_valid(right_program, permutation_low, permutation_high)
            || !native_lobe_mask_is_supported(right_program.response_kind, header.lobe_mask)
            || plan.blended_channels == 0u
            || (plan.blended_channels & ~RESPONSE_CHANNEL_RADIANCE) != 0u
            || !native_channels_are_supported(left_program.response_kind, plan.blended_channels)
            || !native_channels_are_supported(right_program.response_kind, plan.blended_channels)
            || plan.blend_domain > BLEND_DOMAIN_CUSTOM_CHANNELS
            || plan.mask_kind != MASK_KIND_DYNAMIC_CHANNEL || plan.mask_index > 3u
            || plan.execution_family == EXECUTION_FAMILY_SPLIT_RESPONSE {
            evaluations[ray_index] = result;
            return;
        }
        let builtin_pair = left_program.response_kind != RESPONSE_PROGRAM_KIND_CUSTOM
            && right_program.response_kind != RESPONSE_PROGRAM_KIND_CUSTOM;
        let both_lit = left_program.response_kind == RESPONSE_PROGRAM_KIND_LIT
            && right_program.response_kind == RESPONSE_PROGRAM_KIND_LIT;
        if (plan.execution_family == EXECUTION_FAMILY_FUSED_BUILTIN
            && (!builtin_pair || !both_lit))
            || (plan.execution_family == EXECUTION_FAMILY_COMPOSITION
                && (builtin_pair && both_lit))
            || (plan.execution_family != EXECUTION_FAMILY_FUSED_BUILTIN
                && plan.execution_family != EXECUTION_FAMILY_COMPOSITION) {
            evaluations[ray_index] = result;
            return;
        }
        if (plan.blend_domain == BLEND_DOMAIN_FULL_RADIANCE
            && plan.blended_channels != RESPONSE_CHANNEL_RADIANCE)
            || (plan.blend_domain == BLEND_DOMAIN_DIRECT_ONLY
                && plan.blended_channels != RESPONSE_CHANNEL_DIRECT)
            || (plan.blend_domain == BLEND_DOMAIN_CUSTOM_CHANNELS
                && plan.blended_channels == 0u) {
            evaluations[ray_index] = result;
            return;
        }
    }

    let parameter_words = header.parameter_size / 4u;
    if (parameter_words != prism_packed_word_count(header.lobe_mask)
        || parameter_words > PRISM_SURFACE_MAX_WORDS
        || header.parameter_offset > evaluator_config.parameter_word_count
        || parameter_words > evaluator_config.parameter_word_count - header.parameter_offset
        || evaluator_config.parameter_word_count > arrayLength(&material_parameters)) {
        evaluations[ray_index] = result;
        return;
    }
    var packed_words: array<u32, PRISM_SURFACE_MAX_WORDS>;
    for (var word = 0u; word < parameter_words; word += 1u) {
        packed_words[word] = material_parameters[header.parameter_offset + word];
    }
    let surface = prism_unpack_surface(packed_words, header.lobe_mask);
    let uv = vec2<f32>(
        bitcast<f32>(trace_hits[hit_base + 18u]),
        bitcast<f32>(trace_hits[hit_base + 19u]),
    );
    if !finite(uv.x) || !finite(uv.y) {
        evaluations[ray_index] = result;
        return;
    }
    var sampled = sampled_material_defaults(surface);
    for (var row_index = 0u; row_index < header.texture_count; row_index += 1u) {
        let row = material_texture_rows[header.texture_offset + row_index];
        if !native_texture_row_binding_is_valid(row)
            || !native_texture_semantic_is_supported(row.semantic) {
            // A semantic that this evaluator cannot fold is a missing material
            // input, never an identity/white texture.
            evaluations[ray_index] = result;
            return;
        }
        let texel_sample = sample_texture_row(row, uv);
        if !texel_sample.valid {
            // Direct authored images and stale texture generations are not
            // silently replaced by a fallback in this portable evaluator.
            evaluations[ray_index] = result;
            return;
        }
        sampled = fold_material_texel(row, texel_sample.value, surface.normal_scale, sampled);
    }
    let geometry_normal = vec3<f32>(
        bitcast<f32>(trace_hits[hit_base + 20u]),
        bitcast<f32>(trace_hits[hit_base + 21u]),
        bitcast<f32>(trace_hits[hit_base + 22u]),
    );
    let tangent_input = vec4<f32>(
        bitcast<f32>(trace_hits[hit_base + 23u]),
        bitcast<f32>(trace_hits[hit_base + 24u]),
        bitcast<f32>(trace_hits[hit_base + 25u]),
        bitcast<f32>(trace_hits[hit_base + 26u]),
    );
    let view_direction = -vec3<f32>(
        bitcast<f32>(trace_hits[hit_base + 27u]),
        bitcast<f32>(trace_hits[hit_base + 28u]),
        bitcast<f32>(trace_hits[hit_base + 29u]),
    );
    if !finite3(geometry_normal) || !finite3(tangent_input.xyz)
        || !finite(tangent_input.w) || abs(abs(tangent_input.w) - 1.0) > 1.0e-3
        || !finite3(view_direction) {
        evaluations[ray_index] = result;
        return;
    }
    let geometry_frame = native_tangent_frame(geometry_normal, tangent_input);
    let sampled_normal = select(
        geometry_frame.normal,
        native_apply_tangent_normal(sampled.normal_tangent, geometry_frame, geometry_frame.normal),
        sampled.has_normal_map,
    );
    let sampled_frame = ShadingFrame(
        sampled_normal,
        native_normalize_or(view_direction, sampled_normal),
        geometry_frame.tangent,
        geometry_frame.bitangent,
    );
    let sampled_surface = native_surface_sample(surface, sampled, sampled_frame);
    let alpha = clamp(sampled.base_alpha * surface.geometry_opacity, 0.0, 1.0);
    let left = native_arm_response(
        surface, sampled, sampled_surface, header, left_program, sampled_frame, uv, alpha,
        permutation_low, permutation_high,
    );
    if !left.valid {
        evaluations[ray_index] = result;
        return;
    }
    var response = left;
    if plan.right_program != INVALID {
        let right_program = execution_program_at(plan.right_program);
        let right = native_arm_response(
            surface, sampled, sampled_surface, header, right_program, sampled_frame, uv, alpha,
            permutation_low, permutation_high,
        );
        let mask = native_mask_value(header, plan.mask_index);
        if !right.valid || !mask.valid {
            evaluations[ray_index] = result;
            return;
        }
        let blend_mask = mask.value.x;
        if (plan.blended_channels & 1u) != 0u {
            response.direct = mix(left.direct, right.direct, blend_mask);
        }
        if (plan.blended_channels & 2u) != 0u {
            response.indirect_diffuse = mix(left.indirect_diffuse, right.indirect_diffuse, blend_mask);
        }
        if (plan.blended_channels & 4u) != 0u {
            response.reflection = mix(left.reflection, right.reflection, blend_mask);
        }
        if (plan.blended_channels & 8u) != 0u {
            response.transmission = mix(left.transmission, right.transmission, blend_mask);
        }
        if (plan.blended_channels & 16u) != 0u {
            response.emission = mix(left.emission, right.emission, blend_mask);
        }
        // Alpha is a shared coverage contract, not a response channel.
        response.alpha = left.alpha;
    }
    let response_rgb = response.direct + response.indirect_diffuse + response.reflection
        + response.transmission + response.emission;
    if !finite3(response_rgb) || any(response_rgb < vec3<f32>(0.0))
        || !finite(response.alpha) || response.alpha < 0.0 || response.alpha > 1.0 {
        evaluations[ray_index] = result;
        return;
    }

    result.response_rgb = response_rgb;
    result.valid = 1u;
    result.instance_id = instance_id;
    result.geometry_id = geometry_id;
    result.material_index = material_index;
    result.material_generation = material_generation;
    result.material_revision = material_revision;
    result.material_epoch_low = material_epoch_low;
    result.material_epoch_high = material_epoch_high;
    result.execution_plan_index = plan_index;
    result.execution_program_index = program_index;
    result.execution_package_index = package_index;
    result.execution_permutation_low = permutation_low;
    result.execution_permutation_high = permutation_high;
    result.response_kind = program.response_kind;
    result.execution_family = plan.execution_family;
    result.blended_channels = plan.blended_channels;
    result.response_program_id = program.program_id;
    evaluations[ray_index] = result;
}
