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
const RESPONSE_PROGRAM_KIND_LIT: u32 = 0u;
const RESPONSE_PROGRAM_KIND_CUSTOM: u32 = 3u;
const BUILTIN_UNLIT_PROGRAM_ID: u32 = 2u;
const BUILTIN_LIT_PROGRAM_ID: u32 = 0u;
const TEXTURE_SEMANTIC_BASE_COLOR: u32 = 0u;
const TEXTURE_SEMANTIC_EMISSIVE: u32 = 1u;
const NO_VIRTUAL_TEXTURE: u32 = 0xffffffffu;
const PORTABLE_TEXTURE_ATLAS: u32 = 0xfffffffeu;
const EXECUTION_FAMILY_FUSED_BUILTIN: u32 = 0u;
const EXECUTION_FAMILY_SPLIT_RESPONSE: u32 = 1u;
const EXECUTION_FAMILY_COMPOSITION: u32 = 2u;
const MASK_KIND_DYNAMIC_CHANNEL: u32 = 5u;
const BLEND_DOMAIN_FULL_RADIANCE: u32 = 0u;
const BLEND_DOMAIN_DIRECT_ONLY: u32 = 1u;
const BLEND_DOMAIN_CUSTOM_CHANNELS: u32 = 2u;

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

@group(0) @binding(0) var<storage, read> trace_hits: array<u32>;
@group(0) @binding(1) var<storage, read> hit_attributes: array<u32>;
@group(0) @binding(2) var<storage, read> trace_rays: array<u32>;
@group(0) @binding(3) var<uniform> evaluator_config: NativeGiMaterialEvaluatorConfig;
@group(0) @binding(4) var<storage, read_write> evaluations: array<NativeGiMaterialEvaluationRecord>;

@group(1) @binding(0) var<storage, read> material_headers: array<PrismMaterialHeader>;
@group(1) @binding(1) var<storage, read> material_parameters: array<u32>;
@group(1) @binding(2) var<storage, read> material_texture_rows: array<PrismMaterialTexture>;
@group(1) @binding(3) var<storage, read> execution_plans: array<ExecutionPlanRow>;
@group(1) @binding(4) var<storage, read> execution_programs: array<ExecutionProgramRow>;
// Binding 5 is owned by the imported VT resolver. Its runtime array is
// clipped against the published entry_count inside material_vt_exact_slot.
@group(1) @binding(6) var<storage, read> material_vt_config: MaterialVirtualTextureConfig;
@group(1) @binding(7) var material_vt_atlas: texture_2d_array<f32>;
@group(1) @binding(8) var material_vt_sampler: sampler;

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

fn hit_matches_attributes(hit_base: u32, attribute_base: u32) -> bool {
    if (trace_hits[hit_base] != 1u) { return false; }
    // Hit identity is [instance, geometry, material slot, generation, revision,
    // epoch low/high, plan, program, package, permutation low/high].
    for (var word = 0u; word < 12u; word += 1u) {
        if (trace_hits[hit_base + 6u + word] != hit_attributes[attribute_base + word]) {
            return false;
        }
    }
    return true;
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

fn srgb_channel_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        return value / 12.92;
    }
    return pow((value + 0.055) / 1.055, 2.4);
}

fn srgb_to_linear(value: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        srgb_channel_to_linear(value.x),
        srgb_channel_to_linear(value.y),
        srgb_channel_to_linear(value.z),
    );
}

fn sample_vt_row(row: PrismMaterialTexture, uv: vec2<f32>) -> TextureSample {
    var out: TextureSample;
    out.valid = false;
    out.value = vec4<f32>(0.0);
    if row.generation == 0u
        || evaluator_config.vt_page_table_word_count == 0u
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

fn sample_texture_row(row: PrismMaterialTexture, uv: vec2<f32>) -> TextureSample {
    // Direct rows have no image heap in this evaluator ABI. Treating one as a
    // white texel would make a stale or unbound generation look valid.
    return sample_vt_row(row, uv);
}

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
    return invalid;
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

fn native_registered_surface(surface: PrismSurfaceParameters) -> RegisteredSurface {
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
        vec3<f32>(0.0), surface.coat_roughness_anisotropy, vec3<f32>(0.0),
        surface.fuzz_weight, surface.fuzz_roughness, surface.fuzz_color.xyz,
        surface.subsurface_weight, surface.subsurface_color.xyz, surface.subsurface_radius,
        surface.subsurface_radius_scale.xyz, surface.thickness, surface.anisotropy,
        surface.anisotropy_rotation, 0.0, surface.thin_film_weight,
        surface.thin_film_thickness * 1000.0, surface.thin_film_ior, surface.glint_weight,
        surface.glint_roughness, surface.glint_density, surface.glint_bin_half_width,
        -1.0, vec2<f32>(0.0), 0.0,
    );
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
    header: PrismMaterialHeader,
    normal: vec3<f32>,
    view_direction: vec3<f32>,
    uv: vec2<f32>,
    program_id: u32,
    permutation_low: u32,
    permutation_high: u32,
) -> NativeResponse {
    var out = invalid_native_response();
    let frame = RegisteredFrame(normal, view_direction, vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0));
    let geometry = RegisteredGeometry(vec3<f32>(0.0), normal, frame.tangent, frame.bitangent, uv, 0u);
    let environment = RegisteredEnvironment(
        vec3<f32>(0.0), 0u,
        array<vec4<f32>, 9>(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0)),
        0u, 0u, 1.0, 0u,
        RegisteredStylizedParams(1u, 0.0, 0.0, 0.5, 0.1, 0.0, 0.5, 0.1, vec3<f32>(1.0), 0.0, vec3<f32>(1.0), 1.0),
    );
    let lighting = RegisteredLightingContext(
        environment, frame, vec3<f32>(0.0), 0u, 0u, 0u, 0u, 0u,
        vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(1.0),
    );
    let view = RegisteredView(
        view_direction, vec2<u32>(0u), vec2<u32>(0u), 0u,
        mat4x4<f32>(vec4<f32>(1.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 1.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 1.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 1.0)),
        mat4x4<f32>(vec4<f32>(1.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 1.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 1.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 1.0)),
    );
    let custom = RegisteredCustomData(native_registered_header(header), permutation_low, permutation_high);
    let response = prism_native_registered_evaluate(
        RegisteredSurfaceData(native_registered_surface(surface), frame, geometry, native_registered_parameters(surface)),
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

fn native_builtin_response(
    surface: PrismSurfaceParameters,
    base_color: vec3<f32>,
    emission_color: vec3<f32>,
    normal: vec3<f32>,
    view_direction: vec3<f32>,
    response_kind: u32,
) -> NativeResponse {
    var out = invalid_native_response();
    let n = native_normalize_or(normal, vec3<f32>(0.0, 1.0, 0.0));
    let v = native_normalize_or(view_direction, n);
    let n_dot_v = max(dot(n, v), 0.0);
    let emission = emission_color * surface.emission_luminance;
    if response_kind == RESPONSE_PROGRAM_KIND_UNLIT {
        out.direct = base_color;
        out.emission = emission;
        out.alpha = surface.geometry_opacity;
        out.valid = all(finite3(base_color + emission)) && out.alpha >= 0.0 && out.alpha <= 1.0;
        return out;
    }
    if response_kind != RESPONSE_PROGRAM_KIND_LIT {
        return out;
    }
    let metallic = clamp(surface.base_metalness, 0.0, 1.0);
    let roughness = clamp(surface.specular_roughness, 0.04, 1.0);
    let alpha = roughness * roughness;
    let ior = clamp(surface.specular_ior, 1.0, 3.0);
    let dielectric = (ior - 1.0) / (ior + 1.0);
    let f0 = mix(vec3<f32>(dielectric * dielectric) * surface.specular_color.xyz, base_color, metallic);
    let fresnel = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - n_dot_v, 5.0);
    let denominator = max(4.0 * n_dot_v * n_dot_v, 1.0e-4);
    let d = alpha * alpha / max(3.14159265 * pow((n_dot_v * n_dot_v) * (alpha * alpha - 1.0) + 1.0, 2.0), 1.0e-5);
    let k = (roughness + 1.0) * (roughness + 1.0) * 0.125;
    let g = (n_dot_v / max(n_dot_v * (1.0 - k) + k, 1.0e-5));
    let specular = fresnel * d * g * g / denominator;
    let diffuse = base_color * clamp(surface.base_weight, 0.0, 1.0) * (1.0 - metallic)
        * n_dot_v / 3.14159265;
    let response = diffuse + specular + emission;
    out.direct = diffuse + specular;
    out.emission = emission;
    out.alpha = surface.geometry_opacity;
    out.valid = all(finite3(response)) && all(response >= vec3<f32>(0.0))
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
    if program.response_kind == 1u
        || program.response_kind > RESPONSE_PROGRAM_KIND_CUSTOM
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

fn native_arm_response(
    surface: PrismSurfaceParameters,
    header: PrismMaterialHeader,
    program: ExecutionProgramRow,
    normal: vec3<f32>,
    view_direction: vec3<f32>,
    uv: vec2<f32>,
    permutation_low: u32,
    permutation_high: u32,
) -> NativeResponse {
    var shaded = surface;
    if program.response_kind == RESPONSE_PROGRAM_KIND_CUSTOM {
        return native_registered_response(
            shaded, header, normal, view_direction, uv, program.program_id,
            permutation_low, permutation_high,
        );
    }
    if program.response_kind == RESPONSE_PROGRAM_KIND_LIT && program.program_id != BUILTIN_LIT_PROGRAM_ID {
        return invalid_native_response();
    }
    if program.response_kind == RESPONSE_PROGRAM_KIND_UNLIT && program.program_id != BUILTIN_UNLIT_PROGRAM_ID {
        return invalid_native_response();
    }
    return native_builtin_response(
        shaded, shaded.base_color.xyz, shaded.emission_color.xyz,
        normal, view_direction, program.response_kind,
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
    if (triangle_index >= arrayLength(&hit_attributes) / HIT_ATTRIBUTE_WORDS) {
        evaluations[ray_index] = result;
        return;
    }
    let attribute_base = triangle_index * HIT_ATTRIBUTE_WORDS;
    if (trace_hits[hit_base] != 1u
        || triangle_index == INVALID
        || !hit_matches_attributes(hit_base, attribute_base)) {
        evaluations[ray_index] = result;
        return;
    }
    let ray_base = ray_index * 12u;
    if (ray_base + 6u >= arrayLength(&trace_rays)) {
        evaluations[ray_index] = result;
        return;
    }

    if (evaluator_config.abi_version != MATERIAL_EVALUATOR_ABI_VERSION
        || evaluator_config.execution_table_version == 0u
        || evaluator_config.material_header_count == 0u
        || evaluator_config.parameter_word_count == 0u
        || evaluator_config.texture_row_count == 0u
        || evaluator_config.execution_plan_count == 0u
        || evaluator_config.execution_program_count == 0u
        || evaluator_config.material_header_count > arrayLength(&material_headers)
        || evaluator_config.parameter_word_count > arrayLength(&material_parameters)
        || evaluator_config.texture_row_count > arrayLength(&material_texture_rows)
        || evaluator_config.execution_plan_count > arrayLength(&execution_plans)
        || evaluator_config.execution_program_count > arrayLength(&execution_programs)
        || evaluator_config.texture_slot_count != 0u
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

    let material_index = hit_attributes[attribute_base + 2u];
    let material_generation = hit_attributes[attribute_base + 3u];
    let material_revision = hit_attributes[attribute_base + 4u];
    let material_epoch_low = hit_attributes[attribute_base + 5u];
    let material_epoch_high = hit_attributes[attribute_base + 6u];
    let plan_index = hit_attributes[attribute_base + 7u];
    let program_index = hit_attributes[attribute_base + 8u];
    let package_index = hit_attributes[attribute_base + 9u];
    let permutation_low = hit_attributes[attribute_base + 10u];
    let permutation_high = hit_attributes[attribute_base + 11u];
    if (hit_attributes[attribute_base] == INVALID
        || hit_attributes[attribute_base + 1u] == INVALID
        || material_index == INVALID
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
        || plan_index >= arrayLength(&execution_plans)
        || program_index >= evaluator_config.execution_program_count
        || program_index >= arrayLength(&execution_programs)) {
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
        || header.texture_offset + header.texture_count > arrayLength(&material_texture_rows)) {
        evaluations[ray_index] = result;
        return;
    }

    let plan = execution_plans[plan_index];
    if (plan.reserved != 0u
        || plan.execution_family > EXECUTION_FAMILY_COMPOSITION
        || plan.left_program != program_index
        || plan.left_program >= evaluator_config.execution_program_count
        || plan.left_program >= arrayLength(&execution_programs)
        || (plan.right_program != INVALID && (plan.right_program >= evaluator_config.execution_program_count
            || plan.right_program >= arrayLength(&execution_programs))
        )
        || (plan.right_program == INVALID && (plan.execution_family == EXECUTION_FAMILY_COMPOSITION
            || plan.blended_channels != 0u || plan.mask_kind != 0u
            || plan.mask_index != 0u || plan.blend_domain != 0u))
        ) {
        evaluations[ray_index] = result;
        return;
    }
    let program = execution_programs[program_index];
    if !native_execution_program_is_valid(program, package_index, permutation_low, permutation_high) {
        evaluations[ray_index] = result;
        return;
    }
    if plan.right_program == INVALID
        && ((program.response_kind == RESPONSE_PROGRAM_KIND_LIT
                && plan.execution_family != EXECUTION_FAMILY_FUSED_BUILTIN)
            || (program.response_kind == RESPONSE_PROGRAM_KIND_UNLIT
                && plan.execution_family != EXECUTION_FAMILY_SPLIT_RESPONSE)
            || (program.response_kind == RESPONSE_PROGRAM_KIND_CUSTOM
                && plan.execution_family != EXECUTION_FAMILY_SPLIT_RESPONSE)) {
        evaluations[ray_index] = result;
        return;
    }
    var right_program = program;
    if (plan.right_program != INVALID) {
        right_program = execution_programs[plan.right_program];
        if !native_execution_program_is_valid(
            right_program, package_index, permutation_low, permutation_high,
        ) {
            evaluations[ray_index] = result;
            return;
        }
        let builtin_pair = program.response_kind != RESPONSE_PROGRAM_KIND_CUSTOM
            && right_program.response_kind != RESPONSE_PROGRAM_KIND_CUSTOM;
        let both_lit = program.response_kind == RESPONSE_PROGRAM_KIND_LIT
            && right_program.response_kind == RESPONSE_PROGRAM_KIND_LIT;
        if ((builtin_pair
                && ((both_lit && plan.execution_family != EXECUTION_FAMILY_FUSED_BUILTIN)
                    || (!both_lit && plan.execution_family != EXECUTION_FAMILY_COMPOSITION)))
            || (!builtin_pair && plan.execution_family != EXECUTION_FAMILY_COMPOSITION)
            || plan.execution_family == EXECUTION_FAMILY_SPLIT_RESPONSE
            || plan.blended_channels == 0u || plan.blended_channels > 31u
            || plan.blend_domain > BLEND_DOMAIN_CUSTOM_CHANNELS
            || plan.mask_kind != MASK_KIND_DYNAMIC_CHANNEL || plan.mask_index > 3u) {
            evaluations[ray_index] = result;
            return;
        }
        if (plan.blend_domain == BLEND_DOMAIN_FULL_RADIANCE && plan.blended_channels != 31u) {
            evaluations[ray_index] = result;
            return;
        }
        if (plan.blend_domain == BLEND_DOMAIN_DIRECT_ONLY && plan.blended_channels != 1u) {
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
    var base_color = surface.base_color.xyz;
    var emission_color = surface.emission_color.xyz;
    for (var row_index = 0u; row_index < header.texture_count; row_index += 1u) {
        let row = material_texture_rows[header.texture_offset + row_index];
        let sampled = sample_texture_row(row, uv);
        if !sampled.valid {
            // Direct authored images and stale texture generations are not
            // silently replaced by a fallback in this portable evaluator.
            evaluations[ray_index] = result;
            return;
        }
        if row.semantic == TEXTURE_SEMANTIC_BASE_COLOR {
            base_color = base_color * srgb_to_linear(sampled.value.xyz);
        } else if row.semantic == TEXTURE_SEMANTIC_EMISSIVE {
            emission_color = emission_color * srgb_to_linear(sampled.value.xyz);
        } else {
            // A row with an unsupported semantic is not interchangeable with
            // an absent row: reject the tuple instead of silently whitening it.
            evaluations[ray_index] = result;
            return;
        }
    }
    var shaded_surface = surface;
    shaded_surface.base_color = vec4<f32>(base_color, surface.base_color.w);
    shaded_surface.emission_color = vec4<f32>(emission_color, surface.emission_color.w);
    let normal = vec3<f32>(
        bitcast<f32>(trace_hits[hit_base + 20u]),
        bitcast<f32>(trace_hits[hit_base + 21u]),
        bitcast<f32>(trace_hits[hit_base + 22u]),
    );
    let view_direction = -vec3<f32>(
        bitcast<f32>(trace_rays[ray_base + 4u]),
        bitcast<f32>(trace_rays[ray_base + 5u]),
        bitcast<f32>(trace_rays[ray_base + 6u]),
    );
    if !finite3(normal) || !finite3(view_direction) {
        evaluations[ray_index] = result;
        return;
    }
    let left = native_arm_response(
        shaded_surface, header, program, normal, view_direction, uv,
        permutation_low, permutation_high,
    );
    if !left.valid {
        evaluations[ray_index] = result;
        return;
    }
    var response = left;
    if plan.right_program != INVALID {
        let right = native_arm_response(
            shaded_surface, header, right_program, normal, view_direction, uv,
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
    result.instance_id = hit_attributes[attribute_base];
    result.geometry_id = hit_attributes[attribute_base + 1u];
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
