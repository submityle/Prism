// Canonical material-table admission and unpacking for the Native composite.
// The producer identity is reconstructed into a per-pixel storage tuple after
// opaque raster. The frame owner pins the referenced allocations for the
// complete graph lifetime.
import prism_render_material::shaders::material::{PrismMaterialHeader, PrismSurfaceParameters};
import prism_render_material::shaders::material_unpack::{PRISM_LOBE_ALL, PRISM_SURFACE_MAX_WORDS, prism_neutral_surface, prism_packed_word_count, prism_unpack_surface};
import prism_render_app::shaders::brdf::{
    DirectLightSample,
    PrincipledPre,
    ShadingFrame,
    SurfaceSample,
    principled_direct_channels_with,
    principled_pre,
    stylized_direct,
};
import prism_render_app::shaders::lighting::StylizedParams;

struct CompositeMaterialSurface {
    valid: bool,
    program_kind: u32,
    surface: PrismSurfaceParameters,
    npr_modifiers: vec3<f32>,
};

struct SurfaceIdentity {
    material_index: u32,
    generation: u32,
    revision: u32,
    execution_plan_index: u32,
    program_kind: u32,
    lighting_surface_index: u32,
    left_program_index: u32,
    right_program_index: u32,
    left_program_kind: u32,
    right_program_kind: u32,
    blended_channels: u32,
    mask_kind: u32,
    mask_index: u32,
    blend_domain: u32,
    left_program_id: u32,
    right_program_id: u32,
};

@group(2) @binding(0) var<storage, read> composite_material_headers: array<PrismMaterialHeader>;
@group(2) @binding(1) var<storage, read> composite_material_parameters: array<u32>;

fn composite_neutral_surface() -> PrismSurfaceParameters {
    // This value is never consumed when `valid` is false.
    return prism_neutral_surface();
}

fn composite_execution_identity_is_valid(
    identity: SurfaceIdentity,
    header: PrismMaterialHeader,
) -> bool {
    if (identity.left_program_id == 0xffffffffu) { return false; }
    if (identity.left_program_kind < 3u &&
        identity.left_program_id != identity.left_program_kind) {
        return false;
    }

    let selected_left = header.execution_program_index == identity.left_program_index;
    let selected_right = identity.right_program_index != 0xffffffffu &&
        header.execution_program_index == identity.right_program_index;
    if (!selected_left && !selected_right) { return false; }
    if (selected_left && identity.program_kind != identity.left_program_kind) {
        return false;
    }
    if (selected_right && identity.program_kind != identity.right_program_kind) {
        return false;
    }

    if (identity.right_program_index == 0xffffffffu) {
        // A single arm is either a fused Lit builtin, a split-only builtin, or
        // a registered response. The selected header row must name that arm.
        return selected_left && identity.right_program_id == 0xffffffffu;
    }
    if (identity.right_program_id == 0xffffffffu ||
        (identity.right_program_kind < 3u &&
            identity.right_program_id != identity.right_program_kind)) {
        return false;
    }

    // SurfaceIdentity does not carry the execution-family word. Therefore the
    // consumer admits only shapes whose family is derivable from the arms:
    // builtin MaskBlend is the canonical fused Lit/Lit form; every other
    // supported blend must contain a registered arm and is composition.
    let builtin_pair = identity.left_program_kind < 3u &&
        identity.right_program_kind < 3u;
    if (builtin_pair && (identity.left_program_kind != 0u ||
        identity.right_program_kind != 0u)) {
        return false;
    }
    if (!builtin_pair && identity.left_program_kind != 3u &&
        identity.right_program_kind != 3u) {
        return false;
    }
    return true;
}

fn prism_composite_material_surface(identity: SurfaceIdentity) -> CompositeMaterialSurface {
    var result: CompositeMaterialSurface;
    result.valid = false;
    result.program_kind = 0xffffffffu;
    result.surface = composite_neutral_surface();
    result.npr_modifiers = vec3<f32>(1.0);

    let header_count = arrayLength(&composite_material_headers);
    if (identity.material_index >= header_count) { return result; }
    let header = composite_material_headers[identity.material_index];
    if (header.is_active == 0u ||
        (header.generation != identity.generation) ||
        header.revision != identity.revision ||
        header.execution_plan_index != identity.execution_plan_index ||
        (header.generation == 0u && identity.material_index != 0u) ||
        header.execution_plan_index == 0xffffffffu ||
        header.execution_program_index == 0xffffffffu ||
        header.execution_package_index == 0xffffffffu ||
        identity.left_program_index == 0xffffffffu ||
        identity.left_program_kind > 3u ||
        identity.program_kind > 3u ||
        (header.lobe_mask & ~PRISM_LOBE_ALL) != 0u) { return result; }

    if (!composite_execution_identity_is_valid(identity, header)) {
        return result;
    }

    if (identity.right_program_index == 0xffffffffu) {
        if (identity.right_program_kind != 0xffffffffu ||
            identity.blended_channels != 0u || identity.mask_kind != 0u ||
            identity.mask_index != 0u || identity.blend_domain != 0u ||
            identity.right_program_id != 0xffffffffu) {
            return result;
        }
    } else if (identity.right_program_kind > 3u ||
        identity.blended_channels == 0u || identity.mask_kind == 0u ||
        identity.blend_domain > 2u || identity.right_program_id == 0xffffffffu) {
        return result;
    }
    if (identity.right_program_index != 0xffffffffu) {
        if (identity.blend_domain == 0u && identity.blended_channels != 31u) {
            return result;
        }
        if (identity.blend_domain == 1u && identity.blended_channels != 1u) {
            return result;
        }
        if (identity.blend_domain == 2u &&
            (identity.blended_channels == 0u || (identity.blended_channels & ~31u) != 0u)) {
            return result;
        }
        // Dynamic channels are the only complete Native mask producer in this
        // composite ABI. Unsupported vertex/UV/region/texture sources reject
        // before either response arm is evaluated.
        if (identity.mask_kind != 5u || identity.mask_index > 3u) {
            return result;
        }
    }

    let packed_words = prism_packed_word_count(header.lobe_mask);
    if (header.parameter_size != packed_words * 4u) { return result; }
    let parameter_count = arrayLength(&composite_material_parameters);
    if (header.parameter_offset > parameter_count ||
        packed_words > parameter_count - header.parameter_offset) { return result; }

    var words: array<u32, PRISM_SURFACE_MAX_WORDS>;
    for (var index = 0u; index < packed_words; index += 1u) {
        words[index] = composite_material_parameters[header.parameter_offset + index];
    }
    result.surface = prism_unpack_surface(words, header.lobe_mask);
    result.npr_modifiers = vec3<f32>(
        result.surface.npr_stylized_ramp_softness,
        result.surface.npr_stylized_specular_scale,
        result.surface.npr_stylized_rim_scale,
    );
    // The identity prepass has already validated the immutable plan/program
    // tuple. The composite only revalidates the live header against that
    // per-pixel identity and does not bind the plan/program tables again.
    result.program_kind = identity.program_kind;
    result.valid = true;
    return result;
}

struct CompositeDirectChannels {
    valid: bool,
    direct: vec3<f32>,
    transmission: vec3<f32>,
    radiance: vec3<f32>,
}

// Native keeps the same logical response channels as the backend-neutral
// ResponseOutput contract.  The storage representation is deliberately the
// 16-word form (five RGB channels plus alpha), so the buffer can be consumed
// by a later composition pass without depending on vec3 host padding.
struct CompositeResponseOutput {
    words: array<f32, 16>,
}

fn composite_response_zero() -> CompositeResponseOutput {
    return CompositeResponseOutput(array<f32, 16>(
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0,
    ));
}

fn composite_response_from_channels(
    direct: vec3<f32>,
    indirect_diffuse: vec3<f32>,
    reflection: vec3<f32>,
    transmission: vec3<f32>,
    emission: vec3<f32>,
    alpha: f32,
) -> CompositeResponseOutput {
    return CompositeResponseOutput(array<f32, 16>(
        direct.x, direct.y, direct.z,
        indirect_diffuse.x, indirect_diffuse.y, indirect_diffuse.z,
        reflection.x, reflection.y, reflection.z,
        transmission.x, transmission.y, transmission.z,
        emission.x, emission.y, emission.z,
        alpha,
    ));
}

fn composite_response_is_finite(response: CompositeResponseOutput) -> bool {
    for (var index = 0u; index < 16u; index += 1u) {
        if (!finite(response.words[index])) { return false; }
    }
    return true;
}

fn composite_response_radiance(response: CompositeResponseOutput) -> vec3<f32> {
    return vec3<f32>(
        response.words[0] + response.words[3] + response.words[6]
            + response.words[9] + response.words[12],
        response.words[1] + response.words[4] + response.words[7]
            + response.words[10] + response.words[13],
        response.words[2] + response.words[5] + response.words[8]
            + response.words[11] + response.words[14],
    );
}

fn composite_surface_sample(parameters: PrismSurfaceParameters) -> SurfaceSample {
    var surface: SurfaceSample;
    surface.base_color = parameters.base_color.xyz;
    surface.base_weight = parameters.base_weight;
    surface.diffuse_roughness = parameters.base_diffuse_roughness;
    surface.metallic = parameters.base_metalness;
    surface.perceptual_roughness = parameters.specular_roughness;
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
    surface.ambient_occlusion = parameters.ambient_occlusion;
    surface.emissive = parameters.emission_color.xyz * parameters.emission_luminance;
    surface.clearcoat = parameters.coat_weight;
    surface.clearcoat_roughness = parameters.coat_roughness;
    surface.coat_ior = parameters.coat_ior;
    surface.coat_color = parameters.coat_color.xyz;
    surface.coat_normal = vec3<f32>(0.0);
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
    // These three values are shade-time inputs, not part of the material row.
    // Preserve the exact identity/sentinel semantics used by shading_resolve.
    surface.normal_variance = 0.0;
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

// Keep the raw runtime WGSL on this module's stable public surface. Imported
// BRDF declarations are intentionally name-mangled by WESL, so callers must
// not construct or name those imported types directly after linking.
fn composite_make_shading_frame(
    normal: vec3<f32>,
    view: vec3<f32>,
    tangent: vec3<f32>,
    bitangent: vec3<f32>,
) -> ShadingFrame {
    return ShadingFrame(normal, view, tangent, bitangent);
}

fn composite_make_direct_light(
    direction: vec3<f32>,
    illuminance: vec3<f32>,
    visibility: f32,
) -> DirectLightSample {
    return DirectLightSample(direction, illuminance, visibility);
}

fn composite_principled_pre(
    surface: SurfaceSample,
    frame: ShadingFrame,
) -> PrincipledPre {
    return principled_pre(surface, frame);
}

// Canonical builtin Lit evaluator. The channel split comes directly from the
// BRDF implementation; radiance retains its source-defined emission behavior.
fn composite_evaluate_builtin_direct(
    surface: SurfaceSample,
    frame: ShadingFrame,
    light: DirectLightSample,
    pre: PrincipledPre,
) -> CompositeDirectChannels {
    let channels = principled_direct_channels_with(surface, frame, light, pre);
    return CompositeDirectChannels(
        true,
        channels.direct,
        channels.transmission,
        channels.radiance,
    );
}

// Canonical builtin Unlit evaluator. Unlit has no analytic-light or
// transmission response; its radiance is the authored base color plus
// emission exactly once.
fn composite_evaluate_builtin_unlit(surface: SurfaceSample) -> CompositeDirectChannels {
    return CompositeDirectChannels(
        true,
        vec3<f32>(0.0),
        vec3<f32>(0.0),
        surface.base_color + surface.emissive,
    );
}

fn composite_frame_stylized() -> StylizedParams {
    return StylizedParams(
        params.stylized.bands,
        params.stylized.wrap,
        params.stylized.ramp_softness,
        params.stylized.shadow_threshold,
        params.stylized.shadow_softness,
        params.stylized.specular_intensity,
        params.stylized.specular_threshold,
        params.stylized.specular_softness,
        params.stylized.specular_color,
        params.stylized.rim_intensity,
        params.stylized.rim_color,
        params.stylized.rim_power,
    );
}

fn composite_evaluate_builtin_npr(
    surface: SurfaceSample,
    frame: ShadingFrame,
    light: DirectLightSample,
    npr_modifiers: vec3<f32>,
) -> CompositeDirectChannels {
    var stylized = composite_frame_stylized();
    stylized.ramp_softness = stylized.ramp_softness * npr_modifiers.x;
    stylized.specular_intensity = stylized.specular_intensity * npr_modifiers.y;
    stylized.rim_intensity = stylized.rim_intensity * npr_modifiers.z;
    // `stylized_direct` includes emission for the standalone reference API.
    // Native accumulates emission once after the light loop, so remove it from
    // this per-light call to avoid multiplying emission by light count.
    var direct_surface = surface;
    direct_surface.emissive = vec3<f32>(0.0);
    return CompositeDirectChannels(
        true,
        stylized_direct(direct_surface, frame, light, stylized),
        vec3<f32>(0.0),
        vec3<f32>(0.0),
    );
}
