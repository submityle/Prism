struct PrismMaterialHeader {
    generation: u32,
    revision: u32,
    illumination: u32,
    render_class: u32,
    feature_flags: u32,
    closure_mask: u32,
    parameter_offset: u32,
    parameter_size: u32,
    texture_offset: u32,
    texture_count: u32,
    sampler_offset: u32,
    sampler_count: u32,
    custom_program: u32,
    is_active: u32,
    material_epoch_low: u32,
    material_epoch_high: u32,
    specialization_low: u32,
    specialization_high: u32,
    lobe_mask: u32,
    // PCM extension digest (§5.3): `used` and `required`, each split low/high
    // so all 64 registry slots are addressable. Mirrors
    // `GpuMaterialHeader::extensions_{used,required}_{low,high}`.
    extensions_used_low: u32,
    extensions_used_high: u32,
    extensions_required_low: u32,
    extensions_required_high: u32,
    // Runtime dynamic channels (§17.1), each an `f32` bitcast into the word;
    // `0.0` is the exact identity. A runtime state change rewrites only
    // these four words (the row is re-uploaded, the record is not
    // republished), so wetness/snow/damage/dissolve stay out of the
    // permutation space.
    dynamic_wetness: u32,
    dynamic_snow: u32,
    dynamic_damage: u32,
    dynamic_dissolve: u32,
}

struct PrismSurfaceParameters {
    base_color: vec4<f32>,
    emission_color: vec4<f32>,
    // `OpenPBR` `emission_luminance` (nits); radiance = emissive * luminance.
    emission_luminance: f32,
    base_metalness: f32,
    specular_roughness: f32,
    // `OpenPBR` specular group (see `brdf.wesl::SurfaceSample`).
    specular_weight: f32,
    specular_color: vec4<f32>,
    specular_ior: f32,
    // `OpenPBR` `geometry_thin_walled` (single-interface open surface).
    thin_walled: f32,
    // `OpenPBR` `geometry_opacity` in [0, 1].
    geometry_opacity: f32,
    ambient_occlusion: f32,
    normal_scale: f32,
    alpha_cutoff: f32,
    // `OpenPBR` base group: `base_weight` scales the base layer and
    // `diffuse_roughness` drives the Oren-Nayar diffuse lobe.
    base_weight: f32,
    base_diffuse_roughness: f32,
    transmission_weight: f32,
    thickness: f32,
    coat_weight: f32,
    coat_roughness: f32,
    // OpenPBR `coat_ior` (relative coat index) and `coat_color` (highlight
    // tint). Mirrors `GpuSurfaceParameters::coat_ior` / `coat_color`.
    coat_ior: f32,
    coat_color: vec4<f32>,
    anisotropy: f32,
    anisotropy_rotation: f32,
    fuzz_weight: f32,
    subsurface_weight: f32,
    // `OpenPBR` subsurface group (see `brdf.wesl::SurfaceSample`).
    subsurface_color: vec4<f32>,
    subsurface_radius: f32,
    subsurface_radius_scale: vec4<f32>,
    index_of_refraction: f32,
    transmission_dispersion_scale: f32,
    // `OpenPBR` `transmission_color`: tint of the refracted light.
    transmission_color: vec4<f32>,
    transmission_scatter: vec4<f32>,
    transmission_scatter_anisotropy: f32,
    transmission_dispersion_abbe_number: f32,
    coat_darkening: f32,
    subsurface_scatter_anisotropy: f32,
    npr_face_softness: f32,
    // Prism glint lobe (§17.6): a discrete microfacet estimator blended into the
    // specular lobe. `glint_weight == 0` (the default) keeps it inert, so the
    // fields cost 4 words of übers only for materials that declare the lobe.
    glint_weight: f32,
    glint_roughness: f32,
    glint_density: f32,
    glint_bin_half_width: f32,
    // Measured (tabulated) BRDF/BTF lobe (§17.6). The table itself is a material
    // texture (a provider); these are the authored response parameters.
    measured_weight: f32,
    measured_phi_d: f32,
    measured_scale: f32,
}

struct PrismMaterialTexture {
    index: u32,
    generation: u32,
    semantic: u32,
    sampler_index: u32,
}

fn prism_material_is_valid(
    header: PrismMaterialHeader,
    expected_generation: u32,
) -> bool {
    return header.is_active != 0u && header.generation == expected_generation;
}

fn prism_resolve_material_index(
    header: PrismMaterialHeader,
    expected_generation: u32,
    requested_index: u32,
) -> u32 {
    return select(0u, requested_index, prism_material_is_valid(header, expected_generation));
}
