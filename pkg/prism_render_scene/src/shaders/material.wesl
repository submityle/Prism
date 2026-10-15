struct PrismMaterialHeader {
    generation: u32,
    revision: u32,
    render_class: u32,
    render_domain: u32,
    feature_flags: u32,
    closure_mask: u32,
    parameter_offset: u32,
    parameter_size: u32,
    texture_offset: u32,
    texture_count: u32,
    sampler_offset: u32,
    sampler_count: u32,
    is_active: u32,
    material_epoch_low: u32,
    material_epoch_high: u32,
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
    // ABI v8: verified dense plan/program/package identity plus the complete
    // 64-bit permutation index. Each word is 0xffffffff when unbound.
    execution_plan_index: u32,
    execution_program_index: u32,
    execution_package_index: u32,
    execution_permutation_low: u32,
    execution_permutation_high: u32,
}

struct PrismSurfaceParameters {
    base_color: vec4<f32>,
    emission_color: vec4<f32>,
    emission_luminance: f32,
    base_metalness: f32,
    specular_roughness: f32,
    specular_weight: f32,
    ambient_occlusion: f32,
    normal_scale: f32,
    alpha_cutoff: f32,
    base_weight: f32,
    base_diffuse_roughness: f32,
    specular_color: vec4<f32>,
    specular_ior: f32,
    thin_walled: f32,
    geometry_opacity: f32,
    transmission_weight: f32,
    thickness: f32,
    coat_weight: f32,
    coat_roughness: f32,
    coat_ior: f32,
    coat_color: vec4<f32>,
    coat_roughness_anisotropy: f32,
    coat_darkening: f32,
    anisotropy: f32,
    anisotropy_rotation: f32,
    fuzz_weight: f32,
    fuzz_roughness: f32,
    fuzz_color: vec4<f32>,
    subsurface_weight: f32,
    subsurface_color: vec4<f32>,
    subsurface_radius: f32,
    subsurface_radius_scale: vec4<f32>,
    subsurface_scatter_anisotropy: f32,
    index_of_refraction: f32,
    transmission_dispersion_scale: f32,
    transmission_color: vec4<f32>,
    transmission_scatter: vec4<f32>,
    thin_film_weight: f32,
    thin_film_thickness: f32,
    thin_film_ior: f32,
    transmission_scatter_anisotropy: f32,
    transmission_dispersion_abbe_number: f32,
    npr_face_softness: f32,
    npr_stylized_ramp_softness: f32,
    npr_stylized_specular_scale: f32,
    npr_stylized_rim_scale: f32,
    glint_weight: f32,
    glint_roughness: f32,
    glint_density: f32,
    glint_bin_half_width: f32,
    measured_weight: f32,
    measured_phi_d: f32,
    measured_scale: f32,
}

struct PrismMaterialTexture {
    index: u32,
    generation: u32,
    semantic: u32,
    sampler_index: u32,
    virtual_texture_id: u32,
    virtual_texture_width: u32,
    virtual_texture_height: u32,
    virtual_texture_page_size: u32,
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
