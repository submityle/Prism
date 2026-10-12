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
    closure_graph_offset: u32,
    specialization_low: u32,
    specialization_high: u32,
    lobe_mask: u32,
}

struct PrismSurfaceParameters {
    base_color: vec4<f32>,
    emissive: vec4<f32>,
    metallic: f32,
    perceptual_roughness: f32,
    // `OpenPBR` specular group (see `brdf.wesl::SurfaceSample`).
    specular_weight: f32,
    specular_color: vec4<f32>,
    specular_ior: f32,
    // `OpenPBR` `geometry_thin_walled` (single-interface open surface).
    thin_walled: f32,
    ambient_occlusion: f32,
    normal_scale: f32,
    alpha_cutoff: f32,
    // `OpenPBR` base group: `base_weight` scales the base layer and
    // `diffuse_roughness` drives the Oren-Nayar diffuse lobe.
    base_weight: f32,
    diffuse_roughness: f32,
    transmission: f32,
    thickness: f32,
    clearcoat: f32,
    clearcoat_roughness: f32,
    // OpenPBR `coat_ior` (relative coat index) and `coat_color` (highlight
    // tint). Mirrors `GpuSurfaceParameters::coat_ior` / `coat_color`.
    coat_ior: f32,
    coat_color: vec4<f32>,
    anisotropy: f32,
    anisotropy_rotation: f32,
    sheen: f32,
    subsurface: f32,
    index_of_refraction: f32,
    dispersion: f32,
    face_softness: f32,
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
