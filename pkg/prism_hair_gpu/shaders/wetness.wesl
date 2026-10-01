// Compute twin of Prism's wet-hair saturation coupling
// (prism_render_architecture::hair::wetness::wet_hair_response /
// wet_hair_response_map). One thread maps one scalar wetness fraction to the
// five physical parameter modifiers (clump_scale, mass_mul, damping_mul,
// sigma_a_mul, roughness_delta). Every output is a straight linear
// interpolation between the dry endpoint and the wet endpoint, so the kernel is
// pure multiply/add with no transcendental step. The five wet endpoints are
// passed as host uniform so the shader never duplicates the golden's magic
// constants and cannot drift from them.

struct Params {
    // Wet endpoints at full saturation: (clump_scale, mass_mul, damping_mul,
    // sigma_a_mul). The dry endpoint for each of these is 1.0.
    wet: vec4<f32>,
    // Additive roughness offset at full saturation; the dry endpoint is 0.0.
    roughness_delta_wet: f32,
    element_count: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> wetness: array<f32>;
// Flattened output: five consecutive f32 modifiers per element, in the field
// order (clump_scale, mass_mul, damping_mul, sigma_a_mul, roughness_delta).
@group(0) @binding(2) var<storage, read_write> response: array<f32>;

const MAX_FINITE_F32: f32 = 3.40282347e+38;

// Clamp an arbitrary wetness input to the physical [0, 1] saturation range.
// Non-finite inputs (NaN / +/-inf) collapse to fully dry (0.0), bit-faithfully
// to the golden's sanitize_wetness (`is_finite()` then `clamp(0, 1)`): the
// finite test is `w == w` (rejects NaN) plus the finite-magnitude bound
// (rejects +/-inf), after which negative values clamp up to 0.
fn sanitize_wetness(w: f32) -> f32 {
    if w == w && w >= -MAX_FINITE_F32 && w <= MAX_FINITE_F32 {
        return clamp(w, 0.0, 1.0);
    }
    return 0.0;
}

// Linear interpolation between the dry endpoint `dry` (at w = 0) and the wet
// endpoint `wet` (at w = 1). Pure arithmetic, mirroring the golden `lerp`.
fn lerp_dry_wet(dry: f32, wet: f32, w: f32) -> f32 {
    return dry + (wet - dry) * w;
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.element_count {
        return;
    }
    let w = sanitize_wetness(wetness[idx]);
    let base = idx * 5u;
    response[base + 0u] = lerp_dry_wet(1.0, params.wet.x, w);
    response[base + 1u] = lerp_dry_wet(1.0, params.wet.y, w);
    response[base + 2u] = lerp_dry_wet(1.0, params.wet.z, w);
    response[base + 3u] = lerp_dry_wet(1.0, params.wet.w, w);
    response[base + 4u] = lerp_dry_wet(0.0, params.roughness_delta_wet, w);
}
