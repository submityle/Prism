//! GPU light ABI + analytic light math shared by the shading resolve pass.
//!
//! The `struct` layouts below are the byte-for-byte mirror of the Rust
//! `#[repr(C)]` records in `lighting/abi.rs` (validated there against
//! `size_of`), so the same storage buffers bind identically on both sides.
//! The functions mirror the CPU golden reference in `prism_render_shading`
//! (`punctual.rs`, `environment.rs`) so the GPU resolve pass and the CPU
//! reference agree numerically.
//!
//! This module is intentionally self-contained (no external imports) so it can
//! be compiled standalone in a WESL unit test and imported wherever lighting is
//! consumed.

const PI: f32 = 3.1415927;
const INV_PI: f32 = 0.31830987;
const MIN_DISTANCE_SQUARED: f32 = 1.0e-4;
const LIGHT_ENVIRONMENT_FLAG_IMAGE_BASED: u32 = 1u;

// --- SH basis constants (Ramamoorthi & Hanrahan), mirroring environment.rs ---
const A0: f32 = 3.1415927;    // pi,     l = 0
const A1: f32 = 2.0943952;    // 2*pi/3, l = 1
const A2: f32 = 0.7853982;    // pi/4,   l = 2
const K0: f32 = 0.2820948;    // 0.5 * sqrt(1/pi)
const K1: f32 = 0.4886025;    // 0.5 * sqrt(3/pi)
const K2_XY: f32 = 1.0925484; // 0.5 * sqrt(15/pi)
const K2_Z2: f32 = 0.31539157;// 0.25 * sqrt(5/pi)
const K2_X2: f32 = 0.5462742; // 0.25 * sqrt(15/pi)

struct DirectionalLight {
    // Unit vector pointing from the surface toward the emitter.
    direction_to_light: vec3<f32>,
    // Analytic shadow visibility in [0, 1].
    visibility: f32,
    // Linear, pre-exposed illuminance at the surface.
    illuminance: vec3<f32>,
    // Padding to a 16-byte boundary.
    padding: f32,
}

struct PunctualLight {
    // World-space position of the emitter.
    position: vec3<f32>,
    // Influence radius; 0 disables the range window.
    range: f32,
    // Linear, pre-exposed radiant intensity (candela).
    intensity: vec3<f32>,
    // Precomputed 1 / (cos(inner) - cos(outer)); 0 marks a point light.
    spot_scale: f32,
    // Unit cone axis pointing from the light into the scene.
    direction: vec3<f32>,
    // Precomputed -cos(outer) * spot_scale.
    spot_offset: f32,
    // Analytic shadow visibility in [0, 1].
    visibility: f32,
    // Padding to a 16-byte boundary (three scalars, matching [f32; 3]).
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

// Stylized (non-photoreal) front-end controls. Byte-for-byte mirror of
// `GpuStylizedParams` in `lighting/abi.rs`: four 16-byte rows, each `vec3`
// tint sharing its row with a trailing scalar. Consumed by `stylized_direct`
// in `brdf.wesl`.
struct StylizedParams {
    // Cel quantization band count for the diffuse ramp (>= 1).
    bands: u32,
    // Cosine wrap in [0, 1]; 0 is pure Lambert, 0.5 is half-Lambert.
    wrap: f32,
    // Band-edge softness in [0, 1]; 0 gives hard ink-line steps.
    ramp_softness: f32,
    // Stepped-shadow threshold in [0, 1].
    shadow_threshold: f32,
    // Half-width of the stylized shadow transition.
    shadow_softness: f32,
    // Stylized specular intensity; 0 disables the highlight.
    specular_intensity: f32,
    // Highlight cutoff applied to the sharpened N.H response.
    specular_threshold: f32,
    // Half-width of the highlight edge.
    specular_softness: f32,
    // Linear tint of the stylized highlight (row start).
    specular_color: vec3<f32>,
    // Rim (edge) light intensity; 0 disables the rim.
    rim_intensity: f32,
    // Linear tint of the rim light (row start).
    rim_color: vec3<f32>,
    // Fresnel exponent controlling how tightly the rim hugs the silhouette.
    rim_power: f32,
}

struct LightEnvironment {
    // Constant ambient radiance (linear).
    ambient: vec3<f32>,
    // Number of valid directional lights.
    directional_count: u32,
    // Nine L2 SH radiance coefficients; .xyz used, .w padding.
    sh: array<vec4<f32>, 9>,
    // Number of valid punctual lights.
    punctual_count: u32,
    // Reserved padding (formerly toon_bands; band count now in stylized.bands).
    reserved0: u32,
    // Reserved exposure scale (1.0 = neutral).
    exposure: f32,
    // Bit flags; see LIGHT_ENVIRONMENT_FLAG_*.
    flags: u32,
    // Stylized (NPR) front-end controls consumed by the Npr resolve arm.
    stylized: StylizedParams,
}

// The evaluated contribution of a single analytic light at a surface point.
struct LightSample {
    // Unit direction from the surface toward the light.
    direction: vec3<f32>,
    // Illuminance reaching the surface (post attenuation, pre-BSDF).
    illuminance: vec3<f32>,
    // Shadow visibility term in [0, 1].
    visibility: f32,
    // 1.0 when the light contributes, 0.0 when it should be skipped.
    valid: f32,
}

fn environment_has_image_based(env: LightEnvironment) -> bool {
    return (env.flags & LIGHT_ENVIRONMENT_FLAG_IMAGE_BASED) != 0u;
}

// Directional lights reach the surface with constant illuminance.
fn sample_directional_light(light: DirectionalLight) -> LightSample {
    var sample: LightSample;
    sample.direction = normalize(light.direction_to_light);
    sample.illuminance = light.illuminance;
    sample.visibility = clamp(light.visibility, 0.0, 1.0);
    sample.valid = 1.0;
    return sample;
}

// Punctual lights fall off with inverse square, a smooth range window, and (for
// spots) a squared cone term. Mirrors `PunctualLight::sample` in punctual.rs.
fn sample_punctual_light(light: PunctualLight, world_position: vec3<f32>) -> LightSample {
    var sample: LightSample;
    sample.valid = 0.0;

    let to_light = light.position - world_position;
    let distance_squared = dot(to_light, to_light);
    if distance_squared <= 0.0 {
        return sample;
    }
    let distance = sqrt(distance_squared);
    let direction = to_light * (1.0 / distance);

    var attenuation = 1.0 / max(distance_squared, MIN_DISTANCE_SQUARED);

    if light.range > 0.0 {
        let range_squared = light.range * light.range;
        let ratio = min(distance_squared / range_squared, 1.0);
        let window = max(1.0 - ratio * ratio, 0.0);
        attenuation = attenuation * window * window;
    }

    if light.spot_scale != 0.0 {
        let cos_angle = dot(light.direction, -direction);
        let cone = clamp(cos_angle * light.spot_scale + light.spot_offset, 0.0, 1.0);
        attenuation = attenuation * cone * cone;
    }

    if attenuation <= 0.0 {
        return sample;
    }

    sample.direction = direction;
    sample.illuminance = light.intensity * attenuation;
    sample.visibility = clamp(light.visibility, 0.0, 1.0);
    sample.valid = 1.0;
    return sample;
}

// Evaluates the nine real SH basis functions for a unit direction.
fn sh_basis(direction: vec3<f32>) -> array<f32, 9> {
    let x = direction.x;
    let y = direction.y;
    let z = direction.z;
    return array<f32, 9>(
        K0,
        K1 * y,
        K1 * z,
        K1 * x,
        K2_XY * x * y,
        K2_XY * y * z,
        K2_Z2 * (3.0 * z * z - 1.0),
        K2_XY * x * z,
        K2_X2 * (x * x - y * y),
    );
}

// Reconstructs radiance from the probe along a direction (dot of coefficients
// with the SH basis). Mirrors `SphericalHarmonicsL2::radiance`.
fn sh_radiance(env: LightEnvironment, direction: vec3<f32>) -> vec3<f32> {
    let basis = sh_basis(direction);
    var sum = vec3<f32>(0.0);
    for (var i = 0u; i < 9u; i = i + 1u) {
        sum = sum + env.sh[i].xyz * basis[i];
    }
    return sum;
}

// Convolves the probe with the clamped cosine lobe to produce irradiance at a
// surface normal. Mirrors `SphericalHarmonicsL2::irradiance`.
fn sh_irradiance(env: LightEnvironment, normal: vec3<f32>) -> vec3<f32> {
    let basis = sh_basis(normal);
    let band = array<f32, 9>(A0, A1, A1, A1, A2, A2, A2, A2, A2);
    var sum = vec3<f32>(0.0);
    for (var i = 0u; i < 9u; i = i + 1u) {
        sum = sum + env.sh[i].xyz * (band[i] * basis[i]);
    }
    return sum;
}

// Karis' analytic environment BRDF approximation. Mirrors `env_brdf_approx`.
fn env_brdf_approx(n_dot_v: f32, perceptual_roughness: f32) -> vec2<f32> {
    let nv = clamp(n_dot_v, 0.0, 1.0);
    let roughness = clamp(perceptual_roughness, 0.0, 1.0);
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let r = roughness * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * nv)) * r.x + r.y;
    return vec2<f32>(-1.04 * a004 + r.z, 1.04 * a004 + r.w);
}
