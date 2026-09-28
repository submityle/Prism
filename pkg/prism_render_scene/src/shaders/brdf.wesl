//! GPU BSDF closures mirroring the CPU golden reference in
//! `prism_render_shading` (`lighting.rs`, `environment.rs`).
//!
//! These functions are the byte-for-byte numerical twin of the backend-neutral
//! reference so the compute resolve pass agrees with the CPU golden tests
//! within tolerance. The analytic light ABI, spherical-harmonic probe helpers
//! and the environment-BRDF fit are imported from `lighting.wesl`; this module
//! adds the microfacet direct-lighting lobes (GGX principled + clearcoat), the
//! toon lobe and the split-sum image-based term.
//!
//! `pow`, `exp2`, `reflect`, `normalize` and `inverseSqrt` are shader builtins
//! and are used directly; there is no CPU `bevy_math::ops` restriction here.

import prism_render_scene::shaders::lighting::{
    PI,
    INV_PI,
    LightEnvironment,
    StylizedParams,
    sh_radiance,
    sh_irradiance,
    env_brdf_approx,
};

import prism_render_scene::shaders::tangent::{
    tangent_orthonormal_basis,
};

// Clamp floors mirroring `lighting.rs` (`MIN_ROUGHNESS`, `MIN_N_DOT`).
const MIN_ROUGHNESS: f32 = 0.045;
const MIN_N_DOT: f32 = 1.0e-5;
// Per-axis GGX roughness floor keeping the anisotropic lobe finite even when
// one axis approaches zero. Mirrors `MIN_ANISOTROPIC_ALPHA` in `lighting.rs`.
const MIN_ANISOTROPIC_ALPHA: f32 = 1.0e-3;
// Stylized specular exponent range. Mirrors `MIN_SPECULAR_EXPONENT` /
// `MAX_SPECULAR_EXPONENT` in `stylized.rs`.
const MIN_STYLIZED_SPEC_EXPONENT: f32 = 2.0;
const MAX_STYLIZED_SPEC_EXPONENT: f32 = 256.0;

// Physically based surface parameters consumed by the analytic lobes. Mirrors
// `SurfaceSample` in `lighting.rs`; emissive is carried separately so the
// resolve pass can add it exactly once.
struct SurfaceSample {
    base_color: vec3<f32>,
    metallic: f32,
    perceptual_roughness: f32,
    reflectance: f32,
    ambient_occlusion: f32,
    emissive: vec3<f32>,
    clearcoat: f32,
    clearcoat_roughness: f32,
    sheen: f32,
    subsurface: f32,
    thickness: f32,
    // Signed anisotropy in [-1, 1]; positive elongates the highlight along the
    // tangent, negative along the bitangent. Mirrors `SurfaceSample`.
    anisotropy: f32,
    // Rotation of the anisotropy axes around the normal, in radians.
    anisotropy_rotation: f32,
}

// World-space shading basis. Mirrors `ShadingFrame` in `lighting.rs`.
struct ShadingFrame {
    normal: vec3<f32>,
    view: vec3<f32>,
    // Unit tangent (major anisotropy axis) orthogonal to `normal`.
    tangent: vec3<f32>,
    // Unit bitangent carrying the frame handedness.
    bitangent: vec3<f32>,
}

// A single analytic light already reduced to a surface-relative sample.
// Mirrors `DirectLightSample` in `lighting.rs`.
struct DirectLightSample {
    direction: vec3<f32>,
    illuminance: vec3<f32>,
    visibility: f32,
}

// Normalizes `value`, falling back to `fallback` for degenerate inputs.
// Mirrors `normalize_or` in the CPU reference.
fn brdf_normalize_or(value: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let length_squared = dot(value, value);
    if length_squared > 1.0e-12 {
        return value * inverseSqrt(length_squared);
    }
    return fallback;
}

// Schlick's Fresnel approximation. Mirrors `fresnel_schlick`.
fn fresnel_schlick(f0: vec3<f32>, v_dot_h: f32) -> vec3<f32> {
    let one_minus = 1.0 - clamp(v_dot_h, 0.0, 1.0);
    let squared = one_minus * one_minus;
    let factor = squared * squared * one_minus;
    return f0 + (vec3<f32>(1.0) - f0) * factor;
}

// GGX/Trowbridge-Reitz normal distribution. Mirrors `distribution_ggx`.
fn distribution_ggx(n_dot_h: f32, alpha: f32) -> f32 {
    let alpha_squared = alpha * alpha;
    let denominator = n_dot_h * n_dot_h * (alpha_squared - 1.0) + 1.0;
    return alpha_squared / max(PI * denominator * denominator, MIN_N_DOT);
}

// Height-correlated Smith visibility term. Mirrors
// `visibility_smith_ggx_correlated`.
fn vis_smith(n_dot_v: f32, n_dot_l: f32, alpha: f32) -> f32 {
    let alpha_squared = alpha * alpha;
    let gv = n_dot_l * sqrt((n_dot_v - n_dot_v * alpha_squared) * n_dot_v + alpha_squared);
    let gl = n_dot_v * sqrt((n_dot_l - n_dot_l * alpha_squared) * n_dot_l + alpha_squared);
    return 0.5 / max(gv + gl, MIN_N_DOT);
}

// Result of re-orthonormalizing + rotating the interpolated tangent frame.
struct AnisotropicAxes {
    tangent: vec3<f32>,
    bitangent: vec3<f32>,
}

// Re-orthonormalizes the interpolated tangent frame against `normal` and
// rotates it around the normal by `rotation` radians (UE `AnisotropyRotation`).
// Mirrors `anisotropic_axes` in `lighting.rs`.
fn anisotropic_axes(
    normal: vec3<f32>,
    basis_tangent: vec3<f32>,
    basis_bitangent: vec3<f32>,
    rotation: f32,
) -> AnisotropicAxes {
    let fallback = tangent_orthonormal_basis(normal);
    let projected = basis_tangent - normal * dot(normal, basis_tangent);
    let tangent0 = brdf_normalize_or(projected, fallback.tangent);
    let handedness = select(1.0, -1.0, dot(cross(normal, tangent0), basis_bitangent) < 0.0);
    let bitangent0 = cross(normal, tangent0) * handedness;
    let cos_r = cos(rotation);
    let sin_r = sin(rotation);
    var axes: AnisotropicAxes;
    axes.tangent = tangent0 * cos_r + bitangent0 * sin_r;
    axes.bitangent = bitangent0 * cos_r - tangent0 * sin_r;
    return axes;
}

// Anisotropic GGX/Trowbridge-Reitz distribution (UE `D_GGXaniso`). Reduces to
// `distribution_ggx` when `ax == ay`. Mirrors `distribution_ggx_anisotropic`.
fn distribution_ggx_anisotropic(
    x_dot_h: f32,
    y_dot_h: f32,
    n_dot_h: f32,
    ax: f32,
    ay: f32,
) -> f32 {
    let dx = x_dot_h / ax;
    let dy = y_dot_h / ay;
    let d = dx * dx + dy * dy + n_dot_h * n_dot_h;
    return 1.0 / max(PI * ax * ay * d * d, MIN_N_DOT);
}

// Height-correlated anisotropic Smith visibility (UE `Vis_SmithJointAniso`).
// Reduces to `vis_smith` when `ax == ay`. Mirrors
// `visibility_smith_ggx_anisotropic`.
fn vis_smith_anisotropic(
    x_dot_v: f32,
    y_dot_v: f32,
    x_dot_l: f32,
    y_dot_l: f32,
    n_dot_v: f32,
    n_dot_l: f32,
    ax: f32,
    ay: f32,
) -> f32 {
    let ax_v = ax * x_dot_v;
    let ay_v = ay * y_dot_v;
    let lambda_v = n_dot_l * sqrt(ax_v * ax_v + ay_v * ay_v + n_dot_v * n_dot_v);
    let ax_l = ax * x_dot_l;
    let ay_l = ay * y_dot_l;
    let lambda_l = n_dot_v * sqrt(ax_l * ax_l + ay_l * ay_l + n_dot_l * n_dot_l);
    return 0.5 / max(lambda_v + lambda_l, MIN_N_DOT);
}

// Physically based direct lighting for one analytic light, including a
// Kulla-Conty energy-compensation term and a clearcoat lobe. Mirrors
// `evaluate_principled_direct`.
fn principled_direct(
    surface: SurfaceSample,
    frame: ShadingFrame,
    light: DirectLightSample,
) -> vec3<f32> {
    let n = brdf_normalize_or(frame.normal, vec3<f32>(0.0, 1.0, 0.0));
    let v = brdf_normalize_or(frame.view, n);
    let l = brdf_normalize_or(light.direction, n);
    let h = brdf_normalize_or(v + l, n);
    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_v = max(dot(n, v), MIN_N_DOT);
    let n_dot_h = max(dot(n, h), MIN_N_DOT);
    let v_dot_h = max(dot(v, h), 0.0);
    if n_dot_l <= 0.0 {
        return surface.emissive;
    }

    let metallic = clamp(surface.metallic, 0.0, 1.0);
    let roughness = clamp(surface.perceptual_roughness, MIN_ROUGHNESS, 1.0);
    let alpha = roughness * roughness;
    let reflectance = clamp(surface.reflectance, 0.0, 1.0);
    let f0_dielectric = 0.16 * reflectance * reflectance;
    let f0 = mix(vec3<f32>(f0_dielectric), surface.base_color, metallic);
    let f = fresnel_schlick(f0, v_dot_h);

    // Anisotropic axes: re-orthonormalize + rotate the interpolated tangent
    // frame, then split `alpha` onto the two axes via Burley's aspect ratio.
    // With `anisotropy == 0` the axes collapse to `alpha` and the anisotropic
    // D/V reduce exactly to the isotropic GGX pair.
    let axes = anisotropic_axes(n, frame.tangent, frame.bitangent, surface.anisotropy_rotation);
    let anisotropy = clamp(surface.anisotropy, -1.0, 1.0);
    let aspect = sqrt(1.0 - 0.9 * anisotropy);
    let ax = max(alpha / aspect, MIN_ANISOTROPIC_ALPHA);
    let ay = max(alpha * aspect, MIN_ANISOTROPIC_ALPHA);
    let d = distribution_ggx_anisotropic(dot(axes.tangent, h), dot(axes.bitangent, h), n_dot_h, ax, ay);
    let g = vis_smith_anisotropic(
        dot(axes.tangent, v),
        dot(axes.bitangent, v),
        dot(axes.tangent, l),
        dot(axes.bitangent, l),
        n_dot_v,
        n_dot_l,
        ax,
        ay,
    );
    let single_scatter = f * (d * g);

    // Kulla-Conty-style bounded compensation returning energy lost to
    // unresolved multiple scattering while preserving the primary lobe.
    let energy = 1.0 - 0.28 * roughness * roughness;
    let compensation = f0 * min((1.0 / max(energy, 0.25)) - 1.0, 1.0);
    let specular = single_scatter + compensation;

    let diffuse_weight = (vec3<f32>(1.0) - f) * (1.0 - metallic);
    let diffuse = (surface.base_color * INV_PI) * diffuse_weight;
    let base = (diffuse + specular) * n_dot_l;

    // Clearcoat: a second, always-dielectric GGX lobe that attenuates the base
    // response by its own Fresnel reflectance.
    let coat = clamp(surface.clearcoat, 0.0, 1.0);
    let coat_roughness = clamp(surface.clearcoat_roughness, MIN_ROUGHNESS, 1.0);
    let coat_alpha = coat_roughness * coat_roughness;
    let coat_f = fresnel_schlick(vec3<f32>(0.04), v_dot_h);
    let coat_specular = coat_f
        * (coat
            * distribution_ggx(n_dot_h, coat_alpha)
            * vis_smith(n_dot_v, n_dot_l, coat_alpha)
            * n_dot_l);
    let coat_attenuation = 1.0 - coat * clamp(coat_f.x, 0.0, 1.0);
    let reflected = base * coat_attenuation + coat_specular;

    let radiance = reflected * light.illuminance;
    return radiance * clamp(light.visibility, 0.0, 1.0) + surface.emissive;
}

// Non-photoreal quantized diffuse lobe. Mirrors `evaluate_toon_direct`.
fn toon_direct(
    surface: SurfaceSample,
    frame: ShadingFrame,
    light: DirectLightSample,
    bands: u32,
) -> vec3<f32> {
    let n = brdf_normalize_or(frame.normal, vec3<f32>(0.0, 1.0, 0.0));
    let l = brdf_normalize_or(light.direction, n);
    let steps = f32(max(bands, 1u));
    let diffuse = floor(max(dot(n, l), 0.0) * steps) / steps;
    return (surface.base_color * light.illuminance) * (diffuse * clamp(light.visibility, 0.0, 1.0))
        + surface.emissive;
}

// Scalar lerp matching the CPU `mix` helper (a + (b - a) * t) exactly, rather
// than the WGSL builtin (e1 * (1 - e3) + e2 * e3), so the stylized front end
// stays a bit-faithful twin of `stylized.rs`.
fn stylized_mix(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * t;
}

// Hermite smoothstep matching `stylized.rs`: a collapsed or inverted edge pair
// degrades to a hard step at `edge0` (the WGSL builtin is undefined there).
fn stylized_smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if edge1 <= edge0 {
        return select(1.0, 0.0, x < edge0);
    }
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// Re-shapes the cosine term into an optionally wrapped, quantized cel ramp.
// Mirrors `stylized_ramp` in `stylized.rs`.
fn stylized_ramp(n_dot_l: f32, bands: u32, wrap_in: f32, softness: f32) -> f32 {
    let steps = f32(max(bands, 1u));
    let wrap = clamp(wrap_in, 0.0, 1.0);
    var lit: f32;
    if wrap > 0.0 {
        lit = clamp((n_dot_l + wrap) / (1.0 + wrap), 0.0, 1.0);
    } else {
        lit = max(n_dot_l, 0.0);
    }
    let scaled = lit * steps;
    let level = floor(scaled);
    let lower = level / steps;
    if softness <= 0.0 {
        return lower;
    }
    let frac = scaled - level;
    let upper = min((level + 1.0) / steps, 1.0);
    let half_window = clamp(softness * 0.5, 0.0, 0.5);
    let t = stylized_smoothstep(0.5 - half_window, 0.5 + half_window, frac);
    return stylized_mix(lower, upper, t);
}

// Re-shapes the analytic shadow visibility into a stylized step. Mirrors
// `stylized_shadow` in `stylized.rs`.
fn stylized_shadow(visibility: f32, threshold: f32, softness: f32) -> f32 {
    let v = clamp(visibility, 0.0, 1.0);
    if softness <= 0.0 {
        if threshold <= 0.0 {
            return v;
        }
        return select(0.0, 1.0, v >= threshold);
    }
    return stylized_smoothstep(threshold - softness, threshold + softness, v);
}

// Stylized (non-photoreal / NPR) front end for a single analytic light: cel
// ramp diffuse + stepped shadow + thresholded specular blob + view Fresnel rim.
// Byte-for-byte twin of `evaluate_stylized_direct` in `stylized.rs`; with the
// default (legacy) params it reproduces `toon_direct` exactly.
fn stylized_direct(
    surface: SurfaceSample,
    frame: ShadingFrame,
    light: DirectLightSample,
    params: StylizedParams,
) -> vec3<f32> {
    let n = brdf_normalize_or(frame.normal, vec3<f32>(0.0, 1.0, 0.0));
    let v = brdf_normalize_or(frame.view, n);
    let l = brdf_normalize_or(light.direction, n);

    let n_dot_l = dot(n, l);
    let lit_gate = max(n_dot_l, 0.0);
    let shadow = stylized_shadow(light.visibility, params.shadow_threshold, params.shadow_softness);

    let ramp = stylized_ramp(n_dot_l, params.bands, params.wrap, params.ramp_softness);
    var color = (surface.base_color * light.illuminance) * (ramp * shadow);

    if params.specular_intensity > 0.0 && lit_gate > 0.0 {
        let h = brdf_normalize_or(v + l, n);
        let n_dot_h = max(dot(n, h), 0.0);
        let roughness = clamp(surface.perceptual_roughness, MIN_ROUGHNESS, 1.0);
        let gloss = 1.0 - roughness;
        let exponent = stylized_mix(MIN_STYLIZED_SPEC_EXPONENT, MAX_STYLIZED_SPEC_EXPONENT, gloss * gloss);
        let raw = pow(n_dot_h, exponent);
        let half_window = clamp(params.specular_softness * 0.5, 0.0, 0.5);
        let blob = stylized_smoothstep(
            params.specular_threshold - half_window,
            params.specular_threshold + half_window,
            raw,
        );
        let weight = params.specular_intensity * blob * lit_gate * shadow;
        color = color + (params.specular_color * weight) * light.illuminance;
    }

    if params.rim_intensity > 0.0 && lit_gate > 0.0 {
        let n_dot_v = max(dot(n, v), 0.0);
        let rim = pow(max(1.0 - n_dot_v, 0.0), max(params.rim_power, 0.0));
        let weight = params.rim_intensity * rim * lit_gate;
        color = color + (params.rim_color * weight) * light.illuminance;
    }

    return color + surface.emissive;
}

// --- SDF face shadow (stylized NPR) -----------------------------------------
// Backend twin of `face_shadow.rs`: drives an anime face terminator from a
// baked SDF map plus the light's azimuth in the head frame, instead of the
// geometric N.L / shadow map. Returns a [0,1] visibility that feeds the
// `visibility` input of `stylized_direct` (through `stylized_shadow`).

// Projects `light_dir` onto the head's horizontal plane (removing the `up`
// component so head tilt is handled) and returns (forward_cos, right_cos).
// Mirrors `face_shadow_light_cosines`.
fn face_shadow_light_cosines(
    face_forward: vec3<f32>,
    face_right: vec3<f32>,
    face_up: vec3<f32>,
    light_dir: vec3<f32>,
) -> vec2<f32> {
    let up = brdf_normalize_or(face_up, vec3<f32>(0.0, 1.0, 0.0));
    let l = brdf_normalize_or(light_dir, face_forward);
    let planar = l - up * dot(l, up);
    let forward = brdf_normalize_or(face_forward, vec3<f32>(0.0, 0.0, 1.0));
    let l_h = brdf_normalize_or(planar, forward);
    let right = brdf_normalize_or(face_right, vec3<f32>(1.0, 0.0, 0.0));
    return vec2<f32>(dot(l_h, forward), dot(l_h, right));
}

// UV-mirror decision: the map is authored for the light on the character's
// right (right_cos >= 0) and mirrored across u for the left. Mirrors
// `face_shadow_flip_u`.
fn face_shadow_flip_u(right_cos: f32) -> bool {
    return right_cos < 0.0;
}

// Turns the resolved SDF sample + forward cosine into a [0,1] visibility.
// Threshold is (1 - forward_cos)/2; `stylized_smoothstep` shares the collapsed
// edge semantics of the CPU reference. Mirrors `evaluate_face_shadow`.
fn evaluate_face_shadow(sdf: f32, forward_cos: f32, softness: f32) -> f32 {
    let threshold = (1.0 - clamp(forward_cos, -1.0, 1.0)) * 0.5;
    let sample = clamp(sdf, 0.0, 1.0);
    let half_window = clamp(softness, 0.0, 0.5);
    return stylized_smoothstep(threshold - half_window, threshold + half_window, sample);
}

// Split-sum image-based indirect term. Mirrors `evaluate_image_based_light`.
// The GPU probe carries no independent intensity, so intensity is 1.0. Probe
// reconstruction can ring negative, so both halves clamp to non-negative before
// weighting, matching the CPU `.max(0.0)` guards.
fn image_based(
    surface: SurfaceSample,
    frame: ShadingFrame,
    env: LightEnvironment,
) -> vec3<f32> {
    let normal = brdf_normalize_or(frame.normal, vec3<f32>(0.0, 1.0, 0.0));
    let view = brdf_normalize_or(frame.view, normal);
    let n_dot_v = max(dot(normal, view), 1.0e-4);

    let metallic = clamp(surface.metallic, 0.0, 1.0);
    let roughness = clamp(surface.perceptual_roughness, 0.0, 1.0);
    let occlusion = clamp(surface.ambient_occlusion, 0.0, 1.0);
    let reflectance = clamp(surface.reflectance, 0.0, 1.0);
    let f0_dielectric = 0.16 * reflectance * reflectance;
    let f0 = mix(vec3<f32>(f0_dielectric), surface.base_color, metallic);

    // Diffuse: Lambertian albedo lit by cosine-convolved irradiance.
    let irradiance = max(sh_irradiance(env, normal), vec3<f32>(0.0));
    let diffuse_weight = 1.0 - metallic;
    let diffuse = surface.base_color * diffuse_weight * irradiance * INV_PI;

    // Specular: prefiltered radiance along the reflection vector weighted by
    // the analytic environment BRDF.
    let reflection = reflect(-view, normal);
    let prefiltered = max(sh_radiance(env, reflection), vec3<f32>(0.0));
    let dfg = env_brdf_approx(n_dot_v, roughness);
    let specular = prefiltered * (f0 * dfg.x + dfg.y);

    return (diffuse + specular) * occlusion;
}
