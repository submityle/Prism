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

// Specular anti-aliasing kernels: GPU twins of `gi/specular_aa/{normal_variance,toksvig}.rs`.
import prism_render_scene::shaders::specular_aa::{
    spec_aa_filter_alpha_sq,
    SPEC_AA_KAPPA_MAX,
};
// Thin-film iridescence: GPU twin of `gi/material/thin_film.rs`.
import prism_render_scene::shaders::thin_film::{
    tf_iridescent_fresnel_rgb,
};
// Procedural glint: GPU twin of the discrete-microfacet kernel in
// `glint.rs` (§17.6).
import prism_render_scene::shaders::glint::{
    glint_bin_solid_angle,
    glint_half_slope,
    glint_microfacet_count,
    glint_ndf_from_density,
    glint_seed,
    GLINT_MIN_ALPHA,
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
    // `OpenPBR` `base_weight` (scales the base layer) and `diffuse_roughness`
    // (Oren-Nayar micro-slope roughness; 0 is exactly Lambert). Mirrors
    // `SurfaceSample::base_weight` / `diffuse_roughness` in `lighting.rs`.
    base_weight: f32,
    diffuse_roughness: f32,
    metallic: f32,
    perceptual_roughness: f32,
    // `OpenPBR` specular group. `specular_weight` scales the dielectric lobe,
    // `specular_color` tints it and `specular_ior` sets its interface index
    // (`1.5` = the historical `F0 = 0.04`). Mirrors `SurfaceSample` in
    // `lighting.rs`; the former `reflectance` IOR proxy is gone.
    specular_weight: f32,
    specular_color: vec3<f32>,
    specular_ior: f32,
    // `OpenPBR` transmission lobe: weight, refracting index and chromatic
    // dispersion. Mirrors `SurfaceSample` in `lighting.rs`; `transmission == 0`
    // is an exact no-op.
    transmission: f32,
    index_of_refraction: f32,
    dispersion: f32,
    // `OpenPBR` `geometry_thin_walled`: skip the transmission volume absorption.
    thin_walled: f32,
    // `OpenPBR` `transmission_color`: tint of the refracted light (white =
    // identity).
    transmission_color: vec3<f32>,
    // `OpenPBR` `transmission_scatter` (color3): per-channel re-scattered
    // fraction of the absorbed slab energy (black = pure Beer-Lambert).
    transmission_scatter: vec3<f32>,
    // `OpenPBR` `transmission_scatter_anisotropy` in [-1, 1] and
    // `transmission_dispersion_abbe_number` (V_d).
    transmission_scatter_anisotropy: f32,
    transmission_dispersion_abbe_number: f32,
    // `OpenPBR` `coat_darkening`: internal-reflection loss that darkens the base.
    coat_darkening: f32,
    // `OpenPBR` `scatter_anisotropy`: forward/backward bias of the subsurface
    // transmitted lobe (0 = the historical constant sharpness).
    scatter_anisotropy: f32,
    ambient_occlusion: f32,
    emissive: vec3<f32>,
    clearcoat: f32,
    clearcoat_roughness: f32,
    // OpenPBR `coat_ior` (relative coat index) and `coat_color` (highlight
    // tint). Mirrors `SurfaceSample::coat_ior` / `coat_color` in `lighting.rs`.
    coat_ior: f32,
    coat_color: vec3<f32>,
    // World-space coat interface normal (`OpenPBR` `coat_normal`); the zero
    // vector means "use the base shading normal". Mirrors
    // `SurfaceSample::coat_normal` in `lighting.rs`.
    coat_normal: vec3<f32>,
    // `OpenPBR` `coat_roughness_anisotropy` in [0, 1]; 0 is the isotropic
    // identity. Mirrors `SurfaceSample::coat_roughness_anisotropy`.
    coat_roughness_anisotropy: f32,
    // `OpenPBR` `geometry_coat_tangent`; the zero vector inherits the base
    // frame tangent. Mirrors `SurfaceSample::coat_tangent`.
    coat_tangent: vec3<f32>,
    sheen: f32,
    // `OpenPBR` `fuzz_roughness` in [0, 1]: the fuzz lobe's own highlight width.
    fuzz_roughness: f32,
    // `OpenPBR` `fuzz_color`: per-channel tint of the fuzz lobe (white =
    // achromatic). Mirrors `SurfaceSample::fuzz_roughness` / `fuzz_color`.
    fuzz_color: vec3<f32>,
    subsurface: f32,
    // `OpenPBR` subsurface group: scattered-light tint (white = identity) and
    // per-channel scattering extent (white = identity, relative in this lobe).
    subsurface_color: vec3<f32>,
    // `OpenPBR` `subsurface_radius` (scalar, relative here) and
    // `subsurface_radius_scale` (color3 per-channel split).
    subsurface_radius: f32,
    subsurface_radius_scale: vec3<f32>,
    thickness: f32,
    // `OpenPBR` `specular_roughness_anisotropy` in [0, 1]: stretches the
    // highlight along the (rotated) tangent; 0 is exactly isotropic. Mirrors
    // `SurfaceSample::anisotropy` in `lighting.rs`.
    anisotropy: f32,
    // Rotation of the anisotropy axes around the normal, in radians.
    anisotropy_rotation: f32,
    // Specular-AA normal variance in the GGX `alpha^2` domain (`sigma^2`),
    // derived shade-time from the sub-pixel / texel footprint of the shading
    // normal. NOT part of the GPU material ABI; `0` is an exact identity.
    // Mirrors `SurfaceSample::normal_variance` in `lighting.rs`.
    normal_variance: f32,
    // Thin-film iridescence weight in [0, 1] blending the specular Fresnel
    // toward the Airy interference reflectance (glTF KHR_materials_iridescence).
    // Shade-time, NOT part of the GPU material ABI; `0` is an exact identity.
    // Mirrors `SurfaceSample::iridescence` in `lighting.rs`.
    iridescence: f32,
    // Film thickness in nm driving the Airy phase; inert at weight 0.
    iridescence_thickness_nm: f32,
    // Film refractive index (glTF default 1.3); inert at weight 0.
    iridescence_ior: f32,
    // Prism glint lobe (§17.6): blends a discrete (counting) microfacet NDF
    // into the specular lobe. `glint_weight == 0` is an exact identity — the
    // estimator is never entered. Mirrors `SurfaceSample::glint_*`.
    glint_weight: f32,
    // Flake orientation spread (see `brdf_glint_shade_seed`).
    glint_roughness: f32,
    // Facets per unit² of the shade-time surface cell, and the angular
    // half-width of the half-vector acceptance bin (radians).
    glint_density: f32,
    glint_bin_half_width: f32,
    // Glint *mixture state* for the single-lobe sampler (§17.4, P3-e): `< 0`
    // means "use the authored weight" (the exact blend), `[0, 1]` is the state
    // this pixel drew. Sampling the state rather than scaling a signed delta by
    // `1/p` keeps every evaluation a genuine non-negative NDF. Mirrors
    // `SurfaceSample::glint_state`.
    glint_state: f32,
    // Shade-time surface cell the seed quantises, and the area of one pixel in
    // that space (|det dUV/dpixel|).
    glint_cell: vec2<f32>,
    glint_footprint_area: f32,
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

// `OpenPBR` specular group dielectric `F0`: the per-channel normal-incidence
// Fresnel `((ior-1)/(ior+1))^2`, scaled by `specular_weight` and tinted by
// `specular_color`. Byte-for-byte twin of `lighting.rs::dielectric_f0`; the
// defaults (ior 1.5, weight 1, white) give `vec3(0.04)`, the value the removed
// `reflectance = 0.5` proxy produced.
fn dielectric_f0(surface: SurfaceSample) -> vec3<f32> {
    let ior = clamp(surface.specular_ior, 1.0, 3.0);
    let r = (ior - 1.0) / (ior + 1.0);
    let f0 = r * r * clamp(surface.specular_weight, 0.0, 1.0);
    return vec3<f32>(f0) * surface.specular_color;
}

// `OpenPBR` equation `modulated_ior`: the index ratio `η' = (1 + ε)/(1 − ε)`
// with `ε = sgn(η − 1)·sqrt(ξ_s·F_s)`. `ξ_s == 0` removes the interface
// (`η' == 1`, i.e. `F ≡ 0` at every angle — not a grazing rim); `ξ_s == 1`
// returns the authored index. Byte-for-byte twin of `lighting.rs::modulated_ior`.
fn brdf_modulated_ior(eta: f32, specular_weight: f32) -> f32 {
    let w = max(specular_weight, 0.0);
    if w == 0.0 {
        return 1.0;
    }
    let r = (1.0 - eta) / (1.0 + eta);
    let f_s = r * r;
    let sign = select(-1.0, 1.0, eta >= 1.0);
    let epsilon = clamp(sign * sqrt(w * f_s), -(1.0 - 1.0e-4), 1.0 - 1.0e-4);
    let one_minus = 1.0 - epsilon;
    let eta_prime = (1.0 + epsilon) / one_minus;
    if !(eta_prime > 0.0) || !(abs(eta_prime) < 1.0e30) {
        return BRDF_MIN_DIELECTRIC_IOR;
    }
    return eta_prime;
}

// The `OpenPBR` dielectric base specular Fresnel factor: the exact angle-
// dependent Fresnel at the modulated index, tinted by `specular_color`.
// Byte-for-byte twin of `lighting.rs::base_dielectric_fresnel_rgb`. `η' == 1` is
// a non-interface, so the guard returns zero rather than the spurious grazing
// reflection `fresnel_dielectric` would report for a clamped index.
fn brdf_base_dielectric_fresnel_rgb(
    cos_i: f32,
    specular_ior: f32,
    specular_weight: f32,
    specular_color: vec3<f32>,
) -> vec3<f32> {
    let eta = brdf_modulated_ior(clamp(specular_ior, 1.0, 3.0), clamp(specular_weight, 0.0, 1.0));
    var f = 0.0;
    if abs(eta - 1.0) > 1.0e-7 {
        f = fresnel_dielectric(cos_i, eta);
    }
    return vec3<f32>(f) * specular_color;
}

// `OpenPBR` §Metal "F82-tint" conductor Fresnel (Kutz 2021 / Hoffman 2019):
//   F0 = base_weight·base_color
//   F_Schlick(μ) = F0 + (1 − F0)(1 − μ)^5
//   F(μ̄) = specular_color·F_Schlick(μ̄)                     with μ̄ = 1/7
//   F82(μ) = F_Schlick(μ) − μ(1−μ)^6/(μ̄(1−μ̄)^6)·(F_Schlick(μ̄) − F(μ̄))
//   F_metal(μ) = specular_weight · F82(μ)
// White `specular_color` at `specular_weight == 1` collapses the correction
// term to plain Schlick. Byte-for-byte twin of
// `lighting.rs::base_metal_fresnel_rgb`.
const BRDF_METAL_MU_BAR: f32 = 1.0 / 7.0;
fn brdf_base_metal_fresnel_rgb(
    cos_i: f32,
    base_weight: f32,
    base_color: vec3<f32>,
    specular_weight: f32,
    specular_color: vec3<f32>,
) -> vec3<f32> {
    let bw = clamp(base_weight, 0.0, 1.0);
    let f0 = vec3<f32>(bw) * base_color;
    let f_bar_curve = fresnel_schlick(f0, BRDF_METAL_MU_BAR);
    let om_bar = 1.0 - BRDF_METAL_MU_BAR;
    let om_bar2 = om_bar * om_bar;
    let denom = BRDF_METAL_MU_BAR * om_bar2 * om_bar2 * om_bar2;
    let mu = clamp(cos_i, 0.0, 1.0);
    let om = 1.0 - mu;
    let om2 = om * om;
    let correction = mu * om2 * om2 * om2 / denom;
    let curve = fresnel_schlick(f0, mu);
    let f_edge = clamp(specular_color, vec3<f32>(0.0), vec3<f32>(1.0)) * f_bar_curve;
    return clamp(specular_weight, 0.0, 1.0) * (curve - correction * (f_bar_curve - f_edge));
}

// Unpolarized dielectric Fresnel reflectance for a direction whose normal
// cosine is `cos_i`, entering from index 1 into relative index `ior`. Exact twin
// of `lighting.rs::fresnel_dielectric` and the path-traced reference
// `reference_pt::dielectric::fresnel_dielectric(cos_i, 1.0, ior)` (including the
// total-internal-reflection branch). At `cos_i == 1`, `ior == 1.5` it is `0.04`.
const BRDF_MIN_DIELECTRIC_IOR: f32 = 1.0 + 1.0e-3;
// Relative per-channel index spread at `dispersion == 1`; mirrors
// `lighting.rs::TRANSMISSION_DISPERSION_SPREAD`.
const TRANSMISSION_DISPERSION_SPREAD: f32 = 0.05;
fn fresnel_dielectric(cos_i: f32, ior: f32) -> f32 {
    let c = clamp(cos_i, 0.0, 1.0);
    let eta_t = max(ior, BRDF_MIN_DIELECTRIC_IOR);
    let eta = 1.0 / eta_t;
    let sin2_i = max(1.0 - c * c, 0.0);
    let sin2_t = eta * eta * sin2_i;
    if sin2_t >= 1.0 {
        return 1.0;
    }
    let cos_t = sqrt(max(1.0 - sin2_t, 0.0));
    let r_parl = (eta_t * c - cos_t) / (eta_t * c + cos_t);
    let r_perp = (c - eta_t * cos_t) / (c + eta_t * cos_t);
    return 0.5 * (r_parl * r_parl + r_perp * r_perp);
}

// Beer-Lambert transmittance `color^{thickness}` per channel; mirrors
// `gi::translucency::transmission::{transmittance_from_color, channel_pow}` with
// a unit reference thickness.
fn transmittance_pow(color: f32, thickness: f32) -> f32 {
    let c = clamp(color, 0.0, 1.0);
    let e = max(thickness, 0.0);
    if c <= 0.0 {
        if e <= 0.0 { return 1.0; }
        return 0.0;
    }
    return clamp(pow(c, e), 0.0, 1.0);
}

// `OpenPBR` dispersion: the Abbe `d`-line and `F`/`C` reference wavelengths (µm)
// and the Prism RGB primaries (nm). Mirrors `lighting.rs` and `thin_film.wesl`.
const ABBE_LAMBDA_D_UM: f32 = 0.58756;
const ABBE_LAMBDA_F_UM: f32 = 0.48613;
const ABBE_LAMBDA_C_UM: f32 = 0.65627;
const ABBE_MIN_NUMBER: f32 = 1.0;
const ABBE_PRIMARY_R_NM: f32 = 630.0;
const ABBE_PRIMARY_G_NM: f32 = 532.0;
const ABBE_PRIMARY_B_NM: f32 = 465.0;

// Relative refractive offset about the `d`-line in units of `(n_d-1)/V_d`,
// from `V_d = (n_d-1)/(n_F-n_C)` with linear dispersion in `1/lambda`.
fn abbe_relative_offset(wavelength_nm: f32) -> f32 {
    let inv = 1.0 / (wavelength_nm * 1.0e-3);
    return (inv - 1.0 / ABBE_LAMBDA_D_UM)
        / (1.0 / ABBE_LAMBDA_F_UM - 1.0 / ABBE_LAMBDA_C_UM);
}

// Per-channel refractive index of a dispersive dielectric; `scale == 0` (the
// `transmission_dispersion_scale` default) reproduces `n_d` exactly.
fn dispersive_ior(n_d: f32, abbe_number: f32, scale: f32, channel: u32) -> f32 {
    let abbe = clamp(abbe_number, ABBE_MIN_NUMBER, 200.0);
    let delta = (n_d - 1.0) / abbe * clamp(scale, 0.0, 1.0);
    var wavelength = ABBE_PRIMARY_R_NM;
    if channel == 1u {
        wavelength = ABBE_PRIMARY_G_NM;
    } else if channel >= 2u {
        wavelength = ABBE_PRIMARY_B_NM;
    }
    return clamp(n_d + delta * abbe_relative_offset(wavelength), 1.0, 3.0);
}

// Henyey-Greenstein phase function `(1-g^2) / (4pi (1+g^2-2g cos)^1.5)`.
// Mirrors `gi::translucency::foliage::forward_phase`; `g` is clamped just inside
// (-1, 1) so the lobe stays finite.
fn vc_forward_phase(cos_theta: f32, g: f32) -> f32 {
    let c = clamp(cos_theta, -1.0, 1.0);
    let gg = clamp(g, -0.999, 0.999);
    let denom = max(1.0 + gg * gg - 2.0 * gg * c, 1.0e-12);
    // `INV_4PI` = INV_PI / 4, kept consistent with the lighting constants.
    let value = (1.0 - gg * gg) * (INV_PI * 0.25) / pow(denom, 1.5);
    if value >= 0.0 {
        return value;
    }
    return INV_PI * 0.25;
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

// `E(µ_i, r_j)`, row-major over roughness (`µ_i = i/16`, `r_j = j/8`).
const GGX_N_MU: i32 = 17;
const GGX_N_R: i32 = 9;
const GGX_ALBEDO: array<f32, 153> = array<f32, 153>(
    1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
    0.964381, 0.964381, 0.990634, 0.996070, 0.997740, 0.998607, 0.999081, 0.999299, 0.999487, 0.999554, 0.999628, 0.999718, 0.999756, 0.999778, 0.999793, 0.999830, 0.999844,
    0.888715, 0.888715, 0.910110, 0.941431, 0.961581, 0.973538, 0.980757, 0.985280, 0.988275, 0.990371, 0.991976, 0.993108, 0.993935, 0.994528, 0.995015, 0.995406, 0.995791,
    0.911760, 0.911760, 0.877837, 0.876056, 0.888036, 0.903585, 0.918350, 0.930953, 0.941175, 0.949317, 0.955872, 0.961098, 0.965282, 0.968609, 0.971358, 0.973665, 0.975570,
    0.921149, 0.921149, 0.876500, 0.851384, 0.840079, 0.838098, 0.841831, 0.848806, 0.857358, 0.866410, 0.875333, 0.883814, 0.891623, 0.898722, 0.905101, 0.910799, 0.915882,
    0.911364, 0.911364, 0.860054, 0.824380, 0.799449, 0.782489, 0.771666, 0.765553, 0.763024, 0.763164, 0.765273, 0.768800, 0.773327, 0.778502, 0.784061, 0.789830, 0.795653,
    0.888982, 0.888982, 0.825800, 0.779841, 0.744536, 0.716752, 0.694690, 0.677148, 0.663252, 0.652352, 0.643936, 0.637587, 0.632968, 0.629821, 0.627890, 0.626996, 0.626978,
    0.858492, 0.858492, 0.779168, 0.720858, 0.674986, 0.637531, 0.606256, 0.579731, 0.556964, 0.537255, 0.520047, 0.504953, 0.491672, 0.479948, 0.469558, 0.460338, 0.452170,
    0.823009, 0.823009, 0.725394, 0.653967, 0.597712, 0.551612, 0.512862, 0.479666, 0.450817, 0.425447, 0.402928, 0.382790, 0.364667, 0.348231, 0.333275, 0.319598, 0.307007,
);

// `E_avg(r_j)`.
const GGX_AVERAGE: array<f32, 9> = array<f32, 9>(
    0.999995, 0.998962, 0.987304, 0.951403, 0.882399, 0.780349, 0.656290, 0.527348, 0.409282,
);

const BIAS_N_MU: i32 = 17;
const BIAS_N_R: i32 = 9;
const SCHLICK_BIAS: array<f32, 153> = array<f32, 153>(
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.774244, 0.663815, 0.495332, 0.347318, 0.234679, 0.152748, 0.095450, 0.056640, 0.031595, 0.016307, 0.007606, 0.003092, 0.001032, 0.000253, 0.000036, 0.000002, 0.000000,
    0.508795, 0.431615, 0.362063, 0.281038, 0.203548, 0.139449, 0.090800, 0.056082, 0.032642, 0.017708, 0.008802, 0.003916, 0.001500, 0.000479, 0.000124, 0.000025, 0.000003,
    0.313340, 0.263798, 0.218489, 0.176234, 0.136619, 0.101114, 0.071250, 0.047687, 0.030196, 0.017975, 0.009964, 0.005073, 0.002328, 0.000937, 0.000319, 0.000085, 0.000012,
    0.192295, 0.157866, 0.128437, 0.102946, 0.080828, 0.061808, 0.045781, 0.032675, 0.022347, 0.014553, 0.008951, 0.005144, 0.002719, 0.001290, 0.000527, 0.000168, 0.000027,
    0.122265, 0.096436, 0.076552, 0.060369, 0.047053, 0.036096, 0.027142, 0.019915, 0.014187, 0.009753, 0.006421, 0.004007, 0.002336, 0.001243, 0.000579, 0.000213, 0.000042,
    0.082617, 0.061668, 0.047356, 0.036463, 0.027948, 0.021230, 0.015922, 0.011744, 0.008483, 0.005970, 0.004066, 0.002656, 0.001641, 0.000940, 0.000479, 0.000199, 0.000048,
    0.060034, 0.041666, 0.030684, 0.022905, 0.017144, 0.012793, 0.009478, 0.006945, 0.005013, 0.003548, 0.002448, 0.001633, 0.001041, 0.000622, 0.000336, 0.000152, 0.000043,
    0.046881, 0.029659, 0.020799, 0.014960, 0.010876, 0.007932, 0.005773, 0.004175, 0.002989, 0.002108, 0.001457, 0.000979, 0.000633, 0.000387, 0.000217, 0.000104, 0.000034,
);

// Bilinear `B(µ, r)`; arguments clamp to `[0, 1]`.
fn schlick_bias_lookup(n_dot_v: f32, roughness: f32) -> f32 {
    let x = clamp(n_dot_v, 0.0, 1.0) * f32(BIAS_N_MU - 1);
    let y = clamp(roughness, 0.0, 1.0) * f32(BIAS_N_R - 1);
    let i0 = min(i32(x), BIAS_N_MU - 2);
    let j0 = min(i32(y), BIAS_N_R - 2);
    let fx = x - f32(i0);
    let fy = y - f32(j0);
    let a = SCHLICK_BIAS[j0 * BIAS_N_MU + i0];
    let b = SCHLICK_BIAS[j0 * BIAS_N_MU + i0 + 1];
    let c = SCHLICK_BIAS[(j0 + 1) * BIAS_N_MU + i0];
    let d = SCHLICK_BIAS[(j0 + 1) * BIAS_N_MU + i0 + 1];
    let top = a + (b - a) * fx;
    let bottom = c + (d - c) * fx;
    return clamp(top + (bottom - top) * fy, 0.0, 1.0);
}

// Bilinear `E(µ, r)`; both arguments clamp to `[0, 1]`. Mirrors
// `ggx_albedo_table.rs::ggx_albedo_lookup`.
fn ggx_albedo_lookup(n_dot_v: f32, roughness: f32) -> f32 {
    let x = clamp(n_dot_v, 0.0, 1.0) * f32(GGX_N_MU - 1);
    let y = clamp(roughness, 0.0, 1.0) * f32(GGX_N_R - 1);
    let i0 = min(i32(x), GGX_N_MU - 2);
    let j0 = min(i32(y), GGX_N_R - 2);
    let fx = x - f32(i0);
    let fy = y - f32(j0);
    let a = GGX_ALBEDO[j0 * GGX_N_MU + i0];
    let b = GGX_ALBEDO[j0 * GGX_N_MU + i0 + 1];
    let c = GGX_ALBEDO[(j0 + 1) * GGX_N_MU + i0];
    let d = GGX_ALBEDO[(j0 + 1) * GGX_N_MU + i0 + 1];
    let top = a + (b - a) * fx;
    let bottom = c + (d - c) * fx;
    return clamp(top + (bottom - top) * fy, 0.0, 1.0);
}

// Linear `E_avg(r)`; clamped to `[0, 1]`.
fn ggx_average_lookup(roughness: f32) -> f32 {
    let y = clamp(roughness, 0.0, 1.0) * f32(GGX_N_R - 1);
    let j0 = min(i32(y), GGX_N_R - 2);
    let fy = y - f32(j0);
    let a = GGX_AVERAGE[j0];
    let b = GGX_AVERAGE[j0 + 1];
    return clamp(a + (b - a) * fy, 0.0, 1.0);
}

// Lower bound on `1 - E_avg` keeping the `f_ms` denominator finite as the
// single-scattering lobe approaches a mirror. Mirrors `MIN_ENERGY_DEFICIT`.
const MS_MIN_ENERGY_DEFICIT: f32 = 1.0e-4;

const DIEL_N_MU: i32 = 17;
const DIEL_N_R: i32 = 9;
const DIELECTRIC_ALBEDO: array<f32, 153> = array<f32, 153>(
    0.991738, 0.701614, 0.500000, 0.361128, 0.264190, 0.195931, 0.147630, 0.113399, 0.089187, 0.072157, 0.060302, 0.052187, 0.046781, 0.043333, 0.041301, 0.040287, 0.040000,
    0.764850, 0.644223, 0.484574, 0.355432, 0.261916, 0.195040, 0.147497, 0.113488, 0.089340, 0.072315, 0.060440, 0.052298, 0.046862, 0.043389, 0.041335, 0.040302, 0.040001,
    0.509576, 0.428823, 0.365235, 0.296800, 0.233476, 0.181438, 0.141125, 0.110840, 0.088530, 0.072355, 0.060806, 0.052732, 0.047223, 0.043644, 0.041473, 0.040305, 0.039882,
    0.331714, 0.281623, 0.239760, 0.203679, 0.171336, 0.142631, 0.118013, 0.097659, 0.081365, 0.068684, 0.059069, 0.051971, 0.046890, 0.043397, 0.041137, 0.039818, 0.039220,
    0.223957, 0.189272, 0.161310, 0.138173, 0.118762, 0.102329, 0.088397, 0.076662, 0.066899, 0.058912, 0.052510, 0.047500, 0.043691, 0.040902, 0.038963, 0.037724, 0.037051,
    0.161297, 0.133988, 0.113894, 0.097904, 0.084905, 0.074226, 0.065410, 0.058119, 0.052102, 0.047164, 0.043150, 0.039931, 0.037400, 0.035462, 0.034037, 0.033049, 0.032436,
    0.125001, 0.100881, 0.084903, 0.072798, 0.063266, 0.055614, 0.049403, 0.044328, 0.040168, 0.036757, 0.033965, 0.031692, 0.029856, 0.028393, 0.027246, 0.026373, 0.025733,
    0.103650, 0.080339, 0.066429, 0.056394, 0.048741, 0.042727, 0.037914, 0.034007, 0.030806, 0.028165, 0.025972, 0.024146, 0.022621, 0.021346, 0.020281, 0.019392, 0.018652,
    0.090763, 0.066837, 0.053940, 0.045021, 0.038412, 0.033320, 0.029293, 0.026045, 0.023387, 0.021183, 0.019337, 0.017776, 0.016445, 0.015302, 0.014314, 0.013455, 0.012703,
);

// Bilinear `E_dielectric(µ, r)`; arguments clamp to `[0, 1]`. Mirrors
// `dielectric_albedo_table.rs::dielectric_albedo_lookup`.
fn dielectric_albedo_lookup(n_dot_v: f32, roughness: f32) -> f32 {
    let x = clamp(n_dot_v, 0.0, 1.0) * f32(DIEL_N_MU - 1);
    let y = clamp(roughness, 0.0, 1.0) * f32(DIEL_N_R - 1);
    let i0 = min(i32(x), DIEL_N_MU - 2);
    let j0 = min(i32(y), DIEL_N_R - 2);
    let fx = x - f32(i0);
    let fy = y - f32(j0);
    let a = DIELECTRIC_ALBEDO[j0 * DIEL_N_MU + i0];
    let b = DIELECTRIC_ALBEDO[j0 * DIEL_N_MU + i0 + 1];
    let c = DIELECTRIC_ALBEDO[(j0 + 1) * DIEL_N_MU + i0];
    let d = DIELECTRIC_ALBEDO[(j0 + 1) * DIEL_N_MU + i0 + 1];
    let top = a + (b - a) * fx;
    let bottom = c + (d - c) * fx;
    return clamp(top + (bottom - top) * fy, 0.0, 1.0);
}

// Single-scattering directional albedo `E(mu, r)` of the GGX lobe, from the
// baked table (view-dependent; the analytic `scale + bias` fit is not, which is
// what made a white metal furnace reflect 1.11 instead of 1.0). Mirrors
// `multiscatter::multiscatter_direct`.
fn ms_directional_albedo(n_dot: f32, perceptual_roughness: f32) -> f32 {
    return ggx_albedo_lookup(n_dot, perceptual_roughness);
}

// Cosine-weighted hemispherical average albedo `E_avg(r)`, pre-integrated from
// the same table. Mirrors `multiscatter::multiscatter_direct`.
fn ms_average_albedo(perceptual_roughness: f32) -> f32 {
    return ggx_average_lookup(perceptual_roughness);
}

// Added multiple-scattering specular BRDF value `f_ms * F_ms` (Kulla-Conty
// achromatic lobe scaled by Turquin's colored multiple-bounce Fresnel), to be
// added to the single-scattering specular before the `N.L` weight. Mirrors
// `multiscatter::multiscatter_direct`.
fn multiscatter_direct(
    f0: vec3<f32>,
    n_dot_v: f32,
    n_dot_l: f32,
    perceptual_roughness: f32,
) -> vec3<f32> {
    let e_o = ms_directional_albedo(n_dot_v, perceptual_roughness);
    let e_i = ms_directional_albedo(n_dot_l, perceptual_roughness);
    let e_avg = ms_average_albedo(perceptual_roughness);
    return multiscatter_direct_parts(f0, e_o, e_i, e_avg);
}

// Same value with the three albedos supplied by the caller, so a shading loop
// can hoist the view- and material-only parts out of its per-light loop (only
// `E_white(n·l)` actually depends on the light). Mirrors
// `multiscatter::multiscatter_direct_parts`.
fn multiscatter_direct_parts(f0: vec3<f32>, e_o: f32, e_i: f32, e_avg: f32) -> vec3<f32> {
    let f_ms = (1.0 - e_o) * (1.0 - e_i) * INV_PI / (1.0 - e_avg);
    let f_avg = f0 + (vec3<f32>(1.0) - f0) * (1.0 / 21.0);
    let f_ms_color =
        f_avg * f_avg * e_avg / max(vec3<f32>(1.0) - f_avg * (1.0 - e_avg), vec3<f32>(MS_MIN_ENERGY_DEFICIT));
    return f_ms * f_ms_color;
}

// Oren-Nayar "qualitative" (Gotanda / Fujii, trig-free) rough-diffuse factor on
// the Lambertian `albedo / pi` term. Byte-for-byte twin of
// `lighting.rs::oren_nayar_factor` and of the path-traced oracle
// `reference_pt::oren_nayar`. `fuzz == 0` is exactly Lambert.
const OREN_A_OFFSET: f32 = 0.33;
const OREN_B_OFFSET: f32 = 0.09;
const OREN_A_SCALE: f32 = 0.5;
const OREN_B_SCALE: f32 = 0.45;
const OREN_MIN_T: f32 = 1.0e-6;
// `pi / 2`, matching `core::f32::consts::FRAC_PI_2` on the CPU side.
const OREN_MAX_SIGMA: f32 = PI * 0.5;

fn oren_nayar_factor(n_dot_v: f32, n_dot_l: f32, v_dot_l: f32, diffuse_roughness: f32) -> f32 {
    let sigma = clamp(diffuse_roughness, 0.0, 1.0) * OREN_MAX_SIGMA;
    let sigma_sq = sigma * sigma;
    let a = 1.0 - OREN_A_SCALE * sigma_sq / (sigma_sq + OREN_A_OFFSET);
    let b = OREN_B_SCALE * sigma_sq / (sigma_sq + OREN_B_OFFSET);
    let s = v_dot_l - n_dot_v * n_dot_l;
    var t = 1.0;
    if s > 0.0 {
        t = max(max(n_dot_v, n_dot_l), OREN_MIN_T);
    }
    return max(a + b * s / t, 0.0);
}

// `Ê_ON(μ_i, σ_j)`, row-major over σ (`μ_i = i/16`, `σ_j = j/8·π/2`).
const EON_N_MU: i32 = 17;
const EON_N_SIGMA: i32 = 9;
const EON_ALBEDO: array<f32, 153> = array<f32, 153>(
    1.000102, 1.000102, 1.000102, 1.000102, 1.000102, 1.000102, 1.000102, 1.000102, 1.000102, 1.000102, 1.000102, 1.000102, 1.000102, 1.000102, 1.000102, 1.000102, 1.000102,
    0.986645, 0.983973, 0.981022, 0.978077, 0.975073, 0.972048, 0.969053, 0.966109, 0.963232, 0.960478, 0.957871, 0.955444, 0.953240, 0.951300, 0.949674, 0.948444, 0.947782,
    0.922654, 0.917026, 0.910807, 0.904613, 0.898283, 0.891915, 0.885615, 0.879407, 0.873357, 0.867555, 0.862069, 0.856957, 0.852316, 0.848229, 0.844811, 0.842220, 0.840847,
    0.846674, 0.839602, 0.831781, 0.823991, 0.816032, 0.808029, 0.800101, 0.792296, 0.784691, 0.777392, 0.770494, 0.764063, 0.758232, 0.753093, 0.748796, 0.745533, 0.743797,
    0.787386, 0.779612, 0.771021, 0.762455, 0.753710, 0.744912, 0.736201, 0.727622, 0.719264, 0.711242, 0.703662, 0.696595, 0.690184, 0.684535, 0.679813, 0.676232, 0.674332,
    0.746080, 0.737933, 0.728924, 0.719954, 0.710785, 0.701562, 0.692436, 0.683445, 0.674682, 0.666280, 0.658332, 0.650927, 0.644210, 0.638288, 0.633338, 0.629584, 0.627586,
    0.717770, 0.709404, 0.700156, 0.690942, 0.681529, 0.672060, 0.662687, 0.653458, 0.644460, 0.635832, 0.627671, 0.620066, 0.613169, 0.607089, 0.602005, 0.598151, 0.596103,
    0.698071, 0.689569, 0.680169, 0.670805, 0.661239, 0.651613, 0.642086, 0.632702, 0.623559, 0.614790, 0.606495, 0.598766, 0.591756, 0.585576, 0.580408, 0.576491, 0.574419,
    0.684031, 0.675436, 0.665937, 0.656470, 0.646799, 0.637073, 0.627444, 0.617960, 0.608717, 0.599852, 0.591468, 0.583657, 0.576569, 0.570325, 0.565099, 0.561141, 0.559036,
);

// `⟨Ê_ON⟩(σ_j)`.
const EON_AVERAGE: array<f32, 9> = array<f32, 9>(
    1.000102, 0.957581, 0.861456, 0.769724, 0.702815, 0.657445, 0.626761, 0.605569, 0.590533,
);

// Bilinear `Ê_ON(μ, σ)` with `sigma01 = σ/(π/2) ∈ [0, 1]`. Mirrors
// `eon_albedo_table.rs::eon_albedo_lookup`.
fn eon_albedo_lookup(mu: f32, sigma01: f32) -> f32 {
    let x = clamp(mu, 0.0, 1.0) * f32(EON_N_MU - 1);
    let y = clamp(sigma01, 0.0, 1.0) * f32(EON_N_SIGMA - 1);
    let i0 = min(i32(x), EON_N_MU - 2);
    let j0 = min(i32(y), EON_N_SIGMA - 2);
    let fx = x - f32(i0);
    let fy = y - f32(j0);
    let a = EON_ALBEDO[j0 * EON_N_MU + i0];
    let b = EON_ALBEDO[j0 * EON_N_MU + i0 + 1];
    let c = EON_ALBEDO[(j0 + 1) * EON_N_MU + i0];
    let d = EON_ALBEDO[(j0 + 1) * EON_N_MU + i0 + 1];
    let top = a + (b - a) * fx;
    let bottom = c + (d - c) * fx;
    return clamp(top + (bottom - top) * fy, 0.0, 1.0);
}

// Linear `⟨Ê_ON⟩(σ)` with `sigma01 = σ/(π/2) ∈ [0, 1]`.
fn eon_average_lookup(sigma01: f32) -> f32 {
    let y = clamp(sigma01, 0.0, 1.0) * f32(EON_N_SIGMA - 1);
    let j0 = min(i32(y), EON_N_SIGMA - 2);
    let fy = y - f32(j0);
    let a = EON_AVERAGE[j0];
    let b = EON_AVERAGE[j0 + 1];
    return clamp(a + (b - a) * fy, 0.0, 1.0);
}

// `OpenPBR` equation `EON_comp`: reciprocal energy compensation for the
// energy-preserving Oren-Nayar diffuse,
// `f_comp = (ρ_ms/π)(1 − Ê(ωi))(1 − Ê(ωo))` with
// `ρ_ms = ρ²·(⟨Ê⟩/(1−⟨Ê⟩))/(1 − ρ(1−⟨Ê⟩))`. `Ê`/`⟨Ê⟩` come from the baked
// table derived from the specification's own `FON_brdf`. Mirrors
// `lighting.rs::oren_nayar_energy_compensation`; at `diffuse_roughness == 0` the
// lookup saturates at 1 so this is exactly zero.
fn oren_nayar_energy_compensation(
    base_color: vec3<f32>,
    n_dot_v: f32,
    n_dot_l: f32,
    diffuse_roughness: f32,
) -> vec3<f32> {
    let sigma01 = clamp(diffuse_roughness, 0.0, 1.0);
    // Fast path: at `diffuse_roughness == 0` the compensation is *exactly* zero
    // (`Ê_ON ≡ 1`), so the table lookups are skipped for the common case.
    if sigma01 <= 0.0 {
        return vec3<f32>(0.0);
    }
    let e_avg = clamp(eon_average_lookup(sigma01), 0.0, 1.0 - 1.0e-4);
    let e_i = eon_albedo_lookup(max(n_dot_v, 0.0), sigma01);
    let e_o = eon_albedo_lookup(max(n_dot_l, 0.0), sigma01);
    let one_minus_i = max(1.0 - e_i, 0.0);
    let one_minus_o = max(1.0 - e_o, 0.0);
    let ratio = e_avg / (1.0 - e_avg);
    let rho = clamp(base_color, vec3<f32>(0.0), vec3<f32>(1.0));
    let rho_ms = rho * rho * ratio / max(vec3<f32>(1.0) - rho * (1.0 - e_avg), vec3<f32>(1.0e-4));
    return (rho_ms * INV_PI) * (one_minus_i * one_minus_o);
}

// Physically based direct lighting for one analytic light, including a
// Kulla-Conty energy-compensation term and a clearcoat lobe. Mirrors
// `evaluate_principled_direct`.
// View- and material-only terms of the principled lobe, computed once per pixel
// and reused across every analytic light (`4·(N−1)` fewer table lookups per
// pixel). Mirrors `lighting.rs::PrincipledPre`.
struct PrincipledPre {
    e_ggx_view: f32,
    e_ggx_avg: f32,
    e_dielectric: vec3<f32>,
    coat_lobe_albedo: f32,
}

fn principled_pre(surface: SurfaceSample, frame: ShadingFrame) -> PrincipledPre {
    let n = brdf_normalize_or(frame.normal, vec3<f32>(0.0, 1.0, 0.0));
    let v = brdf_normalize_or(frame.view, n);
    let n_dot_v = max(dot(n, v), MIN_N_DOT);
    let roughness = clamp(surface.perceptual_roughness, MIN_ROUGHNESS, 1.0);
    let coat = clamp(surface.clearcoat, 0.0, 1.0);
    let coat_roughness = clamp(surface.clearcoat_roughness, MIN_ROUGHNESS, 1.0);
    let coat_ior = max(surface.coat_ior, 1.0 + 1.0e-3);
    let coat_f0 = pow((coat_ior - 1.0) / (coat_ior + 1.0), 2.0);
    var coat_lobe_albedo = 1.0;
    if coat > 0.0 {
        coat_lobe_albedo = max(
            coat_f0 * ggx_albedo_lookup(n_dot_v, coat_roughness)
                + (1.0 - coat_f0) * schlick_bias_lookup(n_dot_v, coat_roughness),
            1.0e-3,
        );
    }
    let e_ref = dielectric_albedo_lookup(n_dot_v, roughness);
    let dielectric_f0_vec = dielectric_f0(surface);
    return PrincipledPre(
        ggx_albedo_lookup(n_dot_v, roughness),
        ggx_average_lookup(roughness),
        clamp(
            vec3<f32>(e_ref) * dielectric_f0_vec / vec3<f32>(0.04),
            vec3<f32>(0.0),
            vec3<f32>(1.0),
        ),
        coat_lobe_albedo,
    );
}

fn principled_direct(
    surface: SurfaceSample,
    frame: ShadingFrame,
    light: DirectLightSample,
) -> vec3<f32> {
    return principled_direct_with(surface, frame, light, principled_pre(surface, frame));
}

/// Glint seed for one light direction (§17.6) — the GPU twin of
/// `lighting.rs::glint_shade_seed`.
///
/// The discrete estimator is unbiased only if its randomness decorrelates per
/// (space, direction) cell, so the seed mixes the shade-time surface cell with
/// the half-vector slope in the surface tangent frame, quantised at two
/// resolutions: one seed bucket per flake lattice cell (`sqrt(density)` cells
/// per unit) and one slope bin per flake orientation spread
/// (`1 / max(roughness², MIN_ALPHA)`). The latter sets how fast the sparkle
/// pattern decorrelates as the view or light moves; it deliberately does not
/// change the lobe's energy, because the estimator samples the base lobe's own
/// NDF (whose ensemble mean it is).
fn brdf_glint_shade_seed(
    surface: SurfaceSample,
    h: vec3<f32>,
    tangent: vec3<f32>,
    bitangent: vec3<f32>,
    n: vec3<f32>,
) -> u32 {
    let cells_per_unit = sqrt(max(surface.glint_density, 0.0));
    let roughness = clamp(surface.glint_roughness, 0.0, 1.0);
    let flake_alpha = max(roughness * roughness, GLINT_MIN_ALPHA);
    let h_n = max(dot(n, h), MIN_N_DOT);
    let slope_x = glint_half_slope(dot(tangent, h), h_n);
    let slope_y = glint_half_slope(dot(bitangent, h), h_n);
    return glint_seed(
        surface.glint_cell.x,
        surface.glint_cell.y,
        cells_per_unit,
        slope_x,
        slope_y,
        1.0 / flake_alpha,
        0u,
    );
}

fn principled_direct_with(
    surface: SurfaceSample,
    frame: ShadingFrame,
    light: DirectLightSample,
    pre: PrincipledPre,
) -> vec3<f32> {
    return principled_direct_channels_with(surface, frame, light, pre).radiance;
}

// Keep the legacy radiance expression intact for fixed-point comparisons;
// composition consumes the two contributions at their source, never a residual
// obtained by subtracting indirect illumination from the final HDR color.
struct PrincipledDirectChannels {
    radiance: vec3<f32>,
    direct: vec3<f32>,
    transmission: vec3<f32>,
}

// Scattering-only twin of `subsurface.rs::subsurface_diffuse_terms`.
// The standalone subsurface/cloth modules import this module's surface and
// GGX helpers, so the principled mixture cannot import their full integrators
// without a cycle (or double-counting specular/emission).
struct PrincipledSubsurfaceTerms {
    diffuse: vec3<f32>,
    transmission: vec3<f32>,
}

fn principled_subsurface_terms(
    surface: SurfaceSample,
    n: vec3<f32>,
    v: vec3<f32>,
    l: vec3<f32>,
    raw_n_dot_l: f32,
    diffuse_weight: vec3<f32>,
) -> PrincipledSubsurfaceTerms {
    let subsurface = clamp(surface.subsurface, 0.0, 1.0);
    let thickness = clamp(surface.thickness, 0.0, 1.0);
    var diffuse = vec3<f32>(0.0);
    for (var c = 0u; c < 3u; c = c + 1u) {
        let wrap = clamp(
            subsurface * surface.subsurface_radius * surface.subsurface_radius_scale[c],
            0.0,
            1.0,
        );
        let one_plus_w = 1.0 + wrap;
        let e_wrap = max((1.0 + 2.0 * wrap) / (one_plus_w * one_plus_w), 1.0e-3);
        let diffuse_cos = clamp(
            (raw_n_dot_l + wrap) / ((1.0 + wrap) * (1.0 + wrap)),
            0.0,
            1.0,
        ) / e_wrap;
        diffuse[c] = surface.base_color[c]
            * surface.subsurface_color[c]
            * INV_PI
            * diffuse_cos
            * diffuse_weight[c];
    }
    let h_transmission = brdf_normalize_or(l + n * 0.2, n);
    let sharpness = max(
        12.0 * (1.0 + clamp(surface.scatter_anisotropy, -1.0, 1.0)),
        MIN_N_DOT,
    );
    let back = pow(clamp(dot(v, -h_transmission), 0.0, 1.0), sharpness);
    let transmitted = (surface.base_color * surface.subsurface_color)
        * (subsurface * back * (1.0 - thickness));
    return PrincipledSubsurfaceTerms(diffuse, transmitted);
}

// Same baked directional albedo and interpolation as charlie_albedo_table.rs
// and cloth.wesl. Kept local to avoid the cloth -> brdf import cycle.
const PRINCIPLED_FUZZ_N_MU: i32 = 17;
const PRINCIPLED_FUZZ_N_R: i32 = 9;
const PRINCIPLED_FUZZ_ALBEDO: array<f32, 153> = array<f32, 153>(
    1.000000, 1.000000, 0.649052, 0.282295, 0.116762, 0.044961, 0.015821, 0.004987, 0.001375, 0.000322, 0.000061, 0.000009, 0.000001, 0.000000, 0.000000, 0.000000, 0.000000,
    1.000000, 0.868676, 0.677266, 0.538730, 0.431285, 0.345282, 0.275282, 0.217819, 0.170482, 0.131497, 0.099501, 0.073411, 0.052347, 0.035572, 0.022467, 0.012500, 0.005209,
    0.928916, 0.716657, 0.595808, 0.504953, 0.431375, 0.369553, 0.316454, 0.270209, 0.229555, 0.193579, 0.161614, 0.133140, 0.107750, 0.085115, 0.064965, 0.047073, 0.031251,
    0.810533, 0.646225, 0.552173, 0.480724, 0.422060, 0.371965, 0.328132, 0.289150, 0.254069, 0.222217, 0.193104, 0.166352, 0.141669, 0.118817, 0.097613, 0.077898, 0.059546,
    0.744316, 0.604668, 0.524836, 0.463977, 0.413764, 0.370588, 0.332490, 0.298281, 0.267155, 0.238550, 0.212042, 0.187323, 0.164145, 0.142310, 0.121656, 0.102055, 0.083397,
    0.701662, 0.577044, 0.506022, 0.451871, 0.407107, 0.368500, 0.334302, 0.303441, 0.275201, 0.249079, 0.224705, 0.201788, 0.180111, 0.159499, 0.139803, 0.120900, 0.102697,
    0.671750, 0.557241, 0.492234, 0.442722, 0.401779, 0.366425, 0.335052, 0.306665, 0.280621, 0.256435, 0.233778, 0.212381, 0.192041, 0.172592, 0.153896, 0.135837, 0.118325,
    0.649554, 0.542347, 0.481705, 0.435588, 0.397465, 0.364541, 0.335312, 0.308831, 0.284496, 0.261864, 0.240608, 0.220482, 0.201294, 0.182884, 0.165127, 0.147906, 0.131134,
    0.632425, 0.530705, 0.473381, 0.429855, 0.393906, 0.362883, 0.335331, 0.310368, 0.287407, 0.266029, 0.245935, 0.226876, 0.208675, 0.191184, 0.174267, 0.157825, 0.141770,
);

fn principled_fuzz_albedo_lookup(n_dot_v: f32, roughness: f32) -> f32 {
    let x = clamp(n_dot_v, 0.0, 1.0) * f32(PRINCIPLED_FUZZ_N_MU - 1);
    let y = clamp(roughness, 0.0, 1.0) * f32(PRINCIPLED_FUZZ_N_R - 1);
    let i0 = min(i32(x), PRINCIPLED_FUZZ_N_MU - 2);
    let j0 = min(i32(y), PRINCIPLED_FUZZ_N_R - 2);
    let fx = x - f32(i0);
    let fy = y - f32(j0);
    let a = PRINCIPLED_FUZZ_ALBEDO[j0 * PRINCIPLED_FUZZ_N_MU + i0];
    let b = PRINCIPLED_FUZZ_ALBEDO[j0 * PRINCIPLED_FUZZ_N_MU + i0 + 1];
    let c = PRINCIPLED_FUZZ_ALBEDO[(j0 + 1) * PRINCIPLED_FUZZ_N_MU + i0];
    let d = PRINCIPLED_FUZZ_ALBEDO[(j0 + 1) * PRINCIPLED_FUZZ_N_MU + i0 + 1];
    let top = a + (b - a) * fx;
    let bottom = c + (d - c) * fx;
    return clamp(top + (bottom - top) * fy, 0.0, 1.0);
}

fn principled_distribution_charlie(roughness: f32, n_dot_h: f32) -> f32 {
    let inv_r = 1.0 / roughness;
    let sin2h = max(1.0 - n_dot_h * n_dot_h, 0.0078125);
    return (2.0 + inv_r) * pow(sin2h, inv_r * 0.5) * (1.0 / (2.0 * PI));
}

fn principled_visibility_ashikhmin(n_dot_v: f32, n_dot_l: f32) -> f32 {
    return 1.0 / max(4.0 * (n_dot_l + n_dot_v - n_dot_l * n_dot_v), MIN_N_DOT);
}

fn principled_direct_channels_with(
    surface: SurfaceSample,
    frame: ShadingFrame,
    light: DirectLightSample,
    pre: PrincipledPre,
) -> PrincipledDirectChannels {
    let n = brdf_normalize_or(frame.normal, vec3<f32>(0.0, 1.0, 0.0));
    let v = brdf_normalize_or(frame.view, n);
    let l = brdf_normalize_or(light.direction, n);
    let h = brdf_normalize_or(v + l, n);
    let raw_n_dot_l = dot(n, l);
    let n_dot_l = max(raw_n_dot_l, 0.0);
    let n_dot_v = max(dot(n, v), MIN_N_DOT);
    let n_dot_h = max(dot(n, h), MIN_N_DOT);
    let v_dot_h = max(dot(v, h), 0.0);
    if n_dot_l <= 0.0 && surface.subsurface <= 0.0 {
        return PrincipledDirectChannels(surface.emissive, vec3<f32>(0.0), vec3<f32>(0.0));
    }

    let metallic = clamp(surface.metallic, 0.0, 1.0);
    let roughness = clamp(surface.perceptual_roughness, MIN_ROUGHNESS, 1.0);
    // Specular anti-aliasing: widen the GGX width by the surface normal
    // variance (already in the `alpha^2` domain), capped at the
    // Tokuyoshi-Kaplanyan `kappa_max`. `spec_aa_filter_alpha_sq` takes and
    // returns `alpha^2`; `sqrt` returns to the `alpha` the lobe consumes.
    // Byte-identical to `evaluate_principled_direct` in `lighting.rs`; with
    // `normal_variance == 0` it is an exact identity.
    let base_alpha = roughness * roughness;
    let alpha = sqrt(spec_aa_filter_alpha_sq(base_alpha * base_alpha, surface.normal_variance, SPEC_AA_KAPPA_MAX));
    // `OpenPBR` dielectric base specular: the exact angle-dependent Fresnel at
    // the `specular_weight`-modulated index ratio (equations
    // `unmodulated_fresnel` / `modulated_ior`), tinted by `specular_color`. A
    // plain Schlick curve is not equivalent — it reaches total reflection at
    // grazing incidence even at `specular_weight == 0`, where the specification
    // removes the interface. The conductor branch keeps Schlick from
    // `base_color`. Byte-identical to `evaluate_principled_direct` in
    // `lighting.rs`.
    let f0 = mix(dielectric_f0(surface), surface.base_color, metallic);
    let f_dielectric = brdf_base_dielectric_fresnel_rgb(
        v_dot_h,
        surface.specular_ior,
        surface.specular_weight,
        surface.specular_color,
    );
    let f_metal = brdf_base_metal_fresnel_rgb(
        v_dot_h,
        surface.base_weight,
        surface.base_color,
        surface.specular_weight,
        surface.specular_color,
    );
    let f_specular = mix(f_dielectric, f_metal, metallic);
    // Thin-film iridescence: blend the base Fresnel toward the Airy interference
    // reflectance at the same `v_dot_h`; `iridescence == 0` is an exact identity.
    let iridescence = clamp(surface.iridescence, 0.0, 1.0);
    var f = f_specular;
    if iridescence > 0.0 {
        let f_irid = tf_iridescent_fresnel_rgb(
            1.0,
            surface.iridescence_ior,
            f0,
            v_dot_h,
            surface.iridescence_thickness_nm,
        );
        f = mix(f_specular, f_irid, iridescence);
    }

    // Anisotropic axes: re-orthonormalize + rotate the interpolated tangent
    // frame (the rotation is a Prism extension standing in for a per-vertex
    // `geometry_tangent`), then split `alpha` onto the two axes with the
    // `OpenPBR` equation `openpbr-anisotropy-formula`:
    //   α_t = α·sqrt(2/(1+(1−a)²)),  α_b = (1−a)·α_t,  a ∈ [0, 1],  α = r².
    // `a == 0` gives `sqrt(2/2) == 1` exactly, so the axes collapse to `alpha`
    // and the anisotropic D/V reduce to the isotropic GGX pair bit-for-bit.
    // Byte twin of `openpbr_anisotropy_to_alpha` in `lighting.rs`.
    let axes = anisotropic_axes(n, frame.tangent, frame.bitangent, surface.anisotropy_rotation);
    let a = clamp(surface.anisotropy, 0.0, 1.0);
    let one_minus = 1.0 - a;
    let stretch = sqrt(2.0 / (1.0 + one_minus * one_minus));
    let ax = max(alpha * stretch, MIN_ANISOTROPIC_ALPHA);
    let ay = max(one_minus * alpha * stretch, MIN_ANISOTROPIC_ALPHA);
    let d_smooth =
        distribution_ggx_anisotropic(dot(axes.tangent, h), dot(axes.bitangent, h), n_dot_h, ax, ay);
    // §17.6 discrete (counting) microfacet NDF. The estimator is unbiased for
    // whatever density it samples, so it is fed the lobe's *actual* `D` (the
    // anisotropic one included) rather than an isotropic stand-in; at
    // `glint_weight == 0` the bypass returns `d_smooth` unchanged, so the
    // smooth-GGX path is byte-for-byte what it was before this lobe existed.
    // An unknown footprint (no pixel area) keeps the smooth lobe: there is no
    // facet count to draw from, and treating "unknown" as "zero" would blacken
    // the highlight instead of degrading to the smooth model.
    var d = d_smooth;
    if (surface.glint_weight > 0.0 && surface.glint_footprint_area > 0.0) {
        // The single-lobe sampler (§17.4) hands us the mixture state it drew;
        // `< 0` means "no sampling happened, use the authored weight". Either way
        // this stays a convex blend, so the NDF can never go negative.
        let mix = select(
            clamp(surface.glint_state, 0.0, 1.0),
            clamp(surface.glint_weight, 0.0, 1.0),
            surface.glint_state < 0.0,
        );
        if (mix > 0.0) {
            let d_glint = glint_ndf_from_density(
                d_smooth,
                glint_microfacet_count(surface.glint_density, surface.glint_footprint_area),
                glint_bin_solid_angle(surface.glint_bin_half_width),
                brdf_glint_shade_seed(surface, h, axes.tangent, axes.bitangent, n),
            );
            d = d_smooth + (d_glint - d_smooth) * mix;
        }
    }
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

    // Multiple-scattering energy compensation (Kulla-Conty 2017 achromatic lobe
    // + Turquin 2019 colored multiple-bounce Fresnel). Byte-for-byte twin of
    // `multiscatter::multiscatter_direct`; restores the energy the single-
    // scattering GGX lobe loses to unresolved microfacet inter-reflections.
    let compensation = multiscatter_direct_parts(
        f0,
        pre.e_ggx_view,
        ggx_albedo_lookup(n_dot_l, roughness),
        pre.e_ggx_avg,
    );
    let specular = single_scatter + compensation;

    // `OpenPBR` `glossy_diffuse_albedo_constraint`: weight the diffuse slab by
    // one minus the *directional albedo* of the dielectric specular lobe, not by
    // the half-vector Fresnel. The lobe albedo is baked at the specification's
    // default dielectric and scaled by the authored `F0` ratio (exact at
    // `F0 == 0` and at the default). Mirrors `lighting.rs`.
    let diffuse_weight = (vec3<f32>(1.0) - pre.e_dielectric) * (1.0 - metallic);
    // `OpenPBR` `diffuse_roughness`: Oren-Nayar replaces Lambert for
    // microscopically rough diffuse surfaces; `0` collapses the factor to
    // exactly 1.0 and preserves the legacy path.
    let diffuse_factor =
        oren_nayar_factor(n_dot_v, n_dot_l, dot(v, l), surface.diffuse_roughness);
    // `OpenPBR` `EON_brdf` / `EON_comp`: the Fujii term plus the reciprocal
    // multiple-scattering compensation that restores the energy plain
    // Oren-Nayar loses as `base_diffuse_roughness` grows. Zero at roughness 0.
    let diffuse_on = surface.base_color * (INV_PI * diffuse_factor);
    let diffuse_comp = oren_nayar_energy_compensation(
        surface.base_color,
        n_dot_v,
        n_dot_l,
        surface.diffuse_roughness,
    );
    let diffuse = (diffuse_on + diffuse_comp) * diffuse_weight;
    // `OpenPBR` `base_weight` scales the whole base layer; 1.0 is exact.
    var base = (diffuse + specular) * (n_dot_l * clamp(surface.base_weight, 0.0, 1.0));
    let subsurface = clamp(surface.subsurface, 0.0, 1.0);
    if subsurface > 0.0 {
        let scattered = principled_subsurface_terms(surface, n, v, l, raw_n_dot_l, diffuse_weight);
        let diffuse_mix = diffuse * (n_dot_l * (1.0 - subsurface))
            + scattered.diffuse * subsurface;
        // Replace the diffuse share only; GGX and the enclosing coat/fuzz
        // layers remain present exactly once, as in the CPU principled path.
        base = ((diffuse_mix + specular * n_dot_l)
            + scattered.transmission * diffuse_weight)
            * clamp(surface.base_weight, 0.0, 1.0);
    }

    // Clearcoat: a second, always-dielectric GGX lobe that attenuates the base
    // response by its own Fresnel reflectance.
    let coat = clamp(surface.clearcoat, 0.0, 1.0);
    let coat_roughness = clamp(surface.clearcoat_roughness, MIN_ROUGHNESS, 1.0);
    let coat_alpha = coat_roughness * coat_roughness;
    // `OpenPBR` `coat_ior` semantics: the coat IOR drives the coat interface
    // Fresnel (the highlight used to be pinned at the legacy `F0 = 0.04`).
    let coat_f0 = pow((clamp(surface.coat_ior, 1.0 + 1.0e-3, 3.0) - 1.0)
        / (clamp(surface.coat_ior, 1.0 + 1.0e-3, 3.0) + 1.0), 2.0);
    let coat_f = fresnel_schlick(vec3<f32>(coat_f0), v_dot_h);
    let coat_single_scatter = coat_f
        * (coat
            * distribution_ggx(n_dot_h, coat_alpha)
            * vis_smith(n_dot_v, n_dot_l, coat_alpha)
            * n_dot_l);
    // Coat energy closure: the base keeps the closed-form slab transmittance
    // `t_slab = (1−f_v)(1−f_l)/(1−f_v·f_l)` and the coat carries the complementary
    // reflectance `r_slab = f_l + (1−f_l)²f_v/(1−f_v·f_l)` (they sum to exactly 1
    // for a non-absorbing coat). The coat's single-scatter lobe is given `r_slab`
    // of energy by adding a **zero-albedo** redistribution
    // `(r_slab − A)·(n·l/π)`, where `A = F0·E_white + (1−F0)·B` is the exact
    // Schlick split-sum albedo — so the estimate `A` shapes the correction but
    // never the energy budget. Byte-identical to `evaluate_principled_direct` in
    // `lighting.rs`; `clearcoat == 0` is an exact identity (`1 − 0 + 0·t = 1`).
    let coat_f_v = clamp(fresnel_schlick(vec3<f32>(coat_f0), n_dot_v).x, 0.0, 1.0);
    let coat_f_l = clamp(fresnel_schlick(vec3<f32>(coat_f0), n_dot_l).x, 0.0, 1.0);
    let coat_denom = max(1.0 - coat_f_v * coat_f_l, 1.0e-4);
    let coat_t_slab = (1.0 - coat_f_v) * (1.0 - coat_f_l) / coat_denom;
    let coat_r_slab = coat_f_l + (1.0 - coat_f_l) * (1.0 - coat_f_l) * coat_f_v / coat_denom;
    // Fast path: with no coat every term below is scaled away, so the table
    // lookups are skipped entirely for a material that never authors a coat.
    let coat_lobe_albedo = pre.coat_lobe_albedo;
    let coat_specular = coat_single_scatter
        + vec3<f32>(coat * (coat_r_slab - coat_lobe_albedo) * n_dot_l * INV_PI);
    let coat_attenuation = 1.0 + coat * (coat_t_slab - 1.0);

    // `OpenPBR` transmission lobe: interface Fresnel (driven by
    // `index_of_refraction`) gating entry and exit, Beer-Lambert over
    // `thickness`, and a per-channel index spread from `dispersion`. Byte twin
    // of the same block in `lighting.rs`; `transmission == 0` is exactly zero.
    var transmission_term = vec3<f32>(0.0);
    let transmission = clamp(surface.transmission, 0.0, 1.0);
    if transmission > 0.0 {
        let ior = clamp(surface.index_of_refraction, 1.0, 3.0);
        let dispersion = clamp(surface.dispersion, 0.0, 1.0);
        // `OpenPBR` `transmission_depth`: distance at which the slab is exactly
        // `transmission_color` (a length). The path is fixed at one scene unit,
        // so a unit path transmits `slab_color^(1/t_depth)`; `t_depth == 0` is
        // the specification's non-physical tint case (authored tint applied
        // directly). (`depth` is a WGSL reserved word, hence the `t_depth` local.)
        let t_depth = max(surface.thickness, 0.0);
        // `geometry_thin_walled`: an open, single-interface surface with no
        // volume behind it, so the Beer-Lambert depth term is skipped.
        let thin_walled = surface.thin_walled > 0.5;
        // `OpenPBR` `transmission_scatter_anisotropy`: Henyey-Greenstein shape of
        // the in-scatter, normalised so `g == 0` reproduces the isotropic term.
        let scatter_g = clamp(surface.transmission_scatter_anisotropy, -1.0, 1.0);
        let scatter_cos = -dot(v, l);
        let phase = vc_forward_phase(scatter_cos, scatter_g) / vc_forward_phase(scatter_cos, 0.0);
        for (var c = 0u; c < 3u; c = c + 1u) {
            // `OpenPBR` dispersion: the Abbe number sets the per-channel index,
            // scaled by `transmission_dispersion_scale` (0 = achromatic).
            let ior_c = dispersive_ior(
                ior,
                surface.transmission_dispersion_abbe_number,
                dispersion,
                c,
            );
            let f_view = fresnel_dielectric(n_dot_v, ior_c);
            let f_light = fresnel_dielectric(n_dot_l, ior_c);
            var transmittance = 1.0;
            if !thin_walled {
                // `transmission_color` tints the slab (multiplied with the base
                // albedo), so white is the exact identity.
                let slab_color = surface.base_color[c] * surface.transmission_color[c];
                // `t_depth == 0` is the specification's non-physical tint case:
                // apply the authored tint directly. Otherwise Beer-Lambert with
                // the path fixed at one scene unit ⇒ `slab_color^(1/t_depth)`.
                var beer = clamp(surface.transmission_color[c], 0.0, 1.0);
                if t_depth > 0.0 {
                    beer = transmittance_pow(slab_color, 1.0 / t_depth);
                }
                // `transmission_scatter` re-scatters the absorbed fraction
                // instead of losing it (0 = pure Beer-Lambert).
                let scatter = clamp(surface.transmission_scatter[c], 0.0, 1.0);
                transmittance = beer + phase * scatter * (1.0 - beer);
            }
            // Smooth two-interface slab, closed form: the internal bounces form a
            // geometric series with ratio `f_v·f_l`, giving
            //   T = (1−f_v)(1−f_l)/(1 − f_v·f_l)
            //   R = f_l + (1−f_l)² f_v/(1 − f_v·f_l)
            // with exactly `R + T = 1` for a non-absorbing slab. Modelling the
            // lobe as `R + T` (not transmission alone) keeps a grazing view from
            // losing the energy the interface reflects. Mirrors `lighting.rs`.
            let denom = max(1.0 - f_view * f_light, 1.0e-4);
            let t_slab = (1.0 - f_view) * (1.0 - f_light) / denom;
            let one_minus_light = 1.0 - f_light;
            let r_slab = f_light + one_minus_light * one_minus_light * f_view / denom;
            // `OpenPBR` dispersion, physically: a channel with a higher index
            // refracts to a shallower cosine and therefore traverses a *longer*
            // path through the absorbing slab. Extending the Beer-Lambert path by
            // `1/µᵗ` is what keeps dispersion visible now that the slab is
            // energy-closing (`R + T = 1` makes the sum wavelength-independent).
            let s_refr = 1.0 - max(1.0 - n_dot_v * n_dot_v, 0.0) / (ior_c * ior_c);
            let mu_t = select(1.0, sqrt(max(s_refr, 1.0e-3)), s_refr > 0.0);
            let path_scale = 1.0 / mu_t;
            transmission_term[c] =
                transmission * (r_slab + t_slab * pow(transmittance, path_scale));
        }
        // Lambertian transmission is a BRDF: it carries the `1/π` a radiance
        // factor would not (without it the lobe's albedo is `(1−f)²·π`).
        transmission_term = transmission_term * (n_dot_l * INV_PI);
    }

    // `OpenPBR` `transmission_weight` is a **statistical mixture weight** between
    // the reflective base slab and the translucent slab, not an additive lobe;
    // mixing keeps both slabs inside one energy budget. `transmission == 0`
    // leaves the base untouched bit-for-bit.
    var reflected = base * (coat_attenuation * (1.0 - transmission))
        + coat_specular
        + transmission_term;
    var direct_term = base * (coat_attenuation * (1.0 - transmission)) + coat_specular;
    let sheen = clamp(surface.sheen, 0.0, 1.0);
    if sheen > 0.0 {
        let fuzz_roughness = clamp(surface.fuzz_roughness, MIN_ROUGHNESS, 1.0);
        let albedo = principled_fuzz_albedo_lookup(n_dot_v, fuzz_roughness);
        let substrate_scale = clamp(1.0 - sheen * albedo, 0.0, 1.0);
        let fuzz = sheen
            * principled_distribution_charlie(fuzz_roughness, n_dot_h)
            * principled_visibility_ashikhmin(n_dot_v, n_dot_l)
            * n_dot_l;
        direct_term = direct_term * substrate_scale + surface.fuzz_color * fuzz;
        transmission_term = transmission_term * substrate_scale;
        reflected = direct_term + transmission_term;
    }

    let radiance = reflected * light.illuminance;
    let visibility = clamp(light.visibility, 0.0, 1.0);
    let direct = direct_term * light.illuminance * visibility;
    let transmitted = transmission_term * light.illuminance * visibility;
    return PrincipledDirectChannels(radiance * visibility + surface.emissive, direct, transmitted);
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
    let f0 = mix(dielectric_f0(surface), surface.base_color, metallic);

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
