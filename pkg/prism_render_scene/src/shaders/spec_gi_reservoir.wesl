//! Glossy specular ReSTIR — directional GGX-lobe reservoir reuse (GPU twin of
//! the CPU golden `prism_render_shading::gi::spec_gi::glossy_reservoir`).
//!
//! Diffuse ReSTIR resamples a view-independent luminance×geometry target, so a
//! neighbour's radiance can be reused almost verbatim. Glossy specular is
//! different: a stored incident radiance only contributes along a direction the
//! destination's GGX lobe actually values, which depends on the view vector and
//! the roughness. This module swaps the scalar target for a *directional
//! GGX-lobe* target (`p̂ = luminance(f_spec · cos · L)`), re-targeting a
//! neighbour's incoming radiance onto the current pixel's view/roughness instead
//! of copying it blindly. It layers glossy-specific target + reuse-weight logic
//! on top of the generic weighted-reservoir machinery (confidence weights, GRIS
//! merge, M-cap, unbiased contribution weight `W`).
//!
//! Conventions:
//! * `SpecrGlossyPoint` fixes the destination: world position, unit normal, unit
//!   view, perceptual roughness, signed anisotropy, RGB `f0`.
//! * The reservoir payload is `SpecrGiSample`: the secondary sample point and
//!   its outgoing radiance; the incident direction is reconstructed as
//!   `normalize(sample_point - visible_point)` (a reconnection shift).
//! * Reuse is gated by `specr_roughness_reuse_weight` and the temporal M-cap is
//!   tightened for sharp surfaces by `specr_roughness_confidence_cap`.
//!
//! Rust's `Option<GiSample>` is modelled with a `has_sample: u32` flag. WGSL has
//! no `isFinite`; `specr_is_finite(x) = (x - x) == 0.0` reproduces Rust's
//! `f32::is_finite` exactly (true for every finite value, false for ±inf/NaN).
//! Every helper is transcribed op-for-op from the golden so
//! `spec_gi/reservoir_tests.rs` can assert numeric parity to `1e-5`.
//!
//! References: Bitterli et al. 2020 (*Spatiotemporal reservoir resampling*);
//! Lin et al. 2022 (*GRIS*); Heitz 2014/2018 (GGX VNDF + Smith).

const SPECR_PI: f32 = 3.1415926535897932;
const SPECR_MIN_ALPHA: f32 = 1.0e-3;
const SPECR_LUMA_R: f32 = 0.212639;
const SPECR_LUMA_G: f32 = 0.715169;
const SPECR_LUMA_B: f32 = 0.072192;

// Reproduces Rust `f32::is_finite`: for a finite x, `x - x == 0`; for ±inf/NaN
// the subtraction is NaN, which compares unequal to 0.
fn specr_is_finite(x: f32) -> bool {
    return (x - x) == 0.0;
}

fn specr_is_finite3(v: vec3<f32>) -> bool {
    return specr_is_finite(v.x) && specr_is_finite(v.y) && specr_is_finite(v.z);
}

// `v.normalize_or_zero()`.
fn specr_normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let l2 = dot(v, v);
    if l2 > 0.0 {
        return v / sqrt(l2);
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// Rec. 709 relative luminance, negative channels clamped to zero. Mirrors
// `screen_probe::restir::luminance`.
fn specr_luminance(rgb: vec3<f32>) -> f32 {
    return SPECR_LUMA_R * max(rgb.x, 0.0)
        + SPECR_LUMA_G * max(rgb.y, 0.0)
        + SPECR_LUMA_B * max(rgb.z, 0.0);
}

struct SpecrBasis {
    t: vec3<f32>,
    b: vec3<f32>,
}

// Duff et al. 2017 branchless orthonormal basis. Mirrors
// `gi::sample::mapping::orthonormal_basis`.
fn specr_orthonormal_basis(n: vec3<f32>) -> SpecrBasis {
    let sign = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    var out: SpecrBasis;
    out.t = vec3<f32>(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    out.b = vec3<f32>(b, sign + n.y * n.y * a, -n.y);
    return out;
}

// --- GGX lobe primitives (op-for-op copies of the subset `ggx_brdf` needs;
// kept local so this shader stays self-contained for the compile test) --------

fn specr_roughness_to_alpha(roughness: f32) -> f32 {
    let r = clamp(roughness, 0.0, 1.0);
    return max(r * r, SPECR_MIN_ALPHA);
}

fn specr_roughness_to_alpha_anisotropic(roughness: f32, anisotropy: f32) -> vec2<f32> {
    let alpha = specr_roughness_to_alpha(roughness);
    let aniso = clamp(anisotropy, -1.0, 1.0);
    let aspect = sqrt(max(1.0 - 0.9 * aniso, 1.0e-4));
    let ax = max(alpha / aspect, SPECR_MIN_ALPHA);
    let ay = max(alpha * aspect, SPECR_MIN_ALPHA);
    return vec2<f32>(ax, ay);
}

fn specr_ndf_ggx_anisotropic(h: vec3<f32>, alpha_x: f32, alpha_y: f32) -> f32 {
    if h.z <= 0.0 {
        return 0.0;
    }
    let ax = max(alpha_x, SPECR_MIN_ALPHA);
    let ay = max(alpha_y, SPECR_MIN_ALPHA);
    let t = h.x / ax;
    let b = h.y / ay;
    let s = t * t + b * b + h.z * h.z;
    let d = 1.0 / (SPECR_PI * ax * ay * max(s * s, 1.0e-20));
    if specr_is_finite(d) {
        return max(d, 0.0);
    }
    return 0.0;
}

fn specr_smith_lambda_anisotropic(w: vec3<f32>, alpha_x: f32, alpha_y: f32) -> f32 {
    let cz = clamp(abs(w.z), 1.0e-6, 1.0);
    let ax = max(alpha_x, SPECR_MIN_ALPHA);
    let ay = max(alpha_y, SPECR_MIN_ALPHA);
    let num = (ax * w.x) * (ax * w.x) + (ay * w.y) * (ay * w.y);
    let tan2 = num / (cz * cz);
    let lambda = 0.5 * (-1.0 + sqrt(max(1.0 + tan2, 0.0)));
    if specr_is_finite(lambda) {
        return max(lambda, 0.0);
    }
    return 0.0;
}

fn specr_smith_g2_anisotropic(wo: vec3<f32>, wi: vec3<f32>, alpha_x: f32, alpha_y: f32) -> f32 {
    if wo.z <= 0.0 || wi.z <= 0.0 {
        return 0.0;
    }
    let g = 1.0
        / (1.0
            + specr_smith_lambda_anisotropic(wo, alpha_x, alpha_y)
            + specr_smith_lambda_anisotropic(wi, alpha_x, alpha_y));
    return clamp(g, 0.0, 1.0);
}

fn specr_fresnel_schlick(f0: vec3<f32>, cos_theta: f32) -> vec3<f32> {
    let c = max(1.0 - clamp(cos_theta, 0.0, 1.0), 0.0);
    let c5 = (c * c) * (c * c) * c;
    let f = f0 + (vec3<f32>(1.0) - f0) * c5;
    return vec3<f32>(clamp(f.x, 0.0, 1.0), clamp(f.y, 0.0, 1.0), clamp(f.z, 0.0, 1.0));
}

fn specr_ggx_brdf(wo: vec3<f32>, wi: vec3<f32>, alpha_x: f32, alpha_y: f32, f0: vec3<f32>) -> vec3<f32> {
    if wo.z <= 0.0 || wi.z <= 0.0 {
        return vec3<f32>(0.0);
    }
    let h = specr_normalize_or_zero(wo + wi);
    if dot(h, h) == 0.0 {
        return vec3<f32>(0.0);
    }
    let d = specr_ndf_ggx_anisotropic(h, alpha_x, alpha_y);
    let g = specr_smith_g2_anisotropic(wo, wi, alpha_x, alpha_y);
    let f = specr_fresnel_schlick(f0, max(dot(wo, h), 0.0));
    let denom = max(4.0 * wo.z * wi.z, 1.0e-6);
    let scale = d * g / denom;
    let v = f * scale;
    if specr_is_finite3(v) {
        return max(v, vec3<f32>(0.0));
    }
    return vec3<f32>(0.0);
}

// --- Shading point / sample / reservoir payloads -----------------------------

struct SpecrGiSample {
    visible_point: vec3<f32>,
    sample_point: vec3<f32>,
    radiance: vec3<f32>,
}

struct SpecrGlossyPoint {
    position: vec3<f32>,
    normal: vec3<f32>,
    view: vec3<f32>,
    roughness: f32,
    anisotropy: f32,
    f0: vec3<f32>,
}

struct SpecrReservoir {
    sample: SpecrGiSample,
    w_sum: f32,
    m: f32,
    w: f32,
    has_sample: u32,
}

fn specr_reservoir_new() -> SpecrReservoir {
    var r: SpecrReservoir;
    r.sample = SpecrGiSample(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
    r.w_sum = 0.0;
    r.m = 0.0;
    r.w = 0.0;
    r.has_sample = 0u;
    return r;
}

// Projects a world-space direction into the point's local shading frame
// (`+Z` = normal). Mirrors `GlossyShadingPoint::local_dir`.
fn specr_local_dir(point: SpecrGlossyPoint, world: vec3<f32>) -> vec3<f32> {
    let n = point.normal;
    if dot(n, n) <= 0.0 {
        return vec3<f32>(0.0);
    }
    let basis = specr_orthonormal_basis(n);
    return vec3<f32>(dot(world, basis.t), dot(world, basis.b), dot(world, n));
}

// Glossy lobe throughput `f_spec · (n·wi)` towards the sample point.
fn specr_glossy_lobe_throughput(point: SpecrGlossyPoint, sample: SpecrGiSample) -> vec3<f32> {
    let to_sample = sample.sample_point - sample.visible_point;
    let dist2 = dot(to_sample, to_sample);
    if dist2 <= 1.0e-12 {
        return vec3<f32>(0.0);
    }
    let wi_world = to_sample / sqrt(dist2);
    let wo = specr_local_dir(point, point.view);
    let wi = specr_local_dir(point, wi_world);
    if wo.z <= 0.0 || wi.z <= 0.0 {
        return vec3<f32>(0.0);
    }
    let aa = specr_roughness_to_alpha_anisotropic(point.roughness, point.anisotropy);
    let f = specr_ggx_brdf(wo, wi, aa.x, aa.y, point.f0);
    let v = f * wi.z;
    if specr_is_finite3(v) {
        return max(v, vec3<f32>(0.0));
    }
    return vec3<f32>(0.0);
}

// Glossy lobe throughput for an explicit unit world incident direction.
fn specr_glossy_lobe_throughput_dir(point: SpecrGlossyPoint, wi_world: vec3<f32>) -> vec3<f32> {
    let wo = specr_local_dir(point, point.view);
    let wi = specr_local_dir(point, wi_world);
    if wo.z <= 0.0 || wi.z <= 0.0 {
        return vec3<f32>(0.0);
    }
    let aa = specr_roughness_to_alpha_anisotropic(point.roughness, point.anisotropy);
    let f = specr_ggx_brdf(wo, wi, aa.x, aa.y, point.f0);
    let v = f * wi.z;
    if specr_is_finite3(v) {
        return max(v, vec3<f32>(0.0));
    }
    return vec3<f32>(0.0);
}

// Directional glossy resampling target `p̂ = luminance(f_spec · cos · L)`.
fn specr_glossy_target_function(point: SpecrGlossyPoint, sample: SpecrGiSample) -> f32 {
    let throughput = specr_glossy_lobe_throughput(point, sample);
    let contrib = throughput * sample.radiance;
    let t = specr_luminance(contrib);
    if specr_is_finite(t) {
        return max(t, 0.0);
    }
    return 0.0;
}

// Similarity weight in [0,1] combining a Gaussian on the roughness gap with a
// cosine power on the normal gap.
fn specr_roughness_reuse_weight(
    dst_roughness: f32,
    src_roughness: f32,
    dst_normal: vec3<f32>,
    src_normal: vec3<f32>,
    sigma_roughness: f32,
) -> f32 {
    let sigma = max(sigma_roughness, 1.0e-3);
    let dr = (clamp(dst_roughness, 0.0, 1.0) - clamp(src_roughness, 0.0, 1.0)) / sigma;
    let rough_w = exp(-0.5 * dr * dr);

    let n_dst = specr_normalize_or_zero(dst_normal);
    let n_src = specr_normalize_or_zero(src_normal);
    if dot(n_dst, n_dst) == 0.0 || dot(n_src, n_src) == 0.0 {
        return 0.0;
    }
    let cos_n = clamp(dot(n_dst, n_src), 0.0, 1.0);
    let exponent = 8.0 + 56.0 * (1.0 - clamp(dst_roughness, 0.0, 1.0));
    let normal_w = pow(cos_n, exponent);

    let w = rough_w * normal_w;
    if specr_is_finite(w) {
        return clamp(w, 0.0, 1.0);
    }
    return 0.0;
}

// Roughness-dependent temporal confidence cap in [1, base_cap].
fn specr_roughness_confidence_cap(roughness: f32, base_cap: f32) -> f32 {
    let base = max(base_cap, 1.0);
    let r = clamp(roughness, 0.0, 1.0);
    let cap = 1.0 + (base - 1.0) * r;
    return clamp(cap, 1.0, base);
}

// --- Weighted-reservoir machinery (mirrors `screen_probe::restir::Reservoir`) -

fn specr_update(r: ptr<function, SpecrReservoir>, sample: SpecrGiSample, weight: f32, rng_uniform: f32) -> bool {
    if !specr_is_finite(weight) || weight <= 0.0 {
        return false;
    }
    (*r).w_sum = (*r).w_sum + weight;
    (*r).m = (*r).m + 1.0;
    let u = clamp(rng_uniform, 0.0, 1.0);
    if u * (*r).w_sum <= weight {
        (*r).sample = sample;
        (*r).has_sample = 1u;
        return true;
    }
    return false;
}

fn specr_merge(r: ptr<function, SpecrReservoir>, other: SpecrReservoir, other_target_pdf: f32, rng_uniform: f32) -> bool {
    if !specr_is_finite(other.m) || other.m <= 0.0 {
        return false;
    }
    (*r).m = (*r).m + other.m;
    let rw = other.m * max(other_target_pdf, 0.0) * max(other.w, 0.0);
    if !specr_is_finite(rw) || rw <= 0.0 {
        return false;
    }
    (*r).w_sum = (*r).w_sum + rw;
    let u = clamp(rng_uniform, 0.0, 1.0);
    if u * (*r).w_sum <= rw {
        (*r).sample = other.sample;
        (*r).has_sample = other.has_sample;
        return true;
    }
    return false;
}

fn specr_cap_confidence(r: ptr<function, SpecrReservoir>, max_m: f32) {
    if max_m >= 0.0 && (*r).m > max_m {
        (*r).m = max_m;
    }
}

fn specr_finalize_weight(r: ptr<function, SpecrReservoir>, target_pdf: f32) {
    if (*r).has_sample == 0u
        || (*r).m <= 0.0
        || !specr_is_finite((*r).w_sum)
        || !specr_is_finite(target_pdf)
        || target_pdf <= 0.0 {
        (*r).w = 0.0;
        return;
    }
    let w = ((*r).w_sum / (*r).m) / target_pdf;
    if specr_is_finite(w) && w >= 0.0 {
        (*r).w = w;
    } else {
        (*r).w = 0.0;
    }
}

// --- Glossy reservoir operations ---------------------------------------------

fn specr_stream_glossy_candidate(
    r: ptr<function, SpecrReservoir>,
    point: SpecrGlossyPoint,
    sample: SpecrGiSample,
    source_pdf: f32,
    rng_uniform: f32,
) -> bool {
    if !specr_is_finite(source_pdf) || source_pdf <= 0.0 {
        return false;
    }
    let p_hat = specr_glossy_target_function(point, sample);
    let weight = p_hat / source_pdf;
    return specr_update(r, sample, weight, rng_uniform);
}

fn specr_merge_glossy(
    canonical: ptr<function, SpecrReservoir>,
    neighbour: SpecrReservoir,
    point: SpecrGlossyPoint,
    neighbour_roughness: f32,
    neighbour_normal: vec3<f32>,
    sigma_roughness: f32,
    rng_uniform: f32,
) -> bool {
    var shifted_pdf = 0.0;
    if neighbour.has_sample != 0u {
        shifted_pdf = specr_glossy_target_function(point, neighbour.sample);
    }
    let reuse = specr_roughness_reuse_weight(
        point.roughness,
        neighbour_roughness,
        point.normal,
        neighbour_normal,
        sigma_roughness,
    );
    return specr_merge(canonical, neighbour, shifted_pdf * reuse, rng_uniform);
}

fn specr_finalize_glossy(r: ptr<function, SpecrReservoir>, point: SpecrGlossyPoint) -> f32 {
    var p_hat = 0.0;
    if (*r).has_sample != 0u {
        p_hat = specr_glossy_target_function(point, (*r).sample);
    }
    specr_finalize_weight(r, p_hat);
    return (*r).w;
}

fn specr_glossy_contribution(r: SpecrReservoir, point: SpecrGlossyPoint) -> vec3<f32> {
    if r.has_sample == 0u {
        return vec3<f32>(0.0);
    }
    let w = r.w;
    if !specr_is_finite(w) || w <= 0.0 {
        return vec3<f32>(0.0);
    }
    let throughput = specr_glossy_lobe_throughput(point, r.sample);
    let v = throughput * r.sample.radiance * w;
    if specr_is_finite3(v) {
        return max(v, vec3<f32>(0.0));
    }
    return vec3<f32>(0.0);
}

// --- Dead-code anchor --------------------------------------------------------
// Folds every helper so naga keeps them live in the compile test.

struct SpecrProbe {
    value: vec4<f32>,
}

@group(0) @binding(0) var<storage, read_write> specr_probe: SpecrProbe;

@compute @workgroup_size(1, 1, 1)
fn spec_gi_reservoir_probe(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x != 0u {
        return;
    }
    var point: SpecrGlossyPoint;
    point.position = vec3<f32>(0.0);
    point.normal = vec3<f32>(0.0, 0.0, 1.0);
    point.view = specr_normalize_or_zero(vec3<f32>(0.0, 0.2, 1.0));
    point.roughness = 0.3;
    point.anisotropy = 0.1;
    point.f0 = vec3<f32>(0.04);

    var s: SpecrGiSample;
    s.visible_point = point.position;
    s.sample_point = vec3<f32>(0.1, 0.2, 2.0);
    s.radiance = vec3<f32>(1.0, 0.8, 0.5);

    var r = specr_reservoir_new();
    let picked = specr_stream_glossy_candidate(&r, point, s, 0.7, 0.1);
    let cap = specr_roughness_confidence_cap(point.roughness, 16.0);
    specr_cap_confidence(&r, cap);

    var neighbour = specr_reservoir_new();
    let seeded = specr_stream_glossy_candidate(&neighbour, point, s, 1.0, 0.0);
    let wfin_n = specr_finalize_glossy(&neighbour, point);
    let reuse = specr_roughness_reuse_weight(point.roughness, 0.35, point.normal, point.normal, 0.1);
    let merged = specr_merge_glossy(&r, neighbour, point, 0.35, point.normal, 0.1, 0.5);
    let w = specr_finalize_glossy(&r, point);
    let contrib = specr_glossy_contribution(r, point);
    let tput = specr_glossy_lobe_throughput_dir(point, specr_normalize_or_zero(vec3<f32>(0.0, 0.1, 1.0)));
    let tgt = specr_glossy_target_function(point, s);

    let flags = select(0.0, 1.0, picked) + select(0.0, 1.0, seeded) + select(0.0, 1.0, merged);
    specr_probe.value = vec4<f32>(contrib + tput, w + tgt + reuse + cap + wfin_n + flags);
}
