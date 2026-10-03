//! BRDF / light multiple-importance sampling for glossy reflection (GPU twin of
//! the CPU golden `prism_render_shading::gi::spec_gi::brdf_mis`).
//!
//! A glossy reflection integral `∫ f_spec(wo, wi) L(wi) (n·wi) dω` is estimated
//! by combining two complementary samplers — BRDF sampling (sharp lobes) and
//! light sampling (small bright lights) — and blending them with multiple
//! importance sampling (MIS, Veach & Guibas 1995). This module ports the MIS
//! weights (balance + power `β = 2`, with sample-count-aware variants), the
//! BRDF solid-angle pdf of a sampled direction (`specm_brdf_pdf_glossy`, reusing
//! the GGX VNDF reflection pdf), the area→solid-angle Jacobian, the single-
//! sample `f·cos·L / pdf` estimator, and the combined one-BRDF + one-light MIS
//! estimator `specm_mis_estimator_glossy`.
//!
//! Conventions:
//! * All pdfs are expressed w.r.t. **solid angle** at the shading point; the
//!   only measure change lives in `specm_area_to_solid_angle_pdf`.
//! * Weights are deterministic and defend against degeneracy: a zero/non-finite
//!   denominator yields `0`, and paired weights sum to `1` whenever at least one
//!   pdf is positive (partition of unity).
//!
//! WGSL has no `isFinite`; `specm_is_finite(x) = (x - x) == 0.0` reproduces
//! Rust's `f32::is_finite` exactly (true for every finite value, false for
//! ±inf/NaN). Every helper is transcribed op-for-op from the golden so
//! `spec_gi/brdf_mis_tests.rs` can assert numeric parity to `1e-5`.
//!
//! References: Veach & Guibas 1995 (*Optimally Combining Sampling Techniques*);
//! Heitz 2014/2018 (GGX VNDF + Smith).

const SPECM_PI: f32 = 3.1415926535897932;
const SPECM_MIN_ALPHA: f32 = 1.0e-3;
const SPECM_LUMA_R: f32 = 0.212639;
const SPECM_LUMA_G: f32 = 0.715169;
const SPECM_LUMA_B: f32 = 0.072192;

// Reproduces Rust `f32::is_finite`.
fn specm_is_finite(x: f32) -> bool {
    return (x - x) == 0.0;
}

fn specm_is_finite3(v: vec3<f32>) -> bool {
    return specm_is_finite(v.x) && specm_is_finite(v.y) && specm_is_finite(v.z);
}

fn specm_normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let l2 = dot(v, v);
    if l2 > 0.0 {
        return v / sqrt(l2);
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// Rec. 709 relative luminance, negative channels clamped to zero.
fn specm_luminance(rgb: vec3<f32>) -> f32 {
    return SPECM_LUMA_R * max(rgb.x, 0.0)
        + SPECM_LUMA_G * max(rgb.y, 0.0)
        + SPECM_LUMA_B * max(rgb.z, 0.0);
}

struct SpecmBasis {
    t: vec3<f32>,
    b: vec3<f32>,
}

// Duff et al. 2017 branchless orthonormal basis.
fn specm_orthonormal_basis(n: vec3<f32>) -> SpecmBasis {
    let sign = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    var out: SpecmBasis;
    out.t = vec3<f32>(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    out.b = vec3<f32>(b, sign + n.y * n.y * a, -n.y);
    return out;
}

// --- GGX lobe primitives (self-contained subset) ----------------------------

fn specm_roughness_to_alpha(roughness: f32) -> f32 {
    let r = clamp(roughness, 0.0, 1.0);
    return max(r * r, SPECM_MIN_ALPHA);
}

fn specm_roughness_to_alpha_anisotropic(roughness: f32, anisotropy: f32) -> vec2<f32> {
    let alpha = specm_roughness_to_alpha(roughness);
    let aniso = clamp(anisotropy, -1.0, 1.0);
    let aspect = sqrt(max(1.0 - 0.9 * aniso, 1.0e-4));
    let ax = max(alpha / aspect, SPECM_MIN_ALPHA);
    let ay = max(alpha * aspect, SPECM_MIN_ALPHA);
    return vec2<f32>(ax, ay);
}

fn specm_ndf_ggx_anisotropic(h: vec3<f32>, alpha_x: f32, alpha_y: f32) -> f32 {
    if h.z <= 0.0 {
        return 0.0;
    }
    let ax = max(alpha_x, SPECM_MIN_ALPHA);
    let ay = max(alpha_y, SPECM_MIN_ALPHA);
    let t = h.x / ax;
    let b = h.y / ay;
    let s = t * t + b * b + h.z * h.z;
    let d = 1.0 / (SPECM_PI * ax * ay * max(s * s, 1.0e-20));
    if specm_is_finite(d) {
        return max(d, 0.0);
    }
    return 0.0;
}

fn specm_smith_lambda_anisotropic(w: vec3<f32>, alpha_x: f32, alpha_y: f32) -> f32 {
    let cz = clamp(abs(w.z), 1.0e-6, 1.0);
    let ax = max(alpha_x, SPECM_MIN_ALPHA);
    let ay = max(alpha_y, SPECM_MIN_ALPHA);
    let num = (ax * w.x) * (ax * w.x) + (ay * w.y) * (ay * w.y);
    let tan2 = num / (cz * cz);
    let lambda = 0.5 * (-1.0 + sqrt(max(1.0 + tan2, 0.0)));
    if specm_is_finite(lambda) {
        return max(lambda, 0.0);
    }
    return 0.0;
}

fn specm_smith_g2_anisotropic(wo: vec3<f32>, wi: vec3<f32>, alpha_x: f32, alpha_y: f32) -> f32 {
    if wo.z <= 0.0 || wi.z <= 0.0 {
        return 0.0;
    }
    let g = 1.0
        / (1.0
            + specm_smith_lambda_anisotropic(wo, alpha_x, alpha_y)
            + specm_smith_lambda_anisotropic(wi, alpha_x, alpha_y));
    return clamp(g, 0.0, 1.0);
}

fn specm_fresnel_schlick(f0: vec3<f32>, cos_theta: f32) -> vec3<f32> {
    let c = max(1.0 - clamp(cos_theta, 0.0, 1.0), 0.0);
    let c5 = (c * c) * (c * c) * c;
    let f = f0 + (vec3<f32>(1.0) - f0) * c5;
    return vec3<f32>(clamp(f.x, 0.0, 1.0), clamp(f.y, 0.0, 1.0), clamp(f.z, 0.0, 1.0));
}

fn specm_ggx_brdf(wo: vec3<f32>, wi: vec3<f32>, alpha_x: f32, alpha_y: f32, f0: vec3<f32>) -> vec3<f32> {
    if wo.z <= 0.0 || wi.z <= 0.0 {
        return vec3<f32>(0.0);
    }
    let h = specm_normalize_or_zero(wo + wi);
    if dot(h, h) == 0.0 {
        return vec3<f32>(0.0);
    }
    let d = specm_ndf_ggx_anisotropic(h, alpha_x, alpha_y);
    let g = specm_smith_g2_anisotropic(wo, wi, alpha_x, alpha_y);
    let f = specm_fresnel_schlick(f0, max(dot(wo, h), 0.0));
    let denom = max(4.0 * wo.z * wi.z, 1.0e-6);
    let scale = d * g / denom;
    let v = f * scale;
    if specm_is_finite3(v) {
        return max(v, vec3<f32>(0.0));
    }
    return vec3<f32>(0.0);
}

// Solid-angle pdf of the VNDF half vector `h` for local view `wo`.
fn specm_vndf_pdf_h(wo: vec3<f32>, h: vec3<f32>, alpha_x: f32, alpha_y: f32) -> f32 {
    if wo.z <= 0.0 || h.z <= 0.0 {
        return 0.0;
    }
    let v_dot_h = max(dot(wo, h), 0.0);
    if v_dot_h <= 0.0 {
        return 0.0;
    }
    let g1 = 1.0 / (1.0 + specm_smith_lambda_anisotropic(wo, alpha_x, alpha_y));
    let d = specm_ndf_ggx_anisotropic(h, alpha_x, alpha_y);
    let pdf = g1 * v_dot_h * d / max(wo.z, 1.0e-6);
    return max(pdf, 0.0);
}

// Solid-angle pdf of a VNDF-sampled reflected direction `wi`.
fn specm_vndf_pdf_reflect(wo: vec3<f32>, wi: vec3<f32>, alpha_x: f32, alpha_y: f32) -> f32 {
    if wo.z <= 0.0 || wi.z <= 0.0 {
        return 0.0;
    }
    let h = specm_normalize_or_zero(wo + wi);
    if dot(h, h) == 0.0 {
        return 0.0;
    }
    let v_dot_h = max(dot(wo, h), 0.0);
    if v_dot_h <= 1.0e-8 {
        return 0.0;
    }
    let pdf = specm_vndf_pdf_h(wo, h, alpha_x, alpha_y) / (4.0 * v_dot_h);
    return max(pdf, 0.0);
}

// --- Shading point + direction sample ----------------------------------------

struct SpecmGlossyPoint {
    position: vec3<f32>,
    normal: vec3<f32>,
    view: vec3<f32>,
    roughness: f32,
    anisotropy: f32,
    f0: vec3<f32>,
}

struct SpecmDirectionSample {
    wi: vec3<f32>,
    radiance: vec3<f32>,
    pdf: f32,
}

// Projects a world-space direction into the point's local shading frame.
fn specm_local_dir(point: SpecmGlossyPoint, world: vec3<f32>) -> vec3<f32> {
    let n = point.normal;
    if dot(n, n) <= 0.0 {
        return vec3<f32>(0.0);
    }
    let basis = specm_orthonormal_basis(n);
    return vec3<f32>(dot(world, basis.t), dot(world, basis.b), dot(world, n));
}

// Glossy lobe throughput `f_spec · (n·wi)` for an explicit unit world incident
// direction. Mirrors `glossy_lobe_throughput_dir`.
fn specm_glossy_lobe_throughput_dir(point: SpecmGlossyPoint, wi_world: vec3<f32>) -> vec3<f32> {
    let wo = specm_local_dir(point, point.view);
    let wi = specm_local_dir(point, wi_world);
    if wo.z <= 0.0 || wi.z <= 0.0 {
        return vec3<f32>(0.0);
    }
    let aa = specm_roughness_to_alpha_anisotropic(point.roughness, point.anisotropy);
    let f = specm_ggx_brdf(wo, wi, aa.x, aa.y, point.f0);
    let v = f * wi.z;
    if specm_is_finite3(v) {
        return max(v, vec3<f32>(0.0));
    }
    return vec3<f32>(0.0);
}

// --- MIS weights -------------------------------------------------------------

// Balance-heuristic MIS weight `w_a = pdf_a / (pdf_a + pdf_b)`.
fn specm_balance_heuristic_pair(pdf_a: f32, pdf_b: f32) -> f32 {
    let a = max(pdf_a, 0.0);
    let b = max(pdf_b, 0.0);
    if a <= 0.0 {
        return 0.0;
    }
    let denom = a + b;
    if denom <= 0.0 || !specm_is_finite(denom) {
        return 0.0;
    }
    return clamp(a / denom, 0.0, 1.0);
}

// Power-heuristic (`β = 2`) MIS weight `w_a = pdf_a^2 / (pdf_a^2 + pdf_b^2)`.
fn specm_power_heuristic(pdf_a: f32, pdf_b: f32) -> f32 {
    let a = max(pdf_a, 0.0);
    let b = max(pdf_b, 0.0);
    if a <= 0.0 {
        return 0.0;
    }
    let a2 = a * a;
    let b2 = b * b;
    let denom = a2 + b2;
    if denom <= 0.0 || !specm_is_finite(denom) {
        return 0.0;
    }
    return clamp(a2 / denom, 0.0, 1.0);
}

// Sample-count-aware power heuristic for `n_a`/`n_b` samples.
fn specm_power_heuristic_counts(n_a: f32, pdf_a: f32, n_b: f32, pdf_b: f32) -> f32 {
    let a = max(n_a, 0.0) * max(pdf_a, 0.0);
    let b = max(n_b, 0.0) * max(pdf_b, 0.0);
    return specm_power_heuristic(a, b);
}

// Sample-count-aware balance heuristic for `n_a`/`n_b` samples.
fn specm_balance_heuristic_counts(n_a: f32, pdf_a: f32, n_b: f32, pdf_b: f32) -> f32 {
    let a = max(n_a, 0.0) * max(pdf_a, 0.0);
    let b = max(n_b, 0.0) * max(pdf_b, 0.0);
    return specm_balance_heuristic_pair(a, b);
}

// Solid-angle pdf of a BRDF-sampled incident direction `wi_world` at `point`.
fn specm_brdf_pdf_glossy(point: SpecmGlossyPoint, wi_world: vec3<f32>) -> f32 {
    let wo = specm_local_dir(point, point.view);
    let wi = specm_local_dir(point, wi_world);
    let aa = specm_roughness_to_alpha_anisotropic(point.roughness, point.anisotropy);
    return specm_vndf_pdf_reflect(wo, wi, aa.x, aa.y);
}

// Converts an area-measure light pdf to a solid-angle pdf at the shading point:
// `pdf_ω = pdf_A · dist² / |cos_light|`.
fn specm_area_to_solid_angle_pdf(pdf_area: f32, dist: f32, cos_light: f32) -> f32 {
    let p = max(pdf_area, 0.0);
    let d2 = dist * dist;
    let c = abs(cos_light);
    if p <= 0.0 || !specm_is_finite(d2) || c <= 1.0e-6 {
        return 0.0;
    }
    let pdf = p * d2 / c;
    if specm_is_finite(pdf) {
        return max(pdf, 0.0);
    }
    return 0.0;
}

// Single-sample contribution of a direction sample: `f·cos·L / pdf`.
fn specm_brdf_sample_estimate(point: SpecmGlossyPoint, sample: SpecmDirectionSample) -> vec3<f32> {
    if !specm_is_finite(sample.pdf) || sample.pdf <= 0.0 {
        return vec3<f32>(0.0);
    }
    let throughput = specm_glossy_lobe_throughput_dir(point, specm_normalize_or_zero(sample.wi));
    let v = throughput * sample.radiance / sample.pdf;
    if specm_is_finite3(v) {
        return max(v, vec3<f32>(0.0));
    }
    return vec3<f32>(0.0);
}

// Combined one-BRDF-sample + one-light-sample MIS estimator for glossy
// reflection, each term power-heuristic weighted against the other technique's
// pdf for the same direction.
fn specm_mis_estimator_glossy(
    point: SpecmGlossyPoint,
    brdf_sample: SpecmDirectionSample,
    light_pdf_for_brdf_dir: f32,
    light_sample: SpecmDirectionSample,
    brdf_pdf_for_light_dir: f32,
) -> vec3<f32> {
    var out = vec3<f32>(0.0);

    if specm_is_finite(brdf_sample.pdf) && brdf_sample.pdf > 0.0 {
        let w = specm_power_heuristic(brdf_sample.pdf, light_pdf_for_brdf_dir);
        if w > 0.0 {
            out = out + specm_brdf_sample_estimate(point, brdf_sample) * w;
        }
    }

    if specm_is_finite(light_sample.pdf) && light_sample.pdf > 0.0 {
        let w = specm_power_heuristic(light_sample.pdf, brdf_pdf_for_light_dir);
        if w > 0.0 {
            out = out + specm_brdf_sample_estimate(point, light_sample) * w;
        }
    }

    if specm_is_finite3(out) {
        return max(out, vec3<f32>(0.0));
    }
    return vec3<f32>(0.0);
}

// Rec. 709 luminance of a glossy MIS estimate.
fn specm_mis_estimate_luminance(estimate: vec3<f32>) -> f32 {
    return specm_luminance(estimate);
}

// --- Dead-code anchor --------------------------------------------------------
// Folds every helper so naga keeps them live in the compile test.

struct SpecmProbe {
    value: vec4<f32>,
}

@group(0) @binding(0) var<storage, read_write> specm_probe: SpecmProbe;

@compute @workgroup_size(1, 1, 1)
fn spec_gi_brdf_mis_probe(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x != 0u {
        return;
    }
    var point: SpecmGlossyPoint;
    point.position = vec3<f32>(0.0);
    point.normal = vec3<f32>(0.0, 0.0, 1.0);
    point.view = specm_normalize_or_zero(vec3<f32>(0.0, 0.3, 0.954));
    point.roughness = 0.3;
    point.anisotropy = 0.0;
    point.f0 = vec3<f32>(0.5);

    let refl = specm_normalize_or_zero(vec3<f32>(0.0, -0.3, 0.954));
    let brdf_pdf = specm_brdf_pdf_glossy(point, refl);

    var brdf_sample: SpecmDirectionSample;
    brdf_sample.wi = refl;
    brdf_sample.radiance = vec3<f32>(2.0);
    brdf_sample.pdf = brdf_pdf;

    var light_sample: SpecmDirectionSample;
    light_sample.wi = refl;
    light_sample.radiance = vec3<f32>(2.0);
    light_sample.pdf = 1.5;

    let est = specm_mis_estimator_glossy(point, brdf_sample, 1.5, light_sample, brdf_pdf);
    let plain = specm_brdf_sample_estimate(point, brdf_sample);
    let lum = specm_mis_estimate_luminance(est);

    let wb = specm_balance_heuristic_pair(brdf_pdf, 1.5);
    let wp = specm_power_heuristic(brdf_pdf, 1.5);
    let wbc = specm_balance_heuristic_counts(1.0, brdf_pdf, 1.0, 1.5);
    let wpc = specm_power_heuristic_counts(1.0, brdf_pdf, 1.0, 1.5);
    let jac = specm_area_to_solid_angle_pdf(0.25, 2.0, 0.5);
    let tput = specm_glossy_lobe_throughput_dir(point, refl);

    specm_probe.value = vec4<f32>(est + plain + tput, lum + wb + wp + wbc + wpc + jac + brdf_pdf);
}
