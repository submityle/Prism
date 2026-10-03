//! WESL compile coverage and CPU-mirror parity for `spec_gi_brdf_mis.wesl`.
//!
//! The sandbox has no GPU, so the compile test drives the BRDF/light-MIS module
//! through the same `ShaderCache` / `wesl` pipeline the render world uses,
//! proving it parses and type-checks exactly as it will on device (the
//! `spec_gi_brdf_mis_probe` entry folds every helper so naga keeps them live).
//!
//! The parity tests transcribe each WESL helper op-for-op into Rust and assert
//! agreement with the CPU golden
//! ([`prism_render_shading::gi::spec_gi::brdf_mis`]) across a sweep of valid and
//! degenerate inputs — the balance/power MIS weights (and their sample-count
//! variants), the GGX VNDF BRDF pdf, the area→solid-angle Jacobian, the single-
//! sample `f·cos·L / pdf` estimator, and the combined one-BRDF + one-light MIS
//! estimator. WGSL's missing `isFinite` is modelled by `(x - x) == 0.0`,
//! matching the shader exactly.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_math::Vec3;
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};
use prism_render_shading::gi::spec_gi::brdf_mis as golden;
use prism_render_shading::gi::spec_gi::brdf_mis::DirectionSample;
use prism_render_shading::gi::spec_gi::ggx_lobe as golden_lobe;
use prism_render_shading::gi::spec_gi::glossy_reservoir::GlossyShadingPoint;

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("spec_gi brdf_mis shader is WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles `spec_gi_brdf_mis.wesl`, proving the BRDF/light-MIS module parses
/// and type-checks as it will in the render world.
#[test]
fn spec_gi_brdf_mis_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let brdf_mis = shader_id(0x5350_4543_5f47_495f_5252_4446_5f00_0003);
    cache.set_shader(
        brdf_mis,
        Shader::from_wesl(
            include_str!("../../shaders/spec_gi_brdf_mis.wesl"),
            "embedded://prism_render_scene/shaders/spec_gi_brdf_mis.wesl",
        ),
    );

    cache
        .get(0, brdf_mis, &[])
        .unwrap_or_else(|error| panic!("spec_gi_brdf_mis.wesl failed to compile: {error}"));
}

// --- Rust transcription of the WESL maths (op-for-op) -------------------------
// Floors, control flow and finite-guards mirror `spec_gi_brdf_mis.wesl`
// exactly; this is what the GPU evaluates, asserted equal to the golden below.

const MIN_ALPHA: f32 = 1.0e-3;
const PI: f32 = std::f32::consts::PI;
const LUMA_R: f32 = 0.212639;
const LUMA_G: f32 = 0.715169;
const LUMA_B: f32 = 0.072192;

#[inline]
fn is_finite(x: f32) -> bool {
    (x - x) == 0.0
}

#[inline]
fn is_finite3(v: Vec3) -> bool {
    is_finite(v.x) && is_finite(v.y) && is_finite(v.z)
}

#[allow(clippy::disallowed_methods)]
fn nz(v: Vec3) -> Vec3 {
    let l2 = v.dot(v);
    if l2 > 0.0 {
        v / l2.sqrt()
    } else {
        Vec3::ZERO
    }
}

fn luminance(rgb: Vec3) -> f32 {
    LUMA_R * rgb.x.max(0.0) + LUMA_G * rgb.y.max(0.0) + LUMA_B * rgb.z.max(0.0)
}

fn orthonormal_basis(n: Vec3) -> (Vec3, Vec3) {
    let sign = if n.z >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    let t = Vec3::new(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    let bt = Vec3::new(b, sign + n.y * n.y * a, -n.y);
    (t, bt)
}

fn roughness_to_alpha(roughness: f32) -> f32 {
    let r = roughness.clamp(0.0, 1.0);
    (r * r).max(MIN_ALPHA)
}

#[allow(clippy::disallowed_methods)]
fn roughness_to_alpha_anisotropic(roughness: f32, anisotropy: f32) -> (f32, f32) {
    let alpha = roughness_to_alpha(roughness);
    let aniso = anisotropy.clamp(-1.0, 1.0);
    let aspect = (1.0 - 0.9 * aniso).max(1.0e-4).sqrt();
    let ax = (alpha / aspect).max(MIN_ALPHA);
    let ay = (alpha * aspect).max(MIN_ALPHA);
    (ax, ay)
}

fn ndf_ggx_anisotropic(h: Vec3, alpha_x: f32, alpha_y: f32) -> f32 {
    if h.z <= 0.0 {
        return 0.0;
    }
    let ax = alpha_x.max(MIN_ALPHA);
    let ay = alpha_y.max(MIN_ALPHA);
    let t = h.x / ax;
    let b = h.y / ay;
    let s = t * t + b * b + h.z * h.z;
    let d = 1.0 / (PI * ax * ay * (s * s).max(1.0e-20));
    if is_finite(d) {
        d.max(0.0)
    } else {
        0.0
    }
}

#[allow(clippy::disallowed_methods)]
fn smith_lambda_anisotropic(w: Vec3, alpha_x: f32, alpha_y: f32) -> f32 {
    let cz = w.z.abs().clamp(1.0e-6, 1.0);
    let ax = alpha_x.max(MIN_ALPHA);
    let ay = alpha_y.max(MIN_ALPHA);
    let num = (ax * w.x) * (ax * w.x) + (ay * w.y) * (ay * w.y);
    let tan2 = num / (cz * cz);
    let lambda = 0.5 * (-1.0 + (1.0 + tan2).max(0.0).sqrt());
    if is_finite(lambda) {
        lambda.max(0.0)
    } else {
        0.0
    }
}

fn smith_g2_anisotropic(wo: Vec3, wi: Vec3, alpha_x: f32, alpha_y: f32) -> f32 {
    if wo.z <= 0.0 || wi.z <= 0.0 {
        return 0.0;
    }
    let g = 1.0
        / (1.0
            + smith_lambda_anisotropic(wo, alpha_x, alpha_y)
            + smith_lambda_anisotropic(wi, alpha_x, alpha_y));
    g.clamp(0.0, 1.0)
}

fn fresnel_schlick(f0: Vec3, cos_theta: f32) -> Vec3 {
    let c = (1.0 - cos_theta.clamp(0.0, 1.0)).max(0.0);
    let c5 = (c * c) * (c * c) * c;
    let f = f0 + (Vec3::ONE - f0) * c5;
    Vec3::new(
        f.x.clamp(0.0, 1.0),
        f.y.clamp(0.0, 1.0),
        f.z.clamp(0.0, 1.0),
    )
}

fn ggx_brdf(wo: Vec3, wi: Vec3, alpha_x: f32, alpha_y: f32, f0: Vec3) -> Vec3 {
    if wo.z <= 0.0 || wi.z <= 0.0 {
        return Vec3::ZERO;
    }
    let h = nz(wo + wi);
    if h.dot(h) == 0.0 {
        return Vec3::ZERO;
    }
    let d = ndf_ggx_anisotropic(h, alpha_x, alpha_y);
    let g = smith_g2_anisotropic(wo, wi, alpha_x, alpha_y);
    let f = fresnel_schlick(f0, wo.dot(h).max(0.0));
    let denom = (4.0 * wo.z * wi.z).max(1.0e-6);
    let scale = d * g / denom;
    let v = f * scale;
    if is_finite3(v) {
        v.max(Vec3::ZERO)
    } else {
        Vec3::ZERO
    }
}

fn vndf_pdf_h(wo: Vec3, h: Vec3, alpha_x: f32, alpha_y: f32) -> f32 {
    if wo.z <= 0.0 || h.z <= 0.0 {
        return 0.0;
    }
    let v_dot_h = wo.dot(h).max(0.0);
    if v_dot_h <= 0.0 {
        return 0.0;
    }
    let g1 = 1.0 / (1.0 + smith_lambda_anisotropic(wo, alpha_x, alpha_y));
    let d = ndf_ggx_anisotropic(h, alpha_x, alpha_y);
    let pdf = g1 * v_dot_h * d / wo.z.max(1.0e-6);
    pdf.max(0.0)
}

fn vndf_pdf_reflect(wo: Vec3, wi: Vec3, alpha_x: f32, alpha_y: f32) -> f32 {
    if wo.z <= 0.0 || wi.z <= 0.0 {
        return 0.0;
    }
    let h = nz(wo + wi);
    if h.dot(h) == 0.0 {
        return 0.0;
    }
    let v_dot_h = wo.dot(h).max(0.0);
    if v_dot_h <= 1.0e-8 {
        return 0.0;
    }
    let pdf = vndf_pdf_h(wo, h, alpha_x, alpha_y) / (4.0 * v_dot_h);
    pdf.max(0.0)
}

#[derive(Clone, Copy)]
struct MPoint {
    normal: Vec3,
    view: Vec3,
    roughness: f32,
    anisotropy: f32,
    f0: Vec3,
}

#[derive(Clone, Copy)]
struct MDirSample {
    wi: Vec3,
    radiance: Vec3,
    pdf: f32,
}

fn local_dir(point: &MPoint, world: Vec3) -> Vec3 {
    let n = point.normal;
    if n.dot(n) <= 0.0 {
        return Vec3::ZERO;
    }
    let (t, b) = orthonormal_basis(n);
    Vec3::new(world.dot(t), world.dot(b), world.dot(n))
}

#[allow(clippy::disallowed_methods)]
fn glossy_lobe_throughput_dir(point: &MPoint, wi_world: Vec3) -> Vec3 {
    let wo = local_dir(point, point.view);
    let wi = local_dir(point, wi_world);
    if wo.z <= 0.0 || wi.z <= 0.0 {
        return Vec3::ZERO;
    }
    let (ax, ay) = roughness_to_alpha_anisotropic(point.roughness, point.anisotropy);
    let f = ggx_brdf(wo, wi, ax, ay, point.f0);
    let v = f * wi.z;
    if is_finite3(v) {
        v.max(Vec3::ZERO)
    } else {
        Vec3::ZERO
    }
}

fn balance_heuristic_pair(pdf_a: f32, pdf_b: f32) -> f32 {
    let a = pdf_a.max(0.0);
    let b = pdf_b.max(0.0);
    if a <= 0.0 {
        return 0.0;
    }
    let denom = a + b;
    if denom <= 0.0 || !is_finite(denom) {
        return 0.0;
    }
    (a / denom).clamp(0.0, 1.0)
}

fn power_heuristic(pdf_a: f32, pdf_b: f32) -> f32 {
    let a = pdf_a.max(0.0);
    let b = pdf_b.max(0.0);
    if a <= 0.0 {
        return 0.0;
    }
    let a2 = a * a;
    let b2 = b * b;
    let denom = a2 + b2;
    if denom <= 0.0 || !is_finite(denom) {
        return 0.0;
    }
    (a2 / denom).clamp(0.0, 1.0)
}

fn power_heuristic_counts(n_a: f32, pdf_a: f32, n_b: f32, pdf_b: f32) -> f32 {
    let a = n_a.max(0.0) * pdf_a.max(0.0);
    let b = n_b.max(0.0) * pdf_b.max(0.0);
    power_heuristic(a, b)
}

fn balance_heuristic_counts(n_a: f32, pdf_a: f32, n_b: f32, pdf_b: f32) -> f32 {
    let a = n_a.max(0.0) * pdf_a.max(0.0);
    let b = n_b.max(0.0) * pdf_b.max(0.0);
    balance_heuristic_pair(a, b)
}

fn brdf_pdf_glossy(point: &MPoint, wi_world: Vec3) -> f32 {
    let wo = local_dir(point, point.view);
    let wi = local_dir(point, wi_world);
    let (ax, ay) = roughness_to_alpha_anisotropic(point.roughness, point.anisotropy);
    vndf_pdf_reflect(wo, wi, ax, ay)
}

fn area_to_solid_angle_pdf(pdf_area: f32, dist: f32, cos_light: f32) -> f32 {
    let p = pdf_area.max(0.0);
    let d2 = dist * dist;
    let c = cos_light.abs();
    if p <= 0.0 || !is_finite(d2) || c <= 1.0e-6 {
        return 0.0;
    }
    let pdf = p * d2 / c;
    if is_finite(pdf) {
        pdf.max(0.0)
    } else {
        0.0
    }
}

#[allow(clippy::disallowed_methods)]
fn brdf_sample_estimate(point: &MPoint, sample: &MDirSample) -> Vec3 {
    if !is_finite(sample.pdf) || sample.pdf <= 0.0 {
        return Vec3::ZERO;
    }
    let throughput = glossy_lobe_throughput_dir(point, nz(sample.wi));
    let v = throughput * sample.radiance / sample.pdf;
    if is_finite3(v) {
        v.max(Vec3::ZERO)
    } else {
        Vec3::ZERO
    }
}

fn mis_estimator_glossy(
    point: &MPoint,
    brdf_sample: &MDirSample,
    light_pdf_for_brdf_dir: f32,
    light_sample: &MDirSample,
    brdf_pdf_for_light_dir: f32,
) -> Vec3 {
    let mut out = Vec3::ZERO;

    if is_finite(brdf_sample.pdf) && brdf_sample.pdf > 0.0 {
        let w = power_heuristic(brdf_sample.pdf, light_pdf_for_brdf_dir);
        if w > 0.0 {
            out += brdf_sample_estimate(point, brdf_sample) * w;
        }
    }

    if is_finite(light_sample.pdf) && light_sample.pdf > 0.0 {
        let w = power_heuristic(light_sample.pdf, brdf_pdf_for_light_dir);
        if w > 0.0 {
            out += brdf_sample_estimate(point, light_sample) * w;
        }
    }

    if is_finite3(out) {
        out.max(Vec3::ZERO)
    } else {
        Vec3::ZERO
    }
}

fn mis_estimate_luminance(estimate: Vec3) -> f32 {
    luminance(estimate)
}

// --- parity harness ----------------------------------------------------------

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1.0e-5 * (1.0 + a.abs().max(b.abs()))
}

fn close3(a: Vec3, b: Vec3) -> bool {
    close(a.x, b.x) && close(a.y, b.y) && close(a.z, b.z)
}

/// Builds a matched (mirror, golden) destination point from normalised inputs so
/// `local_dir`/alpha derivations are identical on both sides.
fn make_points(
    normal: Vec3,
    view: Vec3,
    roughness: f32,
    anisotropy: f32,
    f0: Vec3,
) -> (MPoint, GlossyShadingPoint) {
    let n = nz(normal);
    let v = nz(view);
    let m = MPoint {
        normal: n,
        view: v,
        roughness,
        anisotropy,
        f0,
    };
    let g = GlossyShadingPoint {
        position: Vec3::ZERO,
        normal: n,
        view: v,
        roughness,
        anisotropy,
        f0,
    };
    (m, g)
}

fn make_dir_samples(wi: Vec3, radiance: Vec3, pdf: f32) -> (MDirSample, DirectionSample) {
    let w = nz(wi);
    let m = MDirSample {
        wi: w,
        radiance,
        pdf,
    };
    let g = DirectionSample {
        wi: w,
        radiance,
        pdf,
    };
    (m, g)
}

#[test]
#[allow(clippy::disallowed_methods)]
fn mirror_ggx_subset_matches_golden() {
    // The GGX helpers the MIS pdf/throughput reuse must agree with the golden
    // `ggx_lobe` numerically, not merely compile.
    for &roughness in &[0.05_f32, 0.2, 0.5, 0.85] {
        for &aniso in &[-0.6_f32, 0.0, 0.4] {
            let (mx, my) = roughness_to_alpha_anisotropic(roughness, aniso);
            let (gx, gy) = golden_lobe::roughness_to_alpha_anisotropic(roughness, aniso);
            assert!(
                close(mx, gx) && close(my, gy),
                "alpha r={roughness} a={aniso}"
            );

            let wo = Vec3::new(0.2, 0.1, 0.974).normalize();
            let wi = Vec3::new(-0.15, 0.25, 0.956).normalize();
            assert!(close(
                vndf_pdf_reflect(wo, wi, mx, my),
                golden_lobe::vndf_pdf_reflect(wo, wi, gx, gy),
            ));
            assert!(close3(
                ggx_brdf(wo, wi, mx, my, Vec3::splat(0.5)),
                golden_lobe::ggx_brdf(wo, wi, gx, gy, Vec3::splat(0.5)),
            ));
        }
    }
}

#[test]
fn mirror_mis_weights_match_golden() {
    let pairs = [
        (3.0_f32, 7.0_f32),
        (4.0, 2.0),
        (1.0, 0.0),
        (0.0, 5.0),
        (2.5, 2.5),
        (0.3, 9.1),
    ];
    for &(a, b) in &pairs {
        assert!(
            close(
                balance_heuristic_pair(a, b),
                golden::balance_heuristic_pair(a, b)
            ),
            "balance a={a} b={b}"
        );
        assert!(
            close(power_heuristic(a, b), golden::power_heuristic(a, b)),
            "power a={a} b={b}"
        );
        assert!(close(
            power_heuristic_counts(2.0, a, 1.0, b),
            golden::power_heuristic_counts(2.0, a, 1.0, b),
        ));
        assert!(close(
            balance_heuristic_counts(1.0, a, 3.0, b),
            golden::balance_heuristic_counts(1.0, a, 3.0, b),
        ));
    }
}

#[test]
fn mirror_area_to_solid_angle_matches_golden() {
    let cases = [
        (0.25_f32, 2.0_f32, 0.5_f32),
        (1.0, 1.0, 1.0),
        (0.5, 3.0, 0.1),
        (2.0, 0.5, 0.9),
        (1.0, 2.0, 0.0),
    ];
    for &(p, d, c) in &cases {
        assert!(
            close(
                area_to_solid_angle_pdf(p, d, c),
                golden::area_to_solid_angle_pdf(p, d, c),
            ),
            "jacobian p={p} d={d} c={c}"
        );
    }
}

#[test]
fn mirror_brdf_pdf_and_estimate_match_golden() {
    let (m, g) = make_points(
        Vec3::Z,
        Vec3::new(0.0, 0.3, 0.954),
        0.3,
        0.1,
        Vec3::splat(0.5),
    );
    let dirs = [
        Vec3::new(0.0, -0.3, 0.954),
        Vec3::new(0.4, 0.1, 0.91),
        Vec3::new(-0.5, -0.2, 0.84),
        Vec3::new(0.9, 0.0, 0.436),
    ];
    for &d in &dirs {
        let wi = d.normalize();
        assert!(
            close(brdf_pdf_glossy(&m, wi), golden::brdf_pdf_glossy(&g, wi)),
            "brdf_pdf dir={wi:?}"
        );

        let (ms, gs) = make_dir_samples(wi, Vec3::new(2.0, 1.5, 0.8), 1.3);
        assert!(close3(
            brdf_sample_estimate(&m, &ms),
            golden::brdf_sample_estimate(&g, &gs),
        ));
    }
}

#[test]
fn mirror_mis_estimator_matches_golden() {
    let configs = [(0.2_f32, 0.0_f32), (0.3, 0.2), (0.6, -0.4)];
    for &(roughness, aniso) in &configs {
        let (m, g) = make_points(
            Vec3::Z,
            Vec3::new(0.0, 0.3, 0.954),
            roughness,
            aniso,
            Vec3::splat(0.5),
        );
        // A direction near the lobe peak (reflection of the view about the normal).
        let v = m.view;
        let n = m.normal;
        let refl = (2.0 * v.dot(n) * n - v).normalize();

        let brdf_pdf = brdf_pdf_glossy(&m, refl).max(1.0e-3);
        let light_pdf = 1.5_f32;
        let (mb, gb) = make_dir_samples(refl, Vec3::splat(2.0), brdf_pdf);
        let (ml, gl) = make_dir_samples(refl, Vec3::splat(2.0), light_pdf);

        let est_m = mis_estimator_glossy(&m, &mb, light_pdf, &ml, brdf_pdf);
        let est_g = golden::mis_estimator_glossy(&g, &gb, light_pdf, &gl, brdf_pdf);
        assert!(close3(est_m, est_g), "estimator r={roughness} a={aniso}");
        assert!(close(
            mis_estimate_luminance(est_m),
            golden::mis_estimate_luminance(est_g),
        ));

        // Single-technique collapse: light dead -> plain BRDF estimator.
        let est_single_m = mis_estimator_glossy(
            &m,
            &mb,
            0.0,
            &MDirSample {
                wi: Vec3::ZERO,
                radiance: Vec3::ZERO,
                pdf: 0.0,
            },
            0.0,
        );
        let est_single_g = golden::mis_estimator_glossy(&g, &gb, 0.0, &DirectionSample::NONE, 0.0);
        assert!(close3(est_single_m, est_single_g));
    }
}

#[test]
fn degenerate_inputs_match_golden_both_sides() {
    // NaN / negative pdfs and null samples must collapse to the same values.
    assert!(close(
        power_heuristic(f32::NAN, 1.0),
        golden::power_heuristic(f32::NAN, 1.0)
    ));
    assert!(close(
        balance_heuristic_pair(1.0, f32::NAN),
        golden::balance_heuristic_pair(1.0, f32::NAN),
    ));
    assert!(close(
        power_heuristic(-1.0, -1.0),
        golden::power_heuristic(-1.0, -1.0)
    ));
    assert!(close(
        area_to_solid_angle_pdf(1.0, 2.0, 0.0),
        golden::area_to_solid_angle_pdf(1.0, 2.0, 0.0),
    ));

    let (m, g) = make_points(
        Vec3::Z,
        Vec3::new(0.0, 0.3, 0.954),
        0.3,
        0.0,
        Vec3::splat(0.5),
    );
    let null_m = MDirSample {
        wi: Vec3::ZERO,
        radiance: Vec3::ZERO,
        pdf: 0.0,
    };
    let est_m = mis_estimator_glossy(&m, &null_m, 0.0, &null_m, 0.0);
    let est_g =
        golden::mis_estimator_glossy(&g, &DirectionSample::NONE, 0.0, &DirectionSample::NONE, 0.0);
    assert!(close3(est_m, est_g));
    assert_eq!(est_m, Vec3::ZERO);
}
