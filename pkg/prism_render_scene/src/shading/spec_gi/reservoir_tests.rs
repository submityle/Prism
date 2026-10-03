//! WESL compile coverage and CPU-mirror parity for `spec_gi_reservoir.wesl`.
//!
//! The sandbox has no GPU, so the compile test drives the glossy-reservoir
//! module through the same `ShaderCache` / `wesl` pipeline the render world
//! uses, proving it parses and type-checks exactly as it will on device (the
//! `spec_gi_reservoir_probe` entry folds every helper so naga keeps them live).
//!
//! The parity tests transcribe each WESL helper op-for-op into Rust and assert
//! agreement with the CPU golden
//! ([`prism_render_shading::gi::spec_gi::glossy_reservoir`]) across a sweep of
//! valid inputs — including the full streaming → cap → merge → finalize →
//! contribution reservoir flow, driven with identical RNG sequences so the two
//! implementations must agree bit-for-bit modulo the floating-point tolerance.
//! `Option<GiSample>` is modelled by a `has_sample` flag and WGSL's missing
//! `isFinite` by `(x - x) == 0.0`, matching the shader exactly.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_math::Vec3;
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};
use prism_render_shading::gi::screen_probe::restir::{GiSample, Reservoir};
use prism_render_shading::gi::spec_gi::glossy_reservoir as golden;
use prism_render_shading::gi::spec_gi::glossy_reservoir::GlossyShadingPoint;

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("spec_gi reservoir shader is WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles `spec_gi_reservoir.wesl`, proving the glossy-reservoir module parses
/// and type-checks as it will in the render world.
#[test]
fn spec_gi_reservoir_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let reservoir = shader_id(0x5350_4543_5f47_495f_5245_5356_5f00_0002);
    cache.set_shader(
        reservoir,
        Shader::from_wesl(
            include_str!("../../shaders/spec_gi_reservoir.wesl"),
            "embedded://prism_render_scene/shaders/spec_gi_reservoir.wesl",
        ),
    );

    cache
        .get(0, reservoir, &[])
        .unwrap_or_else(|error| panic!("spec_gi_reservoir.wesl failed to compile: {error}"));
}

// --- Rust transcription of the WESL maths (op-for-op) -------------------------
// Floors, control flow and finite-guards mirror `spec_gi_reservoir.wesl`
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

#[derive(Clone, Copy)]
struct MPoint {
    normal: Vec3,
    view: Vec3,
    roughness: f32,
    anisotropy: f32,
    f0: Vec3,
}

#[derive(Clone, Copy)]
struct MSample {
    visible_point: Vec3,
    sample_point: Vec3,
    radiance: Vec3,
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
fn glossy_lobe_throughput(point: &MPoint, sample: &MSample) -> Vec3 {
    let to_sample = sample.sample_point - sample.visible_point;
    let dist2 = to_sample.dot(to_sample);
    if dist2 <= 1.0e-12 {
        return Vec3::ZERO;
    }
    let wi_world = to_sample / dist2.sqrt();
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

fn glossy_target_function(point: &MPoint, sample: &MSample) -> f32 {
    let throughput = glossy_lobe_throughput(point, sample);
    let contrib = throughput * sample.radiance;
    let t = luminance(contrib);
    if is_finite(t) {
        t.max(0.0)
    } else {
        0.0
    }
}

#[allow(clippy::disallowed_methods)]
fn roughness_reuse_weight(
    dst_roughness: f32,
    src_roughness: f32,
    dst_normal: Vec3,
    src_normal: Vec3,
    sigma_roughness: f32,
) -> f32 {
    let sigma = sigma_roughness.max(1.0e-3);
    let dr = (dst_roughness.clamp(0.0, 1.0) - src_roughness.clamp(0.0, 1.0)) / sigma;
    let rough_w = (-0.5 * dr * dr).exp();
    let n_dst = nz(dst_normal);
    let n_src = nz(src_normal);
    if n_dst.dot(n_dst) == 0.0 || n_src.dot(n_src) == 0.0 {
        return 0.0;
    }
    let cos_n = n_dst.dot(n_src).clamp(0.0, 1.0);
    let exponent = 8.0 + 56.0 * (1.0 - dst_roughness.clamp(0.0, 1.0));
    let normal_w = cos_n.powf(exponent);
    let w = rough_w * normal_w;
    if is_finite(w) {
        w.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn roughness_confidence_cap(roughness: f32, base_cap: f32) -> f32 {
    let base = base_cap.max(1.0);
    let r = roughness.clamp(0.0, 1.0);
    let cap = 1.0 + (base - 1.0) * r;
    cap.clamp(1.0, base)
}

#[derive(Clone, Copy)]
struct MReservoir {
    sample: MSample,
    w_sum: f32,
    m: f32,
    w: f32,
    has_sample: bool,
}

impl MReservoir {
    fn new() -> Self {
        Self {
            sample: MSample {
                visible_point: Vec3::ZERO,
                sample_point: Vec3::ZERO,
                radiance: Vec3::ZERO,
            },
            w_sum: 0.0,
            m: 0.0,
            w: 0.0,
            has_sample: false,
        }
    }

    fn update(&mut self, sample: MSample, weight: f32, rng_uniform: f32) -> bool {
        if !is_finite(weight) || weight <= 0.0 {
            return false;
        }
        self.w_sum += weight;
        self.m += 1.0;
        let u = rng_uniform.clamp(0.0, 1.0);
        if u * self.w_sum <= weight {
            self.sample = sample;
            self.has_sample = true;
            true
        } else {
            false
        }
    }

    fn merge(&mut self, other: &MReservoir, other_target_pdf: f32, rng_uniform: f32) -> bool {
        if !is_finite(other.m) || other.m <= 0.0 {
            return false;
        }
        self.m += other.m;
        let rw = other.m * other_target_pdf.max(0.0) * other.w.max(0.0);
        if !is_finite(rw) || rw <= 0.0 {
            return false;
        }
        self.w_sum += rw;
        let u = rng_uniform.clamp(0.0, 1.0);
        if u * self.w_sum <= rw {
            self.sample = other.sample;
            self.has_sample = other.has_sample;
            true
        } else {
            false
        }
    }

    fn cap_confidence(&mut self, max_m: f32) {
        if max_m >= 0.0 && self.m > max_m {
            self.m = max_m;
        }
    }

    fn finalize_weight(&mut self, target_pdf: f32) {
        if !self.has_sample
            || self.m <= 0.0
            || !is_finite(self.w_sum)
            || !is_finite(target_pdf)
            || target_pdf <= 0.0
        {
            self.w = 0.0;
            return;
        }
        let w = (self.w_sum / self.m) / target_pdf;
        self.w = if is_finite(w) && w >= 0.0 { w } else { 0.0 };
    }
}

fn stream_glossy_candidate(
    r: &mut MReservoir,
    point: &MPoint,
    sample: MSample,
    source_pdf: f32,
    rng_uniform: f32,
) -> bool {
    if !is_finite(source_pdf) || source_pdf <= 0.0 {
        return false;
    }
    let p_hat = glossy_target_function(point, &sample);
    let weight = p_hat / source_pdf;
    r.update(sample, weight, rng_uniform)
}

#[allow(clippy::too_many_arguments)]
fn merge_glossy(
    canonical: &mut MReservoir,
    neighbour: &MReservoir,
    point: &MPoint,
    neighbour_roughness: f32,
    neighbour_normal: Vec3,
    sigma_roughness: f32,
    rng_uniform: f32,
) -> bool {
    let shifted_pdf = if neighbour.has_sample {
        glossy_target_function(point, &neighbour.sample)
    } else {
        0.0
    };
    let reuse = roughness_reuse_weight(
        point.roughness,
        neighbour_roughness,
        point.normal,
        neighbour_normal,
        sigma_roughness,
    );
    canonical.merge(neighbour, shifted_pdf * reuse, rng_uniform)
}

fn finalize_glossy(r: &mut MReservoir, point: &MPoint) -> f32 {
    let p_hat = if r.has_sample {
        glossy_target_function(point, &r.sample)
    } else {
        0.0
    };
    r.finalize_weight(p_hat);
    r.w
}

fn glossy_contribution(r: &MReservoir, point: &MPoint) -> Vec3 {
    if !r.has_sample {
        return Vec3::ZERO;
    }
    let w = r.w;
    if !is_finite(w) || w <= 0.0 {
        return Vec3::ZERO;
    }
    let throughput = glossy_lobe_throughput(point, &r.sample);
    let v = throughput * r.sample.radiance * w;
    if is_finite3(v) {
        v.max(Vec3::ZERO)
    } else {
        Vec3::ZERO
    }
}

// --- Parity harness ----------------------------------------------------------

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

fn make_samples(visible: Vec3, sample_pt: Vec3, radiance: Vec3) -> (MSample, GiSample) {
    let m = MSample {
        visible_point: visible,
        sample_point: sample_pt,
        radiance,
    };
    let g = GiSample {
        visible_point: visible,
        visible_normal: Vec3::Z,
        sample_point: sample_pt,
        sample_normal: Vec3::Z,
        radiance,
    };
    (m, g)
}

#[test]
fn mirror_ggx_subset_matches_golden() {
    use prism_render_shading::gi::spec_gi::ggx_lobe as glb;
    for ri in 0..6u32 {
        let r = ri as f32 / 5.0;
        for ai in 0..5u32 {
            let aniso = -1.0 + 2.0 * ai as f32 / 4.0;
            let (ax, ay) = roughness_to_alpha_anisotropic(r, aniso);
            let (gx, gy) = glb::roughness_to_alpha_anisotropic(r, aniso);
            assert!(close(ax, gx) && close(ay, gy), "alpha r={r} aniso={aniso}");
            let wo = nz(Vec3::new(0.2, 0.1, 0.9));
            let wi = nz(Vec3::new(-0.1, 0.3, 0.8));
            let f0 = Vec3::splat(0.08);
            assert!(
                close3(
                    ggx_brdf(wo, wi, ax, ay, f0),
                    glb::ggx_brdf(wo, wi, ax, ay, f0)
                ),
                "ggx_brdf r={r} aniso={aniso}"
            );
        }
    }
}

#[test]
fn mirror_lobe_throughput_and_target_match_golden() {
    let f0 = Vec3::splat(0.06);
    for ri in 0..6u32 {
        let r = 0.05 + 0.9 * ri as f32 / 5.0;
        for ai in 0..4u32 {
            let aniso = -0.6 + 1.2 * ai as f32 / 3.0;
            let (mp, gp) = make_points(
                Vec3::new(0.1, 0.2, 1.0),
                Vec3::new(0.0, 0.3, 1.0),
                r,
                aniso,
                f0,
            );
            for sj in 0..5u32 {
                let sx = -0.4 + 0.2 * sj as f32;
                let (ms, gs) = make_samples(
                    Vec3::ZERO,
                    Vec3::new(sx, 0.3, 1.5 + 0.1 * sj as f32),
                    Vec3::new(1.0, 0.7, 0.4) * (0.5 + sj as f32),
                );
                assert!(
                    close3(
                        glossy_lobe_throughput(&mp, &ms),
                        golden::glossy_lobe_throughput(&gp, &gs)
                    ),
                    "throughput r={r} aniso={aniso} sj={sj}"
                );
                assert!(
                    close(
                        glossy_target_function(&mp, &ms),
                        golden::glossy_target_function(&gp, &gs)
                    ),
                    "target r={r} aniso={aniso} sj={sj}"
                );
                let dir = nz(Vec3::new(sx, 0.1, 1.0));
                assert!(
                    close3(
                        glossy_lobe_throughput_dir(&mp, dir),
                        golden::glossy_lobe_throughput_dir(&gp, dir)
                    ),
                    "throughput_dir r={r} aniso={aniso} sj={sj}"
                );
            }
        }
    }
}

#[test]
fn mirror_reuse_weight_matches_golden() {
    let normals = [
        Vec3::Z,
        nz(Vec3::new(0.3, 0.0, 0.95)),
        nz(Vec3::new(0.0, 0.6, 0.8)),
        nz(Vec3::new(0.5, 0.5, 0.7)),
    ];
    for dr in 0..5u32 {
        let dst_r = dr as f32 / 4.0;
        for sr in 0..5u32 {
            let src_r = sr as f32 / 4.0;
            for &nn in &normals {
                for &sigma in &[0.05_f32, 0.1, 0.3] {
                    assert!(
                        close(
                            roughness_reuse_weight(dst_r, src_r, Vec3::Z, nn, sigma),
                            golden::roughness_reuse_weight(dst_r, src_r, Vec3::Z, nn, sigma)
                        ),
                        "reuse dst_r={dst_r} src_r={src_r} sigma={sigma}"
                    );
                }
            }
        }
    }
}

#[test]
fn mirror_confidence_cap_matches_golden() {
    for ri in 0..9u32 {
        let r = ri as f32 / 8.0;
        for &base in &[1.0_f32, 4.0, 16.0, 32.0, 64.0] {
            assert!(
                close(
                    roughness_confidence_cap(r, base),
                    golden::roughness_confidence_cap(r, base)
                ),
                "cap r={r} base={base}"
            );
        }
    }
}

#[test]
fn mirror_full_reservoir_flow_matches_golden() {
    let f0 = Vec3::splat(0.05);
    // A sweep of (roughness, rng, source_pdf) so stream/merge selection branches
    // both ways and finalize/contribution agree with the golden end-to-end.
    for ri in 0..4u32 {
        let r = 0.05 + 0.85 * ri as f32 / 3.0;
        let (mp, gp) = make_points(
            Vec3::new(0.0, 0.1, 1.0),
            Vec3::new(0.0, 0.25, 1.0),
            r,
            0.15,
            f0,
        );

        let (ms0, gs0) = make_samples(
            Vec3::ZERO,
            Vec3::new(0.1, 0.2, 1.6),
            Vec3::new(2.0, 1.5, 1.0),
        );
        let (ms1, gs1) = make_samples(
            Vec3::ZERO,
            Vec3::new(-0.2, 0.1, 1.4),
            Vec3::new(1.0, 0.8, 0.6),
        );
        let (msn, gsn) = make_samples(
            Vec3::ZERO,
            Vec3::new(0.3, 0.0, 1.2),
            Vec3::new(3.0, 2.0, 1.2),
        );

        for &(p0, u0, p1, u1, cap, nr, pn, un) in &[
            (
                0.5_f32, 0.0_f32, 0.8_f32, 0.3_f32, 8.0_f32, 0.2_f32, 1.0_f32, 0.0_f32,
            ),
            (1.0, 0.7, 0.4, 0.9, 16.0, 0.9, 1.0, 0.5),
            (0.3, 0.2, 1.2, 0.6, 4.0, 0.1, 1.0, 0.95),
        ] {
            // Mirror flow.
            let mut mr = MReservoir::new();
            let m_sel0 = stream_glossy_candidate(&mut mr, &mp, ms0, p0, u0);
            let m_sel1 = stream_glossy_candidate(&mut mr, &mp, ms1, p1, u1);
            mr.cap_confidence(cap);
            let mut m_neighbour = MReservoir::new();
            stream_glossy_candidate(&mut m_neighbour, &mp, msn, pn, 0.0);
            finalize_glossy(&mut m_neighbour, &mp);
            let m_merged = merge_glossy(&mut mr, &m_neighbour, &mp, nr, mp.normal, 0.1, un);
            let m_w = finalize_glossy(&mut mr, &mp);
            let m_c = glossy_contribution(&mr, &mp);

            // Golden flow — identical operations and RNG.
            let mut gr = Reservoir::<GiSample>::new();
            let g_sel0 = golden::stream_glossy_candidate(&mut gr, &gp, gs0, p0, u0);
            let g_sel1 = golden::stream_glossy_candidate(&mut gr, &gp, gs1, p1, u1);
            gr.cap_confidence(cap);
            let mut g_neighbour = Reservoir::<GiSample>::new();
            golden::stream_glossy_candidate(&mut g_neighbour, &gp, gsn, pn, 0.0);
            golden::finalize_glossy(&mut g_neighbour, &gp);
            let g_merged = golden::merge_glossy(&mut gr, &g_neighbour, &gp, nr, gp.normal, 0.1, un);
            let g_w = golden::finalize_glossy(&mut gr, &gp);
            let g_c = golden::glossy_contribution(&gr, &gp);

            assert_eq!(m_sel0, g_sel0, "sel0 r={r} p0={p0}");
            assert_eq!(m_sel1, g_sel1, "sel1 r={r}");
            assert_eq!(m_merged, g_merged, "merged r={r}");
            assert!(close(m_w, g_w), "W r={r}: mirror={m_w} golden={g_w}");
            assert!(
                close3(m_c, g_c),
                "contrib r={r}: mirror={m_c:?} golden={g_c:?}"
            );
            let _ = pn;
        }
    }
}

#[test]
fn empty_reservoir_contributes_nothing_both_sides() {
    let (mp, gp) = make_points(
        Vec3::Z,
        Vec3::new(0.0, 0.0, 1.0),
        0.3,
        0.0,
        Vec3::splat(0.04),
    );
    let mr = MReservoir::new();
    let gr = Reservoir::<GiSample>::new();
    assert_eq!(glossy_contribution(&mr, &mp), Vec3::ZERO);
    assert_eq!(golden::glossy_contribution(&gr, &gp), Vec3::ZERO);
}
