//! WESL compile coverage and CPU-mirror parity for `spec_denoise_history_clamp.wesl`.
//!
//! The sandbox has no GPU, so the compile test drives the history-clamp module
//! through the same `ShaderCache` / `wesl` pipeline the render world uses,
//! proving it parses and type-checks exactly as it will on device (the
//! `spec_denoise_history_clamp_probe` entry folds every helper so naga keeps
//! them live).
//!
//! The parity tests transcribe each WESL helper op-for-op into Rust and assert
//! agreement with the CPU golden
//! ([`prism_render_shading::gi::spec_denoise::history_clamp`]) across a sweep of
//! valid and degenerate inputs. WGSL's missing `isFinite` is modelled by
//! `(x - x) == 0.0` and the `+Z` normalise fallback; the golden's INFINITY
//! sentinel for a degenerate colour box is modelled by a `degenerate` flag that
//! returns the box centre (numerically identical). Transcendentals go through
//! `std` here (the golden uses `bevy_math::ops`), so parity holds to a small
//! floating-point tolerance.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_math::Vec3;
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};
use prism_render_shading::gi::spec_denoise::history_clamp as golden;
use prism_render_shading::gi::spec_denoise::history_clamp::HistoryClampParams;

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("spec_denoise history_clamp shader is WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles `spec_denoise_history_clamp.wesl`, proving the history-clamp module
/// parses and type-checks as it will in the render world.
#[test]
fn spec_denoise_history_clamp_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let history_clamp = shader_id(0x5350_4543_5f44_4e5f_4849_5354_5f00_0003);
    cache.set_shader(
        history_clamp,
        Shader::from_wesl(
            include_str!("../../shaders/spec_denoise_history_clamp.wesl"),
            "embedded://prism_render_scene/shaders/spec_denoise_history_clamp.wesl",
        ),
    );

    cache.get(0, history_clamp, &[]).unwrap_or_else(|error| {
        panic!("spec_denoise_history_clamp.wesl failed to compile: {error}")
    });
}

// --- Rust transcription of the WESL maths (op-for-op) -------------------------
// Control flow, floors and finite-guards mirror `spec_denoise_history_clamp.wesl`
// exactly; this is what the GPU evaluates, asserted equal to the golden below.

const MIN_ALPHA: f32 = 1.0e-3;
const PI: f32 = core::f32::consts::PI;

#[inline]
fn is_finite(x: f32) -> bool {
    (x - x) == 0.0
}

#[inline]
fn finite_or_zero(x: f32) -> f32 {
    if is_finite(x) {
        x
    } else {
        0.0
    }
}

#[inline]
fn sanitize_scalar(x: f32) -> f32 {
    finite_or_zero(x)
}

#[inline]
fn sanitize_rgb(c: Vec3) -> Vec3 {
    Vec3::new(
        finite_or_zero(c.x),
        finite_or_zero(c.y),
        finite_or_zero(c.z),
    )
    .max(Vec3::ZERO)
}

#[inline]
fn safe_normalize(v: Vec3) -> Vec3 {
    let len_sq = v.dot(v);
    if is_finite(len_sq) && len_sq > 1.0e-24 {
        v / len_sq.sqrt()
    } else {
        Vec3::Z
    }
}

#[inline]
#[allow(clippy::disallowed_methods)]
fn stable_exp(x: f32) -> f32 {
    if !is_finite(x) {
        return 0.0;
    }
    x.clamp(-80.0, 0.0).exp()
}

#[inline]
fn roughness_to_alpha(roughness: f32) -> f32 {
    let r = roughness.clamp(0.0, 1.0);
    (r * r).max(MIN_ALPHA)
}

#[inline]
fn variance_aabb(mean: Vec3, std: Vec3, sigma: f32) -> (Vec3, Vec3) {
    let m = sanitize_rgb(mean);
    let s = sanitize_rgb(std) * sigma.max(0.0);
    let lo = (m - s).max(Vec3::ZERO);
    let hi = m + s;
    (lo, hi)
}

#[inline]
fn clip_history_aabb(history: Vec3, aabb_min: Vec3, aabb_max: Vec3) -> Vec3 {
    let h = sanitize_rgb(history);
    let lo = sanitize_rgb(aabb_min);
    let hi = sanitize_rgb(aabb_max);
    let center = (lo + hi) * 0.5;
    let extent = (hi - lo) * 0.5;
    let offset = h - center;

    let mut max_ratio = 0.0_f32;
    let mut degenerate = false;

    if extent.x <= 1.0e-12 {
        if offset.x.abs() > 1.0e-12 {
            degenerate = true;
        }
    } else {
        max_ratio = max_ratio.max(offset.x.abs() / extent.x);
    }
    if extent.y <= 1.0e-12 {
        if offset.y.abs() > 1.0e-12 {
            degenerate = true;
        }
    } else {
        max_ratio = max_ratio.max(offset.y.abs() / extent.y);
    }
    if extent.z <= 1.0e-12 {
        if offset.z.abs() > 1.0e-12 {
            degenerate = true;
        }
    } else {
        max_ratio = max_ratio.max(offset.z.abs() / extent.z);
    }

    if degenerate {
        return center;
    }
    if max_ratio <= 1.0 {
        return h;
    }
    center + offset / max_ratio
}

#[inline]
fn roughness_history_length(roughness: f32, params: &HistoryClampParams) -> u32 {
    let r = roughness.clamp(0.0, 1.0);
    let lo = params.min_frames.max(1);
    let hi = params.max_frames.max(lo);
    let span = (hi - lo) as f32;
    let frames = lo as f32 + span * r;
    let rounded = (frames + 0.5) as u32;
    rounded.clamp(lo, hi)
}

#[inline]
#[allow(clippy::disallowed_methods)]
fn lobe_half_angle(roughness: f32) -> f32 {
    let alpha = roughness_to_alpha(roughness);
    alpha.atan().max(1.0e-4)
}

#[inline]
#[allow(clippy::disallowed_methods)]
fn lobe_rejection(
    normal_curr: Vec3,
    normal_prev: Vec3,
    roughness: f32,
    params: &HistoryClampParams,
) -> f32 {
    let a = safe_normalize(normal_curr);
    let b = safe_normalize(normal_prev);
    let cos = a.dot(b).clamp(-1.0, 1.0);
    let angle = cos.acos();
    let window = lobe_half_angle(roughness) * params.lobe_tolerance.max(1.0e-4);
    if angle <= 0.0 {
        return 1.0;
    }
    if angle >= window {
        return 0.0;
    }
    let t = (angle / window).clamp(0.0, 1.0);
    (0.5 * (1.0 + (PI * t).cos())).clamp(0.0, 1.0)
}

#[inline]
fn fast_history_factor(fast_luma: f32, slow_luma: f32, params: &HistoryClampParams) -> f32 {
    let f = sanitize_scalar(fast_luma).max(0.0);
    let s = sanitize_scalar(slow_luma).max(0.0);
    let diff = (f - s).abs();
    let denom = (f + s) * 0.5 + 1.0e-4;
    let rel = diff / denom;
    stable_exp(-params.fast_sensitivity.max(0.0) * rel)
}

#[inline]
fn specular_blend_weight(
    age: u32,
    frame_budget: u32,
    confidence: f32,
    lobe_keep: f32,
    fast_keep: f32,
) -> f32 {
    let budget = frame_budget.max(1);
    let effective = (age + 1).min(budget);
    let base = 1.0 / effective as f32;
    let keep = sanitize_scalar(confidence).clamp(0.0, 1.0)
        * sanitize_scalar(lobe_keep).clamp(0.0, 1.0)
        * sanitize_scalar(fast_keep).clamp(0.0, 1.0);
    let alpha = 1.0 - keep * (1.0 - base);
    alpha.clamp(0.0, 1.0)
}

struct MirrorClampResult {
    clamped_history: Vec3,
    blend_weight: f32,
    frame_budget: u32,
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn specular_accumulation(
    history: Vec3,
    neighbourhood_mean: Vec3,
    neighbourhood_std: Vec3,
    age: u32,
    roughness: f32,
    confidence: f32,
    normal_curr: Vec3,
    normal_prev: Vec3,
    fast_luma: f32,
    slow_luma: f32,
    params: &HistoryClampParams,
) -> MirrorClampResult {
    let (lo, hi) = variance_aabb(neighbourhood_mean, neighbourhood_std, params.clamp_sigma);
    let clamped = clip_history_aabb(history, lo, hi);
    let frame_budget = roughness_history_length(roughness, params);
    let lobe_keep = lobe_rejection(normal_curr, normal_prev, roughness, params);
    let fast_keep = fast_history_factor(fast_luma, slow_luma, params);
    let blend_weight = specular_blend_weight(age, frame_budget, confidence, lobe_keep, fast_keep);
    MirrorClampResult {
        clamped_history: sanitize_rgb(clamped),
        blend_weight,
        frame_budget,
    }
}

// --- helpers ------------------------------------------------------------------

const TOL: f32 = 2.0e-5;

fn close(a: f32, b: f32) {
    assert!(
        (a - b).abs() <= TOL + TOL * a.abs().max(b.abs()),
        "scalar mismatch: mirror={a}, golden={b}"
    );
}

fn close3(a: Vec3, b: Vec3) {
    close(a.x, b.x);
    close(a.y, b.y);
    close(a.z, b.z);
}

fn param_sweep() -> Vec<HistoryClampParams> {
    vec![
        HistoryClampParams::default(),
        HistoryClampParams {
            clamp_sigma: 1.0,
            min_frames: 1,
            max_frames: 8,
            lobe_tolerance: 1.0,
            fast_sensitivity: 2.0,
        },
        HistoryClampParams {
            clamp_sigma: 4.0,
            min_frames: 4,
            max_frames: 64,
            lobe_tolerance: 4.0,
            fast_sensitivity: 16.0,
        },
        HistoryClampParams {
            clamp_sigma: 0.0,
            min_frames: 2,
            max_frames: 2,
            lobe_tolerance: 1.0e-4,
            fast_sensitivity: 0.0,
        },
    ]
}

// --- parity tests -------------------------------------------------------------

#[test]
fn mirror_variance_aabb_matches_golden() {
    let cases = [
        (Vec3::splat(1.0), Vec3::splat(0.25), 2.0_f32),
        (Vec3::new(0.1, 0.5, 2.0), Vec3::new(0.3, 0.1, 1.0), 1.5),
        (Vec3::splat(0.2), Vec3::splat(0.5), 2.0),
        (Vec3::splat(f32::NAN), Vec3::splat(f32::INFINITY), -1.0),
    ];
    for (mean, std, sigma) in cases {
        let (mlo, mhi) = variance_aabb(mean, std, sigma);
        let (glo, ghi) = golden::variance_aabb(mean, std, sigma);
        close3(mlo, glo);
        close3(mhi, ghi);
    }
    // Non-vacuous: box brackets the mean symmetrically and never inverts.
    let (lo, hi) = golden::variance_aabb(Vec3::splat(1.0), Vec3::splat(0.25), 2.0);
    close3(lo, Vec3::splat(0.5));
    close3(hi, Vec3::splat(1.5));
}

#[test]
fn mirror_clip_history_aabb_matches_golden() {
    let lo = Vec3::splat(0.0);
    let hi = Vec3::splat(1.0);
    let cases = [
        (Vec3::new(0.3, 0.7, 0.5), lo, hi),
        (Vec3::new(2.0, 0.5, 0.5), lo, hi),
        (Vec3::new(5.0, -3.0, 10.0), lo, hi),
        // Degenerate (zero-extent) box -> centre.
        (Vec3::splat(5.0), Vec3::splat(0.5), Vec3::splat(0.5)),
        (
            Vec3::new(0.4, 0.6, 0.9),
            Vec3::new(0.2, 0.2, 0.2),
            Vec3::new(0.8, 0.8, 0.8),
        ),
    ];
    for (h, bmin, bmax) in cases {
        close3(
            clip_history_aabb(h, bmin, bmax),
            golden::clip_history_aabb(h, bmin, bmax),
        );
    }
    // Non-vacuous: an inside point is unchanged, an outside point moves inward.
    let inside = Vec3::new(0.3, 0.7, 0.5);
    close3(golden::clip_history_aabb(inside, lo, hi), inside);
    let out = golden::clip_history_aabb(Vec3::new(2.0, 0.5, 0.5), lo, hi);
    assert!(out.x <= 2.0 && out.x <= 1.0 + 1.0e-4 && out.x >= -1.0e-4);
}

#[test]
fn mirror_roughness_history_length_matches_golden() {
    for p in param_sweep() {
        for &r in &[0.0_f32, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] {
            assert_eq!(
                roughness_history_length(r, &p),
                golden::roughness_history_length(r, &p),
                "frame budget mismatch at roughness {r}"
            );
        }
    }
    // Non-vacuous: monotone; mirror -> min_frames, rough -> max_frames.
    let p = HistoryClampParams::default();
    assert_eq!(golden::roughness_history_length(0.0, &p), p.min_frames);
    assert_eq!(golden::roughness_history_length(1.0, &p), p.max_frames);
    let mid = golden::roughness_history_length(0.5, &p);
    assert!(p.min_frames <= mid && mid <= p.max_frames);
}

#[test]
fn mirror_lobe_half_angle_matches_golden() {
    for &r in &[0.0_f32, 0.1, 0.3, 0.5, 0.8, 1.0] {
        close(lobe_half_angle(r), golden::lobe_half_angle(r));
    }
    // Non-vacuous: monotone, mirror tiny, roughness=1 -> atan(1) = pi/4.
    assert!(golden::lobe_half_angle(0.0) > 0.0);
    assert!(golden::lobe_half_angle(0.2) < golden::lobe_half_angle(0.9));
    assert!((golden::lobe_half_angle(1.0) - core::f32::consts::FRAC_PI_4).abs() < 1.0e-3);
}

#[test]
fn mirror_lobe_rejection_matches_golden() {
    for p in param_sweep() {
        for &r in &[0.05_f32, 0.3, 0.7, 1.0] {
            let cases = [
                (Vec3::Z, Vec3::Z),
                (Vec3::Z, Vec3::new(0.05, 0.0, 1.0)),
                (Vec3::Z, Vec3::new(0.4, 0.0, 1.0)),
                (Vec3::Z, Vec3::X),
                (Vec3::Z, Vec3::NEG_Z),
            ];
            for (a, b) in cases {
                close(
                    lobe_rejection(a, b, r, &p),
                    golden::lobe_rejection(a, b, r, &p),
                );
            }
        }
    }
    // Non-vacuous: aligned normals keep fully, a 90 deg shift is rejected.
    let p = HistoryClampParams::default();
    close(golden::lobe_rejection(Vec3::Z, Vec3::Z, 0.1, &p), 1.0);
    assert!(golden::lobe_rejection(Vec3::Z, Vec3::X, 0.1, &p).abs() < 1.0e-5);
}

#[test]
fn mirror_fast_history_factor_matches_golden() {
    for p in param_sweep() {
        let cases = [
            (0.5_f32, 0.5_f32),
            (1.0, 0.0),
            (0.0, 0.0),
            (2.0, 1.9),
            (10.0, 0.1),
        ];
        for (f, s) in cases {
            close(
                fast_history_factor(f, s, &p),
                golden::fast_history_factor(f, s, &p),
            );
        }
    }
    // Non-vacuous: agreement -> 1, divergence decays below it.
    let p = HistoryClampParams::default();
    close(golden::fast_history_factor(0.5, 0.5, &p), 1.0);
    assert!(golden::fast_history_factor(10.0, 0.1, &p) < 1.0);
}

#[test]
fn mirror_specular_blend_weight_matches_golden() {
    for &age in &[0u32, 1, 4, 16, 64] {
        for &budget in &[1u32, 2, 8, 32] {
            for &conf in &[0.0_f32, 0.5, 1.0] {
                for &lobe in &[0.0_f32, 0.5, 1.0] {
                    for &fast in &[0.0_f32, 0.5, 1.0] {
                        close(
                            specular_blend_weight(age, budget, conf, lobe, fast),
                            golden::specular_blend_weight(age, budget, conf, lobe, fast),
                        );
                    }
                }
            }
        }
    }
    // Non-vacuous: full keep -> base EMA (1/budget here, age>=budget);
    // zero keep -> alpha 1 (discard history).
    close(
        golden::specular_blend_weight(64, 32, 1.0, 1.0, 1.0),
        1.0 / 32.0,
    );
    close(golden::specular_blend_weight(64, 32, 0.0, 1.0, 1.0), 1.0);
}

#[test]
fn mirror_specular_accumulation_matches_golden() {
    let p = HistoryClampParams::default();
    let cases = [
        (
            Vec3::new(0.6, 0.5, 0.4),
            Vec3::splat(0.5),
            Vec3::splat(0.1),
            4u32,
            0.4_f32,
            0.8_f32,
            Vec3::Z,
            Vec3::new(0.1, 0.0, 1.0),
            0.5_f32,
            0.45_f32,
        ),
        (
            Vec3::new(5.0, 0.1, 0.1),
            Vec3::splat(0.3),
            Vec3::splat(0.2),
            0,
            0.9,
            0.3,
            Vec3::Z,
            Vec3::X,
            2.0,
            0.1,
        ),
    ];
    for (h, mean, std, age, r, conf, nc, np, fl, sl) in cases {
        let m = specular_accumulation(h, mean, std, age, r, conf, nc, np, fl, sl, &p);
        let g = golden::specular_accumulation(h, mean, std, age, r, conf, nc, np, fl, sl, &p);
        close3(m.clamped_history, g.clamped_history);
        close(m.blend_weight, g.blend_weight);
        assert_eq!(m.frame_budget, g.frame_budget);
    }
}

#[test]
fn mirror_specular_accumulation_is_finite_on_degenerate_inputs() {
    let p = HistoryClampParams::default();
    let m = specular_accumulation(
        Vec3::splat(f32::NAN),
        Vec3::splat(f32::INFINITY),
        Vec3::splat(f32::NAN),
        0,
        2.0,
        f32::NAN,
        Vec3::ZERO,
        Vec3::ZERO,
        f32::INFINITY,
        f32::NAN,
        &p,
    );
    let g = golden::specular_accumulation(
        Vec3::splat(f32::NAN),
        Vec3::splat(f32::INFINITY),
        Vec3::splat(f32::NAN),
        0,
        2.0,
        f32::NAN,
        Vec3::ZERO,
        Vec3::ZERO,
        f32::INFINITY,
        f32::NAN,
        &p,
    );
    close3(m.clamped_history, g.clamped_history);
    close(m.blend_weight, g.blend_weight);
    assert_eq!(m.frame_budget, g.frame_budget);
    assert!(g.clamped_history.is_finite());
    assert!(g.blend_weight.is_finite());
    assert!((0.0..=1.0).contains(&g.blend_weight));
}
