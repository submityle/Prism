//! WESL compile coverage and CPU-mirror parity for `spec_denoise_reproject.wesl`.
//!
//! The sandbox has no GPU, so the compile test drives the reprojection module
//! through the same `ShaderCache` / `wesl` pipeline the render world uses,
//! proving it parses and type-checks exactly as it will on device (the
//! `spec_denoise_reproject_probe` entry folds every helper so naga keeps them
//! live).
//!
//! The parity tests transcribe each WESL helper op-for-op into Rust and assert
//! agreement with the CPU golden
//! ([`prism_render_shading::gi::spec_denoise::reproject`]) across a sweep of
//! valid and degenerate inputs. WGSL's missing `isFinite` is modelled by
//! `(x - x) == 0.0` and the `+Z` normalise fallback, matching the shader
//! exactly; transcendentals go through `std` here (the golden uses
//! `bevy_math::ops`), so parity holds to a small floating-point tolerance.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_math::Vec3;
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};
use prism_render_shading::gi::spec_denoise::reproject as golden;
use prism_render_shading::gi::spec_denoise::reproject::ReprojectParams;

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("spec_denoise reproject shader is WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles `spec_denoise_reproject.wesl`, proving the reprojection module
/// parses and type-checks as it will in the render world.
#[test]
fn spec_denoise_reproject_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let reproject = shader_id(0x5350_4543_5f44_4e5f_5245_5052_4f00_0001);
    cache.set_shader(
        reproject,
        Shader::from_wesl(
            include_str!("../../shaders/spec_denoise_reproject.wesl"),
            "embedded://prism_render_scene/shaders/spec_denoise_reproject.wesl",
        ),
    );

    cache
        .get(0, reproject, &[])
        .unwrap_or_else(|error| panic!("spec_denoise_reproject.wesl failed to compile: {error}"));
}

// --- Rust transcription of the WESL maths (op-for-op) -------------------------
// Control flow, floors and finite-guards mirror `spec_denoise_reproject.wesl`
// exactly; this is what the GPU evaluates, asserted equal to the golden below.

#[derive(Clone, Copy)]
struct MirrorParams {
    parallax_sensitivity: f32,
    virtual_exponent: f32,
    min_confidence: f32,
}

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
fn sanitize_vec(v: Vec3) -> Vec3 {
    Vec3::new(
        finite_or_zero(v.x),
        finite_or_zero(v.y),
        finite_or_zero(v.z),
    )
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
fn reflect(i: Vec3, n: Vec3) -> Vec3 {
    let nn = safe_normalize(n);
    i - nn * (2.0 * i.dot(nn))
}

#[inline]
#[allow(clippy::disallowed_methods)]
fn virtual_history_amount(roughness: f32, p: MirrorParams) -> f32 {
    let r = roughness.clamp(0.0, 1.0);
    let e = p.virtual_exponent.max(0.0);
    (1.0 - r).max(0.0).powf(e).clamp(0.0, 1.0)
}

#[inline]
#[allow(clippy::disallowed_methods)]
fn specular_dominant_factor(n_dot_v: f32, roughness: f32) -> f32 {
    let ndv = n_dot_v.clamp(0.0, 1.0);
    let r = roughness.clamp(0.0, 1.0);
    let a = 0.298475 * (39.4115 - 39.0029 * r).ln();
    let f = (1.0 - ndv).max(0.0).powf(10.8649) * (1.0 - a) + a;
    f.clamp(0.0, 1.0)
}

#[inline]
fn virtual_reflection_point(
    surface_pos: Vec3,
    view_dir: Vec3,
    normal: Vec3,
    hit_distance: f32,
    roughness: f32,
    p: MirrorParams,
) -> Vec3 {
    let v = safe_normalize(view_dir);
    let n = safe_normalize(normal);
    let r = reflect(-v, n);
    let dist = sanitize_scalar(hit_distance).max(0.0);
    let amount = virtual_history_amount(roughness, p);
    sanitize_vec(surface_pos) + r * (dist * amount)
}

#[inline]
fn view_parallax(prev_cam_pos: Vec3, curr_cam_pos: Vec3, point: Vec3) -> f32 {
    let p = sanitize_vec(point);
    let a = sanitize_vec(curr_cam_pos) - p;
    let b = sanitize_vec(prev_cam_pos) - p;
    let la = a.length();
    let lb = b.length();
    if la <= 1.0e-12 || lb <= 1.0e-12 {
        return 0.0;
    }
    let cos_t = (a.dot(b) / (la * lb)).clamp(-1.0, 1.0);
    let sin_t = (1.0 - cos_t * cos_t).max(0.0).sqrt();
    let tan_t = sin_t / cos_t.abs().max(1.0e-4);
    if is_finite(tan_t) {
        tan_t.min(1.0e4)
    } else {
        1.0e4
    }
}

#[inline]
fn reprojection_confidence(roughness: f32, parallax: f32, p: MirrorParams) -> f32 {
    let px = sanitize_scalar(parallax).max(0.0);
    let amount = virtual_history_amount(roughness, p);
    let sensitivity = p.parallax_sensitivity.max(0.0) * amount;
    let conf = stable_exp(-sensitivity * px);
    let floor_c = p.min_confidence.clamp(0.0, 1.0);
    conf.max(floor_c).clamp(0.0, 1.0)
}

#[inline]
fn blend_reprojected_position(
    surface_reproj: Vec3,
    virtual_reproj: Vec3,
    roughness: f32,
    p: MirrorParams,
) -> Vec3 {
    let amount = virtual_history_amount(roughness, p);
    let s = sanitize_vec(surface_reproj);
    let v = sanitize_vec(virtual_reproj);
    sanitize_vec(s.lerp(v, amount))
}

struct MirrorReprojection {
    sample_position: Vec3,
    virtual_point: Vec3,
    parallax: f32,
    confidence: f32,
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn reproject_specular(
    surface_pos: Vec3,
    surface_reproj: Vec3,
    view_dir: Vec3,
    normal: Vec3,
    hit_distance: f32,
    roughness: f32,
    prev_cam_pos: Vec3,
    curr_cam_pos: Vec3,
    p: MirrorParams,
) -> MirrorReprojection {
    let virtual_point =
        virtual_reflection_point(surface_pos, view_dir, normal, hit_distance, roughness, p);
    let parallax = view_parallax(prev_cam_pos, curr_cam_pos, virtual_point);
    let confidence = reprojection_confidence(roughness, parallax, p);
    let sample_position = blend_reprojected_position(surface_reproj, virtual_point, roughness, p);
    MirrorReprojection {
        sample_position,
        virtual_point,
        parallax,
        confidence,
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

fn mirror_params(p: &ReprojectParams) -> MirrorParams {
    MirrorParams {
        parallax_sensitivity: p.parallax_sensitivity,
        virtual_exponent: p.virtual_exponent,
        min_confidence: p.min_confidence,
    }
}

fn param_sweep() -> Vec<ReprojectParams> {
    vec![
        ReprojectParams::default(),
        ReprojectParams {
            parallax_sensitivity: 2.0,
            virtual_exponent: 1.0,
            min_confidence: 0.1,
        },
        ReprojectParams {
            parallax_sensitivity: 16.0,
            virtual_exponent: 4.0,
            min_confidence: 0.0,
        },
        ReprojectParams {
            parallax_sensitivity: 0.0,
            virtual_exponent: 0.0,
            min_confidence: 0.5,
        },
    ]
}

// --- parity tests -------------------------------------------------------------

#[test]
fn mirror_reflect_matches_golden() {
    let cases = [
        (Vec3::NEG_Z, Vec3::Z),
        (Vec3::new(0.3, -0.4, -0.8), Vec3::new(0.0, 0.2, 1.0)),
        (Vec3::new(-1.0, 0.5, -0.2), Vec3::new(1.0, 1.0, 1.0)),
    ];
    for (i, n) in cases {
        close3(reflect(i, n), golden::reflect(i, n));
    }
    // Non-vacuous: a straight-down ray reflects straight up about +Z.
    assert!((golden::reflect(Vec3::NEG_Z, Vec3::Z) - Vec3::Z).length() < 1.0e-5);
}

#[test]
fn mirror_virtual_history_amount_matches_golden() {
    for p in param_sweep() {
        let mp = mirror_params(&p);
        let mut prev = f32::INFINITY;
        for &r in &[0.0_f32, 0.15, 0.3, 0.5, 0.7, 0.9, 1.0] {
            let m = virtual_history_amount(r, mp);
            let g = golden::virtual_history_amount(r, &p);
            close(m, g);
            // Non-vacuous: schedule is non-increasing in roughness.
            assert!(g <= prev + 1.0e-6, "amount not monotone: {g} > {prev}");
            prev = g;
        }
    }
}

#[test]
fn mirror_specular_dominant_factor_matches_golden() {
    for &ndv in &[0.0_f32, 0.1, 0.3, 0.6, 0.9, 1.0] {
        for &r in &[0.0_f32, 0.05, 0.25, 0.5, 0.8, 1.0] {
            let m = specular_dominant_factor(ndv, r);
            let g = golden::specular_dominant_factor(ndv, r);
            close(m, g);
            assert!((0.0..=1.0).contains(&g));
        }
    }
    // Non-vacuous: smoother surface leans more toward the mirror reflection.
    assert!(
        golden::specular_dominant_factor(0.9, 0.02) > golden::specular_dominant_factor(0.9, 0.9)
    );
}

#[test]
fn mirror_virtual_reflection_point_matches_golden() {
    for p in param_sweep() {
        let mp = mirror_params(&p);
        let cases = [
            (Vec3::ZERO, Vec3::Z, Vec3::Z, 3.0, 0.0),
            (Vec3::new(1.0, 2.0, 3.0), Vec3::Z, Vec3::Z, 5.0, 1.0),
            (
                Vec3::new(-2.0, 0.5, 1.0),
                Vec3::new(0.2, 0.1, 0.9),
                Vec3::new(0.0, 0.3, 1.0),
                4.0,
                0.4,
            ),
            (Vec3::ZERO, Vec3::Z, Vec3::Z, -1.0, 0.2),
        ];
        for (sp, vd, n, hit, r) in cases {
            close3(
                virtual_reflection_point(sp, vd, n, hit, r, mp),
                golden::virtual_reflection_point(sp, vd, n, hit, r, &p),
            );
        }
    }
}

#[test]
fn mirror_view_parallax_matches_golden() {
    let point = Vec3::ZERO;
    let prev = Vec3::new(0.0, 0.0, 5.0);
    let cases = [
        (prev, prev),
        (prev, Vec3::new(0.5, 0.0, 5.0)),
        (prev, Vec3::new(3.0, 0.0, 5.0)),
        (prev, Vec3::new(0.0, 0.0, 5.0)),
        (Vec3::new(0.0, 0.0, 1.0e-13), Vec3::new(2.0, 1.0, 0.0)),
    ];
    for (a, b) in cases {
        close(
            view_parallax(a, b, point),
            golden::view_parallax(a, b, point),
        );
    }
    // Non-vacuous: larger camera sweep => larger parallax.
    let small = golden::view_parallax(prev, Vec3::new(0.5, 0.0, 5.0), point);
    let large = golden::view_parallax(prev, Vec3::new(3.0, 0.0, 5.0), point);
    assert!(large > small && small > 0.0);
}

#[test]
fn mirror_reprojection_confidence_matches_golden() {
    for p in param_sweep() {
        let mp = mirror_params(&p);
        for &r in &[0.0_f32, 0.2, 0.5, 0.9] {
            for &px in &[0.0_f32, 0.05, 0.2, 0.5, 1.0, 1.0e4] {
                let m = reprojection_confidence(r, px, mp);
                let g = golden::reprojection_confidence(r, px, &p);
                close(m, g);
                assert!((0.0..=1.0).contains(&g));
            }
        }
    }
    // Non-vacuous: a mirror loses confidence as parallax grows.
    let p = ReprojectParams::default();
    assert!(
        golden::reprojection_confidence(0.0, 0.0, &p)
            > golden::reprojection_confidence(0.0, 0.5, &p)
    );
}

#[test]
fn mirror_blend_reprojected_position_matches_golden() {
    for p in param_sweep() {
        let mp = mirror_params(&p);
        let surf = Vec3::new(1.0, -2.0, 0.5);
        let virt = Vec3::new(10.0, 4.0, -3.0);
        for &r in &[0.0_f32, 0.25, 0.5, 0.75, 1.0] {
            close3(
                blend_reprojected_position(surf, virt, r, mp),
                golden::blend_reprojected_position(surf, virt, r, &p),
            );
        }
    }
}

#[test]
fn mirror_reproject_specular_matches_golden() {
    let p = ReprojectParams::default();
    let mp = mirror_params(&p);
    let cases = [
        (
            Vec3::ZERO,
            Vec3::new(0.05, 0.0, 0.0),
            Vec3::Z,
            Vec3::Z,
            2.0_f32,
            0.3_f32,
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::new(0.3, 0.0, 5.0),
        ),
        (
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.9, 1.1, 0.0),
            Vec3::new(0.2, 0.1, 0.9),
            Vec3::new(0.0, 0.2, 1.0),
            6.0,
            0.05,
            Vec3::new(1.0, 0.0, 8.0),
            Vec3::new(2.5, 0.0, 8.0),
        ),
    ];
    for (sp, sr, vd, n, hit, r, pc, cc) in cases {
        let m = reproject_specular(sp, sr, vd, n, hit, r, pc, cc, mp);
        let g = golden::reproject_specular(sp, sr, vd, n, hit, r, pc, cc, &p);
        close3(m.sample_position, g.sample_position);
        close3(m.virtual_point, g.virtual_point);
        close(m.parallax, g.parallax);
        close(m.confidence, g.confidence);
    }
}

#[test]
fn degenerate_inputs_match_golden_both_sides() {
    let p = ReprojectParams::default();
    let mp = mirror_params(&p);
    let nan = Vec3::splat(f32::NAN);
    let inf = Vec3::splat(f32::INFINITY);

    // view_parallax with non-finite inputs.
    close(
        view_parallax(nan, inf, Vec3::ZERO),
        golden::view_parallax(nan, inf, Vec3::ZERO),
    );

    // Full pipeline with pathological inputs: both sides finite and in range.
    let m = reproject_specular(
        nan,
        inf,
        Vec3::ZERO,
        Vec3::ZERO,
        -5.0,
        2.0,
        nan,
        Vec3::ZERO,
        mp,
    );
    let g = golden::reproject_specular(
        nan,
        inf,
        Vec3::ZERO,
        Vec3::ZERO,
        -5.0,
        2.0,
        nan,
        Vec3::ZERO,
        &p,
    );
    close3(m.sample_position, g.sample_position);
    close3(m.virtual_point, g.virtual_point);
    close(m.parallax, g.parallax);
    close(m.confidence, g.confidence);
    assert!(g.sample_position.is_finite());
    assert!(g.virtual_point.is_finite());
    assert!(g.parallax.is_finite());
    assert!((0.0..=1.0).contains(&g.confidence));
}
