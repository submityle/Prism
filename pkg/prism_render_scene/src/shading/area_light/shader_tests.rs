//! `WESL` compilation coverage and `CPU` math-mirror parity for the area-light
//! `LTC` subsystem.
//!
//! The sandbox has no `GPU`, so these tests compile `area_light_ltc.wesl`
//! through the same `ShaderCache` / `wesl` pipeline the render world uses,
//! proving the kernel parses and type-checks exactly as it will on device. The
//! source is self-contained (no intra-crate `import`s, matching the
//! surface-cache / world-space `GI` kernels), so a green compile also pins the
//! inlined `LTC` maths.
//!
//! The math-mirror tests transcribe the scalar `WESL` `LTC` kernel
//! (`integrate_edge` / `polygon_form_factor_quad` / `ltc_apply` /
//! `ltc_evaluate_quad`) op-for-op into Rust and cross-check it against the `CPU`
//! golden [`prism_render_shading::gi::area_light::polygon`] and
//! [`prism_render_shading::gi::area_light::ltc_lut`], so a divergence between
//! the on-device kernel and the reference fails here.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_math::{ops, Vec3};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};
use prism_render_shading::gi::area_light::ltc_lut::{fit_ltc_default, LtcCoeffs};
use prism_render_shading::gi::area_light::polygon::{
    integrate_edge, polygon_form_factor, quad_ltc_evaluate, rectangle_points,
};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("area-light shaders are WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

#[test]
fn area_light_ltc_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);
    let id = shader_id(0x5052_4953_4d5f_414c_5f4c_5443_0001_0001);
    cache.set_shader(
        id,
        Shader::from_wesl(
            include_str!("../../shaders/area_light_ltc.wesl"),
            "embedded://prism_render_scene/shaders/area_light_ltc.wesl",
        ),
    );
    cache
        .get(0, id, &[])
        .unwrap_or_else(|error| panic!("area_light_ltc.wesl failed to compile: {error}"));
}

// --- CPU mirror of the inlined WESL maths (transcribed op-for-op) -----------

/// Near-parallel edge directions span zero solid angle (`WESL` `MIN_SIN`).
const MIN_SIN: f32 = 1.0e-7;
/// Vertices this close to the shaded point are dropped (`WESL` `MIN_LEN_SQ`).
const MIN_LEN_SQ: f32 = 1.0e-18;
const TWO_PI: f32 = 6.283_185_5;

fn mirror_integrate_edge_vec(v1: Vec3, v2: Vec3) -> Vec3 {
    let cos_theta = v1.dot(v2).clamp(-1.0, 1.0);
    let theta = ops::acos(cos_theta);
    let cross_v = v1.cross(v2);
    let sin_theta = cross_v.length();
    if sin_theta < MIN_SIN {
        return Vec3::ZERO;
    }
    let result = cross_v * (theta / sin_theta);
    if result.is_finite() {
        result
    } else {
        Vec3::ZERO
    }
}

fn mirror_integrate_edge(v1: Vec3, v2: Vec3) -> f32 {
    mirror_integrate_edge_vec(v1, v2).z
}

fn mirror_intersect_horizon(a: Vec3, b: Vec3) -> Option<Vec3> {
    let dz = a.z - b.z;
    if dz.abs() < 1.0e-12 {
        return None;
    }
    let t = a.z / dz;
    if !(0.0..=1.0).contains(&t) {
        return None;
    }
    let p = a + (b - a) * t;
    if p.is_finite() {
        Some(p)
    } else {
        None
    }
}

fn mirror_polygon_form_factor_quad(input: [Vec3; 4]) -> f32 {
    let n = input.len();
    let mut clipped: Vec<Vec3> = Vec::with_capacity(8);
    for i in 0..n {
        let cur = input[i];
        let prev = input[(i + n - 1) % n];
        let cur_in = cur.z >= 0.0;
        let prev_in = prev.z >= 0.0;
        if cur_in {
            if !prev_in && let Some(p) = mirror_intersect_horizon(prev, cur) {
                clipped.push(p);
            }
            clipped.push(cur);
        } else if prev_in && let Some(p) = mirror_intersect_horizon(prev, cur) {
            clipped.push(p);
        }
    }
    if clipped.len() < 3 {
        return 0.0;
    }
    let mut dirs: Vec<Vec3> = Vec::with_capacity(8);
    for p in &clipped {
        let len_sq = p.dot(*p);
        if len_sq <= MIN_LEN_SQ {
            continue;
        }
        let d = *p / ops::sqrt(len_sq);
        if d.dot(d) > 0.5 {
            dirs.push(d);
        }
    }
    let k = dirs.len();
    if k < 3 {
        return 0.0;
    }
    let mut sum = 0.0f32;
    for i in 0..k {
        let v1 = dirs[i];
        let v2 = dirs[(i + 1) % k];
        sum += mirror_integrate_edge(v1, v2);
    }
    let f = sum / TWO_PI;
    if f.is_finite() {
        f.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn mirror_ltc_apply(c: &LtcCoeffs, d: Vec3) -> Vec3 {
    Vec3::new(
        c.a00 * d.x + c.a02 * d.z,
        c.a11 * d.y,
        c.a20 * d.x + c.a22 * d.z,
    )
}

fn mirror_ltc_evaluate_quad(pts: [Vec3; 4], c: &LtcCoeffs) -> f32 {
    let mut t = [Vec3::ZERO; 4];
    for i in 0..4 {
        let v = mirror_ltc_apply(c, pts[i]);
        t[i] = if v.is_finite() { v } else { Vec3::ZERO };
    }
    let f = mirror_polygon_form_factor_quad(t);
    let amplitude = c.amplitude.clamp(0.0, 1.0);
    let r = amplitude * f;
    if r.is_finite() {
        r.max(0.0)
    } else {
        0.0
    }
}

// --- Cross-checks vs the CPU golden -----------------------------------------

#[test]
fn mirror_edge_integral_matches_golden() {
    let samples = [
        (Vec3::new(0.3, 0.1, 0.95), Vec3::new(-0.2, 0.4, 0.89)),
        (Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.5, 0.5, 0.70710677)),
        (Vec3::new(-0.6, 0.2, 0.77), Vec3::new(0.1, -0.9, 0.42)),
    ];
    for (a, b) in samples {
        let a = a.normalize();
        let b = b.normalize();
        let mirror = mirror_integrate_edge(a, b);
        let golden = integrate_edge(a, b);
        assert!(
            (mirror - golden).abs() < 1e-6,
            "edge mirror={mirror} golden={golden}"
        );
    }
}

#[test]
fn mirror_form_factor_matches_golden_quad() {
    let quads = [
        rectangle_points(Vec3::new(0.4, 0.0, 1.0), Vec3::X, Vec3::Y, 0.5, 0.5),
        rectangle_points(Vec3::new(0.0, 0.0, 1.0), Vec3::X, Vec3::Y, 1.0, 1.0),
        rectangle_points(Vec3::new(0.0, 0.0, 1.0), Vec3::X, Vec3::Y, 0.2, 0.2),
        // Straddles the horizon so the clip path is exercised.
        [
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
        ],
    ];
    for quad in quads {
        let mirror = mirror_polygon_form_factor_quad(quad);
        let golden = polygon_form_factor(&quad);
        assert!(
            (mirror - golden).abs() < 1e-6,
            "form-factor mirror={mirror} golden={golden}"
        );
    }
}

#[test]
fn mirror_ltc_evaluate_matches_golden_quad() {
    let coeffs = [
        LtcCoeffs::IDENTITY,
        fit_ltc_default(0.5, 0.4),
        fit_ltc_default(0.8, 0.15),
        fit_ltc_default(0.2, 0.9),
    ];
    let quad = rectangle_points(Vec3::new(0.3, 0.1, 0.8), Vec3::X, Vec3::Y, 0.4, 0.6);
    for c in coeffs {
        let mirror = mirror_ltc_evaluate_quad(quad, &c);
        let golden = quad_ltc_evaluate(quad[0], quad[1], quad[2], quad[3], &c);
        assert!(
            (mirror - golden).abs() < 1e-6,
            "ltc mirror={mirror} golden={golden}"
        );
    }
}

#[test]
fn mirror_identity_ltc_equals_diffuse_form_factor() {
    // With the identity transform the LTC evaluation collapses to the
    // clamped-cosine form factor (amplitude 1), matching the golden anchor.
    let quad = rectangle_points(Vec3::new(0.4, 0.0, 1.0), Vec3::X, Vec3::Y, 0.5, 0.5);
    let ltc = mirror_ltc_evaluate_quad(quad, &LtcCoeffs::IDENTITY);
    let ff = mirror_polygon_form_factor_quad(quad);
    assert!((ltc - ff).abs() < 1e-6, "ltc={ltc} ff={ff}");
    // And the golden agrees with both.
    let golden_ff = polygon_form_factor(&quad);
    assert!((ff - golden_ff).abs() < 1e-6, "ff={ff} golden={golden_ff}");
}

#[test]
fn mirror_form_factor_is_bounded() {
    let big = rectangle_points(Vec3::new(0.0, 0.0, 0.02), Vec3::X, Vec3::Y, 2000.0, 2000.0);
    let f = mirror_polygon_form_factor_quad(big);
    assert!((0.0..=1.0).contains(&f), "f={f}");
    assert!(f > 0.98, "nearly-full hemisphere f={f}");
    // Fully below the horizon integrates to zero.
    let below = [
        Vec3::new(-1.0, -1.0, -0.5),
        Vec3::new(1.0, -1.0, -0.5),
        Vec3::new(1.0, 1.0, -0.5),
        Vec3::new(-1.0, 1.0, -0.5),
    ];
    assert_eq!(mirror_polygon_form_factor_quad(below), 0.0);
}
