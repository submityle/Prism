//! WESL compile coverage and CPU-mirror parity for `spec_denoise_spatial.wesl`.
//!
//! The sandbox has no GPU, so the compile test drives the spatial-filter module
//! through the same `ShaderCache` / `wesl` pipeline the render world uses,
//! proving it parses and type-checks exactly as it will on device (the
//! `spec_denoise_spatial_probe` entry folds every helper so naga keeps them
//! live).
//!
//! The parity tests transcribe each WESL helper op-for-op into Rust and assert
//! agreement with the CPU golden
//! ([`prism_render_shading::gi::spec_denoise::spatial`]) across a sweep of valid
//! and degenerate inputs. WGSL's missing `isFinite` is modelled by
//! `(x - x) == 0.0` and the `+Z` normalise fallback, matching the shader
//! exactly; transcendentals go through `std` here (the golden uses
//! `bevy_math::ops`), so parity holds to a small floating-point tolerance. The
//! on-device `sds_spatial_filter` gathers a bounded neighbour array; the mirror
//! drives the identical accumulation over an arbitrary-length slice, matching
//! the golden `spatial_filter` signature.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_math::Vec3;
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};
use prism_render_shading::gi::spec_denoise::spatial as golden;
use prism_render_shading::gi::spec_denoise::spatial::{SpatialParams, SpecularTap};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("spec_denoise spatial shader is WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles `spec_denoise_spatial.wesl`, proving the spatial-filter module
/// parses and type-checks as it will in the render world.
#[test]
fn spec_denoise_spatial_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let spatial = shader_id(0x5350_4543_5f44_4e5f_5350_4154_5f00_0002);
    cache.set_shader(
        spatial,
        Shader::from_wesl(
            include_str!("../../shaders/spec_denoise_spatial.wesl"),
            "embedded://prism_render_scene/shaders/spec_denoise_spatial.wesl",
        ),
    );

    cache
        .get(0, spatial, &[])
        .unwrap_or_else(|error| panic!("spec_denoise_spatial.wesl failed to compile: {error}"));
}

// --- Rust transcription of the WESL maths (op-for-op) -------------------------
// Control flow, floors and finite-guards mirror `spec_denoise_spatial.wesl`
// exactly; this is what the GPU evaluates, asserted equal to the golden below.

const MIN_ALPHA: f32 = 1.0e-3;

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
fn make_tap(
    color: Vec3,
    position: Vec3,
    normal: Vec3,
    roughness: f32,
    normalized_hit: f32,
) -> SpecularTap {
    SpecularTap {
        color: sanitize_rgb(color),
        position: sanitize_vec(position),
        normal: safe_normalize(normal),
        roughness: roughness.clamp(0.0, 1.0),
        normalized_hit: sanitize_scalar(normalized_hit).clamp(0.0, 1.0),
    }
}

#[inline]
fn hit_normalizer(roughness: f32, abs_view_z: f32) -> f32 {
    let r = roughness.clamp(0.0, 1.0);
    let base = abs_view_z.max(1.0e-3);
    base * (0.5 + 1.5 * r)
}

#[inline]
fn normalize_hit_distance(hit_distance: f32, view_z: f32, roughness: f32) -> f32 {
    let hit = sanitize_scalar(hit_distance).max(0.0);
    let vz = sanitize_scalar(view_z).abs();
    let scale = hit_normalizer(roughness, vz);
    let denom = hit + scale;
    if denom <= 1.0e-12 {
        0.0
    } else {
        (hit / denom).clamp(0.0, 1.0)
    }
}

#[inline]
fn denormalize_hit_distance(normalized: f32, view_z: f32, roughness: f32) -> f32 {
    let n = sanitize_scalar(normalized).clamp(0.0, 1.0 - 1.0e-4);
    let vz = sanitize_scalar(view_z).abs();
    let scale = hit_normalizer(roughness, vz);
    (n * scale / (1.0 - n)).max(0.0)
}

#[inline]
fn contact_hardening_factor(normalized_hit: f32, strength: f32) -> f32 {
    let n = sanitize_scalar(normalized_hit).clamp(0.0, 1.0);
    let s = strength.clamp(0.0, 1.0);
    let hardened = n.sqrt();
    (1.0 - s) + s * hardened
}

#[inline]
fn blur_radius(roughness: f32, normalized_hit: f32, params: &SpatialParams) -> f32 {
    let alpha = roughness_to_alpha(roughness);
    let hardening = contact_hardening_factor(normalized_hit, params.contact_hardening);
    let radius = params.max_radius.max(0.0) * alpha * hardening;
    if is_finite(radius) {
        radius.max(0.0)
    } else {
        0.0
    }
}

#[inline]
fn anisotropic_radii(radius: f32, n_dot_v: f32, params: &SpatialParams) -> (f32, f32) {
    let r = radius.max(0.0);
    let ndv = n_dot_v.clamp(1.0e-3, 1.0);
    let max_aniso = params.max_anisotropy.max(1.0);
    let aniso = 1.0 + (max_aniso - 1.0) * (1.0 - ndv);
    let aniso = aniso.clamp(1.0, max_aniso);
    let sqrt_aniso = aniso.sqrt();
    let major = r * sqrt_aniso;
    let minor = r / sqrt_aniso;
    (major.max(0.0), minor.max(0.0))
}

#[inline]
fn depth_weight(center_pos: Vec3, center_normal: Vec3, sample_pos: Vec3, phi_depth: f32) -> f32 {
    let n = safe_normalize(center_normal);
    let plane = (sanitize_vec(sample_pos) - sanitize_vec(center_pos))
        .dot(n)
        .abs();
    let phi = phi_depth.max(1.0e-6);
    stable_exp(-plane / phi)
}

#[inline]
#[allow(clippy::disallowed_methods)]
fn normal_weight(n0: Vec3, n1: Vec3, phi_normal: f32) -> f32 {
    let a = safe_normalize(n0);
    let b = safe_normalize(n1);
    let cosine = a.dot(b).clamp(0.0, 1.0);
    cosine.powf(phi_normal.max(0.0)).clamp(0.0, 1.0)
}

#[inline]
fn roughness_weight(r0: f32, r1: f32, phi_roughness: f32) -> f32 {
    let diff = (r0.clamp(0.0, 1.0) - r1.clamp(0.0, 1.0)).abs();
    let phi = phi_roughness.max(1.0e-6);
    stable_exp(-diff / phi)
}

#[inline]
fn spatial_weight(center: &SpecularTap, sample: &SpecularTap, params: &SpatialParams) -> f32 {
    let w_depth = depth_weight(
        center.position,
        center.normal,
        sample.position,
        params.phi_depth,
    );
    let w_normal = normal_weight(center.normal, sample.normal, params.phi_normal);
    let w_rough = roughness_weight(center.roughness, sample.roughness, params.phi_roughness);
    (w_depth * w_normal * w_rough).clamp(0.0, 1.0)
}

struct MirrorResult {
    color: Vec3,
    normalized_hit: f32,
}

#[inline]
fn spatial_filter(
    center: &SpecularTap,
    center_kernel: f32,
    neighbours: &[(SpecularTap, f32)],
    params: &SpatialParams,
) -> MirrorResult {
    let w_center = sanitize_scalar(center_kernel).max(0.0);
    let mut color_sum = center.color * w_center;
    let mut hit_sum = center.normalized_hit * w_center;
    let mut weight = w_center;

    for (sample, kernel) in neighbours {
        let k = sanitize_scalar(*kernel).max(0.0);
        if k == 0.0 {
            continue;
        }
        let w = k * spatial_weight(center, sample, params);
        if w <= 0.0 {
            continue;
        }
        color_sum += sample.color * w;
        hit_sum += sample.normalized_hit * w;
        weight += w;
    }

    if weight <= 1.0e-12 {
        MirrorResult {
            color: center.color,
            normalized_hit: center.normalized_hit,
        }
    } else {
        let inv = 1.0 / weight;
        MirrorResult {
            color: sanitize_rgb(color_sum * inv),
            normalized_hit: (hit_sum * inv).clamp(0.0, 1.0),
        }
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

fn tap(color: Vec3, pos: Vec3, normal: Vec3, rough: f32, hit: f32) -> SpecularTap {
    SpecularTap::new(color, pos, normal, rough, hit)
}

fn param_sweep() -> Vec<SpatialParams> {
    vec![
        SpatialParams::default(),
        SpatialParams {
            max_radius: 8.0,
            phi_depth: 0.1,
            phi_normal: 32.0,
            phi_roughness: 0.2,
            contact_hardening: 0.5,
            max_anisotropy: 2.0,
        },
        SpatialParams {
            max_radius: 64.0,
            phi_depth: 2.0,
            phi_normal: 256.0,
            phi_roughness: 0.02,
            contact_hardening: 1.0,
            max_anisotropy: 8.0,
        },
        SpatialParams {
            max_radius: 0.0,
            phi_depth: 1.0e-6,
            phi_normal: 0.0,
            phi_roughness: 1.0e-6,
            contact_hardening: 0.0,
            max_anisotropy: 1.0,
        },
    ]
}

// --- parity tests -------------------------------------------------------------

#[test]
fn mirror_make_tap_matches_golden() {
    let cases = [
        (
            Vec3::new(0.2, 0.4, 0.6),
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::Z,
            0.4_f32,
            0.3_f32,
        ),
        (
            Vec3::new(-1.0, 2.0, f32::NAN),
            Vec3::new(f32::INFINITY, 0.0, 0.0),
            Vec3::new(0.0, 0.3, 1.0),
            2.0,
            5.0,
        ),
        (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, -1.0, -1.0),
    ];
    for (c, p, n, r, h) in cases {
        let m = make_tap(c, p, n, r, h);
        let g = SpecularTap::new(c, p, n, r, h);
        close3(m.color, g.color);
        close3(m.position, g.position);
        close3(m.normal, g.normal);
        close(m.roughness, g.roughness);
        close(m.normalized_hit, g.normalized_hit);
    }
    // Non-vacuous: a zero normal falls back to +Z.
    assert!(
        (SpecularTap::new(Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, 0.5, 0.5).normal - Vec3::Z).length()
            < 1.0e-6
    );
}

#[test]
fn mirror_roughness_to_alpha_matches_golden_via_blur() {
    // roughness_to_alpha is private in the golden; exercise it through
    // blur_radius with full radius and no hardening (hardening -> 1).
    let p = SpatialParams {
        contact_hardening: 0.0,
        ..SpatialParams::default()
    };
    for &r in &[0.0_f32, 0.1, 0.3, 0.5, 0.8, 1.0] {
        close(blur_radius(r, 1.0, &p), golden::blur_radius(r, 1.0, &p));
    }
    // Floor: alpha(0) == MIN_ALPHA -> radius == max_radius * 1e-3.
    close(blur_radius(0.0, 1.0, &p), p.max_radius * MIN_ALPHA);
}

#[test]
fn mirror_normalize_hit_distance_matches_golden() {
    let cases = [
        (0.0_f32, 10.0_f32, 0.3_f32),
        (2.5, 10.0, 0.5),
        (50.0, 10.0, 0.9),
        (5.0, -8.0, 0.1),
        (1.0, 0.0, 1.0),
    ];
    for (hit, vz, r) in cases {
        let m = normalize_hit_distance(hit, vz, r);
        let g = golden::normalize_hit_distance(hit, vz, r);
        close(m, g);
        assert!((0.0..=1.0).contains(&g));
    }
    // Non-vacuous: monotone increasing in hit distance.
    let a = golden::normalize_hit_distance(1.0, 5.0, 0.5);
    let b = golden::normalize_hit_distance(5.0, 5.0, 0.5);
    let c = golden::normalize_hit_distance(20.0, 5.0, 0.5);
    assert!(a < b && b < c);
}

#[test]
fn mirror_denormalize_hit_distance_matches_golden_and_round_trips() {
    let vz = 10.0;
    for &(hit, rough) in &[(0.0_f32, 0.3_f32), (2.5, 0.5), (50.0, 0.9)] {
        let n = normalize_hit_distance(hit, vz, rough);
        close(n, golden::normalize_hit_distance(hit, vz, rough));
        let back = denormalize_hit_distance(n, vz, rough);
        close(back, golden::denormalize_hit_distance(n, vz, rough));
        // Non-vacuous: the inverse recovers the original hit distance.
        assert!((back - hit).abs() < 1.0e-2 * (1.0 + hit));
    }
    // Clamp-away-from-one guard: a stored 1.0 stays finite.
    assert!(golden::denormalize_hit_distance(1.0, vz, 0.5).is_finite());
}

#[test]
fn mirror_contact_hardening_factor_matches_golden() {
    for &s in &[0.0_f32, 0.5, 1.0] {
        for &n in &[0.0_f32, 0.01, 0.25, 0.5, 1.0] {
            close(
                contact_hardening_factor(n, s),
                golden::contact_hardening_factor(n, s),
            );
        }
    }
    // Non-vacuous: monotone in hit, strength 0 disables hardening.
    let a = golden::contact_hardening_factor(0.0, 1.0);
    let b = golden::contact_hardening_factor(0.25, 1.0);
    let c = golden::contact_hardening_factor(1.0, 1.0);
    assert!(a < b && b < c);
    close(golden::contact_hardening_factor(0.0, 0.0), 1.0);
}

#[test]
fn mirror_blur_radius_matches_golden() {
    for p in param_sweep() {
        for &r in &[0.0_f32, 0.2, 0.5, 0.8, 1.0] {
            for &nh in &[0.0_f32, 0.1, 0.5, 1.0] {
                close(blur_radius(r, nh, &p), golden::blur_radius(r, nh, &p));
            }
        }
    }
    // Non-vacuous: a mirror stays near-zero; radius grows with roughness;
    // contact hardening shrinks a near-surface reflection.
    let p = SpatialParams::default();
    assert!(golden::blur_radius(0.0, 1.0, &p) < 0.05);
    assert!(golden::blur_radius(0.2, 1.0, &p) < golden::blur_radius(1.0, 1.0, &p));
    assert!(golden::blur_radius(0.8, 0.01, &p) < golden::blur_radius(0.8, 1.0, &p));
}

#[test]
fn mirror_anisotropic_radii_matches_golden() {
    for p in param_sweep() {
        for &radius in &[0.0_f32, 1.0, 4.0, 16.0] {
            for &ndv in &[0.0_f32, 0.05, 0.3, 0.6, 1.0] {
                let (mmaj, mmin) = anisotropic_radii(radius, ndv, &p);
                let (gmaj, gmin) = golden::anisotropic_radii(radius, ndv, &p);
                close(mmaj, gmaj);
                close(mmin, gmin);
            }
        }
    }
    // Non-vacuous: head-on is isotropic; grazing elongates and conserves area.
    let p = SpatialParams::default();
    let (maj0, min0) = golden::anisotropic_radii(4.0, 1.0, &p);
    close(maj0, 4.0);
    close(min0, 4.0);
    let (maj1, min1) = golden::anisotropic_radii(4.0, 0.05, &p);
    assert!(maj1 > 4.0 && min1 < 4.0);
    assert!((maj1 * min1 - 16.0).abs() < 1.0e-3);
}

#[test]
fn mirror_depth_weight_matches_golden() {
    let center = Vec3::new(1.0, 2.0, 3.0);
    let normal = Vec3::new(0.0, 0.0, 1.0);
    let cases = [
        (Vec3::new(1.0, 2.0, 3.0), 0.5_f32),
        (Vec3::new(1.0, 2.0, 3.5), 0.5),
        (Vec3::new(5.0, 5.0, 3.0), 0.5),
        (Vec3::new(1.0, 2.0, 10.0), 0.1),
    ];
    for (sample, phi) in cases {
        close(
            depth_weight(center, normal, sample, phi),
            golden::depth_weight(center, normal, sample, phi),
        );
    }
    // Non-vacuous: in-plane sample peaks, off-plane sample decays.
    close(
        golden::depth_weight(center, normal, Vec3::new(9.0, 9.0, 3.0), 0.5),
        1.0,
    );
    assert!(golden::depth_weight(center, normal, Vec3::new(1.0, 2.0, 5.0), 0.5) < 1.0);
}

#[test]
fn mirror_normal_weight_matches_golden() {
    for &phi in &[0.0_f32, 1.0, 32.0, 128.0] {
        let cases = [
            (Vec3::Z, Vec3::Z),
            (Vec3::Z, Vec3::new(0.1, 0.0, 1.0)),
            (Vec3::Z, Vec3::new(1.0, 0.0, 0.1)),
            (Vec3::Z, Vec3::NEG_Z),
        ];
        for (a, b) in cases {
            close(normal_weight(a, b, phi), golden::normal_weight(a, b, phi));
        }
    }
    // Non-vacuous: identical normals peak, opposite normals reject.
    close(golden::normal_weight(Vec3::Z, Vec3::Z, 128.0), 1.0);
    assert!(golden::normal_weight(Vec3::Z, Vec3::NEG_Z, 128.0) < 1.0e-3);
}

#[test]
fn mirror_roughness_weight_matches_golden() {
    for &phi in &[1.0e-6_f32, 0.02, 0.08, 0.2] {
        for &(r0, r1) in &[(0.5_f32, 0.5_f32), (0.1, 0.3), (0.0, 1.0), (0.9, 0.95)] {
            close(
                roughness_weight(r0, r1, phi),
                golden::roughness_weight(r0, r1, phi),
            );
        }
    }
    // Non-vacuous: matched roughness peaks, large break rejects.
    close(golden::roughness_weight(0.5, 0.5, 0.08), 1.0);
    assert!(golden::roughness_weight(0.0, 1.0, 0.08) < 1.0e-3);
}

#[test]
fn mirror_spatial_weight_matches_golden() {
    let p = SpatialParams::default();
    let center = tap(Vec3::splat(0.5), Vec3::ZERO, Vec3::Z, 0.4, 0.5);
    let samples = [
        tap(Vec3::splat(0.5), Vec3::ZERO, Vec3::Z, 0.4, 0.5),
        tap(
            Vec3::splat(0.5),
            Vec3::new(0.0, 0.0, 2.0),
            Vec3::Z,
            0.4,
            0.5,
        ),
        tap(Vec3::splat(0.5), Vec3::ZERO, Vec3::NEG_Z, 0.4, 0.5),
        tap(Vec3::splat(0.5), Vec3::ZERO, Vec3::Z, 0.95, 0.5),
    ];
    for s in &samples {
        close(
            spatial_weight(&center, s, &p),
            golden::spatial_weight(&center, s, &p),
        );
    }
    // Non-vacuous: identical tap peaks; normal/roughness breaks reject.
    close(golden::spatial_weight(&center, &center, &p), 1.0);
    assert!(golden::spatial_weight(&center, &samples[2], &p) < 1.0e-3);
    assert!(golden::spatial_weight(&center, &samples[3], &p) < 0.5);
}

#[test]
fn mirror_spatial_filter_matches_golden() {
    let p = SpatialParams::default();
    let center = tap(Vec3::new(0.2, 0.4, 0.6), Vec3::ZERO, Vec3::Z, 0.5, 0.3);
    let neighbours = vec![
        (
            tap(
                Vec3::splat(1.0),
                Vec3::new(0.1, 0.0, 0.0),
                Vec3::Z,
                0.5,
                0.8,
            ),
            1.0_f32,
        ),
        (
            tap(
                Vec3::splat(0.0),
                Vec3::new(-0.1, 0.0, 0.0),
                Vec3::Z,
                0.5,
                0.2,
            ),
            0.5,
        ),
        (
            tap(
                Vec3::splat(0.5),
                Vec3::new(0.0, 0.1, 0.0),
                Vec3::Z,
                0.55,
                0.4,
            ),
            0.75,
        ),
        // Rejected by a normal break -> contributes ~0 weight.
        (
            tap(
                Vec3::splat(9.0),
                Vec3::new(0.0, 0.0, 5.0),
                Vec3::NEG_Z,
                0.9,
                1.0,
            ),
            1.0,
        ),
    ];
    let m = spatial_filter(&center, 1.0, &neighbours, &p);
    let g = golden::spatial_filter(&center, 1.0, &neighbours, &p);
    close3(m.color, g.color);
    close(m.normalized_hit, g.normalized_hit);

    // Empty neighbourhood -> identity on centre.
    let g_empty = golden::spatial_filter(&center, 1.0, &[], &p);
    let m_empty = spatial_filter(&center, 1.0, &[], &p);
    close3(m_empty.color, g_empty.color);
    close3(g_empty.color, center.color);

    // Equal-weight matching neighbour -> midpoint.
    let c0 = tap(Vec3::splat(0.0), Vec3::ZERO, Vec3::Z, 0.5, 0.2);
    let nb = tap(
        Vec3::splat(1.0),
        Vec3::new(0.1, 0.0, 0.0),
        Vec3::Z,
        0.5,
        0.8,
    );
    let g_mid = golden::spatial_filter(&c0, 1.0, &[(nb, 1.0)], &p);
    assert!((g_mid.color - Vec3::splat(0.5)).length() < 1.0e-2);
    assert!((g_mid.normalized_hit - 0.5).abs() < 1.0e-2);
}

#[test]
fn mirror_spatial_filter_is_finite_on_degenerate_inputs() {
    let p = SpatialParams::default();
    let center = SpecularTap::new(
        Vec3::splat(f32::NAN),
        Vec3::splat(f32::INFINITY),
        Vec3::ZERO,
        2.0,
        5.0,
    );
    let nb = SpecularTap::new(
        Vec3::splat(f32::INFINITY),
        Vec3::splat(f32::NAN),
        Vec3::ZERO,
        -1.0,
        -1.0,
    );
    let m = spatial_filter(&center, f32::NAN, &[(nb, f32::INFINITY)], &p);
    let g = golden::spatial_filter(&center, f32::NAN, &[(nb, f32::INFINITY)], &p);
    close3(m.color, g.color);
    close(m.normalized_hit, g.normalized_hit);
    assert!(g.color.is_finite());
    assert!(g.normalized_hit.is_finite());
    assert!((0.0..=1.0).contains(&g.normalized_hit));
}
