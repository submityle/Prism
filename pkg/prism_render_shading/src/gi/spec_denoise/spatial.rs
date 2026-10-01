//! Hit-distance-driven specular spatial filter — CPU golden.
//!
//! After temporal accumulation a specular signal still needs a spatial pass to
//! clean under-converged pixels, but — unlike diffuse — the blur must respect
//! the reflection's geometry.  Two signals drive it (NVIDIA NRD ReBLUR
//! specular):
//!
//! * **Roughness** sets the base kernel size: a mirror must stay razor sharp
//!   (near-zero radius) while a rough surface tolerates a wide blur.
//! * **Hit distance** performs *contact hardening*: a reflected object close to
//!   the surface (small hit distance) projects a sharp, nearby image and must
//!   be blurred little; a distant reflection may be blurred more.  The hit
//!   distance is stored *normalised* ([`normalize_hit_distance`]) so it
//!   survives temporal accumulation, then rebuilt on read.
//!
//! The kernel is additionally **anisotropic**: at grazing angles the reflection
//! stretches along the view azimuth, so [`anisotropic_radii`] elongates the
//! footprint.  Edge-stopping [`spatial_weight`] bilateral terms (plane distance,
//! normal, roughness) keep the blur from crossing geometry or material breaks,
//! and [`spatial_filter`] reconstructs both the filtered colour and a filtered
//! normalised hit distance in one gather.
//!
//! # Conventions
//! * Linear RGB `f32` radiance; world-space positions and unit normals, all
//!   matching the WESL/GPU twin bit-for-bit.
//! * `roughness ∈ [0, 1]` perceptual; GGX `alpha = roughness²` via
//!   [`crate::gi::spec_gi::ggx_lobe::roughness_to_alpha`].
//! * Transcendentals go through [`bevy_math::ops`]; `sqrt` is inherent `f32`.
//! * Every helper is pure (no RNG / IO / GPU / globals / `unsafe`), defends
//!   against degeneracy (empty neighbourhoods, zero normals, non-finite input),
//!   and never emits `NaN`; weights are clamped to `[0, 1]` and radii `≥ 0`.

use bevy_math::{ops, Vec3};

use crate::gi::spec_gi::ggx_lobe::roughness_to_alpha;

/// Tunables for the specular spatial filter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpatialParams {
    /// Maximum blur radius (world units) reached by a fully rough surface.
    pub max_radius: f32,
    /// Plane-distance tolerance (world units) for the depth bilateral term.
    pub phi_depth: f32,
    /// Normal weight exponent; higher preserves normal edges more sharply.
    pub phi_normal: f32,
    /// Roughness tolerance for the roughness bilateral term.
    pub phi_roughness: f32,
    /// Strength of contact hardening: how strongly a small normalised hit
    /// distance shrinks the radius (`0` disables, `1` is full).
    pub contact_hardening: f32,
    /// Maximum anisotropic elongation at grazing angles (ratio ≥ 1).
    pub max_anisotropy: f32,
}

impl Default for SpatialParams {
    fn default() -> Self {
        Self {
            max_radius: 32.0,
            phi_depth: 0.5,
            phi_normal: 128.0,
            phi_roughness: 0.08,
            contact_hardening: 1.0,
            max_anisotropy: 3.0,
        }
    }
}

/// One specular neighbour tap: colour, geometry, material, and the normalised
/// hit distance to reconstruct.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpecularTap {
    /// Linear RGB radiance of the tap.
    pub color: Vec3,
    /// World-space surface position.
    pub position: Vec3,
    /// Unit surface normal.
    pub normal: Vec3,
    /// Perceptual roughness in `[0, 1]`.
    pub roughness: f32,
    /// Normalised hit distance in `[0, 1]` (see [`normalize_hit_distance`]).
    pub normalized_hit: f32,
}

impl SpecularTap {
    /// Construct a tap, sanitising colour/position and normalising the normal.
    #[must_use]
    pub fn new(
        color: Vec3,
        position: Vec3,
        normal: Vec3,
        roughness: f32,
        normalized_hit: f32,
    ) -> Self {
        Self {
            color: sanitize_rgb(color),
            position: sanitize_vec(position),
            normal: safe_normalize(normal),
            roughness: roughness.clamp(0.0, 1.0),
            normalized_hit: sanitize_scalar(normalized_hit).clamp(0.0, 1.0),
        }
    }
}

/// Normalise a world-space hit distance into `[0, 1]` for stable storage.
///
/// ReBLUR stores `hit / (hit + f(roughness)·|view_z|)` so the value is scale
/// and depth-robust: a rough surface's normaliser is larger (its hits read as
/// relatively closer), keeping the stored value well-conditioned across the
/// frame.  `view_z` is the linear view-space depth of the shaded pixel.
#[must_use]
pub fn normalize_hit_distance(hit_distance: f32, view_z: f32, roughness: f32) -> f32 {
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

/// Inverse of [`normalize_hit_distance`]: rebuild a world-space hit distance.
///
/// `normalized ∈ [0, 1]`; the value is clamped away from `1` so the inverse
/// never divides by zero (a stored `1` would mean an infinite hit).
#[must_use]
pub fn denormalize_hit_distance(normalized: f32, view_z: f32, roughness: f32) -> f32 {
    let n = sanitize_scalar(normalized).clamp(0.0, 1.0 - 1.0e-4);
    let vz = sanitize_scalar(view_z).abs();
    let scale = hit_normalizer(roughness, vz);
    // hit = n·scale / (1 - n), from hit/(hit+scale) = n.
    (n * scale / (1.0 - n)).max(0.0)
}

/// Contact-hardening factor in `[0, 1]` from a normalised hit distance.
///
/// A small normalised hit (reflection hugging the surface) returns a small
/// factor → tight blur; a large one returns `≈ 1` → full blur.  `strength`
/// blends between no hardening (`0` → always `1`) and full hardening (`1`).
#[must_use]
pub fn contact_hardening_factor(normalized_hit: f32, strength: f32) -> f32 {
    let n = sanitize_scalar(normalized_hit).clamp(0.0, 1.0);
    let s = strength.clamp(0.0, 1.0);
    // Lerp between 1 (no hardening) and the sqrt-shaped hit response.
    let hardened = n.sqrt();
    (1.0 - s) + s * hardened
}

/// World-space blur radius for one pixel.
///
/// Scales [`SpatialParams::max_radius`] by the GGX width (`alpha = roughness²`)
/// so a mirror blurs essentially not at all, and by the contact-hardening
/// factor so near-surface reflections stay sharp.  Always finite and `≥ 0`.
#[must_use]
pub fn blur_radius(roughness: f32, normalized_hit: f32, params: &SpatialParams) -> f32 {
    let alpha = roughness_to_alpha(roughness);
    let hardening = contact_hardening_factor(normalized_hit, params.contact_hardening);
    let radius = params.max_radius.max(0.0) * alpha * hardening;
    if radius.is_finite() {
        radius.max(0.0)
    } else {
        0.0
    }
}

/// Split a scalar blur radius into anisotropic `(major, minor)` radii.
///
/// At grazing angles (`n_dot_v → 0`) the reflection footprint stretches, so the
/// major axis grows up to `max_anisotropy·radius` while the minor axis shrinks
/// to conserve the footprint area (`major·minor ≈ radius²`).  Head-on
/// (`n_dot_v → 1`) the kernel is isotropic (`major = minor = radius`).
#[must_use]
pub fn anisotropic_radii(radius: f32, n_dot_v: f32, params: &SpatialParams) -> (f32, f32) {
    let r = radius.max(0.0);
    let ndv = n_dot_v.clamp(1.0e-3, 1.0);
    let max_aniso = params.max_anisotropy.max(1.0);
    // Elongation grows as the view grazes; sqrt keeps it gentle.
    let aniso = 1.0 + (max_aniso - 1.0) * (1.0 - ndv);
    let aniso = aniso.clamp(1.0, max_aniso);
    let major = r * aniso.sqrt();
    let minor = r / aniso.sqrt();
    (major.max(0.0), minor.max(0.0))
}

/// Plane-distance edge-stopping weight in `[0, 1]`.
///
/// Measures how far `sample_pos` lies off the tangent plane through
/// `center_pos`/`center_normal`, normalised by `phi_depth`:
/// `exp(-|plane_distance| / phi_depth)`.
#[must_use]
pub fn depth_weight(center_pos: Vec3, center_normal: Vec3, sample_pos: Vec3, phi_depth: f32) -> f32 {
    let n = safe_normalize(center_normal);
    let plane = (sanitize_vec(sample_pos) - sanitize_vec(center_pos)).dot(n).abs();
    let phi = phi_depth.max(1.0e-6);
    stable_exp(-plane / phi)
}

/// Normal edge-stopping weight in `[0, 1]`: `max(0, n0·n1)^phi_normal`.
#[must_use]
pub fn normal_weight(n0: Vec3, n1: Vec3, phi_normal: f32) -> f32 {
    let a = safe_normalize(n0);
    let b = safe_normalize(n1);
    let cosine = a.dot(b).clamp(0.0, 1.0);
    ops::powf(cosine, phi_normal.max(0.0)).clamp(0.0, 1.0)
}

/// Roughness edge-stopping weight in `[0, 1]`: `exp(-|r0 - r1| / phi_roughness)`.
#[must_use]
pub fn roughness_weight(r0: f32, r1: f32, phi_roughness: f32) -> f32 {
    let diff = (r0.clamp(0.0, 1.0) - r1.clamp(0.0, 1.0)).abs();
    let phi = phi_roughness.max(1.0e-6);
    stable_exp(-diff / phi)
}

/// Combined specular bilateral weight for one neighbour against the centre.
///
/// Product of the plane-distance, normal, and roughness terms, all finite and
/// in `[0, 1]`.  (Luminance is deliberately *not* an edge-stop here: specular
/// radiance varies legitimately across a smooth lobe, so a luminance stop would
/// defeat the blur.)
#[must_use]
pub fn spatial_weight(center: &SpecularTap, sample: &SpecularTap, params: &SpatialParams) -> f32 {
    let w_depth = depth_weight(center.position, center.normal, sample.position, params.phi_depth);
    let w_normal = normal_weight(center.normal, sample.normal, params.phi_normal);
    let w_rough = roughness_weight(center.roughness, sample.roughness, params.phi_roughness);
    (w_depth * w_normal * w_rough).clamp(0.0, 1.0)
}

/// Filtered specular output: reconstructed colour and normalised hit distance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpatialResult {
    /// Edge-stopping weighted-mean colour.
    pub color: Vec3,
    /// Edge-stopping weighted-mean normalised hit distance in `[0, 1]`.
    pub normalized_hit: f32,
}

/// One edge-stopping spatial gather over a specular neighbourhood.
///
/// Each neighbour carries a precomputed kernel weight (e.g. a Gaussian of its
/// offset within the [`anisotropic_radii`] footprint); the bilateral
/// [`spatial_weight`] multiplies it.  Reconstructs the colour *and* the
/// normalised hit distance with the same weights so the two stay consistent.
/// Falls back to the centre tap (identity) when every weight collapses, so the
/// filter never produces `NaN`.
#[must_use]
pub fn spatial_filter(
    center: &SpecularTap,
    center_kernel: f32,
    neighbours: &[(SpecularTap, f32)],
    params: &SpatialParams,
) -> SpatialResult {
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
        SpatialResult {
            color: center.color,
            normalized_hit: center.normalized_hit,
        }
    } else {
        let inv = 1.0 / weight;
        SpatialResult {
            color: sanitize_rgb(color_sum * inv),
            normalized_hit: (hit_sum * inv).clamp(0.0, 1.0),
        }
    }
}

// ---------------------------------------------------------------------------
// Defensive numeric helpers (private).
// ---------------------------------------------------------------------------

/// Hit-distance normaliser shared by [`normalize_hit_distance`] and its
/// inverse: a roughness-boosted multiple of the view depth, floored positive.
#[must_use]
fn hit_normalizer(roughness: f32, abs_view_z: f32) -> f32 {
    let r = roughness.clamp(0.0, 1.0);
    // Rougher surfaces get a larger normaliser (hits read relatively closer).
    let base = abs_view_z.max(1.0e-3);
    base * (0.5 + 1.5 * r)
}

/// Numerically safe `exp` on `(-inf, 0]`; never returns `NaN`/`+inf`.
#[must_use]
fn stable_exp(x: f32) -> f32 {
    if !x.is_finite() {
        return 0.0;
    }
    ops::exp(x.clamp(-80.0, 0.0))
}

/// Normalise `v`, falling back to `+Z` for zero-length / non-finite input.
#[must_use]
fn safe_normalize(v: Vec3) -> Vec3 {
    let len_sq = v.length_squared();
    if len_sq.is_finite() && len_sq > 1.0e-24 {
        v / len_sq.sqrt()
    } else {
        Vec3::Z
    }
}

/// Replace any non-finite component of an RGB triple with `0`, clamped `≥ 0`.
#[must_use]
fn sanitize_rgb(c: Vec3) -> Vec3 {
    Vec3::new(finite_or_zero(c.x), finite_or_zero(c.y), finite_or_zero(c.z)).max(Vec3::ZERO)
}

/// Replace any non-finite component of a position with `0`.
#[must_use]
fn sanitize_vec(v: Vec3) -> Vec3 {
    Vec3::new(finite_or_zero(v.x), finite_or_zero(v.y), finite_or_zero(v.z))
}

/// Replace a non-finite scalar with `0`.
#[must_use]
fn sanitize_scalar(x: f32) -> f32 {
    finite_or_zero(x)
}

#[must_use]
fn finite_or_zero(x: f32) -> f32 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-5;

    fn params() -> SpatialParams {
        SpatialParams::default()
    }

    fn tap(color: Vec3, pos: Vec3, normal: Vec3, rough: f32, hit: f32) -> SpecularTap {
        SpecularTap::new(color, pos, normal, rough, hit)
    }

    #[test]
    fn mirror_radius_is_near_zero() {
        let p = params();
        let r = blur_radius(0.0, 1.0, &p);
        // alpha(0) = MIN_ALPHA = 1e-3 -> radius = 32 * 1e-3 = 0.032.
        assert!(r < 0.05);
    }

    #[test]
    fn radius_increases_with_roughness() {
        let p = params();
        let r_low = blur_radius(0.2, 1.0, &p);
        let r_mid = blur_radius(0.5, 1.0, &p);
        let r_high = blur_radius(1.0, 1.0, &p);
        assert!(r_low < r_mid && r_mid < r_high);
    }

    #[test]
    fn contact_hardening_shrinks_near_surface() {
        let p = params();
        let near = blur_radius(0.8, 0.01, &p);
        let far = blur_radius(0.8, 1.0, &p);
        assert!(near < far);
    }

    #[test]
    fn contact_hardening_factor_monotone_and_bounded() {
        let a = contact_hardening_factor(0.0, 1.0);
        let b = contact_hardening_factor(0.25, 1.0);
        let c = contact_hardening_factor(1.0, 1.0);
        assert!(a < b && b < c);
        assert!((0.0..=1.0).contains(&a));
        assert!((c - 1.0).abs() < EPS);
        // strength 0 disables hardening -> always 1.
        assert!((contact_hardening_factor(0.0, 0.0) - 1.0).abs() < EPS);
    }

    #[test]
    fn hit_distance_round_trip() {
        let vz = 10.0;
        for &(hit, rough) in &[(0.0_f32, 0.3_f32), (2.5, 0.5), (50.0, 0.9)] {
            let n = normalize_hit_distance(hit, vz, rough);
            assert!((0.0..=1.0).contains(&n));
            let back = denormalize_hit_distance(n, vz, rough);
            assert!((back - hit).abs() < 1e-2 * (1.0 + hit));
        }
    }

    #[test]
    fn normalize_hit_is_monotone_in_distance() {
        let a = normalize_hit_distance(1.0, 5.0, 0.5);
        let b = normalize_hit_distance(5.0, 5.0, 0.5);
        let c = normalize_hit_distance(20.0, 5.0, 0.5);
        assert!(a < b && b < c);
    }

    #[test]
    fn anisotropy_head_on_is_isotropic() {
        let p = params();
        let (maj, min) = anisotropic_radii(4.0, 1.0, &p);
        assert!((maj - 4.0).abs() < EPS);
        assert!((min - 4.0).abs() < EPS);
    }

    #[test]
    fn anisotropy_grazing_elongates_and_preserves_area() {
        let p = params();
        let (maj, min) = anisotropic_radii(4.0, 0.05, &p);
        assert!(maj > 4.0 && min < 4.0);
        // major * minor conserved at radius^2.
        assert!((maj * min - 16.0).abs() < 1e-3);
    }

    #[test]
    fn weights_peak_for_identical_tap() {
        let p = params();
        let c = tap(Vec3::splat(0.5), Vec3::ZERO, Vec3::Z, 0.4, 0.5);
        let w = spatial_weight(&c, &c, &p);
        assert!((w - 1.0).abs() < EPS);
    }

    #[test]
    fn weight_rejects_normal_and_roughness_breaks() {
        let p = params();
        let c = tap(Vec3::splat(0.5), Vec3::ZERO, Vec3::Z, 0.1, 0.5);
        let flipped = tap(Vec3::splat(0.5), Vec3::ZERO, Vec3::NEG_Z, 0.1, 0.5);
        let rough = tap(Vec3::splat(0.5), Vec3::ZERO, Vec3::Z, 0.9, 0.5);
        assert!(spatial_weight(&c, &flipped, &p) < 1e-3);
        assert!(spatial_weight(&c, &rough, &p) < 0.5);
    }

    #[test]
    fn filter_identity_on_empty_neighbourhood() {
        let p = params();
        let c = tap(Vec3::new(0.2, 0.4, 0.6), Vec3::ZERO, Vec3::Z, 0.5, 0.3);
        let out = spatial_filter(&c, 1.0, &[], &p);
        assert!((out.color - c.color).length() < EPS);
        assert!((out.normalized_hit - c.normalized_hit).abs() < EPS);
    }

    #[test]
    fn filter_rejects_disjoint_neighbour() {
        let p = params();
        let c = tap(Vec3::splat(1.0), Vec3::ZERO, Vec3::Z, 0.1, 0.5);
        // Opposite normal -> weight ~0 -> output stays at centre.
        let bad = tap(Vec3::splat(100.0), Vec3::new(0.0, 0.0, 50.0), Vec3::NEG_Z, 0.9, 1.0);
        let out = spatial_filter(&c, 1.0, &[(bad, 1.0)], &p);
        assert!((out.color - c.color).length() < 1e-2);
    }

    #[test]
    fn filter_averages_matching_neighbour() {
        let p = params();
        let c = tap(Vec3::splat(0.0), Vec3::ZERO, Vec3::Z, 0.5, 0.2);
        let nb = tap(Vec3::splat(1.0), Vec3::new(0.1, 0.0, 0.0), Vec3::Z, 0.5, 0.8);
        let out = spatial_filter(&c, 1.0, &[(nb, 1.0)], &p);
        // Equal weights -> midpoint for both colour and hit distance.
        assert!((out.color - Vec3::splat(0.5)).length() < 1e-2);
        assert!((out.normalized_hit - 0.5).abs() < 1e-2);
    }

    #[test]
    fn filter_is_finite_on_degenerate_inputs() {
        let p = params();
        let c = SpecularTap::new(
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
        let out = spatial_filter(&c, f32::NAN, &[(nb, f32::INFINITY)], &p);
        assert!(out.color.is_finite());
        assert!(out.normalized_hit.is_finite());
        assert!((0.0..=1.0).contains(&out.normalized_hit));
    }

    #[test]
    fn determinism() {
        let p = params();
        let c = tap(Vec3::splat(0.5), Vec3::ZERO, Vec3::Z, 0.3, 0.4);
        let nb = [(tap(Vec3::splat(0.7), Vec3::new(0.2, 0.0, 0.0), Vec3::Z, 0.3, 0.5), 0.5)];
        let a = spatial_filter(&c, 1.0, &nb, &p);
        let b = spatial_filter(&c, 1.0, &nb, &p);
        assert_eq!(a, b);
    }
}
