//! Adaptive probe-grid trilinear interpolation with geometry-aware weights.
//!
//! At shading time an adaptive probe volume gathers the eight probes forming
//! the grid cell that encloses a shading point and blends their irradiance.
//! A naive trilinear blend leaks light through thin walls and across concave
//! corners, so (following DDGI/RTXGI and Unity APV) each corner's trilinear
//! weight is multiplied by two geometry-aware factors:
//!
//! * a **normal / back-face weight** that fades out probes sitting *behind* the
//!   shading surface — a smooth `(0.5 + 0.5 * dot(dir_to_probe, normal))^2`
//!   wrap that keeps a probe only partially contributing as it rotates behind
//!   the surface, and
//! * a **depth (Chebyshev) weight** reusing the sibling
//!   [`chebyshev_weight`](crate::gi::world_space::visibility::chebyshev_weight)
//!   so a probe occluded from the point by nearer geometry is suppressed.
//!
//! The combined weights are renormalised to sum to one; when every weight
//! collapses to zero (a fully degenerate configuration) the blend falls back
//! to a uniform average so the result is always finite and leak-free.
//!
//! # Conventions
//! * The grid cell is axis-aligned; `frac` holds the trilinear fractions in
//!   `[0, 1]^3` of the shading point inside the cell, and corner `c` is indexed
//!   by its `(x, y, z)` bits with bit 0 = x, bit 1 = y, bit 2 = z.
//! * Each [`ProbeCorner`] carries its world position, an [`ShL1Irradiance`]
//!   probe, and the two depth moments (`mean`, `mean_sq`) of the occluder along
//!   the direction from the probe toward the shading point.
//! * Weights are clamped non-negative; `frac` is clamped to `[0, 1]`; a
//!   degenerate zero-length normal disables the normal term (weight `1`).
//! * Every item is a deterministic pure function: no RNG, no I/O, no GPU, no
//!   `unsafe`, and no allocation.

use bevy_math::Vec3;

use crate::gi::world_space::visibility::chebyshev_weight;

use super::sh_irradiance::ShL1Irradiance;

/// One corner probe of a grid cell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProbeCorner {
    /// World-space position of the probe centre.
    pub position: Vec3,
    /// The probe's L1 irradiance payload.
    pub sh: ShL1Irradiance,
    /// Mean occluder depth `E[d]` along the probe→point direction.
    pub depth_mean: f32,
    /// Mean-squared occluder depth `E[d^2]` along the probe→point direction.
    pub depth_mean_sq: f32,
}

impl ProbeCorner {
    /// Builds a corner from its position and irradiance with "fully open" depth
    /// moments (`mean = mean_sq = large`), i.e. no occlusion.
    #[inline]
    pub fn open(position: Vec3, sh: ShL1Irradiance) -> Self {
        // A very large mean makes the Chebyshev test short-circuit to visible
        // for any realistic shading distance.
        let big = 1.0e9;
        Self {
            position,
            sh,
            depth_mean: big,
            depth_mean_sq: big * big,
        }
    }
}

/// Computes the eight trilinear corner weights for cell fractions `frac`.
///
/// `frac` is clamped to `[0, 1]^3`.  Corner `c` is indexed by its `(x, y, z)`
/// bits (bit 0 = x, bit 1 = y, bit 2 = z); the returned weights always sum to
/// exactly one (partition of unity).
#[inline]
pub fn trilinear_weights(frac: Vec3) -> [f32; 8] {
    let fx = frac.x.clamp(0.0, 1.0);
    let fy = frac.y.clamp(0.0, 1.0);
    let fz = frac.z.clamp(0.0, 1.0);
    let wx = [1.0 - fx, fx];
    let wy = [1.0 - fy, fy];
    let wz = [1.0 - fz, fz];
    let mut out = [0.0f32; 8];
    for c in 0..8 {
        let ix = c & 1;
        let iy = (c >> 1) & 1;
        let iz = (c >> 2) & 1;
        out[c] = wx[ix] * wy[iy] * wz[iz];
    }
    out
}

/// Smooth back-face weight for a probe seen from a shading point.
///
/// Returns `(0.5 + 0.5 * dot(dir_to_probe, normal))^2`, the DDGI "wrap"
/// weight: `1` when the probe lies along the surface normal, fading smoothly to
/// `0` as it rotates behind the surface.  A degenerate (zero-length) normal or
/// a coincident probe/point returns `1` so the term is a no-op.
#[inline]
pub fn normal_backface_weight(point: Vec3, normal: Vec3, probe_position: Vec3) -> f32 {
    let to_probe = probe_position - point;
    let len_sq = to_probe.length_squared();
    let n_len_sq = normal.length_squared();
    if len_sq <= f32::MIN_POSITIVE || n_len_sq <= f32::MIN_POSITIVE {
        return 1.0;
    }
    let dir = to_probe * len_sq.sqrt().recip();
    let n = normal * n_len_sq.sqrt().recip();
    let wrap = 0.5 + 0.5 * dir.dot(n);
    let w = wrap.max(0.0);
    w * w
}

/// Computes the fully combined, renormalised corner weights for a blend.
///
/// For each corner the trilinear weight is multiplied by its
/// [`normal_backface_weight`] and its Chebyshev depth weight (distance = the
/// probe→point distance against the stored moments).  The eight products are
/// renormalised to sum to one; if their sum is non-positive (every corner
/// rejected) a uniform `1/8` fallback is returned instead.
#[inline]
pub fn resolve_corner_weights(
    corners: &[ProbeCorner; 8],
    frac: Vec3,
    point: Vec3,
    normal: Vec3,
) -> [f32; 8] {
    let tri = trilinear_weights(frac);
    let mut combined = [0.0f32; 8];
    let mut sum = 0.0f32;
    for c in 0..8 {
        let corner = &corners[c];
        let n_w = normal_backface_weight(point, normal, corner.position);
        let distance = (corner.position - point).length();
        let d_w = chebyshev_weight(corner.depth_mean, corner.depth_mean_sq, distance);
        let w = (tri[c] * n_w * d_w).max(0.0);
        combined[c] = w;
        sum += w;
    }
    if sum > f32::MIN_POSITIVE {
        let inv = sum.recip();
        for w in combined.iter_mut() {
            *w *= inv;
        }
    } else {
        // Every corner rejected: fall back to a uniform blend.
        combined = [0.125; 8];
    }
    combined
}

/// Blends the eight corner probes into a single irradiance for a shading point.
///
/// The corner irradiances (each evaluated at `normal`) are combined with the
/// weights from [`resolve_corner_weights`].  The result is always finite and
/// non-negative; a degenerate configuration falls back to the uniform average.
#[inline]
pub fn sample_probe_grid(
    corners: &[ProbeCorner; 8],
    frac: Vec3,
    point: Vec3,
    normal: Vec3,
) -> Vec3 {
    let weights = resolve_corner_weights(corners, frac, point, normal);
    let mut out = Vec3::ZERO;
    for c in 0..8 {
        out += corners[c].sh.eval_irradiance(normal) * weights[c];
    }
    Vec3::new(out.x.max(0.0), out.y.max(0.0), out.z.max(0.0))
}

/// Blends the eight corner probes' SH coefficients, then evaluates irradiance.
///
/// Unlike [`sample_probe_grid`] this blends in SH space (coefficient-wise)
/// before the cosine convolution, which better preserves directional content
/// when the eight probes differ.  Falls back to the uniform average when the
/// weights degenerate.
#[inline]
pub fn blend_probe_sh(
    corners: &[ProbeCorner; 8],
    frac: Vec3,
    point: Vec3,
    normal: Vec3,
) -> ShL1Irradiance {
    let weights = resolve_corner_weights(corners, frac, point, normal);
    let mut blended = ShL1Irradiance::ZERO;
    for c in 0..8 {
        blended.add_scaled(&corners[c].sh, weights[c]);
    }
    blended
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_corners(color: [f32; 3]) -> [ProbeCorner; 8] {
        let sh = ShL1Irradiance::from_constant(color);
        core::array::from_fn(|c| {
            let ix = (c & 1) as f32;
            let iy = ((c >> 1) & 1) as f32;
            let iz = ((c >> 2) & 1) as f32;
            ProbeCorner::open(Vec3::new(ix, iy, iz), sh)
        })
    }

    #[test]
    fn trilinear_partition_of_unity() {
        for &frac in &[
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(0.3, 0.7, 0.5),
            Vec3::new(0.9, 0.1, 0.25),
        ] {
            let w = trilinear_weights(frac);
            let sum: f32 = w.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "sum {sum} for {frac:?}");
            for wi in w {
                assert!((0.0..=1.0).contains(&wi), "weight {wi}");
            }
        }
    }

    #[test]
    fn trilinear_corner_identity() {
        // At each cell corner the matching trilinear weight is 1, rest 0.
        for c in 0..8 {
            let fx = (c & 1) as f32;
            let fy = ((c >> 1) & 1) as f32;
            let fz = ((c >> 2) & 1) as f32;
            let w = trilinear_weights(Vec3::new(fx, fy, fz));
            for (i, wi) in w.iter().enumerate() {
                if i == c {
                    assert!((wi - 1.0).abs() < 1e-6, "corner {c}: w={wi}");
                } else {
                    assert!(wi.abs() < 1e-6, "corner {c} other {i}: w={wi}");
                }
            }
        }
    }

    #[test]
    fn frac_is_clamped() {
        let inside = trilinear_weights(Vec3::new(0.5, 0.5, 0.5));
        let over = trilinear_weights(Vec3::new(2.0, 2.0, 2.0));
        let under = trilinear_weights(Vec3::new(-2.0, -2.0, -2.0));
        let sum_i: f32 = inside.iter().sum();
        assert!((sum_i - 1.0).abs() < 1e-6);
        // Over-range clamps to the (1,1,1) corner (index 7).
        assert!((over[7] - 1.0).abs() < 1e-6);
        // Under-range clamps to the (0,0,0) corner (index 0).
        assert!((under[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn constant_field_blends_to_constant_irradiance() {
        // All eight probes identical => any blend reproduces that irradiance.
        let corners = uniform_corners([1.0, 0.5, 0.25]);
        let expected = ShL1Irradiance::from_constant([1.0, 0.5, 0.25])
            .eval_irradiance(Vec3::Y);
        for &frac in &[
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(0.1, 0.8, 0.3),
            Vec3::new(0.0, 1.0, 0.0),
        ] {
            let point = Vec3::new(frac.x, frac.y, frac.z);
            let e = sample_probe_grid(&corners, frac, point, Vec3::Y);
            assert!((e - expected).length() < 1e-4, "{e:?} vs {expected:?}");
        }
    }

    #[test]
    fn resolve_weights_are_normalised() {
        let corners = uniform_corners([1.0, 1.0, 1.0]);
        let frac = Vec3::new(0.4, 0.6, 0.2);
        let point = Vec3::new(0.4, 0.6, 0.2);
        let w = resolve_corner_weights(&corners, frac, point, Vec3::Y);
        let sum: f32 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum {sum}");
        for wi in w {
            assert!(wi.is_finite() && wi >= 0.0);
        }
    }

    #[test]
    fn fully_occluded_falls_back_to_uniform() {
        // Zero depth moments => Chebyshev weight 0 for every positive distance.
        let sh = ShL1Irradiance::from_constant([1.0, 1.0, 1.0]);
        let corners: [ProbeCorner; 8] = core::array::from_fn(|c| {
            let ix = (c & 1) as f32;
            let iy = ((c >> 1) & 1) as f32;
            let iz = ((c >> 2) & 1) as f32;
            ProbeCorner {
                position: Vec3::new(ix, iy, iz),
                sh,
                depth_mean: 0.0,
                depth_mean_sq: 0.0,
            }
        });
        let frac = Vec3::new(0.5, 0.5, 0.5);
        let point = Vec3::new(0.5, 0.5, 0.5);
        let w = resolve_corner_weights(&corners, frac, point, Vec3::Y);
        let sum: f32 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        for wi in w {
            assert!((wi - 0.125).abs() < 1e-6, "expected uniform, got {wi}");
        }
    }

    #[test]
    fn backface_weight_fades_behind_surface() {
        let point = Vec3::ZERO;
        let normal = Vec3::Y;
        // Probe straight along the normal: full weight.
        let front = normal_backface_weight(point, normal, Vec3::new(0.0, 1.0, 0.0));
        assert!((front - 1.0).abs() < 1e-6, "front {front}");
        // Probe directly behind: fully faded.
        let back = normal_backface_weight(point, normal, Vec3::new(0.0, -1.0, 0.0));
        assert!(back.abs() < 1e-6, "back {back}");
        // Perpendicular: the wrap gives 0.25.
        let side = normal_backface_weight(point, normal, Vec3::new(1.0, 0.0, 0.0));
        assert!((side - 0.25).abs() < 1e-6, "side {side}");
        // Degenerate normal => no-op.
        let deg = normal_backface_weight(point, Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0));
        assert!((deg - 1.0).abs() < 1e-6);
    }

    #[test]
    fn blend_sh_matches_sample_for_constant_field() {
        let corners = uniform_corners([0.7, 0.2, 0.9]);
        let frac = Vec3::new(0.3, 0.3, 0.3);
        let point = Vec3::new(0.3, 0.3, 0.3);
        let blended = blend_probe_sh(&corners, frac, point, Vec3::Z);
        let via_sh = blended.eval_irradiance(Vec3::Z);
        let direct = sample_probe_grid(&corners, frac, point, Vec3::Z);
        assert!((via_sh - direct).length() < 1e-4, "{via_sh:?} vs {direct:?}");
    }

    #[test]
    fn output_is_finite_and_non_negative() {
        let corners = uniform_corners([2.0, 0.0, 1.0]);
        for &frac in &[Vec3::new(0.5, 0.5, 0.5), Vec3::new(0.9, 0.05, 0.6)] {
            let e = sample_probe_grid(&corners, frac, Vec3::splat(0.5), Vec3::new(0.1, 0.9, 0.0));
            assert!(e.x.is_finite() && e.y.is_finite() && e.z.is_finite());
            assert!(e.x >= 0.0 && e.y >= 0.0 && e.z >= 0.0);
        }
    }
}
