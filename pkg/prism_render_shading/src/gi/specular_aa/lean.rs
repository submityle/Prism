//! LEAN / LEADR normal-distribution mapping — CPU golden reference.
//!
//! LEAN mapping (Olano & Baker 2010) and its displacement-aware successor
//! LEADR (Dupuy et al. 2013) filter *bump* detail by tracking the statistics of
//! the micro-normal **slope** rather than the normal itself.  A slope
//! `(sx, sy) = (nx/nz, ny/nz)` is a linear coordinate, so slopes average
//! correctly under minification: storing the first moment `B = ⟨(sx, sy)⟩` and
//! the second moment `M = ⟨(sx², sy², sx·sy)⟩` lets a mip chain reconstruct the
//! sub-texel slope distribution's covariance
//!
//! ```text
//! Σ = M − B⊗B =  [ Mxx − Bx²      Mxy − Bx·By ]
//!                [ Mxy − Bx·By    Myy − By²   ].
//! ```
//!
//! `Σ` is the covariance of the micro-slope distribution.  Because a GGX/
//! Beckmann lobe is itself a slope-space Gaussian with variance `alpha²/2`, the
//! bump covariance adds *directly* to the base lobe variance, giving an
//! anisotropic filtered width
//!
//! ```text
//! alpha_x² = alpha_base² + 2·Σxx,      alpha_y² = alpha_base² + 2·Σyy,
//! ```
//!
//! with the off-diagonal `Σxy` encoding the lobe's orientation (recovered here
//! via a closed-form 2×2 eigen-decomposition).  Rounding error or aggressive
//! quantisation can push the sampled `Σ` slightly indefinite, so it is clamped
//! to the nearest positive-semi-definite matrix before use.
//!
//! # Conventions
//! * `no_std`: allocation-free; uses [`bevy_math::Vec2`] / [`bevy_math::Vec3`]
//!   only as plain data carriers.  No transcendental functions are needed, so
//!   [`bevy_math::ops`] is *not* imported; the inherent `f32::sqrt` suffices.
//! * The second moment `M` is packed into a `Vec3` as `(Mxx, Myy, Mxy)` and a
//!   covariance likewise as `(Σxx, Σyy, Σxy)`.
//! * A near-vertical micro-normal (`|nz| → 0`) would make the slope diverge;
//!   `nz` is floored in magnitude so a grazing normal yields a large-but-finite
//!   slope instead of `inf`.
//! * Perceptual `roughness ∈ [0, 1]` maps to the GGX width `alpha = roughness²`
//!   through [`roughness_to_alpha`]; this module never re-derives it.
//! * Every covariance is PSD-clamped and every roughness is clamped to `[0, 1]`,
//!   so results are always finite and physically valid (never `NaN`/`inf`).
//!
//! # References
//! * Marc Olano & Dan Baker 2010, *LEAN Mapping* (I3D) — linear efficient
//!   antialiased normal mapping via slope moments.
//! * Jonathan Dupuy et al. 2013, *Linear Efficient Antialiased Displacement and
//!   Reflectance Mapping* (LEADR) — the displacement-aware extension.

use crate::gi::spec_gi::ggx_lobe::{MIN_ALPHA, roughness_to_alpha};
use bevy_math::{Vec2, Vec3, ops};

/// Smallest `|nz|` used when forming a slope `n.xy / n.z`.  Floors the divisor
/// so a grazing micro-normal produces a large-but-finite slope.
pub const MIN_NZ: f32 = 1.0e-4;

/// Largest absolute slope component retained.  Caps the divergence of
/// `n.xy / n.z` near the horizon so the moments stay numerically bounded.
pub const MAX_SLOPE: f32 = 1.0e3;

/// Projects a (not necessarily normalised) micro-normal to its slope
/// `(n.x/n.z, n.y/n.z)`, flooring `|n.z|` at [`MIN_NZ`] and clamping each
/// component to `±`[`MAX_SLOPE`].
///
/// A back-facing or non-finite normal collapses to a zero slope (flat).
#[inline]
pub fn normal_to_slope(normal: Vec3) -> Vec2 {
    if !normal.is_finite() {
        return Vec2::ZERO;
    }
    let nz = if normal.z.abs() < MIN_NZ {
        if normal.z < 0.0 { -MIN_NZ } else { MIN_NZ }
    } else {
        normal.z
    };
    let sx = (normal.x / nz).clamp(-MAX_SLOPE, MAX_SLOPE);
    let sy = (normal.y / nz).clamp(-MAX_SLOPE, MAX_SLOPE);
    Vec2::new(sx, sy)
}

/// Running accumulator of LEAN slope moments.
///
/// Accumulates the weighted first moment `B = Σ wᵢ sᵢ / Σ wᵢ` and second moment
/// `M = Σ wᵢ (sx², sy², sx·sy) / Σ wᵢ`.  Call [`LeanMoments::accumulate`] for
/// each contributing micro-normal, then read the covariance / roughness.
#[derive(Clone, Copy, Debug, Default)]
pub struct LeanMoments {
    /// First moment `B = ⟨slope⟩`.
    pub b: Vec2,
    /// Second moment `M = ⟨(sx², sy², sx·sy)⟩`, packed as `(Mxx, Myy, Mxy)`.
    pub m: Vec3,
    /// Total accumulated weight (`Σ wᵢ`).
    pub weight: f32,
}

impl LeanMoments {
    /// Builds moments from a single micro-normal (unit weight).
    #[inline]
    pub fn from_normal(normal: Vec3) -> Self {
        let mut acc = Self::default();
        acc.accumulate(normal, 1.0);
        acc
    }

    /// Folds one weighted micro-normal into the running moments.
    ///
    /// `weight` is floored at `0`; non-finite normals/weights are ignored so a
    /// bad sample cannot corrupt the accumulation.
    #[inline]
    pub fn accumulate(&mut self, normal: Vec3, weight: f32) {
        let w = if weight.is_finite() { weight.max(0.0) } else { 0.0 };
        if w == 0.0 {
            return;
        }
        let s = normal_to_slope(normal);
        self.b += s * w;
        self.m += Vec3::new(s.x * s.x, s.y * s.y, s.x * s.y) * w;
        self.weight += w;
    }

    /// Returns the normalised moments `(B, M)` by dividing out the total weight.
    ///
    /// With zero accumulated weight this returns flat moments `(0, 0)`.
    #[inline]
    pub fn normalized(&self) -> (Vec2, Vec3) {
        if self.weight <= 0.0 {
            return (Vec2::ZERO, Vec3::ZERO);
        }
        let inv = 1.0 / self.weight;
        (self.b * inv, self.m * inv)
    }

    /// Micro-slope covariance `Σ = M − B⊗B`, clamped to the nearest PSD matrix.
    ///
    /// Returned packed as `(Σxx, Σyy, Σxy)`.
    #[inline]
    pub fn covariance(&self) -> Vec3 {
        let (b, m) = self.normalized();
        let raw = Vec3::new(
            m.x - b.x * b.x,
            m.y - b.y * b.y,
            m.z - b.x * b.y,
        );
        clamp_psd(raw)
    }

    /// Maps the accumulated bump covariance onto a base perceptual roughness,
    /// returning two-axis perceptual roughness `(roughness_x, roughness_y)`.
    ///
    /// `alpha_axis² = alpha_base² + 2·Σaxis`, then converted back to perceptual
    /// roughness.  Both axes are `≥` the base roughness (bump detail only
    /// softens) and clamped to `[0, 1]`.
    #[inline]
    pub fn to_anisotropic_roughness(&self, base_roughness: f32) -> (f32, f32) {
        let cov = self.covariance();
        covariance_to_anisotropic_roughness(base_roughness, cov)
    }
}

/// Clamps a packed symmetric 2×2 covariance `(Σxx, Σyy, Σxy)` to the nearest
/// positive-semi-definite matrix via its closed-form eigen-decomposition.
///
/// Negative eigenvalues (from quantisation / rounding) are raised to `0` and
/// the matrix reconstructed, guaranteeing non-negative variance along every
/// direction.  Non-finite inputs collapse to a zero covariance.
#[inline]
pub fn clamp_psd(cov: Vec3) -> Vec3 {
    if !cov.is_finite() {
        return Vec3::ZERO;
    }
    let a = cov.x;
    let b = cov.y;
    let c = cov.z;
    let half_tr = 0.5 * (a + b);
    let diff = 0.5 * (a - b);
    let radius = (diff * diff + c * c).max(0.0).sqrt();
    let l0 = half_tr - radius;
    let l1 = half_tr + radius;
    let c0 = l0.max(0.0);
    let c1 = l1.max(0.0);
    if radius < 1.0e-12 {
        // Isotropic (degenerate) — eigenvectors are arbitrary; the clamped
        // matrix is just the clamped mean on the diagonal.
        let v = c1.max(c0);
        return Vec3::new(v, v, 0.0);
    }
    // Reconstruct Σ' = Σ cᵢ vᵢ vᵢᵀ from the normalised eigenvectors.  For the
    // symmetric 2×2 case the eigenvector angle θ satisfies tan(2θ) = 2c/(a−b);
    // the direction cosines fall out of (diff, c) without a trig call.
    let inv_r = 1.0 / radius;
    // Unit eigenvector components for the larger eigenvalue l1.
    let ux2 = 0.5 * (1.0 + diff * inv_r); // vx²
    let uy2 = 0.5 * (1.0 - diff * inv_r); // vy²
    let uxy = 0.5 * c * inv_r; // vx·vy
    let xx = c1 * ux2 + c0 * uy2;
    let yy = c1 * uy2 + c0 * ux2;
    let xy = (c1 - c0) * uxy;
    Vec3::new(xx.max(0.0), yy.max(0.0), xy)
}

/// Principal-axis decomposition of a packed covariance `(Σxx, Σyy, Σxy)`.
///
/// Returns `(major_variance, minor_variance, angle)` where `angle` is the
/// orientation (radians) of the major axis measured from `+x`, and
/// `major ≥ minor ≥ 0`.  Input is PSD-clamped first.
#[inline]
pub fn principal_axes(cov: Vec3) -> (f32, f32, f32) {
    let c = clamp_psd(cov);
    let a = c.x;
    let b = c.y;
    let off = c.z;
    let half_tr = 0.5 * (a + b);
    let diff = 0.5 * (a - b);
    let radius = (diff * diff + off * off).max(0.0).sqrt();
    let major = (half_tr + radius).max(0.0);
    let minor = (half_tr - radius).max(0.0);
    // Orientation of the major axis from the symmetric 2x2 half-angle
    // identity tan(2theta) = 2*off / (a - b); atan2 goes through `ops`.
    let angle = 0.5 * ops::atan2(2.0 * off, a - b);
    (major, minor, angle)
}

/// Adds a bump covariance `(Σxx, Σyy, Σxy)` to a base perceptual roughness,
/// returning two-axis perceptual roughness `(roughness_x, roughness_y)`.
///
/// `alpha_axis² = alpha_base² + 2·Σaxis` (axis-aligned); the result is clamped
/// so each axis is a valid GGX width `≥` the base and `≤ 1`.
#[inline]
pub fn covariance_to_anisotropic_roughness(base_roughness: f32, cov: Vec3) -> (f32, f32) {
    let c = clamp_psd(cov);
    let alpha = roughness_to_alpha(base_roughness);
    let base_sq = alpha * alpha;
    let floor = MIN_ALPHA * MIN_ALPHA;
    let ax_sq = (base_sq + 2.0 * c.x).clamp(floor, 1.0);
    let ay_sq = (base_sq + 2.0 * c.y).clamp(floor, 1.0);
    let rx = ax_sq.max(0.0).sqrt().sqrt().clamp(0.0, 1.0);
    let ry = ay_sq.max(0.0).sqrt().sqrt().clamp(0.0, 1.0);
    (rx, ry)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-5;

    #[test]
    fn flat_normal_has_zero_slope() {
        let s = normal_to_slope(Vec3::Z);
        assert!(s.length() < EPS);
    }

    #[test]
    fn single_normal_has_zero_covariance() {
        // One sample ⇒ M = B⊗B ⇒ Σ = 0.
        let n = Vec3::new(0.2, -0.1, 1.0).normalize();
        let moments = LeanMoments::from_normal(n);
        let cov = moments.covariance();
        assert!(cov.length() < 1.0e-4, "cov = {cov:?}");
    }

    #[test]
    fn two_opposing_normals_give_positive_variance() {
        let mut m = LeanMoments::default();
        m.accumulate(Vec3::new(0.3, 0.0, 1.0).normalize(), 1.0);
        m.accumulate(Vec3::new(-0.3, 0.0, 1.0).normalize(), 1.0);
        let cov = m.covariance();
        assert!(cov.x > 0.0, "Σxx = {}", cov.x);
        assert!(cov.y.abs() < 1.0e-4, "Σyy = {}", cov.y);
    }

    #[test]
    fn anisotropic_bump_widens_matching_axis() {
        // Variation only along x should coarsen roughness_x more than _y.
        let mut m = LeanMoments::default();
        m.accumulate(Vec3::new(0.5, 0.0, 1.0).normalize(), 1.0);
        m.accumulate(Vec3::new(-0.5, 0.0, 1.0).normalize(), 1.0);
        let (rx, ry) = m.to_anisotropic_roughness(0.1);
        assert!(rx > ry, "rx {rx} ry {ry}");
        assert!(rx >= 0.1 && ry >= 0.1 - 1.0e-4);
        assert!((0.0..=1.0).contains(&rx) && (0.0..=1.0).contains(&ry));
    }

    #[test]
    fn covariance_only_coarsens() {
        for &r in &[0.0_f32, 0.1, 0.4, 0.8, 1.0] {
            let (rx, ry) =
                covariance_to_anisotropic_roughness(r, Vec3::new(0.05, 0.02, 0.0));
            assert!(rx >= r - 1.0e-4 && ry >= r - 1.0e-4, "r {r} -> {rx},{ry}");
        }
    }

    #[test]
    fn psd_clamp_removes_negative_eigenvalue() {
        // An indefinite matrix (det < 0) must become PSD: both eigenvalues ≥ 0.
        let clamped = clamp_psd(Vec3::new(1.0, 1.0, 2.0));
        let (major, minor, _) = principal_axes(clamped);
        assert!(minor >= -1.0e-6, "minor = {minor}");
        assert!(major >= minor);
    }

    #[test]
    fn psd_clamp_preserves_valid_matrix() {
        let valid = Vec3::new(0.3, 0.2, 0.05);
        let clamped = clamp_psd(valid);
        assert!((clamped.x - valid.x).abs() < 1.0e-4);
        assert!((clamped.y - valid.y).abs() < 1.0e-4);
        assert!((clamped.z - valid.z).abs() < 1.0e-4);
    }

    #[test]
    fn principal_axes_detects_x_dominant() {
        let (major, minor, angle) = principal_axes(Vec3::new(0.4, 0.1, 0.0));
        assert!((major - 0.4).abs() < 1.0e-5);
        assert!((minor - 0.1).abs() < 1.0e-5);
        assert!(angle.abs() < 1.0e-4, "angle = {angle}");
    }

    #[test]
    fn grazing_normal_is_finite() {
        let s = normal_to_slope(Vec3::new(1.0, 0.0, 0.0));
        assert!(s.is_finite());
        assert!(s.x.abs() <= MAX_SLOPE);
    }

    #[test]
    fn defends_against_garbage() {
        let mut m = LeanMoments::default();
        m.accumulate(Vec3::splat(f32::NAN), 1.0);
        m.accumulate(Vec3::Z, f32::NAN);
        // Nothing valid accumulated through garbage ⇒ still flat.
        let cov = m.covariance();
        assert!(cov.is_finite());
        let (rx, ry) = m.to_anisotropic_roughness(0.3);
        assert!((0.0..=1.0).contains(&rx) && (0.0..=1.0).contains(&ry));
        assert!(clamp_psd(Vec3::splat(f32::INFINITY)).length() < EPS);
    }

    #[test]
    fn zero_weight_is_flat() {
        let m = LeanMoments::default();
        let (b, mm) = m.normalized();
        assert!(b.length() < EPS && mm.length() < EPS);
        let (rx, ry) = m.to_anisotropic_roughness(0.25);
        assert!((rx - 0.25).abs() < 1.0e-3 && (ry - 0.25).abs() < 1.0e-3);
    }
}
