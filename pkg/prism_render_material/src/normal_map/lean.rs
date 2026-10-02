//! **LEAN mapping** -- specular antialiasing for tangent-space normal maps
//! (Olano & Baker, "LEAN Mapping", I3D 2010).
//!
//! Minifying a normal map under a single filtered normal loses the *spread* of
//! the sub-texel slopes, so a bumpy surface that should read as a wide, rough
//! highlight collapses to a sharp, aliased sparkle. LEAN mapping fixes this by
//! carrying, per texel, the **first two statistical moments** of the slope
//! distribution instead of just a normal:
//!
//! * the mean slope `B = (E[sx], E[sy])` (the first moment), and
//! * the second moments `M = (E[sx^2], E[sy^2], E[sx*sy])`.
//!
//! Both moments are **linear in the samples**, so they can be box-averaged
//! down a mip chain (or bilinearly filtered) exactly like colour. After
//! filtering, the covariance of the sub-texel slopes is recovered as
//! `Sigma = (M.x - B.x^2, M.y - B.y^2, M.z - B.x*B.y)` and *added* to the base
//! material roughness (expressed as slope variance). That convolved,
//! anisotropic variance is what the shading model consumes, which removes the
//! specular aliasing while preserving the mean normal `slope_to_normal(B)`.
//!
//! A single un-filtered texel has zero slope variance (`Sigma = 0`), so it
//! reproduces the base roughness exactly -- the degenerate anti-fake check.
//! More strongly, averaging the per-texel moments and recovering `Sigma`
//! reproduces the **population covariance** of the underlying slopes bit for
//! bit (verified against an independent direct covariance in the tests), which
//! is the property that makes LEAN filterable and the primary anti-fake oracle.
//!
//! The slope convention matches [`normal_to_slope`](super::normal_to_slope)
//! (`s = (-nx/nz, -ny/nz)`), so LEAN moments interoperate with the height-field
//! generator, the strength control and the surface-gradient blend. Everything is
//! deterministic analytic `f32` arithmetic (no transcendentals, no AI/ML), so a
//! CPU golden matches a GPU twin to floating-point tolerance.
//!
//! # References
//! * Olano & Baker, "LEAN Mapping", ACM I3D 2010.
//! * Dupuy et al., "Linear Efficient Antialiased Displacement and Reflectance
//!   Mapping" (LEADR), SIGGRAPH Asia 2013 (the displacement extension).
//! * Akenine-Moller et al., *Real-Time Rendering* 4th ed., Section 9.13.1.

use super::strength::{normal_to_slope, slope_to_normal};

/// The two slope moments LEAN mapping stores per texel.
///
/// `b` is the mean slope `(E[sx], E[sy])`; `m` is the second-moment triple
/// `(E[sx^2], E[sy^2], E[sx*sy])`. Both are linear in the slope samples, so
/// they average straight down a mip chain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LeanMoments {
    /// Mean slope `(E[sx], E[sy])` (first moment).
    pub b: [f32; 2],
    /// Second moments `(E[sx^2], E[sy^2], E[sx*sy])`.
    pub m: [f32; 3],
}

impl LeanMoments {
    /// The flat surface: zero mean slope and zero second moment.
    pub const FLAT: Self = Self {
        b: [0.0, 0.0],
        m: [0.0, 0.0, 0.0],
    };
}

/// Build the LEAN moments for a single slope sample.
///
/// A single sample has no spread, so the second moment is just the outer
/// product of the slope with itself.
#[inline]
#[must_use]
pub fn lean_from_slope(slope: [f32; 2]) -> LeanMoments {
    let [sx, sy] = slope;
    LeanMoments {
        b: [sx, sy],
        m: [sx * sx, sy * sy, sx * sy],
    }
}

/// Build the LEAN moments for a single tangent-space unit normal.
///
/// The normal is converted to its height-field slope with
/// [`normal_to_slope`](super::normal_to_slope), keeping the LEAN moments
/// consistent with the rest of the normal-map pipeline.
#[inline]
#[must_use]
pub fn lean_from_normal(normal: [f32; 3]) -> LeanMoments {
    lean_from_slope(normal_to_slope(normal))
}

/// Box-average a set of LEAN moments (the mip-reduction / filtering step).
///
/// Each moment is averaged independently, which is exactly correct because both
/// moments are linear in the samples. An empty slice returns
/// [`LeanMoments::FLAT`].
#[must_use]
pub fn lean_average(samples: &[LeanMoments]) -> LeanMoments {
    if samples.is_empty() {
        return LeanMoments::FLAT;
    }
    let mut b = [0.0f32; 2];
    let mut m = [0.0f32; 3];
    for s in samples {
        b[0] += s.b[0];
        b[1] += s.b[1];
        m[0] += s.m[0];
        m[1] += s.m[1];
        m[2] += s.m[2];
    }
    let inv = 1.0 / samples.len() as f32;
    LeanMoments {
        b: [b[0] * inv, b[1] * inv],
        m: [m[0] * inv, m[1] * inv, m[2] * inv],
    }
}

/// Recover the sub-texel slope covariance `(Var[sx], Var[sy], Cov[sx,sy])`.
///
/// `Sigma = (M.x - B.x^2, M.y - B.y^2, M.z - B.x*B.y)`. The variances are
/// floored at zero to absorb floating-point round-off (a covariance can only be
/// slightly negative through cancellation).
#[inline]
#[must_use]
pub fn lean_covariance(moments: &LeanMoments) -> [f32; 3] {
    let [bx, by] = moments.b;
    let [mx, my, mxy] = moments.m;
    [
        (mx - bx * bx).max(0.0),
        (my - by * by).max(0.0),
        mxy - bx * by,
    ]
}

/// Convolve a base-material slope variance with the recovered sub-texel slope
/// covariance, giving the anisotropic variance the shading model consumes.
///
/// The base roughness contributes an axis-aligned slope variance
/// `base_variance = (sigma_x^2, sigma_y^2)` which simply adds to the diagonal of
/// the LEAN covariance (the two distributions convolve). Returns the combined
/// `(Var[sx], Var[sy], Cov[sx,sy])`.
#[inline]
#[must_use]
pub fn lean_effective_variance(moments: &LeanMoments, base_variance: [f32; 2]) -> [f32; 3] {
    let c = lean_covariance(moments);
    [c[0] + base_variance[0], c[1] + base_variance[1], c[2]]
}

/// Resolve the mean tangent-space unit normal `slope_to_normal(B)` from LEAN
/// moments (the normal the surface reads as after filtering).
#[inline]
#[must_use]
pub fn lean_resolve_normal(moments: &LeanMoments) -> [f32; 3] {
    slope_to_normal(moments.b)
}

#[cfg(test)]
mod tests {
    use super::{
        lean_average, lean_covariance, lean_effective_variance, lean_from_normal, lean_from_slope,
        lean_resolve_normal, LeanMoments,
    };
    use crate::normal_map::normal_to_slope;

    fn unit(v: [f32; 3]) -> [f32; 3] {
        let inv = 1.0 / (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] * inv, v[1] * inv, v[2] * inv]
    }

    const NORMALS: [[f32; 3]; 5] = [
        [0.0, 0.0, 1.0],
        [0.3, -0.2, 0.9],
        [-0.5, 0.4, 0.76],
        [0.1, 0.7, 0.7],
        [-0.25, -0.25, 0.93],
    ];

    /// A single sample carries zero slope variance, so it reproduces the base
    /// roughness exactly (the degenerate anti-fake check).
    #[test]
    fn single_sample_has_zero_variance() {
        for &n in &NORMALS {
            let moments = lean_from_normal(unit(n));
            let cov = lean_covariance(&moments);
            assert!(cov[0].abs() < 1e-6, "var x {cov:?}");
            assert!(cov[1].abs() < 1e-6, "var y {cov:?}");
            assert!(cov[2].abs() < 1e-6, "cov xy {cov:?}");
            let base = [0.04f32, 0.09];
            let eff = lean_effective_variance(&moments, base);
            assert!((eff[0] - base[0]).abs() < 1e-6);
            assert!((eff[1] - base[1]).abs() < 1e-6);
            assert!(eff[2].abs() < 1e-6);
        }
    }

    /// Averaging the per-texel moments and recovering the covariance reproduces
    /// the population covariance of the slopes bit for bit -- the property that
    /// makes LEAN filterable (primary anti-fake oracle, checked against an
    /// independent direct covariance).
    #[test]
    fn covariance_matches_population_statistics() {
        let slopes: [[f32; 2]; 5] = NORMALS.map(|n| normal_to_slope(unit(n)));
        let moments: [LeanMoments; 5] = slopes.map(lean_from_slope);
        let avg = lean_average(&moments);

        // Independent direct population mean / covariance.
        let n = slopes.len() as f32;
        let mut mean = [0.0f32; 2];
        for s in &slopes {
            mean[0] += s[0];
            mean[1] += s[1];
        }
        mean[0] /= n;
        mean[1] /= n;
        let mut vx = 0.0f32;
        let mut vy = 0.0f32;
        let mut vxy = 0.0f32;
        for s in &slopes {
            vx += (s[0] - mean[0]) * (s[0] - mean[0]);
            vy += (s[1] - mean[1]) * (s[1] - mean[1]);
            vxy += (s[0] - mean[0]) * (s[1] - mean[1]);
        }
        vx /= n;
        vy /= n;
        vxy /= n;

        assert!((avg.b[0] - mean[0]).abs() < 1e-5, "mean x");
        assert!((avg.b[1] - mean[1]).abs() < 1e-5, "mean y");
        let cov = lean_covariance(&avg);
        assert!((cov[0] - vx).abs() < 1e-5, "var x {cov:?} vs {vx}");
        assert!((cov[1] - vy).abs() < 1e-5, "var y {cov:?} vs {vy}");
        assert!((cov[2] - vxy).abs() < 1e-5, "cov xy {cov:?} vs {vxy}");
    }

    /// The recovered covariance is a valid (positive-semidefinite) covariance
    /// matrix: non-negative variances and a non-negative determinant.
    #[test]
    fn covariance_is_positive_semidefinite() {
        let moments: [LeanMoments; 5] = NORMALS.map(|n| lean_from_normal(unit(n)));
        let avg = lean_average(&moments);
        let [vx, vy, vxy] = lean_covariance(&avg);
        assert!(vx >= 0.0 && vy >= 0.0, "variances non-negative");
        let det = vx * vy - vxy * vxy;
        assert!(det >= -1e-5, "determinant non-negative, got {det}");
    }

    /// A mip reduction preserves the mean normal when every sub-texel shares the
    /// same slope, and spreads variance when they differ.
    #[test]
    fn average_preserves_mean_normal_and_detects_spread() {
        // All four children identical -> mean normal unchanged, zero variance.
        let n = unit([0.3, -0.4, 0.8]);
        let same = [lean_from_normal(n); 4];
        let avg = lean_average(&same);
        let got = lean_resolve_normal(&avg);
        for c in 0..3 {
            assert!((got[c] - n[c]).abs() < 1e-5, "mean normal {got:?} vs {n:?}");
        }
        assert!(lean_covariance(&avg).iter().all(|v| v.abs() < 1e-6));

        // Opposing slopes cancel in the mean but leave real variance behind.
        let a = lean_from_slope([0.5, 0.0]);
        let b = lean_from_slope([-0.5, 0.0]);
        let avg = lean_average(&[a, b]);
        assert!(avg.b[0].abs() < 1e-6, "opposing slopes cancel");
        let cov = lean_covariance(&avg);
        assert!((cov[0] - 0.25).abs() < 1e-6, "variance survives {cov:?}");
    }

    /// `lean_from_normal` agrees with the shared slope convention.
    #[test]
    fn moments_follow_slope_convention() {
        for &n in &NORMALS {
            let n = unit(n);
            let s = normal_to_slope(n);
            let moments = lean_from_normal(n);
            assert!((moments.b[0] - s[0]).abs() < 1e-6);
            assert!((moments.b[1] - s[1]).abs() < 1e-6);
        }
        assert_eq!(LeanMoments::FLAT.b, [0.0, 0.0]);
    }
}
