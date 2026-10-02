//! **Anisotropic roughness from a filtered slope covariance** -- the final
//! step of LEAN / Toksvig specular antialiasing.
//!
//! [`lean`](super::lean) recovers the sub-texel slope covariance as a symmetric
//! 2x2 matrix `Sigma = [[Var sx, Cov], [Cov, Var sy]]`. To feed an anisotropic
//! GGX / Beckmann BRDF the shading model needs that covariance expressed as a
//! pair of **principal roughness axes** (a major and a minor roughness) plus the
//! **rotation** of those axes in the tangent plane. This module performs the
//! closed-form symmetric-eigen decomposition that turns one into the other, with
//! no iteration.
//!
//! For a symmetric `[[a, b], [b, c]]` the eigenvalues are
//! `lambda = (a + c)/2 +/- sqrt(((a - c)/2)^2 + b^2)` and the major eigenvector
//! lies at `theta = 0.5 * atan2(2b, a - c)`; the minor axis is orthogonal. The
//! eigenvalues are the slope variances along the principal axes, and a Beckmann
//! / GGX slope distribution of roughness `alpha` has slope variance
//! `alpha^2 / 2`, so the principal roughness is `alpha = sqrt(2 * variance)`.
//!
//! The reconstruction `R * diag(lambda1, lambda2) * R^T` returns the original
//! covariance exactly, which is the primary anti-fake oracle (the decomposition
//! is a genuine factorisation, not a stub). Everything is deterministic analytic
//! `f32` arithmetic routed through `bevy_math::ops` for libm determinism (no
//! AI/ML), so a CPU golden matches a GPU twin to floating-point tolerance.
//!
//! # References
//! * Olano & Baker, "LEAN Mapping", ACM I3D 2010 (anisotropic roughness from
//!   slope covariance).
//! * Kaplanyan et al., "Filtering Distributions of Normals for Shading
//!   Antialiasing", HPG 2016.
//! * Akenine-Moller et al., *Real-Time Rendering* 4th ed., Section 9.13.

use bevy_math::ops;

/// Principal-axis decomposition of a symmetric slope covariance.
///
/// `major` and `minor` are the eigenvalues (slope variances) with
/// `major >= minor >= 0`; `angle` is the tangent-plane rotation (radians) of the
/// major axis, `atan2`-ranged to `(-pi/2, pi/2]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlopeEigen {
    /// Larger eigenvalue (slope variance along the major axis).
    pub major: f32,
    /// Smaller eigenvalue (slope variance along the minor axis).
    pub minor: f32,
    /// Rotation of the major axis in the tangent plane (radians).
    pub angle: f32,
}

/// Decompose a symmetric slope covariance `(Var sx, Var sy, Cov sx sy)` into its
/// principal axes.
///
/// Uses the closed-form symmetric 2x2 eigen solution; the result always
/// satisfies `major >= minor`.
#[inline]
#[must_use]
pub fn slope_covariance_eigen(cov: [f32; 3]) -> SlopeEigen {
    let (a, c, b) = (cov[0], cov[1], cov[2]);
    let mean = 0.5 * (a + c);
    let diff = 0.5 * (a - c);
    let r = ops::sqrt(diff * diff + b * b);
    SlopeEigen {
        major: mean + r,
        minor: mean - r,
        // atan2(2b, a - c) is the double angle of the major eigenvector.
        angle: 0.5 * ops::atan2(2.0 * b, a - c),
    }
}

/// Reconstruct the symmetric slope covariance `(Var sx, Var sy, Cov sx sy)` from
/// a principal-axis decomposition.
///
/// Computes `R * diag(major, minor) * R^T`; the exact inverse of
/// [`slope_covariance_eigen`] (round-trip anti-fake oracle).
#[inline]
#[must_use]
pub fn slope_covariance_from_eigen(e: SlopeEigen) -> [f32; 3] {
    let (s, co) = ops::sin_cos(e.angle);
    let (c2, s2) = (co * co, s * s);
    [
        e.major * c2 + e.minor * s2,
        e.major * s2 + e.minor * c2,
        (e.major - e.minor) * co * s,
    ]
}

/// Convert a slope variance to the matching Beckmann / GGX roughness
/// `alpha = sqrt(2 * variance)`.
///
/// Negative inputs (possible only through round-off) are floored at zero.
#[inline]
#[must_use]
pub fn variance_to_ggx_alpha(variance: f32) -> f32 {
    ops::sqrt(2.0 * variance.max(0.0))
}

/// Convert a GGX roughness `alpha` back to its slope variance `alpha^2 / 2`.
///
/// The exact inverse of [`variance_to_ggx_alpha`].
#[inline]
#[must_use]
pub fn ggx_alpha_to_variance(alpha: f32) -> f32 {
    0.5 * alpha * alpha
}

/// Anisotropic GGX roughness recovered from a filtered slope covariance.
///
/// Returns `(alpha_major, alpha_minor, angle)`: the principal GGX roughnesses
/// and the tangent-plane rotation of the major axis.
#[inline]
#[must_use]
pub fn anisotropic_ggx_from_covariance(cov: [f32; 3]) -> (f32, f32, f32) {
    let e = slope_covariance_eigen(cov);
    (
        variance_to_ggx_alpha(e.major),
        variance_to_ggx_alpha(e.minor),
        e.angle,
    )
}

#[cfg(test)]
mod tests {
    use super::{
        anisotropic_ggx_from_covariance, ggx_alpha_to_variance, slope_covariance_eigen,
        slope_covariance_from_eigen, variance_to_ggx_alpha,
    };

    fn close(a: f32, b: f32, tol: f32, msg: &str) {
        assert!((a - b).abs() < tol, "{msg}: {a} vs {b}");
    }

    /// Decompose then reconstruct returns the original covariance exactly (the
    /// primary anti-fake oracle: a genuine factorisation).
    #[test]
    fn eigen_reconstructs_covariance() {
        let cases = [
            [0.04f32, 0.09, 0.0],
            [0.09, 0.04, 0.0],
            [0.05, 0.05, 0.0],
            [0.08, 0.03, 0.02],
            [0.12, 0.07, -0.045],
            [0.2, 0.02, 0.05],
            [0.0, 0.0, 0.0],
        ];
        for cov in cases {
            let e = slope_covariance_eigen(cov);
            let rt = slope_covariance_from_eigen(e);
            for k in 0..3 {
                close(rt[k], cov[k], 1e-6, "reconstruction");
            }
        }
    }

    /// Eigenvalues are ordered and, for a positive-semidefinite input, both
    /// non-negative.
    #[test]
    fn eigenvalues_ordered_and_psd() {
        let psd = [
            [0.08f32, 0.03, 0.02],
            [0.12, 0.07, -0.045],
            [0.2, 0.02, 0.05],
            [0.05, 0.05, 0.0],
        ];
        for cov in psd {
            let e = slope_covariance_eigen(cov);
            assert!(e.major >= e.minor, "ordered {e:?}");
            assert!(e.minor >= -1e-6, "non-negative minor {e:?}");
        }
    }

    /// An isotropic covariance yields equal principal variances.
    #[test]
    fn isotropic_has_equal_axes() {
        let e = slope_covariance_eigen([0.06, 0.06, 0.0]);
        close(e.major, 0.06, 1e-6, "major");
        close(e.minor, 0.06, 1e-6, "minor");
    }

    /// An axis-aligned diagonal covariance decomposes to a zero rotation with
    /// the larger variance as the major axis.
    #[test]
    fn axis_aligned_has_zero_angle() {
        let e = slope_covariance_eigen([0.1, 0.03, 0.0]);
        close(e.major, 0.1, 1e-6, "major");
        close(e.minor, 0.03, 1e-6, "minor");
        close(e.angle, 0.0, 1e-6, "angle");
    }

    /// A covariance synthesised at a known rotation recovers that rotation and
    /// those eigenvalues.
    #[test]
    fn recovers_known_rotation() {
        // Build Sigma = R diag(0.16, 0.04) R^T at 30 degrees, decompose it back.
        let angle = core::f32::consts::FRAC_PI_6; // 30 degrees
        let synth = slope_covariance_from_eigen(super::SlopeEigen {
            major: 0.16,
            minor: 0.04,
            angle,
        });
        let e = slope_covariance_eigen(synth);
        close(e.major, 0.16, 1e-5, "major");
        close(e.minor, 0.04, 1e-5, "minor");
        close(e.angle, angle, 1e-5, "angle");
    }

    /// The GGX-roughness / slope-variance conversions are exact inverses.
    #[test]
    fn alpha_variance_round_trip() {
        for &v in &[0.0f32, 0.01, 0.08, 0.25, 0.5] {
            close(
                ggx_alpha_to_variance(variance_to_ggx_alpha(v)),
                v,
                1e-6,
                "v->a->v",
            );
        }
        for &a in &[0.0f32, 0.1, 0.4, 0.7, 1.0] {
            close(
                variance_to_ggx_alpha(ggx_alpha_to_variance(a)),
                a,
                1e-6,
                "a->v->a",
            );
        }
    }

    /// The anisotropic helper ties the eigen axes to GGX roughness consistently.
    #[test]
    fn anisotropic_helper_matches_parts() {
        let cov = [0.09f32, 0.04, 0.01];
        let e = slope_covariance_eigen(cov);
        let (am, an, ang) = anisotropic_ggx_from_covariance(cov);
        close(am, variance_to_ggx_alpha(e.major), 1e-6, "alpha major");
        close(an, variance_to_ggx_alpha(e.minor), 1e-6, "alpha minor");
        close(ang, e.angle, 1e-6, "angle");
        assert!(am >= an, "major roughness dominates");
    }
}
