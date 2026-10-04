//! Power-moment reconstruction for moment-based `OIT`.
//!
//! Moment-based `OIT` (`MBOIT`, Münstermann et al. 2018) summarises every
//! transparent fragment that covers a pixel into a fixed, bounded-size vector of
//! statistical *power moments* of the fragment-absorbance measure, then
//! reconstructs the transmittance at an arbitrary query depth from those moments
//! alone. Because the per-pixel storage is constant regardless of overdraw, it
//! renders order-independent transparency in two lightweight passes without the
//! unbounded per-pixel lists an exact `A-buffer` needs.
//!
//! The reconstruction is the classical *moment problem*: given the first few
//! power moments of a non-negative measure, bound the fraction of its mass that
//! lies in front of a query depth. We provide the provably-correct
//! Chebyshev–Cantelli two-moment bound (identical to variance shadow mapping)
//! and the tighter four-moment canonical bound of Peters & Klein 2015, obtained
//! by solving the `Hankel` moment system and locating the support points of the
//! canonical representing measure. Neither path uses any floating-point
//! transcendental intrinsic: the required `ln`/`exp` come from the sibling
//! `math` module and the `Hankel` solve from its `Cholesky` routine.

use super::math::{exp, ln, quadratic_roots, solve_spd3};

/// Smallest variance / pivot we treat as non-degenerate, to keep the
/// reconstruction finite when every fragment sits at (almost) one depth.
const EPS: f32 = 1.0e-6;

/// Regularisation weight blended toward the uniform-measure moments so the
/// `Hankel` matrix stays positive definite for near-degenerate inputs. Small
/// enough to leave well-conditioned scenes essentially unbiased.
const BIAS: f32 = 3.0e-4;

/// Accumulated power moments of a pixel's transparent-fragment absorbance.
///
/// `b0` is the total optical absorbance `sum(a_i)`; `b1..=b4` are the
/// absorbance-weighted sums of the first four powers of the *warped* view
/// depth. View depths are affinely warped from `[near, far]` into `[-1, 1]`
/// before the powers are taken, which keeps the higher moments numerically
/// balanced and matches the canonical-measure domain assumed by the four-moment
/// reconstruction.
#[derive(Clone, Copy, Debug)]
pub struct PowerMoments {
    b0: f32,
    b1: f32,
    b2: f32,
    b3: f32,
    b4: f32,
    near: f32,
    inv_span: f32,
}

impl PowerMoments {
    /// Creates an empty accumulator spanning the view-depth range `[near, far]`.
    ///
    /// `far` must be strictly greater than `near`; a degenerate range collapses
    /// to a unit span so warping stays finite.
    #[must_use]
    pub fn new(near: f32, far: f32) -> Self {
        let span = far - near;
        let inv_span = if span.abs() <= EPS { 1.0 } else { 1.0 / span };
        Self {
            b0: 0.0,
            b1: 0.0,
            b2: 0.0,
            b3: 0.0,
            b4: 0.0,
            near,
            inv_span,
        }
    }

    /// Warps a raw view depth into the canonical `[-1, 1]` domain.
    #[must_use]
    fn warp(&self, view_depth: f32) -> f32 {
        let t = (view_depth - self.near) * self.inv_span; // [0, 1] for in-range
        (2.0 * t - 1.0).clamp(-1.0, 1.0)
    }

    /// Accumulates one transparent fragment at `view_depth` with opacity
    /// `alpha`.
    ///
    /// The fragment contributes optical absorbance `a = -ln(1 - alpha)` (opacity
    /// is clamped below `1` so a fully opaque fragment stays finite), weighting
    /// the depth-power moments. Non-positive opacity is ignored.
    pub fn add_fragment(&mut self, view_depth: f32, alpha: f32) {
        let alpha = alpha.clamp(0.0, 0.999);
        if alpha <= 0.0 {
            return;
        }
        let a = -ln(1.0 - alpha);
        let w = self.warp(view_depth);
        let w2 = w * w;
        self.b0 += a;
        self.b1 += a * w;
        self.b2 += a * w2;
        self.b3 += a * w2 * w;
        self.b4 += a * w2 * w2;
    }

    /// Total accumulated absorbance (zero when the pixel is untouched).
    #[must_use]
    pub fn total_absorbance(&self) -> f32 {
        self.b0
    }

    /// Normalised first four moments of the probability measure, or `None` when
    /// no absorbance was accumulated.
    #[must_use]
    fn normalized(&self) -> Option<[f32; 4]> {
        if self.b0 <= EPS {
            return None;
        }
        let inv = 1.0 / self.b0;
        // Blend toward the uniform-on-[-1,1] moments (0, 1/3, 0, 1/5) so the
        // Hankel matrix is strictly positive definite even for a delta measure.
        let m = [self.b1 * inv, self.b2 * inv, self.b3 * inv, self.b4 * inv];
        let r = [0.0, 1.0 / 3.0, 0.0, 1.0 / 5.0];
        Some([
            (1.0 - BIAS) * m[0] + BIAS * r[0],
            (1.0 - BIAS) * m[1] + BIAS * r[1],
            (1.0 - BIAS) * m[2] + BIAS * r[2],
            (1.0 - BIAS) * m[3] + BIAS * r[3],
        ])
    }

    /// Fraction of absorbance in front of `view_depth` from two power moments.
    ///
    /// This is the one-sided Chebyshev–Cantelli lower bound on the measure's
    /// cumulative distribution: with mean `mu` and variance `sigma2`, no more
    /// than `sigma2 / (sigma2 + t^2)` of the mass can sit at or beyond
    /// `mu + t`, so at least `t^2 / (sigma2 + t^2)` lies in front. It is the
    /// exact bound realised by a two-point measure and coincides with variance
    /// shadow mapping.
    #[must_use]
    fn cdf2(&self, view_depth: f32) -> f32 {
        let Some(m) = self.normalized() else {
            return 0.0;
        };
        let mu = m[0];
        let sigma2 = (m[1] - mu * mu).max(EPS);
        let z = self.warp(view_depth);
        if z <= mu {
            0.0
        } else {
            let t = z - mu;
            (t * t / (sigma2 + t * t)).clamp(0.0, 1.0)
        }
    }

    /// Fraction of absorbance in front of `view_depth` from four power moments.
    ///
    /// Solves the `Hankel` system `B c = (1, z, z^2)` where `B` is the symmetric
    /// positive-definite moment matrix, so the quadratic `c2 x^2 + c1 x + c0`
    /// has as roots the two support depths of the canonical representing
    /// measure. The query depth's position relative to those roots yields the
    /// tightest cumulative-mass bound consistent with all four moments (Peters &
    /// Klein 2015). Falls back to the two-moment bound if the system is
    /// degenerate.
    #[must_use]
    fn cdf4(&self, view_depth: f32) -> f32 {
        let Some(m) = self.normalized() else {
            return 0.0;
        };
        let z = self.warp(view_depth);
        // Hankel moment matrix (lower triangle), rows/cols indexed by power 0..2.
        // [ 1   m1  m2 ]
        // [ m1  m2  m3 ]
        // [ m2  m3  m4 ]
        let a = [1.0, m[0], m[1], m[1], m[2], m[3]];
        let rhs = [1.0, z, z * z];
        let Some(c) = solve_spd3(a, rhs) else {
            return self.cdf2(view_depth);
        };
        // Roots of c2 x^2 + c1 x + c0 = 0 are the canonical support points.
        let Some((z2, z3)) = quadratic_roots(c[2], c[1], c[0]) else {
            return self.cdf2(view_depth);
        };
        let m1 = m[0];
        let m2 = m[1];
        if z <= z2 {
            0.0
        } else if z <= z3 {
            let num = z * z3 - m1 * (z + z3) + m2;
            let den = (z3 - z2) * (z - z2);
            if den.abs() <= EPS {
                self.cdf2(view_depth)
            } else {
                (num / den).clamp(0.0, 1.0)
            }
        } else {
            let num = z2 * z3 - m1 * (z2 + z3) + m2;
            let den = (z - z2) * (z - z3);
            if den.abs() <= EPS {
                self.cdf2(view_depth)
            } else {
                (1.0 - num / den).clamp(0.0, 1.0)
            }
        }
    }

    /// Reconstructed transmittance in front of `view_depth` using two moments.
    ///
    /// `transmittance = exp(-b0 * G2(view_depth))`, i.e. Beer–Lambert applied to
    /// the absorbance estimated to lie in front of the query depth. Always in
    /// `(0, 1]`.
    #[must_use]
    pub fn transmittance2(&self, view_depth: f32) -> f32 {
        exp(-self.b0 * self.cdf2(view_depth)).clamp(0.0, 1.0)
    }

    /// Reconstructed transmittance in front of `view_depth` using four moments.
    ///
    /// Same Beer–Lambert mapping as [`Self::transmittance2`] but with the
    /// tighter four-moment cumulative-mass estimate, so it tracks the exact
    /// depth-sorted result far more closely for multi-layer pixels.
    #[must_use]
    pub fn transmittance4(&self, view_depth: f32) -> f32 {
        exp(-self.b0 * self.cdf4(view_depth)).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn empty_pixel_is_fully_transparent() {
        let m = PowerMoments::new(0.0, 10.0);
        assert_eq!(m.total_absorbance(), 0.0);
        assert_eq!(m.transmittance2(5.0), 1.0);
        assert_eq!(m.transmittance4(5.0), 1.0);
    }

    #[test]
    fn transmittance_stays_in_unit_interval() {
        let mut m = PowerMoments::new(0.0, 1.0);
        for (d, a) in [(0.1, 0.4), (0.3, 0.6), (0.55, 0.3), (0.8, 0.9)] {
            m.add_fragment(d, a);
        }
        for i in 0..=20 {
            let z = i as f32 / 20.0;
            let t2 = m.transmittance2(z);
            let t4 = m.transmittance4(z);
            assert!((0.0..=1.0).contains(&t2), "t2={t2}");
            assert!((0.0..=1.0).contains(&t4), "t4={t4}");
        }
    }

    #[test]
    fn transmittance_is_monotone_in_depth() {
        let mut m = PowerMoments::new(0.0, 1.0);
        for (d, a) in [(0.2, 0.5), (0.4, 0.5), (0.6, 0.5), (0.85, 0.5)] {
            m.add_fragment(d, a);
        }
        let mut prev2 = f32::INFINITY;
        let mut prev4 = f32::INFINITY;
        for i in 0..=40 {
            let z = i as f32 / 40.0;
            let t2 = m.transmittance2(z);
            let t4 = m.transmittance4(z);
            assert!(t2 <= prev2 + 1e-4, "t2 not monotone: {t2} > {prev2}");
            assert!(t4 <= prev4 + 1e-4, "t4 not monotone: {t4} > {prev4}");
            prev2 = t2;
            prev4 = t4;
        }
    }

    #[test]
    fn single_fragment_brackets_its_depth() {
        // One fragment at d=0.5, alpha=0.75 -> transmittance behind it ~ 0.25.
        let mut m = PowerMoments::new(0.0, 1.0);
        m.add_fragment(0.5, 0.75);
        // Well in front: almost no absorbance -> transmittance ~ 1.
        assert!(m.transmittance4(0.05) > 0.9, "front={}", m.transmittance4(0.05));
        // Well behind: full absorbance -> transmittance ~ (1-alpha)=0.25.
        let behind = m.transmittance4(0.95);
        assert!(approx(behind, 0.25, 0.1), "behind={behind}");
    }

    /// Exact depth-sorted transmittance in front of `z` (the oracle).
    fn exact_transmittance(frags: &[(f32, f32)], z: f32) -> f32 {
        let mut t = 1.0_f32;
        for &(d, a) in frags {
            if d < z {
                t *= 1.0 - a;
            }
        }
        t
    }

    #[test]
    fn four_moments_beat_two_moments_against_exact_oracle() {
        // A representative multi-layer pixel.
        let frags: Vec<(f32, f32)> = vec![
            (0.10, 0.30),
            (0.25, 0.50),
            (0.45, 0.40),
            (0.65, 0.60),
            (0.90, 0.35),
        ];
        let mut m = PowerMoments::new(0.0, 1.0);
        for &(d, a) in &frags {
            m.add_fragment(d, a);
        }
        let mut err2 = 0.0_f32;
        let mut err4 = 0.0_f32;
        let mut n = 0.0_f32;
        for i in 1..20 {
            let z = i as f32 / 20.0;
            let exact = exact_transmittance(&frags, z);
            err2 += (m.transmittance2(z) - exact).abs();
            err4 += (m.transmittance4(z) - exact).abs();
            n += 1.0;
        }
        err2 /= n;
        err4 /= n;
        assert!(
            err4 <= err2 + 1e-4,
            "four-moment error {err4} should not exceed two-moment error {err2}"
        );
        // Moment reconstruction smooths depth discontinuities, so the raw
        // per-depth transmittance differs from the sharp step oracle; the
        // meaningful metric is the composited result (see the mod-level
        // `four_moment_tracks_exact_sorted_resolve` test, which stays < 0.1).
        assert!(err4 < 0.25, "four-moment mean error too large: {err4}");
    }
}
