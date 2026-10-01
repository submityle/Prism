//! Weighted-blended order-independent transparency (OIT) — CPU golden reference.
//!
//! This module is the backend-neutral numerical reference for McGuire &
//! Bavoil's *Weighted Blended Order-Independent Transparency* (2013).  The
//! method approximates correct back-to-front "over" compositing **without any
//! sorting** by replacing the ordered recurrence with two commutative
//! accumulators:
//!
//! * a weighted sum of premultiplied colour — `accum += (C·a)·w(z, a)` with the
//!   weighted alpha stored in `accum.w` (`accum.w += a·w(z, a)`), and
//! * a *revealage* product — `revealage *= (1 - a)` — tracking how much of the
//!   background still shows through the transparent stack.
//!
//! Because addition and multiplication are both order independent, the result
//! is identical for any permutation of the input fragments, which is the whole
//! point of OIT.  At resolve time the weighted average colour
//! `accum.rgb / max(accum.w, eps)` is scaled by the total coverage
//! `1 - revealage` and composited over the opaque background.
//!
//! The accuracy of the approximation is governed entirely by the depth weight
//! `w(z, a)`: fragments nearer the eye must dominate, so `w` decreases with
//! depth.  Several of the weighting functions proposed in the paper are
//! provided through [`WeightScheme`]; all are clamped to `[1e-2, 3e3]` to keep
//! the single-precision accumulators far from overflow/underflow.
//!
//! # Conventions
//! * `color` is **straight** (non-premultiplied) linear RGB and `alpha ∈ [0, 1]`
//!   is coverage; the accumulator premultiplies internally.
//! * `z` is a **positive, view-space** distance from the eye (larger = farther).
//!   [`WeightScheme::Equation7`] instead takes a normalised depth `d ∈ [0, 1]`.
//! * The two accumulators map onto the two render targets of the original
//!   technique: [`WeightedBlendTarget::accum`] (RGBA16F) and
//!   [`WeightedBlendTarget::revealage`] (R8/R16F).
//! * Every public function clamps its result finite and inside its valid range;
//!   degenerate inputs (`NaN`/`inf`, negative weights) fall back to safe values
//!   rather than poisoning the accumulators.
//!
//! # References
//! * McGuire & Bavoil 2013, *Weighted Blended Order-Independent Transparency*,
//!   Journal of Computer Graphics Techniques 2(2).
//! * Bavoil & Myers 2008, *Order Independent Transparency with Dual Depth
//!   Peeling* (the compositing identity this approximates).

use bevy_math::{Vec3, Vec4};

/// Lower clamp applied to every depth weight (paper's `1e-2`).
pub const MIN_WEIGHT: f32 = 1.0e-2;
/// Upper clamp applied to every depth weight (paper's `3e3`).
pub const MAX_WEIGHT: f32 = 3.0e3;
/// Default divide-by-zero guard for the resolve average.
pub const DEFAULT_RESOLVE_EPS: f32 = 1.0e-5;

/// Selectable depth-weighting functions `w(z, a)` from McGuire & Bavoil 2013.
///
/// Each variant trades off how aggressively near fragments dominate far ones.
/// [`WeightScheme::Equation8`] is the general-purpose default used by
/// [`WeightedBlendTarget`] and matches the engine's shader twin.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum WeightScheme {
    /// Equation 7: normalised-depth weight `a·max(1e-2, 3e3·(1 - d)^3)`,
    /// where the depth argument is `d ∈ [0, 1]` rather than a view distance.
    Equation7,
    /// Equation 8 (default): `a·clamp(10 / (1e-5 + (z/5)² + (z/200)⁶))`.
    #[default]
    Equation8,
    /// Equation 9: `a·clamp(0.03 / (1e-5 + (z/200)⁴))` — favours wide ranges.
    Equation9,
    /// Equation 10: `a·clamp(10 / (1e-5 + (z/10)³ + (z/200)⁶))`.
    Equation10,
}

/// Sanitises a scalar to a finite value, substituting `fallback` otherwise.
#[inline]
fn finite_or(x: f32, fallback: f32) -> f32 {
    if x.is_finite() { x } else { fallback }
}

/// Evaluates the depth weight `w(z, a)` for the given [`WeightScheme`].
///
/// `alpha` is clamped to `[0, 1]`; `depth` is treated as `d ∈ [0, 1]` for
/// [`WeightScheme::Equation7`] and as a non-negative view distance otherwise.
/// The returned weight is finite and lies in `[0, MAX_WEIGHT]` (`0` only when
/// `alpha == 0`).
#[inline]
pub fn weight(scheme: WeightScheme, depth: f32, alpha: f32) -> f32 {
    let a = finite_or(alpha, 0.0).clamp(0.0, 1.0);
    if a <= 0.0 {
        return 0.0;
    }
    let w = match scheme {
        WeightScheme::Equation7 => {
            let d = finite_or(depth, 1.0).clamp(0.0, 1.0);
            let one_minus = 1.0 - d;
            (MAX_WEIGHT * one_minus * one_minus * one_minus).max(MIN_WEIGHT)
        }
        WeightScheme::Equation8 => {
            let z = finite_or(depth, 0.0).abs();
            let a2 = z / 5.0;
            let b = z / 200.0;
            let b2 = b * b;
            let b6 = b2 * b2 * b2;
            let denom = 1.0e-5 + a2 * a2 + b6;
            (10.0 / denom).clamp(MIN_WEIGHT, MAX_WEIGHT)
        }
        WeightScheme::Equation9 => {
            let z = finite_or(depth, 0.0).abs();
            let c = z / 200.0;
            let c2 = c * c;
            let c4 = c2 * c2;
            let denom = 1.0e-5 + c4;
            (0.03 / denom).clamp(MIN_WEIGHT, MAX_WEIGHT)
        }
        WeightScheme::Equation10 => {
            let z = finite_or(depth, 0.0).abs();
            let a1 = z / 10.0;
            let a3 = a1 * a1 * a1;
            let b = z / 200.0;
            let b2 = b * b;
            let b6 = b2 * b2 * b2;
            let denom = 1.0e-5 + a3 + b6;
            (10.0 / denom).clamp(MIN_WEIGHT, MAX_WEIGHT)
        }
    };
    a * finite_or(w, MIN_WEIGHT)
}

/// Resolved weighted-blended output: average transparent colour plus coverage.
///
/// `color` is the weighted-average **straight** RGB of the transparent stack and
/// `coverage = 1 - revealage ∈ [0, 1]` is the fraction of the pixel occluded by
/// transparency.  Composite over an opaque background with
/// [`ResolvedTransparency::over_background`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResolvedTransparency {
    /// Weighted-average straight linear RGB of the transparent fragments.
    pub color: Vec3,
    /// Transparent coverage `1 - revealage ∈ [0, 1]`.
    pub coverage: f32,
}

impl ResolvedTransparency {
    /// Composites the resolved transparency over an opaque `background` colour.
    ///
    /// `out = color·coverage + background·(1 - coverage)`.
    #[inline]
    pub fn over_background(&self, background: Vec3) -> Vec3 {
        self.color * self.coverage + background * (1.0 - self.coverage)
    }
}

/// The two commutative accumulators of weighted-blended OIT.
///
/// `accum.xyz` holds `Σ (Cᵢ·aᵢ)·wᵢ`, `accum.w` holds `Σ aᵢ·wᵢ`, and
/// [`revealage`](Self::revealage) holds `Π (1 - aᵢ)`.  Build one with
/// [`WeightedBlendTarget::new`], feed fragments through
/// [`accumulate`](Self::accumulate), and read the result with
/// [`resolve`](Self::resolve).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeightedBlendTarget {
    /// `(Σ C·a·w, Σ a·w)` accumulator (RGBA render target).
    pub accum: Vec4,
    /// Revealage product `Π (1 - a)` (single-channel render target).
    pub revealage: f32,
}

impl Default for WeightedBlendTarget {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl WeightedBlendTarget {
    /// Returns an empty target (`accum = 0`, `revealage = 1`).
    #[inline]
    pub fn new() -> Self {
        Self {
            accum: Vec4::ZERO,
            revealage: 1.0,
        }
    }

    /// Adds one fragment weighted by a precomputed depth weight `w`.
    ///
    /// `color` is straight linear RGB, `alpha ∈ [0, 1]`.  The colour is
    /// premultiplied internally; `w` is clamped non-negative and finite.  This
    /// is the order-independent inner loop: `accum += (C·a·w, a·w)` and
    /// `revealage *= (1 - a)`.
    #[inline]
    pub fn accumulate_weighted(&mut self, color: Vec3, alpha: f32, w: f32) {
        let a = finite_or(alpha, 0.0).clamp(0.0, 1.0);
        let weight = finite_or(w, 0.0).max(0.0);
        let c = Vec3::new(
            finite_or(color.x, 0.0),
            finite_or(color.y, 0.0),
            finite_or(color.z, 0.0),
        );
        let premult = c * a;
        self.accum += Vec4::new(
            premult.x * weight,
            premult.y * weight,
            premult.z * weight,
            a * weight,
        );
        self.revealage *= 1.0 - a;
    }

    /// Adds one fragment, computing its depth weight from `scheme` and `depth`.
    ///
    /// Convenience wrapper over [`accumulate_weighted`](Self::accumulate_weighted)
    /// using [`weight`].
    #[inline]
    pub fn accumulate(&mut self, scheme: WeightScheme, color: Vec3, alpha: f32, depth: f32) {
        let w = weight(scheme, depth, alpha);
        self.accumulate_weighted(color, alpha, w);
    }

    /// Resolves the accumulators into an average colour plus coverage.
    ///
    /// `eps` guards the divide `accum.rgb / max(accum.w, eps)`.  When no
    /// fragment contributed (`accum.w <= 0`) the average colour is black and the
    /// coverage is `0`, so compositing yields the untouched background.
    #[inline]
    pub fn resolve(&self, eps: f32) -> ResolvedTransparency {
        let guard = finite_or(eps, DEFAULT_RESOLVE_EPS).max(f32::MIN_POSITIVE);
        let denom = self.accum.w.max(guard);
        let color = if self.accum.w > 0.0 {
            Vec3::new(self.accum.x, self.accum.y, self.accum.z) / denom
        } else {
            Vec3::ZERO
        };
        let revealage = finite_or(self.revealage, 1.0).clamp(0.0, 1.0);
        ResolvedTransparency {
            color: Vec3::new(
                finite_or(color.x, 0.0),
                finite_or(color.y, 0.0),
                finite_or(color.z, 0.0),
            ),
            coverage: (1.0 - revealage).clamp(0.0, 1.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-5;

    #[test]
    fn weight_is_zero_for_zero_alpha() {
        assert_eq!(weight(WeightScheme::Equation8, 10.0, 0.0), 0.0);
    }

    #[test]
    fn weight_decreases_with_depth() {
        let near = weight(WeightScheme::Equation8, 1.0, 1.0);
        let far = weight(WeightScheme::Equation8, 500.0, 1.0);
        assert!(near > far, "near={near} far={far}");
    }

    #[test]
    fn weight_is_clamped_in_range() {
        for &scheme in &[
            WeightScheme::Equation7,
            WeightScheme::Equation8,
            WeightScheme::Equation9,
            WeightScheme::Equation10,
        ] {
            for i in 0..64 {
                let z = i as f32 * 20.0;
                let w = weight(scheme, z, 1.0);
                assert!(w.is_finite(), "scheme={scheme:?} z={z} w={w}");
                assert!((0.0..=MAX_WEIGHT).contains(&w), "scheme={scheme:?} w={w}");
            }
        }
    }

    #[test]
    fn weight_rejects_nan() {
        let w = weight(WeightScheme::Equation8, f32::NAN, f32::NAN);
        assert!(w.is_finite());
    }

    #[test]
    fn empty_target_is_fully_transparent() {
        let t = WeightedBlendTarget::new();
        let r = t.resolve(EPS);
        assert_eq!(r.coverage, 0.0);
        let bg = Vec3::new(0.2, 0.4, 0.6);
        assert!((r.over_background(bg) - bg).length() < 1e-6);
    }

    #[test]
    fn single_opaque_fragment_covers_fully() {
        let mut t = WeightedBlendTarget::new();
        t.accumulate(WeightScheme::Equation8, Vec3::new(1.0, 0.0, 0.0), 1.0, 5.0);
        let r = t.resolve(EPS);
        assert!((r.coverage - 1.0).abs() < 1e-6, "coverage={}", r.coverage);
        let out = r.over_background(Vec3::ZERO);
        assert!((out - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-5, "out={out:?}");
    }

    #[test]
    fn accumulation_is_order_independent() {
        let frags = [
            (Vec3::new(1.0, 0.0, 0.0), 0.5_f32, 3.0_f32),
            (Vec3::new(0.0, 1.0, 0.0), 0.3, 10.0),
            (Vec3::new(0.0, 0.0, 1.0), 0.8, 1.5),
        ];
        let mut forward = WeightedBlendTarget::new();
        for &(c, a, z) in frags.iter() {
            forward.accumulate(WeightScheme::Equation8, c, a, z);
        }
        let mut reverse = WeightedBlendTarget::new();
        for &(c, a, z) in frags.iter().rev() {
            reverse.accumulate(WeightScheme::Equation8, c, a, z);
        }
        let rf = forward.resolve(EPS);
        let rr = reverse.resolve(EPS);
        assert!((rf.coverage - rr.coverage).abs() < 1e-6);
        assert!((rf.color - rr.color).length() < 1e-6);
    }

    #[test]
    fn revealage_matches_product_of_transmission() {
        let mut t = WeightedBlendTarget::new();
        let alphas = [0.5_f32, 0.25, 0.1];
        for &a in alphas.iter() {
            t.accumulate(WeightScheme::Equation8, Vec3::ONE, a, 10.0);
        }
        let expected_reveal = alphas.iter().fold(1.0_f32, |acc, &a| acc * (1.0 - a));
        let r = t.resolve(EPS);
        assert!(((1.0 - r.coverage) - expected_reveal).abs() < 1e-6);
    }

    #[test]
    fn resolve_never_produces_nan() {
        let mut t = WeightedBlendTarget::new();
        t.accumulate_weighted(Vec3::splat(f32::NAN), f32::NAN, f32::NAN);
        let r = t.resolve(0.0);
        assert!(r.color.is_finite());
        assert!(r.coverage.is_finite());
    }

    #[test]
    fn near_fragment_dominates_average() {
        // A bright near fragment and a dark far fragment of equal alpha: the
        // weighted average must lean toward the near colour.
        let mut t = WeightedBlendTarget::new();
        t.accumulate(WeightScheme::Equation8, Vec3::ONE, 0.5, 2.0);
        t.accumulate(WeightScheme::Equation8, Vec3::ZERO, 0.5, 300.0);
        let r = t.resolve(EPS);
        assert!(r.color.x > 0.5, "near colour should dominate: {}", r.color.x);
    }
}
