//! Moment-based order-independent transparency (OIT) — CPU golden reference.
//!
//! This module is the backend-neutral numerical reference for Münstermann,
//! Krüger, Bavoil & Wyman's *Moment-Based Order-Independent Transparency*
//! (2018).  Where weighted-blended OIT (see
//! [`crate::gi::oit::weighted_blended`]) stores only a single depth-weighted
//! average, moment-based OIT captures a compact statistical summary of the
//! **absorbance distribution along depth** and reconstructs, for any query
//! depth `z`, the transmittance of all closer fragments.
//!
//! Two passes are involved:
//!
//! 1. **Generation** — every transparent fragment contributes an absorbance
//!    `aᵢ = -ln(1 - αᵢ)` placed at warped depth `zᵢ`.  We accumulate the zeroth
//!    power moment `b₀ = Σ aᵢ` and the normalised higher power moments
//!    `bₖ = Σ aᵢ·zᵢᵏ` for `k = 1..=4` (the four-moment variant) or `k = 1..=6`
//!    (the six-moment variant for difficult overlap).  Accumulation is a plain
//!    sum, hence order independent.
//! 2. **Reconstruction** — given the power moments and a query depth, the
//!    *Hamburger* moment problem is solved in closed form (Peters & Klein 2015)
//!    to bound the fraction of total absorbance located in front of `z`.  The
//!    transmittance is then `exp(-b₀·fraction)`.
//!
//! Only the four-moment reconstruction is a true closed form (a Cholesky
//! factorisation of a 3×3 Hankel matrix followed by a quadratic solve); it is
//! the reference reconstruction here.  Six-moment generation is provided for
//! callers that want the extra fidelity, with [`PowerMoments6::truncate`]
//! exposing the four-moment subset consumed by the closed-form solver.
//!
//! # Conventions
//! * `alpha ∈ [0, 1)` is per-fragment coverage; `alpha = 1` is clamped to just
//!   below one so the absorbance `-ln(1 - α)` stays finite.
//! * Depths fed to generation and reconstruction must be **warped to `[-1, 1]`**
//!   consistently; [`warp_depth`] maps a view-space `[near, far]` range onto
//!   that canonical interval for numerical conditioning.
//! * A positive `moment_bias` blends the normalised moments toward the neutral
//!   vector `(0, 0.375, 0, 0.375)` (Peters' power-moment bias), trading a touch
//!   of contrast for freedom from ringing; defaults live in
//!   [`crate::gi::oit::OitParams`].
//! * Every reconstruction clamps its transmittance to `[0, 1]` and never
//!   returns `NaN`/`inf`, falling back to the unoccluded value `1` for
//!   degenerate moment sets.
//!
//! # References
//! * Münstermann, Krüger, Bavoil & Wyman 2018, *Moment-Based Order-Independent
//!   Transparency*, Proc. ACM i3D.
//! * Peters & Klein 2015, *Moment Shadow Mapping* (the four-moment Hamburger
//!   reconstruction reused here).

use bevy_math::{ops, Vec3, Vec4};

/// Neutral bias target for four power moments (Peters 2015).
const POWER_MOMENT_BIAS: Vec4 = Vec4::new(0.0, 0.375, 0.0, 0.375);
/// Minimum total absorbance below which the pixel is treated as empty.
const MIN_ABSORBANCE: f32 = 1.0e-4;
/// Alpha clamp so `-ln(1 - a)` stays finite for near-opaque fragments.
const MAX_ALPHA: f32 = 1.0 - 1.0e-4;

/// Sanitises a scalar to a finite value, substituting `fallback` otherwise.
#[inline]
fn finite_or(x: f32, fallback: f32) -> f32 {
    if x.is_finite() { x } else { fallback }
}

/// Converts a fragment `alpha ∈ [0, 1]` to optical absorbance `-ln(1 - a)`.
///
/// The alpha is clamped to `[0, MAX_ALPHA]` so a fully opaque fragment yields a
/// large but finite absorbance instead of `+inf`.
#[inline]
pub fn alpha_to_absorbance(alpha: f32) -> f32 {
    let a = finite_or(alpha, 0.0).clamp(0.0, MAX_ALPHA);
    let t = (1.0 - a).max(f32::MIN_POSITIVE);
    (-ops::ln(t)).max(0.0)
}

/// Maps a view-space depth in `[near, far]` onto the canonical `[-1, 1]`
/// interval used by moment generation and reconstruction.
///
/// Linearly rescales `z`; the range is degenerate-guarded so `near == far`
/// returns `0`.  Results are clamped to `[-1, 1]`.
#[inline]
pub fn warp_depth(z: f32, near: f32, far: f32) -> f32 {
    let n = finite_or(near, 0.0);
    let f = finite_or(far, 1.0);
    let span = f - n;
    if span.abs() <= f32::MIN_POSITIVE {
        return 0.0;
    }
    let t = (finite_or(z, n) - n) / span;
    (t * 2.0 - 1.0).clamp(-1.0, 1.0)
}

/// Accumulated four-power-moment summary of a pixel's absorbance distribution.
///
/// `b0` is the total absorbance `Σ aᵢ`; `b` holds the raw higher power moments
/// `(Σ aᵢzᵢ, Σ aᵢzᵢ², Σ aᵢzᵢ³, Σ aᵢzᵢ⁴)`.  Build with [`PowerMoments4::new`],
/// feed fragments through [`add`](Self::add), and reconstruct transmittance with
/// [`reconstruct_transmittance`](Self::reconstruct_transmittance).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct PowerMoments4 {
    /// Zeroth moment: total absorbance `Σ aᵢ`.
    pub b0: f32,
    /// Raw power moments `(Σ aᵢzᵢ, Σ aᵢzᵢ², Σ aᵢzᵢ³, Σ aᵢzᵢ⁴)`.
    pub b: Vec4,
}

impl PowerMoments4 {
    /// Returns an empty moment set (all zero).
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds one fragment at warped depth `z ∈ [-1, 1]` with coverage `alpha`.
    #[inline]
    pub fn add(&mut self, z: f32, alpha: f32) {
        let absorbance = alpha_to_absorbance(alpha);
        if absorbance <= 0.0 {
            return;
        }
        let zc = finite_or(z, 0.0).clamp(-1.0, 1.0);
        let z2 = zc * zc;
        let z3 = z2 * zc;
        let z4 = z2 * z2;
        self.b0 += absorbance;
        self.b += Vec4::new(absorbance * zc, absorbance * z2, absorbance * z3, absorbance * z4);
    }

    /// Total transmittance through the whole transparent stack: `exp(-b₀)`.
    #[inline]
    pub fn total_transmittance(&self) -> f32 {
        ops::exp(-self.b0.max(0.0)).clamp(0.0, 1.0)
    }

    /// Reconstructs the transmittance of fragments **in front of** `z`.
    ///
    /// `z` is a warped depth in `[-1, 1]`; `moment_bias ∈ [0, 1]` regularises the
    /// moments toward the neutral power-moment vector.  Returns a value in
    /// `[0, 1]` (`1` = nothing in front), finite for any input.
    #[inline]
    pub fn reconstruct_transmittance(&self, z: f32, moment_bias: f32) -> f32 {
        if self.b0 < MIN_ABSORBANCE {
            return 1.0;
        }
        let fraction = self.absorbance_fraction_in_front(z, moment_bias);
        let absorbance_front = (self.b0 * fraction).max(0.0);
        ops::exp(-absorbance_front).clamp(0.0, 1.0)
    }

    /// Solves the four-moment Hamburger problem for the fraction of total
    /// absorbance located at depths `< z`, in `[0, 1]`.
    #[inline]
    fn absorbance_fraction_in_front(&self, z: f32, moment_bias: f32) -> f32 {
        // Normalise then bias the moments (Peters & Klein 2015).
        let inv_b0 = 1.0 / self.b0;
        let m = self.b * inv_b0;
        let bias = finite_or(moment_bias, 0.0).clamp(0.0, 1.0);
        let b = m.lerp(POWER_MOMENT_BIAS, bias);

        let zf = finite_or(z, 0.0).clamp(-1.0, 1.0);

        // Cholesky factorisation of the 3×3 Hankel matrix (non-trivial entries).
        let l21d11 = b.x.mul_add(-b.y, b.z); // -b0*b1 + b2
        let d11 = b.x.mul_add(-b.x, b.y); // -b0*b0 + b1
        if d11.abs() <= f32::MIN_POSITIVE {
            return 0.0;
        }
        let inv_d11 = 1.0 / d11;
        let l21 = l21d11 * inv_d11;
        let sq_depth_var = b.y.mul_add(-b.y, b.w); // -b1*b1 + b3
        let d22 = l21d11.mul_add(-l21, sq_depth_var);
        if d22.abs() <= f32::MIN_POSITIVE {
            return 0.0;
        }
        let inv_d22 = 1.0 / d22;

        // Solve for the scaled inverse image of (1, z, z²)ᵀ.
        let mut c0 = 1.0_f32;
        let mut c1 = zf;
        let mut c2 = zf * zf;
        // Forward substitution L·c₁ = bz.
        c1 -= b.x;
        c2 -= b.y + l21 * c1;
        // Diagonal solve D·c₂ = c₁.
        c1 *= inv_d11;
        c2 *= inv_d22;
        // Backward substitution Lᵀ·c₃ = c₂.
        c1 -= l21 * c2;
        c0 -= b.x * c1 + b.y * c2;

        // Roots of c0 + c1·z + c2·z² give the two canonical support depths.
        if c2.abs() <= f32::MIN_POSITIVE {
            return 0.0;
        }
        let inv_c2 = 1.0 / c2;
        let p = c1 * inv_c2;
        let q = c0 * inv_c2;
        let disc = (p * p * 0.25 - q).max(0.0);
        let r = disc.sqrt();
        let z1 = -p * 0.5 - r;
        let z2 = -p * 0.5 + r;

        // Select weights for the three canonical cases (Peters & Klein 2015).
        let (s0, s1, s2, s3): (f32, f32, f32, f32) = if z2 < zf {
            (z1, zf, 1.0, 1.0)
        } else if z1 < zf {
            (zf, z1, 0.0, 1.0)
        } else {
            (0.0, 0.0, 0.0, 0.0)
        };
        let denom = (z2 - s1) * (zf - z1);
        if denom.abs() <= f32::MIN_POSITIVE {
            return s2.clamp(0.0, 1.0);
        }
        let quotient = (s0 * z2 - b.x * (s0 + z2) + b.y) / denom;
        let fraction = s2 + s3 * quotient;
        finite_or(fraction, 0.0).clamp(0.0, 1.0)
    }
}

/// Accumulates four power moments from an iterator of `(warped_depth, alpha)`.
///
/// Convenience constructor equivalent to repeated [`PowerMoments4::add`].
pub fn generate_moments<I>(fragments: I) -> PowerMoments4
where
    I: IntoIterator<Item = (f32, f32)>,
{
    let mut moments = PowerMoments4::new();
    for (z, alpha) in fragments {
        moments.add(z, alpha);
    }
    moments
}

/// Reconstructs the front transmittance at `z` from a [`PowerMoments4`] set.
///
/// Free-function form of [`PowerMoments4::reconstruct_transmittance`].
#[inline]
pub fn reconstruct_transmittance(moments: &PowerMoments4, z: f32, moment_bias: f32) -> f32 {
    moments.reconstruct_transmittance(z, moment_bias)
}

/// Accumulated six-power-moment summary for high-overlap scenes.
///
/// Stores the total absorbance and the six raw power moments
/// `Σ aᵢzᵢᵏ, k = 1..=6`.  Six moments resolve more depth structure than four;
/// [`truncate`](Self::truncate) exposes the four-moment subset consumed by the
/// closed-form [`PowerMoments4`] reconstruction.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct PowerMoments6 {
    /// Zeroth moment: total absorbance `Σ aᵢ`.
    pub b0: f32,
    /// Odd power moments `(Σ aᵢzᵢ, Σ aᵢzᵢ³, Σ aᵢzᵢ⁵)`.
    pub odd: Vec3,
    /// Even power moments `(Σ aᵢzᵢ², Σ aᵢzᵢ⁴, Σ aᵢzᵢ⁶)`.
    pub even: Vec3,
}

impl PowerMoments6 {
    /// Returns an empty moment set (all zero).
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds one fragment at warped depth `z ∈ [-1, 1]` with coverage `alpha`.
    #[inline]
    pub fn add(&mut self, z: f32, alpha: f32) {
        let absorbance = alpha_to_absorbance(alpha);
        if absorbance <= 0.0 {
            return;
        }
        let zc = finite_or(z, 0.0).clamp(-1.0, 1.0);
        let z2 = zc * zc;
        let z3 = z2 * zc;
        let z4 = z2 * z2;
        let z5 = z4 * zc;
        let z6 = z4 * z2;
        self.b0 += absorbance;
        self.odd += Vec3::new(absorbance * zc, absorbance * z3, absorbance * z5);
        self.even += Vec3::new(absorbance * z2, absorbance * z4, absorbance * z6);
    }

    /// Total transmittance through the whole transparent stack: `exp(-b₀)`.
    #[inline]
    pub fn total_transmittance(&self) -> f32 {
        ops::exp(-self.b0.max(0.0)).clamp(0.0, 1.0)
    }

    /// Returns the four-power-moment subset for closed-form reconstruction.
    #[inline]
    pub fn truncate(&self) -> PowerMoments4 {
        PowerMoments4 {
            b0: self.b0,
            b: Vec4::new(self.odd.x, self.even.x, self.odd.y, self.even.y),
        }
    }

    /// Reconstructs front transmittance via the four-moment subset.
    #[inline]
    pub fn reconstruct_transmittance(&self, z: f32, moment_bias: f32) -> f32 {
        self.truncate().reconstruct_transmittance(z, moment_bias)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absorbance_matches_definition() {
        let a = alpha_to_absorbance(0.5);
        assert!((a - (-ops::ln(0.5))).abs() < 1e-6);
    }

    #[test]
    fn opaque_alpha_stays_finite() {
        let a = alpha_to_absorbance(1.0);
        assert!(a.is_finite() && a > 0.0);
    }

    #[test]
    fn warp_maps_range_to_unit_interval() {
        assert!((warp_depth(0.0, 0.0, 10.0) + 1.0).abs() < 1e-6);
        assert!(warp_depth(10.0, 0.0, 10.0).abs() - 1.0 < 1e-6);
        assert!((warp_depth(5.0, 0.0, 10.0)).abs() < 1e-6);
        assert_eq!(warp_depth(5.0, 3.0, 3.0), 0.0);
    }

    #[test]
    fn empty_moments_are_transparent() {
        let m = PowerMoments4::new();
        assert_eq!(m.reconstruct_transmittance(0.0, 0.0), 1.0);
        assert!((m.total_transmittance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn transmittance_is_monotonically_decreasing_in_depth() {
        // Three fragments spread across depth; querying deeper must reveal more
        // occluders in front, so transmittance can only drop.
        let m = generate_moments([(-0.6, 0.4), (0.0, 0.5), (0.6, 0.3)]);
        let bias = 5.0e-3;
        let mut prev = m.reconstruct_transmittance(-1.0, bias);
        for i in 1..=20 {
            let z = -1.0 + (i as f32) * 0.1;
            let t = m.reconstruct_transmittance(z, bias);
            assert!(t.is_finite() && (0.0..=1.0).contains(&t), "z={z} t={t}");
            assert!(t <= prev + 1e-3, "not monotone at z={z}: {t} > {prev}");
            prev = t;
        }
    }

    #[test]
    fn front_transmittance_bounds_total() {
        let m = generate_moments([(-0.5, 0.6), (0.2, 0.4)]);
        let bias = 5.0e-3;
        let back = m.reconstruct_transmittance(1.0, bias);
        let total = m.total_transmittance();
        // The deepest query should approach the full-stack transmittance.
        assert!(back >= total - 1e-2, "back={back} total={total}");
    }

    #[test]
    fn generation_is_order_independent() {
        let a = generate_moments([(-0.5, 0.3), (0.1, 0.5), (0.7, 0.2)]);
        let b = generate_moments([(0.7, 0.2), (-0.5, 0.3), (0.1, 0.5)]);
        assert!((a.b0 - b.b0).abs() < 1e-6);
        assert!((a.b - b.b).length() < 1e-6);
    }

    #[test]
    fn reconstruction_never_nan_for_degenerate_moments() {
        let mut m = PowerMoments4::new();
        m.b0 = 2.0; // moments left zero → singular Hankel matrix.
        let t = m.reconstruct_transmittance(0.0, 0.0);
        assert!(t.is_finite() && (0.0..=1.0).contains(&t));
    }

    #[test]
    fn free_function_matches_method() {
        let m = generate_moments([(-0.3, 0.5), (0.4, 0.6)]);
        let a = reconstruct_transmittance(&m, 0.0, 1e-3);
        let b = m.reconstruct_transmittance(0.0, 1e-3);
        assert_eq!(a, b);
    }

    #[test]
    fn six_moment_truncation_preserves_four() {
        let mut m6 = PowerMoments6::new();
        let frags = [(-0.4, 0.5_f32), (0.3, 0.4), (0.8, 0.2)];
        let mut m4 = PowerMoments4::new();
        for &(z, a) in frags.iter() {
            m6.add(z, a);
            m4.add(z, a);
        }
        let t = m6.truncate();
        assert!((t.b0 - m4.b0).abs() < 1e-6);
        assert!((t.b - m4.b).length() < 1e-6);
        let bias = 5e-3;
        assert!(
            (m6.reconstruct_transmittance(0.0, bias) - m4.reconstruct_transmittance(0.0, bias))
                .abs()
                < 1e-6
        );
    }

    #[test]
    fn higher_alpha_lowers_transmittance() {
        let light = generate_moments([(-0.2, 0.2)]);
        let heavy = generate_moments([(-0.2, 0.8)]);
        let bias = 5e-3;
        let tl = light.reconstruct_transmittance(0.5, bias);
        let th = heavy.reconstruct_transmittance(0.5, bias);
        assert!(th < tl, "heavy={th} light={tl}");
    }
}
