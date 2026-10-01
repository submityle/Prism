//! Penner pre-integrated skin shading — the CPU golden reference for the
//! curvature-driven diffuse BRDF lookup baked by AAA real-time skin pipelines.
//!
//! Pre-integrated skin shading (Penner & Borshukov, *GPU Pro 2*, 2011) replaces
//! an expensive screen-space diffusion blur of the *lit* diffuse term with a
//! small two-dimensional lookup `D(N·L, curvature)`.  The insight is that, on a
//! locally spherical patch of skin of radius `r`, light scatters sideways
//! across the geometric terminator; the amount it bleeds depends only on the
//! surface *curvature* `κ = 1/r` and the local cosine `N·L`.  The lookup is
//! produced by convolving the clamped Lambert response around a great circle of
//! the sphere with the skin's 1-D radial diffusion profile:
//!
//! ```text
//!   D(θ, r) = ( ∫_{-π/2}^{π/2} saturate(cos(θ + a)) · R(2 r sin(a/2)) da )
//!             ---------------------------------------------------------------
//!             (            ∫_{-π/2}^{π/2}             R(2 r sin(a/2)) da )
//! ```
//!
//! where `θ = acos(N·L)` and `2 r sin(a/2)` is the chord length between the
//! shaded point and a neighbour at central angle `a` on the circle.  The
//! diffusion profile `R(r)` is the classic three-Gaussian skin fit (one narrow,
//! one medium, one wide lobe) with per-channel weights that give red light the
//! widest spatial tail — reproducing the characteristic red bleed just past the
//! terminator.  Each colour channel is integrated independently, so the result
//! is an RGB [`Vec3`].
//!
//! As curvature tends to zero (`r → ∞`) every neighbour but `a = 0` sits at an
//! effectively infinite chord distance, the profile collapses to a delta, and
//! `D(θ) → saturate(cos θ)` — i.e. a flat surface recovers ordinary Lambert.
//! Increasing curvature widens the soft wrap past the terminator.
//!
//! # Conventions
//! * `ndotl` is the surface cosine `N·L`, clamped to `[-1, 1]`.  Back-facing
//!   light (`N·L < 0`) can still return energy because of the wrap.
//! * `curvature` is `1/radius` in inverse world units (here millimetres);
//!   it is clamped non-negative and a near-zero curvature falls back to the
//!   flat Lambert response.  Radii are measured in the same units as the
//!   Gaussian lobe variances (mm²).
//! * The three-Gaussian profile weights sum to `1` per channel, so `R` is an
//!   energy-normalized radial density; the pre-integration is a *normalized*
//!   weighted average and therefore lies in `[0, 1]`.
//! * Spectral quantities are linear-RGB [`Vec3`]s matching the GPU twin.
//! * Every function is a deterministic pure function: no RNG, no I/O, no GPU,
//!   and no `unsafe`.  Transcendental maths goes through [`bevy_math::ops`].

use alloc::vec::Vec;
use bevy_math::{ops, Vec3};
use core::f32::consts::PI;

/// One Gaussian lobe of the skin diffusion profile.
#[derive(Clone, Copy, Debug)]
struct SkinLobe {
    /// Variance `v` of the lobe (mm²); the lobe is `exp(-r²/(2v)) / (2π v)`.
    variance: f32,
    /// Per-channel linear-RGB weight; the three lobe weights sum to `1`.
    weight: Vec3,
}

/// Three-Gaussian skin diffusion profile (narrow / medium / wide lobes).
///
/// The per-channel weights sum to `Vec3::ONE`, so the radial profile integrates
/// to one per channel over the plane.  The wide lobe is weighted most heavily
/// in red, giving the signature reddish subsurface bleed near the terminator.
const SKIN_LOBES: [SkinLobe; 3] = [
    SkinLobe {
        variance: 0.0516,
        weight: Vec3::new(0.299, 0.391, 0.474),
    },
    SkinLobe {
        variance: 0.2719,
        weight: Vec3::new(0.429, 0.457, 0.439),
    },
    SkinLobe {
        variance: 2.0062,
        weight: Vec3::new(0.272, 0.152, 0.087),
    },
];

/// Smallest curvature treated as curved; below this the surface is flat.
const MIN_CURVATURE: f32 = 1.0e-4;

/// Number of integration subintervals across `[-π/2, π/2]`.
const INTEGRATION_STEPS: usize = 160;

/// Clamp a possibly non-finite scalar into `[lo, hi]`, mapping `NaN` to `lo`.
#[inline]
fn clamp_finite(x: f32, lo: f32, hi: f32) -> f32 {
    if x.is_finite() {
        x.clamp(lo, hi)
    } else {
        lo
    }
}

/// 2-D normalized Gaussian lobe value `exp(-r²/(2v)) / (2π v)` at radius `r`.
#[inline]
fn gaussian_2d(variance: f32, r: f32) -> f32 {
    let v = variance.max(1.0e-8);
    let r = clamp_finite(r, 0.0, f32::MAX);
    let norm = 1.0 / (2.0 * PI * v);
    norm * ops::exp(-(r * r) / (2.0 * v))
}

/// Energy-normalized three-Gaussian skin diffusion profile `R(r)` (per channel).
///
/// Returns `Σ_i w_i · G(v_i, r)` over the three [`SKIN_LOBES`]; each channel is
/// non-negative and finite for every input (non-finite radii collapse to the
/// value at `r = 0`).
#[inline]
pub fn skin_profile(r: f32) -> Vec3 {
    let mut acc = Vec3::ZERO;
    for lobe in SKIN_LOBES {
        acc += lobe.weight * gaussian_2d(lobe.variance, r);
    }
    acc
}

/// Scalar (luminance-free, equal-weight) skin profile `⅓·Σ channels`.
///
/// Convenience reduction of [`skin_profile`] for callers that only need a
/// monochrome diffusion kernel; non-negative and finite.
#[inline]
pub fn skin_profile_scalar(r: f32) -> f32 {
    let p = skin_profile(r);
    (p.x + p.y + p.z) * (1.0 / 3.0)
}

/// Convert a curvature `κ` into a sphere radius `r = 1/κ` (world units).
///
/// `curvature` is clamped non-negative; a near-zero curvature maps to a very
/// large (effectively flat) radius so the lookup degrades gracefully to
/// Lambert.  The returned radius is strictly positive and finite.
#[inline]
pub fn radius_from_curvature(curvature: f32) -> f32 {
    let k = clamp_finite(curvature, 0.0, f32::MAX);
    if k > MIN_CURVATURE {
        1.0 / k
    } else {
        1.0 / MIN_CURVATURE
    }
}

/// Convert a sphere radius into curvature `κ = 1/r` (inverse world units).
///
/// `radius` is clamped strictly positive; the result is non-negative and
/// finite.
#[inline]
pub fn curvature_from_radius(radius: f32) -> f32 {
    let r = clamp_finite(radius, 1.0e-6, f32::MAX);
    1.0 / r
}

/// Penner pre-integrated diffuse lookup `D(N·L, curvature)` (per channel).
///
/// Numerically integrates the clamped Lambert response convolved with the
/// three-Gaussian skin profile around a great circle of the osculating sphere
/// (midpoint rule, [`INTEGRATION_STEPS`] subintervals across `[-π/2, π/2]`).
/// This is the direct golden reference the baked GPU LUT must reproduce.
///
/// `ndotl` is clamped to `[-1, 1]` and `curvature` non-negative.  The result is
/// the per-channel normalized average in `[0, 1]`; a degenerate (near-zero)
/// total weight falls back to `saturate(N·L)`.
pub fn preintegrated_diffuse(ndotl: f32, curvature: f32) -> Vec3 {
    let cos_theta = clamp_finite(ndotl, -1.0, 1.0);
    let theta = ops::acos(cos_theta);
    let radius = radius_from_curvature(curvature);

    let a0 = -PI * 0.5;
    let a1 = PI * 0.5;
    let n = INTEGRATION_STEPS;
    let da = (a1 - a0) / n as f32;

    let mut total_light = Vec3::ZERO;
    let mut total_weight = Vec3::ZERO;
    for i in 0..n {
        // Midpoint of the i-th subinterval.
        let a = a0 + (i as f32 + 0.5) * da;
        let diffuse = ops::cos(theta + a).max(0.0);
        // Chord length between the shaded point and the neighbour at angle `a`.
        let chord = (2.0 * radius * ops::sin(a * 0.5)).abs();
        let w = skin_profile(chord);
        total_light += w * diffuse;
        total_weight += w;
    }

    let fallback = cos_theta.max(0.0);
    channel_ratio(total_light, total_weight, fallback)
}

/// Scalar Penner pre-integrated diffuse lookup using [`skin_profile_scalar`].
///
/// Monochrome convenience variant of [`preintegrated_diffuse`]; returns a value
/// in `[0, 1]` and falls back to `saturate(N·L)` on a degenerate weight.
pub fn preintegrated_diffuse_scalar(ndotl: f32, curvature: f32) -> f32 {
    let cos_theta = clamp_finite(ndotl, -1.0, 1.0);
    let theta = ops::acos(cos_theta);
    let radius = radius_from_curvature(curvature);

    let a0 = -PI * 0.5;
    let n = INTEGRATION_STEPS;
    let da = PI / n as f32;

    let mut total_light = 0.0f32;
    let mut total_weight = 0.0f32;
    for i in 0..n {
        let a = a0 + (i as f32 + 0.5) * da;
        let diffuse = ops::cos(theta + a).max(0.0);
        let chord = (2.0 * radius * ops::sin(a * 0.5)).abs();
        let w = skin_profile_scalar(chord);
        total_light += w * diffuse;
        total_weight += w;
    }

    if total_weight > 1.0e-20 {
        (total_light / total_weight).clamp(0.0, 1.0)
    } else {
        cos_theta.max(0.0)
    }
}

/// Per-channel `light / weight` with a scalar fallback on degenerate weights.
#[inline]
fn channel_ratio(light: Vec3, weight: Vec3, fallback: f32) -> Vec3 {
    Vec3::new(
        safe_ratio(light.x, weight.x, fallback),
        safe_ratio(light.y, weight.y, fallback),
        safe_ratio(light.z, weight.z, fallback),
    )
}

/// `num / den` clamped to `[0, 1]`, returning `fallback` when `den` is tiny.
#[inline]
fn safe_ratio(num: f32, den: f32, fallback: f32) -> f32 {
    if den > 1.0e-20 {
        (num / den).clamp(0.0, 1.0)
    } else {
        fallback.clamp(0.0, 1.0)
    }
}

/// A baked 2-D pre-integrated skin lookup table.
///
/// Rows index curvature (`0 ..= max_curvature`) and columns index `N·L`
/// (`-1 ..= 1`), both sampled on a regular grid.  This mirrors the texture the
/// GPU twin samples at runtime; [`Lut::sample`] performs the matching bilinear
/// fetch so the CPU reference and GPU LUT agree.
#[derive(Clone, Debug)]
pub struct Lut {
    /// Number of `N·L` columns (`≥ 2`).
    pub width: usize,
    /// Number of curvature rows (`≥ 2`).
    pub height: usize,
    /// Maximum curvature stored in the last row.
    pub max_curvature: f32,
    /// Row-major `width · height` RGB samples.
    pub texels: Vec<Vec3>,
}

impl Lut {
    /// Bake a [`Lut`] by evaluating [`preintegrated_diffuse`] on a grid.
    ///
    /// `width` and `height` are clamped to at least `2`; `max_curvature` is
    /// clamped non-negative.  The returned table is finite everywhere.
    pub fn bake(width: usize, height: usize, max_curvature: f32) -> Self {
        let width = width.max(2);
        let height = height.max(2);
        let max_curvature = clamp_finite(max_curvature, 0.0, f32::MAX);
        let mut texels = Vec::with_capacity(width * height);
        for row in 0..height {
            let kv = row as f32 / (height - 1) as f32;
            let curvature = kv * max_curvature;
            for col in 0..width {
                let uv = col as f32 / (width - 1) as f32;
                let ndotl = uv * 2.0 - 1.0;
                texels.push(preintegrated_diffuse(ndotl, curvature));
            }
        }
        Self {
            width,
            height,
            max_curvature,
            texels,
        }
    }

    /// Bilinearly sample the baked table at `(ndotl, curvature)`.
    ///
    /// Inputs are clamped to the stored domain (`N·L ∈ [-1, 1]`, curvature
    /// `∈ [0, max_curvature]`); the result is finite and in `[0, 1]`.
    pub fn sample(&self, ndotl: f32, curvature: f32) -> Vec3 {
        let u = (clamp_finite(ndotl, -1.0, 1.0) * 0.5 + 0.5) * (self.width - 1) as f32;
        let denom = if self.max_curvature > 0.0 {
            self.max_curvature
        } else {
            1.0
        };
        let v = (clamp_finite(curvature, 0.0, self.max_curvature) / denom)
            * (self.height - 1) as f32;

        let x0 = (u.floor() as usize).min(self.width - 1);
        let y0 = (v.floor() as usize).min(self.height - 1);
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let fx = u - x0 as f32;
        let fy = v - y0 as f32;

        let c00 = self.texels[y0 * self.width + x0];
        let c10 = self.texels[y0 * self.width + x1];
        let c01 = self.texels[y1 * self.width + x0];
        let c11 = self.texels[y1 * self.width + x1];
        let top = c00.lerp(c10, fx);
        let bot = c01.lerp(c11, fx);
        top.lerp(bot, fy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn flat_surface_recovers_lambert() {
        // Near-zero curvature must collapse to saturate(N·L) on every channel.
        for &ndotl in &[-0.5f32, -0.1, 0.0, 0.25, 0.5, 0.8, 1.0] {
            let d = preintegrated_diffuse(ndotl, 0.0);
            let lambert = ndotl.max(0.0);
            assert!(approx(d.x, lambert, 2e-2), "x ndotl={ndotl} d={d:?}");
            assert!(approx(d.y, lambert, 2e-2), "y ndotl={ndotl} d={d:?}");
            assert!(approx(d.z, lambert, 2e-2), "z ndotl={ndotl} d={d:?}");
        }
    }

    #[test]
    fn curvature_softens_terminator() {
        // Just past the terminator a curved surface leaks more light than flat.
        let flat = preintegrated_diffuse(-0.05, 0.0);
        let curved = preintegrated_diffuse(-0.05, 2.0);
        assert!(curved.x >= flat.x, "flat={flat:?} curved={curved:?}");
        assert!(curved.x > 0.0, "curved should bleed light: {curved:?}");
    }

    #[test]
    fn red_bleeds_more_than_blue_near_terminator() {
        // The wide red lobe pushes red further past the terminator.
        let d = preintegrated_diffuse(-0.02, 4.0);
        assert!(d.x >= d.z, "expected red≥blue bleed, got {d:?}");
    }

    #[test]
    fn fully_lit_is_near_one() {
        for k in [0.0f32, 1.0, 5.0] {
            let d = preintegrated_diffuse(1.0, k);
            assert!(d.x > 0.6 && d.x <= 1.0, "k={k} d={d:?}");
        }
    }

    #[test]
    fn result_is_bounded_and_finite() {
        for &k in &[0.0f32, 0.5, 2.0, 10.0] {
            for i in 0..=40 {
                let ndotl = i as f32 / 20.0 - 1.0;
                let d = preintegrated_diffuse(ndotl, k);
                assert!(d.is_finite(), "non-finite at ndotl={ndotl} k={k}");
                for c in [d.x, d.y, d.z] {
                    assert!((0.0..=1.0).contains(&c), "out of range {c} k={k}");
                }
            }
        }
    }

    #[test]
    fn monotonic_in_ndotl_for_flat() {
        // On a flat surface the lookup is Lambert: non-decreasing in N·L.
        let mut prev = -1.0;
        for i in 0..=40 {
            let ndotl = i as f32 / 20.0 - 1.0;
            let d = preintegrated_diffuse_scalar(ndotl, 0.0);
            assert!(d >= prev - 1e-2, "not monotonic at ndotl={ndotl}");
            prev = d;
        }
    }

    #[test]
    fn profile_is_normalized_per_channel() {
        // Radial integral ∫ 2π r R(r) dr ≈ 1 per channel.
        let steps = 20_000;
        let r_max = 12.0f32;
        let dr = r_max / steps as f32;
        let mut acc = Vec3::ZERO;
        for i in 0..steps {
            let r = (i as f32 + 0.5) * dr;
            acc += skin_profile(r) * (2.0 * PI * r * dr);
        }
        assert!(approx(acc.x, 1.0, 1e-2), "red integral {}", acc.x);
        assert!(approx(acc.y, 1.0, 1e-2), "green integral {}", acc.y);
        assert!(approx(acc.z, 1.0, 1e-2), "blue integral {}", acc.z);
    }

    #[test]
    fn curvature_radius_roundtrip() {
        for r in [0.5f32, 1.0, 3.0, 10.0] {
            let k = curvature_from_radius(r);
            assert!(approx(radius_from_curvature(k), r, 1e-3));
        }
    }

    #[test]
    fn lut_matches_direct_integration() {
        let lut = Lut::bake(64, 48, 6.0);
        for &(ndotl, k) in &[(-0.3f32, 1.0f32), (0.2, 3.0), (0.7, 5.0)] {
            let direct = preintegrated_diffuse(ndotl, k);
            let baked = lut.sample(ndotl, k);
            assert!((direct - baked).length() < 3e-2, "direct={direct:?} baked={baked:?}");
        }
    }

    #[test]
    fn is_deterministic() {
        assert_eq!(preintegrated_diffuse(0.3, 2.0), preintegrated_diffuse(0.3, 2.0));
        assert_eq!(skin_profile(0.4), skin_profile(0.4));
    }

    #[test]
    fn no_nan_on_degenerate_inputs() {
        assert!(preintegrated_diffuse(f32::NAN, f32::NAN).is_finite());
        assert!(preintegrated_diffuse(f32::INFINITY, -1.0).is_finite());
        assert!(preintegrated_diffuse_scalar(f32::NAN, f32::NAN).is_finite());
        assert!(skin_profile(f32::NAN).is_finite());
        assert!(radius_from_curvature(f32::NAN).is_finite());
        assert!(curvature_from_radius(0.0).is_finite());
        let lut = Lut::bake(2, 2, 0.0);
        assert!(lut.sample(f32::NAN, f32::NAN).is_finite());
    }
}
