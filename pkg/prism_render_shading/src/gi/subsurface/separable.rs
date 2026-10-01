//! Jimenez separable screen-space subsurface scattering — the CPU golden
//! reference for the two-pass (horizontal + vertical) diffuse-blur kernel.
//!
//! Separable SSS (Jimenez et al., *Separable Subsurface Scattering*, 2015)
//! approximates the two-dimensional radial diffusion of light inside skin with
//! two cheap one-dimensional convolutions.  A true radial profile `R(r)` is not
//! generally separable, but a *sum of Gaussians* is: each isotropic Gaussian
//! factors exactly as `G₂(x, y) = G₁(x)·G₁(y)`, so running a normalized 1-D
//! Gaussian kernel horizontally and then vertically reproduces the 2-D blur for
//! every individual lobe (and approximates their weighted sum).
//!
//! This module builds that 1-D kernel.  Given a diffusion profile (the
//! three-Gaussian skin fit, or a single caller-supplied Gaussian), it produces
//! a symmetric set of **texel offsets** and **per-channel weights** normalized
//! to a partition of unity, so a flat (constant-radiance) region is preserved
//! exactly — the defining energy-conservation property of a diffusion blur.
//! The screen-space extent of the kernel is derived from the world-space SSS
//! width, the output resolution (dpi), the authored world scale, and — for a
//! perspective camera — the view-space depth.
//!
//! # Conventions
//! * Kernel taps are placed at integer texel offsets `-(n-1)/2 ..= (n-1)/2`;
//!   `num_taps` is clamped to an odd value `≥ 3` so the kernel is symmetric and
//!   has a well-defined centre.
//! * `weights` are per-channel linear-RGB [`Vec3`]s that **sum to one** on each
//!   channel; the convolution therefore conserves energy and leaves a constant
//!   signal unchanged.
//! * `sigma` / widths are standard deviations in texels (screen space) or world
//!   units, always clamped strictly positive so the Gaussian stays finite.
//! * Convolution uses clamp-to-edge addressing; separability is exact per lobe,
//!   so the 2-D kernel reconstructed from a 1-D kernel is its outer product.
//! * Every function is a deterministic pure function: no RNG, no I/O, no GPU,
//!   and no `unsafe`.  Transcendental maths goes through [`bevy_math::ops`].

use alloc::vec::Vec;
use bevy_math::{ops, Vec3};
use core::f32::consts::PI;

/// Smallest standard deviation, keeping the Gaussian finite and invertible.
const MIN_SIGMA: f32 = 1.0e-4;

/// One Gaussian lobe of the separable skin kernel (1-D variance + RGB weight).
#[derive(Clone, Copy, Debug)]
struct Lobe {
    variance: f32,
    weight: Vec3,
}

/// Three-Gaussian skin profile reused as a 1-D separable kernel basis.
///
/// Weights sum to `Vec3::ONE` per channel; the wide red lobe gives the broad
/// red screen-space bleed characteristic of skin.
const SKIN_LOBES: [Lobe; 3] = [
    Lobe {
        variance: 0.0516,
        weight: Vec3::new(0.299, 0.391, 0.474),
    },
    Lobe {
        variance: 0.2719,
        weight: Vec3::new(0.429, 0.457, 0.439),
    },
    Lobe {
        variance: 2.0062,
        weight: Vec3::new(0.272, 0.152, 0.087),
    },
];

/// Clamp a possibly non-finite scalar into `[lo, hi]`, mapping `NaN` to `lo`.
#[inline]
fn clamp_finite(x: f32, lo: f32, hi: f32) -> f32 {
    if x.is_finite() {
        x.clamp(lo, hi)
    } else {
        lo
    }
}

/// Normalized 1-D Gaussian density `exp(-x²/(2σ²)) / (σ·√(2π))`.
///
/// `sigma` is clamped strictly positive; the result is non-negative and finite,
/// and integrates to one over the real line.
#[inline]
pub fn gaussian_1d(sigma: f32, x: f32) -> f32 {
    let s = clamp_finite(sigma, MIN_SIGMA, f32::MAX);
    let x = clamp_finite(x, f32::MIN, f32::MAX);
    let norm = 1.0 / (s * (2.0 * PI).sqrt());
    norm * ops::exp(-(x * x) / (2.0 * s * s))
}

/// Three-Gaussian skin profile sampled as a 1-D slice at offset `x` (per channel).
///
/// `scale` multiplies every lobe's standard deviation, stretching the kernel to
/// a desired world/screen extent; it is clamped strictly positive.  The result
/// is non-negative and finite on every channel.
#[inline]
pub fn skin_profile_1d(x: f32, scale: f32) -> Vec3 {
    let scale = clamp_finite(scale, MIN_SIGMA, f32::MAX);
    let mut acc = Vec3::ZERO;
    for lobe in SKIN_LOBES {
        let sigma = lobe.variance.sqrt() * scale;
        acc += lobe.weight * gaussian_1d(sigma, x);
    }
    acc
}

/// Texels per world unit for a screen-aligned (orthographic) mapping.
///
/// `dpi` is output texels per inch; `world_scale` is inches per world unit, so
/// the product is texels per world unit.  Both are clamped non-negative and the
/// result is finite.
#[inline]
pub fn texels_per_world_unit(dpi: f32, world_scale: f32) -> f32 {
    let dpi = clamp_finite(dpi, 0.0, f32::MAX);
    let world_scale = clamp_finite(world_scale, 0.0, f32::MAX);
    dpi * world_scale
}

/// Texels per world unit under perspective, attenuated by view-space `depth`.
///
/// Divides the orthographic [`texels_per_world_unit`] by the depth so distant
/// surfaces blur over fewer texels (`scale ∝ 1/depth`).  `depth` is clamped
/// strictly positive; the result is finite and non-negative.
#[inline]
pub fn perspective_texel_scale(dpi: f32, world_scale: f32, depth: f32) -> f32 {
    let base = texels_per_world_unit(dpi, world_scale);
    let depth = clamp_finite(depth, MIN_SIGMA, f32::MAX);
    base / depth
}

/// A symmetric separable 1-D blur kernel (integer texel taps + RGB weights).
///
/// `offsets[k]` is the signed texel offset of tap `k` and `weights[k]` its
/// per-channel contribution.  The weights sum to one on each channel so the
/// kernel is energy-conserving; applying it horizontally then vertically yields
/// a separable approximation of the 2-D diffusion blur.
#[derive(Clone, Debug)]
pub struct SeparableKernel {
    /// Signed integer texel offsets, symmetric about zero.
    pub offsets: Vec<f32>,
    /// Per-channel weights, summing to `Vec3::ONE`.
    pub weights: Vec<Vec3>,
}

impl SeparableKernel {
    /// Build a single-Gaussian separable kernel of standard deviation `sigma`.
    ///
    /// `num_taps` is forced odd and `≥ 3`; `sigma` is in texels.  Weights are
    /// the sampled Gaussian renormalized to a partition of unity, so a
    /// single-lobe kernel reconstructs its 2-D counterpart *exactly*.
    pub fn gaussian(num_taps: usize, sigma: f32) -> Self {
        let n = odd_at_least_three(num_taps);
        let half = (n / 2) as i32;
        let sigma = clamp_finite(sigma, MIN_SIGMA, f32::MAX);
        let mut offsets = Vec::with_capacity(n);
        let mut weights = Vec::with_capacity(n);
        for k in 0..n {
            let tap = k as i32 - half;
            let x = tap as f32;
            offsets.push(x);
            weights.push(Vec3::splat(gaussian_1d(sigma, x)));
        }
        normalize_weights(&mut weights);
        Self { offsets, weights }
    }

    /// Build the three-Gaussian skin kernel spanning `radius_texels` texels.
    ///
    /// Taps are placed on `[-radius_texels, radius_texels]` and weighted by the
    /// skin profile; `scale` stretches the profile lobes to the chosen radius.
    /// `num_taps` is forced odd and `≥ 3`; weights are renormalized to a
    /// partition of unity on every channel.
    pub fn skin(num_taps: usize, radius_texels: f32, scale: f32) -> Self {
        let n = odd_at_least_three(num_taps);
        let half = (n / 2) as i32;
        let radius = clamp_finite(radius_texels, MIN_SIGMA, f32::MAX);
        let step = radius / half as f32;
        let mut offsets = Vec::with_capacity(n);
        let mut weights = Vec::with_capacity(n);
        for k in 0..n {
            let tap = k as i32 - half;
            let x = tap as f32 * step;
            offsets.push(x);
            weights.push(skin_profile_1d(x, scale));
        }
        normalize_weights(&mut weights);
        Self { offsets, weights }
    }

    /// Number of taps in the kernel.
    #[inline]
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Whether the kernel is empty (never true for a built kernel).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// Per-channel sum of the weights (should be `≈ Vec3::ONE`).
    pub fn weight_sum(&self) -> Vec3 {
        let mut acc = Vec3::ZERO;
        for &w in &self.weights {
            acc += w;
        }
        acc
    }

    /// Convolve a 1-D `signal` with the kernel at index `i` (clamp-to-edge).
    ///
    /// Returns `Σ_k weights[k] · signal[clamp(i + tap_k)]`.  An empty signal
    /// yields `Vec3::ZERO`.  Energy conservation guarantees a constant signal is
    /// reproduced exactly.
    pub fn convolve_at(&self, signal: &[Vec3], i: usize) -> Vec3 {
        if signal.is_empty() {
            return Vec3::ZERO;
        }
        let n = self.weights.len();
        let half = (n / 2) as i32;
        let last = signal.len() as i32 - 1;
        let mut acc = Vec3::ZERO;
        for k in 0..n {
            let tap = k as i32 - half;
            let idx = (i as i32 + tap).clamp(0, last) as usize;
            acc += self.weights[k] * signal[idx];
        }
        acc
    }

    /// Reconstruct the full 2-D kernel as the outer product of the 1-D weights.
    ///
    /// Returns an `n·n` row-major table where entry `(r, c)` is
    /// `weights[r] * weights[c]`; its total sums to `≈ Vec3::ONE` and it equals
    /// the true 2-D kernel exactly for a single Gaussian lobe.
    pub fn reconstruct_2d(&self) -> Vec<Vec3> {
        let n = self.weights.len();
        let mut out = Vec::with_capacity(n * n);
        for &wr in &self.weights {
            for &wc in &self.weights {
                out.push(wr * wc);
            }
        }
        out
    }
}

/// Force `n` to an odd value of at least three.
#[inline]
fn odd_at_least_three(n: usize) -> usize {
    let n = n.max(3);
    if n % 2 == 0 {
        n + 1
    } else {
        n
    }
}

/// Renormalize per-channel weights so each channel sums to one.
fn normalize_weights(weights: &mut [Vec3]) {
    let mut sum = Vec3::ZERO;
    for &w in weights.iter() {
        sum += w;
    }
    let inv = Vec3::new(
        safe_inv(sum.x),
        safe_inv(sum.y),
        safe_inv(sum.z),
    );
    for w in weights.iter_mut() {
        *w *= inv;
    }
}

/// `1/x` guarded against a zero or non-finite denominator (returns `0`).
#[inline]
fn safe_inv(x: f32) -> f32 {
    if x.is_finite() && x.abs() > 1.0e-20 {
        1.0 / x
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn gaussian_integrates_to_one() {
        let steps = 20_000;
        let x_max = 8.0f32;
        let dx = 2.0 * x_max / steps as f32;
        let mut acc = 0.0f64;
        for i in 0..steps {
            let x = -x_max + (i as f32 + 0.5) * dx;
            acc += gaussian_1d(1.3, x) as f64 * dx as f64;
        }
        assert!(approx(acc as f32, 1.0, 1e-3), "integral={acc}");
    }

    #[test]
    fn gaussian_weights_sum_to_one() {
        for sigma in [0.5f32, 1.0, 2.5, 5.0] {
            let k = SeparableKernel::gaussian(15, sigma);
            let s = k.weight_sum();
            assert!(approx(s.x, 1.0, 1e-5), "sigma={sigma} sum={s:?}");
            assert!(approx(s.y, 1.0, 1e-5));
            assert!(approx(s.z, 1.0, 1e-5));
        }
    }

    #[test]
    fn skin_weights_sum_to_one() {
        let k = SeparableKernel::skin(17, 6.0, 2.0);
        let s = k.weight_sum();
        assert!(approx(s.x, 1.0, 1e-5), "sum={s:?}");
        assert!(approx(s.y, 1.0, 1e-5));
        assert!(approx(s.z, 1.0, 1e-5));
    }

    #[test]
    fn constant_signal_is_preserved() {
        // Energy conservation: a flat region is returned unchanged.
        let k = SeparableKernel::skin(13, 5.0, 1.5);
        let signal = [Vec3::new(0.4, 0.6, 0.8); 32];
        for i in 0..signal.len() {
            let out = k.convolve_at(&signal, i);
            assert!((out - signal[0]).length() < 1e-5, "i={i} out={out:?}");
        }
    }

    #[test]
    fn delta_response_equals_weights() {
        // Convolving a centred delta reproduces the kernel weights.
        let k = SeparableKernel::gaussian(9, 1.2);
        let n = 41usize;
        let mut signal = alloc::vec![Vec3::ZERO; n];
        let center = n / 2;
        signal[center] = Vec3::ONE;
        let half = k.len() / 2;
        for (k_idx, &w) in k.weights.iter().enumerate() {
            let tap = k_idx as i32 - half as i32;
            let idx = (center as i32 - tap) as usize;
            let out = k.convolve_at(&signal, idx);
            assert!((out - w).length() < 1e-6, "tap={tap} out={out:?} w={w:?}");
        }
    }

    #[test]
    fn single_gaussian_is_exactly_separable() {
        // For one lobe the 2-D outer product equals the true 2-D Gaussian.
        let sigma = 1.7f32;
        let k = SeparableKernel::gaussian(21, sigma);
        let kernel2d = k.reconstruct_2d();
        let n = k.len();
        // Reference 2-D Gaussian, renormalized the same way.
        let mut reference = alloc::vec![0.0f32; n * n];
        let half = (n / 2) as i32;
        let mut sum = 0.0f32;
        for r in 0..n {
            for c in 0..n {
                let y = (r as i32 - half) as f32;
                let x = (c as i32 - half) as f32;
                let g = ops::exp(-(x * x + y * y) / (2.0 * sigma * sigma));
                reference[r * n + c] = g;
                sum += g;
            }
        }
        for v in reference.iter_mut() {
            *v /= sum;
        }
        for (sep, &reff) in kernel2d.iter().zip(reference.iter()) {
            assert!(approx(sep.x, reff, 1e-5), "sep={} ref={}", sep.x, reff);
        }
    }

    #[test]
    fn reconstructed_2d_sums_to_one() {
        let k = SeparableKernel::skin(15, 5.0, 2.0);
        let kernel2d = k.reconstruct_2d();
        let mut sum = Vec3::ZERO;
        for &w in &kernel2d {
            sum += w;
        }
        assert!(approx(sum.x, 1.0, 1e-5), "sum={sum:?}");
        assert!(approx(sum.y, 1.0, 1e-5));
        assert!(approx(sum.z, 1.0, 1e-5));
    }

    #[test]
    fn two_pass_matches_outer_product() {
        // Horizontal then vertical blur of a 2-D delta equals the outer product.
        let k = SeparableKernel::gaussian(7, 1.1);
        let n = 21usize;
        let center = n / 2;
        // Row-major image.
        let mut img = alloc::vec![Vec3::ZERO; n * n];
        img[center * n + center] = Vec3::ONE;
        // Horizontal pass.
        let mut tmp = alloc::vec![Vec3::ZERO; n * n];
        for r in 0..n {
            let row = &img[r * n..r * n + n];
            for c in 0..n {
                tmp[r * n + c] = k.convolve_at(row, c);
            }
        }
        // Vertical pass.
        let mut out = alloc::vec![Vec3::ZERO; n * n];
        let mut col = alloc::vec![Vec3::ZERO; n];
        for c in 0..n {
            for r in 0..n {
                col[r] = tmp[r * n + c];
            }
            for r in 0..n {
                out[r * n + c] = k.convolve_at(&col, r);
            }
        }
        let outer = k.reconstruct_2d();
        let kn = k.len();
        let khalf = kn / 2;
        for dr in 0..kn {
            for dc in 0..kn {
                let r = center + dr - khalf;
                let c = center + dc - khalf;
                let got = out[r * n + c];
                let want = outer[dr * kn + dc];
                assert!((got - want).length() < 1e-6, "got={got:?} want={want:?}");
            }
        }
    }

    #[test]
    fn perspective_scale_falls_with_depth() {
        let near = perspective_texel_scale(96.0, 0.5, 1.0);
        let far = perspective_texel_scale(96.0, 0.5, 4.0);
        assert!(near > far, "near={near} far={far}");
        assert!(approx(near / far, 4.0, 1e-4));
    }

    #[test]
    fn is_deterministic() {
        let a = SeparableKernel::skin(11, 4.0, 1.5);
        let b = SeparableKernel::skin(11, 4.0, 1.5);
        assert_eq!(a.weights, b.weights);
        assert_eq!(a.offsets, b.offsets);
    }

    #[test]
    fn no_nan_on_degenerate_inputs() {
        assert!(gaussian_1d(f32::NAN, f32::NAN).is_finite());
        assert!(gaussian_1d(0.0, 1.0).is_finite());
        assert!(skin_profile_1d(f32::NAN, f32::NAN).is_finite());
        assert!(texels_per_world_unit(f32::NAN, f32::NAN).is_finite());
        assert!(perspective_texel_scale(96.0, 0.5, 0.0).is_finite());
        let k = SeparableKernel::skin(0, f32::NAN, f32::NAN);
        assert!(k.len() >= 3);
        assert!(k.weight_sum().is_finite());
        assert!(k.convolve_at(&[], 0).is_finite());
    }
}
