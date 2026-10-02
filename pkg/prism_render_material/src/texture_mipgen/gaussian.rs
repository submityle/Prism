//! Separable Gaussian mip reduction for decoded `RGBA8`.
//!
//! The box ([`box_filter`](super::box_filter)) reducer is a hard 2-tap average
//! (boxy, aliased), while the Lanczos ([`windowed`](super::windowed)) and
//! Kaiser ([`kaiser`](super::kaiser)) reducers are windowed-sinc kernels whose
//! negative side-lobes **ring** at steep edges. For prefilter targets where
//! ringing is unacceptable -- roughness/gloss maps feeding a specular BRDF,
//! height fields, SDF/coverage data, bloom-style pyramids -- AAA texture tools
//! expose a strictly non-negative Gaussian as the "soft, no-overshoot" option.
//! This module adds a separable Gaussian 2x reduction on the shared
//! gamma-correct resample core in [`resample_core`](super::resample_core); only
//! the 1-D Gaussian kernel is declared here. Everything is deterministic
//! analytic arithmetic -- a CPU golden for a GPU compute down-sampler, no AI/ML.
//!
//! # Conventions
//! * Reduction factor is exactly 2 per axis; a dimension of `1` is carried
//!   through unchanged (GL `max(1, dim >> 1)` rule).
//! * The (unnormalised) kernel is `g(x) = exp(-x^2 / (2 sigma^2))` on
//!   `|x| <= radius` and `0` outside; [`resample_core`](super::resample_core)
//!   divides by the tap sum so a constant is preserved exactly.
//! * `g(x) >= 0` for all `x`, so the reduction is a convex combination of the
//!   source texels: the result can never over- or under-shoot the local texel
//!   range (no ringing), unlike the sinc-based reducers.
//! * Colour is filtered in scene-linear light exactly once; see
//!   [`resample_core`](super::resample_core) for the working-buffer policy.
//!
//! # References
//! * `DirectXTex` `TEX_FILTER_FLAGS` Gaussian / `NVIDIA` Texture Tools Gaussian
//!   mip generation.
//! * Gaussian blur as a non-negative, zero-overshoot low-pass prefilter.

use alloc::vec::Vec;

use super::box_filter::{ColorSpace, Rgba8Image};
use super::resample_core::downsample_2x;
use bevy_math::ops;

/// A separable Gaussian reduction filter.
///
/// `radius` is the half-support in output-sample units (taps beyond it are
/// dropped); `sigma` is the Gaussian standard deviation in the same units.
/// A larger `sigma` (relative to `radius`) is softer; a smaller one is sharper
/// but, if `radius` is not widened to match, truncates more of the tail.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaussianFilter {
    /// Half-support of the kernel, in output-sample units (`> 0`).
    pub radius: f32,
    /// Gaussian standard deviation, in output-sample units (`> 0`).
    pub sigma: f32,
}

impl GaussianFilter {
    /// Texture-tool style default: 3-tap half-support with `sigma = 1`, so the
    /// support spans `3 sigma` (the kernel has decayed to ~1% at the edge).
    pub const DEFAULT: Self = Self {
        radius: 3.0,
        sigma: 1.0,
    };

    /// Construct a filter, clamping `radius` and `sigma` to a sane positive
    /// minimum so the kernel is always well formed.
    #[must_use]
    pub fn new(radius: f32, sigma: f32) -> Self {
        Self {
            radius: radius.max(1.0),
            sigma: sigma.max(1.0e-3),
        }
    }
}

impl Default for GaussianFilter {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Unnormalised Gaussian kernel `exp(-x^2 / (2 sigma^2))` on `|x| <= radius`,
/// else `0`. Strictly non-negative; peaks at `g(0) = 1`.
#[must_use]
fn gaussian_kernel(x: f32, radius: f32, sigma: f32) -> f32 {
    if x.abs() > radius {
        return 0.0;
    }
    let z = x / sigma;
    ops::exp(-0.5 * z * z)
}

/// Produce the next mip level with a separable Gaussian reduction, or `None`
/// when `src` is already `1x1` or has a non-reducible (odd, `>1`) dimension.
#[must_use]
pub fn gaussian_downsample(
    src: &Rgba8Image,
    space: ColorSpace,
    filter: GaussianFilter,
) -> Option<Rgba8Image> {
    let (r, s) = (filter.radius, filter.sigma);
    downsample_2x(src, space, r, |t| gaussian_kernel(t, r, s))
}

/// Build the full Gaussian mip chain from `base` to `1x1` (inclusive of
/// `base`).
#[must_use]
pub fn generate_mip_chain_gaussian(
    base: Rgba8Image,
    space: ColorSpace,
    filter: GaussianFilter,
) -> Vec<Rgba8Image> {
    let mut chain = Vec::new();
    let mut current = base;
    loop {
        let next = gaussian_downsample(&current, space, filter);
        chain.push(current);
        match next {
            Some(level) => current = level,
            None => break,
        }
    }
    chain
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn solid(w: u32, h: u32, c: [u8; 4]) -> Rgba8Image {
        Rgba8Image::new(w, h, vec![c; (w as usize) * (h as usize)]).unwrap()
    }

    #[test]
    fn kernel_is_nonnegative_and_peaks_at_centre() {
        let (r, s) = (3.0f32, 1.0f32);
        assert!((gaussian_kernel(0.0, r, s) - 1.0).abs() < 1e-6);
        let mut t = -3.0f32;
        while t <= 3.0 {
            assert!(gaussian_kernel(t, r, s) >= 0.0, "negative weight at {t}");
            assert!(gaussian_kernel(t, r, s) <= 1.0 + 1e-6, "weight > 1 at {t}");
            t += 0.1;
        }
    }

    #[test]
    fn kernel_is_symmetric() {
        let (r, s) = (3.0f32, 1.3f32);
        for &t in &[0.2f32, 0.9, 1.7, 2.6] {
            let a = gaussian_kernel(t, r, s);
            let b = gaussian_kernel(-t, r, s);
            assert!((a - b).abs() < 1e-7, "asymmetric at {t}: {a} vs {b}");
        }
    }

    #[test]
    fn kernel_is_zero_outside_support() {
        let (r, s) = (3.0f32, 1.0f32);
        assert_eq!(gaussian_kernel(3.0001, r, s), 0.0);
        assert_eq!(gaussian_kernel(-5.0, r, s), 0.0);
    }

    #[test]
    fn kernel_monotone_non_increasing_from_centre() {
        let (r, s) = (3.0f32, 1.0f32);
        let mut prev = gaussian_kernel(0.0, r, s);
        let mut t = 0.1f32;
        while t <= r {
            let cur = gaussian_kernel(t, r, s);
            assert!(cur <= prev + 1e-7, "not monotone at t={t}: {cur} > {prev}");
            prev = cur;
            t += 0.1;
        }
    }

    #[test]
    fn constant_image_is_preserved_linear() {
        let img = solid(8, 8, [12, 34, 56, 210]);
        let mip = gaussian_downsample(&img, ColorSpace::Linear, GaussianFilter::DEFAULT).unwrap();
        assert_eq!((mip.width(), mip.height()), (4, 4));
        for t in mip.as_slice() {
            for (c, &tc) in t.iter().enumerate() {
                let d = i16::from(tc) - i16::from(img.as_slice()[0][c]);
                assert!(d.abs() <= 1, "channel {c} drifted by {d}");
            }
        }
    }

    #[test]
    fn constant_image_is_preserved_srgb() {
        let img = solid(8, 8, [200, 100, 25, 255]);
        let mip =
            gaussian_downsample(&img, ColorSpace::Srgb, GaussianFilter::new(2.0, 0.8)).unwrap();
        for t in mip.as_slice() {
            for (c, &tc) in t.iter().take(3).enumerate() {
                let d = i16::from(tc) - i16::from(img.as_slice()[0][c]);
                assert!(d.abs() <= 1, "srgb channel {c} drifted by {d}");
            }
            assert_eq!(t[3], 255);
        }
    }

    #[test]
    fn no_overshoot_on_hard_edge() {
        // A non-negative kernel is a convex blend of the source texels, so every
        // output must stay inside the *input* range -- here [60, 190]. A ringing
        // sinc kernel (Lanczos/Kaiser) would under/overshoot past those bounds;
        // using a non-extreme step makes that distinction observable (an extreme
        // 0/255 step would be masked by the u8 gamut clamp).
        let (lo, hi) = (60u8, 190u8);
        let mut texels = Vec::new();
        for _y in 0..8u32 {
            for x in 0..8u32 {
                let v = if x < 4 { lo } else { hi };
                texels.push([v, v, v, 255]);
            }
        }
        let img = Rgba8Image::new(8, 8, texels).unwrap();
        let mip = gaussian_downsample(&img, ColorSpace::Linear, GaussianFilter::DEFAULT).unwrap();
        for t in mip.as_slice() {
            for &c in t.iter().take(3) {
                // +/-1 for the single linear<->u8 round trip.
                assert!(
                    c + 1 >= lo && c <= hi + 1,
                    "overshoot past [{lo},{hi}]: {c}"
                );
            }
        }
    }

    #[test]
    fn larger_sigma_blurs_more() {
        // A single bright column; a wider Gaussian spreads its energy more, so
        // the peak output column is not larger than a sharp one's.
        let mut texels = Vec::new();
        for _y in 0..8u32 {
            for x in 0..8u32 {
                let v = if x == 3 { 255 } else { 0 };
                texels.push([v, v, v, 255]);
            }
        }
        let img = Rgba8Image::new(8, 8, texels).unwrap();
        let sharp =
            gaussian_downsample(&img, ColorSpace::Linear, GaussianFilter::new(3.0, 0.5)).unwrap();
        let wide =
            gaussian_downsample(&img, ColorSpace::Linear, GaussianFilter::new(3.0, 2.0)).unwrap();
        let peak = |m: &Rgba8Image| m.as_slice().iter().map(|t| t[0]).max().unwrap();
        assert!(
            peak(&wide) <= peak(&sharp),
            "wider sigma should not sharpen"
        );
    }

    #[test]
    fn non_square_pot_carries_dimension_of_one() {
        let img = solid(4, 1, [50, 60, 70, 80]);
        let m1 = gaussian_downsample(&img, ColorSpace::Linear, GaussianFilter::DEFAULT).unwrap();
        assert_eq!((m1.width(), m1.height()), (2, 1));
        let m2 = gaussian_downsample(&m1, ColorSpace::Linear, GaussianFilter::DEFAULT).unwrap();
        assert_eq!((m2.width(), m2.height()), (1, 1));
        assert!(gaussian_downsample(&m2, ColorSpace::Linear, GaussianFilter::DEFAULT).is_none());
    }

    #[test]
    fn odd_dimension_is_not_reducible() {
        let img = solid(3, 2, [1, 2, 3, 4]);
        assert!(gaussian_downsample(&img, ColorSpace::Linear, GaussianFilter::DEFAULT).is_none());
    }

    #[test]
    fn mip_chain_has_log2_plus_one_levels() {
        let chain = generate_mip_chain_gaussian(
            solid(8, 8, [1, 1, 1, 255]),
            ColorSpace::Linear,
            GaussianFilter::DEFAULT,
        );
        let dims: Vec<(u32, u32)> = chain.iter().map(|i| (i.width(), i.height())).collect();
        assert_eq!(dims, vec![(8, 8), (4, 4), (2, 2), (1, 1)]);
    }
}
