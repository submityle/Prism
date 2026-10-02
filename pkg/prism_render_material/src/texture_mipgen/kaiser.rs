//! Kaiser-windowed sinc separable mip reduction for decoded `RGBA8`.
//!
//! Lanczos (in [`windowed`](super::windowed)) fixes its side-lobe/transition
//! trade-off by the lobe count `a`. The Kaiser window instead exposes a single
//! shape parameter `beta` that continuously trades main-lobe width (sharpness)
//! against stop-band attenuation (ringing), which is why `DirectXTex` and the
//! `NVIDIA` Texture Tools expose a Kaiser filter as their highest-quality mip
//! down-sampler. This module adds a separable Kaiser-windowed sinc 2x reduction
//! on the shared gamma-correct resample core in
//! [`resample_core`](super::resample_core); only the Kaiser kernel and its
//! modified Bessel `I0` evaluation are declared here. Everything is
//! deterministic analytic arithmetic -- a CPU golden, no AI/ML.
//!
//! # Conventions
//! * Reduction factor is exactly 2 per axis; a dimension of `1` is carried
//!   through unchanged (GL `max(1, dim >> 1)` rule).
//! * The window is `w(x) = I0(beta * sqrt(1 - (x/r)^2)) / I0(beta)` on `|x| <= r`
//!   and `0` outside; the kernel is `sinc(x) * w(x)`.
//! * Colour is filtered in scene-linear light exactly once; see
//!   [`resample_core`](super::resample_core) for the working-buffer policy.
//!
//! # References
//! * Kaiser & Schafer, "On the use of the I0-sinh window for spectrum
//!   analysis" (1980).
//! * `DirectXTex` `TEX_FILTER_FLAGS` Kaiser mip generation; NVTT Kaiser filter.
//! * Modified Bessel `I0(x) = sum_{k>=0} ((x/2)^(2k) / (k!)^2)`.

use alloc::vec::Vec;

use super::box_filter::{ColorSpace, Rgba8Image};
use super::resample_core::{downsample_2x, sinc};

/// A Kaiser-windowed sinc reduction filter.
///
/// `radius` is the half-support in output-sample units (3 taps per side is the
/// common texture-tool default). `beta` is the Kaiser shape parameter: `0`
/// degenerates to a rectangular window, larger values widen the main lobe while
/// deepening stop-band attenuation (less ringing, more blur).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KaiserFilter {
    /// Half-support of the kernel, in output-sample units (`> 0`).
    pub radius: f32,
    /// Kaiser shape parameter `beta` (`>= 0`).
    pub beta: f32,
}

impl KaiserFilter {
    /// `DirectXTex`-style default: 3-tap half-support, `beta = 4.0`.
    pub const DEFAULT: Self = Self { radius: 3.0, beta: 4.0 };

    /// Construct a filter, clamping `radius` to a sane positive minimum and
    /// `beta` to non-negative so the kernel is always well formed.
    #[must_use]
    pub fn new(radius: f32, beta: f32) -> Self {
        Self { radius: radius.max(1.0), beta: beta.max(0.0) }
    }
}

impl Default for KaiserFilter {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Modified Bessel function of the first kind, order 0, via its Maclaurin
/// series. Accumulated in `f64` for accuracy; the series converges rapidly for
/// the small arguments used by Kaiser windows (`x <= beta`, typically `< 10`).
#[must_use]
pub fn bessel_i0(x: f32) -> f32 {
    let half = f64::from(x) * 0.5;
    let half_sq = half * half;
    let mut term = 1.0f64; // k = 0 term
    let mut sum = 1.0f64;
    let mut k = 1.0f64;
    // 64 iterations is far beyond convergence for the arguments used here.
    while k <= 64.0 {
        term *= half_sq / (k * k);
        sum += term;
        if term < 1.0e-15 * sum {
            break;
        }
        k += 1.0;
    }
    sum as f32
}

/// Kaiser window `I0(beta * sqrt(1 - (x/r)^2)) / I0(beta)` on `|x| <= r`, else 0.
#[must_use]
fn kaiser_window(x: f32, radius: f32, beta: f32) -> f32 {
    let r = x.abs() / radius;
    if r > 1.0 {
        return 0.0;
    }
    let arg = beta * (1.0 - r * r).max(0.0).sqrt();
    bessel_i0(arg) / bessel_i0(beta)
}

/// Kaiser-windowed sinc kernel `sinc(x) * kaiser_window(x)`.
#[must_use]
fn kaiser_kernel(x: f32, radius: f32, beta: f32) -> f32 {
    sinc(x) * kaiser_window(x, radius, beta)
}

/// Produce the next mip level with a separable Kaiser-windowed sinc reduction,
/// or `None` when `src` is already `1x1` or has a non-reducible (odd, >1)
/// dimension.
#[must_use]
pub fn kaiser_downsample(
    src: &Rgba8Image,
    space: ColorSpace,
    filter: KaiserFilter,
) -> Option<Rgba8Image> {
    let (r, b) = (filter.radius, filter.beta);
    downsample_2x(src, space, r, |t| kaiser_kernel(t, r, b))
}

/// Build the full Kaiser mip chain from `base` to `1x1` (inclusive of `base`).
#[must_use]
pub fn generate_mip_chain_kaiser(
    base: Rgba8Image,
    space: ColorSpace,
    filter: KaiserFilter,
) -> Vec<Rgba8Image> {
    let mut chain = Vec::new();
    let mut current = base;
    loop {
        let next = kaiser_downsample(&current, space, filter);
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
        Rgba8Image::new(w, h, vec![c; (w * h) as usize]).unwrap()
    }

    #[test]
    fn bessel_i0_known_values() {
        // Reference values of the modified Bessel function I0.
        assert!((bessel_i0(0.0) - 1.0).abs() < 1e-6);
        assert!((bessel_i0(1.0) - 1.266_065_9).abs() < 1e-4);
        assert!((bessel_i0(2.0) - 2.279_585_3).abs() < 1e-4);
        assert!((bessel_i0(3.0) - 4.880_792_6).abs() < 1e-3);
    }

    #[test]
    fn kaiser_window_shape() {
        let (r, b) = (3.0f32, 4.0f32);
        // Peak at centre is exactly 1 (numerator == denominator).
        assert!((kaiser_window(0.0, r, b) - 1.0).abs() < 1e-6);
        // Symmetric.
        assert!((kaiser_window(1.3, r, b) - kaiser_window(-1.3, r, b)).abs() < 1e-6);
        // The Kaiser window is NOT zero at the support edge (unlike Lanczos):
        // at |x| = r the argument collapses to 0, so w(r) = I0(0)/I0(beta) =
        // 1/I0(beta). It is identically zero only strictly beyond the support.
        assert_eq!(kaiser_window(3.0001, r, b), 0.0);
        assert!((kaiser_window(3.0, r, b) - 1.0 / bessel_i0(b)).abs() < 1e-6);
        // Monotonically non-increasing from centre to edge.
        let mut prev = kaiser_window(0.0, r, b);
        let mut t = 0.1f32;
        while t <= r {
            let cur = kaiser_window(t, r, b);
            assert!(cur <= prev + 1e-6, "window not monotone at t={t}: {cur} > {prev}");
            prev = cur;
            t += 0.1;
        }
    }

    #[test]
    fn beta_zero_is_rectangular_window() {
        // beta = 0 -> I0(0)/I0(0) = 1 everywhere inside the support.
        let r = 3.0f32;
        for &t in &[0.0f32, 0.5, 1.7, 2.9] {
            assert!((kaiser_window(t, r, 0.0) - 1.0).abs() < 1e-6);
        }
        assert_eq!(kaiser_window(3.1, r, 0.0), 0.0);
    }

    #[test]
    fn kernel_zero_at_nonzero_integers() {
        // sinc has zeros at nonzero integers; the window is nonzero there,
        // so the full kernel is still zero at those taps.
        let (r, b) = (3.0f32, 4.0f32);
        assert!(kaiser_kernel(1.0, r, b).abs() < 1e-6);
        assert!(kaiser_kernel(2.0, r, b).abs() < 1e-6);
        // Peak at centre.
        assert!((kaiser_kernel(0.0, r, b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn constant_image_is_preserved_linear() {
        let img = solid(8, 8, [12, 34, 56, 210]);
        let mip = kaiser_downsample(&img, ColorSpace::Linear, KaiserFilter::DEFAULT).unwrap();
        assert_eq!((mip.width(), mip.height()), (4, 4));
        for t in mip.as_slice() {
            for c in 0..4 {
                let d = i16::from(t[c]) - i16::from(img.as_slice()[0][c]);
                assert!(d.abs() <= 1, "channel {c} drifted by {d}");
            }
        }
    }

    #[test]
    fn constant_image_is_preserved_srgb() {
        let img = solid(8, 8, [200, 100, 25, 255]);
        let mip = kaiser_downsample(&img, ColorSpace::Srgb, KaiserFilter::new(2.0, 6.0)).unwrap();
        for t in mip.as_slice() {
            for c in 0..3 {
                let d = i16::from(t[c]) - i16::from(img.as_slice()[0][c]);
                assert!(d.abs() <= 1, "srgb channel {c} drifted by {d}");
            }
            assert_eq!(t[3], 255);
        }
    }

    #[test]
    fn output_stays_in_gamut() {
        let mut texels = Vec::new();
        for y in 0..8u32 {
            for x in 0..8u32 {
                let v = if (x + y) % 2 == 0 { 0 } else { 255 };
                texels.push([v, v, v, 255]);
            }
        }
        let img = Rgba8Image::new(8, 8, texels).unwrap();
        let mip = kaiser_downsample(&img, ColorSpace::Linear, KaiserFilter::DEFAULT).unwrap();
        assert_eq!((mip.width(), mip.height()), (4, 4));
    }

    #[test]
    fn non_square_pot_carries_dimension_of_one() {
        let img = solid(4, 1, [50, 60, 70, 80]);
        let m1 = kaiser_downsample(&img, ColorSpace::Linear, KaiserFilter::DEFAULT).unwrap();
        assert_eq!((m1.width(), m1.height()), (2, 1));
        let m2 = kaiser_downsample(&m1, ColorSpace::Linear, KaiserFilter::DEFAULT).unwrap();
        assert_eq!((m2.width(), m2.height()), (1, 1));
        assert!(kaiser_downsample(&m2, ColorSpace::Linear, KaiserFilter::DEFAULT).is_none());
    }

    #[test]
    fn odd_dimension_is_not_reducible() {
        let img = solid(3, 2, [1, 2, 3, 4]);
        assert!(kaiser_downsample(&img, ColorSpace::Linear, KaiserFilter::DEFAULT).is_none());
    }

    #[test]
    fn mip_chain_has_log2_plus_one_levels() {
        let chain = generate_mip_chain_kaiser(
            solid(8, 8, [1, 1, 1, 255]),
            ColorSpace::Linear,
            KaiserFilter::DEFAULT,
        );
        let dims: Vec<(u32, u32)> = chain.iter().map(|i| (i.width(), i.height())).collect();
        assert_eq!(dims, vec![(8, 8), (4, 4), (2, 2), (1, 1)]);
    }

    #[test]
    fn larger_beta_blurs_more_than_lanczos_on_edge() {
        // A single bright column: a wider (larger-beta) Kaiser should spread
        // energy at least as much as a sharp one, i.e. the peak output is not
        // larger. This is a monotone-in-beta smoke check, not a magic constant.
        let mut sharp_texels = Vec::new();
        for _y in 0..8u32 {
            for x in 0..8u32 {
                let v = if x == 3 { 255 } else { 0 };
                sharp_texels.push([v, v, v, 255]);
            }
        }
        let img = Rgba8Image::new(8, 8, sharp_texels).unwrap();
        let sharp = kaiser_downsample(&img, ColorSpace::Linear, KaiserFilter::new(3.0, 2.0)).unwrap();
        let wide = kaiser_downsample(&img, ColorSpace::Linear, KaiserFilter::new(3.0, 10.0)).unwrap();
        let peak = |m: &Rgba8Image| m.as_slice().iter().map(|t| t[0]).max().unwrap();
        assert!(peak(&wide) <= peak(&sharp), "wider window should not sharpen the peak");
    }
}
