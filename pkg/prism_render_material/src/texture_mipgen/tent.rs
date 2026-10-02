//! Separable triangle (tent / Bartlett) mip reduction for decoded `RGBA8`.
//!
//! The box ([`box_filter`](super::box_filter)) reducer is a flat 2-tap average
//! and the Gaussian ([`gaussian`](super::gaussian)) reducer is a bell curve;
//! both are strictly non-negative but the box has no roll-off (poor stopband)
//! while the Gaussian needs a `sigma`/`radius` pair to tune. The triangle
//! kernel sits between them: a parameter-free, piecewise-linear "tent" that
//! weights the two nearest output-adjacent texel pairs linearly. It is the
//! classic Bartlett window and the linear (first-order) B-spline, giving a
//! cheap non-negative low-pass with a gentler stopband than the box and no
//! shape parameter to pick. Like the Gaussian it can never over- or
//! under-shoot (convex blend of the source texels), so it is safe for
//! roughness/height/coverage prefiltering where sinc ringing is unacceptable.
//! Only the 1-D tent kernel is declared here; the gamma-correct separable
//! resample loop lives in [`resample_core`](super::resample_core). Everything
//! is deterministic analytic arithmetic -- a CPU golden for a GPU down-sampler,
//! no AI/ML.
//!
//! # Conventions
//! * Reduction factor is exactly 2 per axis; a dimension of `1` is carried
//!   through unchanged (GL `max(1, dim >> 1)` rule).
//! * The (unnormalised) kernel is `t(x) = 1 - |x| / radius` on `|x| < radius`
//!   and `0` outside; [`resample_core`](super::resample_core) divides by the
//!   tap sum so a constant is preserved exactly.
//! * `t(x) >= 0` for all `x`, so the reduction is a convex combination of the
//!   source texels: the result can never over- or under-shoot the local texel
//!   range (no ringing), unlike the sinc-based reducers.
//! * Colour is filtered in scene-linear light exactly once; see
//!   [`resample_core`](super::resample_core) for the working-buffer policy.
//!
//! # References
//! * Bartlett (triangular) window; linear / first-order B-spline reconstruction.
//! * `DirectXTex` `TEX_FILTER_FLAGS` triangle / `NVIDIA` Texture Tools triangle
//!   mip generation.

use alloc::vec::Vec;

use super::box_filter::{ColorSpace, Rgba8Image};
use super::resample_core::downsample_2x;

/// A separable triangle (tent / Bartlett) reduction filter.
///
/// `radius` is the half-support in output-sample units: taps at or beyond it
/// have zero weight. A larger `radius` reaches more source texels and is
/// therefore softer; the parameter-free default `radius = 1` matches the
/// classic tent that spans the two output-adjacent source pairs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TentFilter {
    /// Half-support of the tent, in output-sample units (`>= 1`).
    pub radius: f32,
}

impl TentFilter {
    /// Classic unit tent: `radius = 1`, the parameter-free linear B-spline.
    pub const DEFAULT: Self = Self { radius: 1.0 };

    /// Construct a filter, clamping `radius` to a sane positive minimum so the
    /// kernel always spans at least the nearest source pair.
    #[must_use]
    pub fn new(radius: f32) -> Self {
        Self {
            radius: radius.max(1.0),
        }
    }
}

impl Default for TentFilter {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Unnormalised triangle kernel `1 - |x| / radius` on `|x| < radius`, else `0`.
/// Strictly non-negative; peaks at `t(0) = 1` and decays linearly to `0` at
/// `|x| = radius`.
#[must_use]
fn tent_kernel(x: f32, radius: f32) -> f32 {
    let a = x.abs();
    if a >= radius {
        return 0.0;
    }
    1.0 - a / radius
}

/// Produce the next mip level with a separable triangle reduction, or `None`
/// when `src` is already `1x1` or has a non-reducible (odd, `>1`) dimension.
#[must_use]
pub fn tent_downsample(
    src: &Rgba8Image,
    space: ColorSpace,
    filter: TentFilter,
) -> Option<Rgba8Image> {
    let r = filter.radius;
    downsample_2x(src, space, r, |t| tent_kernel(t, r))
}

/// Build the full triangle mip chain from `base` to `1x1` (inclusive of
/// `base`).
#[must_use]
pub fn generate_mip_chain_tent(
    base: Rgba8Image,
    space: ColorSpace,
    filter: TentFilter,
) -> Vec<Rgba8Image> {
    let mut chain = Vec::new();
    let mut current = base;
    loop {
        let next = tent_downsample(&current, space, filter);
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
        let r = 1.0f32;
        assert!((tent_kernel(0.0, r) - 1.0).abs() < 1e-7);
        let mut t = -1.0f32;
        while t <= 1.0 {
            assert!(tent_kernel(t, r) >= 0.0, "negative weight at {t}");
            t += 0.05;
        }
    }

    #[test]
    fn kernel_decays_linearly_and_is_symmetric() {
        let r = 1.0f32;
        // Linear ramp: t(x) = 1 - |x| for radius 1.
        assert!((tent_kernel(0.25, r) - 0.75).abs() < 1e-7);
        assert!((tent_kernel(0.5, r) - 0.5).abs() < 1e-7);
        assert!((tent_kernel(0.75, r) - 0.25).abs() < 1e-7);
        let mut t = 0.0f32;
        while t <= 1.0 {
            let a = tent_kernel(t, r);
            let b = tent_kernel(-t, r);
            assert!((a - b).abs() < 1e-7, "asymmetric at {t}: {a} vs {b}");
            t += 0.1;
        }
    }

    #[test]
    fn kernel_is_zero_outside_support() {
        let r = 1.0f32;
        assert_eq!(tent_kernel(1.0, r), 0.0);
        assert_eq!(tent_kernel(1.0001, r), 0.0);
        assert_eq!(tent_kernel(-3.0, r), 0.0);
    }

    #[test]
    fn kernel_monotone_non_increasing_from_centre() {
        let r = 1.0f32;
        let mut prev = tent_kernel(0.0, r);
        let mut t = 0.05f32;
        while t <= r {
            let cur = tent_kernel(t, r);
            assert!(cur <= prev + 1e-7, "not monotone at t={t}: {cur} > {prev}");
            prev = cur;
            t += 0.05;
        }
    }

    #[test]
    fn constant_image_is_preserved_linear() {
        let img = solid(8, 8, [12, 34, 56, 210]);
        let mip = tent_downsample(&img, ColorSpace::Linear, TentFilter::DEFAULT).unwrap();
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
        let mip = tent_downsample(&img, ColorSpace::Srgb, TentFilter::new(2.0)).unwrap();
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
        // a non-extreme step makes that observable (0/255 is masked by clamp).
        let (lo, hi) = (60u8, 190u8);
        let mut texels = Vec::new();
        for _y in 0..8u32 {
            for x in 0..8u32 {
                let v = if x < 4 { lo } else { hi };
                texels.push([v, v, v, 255]);
            }
        }
        let img = Rgba8Image::new(8, 8, texels).unwrap();
        let mip = tent_downsample(&img, ColorSpace::Linear, TentFilter::DEFAULT).unwrap();
        for t in mip.as_slice() {
            for &c in t.iter().take(3) {
                assert!(
                    c + 1 >= lo && c <= hi + 1,
                    "overshoot past [{lo},{hi}]: {c}"
                );
            }
        }
    }

    #[test]
    fn wider_radius_is_not_sharper() {
        // A single bright column; a wider tent spreads its energy more, so the
        // peak output column is no larger than a tight tent's.
        let mut texels = Vec::new();
        for _y in 0..8u32 {
            for x in 0..8u32 {
                let v = if x == 3 { 255 } else { 0 };
                texels.push([v, v, v, 255]);
            }
        }
        let img = Rgba8Image::new(8, 8, texels).unwrap();
        let tight = tent_downsample(&img, ColorSpace::Linear, TentFilter::new(1.0)).unwrap();
        let wide = tent_downsample(&img, ColorSpace::Linear, TentFilter::new(3.0)).unwrap();
        let peak = |m: &Rgba8Image| m.as_slice().iter().map(|t| t[0]).max().unwrap();
        assert!(
            peak(&wide) <= peak(&tight),
            "wider radius should not sharpen"
        );
    }

    #[test]
    fn non_square_pot_carries_dimension_of_one() {
        let img = solid(4, 1, [50, 60, 70, 80]);
        let m1 = tent_downsample(&img, ColorSpace::Linear, TentFilter::DEFAULT).unwrap();
        assert_eq!((m1.width(), m1.height()), (2, 1));
        let m2 = tent_downsample(&m1, ColorSpace::Linear, TentFilter::DEFAULT).unwrap();
        assert_eq!((m2.width(), m2.height()), (1, 1));
        assert!(tent_downsample(&m2, ColorSpace::Linear, TentFilter::DEFAULT).is_none());
    }

    #[test]
    fn odd_dimension_is_not_reducible() {
        let img = solid(3, 2, [1, 2, 3, 4]);
        assert!(tent_downsample(&img, ColorSpace::Linear, TentFilter::DEFAULT).is_none());
    }

    #[test]
    fn mip_chain_has_log2_plus_one_levels() {
        let chain = generate_mip_chain_tent(
            solid(8, 8, [1, 1, 1, 255]),
            ColorSpace::Linear,
            TentFilter::DEFAULT,
        );
        let dims: Vec<(u32, u32)> = chain.iter().map(|i| (i.width(), i.height())).collect();
        assert_eq!(dims, vec![(8, 8), (4, 4), (2, 2), (1, 1)]);
    }
}
