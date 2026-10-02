//! Windowed-sinc (Lanczos) separable mip reduction for decoded `RGBA8`.
//!
//! The box filter in [`box_filter`](super::box_filter) is cheap but its boxcar
//! frequency response passes significant energy above the output Nyquist, so
//! AAA texture compressors (`NVIDIA` Texture Tools, `DirectXTex`) down-sample with
//! a windowed-sinc instead to suppress mip shimmering. This module provides a
//! separable Lanczos-2/3 2x reduction built on the shared gamma-correct resample
//! core in [`resample_core`](super::resample_core); only the Lanczos kernel is
//! declared here. The kernel is pure analytic trigonometry -- a deterministic
//! CPU golden, no AI/ML.
//!
//! # Conventions
//! * Reduction factor is exactly 2 per axis; a dimension of `1` is carried
//!   through unchanged (GL `max(1, dim >> 1)` rule), matching the box path.
//! * Colour is filtered in scene-linear light exactly once; see
//!   [`resample_core`](super::resample_core) for the working-buffer policy.
//!
//! # References
//! * Lanczos resampling: windowed sinc `sinc(x) * sinc(x / a)`, support `|x|<a`.
//! * Turkowski, "Filters for Common Resampling Tasks" (1990).

use alloc::vec::Vec;

use super::box_filter::{ColorSpace, Rgba8Image};
use super::resample_core::{downsample_2x, sinc};

/// Windowed-sinc kernel selection for the mip reduction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowedKernel {
    /// Lanczos with `a = 2` (4-tap at unit scale): sharper, slightly more ring.
    Lanczos2,
    /// Lanczos with `a = 3` (6-tap at unit scale): the common texture-tool default.
    Lanczos3,
}

impl WindowedKernel {
    #[must_use]
    fn radius(self) -> f32 {
        match self {
            WindowedKernel::Lanczos2 => 2.0,
            WindowedKernel::Lanczos3 => 3.0,
        }
    }
}

/// Lanczos window `sinc(x) * sinc(x / a)` on `|x| < a`, else 0.
#[must_use]
fn lanczos(x: f32, a: f32) -> f32 {
    if x.abs() < a {
        sinc(x) * sinc(x / a)
    } else {
        0.0
    }
}

/// Produce the next mip level with a separable Lanczos reduction, or `None`
/// when `src` is already `1x1` or has a non-reducible (odd, >1) dimension.
#[must_use]
pub fn windowed_downsample(
    src: &Rgba8Image,
    space: ColorSpace,
    kernel: WindowedKernel,
) -> Option<Rgba8Image> {
    let a = kernel.radius();
    downsample_2x(src, space, a, |t| lanczos(t, a))
}

/// Build the full Lanczos mip chain from `base` to `1x1` (inclusive of `base`).
#[must_use]
pub fn generate_mip_chain_windowed(
    base: Rgba8Image,
    space: ColorSpace,
    kernel: WindowedKernel,
) -> Vec<Rgba8Image> {
    let mut chain = Vec::new();
    let mut current = base;
    loop {
        let next = windowed_downsample(&current, space, kernel);
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
    fn lanczos_identities() {
        // Lanczos peak and symmetry.
        assert!((lanczos(0.0, 3.0) - 1.0).abs() < 1e-6);
        assert!((lanczos(0.7, 3.0) - lanczos(-0.7, 3.0)).abs() < 1e-6);
        // Compact support.
        assert_eq!(lanczos(3.0, 3.0), 0.0);
        assert_eq!(lanczos(2.0, 2.0), 0.0);
        // Zero at nonzero integers inside support (sinc zero, window nonzero).
        assert!(lanczos(1.0, 3.0).abs() < 1e-6);
    }

    #[test]
    fn constant_image_is_preserved_linear() {
        let img = solid(8, 8, [10, 20, 30, 200]);
        let mip = windowed_downsample(&img, ColorSpace::Linear, WindowedKernel::Lanczos3).unwrap();
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
        let img = solid(8, 8, [128, 64, 200, 255]);
        let mip = windowed_downsample(&img, ColorSpace::Srgb, WindowedKernel::Lanczos2).unwrap();
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
        let mip = windowed_downsample(&img, ColorSpace::Linear, WindowedKernel::Lanczos3).unwrap();
        assert_eq!((mip.width(), mip.height()), (4, 4));
    }

    #[test]
    fn non_square_pot_carries_dimension_of_one() {
        let img = solid(4, 1, [50, 60, 70, 80]);
        let m1 = windowed_downsample(&img, ColorSpace::Linear, WindowedKernel::Lanczos2).unwrap();
        assert_eq!((m1.width(), m1.height()), (2, 1));
        let m2 = windowed_downsample(&m1, ColorSpace::Linear, WindowedKernel::Lanczos2).unwrap();
        assert_eq!((m2.width(), m2.height()), (1, 1));
        assert!(windowed_downsample(&m2, ColorSpace::Linear, WindowedKernel::Lanczos2).is_none());
    }

    #[test]
    fn odd_dimension_is_not_reducible() {
        let img = solid(3, 2, [1, 2, 3, 4]);
        assert!(windowed_downsample(&img, ColorSpace::Linear, WindowedKernel::Lanczos3).is_none());
    }

    #[test]
    fn mip_chain_has_log2_plus_one_levels() {
        let chain = generate_mip_chain_windowed(
            solid(8, 8, [1, 1, 1, 255]),
            ColorSpace::Linear,
            WindowedKernel::Lanczos3,
        );
        let dims: Vec<(u32, u32)> = chain.iter().map(|i| (i.width(), i.height())).collect();
        assert_eq!(dims, vec![(8, 8), (4, 4), (2, 2), (1, 1)]);
    }
}
