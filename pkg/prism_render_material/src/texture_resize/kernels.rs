//! One-dimensional reconstruction kernels for the separable resampler.
//!
//! Each kernel is defined on its *normalised* distance domain (source-sample
//! distance divided by the per-axis filter scale) and returns `0` outside its
//! [`ResizeFilter::radius`] support, so the driver in
//! [`super`] can treat every filter uniformly. Pure analytic math -- no AI/ML.
//!
//! # References
//! * Keys, "Cubic Convolution Interpolation for Digital Image Processing"
//!   (1981) -- the Catmull-Rom (`a = -1/2`) cubic.
//! * Turkowski, "Filters for Common Resampling Tasks" (1990) -- box, triangle,
//!   and windowed-sinc resampling conventions.

use core::f32::consts::PI;

use bevy_math::ops;

use crate::texture_filter::mn_kernel;

/// Reconstruction filter selectable for a resize.
///
/// Ordered by cost / support width: [`Box`](ResizeFilter::Box) (nearest-ish,
/// 1-tap) → [`Triangle`](ResizeFilter::Triangle) (linear) →
/// [`CatmullRom`](ResizeFilter::CatmullRom) (C1 cubic, mild overshoot) →
/// [`Mitchell`](ResizeFilter::Mitchell) (balanced B=C=1/3 reconstruction) →
/// [`BSpline`](ResizeFilter::BSpline) (non-negative, ring-free) →
/// [`Lanczos3`](ResizeFilter::Lanczos3) (sharpest, visible ringing).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResizeFilter {
    /// Nearest-neighbour box of half-width `0.5`.
    Box,
    /// Linear (triangle / tent) of radius `1`.
    Triangle,
    /// Catmull-Rom cubic (`a = -1/2`) of radius `2`, interpolating.
    CatmullRom,
    /// Mitchell-Netravali balanced cubic (`B = C = 1/3`) of radius `2` -- the
    /// recommended general-purpose resampling kernel (mild ring, mild blur).
    Mitchell,
    /// Cubic B-spline (`B = 1, C = 0`) of radius `2`: strictly non-negative,
    /// never rings or overshoots -- the smooth reconstruction choice.
    BSpline,
    /// Three-lobe Lanczos windowed sinc of radius `3`.
    Lanczos3,
}

/// Normalised sinc, `sin(pi x) / (pi x)`, with the removable singularity at 0.
#[inline]
fn sinc(x: f32) -> f32 {
    if x.abs() < 1.0e-8 {
        1.0
    } else {
        let p = PI * x;
        ops::sin(p) / p
    }
}

impl ResizeFilter {
    /// Half-width of the kernel support on the normalised-distance domain.
    #[must_use]
    pub fn radius(self) -> f32 {
        match self {
            ResizeFilter::Box => 0.5,
            ResizeFilter::Triangle => 1.0,
            ResizeFilter::CatmullRom | ResizeFilter::Mitchell | ResizeFilter::BSpline => 2.0,
            ResizeFilter::Lanczos3 => 3.0,
        }
    }

    /// Evaluate the kernel at normalised distance `x`.
    #[must_use]
    pub fn eval(self, x: f32) -> f32 {
        let ax = x.abs();
        match self {
            ResizeFilter::Box => {
                if ax <= 0.5 {
                    1.0
                } else {
                    0.0
                }
            }
            ResizeFilter::Triangle => {
                if ax < 1.0 {
                    1.0 - ax
                } else {
                    0.0
                }
            }
            ResizeFilter::CatmullRom => {
                // Keys cubic with a = -1/2.
                let a = -0.5;
                if ax < 1.0 {
                    ((a + 2.0) * ax - (a + 3.0)) * ax * ax + 1.0
                } else if ax < 2.0 {
                    (((ax - 5.0) * ax + 8.0) * ax - 4.0) * a
                } else {
                    0.0
                }
            }
            // Mitchell-Netravali reconstruction cubics reuse the already-tested
            // general `(B, C)` kernel from `texture_filter`; the resize domain is
            // an identity map (`fscale == 1`) so distance units already match.
            ResizeFilter::Mitchell => mn_kernel(ax, 1.0 / 3.0, 1.0 / 3.0),
            ResizeFilter::BSpline => mn_kernel(ax, 1.0, 0.0),
            ResizeFilter::Lanczos3 => {
                if ax < 3.0 {
                    sinc(x) * sinc(x / 3.0)
                } else {
                    0.0
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [ResizeFilter; 6] = [
        ResizeFilter::Box,
        ResizeFilter::Triangle,
        ResizeFilter::CatmullRom,
        ResizeFilter::Mitchell,
        ResizeFilter::BSpline,
        ResizeFilter::Lanczos3,
    ];

    // Interpolating (and box) kernels equal 1 at the origin and 0 at every
    // nonzero integer; the Mitchell/B-spline reconstruction cubics do not.
    const INTERPOLATING: [ResizeFilter; 4] = [
        ResizeFilter::Box,
        ResizeFilter::Triangle,
        ResizeFilter::CatmullRom,
        ResizeFilter::Lanczos3,
    ];

    #[test]
    fn unit_gain_at_zero_for_interpolating_kernels() {
        for f in INTERPOLATING {
            assert!((f.eval(0.0) - 1.0).abs() < 1.0e-6, "{f:?}");
        }
    }

    #[test]
    fn symmetric() {
        for f in ALL {
            for i in 0..30 {
                let x = i as f32 * 0.1;
                assert!((f.eval(x) - f.eval(-x)).abs() < 1.0e-6, "{f:?} x={x}");
            }
        }
    }

    #[test]
    fn zero_outside_radius() {
        for f in ALL {
            let r = f.radius();
            assert_eq!(f.eval(r + 0.01), 0.0, "{f:?}");
            assert_eq!(f.eval(-(r + 0.5)), 0.0, "{f:?}");
        }
    }

    #[test]
    fn interpolating_kernels_vanish_at_nonzero_integers() {
        // Triangle, Catmull-Rom, and Lanczos are interpolating: zero at every
        // nonzero integer offset (so they reproduce samples exactly).
        for f in [
            ResizeFilter::Triangle,
            ResizeFilter::CatmullRom,
            ResizeFilter::Lanczos3,
        ] {
            for k in 1..=3 {
                assert!(f.eval(k as f32).abs() < 1.0e-6, "{f:?} k={k}");
            }
        }
    }

    #[test]
    fn catmull_rom_matches_closed_form_samples() {
        // Spot values of the a=-1/2 Keys cubic.
        let f = ResizeFilter::CatmullRom;
        assert!((f.eval(0.5) - 0.5625).abs() < 1.0e-6);
        assert!((f.eval(1.5) + 0.0625).abs() < 1.0e-6);
    }

    #[test]
    fn catmull_rom_matches_mitchell_general_form() {
        // The resize Keys-form Catmull-Rom (independent closed form here) must
        // agree with the general (B, C) = (0, 1/2) Mitchell kernel reused from
        // `texture_filter` -- two independently derived implementations.
        let f = ResizeFilter::CatmullRom;
        for i in -40..=40 {
            let x = i as f32 * 0.0625;
            assert!(
                (f.eval(x) - mn_kernel(x.abs(), 0.0, 0.5)).abs() < 1.0e-6,
                "x={x}"
            );
        }
    }

    #[test]
    fn bspline_is_strictly_non_negative() {
        // The cubic B-spline (B=1, C=0) has no negative lobe, so it never rings.
        let f = ResizeFilter::BSpline;
        for i in -40..=40 {
            let x = i as f32 * 0.0625;
            assert!(f.eval(x) >= -1.0e-7, "x={x} -> {}", f.eval(x));
        }
    }

    #[test]
    fn mitchell_has_sub_unit_central_lobe() {
        // Reconstruction cubics are not interpolating: the central value is
        // below 1 (Mitchell 16/18, B-spline 2/3), which is why the resampler
        // normalises tap weights.
        assert!((ResizeFilter::Mitchell.eval(0.0) - 16.0 / 18.0).abs() < 1.0e-6);
        assert!((ResizeFilter::BSpline.eval(0.0) - 2.0 / 3.0).abs() < 1.0e-6);
    }

    #[test]
    fn cubics_are_a_partition_of_unity() {
        // Box/triangle/Catmull-Rom/Mitchell/B-spline all sum to 1 over the
        // integer lattice for any shift (Lanczos only approximates this and is
        // excluded). Sample off the half-integers so the box stays single-tap.
        for f in [
            ResizeFilter::Box,
            ResizeFilter::Triangle,
            ResizeFilter::CatmullRom,
            ResizeFilter::Mitchell,
            ResizeFilter::BSpline,
        ] {
            for i in 0..13 {
                let t = i as f32 * 0.07 + 0.013;
                let sum: f32 = (-3..=3).map(|k| f.eval(t - k as f32)).sum();
                assert!((sum - 1.0).abs() < 1.0e-5, "{f:?} t={t} sum={sum}");
            }
        }
    }
}
