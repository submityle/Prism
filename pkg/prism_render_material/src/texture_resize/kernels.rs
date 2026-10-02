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

/// Reconstruction filter selectable for a resize.
///
/// Ordered by cost / support width: [`Box`](ResizeFilter::Box) (nearest-ish,
/// 1-tap) → [`Triangle`](ResizeFilter::Triangle) (linear) →
/// [`CatmullRom`](ResizeFilter::CatmullRom) (C1 cubic, mild overshoot) →
/// [`Lanczos3`](ResizeFilter::Lanczos3) (sharpest, visible ringing).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResizeFilter {
    /// Nearest-neighbour box of half-width `0.5`.
    Box,
    /// Linear (triangle / tent) of radius `1`.
    Triangle,
    /// Catmull-Rom cubic (`a = -1/2`) of radius `2`, interpolating.
    CatmullRom,
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
            ResizeFilter::CatmullRom => 2.0,
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

    const ALL: [ResizeFilter; 4] = [
        ResizeFilter::Box,
        ResizeFilter::Triangle,
        ResizeFilter::CatmullRom,
        ResizeFilter::Lanczos3,
    ];

    #[test]
    fn unit_gain_at_zero() {
        for f in ALL {
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
}
