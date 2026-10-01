//! L1 / L2 spherical-harmonic irradiance encoding — CPU golden.
//!
//! Adaptive probe volumes (APV, à la Unity's Adaptive Probe Volumes and the
//! classic Ramamoorthi–Hanrahan irradiance environment maps) store low-order
//! diffuse irradiance per probe as a handful of spherical-harmonic (SH)
//! coefficients.  A single constant radiance over the sphere projects onto the
//! L0 band alone, and after cosine-lobe convolution reconstructs a *flat*
//! irradiance of `pi * radiance` regardless of surface normal — the analytic
//! identity this module's tests pin down.
//!
//! This file is the backend-neutral reference for that encode/decode algebra:
//!
//! * [`sh_basis_l1`] / [`sh_basis_l2`] evaluate the real SH basis for a unit
//!   direction, matching the `environment` module's `[K0, K1*y, K1*z, K1*x,
//!   K2xy*xy, K2xy*yz, K2z2*(3z^2-1), K2xy*xz, K2x2*(x^2-y^2)]` ordering.
//! * [`ShL1Irradiance`] / [`ShL2Irradiance`] are per-RGB probes with
//!   [`project_radiance`](ShL1Irradiance::project_radiance) baking primitives
//!   and [`eval_irradiance`](ShL1Irradiance::eval_irradiance) /
//!   [`eval_radiance`](ShL1Irradiance::eval_radiance) decode primitives.
//! * [`cosine_convolution_l1`] / [`cosine_convolution_l2`] expose the
//!   per-band clamped-cosine convolution factors `A0 = pi`, `A1 = 2*pi/3`,
//!   `A2 = pi/4` used by the irradiance decode.
//!
//! # Conventions
//! * Directions are right-handed unit `(x, y, z)` vectors; a degenerate (zero)
//!   direction falls back to `+Y` so basis evaluation never produces `NaN`.
//! * Coefficients are per-RGB (`[[f32; 3]; N]`) so the three channels share one
//!   basis, matching the GPU twin's SH texture layout.
//! * `project_radiance` accumulates `radiance * basis * weight`, where `weight`
//!   is the solid angle of the sample; integrating a constant field with
//!   solid-angle weights (`sum(weight) == 4*pi`) reproduces that field.
//! * `eval_irradiance` convolves with the clamped-cosine lobe and clamps each
//!   channel to be non-negative, because SH ringing can otherwise push a
//!   low-frequency reconstruction below zero and inject negative energy.
//! * Every item is a deterministic pure function: no RNG, no I/O, no GPU, no
//!   `unsafe`, and no allocation.

use bevy_math::Vec3;

/// `0.5 * sqrt(1/pi)` — the L0 SH basis constant (mirrors `environment::K0`).
const K0: f32 = 0.282_094_8;
/// `0.5 * sqrt(3/pi)` — the L1 SH basis constant (mirrors `environment::K1`).
const K1: f32 = 0.488_602_5;
/// `0.5 * sqrt(15/pi)` — L2 off-axis basis constant (mirrors `environment`).
const K2_XY: f32 = 1.092_548_4;
/// `0.25 * sqrt(5/pi)` — L2 `(3z^2 - 1)` basis constant.
const K2_Z2: f32 = 0.315_391_57;
/// `0.25 * sqrt(15/pi)` — L2 `(x^2 - y^2)` basis constant.
const K2_X2: f32 = 0.546_274_2;

/// Clamped-cosine convolution factor for band 0 (`pi`).
pub const A0: f32 = core::f32::consts::PI;
/// Clamped-cosine convolution factor for band 1 (`2*pi/3`).
pub const A1: f32 = 2.094_395_2;
/// Clamped-cosine convolution factor for band 2 (`pi/4`).
pub const A2: f32 = core::f32::consts::FRAC_PI_4;

/// `2 * sqrt(pi)` — the L0 coefficient of a unit constant radiance field.
///
/// Projecting a uniform radiance `L` gives `c0 = L * K0 * 4*pi = L * 2*sqrt(pi)`
/// with every higher band zero; evaluating radiance back yields `c0 * K0 == L`.
const L0_CONST: f32 = 3.544_907_7;

/// Normalises `dir`, falling back to `+Y` for a degenerate (zero) input so the
/// basis evaluation never produces `NaN`.
#[inline]
fn normalize_or_up(dir: Vec3) -> Vec3 {
    let len_sq = dir.length_squared();
    if len_sq > f32::MIN_POSITIVE {
        dir * len_sq.sqrt().recip()
    } else {
        Vec3::Y
    }
}

/// Evaluates the four L0+L1 real SH basis functions for a direction.
///
/// Follows the `[K0, K1*y, K1*z, K1*x]` ordering shared with the `environment`
/// and `radiance_cache` modules.  The direction is normalised first (zero folds
/// to `+Y`).
#[inline]
pub fn sh_basis_l1(dir: Vec3) -> [f32; 4] {
    let d = normalize_or_up(dir);
    [K0, K1 * d.y, K1 * d.z, K1 * d.x]
}

/// Evaluates the nine L0+L1+L2 real SH basis functions for a direction.
///
/// Follows the `[K0, K1*y, K1*z, K1*x, K2xy*xy, K2xy*yz, K2z2*(3z^2-1),
/// K2xy*xz, K2x2*(x^2-y^2)]` ordering shared with the `environment` module.
#[inline]
pub fn sh_basis_l2(dir: Vec3) -> [f32; 9] {
    let d = normalize_or_up(dir);
    let (x, y, z) = (d.x, d.y, d.z);
    [
        K0,
        K1 * y,
        K1 * z,
        K1 * x,
        K2_XY * x * y,
        K2_XY * y * z,
        K2_Z2 * (3.0 * z * z - 1.0),
        K2_XY * x * z,
        K2_X2 * (x * x - y * y),
    ]
}

/// Returns the four L1 clamped-cosine convolution factors `[A0, A1, A1, A1]`.
#[inline]
pub fn cosine_convolution_l1() -> [f32; 4] {
    [A0, A1, A1, A1]
}

/// Returns the nine L2 clamped-cosine convolution factors
/// `[A0, A1, A1, A1, A2, A2, A2, A2, A2]`.
#[inline]
pub fn cosine_convolution_l2() -> [f32; 9] {
    [A0, A1, A1, A1, A2, A2, A2, A2, A2]
}

/// An order-one (L1) per-RGB spherical-harmonic irradiance probe.
///
/// Stores four RGB coefficients following the [`sh_basis_l1`] ordering
/// (4 * 3 = 12 floats).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShL1Irradiance {
    /// Four RGB coefficients following the [`sh_basis_l1`] ordering.
    pub coefficients: [[f32; 3]; 4],
}

impl Default for ShL1Irradiance {
    fn default() -> Self {
        Self::ZERO
    }
}

impl ShL1Irradiance {
    /// A probe that radiates nothing.
    pub const ZERO: Self = Self {
        coefficients: [[0.0; 3]; 4],
    };

    /// Builds a probe describing a uniform radiance `color` over the sphere.
    ///
    /// Only the L0 band is populated (`c0 = color * 2*sqrt(pi)`); every higher
    /// coefficient stays zero, exactly as a solid-angle projection of a
    /// constant field would produce.
    #[inline]
    pub fn from_constant(color: [f32; 3]) -> Self {
        let mut probe = Self::ZERO;
        probe.coefficients[0] = [
            color[0] * L0_CONST,
            color[1] * L0_CONST,
            color[2] * L0_CONST,
        ];
        probe
    }

    /// Accumulates a directional radiance sample weighted by its solid angle.
    ///
    /// `direction` points toward the incoming radiance; `weight` is the solid
    /// angle the sample represents (`4*pi/N` for a uniform `N`-point lattice).
    #[inline]
    pub fn project_radiance(&mut self, direction: Vec3, radiance: [f32; 3], weight: f32) {
        let basis = sh_basis_l1(direction);
        for (coefficient, basis_value) in self.coefficients.iter_mut().zip(basis) {
            let scaled = basis_value * weight;
            coefficient[0] += radiance[0] * scaled;
            coefficient[1] += radiance[1] * scaled;
            coefficient[2] += radiance[2] * scaled;
        }
    }

    /// Reconstructs the unconvolved radiance leaving the probe along `dir`.
    ///
    /// Clamped to be non-negative so SH ringing cannot emit negative radiance.
    #[inline]
    pub fn eval_radiance(&self, dir: Vec3) -> Vec3 {
        let basis = sh_basis_l1(dir);
        let mut sum = [0.0f32; 3];
        for index in 0..4 {
            for ch in 0..3 {
                sum[ch] += self.coefficients[index][ch] * basis[index];
            }
        }
        Vec3::new(sum[0].max(0.0), sum[1].max(0.0), sum[2].max(0.0))
    }

    /// Convolves the probe with the clamped-cosine lobe to produce irradiance
    /// at a surface `normal`.  Dividing by `pi` yields Lambertian diffuse.
    ///
    /// The result is clamped to be non-negative per channel.
    #[inline]
    pub fn eval_irradiance(&self, normal: Vec3) -> Vec3 {
        let basis = sh_basis_l1(normal);
        let band = cosine_convolution_l1();
        let mut sum = [0.0f32; 3];
        for index in 0..4 {
            let factor = band[index] * basis[index];
            for ch in 0..3 {
                sum[ch] += self.coefficients[index][ch] * factor;
            }
        }
        Vec3::new(sum[0].max(0.0), sum[1].max(0.0), sum[2].max(0.0))
    }

    /// Adds another probe's coefficients scaled by `weight` in place.
    #[inline]
    pub fn add_scaled(&mut self, other: &ShL1Irradiance, weight: f32) {
        for (dst, src) in self.coefficients.iter_mut().zip(other.coefficients) {
            for ch in 0..3 {
                dst[ch] += src[ch] * weight;
            }
        }
    }

    /// Scales every coefficient by `factor` in place.
    #[inline]
    pub fn scale(&mut self, factor: f32) {
        for c in self.coefficients.iter_mut() {
            for ch in 0..3 {
                c[ch] *= factor;
            }
        }
    }
}

/// An order-two (L2) per-RGB spherical-harmonic irradiance probe.
///
/// Stores nine RGB coefficients following the [`sh_basis_l2`] ordering
/// (9 * 3 = 27 floats).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShL2Irradiance {
    /// Nine RGB coefficients following the [`sh_basis_l2`] ordering.
    pub coefficients: [[f32; 3]; 9],
}

impl Default for ShL2Irradiance {
    fn default() -> Self {
        Self::ZERO
    }
}

impl ShL2Irradiance {
    /// A probe that radiates nothing.
    pub const ZERO: Self = Self {
        coefficients: [[0.0; 3]; 9],
    };

    /// Builds a probe describing a uniform radiance `color` over the sphere.
    ///
    /// Only the L0 band is populated; higher bands stay zero, matching a
    /// solid-angle projection of a constant field.
    #[inline]
    pub fn from_constant(color: [f32; 3]) -> Self {
        let mut probe = Self::ZERO;
        probe.coefficients[0] = [
            color[0] * L0_CONST,
            color[1] * L0_CONST,
            color[2] * L0_CONST,
        ];
        probe
    }

    /// Accumulates a directional radiance sample weighted by its solid angle.
    #[inline]
    pub fn project_radiance(&mut self, direction: Vec3, radiance: [f32; 3], weight: f32) {
        let basis = sh_basis_l2(direction);
        for (coefficient, basis_value) in self.coefficients.iter_mut().zip(basis) {
            let scaled = basis_value * weight;
            coefficient[0] += radiance[0] * scaled;
            coefficient[1] += radiance[1] * scaled;
            coefficient[2] += radiance[2] * scaled;
        }
    }

    /// Reconstructs the unconvolved radiance leaving the probe along `dir`.
    ///
    /// Clamped to be non-negative per channel.
    #[inline]
    pub fn eval_radiance(&self, dir: Vec3) -> Vec3 {
        let basis = sh_basis_l2(dir);
        let mut sum = [0.0f32; 3];
        for index in 0..9 {
            for ch in 0..3 {
                sum[ch] += self.coefficients[index][ch] * basis[index];
            }
        }
        Vec3::new(sum[0].max(0.0), sum[1].max(0.0), sum[2].max(0.0))
    }

    /// Convolves the probe with the clamped-cosine lobe to produce irradiance
    /// at a surface `normal`.  Clamped to be non-negative per channel.
    #[inline]
    pub fn eval_irradiance(&self, normal: Vec3) -> Vec3 {
        let basis = sh_basis_l2(normal);
        let band = cosine_convolution_l2();
        let mut sum = [0.0f32; 3];
        for index in 0..9 {
            let factor = band[index] * basis[index];
            for ch in 0..3 {
                sum[ch] += self.coefficients[index][ch] * factor;
            }
        }
        Vec3::new(sum[0].max(0.0), sum[1].max(0.0), sum[2].max(0.0))
    }

    /// Adds another probe's coefficients scaled by `weight` in place.
    #[inline]
    pub fn add_scaled(&mut self, other: &ShL2Irradiance, weight: f32) {
        for (dst, src) in self.coefficients.iter_mut().zip(other.coefficients) {
            for ch in 0..3 {
                dst[ch] += src[ch] * weight;
            }
        }
    }

    /// Scales every coefficient by `factor` in place.
    #[inline]
    pub fn scale(&mut self, factor: f32) {
        for c in self.coefficients.iter_mut() {
            for ch in 0..3 {
                c[ch] *= factor;
            }
        }
    }

    /// Truncates this L2 probe to its L0+L1 bands as an [`ShL1Irradiance`].
    #[inline]
    pub fn to_l1(&self) -> ShL1Irradiance {
        ShL1Irradiance {
            coefficients: [
                self.coefficients[0],
                self.coefficients[1],
                self.coefficients[2],
                self.coefficients[3],
            ],
        }
    }
}

/// A deterministic Fibonacci-lattice unit direction, index `i` of `n` samples.
///
/// Used by the tests to integrate a constant radiance field with uniform
/// solid-angle weights (`4*pi/n`), reproducing the analytic SH projection.
#[cfg(test)]
fn fibonacci_dir(i: usize, n: usize) -> Vec3 {
    use bevy_math::ops;
    let n_f = n as f32;
    // z uniformly in (-1, 1), azimuth by the golden angle.
    let z = 1.0 - (2.0 * i as f32 + 1.0) / n_f;
    let r = (1.0 - z * z).max(0.0).sqrt();
    let golden = core::f32::consts::PI * (3.0 - (5.0f32).sqrt());
    let phi = golden * i as f32;
    Vec3::new(r * ops::cos(phi), r * ops::sin(phi), z)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FOUR_PI: f32 = 4.0 * core::f32::consts::PI;

    #[test]
    fn basis_matches_known_axis_values() {
        // L0 is constant; L1 picks out y, z, x respectively.
        let b = sh_basis_l1(Vec3::Y);
        assert!((b[0] - K0).abs() < 1e-6);
        assert!((b[1] - K1).abs() < 1e-6);
        assert!(b[2].abs() < 1e-6 && b[3].abs() < 1e-6);

        let b = sh_basis_l1(Vec3::X);
        assert!((b[3] - K1).abs() < 1e-6);
    }

    #[test]
    fn zero_direction_folds_to_up() {
        let b = sh_basis_l1(Vec3::ZERO);
        // +Y fallback: y basis is K1, x/z zero.
        assert!((b[1] - K1).abs() < 1e-6);
        assert!(b[2].abs() < 1e-6 && b[3].abs() < 1e-6);
        for v in sh_basis_l2(Vec3::ZERO) {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn from_constant_reconstructs_radiance_l1() {
        let probe = ShL1Irradiance::from_constant([2.0, 3.0, 4.0]);
        for dir in [Vec3::X, Vec3::Y, Vec3::Z, Vec3::new(0.3, -0.6, 0.7)] {
            let r = probe.eval_radiance(dir);
            assert!((r.x - 2.0).abs() < 1e-4, "{r:?}");
            assert!((r.y - 3.0).abs() < 1e-4, "{r:?}");
            assert!((r.z - 4.0).abs() < 1e-4, "{r:?}");
        }
    }

    #[test]
    fn constant_radiance_gives_flat_irradiance_l1() {
        // Analytic opposite direction: project a constant field, decode
        // irradiance, and expect a flat pi * radiance for every normal.
        let l = [1.0, 0.5, 0.25];
        let n = 4096;
        let w = FOUR_PI / n as f32;
        let mut probe = ShL1Irradiance::ZERO;
        for i in 0..n {
            probe.project_radiance(fibonacci_dir(i, n), l, w);
        }
        let expected = [A0 * l[0], A0 * l[1], A0 * l[2]];
        for normal in [Vec3::X, Vec3::NEG_Y, Vec3::Z, Vec3::new(-0.5, 0.5, 0.7)] {
            let e = probe.eval_irradiance(normal);
            assert!((e.x - expected[0]).abs() < 0.03 * expected[0], "x {e:?}");
            assert!((e.y - expected[1]).abs() < 0.03 * expected[0], "y {e:?}");
            assert!((e.z - expected[2]).abs() < 0.03 * expected[0], "z {e:?}");
        }
    }

    #[test]
    fn constant_radiance_gives_flat_irradiance_l2() {
        let l = [0.8, 0.8, 0.8];
        let n = 4096;
        let w = FOUR_PI / n as f32;
        let mut probe = ShL2Irradiance::ZERO;
        for i in 0..n {
            probe.project_radiance(fibonacci_dir(i, n), l, w);
        }
        let expected = A0 * l[0];
        let mut first = None;
        for normal in [Vec3::X, Vec3::NEG_Y, Vec3::Z, Vec3::new(0.2, -0.3, 0.9)] {
            let e = probe.eval_irradiance(normal);
            assert!((e.x - expected).abs() < 0.03 * expected, "{e:?}");
            // Flatness: all normals agree.
            let v = e.x;
            match first {
                None => first = Some(v),
                Some(f) => assert!((v - f).abs() < 0.03 * expected, "flat {v} vs {f}"),
            }
        }
    }

    #[test]
    fn from_constant_matches_projection_l0() {
        // from_constant should match an actual solid-angle projection on L0.
        let l = [1.0, 2.0, 3.0];
        let n = 8192;
        let w = FOUR_PI / n as f32;
        let mut projected = ShL2Irradiance::ZERO;
        for i in 0..n {
            projected.project_radiance(fibonacci_dir(i, n), l, w);
        }
        let direct = ShL2Irradiance::from_constant(l);
        for ch in 0..3 {
            assert!(
                (projected.coefficients[0][ch] - direct.coefficients[0][ch]).abs() < 0.02,
                "c0[{ch}] {} vs {}",
                projected.coefficients[0][ch],
                direct.coefficients[0][ch]
            );
        }
    }

    #[test]
    fn irradiance_is_non_negative_under_ringing() {
        // A single bright directional sample induces ringing; the clamp must
        // keep every channel non-negative for all normals.
        let mut probe = ShL1Irradiance::ZERO;
        probe.project_radiance(Vec3::Z, [5.0, 5.0, 5.0], 1.0);
        for normal in [Vec3::NEG_Z, Vec3::X, Vec3::new(0.1, 0.2, -0.97)] {
            let e = probe.eval_irradiance(normal);
            assert!(e.x >= 0.0 && e.y >= 0.0 && e.z >= 0.0, "{e:?}");
        }
    }

    #[test]
    fn scale_and_add_scaled_are_linear() {
        let base = ShL1Irradiance::from_constant([1.0, 1.0, 1.0]);
        let mut acc = ShL1Irradiance::ZERO;
        acc.add_scaled(&base, 0.25);
        acc.add_scaled(&base, 0.75);
        let r = acc.eval_radiance(Vec3::Y);
        assert!((r.x - 1.0).abs() < 1e-4, "{r:?}");

        let mut s = base;
        s.scale(2.0);
        let r = s.eval_radiance(Vec3::Y);
        assert!((r.x - 2.0).abs() < 1e-4, "{r:?}");
    }

    #[test]
    fn l2_to_l1_preserves_low_bands() {
        let mut probe = ShL2Irradiance::from_constant([1.0, 2.0, 3.0]);
        probe.coefficients[1] = [0.1, 0.2, 0.3];
        let l1 = probe.to_l1();
        assert_eq!(l1.coefficients[0], probe.coefficients[0]);
        assert_eq!(l1.coefficients[1], probe.coefficients[1]);
    }
}
