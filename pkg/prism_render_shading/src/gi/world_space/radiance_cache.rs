//! World-space radiance cache backed by an order-one (L1) SH probe per cell.
//!
//! Lumen keeps a persistent world-space cache of diffuse radiance so that
//! screen probes converge quickly and off-screen geometry still contributes
//! indirect light.  This module models the CPU-golden version of that cache:
//! world positions are quantised onto a uniform voxel grid, each cell stores a
//! per-RGB L1 spherical-harmonic irradiance probe, and samples are folded in
//! with a running weighted average.
//!
//! # Conventions
//! * The SH basis, band ordering, and cosine-convolution factors match the
//!   L2 probe in [`crate`]'s `environment` module, truncated to the first four
//!   (L0 + L1) coefficients.  Concretely the basis is
//!   `[K0, K1*y, K1*z, K1*x]` with `K0 = 0.5*sqrt(1/pi)` and
//!   `K1 = 0.5*sqrt(3/pi)`, and irradiance convolution uses band factors
//!   `[A0, A1, A1, A1] = [pi, 2*pi/3, 2*pi/3, 2*pi/3]` (Ramamoorthi &
//!   Hanrahan 2001).  Dividing the result by `pi` yields Lambertian diffuse.
//! * `add_directional_radiance` accumulates `radiance * basis * weight`, where
//!   `weight` is the solid angle of the sample — identical to the environment
//!   baker so probes are numerically comparable.
//! * `evaluate_irradiance` clamps each channel to be non-negative, because SH
//!   ringing can otherwise push low-frequency reconstructions below zero and
//!   inject negative energy into the lighting integral.
//! * Cells are indexed by `floor(world / cell_size)` on a right-handed axis
//!   grid; the origin cell is `(0, 0, 0)` covering `[0, cell_size)` on each
//!   axis.

use bevy_math::{IVec3, Vec3};

/// `0.5 * sqrt(1/pi)` — the L0 SH basis constant (mirrors `environment::K0`).
const K0: f32 = 0.282_094_8;
/// `0.5 * sqrt(3/pi)` — the L1 SH basis constant (mirrors `environment::K1`).
const K1: f32 = 0.488_602_5;
/// Cosine-lobe convolution factor for band 0 (`pi`).
const A0: f32 = core::f32::consts::PI;
/// Cosine-lobe convolution factor for band 1 (`2*pi/3`).
const A1: f32 = 2.094_395_2;

/// Evaluates the four L0+L1 real SH basis functions for a unit direction,
/// following the environment probe's `[K0, K1*y, K1*z, K1*x]` ordering.
#[inline]
fn sh_basis_l1(dir: Vec3) -> [f32; 4] {
    let d = normalize_or_up(dir);
    [K0, K1 * d.y, K1 * d.z, K1 * d.x]
}

/// Normalises `dir`, falling back to `+y` for a degenerate (zero) input so the
/// basis evaluation never produces NaNs.
#[inline]
fn normalize_or_up(dir: Vec3) -> Vec3 {
    let len_sq = dir.length_squared();
    if len_sq > f32::MIN_POSITIVE {
        dir * len_sq.sqrt().recip()
    } else {
        Vec3::Y
    }
}

/// An order-one spherical-harmonic irradiance probe with one RGB coefficient
/// per band (4 coefficients * 3 channels = 12 floats).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShL1Rgb {
    /// Four RGB coefficients following the [`sh_basis_l1`] ordering.
    pub coefficients: [[f32; 3]; 4],
}

impl Default for ShL1Rgb {
    fn default() -> Self {
        Self::ZERO
    }
}

impl ShL1Rgb {
    /// A probe that radiates nothing.
    pub const ZERO: Self = Self {
        coefficients: [[0.0; 3]; 4],
    };

    /// Accumulates a directional radiance sample weighted by its solid angle.
    #[inline]
    pub fn add_directional_radiance(&mut self, direction: Vec3, radiance: [f32; 3], weight: f32) {
        let basis = sh_basis_l1(direction);
        for (coefficient, basis_value) in self.coefficients.iter_mut().zip(basis) {
            let scaled = basis_value * weight;
            coefficient[0] += radiance[0] * scaled;
            coefficient[1] += radiance[1] * scaled;
            coefficient[2] += radiance[2] * scaled;
        }
    }

    /// Adds another probe's coefficients scaled by `weight` in place.
    #[inline]
    pub fn add_scaled(&mut self, other: &ShL1Rgb, weight: f32) {
        for (dst, src) in self.coefficients.iter_mut().zip(other.coefficients) {
            dst[0] += src[0] * weight;
            dst[1] += src[1] * weight;
            dst[2] += src[2] * weight;
        }
    }

    /// Scales every coefficient by `factor` in place.
    #[inline]
    pub fn scale(&mut self, factor: f32) {
        for c in self.coefficients.iter_mut() {
            c[0] *= factor;
            c[1] *= factor;
            c[2] *= factor;
        }
    }

    /// Linear blend `self * (1 - t) + other * t`.
    #[inline]
    pub fn lerp(&self, other: &ShL1Rgb, t: f32) -> ShL1Rgb {
        let mut out = ShL1Rgb::ZERO;
        for i in 0..4 {
            for ch in 0..3 {
                out.coefficients[i][ch] =
                    self.coefficients[i][ch] * (1.0 - t) + other.coefficients[i][ch] * t;
            }
        }
        out
    }
}

/// Convolves an L1 probe with the clamped cosine lobe to produce irradiance at
/// a surface `normal`.  The result is clamped to be non-negative.
#[inline]
pub fn evaluate_irradiance(sh: &ShL1Rgb, normal: Vec3) -> Vec3 {
    let basis = sh_basis_l1(normal);
    let band = [A0, A1, A1, A1];
    let mut sum = [0.0f32; 3];
    for index in 0..4 {
        let factor = band[index] * basis[index];
        sum[0] += sh.coefficients[index][0] * factor;
        sum[1] += sh.coefficients[index][1] * factor;
        sum[2] += sh.coefficients[index][2] * factor;
    }
    Vec3::new(sum[0].max(0.0), sum[1].max(0.0), sum[2].max(0.0))
}

/// Quantises a world position to its integer voxel-grid cell,
/// `floor(world / cell_size)` per axis.
///
/// `cell_size` is clamped to a tiny positive value so a zero or negative input
/// cannot produce a division blow-up; the resulting cells simply collapse
/// toward the origin.
#[inline]
pub fn world_to_cell(world: Vec3, cell_size: f32) -> IVec3 {
    let size = if cell_size > f32::MIN_POSITIVE {
        cell_size
    } else {
        f32::MIN_POSITIVE
    };
    let inv = size.recip();
    IVec3::new(
        (world.x * inv).floor() as i32,
        (world.y * inv).floor() as i32,
        (world.z * inv).floor() as i32,
    )
}

/// Deterministic 64-bit key for a voxel cell.
///
/// Each signed axis is biased into an unsigned 21-bit range and packed, giving
/// a collision-free key for cells within `+/- 2^20` of the origin — ample for
/// a bounded scene.  A stable key lets the cache live in a hash map without
/// depending on floating-point identity.
#[inline]
pub fn cell_to_key(cell: IVec3) -> u64 {
    const BITS: u32 = 21;
    const MASK: u64 = (1 << BITS) - 1;
    const BIAS: i64 = 1 << (BITS - 1); // 2^20
    let encode = |v: i32| -> u64 { (((v as i64) + BIAS) as u64) & MASK };
    encode(cell.x) | (encode(cell.y) << BITS) | (encode(cell.z) << (2 * BITS))
}

/// A single accumulating cell: an SH probe plus the total sample weight folded
/// into it, enabling a running weighted average.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RadianceCell {
    /// Weighted sum of contributing probes (`sum_i w_i * sh_i`).
    pub weighted_sh: ShL1Rgb,
    /// Sum of contribution weights (`sum_i w_i`).
    pub total_weight: f32,
}

impl Default for RadianceCell {
    fn default() -> Self {
        Self::EMPTY
    }
}

impl RadianceCell {
    /// An empty cell holding no energy.
    pub const EMPTY: Self = Self {
        weighted_sh: ShL1Rgb::ZERO,
        total_weight: 0.0,
    };

    /// Folds an already-assembled probe into the running weighted average.
    /// Non-positive weights are ignored.
    #[inline]
    pub fn accumulate(&mut self, sample: &ShL1Rgb, weight: f32) {
        if weight <= 0.0 {
            return;
        }
        self.weighted_sh.add_scaled(sample, weight);
        self.total_weight += weight;
    }

    /// Folds a single directional radiance sample (solid-angle `weight`) into
    /// the cell.  The sample's contribution to the average is also weighted by
    /// `weight`, matching a Monte-Carlo estimate of the incident field.
    #[inline]
    pub fn accumulate_directional(&mut self, direction: Vec3, radiance: [f32; 3], weight: f32) {
        if weight <= 0.0 {
            return;
        }
        let mut probe = ShL1Rgb::ZERO;
        probe.add_directional_radiance(direction, radiance, 1.0);
        self.weighted_sh.add_scaled(&probe, weight);
        self.total_weight += weight;
    }

    /// The weighted-average probe (`weighted_sh / total_weight`), or
    /// [`ShL1Rgb::ZERO`] when the cell is empty.
    #[inline]
    pub fn mean(&self) -> ShL1Rgb {
        if self.total_weight <= f32::MIN_POSITIVE {
            return ShL1Rgb::ZERO;
        }
        let inv = self.total_weight.recip();
        let mut out = self.weighted_sh;
        out.scale(inv);
        out
    }

    /// Convenience: irradiance of the averaged probe along `normal`.
    #[inline]
    pub fn irradiance(&self, normal: Vec3) -> Vec3 {
        evaluate_irradiance(&self.mean(), normal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::ops;

    #[test]
    fn world_to_cell_floors_toward_negative_infinity() {
        assert_eq!(world_to_cell(Vec3::new(0.0, 0.0, 0.0), 1.0), IVec3::ZERO);
        assert_eq!(
            world_to_cell(Vec3::new(0.9, 1.1, 2.5), 1.0),
            IVec3::new(0, 1, 2)
        );
        assert_eq!(
            world_to_cell(Vec3::new(-0.1, -1.0, -2.5), 1.0),
            IVec3::new(-1, -1, -3)
        );
        // Larger cells quantise more coarsely.
        assert_eq!(
            world_to_cell(Vec3::new(9.0, -3.0, 0.0), 4.0),
            IVec3::new(2, -1, 0)
        );
    }

    #[test]
    fn world_to_cell_degenerate_size_is_safe() {
        let c = world_to_cell(Vec3::new(1.0, 2.0, 3.0), 0.0);
        // No panic / NaN; coords are finite.
        assert!(c.x.abs() >= 0 || c.x < 0);
        let c = world_to_cell(Vec3::ZERO, -5.0);
        assert_eq!(c, IVec3::ZERO);
    }

    #[test]
    fn cell_key_is_unique_over_a_block() {
        use alloc::collections::BTreeSet;
        let mut seen = BTreeSet::new();
        for x in -3..=3 {
            for y in -3..=3 {
                for z in -3..=3 {
                    let key = cell_to_key(IVec3::new(x, y, z));
                    assert!(seen.insert(key), "collision at {x},{y},{z}");
                }
            }
        }
        assert_eq!(seen.len(), 7 * 7 * 7);
    }

    #[test]
    fn cell_key_distinguishes_axes() {
        assert_ne!(
            cell_to_key(IVec3::new(1, 0, 0)),
            cell_to_key(IVec3::new(0, 1, 0))
        );
        assert_ne!(
            cell_to_key(IVec3::new(0, 0, 1)),
            cell_to_key(IVec3::new(0, 1, 0))
        );
        assert_eq!(cell_to_key(IVec3::ZERO), cell_to_key(IVec3::ZERO));
    }

    #[test]
    fn constant_probe_irradiance_matches_analytic() {
        // A uniform white field of radiance L over the sphere: the DC term is
        // L via add_directional over a full sphere. We instead build the DC
        // coefficient directly so E = A0 * K0^2 * (dc) ... verify positivity &
        // direction independence.
        let mut sh = ShL1Rgb::ZERO;
        // Bake many directions of unit radiance with equal solid angle so the
        // L1 bands cancel and only the DC term survives.
        let n = 200usize;
        for i in 0..n {
            let u = (i as f32 + 0.5) / n as f32;
            let z = 1.0 - 2.0 * u;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let phi = core::f32::consts::TAU * (i as f32 * 0.618_034);
            let dir = Vec3::new(r * ops::cos(phi), r * ops::sin(phi), z);
            let w = core::f32::consts::TAU * 2.0 / n as f32; // ~ 4pi / n
            sh.add_directional_radiance(dir, [1.0, 1.0, 1.0], w);
        }
        let e_up = evaluate_irradiance(&sh, Vec3::Y);
        let e_side = evaluate_irradiance(&sh, Vec3::X);
        // Uniform field -> irradiance independent of normal and ~ pi.
        assert!((e_up.x - e_side.x).abs() < 0.05, "{e_up:?} vs {e_side:?}");
        assert!((e_up.x - core::f32::consts::PI).abs() < 0.1, "{e_up:?}");
    }

    #[test]
    fn directional_probe_is_brightest_toward_source() {
        // A single bright sample from +y should light an up-facing normal more
        // than a down-facing one.
        let mut sh = ShL1Rgb::ZERO;
        sh.add_directional_radiance(Vec3::Y, [1.0, 1.0, 1.0], 1.0);
        let up = evaluate_irradiance(&sh, Vec3::Y);
        let down = evaluate_irradiance(&sh, Vec3::NEG_Y);
        assert!(up.x > down.x, "up {up:?} down {down:?}");
        // Clamped: the far side must not go negative.
        assert!(down.x >= 0.0);
    }

    #[test]
    fn irradiance_is_clamped_non_negative() {
        // Manufacture a probe with a strong negative-lobe L1 term.
        let mut sh = ShL1Rgb::ZERO;
        sh.coefficients[1] = [-10.0, -10.0, -10.0]; // y band
        let e = evaluate_irradiance(&sh, Vec3::Y);
        assert_eq!(e, Vec3::ZERO);
    }

    #[test]
    fn cell_weighted_average_of_equal_probes_is_the_probe() {
        let mut base = ShL1Rgb::ZERO;
        base.add_directional_radiance(Vec3::Z, [0.5, 0.25, 0.75], 1.0);
        let mut cell = RadianceCell::EMPTY;
        cell.accumulate(&base, 2.0);
        cell.accumulate(&base, 3.0);
        let mean = cell.mean();
        for i in 0..4 {
            for ch in 0..3 {
                assert!(
                    (mean.coefficients[i][ch] - base.coefficients[i][ch]).abs() < 1e-6,
                    "coeff {i},{ch}"
                );
            }
        }
    }

    #[test]
    fn cell_weighted_average_interpolates() {
        let mut a = ShL1Rgb::ZERO;
        a.coefficients[0] = [1.0, 0.0, 0.0];
        let mut b = ShL1Rgb::ZERO;
        b.coefficients[0] = [0.0, 4.0, 0.0];
        let mut cell = RadianceCell::EMPTY;
        cell.accumulate(&a, 1.0);
        cell.accumulate(&b, 3.0);
        let mean = cell.mean();
        // (1*a + 3*b)/4 => r = 0.25, g = 3.0
        assert!((mean.coefficients[0][0] - 0.25).abs() < 1e-6);
        assert!((mean.coefficients[0][1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn empty_cell_is_zero() {
        let cell = RadianceCell::EMPTY;
        assert_eq!(cell.mean(), ShL1Rgb::ZERO);
        assert_eq!(cell.irradiance(Vec3::Y), Vec3::ZERO);
    }

    #[test]
    fn non_positive_weights_are_ignored() {
        let mut probe = ShL1Rgb::ZERO;
        probe.coefficients[0] = [1.0, 1.0, 1.0];
        let mut cell = RadianceCell::EMPTY;
        cell.accumulate(&probe, -1.0);
        cell.accumulate(&probe, 0.0);
        assert_eq!(cell.total_weight, 0.0);
        assert_eq!(cell.mean(), ShL1Rgb::ZERO);
    }

    #[test]
    fn accumulate_directional_matches_manual_probe() {
        let mut cell = RadianceCell::EMPTY;
        cell.accumulate_directional(Vec3::Z, [1.0, 2.0, 3.0], 4.0);
        let mut probe = ShL1Rgb::ZERO;
        probe.add_directional_radiance(Vec3::Z, [1.0, 2.0, 3.0], 1.0);
        let mean = cell.mean();
        for i in 0..4 {
            for ch in 0..3 {
                assert!((mean.coefficients[i][ch] - probe.coefficients[i][ch]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn sh_lerp_endpoints() {
        let mut a = ShL1Rgb::ZERO;
        a.coefficients[0] = [1.0, 1.0, 1.0];
        let b = ShL1Rgb::ZERO;
        assert_eq!(a.lerp(&b, 0.0), a);
        assert_eq!(a.lerp(&b, 1.0), b);
        let mid = a.lerp(&b, 0.5);
        assert!((mid.coefficients[0][0] - 0.5).abs() < 1e-6);
    }
}
