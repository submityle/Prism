//! Piecewise-constant sampling distributions for importance sampling.
//!
//! The image-based environment light must draw directions in proportion to the
//! radiance stored in its texels. That is a classic inversion-sampling problem:
//! treat the texels as a piecewise-constant function and sample it by inverting
//! its cumulative distribution. This module provides the one- and
//! two-dimensional building blocks (after Pharr, Jakob and Humphreys,
//! "Physically Based Rendering", the `Distribution1D` / `Distribution2D`
//! construction) using only comparisons, additions and multiplications, so no
//! transcendental is involved and the determinism policy is respected.
//!
//! Both samplers operate on the unit domain `[0, 1)` (and `[0, 1)^2`), return a
//! continuous sample with its probability density in that domain, and expose a
//! matching density query for multiple-importance weighting.

use alloc::vec::Vec;

/// Finds the largest index `i` in `0..cdf.len() - 1` with `cdf[i] <= u`.
///
/// `cdf` is non-decreasing with `cdf[0] == 0` and `cdf[last] == 1`; the result
/// is the bin that contains `u`, clamped into the valid range so a `u` of
/// exactly `1.0` (or tiny overshoot) still addresses the final bin.
fn find_interval(cdf: &[f32], u: f32) -> usize {
    // Linear scan is adequate here: environment rows are short and the search is
    // branch-predictable. A monotone `cdf` makes the first failing step the
    // answer.
    let bins = cdf.len() - 1;
    let mut i = 0;
    while i + 1 < bins && cdf[i + 1] <= u {
        i += 1;
    }
    i.min(bins - 1)
}

/// A continuous sample drawn from a [`Distribution1D`].
#[derive(Clone, Copy, Debug)]
pub struct Sample1D {
    /// The sampled position in `[0, 1)`.
    pub x: f32,
    /// The probability density of `x` with respect to the unit domain.
    pub pdf: f32,
    /// The index of the piecewise-constant bin that produced `x`.
    pub offset: usize,
}

/// A piecewise-constant distribution over `[0, 1)` sampled by inverting its
/// cumulative distribution.
#[derive(Clone, Debug, Default)]
pub struct Distribution1D {
    /// The non-negative per-bin function values.
    func: Vec<f32>,
    /// The normalized cumulative distribution, length `func.len() + 1`, running
    /// from `0.0` to `1.0`.
    cdf: Vec<f32>,
    /// The integral of `func` over `[0, 1)`, i.e. the mean of `func`; zero only
    /// for an all-zero function.
    func_int: f32,
}

impl Distribution1D {
    /// Builds the distribution from a slice of non-negative bin values.
    ///
    /// An all-zero (or empty) function yields a degenerate distribution that
    /// samples the domain uniformly, so callers never divide by a zero integral.
    #[must_use]
    pub fn new(func: &[f32]) -> Self {
        let n = func.len();
        let func = func.to_vec();
        let mut cdf = Vec::with_capacity(n + 1);
        cdf.push(0.0);
        let mut running = 0.0f64;
        for &value in &func {
            running += f64::from(value.max(0.0));
            cdf.push(running as f32);
        }
        // `func_int` is the mean of `func`; the unnormalized `cdf` currently
        // holds the running sum, which equals `func_int * n` at the end.
        let total = running;
        let func_int = (total / f64::from(n.max(1) as u32)) as f32;
        if total > 0.0 {
            let inv = 1.0 / total;
            for value in &mut cdf {
                *value = (f64::from(*value) * inv) as f32;
            }
        } else {
            // Uniform fallback: a linear cumulative distribution.
            for (i, value) in cdf.iter_mut().enumerate() {
                *value = i as f32 / n.max(1) as f32;
            }
        }
        Self {
            func,
            cdf,
            func_int,
        }
    }

    /// The number of piecewise-constant bins.
    #[must_use]
    pub fn count(&self) -> usize {
        self.func.len()
    }

    /// The integral of the function over `[0, 1)` (its mean bin value).
    #[must_use]
    pub fn integral(&self) -> f32 {
        self.func_int
    }

    /// Draws a continuous sample from the distribution using the uniform
    /// variate `u` in `[0, 1)`.
    #[must_use]
    pub fn sample_continuous(&self, u: f32) -> Sample1D {
        let n = self.func.len();
        if n == 0 {
            return Sample1D {
                x: 0.0,
                pdf: 0.0,
                offset: 0,
            };
        }
        if self.func_int <= 0.0 {
            // Uniform fallback over the whole domain.
            let x = u.clamp(0.0, 1.0);
            let offset = ((x * n as f32) as usize).min(n - 1);
            return Sample1D {
                x,
                pdf: 1.0,
                offset,
            };
        }
        let offset = find_interval(&self.cdf, u);
        let mut du = u - self.cdf[offset];
        let span = self.cdf[offset + 1] - self.cdf[offset];
        if span > 0.0 {
            du /= span;
        }
        let pdf = self.func[offset] / self.func_int;
        let x = (offset as f32 + du) / n as f32;
        Sample1D { x, pdf, offset }
    }

    /// The probability density the distribution assigns to a position `x` in
    /// `[0, 1)`.
    #[must_use]
    pub fn pdf(&self, x: f32) -> f32 {
        let n = self.func.len();
        if n == 0 || self.func_int <= 0.0 {
            return if n == 0 { 0.0 } else { 1.0 };
        }
        let offset = ((x.clamp(0.0, 1.0) * n as f32) as usize).min(n - 1);
        self.func[offset] / self.func_int
    }
}

/// A continuous sample drawn from a [`Distribution2D`].
#[derive(Clone, Copy, Debug)]
pub struct Sample2D {
    /// The sampled horizontal position in `[0, 1)`.
    pub u: f32,
    /// The sampled vertical position in `[0, 1)`.
    pub v: f32,
    /// The joint probability density of `(u, v)` over the unit square.
    pub pdf: f32,
}

/// A piecewise-constant distribution over the unit square `[0, 1)^2`.
///
/// Sampling factors into a marginal draw over rows followed by a conditional
/// draw along the chosen row, so the two-dimensional inversion reduces to two
/// one-dimensional ones.
#[derive(Clone, Debug, Default)]
pub struct Distribution2D {
    /// One conditional distribution per row (over the horizontal axis).
    conditional: Vec<Distribution1D>,
    /// The marginal distribution over rows, weighted by each row's integral.
    marginal: Distribution1D,
}

impl Distribution2D {
    /// Builds the distribution from a row-major `func` of `width * height`
    /// non-negative values.
    ///
    /// A mismatched length or a zero dimension yields a degenerate distribution
    /// that samples uniformly, keeping every query well defined.
    #[must_use]
    pub fn new(func: &[f32], width: usize, height: usize) -> Self {
        if width == 0 || height == 0 || func.len() != width * height {
            return Self::default();
        }
        let mut conditional = Vec::with_capacity(height);
        let mut row_integrals = Vec::with_capacity(height);
        for row in 0..height {
            let start = row * width;
            let dist = Distribution1D::new(&func[start..start + width]);
            row_integrals.push(dist.integral());
            conditional.push(dist);
        }
        let marginal = Distribution1D::new(&row_integrals);
        Self {
            conditional,
            marginal,
        }
    }

    /// `true` when the distribution has no sampleable rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.conditional.is_empty()
    }

    /// The integral of the function over the unit square (its mean value).
    #[must_use]
    pub fn integral(&self) -> f32 {
        self.marginal.integral()
    }

    /// Draws a continuous sample from the unit square using two uniform variates.
    #[must_use]
    pub fn sample_continuous(&self, u0: f32, u1: f32) -> Sample2D {
        if self.conditional.is_empty() {
            return Sample2D {
                u: u0,
                v: u1,
                pdf: 1.0,
            };
        }
        let marginal = self.marginal.sample_continuous(u1);
        let row = marginal.offset.min(self.conditional.len() - 1);
        let conditional = self.conditional[row].sample_continuous(u0);
        Sample2D {
            u: conditional.x,
            v: marginal.x,
            pdf: marginal.pdf * conditional.pdf,
        }
    }

    /// The joint probability density assigned to `(u, v)` over the unit square.
    #[must_use]
    pub fn pdf(&self, u: f32, v: f32) -> f32 {
        if self.conditional.is_empty() {
            return 1.0;
        }
        let rows = self.conditional.len();
        let row = ((v.clamp(0.0, 1.0) * rows as f32) as usize).min(rows - 1);
        self.marginal.pdf(v) * self.conditional[row].pdf(u)
    }
}

#[cfg(test)]
mod tests {
    use super::super::sampler::Rng;
    use super::*;

    #[test]
    fn uniform_function_samples_uniformly() {
        let dist = Distribution1D::new(&[1.0, 1.0, 1.0, 1.0]);
        assert!((dist.integral() - 1.0).abs() < 1e-6);
        for k in 0..10 {
            let u = k as f32 / 10.0;
            let s = dist.sample_continuous(u);
            assert!((s.pdf - 1.0).abs() < 1e-5);
            assert!((s.x - u).abs() < 1e-5, "x {} vs u {u}", s.x);
        }
    }

    #[test]
    fn weighted_function_concentrates_samples() {
        // The third bin carries all the weight, so every sample lands in it and
        // the density there is the bin count.
        let dist = Distribution1D::new(&[0.0, 0.0, 4.0, 0.0]);
        assert!((dist.integral() - 1.0).abs() < 1e-6);
        let s = dist.sample_continuous(0.5);
        assert_eq!(s.offset, 2);
        assert!(s.x >= 0.5 && s.x <= 0.75, "x {} outside third bin", s.x);
        assert!((dist.pdf(0.6) - 4.0).abs() < 1e-5);
        assert!(dist.pdf(0.1).abs() < 1e-6);
    }

    #[test]
    fn one_d_density_integrates_to_one() {
        let dist = Distribution1D::new(&[0.5, 2.0, 1.0, 3.0, 0.25]);
        let mut rng = Rng::with_stream(13, 1);
        let count = 200_000u32;
        let mut sum = 0.0f64;
        for _ in 0..count {
            let x = rng.next_f32();
            sum += f64::from(dist.pdf(x));
        }
        let mean = sum / f64::from(count);
        assert!((mean - 1.0).abs() < 5e-3, "mean density {mean} should be 1");
    }

    #[test]
    fn empty_distribution_is_inert() {
        let dist = Distribution1D::new(&[]);
        assert_eq!(dist.count(), 0);
        let s = dist.sample_continuous(0.4);
        assert_eq!(s.pdf, 0.0);
    }

    #[test]
    fn two_d_density_integrates_to_one() {
        let func = [0.2f32, 1.0, 0.5, 2.0, 0.1, 3.0];
        let dist = Distribution2D::new(&func, 3, 2);
        let mut rng = Rng::with_stream(23, 2);
        let count = 300_000u32;
        let mut sum = 0.0f64;
        for _ in 0..count {
            let u = rng.next_f32();
            let v = rng.next_f32();
            sum += f64::from(dist.pdf(u, v));
        }
        let mean = sum / f64::from(count);
        assert!((mean - 1.0).abs() < 1e-2, "mean density {mean} should be 1");
    }

    #[test]
    fn two_d_importance_sampling_is_unbiased() {
        // Drawing from the distribution and averaging `func / pdf` must recover
        // the function's integral over the unit square (its mean value).
        let func = [0.2f32, 1.0, 0.5, 2.0, 0.1, 3.0];
        let width = 3usize;
        let height = 2usize;
        let dist = Distribution2D::new(&func, width, height);
        let expected = dist.integral();
        let mut rng = Rng::with_stream(37, 3);
        let count = 300_000u32;
        let mut sum = 0.0f64;
        for _ in 0..count {
            let s = dist.sample_continuous(rng.next_f32(), rng.next_f32());
            let iu = ((s.u * width as f32) as usize).min(width - 1);
            let iv = ((s.v * height as f32) as usize).min(height - 1);
            let value = func[iv * width + iu];
            if s.pdf > 0.0 {
                sum += f64::from(value) / f64::from(s.pdf);
            }
        }
        let estimate = sum / f64::from(count);
        assert!(
            (estimate / f64::from(expected) - 1.0).abs() < 1e-2,
            "estimate {estimate} should match integral {expected}"
        );
    }
}
