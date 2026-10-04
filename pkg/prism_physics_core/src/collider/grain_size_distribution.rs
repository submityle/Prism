//! Deterministic grain-size distribution sampling for granular scene setup.
//!
//! [`pack_spheres`](super::sphere_packing::pack_spheres) draws each grain's
//! radius *uniformly* from `[radius_min, radius_max]`. Real granular media are
//! rarely uniform: sands, powders and ballast follow characteristic grading
//! curves — most commonly a **log-normal** spread about a median size, or an
//! explicit **sieve** grading expressed as mass fractions retained between
//! successive mesh sizes. Feeding a DEM packing the right size distribution is
//! what makes a column consolidate, a hopper jam, or a drum segregate the way
//! the physical material does.
//!
//! This module turns such a specification into a reproducible stream of grain
//! radii. The sampled radii are the per-grain `radii` array consumed directly
//! by the DEM integrators
//! ([`SphereDemFrictionIntegrator`](super::sphere_dem_friction_integrator::SphereDemFrictionIntegrator),
//! [`GranularPileBody`](super::granular_pile_integrator::GranularPileBody)) and
//! by the boundary driver
//! ([`SphereBoundaryDriver`](super::sphere_boundary_driver::SphereBoundaryDriver)).
//!
//! # Distributions
//!
//! * [`GrainSizeDistribution::uniform`] — a flat spread over `[min, max]`,
//!   matching the built-in sampling of `pack_spheres` but reusable on its own.
//! * [`GrainSizeDistribution::log_normal`] — radii whose *logarithm* is normal,
//!   parameterised by a `median` radius and a dimensionless geometric standard
//!   deviation `gsd ≥ 1`. The spread is truncated to `[min, max]` by rejection
//!   so no grain falls outside the authored band. A sample of the underlying
//!   normal variate uses the Box–Muller transform.
//! * [`GrainSizeDistribution::graded`] — an explicit set of [`SieveBin`]s, each
//!   a half-open radius band `[lower, upper)` carrying a relative weight. A bin
//!   is chosen with probability proportional to its weight and a radius is then
//!   drawn uniformly inside it, reproducing a tabulated grading curve.
//!
//! # Determinism
//!
//! All sampling draws from the crate's [`DeterministicRng`], so a given
//! distribution and seed yield byte-identical radii across machines — the same
//! reproducibility contract the packing relies on for baked caches and
//! networked simulation. The transcendental steps of the log-normal transform
//! are evaluated in `f64` and narrowed to `f32`, matching the crate's scalar
//! policy. Nothing here is derived from Unreal Engine source.

use crate::fracture::rng::DeterministicRng;

/// One band of a graded size distribution: a half-open radius interval
/// `[lower, upper)` carrying a strictly positive relative sampling `weight`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SieveBin {
    lower: f32,
    upper: f32,
    weight: f32,
}

impl SieveBin {
    /// Builds a bin spanning `[lower, upper)` with relative `weight`.
    ///
    /// Returns `None` unless every value is finite, `0 < lower < upper`, and
    /// `weight > 0`. Weights are relative: only their ratios matter, because a
    /// [`GrainSizeDistribution::graded`] normalises the whole set.
    #[must_use]
    pub fn new(lower: f32, upper: f32, weight: f32) -> Option<Self> {
        if !lower.is_finite() || !upper.is_finite() || !weight.is_finite() {
            return None;
        }
        if lower <= 0.0 || upper <= lower || weight <= 0.0 {
            return None;
        }
        Some(Self {
            lower,
            upper,
            weight,
        })
    }

    /// Lower radius bound of the band (inclusive).
    #[must_use]
    pub fn lower(&self) -> f32 {
        self.lower
    }

    /// Upper radius bound of the band (exclusive).
    #[must_use]
    pub fn upper(&self) -> f32 {
        self.upper
    }

    /// Relative sampling weight of the band.
    #[must_use]
    pub fn weight(&self) -> f32 {
        self.weight
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Kind {
    Uniform {
        min: f32,
        max: f32,
    },
    LogNormal {
        log_median: f64,
        log_sigma: f64,
        min: f32,
        max: f32,
    },
    Graded {
        bins: Vec<SieveBin>,
        cumulative: Vec<f32>,
    },
}

/// A deterministic distribution of grain radii.
///
/// Build one with [`GrainSizeDistribution::uniform`],
/// [`GrainSizeDistribution::log_normal`] or [`GrainSizeDistribution::graded`],
/// then draw radii with [`sample`](Self::sample) (single draw against a
/// caller-owned generator) or [`sample_radii`](Self::sample_radii) (a whole
/// seeded array). Every variant guarantees the returned radius lies within the
/// authored `[min_radius, max_radius]` envelope.
#[derive(Clone, Debug, PartialEq)]
pub struct GrainSizeDistribution {
    kind: Kind,
}

impl GrainSizeDistribution {
    /// A flat distribution drawing radii uniformly from `[min, max]`.
    ///
    /// Returns `None` unless both bounds are finite and `0 < min ≤ max`.
    #[must_use]
    pub fn uniform(min: f32, max: f32) -> Option<Self> {
        if !min.is_finite() || !max.is_finite() {
            return None;
        }
        if min <= 0.0 || max < min {
            return None;
        }
        Some(Self {
            kind: Kind::Uniform { min, max },
        })
    }

    /// A log-normal distribution about `median` with geometric standard
    /// deviation `gsd`, truncated to `[min, max]`.
    ///
    /// `gsd` is dimensionless and must be `≥ 1`; `gsd == 1` collapses to a
    /// single size at `median`. Returns `None` unless every argument is finite,
    /// `median > 0`, `gsd ≥ 1`, `0 < min ≤ max`, and `median` lies within
    /// `[min, max]` (so the truncation band actually contains the bulk of the
    /// mass).
    #[must_use]
    pub fn log_normal(median: f32, gsd: f32, min: f32, max: f32) -> Option<Self> {
        if !median.is_finite() || !gsd.is_finite() || !min.is_finite() || !max.is_finite() {
            return None;
        }
        if median <= 0.0 || gsd < 1.0 || min <= 0.0 || max < min {
            return None;
        }
        if !(min..=max).contains(&median) {
            return None;
        }
        // ln of a strictly positive, finite value is finite; evaluate in f64.
        let log_median = (median as f64).ln();
        let log_sigma = (gsd as f64).ln();
        Some(Self {
            kind: Kind::LogNormal {
                log_median,
                log_sigma,
                min,
                max,
            },
        })
    }

    /// A graded distribution over an explicit set of [`SieveBin`]s.
    ///
    /// A bin is chosen with probability proportional to its weight and a radius
    /// is then drawn uniformly inside it. Returns `None` when `bins` is empty.
    /// The relative weights are normalised internally, so only their ratios
    /// matter.
    #[must_use]
    pub fn graded(bins: Vec<SieveBin>) -> Option<Self> {
        if bins.is_empty() {
            return None;
        }
        let total: f32 = bins.iter().map(SieveBin::weight).sum();
        if !(total.is_finite() && total > 0.0) {
            return None;
        }
        let mut cumulative = Vec::with_capacity(bins.len());
        let mut running = 0.0_f32;
        for bin in &bins {
            running += bin.weight / total;
            cumulative.push(running);
        }
        // Guard the final bound against floating-point drift so a draw of
        // exactly the largest `next_unit` still selects the last bin.
        if let Some(last) = cumulative.last_mut() {
            *last = 1.0;
        }
        Some(Self {
            kind: Kind::Graded { bins, cumulative },
        })
    }

    /// The smallest radius the distribution can produce.
    #[must_use]
    pub fn min_radius(&self) -> f32 {
        match &self.kind {
            Kind::Uniform { min, .. } | Kind::LogNormal { min, .. } => *min,
            Kind::Graded { bins, .. } => bins
                .iter()
                .map(SieveBin::lower)
                .fold(f32::INFINITY, f32::min),
        }
    }

    /// The largest radius the distribution can produce.
    #[must_use]
    pub fn max_radius(&self) -> f32 {
        match &self.kind {
            Kind::Uniform { max, .. } | Kind::LogNormal { max, .. } => *max,
            Kind::Graded { bins, .. } => bins.iter().map(SieveBin::upper).fold(0.0_f32, f32::max),
        }
    }

    /// Draws a single radius from the distribution using the caller's `rng`.
    ///
    /// The returned radius is always finite and within
    /// `[min_radius, max_radius]`.
    #[must_use]
    pub fn sample(&self, rng: &mut DeterministicRng) -> f32 {
        match &self.kind {
            Kind::Uniform { min, max } => {
                if max <= min {
                    *min
                } else {
                    rng.next_range(*min, *max)
                }
            }
            Kind::LogNormal {
                log_median,
                log_sigma,
                min,
                max,
            } => self.sample_log_normal(*log_median, *log_sigma, *min, *max, rng),
            Kind::Graded { bins, cumulative } => Self::sample_graded(bins, cumulative, rng),
        }
    }

    /// Draws `count` radii into a fresh `Vec`, seeding a private generator with
    /// `seed`.
    ///
    /// Equivalent to constructing a [`DeterministicRng`] from `seed` and
    /// calling [`sample`](Self::sample) `count` times, so the result is fully
    /// reproducible across machines.
    #[must_use]
    pub fn sample_radii(&self, count: usize, seed: u64) -> Vec<f32> {
        let mut rng = DeterministicRng::new(seed);
        (0..count).map(|_| self.sample(&mut rng)).collect()
    }

    fn sample_log_normal(
        &self,
        log_median: f64,
        log_sigma: f64,
        min: f32,
        max: f32,
        rng: &mut DeterministicRng,
    ) -> f32 {
        // gsd == 1 ⇒ zero spread ⇒ the single median size (clamped for safety).
        // `log_median` is the natural log of the median, so exponentiate it.
        if log_sigma <= 0.0 {
            let median = log_median.exp() as f32;
            return median.clamp(min, max);
        }
        // Rejection-truncate to [min, max]; the bulk of the mass is inside the
        // band because the constructor requires median ∈ [min, max], so a
        // modest attempt budget almost always succeeds.
        let mut last = log_median;
        for _ in 0..32 {
            let z = Self::standard_normal(rng);
            let radius = log_median + log_sigma * z;
            last = radius;
            // exp of a finite argument; evaluate in f64 then narrow.
            let value = radius.exp() as f32;
            if (min..=max).contains(&value) {
                return value;
            }
        }
        // Budget exhausted: clamp the last draw onto the band so the contract
        // (finite, in-range) still holds rather than fabricating a value.
        (last.exp() as f32).clamp(min, max)
    }

    /// A standard normal variate via one half of the Box–Muller transform.
    fn standard_normal(rng: &mut DeterministicRng) -> f64 {
        // next_unit() ∈ [0, 1); nudge off zero so ln() stays finite.
        let u1 = (rng.next_unit() as f64).max(f64::MIN_POSITIVE);
        let u2 = rng.next_unit() as f64;
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        radius * theta.cos()
    }

    fn sample_graded(bins: &[SieveBin], cumulative: &[f32], rng: &mut DeterministicRng) -> f32 {
        let u = rng.next_unit();
        let index = cumulative
            .iter()
            .position(|&c| u < c)
            .unwrap_or(bins.len() - 1);
        let bin = bins[index];
        if bin.upper <= bin.lower {
            bin.lower
        } else {
            rng.next_range(bin.lower, bin.upper)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_rejects_bad_parameters() {
        assert!(GrainSizeDistribution::uniform(0.0, 1.0).is_none());
        assert!(GrainSizeDistribution::uniform(-1.0, 1.0).is_none());
        assert!(GrainSizeDistribution::uniform(2.0, 1.0).is_none());
        assert!(GrainSizeDistribution::uniform(f32::NAN, 1.0).is_none());
        assert!(GrainSizeDistribution::uniform(0.5, 0.5).is_some());
    }

    #[test]
    fn uniform_samples_stay_within_bounds_and_are_deterministic() {
        let dist = GrainSizeDistribution::uniform(0.01, 0.02).unwrap();
        let a = dist.sample_radii(500, 7);
        let b = dist.sample_radii(500, 7);
        assert_eq!(a, b, "same seed must reproduce the identical sequence");
        for r in &a {
            assert!(r.is_finite());
            assert!((0.01..=0.02).contains(r), "radius {r} escaped the band");
        }
        let mean: f32 = a.iter().sum::<f32>() / a.len() as f32;
        assert!(
            (0.013..=0.017).contains(&mean),
            "uniform mean {mean} far from midpoint"
        );
    }

    #[test]
    fn log_normal_rejects_bad_parameters() {
        assert!(GrainSizeDistribution::log_normal(0.0, 1.5, 0.001, 0.1).is_none());
        assert!(GrainSizeDistribution::log_normal(0.01, 0.9, 0.001, 0.1).is_none());
        assert!(GrainSizeDistribution::log_normal(0.01, 1.5, 0.0, 0.1).is_none());
        assert!(GrainSizeDistribution::log_normal(0.01, 1.5, 0.1, 0.05).is_none());
        // median outside [min, max]:
        assert!(GrainSizeDistribution::log_normal(0.5, 1.5, 0.001, 0.1).is_none());
        assert!(GrainSizeDistribution::log_normal(0.01, 1.5, 0.001, 0.1).is_some());
    }

    #[test]
    fn log_normal_samples_stay_within_bounds_and_center_near_median() {
        let dist = GrainSizeDistribution::log_normal(0.01, 1.4, 0.003, 0.03).unwrap();
        let a = dist.sample_radii(4000, 42);
        let b = dist.sample_radii(4000, 42);
        assert_eq!(a, b, "same seed must reproduce the identical sequence");
        for r in &a {
            assert!(r.is_finite());
            assert!((0.003..=0.03).contains(r), "radius {r} escaped the band");
        }
        // The log-normal mean exceeds the median but should sit comfortably
        // inside the truncation band for this modest spread.
        let mean: f32 = a.iter().sum::<f32>() / a.len() as f32;
        assert!(
            (0.008..=0.016).contains(&mean),
            "log-normal mean {mean} unexpectedly far from the median scale"
        );
    }

    #[test]
    fn log_normal_with_unit_gsd_is_a_single_size() {
        let dist = GrainSizeDistribution::log_normal(0.01, 1.0, 0.005, 0.02).unwrap();
        for r in dist.sample_radii(16, 1) {
            assert!((r - 0.01).abs() <= 1.0e-6, "degenerate draw {r} drifted");
        }
    }

    #[test]
    fn sieve_bin_rejects_bad_parameters() {
        assert!(SieveBin::new(0.0, 1.0, 1.0).is_none());
        assert!(SieveBin::new(1.0, 1.0, 1.0).is_none());
        assert!(SieveBin::new(2.0, 1.0, 1.0).is_none());
        assert!(SieveBin::new(1.0, 2.0, 0.0).is_none());
        assert!(SieveBin::new(1.0, 2.0, f32::INFINITY).is_none());
        assert!(SieveBin::new(0.5, 1.5, 2.0).is_some());
    }

    #[test]
    fn graded_rejects_empty_and_reports_envelope() {
        assert!(GrainSizeDistribution::graded(Vec::new()).is_none());
        let dist = GrainSizeDistribution::graded(vec![
            SieveBin::new(0.001, 0.002, 1.0).unwrap(),
            SieveBin::new(0.004, 0.008, 3.0).unwrap(),
        ])
        .unwrap();
        assert!((dist.min_radius() - 0.001).abs() <= 1.0e-9);
        assert!((dist.max_radius() - 0.008).abs() <= 1.0e-9);
    }

    #[test]
    fn graded_samples_respect_bins_and_weights() {
        let fine = SieveBin::new(0.001, 0.002, 1.0).unwrap();
        let coarse = SieveBin::new(0.004, 0.008, 3.0).unwrap();
        let dist = GrainSizeDistribution::graded(vec![fine, coarse]).unwrap();
        let a = dist.sample_radii(4000, 99);
        let b = dist.sample_radii(4000, 99);
        assert_eq!(a, b, "same seed must reproduce the identical sequence");

        let mut in_fine = 0usize;
        let mut in_coarse = 0usize;
        for r in &a {
            assert!(r.is_finite());
            if (0.001..0.002).contains(r) {
                in_fine += 1;
            } else if (0.004..0.008).contains(r) {
                in_coarse += 1;
            } else {
                panic!("radius {r} fell outside every bin");
            }
        }
        assert_eq!(
            in_fine + in_coarse,
            a.len(),
            "every draw must land in a bin"
        );
        // Coarse weight is 3× fine, so it should dominate the draws.
        assert!(
            in_coarse > in_fine * 2,
            "weighting ignored: fine {in_fine} vs coarse {in_coarse}"
        );
    }

    #[test]
    fn single_bin_graded_is_uniform_in_that_bin() {
        let dist =
            GrainSizeDistribution::graded(vec![SieveBin::new(0.01, 0.02, 5.0).unwrap()]).unwrap();
        for r in dist.sample_radii(256, 3) {
            assert!((0.01..0.02).contains(&r), "radius {r} escaped the only bin");
        }
    }
}
