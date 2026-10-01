//! Directional clustering of early-reflection taps.
//!
//! A full image-source early-reflection list can hold up to
//! [`crate::early_reflections::MAX_EARLY_REFLECTIONS`] individual taps. Feeding
//! each one through its own delay line and panner is wasteful when many taps
//! arrive from nearly the same direction. This module collapses a tap set into
//! a small fixed set of directional clusters (buckets), aggregating the taps in
//! each bucket in an energy-preserving way. The result is a compact,
//! direction-tagged set of reflection sends suitable for efficient rendering,
//! the same idea behind baked-reflection pipelines in game audio middleware.
//!
//! It performs no per-sample DSP and re-implements none of the image-source
//! geometry: it consumes the public [`crate::early_reflections::ReflectionTap`]
//! outputs and reduces them.
//!
//! # Clusters
//!
//! The buckets are the six axis-aligned face directions of the listener-local
//! frame (`+X`, `-X`, `+Y`, `-Y`, `+Z`, `-Z`; `-Z` is forward). Each tap is
//! assigned to the bucket whose canonical direction has the largest dot product
//! with the tap arrival direction (nearest cosine). There are always
//! [`CLUSTER_COUNT`] buckets; buckets with no taps report zero gain and keep
//! their canonical direction.
//!
//! # Aggregation
//!
//! Within a bucket the taps are combined incoherently:
//!
//! - the cluster energy is the sum of squared tap amplitudes,
//!   `E = sum(gain^2)`, and the reported `gain` is `sqrt(E)`;
//! - the representative direction is the energy-weighted mean of the tap
//!   directions, renormalised (falling back to the bucket canonical direction
//!   if the weighted sum cancels);
//! - the reported `delay_samples` is the energy-weighted mean tap delay,
//!   rounded to the nearest whole sample.
//!
//! Because every valid tap contributes `gain^2` to exactly one bucket, the sum
//! of the per-cluster energies equals the sum of the input tap energies: the
//! reduction is energy-preserving.
//!
//! # Control rate, not audio rate
//!
//! Clustering operates on stack scalars and fixed-size arrays; there is no heap
//! allocation, no locking, and no panicking. Taps with a non-finite or
//! zero-length direction, a non-finite gain, or zero energy are skipped. All
//! transcendental and length math routes through [`bevy_math::ops`].
//!
//! # Relationship
//!
//! This module is a reduction layer over
//! [`crate::early_reflections`]: it consumes the [`ReflectionTap`] list that
//! [`crate::early_reflections::compute_early_reflections`] produces. It is a
//! sibling of [`crate::reflection_directivity`], which instead weights the same
//! taps by a source radiation pattern; the two can be used together (weight
//! first, then cluster) or independently. It owns none of the reflection
//! physics.
//!
//! # Provenance
//!
//! Collapsing many early reflections into a handful of directional sends is
//! standard practice in geometrical room-acoustics rendering. This module is
//! engine-agnostic and contains **no Unreal Engine, Unity, Godot, Wwise, FMOD,
//! Steam Audio, or Google Resonance Audio source or derived code**; it is a
//! plain energy-preserving bucketing of this crate's own reflection taps.

use bevy_math::Vec3;
use bevy_math::ops;

use prism_audio_core::math::Sample;

use crate::early_reflections::ReflectionTap;

/// The number of directional clusters (the six axis-aligned face directions).
pub const CLUSTER_COUNT: usize = 6;

/// The canonical bucket directions, indexed `+X, -X, +Y, -Y, +Z, -Z`.
const CLUSTER_DIRECTIONS: [Vec3; CLUSTER_COUNT] = [
    Vec3::X,
    Vec3::NEG_X,
    Vec3::Y,
    Vec3::NEG_Y,
    Vec3::Z,
    Vec3::NEG_Z,
];

/// A single directional reflection cluster: the energy-preserving aggregate of
/// every tap that fell into one bucket.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ReflectionCluster {
    /// Representative (energy-weighted mean) arrival direction, unit length.
    pub direction: Vec3,
    /// Aggregate linear amplitude, `sqrt(sum(gain^2))` over the bucket taps.
    pub gain: Sample,
    /// Energy-weighted mean path delay, rounded to whole samples.
    pub delay_samples: usize,
    /// Number of input taps that fell into this bucket.
    pub tap_count: u32,
}

impl ReflectionCluster {
    /// An empty cluster pointing along `dir` with zero energy.
    const fn empty(dir: Vec3) -> Self {
        Self {
            direction: dir,
            gain: 0.0,
            delay_samples: 0,
            tap_count: 0,
        }
    }

    /// Whether this cluster received any taps.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.tap_count > 0
    }
}

/// The fixed set of directional clusters produced from a tap set.
///
/// Always holds [`CLUSTER_COUNT`] buckets in canonical order
/// (`+X, -X, +Y, -Y, +Z, -Z`); inactive buckets report zero gain.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ReflectionClusters {
    clusters: [ReflectionCluster; CLUSTER_COUNT],
}

impl ReflectionClusters {
    /// Clusters a tap set directly (same as [`cluster_taps`]).
    #[must_use]
    pub fn from_taps(taps: &[ReflectionTap]) -> Self {
        cluster_taps(taps)
    }

    /// All [`CLUSTER_COUNT`] clusters in canonical order.
    #[must_use]
    pub fn clusters(&self) -> &[ReflectionCluster] {
        &self.clusters
    }

    /// The cluster at `index`, or `None` when out of range.
    #[must_use]
    pub fn cluster(&self, index: usize) -> Option<&ReflectionCluster> {
        self.clusters.get(index)
    }

    /// The total number of clusters (always [`CLUSTER_COUNT`]).
    #[must_use]
    pub fn count(&self) -> usize {
        CLUSTER_COUNT
    }

    /// The number of clusters that received at least one tap.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.clusters.iter().filter(|c| c.is_active()).count()
    }

    /// The total aggregate energy across all clusters, `sum(gain^2)`.
    #[must_use]
    pub fn total_energy(&self) -> Sample {
        let mut sum = 0.0;
        for c in &self.clusters {
            sum += c.gain * c.gain;
        }
        sum
    }
}

/// Normalises a direction, returning `None` when it is non-finite or
/// zero-length.
fn sanitise_direction(dir: Vec3) -> Option<Vec3> {
    if !dir.is_finite() {
        return None;
    }
    let normalised = dir.normalize_or_zero();
    if normalised.length_squared() > 0.5 {
        Some(normalised)
    } else {
        None
    }
}

/// Returns the index of the canonical bucket nearest (largest dot product) to
/// a unit direction.
fn nearest_cluster(dir: Vec3) -> usize {
    let mut best = 0usize;
    let mut best_dot = Sample::NEG_INFINITY;
    for (index, &canonical) in CLUSTER_DIRECTIONS.iter().enumerate() {
        let d = dir.dot(canonical);
        if d > best_dot {
            best_dot = d;
            best = index;
        }
    }
    best
}

/// Clusters an early-reflection tap set into [`CLUSTER_COUNT`] directional
/// buckets, aggregating each bucket in an energy-preserving way.
///
/// Taps with a non-finite or zero-length direction, a non-finite gain, or zero
/// energy are skipped. The result is control-rate: no heap allocation, no
/// locking, no panicking.
///
/// # Examples
///
/// ```
/// use bevy_math::Vec3;
/// use prism_audio_spatial::early_reflections::ReflectionTap;
/// use prism_audio_spatial::reflection_clustering::{cluster_taps, CLUSTER_COUNT};
///
/// let taps = [
///     ReflectionTap { delay_samples: 100, gain: 1.0, direction: Vec3::X, order: 1, is_direct: false },
///     ReflectionTap { delay_samples: 140, gain: 0.5, direction: Vec3::X, order: 2, is_direct: false },
/// ];
/// let clusters = cluster_taps(&taps);
/// assert_eq!(clusters.count(), CLUSTER_COUNT);
/// // Both taps fall into the +X bucket; the aggregate gain is energy-preserving.
/// let plus_x = clusters.cluster(0).unwrap();
/// assert_eq!(plus_x.tap_count, 2);
/// let expected = (1.0f32 * 1.0 + 0.5 * 0.5).sqrt();
/// assert!((plus_x.gain - expected).abs() < 1e-6);
/// ```
#[must_use]
pub fn cluster_taps(taps: &[ReflectionTap]) -> ReflectionClusters {
    let mut energy = [0.0 as Sample; CLUSTER_COUNT];
    let mut dir_sum = [Vec3::ZERO; CLUSTER_COUNT];
    let mut delay_sum = [0.0 as Sample; CLUSTER_COUNT];
    let mut counts = [0u32; CLUSTER_COUNT];

    for tap in taps {
        let Some(dir) = sanitise_direction(tap.direction) else {
            continue;
        };
        if !tap.gain.is_finite() {
            continue;
        }
        let e = tap.gain * tap.gain;
        if e <= 0.0 {
            continue;
        }
        let index = nearest_cluster(dir);
        energy[index] += e;
        dir_sum[index] += dir * e;
        delay_sum[index] += e * (tap.delay_samples as Sample);
        counts[index] += 1;
    }

    let mut clusters = [ReflectionCluster::empty(Vec3::X); CLUSTER_COUNT];
    for (index, slot) in clusters.iter_mut().enumerate() {
        let canonical = CLUSTER_DIRECTIONS[index];
        let e = energy[index];
        if counts[index] == 0 || e <= 0.0 {
            *slot = ReflectionCluster::empty(canonical);
            continue;
        }
        let normalised = dir_sum[index].normalize_or_zero();
        let direction = if normalised.length_squared() > 0.5 {
            normalised
        } else {
            canonical
        };
        let mean_delay = ops::round(delay_sum[index] / e);
        let delay_samples = if mean_delay.is_finite() && mean_delay >= 0.0 {
            mean_delay as usize
        } else {
            0
        };
        *slot = ReflectionCluster {
            direction,
            gain: ops::sqrt(e),
            delay_samples,
            tap_count: counts[index],
        };
    }

    ReflectionClusters { clusters }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: Sample, b: Sample, eps: Sample) -> bool {
        (a - b).abs() <= eps
    }

    fn tap(gain: Sample, direction: Vec3, delay: usize) -> ReflectionTap {
        ReflectionTap {
            delay_samples: delay,
            gain,
            direction,
            order: 1,
            is_direct: false,
        }
    }

    #[test]
    fn count_is_always_cluster_count() {
        let clusters = cluster_taps(&[]);
        assert_eq!(clusters.count(), CLUSTER_COUNT);
        assert_eq!(clusters.clusters().len(), CLUSTER_COUNT);
    }

    #[test]
    fn empty_set_has_zero_energy_and_no_active() {
        let clusters = cluster_taps(&[]);
        assert_eq!(clusters.active_count(), 0);
        assert!(approx(clusters.total_energy(), 0.0, 1e-9));
        for c in clusters.clusters() {
            assert_eq!(c.tap_count, 0);
            assert!(approx(c.gain, 0.0, 1e-9));
        }
    }

    #[test]
    fn empty_bucket_keeps_canonical_direction() {
        let clusters = cluster_taps(&[tap(1.0, Vec3::X, 10)]);
        // The -X bucket (index 1) got nothing: it keeps its canonical dir.
        let minus_x = clusters.cluster(1).unwrap();
        assert_eq!(minus_x.tap_count, 0);
        assert!(approx(minus_x.direction.dot(Vec3::NEG_X), 1.0, 1e-6));
    }

    #[test]
    fn single_direction_falls_into_one_bucket() {
        let taps = [tap(1.0, Vec3::X, 10), tap(0.5, Vec3::X, 20)];
        let clusters = cluster_taps(&taps);
        assert_eq!(clusters.active_count(), 1);
        let plus_x = clusters.cluster(0).unwrap();
        assert_eq!(plus_x.tap_count, 2);
    }

    #[test]
    fn aggregate_gain_is_energy_sum() {
        let taps = [tap(1.0, Vec3::X, 10), tap(0.5, Vec3::X, 20)];
        let clusters = cluster_taps(&taps);
        let expected = ops::sqrt(1.0 + 0.25);
        assert!(approx(clusters.cluster(0).unwrap().gain, expected, 1e-6));
    }

    #[test]
    fn energy_is_conserved_across_clusters() {
        let taps = [
            tap(1.0, Vec3::X, 5),
            tap(0.7, Vec3::NEG_Z, 9),
            tap(0.4, Vec3::Y, 13),
            tap(0.9, Vec3::X, 17),
            tap(0.3, Vec3::NEG_Y, 21),
        ];
        let tap_energy: Sample = taps.iter().map(|t| t.gain * t.gain).sum();
        let clusters = cluster_taps(&taps);
        assert!(approx(clusters.total_energy(), tap_energy, 1e-5));
    }

    #[test]
    fn representative_direction_is_energy_weighted_mean() {
        // Two equal-energy taps symmetric about +X: the weighted mean points
        // back along +X (the +/-Y offsets cancel).
        let a = Vec3::new(1.0, 0.1, 0.0);
        let b = Vec3::new(1.0, -0.1, 0.0);
        let clusters = cluster_taps(&[tap(1.0, a, 10), tap(1.0, b, 10)]);
        let plus_x = clusters.cluster(0).unwrap();
        assert!(approx(plus_x.direction.dot(Vec3::X), 1.0, 1e-5));
    }

    #[test]
    fn representative_direction_is_unit_length() {
        let clusters = cluster_taps(&[
            tap(1.0, Vec3::new(1.0, 0.3, 0.0), 10),
            tap(0.6, Vec3::new(1.0, -0.2, 0.1), 12),
        ]);
        assert!(approx(clusters.cluster(0).unwrap().direction.length(), 1.0, 1e-5));
    }

    #[test]
    fn delay_is_energy_weighted_mean() {
        // Equal gains, delays 100 and 140: mean = 120.
        let clusters = cluster_taps(&[tap(1.0, Vec3::X, 100), tap(1.0, Vec3::X, 140)]);
        assert_eq!(clusters.cluster(0).unwrap().delay_samples, 120);
    }

    #[test]
    fn delay_weights_toward_louder_tap() {
        // gain 1.0 at delay 100 dominates gain 0.1 at delay 200.
        // mean = (1*100 + 0.01*200) / 1.01 = ~100.99 -> rounds to 101.
        let clusters = cluster_taps(&[tap(1.0, Vec3::X, 100), tap(0.1, Vec3::X, 200)]);
        assert_eq!(clusters.cluster(0).unwrap().delay_samples, 101);
    }

    #[test]
    fn uniform_six_axis_taps_spread_evenly() {
        let taps = [
            tap(1.0, Vec3::X, 10),
            tap(1.0, Vec3::NEG_X, 10),
            tap(1.0, Vec3::Y, 10),
            tap(1.0, Vec3::NEG_Y, 10),
            tap(1.0, Vec3::Z, 10),
            tap(1.0, Vec3::NEG_Z, 10),
        ];
        let clusters = cluster_taps(&taps);
        assert_eq!(clusters.active_count(), CLUSTER_COUNT);
        for c in clusters.clusters() {
            assert_eq!(c.tap_count, 1);
            assert!(approx(c.gain, 1.0, 1e-6));
        }
    }

    #[test]
    fn non_finite_direction_is_skipped() {
        let bad = tap(1.0, Vec3::new(Sample::NAN, 0.0, 0.0), 10);
        let good = tap(1.0, Vec3::X, 10);
        let clusters = cluster_taps(&[bad, good]);
        assert_eq!(clusters.cluster(0).unwrap().tap_count, 1);
        assert!(approx(clusters.total_energy(), 1.0, 1e-6));
    }

    #[test]
    fn non_finite_gain_is_skipped() {
        let bad = tap(Sample::INFINITY, Vec3::X, 10);
        let clusters = cluster_taps(&[bad]);
        assert_eq!(clusters.active_count(), 0);
        assert!(clusters.total_energy().is_finite());
    }

    #[test]
    fn zero_direction_tap_is_skipped() {
        let clusters = cluster_taps(&[tap(1.0, Vec3::ZERO, 10)]);
        assert_eq!(clusters.active_count(), 0);
    }

    #[test]
    fn zero_gain_tap_contributes_nothing() {
        let clusters = cluster_taps(&[tap(0.0, Vec3::X, 10), tap(1.0, Vec3::X, 10)]);
        assert_eq!(clusters.cluster(0).unwrap().tap_count, 1);
    }

    #[test]
    fn out_of_range_cluster_is_none() {
        let clusters = cluster_taps(&[]);
        assert!(clusters.cluster(CLUSTER_COUNT).is_none());
    }

    #[test]
    fn from_taps_matches_free_function() {
        let taps = [tap(1.0, Vec3::X, 10), tap(0.5, Vec3::Y, 20)];
        assert_eq!(ReflectionClusters::from_taps(&taps), cluster_taps(&taps));
    }
}
