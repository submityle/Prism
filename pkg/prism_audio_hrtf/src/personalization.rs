//! Classic (non-ML) personalized HRTF selection from anthropometry.
//!
//! Measured HRTF datasets are captured on specific heads. The timbre and
//! localization cues they encode depend strongly on the subject's anatomy -
//! head width/depth, head circumference, and pinna (outer-ear) geometry. When
//! several standard datasets (SADIE, CIPIC, ...) are available, the closest
//! anatomical match is usually the most convincing default for a listener,
//! which is the classic "best match" personalization used before any
//! data-driven approaches.
//!
//! This module implements exactly that, with **no machine learning**: a plain,
//! deterministic, weighted/normalized Euclidean distance over published
//! anthropometric measurements. Given a listener's [`Anthropometry`] and a set
//! of [`HrtfCandidate`]s, [`select_best`] returns the nearest candidate.
//!
//! Each measurement is optional, so partial subject data (e.g. only head
//! circumference and pinna height) still produces a ranking over the
//! dimensions that both the subject and a candidate share. Dimensions are
//! normalized by a documented per-field population scale so that millimetre
//! differences across heterogeneous fields are comparable before weighting.
//!
//! # Distance helper
//!
//! [`measured_distance_bounds`] is a small bridge to the existing distance
//! machinery: it returns the nearest measured radii bracketing a target
//! distance in a chosen dataset, so callers can hand those to
//! [`crate::nearfield`] / [`crate::interpolation`] for the actual radial blend.
//! It does **not** reimplement interpolation.
//!
//! # Real-time contract
//!
//! Selection runs off the audio thread (it allocates nothing itself but is a
//! setup-time decision). [`Anthropometry::distance_to`] and
//! [`measured_distance_bounds`] are allocation-free and panic-free.
//!
//! # Determinism
//!
//! The only transcendental used is a square root, routed through
//! [`bevy_math::ops`], so rankings are bit-reproducible across targets.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, or Steam Audio
//! source or derived code; no AI/ML.
//!
//! # Relationship
//!
//! Consumes [`crate::dataset::HrtfDataset`] (via [`HrtfCandidate`]) and feeds
//! the chosen dataset into [`crate::interpolation`] / [`crate::nearfield`].
//! Independent of [`crate::sofa`] loading: a candidate may originate from any
//! [`crate::sofa::HrirSource`].

use bevy_math::ops;
use prism_audio_core::math::Sample;

use crate::dataset::HrtfDataset;

/// A single anthropometric field: its population scale and the subject's value.
///
/// `scale` is a representative population spread (metres) used to normalize the
/// field so differences across heterogeneous measurements are comparable.
struct Field {
    scale: Sample,
    weight: Sample,
    value: Option<Sample>,
}

/// Published-style anthropometric measurements for a listener or dataset
/// subject, in **metres**.
///
/// Fields mirror the public CIPIC/anthropometry conventions. Every field is
/// optional so partial data still produces a ranking over shared dimensions.
/// All values are head/ear anatomy only; nothing here is biometric identity.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Anthropometry {
    /// Maximum head breadth (ear-to-ear), metres.
    pub head_width: Option<Sample>,
    /// Maximum head length (front-to-back), metres.
    pub head_depth: Option<Sample>,
    /// Head circumference, metres.
    pub head_circumference: Option<Sample>,
    /// Pinna (cavum concha) height, metres.
    pub pinna_height: Option<Sample>,
    /// Pinna width, metres.
    pub pinna_width: Option<Sample>,
    /// Cavum concha depth, metres.
    pub concha_depth: Option<Sample>,
    /// Biacromial shoulder width, metres.
    pub shoulder_width: Option<Sample>,
}

impl Anthropometry {
    /// Creates an empty measurement set (every field unknown).
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        Self {
            head_width: None,
            head_depth: None,
            head_circumference: None,
            pinna_height: None,
            pinna_width: None,
            concha_depth: None,
            shoulder_width: None,
        }
    }

    /// Sets the head width (metres) and returns `self` for chaining.
    #[must_use]
    #[inline]
    pub const fn with_head_width(mut self, m: Sample) -> Self {
        self.head_width = Some(m);
        self
    }

    /// Sets the head depth (metres) and returns `self` for chaining.
    #[must_use]
    #[inline]
    pub const fn with_head_depth(mut self, m: Sample) -> Self {
        self.head_depth = Some(m);
        self
    }

    /// Sets the head circumference (metres) and returns `self` for chaining.
    #[must_use]
    #[inline]
    pub const fn with_head_circumference(mut self, m: Sample) -> Self {
        self.head_circumference = Some(m);
        self
    }

    /// Sets the pinna height (metres) and returns `self` for chaining.
    #[must_use]
    #[inline]
    pub const fn with_pinna_height(mut self, m: Sample) -> Self {
        self.pinna_height = Some(m);
        self
    }

    /// Sets the pinna width (metres) and returns `self` for chaining.
    #[must_use]
    #[inline]
    pub const fn with_pinna_width(mut self, m: Sample) -> Self {
        self.pinna_width = Some(m);
        self
    }

    /// Sets the cavum concha depth (metres) and returns `self` for chaining.
    #[must_use]
    #[inline]
    pub const fn with_concha_depth(mut self, m: Sample) -> Self {
        self.concha_depth = Some(m);
        self
    }

    /// Sets the shoulder width (metres) and returns `self` for chaining.
    #[must_use]
    #[inline]
    pub const fn with_shoulder_width(mut self, m: Sample) -> Self {
        self.shoulder_width = Some(m);
        self
    }

    /// The fields in a fixed order with their population scales and weights.
    ///
    /// Scales are representative adult population spreads (metres); weights
    /// emphasize the pinna, whose geometry dominates elevation cues, over the
    /// larger but less individuating torso/head-size terms.
    #[inline]
    fn fields(&self) -> [Field; 7] {
        [
            Field {
                scale: 0.012,
                weight: 1.0,
                value: self.head_width,
            },
            Field {
                scale: 0.013,
                weight: 1.0,
                value: self.head_depth,
            },
            Field {
                scale: 0.020,
                weight: 1.0,
                value: self.head_circumference,
            },
            Field {
                scale: 0.004,
                weight: 2.0,
                value: self.pinna_height,
            },
            Field {
                scale: 0.003,
                weight: 2.0,
                value: self.pinna_width,
            },
            Field {
                scale: 0.003,
                weight: 1.5,
                value: self.concha_depth,
            },
            Field {
                scale: 0.030,
                weight: 0.5,
                value: self.shoulder_width,
            },
        ]
    }

    /// The number of fields with a known value.
    #[must_use]
    #[inline]
    pub fn known_count(&self) -> usize {
        self.fields().iter().filter(|f| f.value.is_some()).count()
    }

    /// The normalized, weighted Euclidean distance to `other`.
    ///
    /// Only fields present in **both** sets contribute. Each contributing
    /// field is normalized by its population scale and multiplied by its
    /// weight; the sum is divided by the total participating weight and
    /// square-rooted, so the distance is a dimensionless root-mean-square
    /// deviation that is comparable regardless of how many fields overlap.
    ///
    /// Returns `None` when the two sets share no known field.
    #[must_use]
    pub fn distance_to(&self, other: &Anthropometry) -> Option<Sample> {
        let a = self.fields();
        let b = other.fields();
        let mut acc = 0.0;
        let mut used_weight = 0.0;
        for (fa, fb) in a.iter().zip(b.iter()) {
            if let (Some(va), Some(vb)) = (fa.value, fb.value) {
                let d = (va - vb) / fa.scale;
                acc += fa.weight * d * d;
                used_weight += fa.weight;
            }
        }
        if used_weight <= 0.0 {
            return None;
        }
        Some(ops::sqrt(acc / used_weight))
    }
}

/// A candidate HRTF dataset paired with the subject it was measured on.
#[derive(Debug, Clone, PartialEq)]
pub struct HrtfCandidate {
    /// The measured dataset.
    pub dataset: HrtfDataset,
    /// The anthropometry of the subject the dataset was measured on.
    pub anthropometry: Anthropometry,
}

impl HrtfCandidate {
    /// Pairs a `dataset` with the subject `anthropometry` it was measured on.
    #[must_use]
    #[inline]
    pub fn new(dataset: HrtfDataset, anthropometry: Anthropometry) -> Self {
        Self {
            dataset,
            anthropometry,
        }
    }
}

/// The outcome of [`select_best`]: the winning candidate index and its score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Selection {
    /// Index into the candidate slice of the best match.
    pub index: usize,
    /// The normalized, weighted distance of the winner (smaller is closer).
    pub distance: Sample,
}

/// Selects the candidate whose subject anthropometry is closest to `subject`.
///
/// Uses [`Anthropometry::distance_to`]. Candidates that share no known field
/// with the subject are skipped. On ties the lower index wins (deterministic).
///
/// Returns `None` if there are no candidates or none is comparable to the
/// subject.
#[must_use]
pub fn select_best(subject: &Anthropometry, candidates: &[HrtfCandidate]) -> Option<Selection> {
    let mut best: Option<Selection> = None;
    for (index, candidate) in candidates.iter().enumerate() {
        let Some(distance) = subject.distance_to(&candidate.anthropometry) else {
            continue;
        };
        match best {
            Some(current) if distance >= current.distance => {}
            _ => best = Some(Selection { index, distance }),
        }
    }
    best
}

/// The measured radii in `dataset` that bracket `target_distance` (metres).
///
/// Returns `(lower, upper)` where `lower <= target <= upper` using the nearest
/// measured radii on each side; if the target is outside the measured range
/// both entries clamp to the nearest measured radius. This is only a bracket
/// lookup - the actual radial blend is performed by [`crate::interpolation`]
/// and [`crate::nearfield`], which this helper intentionally does not duplicate.
///
/// Returns `None` only for an empty dataset (impossible for a validated
/// [`HrtfDataset`]), kept for total correctness.
#[must_use]
pub fn measured_distance_bounds(
    dataset: &HrtfDataset,
    target_distance: Sample,
) -> Option<(Sample, Sample)> {
    let measurements = dataset.measurements();
    let first = measurements.first()?;
    let mut lower = first.distance;
    let mut upper = first.distance;
    let mut has_lower = false;
    let mut has_upper = false;
    for m in measurements {
        let r = m.distance;
        if r <= target_distance && (!has_lower || r > lower) {
            lower = r;
            has_lower = true;
        }
        if r >= target_distance && (!has_upper || r < upper) {
            upper = r;
            has_upper = true;
        }
    }
    if !has_lower {
        lower = upper;
    }
    if !has_upper {
        upper = lower;
    }
    Some((lower, upper))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{HrtfDataset, Measurement};
    #[cfg(not(feature = "std"))]
    use alloc::vec;
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

    fn approx(a: Sample, b: Sample, eps: Sample) -> bool {
        (a - b).abs() <= eps
    }

    fn tiny_dataset(radii: &[Sample]) -> HrtfDataset {
        let measurements: Vec<Measurement> = radii
            .iter()
            .map(|&r| Measurement::new(0.0, 0.0, r))
            .collect();
        let n = measurements.len();
        HrtfDataset::from_samples(48_000, 1, measurements, vec![1.0; n], vec![1.0; n]).unwrap()
    }

    #[test]
    fn identical_anthropometry_is_zero_distance() {
        let a = Anthropometry::new()
            .with_head_width(0.15)
            .with_pinna_height(0.065);
        assert!(approx(a.distance_to(&a).unwrap(), 0.0, 1e-6));
    }

    #[test]
    fn no_shared_fields_is_incomparable() {
        let a = Anthropometry::new().with_head_width(0.15);
        let b = Anthropometry::new().with_pinna_height(0.065);
        assert_eq!(a.distance_to(&b), None);
    }

    #[test]
    fn distance_grows_with_difference() {
        let subject = Anthropometry::new().with_head_width(0.150);
        let near = Anthropometry::new().with_head_width(0.152);
        let far = Anthropometry::new().with_head_width(0.170);
        let dn = subject.distance_to(&near).unwrap();
        let df = subject.distance_to(&far).unwrap();
        assert!(df > dn);
    }

    #[test]
    fn single_field_distance_is_normalized_difference() {
        // One field: distance = |dv| / scale (weight cancels in the RMS).
        let subject = Anthropometry::new().with_head_width(0.150);
        let other = Anthropometry::new().with_head_width(0.162);
        // scale for head_width is 0.012, difference 0.012 -> normalized 1.0.
        assert!(approx(subject.distance_to(&other).unwrap(), 1.0, 1e-5));
    }

    #[test]
    fn pinna_is_weighted_above_shoulders() {
        // Same normalized deviation on pinna vs shoulders -> pinna costs more.
        let subject = Anthropometry::new()
            .with_pinna_height(0.065)
            .with_shoulder_width(0.400);
        let pinna_off = Anthropometry::new()
            .with_pinna_height(0.069) // +scale (0.004)
            .with_shoulder_width(0.400);
        let shoulder_off = Anthropometry::new()
            .with_pinna_height(0.065)
            .with_shoulder_width(0.430); // +scale (0.030)
        let dp = subject.distance_to(&pinna_off).unwrap();
        let ds = subject.distance_to(&shoulder_off).unwrap();
        assert!(dp > ds);
    }

    #[test]
    fn select_best_picks_nearest() {
        let subject = Anthropometry::new().with_head_width(0.150);
        let candidates = vec![
            HrtfCandidate::new(
                tiny_dataset(&[1.0]),
                Anthropometry::new().with_head_width(0.180),
            ),
            HrtfCandidate::new(
                tiny_dataset(&[1.0]),
                Anthropometry::new().with_head_width(0.151),
            ),
            HrtfCandidate::new(
                tiny_dataset(&[1.0]),
                Anthropometry::new().with_head_width(0.165),
            ),
        ];
        let sel = select_best(&subject, &candidates).unwrap();
        assert_eq!(sel.index, 1);
        assert!(sel.distance > 0.0);
    }

    #[test]
    fn select_best_skips_incomparable_candidates() {
        let subject = Anthropometry::new().with_head_width(0.150);
        let candidates = vec![
            HrtfCandidate::new(
                tiny_dataset(&[1.0]),
                Anthropometry::new().with_pinna_width(0.030), // no overlap
            ),
            HrtfCandidate::new(
                tiny_dataset(&[1.0]),
                Anthropometry::new().with_head_width(0.158),
            ),
        ];
        let sel = select_best(&subject, &candidates).unwrap();
        assert_eq!(sel.index, 1);
    }

    #[test]
    fn select_best_none_when_no_candidates() {
        let subject = Anthropometry::new().with_head_width(0.150);
        assert_eq!(select_best(&subject, &[]), None);
    }

    #[test]
    fn select_best_ties_prefer_lower_index() {
        let subject = Anthropometry::new().with_head_width(0.150);
        let candidates = vec![
            HrtfCandidate::new(
                tiny_dataset(&[1.0]),
                Anthropometry::new().with_head_width(0.160),
            ),
            HrtfCandidate::new(
                tiny_dataset(&[1.0]),
                Anthropometry::new().with_head_width(0.160),
            ),
        ];
        let sel = select_best(&subject, &candidates).unwrap();
        assert_eq!(sel.index, 0);
    }

    #[test]
    fn distance_bounds_bracket_target() {
        let ds = tiny_dataset(&[0.5, 1.0, 1.5]);
        let (lo, hi) = measured_distance_bounds(&ds, 1.2).unwrap();
        assert!(approx(lo, 1.0, 1e-6));
        assert!(approx(hi, 1.5, 1e-6));
    }

    #[test]
    fn distance_bounds_clamp_below_range() {
        let ds = tiny_dataset(&[0.5, 1.0, 1.5]);
        let (lo, hi) = measured_distance_bounds(&ds, 0.2).unwrap();
        assert!(approx(lo, 0.5, 1e-6));
        assert!(approx(hi, 0.5, 1e-6));
    }

    #[test]
    fn distance_bounds_clamp_above_range() {
        let ds = tiny_dataset(&[0.5, 1.0, 1.5]);
        let (lo, hi) = measured_distance_bounds(&ds, 9.0).unwrap();
        assert!(approx(lo, 1.5, 1e-6));
        assert!(approx(hi, 1.5, 1e-6));
    }

    #[test]
    fn distance_bounds_exact_hit() {
        let ds = tiny_dataset(&[0.5, 1.0, 1.5]);
        let (lo, hi) = measured_distance_bounds(&ds, 1.0).unwrap();
        assert!(approx(lo, 1.0, 1e-6));
        assert!(approx(hi, 1.0, 1e-6));
    }
}
