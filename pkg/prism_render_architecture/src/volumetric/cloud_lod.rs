//! Distance-driven cloud `LOD` bucket selection (design section 11).
//!
//! A cloudscape spanning a large world cannot be `raymarch`ed at full step
//! count and full resolution everywhere: a cloud a few hundred metres away
//! deserves a high step count and a full-resolution buffer, while the same
//! cloud at the horizon should collapse to a coarse `raymarch`, a low-
//! resolution buffer, and finally a camera-facing `imposter` billboard. This
//! module maps a cloud's view distance to a [`CloudLod`] bucket, resolves the
//! per-bucket `raymarch` step count / step length / resolution scale, and
//! computes the `imposter` cross-fade weight so the far transition is a smooth
//! blend rather than a hard switch.
//!
//! Everything here is a pure, deterministic classification: the caller supplies
//! the view distance (and, for binning, a parallel distance slice indexed by
//! layer handle), so no projection or transcendental math beyond the shared
//! `smoothstep` is needed. Bucket thresholds are ordered coarsest-last; the
//! per-bucket workload is monotone in distance (steps only decrease, step
//! length only increases, resolution scale only decreases), matching the
//! screen-coverage `LOD` paradigm the hair subsystem uses.

use alloc::vec::Vec;

use super::math::smoothstep;

/// The discrete cloud `LOD` bucket a view distance falls into.
///
/// Ordered finest-first: [`CloudLod::Near`] keeps the most work and
/// [`CloudLod::Imposter`] the least. The numeric [`CloudLod::rank`] increases
/// monotonically with distance, which is what makes the per-bucket workload
/// monotone.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CloudLod {
    /// Close clouds: highest `raymarch` step count, full-resolution buffer.
    Near,
    /// Mid-distance clouds: reduced step count, half-resolution buffer.
    Mid,
    /// Far clouds: coarse `raymarch`, quarter-resolution buffer.
    Far,
    /// Horizon clouds: camera-facing `imposter` billboard, minimal `raymarch`.
    Imposter,
}

impl CloudLod {
    /// Every bucket in finest-to-coarsest order, used for monotonicity sweeps.
    pub const ALL: [CloudLod; 4] = [
        CloudLod::Near,
        CloudLod::Mid,
        CloudLod::Far,
        CloudLod::Imposter,
    ];

    /// Distance rank: `0` for [`CloudLod::Near`], `3` for [`CloudLod::Imposter`].
    ///
    /// The rank increases with distance, so a larger rank always means less
    /// per-frame work.
    #[must_use]
    pub fn rank(self) -> u8 {
        match self {
            CloudLod::Near => 0,
            CloudLod::Mid => 1,
            CloudLod::Far => 2,
            CloudLod::Imposter => 3,
        }
    }

    /// The fixed `raymarch` / resolution settings this bucket uses.
    ///
    /// The values are authored so that, walking [`CloudLod::ALL`] from finest to
    /// coarsest, [`CloudLodSettings::raymarch_steps`] strictly decreases,
    /// [`CloudLodSettings::step_length`] strictly increases, and
    /// [`CloudLodSettings::resolution_scale`] strictly decreases. This encodes
    /// the design rule "far away: fewer steps, longer steps, lower resolution".
    #[must_use]
    pub fn settings(self) -> CloudLodSettings {
        match self {
            CloudLod::Near => CloudLodSettings {
                raymarch_steps: 128,
                step_length: 1.0,
                resolution_scale: 1.0,
            },
            CloudLod::Mid => CloudLodSettings {
                raymarch_steps: 64,
                step_length: 2.0,
                resolution_scale: 0.5,
            },
            CloudLod::Far => CloudLodSettings {
                raymarch_steps: 32,
                step_length: 4.0,
                resolution_scale: 0.25,
            },
            CloudLod::Imposter => CloudLodSettings {
                raymarch_steps: 8,
                step_length: 8.0,
                resolution_scale: 0.125,
            },
        }
    }
}

/// The `raymarch` step budget and buffer resolution scale for one [`CloudLod`].
///
/// `resolution_scale` is a linear buffer scale in `(0, 1]`: `1.0` is a full-
/// resolution volumetric buffer, `0.5` a half-resolution buffer (a quarter of
/// the pixels), and so on. The temporal upsampler (design section 10)
/// reconstructs the full-resolution image from the scaled buffer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CloudLodSettings {
    /// `raymarch` samples taken along a view ray at this bucket.
    pub raymarch_steps: u32,
    /// World-space length of one `raymarch` step at this bucket.
    pub step_length: f32,
    /// Linear resolution scale of the volumetric buffer in `(0, 1]`.
    pub resolution_scale: f32,
}

/// View-distance boundaries at which clouds drop to the next coarser bucket.
///
/// The invariant `mid_beyond <= far_beyond <= imposter_beyond` is expected. If
/// it is violated the classification in [`select_lod`] still terminates
/// deterministically because it tests the boundaries in ascending order; it
/// simply skips any bucket whose window was collapsed to nothing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CloudLodThresholds {
    /// At or beyond this distance a cloud drops from `Near` to `Mid`.
    pub mid_beyond: f32,
    /// At or beyond this distance a cloud drops from `Mid` to `Far`.
    pub far_beyond: f32,
    /// At or beyond this distance a cloud drops from `Far` to `Imposter`.
    pub imposter_beyond: f32,
}

/// Classifies a view distance into a [`CloudLod`] bucket.
///
/// The distance is any finite non-negative value; negatives simply map to
/// [`CloudLod::Near`] and very large values (including infinity) map to
/// [`CloudLod::Imposter`], so no input can panic. Boundaries are tested in
/// ascending order, which keeps the result deterministic even when the
/// thresholds are not monotone.
#[must_use]
pub fn select_lod(distance: f32, thresholds: CloudLodThresholds) -> CloudLod {
    if distance < thresholds.mid_beyond {
        CloudLod::Near
    } else if distance < thresholds.far_beyond {
        CloudLod::Mid
    } else if distance < thresholds.imposter_beyond {
        CloudLod::Far
    } else {
        CloudLod::Imposter
    }
}

/// Resolves both the bucket and its [`CloudLodSettings`] for a view distance.
///
/// Convenience wrapper over [`select_lod`] followed by [`CloudLod::settings`].
#[must_use]
pub fn resolve_lod(distance: f32, thresholds: CloudLodThresholds) -> (CloudLod, CloudLodSettings) {
    let lod = select_lod(distance, thresholds);
    (lod, lod.settings())
}

/// Cross-fade weight for the `imposter` transition, in `[0, 1]`.
///
/// Returns `0.0` when the volumetric `raymarch` result should be shown in full
/// and `1.0` when the pre-rendered `imposter` billboard should be shown in
/// full; between `start` and `end` it is a `smoothstep` blend so the far
/// transition is a soft cross-fade rather than a hard switch (design section
/// 11 forbids hard switching). The weight is monotonically non-decreasing in
/// distance. When `start == end` the shared `smoothstep` degrades to a hard
/// step without dividing by zero.
#[must_use]
pub fn imposter_fade(distance: f32, start: f32, end: f32) -> f32 {
    smoothstep(start, end, distance)
}

/// A set of cloud layers partitioned by the [`CloudLod`] bucket selected for
/// each this frame.
///
/// Each bucket holds the layer indices (into the caller's distance slice) that
/// landed in it, preserving the input order so downstream dispatch is
/// deterministic. This mirrors the hair `LOD` plan and the virtual-geometry
/// raster `bin`s: the renderer consumes one bucket at a time to build its per-
/// bucket `raymarch` / `imposter` passes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CloudLodBins {
    /// Layer indices classified as [`CloudLod::Near`].
    pub near: Vec<u32>,
    /// Layer indices classified as [`CloudLod::Mid`].
    pub mid: Vec<u32>,
    /// Layer indices classified as [`CloudLod::Far`].
    pub far: Vec<u32>,
    /// Layer indices classified as [`CloudLod::Imposter`].
    pub imposter: Vec<u32>,
}

impl CloudLodBins {
    /// Total number of layer indices across every bucket.
    #[must_use]
    pub fn total(&self) -> usize {
        self.near.len() + self.mid.len() + self.far.len() + self.imposter.len()
    }

    /// Returns `true` when no layer landed in any bucket.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.near.is_empty()
            && self.mid.is_empty()
            && self.far.is_empty()
            && self.imposter.is_empty()
    }

    /// Number of layers routed to a given [`CloudLod`] bucket.
    #[must_use]
    pub fn count_of(&self, lod: CloudLod) -> usize {
        match lod {
            CloudLod::Near => self.near.len(),
            CloudLod::Mid => self.mid.len(),
            CloudLod::Far => self.far.len(),
            CloudLod::Imposter => self.imposter.len(),
        }
    }

    /// Mutable handle to the bucket backing a given [`CloudLod`].
    fn bucket_mut(&mut self, lod: CloudLod) -> &mut Vec<u32> {
        match lod {
            CloudLod::Near => &mut self.near,
            CloudLod::Mid => &mut self.mid,
            CloudLod::Far => &mut self.far,
            CloudLod::Imposter => &mut self.imposter,
        }
    }
}

/// Partitions cloud layers into per-`LOD` `bin`s by view distance.
///
/// Each entry of `layers` is an index into the parallel `distances` slice, so
/// distances stay in one contiguous buffer the culling stage fills. A layer
/// index that falls outside `distances` is skipped rather than panicking, so a
/// stale layer list cannot crash `LOD` selection (the out-of-range-skip rule
/// shared with the virtual-geometry raster `bin`s). Every in-range layer is
/// routed via [`select_lod`], and per-bucket order follows the `layers` input
/// order, keeping the result deterministic.
#[must_use]
pub fn bin_by_distance(
    layers: &[u32],
    distances: &[f32],
    thresholds: CloudLodThresholds,
) -> CloudLodBins {
    let mut bins = CloudLodBins::default();
    for &layer in layers {
        let Some(&distance) = distances.get(layer as usize) else {
            continue;
        };
        let lod = select_lod(distance, thresholds);
        bins.bucket_mut(lod).push(layer);
    }
    bins
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    const THRESHOLDS: CloudLodThresholds = CloudLodThresholds {
        mid_beyond: 1_000.0,
        far_beyond: 5_000.0,
        imposter_beyond: 20_000.0,
    };

    #[test]
    fn select_lod_thresholds_are_ordered() {
        assert_eq!(select_lod(0.0, THRESHOLDS), CloudLod::Near);
        assert_eq!(select_lod(999.0, THRESHOLDS), CloudLod::Near);
        assert_eq!(select_lod(1_000.0, THRESHOLDS), CloudLod::Mid);
        assert_eq!(select_lod(4_999.0, THRESHOLDS), CloudLod::Mid);
        assert_eq!(select_lod(5_000.0, THRESHOLDS), CloudLod::Far);
        assert_eq!(select_lod(19_999.0, THRESHOLDS), CloudLod::Far);
        assert_eq!(select_lod(20_000.0, THRESHOLDS), CloudLod::Imposter);
        assert_eq!(select_lod(1.0e9, THRESHOLDS), CloudLod::Imposter);
    }

    #[test]
    fn extreme_distances_do_not_panic() {
        assert_eq!(select_lod(-100.0, THRESHOLDS), CloudLod::Near);
        assert_eq!(select_lod(f32::INFINITY, THRESHOLDS), CloudLod::Imposter);
    }

    #[test]
    fn per_bucket_workload_is_monotone_in_distance() {
        // Walking finest -> coarsest, steps only decrease, step length only
        // increases, and resolution scale only decreases.
        let mut prev = CloudLod::Near.settings();
        for lod in CloudLod::ALL.iter().skip(1).copied() {
            let cur = lod.settings();
            assert!(
                cur.raymarch_steps < prev.raymarch_steps,
                "steps must decrease at {lod:?}"
            );
            assert!(
                cur.step_length > prev.step_length,
                "step length must increase at {lod:?}"
            );
            assert!(
                cur.resolution_scale < prev.resolution_scale,
                "resolution scale must decrease at {lod:?}"
            );
            assert!(cur.resolution_scale > 0.0 && cur.resolution_scale <= 1.0);
            prev = cur;
        }
    }

    #[test]
    fn increasing_distance_never_increases_work() {
        // Sweeping distance forward, the resolved settings never gain work.
        let mut prev = resolve_lod(0.0, THRESHOLDS).1;
        let mut d = 0.0;
        while d <= 30_000.0 {
            let cur = resolve_lod(d, THRESHOLDS).1;
            assert!(cur.raymarch_steps <= prev.raymarch_steps);
            assert!(cur.step_length >= prev.step_length);
            assert!(cur.resolution_scale <= prev.resolution_scale);
            prev = cur;
            d += 250.0;
        }
    }

    #[test]
    fn imposter_fade_is_saturated_and_monotone() {
        let start = 15_000.0;
        let end = 20_000.0;
        assert_eq!(imposter_fade(0.0, start, end), 0.0);
        assert_eq!(imposter_fade(14_000.0, start, end), 0.0);
        assert_eq!(imposter_fade(25_000.0, start, end), 1.0);
        let mut prev = imposter_fade(0.0, start, end);
        let mut d = 0.0;
        while d <= 25_000.0 {
            let w = imposter_fade(d, start, end);
            assert!((0.0..=1.0).contains(&w), "weight out of range at d={d}");
            assert!(w >= prev - 1e-6, "fade must be non-decreasing at d={d}");
            prev = w;
            d += 100.0;
        }
    }

    #[test]
    fn imposter_fade_collapsed_edges_is_a_hard_step() {
        // start == end must not divide by zero; it becomes a hard step.
        assert_eq!(imposter_fade(9.0, 10.0, 10.0), 0.0);
        assert_eq!(imposter_fade(11.0, 10.0, 10.0), 1.0);
    }

    #[test]
    fn binning_routes_and_preserves_order() {
        // layer 2 -> Near, layer 0 -> Far, layer 1 -> Imposter, layer 3 -> Near.
        let distances = [6_000.0, 25_000.0, 500.0, 100.0];
        let layers = [2, 0, 1, 3];
        let bins = bin_by_distance(&layers, &distances, THRESHOLDS);
        assert_eq!(bins.total(), 4);
        assert_eq!(bins.near, vec![2, 3]);
        assert_eq!(bins.far, vec![0]);
        assert_eq!(bins.imposter, vec![1]);
        assert!(bins.mid.is_empty());
        assert_eq!(bins.count_of(CloudLod::Near), 2);
    }

    #[test]
    fn binning_skips_out_of_range_layers() {
        // layer 7 has no distance entry and must be dropped, not panic.
        let distances = [100.0];
        let layers = [0, 7];
        let bins = bin_by_distance(&layers, &distances, THRESHOLDS);
        assert_eq!(bins.total(), 1);
        assert_eq!(bins.near, vec![0]);
    }

    #[test]
    fn binning_is_deterministic() {
        let distances = [100.0, 6_000.0, 25_000.0];
        let layers = [0, 1, 2, 0];
        let a = bin_by_distance(&layers, &distances, THRESHOLDS);
        let b = bin_by_distance(&layers, &distances, THRESHOLDS);
        assert_eq!(a, b);
    }

    #[test]
    fn empty_input_is_empty_plan() {
        let bins = bin_by_distance(&[], &[], THRESHOLDS);
        assert!(bins.is_empty());
        assert_eq!(bins.total(), 0);
    }
}
