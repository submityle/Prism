//! Octahedral depth-moment sky-visibility occlusion — CPU golden.
//!
//! Adaptive probe volumes bake, alongside diffuse irradiance, a measure of how
//! much open sky each probe can see.  This drives a soft ambient-occlusion-like
//! modulation of the sky/ambient term so points tucked under geometry are not
//! over-lit by the environment.  As in DDGI depth filtering, the directional
//! occluder field is stored as **two depth moments** (`mean`, `mean_sq`) per
//! octahedral texel; the Chebyshev (Cantelli) inequality then turns those
//! moments into a soft per-direction sky visibility, and a cosine-weighted
//! hemisphere integral collapses them into a single scalar sky-access factor.
//!
//! This file is the backend-neutral reference:
//!
//! * [`SkyOcclusion`] packs the per-direction occluder moments into an
//!   octahedral map, reusing the shared
//!   [`dir_to_oct`](crate::gi::world_space::octahedral::dir_to_oct) mapping so a
//!   direction round-trips to the same texel the GPU twin addresses.
//! * [`SkyOcclusion::update`] folds a freshly sampled occluder distance into a
//!   texel with an exponential moving average (the RTXGI hysteresis blend).
//! * [`SkyOcclusion::directional_visibility`] evaluates the Chebyshev weight for
//!   one direction, reusing the sibling
//!   [`chebyshev_weight`](crate::gi::world_space::visibility::chebyshev_weight).
//! * [`SkyOcclusion::sky_visibility`] integrates directional visibility over the
//!   cosine-weighted hemisphere about a normal into a `[0, 1]` scalar.
//!
//! # Conventions
//! * Depths are world-space distances from the probe centre to the nearest
//!   occluder along a direction, in the same units as the sky query distance.
//! * A direction with no occluder stores a very large mean (fully open sky);
//!   a zeroed texel reads fully occluded, the conservative default before any
//!   [`update`](SkyOcclusion::update).
//! * The query compares a large `sky_distance` against the moments: `distance
//!   <= mean` short-circuits to fully visible, matching the DDGI depth test.
//! * Every value is clamped to stay finite and in range; variance is clamped
//!   non-negative so `f32` round-off can never inject a `NaN`.
//! * Every item is a deterministic pure function: no RNG, no I/O, no GPU, no
//!   `unsafe`; the only allocation is the moment buffer owned by
//!   [`SkyOcclusion`].

use alloc::vec::Vec;
use bevy_math::{ops, Vec3};

use crate::gi::world_space::octahedral::dir_to_oct;
use crate::gi::world_space::visibility::chebyshev_weight;

/// A per-probe octahedral map of two-moment occluder depths for sky visibility.
///
/// The map is a square `resolution x resolution` grid of texels in row-major
/// order; each texel stores `[mean, mean_sq]` occluder depth as described in the
/// module docs.  Directions are addressed through the shared octahedral
/// parameterisation.
#[derive(Clone, Debug, PartialEq)]
pub struct SkyOcclusion {
    /// Side length of the square octahedral grid, always `>= 1`.
    resolution: usize,
    /// Row-major `resolution * resolution` texels of `[mean, mean_sq]`.
    moments: Vec<[f32; 2]>,
}

/// Distance treated as "sky" when integrating open-sky visibility.
const SKY_DISTANCE: f32 = 1.0e6;

impl SkyOcclusion {
    /// Creates a zero-initialised (fully occluded) map of the given resolution.
    ///
    /// `resolution` is clamped to at least `1`.  A zeroed texel reads
    /// `mean = mean_sq = 0`, i.e. fully occluded for any positive query
    /// distance until populated by [`update`](Self::update).
    #[inline]
    pub fn new(resolution: usize) -> Self {
        Self::filled(resolution, 0.0, 0.0)
    }

    /// Creates a map whose every texel holds the same moments.
    #[inline]
    pub fn filled(resolution: usize, mean: f32, mean_sq: f32) -> Self {
        let res = resolution.max(1);
        Self {
            resolution: res,
            moments: alloc::vec![[mean, mean_sq]; res * res],
        }
    }

    /// Creates a fully-open-sky map (large mean) of the given resolution.
    ///
    /// Every direction reads as unoccluded until an [`update`](Self::update)
    /// introduces a nearer occluder.
    #[inline]
    pub fn open(resolution: usize) -> Self {
        Self::filled(resolution, SKY_DISTANCE, SKY_DISTANCE * SKY_DISTANCE)
    }

    /// The side length of the square octahedral grid (`>= 1`).
    #[inline]
    pub fn resolution(&self) -> usize {
        self.resolution
    }

    /// The number of texels, `resolution * resolution`.
    #[inline]
    pub fn len(&self) -> usize {
        self.moments.len()
    }

    /// Whether the map holds no texels (never true: resolution is `>= 1`).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.moments.is_empty()
    }

    /// Maps a direction to its row-major texel index via octahedral encoding.
    ///
    /// A degenerate (zero) direction maps to the centre texel; the result is
    /// always a valid in-range index.
    #[inline]
    pub fn index_for_dir(&self, dir: Vec3) -> usize {
        let uv = dir_to_oct(dir);
        let res = self.resolution;
        let max = res - 1;
        let fx = (uv.x.clamp(0.0, 1.0) * res as f32).floor();
        let fy = (uv.y.clamp(0.0, 1.0) * res as f32).floor();
        let ix = (fx as usize).min(max);
        let iy = (fy as usize).min(max);
        iy * res + ix
    }

    /// Reads the raw moments at a texel index, or `[0, 0]` if out of range.
    #[inline]
    pub fn moments_at(&self, index: usize) -> [f32; 2] {
        self.moments.get(index).copied().unwrap_or([0.0, 0.0])
    }

    /// Samples the moments along a direction (nearest texel).
    #[inline]
    pub fn sample(&self, dir: Vec3) -> [f32; 2] {
        self.moments_at(self.index_for_dir(dir))
    }

    /// Folds a freshly sampled occluder distance into a direction's texel with
    /// an exponential moving average blend factor `alpha` in `[0, 1]`.
    ///
    /// `distance` is clamped non-negative; `alpha` is clamped to `[0, 1]`.
    /// `alpha = 1` overwrites, `alpha = 0` is a no-op.
    #[inline]
    pub fn update(&mut self, dir: Vec3, distance: f32, alpha: f32) {
        let a = alpha.clamp(0.0, 1.0);
        let d = distance.max(0.0);
        let idx = self.index_for_dir(dir);
        if let Some(texel) = self.moments.get_mut(idx) {
            let d2 = d * d;
            texel[0] += (d - texel[0]) * a;
            texel[1] += (d2 - texel[1]) * a;
        }
    }

    /// The Chebyshev sky visibility along one direction for a query distance.
    ///
    /// `query_distance` is clamped non-negative; the result is in `[0, 1]`.
    #[inline]
    pub fn directional_visibility(&self, dir: Vec3, query_distance: f32) -> f32 {
        let [mean, mean_sq] = self.sample(dir);
        chebyshev_weight(mean, mean_sq, query_distance.max(0.0))
    }

    /// Integrates sky visibility over the cosine-weighted hemisphere about
    /// `normal`, returning a `[0, 1]` scalar sky-access factor.
    ///
    /// Uses a deterministic `samples`-point Fibonacci sphere lattice, keeping
    /// only directions in the upper hemisphere (`dot(dir, normal) > 0`) and
    /// weighting each by its clamped cosine.  Each kept direction's visibility
    /// is the Chebyshev test of [`SKY_DISTANCE`] against the stored moments, so
    /// an open-sky direction reads `1` and an occluded one reads `0`.
    ///
    /// A degenerate normal falls back to `+Y`; when no sample lands in the
    /// hemisphere (impossible for `samples >= 1`) the function returns `0`.
    #[inline]
    pub fn sky_visibility(&self, normal: Vec3, samples: usize) -> f32 {
        let n = normalize_or_up(normal);
        let count = samples.max(1);
        let golden = core::f32::consts::PI * (3.0 - (5.0f32).sqrt());
        let mut num = 0.0f32;
        let mut den = 0.0f32;
        for i in 0..count {
            let dir = fibonacci_dir(i, count, golden);
            let cos = dir.dot(n);
            if cos <= 0.0 {
                continue;
            }
            let vis = chebyshev_weight(
                self.sample(dir)[0],
                self.sample(dir)[1],
                SKY_DISTANCE,
            );
            num += vis * cos;
            den += cos;
        }
        if den > f32::MIN_POSITIVE {
            (num / den).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

/// Normalises `dir`, falling back to `+Y` for a degenerate (zero) input.
#[inline]
fn normalize_or_up(dir: Vec3) -> Vec3 {
    let len_sq = dir.length_squared();
    if len_sq > f32::MIN_POSITIVE {
        dir * len_sq.sqrt().recip()
    } else {
        Vec3::Y
    }
}

/// Deterministic Fibonacci-lattice unit direction, index `i` of `n`.
#[inline]
fn fibonacci_dir(i: usize, n: usize, golden: f32) -> Vec3 {
    let z = 1.0 - (2.0 * i as f32 + 1.0) / n as f32;
    let r = (1.0 - z * z).max(0.0).sqrt();
    let phi = golden * i as f32;
    Vec3::new(r * ops::cos(phi), r * ops::sin(phi), z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction_clamps_resolution() {
        let m = SkyOcclusion::new(0);
        assert_eq!(m.resolution(), 1);
        assert_eq!(m.len(), 1);
        assert!(!m.is_empty());
        assert_eq!(m.index_for_dir(Vec3::new(1.0, -2.0, 0.3)), 0);
    }

    #[test]
    fn octahedral_indexing_round_trips_in_range() {
        let res = 16;
        let m = SkyOcclusion::new(res);
        assert_eq!(m.len(), res * res);
        let n = 24;
        for i in 0..=n {
            for j in 0..=n {
                let theta = core::f32::consts::PI * (i as f32 / n as f32);
                let phi = core::f32::consts::TAU * (j as f32 / n as f32);
                let dir = Vec3::new(
                    ops::sin(theta) * ops::cos(phi),
                    ops::sin(theta) * ops::sin(phi),
                    ops::cos(theta),
                );
                let idx = m.index_for_dir(dir);
                assert!(idx < m.len(), "idx {idx} >= {}", m.len());
                let s = m.sample(dir);
                assert!(s[0].is_finite() && s[1].is_finite());
            }
        }
    }

    #[test]
    fn filled_samples_back_to_fill_value() {
        let m = SkyOcclusion::filled(8, 3.0, 10.0);
        for dir in [
            Vec3::X,
            Vec3::NEG_X,
            Vec3::Y,
            Vec3::NEG_Y,
            Vec3::Z,
            Vec3::NEG_Z,
            Vec3::new(0.4, 0.5, -0.7),
        ] {
            let [mean, mean_sq] = m.sample(dir);
            assert!((mean - 3.0).abs() < 1e-5, "mean {mean} for {dir:?}");
            assert!((mean_sq - 10.0).abs() < 1e-5, "mean_sq {mean_sq}");
        }
    }

    #[test]
    fn update_applies_exponential_moving_average() {
        let dir = Vec3::new(-0.2, 0.5, 0.8);
        let mut m = SkyOcclusion::new(4);
        m.update(dir, 4.0, 1.0);
        let idx = m.index_for_dir(dir);
        let t = m.moments_at(idx);
        assert!((t[0] - 4.0).abs() < 1e-6 && (t[1] - 16.0).abs() < 1e-6, "{t:?}");

        m.update(dir, 6.0, 0.5);
        let t = m.moments_at(idx);
        assert!((t[0] - 5.0).abs() < 1e-6 && (t[1] - 26.0).abs() < 1e-6, "{t:?}");

        m.update(dir, 100.0, 0.0);
        let t = m.moments_at(idx);
        assert!((t[0] - 5.0).abs() < 1e-6 && (t[1] - 26.0).abs() < 1e-6, "{t:?}");
    }

    #[test]
    fn negative_distance_is_clamped() {
        let dir = Vec3::Z;
        let mut m = SkyOcclusion::new(2);
        m.update(dir, -3.0, 1.0);
        assert_eq!(m.moments_at(m.index_for_dir(dir)), [0.0, 0.0]);
    }

    #[test]
    fn open_sky_is_fully_visible() {
        let m = SkyOcclusion::open(8);
        for dir in [Vec3::Y, Vec3::X, Vec3::new(0.3, 0.8, 0.5)] {
            assert!((m.directional_visibility(dir, 50.0) - 1.0).abs() < 1e-6);
        }
        let v = m.sky_visibility(Vec3::Y, 256);
        assert!((v - 1.0).abs() < 1e-5, "sky visibility {v}");
    }

    #[test]
    fn zero_map_is_fully_occluded() {
        let m = SkyOcclusion::new(8);
        // Positive query distance against zero moments => occluded.
        assert_eq!(m.directional_visibility(Vec3::Y, 10.0), 0.0);
        let v = m.sky_visibility(Vec3::Y, 256);
        assert!(v.abs() < 1e-6, "sky visibility {v}");
    }

    #[test]
    fn chebyshev_boundary_short_circuits() {
        // mean large => query below mean => fully visible.
        let m = SkyOcclusion::filled(4, 100.0, 100.0 * 100.0);
        assert_eq!(m.directional_visibility(Vec3::Z, 50.0), 1.0);
        // query above mean with zero variance => hard occlusion.
        let m2 = SkyOcclusion::filled(4, 5.0, 25.0);
        assert_eq!(m2.directional_visibility(Vec3::Z, 6.0), 0.0);
        assert_eq!(m2.directional_visibility(Vec3::Z, 4.0), 1.0);
    }

    #[test]
    fn partial_occlusion_is_between_zero_and_one() {
        // Open the +Y hemisphere, occlude the rest, then integrate about +Y.
        let mut m = SkyOcclusion::new(32);
        let n = 48;
        for i in 0..n {
            for j in 0..n {
                let theta = core::f32::consts::PI * (i as f32 + 0.5) / n as f32;
                let phi = core::f32::consts::TAU * (j as f32 + 0.5) / n as f32;
                let dir = Vec3::new(
                    ops::sin(theta) * ops::cos(phi),
                    ops::cos(theta),
                    ops::sin(theta) * ops::sin(phi),
                );
                // Open sky above the horizon, blocked below.
                if dir.y > 0.0 {
                    m.update(dir, SKY_DISTANCE, 1.0);
                } else {
                    m.update(dir, 0.1, 1.0);
                }
            }
        }
        let up = m.sky_visibility(Vec3::Y, 512);
        let down = m.sky_visibility(Vec3::NEG_Y, 512);
        assert!(up > 0.7, "up visibility {up}");
        assert!(down < 0.3, "down visibility {down}");
    }

    #[test]
    fn results_are_deterministic() {
        let build = || {
            let mut m = SkyOcclusion::new(8);
            m.update(Vec3::new(0.1, 0.2, 0.9), 4.0, 0.75);
            m.update(Vec3::new(-0.7, 0.3, 0.6), 2.0, 0.9);
            m
        };
        let a = build();
        let b = build();
        assert_eq!(a, b);
        assert_eq!(a.sky_visibility(Vec3::Y, 128), b.sky_visibility(Vec3::Y, 128));
    }

    #[test]
    fn stale_index_reads_zero() {
        let m = SkyOcclusion::new(4);
        assert_eq!(m.moments_at(m.len()), [0.0, 0.0]);
        assert_eq!(m.moments_at(usize::MAX), [0.0, 0.0]);
    }
}
