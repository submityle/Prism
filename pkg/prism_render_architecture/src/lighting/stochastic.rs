//! Stochastic per-cluster light sampling for massive light counts (`MegaLights`).
//!
//! Clustered culling ([`super::culling`]) assigns overlapping lights to each
//! cluster but hard-caps the list at `max_per_cluster`, *dropping* every light
//! past the cap and only recording an overflow count. With hundreds or
//! thousands of overlapping lights that dropping is visible: lights pop in and
//! out as the cap boundary shifts, and whole groups of lights never illuminate
//! the surface at all.
//!
//! MegaLights-style shading (borrowing the *form* of UE5 `MegaLights` — bounded
//! per-tile stochastic light evaluation — without reusing any of its code)
//! fixes this by treating per-cluster light selection as **Monte Carlo
//! importance sampling** instead of truncation: each cluster draws a fixed
//! budget of light *samples* with probability proportional to each light's
//! estimated importance, and carries an unbiased `1/pdf` weight so the shading
//! sum is correct *in expectation* no matter how many lights overlap. Temporal
//! and spatial denoising (owned elsewhere) then resolves the residual variance.
//!
//! The estimator is unbiased: with per-draw selection probability
//! `p_i = importance_i / S` (where `S` is the total importance of the
//! overlapping lights) and budget `K`, a light drawn in a sample carries weight
//! `w_i = S / (K · importance_i) = 1 / (K · p_i)`. For any per-light shading
//! contribution `f_i`, the tile estimate `Σ_samples f · w` has expectation
//! `Σ_i p_i · (f_i / p_i) = Σ_i f_i`, i.e. the exact many-light sum.
//!
//! When the number of overlapping lights does not exceed the budget, sampling
//! is pointless: the tile falls back to an **exact** list (every overlapping
//! light once, weight `1`), so small scenes pay no variance. Selection is fully
//! deterministic given a seed (a pure integer hash, no RNG state, no `std`), so
//! results are reproducible frame-to-frame and bit-exact in golden tests and a
//! future GPU twin.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use super::culling::{ClusterBounds, LightVolume};

/// A light paired with the scalar power used to weight its importance.
///
/// `power` is a radiometric magnitude (e.g. luminous/radiant intensity) in any
/// consistent unit; only ratios between lights matter for sampling. The spatial
/// extent and influence range come from the embedded [`LightVolume`].
#[derive(Clone, Copy, Debug)]
pub struct MegaLight {
    /// View-space bounding sphere and light handle.
    pub volume: LightVolume,
    /// Scalar power driving sampling importance (larger = sampled more often).
    pub power: f32,
}

/// One selected light with its unbiased Monte Carlo weight (`1 / (K · pdf)`).
///
/// Multiply a light's shading contribution by [`LightSample::weight`] and sum
/// over a tile's samples to get an unbiased estimate of the full many-light
/// sum. `light_index` points into the `lights` slice passed to selection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightSample {
    /// Index into the input `lights` slice.
    pub light_index: u32,
    /// Unbiased weight for this light's contribution in the tile estimate.
    pub weight: f32,
}

/// The stochastically chosen light set for a single cluster/tile.
///
/// Samples are deduplicated by light (a with-replacement draw that picks the
/// same light twice merges into one entry with summed weight — identical
/// estimator, fewer shading evaluations) and ordered by `light_index` for
/// deterministic iteration.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StochasticTile {
    /// Selected lights with unbiased weights, ordered by `light_index`.
    pub samples: Vec<LightSample>,
    /// Sum of importances over the overlapping lights (`S`); `0` when none.
    pub total_importance: f32,
    /// Number of lights whose influence sphere overlapped the cluster.
    pub overlap_count: u32,
    /// `true` when every overlapping light is included exactly (no sampling,
    /// zero variance), i.e. `overlap_count <= budget`.
    pub exact: bool,
}

impl StochasticTile {
    /// Number of distinct selected lights.
    #[must_use]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether no light was selected (no overlap or zero total importance).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Unbiased estimate of `Σ_i contribution(light_i)` over all overlapping
    /// lights, using this tile's samples and weights.
    ///
    /// `contribution` is called with the input light index and returns that
    /// light's shading contribution at the shaded point. In expectation (over
    /// the seed) the result equals evaluating `contribution` for *every*
    /// overlapping light and summing — at a fixed cost of [`StochasticTile::len`]
    /// evaluations rather than [`StochasticTile::overlap_count`].
    #[must_use]
    pub fn estimate<F: FnMut(u32) -> f32>(&self, mut contribution: F) -> f32 {
        let mut sum = 0.0_f32;
        for sample in &self.samples {
            sum += contribution(sample.light_index) * sample.weight;
        }
        sum
    }
}

/// Squared distance from a point to an axis-aligned box (per-axis clamped).
fn distance_sq_point_aabb(point: [f32; 3], min: [f32; 3], max: [f32; 3]) -> f32 {
    let mut total = 0.0_f32;
    let mut axis = 0;
    while axis < 3 {
        let p = point[axis];
        if p < min[axis] {
            let d = min[axis] - p;
            total += d * d;
        } else if p > max[axis] {
            let d = p - max[axis];
            total += d * d;
        }
        axis += 1;
    }
    total
}

/// Floor on squared distance so a light centered inside a cluster gets a large
/// but finite importance instead of dividing by zero.
const MIN_DISTANCE_SQ: f32 = 1.0e-4;

/// Importance of `light` for a cluster: `0` when the light's sphere does not
/// reach the cluster, else `power / max(d², ε)` with `d` the nearest distance
/// from the cluster box to the light center.
///
/// This is a conservative inverse-square falloff weight; it does not need to be
/// the exact shading contribution for the estimate to stay unbiased — only
/// nonzero wherever the true contribution is nonzero (which the overlap test
/// guarantees) — but a closer match to the real falloff lowers variance.
#[must_use]
pub fn cluster_light_importance(cluster: ClusterBounds, light: MegaLight) -> f32 {
    let d2 = distance_sq_point_aabb(light.volume.center, cluster.min, cluster.max);
    if d2 > light.volume.radius * light.volume.radius {
        return 0.0;
    }
    let power = if light.power > 0.0 { light.power } else { 0.0 };
    power / d2.max(MIN_DISTANCE_SQ)
}

/// `splitmix64` finalizer — a strong integer hash with no retained state.
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Deterministic hash of three mixing inputs into a 64-bit value.
fn hash3(a: u64, b: u64, c: u64) -> u64 {
    splitmix64(a ^ splitmix64(b ^ splitmix64(c)))
}

/// Maps a hashed `u64` to a uniform `f32` in `[0, 1)` using 24 mantissa bits.
fn unit_f32(bits: u64) -> f32 {
    // Top 24 bits give an integer in [0, 2^24); scale into [0, 1).
    ((bits >> 40) as u32) as f32 * (1.0 / 16_777_216.0)
}

/// Selects a bounded, importance-weighted light set for one cluster.
///
/// `budget` is the maximum number of light *draws* (`K`). `seed` and
/// `cluster_id` make the draw deterministic and decorrelated across clusters.
///
/// - No overlapping lights or zero total importance → an empty tile.
/// - `overlap_count <= budget` → the exact list (each overlapping light once,
///   weight `1`), with [`StochasticTile::exact`] set.
/// - otherwise → `budget` draws with replacement, probability proportional to
///   [`cluster_light_importance`], deduplicated with summed unbiased weights.
#[must_use]
pub fn select_tile_lights(
    cluster: ClusterBounds,
    lights: &[MegaLight],
    budget: u32,
    seed: u64,
    cluster_id: u32,
) -> StochasticTile {
    // Gather overlapping lights with their importances and a running prefix sum.
    let mut indices: Vec<u32> = Vec::new();
    let mut prefix: Vec<f32> = Vec::new();
    let mut total = 0.0_f32;
    for (i, light) in lights.iter().enumerate() {
        let imp = cluster_light_importance(cluster, *light);
        if imp > 0.0 {
            total += imp;
            indices.push(i as u32);
            prefix.push(total);
        }
    }

    let overlap_count = indices.len() as u32;
    let mut tile = StochasticTile {
        samples: Vec::new(),
        total_importance: total,
        overlap_count,
        exact: false,
    };

    if overlap_count == 0 || total <= 0.0 || budget == 0 {
        return tile;
    }

    // Exact fallback: when the budget covers every overlapping light, include
    // them all with unit weight — zero variance, no RNG.
    if overlap_count <= budget {
        tile.samples = indices
            .iter()
            .map(|&light_index| LightSample {
                light_index,
                weight: 1.0,
            })
            .collect();
        tile.exact = true;
        return tile;
    }

    // Stochastic path: `budget` draws, probability ∝ importance, weight
    // S / (K · importance_i) per draw. Merge duplicate draws by summing weight.
    let k = budget as f32;
    let mut merged: BTreeMap<u32, f32> = BTreeMap::new();
    let mut draw = 0u32;
    while draw < budget {
        let u = unit_f32(hash3(seed, u64::from(cluster_id), u64::from(draw)));
        let target = u * total;
        // First prefix strictly greater than `target` is the chosen light.
        let slot = lower_bound(&prefix, target);
        let slot = slot.min(indices.len() - 1);
        let light_index = indices[slot];
        let importance = prefix[slot] - if slot == 0 { 0.0 } else { prefix[slot - 1] };
        // importance > 0 by construction (only overlapping lights are listed).
        let weight = total / (k * importance);
        *merged.entry(light_index).or_insert(0.0) += weight;
        draw += 1;
    }

    tile.samples = merged
        .into_iter()
        .map(|(light_index, weight)| LightSample {
            light_index,
            weight,
        })
        .collect();
    tile
}

/// Index of the first element in `prefix` strictly greater than `target`.
///
/// `prefix` is non-decreasing (a cumulative importance sum). Returns
/// `prefix.len()` when `target` is >= the last element (callers clamp).
fn lower_bound(prefix: &[f32], target: f32) -> usize {
    let mut lo = 0usize;
    let mut hi = prefix.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if prefix[mid] > target {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo
}

/// Selects stochastic light sets for a batch of clusters.
///
/// Each cluster `c` is seeded with its own index so draws are decorrelated;
/// results are returned in cluster order. See [`select_tile_lights`].
#[must_use]
pub fn select_all_tiles(
    clusters: &[ClusterBounds],
    lights: &[MegaLight],
    budget: u32,
    seed: u64,
) -> Vec<StochasticTile> {
    clusters
        .iter()
        .enumerate()
        .map(|(c, &cluster)| select_tile_lights(cluster, lights, budget, seed, c as u32))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lighting::LightHandle;

    fn light(index: u32, center: [f32; 3], radius: f32, power: f32) -> MegaLight {
        MegaLight {
            volume: LightVolume {
                light: LightHandle(index),
                center,
                radius,
            },
            power,
        }
    }

    fn unit_cluster(origin: [f32; 3]) -> ClusterBounds {
        ClusterBounds {
            min: origin,
            max: [origin[0] + 1.0, origin[1] + 1.0, origin[2] + 1.0],
        }
    }

    #[test]
    fn non_overlapping_lights_yield_empty_tile() {
        let cluster = unit_cluster([0.0, 0.0, 0.0]);
        let lights = [light(0, [100.0, 0.0, 0.0], 1.0, 10.0)];
        let tile = select_tile_lights(cluster, &lights, 4, 0, 0);
        assert!(tile.is_empty());
        assert_eq!(tile.overlap_count, 0);
        assert_eq!(tile.total_importance, 0.0);
    }

    #[test]
    fn importance_zero_beyond_radius() {
        let cluster = unit_cluster([0.0, 0.0, 0.0]);
        // Center 3 units from box max (gap 2), radius 1.9 falls short.
        let short = light(0, [3.0, 0.5, 0.5], 1.9, 5.0);
        assert_eq!(cluster_light_importance(cluster, short), 0.0);
        // Radius 2.0 reaches.
        let reach = light(0, [3.0, 0.5, 0.5], 2.0, 5.0);
        assert!(cluster_light_importance(cluster, reach) > 0.0);
    }

    #[test]
    fn importance_decreases_with_distance() {
        let cluster = unit_cluster([0.0, 0.0, 0.0]);
        let near = light(0, [2.0, 0.5, 0.5], 10.0, 1.0);
        let far = light(0, [5.0, 0.5, 0.5], 10.0, 1.0);
        assert!(cluster_light_importance(cluster, near) > cluster_light_importance(cluster, far));
    }

    #[test]
    fn exact_path_when_under_budget() {
        let cluster = unit_cluster([0.0, 0.0, 0.0]);
        let lights = [
            light(0, [0.5, 0.5, 0.5], 2.0, 1.0),
            light(1, [0.5, 0.5, 0.5], 2.0, 3.0),
        ];
        let tile = select_tile_lights(cluster, &lights, 4, 0, 0);
        assert!(tile.exact);
        assert_eq!(tile.len(), 2);
        // Each included once with unit weight → estimate is the exact sum.
        for s in &tile.samples {
            assert_eq!(s.weight, 1.0);
        }
        let est = tile.estimate(|i| if i == 0 { 2.0 } else { 5.0 });
        assert_eq!(est, 7.0);
    }

    #[test]
    fn stochastic_path_respects_budget() {
        let cluster = unit_cluster([0.0, 0.0, 0.0]);
        let lights: Vec<MegaLight> = (0..32)
            .map(|i| light(i, [0.5, 0.5, 0.5], 2.0, 1.0 + i as f32))
            .collect();
        let budget = 4;
        let tile = select_tile_lights(cluster, &lights, budget, 12345, 7);
        assert!(!tile.exact);
        assert_eq!(tile.overlap_count, 32);
        // Distinct lights never exceed the draw budget.
        assert!(tile.len() <= budget as usize);
        assert!(!tile.is_empty());
    }

    #[test]
    fn selection_is_deterministic() {
        let cluster = unit_cluster([0.0, 0.0, 0.0]);
        let lights: Vec<MegaLight> = (0..64)
            .map(|i| light(i, [0.5, 0.5, 0.5], 2.0, 1.0 + (i % 7) as f32))
            .collect();
        let a = select_tile_lights(cluster, &lights, 6, 999, 3);
        let b = select_tile_lights(cluster, &lights, 6, 999, 3);
        assert_eq!(a, b);
    }

    #[test]
    fn different_clusters_decorrelate() {
        let cluster = unit_cluster([0.0, 0.0, 0.0]);
        let lights: Vec<MegaLight> = (0..64)
            .map(|i| light(i, [0.5, 0.5, 0.5], 2.0, 1.0))
            .collect();
        let a = select_tile_lights(cluster, &lights, 4, 42, 0);
        let b = select_tile_lights(cluster, &lights, 4, 42, 1);
        // Equal-power lights with different cluster ids should (almost surely)
        // pick different sample sets.
        assert_ne!(a.samples, b.samples);
    }

    #[test]
    fn estimator_is_unbiased_over_seeds() {
        // Many overlapping lights with varied power and arbitrary per-light
        // contributions; averaging the budgeted estimate over many seeds must
        // converge to the exact many-light sum.
        let cluster = unit_cluster([0.0, 0.0, 0.0]);
        let count = 24u32;
        let lights: Vec<MegaLight> = (0..count)
            // Vary distance and power so importances are non-uniform.
            .map(|i| {
                let x = 0.5 + (i % 5) as f32 * 0.1;
                light(i, [x, 0.5, 0.5], 3.0, 1.0 + (i % 6) as f32)
            })
            .collect();
        // Arbitrary ground-truth contribution per light.
        let contribution = |i: u32| -> f32 { 0.5 + (i as f32) * 0.37 };
        let exact: f32 = (0..count).map(contribution).sum();

        let budget = 4;
        let seeds = 60_000u64;
        let mut acc = 0.0f64;
        for s in 0..seeds {
            let tile = select_tile_lights(cluster, &lights, budget, s.wrapping_mul(2654435761), 0);
            assert!(!tile.exact, "scene must exercise the sampling path");
            acc += f64::from(tile.estimate(contribution));
        }
        let mean = (acc / seeds as f64) as f32;
        let rel_err = (mean - exact).abs() / exact;
        assert!(
            rel_err < 0.01,
            "mean {mean} vs exact {exact} (rel err {rel_err})"
        );
    }

    #[test]
    fn weights_sum_matches_total_over_importance_in_expectation() {
        // With f_i = 1 for all lights, the estimate equals Σ weights, whose
        // expectation is the light count. Check convergence.
        let cluster = unit_cluster([0.0, 0.0, 0.0]);
        let count = 20u32;
        let lights: Vec<MegaLight> = (0..count)
            .map(|i| light(i, [0.5, 0.5, 0.5], 2.0, 1.0 + (i % 4) as f32))
            .collect();
        let budget = 3;
        let seeds = 40_000u64;
        let mut acc = 0.0f64;
        for s in 0..seeds {
            let tile = select_tile_lights(cluster, &lights, budget, s.wrapping_mul(40503), 0);
            acc += f64::from(tile.estimate(|_| 1.0));
        }
        let mean = (acc / seeds as f64) as f32;
        let rel_err = (mean - count as f32).abs() / count as f32;
        assert!(rel_err < 0.01, "mean {mean} vs {count} (rel err {rel_err})");
    }

    #[test]
    fn select_all_tiles_matches_per_tile() {
        let clusters = [unit_cluster([0.0, 0.0, 0.0]), unit_cluster([2.0, 0.0, 0.0])];
        let lights: Vec<MegaLight> = (0..40)
            .map(|i| light(i, [1.0, 0.5, 0.5], 3.0, 1.0 + i as f32))
            .collect();
        let all = select_all_tiles(&clusters, &lights, 5, 77);
        assert_eq!(all.len(), 2);
        assert_eq!(all[0], select_tile_lights(clusters[0], &lights, 5, 77, 0));
        assert_eq!(all[1], select_tile_lights(clusters[1], &lights, 5, 77, 1));
    }

    #[test]
    fn zero_budget_yields_empty() {
        let cluster = unit_cluster([0.0, 0.0, 0.0]);
        let lights = [light(0, [0.5, 0.5, 0.5], 2.0, 1.0)];
        let tile = select_tile_lights(cluster, &lights, 0, 0, 0);
        assert!(tile.is_empty());
    }

    #[test]
    fn lower_bound_finds_first_greater() {
        let prefix = [1.0, 3.0, 6.0, 10.0];
        assert_eq!(lower_bound(&prefix, 0.0), 0);
        assert_eq!(lower_bound(&prefix, 1.0), 1); // not strictly greater at 0
        assert_eq!(lower_bound(&prefix, 2.0), 1);
        assert_eq!(lower_bound(&prefix, 6.0), 3);
        assert_eq!(lower_bound(&prefix, 9.9), 3);
        assert_eq!(lower_bound(&prefix, 10.0), 4);
    }
}
