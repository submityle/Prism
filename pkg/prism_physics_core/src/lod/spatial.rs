//! Distance-based spatial level-of-detail.
//!
//! Bodies far from the point the viewer cares about need less solver effort:
//! their errors are small on screen and rarely interacted with. The spatial
//! controller sorts bodies into distance *tiers* and scales each tier's solver
//! iteration budget, so a distant soft body might run a handful of iterations
//! while a nearby one runs the full count. Because the tiers are ordered and the
//! scale changes gradually, quality degrades smoothly rather than popping.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**.
//! Distance-based level-of-detail bucketing is a standard, publicly documented
//! real-time-rendering and simulation technique.

use crate::math::scalar::Real;

/// Buckets bodies into distance tiers and scales their iteration budgets.
///
/// `tier_distances` holds ascending distance thresholds; a body closer than
/// `tier_distances[0]` is in tier `0`, between the first two thresholds is in
/// tier `1`, and so on, with everything beyond the last threshold in the final
/// tier. `iteration_scales` has exactly one more entry than `tier_distances`,
/// giving the iteration-budget multiplier for each tier (typically decreasing
/// with distance).
#[derive(Clone, PartialEq, Debug, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SpatialController {
    /// Ascending distance thresholds separating the tiers.
    tier_distances: Vec<Real>,
    /// Per-tier iteration-budget multipliers; `tier_distances.len() + 1` entries.
    iteration_scales: Vec<Real>,
    /// Floor on the scaled iteration count so no body is starved entirely.
    min_iterations: u32,
}

impl SpatialController {
    /// Creates a controller from ascending `tier_distances` and per-tier
    /// `iteration_scales`.
    ///
    /// The threshold list is sorted defensively. The scale list is truncated or
    /// padded (with its last value, or `1.0` when empty) to exactly
    /// `tier_distances.len() + 1` entries so [`tier_of`](Self::tier_of) always
    /// has a scale.
    #[must_use]
    pub fn new(
        mut tier_distances: Vec<Real>,
        mut iteration_scales: Vec<Real>,
        min_iterations: u32,
    ) -> Self {
        tier_distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        let wanted = tier_distances.len() + 1;
        let fill = iteration_scales.last().copied().unwrap_or(1.0);
        iteration_scales.resize(wanted, fill);
        SpatialController {
            tier_distances,
            iteration_scales,
            min_iterations: min_iterations.max(1),
        }
    }

    /// Number of tiers (`tier_distances.len() + 1`).
    #[must_use]
    pub fn tier_count(&self) -> usize {
        self.tier_distances.len() + 1
    }

    /// The tier index a body at `distance` belongs to.
    #[must_use]
    pub fn tier_of(&self, distance: Real) -> usize {
        let mut tier = 0;
        for &threshold in &self.tier_distances {
            if distance < threshold {
                break;
            }
            tier += 1;
        }
        tier
    }

    /// The iteration-budget multiplier for a body at `distance`.
    #[must_use]
    pub fn scale_at(&self, distance: Real) -> Real {
        let tier = self.tier_of(distance);
        self.iteration_scales
            .get(tier)
            .copied()
            .unwrap_or(1.0)
            .max(0.0)
    }

    /// Scales a base iteration budget for a body at `distance`, never dropping
    /// below the configured minimum.
    #[must_use]
    pub fn scaled_iterations(&self, base_iterations: u32, distance: Real) -> u32 {
        let scaled = base_iterations as Real * self.scale_at(distance);
        // Round to nearest without a transcendental round call.
        let rounded = (scaled + 0.5) as u32;
        rounded.max(self.min_iterations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn controller() -> SpatialController {
        // Near (<5): full budget; mid (<20): half; far: quarter.
        SpatialController::new(vec![5.0, 20.0], vec![1.0, 0.5, 0.25], 2)
    }

    #[test]
    fn tiers_partition_the_number_line() {
        let c = controller();
        assert_eq!(c.tier_count(), 3);
        assert_eq!(c.tier_of(0.0), 0);
        assert_eq!(c.tier_of(4.9), 0);
        assert_eq!(c.tier_of(5.0), 1);
        assert_eq!(c.tier_of(19.9), 1);
        assert_eq!(c.tier_of(20.0), 2);
        assert_eq!(c.tier_of(1000.0), 2);
    }

    #[test]
    fn iteration_budget_shrinks_with_distance() {
        let c = controller();
        assert_eq!(c.scaled_iterations(16, 1.0), 16);
        assert_eq!(c.scaled_iterations(16, 10.0), 8);
        assert_eq!(c.scaled_iterations(16, 100.0), 4);
    }

    #[test]
    fn minimum_iterations_is_enforced() {
        let c = controller();
        // 2 * 0.25 = 0.5 -> rounds to 1, but floor is 2.
        assert_eq!(c.scaled_iterations(2, 100.0), 2);
    }

    #[test]
    fn unsorted_thresholds_are_sorted() {
        let c = SpatialController::new(vec![20.0, 5.0], vec![1.0, 0.5, 0.25], 1);
        assert_eq!(c.tier_of(4.0), 0);
        assert_eq!(c.tier_of(10.0), 1);
    }

    #[test]
    fn scales_are_padded_to_tier_count() {
        // Fewer scales than tiers: last value repeats.
        let c = SpatialController::new(vec![5.0, 20.0], vec![1.0], 1);
        assert_eq!(c.tier_count(), 3);
        assert!((c.scale_at(100.0) - 1.0).abs() < 1e-6);
    }
}
