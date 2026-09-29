//! Level-of-detail (`LOD`) residency policy and screen-error selection.
//!
//! Each geometry carries several `LOD` levels; higher levels are coarser and
//! cheaper but carry more screen-space error. Two decisions live here:
//!
//! 1. **Selection** — given a screen-error budget, pick the coarsest resident
//!    `LOD` that still stays within budget (the cheapest acceptable level), and
//!    otherwise fall back to the finest resident level available.
//! 2. **Residency** — given a target level, decide which nearby levels should be
//!    kept resident so that small viewpoint changes do not stall on streaming.
//!
//! The physical `GPU` streaming/upload is pending the GPU backend; this module
//! is the deterministic `CPU`-verifiable policy.

use super::{GeometryLodRecord, GeometryRecord};

impl GeometryRecord {
    /// Selects the cheapest resident `LOD` whose screen error is within
    /// `error_budget`.
    ///
    /// Among resident levels with `screen_error <= error_budget`, the coarsest
    /// (highest level) is returned because it is the cheapest that still meets
    /// quality. When none are within budget, the resident level with the
    /// smallest screen error (the finest available) is returned as best effort,
    /// or `None` when nothing is resident.
    #[must_use]
    pub fn select_lod_by_error(&self, error_budget: f32) -> Option<&GeometryLodRecord> {
        let budget = if error_budget.is_nan() {
            0.0
        } else {
            error_budget
        };
        let within_budget = self
            .lods
            .iter()
            .filter(|lod| lod.resident && lod.screen_error <= budget)
            .max_by_key(|lod| lod.level);
        if within_budget.is_some() {
            return within_budget;
        }
        self.lods.iter().filter(|lod| lod.resident).min_by(|a, b| {
            a.screen_error
                .partial_cmp(&b.screen_error)
                .unwrap_or(core::cmp::Ordering::Equal)
        })
    }
}

/// Window of `LOD` levels to keep resident around a target level.
///
/// `finer_margin` levels below the target and `coarser_margin` levels above it
/// are retained so that zooming in or out by a step reuses already-resident
/// data instead of stalling.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct LodResidencyPolicy {
    pub finer_margin: u32,
    pub coarser_margin: u32,
}

impl LodResidencyPolicy {
    /// Builds a policy from explicit finer/coarser margins.
    #[must_use]
    pub const fn new(finer_margin: u32, coarser_margin: u32) -> Self {
        Self {
            finer_margin,
            coarser_margin,
        }
    }

    /// Inclusive `[low, high]` level range that should be resident for `target`,
    /// clamped to `[0, max_level]`.
    #[must_use]
    pub const fn resident_range(self, target: u32, max_level: u32) -> (u32, u32) {
        let low = target.saturating_sub(self.finer_margin);
        let high_unclamped = target.saturating_add(self.coarser_margin);
        let high = if high_unclamped > max_level {
            max_level
        } else {
            high_unclamped
        };
        (low, high)
    }

    /// True when `level` falls inside the resident window for `target`.
    #[must_use]
    pub const fn is_resident_level(self, level: u32, target: u32, max_level: u32) -> bool {
        let (low, high) = self.resident_range(target, max_level);
        level >= low && level <= high
    }
}

/// Plan describing which levels to stream in and which to evict.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ResidencyPlan {
    /// Levels that should be resident but currently are not.
    pub stream_in: Vec<u32>,
    /// Levels that are resident but fall outside the window.
    pub evict: Vec<u32>,
}

impl GeometryRecord {
    /// Highest `LOD` level present in the record, or `0` when empty.
    #[must_use]
    pub fn max_level(&self) -> u32 {
        self.lods.iter().map(|lod| lod.level).max().unwrap_or(0)
    }

    /// Computes the residency plan for a `target` level under `policy`.
    ///
    /// Levels inside the window that are not resident are queued in `stream_in`;
    /// resident levels outside the window are queued in `evict`. Both lists are
    /// ordered by ascending level for deterministic output.
    #[must_use]
    pub fn residency_plan(&self, target: u32, policy: LodResidencyPolicy) -> ResidencyPlan {
        let max_level = self.max_level();
        let mut stream_in = Vec::new();
        let mut evict = Vec::new();
        for lod in &self.lods {
            let wanted = policy.is_resident_level(lod.level, target, max_level);
            if wanted && !lod.resident {
                stream_in.push(lod.level);
            } else if !wanted && lod.resident {
                evict.push(lod.level);
            }
        }
        stream_in.sort_unstable();
        evict.sort_unstable();
        ResidencyPlan { stream_in, evict }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::GeometryPrimitiveKind;

    fn lod(level: u32, screen_error: f32, resident: bool) -> GeometryLodRecord {
        GeometryLodRecord {
            level,
            primitive_kind: GeometryPrimitiveKind::Indexed,
            element_count: 0,
            first_element: 0,
            base_vertex: 0,
            vertex_count: 0,
            screen_error,
            resident,
            fallback: false,
        }
    }

    fn record(lods: Vec<GeometryLodRecord>) -> GeometryRecord {
        GeometryRecord {
            lods,
            ..Default::default()
        }
    }

    #[test]
    fn selects_coarsest_within_budget() {
        let r = record(vec![
            lod(0, 0.1, true),
            lod(1, 0.5, true),
            lod(2, 2.0, true),
        ]);
        // Budget 1.0 admits levels 0 and 1; the coarsest (1) wins.
        assert_eq!(r.select_lod_by_error(1.0).unwrap().level, 1);
        // Budget 5.0 admits all; the coarsest (2) wins.
        assert_eq!(r.select_lod_by_error(5.0).unwrap().level, 2);
    }

    #[test]
    fn falls_back_to_finest_when_over_budget() {
        let r = record(vec![lod(0, 0.4, true), lod(1, 0.9, true)]);
        // Budget below every level: pick the finest (smallest error).
        assert_eq!(r.select_lod_by_error(0.1).unwrap().level, 0);
    }

    #[test]
    fn selection_ignores_non_resident() {
        let r = record(vec![lod(0, 0.1, false), lod(1, 0.2, true)]);
        assert_eq!(r.select_lod_by_error(5.0).unwrap().level, 1);
    }

    #[test]
    fn selection_empty_is_none() {
        let r = record(vec![lod(0, 0.1, false)]);
        assert!(r.select_lod_by_error(5.0).is_none());
    }

    #[test]
    fn nan_budget_is_treated_as_zero() {
        let r = record(vec![lod(0, 0.3, true), lod(1, 0.9, true)]);
        // NaN -> 0 budget -> nothing within budget -> finest fallback.
        assert_eq!(r.select_lod_by_error(f32::NAN).unwrap().level, 0);
    }

    #[test]
    fn resident_range_clamps_to_bounds() {
        let policy = LodResidencyPolicy::new(1, 2);
        assert_eq!(policy.resident_range(0, 5), (0, 2));
        assert_eq!(policy.resident_range(4, 5), (3, 5));
        assert_eq!(policy.resident_range(3, 3), (2, 3));
    }

    #[test]
    fn residency_plan_streams_and_evicts() {
        let policy = LodResidencyPolicy::new(0, 1);
        // Target 1, window [1, 2]. Level 0 resident -> evict; level 2 not
        // resident -> stream in; level 1 resident -> keep.
        let r = record(vec![
            lod(0, 0.1, true),
            lod(1, 0.5, true),
            lod(2, 1.0, false),
        ]);
        let plan = r.residency_plan(1, policy);
        assert_eq!(plan.stream_in, vec![2]);
        assert_eq!(plan.evict, vec![0]);
    }

    #[test]
    fn residency_plan_is_sorted_and_deterministic() {
        let policy = LodResidencyPolicy::new(0, 0);
        let r = record(vec![
            lod(3, 0.1, true),
            lod(2, 0.1, false),
            lod(1, 0.1, true),
            lod(0, 0.1, false),
        ]);
        let plan = r.residency_plan(2, policy);
        // Only level 2 is wanted, and it is present but not resident.
        assert_eq!(plan.stream_in, vec![2]);
        // Resident 1 and 3 are outside the window -> evict sorted [1, 3].
        assert_eq!(plan.evict, vec![1, 3]);
    }
}
