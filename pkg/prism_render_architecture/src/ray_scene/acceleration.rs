//! Acceleration-structure update policy and rebuild budgeting.
//!
//! Bottom-level (`BLAS`) and top-level (`TLAS`) acceleration structures are
//! expensive to build, so each frame the renderer decides the cheapest update
//! that still yields a correct structure:
//!
//! - `Reuse` — nothing changed; keep the existing structure.
//! - `Refit` — vertices moved but topology is intact; refit bounding volumes.
//! - `Rebuild` — topology changed or deformation is too large to refit well.
//! - `BuildAndCompact` — rebuild, then compact because fragmentation is high.
//!
//! The decision is threshold-driven and deterministic. The `GPU` `BVH` build and
//! compaction passes are pending the GPU backend; this module owns the
//! `CPU`-verifiable policy and the per-frame rebuild accounting.

/// Update strategy chosen for an acceleration structure this frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AccelerationUpdate {
    /// Keep the existing structure unchanged.
    Reuse,
    /// Refit bounding volumes in place; topology is unchanged.
    Refit,
    /// Rebuild the structure from scratch.
    Rebuild,
    /// Rebuild and then compact to reclaim fragmented memory.
    BuildAndCompact,
}

impl AccelerationUpdate {
    /// True when the update reconstructs topology (rebuild variants).
    #[must_use]
    pub const fn is_rebuild(self) -> bool {
        matches!(self, Self::Rebuild | Self::BuildAndCompact)
    }
}

/// Per-frame description of how much a structure's geometry changed.
///
/// Ratios are fractions in `[0, 1]`; `max_vertex_deformation` is the largest
/// per-vertex displacement expressed as a fraction of the structure's bounding
/// radius, so it is resolution independent.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GeometryChange {
    /// Primitives touched this frame.
    pub moved_primitives: u32,
    /// Total primitives in the structure.
    pub total_primitives: u32,
    /// Largest per-vertex displacement as a fraction of the bounding radius.
    pub max_vertex_deformation: f64,
    /// Index/primitive counts changed, invalidating the current topology.
    pub topology_changed: bool,
    /// Fraction of the allocation that is stale/fragmented, in `[0, 1]`.
    pub fragmentation: f64,
}

impl GeometryChange {
    /// Fraction of primitives that moved this frame, in `[0, 1]`.
    #[must_use]
    pub fn moved_ratio(self) -> f64 {
        if self.total_primitives == 0 {
            return 0.0;
        }
        let moved = f64::from(self.moved_primitives);
        let total = f64::from(self.total_primitives);
        (moved / total).clamp(0.0, 1.0)
    }
}

/// Thresholds steering [`AccelerationUpdatePolicy::decide`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AccelerationUpdatePolicy {
    /// At or below this deformation a refit preserves quality.
    pub refit_deformation_max: f64,
    /// At or below this moved-primitive ratio a refit is allowed.
    pub refit_moved_ratio_max: f64,
    /// At or above this fragmentation a rebuild also compacts.
    pub compact_fragmentation_min: f64,
    /// At or above this measured refit-quality ratio (current `SAH` cost over a
    /// rebuild's cost, from [`Bvh::refit_quality`](super::bvh::Bvh::refit_quality))
    /// accumulated refits have degraded the tree enough to force a rebuild.
    pub refit_quality_rebuild_ratio: f64,
}

impl Default for AccelerationUpdatePolicy {
    fn default() -> Self {
        Self {
            refit_deformation_max: 0.05,
            refit_moved_ratio_max: 0.25,
            compact_fragmentation_min: 0.5,
            refit_quality_rebuild_ratio: 1.3,
        }
    }
}

impl AccelerationUpdatePolicy {
    /// Chooses the cheapest correct update for `change`.
    ///
    /// Topology changes force a rebuild. Otherwise a structure with no motion is
    /// reused; small motion within both thresholds is refit; anything larger is
    /// rebuilt. Rebuilds escalate to [`AccelerationUpdate::BuildAndCompact`] when
    /// fragmentation reaches [`Self::compact_fragmentation_min`].
    #[must_use]
    pub fn decide(&self, change: GeometryChange) -> AccelerationUpdate {
        let deformation = if change.max_vertex_deformation.is_nan() {
            0.0
        } else {
            change.max_vertex_deformation
        };
        let moved = change.moved_ratio();

        if change.topology_changed {
            return self.rebuild_variant(change.fragmentation);
        }
        if moved <= 0.0 && deformation <= 0.0 {
            return AccelerationUpdate::Reuse;
        }
        if deformation <= self.refit_deformation_max && moved <= self.refit_moved_ratio_max {
            return AccelerationUpdate::Refit;
        }
        self.rebuild_variant(change.fragmentation)
    }

    fn rebuild_variant(&self, fragmentation: f64) -> AccelerationUpdate {
        let frag = if fragmentation.is_nan() {
            0.0
        } else {
            fragmentation
        };
        if frag >= self.compact_fragmentation_min {
            return AccelerationUpdate::BuildAndCompact;
        }
        AccelerationUpdate::Rebuild
    }

    /// Whether an *already refit* structure has degraded enough to warrant a
    /// rebuild, from its measured surface-area-heuristic quality.
    ///
    /// [`decide`](Self::decide) chooses refit-vs-rebuild *before* an update from
    /// predicted motion, but a run of accepted refits keeps the original split
    /// planes while geometry drifts, so quality erodes silently. Feeding the
    /// measured [`Bvh::refit_quality`](super::bvh::Bvh::refit_quality) — `1.0`
    /// for an ideal tree, larger as topology stops matching geometry — back into
    /// the policy closes that loop: once the ratio reaches
    /// [`Self::refit_quality_rebuild_ratio`] the next update is escalated to a
    /// rebuild even though per-frame motion still looked refit-sized. A `NaN`
    /// quality (undefined, e.g. an empty tree) is treated as no degradation and
    /// does not trigger a rebuild.
    #[must_use]
    pub fn should_rebuild_after_refit(&self, refit_quality: f64) -> bool {
        if refit_quality.is_nan() {
            return false;
        }
        refit_quality >= self.refit_quality_rebuild_ratio
    }
}

/// Rough cost, in bytes of build scratch, attributed to an update.
///
/// `Reuse` is free; `Refit` touches only bounding volumes; the rebuild variants
/// pay the full primitive build cost, and compaction adds a copy pass.
#[must_use]
pub fn update_scratch_bytes(
    update: AccelerationUpdate,
    primitives: u32,
    bytes_per_primitive: u32,
) -> u64 {
    let prims = u64::from(primitives);
    let per = u64::from(bytes_per_primitive);
    match update {
        AccelerationUpdate::Reuse => 0,
        AccelerationUpdate::Refit => prims.saturating_mul(per) / 4,
        AccelerationUpdate::Rebuild => prims.saturating_mul(per),
        AccelerationUpdate::BuildAndCompact => {
            let build = prims.saturating_mul(per);
            build.saturating_add(build / 2)
        }
    }
}

/// Accumulates rebuild work against a per-frame byte budget.
///
/// The renderer admits structures until the budget is exhausted, then defers the
/// rest to later frames. This keeps `BVH` rebuild spikes bounded.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RebuildLedger {
    budget_bytes: u64,
    spent_bytes: u64,
    admitted: u32,
    deferred: u32,
}

impl RebuildLedger {
    /// Starts a ledger with `budget_bytes` of rebuild scratch per frame.
    #[must_use]
    pub const fn new(budget_bytes: u64) -> Self {
        Self {
            budget_bytes,
            spent_bytes: 0,
            admitted: 0,
            deferred: 0,
        }
    }

    /// Attempts to admit an update costing `cost_bytes`.
    ///
    /// Returns `true` when the update fits in the remaining budget (and charges
    /// it), or `false` when it is deferred. Zero-cost updates always fit.
    pub fn admit(&mut self, cost_bytes: u64) -> bool {
        if cost_bytes == 0 {
            self.admitted += 1;
            return true;
        }
        let projected = self.spent_bytes.saturating_add(cost_bytes);
        if projected > self.budget_bytes {
            self.deferred += 1;
            return false;
        }
        self.spent_bytes = projected;
        self.admitted += 1;
        true
    }

    /// Bytes charged so far this frame.
    #[must_use]
    pub const fn spent_bytes(self) -> u64 {
        self.spent_bytes
    }

    /// Bytes still available this frame.
    #[must_use]
    pub const fn remaining_bytes(self) -> u64 {
        self.budget_bytes.saturating_sub(self.spent_bytes)
    }

    /// Number of admitted and deferred updates so far.
    #[must_use]
    pub const fn counts(self) -> (u32, u32) {
        (self.admitted, self.deferred)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_geometry_is_reused() {
        let policy = AccelerationUpdatePolicy::default();
        let change = GeometryChange {
            total_primitives: 1000,
            ..Default::default()
        };
        assert_eq!(policy.decide(change), AccelerationUpdate::Reuse);
    }

    #[test]
    fn small_motion_refits() {
        let policy = AccelerationUpdatePolicy::default();
        let change = GeometryChange {
            moved_primitives: 100,
            total_primitives: 1000,
            max_vertex_deformation: 0.02,
            ..Default::default()
        };
        assert_eq!(policy.decide(change), AccelerationUpdate::Refit);
    }

    #[test]
    fn large_deformation_rebuilds() {
        let policy = AccelerationUpdatePolicy::default();
        let change = GeometryChange {
            moved_primitives: 100,
            total_primitives: 1000,
            max_vertex_deformation: 0.5,
            ..Default::default()
        };
        assert_eq!(policy.decide(change), AccelerationUpdate::Rebuild);
    }

    #[test]
    fn wide_motion_rebuilds_even_when_soft() {
        let policy = AccelerationUpdatePolicy::default();
        let change = GeometryChange {
            moved_primitives: 900,
            total_primitives: 1000,
            max_vertex_deformation: 0.01,
            ..Default::default()
        };
        assert_eq!(policy.decide(change), AccelerationUpdate::Rebuild);
    }

    #[test]
    fn topology_change_forces_rebuild() {
        let policy = AccelerationUpdatePolicy::default();
        let change = GeometryChange {
            total_primitives: 1000,
            topology_changed: true,
            ..Default::default()
        };
        assert_eq!(policy.decide(change), AccelerationUpdate::Rebuild);
    }

    #[test]
    fn fragmentation_escalates_to_compaction() {
        let policy = AccelerationUpdatePolicy::default();
        let change = GeometryChange {
            total_primitives: 1000,
            topology_changed: true,
            fragmentation: 0.75,
            ..Default::default()
        };
        assert_eq!(policy.decide(change), AccelerationUpdate::BuildAndCompact);
    }

    #[test]
    fn threshold_boundaries_prefer_refit() {
        let policy = AccelerationUpdatePolicy::default();
        // Exactly on both refit ceilings still refits.
        let change = GeometryChange {
            moved_primitives: 250,
            total_primitives: 1000,
            max_vertex_deformation: 0.05,
            ..Default::default()
        };
        assert_eq!(policy.decide(change), AccelerationUpdate::Refit);
        // Just past the moved-ratio ceiling rebuilds.
        let change_over = GeometryChange {
            moved_primitives: 251,
            total_primitives: 1000,
            max_vertex_deformation: 0.05,
            ..Default::default()
        };
        assert_eq!(policy.decide(change_over), AccelerationUpdate::Rebuild);
    }

    #[test]
    fn nan_inputs_are_treated_as_zero() {
        let policy = AccelerationUpdatePolicy::default();
        let change = GeometryChange {
            moved_primitives: 0,
            total_primitives: 1000,
            max_vertex_deformation: f64::NAN,
            fragmentation: f64::NAN,
            topology_changed: false,
        };
        assert_eq!(policy.decide(change), AccelerationUpdate::Reuse);
    }

    #[test]
    fn scratch_cost_orders_by_update_kind() {
        let reuse = update_scratch_bytes(AccelerationUpdate::Reuse, 1000, 64);
        let refit = update_scratch_bytes(AccelerationUpdate::Refit, 1000, 64);
        let rebuild = update_scratch_bytes(AccelerationUpdate::Rebuild, 1000, 64);
        let compact = update_scratch_bytes(AccelerationUpdate::BuildAndCompact, 1000, 64);
        assert_eq!(reuse, 0);
        assert!(refit < rebuild);
        assert!(rebuild < compact);
    }

    #[test]
    fn ledger_admits_within_budget_then_defers() {
        let mut ledger = RebuildLedger::new(1000);
        assert!(ledger.admit(400));
        assert!(ledger.admit(400));
        assert!(!ledger.admit(400));
        assert!(ledger.admit(0));
        assert_eq!(ledger.spent_bytes(), 800);
        assert_eq!(ledger.remaining_bytes(), 200);
        assert_eq!(ledger.counts(), (3, 1));
    }

    #[test]
    fn is_rebuild_classifies_variants() {
        assert!(!AccelerationUpdate::Reuse.is_rebuild());
        assert!(!AccelerationUpdate::Refit.is_rebuild());
        assert!(AccelerationUpdate::Rebuild.is_rebuild());
        assert!(AccelerationUpdate::BuildAndCompact.is_rebuild());
    }

    #[test]
    fn measured_refit_quality_escalates_to_rebuild_at_threshold() {
        let policy = AccelerationUpdatePolicy::default();
        // Ideal / mildly degraded trees keep refitting.
        assert!(!policy.should_rebuild_after_refit(1.0));
        assert!(!policy.should_rebuild_after_refit(1.29));
        // At and past the ratio a rebuild is forced.
        assert!(policy.should_rebuild_after_refit(policy.refit_quality_rebuild_ratio));
        assert!(policy.should_rebuild_after_refit(2.5));
    }

    #[test]
    fn undefined_refit_quality_never_rebuilds() {
        let policy = AccelerationUpdatePolicy::default();
        assert!(!policy.should_rebuild_after_refit(f64::NAN));
    }
}
