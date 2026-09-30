//! Per-structure acceleration-update scheduler.
//!
//! [`AccelerationUpdatePolicy`] answers two questions in isolation: what update
//! *this* frame's motion warrants ([`decide`](AccelerationUpdatePolicy::decide),
//! a-priori) and whether an accumulated run of refits has silently degraded the
//! tree enough to rebuild
//! ([`should_rebuild_after_refit`](AccelerationUpdatePolicy::should_rebuild_after_refit),
//! a-posteriori). [`RebuildLedger`] bounds how much rebuild scratch a frame may
//! spend. This module ties the three together into one deterministic per-frame
//! lifecycle so a caller drives a `BLAS` or `TLAS` correctly across frames.
//!
//! # Correctness vs. budget
//!
//! The scheduler distinguishes *mandatory* work from an *optional* escalation:
//!
//! - The a-priori [`decide`](AccelerationUpdatePolicy::decide) result is
//!   mandatory. A refit encloses moved geometry and a rebuild absorbs a topology
//!   change; skipping either renders an incorrect structure. Mandatory work is
//!   force-charged to the ledger ([`RebuildLedger::force_admit`]) and may
//!   overrun the frame budget rather than be dropped.
//! - A rebuild armed by a prior frame's degraded refit quality is optional: the
//!   mandatory update already yields a *correct* structure and the rebuild only
//!   restores traversal speed. It is the sole work the budget may defer, retried
//!   on a later frame until it fits.
//!
//! The scheduler is acceleration-structure agnostic — it reasons over decisions,
//! costs, and measured quality, never geometry — so the same type drives both
//! bottom- and top-level structures.

use super::acceleration::{
    update_scratch_bytes, AccelerationUpdate, AccelerationUpdatePolicy, GeometryChange,
    RebuildLedger,
};

/// The update the scheduler resolved for a frame, after folding the carried-over
/// rebuild debt and the per-frame budget into the a-priori decision.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScheduledUpdate {
    /// The update the caller must execute this frame.
    pub update: AccelerationUpdate,
    /// Scratch bytes charged to the ledger for [`update`](Self::update).
    pub cost_bytes: u64,
    /// True when an *optional* quality-driven rebuild the scheduler wanted did
    /// not fit the budget: the caller executes the mandatory
    /// [`update`](Self::update) (a refit or reuse) this frame and the scheduler
    /// retains the rebuild debt to retry on a later frame.
    pub rebuild_deferred: bool,
}

/// Drives one acceleration structure's per-frame update decisions, carrying
/// rebuild debt between frames.
///
/// A scheduler owns the [`AccelerationUpdatePolicy`] and the structure's byte
/// cost per primitive, plus a single bit of cross-frame state: whether an
/// optional quality-driven rebuild is still owed. Feed each frame's
/// [`GeometryChange`] to [`plan`](Self::plan) and, after executing a refit,
/// report the measured quality to [`observe_refit_quality`](Self::observe_refit_quality).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AccelerationScheduler {
    policy: AccelerationUpdatePolicy,
    bytes_per_primitive: u32,
    rebuild_pending: bool,
}

impl AccelerationScheduler {
    /// Creates a scheduler with an explicit `policy` and the structure's build
    /// scratch cost `bytes_per_primitive`, with no rebuild debt owed.
    #[must_use]
    pub const fn new(policy: AccelerationUpdatePolicy, bytes_per_primitive: u32) -> Self {
        Self {
            policy,
            bytes_per_primitive,
            rebuild_pending: false,
        }
    }

    /// Creates a scheduler using the [`AccelerationUpdatePolicy::default`]
    /// thresholds.
    #[must_use]
    pub fn with_default_policy(bytes_per_primitive: u32) -> Self {
        Self::new(AccelerationUpdatePolicy::default(), bytes_per_primitive)
    }

    /// The policy steering this scheduler.
    #[must_use]
    pub const fn policy(&self) -> AccelerationUpdatePolicy {
        self.policy
    }

    /// Whether an optional quality-driven rebuild is currently owed (armed by a
    /// prior [`observe_refit_quality`](Self::observe_refit_quality) or a
    /// budget-deferred rebuild) and will be attempted on the next
    /// [`plan`](Self::plan).
    #[must_use]
    pub const fn rebuild_pending(&self) -> bool {
        self.rebuild_pending
    }

    /// Resolves this frame's update for `change`, charging `ledger`.
    ///
    /// The a-priori [`decide`](AccelerationUpdatePolicy::decide) result is
    /// mandatory and force-charged. When a quality-driven rebuild is owed and the
    /// mandatory decision is not already a rebuild, the scheduler attempts to
    /// upgrade to a rebuild (honouring the fragmentation compaction threshold)
    /// and gates *only that upgrade* on the budget: if it fits, the rebuild runs
    /// and the debt clears; if not, the mandatory update runs this frame,
    /// [`rebuild_deferred`](ScheduledUpdate::rebuild_deferred) is set, and the
    /// debt is retained for a later frame. Executing any rebuild clears the debt.
    pub fn plan(&mut self, change: GeometryChange, ledger: &mut RebuildLedger) -> ScheduledUpdate {
        let base = self.policy.decide(change);

        if self.rebuild_pending && !base.is_rebuild() {
            let wanted = self.policy.rebuild_variant(change.fragmentation);
            let cost = update_scratch_bytes(
                wanted,
                change.total_primitives,
                self.bytes_per_primitive,
            );
            if ledger.admit(cost) {
                self.rebuild_pending = false;
                return ScheduledUpdate {
                    update: wanted,
                    cost_bytes: cost,
                    rebuild_deferred: false,
                };
            }
            // Budget denied the optional rebuild: still run the mandatory base
            // update now (correctness cannot wait) and retry the rebuild later.
            let base_cost =
                update_scratch_bytes(base, change.total_primitives, self.bytes_per_primitive);
            ledger.force_admit(base_cost);
            return ScheduledUpdate {
                update: base,
                cost_bytes: base_cost,
                rebuild_deferred: true,
            };
        }

        let cost = update_scratch_bytes(base, change.total_primitives, self.bytes_per_primitive);
        ledger.force_admit(cost);
        if base.is_rebuild() {
            self.rebuild_pending = false;
        }
        ScheduledUpdate {
            update: base,
            cost_bytes: cost,
            rebuild_deferred: false,
        }
    }

    /// Records a measured refit quality (from
    /// [`Bvh::refit_quality`](super::bvh::Bvh::refit_quality) or
    /// [`Tlas::refit_quality`](super::tlas::Tlas::refit_quality)) after executing
    /// a refit, arming a rebuild for a later frame when the tree has degraded
    /// past [`AccelerationUpdatePolicy::refit_quality_rebuild_ratio`].
    ///
    /// A quality at or below the threshold (or an undefined `NaN`, e.g. an empty
    /// tree) leaves any existing debt untouched; the debt clears only when a
    /// rebuild actually executes via [`plan`](Self::plan).
    pub fn observe_refit_quality(&mut self, quality: f64) {
        if self.policy.should_rebuild_after_refit(quality) {
            self.rebuild_pending = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Per-primitive scratch cost used across the scheduler tests.
    const PER_PRIM: u32 = 64;

    fn moved(moved_primitives: u32, total_primitives: u32) -> GeometryChange {
        GeometryChange {
            moved_primitives,
            total_primitives,
            max_vertex_deformation: 0.0,
            topology_changed: false,
            fragmentation: 0.0,
        }
    }

    #[test]
    fn static_frame_reuses_without_charging() {
        let mut sched = AccelerationScheduler::with_default_policy(PER_PRIM);
        let mut ledger = RebuildLedger::new(1_000_000);
        let out = sched.plan(moved(0, 1000), &mut ledger);
        assert_eq!(out.update, AccelerationUpdate::Reuse);
        assert_eq!(out.cost_bytes, 0);
        assert!(!out.rebuild_deferred);
        assert_eq!(ledger.spent_bytes(), 0);
        assert!(!sched.rebuild_pending());
    }

    #[test]
    fn small_motion_refits_and_charges_refit_cost() {
        let mut sched = AccelerationScheduler::with_default_policy(PER_PRIM);
        let mut ledger = RebuildLedger::new(1_000_000);
        // 10% moved is within the default refit ratio.
        let out = sched.plan(moved(100, 1000), &mut ledger);
        assert_eq!(out.update, AccelerationUpdate::Refit);
        let expected = update_scratch_bytes(AccelerationUpdate::Refit, 1000, PER_PRIM);
        assert_eq!(out.cost_bytes, expected);
        assert_eq!(ledger.spent_bytes(), expected);
    }

    #[test]
    fn topology_change_force_admits_rebuild_over_budget() {
        let mut sched = AccelerationScheduler::with_default_policy(PER_PRIM);
        // Budget far below a full rebuild's cost.
        let mut ledger = RebuildLedger::new(10);
        let change = GeometryChange {
            topology_changed: true,
            total_primitives: 1000,
            ..GeometryChange::default()
        };
        let out = sched.plan(change, &mut ledger);
        assert_eq!(out.update, AccelerationUpdate::Rebuild);
        assert!(!out.rebuild_deferred, "mandatory rebuild is never deferred");
        let expected = update_scratch_bytes(AccelerationUpdate::Rebuild, 1000, PER_PRIM);
        assert_eq!(out.cost_bytes, expected);
        // Over-budget spend is recorded, not dropped.
        assert_eq!(ledger.spent_bytes(), expected);
        assert_eq!(ledger.remaining_bytes(), 0);
    }

    #[test]
    fn high_fragmentation_rebuild_compacts() {
        let mut sched = AccelerationScheduler::with_default_policy(PER_PRIM);
        let mut ledger = RebuildLedger::new(1_000_000);
        let change = GeometryChange {
            topology_changed: true,
            total_primitives: 500,
            fragmentation: 0.9,
            ..GeometryChange::default()
        };
        let out = sched.plan(change, &mut ledger);
        assert_eq!(out.update, AccelerationUpdate::BuildAndCompact);
    }

    #[test]
    fn degraded_refit_quality_arms_and_then_executes_a_rebuild() {
        let mut sched = AccelerationScheduler::with_default_policy(PER_PRIM);
        let mut ledger = RebuildLedger::new(1_000_000);

        // Frame 1: small motion refits; measured quality is still good.
        let f1 = sched.plan(moved(50, 1000), &mut ledger);
        assert_eq!(f1.update, AccelerationUpdate::Refit);
        sched.observe_refit_quality(1.05);
        assert!(!sched.rebuild_pending());

        // The tree has now drifted past the threshold: arm a rebuild.
        sched.observe_refit_quality(1.4);
        assert!(sched.rebuild_pending());

        // Frame 2: motion alone would refit, but the armed debt upgrades to a
        // rebuild that fits the budget, clearing the debt.
        let f2 = sched.plan(moved(50, 1000), &mut ledger);
        assert_eq!(f2.update, AccelerationUpdate::Rebuild);
        assert!(!f2.rebuild_deferred);
        assert!(!sched.rebuild_pending());
    }

    #[test]
    fn armed_rebuild_defers_under_budget_and_retries_next_frame() {
        let mut sched = AccelerationScheduler::with_default_policy(PER_PRIM);
        // Budget fits a refit (16 000 B) but not a full rebuild (64 000 B).
        let refit_cost = update_scratch_bytes(AccelerationUpdate::Refit, 1000, PER_PRIM);
        let rebuild_cost = update_scratch_bytes(AccelerationUpdate::Rebuild, 1000, PER_PRIM);
        assert!(refit_cost < rebuild_cost);
        let mut ledger = RebuildLedger::new(rebuild_cost - 1);

        sched.observe_refit_quality(1.5);
        assert!(sched.rebuild_pending());

        // Frame with small motion: the optional rebuild is denied, so the
        // mandatory refit runs and the debt is retained.
        let out = sched.plan(moved(50, 1000), &mut ledger);
        assert_eq!(out.update, AccelerationUpdate::Refit);
        assert!(out.rebuild_deferred);
        assert_eq!(out.cost_bytes, refit_cost);
        assert!(sched.rebuild_pending(), "deferred rebuild debt is retained");

        // Next frame with a fresh, ample budget: the retry finally rebuilds.
        let mut ledger2 = RebuildLedger::new(1_000_000);
        let retry = sched.plan(moved(50, 1000), &mut ledger2);
        assert_eq!(retry.update, AccelerationUpdate::Rebuild);
        assert!(!retry.rebuild_deferred);
        assert!(!sched.rebuild_pending());
    }

    #[test]
    fn undefined_quality_does_not_arm() {
        let mut sched = AccelerationScheduler::with_default_policy(PER_PRIM);
        sched.observe_refit_quality(f64::NAN);
        assert!(!sched.rebuild_pending());
        sched.observe_refit_quality(1.1);
        assert!(!sched.rebuild_pending());
    }

    #[test]
    fn plan_is_deterministic() {
        let make = || {
            let mut s = AccelerationScheduler::with_default_policy(PER_PRIM);
            s.observe_refit_quality(1.6);
            let mut l = RebuildLedger::new(1_000_000);
            let o = s.plan(moved(50, 1000), &mut l);
            (o, s.rebuild_pending(), l.spent_bytes())
        };
        assert_eq!(make(), make());
    }
}
