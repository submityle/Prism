//! Per-frame water solve/reconstruct/displacement budget arbitration.
//!
//! Ocean spectral stepping, shallow-water stepping, particle-fluid solves,
//! surface reconstruction, clipmap displacement-mesh generation, and foam
//! advection all run on the GPU before shading. Like the shared deformation
//! scheduler ([`crate::deformation::schedule`]), they compete for bounded
//! per-frame work, but water splits that work into *independent* quotas so a
//! saturated particle solve never starves foam advection or displacement mesh
//! generation. This module is the CPU decision layer that admits water jobs
//! within a [`WaterBudget`] while guaranteeing forward progress in every quota.
//!
//! Selection is priority-greedy and fully deterministic: requests are ordered
//! by descending priority, then ascending handle, then a fixed job-kind order.
//! Each job is charged against the single quota its [`WaterJobKind`] maps to,
//! and each quota is accounted separately. Per quota, the first (highest-
//! priority) job is admitted unconditionally so an oversized job can never
//! starve forever; every later job in that quota is admitted only while its
//! quota has room and otherwise defers to a later frame.

use alloc::vec::Vec;

use super::{WaterBodyHandle, WaterBudget};

/// The kind of one water job, which selects the [`WaterBudget`] quota it is
/// charged against.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum WaterJobKind {
    /// A solver step dispatch (spectral / shallow-water / `PBF` / `FLIP`).
    /// Charged against [`WaterBudget::solve_steps_per_frame`].
    SolveStep,
    /// Surface reconstruction for a particle body. Charged against
    /// [`WaterBudget::reconstruct_cells_per_frame`].
    Reconstruct,
    /// Clipmap displacement-mesh generation. Charged against
    /// [`WaterBudget::displacement_vertices_per_frame`].
    Displacement,
    /// Foam-density advection. Charged against
    /// [`WaterBudget::foam_cells_per_frame`].
    FoamAdvect,
}

impl WaterJobKind {
    /// Deterministic ordering rank used to break tie-priority, tie-handle jobs.
    #[must_use]
    fn order(self) -> u8 {
        match self {
            WaterJobKind::SolveStep => 0,
            WaterJobKind::Reconstruct => 1,
            WaterJobKind::Displacement => 2,
            WaterJobKind::FoamAdvect => 3,
        }
    }

    /// The quota cap (from `budget`) this job kind draws from.
    #[must_use]
    fn quota(self, budget: WaterBudget) -> u32 {
        match self {
            WaterJobKind::SolveStep => budget.solve_steps_per_frame,
            WaterJobKind::Reconstruct => budget.reconstruct_cells_per_frame,
            WaterJobKind::Displacement => budget.displacement_vertices_per_frame,
            WaterJobKind::FoamAdvect => budget.foam_cells_per_frame,
        }
    }
}

/// A request to run one water job this frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WaterJobRequest {
    /// Water body this job belongs to.
    pub handle: WaterBodyHandle,
    /// Which stage this job is, selecting its budget quota.
    pub kind: WaterJobKind,
    /// Work units charged against this job's quota (steps / cells / vertices).
    pub cost: u32,
    /// Higher runs first; ties break by ascending handle then job-kind order.
    pub priority: u32,
}

/// The water jobs chosen this frame plus what slipped to a later frame.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WaterSolvePlan {
    /// Admitted jobs in dispatch order (priority desc, handle asc, kind order).
    pub scheduled: Vec<WaterJobRequest>,
    /// Jobs that did not fit their quota and defer to a later frame.
    pub deferred: Vec<WaterJobRequest>,
    /// Solve-step units charged this frame.
    pub steps_used: u32,
    /// Reconstruction cells charged this frame.
    pub reconstruct_used: u32,
    /// Displacement vertices charged this frame.
    pub displacement_used: u32,
    /// Foam-advection cells charged this frame.
    pub foam_used: u32,
}

impl WaterSolvePlan {
    /// Number of admitted jobs.
    #[must_use]
    pub fn scheduled_count(&self) -> usize {
        self.scheduled.len()
    }

    /// Number of admitted jobs of a given kind, for per-stage dispatch.
    #[must_use]
    pub fn count_of_kind(&self, kind: WaterJobKind) -> usize {
        self.scheduled.iter().filter(|job| job.kind == kind).count()
    }

    /// Handles admitted for `kind`, in dispatch order.
    #[must_use]
    pub fn handles_of_kind(&self, kind: WaterJobKind) -> Vec<WaterBodyHandle> {
        self.scheduled
            .iter()
            .filter(|job| job.kind == kind)
            .map(|job| job.handle)
            .collect()
    }

    /// Work units charged against the quota backing `kind`.
    #[must_use]
    pub fn used_of_kind(&self, kind: WaterJobKind) -> u32 {
        match kind {
            WaterJobKind::SolveStep => self.steps_used,
            WaterJobKind::Reconstruct => self.reconstruct_used,
            WaterJobKind::Displacement => self.displacement_used,
            WaterJobKind::FoamAdvect => self.foam_used,
        }
    }

    /// Adds `cost` to the running total for `kind`'s quota.
    fn charge(&mut self, kind: WaterJobKind, cost: u32) {
        match kind {
            WaterJobKind::SolveStep => self.steps_used = self.steps_used.saturating_add(cost),
            WaterJobKind::Reconstruct => {
                self.reconstruct_used = self.reconstruct_used.saturating_add(cost);
            }
            WaterJobKind::Displacement => {
                self.displacement_used = self.displacement_used.saturating_add(cost);
            }
            WaterJobKind::FoamAdvect => self.foam_used = self.foam_used.saturating_add(cost),
        }
    }
}

/// Per-quota running state: current usage and whether the first job landed.
#[derive(Clone, Copy, Default)]
struct QuotaState {
    used: u32,
    seen_first: bool,
}

/// Chooses the water jobs to run this frame within `budget`.
///
/// Requests are ordered by descending priority, then ascending handle, then a
/// fixed job-kind order for full determinism. Each job is charged against the
/// single quota its [`WaterJobKind`] selects. Per quota, the first job is
/// admitted unconditionally (forward-progress guarantee), and every later job
/// in that quota is admitted while `used + cost` stays within the quota cap,
/// otherwise it defers. Quotas are independent, so saturating one never blocks
/// another.
#[must_use]
pub fn plan_water(requests: &[WaterJobRequest], budget: WaterBudget) -> WaterSolvePlan {
    let mut ordered: Vec<WaterJobRequest> = requests.to_vec();
    ordered.sort_by(|a, b| {
        b.priority
            .cmp(&a.priority)
            .then(a.handle.0.cmp(&b.handle.0))
            .then(a.kind.order().cmp(&b.kind.order()))
    });

    let mut plan = WaterSolvePlan::default();
    // One quota state per job kind, indexed by `WaterJobKind::order`.
    let mut quotas = [QuotaState::default(); 4];

    for request in ordered {
        let state = &mut quotas[request.kind.order() as usize];
        let cap = request.kind.quota(budget);
        let next = state.used.saturating_add(request.cost);

        // The first job in each quota is admitted unconditionally so an
        // oversized top-priority job can never starve; later jobs must fit.
        if state.seen_first && next > cap {
            plan.deferred.push(request);
            continue;
        }
        state.seen_first = true;
        state.used = next;
        plan.charge(request.kind, request.cost);
        plan.scheduled.push(request);
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(handle: u32, kind: WaterJobKind, cost: u32, priority: u32) -> WaterJobRequest {
        WaterJobRequest {
            handle: WaterBodyHandle(handle),
            kind,
            cost,
            priority,
        }
    }

    const BUDGET: WaterBudget = WaterBudget {
        solve_steps_per_frame: 1000,
        reconstruct_cells_per_frame: 1000,
        displacement_vertices_per_frame: 1000,
        foam_cells_per_frame: 1000,
    };

    #[test]
    fn admits_by_priority_then_handle_then_kind() {
        let requests = [
            req(2, WaterJobKind::SolveStep, 100, 5),
            req(1, WaterJobKind::SolveStep, 100, 5),
            req(3, WaterJobKind::SolveStep, 100, 9),
        ];
        let plan = plan_water(&requests, BUDGET);
        let order: Vec<u32> = plan.scheduled.iter().map(|j| j.handle.0).collect();
        assert_eq!(order, [3, 1, 2]);
        assert_eq!(plan.steps_used, 300);
        assert!(plan.deferred.is_empty());
    }

    #[test]
    fn kind_order_breaks_full_ties() {
        let requests = [
            req(1, WaterJobKind::FoamAdvect, 1, 5),
            req(1, WaterJobKind::SolveStep, 1, 5),
            req(1, WaterJobKind::Displacement, 1, 5),
            req(1, WaterJobKind::Reconstruct, 1, 5),
        ];
        let plan = plan_water(&requests, BUDGET);
        let kinds: Vec<WaterJobKind> = plan.scheduled.iter().map(|j| j.kind).collect();
        assert_eq!(
            kinds,
            [
                WaterJobKind::SolveStep,
                WaterJobKind::Reconstruct,
                WaterJobKind::Displacement,
                WaterJobKind::FoamAdvect,
            ]
        );
    }

    #[test]
    fn quotas_are_independent() {
        // Saturating the solve quota must not block reconstruction jobs.
        let requests = [
            req(1, WaterJobKind::SolveStep, 900, 10),
            req(2, WaterJobKind::SolveStep, 900, 9),
            req(3, WaterJobKind::Reconstruct, 900, 8),
        ];
        let plan = plan_water(&requests, BUDGET);
        // Solve: first admitted (900), second overflows (900+900>1000) -> defer.
        // Reconstruct: independent quota, admitted.
        assert_eq!(plan.count_of_kind(WaterJobKind::SolveStep), 1);
        assert_eq!(plan.count_of_kind(WaterJobKind::Reconstruct), 1);
        assert_eq!(plan.deferred.len(), 1);
        assert_eq!(plan.deferred[0].handle, WaterBodyHandle(2));
        assert_eq!(plan.steps_used, 900);
        assert_eq!(plan.reconstruct_used, 900);
    }

    #[test]
    fn oversized_first_job_per_quota_never_starves() {
        let requests = [
            req(1, WaterJobKind::SolveStep, 5000, 10),
            req(2, WaterJobKind::FoamAdvect, 5000, 9),
        ];
        let plan = plan_water(&requests, BUDGET);
        // Both are the first job in their own quota, so both are admitted even
        // though each alone exceeds the quota.
        assert_eq!(plan.scheduled_count(), 2);
        assert!(plan.deferred.is_empty());
        assert_eq!(plan.steps_used, 5000);
        assert_eq!(plan.foam_used, 5000);
    }

    #[test]
    fn defers_later_jobs_that_overflow_their_quota() {
        let requests = [
            req(1, WaterJobKind::Displacement, 800, 10),
            req(2, WaterJobKind::Displacement, 800, 9),
        ];
        let plan = plan_water(&requests, BUDGET);
        assert_eq!(plan.scheduled_count(), 1);
        assert_eq!(plan.scheduled[0].handle, WaterBodyHandle(1));
        assert_eq!(plan.deferred, [req(2, WaterJobKind::Displacement, 800, 9)]);
        assert_eq!(plan.displacement_used, 800);
    }

    #[test]
    fn handles_of_kind_preserve_dispatch_order() {
        let requests = [
            req(1, WaterJobKind::SolveStep, 10, 8),
            req(2, WaterJobKind::Reconstruct, 10, 9),
            req(3, WaterJobKind::SolveStep, 10, 10),
        ];
        let plan = plan_water(&requests, BUDGET);
        assert_eq!(
            plan.handles_of_kind(WaterJobKind::SolveStep),
            [WaterBodyHandle(3), WaterBodyHandle(1)]
        );
        assert_eq!(
            plan.handles_of_kind(WaterJobKind::Reconstruct),
            [WaterBodyHandle(2)]
        );
        assert_eq!(plan.used_of_kind(WaterJobKind::SolveStep), 20);
    }

    #[test]
    fn empty_requests_yield_empty_plan() {
        let plan = plan_water(&[], BUDGET);
        assert_eq!(plan.scheduled_count(), 0);
        assert_eq!(plan.steps_used, 0);
        assert!(plan.deferred.is_empty());
    }

    #[test]
    fn zero_budget_still_admits_one_job_per_quota() {
        let zero = WaterBudget::default();
        let requests = [
            req(1, WaterJobKind::SolveStep, 10, 10),
            req(2, WaterJobKind::SolveStep, 10, 9),
        ];
        let plan = plan_water(&requests, zero);
        assert_eq!(plan.scheduled_count(), 1);
        assert_eq!(plan.scheduled[0].handle, WaterBodyHandle(1));
        assert_eq!(plan.deferred, [req(2, WaterJobKind::SolveStep, 10, 9)]);
    }
}
