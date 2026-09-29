//! Per-frame ray-march / modelling / upsample / multi-scatter budget arbitration.
//!
//! View ray-marching, procedural density-field modelling (baking coverage /
//! type / height gradients and detail erosion into the voxel field),
//! temporal-upsample resolves, and multi-scatter probe / `LUT` updates all run
//! on the `GPU` before shading. Like the shared deformation scheduler
//! ([`crate::deformation::schedule`]), they compete for bounded per-frame work,
//! but the volumetric engine splits that work into *independent* quotas so a
//! saturated ray-march never starves modelling, temporal upsampling, or
//! multi-scatter updates. This module is the `CPU` decision layer that admits
//! volumetric jobs within a [`VolumetricBudget`] while guaranteeing forward
//! progress in every quota.
//!
//! Selection is priority-greedy and fully deterministic: requests are ordered
//! by descending priority, then ascending handle, then a fixed job-kind order.
//! Each job is charged against the single quota its [`VolumetricJobKind`] maps
//! to, and each quota is accounted separately. Per quota, the first (highest-
//! priority) job is admitted unconditionally so an oversized job can never
//! starve forever; every later job in that quota is admitted only while its
//! quota has room and otherwise defers to a later frame.

use alloc::vec::Vec;

use super::{CloudLayerHandle, VolumetricBudget};

/// The kind of one volumetric job, which selects the [`VolumetricBudget`] quota
/// it is charged against.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum VolumetricJobKind {
    /// A view ray-march sample dispatch. Charged against
    /// [`VolumetricBudget::raymarch_samples_per_frame`].
    Raymarch,
    /// Procedural density-field modelling / bake. Charged against
    /// [`VolumetricBudget::modeling_voxels_per_frame`].
    Modeling,
    /// Temporal-upsample / reprojection resolve. Charged against
    /// [`VolumetricBudget::upsample_pixels_per_frame`].
    Upsample,
    /// Multi-scatter probe / `LUT` cell update. Charged against
    /// [`VolumetricBudget::multiscatter_cells_per_frame`].
    Multiscatter,
}

impl VolumetricJobKind {
    /// Deterministic ordering rank used to break tie-priority, tie-handle jobs.
    #[must_use]
    fn order(self) -> u8 {
        match self {
            VolumetricJobKind::Raymarch => 0,
            VolumetricJobKind::Modeling => 1,
            VolumetricJobKind::Upsample => 2,
            VolumetricJobKind::Multiscatter => 3,
        }
    }

    /// The quota cap (from `budget`) this job kind draws from.
    #[must_use]
    fn quota(self, budget: VolumetricBudget) -> u32 {
        match self {
            VolumetricJobKind::Raymarch => budget.raymarch_samples_per_frame,
            VolumetricJobKind::Modeling => budget.modeling_voxels_per_frame,
            VolumetricJobKind::Upsample => budget.upsample_pixels_per_frame,
            VolumetricJobKind::Multiscatter => budget.multiscatter_cells_per_frame,
        }
    }
}

/// A request to run one volumetric job this frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VolumetricJobRequest {
    /// Cloud layer this job belongs to.
    pub handle: CloudLayerHandle,
    /// Which stage this job is, selecting its budget quota.
    pub kind: VolumetricJobKind,
    /// Work units charged against this job's quota (samples / voxels / pixels /
    /// cells).
    pub cost: u32,
    /// Higher runs first; ties break by ascending handle then job-kind order.
    pub priority: u32,
}

/// The volumetric jobs chosen this frame plus what slipped to a later frame.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VolumetricPlan {
    /// Admitted jobs in dispatch order (priority desc, handle asc, kind order).
    pub scheduled: Vec<VolumetricJobRequest>,
    /// Jobs that did not fit their quota and defer to a later frame.
    pub deferred: Vec<VolumetricJobRequest>,
    /// Ray-march sample units charged this frame.
    pub raymarch_used: u32,
    /// Modelling voxel units charged this frame.
    pub modeling_used: u32,
    /// Temporal-upsample pixel units charged this frame.
    pub upsample_used: u32,
    /// Multi-scatter cell units charged this frame.
    pub multiscatter_used: u32,
}

impl VolumetricPlan {
    /// Number of admitted jobs.
    #[must_use]
    pub fn scheduled_count(&self) -> usize {
        self.scheduled.len()
    }

    /// Number of admitted jobs of a given kind, for per-stage dispatch.
    #[must_use]
    pub fn count_of_kind(&self, kind: VolumetricJobKind) -> usize {
        self.scheduled.iter().filter(|job| job.kind == kind).count()
    }

    /// Handles admitted for `kind`, in dispatch order.
    #[must_use]
    pub fn handles_of_kind(&self, kind: VolumetricJobKind) -> Vec<CloudLayerHandle> {
        self.scheduled
            .iter()
            .filter(|job| job.kind == kind)
            .map(|job| job.handle)
            .collect()
    }

    /// Work units charged against the quota backing `kind`.
    #[must_use]
    pub fn used_of_kind(&self, kind: VolumetricJobKind) -> u32 {
        match kind {
            VolumetricJobKind::Raymarch => self.raymarch_used,
            VolumetricJobKind::Modeling => self.modeling_used,
            VolumetricJobKind::Upsample => self.upsample_used,
            VolumetricJobKind::Multiscatter => self.multiscatter_used,
        }
    }

    /// Adds `cost` to the running total for `kind`'s quota.
    fn charge(&mut self, kind: VolumetricJobKind, cost: u32) {
        match kind {
            VolumetricJobKind::Raymarch => {
                self.raymarch_used = self.raymarch_used.saturating_add(cost);
            }
            VolumetricJobKind::Modeling => {
                self.modeling_used = self.modeling_used.saturating_add(cost);
            }
            VolumetricJobKind::Upsample => {
                self.upsample_used = self.upsample_used.saturating_add(cost);
            }
            VolumetricJobKind::Multiscatter => {
                self.multiscatter_used = self.multiscatter_used.saturating_add(cost);
            }
        }
    }
}

/// Per-quota running state: current usage and whether the first job landed.
#[derive(Clone, Copy, Default)]
struct QuotaState {
    /// Work units already charged against this quota this frame.
    used: u32,
    /// Whether the unconditional first job for this quota has been admitted.
    seen_first: bool,
}

/// Chooses the volumetric jobs to run this frame within `budget`.
///
/// Requests are ordered by descending priority, then ascending handle, then a
/// fixed job-kind order for full determinism. Each job is charged against the
/// single quota its [`VolumetricJobKind`] selects. Per quota, the first job is
/// admitted unconditionally (forward-progress guarantee), and every later job
/// in that quota is admitted while `used + cost` stays within the quota cap,
/// otherwise it defers. Quotas are independent, so saturating one never blocks
/// another.
#[must_use]
pub fn plan_volumetric(
    requests: &[VolumetricJobRequest],
    budget: VolumetricBudget,
) -> VolumetricPlan {
    let mut ordered: Vec<VolumetricJobRequest> = requests.to_vec();
    ordered.sort_by(|a, b| {
        b.priority
            .cmp(&a.priority)
            .then(a.handle.0.cmp(&b.handle.0))
            .then(a.kind.order().cmp(&b.kind.order()))
    });

    let mut plan = VolumetricPlan::default();
    // One quota state per job kind, indexed by `VolumetricJobKind::order`.
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

    fn req(handle: u32, kind: VolumetricJobKind, cost: u32, priority: u32) -> VolumetricJobRequest {
        VolumetricJobRequest {
            handle: CloudLayerHandle(handle),
            kind,
            cost,
            priority,
        }
    }

    const BUDGET: VolumetricBudget = VolumetricBudget {
        raymarch_samples_per_frame: 1000,
        modeling_voxels_per_frame: 1000,
        upsample_pixels_per_frame: 1000,
        multiscatter_cells_per_frame: 1000,
    };

    #[test]
    fn admits_by_priority_then_handle_then_kind() {
        let requests = [
            req(2, VolumetricJobKind::Raymarch, 100, 5),
            req(1, VolumetricJobKind::Raymarch, 100, 5),
            req(3, VolumetricJobKind::Raymarch, 100, 9),
        ];
        let plan = plan_volumetric(&requests, BUDGET);
        let order: Vec<u32> = plan.scheduled.iter().map(|j| j.handle.0).collect();
        assert_eq!(order, [3, 1, 2]);
        assert_eq!(plan.raymarch_used, 300);
        assert!(plan.deferred.is_empty());
    }

    #[test]
    fn kind_order_breaks_full_ties() {
        let requests = [
            req(1, VolumetricJobKind::Multiscatter, 1, 5),
            req(1, VolumetricJobKind::Raymarch, 1, 5),
            req(1, VolumetricJobKind::Upsample, 1, 5),
            req(1, VolumetricJobKind::Modeling, 1, 5),
        ];
        let plan = plan_volumetric(&requests, BUDGET);
        let kinds: Vec<VolumetricJobKind> = plan.scheduled.iter().map(|j| j.kind).collect();
        assert_eq!(
            kinds,
            [
                VolumetricJobKind::Raymarch,
                VolumetricJobKind::Modeling,
                VolumetricJobKind::Upsample,
                VolumetricJobKind::Multiscatter,
            ]
        );
    }

    #[test]
    fn quotas_are_independent() {
        // Saturating the ray-march quota must not block modelling jobs.
        let requests = [
            req(1, VolumetricJobKind::Raymarch, 900, 10),
            req(2, VolumetricJobKind::Raymarch, 900, 9),
            req(3, VolumetricJobKind::Modeling, 900, 8),
        ];
        let plan = plan_volumetric(&requests, BUDGET);
        // Raymarch: first admitted (900), second overflows (900+900>1000) ->
        // defer. Modeling: independent quota, admitted.
        assert_eq!(plan.count_of_kind(VolumetricJobKind::Raymarch), 1);
        assert_eq!(plan.count_of_kind(VolumetricJobKind::Modeling), 1);
        assert_eq!(plan.deferred.len(), 1);
        assert_eq!(plan.deferred[0].handle, CloudLayerHandle(2));
        assert_eq!(plan.raymarch_used, 900);
        assert_eq!(plan.modeling_used, 900);
    }

    #[test]
    fn oversized_first_job_per_quota_never_starves() {
        let requests = [
            req(1, VolumetricJobKind::Raymarch, 5000, 10),
            req(2, VolumetricJobKind::Multiscatter, 5000, 9),
        ];
        let plan = plan_volumetric(&requests, BUDGET);
        // Both are the first job in their own quota, so both are admitted even
        // though each alone exceeds the quota.
        assert_eq!(plan.scheduled_count(), 2);
        assert!(plan.deferred.is_empty());
        assert_eq!(plan.raymarch_used, 5000);
        assert_eq!(plan.multiscatter_used, 5000);
    }

    #[test]
    fn defers_later_jobs_that_overflow_their_quota() {
        let requests = [
            req(1, VolumetricJobKind::Upsample, 800, 10),
            req(2, VolumetricJobKind::Upsample, 800, 9),
        ];
        let plan = plan_volumetric(&requests, BUDGET);
        assert_eq!(plan.scheduled_count(), 1);
        assert_eq!(plan.scheduled[0].handle, CloudLayerHandle(1));
        assert_eq!(plan.deferred, [req(2, VolumetricJobKind::Upsample, 800, 9)]);
        assert_eq!(plan.upsample_used, 800);
    }

    #[test]
    fn handles_of_kind_preserve_dispatch_order() {
        let requests = [
            req(1, VolumetricJobKind::Raymarch, 10, 8),
            req(2, VolumetricJobKind::Modeling, 10, 9),
            req(3, VolumetricJobKind::Raymarch, 10, 10),
        ];
        let plan = plan_volumetric(&requests, BUDGET);
        assert_eq!(
            plan.handles_of_kind(VolumetricJobKind::Raymarch),
            [CloudLayerHandle(3), CloudLayerHandle(1)]
        );
        assert_eq!(
            plan.handles_of_kind(VolumetricJobKind::Modeling),
            [CloudLayerHandle(2)]
        );
        assert_eq!(plan.used_of_kind(VolumetricJobKind::Raymarch), 20);
    }

    #[test]
    fn empty_requests_yield_empty_plan() {
        let plan = plan_volumetric(&[], BUDGET);
        assert_eq!(plan.scheduled_count(), 0);
        assert_eq!(plan.raymarch_used, 0);
        assert!(plan.deferred.is_empty());
    }

    #[test]
    fn zero_budget_still_admits_one_job_per_quota() {
        let zero = VolumetricBudget::default();
        let requests = [
            req(1, VolumetricJobKind::Raymarch, 10, 10),
            req(2, VolumetricJobKind::Raymarch, 10, 9),
        ];
        let plan = plan_volumetric(&requests, zero);
        assert_eq!(plan.scheduled_count(), 1);
        assert_eq!(plan.scheduled[0].handle, CloudLayerHandle(1));
        assert_eq!(plan.deferred, [req(2, VolumetricJobKind::Raymarch, 10, 9)]);
    }
}
