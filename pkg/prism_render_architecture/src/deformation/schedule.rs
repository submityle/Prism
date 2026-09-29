//! Per-frame deformation job scheduling within a fixed GPU budget.
//!
//! Skinning, morph targets, cloth, vertex animation, and hair all deform
//! geometry on the GPU before visibility and shading run. Each is its own
//! subsystem with its own compute pass, but they compete for one shared budget:
//! the number of vertices the deformation cache can process per frame and the
//! number of acceleration-structure (BLAS) refits ray tracing can afford. This
//! module is the CPU decision layer that picks which requested deformation jobs
//! run this frame and which slip to the next, keeping every subsystem within
//! [`DeformationBudget`] while guaranteeing forward progress.
//!
//! Selection is priority-greedy and fully deterministic: requests are ordered
//! by descending priority then ascending handle, then accepted while the vertex
//! budget allows. The single highest-priority request is always admitted even
//! if it alone exceeds the budget, so an oversized job (a dense hair groom, a
//! high-resolution cloth mesh) can never starve forever. BLAS refits are
//! accounted separately: a scheduled job that wants a refit takes a refit slot
//! when one is free, and otherwise still deforms this frame while its refit
//! defers, matching how a skin-cache update can precede its BLAS rebuild.

use alloc::vec::Vec;

use super::{DeformationBudget, DeformationHandle, DeformationKind};

/// A request to deform one cached geometry this frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeformationRequest {
    /// Deformation cache entry to update.
    pub handle: DeformationHandle,
    /// Which subsystem owns this deformation.
    pub kind: DeformationKind,
    /// Vertices this job will process, charged against the vertex budget.
    pub vertex_count: u32,
    /// Higher runs first; ties break by ascending handle for determinism.
    pub priority: u32,
    /// `true` when the deformed geometry needs a BLAS refit for ray tracing.
    pub needs_blas_refit: bool,
}

/// A deformation request admitted for this frame, with its refit decision.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScheduledDeformation {
    /// The admitted request.
    pub request: DeformationRequest,
    /// `true` when a BLAS refit slot was granted this frame. When the request
    /// wanted a refit but this is `false`, the refit is deferred to a later
    /// frame while the deformation itself still runs now.
    pub refit_granted: bool,
}

/// The deformation jobs chosen for this frame plus what slipped to the next.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DeformationPlan {
    /// Admitted jobs in dispatch order (priority desc, handle asc).
    pub scheduled: Vec<ScheduledDeformation>,
    /// Handles that did not fit the vertex budget and defer to a later frame.
    pub deferred: Vec<DeformationHandle>,
    /// Total vertices charged against the budget this frame.
    pub vertices_used: u32,
    /// BLAS refit slots consumed this frame.
    pub refits_used: u32,
}

impl DeformationPlan {
    /// Number of admitted jobs.
    #[must_use]
    pub fn scheduled_count(&self) -> usize {
        self.scheduled.len()
    }

    /// Number of admitted jobs of a given kind, for per-subsystem dispatch.
    #[must_use]
    pub fn count_of_kind(&self, kind: DeformationKind) -> usize {
        self.scheduled
            .iter()
            .filter(|job| job.request.kind == kind)
            .count()
    }

    /// Handles admitted for `kind`, in dispatch order.
    ///
    /// The backend runs one compute pass per subsystem; filtering the shared
    /// schedule preserves the deterministic priority order within each kind.
    #[must_use]
    pub fn handles_of_kind(&self, kind: DeformationKind) -> Vec<DeformationHandle> {
        self.scheduled
            .iter()
            .filter(|job| job.request.kind == kind)
            .map(|job| job.request.handle)
            .collect()
    }

    /// Handles whose deformation runs this frame but whose BLAS refit deferred.
    #[must_use]
    pub fn refit_deferred(&self) -> Vec<DeformationHandle> {
        self.scheduled
            .iter()
            .filter(|job| job.request.needs_blas_refit && !job.refit_granted)
            .map(|job| job.request.handle)
            .collect()
    }
}

/// Chooses the deformation jobs to run this frame within `budget`.
///
/// Requests are ordered by descending priority then ascending handle, then
/// admitted greedily while the vertex budget allows. The highest-priority
/// request is always admitted so no job starves; every later request that would
/// overflow the vertex budget defers. A refit slot is granted to an admitted
/// job that wants one while [`DeformationBudget::blas_refits_per_frame`] slots
/// remain; otherwise the refit defers but the deformation still runs.
#[must_use]
pub fn plan_deformations(
    requests: &[DeformationRequest],
    budget: DeformationBudget,
) -> DeformationPlan {
    let mut ordered: Vec<DeformationRequest> = requests.to_vec();
    ordered.sort_by(|a, b| {
        b.priority
            .cmp(&a.priority)
            .then_with(|| a.handle.0.cmp(&b.handle.0))
    });

    let mut plan = DeformationPlan::default();
    for request in ordered {
        let first = plan.scheduled.is_empty();
        let next_vertices = plan.vertices_used.saturating_add(request.vertex_count);
        // The very first (highest-priority) job is admitted unconditionally to
        // guarantee forward progress; the rest must fit the vertex budget.
        if !first && next_vertices > budget.vertices_per_frame {
            plan.deferred.push(request.handle);
            continue;
        }
        plan.vertices_used = next_vertices;

        let refit_granted =
            request.needs_blas_refit && plan.refits_used < budget.blas_refits_per_frame;
        if refit_granted {
            plan.refits_used += 1;
        }
        plan.scheduled.push(ScheduledDeformation {
            request,
            refit_granted,
        });
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(
        handle: u32,
        kind: DeformationKind,
        vertex_count: u32,
        priority: u32,
        needs_blas_refit: bool,
    ) -> DeformationRequest {
        DeformationRequest {
            handle: DeformationHandle(handle),
            kind,
            vertex_count,
            priority,
            needs_blas_refit,
        }
    }

    const BUDGET: DeformationBudget = DeformationBudget {
        vertices_per_frame: 1000,
        blas_refits_per_frame: 1,
    };

    #[test]
    fn admits_by_priority_then_handle() {
        let requests = [
            request(2, DeformationKind::Skinning, 100, 5, false),
            request(1, DeformationKind::Skinning, 100, 5, false),
            request(3, DeformationKind::Cloth, 100, 9, false),
        ];
        let plan = plan_deformations(&requests, BUDGET);
        let order: Vec<u32> = plan
            .scheduled
            .iter()
            .map(|job| job.request.handle.0)
            .collect();
        // Priority 9 first, then the two priority-5 jobs by ascending handle.
        assert_eq!(order, [3, 1, 2]);
        assert_eq!(plan.vertices_used, 300);
        assert!(plan.deferred.is_empty());
    }

    #[test]
    fn defers_jobs_that_overflow_the_vertex_budget() {
        let requests = [
            request(1, DeformationKind::Cloth, 800, 10, false),
            request(2, DeformationKind::Hair, 800, 9, false),
        ];
        let plan = plan_deformations(&requests, BUDGET);
        assert_eq!(plan.scheduled_count(), 1);
        assert_eq!(plan.scheduled[0].request.handle, DeformationHandle(1));
        assert_eq!(plan.deferred, [DeformationHandle(2)]);
        assert_eq!(plan.vertices_used, 800);
    }

    #[test]
    fn oversized_top_priority_job_never_starves() {
        // A single job larger than the whole budget is still admitted.
        let requests = [request(1, DeformationKind::Hair, 5000, 10, false)];
        let plan = plan_deformations(&requests, BUDGET);
        assert_eq!(plan.scheduled_count(), 1);
        assert_eq!(plan.vertices_used, 5000);
        assert!(plan.deferred.is_empty());
    }

    #[test]
    fn refit_slots_are_granted_then_deferred() {
        let requests = [
            request(1, DeformationKind::Cloth, 100, 10, true),
            request(2, DeformationKind::Hair, 100, 9, true),
        ];
        let plan = plan_deformations(&requests, BUDGET);
        // Only one refit slot: the higher-priority job gets it.
        assert!(plan.scheduled[0].refit_granted);
        assert!(!plan.scheduled[1].refit_granted);
        assert_eq!(plan.refits_used, 1);
        assert_eq!(plan.refit_deferred(), [DeformationHandle(2)]);
    }

    #[test]
    fn per_kind_dispatch_preserves_order() {
        let requests = [
            request(1, DeformationKind::Skinning, 100, 8, false),
            request(2, DeformationKind::Cloth, 100, 9, false),
            request(3, DeformationKind::Skinning, 100, 10, false),
        ];
        let plan = plan_deformations(&requests, BUDGET);
        assert_eq!(plan.count_of_kind(DeformationKind::Skinning), 2);
        // Dispatch order for skinning follows the global priority order: 3 then 1.
        assert_eq!(
            plan.handles_of_kind(DeformationKind::Skinning),
            [DeformationHandle(3), DeformationHandle(1)]
        );
        assert_eq!(
            plan.handles_of_kind(DeformationKind::Cloth),
            [DeformationHandle(2)]
        );
    }

    #[test]
    fn empty_requests_yield_empty_plan() {
        let plan = plan_deformations(&[], BUDGET);
        assert_eq!(plan.scheduled_count(), 0);
        assert_eq!(plan.vertices_used, 0);
        assert_eq!(plan.refits_used, 0);
    }

    #[test]
    fn zero_vertex_budget_still_admits_one_job() {
        let zero = DeformationBudget {
            vertices_per_frame: 0,
            blas_refits_per_frame: 0,
        };
        let requests = [
            request(1, DeformationKind::Hair, 10, 10, true),
            request(2, DeformationKind::Hair, 10, 9, false),
        ];
        let plan = plan_deformations(&requests, zero);
        assert_eq!(plan.scheduled_count(), 1);
        // No refit budget, so even the admitted job's refit defers.
        assert!(!plan.scheduled[0].refit_granted);
        assert_eq!(plan.refit_deferred(), [DeformationHandle(1)]);
        assert_eq!(plan.deferred, [DeformationHandle(2)]);
    }
}
