//! Dynamic bottom-level acceleration structure (`BLAS`) update executor.
//!
//! [`super::acceleration`] decides *which* update a changed structure needs each
//! frame ([`AccelerationUpdate`]) and budgets rebuild spikes
//! ([`RebuildLedger`]), but it never touches a [`Bvh`]. This module is the
//! executor that closes that loop: given the per-frame [`GeometryChange`] and the
//! moved geometry, it runs the chosen update on an owned `BLAS` and returns a
//! [`BlasUpdate`] receipt.
//!
//! It wires three already-verified pieces into one production path:
//!
//! - [`AccelerationUpdatePolicy::decide`] / `rebuild_variant` /
//!   [`should_rebuild_after_refit`](AccelerationUpdatePolicy::should_rebuild_after_refit)
//!   — the refit-vs-rebuild decision, plus the quality-driven escalation.
//! - [`Bvh::refit`] — the `O(nodes)` in-place bounds refit for topology-intact
//!   motion, and [`Bvh::refit_quality`] — the surface-area-heuristic (`SAH`)
//!   degradation signal fed back into the policy.
//! - [`LinearBvh::build`] — the parallel linear-`BVH` (`LBVH`) rebuild used for
//!   both [`AccelerationUpdate::Rebuild`] and
//!   [`AccelerationUpdate::BuildAndCompact`] (a flattened linear `BVH` is
//!   contiguous by construction, so the rebuild *is* the compaction).
//!
//! # Correctness vs. deferral
//!
//! Two update kinds are *mandatory* — skipping them renders an incorrect
//! structure — so they bypass the budget via
//! [`RebuildLedger::force_admit`]:
//!
//! - A topology change ([`DynamicBlas::retopology`]): the old node/primitive
//!   tables no longer describe the geometry, so a rebuild cannot be deferred.
//! - The bounds refit for moved geometry: stale bounds would miss or mis-report
//!   hits.
//!
//! A *motion-driven* rebuild (deformation outgrew a cheap refit, but topology is
//! intact) is deferrable: if the ledger is exhausted the executor falls back to a
//! correctness-preserving refit this frame and keeps the rebuild queued for a
//! later frame via [`DynamicBlas::rebuild_pending`]. This bounds per-frame
//! rebuild cost without ever leaving the `BLAS` stale.

use super::acceleration::{
    AccelerationUpdate, AccelerationUpdatePolicy, GeometryChange, RebuildLedger,
    update_scratch_bytes,
};
use super::bvh::{Bvh, BvhBuildConfig, Triangle};
use super::lbvh::LinearBvh;

#[cfg(test)]
mod tests;

/// Receipt describing the update a [`DynamicBlas`] applied this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlasUpdate {
    /// Update the policy resolved to *before* any budget fallback — i.e. the
    /// policy decision with the quality-driven escalation already applied. This
    /// is the work the executor intended to do.
    pub requested: AccelerationUpdate,
    /// Update actually executed on the `BLAS`. Equals [`Self::requested`] unless
    /// a motion-driven rebuild was deferred by the budget, in which case it is
    /// [`AccelerationUpdate::Refit`].
    pub executed: AccelerationUpdate,
    /// Whether the rebuild ledger admitted the requested cost. `false` means a
    /// motion-driven rebuild was deferred and a fallback refit ran instead;
    /// mandatory (`force_admit`) work always reports `true`.
    pub admitted: bool,
    /// Build-scratch bytes charged to the ledger for the executed update, from
    /// [`update_scratch_bytes`].
    pub cost_bytes: u64,
    /// Measured post-refit [`Bvh::refit_quality`] when a refit ran, else `None`.
    /// `1.0` is an ideal tree; larger means topology has drifted from geometry.
    pub refit_quality: Option<f64>,
    /// Whether a quality-driven rebuild is now queued for a later frame.
    pub rebuild_pending: bool,
}

/// An owned bottom-level acceleration structure with per-frame dynamic updates.
///
/// Construct once over a triangle soup, then drive it each frame with
/// [`deform`](Self::deform) (topology intact; positions moved) or
/// [`retopology`](Self::retopology) (topology changed; a fresh triangle soup).
/// The executor picks and runs the cheapest correct update under the supplied
/// [`AccelerationUpdatePolicy`] and [`RebuildLedger`].
#[derive(Clone, Debug)]
pub struct DynamicBlas {
    bvh: Bvh,
    policy: AccelerationUpdatePolicy,
    bytes_per_primitive: u32,
    traversal_cost: f32,
    rebuild_pending: bool,
}

impl DynamicBlas {
    /// Builds a dynamic `BLAS` over `triangles` using the parallel `LBVH`
    /// builder.
    ///
    /// `bytes_per_primitive` is the per-primitive build-scratch estimate used to
    /// charge the [`RebuildLedger`] (see [`update_scratch_bytes`]); it does not
    /// affect the tree, only budgeting.
    #[must_use]
    pub fn new(
        triangles: &[Triangle],
        policy: AccelerationUpdatePolicy,
        bytes_per_primitive: u32,
    ) -> Self {
        Self {
            bvh: LinearBvh::build(triangles),
            policy,
            bytes_per_primitive,
            traversal_cost: BvhBuildConfig::default().traversal_cost,
            rebuild_pending: false,
        }
    }

    /// Read-only view of the current acceleration structure.
    #[must_use]
    pub fn bvh(&self) -> &Bvh {
        &self.bvh
    }

    /// Primitives in the current structure.
    #[must_use]
    pub fn primitive_count(&self) -> usize {
        self.bvh.primitive_count()
    }

    /// Whether a quality-driven rebuild is queued for a later frame.
    ///
    /// Set when the last refit degraded [`Bvh::refit_quality`] past the policy's
    /// escalation ratio, or when a motion-driven rebuild was deferred by the
    /// budget. Cleared once a rebuild runs.
    #[must_use]
    pub const fn rebuild_pending(&self) -> bool {
        self.rebuild_pending
    }

    /// Resolves the policy decision for `change`, escalating a `Reuse`/`Refit`
    /// result to a rebuild when a quality-driven rebuild is already queued.
    fn resolve(&self, change: GeometryChange) -> AccelerationUpdate {
        let base = self.policy.decide(change);
        if self.rebuild_pending && !base.is_rebuild() {
            self.policy.rebuild_variant(change.fragmentation)
        } else {
            base
        }
    }

    /// Per-primitive scratch cost of `update` at the current primitive count.
    fn cost_of(&self, update: AccelerationUpdate, prims: u32) -> u64 {
        update_scratch_bytes(update, prims, self.bytes_per_primitive)
    }

    /// Current primitive count as a saturating `u32` for cost accounting.
    fn prim_count_u32(&self) -> u32 {
        u32::try_from(self.bvh.primitive_count()).unwrap_or(u32::MAX)
    }

    /// Applies a topology-intact frame: primitives kept their connectivity but
    /// moved, with `positions(primitive_id)` returning each triangle's new
    /// `[v0, v1, v2]`.
    ///
    /// The policy chooses [`Reuse`](AccelerationUpdate::Reuse),
    /// [`Refit`](AccelerationUpdate::Refit), or a rebuild from the amount of
    /// motion. A refit is mandatory (force-admitted) because stale bounds are
    /// incorrect; a motion-driven rebuild is budgeted and falls back to a refit
    /// when the [`RebuildLedger`] is exhausted, retrying on a later frame.
    ///
    /// `change.topology_changed` must be `false` here; a true topology change has
    /// no valid refit and must go through [`retopology`](Self::retopology), which
    /// supplies the new triangle soup.
    pub fn deform(
        &mut self,
        change: GeometryChange,
        positions: impl Fn(u32) -> [[f32; 3]; 3],
        ledger: &mut RebuildLedger,
    ) -> BlasUpdate {
        let requested = self.resolve(change);
        let prims = self.prim_count_u32();

        match requested {
            AccelerationUpdate::Reuse => BlasUpdate {
                requested,
                executed: AccelerationUpdate::Reuse,
                admitted: true,
                cost_bytes: 0,
                refit_quality: None,
                rebuild_pending: self.rebuild_pending,
            },
            AccelerationUpdate::Refit => {
                let cost = self.cost_of(AccelerationUpdate::Refit, prims);
                ledger.force_admit(cost);
                let quality = self.apply_refit(&positions);
                BlasUpdate {
                    requested,
                    executed: AccelerationUpdate::Refit,
                    admitted: true,
                    cost_bytes: cost,
                    refit_quality: Some(quality),
                    rebuild_pending: self.rebuild_pending,
                }
            }
            AccelerationUpdate::Rebuild | AccelerationUpdate::BuildAndCompact => {
                let cost = self.cost_of(requested, prims);
                if ledger.admit(cost) {
                    self.rebuild_deformed(&positions);
                    self.rebuild_pending = false;
                    BlasUpdate {
                        requested,
                        executed: requested,
                        admitted: true,
                        cost_bytes: cost,
                        refit_quality: None,
                        rebuild_pending: false,
                    }
                } else {
                    // Budget exhausted: a motion-driven rebuild is deferrable, so
                    // keep the structure correct with a refit now and retry the
                    // rebuild on a later frame.
                    let refit_cost = self.cost_of(AccelerationUpdate::Refit, prims);
                    ledger.force_admit(refit_cost);
                    let quality = self.apply_refit(&positions);
                    self.rebuild_pending = true;
                    BlasUpdate {
                        requested,
                        executed: AccelerationUpdate::Refit,
                        admitted: false,
                        cost_bytes: refit_cost,
                        refit_quality: Some(quality),
                        rebuild_pending: true,
                    }
                }
            }
        }
    }

    /// Applies a topology-changing frame: `new_triangles` fully replaces the
    /// current soup (primitives added, removed, or reordered).
    ///
    /// This always rebuilds — a topology change invalidates the existing
    /// node/primitive tables, so the work is mandatory and force-admitted to the
    /// ledger (it cannot be deferred without rendering an incorrect structure).
    /// Fragmentation in `change` selects a plain [`Rebuild`](AccelerationUpdate::Rebuild)
    /// vs. [`BuildAndCompact`](AccelerationUpdate::BuildAndCompact) for cost
    /// reporting; both run the same compacting `LBVH` build.
    pub fn retopology(
        &mut self,
        change: GeometryChange,
        new_triangles: &[Triangle],
        ledger: &mut RebuildLedger,
    ) -> BlasUpdate {
        let requested = self.policy.rebuild_variant(change.fragmentation);
        let prims = u32::try_from(new_triangles.len()).unwrap_or(u32::MAX);
        let cost = self.cost_of(requested, prims);
        ledger.force_admit(cost);
        self.bvh = LinearBvh::build(new_triangles);
        self.rebuild_pending = false;
        BlasUpdate {
            requested,
            executed: requested,
            admitted: true,
            cost_bytes: cost,
            refit_quality: None,
            rebuild_pending: false,
        }
    }

    /// Refits bounds in place and returns the resulting [`Bvh::refit_quality`],
    /// updating [`Self::rebuild_pending`] from the policy's escalation ratio.
    fn apply_refit(&mut self, positions: &impl Fn(u32) -> [[f32; 3]; 3]) -> f64 {
        self.bvh.refit(positions);
        let quality = self.bvh.refit_quality(self.traversal_cost);
        self.rebuild_pending = self.policy.should_rebuild_after_refit(quality);
        quality
    }

    /// Rebuilds the tree from the current primitives moved by `positions`,
    /// preserving each primitive's stable id.
    fn rebuild_deformed(&mut self, positions: &impl Fn(u32) -> [[f32; 3]; 3]) {
        let updated: Vec<Triangle> = self
            .bvh
            .primitives()
            .iter()
            .map(|tri| {
                let [v0, v1, v2] = positions(tri.primitive);
                Triangle::new(v0, v1, v2, tri.primitive)
            })
            .collect();
        self.bvh = LinearBvh::build(&updated);
    }
}
