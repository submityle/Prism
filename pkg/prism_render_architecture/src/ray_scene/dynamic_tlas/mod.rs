//! Dynamic top-level acceleration structure (`TLAS`) update executor.
//!
//! [`super::acceleration`] decides *which* update a changed structure needs each
//! frame ([`AccelerationUpdate`]) and budgets rebuild spikes
//! ([`RebuildLedger`]), but it never touches a [`Tlas`]. This module is the
//! top-level twin of [`super::dynamic_blas`]: given the per-frame
//! [`GeometryChange`] and the moved instance transforms, it runs the chosen
//! update on an owned `TLAS` and returns a [`TlasUpdate`] receipt.
//!
//! It wires three already-verified pieces into one production path:
//!
//! - [`AccelerationUpdatePolicy::decide`] / `rebuild_variant` /
//!   [`should_rebuild_after_refit`](AccelerationUpdatePolicy::should_rebuild_after_refit)
//!   — the refit-vs-rebuild decision, plus the quality-driven escalation.
//! - [`Tlas::refit`] — the `O(nodes)` in-place bounds refit for instance motion
//!   with topology intact, and [`Tlas::refit_quality`] — the
//!   surface-area-heuristic (`SAH`) degradation signal fed back into the policy.
//! - [`Tlas::build`] / [`Tlas::rebuilt`] — the binned-`SAH` rebuild used for both
//!   [`AccelerationUpdate::Rebuild`] and [`AccelerationUpdate::BuildAndCompact`]
//!   (a flattened top-level array is contiguous by construction, so the rebuild
//!   *is* the compaction).
//!
//! Here a "primitive" is one [`Instance`]: the policy's moved-ratio and cost
//! accounting are expressed in instances, and `max_vertex_deformation` is the
//! largest instance displacement as a fraction of the scene's bounding radius.
//!
//! # Ownership
//!
//! The executor owns the `TLAS` (and therefore the reordered instance table) but
//! not the bottom-level pool: every update takes `blases: &[Bvh]`, the same pool
//! passed at construction, exactly like the underlying [`Tlas`] methods. This
//! keeps a single shared `BLAS` pool (which may itself be driven by
//! [`super::dynamic_blas`]) behind both the build and the per-frame refit.
//!
//! # Correctness vs. deferral
//!
//! Two update kinds are *mandatory* — skipping them renders an incorrect
//! structure — so they bypass the budget via [`RebuildLedger::force_admit`]:
//!
//! - A topology change ([`DynamicTlas::reinstance`]): the old node/instance
//!   tables no longer describe the scene, so a rebuild cannot be deferred.
//! - The bounds refit for moved instances: stale world-space boxes would miss or
//!   mis-report hits.
//!
//! A *motion-driven* rebuild (instance motion outgrew a cheap refit, but the
//! instance set is intact) is deferrable: if the ledger is exhausted the executor
//! falls back to a correctness-preserving refit this frame and keeps the rebuild
//! queued for a later frame via [`DynamicTlas::rebuild_pending`]. This bounds
//! per-frame rebuild cost without ever leaving the `TLAS` stale.

use super::acceleration::{
    update_scratch_bytes, AccelerationUpdate, AccelerationUpdatePolicy, GeometryChange,
    RebuildLedger,
};
use super::bvh::{Bvh, BvhBuildConfig};
use super::tlas::{Affine3, Instance, Tlas, TlasHit};
use super::traversal::Ray;

#[cfg(test)]
mod tests;

/// Receipt describing the update a [`DynamicTlas`] applied this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TlasUpdate {
    /// Update the policy resolved to *before* any budget fallback — i.e. the
    /// policy decision with the quality-driven escalation already applied. This
    /// is the work the executor intended to do.
    pub requested: AccelerationUpdate,
    /// Update actually executed on the `TLAS`. Equals [`Self::requested`] unless
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
    /// Measured post-refit [`Tlas::refit_quality`] when a refit ran, else `None`.
    /// `1.0` is an ideal tree; larger means topology has drifted from the
    /// instance layout.
    pub refit_quality: Option<f64>,
    /// Whether a quality-driven rebuild is now queued for a later frame.
    pub rebuild_pending: bool,
}

/// An owned top-level acceleration structure with per-frame dynamic updates.
///
/// Construct once over an instance set, then drive it each frame with
/// [`deform`](Self::deform) (instance set intact; transforms moved) or
/// [`reinstance`](Self::reinstance) (instance set changed; a fresh table). The
/// executor picks and runs the cheapest correct update under the supplied
/// [`AccelerationUpdatePolicy`] and [`RebuildLedger`].
#[derive(Clone, Debug)]
pub struct DynamicTlas {
    tlas: Tlas,
    policy: AccelerationUpdatePolicy,
    bytes_per_instance: u32,
    traversal_cost: f32,
    rebuild_pending: bool,
}

impl DynamicTlas {
    /// Builds a dynamic `TLAS` over `instances` referencing the `blases` pool.
    ///
    /// `bytes_per_instance` is the per-instance build-scratch estimate used to
    /// charge the [`RebuildLedger`] (see [`update_scratch_bytes`]); it does not
    /// affect the tree, only budgeting.
    #[must_use]
    pub fn new(
        instances: &[Instance],
        blases: &[Bvh],
        policy: AccelerationUpdatePolicy,
        bytes_per_instance: u32,
    ) -> Self {
        Self {
            tlas: Tlas::build(instances, blases),
            policy,
            bytes_per_instance,
            traversal_cost: BvhBuildConfig::default().traversal_cost,
            rebuild_pending: false,
        }
    }

    /// Read-only view of the current `TLAS`.
    #[must_use]
    pub const fn tlas(&self) -> &Tlas {
        &self.tlas
    }

    /// Instances in the current structure.
    #[must_use]
    pub fn instance_count(&self) -> usize {
        self.tlas.instances().len()
    }

    /// Whether a quality-driven rebuild is queued for a later frame.
    ///
    /// Set when the last refit degraded [`Tlas::refit_quality`] past the policy's
    /// escalation ratio, or when a motion-driven rebuild was deferred by the
    /// budget. Cleared once a rebuild runs.
    #[must_use]
    pub const fn rebuild_pending(&self) -> bool {
        self.rebuild_pending
    }

    /// Nearest intersection along the world-space `ray`, delegating to
    /// [`Tlas::closest_hit`]. `blases` must be the same pool the executor was
    /// built with.
    #[must_use]
    pub fn closest_hit(&self, ray: &Ray, blases: &[Bvh]) -> Option<TlasHit> {
        self.tlas.closest_hit(ray, blases)
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

    /// Per-instance scratch cost of `update` at the current instance count.
    fn cost_of(&self, update: AccelerationUpdate, instances: u32) -> u64 {
        update_scratch_bytes(update, instances, self.bytes_per_instance)
    }

    /// Current instance count as a saturating `u32` for cost accounting.
    fn instance_count_u32(&self) -> u32 {
        u32::try_from(self.tlas.instances().len()).unwrap_or(u32::MAX)
    }

    /// Applies an instance-set-intact frame: instances kept their identity but
    /// moved, with `transforms(instance_id)` returning each instance's new
    /// object→world [`Affine3`].
    ///
    /// The policy chooses [`Reuse`](AccelerationUpdate::Reuse),
    /// [`Refit`](AccelerationUpdate::Refit), or a rebuild from the amount of
    /// motion. A refit is mandatory (force-admitted) because stale world-space
    /// bounds are incorrect; a motion-driven rebuild is budgeted and falls back
    /// to a refit when the [`RebuildLedger`] is exhausted, retrying on a later
    /// frame. A singular new transform for an instance is ignored by
    /// [`Tlas::refit`], which keeps that instance's previous transform so a
    /// degenerate frame never corrupts the table.
    ///
    /// `change.topology_changed` must be `false` here; adding, removing, or
    /// reordering instances has no valid refit and must go through
    /// [`reinstance`](Self::reinstance), which supplies the new instance table.
    pub fn deform(
        &mut self,
        change: GeometryChange,
        transforms: impl Fn(u32) -> Affine3,
        blases: &[Bvh],
        ledger: &mut RebuildLedger,
    ) -> TlasUpdate {
        let requested = self.resolve(change);
        let instances = self.instance_count_u32();

        match requested {
            AccelerationUpdate::Reuse => TlasUpdate {
                requested,
                executed: AccelerationUpdate::Reuse,
                admitted: true,
                cost_bytes: 0,
                refit_quality: None,
                rebuild_pending: self.rebuild_pending,
            },
            AccelerationUpdate::Refit => {
                let cost = self.cost_of(AccelerationUpdate::Refit, instances);
                ledger.force_admit(cost);
                let quality = self.apply_refit(&transforms, blases);
                TlasUpdate {
                    requested,
                    executed: AccelerationUpdate::Refit,
                    admitted: true,
                    cost_bytes: cost,
                    refit_quality: Some(quality),
                    rebuild_pending: self.rebuild_pending,
                }
            }
            AccelerationUpdate::Rebuild | AccelerationUpdate::BuildAndCompact => {
                let cost = self.cost_of(requested, instances);
                if ledger.admit(cost) {
                    self.rebuild_moved(&transforms, blases);
                    self.rebuild_pending = false;
                    TlasUpdate {
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
                    let refit_cost = self.cost_of(AccelerationUpdate::Refit, instances);
                    ledger.force_admit(refit_cost);
                    let quality = self.apply_refit(&transforms, blases);
                    self.rebuild_pending = true;
                    TlasUpdate {
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

    /// Applies an instance-set-changing frame: `new_instances` fully replaces the
    /// current table (instances added, removed, or reordered).
    ///
    /// This always rebuilds — a changed instance set invalidates the existing
    /// node/instance tables, so the work is mandatory and force-admitted to the
    /// ledger (it cannot be deferred without rendering an incorrect structure).
    /// Fragmentation in `change` selects a plain [`Rebuild`](AccelerationUpdate::Rebuild)
    /// vs. [`BuildAndCompact`](AccelerationUpdate::BuildAndCompact) for cost
    /// reporting; both run the same compacting [`Tlas::build`].
    pub fn reinstance(
        &mut self,
        change: GeometryChange,
        new_instances: &[Instance],
        blases: &[Bvh],
        ledger: &mut RebuildLedger,
    ) -> TlasUpdate {
        let requested = self.policy.rebuild_variant(change.fragmentation);
        let instances = u32::try_from(new_instances.len()).unwrap_or(u32::MAX);
        let cost = self.cost_of(requested, instances);
        ledger.force_admit(cost);
        self.tlas = Tlas::build(new_instances, blases);
        self.rebuild_pending = false;
        TlasUpdate {
            requested,
            executed: requested,
            admitted: true,
            cost_bytes: cost,
            refit_quality: None,
            rebuild_pending: false,
        }
    }

    /// Refits bounds in place and returns the resulting [`Tlas::refit_quality`],
    /// updating [`Self::rebuild_pending`] from the policy's escalation ratio.
    fn apply_refit(&mut self, transforms: &impl Fn(u32) -> Affine3, blases: &[Bvh]) -> f64 {
        self.tlas.refit(transforms, blases);
        let quality = self.tlas.refit_quality(blases, self.traversal_cost);
        self.rebuild_pending = self.policy.should_rebuild_after_refit(quality);
        quality
    }

    /// Rebuilds the tree from the current instances moved by `transforms`.
    ///
    /// Applies the motion via [`Tlas::refit`] (which updates each instance's
    /// cached transform and skips singular frames) and then reconstructs a fresh
    /// compact hierarchy with [`Tlas::rebuilt`], preserving every instance's
    /// stable id.
    fn rebuild_moved(&mut self, transforms: &impl Fn(u32) -> Affine3, blases: &[Bvh]) {
        self.tlas.refit(transforms, blases);
        self.tlas = self.tlas.rebuilt(blases);
    }
}
