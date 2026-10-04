//! Dynamic scene orchestrator: one cohesive per-frame update across the whole
//! acceleration hierarchy.
//!
//! [`super::dynamic_blas`] drives a single bottom-level structure and
//! [`super::dynamic_tlas`] drives the top-level structure, but neither knows
//! about the other. A real frame updates *both*: skinned/deformed meshes refit
//! or rebuild their `BLAS`, and the `TLAS` must then be brought back into
//! agreement with the moved instances **and** the changed `BLAS` bounds. This
//! module owns that whole hierarchy and closes the loop with the one invariant
//! that ties the two levels together.
//!
//! # The cross-level invariant
//!
//! A [`Tlas`] leaf box is the world-space bound of an [`Instance`], computed from
//! `blases[inst.blas].bounds()` transformed by the instance's object→world
//! matrix. So a `BLAS` whose bounds changed this frame silently invalidates the
//! world box of **every instance that references it**, even when that instance
//! never moved. If the orchestrator let the `TLAS` take a cheap
//! [`Reuse`](AccelerationUpdate::Reuse) on such a frame, the stale top-level
//! boxes would miss or mis-report hits.
//!
//! Therefore: **whenever any `BLAS` in the pool was refit or rebuilt this frame,
//! the `TLAS` must at least refit**, re-reading the updated pool bounds — even if
//! no instance transform moved. [`DynamicScene`] enforces this automatically in
//! [`sync_tlas`](DynamicScene::sync_tlas) /
//! [`sync_tlas_bounds_only`](DynamicScene::sync_tlas_bounds_only): it tracks a
//! per-`BLAS` dirty set and, when the top-level policy decision for the caller's
//! [`GeometryChange`] would otherwise be `Reuse`, nudges it up to a `Refit`
//! (never further — a pure bounds refresh never needs a rebuild).
//!
//! # Ownership and the mirrored pool
//!
//! The orchestrator owns the `BLAS` executors ([`DynamicBlas`]) and the `TLAS`
//! executor ([`DynamicTlas`]). The `TLAS` methods all take `blases: &[Bvh]`, so
//! the orchestrator keeps a mirrored [`Bvh`] pool in lock-step with the
//! executors: when a `BLAS` update changes its tree, the matching pool slot is
//! refreshed from [`DynamicBlas::bvh`]. The mirror is only touched for `BLAS`
//! slots that actually changed, so a `Reuse` frame copies nothing.
//!
//! # Per-frame flow
//!
//! 1. For each changed mesh, call [`deform_blas`](DynamicScene::deform_blas) or
//!    [`retopology_blas`](DynamicScene::retopology_blas). Each marks its pool
//!    slot dirty when the executed update was not a `Reuse`.
//! 2. Bring the top level back into agreement with
//!    [`sync_tlas`](DynamicScene::sync_tlas) (instances moved) or
//!    [`sync_tlas_bounds_only`](DynamicScene::sync_tlas_bounds_only) (instances
//!    static, only `BLAS` bounds changed). Both honour the invariant above and
//!    then clear the dirty set.
//! 3. If the instance *set* changed, call
//!    [`reinstance`](DynamicScene::reinstance) instead; a full top-level rebuild
//!    re-reads the whole pool, so it also clears the dirty set.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use super::acceleration::{
    AccelerationUpdate, AccelerationUpdatePolicy, GeometryChange, RebuildLedger,
};
use super::bvh::{Bvh, Triangle};
use super::dynamic_blas::{BlasUpdate, DynamicBlas};
use super::dynamic_tlas::{DynamicTlas, TlasUpdate};
use super::tlas::{Affine3, Instance, Tlas, TlasHit};
use super::traversal::Ray;

#[cfg(test)]
mod tests;

/// Smallest positive deformation signal used to nudge a top-level `Reuse` up to
/// a `Refit` when a `BLAS` changed but no instance moved. It is far below the
/// policy's refit-deformation threshold, so it can never escalate past a refit.
const REFIT_NUDGE: f64 = 1.0e-6;

/// An owned acceleration hierarchy — a pool of bottom-level structures plus the
/// top-level structure over them — with one cohesive per-frame update.
///
/// Construct once from a set of triangle soups and an instance table, then drive
/// each frame with [`deform_blas`](Self::deform_blas) /
/// [`retopology_blas`](Self::retopology_blas) for the meshes followed by
/// [`sync_tlas`](Self::sync_tlas) / [`sync_tlas_bounds_only`](Self::sync_tlas_bounds_only)
/// for the top level. The orchestrator guarantees the top level never observes a
/// stale `BLAS` bound.
#[derive(Clone, Debug)]
pub struct DynamicScene {
    blases: Vec<DynamicBlas>,
    pool: Vec<Bvh>,
    tlas: DynamicTlas,
    policy: AccelerationUpdatePolicy,
    dirty: Vec<bool>,
}

impl DynamicScene {
    /// Builds a scene over `soups` (one triangle soup per `BLAS`) and
    /// `instances` (each referencing a `BLAS` by index into `soups`).
    ///
    /// `bytes_per_primitive` and `bytes_per_instance` are the build-scratch
    /// estimates charged to the [`RebuildLedger`] for bottom-level and top-level
    /// rebuilds respectively (see [`DynamicBlas::new`] / [`DynamicTlas::new`]);
    /// they budget work and do not affect the trees.
    #[must_use]
    pub fn new(
        soups: &[&[Triangle]],
        instances: &[Instance],
        policy: AccelerationUpdatePolicy,
        bytes_per_primitive: u32,
        bytes_per_instance: u32,
    ) -> Self {
        let blases: Vec<DynamicBlas> = soups
            .iter()
            .map(|soup| DynamicBlas::new(soup, policy, bytes_per_primitive))
            .collect();
        let pool: Vec<Bvh> = blases.iter().map(|b| b.bvh().clone()).collect();
        let dirty = alloc::vec![false; blases.len()];
        let tlas = DynamicTlas::new(instances, &pool, policy, bytes_per_instance);
        Self {
            blases,
            pool,
            tlas,
            policy,
            dirty,
        }
    }

    /// Number of bottom-level structures in the pool.
    #[must_use]
    pub fn blas_count(&self) -> usize {
        self.blases.len()
    }

    /// Number of instances in the top-level structure.
    #[must_use]
    pub fn instance_count(&self) -> usize {
        self.tlas.instance_count()
    }

    /// Read-only view of one bottom-level executor.
    #[must_use]
    pub fn blas(&self, index: usize) -> &DynamicBlas {
        &self.blases[index]
    }

    /// Read-only view of the top-level executor.
    #[must_use]
    pub const fn dynamic_tlas(&self) -> &DynamicTlas {
        &self.tlas
    }

    /// Read-only view of the underlying [`Tlas`].
    #[must_use]
    pub const fn tlas(&self) -> &Tlas {
        self.tlas.tlas()
    }

    /// Mirrored bottom-level [`Bvh`] pool, in `BLAS`-index order. Kept in
    /// lock-step with the executors.
    #[must_use]
    pub fn blas_pool(&self) -> &[Bvh] {
        &self.pool
    }

    /// Whether any pool slot changed since the last top-level sync and is still
    /// awaiting a `TLAS` refit/rebuild.
    #[must_use]
    pub fn has_dirty_blas(&self) -> bool {
        self.dirty.iter().any(|&d| d)
    }

    /// Nearest world-space intersection, delegating to [`Tlas::closest_hit`] over
    /// the mirrored pool.
    #[must_use]
    pub fn closest_hit(&self, ray: &Ray) -> Option<TlasHit> {
        self.tlas.closest_hit(ray, &self.pool)
    }

    /// Applies a topology-intact frame to one `BLAS`: its primitives kept their
    /// connectivity but moved, with `positions(primitive_id)` returning each
    /// triangle's new `[v0, v1, v2]`. See [`DynamicBlas::deform`].
    ///
    /// Refreshes the mirrored pool slot and marks it dirty when the executed
    /// update changed the tree (anything but a `Reuse`), so the next top-level
    /// sync re-reads its bounds.
    pub fn deform_blas(
        &mut self,
        index: usize,
        change: GeometryChange,
        positions: impl Fn(u32) -> [[f32; 3]; 3],
        ledger: &mut RebuildLedger,
    ) -> BlasUpdate {
        let receipt = self.blases[index].deform(change, positions, ledger);
        self.absorb_blas_update(index, receipt.executed);
        receipt
    }

    /// Applies a topology-changing frame to one `BLAS`: `new_triangles` fully
    /// replaces its soup. See [`DynamicBlas::retopology`]. Always refreshes the
    /// mirrored pool slot and marks it dirty.
    pub fn retopology_blas(
        &mut self,
        index: usize,
        change: GeometryChange,
        new_triangles: &[Triangle],
        ledger: &mut RebuildLedger,
    ) -> BlasUpdate {
        let receipt = self.blases[index].retopology(change, new_triangles, ledger);
        self.absorb_blas_update(index, receipt.executed);
        receipt
    }

    /// Refreshes the mirrored pool slot for `index` and records whether its tree
    /// changed this frame.
    fn absorb_blas_update(&mut self, index: usize, executed: AccelerationUpdate) {
        if executed != AccelerationUpdate::Reuse {
            self.pool[index] = self.blases[index].bvh().clone();
            self.dirty[index] = true;
        }
    }

    /// Brings the top level back into agreement with the moved instances and any
    /// changed `BLAS` bounds, with `transforms(instance_id)` returning each
    /// instance's new object→world [`Affine3`]. See [`DynamicTlas::deform`].
    ///
    /// Enforces the cross-level invariant: if a `BLAS` changed this frame but the
    /// caller's `change` would resolve to a top-level `Reuse`, the update is
    /// nudged up to a `Refit` so stale world boxes are re-read from the updated
    /// pool. A `change` that already resolves to a refit or rebuild re-reads the
    /// pool on its own and is passed through unchanged. The dirty set is cleared
    /// afterwards.
    pub fn sync_tlas(
        &mut self,
        change: GeometryChange,
        transforms: impl Fn(u32) -> Affine3,
        ledger: &mut RebuildLedger,
    ) -> TlasUpdate {
        let effective = self.effective_change(change);
        let receipt = self.tlas.deform(effective, transforms, &self.pool, ledger);
        self.clear_dirty();
        receipt
    }

    /// Convenience sync for frames where **only** `BLAS` bounds changed and no
    /// instance moved: reuses each instance's current object→world transform so
    /// the refit re-reads the updated pool without altering placement.
    ///
    /// Honours the same invariant as [`sync_tlas`](Self::sync_tlas).
    pub fn sync_tlas_bounds_only(
        &mut self,
        change: GeometryChange,
        ledger: &mut RebuildLedger,
    ) -> TlasUpdate {
        let snapshot: BTreeMap<u32, Affine3> = self
            .tlas
            .tlas()
            .instances()
            .iter()
            .map(|inst| (inst.instance_id(), inst.object_to_world()))
            .collect();
        let effective = self.effective_change(change);
        let receipt = self
            .tlas
            .deform(effective, |id| snapshot[&id], &self.pool, ledger);
        self.clear_dirty();
        receipt
    }

    /// Applies an instance-set-changing frame: `new_instances` fully replaces the
    /// top-level table (instances added, removed, or reordered). See
    /// [`DynamicTlas::reinstance`].
    ///
    /// A full top-level rebuild re-reads the entire pool, so this also satisfies
    /// the cross-level invariant and clears the dirty set.
    pub fn reinstance(
        &mut self,
        change: GeometryChange,
        new_instances: &[Instance],
        ledger: &mut RebuildLedger,
    ) -> TlasUpdate {
        let receipt = self
            .tlas
            .reinstance(change, new_instances, &self.pool, ledger);
        self.clear_dirty();
        receipt
    }

    /// Count of instances whose referenced `BLAS` is dirty this frame.
    fn affected_instances(&self) -> usize {
        self.tlas
            .tlas()
            .instances()
            .iter()
            .filter(|inst| self.dirty.get(inst.blas()).copied().unwrap_or(false))
            .count()
    }

    /// Resolves the effective top-level change, upgrading a would-be `Reuse` to a
    /// minimal `Refit` when a `BLAS` changed this frame (the cross-level
    /// invariant). A change that already resolves to a refit or rebuild — or a
    /// top level that already has a rebuild queued — re-reads the pool on its own
    /// and is returned unchanged.
    fn effective_change(&self, change: GeometryChange) -> GeometryChange {
        if self.affected_instances() == 0 || self.tlas.rebuild_pending() {
            return change;
        }
        match self.policy.decide(change) {
            AccelerationUpdate::Reuse => GeometryChange {
                moved_primitives: 0,
                max_vertex_deformation: change.max_vertex_deformation.max(REFIT_NUDGE),
                topology_changed: false,
                ..change
            },
            AccelerationUpdate::Refit
            | AccelerationUpdate::Rebuild
            | AccelerationUpdate::BuildAndCompact => change,
        }
    }

    /// Clears the per-`BLAS` dirty set after a top-level sync has re-read the
    /// pool.
    fn clear_dirty(&mut self) {
        for d in &mut self.dirty {
            *d = false;
        }
    }
}
