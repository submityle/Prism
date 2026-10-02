//! GPU twin of the capsule-versus-trimesh collider.
//!
//! Runs the same three-stage pipeline as [`cpu_capsule_trimesh_collide`](super::cpu_capsule_trimesh_collide)
//! — `LBVH` broad phase, capsule-triangle narrow phase, deepest-contact
//! reduction — but with the broad and narrow phases executing on the device.
//! Mirrors [`GpuSphereTrimeshCollider`](super::GpuSphereTrimeshCollider)
//! exactly, swapping the sphere proxy for a [`Capsule`].
//!
//! The pair assembly ([`build_pairs`](super::capsule_trimesh::build_pairs)), the
//! broad-phase box ([`capsule_aabb`](super::capsule_trimesh::capsule_aabb)), and
//! the deepest-contact reduction ([`deepest_per_group`](super::reduce::deepest_per_group))
//! are the very functions the CPU golden calls, so this twin is correct by
//! construction and matches the golden lane for lane; only the two sub-phases
//! move to the GPU. Fusing them into one kernel would save buffer round-trips
//! but couple independently verified pieces into one unproven shader, so this
//! slice keeps each verified kernel intact and composes them.
//!
//! Provenance: pipeline composition only; the reused kernels cite their own
//! sources. No Unreal Engine source or derived code.

use crate::bvh::{Aabb, GpuBvhOverlap, Lbvh, OverlapQueryError};
use crate::context::GpuContext;
use crate::narrowphase::{Capsule, Contact, GpuCapsuleTriangleNarrowphase, Triangle};

use super::capsule_trimesh::{build_pairs, capsule_aabb};
use super::reduce::deepest_per_group;
use super::Trimesh;

/// A reusable capsule-versus-trimesh collider that runs the broad and narrow
/// phases on the GPU.
///
/// Holds the two sub-pipelines so repeated frames reuse their compiled shaders
/// and bind-group layouts; build it once per [`GpuContext`] and call
/// [`collide`](GpuCapsuleTrimeshCollider::collide) each frame.
pub struct GpuCapsuleTrimeshCollider {
    /// On-device `LBVH` overlap broad phase.
    overlap: GpuBvhOverlap,
    /// On-device capsule-triangle narrow phase.
    narrowphase: GpuCapsuleTriangleNarrowphase,
}

impl GpuCapsuleTrimeshCollider {
    /// Builds the collider's sub-pipelines on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuCapsuleTrimeshCollider {
        GpuCapsuleTrimeshCollider {
            overlap: GpuBvhOverlap::new(ctx),
            narrowphase: GpuCapsuleTriangleNarrowphase::new(ctx),
        }
    }

    /// Collides a batch of capsules against `mesh` on the GPU, returning the
    /// single deepest contact per capsule in capsule order.
    ///
    /// `lbvh` must be the hierarchy built from `mesh.triangle_aabbs()`; the
    /// overlap query returns original triangle indices that index `mesh`.
    /// `capacity_per_capsule` bounds the broad-phase candidate count per capsule
    /// and mirrors the device output buffer's fixed per-query region.
    ///
    /// The result matches [`cpu_capsule_trimesh_collide`](super::cpu_capsule_trimesh_collide):
    /// a capsule that touches no triangle (or only non-penetrated ones) reports
    /// [`None`]; a contact's `a` is the capsule index, `b` the winning triangle
    /// index, and the normal points from the triangle toward the capsule.
    ///
    /// # Errors
    ///
    /// Returns [`OverlapQueryError::CapacityExceeded`] when a capsule overlaps
    /// more triangle boxes than `capacity_per_capsule` allows.
    pub fn collide(
        &self,
        ctx: &GpuContext,
        mesh: &Trimesh,
        lbvh: &Lbvh,
        capsules: &[Capsule],
        capacity_per_capsule: u32,
    ) -> Result<Vec<Option<Contact>>, OverlapQueryError> {
        // Materialise every triangle once; pair indices address this slice and
        // match the original triangle indices the overlap query reports.
        let triangles: Vec<Triangle> = (0..mesh.triangle_count())
            .map(|i| mesh.triangle(i))
            .collect();

        // Broad phase on-device: one swept-segment box per capsule, assembled by
        // the same helper the CPU golden uses.
        let queries: Vec<Aabb> = capsules.iter().map(capsule_aabb).collect();
        let candidates = self
            .overlap
            .query(ctx, lbvh, &queries, capacity_per_capsule)?;

        // Assemble the pair batch exactly as the CPU golden does (each group
        // sorted ascending by triangle index for the tie-break).
        let (pairs, group_len) = build_pairs(&candidates);

        // Narrow phase on-device over the full pair batch.
        let contacts = self.narrowphase.query(ctx, capsules, &triangles, &pairs);

        // Collapse each capsule's group to its deepest contact through the shared
        // reduction, matching the CPU golden lane for lane.
        Ok(deepest_per_group(&contacts, &group_len))
    }
}
