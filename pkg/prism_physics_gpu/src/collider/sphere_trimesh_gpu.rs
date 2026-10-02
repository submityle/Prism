//! Device twin of [`cpu_sphere_trimesh_collide`](super::cpu_sphere_trimesh_collide):
//! the sphere-versus-trimesh collider running both heavy phases on the GPU.
//!
//! [`GpuSphereTrimeshCollider`] composes the two already device-twinned stages
//! of the collider pipeline:
//!
//! 1. [`GpuBvhOverlap::query`] runs the `LBVH` broad phase on-device, returning
//!    the candidate triangle indices each sphere's padded box overlaps.
//! 2. [`GpuSphereTriangleNarrowphase::query`] runs the per-pair sphere-triangle
//!    test on-device for every surfaced candidate.
//!
//! The only host-side work left is assembling the broad-phase hits into the
//! pair batch the narrow phase consumes and reducing each sphere's contacts to
//! the single deepest one. Both steps are pure, deterministic selection: the
//! candidate triangles are sorted ascending and the reduction keeps a contact
//! only when it is strictly deeper, so ties resolve to the smallest triangle
//! index — identical to the CPU golden. Because each GPU stage is a bit-for-bit
//! twin of its CPU counterpart and the glue is the same ordering and reduction,
//! the composed result matches [`cpu_sphere_trimesh_collide`] to the tolerance
//! the narrow phase's square root and reciprocals impose, which the real-device
//! parity test asserts.
//!
//! # Why host-side glue rather than one fused kernel
//!
//! Fusing broad phase, narrow phase, and reduction into a single kernel would
//! save two buffer round-trips but couple three independently verified pieces
//! into one unproven shader. This slice keeps each verified kernel intact and
//! composes them, so the device twin is correct by construction; a fused kernel
//! is a later optimisation that must prove the same parity.
//!
//! Provenance: pipeline composition only; the reused kernels cite their own
//! sources. No Unreal Engine source or derived code.

use crate::broadphase::Particle;
use crate::bvh::{GpuBvhOverlap, Lbvh, OverlapQueryError};
use crate::context::GpuContext;
use crate::narrowphase::{Contact, GpuSphereTriangleNarrowphase, SphereTrianglePair, Triangle};

use super::Trimesh;

/// A reusable sphere-versus-trimesh collider that runs the broad and narrow
/// phases on the GPU.
///
/// Holds the two sub-pipelines so repeated frames reuse their compiled shaders
/// and bind-group layouts; build it once per [`GpuContext`] and call
/// [`collide`](GpuSphereTrimeshCollider::collide) each frame.
pub struct GpuSphereTrimeshCollider {
    /// On-device `LBVH` overlap broad phase.
    overlap: GpuBvhOverlap,
    /// On-device sphere-triangle narrow phase.
    narrowphase: GpuSphereTriangleNarrowphase,
}

impl GpuSphereTrimeshCollider {
    /// Builds the collider's sub-pipelines on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSphereTrimeshCollider {
        GpuSphereTrimeshCollider {
            overlap: GpuBvhOverlap::new(ctx),
            narrowphase: GpuSphereTriangleNarrowphase::new(ctx),
        }
    }

    /// Collides a batch of spheres against `mesh` on the GPU, returning the
    /// single deepest contact per sphere in sphere order.
    ///
    /// `lbvh` must be the hierarchy built from `mesh.triangle_aabbs()`; the
    /// overlap query returns original triangle indices that index `mesh`.
    /// `capacity_per_sphere` bounds the broad-phase candidate count per sphere
    /// and mirrors the device output buffer's fixed per-query region.
    ///
    /// The result matches [`cpu_sphere_trimesh_collide`](super::cpu_sphere_trimesh_collide):
    /// a sphere that touches no triangle (or only non-penetrated ones) reports
    /// [`None`]; a contact's `a` is the sphere index, `b` the winning triangle
    /// index, and the normal points from the triangle toward the sphere.
    ///
    /// # Errors
    ///
    /// Returns [`OverlapQueryError::CapacityExceeded`] when a sphere overlaps
    /// more triangle boxes than `capacity_per_sphere` allows.
    pub fn collide(
        &self,
        ctx: &GpuContext,
        mesh: &Trimesh,
        lbvh: &Lbvh,
        spheres: &[Particle],
        capacity_per_sphere: u32,
    ) -> Result<Vec<Option<Contact>>, OverlapQueryError> {
        // Materialise every triangle once; pair indices address this slice and
        // match the original triangle indices the overlap query reports.
        let triangles: Vec<Triangle> = (0..mesh.triangle_count())
            .map(|i| mesh.triangle(i))
            .collect();

        // Broad phase on-device: one padded box per sphere.
        let queries: Vec<crate::bvh::Aabb> = spheres
            .iter()
            .map(|s| {
                let r = glam::Vec3::splat(s.radius);
                crate::bvh::Aabb::new(s.position - r, s.position + r)
            })
            .collect();
        let candidates = self
            .overlap
            .query(ctx, lbvh, &queries, capacity_per_sphere)?;

        // Assemble the pair batch, each sphere's candidates sorted ascending so
        // the reduction's strict-deeper rule breaks ties toward the smallest
        // triangle index, matching the CPU golden exactly.
        let mut pairs: Vec<SphereTrianglePair> = Vec::new();
        let mut group_len: Vec<usize> = Vec::with_capacity(spheres.len());
        for (sphere_index, hits) in candidates.iter().enumerate() {
            let mut tris: Vec<u32> = hits.clone();
            tris.sort_unstable();
            group_len.push(tris.len());
            let sphere_u32 = u32::try_from(sphere_index).unwrap_or(u32::MAX);
            for tri in tris {
                pairs.push(SphereTrianglePair::new(sphere_u32, tri));
            }
        }

        // Narrow phase on-device over the full pair batch.
        let contacts = self.narrowphase.query(ctx, spheres, &triangles, &pairs);

        // Reduce each sphere's group to its deepest contact (strict-deeper keeps
        // the smaller triangle index on ties, matching the CPU golden).
        let mut out: Vec<Option<Contact>> = Vec::with_capacity(spheres.len());
        let mut cursor = 0usize;
        for len in group_len {
            let mut best: Option<Contact> = None;
            for c in contacts[cursor..cursor + len].iter().flatten() {
                match best {
                    Some(b) if c.depth <= b.depth => {}
                    _ => best = Some(*c),
                }
            }
            out.push(best);
            cursor += len;
        }

        Ok(out)
    }
}
