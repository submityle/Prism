//! Capsule-versus-trimesh collision: the aggregation layer that turns a mesh,
//! its `LBVH`, and a batch of capsules into one contact manifold per capsule.
//!
//! This is the CPU golden for capsule-against-triangle-mesh collision and the
//! bit-for-bit reference the device twin matches. It mirrors
//! [`cpu_sphere_trimesh_collide`](super::cpu_sphere_trimesh_collide) exactly,
//! swapping the sphere proxy for a [`Capsule`] (a segment swept by a radius):
//!
//! 1. **Broad phase** — each capsule's padded bounding box queries the mesh
//!    `LBVH`. The box bounds the whole swept segment:
//!    `[min(p0, p1) - r, max(p0, p1) + r]`.
//! 2. **Narrow phase** — every surfaced `(capsule, triangle)` pair runs the
//!    shared [`cpu_capsule_triangle_narrowphase`](crate::cpu_capsule_triangle_narrowphase)
//!    test.
//! 3. **Reduction** — the per-capsule contacts collapse to the single deepest
//!    one, with ties resolved to the smallest triangle index.
//!
//! The determinism argument, tie-break rule, and output contract are identical
//! to the sphere collider: candidates are tested in ascending triangle index
//! and a contact replaces the current best only when it is strictly deeper, so
//! the device twin can match the reduction lane for lane. A capsule in contact
//! reports a [`Contact`] whose `a` is the capsule index, `b` the winning
//! triangle index, and whose normal points from the triangle toward the
//! capsule.
//!
//! Provenance: broad-phase-then-narrow-phase mesh collision with a deepest-point
//! reduction is textbook; the reused pieces cite their own sources. No Unreal
//! Engine source or derived code.

use crate::bvh::{cpu_bvh_aabb_overlap, Aabb, Lbvh, OverlapQueryError};
use crate::narrowphase::{
    cpu_capsule_triangle_narrowphase, Capsule, CapsuleTrianglePair, Contact, Triangle,
};

use super::reduce::deepest_per_group;

/// Collides a batch of capsules against a static triangle mesh, returning the
/// single deepest contact per capsule.
///
/// `lbvh` must be the hierarchy built from `mesh.triangle_aabbs()`;
/// `capacity_per_capsule` bounds the broad-phase candidate count per capsule
/// and mirrors the device output buffer's fixed per-query region.
///
/// # Errors
///
/// Returns [`OverlapQueryError::CapacityExceeded`] (naming the offending
/// capsule, since there is one query per capsule) when a capsule's bounding box
/// overlaps more triangle boxes than `capacity_per_capsule` allows.
pub fn cpu_capsule_trimesh_collide(
    mesh: &super::Trimesh,
    lbvh: &Lbvh,
    capsules: &[Capsule],
    capacity_per_capsule: u32,
) -> Result<Vec<Option<Contact>>, OverlapQueryError> {
    let triangles: Vec<Triangle> = (0..mesh.triangle_count())
        .map(|i| mesh.triangle(i))
        .collect();

    let queries: Vec<Aabb> = capsules.iter().map(capsule_aabb).collect();

    let candidates = cpu_bvh_aabb_overlap(lbvh, &queries, capacity_per_capsule)?;

    let (pairs, group_len) = build_pairs(&candidates);

    let contacts = cpu_capsule_triangle_narrowphase(capsules, &triangles, &pairs);

    Ok(deepest_per_group(&contacts, &group_len))
}

/// The axis-aligned bounding box of a capsule: the swept segment padded by the
/// radius on every side.
pub(super) fn capsule_aabb(cap: &Capsule) -> Aabb {
    let lo = cap.p0.min(cap.p1) - glam::Vec3::splat(cap.radius);
    let hi = cap.p0.max(cap.p1) + glam::Vec3::splat(cap.radius);
    Aabb::new(lo, hi)
}

/// Flattens the per-query broad-phase hits into the pair batch the narrow phase
/// consumes, each group sorted ascending by triangle index so the reduction's
/// strict-deeper rule breaks ties toward the smallest index. Returns the pairs
/// and the per-capsule group lengths.
fn build_pairs(candidates: &[Vec<u32>]) -> (Vec<CapsuleTrianglePair>, Vec<usize>) {
    let mut pairs: Vec<CapsuleTrianglePair> = Vec::new();
    let mut group_len: Vec<usize> = Vec::with_capacity(candidates.len());
    for (capsule_index, hits) in candidates.iter().enumerate() {
        let mut tris: Vec<u32> = hits.clone();
        tris.sort_unstable();
        group_len.push(tris.len());
        let capsule_u32 = u32::try_from(capsule_index).unwrap_or(u32::MAX);
        for tri in tris {
            pairs.push(CapsuleTrianglePair::new(capsule_u32, tri));
        }
    }
    (pairs, group_len)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bvh::cpu_build_lbvh;
    use crate::collider::Trimesh;
    use glam::Vec3;

    /// A flat two-triangle quad in the z = 0 plane spanning [0,2] x [0,2].
    fn quad() -> Trimesh {
        Trimesh::new(
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(2.0, 2.0, 0.0),
                Vec3::new(0.0, 2.0, 0.0),
            ],
            vec![[0, 1, 2], [0, 2, 3]],
        )
    }

    #[test]
    fn capsule_above_face_contacts() {
        let mesh = quad();
        let lbvh = cpu_build_lbvh(&mesh.triangle_aabbs());
        // Axis-horizontal capsule hovering 0.3 above triangle 0's interior,
        // radius 0.5: nearest point on the face is directly below, depth 0.2.
        let capsules = vec![Capsule::new(
            Vec3::new(1.2, 0.5, 0.3),
            Vec3::new(1.6, 0.5, 0.3),
            0.5,
        )];
        let out = cpu_capsule_trimesh_collide(&mesh, &lbvh, &capsules, 16).unwrap();
        let c = out[0].expect("capsule above the face must contact");
        assert_eq!(c.a, 0);
        assert!((c.normal - Vec3::Z).length() < 1.0e-6);
        assert!((c.depth - 0.2).abs() < 1.0e-6);
    }

    #[test]
    fn capsule_far_away_reports_none() {
        let mesh = quad();
        let lbvh = cpu_build_lbvh(&mesh.triangle_aabbs());
        let capsules = vec![Capsule::new(
            Vec3::new(1.0, 1.0, 5.0),
            Vec3::new(1.5, 1.0, 5.0),
            0.5,
        )];
        let out = cpu_capsule_trimesh_collide(&mesh, &lbvh, &capsules, 16).unwrap();
        assert!(out[0].is_none());
    }

    #[test]
    fn capsule_on_shared_edge_picks_smallest_triangle_index() {
        let mesh = quad();
        let lbvh = cpu_build_lbvh(&mesh.triangle_aabbs());
        // Capsule hovering over the shared diagonal (0,0,0)->(2,2,0): both
        // triangles are penetrated equally, so the tie must award triangle 0.
        let capsules = vec![Capsule::new(
            Vec3::new(0.8, 0.8, 0.3),
            Vec3::new(1.2, 1.2, 0.3),
            0.5,
        )];
        let out = cpu_capsule_trimesh_collide(&mesh, &lbvh, &capsules, 16).unwrap();
        let c = out[0].expect("capsule on the shared edge must contact");
        assert_eq!(c.b, 0, "tie must resolve to the smallest triangle index");
    }

    #[test]
    fn batched_capsules_stay_in_order() {
        let mesh = quad();
        let lbvh = cpu_build_lbvh(&mesh.triangle_aabbs());
        let capsules = vec![
            Capsule::new(Vec3::new(1.2, 0.5, 0.3), Vec3::new(1.6, 0.5, 0.3), 0.5), // hits
            Capsule::new(Vec3::new(1.0, 1.0, 5.0), Vec3::new(1.5, 1.0, 5.0), 0.5), // misses
            Capsule::new(Vec3::new(0.4, 1.5, -0.3), Vec3::new(0.8, 1.5, -0.3), 0.5), // below
        ];
        let out = cpu_capsule_trimesh_collide(&mesh, &lbvh, &capsules, 16).unwrap();
        assert_eq!(out.len(), 3);
        assert!(out[0].is_some());
        assert!(out[1].is_none());
        assert!(out[2].is_some());
    }

    #[test]
    fn capacity_overflow_is_reported() {
        let mesh = quad();
        let lbvh = cpu_build_lbvh(&mesh.triangle_aabbs());
        let capsules = vec![Capsule::new(
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.1),
            5.0,
        )];
        let err = cpu_capsule_trimesh_collide(&mesh, &lbvh, &capsules, 1).unwrap_err();
        match err {
            OverlapQueryError::CapacityExceeded { query, .. } => assert_eq!(query, 0),
        }
    }
}
