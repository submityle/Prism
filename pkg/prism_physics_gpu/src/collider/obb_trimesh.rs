//! OBB-versus-trimesh collision: the aggregation layer that turns a mesh,
//! its `LBVH`, and a batch of oriented bounding boxes into one contact manifold
//! per box.
//!
//! This is the CPU golden for OBB-against-triangle-mesh collision and the
//! bit-for-bit reference the device twin matches. It mirrors
//! [`cpu_sphere_trimesh_collide`](super::cpu_sphere_trimesh_collide) and
//! [`cpu_capsule_trimesh_collide`](super::cpu_capsule_trimesh_collide) exactly,
//! swapping the proxy for an [`Obb`]:
//!
//! 1. **Broad phase** — each box's world-space axis-aligned bound queries the
//!    mesh `LBVH`. An oriented box's AABB is centred at the box centre with a
//!    half-extent that is the absolute projection of the three scaled local
//!    axes onto each world axis (see [`obb_aabb`]).
//! 2. **Narrow phase** — every surfaced `(box, triangle)` pair runs the shared
//!    [`cpu_obb_triangle_narrowphase`](crate::cpu_obb_triangle_narrowphase) test.
//! 3. **Reduction** — the per-box contacts collapse to the single deepest one
//!    via the shared [`deepest_per_group`](super::reduce::deepest_per_group),
//!    with ties resolved to the smallest triangle index.
//!
//! The determinism argument, tie-break rule, and output contract are identical
//! to the sphere and capsule colliders: candidates are tested in ascending
//! triangle index and a contact replaces the current best only when it is
//! strictly deeper, so the device twin can match the reduction lane for lane. A
//! box in contact reports a [`Contact`] whose `a` is the box index, `b` the
//! winning triangle index, and whose normal points from the triangle toward the
//! box.
//!
//! Provenance: broad-phase-then-narrow-phase mesh collision with a deepest-point
//! reduction is textbook; the reused pieces cite their own sources. No Unreal
//! Engine source or derived code.

use crate::bvh::{cpu_bvh_aabb_overlap, Aabb, Lbvh, OverlapQueryError};
use crate::narrowphase::{cpu_obb_triangle_narrowphase, Contact, Obb, ObbTrianglePair, Triangle};

use super::reduce::deepest_per_group;

/// Collides a batch of oriented bounding boxes against a static triangle mesh,
/// returning the single deepest contact per box.
///
/// `lbvh` must be the hierarchy built from `mesh.triangle_aabbs()`;
/// `capacity_per_box` bounds the broad-phase candidate count per box and mirrors
/// the device output buffer's fixed per-query region.
///
/// # Errors
///
/// Returns [`OverlapQueryError::CapacityExceeded`] (naming the offending box,
/// since there is one query per box) when a box's bounding box overlaps more
/// triangle boxes than `capacity_per_box` allows.
pub fn cpu_obb_trimesh_collide(
    mesh: &super::Trimesh,
    lbvh: &Lbvh,
    boxes: &[Obb],
    capacity_per_box: u32,
) -> Result<Vec<Option<Contact>>, OverlapQueryError> {
    let triangles: Vec<Triangle> = (0..mesh.triangle_count())
        .map(|i| mesh.triangle(i))
        .collect();

    let queries: Vec<Aabb> = boxes.iter().map(obb_aabb).collect();

    let candidates = cpu_bvh_aabb_overlap(lbvh, &queries, capacity_per_box)?;

    let (pairs, group_len) = build_pairs(&candidates);

    let contacts = cpu_obb_triangle_narrowphase(boxes, &triangles, &pairs);

    Ok(deepest_per_group(&contacts, &group_len))
}

/// The world-space axis-aligned bounding box of an oriented box.
///
/// Each world-axis half-extent is the sum of the absolute contributions of the
/// three scaled local axes: `|axis_i| . half_extent_i` accumulated per
/// component. This is the tightest AABB enclosing the rotated box.
pub(super) fn obb_aabb(obb: &Obb) -> Aabb {
    let extent = obb.axes[0].abs() * obb.half_extents.x
        + obb.axes[1].abs() * obb.half_extents.y
        + obb.axes[2].abs() * obb.half_extents.z;
    Aabb::new(obb.center - extent, obb.center + extent)
}

/// Flattens the per-query broad-phase hits into the pair batch the narrow phase
/// consumes, each group sorted ascending by triangle index so the reduction's
/// strict-deeper rule breaks ties toward the smallest index. Returns the pairs
/// and the per-box group lengths.
pub(super) fn build_pairs(candidates: &[Vec<u32>]) -> (Vec<ObbTrianglePair>, Vec<usize>) {
    let mut pairs: Vec<ObbTrianglePair> = Vec::new();
    let mut group_len: Vec<usize> = Vec::with_capacity(candidates.len());
    for (box_index, hits) in candidates.iter().enumerate() {
        let mut tris: Vec<u32> = hits.clone();
        tris.sort_unstable();
        group_len.push(tris.len());
        let box_u32 = u32::try_from(box_index).unwrap_or(u32::MAX);
        for tri in tris {
            pairs.push(ObbTrianglePair::new(box_u32, tri));
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

    /// The identity axis triple.
    fn axes() -> [Vec3; 3] {
        [Vec3::X, Vec3::Y, Vec3::Z]
    }

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
    fn box_above_face_contacts() {
        let mesh = quad();
        let lbvh = cpu_build_lbvh(&mesh.triangle_aabbs());
        // Axis-aligned half-unit box centred 0.4 above triangle 0's interior:
        // the box bottom at z = 0.4 - 0.5 = -0.1 dips 0.1 below the face.
        let boxes = vec![Obb::new(
            Vec3::new(1.5, 0.5, 0.4),
            axes(),
            Vec3::splat(0.5),
        )];
        let out = cpu_obb_trimesh_collide(&mesh, &lbvh, &boxes, 16).unwrap();
        let c = out[0].expect("box above the face must contact");
        assert_eq!(c.a, 0);
        assert!((c.normal - Vec3::Z).length() < 1.0e-6);
        assert!((c.depth - 0.1).abs() < 1.0e-6);
    }

    #[test]
    fn box_far_away_reports_none() {
        let mesh = quad();
        let lbvh = cpu_build_lbvh(&mesh.triangle_aabbs());
        let boxes = vec![Obb::new(
            Vec3::new(1.0, 1.0, 5.0),
            axes(),
            Vec3::splat(0.5),
        )];
        let out = cpu_obb_trimesh_collide(&mesh, &lbvh, &boxes, 16).unwrap();
        assert!(out[0].is_none());
    }

    #[test]
    fn box_on_shared_edge_picks_smallest_triangle_index() {
        let mesh = quad();
        let lbvh = cpu_build_lbvh(&mesh.triangle_aabbs());
        // Box straddling the shared diagonal (0,0,0)->(2,2,0): both triangles
        // are penetrated equally, so the tie must award triangle 0.
        let boxes = vec![Obb::new(
            Vec3::new(1.0, 1.0, 0.4),
            axes(),
            Vec3::splat(0.5),
        )];
        let out = cpu_obb_trimesh_collide(&mesh, &lbvh, &boxes, 16).unwrap();
        let c = out[0].expect("box on the shared edge must contact");
        assert_eq!(c.b, 0, "tie must resolve to the smallest triangle index");
    }

    #[test]
    fn rotated_box_bound_still_overlaps() {
        let mesh = quad();
        let lbvh = cpu_build_lbvh(&mesh.triangle_aabbs());
        // A box rotated 45 degrees about z: its world AABB grows by sqrt(2),
        // but the centre still sits above the face, so it contacts.
        let s = std::f32::consts::FRAC_1_SQRT_2;
        let rotated = [
            Vec3::new(s, s, 0.0),
            Vec3::new(-s, s, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let boxes = vec![Obb::new(Vec3::new(1.0, 1.0, 0.4), rotated, Vec3::splat(0.5))];
        let out = cpu_obb_trimesh_collide(&mesh, &lbvh, &boxes, 16).unwrap();
        assert!(out[0].is_some(), "rotated box over the face must contact");
    }

    #[test]
    fn batched_boxes_stay_in_order() {
        let mesh = quad();
        let lbvh = cpu_build_lbvh(&mesh.triangle_aabbs());
        let boxes = vec![
            Obb::new(Vec3::new(1.5, 0.5, 0.4), axes(), Vec3::splat(0.5)), // hits
            Obb::new(Vec3::new(1.0, 1.0, 5.0), axes(), Vec3::splat(0.5)), // misses
            Obb::new(Vec3::new(0.5, 1.5, -0.4), axes(), Vec3::splat(0.5)), // below
        ];
        let out = cpu_obb_trimesh_collide(&mesh, &lbvh, &boxes, 16).unwrap();
        assert_eq!(out.len(), 3);
        assert!(out[0].is_some());
        assert!(out[1].is_none());
        assert!(out[2].is_some());
    }

    #[test]
    fn capacity_overflow_is_reported() {
        let mesh = quad();
        let lbvh = cpu_build_lbvh(&mesh.triangle_aabbs());
        let boxes = vec![Obb::new(Vec3::new(1.0, 1.0, 0.0), axes(), Vec3::splat(5.0))];
        let err = cpu_obb_trimesh_collide(&mesh, &lbvh, &boxes, 1).unwrap_err();
        match err {
            OverlapQueryError::CapacityExceeded { query, .. } => assert_eq!(query, 0),
        }
    }
}
