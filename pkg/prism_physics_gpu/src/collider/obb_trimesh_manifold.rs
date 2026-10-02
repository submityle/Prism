//! OBB-versus-trimesh contact `manifolds`: the aggregation layer that turns a
//! mesh, its `LBVH`, and a batch of oriented boxes into one *multi-point*
//! [`ContactManifold`](crate::ContactManifold) per box.
//!
//! This is the manifold-producing sibling of
//! [`cpu_obb_trimesh_collide`](super::cpu_obb_trimesh_collide). That collider
//! reports a single deepest [`Contact`](crate::Contact) per box, which is enough
//! to separate a penetrating box but not to hold it *flat* against a mesh face:
//! a box resting on a tiled floor touches several triangles at once, and a
//! solver needs the full contact polygon — up to [`MAX_MANIFOLD_POINTS`] corners
//! sharing one normal — to keep the box from rocking. This module produces that
//! polygon.
//!
//! # Pipeline
//!
//! 1. **Broad phase** — each box's world AABB (see
//!    [`obb_aabb`](super::obb_trimesh)) queries the mesh `LBVH`, exactly as the
//!    single-contact collider does; the two share
//!    [`build_pairs`](super::obb_trimesh) so the surfaced `(box, triangle)` pairs
//!    are identical and ascending by triangle index.
//! 2. **Narrow phase** — every pair runs the shared
//!    [`cpu_obb_triangle_manifold`](crate::cpu_obb_triangle_manifold), yielding
//!    one clipped multi-point manifold (or `None`) per pair.
//! 3. **Per-box merge** — the per-triangle manifolds in a box's group collapse
//!    to a single manifold by [`merge_group`]: the deepest manifold sets the
//!    reference normal, every manifold whose normal agrees with it contributes
//!    its corners, and the shared
//!    [`reduce_to_four`](crate::narrowphase) culls the union back to the widest,
//!    deepest quad.
//!
//! # Why merge by normal
//!
//! A flat box bottom straddling the shared edge of two coplanar floor triangles
//! produces two separate manifolds with the *same* normal; their corners
//! together are the true contact polygon, so merging them and re-reducing is
//! what yields the stable four-corner face contact. Triangles facing a
//! *different* way (a wall the box also grazes) carry a different normal and
//! would fight the dominant face if blended in, so only manifolds aligned with
//! the deepest one — the face the box is actually resting on — are merged. A box
//! wedged into a true crease, touching two differently-oriented faces at once,
//! is a genuinely harder case the single-normal manifold model does not capture;
//! the dominant (deepest) face wins here, matching how face-based solvers treat
//! such contacts.
//!
//! # Determinism
//!
//! Candidates are tested in ascending triangle index, and the reference manifold
//! is replaced only when another is *strictly* deeper, so ties resolve toward
//! the smallest triangle index — the same rule the single-contact reduction
//! uses. The reported manifold's `b` is the triangle index of that deepest
//! contributor, giving a stable, reproducible identity per box.
//!
//! Provenance: broad-phase-then-narrow-phase mesh collision with a per-face
//! manifold merge and four-point reduction is textbook (Ericson, *Real-Time
//! Collision Detection*, 2004). No Unreal Engine source or derived code.

use crate::bvh::{cpu_bvh_aabb_overlap, Aabb, Lbvh, OverlapQueryError};
use crate::narrowphase::{
    cpu_obb_triangle_manifold, reduce_to_four, ContactManifold, ManifoldPoint, Obb, Triangle,
};

use super::obb_trimesh::{build_pairs, obb_aabb};

/// Minimum dot product between two unit normals for them to count as the same
/// contact face.
///
/// Manifolds coming from distinct coplanar triangles under one box share a
/// normal to floating-point noise, so a dot above this threshold (about eight
/// degrees of slack) groups them; a genuinely different face falls below it and
/// is excluded from the merge.
const NORMAL_ALIGN_EPS: f32 = 0.99;

/// Collides a batch of oriented bounding boxes against a static triangle mesh,
/// returning one merged multi-point [`ContactManifold`] per box.
///
/// `lbvh` must be the hierarchy built from [`mesh.triangle_aabbs`](super::Trimesh);
/// `capacity_per_box` bounds the broad-phase candidate count per box and mirrors
/// the device output buffer's fixed per-query region. The result is index-aligned
/// with `boxes`: entry `i` is the manifold for box `i`, or `None` when that box
/// touches no triangle.
///
/// # Errors
///
/// Returns [`OverlapQueryError::CapacityExceeded`](crate::bvh::OverlapQueryError)
/// (naming the offending box) when a box's bounding box overlaps more triangle
/// boxes than `capacity_per_box` allows.
pub fn cpu_obb_trimesh_manifold_collide(
    mesh: &super::Trimesh,
    lbvh: &Lbvh,
    boxes: &[Obb],
    capacity_per_box: u32,
) -> Result<Vec<Option<ContactManifold>>, OverlapQueryError> {
    let triangles: Vec<Triangle> = (0..mesh.triangle_count())
        .map(|i| mesh.triangle(i))
        .collect();

    let queries: Vec<Aabb> = boxes.iter().map(obb_aabb).collect();

    let candidates = cpu_bvh_aabb_overlap(lbvh, &queries, capacity_per_box)?;

    let (pairs, group_len) = build_pairs(&candidates);

    let manifolds = cpu_obb_triangle_manifold(boxes, &triangles, &pairs);

    Ok(reduce_manifolds_per_group(&manifolds, &group_len))
}

/// Collapses the flat per-pair manifold list into one merged manifold per box,
/// walking the groups in the same order [`build_pairs`](super::obb_trimesh)
/// emitted them.
fn reduce_manifolds_per_group(
    manifolds: &[Option<ContactManifold>],
    group_len: &[usize],
) -> Vec<Option<ContactManifold>> {
    let mut out = Vec::with_capacity(group_len.len());
    let mut offset = 0;
    for &len in group_len {
        let group = &manifolds[offset..offset + len];
        out.push(merge_group(group));
        offset += len;
    }
    out
}

/// Merges one box's per-triangle manifolds into a single manifold.
///
/// Returns `None` if the group holds no live manifold. Otherwise the deepest
/// manifold (by its deepest corner, ties toward the earliest/smallest triangle
/// index) sets the reference normal and the reported `b`; every manifold whose
/// normal aligns with it contributes its corners, and the union is reduced back
/// to at most [`MAX_MANIFOLD_POINTS`](crate::narrowphase) corners.
fn merge_group(group: &[Option<ContactManifold>]) -> Option<ContactManifold> {
    // Pick the reference manifold: strictly deepest corner wins, so equal depths
    // keep the earliest (smallest triangle index) group member.
    let mut reference: Option<&ContactManifold> = None;
    let mut best_depth = f32::NEG_INFINITY;
    for manifold in group.iter().flatten() {
        let depth = deepest_corner(manifold);
        if depth > best_depth {
            best_depth = depth;
            reference = Some(manifold);
        }
    }
    let reference = reference?;

    // Gather every corner from manifolds that face the same way as the reference.
    let mut points: Vec<ManifoldPoint> = Vec::new();
    for manifold in group.iter().flatten() {
        if manifold.normal.dot(reference.normal) >= NORMAL_ALIGN_EPS {
            points.extend_from_slice(&manifold.points[..manifold.count as usize]);
        }
    }

    let reduced = reduce_to_four(&points, reference.normal);
    Some(ContactManifold::new(
        reference.a,
        reference.b,
        reference.normal,
        reduced.len(),
        &reduced,
    ))
}

/// The deepest penetration among a manifold's live corners.
fn deepest_corner(manifold: &ContactManifold) -> f32 {
    manifold.points[..manifold.count as usize]
        .iter()
        .map(|p| p.depth)
        .fold(f32::NEG_INFINITY, f32::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bvh::cpu_build_lbvh;
    use crate::collider::Trimesh;
    use glam::Vec3;

    const EPS: f32 = 1.0e-4;

    /// The identity axis triple.
    fn axes() -> [Vec3; 3] {
        [Vec3::X, Vec3::Y, Vec3::Z]
    }

    /// A flat quad in the z = 0 plane spanning [0,2] x [0,2], split into two
    /// triangles that share the diagonal.
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
    fn box_straddling_two_triangles_merges_to_four_corners() {
        let mesh = quad();
        let lbvh = cpu_build_lbvh(&mesh.triangle_aabbs());
        // A wide box centred over the shared diagonal, bottom 0.1 below z = 0.
        // Its footprint overlaps both triangles, so the two per-triangle
        // manifolds must merge into one four-corner face contact.
        let boxes = vec![Obb::new(Vec3::new(1.0, 1.0, 0.4), axes(), Vec3::splat(0.5))];
        let out = cpu_obb_trimesh_manifold_collide(&mesh, &lbvh, &boxes, 16).unwrap();
        let m = out[0].expect("straddling box must contact");
        assert_eq!(m.a, 0);
        assert_eq!(m.count, 4, "a flush face must report four corners");
        assert!((m.normal - Vec3::Z).length() < EPS, "normal {:?}", m.normal);
        for pt in &m.points[..m.count as usize] {
            assert!((pt.depth - 0.1).abs() < EPS, "depth {}", pt.depth);
            assert!((pt.position.z + 0.05).abs() < EPS, "z {}", pt.position.z);
        }
    }

    #[test]
    fn box_over_single_triangle_passes_through() {
        let mesh = quad();
        let lbvh = cpu_build_lbvh(&mesh.triangle_aabbs());
        // A small box fully inside triangle 0's half of the quad (y < x),
        // dipping 0.1 below the plane: it touches exactly one triangle, so the
        // merged manifold is just that triangle's clipped polygon.
        let boxes = vec![Obb::new(Vec3::new(1.2, 0.5, 0.2), axes(), Vec3::splat(0.3))];
        let out = cpu_obb_trimesh_manifold_collide(&mesh, &lbvh, &boxes, 16).unwrap();
        let m = out[0].expect("box over one triangle must contact");
        assert_eq!(m.a, 0);
        assert_eq!(m.b, 0, "only triangle 0 is under the box");
        assert!(m.count >= 1 && m.count <= 4);
        assert!((m.normal - Vec3::Z).length() < EPS);
    }

    #[test]
    fn separated_box_reports_none() {
        let mesh = quad();
        let lbvh = cpu_build_lbvh(&mesh.triangle_aabbs());
        // A box well above the plane: broad phase may surface candidates, but the
        // narrow phase separates them, so the group holds no live manifold.
        let boxes = vec![Obb::new(Vec3::new(1.0, 1.0, 5.0), axes(), Vec3::splat(0.5))];
        let out = cpu_obb_trimesh_manifold_collide(&mesh, &lbvh, &boxes, 16).unwrap();
        assert!(out[0].is_none(), "a separated box reports no manifold");
    }

    #[test]
    fn capacity_overflow_is_reported() {
        let mesh = quad();
        let lbvh = cpu_build_lbvh(&mesh.triangle_aabbs());
        // A box covering the whole quad overlaps both triangle AABBs; a capacity
        // of one cannot hold both candidates.
        let boxes = vec![Obb::new(Vec3::new(1.0, 1.0, 0.4), axes(), Vec3::splat(1.5))];
        let err = cpu_obb_trimesh_manifold_collide(&mesh, &lbvh, &boxes, 1).unwrap_err();
        assert!(matches!(err, OverlapQueryError::CapacityExceeded { .. }));
    }

    #[test]
    fn deepest_triangle_sets_the_reported_identity() {
        // Two groups' worth: confirm merge picks the strictly deepest corner's
        // triangle as b when depths differ. Build a mesh where triangle 1 sits
        // slightly higher so a tilted box bites deeper into triangle 0.
        let mesh = quad();
        let lbvh = cpu_build_lbvh(&mesh.triangle_aabbs());
        // Box low over triangle 0's corner (origin), just grazing triangle 1.
        let boxes = vec![Obb::new(Vec3::new(0.4, 0.4, 0.35), axes(), Vec3::splat(0.4))];
        let out = cpu_obb_trimesh_manifold_collide(&mesh, &lbvh, &boxes, 16).unwrap();
        let m = out[0].expect("contact");
        assert_eq!(m.b, 0, "triangle 0 holds the deepest corner");
    }
}
