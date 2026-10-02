//! Sphere-versus-trimesh collision: the aggregation layer that turns a mesh,
//! its `LBVH`, and a batch of spheres into one contact manifold per sphere.
//!
//! This is the CPU golden for sphere-against-triangle-mesh collision and the
//! bit-for-bit reference the device twin matches. It stitches three already
//! verified pieces into a single collider:
//!
//! 1. **Broad phase** — each sphere's padded bounding box
//!    (`[centre - r, centre + r]`) queries the mesh `LBVH`
//!    ([`cpu_bvh_aabb_overlap`](crate::cpu_bvh_aabb_overlap)), surfacing the
//!    triangles whose boxes overlap the sphere.
//! 2. **Narrow phase** — every surfaced `(sphere, triangle)` pair runs the
//!    shared [`cpu_sphere_triangle_narrowphase`](crate::cpu_sphere_triangle_narrowphase)
//!    test, the single source of truth for sphere-triangle manifolds.
//! 3. **Reduction** — the per-sphere contacts collapse to the single deepest
//!    one, the manifold a sphere collider against static geometry reports.
//!
//! # Determinism and the tie-break
//!
//! A sphere resting in a crease touches several triangles at the same
//! penetration depth. To make the reduction order-independent (and so the
//! device twin can match it lane for lane), candidate triangles for each sphere
//! are tested in **ascending triangle index** and a contact replaces the
//! current best only when it is **strictly deeper**. Ascending order plus a
//! strict comparison means ties resolve to the **smallest triangle index**,
//! deterministically, regardless of the hierarchy-traversal order the broad
//! phase happens to emit.
//!
//! # Output contract
//!
//! The returned vector has one entry per sphere, in sphere order. A sphere that
//! overlaps no triangle, or overlaps only triangles it does not actually
//! penetrate, reports [`None`]. A sphere in contact reports a [`Contact`] whose
//! `a` is the sphere index, `b` the winning triangle index, and whose normal
//! points from the triangle toward the sphere (the push-out direction), exactly
//! as the underlying narrow phase defines it.
//!
//! Provenance: broad-phase-then-narrow-phase mesh collision with a deepest-point
//! reduction is textbook. The reused pieces cite their own sources. No Unreal
//! Engine source or derived code.

use crate::broadphase::Particle;
use crate::bvh::{cpu_bvh_aabb_overlap, Aabb, Lbvh, OverlapQueryError};
use crate::narrowphase::{cpu_sphere_triangle_narrowphase, Contact, SphereTrianglePair, Triangle};

/// Collides a batch of spheres against a static triangle mesh, returning the
/// single deepest contact per sphere.
///
/// `lbvh` must be the hierarchy built from `mesh.triangle_aabbs()` (see
/// [`Trimesh::triangle_aabbs`](super::Trimesh::triangle_aabbs)); the overlap
/// query returns original triangle indices, so the two must describe the same
/// mesh. `capacity_per_sphere` bounds the broad-phase candidate count per
/// sphere and mirrors the device output buffer's fixed per-query region: a
/// sphere overlapping more triangle boxes than that reports overflow rather
/// than silently dropping candidates.
///
/// # Errors
///
/// Returns [`OverlapQueryError::CapacityExceeded`] (naming the offending
/// sphere, since there is one query per sphere) when a sphere's bounding box
/// overlaps more triangle boxes than `capacity_per_sphere` allows.
pub fn cpu_sphere_trimesh_collide(
    mesh: &super::Trimesh,
    lbvh: &Lbvh,
    spheres: &[Particle],
    capacity_per_sphere: u32,
) -> Result<Vec<Option<Contact>>, OverlapQueryError> {
    // Materialise every triangle once so pair indices address a stable slice;
    // the LBVH's overlap query reports these same original triangle indices.
    let triangles: Vec<Triangle> = (0..mesh.triangle_count())
        .map(|i| mesh.triangle(i))
        .collect();

    // One padded bounding box per sphere; the broad phase runs them as a batch.
    let queries: Vec<Aabb> = spheres
        .iter()
        .map(|s| {
            let r = glam::Vec3::splat(s.radius);
            Aabb::new(s.position - r, s.position + r)
        })
        .collect();

    let candidates = cpu_bvh_aabb_overlap(lbvh, &queries, capacity_per_sphere)?;

    // Build the flat pair list grouped by sphere, each group sorted ascending by
    // triangle index so the reduction's strict-deeper rule breaks ties toward
    // the smallest index. `group_len[s]` records how many pairs sphere `s`
    // contributed, so the reduction can walk the batch results per sphere.
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

    let contacts = cpu_sphere_triangle_narrowphase(spheres, &triangles, &pairs);

    // Reduce each sphere's group to its deepest contact. Pairs within a group
    // are in ascending triangle order, so `depth > best_depth` keeps the
    // smallest triangle index on ties.
    let mut out: Vec<Option<Contact>> = Vec::with_capacity(spheres.len());
    let mut cursor = 0usize;
    for len in group_len {
        let mut best: Option<Contact> = None;
        for contact in &contacts[cursor..cursor + len] {
            if let Some(c) = contact {
                match best {
                    Some(b) if c.depth <= b.depth => {}
                    _ => best = Some(*c),
                }
            }
        }
        out.push(best);
        cursor += len;
    }

    Ok(out)
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
    fn sphere_above_face_contacts() {
        let mesh = quad();
        let lbvh = cpu_build_lbvh(&mesh.triangle_aabbs());
        // Sphere centred over triangle 0's interior, 0.4 above the face, r = 0.5.
        let spheres = vec![Particle::new(Vec3::new(1.5, 0.5, 0.4), 0.5)];
        let out = cpu_sphere_trimesh_collide(&mesh, &lbvh, &spheres, 16).unwrap();
        let c = out[0].expect("sphere must contact the face");
        assert_eq!(c.a, 0);
        assert!((c.normal - Vec3::Z).length() < 1.0e-6);
        assert!((c.depth - 0.1).abs() < 1.0e-6);
    }

    #[test]
    fn sphere_far_away_reports_none() {
        let mesh = quad();
        let lbvh = cpu_build_lbvh(&mesh.triangle_aabbs());
        let spheres = vec![Particle::new(Vec3::new(1.0, 1.0, 5.0), 0.5)];
        let out = cpu_sphere_trimesh_collide(&mesh, &lbvh, &spheres, 16).unwrap();
        assert!(out[0].is_none());
    }

    #[test]
    fn sphere_on_shared_edge_picks_smallest_triangle_index() {
        let mesh = quad();
        let lbvh = cpu_build_lbvh(&mesh.triangle_aabbs());
        // The shared diagonal edge runs from (0,0,0) to (2,2,0); a sphere on it
        // (at (1,1, +z)) penetrates both triangles equally, so the tie-break
        // must award the contact to triangle 0.
        let spheres = vec![Particle::new(Vec3::new(1.0, 1.0, 0.3), 0.5)];
        let out = cpu_sphere_trimesh_collide(&mesh, &lbvh, &spheres, 16).unwrap();
        let c = out[0].expect("sphere on the shared edge must contact");
        assert_eq!(c.b, 0, "tie must resolve to the smallest triangle index");
        assert!((c.depth - 0.2).abs() < 1.0e-6);
    }

    #[test]
    fn deepest_contact_wins_across_candidates() {
        // Two triangles at different heights; the sphere penetrates the higher
        // one more deeply, so that contact must win regardless of index order.
        let mesh = Trimesh::new(
            vec![
                // Triangle 0 sits at z = 0.
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                // Triangle 1 sits at z = 0.3 (closer to the sphere above).
                Vec3::new(0.0, 0.0, 0.3),
                Vec3::new(1.0, 0.0, 0.3),
                Vec3::new(0.0, 1.0, 0.3),
            ],
            vec![[0, 1, 2], [3, 4, 5]],
        );
        let lbvh = cpu_build_lbvh(&mesh.triangle_aabbs());
        // Sphere centre at z = 0.5, r = 0.6: depth vs tri1 = 0.6-0.2 = 0.4,
        // depth vs tri0 = 0.6-0.5 = 0.1. Triangle 1 must win.
        let spheres = vec![Particle::new(Vec3::new(0.25, 0.25, 0.5), 0.6)];
        let out = cpu_sphere_trimesh_collide(&mesh, &lbvh, &spheres, 16).unwrap();
        let c = out[0].expect("sphere must contact the stack");
        assert_eq!(c.b, 1, "deepest penetration must win");
        assert!((c.depth - 0.4).abs() < 1.0e-6);
    }

    #[test]
    fn batched_spheres_stay_in_order() {
        let mesh = quad();
        let lbvh = cpu_build_lbvh(&mesh.triangle_aabbs());
        let spheres = vec![
            Particle::new(Vec3::new(1.5, 0.5, 0.4), 0.5), // contacts
            Particle::new(Vec3::new(1.0, 1.0, 5.0), 0.5), // misses
            Particle::new(Vec3::new(0.5, 1.5, -0.3), 0.5), // contacts from below
        ];
        let out = cpu_sphere_trimesh_collide(&mesh, &lbvh, &spheres, 16).unwrap();
        assert_eq!(out.len(), 3);
        assert!(out[0].is_some());
        assert!(out[1].is_none());
        assert!(out[2].is_some());
    }

    #[test]
    fn capacity_overflow_is_reported() {
        let mesh = quad();
        let lbvh = cpu_build_lbvh(&mesh.triangle_aabbs());
        // A big sphere overlaps both triangle boxes; capacity 1 must overflow.
        let spheres = vec![Particle::new(Vec3::new(1.0, 1.0, 0.0), 5.0)];
        let err = cpu_sphere_trimesh_collide(&mesh, &lbvh, &spheres, 1).unwrap_err();
        match err {
            OverlapQueryError::CapacityExceeded { query, .. } => assert_eq!(query, 0),
        }
    }
}
