//! Vertex-count reduction for cooked convex hulls.
//!
//! A convex collider cooked from a dense render mesh can carry far more
//! vertices than a physics solver wants to touch each step: `GJK`/`EPA` support
//! queries and the GPU host packer both scale with the hull vertex count, and
//! most engines expose a hard cap -- `PhysX` `PxConvexFlag::eCOMPUTE_CONVEX`
//! paired with `PxConvexMeshDesc::vertexLimit`, Jolt
//! `ConvexHullShapeSettings::mMaxConvexRadius`/vertex budget, Chaos
//! `FConvexBuilder` simplification. [`simplify_convex_hull`] closes that gap: it
//! reduces a point cloud's convex hull to at most `max_vertices` vertices.
//!
//! # Contract
//!
//! The result is the convex hull of the lowest-error **subset** of the full
//! hull's vertices: every returned vertex is one of the original hull vertices,
//! so the simplified solid is contained in the full hull (an *inner*
//! approximation). This is the right trade-off for a performance proxy that must
//! never report a contact the true shape would not -- callers that instead need
//! a strictly *enclosing* bound should use the shape's AABB or bounding sphere.
//! [`SimplifiedHull::removed_volume`] reports how much volume the reduction gave
//! up so content tools can gate quality.
//!
//! # Algorithm
//!
//! Greedy least-volume vertex decimation. Starting from the exact hull, repeat:
//! for every current hull vertex, measure the hull volume that remains if that
//! vertex is dropped, and remove the vertex whose removal costs the least volume
//! (deterministic lowest-index tie-break). Re-cook the hull after each removal
//! so the vertex set stays canonical -- removing one vertex can push a neighbour
//! strictly interior, which simply converges faster. The loop ends once the cap
//! is met or no further vertex can be removed while leaving a valid solid.
//!
//! All ordering is derived from deterministic index loops over the hull builder,
//! which is itself bit-for-bit reproducible, so the simplified hull is stable
//! across runs -- a hard requirement for cross-run state hashing.
//!
//! # Provenance
//!
//! Greedy convex decimation by least-volume vertex removal is a textbook
//! mesh-simplification strategy (cf. progressive-hull and quadric-decimation
//! literature). This module contains **no Unreal Engine source or derived
//! code**.

use glam::Vec3;

use super::convex_mesh::ConvexMeshData;

/// A convex hull reduced to a bounded vertex count by [`simplify_convex_hull`].
#[derive(Clone, Debug)]
pub struct SimplifiedHull {
    /// Hull vertices in local space; a subset of the full hull's vertices.
    pub vertices: Vec<Vec3>,
    /// Surface triangles as indices into [`Self::vertices`], wound
    /// counter-clockwise as seen from outside (outward normal).
    pub triangles: Vec<[u32; 3]>,
    /// Volume given up by the reduction: `full_hull_volume - simplified_volume`.
    /// Always `>= 0`; zero when no vertex had to be removed.
    pub removed_volume: f32,
}

impl SimplifiedHull {
    /// Number of vertices in the simplified hull.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Builds a cooked [`ConvexMeshData`] from the simplified surface.
    #[must_use]
    pub fn to_convex_mesh(&self) -> ConvexMeshData {
        ConvexMeshData::from_surface(self.vertices.clone(), self.triangles.clone())
    }
}

/// The smallest vertex budget that can still bound a convex solid: a
/// tetrahedron.
pub const MIN_HULL_VERTICES: usize = 4;

/// Reduces the convex hull of `points` to at most `max_vertices` vertices.
///
/// Returns the exact hull unchanged (with [`SimplifiedHull::removed_volume`]
/// `== 0.0`) when it already fits the budget. Returns [`None`] when
/// `max_vertices < MIN_HULL_VERTICES` or when `points` is degenerate (fewer than
/// four points, or all collinear/coplanar) and therefore has no convex solid; see
/// [`convex_hull`](super::convex_hull).
///
/// The reduction is an inner approximation: see the [module docs](self) for the
/// full contract.
#[must_use]
pub fn simplify_convex_hull(points: &[Vec3], max_vertices: usize) -> Option<SimplifiedHull> {
    if max_vertices < MIN_HULL_VERTICES {
        return None;
    }

    let full = ConvexMeshData::from_points(points)?;
    let full_volume = full.volume();
    let mut working: Vec<Vec3> = full.vertices().to_vec();

    if working.len() <= max_vertices {
        return Some(SimplifiedHull {
            vertices: full.vertices().to_vec(),
            triangles: full.triangles().to_vec(),
            removed_volume: 0.0,
        });
    }

    // Greedily drop the least-costly vertex until the budget is met or no
    // removal leaves a valid solid.
    while working.len() > max_vertices {
        let Some(best) = best_removal(&working) else {
            break;
        };
        working = best.vertices().to_vec();
    }

    let final_hull = ConvexMeshData::from_points(&working)?;
    let removed = (full_volume - final_hull.volume()).max(0.0);

    Some(SimplifiedHull {
        vertices: final_hull.vertices().to_vec(),
        triangles: final_hull.triangles().to_vec(),
        removed_volume: removed,
    })
}

/// Among all single-vertex removals from `working`, returns the re-cooked hull
/// that preserves the most volume (loses the least), with a deterministic
/// lowest-index tie-break. Returns [`None`] when no removal leaves a valid
/// convex solid (e.g. every candidate collapses to a degenerate cloud).
fn best_removal(working: &[Vec3]) -> Option<ConvexMeshData> {
    let mut best: Option<ConvexMeshData> = None;
    let mut best_volume = f32::NEG_INFINITY;

    for drop_idx in 0..working.len() {
        let candidate: Vec<Vec3> = working
            .iter()
            .enumerate()
            .filter_map(|(i, &p)| (i != drop_idx).then_some(p))
            .collect();
        if candidate.len() < MIN_HULL_VERTICES {
            continue;
        }
        let Some(mesh) = ConvexMeshData::from_points(&candidate) else {
            continue;
        };
        let volume = mesh.volume();
        // Strictly greater keeps the first (lowest-index) winner on ties, so the
        // choice is deterministic.
        if volume > best_volume {
            best_volume = volume;
            best = Some(mesh);
        }
    }

    best
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The eight corners of an axis-aligned box of the given half-extents.
    fn box_corners(h: Vec3) -> Vec<Vec3> {
        let mut pts = Vec::with_capacity(8);
        for sx in [-1.0_f32, 1.0] {
            for sy in [-1.0_f32, 1.0] {
                for sz in [-1.0_f32, 1.0] {
                    pts.push(Vec3::new(sx * h.x, sy * h.y, sz * h.z));
                }
            }
        }
        pts
    }

    /// True when `p` lies inside or on every face half-space of the hull.
    fn inside_hull(mesh: &ConvexMeshData, p: Vec3, eps: f32) -> bool {
        mesh.face_half_spaces()
            .iter()
            .all(|&(normal, offset)| normal.dot(p) - offset <= eps)
    }

    #[test]
    fn rejects_sub_tetrahedron_budget() {
        let pts = box_corners(Vec3::ONE);
        assert!(simplify_convex_hull(&pts, 3).is_none());
        assert!(simplify_convex_hull(&pts, 0).is_none());
    }

    #[test]
    fn rejects_degenerate_cloud() {
        let coplanar = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        assert!(simplify_convex_hull(&coplanar, 4).is_none());
        assert!(simplify_convex_hull(&coplanar[..3], 4).is_none());
    }

    #[test]
    fn under_budget_is_unchanged() {
        let pts = box_corners(Vec3::new(1.5, 0.75, 2.0));
        let simplified = simplify_convex_hull(&pts, 8).expect("box fits budget");
        assert_eq!(simplified.vertex_count(), 8);
        assert_eq!(simplified.triangles.len(), 12);
        assert_eq!(simplified.removed_volume, 0.0);

        // A generous budget leaves the hull untouched as well.
        let roomy = simplify_convex_hull(&pts, 64).expect("box fits budget");
        assert_eq!(roomy.vertex_count(), 8);
        assert_eq!(roomy.removed_volume, 0.0);
    }

    #[test]
    fn drops_near_coplanar_bumps_back_to_box() {
        // A unit box plus small bumps that barely bulge four faces outward. The
        // bumps are the lowest-volume vertices, so a budget of 8 should recover a
        // near-cube with only a sliver of volume removed.
        let mut pts = box_corners(Vec3::ONE);
        pts.push(Vec3::new(0.0, 0.0, 1.02));
        pts.push(Vec3::new(0.0, 0.0, -1.02));
        pts.push(Vec3::new(1.02, 0.0, 0.0));
        pts.push(Vec3::new(-1.02, 0.0, 0.0));

        let simplified = simplify_convex_hull(&pts, 8).expect("reduce to box budget");
        assert!(simplified.vertex_count() <= 8);
        assert!(
            simplified.removed_volume > 0.0,
            "removing the bumps must give up volume"
        );
        assert!(
            simplified.removed_volume < 0.2,
            "near-coplanar bumps carry little volume, got {}",
            simplified.removed_volume
        );
    }

    #[test]
    fn result_is_subset_and_contained() {
        let pts = sample_cloud();
        let full = ConvexMeshData::from_points(&pts).expect("full hull");
        let simplified = simplify_convex_hull(&pts, 6).expect("reduce");

        assert!(simplified.vertex_count() <= 6);
        assert!(simplified.vertex_count() >= MIN_HULL_VERTICES);

        // Every simplified vertex is one of the full hull's vertices (subset).
        for &v in &simplified.vertices {
            assert!(
                full.vertices().iter().any(|&fv| (fv - v).length() < 1.0e-5),
                "vertex {v:?} is not an original hull vertex"
            );
        }

        // Inner approximation: the simplified solid is contained in the full hull.
        let eps = 1.0e-4 * full.bounding_radius().max(1.0);
        for &v in &simplified.vertices {
            assert!(inside_hull(&full, v, eps), "vertex {v:?} escaped full hull");
        }

        assert!(simplified.removed_volume >= 0.0);
        assert!(simplified.removed_volume < full.volume());
    }

    #[test]
    fn tighter_budget_removes_at_least_as_much() {
        let pts = sample_cloud();
        let loose = simplify_convex_hull(&pts, 10).expect("loose");
        let tight = simplify_convex_hull(&pts, 5).expect("tight");
        assert!(tight.vertex_count() <= loose.vertex_count());
        assert!(
            tight.removed_volume >= loose.removed_volume - 1.0e-6,
            "tighter budget {} should not keep more volume than looser {}",
            tight.removed_volume,
            loose.removed_volume
        );
    }

    #[test]
    fn deterministic_across_runs() {
        let pts = sample_cloud();
        let first = simplify_convex_hull(&pts, 7).expect("hull");
        for _ in 0..8 {
            let again = simplify_convex_hull(&pts, 7).expect("hull");
            assert_eq!(first.vertices, again.vertices, "vertices reproducible");
            assert_eq!(first.triangles, again.triangles, "triangles reproducible");
            assert_eq!(
                first.removed_volume, again.removed_volume,
                "removed volume reproducible"
            );
        }
    }

    #[test]
    fn simplified_surface_cooks_cleanly() {
        let pts = sample_cloud();
        let simplified = simplify_convex_hull(&pts, 6).expect("reduce");
        let mesh = simplified.to_convex_mesh();
        assert_eq!(mesh.vertices().len(), simplified.vertex_count());
        assert!(mesh.volume() > 0.0);
        // Euler relation for a simplicial polytope.
        assert_eq!(mesh.triangles().len(), 2 * mesh.vertices().len() - 4);
    }

    /// A fixed, well-separated point cloud (deterministic, no RNG) whose hull has
    /// many vertices so the decimation loop runs several removals.
    fn sample_cloud() -> Vec<Vec3> {
        let mut pts = Vec::new();
        let coords = [-3.0_f32, -1.0, 1.0, 3.0];
        for (i, &x) in coords.iter().enumerate() {
            for (j, &y) in coords.iter().enumerate() {
                let z = 2.0 - 0.2 * (x * x + y * y) + 0.11 * (i as f32) - 0.07 * (j as f32);
                pts.push(Vec3::new(x, y, z));
            }
        }
        pts.push(Vec3::new(0.0, 0.0, -4.0));
        pts.push(Vec3::new(0.0, 0.0, 5.0));
        pts.push(Vec3::new(4.5, 0.0, 0.0));
        pts.push(Vec3::new(-4.5, 0.0, 0.0));
        pts
    }
}
