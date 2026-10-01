//! Approximate minimal bounding sphere of a triangle mesh for the `CPU` golden
//! path.
//!
//! Culling, LOD selection, and broad-phase queries all lean on a cheap sphere
//! that encloses an object's geometry. Computing the *exact* minimum enclosing
//! sphere (Welzl) needs randomized recursion; `AAA` engines instead bake the
//! classic two-pass heuristic of Jack Ritter, which is linear, deterministic,
//! and in practice within a few percent of optimal. This module implements it
//! over the mesh's *referenced* vertices (those actually touched by an index
//! triple), so stray unused positions never inflate the bound.
//!
//! The algorithm seeds a diameter from a far-apart vertex pair, then sweeps the
//! vertices once more and grows the sphere just enough to swallow any outlier.
//! All arithmetic accumulates in `f64`; the only non-polynomial operation is a
//! square root (permitted by the `CPU` golden-path numeric policy), so the
//! result is reproducible across platforms.
//!
//! [`bounding_sphere`] returns [`BoundingSphere`] (centre plus radius), or
//! [`None`] when the mesh references no vertices.

use super::triangle_mesh::TriangleMesh;

/// An axis-free bounding sphere: a centre point and an enclosing radius.
///
/// The sphere is guaranteed to contain every referenced vertex of the mesh it
/// was built from (up to floating-point rounding). It is produced by
/// [`bounding_sphere`] and is never empty — a one-vertex mesh yields a
/// zero-radius sphere at that vertex.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundingSphere {
    /// Centre of the sphere in the mesh's local space.
    center: [f32; 3],
    /// Non-negative enclosing radius.
    radius: f32,
}

impl BoundingSphere {
    /// Centre of the sphere in the mesh's local space.
    pub fn center(&self) -> [f32; 3] {
        self.center
    }

    /// Non-negative enclosing radius of the sphere.
    pub fn radius(&self) -> f32 {
        self.radius
    }

    /// Returns `true` when `point` lies inside or on the sphere, using a small
    /// relative/absolute tolerance so boundary vertices from the build are not
    /// rejected by rounding.
    pub fn contains(&self, point: [f32; 3]) -> bool {
        let dx = f64::from(point[0]) - f64::from(self.center[0]);
        let dy = f64::from(point[1]) - f64::from(self.center[1]);
        let dz = f64::from(point[2]) - f64::from(self.center[2]);
        let dist_sq = dx * dx + dy * dy + dz * dz;
        let r = f64::from(self.radius);
        // Pad the squared radius by a tolerance proportional to the radius so
        // that vertices sitting exactly on the computed shell test as inside.
        let tolerance = 1e-5 * (1.0 + r);
        let padded = r + tolerance;
        dist_sq <= padded * padded
    }
}

/// Squared Euclidean distance between two points, accumulated in `f64`.
fn distance_squared(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

/// Collects the distinct vertices referenced by at least one triangle index,
/// promoted to `f64`, preserving first-seen order for a deterministic sweep.
fn referenced_vertices(mesh: &TriangleMesh) -> Vec<[f64; 3]> {
    let positions = mesh.positions();
    let mut seen = vec![false; positions.len()];
    let mut points = Vec::new();
    for tri in mesh.indices() {
        for &index in tri {
            let i = index as usize;
            if i < positions.len() && !seen[i] {
                seen[i] = true;
                let p = positions[i];
                points.push([f64::from(p[0]), f64::from(p[1]), f64::from(p[2])]);
            }
        }
    }
    points
}

/// Finds the index of the vertex farthest (by squared distance) from `origin`.
///
/// `points` must be non-empty; the first element is the fallback when every
/// vertex is coincident.
fn farthest_from(points: &[[f64; 3]], origin: [f64; 3]) -> usize {
    let mut best = 0usize;
    let mut best_sq = -1.0f64;
    for (i, &p) in points.iter().enumerate() {
        let d = distance_squared(p, origin);
        if d > best_sq {
            best_sq = d;
            best = i;
        }
    }
    best
}

/// Computes an approximate minimal bounding sphere over the mesh's referenced
/// vertices using Ritter's two-pass heuristic.
///
/// Returns [`None`] when no triangle references a valid vertex (an empty or
/// index-free mesh). Otherwise the returned [`BoundingSphere`] encloses every
/// referenced vertex.
pub fn bounding_sphere(mesh: &TriangleMesh) -> Option<BoundingSphere> {
    let points = referenced_vertices(mesh);
    let (first, _) = points.split_first()?;

    // Seed the diameter: the vertex farthest from an arbitrary start, then the
    // vertex farthest from that one. These two define the initial sphere.
    let a = farthest_from(&points, *first);
    let b = farthest_from(&points, points[a]);
    let pa = points[a];
    let pb = points[b];

    let mut center = [
        0.5 * (pa[0] + pb[0]),
        0.5 * (pa[1] + pb[1]),
        0.5 * (pa[2] + pb[2]),
    ];
    let mut radius = 0.5 * distance_squared(pa, pb).sqrt();

    // Growth pass: expand to cover any vertex left outside the seed sphere.
    for &p in &points {
        let dist_sq = distance_squared(p, center);
        if dist_sq <= radius * radius {
            continue;
        }
        let dist = dist_sq.sqrt();
        // New radius is the mean of the old radius and the outlier distance;
        // slide the centre toward the outlier by exactly the half-gap so both
        // the previous shell and the new point stay enclosed.
        let new_radius = 0.5 * (radius + dist);
        let shift = if dist > 0.0 {
            (new_radius - radius) / dist
        } else {
            0.0
        };
        center[0] += (p[0] - center[0]) * shift;
        center[1] += (p[1] - center[1]) * shift;
        center[2] += (p[2] - center[2]) * shift;
        radius = new_radius;
    }

    Some(BoundingSphere {
        center: [center[0] as f32, center[1] as f32, center[2] as f32],
        radius: radius as f32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ray_scene::triangle_mesh::TriangleMesh;

    /// Builds an index-only triangle mesh from positions (no normals/uvs).
    fn mesh(positions: Vec<[f32; 3]>, indices: Vec<[u32; 3]>) -> TriangleMesh {
        TriangleMesh::new(positions, Vec::new(), Vec::new(), indices).unwrap()
    }

    #[test]
    fn empty_mesh_has_no_sphere() {
        let m = mesh(Vec::new(), Vec::new());
        assert!(bounding_sphere(&m).is_none());
    }

    #[test]
    fn mesh_without_indices_has_no_sphere() {
        // Positions present but no triangle references them.
        let m = mesh(vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], Vec::new());
        assert!(bounding_sphere(&m).is_none());
    }

    #[test]
    fn degenerate_single_point_triangle_is_zero_radius() {
        let m = mesh(vec![[2.0, -1.0, 5.0]], vec![[0, 0, 0]]);
        let s = bounding_sphere(&m).unwrap();
        assert_eq!(s.center(), [2.0, -1.0, 5.0]);
        assert!(s.radius() <= 1e-6, "radius {}", s.radius());
    }

    #[test]
    fn two_points_give_their_midpoint_and_half_distance() {
        let m = mesh(vec![[-3.0, 0.0, 0.0], [3.0, 0.0, 0.0]], vec![[0, 1, 0]]);
        let s = bounding_sphere(&m).unwrap();
        assert!((s.center()[0] - 0.0).abs() < 1e-5);
        assert!((s.radius() - 3.0).abs() < 1e-5, "radius {}", s.radius());
    }

    #[test]
    fn encloses_all_axis_aligned_box_corners() {
        // Unit cube corners; sphere must contain every corner.
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        // Two triangles are enough to reference all eight corners.
        let indices = vec![[0, 1, 2], [3, 4, 5], [6, 7, 0]];
        let m = mesh(positions.clone(), indices);
        let s = bounding_sphere(&m).unwrap();
        for p in &positions {
            assert!(s.contains(*p), "corner {p:?} outside sphere {s:?}");
        }
        // The exact minimal sphere for a unit cube has radius sqrt(3)/2 ~ 0.866;
        // Ritter stays close to that and never below it.
        let exact = (3.0f64).sqrt() / 2.0;
        assert!(f64::from(s.radius()) >= exact - 1e-4, "radius {}", s.radius());
        assert!(f64::from(s.radius()) <= exact * 1.15, "radius {}", s.radius());
    }

    #[test]
    fn ignores_unreferenced_outlier_vertex() {
        // A far outlier vertex that no triangle references must not inflate.
        let positions = vec![
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1000.0, 1000.0, 1000.0],
        ];
        let indices = vec![[0, 1, 2]];
        let m = mesh(positions, indices);
        let s = bounding_sphere(&m).unwrap();
        assert!(f64::from(s.radius()) < 2.0, "radius {}", s.radius());
        assert!(!s.contains([1000.0, 1000.0, 1000.0]));
    }

    #[test]
    fn contains_rejects_far_exterior_point() {
        let m = mesh(vec![[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], vec![[0, 1, 2]]);
        let s = bounding_sphere(&m).unwrap();
        assert!(!s.contains([100.0, 0.0, 0.0]));
        assert!(s.contains(s.center()));
    }
}
