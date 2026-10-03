//! Concavity depth of a point set relative to its convex hull.
//!
//! [`measure_solidity`](crate::collider::measure_solidity) answers "how much
//! *volume* is lost to concavity" as a ratio. That ratio is blind to shape:
//! a part with a single deep, thin slot loses almost no volume yet is clearly
//! not hull-convex, and a cooker that only looks at solidity would wrongly
//! approximate it with one hull. The complementary measure is *concavity
//! depth*: the largest distance, in world units, by which any point sits
//! inside its own convex hull.
//!
//! `V-HACD` and similar approximate-convex-decomposition cookers use exactly
//! this distance-based concavity to decide how aggressively to split a shape.
//! Here we build the convex hull of the input points, then measure how far each
//! point lies from the hull surface. Hull vertices and points lying on a hull
//! face report zero; points buried in a concave pocket report a large depth.
//! The depth is also reported normalised by the hull's bounding diagonal so the
//! threshold is scale-free.
//!
//! This reuses the existing hull builder ([`convex_hull`](crate::collider::convex_hull))
//! and triangle `BVH` ([`MeshBvh`](crate::collider::MeshBvh)); it is pure
//! point-cloud geometry with no coupling to the collision pipeline, and nothing
//! here is derived from Unreal Engine source.

use glam::Vec3;

use crate::collider::hull::convex_hull;
use crate::collider::mesh_bvh::MeshBvh;

/// Default normalised-concavity threshold under which
/// [`MeshConcavity::is_convex`] still reports convex.
pub const DEFAULT_CONCAVITY_TOLERANCE: f32 = 1.0e-3;

/// Concavity statistics of a point set relative to its convex hull.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshConcavity {
    /// Largest distance from any input point to the hull surface, in world
    /// units. Zero for a convex cloud.
    pub max_concavity: f32,
    /// Mean distance from the input points to the hull surface.
    pub mean_concavity: f32,
    /// Length of the diagonal of the hull's axis-aligned bounding box, used as
    /// the scale reference.
    pub hull_diagonal: f32,
    /// `max_concavity / hull_diagonal`: a scale-free concavity depth. A convex
    /// cloud is `0`; deep pockets drive it up.
    pub normalized_concavity: f32,
}

impl MeshConcavity {
    /// Whether the cloud is convex to within `tolerance` (normalised concavity
    /// at most `tolerance`).
    #[must_use]
    pub fn is_convex(&self, tolerance: f32) -> bool {
        self.normalized_concavity <= tolerance
    }

    /// Whether the cloud is convex to within [`DEFAULT_CONCAVITY_TOLERANCE`].
    #[must_use]
    pub fn is_convex_default(&self) -> bool {
        self.is_convex(DEFAULT_CONCAVITY_TOLERANCE)
    }
}

/// Measures the concavity depth of a point cloud against its convex hull.
///
/// Returns [`None`] when fewer than four points are supplied, when a convex
/// hull cannot be built (all points coplanar or degenerate), or when the hull
/// is too small to establish a scale. The result is deterministic.
#[must_use]
pub fn measure_concavity(points: &[Vec3]) -> Option<MeshConcavity> {
    if points.len() < 4 {
        return None;
    }
    let (hull_vertices, hull_indices) = convex_hull(points)?;
    let bvh = MeshBvh::build(&hull_vertices, &hull_indices)?;

    let (aabb_min, aabb_max) = bvh.local_aabb();
    let hull_diagonal = (aabb_max - aabb_min).length();
    if !(hull_diagonal.is_finite() && hull_diagonal > 0.0) {
        return None;
    }

    let mut max_depth = 0.0_f64;
    let mut sum_depth = 0.0_f64;
    for &p in points {
        let hit = bvh.closest_point(p)?;
        let depth = f64::from(hit.distance_sq).max(0.0).sqrt();
        if depth > max_depth {
            max_depth = depth;
        }
        sum_depth += depth;
    }
    let mean = sum_depth / points.len() as f64;
    let max_concavity = max_depth as f32;

    Some(MeshConcavity {
        max_concavity,
        mean_concavity: mean as f32,
        hull_diagonal,
        normalized_concavity: max_concavity / hull_diagonal,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_cube_corners() -> Vec<Vec3> {
        let mut v = Vec::with_capacity(8);
        for &x in &[-1.0_f32, 1.0] {
            for &y in &[-1.0_f32, 1.0] {
                for &z in &[-1.0_f32, 1.0] {
                    v.push(Vec3::new(x, y, z));
                }
            }
        }
        v
    }

    #[test]
    fn rejects_too_few_or_degenerate_points() {
        assert!(measure_concavity(&[]).is_none());
        assert!(measure_concavity(&[Vec3::ZERO, Vec3::X, Vec3::Y]).is_none());
        // Four coplanar points cannot form a hull with volume.
        let planar = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        assert!(measure_concavity(&planar).is_none());
    }

    #[test]
    fn convex_cube_has_zero_concavity() {
        let report = measure_concavity(&unit_cube_corners()).expect("cube");
        assert!(
            report.max_concavity < 1.0e-5,
            "max = {}",
            report.max_concavity
        );
        assert!(report.normalized_concavity < 1.0e-5);
        assert!(report.is_convex_default());
    }

    #[test]
    fn coplanar_face_subdivision_stays_convex() {
        // Extra points on the cube's faces lie on the hull surface, so they
        // must not register as concavity.
        let mut pts = unit_cube_corners();
        pts.push(Vec3::new(0.0, 0.0, 1.0)); // centre of +z face
        pts.push(Vec3::new(0.0, 1.0, 0.0)); // centre of +y face
        pts.push(Vec3::new(0.5, 0.5, 1.0)); // off-centre on +z face
        let report = measure_concavity(&pts).expect("cube+face points");
        assert!(
            report.max_concavity < 1.0e-4,
            "max = {}",
            report.max_concavity
        );
        assert!(report.is_convex_default());
    }

    #[test]
    fn interior_point_registers_its_depth() {
        // A point at the cube centre is 1.0 from the nearest face; the hull is
        // still the cube, so concavity depth is exactly the half-extent.
        let mut pts = unit_cube_corners();
        pts.push(Vec3::ZERO);
        let report = measure_concavity(&pts).expect("cube+centre");
        assert!(
            (report.max_concavity - 1.0).abs() < 1.0e-4,
            "max = {}",
            report.max_concavity
        );
        // Diagonal of the [-1,1]^3 hull is sqrt(12).
        let expected_norm = 1.0 / 12.0_f32.sqrt();
        assert!((report.normalized_concavity - expected_norm).abs() < 1.0e-4);
        assert!(!report.is_convex_default());
    }

    #[test]
    fn normalized_concavity_is_scale_invariant() {
        let mut pts = unit_cube_corners();
        pts.push(Vec3::ZERO);
        let base = measure_concavity(&pts).expect("base");

        let scaled: Vec<Vec3> = pts.iter().map(|p| *p * 10.0).collect();
        let big = measure_concavity(&scaled).expect("scaled");

        assert!((big.max_concavity - 10.0 * base.max_concavity).abs() < 1.0e-3);
        assert!((big.normalized_concavity - base.normalized_concavity).abs() < 1.0e-4);
    }

    #[test]
    fn result_is_deterministic() {
        let mut pts = unit_cube_corners();
        pts.push(Vec3::new(0.1, -0.2, 0.3));
        let a = measure_concavity(&pts).expect("a");
        let b = measure_concavity(&pts).expect("b");
        assert_eq!(a, b);
    }
}
