//! Frustum and occlusion culling decisions for geometry clusters.
//!
//! Culling runs before LOD selection and page residency: a cluster that leaves
//! the view frustum, or that a nearer occluder fully hides, needs neither a LOD
//! nor a resident page. This layer stays GPU-independent and trig/sqrt-free —
//! frustum planes arrive already normalized from the render layer (which owns
//! libm-deterministic math), mirroring the projection convention in
//! [`super::lod`]. The tests here are conservative: they never cull a cluster
//! that could contribute a pixel, at the cost of occasionally keeping one that
//! a tighter test would reject.

use crate::gpu_scene::SceneBounds;

/// An inward-facing frustum plane.
///
/// The half-space `dot(normal, p) + distance >= 0` is the interior. `normal`
/// must be unit length so the signed distance is expressed in world units and
/// the AABB projected-radius test stays exact.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plane {
    pub normal: [f32; 3],
    pub distance: f32,
}

impl Plane {
    /// Builds a plane from a (unit) normal and signed distance to the origin.
    #[must_use]
    pub const fn new(normal: [f32; 3], distance: f32) -> Self {
        Self { normal, distance }
    }

    /// Signed distance from `point` to the plane; positive is interior.
    #[must_use]
    pub fn signed_distance(&self, point: [f32; 3]) -> f32 {
        self.normal[0] * point[0]
            + self.normal[1] * point[1]
            + self.normal[2] * point[2]
            + self.distance
    }

    /// Projection of an AABB's half-extents onto `|normal|`, i.e. the AABB's
    /// support radius along this plane's axis.
    #[must_use]
    fn projected_extent(&self, half_extents: [f32; 3]) -> f32 {
        self.normal[0].abs() * half_extents[0]
            + self.normal[1].abs() * half_extents[1]
            + self.normal[2].abs() * half_extents[2]
    }
}

/// Six inward-facing frustum planes (order is not significant).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frustum {
    pub planes: [Plane; 6],
}

impl Frustum {
    /// Wraps six already-normalized inward-facing planes.
    #[must_use]
    pub const fn from_planes(planes: [Plane; 6]) -> Self {
        Self { planes }
    }

    /// Whether a bounding sphere is at least partially inside the frustum.
    #[must_use]
    pub fn contains_sphere(&self, center: [f32; 3], radius: f32) -> bool {
        self.planes
            .iter()
            .all(|plane| plane.signed_distance(center) >= -radius)
    }

    /// Whether an AABB (from [`SceneBounds::half_extents`]) is at least
    /// partially inside the frustum, using the exact projected-radius test.
    #[must_use]
    pub fn intersects_bounds(&self, bounds: &SceneBounds) -> bool {
        self.planes.iter().all(|plane| {
            plane.signed_distance(bounds.center) >= -plane.projected_extent(bounds.half_extents)
        })
    }
}

/// Conservative occlusion probe for a cluster's screen footprint.
///
/// Depths are view-space linear distances where a smaller value is nearer. A
/// cluster is occluded when its nearest point is strictly farther than the
/// nearest occluder covering its footprint (e.g. a hierarchical-Z sample).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OcclusionProbe {
    /// Depth of the cluster's closest point (smaller is nearer).
    pub closest_depth: f32,
    /// Conservative nearest occluder depth over the cluster footprint.
    pub occluder_depth: f32,
}

impl OcclusionProbe {
    /// Whether the cluster is fully behind the nearest occluder.
    #[must_use]
    pub fn is_occluded(&self) -> bool {
        self.closest_depth > self.occluder_depth
    }
}

/// Outcome of culling one cluster.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CullVerdict {
    /// Passes frustum and occlusion; submit it.
    Visible,
    /// Outside the view frustum.
    #[default]
    FrustumCulled,
    /// Inside the frustum but fully hidden by a nearer occluder.
    OcclusionCulled,
}

/// Culls a cluster against the frustum and an optional occlusion probe.
///
/// Frustum rejection takes precedence; occlusion is only consulted for clusters
/// that survive the frustum test. Passing `None` for `occlusion` skips the
/// occlusion phase (e.g. the first depth-prepass wave with no HZB yet).
#[must_use]
pub fn cluster_cull(
    frustum: &Frustum,
    bounds: &SceneBounds,
    occlusion: Option<OcclusionProbe>,
) -> CullVerdict {
    if !frustum.intersects_bounds(bounds) {
        return CullVerdict::FrustumCulled;
    }
    if let Some(probe) = occlusion
        && probe.is_occluded()
    {
        return CullVerdict::OcclusionCulled;
    }
    CullVerdict::Visible
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Axis-aligned box frustum: |x| <= 10, |y| <= 10, 0 <= z <= 100.
    fn box_frustum() -> Frustum {
        Frustum::from_planes([
            Plane::new([1.0, 0.0, 0.0], 10.0),
            Plane::new([-1.0, 0.0, 0.0], 10.0),
            Plane::new([0.0, 1.0, 0.0], 10.0),
            Plane::new([0.0, -1.0, 0.0], 10.0),
            Plane::new([0.0, 0.0, 1.0], 0.0),
            Plane::new([0.0, 0.0, -1.0], 100.0),
        ])
    }

    fn bounds(center: [f32; 3], half: [f32; 3]) -> SceneBounds {
        SceneBounds {
            center,
            radius: (half[0] * half[0] + half[1] * half[1] + half[2] * half[2]).max(0.0),
            half_extents: half,
            _padding: 0.0,
        }
    }

    #[test]
    fn inside_cluster_is_visible() {
        let f = box_frustum();
        let b = bounds([0.0, 0.0, 50.0], [1.0, 1.0, 1.0]);
        assert!(f.intersects_bounds(&b));
        assert_eq!(cluster_cull(&f, &b, None), CullVerdict::Visible);
    }

    #[test]
    fn outside_cluster_is_frustum_culled() {
        let f = box_frustum();
        let b = bounds([100.0, 0.0, 50.0], [1.0, 1.0, 1.0]);
        assert!(!f.intersects_bounds(&b));
        assert_eq!(cluster_cull(&f, &b, None), CullVerdict::FrustumCulled);
    }

    #[test]
    fn straddling_boundary_is_kept() {
        let f = box_frustum();
        // Centre exactly on the +x=10 face, extent 2 straddles it.
        let b = bounds([10.0, 0.0, 50.0], [2.0, 1.0, 1.0]);
        assert!(f.intersects_bounds(&b));
    }

    #[test]
    fn occluded_cluster_reports_occlusion() {
        let f = box_frustum();
        let b = bounds([0.0, 0.0, 60.0], [1.0, 1.0, 1.0]);
        let probe = OcclusionProbe {
            closest_depth: 60.0,
            occluder_depth: 50.0,
        };
        assert_eq!(
            cluster_cull(&f, &b, Some(probe)),
            CullVerdict::OcclusionCulled
        );
        let visible = OcclusionProbe {
            closest_depth: 40.0,
            occluder_depth: 50.0,
        };
        assert_eq!(cluster_cull(&f, &b, Some(visible)), CullVerdict::Visible);
    }

    #[test]
    fn sphere_test_matches_frustum_extent() {
        let f = box_frustum();
        assert!(f.contains_sphere([0.0, 0.0, 50.0], 1.0));
        assert!(!f.contains_sphere([15.0, 0.0, 50.0], 1.0));
        // A large sphere straddling the face is retained.
        assert!(f.contains_sphere([12.0, 0.0, 50.0], 3.0));
    }
}
