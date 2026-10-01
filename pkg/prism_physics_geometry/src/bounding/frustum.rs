//! View frustum (six inward-facing planes) and its culling tests.

use glam::{Mat4, Vec3};

use super::aabb::Aabb;
use super::plane::Plane;
use super::sphere::BoundingSphere;

/// A view frustum defined by six inward-facing [`Plane`]s.
///
/// Plane order is left, right, bottom, top, near, far. Each plane's normal
/// points toward the interior, so a point is inside the frustum exactly when it
/// has a non-negative signed distance to every plane.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Frustum {
    /// The six bounding planes: `[left, right, bottom, top, near, far]`.
    pub planes: [Plane; 6],
}

impl Frustum {
    /// Creates a frustum from six explicit inward-facing planes in the order
    /// left, right, bottom, top, near, far.
    #[inline]
    pub fn new(planes: [Plane; 6]) -> Frustum {
        Frustum { planes }
    }

    /// Extracts the six frustum planes from a combined view-projection (clip)
    /// matrix using the Gribb–Hartmann method.
    ///
    /// `clip` maps world points to OpenGL-style clip space, where the visible
    /// region is `-w <= x, y, z <= w`. The returned planes are normalized so
    /// that signed distances are metric, which [`Frustum::intersects_sphere`]
    /// relies on. This derives the planes algebraically from the matrix and
    /// contains no engine-specific code.
    pub fn from_clip_matrix(clip: Mat4) -> Frustum {
        let r0 = clip.row(0);
        let r1 = clip.row(1);
        let r2 = clip.row(2);
        let r3 = clip.row(3);
        let planes = [
            Plane::from_vec4(r3 + r0).normalized(), // left:   w + x >= 0
            Plane::from_vec4(r3 - r0).normalized(), // right:  w - x >= 0
            Plane::from_vec4(r3 + r1).normalized(), // bottom: w + y >= 0
            Plane::from_vec4(r3 - r1).normalized(), // top:    w - y >= 0
            Plane::from_vec4(r3 + r2).normalized(), // near:   w + z >= 0
            Plane::from_vec4(r3 - r2).normalized(), // far:    w - z >= 0
        ];
        Frustum { planes }
    }

    /// Returns `true` if `p` lies inside or on every frustum plane.
    #[inline]
    pub fn contains_point(&self, p: Vec3) -> bool {
        self.planes.iter().all(|pl| pl.signed_distance(p) >= 0.0)
    }

    /// Returns `true` if `aabb` is not fully culled by any plane.
    ///
    /// This is the standard conservative frustum test: a box is rejected only
    /// when it lies entirely outside one plane. Boxes that straddle the frustum
    /// corner region may be kept even if they are strictly outside, which is
    /// acceptable because the result only needs to be an over-approximation of
    /// visibility for a broad-phase cull.
    #[inline]
    pub fn intersects_aabb(&self, aabb: &Aabb) -> bool {
        !self.planes.iter().any(|pl| pl.aabb_is_outside(aabb))
    }

    /// Returns `true` if `sphere` is not fully culled by any plane.
    ///
    /// A sphere is rejected when its center is farther than its radius onto the
    /// outside of any plane. This requires metric signed distances, so the
    /// planes must be normalized (as produced by [`Frustum::from_clip_matrix`]).
    #[inline]
    pub fn intersects_sphere(&self, sphere: &BoundingSphere) -> bool {
        self.planes
            .iter()
            .all(|pl| pl.signed_distance(sphere.center) >= -sphere.radius)
    }
}

#[cfg(test)]
mod tests {
    use super::Frustum;
    use crate::bounding::{Aabb, BoundingSphere, Plane};
    use glam::{Mat4, Vec3, Vec4};

    /// An axis-aligned unit cube frustum spanning `[-1, 1]` on every axis,
    /// built from six explicit inward-facing planes.
    fn box_frustum() -> Frustum {
        Frustum::new([
            Plane::new(Vec3::X, 1.0),      // left:   x >= -1
            Plane::new(Vec3::NEG_X, 1.0),  // right:  x <=  1
            Plane::new(Vec3::Y, 1.0),      // bottom: y >= -1
            Plane::new(Vec3::NEG_Y, 1.0),  // top:    y <=  1
            Plane::new(Vec3::Z, 1.0),      // near:   z >= -1
            Plane::new(Vec3::NEG_Z, 1.0),  // far:    z <=  1
        ])
    }

    #[test]
    fn point_containment() {
        let f = box_frustum();
        assert!(f.contains_point(Vec3::ZERO));
        assert!(f.contains_point(Vec3::splat(1.0))); // on the corner
        assert!(!f.contains_point(Vec3::new(1.5, 0.0, 0.0)));
        assert!(!f.contains_point(Vec3::splat(-2.0)));
    }

    #[test]
    fn aabb_culling() {
        let f = box_frustum();
        assert!(f.intersects_aabb(&Aabb::new(Vec3::splat(-0.5), Vec3::splat(0.5))));
        // Overlapping one face is kept.
        assert!(f.intersects_aabb(&Aabb::new(Vec3::new(0.5, -0.5, -0.5), Vec3::new(2.0, 0.5, 0.5))));
        // Entirely to the right of the frustum is culled.
        assert!(!f.intersects_aabb(&Aabb::new(Vec3::splat(2.0), Vec3::splat(3.0))));
    }

    #[test]
    fn sphere_culling() {
        let f = box_frustum();
        assert!(f.intersects_sphere(&BoundingSphere::new(Vec3::ZERO, 0.5)));
        // Center just outside, but radius reaches in.
        assert!(f.intersects_sphere(&BoundingSphere::new(Vec3::new(1.4, 0.0, 0.0), 0.5)));
        // Fully outside with no reach.
        assert!(!f.intersects_sphere(&BoundingSphere::new(Vec3::new(3.0, 0.0, 0.0), 0.5)));
    }

    #[test]
    fn from_clip_matrix_matches_clip_space_test() {
        // OpenGL-convention orthographic projection (clip z in [-1, 1]),
        // built explicitly to avoid depth-convention ambiguity. Bounds:
        // x in [-2, 2], y in [-1, 1], z near = 1, far = 10.
        //
        // Row form (OpenGL RH ortho):
        //   r0 = (2/(r-l), 0,       0,        -(r+l)/(r-l))
        //   r1 = (0,       2/(t-b), 0,        -(t+b)/(t-b))
        //   r2 = (0,       0,       -2/(f-n), -(f+n)/(f-n))
        //   r3 = (0,       0,       0,        1)
        // glam is column-major, so columns are the transposed rows.
        let a = 2.0 / 9.0; // 2/(f-n) with f=10, n=1
        let c = 11.0 / 9.0; // (f+n)/(f-n)
        let clip = Mat4::from_cols(
            Vec4::new(0.5, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, -a, 0.0),
            Vec4::new(0.0, 0.0, -c, 1.0),
        );
        let f = Frustum::from_clip_matrix(clip);
        // Sample points and compare the frustum verdict with the direct
        // clip-space inclusion test `-w <= x,y,z <= w` (w = 1 for ortho).
        let samples = [
            Vec3::new(0.0, 0.0, -5.0),
            Vec3::new(1.9, 0.9, -2.0),
            Vec3::new(0.0, 0.0, -0.5), // in front of the near plane
            Vec3::new(3.0, 0.0, -5.0), // beyond the right plane
            Vec3::new(0.0, 0.0, -20.0), // beyond the far plane
        ];
        for p in samples {
            let clip_pt = clip * Vec4::new(p.x, p.y, p.z, 1.0);
            let w = clip_pt.w;
            let inside = clip_pt.x >= -w
                && clip_pt.x <= w
                && clip_pt.y >= -w
                && clip_pt.y <= w
                && clip_pt.z >= -w
                && clip_pt.z <= w;
            assert_eq!(
                f.contains_point(p),
                inside,
                "mismatch at {p:?}: clip = {clip_pt:?}"
            );
        }
    }
}
