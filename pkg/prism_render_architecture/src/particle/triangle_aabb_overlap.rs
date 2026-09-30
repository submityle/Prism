//! Akenine-Möller triangle-versus-axis-aligned-box boolean overlap test for the
//! particle subsystem's zero-dependency, `CPU`-verifiable geometry contracts
//! (design §10, §13).
//!
//! Several particle stages need a cheap, exact yes/no answer to "does this
//! triangle touch this axis-aligned cell/box?": rasterizing an emitter mesh
//! into a spatial-hash grid, binning collider triangles into broadphase cells,
//! and culling a triangle fan against a tile's bounding box. This module owns
//! the small contract they share: given a triangle (three corners) and an
//! axis-aligned box (center plus half-extents), it decides whether the two
//! overlap. It returns a single boolean — nothing else.
//!
//! # Algorithm (13-axis `SAT`)
//! Two convex sets are disjoint if and only if some axis separates their
//! projections. The classic Akenine-Möller test picks thirteen candidate axes
//! for a triangle against a box. First the triangle and box are translated so
//! the box is centered at the origin; then each axis is checked:
//!
//! * the three box face normals (the world `x`, `y`, `z` axes) — this is
//!   exactly the "triangle `AABB` vs box `AABB`" overlap test;
//! * the one triangle face normal — a plane-versus-box side test;
//! * the nine cross products of a triangle edge with a box axis.
//!
//! On every candidate axis the box projects to a symmetric interval of
//! half-width `hx*|ax| + hy*|ay| + hz*|az|` (because it is centered at the
//! origin) while the triangle projects to `[min, max]` of its three vertex
//! dots. If those intervals leave a positive gap on any axis the shapes are
//! disjoint and the routine returns `false`; if no axis separates them they
//! overlap. An exact touch (zero gap) counts as an overlap, so a shared vertex,
//! edge, or coplanar graze all report `true`.
//!
//! # Box input convention
//! The primary entry point takes the box as a **center plus non-negative
//! half-extents**. A [`Aabb::from_min_max`] convenience constructor accepts the
//! `min`/`max` corner form and derives the center and half-extents from it, so
//! callers may use whichever representation they already hold.
//!
//! # Degenerate triangles
//! A triangle that has collapsed to a segment (collinear corners) or a point
//! (coincident corners) is handled gracefully: its face normal and/or its
//! edge-cross axes vanish, and a vanishing axis cannot separate anything, so it
//! is skipped. The remaining axes then reduce the query to the correct
//! segment-versus-box or point-versus-box overlap test for that degenerate
//! geometry.
//!
//! # Relationship to the sibling modules (strict boundary)
//! This file is deliberately disjoint from its neighbours:
//!
//! * [`crate::particle::sphere_aabb`] tests a **sphere** against a box and
//!   returns a closest point, penetration depth, and normal; this module tests
//!   a **triangle** and returns only a boolean.
//! * [`crate::particle::obb_obb_sat_3d`] is the **`OBB`-`OBB`** boolean `SAT`
//!   (fifteen axes, two oriented boxes); this one is a triangle against an
//!   *axis-aligned* box (thirteen axes).
//! * [`crate::particle::sat_collision_2d`] is the **2D** convex-polygon `SAT`
//!   that also produces a minimum translation vector; this module is 3D and
//!   never computes an `MTV`.
//! * [`crate::particle::tri_tri_intersect`] tests a **triangle against another
//!   triangle**; this module tests a triangle against a box.
//!
//! It reports neither a penetration depth nor a separation vector — a single
//! boolean is the entire contract, and it keeps its own tiny vector helpers
//! rather than importing any sibling type.
//!
//! # No transcendental math
//! Every routine is pure `+`, `-`, `*` arithmetic with `f32::abs`, `f32::min`,
//! and `f32::max`. There is no `sqrt`, `sin`, `cos`, `atan`, `exp`, `ln`,
//! `powf`, `ceil`, `round`, or any other transcendental or rounding call, and
//! no `f32` equality: near-zero and near-touching magnitudes are compared
//! against the module epsilons instead of `==` / `!=`.

/// Separation slack added to the summed projection radii. A configuration is
/// reported disjoint only when the projection gap exceeds this tolerance, so an
/// exact vertex/edge/face contact counts as an overlap rather than flickering
/// between states from rounding noise. Used in place of any `f32` `==` / `!=`.
pub const SEP_EPS: f32 = 1.0e-6;

/// Squared-length threshold below which a candidate axis (a triangle face
/// normal or an edge-cross-box-axis) is treated as degenerate and skipped: a
/// vanishing axis cannot separate two sets, so ignoring it is correct and
/// avoids dividing meaning into numerical noise. Compared against a squared
/// length, so it is the square of a `~1e-6` direction tolerance.
pub const DEGENERATE_AXIS_EPS: f32 = 1.0e-12;

/// The three world axes, which double as the box face normals in the local
/// (box-centered) frame used by the test.
const UNIT_AXES: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// Component-wise difference of two 3D points (`lhs - rhs`).
fn v_sub(lhs: [f32; 3], rhs: [f32; 3]) -> [f32; 3] {
    [lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]]
}

/// Cross product `lhs × rhs` of two 3D vectors.
fn v_cross(lhs: [f32; 3], rhs: [f32; 3]) -> [f32; 3] {
    [
        lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[0] * rhs[1] - lhs[1] * rhs[0],
    ]
}

/// Dot product of two 3D vectors.
fn dot(lhs: [f32; 3], rhs: [f32; 3]) -> f32 {
    lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]
}

/// Smallest of three scalars, using only `f32::min`.
fn min3(first: f32, second: f32, third: f32) -> f32 {
    first.min(second).min(third)
}

/// Largest of three scalars, using only `f32::max`.
fn max3(first: f32, second: f32, third: f32) -> f32 {
    first.max(second).max(third)
}

/// Returns `true` when the box face axis `world_index` separates the triangle
/// from the (origin-centered) box, given the triangle vertex coordinates along
/// that axis and the box half-extent along it.
fn face_separates(coord0: f32, coord1: f32, coord2: f32, half_extent: f32) -> bool {
    let lo = min3(coord0, coord1, coord2);
    let hi = max3(coord0, coord1, coord2);
    lo > half_extent + SEP_EPS || hi < -half_extent - SEP_EPS
}

/// Returns `true` when `axis` is a separating axis for the triangle
/// `v0`/`v1`/`v2` (already expressed in the box-centered frame) against the box
/// with the given `half` extents. A degenerate (near-zero) axis can never
/// separate, so it returns `false`.
fn axis_separates(
    axis: [f32; 3],
    v0: [f32; 3],
    v1: [f32; 3],
    v2: [f32; 3],
    half: [f32; 3],
) -> bool {
    if dot(axis, axis) < DEGENERATE_AXIS_EPS {
        return false;
    }
    let p0 = dot(axis, v0);
    let p1 = dot(axis, v1);
    let p2 = dot(axis, v2);
    let lo = min3(p0, p1, p2);
    let hi = max3(p0, p1, p2);
    let radius = half[0] * axis[0].abs() + half[1] * axis[1].abs() + half[2] * axis[2].abs();
    lo > radius + SEP_EPS || hi < -radius - SEP_EPS
}

/// Tests whether a triangle overlaps an axis-aligned box given as a `center`
/// and non-negative `half` extents.
///
/// The triangle is `[v0, v1, v2]`, each a 3D point. Returns `true` when the two
/// shapes intersect or merely touch (shared vertex, edge, or coplanar graze),
/// and `false` when a separating axis proves they are disjoint. Negative
/// components of `half` are treated by their magnitude so a mis-signed extent
/// does not silently invert the box.
#[must_use]
pub fn triangle_overlaps_aabb(tri: &[[f32; 3]; 3], center: [f32; 3], half: [f32; 3]) -> bool {
    let half = [half[0].abs(), half[1].abs(), half[2].abs()];

    // Move the triangle into the box-centered frame.
    let v0 = v_sub(tri[0], center);
    let v1 = v_sub(tri[1], center);
    let v2 = v_sub(tri[2], center);

    // Axes 1-3: the box face normals (triangle AABB vs box AABB).
    if face_separates(v0[0], v1[0], v2[0], half[0]) {
        return false;
    }
    if face_separates(v0[1], v1[1], v2[1], half[1]) {
        return false;
    }
    if face_separates(v0[2], v1[2], v2[2], half[2]) {
        return false;
    }

    // Triangle edges in the box-centered frame.
    let e0 = v_sub(v1, v0);
    let e1 = v_sub(v2, v1);
    let e2 = v_sub(v0, v2);

    // Axis 4: the triangle face normal (plane vs box).
    let normal = v_cross(e0, e1);
    if axis_separates(normal, v0, v1, v2, half) {
        return false;
    }

    // Axes 5-13: each triangle edge crossed with each box axis.
    let edges = [e0, e1, e2];
    for edge in &edges {
        for unit in &UNIT_AXES {
            if axis_separates(v_cross(*edge, *unit), v0, v1, v2, half) {
                return false;
            }
        }
    }

    true
}

/// An axis-aligned bounding box stored as a center and non-negative
/// half-extents. This mirrors the primary [`triangle_overlaps_aabb`] input and
/// offers a `min`/`max` constructor for callers that hold the corner form.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    /// Box center.
    pub center: [f32; 3],
    /// Non-negative half-extents along each axis.
    pub half: [f32; 3],
}

impl Aabb {
    /// Builds a box directly from a center and half-extents. Half-extents are
    /// stored by magnitude so a mis-signed input cannot invert the box.
    #[must_use]
    pub fn from_center_half(center: [f32; 3], half: [f32; 3]) -> Self {
        Self {
            center,
            half: [half[0].abs(), half[1].abs(), half[2].abs()],
        }
    }

    /// Builds a box from its `min` and `max` corners. The center is the corner
    /// midpoint and each half-extent is half the corner span; the span is taken
    /// by magnitude so swapped corners still yield a valid box.
    #[must_use]
    pub fn from_min_max(min: [f32; 3], max: [f32; 3]) -> Self {
        let center = [
            (min[0] + max[0]) * 0.5,
            (min[1] + max[1]) * 0.5,
            (min[2] + max[2]) * 0.5,
        ];
        let half = [
            (max[0] - min[0]).abs() * 0.5,
            (max[1] - min[1]).abs() * 0.5,
            (max[2] - min[2]).abs() * 0.5,
        ];
        Self { center, half }
    }

    /// Returns `true` when `tri` overlaps or touches this box.
    #[must_use]
    pub fn overlaps_triangle(&self, tri: &[[f32; 3]; 3]) -> bool {
        triangle_overlaps_aabb(tri, self.center, self.half)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The canonical unit box: centered at the origin with half-extents of 1,
    /// i.e. the cube spanning `[-1, 1]` on every axis.
    fn unit_box() -> ([f32; 3], [f32; 3]) {
        ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    }

    #[test]
    fn triangle_fully_inside_overlaps() {
        let (c, h) = unit_box();
        let tri = [[-0.5, -0.5, 0.0], [0.5, -0.25, 0.1], [0.0, 0.5, -0.2]];
        assert!(triangle_overlaps_aabb(&tri, c, h));
    }

    #[test]
    fn triangle_far_on_plus_x_is_separated() {
        let (c, h) = unit_box();
        let tri = [[3.0, 0.0, 0.0], [4.0, 1.0, 0.0], [3.5, -1.0, 1.0]];
        assert!(!triangle_overlaps_aabb(&tri, c, h));
    }

    #[test]
    fn triangle_far_on_minus_y_is_separated() {
        let (c, h) = unit_box();
        let tri = [[0.0, -3.0, 0.0], [1.0, -4.0, 0.0], [-1.0, -3.5, 0.5]];
        assert!(!triangle_overlaps_aabb(&tri, c, h));
    }

    #[test]
    fn triangle_far_on_plus_z_is_separated() {
        let (c, h) = unit_box();
        let tri = [[0.0, 0.0, 3.0], [1.0, 0.0, 4.0], [0.0, 1.0, 3.5]];
        assert!(!triangle_overlaps_aabb(&tri, c, h));
    }

    #[test]
    fn triangle_crossing_plus_x_face_overlaps() {
        let (c, h) = unit_box();
        // Spans from inside the box out through the +x face.
        let tri = [[0.0, 0.0, 0.0], [2.0, 0.5, 0.0], [2.0, -0.5, 0.0]];
        assert!(triangle_overlaps_aabb(&tri, c, h));
    }

    #[test]
    fn triangle_vertex_touching_face_overlaps() {
        let (c, h) = unit_box();
        // One vertex sits exactly on the +x face; the rest are outside.
        let tri = [[1.0, 0.0, 0.0], [3.0, 1.0, 0.0], [3.0, -1.0, 0.0]];
        assert!(triangle_overlaps_aabb(&tri, c, h));
    }

    #[test]
    fn triangle_vertex_touching_corner_overlaps() {
        let (c, h) = unit_box();
        // One vertex sits exactly on the (1,1,1) corner; the rest are outside.
        let tri = [[1.0, 1.0, 1.0], [3.0, 2.0, 1.0], [2.0, 3.0, 1.5]];
        assert!(triangle_overlaps_aabb(&tri, c, h));
    }

    #[test]
    fn triangle_edge_lying_on_face_plane_overlaps() {
        let (c, h) = unit_box();
        // An edge lies in the x = 1 face plane, grazing the box.
        let tri = [[1.0, -0.5, 0.0], [1.0, 0.5, 0.0], [2.0, 0.0, 0.0]];
        assert!(triangle_overlaps_aabb(&tri, c, h));
    }

    #[test]
    fn big_triangle_enclosing_box_overlaps() {
        let (c, h) = unit_box();
        // A large triangle in the z = 0 plane whose interior covers the box.
        let tri = [[-10.0, -10.0, 0.0], [10.0, -10.0, 0.0], [0.0, 10.0, 0.0]];
        assert!(triangle_overlaps_aabb(&tri, c, h));
    }

    #[test]
    fn coplanar_triangle_on_face_overlaps() {
        let (c, h) = unit_box();
        // Triangle lies in the x = 1 face plane and covers part of that face.
        let tri = [[1.0, -0.5, -0.5], [1.0, 0.5, -0.5], [1.0, 0.0, 0.5]];
        assert!(triangle_overlaps_aabb(&tri, c, h));
    }

    #[test]
    fn coplanar_triangle_beyond_face_is_separated() {
        let (c, h) = unit_box();
        // Same shape shifted to x = 1.5: now beyond the +x face.
        let tri = [[1.5, -0.5, -0.5], [1.5, 0.5, -0.5], [1.5, 0.0, 0.5]];
        assert!(!triangle_overlaps_aabb(&tri, c, h));
    }

    #[test]
    fn triangle_slicing_diagonal_overlaps() {
        let (c, h) = unit_box();
        // Passes through the interior along a body diagonal.
        let tri = [[-2.0, -2.0, -2.0], [2.0, 2.0, -2.0], [0.0, 0.0, 2.0]];
        assert!(triangle_overlaps_aabb(&tri, c, h));
    }

    #[test]
    fn degenerate_point_inside_overlaps() {
        let (c, h) = unit_box();
        let p = [0.25, -0.5, 0.75];
        let tri = [p, p, p];
        assert!(triangle_overlaps_aabb(&tri, c, h));
    }

    #[test]
    fn degenerate_point_on_corner_overlaps() {
        let (c, h) = unit_box();
        let p = [1.0, 1.0, 1.0];
        let tri = [p, p, p];
        assert!(triangle_overlaps_aabb(&tri, c, h));
    }

    #[test]
    fn degenerate_point_outside_is_separated() {
        let (c, h) = unit_box();
        let p = [5.0, 0.0, 0.0];
        let tri = [p, p, p];
        assert!(!triangle_overlaps_aabb(&tri, c, h));
    }

    #[test]
    fn degenerate_segment_through_box_overlaps() {
        let (c, h) = unit_box();
        // Collinear vertices form a segment that passes through the box.
        let tri = [[-3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
        assert!(triangle_overlaps_aabb(&tri, c, h));
    }

    #[test]
    fn degenerate_segment_outside_is_separated() {
        let (c, h) = unit_box();
        // Collinear vertices, entirely on the far +y side.
        let tri = [[-3.0, 5.0, 0.0], [0.0, 5.0, 0.0], [3.0, 5.0, 0.0]];
        assert!(!triangle_overlaps_aabb(&tri, c, h));
    }

    #[test]
    fn triangle_face_normal_axis_separates() {
        let (c, h) = unit_box();
        // Large triangle in the plane x + y + z = 4. Its AABB overlaps the box
        // on every world axis and its edge-cross axes do not separate, so the
        // triangle FACE NORMAL (1,1,1) is the only separating axis (box corner
        // sum peaks at 3 < 4).
        let tri = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]];
        assert!(!triangle_overlaps_aabb(&tri, c, h));
    }

    #[test]
    fn triangle_face_normal_plane_touching_overlaps() {
        let (c, h) = unit_box();
        // Same family but the plane x + y + z = 3 touches the (1,1,1) corner.
        let tri = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]];
        assert!(triangle_overlaps_aabb(&tri, c, h));
    }

    #[test]
    fn edge_cross_z_axis_separates() {
        let (c, h) = unit_box();
        // In the z = 0 plane, hypotenuse x + y = 4.2 leaves the box corner
        // (x+y peaks at 2) separated only along the edge-cross axis (1,1,0).
        let tri = [[2.1, 0.1, 0.0], [0.1, 2.1, 0.0], [2.1, 2.1, 0.0]];
        assert!(!triangle_overlaps_aabb(&tri, c, h));
    }

    #[test]
    fn edge_cross_z_axis_touching_overlaps() {
        let (c, h) = unit_box();
        // Hypotenuse x + y = 2 now touches the box corner (1,1).
        let tri = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [2.0, 2.0, 0.0]];
        assert!(triangle_overlaps_aabb(&tri, c, h));
    }

    #[test]
    fn edge_cross_x_axis_separates() {
        let (c, h) = unit_box();
        // Same idea rotated into the x = 0 plane: y + z = 4.2 separates only on
        // the edge-cross axis (0,1,1).
        let tri = [[0.0, 2.1, 0.1], [0.0, 0.1, 2.1], [0.0, 2.1, 2.1]];
        assert!(!triangle_overlaps_aabb(&tri, c, h));
    }

    #[test]
    fn edge_cross_x_axis_touching_overlaps() {
        let (c, h) = unit_box();
        let tri = [[0.0, 2.0, 0.0], [0.0, 0.0, 2.0], [0.0, 2.0, 2.0]];
        assert!(triangle_overlaps_aabb(&tri, c, h));
    }

    #[test]
    fn edge_cross_y_axis_separates() {
        let (c, h) = unit_box();
        // Rotated into the y = 0 plane: x + z = 4.2 separates only on the
        // edge-cross axis (1,0,1).
        let tri = [[2.1, 0.0, 0.1], [0.1, 0.0, 2.1], [2.1, 0.0, 2.1]];
        assert!(!triangle_overlaps_aabb(&tri, c, h));
    }

    #[test]
    fn edge_cross_y_axis_touching_overlaps() {
        let (c, h) = unit_box();
        let tri = [[2.0, 0.0, 0.0], [0.0, 0.0, 2.0], [2.0, 0.0, 2.0]];
        assert!(triangle_overlaps_aabb(&tri, c, h));
    }

    #[test]
    fn translation_invariance() {
        let (_c, h) = unit_box();
        let shift = [7.0, -3.0, 2.0];
        let tri = [[2.1, 0.1, 0.0], [0.1, 2.1, 0.0], [2.1, 2.1, 0.0]];
        let moved = [
            v_add(tri[0], shift),
            v_add(tri[1], shift),
            v_add(tri[2], shift),
        ];
        let base = triangle_overlaps_aabb(&tri, [0.0, 0.0, 0.0], h);
        let shifted = triangle_overlaps_aabb(&moved, shift, h);
        assert_eq!(base, shifted);
    }

    /// Local vector add used only by the translation-invariance test.
    fn v_add(lhs: [f32; 3], rhs: [f32; 3]) -> [f32; 3] {
        [lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]]
    }

    #[test]
    fn offset_box_via_center_half() {
        // Box centered at (5,5,5), half 1 => spans [4,6]^3.
        let center = [5.0, 5.0, 5.0];
        let half = [1.0, 1.0, 1.0];
        let inside = [[4.5, 5.0, 5.0], [5.5, 5.0, 5.5], [5.0, 5.5, 4.5]];
        let outside = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        assert!(triangle_overlaps_aabb(&inside, center, half));
        assert!(!triangle_overlaps_aabb(&outside, center, half));
    }

    #[test]
    fn aabb_from_min_max_matches_center_half() {
        let by_corners = Aabb::from_min_max([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        let by_center = Aabb::from_center_half([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert_eq!(by_corners, by_center);
        let tri = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]];
        assert!(by_corners.overlaps_triangle(&tri));
    }

    #[test]
    fn aabb_from_min_max_handles_swapped_corners() {
        // Swapped corners must still describe the same box.
        let normal = Aabb::from_min_max([-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]);
        let swapped = Aabb::from_min_max([2.0, 2.0, 2.0], [-2.0, -2.0, -2.0]);
        assert_eq!(normal, swapped);
    }

    #[test]
    fn negative_half_extent_is_taken_by_magnitude() {
        let tri = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]];
        let with_pos = triangle_overlaps_aabb(&tri, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let with_neg = triangle_overlaps_aabb(&tri, [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]);
        assert_eq!(with_pos, with_neg);
        assert!(with_pos);
    }

    #[test]
    fn non_cubic_box_respects_each_extent() {
        // Thin slab: half extents (2, 0.1, 2). A triangle at y = 0.5 clears it.
        let center = [0.0, 0.0, 0.0];
        let half = [2.0, 0.1, 2.0];
        let above = [[-1.0, 0.5, -1.0], [1.0, 0.5, -1.0], [0.0, 0.5, 1.0]];
        let through = [[-1.0, -0.5, 0.0], [1.0, -0.5, 0.0], [0.0, 0.5, 0.0]];
        assert!(!triangle_overlaps_aabb(&above, center, half));
        assert!(triangle_overlaps_aabb(&through, center, half));
    }

    #[test]
    fn tilted_triangle_near_corner_is_separated() {
        let (c, h) = unit_box();
        // A small tilted triangle hovering just past the (1,1,1) corner along
        // the (1,1,1) direction: separated but AABBs still nearly touch.
        let tri = [[1.4, 1.0, 1.0], [1.0, 1.4, 1.0], [1.0, 1.0, 1.4]];
        assert!(!triangle_overlaps_aabb(&tri, c, h));
    }

    #[test]
    fn tilted_triangle_cutting_corner_overlaps() {
        let (c, h) = unit_box();
        // Same tilt pulled in so the plane slices off the (1,1,1) corner.
        let tri = [[1.2, 0.6, 0.6], [0.6, 1.2, 0.6], [0.6, 0.6, 1.2]];
        assert!(triangle_overlaps_aabb(&tri, c, h));
    }

    #[test]
    fn shared_axis_epsilon_touch_counts_as_overlap() {
        let (c, h) = unit_box();
        // Hypotenuse x + y = 2 + tiny(< SEP_EPS): still reported as touching.
        let tri = [
            [2.0 + 5.0e-7, 0.0, 0.0],
            [0.0, 2.0 + 5.0e-7, 0.0],
            [2.0, 2.0, 0.0],
        ];
        assert!(triangle_overlaps_aabb(&tri, c, h));
    }
}
