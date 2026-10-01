//! Triangle versus axis-aligned-box overlap via the separating-axis theorem.
//!
//! This is the classic 13-axis Akenine-Möller triangle/box test: three box
//! face normals, one triangle face normal, and the nine cross products of the
//! box axes with the triangle edges. All work happens in box-local space with
//! the box centred at the origin, so each axis test reduces to projecting the
//! three triangle vertices and comparing against the box's projection radius.
//!
//! It is an engine-agnostic implementation of a publicly documented algorithm
//! and contains no Unreal Engine source or derived code. The routine uses only
//! dot/cross products and absolute values, so it is exact up to floating-point
//! rounding and allocation free.

use glam::Vec3;

use crate::bounding::Aabb;

/// Returns `true` when the triangle `(a, b, c)` overlaps `aabb`.
///
/// A shared boundary (touching face, edge, or vertex) counts as an overlap.
/// Degenerate triangles are handled gracefully: a zero-length edge simply
/// drops the cross-product axes it cannot define, and the remaining box-face
/// and triangle-face axes still decide the result.
pub fn triangle_aabb_overlap(a: Vec3, b: Vec3, c: Vec3, aabb: &Aabb) -> bool {
    let center = aabb.center();
    let h = aabb.half_extents();

    // Triangle vertices relative to the box centre.
    let v0 = a - center;
    let v1 = b - center;
    let v2 = c - center;

    // Triangle edge vectors.
    let f0 = v1 - v0;
    let f1 = v2 - v1;
    let f2 = v0 - v2;

    // Nine edge-cross axes: e_i x f_j for box axes e_x, e_y, e_z. A near-zero
    // axis (edge parallel to a box axis) cannot separate, so it is skipped.
    for f in [f0, f1, f2] {
        // e_x x f = (0, -f.z, f.y)
        if separated_on_axis(Vec3::new(0.0, -f.z, f.y), v0, v1, v2, h) {
            return false;
        }
        // e_y x f = (f.z, 0, -f.x)
        if separated_on_axis(Vec3::new(f.z, 0.0, -f.x), v0, v1, v2, h) {
            return false;
        }
        // e_z x f = (-f.y, f.x, 0)
        if separated_on_axis(Vec3::new(-f.y, f.x, 0.0), v0, v1, v2, h) {
            return false;
        }
    }

    // Three box face normals: the triangle's AABB must overlap the box.
    if v0.x.min(v1.x).min(v2.x) > h.x || v0.x.max(v1.x).max(v2.x) < -h.x {
        return false;
    }
    if v0.y.min(v1.y).min(v2.y) > h.y || v0.y.max(v1.y).max(v2.y) < -h.y {
        return false;
    }
    if v0.z.min(v1.z).min(v2.z) > h.z || v0.z.max(v1.z).max(v2.z) < -h.z {
        return false;
    }

    // Triangle face normal: plane/box overlap. The box (centred at the origin)
    // meets the triangle plane iff the plane's signed offset is within the
    // box's projection radius onto the normal.
    let normal = f0.cross(f1);
    let r = h.x * normal.x.abs() + h.y * normal.y.abs() + h.z * normal.z.abs();
    if normal.dot(v0).abs() > r {
        return false;
    }

    true
}

/// Minimum-translation `(normal, depth)` that separates the triangle
/// `(a, b, c)` from `aabb`, or [`None`] when they do not overlap.
///
/// This is the depth-reporting companion to [`triangle_aabb_overlap`]: it runs
/// the same 13-axis separating-axis test but, instead of stopping at the first
/// separating axis, keeps the signed overlap on every axis and returns the
/// axis of *least* penetration. The returned `normal` is a unit vector in the
/// box's frame pointing from the triangle toward the box centre, so
/// translating the box along `normal` by `depth` lifts it clear of the
/// triangle; `depth` is always `>= 0`. Axes degenerate to zero length (a
/// triangle edge parallel to a box axis, or a degenerate triangle) are skipped
/// because they cannot separate.
pub fn triangle_aabb_penetration(a: Vec3, b: Vec3, c: Vec3, aabb: &Aabb) -> Option<(Vec3, f32)> {
    let center = aabb.center();
    let h = aabb.half_extents();

    let v0 = a - center;
    let v1 = b - center;
    let v2 = c - center;

    let f0 = v1 - v0;
    let f1 = v2 - v1;
    let f2 = v0 - v2;

    // The 13 candidate axes: three box faces, the triangle face normal, and
    // the nine box-edge x triangle-edge cross products.
    let axes = [
        Vec3::X,
        Vec3::Y,
        Vec3::Z,
        f0.cross(f1),
        Vec3::new(0.0, -f0.z, f0.y),
        Vec3::new(f0.z, 0.0, -f0.x),
        Vec3::new(-f0.y, f0.x, 0.0),
        Vec3::new(0.0, -f1.z, f1.y),
        Vec3::new(f1.z, 0.0, -f1.x),
        Vec3::new(-f1.y, f1.x, 0.0),
        Vec3::new(0.0, -f2.z, f2.y),
        Vec3::new(f2.z, 0.0, -f2.x),
        Vec3::new(-f2.y, f2.x, 0.0),
    ];

    let mut best_depth = f32::INFINITY;
    let mut best_normal = Vec3::ZERO;

    for axis in axes {
        match penetration_on_axis(axis, v0, v1, v2, h) {
            AxisPenetration::Separated => return None,
            AxisPenetration::Skip => {}
            AxisPenetration::Overlap { normal, depth } => {
                if depth < best_depth {
                    best_depth = depth;
                    best_normal = normal;
                }
            }
        }
    }

    if best_normal == Vec3::ZERO {
        None
    } else {
        Some((best_normal, best_depth))
    }
}

/// Outcome of projecting the triangle and box onto a single axis.
enum AxisPenetration {
    /// The intervals are disjoint: the shapes do not overlap at all.
    Separated,
    /// The axis is degenerate (zero length) and cannot decide anything.
    Skip,
    /// The intervals overlap; `normal` (unit, box-centre-ward) and `depth` give
    /// the minimum push that separates the box along this axis.
    Overlap { normal: Vec3, depth: f32 },
}

/// Projects the (box-local) triangle vertices and the centred box onto `axis`
/// and reports their penetration.
///
/// The box interval is `[-r, r]` with `r = sum h_i * |axis_i|`; the triangle
/// interval is `[tri_min, tri_max]`. The two ways to separate are to push the
/// triangle to `+axis` (cost `r - tri_min`) or to `-axis` (cost
/// `tri_max + r`); the cheaper one is the per-axis penetration. The returned
/// `normal` points the way the *box* must move (opposite the triangle push),
/// i.e. from the triangle toward the box centre.
fn penetration_on_axis(axis: Vec3, v0: Vec3, v1: Vec3, v2: Vec3, h: Vec3) -> AxisPenetration {
    let len2 = axis.length_squared();
    if len2 <= 1e-12 {
        return AxisPenetration::Skip;
    }
    let n = axis * len2.sqrt().recip();
    let p0 = v0.dot(n);
    let p1 = v1.dot(n);
    let p2 = v2.dot(n);
    let tri_min = p0.min(p1).min(p2);
    let tri_max = p0.max(p1).max(p2);
    let r = h.x * n.x.abs() + h.y * n.y.abs() + h.z * n.z.abs();

    let push_pos = r - tri_min; // shift triangle along +n to clear
    let push_neg = tri_max + r; // shift triangle along -n to clear
    if push_pos <= 0.0 || push_neg <= 0.0 {
        return AxisPenetration::Separated;
    }

    // Cheaper push wins; the box moves opposite the triangle, so the box-ward
    // normal is the negated triangle push direction.
    if push_pos <= push_neg {
        AxisPenetration::Overlap {
            normal: -n,
            depth: push_pos,
        }
    } else {
        AxisPenetration::Overlap {
            normal: n,
            depth: push_neg,
        }
    }
}

/// Returns `true` when the triangle and box are separated along `axis`.
///
/// Projects the three (box-local) triangle vertices onto `axis` and compares
/// their span to the box's projection radius `r = sum h_i * |axis_i|`. A
/// degenerate axis (produced when a triangle edge is parallel to a box axis)
/// has no projection extent and never separates.
fn separated_on_axis(axis: Vec3, v0: Vec3, v1: Vec3, v2: Vec3, h: Vec3) -> bool {
    if axis.length_squared() <= 1e-12 {
        return false;
    }
    let p0 = v0.dot(axis);
    let p1 = v1.dot(axis);
    let p2 = v2.dot(axis);
    let min = p0.min(p1).min(p2);
    let max = p0.max(p1).max(p2);
    let r = h.x * axis.x.abs() + h.y * axis.y.abs() + h.z * axis.z.abs();
    min > r || max < -r
}

#[cfg(test)]
mod tests {
    use super::{triangle_aabb_overlap, triangle_aabb_penetration};
    use crate::bounding::Aabb;
    use glam::Vec3;

    fn unit_box() -> Aabb {
        Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0))
    }

    #[test]
    fn triangle_crossing_box_overlaps() {
        // A triangle whose face slices through the box interior.
        let a = Vec3::new(-2.0, 0.0, 0.0);
        let b = Vec3::new(2.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 2.0, 0.5);
        assert!(triangle_aabb_overlap(a, b, c, &unit_box()));
    }

    #[test]
    fn triangle_contained_in_box_overlaps() {
        let a = Vec3::new(-0.3, -0.3, 0.0);
        let b = Vec3::new(0.3, -0.3, 0.0);
        let c = Vec3::new(0.0, 0.3, 0.0);
        assert!(triangle_aabb_overlap(a, b, c, &unit_box()));
    }

    #[test]
    fn box_corner_poking_triangle_overlaps() {
        // Triangle plane just clips the +x +y +z corner region.
        let a = Vec3::new(2.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 2.0, 0.0);
        let c = Vec3::new(0.0, 0.0, 2.0);
        // Plane x+y+z = 2 passes through (1,1,0) etc., inside the box corner.
        assert!(triangle_aabb_overlap(a, b, c, &unit_box()));
    }

    #[test]
    fn triangle_far_away_is_separated() {
        let a = Vec3::new(10.0, 10.0, 10.0);
        let b = Vec3::new(11.0, 10.0, 10.0);
        let c = Vec3::new(10.0, 11.0, 10.0);
        assert!(!triangle_aabb_overlap(a, b, c, &unit_box()));
    }

    #[test]
    fn coplanar_triangle_beyond_face_is_separated() {
        // Triangle sits in the plane z = 5, far above the box.
        let a = Vec3::new(-1.0, -1.0, 5.0);
        let b = Vec3::new(1.0, -1.0, 5.0);
        let c = Vec3::new(0.0, 1.0, 5.0);
        assert!(!triangle_aabb_overlap(a, b, c, &unit_box()));
    }

    #[test]
    fn edge_axis_separation_is_detected() {
        // A thin skew triangle that straddles the box in its AABB but is
        // actually separated by an edge-cross axis.
        let a = Vec3::new(-4.0, 2.0, 0.0);
        let b = Vec3::new(4.0, 2.1, 0.0);
        let c = Vec3::new(-4.0, 2.05, 0.0);
        assert!(!triangle_aabb_overlap(a, b, c, &unit_box()));
    }

    #[test]
    fn touching_face_counts_as_overlap() {
        // Triangle lying exactly on the +z face plane, overlapping in xy.
        let a = Vec3::new(-0.5, -0.5, 1.0);
        let b = Vec3::new(0.5, -0.5, 1.0);
        let c = Vec3::new(0.0, 0.5, 1.0);
        assert!(triangle_aabb_overlap(a, b, c, &unit_box()));
    }

    #[test]
    fn degenerate_point_triangle_inside_overlaps() {
        let p = Vec3::new(0.2, -0.1, 0.3);
        assert!(triangle_aabb_overlap(p, p, p, &unit_box()));
    }

    #[test]
    fn degenerate_point_triangle_outside_is_separated() {
        let p = Vec3::new(5.0, 5.0, 5.0);
        assert!(!triangle_aabb_overlap(p, p, p, &unit_box()));
    }

    #[test]
    fn penetration_flat_triangle_above_centre_pushes_down() {
        // A horizontal triangle at z = 0.5 inside the unit box penetrates from
        // above the centre, so the box must move toward -Z by 0.5 to clear it.
        let a = Vec3::new(-2.0, -2.0, 0.5);
        let b = Vec3::new(2.0, -2.0, 0.5);
        let c = Vec3::new(0.0, 2.0, 0.5);
        let (normal, depth) = triangle_aabb_penetration(a, b, c, &unit_box()).unwrap();
        assert!((depth - 0.5).abs() < 1e-5, "depth = {depth}");
        assert!(normal.z < -0.99, "normal = {normal:?}");
        assert!(normal.x.abs() < 1e-5 && normal.y.abs() < 1e-5);
    }

    #[test]
    fn penetration_flat_triangle_below_centre_pushes_up() {
        // Triangle at z = -0.75: closer to the bottom face, so the shallow
        // escape is +Z by 0.25.
        let a = Vec3::new(-2.0, -2.0, -0.75);
        let b = Vec3::new(2.0, -2.0, -0.75);
        let c = Vec3::new(0.0, 2.0, -0.75);
        let (normal, depth) = triangle_aabb_penetration(a, b, c, &unit_box()).unwrap();
        assert!((depth - 0.25).abs() < 1e-5, "depth = {depth}");
        assert!(normal.z > 0.99, "normal = {normal:?}");
    }

    #[test]
    fn penetration_is_none_when_separated() {
        // Triangle well above the box: no overlap, no MTV.
        let a = Vec3::new(-2.0, -2.0, 2.0);
        let b = Vec3::new(2.0, -2.0, 2.0);
        let c = Vec3::new(0.0, 2.0, 2.0);
        assert!(triangle_aabb_penetration(a, b, c, &unit_box()).is_none());
        // And it agrees with the boolean overlap test.
        assert!(!triangle_aabb_overlap(a, b, c, &unit_box()));
    }

    #[test]
    fn penetration_agrees_with_overlap_boolean() {
        // A triangle slicing the interior overlaps and yields a positive depth.
        let a = Vec3::new(-2.0, 0.0, 0.0);
        let b = Vec3::new(2.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 2.0, 0.5);
        assert!(triangle_aabb_overlap(a, b, c, &unit_box()));
        let (normal, depth) = triangle_aabb_penetration(a, b, c, &unit_box()).unwrap();
        assert!(depth > 0.0);
        assert!((normal.length() - 1.0).abs() < 1e-5, "unit normal");
    }

}
