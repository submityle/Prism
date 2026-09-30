//! Point-in-tetrahedron containment and 3D barycentric coordinates for the
//! particle spawn-in-volume and mesh-sampling contracts (design §8.2, §11).
//!
//! Several particle stages need to place a query point *inside a solid cell*
//! rather than on a surface: a spawn-in-tetrahedral-mesh emitter must reject a
//! seed that falls outside its host tet; a tetrahedral-field interpolator needs
//! the four blend weights that express a sample as a convex mix of the cell's
//! corners; and a collision-proxy pass wants a cheap "is this contact point
//! within the convex cell?" predicate. This module owns the small,
//! `CPU`-verifiable contract those stages share: the signed-volume (`orient3d`)
//! decomposition that turns a tetrahedron and a point into four barycentric
//! weights, the containment test built on their signs, and the guard that
//! reports a flat (coplanar) cell as having no defined interior.
//!
//! The method is the standard four-sub-tetrahedron decomposition. The whole
//! tet `(t0, t1, t2, t3)` has signed volume `V = orient3d(t0, t1, t2, t3)`.
//! Replacing corner `i` with the query point `p` yields the signed volume of
//! the sub-tet opposite that corner, and the barycentric weight is the ratio
//! `Vi / V`. The four weights sum to `1` and reconstruct `p` as `Σ bi * ti`;
//! the point lies inside (boundary included) exactly when every weight is at
//! least `-`[`BOUNDARY_EPS`].
//!
//! # Strict scope
//! This module only decides tetrahedron containment and returns the four
//! barycentric weights. It deliberately differs from its neighbours and neither
//! imports nor reconstructs their contracts:
//! - [`super::tetrahedron_volume`] computes a tetrahedron's signed volume,
//!   orientation, and mesh volume/centroid; it never expresses a point as a
//!   blend of the four corners.
//! - [`super::barycentric_coord`] solves *triangle* (three-corner) barycentrics
//!   in a plane; this module solves the *tetrahedron* (four-corner) case in 3D.
//! - [`super::point_in_polygon`] answers inside/outside for a 2D polygon by ray
//!   crossing or winding; this module works on a 3D solid via signed volumes.
//! - [`super::point_triangle_closest_3d`] projects a point to the nearest point
//!   on a triangle; this module never measures distance, only containment.
//!
//! # No transcendental math
//! Every quantity here is a ratio of scalar triple products, so the module is
//! pure `+`, `-`, `*`, `/`. There is no `sqrt`, `sin`, `cos`, `powf`, `floor`,
//! `round`, or any other transcendental or rounding call, and no `f32`
//! equality: a near-zero total volume is compared against [`DEGENERATE_EPS`]
//! (never `== 0.0`) and reported as [`None`].

/// Magnitude below which the tetrahedron's total signed volume is treated as
/// zero, marking the cell as degenerate (its four vertices are coplanar or
/// coincident) so no interior can be defined.
///
/// This is the comparison rule used instead of `==` on `f32`: a total whose
/// absolute value does not exceed this bound divides by ~zero and is rejected
/// as [`None`].
pub const DEGENERATE_EPS: f32 = 1.0e-6;

/// Slack applied to each barycentric weight when classifying containment.
///
/// A point exactly on a face, edge, or vertex has one or more weights equal to
/// zero; floating-point error can nudge such a weight slightly negative, so a
/// weight is accepted as non-negative when it is at least `-BOUNDARY_EPS`. This
/// makes the boundary inclusive without leaking points that are clearly
/// outside.
pub const BOUNDARY_EPS: f32 = 1.0e-5;

/// Component-wise difference of two 3D points (`lhs - rhs`).
///
/// A free function (rather than an inherent `sub`) keeps the point type a plain
/// `[f32; 3]` and sidesteps the operator-trait lint.
#[must_use]
fn v_sub(lhs: [f32; 3], rhs: [f32; 3]) -> [f32; 3] {
    [lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]]
}

/// Cross product of two 3D vectors (`lhs × rhs`).
#[must_use]
fn v_cross(lhs: [f32; 3], rhs: [f32; 3]) -> [f32; 3] {
    [
        lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[0] * rhs[1] - lhs[1] * rhs[0],
    ]
}

/// Dot product of two 3D vectors.
#[must_use]
fn v_dot(lhs: [f32; 3], rhs: [f32; 3]) -> f32 {
    lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]
}

/// Six times the signed volume of the tetrahedron `(a, b, c, d)`, i.e. the
/// scalar triple product `dot(b - a, cross(c - a, d - a))`.
///
/// This is the orientation determinant whose sign classifies which side of the
/// oriented plane `(a, b, c)` the point `d` lies on. Barycentric weights are
/// ratios of this quantity, so the common factor of six cancels and never needs
/// to be applied.
#[must_use]
pub fn orient3d(a: [f32; 3], b: [f32; 3], c: [f32; 3], d: [f32; 3]) -> f32 {
    v_dot(v_sub(b, a), v_cross(v_sub(c, a), v_sub(d, a)))
}

/// Barycentric weights `[b0, b1, b2, b3]` of `p` with respect to the
/// tetrahedron `tet = [t0, t1, t2, t3]`.
///
/// Each `bi` is the signed volume of the sub-tetrahedron obtained by replacing
/// corner `i` with `p`, divided by the whole tet's signed volume. The weights
/// sum to `1` by construction and satisfy `p == Σ bi * ti`; `bi` is negative
/// exactly when `p` lies on the far side of the face opposite corner `i`.
///
/// Returns [`None`] when the tetrahedron is degenerate: its total signed volume
/// is within [`DEGENERATE_EPS`] of zero (four coplanar or coincident vertices),
/// so the ratios would divide by ~zero and no interior is defined.
#[must_use]
pub fn barycentric_in_tetrahedron(tet: &[[f32; 3]; 4], p: [f32; 3]) -> Option<[f32; 4]> {
    let t0 = tet[0];
    let t1 = tet[1];
    let t2 = tet[2];
    let t3 = tet[3];
    let total = orient3d(t0, t1, t2, t3);
    if total.abs() < DEGENERATE_EPS {
        return None;
    }
    let inv = 1.0 / total;
    let b0 = orient3d(p, t1, t2, t3) * inv;
    let b1 = orient3d(t0, p, t2, t3) * inv;
    let b2 = orient3d(t0, t1, p, t3) * inv;
    let b3 = orient3d(t0, t1, t2, p) * inv;
    Some([b0, b1, b2, b3])
}

/// Returns `true` when `p` lies inside the tetrahedron `tet` (boundary
/// included).
///
/// Containment holds when every barycentric weight is at least
/// `-`[`BOUNDARY_EPS`], which admits points on a face, edge, or vertex while
/// rejecting points clearly outside. A degenerate tetrahedron (no defined
/// interior) always returns `false`.
#[must_use]
pub fn point_in_tetrahedron(tet: &[[f32; 3]; 4], p: [f32; 3]) -> bool {
    match barycentric_in_tetrahedron(tet, p) {
        Some(bary) => {
            bary[0] >= -BOUNDARY_EPS
                && bary[1] >= -BOUNDARY_EPS
                && bary[2] >= -BOUNDARY_EPS
                && bary[3] >= -BOUNDARY_EPS
        }
        None => false,
    }
}

/// Reconstructs the point `Σ bi * ti` from barycentric weights and the tet's
/// four corners.
///
/// This is the inverse of [`barycentric_in_tetrahedron`]: feeding the returned
/// weights back through this affine blend recovers the original query point (up
/// to floating-point error), which the tests exercise as a round-trip check.
#[must_use]
pub fn reconstruct_from_barycentric(tet: &[[f32; 3]; 4], bary: [f32; 4]) -> [f32; 3] {
    let mut out = [0.0f32; 3];
    for (weight, corner) in bary.iter().zip(tet.iter()) {
        out[0] += weight * corner[0];
        out[1] += weight * corner[1];
        out[2] += weight * corner[2];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The canonical unit corner tetrahedron: origin plus the three axis tips.
    /// Its total `orient3d` is `+1`, so barycentric weights of `(x, y, z)` are
    /// exactly `[1 - x - y - z, x, y, z]`.
    fn unit_tet() -> [[f32; 3]; 4] {
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    }

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1.0e-4
    }

    fn close_arr4(a: [f32; 4], b: [f32; 4]) -> bool {
        close(a[0], b[0]) && close(a[1], b[1]) && close(a[2], b[2]) && close(a[3], b[3])
    }

    fn close_vec(a: [f32; 3], b: [f32; 3]) -> bool {
        close(a[0], b[0]) && close(a[1], b[1]) && close(a[2], b[2])
    }

    #[test]
    fn centroid_has_equal_quarter_weights() {
        let tet = unit_tet();
        let c = [0.25, 0.25, 0.25];
        let bary = barycentric_in_tetrahedron(&tet, c).unwrap();
        assert!(close_arr4(bary, [0.25, 0.25, 0.25, 0.25]));
    }

    #[test]
    fn centroid_is_inside() {
        let tet = unit_tet();
        assert!(point_in_tetrahedron(&tet, [0.25, 0.25, 0.25]));
    }

    #[test]
    fn vertex0_weight_is_e0() {
        let tet = unit_tet();
        let bary = barycentric_in_tetrahedron(&tet, tet[0]).unwrap();
        assert!(close_arr4(bary, [1.0, 0.0, 0.0, 0.0]));
    }

    #[test]
    fn vertex1_weight_is_e1() {
        let tet = unit_tet();
        let bary = barycentric_in_tetrahedron(&tet, tet[1]).unwrap();
        assert!(close_arr4(bary, [0.0, 1.0, 0.0, 0.0]));
    }

    #[test]
    fn vertex2_weight_is_e2() {
        let tet = unit_tet();
        let bary = barycentric_in_tetrahedron(&tet, tet[2]).unwrap();
        assert!(close_arr4(bary, [0.0, 0.0, 1.0, 0.0]));
    }

    #[test]
    fn vertex3_weight_is_e3() {
        let tet = unit_tet();
        let bary = barycentric_in_tetrahedron(&tet, tet[3]).unwrap();
        assert!(close_arr4(bary, [0.0, 0.0, 0.0, 1.0]));
    }

    #[test]
    fn every_vertex_is_inside_boundary() {
        let tet = unit_tet();
        for v in tet.iter() {
            assert!(point_in_tetrahedron(&tet, *v));
        }
    }

    #[test]
    fn point_on_face_has_zero_opposite_weight() {
        // Point on the face opposite corner 0 (the plane x + y + z = 1).
        let tet = unit_tet();
        let p = [0.5, 0.25, 0.25];
        let bary = barycentric_in_tetrahedron(&tet, p).unwrap();
        assert!(close(bary[0], 0.0));
        assert!(bary[1] > 0.0 && bary[2] > 0.0 && bary[3] > 0.0);
        assert!(point_in_tetrahedron(&tet, p));
    }

    #[test]
    fn point_on_edge_has_two_zero_weights() {
        // Midpoint of the edge t1--t2 lies on two faces at once.
        let tet = unit_tet();
        let p = [0.5, 0.5, 0.0];
        let bary = barycentric_in_tetrahedron(&tet, p).unwrap();
        assert!(close(bary[0], 0.0));
        assert!(close(bary[3], 0.0));
        assert!(close(bary[1], 0.5));
        assert!(close(bary[2], 0.5));
        assert!(point_in_tetrahedron(&tet, p));
    }

    #[test]
    fn outside_point_has_negative_weight() {
        let tet = unit_tet();
        // Beyond the slanted face: x + y + z = 2 > 1, so b0 < 0.
        let p = [0.7, 0.7, 0.6];
        let bary = barycentric_in_tetrahedron(&tet, p).unwrap();
        assert!(bary[0] < 0.0);
    }

    #[test]
    fn outside_point_is_not_inside() {
        let tet = unit_tet();
        assert!(!point_in_tetrahedron(&tet, [0.7, 0.7, 0.6]));
    }

    #[test]
    fn far_negative_octant_point_is_outside() {
        let tet = unit_tet();
        assert!(!point_in_tetrahedron(&tet, [-1.0, -1.0, -1.0]));
    }

    #[test]
    fn known_exact_barycentric() {
        let tet = unit_tet();
        let p = [0.1, 0.2, 0.3];
        let bary = barycentric_in_tetrahedron(&tet, p).unwrap();
        // For the unit tet, bary == [1 - x - y - z, x, y, z].
        assert!(close_arr4(bary, [0.4, 0.1, 0.2, 0.3]));
    }

    #[test]
    fn weights_sum_to_one_interior() {
        let tet = unit_tet();
        let bary = barycentric_in_tetrahedron(&tet, [0.15, 0.35, 0.2]).unwrap();
        let sum = bary[0] + bary[1] + bary[2] + bary[3];
        assert!(close(sum, 1.0));
    }

    #[test]
    fn weights_sum_to_one_exterior() {
        // The partition-of-unity identity holds even for points outside.
        let tet = unit_tet();
        let bary = barycentric_in_tetrahedron(&tet, [2.0, -1.0, 0.5]).unwrap();
        let sum = bary[0] + bary[1] + bary[2] + bary[3];
        assert!(close(sum, 1.0));
    }

    #[test]
    fn reconstructs_interior_point() {
        let tet = unit_tet();
        let p = [0.2, 0.3, 0.1];
        let bary = barycentric_in_tetrahedron(&tet, p).unwrap();
        let back = reconstruct_from_barycentric(&tet, bary);
        assert!(close_vec(back, p));
    }

    #[test]
    fn reconstructs_exterior_point() {
        let tet = unit_tet();
        let p = [1.5, -0.4, 0.9];
        let bary = barycentric_in_tetrahedron(&tet, p).unwrap();
        let back = reconstruct_from_barycentric(&tet, bary);
        assert!(close_vec(back, p));
    }

    #[test]
    fn reconstructs_origin_point() {
        // Origin coincides with corner 0, so it rebuilds exactly.
        let tet = unit_tet();
        let p = [0.0, 0.0, 0.0];
        let bary = barycentric_in_tetrahedron(&tet, p).unwrap();
        let back = reconstruct_from_barycentric(&tet, bary);
        assert!(close_vec(back, p));
    }

    #[test]
    fn degenerate_coplanar_returns_none() {
        // All four vertices on the z = 0 plane: zero volume.
        let flat = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        assert!(barycentric_in_tetrahedron(&flat, [0.25, 0.25, 0.0]).is_none());
    }

    #[test]
    fn degenerate_coincident_returns_none() {
        // Three coincident vertices collapse the volume to zero.
        let dup = [
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [1.0, 0.0, 0.0],
        ];
        assert!(barycentric_in_tetrahedron(&dup, [0.5, 0.5, 0.5]).is_none());
    }

    #[test]
    fn degenerate_is_never_inside() {
        let flat = [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [2.0, 2.0, 0.0],
        ];
        assert!(!point_in_tetrahedron(&flat, [0.5, 0.5, 0.0]));
    }

    #[test]
    fn negative_winding_tetra_bary_matches() {
        // Swap corners 1 and 2 to flip the orientation (total volume negative).
        let tet = [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let p = [0.1, 0.2, 0.3];
        let bary = barycentric_in_tetrahedron(&tet, p).unwrap();
        // Weights follow the swapped corner order: b1 <-> b2.
        assert!(close_arr4(bary, [0.4, 0.2, 0.1, 0.3]));
        let sum = bary[0] + bary[1] + bary[2] + bary[3];
        assert!(close(sum, 1.0));
    }

    #[test]
    fn negative_winding_tetra_inside() {
        let tet = [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        assert!(point_in_tetrahedron(&tet, [0.2, 0.2, 0.2]));
        assert!(!point_in_tetrahedron(&tet, [0.6, 0.6, 0.6]));
    }

    #[test]
    fn negative_winding_reconstructs_point() {
        let tet = [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let p = [0.3, 0.1, 0.25];
        let bary = barycentric_in_tetrahedron(&tet, p).unwrap();
        let back = reconstruct_from_barycentric(&tet, bary);
        assert!(close_vec(back, p));
    }

    #[test]
    fn just_outside_a_face_is_outside() {
        let tet = unit_tet();
        // Slightly past the slanted face plane x + y + z = 1.
        let p = [0.34, 0.34, 0.34];
        assert!(!point_in_tetrahedron(&tet, p));
    }

    #[test]
    fn just_inside_a_face_is_inside() {
        let tet = unit_tet();
        let p = [0.33, 0.33, 0.33];
        assert!(point_in_tetrahedron(&tet, p));
    }

    #[test]
    fn deep_interior_all_weights_positive() {
        let tet = unit_tet();
        let bary = barycentric_in_tetrahedron(&tet, [0.2, 0.2, 0.2]).unwrap();
        assert!(bary[0] > 0.0 && bary[1] > 0.0 && bary[2] > 0.0 && bary[3] > 0.0);
    }

    #[test]
    fn translation_invariance_of_weights() {
        let tet = unit_tet();
        let shift = [10.0, -5.0, 3.0];
        let shifted = [
            v_add(tet[0], shift),
            v_add(tet[1], shift),
            v_add(tet[2], shift),
            v_add(tet[3], shift),
        ];
        let p = [0.2, 0.3, 0.1];
        let a = barycentric_in_tetrahedron(&tet, p).unwrap();
        let b = barycentric_in_tetrahedron(&shifted, v_add(p, shift)).unwrap();
        assert!(close_arr4(a, b));
    }

    #[test]
    fn uniform_scale_preserves_weights() {
        let tet = unit_tet();
        let s = 4.0f32;
        let scaled = [
            v_scale(tet[0], s),
            v_scale(tet[1], s),
            v_scale(tet[2], s),
            v_scale(tet[3], s),
        ];
        let p = [0.2, 0.3, 0.1];
        let a = barycentric_in_tetrahedron(&tet, p).unwrap();
        let b = barycentric_in_tetrahedron(&scaled, v_scale(p, s)).unwrap();
        assert!(close_arr4(a, b));
    }

    #[test]
    fn general_tetra_reconstructs_and_sums() {
        // A skewed, non-axis-aligned tetrahedron.
        let tet = [
            [1.0, 2.0, -1.0],
            [4.0, 0.0, 1.0],
            [-1.0, 3.0, 2.0],
            [2.0, -2.0, 5.0],
        ];
        let p = [1.5, 0.5, 1.75];
        let bary = barycentric_in_tetrahedron(&tet, p).unwrap();
        let sum = bary[0] + bary[1] + bary[2] + bary[3];
        assert!(close(sum, 1.0));
        let back = reconstruct_from_barycentric(&tet, bary);
        assert!(close_vec(back, p));
    }

    #[test]
    fn determinism_bit_for_bit() {
        let tet = unit_tet();
        let p = [0.11, 0.22, 0.33];
        let first = barycentric_in_tetrahedron(&tet, p).unwrap();
        let second = barycentric_in_tetrahedron(&tet, p).unwrap();
        for i in 0..4 {
            assert_eq!(first[i].to_bits(), second[i].to_bits());
        }
    }

    #[test]
    fn orient3d_sign_flips_on_swap() {
        let tet = unit_tet();
        let pos = orient3d(tet[0], tet[1], tet[2], tet[3]);
        let neg = orient3d(tet[0], tet[2], tet[1], tet[3]);
        assert!(pos > 0.0);
        assert!(neg < 0.0);
        assert!(close(pos, -neg));
    }

    // --- test-only vector helpers ---

    fn v_add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
    }

    fn v_scale(a: [f32; 3], s: f32) -> [f32; 3] {
        [a[0] * s, a[1] * s, a[2] * s]
    }
}
