//! Triangle barycentric coordinates, attribute interpolation and inside tests
//! for the particle mesh-emission and decal-projection contracts (design §8.2).
//!
//! Several particle stages need to express a point as a blend of a triangle's
//! three corners: mesh emission samples a source surface and must interpolate
//! per-vertex attributes at the spawn position; decal and trail projection tests
//! whether a footprint sample lands inside a face; and any `GPU` rasterization
//! reference path needs perspective-correct interpolation to match the
//! hardware. This module owns the small, `CPU`-verifiable contract those stages
//! share: converting a point into barycentric weights, deciding whether it lies
//! inside the triangle, blending vertex attributes by those weights, and
//! applying the perspective divide that a screen-space `NDC` sample requires.
//!
//! # Strict scope
//! This module only computes barycentric weights and blends attributes by them.
//! It deliberately does not build meshes, project through a camera, or own the
//! attribute pool layout; it neither imports nor reconstructs those types.
//!
//! # No transcendental math
//! Every routine is pure ratio-of-cross-products arithmetic: the only
//! operations are `+`, `-`, `*` and one division by a signed-area / Gram
//! determinant. There is no `sqrt`, `sin`, `atan` or any transcendental call.
//! A degenerate (zero-area) triangle can never divide by near-zero: the area /
//! determinant is checked against [`CMP_EPS`] and reported as `None` instead.

/// Magnitude below which a signed area, a determinant, or a weight is treated
/// as zero. Two floats are "equal" when their difference is smaller than this,
/// which is the comparison rule used throughout instead of `==` on `f32`.
const CMP_EPS: f32 = 1.0e-6;

/// A triangle in the 2D plane, given by its three corners in `a`, `b`, `c`
/// order. The winding (clockwise vs. counter-clockwise) only flips the sign of
/// the shared area denominator, so the returned weights are winding-agnostic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Triangle2 {
    /// First corner.
    pub a: [f32; 2],
    /// Second corner.
    pub b: [f32; 2],
    /// Third corner.
    pub c: [f32; 2],
}

/// A triangle in 3D space, given by its three corners in `a`, `b`, `c` order.
/// Barycentric coordinates are solved in the triangle's own plane, so a query
/// point that is off the plane is projected onto it by the Gram-matrix method.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Triangle3 {
    /// First corner.
    pub a: [f32; 3],
    /// Second corner.
    pub b: [f32; 3],
    /// Third corner.
    pub c: [f32; 3],
}

/// Component-wise difference of two 2D points (`lhs - rhs`).
fn minus2(lhs: [f32; 2], rhs: [f32; 2]) -> [f32; 2] {
    [lhs[0] - rhs[0], lhs[1] - rhs[1]]
}

/// Component-wise difference of two 3D points (`lhs - rhs`).
fn minus3(lhs: [f32; 3], rhs: [f32; 3]) -> [f32; 3] {
    [lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]]
}

/// Dot product of two 3D vectors.
fn dot3(lhs: [f32; 3], rhs: [f32; 3]) -> f32 {
    lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]
}

/// Scalar 2D cross product (`z` component of the 3D cross), i.e. twice the
/// signed area of the triangle spanned by `lhs` and `rhs`.
fn cross2(lhs: [f32; 2], rhs: [f32; 2]) -> f32 {
    lhs[0] * rhs[1] - lhs[1] * rhs[0]
}

impl Triangle2 {
    /// Barycentric weights `[u, v, w]` of `p` with respect to `a`, `b`, `c`.
    ///
    /// Each weight is the ratio of the sub-triangle opposite that corner to the
    /// whole triangle, computed from scalar cross products (signed areas). The
    /// weights sum to `1` by construction and are negative outside the
    /// corresponding edge, so [`point_in_triangle`] classifies the point.
    /// Returns `None` when the triangle is degenerate (its area is within
    /// [`CMP_EPS`] of zero), which would otherwise divide by ~zero.
    #[must_use]
    pub fn barycentric2(&self, p: [f32; 2]) -> Option<[f32; 3]> {
        let denom = cross2(minus2(self.b, self.a), minus2(self.c, self.a));
        if denom.abs() < CMP_EPS {
            return None;
        }
        let inv = 1.0 / denom;
        let wa = cross2(minus2(self.c, self.b), minus2(p, self.b)) * inv;
        let wb = cross2(minus2(self.a, self.c), minus2(p, self.c)) * inv;
        let wc = cross2(minus2(self.b, self.a), minus2(p, self.a)) * inv;
        Some([wa, wb, wc])
    }
}

impl Triangle3 {
    /// Barycentric weights `[u, v, w]` of `p` with respect to `a`, `b`, `c`,
    /// solved in the triangle's plane by Christer Ericson's Gram-matrix method.
    ///
    /// Let `v0 = b - a`, `v1 = c - a`, `v2 = p - a`. The 2x2 Gram system in the
    /// `(v0, v1)` basis has determinant `denom = d00*d11 - d01*d01`; the two
    /// projected coordinates are `v` and `w`, and `u = 1 - v - w`. A point off
    /// the plane is implicitly projected onto it. Returns `None` when the
    /// triangle is degenerate (`denom` within [`CMP_EPS`] of zero).
    #[must_use]
    pub fn barycentric3(&self, p: [f32; 3]) -> Option<[f32; 3]> {
        let v0 = minus3(self.b, self.a);
        let v1 = minus3(self.c, self.a);
        let v2 = minus3(p, self.a);
        let d00 = dot3(v0, v0);
        let d01 = dot3(v0, v1);
        let d11 = dot3(v1, v1);
        let d20 = dot3(v2, v0);
        let d21 = dot3(v2, v1);
        let denom = d00 * d11 - d01 * d01;
        if denom.abs() < CMP_EPS {
            return None;
        }
        let inv = 1.0 / denom;
        let v = (d11 * d20 - d01 * d21) * inv;
        let w = (d00 * d21 - d01 * d20) * inv;
        let u = 1.0 - v - w;
        Some([u, v, w])
    }
}

/// Blends three scalar vertex attributes by the barycentric `weights`.
#[must_use]
pub fn interpolate_scalar(weights: [f32; 3], attrs: [f32; 3]) -> f32 {
    weights[0] * attrs[0] + weights[1] * attrs[1] + weights[2] * attrs[2]
}

/// Blends three `vec2` vertex attributes by the barycentric `weights`.
#[must_use]
pub fn interpolate_vec2(weights: [f32; 3], attrs: [[f32; 2]; 3]) -> [f32; 2] {
    [
        interpolate_scalar(weights, [attrs[0][0], attrs[1][0], attrs[2][0]]),
        interpolate_scalar(weights, [attrs[0][1], attrs[1][1], attrs[2][1]]),
    ]
}

/// Blends three `vec3` vertex attributes by the barycentric `weights`.
#[must_use]
pub fn interpolate_vec3(weights: [f32; 3], attrs: [[f32; 3]; 3]) -> [f32; 3] {
    [
        interpolate_scalar(weights, [attrs[0][0], attrs[1][0], attrs[2][0]]),
        interpolate_scalar(weights, [attrs[0][1], attrs[1][1], attrs[2][1]]),
        interpolate_scalar(weights, [attrs[0][2], attrs[1][2], attrs[2][2]]),
    ]
}

/// Blends three `vec4` vertex attributes by the barycentric `weights`.
#[must_use]
pub fn interpolate_vec4(weights: [f32; 3], attrs: [[f32; 4]; 3]) -> [f32; 4] {
    [
        interpolate_scalar(weights, [attrs[0][0], attrs[1][0], attrs[2][0]]),
        interpolate_scalar(weights, [attrs[0][1], attrs[1][1], attrs[2][1]]),
        interpolate_scalar(weights, [attrs[0][2], attrs[1][2], attrs[2][2]]),
        interpolate_scalar(weights, [attrs[0][3], attrs[1][3], attrs[2][3]]),
    ]
}

/// Returns `true` when the barycentric `weights` place the point inside (or on
/// the boundary of) the triangle, i.e. every weight is at least `-CMP_EPS`.
///
/// The small negative tolerance keeps points that land exactly on an edge
/// (where one weight rounds just below zero) classified as inside.
#[must_use]
pub fn point_in_triangle(weights: [f32; 3]) -> bool {
    weights[0] >= -CMP_EPS && weights[1] >= -CMP_EPS && weights[2] >= -CMP_EPS
}

/// Applies the perspective divide to affine barycentric `weights`, given each
/// vertex's `inv_w` (its `1/w` clip-space value).
///
/// Screen-space (`NDC`) barycentric weights interpolate linearly in device
/// space but not in the original attribute space; multiplying each weight by
/// the vertex `1/w` and renormalizing recovers the perspective-correct blend
/// the `GPU` rasterizer produces. When the reweighted sum is within
/// [`CMP_EPS`] of zero (all `inv_w` ~zero) the affine `weights` are returned
/// unchanged so the result is never `NaN`.
#[must_use]
pub fn perspective_correct(weights: [f32; 3], inv_w: [f32; 3]) -> [f32; 3] {
    let n = [
        weights[0] * inv_w[0],
        weights[1] * inv_w[1],
        weights[2] * inv_w[2],
    ];
    let sum = n[0] + n[1] + n[2];
    if sum.abs() < CMP_EPS {
        return weights;
    }
    let inv = 1.0 / sum;
    [n[0] * inv, n[1] * inv, n[2] * inv]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1.0e-4
    }

    fn close3(a: [f32; 3], b: [f32; 3]) -> bool {
        close(a[0], b[0]) && close(a[1], b[1]) && close(a[2], b[2])
    }

    fn close2(a: [f32; 2], b: [f32; 2]) -> bool {
        close(a[0], b[0]) && close(a[1], b[1])
    }

    fn tri2() -> Triangle2 {
        Triangle2 {
            a: [0.0, 0.0],
            b: [4.0, 0.0],
            c: [0.0, 3.0],
        }
    }

    fn tri3() -> Triangle3 {
        Triangle3 {
            a: [0.0, 0.0, 1.0],
            b: [4.0, 0.0, 1.0],
            c: [0.0, 3.0, 1.0],
        }
    }

    #[test]
    fn bary2_at_corner_a_is_unit_u() {
        let t = tri2();
        assert!(close3(t.barycentric2(t.a).unwrap(), [1.0, 0.0, 0.0]));
    }

    #[test]
    fn bary2_at_corner_b_is_unit_v() {
        let t = tri2();
        assert!(close3(t.barycentric2(t.b).unwrap(), [0.0, 1.0, 0.0]));
    }

    #[test]
    fn bary2_at_corner_c_is_unit_w() {
        let t = tri2();
        assert!(close3(t.barycentric2(t.c).unwrap(), [0.0, 0.0, 1.0]));
    }

    #[test]
    fn bary2_at_centroid_is_thirds() {
        let t = tri2();
        let centroid = [
            (t.a[0] + t.b[0] + t.c[0]) / 3.0,
            (t.a[1] + t.b[1] + t.c[1]) / 3.0,
        ];
        let third = 1.0 / 3.0;
        assert!(close3(
            t.barycentric2(centroid).unwrap(),
            [third, third, third]
        ));
    }

    #[test]
    fn bary2_weights_sum_to_one() {
        let t = tri2();
        let w = t.barycentric2([1.0, 1.0]).unwrap();
        assert!(close(w[0] + w[1] + w[2], 1.0));
    }

    #[test]
    fn bary2_is_winding_agnostic() {
        // Reversing the winding negates the shared area denominator but leaves
        // each corner's weight ratio identical.
        let cw = Triangle2 {
            a: [0.0, 0.0],
            b: [0.0, 3.0],
            c: [4.0, 0.0],
        };
        let w = cw.barycentric2([1.0, 1.0]).unwrap();
        assert!(close(w[0] + w[1] + w[2], 1.0));
        assert!(point_in_triangle(w));
    }

    #[test]
    fn bary2_degenerate_returns_none() {
        let line = Triangle2 {
            a: [0.0, 0.0],
            b: [1.0, 1.0],
            c: [2.0, 2.0],
        };
        assert!(line.barycentric2([0.5, 0.5]).is_none());
    }

    #[test]
    fn bary2_outside_point_has_negative_weight() {
        let t = tri2();
        let w = t.barycentric2([-1.0, -1.0]).unwrap();
        assert!(!point_in_triangle(w));
    }

    #[test]
    fn bary2_reconstructs_query_point() {
        let t = tri2();
        let p = [1.5, 0.75];
        let w = t.barycentric2(p).unwrap();
        assert!(close2(interpolate_vec2(w, [t.a, t.b, t.c]), p));
    }

    #[test]
    fn bary3_at_corners_are_unit_weights() {
        let t = tri3();
        assert!(close3(t.barycentric3(t.a).unwrap(), [1.0, 0.0, 0.0]));
        assert!(close3(t.barycentric3(t.b).unwrap(), [0.0, 1.0, 0.0]));
        assert!(close3(t.barycentric3(t.c).unwrap(), [0.0, 0.0, 1.0]));
    }

    #[test]
    fn bary3_at_centroid_is_thirds() {
        let t = tri3();
        let centroid = [
            (t.a[0] + t.b[0] + t.c[0]) / 3.0,
            (t.a[1] + t.b[1] + t.c[1]) / 3.0,
            (t.a[2] + t.b[2] + t.c[2]) / 3.0,
        ];
        let third = 1.0 / 3.0;
        assert!(close3(
            t.barycentric3(centroid).unwrap(),
            [third, third, third]
        ));
    }

    #[test]
    fn bary3_weights_sum_to_one() {
        let t = tri3();
        let w = t.barycentric3([1.0, 1.0, 1.0]).unwrap();
        assert!(close(w[0] + w[1] + w[2], 1.0));
    }

    #[test]
    fn bary3_off_plane_point_projects_onto_plane() {
        let t = tri3();
        // A point above the plane projects to the same weights as on it.
        let on = t.barycentric3([1.0, 1.0, 1.0]).unwrap();
        let off = t.barycentric3([1.0, 1.0, 5.0]).unwrap();
        assert!(close3(on, off));
    }

    #[test]
    fn bary3_degenerate_returns_none() {
        let line = Triangle3 {
            a: [0.0, 0.0, 0.0],
            b: [1.0, 1.0, 1.0],
            c: [2.0, 2.0, 2.0],
        };
        assert!(line.barycentric3([0.5, 0.5, 0.5]).is_none());
    }

    #[test]
    fn bary3_reconstructs_query_point() {
        let t = tri3();
        let p = [1.0, 0.5, 1.0];
        let w = t.barycentric3(p).unwrap();
        assert!(close3(interpolate_vec3(w, [t.a, t.b, t.c]), p));
    }

    #[test]
    fn interpolate_scalar_blends_by_weight() {
        assert!(close(
            interpolate_scalar([0.25, 0.25, 0.5], [4.0, 8.0, 2.0]),
            0.25 * 4.0 + 0.25 * 8.0 + 0.5 * 2.0
        ));
    }

    #[test]
    fn interpolate_vec2_is_componentwise() {
        let w = [0.5, 0.25, 0.25];
        let out = interpolate_vec2(w, [[1.0, 0.0], [0.0, 4.0], [4.0, 0.0]]);
        assert!(close2(out, [1.5, 1.0]));
    }

    #[test]
    fn interpolate_vec4_at_single_corner_selects_it() {
        let w = [0.0, 1.0, 0.0];
        let attrs = [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ];
        let out = interpolate_vec4(w, attrs);
        assert!(close(out[0], 5.0) && close(out[3], 8.0));
    }

    #[test]
    fn point_in_triangle_accepts_interior_and_edge() {
        assert!(point_in_triangle([0.5, 0.25, 0.25]));
        assert!(point_in_triangle([0.0, 0.5, 0.5]));
        assert!(point_in_triangle([1.0, 0.0, 0.0]));
    }

    #[test]
    fn point_in_triangle_rejects_exterior() {
        assert!(!point_in_triangle([-0.1, 0.6, 0.5]));
        assert!(!point_in_triangle([1.2, -0.1, -0.1]));
    }

    #[test]
    fn perspective_correct_is_identity_for_uniform_w() {
        let w = [0.2, 0.3, 0.5];
        assert!(close3(perspective_correct(w, [1.0, 1.0, 1.0]), w));
    }

    #[test]
    fn perspective_correct_renormalizes_and_sums_to_one() {
        let w = [0.25, 0.25, 0.5];
        let out = perspective_correct(w, [1.0, 2.0, 4.0]);
        assert!(close(out[0] + out[1] + out[2], 1.0));
        // The vertex with the largest 1/w gains the most weight.
        assert!(out[2] > w[2]);
    }

    #[test]
    fn perspective_correct_falls_back_when_all_inv_w_zero() {
        let w = [0.2, 0.3, 0.5];
        assert!(close3(perspective_correct(w, [0.0, 0.0, 0.0]), w));
    }
}
