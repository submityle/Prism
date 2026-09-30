//! Expanding-Polytope-Algorithm (`EPA`) penetration depth and contact normal
//! for two convex point sets in 3D (design §7, §10 collision resolution).
//!
//! When two convex particle proxies overlap, the simulation needs more than a
//! yes/no answer: to resolve the contact it needs the *penetration depth* (how
//! far one body has to move to stop overlapping) and the *contact normal* (the
//! direction of that minimal move). This module owns the small,
//! `CPU`-verifiable contract that produces both from the Minkowski difference
//! `A - B` of two convex hulls given as vertex clouds.
//!
//! The pipeline is the classic two-stage one. First a lightweight `GJK`
//! (Gilbert-Johnson-Keerthi) evolves a simplex over the Minkowski difference
//! until it encloses the origin, which both proves the two bodies overlap and
//! hands `EPA` a non-degenerate seed tetrahedron. Then `EPA` grows that
//! tetrahedron into a polytope that hugs the Minkowski boundary: at every step
//! it takes the polytope face closest to the origin, queries a fresh support
//! point along that face's outward normal, and either declares convergence
//! (the support point does not advance past the face) or carves the polytope
//! open along the *horizon* — the loop of edges bordering every face the new
//! point can see — and stitches the new point onto each horizon edge. The
//! closest face at convergence gives the penetration depth (its distance from
//! the origin) and the contact normal (its outward unit normal).
//!
//! # `//!` scope boundary
//! This module is deliberately narrow and does *not* overlap its siblings:
//! * [`super::gjk_3d`] answers only the boolean question "do these two convex
//!   sets intersect?"; it never measures how deep the overlap is. This module
//!   consumes that same overlap test as a seed stage and then quantifies the
//!   penetration `gjk_3d` leaves unmeasured.
//! * [`super::sat_collision_2d`] computes a 2D separating-axis minimum
//!   translation vector; it lives entirely in the plane and enumerates edge
//!   normals, not a 3D expanding polytope.
//! * [`super::obb_obb_sat_3d`] resolves oriented-box overlap by testing the 15
//!   fixed separating axes of two boxes; it is specialised to boxes and never
//!   builds a polytope over an arbitrary Minkowski difference.
//! * [`super::minkowski_sum_2d`] constructs the 2D Minkowski *sum* polygon as an
//!   explicit boundary; this module never materialises the whole Minkowski
//!   set, only the local support points `EPA` needs.
//!
//! # No transcendental math
//! Every routine is pure `+`, `-`, `*` vector arithmetic plus comparison. The
//! only irrational operation is the [`f32::sqrt`] used to normalise a face
//! normal; there is no `sin`, `cos`, `atan`, `exp`, `ln`, `powf`, or any other
//! transcendental call. No `f32` value is ever compared with `==` or `!=`:
//! near-zero magnitudes are tested against an epsilon, and `NaN`-free inputs are
//! assumed from the convex-hull producers upstream.

use alloc::vec::Vec;

/// Magnitude below which a squared length or a signed distance is treated as
/// zero. Used instead of `==` on `f32`: a face whose normal is shorter than
/// this is degenerate and skipped, and a direction shorter than this is
/// considered to have collapsed onto the origin.
pub const EPS: f32 = 1.0e-6;

/// Growth threshold for `EPA` convergence. When a fresh support point advances
/// less than this past the current closest face, the polytope is judged to have
/// reached the Minkowski boundary and iteration stops.
pub const EPA_TOLERANCE: f32 = 1.0e-4;

/// Visibility slack. A polytope face is "seen" by a new support point only when
/// the point lies strictly more than this beyond the face plane, so points that
/// merely graze a face do not trigger a rebuild.
pub const VISIBILITY_EPS: f32 = 1.0e-6;

/// Hard cap on `EPA` refinement iterations, so a pathological input can never
/// spin forever; the best face found so far is returned when the cap is hit.
pub const EPA_MAX_ITERS: usize = 64;

/// Hard cap on the polytope face count, a second guard against runaway growth.
pub const FACE_CAP: usize = 128;

/// Hard cap on `GJK` simplex-evolution iterations before the seed search gives
/// up and reports no overlap.
pub const GJK_MAX_ITERS: usize = 64;

/// The resolved contact: the outward unit `normal` on the Minkowski boundary and
/// the `depth` (penetration distance) along it. Translating body `B` by
/// `normal * depth` moves the two bodies to just-touching.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Penetration {
    /// Unit contact normal pointing from body `B` toward body `A`.
    pub normal: [f32; 3],
    /// Penetration depth: distance from the origin to the closest Minkowski
    /// face along `normal`. Never negative.
    pub depth: f32,
}

/// Sum `a + b` of two 3D vectors.
#[must_use]
fn v_add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Difference `a - b` of two 3D vectors.
#[must_use]
fn v_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Negation `-a` of a 3D vector.
#[must_use]
fn v_neg(a: [f32; 3]) -> [f32; 3] {
    [-a[0], -a[1], -a[2]]
}

/// Scale `a * s` of a 3D vector by a scalar.
#[must_use]
fn v_scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Dot product `a . b` of two 3D vectors.
#[must_use]
fn v_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product `a x b` of two 3D vectors.
#[must_use]
fn v_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Squared Euclidean length of a 3D vector (no `sqrt`).
#[must_use]
fn v_len_sq(a: [f32; 3]) -> f32 {
    v_dot(a, a)
}

/// Unit vector along `a`, or `None` when `a` is shorter than [`EPS`] and has no
/// well-defined direction. This is the module's only use of [`f32::sqrt`].
#[must_use]
fn v_normalize(a: [f32; 3]) -> Option<[f32; 3]> {
    let len = v_len_sq(a).sqrt();
    if len <= EPS {
        None
    } else {
        let inv = 1.0 / len;
        Some(v_scale(a, inv))
    }
}

/// A vector perpendicular to `a`, chosen from a coordinate axis that is least
/// aligned with `a`. Used only to restart a `GJK` search direction that has
/// collapsed along the current edge.
#[must_use]
fn any_perpendicular(a: [f32; 3]) -> [f32; 3] {
    let ax = a[0].abs();
    let ay = a[1].abs();
    let az = a[2].abs();
    let axis = if ax <= ay && ax <= az {
        [1.0, 0.0, 0.0]
    } else if ay <= az {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    v_cross(a, axis)
}

/// Farthest vertex of a convex point cloud along `dir`.
///
/// This is the per-body support function: for a convex set the extreme vertex
/// in a direction is a supporting point of its hull. Assumes `points` is
/// non-empty; callers guard the empty case before reaching here.
#[must_use]
pub fn support(points: &[[f32; 3]], dir: [f32; 3]) -> [f32; 3] {
    let mut best = points[0];
    let mut best_dot = v_dot(best, dir);
    for &p in &points[1..] {
        let d = v_dot(p, dir);
        if d > best_dot {
            best_dot = d;
            best = p;
        }
    }
    best
}

/// Support point of the Minkowski difference `A - B` along `dir`.
///
/// The farthest point of `A - B` in a direction is `support(A, dir) -
/// support(B, -dir)`, which is the single primitive both `GJK` and `EPA` build
/// on.
#[must_use]
pub fn minkowski_support(a: &[[f32; 3]], b: &[[f32; 3]], dir: [f32; 3]) -> [f32; 3] {
    v_sub(support(a, dir), support(b, v_neg(dir)))
}

/// A polytope face: three vertex indices wound so its outward `normal` points
/// away from the polytope interior, with `dist` the signed distance from the
/// origin to the face plane along that normal.
#[derive(Clone, Copy)]
struct Face {
    v: [usize; 3],
    normal: [f32; 3],
    dist: f32,
}

/// Builds a face on `verts[i], verts[j], verts[k]` whose unit normal points away
/// from the interior reference `interior`, returning `None` when the triangle is
/// degenerate (zero area).
///
/// Because the seed tetrahedron centroid stays strictly inside the growing
/// polytope, orienting every face away from that fixed interior point keeps all
/// windings consistent without a separate repair pass, and makes `dist`
/// non-negative for a polytope that encloses the origin.
#[must_use]
fn build_face(
    interior: [f32; 3],
    verts: &[[f32; 3]],
    i: usize,
    j: usize,
    k: usize,
) -> Option<Face> {
    let vi = verts[i];
    let vj = verts[j];
    let vk = verts[k];
    let raw = v_cross(v_sub(vj, vi), v_sub(vk, vi));
    let unit = v_normalize(raw)?;
    let (v, normal) = if v_dot(unit, v_sub(interior, vi)) > 0.0 {
        // The normal points toward the interior; flip both winding and normal
        // so it points outward.
        ([i, k, j], v_neg(unit))
    } else {
        ([i, j, k], unit)
    };
    let dist = v_dot(normal, vi);
    Some(Face { v, normal, dist })
}

/// Evolves a `GJK` simplex one step toward the origin, returning `true` only
/// when the simplex has become a tetrahedron that encloses the origin.
fn do_simplex(simplex: &mut Vec<[f32; 3]>, dir: &mut [f32; 3]) -> bool {
    match simplex.len() {
        2 => {
            line_case(simplex, dir);
            false
        }
        3 => {
            triangle_case(simplex, dir);
            false
        }
        4 => tetra_case(simplex, dir),
        _ => false,
    }
}

/// `GJK` line case: reduce a 2-point simplex toward the origin.
fn line_case(simplex: &mut Vec<[f32; 3]>, dir: &mut [f32; 3]) {
    let a = simplex[1];
    let b = simplex[0];
    let ab = v_sub(b, a);
    let ao = v_neg(a);
    if v_dot(ab, ao) > 0.0 {
        let perp = v_cross(v_cross(ab, ao), ab);
        *dir = if v_len_sq(perp) <= EPS {
            any_perpendicular(ab)
        } else {
            perp
        };
    } else {
        *simplex = Vec::from([a]);
        *dir = ao;
    }
}

/// `GJK` triangle case: reduce a 3-point simplex toward the origin.
fn triangle_case(simplex: &mut Vec<[f32; 3]>, dir: &mut [f32; 3]) {
    let a = simplex[2];
    let b = simplex[1];
    let c = simplex[0];
    let ao = v_neg(a);
    let ab = v_sub(b, a);
    let ac = v_sub(c, a);
    let abc = v_cross(ab, ac);
    if v_dot(v_cross(abc, ac), ao) > 0.0 {
        if v_dot(ac, ao) > 0.0 {
            *simplex = Vec::from([c, a]);
            *dir = v_cross(v_cross(ac, ao), ac);
        } else {
            triangle_edge_ab(simplex, dir, a, b, ab, ao);
        }
    } else if v_dot(v_cross(ab, abc), ao) > 0.0 {
        triangle_edge_ab(simplex, dir, a, b, ab, ao);
    } else if v_dot(abc, ao) > 0.0 {
        *dir = abc;
    } else {
        *simplex = Vec::from([b, c, a]);
        *dir = v_neg(abc);
    }
}

/// Shared edge-`ab` reduction used by two branches of the triangle case.
fn triangle_edge_ab(
    simplex: &mut Vec<[f32; 3]>,
    dir: &mut [f32; 3],
    a: [f32; 3],
    b: [f32; 3],
    ab: [f32; 3],
    ao: [f32; 3],
) {
    if v_dot(ab, ao) > 0.0 {
        *simplex = Vec::from([b, a]);
        *dir = v_cross(v_cross(ab, ao), ab);
    } else {
        *simplex = Vec::from([a]);
        *dir = ao;
    }
}

/// `GJK` tetrahedron case: either confirm the origin is enclosed or drop the one
/// face it lies outside and recurse into the triangle case.
fn tetra_case(simplex: &mut Vec<[f32; 3]>, dir: &mut [f32; 3]) -> bool {
    let a = simplex[3];
    let b = simplex[2];
    let c = simplex[1];
    let d = simplex[0];
    let ao = v_neg(a);
    let ab = v_sub(b, a);
    let ac = v_sub(c, a);
    let ad = v_sub(d, a);
    let mut abc = v_cross(ab, ac);
    let mut acd = v_cross(ac, ad);
    let mut adb = v_cross(ad, ab);
    // Orient each face normal to point away from the opposite vertex.
    if v_dot(abc, ad) > 0.0 {
        abc = v_neg(abc);
    }
    if v_dot(acd, ab) > 0.0 {
        acd = v_neg(acd);
    }
    if v_dot(adb, ac) > 0.0 {
        adb = v_neg(adb);
    }
    if v_dot(abc, ao) > 0.0 {
        *simplex = Vec::from([c, b, a]);
        triangle_case(simplex, dir);
        false
    } else if v_dot(acd, ao) > 0.0 {
        *simplex = Vec::from([d, c, a]);
        triangle_case(simplex, dir);
        false
    } else if v_dot(adb, ao) > 0.0 {
        *simplex = Vec::from([b, d, a]);
        triangle_case(simplex, dir);
        false
    } else {
        true
    }
}

/// Runs `GJK` on the Minkowski difference of two convex clouds, returning the
/// four vertices of a seed tetrahedron that encloses the origin when the bodies
/// overlap, or `None` when they are disjoint.
#[must_use]
pub fn gjk_tetrahedron(a: &[[f32; 3]], b: &[[f32; 3]]) -> Option<[[f32; 3]; 4]> {
    if a.is_empty() || b.is_empty() {
        return None;
    }
    let mut dir = [1.0, 0.0, 0.0];
    let first = minkowski_support(a, b, dir);
    let mut simplex: Vec<[f32; 3]> = Vec::from([first]);
    dir = v_neg(first);
    for _ in 0..GJK_MAX_ITERS {
        if v_len_sq(dir) <= EPS {
            dir = [1.0, 0.0, 0.0];
        }
        let p = minkowski_support(a, b, dir);
        if v_dot(p, dir) < 0.0 {
            return None;
        }
        simplex.push(p);
        if do_simplex(&mut simplex, &mut dir) {
            return Some([simplex[0], simplex[1], simplex[2], simplex[3]]);
        }
    }
    None
}

/// Runs `EPA` from a caller-supplied seed tetrahedron of Minkowski-difference
/// vertices, expanding it against the support function of `a` and `b` until the
/// closest face converges to the Minkowski boundary.
///
/// The four `tetra` points must enclose the origin (as produced by
/// [`gjk_tetrahedron`]); the same `a` and `b` clouds must be passed so `EPA` can
/// query fresh support points while it refines. Returns `None` only when no
/// non-degenerate face can be built from the seed.
#[must_use]
pub fn epa_from_tetrahedron(
    a: &[[f32; 3]],
    b: &[[f32; 3]],
    tetra: [[f32; 3]; 4],
) -> Option<Penetration> {
    if a.is_empty() || b.is_empty() {
        return None;
    }
    let mut verts: Vec<[f32; 3]> = Vec::from(tetra);
    let centroid = v_scale(
        v_add(v_add(verts[0], verts[1]), v_add(verts[2], verts[3])),
        0.25,
    );
    let mut faces: Vec<Face> = Vec::new();
    for (i, j, k) in [(0usize, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)] {
        if let Some(face) = build_face(centroid, &verts, i, j, k) {
            faces.push(face);
        }
    }
    if faces.is_empty() {
        return None;
    }
    let mut best_normal = [0.0, 0.0, 0.0];
    let mut best_depth = f32::INFINITY;
    for _ in 0..EPA_MAX_ITERS {
        let mut idx = 0usize;
        let mut min_dist = f32::INFINITY;
        for (i, face) in faces.iter().enumerate() {
            if face.dist < min_dist {
                min_dist = face.dist;
                idx = i;
            }
        }
        let normal = faces[idx].normal;
        best_normal = normal;
        best_depth = min_dist;
        let p = minkowski_support(a, b, normal);
        let advanced = v_dot(normal, p) - min_dist;
        if advanced < EPA_TOLERANCE {
            return Some(Penetration {
                normal,
                depth: min_dist,
            });
        }
        let p_idx = verts.len();
        verts.push(p);
        let mut horizon: Vec<(usize, usize)> = Vec::new();
        let mut kept: Vec<Face> = Vec::new();
        for face in faces.drain(..) {
            let visible = v_dot(face.normal, p) - face.dist > VISIBILITY_EPS;
            if visible {
                for (x, y) in [
                    (face.v[0], face.v[1]),
                    (face.v[1], face.v[2]),
                    (face.v[2], face.v[0]),
                ] {
                    if let Some(pos) = horizon.iter().position(|&(hx, hy)| hx == y && hy == x) {
                        horizon.remove(pos);
                    } else {
                        horizon.push((x, y));
                    }
                }
            } else {
                kept.push(face);
            }
        }
        faces = kept;
        if horizon.is_empty() {
            break;
        }
        for (i, j) in horizon {
            if let Some(face) = build_face(centroid, &verts, i, j, p_idx) {
                faces.push(face);
            }
        }
        if faces.is_empty() || faces.len() > FACE_CAP {
            break;
        }
    }
    Some(Penetration {
        normal: best_normal,
        depth: best_depth,
    })
}

/// Penetration depth and contact normal of two overlapping convex clouds, or
/// `None` when they are disjoint.
///
/// Seeds `EPA` with a [`gjk_tetrahedron`] and then refines it with
/// [`epa_from_tetrahedron`]. The returned `normal` is a unit vector and `depth`
/// is non-negative; translating `b` by `normal * depth` separates the bodies.
#[must_use]
pub fn penetration(a: &[[f32; 3]], b: &[[f32; 3]]) -> Option<Penetration> {
    let tetra = gjk_tetrahedron(a, b)?;
    epa_from_tetrahedron(a, b, tetra)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// Absolute-difference comparison, the test-side stand-in for `f32` equality.
    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    /// Eight corners of an axis-aligned box centred at `c` with per-axis
    /// half-extent `h`.
    fn boxed(c: [f32; 3], h: f32) -> Vec<[f32; 3]> {
        let mut out: Vec<[f32; 3]> = Vec::new();
        for sx in [-1.0f32, 1.0] {
            for sy in [-1.0f32, 1.0] {
                for sz in [-1.0f32, 1.0] {
                    out.push([c[0] + sx * h, c[1] + sy * h, c[2] + sz * h]);
                }
            }
        }
        out
    }

    /// Six vertices of an axis-aligned octahedron of radius `r` centred at `c`.
    fn octa(c: [f32; 3], r: f32) -> Vec<[f32; 3]> {
        Vec::from([
            [c[0] + r, c[1], c[2]],
            [c[0] - r, c[1], c[2]],
            [c[0], c[1] + r, c[2]],
            [c[0], c[1] - r, c[2]],
            [c[0], c[1], c[2] + r],
            [c[0], c[1], c[2] - r],
        ])
    }

    /// Four vertices of a regular-ish tetrahedron centred at `c`, scaled by `s`.
    fn tetra_body(c: [f32; 3], s: f32) -> Vec<[f32; 3]> {
        Vec::from([
            [c[0] + s, c[1] + s, c[2] + s],
            [c[0] + s, c[1] - s, c[2] - s],
            [c[0] - s, c[1] + s, c[2] - s],
            [c[0] - s, c[1] - s, c[2] + s],
        ])
    }

    /// Shift every point by `t`.
    fn translate(points: &[[f32; 3]], t: [f32; 3]) -> Vec<[f32; 3]> {
        points.iter().map(|&p| v_add(p, t)).collect()
    }

    #[test]
    fn overlap_along_x_has_known_depth_and_normal() {
        let a = boxed([0.0, 0.0, 0.0], 1.0);
        let b = boxed([1.5, 0.0, 0.0], 1.0);
        let pen = penetration(&a, &b).expect("boxes overlap");
        assert!(approx(pen.depth, 0.5, 1.0e-2), "depth {}", pen.depth);
        assert!(pen.normal[0].abs() > 0.99, "normal {:?}", pen.normal);
        assert!(pen.normal[1].abs() < 1.0e-2);
        assert!(pen.normal[2].abs() < 1.0e-2);
    }

    #[test]
    fn overlap_along_y_axis() {
        let a = boxed([0.0, 0.0, 0.0], 1.0);
        let b = boxed([0.0, 1.25, 0.0], 1.0);
        let pen = penetration(&a, &b).expect("overlap");
        assert!(approx(pen.depth, 0.75, 1.0e-2), "depth {}", pen.depth);
        assert!(pen.normal[1].abs() > 0.99);
    }

    #[test]
    fn overlap_along_z_axis() {
        let a = boxed([0.0, 0.0, 0.0], 1.0);
        let b = boxed([0.0, 0.0, 1.75], 1.0);
        let pen = penetration(&a, &b).expect("overlap");
        assert!(approx(pen.depth, 0.25, 1.0e-2), "depth {}", pen.depth);
        assert!(pen.normal[2].abs() > 0.99);
    }

    #[test]
    fn shallow_penetration_small_depth() {
        let a = boxed([0.0, 0.0, 0.0], 1.0);
        let b = boxed([1.98, 0.0, 0.0], 1.0);
        let pen = penetration(&a, &b).expect("thin overlap");
        assert!(approx(pen.depth, 0.02, 1.0e-2), "depth {}", pen.depth);
        assert!(pen.depth > 0.0);
    }

    #[test]
    fn deep_penetration_large_depth() {
        let a = boxed([0.0, 0.0, 0.0], 1.0);
        let b = boxed([0.2, 0.0, 0.0], 1.0);
        let pen = penetration(&a, &b).expect("deep overlap");
        assert!(approx(pen.depth, 1.8, 1.0e-2), "depth {}", pen.depth);
    }

    #[test]
    fn depth_is_never_negative() {
        let a = boxed([0.0, 0.0, 0.0], 1.0);
        let b = boxed([0.7, 0.3, -0.1], 1.0);
        let pen = penetration(&a, &b).expect("overlap");
        assert!(pen.depth >= 0.0);
    }

    #[test]
    fn contact_normal_is_unit_length() {
        let a = boxed([0.0, 0.0, 0.0], 1.0);
        let b = boxed([1.1, 0.4, 0.0], 1.0);
        let pen = penetration(&a, &b).expect("overlap");
        let len_sq = v_len_sq(pen.normal);
        assert!(approx(len_sq, 1.0, 1.0e-3), "len_sq {len_sq}");
    }

    #[test]
    fn disjoint_boxes_return_none() {
        let a = boxed([0.0, 0.0, 0.0], 1.0);
        let b = boxed([3.0, 0.0, 0.0], 1.0);
        assert!(penetration(&a, &b).is_none());
    }

    #[test]
    fn disjoint_along_diagonal_returns_none() {
        let a = boxed([0.0, 0.0, 0.0], 1.0);
        let b = boxed([2.5, 2.5, 2.5], 1.0);
        assert!(penetration(&a, &b).is_none());
    }

    #[test]
    fn containment_depth_is_distance_to_nearest_face() {
        let big = boxed([0.0, 0.0, 0.0], 3.0);
        let small = boxed([0.0, 0.0, 0.0], 0.5);
        let pen = penetration(&big, &small).expect("small inside big");
        assert!(approx(pen.depth, 3.5, 2.0e-2), "depth {}", pen.depth);
        assert!(approx(v_len_sq(pen.normal), 1.0, 1.0e-3));
    }

    #[test]
    fn identical_boxes_depth_is_full_extent() {
        let a = boxed([0.0, 0.0, 0.0], 1.0);
        let b = boxed([0.0, 0.0, 0.0], 1.0);
        let pen = penetration(&a, &b).expect("coincident");
        assert!(approx(pen.depth, 2.0, 2.0e-2), "depth {}", pen.depth);
    }

    #[test]
    fn tangent_tiny_overlap_has_near_zero_depth() {
        let a = boxed([0.0, 0.0, 0.0], 1.0);
        let b = boxed([1.999, 0.0, 0.0], 1.0);
        let pen = penetration(&a, &b).expect("grazing overlap");
        assert!(pen.depth >= 0.0);
        assert!(pen.depth < 0.05, "depth {}", pen.depth);
    }

    #[test]
    fn just_separated_boxes_are_none() {
        let a = boxed([0.0, 0.0, 0.0], 1.0);
        // A hair past touching along x.
        let b = boxed([2.01, 0.0, 0.0], 1.0);
        assert!(penetration(&a, &b).is_none());
    }

    #[test]
    fn overlapping_tetrahedra_report_penetration() {
        let a = tetra_body([0.0, 0.0, 0.0], 1.0);
        let b = tetra_body([0.5, 0.0, 0.0], 1.0);
        let pen = penetration(&a, &b).expect("tetra overlap");
        assert!(pen.depth > 0.0);
        assert!(approx(v_len_sq(pen.normal), 1.0, 1.0e-3));
    }

    #[test]
    fn overlapping_octahedra_report_penetration() {
        let a = octa([0.0, 0.0, 0.0], 1.0);
        let b = octa([1.0, 0.0, 0.0], 1.0);
        let pen = penetration(&a, &b).expect("octa overlap");
        assert!(pen.depth > 0.0);
        assert!(approx(v_len_sq(pen.normal), 1.0, 1.0e-3));
    }

    #[test]
    fn gjk_seed_exists_for_overlap() {
        let a = boxed([0.0, 0.0, 0.0], 1.0);
        let b = boxed([1.0, 0.0, 0.0], 1.0);
        assert!(gjk_tetrahedron(&a, &b).is_some());
    }

    #[test]
    fn gjk_seed_absent_for_disjoint() {
        let a = boxed([0.0, 0.0, 0.0], 1.0);
        let b = boxed([5.0, 0.0, 0.0], 1.0);
        assert!(gjk_tetrahedron(&a, &b).is_none());
    }

    #[test]
    fn epa_from_seed_matches_penetration() {
        let a = boxed([0.0, 0.0, 0.0], 1.0);
        let b = boxed([1.3, 0.0, 0.0], 1.0);
        let seed = gjk_tetrahedron(&a, &b).expect("seed");
        let via_seed = epa_from_tetrahedron(&a, &b, seed).expect("epa");
        let via_full = penetration(&a, &b).expect("full");
        assert!(approx(via_seed.depth, via_full.depth, 1.0e-3));
    }

    #[test]
    fn translating_b_by_normal_times_depth_separates() {
        let a = boxed([0.0, 0.0, 0.0], 1.0);
        let b = boxed([1.5, 0.0, 0.0], 1.0);
        let pen = penetration(&a, &b).expect("overlap");
        // Push slightly past the exact separation distance.
        let shove = v_scale(pen.normal, pen.depth + 1.0e-2);
        let b2 = translate(&b, shove);
        assert!(
            penetration(&a, &b2).is_none(),
            "should separate after moving by depth"
        );
    }

    #[test]
    fn translating_b_by_less_than_depth_still_overlaps() {
        let a = boxed([0.0, 0.0, 0.0], 1.0);
        let b = boxed([1.5, 0.0, 0.0], 1.0);
        let pen = penetration(&a, &b).expect("overlap");
        let shove = v_scale(pen.normal, pen.depth * 0.5);
        let b2 = translate(&b, shove);
        let still = penetration(&a, &b2).expect("still overlapping");
        assert!(still.depth > 0.0);
        // Remaining overlap is roughly half the original depth.
        assert!(approx(still.depth, pen.depth * 0.5, 2.0e-2));
    }

    #[test]
    fn depth_normal_consistency_off_axis() {
        let a = boxed([0.0, 0.0, 0.0], 1.0);
        let b = boxed([1.4, 0.9, 0.0], 1.0);
        let pen = penetration(&a, &b).expect("overlap");
        let shove = v_scale(pen.normal, pen.depth + 1.0e-2);
        let b2 = translate(&b, shove);
        assert!(penetration(&a, &b2).is_none());
    }

    #[test]
    fn result_is_symmetric_in_depth_and_opposite_in_normal() {
        let a = boxed([0.0, 0.0, 0.0], 1.0);
        let b = boxed([1.2, 0.0, 0.0], 1.0);
        let ab = penetration(&a, &b).expect("a,b");
        let ba = penetration(&b, &a).expect("b,a");
        assert!(approx(ab.depth, ba.depth, 1.0e-2));
        let dot = v_dot(ab.normal, ba.normal);
        assert!(dot < -0.99, "normals should be opposite, dot {dot}");
    }

    #[test]
    fn deeper_overlap_gives_larger_depth() {
        let a = boxed([0.0, 0.0, 0.0], 1.0);
        let shallow = penetration(&a, &boxed([1.7, 0.0, 0.0], 1.0)).expect("shallow");
        let deep = penetration(&a, &boxed([0.6, 0.0, 0.0], 1.0)).expect("deep");
        assert!(deep.depth > shallow.depth);
    }

    #[test]
    fn normal_sign_points_from_b_toward_a() {
        // b is on +x of a, so the separating push on b should be +x.
        let a = boxed([0.0, 0.0, 0.0], 1.0);
        let b = boxed([1.5, 0.0, 0.0], 1.0);
        let pen = penetration(&a, &b).expect("overlap");
        assert!(pen.normal[0] > 0.99, "normal {:?}", pen.normal);
    }

    #[test]
    fn iteration_converges_for_offset_octahedra() {
        // Octahedra force several EPA refinements; the run must still terminate
        // with a finite depth well under the iteration cap.
        let a = octa([0.0, 0.0, 0.0], 1.5);
        let b = octa([0.4, 0.3, 0.2], 1.5);
        let pen = penetration(&a, &b).expect("overlap");
        assert!(pen.depth.abs() < f32::INFINITY);
        assert!(pen.depth > 0.0);
    }

    #[test]
    fn empty_input_returns_none() {
        let a: Vec<[f32; 3]> = Vec::new();
        let b = boxed([0.0, 0.0, 0.0], 1.0);
        assert!(penetration(&a, &b).is_none());
        assert!(penetration(&b, &a).is_none());
    }

    #[test]
    fn degenerate_normalize_returns_none() {
        assert!(v_normalize([0.0, 0.0, 0.0]).is_none());
        assert!(v_normalize([EPS * 0.5, 0.0, 0.0]).is_none());
        assert!(v_normalize([3.0, 0.0, 0.0]).is_some());
    }

    #[test]
    fn support_picks_extreme_vertex() {
        let cube = boxed([0.0, 0.0, 0.0], 1.0);
        let s = support(&cube, [1.0, 1.0, 1.0]);
        assert!(approx(s[0], 1.0, 1.0e-6));
        assert!(approx(s[1], 1.0, 1.0e-6));
        assert!(approx(s[2], 1.0, 1.0e-6));
    }

    #[test]
    fn minkowski_support_is_difference_of_supports() {
        let a = boxed([0.0, 0.0, 0.0], 1.0);
        let b = boxed([5.0, 0.0, 0.0], 1.0);
        let dir = [1.0, 0.0, 0.0];
        let got = minkowski_support(&a, &b, dir);
        let want = v_sub(support(&a, dir), support(&b, v_neg(dir)));
        assert!(approx(got[0], want[0], 1.0e-6));
        assert!(approx(got[1], want[1], 1.0e-6));
        assert!(approx(got[2], want[2], 1.0e-6));
    }

    #[test]
    fn rotated_cloud_still_yields_finite_depth() {
        // A tetrahedron that is not axis aligned overlapping a box.
        let a = tetra_body([0.0, 0.0, 0.0], 1.2);
        let b = boxed([0.6, 0.4, 0.3], 1.0);
        let pen = penetration(&a, &b).expect("overlap");
        assert!(pen.depth > 0.0);
        assert!(pen.depth.abs() < f32::INFINITY);
        assert!(approx(v_len_sq(pen.normal), 1.0, 1.0e-3));
    }

    #[test]
    fn diagonal_overlap_reports_penetration() {
        let a = boxed([0.0, 0.0, 0.0], 1.0);
        let b = boxed([1.2, 1.2, 1.2], 1.0);
        let pen = penetration(&a, &b).expect("corner overlap");
        assert!(pen.depth > 0.0);
        assert!(approx(v_len_sq(pen.normal), 1.0, 1.0e-3));
    }
}
