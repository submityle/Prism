//! `Sutherland-Hodgman` convex-window polygon clipping in 2D for the
//! particle screen-space, footprint, and 2D-culling passes (design §12, §13).
//!
//! Several particle stages need the *area* of overlap between a 2D shape and a
//! convex region rather than a single trimmed segment: a splat-footprint pass
//! wants the part of a quad that survives inside a tile, a bounds-reduction pass
//! wants an emitter polygon cropped to the visible viewport, and an authoring
//! overlay wants a guide shape confined to a convex mask. This module owns the
//! small, `CPU`-verifiable reference for that operation so a future `GPU`
//! kernel can reproduce it bit for bit.
//!
//! The algorithm is the classic `Sutherland-Hodgman` polygon clip. The clip
//! window is a *convex* polygon wound counter-clockwise (`CCW`); each of its
//! directed edges `a -> b` defines a half-plane whose *inside* is the region to
//! the left of the edge (a non-negative 2D cross product). The subject polygon
//! is clipped against one window edge at a time: [`clip_edge`] walks the current
//! polygon ring once and, for each edge `cur -> nxt`, emits vertices by the four
//! in/out cases — inside-to-inside keeps the endpoint, inside-to-outside emits
//! the boundary crossing, outside-to-inside emits the crossing then the
//! endpoint, and outside-to-outside emits nothing. Feeding each window edge's
//! output into the next ([`clip_polygon`]) yields the intersection of the subject
//! with every half-plane, i.e. the subject cropped to the convex window. The
//! subject itself may be convex or concave; only the *window* must be convex.
//!
//! # Strict scope
//! This module clips *one 2D polygon against one convex 2D window* and returns
//! the surviving polygon as an ordered vertex ring. It is deliberately distinct
//! from its neighbours:
//! * [`super::plane_clip`] runs the same-named `Sutherland-Hodgman` clip but in
//!   3D against an oriented half-space *plane*; that operates on `vec3`
//!   polygons and plane coefficients, not on a 2D convex window, and this
//!   module neither imports nor reconstructs it.
//! * [`super::cohen_sutherland_clip`] trims a single 2D *segment* to an
//!   axis-aligned rectangle via 4-bit outcodes; it clips a segment, not a
//!   polygon, and only against a box.
//! * [`super::liang_barsky_clip`] is the parametric single-*segment* clip against
//!   a rectangle; again a segment, not a polygon.
//! * [`super::polygon_area_2d`] only *measures* a ring the caller already has
//!   (area, centroid, winding); it never clips one shape against another.
//!
//! # No transcendental math
//! Every step is pure `+`, `-`, `*`, `/` cross-product and linear-interpolation
//! arithmetic; there is no `sqrt`, `sin`, `cos`, `atan`, `exp`, `ln`,
//! `powf`, or any other transcendental call. `f32` magnitudes are never
//! compared with `==`/`!=`: the inside test and the parallel-edge guard both go
//! through [`CMP_EPS`], and ordinary `<`/`>=` ordering drives the in/out
//! decision. Boundary intersections use the crossing parameter
//! `t = sc / (sc - sn)` and a linear interpolation of the two endpoints, so the
//! math stays affine.
//!
//! # Layout
//! [`gpu_storage_bytes`] sizes a `std430` storage buffer holding the clipped ring
//! as one `vec2<f32>` slot per vertex, matching the shared
//! [`crate::particle::gpu_layout`] stride so a `WebGPU` kernel binds a stable
//! `ABI`.

use alloc::vec::Vec;

use crate::particle::gpu_layout::{storage_bytes, VEC2_STRIDE};

/// Magnitude below which a cross product or a coordinate difference is treated
/// as zero.
///
/// This is the comparison rule used throughout instead of `==` on `f32`: a
/// point whose signed side value does not fall below `-`[`CMP_EPS`] counts as
/// inside the half-plane (so a vertex exactly on a window edge is retained),
/// and a window edge is treated as parallel to a subject edge when the crossing
/// denominator's absolute value does not exceed this bound.
pub const CMP_EPS: f32 = 1.0e-6;

/// Componentwise 2D vector difference `lhs - rhs`.
#[must_use]
fn v_sub(lhs: [f32; 2], rhs: [f32; 2]) -> [f32; 2] {
    [lhs[0] - rhs[0], lhs[1] - rhs[1]]
}

/// 2D scalar cross product `lhs.x * rhs.y - lhs.y * rhs.x`.
///
/// Its sign is the turn sense of the two vectors: strictly positive when `rhs`
/// lies to the left of `lhs`, strictly negative to the right, and zero (within
/// [`CMP_EPS`]) when they are collinear.
#[must_use]
fn v_cross(lhs: [f32; 2], rhs: [f32; 2]) -> f32 {
    lhs[0] * rhs[1] - lhs[1] * rhs[0]
}

/// Linear interpolation `a + (b - a) * t` between two 2D points.
///
/// With `t` in `0..=1` this walks from `a` at `t = 0` to `b` at `t = 1`,
/// so a crossing parameter yields the boundary intersection point.
#[must_use]
fn lerp2(a: [f32; 2], b: [f32; 2], t: f32) -> [f32; 2] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
}

/// Signed side of point `p` relative to the directed window edge `a -> b`.
///
/// Equal to the 2D cross product `(b - a) x (p - a)`: strictly positive when `p`
/// is to the left of the edge (inside a `CCW` window), strictly negative to the
/// right (outside), and near zero on the edge line.
#[must_use]
fn edge_side(a: [f32; 2], b: [f32; 2], p: [f32; 2]) -> f32 {
    v_cross(v_sub(b, a), v_sub(p, a))
}

/// Returns `true` when two points coincide within [`CMP_EPS`] on both axes.
#[must_use]
fn points_equal(a: [f32; 2], b: [f32; 2]) -> bool {
    (a[0] - b[0]).abs() <= CMP_EPS && (a[1] - b[1]).abs() <= CMP_EPS
}

/// Crossing parameter where the subject edge with side values `sc` (start) and
/// `sn` (end) meets the window edge line.
///
/// `t = sc / (sc - sn)` is the fraction along the subject edge at which the
/// signed side reaches zero. Returns `None` when the subject edge is parallel to
/// the window edge (the denominator's magnitude does not exceed [`CMP_EPS`]); the
/// result is otherwise clamped into `0..=1` so a rounding wobble never places the
/// intersection off the segment.
#[must_use]
fn crossing_param(sc: f32, sn: f32) -> Option<f32> {
    let denom = sc - sn;
    if denom.abs() <= CMP_EPS {
        return None;
    }
    Some((sc / denom).clamp(0.0, 1.0))
}

/// Removes consecutive duplicate vertices from a ring, including the wraparound
/// pair, so a clip that lands an intersection exactly on an existing vertex does
/// not leave a zero-length edge.
///
/// Duplicates are detected with [`points_equal`], never with `f32` equality.
#[must_use]
fn dedup_ring(ring: Vec<[f32; 2]>) -> Vec<[f32; 2]> {
    if ring.len() < 2 {
        return ring;
    }
    let mut out: Vec<[f32; 2]> = Vec::new();
    for &p in &ring {
        if let Some(&last) = out.last()
            && points_equal(last, p)
        {
            continue;
        }
        out.push(p);
    }
    if out.len() >= 2 && points_equal(out[0], out[out.len() - 1]) {
        out.pop();
    }
    out
}

/// Clips a polygon ring against a single directed window edge `a -> b`
/// (`Sutherland-Hodgman`, endpoint-emitting variant).
///
/// Walks `poly` once as a closed ring. For each subject edge `cur -> nxt` the
/// two endpoints are classified inside/outside the half-plane to the left of
/// `a -> b` via [`edge_side`] (a point within [`CMP_EPS`] of the edge line
/// counts as inside). The four cases emit:
/// * inside to inside — the endpoint `nxt`;
/// * inside to outside — the boundary crossing;
/// * outside to inside — the boundary crossing, then `nxt`;
/// * outside to outside — nothing.
///
/// An empty or single-vertex input yields an empty [`Vec`].
#[must_use]
pub fn clip_edge(poly: &[[f32; 2]], a: [f32; 2], b: [f32; 2]) -> Vec<[f32; 2]> {
    let n = poly.len();
    if n < 2 {
        return Vec::new();
    }
    let mut out: Vec<[f32; 2]> = Vec::new();
    for (i, &cur) in poly.iter().enumerate() {
        let nxt = poly[(i + 1) % n];
        let sc = edge_side(a, b, cur);
        let sn = edge_side(a, b, nxt);
        let cur_in = sc >= -CMP_EPS;
        let nxt_in = sn >= -CMP_EPS;
        if cur_in && nxt_in {
            out.push(nxt);
        } else if cur_in {
            if let Some(t) = crossing_param(sc, sn) {
                out.push(lerp2(cur, nxt, t));
            }
        } else if nxt_in {
            if let Some(t) = crossing_param(sc, sn) {
                out.push(lerp2(cur, nxt, t));
            }
            out.push(nxt);
        }
    }
    out
}

/// Clips the `subject` polygon against the convex `clip` window
/// (`Sutherland-Hodgman`).
///
/// The window must be a convex polygon wound counter-clockwise (`CCW`); the
/// subject may be convex or concave. The subject is clipped against each window
/// edge in turn with [`clip_edge`], so the result is the subject cropped to the
/// intersection of every window half-plane, returned as an ordered vertex ring.
/// Consecutive and wraparound duplicate vertices are collapsed with
/// [`dedup_ring`].
///
/// Returns an empty [`Vec`] when either polygon has fewer than three vertices,
/// when the subject lies wholly outside the window, or when the clip degenerates
/// the ring below three vertices (a sliver or point).
#[must_use]
pub fn clip_polygon(subject: &[[f32; 2]], clip: &[[f32; 2]]) -> Vec<[f32; 2]> {
    let m = clip.len();
    if subject.len() < 3 || m < 3 {
        return Vec::new();
    }
    let mut out: Vec<[f32; 2]> = subject.to_vec();
    for (i, &a) in clip.iter().enumerate() {
        if out.is_empty() {
            break;
        }
        let b = clip[(i + 1) % m];
        out = clip_edge(&out, a, b);
    }
    let out = dedup_ring(out);
    if out.len() < 3 {
        Vec::new()
    } else {
        out
    }
}

/// Byte size of a `std430` storage buffer holding a clipped ring of `count`
/// vertices, one `vec2<f32>` slot each.
///
/// Delegates to [`crate::particle::gpu_layout::storage_bytes`] with the shared
/// [`VEC2_STRIDE`] so a `WebGPU` kernel that consumes the clipped polygon binds
/// the same `ABI` as the rest of the particle 2D contracts.
#[must_use]
pub fn gpu_storage_bytes(count: usize) -> usize {
    storage_bytes(VEC2_STRIDE, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Twice the signed shoelace area of a ring, positive for a `CCW` winding.
    fn signed_area2(poly: &[[f32; 2]]) -> f32 {
        let n = poly.len();
        if n < 3 {
            return 0.0;
        }
        let mut acc = 0.0f32;
        for (i, &p) in poly.iter().enumerate() {
            let q = poly[(i + 1) % n];
            acc += p[0] * q[1] - q[0] * p[1];
        }
        acc
    }

    /// Absolute polygon area from the shoelace sum.
    fn area(poly: &[[f32; 2]]) -> f32 {
        signed_area2(poly).abs() * 0.5
    }

    /// Returns `true` when `p` lies inside the `CCW` convex `window` (left of
    /// every directed edge within tolerance).
    fn inside_window(window: &[[f32; 2]], p: [f32; 2]) -> bool {
        let m = window.len();
        for (i, &a) in window.iter().enumerate() {
            let b = window[(i + 1) % m];
            if edge_side(a, b, p) < -1.0e-4 {
                return false;
            }
        }
        true
    }

    /// A rectangle `[x0, x1] x [y0, y1]` wound `CCW`.
    fn square(x0: f32, y0: f32, x1: f32, y1: f32) -> Vec<[f32; 2]> {
        Vec::from([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    }

    #[test]
    fn v_sub_is_componentwise() {
        assert_eq!(v_sub([3.0, 5.0], [1.0, 2.0]), [2.0, 3.0]);
    }

    #[test]
    fn v_cross_sign_follows_turn() {
        assert!(v_cross([1.0, 0.0], [0.0, 1.0]) > 0.0);
        assert!(v_cross([1.0, 0.0], [0.0, -1.0]) < 0.0);
        assert!(v_cross([1.0, 0.0], [2.0, 0.0]).abs() <= CMP_EPS);
    }

    #[test]
    fn lerp2_midpoint() {
        let m = lerp2([0.0, 0.0], [4.0, 2.0], 0.5);
        assert!((m[0] - 2.0).abs() <= CMP_EPS);
        assert!((m[1] - 1.0).abs() <= CMP_EPS);
    }

    #[test]
    fn edge_side_left_is_positive() {
        assert!(edge_side([0.0, 0.0], [1.0, 0.0], [0.5, 1.0]) > 0.0);
        assert!(edge_side([0.0, 0.0], [1.0, 0.0], [0.5, -1.0]) < 0.0);
    }

    #[test]
    fn crossing_param_parallel_is_none() {
        assert!(crossing_param(1.0, 1.0).is_none());
    }

    #[test]
    fn crossing_param_midpoint() {
        let t = crossing_param(1.0, -1.0).expect("crossing");
        assert!((t - 0.5).abs() <= CMP_EPS);
    }

    #[test]
    fn subject_fully_inside_is_unchanged_area() {
        let subject = square(1.0, 1.0, 3.0, 3.0);
        let window = square(0.0, 0.0, 4.0, 4.0);
        let out = clip_polygon(&subject, &window);
        assert_eq!(out.len(), 4);
        assert!((area(&out) - 4.0).abs() <= 1.0e-4);
    }

    #[test]
    fn subject_fully_inside_keeps_vertices() {
        let subject = square(1.0, 1.0, 3.0, 3.0);
        let window = square(0.0, 0.0, 4.0, 4.0);
        let out = clip_polygon(&subject, &window);
        for v in &subject {
            assert!(out.iter().any(|o| points_equal(*o, *v)));
        }
    }

    #[test]
    fn subject_fully_outside_is_empty() {
        let subject = square(10.0, 10.0, 12.0, 12.0);
        let window = square(0.0, 0.0, 4.0, 4.0);
        assert!(clip_polygon(&subject, &window).is_empty());
    }

    #[test]
    fn square_clips_square_to_overlap() {
        let subject = square(0.0, 0.0, 4.0, 4.0);
        let window = square(2.0, 2.0, 6.0, 6.0);
        let out = clip_polygon(&subject, &window);
        assert_eq!(out.len(), 4);
        assert!((area(&out) - 4.0).abs() <= 1.0e-4);
        for &p in &out {
            assert!(inside_window(&window, p));
        }
    }

    #[test]
    fn triangle_clips_to_pentagon_known_area() {
        let subject = Vec::from([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]]);
        let window = square(0.0, 0.0, 3.0, 3.0);
        let out = clip_polygon(&subject, &window);
        assert_eq!(out.len(), 5);
        assert!((area(&out) - 7.0).abs() <= 1.0e-4);
    }

    #[test]
    fn triangle_clips_triangle() {
        let subject = Vec::from([[0.0, 0.0], [6.0, 0.0], [0.0, 6.0]]);
        let window = Vec::from([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]]);
        let out = clip_polygon(&subject, &window);
        assert!(out.len() >= 3);
        assert!((area(&out) - 0.5).abs() <= 1.0e-4);
    }

    #[test]
    fn rectangle_partial_clip_area() {
        let subject = square(0.0, 0.0, 6.0, 2.0);
        let window = square(3.0, -1.0, 10.0, 3.0);
        let out = clip_polygon(&subject, &window);
        assert!((area(&out) - 6.0).abs() <= 1.0e-4);
        for &p in &out {
            assert!(inside_window(&window, p));
        }
    }

    #[test]
    fn concave_subject_l_shape_clipped() {
        let subject = Vec::from([
            [0.0, 0.0],
            [4.0, 0.0],
            [4.0, 1.0],
            [1.0, 1.0],
            [1.0, 4.0],
            [0.0, 4.0],
        ]);
        let window = square(0.0, 0.0, 4.0, 2.0);
        let out = clip_polygon(&subject, &window);
        assert!(out.len() >= 3);
        assert!(area(&out) > 0.0);
        for &p in &out {
            assert!(inside_window(&window, p));
        }
    }

    #[test]
    fn concave_subject_area_matches_expected() {
        let subject = Vec::from([
            [0.0, 0.0],
            [4.0, 0.0],
            [4.0, 1.0],
            [1.0, 1.0],
            [1.0, 4.0],
            [0.0, 4.0],
        ]);
        let window = square(0.0, 0.0, 4.0, 2.0);
        let out = clip_polygon(&subject, &window);
        assert!((area(&out) - 5.0).abs() <= 1.0e-4);
    }

    #[test]
    fn subject_equal_to_window_is_preserved() {
        let poly = square(0.0, 0.0, 5.0, 5.0);
        let out = clip_polygon(&poly, &poly);
        assert_eq!(out.len(), 4);
        assert!((area(&out) - 25.0).abs() <= 1.0e-3);
    }

    #[test]
    fn single_point_subject_is_empty() {
        let subject = Vec::from([[1.0, 1.0]]);
        let window = square(0.0, 0.0, 4.0, 4.0);
        assert!(clip_polygon(&subject, &window).is_empty());
    }

    #[test]
    fn two_point_subject_is_empty() {
        let subject = Vec::from([[1.0, 1.0], [2.0, 2.0]]);
        let window = square(0.0, 0.0, 4.0, 4.0);
        assert!(clip_polygon(&subject, &window).is_empty());
    }

    #[test]
    fn empty_subject_is_empty() {
        let window = square(0.0, 0.0, 4.0, 4.0);
        assert!(clip_polygon(&[], &window).is_empty());
    }

    #[test]
    fn degenerate_window_is_empty() {
        let subject = square(0.0, 0.0, 4.0, 4.0);
        let window = Vec::from([[0.0, 0.0], [1.0, 1.0]]);
        assert!(clip_polygon(&subject, &window).is_empty());
    }

    #[test]
    fn vertex_on_boundary_counts_as_inside() {
        let subject = square(0.0, 0.0, 4.0, 4.0);
        let window = square(0.0, 0.0, 4.0, 4.0);
        let out = clip_polygon(&subject, &window);
        assert_eq!(out.len(), 4);
        assert!((area(&out) - 16.0).abs() <= 1.0e-3);
    }

    #[test]
    fn all_result_vertices_lie_within_window() {
        let subject = square(-2.0, -2.0, 5.0, 5.0);
        let window = Vec::from([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]]);
        let out = clip_polygon(&subject, &window);
        assert!(!out.is_empty());
        for &p in &out {
            assert!(inside_window(&window, p));
        }
    }

    #[test]
    fn clipped_area_never_exceeds_subject_area() {
        let subject = square(0.0, 0.0, 4.0, 4.0);
        let window = square(1.0, 1.0, 3.0, 3.0);
        let out = clip_polygon(&subject, &window);
        assert!(area(&out) <= area(&subject) + 1.0e-4);
        assert!((area(&out) - 4.0).abs() <= 1.0e-4);
    }

    #[test]
    fn ccw_winding_is_preserved() {
        let subject = square(0.0, 0.0, 4.0, 4.0);
        let window = square(1.0, 1.0, 6.0, 6.0);
        let out = clip_polygon(&subject, &window);
        assert!(signed_area2(&out) > 0.0);
    }

    #[test]
    fn order_is_preserved_for_inside_case() {
        let subject = square(1.0, 1.0, 2.0, 2.0);
        let window = square(0.0, 0.0, 4.0, 4.0);
        let out = clip_polygon(&subject, &window);
        assert_eq!(out.len(), 4);
        let start = out
            .iter()
            .position(|p| points_equal(*p, subject[0]))
            .expect("first vertex present");
        for (k, sv) in subject.iter().enumerate() {
            let got = out[(start + k) % out.len()];
            assert!(points_equal(got, *sv));
        }
    }

    #[test]
    fn corner_clip_recovers_triangle_window() {
        let subject = square(0.0, 0.0, 4.0, 4.0);
        let window = Vec::from([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]]);
        let out = clip_polygon(&subject, &window);
        assert!((area(&out) - 8.0).abs() <= 1.0e-4);
    }

    #[test]
    fn parallel_edges_do_not_panic() {
        let subject = square(1.0, 1.0, 3.0, 3.0);
        let window = square(1.0, 1.0, 3.0, 3.0);
        let out = clip_polygon(&subject, &window);
        assert_eq!(out.len(), 4);
        assert!((area(&out) - 4.0).abs() <= 1.0e-3);
    }

    #[test]
    fn clip_edge_all_inside_keeps_ring() {
        let poly = square(0.0, 1.0, 2.0, 3.0);
        let out = clip_edge(&poly, [0.0, 0.0], [1.0, 0.0]);
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn clip_edge_all_outside_is_empty() {
        let poly = square(0.0, -3.0, 2.0, -1.0);
        let out = clip_edge(&poly, [0.0, 0.0], [1.0, 0.0]);
        assert!(out.is_empty());
    }

    #[test]
    fn clip_edge_straddling_emits_crossings() {
        let poly = square(0.0, -2.0, 2.0, 2.0);
        let out = clip_edge(&poly, [0.0, 0.0], [1.0, 0.0]);
        assert!(out.len() >= 3);
        for &p in &out {
            assert!(p[1] >= -CMP_EPS);
        }
        assert!((area(&out) - 4.0).abs() <= 1.0e-4);
    }

    #[test]
    fn clip_edge_rejects_short_ring() {
        assert!(clip_edge(&[[0.0, 0.0]], [0.0, 0.0], [1.0, 0.0]).is_empty());
    }

    #[test]
    fn far_window_removes_subject() {
        let subject = square(0.0, 0.0, 1.0, 1.0);
        let window = square(5.0, 5.0, 9.0, 9.0);
        assert!(clip_polygon(&subject, &window).is_empty());
    }

    #[test]
    fn gpu_storage_bytes_matches_stride() {
        assert_eq!(gpu_storage_bytes(4), VEC2_STRIDE * 4);
        assert_eq!(gpu_storage_bytes(0), VEC2_STRIDE);
    }

    #[test]
    fn dedup_ring_removes_wraparound_duplicate() {
        let ring = Vec::from([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]);
        let out = dedup_ring(ring);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn offset_window_clips_to_intersection() {
        let subject = square(0.0, 0.0, 4.0, 4.0);
        let window = square(-1.0, 1.0, 2.0, 5.0);
        let out = clip_polygon(&subject, &window);
        assert!((area(&out) - 6.0).abs() <= 1.0e-4);
    }
}
