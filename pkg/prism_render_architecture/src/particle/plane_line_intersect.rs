//! Point intersection of an oriented plane with an infinite line, a finite
//! segment, or a ray, with a parallel / coincident classification.
//!
//! Where [`crate::particle::plane_clip`] runs the `Sutherland-Hodgman` convex
//! polygon-versus-plane clip and [`crate::particle::plane_aabb_classify`] only
//! labels an axis-aligned box as `above` / `below` / `intersecting` without
//! ever solving for a point, this module owns the lower-level primitive both of
//! them build on: the single scalar solve that turns a plane and a line into a
//! concrete intersection point plus its ray parameter `t`. It is deliberately
//! narrower than the 2D clippers `sutherland_hodgman_2d` and
//! `liang_barsky_clip`, which chain many of these solves along polygon edges or
//! clip windows; here a single line meets a single plane and the result is one
//! of three cases: a [`LinePlaneResult::Point`], a
//! [`LinePlaneResult::Parallel`] (the line misses the plane), or a
//! [`LinePlaneResult::Coincident`] (the line lies inside the plane).
//!
//! # Conventions
//! A plane is `normal · x = d`, i.e. the point-normal / Hessian form scaled so
//! the right-hand side is the constant `d`. `normal` is expected to be unit
//! length; then [`signed_distance`] returns the true signed distance
//! `normal · point - d`, positive on the side the normal points toward. The
//! intersection point and `t` are invariant when `normal` and `d` are scaled by
//! the same nonzero factor, so a non-unit normal still yields the correct point
//! (only the reported distance rescales); a test pins this down.
//!
//! A line is parameterized `p(t) = p0 + t * dir`. For an infinite line `t` is
//! unbounded; a segment keeps only `t` in `[0, 1]` (endpoints inclusive); a ray
//! keeps only `t >= 0`. Because a coincident or parallel line has no single
//! crossing point, the bounded [`segment_plane_intersect`] and
//! [`ray_plane_intersect`] helpers report [`None`] in those two cases.
//!
//! # Determinism
//! The only non-`+ - * /` primitive here is [`f32::abs`] for the tolerant
//! parallel / on-plane tests; there are no transcendental functions and no
//! platform math, so the reference matches a future `GPU` kernel bit for bit.
//! `f32` magnitudes are never compared with `==` / `!=`: every equality-like
//! test goes through [`EPS`]. `NaN` inputs are not special-cased; callers are
//! expected to pass finite geometry.

/// Epsilon for tolerant `f32` comparisons; direct `==` / `!=` is forbidden.
///
/// Used both as the floor on `|normal · dir|` below which the line is treated as
/// parallel to the plane, and as the band around zero within which a point is
/// treated as lying on the plane.
pub const EPS: f32 = 1.0e-6;

/// Dot product of two 3-vectors.
pub fn v_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Component-wise difference `a - b`.
pub fn v_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Component-wise sum `a + b`.
pub fn v_add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Scale a 3-vector by a scalar.
pub fn v_scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Signed distance from `point` to the plane `normal · x = d`.
///
/// Returns `normal · point - d`, which is the true signed distance when
/// `normal` is unit length and is positive on the side the normal points
/// toward.
pub fn signed_distance(point: [f32; 3], normal: [f32; 3], d: f32) -> f32 {
    v_dot(normal, point) - d
}

/// Outcome of intersecting an infinite line with a plane.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LinePlaneResult {
    /// The line crosses the plane at a single point, given with its ray
    /// parameter `t` so `point == p0 + t * dir`.
    Point([f32; 3], f32),
    /// The line is parallel to the plane and lies off it: there is no
    /// intersection.
    Parallel,
    /// The line lies inside the plane: every point of the line intersects it.
    Coincident,
}

/// Intersect the infinite line `p0 + t * dir` with the plane `normal · x = d`.
///
/// When `|normal · dir| <= EPS` the line is parallel to the plane and the
/// result depends on `p0`: [`LinePlaneResult::Coincident`] if `p0` lies on the
/// plane, otherwise [`LinePlaneResult::Parallel`]. Otherwise the unique
/// crossing is returned as [`LinePlaneResult::Point`] with an unbounded `t`.
pub fn line_plane_intersect(
    p0: [f32; 3],
    dir: [f32; 3],
    normal: [f32; 3],
    d: f32,
) -> LinePlaneResult {
    let denom = v_dot(normal, dir);
    if denom.abs() <= EPS {
        if signed_distance(p0, normal, d).abs() <= EPS {
            return LinePlaneResult::Coincident;
        }
        return LinePlaneResult::Parallel;
    }
    let t = (d - v_dot(normal, p0)) / denom;
    LinePlaneResult::Point(v_add(p0, v_scale(dir, t)), t)
}

/// Intersect the finite segment from `p0` to `p1` with the plane.
///
/// Returns the crossing point and its parameter `t` only when `t` falls in
/// `[0, 1]` (endpoints on the plane are included via [`EPS`]). A segment that is
/// parallel to or coincident with the plane has no single crossing point and
/// returns [`None`].
pub fn segment_plane_intersect(
    p0: [f32; 3],
    p1: [f32; 3],
    normal: [f32; 3],
    d: f32,
) -> Option<([f32; 3], f32)> {
    let dir = v_sub(p1, p0);
    match line_plane_intersect(p0, dir, normal, d) {
        LinePlaneResult::Point(point, t) => {
            if (-EPS..=1.0 + EPS).contains(&t) {
                Some((point, t))
            } else {
                None
            }
        }
        LinePlaneResult::Parallel | LinePlaneResult::Coincident => None,
    }
}

/// Intersect the ray `origin + t * dir` (with `t >= 0`) with the plane.
///
/// Returns the crossing point and its parameter `t` only when `t >= 0` (the
/// origin on the plane is included via [`EPS`]). A ray that is parallel to or
/// coincident with the plane has no single crossing point and returns [`None`].
pub fn ray_plane_intersect(
    origin: [f32; 3],
    dir: [f32; 3],
    normal: [f32; 3],
    d: f32,
) -> Option<([f32; 3], f32)> {
    match line_plane_intersect(origin, dir, normal, d) {
        LinePlaneResult::Point(point, t) => {
            if t >= -EPS {
                Some((point, t))
            } else {
                None
            }
        }
        LinePlaneResult::Parallel | LinePlaneResult::Coincident => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1.0e-4
    }

    fn approx_vec(a: [f32; 3], b: [f32; 3]) -> bool {
        approx(a[0], b[0]) && approx(a[1], b[1]) && approx(a[2], b[2])
    }

    // A small `splitmix32`-style generator for the randomized on-plane check;
    // it stays inside integer wrapping arithmetic plus an `as f32` cast.
    struct Rng {
        state: u32,
    }

    impl Rng {
        fn new(seed: u32) -> Self {
            Self { state: seed }
        }

        fn next_u32(&mut self) -> u32 {
            self.state = self.state.wrapping_add(0x9E37_79B9);
            let mut z = self.state;
            z = (z ^ (z >> 16)).wrapping_mul(0x21F0_AAAD);
            z = (z ^ (z >> 15)).wrapping_mul(0x735A_2D97);
            z ^ (z >> 15)
        }

        // Uniform-ish value in `[-1, 1)`.
        fn unit(&mut self) -> f32 {
            let v = (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32;
            v * 2.0 - 1.0
        }
    }

    fn normalize(v: [f32; 3]) -> [f32; 3] {
        let len = v_dot(v, v).sqrt();
        if len <= EPS {
            [0.0, 0.0, 1.0]
        } else {
            v_scale(v, 1.0 / len)
        }
    }

    #[test]
    fn line_through_plane_basic() {
        // Plane x = 0, line along +x from x = -2.
        let r = line_plane_intersect([-2.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.0);
        match r {
            LinePlaneResult::Point(p, t) => {
                assert!(approx(t, 2.0));
                assert!(approx_vec(p, [0.0, 0.0, 0.0]));
            }
            _ => panic!("expected point"),
        }
    }

    #[test]
    fn line_point_lies_on_plane() {
        let normal = [0.0, 1.0, 0.0];
        let d = 3.0;
        let r = line_plane_intersect([5.0, -1.0, 2.0], [0.0, 1.0, 0.0], normal, d);
        match r {
            LinePlaneResult::Point(p, _) => {
                assert!(approx(signed_distance(p, normal, d), 0.0));
            }
            _ => panic!("expected point"),
        }
    }

    #[test]
    fn line_parallel_off_plane_returns_parallel() {
        // Plane y = 0, line along +x at y = 5.
        let r = line_plane_intersect([0.0, 5.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.0);
        assert_eq!(r, LinePlaneResult::Parallel);
    }

    #[test]
    fn line_coincident_returns_coincident() {
        // Plane y = 0, line along +x at y = 0.
        let r = line_plane_intersect([-3.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.0);
        assert_eq!(r, LinePlaneResult::Coincident);
    }

    #[test]
    fn line_negative_t_behind_origin() {
        // Plane x = 0, p0 at x = 2 heading +x: crossing is behind at t = -2.
        let r = line_plane_intersect([2.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.0);
        match r {
            LinePlaneResult::Point(_, t) => assert!(approx(t, -2.0)),
            _ => panic!("expected point"),
        }
    }

    #[test]
    fn line_flipped_normal_same_point() {
        let a = line_plane_intersect([-2.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.0);
        let b = line_plane_intersect([-2.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], 0.0);
        match (a, b) {
            (LinePlaneResult::Point(pa, _), LinePlaneResult::Point(pb, _)) => {
                assert!(approx_vec(pa, pb));
            }
            _ => panic!("expected points"),
        }
    }

    #[test]
    fn line_offset_plane_d_nonzero() {
        // Plane x = 4, line along +x from origin: t = 4.
        let r = line_plane_intersect([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 4.0);
        match r {
            LinePlaneResult::Point(p, t) => {
                assert!(approx(t, 4.0));
                assert!(approx_vec(p, [4.0, 0.0, 0.0]));
            }
            _ => panic!("expected point"),
        }
    }

    #[test]
    fn segment_crosses_two_sides() {
        // Plane z = 0, segment from z = -1 to z = 1: crosses at t = 0.5.
        let hit = segment_plane_intersect([0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], 0.0);
        let (p, t) = hit.expect("expected hit");
        assert!(approx(t, 0.5));
        assert!(approx_vec(p, [0.0, 0.0, 0.0]));
    }

    #[test]
    fn segment_same_side_no_hit() {
        // Both endpoints at z > 0.
        let hit = segment_plane_intersect([0.0, 0.0, 1.0], [0.0, 0.0, 3.0], [0.0, 0.0, 1.0], 0.0);
        assert!(hit.is_none());
    }

    #[test]
    fn segment_endpoint_on_plane_t0() {
        let hit = segment_plane_intersect([0.0, 0.0, 0.0], [0.0, 0.0, 4.0], [0.0, 0.0, 1.0], 0.0);
        let (p, t) = hit.expect("expected hit");
        assert!(approx(t, 0.0));
        assert!(approx_vec(p, [0.0, 0.0, 0.0]));
    }

    #[test]
    fn segment_endpoint_on_plane_t1() {
        let hit = segment_plane_intersect([0.0, 0.0, -4.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.0);
        let (_, t) = hit.expect("expected hit");
        assert!(approx(t, 1.0));
    }

    #[test]
    fn segment_parallel_none() {
        let hit = segment_plane_intersect([0.0, 2.0, 0.0], [4.0, 2.0, 0.0], [0.0, 1.0, 0.0], 0.0);
        assert!(hit.is_none());
    }

    #[test]
    fn segment_coincident_none() {
        let hit = segment_plane_intersect([0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.0);
        assert!(hit.is_none());
    }

    #[test]
    fn segment_quarter_crossing() {
        // Plane x = 0, segment from x = -1 to x = 3: crossing at t = 0.25.
        let hit = segment_plane_intersect([-1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.0);
        let (_, t) = hit.expect("expected hit");
        assert!(approx(t, 0.25));
    }

    #[test]
    fn segment_beyond_far_end_none() {
        // Plane x = 4 but the segment only reaches x = 3.
        let hit = segment_plane_intersect([0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.0, 0.0, 0.0], 4.0);
        assert!(hit.is_none());
    }

    #[test]
    fn ray_forward_hit() {
        let hit = ray_plane_intersect([-2.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.0);
        let (p, t) = hit.expect("expected hit");
        assert!(approx(t, 2.0));
        assert!(approx_vec(p, [0.0, 0.0, 0.0]));
    }

    #[test]
    fn ray_backward_miss() {
        // Origin past the plane heading away: crossing is at t < 0.
        let hit = ray_plane_intersect([2.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.0);
        assert!(hit.is_none());
    }

    #[test]
    fn ray_origin_on_plane_t0() {
        let hit = ray_plane_intersect([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.0);
        let (_, t) = hit.expect("expected hit");
        assert!(approx(t, 0.0));
    }

    #[test]
    fn ray_parallel_none() {
        let hit = ray_plane_intersect([0.0, 3.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.0);
        assert!(hit.is_none());
    }

    #[test]
    fn ray_coincident_none() {
        let hit = ray_plane_intersect([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.0);
        assert!(hit.is_none());
    }

    #[test]
    fn signed_distance_positive() {
        // Point on the +normal side of x = 0.
        assert!(signed_distance([2.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.0) > 0.0);
    }

    #[test]
    fn signed_distance_negative() {
        assert!(signed_distance([-2.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.0) < 0.0);
    }

    #[test]
    fn signed_distance_zero_on_plane() {
        assert!(approx(
            signed_distance([4.0, 1.0, 9.0], [1.0, 0.0, 0.0], 4.0),
            0.0
        ));
    }

    #[test]
    fn non_unit_normal_scaled_same_point() {
        // Scaling normal and d by the same factor leaves the point unchanged.
        let a = line_plane_intersect([-2.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.5);
        let b = line_plane_intersect([-2.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0], 1.5);
        match (a, b) {
            (LinePlaneResult::Point(pa, ta), LinePlaneResult::Point(pb, tb)) => {
                assert!(approx_vec(pa, pb));
                assert!(approx(ta, tb));
            }
            _ => panic!("expected points"),
        }
    }

    #[test]
    fn different_normal_orientations_all_solve() {
        let normals = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        for n in normals {
            // Fire a line along the normal from -2 * normal; it must hit at the
            // origin plane `n · x = 0`.
            let p0 = v_scale(n, -2.0);
            match line_plane_intersect(p0, n, n, 0.0) {
                LinePlaneResult::Point(p, t) => {
                    assert!(approx(t, 2.0));
                    assert!(approx(signed_distance(p, n, 0.0), 0.0));
                }
                _ => panic!("expected point"),
            }
        }
    }

    #[test]
    fn v_dot_value() {
        assert!(approx(v_dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), 32.0));
    }

    #[test]
    fn v_sub_add_scale_values() {
        assert!(approx_vec(
            v_sub([3.0, 5.0, 7.0], [1.0, 2.0, 3.0]),
            [2.0, 3.0, 4.0]
        ));
        assert!(approx_vec(
            v_add([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
            [5.0, 7.0, 9.0]
        ));
        assert!(approx_vec(v_scale([1.0, -2.0, 3.0], 2.0), [2.0, -4.0, 6.0]));
    }

    #[test]
    fn eps_is_small_positive() {
        assert!(EPS > 0.0);
        assert!(EPS < 1.0e-3);
    }

    #[test]
    fn random_intersections_land_on_plane() {
        let mut rng = Rng::new(0x1234_5678);
        let mut checked = 0u32;
        for _ in 0..512 {
            let normal = normalize([rng.unit(), rng.unit(), rng.unit()]);
            let d = rng.unit() * 4.0;
            let p0 = [rng.unit() * 6.0, rng.unit() * 6.0, rng.unit() * 6.0];
            let dir = [rng.unit(), rng.unit(), rng.unit()];
            // Skip near-parallel lines to keep the solve well conditioned.
            if v_dot(normal, dir).abs() <= 1.0e-2 {
                continue;
            }
            match line_plane_intersect(p0, dir, normal, d) {
                LinePlaneResult::Point(p, t) => {
                    // The reported point sits on the plane...
                    assert!(signed_distance(p, normal, d).abs() <= 1.0e-3);
                    // ...and matches the parametric reconstruction.
                    assert!(approx_vec(p, v_add(p0, v_scale(dir, t))));
                    checked += 1;
                }
                _ => panic!("non-parallel line should hit"),
            }
        }
        assert!(checked > 0);
    }
}
