//! Exact minimum-enclosing-ball contract for the particle subsystem: Welzl's
//! algorithm for the smallest enclosing circle (2D) and smallest enclosing
//! sphere (3D) of a particle position cloud (design 12, 13).
//!
//! Given a set of particle positions this module returns the provably
//! *smallest* circle or sphere that still contains every point. Unlike a
//! bounding box, the result is rotation invariant and is uniquely determined by
//! its support set: 1, 2, or 3 points on the boundary in 2D, and 1, 2, 3, or 4
//! points on the boundary in 3D. The construction is Emo Welzl's expected
//! linear-time incremental algorithm in its move-to-front form: walk the points
//! once; whenever a point falls outside the current ball, rebuild the ball with
//! that point pinned to the boundary and recurse over the earlier points. The
//! tiny "trivial" balls resting on a support set are solved in closed form by
//! Cramer's rule (a determinant solve for the circumcenter), so no iterative
//! optimizer and no transcendental function is ever needed.
//!
//! # Relationship to the sibling modules (strict boundary)
//!
//! * [`crate::particle::ritter_bounding_sphere`] builds a *fast approximate*
//!   bounding sphere in two linear passes; it is guaranteed to enclose every
//!   point but is typically a few percent larger than optimal. This module is
//!   the *exact* counterpart: it returns the true minimum-enclosing ball, at
//!   the cost of the heavier Welzl recursion. Reach for Ritter when speed
//!   dominates and for Welzl when tightness dominates.
//! * [`crate::particle::sphere_aabb`] answers a *sphere-versus-box proximity*
//!   query (closest point, overlap, penetration). It consumes a sphere; it
//!   never constructs one. This module only *constructs* the ball and never
//!   performs an intersection or contact query.
//!
//! # Numerical discipline
//!
//! Everything is a zero-dependency contract. The vector math is hand-rolled in
//! this file with only `+ - * /` plus `f32::sqrt` for the radius and
//! `f32::abs`, `f32::min`, `f32::max` for guards. No transcendental function
//! (no `sin`/`cos`/`atan`/`powf`/`exp`/`ln`) is ever called, and no exact `==`
//! or `!=` is ever written on a production `f32`: degeneracy is detected by
//! comparing a determinant against a relative epsilon, and containment is a
//! tolerant `<=` test. That keeps the `CPU` reference here in lockstep with a
//! future `GPU` kernel and free of `NaN`-producing divisions.

use alloc::vec::Vec;

/// Relative epsilon used to decide that a defining determinant has collapsed
/// (collinear points in 2D, coplanar points in 3D). It is compared against the
/// product of the squared or plain edge magnitudes, so the test is scale free.
const DEGEN_REL: f32 = 1.0e-7;

/// Small absolute floor added to the degeneracy threshold so that an all-zero
/// (fully coincident) support set is classified as degenerate rather than
/// dividing by a hard `0.0`.
const DEGEN_ABS: f32 = 1.0e-30;

/// Absolute slack added to every containment test so a boundary point that is
/// mathematically *on* the ball is never rejected by rounding.
const CONTAIN_ABS: f32 = 1.0e-5;

/// Radius-relative slack added to every containment test so the tolerance grows
/// with the working scale of the point cloud.
const CONTAIN_REL: f32 = 1.0e-6;

/// A smallest enclosing circle in the plane: a center and a non-negative radius.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sphere2 {
    /// The circle center `[x, y]`.
    pub center: [f32; 2],
    /// The circle radius; never negative.
    pub radius: f32,
}

impl Sphere2 {
    /// Returns `true` when `p` lies inside or on the circle within tolerance.
    #[must_use]
    pub fn contains(self, p: [f32; 2]) -> bool {
        let dx = p[0] - self.center[0];
        let dy = p[1] - self.center[1];
        let d = (dx * dx + dy * dy).sqrt();
        d <= self.radius + CONTAIN_ABS + CONTAIN_REL * self.radius
    }
}

/// A smallest enclosing sphere in space: a center and a non-negative radius.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sphere3 {
    /// The sphere center `[x, y, z]`.
    pub center: [f32; 3],
    /// The sphere radius; never negative.
    pub radius: f32,
}

impl Sphere3 {
    /// Returns `true` when `p` lies inside or on the sphere within tolerance.
    #[must_use]
    pub fn contains(self, p: [f32; 3]) -> bool {
        let dx = p[0] - self.center[0];
        let dy = p[1] - self.center[1];
        let dz = p[2] - self.center[2];
        let d = (dx * dx + dy * dy + dz * dz).sqrt();
        d <= self.radius + CONTAIN_ABS + CONTAIN_REL * self.radius
    }
}

// ----------------------------------------------------------------------------
// Hand-rolled vector helpers (free functions to avoid operator-trait lints).
// ----------------------------------------------------------------------------

/// Component-wise difference `a - b` in the plane.
fn sub2(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

/// Dot product of two planar vectors.
fn dot2(a: [f32; 2], b: [f32; 2]) -> f32 {
    a[0] * b[0] + a[1] * b[1]
}

/// Squared distance between two planar points.
fn dist2_2(a: [f32; 2], b: [f32; 2]) -> f32 {
    let d = sub2(a, b);
    dot2(d, d)
}

/// Component-wise difference `a - b` in space.
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Dot product of two spatial vectors.
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Squared distance between two spatial points.
fn dist2_3(a: [f32; 3], b: [f32; 3]) -> f32 {
    let d = sub3(a, b);
    dot3(d, d)
}

/// Determinant of a 3x3 matrix given as three row vectors.
fn det3x3(m: [[f32; 3]; 3]) -> f32 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

// ----------------------------------------------------------------------------
// Closed-form "trivial" balls resting on a small support set.
// ----------------------------------------------------------------------------

/// The circle whose diameter is the segment `a`-`b`.
fn circle_diameter_2(a: [f32; 2], b: [f32; 2]) -> Sphere2 {
    let center = [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5];
    let radius = dist2_2(a, b).sqrt() * 0.5;
    Sphere2 { center, radius }
}

/// The circumscribed circle of the triangle `a`, `b`, `c`, or `None` when the
/// three points are (near) collinear.
fn circumcircle_2(a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> Option<Sphere2> {
    let ab = sub2(b, a);
    let ac = sub2(c, a);
    let d11 = dot2(ab, ab);
    let d12 = dot2(ab, ac);
    let d22 = dot2(ac, ac);
    let det = d11 * d22 - d12 * d12;
    if det.abs() <= DEGEN_REL * d11 * d22 + DEGEN_ABS {
        return None;
    }
    let r1 = d11 * 0.5;
    let r2 = d22 * 0.5;
    let alpha = (r1 * d22 - r2 * d12) / det;
    let beta = (d11 * r2 - d12 * r1) / det;
    let center = [
        a[0] + alpha * ab[0] + beta * ac[0],
        a[1] + alpha * ab[1] + beta * ac[1],
    ];
    let radius = dist2_2(center, a).sqrt();
    Some(Sphere2 { center, radius })
}

/// The sphere whose diameter is the segment `a`-`b`.
fn sphere_diameter_3(a: [f32; 3], b: [f32; 3]) -> Sphere3 {
    let center = [
        (a[0] + b[0]) * 0.5,
        (a[1] + b[1]) * 0.5,
        (a[2] + b[2]) * 0.5,
    ];
    let radius = dist2_3(a, b).sqrt() * 0.5;
    Sphere3 { center, radius }
}

/// The circumscribed circle of the spatial triangle `a`, `b`, `c` (center lies
/// in the triangle plane), or `None` when the three points are (near)
/// collinear.
fn circumcircle_3(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> Option<Sphere3> {
    let ab = sub3(b, a);
    let ac = sub3(c, a);
    let d11 = dot3(ab, ab);
    let d12 = dot3(ab, ac);
    let d22 = dot3(ac, ac);
    let det = d11 * d22 - d12 * d12;
    if det.abs() <= DEGEN_REL * d11 * d22 + DEGEN_ABS {
        return None;
    }
    let r1 = d11 * 0.5;
    let r2 = d22 * 0.5;
    let alpha = (r1 * d22 - r2 * d12) / det;
    let beta = (d11 * r2 - d12 * r1) / det;
    let center = [
        a[0] + alpha * ab[0] + beta * ac[0],
        a[1] + alpha * ab[1] + beta * ac[1],
        a[2] + alpha * ab[2] + beta * ac[2],
    ];
    let radius = dist2_3(center, a).sqrt();
    Some(Sphere3 { center, radius })
}

/// The circumscribed sphere of the tetrahedron `a`, `b`, `c`, `d`, or `None`
/// when the four points are (near) coplanar.
fn circumsphere_4(a: [f32; 3], b: [f32; 3], c: [f32; 3], d: [f32; 3]) -> Option<Sphere3> {
    let ab = sub3(b, a);
    let ac = sub3(c, a);
    let ad = sub3(d, a);
    let rb = dot3(ab, ab) * 0.5;
    let rc = dot3(ac, ac) * 0.5;
    let rd = dot3(ad, ad) * 0.5;
    let m = [ab, ac, ad];
    let det = det3x3(m);
    let na = dot3(ab, ab).sqrt();
    let nb = dot3(ac, ac).sqrt();
    let nc = dot3(ad, ad).sqrt();
    if det.abs() <= DEGEN_REL * na * nb * nc + DEGEN_ABS {
        return None;
    }
    let ux = det3x3([[rb, ab[1], ab[2]], [rc, ac[1], ac[2]], [rd, ad[1], ad[2]]]) / det;
    let uy = det3x3([[ab[0], rb, ab[2]], [ac[0], rc, ac[2]], [ad[0], rd, ad[2]]]) / det;
    let uz = det3x3([[ab[0], ab[1], rb], [ac[0], ac[1], rc], [ad[0], ad[1], rd]]) / det;
    let center = [a[0] + ux, a[1] + uy, a[2] + uz];
    let radius = dist2_3(center, a).sqrt();
    Some(Sphere3 { center, radius })
}

// ----------------------------------------------------------------------------
// Exact minimum-enclosing ball of a tiny support set (robust to degeneracy).
// ----------------------------------------------------------------------------

/// Keeps `cand` in `best` when it is the first candidate or strictly smaller.
fn keep_smaller_2(best: &mut Option<Sphere2>, cand: Sphere2) {
    match *best {
        Some(cur) if cur.radius <= cand.radius => {}
        _ => *best = Some(cand),
    }
}

/// Keeps `cand` in `best` when it is the first candidate or strictly smaller.
fn keep_smaller_3(best: &mut Option<Sphere3>, cand: Sphere3) {
    match *best {
        Some(cur) if cur.radius <= cand.radius => {}
        _ => *best = Some(cand),
    }
}

/// Exact smallest enclosing circle of a support set of 1..=3 points, tolerant
/// of collinear or coincident inputs.
fn min_circle_of_set(s: &[[f32; 2]]) -> Sphere2 {
    if s.len() == 1 {
        return Sphere2 {
            center: s[0],
            radius: 0.0,
        };
    }
    let mut best: Option<Sphere2> = None;
    for (i, &pi) in s.iter().enumerate() {
        for &pj in &s[i + 1..] {
            let cand = circle_diameter_2(pi, pj);
            if s.iter().all(|&p| cand.contains(p)) {
                keep_smaller_2(&mut best, cand);
            }
        }
    }
    if s.len() == 3
        && let Some(cand) = circumcircle_2(s[0], s[1], s[2])
        && s.iter().all(|&p| cand.contains(p))
    {
        keep_smaller_2(&mut best, cand);
    }
    best.unwrap_or(Sphere2 {
        center: s[0],
        radius: 0.0,
    })
}

/// Exact smallest enclosing sphere of a support set of 1..=4 points, tolerant
/// of collinear or coplanar inputs.
fn min_sphere_of_set(s: &[[f32; 3]]) -> Sphere3 {
    if s.len() == 1 {
        return Sphere3 {
            center: s[0],
            radius: 0.0,
        };
    }
    let mut best: Option<Sphere3> = None;
    for (i, &pi) in s.iter().enumerate() {
        for &pj in &s[i + 1..] {
            let cand = sphere_diameter_3(pi, pj);
            if s.iter().all(|&p| cand.contains(p)) {
                keep_smaller_3(&mut best, cand);
            }
        }
    }
    if s.len() >= 3 {
        for (i, &pi) in s.iter().enumerate() {
            for (j, &pj) in s[i + 1..].iter().enumerate() {
                for &pk in &s[i + 1 + j + 1..] {
                    if let Some(cand) = circumcircle_3(pi, pj, pk)
                        && s.iter().all(|&p| cand.contains(p))
                    {
                        keep_smaller_3(&mut best, cand);
                    }
                }
            }
        }
    }
    if s.len() == 4
        && let Some(cand) = circumsphere_4(s[0], s[1], s[2], s[3])
        && s.iter().all(|&p| cand.contains(p))
    {
        keep_smaller_3(&mut best, cand);
    }
    best.unwrap_or(Sphere3 {
        center: s[0],
        radius: 0.0,
    })
}

// ----------------------------------------------------------------------------
// Deterministic shuffle (built-in LCG; no external randomness).
// ----------------------------------------------------------------------------

/// In-place Fisher-Yates shuffle driven by a fixed-seed linear congruential
/// generator. Shuffling only affects the expected running time of Welzl's
/// recursion; the returned ball is unique and therefore order independent.
fn lcg_shuffle_2(v: &mut [[f32; 2]]) {
    let mut state: u32 = 0x9E37_79B9;
    let n = v.len();
    for i in (1..n).rev() {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let j = ((state >> 8) as usize) % (i + 1);
        v.swap(i, j);
    }
}

/// In-place Fisher-Yates shuffle driven by a fixed-seed linear congruential
/// generator, for spatial points.
fn lcg_shuffle_3(v: &mut [[f32; 3]]) {
    let mut state: u32 = 0x9E37_79B9;
    let n = v.len();
    for i in (1..n).rev() {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let j = ((state >> 8) as usize) % (i + 1);
        v.swap(i, j);
    }
}

// ----------------------------------------------------------------------------
// Public entry points: Welzl's algorithm, move-to-front form.
// ----------------------------------------------------------------------------

/// Returns the exact smallest enclosing circle of `points`, or `None` when the
/// slice is empty.
///
/// A single point yields a radius-`0.0` circle centered on it; fully coincident
/// points collapse to the same radius-`0.0` circle. Every input point is
/// guaranteed to satisfy [`Sphere2::contains`] on the result, and the 1, 2, or
/// 3 support points rest exactly on the boundary.
#[must_use]
pub fn min_enclosing_circle(points: &[[f32; 2]]) -> Option<Sphere2> {
    if points.is_empty() {
        return None;
    }
    let mut pts: Vec<[f32; 2]> = points.to_vec();
    lcg_shuffle_2(&mut pts);
    let n = pts.len();
    let mut ball = min_circle_of_set(&[pts[0]]);
    for i in 1..n {
        let pi = pts[i];
        if !ball.contains(pi) {
            ball = min_circle_of_set(&[pi]);
            for (j, &pj) in pts[..i].iter().enumerate() {
                if !ball.contains(pj) {
                    ball = min_circle_of_set(&[pi, pj]);
                    for &pk in &pts[..j] {
                        if !ball.contains(pk) {
                            ball = min_circle_of_set(&[pi, pj, pk]);
                        }
                    }
                }
            }
        }
    }
    Some(ball)
}

/// Returns the exact smallest enclosing sphere of `points`, or `None` when the
/// slice is empty.
///
/// A single point yields a radius-`0.0` sphere centered on it; fully coincident
/// points collapse to the same radius-`0.0` sphere. Every input point is
/// guaranteed to satisfy [`Sphere3::contains`] on the result, and the 1, 2, 3,
/// or 4 support points rest exactly on the boundary.
#[must_use]
pub fn min_enclosing_sphere(points: &[[f32; 3]]) -> Option<Sphere3> {
    if points.is_empty() {
        return None;
    }
    let mut pts: Vec<[f32; 3]> = points.to_vec();
    lcg_shuffle_3(&mut pts);
    let n = pts.len();
    let mut ball = min_sphere_of_set(&[pts[0]]);
    for i in 1..n {
        let pi = pts[i];
        if !ball.contains(pi) {
            ball = min_sphere_of_set(&[pi]);
            for (j, &pj) in pts[..i].iter().enumerate() {
                if !ball.contains(pj) {
                    ball = min_sphere_of_set(&[pi, pj]);
                    for (k, &pk) in pts[..j].iter().enumerate() {
                        if !ball.contains(pk) {
                            ball = min_sphere_of_set(&[pi, pj, pk]);
                            for &pl in &pts[..k] {
                                if !ball.contains(pl) {
                                    ball = min_sphere_of_set(&[pi, pj, pk, pl]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Some(ball)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Loose absolute tolerance for geometric assertions in tests.
    const TEST_EPS: f32 = 1.0e-3;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= TEST_EPS * (a.abs() + b.abs() + 1.0)
    }

    fn approx2(a: [f32; 2], b: [f32; 2]) -> bool {
        approx(a[0], b[0]) && approx(a[1], b[1])
    }

    fn approx3(a: [f32; 3], b: [f32; 3]) -> bool {
        approx(a[0], b[0]) && approx(a[1], b[1]) && approx(a[2], b[2])
    }

    /// Naive minimum enclosing circle: try every pair-diameter and triple
    /// circumcircle, keep the smallest that encloses all points.
    fn brute_circle(points: &[[f32; 2]]) -> Sphere2 {
        let n = points.len();
        let mut best: Option<Sphere2> = None;
        if n == 1 {
            return Sphere2 {
                center: points[0],
                radius: 0.0,
            };
        }
        for i in 0..n {
            for j in (i + 1)..n {
                let c = circle_diameter_2(points[i], points[j]);
                if points.iter().all(|&p| c.contains(p)) {
                    keep_smaller_2(&mut best, c);
                }
            }
        }
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    #[expect(
                        clippy::collapsible_if,
                        reason = "paired geometric predicate and containment test read clearer kept nested in this brute-force reference"
                    )]
                    if let Some(c) = circumcircle_2(points[i], points[j], points[k]) {
                        if points.iter().all(|&p| c.contains(p)) {
                            keep_smaller_2(&mut best, c);
                        }
                    }
                }
            }
        }
        best.expect("non-empty")
    }

    /// Naive minimum enclosing sphere over pair/triple/quad supports.
    fn brute_sphere(points: &[[f32; 3]]) -> Sphere3 {
        let n = points.len();
        let mut best: Option<Sphere3> = None;
        if n == 1 {
            return Sphere3 {
                center: points[0],
                radius: 0.0,
            };
        }
        for i in 0..n {
            for j in (i + 1)..n {
                let c = sphere_diameter_3(points[i], points[j]);
                if points.iter().all(|&p| c.contains(p)) {
                    keep_smaller_3(&mut best, c);
                }
            }
        }
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    #[expect(
                        clippy::collapsible_if,
                        reason = "paired geometric predicate and containment test read clearer kept nested in this brute-force reference"
                    )]
                    if let Some(c) = circumcircle_3(points[i], points[j], points[k]) {
                        if points.iter().all(|&p| c.contains(p)) {
                            keep_smaller_3(&mut best, c);
                        }
                    }
                }
            }
        }
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    for l in (k + 1)..n {
                        #[expect(
                            clippy::collapsible_if,
                            reason = "paired geometric predicate and containment test read clearer kept nested in this brute-force reference"
                        )]
                        if let Some(c) = circumsphere_4(points[i], points[j], points[k], points[l])
                        {
                            if points.iter().all(|&p| c.contains(p)) {
                                keep_smaller_3(&mut best, c);
                            }
                        }
                    }
                }
            }
        }
        best.expect("non-empty")
    }

    struct Lcg {
        state: u32,
    }

    impl Lcg {
        fn new(seed: u32) -> Self {
            Self { state: seed }
        }

        fn next_unit(&mut self) -> f32 {
            self.state = self
                .state
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223);
            (self.state >> 8) as f32 / (1_u32 << 24) as f32
        }

        fn next_range(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * self.next_unit()
        }
    }

    // ---- 2D ----------------------------------------------------------------

    #[test]
    fn circle_empty_returns_none() {
        let pts: [[f32; 2]; 0] = [];
        assert!(min_enclosing_circle(&pts).is_none());
    }

    #[test]
    fn circle_single_point_zero_radius() {
        let p = [3.0, -2.0];
        let s = min_enclosing_circle(&[p]).expect("non-empty");
        assert!(approx2(s.center, p));
        assert!(s.radius <= TEST_EPS);
    }

    #[test]
    fn circle_two_points_diameter() {
        let a = [-1.0, 0.0];
        let b = [3.0, 0.0];
        let s = min_enclosing_circle(&[a, b]).expect("non-empty");
        assert!(approx2(s.center, [1.0, 0.0]));
        assert!(approx(s.radius, 2.0));
    }

    #[test]
    fn circle_coincident_points_zero_radius() {
        let p = [1.5, -4.0];
        let s = min_enclosing_circle(&[p, p, p, p]).expect("non-empty");
        assert!(approx2(s.center, p));
        assert!(s.radius <= TEST_EPS);
    }

    #[test]
    fn circle_collinear_points_use_extremes() {
        let pts = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [5.0, 0.0], [3.0, 0.0]];
        let s = min_enclosing_circle(&pts).expect("non-empty");
        assert!(approx2(s.center, [2.5, 0.0]));
        assert!(approx(s.radius, 2.5));
        for &p in &pts {
            assert!(s.contains(p));
        }
    }

    #[test]
    fn circle_unit_square_exact() {
        let pts = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let s = min_enclosing_circle(&pts).expect("non-empty");
        assert!(approx2(s.center, [0.5, 0.5]));
        assert!(approx(s.radius, 0.5_f32.sqrt()));
    }

    #[test]
    fn circle_equilateral_circumcircle() {
        let h = 3.0_f32.sqrt() * 0.5;
        let pts = [[0.0, 0.0], [1.0, 0.0], [0.5, h]];
        let s = min_enclosing_circle(&pts).expect("non-empty");
        let expected_r = 1.0 / 3.0_f32.sqrt();
        assert!(approx(s.radius, expected_r));
        assert!(approx2(s.center, [0.5, 3.0_f32.sqrt() / 6.0]));
    }

    #[test]
    fn circle_right_triangle_hypotenuse_diameter() {
        let pts = [[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]];
        let s = min_enclosing_circle(&pts).expect("non-empty");
        assert!(approx2(s.center, [2.0, 1.5]));
        assert!(approx(s.radius, 2.5));
    }

    #[test]
    fn circle_obtuse_triangle_longest_edge() {
        let pts = [[0.0, 0.0], [4.0, 0.0], [1.0, 0.5]];
        let s = min_enclosing_circle(&pts).expect("non-empty");
        assert!(approx2(s.center, [2.0, 0.0]));
        assert!(approx(s.radius, 2.0));
    }

    #[test]
    fn circle_points_on_known_circle() {
        // All points lie on the circle centered at (3, -2) with radius 5.
        let pts = [
            [8.0, -2.0],
            [-2.0, -2.0],
            [3.0, 3.0],
            [3.0, -7.0],
            [6.0, 2.0],
            [0.0, -6.0],
            [7.0, 1.0],
        ];
        let s = min_enclosing_circle(&pts).expect("non-empty");
        assert!(approx2(s.center, [3.0, -2.0]));
        assert!(approx(s.radius, 5.0));
    }

    #[test]
    fn circle_all_points_contained_random() {
        let mut rng = Lcg::new(0x1234_5678);
        for _ in 0..40 {
            let n = 3 + (rng.state as usize % 8);
            let mut pts = Vec::new();
            for _ in 0..n {
                pts.push([rng.next_range(-10.0, 10.0), rng.next_range(-10.0, 10.0)]);
            }
            let s = min_enclosing_circle(&pts).expect("non-empty");
            for &p in &pts {
                assert!(s.contains(p), "point {p:?} escaped {s:?}");
            }
        }
    }

    #[test]
    fn circle_matches_brute_force_random() {
        let mut rng = Lcg::new(0xDEAD_BEEF);
        for _ in 0..40 {
            let n = 3 + (rng.state as usize % 6);
            let mut pts = Vec::new();
            for _ in 0..n {
                pts.push([rng.next_range(-6.0, 6.0), rng.next_range(-6.0, 6.0)]);
            }
            let got = min_enclosing_circle(&pts).expect("non-empty");
            let want = brute_circle(&pts);
            assert!(
                approx(got.radius, want.radius),
                "radius {} vs {}",
                got.radius,
                want.radius
            );
            assert!(approx2(got.center, want.center));
        }
    }

    #[test]
    fn circle_translation_invariant() {
        let pts = [[0.0, 0.0], [2.0, 1.0], [1.0, 3.0], [-1.0, 1.0]];
        let base = min_enclosing_circle(&pts).expect("non-empty");
        let shift = [10.0, -7.0];
        let moved: Vec<[f32; 2]> = pts
            .iter()
            .map(|p| [p[0] + shift[0], p[1] + shift[1]])
            .collect();
        let s = min_enclosing_circle(&moved).expect("non-empty");
        assert!(approx(s.radius, base.radius));
        assert!(approx2(
            s.center,
            [base.center[0] + shift[0], base.center[1] + shift[1]]
        ));
    }

    #[test]
    fn circle_scale_invariant() {
        let pts = [[0.0, 0.0], [2.0, 1.0], [1.0, 3.0], [-1.0, 1.0]];
        let base = min_enclosing_circle(&pts).expect("non-empty");
        let k = 3.5_f32;
        let scaled: Vec<[f32; 2]> = pts.iter().map(|p| [p[0] * k, p[1] * k]).collect();
        let s = min_enclosing_circle(&scaled).expect("non-empty");
        assert!(approx(s.radius, base.radius * k));
        assert!(approx2(s.center, [base.center[0] * k, base.center[1] * k]));
    }

    #[test]
    fn circle_permutation_invariant() {
        let pts = [[0.0, 0.0], [4.0, 0.0], [0.0, 3.0], [1.0, 1.0], [2.0, 2.0]];
        let a = min_enclosing_circle(&pts).expect("non-empty");
        let rev: Vec<[f32; 2]> = pts.iter().rev().copied().collect();
        let b = min_enclosing_circle(&rev).expect("non-empty");
        assert!(approx(a.radius, b.radius));
        assert!(approx2(a.center, b.center));
    }

    #[test]
    fn circle_boundary_points_on_surface() {
        let pts = [[0.0, 0.0], [4.0, 0.0], [2.0, 5.0]];
        let s = min_enclosing_circle(&pts).expect("non-empty");
        // Every vertex of an acute triangle sits on the circumcircle boundary.
        for &p in &pts {
            let d = dist2_2(p, s.center).sqrt();
            assert!(approx(d, s.radius), "vertex {p:?} off boundary");
        }
    }

    // ---- 3D ----------------------------------------------------------------

    #[test]
    fn sphere_empty_returns_none() {
        let pts: [[f32; 3]; 0] = [];
        assert!(min_enclosing_sphere(&pts).is_none());
    }

    #[test]
    fn sphere_single_point_zero_radius() {
        let p = [3.0, -2.0, 7.0];
        let s = min_enclosing_sphere(&[p]).expect("non-empty");
        assert!(approx3(s.center, p));
        assert!(s.radius <= TEST_EPS);
    }

    #[test]
    fn sphere_two_points_diameter() {
        let a = [-1.0, 0.0, 0.0];
        let b = [3.0, 0.0, 0.0];
        let s = min_enclosing_sphere(&[a, b]).expect("non-empty");
        assert!(approx3(s.center, [1.0, 0.0, 0.0]));
        assert!(approx(s.radius, 2.0));
    }

    #[test]
    fn sphere_coincident_points_zero_radius() {
        let p = [1.0, 1.0, 1.0];
        let s = min_enclosing_sphere(&[p, p, p, p, p]).expect("non-empty");
        assert!(approx3(s.center, p));
        assert!(s.radius <= TEST_EPS);
    }

    #[test]
    fn sphere_collinear_points_use_extremes() {
        let pts = [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 6.0, 0.0],
        ];
        let s = min_enclosing_sphere(&pts).expect("non-empty");
        assert!(approx3(s.center, [0.0, 3.0, 0.0]));
        assert!(approx(s.radius, 3.0));
    }

    #[test]
    fn sphere_coplanar_square_matches_2d() {
        let pts = [
            [0.0, 0.0, 5.0],
            [1.0, 0.0, 5.0],
            [1.0, 1.0, 5.0],
            [0.0, 1.0, 5.0],
        ];
        let s = min_enclosing_sphere(&pts).expect("non-empty");
        assert!(approx3(s.center, [0.5, 0.5, 5.0]));
        assert!(approx(s.radius, 0.5_f32.sqrt()));
    }

    #[test]
    fn sphere_regular_tetrahedron_exact() {
        let pts = [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ];
        let s = min_enclosing_sphere(&pts).expect("non-empty");
        assert!(approx3(s.center, [0.0, 0.0, 0.0]));
        assert!(approx(s.radius, 3.0_f32.sqrt()));
    }

    #[test]
    fn sphere_coplanar_triangle_circumcircle() {
        let pts = [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [2.0, 4.0, 0.0]];
        let s = min_enclosing_sphere(&pts).expect("non-empty");
        // Center stays in the z = 0 plane; every vertex on the boundary.
        assert!(approx(s.center[2], 0.0));
        for &p in &pts {
            let d = dist2_3(p, s.center).sqrt();
            assert!(approx(d, s.radius));
        }
    }

    #[test]
    fn sphere_points_on_known_sphere() {
        let c = [1.0, 2.0, 3.0];
        let r = 2.0_f32;
        let pts = [
            [c[0] + r, c[1], c[2]],
            [c[0] - r, c[1], c[2]],
            [c[0], c[1] + r, c[2]],
            [c[0], c[1] - r, c[2]],
            [c[0], c[1], c[2] + r],
            [c[0], c[1], c[2] - r],
        ];
        let s = min_enclosing_sphere(&pts).expect("non-empty");
        assert!(approx3(s.center, c));
        assert!(approx(s.radius, r));
    }

    #[test]
    fn sphere_all_points_contained_random() {
        let mut rng = Lcg::new(0x0BAD_F00D);
        for _ in 0..40 {
            let n = 4 + (rng.state as usize % 8);
            let mut pts = Vec::new();
            for _ in 0..n {
                pts.push([
                    rng.next_range(-8.0, 8.0),
                    rng.next_range(-8.0, 8.0),
                    rng.next_range(-8.0, 8.0),
                ]);
            }
            let s = min_enclosing_sphere(&pts).expect("non-empty");
            for &p in &pts {
                assert!(s.contains(p), "point {p:?} escaped {s:?}");
            }
        }
    }

    #[test]
    fn sphere_matches_brute_force_random() {
        let mut rng = Lcg::new(0xCAFE_1234);
        for _ in 0..30 {
            let n = 4 + (rng.state as usize % 5);
            let mut pts = Vec::new();
            for _ in 0..n {
                pts.push([
                    rng.next_range(-5.0, 5.0),
                    rng.next_range(-5.0, 5.0),
                    rng.next_range(-5.0, 5.0),
                ]);
            }
            let got = min_enclosing_sphere(&pts).expect("non-empty");
            let want = brute_sphere(&pts);
            assert!(
                approx(got.radius, want.radius),
                "radius {} vs {}",
                got.radius,
                want.radius
            );
            assert!(approx3(got.center, want.center));
        }
    }

    #[test]
    fn sphere_translation_invariant() {
        let pts = [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 4.0],
        ];
        let base = min_enclosing_sphere(&pts).expect("non-empty");
        let shift = [-5.0, 6.0, 7.0];
        let moved: Vec<[f32; 3]> = pts
            .iter()
            .map(|p| [p[0] + shift[0], p[1] + shift[1], p[2] + shift[2]])
            .collect();
        let s = min_enclosing_sphere(&moved).expect("non-empty");
        assert!(approx(s.radius, base.radius));
        assert!(approx3(
            s.center,
            [
                base.center[0] + shift[0],
                base.center[1] + shift[1],
                base.center[2] + shift[2],
            ]
        ));
    }

    #[test]
    fn sphere_scale_invariant() {
        let pts = [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 4.0],
            [1.0, 1.0, 1.0],
        ];
        let base = min_enclosing_sphere(&pts).expect("non-empty");
        let k = 2.5_f32;
        let scaled: Vec<[f32; 3]> = pts.iter().map(|p| [p[0] * k, p[1] * k, p[2] * k]).collect();
        let s = min_enclosing_sphere(&scaled).expect("non-empty");
        assert!(approx(s.radius, base.radius * k));
        assert!(approx3(
            s.center,
            [base.center[0] * k, base.center[1] * k, base.center[2] * k]
        ));
    }

    #[test]
    fn sphere_permutation_invariant() {
        let pts = [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 5.0],
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 2.0],
        ];
        let a = min_enclosing_sphere(&pts).expect("non-empty");
        let rev: Vec<[f32; 3]> = pts.iter().rev().copied().collect();
        let b = min_enclosing_sphere(&rev).expect("non-empty");
        assert!(approx(a.radius, b.radius));
        assert!(approx3(a.center, b.center));
    }

    #[test]
    fn sphere_boundary_points_on_surface() {
        let pts = [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [2.0, 4.0, 0.0],
            [2.0, 1.0, 4.0],
        ];
        let s = min_enclosing_sphere(&pts).expect("non-empty");
        for &p in &pts {
            let d = dist2_3(p, s.center).sqrt();
            assert!(approx(d, s.radius), "vertex {p:?} off boundary");
        }
    }
}
