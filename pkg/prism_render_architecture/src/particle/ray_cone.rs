//! Analytic `ray`-cone intersection for the particle subsystem's picking,
//! collision-probe, and analytic-primitive raytrace contracts (design
//! sections 10 and 14).
//!
//! This module owns the *closed-form* solution of the `ray`-against-a-cone
//! problem, where the cone is described by an apex point, a unit axis that
//! points toward the opening direction, and the **cosine of its half-angle**
//! (`cos_half_angle`, with `0 < cos < 1`). The infinite variant
//! ([`ray_infinite_cone`]) solves the scalar quadratic obtained by substituting
//! the ray `origin + t * dir` into the cone identity
//! `dot(v, axis)^2 = cos^2 * dot(v, v)` (with `v = origin + t * dir - apex`),
//! and the finite variant ([`ray_finite_cone`]) additionally clips the lateral
//! surface to the axial band `0 <= proj <= height` and closes the wide end with
//! a flat base cap.
//!
//! # Strict scope
//! This is the *quadric cone* kernel and nothing else. It is the analytic
//! sibling of the other single-`ray`-single-primitive contracts and is
//! deliberately distinct from each of them:
//! * [`ray_cylinder`](crate::particle::ray_cylinder) sweeps an **equal-radius**
//!   tube: its side wall is a fixed-radius quadric, whereas a cone's radius
//!   grows linearly with axial distance from the apex.
//! * [`ray_sphere`](crate::particle::ray_sphere) solves a quadratic against a
//!   full spherical surface, which has no axis and no nappe to cull.
//! * [`ray_disk`](crate::particle::ray_disk) is a flat bounded circle in a
//!   plane; this module reuses that same disk idea *only* for the finite cone's
//!   base cap, while its body is the genuine second-degree cone surface.
//!
//! # No transcendental math and no reflected nappe
//! The half-angle enters exclusively as its cosine (and, for the base-cap
//! radius, its sine recovered as `sqrt(1 - cos^2)`); the caller is responsible
//! for any angle-to-trigonometry conversion, so this file never calls `sin`,
//! `cos`, `tan`, `acos`, or `atan`. Every routine uses only `+ - * /`,
//! `f32::sqrt`, `f32::abs`, `f32::min`, and `f32::max`, and no exact `==` /
//! `!=` is ever written on a production `f32`: divisors and the discriminant are
//! classified against an explicit epsilon so a degenerate direction, a
//! near-linear quadratic, or a grazing (double-root) ray reports a clean miss or
//! single hit instead of a `NaN`. Crucially, the algebraic cone is a *double*
//! cone (two nappes meeting at the apex), so every candidate root is tested
//! against `dot(hit - apex, axis) >= 0` and any solution that lands on the
//! **reflected backward nappe** is discarded. This keeps the `CPU` reference
//! here in agreement with a future `GPU` kernel that packs the same cone.

/// Epsilon used to guard divisions, to classify the quadratic discriminant, and
/// to compare quantities against zero without ever writing an exact `==` / `!=`
/// on a production `f32`.
const CMP_EPS: f32 = 1.0e-6;

/// A single forward ray-cone surface hit: the ray parameter `t` (with
/// `t >= 0`) and the world-space intersection `point`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayConeHit {
    /// Ray parameter of the hit, measured in units of `dir`'s length.
    pub t: f32,
    /// World-space position of the hit, equal to `origin + t * dir`.
    pub point: [f32; 3],
}

/// Component-wise difference `a - b`.
#[must_use]
fn v_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Component-wise sum `a + b`.
#[must_use]
fn v_add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Scales `a` by the scalar `s`.
#[must_use]
fn v_scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Euclidean dot product `a . b`.
#[must_use]
fn v_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Evaluates the ray at parameter `t`, returning `origin + t * dir`.
#[must_use]
fn point_at(origin: [f32; 3], dir: [f32; 3], t: f32) -> [f32; 3] {
    v_add(origin, v_scale(dir, t))
}

/// Solves `a t^2 + b t + c = 0` for real roots, degrading gracefully to the
/// linear case when `a` is negligible and to a single (double) root when the
/// discriminant is negligible.
///
/// Returns the roots in the first `count` lanes of the array; `count` is `0`,
/// `1`, or `2`. Roots are not sorted; callers scan them for the best candidate.
#[must_use]
fn solve_quadratic(a: f32, b: f32, c: f32) -> ([f32; 2], usize) {
    if a.abs() < CMP_EPS {
        // Near-linear: b t + c = 0.
        if b.abs() < CMP_EPS {
            return ([0.0; 2], 0);
        }
        return ([-c / b, 0.0], 1);
    }
    let disc = b * b - 4.0 * a * c;
    if disc < -CMP_EPS {
        return ([0.0; 2], 0);
    }
    let sqrt_disc = disc.max(0.0).sqrt();
    if sqrt_disc < CMP_EPS {
        // Grazing: a single double root.
        return ([-b / (2.0 * a), 0.0], 1);
    }
    let inv = 0.5 / a;
    let t0 = (-b - sqrt_disc) * inv;
    let t1 = (-b + sqrt_disc) * inv;
    ([t0, t1], 2)
}

/// Rejects a half-angle cosine that is not strictly inside `(0, 1)`.
#[must_use]
fn cos_out_of_range(cos_half_angle: f32) -> bool {
    if cos_half_angle <= CMP_EPS {
        return true;
    }
    cos_half_angle >= 1.0
}

/// Builds the cone quadratic coefficients `(a, b, c)` and the axial scalars
/// `(cd, dd)` shared by both the infinite and finite solvers.
///
/// Here `cd = dot(origin - apex, axis)` and `dd = dot(dir, axis)`, so the axial
/// coordinate of a hit at parameter `t` is simply `cd + t * dd`.
#[must_use]
fn cone_coefficients(
    origin: [f32; 3],
    dir: [f32; 3],
    apex: [f32; 3],
    axis: [f32; 3],
    cos_half_angle: f32,
) -> (f32, f32, f32, f32, f32) {
    let co = v_sub(origin, apex);
    let dd = v_dot(dir, axis);
    let cd = v_dot(co, axis);
    let dirdir = v_dot(dir, dir);
    let codir = v_dot(co, dir);
    let coco = v_dot(co, co);
    let k2 = cos_half_angle * cos_half_angle;
    let a = dd * dd - k2 * dirdir;
    let b = 2.0 * (cd * dd - k2 * codir);
    let c = cd * cd - k2 * coco;
    (a, b, c, cd, dd)
}

/// Keeps `candidate` if it improves on `best` (smaller non-negative `t`).
#[must_use]
fn keep_min(best: Option<f32>, candidate: f32) -> Option<f32> {
    Some(best.map_or(candidate, |b| b.min(candidate)))
}

/// Intersects a ray with an **infinite** cone.
///
/// The cone has its apex at `apex`, opens along the unit vector `axis`, and has
/// a half-angle whose cosine is `cos_half_angle` (which must lie strictly inside
/// `(0, 1)`). Returns the nearest forward hit (smallest `t >= 0`) on the forward
/// nappe; solutions on the reflected backward nappe are discarded. A degenerate
/// direction or an out-of-range cosine yields `None`.
#[must_use]
pub fn ray_infinite_cone(
    origin: [f32; 3],
    dir: [f32; 3],
    apex: [f32; 3],
    axis: [f32; 3],
    cos_half_angle: f32,
) -> Option<RayConeHit> {
    if cos_out_of_range(cos_half_angle) {
        return None;
    }
    if v_dot(dir, dir) < CMP_EPS {
        return None;
    }
    let (a, b, c, cd, dd) = cone_coefficients(origin, dir, apex, axis, cos_half_angle);
    let (roots, count) = solve_quadratic(a, b, c);
    let mut best: Option<f32> = None;
    for &t in &roots[..count] {
        if t < 0.0 {
            continue;
        }
        // Cull the reflected backward nappe: keep dot(hit - apex, axis) >= 0.
        if cd + t * dd < -CMP_EPS {
            continue;
        }
        best = keep_min(best, t);
    }
    best.map(|t| RayConeHit {
        t,
        point: point_at(origin, dir, t),
    })
}

/// Intersects a ray with a **finite** capped cone.
///
/// This is [`ray_infinite_cone`] restricted to the axial band
/// `0 <= dot(hit - apex, axis) <= height` and closed at the wide end by a flat
/// base cap. The base cap is a disk centered at `apex + height * axis`, oriented
/// by `axis`, whose radius is `height * sin / cos` with `sin = sqrt(1 - cos^2)`
/// (no `tan` is called). Returns the nearest forward hit across the lateral
/// surface and the base cap, or `None` for a degenerate direction, a
/// non-positive `height`, or an out-of-range cosine.
#[must_use]
pub fn ray_finite_cone(
    origin: [f32; 3],
    dir: [f32; 3],
    apex: [f32; 3],
    axis: [f32; 3],
    cos_half_angle: f32,
    height: f32,
) -> Option<RayConeHit> {
    if cos_out_of_range(cos_half_angle) {
        return None;
    }
    if height <= CMP_EPS {
        return None;
    }
    if v_dot(dir, dir) < CMP_EPS {
        return None;
    }
    let (a, b, c, cd, dd) = cone_coefficients(origin, dir, apex, axis, cos_half_angle);
    let (roots, count) = solve_quadratic(a, b, c);
    let mut best: Option<f32> = None;

    // Lateral surface, clipped to the axial band [0, height].
    for &t in &roots[..count] {
        if t < 0.0 {
            continue;
        }
        let axial = cd + t * dd;
        if axial < -CMP_EPS {
            continue;
        }
        if axial > height + CMP_EPS {
            continue;
        }
        best = keep_min(best, t);
    }

    // Base cap: the flat disk that closes the wide end.
    if dd.abs() > CMP_EPS {
        let t = (height - cd) / dd;
        if t >= 0.0 {
            let hit = point_at(origin, dir, t);
            let center = v_add(apex, v_scale(axis, height));
            let radial = v_sub(hit, center);
            let k2 = cos_half_angle * cos_half_angle;
            let sin2 = 1.0 - k2;
            let radius2 = height * height * sin2 / k2;
            if v_dot(radial, radial) <= radius2 + CMP_EPS {
                best = keep_min(best, t);
            }
        }
    }

    best.map(|t| RayConeHit {
        t,
        point: point_at(origin, dir, t),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1.0e-3;

    // Cosine of 45 degrees, a convenient half-angle where radius equals axial
    // distance. Kept as a literal so the module stays trig-free.
    const COS_45: f32 = 0.707_106_77;

    fn approx(a: f32, b: f32) -> bool {
        // Absolute floor absorbs the sqrt-scale error of a grazing double root
        // in f32; the relative term keeps large-magnitude hits meaningful.
        (a - b).abs() <= 4.0e-3 + TOL * (a.abs() + b.abs())
    }

    fn approx_pt(a: [f32; 3], b: [f32; 3]) -> bool {
        approx(a[0], b[0]) && approx(a[1], b[1]) && approx(a[2], b[2])
    }

    fn normalize(v: [f32; 3]) -> [f32; 3] {
        let len = v_dot(v, v).sqrt();
        v_scale(v, 1.0 / len)
    }

    // Residual of the infinite-cone identity dot(v,axis)^2 - cos^2 dot(v,v).
    fn cone_residual(hit: [f32; 3], apex: [f32; 3], axis: [f32; 3], cos: f32) -> f32 {
        let v = v_sub(hit, apex);
        let vd = v_dot(v, axis);
        vd * vd - cos * cos * v_dot(v, v)
    }

    // Small linear-congruential generator for reproducible pseudo-random tests.
    struct Lcg(u64);
    impl Lcg {
        fn next_u16(&mut self) -> u16 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // Take high-quality upper state bits as a uniform 16-bit sample.
            (self.0 >> 40) as u16
        }
        fn unit(&mut self) -> f32 {
            f32::from(self.next_u16()) / f32::from(u16::MAX)
        }
        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * self.unit()
        }
    }

    #[test]
    fn axis_ray_hits_apex() {
        let hit = ray_infinite_cone(
            [0.0, 0.0, -5.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            COS_45,
        )
        .expect("axis ray should hit apex");
        assert!(approx(hit.t, 5.0));
        assert!(approx_pt(hit.point, [0.0, 0.0, 0.0]));
    }

    #[test]
    fn perpendicular_ray_takes_nearest_of_two() {
        let hit = ray_infinite_cone(
            [-10.0, 0.0, 2.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            COS_45,
        )
        .expect("perpendicular ray should hit");
        assert!(approx(hit.t, 8.0));
        assert!(approx_pt(hit.point, [-2.0, 0.0, 2.0]));
    }

    #[test]
    fn ray_pointing_away_misses() {
        let miss = ray_infinite_cone(
            [20.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            COS_45,
        );
        assert!(miss.is_none());
    }

    #[test]
    fn backward_nappe_is_culled() {
        // A horizontal ray below the apex would cross the reflected nappe; the
        // forward nappe is not crossed, so the result must be a miss.
        let miss = ray_infinite_cone(
            [-10.0, 0.0, -2.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            COS_45,
        );
        assert!(miss.is_none());
    }

    #[test]
    fn backward_nappe_hit_would_exist_without_cull() {
        // Sanity: the algebraic quadratic does have roots here (they land on the
        // backward nappe), confirming the miss above is due to culling.
        let (a, b, c, _, _) = cone_coefficients(
            [-10.0, 0.0, -2.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            COS_45,
        );
        let (_, count) = solve_quadratic(a, b, c);
        assert_eq!(count, 2);
    }

    #[test]
    fn origin_inside_forward_exit() {
        let hit = ray_infinite_cone(
            [0.0, 0.0, 3.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            COS_45,
        )
        .expect("ray from inside should exit through the surface");
        assert!(approx(hit.t, 3.0));
        assert!(approx_pt(hit.point, [3.0, 0.0, 3.0]));
    }

    #[test]
    fn parallel_to_axis_from_outside_hits_apex() {
        // dir is parallel to the axis; a = sin^2 > 0 and the discriminant is
        // exactly zero, so this exercises the grazing single-root branch too.
        let hit = ray_infinite_cone(
            [0.0, 0.0, -5.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            COS_45,
        )
        .expect("parallel ray should graze the apex");
        assert!(approx(hit.t, 5.0));
        assert!(approx_pt(hit.point, [0.0, 0.0, 0.0]));
    }

    #[test]
    fn parallel_to_axis_from_inside_no_forward_hit() {
        let miss = ray_infinite_cone(
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            COS_45,
        );
        assert!(miss.is_none());
    }

    #[test]
    fn grazing_double_root_is_single_hit() {
        let (roots, count) = solve_quadratic(1.0, -4.0, 4.0);
        assert_eq!(count, 1);
        assert!(approx(roots[0], 2.0));
    }

    #[test]
    fn degenerate_linear_a_near_zero() {
        // dir points along a cone generator, so a = dd^2 - cos^2 = 0.
        let dir = normalize([1.0, 0.0, 1.0]);
        let hit = ray_infinite_cone(
            [-3.0, 0.0, 0.0],
            dir,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            COS_45,
        )
        .expect("generator-parallel ray should hit the far side once");
        assert!(approx_pt(hit.point, [-1.5, 0.0, 1.5]));
        assert!(cone_residual(hit.point, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], COS_45).abs() <= TOL);
    }

    #[test]
    fn solve_quadratic_linear_branch() {
        // 0 t^2 + 2 t - 4 = 0 -> t = 2.
        let (roots, count) = solve_quadratic(0.0, 2.0, -4.0);
        assert_eq!(count, 1);
        assert!(approx(roots[0], 2.0));
    }

    #[test]
    fn solve_quadratic_no_real_roots() {
        let (_, count) = solve_quadratic(1.0, 0.0, 4.0);
        assert_eq!(count, 0);
    }

    #[test]
    fn lateral_hit_satisfies_cone_identity() {
        let apex = [0.0, 0.0, 0.0];
        let axis = [0.0, 0.0, 1.0];
        let hit =
            ray_infinite_cone([-10.0, 0.0, 2.0], [1.0, 0.0, 0.0], apex, axis, COS_45).expect("hit");
        assert!(cone_residual(hit.point, apex, axis, COS_45).abs() <= TOL);
    }

    #[test]
    fn invalid_cos_zero_returns_none() {
        assert!(ray_infinite_cone(
            [0.0, 0.0, -5.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            0.0,
        )
        .is_none());
    }

    #[test]
    fn invalid_cos_one_returns_none() {
        assert!(ray_infinite_cone(
            [0.0, 0.0, -5.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            1.0,
        )
        .is_none());
    }

    #[test]
    fn zero_direction_returns_none() {
        assert!(ray_infinite_cone(
            [0.0, 0.0, -5.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            COS_45,
        )
        .is_none());
    }

    #[test]
    fn finite_zero_height_returns_none() {
        assert!(ray_finite_cone(
            [-10.0, 0.0, 2.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            COS_45,
            0.0,
        )
        .is_none());
    }

    #[test]
    fn finite_height_clip_rejects_far_lateral() {
        // Lateral hit is at axial = 5, beyond height = 2, and no cap is crossed.
        let miss = ray_finite_cone(
            [-10.0, 0.0, 5.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            COS_45,
            2.0,
        );
        assert!(miss.is_none());
    }

    #[test]
    fn finite_lateral_within_height_hits() {
        let hit = ray_finite_cone(
            [-10.0, 0.0, 2.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            COS_45,
            5.0,
        )
        .expect("lateral hit within the band");
        assert!(approx(hit.t, 8.0));
        assert!(approx_pt(hit.point, [-2.0, 0.0, 2.0]));
    }

    #[test]
    fn finite_base_cap_hit() {
        // Descending along -z, the cap at z = 2 (t = 3) is nearer than the
        // lateral surface (t = 4.5).
        let hit = ray_finite_cone(
            [0.5, 0.0, 5.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            COS_45,
            2.0,
        )
        .expect("base cap hit");
        assert!(approx(hit.t, 3.0));
        assert!(approx_pt(hit.point, [0.5, 0.0, 2.0]));
    }

    #[test]
    fn finite_base_cap_center_axial_hit() {
        let hit = ray_finite_cone(
            [0.0, 0.0, 5.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            COS_45,
            2.0,
        )
        .expect("center of base cap");
        assert!(approx(hit.t, 3.0));
        assert!(approx_pt(hit.point, [0.0, 0.0, 2.0]));
    }

    #[test]
    fn finite_ray_missing_cap_outside_radius() {
        // Straight down but outside the cap radius (radius = 2 at height 2) and
        // it never reaches the lateral band -> miss.
        let miss = ray_finite_cone(
            [5.0, 0.0, 5.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            COS_45,
            2.0,
        );
        assert!(miss.is_none());
    }

    #[test]
    fn infinite_hits_where_finite_clips() {
        let origin = [-10.0, 0.0, 5.0];
        let dir = [1.0, 0.0, 0.0];
        let apex = [0.0, 0.0, 0.0];
        let axis = [0.0, 0.0, 1.0];
        let inf = ray_infinite_cone(origin, dir, apex, axis, COS_45)
            .expect("infinite cone still hits far lateral point");
        assert!(approx(inf.t, 5.0));
        let fin = ray_finite_cone(origin, dir, apex, axis, COS_45, 2.0);
        assert!(fin.is_none());
    }

    #[test]
    fn point_lies_on_ray() {
        let origin = [1.0, -2.0, 4.0];
        let dir = normalize([0.3, 0.9, -0.2]);
        let apex = [0.0, 0.0, 0.0];
        let axis = [0.0, 0.0, 1.0];
        if let Some(hit) = ray_infinite_cone(origin, dir, apex, axis, COS_45) {
            assert!(approx_pt(hit.point, point_at(origin, dir, hit.t)));
        }
    }

    #[test]
    fn narrow_cone_misses_where_wide_cone_hits() {
        let origin = [-10.0, 0.0, 2.0];
        let dir = [1.0, 0.0, 0.0];
        let apex = [0.0, 0.0, 0.0];
        let axis = [0.0, 0.0, 1.0];
        // Wide 45-degree cone: radius 2 at z = 2, so the ray hits.
        assert!(ray_infinite_cone(origin, dir, apex, axis, COS_45).is_some());
        // Very narrow cone: radius at z = 2 is tiny, and a ray offset in y never
        // reaches the slim surface.
        let off = ray_infinite_cone([-10.0, 3.0, 2.0], dir, apex, axis, 0.999_5);
        assert!(off.is_none());
    }

    #[test]
    fn finite_hit_stays_within_axial_band() {
        let apex = [0.0, 0.0, 0.0];
        let axis = [0.0, 0.0, 1.0];
        let hit = ray_finite_cone([-10.0, 0.0, 2.0], [1.0, 0.0, 0.0], apex, axis, COS_45, 5.0)
            .expect("hit");
        let axial = v_dot(v_sub(hit.point, apex), axis);
        assert!(axial >= -TOL);
        assert!(axial <= 5.0 + TOL);
    }

    #[test]
    fn tilted_axis_hit_is_on_surface() {
        let apex = [1.0, 2.0, -1.0];
        let axis = normalize([1.0, 1.0, 1.0]);
        let origin = [5.0, 2.0, -1.0];
        let dir = normalize([-1.0, 0.0, 0.0]);
        if let Some(hit) = ray_infinite_cone(origin, dir, apex, axis, COS_45) {
            assert!(cone_residual(hit.point, apex, axis, COS_45).abs() <= TOL * 10.0);
            let axial = v_dot(v_sub(hit.point, apex), axis);
            assert!(axial >= -TOL);
        }
    }

    #[test]
    fn lcg_random_infinite_residual_and_nappe() {
        let mut rng = Lcg(0x1234_5678_9abc_def0);
        let apex = [0.0, 0.0, 0.0];
        let axis = [0.0, 0.0, 1.0];
        let cos = COS_45;
        let mut hits = 0_u32;
        for _ in 0..400 {
            let origin = [
                rng.range(-6.0, 6.0),
                rng.range(-6.0, 6.0),
                rng.range(-6.0, 6.0),
            ];
            let dir = normalize([
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
            ]);
            if let Some(hit) = ray_infinite_cone(origin, dir, apex, axis, cos) {
                hits += 1;
                assert!(hit.t >= -TOL);
                // On the forward nappe.
                let axial = v_dot(v_sub(hit.point, apex), axis);
                assert!(axial >= -TOL, "axial = {axial}");
                // On the cone surface, scaled by magnitude.
                let v = v_sub(hit.point, apex);
                let scale = 1.0 + v_dot(v, v);
                assert!(
                    cone_residual(hit.point, apex, axis, cos).abs() <= TOL * scale * 10.0,
                    "residual too large"
                );
                // Point lies on the ray.
                assert!(approx_pt(hit.point, point_at(origin, dir, hit.t)));
            }
        }
        assert!(hits > 10, "expected several random hits, got {hits}");
    }

    #[test]
    fn lcg_random_finite_within_bounds() {
        let mut rng = Lcg(0x0bad_c0ff_ee00_1357);
        let apex = [0.0, 0.0, 0.0];
        let axis = [0.0, 0.0, 1.0];
        let cos = COS_45;
        let height = 4.0;
        let sin = (1.0_f32 - cos * cos).sqrt();
        let cap_radius2 = height * height * (sin * sin) / (cos * cos);
        let mut checked = 0_u32;
        for _ in 0..400 {
            let origin = [
                rng.range(-8.0, 8.0),
                rng.range(-8.0, 8.0),
                rng.range(-4.0, 10.0),
            ];
            let dir = normalize([
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
            ]);
            if let Some(hit) = ray_finite_cone(origin, dir, apex, axis, cos, height) {
                checked += 1;
                assert!(hit.t >= -TOL);
                let axial = v_dot(v_sub(hit.point, apex), axis);
                assert!(axial >= -TOL && axial <= height + TOL, "axial = {axial}");
                // Each finite hit is either on the lateral cone or on the cap.
                let v = v_sub(hit.point, apex);
                let on_lateral = cone_residual(hit.point, apex, axis, cos).abs()
                    <= TOL * (1.0 + v_dot(v, v)) * 20.0;
                let center = v_add(apex, v_scale(axis, height));
                let radial = v_sub(hit.point, center);
                let on_cap = (axial - height).abs() <= TOL * 10.0
                    && v_dot(radial, radial) <= cap_radius2 + TOL * 10.0;
                assert!(on_lateral || on_cap, "hit is neither lateral nor cap");
            }
        }
        assert!(checked > 5, "expected several finite hits, got {checked}");
    }

    #[test]
    fn lcg_random_finite_lateral_matches_infinite() {
        // Any finite lateral hit must also be an infinite-cone hit at the same
        // or nearer parameter (the finite solid is a subset along the surface).
        let mut rng = Lcg(0xdead_beef_cafe_babe);
        let apex = [0.0, 0.0, 0.0];
        let axis = [0.0, 0.0, 1.0];
        let cos = COS_45;
        let height = 3.0;
        for _ in 0..300 {
            let origin = [
                rng.range(-5.0, 5.0),
                rng.range(-5.0, 5.0),
                rng.range(-1.0, 8.0),
            ];
            let dir = normalize([
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
            ]);
            if let Some(fh) = ray_finite_cone(origin, dir, apex, axis, cos, height) {
                let axial = v_dot(v_sub(fh.point, apex), axis);
                let is_cap = (axial - height).abs() <= TOL * 10.0;
                if !is_cap {
                    let inf = ray_infinite_cone(origin, dir, apex, axis, cos)
                        .expect("lateral finite hit implies infinite hit");
                    assert!(inf.t <= fh.t + TOL);
                }
            }
        }
    }

    #[test]
    fn hit_ordering_prefers_smaller_t() {
        // Two lateral crossings; confirm the smaller t is returned.
        let apex = [0.0, 0.0, 0.0];
        let axis = [0.0, 0.0, 1.0];
        let hit =
            ray_infinite_cone([-10.0, 0.0, 3.0], [1.0, 0.0, 0.0], apex, axis, COS_45).expect("hit");
        // Crossings at x = -3 (t = 7) and x = 3 (t = 13); nearest is t = 7.
        assert!(approx(hit.t, 7.0));
    }

    #[test]
    fn finite_origin_inside_returns_forward_surface() {
        // Origin inside the finite cone; the ray exits through the lateral wall.
        let apex = [0.0, 0.0, 0.0];
        let axis = [0.0, 0.0, 1.0];
        let hit = ray_finite_cone([0.0, 0.0, 3.0], [1.0, 0.0, 0.0], apex, axis, COS_45, 10.0)
            .expect("exit through wall");
        assert!(approx(hit.t, 3.0));
        assert!(approx_pt(hit.point, [3.0, 0.0, 3.0]));
    }

    #[test]
    fn behind_origin_only_returns_none() {
        // Both algebraic roots are negative (cone entirely behind the origin).
        let miss = ray_infinite_cone(
            [0.0, 0.0, 10.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            COS_45,
        );
        assert!(miss.is_none());
    }
}
