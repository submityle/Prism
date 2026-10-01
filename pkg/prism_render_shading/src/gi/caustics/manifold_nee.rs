//! Manifold next-event estimation across a specular interface — CPU golden.
//!
//! Photon splatting alone cannot connect a *diffuse* shading point to a light
//! that is only reachable *through* a specular interface (the water surface
//! above a submerged floor, light through a glass pane).  Manifold NEE closes
//! that gap: given the diffuse receiver point and the light, it solves for the
//! point on the specular interface through which a valid reflected or refracted
//! path connects them, by driving the *half-vector constraint* to zero with a
//! Newton iteration on the interface's 2-D parameterisation.
//!
//! This module models the specular interface as a plane ([`SpecularPlane`]) —
//! the common water-surface / glass-pane case — which makes the Newton solve
//! smooth and gives an analytic ground truth for the tests (the mirror-image
//! reflection point).  The same half-vector residual handles both reflection
//! and Snell refraction, selected by [`Interaction`].
//!
//! The constraint is the generalised half-vector of Walter et al.: with `a` the
//! unit direction from the interface point to the receiver (in medium `n_a`) and
//! `b` the unit direction to the light (in medium `n_b`), the vector
//! `g = n_a * a + n_b * b` must be parallel to the surface normal on a valid
//! specular path.  Its two tangential components form the residual; zeroing them
//! yields the law of reflection (`n_a = n_b`) or Snell's law (`n_a != n_b`).
//!
//! It provides:
//!
//! * [`SpecularPlane`] — an oriented plane with a cached orthonormal tangent
//!   frame.
//! * [`Interaction`] — reflection, or refraction with the two media's indices.
//! * [`NewtonConfig`] — iteration budget, convergence tolerance and step/
//!   Jacobian clamps.
//! * [`solve_manifold`] — the Newton solver, returning the connection point and
//!   whether it converged.
//!
//! # Conventions
//! * The plane normal points toward the receiver's side by convention, but the
//!   residual is symmetric, so either orientation converges.
//! * `no_std`: math via `bevy_math`; transcendentals via `bevy_math::ops`;
//!   square roots via the `f32::sqrt` method (never `f32::exp`).
//! * The Newton step guards against a singular Jacobian (determinant clamped
//!   away from zero), clamps the step length, caps the iteration count and
//!   returns a finite fall-back point on failure — never `NaN`, never a
//!   runaway.
//! * Every function is a deterministic, allocation-free pure function: no RNG,
//!   no I/O, no GPU, no global state.

use bevy_math::Vec3;

/// Smallest index of refraction accepted; physical dielectrics have `n >= 1`.
const MIN_IOR: f32 = 1.0;

/// Clamps an index of refraction to the physical dielectric range `[1, inf)`.
#[inline]
fn clamp_ior(n: f32) -> f32 {
    if n.is_finite() {
        n.max(MIN_IOR)
    } else {
        MIN_IOR
    }
}

/// Normalises `v`, returning `fallback` for a degenerate (near-zero) input.
#[inline]
fn normalize_or(v: Vec3, fallback: Vec3) -> Vec3 {
    let len_sq = v.length_squared();
    if len_sq > f32::MIN_POSITIVE {
        v * len_sq.sqrt().recip()
    } else {
        fallback
    }
}

/// Branch-light orthonormal basis `(t1, t2)` tangent to a unit normal `n`
/// (Duff et al., *Building an Orthonormal Basis, Revisited*).
#[inline]
fn orthonormal_basis(n: Vec3) -> (Vec3, Vec3) {
    let sign = if n.z >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    let t1 = Vec3::new(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    let t2 = Vec3::new(b, sign + n.y * n.y * a, -n.y);
    (t1, t2)
}

/// An oriented specular plane with a cached orthonormal tangent frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpecularPlane {
    /// A point on the plane (the parameterisation origin).
    pub point: Vec3,
    /// Unit surface normal.
    pub normal: Vec3,
    /// First tangent axis (unit, orthogonal to `normal`).
    pub tangent_u: Vec3,
    /// Second tangent axis (unit, orthogonal to `normal` and `tangent_u`).
    pub tangent_v: Vec3,
}

impl SpecularPlane {
    /// Builds a plane from a point and a normal, deriving an orthonormal tangent
    /// frame.  A degenerate normal falls back to `+Y`.
    #[inline]
    pub fn new(point: Vec3, normal: Vec3) -> Self {
        let n = normalize_or(normal, Vec3::Y);
        let (tangent_u, tangent_v) = orthonormal_basis(n);
        Self {
            point,
            normal: n,
            tangent_u,
            tangent_v,
        }
    }

    /// Maps 2-D plane coordinates `(u, v)` to a world-space point.
    #[inline]
    pub fn position(&self, u: f32, v: f32) -> Vec3 {
        self.point + self.tangent_u * u + self.tangent_v * v
    }

    /// Projects a world point onto the plane and returns its `(u, v)`
    /// coordinates.
    #[inline]
    pub fn project(&self, world: Vec3) -> (f32, f32) {
        let d = world - self.point;
        (d.dot(self.tangent_u), d.dot(self.tangent_v))
    }
}

/// The specular interaction a manifold path makes at the interface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Interaction {
    /// Mirror reflection: both endpoints are in the same medium.
    Reflect,
    /// Refraction: the receiver sits in medium `n_a`, the light in medium `n_b`.
    Refract {
        /// Index of refraction on the receiver's side.
        n_a: f32,
        /// Index of refraction on the light's side.
        n_b: f32,
    },
}

impl Interaction {
    /// The `(n_a, n_b)` index pair the half-vector residual uses; reflection is
    /// `(1, 1)`.
    #[inline]
    fn indices(&self) -> (f32, f32) {
        match *self {
            Interaction::Reflect => (1.0, 1.0),
            Interaction::Refract { n_a, n_b } => (clamp_ior(n_a), clamp_ior(n_b)),
        }
    }
}

/// Tuning for the Newton manifold solve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NewtonConfig {
    /// Maximum Newton iterations before giving up.
    pub max_iters: u32,
    /// Convergence tolerance on the residual's magnitude.
    pub tolerance: f32,
    /// Finite-difference step used to build the Jacobian.
    pub fd_step: f32,
    /// Maximum per-component step length (prevents overshoot / runaway).
    pub max_step: f32,
    /// Smallest Jacobian determinant magnitude treated as invertible; below
    /// this the solve bails out to the fall-back point.
    pub min_det: f32,
}

impl Default for NewtonConfig {
    #[inline]
    fn default() -> Self {
        Self {
            max_iters: 32,
            tolerance: 1.0e-5,
            fd_step: 1.0e-3,
            max_step: 16.0,
            min_det: 1.0e-9,
        }
    }
}

/// The outcome of a manifold solve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ManifoldResult {
    /// The connection point on the interface (world space).  On failure this is
    /// the fall-back projection of the endpoints' midpoint.
    pub point: Vec3,
    /// Whether the Newton iteration converged to the tolerance.
    pub converged: bool,
    /// Number of Newton iterations actually performed.
    pub iterations: u32,
    /// Residual magnitude at the returned point.
    pub residual: f32,
}

/// The two tangential components of the generalised half-vector at plane
/// coordinates `(u, v)` — the quantity the Newton solve drives to zero.
#[inline]
fn residual(
    plane: &SpecularPlane,
    u: f32,
    v: f32,
    receiver: Vec3,
    light: Vec3,
    n_a: f32,
    n_b: f32,
) -> [f32; 2] {
    let p = plane.position(u, v);
    let a = normalize_or(receiver - p, plane.normal);
    let b = normalize_or(light - p, plane.normal);
    let g = n_a * a + n_b * b;
    [g.dot(plane.tangent_u), g.dot(plane.tangent_v)]
}

/// Solves for the specular connection point via a damped, Jacobian-clamped
/// Newton iteration on the half-vector constraint.
///
/// Starting from the plane projection of the midpoint of `receiver` and
/// `light`, it drives the two tangential components of `g = n_a*a + n_b*b` to
/// zero.  On success the returned [`ManifoldResult::point`] satisfies the law of
/// reflection (for [`Interaction::Reflect`]) or Snell's law (for
/// [`Interaction::Refract`]) at the interface.
///
/// Robustness: the 2x2 Jacobian is built by central differences and inverted
/// only when its determinant magnitude exceeds [`NewtonConfig::min_det`];
/// otherwise the iteration stops.  Each step is clamped to
/// [`NewtonConfig::max_step`] per component.  If the tolerance is never met the
/// best-so-far point is returned with `converged == false`, so callers can fall
/// back to a different estimator.  The result is always finite.
#[inline]
pub fn solve_manifold(
    plane: &SpecularPlane,
    receiver: Vec3,
    light: Vec3,
    interaction: Interaction,
    config: NewtonConfig,
) -> ManifoldResult {
    let (n_a, n_b) = interaction.indices();
    let max_iters = config.max_iters.max(1);
    let tol = if config.tolerance.is_finite() {
        config.tolerance.max(0.0)
    } else {
        1.0e-5
    };
    let eps = if config.fd_step.is_finite() && config.fd_step > 0.0 {
        config.fd_step
    } else {
        1.0e-3
    };
    let max_step = if config.max_step.is_finite() && config.max_step > 0.0 {
        config.max_step
    } else {
        16.0
    };
    let min_det = if config.min_det.is_finite() && config.min_det > 0.0 {
        config.min_det
    } else {
        1.0e-9
    };

    // Initial guess: project the midpoint of the endpoints onto the plane.
    let midpoint = 0.5 * (receiver + light);
    let (mut u, mut v) = plane.project(midpoint);
    if !u.is_finite() {
        u = 0.0;
    }
    if !v.is_finite() {
        v = 0.0;
    }

    let mut best_u = u;
    let mut best_v = v;
    let mut best_res = f32::INFINITY;
    let mut iterations = 0u32;
    let mut converged = false;

    for _ in 0..max_iters {
        iterations += 1;
        let r = residual(plane, u, v, receiver, light, n_a, n_b);
        let r_mag = (r[0] * r[0] + r[1] * r[1]).sqrt();
        if r_mag < best_res {
            best_res = r_mag;
            best_u = u;
            best_v = v;
        }
        if r_mag <= tol {
            converged = true;
            break;
        }

        // Central-difference Jacobian J[i][j] = d r_i / d (u_j).
        let r_up = residual(plane, u + eps, v, receiver, light, n_a, n_b);
        let r_um = residual(plane, u - eps, v, receiver, light, n_a, n_b);
        let r_vp = residual(plane, u, v + eps, receiver, light, n_a, n_b);
        let r_vm = residual(plane, u, v - eps, receiver, light, n_a, n_b);
        let inv_2eps = 1.0 / (2.0 * eps);
        let j00 = (r_up[0] - r_um[0]) * inv_2eps;
        let j10 = (r_up[1] - r_um[1]) * inv_2eps;
        let j01 = (r_vp[0] - r_vm[0]) * inv_2eps;
        let j11 = (r_vp[1] - r_vm[1]) * inv_2eps;

        let det = j00 * j11 - j01 * j10;
        if !det.is_finite() || det.abs() < min_det {
            // Singular / ill-conditioned Jacobian: stop and fall back.
            break;
        }
        let inv_det = 1.0 / det;
        // delta = -J^{-1} r.
        let du = -inv_det * (j11 * r[0] - j01 * r[1]);
        let dv = -inv_det * (-j10 * r[0] + j00 * r[1]);
        if !du.is_finite() || !dv.is_finite() {
            break;
        }
        let du = du.clamp(-max_step, max_step);
        let dv = dv.clamp(-max_step, max_step);
        u += du;
        v += dv;
        if !u.is_finite() || !v.is_finite() {
            break;
        }
    }

    let (ru, rv) = (best_u, best_v);
    let final_res = residual(plane, ru, rv, receiver, light, n_a, n_b);
    let residual_mag = (final_res[0] * final_res[0] + final_res[1] * final_res[1]).sqrt();

    ManifoldResult {
        point: plane.position(ru, rv),
        converged,
        iterations,
        residual: residual_mag,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::Vec3;

    /// Analytic reflection point on the y=0 plane for two points above it.
    fn analytic_reflection(x: Vec3, l: Vec3) -> Vec3 {
        // Mirror the light below the plane; the straight segment x -> l' crosses
        // the plane at the reflection point (similar triangles on height).
        let l_mirror = Vec3::new(l.x, -l.y, l.z);
        let t = x.y / (x.y - l_mirror.y);
        x + (l_mirror - x) * t
    }

    #[test]
    fn reflection_converges_to_analytic_point() {
        let plane = SpecularPlane::new(Vec3::ZERO, Vec3::Y);
        let x = Vec3::new(-1.0, 2.0, 0.3);
        let l = Vec3::new(2.0, 1.0, -0.4);
        let res = solve_manifold(&plane, x, l, Interaction::Reflect, NewtonConfig::default());
        assert!(res.converged, "did not converge: {res:?}");
        let expected = analytic_reflection(x, l);
        assert!(
            (res.point - expected).length() < 1.0e-3,
            "point={:?} expected={:?}",
            res.point,
            expected
        );
    }

    #[test]
    fn reflection_obeys_law_of_reflection() {
        let plane = SpecularPlane::new(Vec3::ZERO, Vec3::Y);
        let x = Vec3::new(-1.5, 1.0, 0.0);
        let l = Vec3::new(1.0, 2.0, 0.0);
        let res = solve_manifold(&plane, x, l, Interaction::Reflect, NewtonConfig::default());
        assert!(res.converged);
        let p = res.point;
        let a = (x - p).normalize();
        let b = (l - p).normalize();
        // Angles to the normal are equal => equal cosines.
        let ca = a.dot(Vec3::Y);
        let cb = b.dot(Vec3::Y);
        assert!((ca - cb).abs() < 1.0e-3, "ca={ca} cb={cb}");
    }

    #[test]
    fn refraction_obeys_snell_at_connection() {
        // Receiver submerged (below plane, in water n_a=1.33); light in air above.
        let plane = SpecularPlane::new(Vec3::ZERO, Vec3::Y);
        let x = Vec3::new(-0.8, -1.0, 0.0); // under water
        let l = Vec3::new(1.2, 2.0, 0.0); // in air
        let interaction = Interaction::Refract { n_a: 1.33, n_b: 1.0 };
        let res = solve_manifold(&plane, x, l, interaction, NewtonConfig::default());
        assert!(res.converged, "did not converge: {res:?}");
        let p = res.point;
        let a = (x - p).normalize();
        let b = (l - p).normalize();
        // Angle each side makes with the normal.
        let cos_a = a.dot(Vec3::NEG_Y).abs();
        let cos_b = b.dot(Vec3::Y).abs();
        let sin_a = (1.0 - cos_a * cos_a).max(0.0).sqrt();
        let sin_b = (1.0 - cos_b * cos_b).max(0.0).sqrt();
        // Snell: n_a sin_a = n_b sin_b.
        assert!(
            (1.33 * sin_a - 1.0 * sin_b).abs() < 2.0e-3,
            "n_a sin_a={} n_b sin_b={}",
            1.33 * sin_a,
            sin_b
        );
    }

    #[test]
    fn reflection_handles_tilted_plane() {
        // Tilted plane still converges to a valid reflection (equal angles).
        let normal = Vec3::new(0.2, 1.0, -0.1).normalize();
        let plane = SpecularPlane::new(Vec3::new(0.1, 0.0, 0.0), normal);
        let x = plane.point + normal * 1.5 + plane.tangent_u * -1.0;
        let l = plane.point + normal * 2.0 + plane.tangent_u * 1.3 + plane.tangent_v * 0.5;
        let res = solve_manifold(&plane, x, l, Interaction::Reflect, NewtonConfig::default());
        assert!(res.converged, "tilted did not converge: {res:?}");
        let p = res.point;
        let a = (x - p).normalize();
        let b = (l - p).normalize();
        let ca = a.dot(normal);
        let cb = b.dot(normal);
        assert!((ca - cb).abs() < 1.0e-3, "ca={ca} cb={cb}");
    }

    #[test]
    fn tangent_frame_is_orthonormal() {
        for n in [
            Vec3::Y,
            Vec3::new(0.3, 0.9, -0.2).normalize(),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
        ] {
            let plane = SpecularPlane::new(Vec3::ZERO, n);
            assert!((plane.tangent_u.length() - 1.0).abs() < 1.0e-5);
            assert!((plane.tangent_v.length() - 1.0).abs() < 1.0e-5);
            assert!(plane.tangent_u.dot(plane.normal).abs() < 1.0e-5);
            assert!(plane.tangent_v.dot(plane.normal).abs() < 1.0e-5);
            assert!(plane.tangent_u.dot(plane.tangent_v).abs() < 1.0e-5);
        }
    }

    #[test]
    fn project_round_trips() {
        let plane = SpecularPlane::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.1, 1.0, 0.2));
        let (u, v) = (0.7, -1.3);
        let w = plane.position(u, v);
        let (u2, v2) = plane.project(w);
        assert!((u - u2).abs() < 1.0e-5 && (v - v2).abs() < 1.0e-5);
    }

    #[test]
    fn failure_returns_finite_fallback() {
        // Degenerate: receiver and light coincident on the plane -> no valid
        // half-vector direction. Must not NaN and must report non-convergence
        // or a finite point.
        let plane = SpecularPlane::new(Vec3::ZERO, Vec3::Y);
        let p = Vec3::new(0.5, 0.0, 0.5);
        let res = solve_manifold(&plane, p, p, Interaction::Reflect, NewtonConfig::default());
        assert!(res.point.is_finite());
        assert!(res.residual.is_finite());
    }

    #[test]
    fn iteration_budget_is_respected() {
        let plane = SpecularPlane::new(Vec3::ZERO, Vec3::Y);
        let x = Vec3::new(-1.0, 2.0, 0.0);
        let l = Vec3::new(2.0, 1.0, 0.0);
        let cfg = NewtonConfig {
            max_iters: 3,
            ..NewtonConfig::default()
        };
        let res = solve_manifold(&plane, x, l, Interaction::Reflect, cfg);
        assert!(res.iterations <= 3);
        assert!(res.point.is_finite());
    }

    #[test]
    fn degenerate_config_does_not_nan() {
        let plane = SpecularPlane::new(Vec3::ZERO, Vec3::ZERO);
        let cfg = NewtonConfig {
            max_iters: 0,
            tolerance: f32::NAN,
            fd_step: 0.0,
            max_step: -1.0,
            min_det: f32::NAN,
        };
        let res = solve_manifold(
            &plane,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 1.0),
            Interaction::Refract { n_a: 0.0, n_b: f32::NAN },
            cfg,
        );
        assert!(res.point.is_finite() && res.residual.is_finite());
    }

    #[test]
    fn is_deterministic() {
        let plane = SpecularPlane::new(Vec3::ZERO, Vec3::Y);
        let x = Vec3::new(-1.0, 2.0, 0.3);
        let l = Vec3::new(2.0, 1.0, -0.4);
        let a = solve_manifold(&plane, x, l, Interaction::Reflect, NewtonConfig::default());
        let b = solve_manifold(&plane, x, l, Interaction::Reflect, NewtonConfig::default());
        assert_eq!(a, b);
    }
}
