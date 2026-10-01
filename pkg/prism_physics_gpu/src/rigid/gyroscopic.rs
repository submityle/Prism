//! Implicit gyroscopic integration for the rigid-body angular update.
//!
//! Euler's rigid-body equations couple the three body-frame angular velocity
//! components through the gyroscopic term `omega x (I * omega)`. The base
//! integrator in [`cpu`](super::cpu) treats that term **explicitly**: it
//! evaluates the coupling at the start-of-substep angular velocity and subtracts
//! it. Explicit treatment is exact in the parity sense and fine for the moderate
//! spin rates of typical gameplay, but it can *inject* energy for a body spun
//! fast about its intermediate principal axis — the Dzhanibekov (tennis-racket)
//! regime — eventually blowing the spin up instead of letting it tumble
//! stably.
//!
//! This module adds the **implicit** alternative. Rather than evaluating the
//! coupling at the old angular velocity, it solves for the end-of-substep
//! body-frame angular velocity `omega1` that satisfies
//!
//! ```text
//! I * omega1 + h * (omega1 x (I * omega1)) = I * omega0
//! ```
//!
//! where `omega0` is the body-frame angular velocity *after* the external
//! torque has been applied and `I = diag(Ix, Iy, Iz)` is the body-frame
//! principal inertia. The left side is the exact backward-Euler form of the
//! torque-free Euler equation. Dotting it with the solved angular velocity
//! shows `|I*omega0|^2 = |I*omega1|^2 + h^2 * |omega1 x (I*omega1)|^2`, so the
//! scheme is slightly *dissipative* in angular-momentum magnitude rather than
//! exactly conservative — and that dissipation is exactly what makes it
//! unconditionally stable: unlike the explicit form it never gains energy, so a
//! body spun fast about its intermediate axis tumbles stably instead of blowing
//! up. The nonlinear system is solved with one or
//! more Newton iterations (one is the default and matches Bullet's
//! `btRigidBody::computeGyroscopicImpulseImplicit_Body`).
//!
//! # Newton step
//!
//! With residual `f(w) = I*w - L0 + h*(w x (I*w))` and `L0 = I*omega0` constant,
//! the Jacobian is
//!
//! ```text
//! J = I_mat + h * (skew(w) * I_mat - skew(I*w))
//! ```
//!
//! where `skew(v)` is the cross-product matrix. Each iteration solves
//! `J * delta = -f` with a hand-written 3x3 cofactor inverse and updates
//! `w += delta`. When the Jacobian is numerically singular the step is dropped
//! (`delta = 0`), leaving the explicit-free value in place rather than
//! producing a non-finite result.
//!
//! # Honest limits
//!
//! The implicit solve requires a strictly positive inertia on every axis so the
//! `I_mat` block of the Jacobian is non-singular. A body with any locked axis
//! (`inverse_inertia` component zero, mapped to zero inertia) falls back to the
//! explicit path; the caller enforces this guard identically on the `CPU` and
//! `GPU` so the two stay in parity. The solver operates purely in the body
//! frame on the diagonal inertia; a full dense inertia tensor is a later
//! concern.
//!
//! The 3x3 inverse is written as an explicit cofactor expansion over scalar
//! `f32` values — never [`glam`]'s `Mat3::inverse`, whose internal arithmetic
//! cannot be reproduced byte-for-byte — so the device shader
//! (`shaders/rigid_integrate.wgsl`) performs the identical sequence of multiplies
//! and divides.
//!
//! Provenance: implicit (backward-Euler) gyroscopic integration of Euler's
//! rigid-body equations, as in Bullet's `computeGyroscopicImpulseImplicit_Body`
//! and `PhysX`'s `eENABLE_GYROSCOPIC_FORCES`. No Unreal Engine source or derived
//! code.

use glam::Vec3;

/// Determinant magnitude below which the Newton Jacobian is treated as
/// singular and the step is dropped. Matches `GYRO_DET_EPSILON` in
/// `shaders/rigid_integrate.wgsl`.
pub(crate) const GYRO_DET_EPSILON: f32 = 1.0e-20;

/// Selects how the gyroscopic coupling term of the angular update is
/// integrated.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GyroscopicMode {
    /// Evaluate `omega x (I * omega)` at the start-of-substep angular velocity
    /// and subtract it. Cheap and exact in the parity sense; can gain energy
    /// for a body spun fast about its intermediate axis.
    #[default]
    Explicit,
    /// Solve the backward-Euler coupling for the end-of-substep angular
    /// velocity with Newton iteration. Conserves body-frame angular-momentum
    /// magnitude and stays stable in the intermediate-axis regime.
    Implicit,
}

/// Tunables controlling the gyroscopic treatment of the angular update.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GyroscopicConfig {
    /// Which integration scheme the gyroscopic coupling uses.
    pub mode: GyroscopicMode,
    /// Newton iterations per substep when `mode` is
    /// [`GyroscopicMode::Implicit`]. Clamped to at least `1` at use. Ignored by
    /// the explicit scheme.
    pub iterations: u32,
}

impl GyroscopicConfig {
    /// Default Newton iteration count for the implicit scheme.
    pub const DEFAULT_ITERATIONS: u32 = 1;

    /// Creates a gyroscopic configuration.
    #[must_use]
    pub fn new(mode: GyroscopicMode, iterations: u32) -> GyroscopicConfig {
        GyroscopicConfig { mode, iterations }
    }

    /// A configuration selecting the explicit scheme (the integrator default).
    #[must_use]
    pub fn explicit() -> GyroscopicConfig {
        GyroscopicConfig {
            mode: GyroscopicMode::Explicit,
            iterations: Self::DEFAULT_ITERATIONS,
        }
    }

    /// A configuration selecting the implicit scheme with `iterations` Newton
    /// steps (clamped to at least `1` at use).
    #[must_use]
    pub fn implicit(iterations: u32) -> GyroscopicConfig {
        GyroscopicConfig {
            mode: GyroscopicMode::Implicit,
            iterations,
        }
    }

    /// Returns the effective Newton iteration count (at least `1`).
    #[must_use]
    pub fn effective_iterations(&self) -> u32 {
        self.iterations.max(1)
    }
}

impl Default for GyroscopicConfig {
    fn default() -> Self {
        GyroscopicConfig::explicit()
    }
}

/// Solves the implicit (backward-Euler) gyroscopic update for the end-of-substep
/// body-frame angular velocity.
///
/// Given the body-frame angular velocity `omega_body` after the external torque
/// has been applied, the diagonal principal `inertia`, the substep size `h`, and
/// the Newton `iterations` count, returns `omega1` approximately satisfying
/// `I*omega1 + h*(omega1 x (I*omega1)) = I*omega_body`. The caller must ensure
/// every `inertia` component is strictly positive; a zero (locked) axis makes
/// the Jacobian singular and must use the explicit path instead.
pub(crate) fn implicit_gyroscopic_body(
    omega_body: Vec3,
    inertia: Vec3,
    h: f32,
    iterations: u32,
) -> Vec3 {
    let ix = inertia.x;
    let iy = inertia.y;
    let iz = inertia.z;
    // Constant right-hand-side angular momentum L0 = I * omega0.
    let l0x = ix * omega_body.x;
    let l0y = iy * omega_body.y;
    let l0z = iz * omega_body.z;

    let mut omega = omega_body;
    let steps = iterations.max(1);
    let mut iter = 0u32;
    while iter < steps {
        let wx = omega.x;
        let wy = omega.y;
        let wz = omega.z;
        // Current angular momentum L = I * omega.
        let lx = ix * wx;
        let ly = iy * wy;
        let lz = iz * wz;
        // Residual f = I*omega - L0 + h*(omega x L).
        let cross_x = wy * lz - wz * ly;
        let cross_y = wz * lx - wx * lz;
        let cross_z = wx * ly - wy * lx;
        let fx = lx - l0x + h * cross_x;
        let fy = ly - l0y + h * cross_y;
        let fz = lz - l0z + h * cross_z;
        // Jacobian J = I_mat + h*(skew(omega)*I_mat - skew(L)), row-major.
        // Diagonal entries carry the I_mat term; off-diagonals are h*A where
        // A = skew(omega)*I_mat - skew(L).
        let a01 = -wz * iy + lz;
        let a02 = wy * iz - ly;
        let a10 = wz * ix - lz;
        let a12 = -wx * iz + lx;
        let a20 = -wy * ix + ly;
        let a21 = wx * iy - lx;
        let m = [
            ix,
            h * a01,
            h * a02,
            h * a10,
            iy,
            h * a12,
            h * a20,
            h * a21,
            iz,
        ];
        // Solve J * delta = -f.
        let delta = solve_3x3(&m, -fx, -fy, -fz);
        omega.x += delta.x;
        omega.y += delta.y;
        omega.z += delta.z;
        iter += 1;
    }
    omega
}

/// Solves the 3x3 system `m * x = b` for `x` using an explicit cofactor
/// (adjugate over determinant) inverse, where `m` is stored row-major as
/// `[m00, m01, m02, m10, m11, m12, m20, m21, m22]` and `b = (bx, by, bz)`.
///
/// Returns the zero vector when the determinant magnitude is below
/// [`GYRO_DET_EPSILON`], so a numerically singular Jacobian drops the Newton
/// step instead of producing a non-finite result.
fn solve_3x3(m: &[f32; 9], bx: f32, by: f32, bz: f32) -> Vec3 {
    // Cofactors of the first column, reused for the determinant.
    let c0 = m[4] * m[8] - m[5] * m[7];
    let c1 = m[5] * m[6] - m[3] * m[8];
    let c2 = m[3] * m[7] - m[4] * m[6];
    let det = m[0] * c0 + m[1] * c1 + m[2] * c2;
    if det.abs() < GYRO_DET_EPSILON {
        return Vec3::ZERO;
    }
    let inv_det = 1.0 / det;
    // x = (adjugate(m) * b) / det, where adjugate is the transpose of the
    // cofactor matrix.
    let x = (c0 * bx + (m[2] * m[7] - m[1] * m[8]) * by + (m[1] * m[5] - m[2] * m[4]) * bz)
        * inv_det;
    let y = (c1 * bx + (m[0] * m[8] - m[2] * m[6]) * by + (m[2] * m[3] - m[0] * m[5]) * bz)
        * inv_det;
    let z = (c2 * bx + (m[1] * m[6] - m[0] * m[7]) * by + (m[0] * m[4] - m[1] * m[3]) * bz)
        * inv_det;
    Vec3::new(x, y, z)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Body-frame angular momentum magnitude `|I * omega|`.
    fn momentum_magnitude(inertia: Vec3, omega: Vec3) -> f32 {
        Vec3::new(
            inertia.x * omega.x,
            inertia.y * omega.y,
            inertia.z * omega.z,
        )
        .length()
    }

    /// Evaluates the implicit residual `I*w - I*w0 + h*(w x (I*w))` whose root
    /// the Newton iteration seeks.
    fn residual(omega_new: Vec3, omega_old: Vec3, inertia: Vec3, h: f32) -> Vec3 {
        let l_new = Vec3::new(
            inertia.x * omega_new.x,
            inertia.y * omega_new.y,
            inertia.z * omega_new.z,
        );
        let l_old = Vec3::new(
            inertia.x * omega_old.x,
            inertia.y * omega_old.y,
            inertia.z * omega_old.z,
        );
        l_new - l_old + omega_new.cross(l_new) * h
    }

    #[test]
    fn config_defaults_to_explicit_single_iteration() {
        let config = GyroscopicConfig::default();
        assert_eq!(config.mode, GyroscopicMode::Explicit);
        assert_eq!(config.effective_iterations(), 1);
    }

    #[test]
    fn zero_iterations_clamps_to_one() {
        let config = GyroscopicConfig::implicit(0);
        assert_eq!(config.effective_iterations(), 1);
    }

    #[test]
    fn newton_reduces_residual_for_asymmetric_body() {
        // Classic tumbler inertia, moderate spin, substantial substep.
        let inertia = Vec3::new(1.0, 2.0, 4.0);
        let omega0 = Vec3::new(3.0, 2.0, 1.0);
        let h = 0.02;

        // One Newton step should already drive the residual well below the
        // explicit scheme's leftover coupling.
        let omega1 = implicit_gyroscopic_body(omega0, inertia, h, 1);
        let r1 = residual(omega1, omega0, inertia, h).length();

        // Three steps should drive it to near machine precision.
        let omega3 = implicit_gyroscopic_body(omega0, inertia, h, 3);
        let r3 = residual(omega3, omega0, inertia, h).length();

        assert!(r1 < 1e-2, "single-step residual too large: {r1}");
        assert!(r3 < 1e-5, "three-step residual not converged: {r3}");
        assert!(r3 < r1, "more iterations did not reduce the residual");
    }

    #[test]
    fn nearly_conserves_body_frame_momentum_magnitude() {
        // Dotting the backward-Euler update `L1 + h*(w1 x L1) = L0` with itself
        // gives `|L0|^2 = |L1|^2 + h^2*|w1 x L1|^2`, so the scheme is slightly
        // *dissipative* in momentum magnitude (|L1| <= |L0|) — the very
        // property that makes it unconditionally stable. Over a long run the
        // magnitude must therefore stay bounded, drift downward monotonically,
        // and only by a small amount.
        let inertia = Vec3::new(1.0, 2.0, 4.0);
        let omega0 = Vec3::new(2.0, 0.1, 0.1);
        let h = 1.0 / 240.0;

        let mut omega = omega0;
        let m0 = momentum_magnitude(inertia, omega);
        let mut previous = m0;
        // Integrate many substeps with a well-converged solve.
        for _ in 0..2000 {
            omega = implicit_gyroscopic_body(omega, inertia, h, 4);
            let m = momentum_magnitude(inertia, omega);
            // Dissipative: never gains magnitude (allow a hair of f32 noise).
            assert!(m <= previous + 1e-6, "momentum grew {previous} -> {m}");
            previous = m;
        }
        let m1 = momentum_magnitude(inertia, omega);
        // The accumulated dissipation stays tiny for gameplay substeps.
        assert!(
            (m0 - m1) < 1e-2 * m0,
            "momentum magnitude dissipated too fast {m0} -> {m1}"
        );
    }

    #[test]
    fn intermediate_axis_spin_stays_bounded() {
        // Spin dominantly about the intermediate axis (y, inertia 2) with a
        // small perturbation: the Dzhanibekov regime where *explicit*
        // integration gains energy and blows the spin up. The conserved
        // invariant is the angular-momentum magnitude |I*omega| (the backward
        // Euler scheme is slightly dissipative in it, never gaining); the
        // angular-velocity magnitude itself legitimately oscillates during a
        // flip because the inertia is anisotropic, so it is not the quantity to
        // bound. We assert momentum never grows and the spin stays finite and
        // bounded over thousands of steps at a fast 8 rad/s spin.
        let inertia = Vec3::new(1.0, 2.0, 4.0);
        let omega0 = Vec3::new(0.01, 8.0, 0.01);
        let h = 1.0 / 120.0;

        let mut omega = omega0;
        let m0 = momentum_magnitude(inertia, omega);
        let mut previous_m = m0;
        for _ in 0..5000 {
            omega = implicit_gyroscopic_body(omega, inertia, h, 2);
            assert!(omega.is_finite(), "angular velocity went non-finite");
            let m = momentum_magnitude(inertia, omega);
            // Momentum magnitude is the stable invariant: never grows.
            assert!(m <= previous_m + 1e-5, "momentum grew {previous_m} -> {m}");
            previous_m = m;
            // Angular velocity oscillates but must never blow up the way the
            // explicit scheme does.
            assert!(
                omega.length() < 4.0 * omega0.length(),
                "intermediate-axis spin blew up to {}",
                omega.length()
            );
        }
        // The tumble persists rather than decaying to rest.
        assert!(
            momentum_magnitude(inertia, omega) > 0.5 * m0,
            "intermediate-axis spin dissipated to rest"
        );
    }

    #[test]
    fn isotropic_body_has_no_coupling() {
        // An isotropic inertia produces zero gyroscopic coupling, so the
        // implicit solve returns the input unchanged (the residual is already
        // zero at omega0).
        let inertia = Vec3::splat(2.5);
        let omega0 = Vec3::new(1.3, -0.7, 0.4);
        let omega1 = implicit_gyroscopic_body(omega0, inertia, 0.01, 1);
        assert!(
            (omega1 - omega0).length() < 1e-6,
            "isotropic body coupled: {omega0:?} -> {omega1:?}"
        );
    }

    #[test]
    fn singular_jacobian_drops_the_step() {
        // A zero determinant must yield a zero solve rather than NaN/inf.
        let singular = [0.0f32; 9];
        let delta = solve_3x3(&singular, 1.0, 2.0, 3.0);
        assert_eq!(delta, Vec3::ZERO);
    }

    #[test]
    fn solve_3x3_matches_known_inverse() {
        // Diagonal system is trivially invertible: m * x = b => x = b / diag.
        let m = [2.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 8.0];
        let x = solve_3x3(&m, 6.0, 8.0, 24.0);
        assert!((x.x - 3.0).abs() < 1e-6);
        assert!((x.y - 2.0).abs() < 1e-6);
        assert!((x.z - 3.0).abs() < 1e-6);
    }
}
