//! Bit-exact Vertex Block Descent (VBD) sweep primitives for the GPU-mirrored
//! cloth solver.
//!
//! This module is the single authoritative home for the per-vertex VBD Newton
//! step used by the render cloth solver and its `cloth_vbd_sweep_color` GPU
//! kernel. The render crate keeps only guards, SoA packing, adjacency, and
//! sweep orchestration; the actual gradient / PSD-Hessian accumulation and the
//! `3x3` solve live here so there is a single source of truth.
//!
//! # Why a dedicated path (not [`crate::vbd`])
//!
//! The general-purpose soft-body VBD in [`crate::vbd`] assembles its per-vertex
//! system with `glam`'s column-major [`glam::Mat3`] and inverts it with
//! [`glam::Mat3::inverse`]. That path is pinned bit-for-bit to the *physics* GPU
//! kernel (`prism_physics_gpu`'s `GpuVbd`) and its own `vbd_parity` golden.
//!
//! The render cloth kernel targets a *different* GPU program
//! (`cloth_vbd_sweep_color`) whose WGSL performs an explicit row-major cofactor
//! inversion. Floating-point addition/multiplication is not associative, so the
//! two inversion schemes do not agree to the last bit. To keep the render CPU
//! reference bit-identical to its WGSL kernel (asserted by the render
//! `vbd_parity` golden via `f32::to_bits`), this module reproduces that exact
//! row-major cofactor arithmetic rather than reusing [`crate::vbd`]. The two
//! paths are the same VBD formulation; only the operation order differs, by
//! GPU-kernel contract.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! per-vertex variational system (implicit-Euler inertia plus summed,
//! PSD-projected spring Hessians) and the Gauss-Seidel block descent are the
//! formulation published by Chen et al., "Vertex Block Descent" (2024); the
//! Hookean spring energy, its gradient/Hessian, and the positive-semidefinite
//! projection are standard, publicly documented results (Liu et al., "Fast
//! Simulation of Mass-Spring Systems", 2013).

use glam::Vec3;

use crate::math::scalar::Real;

/// A `3x3` solve is skipped when the determinant is below this, so a degenerate
/// (singular) Hessian never divides by ~0. Matches the render / WGSL contract.
pub const EPS_DET: Real = 1.0e-20;

/// Upper bound on per-constraint stiffness. A perfectly rigid constraint
/// (`compliance == 0`) or a vanishingly small compliance would otherwise imply
/// an infinite stiffness; capping it keeps the Hessian finite while still
/// dominating the inertial term by many orders of magnitude. Matches the
/// render / WGSL contract.
pub const MAX_STIFFNESS: Real = 1.0e9;

/// Below this squared separation the spring direction is undefined, so the
/// constraint contributes nothing. Matches the render / WGSL contract.
pub const EPS_LEN_SQ: Real = 1e-12;

/// A dense symmetric `3x3` matrix in **row-major** order, used only for the
/// per-vertex Hessian.
///
/// The explicit row-major layout and the hand-written cofactor solve are load
/// bearing: they reproduce the exact floating-point operation order of the
/// `cloth_vbd_sweep_color` WGSL kernel so the CPU reference stays bit-identical
/// to the GPU (see the module docs). Do not swap this for [`glam::Mat3`].
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SweepHessian {
    /// Row-major entries `[m00, m01, m02, m10, m11, m12, m20, m21, m22]`.
    m: [Real; 9],
}

impl SweepHessian {
    /// The all-zero Hessian accumulator.
    #[must_use]
    pub const fn zero() -> Self {
        Self { m: [0.0; 9] }
    }

    /// A scalar multiple of the identity.
    #[must_use]
    pub fn scaled_identity(s: Real) -> Self {
        let mut m = [0.0; 9];
        m[0] = s;
        m[4] = s;
        m[8] = s;
        Self { m }
    }

    /// The outer product `v vᵀ` scaled by `s`.
    #[must_use]
    pub fn scaled_outer(v: Vec3, s: Real) -> Self {
        Self {
            m: [
                s * v.x * v.x,
                s * v.x * v.y,
                s * v.x * v.z,
                s * v.y * v.x,
                s * v.y * v.y,
                s * v.y * v.z,
                s * v.z * v.x,
                s * v.z * v.y,
                s * v.z * v.z,
            ],
        }
    }

    /// Component-wise matrix sum `self + rhs`.
    #[must_use]
    pub fn add(self, rhs: Self) -> Self {
        let mut m = [0.0; 9];
        for (out, (a, b)) in m.iter_mut().zip(self.m.iter().zip(rhs.m.iter())) {
            *out = a + b;
        }
        Self { m }
    }

    /// Solves `self * x = rhs` by explicit cofactor inversion. Returns [`None`]
    /// when the matrix is (near-)singular or the result is non-finite, so the
    /// caller can simply not move the vertex this sweep instead of producing a
    /// `NaN` position.
    #[must_use]
    pub fn solve(self, rhs: Vec3) -> Option<Vec3> {
        let m = &self.m;
        let c00 = m[4] * m[8] - m[5] * m[7];
        let c01 = m[5] * m[6] - m[3] * m[8];
        let c02 = m[3] * m[7] - m[4] * m[6];
        let det = m[0] * c00 + m[1] * c01 + m[2] * c02;
        if det.abs() < EPS_DET {
            return None;
        }
        let inv_det = 1.0 / det;
        // Cofactor (adjugate) columns; the inverse is adjugateᵀ / det.
        let c10 = m[2] * m[7] - m[1] * m[8];
        let c11 = m[0] * m[8] - m[2] * m[6];
        let c12 = m[1] * m[6] - m[0] * m[7];
        let c20 = m[1] * m[5] - m[2] * m[4];
        let c21 = m[2] * m[3] - m[0] * m[5];
        let c22 = m[0] * m[4] - m[1] * m[3];
        let x = (c00 * rhs.x + c10 * rhs.y + c20 * rhs.z) * inv_det;
        let y = (c01 * rhs.x + c11 * rhs.y + c21 * rhs.z) * inv_det;
        let z = (c02 * rhs.x + c12 * rhs.y + c22 * rhs.z) * inv_det;
        let out = Vec3::new(x, y, z);
        if out.x.is_finite() && out.y.is_finite() && out.z.is_finite() {
            Some(out)
        } else {
            None
        }
    }
}

impl Default for SweepHessian {
    fn default() -> Self {
        SweepHessian::zero()
    }
}

/// Converts an XPBD compliance `alpha` into the VBD energy stiffness `k` for one
/// substep of squared size `dt_sub_sq`.
///
/// XPBD compliance `α` relates to stiffness by `k = 1 / (α · dt²)`. A rigid
/// constraint (`α == 0`) or a tiny compliance would blow that up, so the result
/// is capped at [`MAX_STIFFNESS`]. `dt_sub_sq` is always positive at the call
/// site (guaranteed by the caller), so the division is safe.
#[must_use]
pub fn constraint_stiffness(alpha: Real, dt_sub_sq: Real) -> Real {
    if alpha > 0.0 {
        (1.0 / (alpha * dt_sub_sq)).min(MAX_STIFFNESS)
    } else {
        MAX_STIFFNESS
    }
}

/// Accumulates one distance constraint's gradient and (PSD-projected) Hessian
/// contribution for the vertex at `x`, connected to `other` with rest length
/// `rest` and stiffness `k`.
///
/// The Hessian uses the standard positive-semidefinite spring form
/// `k·nnᵀ + k·max(0, 1 - rest/len)·(I - nnᵀ)`, which drops the indefinite part
/// when the constraint is compressed (`len < rest`) so the per-vertex Newton
/// step stays a descent direction. When `one_sided` is set (LRA / tether), a
/// constraint that is slack or at rest (`len <= rest`) contributes nothing,
/// matching the XPBD over-extension gate so an anchor never yanks slack cloth
/// inward.
pub fn accumulate_constraint(
    grad: &mut Vec3,
    hess: &mut SweepHessian,
    x: Vec3,
    other: Vec3,
    rest: Real,
    k: Real,
    one_sided: bool,
) {
    let d = x - other;
    let len_sq = d.length_squared();
    if len_sq < EPS_LEN_SQ {
        return;
    }
    let len = len_sq.sqrt();
    if one_sided && len <= rest {
        return;
    }
    let n = d * (1.0 / len);
    *grad += n * (k * (len - rest));
    let tangential = (1.0 - rest / len).max(0.0);
    // k·tangential·(I - nnᵀ) + k·nnᵀ, regrouped as k·tangential·I plus the
    // remaining k·(1 - tangential)·nnᵀ.
    *hess = hess.add(SweepHessian::scaled_identity(k * tangential));
    *hess = hess.add(SweepHessian::scaled_outer(n, k * (1.0 - tangential)));
}

/// A single incident distance constraint seen from the vertex being relaxed:
/// the neighbour position, the rest length, the already-resolved stiffness `k`,
/// and whether the constraint is one-sided (LRA / tether).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct IncidentConstraint {
    /// Current position of the other endpoint.
    pub other: Vec3,
    /// Rest (unstressed) separation.
    pub rest: Real,
    /// VBD energy stiffness for this substep (see [`constraint_stiffness`]).
    pub k: Real,
    /// Whether the constraint only resists over-extension (LRA / tether).
    pub one_sided: bool,
}

/// Computes one exact per-vertex Newton step `dx = H⁻¹ g` for a free vertex.
///
/// `x` is the vertex position, `target` its inertial prediction
/// `y = x_prev + v·dt + g·dt²`, `inverse_mass` its inverse mass (must be `> 0`;
/// pinned vertices are handled by the caller), `dt_sub_sq` the squared substep,
/// and `incident` the vertex's constraints in a fixed order (the render CSR
/// adjacency order, which the GPU upload preserves). Returns the position delta
/// to **subtract** from `x`, or [`None`] when the assembled Hessian is singular
/// or the solve is non-finite, in which case the vertex should not move.
#[must_use]
pub fn relax_delta<I>(
    x: Vec3,
    target: Vec3,
    inverse_mass: Real,
    dt_sub_sq: Real,
    incident: I,
) -> Option<Vec3>
where
    I: IntoIterator<Item = IncidentConstraint>,
{
    let mass = 1.0 / inverse_mass;
    let inertia = mass / dt_sub_sq;

    let mut grad = (x - target) * inertia;
    let mut hess = SweepHessian::scaled_identity(inertia);

    for c in incident {
        accumulate_constraint(&mut grad, &mut hess, x, c.other, c.rest, c.k, c.one_sided);
    }

    hess.solve(grad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn scaled_identity_solve_recovers_scaled_rhs() {
        let h = SweepHessian::scaled_identity(4.0);
        let x = h.solve(Vec3::new(8.0, -4.0, 2.0)).expect("invertible");
        assert_eq!(x, Vec3::new(2.0, -1.0, 0.5));
    }

    #[test]
    fn singular_hessian_returns_none() {
        assert!(SweepHessian::zero().solve(Vec3::ONE).is_none());
    }

    #[test]
    fn stiffness_saturates_for_rigid_and_tiny_compliance() {
        assert_eq!(constraint_stiffness(0.0, 1.0e-4), MAX_STIFFNESS);
        assert_eq!(constraint_stiffness(1.0e-30, 1.0e-4), MAX_STIFFNESS);
        // Finite compliance stays below the cap.
        assert!(constraint_stiffness(1.0, 1.0e-4) < MAX_STIFFNESS);
    }

    #[test]
    fn one_sided_slack_constraint_contributes_nothing() {
        let mut grad = Vec3::ZERO;
        let mut hess = SweepHessian::zero();
        // len = 1.0, rest = 2.0 -> slack, one-sided gate drops it.
        accumulate_constraint(
            &mut grad,
            &mut hess,
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            2.0,
            100.0,
            true,
        );
        assert_eq!(grad, Vec3::ZERO);
        assert_eq!(hess, SweepHessian::zero());
    }

    #[test]
    fn compressed_constraint_drops_indefinite_transverse_term() {
        // len = 1.0 < rest = 2.0: tangential = max(0, 1 - 2) = 0, so only the
        // normal (nnᵀ) term survives.
        let mut grad = Vec3::ZERO;
        let mut hess = SweepHessian::zero();
        accumulate_constraint(
            &mut grad,
            &mut hess,
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            2.0,
            10.0,
            false,
        );
        // n = (-1, 0, 0); grad = n * (k * (len - rest)) = (-1,0,0) * (10 * -1) = (10,0,0).
        assert_eq!(grad, Vec3::new(10.0, 0.0, 0.0));
        // Hessian = k * nnᵀ = 10 on the xx entry only.
        let probe = hess.solve(Vec3::new(10.0, 0.0, 0.0));
        // xx = 10, rest of diagonal 0 -> singular, so solve returns None.
        assert!(probe.is_none());
    }

    #[test]
    fn relax_single_edge_matches_hand_assembly() {
        // One free vertex at origin, inertial target at origin, one stretched
        // edge to (2,0,0) with rest 1.0. Hand-assemble and compare.
        let x = Vec3::ZERO;
        let target = Vec3::ZERO;
        let inverse_mass = 1.0;
        let dt_sub_sq = 1.0e-4;
        let k = constraint_stiffness(0.01, dt_sub_sq);
        let incident = vec![IncidentConstraint {
            other: Vec3::new(2.0, 0.0, 0.0),
            rest: 1.0,
            k,
            one_sided: false,
        }];

        let mut grad = (x - target) * (1.0 / inverse_mass / dt_sub_sq);
        let mut hess = SweepHessian::scaled_identity(1.0 / inverse_mass / dt_sub_sq);
        accumulate_constraint(
            &mut grad,
            &mut hess,
            x,
            Vec3::new(2.0, 0.0, 0.0),
            1.0,
            k,
            false,
        );
        let expected = hess.solve(grad);

        let got = relax_delta(x, target, inverse_mass, dt_sub_sq, incident);
        assert_eq!(got, expected);
    }
}
