//! Multi-solver slot: XPBD baseline vs VBD high-fidelity cloth solve.
//!
//! The baseline garment integrator in [`super::dynamics`] is an XPBD solver:
//! cheap, unconditionally stable, and the right default for ordinary drape. But
//! *stiff* fabrics — leather, structured tailoring, starched or laminated
//! panels, tightly woven technical cloth — need much higher effective stiffness
//! than position-based projection delivers without either ballooning the
//! iteration count or going numerically soft (a rigid collar collapsing into a
//! rubber sheet). Production cloth reaches for a variational solver here
//! (design §6.1 "多求解器插槽": XPBD 基线 / VBD 高保真), mirroring the SIGGRAPH
//! 2024 Vertex Block Descent method used by engines such as UE5 `Chaos` Cloth,
//! NVIDIA `NvCloth` / `PhysX` Clothing, and Houdini `Vellum`, at the algorithm
//! level without reusing any of their code.
//!
//! This module adds that second solver: a Vertex Block Descent (VBD) mesh
//! integrator over the shared two-particle [`Constraint`] graph. VBD minimizes
//! the same backward-Euler incremental potential a full Newton solve would, but
//! *block-locally*: it sweeps the mesh vertices in a fixed Gauss-Seidel order
//! and takes one exact per-vertex Newton step against that vertex's own 3x3
//! Hessian each iteration. That makes very high stretch stiffness stable and
//! convergent (a stiff panel stops behaving like a soft spring) while staying a
//! simple, allocation-light, deterministic array-in / array-out kernel
//! (design §9) — no global matrix, no sparse solve, GPU-dispatch friendly.
//!
//! [`ClothSolverKind`] plus [`SolverSelection`] pick between the two solvers per
//! piece: ordinary garments stay on the cheap XPBD path, and only fabrics whose
//! authored stiffness crosses a threshold pay for VBD. The constraint graph,
//! particle layout, and one-sided (LRA / tether) semantics are shared verbatim
//! with the XPBD path, so a piece can switch solvers without re-authoring.

use alloc::vec;
use alloc::vec::Vec;

use super::{ClothParticle, Compliance, Constraint, Vec3, EPS_LEN_SQ};

/// Which cloth solver a piece uses.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ClothSolverKind {
    /// Position-based XPBD (see [`super::dynamics::solve_cloth`]). The cheap
    /// default: fast, stable, good enough for soft and medium drape.
    Xpbd,
    /// Vertex Block Descent (see [`solve_cloth_vbd`]). The high-fidelity path
    /// for stiff fabrics that XPBD would leave rubbery.
    Vbd,
}

/// Chooses a cloth solver from a piece's authored stretch stiffness.
///
/// The single knob is a threshold: a piece whose stretch stiffness is at or
/// above `vbd_stiffness_threshold` is stiff enough to be worth VBD; everything
/// softer stays on the cheaper XPBD path. Stiffness here is the direct energy
/// coefficient (larger is stiffer), the reciprocal of the [`Compliance`] the
/// solvers consume internally.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolverSelection {
    /// Stretch stiffness at or above which a piece is routed to VBD.
    pub vbd_stiffness_threshold: f32,
}

impl SolverSelection {
    /// Returns the solver to run for a piece with the given stretch stiffness.
    ///
    /// A non-finite stiffness (`NaN` / infinity) is treated as "not stiff" and
    /// stays on XPBD, so a bad authored value can never route a piece onto the
    /// expensive path by accident.
    #[must_use]
    pub fn choose(self, stretch_stiffness: f32) -> ClothSolverKind {
        if stretch_stiffness.is_finite() && stretch_stiffness >= self.vbd_stiffness_threshold {
            ClothSolverKind::Vbd
        } else {
            ClothSolverKind::Xpbd
        }
    }
}

/// Parameters for the VBD cloth solver.
///
/// Unlike XPBD, which projects positions directly, VBD integrates an implicit
/// backward-Euler step: each substep predicts an inertial target from the
/// current velocity and gravity, then relaxes it with a handful of Gauss-Seidel
/// Newton sweeps. Per-constraint stiffness is derived from the shared
/// [`Compliance`] so authored materials transfer between the two solvers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VbdParams {
    /// Semi-implicit substeps the frame `dt` is split into. Clamped to at least
    /// one by [`VbdParams::sanitized`].
    pub substeps: u32,
    /// Gauss-Seidel vertex sweeps per substep. VBD converges in a handful.
    /// Clamped to at least one by [`VbdParams::sanitized`].
    pub iterations: u32,
    /// Constant external acceleration (gravity), in world units per second².
    pub gravity: Vec3,
    /// Per-substep velocity damping in `0..=1` (`0` keeps all velocity, `1`
    /// removes it); values outside the range are clamped.
    pub damping: f32,
}

impl Default for VbdParams {
    /// A stiff-friendly default: eight substeps and eight sweeps under Earth
    /// gravity with light numerical damping.
    fn default() -> Self {
        Self {
            substeps: 8,
            iterations: 8,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            damping: 0.02,
        }
    }
}

impl VbdParams {
    /// Returns a copy with the counts forced to at least one and the damping
    /// clamped into `0..=1`, so a caller can hand raw authored values straight
    /// to [`solve_cloth_vbd`] without tripping a divide-by-zero or a runaway
    /// negative damping that would *inject* energy.
    #[must_use]
    pub fn sanitized(self) -> Self {
        Self {
            substeps: self.substeps.max(1),
            iterations: self.iterations.max(1),
            gravity: self.gravity,
            damping: self.damping.clamp(0.0, 1.0),
        }
    }
}

/// A symmetric 3x3 solve is skipped when the determinant is below this, so a
/// degenerate (singular) Hessian never divides by ~0.
const EPS_DET: f32 = 1.0e-20;
/// Upper bound on per-constraint stiffness. A perfectly rigid constraint
/// (`compliance == 0`) or a vanishingly small compliance would otherwise imply
/// an infinite stiffness; capping it keeps the Hessian finite while still
/// dominating the inertial term by many orders of magnitude, which reads as
/// "rigid" to the per-vertex Newton step.
const MAX_STIFFNESS: f32 = 1.0e9;

/// A dense 3x3 matrix in row-major order; used only for per-vertex Hessians.
#[derive(Clone, Copy)]
struct Mat3 {
    /// Row-major entries `[m00, m01, m02, m10, m11, m12, m20, m21, m22]`.
    m: [f32; 9],
}

impl Mat3 {
    /// A scalar multiple of the identity.
    fn scaled_identity(s: f32) -> Self {
        let mut m = [0.0; 9];
        m[0] = s;
        m[4] = s;
        m[8] = s;
        Self { m }
    }

    /// The outer product `v vᵀ` scaled by `s`.
    fn scaled_outer(v: Vec3, s: f32) -> Self {
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
    fn add(self, rhs: Self) -> Self {
        let mut m = [0.0; 9];
        for (out, (a, b)) in m.iter_mut().zip(self.m.iter().zip(rhs.m.iter())) {
            *out = a + b;
        }
        Self { m }
    }

    /// Solves `self * x = rhs` by explicit cofactor inversion. Returns `None`
    /// when the matrix is (near-)singular or the result is non-finite, so the
    /// caller can simply not move the vertex this sweep instead of producing a
    /// `NaN` position.
    fn solve(self, rhs: Vec3) -> Option<Vec3> {
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

/// Converts an XPBD [`Compliance`] into the VBD energy stiffness `k` for one
/// substep of size `dt_sub`.
///
/// XPBD compliance `α` relates to stiffness by `k = 1 / (α · dt²)`. A rigid
/// constraint (`α == 0`) or a tiny compliance would blow that up, so the result
/// is capped at [`MAX_STIFFNESS`]: still stiff enough to dominate inertia, but
/// finite. `dt_sub` is always positive here (guaranteed by the caller), so the
/// division is safe.
fn constraint_stiffness(compliance: Compliance, dt_sub_sq: f32) -> f32 {
    let alpha = compliance.value();
    if alpha > 0.0 {
        (1.0 / (alpha * dt_sub_sq)).min(MAX_STIFFNESS)
    } else {
        MAX_STIFFNESS
    }
}

/// Builds the per-vertex adjacency: `out[i]` lists the indices (into
/// `constraints`) of every constraint that touches particle `i`.
///
/// The build is `O(constraints)` — one pass appending each constraint to both
/// endpoints — and stores constraint indices (not copies) so the sweep reads
/// the live `constraints` slice. Endpoints outside `particle_count`, or a
/// degenerate self-constraint (`a == b`), are skipped so the sweep never
/// indexes out of bounds.
fn build_adjacency(constraints: &[Constraint], particle_count: usize) -> Vec<Vec<u32>> {
    let mut adjacency: Vec<Vec<u32>> = vec![Vec::new(); particle_count];
    for (index, constraint) in constraints.iter().enumerate() {
        let a = constraint.a as usize;
        let b = constraint.b as usize;
        if a == b || a >= particle_count || b >= particle_count {
            continue;
        }
        adjacency[a].push(index as u32);
        adjacency[b].push(index as u32);
    }
    adjacency
}

/// Accumulates one distance constraint's gradient and (PSD-projected) Hessian
/// contribution for the vertex at `x`, connected to `other` with rest length
/// `rest` and stiffness `k`.
///
/// The Hessian uses the standard positive-semidefinite spring form
/// `k·nnᵀ + k·max(0, 1 - rest/len)·(I - nnᵀ)`, which drops the indefinite part
/// when the constraint is compressed (`len < rest`) so the per-vertex Newton
/// step stays a descent direction and the sweep never blows up. When
/// `one_sided` is set (LRA / tether), a constraint that is slack or at rest
/// (`len <= rest`) contributes nothing, matching the XPBD path's over-extension
/// gate so an anchor never yanks slack cloth inward.
fn accumulate_constraint(
    grad: &mut Vec3,
    hess: &mut Mat3,
    x: Vec3,
    other: Vec3,
    rest: f32,
    k: f32,
    one_sided: bool,
) {
    let d = x.sub(other);
    let len_sq = d.length_squared();
    if len_sq < EPS_LEN_SQ {
        return;
    }
    let len = len_sq.sqrt();
    if one_sided && len <= rest {
        return;
    }
    let n = d.scale(1.0 / len);
    *grad = grad.add(n.scale(k * (len - rest)));
    let tangential = (1.0 - rest / len).max(0.0);
    // k·tangential·(I - nnᵀ) + k·nnᵀ, regrouped as k·tangential·I plus the
    // remaining k·(1 - tangential)·nnᵀ.
    *hess = hess.add(Mat3::scaled_identity(k * tangential));
    *hess = hess.add(Mat3::scaled_outer(n, k * (1.0 - tangential)));
}

/// Advances a cloth patch by `dt` seconds in place with the VBD solver.
///
/// `particles` is the sim-mesh particle array; `constraints` is the shared
/// two-particle distance graph (stretch / bend / shear / LRA / tether) — the
/// same list the XPBD path in [`super::dynamics`] consumes, so no separate
/// authoring is required. `params` tunes substep and sweep counts, gravity, and
/// damping; it is sanitized internally, so raw authored values are safe.
///
/// Pinned particles (`inverse_mass <= 0`) are held exactly in place. Constraint
/// endpoints that fall outside `particles`, or that name the same particle
/// twice, are skipped rather than panicking. A `dt` at or below zero, an empty
/// particle array, or a non-finite `dt` is a no-op (a paused or first frame), so
/// velocity recovery never divides by zero.
///
/// Each substep predicts an inertial target `y = x + v·dt + g·dt²` from the
/// current (damped) velocity, runs `iterations` Gauss-Seidel sweeps that take
/// one exact per-vertex Newton step against the vertex's inertia + constraint
/// Hessian, then recovers `v = (x - x_prev)/dt`. The vertex order and the
/// per-vertex constraint order are fixed, so the result is deterministic given
/// identical inputs.
pub fn solve_cloth_vbd(
    particles: &mut [ClothParticle],
    constraints: &[Constraint],
    params: VbdParams,
    dt: f32,
) {
    if particles.is_empty() || dt <= 0.0 || !dt.is_finite() {
        return;
    }
    let params = params.sanitized();
    let substeps = params.substeps;
    let iterations = params.iterations;
    let dt_sub = dt / substeps as f32;
    let dt_sub_sq = dt_sub * dt_sub;
    let inv_dt_sub = 1.0 / dt_sub;
    let retain = 1.0 - params.damping;
    let gravity_step = params.gravity.scale(dt_sub_sq);

    let count = particles.len();
    let adjacency = build_adjacency(constraints, count);

    // Scratch buffers reused across substeps: inertial targets and the
    // pre-solve positions used for velocity recovery.
    let mut targets: Vec<Vec3> = vec![Vec3::ZERO; count];
    let mut previous: Vec<Vec3> = vec![Vec3::ZERO; count];

    for _ in 0..substeps {
        // 1. Predict the inertial target y = x + v·retain·dt + g·dt² from the
        //    current velocity and snapshot the pre-solve position.
        for (i, particle) in particles.iter().enumerate() {
            previous[i] = particle.position;
            let velocity = particle.velocity.scale(retain);
            targets[i] = particle
                .position
                .add(velocity.scale(dt_sub))
                .add(gravity_step);
        }

        // 2. Gauss-Seidel vertex sweeps: one exact Newton step per free vertex.
        for _ in 0..iterations {
            for i in 0..count {
                if particles[i].is_pinned() {
                    continue; // pinned: fixed at its anim-driven pose.
                }
                let x = particles[i].position;
                let mass = 1.0 / particles[i].inverse_mass;
                let inertia = mass / dt_sub_sq;

                let mut grad = x.sub(targets[i]).scale(inertia);
                let mut hess = Mat3::scaled_identity(inertia);

                for &c_index in &adjacency[i] {
                    let constraint = constraints[c_index as usize];
                    let other_index = if constraint.a as usize == i {
                        constraint.b as usize
                    } else {
                        constraint.a as usize
                    };
                    let k = constraint_stiffness(constraint.compliance, dt_sub_sq);
                    accumulate_constraint(
                        &mut grad,
                        &mut hess,
                        x,
                        particles[other_index].position,
                        constraint.rest_length,
                        k,
                        constraint.kind.is_one_sided(),
                    );
                }

                if let Some(delta) = hess.solve(grad) {
                    particles[i].position = x.sub(delta);
                }
            }
        }

        // 3. Recover velocity from the position delta; pinned vertices stay put.
        for (i, particle) in particles.iter_mut().enumerate() {
            if particle.is_pinned() {
                particle.velocity = Vec3::ZERO;
                continue;
            }
            particle.velocity = particle.position.sub(previous[i]).scale(inv_dt_sub);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cloth::ConstraintKind;
    use alloc::vec;
    use alloc::vec::Vec;

    /// A stretch constraint with the given compliance between two indices.
    fn stretch(a: u32, b: u32, rest: f32, compliance: f32) -> Constraint {
        Constraint {
            a,
            b,
            rest_length: rest,
            compliance: Compliance(compliance),
            kind: ConstraintKind::Stretch,
        }
    }

    /// The default solver parameters used by most tests, with explicit fields.
    fn params(substeps: u32, iterations: u32, damping: f32) -> VbdParams {
        VbdParams {
            substeps,
            iterations,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            damping,
        }
    }

    /// A horizontal chain of `n` particles, unit spacing, root (index 0) pinned.
    fn horizontal_chain(n: usize) -> (Vec<ClothParticle>, Vec<Constraint>) {
        let mut particles: Vec<ClothParticle> = Vec::with_capacity(n);
        particles.push(ClothParticle::pinned(Vec3::ZERO));
        for i in 1..n {
            particles.push(ClothParticle::new(Vec3::new(i as f32, 0.0, 0.0), 1.0));
        }
        (particles, Vec::new())
    }

    #[test]
    fn selection_routes_stiff_to_vbd_soft_to_xpbd() {
        let s = SolverSelection {
            vbd_stiffness_threshold: 100.0,
        };
        assert_eq!(s.choose(10.0), ClothSolverKind::Xpbd);
        assert_eq!(s.choose(100.0), ClothSolverKind::Vbd);
        assert_eq!(s.choose(1000.0), ClothSolverKind::Vbd);
    }

    #[test]
    fn selection_treats_non_finite_stiffness_as_soft() {
        let s = SolverSelection {
            vbd_stiffness_threshold: 100.0,
        };
        assert_eq!(s.choose(f32::NAN), ClothSolverKind::Xpbd);
        assert_eq!(s.choose(f32::INFINITY), ClothSolverKind::Xpbd);
    }

    #[test]
    fn sanitized_clamps_counts_and_damping() {
        let raw = VbdParams {
            substeps: 0,
            iterations: 0,
            gravity: Vec3::new(0.0, -1.0, 0.0),
            damping: 5.0,
        };
        let s = raw.sanitized();
        assert_eq!(s.substeps, 1);
        assert_eq!(s.iterations, 1);
        assert!((s.damping - 1.0).abs() < 1.0e-7);
        let neg = VbdParams {
            substeps: 3,
            iterations: 4,
            gravity: Vec3::ZERO,
            damping: -2.0,
        };
        assert!(neg.sanitized().damping.abs() < 1.0e-7);
    }

    #[test]
    fn empty_and_degenerate_inputs_are_no_ops() {
        let mut empty: Vec<ClothParticle> = Vec::new();
        solve_cloth_vbd(&mut empty, &[], params(4, 4, 0.0), 1.0 / 60.0);
        assert!(empty.is_empty());

        // Non-positive / non-finite dt leaves everything untouched.
        let (mut chain, cons) = horizontal_chain(3);
        let before: Vec<Vec3> = chain.iter().map(|p| p.position).collect();
        solve_cloth_vbd(&mut chain, &cons, params(4, 4, 0.0), 0.0);
        solve_cloth_vbd(&mut chain, &cons, params(4, 4, 0.0), f32::NAN);
        for (p, b) in chain.iter().zip(before.iter()) {
            assert!(p.position.sub(*b).length_squared() < 1.0e-12);
        }
    }

    #[test]
    fn out_of_range_constraints_are_skipped() {
        let mut particles = vec![
            ClothParticle::pinned(Vec3::ZERO),
            ClothParticle::new(Vec3::new(1.0, 0.0, 0.0), 1.0),
        ];
        // Endpoints beyond the array and a self-constraint must not panic.
        let cons = vec![
            stretch(0, 9, 1.0, 0.0),
            stretch(5, 6, 1.0, 0.0),
            stretch(1, 1, 1.0, 0.0),
        ];
        solve_cloth_vbd(&mut particles, &cons, params(4, 4, 0.0), 1.0 / 60.0);
        for p in &particles {
            assert!(p.position.x.is_finite() && p.position.y.is_finite());
        }
    }

    #[test]
    fn pinned_root_never_moves() {
        let (mut chain, _) = horizontal_chain(4);
        let cons = vec![
            stretch(0, 1, 1.0, 0.0),
            stretch(1, 2, 1.0, 0.0),
            stretch(2, 3, 1.0, 0.0),
        ];
        let root = chain[0].position;
        for _ in 0..30 {
            solve_cloth_vbd(&mut chain, &cons, params(8, 8, 0.02), 1.0 / 60.0);
        }
        assert!(chain[0].position.sub(root).length_squared() < 1.0e-12);
    }

    #[test]
    fn deterministic_across_identical_runs() {
        let cons = vec![
            stretch(0, 1, 1.0, 0.0),
            stretch(1, 2, 1.0, 0.0),
            stretch(2, 3, 1.0, 0.0),
        ];
        let run = || {
            let (mut chain, _) = horizontal_chain(4);
            for _ in 0..50 {
                solve_cloth_vbd(&mut chain, &cons, params(8, 8, 0.02), 1.0 / 60.0);
            }
            chain.iter().map(|p| p.position).collect::<Vec<_>>()
        };
        let a = run();
        let b = run();
        for (pa, pb) in a.iter().zip(b.iter()) {
            assert!((pa.x - pb.x).abs() < 1.0e-20);
            assert!((pa.y - pb.y).abs() < 1.0e-20);
            assert!((pa.z - pb.z).abs() < 1.0e-20);
        }
    }

    /// Runs a hanging chain (root pinned, gravity down) to rest and returns its
    /// total arc length: the sum of the settled segment lengths. Arc length is
    /// the direct readout of stretch — it equals the summed rest length when the
    /// springs hold and grows as they give — unlike straight-line end-to-end
    /// distance, which also shrinks as a stiff chain merely reorients.
    fn settled_arc_length(compliance: f32) -> f32 {
        let n = 4;
        let (mut chain, _) = horizontal_chain(n);
        let cons = vec![
            stretch(0, 1, 1.0, compliance),
            stretch(1, 2, 1.0, compliance),
            stretch(2, 3, 1.0, compliance),
        ];
        // The soft chain is a low-frequency pendulum, so settle generously.
        for _ in 0..2000 {
            solve_cloth_vbd(&mut chain, &cons, params(4, 12, 0.2), 1.0 / 60.0);
        }
        let mut arc = 0.0;
        for i in 0..n - 1 {
            arc += chain[i + 1].position.sub(chain[i].position).length();
        }
        arc
    }

    #[test]
    fn rigid_material_does_not_sag_like_a_soft_spring() {
        // The chain's summed rest length is 3.0. A near-rigid chain (compliance
        // ~ 0) must hold that length under load; a soft chain stretches well
        // past it. This is exactly the failure VBD exists to prevent: a stiff
        // panel going rubbery.
        let rigid = settled_arc_length(0.0);
        let soft = settled_arc_length(800.0);
        assert!(
            rigid < soft,
            "rigid arc {rigid} should stay under soft arc {soft}"
        );
        // The rigid chain stays essentially at its rest length ...
        assert!((rigid - 3.0).abs() < 0.03, "rigid arc {rigid} sagged");
        // ... while the soft chain overshoots it substantially.
        assert!(soft > 3.2, "soft arc {soft} did not stretch");
    }

    #[test]
    fn one_sided_tether_only_pulls_when_over_extended() {
        // A tether from a pinned anchor at the origin to a free particle sitting
        // *inside* the rest radius must not drag it inward.
        let mut particles = vec![
            ClothParticle::pinned(Vec3::ZERO),
            ClothParticle::new(Vec3::new(0.5, 0.0, 0.0), 1.0),
        ];
        let tether = Constraint {
            a: 0,
            b: 1,
            rest_length: 2.0,
            compliance: Compliance::RIGID,
            kind: ConstraintKind::Tether,
        };
        // Zero gravity so only the tether could move it.
        let p = VbdParams {
            substeps: 4,
            iterations: 8,
            gravity: Vec3::ZERO,
            damping: 0.0,
        };
        let before = particles[1].position;
        solve_cloth_vbd(&mut particles, &[tether], p, 1.0 / 60.0);
        assert!(
            particles[1].position.sub(before).length() < 1.0e-4,
            "slack tether moved the particle"
        );
    }

    #[test]
    fn chain_stays_finite_over_a_long_run() {
        let (mut chain, _) = horizontal_chain(5);
        let cons = vec![
            stretch(0, 1, 1.0, 0.0),
            stretch(1, 2, 1.0, 0.0),
            stretch(2, 3, 1.0, 0.0),
            stretch(3, 4, 1.0, 0.0),
        ];
        for _ in 0..600 {
            solve_cloth_vbd(&mut chain, &cons, params(8, 8, 0.05), 1.0 / 60.0);
        }
        for p in &chain {
            assert!(p.position.x.is_finite());
            assert!(p.position.y.is_finite());
            assert!(p.position.z.is_finite());
        }
        // A rigid 4-segment chain must not have exploded far past its 4.0 rest
        // length.
        let total = chain[4].position.sub(chain[0].position).length();
        assert!(total < 6.0, "chain length {total} exploded");
    }
}
