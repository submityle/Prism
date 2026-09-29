//! Multi-solver slot: XPBD baseline vs VBD high-fidelity.
//!
//! The baseline strand integrator in [`super::dynamics`] is an XPBD solver:
//! cheap, unconditionally stable, and the right default for ordinary hair. But
//! *stiff* grooms — braids, dreadlocks, wax/gel-set styling — need much higher
//! effective stiffness than position-based projection delivers without either
//! ballooning the iteration count or going numerically soft. Production hair
//! reaches for a variational solver here (design §6.7 "多求解器插槽": XPBD 基线
//! / VBD 高保真).
//!
//! This module adds that second solver: a Vertex Block Descent (VBD) strand
//! integrator. VBD minimizes the same backward-Euler incremental potential a
//! full Newton solve would, but block-locally: it sweeps the vertices in a
//! Gauss-Seidel order and takes one exact per-vertex Newton step against that
//! vertex's own 3x3 Hessian each iteration. That makes very high stretch
//! stiffness stable and convergent (a braid stops looking like a soft spring)
//! while staying a simple, allocation-light, deterministic array-in/array-out
//! kernel (design §9) — no global matrix, no sparse solve.
//!
//! [`HairSolverKind`] plus [`SolverSelection`] pick between the two per groom:
//! ordinary hair stays on the cheap XPBD path, and only grooms whose authored
//! stiffness crosses a threshold pay for VBD. Collision projection reuses the
//! shared [`super::collision`] service, exactly as the XPBD path does.

use alloc::vec::Vec;

use super::collision::{resolve_strand_collisions, Collider};
use super::dynamics::{StrandParticle, Vec3};

/// Which strand solver a groom uses.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum HairSolverKind {
    /// Position-based XPBD (see [`super::dynamics::simulate_strand`]). The cheap
    /// default: fast, stable, good enough for soft and medium hair.
    Xpbd,
    /// Vertex Block Descent (see [`simulate_strand_vbd`]). The high-fidelity
    /// path for stiff styling that XPBD would leave rubbery.
    Vbd,
}

/// Chooses a strand solver from a groom's authored stretch stiffness.
///
/// The single knob is a threshold: a groom whose stretch stiffness is at or
/// above `vbd_stiffness_threshold` is stiff enough to be worth VBD; everything
/// softer stays on the cheaper XPBD path.
#[derive(Clone, Copy, Debug)]
pub struct SolverSelection {
    /// Stretch stiffness at or above which a groom is routed to VBD.
    pub vbd_stiffness_threshold: f32,
}

impl SolverSelection {
    /// Returns the solver to run for a groom with the given stretch stiffness.
    ///
    /// A non-finite stiffness is treated as "not stiff" and stays on XPBD, so a
    /// bad authored value can never route a groom onto the expensive path by
    /// accident.
    #[must_use]
    pub fn choose(self, stretch_stiffness: f32) -> HairSolverKind {
        if stretch_stiffness.is_finite() && stretch_stiffness >= self.vbd_stiffness_threshold {
            HairSolverKind::Vbd
        } else {
            HairSolverKind::Xpbd
        }
    }
}

/// Parameters for the VBD strand solver.
///
/// Unlike XPBD's compliance (an inverse stiffness), VBD takes stiffness
/// *directly* as an energy weight, so larger values mean stiffer. Stiffness is
/// an absolute energy coefficient, not a `0..=1` blend.
#[derive(Clone, Copy, Debug)]
pub struct VbdParams {
    /// Uniform acceleration applied to every free particle (world units/s^2).
    pub gravity: Vec3,
    /// Frame time step advanced by one call (seconds).
    pub dt: f32,
    /// Semi-implicit substeps `dt` is split into.
    pub substeps: u32,
    /// Gauss-Seidel vertex sweeps per substep. VBD converges in a handful.
    pub iterations: u32,
    /// Edge (stretch) energy stiffness; large values hold segment length hard.
    pub stretch_stiffness: f32,
    /// Bending energy stiffness pulling each interior vertex toward the midpoint
    /// of its neighbors (straightens the strand). `0` disables bending.
    pub bending_stiffness: f32,
    /// Velocity retention fraction in `0..=1` (`0` keeps all velocity, `1`
    /// removes it); models drag / numerical damping.
    pub damping: f32,
}

/// Vectors shorter than the square root of this are treated as zero-length.
const EPS_LEN_SQ: f32 = 1.0e-24;
/// A symmetric 3x3 solve is skipped when the determinant is below this, so a
/// degenerate (singular) Hessian never divides by ~0.
const EPS_DET: f32 = 1.0e-20;

/// A dense 3x3 matrix in row-major order; used only for per-vertex Hessians.
#[derive(Clone, Copy)]
struct Mat3 {
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

    fn add(self, rhs: Self) -> Self {
        let mut m = [0.0; 9];
        for (out, (a, b)) in m.iter_mut().zip(self.m.iter().zip(rhs.m.iter())) {
            *out = a + b;
        }
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

    /// Solves `self * x = rhs` by explicit cofactor inversion. Returns `None`
    /// when the matrix is (near-)singular so the caller can simply not move the
    /// vertex this sweep instead of producing a non-finite position.
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
        // Cofactor (adjugate) rows; the inverse is adjugateᵀ / det.
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

/// Accumulates one spring's gradient and (PSD-projected) Hessian contribution
/// for the vertex at `x`, connected to `other` with rest length `rest` and
/// stiffness `k`.
///
/// The Hessian uses the standard positive-semidefinite spring form
/// `k·nnᵀ + k·max(0, 1 - rest/len)·(I - nnᵀ)`, which drops the indefinite part
/// when the spring is compressed (`len < rest`) so the per-vertex Newton step
/// stays a descent direction and the sweep never blows up.
fn accumulate_spring(grad: &mut Vec3, hess: &mut Mat3, x: Vec3, other: Vec3, rest: f32, k: f32) {
    let d = x.sub(other);
    let len_sq = d.length_squared();
    if len_sq < EPS_LEN_SQ {
        return;
    }
    let len = len_sq.sqrt();
    let n = d.scale(1.0 / len);
    *grad = grad.add(n.scale(k * (len - rest)));
    let tangential = (1.0 - rest / len).max(0.0);
    // k·tangential·I + k·(1 - tangential)·nnᵀ
    *hess = hess.add(Mat3::scaled_identity(k * tangential));
    *hess = hess.add(Mat3::scaled_outer(n, k * (1.0 - tangential)));
}

/// Advances a single guide strand by `params.dt` with the VBD solver.
///
/// The contract matches [`super::dynamics::simulate_strand`]: `particles` is the
/// strand poly-line in root-to-tip order and is updated in place;
/// `rest_lengths[i]` is the rest length of the segment between particle `i` and
/// `i + 1`; `colliders` are projected once per substep (pass `&[]` for none).
/// Pinned particles (`inverse_mass <= 0`) are held exactly in place. The call is
/// a no-op for an empty strand, zero substeps, or a non-positive/non-finite
/// `dt`.
///
/// Each substep predicts an inertial target from the implicit velocity, then
/// runs `iterations` Gauss-Seidel sweeps that take one exact per-vertex Newton
/// step against the vertex's inertia + stretch + bending Hessian.
pub fn simulate_strand_vbd(
    particles: &mut [StrandParticle],
    rest_lengths: &[f32],
    colliders: &[Collider],
    params: VbdParams,
) {
    if particles.is_empty() || params.substeps == 0 || params.dt <= 0.0 || !params.dt.is_finite() {
        return;
    }

    let sub_dt = params.dt / params.substeps as f32;
    let sub_dt_sq = sub_dt * sub_dt;
    let velocity_retain = 1.0 - params.damping.clamp(0.0, 1.0);
    let stretch = params.stretch_stiffness.max(0.0);
    let bending = params.bending_stiffness.max(0.0);
    let gravity_step = params.gravity.scale(sub_dt_sq);

    // Reused inertial-target scratch, one entry per particle.
    let mut targets: Vec<Vec3> = Vec::with_capacity(particles.len());

    for _ in 0..params.substeps {
        // 1. Predict each free vertex's inertial target y = x + v·retain + g·dt²
        //    from the *current* implicit velocity, then snapshot prev = x so the
        //    next substep's velocity is measured from here.
        targets.clear();
        for particle in particles.iter() {
            let velocity = particle.position.sub(particle.prev_position);
            let y = particle
                .position
                .add(velocity.scale(velocity_retain))
                .add(gravity_step);
            targets.push(y);
        }
        for particle in particles.iter_mut() {
            particle.prev_position = particle.position;
        }

        // 2. Gauss-Seidel vertex sweeps: one exact Newton step per free vertex.
        let count = particles.len();
        for _ in 0..params.iterations {
            for i in 0..count {
                if particles[i].inverse_mass <= 0.0 {
                    continue; // pinned: fixed at its skinned pose.
                }
                let x = particles[i].position;
                let mass = 1.0 / particles[i].inverse_mass;
                let inertia = mass / sub_dt_sq;

                let mut grad = x.sub(targets[i]).scale(inertia);
                let mut hess = Mat3::scaled_identity(inertia);

                // Segment to the next particle uses rest_lengths[i].
                if i + 1 < count
                    && let Some(&rest) = rest_lengths.get(i)
                {
                    accumulate_spring(
                        &mut grad,
                        &mut hess,
                        x,
                        particles[i + 1].position,
                        rest,
                        stretch,
                    );
                }
                // Segment to the previous particle uses rest_lengths[i - 1].
                if i >= 1
                    && let Some(&rest) = rest_lengths.get(i - 1)
                {
                    accumulate_spring(
                        &mut grad,
                        &mut hess,
                        x,
                        particles[i - 1].position,
                        rest,
                        stretch,
                    );
                }
                // Bending: pull an interior vertex toward its neighbors' midpoint.
                if bending > 0.0 && i >= 1 && i + 1 < count {
                    let mid = particles[i - 1]
                        .position
                        .add(particles[i + 1].position)
                        .scale(0.5);
                    grad = grad.add(x.sub(mid).scale(bending));
                    hess = hess.add(Mat3::scaled_identity(bending));
                }

                if let Some(delta) = hess.solve(grad) {
                    particles[i].position = x.sub(delta);
                }
            }
        }

        // 3. Project out of body colliders once the interior solve has settled.
        resolve_strand_collisions(particles, colliders);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    fn selection() -> SolverSelection {
        SolverSelection {
            vbd_stiffness_threshold: 100.0,
        }
    }

    #[test]
    fn selection_routes_stiff_to_vbd_soft_to_xpbd() {
        let s = selection();
        assert_eq!(s.choose(10.0), HairSolverKind::Xpbd);
        assert_eq!(s.choose(100.0), HairSolverKind::Vbd);
        assert_eq!(s.choose(1000.0), HairSolverKind::Vbd);
    }

    #[test]
    fn selection_treats_non_finite_stiffness_as_soft() {
        let s = selection();
        assert_eq!(s.choose(f32::NAN), HairSolverKind::Xpbd);
        assert_eq!(s.choose(f32::INFINITY), HairSolverKind::Xpbd);
    }

    fn params(stretch: f32) -> VbdParams {
        VbdParams {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            dt: 1.0 / 60.0,
            substeps: 2,
            iterations: 8,
            stretch_stiffness: stretch,
            bending_stiffness: 0.0,
            damping: 0.0,
        }
    }

    fn two_segment_strand() -> Vec<StrandParticle> {
        vec![
            StrandParticle::pinned(Vec3::ZERO),
            StrandParticle::free(Vec3::new(1.0, 0.0, 0.0)),
            StrandParticle::free(Vec3::new(2.0, 0.0, 0.0)),
        ]
    }

    #[test]
    fn empty_and_degenerate_inputs_are_no_ops() {
        let mut empty: Vec<StrandParticle> = Vec::new();
        simulate_strand_vbd(&mut empty, &[], &[], params(1000.0));
        assert!(empty.is_empty());

        let mut one = vec![StrandParticle::free(Vec3::ZERO)];
        simulate_strand_vbd(&mut one, &[], &[], params(1000.0));
        // Single free vertex with no edges just falls under gravity a touch;
        // must stay finite and not panic.
        assert!(one[0].position.x.is_finite());
        assert!(one[0].position.y.is_finite());

        // Zero substeps / non-positive dt leave the strand untouched.
        let mut s = two_segment_strand();
        let before: Vec<Vec3> = s.iter().map(|p| p.position).collect();
        let mut p0 = params(1000.0);
        p0.substeps = 0;
        simulate_strand_vbd(&mut s, &[1.0, 1.0], &[], p0);
        for (p, b) in s.iter().zip(before.iter()) {
            assert!((p.position.sub(*b)).length_squared() < 1.0e-12);
        }
    }

    #[test]
    fn pinned_root_never_moves() {
        let mut s = two_segment_strand();
        let root = s[0].position;
        simulate_strand_vbd(&mut s, &[1.0, 1.0], &[], params(1000.0));
        assert!((s[0].position.sub(root)).length_squared() < 1.0e-12);
    }

    #[test]
    fn stiffer_strand_stretches_less_under_gravity() {
        let rest = [1.0_f32, 1.0_f32];

        // Run each strand for a while and measure how far the tip segment has
        // stretched past its rest length.
        fn tip_stretch(stretch: f32, rest: &[f32]) -> f32 {
            let mut s = two_segment_strand();
            let p = VbdParams {
                gravity: Vec3::new(0.0, -9.81, 0.0),
                dt: 1.0 / 60.0,
                substeps: 4,
                iterations: 12,
                stretch_stiffness: stretch,
                bending_stiffness: 0.0,
                damping: 0.2,
            };
            for _ in 0..120 {
                simulate_strand_vbd(&mut s, rest, &[], p);
            }
            let seg = s[2].position.sub(s[1].position).length();
            (seg - rest[1]).abs()
        }

        let soft = tip_stretch(50.0, &rest);
        let stiff = tip_stretch(5000.0, &rest);
        // A stiffer stretch energy must hold the segment closer to rest length.
        assert!(
            stiff < soft,
            "stiff stretch {stiff} should be < soft stretch {soft}"
        );
        // And all positions stay finite.
        let mut s = two_segment_strand();
        simulate_strand_vbd(&mut s, &rest, &[], params(5000.0));
        for p in &s {
            assert!(p.position.x.is_finite() && p.position.y.is_finite());
        }
    }

    #[test]
    fn strand_settles_and_stays_finite_over_many_frames() {
        let mut s = two_segment_strand();
        let rest = [1.0_f32, 1.0_f32];
        let mut p = params(2000.0);
        p.damping = 0.1;
        for _ in 0..300 {
            simulate_strand_vbd(&mut s, &rest, &[], p);
        }
        // Free tips should hang roughly below the root, never NaN/inf.
        for particle in &s {
            assert!(particle.position.x.is_finite());
            assert!(particle.position.y.is_finite());
            assert!(particle.position.z.is_finite());
        }
        // The chain should not have exploded far past its total rest length.
        let total = s[2].position.sub(s[0].position).length();
        assert!(total < 4.0, "chain length {total} exploded");
    }

    #[test]
    fn collider_pushes_strand_out() {
        use super::super::collision::Collider;
        let mut s = vec![
            StrandParticle::pinned(Vec3::new(0.0, 2.0, 0.0)),
            StrandParticle::free(Vec3::new(0.0, 1.0, 0.0)),
        ];
        // A sphere straddling the free vertex must push it to the surface.
        let sphere = Collider::Sphere {
            center: Vec3::new(0.0, 1.0, 0.0),
            radius: 0.5,
        };
        simulate_strand_vbd(&mut s, &[1.0], &[sphere], params(1000.0));
        let d = s[1].position.sub(Vec3::new(0.0, 1.0, 0.0)).length();
        assert!(d >= 0.5 - 1.0e-4, "vertex at {d} not pushed to radius 0.5");
    }
}
