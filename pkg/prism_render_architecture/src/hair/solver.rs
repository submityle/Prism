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
//! The per-vertex Newton step itself — the implicit-Euler inertia term, the
//! stretch-spring force / `PSD`-Hessian, and the `3x3` local solve — is **not**
//! re-implemented here: it is delegated to the shared
//! [`prism_physics_core::vbd`] kernel through
//! [`super::physics_bridge::solve_vbd_vertex`]. This module owns only the
//! poly-line topology, the Gauss-Seidel sweep order, and the strand-specific
//! point-to-midpoint bending pull.
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

        // 2. Gauss-Seidel vertex sweeps: one exact per-vertex Newton step,
        //    delegated to the shared physics-core VBD kernel so the stretch and
        //    bending constitutive math is never re-implemented here.
        let count = particles.len();
        for _ in 0..params.iterations {
            for i in 0..count {
                if particles[i].inverse_mass <= 0.0 {
                    continue; // pinned: fixed at its skinned pose.
                }
                let x = particles[i].position;
                let mass = 1.0 / particles[i].inverse_mass;

                // Segment to the next particle uses rest_lengths[i]; the segment
                // to the previous particle uses rest_lengths[i - 1].
                let next_neighbor = if i + 1 < count {
                    rest_lengths
                        .get(i)
                        .map(|&rest| (particles[i + 1].position, rest))
                } else {
                    None
                };
                let prev_neighbor = if i >= 1 {
                    rest_lengths
                        .get(i - 1)
                        .map(|&rest| (particles[i - 1].position, rest))
                } else {
                    None
                };
                // Bending pulls an interior vertex toward its neighbors' midpoint.
                let bending_pull = if bending > 0.0 && i >= 1 && i + 1 < count {
                    let mid = particles[i - 1]
                        .position
                        .add(particles[i + 1].position)
                        .scale(0.5);
                    Some((mid, bending))
                } else {
                    None
                };

                let dx = super::physics_bridge::solve_vbd_vertex(
                    x,
                    targets[i],
                    mass,
                    sub_dt,
                    prev_neighbor,
                    next_neighbor,
                    stretch,
                    bending_pull,
                );
                particles[i].position = x.add(dx);
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
