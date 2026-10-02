//! XPBD substepping cloth solver with compliance and strain limiting.
//!
//! This is the per-frame dynamics core (design §3, stage 3), aligned with the
//! unified position-based kernel in the physics design (§3.3–§3.5). Each frame
//! is split into a fixed number of substeps; every substep predicts positions
//! from velocity and gravity, projects the colored constraint graph once, then
//! recovers velocity from the position delta. Substepping (many small steps,
//! one projection each) is what makes the solver stiff and time-step
//! independent, matching `Chaos` Cloth and Houdini `Vellum`.
//!
//! Everything here is deterministic array-in / array-out CPU math so it can be
//! golden-tested value by value:
//!
//! * A fixed substep count and a fixed color-batch order make the result
//!   bit-reproducible for networked / replay determinism.
//! * Constraints are projected per color batch: a batch shares no particle, so
//!   its projections are order-independent (Jacobi within a color, Gauss-Seidel
//!   across colors) and map onto a GPU dispatch unchanged.
//! * One-sided constraints (LRA / tether) only project when over-extended.
//! * A post-projection strain-limiting pass hard-clamps structural edge lengths
//!   so cloth never stretches like rubber.
//! * Body collision is applied through a caller-supplied hook, so the solver
//!   stays independent of the collision module (integration is wired at the
//!   subsystem level, not by cross-importing sibling modules).
//!
//! Per-frame vertex work is charged against the shared deformation budget via
//! [`crate::deformation::schedule`]; this module emits a request, it never owns
//! the budget.

use alloc::vec::Vec;

use super::{
    physics_bridge, ClothParticle, Constraint, ConstraintGraph, ConstraintKind, Vec3, EPS_LEN_SQ,
};
use crate::deformation::schedule::DeformationRequest;
use crate::deformation::{DeformationHandle, DeformationKind};

/// Tuning for one cloth solve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolverParams {
    /// Number of substeps per frame; more substeps stiffen the response and
    /// improve energy stability. Clamped to at least one.
    pub substeps: u32,
    /// Constraint-projection iterations per substep. One is the substep-heavy
    /// default; clamped to at least one.
    pub iterations: u32,
    /// Constant external acceleration (gravity), in world units per second².
    pub gravity: Vec3,
    /// Per-substep velocity damping in `0..=1` (0 keeps all velocity, 1 kills
    /// it); values outside the range are clamped.
    pub damping: f32,
    /// Maximum fractional stretch of a structural edge before the strain
    /// limiter clamps it (0.1 allows 10% stretch). Non-positive disables the
    /// limiter.
    pub strain_limit: f32,
}

impl Default for SolverParams {
    fn default() -> Self {
        Self {
            substeps: 8,
            iterations: 1,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            damping: 0.01,
            strain_limit: 0.1,
        }
    }
}

/// Advances a cloth patch by `dt` seconds in place.
///
/// `particles` is the sim-mesh particle array; `graph` is the colored
/// constraint graph from [`super::constraints`]. Pinned particles
/// (`inverse_mass <= 0`) are never moved by integration or projection, which is
/// how attachment points and anim-driven vertices stay locked. Constraint
/// endpoints that fall outside `particles` are skipped rather than panicking.
///
/// `dt` at or below zero is a no-op (a paused or first frame), so velocity
/// recovery never divides by zero.
pub fn solve_cloth(particles: &mut [ClothParticle], graph: &ConstraintGraph, params: SolverParams) {
    solve_cloth_with_collision(particles, graph, params, 1.0 / 60.0, |p| p.position);
}

/// Advances a cloth patch by an explicit `dt`, applying `collision` to every
/// particle after each substep's projection.
///
/// The `collision` hook receives one particle at a time (already integrated and
/// projected) and returns its corrected position; a no-op hook (`|p| p.position`
/// or ignoring the argument) disables collision. Keeping collision a caller
/// hook lets the body-proxy and self-collision code in [`super::collision`]
/// plug in at the subsystem level without this module importing it.
pub fn solve_cloth_with_collision<F>(
    particles: &mut [ClothParticle],
    graph: &ConstraintGraph,
    params: SolverParams,
    dt: f32,
    mut collision: F,
) where
    F: FnMut(ClothParticle) -> Vec3,
{
    if dt <= 0.0 || particles.is_empty() {
        return;
    }
    let substeps = params.substeps.max(1);
    let iterations = params.iterations.max(1);
    let dt_sub = dt / substeps as f32;
    let damping = params.damping.clamp(0.0, 1.0);
    let retain = 1.0 - damping;
    let gravity_step = params.gravity.scale(dt_sub);

    for _ in 0..substeps {
        // 1. Predict positions from damped velocity + gravity.
        let mut previous: Vec<Vec3> = Vec::with_capacity(particles.len());
        for particle in particles.iter_mut() {
            previous.push(particle.position);
            if particle.is_pinned() {
                continue;
            }
            particle.velocity = particle.velocity.scale(retain).add(gravity_step);
            particle.position = particle.position.add(particle.velocity.scale(dt_sub));
        }

        // 2. Project the colored constraint graph.
        for _ in 0..iterations {
            for batch in &graph.batches {
                for constraint in graph.batch(*batch) {
                    project_distance(particles, *constraint, dt_sub);
                }
            }
        }

        // 3. Strain limiting: hard-clamp structural edge overstretch.
        if params.strain_limit > 0.0 {
            apply_strain_limit(particles, graph, params.strain_limit);
        }

        // 4. Body / self collision hook.
        for particle in particles.iter_mut() {
            if particle.is_pinned() {
                continue;
            }
            particle.position = collision(*particle);
        }

        // 5. Recover velocity from the position delta.
        for (particle, &prev) in particles.iter_mut().zip(previous.iter()) {
            if particle.is_pinned() {
                particle.velocity = Vec3::ZERO;
                continue;
            }
            particle.velocity = particle.position.sub(prev).scale(1.0 / dt_sub);
        }
    }
}

/// Projects one distance constraint, moving its endpoints toward the rest
/// length with XPBD compliance. One-sided constraints only pull when the
/// current distance exceeds the rest length.
fn project_distance(particles: &mut [ClothParticle], constraint: Constraint, dt_sub: f32) {
    let a = constraint.a as usize;
    let b = constraint.b as usize;
    if a == b || a >= particles.len() || b >= particles.len() {
        return;
    }
    let pa = particles[a];
    let pb = particles[b];
    let wa = inverse_mass(pa);
    let wb = inverse_mass(pb);
    let w_sum = wa + wb;
    if w_sum <= 0.0 {
        return;
    }
    let delta = pa.position.sub(pb.position);
    let dist_sq = delta.length_squared();
    if dist_sq <= EPS_LEN_SQ {
        return;
    }
    let dist = dist_sq.sqrt();
    let error = dist - constraint.rest_length;
    if constraint.kind.is_one_sided() && error <= 0.0 {
        return;
    }
    // Delegate the XPBD projection to the single authoritative physics step
    // (`prism_physics_core::soft::constraint::project_distance_constraint`).
    // A fresh (zero) multiplier per call reproduces this solver's stateless
    // per-projection convention; the physics engine owns the arithmetic so the
    // CPU golden, this path, and the `cloth_sim.wesl` GPU twin stay bit-identical
    // (separation direction `delta / dist`, component-wise). The render-side
    // bounds, mass, degeneracy, and one-sided guards above gate the call so the
    // delegated step only runs on the exact pairs the golden projects.
    let mut positions = [
        physics_bridge::to_glam(pa.position),
        physics_bridge::to_glam(pb.position),
    ];
    let inverse_masses = [wa, wb];
    let _ = prism_physics_core::soft::constraint::project_distance_constraint(
        &mut positions,
        &inverse_masses,
        0,
        1,
        constraint.rest_length,
        constraint.compliance.value(),
        0.0,
        dt_sub,
    );
    particles[a].position = physics_bridge::from_glam(positions[0]);
    particles[b].position = physics_bridge::from_glam(positions[1]);
}

/// Hard-clamps structural (stretch) edge lengths to `1 + limit` of their rest
/// length after projection, so a fast pull can never stretch cloth like rubber.
/// The clamp is mass-weighted and never moves pinned particles.
fn apply_strain_limit(particles: &mut [ClothParticle], graph: &ConstraintGraph, limit: f32) {
    let max_scale = 1.0 + limit;
    for batch in &graph.batches {
        for constraint in graph.batch(*batch) {
            if constraint.kind != ConstraintKind::Stretch {
                continue;
            }
            let a = constraint.a as usize;
            let b = constraint.b as usize;
            if a == b || a >= particles.len() || b >= particles.len() {
                continue;
            }
            let pa = particles[a];
            let pb = particles[b];
            let wa = inverse_mass(pa);
            let wb = inverse_mass(pb);
            let w_sum = wa + wb;
            if w_sum <= 0.0 {
                continue;
            }
            let max_len = constraint.rest_length * max_scale;
            let delta = pa.position.sub(pb.position);
            let dist_sq = delta.length_squared();
            let max_sq = max_len * max_len;
            if dist_sq <= max_sq || dist_sq <= EPS_LEN_SQ {
                continue;
            }
            let dist = dist_sq.sqrt();
            let excess = dist - max_len;
            let direction = delta.scale(1.0 / dist);
            let correction = direction.scale(excess);
            particles[a].position = pa.position.sub(correction.scale(wa / w_sum));
            particles[b].position = pb.position.add(correction.scale(wb / w_sum));
        }
    }
}

/// Effective inverse mass of a particle: zero when pinned.
fn inverse_mass(particle: ClothParticle) -> f32 {
    if particle.is_pinned() {
        0.0
    } else {
        particle.inverse_mass
    }
}

/// Copies the current particle positions into a fresh vector — the deformed
/// sim-mesh vertices the render mesh embeds against and the deformation cache
/// stores.
#[must_use]
pub fn extract_positions(particles: &[ClothParticle]) -> Vec<Vec3> {
    particles.iter().map(|p| p.position).collect()
}

/// Builds the deformation-budget request for a cloth solve.
///
/// Cloth dynamics never runs its own scheduler: it emits a
/// [`DeformationRequest`] tagged [`DeformationKind::Cloth`] and lets
/// [`crate::deformation::schedule::plan_deformations`] arbitrate it against
/// every other subsystem. The simulated particle count is the dominant
/// per-frame cost and the deformed geometry needs a BLAS refit for ray tracing,
/// so `needs_blas_refit` is set.
#[must_use]
pub fn cloth_solve_request(
    handle: DeformationHandle,
    particle_count: u32,
    priority: u32,
) -> DeformationRequest {
    DeformationRequest {
        handle,
        kind: DeformationKind::Cloth,
        vertex_count: particle_count,
        priority,
        needs_blas_refit: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cloth::constraints::{
        build_grid_constraints, color_constraints, ClothGrid, GridConstraintParams,
    };
    use crate::cloth::Compliance;
    use crate::deformation::schedule::plan_deformations;
    use crate::deformation::DeformationBudget;

    fn rigid_params() -> GridConstraintParams {
        GridConstraintParams {
            warp: Compliance::RIGID,
            weft: Compliance::RIGID,
            shear: Compliance(0.001),
            bend: Compliance(0.01),
        }
    }

    #[test]
    fn pinned_particles_never_move() {
        let mut particles = [
            ClothParticle::pinned(Vec3::new(0.0, 0.0, 0.0)),
            ClothParticle::new(Vec3::new(0.0, -1.0, 0.0), 1.0),
        ];
        let graph = color_constraints(&[Constraint::new(
            0,
            1,
            1.0,
            Compliance::RIGID,
            ConstraintKind::Stretch,
        )]);
        solve_cloth(&mut particles, &graph, SolverParams::default());
        assert_eq!(particles[0].position, Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(particles[0].velocity, Vec3::ZERO);
    }

    #[test]
    fn rigid_link_holds_rest_length_under_gravity() {
        // Particle 0 pinned; particle 1 hangs on a rigid link and should settle
        // near one unit below without stretching much.
        let mut particles = [
            ClothParticle::pinned(Vec3::new(0.0, 0.0, 0.0)),
            ClothParticle::new(Vec3::new(0.0, -1.0, 0.0), 1.0),
        ];
        let graph = color_constraints(&[Constraint::new(
            0,
            1,
            1.0,
            Compliance::RIGID,
            ConstraintKind::Stretch,
        )]);
        let params = SolverParams::default();
        for _ in 0..120 {
            solve_cloth(&mut particles, &graph, params);
        }
        let dist = particles[0].position.distance(particles[1].position);
        assert!((dist - 1.0).abs() < 0.05, "rigid link stretched to {dist}");
        // It hangs below the anchor.
        assert!(particles[1].position.y < 0.0);
    }

    #[test]
    fn free_particle_falls_under_gravity() {
        let mut particles = [ClothParticle::new(Vec3::ZERO, 1.0)];
        let graph = ConstraintGraph::default();
        solve_cloth(&mut particles, &graph, SolverParams::default());
        assert!(particles[0].position.y < 0.0);
        assert!(particles[0].velocity.y < 0.0);
    }

    #[test]
    fn solve_is_deterministic() {
        let grid = ClothGrid::new(4, 4);
        let mut positions = Vec::new();
        for r in 0..4 {
            for c in 0..4 {
                positions.push(Vec3::new(c as f32, 0.0, r as f32));
            }
        }
        let constraints = build_grid_constraints(grid, &positions, rigid_params());
        let graph = color_constraints(&constraints);
        let make = || {
            let mut ps: Vec<ClothParticle> = positions
                .iter()
                .enumerate()
                .map(|(i, &p)| {
                    if i < 4 {
                        ClothParticle::pinned(p)
                    } else {
                        ClothParticle::new(p, 1.0)
                    }
                })
                .collect();
            for _ in 0..20 {
                solve_cloth(&mut ps, &graph, SolverParams::default());
            }
            ps
        };
        let a = make();
        let b = make();
        for (pa, pb) in a.iter().zip(b.iter()) {
            assert_eq!(pa.position, pb.position);
        }
    }

    #[test]
    fn strain_limit_caps_overstretch() {
        // A heavy free particle on a soft link would stretch far without the
        // limiter; with a 10% limit the edge stays within 1.1x rest.
        let link = [Constraint::new(
            0,
            1,
            1.0,
            Compliance(0.05),
            ConstraintKind::Stretch,
        )];
        let graph = color_constraints(&link);
        let params = SolverParams {
            strain_limit: 0.1,
            gravity: Vec3::new(0.0, -50.0, 0.0),
            ..Default::default()
        };
        let mut particles = [
            ClothParticle::pinned(Vec3::ZERO),
            ClothParticle::new(Vec3::new(0.0, -1.0, 0.0), 5.0),
        ];
        for _ in 0..60 {
            solve_cloth(&mut particles, &graph, params);
        }
        let dist = particles[0].position.distance(particles[1].position);
        assert!(dist <= 1.1 + 1.0e-3, "strain limiter failed: {dist}");
    }

    #[test]
    fn one_sided_constraint_ignores_slack() {
        // An LRA leash with rest 2.0 must not pull a particle that is only 1
        // unit away (within the leash).
        let leash = [Constraint::new(
            0,
            1,
            2.0,
            Compliance::RIGID,
            ConstraintKind::Lra,
        )];
        let graph = color_constraints(&leash);
        let mut particles = [
            ClothParticle::pinned(Vec3::ZERO),
            ClothParticle::new(Vec3::new(0.0, -1.0, 0.0), 1.0),
        ];
        let params = SolverParams {
            gravity: Vec3::ZERO,
            strain_limit: 0.0,
            ..Default::default()
        };
        solve_cloth(&mut particles, &graph, params);
        // No gravity, slack leash: the particle stays put.
        assert!((particles[1].position.distance(Vec3::new(0.0, -1.0, 0.0))) < 1.0e-6);
    }

    #[test]
    fn one_sided_constraint_pulls_when_overextended() {
        let leash = [Constraint::new(
            0,
            1,
            1.0,
            Compliance::RIGID,
            ConstraintKind::Lra,
        )];
        let graph = color_constraints(&leash);
        let mut particles = [
            ClothParticle::pinned(Vec3::ZERO),
            ClothParticle::new(Vec3::new(0.0, -3.0, 0.0), 1.0),
        ];
        let params = SolverParams {
            gravity: Vec3::ZERO,
            strain_limit: 0.0,
            ..Default::default()
        };
        solve_cloth(&mut particles, &graph, params);
        let dist = particles[0].position.distance(particles[1].position);
        assert!(dist < 3.0, "over-extended leash should pull inward: {dist}");
    }

    #[test]
    fn collision_hook_is_applied() {
        // A floor at y = -0.5: the falling particle is clamped above it.
        let mut particles = [ClothParticle::new(Vec3::ZERO, 1.0)];
        let graph = ConstraintGraph::default();
        solve_cloth_with_collision(
            &mut particles,
            &graph,
            SolverParams::default(),
            1.0 / 60.0,
            |p| {
                let mut pos = p.position;
                if pos.y < -0.5 {
                    pos.y = -0.5;
                }
                pos
            },
        );
        assert!(particles[0].position.y >= -0.5);
    }

    #[test]
    fn zero_dt_is_a_no_op() {
        let mut particles = [ClothParticle::new(Vec3::ZERO, 1.0)];
        let graph = ConstraintGraph::default();
        solve_cloth_with_collision(&mut particles, &graph, SolverParams::default(), 0.0, |p| {
            p.position
        });
        assert_eq!(particles[0].position, Vec3::ZERO);
        assert_eq!(particles[0].velocity, Vec3::ZERO);
    }

    #[test]
    fn out_of_range_constraint_is_skipped() {
        let mut particles = [ClothParticle::new(Vec3::ZERO, 1.0)];
        // Constraint references particle index 9 which does not exist.
        let graph = color_constraints(&[Constraint::new(
            0,
            9,
            1.0,
            Compliance::RIGID,
            ConstraintKind::Stretch,
        )]);
        // Must not panic.
        solve_cloth(&mut particles, &graph, SolverParams::default());
    }

    #[test]
    fn extract_positions_matches_particles() {
        let particles = [
            ClothParticle::new(Vec3::new(1.0, 2.0, 3.0), 1.0),
            ClothParticle::pinned(Vec3::new(4.0, 5.0, 6.0)),
        ];
        let positions = extract_positions(&particles);
        assert_eq!(positions.len(), 2);
        assert_eq!(positions[0], Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(positions[1], Vec3::new(4.0, 5.0, 6.0));
    }

    #[test]
    fn solve_request_charges_through_shared_budget() {
        let request = cloth_solve_request(DeformationHandle(3), 4096, 5);
        assert_eq!(request.kind, DeformationKind::Cloth);
        assert_eq!(request.vertex_count, 4096);
        assert!(request.needs_blas_refit);
        let budget = DeformationBudget {
            vertices_per_frame: 8192,
            blas_refits_per_frame: 4,
        };
        let plan = plan_deformations(&[request], budget);
        assert_eq!(plan.scheduled_count(), 1);
        assert_eq!(plan.count_of_kind(DeformationKind::Cloth), 1);
    }
}
