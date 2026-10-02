//! Guide-strand XPBD dynamics: the strand-based sim stage of the hair engine.
//!
//! A groom simulates only its sparse set of *guide* strands; the many render
//! strands are interpolated from them and never simulate (see [`super`] stage
//! 1). This module advances one guide strand with an Extended Position Based
//! Dynamics (XPBD) solver, mirroring the local/global shape constraints of
//! `TressFX`-class strand dynamics at the algorithm level without reusing any
//! of its code. Collision against analytic body proxies is optional (see
//! [`super::collision`]): the collision-free solve is the cheap base tier, and
//! passing colliders enables the head/neck/shoulder projection on top.
//!
//! Each guide is a poly-line of [`StrandParticle`]s. The root particle is
//! *pinned* (`inverse_mass == 0`) so it rides the skinned scalp instead of
//! falling; every other particle is free. One `simulate_strand` call advances a
//! strand by `dt`, split into `substeps` semi-implicit integration steps, each
//! projecting four constraint families for `iterations` Gauss-Seidel sweeps:
//!
//! 1. **Edge-length** — a compliant distance constraint per segment keeps the
//!    strand from stretching. The projection is delegated to the authoritative
//!    physics engine (`prism_physics_core`) through
//!    [`super::physics_bridge::solve_edges`] rather than re-implemented here, so
//!    there is a single copy of the XPBD distance arithmetic; the engine
//!    normalizes compliance by `dt_sub^2` so stiffness is step-size independent.
//! 2. **Local shape (bending)** — each interior particle is pulled toward the
//!    midpoint of its two neighbors, a discrete Laplacian that resists kinks
//!    and stops the strand collapsing onto itself.
//! 3. **Global shape** — every free particle is pulled toward its goal (rest /
//!    animated target) pose so the hairstyle cannot drift away over time.
//! 4. **Long-range attachment (LRA / tether)** — each free particle is capped
//!    to the cumulative rest distance from the strand root, so fast head
//!    motion can never stretch a strand past its authored length (the
//!    `TressFX`-style tether, design §6.5). It only ever pulls a particle
//!    back toward the root, never pushes it out.
//!
//! Everything here is deterministic array-in/array-out CPU math (design §7,
//! the "compute-portable" bucket): identical inputs produce bit-identical
//! outputs, so results can be golden-tested by hand. This module only *solves*;
//! charging the solved vertices against the shared deformation budget is
//! [`super::lod::hair_deformation_request`]'s job, not duplicated here.
//!
//! Collision against analytic body proxies is optional and layered on top of
//! the constraint solve: pass a slice of [`Collider`]s to push free particles
//! out of the head/neck/shoulder proxies after each constraint sweep, or pass
//! `&[]` for the collision-free base tier (design §6.2). The projection lives
//! in [`super::collision`].

use super::collision::{resolve_strand_collisions, Collider};

/// A minimal 3-component vector for strand math.
///
/// The crate carries no linear-algebra dependency, so the solver defines its
/// own value type. All operations are plain `f32` arithmetic evaluated in a
/// fixed order, which is what makes the solver bit-for-bit reproducible.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    /// X component.
    pub x: f32,
    /// Y component.
    pub y: f32,
    /// Z component.
    pub z: f32,
}

impl Vec3 {
    /// Constructs a vector from its components.
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// The zero vector.
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// Component-wise sum.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "The strand math API is specified with named add/sub methods for call-site uniformity; operator traits are intentionally not part of this internal type."
    )]
    pub fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }

    /// Component-wise difference `self - rhs`.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "See add: the specified API uses named sub for call-site uniformity, not operator traits."
    )]
    pub fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }

    /// Uniform scale by a scalar.
    #[must_use]
    pub fn scale(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }

    /// Dot (inner) product.
    #[must_use]
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    /// Squared Euclidean length; cheaper than [`Vec3::length`] when only
    /// comparisons are needed.
    #[must_use]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    /// Euclidean length.
    #[must_use]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Returns the unit vector along `self`, or [`Vec3::ZERO`] when `self` is
    /// (numerically) the zero vector, so normalization never yields NaN.
    #[must_use]
    pub fn normalize_or_zero(self) -> Self {
        let len_sq = self.length_squared();
        if len_sq > EPS_LEN_SQ {
            self.scale(1.0 / len_sq.sqrt())
        } else {
            Self::ZERO
        }
    }
}

/// One control point of a guide strand.
///
/// Position uses a Verlet-style pair (`position` plus `prev_position`) so
/// velocity is stored implicitly as their difference; the integrator advances
/// `position` and rolls `prev_position` forward. A particle with
/// `inverse_mass == 0` (equivalently non-positive) is *pinned*: it is treated
/// as infinitely heavy, never integrated, and never moved by any constraint,
/// which is how the root rides the skinned scalp.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StrandParticle {
    /// Current world-space position.
    pub position: Vec3,
    /// Position at the start of the previous substep (velocity bookkeeping).
    pub prev_position: Vec3,
    /// Reciprocal mass; `0` pins the particle in place.
    pub inverse_mass: f32,
}

impl StrandParticle {
    /// Convenience constructor for a free particle of unit inverse mass.
    #[must_use]
    pub fn free(position: Vec3) -> Self {
        Self {
            position,
            prev_position: position,
            inverse_mass: 1.0,
        }
    }

    /// Convenience constructor for a pinned (kinematic) particle.
    #[must_use]
    pub fn pinned(position: Vec3) -> Self {
        Self {
            position,
            prev_position: position,
            inverse_mass: 0.0,
        }
    }

    /// Returns `true` when this particle is pinned and must not move.
    #[must_use]
    pub fn is_pinned(&self) -> bool {
        self.inverse_mass <= 0.0
    }
}

/// Tunable parameters for one XPBD solve.
///
/// The four `*_stiffness`/`damping` factors are normalized fractions in
/// `0..=1` and are clamped to that range internally, so out-of-range authoring
/// values saturate rather than destabilize the solve. `edge_compliance` is a
/// physical inverse-stiffness in `>= 0` (`0` is perfectly rigid); it is clamped
/// to be non-negative.
#[derive(Clone, Copy, Debug)]
pub struct XpbdParams {
    /// Uniform acceleration applied to every free particle (world units/s^2).
    pub gravity: Vec3,
    /// Frame time step advanced by one `simulate_strand` call (seconds).
    pub dt: f32,
    /// Number of semi-implicit integration steps `dt` is split into.
    pub substeps: u32,
    /// Constraint projection sweeps per substep.
    pub iterations: u32,
    /// XPBD inverse stiffness of the edge-length constraint (`0` = rigid).
    pub edge_compliance: f32,
    /// Local (bending) shape-matching strength in `0..=1`.
    pub local_stiffness: f32,
    /// Global shape-matching strength toward the goal pose in `0..=1`.
    pub global_stiffness: f32,
    /// Long-range attachment (tether) strength in `0..=1`: the fraction of
    /// the over-stretch pulled back toward the root each sweep (`1` snaps a
    /// too-far particle exactly onto its tether sphere). `0` disables the
    /// tether.
    pub lra_stiffness: f32,
    /// Velocity retention fraction in `0..=1` (`0` keeps all velocity, `1`
    /// removes it); models aerodynamic drag / numerical damping.
    pub damping: f32,
}

/// Vectors shorter than the square root of this are treated as zero-length.
const EPS_LEN_SQ: f32 = 1.0e-24;
/// Segments shorter than this are skipped to avoid dividing by ~0.
const EPS_LEN: f32 = 1.0e-12;

/// Advances a single guide strand by `params.dt` with the XPBD solver.
///
/// `particles` is the strand's poly-line in root-to-tip order and is updated in
/// place. `rest_lengths[i]` is the target length of the segment between
/// particle `i` and `i + 1`; `goal_positions[i]` is particle `i`'s global
/// target pose. Both companion slices are read defensively with `get`, so a
/// short or empty slice simply disables the corresponding constraint for the
/// missing indices instead of panicking. `colliders` are projected once per
/// substep after the constraint sweeps; pass `&[]` to run the collision-free
/// base tier.
///
/// The call is a no-op when there is nothing to advance: an empty strand, zero
/// substeps, or a non-positive `dt`. Pinned particles are left exactly where
/// the caller placed them (their skinned pose).
pub fn simulate_strand(
    particles: &mut [StrandParticle],
    rest_lengths: &[f32],
    goal_positions: &[Vec3],
    colliders: &[Collider],
    params: XpbdParams,
) {
    if particles.is_empty() || params.substeps == 0 || params.dt <= 0.0 || !params.dt.is_finite() {
        return;
    }

    let sub_dt = params.dt / params.substeps as f32;
    let sub_dt_sq = sub_dt * sub_dt;
    let damping = params.damping.clamp(0.0, 1.0);
    let local_stiffness = params.local_stiffness.clamp(0.0, 1.0);
    let global_stiffness = params.global_stiffness.clamp(0.0, 1.0);
    let lra_stiffness = params.lra_stiffness.clamp(0.0, 1.0);
    let edge_compliance = params.edge_compliance.max(0.0);
    let gravity_step = params.gravity.scale(sub_dt_sq);
    let velocity_retain = 1.0 - damping;

    for _ in 0..params.substeps {
        integrate(particles, gravity_step, velocity_retain);
        for _ in 0..params.iterations {
            super::physics_bridge::solve_edges(particles, rest_lengths, edge_compliance, sub_dt);
            super::physics_bridge::solve_local_smooth(particles, local_stiffness);
            super::physics_bridge::solve_global_pull(particles, goal_positions, global_stiffness);
            super::physics_bridge::solve_lra_tether(
                particles,
                rest_lengths,
                lra_stiffness,
                EPS_LEN,
            );
        }
        // Collision is projected once per substep, after the constraint sweeps,
        // so hair settles against the body without fighting the shape solve.
        resolve_strand_collisions(particles, colliders);
    }
}

/// Semi-implicit (Verlet) prediction step.
///
/// Free particles carry their implicit velocity `position - prev_position`
/// forward, scaled by `velocity_retain` for damping, and gain the gravity
/// displacement. Pinned particles stay put but still roll `prev_position`
/// forward so their implied velocity stays zero.
fn integrate(particles: &mut [StrandParticle], gravity_step: Vec3, velocity_retain: f32) {
    for particle in particles.iter_mut() {
        if particle.is_pinned() {
            particle.prev_position = particle.position;
            continue;
        }
        let velocity = particle
            .position
            .sub(particle.prev_position)
            .scale(velocity_retain);
        particle.prev_position = particle.position;
        particle.position = particle.position.add(velocity).add(gravity_step);
    }
}

/// Advances many concatenated guide strands in a single deterministic pass.
///
/// `particles`, `rest_lengths`, and `goal_positions` are flat arrays shared by
/// all strands; `strand_lengths[k]` is the particle count of strand `k`, and
/// the strands occupy consecutive ranges in that order (the same
/// offset-slicing discipline used by the geometry binning pass in
/// [`crate::virtual_geometry`]). Each strand is solved with
/// [`simulate_strand`] over its own sub-slice, sharing the same `colliders`
/// across every strand. A `strand_lengths` entry that would run past the end
/// of `particles` stops the walk, so malformed layouts truncate
/// deterministically instead of panicking.
pub fn simulate_guides(
    particles: &mut [StrandParticle],
    strand_lengths: &[usize],
    rest_lengths: &[Vec3Len],
    goal_positions: &[Vec3],
    colliders: &[Collider],
    params: XpbdParams,
) {
    let mut offset = 0usize;
    for &length in strand_lengths {
        let Some(end) = offset.checked_add(length) else {
            break;
        };
        if end > particles.len() {
            break;
        }
        let strand = &mut particles[offset..end];
        let rest = rest_lengths.get(offset..end).unwrap_or(&[]);
        let goal = goal_positions.get(offset..end).unwrap_or(&[]);
        simulate_strand(strand, rest, goal, colliders, params);
        offset = end;
    }
}

/// A per-particle rest length, aligned index-for-index with the flat particle
/// array consumed by [`simulate_guides`]. Entry `i` is the rest length of the
/// segment leaving particle `i`; the last particle of each strand has no
/// outgoing segment and its entry is ignored.
pub type Vec3Len = f32;

/// Total control points across a set of guide strands, saturating at
/// [`u32::MAX`].
///
/// This is the count a caller reports to the deformation budget (via
/// [`super::lod::hair_deformation_request`]); it is provided here only as a
/// convenience and deliberately builds no budget request of its own.
#[must_use]
pub fn solved_guide_vertex_count(strand_lengths: &[usize]) -> u32 {
    let total = strand_lengths
        .iter()
        .copied()
        .fold(0usize, usize::saturating_add);
    u32::try_from(total).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    /// Tolerance for approximate float comparisons in the golden tests.
    const EPS: f32 = 1.0e-3;

    fn params() -> XpbdParams {
        XpbdParams {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            dt: 1.0 / 60.0,
            substeps: 2,
            iterations: 8,
            edge_compliance: 0.0,
            local_stiffness: 0.0,
            global_stiffness: 0.0,
            lra_stiffness: 0.0,
            damping: 0.0,
        }
    }

    #[test]
    fn pinned_root_never_moves_under_gravity() {
        let mut particles = vec![
            StrandParticle::pinned(Vec3::new(1.0, 2.0, 3.0)),
            StrandParticle::free(Vec3::new(1.0, 1.0, 3.0)),
        ];
        let rest = [1.0, 0.0];
        for _ in 0..20 {
            simulate_strand(&mut particles, &rest, &[], &[], params());
        }
        let root = particles[0].position;
        assert!((root.x - 1.0).abs() < EPS);
        assert!((root.y - 2.0).abs() < EPS);
        assert!((root.z - 3.0).abs() < EPS);
    }

    #[test]
    fn free_particle_falls_along_gravity() {
        let mut particles = vec![StrandParticle::free(Vec3::ZERO)];
        let start = particles[0].position.y;
        for _ in 0..10 {
            simulate_strand(&mut particles, &[], &[], &[], params());
        }
        // Gravity points down (-y), so the particle must descend.
        assert!(particles[0].position.y < start - EPS);
        // No lateral forces, so x/z stay put.
        assert!(particles[0].position.x.abs() < EPS);
        assert!(particles[0].position.z.abs() < EPS);
    }

    #[test]
    fn edge_lengths_converge_to_rest() {
        // Root pinned; two free particles start over-stretched.
        let mut particles = vec![
            StrandParticle::pinned(Vec3::ZERO),
            StrandParticle::free(Vec3::new(2.0, 0.0, 0.0)),
            StrandParticle::free(Vec3::new(4.0, 0.0, 0.0)),
        ];
        let rest = [1.0, 1.0, 0.0];
        let mut p = params();
        p.gravity = Vec3::ZERO;
        // One substep isolates constraint relaxation from cross-substep Verlet
        // momentum, so this reads the converged rest state directly.
        p.substeps = 1;
        p.iterations = 40;
        simulate_strand(&mut particles, &rest, &[], &[], p);
        let e0 = particles[1].position.sub(particles[0].position).length();
        let e1 = particles[2].position.sub(particles[1].position).length();
        assert!((e0 - 1.0).abs() < EPS, "edge0 = {e0}");
        assert!((e1 - 1.0).abs() < EPS, "edge1 = {e1}");
    }

    #[test]
    fn global_stiffness_one_reaches_goal() {
        let mut particles = vec![
            StrandParticle::pinned(Vec3::ZERO),
            StrandParticle::free(Vec3::new(1.0, 0.0, 0.0)),
        ];
        let goals = [Vec3::ZERO, Vec3::new(0.0, -5.0, 2.0)];
        let mut p = params();
        p.gravity = Vec3::ZERO;
        p.global_stiffness = 1.0;
        simulate_strand(&mut particles, &[], &goals, &[], p);
        let tip = particles[1].position;
        assert!((tip.x - 0.0).abs() < EPS);
        assert!((tip.y + 5.0).abs() < EPS);
        assert!((tip.z - 2.0).abs() < EPS);
    }

    #[test]
    fn local_stiffness_straightens_a_kink() {
        // Middle particle kicked off the line; a full-strength local pull
        // should move it toward the neighbor midpoint (back onto the line).
        let mut particles = vec![
            StrandParticle::pinned(Vec3::new(-1.0, 0.0, 0.0)),
            StrandParticle::free(Vec3::new(0.0, 1.0, 0.0)),
            StrandParticle::pinned(Vec3::new(1.0, 0.0, 0.0)),
        ];
        let mut p = params();
        p.gravity = Vec3::ZERO;
        p.local_stiffness = 1.0;
        p.iterations = 1;
        p.substeps = 1;
        simulate_strand(&mut particles, &[], &[], &[], p);
        // Midpoint of the two pinned neighbors is the origin.
        assert!(particles[1].position.y.abs() < EPS);
        assert!(particles[1].position.x.abs() < EPS);
    }

    #[test]
    fn solve_is_bit_deterministic() {
        let build = || {
            vec![
                StrandParticle::pinned(Vec3::ZERO),
                StrandParticle::free(Vec3::new(0.5, 0.0, 0.0)),
                StrandParticle::free(Vec3::new(1.0, 0.0, 0.0)),
                StrandParticle::free(Vec3::new(1.5, 0.0, 0.0)),
            ]
        };
        let rest = [0.5, 0.5, 0.5, 0.0];
        let goals = [
            Vec3::ZERO,
            Vec3::new(0.5, 0.1, 0.0),
            Vec3::new(1.0, 0.2, 0.0),
            Vec3::new(1.5, 0.3, 0.0),
        ];
        let mut p = params();
        p.local_stiffness = 0.3;
        p.global_stiffness = 0.2;
        p.damping = 0.1;

        let mut a = build();
        let mut b = build();
        for _ in 0..5 {
            simulate_strand(&mut a, &rest, &goals, &[], p);
            simulate_strand(&mut b, &rest, &goals, &[], p);
        }
        for (pa, pb) in a.iter().zip(b.iter()) {
            assert_eq!(pa.position.x.to_bits(), pb.position.x.to_bits());
            assert_eq!(pa.position.y.to_bits(), pb.position.y.to_bits());
            assert_eq!(pa.position.z.to_bits(), pb.position.z.to_bits());
        }
    }

    #[test]
    fn degenerate_inputs_do_not_panic() {
        // Empty strand.
        let mut empty: Vec<StrandParticle> = Vec::new();
        simulate_strand(&mut empty, &[], &[], &[], params());
        assert!(empty.is_empty());

        // Single particle with empty companions.
        let mut one = vec![StrandParticle::free(Vec3::ZERO)];
        simulate_strand(&mut one, &[], &[], &[], params());

        // Zero substeps / non-positive dt are no-ops.
        let mut two = vec![
            StrandParticle::pinned(Vec3::ZERO),
            StrandParticle::free(Vec3::new(0.0, -1.0, 0.0)),
        ];
        let before = two[1].position.y;
        let mut p = params();
        p.substeps = 0;
        simulate_strand(&mut two, &[1.0, 0.0], &[], &[], p);
        assert!((two[1].position.y - before).abs() < EPS);
        p.substeps = 2;
        p.dt = 0.0;
        simulate_strand(&mut two, &[1.0, 0.0], &[], &[], p);
        assert!((two[1].position.y - before).abs() < EPS);
    }

    #[test]
    fn simulate_guides_solves_each_strand_and_truncates_safely() {
        // Two strands of two particles each, laid out flat.
        let mut particles = vec![
            StrandParticle::pinned(Vec3::ZERO),
            StrandParticle::free(Vec3::new(1.0, 0.0, 0.0)),
            StrandParticle::pinned(Vec3::new(0.0, 10.0, 0.0)),
            StrandParticle::free(Vec3::new(1.0, 10.0, 0.0)),
        ];
        let goals = [
            Vec3::ZERO,
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 10.0, 0.0),
            Vec3::new(0.0, 9.0, 0.0),
        ];
        let rest: [Vec3Len; 4] = [1.0, 0.0, 1.0, 0.0];
        // The third "strand" length overruns the array and must be ignored.
        let lengths = [2usize, 2, 2];
        let mut p = params();
        p.gravity = Vec3::ZERO;
        p.global_stiffness = 1.0;
        simulate_guides(&mut particles, &lengths, &rest, &goals, &[], p);
        assert!((particles[1].position.y + 1.0).abs() < EPS);
        assert!((particles[3].position.y - 9.0).abs() < EPS);
        // Roots stayed pinned.
        assert!((particles[0].position.y - 0.0).abs() < EPS);
        assert!((particles[2].position.y - 10.0).abs() < EPS);
    }

    #[test]
    fn vertex_count_sums_and_saturates() {
        assert_eq!(solved_guide_vertex_count(&[2, 3, 5]), 10);
        assert_eq!(solved_guide_vertex_count(&[]), 0);
        assert_eq!(
            solved_guide_vertex_count(&[usize::MAX, 1]),
            u32::MAX,
            "overflow saturates instead of wrapping",
        );
    }

    #[test]
    fn out_of_range_stiffness_is_clamped() {
        // Absurd stiffness/damping must not blow up; global > 1 still lands on
        // (not past) the goal because the fraction saturates at 1.
        let mut particles = vec![
            StrandParticle::pinned(Vec3::ZERO),
            StrandParticle::free(Vec3::new(1.0, 0.0, 0.0)),
        ];
        let goals = [Vec3::ZERO, Vec3::new(0.0, -3.0, 0.0)];
        let mut p = params();
        p.gravity = Vec3::ZERO;
        p.global_stiffness = 5.0;
        p.local_stiffness = -2.0;
        p.damping = 9.0;
        simulate_strand(&mut particles, &[], &goals, &[], p);
        assert!((particles[1].position.y + 3.0).abs() < EPS);
    }

    #[test]
    fn lra_tether_caps_overstretch_but_never_pushes_out() {
        // Edge constraint is made nearly free (huge compliance) and the shape
        // constraints are off, so only the tether acts and can be checked in
        // isolation.
        let mut p = params();
        p.gravity = Vec3::ZERO;
        p.substeps = 1;
        p.iterations = 1;
        p.edge_compliance = 1.0e6;
        p.lra_stiffness = 1.0;

        // A particle flung to 5 units with a 1-unit tether is snapped back
        // onto the tether sphere along the same radial direction.
        let mut over = vec![
            StrandParticle::pinned(Vec3::ZERO),
            StrandParticle::free(Vec3::new(5.0, 0.0, 0.0)),
        ];
        simulate_strand(&mut over, &[1.0, 0.0], &[], &[], p);
        assert!((over[1].position.x - 1.0).abs() < EPS);
        assert!(over[1].position.y.abs() < EPS);
        assert!(over[1].position.z.abs() < EPS);

        // A particle already inside its tether is left where it is (one-sided).
        let mut inside = vec![
            StrandParticle::pinned(Vec3::ZERO),
            StrandParticle::free(Vec3::new(0.5, 0.0, 0.0)),
        ];
        simulate_strand(&mut inside, &[1.0, 0.0], &[], &[], p);
        assert!((inside[1].position.x - 0.5).abs() < EPS);
    }

    #[test]
    fn collider_keeps_strand_out_of_body_sphere() {
        // A free particle just below a scalp-sized sphere is dragged into it by
        // gravity; the collider must project it back onto the surface each step.
        let mut particles = vec![
            StrandParticle::pinned(Vec3::new(0.0, 2.0, 0.0)),
            StrandParticle::free(Vec3::new(0.0, 1.05, 0.0)),
        ];
        let rest = [1.0, 0.0];
        let sphere = Collider::Sphere {
            center: Vec3::ZERO,
            radius: 1.0,
        };
        let mut p = params();
        p.gravity = Vec3::new(0.0, -9.81, 0.0);
        p.dt = 0.1;
        p.substeps = 4;
        simulate_strand(&mut particles, &rest, &[], &[sphere], p);
        // The free particle can never end up inside the sphere.
        let r = particles[1].position.length();
        assert!(r >= 1.0 - 1.0e-4, "particle penetrated collider: r = {r}");
    }

    #[test]
    fn empty_colliders_match_collision_free_solve() {
        // Passing no colliders must be bit-identical to the base tier.
        let make = || {
            vec![
                StrandParticle::pinned(Vec3::ZERO),
                StrandParticle::free(Vec3::new(1.0, 0.0, 0.0)),
                StrandParticle::free(Vec3::new(2.0, 0.0, 0.0)),
            ]
        };
        let mut a = make();
        let mut b = make();
        let rest = [1.0, 1.0, 0.0];
        let mut p = params();
        p.gravity = Vec3::new(0.0, -9.81, 0.0);
        p.dt = 0.05;
        p.substeps = 3;
        simulate_strand(&mut a, &rest, &[], &[], p);
        simulate_strand(&mut b, &rest, &[], &[], p);
        for (pa, pb) in a.iter().zip(b.iter()) {
            assert_eq!(pa.position, pb.position);
        }
    }

    #[test]
    fn vec3_helpers_are_correct() {
        let a = Vec3::new(3.0, 4.0, 0.0);
        assert!((a.length() - 5.0).abs() < EPS);
        assert!((a.length_squared() - 25.0).abs() < EPS);
        let unit = a.normalize_or_zero();
        assert!((unit.length() - 1.0).abs() < EPS);
        assert_eq!(Vec3::ZERO.normalize_or_zero(), Vec3::ZERO);
        let b = Vec3::new(1.0, 0.0, 2.0);
        assert!((a.dot(b) - 3.0).abs() < EPS);
        assert_eq!(a.add(b), Vec3::new(4.0, 4.0, 2.0));
        assert_eq!(a.sub(b), Vec3::new(2.0, 4.0, -2.0));
        assert_eq!(a.scale(2.0), Vec3::new(6.0, 8.0, 0.0));
    }
}
