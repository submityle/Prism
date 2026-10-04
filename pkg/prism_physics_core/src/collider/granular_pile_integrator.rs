//! Time-steppable granular pile: grain-grain plus grain-boundary rotational
//! contact under gravity.
//!
//! [`RotationalContactBody`](super::rotational_contact_integrator::RotationalContactBody)
//! advances a free-floating sphere packing but has nothing to rest on. A pile,
//! a hopper, or a rotating drum needs the grains to settle against solid
//! boundaries. This integrator composes the two resolvers it owns —
//! [`RollingContactResolver`](super::rotational_contact_resolver::RollingContactResolver)
//! for grain-grain contact and
//! [`BoundaryContactResolver`](super::rotational_boundary_resolver::BoundaryContactResolver)
//! for grain versus a set of [`HalfSpace`] boundaries — and integrates the
//! combined force and torque with a constant gravitational body force using a
//! semi-implicit (symplectic) Euler step.
//!
//! The rolling-resistance term carried through both resolvers is what lets a
//! settled pile hold a finite angle of repose on a floor: without it idealised
//! spheres roll almost freely and the heap slumps flat.

use super::rotational_boundary_contact::HalfSpace;
use super::rotational_boundary_resolver::{BoundaryContactResolution, BoundaryContactResolver};
use super::rotational_contact::RollingContactModel;
use super::rotational_contact_resolver::{RollingContactResolution, RollingContactResolver};
use glam::Vec3;

/// Per-step diagnostics returned by [`GranularPileBody::step`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GranularPileStepReport {
    /// Number of grain-grain contacts resolved this step.
    pub grain_contact_count: usize,
    /// Number of grain-boundary contacts resolved this step.
    pub boundary_contact_count: usize,
    /// Largest overlap across all contacts (grain and boundary) this step.
    pub max_overlap: f32,
    /// Largest normal force magnitude across all contacts this step.
    pub max_normal_force: f32,
    /// Number of contacts at the Coulomb (sliding) limit this step.
    pub sliding_count: usize,
    /// Total kinetic energy (translational plus rotational) of the finite-mass
    /// grains after the step.
    pub kinetic_energy: f32,
}

/// A packing of spheres settling against a set of static boundaries under
/// gravity, carrying linear and angular velocities and persistent contact
/// springs for both the grain-grain and grain-boundary interactions.
#[derive(Clone, Debug)]
pub struct GranularPileBody {
    model: RollingContactModel,
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    angular: Vec<Vec3>,
    radii: Vec<f32>,
    masses: Vec<f32>,
    inertias: Vec<f32>,
    planes: Vec<HalfSpace>,
    gravity: Vec3,
    grain_resolver: RollingContactResolver,
    boundary_resolver: BoundaryContactResolver,
}

impl GranularPileBody {
    /// Builds a pile from a contact `model`, initial `positions`, per-grain
    /// `radii` and `masses`, the set of boundary `planes`, and a constant
    /// `gravity` acceleration. Linear and angular velocities start at rest and
    /// both contact histories are empty. Each finite-mass grain is assigned the
    /// isotropic solid-sphere inertia `I = ⅖·m·r²`.
    ///
    /// A grain whose mass is non-finite (infinite) is *pinned*: it never
    /// integrates and is unaffected by gravity, acting as a movable obstacle.
    /// Returns `None` when the arrays disagree in length, when any position is
    /// non-finite, when any radius is not finite and strictly positive, when
    /// any mass is NaN or finite but non-positive, or when `gravity` is
    /// non-finite.
    #[must_use]
    pub fn new(
        model: RollingContactModel,
        positions: Vec<Vec3>,
        radii: Vec<f32>,
        masses: Vec<f32>,
        planes: Vec<HalfSpace>,
        gravity: Vec3,
    ) -> Option<Self> {
        let n = positions.len();
        if radii.len() != n || masses.len() != n {
            return None;
        }
        if positions
            .iter()
            .any(|p| !(p.x.is_finite() && p.y.is_finite() && p.z.is_finite()))
        {
            return None;
        }
        if !radii.iter().all(|&r| r.is_finite() && r > 0.0) {
            return None;
        }
        let valid_mass = |m: f32| !m.is_nan() && (m.is_infinite() || m > 0.0);
        if !masses.iter().copied().all(valid_mass) {
            return None;
        }
        if !(gravity.x.is_finite() && gravity.y.is_finite() && gravity.z.is_finite()) {
            return None;
        }
        let inertias: Vec<f32> = masses
            .iter()
            .zip(radii.iter())
            .map(|(&m, &r)| 0.4 * m * r * r)
            .collect();
        Some(Self {
            model,
            velocities: vec![Vec3::ZERO; n],
            angular: vec![Vec3::ZERO; n],
            positions,
            radii,
            masses,
            inertias,
            planes,
            gravity,
            grain_resolver: RollingContactResolver::new(),
            boundary_resolver: BoundaryContactResolver::new(),
        })
    }

    /// Number of grains in the pile.
    #[must_use]
    pub fn particle_count(&self) -> usize {
        self.positions.len()
    }

    /// Current grain centres, parallel to the construction `positions`.
    #[must_use]
    pub fn positions(&self) -> &[Vec3] {
        &self.positions
    }

    /// Current grain linear velocities.
    #[must_use]
    pub fn velocities(&self) -> &[Vec3] {
        &self.velocities
    }

    /// Current grain angular velocities.
    #[must_use]
    pub fn angular_velocities(&self) -> &[Vec3] {
        &self.angular
    }

    /// Grain radii.
    #[must_use]
    pub fn radii(&self) -> &[f32] {
        &self.radii
    }

    /// Per-grain moments of inertia.
    #[must_use]
    pub fn inertias(&self) -> &[f32] {
        &self.inertias
    }

    /// The boundary planes confining the pile.
    #[must_use]
    pub fn planes(&self) -> &[HalfSpace] {
        &self.planes
    }

    /// The constant gravitational acceleration applied each step.
    #[must_use]
    pub fn gravity(&self) -> Vec3 {
        self.gravity
    }

    /// Number of live grain-grain contact springs currently stored.
    #[must_use]
    pub fn grain_contacts(&self) -> usize {
        self.grain_resolver.active_contacts()
    }

    /// Number of live grain-boundary contact springs currently stored.
    #[must_use]
    pub fn boundary_contacts(&self) -> usize {
        self.boundary_resolver.active_contacts()
    }

    /// Whether grain `i` is pinned (non-finite mass). Out-of-range indices
    /// report `false`.
    #[must_use]
    pub fn is_pinned(&self, i: usize) -> bool {
        self.masses.get(i).is_some_and(|m| !m.is_finite())
    }

    /// Overrides the linear velocity of grain `i`; out-of-range indices are
    /// ignored.
    pub fn set_velocity(&mut self, i: usize, velocity: Vec3) {
        if let Some(v) = self.velocities.get_mut(i) {
            *v = velocity;
        }
    }

    /// Overrides the angular velocity of grain `i`; out-of-range indices are
    /// ignored.
    pub fn set_angular_velocity(&mut self, i: usize, angular: Vec3) {
        if let Some(w) = self.angular.get_mut(i) {
            *w = angular;
        }
    }

    /// Translational kinetic energy `Σ ½·mᵢ·|vᵢ|²` over the finite-mass grains.
    #[must_use]
    pub fn translational_kinetic_energy(&self) -> f32 {
        let mut e = 0.0;
        for i in 0..self.positions.len() {
            if self.masses[i].is_finite() {
                e += 0.5 * self.masses[i] * self.velocities[i].length_squared();
            }
        }
        e
    }

    /// Rotational kinetic energy `Σ ½·Iᵢ·|ωᵢ|²` over the finite-mass grains.
    #[must_use]
    pub fn rotational_kinetic_energy(&self) -> f32 {
        let mut e = 0.0;
        for i in 0..self.positions.len() {
            if self.masses[i].is_finite() {
                e += 0.5 * self.inertias[i] * self.angular[i].length_squared();
            }
        }
        e
    }

    /// Total kinetic energy (translational plus rotational).
    #[must_use]
    pub fn kinetic_energy(&self) -> f32 {
        self.translational_kinetic_energy() + self.rotational_kinetic_energy()
    }

    /// Linear momentum `Σ mᵢ·vᵢ` over the finite-mass grains.
    #[must_use]
    pub fn linear_momentum(&self) -> Vec3 {
        let mut p = Vec3::ZERO;
        for i in 0..self.positions.len() {
            if self.masses[i].is_finite() {
                p += self.masses[i] * self.velocities[i];
            }
        }
        p
    }

    /// Advances the pile by `dt`, re-resolving the grain-grain and
    /// grain-boundary contact sets, applying gravity and the optional
    /// per-grain external forces and torques, and advancing both spring maps.
    ///
    /// `external_forces` and `external_torques` must each have one entry per
    /// grain. Returns `None` when `dt` is not strictly positive and finite,
    /// when either external slice has the wrong length, or when an internal
    /// resolver rejects the state (which cannot happen for a well-formed body).
    /// On `None` the body is left unchanged.
    #[must_use]
    pub fn step(
        &mut self,
        dt: f32,
        external_forces: &[Vec3],
        external_torques: &[Vec3],
    ) -> Option<GranularPileStepReport> {
        let n = self.positions.len();
        if !(dt.is_finite() && dt > 0.0) {
            return None;
        }
        if external_forces.len() != n || external_torques.len() != n {
            return None;
        }

        let grain: RollingContactResolution = self.grain_resolver.resolve(
            &self.positions,
            &self.radii,
            &self.velocities,
            &self.angular,
            &self.model,
            dt,
        )?;
        let boundary: BoundaryContactResolution = self.boundary_resolver.resolve(
            &self.positions,
            &self.radii,
            &self.velocities,
            &self.angular,
            &self.planes,
            &self.model,
            dt,
        )?;

        for i in 0..n {
            if self.masses[i].is_finite() {
                let force = grain.forces[i]
                    + boundary.forces[i]
                    + external_forces[i]
                    + self.masses[i] * self.gravity;
                self.velocities[i] += (dt / self.masses[i]) * force;
                let torque = grain.torques[i] + boundary.torques[i] + external_torques[i];
                self.angular[i] += (dt / self.inertias[i]) * torque;
            }
        }
        for (pos, &vel) in self.positions.iter_mut().zip(self.velocities.iter()) {
            *pos += vel * dt;
        }

        Some(GranularPileStepReport {
            grain_contact_count: grain.contact_count(),
            boundary_contact_count: boundary.contact_count(),
            max_overlap: grain.max_overlap().max(boundary.max_overlap()),
            max_normal_force: grain.max_normal_force().max(boundary.max_normal_force()),
            sliding_count: grain.sliding_count() + boundary.sliding_count(),
            kinetic_energy: self.kinetic_energy(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> RollingContactModel {
        // Stiff but well-damped so dropped grains settle instead of ringing.
        RollingContactModel::new((2.0e4, 80.0), (2.0e4, 80.0, 0.6), (2.0e3, 40.0, 0.4)).unwrap()
    }

    fn frictionless_rolling() -> RollingContactModel {
        // Same contact, but rolling resistance disabled (μ_r = 0).
        RollingContactModel::new((2.0e4, 80.0), (2.0e4, 80.0, 0.6), (2.0e3, 40.0, 0.0)).unwrap()
    }

    fn floor() -> HalfSpace {
        HalfSpace::new(Vec3::ZERO, Vec3::Z).unwrap()
    }

    fn gravity() -> Vec3 {
        Vec3::new(0.0, 0.0, -9.81)
    }

    fn run(body: &mut GranularPileBody, steps: usize, dt: f32) {
        let n = body.particle_count();
        let zero = vec![Vec3::ZERO; n];
        for _ in 0..steps {
            let _ = body.step(dt, &zero, &zero).unwrap();
        }
    }

    #[test]
    fn new_validates_inputs() {
        assert!(GranularPileBody::new(
            model(),
            vec![Vec3::ZERO, Vec3::X],
            vec![1.0],
            vec![1.0, 1.0],
            vec![floor()],
            gravity(),
        )
        .is_none());
        assert!(GranularPileBody::new(
            model(),
            vec![Vec3::ZERO],
            vec![0.0],
            vec![1.0],
            vec![floor()],
            gravity(),
        )
        .is_none());
        assert!(GranularPileBody::new(
            model(),
            vec![Vec3::ZERO],
            vec![1.0],
            vec![1.0],
            vec![floor()],
            Vec3::new(0.0, 0.0, f32::NAN),
        )
        .is_none());
        assert!(GranularPileBody::new(
            model(),
            vec![Vec3::ZERO],
            vec![1.0],
            vec![1.0],
            vec![floor()],
            gravity(),
        )
        .is_some());
    }

    #[test]
    fn step_rejects_bad_dt_and_lengths() {
        let mut body = GranularPileBody::new(
            model(),
            vec![Vec3::new(0.0, 0.0, 1.0)],
            vec![1.0],
            vec![1.0],
            vec![floor()],
            gravity(),
        )
        .unwrap();
        assert!(body.step(0.0, &[Vec3::ZERO], &[Vec3::ZERO]).is_none());
        assert!(body.step(1.0e-3, &[], &[Vec3::ZERO]).is_none());
    }

    #[test]
    fn free_grain_falls_under_gravity() {
        // A grain well above the floor accelerates downward.
        let mut body = GranularPileBody::new(
            model(),
            vec![Vec3::new(0.0, 0.0, 10.0)],
            vec![0.5],
            vec![1.0],
            vec![floor()],
            gravity(),
        )
        .unwrap();
        let report = body.step(1.0e-3, &[Vec3::ZERO], &[Vec3::ZERO]).unwrap();
        assert_eq!(report.boundary_contact_count, 0);
        assert!(body.velocities()[0].z < 0.0);
    }

    #[test]
    fn single_grain_settles_on_floor() {
        let radius = 0.5;
        let mass = 1.0;
        let mut body = GranularPileBody::new(
            model(),
            // Start exactly touching the floor so impact injects no energy.
            vec![Vec3::new(0.0, 0.0, radius)],
            vec![radius],
            vec![mass],
            vec![floor()],
            gravity(),
        )
        .unwrap();
        run(&mut body, 4000, 1.0e-4);
        // Equilibrium overlap balances weight: kₙ·δ = m·g.
        let expected_overlap = mass * 9.81 / 2.0e4;
        let z = body.positions()[0].z;
        assert!((z - (radius - expected_overlap)).abs() < 5.0e-3);
        // The grain has come to rest.
        assert!(body.kinetic_energy() < 1.0e-4);
        assert_eq!(body.boundary_contacts(), 1);
    }

    #[test]
    fn two_grains_stack_without_tunneling() {
        let radius = 0.5;
        let mut body = GranularPileBody::new(
            model(),
            vec![
                Vec3::new(0.0, 0.0, radius),
                Vec3::new(0.0, 0.0, 3.0 * radius),
            ],
            vec![radius, radius],
            vec![1.0, 1.0],
            vec![floor()],
            gravity(),
        )
        .unwrap();
        run(&mut body, 6000, 1.0e-4);
        let lower = body.positions()[0];
        let upper = body.positions()[1];
        // Neither grain tunnels through the floor.
        assert!(lower.z > radius - 0.05);
        assert!(upper.z > radius - 0.05);
        // The upper grain rests above the lower one.
        assert!(upper.z > lower.z + radius);
        assert!(body.kinetic_energy() < 1.0e-3);
    }

    #[test]
    fn dropped_cluster_settles_into_stable_pile() {
        let radius = 0.5;
        let sqrt3 = (3.0_f64).sqrt() as f32;
        // Start near a stable three-grain pyramid: two base grains resting on
        // the floor and touching each other, with the apex grain released from
        // a small height into the valley between them.
        let positions = vec![
            Vec3::new(-radius, 0.0, radius),
            Vec3::new(radius, 0.0, radius),
            Vec3::new(0.0, 0.0, radius + radius * sqrt3 + 0.05),
        ];
        let radii = vec![radius; 3];
        let masses = vec![1.0; 3];
        let mut body =
            GranularPileBody::new(model(), positions, radii, masses, vec![floor()], gravity())
                .unwrap();
        // Track the peak kinetic energy as the apex drops into the valley.
        let n = body.particle_count();
        let zero = vec![Vec3::ZERO; n];
        let mut peak = 0.0_f32;
        for _ in 0..12000 {
            let _ = body.step(1.0e-4, &zero, &zero).unwrap();
            peak = peak.max(body.kinetic_energy());
        }
        // The apex grain actually moved under gravity.
        assert!(peak > 1.0e-3);
        // Almost all of that kinetic energy has since been dissipated.
        assert!(body.kinetic_energy() < peak * 0.02);
        assert!(body.kinetic_energy() < 5.0e-2);
        // No grain tunnels through the floor.
        for pos in body.positions() {
            assert!(pos.z > radius - 0.05);
        }
        // The pile rests on the floor.
        assert!(body.boundary_contacts() >= 1);
    }

    #[test]
    fn rolling_resistance_steepens_the_pile() {
        // Drop the same stacked pair and let the top grain roll off. Rolling
        // resistance should leave it less far from the axis than the
        // free-rolling case: a proxy for a steeper angle of repose.
        let radius = 0.5;
        let build = |m: RollingContactModel| {
            GranularPileBody::new(
                m,
                vec![
                    Vec3::new(0.0, 0.0, radius),
                    // Slightly off-axis so it rolls down the side.
                    Vec3::new(0.15, 0.0, 3.0 * radius),
                ],
                vec![radius, radius],
                vec![1.0, 1.0],
                vec![floor()],
                gravity(),
            )
            .unwrap()
        };
        let mut resisted = build(model());
        let mut free = build(frictionless_rolling());
        run(&mut resisted, 12000, 1.0e-4);
        run(&mut free, 12000, 1.0e-4);
        let spread = |b: &GranularPileBody| {
            b.positions()
                .iter()
                .map(|p| (p.x * p.x + p.y * p.y).sqrt())
                .fold(0.0_f32, f32::max)
        };
        // Rolling resistance holds the heap tighter (or at least no looser).
        assert!(spread(&resisted) <= spread(&free) + 1.0e-2);
        // The rolling-resistance pile dissipates its spin and settles. The
        // free-rolling pile may keep rolling indefinitely (a grain rolling
        // without slipping never loses energy), so only the resisted case is
        // required to come to rest.
        assert!(resisted.kinetic_energy() < 5.0e-2);
    }

    #[test]
    fn pinned_grain_ignores_gravity() {
        let radius = 0.5;
        let mut body = GranularPileBody::new(
            model(),
            vec![Vec3::new(0.0, 0.0, 5.0)],
            vec![radius],
            vec![f32::INFINITY],
            vec![floor()],
            gravity(),
        )
        .unwrap();
        assert!(body.is_pinned(0));
        run(&mut body, 100, 1.0e-3);
        // A pinned grain never moves.
        assert!((body.positions()[0] - Vec3::new(0.0, 0.0, 5.0)).length() < 1.0e-6);
        assert!(body.kinetic_energy() < 1.0e-9);
    }

    #[test]
    fn hopper_walls_confine_grains() {
        let radius = 0.5;
        let floor = HalfSpace::new(Vec3::ZERO, Vec3::Z).unwrap();
        // Two inward-facing vertical walls at x = ±1.
        let left = HalfSpace::new(Vec3::new(-1.0, 0.0, 0.0), Vec3::X).unwrap();
        let right = HalfSpace::new(Vec3::new(1.0, 0.0, 0.0), -Vec3::X).unwrap();
        let mut body = GranularPileBody::new(
            model(),
            vec![Vec3::new(0.0, 0.0, radius + 0.3)],
            vec![radius],
            vec![1.0],
            vec![floor, left, right],
            gravity(),
        )
        .unwrap();
        // Give it a sideways shove toward the right wall.
        body.set_velocity(0, Vec3::new(5.0, 0.0, 0.0));
        run(&mut body, 6000, 1.0e-4);
        let p = body.positions()[0];
        // The grain stays inside the box (its surface cannot pass the walls).
        assert!(p.x - radius > -1.0 - 0.05);
        assert!(p.x + radius < 1.0 + 0.05);
        assert!(p.z > radius - 0.05);
    }
}
