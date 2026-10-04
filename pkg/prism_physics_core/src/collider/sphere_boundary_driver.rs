//! Static half-space boundary forces for a sphere cloud.
//!
//! The DEM integrators in
//! [`sphere_dem_integrator`](super::sphere_dem_integrator) and
//! [`sphere_dem_friction_integrator`](super::sphere_dem_friction_integrator)
//! resolve grain–grain contacts, but a real granular scene also needs the
//! grains to collide with *static boundaries*: the floor of a box, the slanted
//! walls of a hopper, or the curved shell of a rotating drum. This module
//! supplies that missing half of the contact set as a small, self-contained
//! driver that is deliberately decoupled from the time integrator so it can be
//! composed with any of them.
//!
//! # Model
//!
//! Each boundary is an infinite [`HalfSpace`] with outward unit normal `n̂`
//! (pointing into the region the grains occupy). A grain of radius `r` whose
//! centre has signed distance `d` from the plane penetrates it when
//!
//! ```text
//!   δ = r − d > 0,
//! ```
//!
//! and the contact is resolved with the same [`HertzModel`] used for
//! grain–grain pairs. The plane is treated as the stationary partner `a` and
//! the grain as the moving partner `b`, so the contact axis is the plane normal
//! and the relative velocity is simply the grain velocity. Because a plane has
//! infinite radius the reduced contact radius collapses to the grain radius
//! `R* = r`, which is passed to [`evaluate_hertz_contact`].
//!
//! The response therefore inherits the full Hertz law: a `δ^{3/2}` elastic
//! penalty, viscous normal damping that dissipates energy on approach, and a
//! viscous-plus-Coulomb tangential friction opposing sliding along the wall.
//! The driver is **memoryless** — it stores no per-contact history — so it adds
//! no rotational state and can be re-run every substep without bookkeeping.
//!
//! Forces from every boundary a grain touches are accumulated into a single
//! per-grain vector, so corners (two or three mutually perpendicular planes)
//! and wedges fall out naturally.

use crate::collider::hertz_contact::{evaluate_hertz_contact, HertzModel};
use crate::collider::rotational_boundary_contact::HalfSpace;
use glam::Vec3;

/// Resolved static-boundary forces for a sphere cloud.
///
/// `forces` is indexed in lockstep with the grain arrays passed to
/// [`SphereBoundaryDriver::resolve`]; a grain touching no boundary keeps a zero
/// entry. The scalar summaries describe the whole boundary contact set.
#[derive(Clone, Debug, PartialEq)]
pub struct SphereBoundaryResolution {
    forces: Vec<Vec3>,
    contact_count: u32,
    sliding_count: u32,
    max_normal_force: f32,
    max_penetration: f32,
    total_force: Vec3,
}

impl SphereBoundaryResolution {
    /// Per-grain accumulated boundary force, indexed like the input cloud.
    #[must_use]
    pub fn forces(&self) -> &[Vec3] {
        &self.forces
    }

    /// Number of grain–boundary contacts resolved (a grain in a corner counts
    /// once per plane it penetrates).
    #[must_use]
    pub fn contact_count(&self) -> u32 {
        self.contact_count
    }

    /// Number of resolved contacts whose tangential response saturated the
    /// Coulomb limit, i.e. the grain was sliding along the wall.
    #[must_use]
    pub fn sliding_count(&self) -> u32 {
        self.sliding_count
    }

    /// Largest normal-force magnitude over every resolved contact.
    #[must_use]
    pub fn max_normal_force(&self) -> f32 {
        self.max_normal_force
    }

    /// Deepest penetration `δ` observed over every resolved contact.
    #[must_use]
    pub fn max_penetration(&self) -> f32 {
        self.max_penetration
    }

    /// Vector sum of all per-grain boundary forces. Unlike grain–grain
    /// contacts this does **not** cancel: a static wall is an external agent, so
    /// the net force on the cloud is generally non-zero.
    #[must_use]
    pub fn total_force(&self) -> Vec3 {
        self.total_force
    }
}

/// A frictional Hertz contact driver between a sphere cloud and a set of static
/// [`HalfSpace`] boundaries.
#[derive(Clone, Debug)]
pub struct SphereBoundaryDriver {
    model: HertzModel,
    boundaries: Vec<HalfSpace>,
}

impl SphereBoundaryDriver {
    /// Creates a driver with the given contact law and no boundaries yet.
    #[must_use]
    pub fn new(model: HertzModel) -> Self {
        Self {
            model,
            boundaries: Vec::new(),
        }
    }

    /// Creates a driver seeded with an initial set of boundaries.
    #[must_use]
    pub fn with_boundaries(model: HertzModel, boundaries: Vec<HalfSpace>) -> Self {
        Self { model, boundaries }
    }

    /// Appends a static boundary. Returns `&mut self` for fluent construction.
    pub fn add_boundary(&mut self, boundary: HalfSpace) -> &mut Self {
        self.boundaries.push(boundary);
        self
    }

    /// Number of static boundaries currently registered.
    #[must_use]
    pub fn boundary_count(&self) -> usize {
        self.boundaries.len()
    }

    /// The static boundaries, in insertion order.
    #[must_use]
    pub fn boundaries(&self) -> &[HalfSpace] {
        &self.boundaries
    }

    /// The contact law applied at every boundary.
    #[must_use]
    pub fn model(&self) -> &HertzModel {
        &self.model
    }

    /// Resolves the boundary forces acting on a sphere cloud.
    ///
    /// `positions`, `radii`, and `velocities` describe the grains and must have
    /// equal, non-zero length with finite entries and strictly positive radii;
    /// any violation returns `None`. With no boundaries registered the result
    /// holds all-zero forces and empty summaries.
    #[must_use]
    pub fn resolve(
        &self,
        positions: &[Vec3],
        radii: &[f32],
        velocities: &[Vec3],
    ) -> Option<SphereBoundaryResolution> {
        let count = positions.len();
        if count == 0 || radii.len() != count || velocities.len() != count {
            return None;
        }
        for (&center, (&radius, &velocity)) in
            positions.iter().zip(radii.iter().zip(velocities.iter()))
        {
            if !(center.is_finite() && velocity.is_finite() && radius.is_finite() && radius > 0.0) {
                return None;
            }
        }

        let mut forces = vec![Vec3::ZERO; count];
        let mut contact_count = 0_u32;
        let mut sliding_count = 0_u32;
        let mut max_normal_force = 0.0_f32;
        let mut max_penetration = 0.0_f32;
        let mut total_force = Vec3::ZERO;

        for (i, &center) in positions.iter().enumerate() {
            let radius = radii[i];
            let velocity = velocities[i];
            for boundary in &self.boundaries {
                let distance = boundary.signed_distance(center);
                let overlap = radius - distance;
                if !(overlap.is_finite() && overlap > 0.0) {
                    continue;
                }
                let contact = evaluate_hertz_contact(
                    &self.model,
                    boundary.normal(),
                    overlap,
                    radius,
                    velocity,
                );
                if contact.normal_magnitude <= 0.0 {
                    continue;
                }
                forces[i] += contact.force_on_b;
                total_force += contact.force_on_b;
                contact_count += 1;
                if contact.sliding {
                    sliding_count += 1;
                }
                max_normal_force = max_normal_force.max(contact.normal_magnitude);
                max_penetration = max_penetration.max(overlap);
            }
        }

        Some(SphereBoundaryResolution {
            forces,
            contact_count,
            sliding_count,
            max_normal_force,
            max_penetration,
            total_force,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> HertzModel {
        HertzModel::new(1.0e7, 0.3, 10.0, 10.0, 0.5).expect("valid Hertz model")
    }

    fn floor() -> HalfSpace {
        HalfSpace::new(Vec3::ZERO, Vec3::Z).expect("valid floor")
    }

    #[test]
    fn rejects_mismatched_or_degenerate_input() {
        let driver = SphereBoundaryDriver::with_boundaries(model(), vec![floor()]);
        // Length mismatch.
        assert!(driver
            .resolve(&[Vec3::new(0.0, 0.0, 0.4)], &[0.5, 0.5], &[Vec3::ZERO])
            .is_none());
        // Empty cloud.
        assert!(driver.resolve(&[], &[], &[]).is_none());
        // Non-positive radius.
        assert!(driver
            .resolve(&[Vec3::new(0.0, 0.0, 0.4)], &[0.0], &[Vec3::ZERO])
            .is_none());
        // Non-finite position.
        assert!(driver
            .resolve(&[Vec3::new(f32::NAN, 0.0, 0.4)], &[0.5], &[Vec3::ZERO])
            .is_none());
    }

    #[test]
    fn no_boundaries_yields_zero_forces() {
        let driver = SphereBoundaryDriver::new(model());
        assert_eq!(driver.boundary_count(), 0);
        let res = driver
            .resolve(&[Vec3::new(0.0, 0.0, 0.4)], &[0.5], &[Vec3::ZERO])
            .expect("resolve");
        assert_eq!(res.contact_count(), 0);
        assert_eq!(res.forces(), &[Vec3::ZERO]);
        assert_eq!(res.total_force(), Vec3::ZERO);
    }

    #[test]
    fn grain_above_floor_has_no_contact() {
        let driver = SphereBoundaryDriver::with_boundaries(model(), vec![floor()]);
        // Centre at z = 1.0 with radius 0.5 sits clear of the floor.
        let res = driver
            .resolve(&[Vec3::new(0.0, 0.0, 1.0)], &[0.5], &[Vec3::ZERO])
            .expect("resolve");
        assert_eq!(res.contact_count(), 0);
        assert_eq!(res.forces()[0], Vec3::ZERO);
    }

    #[test]
    fn penetrating_grain_is_pushed_up() {
        let driver = SphereBoundaryDriver::with_boundaries(model(), vec![floor()]);
        // Centre at z = 0.4 with radius 0.5 penetrates the floor by 0.1.
        let res = driver
            .resolve(&[Vec3::new(0.0, 0.0, 0.4)], &[0.5], &[Vec3::ZERO])
            .expect("resolve");
        assert_eq!(res.contact_count(), 1);
        let force = res.forces()[0];
        assert!(force.z > 0.0, "floor must push the grain up, got {force:?}");
        assert!(force.x.abs() <= 1.0e-5 && force.y.abs() <= 1.0e-5);
        assert!(res.max_normal_force() > 0.0);
        assert!((res.max_penetration() - 0.1).abs() <= 1.0e-5);
        // A single static wall imparts a net upward force on the cloud.
        assert_eq!(res.total_force(), force);
    }

    #[test]
    fn approach_velocity_adds_damping_force() {
        let driver = SphereBoundaryDriver::with_boundaries(model(), vec![floor()]);
        let positions = [Vec3::new(0.0, 0.0, 0.4)];
        let radii = [0.5];
        let resting = driver
            .resolve(&positions, &radii, &[Vec3::ZERO])
            .expect("resting");
        let approaching = driver
            .resolve(&positions, &radii, &[Vec3::new(0.0, 0.0, -2.0)])
            .expect("approaching");
        // Downward approach velocity dissipates energy via a larger repulsion.
        assert!(
            approaching.forces()[0].z > resting.forces()[0].z,
            "approach damping should increase the normal force"
        );
    }

    #[test]
    fn corner_accumulates_two_boundaries() {
        let wall = HalfSpace::new(Vec3::ZERO, Vec3::X).expect("wall");
        let driver = SphereBoundaryDriver::with_boundaries(model(), vec![floor(), wall]);
        // Centre near the floor/wall corner penetrates both planes.
        let res = driver
            .resolve(&[Vec3::new(0.4, 2.0, 0.4)], &[0.5], &[Vec3::ZERO])
            .expect("resolve");
        assert_eq!(res.contact_count(), 2);
        let force = res.forces()[0];
        assert!(force.z > 0.0 && force.x > 0.0, "corner force {force:?}");
    }

    #[test]
    fn sliding_grain_saturates_coulomb_limit() {
        let driver = SphereBoundaryDriver::with_boundaries(model(), vec![floor()]);
        // Deeply penetrating grain dragged fast along +x: viscous friction
        // exceeds the Coulomb cap, so the contact is flagged as sliding and the
        // tangential force opposes the motion.
        let res = driver
            .resolve(
                &[Vec3::new(0.0, 0.0, 0.4)],
                &[0.5],
                &[Vec3::new(1.0e5, 0.0, 0.0)],
            )
            .expect("resolve");
        assert_eq!(res.contact_count(), 1);
        assert_eq!(res.sliding_count(), 1);
        let force = res.forces()[0];
        assert!(
            force.x < 0.0,
            "friction must oppose +x motion, got {force:?}"
        );
        assert!(force.z > 0.0, "normal response must remain upward");
    }

    #[test]
    fn total_force_matches_sum_of_per_grain_forces() {
        let driver = SphereBoundaryDriver::with_boundaries(model(), vec![floor()]);
        let positions = [
            Vec3::new(0.0, 0.0, 0.45),
            Vec3::new(3.0, 0.0, 0.40),
            Vec3::new(6.0, 0.0, 2.00),
        ];
        let radii = [0.5, 0.5, 0.5];
        let velocities = [Vec3::ZERO; 3];
        let res = driver
            .resolve(&positions, &radii, &velocities)
            .expect("resolve");
        let summed: Vec3 = res.forces().iter().copied().sum();
        assert!((summed - res.total_force()).length() <= 1.0e-3);
        // Two grains touch the floor, the third is clear.
        assert_eq!(res.contact_count(), 2);
    }
}
