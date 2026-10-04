//! Wet granular pile: dry rotational-contact grains augmented by pendular
//! capillary liquid bridges.
//!
//! A real wet sand pile owes its cohesion (and therefore its ability to stand
//! at steep angles, be moulded, and clump) to microscopic liquid bridges that
//! form between touching grains and resist their separation up to a rupture
//! distance. This module composes two existing, independently verified pieces
//! into a single driver without touching either of them:
//!
//! * [`GranularPileBody`] — dry grain-grain and grain-boundary contact with
//!   rolling resistance, friction, and boundary half-spaces.
//! * [`CapillaryBridgeResolver`] — persistent pendular liquid-bridge cohesion
//!   with formation/rupture hysteresis (Rabinovich 2005 / Lian rupture).
//!
//! The composition is strictly additive: every integration step the capillary
//! resolver produces a per-grain cohesive force from the *current* grain
//! configuration, those forces are summed with the caller's external forces,
//! and the dry body advances under the combined load. Capillary forces are
//! central (acting along the line joining two grain centres), so they exert no
//! torque and the caller's external torques pass through unchanged.
//!
//! The structure keeps a clean zero-coupling boundary: it owns the dry body,
//! the resolver, and the bridge model, and exposes forwarding accessors so
//! callers never need to reach inside.

use glam::Vec3;

use crate::collider::capillary_bridge::CapillaryBridgeModel;
use crate::collider::capillary_bridge_resolver::CapillaryBridgeResolver;
use crate::collider::granular_pile_integrator::GranularPileBody;

/// Diagnostics returned by [`WetGranularPileBody::step`].
///
/// Combines the dry contact report with the capillary-bridge state resolved
/// for the same step, so a caller can monitor both the frictional contact
/// network and the cohesive bridge network from a single value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WetGranularPileStepReport {
    /// Number of grain-grain contacts resolved this step.
    pub grain_contact_count: usize,
    /// Number of grain-boundary contacts resolved this step.
    pub boundary_contact_count: usize,
    /// Number of live capillary bridges contributing cohesion this step.
    pub bridge_count: usize,
    /// Largest overlap across all contacts (grain and boundary) this step.
    pub max_overlap: f32,
    /// Largest normal (repulsive) contact force magnitude this step.
    pub max_normal_force: f32,
    /// Largest cohesive (attractive) capillary-bridge force magnitude this
    /// step. Zero when no bridges are live.
    pub max_cohesive_force: f32,
    /// Number of contacts at the Coulomb (sliding) limit this step.
    pub sliding_count: usize,
    /// Total kinetic energy (translational plus rotational) after the step.
    pub kinetic_energy: f32,
}

/// A wet granular pile: dry grains plus pendular capillary cohesion.
///
/// Construct from an already-validated [`GranularPileBody`] and a
/// [`CapillaryBridgeModel`]; the body carries all grain state (positions,
/// velocities, radii, masses, boundaries, gravity), while this wrapper adds a
/// persistent bridge network on top.
#[derive(Clone, Debug)]
pub struct WetGranularPileBody {
    dry: GranularPileBody,
    capillary: CapillaryBridgeResolver,
    model: CapillaryBridgeModel,
}

impl WetGranularPileBody {
    /// Wrap a dry granular pile with a capillary-bridge cohesion model.
    ///
    /// The dry body is assumed already validated by
    /// [`GranularPileBody::new`]; the capillary model is already validated by
    /// [`CapillaryBridgeModel::new`]. The bridge network starts empty and
    /// nucleates on the first step where grains touch.
    pub fn new(dry: GranularPileBody, capillary_model: CapillaryBridgeModel) -> Self {
        Self {
            dry,
            capillary: CapillaryBridgeResolver::new(),
            model: capillary_model,
        }
    }

    /// Advance the wet pile by `dt`, combining caller-supplied external forces
    /// with cohesive capillary-bridge forces resolved from the current state.
    ///
    /// Returns `None` if `ext_forces` does not have exactly one entry per
    /// grain, if the capillary resolver rejects the current configuration, or
    /// if the underlying dry step rejects its inputs (e.g. a torque-length
    /// mismatch or a non-positive `dt`). On success the bridge network is
    /// updated in place (new bridges nucleated, stretched bridges ruptured).
    pub fn step(
        &mut self,
        dt: f32,
        ext_forces: &[Vec3],
        ext_torques: &[Vec3],
    ) -> Option<WetGranularPileStepReport> {
        let n = self.dry.particle_count();
        if ext_forces.len() != n {
            return None;
        }

        // Cohesive forces are evaluated against the configuration at the start
        // of the step, matching the explicit treatment of the dry contacts.
        let cap = self
            .capillary
            .resolve(self.dry.positions(), self.dry.radii(), &self.model)?;
        let bridge_count = cap.bridge_count();
        let max_cohesive_force = cap.max_force();

        let total_ext: Vec<Vec3> = ext_forces
            .iter()
            .zip(cap.forces().iter())
            .map(|(external, cohesive)| *external + *cohesive)
            .collect();

        // Capillary forces are central, so they contribute no torque and the
        // caller's external torques pass through unchanged.
        let dry_report = self.dry.step(dt, &total_ext, ext_torques)?;

        Some(WetGranularPileStepReport {
            grain_contact_count: dry_report.grain_contact_count,
            boundary_contact_count: dry_report.boundary_contact_count,
            bridge_count,
            max_overlap: dry_report.max_overlap,
            max_normal_force: dry_report.max_normal_force,
            max_cohesive_force,
            sliding_count: dry_report.sliding_count,
            kinetic_energy: dry_report.kinetic_energy,
        })
    }

    /// Number of grains in the pile.
    pub fn particle_count(&self) -> usize {
        self.dry.particle_count()
    }

    /// Grain centre positions.
    pub fn positions(&self) -> &[Vec3] {
        self.dry.positions()
    }

    /// Grain linear velocities.
    pub fn velocities(&self) -> &[Vec3] {
        self.dry.velocities()
    }

    /// Grain angular velocities.
    pub fn angular_velocities(&self) -> &[Vec3] {
        self.dry.angular_velocities()
    }

    /// Grain radii.
    pub fn radii(&self) -> &[f32] {
        self.dry.radii()
    }

    /// Total kinetic energy (translational plus rotational).
    pub fn kinetic_energy(&self) -> f32 {
        self.dry.kinetic_energy()
    }

    /// Grain-grain contacts resolved on the most recent step.
    pub fn grain_contacts(&self) -> usize {
        self.dry.grain_contacts()
    }

    /// Grain-boundary contacts resolved on the most recent step.
    pub fn boundary_contacts(&self) -> usize {
        self.dry.boundary_contacts()
    }

    /// Number of live capillary bridges currently tracked.
    pub fn bridge_count(&self) -> usize {
        self.capillary.active_bridges()
    }

    /// Whether grains `a` and `b` currently share a live capillary bridge.
    /// Order-independent.
    pub fn is_bridged(&self, a: u32, b: u32) -> bool {
        self.capillary.is_bridged(a, b)
    }

    /// Overwrite grain `i`'s linear velocity.
    pub fn set_velocity(&mut self, i: usize, velocity: Vec3) {
        self.dry.set_velocity(i, velocity);
    }

    /// Overwrite grain `i`'s angular velocity.
    pub fn set_angular_velocity(&mut self, i: usize, angular: Vec3) {
        self.dry.set_angular_velocity(i, angular);
    }

    /// Forget all live capillary bridges (e.g. after reconfiguring the pile).
    pub fn clear_bridges(&mut self) {
        self.capillary.clear();
    }

    /// The capillary-bridge model governing cohesion.
    pub fn capillary_model(&self) -> &CapillaryBridgeModel {
        &self.model
    }

    /// Shared access to the underlying dry granular body.
    pub fn dry(&self) -> &GranularPileBody {
        &self.dry
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::granular_pile_integrator::GranularPileBody;
    use crate::collider::rotational_boundary_contact::HalfSpace;
    use crate::collider::rotational_contact::RollingContactModel;

    fn contact_model() -> RollingContactModel {
        RollingContactModel::new((2.0e4, 80.0), (2.0e4, 80.0, 0.6), (2.0e3, 40.0, 0.4)).unwrap()
    }

    fn floor() -> HalfSpace {
        HalfSpace::new(Vec3::ZERO, Vec3::Z).unwrap()
    }

    fn gravity() -> Vec3 {
        Vec3::new(0.0, 0.0, -9.81)
    }

    /// A strong capillary model so cohesion is clearly visible in a handful of
    /// steps: surface tension 1 N/m, zero contact angle, 1 mm^3 of liquid.
    fn strong_model() -> CapillaryBridgeModel {
        CapillaryBridgeModel::new(1.0, 0.0, 1.0e-3).unwrap()
    }

    /// Two unit-diameter grains resting on the floor, touching (gap == 0).
    fn two_touching_grains() -> GranularPileBody {
        let radius = 0.5;
        GranularPileBody::new(
            contact_model(),
            vec![
                Vec3::new(-radius, 0.0, radius),
                Vec3::new(radius, 0.0, radius),
            ],
            vec![radius, radius],
            vec![1.0, 1.0],
            vec![floor()],
            gravity(),
        )
        .unwrap()
    }

    #[test]
    fn construction_forwards_grain_state() {
        let wet = WetGranularPileBody::new(two_touching_grains(), strong_model());
        assert_eq!(wet.particle_count(), 2);
        assert_eq!(wet.positions().len(), 2);
        assert_eq!(wet.velocities().len(), 2);
        assert_eq!(wet.angular_velocities().len(), 2);
        assert_eq!(wet.radii().len(), 2);
        // No step taken yet: no bridges, no contacts recorded.
        assert_eq!(wet.bridge_count(), 0);
        assert!(!wet.is_bridged(0, 1));
        assert_eq!(wet.capillary_model().surface_tension(), 1.0);
    }

    #[test]
    fn step_rejects_force_length_mismatch() {
        let mut wet = WetGranularPileBody::new(two_touching_grains(), strong_model());
        // One force for a two-grain pile.
        assert!(wet
            .step(1.0e-3, &[Vec3::ZERO], &[Vec3::ZERO, Vec3::ZERO])
            .is_none());
    }

    #[test]
    fn step_rejects_torque_length_mismatch() {
        let mut wet = WetGranularPileBody::new(two_touching_grains(), strong_model());
        // Correct force count, wrong torque count: the dry step must reject it.
        assert!(wet
            .step(1.0e-3, &[Vec3::ZERO, Vec3::ZERO], &[Vec3::ZERO])
            .is_none());
    }

    #[test]
    fn touching_grains_nucleate_a_bridge() {
        let mut wet = WetGranularPileBody::new(two_touching_grains(), strong_model());
        let report = wet
            .step(1.0e-3, &[Vec3::ZERO, Vec3::ZERO], &[Vec3::ZERO, Vec3::ZERO])
            .unwrap();
        assert_eq!(report.bridge_count, 1);
        assert!(report.max_cohesive_force > 0.0);
        assert!(wet.is_bridged(0, 1));
        assert!(wet.is_bridged(1, 0));
    }

    #[test]
    fn cohesion_resists_separation_versus_dry() {
        let dt = 1.0e-3;
        let outward = 0.05;
        let steps = 200;

        // Wet pile with strong capillary bridges.
        let mut wet = WetGranularPileBody::new(two_touching_grains(), strong_model());
        wet.set_velocity(0, Vec3::new(-outward, 0.0, 0.0));
        wet.set_velocity(1, Vec3::new(outward, 0.0, 0.0));

        // Identical dry pile for comparison.
        let mut dry = two_touching_grains();
        dry.set_velocity(0, Vec3::new(-outward, 0.0, 0.0));
        dry.set_velocity(1, Vec3::new(outward, 0.0, 0.0));

        let zero = [Vec3::ZERO, Vec3::ZERO];
        for _ in 0..steps {
            wet.step(dt, &zero, &zero).unwrap();
            dry.step(dt, &zero, &zero).unwrap();
        }

        let wet_sep = (wet.positions()[1] - wet.positions()[0]).length();
        let dry_sep = (dry.positions()[1] - dry.positions()[0]).length();

        // Cohesion must leave the wet grains closer together than the dry ones.
        assert!(
            wet_sep < dry_sep,
            "wet_sep {wet_sep} should be < dry_sep {dry_sep}"
        );
        // And the bridge must survive this modest separation.
        assert!(wet.is_bridged(0, 1), "bridge ruptured unexpectedly");
    }

    #[test]
    fn bridge_ruptures_when_pulled_far_apart() {
        let dt = 1.0e-3;
        // A weak model with a short rupture distance so a fast pull ruptures it.
        let model = CapillaryBridgeModel::new(0.05, 0.0, 1.0e-9).unwrap();
        let mut wet = WetGranularPileBody::new(two_touching_grains(), model);

        // Nucleate the bridge while touching.
        let zero = [Vec3::ZERO, Vec3::ZERO];
        wet.step(dt, &zero, &zero).unwrap();
        assert!(wet.is_bridged(0, 1));

        // Yank the grains apart far beyond any plausible rupture distance.
        wet.set_velocity(0, Vec3::new(-5.0, 0.0, 0.0));
        wet.set_velocity(1, Vec3::new(5.0, 0.0, 0.0));
        for _ in 0..50 {
            wet.step(dt, &zero, &zero).unwrap();
        }
        assert!(!wet.is_bridged(0, 1), "bridge should have ruptured");
        assert_eq!(wet.bridge_count(), 0);
    }

    #[test]
    fn clear_bridges_forgets_network() {
        let mut wet = WetGranularPileBody::new(two_touching_grains(), strong_model());
        let zero = [Vec3::ZERO, Vec3::ZERO];
        wet.step(1.0e-3, &zero, &zero).unwrap();
        assert_eq!(wet.bridge_count(), 1);
        wet.clear_bridges();
        assert_eq!(wet.bridge_count(), 0);
        assert!(!wet.is_bridged(0, 1));
    }

    #[test]
    fn settles_without_tunnelling_through_floor() {
        let dt = 1.0e-3;
        let mut wet = WetGranularPileBody::new(two_touching_grains(), strong_model());
        let zero = [Vec3::ZERO, Vec3::ZERO];
        for _ in 0..500 {
            wet.step(dt, &zero, &zero).unwrap();
        }
        // Grain bottoms must stay at or above the floor (z >= 0), allowing a
        // small penetration tolerance from the penalty contact.
        for p in wet.positions() {
            assert!(p.z > 0.4, "grain sank through the floor: z = {}", p.z);
        }
    }
}
