//! Global assembly of parallel-bond (bonded-particle) forces and torques over a
//! network of discrete particles.
//!
//! This is the discrete-element counterpart of the surface
//! [`cohesive_zone_assembly`](super::cohesive_zone_assembly): it takes the
//! per-bond incremental beam kernel from
//! [`bonded_particle`](super::bonded_particle) and scatters each bond's force
//! and moment onto the two particles it connects, carrying one irreversible
//! [`BondState`] per bond.
//!
//! # Network model
//!
//! A [`ParticleBond`] couples two particles `a` and `b` by their indices. The
//! bond axis is taken from the *current* particle centres,
//! `n̂ = (x_b − x_a)/‖x_b − x_a‖`, with bond length `L = ‖x_b − x_a‖`. Each pass
//! advances every bond by the relative kinematic increments of its endpoints,
//! using exactly the convention of [`update_bond`] (particle `b` relative to
//! `a`):
//!
//! ```text
//!   Δu = Δu_b − Δu_a            relative translation increment
//!   Δθ = Δθ_b − Δθ_a            relative rotation increment
//! ```
//!
//! and then scatters the resulting internal loads onto the two particles.
//!
//! # Force and moment scatter
//!
//! After [`update_bond`] the bond holds an axial force `Fₙ` (tension positive),
//! a shear force vector `F_s`, a bending moment `M_b`, and a twisting moment
//! `Mₜ`. The force the bond exerts **on particle `a`** is
//!
//! ```text
//!   F_a = Fₙ·n̂ + F_s
//! ```
//!
//! and particle `b` receives `F_b = −F_a` by Newton's third law, so an
//! assembled pass conserves linear momentum exactly.
//!
//! The bond couple `M = Mₜ·n̂ + M_b` is applied equal and opposite to the two
//! particles. Because the bond acts at the mid-point `x_c = (x_a + x_b)/2`, the
//! off-centre force must be transported to each particle centre, adding a
//! lever-arm moment. With the mid-point application point the two transported
//! moments are
//!
//! ```text
//!   τ_a = +M + ½·(x_b − x_a) × F_a = +M + ½·L·(n̂ × F_s)
//!   τ_b = −M + ½·(x_b − x_a) × F_a = −M + ½·L·(n̂ × F_s)
//! ```
//!
//! (the axial part of `F_a` is parallel to the axis and contributes no
//! moment). This choice conserves total angular momentum of the pair about any
//! origin: `Σ xᵢ × Fᵢ + Σ τᵢ = (x_a − x_b) × F_a + 2·(½·(x_b − x_a) × F_a) = 0`.
//!
//! When a bond breaks on a step, [`update_bond`] zeroes its state, so the
//! particles receive no force or moment from it that step and none afterwards:
//! the snap is an instantaneous release, as in the standard bonded-particle
//! model.
//!
//! The returned [`BondAssembly`] keeps a per-bond [`BondNetworkStep`] alongside
//! the global force and torque vectors so a caller can report how many bonds
//! snapped and recover peak stresses without re-advancing the state. This
//! module holds no solver state of its own and performs no time integration.

use crate::collider::bonded_particle::{update_bond, BondModel, BondState, BondStep};
use glam::Vec3;

/// One parallel bond between two particles, identified by their indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParticleBond {
    /// Index of the first ("a") particle.
    pub a: u32,
    /// Index of the second ("b") particle.
    pub b: u32,
}

impl ParticleBond {
    /// Builds a bond between particles `a` and `b`.
    #[must_use]
    pub fn new(a: u32, b: u32) -> Self {
        Self { a, b }
    }

    /// Highest particle index referenced by the bond, used for bounds checks.
    #[must_use]
    fn max_index(&self) -> u32 {
        self.a.max(self.b)
    }
}

/// Per-bond outcome of one network assembly pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BondNetworkStep {
    /// The underlying parallel-bond evaluation for this bond.
    pub step: BondStep,
    /// Current bond length `L = ‖x_b − x_a‖` used this pass.
    pub length: f32,
    /// Magnitude `‖F_a‖` of the resultant force transmitted by the bond.
    pub force_magnitude: f32,
    /// Magnitude `‖M‖` of the bond couple transmitted by the bond.
    pub torque_magnitude: f32,
}

/// Aggregate result of one global bonded-particle assembly pass.
#[derive(Clone, Debug, PartialEq)]
pub struct BondAssembly {
    /// Per-particle force, one entry per input position.
    pub forces: Vec<Vec3>,
    /// Per-particle torque, one entry per input position.
    pub torques: Vec<Vec3>,
    /// Per-bond outcome, in bond order.
    pub steps: Vec<BondNetworkStep>,
}

impl BondAssembly {
    /// Number of bonds that broke on this pass.
    #[must_use]
    pub fn broke_this_step_count(&self) -> usize {
        self.steps.iter().filter(|s| s.step.broke).count()
    }

    /// Highest extreme-fibre tensile stress `σ_max` across all bonds, or `0.0`
    /// for an empty network.
    #[must_use]
    pub fn max_normal_stress(&self) -> f32 {
        self.steps
            .iter()
            .map(|s| s.step.normal_stress)
            .fold(0.0, f32::max)
    }

    /// Highest extreme-fibre shear stress `τ_max` across all bonds, or `0.0`
    /// for an empty network.
    #[must_use]
    pub fn max_shear_stress(&self) -> f32 {
        self.steps
            .iter()
            .map(|s| s.step.shear_stress)
            .fold(0.0, f32::max)
    }

    /// Sum of the force magnitudes transmitted by every bond this pass.
    #[must_use]
    pub fn total_force_magnitude(&self) -> f32 {
        self.steps.iter().map(|s| s.force_magnitude).sum()
    }
}

/// Builds `count` fresh intact bond states, one per bond.
#[must_use]
pub fn rest_bond_states(count: usize) -> Vec<BondState> {
    vec![BondState::intact(); count]
}

/// Number of bonds in `states` that have broken.
#[must_use]
pub fn broken_bond_count(states: &[BondState]) -> usize {
    states.iter().filter(|s| s.is_broken()).count()
}

/// Number of bonds in `states` that are still intact.
#[must_use]
pub fn intact_bond_count(states: &[BondState]) -> usize {
    states.iter().filter(|s| !s.is_broken()).count()
}

/// Validates that the bonds, position / increment arrays, and state slice are
/// mutually consistent: one state per bond, equal-length position and
/// increment arrays, no self-bonds, and every referenced particle addressable.
fn is_consistent(
    bonds: &[ParticleBond],
    positions: &[Vec3],
    delta_disp: &[Vec3],
    delta_rot: &[Vec3],
    states: &[BondState],
) -> bool {
    if states.len() != bonds.len() {
        return false;
    }
    if delta_disp.len() != positions.len() || delta_rot.len() != positions.len() {
        return false;
    }
    let n = positions.len() as u32;
    bonds
        .iter()
        .all(|bond| bond.a != bond.b && bond.max_index() < n)
}

/// Assembles the per-particle force and torque vectors of a bonded-particle
/// network, advancing each bond's state once.
///
/// `states` must hold exactly one [`BondState`] per bond (see
/// [`rest_bond_states`]); each is updated in place with the relative kinematic
/// increments of its endpoints. `delta_disp` and `delta_rot` are the
/// per-particle translational and rotational increments since the previous
/// pass, one entry per particle position.
///
/// Returns `None` when the lengths disagree, when any bond is a self-bond, or
/// when a bond references a particle index out of range; in that case no state
/// is mutated. A bond whose endpoints are coincident (`L ≈ 0`) has an undefined
/// axis and is treated as a no-op for that pass, leaving its state untouched
/// and contributing no force or moment.
#[must_use]
pub fn assemble_bond_forces(
    bonds: &[ParticleBond],
    positions: &[Vec3],
    delta_disp: &[Vec3],
    delta_rot: &[Vec3],
    model: &BondModel,
    states: &mut [BondState],
) -> Option<BondAssembly> {
    if !is_consistent(bonds, positions, delta_disp, delta_rot, states) {
        return None;
    }

    let mut forces = vec![Vec3::ZERO; positions.len()];
    let mut torques = vec![Vec3::ZERO; positions.len()];
    let mut steps = Vec::with_capacity(bonds.len());

    for (bond, state) in bonds.iter().zip(states.iter_mut()) {
        let a = bond.a as usize;
        let b = bond.b as usize;

        let axis = positions[b] - positions[a];
        let length = axis.length();

        // Relative increments of particle b with respect to particle a, matching
        // the convention of `update_bond`.
        let delta_u = delta_disp[b] - delta_disp[a];
        let delta_theta = delta_rot[b] - delta_rot[a];

        let step = update_bond(model, axis, delta_u, delta_theta, state);

        // Resolve the bond loads into particle contributions. A degenerate axis
        // (coincident particles) or a bond broken this step both leave the state
        // with no stored load, so the contributions below vanish.
        let (force_on_a, couple) = if length > f32::EPSILON {
            let n = axis / length;
            let force = state.normal_force() * n + state.shear_force();
            let moment = state.twisting_moment() * n + state.bending_moment();
            (force, moment)
        } else {
            (Vec3::ZERO, Vec3::ZERO)
        };

        // Transport the off-centre force from the bond mid-point to each centre;
        // only the shear part survives the cross product with the axis.
        let transport = 0.5 * axis.cross(force_on_a);

        forces[a] += force_on_a;
        forces[b] -= force_on_a;
        torques[a] += couple + transport;
        torques[b] += -couple + transport;

        steps.push(BondNetworkStep {
            step,
            length,
            force_magnitude: force_on_a.length(),
            torque_magnitude: couple.length(),
        });
    }

    Some(BondAssembly {
        forces,
        torques,
        steps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const X: Vec3 = Vec3::new(1.0, 0.0, 0.0);

    // kn = ks = 1e9 (per area), R = 0.01 → A ≈ 3.14e-4.
    // σ_c = 1e6, c = 1e6, μ = 0.5.
    fn model() -> BondModel {
        BondModel::new(1.0e9, 1.0e9, 0.01, 1.0e6, 1.0e6, 0.5).unwrap()
    }

    // Two particles one unit apart along +x.
    fn pair() -> Vec<Vec3> {
        vec![Vec3::ZERO, X]
    }

    fn zeros(n: usize) -> Vec<Vec3> {
        vec![Vec3::ZERO; n]
    }

    #[test]
    fn rest_bond_states_are_intact() {
        let states = rest_bond_states(3);
        assert_eq!(states.len(), 3);
        assert!(states.iter().all(|s| !s.is_broken()));
    }

    #[test]
    fn rejects_mismatched_state_length() {
        let bonds = [ParticleBond::new(0, 1)];
        let pos = pair();
        let mut states = rest_bond_states(2);
        assert!(
            assemble_bond_forces(&bonds, &pos, &zeros(2), &zeros(2), &model(), &mut states)
                .is_none()
        );
    }

    #[test]
    fn rejects_mismatched_increment_length() {
        let bonds = [ParticleBond::new(0, 1)];
        let pos = pair();
        let mut states = rest_bond_states(1);
        assert!(
            assemble_bond_forces(&bonds, &pos, &zeros(1), &zeros(2), &model(), &mut states)
                .is_none()
        );
    }

    #[test]
    fn rejects_out_of_range_index() {
        let bonds = [ParticleBond::new(0, 5)];
        let pos = pair();
        let mut states = rest_bond_states(1);
        assert!(
            assemble_bond_forces(&bonds, &pos, &zeros(2), &zeros(2), &model(), &mut states)
                .is_none()
        );
    }

    #[test]
    fn rejects_self_bond() {
        let bonds = [ParticleBond::new(1, 1)];
        let pos = pair();
        let mut states = rest_bond_states(1);
        assert!(
            assemble_bond_forces(&bonds, &pos, &zeros(2), &zeros(2), &model(), &mut states)
                .is_none()
        );
    }

    #[test]
    fn empty_network_is_empty_assembly() {
        let mut states = rest_bond_states(0);
        let out = assemble_bond_forces(&[], &pair(), &zeros(2), &zeros(2), &model(), &mut states)
            .expect("empty network is consistent");
        assert_eq!(out.steps.len(), 0);
        assert_eq!(out.forces.len(), 2);
        assert!(out.forces.iter().all(|f| *f == Vec3::ZERO));
        assert!(out.torques.iter().all(|t| *t == Vec3::ZERO));
    }

    #[test]
    fn axial_tension_scatters_opposite_forces() {
        let bonds = [ParticleBond::new(0, 1)];
        let pos = pair();
        // b displaced +x by 1e-4 → σ = kn·δn = 1e5 < σ_c, intact tension.
        let mut dd = zeros(2);
        dd[1] = X * 1.0e-4;
        let mut states = rest_bond_states(1);
        let out = assemble_bond_forces(&bonds, &pos, &dd, &zeros(2), &model(), &mut states)
            .expect("consistent");
        assert!(!out.steps[0].step.broke);
        // Tension pulls a toward b (+x) and b toward a (−x).
        assert!(out.forces[0].x > 0.0);
        assert!(out.forces[1].x < 0.0);
        // Linear momentum conserved; axial load gives no torque.
        assert!((out.forces[0] + out.forces[1]).length() < 1e-3);
        assert!(out.torques[0].length() < 1e-6);
        assert!(out.torques[1].length() < 1e-6);
    }

    #[test]
    fn axial_compression_pushes_particles_apart() {
        let bonds = [ParticleBond::new(0, 1)];
        let pos = pair();
        // b displaced −x: compression. Force on a points −x (away from b).
        let mut dd = zeros(2);
        dd[1] = X * -1.0e-4;
        let mut states = rest_bond_states(1);
        let out = assemble_bond_forces(&bonds, &pos, &dd, &zeros(2), &model(), &mut states)
            .expect("consistent");
        assert!(out.forces[0].x < 0.0, "compression pushes a away from b");
        assert!(out.forces[1].x > 0.0);
        assert!((out.forces[0] + out.forces[1]).length() < 1e-3);
    }

    #[test]
    fn shear_produces_transport_torque() {
        let bonds = [ParticleBond::new(0, 1)];
        let pos = pair();
        // b displaced +y by 9e-4 → τ = 9e5 < c, intact shear.
        let mut dd = zeros(2);
        dd[1] = Vec3::new(0.0, 9.0e-4, 0.0);
        let mut states = rest_bond_states(1);
        let out = assemble_bond_forces(&bonds, &pos, &dd, &zeros(2), &model(), &mut states)
            .expect("consistent");
        assert!(!out.steps[0].step.broke);
        // Shear resultant along ±y, both particles get a +z transport torque.
        assert!(out.forces[0].y > 0.0);
        assert!(out.torques[0].z > 0.0);
        assert!(out.torques[1].z > 0.0);
        // Total angular momentum about the origin vanishes.
        let orbital = pos[0].cross(out.forces[0]) + pos[1].cross(out.forces[1]);
        let spin = out.torques[0] + out.torques[1];
        assert!((orbital + spin).length() < 1e-3);
    }

    #[test]
    fn twist_scatters_opposite_couples() {
        let bonds = [ParticleBond::new(0, 1)];
        let pos = pair();
        // Relative twist about the +x bond axis, kept below the shear limit.
        let mut dr = zeros(2);
        dr[1] = X * 5.0e-5;
        let mut states = rest_bond_states(1);
        let out = assemble_bond_forces(&bonds, &pos, &zeros(2), &dr, &model(), &mut states)
            .expect("consistent");
        assert!(!out.steps[0].step.broke);
        // No translation → no transport term → equal and opposite couples.
        assert!(out.torques[0].x > 0.0);
        assert!(out.torques[1].x < 0.0);
        assert!((out.torques[0] + out.torques[1]).length() < 1e-6);
        assert!(out.forces[0].length() < 1e-6);
    }

    #[test]
    fn breaking_bond_releases_all_load() {
        let bonds = [ParticleBond::new(0, 1)];
        let pos = pair();
        // δn = 2e-3 ⇒ σ = 2e6 ≥ σ_c → tensile break.
        let mut dd = zeros(2);
        dd[1] = X * 2.0e-3;
        let mut states = rest_bond_states(1);
        let out = assemble_bond_forces(&bonds, &pos, &dd, &zeros(2), &model(), &mut states)
            .expect("consistent");
        assert!(out.steps[0].step.broke);
        assert_eq!(out.broke_this_step_count(), 1);
        assert_eq!(broken_bond_count(&states), 1);
        assert_eq!(intact_bond_count(&states), 0);
        // A snapped bond releases its load instantly.
        assert!(out.forces[0].length() < 1e-6);
        assert!(out.forces[1].length() < 1e-6);
        assert!(out.torques[0].length() < 1e-6);
    }

    #[test]
    fn coincident_particles_are_a_noop() {
        let bonds = [ParticleBond::new(0, 1)];
        let pos = vec![Vec3::ZERO, Vec3::ZERO];
        let mut dd = zeros(2);
        dd[1] = X * 1.0;
        let mut states = rest_bond_states(1);
        let out = assemble_bond_forces(&bonds, &pos, &dd, &zeros(2), &model(), &mut states)
            .expect("consistent");
        assert!(!out.steps[0].step.broke);
        assert_eq!(out.steps[0].length, 0.0);
        assert!(out.forces[0].length() < 1e-9);
        assert!(!states[0].is_broken());
    }

    #[test]
    fn multi_bond_conserves_momentum_and_angular_momentum() {
        // A small chain of three particles with two bonds, loaded with mixed
        // translational and rotational increments that stay below strength.
        let pos = vec![Vec3::ZERO, X, Vec3::new(2.0, 0.0, 0.0)];
        let bonds = [ParticleBond::new(0, 1), ParticleBond::new(1, 2)];
        let dd = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0e-4, 2.0e-4, -1.0e-4),
            Vec3::new(-5.0e-5, 1.0e-4, 3.0e-5),
        ];
        let dr = vec![
            Vec3::ZERO,
            Vec3::new(1.0e-5, -2.0e-5, 1.0e-5),
            Vec3::new(0.0, 1.0e-5, -1.0e-5),
        ];
        let mut states = rest_bond_states(2);
        let out = assemble_bond_forces(&bonds, &pos, &dd, &dr, &model(), &mut states)
            .expect("consistent");
        assert!(out.steps.iter().all(|s| !s.step.broke));
        // Linear momentum: Σ F = 0.
        let net_force: Vec3 = out.forces.iter().copied().sum();
        assert!(net_force.length() < 1e-3, "net force {net_force:?}");
        // Angular momentum about the origin: Σ xᵢ × Fᵢ + Σ τᵢ = 0.
        let orbital: Vec3 = pos
            .iter()
            .zip(out.forces.iter())
            .map(|(x, f)| x.cross(*f))
            .sum();
        let spin: Vec3 = out.torques.iter().copied().sum();
        assert!((orbital + spin).length() < 1e-3, "angular residual");
    }

    #[test]
    fn stress_maxima_track_the_loaded_bond() {
        let pos = vec![Vec3::ZERO, X, Vec3::new(2.0, 0.0, 0.0)];
        let bonds = [ParticleBond::new(0, 1), ParticleBond::new(1, 2)];
        // Only the second bond is loaded (particle 2 pulled +x).
        let mut dd = zeros(3);
        dd[2] = X * 1.0e-4;
        let mut states = rest_bond_states(2);
        let out =
            assemble_bond_forces(&bonds, &pos, &dd, &zeros(3), &model(), &mut states).expect("ok");
        assert!(out.max_normal_stress() > 0.0);
        assert!(out.total_force_magnitude() > 0.0);
        assert_eq!(out.broke_this_step_count(), 0);
    }
}
