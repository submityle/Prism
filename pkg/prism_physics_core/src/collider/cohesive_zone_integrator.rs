//! Explicit dynamic driver for a cohesive-interface body.
//!
//! This closes the cohesive-zone pipeline into something that advances in time.
//! The interface builder
//! ([`insert_cohesive_interfaces`](super::cohesive_interface_builder::insert_cohesive_interfaces))
//! splits a tetrahedral mesh and installs one
//! [`CohesiveInterface`](super::cohesive_zone_assembly::CohesiveInterface) per
//! opened facet; the assembly layer
//! ([`assemble_cohesive_forces`](super::cohesive_zone_assembly::assemble_cohesive_forces))
//! scatters the mixed-mode traction–separation response of every facet onto its
//! vertices; and this module integrates the resulting vertex motion with a
//! symplectic (semi-implicit Euler) step while carrying each facet's
//! irreversible decohesion history across steps.
//!
//! # State
//!
//! A [`CohesiveBody`] owns the *rest* positions of every vertex (the reference
//! configuration the traction–separation law measures the displacement jump
//! against), the *current* positions and velocities, the per-vertex mass, the
//! installed interfaces, and one
//! [`CohesiveState`](super::cohesive_zone::CohesiveState) per interface. A
//! vertex whose mass is non-finite is treated as *pinned*: it feels no
//! acceleration and keeps its prescribed velocity, which is the usual way to
//! impose a kinematic boundary condition on a fracturing solid.
//!
//! # Time step
//!
//! Each [`CohesiveBody::step`] advances the body by `dt`:
//!
//! ```text
//!   F            = assemble_cohesive_forces(rest, current) + external
//!   vᵢ += dt·Fᵢ/mᵢ            (pinned vertices unchanged)
//!   xᵢ += dt·vᵢ               (symplectic position update)
//! ```
//!
//! Because the cohesive force an interface exerts on its two faces is
//! equal-and-opposite, a step with no external loads conserves linear momentum
//! exactly (`Σ mᵢ Δvᵢ = dt·Σ Fᵢ = 0`).

use crate::collider::cohesive_zone::{dissipated_energy, CohesiveModel, CohesiveState};
use crate::collider::cohesive_zone_assembly::{
    assemble_cohesive_forces, rest_states, CohesiveInterface,
};
use glam::Vec3;

/// Diagnostic summary of one cohesive integration step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CohesiveStepReport {
    /// Number of facets whose irreversible history advanced on this step.
    pub advanced_facets: usize,
    /// Number of facets carrying any decohesion damage (`d > 0`).
    pub damaged_facets: usize,
    /// Number of facets that are fully decohered (`d = 1`).
    pub decohered_facets: usize,
    /// Mean decohesion damage `d ∈ [0, 1]` across all facets.
    pub mean_damage: f32,
    /// Largest resultant interface force magnitude across all facets this step.
    pub max_force_magnitude: f32,
    /// Total fracture energy dissipated by decohesion so far.
    pub dissipated_energy: f32,
    /// Total translational kinetic energy after the step.
    pub kinetic_energy: f32,
}

/// A time-steppable cohesive-interface body.
#[derive(Clone, Debug, PartialEq)]
pub struct CohesiveBody {
    model: CohesiveModel,
    interfaces: Vec<CohesiveInterface>,
    states: Vec<CohesiveState>,
    /// Reference (rest) configuration the traction law measures jumps against.
    rest_positions: Vec<Vec3>,
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    masses: Vec<f32>,
    /// Reference facet areas captured once at construction (rest geometry is
    /// fixed), used to report dissipated energy without re-assembling.
    facet_areas: Vec<f32>,
}

impl CohesiveBody {
    /// Builds a body from a cohesive `model`, the installed `interfaces`, the
    /// reference `rest_positions`, and per-vertex `masses`. Current positions
    /// start at the rest configuration, velocities start at rest, and every
    /// interface starts pristine (`κ = 0`).
    ///
    /// A vertex whose mass is non-finite is *pinned*. Returns `None` when the
    /// arrays disagree in length, when any rest position is non-finite, when a
    /// mass is NaN or finite-but-non-positive, when an interface references an
    /// out-of-range vertex, or when any facet's rest triangle is degenerate
    /// (zero area) — the latter two are detected by a dry-run assembly on the
    /// rest configuration, which leaves every facet at rest.
    #[must_use]
    pub fn new(
        model: CohesiveModel,
        interfaces: Vec<CohesiveInterface>,
        rest_positions: Vec<Vec3>,
        masses: Vec<f32>,
    ) -> Option<Self> {
        let n = rest_positions.len();
        if masses.len() != n {
            return None;
        }
        if rest_positions
            .iter()
            .any(|p| !(p.x.is_finite() && p.y.is_finite() && p.z.is_finite()))
        {
            return None;
        }
        // A mass is valid if it is NaN-free and either infinite (pinned) or
        // strictly positive.
        let valid_scalar = |v: f32| !v.is_nan() && (v.is_infinite() || v > 0.0);
        if !masses.iter().copied().all(valid_scalar) {
            return None;
        }

        // Dry-run assembly on the rest configuration validates every vertex
        // index and rejects degenerate facets; with current == rest the jump is
        // zero everywhere, so no state advances. The returned per-facet areas
        // are the fixed reference areas we keep for energy reporting.
        let mut scratch = rest_states(interfaces.len());
        let probe = assemble_cohesive_forces(
            &interfaces,
            &rest_positions,
            &rest_positions,
            &model,
            &mut scratch,
        )?;
        let facet_areas: Vec<f32> = probe.steps.iter().map(|s| s.area).collect();

        let positions = rest_positions.clone();
        Some(Self {
            model,
            states: rest_states(interfaces.len()),
            interfaces,
            rest_positions,
            velocities: vec![Vec3::ZERO; n],
            positions,
            masses,
            facet_areas,
        })
    }

    /// Number of vertices in the body.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    /// Number of cohesive interface facets in the body.
    #[must_use]
    pub fn interface_count(&self) -> usize {
        self.interfaces.len()
    }

    /// Reference (rest) vertex positions.
    #[must_use]
    pub fn rest_positions(&self) -> &[Vec3] {
        &self.rest_positions
    }

    /// Current vertex positions.
    #[must_use]
    pub fn positions(&self) -> &[Vec3] {
        &self.positions
    }

    /// Current vertex velocities.
    #[must_use]
    pub fn velocities(&self) -> &[Vec3] {
        &self.velocities
    }

    /// Whether vertex `i` is pinned (non-finite mass).
    #[must_use]
    pub fn is_pinned(&self, i: usize) -> bool {
        self.masses.get(i).is_some_and(|m| !m.is_finite())
    }

    /// Sets the velocity of vertex `i`, if it exists.
    pub fn set_velocity(&mut self, i: usize, velocity: Vec3) {
        if let Some(v) = self.velocities.get_mut(i) {
            *v = velocity;
        }
    }

    /// Number of facets that are fully decohered (`d = 1`).
    #[must_use]
    pub fn decohered_count(&self) -> usize {
        self.states
            .iter()
            .filter(|s| self.model.damage_at(s.kappa()) >= 1.0)
            .count()
    }

    /// Mean decohesion damage `d ∈ [0, 1]` across all facets, or `0.0` when the
    /// body has no interfaces.
    #[must_use]
    pub fn mean_damage(&self) -> f32 {
        if self.states.is_empty() {
            return 0.0;
        }
        let sum: f32 = self
            .states
            .iter()
            .map(|s| self.model.damage_at(s.kappa()))
            .sum();
        sum / self.states.len() as f32
    }

    /// Total fracture energy dissipated by decohesion across all facets, using
    /// the fixed reference areas and the current irreversible histories.
    #[must_use]
    pub fn dissipated_energy(&self) -> f32 {
        let mut total = 0.0;
        for (state, &area) in self.states.iter().zip(self.facet_areas.iter()) {
            total += area * dissipated_energy(&self.model, state.kappa());
        }
        total
    }

    /// Total translational kinetic energy of the finite-mass vertices.
    #[must_use]
    pub fn kinetic_energy(&self) -> f32 {
        let mut energy = 0.0;
        for i in 0..self.positions.len() {
            if self.masses[i].is_finite() {
                energy += 0.5 * self.masses[i] * self.velocities[i].length_squared();
            }
        }
        energy
    }

    /// Total linear momentum of the finite-mass vertices.
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

    /// Advances the body by `dt` under optional per-vertex external forces,
    /// re-evaluating and integrating the cohesive interface network once.
    ///
    /// `external_forces` must have one entry per vertex. Returns `None` when
    /// `dt` is not strictly positive and finite, when `external_forces` has the
    /// wrong length, or when the internal cohesive assembly rejects the state
    /// (which cannot happen for a well-formed body). On `None` the body is left
    /// unchanged.
    #[must_use]
    pub fn step(&mut self, dt: f32, external_forces: &[Vec3]) -> Option<CohesiveStepReport> {
        let n = self.positions.len();
        if !(dt.is_finite() && dt > 0.0) {
            return None;
        }
        if external_forces.len() != n {
            return None;
        }

        let assembly = assemble_cohesive_forces(
            &self.interfaces,
            &self.rest_positions,
            &self.positions,
            &self.model,
            &mut self.states,
        )?;

        // Integrate velocities (pinned vertices untouched), then positions.
        for (i, &fext) in external_forces.iter().enumerate() {
            if self.masses[i].is_finite() {
                let force = assembly.forces[i] + fext;
                self.velocities[i] += (dt / self.masses[i]) * force;
            }
        }
        for i in 0..n {
            self.positions[i] += self.velocities[i] * dt;
        }

        let max_force_magnitude = assembly
            .steps
            .iter()
            .map(|s| s.force_magnitude)
            .fold(0.0_f32, f32::max);

        Some(CohesiveStepReport {
            advanced_facets: assembly.advanced_facet_count(),
            damaged_facets: assembly.damaged_facet_count(),
            decohered_facets: assembly.decohered_facet_count(),
            mean_damage: assembly.mean_damage(),
            max_force_magnitude,
            dissipated_energy: self.dissipated_energy(),
            kinetic_energy: self.kinetic_energy(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A single square interface in the x–y plane with outward normal +z. Side a
    // (vertices 0,1,2) lies below side b (vertices 3,4,5); they start
    // coincident, so the rest configuration has zero jump.
    fn unit_interface() -> (Vec<CohesiveInterface>, Vec<Vec3>) {
        let rest = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let interfaces = vec![CohesiveInterface::new([0, 1, 2], [3, 4, 5])];
        (interfaces, rest)
    }

    // K = 1e6, σ_c = 1e3 → δ₀ = 1e-3. G_c = 1.0 → δ_f = 2e-3 (softening exists).
    fn model() -> CohesiveModel {
        CohesiveModel::new(1.0e6, 1.0e3, 1.0, 1.0).unwrap()
    }

    fn body() -> CohesiveBody {
        let (interfaces, rest) = unit_interface();
        CohesiveBody::new(model(), interfaces, rest, vec![1.0; 6]).expect("valid body")
    }

    fn zeros(n: usize) -> Vec<Vec3> {
        vec![Vec3::ZERO; n]
    }

    #[test]
    fn new_rejects_mismatched_arrays() {
        let (interfaces, rest) = unit_interface();
        assert!(CohesiveBody::new(model(), interfaces, rest, vec![1.0; 5]).is_none());
    }

    #[test]
    fn new_rejects_bad_mass_and_out_of_range_interface() {
        let (interfaces, rest) = unit_interface();
        // Finite non-positive mass is invalid.
        let mut bad_mass = vec![1.0; 6];
        bad_mass[2] = 0.0;
        assert!(CohesiveBody::new(model(), interfaces.clone(), rest.clone(), bad_mass).is_none());
        // Interface referencing a missing vertex.
        let bad_iface = vec![CohesiveInterface::new([0, 1, 2], [3, 4, 9])];
        assert!(CohesiveBody::new(model(), bad_iface, rest, vec![1.0; 6]).is_none());
    }

    #[test]
    fn new_rejects_degenerate_rest_facet() {
        // All three corners of side a coincide → zero rest area.
        let rest = vec![Vec3::ZERO; 6];
        let interfaces = vec![CohesiveInterface::new([0, 1, 2], [3, 4, 5])];
        assert!(CohesiveBody::new(model(), interfaces, rest, vec![1.0; 6]).is_none());
    }

    #[test]
    fn pinned_vertex_has_infinite_mass() {
        let (interfaces, rest) = unit_interface();
        let mut masses = vec![1.0; 6];
        masses[0] = f32::INFINITY;
        let b = CohesiveBody::new(model(), interfaces, rest, masses).expect("valid");
        assert!(b.is_pinned(0));
        assert!(!b.is_pinned(1));
    }

    #[test]
    fn step_rejects_bad_dt_and_lengths() {
        let mut b = body();
        assert!(b.step(0.0, &zeros(6)).is_none());
        assert!(b.step(-1.0, &zeros(6)).is_none());
        assert!(b.step(f32::NAN, &zeros(6)).is_none());
        assert!(b.step(1.0e-6, &zeros(5)).is_none());
    }

    #[test]
    fn pristine_interface_is_undamaged() {
        let mut b = body();
        // A sub-onset normal pull: elastic penalty only, no history advance.
        let mut fext = zeros(6);
        for v in fext.iter_mut().take(3) {
            v.z = -1.0;
        }
        for v in fext.iter_mut().skip(3) {
            v.z = 1.0;
        }
        let r = b.step(1.0e-6, &fext).expect("ok");
        assert_eq!(r.advanced_facets, 0);
        assert_eq!(r.decohered_facets, 0);
        assert!(r.mean_damage < 1e-6);
    }

    #[test]
    fn rigid_translation_develops_no_cohesive_force() {
        let mut b = body();
        let v = Vec3::new(0.2, -0.1, 0.3);
        for i in 0..6 {
            b.set_velocity(i, v);
        }
        let r = b.step(1.0e-4, &zeros(6)).expect("ok");
        // Both faces move together, so the jump — and the force — stay zero.
        assert!(r.max_force_magnitude < 1e-3);
        for i in 0..6 {
            assert!((b.velocities()[i] - v).length() < 1e-6);
        }
    }

    #[test]
    fn internal_cohesive_forces_conserve_linear_momentum() {
        let mut b = body();
        // Open the interface along the normal with equal/opposite velocities.
        for i in 0..3 {
            b.set_velocity(i, Vec3::new(0.0, 0.0, -0.1));
        }
        for i in 3..6 {
            b.set_velocity(i, Vec3::new(0.0, 0.0, 0.1));
        }
        let p0 = b.linear_momentum();
        for _ in 0..40 {
            let _ = b.step(1.0e-6, &zeros(6)).expect("ok");
        }
        let p1 = b.linear_momentum();
        assert!((p1 - p0).length() < 1e-3, "momentum drift {p0:?} -> {p1:?}");
    }

    #[test]
    fn opening_interface_damages_then_fully_decoheres() {
        let mut b = body();
        // Strong opening pull well past the final separation δ_f = 2e-3.
        let mut fext = zeros(6);
        for v in fext.iter_mut().take(3) {
            v.z = -2.0e3;
        }
        for v in fext.iter_mut().skip(3) {
            v.z = 2.0e3;
        }
        let dt = 1.0e-5;
        let mut decohered = false;
        for _ in 0..400 {
            let r = b.step(dt, &fext).expect("ok");
            if r.decohered_facets == 1 {
                decohered = true;
                break;
            }
        }
        assert!(decohered, "sustained opening must fully decohere the facet");
        assert_eq!(b.decohered_count(), 1);
        // A fully decohered facet dissipated the whole fracture energy
        // G_c · A = 1.0 · 0.5 = 0.5 J.
        assert!((b.dissipated_energy() - 0.5).abs() < 1e-2);
    }

    #[test]
    fn pinned_side_holds_while_free_side_opens() {
        let (interfaces, rest) = unit_interface();
        // Pin side a (vertices 0,1,2); side b is free.
        let masses = vec![f32::INFINITY, f32::INFINITY, f32::INFINITY, 1.0, 1.0, 1.0];
        let mut b = CohesiveBody::new(model(), interfaces, rest, masses).expect("valid");
        // Pull the free side outward along +z, below the force that would snap
        // it immediately, for enough steps to drift measurably.
        let mut fext = zeros(6);
        for v in fext.iter_mut().skip(3) {
            v.z = 2.0e2;
        }
        for _ in 0..200 {
            let _ = b.step(1.0e-5, &fext).expect("ok");
        }
        // Anchored side has not moved at all.
        for i in 0..3 {
            assert_eq!(b.positions()[i], b.rest_positions()[i]);
            assert_eq!(b.velocities()[i], Vec3::ZERO);
        }
        // Free side has lifted off along +z.
        for i in 3..6 {
            assert!(b.positions()[i].z > 0.0, "free side opened");
        }
    }

    #[test]
    fn integrator_matches_builder_output() {
        use crate::collider::cohesive_interface_builder::insert_cohesive_interfaces;
        // Two tets sharing one interior facet; cut it to install one interface.
        let rest = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, -1.0),
        ];
        let tets = vec![[0, 1, 2, 3], [0, 1, 2, 4]];
        let cut = [[0, 1, 2]];
        let mesh = insert_cohesive_interfaces(5, &tets, &cut).expect("valid mesh");
        assert_eq!(mesh.interfaces.len(), 1);
        let positions = mesh.expand_positions(&rest).expect("expanded");
        let masses = vec![1.0; mesh.vertex_count];
        let mut b =
            CohesiveBody::new(model(), mesh.interfaces, positions, masses).expect("valid body");
        let r = b.step(1.0e-6, &zeros(b.vertex_count())).expect("ok");
        assert_eq!(r.advanced_facets, 0);
        assert_eq!(b.interface_count(), 1);
    }
}
