//! Decoupled implicit integration of reduced (modal) coordinates.
//!
//! With a mass-orthonormal modal basis the linearised equations of motion
//! diagonalise: each generalised coordinate `q_k` obeys an independent
//! second-order ODE
//!
//! ```text
//! q_k'' + d_k q_k' + lambda_k q_k = phi_k,
//! ```
//!
//! where `lambda_k` is the mode's eigenvalue (squared angular frequency),
//! `phi_k` is the modal force (the external per-vertex force projected onto the
//! mode shape), and `d_k = alpha + beta * lambda_k` is the modal Rayleigh
//! damping. Because the modes are decoupled, each scalar ODE is advanced with an
//! unconditionally stable backward-Euler step in closed form, so the whole body
//! integrates in `O(num_modes)` time independent of the mesh resolution.
//!
//! Writing `v_k = q_k'`, backward Euler over a substep `h` gives
//!
//! ```text
//! v_k^{n+1} = (v_k^n + h phi_k - h lambda_k q_k^n)
//!             / (1 + h d_k + h^2 lambda_k),
//! q_k^{n+1} = q_k^n + h v_k^{n+1}.
//! ```
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. Modal
//! superposition, Rayleigh damping, and the closed-form backward-Euler update of
//! a decoupled linear oscillator are standard, publicly documented
//! structural-dynamics results.

use glam::Vec3;

use crate::math::scalar::Real;

use super::config::ReducedConfig;
use super::subspace::ReducedModel;

/// The evolving reduced state of one soft body: a generalised position `q` and
/// generalised velocity `qd`, one scalar per retained mode.
///
/// The state is decoupled from the [`ReducedModel`] (which is immutable after
/// construction) so several bodies can share a model while keeping their own
/// motion, and so a state can be reset without rebuilding the modal basis.
#[derive(Clone, PartialEq, Debug, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ReducedState {
    /// Generalised coordinates `q_k`, one per mode.
    q: Vec<Real>,
    /// Generalised velocities `qd_k = dq_k/dt`, one per mode.
    qd: Vec<Real>,
}

impl ReducedState {
    /// Creates a rest state (`q = qd = 0`) with `num_modes` coordinates.
    #[must_use]
    pub fn new(num_modes: usize) -> ReducedState {
        ReducedState {
            q: vec![0.0; num_modes],
            qd: vec![0.0; num_modes],
        }
    }

    /// Creates a rest state sized to a model's retained-mode count.
    #[must_use]
    pub fn for_model(model: &ReducedModel) -> ReducedState {
        ReducedState::new(model.num_modes())
    }

    /// Returns the generalised coordinates `q`.
    #[must_use]
    pub fn coordinates(&self) -> &[Real] {
        &self.q
    }

    /// Returns the generalised velocities `qd`.
    #[must_use]
    pub fn velocities(&self) -> &[Real] {
        &self.qd
    }

    /// Number of modal coordinates tracked.
    #[must_use]
    pub fn len(&self) -> usize {
        self.q.len()
    }

    /// Returns `true` when no coordinates are tracked.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.q.is_empty()
    }

    /// Resets both `q` and `qd` to zero, returning the body to its rest shape.
    pub fn reset(&mut self) {
        for value in &mut self.q {
            *value = 0.0;
        }
        for value in &mut self.qd {
            *value = 0.0;
        }
    }

    /// Reconstructs the world-space vertex positions for the current state.
    #[must_use]
    pub fn positions(&self, model: &ReducedModel) -> Vec<Vec3> {
        model.reconstruct(&self.q)
    }

    /// Advances the reduced state by `dt` seconds under gravity plus an optional
    /// per-vertex external force field.
    ///
    /// The frame step is split into [`ReducedConfig::substeps`] equal substeps
    /// (clamped to at least one). Within each substep every vertex feels the
    /// body force `m_i * gravity + external[i]`, which is projected onto the
    /// modal basis and integrated with the closed-form backward-Euler update.
    /// An empty `external` slice is treated as zero force; a shorter slice pads
    /// missing entries with zero.
    pub fn step(
        &mut self,
        model: &ReducedModel,
        config: &ReducedConfig,
        dt: Real,
        external: &[Vec3],
    ) {
        let modes = model.modes();
        let count = self.q.len().min(modes.len());
        if count == 0 || dt <= 0.0 {
            return;
        }
        let substeps = config.substeps.max(1);
        let h = dt / substeps as Real;

        // Body force per vertex is constant across substeps, so its modal
        // projection is computed once and reused.
        let force = body_force(model, config.gravity, external);
        let phi = model.modal_force(&force);

        for _ in 0..substeps {
            for k in 0..count {
                let lambda = modes[k].frequency_squared;
                let damping = config.rayleigh_alpha + config.rayleigh_beta * lambda;
                let denom = 1.0 + h * damping + h * h * lambda;
                let numerator = self.qd[k] + h * phi[k] - h * lambda * self.q[k];
                let qd_next = numerator / denom;
                self.qd[k] = qd_next;
                self.q[k] += h * qd_next;
            }
        }
    }
}

/// Builds the per-vertex body force `m_i * gravity + external[i]`.
fn body_force(model: &ReducedModel, gravity: Vec3, external: &[Vec3]) -> Vec<Vec3> {
    let masses = model.masses();
    let mut force = Vec::with_capacity(masses.len());
    for (i, &mass) in masses.iter().enumerate() {
        let mut f = gravity * mass;
        if let Some(&e) = external.get(i) {
            f += e;
        }
        force.push(f);
    }
    force
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::soft::particle::ParticleHandle;
    use crate::vbd::element::{SpringElement, SpringSet};

    fn handle(i: u32) -> ParticleHandle {
        ParticleHandle::from_index(i)
    }

    /// Vertex 0 anchored, vertex 1 free, joined by an axial spring. Under
    /// gravity the free vertex must settle to a static equilibrium where the
    /// modal restoring force balances the projected gravity, and must stay
    /// there once at rest.
    fn cantilever() -> ReducedModel {
        let rest = vec![Vec3::ZERO, Vec3::new(0.0, -1.0, 0.0)];
        let masses = vec![1.0, 1.0];
        let fixed = vec![true, false];
        let mut springs = SpringSet::new();
        springs.push(SpringElement::new(handle(0), handle(1), 1.0, 100.0));
        ReducedModel::from_springs(&rest, &masses, &fixed, &springs, 3)
    }

    #[test]
    fn rest_state_is_zero() {
        let model = cantilever();
        let state = ReducedState::for_model(&model);
        assert_eq!(state.len(), model.num_modes());
        assert!(state.coordinates().iter().all(|&q| q == 0.0));
        assert!(state.velocities().iter().all(|&q| q == 0.0));
        // Rest state reconstructs the rest shape exactly.
        assert_eq!(state.positions(&model), model.rest());
    }

    #[test]
    fn gravity_pulls_the_free_vertex_down() {
        let model = cantilever();
        let mut state = ReducedState::for_model(&model);
        let config = ReducedConfig {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            num_modes: 3,
            rayleigh_alpha: 2.0,
            rayleigh_beta: 0.05,
            substeps: 4,
        };
        for _ in 0..2000 {
            state.step(&model, &config, 1.0 / 60.0, &[]);
        }
        let positions = state.positions(&model);
        // Anchor never moves.
        assert_eq!(positions[0], Vec3::ZERO);
        // Free vertex sags below its rest height along the pull direction.
        assert!(
            positions[1].y < model.rest()[1].y - 1e-3,
            "free vertex y = {}",
            positions[1].y
        );
    }

    #[test]
    fn damped_system_reaches_static_equilibrium() {
        // After long damped integration the velocities vanish and successive
        // steps no longer move the body.
        let model = cantilever();
        let mut state = ReducedState::for_model(&model);
        let config = ReducedConfig {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            num_modes: 3,
            rayleigh_alpha: 4.0,
            rayleigh_beta: 0.1,
            substeps: 4,
        };
        for _ in 0..4000 {
            state.step(&model, &config, 1.0 / 60.0, &[]);
        }
        let before = state.positions(&model);
        state.step(&model, &config, 1.0 / 60.0, &[]);
        let after = state.positions(&model);
        for (b, a) in before.iter().zip(after.iter()) {
            assert!((*a - *b).length() < 1e-4, "not settled: {b} -> {a}");
        }
        assert!(state.velocities().iter().all(|v| v.abs() < 1e-3));
    }

    #[test]
    fn zero_dt_is_a_no_op() {
        let model = cantilever();
        let mut state = ReducedState::for_model(&model);
        let config = ReducedConfig::default();
        state.step(&model, &config, 0.0, &[]);
        assert!(state.coordinates().iter().all(|&q| q == 0.0));
    }

    #[test]
    fn external_force_projects_onto_modes() {
        // A one-shot symmetric integration under a pure external force (no
        // gravity) must move the free vertex; an empty model stays put.
        let model = cantilever();
        let mut state = ReducedState::for_model(&model);
        let config = ReducedConfig {
            gravity: Vec3::ZERO,
            num_modes: 3,
            rayleigh_alpha: 0.0,
            rayleigh_beta: 0.0,
            substeps: 1,
        };
        let external = vec![Vec3::ZERO, Vec3::new(0.0, -5.0, 0.0)];
        state.step(&model, &config, 1.0 / 60.0, &external);
        assert!(state.coordinates().iter().any(|&q| q.abs() > 0.0));
    }
}
