//! Particle state consumed and produced by the `XPBD` solvers.
//!
//! [`ParticleState`] is a Structure-of-Arrays bundle of the per-particle
//! quantities the solver reads and writes: current positions, velocities, and
//! inverse masses. A particle with inverse mass `0` is *pinned* (an immovable
//! anchor). Both the `CPU` golden and the `GPU` kernel mutate positions and
//! velocities in place; the caller owns the arrays across steps so state
//! carries forward frame to frame.
//!
//! The `GPU` path uploads positions and velocities as `vec4<f32>` (`xyz` plus a
//! zero pad) for natural `std430` alignment; [`ParticleState`] keeps the tidy
//! [`Vec3`] form and converts at the boundary.
//!
//! Provenance: standard position-based-dynamics particle state. No Unreal
//! Engine source or derived code.

use glam::Vec3;

/// A Structure-of-Arrays bundle of per-particle solver state.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ParticleState {
    /// World-space positions.
    pub positions: Vec<Vec3>,
    /// Linear velocities.
    pub velocities: Vec<Vec3>,
    /// Inverse masses; `0` marks a pinned (immovable) particle.
    pub inverse_masses: Vec<f32>,
}

impl ParticleState {
    /// Creates an empty state.
    #[must_use]
    pub fn new() -> ParticleState {
        ParticleState {
            positions: Vec::new(),
            velocities: Vec::new(),
            inverse_masses: Vec::new(),
        }
    }

    /// Appends a particle with the given position and inverse mass, starting at
    /// rest, and returns its index.
    pub fn push(&mut self, position: Vec3, inverse_mass: f32) -> u32 {
        let index = self.positions.len() as u32;
        self.positions.push(position);
        self.velocities.push(Vec3::ZERO);
        self.inverse_masses.push(inverse_mass.max(0.0));
        index
    }

    /// Number of particles in the state.
    #[must_use]
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Whether the state has no particles.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Returns `true` when the three arrays have equal length.
    ///
    /// A mismatched state is a caller construction error; the solvers check
    /// this before touching the arrays.
    #[must_use]
    pub fn is_consistent(&self) -> bool {
        self.positions.len() == self.velocities.len()
            && self.positions.len() == self.inverse_masses.len()
    }
}
