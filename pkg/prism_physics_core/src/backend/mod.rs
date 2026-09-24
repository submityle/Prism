//! Backends: the pluggable execution extension point.
//!
//! A [`PhysicsBackend`] owns whatever resources are needed to step a
//! [`PhysicsWorld`] and exposes a single per-frame [`PhysicsBackend::step`]
//! entry point. M0 ships one real reference backend, [`CpuBackend`], which runs
//! a boxed [`Solver`] for a fixed number of sub-steps on the CPU.

use crate::solver::{IntegrateOnlySolver, Solver};
use crate::world::PhysicsWorld;

/// A pluggable engine execution backend.
///
/// This abstracts *where and how* stepping runs (CPU today; other targets in
/// later milestones) from *what* stepping does (the [`Solver`]).
pub trait PhysicsBackend {
    /// Advances `world` by `dt` seconds.
    fn step(&mut self, world: &mut PhysicsWorld, dt: f32);

    /// Returns a short, stable name identifying the backend.
    fn name(&self) -> &str;
}

/// Reference CPU backend that drives a boxed [`Solver`].
pub struct CpuBackend {
    solver: Box<dyn Solver>,
    substeps: u32,
}

impl CpuBackend {
    /// The default sub-step count used by [`CpuBackend::with_defaults`].
    pub const DEFAULT_SUBSTEPS: u32 = 4;

    /// Creates a backend that runs `solver` with `substeps` sub-steps per
    /// frame. `substeps` is clamped to at least `1`.
    #[must_use]
    pub fn new(solver: Box<dyn Solver>, substeps: u32) -> CpuBackend {
        CpuBackend {
            solver,
            substeps: substeps.max(1),
        }
    }

    /// Creates a backend using the [`IntegrateOnlySolver`] and
    /// [`CpuBackend::DEFAULT_SUBSTEPS`].
    #[must_use]
    pub fn with_defaults() -> CpuBackend {
        CpuBackend::new(Box::new(IntegrateOnlySolver), Self::DEFAULT_SUBSTEPS)
    }

    /// Returns the configured number of sub-steps.
    #[must_use]
    pub fn substeps(&self) -> u32 {
        self.substeps
    }
}

impl PhysicsBackend for CpuBackend {
    fn step(&mut self, world: &mut PhysicsWorld, dt: f32) {
        self.solver.step(world, dt, self.substeps);
    }

    fn name(&self) -> &str {
        "cpu"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::body::BodyDesc;
    use glam::Vec3;

    #[test]
    fn cpu_backend_steps_world() {
        let mut world = PhysicsWorld::with_gravity(Vec3::new(0.0, -9.81, 0.0));
        let h = world.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
        let mut backend = CpuBackend::with_defaults();
        assert_eq!(backend.name(), "cpu");
        assert_eq!(backend.substeps(), CpuBackend::DEFAULT_SUBSTEPS);
        backend.step(&mut world, 0.1);
        assert!(world.bodies.position(h).unwrap().y < 0.0);
    }

    #[test]
    fn substeps_clamped_to_one() {
        let backend = CpuBackend::new(Box::new(IntegrateOnlySolver), 0);
        assert_eq!(backend.substeps(), 1);
    }
}
