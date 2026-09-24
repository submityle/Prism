//! Solvers: the pluggable stepping strategy extension point.
//!
//! A [`Solver`] advances a [`PhysicsWorld`] by one frame, internally splitting
//! the frame into sub-steps. M0 ships one real reference solver,
//! [`IntegrateOnlySolver`], which performs sub-stepped free-body integration
//! (bodies really fall under gravity). Later milestones add constraint-solving
//! implementations behind the same trait.

use crate::dynamics::Integrator;
use crate::world::PhysicsWorld;
use std::collections::HashMap;

/// A pluggable simulation stepping strategy.
///
/// Implementors advance `world` by `dt` seconds, dividing the work into
/// `substeps` equal sub-steps. `substeps` is always at least `1` by contract;
/// implementors should treat `0` as `1`.
pub trait Solver {
    /// Advances `world` by `dt` seconds using `substeps` sub-steps.
    fn step(&mut self, world: &mut PhysicsWorld, dt: f32, substeps: u32);

    /// Returns a short, stable name identifying the solver.
    fn name(&self) -> &str;
}

/// Reference solver that only integrates free-body motion.
///
/// It divides `dt` into `substeps` equal slices and applies the semi-implicit
/// Euler [`Integrator`] on each, so dynamic bodies accelerate under the world's
/// gravity and damping. It performs no collision or constraint resolution.
#[derive(Clone, Copy, Debug, Default)]
pub struct IntegrateOnlySolver;

impl Solver for IntegrateOnlySolver {
    fn step(&mut self, world: &mut PhysicsWorld, dt: f32, substeps: u32) {
        let n = substeps.max(1);
        let sub_dt = dt / n as f32;
        let gravity = world.config.gravity;
        for _ in 0..n {
            Integrator::integrate(&mut world.bodies, gravity, sub_dt);
        }
    }

    fn name(&self) -> &str {
        "integrate-only"
    }
}

/// A registry of named solvers.
///
/// Solvers are stored as boxed trait objects keyed by name, letting callers
/// register several strategies and select one at run time.
#[derive(Default)]
pub struct SolverRegistry {
    solvers: HashMap<String, Box<dyn Solver>>,
}

impl SolverRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> SolverRegistry {
        SolverRegistry {
            solvers: HashMap::new(),
        }
    }

    /// Registers `solver` under `name`, replacing any existing entry and
    /// returning the previous solver if one was present.
    pub fn register(&mut self, name: &str, solver: Box<dyn Solver>) -> Option<Box<dyn Solver>> {
        self.solvers.insert(name.to_owned(), solver)
    }

    /// Returns a shared reference to the solver registered under `name`.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&dyn Solver> {
        self.solvers.get(name).map(AsRef::as_ref)
    }

    /// Returns a mutable reference to the solver registered under `name`.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut (dyn Solver + 'static)> {
        self.solvers.get_mut(name).map(AsMut::as_mut)
    }

    /// Returns the number of registered solvers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.solvers.len()
    }

    /// Returns `true` if no solvers are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.solvers.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::body::BodyDesc;
    use glam::Vec3;

    #[test]
    fn integrate_only_makes_bodies_fall() {
        let mut world = PhysicsWorld::with_gravity(Vec3::new(0.0, -10.0, 0.0));
        let h = world.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
        let mut solver = IntegrateOnlySolver;
        solver.step(&mut world, 1.0, 4);
        // Body must have fallen (negative y).
        assert!(world.bodies.position(h).unwrap().y < -0.5);
        assert_eq!(solver.name(), "integrate-only");
    }

    #[test]
    fn registry_register_and_get() {
        let mut reg = SolverRegistry::new();
        assert!(reg.is_empty());
        assert!(reg
            .register("integrate-only", Box::new(IntegrateOnlySolver))
            .is_none());
        assert_eq!(reg.len(), 1);
        assert_eq!(
            reg.get("integrate-only").map(Solver::name),
            Some("integrate-only")
        );
        assert!(reg.get("missing").is_none());
        assert!(reg.get_mut("integrate-only").is_some());
    }
}
