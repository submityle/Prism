//! Extended Position Based Dynamics (XPBD) rigid-body solver.
//!
//! This is Prism's first constraint-solving [`Solver`]. It advances the world
//! with sub-stepped position-based dynamics: each sub-step predicts new poses
//! from the current velocities, detects contacts against those predicted poses,
//! resolves penetration and static friction with a compliant position solve,
//! recovers velocities from the net pose change, and finally applies
//! restitution and dynamic friction at the velocity level.
//!
//! The sub-modules split the pipeline into focused pieces:
//!
//! - [`rigid`] — shared rigid-body impulse algebra.
//! - [`integrate`] — pose prediction and velocity recovery.
//! - [`island_solve`] — grouping of contacts/joints into independent islands.
//! - [`contact_constraint`] — the position-level contact + static-friction solve.
//! - [`velocity_solve`] — the velocity-level restitution + dynamic-friction solve.
//! - [`config`] — the [`XpbdConfig`] tuning parameters.
//!
//! # Small steps
//!
//! Following Müller et al. (2020), the solver relies on many small sub-steps
//! rather than many iterations per step; with sub-stepping a single position
//! iteration per sub-step already gives stable stacks. Contacts are detected
//! once per sub-step at the predicted poses; incremental contact reuse is a
//! later-milestone optimisation.
//!
//! # Provenance
//!
//! The whole predict / solve-positions / recover / solve-velocities structure
//! follows Müller et al., *Detailed Rigid Body Simulation with Extended
//! Position Based Dynamics* (2020). This file contains no Unreal Engine source
//! or derived code.

pub mod config;
pub mod contact_constraint;
pub mod integrate;
pub mod island_solve;
pub mod joint_constraint;
pub mod rigid;
pub mod velocity_solve;

pub use config::XpbdConfig;
pub use contact_constraint::ContactConstraint;
pub use island_solve::SolveIslands;

use crate::joint::Joint;
use crate::pipeline::detect_contacts;
use crate::solver::Solver;
use crate::world::PhysicsWorld;

/// A sub-stepped Extended Position Based Dynamics rigid-body solver.
#[derive(Clone, Copy, Debug, Default)]
pub struct XpbdSolver {
    /// Tuning parameters for the position and velocity passes.
    pub config: XpbdConfig,
}

impl XpbdSolver {
    /// Creates a solver with the default [`XpbdConfig`].
    #[must_use]
    pub fn new() -> XpbdSolver {
        XpbdSolver::default()
    }

    /// Creates a solver with the given configuration.
    #[must_use]
    pub fn with_config(config: XpbdConfig) -> XpbdSolver {
        XpbdSolver { config }
    }

    /// Advances the world by a single sub-step of duration `h`.
    fn substep(&self, world: &mut PhysicsWorld, h: f32) {
        let gravity = world.config.gravity;

        // Phase 1: predict poses forward under gravity and damping.
        {
            let mut view = world.bodies.solver_view_mut();
            integrate::predict(&mut view, gravity, h);
        }

        // Phase 2: detect contacts at the predicted poses. This needs an
        // immutable borrow of the whole world, so it must run outside the
        // mutable solver view scopes.
        let manifolds = detect_contacts(world);

        // Phase 3: resolve positions, recover velocities, resolve velocities.
        {
            // Split-borrow the world so the joint storage stays readable while
            // the solver view mutably borrows only the body columns.
            let PhysicsWorld { bodies, joints, .. } = world;
            let mut view = bodies.solver_view_mut();
            let mut constraints = ContactConstraint::build(&view, &manifolds);
            // Collect the active joints once so they can be indexed per island.
            let active: Vec<&Joint> = joints.active_joints().collect();
            // Partition the constraints/joints into independent islands. Solving
            // islands one after another is numerically identical to a single
            // global Gauss-Seidel pass because islands touch disjoint dynamic
            // bodies (statics are read-only separators).
            let islands = SolveIslands::build(&view, &constraints, &active);
            let iterations = self.config.position_iterations.max(1);
            for _ in 0..iterations {
                for island in 0..islands.island_count() {
                    for &joint_index in islands.joints(island) {
                        joint_constraint::solve_joint(&mut view, active[joint_index], h);
                    }
                    contact_constraint::solve_positions_indexed(
                        &mut view,
                        &mut constraints,
                        islands.contacts(island),
                        &self.config,
                        h,
                    );
                }
            }
            // Velocity recovery from the net pose change is a per-body pass and
            // stays global; the velocity-level solve is then applied per island.
            integrate::recover_velocities(&mut view, h);
            for island in 0..islands.island_count() {
                velocity_solve::solve_indexed(
                    &mut view,
                    &constraints,
                    islands.contacts(island),
                    &self.config,
                    h,
                );
            }
        }
    }
}

impl Solver for XpbdSolver {
    fn step(&mut self, world: &mut PhysicsWorld, dt: f32, substeps: u32) {
        let n = substeps.max(1);
        let h = dt / n as f32;
        if h <= 0.0 {
            return;
        }
        for _ in 0..n {
            self.substep(world, h);
        }
    }

    fn name(&self) -> &str {
        "xpbd"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::{ColliderShape, PhysicsMaterial};
    use crate::state::body::BodyDesc;
    use glam::Vec3;

    /// Spawns a static ground plane and returns the world.
    fn world_with_ground() -> PhysicsWorld {
        let mut world = PhysicsWorld::with_gravity(Vec3::new(0.0, -9.81, 0.0));
        let plane = world.shapes.insert(ColliderShape::Plane {
            normal: Vec3::Y,
            offset: 0.0,
        });
        world.spawn(
            BodyDesc::static_at(Vec3::ZERO)
                .with_collider(plane)
                .with_material(PhysicsMaterial::DEFAULT),
        );
        world
    }

    /// Spawns a unit cube (half-extent 0.5) at `y` and returns its handle.
    fn spawn_box(world: &mut PhysicsWorld, y: f32) -> crate::state::handle::BodyHandle {
        let shape = ColliderShape::Cuboid {
            half_extents: Vec3::splat(0.5),
        };
        let cuboid = world.shapes.insert(shape);
        let mp = shape.mass_properties(1.0);
        world.spawn(
            BodyDesc::dynamic_at(Vec3::new(0.0, y, 0.0))
                .with_collider(cuboid)
                .with_mass_properties(mp)
                .with_material(PhysicsMaterial::DEFAULT),
        )
    }

    /// Spawns a unit cube at horizontal offset `x` and height `y`.
    fn spawn_box_at(world: &mut PhysicsWorld, x: f32, y: f32) -> crate::state::handle::BodyHandle {
        let shape = ColliderShape::Cuboid {
            half_extents: Vec3::splat(0.5),
        };
        let cuboid = world.shapes.insert(shape);
        let mp = shape.mass_properties(1.0);
        world.spawn(
            BodyDesc::dynamic_at(Vec3::new(x, y, 0.0))
                .with_collider(cuboid)
                .with_mass_properties(mp)
                .with_material(PhysicsMaterial::DEFAULT),
        )
    }

    #[test]
    fn single_box_settles_on_ground() {
        let mut world = world_with_ground();
        // Drop the box from just above its resting height (0.5).
        let handle = spawn_box(&mut world, 1.2);
        let mut solver = XpbdSolver::new();
        for _ in 0..120 {
            solver.step(&mut world, 1.0 / 60.0, 8);
        }
        let y = world.bodies.position(handle).unwrap().y;
        // The box rests with its center half an extent above the plane.
        assert!(
            (y - 0.5).abs() < 0.02,
            "box settled at y = {y}, expected ~0.5"
        );
        let v = world.bodies.linear_velocity(handle).unwrap();
        assert!(v.length() < 0.05, "box should be at rest, v = {v:?}");
    }

    #[test]
    fn penetrating_box_is_pushed_out() {
        let mut world = world_with_ground();
        // Start already sunk into the ground.
        let handle = spawn_box(&mut world, 0.1);
        let mut solver = XpbdSolver::new();
        for _ in 0..60 {
            solver.step(&mut world, 1.0 / 60.0, 8);
        }
        let y = world.bodies.position(handle).unwrap().y;
        assert!(y > 0.45, "box should be lifted out of the ground, y = {y}");
    }

    #[test]
    fn box_stack_stays_stable() {
        let mut world = world_with_ground();
        // Three stacked unit cubes at rest heights 0.5, 1.5, 2.5 with a small
        // seating gap so the solver must settle them onto each other.
        let handles = [
            spawn_box(&mut world, 0.5),
            spawn_box(&mut world, 1.52),
            spawn_box(&mut world, 2.54),
        ];
        let mut solver = XpbdSolver::new();
        for _ in 0..240 {
            solver.step(&mut world, 1.0 / 60.0, 12);
        }
        let expected = [0.5, 1.5, 2.5];
        for (h, want) in handles.iter().zip(expected) {
            let pos = world.bodies.position(*h).unwrap();
            let vel = world.bodies.linear_velocity(*h).unwrap();
            assert!(
                (pos.y - want).abs() < 0.08,
                "box settled at y = {}, expected ~{want}",
                pos.y
            );
            // No horizontal drift and no residual jitter.
            assert!(
                pos.x.abs() < 0.05 && pos.z.abs() < 0.05,
                "lateral drift: {pos:?}"
            );
            assert!(vel.length() < 0.1, "stack should be at rest, v = {vel:?}");
        }
    }

    /// Two identical two-box stacks placed far apart horizontally form two
    /// separate islands (they share only the static ground). Solving both
    /// together must match solving each stack alone, proving that per-island
    /// solving is independent and equivalent to the global pass.
    #[test]
    fn separated_stacks_solve_independently() {
        fn run(offsets: &[f32]) -> Vec<Vec3> {
            let mut world = world_with_ground();
            let mut handles = Vec::new();
            for &x in offsets {
                handles.push(spawn_box_at(&mut world, x, 0.5));
                handles.push(spawn_box_at(&mut world, x, 1.52));
            }
            let mut solver = XpbdSolver::new();
            for _ in 0..120 {
                solver.step(&mut world, 1.0 / 60.0, 8);
            }
            handles
                .iter()
                .map(|h| world.bodies.position(*h).unwrap())
                .collect()
        }

        let a_alone = run(&[-50.0]);
        let b_alone = run(&[50.0]);
        let both = run(&[-50.0, 50.0]);

        assert_eq!(both.len(), 4);
        for (i, want) in a_alone.iter().enumerate() {
            assert!(
                (both[i] - *want).length() < 1e-5,
                "stack A body {i} diverged: {:?} vs {:?}",
                both[i],
                want
            );
        }
        for (i, want) in b_alone.iter().enumerate() {
            assert!(
                (both[2 + i] - *want).length() < 1e-5,
                "stack B body {i} diverged: {:?} vs {:?}",
                both[2 + i],
                want
            );
        }
    }
}
