//! The fixed-timestep, interpolated simulation pipeline.
//!
//! This is the heart of M2.5. It decouples the physics tick rate from the
//! render frame rate so the simulation is deterministic (it only ever advances
//! in whole [`FixedStepPipeline::fixed_dt`] steps) while the render output stays
//! smooth (each frame interpolates between the two most recently published
//! [`StateSnapshot`]s). The classic problem it solves is temporal aliasing: if
//! the renderer drew the raw solver state, a physics rate that does not evenly
//! divide the frame rate would make bodies visibly stutter. Interpolating
//! between the bracketing physics states removes that jitter entirely.
//!
//! The loop follows the standard fixed-step accumulator:
//!
//! 1. Add the real frame time to an accumulator.
//! 2. While a whole [`FixedStepPipeline::fixed_dt`] has accumulated, drain the
//!    [`CommandQueue`] (so game-logic input is applied at a deterministic step
//!    boundary), step the backend by exactly `fixed_dt`, publish a snapshot, and
//!    subtract `fixed_dt` from the accumulator.
//! 3. The leftover accumulator, divided by `fixed_dt`, is the interpolation
//!    factor [`FixedStepPipeline::alpha`] the renderer blends the last two
//!    snapshots with.
//!
//! A spiral-of-death guard caps how many sub-steps a single
//! [`FixedStepPipeline::advance`] may run so a slow frame cannot make the
//! simulation fall permanently behind.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! fixed-timestep accumulator with state interpolation is a standard, publicly
//! documented real-time-simulation technique implemented from scratch.

use crate::backend::PhysicsBackend;
use crate::command::queue::CommandQueue;
use crate::command::PhysicsCommand;
use crate::snapshot::buffer::TripleBuffer;
use crate::snapshot::hash::{hash_state, StateHash};
use crate::snapshot::pose::BodyPose;
use crate::snapshot::StateSnapshot;
use crate::state::handle::BodyHandle;
use crate::world::PhysicsWorld;

/// The smallest fixed timestep the pipeline will accept, in seconds.
///
/// A non-positive or vanishingly small step is clamped to this so the
/// accumulator loop can never divide by zero or spin forever.
const MIN_FIXED_DT: f32 = 1.0e-6;

/// The default spiral-of-death cap on sub-steps per [`FixedStepPipeline::advance`].
const DEFAULT_MAX_SUBSTEPS: u32 = 8;

/// What a single [`FixedStepPipeline::advance`] did.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct AdvanceReport {
    /// Number of fixed sub-steps executed this call.
    pub steps: u32,
    /// Number of queued commands that took effect this call.
    pub commands_applied: usize,
    /// `true` if the accumulator hit the spiral-of-death cap and time was
    /// dropped to keep the simulation responsive.
    pub clamped: bool,
}

/// A fixed-timestep driver that publishes interpolatable snapshots.
///
/// Construct it with [`FixedStepPipeline::new`], seed the initial pose with
/// [`FixedStepPipeline::prime`], push game-logic input through
/// [`FixedStepPipeline::queue_command`], call [`FixedStepPipeline::advance`]
/// once per render frame with the real elapsed time, and read a smooth pose per
/// body with [`FixedStepPipeline::interpolated_pose`].
#[derive(Debug)]
pub struct FixedStepPipeline {
    /// Fixed physics timestep in seconds (always `>= MIN_FIXED_DT`).
    fixed_dt: f32,
    /// Unspent real time carried between calls.
    accumulator: f32,
    /// Maximum sub-steps a single advance may run.
    max_substeps: u32,
    /// Triple-buffered published snapshots for the renderer to interpolate.
    buffer: TripleBuffer<StateSnapshot>,
    /// Deferred game-logic commands, drained at each step boundary.
    commands: CommandQueue,
    /// Hash of the most recently published snapshot.
    last_hash: StateHash,
}

impl FixedStepPipeline {
    /// Creates a pipeline stepping at `fixed_dt` seconds (clamped to a small
    /// positive minimum) with the default sub-step cap.
    #[must_use]
    pub fn new(fixed_dt: f32) -> FixedStepPipeline {
        FixedStepPipeline {
            fixed_dt: fixed_dt.max(MIN_FIXED_DT),
            accumulator: 0.0,
            max_substeps: DEFAULT_MAX_SUBSTEPS,
            buffer: TripleBuffer::new(StateSnapshot::new()),
            commands: CommandQueue::new(),
            last_hash: StateHash(0),
        }
    }

    /// Creates a pipeline stepping at a rate of `hz` steps per second.
    #[must_use]
    pub fn from_hz(hz: f32) -> FixedStepPipeline {
        let dt = if hz > 0.0 { 1.0 / hz } else { MIN_FIXED_DT };
        FixedStepPipeline::new(dt)
    }

    /// Sets the spiral-of-death cap on sub-steps per advance (clamped to `>= 1`).
    #[must_use]
    pub fn with_max_substeps(mut self, max_substeps: u32) -> FixedStepPipeline {
        self.max_substeps = max_substeps.max(1);
        self
    }

    /// Returns the fixed timestep in seconds.
    #[must_use]
    pub fn fixed_dt(&self) -> f32 {
        self.fixed_dt
    }

    /// Returns the current unspent accumulator time in seconds.
    #[must_use]
    pub fn accumulator(&self) -> f32 {
        self.accumulator
    }

    /// Returns the sub-step cap.
    #[must_use]
    pub fn max_substeps(&self) -> u32 {
        self.max_substeps
    }

    /// Returns the interpolation factor in `[0, 1]` for the current accumulator.
    ///
    /// This is the fraction of the way from the previous published snapshot to
    /// the current one that the renderer should display.
    #[must_use]
    pub fn alpha(&self) -> f32 {
        (self.accumulator / self.fixed_dt).clamp(0.0, 1.0)
    }

    /// Enqueues a game-logic command to apply at the next step boundary.
    pub fn queue_command(&mut self, command: PhysicsCommand) {
        self.commands.push(command);
    }

    /// Returns the number of commands currently queued.
    #[must_use]
    pub fn queued_commands(&self) -> usize {
        self.commands.len()
    }

    /// Publishes the current world state as the initial snapshot so the first
    /// interpolation has a baseline to blend from.
    ///
    /// Call this once after spawning the initial bodies and before the first
    /// [`FixedStepPipeline::advance`]. It performs no simulation.
    pub fn prime(&mut self, world: &PhysicsWorld) {
        self.buffer.write_slot().recapture(world);
        self.buffer.publish();
        if let Some(current) = self.buffer.current() {
            self.last_hash = hash_state(current);
        }
    }

    /// Advances real time by `real_dt` seconds, running as many fixed sub-steps
    /// as have accumulated and publishing a snapshot after each.
    ///
    /// Negative `real_dt` is treated as zero. Returns an [`AdvanceReport`]
    /// describing the work performed.
    pub fn advance(
        &mut self,
        backend: &mut dyn PhysicsBackend,
        world: &mut PhysicsWorld,
        real_dt: f32,
    ) -> AdvanceReport {
        self.accumulator += real_dt.max(0.0);

        // Spiral-of-death guard: never try to run more than max_substeps worth
        // of time in one advance; drop the excess so we stay responsive.
        let max_accum = self.fixed_dt * self.max_substeps as f32;
        let clamped = self.accumulator > max_accum;
        if clamped {
            self.accumulator = max_accum;
        }

        let mut steps = 0;
        let mut commands_applied = 0;
        while self.accumulator >= self.fixed_dt && steps < self.max_substeps {
            // Apply deferred game-logic input at the deterministic boundary
            // *before* the solver runs this sub-step.
            commands_applied += self.commands.drain_apply(world);
            backend.step(world, self.fixed_dt);
            self.buffer.write_slot().recapture(world);
            self.buffer.publish();
            self.accumulator -= self.fixed_dt;
            steps += 1;
        }

        if steps > 0
            && let Some(current) = self.buffer.current()
        {
            self.last_hash = hash_state(current);
        }

        AdvanceReport {
            steps,
            commands_applied,
            clamped,
        }
    }

    /// Returns the most recently published snapshot, or `None` before the first
    /// [`FixedStepPipeline::prime`]/[`FixedStepPipeline::advance`].
    #[must_use]
    pub fn current_snapshot(&self) -> Option<&StateSnapshot> {
        self.buffer.current()
    }

    /// Returns the snapshot published just before the current one, or `None`
    /// until two snapshots have been published.
    #[must_use]
    pub fn previous_snapshot(&self) -> Option<&StateSnapshot> {
        self.buffer.previous()
    }

    /// Returns the interpolated render pose for `handle` at the current
    /// [`FixedStepPipeline::alpha`].
    ///
    /// When a full pair of snapshots is available the pose is blended between
    /// them; with only one snapshot it returns that snapshot's pose directly;
    /// with none it returns `None`.
    #[must_use]
    pub fn interpolated_pose(&self, handle: BodyHandle) -> Option<BodyPose> {
        match self.buffer.pair() {
            Some((prev, curr)) => prev.interpolate(curr, handle, self.alpha()),
            None => self.buffer.current().and_then(|c| c.pose(handle)),
        }
    }

    /// Returns the hash of the most recently published snapshot for desync
    /// checking.
    #[must_use]
    pub fn state_hash(&self) -> StateHash {
        self.last_hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;
    use crate::solver::XpbdSolver;
    use crate::state::body::BodyDesc;
    use glam::Vec3;

    fn dynamics_backend() -> CpuBackend {
        CpuBackend::new(Box::new(XpbdSolver::new()), 4)
    }

    #[test]
    fn accumulator_runs_whole_steps_and_keeps_remainder() {
        let mut world = PhysicsWorld::with_gravity(Vec3::new(0.0, -9.81, 0.0));
        world.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
        let mut backend = dynamics_backend();
        let mut pipe = FixedStepPipeline::new(0.01);
        pipe.prime(&world);
        // 0.025s of real time at a 0.01s step => 2 steps, 0.005 left over.
        let report = pipe.advance(&mut backend, &mut world, 0.025);
        assert_eq!(report.steps, 2);
        assert!((pipe.accumulator() - 0.005).abs() < 1.0e-6);
        assert!((pipe.alpha() - 0.5).abs() < 1.0e-4);
    }

    #[test]
    fn spiral_of_death_is_capped() {
        let mut world = PhysicsWorld::default();
        world.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
        let mut backend = dynamics_backend();
        let mut pipe = FixedStepPipeline::new(0.01).with_max_substeps(4);
        pipe.prime(&world);
        // A huge frame would need 100 steps; the cap limits it to 4.
        let report = pipe.advance(&mut backend, &mut world, 1.0);
        assert_eq!(report.steps, 4);
        assert!(report.clamped);
    }

    #[test]
    fn queued_command_applies_at_step_boundary() {
        let mut world = PhysicsWorld::default();
        let h = world.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
        let mut backend = dynamics_backend();
        let mut pipe = FixedStepPipeline::new(0.01);
        pipe.prime(&world);
        pipe.queue_command(PhysicsCommand::SetLinearVelocity {
            body: h,
            velocity: Vec3::new(2.0, 0.0, 0.0),
        });
        assert_eq!(pipe.queued_commands(), 1);
        let report = pipe.advance(&mut backend, &mut world, 0.01);
        assert_eq!(report.commands_applied, 1);
        assert_eq!(pipe.queued_commands(), 0);
        // The body should have moved along +x from the applied velocity.
        assert!(world.bodies.position(h).unwrap().x > 0.0);
    }

    #[test]
    fn interpolated_pose_brackets_the_two_snapshots() {
        let mut world = PhysicsWorld::with_gravity(Vec3::new(0.0, -10.0, 0.0));
        let h = world.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
        let mut backend = dynamics_backend();
        let mut pipe = FixedStepPipeline::new(0.01);
        pipe.prime(&world);
        // Two steps then a partial: alpha in (0,1), pose between the last two.
        pipe.advance(&mut backend, &mut world, 0.02);
        pipe.advance(&mut backend, &mut world, 0.005);
        let prev_y = pipe
            .previous_snapshot()
            .unwrap()
            .pose(h)
            .unwrap()
            .position
            .y;
        let curr_y = pipe.current_snapshot().unwrap().pose(h).unwrap().position.y;
        let interp_y = pipe.interpolated_pose(h).unwrap().position.y;
        // Falling body: curr is below prev, interp strictly between them.
        assert!(curr_y < prev_y);
        assert!(interp_y <= prev_y && interp_y >= curr_y);
    }

    #[test]
    fn state_hash_is_stable_across_identical_runs() {
        let run = || {
            let mut world = PhysicsWorld::with_gravity(Vec3::new(0.0, -9.81, 0.0));
            world.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
            let mut backend = dynamics_backend();
            let mut pipe = FixedStepPipeline::new(0.01);
            pipe.prime(&world);
            for _ in 0..20 {
                pipe.advance(&mut backend, &mut world, 0.01);
            }
            pipe.state_hash()
        };
        assert_eq!(run(), run());
    }
}
