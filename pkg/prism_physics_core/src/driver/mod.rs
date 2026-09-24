//! Simulation driver: the run-mode dispatch extension point.
//!
//! The [`SimulationDriver`] selects *how* a simulation is advanced over time.
//! M0 implements exactly one mode for real, [`DriveMode::Realtime`], which
//! steps a [`PhysicsBackend`] once per call. The [`DriveMode::Offline`] and
//! [`DriveMode::Playback`] variants are genuine, reserved dispatch points for
//! later milestones: they are not faked. Calling [`SimulationDriver::drive`]
//! with a reserved mode returns an honest [`DriveOutcome::Unsupported`] result
//! rather than panicking or pretending to do work.

pub mod fixed_step;

use crate::backend::PhysicsBackend;
use crate::world::PhysicsWorld;

/// An opaque reference to a cached/baked simulation, reserved for playback.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct CacheHandle(pub u64);

/// How a [`SimulationDriver`] advances the simulation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum DriveMode {
    /// Advance one step per [`SimulationDriver::drive`] call, in lockstep with
    /// the caller's frame loop. This is the only mode executed in M0.
    Realtime,
    /// Reserved: run a fixed-size batch offline (baking). Not executed in M0.
    Offline {
        /// Sub-steps per frame for the offline run.
        substeps: u32,
        /// Solver iterations per sub-step (reserved for constraint solving).
        iterations: u32,
        /// Total number of frames to bake.
        target_frames: u32,
    },
    /// Reserved: play back a previously baked cache. Not executed in M0.
    Playback(CacheHandle),
}

/// The result of a [`SimulationDriver::drive`] call.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DriveOutcome {
    /// The world was advanced by one real step.
    Stepped,
    /// The requested mode is a reserved extension point not yet implemented in
    /// M0. `reason` names the unsupported mode.
    Unsupported {
        /// Human-readable name of the unsupported mode.
        reason: &'static str,
    },
}

/// Drives a simulation according to a [`DriveMode`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SimulationDriver {
    /// The active drive mode.
    pub mode: DriveMode,
}

impl SimulationDriver {
    /// Creates a driver for the given `mode`.
    #[must_use]
    pub fn new(mode: DriveMode) -> SimulationDriver {
        SimulationDriver { mode }
    }

    /// Returns `true` if the driver is in [`DriveMode::Realtime`].
    #[must_use]
    pub fn is_realtime(&self) -> bool {
        matches!(self.mode, DriveMode::Realtime)
    }

    /// Advances the simulation by one step using `backend`, honouring the
    /// current mode.
    ///
    /// Only [`DriveMode::Realtime`] performs work in M0; the reserved modes
    /// return [`DriveOutcome::Unsupported`] with a descriptive reason. This is
    /// intentional: the reserved dispatch points are real branches that report
    /// their status honestly instead of using `unimplemented!()`.
    pub fn drive(
        &self,
        backend: &mut dyn PhysicsBackend,
        world: &mut PhysicsWorld,
        dt: f32,
    ) -> DriveOutcome {
        match self.mode {
            DriveMode::Realtime => {
                backend.step(world, dt);
                DriveOutcome::Stepped
            }
            DriveMode::Offline { .. } => DriveOutcome::Unsupported {
                reason:
                    "Offline baking is reserved for a later milestone (M0 executes Realtime only)",
            },
            DriveMode::Playback(_) => DriveOutcome::Unsupported {
                reason:
                    "Cache playback is reserved for a later milestone (M0 executes Realtime only)",
            },
        }
    }
}

impl Default for SimulationDriver {
    fn default() -> Self {
        SimulationDriver::new(DriveMode::Realtime)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;
    use crate::state::body::BodyDesc;
    use glam::Vec3;

    #[test]
    fn realtime_steps_the_world() {
        let mut world = PhysicsWorld::with_gravity(Vec3::new(0.0, -9.81, 0.0));
        let h = world.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
        let mut backend = CpuBackend::with_defaults();
        let driver = SimulationDriver::new(DriveMode::Realtime);
        assert!(driver.is_realtime());
        assert_eq!(
            driver.drive(&mut backend, &mut world, 0.1),
            DriveOutcome::Stepped
        );
        assert!(world.bodies.position(h).unwrap().y < 0.0);
    }

    #[test]
    fn reserved_modes_report_unsupported() {
        let mut world = PhysicsWorld::default();
        let mut backend = CpuBackend::with_defaults();

        let offline = SimulationDriver::new(DriveMode::Offline {
            substeps: 4,
            iterations: 8,
            target_frames: 100,
        });
        assert!(!offline.is_realtime());
        assert!(matches!(
            offline.drive(&mut backend, &mut world, 0.1),
            DriveOutcome::Unsupported { .. }
        ));

        let playback = SimulationDriver::new(DriveMode::Playback(CacheHandle(7)));
        assert!(matches!(
            playback.drive(&mut backend, &mut world, 0.1),
            DriveOutcome::Unsupported { .. }
        ));
    }
}
