//! A [`SubApp`]: one [`World`] plus its label-keyed [`Schedules`].
//!
//! An [`App`](crate::app::App) owns a *main* `SubApp` and (in later milestones)
//! named secondary sub-apps such as a render sub-app driven by a one-way
//! extract step. M0 ships only the main sub-app and the variable-step frame
//! loop over it.

use prism_ecs::world::World;

use crate::schedule_label::{CoreSchedule, ScheduleLabel};
use crate::schedules::Schedules;

/// A self-contained unit of simulation: a [`World`] and the [`Schedules`] that
/// drive it.
pub struct SubApp {
    /// The ECS world this sub-app simulates.
    pub world: World,
    /// The label-keyed schedules run against [`world`](SubApp::world).
    pub schedules: Schedules,
}

impl Default for SubApp {
    fn default() -> Self {
        Self::new()
    }
}

impl SubApp {
    /// Create a sub-app with a fresh [`World`] and no schedules.
    pub fn new() -> Self {
        Self {
            world: World::new(),
            schedules: Schedules::new(),
        }
    }

    /// Run the schedule registered under `label` against this sub-app's world.
    ///
    /// A missing label is a no-op: a phase with no systems simply does nothing.
    /// This keeps the frame loop total even when a user never adds systems to,
    /// say, `PreUpdate`.
    pub fn run_schedule(&mut self, label: impl ScheduleLabel) {
        if let Some(schedule) = self.schedules.get_mut(label) {
            schedule.run(&mut self.world);
        }
    }

    /// Run one variable-step frame: the M0 subset of the design-doc main-frame
    /// order (§7), `First → PreUpdate → Update → PostUpdate → Last`.
    ///
    /// # Honestly deferred
    ///
    /// The full order also interleaves `RunFixedMainLoop` (fixed-timestep inner
    /// loop, M2) between `First` and `PreUpdate`, and `StateTransition` (the
    /// state machine, M1) between `PreUpdate` and `Update`. Those phases are
    /// absent here, not stubbed, and land with the milestones that implement
    /// their behavior.
    pub fn update(&mut self) {
        for label in CoreSchedule::FRAME_ORDER {
            self.run_schedule(label);
        }
    }
}
