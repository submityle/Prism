//! A [`SubApp`]: one [`World`] whose world-owned
//! [`Schedules`] resource holds the phase
//! schedules that drive it.
//!
//! An [`App`](crate::app::App) owns a *main* `SubApp` and (in later
//! milestones) named secondary sub-apps such as a render sub-app driven by a
//! one-way extract step. M1 ships only the main sub-app and the variable-step
//! frame loop over it.
//!
//! Per design §5 the sub-app **reuses the `prism_ecs` scheduling graph**: the
//! schedules live inside the world (not a separate hand-rolled registry), and
//! each phase is run by label through
//! [`World::run_schedule`](prism_ecs::world::World::run_schedule).

use prism_ecs::schedule::{ScheduleLabel, Schedules};
use prism_ecs::world::World;

use crate::schedule::{
    First, Last, PostStartup, PostUpdate, PreStartup, PreUpdate, StateTransition, Startup, Update,
};

/// A self-contained unit of simulation: a [`World`] whose
/// [`Schedules`] resource drives it.
pub struct SubApp {
    /// The ECS world this sub-app simulates. Its [`Schedules`] resource owns
    /// the phase schedules.
    pub world: World,
}

impl Default for SubApp {
    fn default() -> Self {
        Self::new()
    }
}

impl SubApp {
    /// Create a sub-app with a fresh [`World`] holding an empty
    /// [`Schedules`] resource.
    pub fn new() -> Self {
        let mut world = World::new();
        world.init_resource::<Schedules>();
        Self { world }
    }

    /// Run the schedule registered under `label` against this sub-app's world.
    ///
    /// A missing label is a no-op: a phase with no schedule simply does
    /// nothing. This keeps the frame loop total even when a user never adds
    /// systems to, say, `PreUpdate`.
    pub fn run_schedule(&mut self, label: impl ScheduleLabel) {
        self.world.run_schedule(label);
    }

    /// Run the startup phases exactly once, in order
    /// (`PreStartup → Startup → PostStartup`, design §7).
    pub fn run_startup(&mut self) {
        self.run_schedule(PreStartup);
        self.run_schedule(Startup);
        self.run_schedule(PostStartup);
    }

    /// Run one variable-step frame: the M1 main-frame order (design §7),
    /// `First → PreUpdate → StateTransition → Update → PostUpdate → Last`.
    ///
    /// # Honestly deferred
    ///
    /// The full order also interleaves `RunFixedMainLoop` (the fixed-timestep
    /// inner loop, M2) between `First` and `PreUpdate`. That phase is absent
    /// here, not stubbed, and lands with the milestone that implements its
    /// accumulator behavior.
    pub fn update(&mut self) {
        self.run_schedule(First);
        self.run_schedule(PreUpdate);
        self.run_schedule(StateTransition);
        self.run_schedule(Update);
        self.run_schedule(PostUpdate);
        self.run_schedule(Last);
    }
}
