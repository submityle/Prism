//! Relative insertion into a sub-app's startup and variable-frame phase order.
//!
//! Core phases cannot be removed, moved, or inserted a second time. The native
//! fixed loop is one frame-level anchor; its accumulator and inner tick order
//! remain owned by [`crate::fixed`].

use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use prism_ecs::schedule::{BoxedScheduleLabel, Schedule, ScheduleLabel, Schedules};
use prism_ecs::world::World;

use crate::fixed::{
    AfterFixedMainLoop, BeforeFixedMainLoop, FixedFirst, FixedLast, FixedPostUpdate,
    FixedPreUpdate, FixedUpdate,
};
use crate::schedule::{
    ExitConfirmation, First, Last, PostStartup, PostUpdate, PreStartup, PreUpdate, Shutdown,
    Startup, StateTransition, Update,
};
use crate::sub_app::SubApp;

/// Frame-level anchor for the native fixed-step loop, between [`First`] and
/// [`PreUpdate`]. Inserting before/after it runs once per frame, even when the
/// loop takes zero or multiple steps. This marker is not a runnable schedule;
/// put fixed-rate systems in [`FixedUpdate`] instead.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct RunFixedMainLoop;

/// A rejected relative phase insertion. Validation checks duplicates before
/// looking for the anchor and leaves both the order and schedule registry intact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PhaseOrderError {
    /// This label is already in either phase sequence, or is reserved for the
    /// fixed tick group, its hooks, or the exit lifecycle.
    DuplicatePhase(BoxedScheduleLabel),
    /// The anchor is absent from both the startup and variable-frame sequences.
    /// Inner fixed phases and exit schedules are not frame-level anchors.
    MissingAnchor(BoxedScheduleLabel),
}

impl core::fmt::Display for PhaseOrderError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::DuplicatePhase(label) => {
                write!(f, "phase {label:?} is already present or reserved")
            }
            Self::MissingAnchor(label) => write!(f, "phase anchor {label:?} is missing"),
        }
    }
}

impl core::error::Error for PhaseOrderError {}

pub(crate) struct Phase {
    label: BoxedScheduleLabel,
    #[cfg_attr(not(any(feature = "std", test)), allow(dead_code))]
    pub(crate) name: &'static str,
    pub(crate) run: Arc<dyn Fn(&mut SubApp) + Send + Sync>,
}

impl Phase {
    fn schedule<L: ScheduleLabel + Clone>(label: L, name: &'static str, startup: bool) -> Self {
        Self {
            label: BoxedScheduleLabel::new(label.clone()),
            name,
            // Keep the concrete label at dispatch: passing a BoxedScheduleLabel
            // to World::run_schedule would look up a different (double-boxed) key.
            run: Arc::new(move |sub_app| {
                if startup {
                    sub_app.run_schedule(label.clone());
                } else {
                    sub_app.run_frame_schedule(label.clone());
                }
            }),
        }
    }
}

pub(crate) struct PhaseOrder {
    pub(crate) startup: Vec<Phase>,
    pub(crate) frame: Vec<Phase>,
}

impl Default for PhaseOrder {
    fn default() -> Self {
        Self {
            startup: vec![
                Phase::schedule(PreStartup, "PreStartup", true),
                Phase::schedule(Startup, "Startup", true),
                Phase::schedule(PostStartup, "PostStartup", true),
            ],
            frame: vec![
                Phase::schedule(First, "First", false),
                Phase {
                    label: BoxedScheduleLabel::new(RunFixedMainLoop),
                    name: "RunFixedMainLoop",
                    run: Arc::new(|sub_app| crate::fixed::run_fixed_main_loop(&mut sub_app.world)),
                },
                Phase::schedule(PreUpdate, "PreUpdate", false),
                Phase::schedule(StateTransition, "StateTransition", false),
                Phase::schedule(Update, "Update", false),
                Phase::schedule(PostUpdate, "PostUpdate", false),
                Phase::schedule(Last, "Last", false),
            ],
        }
    }
}

impl PhaseOrder {
    pub(crate) fn insert<L: ScheduleLabel + Clone>(
        &mut self,
        world: &mut World,
        anchor: impl ScheduleLabel,
        label: L,
        after: bool,
    ) -> Result<(), PhaseOrderError> {
        let key = BoxedScheduleLabel::new(label.clone());
        if self
            .startup
            .iter()
            .chain(&self.frame)
            .any(|p| p.label == key)
            || reserved_label(&key)
        {
            return Err(PhaseOrderError::DuplicatePhase(key));
        }
        let anchor = BoxedScheduleLabel::new(anchor);
        let (phases, index, startup) =
            if let Some(index) = self.startup.iter().position(|p| p.label == anchor) {
                (&mut self.startup, index, true)
            } else if let Some(index) = self.frame.iter().position(|p| p.label == anchor) {
                (&mut self.frame, index, false)
            } else {
                return Err(PhaseOrderError::MissingAnchor(anchor));
            };
        let phase = Phase::schedule(label.clone(), core::any::type_name::<L>(), startup);
        world.init_resource::<Schedules>();
        let schedules = world.resource_mut::<Schedules>();
        if !schedules.contains(label.clone()) {
            schedules.insert(label, Schedule::new());
        }
        phases.insert(index + usize::from(after), phase);
        Ok(())
    }
}

fn reserved_label(key: &BoxedScheduleLabel) -> bool {
    // These schedules already have their own cadence. Reusing them as custom
    // frame/startup phases would execute them twice or at the wrong frequency.
    key.as_dyn().dyn_eq(&FixedFirst)
        || key.as_dyn().dyn_eq(&FixedPreUpdate)
        || key.as_dyn().dyn_eq(&FixedUpdate)
        || key.as_dyn().dyn_eq(&FixedPostUpdate)
        || key.as_dyn().dyn_eq(&FixedLast)
        || key.as_dyn().dyn_eq(&BeforeFixedMainLoop)
        || key.as_dyn().dyn_eq(&AfterFixedMainLoop)
        || key.as_dyn().dyn_eq(&Shutdown)
        || key.as_dyn().dyn_eq(&ExitConfirmation)
}

#[cfg(test)]
mod tests;
