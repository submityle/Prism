//! A label-keyed collection of [`Schedule`]s owned by a
//! [`SubApp`](crate::sub_app::SubApp).
//!
//! This is the registry the main-frame loop walks: for each phase label it
//! looks up the matching [`Schedule`] and runs it against the world.

use std::collections::HashMap;

use prism_ecs::schedule::Schedule;

use crate::schedule_label::{ScheduleLabel, ScheduleLabelId};

/// A map from [`ScheduleLabelId`] to the [`Schedule`] registered for that
/// label.
///
/// Schedules are created lazily: [`entry`](Schedules::entry) inserts an empty
/// [`Schedule`] the first time a label is used, so `add_systems(Update, ..)`
/// works before any schedule was explicitly installed.
#[derive(Default)]
pub struct Schedules {
    map: HashMap<ScheduleLabelId, Schedule>,
}

impl Schedules {
    /// Create an empty collection.
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    /// Insert (or replace) the schedule stored under `label`, returning the
    /// previous schedule if one existed.
    pub fn insert(&mut self, label: impl ScheduleLabel, schedule: Schedule) -> Option<Schedule> {
        self.map.insert(label.id(), schedule)
    }

    /// Get a shared reference to the schedule for `label`, if present.
    pub fn get(&self, label: impl ScheduleLabel) -> Option<&Schedule> {
        self.map.get(&label.id())
    }

    /// Get a mutable reference to the schedule for `label`, if present.
    pub fn get_mut(&mut self, label: impl ScheduleLabel) -> Option<&mut Schedule> {
        self.map.get_mut(&label.id())
    }

    /// Get a mutable reference to the schedule for `label`, inserting an empty
    /// [`Schedule`] first if the label has no schedule yet.
    pub fn entry(&mut self, label: impl ScheduleLabel) -> &mut Schedule {
        self.map.entry(label.id()).or_default()
    }

    /// Whether a schedule is registered for `label`.
    pub fn contains(&self, label: impl ScheduleLabel) -> bool {
        self.map.contains_key(&label.id())
    }

    /// The number of registered schedules.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether no schedules are registered.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}
