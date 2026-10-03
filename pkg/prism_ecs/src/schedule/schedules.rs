//! The [`Schedules`] resource: a world-owned registry mapping
//! [`ScheduleLabel`]s to runnable [`Schedule`]s, plus
//! [`World::run_schedule`] to execute one by label (design §8.2).
//!
//! This is what lets a world hold *more than one* schedule — a main loop
//! schedule, plus per-state-edge transition schedules keyed by
//! [`OnEnter`](crate::schedule::OnEnter)/[`OnExit`](crate::schedule::OnExit).
//! The [`States`](crate::schedule::State) machine (`state.rs`) drives those
//! transition schedules through [`World::run_schedule`].

use crate::collections::HashMap;
use crate::resource::Resource;
use crate::schedule::graph::Schedule;
use crate::schedule::label::{BoxedScheduleLabel, ScheduleLabel};
use crate::world::World;

/// A world-global map of [`ScheduleLabel`] → [`Schedule`].
///
/// Insert a schedule under a label with [`insert`](Schedules::insert), then run
/// it with [`World::run_schedule`]. Labels of different concrete types coexist
/// via [`BoxedScheduleLabel`].
#[derive(Default)]
pub struct Schedules {
    inner: HashMap<BoxedScheduleLabel, Schedule>,
}

impl Resource for Schedules {}

impl Schedules {
    /// Create an empty registry.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: HashMap::new(),
        }
    }

    /// Insert `schedule` under `label`, returning any schedule it displaced.
    #[inline]
    pub fn insert(&mut self, label: impl ScheduleLabel, schedule: Schedule) -> Option<Schedule> {
        self.inner.insert(BoxedScheduleLabel::new(label), schedule)
    }

    /// Borrow the schedule stored under `label`, if any.
    #[inline]
    #[must_use]
    pub fn get(&self, label: impl ScheduleLabel) -> Option<&Schedule> {
        self.inner.get(&BoxedScheduleLabel::new(label))
    }

    /// Mutably borrow the schedule stored under `label`, if any.
    #[inline]
    pub fn get_mut(&mut self, label: impl ScheduleLabel) -> Option<&mut Schedule> {
        self.inner.get_mut(&BoxedScheduleLabel::new(label))
    }

    /// Whether a schedule is stored under `label`.
    #[inline]
    #[must_use]
    pub fn contains(&self, label: impl ScheduleLabel) -> bool {
        self.inner.contains_key(&BoxedScheduleLabel::new(label))
    }

    /// Remove and return the schedule stored under `label`, if any.
    #[inline]
    pub fn remove(&mut self, label: impl ScheduleLabel) -> Option<Schedule> {
        self.inner.remove(&BoxedScheduleLabel::new(label))
    }

    /// How many schedules are registered.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether no schedules are registered.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Insert via an already-boxed label (internal fast path for re-insertion
    /// after a run, avoiding a second box allocation).
    #[inline]
    pub(crate) fn insert_boxed(
        &mut self,
        label: BoxedScheduleLabel,
        schedule: Schedule,
    ) -> Option<Schedule> {
        self.inner.insert(label, schedule)
    }

    /// Remove via an already-boxed label (internal fast path).
    #[inline]
    pub(crate) fn remove_boxed(&mut self, label: &BoxedScheduleLabel) -> Option<Schedule> {
        self.inner.remove(label)
    }
}

impl World {
    /// Run the schedule registered under `label` against this world.
    ///
    /// The schedule is temporarily **removed** from the [`Schedules`] resource
    /// (taking ownership) so it can run against `&mut World` without aliasing a
    /// `&mut Schedule` that still lives inside the world, then re-inserted
    /// afterwards. A missing [`Schedules`] resource or an unregistered label is
    /// a no-op, and a schedule that recursively asks to run itself finds its own
    /// slot temporarily empty and safely no-ops.
    pub fn run_schedule(&mut self, label: impl ScheduleLabel) {
        let boxed = BoxedScheduleLabel::new(label);
        let Some(mut schedule) = self
            .get_resource_mut::<Schedules>()
            .and_then(|schedules| schedules.remove_boxed(&boxed))
        else {
            return;
        };
        schedule.run(self);
        if let Some(schedules) = self.get_resource_mut::<Schedules>() {
            schedules.insert_boxed(boxed, schedule);
        }
    }
}
