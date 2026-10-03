//! Finite state machine (`States`) wired to transition schedules (design
//! §8.2 — `States`).
//!
//! A [`States`] type (usually a small `enum`) names the discrete modes a world
//! can be in — `MainMenu`, `Loading`, `InGame`, `Paused`. The world holds the
//! live mode in a [`State<S>`] resource and a pending request in a
//! [`NextState<S>`] resource. Each tick an exclusive
//! [`apply_state_transition::<S>`] system consumes any pending request and, when
//! it differs from the current mode, runs the [`OnExit`] schedule for the old
//! mode followed by the [`OnEnter`] schedule for the new one.
//!
//! [`OnEnter`]/[`OnExit`] are [`ScheduleLabel`](crate::schedule::ScheduleLabel)s
//! carrying a state value, so setup code registers one
//! [`Schedule`](crate::schedule::Schedule) per `(mode, edge)` in the
//! [`Schedules`](crate::schedule::Schedules) resource. Gate ordinary systems on
//! the current mode with the [`in_state`] run condition.

use core::fmt::Debug;
use core::hash::Hash;

use crate::resource::Resource;
use crate::system::Res;
use crate::world::World;

/// A discrete world mode usable with the [`State`]/[`NextState`] machinery.
///
/// Blanket requirements mirror a schedule label (so [`OnEnter`]/[`OnExit`] are
/// valid labels) plus [`Default`] is *not* required — the initial mode is
/// provided explicitly when the [`State`] resource is inserted.
pub trait States: Clone + PartialEq + Eq + Hash + Debug + Send + Sync + 'static {}

/// The world's current [`States`] value, stored as a resource.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct State<S: States>(
    /// The active mode.
    pub S,
);

impl<S: States> State<S> {
    /// Construct a `State` wrapping `value`.
    #[inline]
    #[must_use]
    pub fn new(value: S) -> Self {
        Self(value)
    }

    /// Borrow the active mode.
    #[inline]
    #[must_use]
    pub fn get(&self) -> &S {
        &self.0
    }
}

impl<S: States> Resource for State<S> {}

/// A pending request to switch to another [`States`] value next time
/// [`apply_state_transition`] runs. `None` means "no change requested".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NextState<S: States>(
    /// The queued mode, or `None` when no transition is pending.
    pub Option<S>,
);

impl<S: States> NextState<S> {
    /// Queue a transition to `value`, overwriting any earlier pending request.
    #[inline]
    pub fn set(&mut self, value: S) {
        self.0 = Some(value);
    }

    /// Whether a transition is currently queued.
    #[inline]
    #[must_use]
    pub fn is_pending(&self) -> bool {
        self.0.is_some()
    }
}

impl<S: States> Default for NextState<S> {
    #[inline]
    fn default() -> Self {
        Self(None)
    }
}

impl<S: States> Resource for NextState<S> {}

/// A [`ScheduleLabel`](crate::schedule::ScheduleLabel) for the schedule run when
/// *entering* the given [`States`] value.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OnEnter<S: States>(
    /// The mode being entered.
    pub S,
);

/// A [`ScheduleLabel`](crate::schedule::ScheduleLabel) for the schedule run when
/// *exiting* the given [`States`] value.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OnExit<S: States>(
    /// The mode being exited.
    pub S,
);

/// A run condition that is `true` while the current [`State<S>`] equals `value`.
///
/// Returns `false` if no [`State<S>`] resource exists yet, so systems gated on a
/// mode simply don't run before the state machine is initialised.
#[inline]
pub fn in_state<S: States>(value: S) -> impl FnMut(Option<Res<State<S>>>) -> bool {
    move |current: Option<Res<State<S>>>| current.is_some_and(|current| current.0 == value)
}

/// Exclusive system that applies one pending [`NextState<S>`] transition.
///
/// Takes (and clears) any queued [`NextState<S>`]. If a value was queued and it
/// differs from the current [`State<S>`], it runs `OnExit(old)` then sets the
/// new state then runs `OnEnter(new)`. With no current state yet, it sets the
/// state and runs `OnEnter(new)` (the first entry). Equal old/new is consumed
/// without running either edge, so a redundant request can't re-fire a
/// transition.
pub fn apply_state_transition<S: States>(world: &mut World) {
    let pending = match world.get_resource_mut::<NextState<S>>() {
        Some(next) => next.0.take(),
        None => return,
    };
    let Some(next) = pending else {
        return;
    };

    match world.get_resource::<State<S>>().map(|state| state.0.clone()) {
        Some(current) if current == next => {
            // Redundant request: already in this mode. Consumed, no edges run.
        }
        Some(current) => {
            world.run_schedule(OnExit(current));
            world.insert_resource(State(next.clone()));
            world.run_schedule(OnEnter(next));
        }
        None => {
            world.insert_resource(State(next.clone()));
            world.run_schedule(OnEnter(next));
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use std::vec::Vec;

    use super::*;
    use crate::schedule::{IntoSystemConfigs, Schedule, Schedules};
    use crate::system::ResMut;

    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    enum Mode {
        Menu,
        Game,
    }
    impl States for Mode {}

    /// A resource recording the order in which transition hooks fired.
    #[derive(Default)]
    struct Trace(Vec<&'static str>);
    impl Resource for Trace {}

    fn transition_world() -> World {
        let mut world = World::new();
        world.init_resource::<Trace>();
        world.insert_resource(NextState::<Mode>(None));

        let mut schedules = Schedules::new();

        let mut exit_menu = Schedule::new();
        exit_menu.add_systems(|mut t: ResMut<Trace>| t.0.push("exit:menu"));
        schedules.insert(OnExit(Mode::Menu), exit_menu);

        let mut enter_game = Schedule::new();
        enter_game.add_systems(|mut t: ResMut<Trace>| t.0.push("enter:game"));
        schedules.insert(OnEnter(Mode::Game), enter_game);

        let mut enter_menu = Schedule::new();
        enter_menu.add_systems(|mut t: ResMut<Trace>| t.0.push("enter:menu"));
        schedules.insert(OnEnter(Mode::Menu), enter_menu);

        world.insert_resource(schedules);
        world
    }

    #[test]
    fn first_transition_runs_only_on_enter() {
        let mut world = transition_world();
        world.resource_mut::<NextState<Mode>>().set(Mode::Menu);
        apply_state_transition::<Mode>(&mut world);

        assert_eq!(world.resource::<State<Mode>>().get(), &Mode::Menu);
        assert_eq!(world.resource::<Trace>().0, ["enter:menu"]);
        // NextState consumed exactly once.
        assert!(!world.resource::<NextState<Mode>>().is_pending());
    }

    #[test]
    fn transition_runs_on_exit_then_on_enter_in_order() {
        let mut world = transition_world();
        world.insert_resource(State(Mode::Menu));
        world.resource_mut::<NextState<Mode>>().set(Mode::Game);

        apply_state_transition::<Mode>(&mut world);

        assert_eq!(world.resource::<State<Mode>>().get(), &Mode::Game);
        assert_eq!(world.resource::<Trace>().0, ["exit:menu", "enter:game"]);
    }

    #[test]
    fn redundant_transition_is_consumed_without_running_edges() {
        let mut world = transition_world();
        world.insert_resource(State(Mode::Menu));
        world.resource_mut::<NextState<Mode>>().set(Mode::Menu);

        apply_state_transition::<Mode>(&mut world);

        assert_eq!(world.resource::<State<Mode>>().get(), &Mode::Menu);
        assert!(world.resource::<Trace>().0.is_empty());
        assert!(!world.resource::<NextState<Mode>>().is_pending());
    }

    #[test]
    fn next_state_consumed_once_second_apply_is_noop() {
        let mut world = transition_world();
        world.insert_resource(State(Mode::Menu));
        world.resource_mut::<NextState<Mode>>().set(Mode::Game);

        apply_state_transition::<Mode>(&mut world);
        apply_state_transition::<Mode>(&mut world);

        // Only the first apply produced edges; the second found no pending req.
        assert_eq!(world.resource::<Trace>().0, ["exit:menu", "enter:game"]);
    }

    #[test]
    fn in_state_gates_on_current_mode() {
        let mut world = World::new();
        let mut schedule = Schedule::new();
        schedule.add_systems(
            (|mut count: ResMut<Trace>| count.0.push("ran"))
                .run_if(in_state(Mode::Game)),
        );
        world.init_resource::<Trace>();

        // No State resource yet: condition is false, system skipped.
        schedule.run(&mut world);
        assert!(world.resource::<Trace>().0.is_empty());

        // In the wrong mode: still skipped.
        world.insert_resource(State(Mode::Menu));
        schedule.run(&mut world);
        assert!(world.resource::<Trace>().0.is_empty());

        // In the matching mode: runs.
        world.insert_resource(State(Mode::Game));
        schedule.run(&mut world);
        assert_eq!(world.resource::<Trace>().0, ["ran"]);
    }

    #[test]
    fn apply_transition_is_addable_as_exclusive_system() {
        // apply_state_transition::<S> is a plain fn(&mut World): it must be
        // addable to a schedule as an exclusive system and drive transitions.
        let mut world = transition_world();
        world.resource_mut::<NextState<Mode>>().set(Mode::Menu);

        let mut schedule = Schedule::new();
        schedule.add_systems(apply_state_transition::<Mode>);
        schedule.run(&mut world);

        assert_eq!(world.resource::<State<Mode>>().get(), &Mode::Menu);
        assert_eq!(world.resource::<Trace>().0, ["enter:menu"]);
    }

    #[test]
    fn run_schedule_is_reentrant_safe_noop_on_self() {
        // A schedule that asks to run itself finds its slot empty and no-ops
        // rather than aliasing or recursing forever.
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
        struct SelfLabel;

        let mut world = World::new();
        world.init_resource::<Trace>();
        let mut schedules = Schedules::new();
        let mut sched = Schedule::new();
        sched.add_systems((
            |mut t: ResMut<Trace>| t.0.push("outer"),
            |w: &mut World| w.run_schedule(SelfLabel),
        )
            .chain());
        schedules.insert(SelfLabel, sched);
        world.insert_resource(schedules);

        world.run_schedule(SelfLabel);
        // Ran once; the re-entrant self-run found an empty slot and no-oped.
        assert_eq!(world.resource::<Trace>().0, ["outer"]);
        // And the schedule was re-inserted afterward.
        assert!(world.resource::<Schedules>().contains(SelfLabel));
    }

    #[test]
    fn boxed_schedule_labels_of_different_types_do_not_collide() {
        use crate::schedule::label::BoxedScheduleLabel;
        // Distinct label types with structurally equal inner values must remain
        // distinct keys (TypeId mixed into the hash; dyn_eq checks type).
        let a = BoxedScheduleLabel::new(OnEnter(Mode::Menu));
        let b = BoxedScheduleLabel::new(OnExit(Mode::Menu));
        assert_ne!(a, b);
        assert_eq!(a, BoxedScheduleLabel::new(OnEnter(Mode::Menu)));
    }
}
