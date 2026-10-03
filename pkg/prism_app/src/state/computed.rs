//! Computed states: modes *derived* from another state each frame (design §11).
//!
//! A computed state is not something gameplay code sets directly. Instead it is
//! a pure function of one *source* [`States`] value, recomputed every
//! [`StateTransition`] after the source has settled. The classic example is
//! "simulating" = "in game and not paused":
//!
//! ```
//! use prism_app::prelude::*;
//! use prism_app::state::ComputedStates;
//!
//! #[derive(Clone, PartialEq, Eq, Hash, Debug, Default)]
//! enum AppState { #[default] Menu, InGame, Paused }
//! impl States for AppState {}
//!
//! #[derive(Clone, PartialEq, Eq, Hash, Debug)]
//! struct Simulating;
//! impl States for Simulating {}
//! impl ComputedStates for Simulating {
//!     type SourceStates = AppState;
//!     fn compute(source: &AppState) -> Option<Self> {
//!         matches!(source, AppState::InGame).then_some(Simulating)
//!     }
//! }
//!
//! App::new()
//!     .insert_state(AppState::Menu)
//!     .add_computed_state::<Simulating>();
//! ```
//!
//! # Semantics
//!
//! Each frame [`recompute_computed_state::<C>`] reads
//! [`State<C::SourceStates>`](State) and calls [`C::compute`](ComputedStates::compute):
//!
//! - `Some(new)` where no [`State<C>`] exists yet → insert `State(new)`, run
//!   `OnEnter(new)` (the computed state *appears*).
//! - `Some(new)` differing from the current value → run `OnExit(old)`, insert
//!   `State(new)`, run `OnEnter(new)`.
//! - `None` while a [`State<C>`] exists → run `OnExit(old)` and **remove** the
//!   [`State<C>`] resource (the computed state *disappears*).
//! - unchanged (equal, or absent both before and after) → nothing runs.
//!
//! Because a computed state is single-direction, it has **no**
//! [`NextState<C>`](prism_ecs::schedule::NextState): setting it directly is
//! meaningless, so none is installed. Gate systems on it with the ordinary
//! [`in_state`](prism_ecs::schedule::in_state) condition and hook work on
//! `OnEnter(..)` / `OnExit(..)` exactly like a base state.
//!
//! # Scope
//!
//! The source is a single [`States`] type ([`SourceStates`](ComputedStates::SourceStates)).
//! Deriving from several states at once is expressed by making the source a
//! composite `States` value, and deriving a computed state from *another*
//! computed state (a second `Compute` tier) is intentionally not modelled —
//! both the source and this computed state settle in the one
//! [`StateTransitionSet::Compute`] group, so a computed-of-computed chain would
//! have no ordering guarantee. That remains future work and is documented here
//! rather than faked.
//!
//! [`StateTransition`]: crate::schedule::StateTransition

use core::any::TypeId;

use prism_ecs::schedule::{IntoSystemConfigs, OnEnter, OnExit, State, States};
use prism_ecs::world::World;

use crate::app::App;
use crate::schedule::StateTransition;
use crate::state::StateTransitionSet;

/// A [`States`] value computed each frame from another state.
///
/// Implement this on a type that also implements [`States`] (so its
/// `OnEnter`/`OnExit` schedules are valid labels). The engine recomputes it in
/// the [`StateTransition`] phase after the source transitions have settled; see
/// the [module docs](crate::state::computed) for the full lifecycle.
pub trait ComputedStates: States {
    /// The source state this value is derived from.
    type SourceStates: States;

    /// Compute this state from the current `source` mode.
    ///
    /// Return `Some(value)` when the computed state should exist (with that
    /// value) and `None` when it should not exist at all.
    fn compute(source: &Self::SourceStates) -> Option<Self>
    where
        Self: Sized;
}

/// Exclusive system that recomputes one [`ComputedStates`] type `C` from its
/// source and runs the matching `OnExit`/`OnEnter` edges.
///
/// Registered by [`App::add_computed_state`] into
/// [`StateTransitionSet::Compute`]. See the [module docs](crate::state::computed)
/// for the transition table.
pub fn recompute_computed_state<C: ComputedStates>(world: &mut World) {
    let next = world
        .get_resource::<State<C::SourceStates>>()
        .and_then(|source| C::compute(source.get()));
    let current = world.get_resource::<State<C>>().map(|state| state.0.clone());

    match (current, next) {
        (None, None) => {}
        (Some(old), Some(new)) if old == new => {}
        (Some(old), Some(new)) => {
            world.run_schedule(OnExit(old));
            world.insert_resource(State::new(new.clone()));
            world.run_schedule(OnEnter(new));
        }
        (Some(old), None) => {
            world.run_schedule(OnExit(old));
            world.remove_resource::<State<C>>();
        }
        (None, Some(new)) => {
            world.insert_resource(State::new(new.clone()));
            world.run_schedule(OnEnter(new));
        }
    }
}

impl App {
    /// Register a [`ComputedStates`] type `C`, derived each frame from its
    /// [`SourceStates`](ComputedStates::SourceStates) (design §11).
    ///
    /// Wires [`recompute_computed_state::<C>`] into the [`StateTransition`]
    /// phase in [`StateTransitionSet::Compute`], ordered after
    /// [`StateTransitionSet::Apply`] so it reads the frame's settled source
    /// mode. The source state must itself be registered (e.g. with
    /// [`insert_state`](App::insert_state)); otherwise `C::compute` simply never
    /// sees a source and the computed state stays absent.
    ///
    /// Idempotent: wiring happens only once per `C`.
    pub fn add_computed_state<C: ComputedStates>(&mut self) -> &mut Self {
        if self.initialized_states.insert(TypeId::of::<State<C>>()) {
            self.add_systems(
                StateTransition,
                recompute_computed_state::<C>
                    .in_set(StateTransitionSet::Compute)
                    .after(StateTransitionSet::Apply),
            );
        }
        self
    }
}
