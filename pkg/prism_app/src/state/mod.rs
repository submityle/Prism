//! Finite state-machine assembly on the [`App`] (design §11).
//!
//! A [`States`] type (usually a small `enum`) names the discrete modes a world
//! can be in — `MainMenu`, `Loading`, `InGame`, `Paused`. This module wires
//! such types into an app and layers the three state-machine *depth* features
//! the design calls for on top of the base [`prism_ecs`] primitives:
//!
//! - **Base states** — [`App::insert_state`] / [`App::init_state`] install an
//!   explicit (or [`Default`]) initial mode and the per-frame transition.
//! - **[Computed states](computed)** — [`App::add_computed_state`] installs a
//!   [`ComputedStates`] type that is *derived* from another state each frame
//!   (single-direction, no queue of its own).
//! - **[Sub-states](sub)** — [`App::add_sub_state`] installs a [`SubStates`]
//!   type that only exists while a parent mode is active and auto-exits when
//!   the parent leaves.
//! - **[State-scoped entities](scoped)** — [`App::enable_state_scoped_entities`]
//!   despawns entities tagged with [`StateScoped<S>`] once their owning mode is
//!   no longer current, so switching scenes leaves no stragglers.
//!
//! The underlying machinery ([`State`], [`NextState`], [`OnEnter`], [`OnExit`],
//! [`in_state`], [`apply_state_transition`]) lives in [`prism_ecs`] and is
//! re-exported through the crate prelude; this module provides the App-side
//! wiring, the fixed point in the frame order where transitions are applied,
//! and the derived/scoped features that `prism_ecs` deliberately leaves to the
//! App layer.
//!
//! # Transition ordering
//!
//! Everything the state machine does happens in the [`StateTransition`] phase
//! (design §7: between `PreUpdate` and `Update`). Within that phase the work is
//! split into two ordered groups via [`StateTransitionSet`]:
//!
//! 1. [`StateTransitionSet::Apply`] — base-state transitions
//!    ([`apply_state_transition::<S>`]). The authoritative [`State<S>`] values
//!    settle here.
//! 2. [`StateTransitionSet::Compute`] — derived work that *reads* the settled
//!    base states: computed-state recomputation, sub-state existence/transition,
//!    and state-scoped despawn. Ordered strictly `after` `Apply` so it always
//!    observes the current frame's base modes.
//!
//! Computed and sub states derive from **base** [`States`] (their
//! `SourceStates` is a `States`), so a single `Compute` tier after `Apply` is
//! sufficient; chaining a computed state off another computed state is not
//! modelled (documented in [`computed`]).
//!
//! # Deferred-entry model
//!
//! Inserting a base state does **not** eagerly create the [`State<S>`] resource
//! or run `OnEnter(initial)`. Instead it:
//!
//! 1. inserts [`NextState<S>`] holding the initial mode as a *pending* request,
//!    and
//! 2. registers the exclusive [`apply_state_transition::<S>`] system into the
//!    [`StateTransition`] phase in [`StateTransitionSet::Apply`].
//!
//! The first time the [`StateTransition`] phase runs, that system sees no
//! current [`State<S>`], so it inserts `State(initial)` and runs the
//! `OnEnter(initial)` schedule. From then on it consumes any queued
//! [`NextState<S>`] and runs the `OnExit(old)` then `OnEnter(new)` edges.
//!
//! This matches the ECS transition semantics exactly (one code path serves both
//! "first entry" and "later transition"), so `OnEnter(initial)` fires through
//! the same mechanism as every subsequent enter — no special-cased bootstrap.
//! The observable consequence, documented rather than hidden: the [`State<S>`]
//! resource and `OnEnter(initial)` systems become live on the **first frame's**
//! [`StateTransition`] phase, not at `insert_state` time. Code that must observe
//! the initial mode before the first frame should read the value it passed in,
//! not query [`State<S>`].
//!
//! [`State`]: prism_ecs::schedule::State
//! [`State<S>`]: prism_ecs::schedule::State
//! [`NextState`]: prism_ecs::schedule::NextState
//! [`NextState<S>`]: prism_ecs::schedule::NextState
//! [`OnEnter`]: prism_ecs::schedule::OnEnter
//! [`OnExit`]: prism_ecs::schedule::OnExit
//! [`in_state`]: prism_ecs::schedule::in_state
//! [`apply_state_transition`]: prism_ecs::schedule::apply_state_transition
//! [`apply_state_transition::<S>`]: prism_ecs::schedule::apply_state_transition

use core::any::TypeId;

use prism_ecs::schedule::{
    apply_state_transition, IntoSystemConfigs, NextState, States, SystemSet, SystemSetId,
};

use crate::app::App;
use crate::schedule::StateTransition;

pub mod computed;
pub mod scoped;
pub mod sub;

pub use computed::ComputedStates;
pub use scoped::StateScoped;
pub use sub::SubStates;

/// Ordering anchors inside the [`StateTransition`] phase (design §7, §11).
///
/// Base-state transitions run in [`Apply`](StateTransitionSet::Apply); all
/// derived work (computed states, sub-states, state-scoped cleanup) runs in
/// [`Compute`](StateTransitionSet::Compute), ordered strictly after `Apply` so
/// it reads the frame's settled [`State<S>`](prism_ecs::schedule::State)
/// values. Both groups live in the single [`StateTransition`] schedule.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum StateTransitionSet {
    /// Base [`States`] transitions ([`apply_state_transition`]).
    Apply,
    /// Derived state work that reads the settled base states: computed-state
    /// recomputation, sub-state existence/transition, and state-scoped despawn.
    Compute,
}

impl SystemSet for StateTransitionSet {
    #[inline]
    fn set_id(&self) -> SystemSetId {
        SystemSetId::with::<Self>(*self as u64)
    }
}

impl App {
    /// Register a state machine for `S` with `initial` as its starting mode.
    ///
    /// Follows the [deferred-entry model](crate::state): the initial mode is
    /// queued as a [`NextState<S>`](prism_ecs::schedule::NextState) request and
    /// the exclusive
    /// [`apply_state_transition::<S>`](prism_ecs::schedule::apply_state_transition)
    /// system is wired into the [`StateTransition`] phase in
    /// [`StateTransitionSet::Apply`]. On the first frame's `StateTransition`,
    /// `State(initial)` is inserted and `OnEnter(initial)` runs. Thereafter,
    /// queue a transition with
    /// [`NextState::set`](prism_ecs::schedule::NextState::set) and it is applied
    /// on the next `StateTransition`.
    ///
    /// Calling this (or [`init_state`](App::init_state)) more than once for the
    /// same `S` is allowed: the transition system is wired only once, and the
    /// later call simply re-queues the pending initial value.
    pub fn insert_state<S: States>(&mut self, initial: S) -> &mut Self {
        self.insert_resource(NextState::<S>(Some(initial)));
        if self.initialized_states.insert(TypeId::of::<S>()) {
            self.add_systems(
                StateTransition,
                apply_state_transition::<S>.in_set(StateTransitionSet::Apply),
            );
        }
        self
    }

    /// Register a state machine for `S` starting in its [`Default`] mode.
    ///
    /// Convenience over [`insert_state`](App::insert_state) for the common case
    /// where the starting mode is `S::default()` (e.g. a `#[default] MainMenu`
    /// variant). Semantics are otherwise identical, including the
    /// [deferred-entry model](crate::state).
    pub fn init_state<S: States + Default>(&mut self) -> &mut Self {
        self.insert_state(S::default())
    }
}
