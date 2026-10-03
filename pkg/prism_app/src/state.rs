//! Finite state-machine assembly on the [`App`] (design §11).
//!
//! A [`States`] type (usually a small `enum`) names the discrete modes a world
//! can be in — `MainMenu`, `Loading`, `InGame`, `Paused`. This module adds the
//! two assembly entry points that wire such a type into an app:
//!
//! - [`App::insert_state`] installs an explicit initial mode.
//! - [`App::init_state`] installs the [`Default`] mode.
//!
//! The underlying machinery ([`State`], [`NextState`], [`OnEnter`], [`OnExit`],
//! [`in_state`], [`apply_state_transition`]) lives in [`prism_ecs`] and is
//! re-exported through the crate prelude; this module only provides the App-side
//! wiring and the fixed point in the frame order where transitions are applied.
//!
//! # Deferred-entry model
//!
//! Inserting a state does **not** eagerly create the [`State<S>`] resource or
//! run `OnEnter(initial)`. Instead it:
//!
//! 1. inserts [`NextState<S>`] holding the initial mode as a *pending* request,
//!    and
//! 2. registers the exclusive [`apply_state_transition::<S>`] system into the
//!    [`StateTransition`] phase (design §7: it runs each frame between
//!    `PreUpdate` and `Update`).
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

use prism_ecs::schedule::{apply_state_transition, NextState, States};

use crate::app::App;
use crate::schedule::StateTransition;

impl App {
    /// Register a state machine for `S` with `initial` as its starting mode.
    ///
    /// Follows the [deferred-entry model](crate::state): the initial mode is
    /// queued as a [`NextState<S>`](prism_ecs::schedule::NextState) request and
    /// the exclusive
    /// [`apply_state_transition::<S>`](prism_ecs::schedule::apply_state_transition)
    /// system is wired into the [`StateTransition`] phase. On the first frame's
    /// `StateTransition`, `State(initial)` is inserted and `OnEnter(initial)`
    /// runs. Thereafter, queue a transition with
    /// [`NextState::set`](prism_ecs::schedule::NextState::set) and it is applied
    /// on the next `StateTransition`.
    ///
    /// Calling this (or [`init_state`](App::init_state)) more than once for the
    /// same `S` is allowed: the transition system is wired only once, and the
    /// later call simply re-queues the pending initial value.
    pub fn insert_state<S: States>(&mut self, initial: S) -> &mut Self {
        self.insert_resource(NextState::<S>(Some(initial)));
        if self.initialized_states.insert(TypeId::of::<S>()) {
            self.add_systems(StateTransition, apply_state_transition::<S>);
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
