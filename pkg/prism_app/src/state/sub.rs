//! Sub-states: modes that only exist while a parent mode is active (design §11).
//!
//! A sub-state is a nested state machine scoped to a parent [`States`] value —
//! e.g. `Combat` / `Explore` only make sense while the app is `InGame`. Unlike
//! a [computed state](crate::state::computed), a sub-state has its own
//! [`NextState<S>`] queue: while it exists, gameplay transitions it freely; when
//! the parent leaves, the whole sub-machine exits and disappears.
//!
//! ```
//! use prism_app::prelude::*;
//! use prism_app::state::SubStates;
//!
//! #[derive(Clone, PartialEq, Eq, Hash, Debug, Default)]
//! enum AppState { #[default] Menu, InGame }
//! impl States for AppState {}
//!
//! #[derive(Clone, PartialEq, Eq, Hash, Debug)]
//! enum InGameMode { Explore, Combat }
//! impl States for InGameMode {}
//! impl SubStates for InGameMode {
//!     type SourceStates = AppState;
//!     fn should_exist(source: &AppState) -> Option<Self> {
//!         matches!(source, AppState::InGame).then_some(InGameMode::Explore)
//!     }
//! }
//!
//! App::new()
//!     .insert_state(AppState::Menu)
//!     .add_sub_state::<InGameMode>();
//! ```
//!
//! # Semantics
//!
//! Each frame [`apply_sub_state::<S>`] reads the parent
//! [`State<S::SourceStates>`](State) and calls
//! [`S::should_exist`](SubStates::should_exist):
//!
//! - parent says **exist** and the sub-state is **absent** → it *activates*:
//!   enter the parent-provided initial value (or a value already queued in
//!   [`NextState<S>`] if gameplay pre-queued one), running `OnEnter`.
//! - parent says **exist** and the sub-state is **present** → consume any
//!   queued [`NextState<S>`] like a base state: on a real change run
//!   `OnExit(old)` then `OnEnter(new)`.
//! - parent says **not exist** → the sub-state *deactivates*: run `OnExit(old)`,
//!   remove the [`State<S>`] resource, and clear any stale queued request.
//!
//! The activation initial comes from `should_exist`; its value is otherwise
//! ignored while the sub-state is already live (gameplay drives it through
//! [`NextState<S>`]).
//!
//! # Chaining on a computed or sub source
//!
//! A sub-state's parent is any [`States`] value, so it may be gated by a
//! [computed state](crate::state::computed) or by another sub-state rather than
//! a base state. Because every state settles inside the one
//! [`StateTransitionSet::Compute`] group, such a sub-state must apply *after*
//! its source has recomputed this frame — otherwise it reads a stale parent and
//! lags a frame behind. [`App::add_sub_state`] reuses the same
//! [`ComputeDepth`] ordering as computed states:
//! declare [`DEPENDENCY_DEPTH`](SubStates::DEPENDENCY_DEPTH) as the source's
//! depth plus one and the apply is pinned after every shallower depth, so the
//! whole chain (base → computed → sub, or sub → sub) settles in a single frame.
//!
//! [`NextState<S>`]: prism_ecs::schedule::NextState

use core::any::TypeId;

use prism_ecs::schedule::{IntoSystemConfigs, NextState, OnEnter, OnExit, State, States};
use prism_ecs::world::World;

use crate::app::App;
use crate::schedule::StateTransition;
use crate::state::computed::ComputeDepth;
use crate::state::StateTransitionSet;

/// A [`States`] value that only exists while a parent mode is active.
///
/// Implement this on a type that also implements [`States`]. The engine drives
/// its existence and transitions in the [`StateTransition`]
/// phase after the parent has settled; see the [module docs](crate::state::sub)
/// for the lifecycle.
pub trait SubStates: States {
    /// The parent state whose value gates this sub-state's existence.
    type SourceStates: States;

    /// Position of this sub-state in a derivation chain, used to order
    /// [`apply_sub_state`] within [`StateTransitionSet::Compute`] relative to
    /// any computed/sub source it reads (see the [module docs](crate::state::sub#chaining-on-a-computed-or-sub-source)).
    ///
    /// A sub-state gated by a base [`States`] value has depth `1` (the default):
    /// its parent is already settled by [`StateTransitionSet::Apply`]. A
    /// sub-state gated by a *computed* state or *another* sub-state at depth `n`
    /// must set this to `n + 1` so it runs after the source has recomputed this
    /// frame instead of reading a stale value and lagging a frame; compute it
    /// from the source — `<Source as ComputedStates>::DEPENDENCY_DEPTH + 1` (or
    /// the `SubStates` equivalent) — rather than hard-coding a literal. A depth
    /// of `0` is clamped to `1`, since a sub-state always settles at least one
    /// tier after [`Apply`](StateTransitionSet::Apply).
    const DEPENDENCY_DEPTH: usize = 1;

    /// Whether the sub-state should exist for the given parent `source`.
    ///
    /// Return `Some(initial)` to request that the sub-machine exist (entering
    /// `initial` when it first activates) and `None` to request that it not
    /// exist at all.
    fn should_exist(source: &Self::SourceStates) -> Option<Self>
    where
        Self: Sized;
}

/// Exclusive system that drives one [`SubStates`] type `S`: activation,
/// deactivation, and queued transitions while active.
///
/// Registered by [`App::add_sub_state`] into [`StateTransitionSet::Compute`].
/// See the [module docs](crate::state::sub) for the transition table.
pub fn apply_sub_state<S: SubStates>(world: &mut World) {
    let desired = world
        .get_resource::<State<S::SourceStates>>()
        .and_then(|source| S::should_exist(source.get()));
    let current = world
        .get_resource::<State<S>>()
        .map(|state| state.0.clone());

    match (current, desired) {
        (None, None) => {
            // Inactive and should stay inactive: drop any stale queued request
            // so it cannot leak into a future activation.
            clear_pending::<S>(world);
        }
        (Some(old), None) => {
            // Parent left: exit the whole sub-machine and remove it.
            world.run_schedule(OnExit(old));
            world.remove_resource::<State<S>>();
            clear_pending::<S>(world);
        }
        (None, Some(initial)) => {
            // Just activated: enter the parent-provided initial, unless gameplay
            // already pre-queued a specific entry value.
            let entering = take_pending::<S>(world).unwrap_or(initial);
            world.insert_resource(State::new(entering.clone()));
            world.run_schedule(OnEnter(entering));
        }
        (Some(current), Some(_initial)) => {
            // Already active: apply one queued transition, if any, like a base
            // state. A redundant (equal) request is consumed without edges.
            if let Some(next) = take_pending::<S>(world)
                && next != current
            {
                world.run_schedule(OnExit(current));
                world.insert_resource(State::new(next.clone()));
                world.run_schedule(OnEnter(next));
            }
        }
    }
}

/// Take and clear any pending [`NextState<S>`] request.
fn take_pending<S: SubStates>(world: &mut World) -> Option<S> {
    world
        .get_resource_mut::<NextState<S>>()
        .and_then(|next| next.0.take())
}

/// Clear any pending [`NextState<S>`] request without reading it.
fn clear_pending<S: SubStates>(world: &mut World) {
    if let Some(next) = world.get_resource_mut::<NextState<S>>() {
        next.0 = None;
    }
}

impl App {
    /// Register a [`SubStates`] type `S`, gated by its parent
    /// [`SourceStates`](SubStates::SourceStates) (design §11).
    ///
    /// Installs an empty [`NextState<S>`](prism_ecs::schedule::NextState) queue
    /// and wires [`apply_sub_state::<S>`] into the
    /// [`StateTransition`] phase in
    /// [`StateTransitionSet::Compute`], ordered after
    /// [`StateTransitionSet::Apply`] so it reads the frame's settled parent
    /// mode. The parent state must itself be registered — a base state via
    /// [`insert_state`](App::insert_state), or a computed/sub parent via its own
    /// `add_computed_state`/`add_sub_state`.
    ///
    /// When the parent is itself a computed state or another sub-state, the
    /// apply is additionally placed in the [`ComputeDepth`] sub-tier for `S`'s
    /// [`DEPENDENCY_DEPTH`](SubStates::DEPENDENCY_DEPTH) and ordered after the
    /// previous depth. This shares one depth ordering with computed states, so a
    /// sub-state gated by a depth-`n` source settles in the *same* frame as its
    /// source rather than lagging. Registration order between the source and the
    /// sub-state does not matter; the depth edges fix the order.
    ///
    /// Idempotent: wiring and the queue install happen only once per `S`, so a
    /// repeated call never clobbers an in-flight queued transition.
    pub fn add_sub_state<S: SubStates>(&mut self) -> &mut Self {
        if self.initialized_states.insert(TypeId::of::<NextState<S>>()) {
            self.insert_resource(NextState::<S>(None));
            // Depth 0 is meaningless (a sub-state always settles at least one
            // tier after `Apply`), so clamp it up to 1.
            let depth = S::DEPENDENCY_DEPTH.max(1);
            let mut config = apply_sub_state::<S>
                .in_set(StateTransitionSet::Compute)
                .in_set(ComputeDepth(depth))
                .after(StateTransitionSet::Apply);
            if depth > 1 {
                // Run strictly after the previous depth's recomputations so a
                // computed/sub parent is already settled this frame. The edge is
                // transitive, so this also follows every shallower depth.
                config = config.after(ComputeDepth(depth - 1));
            }
            self.add_systems(StateTransition, config);
        }
        self
    }
}
