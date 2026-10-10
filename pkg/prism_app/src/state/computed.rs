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
//! composite `States` value.
//!
//! # Chaining: computed-of-computed
//!
//! Because [`SourceStates`](ComputedStates::SourceStates) only requires
//! [`States`] and every `ComputedStates` is itself a `States`, a computed state
//! may derive from *another* computed state (`AppState -> Loaded -> Ready`).
//! For the chain to settle in a single frame, the derived state must recompute
//! **after** its source has recomputed — otherwise it would read a stale source
//! value and lag one frame behind.
//!
//! All computed recomputations run in the one
//! [`StateTransitionSet::Compute`] group, so ordering inside it is pinned by a
//! per-chain *dependency depth*: a state derived directly from a base
//! [`States`] value has [`DEPENDENCY_DEPTH`](ComputedStates::DEPENDENCY_DEPTH)
//! `1`; one derived from a depth-`n` computed state declares depth `n + 1`.
//! [`App::add_computed_state`] places each recompute in the [`ComputeDepth`]
//! sub-tier for its depth and orders depth `n` strictly after depth `n - 1`, so
//! sources at every shallower depth are already settled when a deeper state
//! derives from them. The idiomatic way to keep a chain's depths consistent is
//! to compute them from the source rather than hard-code a literal:
//!
//! ```
//! use prism_app::prelude::*;
//! use prism_app::state::ComputedStates;
//!
//! #[derive(Clone, PartialEq, Eq, Hash, Debug, Default)]
//! enum AppState { #[default] Menu, InGame }
//! impl States for AppState {}
//!
//! #[derive(Clone, PartialEq, Eq, Hash, Debug)]
//! struct Loaded; // depth 1: derived from the base AppState.
//! impl States for Loaded {}
//! impl ComputedStates for Loaded {
//!     type SourceStates = AppState;
//!     fn compute(source: &AppState) -> Option<Self> {
//!         matches!(source, AppState::InGame).then_some(Loaded)
//!     }
//! }
//!
//! #[derive(Clone, PartialEq, Eq, Hash, Debug)]
//! struct Ready; // depth 2: derived from the computed Loaded.
//! impl States for Ready {}
//! impl ComputedStates for Ready {
//!     type SourceStates = Loaded;
//!     const DEPENDENCY_DEPTH: usize = <Loaded as ComputedStates>::DEPENDENCY_DEPTH + 1;
//!     fn compute(_source: &Loaded) -> Option<Self> {
//!         Some(Ready)
//!     }
//! }
//!
//! App::new()
//!     .insert_state(AppState::Menu)
//!     .add_computed_state::<Loaded>()
//!     .add_computed_state::<Ready>();
//! ```
//!
//! [`StateTransition`]: crate::schedule::StateTransition

use core::any::TypeId;

use prism_ecs::schedule::{
    IntoSystemConfigs, OnEnter, OnExit, State, States, SystemSet, SystemSetId,
};
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

    /// Position of this state in a computed-of-computed chain, used to order
    /// recomputations within [`StateTransitionSet::Compute`] (see the
    /// [module docs](crate::state::computed#chaining-computed-of-computed)).
    ///
    /// A state derived directly from a base [`States`] value has depth `1`
    /// (the default). A state derived from a depth-`n` computed state must set
    /// this to `n + 1` so its recompute runs after the source has settled;
    /// compute it from the source — `<Source as ComputedStates>::DEPENDENCY_DEPTH
    /// + 1` — rather than hard-coding a literal, so refactoring the chain keeps
    /// the depths consistent. A depth of `0` is clamped to `1`, since every
    /// computed state recomputes at least one tier after the base
    /// [`Apply`](StateTransitionSet::Apply) step.
    const DEPENDENCY_DEPTH: usize = 1;

    /// Compute this state from the current `source` mode.
    ///
    /// Return `Some(value)` when the computed state should exist (with that
    /// value) and `None` when it should not exist at all.
    fn compute(source: &Self::SourceStates) -> Option<Self>
    where
        Self: Sized;
}

/// Ordering sub-tier inside [`StateTransitionSet::Compute`], keyed by a computed
/// state's [`DEPENDENCY_DEPTH`](ComputedStates::DEPENDENCY_DEPTH).
///
/// Computed-of-computed chains all recompute in the single
/// [`StateTransitionSet::Compute`] group, so a derived state could otherwise
/// read a source that has not recomputed yet this frame. [`App::add_computed_state`]
/// places each recompute in the `ComputeDepth` for its depth and orders depth
/// `n` after depth `n - 1`; because the edges are transitive, a depth-`n` state
/// is ordered after *every* shallower depth and therefore sees all of its
/// (possibly transitive) computed sources already settled. Sets at depths with
/// no registered members simply contribute no ordering, so the chain stays
/// correct however sparse the depths are.
///
/// This is a public ordering anchor (like [`StateTransitionSet`]): user systems
/// may order relative to a specific depth, but most code never names it
/// directly — declaring [`DEPENDENCY_DEPTH`](ComputedStates::DEPENDENCY_DEPTH)
/// is enough.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ComputeDepth(pub usize);

impl SystemSet for ComputeDepth {
    #[inline]
    fn set_id(&self) -> SystemSetId {
        SystemSetId::with::<Self>(self.0 as u64)
    }
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
    let current = world
        .get_resource::<State<C>>()
        .map(|state| state.0.clone());

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
    /// mode. The source state must itself be registered — a base state via
    /// [`insert_state`](App::insert_state), or a computed source via its own
    /// `add_computed_state` — otherwise `C::compute` simply never sees a source
    /// and the computed state stays absent.
    ///
    /// For a computed-of-computed chain, the recompute is additionally placed in
    /// the [`ComputeDepth`] sub-tier for `C`'s
    /// [`DEPENDENCY_DEPTH`](ComputedStates::DEPENDENCY_DEPTH) and ordered after
    /// the previous depth, so a derived state settles in the *same* frame as its
    /// source rather than lagging a frame. Registration order between the source
    /// and the derived state does not matter; the depth edges fix the order.
    ///
    /// Idempotent: wiring happens only once per `C`.
    pub fn add_computed_state<C: ComputedStates>(&mut self) -> &mut Self {
        if self.initialized_states.insert(TypeId::of::<State<C>>()) {
            // Depth 0 is meaningless (a computed state always settles at least
            // one tier after `Apply`), so clamp it up to 1.
            let depth = C::DEPENDENCY_DEPTH.max(1);
            let mut config = recompute_computed_state::<C>
                .in_set(StateTransitionSet::Compute)
                .in_set(ComputeDepth(depth))
                .after(StateTransitionSet::Apply);
            if depth > 1 {
                // Run strictly after the previous depth's recomputations so a
                // computed-of-computed source is already settled this frame.
                // The edge is transitive, so this also follows every shallower
                // depth.
                config = config.after(ComputeDepth(depth - 1));
            }
            self.add_systems(StateTransition, config);
        }
        self
    }
}
