//! `OnTransition` hooks: schedules keyed to a specific `from -> to` state edge
//! (design §11, §24.8 — *"转换钩子:`OnEnter/OnExit/OnTransition` 系统集,过渡动画/资源加卸载挂此"*).
//!
//! `prism_ecs` ships [`OnEnter`](prism_ecs::schedule::OnEnter) and
//! [`OnExit`](prism_ecs::schedule::OnExit) edges but deliberately leaves the
//! two-sided `OnTransition { from, to }` edge to the App layer, because it is a
//! derived observation of the base machinery rather than part of the minimal
//! transition primitive. This module supplies that edge without modifying
//! `prism_ecs`.
//!
//! # What a transition hook is for
//!
//! `OnEnter(to)` fires on *every* way of entering `to`, and `OnExit(from)` on
//! *every* way of leaving `from`. A transition hook is narrower: it fires only
//! for the one specific edge `from -> to`. That is exactly what cross-fades,
//! directional load/unload, and "coming from the pause menu vs. the main menu"
//! logic need — work that depends on *both* endpoints at once.
//!
//! ```
//! use prism_app::prelude::*;
//! use prism_app::state::OnTransition;
//!
//! #[derive(Clone, PartialEq, Eq, Hash, Debug, Default)]
//! enum Mode { #[default] Menu, Loading, InGame }
//! impl States for Mode {}
//!
//! fn fade_menu_to_loading() { /* start a cross-fade */ }
//!
//! App::new()
//!     .insert_state(Mode::Menu)
//!     .add_state_transition_hooks::<Mode>()
//!     .add_systems(OnTransition { from: Mode::Menu, to: Mode::Loading }, fade_menu_to_loading);
//! ```
//!
//! # Ordering (honest boundary)
//!
//! Because the hook is driven at the App layer rather than from inside
//! [`apply_state_transition`](prism_ecs::schedule::apply_state_transition), the
//! observed order of a single edge is:
//!
//! ```text
//! OnExit(from)  →  OnEnter(to)  →  OnTransition { from, to }
//! ```
//!
//! i.e. the transition hook runs **after** both the exit and enter edges, not
//! *between* them. `prism_ecs` runs `OnExit` then `OnEnter` atomically inside
//! the exclusive apply system, and this crate does not reach inside that to
//! splice a third schedule in the middle (that would mean forking the ECS
//! primitive). Running the hook afterwards keeps the ECS contract intact and is
//! the natural place for "both endpoints are now settled" work. This differs
//! from engines that interleave `OnTransition` between exit and enter; it is
//! documented here rather than hidden.
//!
//! # Which states it observes
//!
//! The driver runs in [`StateTransitionSet::Notify`], ordered strictly after
//! [`StateTransitionSet::Compute`] (and therefore after
//! [`StateTransitionSet::Apply`]). By that point every state kind this crate
//! models has settled its [`State<S>`] for the frame — base states in `Apply`,
//! and [computed](crate::state::computed) / [sub](crate::state::sub) states in
//! `Compute`. So `add_state_transition_hooks::<S>` works uniformly for a base,
//! computed, or sub state `S`.
//!
//! # When it fires
//!
//! Only on a genuine mode-to-mode edge. The driver tracks the last observed
//! [`State<S>`] value and compares it against the current one each frame:
//!
//! - `from -> to` with `from != to` → run `OnTransition { from, to }`.
//! - first entry / appearance (`None -> to`) → **no** hook (there is no `from`).
//! - disappearance (`from -> None`, e.g. a computed/sub state ceasing to exist)
//!   → **no** hook (there is no `to`).
//! - unchanged → nothing.
//!
//! The tracker is still updated on entry and disappearance so the *next* real
//! edge computes the correct `from`.
//!
//! [`StateTransition`]: crate::schedule::StateTransition

use core::any::TypeId;

use prism_ecs::resource::Resource;
use prism_ecs::schedule::{IntoSystemConfigs, State, States};
use prism_ecs::world::World;

use crate::app::App;
use crate::schedule::StateTransition;
use crate::state::StateTransitionSet;

/// A [`ScheduleLabel`](prism_ecs::schedule::ScheduleLabel) run on the specific
/// `from -> to` edge of a [`States`] type `S`.
///
/// Register systems on it with
/// [`add_systems(OnTransition { from, to }, ..)`](App::add_systems) and enable
/// the driver once with
/// [`add_state_transition_hooks::<S>`](App::add_state_transition_hooks). See the
/// [module docs](crate::state::transition) for ordering and firing rules.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OnTransition<S: States> {
    /// The mode being left.
    pub from: S,
    /// The mode being entered.
    pub to: S,
}

/// Tracks the last [`State<S>`] value observed by the transition driver, so the
/// next frame can tell whether a `from -> to` edge occurred and what `from`
/// was.
///
/// Installed per observed state type `S` by
/// [`add_state_transition_hooks`](App::add_state_transition_hooks); not part of
/// the public API surface beyond that wiring.
struct LastObservedState<S: States>(Option<S>);

impl<S: States> Resource for LastObservedState<S> {}

/// Exclusive system that detects a `from -> to` edge for `S` and runs the
/// matching [`OnTransition`] schedule.
///
/// Registered by [`App::add_state_transition_hooks`] into
/// [`StateTransitionSet::Notify`]. See the [module docs](crate::state::transition)
/// for the firing table and ordering.
pub fn run_transition_hooks<S: States>(world: &mut World) {
    let current = world
        .get_resource::<State<S>>()
        .map(|state| state.0.clone());
    let last = world
        .get_resource::<LastObservedState<S>>()
        .and_then(|tracked| tracked.0.clone());

    if last == current {
        // No change this frame (covers "absent before and after" too).
        return;
    }

    // A genuine mode-to-mode edge runs the hook; first-entry (None -> to) and
    // disappearance (from -> None) update the tracker but run no hook, because
    // one endpoint of the edge is missing.
    if let (Some(from), Some(to)) = (last, current.clone()) {
        world.run_schedule(OnTransition { from, to });
    }

    world.insert_resource(LastObservedState(current));
}

impl App {
    /// Enable [`OnTransition`] hooks for the state type `S` (design §11, §24.8).
    ///
    /// Wires [`run_transition_hooks::<S>`] into the [`StateTransition`] phase in
    /// [`StateTransitionSet::Notify`], ordered after
    /// [`StateTransitionSet::Compute`] so it observes the frame's fully settled
    /// [`State<S>`] (base, computed, or sub). After calling this, register
    /// per-edge systems with
    /// [`add_systems(OnTransition { from, to }, ..)`](App::add_systems).
    ///
    /// This is opt-in and off by default (design §3: advanced capabilities cost
    /// nothing until enabled). An edge with no registered systems is a cheap
    /// no-op [`run_schedule`](prism_ecs::world::World::run_schedule). The state
    /// type `S` must itself be registered (e.g. with
    /// [`insert_state`](App::insert_state),
    /// [`add_computed_state`](App::add_computed_state), or
    /// [`add_sub_state`](App::add_sub_state)); otherwise no `State<S>` ever
    /// settles and no edge is ever observed.
    ///
    /// Idempotent: wiring happens only once per `S`.
    pub fn add_state_transition_hooks<S: States>(&mut self) -> &mut Self {
        if self
            .initialized_states
            .insert(TypeId::of::<OnTransition<S>>())
        {
            self.add_systems(
                StateTransition,
                run_transition_hooks::<S>
                    .in_set(StateTransitionSet::Notify)
                    .after(StateTransitionSet::Compute),
            );
        }
        self
    }
}
