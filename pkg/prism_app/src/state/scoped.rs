//! State-scoped entities: entities that live only while a mode is current (design §11).
//!
//! Tagging an entity with [`StateScoped<S>(mode)`](StateScoped) declares "this
//! entity belongs to `mode`". Once [`State<S>`](State) is no longer that value,
//! the entity is despawned automatically at the next
//! [`StateTransition`]. This removes the
//! hand-written teardown that otherwise leaks "stragglers" across a scene
//! switch — the menu's buttons vanish when the menu does, the level's props
//! vanish when the level does.
//!
//! ```
//! use prism_app::prelude::*;
//! use prism_app::state::StateScoped;
//!
//! #[derive(Clone, PartialEq, Eq, Hash, Debug, Default)]
//! enum AppState { #[default] Menu, InGame }
//! impl States for AppState {}
//!
//! let mut app = App::new();
//! app.insert_state(AppState::Menu)
//!     .enable_state_scoped_entities::<AppState>();
//!
//! // An entity tagged for `Menu` survives while `Menu` is current and is
//! // despawned the first `StateTransition` after the app leaves `Menu`.
//! let _menu_entity = app.world_mut().spawn(StateScoped(AppState::Menu));
//! ```
//!
//! # Semantics
//!
//! Each frame [`clean_up_state_scoped_entities::<S>`] reads the current
//! [`State<S>`](State) and despawns every entity whose
//! [`StateScoped<S>`] tag does **not** equal that value:
//!
//! - mode unchanged → matching entities survive, nothing is despawned.
//! - mode changed to `new` → entities tagged for any value other than `new`
//!   are despawned (including entities tagged for the mode just left).
//! - [`State<S>`](State) not installed yet (the state machine has not run its
//!   first transition) → nothing is despawned; the tags simply wait.
//!
//! The cleanup runs in [`StateTransitionSet::Compute`] after
//! [`StateTransitionSet::Apply`], so it observes the mode the frame just
//! transitioned *into*. An entity spawned with `StateScoped(new)` during the
//! same frame's `Update` is therefore kept; only entities whose owning mode is
//! no longer current are removed.
//!
//! # Scope: flat despawn only
//!
//! The despawn is **flat**: exactly the tagged entities are removed, not their
//! descendants. `prism_ecs` does not yet model an entity hierarchy
//! (parent/child links or recursive despawn), so a scene graph's children are
//! not cascaded here. Hierarchical / recursive state-scoped despawn is deferred
//! until `prism_ecs` grows a hierarchy; it is documented as absent rather than
//! faked. Until then, tag each entity that must be cleaned up, or despawn
//! children explicitly from an `OnExit` system.
//!
//! [`State`]: prism_ecs::schedule::State
//! [`State<S>`]: prism_ecs::schedule::State

use core::any::TypeId;

use prism_ecs::component::Component;
use prism_ecs::entity::Entity;
use prism_ecs::schedule::{IntoSystemConfigs, State, States};
use prism_ecs::world::World;

use crate::app::App;
use crate::schedule::StateTransition;
use crate::state::StateTransitionSet;

/// Marks an entity as owned by a specific [`States`] value `S`.
///
/// While [`State<S>`](State) equals the wrapped value, the entity lives
/// normally. Once the mode changes, [`clean_up_state_scoped_entities::<S>`]
/// despawns it at the next [`StateTransition`].
/// Enable the cleanup for a state type with
/// [`App::enable_state_scoped_entities`]; see the [module docs](crate::state::scoped)
/// for the full lifecycle and the flat-despawn scope limit.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StateScoped<S: States>(pub S);

// A marker component: `StateScoped<S>` carries data but needs no per-component
// storage behaviour beyond the blanket component contract.
impl<S: States> Component for StateScoped<S> {}

/// Exclusive system that despawns every [`StateScoped<S>`] entity whose owning
/// mode is no longer the current [`State<S>`](State).
///
/// Registered by [`App::enable_state_scoped_entities`] into
/// [`StateTransitionSet::Compute`]. Does nothing while [`State<S>`](State) is
/// absent. The despawn is flat (see the [module docs](crate::state::scoped)).
pub fn clean_up_state_scoped_entities<S: States>(world: &mut World) {
    let Some(current) = world
        .get_resource::<State<S>>()
        .map(|state| state.get().clone())
    else {
        // The state machine has not produced a current mode yet; nothing to do.
        return;
    };

    let scoped = world.query_filtered::<(Entity, &StateScoped<S>), ()>();
    let doomed: Vec<Entity> = scoped
        .iter(world)
        .filter(|(_, tag)| tag.0 != current)
        .map(|(entity, _)| entity)
        .collect();

    for entity in doomed {
        world.despawn(entity);
    }
}

impl App {
    /// Enable automatic despawn of [`StateScoped<S>`] entities when `S` changes
    /// (design §11).
    ///
    /// Wires [`clean_up_state_scoped_entities::<S>`] into the
    /// [`StateTransition`] phase in
    /// [`StateTransitionSet::Compute`], ordered after
    /// [`StateTransitionSet::Apply`] so it sees the mode the frame transitioned
    /// into. After this, spawning an entity with a [`StateScoped<S>`] tag ties
    /// its lifetime to that mode. The state type `S` should itself be registered
    /// (e.g. with [`insert_state`](App::insert_state)); until a current
    /// [`State<S>`](prism_ecs::schedule::State) exists, tagged entities are left
    /// untouched.
    ///
    /// Idempotent: wiring happens only once per `S`. The despawn is flat; see
    /// the [module docs](crate::state::scoped) for the hierarchy limitation.
    pub fn enable_state_scoped_entities<S: States>(&mut self) -> &mut Self {
        if self
            .initialized_states
            .insert(TypeId::of::<StateScoped<S>>())
        {
            self.add_systems(
                StateTransition,
                clean_up_state_scoped_entities::<S>
                    .in_set(StateTransitionSet::Compute)
                    .after(StateTransitionSet::Apply),
            );
        }
        self
    }
}
