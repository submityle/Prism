//! `prism_ui_ecs` — field-level, change-detected bridge between Prism ECS
//! components and Loom reactive [`Signal`](prism_ui_reactive::Signal)s.
//!
//! # Architecture
//!
//! Loom's reactivity and the ECS already both track change. This crate wires
//! them together at *field granularity*, using the ECS changed/lifecycle
//! observers to invalidate only bindings of the affected entities:
//!
//! * A [`FieldBinding`] projects a single field of a component `C` to and from a
//!   `Signal<T>` through a `reader` (and optional `writer`) closure.
//! * **Pulling** (ECS → signal) routes coalesced entity notifications to bound
//!   fields. Quiet frames read no bound values. First attachment and snapshot
//!   restore refresh all bindings once. Explicit `pull_all` uses tick checks.
//! * **Pushing** (signal → ECS) uses an equality guard: the component is only
//!   mutated when the projected field actually differs from the signal, so a
//!   write never advances a change tick needlessly.
//! * [`EntityBinding`] attaches a binding to a concrete entity and erases its
//!   types behind the object-safe [`SyncBinding`] trait; [`EcsBridge`] collects
//!   many such bindings and owns their observer subscriptions. Scheduled pushes
//!   drain value-free signal notifications; clean frames visit no bindings and
//!   neither read nor clone signal values.
//!
//! Because both directions are equality-guarded, an `ECS → signal → ECS` round
//! trip settles rather than oscillating.
//!
//! The [`schedule`] module adapts the bridge to a Prism frame loop: it is
//! installed as a non-send resource and driven by two exclusive systems
//! ([`loom_pull_system`] at frame start, [`loom_push_system`] at frame end).
//!
//! # `std`-only exception
//!
//! Unlike the other Loom crates, this is an **engine-integration bridge** and is
//! deliberately `std`-only: it enables `prism_ecs/std` and uses synchronized
//! observer routes, so there is no `no_std` build to support and it omits the usual
//! `no_std`/`alloc` crate attributes.
//!
//! # Example
//!
//! ```
//! use prism_ecs::prelude::{Component, World};
//! use prism_ui_ecs::EcsBridge;
//! use prism_ui_reactive::Runtime;
//!
//! #[derive(Component)]
//! struct Counter {
//!     value: i32,
//! }
//!
//! let mut world = World::new();
//! let entity = world.spawn(Counter { value: 1 });
//!
//! let rt = Runtime::new();
//! let value = rt.signal(0i32);
//!
//! let mut bridge = EcsBridge::new();
//! bridge.bind_two_way::<Counter, i32>(
//!     entity,
//!     value.clone(),
//!     |c| c.value,
//!     |c, v| c.value = *v,
//! );
//!
//! // Mutate the component, then pull: the signal catches up.
//! world.get_mut::<Counter>(entity).unwrap().value = 42;
//! assert_eq!(bridge.pull_all(&world), 1);
//! assert_eq!(value.get_untracked(), 42);
//!
//! // Mutate the signal, then push: the component is written back.
//! value.set(7);
//! assert_eq!(bridge.push_all(&mut world), 1);
//! assert_eq!(world.get::<Counter>(entity).unwrap().value, 7);
//! ```
#![forbid(unsafe_code)]

pub mod binding;
pub mod bridge;
pub mod entity_binding;
mod invalidation;
pub mod schedule;
pub mod structural;

pub use binding::FieldBinding;
pub use bridge::{BindingId, BridgeWorldMismatch, EcsBridge};
pub use entity_binding::{EntityBinding, SyncBinding};
pub use invalidation::cleanup_bridge_observers;
pub use schedule::{
    add_loom_sync_systems, insert_bridge, loom_pull_system, loom_push_system, remove_bridge,
    LoomSyncSet,
};
pub use structural::{
    ForBinding, ShowBinding, StructuralBinding, StructuralScope, StructuralStats,
};
