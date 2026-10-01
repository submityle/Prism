//! `prism_ui_ecs` — field-level, change-detected bridge between Prism ECS
//! components and Loom reactive [`Signal`](prism_ui_reactive::Signal)s.
//!
//! # Architecture
//!
//! Loom's reactivity and the ECS already both track change. This crate wires
//! them together at *field granularity* and reuses the ECS **tick-based change
//! detection** as the transport for reactivity instead of inventing a parallel
//! dirty-flag scheme:
//!
//! * A [`FieldBinding`] projects a single field of a component `C` to and from a
//!   `Signal<T>` through a `reader` (and optional `writer`) closure.
//! * **Pulling** (ECS → signal) consults each component's change tick: a field
//!   is only re-read when the component changed since the binding last observed
//!   it, so idle entities cost nothing.
//! * **Pushing** (signal → ECS) uses an equality guard: the component is only
//!   mutated when the projected field actually differs from the signal, so a
//!   write never advances a change tick needlessly.
//! * [`EntityBinding`] attaches a binding to a concrete entity and erases its
//!   types behind the object-safe [`SyncBinding`] trait; [`EcsBridge`] collects
//!   many such bindings and drives whole-frame `pull_all`/`push_all` passes.
//!
//! Because both directions are equality-guarded, an `ECS → signal → ECS` round
//! trip settles rather than oscillating.
//!
//! # `std`-only exception
//!
//! Unlike the other Loom crates, this is an **engine-integration bridge** and is
//! deliberately `std`-only: it links `bevy_ecs`, which itself requires `std`, so
//! there is no `no_std` build to support and the crate omits the usual
//! `no_std`/`alloc` crate attributes.
//!
//! # Example
//!
//! ```
//! use bevy_ecs::prelude::{Component, World};
//! use prism_ui_ecs::EcsBridge;
//! use prism_ui_reactive::Runtime;
//!
//! #[derive(Component)]
//! struct Counter {
//!     value: i32,
//! }
//!
//! let mut world = World::new();
//! let entity = world.spawn(Counter { value: 1 }).id();
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

pub use binding::FieldBinding;
pub use bridge::EcsBridge;
pub use entity_binding::{EntityBinding, SyncBinding};
