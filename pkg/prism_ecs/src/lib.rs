//! `prism_ecs` — Prism's next-generation, data-oriented Entity-Component-System
//! kernel.
//!
//! This is a **greenfield, chunked-archetype** ECS designed to be the
//! authoritative simulation core for the Prism engine after it leaves Bevy.
//! The architecture, milestones, and invariants are specified in
//! `docs/prism_ecs_design_zh.md`.
//!
//! # Milestone status
//!
//! This crate is built up in milestones (see the design doc, §21):
//!
//! - **M0 (this module set)** — generational [`Entity`] + allocator, the
//!   [`Component`] registry, the archetype graph with columnar
//!   Structure-of-Arrays [storage](crate::storage), ergonomic
//!   [`World`](crate::world::World) spawn/get/insert/remove/despawn, basic
//!   [queries](crate::query) with `With`/`Without`/`Option` filters, and a
//!   deferred [`Commands`](crate::command::Commands) buffer.
//!
//! Everything here is a **real, working implementation** — there are no
//! `todo!()`, `unimplemented!()`, or hollow stubs on the public M0 surface.
//! Later milestones (chunk change-versions, SparseSet storage, SIMD iteration,
//! the fiber job graph, relations, reactivity, world partitioning,
//! determinism/rollback, and GPU-resident columns) layer on top of these
//! foundations without rewriting them.
//!
//! # `no_std`
//!
//! The core kernel is `no_std + alloc`. The `std` feature (on by default) only
//! lights up std-backed test and future executor facilities.
//!
//! # Provenance
//!
//! This crate is engine-agnostic and contains **no Unreal Engine source or
//! derived code**, and depends on **no `bevy_*` crate**. All data structures
//! (generational-index allocators, type-erased columnar storage, archetype
//! graphs) are implemented from standard, publicly documented data-structure
//! and ECS knowledge.

#![no_std]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

extern crate alloc;

#[cfg(any(feature = "std", test))]
extern crate std;

pub mod archetype;
pub mod bundle;
pub mod command;
pub mod component;
pub mod entity;
pub mod event;
pub mod query;
pub mod resource;
pub mod schedule;
pub mod storage;
pub mod system;
pub mod world;

/// Commonly used exports. Mirrors the ergonomics of `bevy_ecs::prelude` to keep
/// the eventual migration a near "change-the-import" exercise.
pub mod prelude {
    pub use crate::bundle::Bundle;
    pub use crate::command::{CommandQueue, Commands};
    pub use crate::component::Component;
    pub use crate::entity::Entity;
    pub use crate::event::{Event, EventCursor, EventId, Events};
    pub use crate::resource::{Resource, ResourceId, Resources};
    pub use crate::query::{With, Without};
    pub use crate::schedule::{
        resource_equals, resource_exists, run_once, IntoSystemConfigs, Schedule, SystemConfigs,
    };
    pub use crate::system::{
        IntoSystem, Local, Query, Res, ResMut, System, SystemParam,
    };
    pub use crate::world::World;
}

/// Internal type aliases for the hash maps/sets used throughout the kernel.
///
/// Centralised so the hasher can be swapped (e.g. for a deterministic hasher
/// under the future `determinism` feature) in exactly one place.
pub(crate) mod collections {
    /// A `no_std`-friendly hash map backed by `hashbrown`.
    pub type HashMap<K, V> = hashbrown::HashMap<K, V>;
}
