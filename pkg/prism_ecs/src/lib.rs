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
pub mod blob;
pub mod bundle;
pub mod change;
pub mod command;
pub mod component;
pub mod component_hooks;
#[cfg(test)]
mod component_hooks_tests;
pub mod diagnostics;
pub mod entity;
pub mod event;
pub mod gpu_resident;
pub mod observer;
pub mod partition;
pub mod prefab;
pub mod query;
pub mod reaction;
pub mod relation;
pub mod resource;
pub mod schedule;
pub mod storage;
pub mod system;
pub mod world;

/// Commonly used exports. Mirrors the ergonomics of `bevy_ecs::prelude` to keep
/// the eventual migration a near "change-the-import" exercise.
pub mod prelude {
    pub use crate::blob::{BlobHandle, BlobStore};
    pub use crate::bundle::Bundle;
    pub use crate::change::{ComponentTicks, DetectChanges, DetectChangesMut, Mut, Ref, Tick};
    pub use crate::command::{CommandQueue, Commands};
    pub use crate::component::Component;
    pub use crate::component_hooks::{ComponentHook, ComponentHooks, HookContext};
    pub use crate::entity::Entity;
    pub use crate::event::{Event, EventCursor, EventId, Events};
    pub use crate::gpu_resident::{DirtyBlock, GpuResidentColumn, GpuResidentColumns};
    pub use crate::observer::{
        EventContext, LifecycleEvent, ObserverContext, ObserverId, Observers,
    };
    pub use crate::partition::cell::{
        CellCoord, CellState, CellStreamer, StreamingDelta, WorldPartitionCell,
    };
    pub use crate::partition::dormant::{DormancySet, Dormant};
    pub use crate::partition::floating_origin::{FloatingOrigin, GridCell, LocalPos, WorldPos};
    pub use crate::partition::lod::{
        distance_sq, LodBand, LodDecision, LodLevel, LodSchedule, OutOfRange,
    };
    pub use crate::prefab::IsA;
    pub use crate::query::{Added, Changed, Or, With, Without};
    pub use crate::reaction::{NodeId, ReactionGraph};
    pub use crate::relation::{
        CascadeEdge, CascadePlan, CleanupPolicy, Pair, PairKey, RelationId, RelationIndex,
        RelationKind, RelationTarget, Relations, TargetId,
    };
    pub use crate::resource::{Resource, ResourceId, Resources};
    pub use crate::schedule::{
        apply_state_transition, in_state, resource_equals, resource_exists, run_once,
        IntoSystemConfigs, IntoSystemConfigs as IntoScheduleConfigs, NextState, OnEnter, OnExit,
        Phase, Schedule, ScheduleLabel, Schedules, SetConfig, State, States, SystemConfigs,
        SystemSet,
    };
    pub use crate::system::{IntoSystem, Local, Query, Res, ResMut, System, SystemParam};
    pub use crate::world::snapshot::{FnvHasher, SnapshotDelta, SnapshotRing, WorldSnapshot};
    pub use crate::world::{EntityRef, World};
    // Derive macros. These live in the macro namespace and coexist with the
    // same-named traits (`Component`, `Bundle`) re-exported above.
    pub use prism_ecs_macros::{Bundle, Component, SystemSet};
}

/// Internal type aliases for the hash maps/sets used throughout the kernel.
///
/// Centralised so the hasher can be swapped (e.g. for a deterministic hasher
/// under the future `determinism` feature) in exactly one place.
pub(crate) mod collections {
    /// A `no_std`-friendly hash map backed by `hashbrown`.
    pub type HashMap<K, V> = hashbrown::HashMap<K, V>;
}
