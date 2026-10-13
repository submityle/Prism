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
//! The design doc's roadmap (§21) is **M0–M5 plus the §23 additions**: all of it
//! is implemented here, with tests, and there are no `todo!()`,
//! `unimplemented!()`, or hollow stubs on the public surface.
//!
//! - **M0 core** — generational [`Entity`] + allocator, the [`Component`]
//!   registry, the archetype graph with columnar Structure-of-Arrays
//!   [storage](crate::storage), [`World`](crate::world::World)
//!   spawn/get/insert/remove/despawn (plus `prewarm` for first-frame bursts),
//!   basic [queries](crate::query), and deferred
//!   [`Commands`](crate::command::Commands).
//! - **M1 scheduling** — [`System`](crate::system::System) param derivation, the
//!   conflict-graph executor, resources, events, run conditions and
//!   [`SystemSet`](crate::schedule::SystemSet)s.
//! - **M2 performance core** — double-layer change detection with the dirty
//!   [chunk](crate::storage::ChunkVersions) accessor, sparse sets, SIMD column
//!   kernels, owning groups, the query match cache, and ordered command replay.
//! - **M3 jobs** — the [fiber job graph](crate::system::JobGraph) and
//!   system-internal chunk parallelism on [`prism_tasks`].
//! - **M4 relations and reactivity** — fragmenting relations with transitive
//!   queries, the canonical [`ChildOf`](crate::hierarchy::ChildOf) hierarchy,
//!   [observers](crate::observer) and hooks, the push reaction graph, required
//!   components, prefabs, and blobs.
//! - **M5 scale / determinism / GPU** — world-partition streaming with entity
//!   LOD and a 64-bit floating origin, snapshots with rollback, the
//!   `determinism`/`trace` diagnostics, and GPU-resident columns.
//!
//! Still **PLANNED** (and recorded as such in the design doc): M6 migration and
//! tooling, physical 16 KiB chunk arenas, large-page / NUMA-aware column
//! allocation, the render side of the §23.5 extraction pipeline, and the
//! `ParamSet` system param.
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
pub mod change_observers;
#[cfg(test)]
mod change_observers_tests;
pub mod command;
pub mod component;
pub mod component_hooks;
#[cfg(test)]
mod component_hooks_tests;
pub mod culling;
pub mod diagnostics;
pub mod entity;
pub mod event;
pub mod extract;
#[cfg(test)]
mod extract_tests;
#[cfg(feature = "gpu_resident")]
pub mod gpu_batch;
#[cfg(feature = "gpu_resident")]
pub mod gpu_resident;
pub mod hierarchy;
#[cfg(test)]
mod hierarchy_tests;
pub mod observer;
#[cfg(feature = "partition")]
pub mod partition;
pub mod prefab;
pub mod query;
pub mod reaction;
#[cfg(feature = "serialize")]
pub mod reflect_archive;
#[cfg(all(test, feature = "serialize"))]
mod reflect_archive_tests;
#[cfg(feature = "reflect")]
pub mod reflect_bridge;
#[cfg(all(test, feature = "reflect"))]
mod reflect_bridge_tests;
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
    pub use crate::change_observers::ChangeContext;
    pub use crate::command::{ChildBuilder, CommandQueue, Commands};
    pub use crate::component::Component;
    pub use crate::component_hooks::{ComponentHook, ComponentHooks, HookContext};
    pub use crate::culling::{ChunkCullStats, Frustum, Plane};
    pub use crate::entity::Entity;
    pub use crate::event::{Event, EventCursor, EventId, Events};
    pub use crate::extract::{
        extract_components, extract_resource, ExtractComponent, ExtractResource,
    };
    #[cfg(feature = "gpu_resident")]
    pub use crate::gpu_batch::{
        build_draw_batches, DrawBatch, GpuBatchBuilder, GpuBatchPlan, InstanceVisibility,
    };
    #[cfg(feature = "gpu_resident")]
    pub use crate::gpu_resident::{
        DirtyBlock, GpuReallocation, GpuResidentColumn, GpuResidentColumns, GpuUpload,
    };
    pub use crate::hierarchy::ChildOf;
    pub use crate::observer::{
        EventContext, LifecycleEvent, ObserverContext, ObserverId, Observers,
    };
    #[cfg(feature = "partition")]
    pub use crate::partition::cell::{
        CellCoord, CellState, CellStreamer, StreamingDelta, WorldPartitionCell,
    };
    #[cfg(feature = "partition")]
    pub use crate::partition::data_layer::{DataLayerId, DataLayerState, DataLayers};
    #[cfg(feature = "partition")]
    pub use crate::partition::dormant::{DormancySet, Dormant};
    #[cfg(feature = "partition")]
    pub use crate::partition::driver::{StreamDriver, StreamResult};
    #[cfg(feature = "partition")]
    pub use crate::partition::floating_origin::{FloatingOrigin, GridCell, LocalPos, WorldPos};
    #[cfg(feature = "partition")]
    pub use crate::partition::hlod::{Hlod, HlodDelta, HlodFootprint, HlodLayer, HlodProxyId};
    #[cfg(feature = "partition")]
    pub use crate::partition::interest::InterestGrid;
    #[cfg(feature = "partition")]
    pub use crate::partition::lod::{
        distance_sq, LodBand, LodDecision, LodLevel, LodQuality, LodSchedule, OutOfRange,
    };
    #[cfg(feature = "partition")]
    pub use crate::partition::processor::{EntityLodProcessor, LodTickResult, PhasePolicy};
    #[cfg(feature = "partition")]
    pub use crate::partition::streaming::{CellEntityIndex, WeakEntity, WeakRefs};
    #[cfg(feature = "partition")]
    pub use crate::partition::view::OriginView;
    #[cfg(feature = "partition")]
    pub use crate::partition::world_partition::WorldPartition;
    pub use crate::prefab::IsA;
    pub use crate::query::{
        Added, AnyOf, Changed, Has, Or, QueryEntityError, QuerySingleError, With, Without,
    };
    pub use crate::reaction::{NodeId, ReactionGraph};
    pub use crate::relation::{
        CascadeEdge, CascadePlan, CleanupPolicy, Pair, PairKey, Relation, RelationId,
        RelationIndex, RelationKind, RelationTarget, Relations, TargetId,
    };
    pub use crate::resource::{FromWorld, Resource, ResourceId, Resources};
    pub use crate::schedule::{
        apply_state_transition, in_state, resource_equals, resource_exists, run_once,
        ContinueReport, ContinueStop, IntoSystemConfigs, IntoSystemConfigs as IntoScheduleConfigs,
        NextState, OnEnter, OnExit, Phase, Schedule, ScheduleLabel, Schedules, SetConfig, State,
        States, StepReport, Stepping, SystemConfigs, SystemLane, SystemSet,
    };
    pub use crate::system::{IntoSystem, Local, Query, Res, ResMut, System, SystemParam};
    #[cfg(feature = "determinism")]
    pub use crate::world::snapshot::{
        locate_divergence, FrameHash, FrameHashLog, SnapshotDivergence, TickDivergence,
    };
    pub use crate::world::snapshot::{FnvHasher, SnapshotDelta, SnapshotRing, WorldSnapshot};
    pub use crate::world::{EntityRef, World, WorldId};
    // Derive macros. These live in the macro namespace and coexist with the
    // same-named traits (`Component`, `Bundle`) re-exported above.
    pub use prism_ecs_macros::{
        Bundle, Component, Event, Relation, Resource, SystemParam, SystemSet,
    };
}

/// Internal type aliases for the hash maps/sets used throughout the kernel.
///
/// Centralised so the hasher can be swapped (e.g. for a deterministic hasher
/// under the future `determinism` feature) in exactly one place.
pub(crate) mod collections {
    /// A `no_std`-friendly hash map backed by `hashbrown`.
    pub type HashMap<K, V> = hashbrown::HashMap<K, V>;
}
