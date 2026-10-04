//! Diagnostics: ECS inspector, change-volume accounting, and time-travel
//! (design §16.6).
//!
//! The kernel exposes read-only diagnostic surfaces so an external editor can
//! build an inspector, a performance panel, and a time-travel debugger
//! (design §16.6 对接 `prism_ui_devtools` / `prism_ui_inspector` /
//! `prism_ui_timetravel`). None of these facilities mutate simulation state
//! during capture, so they are safe to drive from a diagnostics system.
//!
//! * [`inspector`] — archetype / chunk occupancy snapshots.
//! * [`memory_footprint`] — byte-level resident-memory accounting:
//!   per-archetype reserved / live / wasted column payload derived from
//!   row width and chunk geometry, ranked most-wasted-first for reclamation
//!   (design §16.6 / §5.3 / §17).
//! * [`archetype_fragmentation`] — archetype-explosion and chunk
//!   internal-fragmentation accounting with a worst-occupancy-first
//!   ranking (design §16.6 / §22 risk #5).
//! * [`blob_reference_health`] — reference-count and orphan census over a
//!   working set of [`BlobHandle`](crate::blob::BlobHandle)s resolved
//!   against a [`BlobStore`](crate::blob::BlobStore): per-handle liveness /
//!   refcount / byte length / working-set reference sites, deduplication
//!   visibility, and resident-but-unreferenced orphan accounting — the leak
//!   signal a content-addressed refcounted arena cannot otherwise raise
//!   (design §16.4 / §16.6).
//! * [`change_tick_health`] — change-detection tick-saturation census
//!   (design §10 / §14): per-archetype oldest added / changed / chunk-version
//!   age relative to the world change tick, with a registry-wide saturation
//!   per-mille and `needs_check` / saturated flags — surfacing columns whose
//!   ticks approach `MAX_CHANGE_AGE` and risk aliasing stale `Changed<T>`
//!   unless `check_change_ticks` clamps them (design §16.6 / §10 / §14).
//! * [`change_volume`] — dirty-chunk and changed/added-cell accounting that
//!   quantifies the design headline "成本 ∝ 变化量".
//! * [`churn_cost`] — the *gross* dual of [`system_cost`]: joins per-system
//!   self time ([`profiler`]) with the *gross* structural churn of
//!   [`structural_churn`] by name, surfacing expensive oscillators — systems
//!   burning time thrashing structure that nets out — that net-change cost
//!   accounting cannot see (design §16.6 / §9 / §10 / §17).
//! * [`component_alignment`] — per-component column layout + SIMD-packing
//!   readiness: element size / align / packed stride / per-element padding,
//!   an alignment histogram, and the SIMD-ready share — the per-type view
//!   neither [`storage_distribution`] (per-bucket `max_align` only) nor
//!   [`component_memory`] (padding-free size model) can give
//!   (design §16.6 / §17 / §7).
//! * [`component_distribution`] — component-first distribution / spread view:
//!   per-component archetype spread, live-instance count, and chunk
//!   footprint, ranked most-spread-first (design §16.6 / §5.3 / §22 risk #5).
//! * [`component_memory`] — component-first byte-level memory: per-type
//!   reserved / live / wasted column payload, ranked heaviest-first; the
//!   byte dual of [`memory_footprint`] (design §16.6 / §5.3 / §17).
//! * [`storage_distribution`] — registration-time classification of every
//!   component type by physical [`StorageType`](crate::component::StorageType):
//!   per-bucket type count, zero-sized / dynamic / drop-glue counts, and the
//!   element size / alignment profile — a glance at the §6 four-state storage
//!   mix (design §16.6 / §6 / §17).
//! * [`frame_profile`] — per-system change-volume ranking folded from a
//!   [`step_inspector`] frame trace (design §16.6 "帧级变更量").
//! * [`profiler`] — nested system-span timing and flame-graph export
//!   (design §16.6 "系统火焰图").
//! * [`step_inspector`] — per-system structural / change-volume observation
//!   while single-stepping a schedule (design §23.4 / §16.6).
//! * [`structural_churn`] — dual, *gross* lens over a [`step_inspector`]
//!   trace: per-system absolute entity / archetype movement so opposing
//!   spawns and despawns add instead of cancelling, surfacing the
//!   oscillation (thrash) that net accounting hides and that command
//!   batching targets (design §16.6 / §9 / §10).
//! * [`system_cost`] — joins per-system self time ([`profiler`]) with
//!   per-system change volume ([`frame_profile`]) by name, flagging
//!   hot-and-churny systems (batching targets) vs. compute-bound
//!   systems (SIMD / owning-group targets) (design §16.6 / §17 / §10).
//! * [`schedule_ambiguity_audit`] — schedule-ambiguity blast-radius and
//!   contention census (design §23.4): folds the detector's ambiguous-pair
//!   list into per-system involvement (hotspot) and per-datum contention
//!   rankings plus a whole-world / component / resource roll-up, pointing
//!   at the highest-leverage ordering fix (design §16.6 / §23.4).
//! * [`relation_cascade`] — cleanup-policy and cascade blast-radius
//!   census (design §23.2): per-kind [`CleanupPolicy`](crate::relation::CleanupPolicy)
//!   metadata joined with the pure cascade planner, giving — for every
//!   entity in a relation edge — how many holders a despawn recursively
//!   deletes, how many edges unlink, and how many trip a panic guard;
//!   the operational dual of [`relation_graph`]'s shape view
//!   (design §16.6 / §23.2 / §13.1).
//! * [`relation_closure`] — transitive-closure amplification census
//!   (design §11 / §16.6): for every relation kind registered as
//!   transitive, how large its reachability closure grows versus
//!   its direct edges, quantifying the cost of not caching the
//!   closure (design §16.6 / §11).
//! * [`relation_degree`] — per-entity relation connectivity census
//!   (design §11 / §16.6): rolls every relation kind up by entity to
//!   surface the aggregate relation hubs (out-degree / in-degree) and
//!   each entity's structural role (pure source / sink / relay), the
//!   entity-centric dual of the per-kind shape views (design §16.6 / §11).
//! * [`relation_graph`] — read-only relation-kind and edge-topology summary
//!   (design §16.6 "关系图谱").
//! * [`relation_cycles`] — per-relation directed-cycle detection over the
//!   relation index, flagging suspicious hierarchy / closure cycles
//!   (design §16.6 / §11 / §23.2).
//! * [`relation_topology`] — per-relation directed-graph shape: longest
//!   chain depth, fan-out / fan-in extremes, and root / sink counts
//!   (design §16.6 / §11 / §22 risk #5).
//! * [`required_closure`] — required-components graph shape (design §16.1):
//!   per-component closure size (insert blast radius) and fan-in (hot
//!   shared dependency), ranked hottest-dependency first (design §16.6 /
//!   §16.1).
//! * [`determinism_readiness`] — per-component determinism / replication
//!   readiness (design §14 / §16.5): classifies every component by its
//!   snapshot (clone) and desync-hash glue into fully-deterministic,
//!   snapshot-only (a desync blind spot — rolled back but invisible to
//!   the state hash), hash-only (rollback gap), or opaque, surfacing the
//!   gaps that break rollback networking (design §16.6 / §14).
//! * [`event_flow_health`] — double-buffered event-queue flow and
//!   reader-backpressure census (design §16.7): live buffer occupancy
//!   versus lifetime throughput, and per reader cursor how far it lags,
//!   whether it is caught up, and whether it is *saturated* (pinned at the
//!   oldest event and about to drop the older half on the next buffer
//!   rotation) — surfacing readers that run hot enough to silently miss
//!   events (design §16.6 / §16.7).
//! * [`hook_coverage`] — component lifecycle-hook coverage (design §12):
//!   which of `on_add` / `on_insert` / `on_replace` / `on_remove` each
//!   component registers, flagging acquire/release asymmetries (acquire
//!   on add with no teardown; teardown only in `on_remove`, which an
//!   in-place overwrite skips) as concrete resource-leak smells
//!   (design §16.6 / §12).
//! * [`owning_group_packing`] — per owning group packing efficiency
//!   (design §6 / §17): packed members vs. tracked entities, the
//!   tracked-but-unpacked overhead between them, and a per-mille packing
//!   density, flagging declared-but-empty or low-density groups that pay
//!   registration / maintenance cost without earning branch-free iteration
//!   (design §16.6 / §6 / §17).
//! * [`gpu_batch_efficiency`] — GPU draw-batch efficiency census
//!   (design §15): per-batch instance fill over a
//!   [`GpuBatchPlan`](crate::gpu_batch::GpuBatchPlan) — indirect-draw count,
//!   mean instances-per-draw (the amortisation factor), singleton-draw waste,
//!   and the largest / smallest batch — surfacing plans that barely batch
//!   (too many unique mesh+material keys) (design §16.6 / §15 / §6;
//!   `gpu_resident` feature).
//! * [`gpu_resident_upload`] — GPU-resident column upload / dirty-block
//!   census (design §15 脏块增量上传): per-column pending (coalesced) upload
//!   bytes, upload amplification (pending over live — `1000` means the whole
//!   column is being re-sent), dirty-span contiguity, and capacity pressure
//!   over a set of [`GpuResidentColumn`](crate::gpu_resident::GpuResidentColumn)s,
//!   plus a registry-wide roll-up — surfacing frames where incremental upload
//!   degenerates into a full re-upload (design §16.6 / §15 / §1;
//!   `gpu_resident` feature).
//! * [`hlod_pyramid`] — World-Partition HLOD proxy-pyramid structure and
//!   shown-proxy census (design §13.1): a pure read of an
//!   [`Hlod`](crate::partition::hlod::Hlod) that reports each layer's draw
//!   band geometry and `extent³` footprint, flags the one convention its
//!   constructor does not enforce — layers that fail to coarsen outward
//!   (`extent` regressing with level) — and buckets the currently shown
//!   proxies by level with their covered source-cell area, plus a
//!   cross-check between the layer table and the shown set
//!   (design §16.6 / §13.1; `partition` feature).
//! * [`lod_schedule_audit`] — LOD schedule band-table well-formedness
//!   audit (design §13.2 / §23.7): a pure read of a
//!   [`LodSchedule`](crate::partition::lod::LodSchedule) that flags the
//!   semantic smells its constructor does not enforce — unreachable bands
//!   whose upper edge does not exceed the nearer band, cadence regressions
//!   (a farther band ticking more often than a nearer one), and quality
//!   regressions (a farther band more precise than a nearer one) — per band
//!   plus schedule-wide roll-ups for a tuning tool or CI gate
//!   (design §16.6 / §13.2 / §23.7; `partition` feature).
//! * [`partition_occupancy`] — world-partition cell occupancy / streaming
//!   census (design §13.1): joins the cell streamer with the entity index
//!   per cell to report per-cell [`CellState`](crate::partition::cell::CellState)
//!   and entity counts, empty-loaded cells (resident but carrying nothing),
//!   plus the two streaming consistency gaps the join exposes — entities in
//!   non-resident cells and entities stranded in untracked cells
//!   (design §16.6 / §13.1 / §17; `partition` feature).
//! * [`dormancy_census`] — dormant-entity id-space census
//!   (design §13.2): reads a [`DormancySet`](crate::partition::dormant::DormancySet)
//!   and reports the *shape* of the dormant population in the entity
//!   id-space — slot-index span and fill density, contiguity (maximal
//!   runs of consecutive indices and the longest run), a bit-width
//!   histogram of slot indices and of generations (recycle depth), and the
//!   pending-wake churn — distinguishing a tightly-pooled, serialise-cheap
//!   block from a scattered, heavily-recycled one
//!   (design §16.6 / §13.2; `partition` feature).
//! * [`floating_origin_precision`] — 64-bit floating-origin rebase
//!   precision budget audit (design §13.3): a power-of-two ring ladder
//!   reporting the exact `f32` ULP available at each distance from the
//!   active origin (flagging the first ring that no longer resolves a
//!   caller-supplied target), an analytic safe radius (`target × 2^23`) in
//!   metres and cells, and an optional per-entity rebase census that
//!   measures each `(cell, local)` sample's rebased magnitude, resolution
//!   and whether its local offset has drifted out of its cell
//!   (design §16.6 / §13.3; `partition` feature).
//! * [`prefab_inheritance`] — prefab / `IsA` inheritance shape
//!   (design §16.3): per-instance chain depth and resolved-component
//!   census (overridden vs. inherited vs. own), per-template fan-in for
//!   hot reused templates, and `IsA` cycle detection — surfacing deep
//!   resolution chains, authoring hubs, and modelling bugs
//!   (design §16.6 / §16.3 / §11).
//! * [`sparse_set_occupancy`] — sparse-set storage occupancy / keyspace-
//!   fragmentation census (design §6 四态存储): the out-of-band counterpart
//!   to the Table-path reports above — per sparse-set component and
//!   world-wide live count, dense payload / bookkeeping bytes, the
//!   sparse-index bytes implied by the current entity-index span, and a
//!   `sparse_overhead_permille` that flags sets which have decayed into
//!   mostly-empty index space (design §16.6 / §6).
//! * [`time_travel`] — frame-indexed world snapshot record/seek (`std` only).
//! * [`weak_reference_health`] — weak cross-cell reference table health
//!   census (design §13.1 / §22 risk 4): resolves every stored generational
//!   weak handle against the live world and reports liveness / prune pressure,
//!   duplicate multiplicity and max fan-in, plus a deterministic per-target
//!   breakdown — surfacing stale handles left after streaming unloads and
//!   tables accreting redundant references (design §16.6 / §13.1;
//!   `partition` feature).
//!
//! Every occupancy and change-volume figure except [`sparse_set_occupancy`]
//! describes the chunked Table-backed storage (design §6); `SparseSet`
//! components are not laid out in archetype chunks, so that module accounts
//! them separately out of the world's sparse-set registry.

pub mod archetype_fragmentation;
pub mod blob_reference_health;
pub mod change_tick_health;
pub mod change_volume;
pub mod churn_cost;
pub mod component_alignment;
pub mod component_distribution;
pub mod component_memory;
pub mod determinism_readiness;
#[cfg(feature = "partition")]
pub mod dormancy_census;
pub mod event_flow_health;
#[cfg(feature = "partition")]
pub mod floating_origin_precision;
pub mod frame_profile;
#[cfg(feature = "gpu_resident")]
pub mod gpu_batch_efficiency;
#[cfg(feature = "gpu_resident")]
pub mod gpu_resident_upload;
#[cfg(feature = "partition")]
pub mod hlod_pyramid;
pub mod hook_coverage;
pub mod inspector;
#[cfg(feature = "partition")]
pub mod lod_schedule_audit;
pub mod memory_footprint;
pub mod owning_group_packing;
#[cfg(feature = "partition")]
pub mod partition_occupancy;
pub mod prefab_inheritance;
pub mod profiler;
pub mod relation_cascade;
pub mod relation_closure;
pub mod relation_degree;
pub mod relation_graph;
pub mod relation_cycles;
pub mod relation_topology;
pub mod required_closure;
pub mod schedule_ambiguity_audit;
pub mod sparse_set_occupancy;
pub mod step_inspector;
pub mod storage_distribution;
pub mod structural_churn;
pub mod system_cost;
#[cfg(feature = "std")]
pub mod time_travel;
#[cfg(feature = "partition")]
pub mod weak_reference_health;

pub use archetype_fragmentation::{ArchetypeFragmentEntry, ArchetypeFragmentationReport};
pub use blob_reference_health::{BlobReferenceEntry, BlobReferenceHealth};
pub use change_tick_health::{ArchetypeTickAgeEntry, TickAgeReport};
pub use change_volume::{ArchetypeChangeReport, ChangeReport};
pub use churn_cost::{ChurnCostEntry, ChurnCostProfile};
pub use component_alignment::{
    ComponentAlignmentBucket, ComponentAlignmentEntry, ComponentAlignmentReport,
};
pub use component_distribution::{ComponentDistributionEntry, ComponentDistributionReport};
pub use component_memory::{ComponentMemoryEntry, ComponentMemoryReport};
pub use determinism_readiness::{
    DeterminismClass, DeterminismReadinessEntry, DeterminismReadinessReport,
};
#[cfg(feature = "partition")]
pub use dormancy_census::{DormancyCensus, GenerationBucketEntry, IndexBucketEntry};
pub use event_flow_health::{EventFlowHealth, EventReaderEntry};
#[cfg(feature = "partition")]
pub use floating_origin_precision::{
    FloatingOriginPrecisionAudit, PrecisionRingEntry, RebaseSampleAudit,
};
pub use frame_profile::{FrameChangeProfile, SystemChangeEntry};
#[cfg(feature = "gpu_resident")]
pub use gpu_batch_efficiency::{BatchEntry, GpuBatchEfficiencyReport};
#[cfg(feature = "gpu_resident")]
pub use gpu_resident_upload::{ColumnUploadEntry, GpuUploadReport};
#[cfg(feature = "partition")]
pub use hlod_pyramid::{HlodLayerAudit, HlodPyramidAudit};
pub use hook_coverage::{HookCoverageEntry, HookCoverageReport};
pub use inspector::{ArchetypeReport, OccupancyStats, WorldReport};
#[cfg(feature = "partition")]
pub use lod_schedule_audit::{LodBandAudit, LodScheduleAudit};
pub use memory_footprint::{ArchetypeMemoryEntry, MemoryFootprintReport};
pub use owning_group_packing::{OwningGroupPackingEntry, OwningGroupPackingReport};
#[cfg(feature = "partition")]
pub use partition_occupancy::{CellOccupancyEntry, PartitionOccupancyReport};
pub use prefab_inheritance::{PrefabInheritanceReport, PrefabInstanceEntry, PrefabTemplateEntry};
pub use relation_cascade::{CascadeBlastEntry, RelationCascadeReport, RelationPolicyEntry};
pub use relation_closure::{RelationClosureReport, TransitiveClosureEntry};
pub use relation_degree::{RelationDegreeEntry, RelationDegreeReport};
pub use relation_graph::{RelationGraphReport, RelationKindReport};
pub use relation_cycles::{RelationCycleEntry, RelationCycleReport};
pub use relation_topology::{RelationTopologyEntry, RelationTopologyReport};
pub use required_closure::{RequiredClosureEntry, RequiredClosureReport};
pub use schedule_ambiguity_audit::{
    AmbiguousSystemEntry, ContendedComponentEntry, ContendedResourceEntry, ScheduleAmbiguityAudit,
};
pub use sparse_set_occupancy::{SparseSetOccupancyEntry, SparseSetOccupancyReport};
pub use step_inspector::{StepObservation, SteppingInspector};
pub use storage_distribution::{StorageBucketEntry, StorageDistributionReport};
pub use structural_churn::{StructuralChurnProfile, SystemChurnEntry};
pub use system_cost::{SystemCostEntry, SystemCostProfile};
#[cfg(feature = "std")]
pub use profiler::SpanRecorder;
pub use profiler::{FlameGraph, SpanNode, SystemInstrument};
#[cfg(feature = "std")]
pub use time_travel::TimeTravel;
#[cfg(feature = "partition")]
pub use weak_reference_health::{WeakReferenceHealth, WeakTargetEntry};
