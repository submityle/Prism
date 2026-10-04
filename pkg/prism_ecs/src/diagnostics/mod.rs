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
//! * [`change_volume`] — dirty-chunk and changed/added-cell accounting that
//!   quantifies the design headline "成本 ∝ 变化量".
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
//! * [`relation_graph`] — read-only relation-kind and edge-topology summary
//!   (design §16.6 "关系图谱").
//! * [`relation_cycles`] — per-relation directed-cycle detection over the
//!   relation index, flagging suspicious hierarchy / closure cycles
//!   (design §16.6 / §11 / §23.2).
//! * [`relation_topology`] — per-relation directed-graph shape: longest
//!   chain depth, fan-out / fan-in extremes, and root / sink counts
//!   (design §16.6 / §11 / §22 risk #5).
//! * [`time_travel`] — frame-indexed world snapshot record/seek (`std` only).
//!
//! All occupancy and change-volume figures describe the chunked Table-backed
//! storage (design §6); `SparseSet` components are not laid out in archetype
//! chunks and are therefore outside these reports.

pub mod archetype_fragmentation;
pub mod change_volume;
pub mod component_distribution;
pub mod component_memory;
pub mod frame_profile;
pub mod inspector;
pub mod memory_footprint;
pub mod profiler;
pub mod relation_graph;
pub mod relation_cycles;
pub mod relation_topology;
pub mod step_inspector;
pub mod storage_distribution;
pub mod structural_churn;
pub mod system_cost;
#[cfg(feature = "std")]
pub mod time_travel;

pub use archetype_fragmentation::{ArchetypeFragmentEntry, ArchetypeFragmentationReport};
pub use change_volume::{ArchetypeChangeReport, ChangeReport};
pub use component_distribution::{ComponentDistributionEntry, ComponentDistributionReport};
pub use component_memory::{ComponentMemoryEntry, ComponentMemoryReport};
pub use frame_profile::{FrameChangeProfile, SystemChangeEntry};
pub use inspector::{ArchetypeReport, OccupancyStats, WorldReport};
pub use memory_footprint::{ArchetypeMemoryEntry, MemoryFootprintReport};
pub use relation_graph::{RelationGraphReport, RelationKindReport};
pub use relation_cycles::{RelationCycleEntry, RelationCycleReport};
pub use relation_topology::{RelationTopologyEntry, RelationTopologyReport};
pub use step_inspector::{StepObservation, SteppingInspector};
pub use storage_distribution::{StorageBucketEntry, StorageDistributionReport};
pub use structural_churn::{StructuralChurnProfile, SystemChurnEntry};
pub use system_cost::{SystemCostEntry, SystemCostProfile};
#[cfg(feature = "std")]
pub use profiler::SpanRecorder;
pub use profiler::{FlameGraph, SpanNode, SystemInstrument};
#[cfg(feature = "std")]
pub use time_travel::TimeTravel;
