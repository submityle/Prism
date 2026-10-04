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
//! * [`archetype_fragmentation`] — archetype-explosion and chunk
//!   internal-fragmentation accounting with a worst-occupancy-first
//!   ranking (design §16.6 / §22 risk #5).
//! * [`change_volume`] — dirty-chunk and changed/added-cell accounting that
//!   quantifies the design headline "成本 ∝ 变化量".
//! * [`frame_profile`] — per-system change-volume ranking folded from a
//!   [`step_inspector`] frame trace (design §16.6 "帧级变更量").
//! * [`profiler`] — nested system-span timing and flame-graph export
//!   (design §16.6 "系统火焰图").
//! * [`step_inspector`] — per-system structural / change-volume observation
//!   while single-stepping a schedule (design §23.4 / §16.6).
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
pub mod frame_profile;
pub mod inspector;
pub mod profiler;
pub mod relation_graph;
pub mod relation_cycles;
pub mod relation_topology;
pub mod step_inspector;
#[cfg(feature = "std")]
pub mod time_travel;

pub use archetype_fragmentation::{ArchetypeFragmentEntry, ArchetypeFragmentationReport};
pub use change_volume::{ArchetypeChangeReport, ChangeReport};
pub use frame_profile::{FrameChangeProfile, SystemChangeEntry};
pub use inspector::{ArchetypeReport, OccupancyStats, WorldReport};
pub use relation_graph::{RelationGraphReport, RelationKindReport};
pub use relation_cycles::{RelationCycleEntry, RelationCycleReport};
pub use relation_topology::{RelationTopologyEntry, RelationTopologyReport};
pub use step_inspector::{StepObservation, SteppingInspector};
#[cfg(feature = "std")]
pub use profiler::SpanRecorder;
pub use profiler::{FlameGraph, SpanNode, SystemInstrument};
#[cfg(feature = "std")]
pub use time_travel::TimeTravel;
