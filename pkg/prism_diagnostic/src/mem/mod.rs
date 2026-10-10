//! §24.3 memory tracking: leak reconciliation, per-category memory budgets, and
//! fragmentation analysis — the reconciliation/guard layer built *on top of* the
//! tagged allocator counters in [`crate::alloc_track`].
//!
//! [`crate::alloc_track`] already delivers the hot-path core (tagged allocation
//! with exact live/peak byte accounting). This module adds the three §24.3
//! pieces that sit at frame/scope boundaries and never touch the allocator hot
//! path, so they are pure `core`/`alloc` arithmetic (deterministic, `no_std` +
//! `alloc`, no `unsafe`) and are always compiled regardless of the
//! `alloc-track` feature:
//!
//! - **Leak detection** ([`leak`]): snapshot live bytes/allocations at a
//!   scope/frame boundary, reconcile at the matching boundary, and report any
//!   residual that a pool expected to return to zero (e.g. after a level
//!   switch) failed to release.
//! - **Memory budget guard** ([`budget`]): declare per-category byte budgets
//!   (assets/render/gameplay) and red-flag categories that exceed them, to
//!   prevent shipping-build OOM.
//! - **Fragmentation visualization** ([`fragmentation`]): turn a pool/virtual
//!   address occupancy list into free-run metrics and a coarse occupancy map
//!   that exposes fragmentation and large holes to guide allocator tuning.
//!
//! When the `alloc-track` feature is on, convenience constructors read directly
//! from [`crate::alloc_track`] snapshots; the core types stay feature-neutral so
//! they can be unit-tested without installing a global allocator.

pub mod budget;
pub mod fragmentation;
pub mod leak;

pub use budget::{MemBudget, MemBudgetRegistry, MemBudgetReport, MemBudgetStatus};
pub use fragmentation::{analyze_fragmentation, occupancy_map, FragmentationReport, Span};
pub use leak::{LeakCheckpoint, LeakReport};
#[cfg(feature = "alloc-track")]
pub use leak::{TagLeakCheckpoint, TagLeakReport, TagLeakResidual};
