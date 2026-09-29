//! `GPU` heap and frame-budget contracts, plus their `CPU`-side allocators.
//!
//! The contract types below (heap classes, per-heap statistics, and the
//! per-frame work budget) describe *what* the backend must report. The
//! submodules implement the deterministic, integer-only bookkeeping that
//! produces those numbers on the `CPU`:
//!
//! - [`suballocator`]: general free-list offset allocator with coalescing.
//! - [`linear`]: bump allocator for per-frame transient memory.
//! - [`aliasing`]: shares one region between resources with disjoint lifetimes.
//! - [`budget`]: frame work-budget ledger and per-heap statistics registry.
//!
//! Binding real device memory to these offsets is *pending the GPU backend*.

pub mod aliasing;
pub mod budget;
pub mod linear;
pub mod suballocator;

pub use aliasing::{plan_aliasing, AliasError, AliasGroup, AliasPlan, ResourceLifetime};
pub use budget::{BudgetOverage, FrameBudgetLedger, HeapStatsRegistry};
pub use linear::{LinearAllocation, LinearAllocator, LinearError};
pub use suballocator::{FitStrategy, SubAllocError, SubAllocation, SubAllocator, SubFreeError};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum HeapClass {
    DeviceLocal,
    Upload,
    Readback,
    Transient,
    AccelerationStructure,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct HeapStats {
    pub committed_bytes: u64,
    pub used_bytes: u64,
    pub largest_free_block: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FrameWorkBudget {
    pub upload_bytes: u64,
    pub readback_bytes: u64,
    pub relocation_bytes: u64,
    pub acceleration_structure_builds: u32,
}
