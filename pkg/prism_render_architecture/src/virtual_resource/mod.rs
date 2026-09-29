//! Shared virtual-resource scheduling contracts.
//!
//! Several rendering subsystems — texture streaming, virtual shadow maps,
//! virtual geometry, and more — all page bounded slices of a much larger virtual
//! address space into limited physical memory. Rather than each reinventing the
//! same residency logic, they share this contract layer. It has four cooperating
//! parts:
//!
//! * [`state`] — the [`ResidencyState`] lifecycle machine and its transition
//!   validation, so a resource can never skip an upload or resurrect out of
//!   order.
//! * [`budget`] — the [`ResidencyBudget`] limits (soft target, hard ceiling, and
//!   per-frame upload cap) and the [`budget::BudgetLedger`] that accounts for
//!   them during a pass.
//! * [`registry`] — the [`registry::VirtualResourceTable`] that stores each
//!   resource's priority, parent dependency, byte cost, state, and invalidation
//!   epoch, and implements [`VirtualResourceClient`].
//! * [`scheduler`] — a deterministic, priority-ordered greedy pass that honours
//!   both budgets and parent chains, producing a [`scheduler::ResidencyPlan`] of
//!   uploads, evictions, and deferrals.
//!
//! No layer holds a `GPU` handle. Residency here is `CPU`-side truth over
//! integer state, so scheduling is exact and reproducible; the backend issues
//! the real uploads, pending the `GPU` backend.

use std::hash::Hash;

pub mod budget;
pub mod registry;
pub mod scheduler;
pub mod state;

pub use budget::BudgetLedger;
pub use registry::{ResourceEntry, ResourceKey, TransitionError, VirtualResourceTable};
pub use scheduler::{schedule, schedule_and_apply, ResidencyPlan};
pub use state::IllegalTransition;

/// Scheduling priority of a residency request; greater is admitted first.
///
/// Ordered so the scheduler and residency containers break ties deterministically.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RequestPriority(pub u32);

/// The three independent limits that gate residency for one frame.
///
/// `soft_bytes` is a comfort target, `hard_bytes` the physical ceiling the
/// resident set may never exceed, and `upload_bytes_per_frame` the cap on *fresh*
/// residency the backend can stream in a single frame. See [`budget`] for the
/// accounting rules built on these fields.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ResidencyBudget {
    /// Comfort target; staying at or under it means no memory pressure.
    pub soft_bytes: u64,
    /// Physical ceiling; the resident set may never exceed it.
    pub hard_bytes: u64,
    /// Cap on bytes of fresh residency uploaded in a single frame.
    pub upload_bytes_per_frame: u64,
}

/// Lifecycle state of a tracked virtual resource.
///
/// A resource climbs `Missing -> Requested -> Uploading -> Resident -> Retiring`
/// as it is demanded, uploaded, kept, and wound down. The legal transitions
/// between these states are defined in [`state`].
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ResidencyState {
    /// Not backed by physical storage and not currently requested.
    #[default]
    Missing,
    /// A view has asked for the resource; no upload has started yet.
    Requested,
    /// An upload is in flight to make the resource resident.
    Uploading,
    /// Backed by physical storage and ready to use.
    Resident,
    /// Being wound down; its storage will be reclaimed.
    Retiring,
}

/// Contract a subsystem implements to expose its resources to shared scheduling.
///
/// The [`registry::VirtualResourceTable`] provides a ready-made implementation;
/// subsystems with their own storage can implement this directly to describe a
/// resource's priority, parent dependency, and invalidation without adopting the
/// table.
pub trait VirtualResourceClient {
    /// Key identifying one resource.
    type Key: Copy + Eq + Hash;

    /// Scheduling priority of `key`.
    fn priority(&self, key: Self::Key) -> RequestPriority;
    /// Parent `key` depends on, if any; the parent must be resident first.
    fn parent(&self, key: Self::Key) -> Option<Self::Key>;
    /// Applies an invalidation stamped with `epoch`, forcing a re-upload when the
    /// epoch is newer than the one last seen for `key`.
    fn invalidate(&mut self, key: Self::Key, epoch: u64);
}
