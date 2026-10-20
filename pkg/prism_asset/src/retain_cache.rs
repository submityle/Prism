//! Bounded, arena-owned retention reservations (design §6.4).
//!
//! Admission pre-charges every retained value, even while externally owned.
//! Pinned and KeepAlive reservations cannot be evicted by memory pressure;
//! only idle LRU reservations can. An impossible admission/shrink is atomic.
//! Byte charges describe caller-declared payload residency, not GPU pages,
//! allocator overhead, or all externally owned assets in the arena.
//!
//! The design's `Lru { budget_bytes }` is split into `RetentionPolicy::Lru`
//! and one shared `RetainBudget` per arena, avoiding contradictory per-entry
//! pool limits. Entry count also bounds zero-byte values and metadata.
//! This opt-in control-plane cache uses bounded linear scans, no locks and no
//! timestamp counters: vector order itself is the deterministic LRU order.

use crate::asset::Asset;
use crate::handle::{Handle, HandleId};
use crate::id::{AssetId, UntypedAssetId};
use alloc::vec::Vec;
use core::fmt;

/// Whole-asset retention after the last external owner releases its handle.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RetentionPolicy {
    /// Do not retain a cache-owned handle (the default).
    #[default]
    Immediate,
    /// Survive this many additional reclaim points after first observed idle.
    KeepAlive {
        /// Number of reclaim points; zero releases on the first idle observation.
        frames: u32,
    },
    /// Retain until pressure evicts the least recently explicitly touched idle entry.
    Lru,
    /// Never automatically evict; explicit removal or invalidation is required.
    Pinned,
}

/// Hard reservation limits shared by all retention policies in an arena.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RetainBudget {
    /// Maximum sum of caller-declared payload bytes, including pinned values.
    pub max_bytes: u64,
    /// Maximum number of retained handles, including zero-byte values.
    pub max_entries: usize,
}

/// Current pre-charged retention usage (including externally referenced entries).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RetainUsage {
    /// Sum of all admitted byte charges.
    pub bytes: u64,
    /// Number of admitted reservations.
    pub entries: usize,
}

/// Admission failures never partially evict or change the previous policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RetainError {
    /// The handle is stale, retired, or belongs to a different arena allocation.
    InvalidHandle,
    /// Pending and failed slots do not have a resident payload to retain.
    NotReady,
    /// Non-evictable usage plus the candidate cannot fit after evicting all idle LRU entries.
    BudgetExceeded {
        /// Remaining required bytes; u128 preserves overflowing u64 proposals.
        required_bytes: u128,
        /// Remaining required entries after all eligible evictions.
        required_entries: u128,
        /// Requested limits; the live budget is unchanged on rejection.
        budget: RetainBudget,
    },
}

impl fmt::Display for RetainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidHandle => f.write_str("invalid or foreign retention handle"),
            Self::NotReady => f.write_str("only ready payloads can be retained"),
            Self::BudgetExceeded { required_bytes, required_entries, budget } => write!(
                f, "retention needs {required_bytes} bytes/{required_entries} entries; budget is {} bytes/{} entries",
                budget.max_bytes, budget.max_entries
            ),
        }
    }
}

impl core::error::Error for RetainError {}

/// Why a cache-owned handle was released; external handles remain valid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RetainRemovalReason {
    /// An idle LRU entry made room for an admission or a smaller budget.
    Budget,
    /// A KeepAlive window expired.
    Expired,
    /// Explicit unpin/release or Immediate policy.
    Explicit,
    /// Unaccounted payload mutation/replacement or failure revoked its reservation.
    PayloadChanged,
    /// The arena explicitly removed this generation.
    Removed,
    /// A cycle collector retired the handle allocation.
    Retired,
}

/// An observable release of one reservation, not necessarily payload destruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RetainRemoval {
    /// Exact type and generation formerly retained.
    pub id: UntypedAssetId,
    /// Allocation identity disambiguates independent arenas with identical ids.
    pub handle_id: HandleId,
    /// Byte reservation released by this operation.
    pub bytes: u64,
    /// Cause of release.
    pub reason: RetainRemovalReason,
}

/// Read-only reservation details, useful for budget/policy diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RetainedAsset {
    /// Exact type and generation.
    pub id: UntypedAssetId,
    /// The retained allocation.
    pub handle_id: HandleId,
    /// Pre-charged payload bytes.
    pub bytes: u64,
    /// Current policy.
    pub policy: RetentionPolicy,
    /// Reclaim points since the first observation with no external handle.
    pub idle_frames: Option<u32>,
}

/// Diagnostics returned by a combined retention/reclaim point.
#[derive(Debug, Default)]
pub struct RetainCollection {
    /// Arena payload slots actually destroyed this call.
    pub reclaimed: usize,
    /// Cache-owned handles released, in deterministic LRU order.
    pub removals: Vec<RetainRemoval>,
}

struct Retained<A> {
    handle: Handle<A>,
    bytes: u64,
    policy: RetentionPolicy,
    idle_frames: Option<u32>,
}

impl<A: Asset> Retained<A> {
    fn snapshot(&self) -> RetainedAsset {
        RetainedAsset {
            id: self.handle.untyped_id(),
            handle_id: self.handle.handle_id(),
            bytes: self.bytes,
            policy: self.policy,
            idle_frames: self.idle_frames,
        }
    }

    fn removal(&self, reason: RetainRemovalReason) -> RetainRemoval {
        RetainRemoval {
            id: self.handle.untyped_id(),
            handle_id: self.handle.handle_id(),
            bytes: self.bytes,
            reason,
        }
    }
}

/// Read-only cache inspection; checked mutations are owned by `Assets<A>`.
///
/// The arena validates generation **and** handle allocation before admitting a
/// strong reference. A cache never resurrects an expired weak handle or exposes
/// unchecked insertion by bare slot id. `Assets::get` remains a pure read;
/// recency changes only on admission, `acquire_retained`, or `touch_retained`.
pub struct RetainCache<A> {
    budget: RetainBudget,
    bytes: u64,
    entries: Vec<Retained<A>>,
}

impl<A: Asset> RetainCache<A> {
    /// Creates an empty cache. Both limits must permit an entry for it to be admitted.
    #[must_use]
    pub const fn new(budget: RetainBudget) -> Self {
        Self {
            budget,
            bytes: 0,
            entries: Vec::new(),
        }
    }

    /// The current hard reservation limits.
    #[must_use]
    pub const fn budget(&self) -> RetainBudget {
        self.budget
    }

    /// Current reserved payload bytes and handles.
    #[must_use]
    pub fn usage(&self) -> RetainUsage {
        RetainUsage {
            bytes: self.bytes,
            entries: self.entries.len(),
        }
    }

    /// Snapshots in oldest-to-newest explicit access order.
    pub fn iter(&self) -> impl Iterator<Item = RetainedAsset> + '_ {
        self.entries.iter().map(Retained::snapshot)
    }

    /// Reservation metadata for an exact generation.
    #[must_use]
    pub fn get(&self, id: AssetId<A>) -> Option<RetainedAsset> {
        self.entries
            .iter()
            .find(|e| e.handle.id() == id)
            .map(Retained::snapshot)
    }

    pub(crate) fn set_budget(
        &mut self,
        budget: RetainBudget,
    ) -> Result<Vec<RetainRemoval>, RetainError> {
        let victims = self.plan(budget, None, 0, 0)?;
        let removals = self.evict(victims);
        self.budget = budget;
        Ok(removals)
    }

    pub(crate) fn retain(
        &mut self,
        handle: &Handle<A>,
        bytes: u64,
        policy: RetentionPolicy,
    ) -> Result<Vec<RetainRemoval>, RetainError> {
        if policy == RetentionPolicy::Immediate {
            return Ok(self
                .remove(handle.id(), RetainRemovalReason::Explicit)
                .into_iter()
                .collect());
        }
        let existing = self
            .entries
            .iter()
            .position(|e| e.handle.handle_id() == handle.handle_id());
        let victims = self.plan(self.budget, existing, bytes, 1)?;
        let removals = self.evict(victims);
        // Re-find after eviction; positions may have shifted.
        if let Some(index) = self
            .entries
            .iter()
            .position(|e| e.handle.handle_id() == handle.handle_id())
        {
            self.bytes -= self.entries.remove(index).bytes;
        }
        // The u128 plan proved the following addition fits the u64 budget.
        self.bytes += bytes;
        self.entries.push(Retained {
            handle: handle.clone(),
            bytes,
            policy,
            idle_frames: None,
        });
        Ok(removals)
    }

    // Planning may allocate bounded control-plane scratch, but never changes
    // handles or recency until all constraints have been proven satisfiable.
    fn plan(
        &self,
        budget: RetainBudget,
        replacing: Option<usize>,
        bytes: u64,
        count: usize,
    ) -> Result<Vec<usize>, RetainError> {
        let old_bytes = replacing.map_or(0, |i| self.entries[i].bytes);
        let mut required_bytes = u128::from(self.bytes - old_bytes) + u128::from(bytes);
        let mut required_entries =
            (self.entries.len() - usize::from(replacing.is_some())) as u128 + count as u128;
        let fits =
            |b: u128, n: u128| b <= u128::from(budget.max_bytes) && n <= budget.max_entries as u128;
        let mut victims = Vec::new();
        for (index, entry) in self.entries.iter().enumerate() {
            if fits(required_bytes, required_entries) {
                break;
            }
            if Some(index) != replacing
                && entry.policy == RetentionPolicy::Lru
                && entry.handle.strong_count() == 1
            {
                required_bytes -= u128::from(entry.bytes);
                required_entries -= 1;
                victims.push(index);
            }
        }
        if !fits(required_bytes, required_entries) {
            return Err(RetainError::BudgetExceeded {
                required_bytes,
                required_entries,
                budget,
            });
        }
        Ok(victims)
    }

    fn evict(&mut self, victims: Vec<usize>) -> Vec<RetainRemoval> {
        let removals = victims
            .iter()
            .map(|&i| self.entries[i].removal(RetainRemovalReason::Budget))
            .collect();
        for index in victims.into_iter().rev() {
            self.bytes -= self.entries.remove(index).bytes;
        }
        removals
    }

    pub(crate) fn remove(
        &mut self,
        id: AssetId<A>,
        reason: RetainRemovalReason,
    ) -> Option<RetainRemoval> {
        let index = self.entries.iter().position(|e| e.handle.id() == id)?;
        let entry = self.entries.remove(index);
        self.bytes -= entry.bytes;
        Some(entry.removal(reason))
    }

    pub(crate) fn acquire(&mut self, id: AssetId<A>) -> Option<Handle<A>> {
        let index = self.entries.iter().position(|e| e.handle.id() == id)?;
        // Honor cycle-collector tombstones instead of cloning a retired handle.
        let handle = self.entries[index].handle.downgrade().upgrade()?;
        self.touch(id);
        Some(handle)
    }

    pub(crate) fn touch(&mut self, id: AssetId<A>) -> bool {
        let Some(index) = self.entries.iter().position(|e| e.handle.id() == id) else {
            return false;
        };
        let mut entry = self.entries.remove(index);
        entry.idle_frames = None;
        self.entries.push(entry);
        true
    }

    pub(crate) fn advance(
        &mut self,
        activity: &mut impl FnMut(AssetId<A>, bool),
    ) -> Vec<RetainRemoval> {
        let mut removals = Vec::new();
        let mut index = 0;
        while index < self.entries.len() {
            let entry = &mut self.entries[index];
            // Drop the probe's temporary strong handle before measuring idle
            // ownership; the probe itself is not an external cache consumer.
            let retired = entry.handle.downgrade().upgrade().is_none();
            let reason = if retired {
                Some(RetainRemovalReason::Retired)
            } else if entry.handle.strong_count() == 1 {
                entry.idle_frames = Some(match entry.idle_frames {
                    None => {
                        activity(entry.handle.id(), true);
                        0
                    }
                    Some(frames) => frames.saturating_add(1),
                });
                match entry.policy {
                    RetentionPolicy::KeepAlive { frames }
                        if entry.idle_frames.is_some_and(|age| age >= frames) =>
                    {
                        Some(RetainRemovalReason::Expired)
                    }
                    _ => None,
                }
            } else {
                entry.idle_frames = None;
                activity(entry.handle.id(), false);
                None
            };
            if let Some(reason) = reason {
                let entry = self.entries.remove(index);
                self.bytes -= entry.bytes;
                removals.push(entry.removal(reason));
            } else {
                index += 1;
            }
        }
        removals
    }
}

impl<A: Asset> Default for RetainCache<A> {
    fn default() -> Self {
        Self::new(RetainBudget::default())
    }
}
