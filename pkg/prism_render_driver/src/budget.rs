//! GPU memory budget accounting across logical heaps.
//!
//! Discrete GPUs expose a few distinct memory pools — device-local VRAM,
//! system memory visible to the GPU, and a (usually small) host-visible +
//! device-local "upload" pool. Mapping APIs (DXGI budgets, `VK_EXT_memory_budget`,
//! Metal residency sets) report both the pool size and an OS-provided *budget*
//! that can shrink when other apps compete for VRAM. A renderer that ignores
//! the budget and over-commits gets its allocations demoted to system memory
//! (a silent, severe perf cliff) or fails outright.
//!
//! [`MemoryBudget`] tracks reserved vs. budgeted bytes per [`MemoryHeap`] and
//! answers the two questions a streaming/residency system needs: "can this
//! allocation fit?" and "how far am I over/under budget right now?". It is a
//! pure accounting structure — it holds no memory — so it is trivially testable
//! and backend-agnostic. `no_std`, no `unsafe`, all arithmetic saturating.

/// The logical GPU memory pools Prism distinguishes for budgeting. A backend
/// maps each to the nearest native heap; integrated GPUs may alias several onto
/// one physical pool.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MemoryHeap {
    /// Device-local VRAM: the fast path for render targets, textures, and
    /// vertex/index data on discrete GPUs.
    DeviceLocal,
    /// Host-visible, device-local memory: a small pool ideal for per-frame
    /// upload staging that the GPU can read directly.
    HostVisibleDeviceLocal,
    /// System memory the GPU can access over the bus: large but slow, used for
    /// spill and bulk staging.
    HostVisible,
}

impl MemoryHeap {
    /// Every heap kind, in a fixed order, for iteration/reporting.
    #[must_use]
    pub const fn all() -> [MemoryHeap; 3] {
        [
            MemoryHeap::DeviceLocal,
            MemoryHeap::HostVisibleDeviceLocal,
            MemoryHeap::HostVisible,
        ]
    }

    /// A stable array index in `0..3` for storing per-heap state compactly.
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            MemoryHeap::DeviceLocal => 0,
            MemoryHeap::HostVisibleDeviceLocal => 1,
            MemoryHeap::HostVisible => 2,
        }
    }
}

/// Per-heap accounting: how many bytes the OS currently budgets for this app
/// and how many Prism has reserved.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct HeapAccount {
    /// Total physical bytes in the heap.
    capacity: u64,
    /// Bytes the OS currently allows this process to use (<= capacity, and may
    /// shrink under memory pressure).
    budget: u64,
    /// Bytes Prism has currently reserved.
    reserved: u64,
    /// The high-water mark of `reserved`, for diagnostics.
    peak: u64,
}

impl HeapAccount {
    const fn new() -> Self {
        Self {
            capacity: 0,
            budget: 0,
            reserved: 0,
            peak: 0,
        }
    }
}

/// Tracks reservations against OS-reported budgets for every [`MemoryHeap`].
pub struct MemoryBudget {
    heaps: [HeapAccount; 3],
}

/// A snapshot of one heap's budget accounting.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct HeapUsage {
    /// Total physical bytes in the heap.
    pub capacity: u64,
    /// Bytes currently budgeted to this process by the OS.
    pub budget: u64,
    /// Bytes currently reserved by Prism.
    pub reserved: u64,
    /// High-water mark of `reserved`.
    pub peak: u64,
}

impl HeapUsage {
    /// Bytes still available before hitting the budget (saturating at 0).
    #[must_use]
    pub const fn available(self) -> u64 {
        self.budget.saturating_sub(self.reserved)
    }

    /// Bytes reserved beyond the budget (0 when within budget).
    #[must_use]
    pub const fn overcommit(self) -> u64 {
        self.reserved.saturating_sub(self.budget)
    }

    /// Whether reservations currently exceed the OS budget.
    #[must_use]
    pub const fn is_over_budget(self) -> bool {
        self.reserved > self.budget
    }

    /// Fraction of the budget in use, `reserved / budget`, saturating. Returns
    /// `0.0` when the budget is zero.
    #[must_use]
    pub fn utilization(self) -> f64 {
        if self.budget == 0 {
            0.0
        } else {
            self.reserved as f64 / self.budget as f64
        }
    }
}

impl MemoryBudget {
    /// Creates a tracker with every heap zero-sized. Call [`Self::set_limits`]
    /// once the backend has queried the adapter.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            heaps: [HeapAccount::new(), HeapAccount::new(), HeapAccount::new()],
        }
    }

    /// Sets a heap's physical capacity and current OS budget. The budget is
    /// clamped to the capacity. Backends call this at init and whenever the OS
    /// reports a budget change (VRAM pressure from other apps).
    pub fn set_limits(&mut self, heap: MemoryHeap, capacity: u64, budget: u64) {
        let h = &mut self.heaps[heap.index()];
        h.capacity = capacity;
        h.budget = budget.min(capacity);
    }

    /// Whether `bytes` can be reserved from `heap` without exceeding its
    /// budget.
    #[must_use]
    pub fn can_reserve(&self, heap: MemoryHeap, bytes: u64) -> bool {
        let h = &self.heaps[heap.index()];
        h.reserved.saturating_add(bytes) <= h.budget
    }

    /// Reserves `bytes` from `heap` unconditionally, updating the peak. Returns
    /// the resulting [`HeapUsage`]. Use when the allocation has already
    /// happened and must be accounted even if it pushes over budget (the
    /// caller then reacts to [`HeapUsage::is_over_budget`]).
    pub fn reserve(&mut self, heap: MemoryHeap, bytes: u64) -> HeapUsage {
        let h = &mut self.heaps[heap.index()];
        h.reserved = h.reserved.saturating_add(bytes);
        if h.reserved > h.peak {
            h.peak = h.reserved;
        }
        Self::usage_of(h)
    }

    /// Attempts to reserve `bytes` from `heap`, succeeding only if it stays
    /// within budget. Returns the new [`HeapUsage`] on success or `None` if the
    /// reservation would overcommit (nothing is reserved in that case).
    pub fn try_reserve(&mut self, heap: MemoryHeap, bytes: u64) -> Option<HeapUsage> {
        if self.can_reserve(heap, bytes) {
            Some(self.reserve(heap, bytes))
        } else {
            None
        }
    }

    /// Releases `bytes` back to `heap` (saturating at 0 so double-frees cannot
    /// underflow the counter).
    pub fn release(&mut self, heap: MemoryHeap, bytes: u64) {
        let h = &mut self.heaps[heap.index()];
        h.reserved = h.reserved.saturating_sub(bytes);
    }

    /// The current usage snapshot for `heap`.
    #[must_use]
    pub fn usage(&self, heap: MemoryHeap) -> HeapUsage {
        Self::usage_of(&self.heaps[heap.index()])
    }

    /// Whether any heap is currently over its budget.
    #[must_use]
    pub fn any_over_budget(&self) -> bool {
        self.heaps.iter().any(|h| h.reserved > h.budget)
    }

    /// Total reserved bytes across every heap.
    #[must_use]
    pub fn total_reserved(&self) -> u64 {
        self.heaps
            .iter()
            .fold(0u64, |acc, h| acc.saturating_add(h.reserved))
    }

    fn usage_of(h: &HeapAccount) -> HeapUsage {
        HeapUsage {
            capacity: h.capacity,
            budget: h.budget,
            reserved: h.reserved,
            peak: h.peak,
        }
    }
}

impl Default for MemoryBudget {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserve_and_release_within_budget() {
        let mut b = MemoryBudget::new();
        b.set_limits(MemoryHeap::DeviceLocal, 8 << 30, 6 << 30);
        assert!(b.can_reserve(MemoryHeap::DeviceLocal, 4 << 30));
        let u = b.try_reserve(MemoryHeap::DeviceLocal, 4 << 30).unwrap();
        assert_eq!(u.reserved, 4 << 30);
        assert_eq!(u.available(), 2 << 30);
        assert!(!u.is_over_budget());
        b.release(MemoryHeap::DeviceLocal, 4 << 30);
        assert_eq!(b.usage(MemoryHeap::DeviceLocal).reserved, 0);
    }

    #[test]
    fn try_reserve_refuses_overcommit() {
        let mut b = MemoryBudget::new();
        b.set_limits(MemoryHeap::DeviceLocal, 4 << 30, 4 << 30);
        assert!(b.try_reserve(MemoryHeap::DeviceLocal, 3 << 30).is_some());
        assert!(b.try_reserve(MemoryHeap::DeviceLocal, 2 << 30).is_none());
        // Nothing was reserved by the failed call.
        assert_eq!(b.usage(MemoryHeap::DeviceLocal).reserved, 3 << 30);
    }

    #[test]
    fn forced_reserve_reports_overcommit() {
        let mut b = MemoryBudget::new();
        b.set_limits(MemoryHeap::DeviceLocal, 4 << 30, 2 << 30);
        let u = b.reserve(MemoryHeap::DeviceLocal, 3 << 30);
        assert!(u.is_over_budget());
        assert_eq!(u.overcommit(), 1 << 30);
        assert!(b.any_over_budget());
    }

    #[test]
    fn budget_clamped_to_capacity() {
        let mut b = MemoryBudget::new();
        b.set_limits(MemoryHeap::HostVisible, 1 << 30, 8 << 30);
        assert_eq!(b.usage(MemoryHeap::HostVisible).budget, 1 << 30);
    }

    #[test]
    fn peak_tracks_high_water_mark() {
        let mut b = MemoryBudget::new();
        b.set_limits(MemoryHeap::DeviceLocal, 8 << 30, 8 << 30);
        b.reserve(MemoryHeap::DeviceLocal, 5 << 30);
        b.release(MemoryHeap::DeviceLocal, 5 << 30);
        b.reserve(MemoryHeap::DeviceLocal, 1 << 30);
        assert_eq!(b.usage(MemoryHeap::DeviceLocal).peak, 5 << 30);
    }

    #[test]
    fn release_saturates_at_zero() {
        let mut b = MemoryBudget::new();
        b.set_limits(MemoryHeap::DeviceLocal, 8 << 30, 8 << 30);
        b.reserve(MemoryHeap::DeviceLocal, 1 << 30);
        b.release(MemoryHeap::DeviceLocal, 4 << 30);
        assert_eq!(b.usage(MemoryHeap::DeviceLocal).reserved, 0);
    }

    #[test]
    fn utilization_and_totals() {
        let mut b = MemoryBudget::new();
        b.set_limits(MemoryHeap::DeviceLocal, 100, 100);
        b.set_limits(MemoryHeap::HostVisible, 100, 100);
        b.reserve(MemoryHeap::DeviceLocal, 50);
        b.reserve(MemoryHeap::HostVisible, 25);
        assert!((b.usage(MemoryHeap::DeviceLocal).utilization() - 0.5).abs() < 1e-9);
        assert_eq!(b.total_reserved(), 75);
    }
}
