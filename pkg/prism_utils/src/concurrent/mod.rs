//! # Concurrent containers (M5, `concurrent` feature)
//!
//! Lock-free and sharded building blocks for the multi-threaded parts of the
//! engine (work-stealing task queues in `prism_tasks`, the per-thread
//! diagnostic ring in `prism_diagnostic`, asset/type caches shared across
//! worker threads). This is the realisation of design doc §11 (并发容器) and
//! §24.2 (无锁进阶).
//!
//! ## What ships here
//! - [`SpscQueue`](spsc::SpscQueue): a bounded, lock-free single-producer /
//!   single-consumer ring. Cache-line padded head/tail so the two endpoints
//!   never share a cache line. This is the diagnostic thread-local buffer form.
//! - [`MpmcQueue`](mpmc::MpmcQueue): a bounded, lock-free multi-producer /
//!   multi-consumer queue using Dmitry Vyukov's per-slot sequence-number
//!   algorithm. This is the cross-thread message / task-queue form.
//! - [`ConcurrentHashMap`](hash::ConcurrentHashMap): a sharded concurrent hash
//!   map (Java `ConcurrentHashMap` / folly `F14` form) whose read path takes a
//!   per-shard read lock, so disjoint keys scale across cores. This is the
//!   read-mostly asset-cache / type-registry form.
//! - [`Collector`](epoch::Collector) / [`Guard`](epoch::Guard): epoch-based
//!   reclamation (`crossbeam-epoch` form) that defers freeing memory until no
//!   pinned thread can still observe it, solving use-after-free and the `ABA`
//!   problem for lock-free structures.
//! - [`TreiberStack`](stack::TreiberStack): a lock-free stack that reclaims its
//!   popped nodes through [`epoch`], serving as the worked example that the
//!   reclaimer is correct under contention.
//!
//! ## Correctness posture
//! The design doc is explicit (§23): lock-free containers are notoriously hard
//! to get right, and `ABA` / memory-ordering / reclamation-timing bugs surface
//! only as rare crashes or corruption. This module therefore:
//! - uses conservative memory orderings (sequentially consistent fences on the
//!   epoch fast path, acquire/release pairing on the queues),
//! - never recycles memory while a reader can observe it (that is the whole job
//!   of [`epoch`]),
//! - and is covered by multi-threaded stress tests that assert *no* item is
//!   ever lost or duplicated and *every* deferred reclamation runs exactly once.
//!
//! The whole module is gated behind the `concurrent` feature so a single-thread
//! build pays nothing for it.

pub mod epoch;
pub mod hash;
pub mod mpmc;
pub mod spsc;
pub mod stack;

pub use epoch::{Collector, Guard, LocalHandle};
pub use hash::ConcurrentHashMap;
pub use mpmc::MpmcQueue;
pub use spsc::{SpscConsumer, SpscProducer, SpscQueue};
pub use stack::TreiberStack;

use core::ops::{Deref, DerefMut};

/// Pads and aligns a value to (at least) a typical cache line so that two
/// independently-updated fields never land on the same line and thrash each
/// other's caches ("false sharing").
///
/// 64 bytes matches the dominant cache-line size on `x86-64` and `AArch64`.
/// The lock-free queues store their producer and consumer cursors inside a
/// `CachePadded` precisely so a producer spinning on its cursor does not
/// invalidate the consumer's cache line on every store.
#[derive(Clone, Copy, Default, Hash, PartialEq, Eq)]
#[repr(align(64))]
pub struct CachePadded<T> {
    value: T,
}

impl<T> CachePadded<T> {
    /// Wraps `value`, aligning it to a cache line.
    #[inline]
    #[must_use]
    pub const fn new(value: T) -> Self {
        Self { value }
    }

    /// Unwraps and returns the inner value.
    #[inline]
    pub fn into_inner(self) -> T {
        self.value
    }
}

impl<T> Deref for CachePadded<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        &self.value
    }
}

impl<T> DerefMut for CachePadded<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        &mut self.value
    }
}

impl<T: core::fmt::Debug> core::fmt::Debug for CachePadded<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("CachePadded").field(&self.value).finish()
    }
}

#[cfg(test)]
mod tests;
