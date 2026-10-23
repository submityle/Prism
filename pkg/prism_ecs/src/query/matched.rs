//! The incremental archetype-match cache and the spin lock that guards it
//! (design §7 query cache).
//!
//! A [`QueryState`](crate::query::QueryState) must be `Sync`: it satisfies the
//! `SystemParam::State: Send + Sync` bound and can legitimately be shared behind
//! an `Arc` while iterated from several threads. Filling the match cache is a
//! mutation, so it cannot live in a bare [`UnsafeCell`](core::cell::UnsafeCell)
//! (which would let two threads race on `Vec::push`/`clone`). [`SpinLock`]
//! makes the refresh a critical section instead: readers serialize briefly on a
//! single `AtomicBool`, so `&self` refresh is race-free with no `std` dependency.

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::cell::UnsafeCell;
use core::hint::spin_loop;
use core::ops::{Deref, DerefMut};
use core::sync::atomic::{AtomicBool, Ordering};

use crate::archetype::ArchetypeId;

/// Incrementally-maintained set of archetypes a query matches.
///
/// Archetype component sets are immutable and archetype ids are monotonic and
/// never reused (except by a wholesale registry replacement on restore), so a
/// match decision for a live [`ArchetypeId`] is permanent within a structural
/// generation. Evicted IDs are removed from `ids` when an eviction event
/// intersects the cached matches. The cache records the highest ID examined,
/// the ids that matched, and its structural generation and eviction-log cursor.
pub(crate) struct MatchedArchetypes {
    /// Immutable, shareable ids of every non-evicted archetype matched so far,
    /// in ascending id order. Query iterators retain this snapshot without
    /// copying the full list on every iteration; a new allocation is published
    /// only when the structural generation changes, matching ids are added, or
    /// evicted ids are compacted.
    pub(crate) ids: Arc<Vec<ArchetypeId>>,
    /// Highest monotonically-issued ID tested, including IDs since evicted.
    /// `None` means no registry ID has been examined yet.
    pub(crate) checked_through: Option<ArchetypeId>,
    /// The `World::archetype_generation` these `ids`/`checked_through` were computed
    /// against; a mismatch clears the cache and re-scans from scratch.
    pub(crate) generation: u64,
    /// Number of `Archetypes::evicted_ids` events already checked. Unlike a
    /// structural-generation reset, an eviction only removes stable ids from
    /// this snapshot; the archetype scan cursor remains valid.
    pub(crate) eviction_checked: usize,
}

impl MatchedArchetypes {
    /// An empty cache stamped with the world's structural `generation`.
    pub(crate) fn new(generation: u64, eviction_checked: usize) -> Self {
        Self {
            ids: Arc::new(Vec::new()),
            checked_through: None,
            generation,
            eviction_checked,
        }
    }
}

/// A minimal `no_std` spin lock.
///
/// The critical section (one incremental match scan) is short and a given
/// [`QueryState`](crate::query::QueryState) is almost always touched by one
/// thread, so spinning is an acceptable trade for staying allocation-free and
/// dependency-free. Correctness does not depend on low contention: the
/// `AtomicBool` + `UnsafeCell` pairing is the standard lock idiom.
pub(crate) struct SpinLock<T> {
    locked: AtomicBool,
    value: UnsafeCell<T>,
}

// SAFETY: the only route to the inner `T` is through `lock`, which hands out a
// `&mut T` exclusively (one holder at a time, an `Acquire`/`Release` pair around
// every section). Thus `&SpinLock<T>` grants other threads nothing more than
// `&T`-equivalent access serialized against writers, which is exactly what
// `Sync` promises given `T: Send`.
unsafe impl<T: Send> Sync for SpinLock<T> {}
// SAFETY: moving the lock moves the `UnsafeCell<T>` with it; that is sound
// whenever `T` is `Send`.
unsafe impl<T: Send> Send for SpinLock<T> {}

impl<T> SpinLock<T> {
    /// Create an unlocked spin lock holding `value`.
    pub(crate) fn new(value: T) -> Self {
        Self {
            locked: AtomicBool::new(false),
            value: UnsafeCell::new(value),
        }
    }

    /// Acquire the lock, spinning until it is free.
    #[inline]
    pub(crate) fn lock(&self) -> SpinGuard<'_, T> {
        while self
            .locked
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            spin_loop();
        }
        SpinGuard { lock: self }
    }
}

/// RAII guard returned by [`SpinLock::lock`]; releases on drop (including when a
/// panic unwinds out of the critical section).
pub(crate) struct SpinGuard<'a, T> {
    lock: &'a SpinLock<T>,
}

impl<T> Deref for SpinGuard<'_, T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: holding the guard means this thread owns the critical section,
        // so no other reference to the inner value exists.
        unsafe { &*self.lock.value.get() }
    }
}

impl<T> DerefMut for SpinGuard<'_, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: `&mut self` proves this guard is the unique holder, and the
        // guard is the only route to the inner value.
        unsafe { &mut *self.lock.value.get() }
    }
}

impl<T> Drop for SpinGuard<'_, T> {
    #[inline]
    fn drop(&mut self) {
        self.lock.locked.store(false, Ordering::Release);
    }
}
