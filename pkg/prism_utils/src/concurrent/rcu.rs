//! Read-copy-update (`RCU`) for read-mostly shared state.
//!
//! `RCU` is the read-mostly concurrency primitive called for by design doc
//! §24.2 (无锁进阶): *readers never lock and never wait*, while a writer
//! publishes an update by building a fresh copy and atomically swapping it in.
//! It is the right tool for globally shared, overwhelmingly-read tables — a
//! type registry (`prism_reflect`), an asset index (`prism_asset`), a hot
//! config snapshot — where readers vastly outnumber writers and must not pay
//! any synchronisation cost on the hot path.
//!
//! ## How it works
//! [`Rcu<T>`] owns the current value in a heap [`Box`] published through an
//! [`AtomicPtr`]. A reader [`read`](Rcu::read)s by pinning the shared
//! [`epoch`](super::epoch) reclaimer and loading the pointer: a single atomic
//! load plus an epoch pin, with no compare-exchange and no spinning, so disjoint
//! readers scale perfectly across cores and never block each other or a writer.
//! The returned [`RcuGuard`] hands out a `&T` that stays valid for as long as
//! the guard lives.
//!
//! A writer never mutates the live value in place. Instead it builds a brand-new
//! value and [`store`](Rcu::store)s it (or derives one from the current snapshot
//! with [`update`](Rcu::update)), atomically swapping the published pointer. The
//! superseded [`Box`] is handed to the epoch reclaimer via
//! [`Guard::defer`](super::epoch::Guard::defer), which frees it only once no
//! pinned reader can still observe it. That is what makes a reader's borrowed
//! `&T` safe without any reference counting on the read path, and it is also how
//! `RCU` sidesteps the `ABA` / use-after-free hazard shared by every pointer-based
//! lock-free structure (see [`TreiberStack`](super::stack::TreiberStack)).
//!
//! ## When *not* to use it
//! `RCU` shines only when writes are rare: every writer copies the whole value,
//! so a write-heavy workload will thrash. For balanced read/write sharing reach
//! for the sharded [`ConcurrentHashMap`](super::hash::ConcurrentHashMap) instead.
//! Readers also observe a *consistent snapshot* taken at [`read`](Rcu::read)
//! time; a value published by a later writer is simply not seen by an
//! already-held guard. That is the defining semantics of `RCU`, not a bug.

extern crate alloc;

use alloc::boxed::Box;
use core::fmt;
use core::ops::Deref;
use core::sync::atomic::{AtomicPtr, Ordering};

use super::epoch::{pin_with, Collector, Guard};

/// A `Send` wrapper around a raw `Box` pointer so a deferred free closure (which
/// may run on another thread) can own it. Sound exactly when the payload may
/// itself cross threads, i.e. `T: Send`.
struct SendPtr<T>(*mut T);

#[expect(
    unsafe_code,
    reason = "a superseded box may be reclaimed on another thread when T: Send"
)]
// SAFETY: the pointer names a box this writer has already unlinked from the
// `Rcu`; moving responsibility for freeing it to another thread is sound
// whenever the payload may cross threads (`T: Send`).
unsafe impl<T: Send> Send for SendPtr<T> {}

/// A read-copy-update cell holding a read-mostly value of type `T`.
///
/// Clones of readers never block each other or a concurrent writer. See the
/// [module docs](self) for the full posture.
pub struct Rcu<T> {
    /// The currently-published value. Always non-null and always the result of
    /// [`Box::into_raw`]; only a writer ever swaps it, and the old box is freed
    /// through the epoch reclaimer, never inline.
    ptr: AtomicPtr<T>,
    /// Reclamation domain shared by this cell's readers and writers.
    collector: Collector,
}

#[expect(
    unsafe_code,
    reason = "all shared access is mediated by the atomic pointer and the epoch reclaimer"
)]
// SAFETY: readers observe `&T` across threads (needs `T: Sync`) and a writer may
// hand a superseded value to another thread for reclamation (needs `T: Send`);
// all pointer mutation goes through the atomic `ptr`, so these bounds make the
// cell soundly shareable.
unsafe impl<T: Send + Sync> Send for Rcu<T> {}
#[expect(
    unsafe_code,
    reason = "all shared access is mediated by the atomic pointer and the epoch reclaimer"
)]
// SAFETY: see the `Send` impl.
unsafe impl<T: Send + Sync> Sync for Rcu<T> {}

impl<T> Rcu<T> {
    /// Creates a cell published with `value`, using a fresh private reclamation
    /// domain.
    #[must_use]
    pub fn new(value: T) -> Self {
        Self::with_collector(value, Collector::new())
    }

    /// Creates a cell that reclaims superseded versions through `collector`,
    /// letting several structures share one reclamation domain.
    #[must_use]
    pub fn with_collector(value: T, collector: Collector) -> Self {
        Self {
            ptr: AtomicPtr::new(Box::into_raw(Box::new(value))),
            collector,
        }
    }

    /// Takes a consistent read snapshot.
    ///
    /// This is wait-free bar the epoch pin: one atomic load, no spinning. The
    /// returned [`RcuGuard`] borrows the current value; a writer that publishes
    /// a newer value afterwards does not disturb this guard, and the observed
    /// value is kept alive until the guard is dropped.
    #[must_use]
    pub fn read(&self) -> RcuGuard<'_, T> {
        // Pin *before* loading: this keeps whatever we observe from being
        // reclaimed out from under us for the guard's whole lifetime.
        let guard = pin_with(&self.collector);
        let ptr = self.ptr.load(Ordering::Acquire);
        #[expect(
            unsafe_code,
            reason = "ptr is a live published box and the pin defers its reclamation"
        )]
        // SAFETY: `ptr` was published by `Box::into_raw` and is non-null; while
        // we hold `guard` the epoch reclaimer cannot free the box, and the value
        // is never mutated in place (writers only swap in fresh boxes), so this
        // shared borrow is valid for the guard's lifetime.
        let value = unsafe { &*ptr };
        RcuGuard {
            value,
            _guard: guard,
        }
    }

    /// Returns a clone of the current value, releasing the read guard
    /// immediately.
    ///
    /// Prefer [`read`](Rcu::read) when a borrow suffices; this exists for
    /// callers that want to hold the snapshot past the guard without pinning the
    /// reclaimer for that whole time.
    #[must_use]
    pub fn load_cloned(&self) -> T
    where
        T: Clone,
    {
        self.read().clone()
    }

    /// Publishes `value`, replacing the current one wholesale.
    ///
    /// The superseded value is reclaimed once no reader can observe it.
    pub fn store(&self, value: T)
    where
        T: Send + 'static,
    {
        let new = Box::into_raw(Box::new(value));
        let old = self.ptr.swap(new, Ordering::AcqRel);
        self.retire(old);
    }

    /// Read-copy-update: derives a new value from a snapshot of the current one
    /// and publishes it, retrying if a racing writer wins first.
    ///
    /// `f` may run more than once under contention, so it must be free of
    /// observable side effects; it only reads the snapshot and returns the
    /// replacement.
    pub fn update<F>(&self, mut f: F)
    where
        F: FnMut(&T) -> T,
        T: Send + 'static,
    {
        loop {
            // Pin spans the read-copy so the snapshot cannot be reclaimed while
            // `f` reads it to build the replacement.
            let guard = pin_with(&self.collector);
            let current = self.ptr.load(Ordering::Acquire);
            #[expect(
                unsafe_code,
                reason = "current is a live published box kept alive by the pin"
            )]
            // SAFETY: `current` is non-null and, while pinned, cannot be
            // reclaimed, so reading it to build the next value is valid.
            let next = f(unsafe { &*current });
            let new = Box::into_raw(Box::new(next));
            match self
                .ptr
                .compare_exchange(current, new, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => {
                    // We unlinked `current`; retire it through the same pin.
                    self.defer_free(&guard, current);
                    return;
                }
                Err(_) => {
                    // Lost the race; drop the just-built box and retry against
                    // the newly-published value.
                    #[expect(
                        unsafe_code,
                        reason = "new is solely owned here; the CAS never published it"
                    )]
                    // SAFETY: the failed `compare_exchange` means `new` was never
                    // published, so this thread is its unique owner and may free
                    // it exactly once.
                    drop(unsafe { Box::from_raw(new) });
                }
            }
        }
    }

    /// A reclamation handle sharing this cell's domain, for building further
    /// structures that must reclaim in lock-step with it.
    #[must_use]
    pub fn collector(&self) -> Collector {
        self.collector.clone()
    }

    /// Retires an unlinked box: defer its free until unobservable.
    fn retire(&self, old: *mut T)
    where
        T: Send + 'static,
    {
        let guard = pin_with(&self.collector);
        self.defer_free(&guard, old);
    }

    /// Defers freeing `old` through `guard`'s epoch domain.
    fn defer_free(&self, guard: &Guard, old: *mut T)
    where
        T: Send + 'static,
    {
        let old = SendPtr(old);
        guard.defer(move || {
            let old = old;
            #[expect(
                unsafe_code,
                reason = "old was unlinked and is freed once no reader can observe it"
            )]
            // SAFETY: `old.0` came from `Box::into_raw`, has been unlinked from
            // the cell, and the deferred closure runs only once every reader
            // that could observe it has unpinned, so rebuilding the box frees it
            // exactly once with no live borrow outstanding.
            drop(unsafe { Box::from_raw(old.0) });
        });
    }
}

impl<T> Drop for Rcu<T> {
    fn drop(&mut self) {
        // Exclusive access at drop: free the current box directly, no deferral
        // needed since nothing can observe it any more. Superseded versions were
        // already handed to the reclaimer and are freed as its epoch advances.
        let ptr = *self.ptr.get_mut();
        debug_assert!(!ptr.is_null(), "Rcu pointer is never null");
        #[expect(
            unsafe_code,
            reason = "drop has exclusive ownership of the currently-published box"
        )]
        // SAFETY: `&mut self` proves no reader or writer can race us, so the
        // published box is uniquely owned and freed exactly once here.
        drop(unsafe { Box::from_raw(ptr) });
    }
}

impl<T: Default> Default for Rcu<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T: fmt::Debug> fmt::Debug for Rcu<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Rcu").field("value", &*self.read()).finish()
    }
}

/// A consistent read snapshot of an [`Rcu`], borrowing the value it observed.
///
/// Deref to `&T`. The snapshot stays valid — and pins the reclaimer — for as
/// long as the guard is alive, so keep it short-lived on the hot path. It is
/// neither [`Send`] nor [`Sync`]: an epoch pin is bound to the thread that took
/// it (`_guard` carries that restriction).
pub struct RcuGuard<'a, T> {
    value: &'a T,
    /// Keeps the thread pinned so `value` cannot be reclaimed; its drop unpins.
    _guard: Guard,
}

impl<T> RcuGuard<'_, T> {
    /// Borrows the snapshotted value explicitly (same as deref).
    #[must_use]
    pub fn get(&self) -> &T {
        self.value
    }
}

impl<T> Deref for RcuGuard<'_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        self.value
    }
}

impl<T: fmt::Debug> fmt::Debug for RcuGuard<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("RcuGuard").field(&self.value).finish()
    }
}
