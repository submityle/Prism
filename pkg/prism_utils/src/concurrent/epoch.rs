//! Epoch-based reclamation (`EBR`).
//!
//! Lock-free structures cannot free a node the instant it is unlinked: another
//! thread may still be about to dereference it. Epoch-based reclamation (the
//! `crossbeam-epoch` form from design doc §24.2) solves this by *deferring*
//! each free until it is provably unobservable.
//!
//! ## The protocol
//! A global [epoch](Collector) counter advances through the cycle
//! `… → e → e + 1 → e + 2 → …`. Each participating thread registers a
//! [`LocalHandle`] and, before touching shared pointers, [`pin`](LocalHandle::pin)s
//! itself — announcing "I am active in the current epoch". A thread may only
//! advance the global epoch once **every** pinned participant has caught up to
//! the current epoch.
//!
//! Garbage retired while pinned at epoch `e` is bucketed by `e`. Because the
//! global epoch advances only when no thread lags, by the time the global epoch
//! reaches `e + 2` no pinned thread can still reference anything from epoch `e`,
//! so epoch-`e` garbage is safe to drop. Three buckets therefore suffice.
//!
//! This directly defeats the `ABA` problem for pointer-based lock-free
//! structures (see [`TreiberStack`](super::stack::TreiberStack)): a node's
//! memory is never recycled while any thread is pinned, so a stale pointer can
//! never be observed pointing at a *different* live node that merely reused the
//! same address.
//!
//! ## Conservative orderings
//! Pinning stores and the advance scan use sequentially-consistent ordering.
//! That is stronger than strictly necessary but matches the design doc's
//! "保守默认" stance for code whose bugs are rare and catastrophic.

extern crate alloc;

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt;
use core::marker::PhantomData;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

/// Number of epoch buckets. Three is the minimum that lets epoch `e` garbage be
/// reclaimed when the global epoch reaches `e + 2` while epoch `e + 1` is still
/// in flight.
const EPOCHS: usize = 3;

/// How many deferrals trigger an automatic collection attempt, amortising the
/// scan cost across many retirements.
const COLLECT_THRESHOLD: usize = 128;

/// A unit of deferred work (typically "free this node"), run once it is safe.
struct Deferred {
    task: Box<dyn FnOnce() + Send + 'static>,
}

impl Deferred {
    #[inline]
    fn run(self) {
        (self.task)();
    }
}

/// Shared reclamation state behind a [`Collector`].
struct Global {
    /// The current global epoch (monotonically increasing, compared by value).
    epoch: AtomicUsize,
    /// Registered participants. Guarded by a mutex: registration and the
    /// advance scan are rare relative to pin/defer, so a lock here is cheap.
    locals: Mutex<Vec<Arc<LocalInner>>>,
    /// Garbage buckets indexed by `epoch % EPOCHS`.
    bags: [Mutex<Vec<Deferred>>; EPOCHS],
    /// Running count of deferrals, used to amortise collection attempts.
    defer_count: AtomicUsize,
}

impl Global {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            epoch: AtomicUsize::new(0),
            locals: Mutex::new(Vec::new()),
            bags: [
                Mutex::new(Vec::new()),
                Mutex::new(Vec::new()),
                Mutex::new(Vec::new()),
            ],
            defer_count: AtomicUsize::new(0),
        })
    }

    /// Attempts to advance the global epoch and reclaim one bucket of garbage.
    ///
    /// Returns the garbage that became safe (so the caller can run it *after*
    /// releasing every lock), or an empty vector if the epoch could not be
    /// advanced.
    fn try_advance(&self) -> Vec<Deferred> {
        let global_epoch = self.epoch.load(Ordering::SeqCst);

        {
            let mut locals = lock(&self.locals);
            let mut i = 0;
            while i < locals.len() {
                let state = locals[i].epoch_state.load(Ordering::SeqCst);
                if state & 1 == 0 {
                    // Unpinned. If the owner retired, physically reap it now so
                    // dead threads never block future advances.
                    if locals[i].retired.load(Ordering::Acquire) {
                        locals.swap_remove(i);
                        continue;
                    }
                    i += 1;
                    continue;
                }
                // Pinned: if it lags the global epoch, we cannot advance yet.
                if (state >> 1) != global_epoch {
                    return Vec::new();
                }
                i += 1;
            }
        }

        // Every pinned participant is at `global_epoch`; try to advance.
        let next = global_epoch.wrapping_add(1);
        if self
            .epoch
            .compare_exchange(global_epoch, next, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            // Another thread advanced first; let it do the collecting.
            return Vec::new();
        }

        // Reaching `next` makes epoch `next - 2 == global_epoch - 1` garbage
        // safe. Its bucket index is `(global_epoch - 1) % EPOCHS`, which equals
        // `(next + 1) % EPOCHS`.
        let idx = next.wrapping_add(1) % EPOCHS;
        let mut bag = lock(&self.bags[idx]);
        core::mem::take(&mut *bag)
    }

    /// Runs a collection attempt and executes any reclaimed garbage.
    fn collect(&self) {
        let garbage = self.try_advance();
        for d in garbage {
            d.run();
        }
    }
}

impl Drop for Global {
    fn drop(&mut self) {
        // The collector and every handle/guard are gone, so all remaining
        // garbage is unobservable and must be run to honour deferred frees.
        for bag in &self.bags {
            let drained = core::mem::take(&mut *lock(bag));
            for d in drained {
                d.run();
            }
        }
    }
}

/// Per-thread participation record. Shared (via [`Arc`]) between the owning
/// [`LocalHandle`]/[`Guard`] and the global participant list, which reads
/// `epoch_state` during an advance scan.
struct LocalInner {
    /// Encoded state: bit 0 is the pinned flag, the remaining bits are the
    /// pinned epoch. `0` means unpinned.
    epoch_state: AtomicUsize,
    /// Pin nesting depth. Only ever mutated by the owning thread, but atomic so
    /// the record stays `Sync` for the shared participant list.
    guard_count: AtomicUsize,
    /// Set when the owning [`LocalHandle`] is dropped, so the advance scan can
    /// reap this record once it is also unpinned.
    retired: AtomicBool,
    /// Back-reference to the shared state.
    global: Arc<Global>,
}

/// A handle to the reclamation machinery, cheaply clonable (an [`Arc`] inside).
/// Clone it to share one reclamation domain across threads; each thread then
/// calls [`register`](Collector::register) for its own participation record.
#[derive(Clone)]
pub struct Collector {
    global: Arc<Global>,
}

impl Collector {
    /// Creates a fresh, independent reclamation domain.
    #[must_use]
    pub fn new() -> Self {
        Self {
            global: Global::new(),
        }
    }

    /// Registers the calling thread, returning its [`LocalHandle`]. Each thread
    /// that pins must hold its own handle.
    #[must_use]
    pub fn register(&self) -> LocalHandle {
        let inner = Arc::new(LocalInner {
            epoch_state: AtomicUsize::new(0),
            guard_count: AtomicUsize::new(0),
            retired: AtomicBool::new(false),
            global: Arc::clone(&self.global),
        });
        lock(&self.global.locals).push(Arc::clone(&inner));
        LocalHandle {
            inner,
            _not_sync: PhantomData,
        }
    }

    /// The current global epoch value. Primarily useful for tests and
    /// diagnostics.
    #[must_use]
    pub fn epoch(&self) -> usize {
        self.global.epoch.load(Ordering::SeqCst)
    }
}

impl Default for Collector {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for Collector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Collector")
            .field("epoch", &self.epoch())
            .finish()
    }
}

/// A thread's participation handle. Create [`Guard`]s from it with
/// [`pin`](LocalHandle::pin). It is `Send` (you may move it onto a worker
/// thread) but intentionally not `Sync` (it must not be shared between threads
/// at once).
pub struct LocalHandle {
    inner: Arc<LocalInner>,
    /// Makes the handle `!Sync`: pin bookkeeping assumes a single accessor.
    _not_sync: PhantomData<core::cell::Cell<()>>,
}

impl LocalHandle {
    /// Pins the calling thread, returning a [`Guard`]. While any guard is held
    /// the thread is announced as active in the current epoch, so no epoch it
    /// can observe will be reclaimed. Pins nest: inner pins are cheap no-ops.
    #[must_use]
    pub fn pin(&self) -> Guard {
        let inner = &*self.inner;
        let count = inner.guard_count.load(Ordering::Relaxed);
        if count == 0 {
            // First pin on this thread: announce the current global epoch.
            let global_epoch = inner.global.epoch.load(Ordering::SeqCst);
            inner
                .epoch_state
                .store((global_epoch << 1) | 1, Ordering::SeqCst);
        }
        inner.guard_count.store(count + 1, Ordering::Relaxed);
        Guard {
            local: Arc::clone(&self.inner),
            _not_send: PhantomData,
        }
    }

    /// Whether the thread is currently pinned (any guard outstanding).
    #[must_use]
    pub fn is_pinned(&self) -> bool {
        self.inner.guard_count.load(Ordering::Relaxed) > 0
    }
}

impl Drop for LocalHandle {
    fn drop(&mut self) {
        // Mark the record retired; the next advance scan reaps it once it is
        // unpinned. (A still-pinned record keeps blocking advances, as it must.)
        self.inner.retired.store(true, Ordering::Release);
    }
}

impl fmt::Debug for LocalHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LocalHandle")
            .field("pinned", &self.is_pinned())
            .finish()
    }
}

/// Proof that the owning thread is pinned. Dropping it unpins (once nesting
/// reaches zero). It is neither `Send` nor `Sync`: a pin is bound to the thread
/// that created it.
pub struct Guard {
    local: Arc<LocalInner>,
    /// Makes the guard `!Send + !Sync`.
    _not_send: PhantomData<*const ()>,
}

impl Guard {
    /// Defers `f` until no thread pinned in the current (or an earlier) epoch
    /// can still observe whatever `f` reclaims. `f` runs at most once, possibly
    /// on another thread, hence the `Send + 'static` bound.
    pub fn defer(&self, f: impl FnOnce() + Send + 'static) {
        let inner = &*self.local;
        let state = inner.epoch_state.load(Ordering::SeqCst);
        debug_assert!(state & 1 == 1, "defer requires an active pin");
        let epoch = state >> 1;
        {
            let mut bag = lock(&inner.global.bags[epoch % EPOCHS]);
            bag.push(Deferred { task: Box::new(f) });
        }
        // Amortised collection: every COLLECT_THRESHOLD deferrals, try to
        // advance the epoch and run anything that became safe.
        let n = inner.global.defer_count.fetch_add(1, Ordering::Relaxed) + 1;
        if n.is_multiple_of(COLLECT_THRESHOLD) {
            inner.global.collect();
        }
    }

    /// Forces a collection attempt: advance the epoch if possible and run any
    /// garbage that became safe. Making forward progress never requires this,
    /// but it is useful to bound memory in tests and quiescent periods.
    pub fn flush(&self) {
        self.local.global.collect();
    }

    /// The epoch this guard is pinned at.
    #[must_use]
    pub fn epoch(&self) -> usize {
        self.local.epoch_state.load(Ordering::SeqCst) >> 1
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        let inner = &*self.local;
        let count = inner.guard_count.load(Ordering::Relaxed);
        debug_assert!(count > 0, "guard count underflow");
        let count = count - 1;
        inner.guard_count.store(count, Ordering::Relaxed);
        if count == 0 {
            // Last pin released: announce unpinned so advances can proceed.
            inner.epoch_state.store(0, Ordering::SeqCst);
        }
    }
}

impl fmt::Debug for Guard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Guard")
            .field("epoch", &self.epoch())
            .finish()
    }
}

/// Locks a mutex, recovering transparently from poisoning so a panic in one
/// participant cannot wedge the whole reclaimer.
#[inline]
fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The process-wide default reclamation domain, used by the free
/// [`pin`] function and by [`TreiberStack::new`](super::stack::TreiberStack::new).
fn default_collector() -> &'static Collector {
    static DEFAULT: OnceLock<Collector> = OnceLock::new();
    DEFAULT.get_or_init(Collector::new)
}

std::thread_local! {
    /// Each thread's handle into the [`default_collector`], created on first use.
    static DEFAULT_HANDLE: LocalHandle = default_collector().register();
}

/// Returns a clone of the process-wide default [`Collector`].
#[must_use]
pub fn collector() -> Collector {
    default_collector().clone()
}

/// Pins the calling thread in the process-wide default reclamation domain.
///
/// This is the ergonomic entry point used by structures that do not manage
/// their own [`Collector`].
#[must_use]
pub fn pin() -> Guard {
    DEFAULT_HANDLE.with(LocalHandle::pin)
}

/// Pins the calling thread in a specific [`Collector`]'s domain, lazily
/// creating (and caching) a per-thread [`LocalHandle`] for that collector.
///
/// This is what per-structure reclamation domains (e.g.
/// [`TreiberStack`](super::stack::TreiberStack)) use so their reclamation does
/// not interleave with the process-wide default domain.
#[must_use]
pub fn pin_with(collector: &Collector) -> Guard {
    std::thread_local! {
        /// Cached handles, one per collector this thread has pinned in.
        static HANDLES: core::cell::RefCell<Vec<(usize, LocalHandle)>> =
            const { core::cell::RefCell::new(Vec::new()) };
    }
    let id = Arc::as_ptr(&collector.global) as usize;
    HANDLES.with(|cell| {
        {
            let handles = cell.borrow();
            if let Some((_, handle)) = handles.iter().find(|(cid, _)| *cid == id) {
                return handle.pin();
            }
        }
        let handle = collector.register();
        let guard = handle.pin();
        cell.borrow_mut().push((id, handle));
        guard
    })
}
