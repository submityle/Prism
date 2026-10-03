//! Lightweight synchronization primitives.
//!
//! - [`SpinLock`]: a mutex that busy-waits (with [`Backoff`]) instead of
//!   blocking the OS thread, for very short critical sections. It layers over
//!   [`std::sync::Mutex`]'s userspace fast path, so the whole type is safe —
//!   no `unsafe`, no hand-rolled `UnsafeCell`.
//! - [`Backoff`]: an exponential spin/yield helper for hand-written wait loops.
//! - [`Once`]: a one-time initializer (thin wrapper over [`std::sync::Once`]).

use std::sync::{Mutex, MutexGuard, TryLockError};

/// Upper bound (as a power of two) on pure spin iterations before yielding.
const SPIN_LIMIT: u32 = 6;
/// Upper bound (as a power of two) on backoff steps before completion.
const YIELD_LIMIT: u32 = 10;

/// Exponential backoff helper for spin-wait loops.
///
/// Early calls emit a growing number of CPU spin hints ([`crate::atomics::spin_hint`]);
/// once the spin budget is exhausted it yields the thread to the scheduler.
#[derive(Clone, Debug)]
pub struct Backoff {
    step: u32,
}

impl Backoff {
    /// Create a fresh backoff at step zero.
    pub const fn new() -> Self {
        Self { step: 0 }
    }

    /// Reset the backoff to its initial state.
    pub fn reset(&mut self) {
        self.step = 0;
    }

    /// Emit an exponentially growing burst of CPU spin hints. Use when the
    /// awaited event is expected imminently and yielding would be wasteful.
    pub fn spin(&mut self) {
        for _ in 0..(1u32 << self.step.min(SPIN_LIMIT)) {
            crate::atomics::spin_hint();
        }
        if self.step <= SPIN_LIMIT {
            self.step += 1;
        }
    }

    /// Back off, spinning while cheap and yielding to the scheduler once the
    /// spin budget is spent. Use in general-purpose wait loops.
    pub fn snooze(&mut self) {
        if self.step <= SPIN_LIMIT {
            for _ in 0..(1u32 << self.step) {
                crate::atomics::spin_hint();
            }
        } else {
            std::thread::yield_now();
        }
        if self.step <= YIELD_LIMIT {
            self.step += 1;
        }
    }

    /// Returns `true` once further [`Backoff::snooze`] calls will only yield,
    /// signalling that a blocking wait would be more appropriate.
    pub fn is_completed(&self) -> bool {
        self.step > YIELD_LIMIT
    }
}

impl Default for Backoff {
    fn default() -> Self {
        Self::new()
    }
}

/// A spin lock protecting a value of type `T`.
///
/// Acquiring busy-waits (via [`Backoff`]) on the underlying userspace lock word
/// instead of parking the thread, which is cheaper than an OS mutex for very
/// short critical sections but wastes cycles if held long. Prefer a blocking
/// mutex when the critical section may be long or contended for a long time.
#[derive(Debug, Default)]
pub struct SpinLock<T> {
    inner: Mutex<T>,
}

impl<T> SpinLock<T> {
    /// Create a new spin lock wrapping `value`.
    pub const fn new(value: T) -> Self {
        Self {
            inner: Mutex::new(value),
        }
    }

    /// Acquire the lock, busy-waiting until it is free, and return a guard.
    pub fn lock(&self) -> SpinLockGuard<'_, T> {
        let mut backoff = Backoff::new();
        loop {
            match self.inner.try_lock() {
                Ok(guard) => return SpinLockGuard { guard },
                Err(TryLockError::WouldBlock) => backoff.snooze(),
                // A previous holder panicked; recover the data and continue.
                Err(TryLockError::Poisoned(poison)) => {
                    return SpinLockGuard {
                        guard: poison.into_inner(),
                    };
                }
            }
        }
    }

    /// Try to acquire the lock without waiting, returning [`None`] if held.
    pub fn try_lock(&self) -> Option<SpinLockGuard<'_, T>> {
        match self.inner.try_lock() {
            Ok(guard) => Some(SpinLockGuard { guard }),
            Err(TryLockError::WouldBlock) => None,
            Err(TryLockError::Poisoned(poison)) => Some(SpinLockGuard {
                guard: poison.into_inner(),
            }),
        }
    }

    /// Consume the lock and return the protected value.
    pub fn into_inner(self) -> T {
        self.inner.into_inner().unwrap_or_else(PoisonRecover::recover)
    }

    /// Borrow the protected value mutably without locking (statically unique).
    pub fn get_mut(&mut self) -> &mut T {
        self.inner.get_mut().unwrap_or_else(PoisonRecover::recover)
    }
}

/// Small helper to recover data from a poisoned [`std::sync::Mutex`] without
/// panicking, used by [`SpinLock::into_inner`] and [`SpinLock::get_mut`].
trait PoisonRecover<T> {
    fn recover(self) -> T;
}

impl<T> PoisonRecover<T> for std::sync::PoisonError<T> {
    fn recover(self) -> T {
        self.into_inner()
    }
}

/// RAII guard granting access to a [`SpinLock`]'s value; releases on drop.
#[derive(Debug)]
pub struct SpinLockGuard<'a, T> {
    guard: MutexGuard<'a, T>,
}

impl<T> core::ops::Deref for SpinLockGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.guard
    }
}

impl<T> core::ops::DerefMut for SpinLockGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.guard
    }
}

/// A one-time initializer: the first [`Once::call_once`] runs its closure, and
/// all callers block until that initialization completes.
///
/// Thin wrapper over [`std::sync::Once`] exposed through the platform surface.
#[derive(Debug)]
pub struct Once {
    inner: std::sync::Once,
}

impl Once {
    /// Create a new, not-yet-run initializer.
    pub const fn new() -> Self {
        Self {
            inner: std::sync::Once::new(),
        }
    }

    /// Run `f` exactly once across all threads; later calls return after the
    /// first completes.
    pub fn call_once<F: FnOnce()>(&self, f: F) {
        self.inner.call_once(f);
    }

    /// Returns `true` once the initializer's closure has finished running.
    pub fn is_completed(&self) -> bool {
        self.inner.is_completed()
    }
}

impl Default for Once {
    fn default() -> Self {
        Self::new()
    }
}
