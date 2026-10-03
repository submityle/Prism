//! A registry-based typed thread-local store.
//!
//! Unlike the [`std::thread_local!`] macro, [`ThreadLocal`] is a *value* you
//! can own, pass around (behind an [`std::sync::Arc`], say), and drop. Each
//! thread that touches it gets its own `T`, created lazily on first access.
//!
//! Access is mediated by a single [`std::sync::Mutex`] guarding a map from
//! [`std::thread::ThreadId`] to a per-thread box. The lock is held only while
//! looking up (and, for [`ThreadLocal::with`], operating on) a thread's own
//! slot; distinct threads never share a slot, so values need only be [`Send`].
//! The design is fully safe — no `unsafe` — at the cost of serializing slot
//! access through the registry lock.

use std::collections::HashMap;
use std::sync::Mutex;
use std::thread::ThreadId;

/// A typed thread-local store with one independently owned `T` per thread.
#[derive(Debug)]
pub struct ThreadLocal<T: Send> {
    slots: Mutex<HashMap<ThreadId, Box<T>>>,
}

impl<T: Send> ThreadLocal<T> {
    /// Create an empty store. The first access on each thread initializes that
    /// thread's slot.
    pub fn new() -> Self {
        Self {
            slots: Mutex::new(HashMap::new()),
        }
    }

    /// Run `f` against the calling thread's slot, creating it with `init` if
    /// this thread has not touched the store yet.
    ///
    /// The registry lock is held for the duration of `f`; `f` must not call
    /// back into the same [`ThreadLocal`] or it will deadlock.
    pub fn with<R>(&self, init: impl FnOnce() -> T, f: impl FnOnce(&mut T) -> R) -> R {
        let mut slots = self.slots.lock().expect("ThreadLocal registry poisoned");
        let slot = slots
            .entry(std::thread::current().id())
            .or_insert_with(|| Box::new(init()));
        f(slot)
    }

    /// Returns `true` if the calling thread already has an initialized slot.
    pub fn is_set(&self) -> bool {
        let slots = self.slots.lock().expect("ThreadLocal registry poisoned");
        slots.contains_key(&std::thread::current().id())
    }

    /// Remove and drop the calling thread's slot, returning whether one existed.
    pub fn clear(&self) -> bool {
        let mut slots = self.slots.lock().expect("ThreadLocal registry poisoned");
        slots.remove(&std::thread::current().id()).is_some()
    }

    /// Number of threads that currently hold a slot.
    pub fn len(&self) -> usize {
        self.slots
            .lock()
            .expect("ThreadLocal registry poisoned")
            .len()
    }

    /// Returns `true` if no thread holds a slot.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T: Send + Clone> ThreadLocal<T> {
    /// Return a clone of the calling thread's value, creating it with `init`
    /// first if needed.
    pub fn get_or(&self, init: impl FnOnce() -> T) -> T {
        self.with(init, |value| value.clone())
    }

    /// Overwrite the calling thread's slot with `value`.
    pub fn set(&self, value: T) {
        let mut slots = self.slots.lock().expect("ThreadLocal registry poisoned");
        slots.insert(std::thread::current().id(), Box::new(value));
    }
}

impl<T: Send> Default for ThreadLocal<T> {
    fn default() -> Self {
        Self::new()
    }
}
