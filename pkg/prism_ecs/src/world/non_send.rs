//! Non-`Send` resources: thread-local singletons a [`World`](crate::world::World)
//! owns but that may not be moved or accessed off their origin thread
//! (design §16 / §18 bevy-compat surface).
//!
//! A [`Resource`](crate::resource::Resource) is `Send + Sync`, which lets the
//! parallel executor touch it from any worker. Some values — a GPU device
//! handle, a windowing context, a `Box<dyn Fn>` reactive bridge
//! (`prism_ui_ecs`) — are intrinsically **not** `Send`. They still need a home
//! inside the world, so this module provides a type-erased, thread-origin-guarded
//! store for them.
//!
//! # Soundness
//!
//! The store holds `Box<dyn Any>` values that are *not* `Send`/`Sync`. To keep
//! [`World`](crate::world::World) itself `Send + Sync` (so it can be moved
//! between threads at synchronization points and shared behind the
//! [`UnsafeWorldCell`](crate::system::world_cell::UnsafeWorldCell) raw-pointer
//! discipline), [`NonSendResources`] carries an `unsafe impl Send + Sync`. That
//! impl is justified by a hard runtime invariant: **every** path that observes
//! or mutates a stored value — insertion, lookup, removal, and `Drop` — first
//! asserts it runs on the thread that originally inserted a value. A value is
//! therefore never read, written, or dropped from any thread other than its
//! origin, which is exactly the guarantee `!Send`/`!Sync` types require. Moving
//! the (type-erased) handle across threads is sound because no access to the
//! inner value can occur there without tripping the origin assertion first.
//!
//! This is `std`-only: thread identity (`std::thread::ThreadId`) is the guard,
//! and non-`Send` singletons are a `std`-world concept.

use alloc::boxed::Box;
use core::any::{type_name, Any, TypeId};
use std::thread::ThreadId;

use crate::collections::HashMap;

/// A type-erased, thread-origin-guarded store for non-`Send` resources.
///
/// Keyed by [`TypeId`]; at most one value per type. The first insertion latches
/// the origin thread; all later access must occur on that same thread.
#[derive(Default)]
pub(crate) struct NonSendResources {
    /// The thread that owns the stored values. `None` until the first value is
    /// inserted (an empty store is trivially safe to touch from any thread).
    origin: Option<ThreadId>,
    map: HashMap<TypeId, Box<dyn Any>>,
}

// SAFETY: `NonSendResources` type-erases values that may be neither `Send` nor
// `Sync`. Every observing or mutating method (`insert`, `get`, `get_mut`,
// `remove`, `contains`, `len`) and the `Drop` impl first calls
// `assert_origin`, which panics unless the caller runs on the thread that
// inserted the first value. Consequently no inner value is ever accessed or
// dropped off its origin thread, so transferring the (opaque) store handle
// between threads — the only thing these auto-trait impls enable — cannot
// create a data race or an off-thread drop. An empty store latches no origin
// and owns nothing to misuse.
unsafe impl Send for NonSendResources {}
// SAFETY: see the `Send` justification above — shared references only reach the
// inner values through origin-guarded methods, so `&NonSendResources` cannot be
// used to touch a `!Sync` value from a foreign thread.
unsafe impl Sync for NonSendResources {}

impl NonSendResources {
    /// Create an empty store.
    #[inline]
    pub(crate) fn new() -> Self {
        Self {
            origin: None,
            map: HashMap::default(),
        }
    }

    /// Panic unless the current thread is the store's origin thread. Latches
    /// the origin on first use.
    #[inline]
    fn assert_origin(&mut self) {
        let current = std::thread::current().id();
        match self.origin {
            Some(origin) => assert!(
                origin == current,
                "non-send resource accessed off its origin thread (origin={:?}, current={:?})",
                origin,
                current,
            ),
            None => self.origin = Some(current),
        }
    }

    /// Panic unless the current thread is the origin, for shared-reference
    /// paths. A store with no latched origin (empty) is always allowed.
    #[inline]
    fn assert_origin_shared(&self) {
        if let Some(origin) = self.origin {
            let current = std::thread::current().id();
            assert!(
                origin == current,
                "non-send resource accessed off its origin thread (origin={:?}, current={:?})",
                origin,
                current,
            );
        }
    }

    /// Insert (or replace) the non-`Send` value of type `T`, returning the
    /// previous value if one was present.
    pub(crate) fn insert<T: 'static>(&mut self, value: T) -> Option<T> {
        self.assert_origin();
        let prev = self.map.insert(TypeId::of::<T>(), Box::new(value));
        prev.map(|boxed| {
            *boxed.downcast::<T>().unwrap_or_else(|_| {
                panic!("non-send slot for {} held the wrong type", type_name::<T>())
            })
        })
    }

    /// Shared borrow of the non-`Send` value of type `T`.
    pub(crate) fn get<T: 'static>(&self) -> Option<&T> {
        self.assert_origin_shared();
        self.map.get(&TypeId::of::<T>()).map(|boxed| {
            boxed.downcast_ref::<T>().unwrap_or_else(|| {
                panic!("non-send slot for {} held the wrong type", type_name::<T>())
            })
        })
    }

    /// Exclusive borrow of the non-`Send` value of type `T`.
    pub(crate) fn get_mut<T: 'static>(&mut self) -> Option<&mut T> {
        self.assert_origin();
        self.map.get_mut(&TypeId::of::<T>()).map(|boxed| {
            boxed.downcast_mut::<T>().unwrap_or_else(|| {
                panic!("non-send slot for {} held the wrong type", type_name::<T>())
            })
        })
    }

    /// Remove and return the non-`Send` value of type `T`, if present.
    pub(crate) fn remove<T: 'static>(&mut self) -> Option<T> {
        self.assert_origin();
        self.map.remove(&TypeId::of::<T>()).map(|boxed| {
            *boxed.downcast::<T>().unwrap_or_else(|_| {
                panic!("non-send slot for {} held the wrong type", type_name::<T>())
            })
        })
    }

    /// Whether a non-`Send` value of type `T` is currently stored.
    #[inline]
    pub(crate) fn contains<T: 'static>(&self) -> bool {
        self.assert_origin_shared();
        self.map.contains_key(&TypeId::of::<T>())
    }
}

impl Drop for NonSendResources {
    fn drop(&mut self) {
        // Dropping the stored `!Send` values off their origin thread would be
        // unsound. If any value remains, enforce the origin invariant; an empty
        // store (no latched origin) is safe to drop anywhere.
        if !self.map.is_empty()
            && let Some(origin) = self.origin
        {
            let current = std::thread::current().id();
            assert!(
                origin == current,
                "non-send resources dropped off their origin thread (origin={:?}, current={:?})",
                origin,
                current,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_get_remove_roundtrip() {
        let mut store = NonSendResources::new();
        assert!(!store.contains::<u32>());
        assert_eq!(store.insert(7u32), None);
        assert!(store.contains::<u32>());
        assert_eq!(store.get::<u32>().copied(), Some(7));
        *store.get_mut::<u32>().unwrap() = 9;
        assert_eq!(store.get::<u32>().copied(), Some(9));
        assert_eq!(store.insert(10u32), Some(9));
        assert_eq!(store.remove::<u32>(), Some(10));
        assert!(!store.contains::<u32>());
    }

    #[test]
    fn distinct_types_are_independent() {
        let mut store = NonSendResources::new();
        store.insert(1u8);
        store.insert(2u16);
        assert_eq!(store.get::<u8>().copied(), Some(1));
        assert_eq!(store.get::<u16>().copied(), Some(2));
        assert_eq!(store.remove::<u8>(), Some(1));
        assert!(store.contains::<u16>());
    }

    #[test]
    fn holds_non_send_value() {
        // A `Box<dyn Fn>` is neither `Send` nor `Sync`; the store must accept it.
        let mut store = NonSendResources::new();
        let f: Box<dyn Fn() -> i32> = Box::new(|| 42);
        store.insert(f);
        assert_eq!((store.get::<Box<dyn Fn() -> i32>>().unwrap())(), 42);
    }
}
