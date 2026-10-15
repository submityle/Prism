//! Writable reactive source values.

use alloc::rc::Rc;
use core::cell::RefCell;

use crate::invalidation::{SignalChangeQueue, SignalSubscribers, SignalSubscription};
use crate::node::{NodeId, NodeKind, State};
use crate::runtime::Runtime;

/// A reactive container for a value of type `T`.
///
/// Reading via [`Signal::get`]/[`Signal::with`] inside a memo or effect records
/// a dependency; writing via [`Signal::set`]/[`Signal::update`] notifies every
/// dependent. A `Signal` is a cheap, clonable handle: clones refer to the same
/// underlying value and graph node.
pub struct Signal<T: 'static> {
    runtime: Runtime,
    id: NodeId,
    value: Rc<RefCell<T>>,
    subscribers: Rc<RefCell<SignalSubscribers>>,
}

impl<T: 'static> Clone for Signal<T> {
    fn clone(&self) -> Self {
        Self {
            runtime: self.runtime.clone(),
            id: self.id,
            value: self.value.clone(),
            subscribers: self.subscribers.clone(),
        }
    }
}

impl<T: 'static> Signal<T> {
    pub(crate) fn new(runtime: &Runtime, value: T) -> Self {
        let id = runtime.insert_node(NodeKind::Signal, State::Clean, None);
        Self {
            runtime: runtime.clone(),
            id,
            value: Rc::new(RefCell::new(value)),
            subscribers: Rc::default(),
        }
    }

    /// The runtime this signal belongs to.
    pub fn runtime(&self) -> &Runtime {
        &self.runtime
    }

    /// Subscribe to value-free write invalidations, including one initial
    /// notification. Repeated writes coalesce in the queue, independently of
    /// reactive batching. Drop the lease to detach. Notifications are enqueued
    /// before effects run, including when a mutation unwinds after writing.
    pub fn subscribe_changes(&self, queue: &SignalChangeQueue) -> SignalSubscription {
        SignalSubscribers::subscribe(&self.subscribers, queue)
    }

    /// Read a clone of the value, recording a dependency on this signal.
    pub fn get(&self) -> T
    where
        T: Clone,
    {
        self.runtime.track(self.id);
        self.value.borrow().clone()
    }

    /// Borrow the value through `f`, recording a dependency on this signal.
    pub fn with<R>(&self, f: impl FnOnce(&T) -> R) -> R {
        self.runtime.track(self.id);
        f(&self.value.borrow())
    }

    /// Read a clone of the value without recording a dependency.
    pub fn get_untracked(&self) -> T
    where
        T: Clone,
    {
        self.value.borrow().clone()
    }

    /// Borrow the value through `f` without recording a dependency.
    pub fn with_untracked<R>(&self, f: impl FnOnce(&T) -> R) -> R {
        f(&self.value.borrow())
    }

    /// Replace the value and notify dependents unconditionally.
    pub fn set(&self, value: T) {
        {
            let mut current = self.value.borrow_mut();
            let _notify = NotifyOnDrop(&self.subscribers);
            *current = value;
        }
        self.runtime.notify_changed(self.id);
    }

    /// Mutate the value in place through `f`, then notify dependents. If `f`
    /// panics, deferred change queues are still invalidated (the value may have
    /// changed); reactive effects are not executed while unwinding.
    pub fn update(&self, f: impl FnOnce(&mut T)) {
        {
            let mut current = self.value.borrow_mut();
            let _notify = NotifyOnDrop(&self.subscribers);
            f(&mut current);
        }
        self.runtime.notify_changed(self.id);
    }

    /// Replace the value only when it differs from the current one, avoiding a
    /// spurious notification. Returns `true` when a change was propagated.
    pub fn set_if_changed(&self, value: T) -> bool
    where
        T: PartialEq,
    {
        {
            let current = self.value.borrow();
            if *current == value {
                return false;
            }
        }
        self.set(value);
        true
    }
}

// Only queue bookkeeping runs from Drop: no user callbacks or value access.
struct NotifyOnDrop<'a>(&'a RefCell<SignalSubscribers>);

impl Drop for NotifyOnDrop<'_> {
    fn drop(&mut self) {
        self.0.borrow().notify();
    }
}

impl Runtime {
    /// Create a new [`Signal`] seeded with `value`.
    pub fn signal<T: 'static>(&self, value: T) -> Signal<T> {
        Signal::new(self, value)
    }
}
