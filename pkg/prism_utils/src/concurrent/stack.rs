//! Lock-free Treiber stack with epoch-based node reclamation.
//!
//! A [`TreiberStack`] is the canonical lock-free `LIFO` stack: push and pop are
//! a single `compare_exchange` on the head pointer. Its reason for living here
//! is to be the worked example (and stress target) proving the [`epoch`]
//! reclaimer correct: popped nodes are *deferred* through a [`Guard`], never
//! freed inline, so no thread can dereference a node whose memory has been
//! recycled — which is exactly how epoch reclamation defeats the `ABA` problem.
//!
//! [`epoch`]: super::epoch
//! [`Guard`]: super::epoch::Guard

extern crate alloc;

use alloc::boxed::Box;
use core::fmt;
use core::mem::ManuallyDrop;
use core::ptr;
use core::sync::atomic::{AtomicPtr, Ordering};

use super::epoch::{pin_with, Collector};

/// A stack node. `value` is wrapped in [`ManuallyDrop`] because its ownership is
/// handed to the popper via `ptr::read`-style extraction before the node is
/// reclaimed, so the node's own destructor must not drop it a second time.
struct Node<T> {
    value: ManuallyDrop<T>,
    next: *mut Node<T>,
}

/// A `Send` wrapper around a node pointer so a deferred free closure (which may
/// run on another thread) can own it. Sound exactly when the node payload may
/// itself cross threads.
struct SendPtr<T>(*mut Node<T>);

// SAFETY: the pointer names a node this thread has exclusively unlinked from
// the stack; moving responsibility for freeing it to another thread is sound
// whenever the payload may cross threads (`T: Send`).
#[expect(
    unsafe_code,
    reason = "an unlinked node may be reclaimed on another thread when T: Send"
)]
// SAFETY: an unlinked node may be reclaimed on another thread when T: Send
unsafe impl<T: Send> Send for SendPtr<T> {}

/// A lock-free multi-producer / multi-consumer `LIFO` stack.
pub struct TreiberStack<T> {
    head: AtomicPtr<Node<T>>,
    collector: Collector,
}

// SAFETY: all shared mutation goes through the atomic `head`, and payloads are
// only ever moved (never shared by reference) across threads, so `T: Send`
// suffices for the stack to be both `Send` and `Sync`.
#[expect(
    unsafe_code,
    reason = "shared access is mediated by the atomic head; payloads only move across threads"
)]
// SAFETY: shared access is mediated by the atomic head; payloads only move across threads
unsafe impl<T: Send> Send for TreiberStack<T> {}
// SAFETY: see the `Send` impl.
#[expect(
    unsafe_code,
    reason = "shared access is mediated by the atomic head; payloads only move across threads"
)]
// SAFETY: shared access is mediated by the atomic head; payloads only move across threads
unsafe impl<T: Send> Sync for TreiberStack<T> {}

impl<T> TreiberStack<T> {
    /// Creates an empty stack with its own private reclamation domain.
    #[must_use]
    pub fn new() -> Self {
        Self {
            head: AtomicPtr::new(ptr::null_mut()),
            collector: Collector::new(),
        }
    }

    /// Creates an empty stack that reclaims through the given [`Collector`],
    /// letting several structures share one reclamation domain.
    #[must_use]
    pub fn with_collector(collector: Collector) -> Self {
        Self {
            head: AtomicPtr::new(ptr::null_mut()),
            collector,
        }
    }

    /// Pushes `value` onto the top of the stack.
    pub fn push(&self, value: T) {
        let node = Box::into_raw(Box::new(Node {
            value: ManuallyDrop::new(value),
            next: ptr::null_mut(),
        }));
        // Pushing never dereferences existing nodes, so no pin is required.
        loop {
            let head = self.head.load(Ordering::Acquire);
            #[expect(
                unsafe_code,
                reason = "the new node is owned solely by this thread until it is published"
            )]
            // SAFETY: `node` was just allocated here and is not yet reachable by
            // any other thread, so we may freely write its `next` link.
            unsafe {
                (*node).next = head;
            }
            if self
                .head
                .compare_exchange_weak(head, node, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return;
            }
        }
    }

    /// Pops the top element, returning `None` if the stack is empty.
    pub fn pop(&self) -> Option<T>
    where
        T: Send + 'static,
    {
        // Pin for the whole operation: this keeps every node we might observe
        // from being reclaimed out from under us.
        let guard = pin_with(&self.collector);
        loop {
            let head = self.head.load(Ordering::Acquire);
            if head.is_null() {
                return None;
            }
            #[expect(
                unsafe_code,
                reason = "the pin guarantees head stays allocated while we read its next link"
            )]
            // SAFETY: we are pinned, so `head` cannot have been reclaimed since
            // we loaded it; reading its `next` link is therefore valid.
            let next = unsafe { (*head).next };
            if self
                .head
                .compare_exchange_weak(head, next, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                #[expect(
                    unsafe_code,
                    reason = "winning the CAS gives this thread exclusive ownership of the node"
                )]
                // SAFETY: winning the `compare_exchange` unlinks `head` and
                // makes this thread its unique owner, so moving the value out is
                // sound and happens exactly once.
                let value = unsafe { ManuallyDrop::take(&mut (*head).value) };
                let ptr = SendPtr(head);
                guard.defer(move || {
                    let ptr = ptr;
                    #[expect(
                        unsafe_code,
                        reason = "the node was unlinked and its value already moved out"
                    )]
                    // SAFETY: `ptr.0` came from `Box::into_raw`, has been
                    // unlinked, and its value was already taken, so rebuilding
                    // the box frees exactly the node once it is unobservable.
                    unsafe {
                        drop(Box::from_raw(ptr.0));
                    }
                });
                return Some(value);
            }
        }
    }

    /// Returns `true` if the stack currently has no elements. Momentary in a
    /// concurrent setting.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.head.load(Ordering::Acquire).is_null()
    }

    /// A reclamation handle sharing this stack's domain, for building further
    /// structures that must reclaim in lock-step with it.
    #[must_use]
    pub fn collector(&self) -> Collector {
        self.collector.clone()
    }
}

impl<T> Default for TreiberStack<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Drop for TreiberStack<T> {
    fn drop(&mut self) {
        // Exclusive access at drop: walk the chain, dropping each payload and
        // freeing each node directly (no deferral needed, nothing can observe
        // them any more).
        let mut head = *self.head.get_mut();
        while !head.is_null() {
            #[expect(
                unsafe_code,
                reason = "drop has exclusive ownership of the remaining node chain"
            )]
            // SAFETY: at drop no other thread holds the stack, so each linked
            // node is uniquely owned; we read its `next`, drop its still-live
            // value, and free the box exactly once.
            let next = unsafe {
                let mut boxed = Box::from_raw(head);
                let next = boxed.next;
                ManuallyDrop::drop(&mut boxed.value);
                next
            };
            head = next;
        }
    }
}

impl<T> fmt::Debug for TreiberStack<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TreiberStack")
            .field("empty", &self.is_empty())
            .finish()
    }
}
