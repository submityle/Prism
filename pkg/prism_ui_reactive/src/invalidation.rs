//! Value-free signal invalidations for deferred consumers.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::rc::{Rc, Weak};
use core::cell::RefCell;

/// Queue-local identity of one live signal subscription.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SignalChangeId(u64);

#[derive(Default)]
struct QueueInner {
    next: u64,
    live: BTreeSet<SignalChangeId>,
    pending: BTreeSet<SignalChangeId>,
}

/// Coalesced, registration-ordered signal invalidations. No values are read or
/// cloned and no user callbacks run when a signal enqueues an invalidation.
/// Clones refer to the same single-threaded queue.
#[derive(Clone, Default)]
pub struct SignalChangeQueue(Rc<RefCell<QueueInner>>);

impl SignalChangeQueue {
    /// Create an empty queue.
    pub fn new() -> Self {
        Self::default()
    }

    /// Take the currently pending batch. Writes during processing belong to
    /// the next batch, even if the same subscription is still in this batch.
    /// Dropping the batch requeues its unacknowledged, still-live entries.
    pub fn take_pending(&self) -> SignalChangeBatch {
        SignalChangeBatch {
            queue: self.clone(),
            pending: core::mem::take(&mut self.0.borrow_mut().pending),
        }
    }
}

/// An owned batch that retains unfinished work across unwinding.
pub struct SignalChangeBatch {
    queue: SignalChangeQueue,
    pending: BTreeSet<SignalChangeId>,
}

impl SignalChangeBatch {
    /// The earliest unacknowledged subscription, if any. A subscription may
    /// have been dropped since capture; consumers should ignore unknown IDs.
    pub fn first(&self) -> Option<SignalChangeId> {
        self.pending.first().copied()
    }

    /// Complete one entry only after its consumer has returned successfully.
    /// This does not discard a new notification enqueued during processing.
    pub fn acknowledge(&mut self, id: SignalChangeId) {
        self.pending.remove(&id);
    }
}

impl Drop for SignalChangeBatch {
    fn drop(&mut self) {
        let mut queue = self.queue.0.borrow_mut();
        for id in &self.pending {
            if queue.live.contains(id) {
                queue.pending.insert(*id);
            }
        }
    }
}

#[derive(Default)]
pub(crate) struct SignalSubscribers {
    next: u64,
    routes: BTreeMap<u64, Weak<SubscriptionInner>>,
}

impl SignalSubscribers {
    pub(crate) fn subscribe(
        source: &Rc<RefCell<Self>>,
        queue: &SignalChangeQueue,
    ) -> SignalSubscription {
        let mut source_routes = source.borrow_mut();
        let source_id = source_routes.next;
        let next_source = source_id
            .checked_add(1)
            .expect("signal subscriptions exhausted");
        let mut queue_inner = queue.0.borrow_mut();
        let id = SignalChangeId(queue_inner.next);
        let next_queue = queue_inner
            .next
            .checked_add(1)
            .expect("signal queue exhausted");
        source_routes.next = next_source;
        queue_inner.next = next_queue;
        let inner = Rc::new(SubscriptionInner {
            id,
            source_id,
            source: Rc::downgrade(source),
            queue: Rc::downgrade(&queue.0),
        });
        source_routes
            .routes
            .insert(source_id, Rc::downgrade(&inner));
        queue_inner.live.insert(id);
        // Initial synchronization must work even without a signal write.
        queue_inner.pending.insert(id);
        SignalSubscription(inner)
    }

    pub(crate) fn notify(&self) {
        for route in self.routes.values() {
            if let Some(subscription) = route.upgrade()
                && let Some(queue) = subscription.queue.upgrade()
            {
                queue.borrow_mut().pending.insert(subscription.id);
            }
        }
    }
}

struct SubscriptionInner {
    id: SignalChangeId,
    source_id: u64,
    source: Weak<RefCell<SignalSubscribers>>,
    queue: Weak<RefCell<QueueInner>>,
}

impl Drop for SubscriptionInner {
    fn drop(&mut self) {
        if let Some(source) = self.source.upgrade() {
            source.borrow_mut().routes.remove(&self.source_id);
        }
        if let Some(queue) = self.queue.upgrade() {
            let mut queue = queue.borrow_mut();
            queue.live.remove(&self.id);
            queue.pending.remove(&self.id);
        }
    }
}

/// A signal-to-queue lease. Drop detaches immediately and cancels pending work;
/// it does not own the signal's value, runtime or queue, and creates no cycle.
pub struct SignalSubscription(Rc<SubscriptionInner>);

impl SignalSubscription {
    /// This subscription's identity within its queue. IDs are never recycled.
    pub fn id(&self) -> SignalChangeId {
        self.0.id
    }
}
