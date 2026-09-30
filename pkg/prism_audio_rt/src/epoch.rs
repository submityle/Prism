//! Deferred reclamation and graph hand-off.
//!
//! The audio thread must never call the allocator, which means it must never
//! run a destructor that frees memory. Two primitives cooperate to uphold that
//! invariant while still letting the graph and its resources evolve at runtime:
//!
//! - [`RetireQueue`] lets the audio thread *retire* an owned, heap-backed
//!   resource ([`Box<dyn Any + Send>`]) by pushing it onto a lock-free queue.
//!   A [`Collector`] running on a task thread later drains the queue and drops
//!   the resources there, so the actual `free` happens off the audio thread.
//! - [`GraphHandoff`] is a capacity-one slot used to publish a freshly compiled
//!   [`AudioGraph`] to the audio thread. The audio thread swaps it in at a
//!   block boundary and retires the previous graph through a [`RetireQueue`],
//!   so graph replacement is wait-free on the audio side and allocation-free in
//!   both directions of the hot path.
//!
//! This mirrors the epoch / hazard-pointer style reclamation used by production
//! audio engines, but is built entirely on the safe, bounded
//! [`crossbeam_queue::ArrayQueue`]; this crate contains no `unsafe` code.

use std::any::Any;
use alloc::sync::Arc;

use crossbeam_queue::ArrayQueue;
use prism_audio_core::graph::AudioGraph;

/// A resource whose destructor must run off the audio thread.
type Retired = Box<dyn Any + Send>;

/// The audio-thread half of a [`RetireQueue`]: pushes resources to be dropped
/// elsewhere.
#[derive(Debug, Clone)]
pub struct Retirer {
    /// Shared retirement queue; pre-allocated at construction.
    queue: Arc<ArrayQueue<Retired>>,
}

impl Retirer {
    /// Hands `resource` off to be dropped by a [`Collector`].
    ///
    /// This does not run the resource's destructor; it only moves ownership
    /// into the queue, which never allocates. On success returns `Ok(())`.
    ///
    /// # Errors
    ///
    /// Returns `Err(resource)` when the queue is full. The caller (the audio
    /// runtime) must hold the resource and retry on a later block rather than
    /// drop it, so that no `free` ever runs on the audio thread.
    #[inline]
    pub fn retire(&self, resource: Retired) -> Result<(), Retired> {
        self.queue.push(resource)
    }

    /// Number of resources currently awaiting collection (racy snapshot).
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Whether no resources are currently awaiting collection (racy snapshot).
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

/// The task-thread half of a [`RetireQueue`]: drains and drops retired
/// resources.
#[derive(Debug)]
pub struct Collector {
    /// Shared retirement queue; pre-allocated at construction.
    queue: Arc<ArrayQueue<Retired>>,
}

impl Collector {
    /// Drains every pending resource and drops it on the calling thread.
    ///
    /// Returns the number of resources reclaimed. Intended to run periodically
    /// on a task thread (never the audio thread), where allocation is allowed.
    pub fn collect(&self) -> usize {
        let mut reclaimed = 0;
        while let Some(resource) = self.queue.pop() {
            drop(resource);
            reclaimed += 1;
        }
        reclaimed
    }

    /// Number of resources currently awaiting collection (racy snapshot).
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Whether no resources are currently awaiting collection (racy snapshot).
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

/// A bounded queue for deferred reclamation. Construct with [`RetireQueue::new`]
/// to obtain the paired [`Retirer`] (audio thread) and [`Collector`] (task
/// thread).
#[derive(Debug)]
pub struct RetireQueue;

impl RetireQueue {
    /// Creates a retirement queue with `capacity` pre-allocated slots and
    /// returns the paired [`Retirer`] and [`Collector`].
    ///
    /// `capacity` is clamped up to `1`.
    #[must_use]
    #[expect(
        clippy::new_ret_no_self,
        reason = "constructor returns the paired producer/consumer halves, like a channel"
    )]
    pub fn new(capacity: usize) -> (Retirer, Collector) {
        let queue = Arc::new(ArrayQueue::new(capacity.max(1)));
        (
            Retirer {
                queue: Arc::clone(&queue),
            },
            Collector { queue },
        )
    }
}

/// The task-thread half of a [`GraphHandoff`]: publishes a compiled graph to
/// the audio thread.
#[derive(Debug, Clone)]
pub struct GraphProducer {
    /// Capacity-one slot holding the most recently published graph.
    slot: Arc<ArrayQueue<Box<AudioGraph>>>,
}

impl GraphProducer {
    /// Publishes `graph` for the audio thread to swap in on its next block.
    ///
    /// The slot holds a single pending graph. If a previously published graph
    /// has not yet been consumed, it is popped and returned as
    /// `Some(previous)` so the caller can drop or retire it on this (task)
    /// thread; the freshly published graph always wins. Returns `None` when no
    /// stale graph was displaced.
    pub fn publish(&self, graph: Box<AudioGraph>) -> Option<Box<AudioGraph>> {
        // Try the fast path first; only displace a stale graph if the slot is
        // occupied. The audio thread is the only consumer, so at most one stale
        // graph can be present.
        match self.slot.push(graph) {
            Ok(()) => None,
            Err(graph) => {
                let stale = self.slot.pop();
                // The slot is capacity-one and we are the only producer, so this
                // push cannot fail after a successful pop; if it somehow does,
                // hand the new graph back to the caller rather than panic.
                match self.slot.push(graph) {
                    Ok(()) => stale,
                    Err(graph) => Some(graph),
                }
            }
        }
    }
}

/// The audio-thread half of a [`GraphHandoff`]: takes a newly published graph.
#[derive(Debug)]
pub struct GraphConsumer {
    /// Capacity-one slot holding the most recently published graph.
    slot: Arc<ArrayQueue<Box<AudioGraph>>>,
}

impl GraphConsumer {
    /// Takes the pending published graph, if any, transferring ownership to the
    /// caller. Never allocates; safe to call from the audio thread.
    #[inline]
    pub fn take(&self) -> Option<Box<AudioGraph>> {
        self.slot.pop()
    }
}

/// A capacity-one hand-off channel that publishes a compiled [`AudioGraph`] to
/// the audio thread. Construct with [`GraphHandoff::new`].
#[derive(Debug)]
pub struct GraphHandoff;

impl GraphHandoff {
    /// Creates the paired [`GraphProducer`] (task thread) and [`GraphConsumer`]
    /// (audio thread).
    #[must_use]
    #[expect(
        clippy::new_ret_no_self,
        reason = "constructor returns the paired producer/consumer halves, like a channel"
    )]
    pub fn new() -> (GraphProducer, GraphConsumer) {
        let slot = Arc::new(ArrayQueue::new(1));
        (
            GraphProducer {
                slot: Arc::clone(&slot),
            },
            GraphConsumer { slot },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{GraphHandoff, RetireQueue};
    use prism_audio_core::graph::AudioGraph;

    #[test]
    fn collector_drops_retired_resources_off_thread() {
        let (retirer, collector) = RetireQueue::new(8);
        assert!(retirer.is_empty());
        retirer
            .retire(Box::new(vec![1u8, 2, 3]))
            .expect("queue has room");
        retirer.retire(Box::new(42u64)).expect("queue has room");
        assert_eq!(retirer.len(), 2);
        assert_eq!(collector.collect(), 2);
        assert!(collector.is_empty());
        assert_eq!(collector.collect(), 0);
    }

    #[test]
    fn retire_hands_back_resource_when_full() {
        let (retirer, _collector) = RetireQueue::new(1);
        retirer.retire(Box::new(1u8)).expect("first fits");
        let rejected = retirer.retire(Box::new(2u8));
        assert!(rejected.is_err());
    }

    #[test]
    fn graph_handoff_publishes_newest_and_displaces_stale() {
        let (producer, consumer) = GraphHandoff::new();
        assert!(consumer.take().is_none());
        assert!(producer.publish(Box::new(AudioGraph::new(48_000, 512))).is_none());
        // A second publish before the consumer takes displaces the stale graph.
        let displaced = producer.publish(Box::new(AudioGraph::new(44_100, 256)));
        let displaced = displaced.expect("stale graph handed back");
        assert_eq!(displaced.sample_rate(), 48_000);
        let taken = consumer.take().expect("newest graph is present");
        assert_eq!(taken.sample_rate(), 44_100);
        assert!(consumer.take().is_none());
    }
}
