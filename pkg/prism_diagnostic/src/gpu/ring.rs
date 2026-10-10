//! N-frame GPU timestamp readback ring.
//!
//! GPU timestamp queries cannot be read back on the frame they are issued: the
//! GPU is still in flight, so the CPU would have to stall. Instead the backend
//! issues a query, keeps rendering, and reads the result back `N` frames later
//! (`N` = the number of frames the CPU is allowed to run ahead of the GPU).
//!
//! [`GpuReadbackRing`] models exactly that latency without ever blocking:
//!
//! 1. The backend calls [`issue`](GpuReadbackRing::issue) when it inserts a
//!    timestamp-query pair, recording the frame number and metadata and getting
//!    back a [`GpuQueryId`].
//! 2. Later — typically `latency_frames` later — the backend reads the two
//!    ticks and calls [`resolve`](GpuReadbackRing::resolve) with them. The
//!    pending query becomes a finished [`GpuSpan`] in the resolved ring.
//! 3. [`ready_queries`](GpuReadbackRing::ready_queries) tells the backend which
//!    in-flight queries *should* have results available for a given current
//!    frame, so it can resolve them opportunistically. Resolution is always
//!    driven by the backend reporting real results — this type never fabricates
//!    GPU data.
//!
//! The resolved side reuses the crate's [`RingBuffer`] so it bounds memory and
//! overwrites the oldest resolved span when full, exactly like the CPU span
//! rings.

extern crate alloc;

use alloc::collections::VecDeque;
use alloc::string::String;
use alloc::vec::Vec;

use super::scope::{CorrelationId, GpuQueueId, GpuSpan, GpuTick};
use crate::trace::ring::RingBuffer;

/// Default capacity of the resolved-span ring.
pub const DEFAULT_RESOLVED_CAPACITY: usize = 4096;

/// Handle to an in-flight timestamp query returned by
/// [`GpuReadbackRing::issue`] and passed back to
/// [`GpuReadbackRing::resolve`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuQueryId(pub u64);

/// An issued-but-not-yet-resolved timestamp query.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PendingQuery {
    /// Identity of this query.
    pub id: GpuQueryId,
    /// Frame the query pair was issued on.
    pub frame: u64,
    /// Scope label carried through to the resolved span.
    pub label: String,
    /// Queue the timestamps are taken on.
    pub queue: GpuQueueId,
    /// Optional cross-queue correlation token.
    pub correlation: Option<CorrelationId>,
    /// Nesting depth within the frame's GPU scopes.
    pub depth: u32,
}

/// A ring that correlates in-flight timestamp queries to the frame they were
/// issued on and resolves them when the backend reports ticks, without blocking
/// the current frame.
#[derive(Debug)]
pub struct GpuReadbackRing {
    latency_frames: u64,
    next_id: u64,
    pending: VecDeque<PendingQuery>,
    resolved: RingBuffer<GpuSpan>,
}

impl GpuReadbackRing {
    /// Create a ring expecting readback to lag `latency_frames` frames, with a
    /// resolved-span capacity of [`DEFAULT_RESOLVED_CAPACITY`].
    #[must_use]
    pub fn new(latency_frames: u64) -> Self {
        Self::with_capacity(latency_frames, DEFAULT_RESOLVED_CAPACITY)
    }

    /// Create a ring with an explicit resolved-span capacity.
    #[must_use]
    pub fn with_capacity(latency_frames: u64, resolved_capacity: usize) -> Self {
        Self {
            latency_frames,
            next_id: 1,
            pending: VecDeque::new(),
            resolved: RingBuffer::with_capacity(resolved_capacity),
        }
    }

    /// The configured readback latency in frames.
    #[must_use]
    pub fn latency_frames(&self) -> u64 {
        self.latency_frames
    }

    /// Register a newly issued timestamp-query pair for `frame` and return its
    /// [`GpuQueryId`]. Never blocks.
    pub fn issue(
        &mut self,
        frame: u64,
        label: impl Into<String>,
        queue: GpuQueueId,
        correlation: Option<CorrelationId>,
        depth: u32,
    ) -> GpuQueryId {
        let id = GpuQueryId(self.next_id);
        self.next_id += 1;
        self.pending.push_back(PendingQuery {
            id,
            frame,
            label: label.into(),
            queue,
            correlation,
            depth,
        });
        id
    }

    /// Resolve a previously issued query with the ticks the backend read back,
    /// moving it from the pending set into the resolved ring.
    ///
    /// Returns the finished [`GpuSpan`], or `None` if `id` is unknown (already
    /// resolved, or never issued). Resolution may be out of order; the common
    /// FIFO case is handled by a fast front-of-queue check.
    pub fn resolve(
        &mut self,
        id: GpuQueryId,
        begin_tick: GpuTick,
        end_tick: GpuTick,
    ) -> Option<GpuSpan> {
        let idx = if self.pending.front().map(|q| q.id) == Some(id) {
            0
        } else {
            self.pending.iter().position(|q| q.id == id)?
        };
        let query = self.pending.remove(idx)?;
        let span = GpuSpan {
            label: query.label,
            queue: query.queue,
            begin_tick,
            end_tick,
            correlation: query.correlation,
            frame: query.frame,
            depth: query.depth,
        };
        self.resolved.push(span.clone());
        Some(span)
    }

    /// Query ids that *should* have results available at `current_frame` given
    /// the configured latency (`issue_frame + latency_frames <= current_frame`),
    /// oldest first. The backend uses this to know which readbacks to attempt.
    #[must_use]
    pub fn ready_queries(&self, current_frame: u64) -> Vec<GpuQueryId> {
        self.pending
            .iter()
            .filter(|q| q.frame.saturating_add(self.latency_frames) <= current_frame)
            .map(|q| q.id)
            .collect()
    }

    /// Number of in-flight (unresolved) queries.
    #[must_use]
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Snapshot the in-flight queries in issue order.
    #[must_use]
    pub fn pending(&self) -> Vec<PendingQuery> {
        self.pending.iter().cloned().collect()
    }

    /// Number of resolved spans currently retained.
    #[must_use]
    pub fn resolved_len(&self) -> usize {
        self.resolved.len()
    }

    /// Snapshot the resolved spans in resolution order (oldest first).
    #[must_use]
    pub fn resolved_snapshot(&self) -> Vec<GpuSpan> {
        self.resolved.snapshot()
    }

    /// Remove and return all resolved spans in resolution order.
    pub fn drain_resolved(&mut self) -> Vec<GpuSpan> {
        self.resolved.drain()
    }

    /// Count of resolved spans overwritten because the resolved ring was full.
    #[must_use]
    pub fn dropped_resolved(&self) -> u64 {
        self.resolved.dropped()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn issue_then_resolve_fifo() {
        let mut ring = GpuReadbackRing::new(2);
        let a = ring.issue(10, "Depth", GpuQueueId::GRAPHICS, None, 0);
        let b = ring.issue(10, "Color", GpuQueueId::GRAPHICS, None, 0);
        assert_eq!(ring.pending_len(), 2);

        let span_a = ring.resolve(a, GpuTick(100), GpuTick(200)).unwrap();
        assert_eq!(span_a.label, "Depth");
        assert_eq!(span_a.duration_ticks(), 100);
        assert_eq!(ring.pending_len(), 1);

        let span_b = ring.resolve(b, GpuTick(200), GpuTick(260)).unwrap();
        assert_eq!(span_b.label, "Color");
        assert_eq!(ring.pending_len(), 0);
        assert_eq!(ring.resolved_len(), 2);

        let snap = ring.resolved_snapshot();
        assert_eq!(snap[0].label, "Depth");
        assert_eq!(snap[1].label, "Color");
    }

    #[test]
    fn resolve_out_of_order() {
        let mut ring = GpuReadbackRing::new(1);
        let a = ring.issue(1, "A", GpuQueueId::GRAPHICS, None, 0);
        let b = ring.issue(1, "B", GpuQueueId::GRAPHICS, None, 0);
        let c = ring.issue(1, "C", GpuQueueId::GRAPHICS, None, 0);
        // Resolve the middle one first.
        assert!(ring.resolve(b, GpuTick(0), GpuTick(5)).is_some());
        assert_eq!(ring.pending_len(), 2);
        assert!(ring.resolve(c, GpuTick(0), GpuTick(7)).is_some());
        assert!(ring.resolve(a, GpuTick(0), GpuTick(3)).is_some());
        let labels: Vec<_> = ring
            .resolved_snapshot()
            .into_iter()
            .map(|s| s.label)
            .collect();
        assert_eq!(labels, ["B", "C", "A"]);
    }

    #[test]
    fn unknown_id_resolves_to_none() {
        let mut ring = GpuReadbackRing::new(1);
        let a = ring.issue(0, "A", GpuQueueId::GRAPHICS, None, 0);
        assert!(ring.resolve(a, GpuTick(0), GpuTick(1)).is_some());
        // Already resolved.
        assert!(ring.resolve(a, GpuTick(0), GpuTick(1)).is_none());
        // Never issued.
        assert!(ring
            .resolve(GpuQueryId(9999), GpuTick(0), GpuTick(1))
            .is_none());
    }

    #[test]
    fn ready_queries_respect_latency() {
        let mut ring = GpuReadbackRing::new(2);
        let f0 = ring.issue(0, "f0", GpuQueueId::GRAPHICS, None, 0);
        let f1 = ring.issue(1, "f1", GpuQueueId::GRAPHICS, None, 0);
        let _f3 = ring.issue(3, "f3", GpuQueueId::GRAPHICS, None, 0);

        // At frame 2, only the query from frame 0 is ready (0 + 2 <= 2).
        assert_eq!(ring.ready_queries(2), [f0]);
        // At frame 3, frames 0 and 1 are ready; frame 3's query is not.
        assert_eq!(ring.ready_queries(3), [f0, f1]);
    }

    #[test]
    fn resolved_ring_bounds_memory() {
        let mut ring = GpuReadbackRing::with_capacity(0, 2);
        for i in 0..5 {
            let id = ring.issue(i, "s", GpuQueueId::GRAPHICS, None, 0);
            ring.resolve(id, GpuTick(0), GpuTick(10)).unwrap();
        }
        assert_eq!(ring.resolved_len(), 2);
        assert_eq!(ring.dropped_resolved(), 3);
    }

    #[test]
    fn correlation_token_survives_resolution() {
        let mut ring = GpuReadbackRing::new(1);
        let id = ring.issue(4, "Shadow", GpuQueueId(1), Some(CorrelationId(77)), 2);
        let span = ring.resolve(id, GpuTick(10), GpuTick(40)).unwrap();
        assert_eq!(span.correlation, Some(CorrelationId(77)));
        assert_eq!(span.queue, GpuQueueId(1));
        assert_eq!(span.frame, 4);
        assert_eq!(span.depth, 2);
    }
}
