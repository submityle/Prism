//! [`TraceBuffer`]: the ordered, optionally-bounded sink that collects
//! [`TraceEvent`]s in record order (design §16.6).
//!
//! The buffer is a pure data structure — no clock, no threads. Callers (or the
//! `std` [`TraceRecorder`](super::recorder::TraceRecorder)) push events; the
//! buffer stamps each with a monotonically increasing sequence number so the
//! stream has a total, stable order even when several events share a timestamp.
//!
//! # Bounding policy
//!
//! A tracing sink that runs every frame must not grow without bound. A buffer
//! built with [`with_capacity`](TraceBuffer::with_capacity) keeps at most `cap`
//! events: once full, each new push evicts the **oldest** event (a FIFO
//! window onto the most recent activity) and increments
//! [`dropped`](TraceBuffer::dropped). A buffer built with
//! [`new`](TraceBuffer::new) is unbounded (devtools captures a bounded window
//! of frames explicitly). Eviction is `O(n)` on the retained window, which is
//! immaterial for a devtools-scale window and is never on the simulation hot
//! path.

use alloc::vec::Vec;

use super::event::{EventPhase, TraceArgValue, TraceEvent, TrackId};

/// An ordered sink of [`TraceEvent`]s with an optional most-recent-window
/// bound. See the [module docs](self) for the eviction policy.
#[derive(Clone, Debug, Default)]
pub struct TraceBuffer {
    events: Vec<TraceEvent>,
    /// `0` means unbounded; otherwise the maximum retained event count.
    capacity: usize,
    next_seq: u64,
    dropped: u64,
}

impl TraceBuffer {
    /// An unbounded buffer. Suitable for capturing a known-bounded number of
    /// frames; prefer [`with_capacity`](TraceBuffer::with_capacity) for an
    /// always-on sink.
    #[must_use]
    #[inline]
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            capacity: 0,
            next_seq: 0,
            dropped: 0,
        }
    }

    /// A buffer that retains at most `cap` most-recent events, evicting the
    /// oldest on overflow. A `cap` of `0` is treated as unbounded (same as
    /// [`new`](TraceBuffer::new)).
    #[must_use]
    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            events: Vec::with_capacity(cap),
            capacity: cap,
            next_seq: 0,
            dropped: 0,
        }
    }

    /// Push a pre-built event, stamping it with the next sequence number and
    /// applying the bounding policy. Returns the sequence number assigned.
    pub fn push(&mut self, mut event: TraceEvent) -> u64 {
        let seq = self.next_seq;
        event.set_seq(seq);
        self.next_seq += 1;

        if self.capacity != 0 && self.events.len() >= self.capacity {
            // Window is full: evict the oldest to make room for the newest.
            self.events.remove(0);
            self.dropped += 1;
        }
        self.events.push(event);
        seq
    }

    /// Record a [`Begin`](EventPhase::Begin) span-open on `track` at
    /// `timestamp_ns`. Pair with [`end`](TraceBuffer::end).
    pub fn begin(&mut self, track: TrackId, timestamp_ns: u64, name: &str) -> u64 {
        self.push(TraceEvent::begin(track, timestamp_ns, name))
    }

    /// Record an [`End`](EventPhase::End) span-close on `track` at
    /// `timestamp_ns`.
    pub fn end(&mut self, track: TrackId, timestamp_ns: u64, name: &str) -> u64 {
        self.push(TraceEvent::end(track, timestamp_ns, name))
    }

    /// Record an [`Instant`](EventPhase::Instant) marker on `track`.
    pub fn instant(&mut self, track: TrackId, timestamp_ns: u64, name: &str) -> u64 {
        self.push(TraceEvent::instant(track, timestamp_ns, name))
    }

    /// Record a [`Complete`](EventPhase::Complete) span of `duration_ns` on
    /// `track` starting at `timestamp_ns`.
    pub fn complete(
        &mut self,
        track: TrackId,
        timestamp_ns: u64,
        duration_ns: u64,
        name: &str,
    ) -> u64 {
        self.push(TraceEvent::complete(track, timestamp_ns, duration_ns, name))
    }

    /// Record a [`Complete`](EventPhase::Complete) span with a category and a
    /// single unsigned argument — the common devtools annotation (e.g. tagging
    /// a system span with its dirty-chunk count). For richer annotations build
    /// a [`TraceEvent`] directly and [`push`](TraceBuffer::push) it.
    pub fn complete_counted(
        &mut self,
        track: TrackId,
        timestamp_ns: u64,
        duration_ns: u64,
        category: &'static str,
        name: &str,
        arg_key: &'static str,
        arg_value: u64,
    ) -> u64 {
        self.push(
            TraceEvent::complete(track, timestamp_ns, duration_ns, name)
                .with_category(category)
                .with_arg(arg_key, TraceArgValue::Uint(arg_value)),
        )
    }

    /// The retained events, oldest-first, in record (sequence) order.
    #[must_use]
    #[inline]
    pub fn events(&self) -> &[TraceEvent] {
        &self.events
    }

    /// The number of currently retained events.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether no events are currently retained.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// The configured retention bound (`0` = unbounded).
    #[must_use]
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Total events evicted by the bounding policy over this buffer's lifetime.
    /// Non-zero means the window dropped older activity.
    #[must_use]
    #[inline]
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// The sequence number the next pushed event will receive (also the total
    /// number of events ever pushed, retained or evicted).
    #[must_use]
    #[inline]
    pub fn total_pushed(&self) -> u64 {
        self.next_seq
    }

    /// Whether every [`Begin`](EventPhase::Begin) in the retained window is
    /// matched by a later [`End`](EventPhase::End) on the same track, with no
    /// unmatched `End`. Evicted events are not considered, so a bounded buffer
    /// mid-stream may legitimately report `false`; use this on a complete
    /// capture to validate recorder stack discipline.
    #[must_use]
    pub fn is_balanced(&self) -> bool {
        // Per-track open-span depth. Small track counts, so a linear scan of a
        // `(track, depth)` association list beats pulling in a map.
        let mut depths: Vec<(TrackId, i64)> = Vec::new();
        for ev in &self.events {
            match ev.phase() {
                EventPhase::Begin => {
                    Self::bump_depth(&mut depths, ev.track(), 1);
                }
                EventPhase::End => {
                    if Self::bump_depth(&mut depths, ev.track(), -1) < 0 {
                        return false;
                    }
                }
                EventPhase::Instant | EventPhase::Complete => {}
            }
        }
        depths.iter().all(|&(_, d)| d == 0)
    }

    fn bump_depth(depths: &mut Vec<(TrackId, i64)>, track: TrackId, delta: i64) -> i64 {
        match depths.iter_mut().find(|(t, _)| *t == track) {
            Some((_, d)) => {
                *d += delta;
                *d
            }
            None => {
                depths.push((track, delta));
                delta
            }
        }
    }

    /// Drop all retained events and reset the sequence counter and drop count.
    /// Reuses the backing allocation.
    pub fn clear(&mut self) {
        self.events.clear();
        self.next_seq = 0;
        self.dropped = 0;
    }
}
