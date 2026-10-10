//! `std` bridge from the executor's [`SystemInstrument`] hook to a
//! [`TraceBuffer`] (design §16.6).
//!
//! The `no_std` core owns no clock, so timeline capture — like the flame-graph
//! [`SpanRecorder`](crate::diagnostics::profiler::SpanRecorder) — needs a `std`
//! shim to supply timestamps. [`TraceRecorder`] implements [`SystemInstrument`]:
//! the executor's perfectly-nested `begin`/`end` calls become
//! [`Begin`](super::event::EventPhase::Begin) /
//! [`End`](super::event::EventPhase::End) events timestamped from a monotonic
//! [`Instant`] origin, all on one configurable [`TrackId`].
//!
//! Pass a [`TraceRecorder`] to
//! [`run_instrumented`](crate::schedule::SingleThreadedExecutor::run_instrumented)
//! exactly where a [`SpanRecorder`](crate::diagnostics::profiler::SpanRecorder)
//! would go; the two capture the same spans into different shapes (timeline vs
//! flame graph). The default `()` instrument path stays zero-overhead.
//!
//! [`SystemInstrument`]: crate::diagnostics::profiler::SystemInstrument
//! [`Instant`]: std::time::Instant

use alloc::string::String;
use alloc::vec::Vec;
use std::time::Instant;

use super::buffer::TraceBuffer;
use super::event::TrackId;
use crate::diagnostics::profiler::SystemInstrument;

/// A [`SystemInstrument`] that records executor spans into a [`TraceBuffer`] as
/// a timestamped `Begin`/`End` stream on a single track.
#[derive(Debug)]
pub struct TraceRecorder {
    buffer: TraceBuffer,
    origin: Instant,
    track: TrackId,
    category: &'static str,
    /// Labels of currently-open spans, so each `end` can name the span it
    /// closes (the executor guarantees stack discipline).
    open: Vec<String>,
}

impl TraceRecorder {
    /// A recorder writing to an unbounded buffer on [`TrackId::MAIN`], with the
    /// monotonic origin set to now and category `"system"`.
    #[must_use]
    pub fn new() -> Self {
        Self::with_buffer(TraceBuffer::new())
    }

    /// A recorder whose buffer retains at most `cap` most-recent events (an
    /// always-on sink; see [`TraceBuffer::with_capacity`]).
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        Self::with_buffer(TraceBuffer::with_capacity(cap))
    }

    /// A recorder writing into a caller-provided buffer (e.g. to continue a
    /// capture or pre-size it).
    #[must_use]
    pub fn with_buffer(buffer: TraceBuffer) -> Self {
        Self {
            buffer,
            origin: Instant::now(),
            track: TrackId::MAIN,
            category: "system",
            open: Vec::new(),
        }
    }

    /// Builder: record onto `track` instead of [`TrackId::MAIN`].
    #[must_use]
    pub fn on_track(mut self, track: TrackId) -> Self {
        self.track = track;
        self
    }

    /// Builder: tag recorded spans with `category` instead of `"system"`.
    #[must_use]
    pub fn with_category(mut self, category: &'static str) -> Self {
        self.category = category;
        self
    }

    /// Reset the monotonic origin to now, so subsequent timestamps are measured
    /// from this point (e.g. at a frame boundary).
    pub fn reset_origin(&mut self) {
        self.origin = Instant::now();
    }

    /// Nanoseconds elapsed since the monotonic origin, saturating at
    /// [`u64::MAX`] (≈ 584 years, so saturation is unreachable in practice).
    #[must_use]
    pub fn elapsed_ns(&self) -> u64 {
        u64::try_from(self.origin.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }

    /// Record a zero-width [`Instant`](super::event::EventPhase::Instant) marker
    /// now (e.g. a frame boundary), outside the span stack.
    pub fn mark(&mut self, name: &str) {
        let ts = self.elapsed_ns();
        self.buffer.push(
            super::event::TraceEvent::instant(self.track, ts, name).with_category(self.category),
        );
    }

    /// The accumulated buffer.
    #[must_use]
    pub fn buffer(&self) -> &TraceBuffer {
        &self.buffer
    }

    /// Consume the recorder and return its buffer.
    #[must_use]
    pub fn into_buffer(self) -> TraceBuffer {
        self.buffer
    }

    /// Whether every open span has been closed (no dangling `begin`).
    #[must_use]
    pub fn is_balanced(&self) -> bool {
        self.open.is_empty()
    }
}

impl Default for TraceRecorder {
    fn default() -> Self {
        Self::new()
    }
}

impl SystemInstrument for TraceRecorder {
    fn begin(&mut self, label: &str) {
        let ts = self.elapsed_ns();
        self.open.push(String::from(label));
        self.buffer.push(
            super::event::TraceEvent::begin(self.track, ts, label).with_category(self.category),
        );
    }

    fn end(&mut self) {
        let ts = self.elapsed_ns();
        // The executor nests perfectly; if `open` is empty we still emit a
        // best-effort unnamed close rather than panic in a diagnostics path.
        let name = self.open.pop().unwrap_or_default();
        self.buffer.push(
            super::event::TraceEvent::end(self.track, ts, name).with_category(self.category),
        );
    }
}
