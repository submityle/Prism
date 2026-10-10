//! The trace event model: tracks, phases, typed arguments, and a single
//! timeline event (design §16.6 "系统火焰图" 的时间线对偶).
//!
//! Where [`profiler`](crate::diagnostics::profiler) folds nested `begin`/`end`
//! calls into a self-time *tree* (a flame graph), the trace model keeps the raw
//! *ordered event stream* with explicit timestamps and tracks. A tree answers
//! "where did the time go?"; an event stream answers "what ran when, on which
//! track, next to what?" — the timeline / Perfetto view every AAA engine ships.
//! The two are complementary and share the same [`SystemInstrument`] capture
//! hook (see [`recorder`](super::recorder)).
//!
//! The core is `no_std + alloc` and owns no clock: timestamps are supplied by
//! the caller in **nanoseconds** on a monotonic timeline of its choosing,
//! exactly as [`profiler`](crate::diagnostics::profiler) consumes caller-measured
//! [`Duration`](core::time::Duration)s. A `std` bridge that fills timestamps
//! from [`std::time::Instant`] lives in [`recorder`](super::recorder).
//!
//! [`SystemInstrument`]: crate::diagnostics::profiler::SystemInstrument

use alloc::string::String;
use alloc::vec::Vec;

/// A logical track an event belongs to — a thread, a job-system lane, or any
/// other serial timeline. Maps to the Chrome Trace Event Format `tid` field.
///
/// Tracks are purely a presentation axis: events on different tracks render as
/// parallel lanes, events on the same track stack by nesting. The kernel does
/// not assign tracks; the caller picks a stable id per thread/lane so repeated
/// runs render identically.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TrackId(pub u32);

impl TrackId {
    /// The conventional main-thread / primary-schedule track.
    pub const MAIN: TrackId = TrackId(0);

    /// The raw track index.
    #[must_use]
    #[inline]
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl Default for TrackId {
    #[inline]
    fn default() -> Self {
        TrackId::MAIN
    }
}

/// The phase of a timeline event — which half of a span it is, or that it is a
/// standalone marker. The names mirror the Chrome Trace Event Format `ph`
/// codes so [`chrome_ph`](EventPhase::chrome_ph) is a direct mapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EventPhase {
    /// Opens a duration span (Chrome `"B"`). Must be matched by a later
    /// [`End`](EventPhase::End) on the same track.
    Begin,
    /// Closes the most recently opened duration span on its track (Chrome
    /// `"E"`).
    End,
    /// A zero-width marker at a single instant (Chrome `"i"`), e.g. a frame
    /// boundary or a one-shot event.
    Instant,
    /// A complete duration span carrying its own length (Chrome `"X"`), so it
    /// needs no separate [`End`](EventPhase::End). The length lives in
    /// [`TraceEvent::duration_ns`].
    Complete,
}

impl EventPhase {
    /// The single-character Chrome Trace Event Format phase code.
    #[must_use]
    #[inline]
    pub const fn chrome_ph(self) -> &'static str {
        match self {
            EventPhase::Begin => "B",
            EventPhase::End => "E",
            EventPhase::Instant => "i",
            EventPhase::Complete => "X",
        }
    }

    /// Whether this phase carries a meaningful [`TraceEvent::duration_ns`].
    #[must_use]
    #[inline]
    pub const fn has_duration(self) -> bool {
        matches!(self, EventPhase::Complete)
    }
}

/// A typed value attached to a [`TraceArg`]. Kept to integer / boolean / string
/// variants so [`TraceEvent`] stays `Eq` (no floating point) and so the Chrome
/// JSON exporter needs no float formatting.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TraceArgValue {
    /// A signed integer argument (e.g. a net delta).
    Int(i64),
    /// An unsigned integer argument (e.g. a dirty-chunk or entity count).
    Uint(u64),
    /// A boolean argument.
    Bool(bool),
    /// A free-form string argument (JSON-escaped on export).
    Str(String),
}

/// A single `key: value` annotation on an event, surfaced under the Chrome
/// `args` object. The key is a `&'static str` to keep annotations allocation-
/// free at the call site.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraceArg {
    key: &'static str,
    value: TraceArgValue,
}

impl TraceArg {
    /// Build an argument from a static key and a typed value.
    #[must_use]
    #[inline]
    pub fn new(key: &'static str, value: TraceArgValue) -> Self {
        Self { key, value }
    }

    /// The argument key.
    #[must_use]
    #[inline]
    pub fn key(&self) -> &'static str {
        self.key
    }

    /// The argument value.
    #[must_use]
    #[inline]
    pub fn value(&self) -> &TraceArgValue {
        &self.value
    }
}

/// One recorded point on the timeline: a phase, a label, a category, the track
/// it belongs to, its monotonic timestamp, an optional duration (for
/// [`Complete`](EventPhase::Complete)), and any typed arguments.
///
/// `seq` is a buffer-assigned monotonically increasing sequence number: it
/// gives a total, stable order to events that share a timestamp, so exports and
/// comparisons are deterministic regardless of how the recorder batched them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraceEvent {
    seq: u64,
    phase: EventPhase,
    track: TrackId,
    timestamp_ns: u64,
    duration_ns: u64,
    category: &'static str,
    name: String,
    args: Vec<TraceArg>,
}

impl TraceEvent {
    /// Build an event with an explicit phase. Prefer the phase-specific
    /// constructors ([`begin`](TraceEvent::begin) etc.) at call sites; this is
    /// the general form the buffer and tests use. `seq` is `0` until the buffer
    /// stamps it on [`push`](super::buffer::TraceBuffer::push).
    #[must_use]
    pub fn new(
        phase: EventPhase,
        track: TrackId,
        timestamp_ns: u64,
        name: impl Into<String>,
    ) -> Self {
        Self {
            seq: 0,
            phase,
            track,
            timestamp_ns,
            duration_ns: 0,
            category: "",
            name: name.into(),
            args: Vec::new(),
        }
    }

    /// A [`Begin`](EventPhase::Begin) span-open event.
    #[must_use]
    #[inline]
    pub fn begin(track: TrackId, timestamp_ns: u64, name: impl Into<String>) -> Self {
        Self::new(EventPhase::Begin, track, timestamp_ns, name)
    }

    /// An [`End`](EventPhase::End) span-close event.
    #[must_use]
    #[inline]
    pub fn end(track: TrackId, timestamp_ns: u64, name: impl Into<String>) -> Self {
        Self::new(EventPhase::End, track, timestamp_ns, name)
    }

    /// An [`Instant`](EventPhase::Instant) zero-width marker.
    #[must_use]
    #[inline]
    pub fn instant(track: TrackId, timestamp_ns: u64, name: impl Into<String>) -> Self {
        Self::new(EventPhase::Instant, track, timestamp_ns, name)
    }

    /// A [`Complete`](EventPhase::Complete) span carrying its own duration.
    #[must_use]
    #[inline]
    pub fn complete(
        track: TrackId,
        timestamp_ns: u64,
        duration_ns: u64,
        name: impl Into<String>,
    ) -> Self {
        let mut ev = Self::new(EventPhase::Complete, track, timestamp_ns, name);
        ev.duration_ns = duration_ns;
        ev
    }

    /// Builder: set the category (a `&'static str` grouping tag, e.g.
    /// `"system"` / `"commands"` / `"frame"`).
    #[must_use]
    pub fn with_category(mut self, category: &'static str) -> Self {
        self.category = category;
        self
    }

    /// Builder: attach one typed argument. Repeatable.
    #[must_use]
    pub fn with_arg(mut self, key: &'static str, value: TraceArgValue) -> Self {
        self.args.push(TraceArg::new(key, value));
        self
    }

    /// The buffer-assigned sequence number (total stable order across equal
    /// timestamps). Zero before the event is pushed into a buffer.
    #[must_use]
    #[inline]
    pub fn seq(&self) -> u64 {
        self.seq
    }

    /// The event phase.
    #[must_use]
    #[inline]
    pub fn phase(&self) -> EventPhase {
        self.phase
    }

    /// The track this event belongs to.
    #[must_use]
    #[inline]
    pub fn track(&self) -> TrackId {
        self.track
    }

    /// The monotonic timestamp in nanoseconds.
    #[must_use]
    #[inline]
    pub fn timestamp_ns(&self) -> u64 {
        self.timestamp_ns
    }

    /// The duration in nanoseconds. Meaningful only when
    /// [`phase`](TraceEvent::phase) is [`Complete`](EventPhase::Complete);
    /// otherwise `0`.
    #[must_use]
    #[inline]
    pub fn duration_ns(&self) -> u64 {
        self.duration_ns
    }

    /// The category tag (empty when none was set).
    #[must_use]
    #[inline]
    pub fn category(&self) -> &'static str {
        self.category
    }

    /// The event label.
    #[must_use]
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The typed arguments attached to this event.
    #[must_use]
    #[inline]
    pub fn args(&self) -> &[TraceArg] {
        &self.args
    }

    /// Internal: stamp the buffer-assigned sequence number.
    pub(super) fn set_seq(&mut self, seq: u64) {
        self.seq = seq;
    }
}
