//! The sampling data model and collector (§24.5).
//!
//! Instrumented spans only cover code that was explicitly instrumented; they
//! miss third-party libraries and un-instrumented hot paths. A statistical
//! sampling profiler periodically interrupts each thread, captures its current
//! call stack, and *statistically* reconstructs where time went: a frame that
//! appears in a larger fraction of samples ran for a larger fraction of wall
//! time. Capturing the real timer interrupt + stack walk is a platform concern
//! (`prism_platform`); this module owns the deterministic data model and the
//! sample buffer those captures feed, so the aggregation downstream
//! ([`fold`](super::fold)) is fully testable offline.
//!
//! Everything here is pure `core`/`alloc` integer arithmetic — deterministic,
//! `no_std` + `alloc`, no `unsafe`.

extern crate alloc;

use alloc::vec::Vec;

use super::symbol::{FrameId, SymbolTable};

/// The lane (thread class) a sample was captured on.
///
/// Faceting hotspots by lane separates main-thread stalls from compute/I-O work
/// (tasks §24.2 thread-class separation). The enum order is the canonical
/// faceting order used by [`facet_by_lane`](super::facet::facet_by_lane).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LaneKind {
    /// The main/simulation thread.
    Main,
    /// A render-submission thread.
    Render,
    /// A compute/worker (job-system) thread.
    Compute,
    /// An I-O / streaming thread.
    Io,
    /// An audio-mixing thread.
    Audio,
    /// Any other / unclassified thread.
    Other,
}

impl LaneKind {
    /// A short, stable label for folded output and HUD text.
    #[inline]
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Main => "main",
            Self::Render => "render",
            Self::Compute => "compute",
            Self::Io => "io",
            Self::Audio => "audio",
            Self::Other => "other",
        }
    }

    /// All lane kinds in canonical faceting order.
    #[inline]
    #[must_use]
    pub const fn all() -> [LaneKind; 6] {
        [
            Self::Main,
            Self::Render,
            Self::Compute,
            Self::Io,
            Self::Audio,
            Self::Other,
        ]
    }
}

/// One captured stack sample: the lane and thread it came from, when it was
/// taken, how many timer ticks it represents, and the call stack itself.
///
/// The `stack` is ordered outermost-first (root to leaf): `stack[0]` is the
/// entry point and the last element is the function executing when the sample
/// fired. `weight` lets consecutive identical captures coalesce into one entry
/// (weight = number of ticks) without changing any aggregation result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StackSample {
    /// The lane (thread class) this sample came from.
    pub lane: LaneKind,
    /// The OS/logical thread id this sample came from.
    pub thread_id: u64,
    /// Capture timestamp in nanoseconds (caller-supplied; monotonic per lane).
    pub timestamp_nanos: u64,
    /// Number of timer ticks this sample represents (coalescing weight, `>= 1`).
    pub weight: u64,
    /// The captured call stack, outermost frame first, leaf last.
    pub stack: Vec<FrameId>,
}

impl StackSample {
    /// Construct a sample with an explicit weight.
    ///
    /// A zero `weight` is clamped to `1`: a captured sample always represents
    /// at least one tick, and a zero weight would silently vanish from every
    /// aggregation.
    #[must_use]
    pub fn new(
        lane: LaneKind,
        thread_id: u64,
        timestamp_nanos: u64,
        weight: u64,
        stack: Vec<FrameId>,
    ) -> Self {
        Self {
            lane,
            thread_id,
            timestamp_nanos,
            weight: weight.max(1),
            stack,
        }
    }

    /// The leaf frame (the function executing when the sample fired), or `None`
    /// for an empty stack.
    #[inline]
    #[must_use]
    pub fn leaf(&self) -> Option<FrameId> {
        self.stack.last().copied()
    }

    /// Stack depth (number of frames).
    #[inline]
    #[must_use]
    pub fn depth(&self) -> usize {
        self.stack.len()
    }
}

/// A collector that owns the frame [`SymbolTable`] and the captured sample
/// buffer, plus the nominal sampling interval used to turn sample counts into
/// estimated nanoseconds.
///
/// The platform layer feeds captured stacks here (by name or by pre-interned
/// [`FrameId`]); the fold/facet/fusion layers read [`samples`](Self::samples)
/// and [`symbols`](Self::symbols) to reconstruct hotspots. The collector keeps
/// samples in capture order and never reorders or drops them, so aggregation is
/// a pure function of the recorded buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SamplingProfiler {
    /// Interned frame names shared by every recorded stack.
    symbols: SymbolTable,
    /// Captured samples, in capture order.
    samples: Vec<StackSample>,
    /// Nominal time represented by one sample tick, in nanoseconds.
    interval_nanos: u64,
}

impl SamplingProfiler {
    /// A collector with the given nominal sampling interval in nanoseconds
    /// (e.g. `1_000_000` for a 1 kHz / 1 ms profiler).
    ///
    /// A zero interval is clamped to `1` so a sample tick is never worth zero
    /// nanoseconds (which would collapse every reconstructed duration to `0`).
    #[inline]
    #[must_use]
    pub fn new(interval_nanos: u64) -> Self {
        Self {
            symbols: SymbolTable::new(),
            samples: Vec::new(),
            interval_nanos: interval_nanos.max(1),
        }
    }

    /// The nominal time one sample tick represents, in nanoseconds.
    #[inline]
    #[must_use]
    pub fn interval_nanos(&self) -> u64 {
        self.interval_nanos
    }

    /// Borrow the shared symbol table.
    #[inline]
    #[must_use]
    pub fn symbols(&self) -> &SymbolTable {
        &self.symbols
    }

    /// Borrow the captured samples in capture order.
    #[inline]
    #[must_use]
    pub fn samples(&self) -> &[StackSample] {
        &self.samples
    }

    /// Number of recorded samples (entries, not summed weight).
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether no samples have been recorded.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Total weight (tick count) across every recorded sample.
    #[must_use]
    pub fn total_weight(&self) -> u64 {
        self.samples.iter().map(|s| s.weight).sum()
    }

    /// Intern a frame name, returning its [`FrameId`] for reuse.
    #[inline]
    pub fn intern(&mut self, name: &str) -> FrameId {
        self.symbols.intern(name)
    }

    /// Record a sample whose stack is given by name (outermost first), interning
    /// each name as needed. Returns the recorded sample's leaf, if any.
    pub fn record_named(
        &mut self,
        lane: LaneKind,
        thread_id: u64,
        timestamp_nanos: u64,
        stack: &[&str],
    ) -> Option<FrameId> {
        self.record_named_weighted(lane, thread_id, timestamp_nanos, 1, stack)
    }

    /// Record a weighted sample whose stack is given by name (outermost first).
    pub fn record_named_weighted(
        &mut self,
        lane: LaneKind,
        thread_id: u64,
        timestamp_nanos: u64,
        weight: u64,
        stack: &[&str],
    ) -> Option<FrameId> {
        let frames: Vec<FrameId> = stack.iter().map(|name| self.symbols.intern(name)).collect();
        let leaf = frames.last().copied();
        self.samples.push(StackSample::new(
            lane,
            thread_id,
            timestamp_nanos,
            weight,
            frames,
        ));
        leaf
    }

    /// Push a pre-built sample (its [`FrameId`]s must come from this collector's
    /// symbol table).
    #[inline]
    pub fn push(&mut self, sample: StackSample) {
        self.samples.push(sample);
    }

    /// Drop all recorded samples, keeping the interned symbol table.
    #[inline]
    pub fn clear_samples(&mut self) {
        self.samples.clear();
    }
}
