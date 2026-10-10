//! §24.5 statistical sampling profiler (low overhead) — deterministic core.
//!
//! Instrumented spans ([`crate::span`]/[`crate::instrument`]) are precise but
//! only cover code that was explicitly instrumented; they are blind to
//! third-party libraries and un-instrumented hot paths. A statistical sampling
//! profiler fills that gap: periodically interrupt each thread, capture its
//! call stack, and reconstruct where time went from the sample distribution — a
//! frame seen in a larger fraction of samples ran for a larger fraction of wall
//! time. The real timer interrupt and stack walk are a platform concern
//! (`prism_platform` high-resolution clock + unwinder); **this module owns the
//! deterministic data model and the offline aggregation those captures feed**,
//! so every reconstruction is a pure, testable function of the sample buffer.
//!
//! The module delivers the three §24.5 pieces as pure `core`/`alloc` integer
//! arithmetic (deterministic, `no_std` + `alloc`, no `unsafe`), always compiled
//! regardless of crate features:
//!
//! 1. **Sampling data model + collector** ([`symbol`] + [`sample`]):
//!    [`SymbolTable`] interns frame names to compact [`FrameId`]s;
//!    [`StackSample`] is one captured stack (lane, thread, timestamp, weight,
//!    frames); [`SamplingProfiler`] owns the symbol table + sample buffer + the
//!    nominal interval used to turn sample counts into estimated nanoseconds.
//! 2. **Hotspot folding** ([`fold`]): a [`FlatProfile`] (self/inclusive per
//!    frame), merged [`CallTree`]s folded [`FoldDirection::TopDown`] or
//!    [`FoldDirection::BottomUp`] (flame graph / inverted flame graph), and
//!    [`CollapsedStack`]s in Brendan Gregg's folded format.
//! 3. **Faceting + fusion** ([`facet`] + [`fusion`]): per-lane
//!    ([`facet_by_lane`]) and per-thread ([`facet_by_thread`]) hotspot
//!    distributions (tasks §24.2 thread-class separation), and [`fuse`], which
//!    overlays precise [`InstrumentedSpan`]s onto the sampled profile on one
//!    flame-graph surface and flags statistical disagreement.
//!
//! Honest boundary: the real interrupt-driven capture (timer + stack unwind,
//! and `<1%` overhead at runtime) is upper-layer wiring into `prism_platform`;
//! this layer's sampling aggregation and statistical reconstruction are real,
//! usable, and oracle-checked offline.

pub mod facet;
pub mod fold;
pub mod fusion;
pub mod sample;
pub mod symbol;

pub use facet::{facet_by_lane, facet_by_thread, LaneProfile, ThreadProfile};
pub use fold::{
    call_tree, collapsed_stacks, flat_profile, CallNode, CallTree, CollapsedStack, FlatProfile,
    FoldDirection, FrameStat,
};
pub use fusion::{fuse, FusedEntry, FusedProfile, FusionSource, InstrumentedSpan};
pub use sample::{LaneKind, SamplingProfiler, StackSample};
pub use symbol::{FrameId, SymbolTable};

impl SamplingProfiler {
    /// Fold the recorded samples into a flat per-frame hotspot table.
    #[must_use]
    pub fn flat_profile(&self) -> FlatProfile {
        flat_profile(self.samples(), self.interval_nanos())
    }

    /// Fold the recorded samples into a merged [`CallTree`] in `direction`.
    #[must_use]
    pub fn call_tree(&self, direction: FoldDirection) -> CallTree {
        call_tree(self.samples(), direction, self.interval_nanos())
    }

    /// Fold the recorded samples into collapsed stacks (folded flame format).
    #[must_use]
    pub fn collapsed_stacks(&self) -> Vec<CollapsedStack> {
        collapsed_stacks(self.samples())
    }

    /// Facet the recorded samples by lane (thread class).
    #[must_use]
    pub fn facet_by_lane(&self) -> Vec<LaneProfile> {
        facet_by_lane(self.samples(), self.interval_nanos())
    }

    /// Facet the recorded samples by thread id.
    #[must_use]
    pub fn facet_by_thread(&self) -> Vec<ThreadProfile> {
        facet_by_thread(self.samples(), self.interval_nanos())
    }

    /// Fuse the recorded samples' flat profile with instrumented spans onto a
    /// single flame-graph surface.
    #[must_use]
    pub fn fuse(&self, spans: &[InstrumentedSpan]) -> FusedProfile {
        fuse(&self.flat_profile(), self.symbols(), spans)
    }
}

extern crate alloc;
