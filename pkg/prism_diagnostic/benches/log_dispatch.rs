//! Logging hot-path overhead (roadmap M-core "基准即规格").
//!
//! The design spec (`docs/prism_diagnostic_design_zh.md`) makes a *near-free
//! disabled-log fast path* the core value of the diagnostic kernel: an
//! [`event!`](prism_diagnostic::event) call whose level is below the runtime
//! threshold must cost only an atomic load and a compare — the message is never
//! built or dispatched — so verbose instrumentation can stay compiled into
//! shipping builds. This benchmark measures the filtered (disabled) path and
//! the enabled dispatch path through a
//! [`CaptureSink`](prism_diagnostic::CaptureSink), reporting the per-call cost
//! of each, guarded by capture-count assertions so a filter that leaked events
//! (or an enabled path that silently dropped them) fails loudly.
//!
//! Dependency-free `harness = false` plain `main` using [`std::time::Instant`]:
//!
//! ```text
//! cargo bench -p prism_diagnostic --bench log_dispatch
//! ```
//!
//! # Provenance
//!
//! This is an original benchmark authored for Prism. It contains
//! **no Unreal Engine source or derived code**.
#![expect(
    clippy::print_stdout,
    reason = "a benchmark binary reports its timing results to stdout"
)]

extern crate alloc;

use alloc::sync::Arc;
use std::hint::black_box;
use std::time::Instant;

use prism_diagnostic::{clear_sink, event, set_max_level, set_sink, CaptureSink, Level};

/// Filtered (disabled) calls per timed pass — large, since each is ~1 atomic load.
const DISABLED_CALLS: usize = 1 << 24;
/// Enabled dispatch calls per timed pass — smaller, each locks + clones + pushes.
const ENABLED_CALLS: usize = 1 << 16;
/// Timed passes; the reported figure is the best (lowest-noise) pass.
const PASSES: usize = 16;

/// Best-of-`PASSES` nanoseconds per filtered (below-threshold) log call.
fn bench_disabled() -> f64 {
    // Only `Error` passes, so every `Info` call below takes the fast path.
    set_max_level(Level::Error);
    let mut best = f64::INFINITY;
    for _ in 0..PASSES {
        let start = Instant::now();
        for i in 0..DISABLED_CALLS {
            event!(Level::Info, "disabled message {}", black_box(i));
        }
        let per = start.elapsed().as_nanos() as f64 / DISABLED_CALLS as f64;
        best = best.min(per);
    }
    best
}

/// Best-of-`PASSES` nanoseconds per enabled (dispatched) log call. Installs a
/// fresh capture sink each pass (the facade has no reset) and returns the best
/// time plus the final pass's capture count for the correctness guard.
fn bench_enabled() -> (f64, usize) {
    set_max_level(Level::Trace); // everything passes
    let mut best = f64::INFINITY;
    let mut last_len = 0;
    for _ in 0..PASSES {
        let sink = Arc::new(CaptureSink::new());
        set_sink(sink.clone());
        let start = Instant::now();
        for i in 0..ENABLED_CALLS {
            event!(Level::Error, "enabled message {}", black_box(i));
        }
        let per = start.elapsed().as_nanos() as f64 / ENABLED_CALLS as f64;
        best = best.min(per);
        last_len = sink.len();
    }
    (best, last_len)
}

fn main() {
    // --- Filtered fast path ---
    // Prove the filter actually suppresses: nothing may reach the sink.
    let guard_sink = Arc::new(CaptureSink::new());
    set_sink(guard_sink.clone());
    set_max_level(Level::Error);
    event!(Level::Info, "should be filtered");
    assert_eq!(
        guard_sink.len(),
        0,
        "disabled-level event leaked to the sink"
    );

    let disabled_ns = bench_disabled();
    assert_eq!(
        guard_sink.len(),
        0,
        "disabled path dispatched events it should have skipped"
    );

    // --- Enabled dispatch path ---
    let (enabled_ns, captured) = bench_enabled();
    assert_eq!(
        captured, ENABLED_CALLS,
        "enabled path captured {captured} events, expected {ENABLED_CALLS}"
    );

    clear_sink();

    let disabled_mps = 1_000.0 / disabled_ns;
    let overhead_ratio = enabled_ns / disabled_ns;

    println!("prism_diagnostic log_dispatch (filtered vs dispatched)");
    println!("  disabled call : {disabled_ns:.3} ns  ({disabled_mps:.0} Mcalls/s, filtered)");
    println!("  enabled call  : {enabled_ns:.1} ns  (build + capture-sink dispatch)");
    println!("  dispatch cost : {overhead_ratio:.0}x the filtered fast path");
    println!("  guard         : 0 leaked while filtered, {ENABLED_CALLS} captured while enabled");
}
