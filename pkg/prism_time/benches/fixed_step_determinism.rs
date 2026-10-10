//! Deterministic fixed-step accumulation throughput (roadmap M-core "基准即规格").
//!
//! The design spec (`docs/prism_time_design_zh.md`) makes *deterministic,
//! drift-free fixed-step accumulation* the core value of the time kernel: the
//! integer [`TickClock`](prism_time::determinism::TickClock) must advance by an
//! exact tick count under a jittered real-delta stream and reproduce
//! bit-identically on a re-run, so rollback/lockstep netcode stays in sync.
//! This benchmark drives two independent clocks with the *same* jittered delta
//! sequence, measures the accumulate/drain throughput, and guards the number
//! with a determinism assertion (both runs' serialized snapshots are
//! byte-identical) plus an anti-vacuous assertion (ticks actually advanced and
//! the interpolation alpha stays in `[0, 1)`).
//!
//! Dependency-free `harness = false` plain `main` using [`std::time::Instant`]:
//!
//! ```text
//! cargo bench -p prism_time --bench fixed_step_determinism
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

use std::hint::black_box;
use std::time::{Duration, Instant};

use prism_time::determinism::{RationalStep, TickClock};

/// Simulated frames per timed pass.
const FRAMES: usize = 1 << 20;
/// Timed passes; the reported figure is the best (lowest-noise) pass.
const PASSES: usize = 32;
/// Fixed timestep: a 120 Hz tick expressed as exact integer nanoseconds.
const STEP_NANOS: u64 = 8_333_333;

/// A trig-free deterministic jittered real-delta stream in nanoseconds. A plain
/// integer LCG keeps the sequence bit-stable across runs and thread-free, so
/// the determinism guard is meaningful; deltas stay within `[0.5, 1.5]` of the
/// step so the per-frame drain count stays under the death-spiral cap.
#[inline]
fn jitter_nanos(state: &mut u64) -> u64 {
    // Numerical Recipes LCG constants.
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    // Map the high bits into [STEP/2, STEP*3/2].
    let span = STEP_NANOS; // width of the jitter window
    let frac = (*state >> 40) % (span + 1); // 0..=span
    STEP_NANOS / 2 + frac
}

/// Run one clock over a freshly-regenerated delta sequence, returning the final
/// clock state and the total ticks drained.
fn run_clock(seed: u64) -> (TickClock, u64) {
    let mut clock = TickClock::new(RationalStep::from_nanos(STEP_NANOS));
    // A generous cap so the jitter window never trips the death-spiral guard
    // and the tick total is a pure function of the delta stream.
    clock.set_max_substeps(16);
    let mut state = seed;
    let mut total = 0u64;
    for _ in 0..FRAMES {
        let delta = Duration::from_nanos(jitter_nanos(&mut state));
        clock.accumulate(delta);
        total += clock.expend_all();
    }
    (clock, total)
}

/// Best-of-`PASSES` wall time (seconds) for the accumulate/drain loop.
fn bench(seed: u64) -> (f64, TickClock, u64) {
    let mut best = f64::INFINITY;
    let mut last = (TickClock::new(RationalStep::from_nanos(STEP_NANOS)), 0u64);
    for _ in 0..PASSES {
        let start = Instant::now();
        let out = run_clock(black_box(seed));
        best = best.min(start.elapsed().as_secs_f64());
        black_box(out.1);
        last = out;
    }
    (best, last.0, last.1)
}

fn main() {
    const SEED: u64 = 0x9E37_79B9_7F4A_7C15;

    // Warm up (page/branch-predictor) before timing.
    run_clock(SEED);

    let (secs, clock_a, ticks_a) = bench(SEED);

    // Determinism guard: a second independent run over the identical jittered
    // delta sequence must reproduce the exact same tick count and a
    // byte-identical serialized snapshot.
    let (clock_b, ticks_b) = run_clock(SEED);
    assert_eq!(
        ticks_a, ticks_b,
        "tick count diverged across identical runs"
    );
    assert_eq!(
        clock_a.snapshot().to_bytes(),
        clock_b.snapshot().to_bytes(),
        "serialized clock state diverged across identical runs"
    );

    // Anti-vacuous guards: real work happened and the interpolation alpha is a
    // valid presentation fraction.
    assert!(ticks_a > 0, "no ticks drained — the workload did nothing");
    assert!(
        ticks_a as usize >= FRAMES / 2 && ticks_a as usize <= FRAMES * 2,
        "tick total {ticks_a} outside the expected jitter band for {FRAMES} frames"
    );
    let alpha = clock_a.overstep_fraction_f64();
    assert!(
        (0.0..1.0).contains(&alpha),
        "overstep fraction {alpha} not in [0, 1)"
    );

    let frames_per_s = FRAMES as f64 / secs;
    let ns_per_frame = secs * 1e9 / FRAMES as f64;

    println!("prism_time fixed_step_determinism (deterministic fixed-step)");
    println!("  frames/pass   : {FRAMES}");
    println!(
        "  timestep      : {STEP_NANOS} ns  (~{:.1} Hz)",
        1e9 / STEP_NANOS as f64
    );
    println!("  wall time     : {:.3} ms", secs * 1e3);
    println!("  throughput    : {frames_per_s:.3e} frames/s  ({ns_per_frame:.2} ns/frame)");
    println!("  ticks drained : {ticks_a}  (final alpha {alpha:.4}, bit-identical re-run)");
}
