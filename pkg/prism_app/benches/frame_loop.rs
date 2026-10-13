//! `prism_app` frame-loop microbenchmarks (roadmap §22 "基准即规格").
//!
//! The design spec makes two of the App shell's core claims *measurable*, not
//! just structural: the empty frame loop must be cheap, and the fixed-timestep
//! heartbeat must be **deterministic** — identical jitter input must produce an
//! identical number of fixed substeps on every run. This benchmark reports the
//! empty-loop cost and asserts the determinism claim (a red/green criterion),
//! so a regression that makes the loop non-deterministic fails the benchmark.
//!
//! Dependency-free `harness = false` plain `main` using [`std::time::Instant`]:
//!
//! ```text
//! cargo bench -p prism_app --bench frame_loop
//! ```
//!
//! # Provenance
//!
//! Original benchmark authored for Prism. Contains no Unreal Engine or Unity
//! source or derived code.
#![expect(
    clippy::print_stdout,
    reason = "a benchmark binary reports its timing results to stdout"
)]

use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use prism_app::prelude::{App, FixedUpdate, TimeUpdateStrategy};
use prism_time::Duration;

/// Frames in the empty-loop timing pass.
const EMPTY_FRAMES: u64 = 50_000;
/// Timed empty-loop passes; the reported figure is the best (lowest-noise) pass.
const PASSES: usize = 5;

/// A deterministic jitter sequence (microseconds): mostly 60 Hz with a short
/// frame, a 2-substep frame, and one 250 ms hitch that the death-spiral guard
/// must absorb without an unbounded catch-up burst.
const JITTER_MICROS: [u64; 6] = [16_667, 20_000, 12_000, 33_000, 250_000, 16_667];

fn main() {
    let ns_per_frame = empty_frame_loop();
    println!(
        "empty frame loop: {ns_per_frame:.0} ns/frame (best of {PASSES} passes of {EMPTY_FRAMES} frames)"
    );

    let first = fixed_step_substeps();
    let second = fixed_step_substeps();
    assert_eq!(
        first, second,
        "identical jitter input must produce an identical fixed-substep count \
         (fixed-timestep determinism, design §8/§15)"
    );
    println!(
        "fixed-step under {}-frame jitter: {first} substeps (identical across two runs)",
        JITTER_MICROS.len()
    );
}

/// Time the empty frame loop (no plugins, no systems) and return ns/frame.
fn empty_frame_loop() -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..PASSES {
        let mut app = App::new();
        let start = Instant::now();
        for _ in 0..EMPTY_FRAMES {
            app.update();
        }
        let elapsed = start.elapsed();
        black_box(app.plugins_state());
        let per_frame = elapsed.as_nanos() as f64 / EMPTY_FRAMES as f64;
        best = best.min(per_frame);
    }
    best
}

/// Count the fixed substeps one full jitter sequence produces on a fresh app.
fn fixed_step_substeps() -> u64 {
    let steps = Arc::new(AtomicU64::new(0));
    let counter = Arc::clone(&steps);

    let mut app = App::new();
    app.set_fixed_timestep_hz(60.0);
    app.add_systems(FixedUpdate, move || {
        counter.fetch_add(1, Ordering::Relaxed);
    });

    for micros in JITTER_MICROS {
        app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_micros(
            micros,
        )));
        app.update();
    }
    steps.load(Ordering::Relaxed)
}
