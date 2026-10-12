//! The printed CPU-submission summary.
//!
//! Criterion reports the full statistics; this is the one-number takeaway the
//! M2 milestone asks for — how much per-frame CPU submission work the GPU-driven
//! path removes relative to one draw call per object. Two quantities are
//! reported per scene:
//!
//!  * **record** — the CPU time to assemble the frame's commands, with no GPU
//!    work submitted. This is the cost that scales with the object count, and it
//!    is what the reduction is computed from.
//!  * **end to end** — one frame from submission to GPU completion. This is a
//!    latency, so it carries a fixed floor that every frame pays (the commit
//!    plus the completion signal: a couple of hundred microseconds through wgpu
//!    here). That is why the GPU-driven path only wins this number once the
//!    object count is large enough for the per-object cost to dominate the floor.

use std::hint::black_box;
use std::time::{Duration, Instant};

use crate::harness::{Harness, Scene, SCENE_SIZES};

/// How many frames each manual measurement submits.
const SAMPLES: u32 = 64;

/// Measures and prints both quantities for both paths, and the ratios between
/// them, for every scene size in the sweep.
pub fn print_submission_reduction(harness: &Harness) {
    println!();
    println!("prism_gpu_cull: per-frame cost (mean of {SAMPLES})");
    println!(
        "{:>8}  {:>11}  {:>11}  {:>10}  {:>11}  {:>11}  {:>9}",
        "objects", "rec driven", "rec naive", "cpu x", "e2e driven", "e2e naive", "e2e x"
    );

    for &count in &SCENE_SIZES {
        let scene = Scene::new(count);
        harness.upload(&scene);

        let record_driven = measure_record(|| harness.encode_gpu_driven());
        let record_naive = measure_record(|| harness.encode_naive(&scene.visible));
        let e2e_driven = measure_end_to_end(harness, Harness::submit_gpu_driven);
        let e2e_naive = measure_end_to_end(harness, |harness| harness.submit_naive(&scene.visible));

        println!(
            "{:>8}  {:>9.2?}  {:>9.2?}  {:>9.1}x  {:>9.2?}  {:>9.2?}  {:>8.1}x",
            scene.object_count(),
            record_driven,
            record_naive,
            ratio(record_naive, record_driven),
            e2e_driven,
            e2e_naive,
            ratio(e2e_naive, e2e_driven),
        );
    }
    println!();
}

/// How much cheaper `a` is than `b`, as a multiplier.
fn ratio(a: Duration, b: Duration) -> f64 {
    a.as_secs_f64() / b.as_secs_f64()
}

/// The mean wall time of `SAMPLES` pure command-assembly calls.
///
/// The result is dropped inside the timed region, exactly as a real frame's
/// command buffer is dropped after submission.
fn measure_record<T>(mut encode: impl FnMut() -> T) -> Duration {
    black_box(encode());
    let start = Instant::now();
    for _ in 0..SAMPLES {
        black_box(encode());
    }
    start.elapsed() / SAMPLES
}

/// The mean wall time of one frame from submission to GPU completion.
///
/// Each frame is drained before the next is submitted, so no measurement
/// includes another frame's GPU time.
fn measure_end_to_end(harness: &Harness, mut submit: impl FnMut(&Harness)) -> Duration {
    // One warm-up frame: the first submission of a freshly uploaded scene is not
    // representative (lazily created GPU objects, cold caches).
    submit(harness);
    harness.wait_idle();

    let start = Instant::now();
    for _ in 0..SAMPLES {
        submit(harness);
        harness.wait_idle();
    }
    let elapsed = start.elapsed();
    elapsed / SAMPLES
}
