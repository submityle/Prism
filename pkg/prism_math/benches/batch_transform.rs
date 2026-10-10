//! Batch `transform_points3` throughput benchmark (roadmap M5 "基准即规格").
//!
//! The design spec (`docs/prism_math_design_zh.md` §22) names *batch
//! `transform_points` throughput* as an acceptance metric: the SIMD-shaped
//! batch path in [`prism_math::batch`] should beat a naive per-point loop over
//! the public [`Mat4::transform_point3`] helper. This benchmark measures both
//! and reports the achieved speedup plus the runtime-detected active backend
//! (SSE2/NEON/scalar) so the number is interpretable on any host.
//!
//! Like the other Prism benchmarks this is intentionally dependency-free: it
//! uses a `harness = false` plain `main` and [`std::time::Instant`], so it runs
//! fully offline with no extra crates. Run it with:
//!
//! ```text
//! cargo bench -p prism_math --bench batch_transform
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
use std::time::Instant;

use prism_math::{batch, Mat4, MathCaps, Quat, Vec3};

/// Points transformed per timed pass. Large enough to dwarf loop overhead and
/// to exercise the batched inner kernel across many cache lines.
const POINTS: usize = 1 << 16;
/// Timed passes; the reported figure is the best (lowest-noise) pass.
const PASSES: usize = 64;

fn build_matrix() -> Mat4 {
    // A full non-trivial TRS so no column degenerates to the identity fast path.
    let rotation = Quat::from_axis_angle(Vec3::new(0.3, 0.7, 0.2).normalize(), 0.97);
    Mat4::from_scale_rotation_translation(
        Vec3::new(1.3, 0.8, 1.1),
        rotation,
        Vec3::new(12.0, -4.0, 7.5),
    )
}

fn build_points() -> Vec<Vec3> {
    // A cheap deterministic spread so successive points differ in all lanes.
    (0..POINTS)
        .map(|i| {
            let f = i as f32;
            // Deterministic, trig-free spread so every lane varies per index.
            let y = (f * 0.37) % 100.0 - 50.0;
            Vec3::new(f * 0.013 - 400.0, y, f * 0.019 - 620.0)
        })
        .collect()
}

/// Best-of-`PASSES` nanoseconds-per-point for the batched path.
fn bench_batch(m: &Mat4, src: &[Vec3], dst: &mut [Vec3]) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..PASSES {
        let start = Instant::now();
        batch::transform_points3(black_box(m), black_box(src), black_box(dst));
        black_box(&dst[dst.len() - 1]);
        let per = start.elapsed().as_nanos() as f64 / src.len() as f64;
        best = best.min(per);
    }
    best
}

/// Best-of-`PASSES` nanoseconds-per-point for a naive scalar loop.
fn bench_naive(m: &Mat4, src: &[Vec3], dst: &mut [Vec3]) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..PASSES {
        let start = Instant::now();
        for (d, s) in dst.iter_mut().zip(src.iter()) {
            *d = black_box(m).transform_point3(black_box(*s));
        }
        black_box(&dst[dst.len() - 1]);
        let per = start.elapsed().as_nanos() as f64 / src.len() as f64;
        best = best.min(per);
    }
    best
}

fn main() {
    let caps = MathCaps::detect();
    let m = build_matrix();
    let src = build_points();
    let mut dst = vec![Vec3::ZERO; src.len()];

    // Warm the caches / branch predictors before the timed passes.
    batch::transform_points3(&m, &src, &mut dst);

    let batch_ns = bench_batch(&m, &src, &mut dst);
    let naive_ns = bench_naive(&m, &src, &mut dst);

    // Correctness guard: the batch path must agree with the scalar reference,
    // otherwise the throughput number would be meaningless.
    let mut reference = vec![Vec3::ZERO; src.len()];
    for (d, s) in reference.iter_mut().zip(src.iter()) {
        *d = m.transform_point3(*s);
    }
    let mut worst = 0.0f32;
    for (a, b) in dst.iter().zip(reference.iter()) {
        worst = worst.max((*a - *b).length());
    }
    assert!(
        worst < 1e-2,
        "batch path diverged from scalar reference: {worst}"
    );

    let batch_mps = 1_000.0 / batch_ns;
    let naive_mps = 1_000.0 / naive_ns;
    let speedup = naive_ns / batch_ns;

    println!("prism_math batch_transform (M5 batch throughput)");
    println!("  active backend : {:?}", caps.active_backend());
    println!("  points/pass    : {POINTS}");
    println!("  batch path     : {batch_ns:.3} ns/point  ({batch_mps:.1} Mpoints/s)");
    println!("  naive path     : {naive_ns:.3} ns/point  ({naive_mps:.1} Mpoints/s)");
    println!("  speedup        : {speedup:.2}x  (max abs error {worst:.2e})");
}
