//! Multi-core parallel-scaling benchmark (roadmap M-core "基准即规格").
//!
//! The design spec (`docs/prism_tasks_design_zh.md`) makes *near-linear
//! multi-core scaling* the core value of the parallel layer: spreading an
//! embarrassingly-parallel workload across N worker threads via
//! [`TaskPool::par_for_each_mut`] should approach an N× speedup over a
//! single-threaded pool. This benchmark runs the identical workload on a
//! 1-thread pool and on an all-cores pool and reports the achieved scaling,
//! with a result checksum guard so a "fast" run that skipped work fails loudly.
//!
//! Dependency-free `harness = false` plain `main` using [`std::time::Instant`]:
//!
//! ```text
//! cargo bench -p prism_tasks --bench parallel_scaling
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

use prism_tasks::TaskPool;

/// Elements processed per pass.
const N: usize = 1 << 20;
/// Timed passes; the reported figure is the best (lowest-noise) pass.
const PASSES: usize = 32;

/// A deterministic, moderately expensive per-element kernel: a short integer
/// hash mix. Pure integer math keeps the result bit-stable so the checksum
/// guard is meaningful across runs and thread counts.
#[inline]
fn kernel(x: u64) -> u64 {
    let mut h = x;
    for _ in 0..24 {
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 29;
        h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    }
    h
}

fn seed(data: &mut [u64]) {
    for (i, slot) in data.iter_mut().enumerate() {
        *slot = i as u64;
    }
}

fn checksum(data: &[u64]) -> u64 {
    data.iter().fold(0u64, |acc, &x| acc.wrapping_add(x))
}

/// Best-of-`PASSES` wall time (seconds) for one pool, plus the result checksum.
fn bench_pool(pool: &TaskPool, data: &mut [u64]) -> (f64, u64) {
    let mut best = f64::INFINITY;
    let mut sum = 0;
    for _ in 0..PASSES {
        seed(data);
        let start = Instant::now();
        pool.par_for_each_mut(black_box(data), |x| *x = kernel(*x));
        let secs = start.elapsed().as_secs_f64();
        best = best.min(secs);
        sum = checksum(data);
        black_box(sum);
    }
    (best, sum)
}

fn main() {
    let cores = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);

    let single = TaskPool::with_threads(1);
    let multi = TaskPool::with_threads(cores);

    let mut data = vec![0u64; N];

    // Warm up both pools (thread spawn, first-touch pages).
    bench_pool(&single, &mut data);
    bench_pool(&multi, &mut data);

    let (single_s, sum1) = bench_pool(&single, &mut data);
    let (multi_s, sum2) = bench_pool(&multi, &mut data);

    // Correctness guard: identical work regardless of thread count.
    assert_eq!(
        sum1, sum2,
        "parallel result diverged from single-threaded: {sum1:#x} != {sum2:#x}"
    );

    let speedup = single_s / multi_s;
    let efficiency = speedup / cores as f64 * 100.0;
    let melems = N as f64 / multi_s / 1e6;

    println!("prism_tasks parallel_scaling (multi-core scaling)");
    println!("  elements/pass : {N}");
    println!("  worker threads: {cores}");
    println!("  1 thread      : {:.3} ms", single_s * 1e3);
    println!("  {cores} threads : {:.3} ms  ({melems:.1} Melems/s)", multi_s * 1e3);
    println!("  speedup       : {speedup:.2}x  ({efficiency:.0}% parallel efficiency)");
    println!("  checksum      : {sum1:#018x}  (thread-count invariant)");
}
