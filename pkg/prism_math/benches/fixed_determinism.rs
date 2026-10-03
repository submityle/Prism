//! Fixed-point transcendental throughput + determinism benchmark (roadmap M4).
//!
//! The design spec (`docs/prism_math_design_zh.md` §22) makes *fixed-point,
//! cross-platform bit-level determinism* the core value of M4: the same code
//! must produce a bit-identical state hash on every run and every machine,
//! otherwise networked/replay simulations desync. This benchmark does two
//! things:
//!
//! 1. measures the throughput of the fixed-point `sin`/`cos`/`exp` kernels, and
//! 2. folds every intermediate into a [`StateHasher`] and asserts the hash is
//!    bit-identical across two independent runs — an anti-vacuous determinism
//!    guard, so a "fast" result that is actually non-deterministic fails loudly.
//!
//! Like the other Prism benchmarks this is dependency-free: a `harness = false`
//! plain `main` using [`std::time::Instant`]. Run it with:
//!
//! ```text
//! cargo bench -p prism_math --bench fixed_determinism
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

use prism_math::{Fixed, StateHasher};

/// Transcendental evaluations per timed pass.
const OPS: usize = 1 << 18;
/// Timed passes; the reported figure is the best (lowest-noise) pass.
const PASSES: usize = 16;

/// Run the fixed-point kernel over a deterministic angle sweep, folding every
/// output into the hasher. Returns the final state hash so callers can compare
/// runs for bit-exactness.
fn run_kernel(hasher: &mut StateHasher) {
    // Sweep roughly [-2π, 2π) in fixed-point steps; derived purely from the
    // integer index so the sequence is identical on every platform.
    let step = Fixed::from_f32(0.000_1);
    // -2π, built from the crate's own PI constant to avoid an approx-TAU literal.
    let base = Fixed::PI.saturating_mul(Fixed::from_f32(-2.0));
    let mut acc = Fixed::from_f32(0.0);
    for i in 0..OPS {
        let angle = base.saturating_add(step.saturating_mul(Fixed::from_bits(i as i64)));
        let (s, c) = angle.sin_cos();
        let e = s.saturating_mul(c).exp();
        acc = acc.saturating_add(e);
        hasher.write_fixed(s);
        hasher.write_fixed(c);
        hasher.write_fixed(e);
    }
    hasher.write_fixed(acc);
}

fn main() {
    // --- determinism guard: two independent runs must hash identically -------
    let mut h1 = StateHasher::new();
    run_kernel(&mut h1);
    let hash1 = h1.finish();

    let mut h2 = StateHasher::new();
    run_kernel(&mut h2);
    let hash2 = h2.finish();

    assert_eq!(
        hash1, hash2,
        "fixed-point kernel is non-deterministic across runs: {hash1:#018x} != {hash2:#018x}"
    );

    // --- throughput ----------------------------------------------------------
    let mut best = f64::INFINITY;
    for _ in 0..PASSES {
        let mut h = StateHasher::new();
        let start = Instant::now();
        run_kernel(black_box(&mut h));
        let per = start.elapsed().as_nanos() as f64 / (OPS as f64 * 3.0);
        best = best.min(per);
        black_box(h.finish());
    }
    let mops = 1_000.0 / best;

    println!("prism_math fixed_determinism (M4 deterministic fixed-point)");
    println!("  ops/pass    : {} (sin+cos+exp)", OPS * 3);
    println!("  throughput  : {best:.3} ns/op  ({mops:.1} Mops/s)");
    println!("  state hash  : {hash1:#018x}  (bit-identical across 2 runs)");
}
