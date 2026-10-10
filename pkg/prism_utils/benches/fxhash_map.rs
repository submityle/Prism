//! `FxHashMap` insert/lookup throughput (roadmap M-core "基准即规格").
//!
//! The design spec (`docs/prism_utils_design_zh.md`) makes a *fast
//! engine-wide `HashMap`* the core value of the container kernel: the
//! [`prism_utils::HashMap`] facade over the integer-friendly `FxHasher` should
//! comfortably beat the standard-library default (`SipHash`) `HashMap` on the
//! small-integer keys that dominate ECS/handle lookups. This benchmark fills
//! and probes both maps with the identical key stream, reports the achieved
//! speedup, and guards the number with a correctness assertion (both maps
//! return the same probed-value checksum) so a "fast" run that lost entries
//! fails loudly.
//!
//! Dependency-free `harness = false` plain `main` using [`std::time::Instant`]:
//!
//! ```text
//! cargo bench -p prism_utils --bench fxhash_map
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

use std::collections::HashMap as StdHashMap;
use std::hint::black_box;
use std::time::Instant;

use prism_utils::hash::FxBuildHasher;
use prism_utils::HashMap as FxHashMap;

/// Entries inserted and then probed per pass.
const N: u64 = 1 << 20;
/// Timed passes; the reported figure is the best (lowest-noise) pass.
const PASSES: usize = 24;

/// A trig-free deterministic key stream: an integer LCG scramble so keys are
/// spread across the whole `u64` range (not densely sequential) while staying
/// bit-stable across runs for the checksum guard.
#[inline]
fn key(i: u64) -> u64 {
    let mut h = i.wrapping_add(0x9E37_79B9_7F4A_7C15);
    h ^= h >> 30;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^ (h >> 31)
}

/// Fill an `FxHashMap`, then probe every key, returning (best secs, checksum).
fn bench_fx() -> (f64, u64) {
    let mut best = f64::INFINITY;
    let mut sum = 0u64;
    for _ in 0..PASSES {
        let mut map: FxHashMap<u64, u64> =
            FxHashMap::with_capacity_and_hasher(N as usize, FxBuildHasher::default());
        let start = Instant::now();
        for i in 0..N {
            map.insert(black_box(key(i)), i);
        }
        let mut s = 0u64;
        for i in 0..N {
            s = s.wrapping_add(*map.get(&black_box(key(i))).unwrap());
        }
        best = best.min(start.elapsed().as_secs_f64());
        sum = s;
        black_box(&map);
    }
    (best, sum)
}

/// Fill a std default (`SipHash`) `HashMap`, then probe, returning (best secs, checksum).
fn bench_std() -> (f64, u64) {
    let mut best = f64::INFINITY;
    let mut sum = 0u64;
    for _ in 0..PASSES {
        let mut map: StdHashMap<u64, u64> = StdHashMap::with_capacity(N as usize);
        let start = Instant::now();
        for i in 0..N {
            map.insert(black_box(key(i)), i);
        }
        let mut s = 0u64;
        for i in 0..N {
            s = s.wrapping_add(*map.get(&black_box(key(i))).unwrap());
        }
        best = best.min(start.elapsed().as_secs_f64());
        sum = s;
        black_box(&map);
    }
    (best, sum)
}

fn main() {
    // Warm up allocator / caches.
    bench_fx();
    bench_std();

    let (fx_s, fx_sum) = bench_fx();
    let (std_s, std_sum) = bench_std();

    // Correctness guard: both maps must hold the identical key→value mapping,
    // so the probed-value checksums must match. Otherwise the hasher dropped or
    // corrupted entries and the throughput number would be meaningless.
    assert_eq!(
        fx_sum, std_sum,
        "FxHashMap checksum diverged from std HashMap: {fx_sum:#x} != {std_sum:#x}"
    );
    // Anti-vacuous: the checksum must reflect real probed values.
    assert_ne!(
        fx_sum, 0,
        "probe checksum is zero — the maps did no useful work"
    );

    let fx_mops = (2 * N) as f64 / fx_s / 1e6; // insert + lookup
    let std_mops = (2 * N) as f64 / std_s / 1e6;
    let speedup = std_s / fx_s;

    println!("prism_utils fxhash_map (FxHashMap vs std SipHash)");
    println!("  entries/pass  : {N}  (insert + probe)");
    println!(
        "  FxHashMap     : {:.3} ms  ({fx_mops:.1} Mops/s)",
        fx_s * 1e3
    );
    println!(
        "  std HashMap   : {:.3} ms  ({std_mops:.1} Mops/s)",
        std_s * 1e3
    );
    println!("  speedup       : {speedup:.2}x  (checksum {fx_sum:#018x})");
}
