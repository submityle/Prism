//! Reflection path-access throughput (roadmap M-core "基准即规格").
//!
//! The design spec (`docs/prism_reflect_design_zh.md`) makes *fast runtime
//! path navigation* a core value of the reflection layer: an editor, a network
//! replicator, or a scripting binding resolves a pre-parsed
//! [`ParsedPath`](prism_reflect::ParsedPath) against a `dyn Reflect` root every
//! frame, so [`reflect_path`](prism_reflect::reflect_path) + a concrete
//! [`downcast_ref`](prism_reflect::Reflect) must stay cheap. This benchmark
//! amortizes the parse, resolves a two-segment path across a batch of values,
//! and reports both the achieved rate and the overhead factor versus a direct
//! native field read, guarded by a checksum that the resolved values match the
//! native ones so a "fast" run that returned `None`/garbage fails loudly.
//!
//! Dependency-free `harness = false` plain `main` using [`std::time::Instant`]:
//!
//! ```text
//! cargo bench -p prism_reflect --bench path_access
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

use prism_reflect::{ParsedPath, Reflect, reflect_path};

#[derive(Reflect)]
struct Stats {
    health: i32,
    speed: i32,
}

#[derive(Reflect)]
struct Entity {
    stats: Stats,
    level: i32,
}

/// Values in the resolved batch (so successive resolutions hit different data).
const BATCH: usize = 1 << 12;
/// Resolutions per timed pass.
const RESOLUTIONS: usize = 1 << 20;
/// Timed passes; the reported figure is the best (lowest-noise) pass.
const PASSES: usize = 24;

fn build_batch() -> Vec<Entity> {
    (0..BATCH)
        .map(|i| Entity {
            stats: Stats { health: i as i32, speed: (i as i32) * 2 - 7 },
            level: i as i32 % 60,
        })
        .collect()
}

/// Best-of-`PASSES` seconds for reflective path resolution + downcast, plus checksum.
fn bench_reflect(batch: &[Entity], path: &ParsedPath) -> (f64, i64) {
    let mut best = f64::INFINITY;
    let mut sum = 0i64;
    for _ in 0..PASSES {
        let start = Instant::now();
        let mut s = 0i64;
        for i in 0..RESOLUTIONS {
            let root: &dyn Reflect = &batch[i & (BATCH - 1)];
            let found = reflect_path(black_box(root), black_box(path)).unwrap();
            s += i64::from(*found.downcast_ref::<i32>().unwrap());
        }
        best = best.min(start.elapsed().as_secs_f64());
        sum = s;
        black_box(s);
    }
    (best, sum)
}

/// Best-of-`PASSES` seconds for the equivalent direct native field read, plus checksum.
fn bench_native(batch: &[Entity]) -> (f64, i64) {
    let mut best = f64::INFINITY;
    let mut sum = 0i64;
    for _ in 0..PASSES {
        let start = Instant::now();
        let mut s = 0i64;
        for i in 0..RESOLUTIONS {
            let e = &batch[i & (BATCH - 1)];
            s += i64::from(black_box(e).stats.health);
        }
        best = best.min(start.elapsed().as_secs_f64());
        sum = s;
        black_box(s);
    }
    (best, sum)
}

fn main() {
    assert!(BATCH.is_power_of_two(), "BATCH must be a power of two for the index mask");
    let batch = build_batch();
    let path = ParsedPath::parse(".stats.health").expect("valid access path");

    // Warm up.
    bench_reflect(&batch, &path);
    bench_native(&batch);

    let (reflect_s, reflect_sum) = bench_reflect(&batch, &path);
    let (native_s, native_sum) = bench_native(&batch);

    // Correctness guard: reflective resolution must return exactly the native
    // field values, otherwise the throughput number would be meaningless.
    assert_eq!(
        reflect_sum, native_sum,
        "reflected path checksum diverged from native field read: {reflect_sum} != {native_sum}"
    );
    assert_ne!(reflect_sum, 0, "checksum is zero — no useful work was measured");

    let reflect_mps = RESOLUTIONS as f64 / reflect_s / 1e6;
    let reflect_ns = reflect_s * 1e9 / RESOLUTIONS as f64;
    let overhead = reflect_s / native_s;

    println!("prism_reflect path_access (ParsedPath resolve + downcast)");
    println!("  path          : .stats.health  (2 segments)");
    println!("  resolutions   : {RESOLUTIONS}  over a {BATCH}-value batch");
    println!("  reflective    : {:.3} ms  ({reflect_mps:.1} Mresolve/s, {reflect_ns:.1} ns each)", reflect_s * 1e3);
    println!("  native field  : {:.3} ms", native_s * 1e3);
    println!("  overhead      : {overhead:.1}x native  (checksum {reflect_sum})");
}
