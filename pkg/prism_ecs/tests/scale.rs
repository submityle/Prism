//! Hardware-scale evidence for the design's scale claims (§17 性能工程, §22
//! 诚实边界: 百万级实体 / 多核近线性).
//!
//! These tests allocate millions of entities, so they are `#[ignore]`d by
//! default and meant to be run explicitly on a real machine in release mode:
//!
//! ```text
//! cargo test -p prism_ecs --release --features multi_thread \
//!     --test scale -- --ignored --nocapture
//! ```
//!
//! They **assert correctness** (the arithmetic must be exact) and *print*
//! timings; they deliberately do not assert a wall-clock threshold, so they
//! stay informative on any machine instead of flaky. `perf_gate` below adds
//! non-ignored operation-count contracts and an ignored machine-readable sample
//! producer for `tools/ecs-perf/gate.py`. Only that external, fingerprint-checked
//! comparator enforces timing regressions on a controlled machine.

use std::hint::black_box;
use std::time::{Duration, Instant};

use prism_ecs::prelude::*;

/// Entities per scenario: the design's "百万级" target.
const ENTITIES: usize = 1_000_000;

/// Iterations of each timed body; the min is reported (least noise).
const REPEATS: usize = 3;

/// Table-backed position.
#[derive(Debug, Clone, Copy)]
struct Pos(f32, f32, f32);
impl Component for Pos {}

/// 512-byte table row used to exercise the high-cardinality chunk-index path.
#[derive(Debug, Clone, Copy)]
struct DirtyCell {
    value: u64,
    _padding: [u8; 504],
}
impl Component for DirtyCell {}

/// Table-backed velocity.
#[derive(Debug, Clone, Copy)]
struct Vel(f32, f32, f32);
impl Component for Vel {}

/// Shared key used to measure archetype creation and final-reference eviction.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[component(storage = "shared")]
struct ChurnKey(u32);

/// Spawn `ENTITIES` `(Pos, Vel)` entities into a fresh world.
fn spawn_world() -> (World, Entity) {
    let mut world = World::new();
    let mut first = None;
    for i in 0..ENTITIES {
        let e = world.spawn((Pos(i as f32, 0.0, 0.0), Vel(1.0, 2.0, 3.0)));
        first.get_or_insert(e);
    }
    (world, first.expect("ENTITIES > 0"))
}

/// Run `body` `REPEATS` times and return the fastest wall-clock duration.
fn best_of(mut body: impl FnMut()) -> Duration {
    let mut best = Duration::MAX;
    for _ in 0..REPEATS {
        let start = Instant::now();
        body();
        best = best.min(start.elapsed());
    }
    best
}

/// `ns` per entity, for a run over `ENTITIES` rows.
fn ns_per_entity(elapsed: Duration) -> f64 {
    elapsed.as_nanos() as f64 / ENTITIES as f64
}

/// Wall-clock samples from repeated timed runs — the raw material for a
/// reproducible scaling curve. Unlike [`best_of`] (which reports the fastest
/// run, the least noisy single number for a *cost* measurement), the curve needs
/// the spread so it can report a confidence interval.
#[cfg(feature = "multi_thread")]
struct Samples(Vec<Duration>);

#[cfg(feature = "multi_thread")]
impl Samples {
    /// Run `body` `n` times, collecting each wall-clock duration.
    fn collect(n: usize, mut body: impl FnMut()) -> Self {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let start = Instant::now();
            body();
            out.push(start.elapsed());
        }
        Self(out)
    }

    /// The `ns`-per-entity median (robust to a single descheduled run).
    fn median_ns_per_entity(&self) -> f64 {
        let mut ns: Vec<f64> = self.0.iter().map(|d| ns_per_entity(*d)).collect();
        ns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ns[ns.len() / 2]
    }

    /// Mean and 95% confidence-interval half-width (normal approximation) of the
    /// `ns`-per-entity samples, so a curve point can say how sure it is.
    fn mean_and_ci95_ns_per_entity(&self) -> (f64, f64) {
        let ns: Vec<f64> = self.0.iter().map(|d| ns_per_entity(*d)).collect();
        let n = ns.len() as f64;
        let mean = ns.iter().sum::<f64>() / n;
        if ns.len() < 2 {
            return (mean, 0.0);
        }
        let var = ns.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let se = (var / n).sqrt();
        (mean, 1.96 * se)
    }
}

#[test]
#[ignore = "allocates 1M entities; run explicitly with --release --ignored"]
fn scale_spawn_and_single_thread_propagate_1m() {
    let spawn_start = Instant::now();
    let (mut world, first) = spawn_world();
    let spawn_elapsed = spawn_start.elapsed();

    let q = world.query::<(&Vel, &mut Pos)>();
    // Warm the incremental match cache so the timed pass is steady-state.
    assert_eq!(q.iter_mut(&mut world).count(), ENTITIES);

    let run = best_of(|| {
        for (v, mut p) in q.iter_mut(&mut world) {
            p.0 += v.0;
            p.1 += v.1;
            p.2 += v.2;
        }
    });

    // Correctness: `best_of` runs the body `REPEATS` times, so the first entity
    // has been advanced exactly that many times (one propagate each).
    let reps = REPEATS as f32;
    let p = world.get::<Pos>(first).expect("first entity is live");
    assert_eq!((p.0, p.1, p.2), (1.0 * reps, 2.0 * reps, 3.0 * reps));

    println!(
        "spawn {ENTITIES}: {spawn_elapsed:?} ({:.0} ns/entity)",
        spawn_elapsed.as_nanos() as f64 / ENTITIES as f64
    );
    println!(
        "iter/&Vel,&mut Pos (1 thread): {run:?} ({:.2} ns/entity, {:.1} M entity/s)",
        ns_per_entity(run),
        ENTITIES as f64 / run.as_secs_f64() / 1.0e6,
    );
}

#[cfg(feature = "multi_thread")]
#[test]
#[ignore = "allocates 1M entities per pool size; run explicitly with --release --ignored"]
fn scale_par_propagate_1m_scaling() {
    let mut baseline = None;
    for threads in [1usize, 2, 4, 8] {
        let pool = prism_tasks::TaskPool::with_threads(threads);
        let (mut world, first) = spawn_world();
        let q = world.query::<(&Vel, &mut Pos)>();

        let run = best_of(|| {
            q.par_for_each_mut(&mut world, &pool, 1024, |(v, mut p)| {
                p.0 += v.0;
                p.1 += v.1;
                p.2 += v.2;
            });
        });
        let base = *baseline.get_or_insert(run);

        // Each repeat applies one propagate to every entity.
        let reps = REPEATS as f32;
        let p = world.get::<Pos>(first).expect("first entity is live");
        assert_eq!((p.0, p.1, p.2), (1.0 * reps, 2.0 * reps, 3.0 * reps));

        println!(
            "par(&Vel,&mut Pos) threads={threads}: {run:?} ({:.2} ns/entity, {:.1} M entity/s, speedup {:.2}x)",
            ns_per_entity(run),
            ENTITIES as f64 / run.as_secs_f64() / 1.0e6,
            base.as_secs_f64() / run.as_secs_f64(),
        );
    }
}

/// Reproducible **multi-core scaling curve** (design §17 / §22: the
/// near-linear-scaling claim needs an exclusive machine).
///
/// Reports, for a set of pool sizes, the median ns/entity, its 95% confidence
/// interval, the speedup over one worker and the parallel efficiency
/// (`speedup / threads`). The numbers are only meaningful on an idle, exclusive
/// host — on a shared dev box other agents' builds dominate the variance, which
/// is exactly why the curve is `#[ignore]`d and prints its own caveat — but the
/// *procedure* is fixed here so it can be reproduced verbatim on a dedicated
/// machine, and the confidence intervals make a noisy run self-evident.
///
/// Run with:
/// `cargo test -p prism_ecs --release --features multi_thread --test scale \
///  -- --ignored --nocapture scale_parallel_speedup_curve`
#[cfg(feature = "multi_thread")]
#[test]
#[ignore = "allocates 1M entities; run explicitly with --release --ignored"]
fn scale_parallel_speedup_curve() {
    const SAMPLES: usize = 9;
    let host = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    println!(
        "parallel speedup curve: {ENTITIES} entities, {SAMPLES} samples/point, host_parallelism={host}"
    );
    println!("NOTE: an idle, exclusive host is required for a trustworthy curve; on a shared");
    println!("      machine the 95% CI below will be wide and the speedups pessimistic.");
    println!();
    println!("threads | median ns/ent | mean ns/ent ± 95% CI | speedup | efficiency");

    // One world, reused across every pool size, so the pool is the only variable.
    let (mut world, first) = spawn_world();
    let q = world.query::<(&Vel, &mut Pos)>();
    let _ = q.iter_mut(&mut world).count(); // warm the match cache

    let mut sizes: Vec<usize> = vec![1, 2, 4, 8];
    if !sizes.contains(&host) {
        sizes.push(host);
    }
    sizes.sort_unstable();
    sizes.dedup();

    let mut total_runs = 0usize;
    let mut baseline = None;
    for threads in sizes {
        let pool = prism_tasks::TaskPool::with_threads(threads);
        let samples = Samples::collect(SAMPLES, || {
            q.par_for_each_mut(&mut world, &pool, 1024, |(v, mut p)| {
                p.0 += v.0;
                p.1 += v.1;
                p.2 += v.2;
            });
        });
        total_runs += SAMPLES;
        let median = samples.median_ns_per_entity();
        let (mean, ci) = samples.mean_and_ci95_ns_per_entity();
        let base = *baseline.get_or_insert(mean);
        println!(
            "{threads:>7} | {median:>13.2} | {mean:>8.2} ± {ci:>6.2}        | {:>6.2}x  | {:>5.1}%",
            base / mean,
            100.0 * base / mean / threads as f64,
        );
    }

    // Correctness: every sample propagated every entity, so the first entity was
    // advanced by the total number of runs (one propagate each).
    let runs = total_runs as f32;
    let p = world.get::<Pos>(first).expect("first entity is live");
    assert!(p.0.is_finite() && p.1.is_finite() && p.2.is_finite());
    assert_eq!(p.0, 1.0 * runs);
}

/// Diagnose the serial-vs-batch gap measured in
/// [`scale_spawn_and_single_thread_propagate_1m`] / [`scale_par_propagate_1m_scaling`]:
/// split it into read-only vs mutable and serial vs one-worker batch, so we can
/// tell whether the serial driver loop itself is slow or whether the gap comes
/// from per-row change-tick stamping (which the batch path amortises).
#[cfg(feature = "multi_thread")]
#[test]
#[ignore = "allocates 1M entities per case; run explicitly with --release --ignored"]
fn scale_serial_vs_batch_diagnosis() {
    let pool = prism_tasks::TaskPool::with_threads(1);

    // Read-only: no change-tick store at all.
    let (mut world, _) = spawn_world();
    let q_ro = world.query::<(&Vel, &Pos)>();
    let _ = q_ro.iter(&world).count();
    let serial_ro = best_of(|| {
        let mut acc = 0.0f32;
        for (v, p) in q_ro.iter(&world) {
            acc += v.0 + p.0;
        }
        black_box(acc);
    });
    // The batched `fold` fast path (design §17.1): `for_each`/`sum`/`count`
    // dispatch through `QueryIter::fold`, which walks each archetype as a tight
    // inner loop instead of the per-row `next()` protocol.
    let fold_ro = best_of(|| {
        let mut acc = 0.0f32;
        q_ro.iter(&world).for_each(|(v, p)| {
            acc += v.0 + p.0;
        });
        black_box(acc);
    });
    let par_ro = best_of(|| {
        q_ro.par_for_each(&world, &pool, 1024, |(v, p)| {
            black_box(v.0 + p.0);
        });
    });
    drop(world);

    // Mutable: each row stamps its change tick.
    let (mut world, _) = spawn_world();
    let q_rw = world.query::<(&Vel, &mut Pos)>();
    let _ = q_rw.iter_mut(&mut world).count();
    let serial_rw = best_of(|| {
        for (v, mut p) in q_rw.iter_mut(&mut world) {
            p.0 += v.0;
            p.1 += v.1;
            p.2 += v.2;
        }
    });
    let fold_rw = best_of(|| {
        q_rw.iter_mut(&mut world).for_each(|(v, mut p)| {
            p.0 += v.0;
            p.1 += v.1;
            p.2 += v.2;
        });
    });
    let par_rw = best_of(|| {
        q_rw.par_for_each_mut(&mut world, &pool, 1024, |(v, mut p)| {
            p.0 += v.0;
            p.1 += v.1;
            p.2 += v.2;
        });
    });

    println!(
        "read-only  for-next {:>6.2} | iter.for_each {:>6.2} | par(1) {:>6.2} ns/entity | fold speedup {:.2}x",
        ns_per_entity(serial_ro),
        ns_per_entity(fold_ro),
        ns_per_entity(par_ro),
        serial_ro.as_secs_f64() / fold_ro.as_secs_f64(),
    );
    println!(
        "mutable    for-next {:>6.2} | iter_mut.for_each {:>6.2} | par(1) {:>6.2} ns/entity | fold speedup {:.2}x",
        ns_per_entity(serial_rw),
        ns_per_entity(fold_rw),
        ns_per_entity(par_rw),
        serial_rw.as_secs_f64() / fold_rw.as_secs_f64(),
    );
}

/// `World::spawn_batch` resolves the archetype once per run instead of once per
/// entity; this measures what that is worth on a mass-spawn.
#[test]
#[ignore = "allocates 1M entities twice; run explicitly with --release --ignored"]
fn scale_spawn_batch_vs_individual_1m() {
    let individual = best_of(|| {
        let mut world = World::new();
        for i in 0..ENTITIES {
            world.spawn((Pos(i as f32, 0.0, 0.0), Vel(1.0, 2.0, 3.0)));
        }
        black_box(world.entity_count());
    });

    let batched = best_of(|| {
        let mut world = World::new();
        let spawned =
            world.spawn_batch((0..ENTITIES).map(|i| (Pos(i as f32, 0.0, 0.0), Vel(1.0, 2.0, 3.0))));
        black_box(spawned.len());
    });

    println!(
        "spawn 1M: individual {individual:?} ({:.0} ns/entity) | spawn_batch {batched:?} ({:.0} ns/entity) | {:.2}x",
        ns_per_entity(individual),
        ns_per_entity(batched),
        individual.as_secs_f64() / batched.as_secs_f64(),
    );
}

/// Measure shared-value archetype eviction while unrelated archetypes sharing
/// the same table columns remain live. This exercises the `by_shared_binding`
/// candidate posting and affected-posting cleanup cost.
///
/// The output is diagnostic, not a timing gate: run on an otherwise idle host
/// before comparing sizes.
#[test]
#[ignore = "creates up to 4097 archetypes; run explicitly with --release --ignored"]
fn scale_shared_binding_eviction_churn() {
    const EVICTIONS_PER_SAMPLE: usize = 1_024;
    const SAMPLES: usize = 7;
    for live_values in [64usize, 256, 1_024, 4_096] {
        let mut world = World::new();
        world.spawn((Pos(-1.0, 0.0, 0.0), ChurnKey(u32::MAX)));
        for value in 0..live_values {
            world.spawn((Pos(value as f32, 0.0, 0.0), ChurnKey(value as u32)));
        }

        let mut samples = Vec::with_capacity(SAMPLES);
        for sample in 0..SAMPLES {
            let start = Instant::now();
            for offset in 0..EVICTIONS_PER_SAMPLE {
                let value = (live_values + sample * EVICTIONS_PER_SAMPLE + offset) as u32;
                let entity = world.spawn((Pos(value as f32, 1.0, 0.0), ChurnKey(value)));
                assert!(world.despawn(entity));
            }
            samples.push(start.elapsed());
            assert_eq!(world.entity_count() as usize, live_values + 1);
        }
        samples.sort_unstable();
        let median = samples[SAMPLES / 2];
        println!(
            "shared_binding_eviction_churn live_values={live_values} samples={SAMPLES} evictions/sample={EVICTIONS_PER_SAMPLE} median={median:?} median_ns/eviction={:.0}",
            median.as_nanos() as f64 / EVICTIONS_PER_SAMPLE as f64
        );
        black_box(&world);
    }
}

/// Measure the conservative non-suffix cleanup path. Target archetypes are
/// created before the background archetypes, so each target eviction removes
/// an older ID from the head of a still-live table-component posting.
///
/// The output is diagnostic, not a timing gate: run on an otherwise idle host
/// before comparing sizes.
#[test]
#[ignore = "creates up to 5120 archetypes; run explicitly with --release --ignored"]
fn scale_shared_binding_eviction_non_tail() {
    const EVICTIONS_PER_SAMPLE: usize = 1_024;
    const SAMPLES: usize = 7;
    for live_values in [64usize, 256, 1_024, 4_096] {
        let mut samples = Vec::with_capacity(SAMPLES);
        for sample in 0..SAMPLES {
            let mut world = World::new();
            let targets: Vec<_> = (0..EVICTIONS_PER_SAMPLE)
                .map(|offset| {
                    let value = (sample * EVICTIONS_PER_SAMPLE + offset) as u32;
                    world.spawn((Pos(value as f32, 1.0, 0.0), ChurnKey(value)))
                })
                .collect();

            for value in 0..live_values {
                let key = (EVICTIONS_PER_SAMPLE * SAMPLES + value) as u32;
                world.spawn((Pos(value as f32, 0.0, 0.0), ChurnKey(key)));
            }

            let start = Instant::now();
            for entity in targets {
                assert!(world.despawn(entity));
            }
            samples.push(start.elapsed());
            assert_eq!(world.entity_count() as usize, live_values);
        }
        samples.sort_unstable();
        let median = samples[SAMPLES / 2];
        println!(
            "shared_binding_eviction_non_tail live_values={live_values} samples={SAMPLES} evictions/sample={EVICTIONS_PER_SAMPLE} median={median:?} median_ns/eviction={:.0}",
            median.as_nanos() as f64 / EVICTIONS_PER_SAMPLE as f64
        );
    }
}

/// Compare the normal forward-journal path against a per-row Changed<T> scan on
/// 1M rows, with touched rows clustered and scattered. Each sample uses a fresh
/// World so retained journal history from earlier samples cannot force the
/// bounded-log fallback. TOUCHED stays below the per-world journal capacity.
#[test]
#[ignore = "allocates 1M entities twice; run explicitly with --release --ignored"]
fn scale_dirty_chunks_vs_changed_scan_1m() {
    use prism_ecs::entity::Entity;

    const TOUCHED: usize = 4_000; // 0.4% of the world; below journal capacity
    const SAMPLES: usize = 9;

    fn summarize(samples: &[Duration]) -> (Duration, Duration, Duration) {
        let mut sorted = samples.to_vec();
        sorted.sort_unstable();
        (
            sorted[sorted.len() / 2],
            sorted[0],
            sorted[sorted.len() - 1],
        )
    }

    fn run_scenario(clustered: bool) {
        let stride = ENTITIES / TOUCHED;
        let mut scan_samples = Vec::with_capacity(SAMPLES);
        let mut dirty_samples = Vec::with_capacity(SAMPLES);
        let mut changed_rows = 0;
        let mut dirty_rows = 0;
        for _ in 0..SAMPLES {
            let mut world = World::new();
            let entities: Vec<Entity> = world
                .spawn_batch((0..ENTITIES).map(|i| (Pos(i as f32, 0.0, 0.0), Vel(1.0, 2.0, 3.0))));

            // Build both states and establish the dirty journal cursor before
            // any measured writes. This makes the measured pass exercise the
            // forward journal path rather than query-activation fallback.
            let any = world.query::<&Vel>();
            let changed = world.query_filtered::<&Vel, Changed<Vel>>();
            let last_run = world.change_tick();
            assert!(any
                .dirty_chunks_with_stats(&world, last_run, last_run)
                .chunks
                .is_empty());

            world.set_last_change_tick(last_run);
            world.increment_change_tick();
            for k in 0..TOUCHED {
                let index = if clustered { k } else { k * stride };
                world.get_mut::<Vel>(entities[index]).expect("live").0 += 1.0;
            }
            let this_run = world.change_tick();

            // Per-row change-detection scan: visits every matched row.
            let scan_start = Instant::now();
            changed_rows = changed.iter(&world).count();
            scan_samples.push(scan_start.elapsed());

            // Coarse chunk-granular pass. Assert that this is a complete
            // forward journal sample, not a fallback or a missed write.
            let dirty_start = Instant::now();
            let report = any.dirty_chunks_with_stats(&world, last_run, this_run);
            dirty_samples.push(dirty_start.elapsed());
            assert!(!report.stats.journal_gap);
            assert_eq!(report.stats.journal_events, TOUCHED);
            dirty_rows = report.chunks.iter().map(|chunk| chunk.len()).sum();
            assert_eq!(changed_rows, TOUCHED);
            assert!(dirty_rows >= changed_rows);
        }

        let (scan_median, scan_min, scan_max) = summarize(&scan_samples);
        let (dirty_median, dirty_min, dirty_max) = summarize(&dirty_samples);
        let label = if clustered { "clustered" } else { "scattered" };
        println!(
            "{label:>9}: Changed<T> median {scan_median:?} ({changed_rows} rows; {scan_min:?}..{scan_max:?}) | dirty journal median {dirty_median:?} ({dirty_rows} rows covered; {dirty_min:?}..{dirty_max:?}; {SAMPLES} samples) | {:.1}x",
            scan_median.as_secs_f64() / dirty_median.as_secs_f64(),
        );
    }

    run_scenario(true);
    run_scenario(false);
}

/// The `simd` feature ships bulk column kernels against a scalar ground truth;
/// the correctness dual-run lives in the crate, this measures what the vector
/// path is actually worth on this host.
#[cfg(feature = "simd")]
#[test]
#[ignore = "allocates 4 MiB of f32; run explicitly with --release --ignored"]
fn scale_simd_kernels_vs_scalar() {
    use prism_ecs::query::{active, add_assign, axpy_assign, scale_assign, sum};

    println!("simd backend: {:?}", active());

    // Two working-set sizes per operand: 4 KiB stays L1-resident (ALU-bound,
    // where the explicit lanes help) and 4 MiB streams past cache
    // (bandwidth-bound, where a scalar loop is already memory-limited and the
    // vector path has no headroom). Reporting both keeps the claim honest.
    for &n in &[1usize << 10, 1 << 20] {
        let mut dst = vec![1.0f32; n];
        let src = vec![2.0f32; n];

        // Warm every path once so first-touch page faults are not measured.
        add_assign(&mut dst, &src);
        for (d, s) in dst.iter_mut().zip(&src) {
            *d += *s;
        }
        scale_assign(&mut dst, 0.5);
        axpy_assign(&mut dst, &src, 0.25);
        black_box(sum(&dst));
        let mut warm = 0.0f32;
        for v in &dst {
            warm += *v;
        }
        black_box(warm);

        let vector_add = best_of(|| add_assign(&mut dst, &src));
        let scalar_add = best_of(|| {
            for (d, s) in dst.iter_mut().zip(&src) {
                *d += *s;
            }
        });
        let vector_scale = best_of(|| scale_assign(&mut dst, 0.5));
        let vector_axpy = best_of(|| axpy_assign(&mut dst, &src, 0.25));
        let vector_sum = best_of(|| {
            black_box(sum(&dst));
        });
        let scalar_sum = best_of(|| {
            let mut acc = 0.0f32;
            for v in &dst {
                acc += *v;
            }
            black_box(acc);
        });

        let ns_per = |d: Duration| d.as_nanos() as f64 / n as f64;
        let kib = (n * size_of::<f32>()) as f64 / 1024.0;
        println!(
            "n={n} ({kib:.0} KiB) add: vec {:?} ({:.3} ns/elem) | scalar {:?} ({:.3} ns/elem) | {:.2}x",
            vector_add,
            ns_per(vector_add),
            scalar_add,
            ns_per(scalar_add),
            scalar_add.as_secs_f64() / vector_add.as_secs_f64(),
        );
        println!(
            "n={n} ({kib:.0} KiB) scale: vec {:?} ({:.3} ns/elem) | axpy: vec {:?} ({:.3} ns/elem)",
            vector_scale,
            ns_per(vector_scale),
            vector_axpy,
            ns_per(vector_axpy),
        );
        println!(
            "n={n} ({kib:.0} KiB) sum: vec {:?} ({:.3} ns/elem) | scalar {:?} ({:.3} ns/elem) | {:.2}x",
            vector_sum,
            ns_per(vector_sum),
            scalar_sum,
            ns_per(scalar_sum),
            scalar_sum.as_secs_f64() / vector_sum.as_secs_f64(),
        );
    }
}

mod perf_gate {
    use super::*;
    use prism_ecs::archetype::{Archetype, COLUMN_CHANGE_LOG_CAPACITY};
    use prism_ecs::change::Tick;
    use prism_ecs::component::Components;
    use prism_ecs::query::{Access, QueryFilter, QueryState};
    use prism_ecs::storage::SparseSets;
    use std::collections::BTreeSet;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    static MATCH_CALLS: AtomicUsize = AtomicUsize::new(0);
    static MATCH_TEST_LOCK: Mutex<()> = Mutex::new(());

    struct CountMatches;

    // SAFETY: this is an always-true filter; it accesses no component values or
    // ticks and only counts calls to the archetype predicate.
    unsafe impl QueryFilter for CountMatches {
        type State = ();
        type Fetch<'w> = ();

        fn init_state(_: &mut Components) -> Self::State {}

        fn matches(_: &(), _: &Archetype) -> bool {
            MATCH_CALLS.fetch_add(1, Ordering::Relaxed);
            true
        }

        fn update_access(_: &(), _: &mut Access) {}

        unsafe fn init_fetch<'w>(
            _: &Self::State,
            _: &'w Archetype,
            _: &'w SparseSets,
            _: Tick,
            _: Tick,
        ) -> Self::Fetch<'w> {
        }

        unsafe fn filter_fetch<'w>(_: Self::Fetch<'w>, _: Entity, _: usize) -> bool {
            true
        }
    }

    #[derive(Clone, Copy)]
    struct Marker;
    impl Component for Marker {}

    #[test]
    fn perf_contract_query_matching_is_incremental_and_restore_invalidates() {
        let _lock = MATCH_TEST_LOCK.lock().unwrap();
        let mut world = World::new();
        world.register_snapshot_component::<Pos>();
        world.register_snapshot_component::<Vel>();
        world.spawn((Pos(1.0, 0.0, 0.0),));
        world.spawn((Pos(2.0, 0.0, 0.0), Vel(1.0, 0.0, 0.0)));
        let snapshot = world.snapshot();
        // Entity data matches every archetype, so short-circuiting D::matches
        // cannot hide any filter predicate calls, including empty archetypes.
        let query = world.query_filtered::<Entity, CountMatches>();
        MATCH_CALLS.store(0, Ordering::Relaxed);
        assert_eq!(query.iter(&world).count(), 2);
        assert_eq!(
            MATCH_CALLS.swap(0, Ordering::Relaxed),
            world.archetypes().len()
        );

        for _ in 0..32 {
            assert_eq!(query.iter(&world).count(), 2);
        }
        assert_eq!(MATCH_CALLS.swap(0, Ordering::Relaxed), 0);

        world.spawn((Pos(3.0, 0.0, 0.0), Marker));
        assert_eq!(query.iter(&world).count(), 3);
        assert_eq!(MATCH_CALLS.swap(0, Ordering::Relaxed), 1);
        world.spawn((Pos(4.0, 0.0, 0.0), Marker));
        assert_eq!(query.iter(&world).count(), 4);
        assert_eq!(MATCH_CALLS.swap(0, Ordering::Relaxed), 0);

        world.restore(&snapshot);
        assert_eq!(query.iter(&world).count(), 2);
        assert_eq!(
            MATCH_CALLS.swap(0, Ordering::Relaxed),
            world.archetypes().len()
        );
        assert_eq!(query.iter(&world).count(), 2);
        assert_eq!(MATCH_CALLS.swap(0, Ordering::Relaxed), 0);
    }

    // These assertions bound reported output, not internal discovery work.
    // The indexed workload below separately checks actual traversal counters.
    #[test]
    fn perf_contract_dirty_output_tracks_touched_chunks_not_world_rows() {
        for count in [256usize, 16_384] {
            let mut world = World::new();
            let entities = world.spawn_batch((0..count).map(|i| (Pos(i as f32, 0.0, 0.0),)));
            let query = world.query::<&Pos>();
            let last = world.change_tick();
            world.set_last_change_tick(last);
            world.increment_change_tick();
            let now = world.change_tick();
            assert!(query.dirty_chunks(&world, last, now).is_empty());

            let touched = [entities[0], entities[count / 2], entities[count - 1]];
            let mut expected = BTreeSet::new();
            for entity in touched {
                world.get_mut::<Pos>(entity).unwrap().0 += 1.0;
                let loc = world.entities().location(entity).unwrap();
                let rpc = world
                    .archetypes()
                    .get(loc.archetype_id)
                    .unwrap()
                    .table()
                    .rows_per_chunk();
                expected.insert((loc.archetype_id, loc.row as usize / rpc * rpc));
            }
            let dirty = query.dirty_chunks(&world, last, now);
            assert_eq!(dirty.len(), expected.len());
            assert_eq!(
                dirty
                    .iter()
                    .map(|d| (d.archetype, d.start))
                    .collect::<BTreeSet<_>>(),
                expected
            );
            assert_eq!(
                world
                    .query_filtered::<&Pos, Changed<Pos>>()
                    .iter(&world)
                    .count(),
                3
            );
            world.set_last_change_tick(now);
            world.increment_change_tick();
            assert!(query
                .dirty_chunks(&world, now, world.change_tick())
                .is_empty());
        }
    }

    #[cfg(all(feature = "multi_thread", feature = "simd"))]
    const SAMPLES: usize = 101;

    #[cfg(all(feature = "multi_thread", feature = "simd"))]
    const WARMUPS: usize = 5;

    // Keep all 10,000 writes in each sample, but consume two forward windows
    // before the bounded journal can lose history. Never time a replay/fallback
    // and present it as normal incremental discovery.
    const DIRTY_WRITES: usize = 10_000;
    const DIRTY_WINDOW_WRITES: usize = 5_000;
    const DIRTY_QUERIES: usize = DIRTY_WRITES.div_ceil(DIRTY_WINDOW_WRITES);
    const DIRTY_ROWS_PER_CHUNK: usize = 32;

    fn dirty_world(count: usize) -> (World, Vec<Entity>, QueryState<&'static DirtyCell>) {
        let mut world = World::new();
        let entities = world.spawn_batch((0..count).map(|_| DirtyCell {
            value: 0,
            _padding: [0; 504],
        }));
        let location = world.entities().location(entities[0]).unwrap();
        let table = world
            .archetypes()
            .get(location.archetype_id)
            .unwrap()
            .table();
        assert_eq!(size_of::<DirtyCell>(), 512);
        assert_eq!(table.rows_per_chunk(), DIRTY_ROWS_PER_CHUNK);
        let query = world.query::<&DirtyCell>();
        let tick = world.change_tick();
        // Establish a cursor after spawning; spawn overflow is not the workload.
        assert!(query.dirty_chunks(&world, tick, tick).is_empty());
        (world, entities, query)
    }

    // All writes, expected-set construction, assertions and report destruction
    // are outside the timed interval. Both paths verify the exact chunk set.
    fn dirty_window_sample(
        world: &mut World,
        entities: &[Entity],
        query: &QueryState<&DirtyCell>,
        stride: usize,
        indices: std::ops::Range<usize>,
        expect_gap: bool,
    ) -> Duration {
        let writes = indices.len();
        assert!(stride > 0 && (indices.end - 1) * stride < entities.len());
        let last = world.change_tick();
        world.set_last_change_tick(last);
        world.increment_change_tick();
        let now = world.change_tick();
        let mut expected = BTreeSet::new();
        for index in indices {
            let entity = entities[index * stride];
            world.get_mut::<DirtyCell>(entity).unwrap().value += 1;
            let location = world.entities().location(entity).unwrap();
            expected.insert((
                location.archetype_id,
                location.row as usize / DIRTY_ROWS_PER_CHUNK * DIRTY_ROWS_PER_CHUNK,
            ));
        }
        // More than INLINE_CHUNKS (currently 8): clustered Pos used to stay
        // inline and could not measure the indexed high-cardinality path.
        assert!(
            expected.len() > 8,
            "workload no longer exercises indexed candidates"
        );
        let start = Instant::now();
        let report = query.dirty_chunks_with_stats(world, last, now);
        let elapsed = start.elapsed();
        assert_eq!(
            report.stats.journal_gap, expect_gap,
            "unexpected dirty path"
        );
        if expect_gap {
            assert_eq!(report.stats.journal_events, 0);
            assert_eq!(report.stats.journal_records_scanned, 0);
            assert!(
                report.stats.summaries_tested > 0,
                "fallback did not visit interval index"
            );
            assert!(report.stats.archetype_index_entries_scanned > 0);
        } else {
            assert_eq!(report.stats.journal_events, writes);
            assert_eq!(report.stats.journal_records_scanned, writes);
            assert_eq!(report.stats.candidate_chunk_marks, writes);
            assert_eq!(
                report.stats.summaries_tested, 0,
                "unexpected index fallback"
            );
            assert_eq!(report.stats.archetype_index_entries_scanned, 0);
        }
        assert_eq!(report.stats.candidate_archetype_checks, 1);
        assert_eq!(report.stats.matched_archetypes, 1);
        assert_eq!(report.stats.merged_candidates, expected.len());
        assert_eq!(report.stats.reported_chunks, expected.len());
        assert_eq!(report.chunks.len(), expected.len());
        assert_eq!(
            report
                .chunks
                .iter()
                .map(|chunk| (chunk.archetype, chunk.start))
                .collect::<BTreeSet<_>>(),
            expected
        );
        assert!(report.chunks.windows(2).all(|pair| {
            (pair[0].archetype, pair[0].start) < (pair[1].archetype, pair[1].start)
        }));
        black_box(report);
        elapsed
    }

    fn forward_dirty_sample(
        world: &mut World,
        entities: &[Entity],
        query: &QueryState<&DirtyCell>,
        stride: usize,
    ) -> Duration {
        assert!(DIRTY_WINDOW_WRITES <= COLUMN_CHANGE_LOG_CAPACITY);
        assert!(stride > 0 && (DIRTY_WRITES - 1) * stride < entities.len());
        let mut elapsed = Duration::ZERO;
        for first in (0..DIRTY_WRITES).step_by(DIRTY_WINDOW_WRITES) {
            let end = (first + DIRTY_WINDOW_WRITES).min(DIRTY_WRITES);
            elapsed += dirty_window_sample(world, entities, query, stride, first..end, false);
        }
        elapsed // Sum of DIRTY_QUERIES calls, not an average or write timing.
    }

    #[test]
    fn perf_contract_dirty_forward_windows_retain_all_ten_thousand_writes() {
        assert_eq!(DIRTY_QUERIES, 2);
        assert!(DIRTY_WRITES > COLUMN_CHANGE_LOG_CAPACITY);
        for stride in [1, 2] {
            let (mut world, entities, query) = dirty_world(DIRTY_WRITES * stride);
            // Reuse the actual producer body across samples: retained history
            // wraps repeatedly, while each forward read must remain gap-free.
            for _ in 0..3 {
                forward_dirty_sample(&mut world, &entities, &query, stride);
            }
            for (index, entity) in entities.iter().enumerate() {
                assert_eq!(
                    world.get::<DirtyCell>(*entity).unwrap().value,
                    if index % stride == 0 { 3 } else { 0 }
                );
            }
        }
    }

    #[test]
    fn perf_contract_dirty_single_window_overflow_is_explicit_fallback() {
        assert!(DIRTY_WRITES > COLUMN_CHANGE_LOG_CAPACITY);
        let (mut world, entities, query) = dirty_world(DIRTY_WRITES * 2);
        for stride in [1, 2] {
            dirty_window_sample(&mut world, &entities, &query, stride, 0..DIRTY_WRITES, true);
            assert_eq!(
                world
                    .query_filtered::<&DirtyCell, Changed<DirtyCell>>()
                    .iter(&world)
                    .count(),
                DIRTY_WRITES
            );
            // A gap is not sticky: the same cursor must resume strict forward discovery.
            forward_dirty_sample(&mut world, &entities, &query, stride);
        }
    }

    #[test]
    #[should_panic(expected = "unexpected dirty path")]
    fn perf_contract_dirty_overflow_cannot_masquerade_as_forward() {
        let (mut world, entities, query) = dirty_world(DIRTY_WRITES);
        dirty_window_sample(&mut world, &entities, &query, 1, 0..DIRTY_WRITES, false);
    }

    #[test]
    #[should_panic(expected = "unexpected dirty path")]
    fn perf_contract_dirty_retained_window_cannot_masquerade_as_gap() {
        let (mut world, entities, query) = dirty_world(DIRTY_WRITES);
        dirty_window_sample(
            &mut world,
            &entities,
            &query,
            1,
            0..DIRTY_WINDOW_WRITES,
            true,
        );
    }

    #[cfg(all(feature = "multi_thread", feature = "simd"))]
    fn emit_controlled_contract() {
        println!(
            "PRISM_ECS_PERF_CONTRACT_V1 {{\"protocol\":\"PRISM_ECS_PERF_V2\",\"samples\":{SAMPLES},\"warmups\":{WARMUPS},\"dirty_writes_per_sample\":{DIRTY_WRITES},\"dirty_writes_per_window\":{DIRTY_WINDOW_WRITES},\"dirty_queries_per_sample\":{DIRTY_QUERIES},\"dirty_rows_per_chunk\":{DIRTY_ROWS_PER_CHUNK},\"column_change_log_capacity\":{COLUMN_CHANGE_LOG_CAPACITY},\"dirty_gap_writes_per_window\":{DIRTY_WRITES},\"dirty_gap_queries_per_sample\":1}}"
        );
    }

    #[cfg(all(feature = "multi_thread", feature = "simd"))]
    fn emit_samples(metric: &str, units: usize, mut run: impl FnMut() -> Duration) {
        for _ in 0..WARMUPS {
            black_box(run());
        }
        for sample in 0..SAMPLES {
            let ns = run().as_nanos();
            assert!(ns > 0, "sample is below clock resolution");
            println!(
                "PRISM_ECS_PERF_V2 {{\"metric\":\"{metric}\",\"sample\":{sample},\"ns\":{ns},\"units\":{units}}}"
            );
        }
    }

    #[cfg(all(feature = "multi_thread", feature = "simd"))]
    fn sample_body(metric: &str, units: usize, mut body: impl FnMut()) {
        emit_samples(metric, units, || {
            let start = Instant::now();
            body();
            start.elapsed()
        });
    }

    #[cfg(all(feature = "multi_thread", feature = "simd"))]
    #[test]
    #[ignore = "controlled-machine sample producer; invoke through tools/ecs-perf/gate.py"]
    fn controlled_samples() {
        if cfg!(debug_assertions) {
            panic!("performance samples require --release");
        }
        emit_controlled_contract();
        emit_samples("spawn_batch", ENTITIES, || {
            let mut world = World::new();
            let start = Instant::now();
            let entities = world
                .spawn_batch((0..ENTITIES).map(|i| (Pos(i as f32, 0.0, 0.0), Vel(1.0, 2.0, 3.0))));
            let elapsed = start.elapsed();
            assert_eq!(entities.len(), ENTITIES);
            assert_eq!(world.entity_count(), ENTITIES as u32);
            black_box(&world);
            elapsed // World/Vec destruction is outside the timed interval.
        });

        let (mut world, _) = spawn_world();
        let query = world.query::<(&Vel, &mut Pos)>();
        assert_eq!(query.iter_mut(&mut world).count(), ENTITIES);
        sample_body("serial_propagate", ENTITIES, || {
            for (v, mut p) in query.iter_mut(&mut world) {
                p.0 += v.0;
                p.1 += v.1;
                p.2 += v.2;
            }
        });
        for threads in [1, 4] {
            let pool = prism_tasks::TaskPool::with_threads(threads);
            sample_body(&format!("parallel_propagate_{threads}"), ENTITIES, || {
                query.par_for_each_mut(&mut world, &pool, 1024, |(v, mut p)| {
                    p.0 += v.0;
                    p.1 += v.1;
                    p.2 += v.2;
                });
            });
        }
        let runs = (3 * (SAMPLES + WARMUPS)) as f32;
        for (i, p) in world.query::<&Pos>().iter(&world).enumerate() {
            assert_eq!((p.0, p.1, p.2), (i as f32 + runs, 2.0 * runs, 3.0 * runs));
        }

        let query = world.query::<&Pos>();
        sample_body("query_cache_idle", 1024, || {
            for _ in 0..1024 {
                black_box(query.iter(&world));
            }
        });
        drop(world); // Do not retain the propagation world beside the 512-byte rows.
        let (mut world, entities, query) = dirty_world(ENTITIES);
        let last = world.change_tick();
        world.set_last_change_tick(last);
        world.increment_change_tick();
        emit_samples("dirty_idle", 10_000, || {
            let start = Instant::now();
            let mut reported_chunks = 0;
            for _ in 0..10_000 {
                reported_chunks +=
                    black_box(query.dirty_chunks(&world, last, world.change_tick())).len();
            }
            let elapsed = start.elapsed();
            assert_eq!(reported_chunks, 0);
            elapsed
        });

        for (metric, stride) in [("dirty_clustered", 1), ("dirty_scattered", 100)] {
            emit_samples(metric, DIRTY_QUERIES, || {
                forward_dirty_sample(&mut world, &entities, &query, stride)
            });
        }
        emit_samples("dirty_gap_fallback", 1, || {
            dirty_window_sample(&mut world, &entities, &query, 100, 0..DIRTY_WRITES, true)
        });

        // A separate row-scan window retains the original 10,000 changed rows;
        // it does not claim to consume the bounded journal incrementally.
        world.set_last_change_tick(world.change_tick());
        world.increment_change_tick();
        for k in 0..DIRTY_WRITES {
            world.get_mut::<DirtyCell>(entities[k * 100]).unwrap().value += 1;
        }
        let changed = world.query_filtered::<&DirtyCell, Changed<DirtyCell>>();
        emit_samples("changed_scan", ENTITIES, || {
            let start = Instant::now();
            let changed_count = black_box(changed.iter(&world).count());
            let elapsed = start.elapsed();
            assert_eq!(changed_count, DIRTY_WRITES);
            elapsed
        });

        let mut dst = vec![1.0f32; ENTITIES];
        let src = vec![2.0f32; ENTITIES];
        sample_body("simd_add", ENTITIES, || {
            prism_ecs::query::add_assign(black_box(&mut dst), black_box(&src));
        });
        assert!(dst
            .iter()
            .all(|&v| v == 1.0 + 2.0 * (SAMPLES + WARMUPS) as f32));
    }

    /// Small synthetic wire-protocol check; no ECS workload or hardware claim.
    #[cfg(all(feature = "multi_thread", feature = "simd"))]
    #[test]
    fn perf_contract_controlled_protocol_smoke() {
        emit_controlled_contract();
        for (metric, units) in [
            ("spawn_batch", ENTITIES),
            ("serial_propagate", ENTITIES),
            ("parallel_propagate_1", ENTITIES),
            ("parallel_propagate_4", ENTITIES),
            ("query_cache_idle", 1024),
            ("dirty_idle", 10_000),
            ("dirty_clustered", DIRTY_QUERIES),
            ("dirty_scattered", DIRTY_QUERIES),
            ("dirty_gap_fallback", 1),
            ("changed_scan", ENTITIES),
            ("simd_add", ENTITIES),
        ] {
            let mut calls = 0;
            emit_samples(metric, units, || {
                calls += 1;
                Duration::from_nanos(100)
            });
            assert_eq!(calls, WARMUPS + SAMPLES);
        }
    }

    const PORTABLE_PREFIX: &str = "PRISM_ECS_COMPLEXITY_V1 ";
    const PORTABLE_TIMING_PREFIX: &str = "PRISM_ECS_TIMING_V1 ";
    const PORTABLE_SAMPLES: usize = 9;

    fn emit_complexity(metric: &str, world_units: usize, changed_units: usize, value: usize) {
        println!(
            "{PORTABLE_PREFIX}{{\"metric\":\"{metric}\",\"world_units\":{world_units},\"changed_units\":{changed_units},\"value\":{value}}}"
        );
    }

    fn emit_timing(metric: &str, units: usize, mut run: impl FnMut() -> usize) {
        black_box(run());
        for sample in 0..PORTABLE_SAMPLES {
            let start = Instant::now();
            black_box(run());
            let ns = start.elapsed().as_nanos();
            assert!(ns > 0, "portable sample is below clock resolution");
            println!(
                "{PORTABLE_TIMING_PREFIX}{{\"metric\":\"{metric}\",\"sample\":{sample},\"ns\":{ns},\"units\":{units}}}"
            );
        }
    }

    /// Portable regression input. Complexity records are the blocking contract;
    /// timing records are diagnostic and deliberately reported as percentiles by
    /// the external gate because shared runners do not provide hardware isolation.
    #[test]
    fn portable_complexity_samples() {
        let _lock = MATCH_TEST_LOCK.lock().unwrap();
        let mut world = World::new();
        world.spawn((Pos(1.0, 0.0, 0.0),));
        world.spawn((Pos(2.0, 0.0, 0.0), Vel(1.0, 0.0, 0.0)));
        let query = world.query_filtered::<Entity, CountMatches>();

        MATCH_CALLS.store(0, Ordering::Relaxed);
        assert_eq!(query.iter(&world).count(), 2);
        let cold_calls = MATCH_CALLS.swap(0, Ordering::Relaxed);
        assert_eq!(cold_calls, world.archetypes().len());
        emit_complexity(
            "query_match_cold_calls",
            world.archetypes().len(),
            0,
            cold_calls,
        );

        for _ in 0..8 {
            assert_eq!(query.iter(&world).count(), 2);
        }
        let warm_calls = MATCH_CALLS.swap(0, Ordering::Relaxed);
        assert_eq!(warm_calls, 0);
        emit_complexity(
            "query_match_warm_calls",
            world.archetypes().len(),
            0,
            warm_calls,
        );

        world.spawn((Pos(3.0, 0.0, 0.0), Marker));
        assert_eq!(query.iter(&world).count(), 3);
        let new_archetype_calls = MATCH_CALLS.swap(0, Ordering::Relaxed);
        assert_eq!(new_archetype_calls, 1);
        emit_complexity(
            "query_match_new_archetype_calls",
            world.archetypes().len(),
            1,
            new_archetype_calls,
        );

        for count in [256usize, 16_384] {
            let mut world = World::new();
            let entities = world.spawn_batch((0..count).map(|i| (Pos(i as f32, 0.0, 0.0),)));
            let query = world.query::<&Pos>();
            let last = world.change_tick();
            world.set_last_change_tick(last);
            world.increment_change_tick();
            let now = world.change_tick();
            let mut expected = BTreeSet::new();
            for entity in [entities[0], entities[count / 2], entities[count - 1]] {
                world.get_mut::<Pos>(entity).expect("live entity").0 += 1.0;
                let location = world.entities().location(entity).expect("live location");
                let archetype = world.archetypes().get(location.archetype_id).unwrap();
                let rows_per_chunk = archetype.table().rows_per_chunk();
                expected.insert((
                    location.archetype_id,
                    location.row as usize / rows_per_chunk,
                ));
            }

            let dirty = query.dirty_chunks(&world, last, now);
            let changed_rows = world
                .query_filtered::<&Pos, Changed<Pos>>()
                .iter(&world)
                .count();
            assert_eq!(changed_rows, 3);
            assert_eq!(dirty.len(), expected.len());
            emit_complexity("dirty_chunks_output", count, 3, dirty.len());
            emit_complexity("changed_scan_rows", count, 3, changed_rows);

            emit_timing(&format!("dirty_chunks_{count}"), count, || {
                black_box(query.dirty_chunks(&world, last, now).len())
            });
            let changed = world.query_filtered::<&Pos, Changed<Pos>>();
            emit_timing(&format!("changed_scan_{count}"), count, || {
                black_box(changed.iter(&world).count())
            });
        }

        // Exercise indexed dirty discovery at enough distinct chunks to bypass
        // the small-candidate inline path. The counters make internal work
        // observable; output cardinality alone cannot detect a full chunk scan.
        {
            const WORLD_CHUNKS: usize = 4096;
            const CHANGED_CHUNKS: usize = 1024;
            const WRITES_PER_CHUNK: usize = 4;
            const ROW_BYTES: usize = 512;
            const ROWS_PER_CHUNK: usize = 32;
            let mut world = World::new();
            let archetype_id = world.prewarm::<DirtyCell>(WORLD_CHUNKS * ROWS_PER_CHUNK);
            let rows_per_chunk = world
                .archetypes()
                .get(archetype_id)
                .unwrap()
                .table()
                .rows_per_chunk();
            assert_eq!(rows_per_chunk, ROWS_PER_CHUNK);
            let entities =
                world.spawn_batch((0..WORLD_CHUNKS * rows_per_chunk).map(|index| DirtyCell {
                    value: index as u64,
                    _padding: [0; ROW_BYTES - size_of::<u64>()],
                }));
            let chunk_count = WORLD_CHUNKS;
            assert_eq!(
                world
                    .archetypes()
                    .get(archetype_id)
                    .unwrap()
                    .table()
                    .chunk_count(),
                chunk_count,
            );
            let query = world.query::<&DirtyCell>();
            let last_run = world.change_tick();
            let baseline = query.dirty_chunks_with_stats(&world, last_run, last_run);
            assert!(baseline.chunks.is_empty());

            world.increment_change_tick();
            let this_run = world.change_tick();
            for chunk in 0..CHANGED_CHUNKS {
                for _ in 0..WRITES_PER_CHUNK {
                    world
                        .get_mut::<DirtyCell>(entities[chunk * rows_per_chunk])
                        .unwrap()
                        .value += 1;
                }
            }
            let report = query.dirty_chunks_with_stats(&world, last_run, this_run);
            let writes = CHANGED_CHUNKS * WRITES_PER_CHUNK;
            assert_eq!(report.stats.journal_records_scanned, writes);
            assert_eq!(report.stats.candidate_chunk_marks, writes);
            assert_eq!(report.stats.candidate_archetype_checks, 1);
            assert_eq!(report.stats.merged_candidates, CHANGED_CHUNKS);
            assert_eq!(report.stats.reported_chunks, CHANGED_CHUNKS);
            assert_eq!(report.stats.matched_archetypes, 1);
            assert_eq!(report.stats.summaries_tested, 0);
            emit_complexity(
                "dirty_journal_records_scanned",
                WORLD_CHUNKS * ROWS_PER_CHUNK,
                writes,
                report.stats.journal_records_scanned,
            );
            emit_complexity(
                "dirty_candidate_chunk_marks",
                WORLD_CHUNKS * ROWS_PER_CHUNK,
                writes,
                report.stats.candidate_chunk_marks,
            );
            emit_complexity(
                "dirty_candidate_archetype_checks",
                WORLD_CHUNKS * ROWS_PER_CHUNK,
                writes,
                report.stats.candidate_archetype_checks,
            );
            emit_complexity(
                "dirty_index_summaries_tested",
                WORLD_CHUNKS * ROWS_PER_CHUNK,
                writes,
                report.stats.summaries_tested,
            );
            emit_complexity(
                "dirty_merged_candidates",
                WORLD_CHUNKS * ROWS_PER_CHUNK,
                writes,
                report.stats.merged_candidates,
            );
            emit_complexity(
                "dirty_reported_chunks",
                WORLD_CHUNKS * ROWS_PER_CHUNK,
                writes,
                report.stats.reported_chunks,
            );
        }
    }
}
