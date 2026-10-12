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
//! stay informative on any machine instead of flaky. The numbers are recorded
//! in `docs/prism_ecs_design_zh.md` as measured evidence.

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

/// Table-backed velocity.
#[derive(Debug, Clone, Copy)]
struct Vel(f32, f32, f32);
impl Component for Vel {}

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
