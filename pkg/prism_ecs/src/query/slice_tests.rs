//! Behavioural tests for the chunk-aligned, typed-column-slice parallel path
//! ([`QueryState::par_chunks`](crate::query::QueryState::par_chunks) /
//! [`par_chunks_mut`](crate::query::QueryState::par_chunks_mut), design §8.3's
//! headline `jobs.par_chunks(&mut q, |chunk| ...)`).
//!
//! Unlike the row path ([`par_tests`](crate::query::par_tests)), each sub-job
//! here receives one aligned, contiguous typed slice (`&[T]` / `&mut [T]`)
//! covering a whole run of rows. These tests assert that:
//! * a `&mut [T]` pass mutates every matched row exactly once, independent of
//!   `batch_size` and pool thread count (chunk-aligned disjoint partition);
//! * a read-only `&[T]` pass reduces to the same value as the serial iterator;
//! * the slice path spans multiple archetypes and multiple chunks per archetype;
//! * a `&mut T` term conservatively stamps the whole handed-out range changed so
//!   `Changed<T>` fires for every visited row afterwards (design §10);
//! * the [`Entity`] term yields each entity once;
//! * tuple terms (`(&mut A, &B)`) slice each column over the same range;
//! * [`With`] / [`Without`] archetype filters are honoured;
//! * an empty match set is a no-op and `batch_size` 0 is clamped.
//!
//! Enabled only under `cfg(all(test, feature = "multi_thread"))`.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicI64, AtomicUsize, Ordering};

use prism_tasks::TaskPool;

use crate::component::Component;
use crate::entity::Entity;
use crate::query::{Changed, With, Without};
use crate::world::World;

#[derive(Debug, PartialEq, Clone, Copy)]
struct Position(i64);
impl Component for Position {}

#[derive(Debug, PartialEq, Clone, Copy)]
struct Velocity(i64);
impl Component for Velocity {}

#[derive(Debug, PartialEq, Clone, Copy)]
struct Tag;
impl Component for Tag {}

/// Advance the world as if a system had just finished a run: the next window
/// starts where this one ended, and the frame tick advances by one. Mirrors the
/// harness in [`change_detection_tests`](crate::query::change_detection_tests).
fn advance_frame(w: &mut World) {
    w.set_last_change_tick(w.change_tick());
    w.increment_change_tick();
}

/// `Position` is 8 bytes, so an archetype holding only it packs
/// `TARGET_CHUNK_BYTES / 8 == 2048` rows per chunk. Spawning well past that
/// guarantees several whole chunks plus a trailing partial chunk, so the
/// chunk-aligned partition genuinely produces multiple batches.
const MANY: i64 = 5000;

#[test]
fn par_chunks_mut_doubles_every_row_across_many_chunks() {
    let mut w = World::new();
    let mut ents = Vec::new();
    for i in 0..MANY {
        ents.push(w.spawn(Position(i)));
    }

    let state = w.query::<&mut Position>();
    let pool = TaskPool::with_threads(4);
    // A batch_size smaller than rows_per_chunk still rounds up to one whole
    // chunk per batch, so this fans out across every chunk of the archetype.
    state.par_chunks_mut(&mut w, &pool, 1, |slice: &mut [Position]| {
        for p in slice {
            p.0 *= 2;
        }
    });

    for (i, &e) in ents.iter().enumerate() {
        assert_eq!(w.get::<Position>(e), Some(&Position(i as i64 * 2)));
    }
}

#[test]
fn par_chunks_read_only_sum_matches_serial() {
    let mut w = World::new();
    let mut expected = 0i64;
    for i in 0..MANY {
        w.spawn(Position(i));
        expected += i;
    }

    // Serial ground truth over the row iterator.
    let state = w.query::<&Position>();
    let serial: i64 = state.iter(&w).map(|p| p.0).sum();
    assert_eq!(serial, expected);

    // Parallel read-only reduction: each sub-job sums a whole typed slice.
    let acc = AtomicI64::new(0);
    let pool = TaskPool::with_threads(4);
    state.par_chunks(&w, &pool, 4, |slice: &[Position]| {
        let partial: i64 = slice.iter().map(|p| p.0).sum();
        acc.fetch_add(partial, Ordering::Relaxed);
    });
    assert_eq!(acc.load(Ordering::Relaxed), expected);
}

#[test]
fn par_chunks_read_only_visits_every_row_once() {
    let mut w = World::new();
    for i in 0..MANY {
        w.spawn(Position(i));
    }

    let state = w.query::<&Position>();
    let seen = AtomicUsize::new(0);
    let pool = TaskPool::with_threads(3);
    state.par_chunks(&w, &pool, 2, |slice: &[Position]| {
        seen.fetch_add(slice.len(), Ordering::Relaxed);
    });
    assert_eq!(seen.load(Ordering::Relaxed), MANY as usize);
}

#[test]
fn par_chunks_mut_spans_multiple_archetypes() {
    let mut w = World::new();
    let mut only_pos = Vec::new();
    let mut pos_vel = Vec::new();
    for i in 0..3000i64 {
        only_pos.push(w.spawn(Position(i)));
    }
    for i in 0..3000i64 {
        pos_vel.push(w.spawn((Position(10_000 + i), Velocity(i))));
    }

    let state = w.query::<&mut Position>();
    let pool = TaskPool::with_threads(4);
    state.par_chunks_mut(&mut w, &pool, 1, |slice: &mut [Position]| {
        for p in slice {
            p.0 += 1;
        }
    });

    for (i, &e) in only_pos.iter().enumerate() {
        assert_eq!(w.get::<Position>(e), Some(&Position(i as i64 + 1)));
    }
    for (i, &e) in pos_vel.iter().enumerate() {
        assert_eq!(w.get::<Position>(e), Some(&Position(10_000 + i as i64 + 1)));
    }
}

#[test]
fn par_chunks_mut_marks_whole_range_changed() {
    let mut w = World::new();
    let n = 3000usize;
    for i in 0..n as i64 {
        w.spawn(Position(i));
    }

    // Move past the spawn frame so "added" no longer dominates the window.
    advance_frame(&mut w);

    let state = w.query::<&mut Position>();
    let pool = TaskPool::with_threads(4);
    state.par_chunks_mut(&mut w, &pool, 1, |slice: &mut [Position]| {
        for p in slice {
            p.0 += 100;
        }
    });

    // Every row the mut slice handed out is stamped `changed = this_run`, so the
    // whole set fires `Changed<Position>` in the same window (design §10).
    let changed = w.query_filtered::<Entity, Changed<Position>>();
    assert_eq!(
        changed.iter(&w).count(),
        n,
        "a mut slice pass marks its whole handed-out range changed"
    );

    // After the frame advances with no further write, nothing is changed.
    advance_frame(&mut w);
    let changed = w.query_filtered::<Entity, Changed<Position>>();
    assert_eq!(
        changed.iter(&w).count(),
        0,
        "rows written in a previous frame are no longer Changed"
    );
}

#[test]
fn par_chunks_entity_slice_visits_each_entity_once() {
    let mut w = World::new();
    let mut ents = Vec::new();
    for i in 0..MANY {
        ents.push(w.spawn(Position(i)));
    }

    let state = w.query::<Entity>();
    let count = AtomicUsize::new(0);
    let pool = TaskPool::with_threads(4);
    state.par_chunks(&w, &pool, 3, |slice: &[Entity]| {
        count.fetch_add(slice.len(), Ordering::Relaxed);
    });
    assert_eq!(count.load(Ordering::Relaxed), ents.len());
}

#[test]
fn par_chunks_mut_tuple_slices_each_column() {
    let mut w = World::new();
    let n = 3000i64;
    let mut ents = Vec::new();
    for i in 0..n {
        ents.push(w.spawn((Position(i), Velocity(i * 2))));
    }

    // `(&mut Position, &Velocity)`: write Position from the read-only Velocity
    // slice, exercising a mixed mutable/shared tuple over the same range.
    let state = w.query::<(&mut Position, &Velocity)>();
    let pool = TaskPool::with_threads(4);
    state.par_chunks_mut(
        &mut w,
        &pool,
        1,
        |(positions, velocities): (&mut [Position], &[Velocity])| {
            assert_eq!(positions.len(), velocities.len());
            for (p, v) in positions.iter_mut().zip(velocities.iter()) {
                p.0 += v.0;
            }
        },
    );

    for (i, &e) in ents.iter().enumerate() {
        let i = i as i64;
        assert_eq!(w.get::<Position>(e), Some(&Position(i + i * 2)));
    }
}

#[test]
fn par_chunks_honours_with_filter() {
    let mut w = World::new();
    for i in 0..2000i64 {
        w.spawn(Position(i)); // no Velocity -> excluded
    }
    let mut expected = 0i64;
    for i in 0..2000i64 {
        w.spawn((Position(10_000 + i), Velocity(i)));
        expected += 10_000 + i;
    }

    let state = w.query_filtered::<&Position, With<Velocity>>();
    let acc = AtomicI64::new(0);
    let pool = TaskPool::with_threads(4);
    state.par_chunks(&w, &pool, 2, |slice: &[Position]| {
        let partial: i64 = slice.iter().map(|p| p.0).sum();
        acc.fetch_add(partial, Ordering::Relaxed);
    });
    assert_eq!(acc.load(Ordering::Relaxed), expected);
}

#[test]
fn par_chunks_honours_without_filter() {
    let mut w = World::new();
    let mut expected = 0i64;
    for i in 0..2000i64 {
        w.spawn(Position(i)); // kept
        expected += i;
    }
    for i in 0..2000i64 {
        w.spawn((Position(10_000 + i), Velocity(i))); // excluded by Without
    }

    let state = w.query_filtered::<&Position, Without<Velocity>>();
    let acc = AtomicI64::new(0);
    let pool = TaskPool::with_threads(4);
    state.par_chunks(&w, &pool, 2, |slice: &[Position]| {
        let partial: i64 = slice.iter().map(|p| p.0).sum();
        acc.fetch_add(partial, Ordering::Relaxed);
    });
    assert_eq!(acc.load(Ordering::Relaxed), expected);
}

#[test]
fn par_chunks_result_independent_of_thread_count_and_batch_size() {
    let mut w = World::new();
    let mut expected = 0i64;
    for i in 0..4096i64 {
        w.spawn((Position(i), Tag));
        expected += i;
    }

    let state = w.query_filtered::<&Position, With<Tag>>();
    for (threads, batch) in [(1usize, 1usize), (1, 64), (3, 7), (8, 1), (8, 100_000)] {
        let pool = TaskPool::with_threads(threads);
        let acc = AtomicI64::new(0);
        state.par_chunks(&w, &pool, batch, |slice: &[Position]| {
            let partial: i64 = slice.iter().map(|p| p.0).sum();
            acc.fetch_add(partial, Ordering::Relaxed);
        });
        assert_eq!(
            acc.load(Ordering::Relaxed),
            expected,
            "threads={threads} batch={batch}"
        );
    }
}

#[test]
fn par_chunks_empty_query_is_noop() {
    let mut w = World::new();
    w.spawn(Velocity(1)); // no Position

    let state = w.query::<&Position>();
    let acc = AtomicI64::new(0);
    let pool = TaskPool::new();
    // batch_size 0 must be clamped to >= 1 internally and visit nothing.
    state.par_chunks(&w, &pool, 0, |slice: &[Position]| {
        let partial: i64 = slice.iter().map(|p| p.0).sum();
        acc.fetch_add(partial, Ordering::Relaxed);
    });
    assert_eq!(acc.load(Ordering::Relaxed), 0);
}

#[test]
fn par_chunks_mut_matches_serial_mutation() {
    // Run the same kernel serially and via the slice path on two identical
    // worlds; assert the final column states are bit-identical.
    let build = || {
        let mut w = World::new();
        let mut es = Vec::new();
        for i in 0..4321i64 {
            es.push(w.spawn(Position(i)));
        }
        (w, es)
    };

    let (mut serial_w, serial_es) = build();
    {
        let state = serial_w.query::<&mut Position>();
        for mut p in state.iter_mut(&mut serial_w) {
            p.0 = p.0 * 3 + 1;
        }
    }

    let (mut par_w, par_es) = build();
    {
        let state = par_w.query::<&mut Position>();
        let pool = TaskPool::with_threads(4);
        state.par_chunks_mut(&mut par_w, &pool, 1, |slice: &mut [Position]| {
            for p in slice {
                p.0 = p.0 * 3 + 1;
            }
        });
    }

    for (&se, &pe) in serial_es.iter().zip(par_es.iter()) {
        assert_eq!(serial_w.get::<Position>(se), par_w.get::<Position>(pe));
    }
}
