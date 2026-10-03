//! Behavioural tests for chunk-parallel query iteration
//! ([`QueryState::par_for_each`](crate::query::QueryState::par_for_each) /
//! [`par_for_each_mut`](crate::query::QueryState::par_for_each_mut), design §7
//! par_iter / §8.3). These assert that the parallel pass visits exactly the
//! same rows as the serial [`QueryIter`](crate::query::QueryIter), that `&mut`
//! terms mutate disjoint storage without races, that archetype filters are
//! honoured, and that the result is independent of pool thread count and
//! `batch_size` (determinism of the *set* of visited rows).
//!
//! Enabled only under `cfg(all(test, feature = "multi_thread"))`.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicI64, Ordering};

use prism_tasks::TaskPool;

use crate::component::Component;
use crate::query::{With, Without};
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

#[test]
fn par_for_each_mut_doubles_every_row_across_many_batches() {
    let mut w = World::new();
    let mut ents = Vec::new();
    for i in 0..1000i64 {
        ents.push(w.spawn(Position(i)));
    }

    let state = w.query::<&mut Position>();
    let pool = TaskPool::new();
    // Small batch size forces many (ceil(1000/7) = 143) disjoint batches so the
    // work genuinely fans out across the pool.
    state.par_for_each_mut(&mut w, &pool, 7, |mut p| {
        p.0 *= 2;
    });

    for (i, &e) in ents.iter().enumerate() {
        assert_eq!(w.get::<Position>(e), Some(&Position(i as i64 * 2)));
    }
}

#[test]
fn par_for_each_sum_matches_serial_iter() {
    let mut w = World::new();
    let mut expected = 0i64;
    for i in 0..777i64 {
        w.spawn(Position(i));
        expected += i;
    }

    // Serial ground truth.
    let state = w.query::<&Position>();
    let serial: i64 = state.iter(&w).map(|p| p.0).sum();
    assert_eq!(serial, expected);

    // Parallel read-only reduction over an atomic accumulator.
    let acc = AtomicI64::new(0);
    let pool = TaskPool::new();
    state.par_for_each(&w, &pool, 13, |p| {
        acc.fetch_add(p.0, Ordering::Relaxed);
    });
    assert_eq!(acc.load(Ordering::Relaxed), expected);
}

#[test]
fn par_for_each_mut_spans_multiple_archetypes() {
    let mut w = World::new();
    let mut only_pos = Vec::new();
    let mut pos_vel = Vec::new();
    for i in 0..300i64 {
        only_pos.push(w.spawn(Position(i)));
    }
    for i in 0..300i64 {
        pos_vel.push(w.spawn((Position(1000 + i), Velocity(i))));
    }

    let state = w.query::<&mut Position>();
    let pool = TaskPool::with_threads(4);
    state.par_for_each_mut(&mut w, &pool, 5, |mut p| {
        p.0 += 1;
    });

    for (i, &e) in only_pos.iter().enumerate() {
        assert_eq!(w.get::<Position>(e), Some(&Position(i as i64 + 1)));
    }
    for (i, &e) in pos_vel.iter().enumerate() {
        assert_eq!(w.get::<Position>(e), Some(&Position(1000 + i as i64 + 1)));
    }
}

#[test]
fn par_for_each_honours_with_filter() {
    let mut w = World::new();
    for i in 0..200i64 {
        w.spawn(Position(i)); // no Velocity -> excluded
    }
    let mut expected = 0i64;
    for i in 0..200i64 {
        w.spawn((Position(1000 + i), Velocity(i)));
        expected += 1000 + i;
    }

    let state = w.query_filtered::<&Position, With<Velocity>>();
    let acc = AtomicI64::new(0);
    let pool = TaskPool::new();
    state.par_for_each(&w, &pool, 9, |p| {
        acc.fetch_add(p.0, Ordering::Relaxed);
    });
    assert_eq!(acc.load(Ordering::Relaxed), expected);
}

#[test]
fn par_for_each_honours_without_filter() {
    let mut w = World::new();
    let mut expected = 0i64;
    for i in 0..200i64 {
        w.spawn(Position(i)); // kept
        expected += i;
    }
    for i in 0..200i64 {
        w.spawn((Position(1000 + i), Velocity(i))); // excluded by Without
    }

    let state = w.query_filtered::<&Position, Without<Velocity>>();
    let acc = AtomicI64::new(0);
    let pool = TaskPool::new();
    state.par_for_each(&w, &pool, 11, |p| {
        acc.fetch_add(p.0, Ordering::Relaxed);
    });
    assert_eq!(acc.load(Ordering::Relaxed), expected);
}

#[test]
fn par_result_independent_of_thread_count_and_batch_size() {
    let mut w = World::new();
    let mut expected = 0i64;
    for i in 0..512i64 {
        w.spawn((Position(i), Tag));
        expected += i;
    }

    let state = w.query_filtered::<&Position, With<Tag>>();

    for (threads, batch) in [(1usize, 1usize), (1, 64), (3, 7), (8, 1), (8, 1000)] {
        let pool = TaskPool::with_threads(threads);
        let acc = AtomicI64::new(0);
        state.par_for_each(&w, &pool, batch, |p| {
            acc.fetch_add(p.0, Ordering::Relaxed);
        });
        assert_eq!(
            acc.load(Ordering::Relaxed),
            expected,
            "threads={threads} batch={batch}"
        );
    }
}

#[test]
fn par_for_each_empty_query_is_noop() {
    let mut w = World::new();
    w.spawn(Velocity(1)); // no Position

    let state = w.query::<&Position>();
    let acc = AtomicI64::new(0);
    let pool = TaskPool::new();
    // batch_size 0 must be clamped to >= 1 internally and simply visit nothing.
    state.par_for_each(&w, &pool, 0, |p| {
        acc.fetch_add(p.0, Ordering::Relaxed);
    });
    assert_eq!(acc.load(Ordering::Relaxed), 0);
}

#[test]
fn par_for_each_mut_then_read_back_matches_serial_mutation() {
    // Run the same mutation serially and in parallel on two identical worlds and
    // assert the final column states are bit-identical.
    let build = || {
        let mut w = World::new();
        let mut es = Vec::new();
        for i in 0..431i64 {
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
        state.par_for_each_mut(&mut par_w, &pool, 6, |mut p| {
            p.0 = p.0 * 3 + 1;
        });
    }

    for (&se, &pe) in serial_es.iter().zip(par_es.iter()) {
        assert_eq!(serial_w.get::<Position>(se), par_w.get::<Position>(pe));
    }
}
