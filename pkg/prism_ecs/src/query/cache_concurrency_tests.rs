//! Concurrency regression test for the query match cache (design §7).
//!
//! [`QueryState`](crate::query::QueryState) is `Sync` (it satisfies the
//! `SystemParam::State: Send + Sync` bound), so a shared `&QueryState` may be
//! iterated from several threads at once. The first iteration lazily *fills* the
//! archetype-match cache, so if that fill were an unsynchronized interior
//! mutation two threads would race on the cache `Vec`. This test hammers that
//! path from many threads at a common barrier; under the spin-locked cache it is
//! race-free and every thread observes the same totals.

use alloc::sync::Arc;
use alloc::vec::Vec;
use std::sync::Barrier;
use std::thread;

use crate::component::Component;
use crate::query::QueryState;
use crate::world::World;

/// Table-backed position.
#[derive(Debug, PartialEq, Clone, Copy)]
struct Pos(i32);
impl Component for Pos {}

/// Table-backed velocity (a second archetype, so the cache has real work).
#[derive(Debug, PartialEq, Clone, Copy)]
struct Vel(i32);
impl Component for Vel {}

/// The state must stay `Send + Sync` (the `SystemParam::State` bound) now that
/// the match cache is spin-locked rather than wrapped in an `UnsafeCell`.
#[test]
fn query_state_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<QueryState<&'static Pos>>();
    assert_send_sync::<QueryState<(&'static Pos, &'static mut Vel)>>();
}

#[test]
fn concurrent_iteration_fills_the_match_cache_without_a_race() {
    const THREADS: usize = 8;
    const ROUNDS: usize = 200;

    let mut world = World::new();
    let mut expected = 0i32;
    for i in 0..64 {
        if i % 2 == 0 {
            world.spawn(Pos(i));
        } else {
            world.spawn((Pos(i), Vel(i * 10)));
        }
        expected += i;
    }

    // A fresh state: its cache is empty, so every thread races to populate it.
    let state = Arc::new(world.query::<&Pos>());
    let barrier = Arc::new(Barrier::new(THREADS + 1));

    // `thread::scope` lets the workers borrow `world` (which `spawn` would
    // reject as non-`'static`); the scope joins them before returning.
    thread::scope(|scope| {
        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let state = Arc::clone(&state);
                let barrier = Arc::clone(&barrier);
                let world = &world;
                scope.spawn(move || {
                    barrier.wait();
                    let mut last = 0i32;
                    for _ in 0..ROUNDS {
                        last = state.iter(world).map(|p| p.0).sum();
                    }
                    last
                })
            })
            .collect();

        barrier.wait();
        for handle in handles {
            assert_eq!(handle.join().expect("worker did not panic"), expected);
        }
    });
}
