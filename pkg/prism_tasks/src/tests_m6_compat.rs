//! M6 tests: the `bevy_tasks`-compatible prelude (behind `compat-bevy`).
//!
//! Anti-vacuous contract: the compat scope runs synchronous closures on the
//! real pool and returns their results in spawn order; the builder honours the
//! requested thread count; the global pool accessors initialize once and then
//! hand back the same instance; and `spawn` drives a future to completion.
//!
//! The `single` feature forces the synchronous fallback (`worker_count == 0`),
//! so thread-count assertions below branch on `cfg!(feature = "single")` to
//! stay correct in every feature combo while still verifying the mapping.

use crate::TaskPool;
use crate::bevy_prelude::{CompatTaskPool, ComputeTaskPool, TaskPoolBuilder};
use alloc::vec::Vec;

/// Worker count a request of `n` threads resolves to, accounting for the
/// synchronous `single` fallback.
fn expected_threads(requested: usize) -> usize {
    if cfg!(feature = "single") {
        0
    } else {
        requested
    }
}

#[test]
fn compat_scope_returns_results_in_spawn_order() {
    let pool = CompatTaskPool::new();
    let results = pool.scope(|s| {
        for i in 0..256usize {
            // Reverse the per-task cost so completion order differs from spawn
            // order; the result vector must still be in spawn order.
            s.spawn(move || i * i);
        }
    });
    let expect: Vec<usize> = (0..256usize).map(|i| i * i).collect();
    assert_eq!(results, expect);
}

#[test]
fn compat_scope_empty_returns_empty() {
    let pool = CompatTaskPool::new();
    let results: Vec<usize> = pool.scope(|_s| {});
    assert!(results.is_empty());
}

#[test]
fn builder_honours_thread_count() {
    let pool = TaskPoolBuilder::new().num_threads(3).build();
    let expect = expected_threads(3);
    assert_eq!(pool.thread_num(), expect);
    // Deref exposes the native pool API.
    assert_eq!(pool.worker_count(), expect);
}

#[test]
fn builder_zero_threads_is_single_threaded() {
    let pool = TaskPoolBuilder::new().num_threads(0).build();
    assert!(pool.is_single_threaded());
    let results = pool.scope(|s| {
        for i in 0..10usize {
            s.spawn(move || i + 1);
        }
    });
    assert_eq!(results, (1..=10usize).collect::<Vec<_>>());
}

#[test]
fn global_compute_pool_initializes_once() {
    assert!(ComputeTaskPool::try_get().is_none());
    let first = ComputeTaskPool::get_or_init(|| TaskPool::with_threads(2));
    assert_eq!(first.thread_num(), expected_threads(2));
    // A second init with a different request must be ignored (same instance).
    let second = ComputeTaskPool::get_or_init(|| TaskPool::with_threads(8));
    assert_eq!(second.thread_num(), expected_threads(2));
    assert!(core::ptr::eq(first, second), "same global instance");
    assert!(
        core::ptr::eq(first, ComputeTaskPool::get()),
        "get returns the init'd pool"
    );
}

#[test]
fn compat_spawn_runs_a_future() {
    let pool = CompatTaskPool::new();
    let task = pool.spawn(async { 7 * 6 });
    assert_eq!(pool.block_on(task), 42);
}
