use crate::{Counter, TaskPool};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn spawn_and_wait_runs_all_jobs() {
    let pool = TaskPool::with_threads(4);
    let counter = Counter::new();
    let sum = Arc::new(AtomicUsize::new(0));
    for i in 1..=100 {
        let sum = Arc::clone(&sum);
        pool.spawn(&counter, move || {
            sum.fetch_add(i, Ordering::Relaxed);
        });
    }
    pool.wait(&counter);
    assert!(counter.is_complete());
    assert_eq!(sum.load(Ordering::Relaxed), (1..=100).sum());
}

#[test]
fn nested_fork_join_does_not_deadlock() {
    // Recursion depth far exceeds the worker count; help-on-wait must drain it.
    let pool = TaskPool::with_threads(2);

    fn fib(pool: &TaskPool, n: u64) -> u64 {
        if n < 2 {
            return n;
        }
        let p = pool.clone();
        let (a, b) = pool.join(move || fib(&p, n - 1), {
            let p2 = pool.clone();
            move || fib(&p2, n - 2)
        });
        a + b
    }

    assert_eq!(fib(&pool, 12), 144);
}

#[test]
fn join_returns_both_results() {
    let pool = TaskPool::with_threads(3);
    let (a, b) = pool.join(|| 2 + 2, || 3 * 3);
    assert_eq!((a, b), (4, 9));
}

#[test]
fn single_threaded_fallback_runs_inline() {
    let pool = TaskPool::with_threads(0);
    assert!(pool.is_single_threaded());
    assert_eq!(pool.worker_count(), 0);
    let counter = Counter::new();
    let hit = Arc::new(AtomicUsize::new(0));
    let h = Arc::clone(&hit);
    pool.spawn(&counter, move || {
        h.fetch_add(1, Ordering::Relaxed);
    });
    // In the fallback the job already ran before spawn returned.
    assert_eq!(hit.load(Ordering::Relaxed), 1);
    assert!(counter.is_complete());
    pool.wait(&counter);
}
