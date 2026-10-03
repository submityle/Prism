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

// ---------------------------------------------------------------------------
// M1: structured parallelism
// ---------------------------------------------------------------------------

#[test]
fn scope_mutates_borrowed_data_and_joins_all() {
    let pool = TaskPool::with_threads(4);
    let mut data = vec![0usize; 1000];
    pool.scope(|s| {
        for (i, slot) in data.iter_mut().enumerate() {
            s.spawn(move || *slot = i * 3);
        }
    });
    for (i, v) in data.iter().enumerate() {
        assert_eq!(*v, i * 3);
    }
}

#[test]
fn scope_join_borrowed() {
    let pool = TaskPool::with_threads(3);
    let left = vec![1u64; 500];
    let right = vec![2u64; 500];
    let (a, b) = pool.scope(|s| {
        s.join(
            || left.iter().sum::<u64>(),
            || right.iter().sum::<u64>(),
        )
    });
    assert_eq!((a, b), (500, 1000));
}

#[test]
fn parallel_for_doubles_large_vec_matches_serial() {
    for &threads in &[0usize, 1, 2, 4, 8] {
        let pool = TaskPool::with_threads(threads);
        let n = 200_000usize;
        let mut data: Vec<u64> = (0..n as u64).collect();
        let expected: Vec<u64> = data.iter().map(|x| x * 2).collect();
        pool.par_for_each_mut(&mut data, |x| *x *= 2);
        assert_eq!(data, expected, "threads={threads}");
    }
}

#[test]
fn parallel_for_chunk_variant() {
    let pool = TaskPool::with_threads(4);
    let n = 123_457usize;
    let mut data: Vec<u64> = (0..n as u64).collect();
    let expected: Vec<u64> = data.iter().map(|x| x + 7).collect();
    pool.parallel_for(&mut data, |chunk| {
        for x in chunk {
            *x += 7;
        }
    });
    assert_eq!(data, expected);
}

#[test]
fn par_for_each_read_only() {
    let pool = TaskPool::with_threads(4);
    let data: Vec<u64> = (1..=10_000).collect();
    let sum = Arc::new(AtomicUsize::new(0));
    let sum2 = Arc::clone(&sum);
    pool.par_for_each(&data, move |x| {
        sum2.fetch_add(*x as usize, Ordering::Relaxed);
    });
    assert_eq!(sum.load(Ordering::Relaxed), (1..=10_000usize).sum());
}

#[test]
fn reduce_sum_matches_serial_for_multiple_worker_counts() {
    for &threads in &[0usize, 1, 2, 4, 8] {
        let pool = TaskPool::with_threads(threads);
        let n: u64 = 100_000;
        let data: Vec<u64> = (1..=n).collect();
        let sum = pool.reduce(&data, || 0u64, |&x| x, |a, b| a + b);
        assert_eq!(sum, n * (n + 1) / 2, "threads={threads}");
    }
}

#[test]
fn reduce_empty_returns_identity() {
    let pool = TaskPool::with_threads(4);
    let data: Vec<u64> = Vec::new();
    let sum = pool.reduce(&data, || 42u64, |&x| x, |a, b| a + b);
    assert_eq!(sum, 42);
}

#[test]
fn prefix_sum_matches_serial_scan() {
    for &threads in &[0usize, 1, 2, 4, 8] {
        let pool = TaskPool::with_threads(threads);
        let n = 100_000usize;
        let mut data: Vec<u64> = (0..n as u64).map(|x| x % 7 + 1).collect();
        let mut expected = data.clone();
        let mut acc = 0u64;
        for v in expected.iter_mut() {
            acc += *v;
            *v = acc;
        }
        pool.prefix_sum(&mut data);
        assert_eq!(data, expected, "threads={threads}");
    }
}

#[test]
fn prefix_sum_small_inputs() {
    let pool = TaskPool::with_threads(4);
    let mut empty: Vec<u64> = Vec::new();
    pool.prefix_sum(&mut empty);
    assert!(empty.is_empty());

    let mut one = vec![9u64];
    pool.prefix_sum(&mut one);
    assert_eq!(one, vec![9]);

    let mut three = vec![1u64, 2, 3];
    pool.prefix_sum(&mut three);
    assert_eq!(three, vec![1, 3, 6]);
}

#[test]
fn nested_scope_and_parallel_for_do_not_deadlock() {
    // Only two workers but deep nesting; help-while-waiting must drain it.
    let pool = TaskPool::with_threads(2);
    let mut matrix = vec![vec![0u64; 500]; 200];
    pool.scope(|s| {
        for (r, row) in matrix.iter_mut().enumerate() {
            let pool_ref = &pool;
            s.spawn(move || {
                pool_ref.par_for_each_mut(row, |x| *x += 1);
                for (c, v) in row.iter_mut().enumerate() {
                    *v += (r + c) as u64;
                }
            });
        }
    });
    for (r, row) in matrix.iter().enumerate() {
        for (c, v) in row.iter().enumerate() {
            assert_eq!(*v, 1 + (r + c) as u64);
        }
    }
}

#[test]
fn stress_many_tasks() {
    let pool = TaskPool::with_threads(4);
    for _ in 0..50 {
        let n = 50_000usize;
        let mut data: Vec<u64> = (0..n as u64).collect();
        pool.par_for_each_mut(&mut data, |x| *x = x.wrapping_mul(2654435761) ^ (*x >> 3));
        let reduced = pool.reduce(&data, || 0u64, |&x| x, u64::wrapping_add);
        let serial: u64 = (0..n as u64)
            .map(|x| x.wrapping_mul(2654435761) ^ (x >> 3))
            .fold(0u64, u64::wrapping_add);
        assert_eq!(reduced, serial);
    }
}

#[test]
fn scope_propagates_child_panic() {
    let pool = TaskPool::with_threads(4);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pool.scope(|s| {
            s.spawn(|| panic!("boom"));
            s.spawn(|| { /* ok */ });
        });
    }));
    assert!(result.is_err());
}
