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

// ---------------------------------------------------------------------------
// M2: fibers (feature `fibers`)
// ---------------------------------------------------------------------------

#[cfg(feature = "fibers")]
mod fiber_tests {
    use super::*;
    use crate::fiber::{run_fiber_switch, spawn_fiber};
    use crate::job::Job;
    use crate::scheduler::Shared;
    use std::time::{Duration, Instant};

    /// A fiber runs its job to completion on its own stack and switches a value
    /// back out: driving `spawn_fiber` + `run_fiber_switch` directly on a bare
    /// `Shared` must round-trip the write the job performs.
    #[test]
    fn fiber_switch_round_trips_value() {
        let shared = Shared::new(1);
        let out = Arc::new(AtomicUsize::new(0));
        let o = Arc::clone(&out);
        let job: Job = Box::new(move || {
            o.store(42, Ordering::SeqCst);
        });
        let fiber = spawn_fiber(&shared, job);
        run_fiber_switch(&shared, fiber);
        assert_eq!(out.load(Ordering::SeqCst), 42);
    }

    /// Several independent fibers driven one after another on the same `Shared`
    /// each run to completion and reuse pooled stacks without corruption.
    #[test]
    fn many_sequential_fibers_reuse_stacks() {
        let shared = Shared::new(1);
        let sum = Arc::new(AtomicUsize::new(0));
        for i in 0..1000usize {
            let s = Arc::clone(&sum);
            let job: Job = Box::new(move || {
                // Touch a chunk of stack to exercise real stack usage.
                let buf = [i as u8; 1024];
                s.fetch_add(buf[i % 1024] as usize, Ordering::Relaxed);
            });
            let fiber = spawn_fiber(&shared, job);
            run_fiber_switch(&shared, fiber);
        }
        let expected: usize = (0..1000usize).map(|i| (i as u8) as usize).sum();
        assert_eq!(sum.load(Ordering::Relaxed), expected);
    }

    /// A job that `wait`s on a counter completed by *another* job must suspend
    /// its fiber (yielding the worker) and resume once the counter hits zero,
    /// with no deadlock even with a single worker thread.
    #[test]
    fn waiting_job_yields_and_resumes() {
        let pool = TaskPool::with_threads(1);
        let order = Arc::new(AtomicUsize::new(0));
        let outer = Counter::new();

        let p = pool.clone();
        let ord = Arc::clone(&order);
        pool.spawn(&outer, move || {
            let inner = Counter::new();
            let ord2 = Arc::clone(&ord);
            p.spawn(&inner, move || {
                // Give the waiter a chance to actually suspend first.
                std::thread::sleep(Duration::from_millis(10));
                ord2.fetch_add(1, Ordering::SeqCst);
            });
            // Single worker: this must suspend so the inner job can run.
            p.wait(&inner);
            // Resumed only after inner completed.
            assert_eq!(ord.load(Ordering::SeqCst), 1);
            ord.fetch_add(10, Ordering::SeqCst);
        });
        pool.wait(&outer);
        assert_eq!(order.load(Ordering::SeqCst), 11);
    }

    /// Nested waits: an outer fiber waits on a middle job that itself waits on
    /// an inner job. Each level suspends; the chain must unwind without wedging.
    #[test]
    fn nested_waits_resume_in_order() {
        let pool = TaskPool::with_threads(2);
        let steps = Arc::new(AtomicUsize::new(0));
        let top = Counter::new();

        let p0 = pool.clone();
        let s0 = Arc::clone(&steps);
        pool.spawn(&top, move || {
            let mid = Counter::new();
            let p1 = p0.clone();
            let s1 = Arc::clone(&s0);
            p0.spawn(&mid, move || {
                let inner = Counter::new();
                let s2 = Arc::clone(&s1);
                p1.spawn(&inner, move || {
                    std::thread::sleep(Duration::from_millis(5));
                    s2.fetch_add(1, Ordering::SeqCst);
                });
                p1.wait(&inner);
                s1.fetch_add(1, Ordering::SeqCst);
            });
            p0.wait(&mid);
            s0.fetch_add(1, Ordering::SeqCst);
        });
        pool.wait(&top);
        assert_eq!(steps.load(Ordering::SeqCst), 3);
    }

    /// Stress: many jobs each suspend on their own sub-counter completed by a
    /// spawned sub-job. With far fewer workers than concurrent waiters, this
    /// only completes if suspension frees the worker to run other fibers.
    #[test]
    fn stress_many_suspending_fibers() {
        let pool = TaskPool::with_threads(3);
        let completed = Arc::new(AtomicUsize::new(0));
        let outer = Counter::new();
        const N: usize = 500;
        for i in 0..N {
            let p = pool.clone();
            let done = Arc::clone(&completed);
            pool.spawn(&outer, move || {
                let inner = Counter::new();
                p.spawn(&inner, move || {
                    // tiny variable work
                    let mut acc = 0u64;
                    for k in 0..(i as u64 % 32) {
                        acc = acc.wrapping_add(k);
                    }
                    std::hint::black_box(acc);
                });
                p.wait(&inner);
                done.fetch_add(1, Ordering::SeqCst);
            });
        }
        pool.wait(&outer);
        assert_eq!(completed.load(Ordering::SeqCst), N);
    }

    /// Deep recursive fork-join on a single worker: every `join` on the worker's
    /// fiber suspends, so correctness here proves the suspend/resume machinery
    /// composes recursively without a per-level worker.
    #[test]
    fn single_worker_deep_recursion_via_fibers() {
        let pool = TaskPool::with_threads(1);

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

        assert_eq!(fib(&pool, 13), 233);
    }

    /// Micro-measurement of a full spawn + switch-in + run + teardown cycle.
    /// Not an ns assertion; just a generous ceiling that would only trip on a
    /// gross regression (e.g. accidental blocking in the switch path).
    #[test]
    fn micro_measure_fiber_cycle_cost() {
        let shared = Shared::new(1);
        const ITERS: usize = 20_000;
        let count = Arc::new(AtomicUsize::new(0));
        let start = Instant::now();
        for _ in 0..ITERS {
            let c = Arc::clone(&count);
            let job: Job = Box::new(move || {
                c.fetch_add(1, Ordering::Relaxed);
            });
            let fiber = spawn_fiber(&shared, job);
            run_fiber_switch(&shared, fiber);
        }
        let elapsed = start.elapsed();
        assert_eq!(count.load(Ordering::Relaxed), ITERS);
        // Generous: < 100 µs per full cycle on any sane machine.
        let ceiling = Duration::from_micros(100) * ITERS as u32;
        assert!(elapsed < ceiling, "fiber cycle unexpectedly slow: {elapsed:?}");
    }

    /// A panic inside a fiber job is captured across the asm boundary and
    /// re-raised on the worker, matching the non-fiber propagation semantics.
    #[test]
    fn fiber_job_panic_propagates() {
        let shared = Shared::new(1);
        let job: Job = Box::new(|| panic!("boom in fiber"));
        let fiber = spawn_fiber(&shared, job);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_fiber_switch(&shared, fiber);
        }));
        assert!(result.is_err());
    }
}

/// With the `fibers` feature OFF, `wait` must keep using the help-on-wait
/// fallback: deep nested fork-join still completes on few workers.
#[cfg(not(feature = "fibers"))]
#[test]
fn help_on_wait_fallback_handles_nested_waits() {
    let pool = TaskPool::with_threads(2);
    let steps = Arc::new(AtomicUsize::new(0));
    let top = Counter::new();

    let p0 = pool.clone();
    let s0 = Arc::clone(&steps);
    pool.spawn(&top, move || {
        let mid = Counter::new();
        let s1 = Arc::clone(&s0);
        p0.spawn(&mid, move || {
            s1.fetch_add(1, Ordering::SeqCst);
        });
        // In the fallback this busy-helps rather than suspending.
        p0.wait(&mid);
        s0.fetch_add(1, Ordering::SeqCst);
    });
    pool.wait(&top);
    assert_eq!(steps.load(Ordering::SeqCst), 2);
}

// ---------------------------------------------------------------------------
// M3: async executor + named threads
// ---------------------------------------------------------------------------

/// Async-executor and named-thread tests. These are written to pass in every
/// feature combo the crate ships (default `std`+`multi_thread`, `fibers`, and
/// `--no-default-features --features single`); the few that need true worker
/// concurrency are gated out of the single-threaded fallback.
mod m3 {
    use crate::{Counter, NamedThreads, NamedThreadsConfig, TaskPool, ThreadCategory};
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::sync::{Arc, Condvar, Mutex};
    use std::task::{Context, Poll};
    #[cfg(not(feature = "single"))]
    use std::task::Waker;

    /// Spin until `counter` drains. Combo-independent (works with or without
    /// workers), used where `TaskPool::wait` is unavailable (e.g. waiting on a
    /// named-lane job from the single-threaded fallback, whose `wait` asserts
    /// the counter is already complete).
    fn spin_until_complete(counter: &Counter) {
        while !counter.is_complete() {
            std::thread::yield_now();
        }
    }

    /// A future that returns `Pending` (re-waking itself) `n` times before
    /// yielding `n`. Exercises the waker's "wake while being polled" path.
    struct YieldN {
        remaining: usize,
        total: usize,
        polls: Arc<AtomicUsize>,
    }

    impl Future for YieldN {
        type Output = usize;

        fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<usize> {
            self.polls.fetch_add(1, Ordering::Relaxed);
            if self.remaining == 0 {
                return Poll::Ready(self.total);
            }
            self.remaining -= 1;
            // Re-schedule ourselves: the harness must poll us again.
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }

    /// A single-shot channel future: resolves once another thread calls
    /// [`OneShot::send`], which registers-and-fires a stored waker. Exercises
    /// the harness's "wake after poll returned Pending" re-enqueue path. Only
    /// the worker-backed tests use it; the single-threaded fallback cannot
    /// `block_on` an event that only an external thread delivers.
    #[cfg(not(feature = "single"))]
    struct OneShot<T> {
        state: Mutex<OneShotState<T>>,
    }

    #[cfg(not(feature = "single"))]
    struct OneShotState<T> {
        value: Option<T>,
        waker: Option<Waker>,
    }

    #[cfg(not(feature = "single"))]
    impl<T> OneShot<T> {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                state: Mutex::new(OneShotState {
                    value: None,
                    waker: None,
                }),
            })
        }

        fn send(&self, value: T) {
            let waker = {
                let mut state = self.state.lock().unwrap();
                state.value = Some(value);
                state.waker.take()
            };
            if let Some(waker) = waker {
                waker.wake();
            }
        }

        fn recv(self: &Arc<Self>) -> OneShotRecv<T> {
            OneShotRecv {
                shared: Arc::clone(self),
            }
        }
    }

    #[cfg(not(feature = "single"))]
    struct OneShotRecv<T> {
        shared: Arc<OneShot<T>>,
    }

    #[cfg(not(feature = "single"))]
    impl<T> Future for OneShotRecv<T> {
        type Output = T;

        fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<T> {
            let mut state = self.shared.state.lock().unwrap();
            if let Some(value) = state.value.take() {
                return Poll::Ready(value);
            }
            state.waker = Some(cx.waker().clone());
            Poll::Pending
        }
    }

    #[test]
    fn block_on_returns_immediate_value() {
        let pool = TaskPool::new();
        let out = pool.block_on(async { 2 + 40 });
        assert_eq!(out, 42);
    }

    #[test]
    fn spawn_async_output_via_block_on() {
        let pool = TaskPool::new();
        let task = pool.spawn_async(async { 7 * 6 });
        let out = pool.block_on(task);
        assert_eq!(out, 42);
    }

    #[test]
    fn block_on_awaits_nested_spawn() {
        let pool = TaskPool::new();
        let p = pool.clone();
        let out = pool.block_on(async move {
            let a = p.spawn_async(async { 10usize });
            let b = p.spawn_async(async { 32usize });
            a.await + b.await
        });
        assert_eq!(out, 42);
    }

    #[test]
    fn waker_reschedules_yielding_future() {
        let pool = TaskPool::new();
        let polls = Arc::new(AtomicUsize::new(0));
        let task = pool.spawn_async(YieldN {
            remaining: 5,
            total: 5,
            polls: Arc::clone(&polls),
        });
        let out = pool.block_on(task);
        assert_eq!(out, 5);
        // One poll per yield (5) plus the final `Ready` poll.
        assert_eq!(polls.load(Ordering::Relaxed), 6);
    }

    #[test]
    fn counter_completion_wakes_future() {
        // In the multi-threaded pool the jobs run on workers while `block_on`
        // parks; in the single-threaded fallback `spawn` runs them inline, so
        // the counter is already drained when the future first polls. Either
        // way the awaiting future must observe completion.
        let pool = TaskPool::new();
        let counter = Counter::new();
        let hits = Arc::new(AtomicUsize::new(0));
        for _ in 0..16 {
            let hits = Arc::clone(&hits);
            pool.spawn(&counter, move || {
                hits.fetch_add(1, Ordering::Relaxed);
            });
        }
        pool.block_on(counter.wait_async());
        assert!(counter.is_complete());
        assert_eq!(hits.load(Ordering::Relaxed), 16);
    }

    #[test]
    fn await_many_spawned_tasks() {
        let pool = TaskPool::new();
        let p = pool.clone();
        let out = pool.block_on(async move {
            let tasks: Vec<_> = (0..64u64).map(|i| p.spawn_async(async move { i * i })).collect();
            let mut sum = 0u64;
            for task in tasks {
                sum += task.await;
            }
            sum
        });
        assert_eq!(out, (0..64u64).map(|i| i * i).sum());
    }

    #[test]
    fn task_is_finished_reports_completion() {
        let pool = TaskPool::new();
        let task = pool.spawn_async(async { 1u8 });
        let value = pool.block_on(task);
        assert_eq!(value, 1);
    }

    // --- Named threads (combo-independent: lanes are their own OS threads) ---

    #[test]
    fn dispatch_render_io_async_compute_run_off_pool() {
        let named = NamedThreads::new();
        for category in [
            ThreadCategory::Render,
            ThreadCategory::Io,
            ThreadCategory::AsyncCompute,
        ] {
            let (tx, rx) = mpsc::channel();
            let counter = named.dispatch(category, move || {
                tx.send(category).unwrap();
            });
            // The job's side effect arrives via the channel from the lane
            // thread; then the tracking counter drains.
            assert_eq!(rx.recv().unwrap(), category);
            spin_until_complete(&counter);
            assert!(counter.is_complete());
        }
    }

    #[test]
    fn dispatch_io_fans_out_across_lane_threads() {
        let named = NamedThreads::with_config(NamedThreadsConfig {
            io_threads: 4,
            async_compute_threads: 1,
        });
        let done = Arc::new(AtomicUsize::new(0));
        let mut counters = Vec::new();
        for _ in 0..64 {
            let done = Arc::clone(&done);
            counters.push(named.dispatch(ThreadCategory::Io, move || {
                done.fetch_add(1, Ordering::Relaxed);
            }));
        }
        for c in &counters {
            spin_until_complete(c);
        }
        assert_eq!(done.load(Ordering::Relaxed), 64);
    }

    #[test]
    fn main_lane_runs_only_when_pumped() {
        let named = NamedThreads::new();
        let ran = Arc::new(AtomicUsize::new(0));
        let r = Arc::clone(&ran);
        let counter = named.dispatch(ThreadCategory::Main, move || {
            r.fetch_add(1, Ordering::Relaxed);
        });
        // Nothing runs the Main lane until we pump it.
        assert_eq!(named.main_pending(), 1);
        assert_eq!(ran.load(Ordering::Relaxed), 0);
        assert!(!counter.is_complete());

        let executed = named.run_main_pending();
        assert_eq!(executed, 1);
        assert_eq!(ran.load(Ordering::Relaxed), 1);
        assert_eq!(named.main_pending(), 0);
        assert!(counter.is_complete());
    }

    #[test]
    fn main_lane_pump_is_batched_against_self_enqueue() {
        // A Main job that re-dispatches to Main must not spin the pump forever:
        // `run_main_pending` only runs the jobs queued at entry.
        let named = Arc::new(NamedThreads::new());
        let ran = Arc::new(AtomicUsize::new(0));
        let n2 = Arc::clone(&named);
        let r = Arc::clone(&ran);
        named.dispatch(ThreadCategory::Main, move || {
            r.fetch_add(1, Ordering::Relaxed);
            let r2 = Arc::clone(&r);
            n2.dispatch(ThreadCategory::Main, move || {
                r2.fetch_add(1, Ordering::Relaxed);
            });
        });
        let first = named.run_main_pending();
        assert_eq!(first, 1);
        assert_eq!(ran.load(Ordering::Relaxed), 1);
        // The re-dispatched job waits for the next pump.
        assert_eq!(named.main_pending(), 1);
        let second = named.run_main_pending();
        assert_eq!(second, 1);
        assert_eq!(ran.load(Ordering::Relaxed), 2);
    }

    #[cfg(not(feature = "single"))]
    #[test]
    fn counter_wait_async_bridges_named_dispatch() {
        // A named-lane job's completion counter resolves a future.
        let pool = TaskPool::new();
        let named = NamedThreads::new();
        let flag = Arc::new(AtomicUsize::new(0));
        let f = Arc::clone(&flag);
        let counter = named.dispatch(ThreadCategory::AsyncCompute, move || {
            f.fetch_add(99, Ordering::Relaxed);
        });
        pool.block_on(counter.wait_async());
        assert_eq!(flag.load(Ordering::Relaxed), 99);
    }

    // --- Tests that require true worker concurrency (not the inline fallback) ---

    /// External wake (from another OS thread) must re-enqueue a parked harness.
    /// Skipped under `single`, whose `block_on` fails fast rather than waiting
    /// on an event only an external thread can deliver.
    #[cfg(not(feature = "single"))]
    #[test]
    fn external_wake_reenqueues_parked_task() {
        let pool = TaskPool::new();
        let channel = OneShot::<u64>::new();
        let producer = Arc::clone(&channel);
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(20));
            producer.send(1234);
        });
        let recv = channel.recv();
        let task = pool.spawn_async(async move { recv.await });
        let out = pool.block_on(task);
        assert_eq!(out, 1234);
    }

    /// A *job* can wait on an async task's completion counter, bridging the
    /// async result back into the fork-join world. Needs real workers to drive
    /// the spawned future while the waiting job helps.
    #[cfg(not(feature = "single"))]
    #[test]
    fn job_waits_on_task_counter() {
        let pool = TaskPool::new();
        let shared = Arc::new(AtomicUsize::new(0));
        let s = Arc::clone(&shared);
        let task = pool.spawn_async(async move {
            s.store(55, Ordering::SeqCst);
        });
        let task_counter = task.counter();
        task.detach();

        let outer = Counter::new();
        let p = pool.clone();
        let probe = Arc::clone(&shared);
        pool.spawn(&outer, move || {
            // Wait on the async task from inside a job.
            p.wait(&task_counter);
            assert_eq!(probe.load(Ordering::SeqCst), 55);
        });
        pool.wait(&outer);
    }

    /// Stress: many independent futures resolved by a counter completed on the
    /// pool, driven by `block_on` while workers run the jobs. Must not deadlock.
    #[cfg(not(feature = "single"))]
    #[test]
    fn stress_many_counter_futures_no_deadlock() {
        let pool = TaskPool::with_threads(3);
        let p = pool.clone();
        let total = pool.block_on(async move {
            let mut sum = 0usize;
            for round in 0..50 {
                let counter = Counter::new();
                let hits = Arc::new(AtomicUsize::new(0));
                for _ in 0..8 {
                    let hits = Arc::clone(&hits);
                    p.spawn(&counter, move || {
                        hits.fetch_add(1, Ordering::Relaxed);
                    });
                }
                counter.wait_async().await;
                assert_eq!(hits.load(Ordering::Relaxed), 8, "round {round}");
                sum += hits.load(Ordering::Relaxed);
            }
            sum
        });
        assert_eq!(total, 50 * 8);
    }

    /// A oneshot completed by a pool job must wake a future blocked in
    /// `block_on` on the main thread. Exercises cross-worker wakeups.
    #[cfg(not(feature = "single"))]
    #[test]
    fn pool_job_wakes_block_on() {
        let pool = TaskPool::with_threads(2);
        let channel = OneShot::<&'static str>::new();
        let producer = Arc::clone(&channel);
        let counter = Counter::new();
        pool.spawn(&counter, move || {
            producer.send("ready");
        });
        let recv = channel.recv();
        let out = pool.block_on(recv);
        assert_eq!(out, "ready");
        pool.wait(&counter);
    }

    /// Condvar-backed sanity: dispatch to every lane and confirm all run,
    /// ordering-independent, using a shared tally guarded by a condvar.
    #[test]
    fn all_categories_execute() {
        let named = NamedThreads::new();
        let state = Arc::new((Mutex::new(0usize), Condvar::new()));
        let categories = [
            ThreadCategory::Render,
            ThreadCategory::Io,
            ThreadCategory::AsyncCompute,
        ];
        for category in categories {
            let state = Arc::clone(&state);
            named.dispatch(category, move || {
                let (lock, cvar) = &*state;
                *lock.lock().unwrap() += 1;
                cvar.notify_all();
            });
        }
        // Main runs inline on pump.
        let state_main = Arc::clone(&state);
        named.dispatch(ThreadCategory::Main, move || {
            let (lock, cvar) = &*state_main;
            *lock.lock().unwrap() += 1;
            cvar.notify_all();
        });
        named.run_main_pending();

        let (lock, cvar) = &*state;
        let mut done = lock.lock().unwrap();
        while *done < 4 {
            done = cvar.wait(done).unwrap();
        }
        assert_eq!(*done, 4);
    }
}

/// M4 (内存/亲和/NUMA) pool-level wiring: frame arenas, worker pinning,
/// worker-index lookup, and the affinity→NUMA steal-order plumbing. The
/// mechanism-level unit tests live in `arena.rs`, `numa.rs`, and `affinity.rs`;
/// these exercise the `TaskPool` surface that stitches them together.
mod m4 {
    use crate::{
        CoreClass, CoreClassPolicy, CoreInfo, NumaNodeId, TaskPool, TaskPoolConfig, Topology,
        affinity_supported, plan_worker_cores, steal_order,
    };
    // These are only touched by the worker-backed tests below, which the
    // synchronous `single` fallback compiles out.
    #[cfg(not(feature = "single"))]
    use crate::{Counter, DEFAULT_ARENA_CAPACITY};
    #[cfg(not(feature = "single"))]
    use std::sync::atomic::{AtomicUsize, Ordering};
    #[cfg(not(feature = "single"))]
    use std::sync::{Arc, Mutex};

    /// Drain `counter` by spinning *without* helping, so queued jobs run only
    /// on the pool's worker threads (never inline on this external thread).
    /// Lets the worker-identity / per-worker-arena assertions below be
    /// deterministic rather than racing `TaskPool::wait`'s help-on-wait.
    #[cfg(not(feature = "single"))]
    fn drain_on_workers(counter: &Counter) {
        while !counter.is_complete() {
            std::thread::yield_now();
        }
    }

    #[test]
    fn default_config_does_not_pin() {
        let cfg = TaskPoolConfig::default();
        assert!(!cfg.pin_workers);
        assert_eq!(cfg.core_class_policy, CoreClassPolicy::PerformanceFirst);
    }

    #[test]
    fn pool_affinity_supported_matches_free_fn() {
        // The pool accessor must forward the platform capability verbatim —
        // never fake support where the OS lacks it (macOS reports false).
        let pool = TaskPool::with_threads(2);
        assert_eq!(pool.affinity_supported(), affinity_supported());
    }

    #[test]
    fn single_pool_frame_arenas_has_one_arena() {
        // Even the synchronous fallback yields a usable (>=1) arena set so
        // frame-scratch code works regardless of the configured thread count.
        let pool = TaskPool::with_threads(0);
        let arenas = pool.new_frame_arenas(4096);
        assert_eq!(arenas.len(), 1);
        assert_eq!(arenas.arena(0).unwrap().capacity(), 4096);
    }

    #[test]
    fn current_worker_index_none_on_external_thread() {
        // The calling (test) thread is never one of the pool's workers.
        let pool = TaskPool::with_threads(2);
        assert_eq!(pool.current_worker_index(), None);
    }

    #[test]
    fn plan_feeds_steal_order_end_to_end() {
        // A synthetic two-node topology drives the worker->node vector that the
        // steal policy consumes — the affinity+NUMA wiring, exercised without
        // any platform NUMA map (honest, machine-independent).
        let topo = Topology::from_cores(vec![
            CoreInfo { id: 0, node: NumaNodeId::new(0), class: CoreClass::Performance },
            CoreInfo { id: 1, node: NumaNodeId::new(0), class: CoreClass::Performance },
            CoreInfo { id: 2, node: NumaNodeId::new(1), class: CoreClass::Performance },
            CoreInfo { id: 3, node: NumaNodeId::new(1), class: CoreClass::Performance },
        ]);
        let plan = plan_worker_cores(&topo, 4, CoreClassPolicy::Flat);
        let nodes = plan.worker_nodes();
        // Worker 0 (node 0): same-node victim 1 first, then cross-node 2, 3.
        assert_eq!(steal_order(&nodes, 0), vec![1, 2, 3]);
        // Worker 2 (node 1): same-node victim 3 first, then cross-node 0, 1.
        assert_eq!(steal_order(&nodes, 2), vec![3, 0, 1]);
    }

    #[cfg(not(feature = "single"))]
    #[test]
    fn pinned_pool_runs_all_jobs() {
        // Pinning is best-effort: whether or not the OS honors it, every job
        // must still run exactly once and the pool must not deadlock.
        let pool = TaskPool::with_config(TaskPoolConfig {
            threads: 4,
            pin_workers: true,
            core_class_policy: CoreClassPolicy::PerformanceFirst,
        });
        let counter = Counter::new();
        let sum = Arc::new(AtomicUsize::new(0));
        for i in 1..=200 {
            let sum = Arc::clone(&sum);
            pool.spawn(&counter, move || {
                sum.fetch_add(i, Ordering::Relaxed);
            });
        }
        pool.wait(&counter);
        assert!(counter.is_complete());
        assert_eq!(sum.load(Ordering::Relaxed), (1..=200).sum());
    }

    #[cfg(not(feature = "single"))]
    #[test]
    fn new_frame_arenas_sized_per_worker() {
        let pool = TaskPool::with_threads(4);
        let arenas = pool.new_frame_arenas(DEFAULT_ARENA_CAPACITY);
        assert_eq!(arenas.len(), pool.worker_count());
        for w in 0..arenas.len() {
            let arena = arenas.arena(w).unwrap();
            assert_eq!(arena.capacity(), DEFAULT_ARENA_CAPACITY);
            // Unpinned pool on a platform with no NUMA map => node 0.
            assert_eq!(arena.node(), NumaNodeId::ZERO);
        }
    }

    #[cfg(not(feature = "single"))]
    #[test]
    fn current_worker_index_some_inside_job() {
        let pool = TaskPool::with_threads(3);
        let counter = Counter::new();
        let seen = Arc::new(Mutex::new(Vec::new()));
        for _ in 0..64 {
            let p = pool.clone();
            let seen = Arc::clone(&seen);
            pool.spawn(&counter, move || {
                if let Some(idx) = p.current_worker_index() {
                    seen.lock().unwrap().push(idx);
                }
            });
        }
        drain_on_workers(&counter);
        let seen = seen.lock().unwrap();
        // Draining only on workers means every job observed a worker index,
        // and each index is a valid worker of this pool.
        assert_eq!(seen.len(), 64);
        assert!(seen.iter().all(|&i| i < pool.worker_count()));
    }

    #[cfg(not(feature = "single"))]
    #[test]
    fn jobs_allocate_from_their_worker_arena() {
        let pool = TaskPool::with_threads(3);
        let arenas = Arc::new(pool.new_frame_arenas(64 * 1024));
        let counter = Counter::new();
        for _ in 0..128 {
            let p = pool.clone();
            let arenas = Arc::clone(&arenas);
            pool.spawn(&counter, move || {
                if let Some(idx) = p.current_worker_index() {
                    if let Some(arena) = arenas.arena(idx) {
                        // Bump a small per-job record; overflow just falls back
                        // to the heap (alloc returns Err), which is also fine.
                        let _ = arena.alloc([0u8; 128]);
                    }
                }
            });
        }
        drain_on_workers(&counter);
        // Lock-free bump allocation from inside live worker jobs recorded a
        // peak across the per-worker arenas.
        assert!(arenas.total_high_water() > 0);
    }

    #[cfg(not(feature = "single"))]
    #[test]
    fn frame_arenas_reset_between_frames() {
        let pool = TaskPool::with_threads(2);
        let mut arenas = pool.new_frame_arenas(4096);
        // Frame 1: fill scratch directly (single owner, no sharing needed).
        for w in 0..arenas.len() {
            let _ = arenas.arena(w).unwrap().alloc([1u8; 256]).unwrap();
            assert!(arenas.arena(w).unwrap().used() >= 256);
        }
        let peak = arenas.total_high_water();
        assert!(peak >= 256 * arenas.len());
        // Frame 2: whole-arena reset rewinds every cursor; peak survives.
        arenas.reset_all();
        for w in 0..arenas.len() {
            assert_eq!(arenas.arena(w).unwrap().used(), 0);
        }
        assert_eq!(arenas.total_high_water(), peak);
    }

    // Running jobs drives them onto large-class fibers (the default class), so
    // the pool reports a non-zero large-class peak and a still-cold small class
    // (design §8 大/小两档; §16 water mark). A freshly built pool reports zero.
    #[cfg(all(feature = "fibers", not(feature = "single")))]
    #[test]
    fn fiber_stack_high_water_tracks_large_class_usage() {
        let pool = TaskPool::with_threads(3);
        // Nothing has run yet: peak occupancy is zero for both classes.
        let fresh = pool.fiber_stack_high_water();
        assert_eq!(fresh.small_high_water, 0);
        assert_eq!(fresh.large_high_water, 0);
        // Run a batch of jobs; each executes on a large-class fiber.
        let counter = Counter::new();
        for _ in 0..8 {
            pool.spawn(&counter, || {});
        }
        pool.wait(&counter);
        let after = pool.fiber_stack_high_water();
        assert!(after.large_high_water >= 1);
        // Shallow jobs never request the small class, so it stays cold.
        assert_eq!(after.small_high_water, 0);
    }

    // In the synchronous fallback no fibers run, so occupancy is zero.
    #[cfg(all(feature = "fibers", feature = "single"))]
    #[test]
    fn fiber_stack_high_water_zero_in_single_fallback() {
        let pool = TaskPool::new();
        let stats = pool.fiber_stack_high_water();
        assert_eq!(stats.small_high_water, 0);
        assert_eq!(stats.large_high_water, 0);
    }
}
