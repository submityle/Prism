use super::{HarnessInner, RunState, RunnableTask, TaskHarness, POLL_QUANTUM, SINGLE_READY_BATCH};
use crate::job::Job;
use crate::scheduler::Shared;
use crate::{Counter, TaskPool};
use alloc::sync::Arc;
use core::future::poll_fn;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use core::task::{Poll, Waker};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{mpsc, Barrier, Mutex};
use std::time::Duration;

const DEADLINE: Duration = Duration::from_secs(2);
// A broken executor still terminates these tests, but only after violating the
// poll-count bounds. Timeouts guard handshakes, not scheduling/fairness claims.
const POLL_FUSE: usize = POLL_QUANTUM * SINGLE_READY_BATCH * 4;

#[test]
fn zero_worker_self_wake_cannot_starve_compute_or_root() {
    let pool = TaskPool::with_threads(0);
    let stop = Arc::new(AtomicBool::new(false));
    let polls = Arc::new(AtomicUsize::new(0));
    let background = pool.spawn_async({
        let stop = Arc::clone(&stop);
        let polls = Arc::clone(&polls);
        poll_fn(move |cx| {
            let count = polls.fetch_add(1, Ordering::SeqCst) + 1;
            if stop.load(Ordering::SeqCst) || count >= POLL_FUSE {
                Poll::Ready(())
            } else {
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        })
    });
    // An independently queued, finite computation must get past the background
    // harness too, not just the directly-polled root future.
    let compute = pool.spawn_async(async { (0..7).sum::<usize>() * 2 });
    let counter = Counter::new();
    let ordinary_at = Arc::new(AtomicUsize::new(usize::MAX));
    let mut root_observations = Vec::new();
    pool.block_on(poll_fn(|cx| {
        root_observations.push(polls.load(Ordering::SeqCst));
        if root_observations.len() == 1 {
            // Zero-worker ordinary jobs run inline. Placing one in the root
            // proves background draining cannot prevent its submission.
            let polls = Arc::clone(&polls);
            let ordinary_at = Arc::clone(&ordinary_at);
            pool.spawn(&counter, move || {
                ordinary_at.store(polls.load(Ordering::SeqCst), Ordering::SeqCst);
            });
            cx.waker().wake_by_ref();
            Poll::Pending
        } else {
            stop.store(true, Ordering::SeqCst);
            Poll::Ready(())
        }
    }));
    let compute_finished_before_cleanup = compute.is_finished();
    pool.block_on(background);
    assert_eq!(pool.block_on(compute), 42);

    assert_eq!(root_observations.len(), 2);
    assert!(root_observations[0] <= POLL_QUANTUM * SINGLE_READY_BATCH);
    assert!(root_observations[1] <= 2 * POLL_QUANTUM * SINGLE_READY_BATCH);
    assert_eq!(ordinary_at.load(Ordering::SeqCst), root_observations[0]);
    assert!(counter.is_complete());
    assert!(compute_finished_before_cleanup);
}

#[test]
#[cfg(not(feature = "single"))]
fn one_worker_self_wake_yields_to_local_and_injected_compute_and_wakes_root() {
    let pool = TaskPool::with_threads(1);
    let stop = Arc::new(AtomicBool::new(false));
    let polls = Arc::new(AtomicUsize::new(0));
    let local_at = Arc::new(AtomicUsize::new(usize::MAX));
    let injected_at = Arc::new(AtomicUsize::new(usize::MAX));
    let local_counter = Counter::new();
    let injected_counter = Counter::new();
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let background = pool.spawn_async({
        let pool = pool.clone();
        let stop = Arc::clone(&stop);
        let polls = Arc::clone(&polls);
        let local_at = Arc::clone(&local_at);
        let local_counter = local_counter.clone();
        poll_fn(move |cx| {
            let count = polls.fetch_add(1, Ordering::SeqCst) + 1;
            if count == 1 {
                let local_polls = Arc::clone(&polls);
                let local_at = Arc::clone(&local_at);
                pool.spawn(&local_counter, move || {
                    local_at.store(local_polls.load(Ordering::SeqCst), Ordering::SeqCst);
                });
                entered_tx.send(()).unwrap();
                release_rx.recv_timeout(DEADLINE).unwrap();
            }
            if stop.load(Ordering::SeqCst) || count >= POLL_FUSE {
                Poll::Ready(())
            } else {
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        })
    });
    entered_rx.recv_timeout(DEADLINE).unwrap();
    // The only worker is inside poll #1, so this ordinary job is definitely in
    // the injector before the background harness can yield its first quantum.
    pool.spawn(&injected_counter, {
        let stop = Arc::clone(&stop);
        let polls = Arc::clone(&polls);
        let injected_at = Arc::clone(&injected_at);
        move || {
            injected_at.store(polls.load(Ordering::SeqCst), Ordering::SeqCst);
            stop.store(true, Ordering::SeqCst);
        }
    });
    pool.block_on(async {
        release_tx.send(()).unwrap();
        // Do not use pool.wait: its caller-side help could hide starvation on
        // the sole worker. Completion must wake this root through the counter.
        injected_counter.wait_async().await;
    });
    pool.block_on(background);

    assert_eq!(local_at.load(Ordering::SeqCst), POLL_QUANTUM);
    assert_eq!(injected_at.load(Ordering::SeqCst), POLL_QUANTUM);
    assert!(local_counter.is_complete());
    assert!(injected_counter.is_complete());
    assert_eq!(polls.load(Ordering::SeqCst), POLL_QUANTUM + 1);
}

#[test]
#[cfg(not(feature = "single"))]
fn worker_spawned_async_starts_behind_already_injected_work() {
    let pool = TaskPool::with_threads(1);
    let parent_counter = Counter::new();
    let injected_counter = Counter::new();
    let injected_done = Arc::new(AtomicBool::new(false));
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let (task_tx, task_rx) = mpsc::channel();
    pool.spawn(&parent_counter, {
        let pool = pool.clone();
        let injected_done = Arc::clone(&injected_done);
        move || {
            entered_tx.send(()).unwrap();
            release_rx.recv_timeout(DEADLINE).unwrap();
            // The first dispatch, not only quantum continuations, must use
            // the injector even though spawn_async is called on a worker.
            let task = pool.spawn_async(async move { injected_done.load(Ordering::SeqCst) });
            task_tx.send(task).unwrap();
        }
    });
    entered_rx.recv_timeout(DEADLINE).unwrap();
    pool.spawn(&injected_counter, {
        let injected_done = Arc::clone(&injected_done);
        move || injected_done.store(true, Ordering::SeqCst)
    });
    release_tx.send(()).unwrap();
    let task = task_rx.recv_timeout(DEADLINE).unwrap();
    // Caller-side help would let the ordinary job run on the caller and hide
    // a local-LIFO enqueue regression, so await only the async result here.
    assert!(pool.block_on(task));
    assert!(parent_counter.is_complete());
    assert!(injected_counter.is_complete());
}

fn wake_concurrently(waker: &Waker) {
    const WAKERS: usize = 8;
    let start = Barrier::new(WAKERS + 1);
    std::thread::scope(|scope| {
        for _ in 0..WAKERS {
            let start = &start;
            let waker = waker.clone();
            scope.spawn(move || {
                start.wait();
                for _ in 0..32 {
                    waker.wake_by_ref();
                }
                waker.wake();
            });
        }
        start.wait();
    });
}

fn pop_dispatch(shared: &Shared, single: bool) -> Option<Job> {
    if single {
        shared
            .pop_async_ready()
            .map(|task| Box::new(move || task.run_boxed()) as Job)
    } else {
        shared.find_task(Some(0))
    }
}

fn last_quantum_poll(ready: bool, prioritized: bool) {
    for single in [true, false] {
        let shared = Arc::new(Shared::new(usize::from(!single)));
        let polls = Arc::new(AtomicUsize::new(0));
        let saved_waker = Arc::new(Mutex::new(None));
        let harness = Arc::new(TaskHarness {
            inner: Mutex::new(HarnessInner {
                future: Some(Box::pin({
                    let polls = Arc::clone(&polls);
                    let saved_waker = Arc::clone(&saved_waker);
                    poll_fn(move |cx| {
                        let count = polls.fetch_add(1, Ordering::SeqCst) + 1;
                        if count == POLL_QUANTUM {
                            *saved_waker.lock().unwrap() = Some(cx.waker().clone());
                        }
                        if count < POLL_QUANTUM || ready {
                            cx.waker().wake_by_ref();
                        }
                        if count > POLL_QUANTUM || (ready && count == POLL_QUANTUM) {
                            Poll::Ready(())
                        } else {
                            Poll::Pending
                        }
                    })
                })),
                run: RunState::Queued,
            }),
            shared: Arc::clone(&shared),
            single,
            priority: prioritized.then(|| Arc::new(crate::PriorityCell::new(crate::Priority::Low))),
            on_panic: Box::new(|payload| std::panic::resume_unwind(payload)),
        });
        Arc::clone(&harness).enqueue();
        pop_dispatch(&shared, single).unwrap()();
        assert_eq!(polls.load(Ordering::SeqCst), POLL_QUANTUM);
        {
            let inner = harness.inner.lock().unwrap();
            assert!(
                inner.run
                    == if ready {
                        RunState::Done
                    } else {
                        RunState::Idle
                    }
            );
            assert_eq!(inner.future.is_none(), ready);
        }
        // Reaching the quantum alone must not enqueue a continuation: Ready
        // wins over its wake, and an unwoken Pending must become Idle.
        assert!(pop_dispatch(&shared, single).is_none());
        let waker = saved_waker.lock().unwrap().take().unwrap();
        wake_concurrently(&waker);
        if !ready {
            let finish = pop_dispatch(&shared, single).unwrap();
            assert!(pop_dispatch(&shared, single).is_none());
            finish();
        }
        assert!(pop_dispatch(&shared, single).is_none());
        assert_eq!(
            polls.load(Ordering::SeqCst),
            POLL_QUANTUM + usize::from(!ready)
        );
        let inner = harness.inner.lock().unwrap();
        assert!(inner.run == RunState::Done);
        assert!(inner.future.is_none());
    }
}

#[test]
fn ready_on_last_quantum_poll_does_not_requeue_even_when_woken() {
    last_quantum_poll(true, false);
}

#[test]
fn unwoken_pending_on_last_quantum_poll_waits_for_an_external_wake() {
    last_quantum_poll(false, false);
}

#[test]
fn prioritized_ready_on_last_quantum_poll_does_not_requeue_even_when_woken() {
    last_quantum_poll(true, true);
}

#[test]
fn prioritized_unwoken_pending_on_last_quantum_poll_waits_for_an_external_wake() {
    last_quantum_poll(false, true);
}

fn concurrent_wakes_across_quantum(single: bool, prioritized: bool) {
    // Drive queues manually: no scheduling luck can hide an extra entry or a
    // lost wake. The only runner thread is held inside the last quantum poll.
    let shared = Arc::new(Shared::new(usize::from(!single)));
    let polls = Arc::new(AtomicUsize::new(0));
    let in_poll = Arc::new(AtomicBool::new(false));
    let (boundary_tx, boundary_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let harness = Arc::new(TaskHarness {
        inner: Mutex::new(HarnessInner {
            future: Some(Box::pin({
                let polls = Arc::clone(&polls);
                let in_poll = Arc::clone(&in_poll);
                poll_fn(move |cx| {
                    assert!(!in_poll.swap(true, Ordering::SeqCst), "overlapping polls");
                    let count = polls.fetch_add(1, Ordering::SeqCst) + 1;
                    let result = if count < POLL_QUANTUM {
                        cx.waker().wake_by_ref();
                        Poll::Pending
                    } else if count == POLL_QUANTUM {
                        boundary_tx.send(cx.waker().clone()).unwrap();
                        release_rx.recv_timeout(DEADLINE).unwrap();
                        // No self-wake here: the external concurrent wakes
                        // must carry the harness across the quantum boundary.
                        Poll::Pending
                    } else if count == POLL_QUANTUM + 1 {
                        Poll::Pending // Exercise Running -> Idle next.
                    } else {
                        Poll::Ready(())
                    };
                    in_poll.store(false, Ordering::SeqCst);
                    result
                })
            })),
            run: RunState::Queued,
        }),
        shared: Arc::clone(&shared),
        single,
        priority: prioritized.then(|| Arc::new(crate::PriorityCell::new(crate::Priority::Low))),
        on_panic: Box::new(|payload| std::panic::resume_unwind(payload)),
    });
    Arc::clone(&harness).enqueue();
    let dispatch = pop_dispatch(&shared, single).unwrap();
    let runner = std::thread::spawn(dispatch);
    let waker = boundary_rx.recv_timeout(DEADLINE).unwrap();

    // A spurious runner cannot claim the in-flight future, even on a different
    // driver thread. Concurrent wakes mark exactly one pending continuation.
    Arc::clone(&harness).run_boxed();
    wake_concurrently(&waker);
    release_tx.send(()).unwrap();
    runner.join().unwrap();
    assert_eq!(polls.load(Ordering::SeqCst), POLL_QUANTUM);
    assert!(matches!(
        harness.inner.lock().unwrap().run,
        RunState::Queued
    ));

    let continuation = pop_dispatch(&shared, single).unwrap();
    wake_concurrently(&waker); // Queued, including the dequeue-to-claim window.
    assert!(pop_dispatch(&shared, single).is_none());
    continuation();
    assert_eq!(polls.load(Ordering::SeqCst), POLL_QUANTUM + 1);
    assert!(matches!(harness.inner.lock().unwrap().run, RunState::Idle));
    assert!(pop_dispatch(&shared, single).is_none());

    wake_concurrently(&waker); // Idle -> Queued must enqueue exactly once.
    let finish = pop_dispatch(&shared, single).unwrap();
    assert!(pop_dispatch(&shared, single).is_none());
    finish();
    assert_eq!(polls.load(Ordering::SeqCst), POLL_QUANTUM + 2);
    {
        let inner = harness.inner.lock().unwrap();
        assert!(matches!(inner.run, RunState::Done));
        assert!(inner.future.is_none());
    }
    wake_concurrently(&waker); // Stale wakes and runners cannot poll after Ready.
    Arc::clone(&harness).run_boxed();
    assert!(pop_dispatch(&shared, single).is_none());
    assert_eq!(polls.load(Ordering::SeqCst), POLL_QUANTUM + 2);
    assert!(!in_poll.load(Ordering::SeqCst));
}

#[test]
fn concurrent_wakes_are_coalesced_on_the_zero_worker_ready_queue() {
    concurrent_wakes_across_quantum(true, false);
}

#[test]
fn concurrent_wakes_are_coalesced_on_the_worker_injector() {
    concurrent_wakes_across_quantum(false, false);
}

#[test]
fn prioritized_concurrent_wakes_are_coalesced_on_the_zero_worker_ready_queue() {
    concurrent_wakes_across_quantum(true, true);
}

#[test]
fn prioritized_concurrent_wakes_are_coalesced_on_the_worker_priority_inbox() {
    concurrent_wakes_across_quantum(false, true);
}

struct DropSignal(mpsc::Sender<()>);

impl Drop for DropSignal {
    fn drop(&mut self) {
        let _ = self.0.send(());
    }
}

fn panic_after_yield(workers: usize) {
    let pool = TaskPool::with_threads(workers);
    let polls = Arc::new(AtomicUsize::new(0));
    let saved_waker = Arc::new(Mutex::new(None));
    let (dropped_tx, dropped_rx) = mpsc::channel();
    let task = pool.spawn_async({
        let polls = Arc::clone(&polls);
        let saved_waker = Arc::clone(&saved_waker);
        let drop_signal = DropSignal(dropped_tx);
        poll_fn(move |cx| -> Poll<()> {
            let _keep_alive = &drop_signal;
            *saved_waker.lock().unwrap() = Some(cx.waker().clone());
            let count = polls.fetch_add(1, Ordering::SeqCst) + 1;
            cx.waker().wake_by_ref();
            assert!(count <= POLL_QUANTUM, "panic after a cooperative yield");
            Poll::Pending
        })
    });
    let counter = task.counter();
    assert!(catch_unwind(AssertUnwindSafe(|| pool.block_on(task))).is_err());
    dropped_rx.recv_timeout(DEADLINE).unwrap();
    assert!(counter.is_complete());
    assert_eq!(polls.load(Ordering::SeqCst), POLL_QUANTUM + 1);
    wake_concurrently(&saved_waker.lock().unwrap().take().unwrap());
    // A panic must not poison the harness, requeue it via stale wakes, or kill
    // the worker: unrelated follow-up work still completes.
    assert_eq!(pool.block_on(pool.spawn_async(async { 42 })), 42);
    assert_eq!(polls.load(Ordering::SeqCst), POLL_QUANTUM + 1);
    if pool.is_single_threaded() {
        assert!(pool.shared_state().async_ready_is_empty());
    }
}

#[test]
fn zero_worker_panic_after_yield_finishes_and_ignores_stale_wakes() {
    panic_after_yield(0);
}

#[test]
#[cfg(not(feature = "single"))]
fn one_worker_panic_after_yield_finishes_and_ignores_stale_wakes() {
    panic_after_yield(1);
}
