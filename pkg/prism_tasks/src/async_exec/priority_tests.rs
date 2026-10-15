use super::POLL_QUANTUM;
use crate::job::Job;
use crate::scheduler::Shared;
use crate::{Priority, PriorityCell, Task, TaskPool};
use alloc::sync::Arc;
use core::future::poll_fn;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use core::task::{Poll, Waker};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{mpsc, Mutex};
use std::time::Duration;

const DEADLINE: Duration = Duration::from_secs(5);

struct DropSignal(mpsc::Sender<()>);

impl Drop for DropSignal {
    fn drop(&mut self) {
        let _ = self.0.send(());
    }
}

// Exercise the actual spawn API and scheduler queues without OS workers: each
// selection and quantum boundary is controlled by the test, not timing.
fn manual_pool(workers: usize) -> TaskPool {
    TaskPool {
        shared: Arc::new(Shared::new(workers)),
        handles: Arc::new(Vec::new()),
        single: workers == 0,
    }
}

fn pop(pool: &TaskPool) -> Option<Job> {
    if pool.is_single_threaded() {
        pool.shared_state()
            .pop_async_ready()
            .map(|task| Box::new(move || task.run_boxed()) as Job)
    } else {
        pool.shared_state().find_task(Some(0))
    }
}

fn marker(
    pool: &TaskPool,
    priority: Arc<PriorityCell>,
    order: &Arc<Mutex<Vec<usize>>>,
    value: usize,
) -> Task<usize> {
    let order = Arc::clone(order);
    pool.spawn_future_with_priority(priority, async move {
        order.lock().unwrap().push(value);
        value
    })
}

#[test]
fn queued_late_boost_is_live_and_equal_priority_is_fifo() {
    for workers in [0, 1, 4] {
        let pool = manual_pool(workers);
        let order = Arc::new(Mutex::new(Vec::new()));
        let group = pool.priority_group(Priority::Low);
        let inherited = group.priority_cell();
        assert!(Arc::ptr_eq(&inherited, &group.priority_cell()));
        let high = Arc::new(PriorityCell::new(Priority::High));
        let ordinary = pool.spawn_future({
            let order = Arc::clone(&order);
            async move { order.lock().unwrap().push(9) }
        });
        let tasks = [
            marker(&pool, Arc::clone(&high), &order, 1),
            marker(&pool, Arc::clone(&inherited), &order, 2),
            marker(&pool, high, &order, 3),
            marker(&pool, inherited, &order, 4),
        ];
        assert!(tasks.iter().all(|task| !task.is_finished()));
        // Sharing inheritance does not silently join the group's counter.
        assert!(group.is_complete());
        group.boost_to(Priority::Critical);
        while let Some(dispatch) = pop(&pool) {
            dispatch();
        }
        assert_eq!(*order.lock().unwrap(), [2, 4, 1, 3, 9]);
        for (task, value) in tasks.into_iter().zip(1..=4) {
            assert!(task.counter().is_complete());
            assert_eq!(pool.block_on(task), value);
        }
        pool.block_on(ordinary);
    }
}

#[test]
fn background_explicit_queue_precedes_ordinary_fifo_not_implicit_normal() {
    for workers in [0, 1, 4] {
        let pool = TaskPool::with_threads(workers);
        let mut gate = WorkerGate::new(&pool);
        let order = Arc::new(Mutex::new(Vec::new()));
        let ordinary = pool.spawn_future({
            let order = Arc::clone(&order);
            async move { order.lock().unwrap().push(8) }
        });
        let legacy_ordinary = pool.spawn_async({
            let order = Arc::clone(&order);
            async move { order.lock().unwrap().push(9) }
        });
        let cell = Arc::new(PriorityCell::new(Priority::Background));
        let first = marker(&pool, Arc::clone(&cell), &order, 1);
        let second = marker(&pool, cell, &order, 2);
        let normal = marker(
            &pool,
            Arc::new(PriorityCell::new(Priority::Normal)),
            &order,
            0,
        );
        // Ordinary futures were queued first. They are not implicitly Normal:
        // the separate explicit-priority lane wins even at Background.
        gate.release_one();
        pool.block_on(async {
            ordinary.await;
            legacy_ordinary.await;
            assert_eq!(normal.await, 0);
            assert_eq!(first.await, 1);
            assert_eq!(second.await, 2);
        });
        assert_eq!(*order.lock().unwrap(), [0, 1, 2, 8, 9]);
    }
}

#[test]
fn quantum_continuation_and_idle_wake_reuse_the_live_cell() {
    for workers in [0, 1, 4] {
        let pool = manual_pool(workers);
        let priority = Arc::new(PriorityCell::new(Priority::Low));
        let polls = Arc::new(AtomicUsize::new(0));
        let order = Arc::new(Mutex::new(Vec::new()));
        let saved = Arc::new(Mutex::new(None::<Waker>));
        let task = pool.spawn_future_with_priority(Arc::clone(&priority), {
            let polls = Arc::clone(&polls);
            let saved = Arc::clone(&saved);
            let order = Arc::clone(&order);
            poll_fn(move |cx| {
                *saved.lock().unwrap() = Some(cx.waker().clone());
                let count = polls.fetch_add(1, Ordering::SeqCst) + 1;
                if count <= POLL_QUANTUM {
                    cx.waker().wake_by_ref();
                    Poll::Pending
                } else if count == POLL_QUANTUM + 1 {
                    order.lock().unwrap().push(1);
                    Poll::Pending // Must remain idle until an external wake.
                } else {
                    order.lock().unwrap().push(2);
                    Poll::Ready(42)
                }
            })
        });
        pop(&pool).unwrap()();
        assert_eq!(polls.load(Ordering::SeqCst), POLL_QUANTUM);
        let normal = marker(
            &pool,
            Arc::new(PriorityCell::new(Priority::Normal)),
            &order,
            3,
        );
        // The continuation was queued at Low before this boost.
        priority.boost_to(Priority::High);
        pop(&pool).unwrap()();
        assert_eq!(*order.lock().unwrap(), [1]);
        assert_eq!(polls.load(Ordering::SeqCst), POLL_QUANTUM + 1);
        let waker = saved.lock().unwrap().take().unwrap();
        for _ in 0..8 {
            waker.wake_by_ref();
        }
        let critical = marker(
            &pool,
            Arc::new(PriorityCell::new(Priority::Critical)),
            &order,
            4,
        );
        // The wake dispatch was queued at High. The same live cell must now
        // tie Critical and win FIFO, rather than retaining a priority snapshot.
        priority.boost_to(Priority::Critical);
        pop(&pool).unwrap()();
        assert_eq!(*order.lock().unwrap(), [1, 2]);
        while let Some(dispatch) = pop(&pool) {
            dispatch();
        }
        assert_eq!(*order.lock().unwrap(), [1, 2, 4, 3]);
        waker.wake_by_ref();
        assert!(pop(&pool).is_none());
        assert_eq!(polls.load(Ordering::SeqCst), POLL_QUANTUM + 2);
        assert_eq!(pool.block_on(task), 42);
        assert_eq!(pool.block_on(normal), 3);
        assert_eq!(pool.block_on(critical), 4);
    }
}

#[test]
fn zero_worker_priority_enqueue_and_external_wake_notify_the_driver() {
    let pool = TaskPool::with_threads(0);
    let shared = pool.shared_state();
    let epoch = shared.async_epoch();
    let (waker_tx, waker_rx) = mpsc::channel();
    let mut first = true;
    let task = pool.spawn_future_with_priority(
        Arc::new(PriorityCell::new(Priority::Background)),
        poll_fn(move |cx| {
            if first {
                first = false;
                waker_tx.send(cx.waker().clone()).unwrap();
                Poll::Pending
            } else {
                Poll::Ready(42)
            }
        }),
    );
    assert!(!shared.async_ready_is_empty());
    assert!(shared.wait_async_ready(epoch, Duration::ZERO));
    pop(&pool).unwrap()();
    assert!(shared.async_ready_is_empty());
    let epoch = shared.async_epoch();
    let waker = waker_rx.recv_timeout(DEADLINE).unwrap();
    waker.wake();
    assert!(shared.wait_async_ready(epoch, Duration::ZERO));
    assert_eq!(pool.block_on(task), 42);
    assert!(shared.async_ready_is_empty());
}

// Hold all real workers, then release just one. Even on a multi-worker pool
// this makes poll order deterministic, without caller-side help masking bugs.
struct WorkerGate(Vec<mpsc::Sender<()>>);

impl WorkerGate {
    fn new(pool: &TaskPool) -> Self {
        let mut gate = Self(Vec::new());
        let (entered_tx, entered_rx) = mpsc::channel();
        for _ in 0..pool.worker_count() {
            let (release_tx, release_rx) = mpsc::channel();
            gate.0.push(release_tx);
            let entered_tx = entered_tx.clone();
            pool.spawn(&crate::Counter::new(), move || {
                entered_tx.send(()).unwrap();
                release_rx.recv_timeout(DEADLINE).unwrap();
            });
        }
        for _ in 0..pool.worker_count() {
            entered_rx.recv_timeout(DEADLINE).unwrap();
        }
        gate
    }

    fn release_one(&mut self) {
        if let Some(release) = self.0.pop() {
            release.send(()).unwrap();
        }
    }
}

impl Drop for WorkerGate {
    fn drop(&mut self) {
        for release in self.0.drain(..) {
            let _ = release.send(());
        }
    }
}

#[test]
fn real_pools_apply_late_boost_across_future_and_group_jobs() {
    for workers in [0, 1, 4] {
        let pool = TaskPool::with_threads(workers);
        let mut gate = WorkerGate::new(&pool);
        let order = Arc::new(Mutex::new(Vec::new()));
        let group = pool.priority_group(Priority::Low);
        let high = Arc::new(PriorityCell::new(Priority::High));
        let first = marker(&pool, high, &order, 1);
        let inherited = marker(&pool, group.priority_cell(), &order, 2);
        if !pool.is_single_threaded() {
            let order = Arc::clone(&order);
            pool.spawn_in_group(&group, move || order.lock().unwrap().push(3));
        }
        group.boost_to(Priority::Critical);
        gate.release_one();
        pool.block_on(async {
            assert_eq!(first.await, 1);
            assert_eq!(inherited.await, 2);
            group.counter().wait_async().await;
        });
        assert_eq!(
            *order.lock().unwrap(),
            if pool.is_single_threaded() {
                vec![2, 1]
            } else {
                vec![2, 3, 1]
            }
        );
    }
}

#[test]
fn real_pools_complete_after_quantum_then_external_wake() {
    for workers in [0, 1, 4] {
        let pool = TaskPool::with_threads(workers);
        let priority = Arc::new(PriorityCell::new(Priority::Low));
        let polls = Arc::new(AtomicUsize::new(0));
        let (idle_tx, idle_rx) = mpsc::channel();
        let task = pool.spawn_future_with_priority(Arc::clone(&priority), {
            let polls = Arc::clone(&polls);
            poll_fn(move |cx| {
                let count = polls.fetch_add(1, Ordering::SeqCst) + 1;
                if count <= POLL_QUANTUM {
                    cx.waker().wake_by_ref();
                    Poll::Pending
                } else if count == POLL_QUANTUM + 1 {
                    idle_tx.send(cx.waker().clone()).unwrap();
                    Poll::Pending
                } else {
                    Poll::Ready(42)
                }
            })
        });
        let (done_tx, done_rx) = mpsc::channel();
        let driver = std::thread::spawn(move || done_tx.send(pool.block_on(task)).unwrap());
        let waker = idle_rx.recv_timeout(DEADLINE).unwrap();
        assert_eq!(polls.load(Ordering::SeqCst), POLL_QUANTUM + 1);
        priority.boost_to(Priority::Critical);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let waker = waker.clone();
                scope.spawn(move || {
                    for _ in 0..32 {
                        waker.wake_by_ref();
                    }
                });
            }
        });
        assert_eq!(done_rx.recv_timeout(DEADLINE).unwrap(), 42);
        driver.join().unwrap();
        assert_eq!(polls.load(Ordering::SeqCst), POLL_QUANTUM + 2);
    }
}

#[test]
fn prioritized_panics_after_yield_finish_counters_and_ignore_stale_wakes() {
    for workers in [0, 1, 4] {
        let pool = TaskPool::with_threads(workers);
        let polls = Arc::new(AtomicUsize::new(0));
        let saved = Arc::new(Mutex::new(None::<Waker>));
        let (dropped_tx, dropped_rx) = mpsc::channel();
        let task =
            pool.spawn_future_with_priority(Arc::new(PriorityCell::new(Priority::Critical)), {
                let polls = Arc::clone(&polls);
                let saved = Arc::clone(&saved);
                let drop_signal = DropSignal(dropped_tx);
                poll_fn(move |cx| -> Poll<()> {
                    let _keep_alive = &drop_signal;
                    *saved.lock().unwrap() = Some(cx.waker().clone());
                    let count = polls.fetch_add(1, Ordering::SeqCst) + 1;
                    cx.waker().wake_by_ref();
                    assert!(count <= POLL_QUANTUM, "prioritized panic payload");
                    Poll::Pending
                })
            });
        let counter = task.counter();
        let panic = catch_unwind(AssertUnwindSafe(|| pool.block_on(task))).unwrap_err();
        assert_eq!(
            panic.downcast_ref::<&str>(),
            Some(&"prioritized panic payload")
        );
        dropped_rx.recv_timeout(DEADLINE).unwrap();
        assert!(counter.is_complete());
        saved.lock().unwrap().take().unwrap().wake();
        let next = pool
            .spawn_future_with_priority(Arc::new(PriorityCell::new(Priority::Low)), async { 42 });
        assert_eq!(pool.block_on(next), 42);
        assert_eq!(polls.load(Ordering::SeqCst), POLL_QUANTUM + 1);
    }
}

#[test]
fn equal_priority_self_wake_yields_to_its_queued_peer() {
    for workers in [0, 1, 4] {
        let pool = manual_pool(workers);
        let cell = Arc::new(PriorityCell::new(Priority::Normal));
        let polls = Arc::new(AtomicUsize::new(0));
        let background = pool.spawn_future_with_priority(Arc::clone(&cell), {
            let polls = Arc::clone(&polls);
            poll_fn(move |cx| {
                let count = polls.fetch_add(1, Ordering::SeqCst) + 1;
                if count > POLL_QUANTUM {
                    Poll::Ready(())
                } else {
                    cx.waker().wake_by_ref();
                    Poll::Pending
                }
            })
        });
        let peer = pool.spawn_future_with_priority(cell, {
            let polls = Arc::clone(&polls);
            async move { polls.load(Ordering::SeqCst) }
        });
        pop(&pool).unwrap()();
        assert!(!background.is_finished());
        pop(&pool).unwrap()();
        assert_eq!(pool.block_on(peer), POLL_QUANTUM);
        while let Some(dispatch) = pop(&pool) {
            dispatch();
        }
        pool.block_on(background);
    }
}

#[test]
fn zero_worker_prioritized_self_wake_still_yields_to_direct_root() {
    let pool = TaskPool::with_threads(0);
    let polls = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let batch_bound = POLL_QUANTUM * super::SINGLE_READY_BATCH;
    let background =
        pool.spawn_future_with_priority(Arc::new(PriorityCell::new(Priority::Critical)), {
            let polls = Arc::clone(&polls);
            let stop = Arc::clone(&stop);
            poll_fn(move |cx| {
                let count = polls.fetch_add(1, Ordering::SeqCst) + 1;
                if stop.load(Ordering::SeqCst) || count >= batch_bound * 4 {
                    Poll::Ready(())
                } else {
                    cx.waker().wake_by_ref();
                    Poll::Pending
                }
            })
        });
    let ordinary = pool.spawn_future(async { 42 });
    pool.block_on(async {
        assert_eq!(polls.load(Ordering::SeqCst), batch_bound);
        assert!(!ordinary.is_finished()); // No cross-priority fairness promise.
        stop.store(true, Ordering::SeqCst);
    });
    pool.block_on(background);
    assert_eq!(pool.block_on(ordinary), 42);
    assert_eq!(polls.load(Ordering::SeqCst), batch_bound + 1);
}
