use crate::TaskPool;
use alloc::sync::Arc;
use core::future::poll_fn;
use core::sync::atomic::{AtomicBool, Ordering};
use core::task::{Poll, Waker};
use std::sync::mpsc;
use std::time::Duration;

const DEADLINE: Duration = Duration::from_secs(2);

fn external_completion(spawned: bool, drivers: usize) {
    let pool = TaskPool::with_threads(0);
    let mut joins = Vec::new();
    let mut signals = Vec::new();
    let (done_tx, done_rx) = mpsc::channel();
    for index in 0..drivers {
        let driver_pool = pool.clone();
        let ready = Arc::new(AtomicBool::new(false));
        let future_ready = ready.clone();
        let (parked_tx, parked_rx) = mpsc::channel::<Waker>();
        let done_tx = done_tx.clone();
        joins.push(std::thread::spawn(move || {
            let future = poll_fn(move |cx| {
                // Register before checking the producer's release so both
                // wake-during-poll and wake-after-park are valid executions.
                parked_tx.send(cx.waker().clone()).unwrap();
                if future_ready.load(Ordering::Acquire) {
                    Poll::Ready(index)
                } else {
                    Poll::Pending
                }
            });
            let result = if spawned {
                driver_pool.block_on(driver_pool.spawn_async(future))
            } else {
                driver_pool.block_on(future)
            };
            done_tx.send(result).unwrap();
        }));
        signals.push((ready, parked_rx));
    }
    let wakers: Vec<_> = signals
        .iter()
        .map(|(_, parked)| parked.recv_timeout(DEADLINE).unwrap())
        .collect();
    for ((ready, _), waker) in signals.iter().zip(wakers) {
        ready.store(true, Ordering::Release);
        waker.wake();
    }
    let mut results = Vec::new();
    for _ in 0..drivers {
        results.push(
            done_rx
                .recv_timeout(DEADLINE)
                .expect("external wake must not wait for the five-second stall timer"),
        );
    }
    results.sort_unstable();
    assert_eq!(results, (0..drivers).collect::<Vec<_>>());
    for join in joins {
        join.join().unwrap();
    }
}

#[test]
fn externally_woken_harness_notifies_a_zero_worker_driver_promptly() {
    external_completion(true, 1);
}

#[test]
fn directly_polled_external_future_notifies_a_zero_worker_driver_promptly() {
    external_completion(false, 1);
}

#[test]
fn concurrent_zero_worker_drivers_do_not_consume_each_others_notifications() {
    external_completion(true, 2);
    external_completion(false, 2);
}

#[test]
fn a_wake_before_wait_is_observed_without_parking() {
    let shared = crate::scheduler::Shared::new(0);
    let epoch = shared.async_epoch();
    shared.signal_async_driver();
    assert!(shared.wait_async_ready(epoch, Duration::ZERO));
    assert!(!shared.wait_async_ready(shared.async_epoch(), Duration::ZERO));
}
