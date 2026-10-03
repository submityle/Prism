//! Unit tests for the M2 threading layer.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::thread::affinity::{self, AffinityError};
use crate::thread::park::Parker;
use crate::thread::spawn::{self, Builder};
use crate::thread::sync::{Backoff, Once, SpinLock};
use crate::thread::tls::ThreadLocal;

#[test]
fn spawn_join_returns_value() {
    let handle = Builder::new()
        .name("prism-worker")
        .stack_size(256 * 1024)
        .spawn(|| 20 + 22)
        .expect("spawn should succeed");
    assert_eq!(handle.join().expect("thread must not panic"), 42);

    // The free function should also round-trip a value.
    let h2 = spawn::spawn(|| "hello".to_string());
    assert_eq!(h2.join().unwrap(), "hello");
}

#[test]
fn hardware_concurrency_is_positive() {
    assert!(spawn::hardware_concurrency() >= 1);
}

#[test]
fn tls_isolates_values_across_threads() {
    let tls: Arc<ThreadLocal<u64>> = Arc::new(ThreadLocal::new());

    // The main thread owns value 100.
    tls.with(|| 100, |v| *v += 0);
    assert!(tls.is_set());

    let mut handles = Vec::new();
    for seed in [1_u64, 2, 3] {
        let tls = Arc::clone(&tls);
        handles.push(spawn::spawn(move || {
            // Each thread initializes its own slot and mutates only that slot.
            let first = tls.get_or(|| seed * 10);
            tls.with(|| seed * 10, |v| *v += 5);
            let second = tls.get_or(|| 0);
            (first, second)
        }));
    }
    for (i, h) in handles.into_iter().enumerate() {
        let seed = (i as u64) + 1;
        let (first, second) = h.join().unwrap();
        assert_eq!(first, seed * 10, "per-thread init must be independent");
        assert_eq!(second, seed * 10 + 5, "per-thread mutation stays local");
    }

    // The main thread's value is untouched by the workers.
    assert_eq!(tls.get_or(|| 0), 100);
    // One slot per thread that touched the store (main + 3 workers).
    assert_eq!(tls.len(), 4);
    assert!(tls.clear());
    assert!(!tls.is_set());
}

#[test]
fn spinlock_provides_mutual_exclusion_under_contention() {
    const THREADS: usize = 8;
    const ITERS: usize = 10_000;

    let lock = Arc::new(SpinLock::new(0usize));
    let mut handles = Vec::new();
    for _ in 0..THREADS {
        let lock = Arc::clone(&lock);
        handles.push(spawn::spawn(move || {
            for _ in 0..ITERS {
                let mut guard = lock.lock();
                // A non-atomic read-modify-write; only mutual exclusion keeps
                // this race-free.
                *guard += 1;
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    assert_eq!(*lock.lock(), THREADS * ITERS);
}

#[test]
fn spinlock_try_lock_reports_contention() {
    let lock = SpinLock::new(7);
    let guard = lock.lock();
    assert!(lock.try_lock().is_none(), "held lock must not be re-acquired");
    drop(guard);
    assert_eq!(*lock.try_lock().expect("lock is free now"), 7);
}

#[test]
fn once_runs_exactly_once() {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let once = Arc::new(Once::new());
    let mut handles = Vec::new();
    for _ in 0..16 {
        let once = Arc::clone(&once);
        handles.push(spawn::spawn(move || {
            once.call_once(|| {
                COUNTER.fetch_add(1, Ordering::SeqCst);
            });
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    assert_eq!(COUNTER.load(Ordering::SeqCst), 1);
    assert!(once.is_completed());
}

#[test]
fn backoff_completes_after_enough_steps() {
    let mut backoff = Backoff::new();
    for _ in 0..64 {
        backoff.snooze();
    }
    assert!(backoff.is_completed());
    backoff.reset();
    assert!(!backoff.is_completed());
}

#[test]
fn park_unpark_wakes_the_parked_thread() {
    let parker = Parker::new();
    let unparker = parker.unparker();

    // Wake from another thread after a short delay; the main thread blocks in
    // `park` until the token arrives.
    let waker = spawn::spawn(move || {
        spawn::sleep(Duration::from_millis(50));
        unparker.unpark();
    });

    let start = Instant::now();
    parker.park();
    assert!(
        start.elapsed() >= Duration::from_millis(40),
        "park should have blocked until unpark"
    );
    waker.join().unwrap();
}

#[test]
fn unpark_before_park_is_remembered() {
    let parker = Parker::new();
    let unparker = parker.unparker();
    // Token posted before parking: the subsequent park must return at once.
    unparker.unpark();
    let start = Instant::now();
    parker.park();
    assert!(
        start.elapsed() < Duration::from_millis(500),
        "a pre-posted token must make park return immediately"
    );
}

#[test]
fn park_timeout_reports_timeout_without_token() {
    let parker = Parker::new();
    let start = Instant::now();
    let got_token = parker.park_timeout(Duration::from_millis(30));
    assert!(!got_token, "no token was posted");
    assert!(start.elapsed() >= Duration::from_millis(20));
}

#[test]
fn affinity_set_is_ok_or_documented_unsupported() {
    match affinity::set_current_thread_affinity(0) {
        // Supported platforms (Linux/Windows) pin to core 0.
        Ok(()) => assert!(affinity::affinity_supported()),
        // Honest unsupported result (e.g. macOS) must not crash.
        Err(AffinityError::Unsupported) => assert!(!affinity::affinity_supported()),
        Err(other) => panic!("unexpected affinity error on this host: {other}"),
    }

    // An empty core set is always rejected as invalid.
    assert_eq!(
        affinity::set_current_thread_affinity_mask(&[]),
        Err(AffinityError::InvalidCore)
    );

    // On platforms that can pin, an out-of-range core is rejected as invalid.
    if affinity::affinity_supported() {
        assert_eq!(
            affinity::set_current_thread_affinity(usize::MAX),
            Err(AffinityError::InvalidCore)
        );
    }
}
