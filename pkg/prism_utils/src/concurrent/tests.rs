//! Tests for the M5 concurrent containers: single-threaded correctness plus
//! multi-threaded stress asserting that no item is ever lost or duplicated, no
//! payload is leaked or double-dropped, and epoch reclamation runs each
//! deferred free exactly once and never while a reader is pinned.

extern crate alloc;

use alloc::collections::BTreeSet;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use super::epoch::{pin_with, Collector};
use super::{CachePadded, ConcurrentHashMap, MpmcQueue, SpscQueue, TreiberStack};

// --- CachePadded -----------------------------------------------------------

#[test]
fn cache_padded_aligns_and_derefs() {
    assert_eq!(align_of::<CachePadded<u8>>(), 64);
    let mut p = CachePadded::new(7_u32);
    assert_eq!(*p, 7);
    *p += 1;
    assert_eq!(p.into_inner(), 8);
}

// --- SPSC ------------------------------------------------------------------

#[test]
fn spsc_single_thread_fifo() {
    let (tx, rx) = SpscQueue::with_capacity(4);
    assert!(rx.is_empty());
    for i in 0..4 {
        assert_eq!(tx.push(i), Ok(()));
    }
    assert!(tx.is_full());
    assert_eq!(tx.push(99), Err(99)); // full
    assert_eq!(rx.len(), 4);
    for i in 0..4 {
        assert_eq!(rx.pop(), Some(i));
    }
    assert_eq!(rx.pop(), None);
}

#[test]
fn spsc_wraps_around_many_times() {
    let (tx, rx) = SpscQueue::with_capacity(3);
    // Push/pop far past capacity to exercise index wrap-around.
    for i in 0..1000 {
        assert_eq!(tx.push(i), Ok(()));
        assert_eq!(rx.pop(), Some(i));
    }
    assert!(rx.is_empty());
}

#[test]
fn spsc_concurrent_stream_preserves_order() {
    const N: usize = 200_000;
    let (tx, rx) = SpscQueue::<usize>::with_capacity(1024);

    let producer = thread::spawn(move || {
        let mut i = 0;
        while i < N {
            if tx.push(i).is_ok() {
                i += 1;
            } else {
                std::hint::spin_loop();
            }
        }
    });

    let consumer = thread::spawn(move || {
        let mut expected = 0;
        while expected < N {
            match rx.pop() {
                // SPSC is strictly FIFO: values must arrive in send order.
                Some(v) => {
                    assert_eq!(v, expected, "SPSC reordered or dropped an item");
                    expected += 1;
                }
                None => std::hint::spin_loop(),
            }
        }
    });

    producer.join().unwrap();
    consumer.join().unwrap();
}

#[test]
fn spsc_drops_remaining_elements() {
    let dropped = Arc::new(AtomicUsize::new(0));
    {
        let (tx, _rx) = SpscQueue::with_capacity(8);
        for _ in 0..5 {
            tx.push(DropProbe::new(0, &dropped)).unwrap();
        }
        // Drop the queue with 5 live elements still inside.
    }
    assert_eq!(dropped.load(Ordering::SeqCst), 5);
}

// --- MPMC ------------------------------------------------------------------

#[test]
fn mpmc_single_thread_fifo_and_bounds() {
    let q = MpmcQueue::with_capacity(3); // rounds up to 4
    assert_eq!(q.capacity(), 4);
    assert!(q.is_empty());
    for i in 0..4 {
        assert_eq!(q.push(i), Ok(()));
    }
    assert_eq!(q.push(99), Err(99)); // full
    for i in 0..4 {
        assert_eq!(q.pop(), Some(i));
    }
    assert_eq!(q.pop(), None);
}

#[test]
fn mpmc_multi_producer_multi_consumer_no_loss_no_dup() {
    const PRODUCERS: usize = 4;
    const CONSUMERS: usize = 4;
    const PER: usize = 25_000;
    const TOTAL: usize = PRODUCERS * PER;

    let q = MpmcQueue::<usize>::with_capacity(1024);
    let produced = Arc::new(AtomicUsize::new(0));
    let consumed = Arc::new(AtomicUsize::new(0));

    thread::scope(|scope| {
        for p in 0..PRODUCERS {
            let q = q.clone();
            scope.spawn(move || {
                for i in 0..PER {
                    // Unique id per (producer, index) so duplicates are visible.
                    let id = p * PER + i;
                    while q.push(id).is_err() {
                        std::hint::spin_loop();
                    }
                }
            });
        }

        let mut handles = Vec::new();
        for _ in 0..CONSUMERS {
            let q = q.clone();
            let consumed = Arc::clone(&consumed);
            handles.push(scope.spawn(move || {
                let mut seen = Vec::new();
                loop {
                    if let Some(v) = q.pop() {
                        seen.push(v);
                        consumed.fetch_add(1, Ordering::Relaxed);
                    } else if consumed.load(Ordering::Relaxed) >= TOTAL {
                        break;
                    } else {
                        std::hint::spin_loop();
                    }
                }
                seen
            }));
        }

        // Mark everything produced (consumers already racing).
        produced.store(TOTAL, Ordering::Relaxed);

        let mut all = BTreeSet::new();
        for h in handles {
            for v in h.join().unwrap() {
                assert!(all.insert(v), "MPMC yielded a duplicate: {v}");
            }
        }
        assert_eq!(all.len(), TOTAL, "MPMC lost items");
        assert_eq!(*all.iter().next().unwrap(), 0);
        assert_eq!(*all.iter().next_back().unwrap(), TOTAL - 1);
    });
}

#[test]
fn mpmc_drops_remaining_elements() {
    let dropped = Arc::new(AtomicUsize::new(0));
    {
        let q = MpmcQueue::with_capacity(8);
        for _ in 0..6 {
            q.push(DropProbe::new(0, &dropped)).unwrap();
        }
    }
    assert_eq!(dropped.load(Ordering::SeqCst), 6);
}

// --- ConcurrentHashMap -----------------------------------------------------

#[test]
fn concurrent_map_single_thread_basics() {
    let m: ConcurrentHashMap<u32, u32> = ConcurrentHashMap::new();
    assert!(m.is_empty());
    assert_eq!(m.insert(1, 10), None);
    assert_eq!(m.insert(1, 11), Some(10));
    assert_eq!(m.get(&1), Some(11));
    assert!(m.contains_key(&1));
    assert!(m.with(&1, |v| v == Some(&11)));
    m.with_mut(&1, |v| {
        if let Some(v) = v {
            *v += 100;
        }
    });
    assert_eq!(m.get(&1), Some(111));
    assert_eq!(m.get_or_insert_with(2, || 20), 20);
    assert_eq!(m.get_or_insert_with(2, || 999), 20);
    assert_eq!(m.len(), 2);
    assert_eq!(m.remove(&1), Some(111));
    assert_eq!(m.remove(&1), None);
    assert_eq!(m.len(), 1);
    m.clear();
    assert!(m.is_empty());
}

#[test]
fn concurrent_map_for_each_visits_all() {
    let m: ConcurrentHashMap<u32, u32> = ConcurrentHashMap::with_shards(8);
    for i in 0..100 {
        m.insert(i, i * 2);
    }
    let mut sum = 0_u64;
    let mut count = 0_u64;
    m.for_each(|_, v| {
        sum += u64::from(*v);
        count += 1;
    });
    assert_eq!(count, 100);
    assert_eq!(sum, (0u32..100).map(|i| u64::from(i * 2)).sum());
}

#[test]
fn concurrent_map_parallel_disjoint_inserts() {
    const THREADS: usize = 8;
    const PER: usize = 10_000;
    let m: Arc<ConcurrentHashMap<usize, usize>> = Arc::new(ConcurrentHashMap::with_shards(16));

    thread::scope(|scope| {
        for t in 0..THREADS {
            let m = Arc::clone(&m);
            scope.spawn(move || {
                for i in 0..PER {
                    let k = t * PER + i;
                    assert_eq!(m.insert(k, k * 2), None);
                }
                // Each thread reads its own keys back.
                for i in 0..PER {
                    let k = t * PER + i;
                    assert_eq!(m.get(&k), Some(k * 2));
                }
            });
        }
    });

    assert_eq!(m.len(), THREADS * PER);
    for k in 0..THREADS * PER {
        assert_eq!(m.get(&k), Some(k * 2));
    }
}

#[test]
fn concurrent_map_contended_counter() {
    // Many threads increment the same small set of keys; the final totals must
    // be exact (no lost updates), proving the write path is correctly locked.
    const THREADS: usize = 8;
    const ITERS: usize = 20_000;
    const KEYS: usize = 4;
    let m: Arc<ConcurrentHashMap<usize, usize>> = Arc::new(ConcurrentHashMap::new());
    for k in 0..KEYS {
        m.insert(k, 0);
    }

    thread::scope(|scope| {
        for _ in 0..THREADS {
            let m = Arc::clone(&m);
            scope.spawn(move || {
                for i in 0..ITERS {
                    m.with_mut(&(i % KEYS), |v| {
                        if let Some(v) = v {
                            *v += 1;
                        }
                    });
                }
            });
        }
    });

    let mut total = 0;
    for k in 0..KEYS {
        total += m.get(&k).unwrap();
    }
    assert_eq!(total, THREADS * ITERS);
}

// --- Epoch reclamation -----------------------------------------------------

#[test]
fn epoch_defers_until_safe_and_never_while_pinned() {
    let c = Collector::new();
    let h = c.register();
    let ran = Arc::new(AtomicUsize::new(0));

    let g = h.pin(); // pinned at epoch 0
    {
        let ran = Arc::clone(&ran);
        g.defer(move || {
            ran.fetch_add(1, Ordering::SeqCst);
        });
    }
    // Advances 0 -> 1 and collects epoch-(-1)'s (empty) bucket; our garbage was
    // retired in epoch 0 and must NOT run while we are still pinned there.
    g.flush();
    assert_eq!(ran.load(Ordering::SeqCst), 0, "ran a deferred while pinned");
    drop(g);

    let g2 = h.pin(); // pinned at epoch 1
                      // Advances 1 -> 2; epoch-0 garbage is now unobservable and must run exactly once.
    g2.flush();
    assert_eq!(ran.load(Ordering::SeqCst), 1);
    g2.flush();
    assert_eq!(ran.load(Ordering::SeqCst), 1, "deferred ran more than once");
    drop(g2);
}

#[test]
fn epoch_stress_runs_every_deferred_exactly_once() {
    const THREADS: usize = 8;
    const PER: usize = 4_000;
    const TOTAL: usize = THREADS * PER;

    let collector = Collector::new();
    let freed = Arc::new(AtomicUsize::new(0));

    thread::scope(|scope| {
        for _ in 0..THREADS {
            let collector = collector.clone();
            let freed = Arc::clone(&freed);
            scope.spawn(move || {
                for i in 0..PER {
                    let g = pin_with(&collector);
                    let freed = Arc::clone(&freed);
                    g.defer(move || {
                        freed.fetch_add(1, Ordering::SeqCst);
                    });
                    if i % 64 == 0 {
                        g.flush();
                    }
                }
            });
        }
    });

    // Workers have exited (their handles retired). Drive the epoch forward from
    // this thread until every deferred free has run.
    for _ in 0..200 {
        if freed.load(Ordering::SeqCst) == TOTAL {
            break;
        }
        let g = pin_with(&collector);
        g.flush();
        drop(g);
    }
    assert_eq!(
        freed.load(Ordering::SeqCst),
        TOTAL,
        "epoch dropped or double-ran a deferred free"
    );
}

// --- Treiber stack ---------------------------------------------------------

#[test]
fn treiber_single_thread_lifo() {
    let s = TreiberStack::new();
    assert!(s.is_empty());
    for i in 0..5 {
        s.push(i);
    }
    assert!(!s.is_empty());
    for i in (0..5).rev() {
        assert_eq!(s.pop(), Some(i));
    }
    assert_eq!(s.pop(), None);
    assert!(s.is_empty());
}

#[test]
fn treiber_drops_remaining_on_drop() {
    let dropped = Arc::new(AtomicUsize::new(0));
    {
        let s = TreiberStack::new();
        for i in 0..7 {
            s.push(DropProbe::new(i, &dropped));
        }
        // Drop stack with 7 live nodes; each payload must be dropped once.
    }
    assert_eq!(dropped.load(Ordering::SeqCst), 7);
}

#[test]
fn treiber_concurrent_push_pop_no_loss_no_dup() {
    const PRODUCERS: usize = 4;
    const CONSUMERS: usize = 4;
    const PER: usize = 20_000;
    const TOTAL: usize = PRODUCERS * PER;

    let stack: Arc<TreiberStack<usize>> = Arc::new(TreiberStack::new());
    let popped = Arc::new(AtomicUsize::new(0));

    thread::scope(|scope| {
        for p in 0..PRODUCERS {
            let stack = Arc::clone(&stack);
            scope.spawn(move || {
                for i in 0..PER {
                    stack.push(p * PER + i);
                }
            });
        }

        let mut handles = Vec::new();
        for _ in 0..CONSUMERS {
            let stack = Arc::clone(&stack);
            let popped = Arc::clone(&popped);
            handles.push(scope.spawn(move || {
                let mut seen = Vec::new();
                loop {
                    if let Some(v) = stack.pop() {
                        seen.push(v);
                        popped.fetch_add(1, Ordering::Relaxed);
                    } else if popped.load(Ordering::Relaxed) >= TOTAL {
                        break;
                    } else {
                        std::hint::spin_loop();
                    }
                }
                seen
            }));
        }

        let mut all = BTreeSet::new();
        for h in handles {
            for v in h.join().unwrap() {
                assert!(all.insert(v), "Treiber stack yielded a duplicate: {v}");
            }
        }
        assert_eq!(
            all.len(),
            TOTAL,
            "Treiber stack lost items (ABA/reclaim bug?)"
        );
    });

    assert!(stack.is_empty());
}

// --- shared test helper ----------------------------------------------------

/// A payload that bumps a shared counter when dropped, so tests can detect
/// leaked or double-dropped elements.
#[derive(Debug)]
struct DropProbe {
    #[expect(
        dead_code,
        reason = "identity is carried for debugging; tests assert on drops, not reads"
    )]
    id: usize,
    dropped: Arc<AtomicUsize>,
}

impl DropProbe {
    fn new(id: usize, dropped: &Arc<AtomicUsize>) -> Self {
        Self {
            id,
            dropped: Arc::clone(dropped),
        }
    }
}

impl Drop for DropProbe {
    fn drop(&mut self) {
        self.dropped.fetch_add(1, Ordering::SeqCst);
    }
}
