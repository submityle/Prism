//! Tests for the §24.2 read-copy-update cell [`Rcu`](super::rcu::Rcu).
//!
//! Single-threaded correctness and snapshot semantics plus multi-threaded
//! stress asserting that: readers always observe a consistent, well-formed
//! snapshot (never a torn or freed value), concurrent updates are serialised so
//! no writer's publish is lost, and every superseded version is reclaimed
//! exactly once (no leak, no double free, never while a reader is pinned).
//!
//! Correctness is checked against a simple sequential oracle: a value's shape
//! is always internally consistent, and after a known sequence of stores the
//! final published value is exactly the oracle's.

extern crate alloc;

use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use super::epoch::{pin_with, Collector};
use super::rcu::Rcu;

// --- single-threaded correctness -------------------------------------------

#[test]
fn new_then_read_returns_initial_value() {
    let rcu = Rcu::new(42_u32);
    assert_eq!(*rcu.read(), 42);
    assert_eq!(*rcu.read().get(), 42);
}

#[test]
fn store_replaces_published_value() {
    let rcu = Rcu::new(1_u32);
    rcu.store(2);
    assert_eq!(*rcu.read(), 2);
    rcu.store(3);
    rcu.store(4);
    assert_eq!(*rcu.read(), 4);
}

#[test]
fn update_derives_from_current_snapshot() {
    let rcu = Rcu::new(10_u64);
    rcu.update(|&x| x + 5);
    assert_eq!(*rcu.read(), 15);
    rcu.update(|&x| x * 2);
    assert_eq!(*rcu.read(), 30);
}

#[test]
fn load_cloned_matches_read() {
    let rcu = Rcu::new(vec![1_u32, 2, 3]);
    let snap = rcu.load_cloned();
    assert_eq!(snap, vec![1, 2, 3]);
    rcu.store(vec![9]);
    assert_eq!(rcu.load_cloned(), vec![9]);
}

#[test]
fn held_guard_keeps_its_snapshot_across_a_store() {
    // RCU semantics: an outstanding reader keeps observing the version it took,
    // even after a writer publishes a newer one.
    let rcu = Rcu::new(vec![1_u32, 2, 3]);
    let guard = rcu.read();
    assert_eq!(*guard, vec![1, 2, 3]);
    rcu.store(vec![100, 200]);
    // Old guard unchanged; a fresh read sees the new value.
    assert_eq!(*guard, vec![1, 2, 3]);
    assert_eq!(*rcu.read(), vec![100, 200]);
    drop(guard);
    assert_eq!(*rcu.read(), vec![100, 200]);
}

#[test]
fn default_and_debug() {
    let rcu: Rcu<u32> = Rcu::default();
    assert_eq!(*rcu.read(), 0);
    let dbg = alloc::format!("{rcu:?}");
    assert!(dbg.contains("Rcu"), "unexpected debug output: {dbg}");
    let g = rcu.read();
    let gdbg = alloc::format!("{g:?}");
    assert!(gdbg.contains("RcuGuard"), "unexpected guard debug: {gdbg}");
}

#[test]
fn empty_and_single_element_payloads() {
    // Boundary: zero-length and one-length collections round-trip cleanly.
    let rcu = Rcu::new(Vec::<u8>::new());
    assert!(rcu.read().is_empty());
    rcu.store(vec![7]);
    assert_eq!(*rcu.read(), vec![7]);
    rcu.update(|v| {
        let mut next = v.clone();
        next.clear();
        next
    });
    assert!(rcu.read().is_empty());
}

// --- drop accounting --------------------------------------------------------

/// A payload that bumps a shared counter when dropped, so tests can detect a
/// leaked or double-dropped version.
#[derive(Clone, Debug)]
struct DropProbe {
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

/// Forces the epoch reclaimer in `collector`'s domain to drain all pending
/// deferred frees. Epoch reclamation is amortised (it only advances when all
/// pinned participants have caught up), so a test that wants to assert exact
/// reclamation must pump it to quiescence. Each pump pins afresh, so the epoch
/// can advance one step and the bucket that became safe is run.
fn drain(collector: &Collector) {
    for _ in 0..16 {
        let g = pin_with(collector);
        g.flush();
        drop(g);
    }
}

#[test]
fn every_superseded_version_is_reclaimed_once() {
    let dropped = Arc::new(AtomicUsize::new(0));
    // Each `store` creates one probe and supersedes the previous one. With N
    // stores on top of the initial value there are N + 1 distinct versions.
    const N: usize = 500;
    let collector;
    {
        let rcu = Rcu::new(DropProbe::new(0, &dropped));
        collector = rcu.collector();
        for id in 1..=N {
            rcu.store(DropProbe::new(id, &dropped));
        }
        // The current version is still live, so at most the N superseded ones
        // can have been freed — never more (no double free).
        assert!(
            dropped.load(Ordering::SeqCst) <= N,
            "a version was freed more than once"
        );
        assert_eq!(rcu.read().id, N);
    }
    // The cell's drop freed the current version directly; drain the reclaimer
    // so every deferred free of a superseded version also runs. Then every one
    // of the N + 1 versions is freed exactly once (no leak, no double free).
    drain(&collector);
    assert_eq!(
        dropped.load(Ordering::SeqCst),
        N + 1,
        "exactly one drop per version expected"
    );
}

// --- multi-threaded stress --------------------------------------------------

#[test]
fn concurrent_readers_never_observe_a_torn_or_freed_value() {
    // Invariant published by every writer: a vector of `len` copies of a
    // monotonically increasing stamp. A reader that ever sees a vector whose
    // elements disagree, or whose stamp went backwards within a single
    // snapshot, caught a torn / recycled value.
    const WRITES: u32 = 20_000;
    let rcu = Arc::new(Rcu::new(vec![0_u32; 8]));

    thread::scope(|s| {
        // Writer: publish ever-higher stamps via copy-update.
        {
            let rcu = Arc::clone(&rcu);
            s.spawn(move || {
                for stamp in 1..=WRITES {
                    rcu.update(|cur| vec![stamp; cur.len()]);
                }
            });
        }
        // Several readers hammer the read path and assert each snapshot is
        // internally consistent.
        for _ in 0..4 {
            let rcu = Arc::clone(&rcu);
            s.spawn(move || {
                let mut last_seen = 0_u32;
                for _ in 0..200_000 {
                    let snap = rcu.read();
                    assert!(!snap.is_empty());
                    let first = snap[0];
                    for &x in snap.iter() {
                        assert_eq!(x, first, "torn snapshot: elements disagree");
                    }
                    assert!(first >= last_seen, "stamp went backwards: reused memory?");
                    last_seen = first;
                }
            });
        }
    });

    assert_eq!(*rcu.read(), vec![WRITES; 8]);
}

#[test]
fn concurrent_updates_are_serialised_no_lost_increment() {
    // Many threads each do `PER` read-copy-update increments; the final value
    // must equal the total number of increments (CAS retry loop loses none).
    const THREADS: usize = 6;
    const PER: u64 = 5_000;
    let rcu = Arc::new(Rcu::new(0_u64));

    thread::scope(|s| {
        for _ in 0..THREADS {
            let rcu = Arc::clone(&rcu);
            s.spawn(move || {
                for _ in 0..PER {
                    rcu.update(|&x| x + 1);
                }
            });
        }
    });

    assert_eq!(*rcu.read(), THREADS as u64 * PER);
}

#[test]
fn shared_collector_reclaims_across_cells() {
    // Two cells sharing one reclamation domain both exercise defer/collect
    // without leaking or double-freeing.
    let dropped = Arc::new(AtomicUsize::new(0));
    let collector = Collector::new();
    {
        let a = Rcu::with_collector(DropProbe::new(0, &dropped), collector.clone());
        let b = Rcu::with_collector(DropProbe::new(1, &dropped), collector.clone());
        for id in 2..200 {
            a.store(DropProbe::new(id, &dropped));
            b.store(DropProbe::new(id, &dropped));
        }
    }
    // Drain the shared domain, then: 2 initial + (198 * 2) stored = 398
    // versions, each reclaimed exactly once.
    drain(&collector);
    assert_eq!(dropped.load(Ordering::SeqCst), 2 + 198 * 2);
}
