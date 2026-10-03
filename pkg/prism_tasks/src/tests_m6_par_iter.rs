//! M6 tests: ECS-style `par_iter` data-parallel iteration.
//!
//! Anti-vacuous contract: every parallel terminal must equal the serial
//! computation across many sizes (including empty and lengths that are not a
//! multiple of the chunk grain) and across several seeds, with a small
//! `with_min_len` so the work genuinely spreads across workers.

use crate::TaskPool;
use alloc::vec::Vec;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Tiny deterministic xorshift64 PRNG so tests seed data without a dependency.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// Sizes chosen to include the empty slice, single element, small lengths, and
/// several lengths that are not a multiple of the forced grain (`min_len`).
const SIZES: &[usize] = &[0, 1, 2, 3, 5, 7, 16, 17, 63, 64, 65, 100, 1000, 4097];

#[test]
fn par_iter_range_for_each_sums_equal_serial() {
    let pool = TaskPool::with_threads(4);
    for &n in SIZES {
        let serial: usize = (0..n).sum();
        let acc = AtomicUsize::new(0);
        pool.par_iter_range(0..n)
            .with_min_len(4)
            .for_each(|i| {
                acc.fetch_add(i, Ordering::Relaxed);
            });
        assert_eq!(acc.load(Ordering::Relaxed), serial, "range sum, n={n}");
    }
}

#[test]
fn par_iter_range_map_collect_is_ordered_and_equal_serial() {
    let pool = TaskPool::with_threads(4);
    for &n in SIZES {
        let serial: Vec<u64> = (0..n).map(|i| (i as u64) * 3 + 1).collect();
        let parallel = pool
            .par_iter_range(0..n)
            .with_min_len(4)
            .map_collect(|i| (i as u64) * 3 + 1);
        assert_eq!(parallel, serial, "range map_collect order, n={n}");
    }
}

#[test]
fn par_iter_slice_for_each_and_enumerate_equal_serial() {
    let pool = TaskPool::with_threads(4);
    for (seed, &n) in SIZES.iter().enumerate() {
        let mut rng = Rng::new(seed as u64 + 1);
        let data: Vec<u64> = (0..n).map(|_| rng.next_u64() % 1000).collect();

        // for_each over a read-only slice: sum must match serial.
        let serial_sum: u128 = data.iter().map(|&x| x as u128).sum();
        let acc = AtomicUsize::new(0);
        pool.par_iter(&data).with_min_len(4).for_each(|&x| {
            acc.fetch_add(x as usize, Ordering::Relaxed);
        });
        assert_eq!(acc.load(Ordering::Relaxed) as u128, serial_sum, "slice for_each, n={n}");

        // enumerate_for_each must observe the correct (index, value) pairs.
        let seen: Mutex<Vec<Option<u64>>> = Mutex::new((0..n).map(|_| None).collect());
        pool.par_iter(&data)
            .with_min_len(4)
            .enumerate_for_each(|i, &x| {
                seen.lock().unwrap()[i] = Some(x);
            });
        let seen = seen.into_inner().unwrap();
        let expect: Vec<Option<u64>> = data.iter().map(|&x| Some(x)).collect();
        assert_eq!(seen, expect, "slice enumerate_for_each, n={n}");
    }
}

#[test]
fn par_iter_slice_map_collect_is_ordered_and_equal_serial() {
    let pool = TaskPool::with_threads(4);
    for (seed, &n) in SIZES.iter().enumerate() {
        let mut rng = Rng::new(seed as u64 + 100);
        let data: Vec<u64> = (0..n).map(|_| rng.next_u64() % 1000).collect();
        let serial: Vec<u64> = data.iter().map(|&x| x.wrapping_mul(7).wrapping_add(5)).collect();
        let parallel = pool
            .par_iter(&data)
            .with_min_len(4)
            .map_collect(|&x| x.wrapping_mul(7).wrapping_add(5));
        assert_eq!(parallel, serial, "slice map_collect order, n={n}");
    }
}

#[test]
fn par_iter_mut_for_each_and_enumerate_equal_serial() {
    let pool = TaskPool::with_threads(4);
    for (seed, &n) in SIZES.iter().enumerate() {
        let mut rng = Rng::new(seed as u64 + 7);
        let base: Vec<u64> = (0..n).map(|_| rng.next_u64() % 1000).collect();

        // In-place for_each must match the serial transform element-for-element.
        let mut parallel = base.clone();
        pool.par_iter_mut(&mut parallel)
            .with_min_len(4)
            .for_each(|x| *x = x.wrapping_mul(3).wrapping_add(1));
        let serial: Vec<u64> = base.iter().map(|&x| x.wrapping_mul(3).wrapping_add(1)).collect();
        assert_eq!(parallel, serial, "mut for_each, n={n}");

        // enumerate_for_each writes index-dependent values.
        let mut parallel2 = base.clone();
        pool.par_iter_mut(&mut parallel2)
            .with_min_len(4)
            .enumerate_for_each(|i, x| *x = x.wrapping_add(i as u64));
        let serial2: Vec<u64> = base.iter().enumerate().map(|(i, &x)| x.wrapping_add(i as u64)).collect();
        assert_eq!(parallel2, serial2, "mut enumerate_for_each, n={n}");
    }
}

#[test]
fn par_iter_spreads_across_multiple_workers() {
    // With a tiny grain and many elements, several distinct workers must each
    // run at least one chunk — proving the work actually fans out.
    let pool = TaskPool::with_threads(4);
    if pool.worker_count() < 2 {
        return;
    }
    let n = 10_000usize;
    let buckets: Vec<AtomicUsize> = (0..pool.worker_count() + 1).map(|_| AtomicUsize::new(0)).collect();
    pool.par_iter_range(0..n).with_min_len(1).for_each(|_| {
        let w = pool.current_worker_index().unwrap_or(pool.worker_count());
        buckets[w].fetch_add(1, Ordering::Relaxed);
    });
    let total: usize = buckets.iter().map(|b| b.load(Ordering::Relaxed)).sum();
    assert_eq!(total, n, "every element visited exactly once");
    let active = buckets.iter().filter(|b| b.load(Ordering::Relaxed) > 0).count();
    assert!(active >= 2, "work must spread across >= 2 workers, saw {active}");
}

#[test]
fn par_iter_single_threaded_fallback_matches_serial() {
    let pool = TaskPool::with_threads(0);
    assert!(pool.is_single_threaded());
    let data: Vec<u64> = (0..1000).collect();
    let parallel = pool.par_iter(&data).map_collect(|&x| x * 2);
    let serial: Vec<u64> = data.iter().map(|&x| x * 2).collect();
    assert_eq!(parallel, serial);
}
