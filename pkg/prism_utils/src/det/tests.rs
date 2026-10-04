//! Tests for the deterministic merge + reproducible hashing tier (`det`, §24.7).
//!
//! The hash tests are **known-answer** tests: the expected digests were
//! computed once with an independent big-integer reference implementation of
//! the exact published formulas and frozen here, so any accidental change to
//! the algorithm (which would silently break every stored digest) fails the
//! suite. The merge tests pin order-independence of the canonical result.

extern crate alloc;

use super::hash::{
    mix64, reproducible_hash_ordered, reproducible_hash_unordered, OrderedHashCombiner,
    UnorderedHashCombiner,
};
use super::merge::DeterministicMerge;
use alloc::vec::Vec;

/// The golden-ratio seed constant, mirrored from the implementation so the
/// known-answer vectors can reference it by value.
const SEED: u64 = 0x9E37_79B9_7F4A_7C15;

// ---------------------------------------------------------------------------
// mix64 known-answer vectors (frozen from an independent reference).
// ---------------------------------------------------------------------------

#[test]
fn mix64_known_vectors() {
    // 0 is the fixed point of the SplitMix64 finaliser.
    assert_eq!(mix64(0), 0x0000_0000_0000_0000);
    assert_eq!(mix64(1), 0x5692_161d_100b_05e5);
    assert_eq!(mix64(SEED), 0xe220_a839_7b1d_cdaf);
    assert_eq!(mix64(0xdead_beef), 0x4e06_2702_ec92_9eea);
}

#[test]
fn mix64_is_injective_on_a_sample() {
    // The finaliser is a bijection; spot-check that a block of distinct inputs
    // maps to distinct outputs.
    let mut seen = alloc::collections::BTreeSet::new();
    for i in 0u64..5000 {
        assert!(seen.insert(mix64(i)), "mix64 collision at {i}");
    }
}

// ---------------------------------------------------------------------------
// Ordered combiner: order-sensitive, known-answer.
// ---------------------------------------------------------------------------

#[test]
fn ordered_hash_known_vectors() {
    // Empty stream: mix64(SEED ^ 0) == mix64(SEED).
    assert_eq!(reproducible_hash_ordered([]), 0xe220_a839_7b1d_cdaf);
    assert_eq!(reproducible_hash_ordered([1]), 0x713c_fa22_ac69_d39e);
    assert_eq!(reproducible_hash_ordered([1, 2, 3]), 0xee33_14c6_44c0_36cd);
    assert_eq!(reproducible_hash_ordered([3, 2, 1]), 0x0b83_bf10_2a1a_89d4);
}

#[test]
fn ordered_hash_is_order_sensitive() {
    assert_ne!(
        reproducible_hash_ordered([1, 2, 3]),
        reproducible_hash_ordered([3, 2, 1])
    );
    assert_ne!(
        reproducible_hash_ordered([1, 2, 3]),
        reproducible_hash_ordered([1, 3, 2])
    );
}

#[test]
fn ordered_hash_length_disambiguates_zero_streams() {
    // All-zero lanes of different lengths must not collide (count is folded in).
    let one = reproducible_hash_ordered([0]);
    let two = reproducible_hash_ordered([0, 0]);
    let three = reproducible_hash_ordered([0, 0, 0]);
    assert_ne!(one, two);
    assert_ne!(two, three);
    assert_ne!(one, three);
}

#[test]
fn ordered_combiner_incremental_matches_free_fn() {
    let mut h = OrderedHashCombiner::new();
    assert_eq!(h.count(), 0);
    h.write(1);
    h.write(2);
    h.write(3);
    assert_eq!(h.count(), 3);
    assert_eq!(h.finish(), reproducible_hash_ordered([1, 2, 3]));
    // write_all is equivalent to repeated write.
    let mut g = OrderedHashCombiner::new();
    g.write_all([1u64, 2, 3]);
    assert_eq!(g.finish(), h.finish());
}

// ---------------------------------------------------------------------------
// Unordered combiner: permutation-invariant, known-answer.
// ---------------------------------------------------------------------------

#[test]
fn unordered_hash_known_vectors() {
    assert_eq!(reproducible_hash_unordered([]), 0x4821_8226_ff3c_d4bf);
    assert_eq!(reproducible_hash_unordered([1]), 0x92f4_ba17_16f6_94f1);
    assert_eq!(reproducible_hash_unordered([1, 2, 3]), 0xb8c1_90a9_4743_4478);
}

#[test]
fn unordered_hash_is_permutation_invariant() {
    let base = reproducible_hash_unordered([1, 2, 3]);
    assert_eq!(reproducible_hash_unordered([3, 2, 1]), base);
    assert_eq!(reproducible_hash_unordered([2, 1, 3]), base);
    assert_eq!(reproducible_hash_unordered([2, 3, 1]), base);
}

#[test]
fn unordered_hash_distinguishes_multisets() {
    // Same sum of mixes is astronomically unlikely for genuinely different
    // multisets; also the count differs here.
    assert_ne!(
        reproducible_hash_unordered([1, 2, 3]),
        reproducible_hash_unordered([1, 2, 3, 4])
    );
    // Multiplicity matters.
    assert_ne!(
        reproducible_hash_unordered([1, 1, 2]),
        reproducible_hash_unordered([1, 2, 2])
    );
}

#[test]
fn unordered_combiner_merge_is_associative() {
    // Fold the full set in one combiner...
    let full = reproducible_hash_unordered([10, 20, 30, 40, 50, 60]);

    // ...vs three partial combiners merged in an arbitrary order.
    let mut a = UnorderedHashCombiner::new();
    a.write_all([30u64, 10]);
    let mut b = UnorderedHashCombiner::new();
    b.write_all([60u64]);
    let mut c = UnorderedHashCombiner::new();
    c.write_all([50u64, 20, 40]);

    let mut merged = UnorderedHashCombiner::new();
    merged.merge(&c);
    merged.merge(&a);
    merged.merge(&b);
    assert_eq!(merged.count(), 6);
    assert_eq!(merged.finish(), full);
}

// ---------------------------------------------------------------------------
// DeterministicMerge: canonical order independent of contribution order.
// ---------------------------------------------------------------------------

#[test]
fn merge_into_sorted_is_order_independent() {
    let mut a = DeterministicMerge::new();
    a.contribute(3u32, 30i32);
    a.contribute(1, 10);
    a.contribute(2, 20);
    a.contribute(1, 11);

    let mut b = DeterministicMerge::with_capacity(4);
    // Reverse contribution order.
    b.contribute(1, 11);
    b.contribute(2, 20);
    b.contribute(1, 10);
    b.contribute(3, 30);

    let sa = a.into_sorted();
    let sb = b.into_sorted();
    assert_eq!(sa, sb);
    // Hand oracle: sorted by (key, value).
    assert_eq!(sa, alloc::vec![(1, 10), (1, 11), (2, 20), (3, 30)]);
}

#[test]
fn merge_reduce_sums_per_key() {
    let mut m = DeterministicMerge::new();
    for (k, v) in [(2u32, 5i64), (1, 100), (2, 7), (3, 1), (1, 1), (2, 8)] {
        m.contribute(k, v);
    }
    let reduced = m.reduce(|a, b| a + b);
    // key 1: 100+1=101; key 2: 5+7+8=20; key 3: 1.
    assert_eq!(reduced, alloc::vec![(1, 101), (2, 20), (3, 1)]);
}

#[test]
fn merge_reduce_is_order_independent_for_commutative_combine() {
    let data = [(1u32, 4u64), (2, 9), (1, 6), (3, 2), (2, 1), (1, 10)];

    let mut forward = DeterministicMerge::new();
    for &(k, v) in &data {
        forward.contribute(k, v);
    }
    let mut backward = DeterministicMerge::new();
    for &(k, v) in data.iter().rev() {
        backward.contribute(k, v);
    }

    // max is commutative + associative.
    let rf = forward.reduce(u64::max);
    let rb = backward.reduce(u64::max);
    assert_eq!(rf, rb);
    assert_eq!(rf, alloc::vec![(1, 10), (2, 9), (3, 2)]);
}

#[test]
fn merge_empty_and_single() {
    let empty: DeterministicMerge<u32, u32> = DeterministicMerge::new();
    assert!(empty.is_empty());
    assert_eq!(empty.len(), 0);
    assert_eq!(empty.into_sorted(), Vec::new());

    let mut one = DeterministicMerge::new();
    one.contribute(42u32, 7u32);
    assert_eq!(one.len(), 1);
    assert_eq!(one.reduce(|a, b| a + b), alloc::vec![(42, 7)]);
}

#[test]
fn merge_result_feeds_reproducible_hash() {
    // The canonical merge order can be fingerprinted with the ordered hash, and
    // the digest is independent of contribution order.
    let mut a = DeterministicMerge::new();
    a.contribute(5u64, 50u64);
    a.contribute(1, 10);
    a.contribute(9, 90);
    let mut b = DeterministicMerge::new();
    b.contribute(9, 90);
    b.contribute(5, 50);
    b.contribute(1, 10);

    let lanes_a: Vec<u64> = a.into_sorted().into_iter().flat_map(|(k, v)| [k, v]).collect();
    let lanes_b: Vec<u64> = b.into_sorted().into_iter().flat_map(|(k, v)| [k, v]).collect();
    assert_eq!(
        reproducible_hash_ordered(lanes_a),
        reproducible_hash_ordered(lanes_b)
    );
}

// ---------------------------------------------------------------------------
// Concurrent wrapper (gated): many-thread contribution is deterministic.
// ---------------------------------------------------------------------------

#[cfg(feature = "concurrent")]
#[test]
fn concurrent_merge_matches_sequential_oracle() {
    use super::merge::ConcurrentMerge;
    use alloc::sync::Arc;
    use std::thread;

    // Build the expected sequential multiset: 8 workers each contribute
    // (worker, i) for i in 0..100 — but the concurrent result must equal the
    // sorted oracle regardless of interleaving.
    const WORKERS: u64 = 8;
    const PER: u64 = 100;

    let run = || {
        let merge = Arc::new(ConcurrentMerge::<u64, u64>::new());
        let mut handles = Vec::new();
        for w in 0..WORKERS {
            let m = Arc::clone(&merge);
            handles.push(thread::spawn(move || {
                for i in 0..PER {
                    m.contribute(i % 10, w * PER + i);
                }
            }));
        }
        for h in handles {
            h.join().expect("worker thread panicked");
        }
        Arc::try_unwrap(merge)
            .expect("all workers joined, Arc is unique")
            .into_sorted()
    };

    let first = run();
    // Determinism across independent runs with fresh thread scheduling.
    for _ in 0..3 {
        assert_eq!(run(), first);
    }
    assert_eq!(first.len() as u64, WORKERS * PER);

    // Equal to the single-threaded oracle over the same contribution multiset.
    let mut oracle = DeterministicMerge::new();
    for w in 0..WORKERS {
        for i in 0..PER {
            oracle.contribute(i % 10, w * PER + i);
        }
    }
    assert_eq!(first, oracle.into_sorted());
}

#[cfg(feature = "concurrent")]
#[test]
fn concurrent_merge_reduce_sums() {
    use super::merge::ConcurrentMerge;
    use alloc::sync::Arc;
    use std::thread;

    let merge = Arc::new(ConcurrentMerge::<u32, u64>::new());
    let mut handles = Vec::new();
    for _ in 0..4 {
        let m = Arc::clone(&merge);
        handles.push(thread::spawn(move || {
            for _ in 0..1000 {
                m.contribute(0u32, 1u64);
                m.contribute(1u32, 2u64);
            }
        }));
    }
    for h in handles {
        h.join().expect("worker thread panicked");
    }
    let reduced = Arc::try_unwrap(merge)
        .expect("unique Arc")
        .into_reduced(|a, b| a + b);
    // key 0: 4*1000*1 = 4000; key 1: 4*1000*2 = 8000.
    assert_eq!(reduced, alloc::vec![(0, 4000), (1, 8000)]);
}
