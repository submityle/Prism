//! §24.7 deterministic parallel record/replay tests.
//!
//! Anti-vacuous contract: the pure order core
//! ([`ExecutionOrder`](crate::ExecutionOrder) + [`fold_in_order`](crate::fold_in_order))
//! is checked directly — permutation validation, text/byte round-trips, and an
//! order-sensitive fold whose result provably depends on the order. The façade
//! ([`TaskPool::record_ordered`](crate::TaskPool::record_ordered) /
//! [`replay_ordered`](crate::TaskPool::replay_ordered) /
//! [`deterministic_ordered`](crate::TaskPool::deterministic_ordered)) is then
//! exercised on real multi-threaded pools and the single-threaded fallback:
//! * a recorded run equals the independent serial oracle folded over its own
//!   captured order,
//! * replaying that order reproduces the value bit-for-bit on 1/2/4/8-worker
//!   and single-threaded pools, and
//! * the canonical fold is worker-count invariant (same seed → same result).

use alloc::vec;
use alloc::vec::Vec;

use crate::{fold_in_order, ExecutionOrder, ReplayOrderError, SeedStream, TaskPool};

// ----------------------------------------------------------------------------
// Shared workload: a pure per-task value and an order-sensitive combine.
// ----------------------------------------------------------------------------

/// FNV-1a offset basis / prime, used to build an order-sensitive hashing fold.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Pure per-task value: a reproducible function of `(seed, index)`.
fn task_value(seed: u64, i: usize) -> u64 {
    SeedStream::for_task(seed, i as u64)
}

/// Order-sensitive combine (non-commutative hashing fold): swapping two commit
/// positions changes the result, which is exactly why the commit order has to
/// be recorded/replayed to be reproducible.
fn mix(acc: u64, _i: usize, v: u64) -> u64 {
    (acc ^ v).wrapping_mul(FNV_PRIME)
}

/// Pools to prove worker-count invariance: 1/2/4/8 real workers plus the
/// single-threaded inline fallback (`with_threads(0)`).
fn pools() -> Vec<TaskPool> {
    vec![
        TaskPool::with_threads(0),
        TaskPool::with_threads(1),
        TaskPool::with_threads(2),
        TaskPool::with_threads(4),
        TaskPool::with_threads(8),
    ]
}

// ----------------------------------------------------------------------------
// Pure core: ExecutionOrder + fold_in_order.
// ----------------------------------------------------------------------------

#[test]
fn canonical_is_identity_permutation() {
    let order = ExecutionOrder::canonical(7, 5);
    assert_eq!(order.seed(), 7);
    assert_eq!(order.len(), 5);
    assert!(!order.is_empty());
    assert_eq!(order.order(), &[0, 1, 2, 3, 4]);
    order.validate().unwrap();
}

#[test]
fn empty_order_is_valid_and_empty() {
    let order = ExecutionOrder::canonical(1, 0);
    assert!(order.is_empty());
    assert_eq!(order.len(), 0);
    order.validate().unwrap();
}

#[test]
fn new_rejects_non_permutations() {
    // Repeated index.
    assert_eq!(
        ExecutionOrder::new(0, vec![0, 1, 1]),
        Err(ReplayOrderError::NotAPermutation { len: 3 })
    );
    // Index out of range for the length.
    assert_eq!(
        ExecutionOrder::new(0, vec![0, 1, 9]),
        Err(ReplayOrderError::TaskOutOfRange { task: 9, len: 3 })
    );
    // A genuine permutation is accepted.
    assert!(ExecutionOrder::new(0, vec![2, 0, 1]).is_ok());
}

#[test]
fn text_round_trip() {
    let order = ExecutionOrder::new(0x0123_4567_89ab_cdef, vec![3, 1, 0, 2]).unwrap();
    let text = order.to_text();
    let parsed = ExecutionOrder::from_text(&text).unwrap();
    assert_eq!(order, parsed);
}

#[test]
fn bytes_round_trip() {
    let order = ExecutionOrder::new(42, vec![1, 0, 3, 2, 4]).unwrap();
    let bytes = order.to_bytes();
    let parsed = ExecutionOrder::from_bytes(&bytes).unwrap();
    assert_eq!(order, parsed);
}

#[test]
fn malformed_text_is_rejected() {
    assert_eq!(ExecutionOrder::from_text(""), Err(ReplayOrderError::Truncated));
    assert_eq!(
        ExecutionOrder::from_text("not-a-header\n"),
        Err(ReplayOrderError::BadHeader)
    );
    assert_eq!(
        ExecutionOrder::from_text("prism-order v1 seed=zz len=0\n"),
        Err(ReplayOrderError::BadHeader)
    );
    // Declares two entries but only one follows.
    assert_eq!(
        ExecutionOrder::from_text("prism-order v1 seed=01 len=2\n0\n"),
        Err(ReplayOrderError::LengthMismatch {
            expected: 2,
            found: 1
        })
    );
    // Non-numeric entry.
    assert_eq!(
        ExecutionOrder::from_text("prism-order v1 seed=01 len=1\nnope\n"),
        Err(ReplayOrderError::BadEntry)
    );
    // Right count, but not a permutation.
    assert_eq!(
        ExecutionOrder::from_text("prism-order v1 seed=01 len=2\n0\n0\n"),
        Err(ReplayOrderError::NotAPermutation { len: 2 })
    );
}

#[test]
fn malformed_bytes_are_rejected() {
    assert_eq!(ExecutionOrder::from_bytes(&[]), Err(ReplayOrderError::Truncated));
    let mut good = ExecutionOrder::new(1, vec![1, 0]).unwrap().to_bytes();
    good[0] = b'X';
    assert_eq!(
        ExecutionOrder::from_bytes(&good),
        Err(ReplayOrderError::BadHeader)
    );
    // Truncated body (header promises more indices than are present).
    let order = ExecutionOrder::new(1, vec![0, 1, 2]).unwrap();
    let mut bytes = order.to_bytes();
    bytes.truncate(bytes.len() - 4);
    assert!(matches!(
        ExecutionOrder::from_bytes(&bytes),
        Err(ReplayOrderError::LengthMismatch { .. })
    ));
}

#[test]
fn seed_stream_is_reproducible_and_index_addressed() {
    // `for_task` is a pure function of (seed, index).
    assert_eq!(
        SeedStream::for_task(99, 7),
        SeedStream::for_task(99, 7),
        "for_task must be deterministic"
    );
    assert_ne!(
        SeedStream::for_task(99, 7),
        SeedStream::for_task(99, 8),
        "different tasks should (overwhelmingly) differ"
    );
    // The sequential stream is also reproducible from a seed.
    let mut a = SeedStream::new(5);
    let mut b = SeedStream::new(5);
    for _ in 0..16 {
        assert_eq!(a.next_u64(), b.next_u64());
    }
}

#[test]
fn fold_in_order_is_order_sensitive() {
    // The whole machinery exists because `mix` is non-commutative: folding the
    // same values in a different order yields a different result.
    let compute = |i: usize| task_value(1234, i);
    let forward = fold_in_order(&[0, 1, 2, 3], FNV_OFFSET, compute, mix);
    let reversed = fold_in_order(&[3, 2, 1, 0], FNV_OFFSET, compute, mix);
    assert_ne!(
        forward, reversed,
        "an order-sensitive fold must depend on the order"
    );
}

// ----------------------------------------------------------------------------
// Façade: record / replay / deterministic over real pools.
// ----------------------------------------------------------------------------

#[test]
fn record_equals_serial_oracle_over_its_own_order() {
    let pool = TaskPool::with_threads(8);
    let seed = 0xC0FF_EE00_u64;
    let len = 2000;
    let compute = |i: usize| task_value(seed, i);

    let out = pool.record_ordered(seed, len, compute, FNV_OFFSET, mix);

    // The captured order is a genuine permutation of 0..len.
    assert_eq!(out.order.len(), len);
    out.order.validate().unwrap();

    // The recorded run equals the independent serial oracle folded over the
    // exact order it captured — the record is faithful.
    let oracle = fold_in_order(out.order.order(), FNV_OFFSET, compute, mix);
    assert_eq!(out.value, oracle);
}

#[test]
fn replay_reproduces_recorded_value_on_every_pool() {
    let record_pool = TaskPool::with_threads(8);
    let seed = 0xABCD_1234_u64;
    let len = 1500;
    let compute = |i: usize| task_value(seed, i);

    let out = record_pool.record_ordered(seed, len, compute, FNV_OFFSET, mix);

    // Replaying the captured order reproduces the value bit-for-bit regardless
    // of how many workers the replay pool has.
    for pool in pools() {
        let replayed = pool.replay_ordered(&out.order, compute, FNV_OFFSET, mix).unwrap();
        assert_eq!(
            replayed, out.value,
            "replay diverged on a {}-worker pool",
            pool.worker_count()
        );
    }
}

#[test]
fn replay_survives_serialization_round_trip() {
    let pool = TaskPool::with_threads(4);
    let seed = 0x5EED_u64;
    let len = 777;
    let compute = |i: usize| task_value(seed, i);

    let out = pool.record_ordered(seed, len, compute, FNV_OFFSET, mix);

    // Serialize the order to text and bytes, parse both back, and replay.
    let via_text = ExecutionOrder::from_text(&out.order.to_text()).unwrap();
    let via_bytes = ExecutionOrder::from_bytes(&out.order.to_bytes()).unwrap();
    assert_eq!(via_text, out.order);
    assert_eq!(via_bytes, out.order);

    let replay_pool = TaskPool::with_threads(2);
    assert_eq!(
        replay_pool.replay_ordered(&via_text, compute, FNV_OFFSET, mix).unwrap(),
        out.value
    );
    assert_eq!(
        replay_pool.replay_ordered(&via_bytes, compute, FNV_OFFSET, mix).unwrap(),
        out.value
    );
}

#[test]
fn deterministic_ordered_is_worker_count_invariant() {
    let seed = 0x1357_9BDF_u64;
    let len = 3000;
    let compute = |i: usize| task_value(seed, i);

    // The canonical fold must equal the serial oracle in canonical order, on
    // every pool size — "same seed → same result, no matter how many cores".
    let expected = fold_in_order(
        ExecutionOrder::canonical(seed, len).order(),
        FNV_OFFSET,
        compute,
        mix,
    );
    for pool in pools() {
        let got = pool.deterministic_ordered(seed, len, compute, FNV_OFFSET, mix);
        assert_eq!(
            got,
            expected,
            "deterministic fold diverged on a {}-worker pool",
            pool.worker_count()
        );
    }
}

#[test]
fn deterministic_differs_from_a_scrambled_replay_order() {
    // Canonical order and a non-trivial recorded order generally disagree for
    // an order-sensitive fold, confirming the fold really honours the order.
    let pool = TaskPool::with_threads(8);
    let seed = 0x2468_ACE0_u64;
    let len = 1024;
    let compute = |i: usize| task_value(seed, i);

    let canonical = pool.deterministic_ordered(seed, len, compute, FNV_OFFSET, mix);

    // A deliberately reversed order (a valid permutation) folds differently.
    let reversed: Vec<u32> = (0..len as u32).rev().collect();
    let order = ExecutionOrder::new(seed, reversed).unwrap();
    let scrambled = pool.replay_ordered(&order, compute, FNV_OFFSET, mix).unwrap();

    assert_ne!(canonical, scrambled);
}

#[test]
fn different_seeds_give_different_results() {
    let pool = TaskPool::with_threads(4);
    let len = 1000;
    let a = pool.deterministic_ordered(1, len, |i| task_value(1, i), FNV_OFFSET, mix);
    let b = pool.deterministic_ordered(2, len, |i| task_value(2, i), FNV_OFFSET, mix);
    assert_ne!(a, b, "distinct seeds should produce distinct results");
}

#[test]
fn repeated_records_are_each_internally_faithful() {
    // Finish order may vary run-to-run (we do not assert it does), but every
    // recording must equal the oracle folded over the order it captured.
    let pool = TaskPool::with_threads(8);
    let seed = 0xFEED_FACE_u64;
    let len = 1200;
    let compute = |i: usize| task_value(seed, i);

    for _ in 0..8 {
        let out = pool.record_ordered(seed, len, compute, FNV_OFFSET, mix);
        out.order.validate().unwrap();
        assert_eq!(
            out.value,
            fold_in_order(out.order.order(), FNV_OFFSET, compute, mix)
        );
        // And replaying it reproduces the same value.
        assert_eq!(
            pool.replay_ordered(&out.order, compute, FNV_OFFSET, mix).unwrap(),
            out.value
        );
    }
}

#[test]
fn empty_workload_round_trips() {
    let pool = TaskPool::with_threads(4);
    let out = pool.record_ordered(7, 0, |i| task_value(7, i), FNV_OFFSET, mix);
    assert!(out.order.is_empty());
    assert_eq!(out.value, FNV_OFFSET); // nothing folded
    assert_eq!(
        pool.replay_ordered(&out.order, |i| task_value(7, i), FNV_OFFSET, mix).unwrap(),
        FNV_OFFSET
    );
    assert_eq!(
        pool.deterministic_ordered(7, 0, |i| task_value(7, i), FNV_OFFSET, mix),
        FNV_OFFSET
    );
}

#[test]
fn single_threaded_record_captures_canonical_order() {
    // The inline fallback commits tasks in submission order, so its recorded
    // order is the canonical identity and matches the deterministic fold.
    let pool = TaskPool::with_threads(0);
    assert!(pool.is_single_threaded());
    let seed = 0x0D15_EA5E_u64;
    let len = 64;
    let compute = |i: usize| task_value(seed, i);

    let out = pool.record_ordered(seed, len, compute, FNV_OFFSET, mix);
    let canonical: Vec<u32> = (0..len as u32).collect();
    assert_eq!(out.order.order(), canonical.as_slice());
    assert_eq!(
        out.value,
        pool.deterministic_ordered(seed, len, compute, FNV_OFFSET, mix)
    );
}

#[test]
fn replay_rejects_invalid_order() {
    let pool = TaskPool::with_threads(2);
    // Hand-craft an order that is a valid permutation, then corrupt a copy by
    // bypassing the validating constructor via serialization tampering.
    let order = ExecutionOrder::new(0, vec![0, 1, 2]).unwrap();
    let mut bytes = order.to_bytes();
    // Overwrite the first index (0) with 9 -> out of range on parse.
    let first = bytes.len() - 12;
    bytes[first] = 9;
    assert!(matches!(
        ExecutionOrder::from_bytes(&bytes),
        Err(ReplayOrderError::TaskOutOfRange { task: 9, len: 3 })
    ));
    // A directly built invalid order (constructed via from_parts path is not
    // public) cannot reach replay; the public `new` rejects it up front.
    assert!(ExecutionOrder::new(0, vec![0, 0, 0]).is_err());
    // Sanity: a valid order replays fine.
    assert!(pool
        .replay_ordered(&order, |i| task_value(0, i), FNV_OFFSET, mix)
        .is_ok());
}
