//! Tests for the determinism tier: deterministic iteration order, fixed-seed
//! stable hashing, container correctness vs the standard library, and
//! deterministic (address-independent) allocation order.

use super::{OrderedMap, OrderedSet};
use crate::alloc_::{Allocator, FrameAllocator};
use crate::hash::{stable_hash_bytes, StableBuildHasher, StableHasher};
use core::alloc::Layout;
use core::hash::{BuildHasher, Hasher};

/// A deliberately degenerate hasher state is not required: we only need to show
/// that the index hasher carries no entropy. Build two hashers and confirm they
/// agree bit-for-bit on the same input.
#[test]
fn stable_build_hasher_is_fixed_seed() {
    let a = StableBuildHasher.build_hasher().finish();
    let b = StableBuildHasher.build_hasher().finish();
    // Fresh hashers start from the identical canonical seed.
    assert_eq!(a, b);

    let mut h1 = StableBuildHasher.build_hasher();
    let mut h2 = StableBuildHasher.build_hasher();
    h1.write(b"prism::determinism");
    h2.write(b"prism::determinism");
    assert_eq!(h1.finish(), h2.finish());
    assert_eq!(h1.finish(), {
        let mut h = StableHasher::new();
        h.write(b"prism::determinism");
        h.finish()
    });
    assert_eq!(stable_hash_bytes(b"prism::determinism"), h1.finish());
}

#[test]
fn ordered_map_preserves_insertion_order() {
    let mut m = OrderedMap::new();
    for k in [7, 3, 9, 1, 42, 5] {
        m.insert(k, k * 10);
    }
    let order: Vec<i32> = m.keys().copied().collect();
    assert_eq!(order, vec![7, 3, 9, 1, 42, 5]);

    // Overwriting an existing key keeps its position.
    assert_eq!(m.insert(9, 999), Some(90));
    let order: Vec<i32> = m.keys().copied().collect();
    assert_eq!(order, vec![7, 3, 9, 1, 42, 5]);
    assert_eq!(m.get(&9), Some(&999));
}

#[test]
fn ordered_map_remove_preserves_order() {
    let mut m: OrderedMap<i32, i32> = (0..10).map(|k| (k, k)).collect();
    assert_eq!(m.remove(&3), Some(3));
    assert_eq!(m.remove(&0), Some(0));
    assert_eq!(m.remove(&9), Some(9));
    let order: Vec<i32> = m.keys().copied().collect();
    assert_eq!(order, vec![1, 2, 4, 5, 6, 7, 8]);
    // Surviving values remain correctly addressable after the index fix-ups.
    for k in [1, 2, 4, 5, 6, 7, 8] {
        assert_eq!(m.get(&k), Some(&k));
    }
    assert_eq!(m.remove(&3), None);
}

#[test]
fn ordered_map_swap_remove_is_deterministic() {
    let build = || -> OrderedMap<i32, i32> { (0..6).map(|k| (k, k)).collect() };
    let mut a = build();
    let mut b = build();
    assert_eq!(a.swap_remove(&1), Some(1));
    assert_eq!(b.swap_remove(&1), Some(1));
    // Last element (5) moved into index 1.
    assert_eq!(a.keys().copied().collect::<Vec<_>>(), vec![0, 5, 2, 3, 4]);
    assert_eq!(
        a.keys().copied().collect::<Vec<_>>(),
        b.keys().copied().collect::<Vec<_>>()
    );
    // The moved key is still addressable (index was repointed).
    assert_eq!(a.get(&5), Some(&5));
    assert_eq!(a.get_index_of(&5), Some(1));
}

#[test]
fn ordered_map_iteration_order_independent_of_value_hash() {
    // Two maps built from the SAME key sequence must iterate identically,
    // regardless of the keys' hash values (seed-free, order driven by inserts).
    let seq = [1000003u64, 17, 0, u64::MAX, 42, 7, 99999999];
    let m1: OrderedMap<u64, u64> = seq.iter().map(|&k| (k, k)).collect();
    let m2: OrderedMap<u64, u64> = seq.iter().map(|&k| (k, k)).collect();
    let o1: Vec<u64> = m1.keys().copied().collect();
    let o2: Vec<u64> = m2.keys().copied().collect();
    assert_eq!(o1, seq.to_vec());
    assert_eq!(o1, o2);
}

#[test]
fn ordered_map_reconstruction_is_bit_identical() {
    // Replaying a mixed insert/remove script on fresh maps reproduces the exact
    // same iteration order every time — the reconstruction-stability guarantee.
    fn replay() -> Vec<(i32, i32)> {
        let mut m = OrderedMap::new();
        for k in [5, 1, 8, 3, 2, 9, 4] {
            m.insert(k, k + 100);
        }
        m.remove(&8);
        m.insert(7, 107);
        m.swap_remove(&1);
        m.insert(5, 500); // overwrite, keeps position
        m.remove(&3);
        m.iter().map(|(k, v)| (*k, *v)).collect()
    }
    let first = replay();
    for _ in 0..50 {
        assert_eq!(replay(), first);
    }
}

#[test]
fn ordered_map_entry_api() {
    let mut m: OrderedMap<&str, i32> = OrderedMap::new();
    *m.entry("a").or_insert(1) += 10;
    *m.entry("b").or_insert_with(|| 2) += 20;
    *m.entry("a").or_insert(999) += 1; // already present -> 11 -> 12
    m.entry("c").and_modify(|v| *v += 1).or_insert(3);
    m.entry("c").and_modify(|v| *v += 100).or_insert(999); // present -> 103
    *m.entry("d").or_default() += 7;

    assert_eq!(m.get("a"), Some(&12));
    assert_eq!(m.get("b"), Some(&22));
    assert_eq!(m.get("c"), Some(&103));
    assert_eq!(m.get("d"), Some(&7));
    // Entry insertion keeps first-seen order.
    assert_eq!(m.keys().copied().collect::<Vec<_>>(), vec!["a", "b", "c", "d"]);
    assert_eq!(m.entry("b").key(), &"b");
}

#[test]
fn ordered_map_matches_std_hashmap_contents() {
    use std::collections::HashMap as Std;
    let mut ours = OrderedMap::new();
    let mut std = Std::new();
    let script = [(1, 10), (2, 20), (3, 30), (2, 25), (4, 40), (1, 11)];
    for (k, v) in script {
        assert_eq!(ours.insert(k, v), std.insert(k, v));
    }
    ours.remove(&3);
    std.remove(&3);
    assert_eq!(ours.len(), std.len());
    for (k, v) in &std {
        assert_eq!(ours.get(k), Some(v));
    }
    for (k, v) in &ours {
        assert_eq!(std.get(k), Some(v));
    }
}

#[test]
fn ordered_map_index_and_iter_mut() {
    let mut m: OrderedMap<&str, i32> = OrderedMap::new();
    m.insert("x", 1);
    m.insert("y", 2);
    assert_eq!(m["x"], 1);
    for (_, v) in m.iter_mut() {
        *v *= 10;
    }
    assert_eq!(m["x"], 10);
    assert_eq!(m["y"], 20);
    assert_eq!(m.get_index(0), Some((&"x", &10)));
    assert_eq!(m.get_index(1), Some((&"y", &20)));
    assert_eq!(m.get_index(2), None);
}

#[test]
fn ordered_map_equality_is_content_based() {
    let a: OrderedMap<i32, i32> = [(1, 1), (2, 2), (3, 3)].into_iter().collect();
    let b: OrderedMap<i32, i32> = [(3, 3), (2, 2), (1, 1)].into_iter().collect();
    // Different insertion order, same contents -> equal.
    assert_eq!(a, b);
    // But iteration order still differs.
    assert_ne!(
        a.keys().copied().collect::<Vec<_>>(),
        b.keys().copied().collect::<Vec<_>>()
    );
}

#[test]
fn ordered_map_into_iter_owns_in_order() {
    let m: OrderedMap<i32, i32> = [(9, 1), (4, 2), (7, 3)].into_iter().collect();
    let collected: Vec<(i32, i32)> = m.into_iter().collect();
    assert_eq!(collected, vec![(9, 1), (4, 2), (7, 3)]);
}

#[test]
fn ordered_set_preserves_order_and_dedups() {
    let mut s = OrderedSet::new();
    assert!(s.insert(5));
    assert!(s.insert(1));
    assert!(s.insert(9));
    assert!(!s.insert(5)); // duplicate, order unchanged
    assert_eq!(s.iter().copied().collect::<Vec<_>>(), vec![5, 1, 9]);
    assert_eq!(s.len(), 3);
    assert!(s.contains(&9));
    assert!(s.remove(&1));
    assert_eq!(s.iter().copied().collect::<Vec<_>>(), vec![5, 9]);
}

#[test]
fn ordered_set_algebra_is_deterministic() {
    let a: OrderedSet<i32> = [1, 2, 3, 4].into_iter().collect();
    let b: OrderedSet<i32> = [3, 4, 5, 6].into_iter().collect();
    assert_eq!(
        a.union(&b).iter().copied().collect::<Vec<_>>(),
        vec![1, 2, 3, 4, 5, 6]
    );
    assert_eq!(
        a.intersection(&b).iter().copied().collect::<Vec<_>>(),
        vec![3, 4]
    );
    assert_eq!(
        a.difference(&b).iter().copied().collect::<Vec<_>>(),
        vec![1, 2]
    );
    let sub: OrderedSet<i32> = [2, 3].into_iter().collect();
    assert!(sub.is_subset(&a));
    assert!(a.is_superset(&sub));
    assert!(!a.is_subset(&b));
}

#[test]
fn ordered_set_matches_std_hashset_contents() {
    use std::collections::HashSet as Std;
    let script = [4, 1, 4, 7, 2, 1, 9];
    let mut ours = OrderedSet::new();
    let mut std = Std::new();
    for v in script {
        assert_eq!(ours.insert(v), std.insert(v));
    }
    ours.remove(&7);
    std.remove(&7);
    assert_eq!(ours.len(), std.len());
    for v in &std {
        assert!(ours.contains(v));
    }
    for v in &ours {
        assert!(std.contains(v));
    }
}

#[test]
fn frame_allocator_offsets_are_address_independent() {
    // The deterministic-allocation guarantee: for a fixed request sequence, the
    // *offset* progression (used()) is identical across two independent
    // allocators, even though their absolute base addresses differ.
    fn run() -> Vec<usize> {
        let fa = FrameAllocator::with_align(4096, 64);
        let layouts = [
            Layout::from_size_align(10, 1).unwrap(),
            Layout::from_size_align(8, 8).unwrap(),
            Layout::from_size_align(1, 1).unwrap(),
            Layout::from_size_align(32, 16).unwrap(),
            Layout::from_size_align(100, 4).unwrap(),
        ];
        let mut used = Vec::new();
        for l in layouts {
            fa.allocate(l).expect("capacity");
            used.push(fa.used());
        }
        used
    }
    let a = run();
    let b = run();
    assert_eq!(a, b);
    // Offsets are monotonically increasing (pure bump order).
    assert!(a.windows(2).all(|w| w[0] <= w[1]));
}
