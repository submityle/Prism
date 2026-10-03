//! M6 tests: the advanced-layout + migration tier — a hierarchical bit set,
//! structure-of-arrays columnar storage, a copy-on-write container, the buddy
//! and `TLSF` region sub-allocators, and (behind `compat-bevy`) the
//! `bevy_utils`-shaped container aliases.

use crate::alloc_::{BuddyAllocator, TlsfAllocator};
use crate::cow::Cow;
use crate::hbitset::HierarchicalBitSet;
use crate::soa::SoaVec;

// --- Hierarchical bit set. --------------------------------------------------

#[test]
fn hbitset_insert_contains_remove_over_sparse_domain() {
    // A domain far larger than one word to force multiple summary levels.
    let mut bs = HierarchicalBitSet::with_capacity(1 << 18);
    assert!(bs.capacity() >= (1 << 18));
    assert!(bs.is_empty());

    let marks = [0usize, 63, 64, 4095, 4096, 200_000, (1 << 18) - 1];
    for &i in &marks {
        assert!(bs.insert(i), "first insert of {i} reports newly set");
        assert!(!bs.insert(i), "re-inserting {i} reports already set");
    }
    assert_eq!(bs.count_ones(), marks.len());
    for &i in &marks {
        assert!(bs.contains(i), "{i} must be present");
    }
    assert!(!bs.contains(1));
    assert!(!bs.contains(100_000));

    assert!(bs.remove(64));
    assert!(!bs.remove(64), "removing a clear bit reports nothing removed");
    assert!(!bs.contains(64));
    assert_eq!(bs.count_ones(), marks.len() - 1);
}

#[test]
fn hbitset_find_next_skips_empty_regions_and_iterates_in_order() {
    let mut bs = HierarchicalBitSet::with_capacity(1 << 16);
    let marks = [5usize, 70, 5000, 60_000];
    for &i in &marks {
        bs.insert(i);
    }

    assert_eq!(bs.find_first(), Some(5));
    // `find_next` is inclusive of `from`: querying a set index returns it.
    assert_eq!(bs.find_next(5), Some(5));
    // Advancing past a hit jumps across the empty gap to the next set bit.
    assert_eq!(bs.find_next(6), Some(70));
    assert_eq!(bs.find_next(71), Some(5000));
    assert_eq!(bs.find_next(5001), Some(60_000));
    assert_eq!(bs.find_next(60_001), None);

    let collected: Vec<usize> = bs.iter().collect();
    assert_eq!(collected, marks);
}

#[test]
fn hbitset_clear_empties_every_level() {
    let mut bs = HierarchicalBitSet::with_capacity(8192);
    for i in (0..8192).step_by(37) {
        bs.insert(i);
    }
    assert!(!bs.is_empty());
    bs.clear();
    assert!(bs.is_empty());
    assert_eq!(bs.count_ones(), 0);
    assert_eq!(bs.find_first(), None);
}

// --- Structure-of-arrays. ---------------------------------------------------

#[test]
fn soa_vec_stores_fields_in_separate_columns() {
    let mut v: SoaVec<(u32, f32)> = SoaVec::new();
    assert!(v.is_empty());
    v.push((1, 1.5));
    v.push((2, 2.5));
    v.push((3, 3.5));
    assert_eq!(v.len(), 3);

    assert_eq!(v.get(1), Some((&2, &2.5)));
    let (ids, weights) = v.columns();
    assert_eq!(ids, &[1, 2, 3]);
    assert_eq!(weights, &[1.5, 2.5, 3.5]);

    // Columns really are contiguous and disjoint.
    if let Some((id, w)) = v.get_mut(0) {
        *id = 10;
        *w = -1.0;
    }
    assert_eq!(v.get(0), Some((&10, &-1.0)));
    assert_eq!(v.get(2), Some((&3, &3.5)));
}

#[test]
fn soa_vec_swap_remove_and_iter() {
    let mut v: SoaVec<(u8, u16, u32)> = SoaVec::with_capacity(4);
    v.push((1, 10, 100));
    v.push((2, 20, 200));
    v.push((3, 30, 300));

    // swap_remove returns the removed tuple and moves the last element in.
    assert_eq!(v.swap_remove(0), Some((1, 10, 100)));
    assert_eq!(v.len(), 2);
    assert_eq!(v.get(0), Some((&3, &30, &300)));
    assert_eq!(v.swap_remove(5), None);

    let rows: Vec<(u8, u16, u32)> = v.iter().map(|(a, b, c)| (*a, *b, *c)).collect();
    assert_eq!(rows, vec![(3, 30, 300), (2, 20, 200)]);

    v.clear();
    assert!(v.is_empty());
}

// --- Copy-on-write. ---------------------------------------------------------

#[test]
fn cow_shares_until_first_mutation() {
    let a = Cow::new(vec![1, 2, 3]);
    let b = a.clone();
    assert!(Cow::ptr_eq(&a, &b), "clone shares the same allocation");
    assert!(a.is_shared());
    assert_eq!(a.ref_count(), 2);
    // A shared value cannot be mutated in place.
    let mut b = b;
    assert!(b.get_mut().is_none());

    b.make_mut().push(4); // b diverges here (deep copy)
    assert!(!Cow::ptr_eq(&a, &b));
    assert_eq!(a.get().as_slice(), &[1, 2, 3], "a is untouched");
    assert_eq!(b.get().as_slice(), &[1, 2, 3, 4]);
    assert!(!a.is_shared());
    assert_eq!(a.ref_count(), 1);
}

#[test]
fn cow_unique_value_mutates_in_place_without_copy() {
    let mut a = Cow::new(String::from("hi"));
    assert!(!a.is_shared());
    let before = core::ptr::from_ref(a.get());
    a.make_mut().push_str(" there");
    let after = core::ptr::from_ref(a.get());
    assert_eq!(before, after, "a unique CoW must mutate in place");
    assert_eq!(a.get(), "hi there");
    assert_eq!(a.into_inner(), "hi there");
}

// --- Buddy sub-allocator. ---------------------------------------------------

#[test]
fn buddy_allocates_splits_and_coalesces() {
    // 16-byte min block, 2^6 = 64 blocks => 1024-byte region.
    let mut b = BuddyAllocator::new(16, 6);
    assert_eq!(b.capacity(), 1024);
    assert_eq!(b.free_bytes(), 1024);

    // Two small allocations are naturally aligned and distinct.
    let a0 = b.allocate(16).expect("first 16B block");
    let a1 = b.allocate(16).expect("second 16B block");
    assert_ne!(a0, a1);
    assert_eq!(a0 % 16, 0);
    assert_eq!(a1 % 16, 0);
    assert_eq!(b.allocated_bytes(), 32);

    // A 300-byte request rounds up to a 512-byte block.
    let big = b.allocate(300).expect("one 512B block");
    assert_eq!(big % 512, 0);
    assert_eq!(b.allocated_bytes(), 32 + 512);

    // Freeing everything coalesces back to one full-size free block.
    b.deallocate(a0, 16);
    b.deallocate(a1, 16);
    b.deallocate(big, 300);
    assert_eq!(b.allocated_bytes(), 0);
    assert_eq!(b.free_bytes(), 1024);
    // The whole region is free again, so a full-capacity request succeeds.
    assert!(b.allocate(1024).is_some());
}

#[test]
fn buddy_reports_exhaustion_honestly() {
    let mut b = BuddyAllocator::new(32, 2); // 128-byte region.
    assert_eq!(b.capacity(), 128);
    let _whole = b.allocate(128).expect("the whole region");
    assert!(b.allocate(32).is_none(), "nothing left to give");
    // A request larger than the region can never be satisfied.
    let mut b2 = BuddyAllocator::new(32, 2);
    assert!(b2.allocate(129).is_none());
    assert!(b2.allocate(0).is_none());
}

// --- TLSF sub-allocator. ----------------------------------------------------

#[test]
fn tlsf_allocates_frees_and_recoalesces() {
    let mut t = TlsfAllocator::new(4096);
    assert_eq!(t.capacity(), 4096);

    let a = t.allocate(100).expect("a");
    let b = t.allocate(200).expect("b");
    let c = t.allocate(300).expect("c");
    assert!(a != b && b != c && a != c);
    assert_eq!(a % 8, 0);
    assert!(t.allocated_bytes() >= 100 + 200 + 300);

    assert!(t.deallocate(b));
    assert!(!t.deallocate(b), "double free is reported, not panicked");
    assert!(t.deallocate(a));
    assert!(t.deallocate(c));
    assert_eq!(t.allocated_bytes(), 0);

    // After everything is freed and coalesced, a near-capacity block fits.
    assert!(t.allocate(4000).is_some());
}

#[test]
fn tlsf_honours_alignment_and_exhaustion() {
    let mut t = TlsfAllocator::new(8192);
    let p = t.allocate_aligned(64, 256).expect("256-aligned block");
    assert_eq!(p % 256, 0, "returned offset must honour the requested align");

    assert!(t.allocate(0).is_none());
    // A request larger than the region fails cleanly.
    assert!(t.allocate(1 << 20).is_none());
}

// --- Bevy migration aliases (feature-gated). --------------------------------

#[cfg(feature = "compat-bevy")]
#[test]
fn compat_bevy_aliases_resolve_to_prism_containers() {
    use crate::compat_bevy::prelude::*;

    let mut map: HashMap<&str, u32> = HashMap::default();
    map.insert("a", 1);
    map.insert("b", 2);
    assert_eq!(map.get("a"), Some(&1));

    let mut set: HashSet<u32> = HashSet::default();
    assert!(set.insert(7));
    assert!(!set.insert(7));
    assert!(set.contains(&7));

    let mut stable: StableHashMap<u32, u32> = StableHashMap::default();
    stable.insert(1, 10);
    assert_eq!(stable.get(&1), Some(&10));

    let mut sv: SmallVec<u8, 4> = SmallVec::new();
    sv.push(1);
    sv.push(2);
    assert_eq!(sv.len(), 2);
}
