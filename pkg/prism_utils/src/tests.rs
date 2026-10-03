use crate::prelude::*;

#[test]
fn array_vec_push_pop_capacity() {
    let mut v: ArrayVec<i32, 3> = ArrayVec::new();
    assert!(v.is_empty());
    v.push(1);
    v.push(2);
    v.push(3);
    assert!(v.is_full());
    assert_eq!(v.try_push(4), Err(4));
    assert_eq!(v.len(), 3);
    assert_eq!(v.get(1), Some(&2));
    assert_eq!(v.pop(), Some(3));
    assert_eq!(v.len(), 2);
    let collected: Vec<i32> = v.iter().copied().collect();
    assert_eq!(collected, vec![1, 2]);
}

#[test]
fn small_vec_spills_preserving_order() {
    let mut v: SmallVec<i32, 2> = SmallVec::new();
    v.push(10);
    v.push(20);
    assert!(!v.spilled());
    v.push(30); // triggers spill
    assert!(v.spilled());
    assert_eq!(v.len(), 3);
    assert_eq!(v.get(0), Some(&10));
    assert_eq!(v.get(1), Some(&20));
    assert_eq!(v.get(2), Some(&30));
    assert_eq!(v.pop(), Some(30));
}

#[test]
fn arena_insert_and_get() {
    let mut a: Arena<&str> = Arena::new();
    let i0 = a.insert("alpha");
    let i1 = a.insert("beta");
    assert_eq!(a.len(), 2);
    assert_eq!(a.get(i0), Some(&"alpha"));
    assert_eq!(a.get(i1), Some(&"beta"));
    *a.get_mut(i1).unwrap() = "gamma";
    assert_eq!(a.get(i1), Some(&"gamma"));
}

#[test]
fn hashmap_facade_works() {
    let mut m: HashMap<&str, i32> = HashMap::default();
    m.insert("x", 1);
    m.insert("y", 2);
    assert_eq!(m.get("x"), Some(&1));
    assert_eq!(m.len(), 2);

    let mut s: HashSet<i32> = HashSet::default();
    assert!(s.insert(7));
    assert!(!s.insert(7));
    assert!(s.contains(&7));
}

#[test]
fn fxhasher_is_deterministic() {
    use core::hash::{Hash, Hasher};
    let hash_of = |v: &u64| {
        let mut h = FxHasher::default();
        v.hash(&mut h);
        h.finish()
    };
    assert_eq!(hash_of(&12345), hash_of(&12345));
    assert_ne!(hash_of(&1), hash_of(&2));
}

#[test]
fn slot_map_generational_invalidation() {
    let mut m: SlotMap<&str> = SlotMap::new();
    let a = m.insert("a");
    let b = m.insert("b");
    assert_eq!(m.len(), 2);
    assert_eq!(m.get(a), Some(&"a"));
    assert!(m.contains_key(b));

    // Remove `a`; its key must become stale.
    assert_eq!(m.remove(a), Some("a"));
    assert_eq!(m.get(a), None);
    assert!(!m.contains_key(a));
    assert_eq!(m.len(), 1);

    // Reusing the freed slot must produce a key that does NOT alias `a`.
    let c = m.insert("c");
    assert_eq!(c.index(), a.index(), "freed slot should be recycled");
    assert_ne!(c.generation(), a.generation(), "generation must advance");
    assert_eq!(m.get(c), Some(&"c"));
    assert_eq!(m.get(a), None, "old key must not resurface as the new value");

    *m.get_mut(c).unwrap() = "c2";
    assert_eq!(m.get(c), Some(&"c2"));

    // Iteration yields only live entries.
    let mut live: Vec<&str> = m.values().copied().collect();
    live.sort_unstable();
    assert_eq!(live, vec!["b", "c2"]);

    m.clear();
    assert!(m.is_empty());
    assert_eq!(m.get(b), None);
    assert_eq!(m.get(c), None);
}

#[test]
fn sparse_set_iteration_after_removals() {
    let mut s: SparseSet<i32> = SparseSet::new();
    assert_eq!(s.insert(10, 100), None);
    assert_eq!(s.insert(3, 30), None);
    assert_eq!(s.insert(7, 70), None);
    assert_eq!(s.insert(10, 101), Some(100)); // overwrite returns old
    assert_eq!(s.len(), 3);
    assert!(s.contains(3));
    assert!(!s.contains(42));
    assert_eq!(s.get(7), Some(&70));

    // swap-remove the middle id; remaining ids/values must stay consistent.
    assert_eq!(s.remove(3), Some(30));
    assert_eq!(s.remove(3), None);
    assert_eq!(s.len(), 2);
    assert!(!s.contains(3));

    let mut pairs: Vec<(u32, i32)> = s.iter().map(|(id, v)| (id, *v)).collect();
    pairs.sort_unstable();
    assert_eq!(pairs, vec![(7, 70), (10, 101)]);

    for v in s.values_mut() {
        *v += 1;
    }
    assert_eq!(s.get(7), Some(&71));
    assert_eq!(s.get(10), Some(&102));

    s.clear();
    assert!(s.is_empty());
    assert!(!s.contains(7));
}

#[test]
fn bit_set_operations_and_iteration() {
    let mut a = BitSet::new();
    assert!(a.set(1));
    assert!(a.set(64));
    assert!(a.set(130));
    assert!(!a.set(64)); // already set
    assert_eq!(a.count_ones(), 3);
    assert_eq!(a.len(), 3);
    assert!(a.contains(64));
    assert!(!a.contains(2));

    let collected: Vec<usize> = a.iter().collect();
    assert_eq!(collected, vec![1, 64, 130]);

    assert!(!a.toggle(1)); // 1 was set; toggle clears it, new state is false
    assert!(!a.contains(1));
    assert!(a.toggle(1)); // toggle sets it back, new state is true
    assert!(a.contains(1));

    assert!(a.clear(130));
    assert!(!a.clear(130));
    assert!(!a.contains(130));

    let mut b = BitSet::new();
    b.set(1);
    b.set(2);
    b.set(64);

    // a = {1, 64}, b = {1, 2, 64}
    let union: Vec<usize> = a.union(&b).iter().collect();
    assert_eq!(union, vec![1, 2, 64]);
    let inter: Vec<usize> = a.intersection(&b).iter().collect();
    assert_eq!(inter, vec![1, 64]);
    let diff: Vec<usize> = b.difference(&a).iter().collect();
    assert_eq!(diff, vec![2]);

    // In-place variants mirror the producing variants.
    let mut c = a.clone();
    c.intersect_with(&b);
    assert_eq!(c, a.intersection(&b));
    c.union_with(&b);
    assert_eq!(c, b);
    c.difference_with(&a);
    let left: Vec<usize> = c.iter().collect();
    assert_eq!(left, vec![2]);

    b.clear_all();
    assert!(b.is_empty());
    assert_eq!(b.count_ones(), 0);
}

// ---------------------------------------------------------------------------
// M2: allocators
// ---------------------------------------------------------------------------

use core::alloc::Layout;

#[test]
fn global_allocator_round_trip_varied_layouts() {
    let global = Global;
    // A spread of sizes and alignments, including over-aligned and zero-sized.
    let layouts = [
        Layout::from_size_align(1, 1).unwrap(),
        Layout::from_size_align(8, 8).unwrap(),
        Layout::from_size_align(64, 32).unwrap(),
        Layout::from_size_align(4096, 4096).unwrap(),
        Layout::from_size_align(0, 16).unwrap(),
    ];
    for layout in layouts {
        let block = global.allocate(layout).expect("allocation must succeed");
        assert!(block.len() >= layout.size());
        let addr = block.as_ptr().cast::<u8>() as usize;
        assert_eq!(addr % layout.align(), 0, "returned block must be aligned");
        #[expect(unsafe_code, reason = "test exercises the raw allocator deallocation path")]
        // SAFETY: `block` was just produced by `global.allocate(layout)` and is
        // handed straight back with the same layout, used nowhere else.
        unsafe {
            global.deallocate(block.cast::<u8>(), layout);
        }
    }
}

#[test]
fn pool_recycles_the_same_block() {
    let pool = Pool::new(Layout::from_size_align(32, 8).unwrap(), 4);
    assert!(pool.block_size() >= 32);
    assert_eq!(pool.live(), 0);

    let a = pool.allocate_block().unwrap();
    assert_eq!(pool.live(), 1);
    #[expect(unsafe_code, reason = "test exercises the raw allocator deallocation path")]
    // SAFETY: `a` is a live block from this pool, freed exactly once.
    unsafe {
        pool.deallocate_block(a);
    }
    assert_eq!(pool.live(), 0);

    // The very next allocation must reuse the block we just freed.
    let b = pool.allocate_block().unwrap();
    assert_eq!(a, b, "freed block should be recycled");
    #[expect(unsafe_code, reason = "test exercises the raw allocator deallocation path")]
    // SAFETY: `b` is live and freed exactly once.
    unsafe {
        pool.deallocate_block(b);
    }
}

#[test]
fn pool_grows_across_chunks() {
    // 2 blocks per chunk: allocating 5 blocks must force 3 chunks.
    let pool = Pool::new(Layout::from_size_align(16, 8).unwrap(), 2);
    let mut blocks = Vec::new();
    for _ in 0..5 {
        blocks.push(pool.allocate_block().unwrap());
    }
    assert_eq!(pool.live(), 5);
    assert_eq!(pool.chunk_count(), 3, "5 blocks / 2 per chunk => 3 chunks");

    // All handed-out blocks must be distinct, non-overlapping addresses.
    let mut addrs: Vec<usize> = blocks.iter().map(|p| p.as_ptr() as usize).collect();
    addrs.sort_unstable();
    addrs.dedup();
    assert_eq!(addrs.len(), 5, "blocks must not alias");

    for b in blocks {
        #[expect(unsafe_code, reason = "test exercises the raw allocator deallocation path")]
        // SAFETY: each block is live and freed exactly once here.
        unsafe {
            pool.deallocate_block(b);
        }
    }
    assert_eq!(pool.live(), 0);
    // Freeing does not release chunks; capacity is retained for reuse.
    assert_eq!(pool.chunk_count(), 3);
}

#[test]
fn frame_allocator_alignment_and_reset() {
    let mut frame = FrameAllocator::new(1024);
    assert_eq!(frame.used(), 0);

    // A 1-byte allocation, then an over-aligned one: the second must be padded
    // up to its alignment.
    let one = frame.allocate(Layout::from_size_align(1, 1).unwrap()).unwrap();
    assert_eq!(one.len(), 1);

    let aligned = frame
        .allocate(Layout::from_size_align(32, 64).unwrap())
        .unwrap();
    let addr = aligned.as_ptr().cast::<u8>() as usize;
    assert_eq!(addr % 64, 0, "block must honor requested alignment");
    assert!(frame.used() >= 33);

    // reset reclaims the whole frame in O(1).
    frame.reset();
    assert_eq!(frame.used(), 0);

    // After reset the cursor restarts, so the first block address repeats.
    let again = frame.allocate(Layout::from_size_align(1, 1).unwrap()).unwrap();
    assert_eq!(
        again.as_ptr().cast::<u8>() as usize,
        one.as_ptr().cast::<u8>() as usize,
        "reset must rewind to the start of the frame"
    );
}

#[test]
fn frame_allocator_reports_exhaustion() {
    let frame = FrameAllocator::with_align(64, 16);
    assert!(
        frame
            .allocate(Layout::from_size_align(128, 1).unwrap())
            .is_err(),
        "requests larger than capacity must fail cleanly"
    );
}

#[test]
fn alloc_box_over_global_and_pool() {
    // Over the global heap.
    let mut boxed = AllocBox::new_in(1234u64, Global);
    assert_eq!(*boxed, 1234);
    *boxed += 1;
    assert_eq!(*boxed, 1235);
    drop(boxed);

    // Over a shared pool: the box borrows `&pool`, so one pool backs many boxes.
    let pool = Pool::new(Layout::new::<u64>(), 8);
    {
        let a = AllocBox::new_in(7u64, &pool);
        let b = AllocBox::new_in(8u64, &pool);
        assert_eq!(*a + *b, 15);
        assert_eq!(pool.live(), 2);
    }
    // Both boxes dropped: their blocks returned to the pool.
    assert_eq!(pool.live(), 0);

    // Dropped blocks are recycled on the next allocation.
    let c = AllocBox::new_in(99u64, &pool);
    assert_eq!(*c, 99);
    assert_eq!(pool.live(), 1);
}

#[test]
fn alloc_box_runs_destructors() {
    use core::cell::Cell;

    // A payload that bumps a borrowed counter when dropped.
    struct Dropper<'a>(&'a Cell<u32>);
    impl Drop for Dropper<'_> {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }

    let counter = Cell::new(0u32);
    let boxed = AllocBox::new_in(Dropper(&counter), Global);
    assert_eq!(counter.get(), 0);
    drop(boxed);
    assert_eq!(counter.get(), 1, "AllocBox must run the value's destructor once");
}
