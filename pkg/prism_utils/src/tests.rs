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
