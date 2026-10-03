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
