//! Oracle-backed tests for the content-addressed [`InternCache`].
//!
//! The oracle is a deliberately naive deduplicator: a `Vec<T>` scanned
//! linearly for an equal value. For any operation sequence the production
//! cache must agree with the oracle on (a) the handle index assigned to each
//! value and (b) the stored contents, which pins down both the dedup and the
//! `O(1)`-handle-equality contracts.

use super::cache::{InternCache, Interned};
use super::domain;

extern crate alloc;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

/// Naive linear-scan interner used as the correctness oracle.
#[derive(Default)]
struct Oracle<T> {
    items: Vec<T>,
}

impl<T: PartialEq + Clone> Oracle<T> {
    fn intern(&mut self, value: &T) -> u32 {
        if let Some(i) = self.items.iter().position(|v| v == value) {
            return i as u32;
        }
        self.items.push(value.clone());
        (self.items.len() - 1) as u32
    }

    fn get(&self, value: &T) -> Option<u32> {
        self.items.iter().position(|v| v == value).map(|i| i as u32)
    }
}

#[test]
fn equal_content_shares_a_handle() {
    let mut cache: InternCache<Vec<u8>> = InternCache::new();
    let a = cache.intern(vec![1, 2, 3]);
    let b = cache.intern(vec![1, 2, 3]);
    let c = cache.intern(vec![1, 2, 4]);
    assert_eq!(a, b, "identical content must share a handle");
    assert_ne!(a, c, "different content must get different handles");
    assert_eq!(cache.len(), 2, "only two distinct values are stored");
}

#[test]
fn handle_equality_is_content_equality() {
    // The whole point of §24.6: comparing handles (`O(1)`) is equivalent to
    // comparing the deep content.
    let mut cache: InternCache<String> = InternCache::new();
    let samples = ["alpha", "beta", "alpha", "gamma", "beta"];
    let handles: Vec<_> = samples
        .iter()
        .map(|s| cache.intern(s.to_string()))
        .collect();
    for i in 0..samples.len() {
        for j in 0..samples.len() {
            assert_eq!(
                handles[i] == handles[j],
                samples[i] == samples[j],
                "handle equality must mirror content equality at ({i}, {j})"
            );
        }
    }
}

#[test]
fn resolve_round_trips() {
    let mut cache: InternCache<String> = InternCache::new();
    let h = cache.intern("prism::Transform".to_string());
    assert_eq!(cache.resolve(h), "prism::Transform");
    assert_eq!(
        cache.try_resolve(h).map(String::as_str),
        Some("prism::Transform")
    );
    assert_eq!(cache.try_resolve(Interned::from_index(999)), None);
}

#[test]
fn get_and_contains_do_not_insert() {
    let mut cache: InternCache<Vec<u8>> = InternCache::new();
    assert!(cache.get(&vec![7]).is_none());
    assert!(!cache.contains(&vec![7]));
    assert_eq!(cache.len(), 0);
    let h = cache.intern(vec![7]);
    assert_eq!(cache.get(&vec![7]), Some(h));
    assert!(cache.contains(&vec![7]));
    assert_eq!(cache.len(), 1);
}

#[test]
fn intern_ref_clones_only_when_absent() {
    let mut cache: InternCache<String> = InternCache::new();
    let owned = "shared".to_string();
    let a = cache.intern_ref(&owned);
    let b = cache.intern_ref(&owned); // already present: no new entry
    assert_eq!(a, b);
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.resolve(a), "shared");
}

#[test]
fn matches_oracle_over_a_long_mixed_sequence() {
    // A deterministic pseudo-random stream of small values exercises hash
    // collisions, repeats and distinct inserts against the naive oracle.
    let mut cache: InternCache<Vec<u8>> = InternCache::new();
    let mut oracle: Oracle<Vec<u8>> = Oracle::default();
    let mut state: u64 = 0x1234_5678_9abc_def0;
    let mut handles: Vec<(Interned<Vec<u8>>, Vec<u8>)> = Vec::new();
    for _ in 0..4000 {
        // xorshift64 for a reproducible stream.
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let len = (state % 4) as usize;
        let value: Vec<u8> = (0..len)
            .map(|k| ((state >> (k * 8)) & 0x07) as u8)
            .collect();

        let got = cache.intern(value.clone());
        let want = oracle.intern(&value);
        assert_eq!(got.index(), want, "handle index must match the oracle");
        assert_eq!(
            cache.resolve(got),
            &value,
            "resolve must return the content"
        );
        assert_eq!(
            cache.get(&value),
            oracle.get(&value).map(Interned::from_index)
        );
        handles.push((got, value));
    }
    assert_eq!(cache.len(), oracle.items.len());
    // Every handle ever issued still resolves to its original content and
    // equal content still shares a handle.
    for (h, v) in &handles {
        assert_eq!(cache.resolve(*h), v);
        assert_eq!(cache.get(v), Some(*h));
    }
}

#[test]
fn iter_yields_insertion_order() {
    let mut cache: InternCache<u32> = InternCache::new();
    cache.intern(10);
    cache.intern(20);
    cache.intern(10); // dedup, no new slot
    cache.intern(30);
    let collected: Vec<_> = cache.iter().map(|(_, v)| *v).collect();
    assert_eq!(collected, [10, 20, 30]);
    // Handle indices are dense and in insertion order.
    let indices: Vec<_> = cache.iter().map(|(h, _)| h.index()).collect();
    assert_eq!(indices, [0, 1, 2]);
}

#[test]
fn clear_reclaims_the_domain_and_restarts_indices() {
    let mut cache: InternCache<String> = InternCache::new();
    cache.intern("a".to_string());
    cache.intern("b".to_string());
    assert_eq!(cache.len(), 2);
    cache.clear();
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
    // After reclamation, interning starts fresh from index 0.
    let h = cache.intern("c".to_string());
    assert_eq!(h.index(), 0);
    assert_eq!(cache.resolve(h), "c");
}

#[test]
fn content_hash_is_stable_and_collision_safe() {
    // Equal content hashes equal; the full `Eq` check still separates two
    // values that happened to collide (simulated here by forcing both into the
    // same bucket is not possible from the public API, so we assert the weaker
    // but meaningful property: distinct content never shares a handle even when
    // hashes are close).
    let h1 = InternCache::<Vec<u8>>::content_hash(&vec![1, 2, 3]);
    let h2 = InternCache::<Vec<u8>>::content_hash(&vec![1, 2, 3]);
    assert_eq!(h1, h2, "equal content must hash equal");

    let mut cache: InternCache<Vec<u8>> = InternCache::new();
    let a = cache.intern(vec![0, 0]);
    let b = cache.intern(vec![0, 0, 0]);
    assert_ne!(a, b);
    assert_eq!(cache.len(), 2);
}

#[test]
fn domains_do_not_collide_in_the_type_system() {
    let mut meshes: InternCache<Vec<u8>, domain::Mesh> = InternCache::new();
    let mut textures: InternCache<Vec<u8>, domain::Texture> = InternCache::new();
    let m = meshes.intern(vec![1, 2, 3]);
    let t = textures.intern(vec![1, 2, 3]);
    // Same raw index in each cache...
    assert_eq!(m.index(), t.index());
    // ...but the handles are different *types*: `m == t` does not type-check.
    assert_eq!(meshes.resolve(m), &[1, 2, 3]);
    assert_eq!(textures.resolve(t), &[1, 2, 3]);
}

#[test]
fn handle_is_u32_sized_and_copy() {
    assert_eq!(size_of::<Interned<Vec<u8>>>(), size_of::<u32>());
    let mut cache: InternCache<u32> = InternCache::new();
    let a = cache.intern(42);
    let b = a; // Copy, not move
    assert_eq!(a, b);
    assert_eq!(cache.resolve(a), &42);
}
