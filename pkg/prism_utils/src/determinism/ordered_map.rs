//! [`OrderedMap`]: a deterministic, insertion-ordered hash map (an `IndexMap`
//! form).
//!
//! The map keeps a packed [`Vec`] of `(key, value)` entries in **insertion
//! order** plus a parallel hash index for `O(1)` lookup. Iteration walks the
//! entry vector directly, so the visit order is a pure function of the
//! insert/remove operation sequence — it never depends on a random hash seed,
//! on pointer addresses, or on the target platform. That is exactly what the
//! four-way determinism contract (see the [module docs](super)) needs: the same
//! inputs always produce the same iteration order, so replicated simulations
//! stay in lock-step.
//!
//! The index uses the crate's fixed-seed
//! [`StableBuildHasher`](crate::hash::StableBuildHasher) rather than the
//! standard library's `RandomState`, so even the internal probe sequence is
//! reproducible. (Iteration order does not depend on the hasher regardless,
//! because it is driven by the entry vector, but the fixed seed keeps the whole
//! structure free of run-to-run entropy.)

use core::borrow::Borrow;
use core::fmt;
use core::hash::Hash;
use core::ops::Index;
use std::collections::HashMap as StdHashMap;

use crate::hash::StableBuildHasher;

/// A deterministic, insertion-ordered map.
///
/// Keys keep the order in which they were first inserted. Overwriting an
/// existing key via [`insert`](OrderedMap::insert) updates the value in place
/// and leaves the key's position unchanged. [`remove`](OrderedMap::remove)
/// preserves the relative order of the surviving keys (an `O(n)` shift), while
/// [`swap_remove`](OrderedMap::swap_remove) is `O(1)` but moves the last entry
/// into the hole. Both removal strategies are fully deterministic for a given
/// operation sequence.
///
/// Equality is content-based (order-insensitive), matching the semantics of a
/// plain hash map: two maps are equal when they hold the same key/value pairs
/// regardless of insertion order.
pub struct OrderedMap<K, V> {
    /// Key -> position in `entries`. Keyed by the fixed-seed stable hasher so
    /// the structure carries no run-to-run entropy.
    indices: StdHashMap<K, usize, StableBuildHasher>,
    /// Packed `(key, value)` pairs in insertion order; the single source of
    /// truth for iteration order.
    entries: Vec<(K, V)>,
}

impl<K, V> OrderedMap<K, V> {
    /// Create an empty map.
    #[inline]
    pub fn new() -> Self {
        Self {
            indices: StdHashMap::with_hasher(StableBuildHasher),
            entries: Vec::new(),
        }
    }

    /// Create an empty map with capacity pre-reserved for `cap` entries.
    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            indices: StdHashMap::with_capacity_and_hasher(cap, StableBuildHasher),
            entries: Vec::with_capacity(cap),
        }
    }

    /// Number of key/value pairs stored.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the map holds no entries.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Remove every entry, keeping allocated capacity.
    #[inline]
    pub fn clear(&mut self) {
        self.indices.clear();
        self.entries.clear();
    }

    /// Borrow the key/value pair at insertion position `index`, if any.
    #[inline]
    pub fn get_index(&self, index: usize) -> Option<(&K, &V)> {
        self.entries.get(index).map(|(k, v)| (k, v))
    }

    /// Mutably borrow the value at insertion position `index`, if any.
    #[inline]
    pub fn get_index_mut(&mut self, index: usize) -> Option<(&K, &mut V)> {
        self.entries.get_mut(index).map(|(k, v)| (&*k, v))
    }

    /// Iterate over `(&K, &V)` in insertion order.
    #[inline]
    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter {
            inner: self.entries.iter(),
        }
    }

    /// Iterate over `(&K, &mut V)` in insertion order.
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        IterMut {
            inner: self.entries.iter_mut(),
        }
    }

    /// Iterate over keys in insertion order.
    #[inline]
    pub fn keys(&self) -> impl Iterator<Item = &K> + '_ {
        self.entries.iter().map(|(k, _)| k)
    }

    /// Iterate over values in insertion order.
    #[inline]
    pub fn values(&self) -> impl Iterator<Item = &V> + '_ {
        self.entries.iter().map(|(_, v)| v)
    }

    /// Mutably iterate over values in insertion order.
    #[inline]
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut V> + '_ {
        self.entries.iter_mut().map(|(_, v)| v)
    }
}

impl<K: Hash + Eq, V> OrderedMap<K, V> {
    /// Reserve capacity for at least `additional` more entries.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.indices.reserve(additional);
        self.entries.reserve(additional);
    }

    /// Whether `key` is present.
    #[inline]
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.indices.contains_key(key)
    }

    /// Borrow the value for `key`, if present.
    #[inline]
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.indices.get(key).map(|&i| &self.entries[i].1)
    }

    /// Mutably borrow the value for `key`, if present.
    #[inline]
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let i = *self.indices.get(key)?;
        Some(&mut self.entries[i].1)
    }

    /// Borrow the full `(&K, &V)` pair for `key`, if present.
    #[inline]
    pub fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let i = *self.indices.get(key)?;
        let (k, v) = &self.entries[i];
        Some((k, v))
    }

    /// Return the insertion position of `key`, if present.
    #[inline]
    pub fn get_index_of<Q>(&self, key: &Q) -> Option<usize>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.indices.get(key).copied()
    }
}

impl<K: Hash + Eq + Clone, V> OrderedMap<K, V> {
    /// Insert `value` for `key`, returning the previous value if the key was
    /// already present.
    ///
    /// A brand-new key is appended at the end (preserving insertion order); an
    /// existing key keeps its position and only its value is replaced.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        if let Some(&i) = self.indices.get(&key) {
            Some(core::mem::replace(&mut self.entries[i].1, value))
        } else {
            let idx = self.entries.len();
            self.indices.insert(key.clone(), idx);
            self.entries.push((key, value));
            None
        }
    }

    /// Remove `key`, preserving the relative order of the remaining entries.
    ///
    /// This shifts every later entry down by one (`O(n)`). The result is
    /// deterministic: the surviving entries keep their insertion order.
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let idx = self.indices.remove(key)?;
        let (_k, v) = self.entries.remove(idx);
        for pos in self.indices.values_mut() {
            if *pos > idx {
                *pos -= 1;
            }
        }
        Some(v)
    }

    /// Remove `key` in `O(1)` by moving the last entry into its slot.
    ///
    /// This changes the iteration order (the former last entry takes the
    /// removed entry's position), but the outcome is still deterministic for a
    /// given operation sequence. Prefer [`remove`](OrderedMap::remove) when
    /// insertion order must be preserved.
    pub fn swap_remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let idx = self.indices.remove(key)?;
        let last = self.entries.len() - 1;
        let (_k, v) = self.entries.swap_remove(idx);
        if idx != last {
            // The entry that lived at `last` now sits at `idx`; repoint it.
            let moved_key = self.entries[idx].0.clone();
            self.indices.insert(moved_key, idx);
        }
        Some(v)
    }

    /// Get the [`Entry`] for `key` for in-place insert-or-update.
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V> {
        match self.indices.get(&key) {
            Some(&index) => Entry::Occupied(OccupiedEntry { map: self, index }),
            None => Entry::Vacant(VacantEntry { map: self, key }),
        }
    }
}

impl<K, V> Default for OrderedMap<K, V> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Clone, V: Clone> Clone for OrderedMap<K, V> {
    fn clone(&self) -> Self {
        Self {
            indices: self.indices.clone(),
            entries: self.entries.clone(),
        }
    }
}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for OrderedMap<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<K: Hash + Eq, V: PartialEq> PartialEq for OrderedMap<K, V> {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter()
            .all(|(k, v)| other.get(k).is_some_and(|ov| ov == v))
    }
}

impl<K: Hash + Eq, V: Eq> Eq for OrderedMap<K, V> {}

impl<K: Hash + Eq, Q, V> Index<&Q> for OrderedMap<K, V>
where
    K: Borrow<Q>,
    Q: Hash + Eq + ?Sized,
{
    type Output = V;

    fn index(&self, key: &Q) -> &V {
        self.get(key).expect("no entry found for key")
    }
}

impl<K: Hash + Eq + Clone, V> FromIterator<(K, V)> for OrderedMap<K, V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut map = Self::with_capacity(iter.size_hint().0);
        for (k, v) in iter {
            map.insert(k, v);
        }
        map
    }
}

impl<K: Hash + Eq + Clone, V> Extend<(K, V)> for OrderedMap<K, V> {
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

/// Owning iterator over `(K, V)` in insertion order.
pub struct IntoIter<K, V> {
    inner: std::vec::IntoIter<(K, V)>,
}

impl<K, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<K, V> ExactSizeIterator for IntoIter<K, V> {}

impl<K, V> IntoIterator for OrderedMap<K, V> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            inner: self.entries.into_iter(),
        }
    }
}

/// Borrowing iterator over `(&K, &V)` in insertion order.
pub struct Iter<'a, K, V> {
    inner: core::slice::Iter<'a, (K, V)>,
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, v)| (k, v))
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<K, V> ExactSizeIterator for Iter<'_, K, V> {}

impl<'a, K, V> IntoIterator for &'a OrderedMap<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Mutable borrowing iterator over `(&K, &mut V)` in insertion order.
pub struct IterMut<'a, K, V> {
    inner: core::slice::IterMut<'a, (K, V)>,
}

impl<'a, K, V> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, v)| (&*k, v))
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<K, V> ExactSizeIterator for IterMut<'_, K, V> {}

impl<'a, K, V> IntoIterator for &'a mut OrderedMap<K, V> {
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

/// A view into a single [`OrderedMap`] slot, obtained from
/// [`OrderedMap::entry`].
pub enum Entry<'a, K, V> {
    /// The key is already present.
    Occupied(OccupiedEntry<'a, K, V>),
    /// The key is absent.
    Vacant(VacantEntry<'a, K, V>),
}

/// An occupied [`Entry`].
pub struct OccupiedEntry<'a, K, V> {
    map: &'a mut OrderedMap<K, V>,
    index: usize,
}

/// A vacant [`Entry`].
pub struct VacantEntry<'a, K, V> {
    map: &'a mut OrderedMap<K, V>,
    key: K,
}

impl<'a, K: Hash + Eq + Clone, V> Entry<'a, K, V> {
    /// Ensure a value is present, inserting `default` if vacant, then return a
    /// mutable reference to it.
    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => e.insert(default),
        }
    }

    /// Like [`or_insert`](Entry::or_insert) but computes the default lazily.
    pub fn or_insert_with<F: FnOnce() -> V>(self, default: F) -> &'a mut V {
        match self {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => e.insert(default()),
        }
    }

    /// Run `f` on the value if the entry is occupied, then return the entry.
    pub fn and_modify<F: FnOnce(&mut V)>(mut self, f: F) -> Self {
        if let Entry::Occupied(e) = &mut self {
            f(e.get_mut());
        }
        self
    }

    /// The key this entry refers to.
    pub fn key(&self) -> &K {
        match self {
            Entry::Occupied(e) => e.key(),
            Entry::Vacant(e) => e.key(),
        }
    }
}

impl<'a, K: Hash + Eq + Clone, V: Default> Entry<'a, K, V> {
    /// Ensure a value is present, inserting `V::default()` if vacant.
    pub fn or_default(self) -> &'a mut V {
        self.or_insert_with(V::default)
    }
}

impl<'a, K, V> OccupiedEntry<'a, K, V> {
    /// The key of this entry.
    #[inline]
    pub fn key(&self) -> &K {
        &self.map.entries[self.index].0
    }

    /// Borrow the value.
    #[inline]
    pub fn get(&self) -> &V {
        &self.map.entries[self.index].1
    }

    /// Mutably borrow the value.
    #[inline]
    pub fn get_mut(&mut self) -> &mut V {
        &mut self.map.entries[self.index].1
    }

    /// Consume the entry and return a mutable reference tied to the map.
    #[inline]
    pub fn into_mut(self) -> &'a mut V {
        &mut self.map.entries[self.index].1
    }

    /// Replace the value, returning the old one.
    #[inline]
    pub fn insert(&mut self, value: V) -> V {
        core::mem::replace(self.get_mut(), value)
    }
}

impl<'a, K: Hash + Eq + Clone, V> VacantEntry<'a, K, V> {
    /// The key that would be inserted.
    #[inline]
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Take back ownership of the key.
    #[inline]
    pub fn into_key(self) -> K {
        self.key
    }

    /// Insert `value` for this entry's key and return a mutable reference.
    pub fn insert(self, value: V) -> &'a mut V {
        let idx = self.map.entries.len();
        self.map.indices.insert(self.key.clone(), idx);
        self.map.entries.push((self.key, value));
        &mut self.map.entries[idx].1
    }
}
