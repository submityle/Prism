//! [`OrderedSet`]: a deterministic, insertion-ordered hash set.
//!
//! Built on top of [`OrderedMap`](super::ordered_map::OrderedMap) with a unit
//! value, so it inherits the same determinism guarantee: iteration visits
//! elements in insertion order, independent of hash seed, address, or platform.
//! The set-algebra helpers ([`union`](OrderedSet::union),
//! [`intersection`](OrderedSet::intersection),
//! [`difference`](OrderedSet::difference)) all produce their results in a fixed,
//! insertion-derived order so downstream consumers stay reproducible.

use core::borrow::Borrow;
use core::fmt;
use core::hash::Hash;

use super::ordered_map::OrderedMap;

/// A deterministic, insertion-ordered set.
///
/// Elements keep the order in which they were first inserted. As with
/// [`OrderedMap`], [`remove`](OrderedSet::remove) preserves the relative order
/// of the survivors while [`swap_remove`](OrderedSet::swap_remove) is `O(1)`
/// but reorders. Equality is content-based (order-insensitive).
pub struct OrderedSet<T> {
    map: OrderedMap<T, ()>,
}

impl<T> OrderedSet<T> {
    /// Create an empty set.
    #[inline]
    pub fn new() -> Self {
        Self {
            map: OrderedMap::new(),
        }
    }

    /// Create an empty set with capacity pre-reserved for `cap` elements.
    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            map: OrderedMap::with_capacity(cap),
        }
    }

    /// Number of elements stored.
    #[inline]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether the set holds no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Remove every element, keeping allocated capacity.
    #[inline]
    pub fn clear(&mut self) {
        self.map.clear();
    }

    /// Borrow the element at insertion position `index`, if any.
    #[inline]
    pub fn get_index(&self, index: usize) -> Option<&T> {
        self.map.get_index(index).map(|(k, _)| k)
    }

    /// Iterate over elements in insertion order.
    #[inline]
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            inner: self.map.iter(),
        }
    }
}

impl<T: Hash + Eq> OrderedSet<T> {
    /// Reserve capacity for at least `additional` more elements.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.map.reserve(additional);
    }

    /// Whether `value` is present.
    #[inline]
    pub fn contains<Q>(&self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.map.contains_key(value)
    }

    /// Return the insertion position of `value`, if present.
    #[inline]
    pub fn get_index_of<Q>(&self, value: &Q) -> Option<usize>
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.map.get_index_of(value)
    }
}

impl<T: Hash + Eq + Clone> OrderedSet<T> {
    /// Insert `value`. Returns `true` if the value was newly inserted, `false`
    /// if it was already present (in which case order is unchanged).
    #[inline]
    pub fn insert(&mut self, value: T) -> bool {
        self.map.insert(value, ()).is_none()
    }

    /// Remove `value`, preserving the relative order of the survivors (`O(n)`).
    /// Returns `true` if the value was present.
    #[inline]
    pub fn remove<Q>(&mut self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.map.remove(value).is_some()
    }

    /// Remove `value` in `O(1)` by moving the last element into its slot.
    /// Returns `true` if the value was present.
    #[inline]
    pub fn swap_remove<Q>(&mut self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.map.swap_remove(value).is_some()
    }

    /// The union `self ∪ other`, in deterministic order: `self`'s elements in
    /// insertion order, followed by `other`'s not already present.
    pub fn union(&self, other: &Self) -> Self {
        let mut out = Self::with_capacity(self.len() + other.len());
        for v in self.iter() {
            out.insert(v.clone());
        }
        for v in other.iter() {
            out.insert(v.clone());
        }
        out
    }

    /// The intersection `self ∩ other`, in `self`'s insertion order.
    pub fn intersection(&self, other: &Self) -> Self {
        let mut out = Self::new();
        for v in self.iter() {
            if other.contains(v) {
                out.insert(v.clone());
            }
        }
        out
    }

    /// The difference `self \ other`, in `self`'s insertion order.
    pub fn difference(&self, other: &Self) -> Self {
        let mut out = Self::new();
        for v in self.iter() {
            if !other.contains(v) {
                out.insert(v.clone());
            }
        }
        out
    }

    /// Whether every element of `self` is also in `other`.
    pub fn is_subset(&self, other: &Self) -> bool {
        self.iter().all(|v| other.contains(v))
    }

    /// Whether every element of `other` is also in `self`.
    pub fn is_superset(&self, other: &Self) -> bool {
        other.is_subset(self)
    }
}

impl<T> Default for OrderedSet<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> Clone for OrderedSet<T> {
    fn clone(&self) -> Self {
        Self {
            map: self.map.clone(),
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for OrderedSet<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl<T: Hash + Eq> PartialEq for OrderedSet<T> {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().all(|v| other.contains(v))
    }
}

impl<T: Hash + Eq> Eq for OrderedSet<T> {}

impl<T: Hash + Eq + Clone> FromIterator<T> for OrderedSet<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut set = Self::with_capacity(iter.size_hint().0);
        for v in iter {
            set.insert(v);
        }
        set
    }
}

impl<T: Hash + Eq + Clone> Extend<T> for OrderedSet<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for v in iter {
            self.insert(v);
        }
    }
}

/// Owning iterator over set elements in insertion order.
pub struct IntoIter<T> {
    inner: super::ordered_map::IntoIter<T, ()>,
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;
    #[inline]
    fn next(&mut self) -> Option<T> {
        self.inner.next().map(|(k, _)| k)
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<T> ExactSizeIterator for IntoIter<T> {}

impl<T> IntoIterator for OrderedSet<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            inner: self.map.into_iter(),
        }
    }
}

/// Borrowing iterator over set elements in insertion order.
pub struct Iter<'a, T> {
    inner: super::ordered_map::Iter<'a, T, ()>,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;
    #[inline]
    fn next(&mut self) -> Option<&'a T> {
        self.inner.next().map(|(k, _)| k)
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<T> ExactSizeIterator for Iter<'_, T> {}

impl<'a, T> IntoIterator for &'a OrderedSet<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
