//! Content-addressed deduplication cache (design doc §24.6).
//!
//! [`InternCache<T, D>`] stores each *distinct value* of content type `T`
//! exactly once and hands back a cheap, `Copy` [`Interned<T, D>`] handle. It is
//! the general-value analogue of the string [`Interner`](super::Interner):
//!
//! - **Content addressing.** Values are keyed by a classic, non-cryptographic
//!   content hash — the crate's [`StableHasher`](crate::hash::StableHasher)
//!   (FNV-1a, the xxHash/FNV family the design doc calls for; it is neither
//!   cryptographic nor machine-learned). The hash only selects a bucket; a full
//!   [`Eq`] comparison then confirms identity, so hash collisions never alias
//!   distinct content.
//! - **Deduplication.** Interning the same content twice returns the same
//!   handle and stores one copy, so a scene full of identical meshes/textures
//!   costs one entry (asset/mesh/texture dedup for the bake pipeline).
//! - **`O(1)` equality.** Two [`Interned`] handles compare equal iff they came
//!   from the same cache and address the same content, so a content equality
//!   test collapses from a deep compare to a `u32` compare.
//! - **Domain separation.** `D` is an uninhabited [`domain`](super::domain)
//!   marker, so a mesh handle and a type handle are different types. Each
//!   domain is a separate cache; reclaim a whole domain by dropping or
//!   [`clear`](InternCache::clear)-ing it, which bounds the table's lifetime
//!   (design doc §23: no single, unbounded global intern table).
//!
//! ```
//! use prism_utils::intern::{InternCache, domain};
//!
//! let mut meshes: InternCache<Vec<u8>, domain::Mesh> = InternCache::new();
//! let a = meshes.intern(vec![1, 2, 3]);
//! let b = meshes.intern(vec![1, 2, 3]); // identical content -> same handle
//! let c = meshes.intern(vec![9]);
//! assert_eq!(a, b);
//! assert_ne!(a, c);
//! assert_eq!(meshes.len(), 2); // only two distinct blocks stored
//! assert_eq!(meshes.resolve(a), &[1, 2, 3]);
//! ```
//!
//! ## Identity contract
//! An [`Interned`] is only meaningful to the [`InternCache`] that produced it.
//! Resolving a handle with a different cache instance (even of the same domain)
//! is a logic error; [`resolve`](InternCache::resolve) panics on an
//! out-of-range handle and [`try_resolve`](InternCache::try_resolve) returns
//! `None`. [`clear`](InternCache::clear) invalidates every previously issued
//! handle (new handles start again from index 0).

extern crate alloc;

use alloc::vec::Vec;
use core::hash::Hash;
use core::marker::PhantomData;

use super::domain;
use crate::hash::{stable_hash, FxBuildHasher, HashMap};

/// A content-addressed handle: a cheap, `Copy`, domain-tagged `u32` index into
/// an [`InternCache`].
///
/// All trait impls are hand-written so they never require the content type `T`
/// or the domain marker `D` to implement anything (both are compile-time tags
/// held in `PhantomData<fn() -> (T, D)>`, which is always
/// `Copy`/`Send`/`Sync`). Two handles compare equal iff they index the same
/// slot of the same cache, which is the `O(1)` equality the module promises.
pub struct Interned<T, D = ()> {
    raw: u32,
    _marker: PhantomData<fn() -> (T, D)>,
}

impl<T, D> Interned<T, D> {
    /// Wrap a raw index. Used internally and for deserialization; the caller
    /// must guarantee the index came from a matching [`InternCache`].
    #[inline]
    pub const fn from_index(raw: u32) -> Self {
        Self {
            raw,
            _marker: PhantomData,
        }
    }

    /// The raw `u32` index backing this handle.
    #[inline]
    pub const fn index(self) -> u32 {
        self.raw
    }
}

impl<T, D> Clone for Interned<T, D> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, D> Copy for Interned<T, D> {}

impl<T, D> PartialEq for Interned<T, D> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw
    }
}

impl<T, D> Eq for Interned<T, D> {}

impl<T, D> PartialOrd for Interned<T, D> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T, D> Ord for Interned<T, D> {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.raw.cmp(&other.raw)
    }
}

impl<T, D> Hash for Interned<T, D> {
    #[inline]
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.raw.hash(state);
    }
}

impl<T, D> core::fmt::Debug for Interned<T, D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Interned({})", self.raw)
    }
}

/// A convenience alias for a content cache keyed in the asset-ID domain.
pub type AssetCache<T> = InternCache<T, domain::Asset>;

/// A domain-separated, content-addressed deduplication cache.
///
/// Backed by a `Vec<T>` of distinct values plus a hash bucket index keyed by
/// the crate's cross-run [`stable_hash`]. Interning and lookup are amortized
/// `O(1)`; stored values are never moved or dropped (except by
/// [`clear`](Self::clear)) for the cache's lifetime, so issued handles stay
/// valid.
pub struct InternCache<T, D = ()> {
    items: Vec<T>,
    buckets: HashMap<u64, Vec<u32>>,
    _marker: PhantomData<fn() -> (T, D)>,
}

impl<T, D> InternCache<T, D> {
    /// Create an empty cache.
    #[inline]
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            buckets: HashMap::default(),
            _marker: PhantomData,
        }
    }

    /// Create an empty cache with room for `cap` distinct values.
    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            items: Vec::with_capacity(cap),
            buckets: HashMap::with_capacity_and_hasher(cap, FxBuildHasher::default()),
            _marker: PhantomData,
        }
    }

    /// Number of distinct values stored.
    #[inline]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether no values have been interned.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Resolve a handle back to its stored value.
    ///
    /// # Panics
    /// Panics if `handle` did not come from this cache (out-of-range index),
    /// for example a handle issued before a [`clear`](Self::clear).
    #[inline]
    pub fn resolve(&self, handle: Interned<T, D>) -> &T {
        &self.items[handle.raw as usize]
    }

    /// Resolve a handle back to its stored value, returning `None` for an
    /// out-of-range handle instead of panicking.
    #[inline]
    pub fn try_resolve(&self, handle: Interned<T, D>) -> Option<&T> {
        self.items.get(handle.raw as usize)
    }

    /// Iterate over every `(handle, value)` pair in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (Interned<T, D>, &T)> {
        self.items
            .iter()
            .enumerate()
            .map(|(i, v)| (Interned::from_index(i as u32), v))
    }

    /// Reclaim the entire domain: drop every stored value and invalidate every
    /// previously issued handle. New handles start again from index 0.
    ///
    /// This is the design doc §24.6 "lifetime by domain" reclamation: a whole
    /// cache (one domain) is reset at once rather than reference-counting
    /// individual entries, which keeps handles stable for the cache's lifetime.
    pub fn clear(&mut self) {
        self.items.clear();
        self.buckets.clear();
    }

    /// Shrink the backing storage to fit the current contents.
    pub fn shrink_to_fit(&mut self) {
        self.items.shrink_to_fit();
        self.buckets.shrink_to_fit();
        for ids in self.buckets.values_mut() {
            ids.shrink_to_fit();
        }
    }
}

impl<T: Hash + Eq, D> InternCache<T, D> {
    /// The content-hash key the cache uses to bucket `value`.
    ///
    /// Exposed so callers can pre-compute or persist the content address; it is
    /// a classic FNV-1a hash and stable across runs of the same build.
    #[inline]
    #[must_use]
    pub fn content_hash(value: &T) -> u64 {
        stable_hash(value)
    }

    /// Intern `value`, returning its handle. Idempotent: equal content always
    /// maps to the same [`Interned`] and is stored exactly once.
    ///
    /// If an equal value is already present `value` is dropped and the existing
    /// handle is returned; otherwise `value` is stored.
    ///
    /// # Panics
    /// Panics if more than `u32::MAX` distinct values would be interned.
    pub fn intern(&mut self, value: T) -> Interned<T, D> {
        let hash = stable_hash(&value);
        if let Some(ids) = self.buckets.get(&hash) {
            for &id in ids {
                if self.items[id as usize] == value {
                    return Interned::from_index(id);
                }
            }
        }
        let id = u32::try_from(self.items.len())
            .expect("intern cache capacity exceeded (> u32::MAX distinct values)");
        self.items.push(value);
        self.buckets.entry(hash).or_default().push(id);
        Interned::from_index(id)
    }

    /// Return the handle for `value` if it has already been interned, without
    /// inserting it.
    pub fn get(&self, value: &T) -> Option<Interned<T, D>> {
        let hash = stable_hash(value);
        let ids = self.buckets.get(&hash)?;
        ids.iter()
            .copied()
            .find(|&id| &self.items[id as usize] == value)
            .map(Interned::from_index)
    }

    /// Whether `value` has already been interned.
    #[inline]
    pub fn contains(&self, value: &T) -> bool {
        self.get(value).is_some()
    }
}

impl<T: Hash + Eq + Clone, D> InternCache<T, D> {
    /// Intern `value` by reference, cloning it only if it is not already
    /// present. Returns the same handle as [`intern`](Self::intern) would.
    pub fn intern_ref(&mut self, value: &T) -> Interned<T, D> {
        if let Some(handle) = self.get(value) {
            return handle;
        }
        self.intern(value.clone())
    }
}

impl<T, D> Default for InternCache<T, D> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T, D> core::fmt::Debug for InternCache<T, D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("InternCache")
            .field("len", &self.items.len())
            .finish()
    }
}
