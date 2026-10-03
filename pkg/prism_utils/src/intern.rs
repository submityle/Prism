//! String interning with domain separation and `O(1)` handle comparison.
//!
//! Interning stores each distinct string exactly once and hands back a small
//! [`Istr`] handle (a 32-bit index). Comparing two interned strings becomes an
//! integer compare instead of a byte-by-byte `strcmp`, which is the standard
//! trick behind engine "names" (Unreal's `FName`, asset paths, debug labels).
//!
//! ## Domain separation
//! [`Istr<D>`] and [`Interner<D>`] are generic over an uninhabited *domain*
//! marker `D`. Handles from different domains are **distinct types**, so the
//! compiler rejects mixing an asset-path handle with a tag handle even though
//! both are `u32` under the hood. The [`domain`] module ships the common
//! markers; [`FName`] is a convenience alias for the default tag domain.
//!
//! ```
//! use prism_utils::intern::{Interner, domain};
//!
//! let mut tags: Interner<domain::Tag> = Interner::new();
//! let a = tags.intern("Player");
//! let b = tags.intern("Player");
//! assert_eq!(a, b); // same string -> same handle
//! assert_eq!(tags.resolve(a), "Player");
//! ```
//!
//! ## Identity contract
//! An [`Istr`] is only meaningful to the [`Interner`] that produced it.
//! Resolving a handle with a different interner instance (even of the same
//! domain) is a logic error; [`Interner::resolve`] panics on an out-of-range
//! handle and [`Interner::try_resolve`] returns `None`.

use core::marker::PhantomData;

use crate::hash::{stable_hash_str, FxBuildHasher, HashMap};

/// Uninhabited domain markers for [`Istr`]/[`Interner`].
///
/// Each marker is a separate type, so handles from different domains cannot be
/// confused at the type level. The markers are uninhabited (no values ever
/// exist); they live only in a [`PhantomData`] and cost nothing at runtime.
pub mod domain {
    /// General-purpose tags / names (the default engine "name" domain).
    #[derive(Debug)]
    pub enum Tag {}
    /// Filesystem- or asset-style paths.
    #[derive(Debug)]
    pub enum Path {}
    /// Human-readable debug labels.
    #[derive(Debug)]
    pub enum Debug {}
}

/// An interned-string handle: a cheap, `Copy`, domain-tagged `u32` index.
///
/// All trait impls are hand-written so they never require the domain marker
/// `D` to implement anything (the marker is purely a compile-time tag held in
/// `PhantomData<fn() -> D>`, which is always `Copy`/`Send`/`Sync`).
pub struct Istr<D = ()> {
    raw: u32,
    _marker: PhantomData<fn() -> D>,
}

impl<D> Istr<D> {
    /// Wrap a raw index. Used internally and for deserialization; the caller
    /// must guarantee the index came from a matching [`Interner`].
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

impl<D> Clone for Istr<D> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<D> Copy for Istr<D> {}

impl<D> PartialEq for Istr<D> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw
    }
}

impl<D> Eq for Istr<D> {}

impl<D> PartialOrd for Istr<D> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<D> Ord for Istr<D> {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.raw.cmp(&other.raw)
    }
}

impl<D> core::hash::Hash for Istr<D> {
    #[inline]
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.raw.hash(state);
    }
}

impl<D> core::fmt::Debug for Istr<D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Istr({})", self.raw)
    }
}

/// A convenience alias for the common tag/name domain, mirroring Unreal's
/// `FName`.
pub type FName = Istr<domain::Tag>;

/// A domain-separated string interner.
///
/// Backed by `Vec<Box<str>>` storage plus a hash bucket index keyed by the
/// crate's cross-run [`stable_hash_str`]. Lookups and inserts are amortized
/// `O(1)`; stored strings are never moved or freed for the interner's lifetime,
/// so returned handles stay valid.
pub struct Interner<D = ()> {
    strings: Vec<Box<str>>,
    buckets: HashMap<u64, Vec<u32>>,
    _marker: PhantomData<fn() -> D>,
}

impl<D> Interner<D> {
    /// Create an empty interner.
    #[inline]
    pub fn new() -> Self {
        Self {
            strings: Vec::new(),
            buckets: HashMap::default(),
            _marker: PhantomData,
        }
    }

    /// Create an empty interner with room for `cap` distinct strings.
    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            strings: Vec::with_capacity(cap),
            buckets: HashMap::with_capacity_and_hasher(cap, FxBuildHasher::default()),
            _marker: PhantomData,
        }
    }

    /// Number of distinct interned strings.
    #[inline]
    pub fn len(&self) -> usize {
        self.strings.len()
    }

    /// Whether no strings have been interned.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.strings.is_empty()
    }

    /// Intern `s`, returning its handle. Idempotent: equal strings always map
    /// to the same [`Istr`].
    ///
    /// # Panics
    /// Panics if more than `u32::MAX` distinct strings would be interned.
    pub fn intern(&mut self, s: &str) -> Istr<D> {
        let hash = stable_hash_str(s);
        // Immutable probe first; the borrow of `self.strings`/`self.buckets`
        // ends with this block, so the mutable insert below cannot conflict.
        if let Some(ids) = self.buckets.get(&hash) {
            for &id in ids {
                if &*self.strings[id as usize] == s {
                    return Istr::from_index(id);
                }
            }
        }
        let id = u32::try_from(self.strings.len())
            .expect("interner capacity exceeded (> u32::MAX distinct strings)");
        self.strings.push(Box::from(s));
        self.buckets.entry(hash).or_default().push(id);
        Istr::from_index(id)
    }

    /// Return the handle for `s` if it has already been interned, without
    /// inserting it.
    pub fn get(&self, s: &str) -> Option<Istr<D>> {
        let hash = stable_hash_str(s);
        let ids = self.buckets.get(&hash)?;
        ids.iter()
            .copied()
            .find(|&id| &*self.strings[id as usize] == s)
            .map(Istr::from_index)
    }

    /// Whether `s` has already been interned.
    #[inline]
    pub fn contains(&self, s: &str) -> bool {
        self.get(s).is_some()
    }

    /// Resolve a handle back to its string.
    ///
    /// # Panics
    /// Panics if `istr` did not come from this interner (out-of-range index).
    #[inline]
    pub fn resolve(&self, istr: Istr<D>) -> &str {
        &self.strings[istr.raw as usize]
    }

    /// Resolve a handle back to its string, returning `None` for an
    /// out-of-range handle instead of panicking.
    #[inline]
    pub fn try_resolve(&self, istr: Istr<D>) -> Option<&str> {
        self.strings.get(istr.raw as usize).map(|b| &**b)
    }

    /// Iterate over every `(handle, string)` pair in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (Istr<D>, &str)> {
        self.strings
            .iter()
            .enumerate()
            .map(|(i, s)| (Istr::from_index(i as u32), &**s))
    }
}

impl<D> Default for Interner<D> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<D> core::fmt::Debug for Interner<D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Interner")
            .field("len", &self.strings.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_strings_share_a_handle() {
        let mut i: Interner = Interner::new();
        let a = i.intern("hello");
        let b = i.intern("hello");
        let c = i.intern("world");
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(i.len(), 2);
    }

    #[test]
    fn resolve_round_trips() {
        let mut i: Interner = Interner::new();
        let a = i.intern("prism::Transform");
        assert_eq!(i.resolve(a), "prism::Transform");
        assert_eq!(i.try_resolve(a), Some("prism::Transform"));
        assert_eq!(i.try_resolve(Istr::from_index(999)), None);
    }

    #[test]
    fn get_does_not_insert() {
        let mut i: Interner = Interner::new();
        assert!(i.get("x").is_none());
        assert!(!i.contains("x"));
        let h = i.intern("x");
        assert_eq!(i.get("x"), Some(h));
        assert!(i.contains("x"));
    }

    #[test]
    fn handles_are_dense_and_ordered() {
        let mut i: Interner = Interner::new();
        let a = i.intern("a");
        let b = i.intern("b");
        assert_eq!(a.index(), 0);
        assert_eq!(b.index(), 1);
        assert!(a < b);
    }

    #[test]
    fn iter_yields_insertion_order() {
        let mut i: Interner = Interner::new();
        i.intern("first");
        i.intern("second");
        let collected: Vec<_> = i.iter().map(|(_, s)| s.to_string()).collect();
        assert_eq!(collected, ["first", "second"]);
    }

    #[test]
    fn domains_do_not_collide_in_the_type_system() {
        // Both interners allocate index 0 for their first string, but the
        // handles are different *types* and cannot be compared or swapped.
        let mut tags: Interner<domain::Tag> = Interner::new();
        let mut paths: Interner<domain::Path> = Interner::new();
        let t = tags.intern("Player");
        let p = paths.intern("/game/player");
        assert_eq!(t.index(), p.index()); // same raw index...
        assert_eq!(tags.resolve(t), "Player");
        assert_eq!(paths.resolve(p), "/game/player");
    }

    #[test]
    fn fname_alias_works() {
        let mut names: Interner<domain::Tag> = Interner::new();
        let n: FName = names.intern("Health");
        assert_eq!(names.resolve(n), "Health");
    }

    #[test]
    fn handle_is_pointer_sized_and_copy() {
        assert_eq!(size_of::<Istr>(), size_of::<u32>());
        let mut i: Interner = Interner::new();
        let a = i.intern("copy-me");
        let b = a; // Copy, not move
        assert_eq!(a, b);
    }
}
