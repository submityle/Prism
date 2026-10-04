//! Deterministic frame-name interning for the sampling profiler (§24.5).
//!
//! A statistical profiler captures the same function names in thousands of
//! stacks, so storing them as owned strings per sample is wasteful and makes
//! equality comparisons slow. [`SymbolTable`] interns each distinct frame name
//! to a small [`FrameId`] in first-seen order and resolves it back on demand.
//!
//! Interning is deterministic: identical name sequences always produce
//! identical ids, so two independently collected profiles fold and compare
//! bit-for-bit. This is pure `core`/`alloc` (no `unsafe`, `no_std` + `alloc`).

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

/// A stable, compact identifier for an interned frame (function) name.
///
/// Ids are assigned in first-seen order starting at `0`, so a given name
/// sequence always yields the same ids. Ordering by [`FrameId`] therefore
/// matches insertion order, which keeps folded output deterministic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FrameId(pub u32);

impl FrameId {
    /// The raw index backing this id.
    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// A deterministic frame-name interner mapping names to [`FrameId`]s.
///
/// Lookups use a linear scan paired with the stored name vector. Profiler
/// symbol counts are modest (hundreds to low thousands of distinct frames) and
/// a scan keeps the type `no_std`-friendly without a hash map; callers that
/// intern in bulk should reuse the returned [`FrameId`] rather than re-intern.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SymbolTable {
    /// Interned names, indexed by [`FrameId`].
    names: Vec<String>,
}

impl SymbolTable {
    /// An empty table.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self { names: Vec::new() }
    }

    /// Intern `name`, returning its [`FrameId`]. Re-interning an existing name
    /// returns the same id.
    pub fn intern(&mut self, name: &str) -> FrameId {
        if let Some(index) = self.names.iter().position(|existing| existing == name) {
            return FrameId(index as u32);
        }
        let id = FrameId(self.names.len() as u32);
        self.names.push(String::from(name));
        id
    }

    /// Look up an already-interned name without inserting it.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<FrameId> {
        self.names
            .iter()
            .position(|existing| existing == name)
            .map(|index| FrameId(index as u32))
    }

    /// Resolve a [`FrameId`] back to its name, or `None` if out of range.
    #[inline]
    #[must_use]
    pub fn resolve(&self, id: FrameId) -> Option<&str> {
        self.names.get(id.index()).map(String::as_str)
    }

    /// Number of distinct interned names.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Whether no names have been interned.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}
