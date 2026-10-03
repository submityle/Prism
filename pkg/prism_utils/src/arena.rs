//! A typed, index-based arena (safe "slab" allocation).

extern crate alloc;

use alloc::vec::Vec;

/// A stable handle into an [`Arena`]. Indices remain valid for the lifetime of
/// the arena (this M0 arena does not reuse slots or track generations; that is
/// the job of the later `SlotMap`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ArenaIndex(usize);

impl ArenaIndex {
    /// Raw numeric index.
    pub const fn raw(self) -> usize {
        self.0
    }
}

/// A growable arena that owns its elements and hands out stable
/// [`ArenaIndex`] handles. This is a fully safe bump/arena substitute for M0;
/// a lock-free byte-bump allocator lands in a later milestone.
pub struct Arena<T> {
    items: Vec<T>,
}

impl<T> Arena<T> {
    /// Create an empty arena.
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    /// Create an arena with pre-reserved capacity.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            items: Vec::with_capacity(cap),
        }
    }

    /// Insert `value`, returning its stable handle.
    pub fn insert(&mut self, value: T) -> ArenaIndex {
        let idx = self.items.len();
        self.items.push(value);
        ArenaIndex(idx)
    }

    /// Borrow the value at `index`.
    pub fn get(&self, index: ArenaIndex) -> Option<&T> {
        self.items.get(index.0)
    }

    /// Mutably borrow the value at `index`.
    pub fn get_mut(&mut self, index: ArenaIndex) -> Option<&mut T> {
        self.items.get_mut(index.0)
    }

    /// Number of stored elements.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether the arena is empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Iterate over `(index, &value)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (ArenaIndex, &T)> {
        self.items
            .iter()
            .enumerate()
            .map(|(i, v)| (ArenaIndex(i), v))
    }
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Self::new()
    }
}
