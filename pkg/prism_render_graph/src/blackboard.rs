//! The frame blackboard: a typed, per-frame scratch map for sharing handles.
//!
//! Passes often need to hand resources to later passes without threading them
//! through every call site (the G-buffer a lighting pass needs, the shadow
//! atlas a dozen passes sample). The blackboard is a small type-keyed store:
//! one value per Rust type, set by a producer and fetched by consumers. It
//! holds plain data (typically [`crate::TextureHandle`] bundles), never GPU
//! objects, and is cleared each frame.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::any::{Any, TypeId};

/// A type-keyed store of per-frame values.
///
/// Keys are Rust types: [`set`](Self::set) replaces the value of type `T`, and
/// [`get`](Self::get) borrows it back. Backed by a small linear vector because
/// a frame rarely registers more than a handful of entries.
#[derive(Default)]
pub struct Blackboard {
    entries: Vec<(TypeId, Box<dyn Any>)>,
}

impl Blackboard {
    /// Creates an empty blackboard.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Stores `value` under its type `T`, replacing any previous value of that
    /// type and returning the old value if present.
    pub fn set<T: 'static>(&mut self, value: T) -> Option<T> {
        let key = TypeId::of::<T>();
        for entry in &mut self.entries {
            if entry.0 == key {
                let old = core::mem::replace(&mut entry.1, Box::new(value));
                return old.downcast::<T>().ok().map(|b| *b);
            }
        }
        self.entries.push((key, Box::new(value)));
        None
    }

    /// Borrows the stored value of type `T`, if any.
    #[must_use]
    pub fn get<T: 'static>(&self) -> Option<&T> {
        let key = TypeId::of::<T>();
        self.entries
            .iter()
            .find(|e| e.0 == key)
            .and_then(|e| e.1.downcast_ref::<T>())
    }

    /// Mutably borrows the stored value of type `T`, if any.
    pub fn get_mut<T: 'static>(&mut self) -> Option<&mut T> {
        let key = TypeId::of::<T>();
        self.entries
            .iter_mut()
            .find(|e| e.0 == key)
            .and_then(|e| e.1.downcast_mut::<T>())
    }

    /// Removes and returns the stored value of type `T`, if any.
    pub fn take<T: 'static>(&mut self) -> Option<T> {
        let key = TypeId::of::<T>();
        let pos = self.entries.iter().position(|e| e.0 == key)?;
        let (_, boxed) = self.entries.swap_remove(pos);
        boxed.downcast::<T>().ok().map(|b| *b)
    }

    /// Whether a value of type `T` is present.
    #[must_use]
    pub fn contains<T: 'static>(&self) -> bool {
        let key = TypeId::of::<T>();
        self.entries.iter().any(|e| e.0 == key)
    }

    /// The number of stored entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the blackboard holds no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Drops every stored value, readying the blackboard for the next frame.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}
