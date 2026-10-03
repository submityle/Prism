//! Small-vector optimization: inline storage that spills to the heap.

extern crate alloc;

use alloc::vec::Vec;
use core::fmt;

use crate::array_vec::ArrayVec;

/// A growable sequence that stores up to `N` elements inline (no allocation)
/// and transparently spills to a heap [`Vec`] once that capacity is exceeded.
///
/// This is the common engine pattern for "usually tiny, occasionally large"
/// collections (component lists, child indices, query terms). The
/// implementation is fully safe.
pub enum SmallVec<T, const N: usize> {
    /// All elements fit inline.
    Inline(ArrayVec<T, N>),
    /// Elements have spilled onto the heap.
    Heap(Vec<T>),
}

impl<T, const N: usize> SmallVec<T, N> {
    /// Create an empty `SmallVec` using inline storage.
    pub fn new() -> Self {
        SmallVec::Inline(ArrayVec::new())
    }

    /// Number of stored elements.
    pub fn len(&self) -> usize {
        match self {
            SmallVec::Inline(v) => v.len(),
            SmallVec::Heap(v) => v.len(),
        }
    }

    /// Whether the collection is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether the elements currently live on the heap.
    pub fn spilled(&self) -> bool {
        matches!(self, SmallVec::Heap(_))
    }

    /// Append `value`, spilling to the heap if inline capacity is exceeded.
    pub fn push(&mut self, value: T) {
        match self {
            SmallVec::Heap(v) => v.push(value),
            SmallVec::Inline(v) => {
                if let Err(value) = v.try_push(value) {
                    // Inline storage is full: migrate contents to a heap Vec
                    // (preserving order), then append the overflow element.
                    let mut tmp = ArrayVec::<T, N>::new();
                    core::mem::swap(v, &mut tmp);
                    let mut heap = into_iter_arrayvec(tmp);
                    heap.reserve(N);
                    heap.push(value);
                    *self = SmallVec::Heap(heap);
                }
            }
        }
    }

    /// Remove and return the last element, if any.
    pub fn pop(&mut self) -> Option<T> {
        match self {
            SmallVec::Inline(v) => v.pop(),
            SmallVec::Heap(v) => v.pop(),
        }
    }

    /// Borrow element `index`.
    pub fn get(&self, index: usize) -> Option<&T> {
        match self {
            SmallVec::Inline(v) => v.get(index),
            SmallVec::Heap(v) => v.get(index),
        }
    }

    /// Mutably borrow element `index`.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        match self {
            SmallVec::Inline(v) => v.get_mut(index),
            SmallVec::Heap(v) => v.get_mut(index),
        }
    }
}

/// Consume an [`ArrayVec`] yielding its elements in order.
fn into_iter_arrayvec<T, const N: usize>(mut v: ArrayVec<T, N>) -> Vec<T> {
    let mut out = Vec::with_capacity(v.len());
    // ArrayVec::pop removes from the back; collect then reverse to keep order.
    let mut rev = Vec::with_capacity(v.len());
    while let Some(item) = v.pop() {
        rev.push(item);
    }
    while let Some(item) = rev.pop() {
        out.push(item);
    }
    out
}

impl<T, const N: usize> Default for SmallVec<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: fmt::Debug, const N: usize> fmt::Debug for SmallVec<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut list = f.debug_list();
        for i in 0..self.len() {
            list.entry(&self.get(i).unwrap());
        }
        list.finish()
    }
}

impl<T: Clone, const N: usize> Clone for SmallVec<T, N> {
    fn clone(&self) -> Self {
        match self {
            SmallVec::Inline(v) => SmallVec::Inline(v.clone()),
            SmallVec::Heap(v) => SmallVec::Heap(v.clone()),
        }
    }
}
