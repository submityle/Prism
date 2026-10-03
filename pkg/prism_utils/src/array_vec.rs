//! Fixed-capacity, stack-allocated vector (`no_std`, zero heap allocation).

use core::fmt;

/// A vector with a compile-time maximum capacity `N`, stored inline on the
/// stack. Pushing beyond `N` returns the value back via [`ArrayVec::try_push`]
/// (or panics via [`ArrayVec::push`]). This implementation is fully safe: it
/// uses `[Option<T>; N]` as backing storage rather than `MaybeUninit`.
pub struct ArrayVec<T, const N: usize> {
    storage: [Option<T>; N],
    len: usize,
}

impl<T, const N: usize> ArrayVec<T, N> {
    /// Create an empty `ArrayVec`.
    pub fn new() -> Self {
        Self {
            storage: core::array::from_fn(|_| None),
            len: 0,
        }
    }

    /// Maximum number of elements this vector can hold.
    pub const fn capacity(&self) -> usize {
        N
    }

    /// Number of stored elements.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the vector holds no elements.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether the vector is at capacity.
    pub const fn is_full(&self) -> bool {
        self.len == N
    }

    /// Append `value`, returning `Err(value)` if the vector is full.
    pub fn try_push(&mut self, value: T) -> Result<(), T> {
        if self.len == N {
            return Err(value);
        }
        self.storage[self.len] = Some(value);
        self.len += 1;
        Ok(())
    }

    /// Append `value`, panicking if the vector is full.
    pub fn push(&mut self, value: T) {
        if self.try_push(value).is_err() {
            panic!("ArrayVec capacity ({N}) exceeded");
        }
    }

    /// Remove and return the last element, if any.
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        self.len -= 1;
        self.storage[self.len].take()
    }

    /// Borrow element `index`.
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len {
            return None;
        }
        self.storage[index].as_ref()
    }

    /// Mutably borrow element `index`.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index >= self.len {
            return None;
        }
        self.storage[index].as_mut()
    }

    /// Remove every element.
    pub fn clear(&mut self) {
        for slot in self.storage.iter_mut().take(self.len) {
            *slot = None;
        }
        self.len = 0;
    }

    /// Iterate over the stored elements in order.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.storage.iter().take(self.len).filter_map(Option::as_ref)
    }
}

impl<T, const N: usize> Default for ArrayVec<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: fmt::Debug, const N: usize> fmt::Debug for ArrayVec<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T: Clone, const N: usize> Clone for ArrayVec<T, N> {
    fn clone(&self) -> Self {
        let mut out = ArrayVec::new();
        for item in self.iter() {
            out.push(item.clone());
        }
        out
    }
}
