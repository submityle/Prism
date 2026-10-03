//! A copy-on-write (`CoW`) container wrapper: [`Cow`], an `Arc`-backed value
//! that is cheap to clone (shared, refcount bump) and only pays for a deep copy
//! on the first mutation of a shared value.
//!
//! # Why `CoW`
//! Assets and configuration are often read by many consumers but mutated
//! rarely. [`Cow`] lets every reader hold a clone for the price of an atomic
//! increment; the backing data is physically shared until someone calls
//! [`make_mut`](Cow::make_mut), at which point *that* holder (and only if the
//! value is still shared) transparently gets its own private copy. This is the
//! "多读者零拷贝共享" form from the design doc §17.
//!
//! ```
//! use prism_utils::cow::Cow;
//!
//! let a = Cow::new(vec![1, 2, 3]);
//! let b = a.clone();
//! assert!(Cow::ptr_eq(&a, &b)); // physically shared, no copy yet
//!
//! let mut b = b;
//! b.make_mut().push(4); // b diverges here
//! assert!(!Cow::ptr_eq(&a, &b));
//! assert_eq!(&*a, &[1, 2, 3]); // a is untouched
//! assert_eq!(&*b, &[1, 2, 3, 4]);
//! ```

extern crate alloc;

use alloc::sync::Arc;
use core::fmt;
use core::ops::Deref;

/// A clone-cheap, copy-on-write wrapper around a `T`.
///
/// Cloning a `Cow` shares the backing allocation (an atomic refcount bump).
/// Mutating through [`make_mut`](Cow::make_mut) copies the value out first iff
/// it is currently shared, so no other holder ever observes the mutation.
pub struct Cow<T> {
    inner: Arc<T>,
}

impl<T> Cow<T> {
    /// Wrap `value`, taking sole ownership of a fresh backing allocation.
    #[must_use]
    pub fn new(value: T) -> Self {
        Self {
            inner: Arc::new(value),
        }
    }

    /// Borrow the shared value.
    #[must_use]
    #[inline]
    pub fn get(&self) -> &T {
        &self.inner
    }

    /// Returns `true` if more than one [`Cow`] currently shares this backing
    /// allocation (so the next [`make_mut`](Cow::make_mut) would copy).
    #[must_use]
    #[inline]
    pub fn is_shared(&self) -> bool {
        Arc::strong_count(&self.inner) > 1
    }

    /// The number of [`Cow`] handles sharing this backing allocation.
    #[must_use]
    #[inline]
    pub fn ref_count(&self) -> usize {
        Arc::strong_count(&self.inner)
    }

    /// Returns `true` if `a` and `b` point at the *same* backing allocation
    /// (i.e. one was cloned from the other and neither has diverged yet).
    #[must_use]
    #[inline]
    pub fn ptr_eq(a: &Self, b: &Self) -> bool {
        Arc::ptr_eq(&a.inner, &b.inner)
    }

    /// Try to borrow the value exclusively *without* copying, succeeding only
    /// if this handle is the sole owner. Returns `None` if the value is shared.
    #[must_use]
    pub fn get_mut(&mut self) -> Option<&mut T> {
        Arc::get_mut(&mut self.inner)
    }
}

impl<T: Clone> Cow<T> {
    /// Borrow the value exclusively, copying it out first iff it is currently
    /// shared. After this call this handle is guaranteed to be the sole owner,
    /// so the mutation is private to it (copy-on-write).
    #[inline]
    pub fn make_mut(&mut self) -> &mut T {
        Arc::make_mut(&mut self.inner)
    }

    /// Consume the wrapper and return the owned value, cloning out of the
    /// shared allocation only if it is still shared.
    #[must_use]
    pub fn into_inner(self) -> T {
        Arc::try_unwrap(self.inner).unwrap_or_else(|arc| (*arc).clone())
    }
}

impl<T> Clone for Cow<T> {
    /// Cheap: shares the backing allocation via an atomic refcount bump; no
    /// `T` is copied.
    #[inline]
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T> Deref for Cow<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        &self.inner
    }
}

impl<T: Default> Default for Cow<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T: fmt::Debug> fmt::Debug for Cow<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Cow").field(&*self.inner).finish()
    }
}

impl<T: PartialEq> PartialEq for Cow<T> {
    fn eq(&self, other: &Self) -> bool {
        // Fast path: identical allocation is trivially equal.
        Arc::ptr_eq(&self.inner, &other.inner) || *self.inner == *other.inner
    }
}

impl<T: Eq> Eq for Cow<T> {}

impl<T> From<T> for Cow<T> {
    fn from(value: T) -> Self {
        Self::new(value)
    }
}
