//! An allocator-aware owning pointer ([`AllocBox`]).
//!
//! [`AllocBox`] is a minimal `Box`-like container that is generic over any
//! Prism [`Allocator`]. It demonstrates the roadmap's "容器带分配器" goal: a
//! container whose storage comes from a caller-chosen allocator (the global
//! heap, a [`Pool`](crate::alloc_::pool::Pool), or a
//! [`FrameAllocator`](crate::alloc_::frame::FrameAllocator)) rather than always
//! the global heap.

use core::alloc::Layout;
use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;

use super::{AllocError, Allocator};

/// A heap-like box whose single `T` is stored in memory obtained from an
/// [`Allocator`] `A`.
///
/// The box owns both the value and the allocator handle; on drop it runs the
/// value's destructor and returns the storage to the allocator. Passing a
/// shared reference (e.g. `&Pool`) as `A` lets many boxes share one allocator.
pub struct AllocBox<T, A: Allocator> {
    /// Pointer to the owned, initialized value.
    ptr: NonNull<T>,
    /// The allocator that owns the backing storage.
    alloc: A,
}

impl<T, A: Allocator> AllocBox<T, A> {
    /// Allocate storage from `alloc` and move `value` into it.
    ///
    /// Returns [`AllocError`] if the allocator cannot satisfy the request.
    pub fn try_new_in(value: T, alloc: A) -> Result<Self, AllocError> {
        let layout = Layout::new::<T>();
        let storage = alloc.allocate(layout)?;
        let ptr = storage.cast::<T>();
        #[expect(
            unsafe_code,
            reason = "initializing freshly allocated storage requires a raw write"
        )]
        // SAFETY: `allocate` returned a block of at least `size_of::<T>()` bytes
        // aligned to `align_of::<T>()` (zero-sized `T` yields a dangling-aligned
        // pointer, for which `write` is a valid no-op), so moving `value` in is
        // sound and does not read the uninitialized memory.
        unsafe {
            ptr.as_ptr().write(value);
        }
        Ok(Self { ptr, alloc })
    }

    /// Allocate storage from `alloc` and move `value` into it, panicking on
    /// allocation failure.
    ///
    /// # Panics
    /// Panics if `alloc` cannot satisfy the request.
    pub fn new_in(value: T, alloc: A) -> Self {
        match Self::try_new_in(value, alloc) {
            Ok(b) => b,
            Err(_) => panic!("AllocBox: allocation failed"),
        }
    }

    /// Borrow the owned value.
    pub fn get(&self) -> &T {
        #[expect(
            unsafe_code,
            reason = "dereferencing the owned, initialized value"
        )]
        // SAFETY: `ptr` points to a value initialized in `try_new_in` and not
        // yet dropped, and the borrow is tied to `&self`.
        unsafe {
            self.ptr.as_ref()
        }
    }

    /// Mutably borrow the owned value.
    pub fn get_mut(&mut self) -> &mut T {
        #[expect(
            unsafe_code,
            reason = "dereferencing the owned, initialized value"
        )]
        // SAFETY: `ptr` points to a live value and the exclusive borrow is tied
        // to `&mut self`, so no other reference can alias it.
        unsafe {
            self.ptr.as_mut()
        }
    }
}

impl<T, A: Allocator> Deref for AllocBox<T, A> {
    type Target = T;

    fn deref(&self) -> &T {
        self.get()
    }
}

impl<T, A: Allocator> DerefMut for AllocBox<T, A> {
    fn deref_mut(&mut self) -> &mut T {
        self.get_mut()
    }
}

impl<T, A: Allocator> Drop for AllocBox<T, A> {
    fn drop(&mut self) {
        #[expect(
            unsafe_code,
            reason = "dropping the value and freeing its backing storage"
        )]
        // SAFETY: `ptr` holds a live value allocated with `Layout::new::<T>()`
        // from `self.alloc`. We drop it in place exactly once, then return the
        // same block with the same layout to the same allocator.
        unsafe {
            core::ptr::drop_in_place(self.ptr.as_ptr());
            self.alloc
                .deallocate(self.ptr.cast::<u8>(), Layout::new::<T>());
        }
    }
}
