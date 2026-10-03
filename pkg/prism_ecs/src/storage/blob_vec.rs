//! A type-erased, growable column of equally-sized values — the physical
//! backbone of the columnar (Structure-of-Arrays) storage.
//!
//! A [`BlobVec`] is like `Vec<T>` with `T` erased to a runtime [`Layout`] plus
//! an optional [`DropFn`]. It owns its elements: dropping the [`BlobVec`] drops
//! every live element. All element access is `unsafe` because the type is
//! erased — callers must pass pointers to values of the exact type the column
//! was created for.

use alloc::alloc::{alloc, dealloc, handle_alloc_error, realloc};
use core::alloc::Layout;
use core::ptr::NonNull;

use crate::component::DropFn;

/// A type-erased, owning, growable array of `item_layout`-sized values.
pub struct BlobVec {
    item_layout: Layout,
    /// Allocated capacity in elements. For zero-sized items this is
    /// `usize::MAX` and no allocation is performed.
    capacity: usize,
    len: usize,
    /// Pointer to the backing allocation. Dangling (but aligned) when
    /// `capacity == 0` or the item is zero-sized.
    data: NonNull<u8>,
    drop: Option<DropFn>,
}

// SAFETY: `BlobVec` only stores component data for `Component: Send + Sync`
// types (or dynamically-registered data the caller promises is `Send + Sync`),
// so transferring/aliasing the column across threads is as safe as for the
// underlying values. The raw pointer is an owning allocation, not shared state.
unsafe impl Send for BlobVec {}
// SAFETY: see the `Send` impl above.
unsafe impl Sync for BlobVec {}

impl BlobVec {
    /// Create an empty column for values described by `item_layout`, dropped by
    /// `drop` (or none if the type needs no drop).
    pub fn new(item_layout: Layout, drop: Option<DropFn>) -> Self {
        let capacity = if item_layout.size() == 0 {
            usize::MAX
        } else {
            0
        };
        Self {
            item_layout,
            capacity,
            len: 0,
            // A dangling-but-aligned pointer is valid for zero-sized reads and
            // for an empty, never-dereferenced allocation.
            data: dangling_aligned(item_layout.align()),
            drop,
        }
    }

    /// Number of live elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the column holds no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The per-element layout.
    #[inline]
    pub fn item_layout(&self) -> Layout {
        self.item_layout
    }

    /// Compute the layout of a backing array of `n` elements.
    fn array_layout(&self, n: usize) -> Layout {
        // Rust type layouts always have `size` a multiple of `align`, so the
        // element stride equals the size and the array size is `size * n`.
        let size = self
            .item_layout
            .size()
            .checked_mul(n)
            .expect("column allocation size overflow");
        Layout::from_size_align(size, self.item_layout.align())
            .expect("invalid column array layout")
    }

    /// Ensure there is capacity for at least `additional` more elements.
    pub fn reserve(&mut self, additional: usize) {
        let required = self.len + additional;
        if required <= self.capacity {
            return;
        }
        // Zero-sized items have `capacity == usize::MAX`; we can never get here.
        debug_assert!(self.item_layout.size() != 0);
        // Amortised growth: at least double, at least `required`, start at 4.
        let new_cap = required.max(self.capacity.saturating_mul(2)).max(4);
        self.grow_exact(new_cap);
    }

    /// Grow the backing allocation to exactly `new_cap` elements.
    fn grow_exact(&mut self, new_cap: usize) {
        debug_assert!(new_cap > self.capacity);
        let new_layout = self.array_layout(new_cap);
        let new_ptr = if self.capacity == 0 {
            // SAFETY: `new_layout` has non-zero size (item is non-ZST and
            // `new_cap > 0`), as required by `alloc`.
            unsafe { alloc(new_layout) }
        } else {
            let old_layout = self.array_layout(self.capacity);
            // SAFETY: `self.data` was allocated with `old_layout` by a previous
            // `alloc`/`realloc`, and `new_layout.size()` is non-zero and larger.
            unsafe { realloc(self.data.as_ptr(), old_layout, new_layout.size()) }
        };
        self.data = NonNull::new(new_ptr).unwrap_or_else(|| handle_alloc_error(new_layout));
        self.capacity = new_cap;
    }

    /// Pointer to the start of element `index` (may be past the live length;
    /// used for writing a freshly-reserved slot).
    ///
    /// # Safety
    /// `index` must be `<= capacity` and the returned pointer must only be used
    /// for `item_layout`-typed accesses.
    #[inline]
    pub unsafe fn get_ptr(&self, index: usize) -> *mut u8 {
        // SAFETY: the caller guarantees `index <= capacity`; for ZSTs `size` is
        // 0 so the offset is 0 and the dangling pointer is returned unchanged.
        unsafe { self.data.as_ptr().add(index * self.item_layout.size()) }
    }

    /// Append a value by copying `item_layout.size()` bytes from `value`.
    ///
    /// Ownership of the value is transferred into the column (a bitwise move);
    /// the caller must not drop the source afterwards.
    ///
    /// # Safety
    /// `value` must point to a valid, initialized value of the exact type this
    /// column was created for.
    pub unsafe fn push(&mut self, value: *const u8) {
        self.reserve(1);
        let index = self.len;
        // SAFETY: we just reserved space, so `index < capacity`; `value` is a
        // valid source of `size` bytes per the caller's contract, and the dest
        // slot is uninitialized so a bitwise copy establishes a valid value.
        unsafe {
            let dst = self.get_ptr(index);
            core::ptr::copy_nonoverlapping(value, dst, self.item_layout.size());
        }
        self.len = index + 1;
    }

    /// Remove the element at `index` by swapping the last element into its
    /// place, dropping the removed element.
    ///
    /// # Safety
    /// `index < len`.
    pub unsafe fn swap_remove_and_drop(&mut self, index: usize) {
        debug_assert!(index < self.len);
        let size = self.item_layout.size();
        // SAFETY: `index < len <= capacity`, so the pointer is in-bounds.
        let removed = unsafe { self.get_ptr(index) };
        if let Some(drop) = self.drop {
            // SAFETY: `removed` points at a valid, initialized element; dropping
            // it exactly once (and not using it afterwards) is sound.
            unsafe { drop(removed) };
        }
        let last = self.len - 1;
        if index != last {
            // SAFETY: both `index` and `last` are in-bounds; the source slot is
            // being logically removed so moving its bytes over the (already
            // dropped) `index` slot leaves exactly one owner.
            unsafe {
                let src = self.get_ptr(last);
                core::ptr::copy_nonoverlapping(src, removed, size);
            }
        }
        self.len = last;
    }

    /// Remove the element at `index` by swapping the last element into its
    /// place, copying the removed element's bytes to `dst` (transferring
    /// ownership to the caller) instead of dropping it.
    ///
    /// This is how a component value is moved from one archetype's column to
    /// another's during a structural change.
    ///
    /// # Safety
    /// `index < len`, and `dst` must be valid for writes of
    /// `item_layout.size()` bytes and properly aligned. The caller takes
    /// ownership of the value written to `dst`.
    pub unsafe fn swap_remove_and_copy_out(&mut self, index: usize, dst: *mut u8) {
        debug_assert!(index < self.len);
        let size = self.item_layout.size();
        // SAFETY: `index < len`, so in-bounds.
        let removed = unsafe { self.get_ptr(index) };
        // SAFETY: `removed` is a valid initialized element and `dst` is valid
        // for `size` bytes per the caller's contract; this moves the value out.
        unsafe { core::ptr::copy_nonoverlapping(removed, dst, size) };
        let last = self.len - 1;
        if index != last {
            // SAFETY: see `swap_remove_and_drop`; both slots are in-bounds.
            unsafe {
                let src = self.get_ptr(last);
                core::ptr::copy_nonoverlapping(src, removed, size);
            }
        }
        self.len = last;
    }

    /// Drop all live elements, leaving the column empty (allocation retained).
    pub fn clear(&mut self) {
        if let Some(drop) = self.drop {
            for i in 0..self.len {
                // SAFETY: `i < len`, so each pointer is a valid, initialized
                // element dropped exactly once.
                unsafe { drop(self.get_ptr(i)) };
            }
        }
        self.len = 0;
    }
}

impl BlobVec {
    /// Relocate the element at `src_index` out of `src` and append it to
    /// `self`. The value is bitwise-copied, transferring ownership into `self`;
    /// the source slot is **not** removed or shrunk here, so after this call
    /// `src` holds a logically moved-out duplicate at `src_index` that must be
    /// reconciled by the caller via [`BlobVec::swap_remove_forget`] (never
    /// dropped). Used to relocate a component value from one archetype column
    /// to another during a structural change.
    ///
    /// Leaving the source slot in place lets the owning table swap-remove every
    /// column (moved and unmoved alike) in one consistent pass, preserving the
    /// invariant that all columns and the entity list shrink against the same
    /// `last` row.
    ///
    /// # Safety
    /// `self` and `src` must have identical item layouts, and
    /// `src_index < src.len()`.
    pub unsafe fn push_from(&mut self, src: &BlobVec, src_index: usize) {
        debug_assert_eq!(self.item_layout.size(), src.item_layout.size());
        debug_assert_eq!(self.item_layout.align(), src.item_layout.align());
        debug_assert!(src_index < src.len);
        self.reserve(1);
        // SAFETY: we reserved a slot, so `self.len` is a valid write target;
        // `src_index < src.len()` per the contract, and the layouts match so
        // the copied bytes form a valid value in `self`. The source slot keeps
        // its bits (a moved-out duplicate) for the caller to forget later.
        unsafe {
            let dst = self.get_ptr(self.len);
            let src_ptr = src.get_ptr(src_index);
            core::ptr::copy_nonoverlapping(src_ptr, dst, self.item_layout.size());
        }
        self.len += 1;
    }
}

impl BlobVec {
    /// Overwrite the element at `index` with the value at `value`, dropping
    /// the previous element. Used for last-wins component insertion.
    ///
    /// # Safety
    /// `index < len()`, and `value` points to a valid, initialized value of
    /// this column's type whose ownership transfers into the column.
    pub unsafe fn replace(&mut self, index: usize, value: *const u8) {
        debug_assert!(index < self.len);
        // SAFETY: `index < len`, so the slot is a valid initialized element.
        let slot = unsafe { self.get_ptr(index) };
        if let Some(drop) = self.drop {
            // SAFETY: `slot` is a valid initialized element dropped exactly once.
            unsafe { drop(slot) };
        }
        // SAFETY: `value` is a valid source of `size` bytes; the slot is now
        // logically uninitialized, so the copy establishes a new valid value.
        unsafe { core::ptr::copy_nonoverlapping(value, slot, self.item_layout.size()) };
    }

    /// Remove the element at `index` by moving the last element over it,
    /// WITHOUT dropping the element at `index`.
    ///
    /// Used after the value at `index` was already moved out via
    /// [`BlobVec::swap_remove_and_copy_out`] or [`BlobVec::push_from`]; running
    /// drop glue again would be a double free.
    ///
    /// # Safety
    /// `index < len()`, and the element at `index` must already have been
    /// moved out (its ownership transferred elsewhere).
    pub unsafe fn swap_remove_forget(&mut self, index: usize) {
        debug_assert!(index < self.len);
        let last = self.len - 1;
        if index != last {
            // SAFETY: both `index` and `last` are in-bounds; the `index` slot is
            // logically uninitialized (moved out), so overwriting it with the
            // last element and shrinking leaves exactly one owner.
            unsafe {
                let src = self.get_ptr(last);
                let dst = self.get_ptr(index);
                core::ptr::copy_nonoverlapping(src, dst, self.item_layout.size());
            }
        }
        self.len = last;
    }
}

impl Drop for BlobVec {
    fn drop(&mut self) {
        self.clear();
        if self.capacity != 0 && self.item_layout.size() != 0 {
            let layout = self.array_layout(self.capacity);
            // SAFETY: `self.data` was allocated with this exact layout and has
            // not been freed; `capacity != 0` and non-ZST guarantee a real
            // allocation.
            unsafe { dealloc(self.data.as_ptr(), layout) };
        }
    }
}

/// A dangling pointer with the given alignment, valid as an empty/ZST base.
fn dangling_aligned(align: usize) -> NonNull<u8> {
    debug_assert!(align.is_power_of_two());
    // SAFETY: a power-of-two alignment is a non-zero address, valid for a
    // dangling `NonNull` that is never dereferenced for a real read/write.
    unsafe { NonNull::new_unchecked(align as *mut u8) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::rc::Rc;
    use alloc::vec::Vec;
    use core::cell::Cell;

    fn layout_of<T>() -> Layout {
        Layout::new::<T>()
    }

    /// Drop glue used by the column tests.
    ///
    /// # Safety
    /// `p` must point at a valid, initialized `T`.
    /// # Safety
    /// `p` must point at a valid, initialized `T` that is dropped exactly once.
    unsafe fn drop_t<T>(p: *mut u8) {
        // SAFETY: tests only push valid `T` values.
        unsafe { core::ptr::drop_in_place(p.cast::<T>()) }
    }

    #[test]
    fn push_and_read_u32() {
        let mut v = BlobVec::new(layout_of::<u32>(), None);
        for i in 0u32..10 {
            // SAFETY: pushing a valid u32.
            unsafe { v.push((&i as *const u32).cast::<u8>()) };
        }
        assert_eq!(v.len(), 10);
        for i in 0..10usize {
            // SAFETY: index in-bounds, column holds u32.
            let val = unsafe { *v.get_ptr(i).cast::<u32>() };
            assert_eq!(val, i as u32);
        }
    }

    #[test]
    fn swap_remove_moves_last_into_hole() {
        let mut v = BlobVec::new(layout_of::<u32>(), None);
        for i in 0u32..5 {
            // SAFETY: valid u32.
            unsafe { v.push((&i as *const u32).cast::<u8>()) };
        }
        // Remove index 1 (value 1); last (value 4) should move into slot 1.
        // SAFETY: index < len.
        unsafe { v.swap_remove_and_drop(1) };
        assert_eq!(v.len(), 4);
        let got: Vec<u32> = (0..4)
            // SAFETY: in-bounds u32 reads.
            .map(|i| unsafe { *v.get_ptr(i).cast::<u32>() })
            .collect();
        assert_eq!(got, alloc::vec![0, 4, 2, 3]);
    }

    #[test]
    fn drop_glue_runs_for_every_element() {
        let counter = Rc::new(Cell::new(0usize));
        struct Guard(Rc<Cell<usize>>);
        impl Drop for Guard {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }
        {
            let mut v = BlobVec::new(layout_of::<Guard>(), Some(drop_t::<Guard> as DropFn));
            for _ in 0..3 {
                let g = core::mem::ManuallyDrop::new(Guard(counter.clone()));
                // SAFETY: moving a valid Guard into the column; source is
                // wrapped in ManuallyDrop so it is not double-dropped.
                unsafe { v.push((&*g as *const Guard).cast::<u8>()) };
            }
            // SAFETY: index < len. Dropping element 0 should run Guard::drop.
            unsafe { v.swap_remove_and_drop(0) };
            assert_eq!(counter.get(), 1);
        }
        // Dropping the column drops the remaining 2 guards.
        assert_eq!(counter.get(), 3);
    }

    #[test]
    fn copy_out_transfers_ownership_without_drop() {
        let counter = Rc::new(Cell::new(0usize));
        struct Guard(#[allow(dead_code)] Rc<Cell<usize>>);
        impl Drop for Guard {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }
        let mut v = BlobVec::new(layout_of::<Guard>(), Some(drop_t::<Guard> as DropFn));
        let g = core::mem::ManuallyDrop::new(Guard(counter.clone()));
        // SAFETY: valid Guard, not double-dropped (ManuallyDrop source).
        unsafe { v.push((&*g as *const Guard).cast::<u8>()) };

        let mut out = core::mem::MaybeUninit::<Guard>::uninit();
        // SAFETY: index 0 < len; `out` is valid for writes of a Guard.
        unsafe { v.swap_remove_and_copy_out(0, out.as_mut_ptr().cast::<u8>()) };
        assert_eq!(v.len(), 0);
        // No drop happened during copy_out.
        assert_eq!(counter.get(), 0);
        // SAFETY: `out` was initialized by copy_out.
        let owned = unsafe { out.assume_init() };
        drop(owned);
        assert_eq!(counter.get(), 1);
    }

    #[test]
    fn zero_sized_items() {
        struct Zst;
        let mut v = BlobVec::new(layout_of::<Zst>(), None);
        for _ in 0..1000 {
            let z = Zst;
            // SAFETY: ZST push copies 0 bytes.
            unsafe { v.push((&z as *const Zst).cast::<u8>()) };
        }
        assert_eq!(v.len(), 1000);
        // SAFETY: index < len.
        unsafe { v.swap_remove_and_drop(0) };
        assert_eq!(v.len(), 999);
    }
}
