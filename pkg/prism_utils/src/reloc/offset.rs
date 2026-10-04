//! Self-relative offset-pointer primitives: [`OffsetPtr`] and [`OffsetSlice`].
//!
//! The stored offset is measured from the byte position of the field itself
//! (not from the blob start), so a struct that embeds an [`OffsetPtr`] relocates
//! together with whatever it points at as long as both move by the same amount
//! — which is exactly what a whole-blob `memcpy`/`mmap` does. Resolution is
//! always bounds-checked against the backing blob.

use core::marker::PhantomData;

use super::{resolve_pos, Reloc, RelocError};

/// A self-relative pointer to a single [`Reloc`] value inside a blob.
///
/// Stored as an `i32` byte offset from the pointer's own position. A zero
/// offset is treated as a *null* pointer (it would otherwise point at itself,
/// which is never a valid target), so [`get`](OffsetPtr::get) returns `None`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct OffsetPtr<T> {
    offset: i32,
    marker: PhantomData<fn() -> T>,
}

impl<T> OffsetPtr<T> {
    /// The null offset pointer.
    pub const NULL: Self = Self {
        offset: 0,
        marker: PhantomData,
    };

    /// Construct an offset pointer from a raw self-relative byte offset.
    #[inline]
    #[must_use]
    pub const fn from_raw(offset: i32) -> Self {
        Self {
            offset,
            marker: PhantomData,
        }
    }

    /// The raw self-relative byte offset.
    #[inline]
    #[must_use]
    pub const fn raw(self) -> i32 {
        self.offset
    }

    /// Whether this pointer is null (a zero self-relative offset).
    #[inline]
    #[must_use]
    pub const fn is_null(self) -> bool {
        self.offset == 0
    }
}

impl<T: Reloc> OffsetPtr<T> {
    /// Resolve and decode the pointee against `blob`, given the byte position
    /// `field_pos` at which this pointer is stored.
    ///
    /// Returns `Ok(None)` when the pointer is [null](OffsetPtr::is_null), and a
    /// [`RelocError`] when the target runs outside `blob`.
    pub fn get(self, blob: &[u8], field_pos: usize) -> Result<Option<T>, RelocError> {
        if self.is_null() {
            return Ok(None);
        }
        let start = resolve_pos(field_pos, self.offset)?;
        let end = start.checked_add(T::SIZE).ok_or(RelocError::Overflow)?;
        let bytes = blob.get(start..end).ok_or(RelocError::OutOfBounds)?;
        Ok(Some(T::decode(bytes)))
    }
}

/// A self-relative pointer to a contiguous run of `len` [`Reloc`] values.
///
/// Stored as an `i32` byte offset (from this field's own position) to the first
/// element plus a `u32` element count. This is the primitive a relocatable
/// container uses to address a variable-length region of its blob.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct OffsetSlice<T> {
    offset: i32,
    len: u32,
    marker: PhantomData<fn() -> T>,
}

impl<T> OffsetSlice<T> {
    /// The empty slice (zero length, null offset).
    pub const EMPTY: Self = Self {
        offset: 0,
        len: 0,
        marker: PhantomData,
    };

    /// Construct a slice from a raw self-relative byte offset and element count.
    #[inline]
    #[must_use]
    pub const fn from_raw(offset: i32, len: u32) -> Self {
        Self {
            offset,
            len,
            marker: PhantomData,
        }
    }

    /// The raw self-relative byte offset to the first element.
    #[inline]
    #[must_use]
    pub const fn raw_offset(self) -> i32 {
        self.offset
    }

    /// The number of elements addressed by this slice.
    #[inline]
    #[must_use]
    pub const fn len(self) -> usize {
        self.len as usize
    }

    /// Whether the slice is empty.
    #[inline]
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }
}

impl<T: Reloc> OffsetSlice<T> {
    /// The absolute byte position of element 0, given this field's position.
    #[inline]
    pub(crate) fn base(self, field_pos: usize) -> Result<usize, RelocError> {
        resolve_pos(field_pos, self.offset)
    }

    /// Resolve and decode element `index`, given this field's position.
    ///
    /// Returns `Ok(None)` when `index` is out of range, and a [`RelocError`]
    /// when the computed region runs outside `blob`.
    pub fn get(
        self,
        blob: &[u8],
        field_pos: usize,
        index: usize,
    ) -> Result<Option<T>, RelocError> {
        if index >= self.len() {
            return Ok(None);
        }
        let base = self.base(field_pos)?;
        let start = base
            .checked_add(index.checked_mul(T::SIZE).ok_or(RelocError::Overflow)?)
            .ok_or(RelocError::Overflow)?;
        let end = start.checked_add(T::SIZE).ok_or(RelocError::Overflow)?;
        let bytes = blob.get(start..end).ok_or(RelocError::OutOfBounds)?;
        Ok(Some(T::decode(bytes)))
    }

    /// Validate that the whole slice region lies within `blob`.
    ///
    /// A view that passes this check can index any element in `0..len` without
    /// a per-element bounds failure.
    pub(crate) fn validate(self, blob: &[u8], field_pos: usize) -> Result<(), RelocError> {
        if self.len == 0 {
            return Ok(());
        }
        let base = self.base(field_pos)?;
        let span = self
            .len()
            .checked_mul(T::SIZE)
            .ok_or(RelocError::Overflow)?;
        let end = base.checked_add(span).ok_or(RelocError::Overflow)?;
        if end > blob.len() {
            return Err(RelocError::OutOfBounds);
        }
        Ok(())
    }
}

impl<T: Reloc> Reloc for OffsetPtr<T> {
    const SIZE: usize = 4;

    #[inline]
    fn encode(&self, dst: &mut [u8]) {
        dst[..4].copy_from_slice(&self.offset.to_le_bytes());
    }

    #[inline]
    fn decode(src: &[u8]) -> Self {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&src[..4]);
        Self::from_raw(i32::from_le_bytes(buf))
    }
}

impl<T: Reloc> Reloc for OffsetSlice<T> {
    const SIZE: usize = 8;

    #[inline]
    fn encode(&self, dst: &mut [u8]) {
        dst[..4].copy_from_slice(&self.offset.to_le_bytes());
        dst[4..8].copy_from_slice(&self.len.to_le_bytes());
    }

    #[inline]
    fn decode(src: &[u8]) -> Self {
        let mut off = [0u8; 4];
        off.copy_from_slice(&src[..4]);
        let mut len = [0u8; 4];
        len.copy_from_slice(&src[4..8]);
        Self::from_raw(i32::from_le_bytes(off), u32::from_le_bytes(len))
    }
}
