//! [`RelocVec`]: a relocatable, length-prefixed vector of [`Reloc`] values.
//!
//! The container owns one contiguous little-endian blob laid out as a fixed
//! 16-byte header followed by the packed elements:
//!
//! ```text
//! +--------+-----------+-------------------------------+------------------+
//! | magic  | elem_size |   data: OffsetSlice<T>        |   elements ...   |
//! | u32 @0 | u32 @4    | i32 off @8 | u32 len @12      |   @16            |
//! +--------+-----------+-------------------------------+------------------+
//! ```
//!
//! The element region is addressed through an [`OffsetSlice`] whose offset is
//! self-relative (points from byte 8 to byte 16), so the entire blob is
//! position-independent: copy the bytes anywhere and [`RelocVecView`] reads them
//! back without a fix-up pass.

extern crate alloc;

use alloc::vec::Vec;
use core::marker::PhantomData;

use super::offset::OffsetSlice;
use super::{read_u32, Reloc, RelocError};

/// Magic tag at the start of a [`RelocVec`] blob (`b"RVC1"`, little-endian).
const MAGIC: u32 = 0x3143_5652;
/// Fixed header length in bytes.
const HEADER_LEN: usize = 16;
/// Byte position of the data [`OffsetSlice`] field.
const DATA_FIELD_POS: usize = 8;
/// Self-relative offset from the data field (byte 8) to the elements (byte 16).
const DATA_SELF_REL_OFFSET: i32 = (HEADER_LEN - DATA_FIELD_POS) as i32;

/// Decode and validate the header, returning the element count and the data
/// [`OffsetSlice`]. Confirms magic, element size, and that the whole element
/// region lies within `blob`.
fn decode_header<T: Reloc>(blob: &[u8]) -> Result<(usize, OffsetSlice<T>), RelocError> {
    if blob.len() < HEADER_LEN {
        return Err(RelocError::OutOfBounds);
    }
    if read_u32(blob, 0)? != MAGIC {
        return Err(RelocError::BadMagic);
    }
    if read_u32(blob, 4)? as usize != T::SIZE {
        return Err(RelocError::SizeMismatch);
    }
    let slice = OffsetSlice::<T>::decode(&blob[DATA_FIELD_POS..HEADER_LEN]);
    slice.validate(blob, DATA_FIELD_POS)?;
    Ok((slice.len(), slice))
}

/// An owned, relocatable vector of [`Reloc`] values.
///
/// Build it with [`new`](RelocVec::new) / [`from_slice`](RelocVec::from_slice)
/// and [`push`](RelocVec::push), then hand [`as_bytes`](RelocVec::as_bytes) to a
/// serializer or `mmap` writer. Re-open a copied blob with
/// [`from_bytes`](RelocVec::from_bytes) (owning) or [`RelocVecView::new`]
/// (borrowing, zero-copy).
#[derive(Clone)]
pub struct RelocVec<T> {
    blob: Vec<u8>,
    marker: PhantomData<fn() -> T>,
}

impl<T: Reloc> RelocVec<T> {
    /// Create an empty vector (a bare, valid 16-byte header).
    #[must_use]
    pub fn new() -> Self {
        let mut blob = Vec::with_capacity(HEADER_LEN);
        blob.extend_from_slice(&MAGIC.to_le_bytes());
        blob.extend_from_slice(&(T::SIZE as u32).to_le_bytes());
        let data = OffsetSlice::<T>::from_raw(DATA_SELF_REL_OFFSET, 0);
        let mut field = [0u8; 8];
        data.encode(&mut field);
        blob.extend_from_slice(&field);
        Self {
            blob,
            marker: PhantomData,
        }
    }

    /// Build a vector from a slice of elements.
    #[must_use]
    pub fn from_slice(items: &[T]) -> Self {
        let mut v = Self::new();
        v.blob.reserve(items.len() * T::SIZE);
        for item in items {
            v.push(*item);
        }
        v
    }

    /// Append one element, growing the blob by [`T::SIZE`](Reloc::SIZE) bytes.
    pub fn push(&mut self, value: T) {
        let start = self.blob.len();
        self.blob.resize(start + T::SIZE, 0);
        value.encode(&mut self.blob[start..start + T::SIZE]);
        // Bump the length field (the second u32 of the data OffsetSlice at
        // byte 12), which never overflows a real asset.
        let new_len = self.len() + 1;
        let len_bytes = (new_len as u32).to_le_bytes();
        self.blob[DATA_FIELD_POS + 4..HEADER_LEN].copy_from_slice(&len_bytes);
    }

    /// The number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        // The length is the second u32 of the data OffsetSlice (byte 12). The
        // header is always well-formed for an owned value, so read directly.
        let b = &self.blob[DATA_FIELD_POS + 4..HEADER_LEN];
        u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize
    }

    /// Whether the vector is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Decode and return element `index`, or `None` if out of range.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<T> {
        self.view().ok()?.get(index)
    }

    /// Iterate over the decoded elements in order.
    #[must_use]
    pub fn iter(&self) -> RelocVecIter<'_, T> {
        RelocVecIter {
            blob: &self.blob,
            slice: OffsetSlice::<T>::from_raw(DATA_SELF_REL_OFFSET, self.len() as u32),
            index: 0,
        }
    }

    /// The relocatable blob backing this vector.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.blob
    }

    /// Consume the vector and return the owned blob.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.blob
    }

    /// Re-open an owned vector from a (possibly copied) blob, validating it.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, RelocError> {
        decode_header::<T>(bytes)?;
        Ok(Self {
            blob: bytes.to_vec(),
            marker: PhantomData,
        })
    }

    /// Borrow a zero-copy [`view`](RelocVecView) over this vector's blob.
    pub fn view(&self) -> Result<RelocVecView<'_, T>, RelocError> {
        RelocVecView::new(&self.blob)
    }
}

impl<T: Reloc> Default for RelocVec<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Reloc + core::fmt::Debug> core::fmt::Debug for RelocVec<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

/// A borrowing, zero-copy view over a (possibly `memcpy`'d / `mmap`'d) blob.
///
/// Construction validates the header once; afterwards every [`get`](RelocVecView::get)
/// is an in-bounds decode. This is the "map the baked blob and use it in place"
/// read path.
#[derive(Clone, Copy)]
pub struct RelocVecView<'a, T> {
    blob: &'a [u8],
    len: usize,
    slice: OffsetSlice<T>,
}

impl<'a, T: Reloc> RelocVecView<'a, T> {
    /// Validate `blob` as a [`RelocVec`] blob and borrow a view over it.
    pub fn new(blob: &'a [u8]) -> Result<Self, RelocError> {
        let (len, slice) = decode_header::<T>(blob)?;
        Ok(Self { blob, len, slice })
    }

    /// The number of elements.
    #[inline]
    #[must_use]
    pub fn len(self) -> usize {
        self.len
    }

    /// Whether the view is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.len == 0
    }

    /// Decode and return element `index`, or `None` if out of range.
    #[must_use]
    pub fn get(self, index: usize) -> Option<T> {
        // The header was validated at construction, so the whole region is in
        // bounds and only the index needs checking.
        self.slice
            .get(self.blob, DATA_FIELD_POS, index)
            .ok()
            .flatten()
    }

    /// Iterate over the decoded elements in order.
    #[must_use]
    pub fn iter(self) -> RelocVecIter<'a, T> {
        RelocVecIter {
            blob: self.blob,
            slice: self.slice,
            index: 0,
        }
    }
}

/// Iterator over the decoded elements of a [`RelocVec`] / [`RelocVecView`].
pub struct RelocVecIter<'a, T> {
    blob: &'a [u8],
    slice: OffsetSlice<T>,
    index: usize,
}

impl<T: Reloc> Iterator for RelocVecIter<'_, T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        let item = self.slice.get(self.blob, DATA_FIELD_POS, self.index).ok()?;
        if item.is_some() {
            self.index += 1;
        }
        item
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.slice.len().saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

impl<T: Reloc> ExactSizeIterator for RelocVecIter<'_, T> {}
