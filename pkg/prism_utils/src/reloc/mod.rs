//! # Relocatable, offset-pointer containers (`reloc`) — design doc §24.5
//!
//! Containers that store their internal references as **relative byte offsets**
//! instead of absolute pointers, so the whole backing blob can be `memcpy`'d,
//! serialised to disk, or `mmap`'d to an arbitrary address and remain valid
//! without any fix-up pass. This is the Unreal `TArray`-serialisation /
//! `FlatBuffers` shape: *bake a container into a contiguous blob, then at runtime
//! use it in place with zero parsing* (design doc §24.5, feeding `prism_asset`
//! baking and `prism_platform` §6 `mmap`).
//!
//! Everything here is **pure safe code** (the module sets
//! `#![forbid(unsafe_code)]`); relocation safety comes from storing offsets and
//! bounds-checking every access against the blob length, never from raw pointer
//! arithmetic. The byte encoding is fixed **little-endian**, so a blob is
//! identical across runs *and* across target endianness — a baked asset built
//! on one machine deserialises bit-for-bit on another.
//!
//! ## What ships here
//! - [`Reloc`]: the plain-old-data trait describing how a value is encoded to
//!   and decoded from a fixed-width little-endian byte run. Implemented for the
//!   integer and float primitives, `bool`, and the offset-pointer primitives.
//! - [`OffsetPtr`] / [`OffsetSlice`]: *self-relative* offset pointers (the
//!   offset is measured from the field's own position in the blob), the
//!   building block that makes an embedded sub-structure relocate with its
//!   parent.
//! - [`RelocVec`] (module [`vec`]): a relocatable, length-prefixed vector. It
//!   owns a contiguous blob, can [`push`](RelocVec::push)/build incrementally,
//!   and exposes the blob via [`as_bytes`](RelocVec::as_bytes). A copied blob is
//!   re-opened zero-copy through [`RelocVecView`].
//! - [`RelocMap`] (module [`map`]): an immutable, sorted, relocatable map built
//!   from key/value pairs with `O(log n)` binary-search lookup. Its root header
//!   addresses the key and value regions through two [`OffsetSlice`]s, so it is
//!   the worked example of offset pointers inside a container. A copied blob is
//!   re-opened zero-copy through [`RelocMapView`].
//!
//! ## Honest boundary (design doc §24.8)
//! - These containers encode scalar [`Reloc`] element types. A nested
//!   variable-length graph (a reloc container *inside* a reloc container) is not
//!   provided here; the [`OffsetSlice`] primitive is the hook a future baker
//!   would compose for that, and remains PLANNED.
//! - `mmap` itself (mapping a file into memory) is a `prism_platform` §6 syscall
//!   concern; this module provides the position-independent byte format and the
//!   zero-copy [`view`](RelocVecView) over whatever `&[u8]` the platform layer
//!   hands back.
//! - POD-layout *versioning* (the bytemuck/§14 bullet) is represented here only
//!   by the per-container magic tag and element-size check that reject a
//!   foreign or corrupt blob; a full schema-evolution layer is out of scope.

#![forbid(unsafe_code)]


use core::fmt;

pub mod map;
pub mod offset;
pub mod vec;

pub use map::{RelocMap, RelocMapView};
pub use offset::{OffsetPtr, OffsetSlice};
pub use vec::{RelocVec, RelocVecView};

/// An error raised while decoding or validating a relocatable blob.
///
/// Every accessor on a [`view`](RelocVecView) and every `from_bytes`
/// constructor is total: a malformed, truncated, or foreign blob yields one of
/// these variants rather than reading out of bounds (the module is
/// `#![forbid(unsafe_code)]`, so an out-of-range read is impossible anyway, but
/// these turn it into a precise, reported error).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RelocError {
    /// The blob is shorter than the fixed header, or a resolved region runs
    /// past the end of the blob.
    OutOfBounds,
    /// The leading magic tag did not match the expected container kind (the
    /// blob is not a blob of this type, or is corrupt).
    BadMagic,
    /// The element size recorded in the header disagrees with the element type
    /// the blob is being opened as (a layout/version mismatch).
    SizeMismatch,
    /// A resolved offset was negative or a region length overflowed the
    /// pointer-sized address space.
    Overflow,
}

impl RelocError {
    /// A stable, human-readable description of the error.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OutOfBounds => "relocatable blob: region runs past the end of the buffer",
            Self::BadMagic => "relocatable blob: magic tag does not match this container kind",
            Self::SizeMismatch => "relocatable blob: element size does not match the element type",
            Self::Overflow => "relocatable blob: offset/length overflowed the address space",
        }
    }
}

impl fmt::Display for RelocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl core::error::Error for RelocError {}

/// A fixed-width plain-old-data value that can live inside a relocatable blob.
///
/// Encoding is **little-endian** and exactly [`SIZE`](Reloc::SIZE) bytes wide,
/// which is what makes a baked blob portable across target endianness. The
/// trait is implemented for the integer primitives (`u8`..`u64`, `i8`..`i64`),
/// the IEEE-754 floats (`f32`, `f64`), `bool`, and the offset-pointer
/// primitives [`OffsetPtr`] / [`OffsetSlice`].
///
/// # Contract
/// - [`encode`](Reloc::encode) writes exactly [`SIZE`](Reloc::SIZE) bytes into
///   the start of `dst` (callers always pass a slice at least that long).
/// - [`decode`](Reloc::decode) reads exactly [`SIZE`](Reloc::SIZE) bytes from
///   the start of `src` and is the exact inverse of `encode`.
pub trait Reloc: Copy {
    /// The fixed width of the little-endian encoding, in bytes.
    const SIZE: usize;

    /// Encode `self` into the first [`SIZE`](Reloc::SIZE) bytes of `dst`.
    fn encode(&self, dst: &mut [u8]);

    /// Decode a value from the first [`SIZE`](Reloc::SIZE) bytes of `src`.
    fn decode(src: &[u8]) -> Self;
}

macro_rules! impl_reloc_le {
    ($($t:ty),* $(,)?) => {
        $(
            impl Reloc for $t {
                const SIZE: usize = core::mem::size_of::<$t>();

                #[inline]
                fn encode(&self, dst: &mut [u8]) {
                    dst[..core::mem::size_of::<$t>()].copy_from_slice(&self.to_le_bytes());
                }

                #[inline]
                fn decode(src: &[u8]) -> Self {
                    let mut buf = [0u8; core::mem::size_of::<$t>()];
                    buf.copy_from_slice(&src[..core::mem::size_of::<$t>()]);
                    <$t>::from_le_bytes(buf)
                }
            }
        )*
    };
}

impl_reloc_le!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

impl Reloc for bool {
    const SIZE: usize = 1;

    #[inline]
    fn encode(&self, dst: &mut [u8]) {
        dst[0] = u8::from(*self);
    }

    #[inline]
    fn decode(src: &[u8]) -> Self {
        src[0] != 0
    }
}

/// Resolve a self-relative offset against the position of the field that holds
/// it, returning the absolute byte position in the blob.
///
/// Fails with [`RelocError::Overflow`] if the result is negative or does not fit
/// in a `usize`.
#[inline]
pub(crate) fn resolve_pos(field_pos: usize, offset: i32) -> Result<usize, RelocError> {
    let target = field_pos as i64 + i64::from(offset);
    if target < 0 {
        return Err(RelocError::Overflow);
    }
    usize::try_from(target).map_err(|_| RelocError::Overflow)
}

/// Read a little-endian `u32` at `pos`, bounds-checked against `blob`.
#[inline]
pub(crate) fn read_u32(blob: &[u8], pos: usize) -> Result<u32, RelocError> {
    let end = pos.checked_add(4).ok_or(RelocError::Overflow)?;
    let bytes = blob.get(pos..end).ok_or(RelocError::OutOfBounds)?;
    let mut buf = [0u8; 4];
    buf.copy_from_slice(bytes);
    Ok(u32::from_le_bytes(buf))
}


#[cfg(test)]
mod tests;
