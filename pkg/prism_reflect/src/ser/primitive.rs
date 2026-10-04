//! Leaf primitive classification, wire tags, and low-level byte helpers.
//!
//! The reflected leaf subset is the fixed set of built-in `Value`-kind types.
//! Each is assigned a stable 1-byte *primitive tag* and each composite kind a
//! stable 1-byte *node tag*; both are frozen on-disk contracts (design §22).
//! The binary format reads/writes unsigned lengths and indices as LEB128
//! varints, implemented here alongside a bounds-checked [`ByteReader`].

use crate::ser::error::DeserializeError;
use alloc::vec::Vec;

/// The classification of a reflected leaf (`Value`-kind) type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Primitive {
    /// A `bool`.
    Bool,
    /// A `char`.
    Char,
    /// An `i8`.
    I8,
    /// An `i16`.
    I16,
    /// An `i32`.
    I32,
    /// An `i64`.
    I64,
    /// An `i128`.
    I128,
    /// An `isize` (encoded as `i64`).
    Isize,
    /// A `u8`.
    U8,
    /// A `u16`.
    U16,
    /// A `u32`.
    U32,
    /// A `u64`.
    U64,
    /// A `u128`.
    U128,
    /// A `usize` (encoded as `u64`).
    Usize,
    /// An `f32`.
    F32,
    /// An `f64`.
    F64,
    /// A `String`.
    String,
}

/// Frozen primitive tag bytes (do not reorder: on-disk contract).
pub mod prim_tag {
    /// Tag for [`Primitive::Bool`](super::Primitive::Bool).
    pub const BOOL: u8 = 0;
    /// Tag for [`Primitive::Char`](super::Primitive::Char).
    pub const CHAR: u8 = 1;
    /// Tag for [`Primitive::I8`](super::Primitive::I8).
    pub const I8: u8 = 2;
    /// Tag for [`Primitive::I16`](super::Primitive::I16).
    pub const I16: u8 = 3;
    /// Tag for [`Primitive::I32`](super::Primitive::I32).
    pub const I32: u8 = 4;
    /// Tag for [`Primitive::I64`](super::Primitive::I64).
    pub const I64: u8 = 5;
    /// Tag for [`Primitive::I128`](super::Primitive::I128).
    pub const I128: u8 = 6;
    /// Tag for [`Primitive::Isize`](super::Primitive::Isize).
    pub const ISIZE: u8 = 7;
    /// Tag for [`Primitive::U8`](super::Primitive::U8).
    pub const U8: u8 = 8;
    /// Tag for [`Primitive::U16`](super::Primitive::U16).
    pub const U16: u8 = 9;
    /// Tag for [`Primitive::U32`](super::Primitive::U32).
    pub const U32: u8 = 10;
    /// Tag for [`Primitive::U64`](super::Primitive::U64).
    pub const U64: u8 = 11;
    /// Tag for [`Primitive::U128`](super::Primitive::U128).
    pub const U128: u8 = 12;
    /// Tag for [`Primitive::Usize`](super::Primitive::Usize).
    pub const USIZE: u8 = 13;
    /// Tag for [`Primitive::F32`](super::Primitive::F32).
    pub const F32: u8 = 14;
    /// Tag for [`Primitive::F64`](super::Primitive::F64).
    pub const F64: u8 = 15;
    /// Tag for [`Primitive::String`](super::Primitive::String).
    pub const STRING: u8 = 16;
}

/// Frozen node (composite kind) tag bytes (do not reorder: on-disk contract).
pub mod node_tag {
    /// Tag for a named-field struct node.
    pub const STRUCT: u8 = 0;
    /// Tag for a tuple-struct node.
    pub const TUPLE_STRUCT: u8 = 1;
    /// Tag for an enum node.
    pub const ENUM: u8 = 2;
    /// Tag for a list node.
    pub const LIST: u8 = 3;
    /// Tag for an array node.
    pub const ARRAY: u8 = 4;
    /// Tag for a map node.
    pub const MAP: u8 = 5;
    /// Tag for a set node.
    pub const SET: u8 = 6;
    /// Tag for a leaf value node.
    pub const VALUE: u8 = 7;
    /// Tag for a bulk POD (fixed-width numeric) sequence node (design §24.3).
    ///
    /// A backward-compatible addition: streams written before this tag existed
    /// never contain it, and the reader accepts both this and the per-element
    /// `LIST`/`ARRAY` encodings for the same logical value.
    pub const POD_BLOB: u8 = 8;
}

impl Primitive {
    /// The frozen wire tag byte for this primitive.
    #[must_use]
    pub const fn tag(self) -> u8 {
        match self {
            Primitive::Bool => prim_tag::BOOL,
            Primitive::Char => prim_tag::CHAR,
            Primitive::I8 => prim_tag::I8,
            Primitive::I16 => prim_tag::I16,
            Primitive::I32 => prim_tag::I32,
            Primitive::I64 => prim_tag::I64,
            Primitive::I128 => prim_tag::I128,
            Primitive::Isize => prim_tag::ISIZE,
            Primitive::U8 => prim_tag::U8,
            Primitive::U16 => prim_tag::U16,
            Primitive::U32 => prim_tag::U32,
            Primitive::U64 => prim_tag::U64,
            Primitive::U128 => prim_tag::U128,
            Primitive::Usize => prim_tag::USIZE,
            Primitive::F32 => prim_tag::F32,
            Primitive::F64 => prim_tag::F64,
            Primitive::String => prim_tag::STRING,
        }
    }

    /// Recover a primitive from its wire tag byte.
    ///
    /// # Errors
    /// Returns [`DeserializeError::UnknownPrimitiveTag`] for an out-of-range tag.
    pub const fn from_tag(tag: u8) -> Result<Self, DeserializeError> {
        Ok(match tag {
            prim_tag::BOOL => Primitive::Bool,
            prim_tag::CHAR => Primitive::Char,
            prim_tag::I8 => Primitive::I8,
            prim_tag::I16 => Primitive::I16,
            prim_tag::I32 => Primitive::I32,
            prim_tag::I64 => Primitive::I64,
            prim_tag::I128 => Primitive::I128,
            prim_tag::ISIZE => Primitive::Isize,
            prim_tag::U8 => Primitive::U8,
            prim_tag::U16 => Primitive::U16,
            prim_tag::U32 => Primitive::U32,
            prim_tag::U64 => Primitive::U64,
            prim_tag::U128 => Primitive::U128,
            prim_tag::USIZE => Primitive::Usize,
            prim_tag::F32 => Primitive::F32,
            prim_tag::F64 => Primitive::F64,
            prim_tag::STRING => Primitive::String,
            other => return Err(DeserializeError::UnknownPrimitiveTag(other)),
        })
    }

    /// The fully-qualified type name this primitive reflects as.
    #[must_use]
    pub fn type_name(self) -> &'static str {
        match self {
            Primitive::Bool => ::core::any::type_name::<bool>(),
            Primitive::Char => ::core::any::type_name::<char>(),
            Primitive::I8 => ::core::any::type_name::<i8>(),
            Primitive::I16 => ::core::any::type_name::<i16>(),
            Primitive::I32 => ::core::any::type_name::<i32>(),
            Primitive::I64 => ::core::any::type_name::<i64>(),
            Primitive::I128 => ::core::any::type_name::<i128>(),
            Primitive::Isize => ::core::any::type_name::<isize>(),
            Primitive::U8 => ::core::any::type_name::<u8>(),
            Primitive::U16 => ::core::any::type_name::<u16>(),
            Primitive::U32 => ::core::any::type_name::<u32>(),
            Primitive::U64 => ::core::any::type_name::<u64>(),
            Primitive::U128 => ::core::any::type_name::<u128>(),
            Primitive::Usize => ::core::any::type_name::<usize>(),
            Primitive::F32 => ::core::any::type_name::<f32>(),
            Primitive::F64 => ::core::any::type_name::<f64>(),
            Primitive::String => ::core::any::type_name::<::alloc::string::String>(),
        }
    }
}

/// Classify a `Value`-kind type name as a built-in leaf primitive, if it is one.
#[must_use]
pub fn leaf_primitive(type_name: &str) -> Option<Primitive> {
    const CANDIDATES: &[Primitive] = &[
        Primitive::Bool,
        Primitive::Char,
        Primitive::I8,
        Primitive::I16,
        Primitive::I32,
        Primitive::I64,
        Primitive::I128,
        Primitive::Isize,
        Primitive::U8,
        Primitive::U16,
        Primitive::U32,
        Primitive::U64,
        Primitive::U128,
        Primitive::Usize,
        Primitive::F32,
        Primitive::F64,
        Primitive::String,
    ];
    CANDIDATES
        .iter()
        .copied()
        .find(|p| p.type_name() == type_name)
}

/// Append an unsigned LEB128 varint to `out`.
pub fn write_varint(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if value == 0 {
            break;
        }
    }
}

/// A forward, bounds-checked cursor over an input byte slice.
pub struct ByteReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    /// Wrap a slice at offset zero.
    #[must_use]
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    /// Whether every byte has been consumed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pos >= self.bytes.len()
    }

    /// How many bytes remain unread.
    ///
    /// Used to clamp speculative pre-allocation during deserialization: every
    /// element costs at least one byte, so a collection can hold no more
    /// elements than there are bytes left in the stream.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }

    /// Read a single byte.
    ///
    /// # Errors
    /// Returns [`DeserializeError::UnexpectedEof`] at end of input.
    pub fn read_u8(&mut self) -> Result<u8, DeserializeError> {
        let byte = *self
            .bytes
            .get(self.pos)
            .ok_or(DeserializeError::UnexpectedEof)?;
        self.pos += 1;
        Ok(byte)
    }

    /// Read exactly `len` bytes as a borrowed slice.
    ///
    /// # Errors
    /// Returns [`DeserializeError::UnexpectedEof`] when fewer than `len` bytes
    /// remain.
    pub fn read_bytes(&mut self, len: usize) -> Result<&'a [u8], DeserializeError> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or(DeserializeError::UnexpectedEof)?;
        let slice = self
            .bytes
            .get(self.pos..end)
            .ok_or(DeserializeError::UnexpectedEof)?;
        self.pos = end;
        Ok(slice)
    }

    /// Read a fixed-size little-endian byte array.
    ///
    /// # Errors
    /// Returns [`DeserializeError::UnexpectedEof`] at end of input.
    pub fn read_array<const N: usize>(&mut self) -> Result<[u8; N], DeserializeError> {
        let slice = self.read_bytes(N)?;
        let mut out = [0u8; N];
        out.copy_from_slice(slice);
        Ok(out)
    }

    /// Read an unsigned LEB128 varint.
    ///
    /// # Errors
    /// Returns [`DeserializeError::UnexpectedEof`] on truncation, or
    /// [`DeserializeError::TrailingData`] if the encoding overflows 64 bits.
    pub fn read_varint(&mut self) -> Result<u64, DeserializeError> {
        let mut result: u64 = 0;
        let mut shift: u32 = 0;
        loop {
            let byte = self.read_u8()?;
            if shift >= 64 {
                return Err(DeserializeError::TrailingData);
            }
            result |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
        }
        Ok(result)
    }

    /// Read a varint and narrow it to a `usize` length.
    ///
    /// # Errors
    /// Propagates [`read_varint`](Self::read_varint) errors; a value that does
    /// not fit `usize` yields [`DeserializeError::TrailingData`].
    pub fn read_len(&mut self) -> Result<usize, DeserializeError> {
        usize::try_from(self.read_varint()?).map_err(|_| DeserializeError::TrailingData)
    }
}
