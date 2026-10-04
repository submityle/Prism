//! Bulk codec for homogeneous fixed-width numeric (POD) sequences (design §24.3).
//!
//! The generic traversal driver encodes a `List`/`Array` element-by-element,
//! prefixing every element with its node tag and primitive tag. For large,
//! homogeneous sequences of fixed-width numeric scalars that overhead dominates
//! both size and time. This module recognizes such sequences and encodes them
//! as a single *bulk blob*: one primitive tag, one varint element count, then
//! the elements' little-endian bytes back-to-back with no per-element framing.
//!
//! This is a **safe** bulk path, not a pointer-level `memcpy`: each element is
//! still visited through reflection and written via `to_le_bytes`, and read
//! back via `from_le_bytes`. The win is the elimination of per-element tag
//! bytes and recursive dispatch, plus a single contiguous `extend_from_slice`.
//! Because every scalar is encoded little-endian, the blob is byte-identical
//! across target endianness, preserving the format's cross-platform contract.
//!
//! POD eligibility is deliberately restricted to the fixed-width *numeric*
//! leaves. `bool` and `char` are excluded because they require validation on
//! read (`char` must be a valid scalar value), and `String` is excluded because
//! it is variable-width. `isize`/`usize` are encoded as 8-byte values, matching
//! the element framing used by the per-element path.

use crate::reflect::Reflect;
use crate::ser::error::DeserializeError;
use crate::ser::primitive::{ByteReader, Primitive};
use alloc::boxed::Box;
use alloc::vec::Vec;

/// The fixed little-endian width, in bytes, of a POD-eligible primitive.
///
/// Returns `None` for the non-POD leaves (`bool`, `char`, `String`), which must
/// use the per-element framing path.
#[must_use]
pub fn pod_width(primitive: Primitive) -> Option<usize> {
    Some(match primitive {
        Primitive::I8 | Primitive::U8 => 1,
        Primitive::I16 | Primitive::U16 => 2,
        Primitive::I32 | Primitive::U32 | Primitive::F32 => 4,
        Primitive::I64 | Primitive::U64 | Primitive::F64 | Primitive::Isize | Primitive::Usize => 8,
        Primitive::I128 | Primitive::U128 => 16,
        Primitive::Bool | Primitive::Char | Primitive::String => return None,
    })
}

/// Classify a leaf's concrete type as a POD-eligible primitive, if it is one.
fn pod_class(leaf: &dyn Reflect) -> Option<Primitive> {
    let any = leaf.as_any();
    if any.is::<i8>() {
        Some(Primitive::I8)
    } else if any.is::<i16>() {
        Some(Primitive::I16)
    } else if any.is::<i32>() {
        Some(Primitive::I32)
    } else if any.is::<i64>() {
        Some(Primitive::I64)
    } else if any.is::<i128>() {
        Some(Primitive::I128)
    } else if any.is::<isize>() {
        Some(Primitive::Isize)
    } else if any.is::<u16>() {
        Some(Primitive::U16)
    } else if any.is::<u32>() {
        Some(Primitive::U32)
    } else if any.is::<u64>() {
        Some(Primitive::U64)
    } else if any.is::<u128>() {
        Some(Primitive::U128)
    } else if any.is::<usize>() {
        Some(Primitive::Usize)
    } else if any.is::<u8>() {
        Some(Primitive::U8)
    } else if any.is::<f32>() {
        Some(Primitive::F32)
    } else if any.is::<f64>() {
        Some(Primitive::F64)
    } else {
        None
    }
}

/// Append `leaf`'s little-endian bytes for the already-classified `primitive`.
///
/// Returns `None` only if `leaf` does not downcast to `primitive`'s concrete
/// type, which cannot happen for a correctly classified homogeneous sequence
/// but is handled defensively rather than panicking.
fn append_le(out: &mut Vec<u8>, primitive: Primitive, leaf: &dyn Reflect) -> Option<()> {
    let any = leaf.as_any();
    match primitive {
        Primitive::I8 => out.extend_from_slice(&any.downcast_ref::<i8>()?.to_le_bytes()),
        Primitive::I16 => out.extend_from_slice(&any.downcast_ref::<i16>()?.to_le_bytes()),
        Primitive::I32 => out.extend_from_slice(&any.downcast_ref::<i32>()?.to_le_bytes()),
        Primitive::I64 => out.extend_from_slice(&any.downcast_ref::<i64>()?.to_le_bytes()),
        Primitive::I128 => out.extend_from_slice(&any.downcast_ref::<i128>()?.to_le_bytes()),
        Primitive::Isize => {
            let value = *any.downcast_ref::<isize>()?;
            out.extend_from_slice(&(value as i64).to_le_bytes());
        }
        Primitive::U8 => out.extend_from_slice(&any.downcast_ref::<u8>()?.to_le_bytes()),
        Primitive::U16 => out.extend_from_slice(&any.downcast_ref::<u16>()?.to_le_bytes()),
        Primitive::U32 => out.extend_from_slice(&any.downcast_ref::<u32>()?.to_le_bytes()),
        Primitive::U64 => out.extend_from_slice(&any.downcast_ref::<u64>()?.to_le_bytes()),
        Primitive::U128 => out.extend_from_slice(&any.downcast_ref::<u128>()?.to_le_bytes()),
        Primitive::Usize => {
            let value = *any.downcast_ref::<usize>()?;
            out.extend_from_slice(&(value as u64).to_le_bytes());
        }
        Primitive::F32 => {
            out.extend_from_slice(&any.downcast_ref::<f32>()?.to_bits().to_le_bytes());
        }
        Primitive::F64 => {
            out.extend_from_slice(&any.downcast_ref::<f64>()?.to_bits().to_le_bytes());
        }
        Primitive::Bool | Primitive::Char | Primitive::String => return None,
    }
    Some(())
}

/// Try to collect a homogeneous POD sequence into `(primitive, le_blob)`.
///
/// Returns `None` when the sequence is empty, its first element is not a
/// POD-eligible numeric leaf, or any element's concrete type differs from the
/// first. Callers fall back to the per-element framing path in that case, so
/// correctness never depends on this fast path firing.
#[must_use]
pub fn collect<'a>(
    mut elements: impl Iterator<Item = &'a dyn Reflect>,
    len: usize,
) -> Option<(Primitive, Vec<u8>)> {
    let first = elements.next()?;
    let primitive = pod_class(first)?;
    let width = pod_width(primitive)?;
    let mut raw = Vec::with_capacity(len.saturating_mul(width));
    append_le(&mut raw, primitive, first)?;
    for element in elements {
        if pod_class(element)? != primitive {
            return None;
        }
        append_le(&mut raw, primitive, element)?;
    }
    Some((primitive, raw))
}

/// Read a single POD element of `primitive` (no per-element tag) from `reader`.
fn read_one(
    reader: &mut ByteReader<'_>,
    primitive: Primitive,
) -> Result<Box<dyn Reflect>, DeserializeError> {
    Ok(match primitive {
        Primitive::I8 => Box::new(i8::from_le_bytes(reader.read_array::<1>()?)),
        Primitive::I16 => Box::new(i16::from_le_bytes(reader.read_array::<2>()?)),
        Primitive::I32 => Box::new(i32::from_le_bytes(reader.read_array::<4>()?)),
        Primitive::I64 => Box::new(i64::from_le_bytes(reader.read_array::<8>()?)),
        Primitive::I128 => Box::new(i128::from_le_bytes(reader.read_array::<16>()?)),
        Primitive::Isize => {
            let value = i64::from_le_bytes(reader.read_array::<8>()?);
            Box::new(isize::try_from(value).map_err(|_| DeserializeError::TrailingData)?)
        }
        Primitive::U8 => Box::new(reader.read_u8()?),
        Primitive::U16 => Box::new(u16::from_le_bytes(reader.read_array::<2>()?)),
        Primitive::U32 => Box::new(u32::from_le_bytes(reader.read_array::<4>()?)),
        Primitive::U64 => Box::new(u64::from_le_bytes(reader.read_array::<8>()?)),
        Primitive::U128 => Box::new(u128::from_le_bytes(reader.read_array::<16>()?)),
        Primitive::Usize => {
            let value = u64::from_le_bytes(reader.read_array::<8>()?);
            Box::new(usize::try_from(value).map_err(|_| DeserializeError::TrailingData)?)
        }
        Primitive::F32 => Box::new(f32::from_bits(u32::from_le_bytes(
            reader.read_array::<4>()?,
        ))),
        Primitive::F64 => Box::new(f64::from_bits(u64::from_le_bytes(
            reader.read_array::<8>()?,
        ))),
        Primitive::Bool | Primitive::Char | Primitive::String => {
            return Err(DeserializeError::LeafTypeMismatch {
                expected: primitive.type_name(),
            });
        }
    })
}

/// Read `count` POD elements of `primitive` from a bulk blob into boxed values.
///
/// # Errors
/// Returns [`DeserializeError::UnexpectedEof`] on truncation, or
/// [`DeserializeError::LeafTypeMismatch`] if `primitive` is not POD-eligible.
pub fn read_blob(
    reader: &mut ByteReader<'_>,
    primitive: Primitive,
    count: usize,
) -> Result<Vec<Box<dyn Reflect>>, DeserializeError> {
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(read_one(reader, primitive)?);
    }
    Ok(out)
}
