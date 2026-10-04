//! Compact field-delta wire codec (design §24.5).
//!
//! A delta is the minimal set of `(field index, encoded value)` pairs selected
//! by a [`DirtyMask`](crate::net_delta::DirtyMask). The wire form is a dense
//! byte string:
//!
//! ```text
//! delta := varint(field_count) entry*
//! entry := varint(field_index) varint(byte_len) value_bytes
//! ```
//!
//! `field_index` is the field's positional index (not its name), so a struct
//! with up to 128 fields spends a single byte per field id. `value_bytes` is
//! the field value encoded with the existing reflection serializer
//! ([`to_binary`](crate::to_binary)), which already pins the value's type with a
//! [`StableTypeId`](crate::StableTypeId) and hardens decoding against untrusted
//! input (design §24.3/§24.8); the delta reuses it rather than inventing a
//! second value encoding.
//!
//! [`apply_delta`] decodes each field value against the *live target field's*
//! [`TypeInfo`](crate::TypeInfo) and applies it in place, so applying the same
//! delta twice is idempotent and the result equals a full snapshot of the new
//! value (verified by the oracle tests).

use crate::net_delta::DeltaError;
use crate::reflect::{Reflect, ReflectMut};
use crate::ser::{from_binary, to_binary};
use crate::net_delta::DirtyMask;
use crate::TypeRegistry;
use alloc::vec::Vec;

/// A decoded field delta: `(field index, encoded value bytes)` pairs in
/// ascending index order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FieldDelta {
    entries: Vec<(usize, Vec<u8>)>,
}

impl FieldDelta {
    /// The decoded `(field index, value bytes)` entries.
    #[must_use]
    pub fn entries(&self) -> &[(usize, Vec<u8>)] {
        &self.entries
    }

    /// The number of fields carried by this delta.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the delta carries no fields (nothing changed).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Serialize this delta back to its compact wire form.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        write_varint(&mut out, self.entries.len() as u64);
        for (index, bytes) in &self.entries {
            write_varint(&mut out, *index as u64);
            write_varint(&mut out, bytes.len() as u64);
            out.extend_from_slice(bytes);
        }
        out
    }
}

/// Encode the fields marked in `dirty` of the struct `value` into a compact
/// delta byte string.
///
/// Only fields whose bit is set are serialized; the result is empty-but-valid
/// (a single `0` count byte) when nothing is marked. Field order follows
/// ascending field index.
///
/// # Errors
/// Returns [`DeltaError::NotAStruct`] if `value` is not a named-field struct,
/// [`DeltaError::FieldIndexOutOfRange`] if a marked index exceeds the struct's
/// field count, or [`DeltaError::Serialize`] if a field value cannot be encoded.
pub fn encode_delta(value: &dyn Reflect, dirty: &DirtyMask) -> Result<Vec<u8>, DeltaError> {
    let crate::ReflectRef::Struct(structure) = value.reflect_ref() else {
        return Err(DeltaError::NotAStruct);
    };
    let field_count = structure.field_count();

    let mut out = Vec::new();
    write_varint(&mut out, dirty.count() as u64);
    for index in dirty.iter() {
        let field = structure
            .field_at(index)
            .ok_or(DeltaError::FieldIndexOutOfRange { index, field_count })?;
        let bytes = to_binary(field).map_err(DeltaError::Serialize)?;
        write_varint(&mut out, index as u64);
        write_varint(&mut out, bytes.len() as u64);
        out.extend_from_slice(&bytes);
    }
    Ok(out)
}

/// Decode a delta byte string produced by [`encode_delta`] (or
/// [`FieldDelta::to_bytes`]) into a [`FieldDelta`].
///
/// This does not need a [`TypeRegistry`]: it only splits the frame into
/// `(index, value bytes)` entries, deferring value decoding to [`apply_delta`]
/// where the live target field's type guides reconstruction.
///
/// # Errors
/// Returns [`DeltaError::Malformed`] if the frame is truncated, carries a
/// malformed varint, declares a byte length past the end of the buffer, or has
/// trailing bytes after the final entry.
pub fn decode_delta(bytes: &[u8]) -> Result<FieldDelta, DeltaError> {
    let mut cursor = Cursor::new(bytes);
    let count = cursor.read_varint()?;
    let mut entries = Vec::with_capacity(count.min(bytes.len() as u64) as usize);
    for _ in 0..count {
        let index = cursor.read_varint()?;
        let len = cursor.read_varint()?;
        let value = cursor.read_slice(len)?;
        entries.push((
            usize::try_from(index).map_err(|_| DeltaError::Malformed)?,
            value.to_vec(),
        ));
    }
    if !cursor.is_empty() {
        return Err(DeltaError::Malformed);
    }
    Ok(FieldDelta { entries })
}

/// Apply a decoded [`FieldDelta`] onto `target`, decoding each field value
/// against the live target field's [`TypeInfo`](crate::TypeInfo).
///
/// Applying is idempotent: because each entry carries the full new value of its
/// field and is applied in place, replaying the same delta leaves `target`
/// unchanged after the first application.
///
/// # Errors
/// Returns [`DeltaError::NotAStruct`] if `target` is not a named-field struct,
/// [`DeltaError::FieldIndexOutOfRange`] if an entry names a missing field,
/// [`DeltaError::Deserialize`] if a value fails to decode against the target
/// field, or [`DeltaError::Apply`] if the decoded value is incompatible with
/// the field.
pub fn apply_delta(
    target: &mut dyn Reflect,
    delta: &FieldDelta,
    registry: &TypeRegistry,
) -> Result<(), DeltaError> {
    if !matches!(target.reflect_mut(), ReflectMut::Struct(_)) {
        return Err(DeltaError::NotAStruct);
    }
    for (index, bytes) in &delta.entries {
        let ReflectMut::Struct(structure) = target.reflect_mut() else {
            return Err(DeltaError::NotAStruct);
        };
        let field_count = structure.field_count();
        let field = structure
            .field_at_mut(*index)
            .ok_or(DeltaError::FieldIndexOutOfRange {
                index: *index,
                field_count,
            })?;
        let info = field.type_info();
        let decoded = from_binary(bytes, registry, info).map_err(DeltaError::Deserialize)?;
        field.apply(&*decoded).map_err(DeltaError::Apply)?;
    }
    Ok(())
}

/// Decode `bytes` and apply the resulting delta onto `target` in one step.
///
/// # Errors
/// Propagates any [`DeltaError`] from [`decode_delta`] or [`apply_delta`].
pub fn decode_and_apply(
    target: &mut dyn Reflect,
    bytes: &[u8],
    registry: &TypeRegistry,
) -> Result<(), DeltaError> {
    let delta = decode_delta(bytes)?;
    apply_delta(target, &delta, registry)
}

/// Append an unsigned LEB128 varint to `out`.
fn write_varint(out: &mut Vec<u8>, mut value: u64) {
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

/// A forward byte cursor with bounds-checked LEB128 and slice reads.
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn is_empty(&self) -> bool {
        self.pos >= self.bytes.len()
    }

    fn read_varint(&mut self) -> Result<u64, DeltaError> {
        let mut result: u64 = 0;
        let mut shift = 0u32;
        loop {
            let byte = *self.bytes.get(self.pos).ok_or(DeltaError::Malformed)?;
            self.pos += 1;
            // A u64 LEB128 is at most 10 groups of 7 bits; reject overlong input.
            if shift >= 64 {
                return Err(DeltaError::Malformed);
            }
            result |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(result);
            }
            shift += 7;
        }
    }

    fn read_slice(&mut self, len: u64) -> Result<&'a [u8], DeltaError> {
        let len = usize::try_from(len).map_err(|_| DeltaError::Malformed)?;
        let end = self.pos.checked_add(len).ok_or(DeltaError::Malformed)?;
        let slice = self.bytes.get(self.pos..end).ok_or(DeltaError::Malformed)?;
        self.pos = end;
        Ok(slice)
    }
}
