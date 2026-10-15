//! Versioned little-endian binary codec.
//!
//! Every encoded value uses the frame:
//! `PWP1 | kind:u8 | payload_len:u32le | payload | crc32:u32le`.
//! The CRC covers only the payload. Integer fields inside payloads are
//! little-endian, and collections are preceded by `u32` item counts.

use alloc::vec::Vec;
use core::convert::TryFrom;

use crate::hash::checksum32;
use crate::{ActorId, LogicalClock};
use crate::{AppendOnlyLog, DeltaError, FieldOverride, OverrideDelta, OverrideValue};
use crate::{PersistentId, Snapshot, Transaction, TransactionId, WorldState};

const MAGIC: [u8; 4] = *b"PWP1";
const KIND_DELTA: u8 = 1;
const KIND_TRANSACTION: u8 = 2;
const KIND_LOG: u8 = 3;
const KIND_SNAPSHOT: u8 = 4;
const MAX_PAYLOAD: usize = 16 * 1024 * 1024;
const MAX_ITEMS: u32 = 1_000_000;

/// Binary codec failures. All malformed input is reported without panicking.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CodecError {
    /// The input is shorter than the requested field.
    UnexpectedEof,
    /// The four-byte frame marker is wrong.
    InvalidMagic,
    /// The frame kind is not supported.
    InvalidKind(u8),
    /// The declared payload is larger than the codec safety limit.
    PayloadTooLarge(u32),
    /// The declared item count is larger than the codec safety limit.
    ItemCountTooLarge(u32),
    /// The CRC-32 does not match the payload.
    ChecksumMismatch {
        /// CRC stored in the frame.
        expected: u32,
        /// CRC computed from the payload.
        actual: u32,
    },
    /// Extra bytes follow a complete frame or payload.
    TrailingBytes,
    /// A value tag is not defined by this codec version.
    InvalidValueTag(u8),
    /// A boolean payload is not zero or one.
    InvalidBoolean(u8),
    /// A snapshot attempted to encode a deletion tombstone.
    InvalidSnapshotValue,
    /// A value failed deterministic validation.
    InvalidValue(crate::ValueError),
    /// A delta failed canonical validation.
    InvalidDelta(DeltaError),
    /// A log transaction failed append validation.
    InvalidLog(crate::LogError),
    /// A length could not be represented by the wire format.
    LengthOverflow,
}

/// Encodes a canonical override delta.
pub fn encode_delta(delta: &OverrideDelta) -> Result<Vec<u8>, CodecError> {
    let mut payload = Vec::new();
    write_delta(&mut payload, delta)?;
    frame(KIND_DELTA, payload)
}

/// Decodes and validates a canonical override delta.
pub fn decode_delta(bytes: &[u8]) -> Result<OverrideDelta, CodecError> {
    let payload = unframe(bytes, KIND_DELTA)?;
    let mut reader = Reader::new(payload);
    let delta = read_delta(&mut reader)?;
    reader.finish()?;
    Ok(delta)
}

/// Encodes one immutable transaction.
pub fn encode_transaction(transaction: &Transaction) -> Result<Vec<u8>, CodecError> {
    let mut payload = Vec::new();
    write_transaction(&mut payload, transaction)?;
    frame(KIND_TRANSACTION, payload)
}

/// Decodes one immutable transaction.
pub fn decode_transaction(bytes: &[u8]) -> Result<Transaction, CodecError> {
    let payload = unframe(bytes, KIND_TRANSACTION)?;
    let mut reader = Reader::new(payload);
    let transaction = read_transaction(&mut reader)?;
    reader.finish()?;
    Ok(transaction)
}

/// Encodes an append-only log in sequence order.
pub fn encode_log(log: &AppendOnlyLog) -> Result<Vec<u8>, CodecError> {
    let mut payload = Vec::new();
    put_count(&mut payload, log.len())?;
    for transaction in log.iter() {
        write_transaction(&mut payload, transaction)?;
    }
    frame(KIND_LOG, payload)
}

/// Decodes a log and rechecks sequence, duplicate, clock, and field rules.
pub fn decode_log(bytes: &[u8]) -> Result<AppendOnlyLog, CodecError> {
    let payload = unframe(bytes, KIND_LOG)?;
    let mut reader = Reader::new(payload);
    let count = reader.count()?;
    let mut log = AppendOnlyLog::new();
    for _ in 0..count {
        let transaction = read_transaction(&mut reader)?;
        log.append(transaction).map_err(CodecError::InvalidLog)?;
    }
    reader.finish()?;
    Ok(log)
}

/// Encodes a materialized replay snapshot.
pub fn encode_snapshot(snapshot: &Snapshot) -> Result<Vec<u8>, CodecError> {
    let mut payload = Vec::new();
    put_u64(&mut payload, snapshot.sequence());
    write_state(&mut payload, snapshot.state())?;
    frame(KIND_SNAPSHOT, payload)
}

/// Decodes a materialized replay snapshot.
pub fn decode_snapshot(bytes: &[u8]) -> Result<Snapshot, CodecError> {
    let payload = unframe(bytes, KIND_SNAPSHOT)?;
    let mut reader = Reader::new(payload);
    let sequence = reader.u64()?;
    let state = read_state(&mut reader)?;
    reader.finish()?;
    Ok(Snapshot::new(sequence, state))
}

fn frame(kind: u8, payload: Vec<u8>) -> Result<Vec<u8>, CodecError> {
    if payload.len() > MAX_PAYLOAD {
        return Err(CodecError::LengthOverflow);
    }
    let length = u32::try_from(payload.len()).map_err(|_| CodecError::LengthOverflow)?;
    let mut bytes = Vec::with_capacity(4 + 1 + 4 + payload.len() + 4);
    bytes.extend_from_slice(&MAGIC);
    bytes.push(kind);
    put_u32(&mut bytes, length);
    bytes.extend_from_slice(&payload);
    put_u32(&mut bytes, checksum32(&payload));
    Ok(bytes)
}

fn unframe(bytes: &[u8], expected_kind: u8) -> Result<&[u8], CodecError> {
    let mut reader = Reader::new(bytes);
    if reader.take(4)? != MAGIC {
        return Err(CodecError::InvalidMagic);
    }
    let kind = reader.u8()?;
    if kind != expected_kind {
        return Err(CodecError::InvalidKind(kind));
    }
    let length = reader.u32()?;
    let length_usize = usize::try_from(length).map_err(|_| CodecError::LengthOverflow)?;
    if length_usize > MAX_PAYLOAD {
        return Err(CodecError::PayloadTooLarge(length));
    }
    let payload = reader.take(length_usize)?;
    let expected = reader.u32()?;
    reader.finish()?;
    let actual = checksum32(payload);
    if actual != expected {
        return Err(CodecError::ChecksumMismatch { expected, actual });
    }
    Ok(payload)
}

fn write_state(output: &mut Vec<u8>, state: &WorldState) -> Result<(), CodecError> {
    put_count(output, state.len())?;
    for (object, object_state) in state.iter() {
        put_u128(output, object.0);
        put_count(output, object_state.len())?;
        for (field, value) in object_state.iter() {
            put_u64(output, field.0);
            write_value(output, value)?;
        }
    }
    Ok(())
}

fn read_state(reader: &mut Reader<'_>) -> Result<WorldState, CodecError> {
    let object_count = reader.count()?;
    let mut state = WorldState::new();
    for _ in 0..object_count {
        let object = PersistentId(reader.u128()?);
        let field_count = reader.count()?;
        let mut fields = Vec::new();
        for _ in 0..field_count {
            let field = crate::FieldId(reader.u64()?);
            let value = read_value(reader)?;
            if matches!(value, OverrideValue::Delete) {
                return Err(CodecError::InvalidSnapshotValue);
            }
            fields.push(FieldOverride { field, value });
        }
        let delta = OverrideDelta::new(fields).map_err(CodecError::InvalidDelta)?;
        state.apply_delta(object, &delta);
    }
    Ok(state)
}

fn write_transaction(output: &mut Vec<u8>, transaction: &Transaction) -> Result<(), CodecError> {
    put_u128(output, transaction.id().0);
    put_u128(output, transaction.actor().0);
    put_u64(output, transaction.clock().0);
    put_u64(output, transaction.sequence());
    put_u128(output, transaction.object().0);
    write_delta(output, transaction.delta())?;
    write_delta(output, transaction.inverse())?;
    Ok(())
}

fn read_transaction(reader: &mut Reader<'_>) -> Result<Transaction, CodecError> {
    let id = TransactionId(reader.u128()?);
    let actor = ActorId(reader.u128()?);
    let clock = LogicalClock(reader.u64()?);
    let sequence = reader.u64()?;
    let object = PersistentId(reader.u128()?);
    let delta = read_delta(reader)?;
    let inverse = read_delta(reader)?;
    Transaction::new(id, actor, clock, sequence, object, delta, inverse)
        .map_err(CodecError::InvalidDelta)
}

fn write_delta(output: &mut Vec<u8>, delta: &OverrideDelta) -> Result<(), CodecError> {
    put_count(output, delta.len())?;
    for change in delta.iter() {
        put_u64(output, change.field.0);
        write_value(output, &change.value)?;
    }
    Ok(())
}

fn read_delta(reader: &mut Reader<'_>) -> Result<OverrideDelta, CodecError> {
    let count = reader.count()?;
    let mut changes = Vec::new();
    for _ in 0..count {
        changes.push(FieldOverride {
            field: crate::FieldId(reader.u64()?),
            value: read_value(reader)?,
        });
    }
    OverrideDelta::new(changes).map_err(CodecError::InvalidDelta)
}

fn write_value(output: &mut Vec<u8>, value: &OverrideValue) -> Result<(), CodecError> {
    value.validate().map_err(CodecError::InvalidValue)?;
    output.push(value.tag());
    match value {
        OverrideValue::Integer(value) => put_u64(output, *value as u64),
        OverrideValue::Unsigned(value) => put_u64(output, *value),
        OverrideValue::Float(value) => put_u64(output, value.to_bits()),
        OverrideValue::Boolean(value) => output.push(u8::from(*value)),
        OverrideValue::Bytes(value) => {
            put_count(output, value.len())?;
            output.extend_from_slice(value);
        }
        OverrideValue::Delete => {}
    }
    Ok(())
}

fn read_value(reader: &mut Reader<'_>) -> Result<OverrideValue, CodecError> {
    let tag = reader.u8()?;
    match tag {
        1 => Ok(OverrideValue::Integer(reader.u64()? as i64)),
        2 => Ok(OverrideValue::Unsigned(reader.u64()?)),
        3 => OverrideValue::float(f64::from_bits(reader.u64()?)).map_err(CodecError::InvalidValue),
        4 => match reader.u8()? {
            0 => Ok(OverrideValue::Boolean(false)),
            1 => Ok(OverrideValue::Boolean(true)),
            value => Err(CodecError::InvalidBoolean(value)),
        },
        5 => {
            let length = reader.count()? as usize;
            let bytes = reader.take(length)?.to_vec();
            Ok(OverrideValue::Bytes(bytes))
        }
        6 => Ok(OverrideValue::Delete),
        _ => Err(CodecError::InvalidValueTag(tag)),
    }
}

fn put_count(output: &mut Vec<u8>, count: usize) -> Result<(), CodecError> {
    let count = u32::try_from(count).map_err(|_| CodecError::LengthOverflow)?;
    if count > MAX_ITEMS {
        return Err(CodecError::ItemCountTooLarge(count));
    }
    put_u32(output, count);
    Ok(())
}

fn put_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(output: &mut Vec<u8>, value: u64) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn put_u128(output: &mut Vec<u8>, value: u128) {
    output.extend_from_slice(&value.to_le_bytes());
}

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], CodecError> {
        let end = self
            .position
            .checked_add(length)
            .ok_or(CodecError::UnexpectedEof)?;
        if end > self.bytes.len() {
            return Err(CodecError::UnexpectedEof);
        }
        let value = &self.bytes[self.position..end];
        self.position = end;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8, CodecError> {
        self.take(1)?
            .first()
            .copied()
            .ok_or(CodecError::UnexpectedEof)
    }

    fn u32(&mut self) -> Result<u32, CodecError> {
        let mut bytes = [0_u8; 4];
        bytes.copy_from_slice(self.take(4)?);
        Ok(u32::from_le_bytes(bytes))
    }

    fn u64(&mut self) -> Result<u64, CodecError> {
        let mut bytes = [0_u8; 8];
        bytes.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(bytes))
    }

    fn u128(&mut self) -> Result<u128, CodecError> {
        let mut bytes = [0_u8; 16];
        bytes.copy_from_slice(self.take(16)?);
        Ok(u128::from_le_bytes(bytes))
    }

    fn count(&mut self) -> Result<u32, CodecError> {
        let count = self.u32()?;
        if count > MAX_ITEMS {
            Err(CodecError::ItemCountTooLarge(count))
        } else {
            Ok(count)
        }
    }

    fn finish(&self) -> Result<(), CodecError> {
        if self.position == self.bytes.len() {
            Ok(())
        } else {
            Err(CodecError::TrailingBytes)
        }
    }
}
