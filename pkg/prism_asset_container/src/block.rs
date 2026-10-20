//! Version-2 block-addressed raw-DEFLATE payloads.
//!
//! The block codec is intentionally independent from the outer container
//! layout.  The outer reader supplies a byte range and a shared
//! [`ContainerReadBudget`]; this module authenticates its own index and each
//! decoded block, then reads only blocks intersecting a requested raw range.

use alloc::vec::Vec;
use core::cmp::{max, min};
use std::io::Write;

use flate2::{write::DeflateEncoder, Compression, Decompress, FlushDecompress, Status};

use crate::budget::{ContainerReadBudget, ReadBuffer};
use crate::format::ContainerError;
use crate::range_source::RangeSource;

/// The outer-container codec value reserved for this wire format.
pub const BLOCK_DEFLATE_CODEC: u8 = 2;
/// The block wire-format version.  It is deliberately independent of the
/// outer container version.
pub const BLOCK_FORMAT_VERSION: u16 = 2;
/// Default raw size of each independently encoded block.
pub const DEFAULT_BLOCK_SIZE: u32 = 64 * 1024;
/// Smallest accepted block size for an encoder or wire payload.
pub const MIN_BLOCK_SIZE: u32 = 1;
/// Largest accepted block size for an encoder or wire payload.
pub const MAX_BLOCK_SIZE: u32 = 16 * 1024 * 1024;

const MAGIC: [u8; 8] = *b"PRBLKDF2";
const HEADER_SIZE: usize = 144;
const INDEX_RECORD_SIZE: usize = 80;
const HEADER_RESERVED_OFFSET: usize = 136;
const INDEX_HASH_OFFSET: usize = 104;
const RAW_HASH_OFFSET: usize = 72;

/// Encodes a payload with the canonical 64 KiB block size.
pub fn encode(raw: &[u8]) -> Result<Vec<u8>, ContainerError> {
    encode_with_block_size(raw, DEFAULT_BLOCK_SIZE)
}

/// Encodes a payload as independently stored raw or raw-DEFLATE blocks.
///
/// A block uses raw bytes when DEFLATE would not reduce its size.  This keeps
/// the wire deterministic and ensures every block remains independently
/// addressable.  The block size is part of the authenticated header and may
/// be selected independently for each encoded payload.
pub fn encode_with_block_size(raw: &[u8], block_size: u32) -> Result<Vec<u8>, ContainerError> {
    validate_block_size(block_size)?;
    let block_size = usize::try_from(block_size).map_err(|_| ContainerError::IntegerOverflow)?;
    let block_count = if raw.is_empty() {
        0
    } else {
        raw.len()
            .checked_add(block_size - 1)
            .ok_or(ContainerError::IntegerOverflow)?
            / block_size
    };
    let block_count_u32 =
        u32::try_from(block_count).map_err(|_| ContainerError::IntegerOverflow)?;
    let index_len = block_count
        .checked_mul(INDEX_RECORD_SIZE)
        .ok_or(ContainerError::IntegerOverflow)?;

    let mut blocks = Vec::new();
    blocks
        .try_reserve_exact(block_count)
        .map_err(|_| ContainerError::AllocationFailed {
            bytes: u64::try_from(block_count).unwrap_or(u64::MAX),
        })?;
    let mut data = Vec::new();
    let mut index = Vec::new();
    index
        .try_reserve_exact(index_len)
        .map_err(|_| ContainerError::AllocationFailed {
            bytes: u64::try_from(index_len).unwrap_or(u64::MAX),
        })?;

    for (block_index, block) in raw.chunks(block_size).enumerate() {
        let raw_hash = *blake3::hash(block).as_bytes();
        let compressed = encode_deflate(block)?;
        let (codec, stored) = if compressed.len() < block.len() {
            (1u8, compressed)
        } else {
            (0u8, block.to_vec())
        };
        let raw_offset = block_index
            .checked_mul(block_size)
            .ok_or(ContainerError::IntegerOverflow)?;
        let stored_offset = data.len();
        data.extend_from_slice(&stored);
        blocks.push(EncodedBlock {
            raw_offset: u64::try_from(raw_offset).map_err(|_| ContainerError::IntegerOverflow)?,
            raw_len: u64::try_from(block.len()).map_err(|_| ContainerError::IntegerOverflow)?,
            stored_offset: u64::try_from(stored_offset)
                .map_err(|_| ContainerError::IntegerOverflow)?,
            stored_len: u64::try_from(stored.len()).map_err(|_| ContainerError::IntegerOverflow)?,
            codec,
            raw_hash,
        });
    }

    for block in &blocks {
        append_u64(&mut index, block.raw_offset);
        append_u64(&mut index, block.raw_len);
        append_u64(&mut index, block.stored_offset);
        append_u64(&mut index, block.stored_len);
        index.push(block.codec);
        index.extend_from_slice(&[0, 0, 0]);
        index.extend_from_slice(&block.raw_hash);
        index.extend_from_slice(&[0; 12]);
    }

    let index_offset = u64::try_from(HEADER_SIZE).map_err(|_| ContainerError::IntegerOverflow)?;
    let index_len_u64 = u64::try_from(index.len()).map_err(|_| ContainerError::IntegerOverflow)?;
    let data_offset = index_offset
        .checked_add(index_len_u64)
        .ok_or(ContainerError::IntegerOverflow)?;
    let data_len = u64::try_from(data.len()).map_err(|_| ContainerError::IntegerOverflow)?;
    let total_len = data_offset
        .checked_add(data_len)
        .ok_or(ContainerError::IntegerOverflow)?;
    let raw_hash = *blake3::hash(raw).as_bytes();
    let index_hash = *blake3::hash(&index).as_bytes();

    let mut header = Vec::new();
    header
        .try_reserve_exact(HEADER_SIZE)
        .map_err(|_| ContainerError::AllocationFailed {
            bytes: HEADER_SIZE as u64,
        })?;
    header.extend_from_slice(&MAGIC);
    append_u16(&mut header, BLOCK_FORMAT_VERSION);
    append_u16(&mut header, HEADER_SIZE as u16);
    append_u32(&mut header, 0);
    append_u32(&mut header, block_size as u32);
    append_u32(&mut header, block_count_u32);
    append_u64(&mut header, raw.len() as u64);
    append_u64(&mut header, total_len);
    append_u64(&mut header, index_offset);
    append_u64(&mut header, index_len_u64);
    append_u64(&mut header, data_offset);
    append_u64(&mut header, data_len);
    header.extend_from_slice(&raw_hash);
    header.extend_from_slice(&index_hash);
    header.extend_from_slice(&[0; HEADER_SIZE - HEADER_RESERVED_OFFSET]);

    let mut wire = Vec::new();
    wire.try_reserve_exact(
        usize::try_from(total_len).map_err(|_| ContainerError::IntegerOverflow)?,
    )
    .map_err(|_| ContainerError::AllocationFailed { bytes: total_len })?;
    wire.extend_from_slice(&header);
    wire.extend_from_slice(&index);
    wire.extend_from_slice(&data);
    Ok(wire)
}

/// A parsed, authenticated block index.
///
/// The index bytes remain in a budgeted [`ReadBuffer`] instead of becoming an
/// unbounded collection of host objects.  Callers normally retain this value
/// for the lifetime of a mounted chunk and use [`Self::decode_range`] for
/// demand reads.
pub struct BlockIndex {
    source_offset: u64,
    wire_len: u64,
    block_size: u32,
    block_count: u32,
    raw_len: u64,
    data_offset: u64,
    raw_hash: [u8; 32],
    index: ReadBuffer,
}

impl core::fmt::Debug for BlockIndex {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BlockIndex")
            .field("source_offset", &self.source_offset)
            .field("wire_len", &self.wire_len)
            .field("block_size", &self.block_size)
            .field("block_count", &self.block_count)
            .field("raw_len", &self.raw_len)
            .field("data_offset", &self.data_offset)
            .finish_non_exhaustive()
    }
}

impl BlockIndex {
    /// Returns the canonical raw block size declared by the payload.
    #[must_use]
    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Returns the number of independently stored blocks.
    #[must_use]
    pub fn block_count(&self) -> u32 {
        self.block_count
    }

    /// Returns the complete decoded payload length.
    #[must_use]
    pub fn raw_len(&self) -> u64 {
        self.raw_len
    }

    /// Returns the exact encoded byte length, including header and index.
    #[must_use]
    pub fn wire_len(&self) -> u64 {
        self.wire_len
    }

    /// Decodes `out.len()` bytes beginning at `raw_offset`.
    ///
    /// Only blocks intersecting the requested range are fetched and decoded.
    /// The caller owns `out`; temporary stored and decoded block buffers are
    /// admitted through `budget`.  Per-block hashes are always checked.
    pub fn decode_range(
        &self,
        source: &dyn RangeSource,
        raw_offset: u64,
        out: &mut [u8],
        budget: &ContainerReadBudget,
    ) -> Result<(), ContainerError> {
        let output_len = u64::try_from(out.len()).map_err(|_| ContainerError::IntegerOverflow)?;
        let end = raw_offset
            .checked_add(output_len)
            .ok_or(ContainerError::IntegerOverflow)?;
        if raw_offset > self.raw_len || end > self.raw_len {
            return Err(ContainerError::InvalidRange);
        }
        if out.is_empty() {
            return Ok(());
        }

        let first = raw_offset / u64::from(self.block_size);
        let last = (end - 1) / u64::from(self.block_size);
        let first = u32::try_from(first).map_err(|_| ContainerError::IntegerOverflow)?;
        let last = u32::try_from(last).map_err(|_| ContainerError::IntegerOverflow)?;
        let mut copied = 0usize;

        for block_number in first..=last {
            let record = self.record(block_number)?;
            let block_end = record
                .raw_offset
                .checked_add(record.raw_len)
                .ok_or(ContainerError::IntegerOverflow)?;
            let copy_start = max(raw_offset, record.raw_offset);
            let copy_end = min(end, block_end);
            if copy_start >= copy_end {
                continue;
            }
            let decoded = self.decode_block(source, &record, budget)?;
            let source_start = usize::try_from(copy_start - record.raw_offset)
                .map_err(|_| ContainerError::IntegerOverflow)?;
            let amount = usize::try_from(copy_end - copy_start)
                .map_err(|_| ContainerError::IntegerOverflow)?;
            let destination = usize::try_from(copy_start - raw_offset)
                .map_err(|_| ContainerError::IntegerOverflow)?;
            let source_end = source_start
                .checked_add(amount)
                .ok_or(ContainerError::IntegerOverflow)?;
            let destination_end = destination
                .checked_add(amount)
                .ok_or(ContainerError::IntegerOverflow)?;
            out.get_mut(destination..destination_end)
                .ok_or(ContainerError::OutOfBounds)?
                .copy_from_slice(
                    decoded
                        .get(source_start..source_end)
                        .ok_or(ContainerError::OutOfBounds)?,
                );
            copied = copied
                .checked_add(amount)
                .ok_or(ContainerError::IntegerOverflow)?;
        }
        if copied != out.len() {
            return Err(ContainerError::InvalidDirectory);
        }
        Ok(())
    }

    /// Validates every stored block and the complete decoded-payload hash.
    ///
    /// This is intended for mount-time validation.  It processes one block at
    /// a time, so its temporary memory is bounded by the largest block rather
    /// than by the complete decoded payload.
    pub fn validate_all(
        &self,
        source: &dyn RangeSource,
        budget: &ContainerReadBudget,
    ) -> Result<(), ContainerError> {
        let mut hasher = blake3::Hasher::new();
        for block_number in 0..self.block_count {
            let record = self.record(block_number)?;
            let decoded = self.decode_block(source, &record, budget)?;
            hasher.update(&decoded);
        }
        if *hasher.finalize().as_bytes() != self.raw_hash {
            return Err(ContainerError::HashMismatch);
        }
        Ok(())
    }

    fn record(&self, block_number: u32) -> Result<BlockRecord, ContainerError> {
        if block_number >= self.block_count {
            return Err(ContainerError::OutOfBounds);
        }
        let start = usize::try_from(block_number)
            .ok()
            .and_then(|number| number.checked_mul(INDEX_RECORD_SIZE))
            .ok_or(ContainerError::IntegerOverflow)?;
        let end = start
            .checked_add(INDEX_RECORD_SIZE)
            .ok_or(ContainerError::IntegerOverflow)?;
        let bytes = self
            .index
            .get(start..end)
            .ok_or(ContainerError::Truncated)?;
        let codec = match bytes[32] {
            0 | 1 => bytes[32],
            other => return Err(ContainerError::UnsupportedCodec(other)),
        };
        Ok(BlockRecord {
            raw_offset: read_u64(bytes, 0)?,
            raw_len: read_u64(bytes, 8)?,
            stored_offset: read_u64(bytes, 16)?,
            stored_len: read_u64(bytes, 24)?,
            codec,
            raw_hash: bytes[36..68]
                .try_into()
                .map_err(|_| ContainerError::Truncated)?,
        })
    }

    fn decode_block(
        &self,
        source: &dyn RangeSource,
        record: &BlockRecord,
        budget: &ContainerReadBudget,
    ) -> Result<ReadBuffer, ContainerError> {
        budget.limits().unit(record.stored_len, record.raw_len)?;
        let absolute = self
            .source_offset
            .checked_add(self.data_offset)
            .and_then(|value| value.checked_add(record.stored_offset))
            .ok_or(ContainerError::IntegerOverflow)?;
        let mut stored = budget.buffer(record.stored_len)?;
        source
            .read_range(absolute, &mut stored[..])
            .map_err(|error| ContainerError::Source(error.to_string()))?;

        let decoded = if record.codec == 0 {
            if record.stored_len != record.raw_len {
                return Err(ContainerError::InvalidDirectory);
            }
            stored
        } else {
            let raw_len =
                usize::try_from(record.raw_len).map_err(|_| ContainerError::IntegerOverflow)?;
            let capacity = raw_len
                .checked_add(1)
                .ok_or(ContainerError::IntegerOverflow)?;
            let mut output = budget
                .buffer(u64::try_from(capacity).map_err(|_| ContainerError::IntegerOverflow)?)?;
            let mut decompressor = Decompress::new(false);
            loop {
                let written = usize::try_from(decompressor.total_out())
                    .map_err(|_| ContainerError::CompressionFailed)?;
                if written >= capacity {
                    return Err(ContainerError::CompressionFailed);
                }
                let consumed = usize::try_from(decompressor.total_in())
                    .map_err(|_| ContainerError::CompressionFailed)?;
                let input = stored
                    .get(consumed..)
                    .ok_or(ContainerError::CompressionFailed)?;
                let before_in = decompressor.total_in();
                let before_out = decompressor.total_out();
                let status = decompressor
                    .decompress(input, &mut output[written..], FlushDecompress::Finish)
                    .map_err(|_| ContainerError::CompressionFailed)?;
                if status == Status::StreamEnd {
                    if decompressor.total_in() != record.stored_len
                        || decompressor.total_out() != record.raw_len
                    {
                        return Err(ContainerError::CompressionFailed);
                    }
                    break;
                }
                if status == Status::BufError
                    || (decompressor.total_in() == before_in
                        && decompressor.total_out() == before_out)
                {
                    return Err(ContainerError::CompressionFailed);
                }
            }
            output.truncate(raw_len);
            output
        };

        if decoded.len()
            != usize::try_from(record.raw_len).map_err(|_| ContainerError::IntegerOverflow)?
            || *blake3::hash(&decoded).as_bytes() != record.raw_hash
        {
            return Err(ContainerError::HashMismatch);
        }
        Ok(decoded)
    }
}

/// Parses and validates a block payload's fixed header and index.
///
/// `source_offset` and `wire_len` identify the exact block payload inside a
/// larger container. `existing_metadata_bytes` lets the caller account for
/// outer directory/table bytes in the same `max_metadata_bytes` ceiling.
pub fn parse_index(
    source: &dyn RangeSource,
    source_offset: u64,
    wire_len: u64,
    existing_metadata_bytes: u64,
    budget: &ContainerReadBudget,
) -> Result<BlockIndex, ContainerError> {
    let source_end = source_offset
        .checked_add(wire_len)
        .ok_or(ContainerError::IntegerOverflow)?;
    if source_end > source.len() || wire_len < HEADER_SIZE as u64 {
        return Err(ContainerError::OutOfBounds);
    }
    let mut header = budget.buffer(HEADER_SIZE as u64)?;
    source
        .read_range(source_offset, &mut header[..])
        .map_err(|error| ContainerError::Source(error.to_string()))?;
    if header.get(..8) != Some(MAGIC.as_slice()) {
        return Err(ContainerError::BadMagic);
    }
    if read_u16(&header, 8)? != BLOCK_FORMAT_VERSION {
        return Err(ContainerError::UnsupportedVersion(read_u16(&header, 8)?));
    }
    if read_u16(&header, 10)? as usize != HEADER_SIZE
        || read_u32(&header, 12)? != 0
        || header[HEADER_RESERVED_OFFSET..]
            .iter()
            .any(|byte| *byte != 0)
    {
        return Err(ContainerError::InvalidHeader);
    }
    let block_size = read_u32(&header, 16)?;
    validate_block_size(block_size)?;
    let block_count = read_u32(&header, 20)?;
    let raw_len = read_u64(&header, 24)?;
    let total_len = read_u64(&header, 32)?;
    let index_offset = read_u64(&header, 40)?;
    let index_len = read_u64(&header, 48)?;
    let data_offset = read_u64(&header, 56)?;
    let data_len = read_u64(&header, 64)?;
    let expected_blocks = if raw_len == 0 {
        0
    } else {
        raw_len
            .checked_add(u64::from(block_size) - 1)
            .ok_or(ContainerError::IntegerOverflow)?
            / u64::from(block_size)
    };
    if u64::from(block_count) != expected_blocks
        || index_offset != HEADER_SIZE as u64
        || index_len
            != u64::from(block_count)
                .checked_mul(INDEX_RECORD_SIZE as u64)
                .ok_or(ContainerError::IntegerOverflow)?
        || data_offset
            != index_offset
                .checked_add(index_len)
                .ok_or(ContainerError::IntegerOverflow)?
        || total_len
            != data_offset
                .checked_add(data_len)
                .ok_or(ContainerError::IntegerOverflow)?
        || total_len != wire_len
    {
        return Err(ContainerError::InvalidHeader);
    }

    let metadata = existing_metadata_bytes
        .checked_add(HEADER_SIZE as u64)
        .and_then(|value| value.checked_add(index_len))
        .ok_or(ContainerError::IntegerOverflow)?;
    budget.limits().check(
        "block count",
        u64::from(block_count),
        u64::from(budget.limits().max_records),
    )?;
    budget.limits().check(
        "metadata bytes",
        metadata,
        budget.limits().max_metadata_bytes,
    )?;
    let index_len_usize =
        usize::try_from(index_len).map_err(|_| ContainerError::IntegerOverflow)?;
    let mut index = budget.buffer(index_len)?;
    if index.len() != index_len_usize {
        return Err(ContainerError::IntegerOverflow);
    }
    if index_len != 0 {
        source
            .read_range(
                source_offset
                    .checked_add(index_offset)
                    .ok_or(ContainerError::IntegerOverflow)?,
                &mut index[..],
            )
            .map_err(|error| ContainerError::Source(error.to_string()))?;
    }
    let expected_index_hash: [u8; 32] = header[INDEX_HASH_OFFSET..INDEX_HASH_OFFSET + 32]
        .try_into()
        .map_err(|_| ContainerError::Truncated)?;
    if *blake3::hash(&index).as_bytes() != expected_index_hash {
        return Err(ContainerError::HashMismatch);
    }

    let data_end = data_offset
        .checked_add(data_len)
        .ok_or(ContainerError::IntegerOverflow)?;
    let mut raw_cursor = 0u64;
    let mut stored_cursor = 0u64;
    for block_number in 0..block_count {
        let start = usize::try_from(block_number)
            .ok()
            .and_then(|number| number.checked_mul(INDEX_RECORD_SIZE))
            .ok_or(ContainerError::IntegerOverflow)?;
        let bytes = index
            .get(start..start + INDEX_RECORD_SIZE)
            .ok_or(ContainerError::Truncated)?;
        let record = BlockRecord {
            raw_offset: read_u64(bytes, 0)?,
            raw_len: read_u64(bytes, 8)?,
            stored_offset: read_u64(bytes, 16)?,
            stored_len: read_u64(bytes, 24)?,
            codec: match bytes[32] {
                0 | 1 => bytes[32],
                other => return Err(ContainerError::UnsupportedCodec(other)),
            },
            raw_hash: bytes[36..68]
                .try_into()
                .map_err(|_| ContainerError::Truncated)?,
        };
        if bytes[33..36].iter().any(|byte| *byte != 0)
            || bytes[68..].iter().any(|byte| *byte != 0)
            || record.raw_offset != raw_cursor
            || record.raw_len == 0
            || record.raw_len > u64::from(block_size)
            || record.stored_offset != stored_cursor
            || record.stored_len == 0
        {
            return Err(ContainerError::InvalidDirectory);
        }
        let expected_raw_len = min(
            u64::from(block_size),
            raw_len
                .checked_sub(raw_cursor)
                .ok_or(ContainerError::InvalidDirectory)?,
        );
        if record.raw_len != expected_raw_len {
            return Err(ContainerError::InvalidDirectory);
        }
        if record.codec == 0 && record.stored_len != record.raw_len {
            return Err(ContainerError::InvalidDirectory);
        }
        budget.limits().unit(record.stored_len, record.raw_len)?;
        stored_cursor = stored_cursor
            .checked_add(record.stored_len)
            .ok_or(ContainerError::IntegerOverflow)?;
        raw_cursor = raw_cursor
            .checked_add(record.raw_len)
            .ok_or(ContainerError::IntegerOverflow)?;
        if stored_cursor > data_len
            || data_offset
                .checked_add(stored_cursor)
                .ok_or(ContainerError::IntegerOverflow)?
                > data_end
        {
            return Err(ContainerError::OutOfBounds);
        }
    }
    if raw_cursor != raw_len || stored_cursor != data_len {
        return Err(ContainerError::InvalidDirectory);
    }

    let raw_hash = header[RAW_HASH_OFFSET..RAW_HASH_OFFSET + 32]
        .try_into()
        .map_err(|_| ContainerError::Truncated)?;
    Ok(BlockIndex {
        source_offset,
        wire_len,
        block_size,
        block_count,
        raw_len,
        data_offset,
        raw_hash,
        index,
    })
}

#[derive(Debug)]
struct EncodedBlock {
    raw_offset: u64,
    raw_len: u64,
    stored_offset: u64,
    stored_len: u64,
    codec: u8,
    raw_hash: [u8; 32],
}

#[derive(Clone, Copy, Debug)]
struct BlockRecord {
    raw_offset: u64,
    raw_len: u64,
    stored_offset: u64,
    stored_len: u64,
    codec: u8,
    raw_hash: [u8; 32],
}

fn encode_deflate(raw: &[u8]) -> Result<Vec<u8>, ContainerError> {
    let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
    encoder
        .write_all(raw)
        .map_err(|_| ContainerError::CompressionFailed)?;
    encoder
        .finish()
        .map_err(|_| ContainerError::CompressionFailed)
}

fn validate_block_size(block_size: u32) -> Result<(), ContainerError> {
    if !(MIN_BLOCK_SIZE..=MAX_BLOCK_SIZE).contains(&block_size) {
        return Err(ContainerError::InvalidHeader);
    }
    Ok(())
}

fn append_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn append_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn append_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, ContainerError> {
    let value = bytes
        .get(offset..offset + 2)
        .ok_or(ContainerError::Truncated)?;
    Ok(u16::from_le_bytes(
        value.try_into().map_err(|_| ContainerError::Truncated)?,
    ))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, ContainerError> {
    let value = bytes
        .get(offset..offset + 4)
        .ok_or(ContainerError::Truncated)?;
    Ok(u32::from_le_bytes(
        value.try_into().map_err(|_| ContainerError::Truncated)?,
    ))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, ContainerError> {
    let value = bytes
        .get(offset..offset + 8)
        .ok_or(ContainerError::Truncated)?;
    Ok(u64::from_le_bytes(
        value.try_into().map_err(|_| ContainerError::Truncated)?,
    ))
}
