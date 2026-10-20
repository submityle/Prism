//! Zero-copy v2 directory lookup over a validated read-only mapping.

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt;
use core::mem::size_of;
use core::ops::Range;

use prism_asset::StableGuid;

use crate::format::{
    read_u16, read_u32, read_u64, ChunkInfo, Codec, ContainerError, ContainerInfo, EntryInfo,
    CHUNK_FORMAT_VERSION, CHUNK_HEADER_SIZE, CHUNK_RECORD_SIZE, CHUNK_TABLE_RECORD_SIZE, MAGIC,
};
use crate::{ContainerReadBudget, MmapRangeSource, RangeSource};

use super::decode_chunk_source;

/// Read-only zero-copy index for a fully validated v2 container mapping.
///
/// Unlike [`super::ContainerReader`], this view does not retain one tree node
/// per entry. Lookups binary-search the canonical on-disk directory records.
#[derive(Clone)]
pub struct MmapV2DirectoryView {
    source: Arc<MmapRangeSource>,
    info: ContainerInfo,
    directory_offset: usize,
    entry_count: usize,
    table_offset: usize,
    chunk_count: usize,
}

impl fmt::Debug for MmapV2DirectoryView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MmapV2DirectoryView")
            .field("info", &self.info)
            .field("entry_count", &self.entry_count)
            .field("chunk_count", &self.chunk_count)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy)]
struct EntryValidation {
    guid: StableGuid,
    chunk_id: u32,
    raw_offset: u64,
    raw_len: u64,
    hash: [u8; 32],
}

impl MmapV2DirectoryView {
    /// Opens and fully validates a v2 chunk-table container using a read-only
    /// mapping. The file must remain immutable for the lifetime of this view.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self, ContainerError> {
        Self::open_with_budget(path, ContainerReadBudget::default())
    }

    /// Opens a v2 container under an explicit shared read budget.
    pub fn open_with_budget(
        path: impl AsRef<std::path::Path>,
        budget: ContainerReadBudget,
    ) -> Result<Self, ContainerError> {
        let source = MmapRangeSource::open(path)
            .map_err(|error| ContainerError::Source(error.to_string()))?;
        Self::from_source(Arc::new(source), budget)
    }

    /// Fully validates an already-open immutable mapping and retains only its
    /// directory/table ranges rather than a per-entry tree.
    pub fn from_source(
        source: Arc<MmapRangeSource>,
        budget: ContainerReadBudget,
    ) -> Result<Self, ContainerError> {
        let bytes = source.mapped_bytes();
        let limits = budget.limits();
        let source_len = u64::try_from(bytes.len()).map_err(|_| ContainerError::IntegerOverflow)?;
        limits.check("container bytes", source_len, limits.max_container_bytes)?;
        if source_len != source.len() {
            return Err(ContainerError::Truncated);
        }
        if bytes.len() < 12 {
            return Err(ContainerError::Truncated);
        }
        if bytes.get(..8) != Some(&MAGIC) {
            return Err(ContainerError::BadMagic);
        }
        let version = read_u16(bytes, 8)?;
        if version != CHUNK_FORMAT_VERSION {
            return Err(ContainerError::UnsupportedVersion(version));
        }
        if bytes.len() < CHUNK_HEADER_SIZE {
            return Err(ContainerError::Truncated);
        }
        let header = &bytes[..CHUNK_HEADER_SIZE];
        if usize::from(read_u16(header, 10)?) != CHUNK_HEADER_SIZE
            || read_u32(header, 12)? != 0
            || header[136..CHUNK_HEADER_SIZE].iter().any(|byte| *byte != 0)
        {
            return Err(ContainerError::InvalidHeader);
        }

        let entry_count_u32 = read_u32(header, 16)?;
        let chunk_count_u32 = read_u32(header, 20)?;
        let entry_count =
            usize::try_from(entry_count_u32).map_err(|_| ContainerError::IntegerOverflow)?;
        let chunk_count =
            usize::try_from(chunk_count_u32).map_err(|_| ContainerError::IntegerOverflow)?;
        let directory_offset_u64 = read_u64(header, 24)?;
        let directory_len = read_u64(header, 32)?;
        let table_offset_u64 = read_u64(header, 40)?;
        let table_len = read_u64(header, 48)?;
        let payload_offset = read_u64(header, 56)?;
        let total_len = read_u64(header, 64)?;
        let expected_directory_len = u64::from(entry_count_u32)
            .checked_mul(CHUNK_RECORD_SIZE as u64)
            .ok_or(ContainerError::IntegerOverflow)?;
        let expected_table_len = u64::from(chunk_count_u32)
            .checked_mul(CHUNK_TABLE_RECORD_SIZE as u64)
            .ok_or(ContainerError::IntegerOverflow)?;
        let directory_end = directory_offset_u64
            .checked_add(directory_len)
            .ok_or(ContainerError::IntegerOverflow)?;
        let table_end = table_offset_u64
            .checked_add(table_len)
            .ok_or(ContainerError::IntegerOverflow)?;
        if total_len > source_len {
            return Err(ContainerError::Truncated);
        }
        if directory_offset_u64 != CHUNK_HEADER_SIZE as u64
            || directory_len != expected_directory_len
            || table_offset_u64 < directory_end
            || table_offset_u64 % 8 != 0
            || table_len != expected_table_len
            || payload_offset < table_end
            || total_len != source_len
            || payload_offset > total_len
        {
            return Err(ContainerError::InvalidHeader);
        }
        limits.metadata(directory_len, table_len, entry_count_u32, chunk_count_u32)?;

        let directory_range = checked_range(directory_offset_u64, directory_len, bytes.len())?;
        let table_range = checked_range(table_offset_u64, table_len, bytes.len())?;
        let directory = &bytes[directory_range.clone()];
        let table = &bytes[table_range];
        if blake3::hash(directory).as_bytes() != &header[72..104]
            || blake3::hash(table).as_bytes() != &header[104..136]
        {
            return Err(ContainerError::HashMismatch);
        }
        validate_zero(
            bytes,
            directory_end,
            table_offset_u64,
            ContainerError::InvalidHeader,
        )?;
        validate_zero(
            bytes,
            table_end,
            payload_offset,
            ContainerError::InvalidHeader,
        )?;

        let chunk_vector_bytes = u64::from(chunk_count_u32)
            .checked_mul(size_of::<ChunkInfo>() as u64)
            .ok_or(ContainerError::IntegerOverflow)?;
        let chunk_scratch_bytes = chunk_vector_bytes
            .checked_mul(2)
            .ok_or(ContainerError::IntegerOverflow)?;
        let entry_scratch_bytes = u64::from(entry_count_u32)
            .checked_mul(size_of::<EntryValidation>() as u64)
            .ok_or(ContainerError::IntegerOverflow)?;
        let _scratch_reservation = budget.reserve(
            chunk_scratch_bytes
                .checked_add(entry_scratch_bytes)
                .ok_or(ContainerError::IntegerOverflow)?,
        )?;

        let mut chunks = Vec::new();
        chunks
            .try_reserve_exact(chunk_count)
            .map_err(|_| allocation_failed(chunk_scratch_bytes))?;
        let mut validation_decoded_bytes = 0_u64;
        for (index, record) in table.chunks_exact(CHUNK_TABLE_RECORD_SIZE).enumerate() {
            let id = u32::try_from(index).map_err(|_| ContainerError::IntegerOverflow)?;
            let chunk = parse_chunk_record(record, id)?;
            if chunk.codec == Codec::Raw && chunk.stored_len != chunk.raw_len {
                return Err(ContainerError::InvalidDirectory);
            }
            limits.unit(chunk.stored_len, chunk.raw_len)?;
            validation_decoded_bytes = validation_decoded_bytes
                .checked_add(chunk.raw_len)
                .ok_or(ContainerError::IntegerOverflow)?;
            limits.check(
                "validation decoded bytes",
                validation_decoded_bytes,
                limits.max_validation_decoded_bytes,
            )?;
            let chunk_end = chunk
                .offset
                .checked_add(chunk.stored_len)
                .ok_or(ContainerError::IntegerOverflow)?;
            if chunk.offset < payload_offset || chunk_end > total_len {
                return Err(ContainerError::OutOfBounds);
            }
            chunks.push(chunk);
        }

        let mut physical_chunks = Vec::new();
        physical_chunks
            .try_reserve_exact(chunks.len())
            .map_err(|_| allocation_failed(chunk_scratch_bytes))?;
        physical_chunks.extend_from_slice(&chunks);
        physical_chunks.sort_unstable_by_key(|chunk| (chunk.offset, chunk.id));
        let mut stored_cursor = payload_offset;
        let mut raw_cursor = payload_offset;
        for &chunk in &physical_chunks {
            if chunk.offset < stored_cursor || chunk.offset < raw_cursor {
                return Err(ContainerError::OutOfBounds);
            }
            let raw_end = chunk
                .offset
                .checked_add(chunk.raw_len)
                .ok_or(ContainerError::IntegerOverflow)?;
            validate_zero(
                bytes,
                stored_cursor,
                chunk.offset,
                ContainerError::InvalidDirectory,
            )?;
            stored_cursor = chunk
                .offset
                .checked_add(chunk.stored_len)
                .ok_or(ContainerError::IntegerOverflow)?;
            raw_cursor = raw_end;
        }
        if let Some(last) = chunks.iter().max_by_key(|chunk| chunk.offset) {
            let expected_end = last
                .offset
                .checked_add(last.raw_len.max(last.stored_len))
                .ok_or(ContainerError::IntegerOverflow)?;
            if expected_end != total_len {
                return Err(ContainerError::InvalidDirectory);
            }
        } else if payload_offset != total_len {
            return Err(ContainerError::InvalidDirectory);
        }
        validate_zero(
            bytes,
            stored_cursor,
            total_len,
            ContainerError::InvalidDirectory,
        )?;

        let mut entries = Vec::new();
        entries
            .try_reserve_exact(entry_count)
            .map_err(|_| allocation_failed(entry_scratch_bytes))?;
        let mut previous_guid = None;
        for record in directory.chunks_exact(CHUNK_RECORD_SIZE) {
            let guid = read_guid(record)?;
            if previous_guid.is_some_and(|previous| previous >= guid) {
                return Err(ContainerError::InvalidDirectory);
            }
            previous_guid = Some(guid);
            let chunk_id = read_u32(record, 16)?;
            let chunk = chunks
                .get(usize::try_from(chunk_id).map_err(|_| ContainerError::IntegerOverflow)?)
                .copied()
                .ok_or(ContainerError::InvalidDirectory)?;
            let raw_offset = read_u64(record, 24)?;
            let raw_len = read_u64(record, 32)?;
            let raw_end = raw_offset
                .checked_add(raw_len)
                .ok_or(ContainerError::IntegerOverflow)?;
            if record[20..24].iter().any(|byte| *byte != 0)
                || record[76..80].iter().any(|byte| *byte != 0)
                || raw_end > chunk.raw_len
            {
                return Err(ContainerError::InvalidDirectory);
            }
            let alignment = u32::from_le_bytes(
                record[72..76]
                    .try_into()
                    .map_err(|_| ContainerError::InvalidDirectory)?,
            );
            let absolute = chunk
                .offset
                .checked_add(raw_offset)
                .ok_or(ContainerError::IntegerOverflow)?;
            if alignment == 0
                || !alignment.is_power_of_two()
                || absolute % u64::from(alignment) != 0
            {
                return Err(ContainerError::InvalidDirectory);
            }
            let hash = record[40..72]
                .try_into()
                .map_err(|_| ContainerError::InvalidDirectory)?;
            entries.push(EntryValidation {
                guid,
                chunk_id,
                raw_offset,
                raw_len,
                hash,
            });
        }
        let actual_chunk_bytes = capacity_bytes(chunks.capacity(), size_of::<ChunkInfo>())?
            .checked_add(capacity_bytes(
                physical_chunks.capacity(),
                size_of::<ChunkInfo>(),
            )?)
            .ok_or(ContainerError::IntegerOverflow)?;
        let actual_entry_bytes = capacity_bytes(entries.capacity(), size_of::<EntryValidation>())?;
        let actual_scratch_bytes = actual_chunk_bytes
            .checked_add(actual_entry_bytes)
            .ok_or(ContainerError::IntegerOverflow)?;
        let requested_scratch_bytes = chunk_scratch_bytes
            .checked_add(entry_scratch_bytes)
            .ok_or(ContainerError::IntegerOverflow)?;
        let _capacity_adjustment = if actual_scratch_bytes > requested_scratch_bytes {
            Some(budget.reserve(actual_scratch_bytes - requested_scratch_bytes)?)
        } else {
            None
        };
        entries.sort_unstable_by_key(|entry| {
            (entry.chunk_id, entry.raw_offset, entry.raw_len, entry.guid)
        });

        let mapped_source: &dyn RangeSource = source.as_ref();
        let metadata_bytes = directory_len
            .checked_add(table_len)
            .ok_or(ContainerError::IntegerOverflow)?;
        let mut position = 0;
        for chunk_id in 0..chunk_count_u32 {
            let begin = position;
            while position < entries.len() && entries[position].chunk_id == chunk_id {
                position += 1;
            }
            if position == begin {
                return Err(ContainerError::InvalidDirectory);
            }
            let chunk_index =
                usize::try_from(chunk_id).map_err(|_| ContainerError::IntegerOverflow)?;
            let chunk = *chunks
                .get(chunk_index)
                .ok_or(ContainerError::InvalidDirectory)?;
            let decoded = if matches!(chunk.codec, Codec::Deflate | Codec::BlockDeflate) {
                let decoded = decode_chunk_source(
                    mapped_source,
                    chunk,
                    metadata_bytes,
                    &budget,
                    &crate::block::CpuBlockDecodeBackend,
                )?;
                if blake3::hash(&decoded).as_bytes() != &chunk.hash {
                    return Err(ContainerError::HashMismatch);
                }
                Some(decoded)
            } else {
                let range = checked_range(chunk.offset, chunk.raw_len, bytes.len())?;
                if blake3::hash(&bytes[range]).as_bytes() != &chunk.hash {
                    return Err(ContainerError::HashMismatch);
                }
                None
            };
            let mut raw_cursor = 0_u64;
            for entry in &entries[begin..position] {
                if entry.raw_offset < raw_cursor {
                    return Err(ContainerError::InvalidDirectory);
                }
                let entry_end = entry
                    .raw_offset
                    .checked_add(entry.raw_len)
                    .ok_or(ContainerError::IntegerOverflow)?;
                if let Some(decoded) = &decoded {
                    let gap_start =
                        usize::try_from(raw_cursor).map_err(|_| ContainerError::IntegerOverflow)?;
                    let start = usize::try_from(entry.raw_offset)
                        .map_err(|_| ContainerError::IntegerOverflow)?;
                    let end =
                        usize::try_from(entry_end).map_err(|_| ContainerError::IntegerOverflow)?;
                    if decoded[gap_start..start].iter().any(|byte| *byte != 0) {
                        return Err(ContainerError::InvalidDirectory);
                    }
                    if blake3::hash(&decoded[start..end]).as_bytes() != &entry.hash {
                        return Err(ContainerError::HashMismatch);
                    }
                } else {
                    let gap_start = chunk
                        .offset
                        .checked_add(raw_cursor)
                        .ok_or(ContainerError::IntegerOverflow)?;
                    let start = chunk
                        .offset
                        .checked_add(entry.raw_offset)
                        .ok_or(ContainerError::IntegerOverflow)?;
                    let end = chunk
                        .offset
                        .checked_add(entry_end)
                        .ok_or(ContainerError::IntegerOverflow)?;
                    let gap = checked_range(gap_start, start - gap_start, bytes.len())?;
                    if bytes[gap].iter().any(|byte| *byte != 0) {
                        return Err(ContainerError::InvalidDirectory);
                    }
                    let payload = checked_range(start, end - start, bytes.len())?;
                    if blake3::hash(&bytes[payload]).as_bytes() != &entry.hash {
                        return Err(ContainerError::HashMismatch);
                    }
                }
                raw_cursor = entry_end;
            }
            if raw_cursor != chunk.raw_len {
                return Err(ContainerError::InvalidDirectory);
            }
        }
        if position != entries.len() {
            return Err(ContainerError::InvalidDirectory);
        }

        let content_hash = *blake3::hash(bytes).as_bytes();
        let generation = u64::from_le_bytes(
            content_hash[..8]
                .try_into()
                .map_err(|_| ContainerError::InvalidHeader)?,
        );
        Ok(Self {
            source,
            info: ContainerInfo {
                version: CHUNK_FORMAT_VERSION,
                entry_count: entry_count_u32,
                chunk_count: chunk_count_u32,
                total_len,
                content_hash,
                generation,
            },
            directory_offset: usize::try_from(directory_offset_u64)
                .map_err(|_| ContainerError::IntegerOverflow)?,
            entry_count,
            table_offset: usize::try_from(table_offset_u64)
                .map_err(|_| ContainerError::IntegerOverflow)?,
            chunk_count,
        })
    }

    /// Returns the validated container identity and counts.
    #[must_use]
    pub const fn info(&self) -> ContainerInfo {
        self.info
    }

    /// Returns validated directory metadata for one GUID using binary search.
    #[must_use]
    pub fn entry(&self, guid: StableGuid) -> Option<EntryInfo> {
        let bytes = self.source.mapped_bytes();
        let mut low = 0_usize;
        let mut high = self.entry_count;
        while low < high {
            let middle = low + (high - low) / 2;
            let start = self.directory_offset + middle * CHUNK_RECORD_SIZE;
            let record = bytes.get(start..start + CHUNK_RECORD_SIZE)?;
            let current = read_guid(record).ok()?;
            match current.cmp(&guid) {
                core::cmp::Ordering::Less => low = middle + 1,
                core::cmp::Ordering::Greater => high = middle,
                core::cmp::Ordering::Equal => return self.decode_entry_record(record),
            }
        }
        None
    }

    /// Returns whether the validated directory contains `guid`.
    #[must_use]
    pub fn contains(&self, guid: StableGuid) -> bool {
        self.entry(guid).is_some()
    }

    /// Returns validated chunk metadata by its canonical ordinal.
    #[must_use]
    pub fn chunk(&self, id: u32) -> Option<ChunkInfo> {
        let index = usize::try_from(id).ok()?;
        if index >= self.chunk_count {
            return None;
        }
        let start = self
            .table_offset
            .checked_add(index.checked_mul(CHUNK_TABLE_RECORD_SIZE)?)?;
        let record = self
            .source
            .mapped_bytes()
            .get(start..start.checked_add(CHUNK_TABLE_RECORD_SIZE)?)?;
        parse_chunk_record(record, id).ok()
    }

    /// Iterates GUIDs in canonical on-disk order without allocating.
    pub fn guids(&self) -> impl ExactSizeIterator<Item = StableGuid> + '_ {
        self.source.mapped_bytes()
            [self.directory_offset..self.directory_offset + self.entry_count * CHUNK_RECORD_SIZE]
            .chunks_exact(CHUNK_RECORD_SIZE)
            .map(|record| read_guid(record).expect("validated fixed-width GUID record"))
    }

    fn decode_entry_record(&self, record: &[u8]) -> Option<EntryInfo> {
        let guid = read_guid(record).ok()?;
        let chunk_id = read_u32(record, 16).ok()?;
        let chunk = self.chunk(chunk_id)?;
        Some(EntryInfo {
            guid,
            chunk_id,
            raw_offset: read_u64(record, 24).ok()?,
            offset: chunk.offset,
            stored_len: chunk.stored_len,
            raw_len: read_u64(record, 32).ok()?,
            codec: chunk.codec,
            hash: record[40..72].try_into().ok()?,
            alignment: read_u32(record, 72).ok()?,
        })
    }
}

fn parse_chunk_record(record: &[u8], id: u32) -> Result<ChunkInfo, ContainerError> {
    if record.len() != CHUNK_TABLE_RECORD_SIZE
        || record[25..28].iter().any(|byte| *byte != 0)
        || record[60..64].iter().any(|byte| *byte != 0)
    {
        return Err(ContainerError::InvalidDirectory);
    }
    Ok(ChunkInfo {
        id,
        offset: read_u64(record, 0)?,
        stored_len: read_u64(record, 8)?,
        raw_len: read_u64(record, 16)?,
        codec: Codec::decode(record[24])?,
        hash: record[28..60]
            .try_into()
            .map_err(|_| ContainerError::InvalidDirectory)?,
    })
}

fn read_guid(record: &[u8]) -> Result<StableGuid, ContainerError> {
    let raw = record.get(..16).ok_or(ContainerError::InvalidDirectory)?;
    let bytes: [u8; 16] = raw
        .try_into()
        .map_err(|_| ContainerError::InvalidDirectory)?;
    Ok(StableGuid::from_u128(u128::from_le_bytes(bytes)))
}

fn checked_range(offset: u64, len: u64, total: usize) -> Result<Range<usize>, ContainerError> {
    let end = offset
        .checked_add(len)
        .ok_or(ContainerError::IntegerOverflow)?;
    let start = usize::try_from(offset).map_err(|_| ContainerError::IntegerOverflow)?;
    let end = usize::try_from(end).map_err(|_| ContainerError::IntegerOverflow)?;
    if end > total {
        return Err(ContainerError::Truncated);
    }
    Ok(start..end)
}

fn validate_zero(
    bytes: &[u8],
    start: u64,
    end: u64,
    error: ContainerError,
) -> Result<(), ContainerError> {
    if end < start {
        return Err(ContainerError::IntegerOverflow);
    }
    let range = checked_range(start, end - start, bytes.len())?;
    if bytes[range].iter().any(|byte| *byte != 0) {
        return Err(error);
    }
    Ok(())
}

fn allocation_failed(bytes: u64) -> ContainerError {
    ContainerError::AllocationFailed { bytes }
}

fn capacity_bytes(capacity: usize, element_size: usize) -> Result<u64, ContainerError> {
    let capacity = u64::try_from(capacity).map_err(|_| ContainerError::IntegerOverflow)?;
    let element_size = u64::try_from(element_size).map_err(|_| ContainerError::IntegerOverflow)?;
    capacity
        .checked_mul(element_size)
        .ok_or(ContainerError::IntegerOverflow)
}
