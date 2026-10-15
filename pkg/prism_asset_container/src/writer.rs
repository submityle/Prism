use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use std::io::Write;

use flate2::{write::DeflateEncoder, Compression};
use prism_asset::StableGuid;

use crate::chunk::ChunkPlan;
use crate::format::{
    Codec, ContainerError, CHUNK_FORMAT_VERSION, CHUNK_HEADER_SIZE, CHUNK_RECORD_SIZE,
    CHUNK_TABLE_RECORD_SIZE, FORMAT_VERSION, HEADER_SIZE, MAGIC, RECORD_SIZE,
};

/// One uncompressed asset supplied to the deterministic writer.
#[derive(Clone, Debug)]
pub struct WriteEntry {
    /// Stable identity to store in the table of contents.
    pub guid: StableGuid,
    /// Raw, uncompressed asset bytes.
    pub bytes: Vec<u8>,
}

#[derive(Debug)]
struct EncodedEntry {
    raw: Vec<u8>,
    stored: Vec<u8>,
    codec: Codec,
    alignment: u32,
}

/// Deterministic version-1 runtime container writer.
#[derive(Default, Debug)]
pub struct ContainerWriter {
    entries: BTreeMap<StableGuid, EncodedEntry>,
}

impl ContainerWriter {
    /// Creates an empty container writer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a raw payload. Duplicate GUIDs are rejected, never silently replaced.
    pub fn add(
        &mut self,
        guid: StableGuid,
        bytes: impl Into<Vec<u8>>,
    ) -> Result<(), ContainerError> {
        self.add_aligned(guid, bytes, 1)
    }

    /// Adds a raw payload with an absolute power-of-two start alignment.
    pub fn add_aligned(
        &mut self,
        guid: StableGuid,
        bytes: impl Into<Vec<u8>>,
        alignment: u32,
    ) -> Result<(), ContainerError> {
        if self.entries.contains_key(&guid) {
            return Err(ContainerError::DuplicateGuid(guid));
        }
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(ContainerError::InvalidAlignment);
        }
        let bytes = bytes.into();
        self.entries.insert(
            guid,
            EncodedEntry {
                stored: bytes.clone(),
                raw: bytes,
                codec: Codec::Raw,
                alignment,
            },
        );
        Ok(())
    }

    /// Adds a raw payload encoded as a deterministic raw-DEFLATE stream.
    ///
    /// The directory hash always covers the decoded bytes, while the physical
    /// container hash covers the compressed bytes through the normal container
    /// identity.  Range reads for this codec decode the complete payload and
    /// then return the requested raw subrange.
    pub fn add_deflate(
        &mut self,
        guid: StableGuid,
        bytes: impl Into<Vec<u8>>,
    ) -> Result<(), ContainerError> {
        self.add_deflate_aligned(guid, bytes, 1)
    }

    /// Adds a raw payload encoded as raw-DEFLATE with an absolute alignment.
    pub fn add_deflate_aligned(
        &mut self,
        guid: StableGuid,
        bytes: impl Into<Vec<u8>>,
        alignment: u32,
    ) -> Result<(), ContainerError> {
        if self.entries.contains_key(&guid) {
            return Err(ContainerError::DuplicateGuid(guid));
        }
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(ContainerError::InvalidAlignment);
        }
        let raw = bytes.into();
        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
        encoder
            .write_all(&raw)
            .map_err(|_| ContainerError::CompressionFailed)?;
        let stored = encoder
            .finish()
            .map_err(|_| ContainerError::CompressionFailed)?;
        self.entries.insert(
            guid,
            EncodedEntry {
                raw,
                stored,
                codec: Codec::Deflate,
                alignment,
            },
        );
        Ok(())
    }

    /// Emits canonical little-endian bytes with GUID-sorted records and BLAKE3
    /// hashes for both the directory and each payload.
    pub fn finish(self) -> Result<Vec<u8>, ContainerError> {
        let entries: Vec<_> = self.entries.into_iter().collect();
        let count = u32::try_from(entries.len()).map_err(|_| ContainerError::IntegerOverflow)?;
        let directory_len = entries
            .len()
            .checked_mul(RECORD_SIZE)
            .ok_or(ContainerError::IntegerOverflow)?;
        let payload_start = HEADER_SIZE
            .checked_add(directory_len)
            .ok_or(ContainerError::IntegerOverflow)?;
        let mut payloads = Vec::with_capacity(entries.len());
        let mut directory = Vec::with_capacity(directory_len);
        let mut offset =
            u64::try_from(payload_start).map_err(|_| ContainerError::IntegerOverflow)?;
        for (guid, entry) in &entries {
            let bytes = &entry.stored;
            let alignment = entry.alignment;
            let align = u64::from(alignment);
            let remainder = offset % align;
            let padding = if remainder == 0 { 0 } else { align - remainder };
            offset = offset
                .checked_add(padding)
                .ok_or(ContainerError::IntegerOverflow)?;
            directory.extend_from_slice(&guid.to_u128().to_le_bytes());
            directory.extend_from_slice(&offset.to_le_bytes());
            let len = u64::try_from(bytes.len()).map_err(|_| ContainerError::IntegerOverflow)?;
            let raw_len =
                u64::try_from(entry.raw.len()).map_err(|_| ContainerError::IntegerOverflow)?;
            directory.extend_from_slice(&len.to_le_bytes());
            directory.extend_from_slice(&raw_len.to_le_bytes());
            directory.push(entry.codec as u8);
            directory.extend_from_slice(&[0; 3]);
            directory.extend_from_slice(blake3::hash(&entry.raw).as_bytes());
            directory.extend_from_slice(&alignment.to_le_bytes());
            let next = offset
                .checked_add(len)
                .ok_or(ContainerError::IntegerOverflow)?;
            offset = next;
            payloads.push(bytes.clone());
        }
        let total_len = usize::try_from(offset).map_err(|_| ContainerError::IntegerOverflow)?;
        let mut out = vec![0; HEADER_SIZE];
        out[0..8].copy_from_slice(&MAGIC);
        out[8..10].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        out[10..12].copy_from_slice(&(HEADER_SIZE as u16).to_le_bytes());
        out[12..16].copy_from_slice(&0_u32.to_le_bytes());
        out[16..20].copy_from_slice(&count.to_le_bytes());
        out[20..28].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
        out[28..36].copy_from_slice(&(directory_len as u64).to_le_bytes());
        out[36..44].copy_from_slice(&(payload_start as u64).to_le_bytes());
        out[44..52].copy_from_slice(&(total_len as u64).to_le_bytes());
        out[52..84].copy_from_slice(blake3::hash(&directory).as_bytes());
        out[84..96].fill(0);
        out.extend_from_slice(&directory);
        let mut cursor =
            u64::try_from(payload_start).map_err(|_| ContainerError::IntegerOverflow)?;
        for ((_, entry), payload) in entries.iter().zip(payloads.iter()) {
            let align = u64::from(entry.alignment);
            let remainder = cursor % align;
            let padding = if remainder == 0 { 0 } else { align - remainder };
            let padding_len =
                usize::try_from(padding).map_err(|_| ContainerError::IntegerOverflow)?;
            out.resize(
                out.len()
                    .checked_add(padding_len)
                    .ok_or(ContainerError::IntegerOverflow)?,
                0,
            );
            cursor = cursor
                .checked_add(padding)
                .ok_or(ContainerError::IntegerOverflow)?;
            out.extend_from_slice(payload);
            cursor = cursor
                .checked_add(
                    u64::try_from(payload.len()).map_err(|_| ContainerError::IntegerOverflow)?,
                )
                .ok_or(ContainerError::IntegerOverflow)?;
        }
        // Directory offsets were computed with the same alignment pass below;
        // keeping this assertion catches accidental divergence in maintenance.
        debug_assert_eq!(out.len(), usize::try_from(cursor).unwrap());
        debug_assert_eq!(out.len(), total_len);
        Ok(out)
    }

    /// Emits a version-1 container using an already validated chunk plan.
    ///
    /// The plan is consumed as layout data, rather than merely checked: every
    /// placement controls the physical payload offset and every chunk gap is
    /// materialized as zero padding. `chunk_alignment` is passed explicitly
    /// because [`ChunkPlan`] intentionally contains only the resulting layout
    /// and not the policy that produced it.
    pub fn finish_with_chunk_plan(
        self,
        plan: &ChunkPlan,
        chunk_alignment: u64,
    ) -> Result<Vec<u8>, ContainerError> {
        if chunk_alignment == 0 || !chunk_alignment.is_power_of_two() {
            return Err(ContainerError::InvalidAlignment);
        }

        let entries = self.entries;
        if let Some(entry) = entries.values().find(|entry| entry.codec != Codec::Raw) {
            return Err(ContainerError::CodecNotSupportedForLayout(entry.codec));
        }
        if plan.assignments.len() != entries.len() {
            return Err(ContainerError::InvalidDirectory);
        }

        let mut placements = BTreeMap::new();
        let mut previous_chunk_end = 0_u64;
        for (index, chunk) in plan.chunks.iter().enumerate() {
            if chunk.start_alignment == 0 || !chunk.start_alignment.is_power_of_two() {
                return Err(ContainerError::InvalidDirectory);
            }
            let expected_chunk_offset = align_up(
                previous_chunk_end,
                chunk_alignment.max(chunk.start_alignment),
            )?;
            if chunk.id != index || chunk.offset != expected_chunk_offset {
                return Err(ContainerError::InvalidDirectory);
            }
            if chunk.placements.is_empty() {
                return Err(ContainerError::InvalidDirectory);
            }
            let mut placement_cursor = 0_u64;
            for placement in &chunk.placements {
                if plan.assignments.get(&placement.guid) != Some(&chunk.id)
                    || placements
                        .insert(placement.guid, (chunk.id, *placement))
                        .is_some()
                {
                    return Err(ContainerError::InvalidDirectory);
                }
                let (_, entry) = entries
                    .get_key_value(&placement.guid)
                    .ok_or(ContainerError::InvalidDirectory)?;
                let required_alignment = chunk_alignment.max(u64::from(entry.alignment));
                let expected_offset = align_up(placement_cursor, required_alignment)?;
                if placement.offset != expected_offset {
                    return Err(ContainerError::InvalidDirectory);
                }
                placement_cursor = placement
                    .offset
                    .checked_add(placement.size)
                    .ok_or(ContainerError::IntegerOverflow)?;
                if placement_cursor > chunk.raw_size {
                    return Err(ContainerError::OutOfBounds);
                }
            }
            if placement_cursor != chunk.raw_size {
                return Err(ContainerError::InvalidDirectory);
            }
            previous_chunk_end = chunk
                .offset
                .checked_add(chunk.raw_size)
                .ok_or(ContainerError::IntegerOverflow)?;
        }
        if placements.len() != entries.len()
            || plan
                .assignments
                .keys()
                .any(|guid| !placements.contains_key(guid))
        {
            return Err(ContainerError::InvalidDirectory);
        }

        let count = u32::try_from(entries.len()).map_err(|_| ContainerError::IntegerOverflow)?;
        let directory_len = entries
            .len()
            .checked_mul(RECORD_SIZE)
            .ok_or(ContainerError::IntegerOverflow)?;
        let directory_end = HEADER_SIZE
            .checked_add(directory_len)
            .ok_or(ContainerError::IntegerOverflow)?;
        let payload_alignment = plan
            .chunks
            .iter()
            .fold(chunk_alignment, |alignment, chunk| {
                alignment.max(chunk.start_alignment)
            });
        let payload_start = align_up(
            u64::try_from(directory_end).map_err(|_| ContainerError::IntegerOverflow)?,
            payload_alignment,
        )?;

        let mut records = Vec::with_capacity(directory_len);
        let mut payload_ranges = Vec::with_capacity(entries.len());
        for (guid, entry) in &entries {
            let bytes = &entry.raw;
            let alignment = entry.alignment;
            let (_, placement) = placements
                .get(guid)
                .copied()
                .ok_or(ContainerError::InvalidDirectory)?;
            let chunk_offset = plan
                .chunks
                .get(plan.assignments[guid])
                .ok_or(ContainerError::InvalidDirectory)?
                .offset;
            let offset = payload_start
                .checked_add(chunk_offset)
                .and_then(|value| value.checked_add(placement.offset))
                .ok_or(ContainerError::IntegerOverflow)?;
            if offset % u64::from(alignment) != 0 {
                return Err(ContainerError::InvalidDirectory);
            }
            let len = u64::try_from(bytes.len()).map_err(|_| ContainerError::IntegerOverflow)?;
            if len != placement.size {
                return Err(ContainerError::InvalidDirectory);
            }
            let end = offset
                .checked_add(len)
                .ok_or(ContainerError::IntegerOverflow)?;
            records.extend_from_slice(&guid.to_u128().to_le_bytes());
            records.extend_from_slice(&offset.to_le_bytes());
            records.extend_from_slice(&len.to_le_bytes());
            records.extend_from_slice(&len.to_le_bytes());
            records.push(Codec::Raw as u8);
            records.extend_from_slice(&[0; 3]);
            records.extend_from_slice(blake3::hash(bytes).as_bytes());
            records.extend_from_slice(&alignment.to_le_bytes());
            payload_ranges.push((offset, end, bytes.as_slice()));
        }

        let total_len_u64 = payload_start
            .checked_add(previous_chunk_end)
            .ok_or(ContainerError::IntegerOverflow)?;
        let total_len =
            usize::try_from(total_len_u64).map_err(|_| ContainerError::IntegerOverflow)?;
        let payload_start_u64 = payload_start;
        let mut out = vec![0; total_len];
        out[0..8].copy_from_slice(&MAGIC);
        out[8..10].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        out[10..12].copy_from_slice(&(HEADER_SIZE as u16).to_le_bytes());
        out[12..16].copy_from_slice(&0_u32.to_le_bytes());
        out[16..20].copy_from_slice(&count.to_le_bytes());
        out[20..28].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
        out[28..36].copy_from_slice(&(directory_len as u64).to_le_bytes());
        out[36..44].copy_from_slice(&payload_start_u64.to_le_bytes());
        out[44..52].copy_from_slice(&total_len_u64.to_le_bytes());
        out[52..84].copy_from_slice(blake3::hash(&records).as_bytes());
        out[84..96].fill(0);
        out[HEADER_SIZE..directory_end].copy_from_slice(&records);
        for (offset, end, bytes) in payload_ranges {
            let start = usize::try_from(offset).map_err(|_| ContainerError::IntegerOverflow)?;
            let finish = usize::try_from(end).map_err(|_| ContainerError::IntegerOverflow)?;
            out.get_mut(start..finish)
                .ok_or(ContainerError::OutOfBounds)?
                .copy_from_slice(bytes);
        }
        Ok(out)
    }

    /// Emits the v2 chunk-table layout. Each planned chunk is encoded
    /// independently, preserving raw offsets inside the chunk so a request
    /// touches only its owning chunk.
    pub fn finish_with_chunk_compression(
        self,
        plan: &ChunkPlan,
        chunk_alignment: u64,
        codec: Codec,
    ) -> Result<Vec<u8>, ContainerError> {
        if chunk_alignment == 0 || !chunk_alignment.is_power_of_two() {
            return Err(ContainerError::InvalidAlignment);
        }
        if !matches!(codec, Codec::Raw | Codec::Deflate) {
            return Err(ContainerError::UnsupportedCodec(codec as u8));
        }
        if self.entries.values().any(|entry| entry.codec != Codec::Raw)
            || plan.assignments.len() != self.entries.len()
        {
            return Err(ContainerError::InvalidDirectory);
        }

        let mut placements = BTreeMap::new();
        let mut previous_end = 0_u64;
        for (index, chunk) in plan.chunks.iter().enumerate() {
            if chunk.id != index
                || chunk.start_alignment == 0
                || !chunk.start_alignment.is_power_of_two()
                || chunk.offset
                    != align_up(previous_end, chunk_alignment.max(chunk.start_alignment))?
                || chunk.placements.is_empty()
            {
                return Err(ContainerError::InvalidDirectory);
            }
            let mut cursor = 0_u64;
            for placement in &chunk.placements {
                if plan.assignments.get(&placement.guid) != Some(&chunk.id)
                    || placements
                        .insert(placement.guid, (chunk.id, *placement))
                        .is_some()
                {
                    return Err(ContainerError::InvalidDirectory);
                }
                let entry = self
                    .entries
                    .get(&placement.guid)
                    .ok_or(ContainerError::InvalidDirectory)?;
                if placement.offset
                    != align_up(cursor, chunk_alignment.max(u64::from(entry.alignment)))?
                    || placement.size
                        != u64::try_from(entry.raw.len())
                            .map_err(|_| ContainerError::IntegerOverflow)?
                {
                    return Err(ContainerError::InvalidDirectory);
                }
                cursor = placement
                    .offset
                    .checked_add(placement.size)
                    .ok_or(ContainerError::IntegerOverflow)?;
                if cursor > chunk.raw_size {
                    return Err(ContainerError::OutOfBounds);
                }
            }
            if cursor != chunk.raw_size {
                return Err(ContainerError::InvalidDirectory);
            }
            previous_end = chunk
                .offset
                .checked_add(chunk.raw_size)
                .ok_or(ContainerError::IntegerOverflow)?;
        }
        if placements.len() != self.entries.len() {
            return Err(ContainerError::InvalidDirectory);
        }

        let count =
            u32::try_from(self.entries.len()).map_err(|_| ContainerError::IntegerOverflow)?;
        let chunk_count =
            u32::try_from(plan.chunks.len()).map_err(|_| ContainerError::IntegerOverflow)?;
        let directory_len = self
            .entries
            .len()
            .checked_mul(CHUNK_RECORD_SIZE)
            .ok_or(ContainerError::IntegerOverflow)?;
        let table_len = plan
            .chunks
            .len()
            .checked_mul(CHUNK_TABLE_RECORD_SIZE)
            .ok_or(ContainerError::IntegerOverflow)?;
        let directory_end = CHUNK_HEADER_SIZE
            .checked_add(directory_len)
            .ok_or(ContainerError::IntegerOverflow)?;
        let table_offset = align_up(
            u64::try_from(directory_end).map_err(|_| ContainerError::IntegerOverflow)?,
            8,
        )?;
        let table_end = table_offset
            .checked_add(u64::try_from(table_len).map_err(|_| ContainerError::IntegerOverflow)?)
            .ok_or(ContainerError::IntegerOverflow)?;
        let payload_alignment = plan
            .chunks
            .iter()
            .fold(chunk_alignment, |a, c| a.max(c.start_alignment));
        let payload_start = align_up(table_end, payload_alignment)?;

        let mut raw_chunks = Vec::with_capacity(plan.chunks.len());
        for chunk in &plan.chunks {
            let mut raw = vec![
                0;
                usize::try_from(chunk.raw_size)
                    .map_err(|_| ContainerError::IntegerOverflow)?
            ];
            for placement in &chunk.placements {
                let bytes = &self
                    .entries
                    .get(&placement.guid)
                    .ok_or(ContainerError::InvalidDirectory)?
                    .raw;
                let start = usize::try_from(placement.offset)
                    .map_err(|_| ContainerError::IntegerOverflow)?;
                let end = start
                    .checked_add(bytes.len())
                    .ok_or(ContainerError::IntegerOverflow)?;
                raw.get_mut(start..end)
                    .ok_or(ContainerError::OutOfBounds)?
                    .copy_from_slice(bytes);
            }
            let stored = encode_chunk(&raw, codec)?;
            raw_chunks.push((raw, stored));
        }

        let mut directory = Vec::with_capacity(directory_len);
        for (guid, entry) in &self.entries {
            let (chunk_id, placement) = placements
                .get(guid)
                .copied()
                .ok_or(ContainerError::InvalidDirectory)?;
            let chunk = plan
                .chunks
                .get(chunk_id)
                .ok_or(ContainerError::InvalidDirectory)?;
            let absolute = payload_start
                .checked_add(chunk.offset)
                .and_then(|v| v.checked_add(placement.offset))
                .ok_or(ContainerError::IntegerOverflow)?;
            if absolute % u64::from(entry.alignment) != 0 {
                return Err(ContainerError::InvalidDirectory);
            }
            directory.extend_from_slice(&guid.to_u128().to_le_bytes());
            directory.extend_from_slice(
                &u32::try_from(chunk_id)
                    .map_err(|_| ContainerError::IntegerOverflow)?
                    .to_le_bytes(),
            );
            directory.extend_from_slice(&0_u32.to_le_bytes());
            directory.extend_from_slice(&placement.offset.to_le_bytes());
            directory.extend_from_slice(&placement.size.to_le_bytes());
            directory.extend_from_slice(blake3::hash(&entry.raw).as_bytes());
            directory.extend_from_slice(&entry.alignment.to_le_bytes());
            directory.extend_from_slice(&0_u32.to_le_bytes());
        }
        let mut table = Vec::with_capacity(table_len);
        for (chunk, (raw, stored)) in plan.chunks.iter().zip(raw_chunks.iter()) {
            table.extend_from_slice(
                &payload_start
                    .checked_add(chunk.offset)
                    .ok_or(ContainerError::IntegerOverflow)?
                    .to_le_bytes(),
            );
            table.extend_from_slice(
                &u64::try_from(stored.len())
                    .map_err(|_| ContainerError::IntegerOverflow)?
                    .to_le_bytes(),
            );
            table.extend_from_slice(
                &u64::try_from(raw.len())
                    .map_err(|_| ContainerError::IntegerOverflow)?
                    .to_le_bytes(),
            );
            table.push(codec as u8);
            table.extend_from_slice(&[0; 3]);
            table.extend_from_slice(blake3::hash(raw).as_bytes());
            table.extend_from_slice(&0_u32.to_le_bytes());
        }

        let stored_end = plan.chunks.iter().zip(raw_chunks.iter()).try_fold(
            0_u64,
            |end, (chunk, (_, stored))| {
                chunk
                    .offset
                    .checked_add(
                        u64::try_from(stored.len()).map_err(|_| ContainerError::IntegerOverflow)?,
                    )
                    .map(|chunk_end| end.max(chunk_end))
                    .ok_or(ContainerError::IntegerOverflow)
            },
        )?;
        let physical_end = previous_end.max(stored_end);
        let total_len_u64 = payload_start
            .checked_add(physical_end)
            .ok_or(ContainerError::IntegerOverflow)?;
        let mut out =
            vec![0; usize::try_from(total_len_u64).map_err(|_| ContainerError::IntegerOverflow)?];
        out[0..8].copy_from_slice(&MAGIC);
        out[8..10].copy_from_slice(&CHUNK_FORMAT_VERSION.to_le_bytes());
        out[10..12].copy_from_slice(&(CHUNK_HEADER_SIZE as u16).to_le_bytes());
        out[16..20].copy_from_slice(&count.to_le_bytes());
        out[20..24].copy_from_slice(&chunk_count.to_le_bytes());
        out[24..32].copy_from_slice(&(CHUNK_HEADER_SIZE as u64).to_le_bytes());
        out[32..40].copy_from_slice(&(directory_len as u64).to_le_bytes());
        out[40..48].copy_from_slice(&table_offset.to_le_bytes());
        out[48..56].copy_from_slice(&(table_len as u64).to_le_bytes());
        out[56..64].copy_from_slice(&payload_start.to_le_bytes());
        out[64..72].copy_from_slice(&total_len_u64.to_le_bytes());
        out[72..104].copy_from_slice(blake3::hash(&directory).as_bytes());
        out[104..136].copy_from_slice(blake3::hash(&table).as_bytes());
        out[CHUNK_HEADER_SIZE..directory_end].copy_from_slice(&directory);
        let table_start =
            usize::try_from(table_offset).map_err(|_| ContainerError::IntegerOverflow)?;
        out[table_start..table_start + table_len].copy_from_slice(&table);
        for (chunk, (_, stored)) in plan.chunks.iter().zip(raw_chunks.iter()) {
            let start = usize::try_from(
                payload_start
                    .checked_add(chunk.offset)
                    .ok_or(ContainerError::IntegerOverflow)?,
            )
            .map_err(|_| ContainerError::IntegerOverflow)?;
            out[start..start + stored.len()].copy_from_slice(stored);
        }
        Ok(out)
    }
}

fn encode_chunk(raw: &[u8], codec: Codec) -> Result<Vec<u8>, ContainerError> {
    match codec {
        Codec::Raw => Ok(raw.to_vec()),
        Codec::Deflate => {
            let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
            encoder
                .write_all(raw)
                .map_err(|_| ContainerError::CompressionFailed)?;
            encoder
                .finish()
                .map_err(|_| ContainerError::CompressionFailed)
        }
    }
}

fn align_up(value: u64, alignment: u64) -> Result<u64, ContainerError> {
    let mask = alignment
        .checked_sub(1)
        .ok_or(ContainerError::IntegerOverflow)?;
    value
        .checked_add(mask)
        .map(|value| value & !mask)
        .ok_or(ContainerError::IntegerOverflow)
}
