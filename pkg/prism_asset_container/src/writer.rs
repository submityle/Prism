use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use prism_asset::StableGuid;

use crate::chunk::ChunkPlan;
use crate::format::{Codec, ContainerError, FORMAT_VERSION, HEADER_SIZE, MAGIC, RECORD_SIZE};

const PLANNED_FORMAT_VERSION: u16 = 2;

/// One uncompressed asset supplied to the deterministic writer.
#[derive(Clone, Debug)]
pub struct WriteEntry {
    /// Stable identity to store in the table of contents.
    pub guid: StableGuid,
    /// Raw, uncompressed asset bytes.
    pub bytes: Vec<u8>,
}

/// Deterministic version-1 runtime container writer.
#[derive(Default, Debug)]
pub struct ContainerWriter {
    entries: BTreeMap<StableGuid, (Vec<u8>, u32)>,
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
        self.entries.insert(guid, (bytes.into(), alignment));
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
        for (guid, (bytes, alignment)) in &entries {
            let align = u64::from(*alignment);
            let remainder = offset % align;
            let padding = if remainder == 0 { 0 } else { align - remainder };
            offset = offset
                .checked_add(padding)
                .ok_or(ContainerError::IntegerOverflow)?;
            directory.extend_from_slice(&guid.to_u128().to_le_bytes());
            directory.extend_from_slice(&offset.to_le_bytes());
            let len = u64::try_from(bytes.len()).map_err(|_| ContainerError::IntegerOverflow)?;
            directory.extend_from_slice(&len.to_le_bytes());
            directory.extend_from_slice(&len.to_le_bytes());
            directory.push(Codec::Raw as u8);
            directory.extend_from_slice(&[0; 3]);
            directory.extend_from_slice(blake3::hash(&bytes).as_bytes());
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
        for ((_, (_, alignment)), payload) in entries.iter().zip(payloads.iter()) {
            let align = u64::from(*alignment);
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

    /// Emits a container using an already validated chunk plan.
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
        if plan.assignments.len() != entries.len() {
            return Err(ContainerError::InvalidDirectory);
        }

        let mut placements = BTreeMap::new();
        let mut previous_chunk_end = 0_u64;
        for (index, chunk) in plan.chunks.iter().enumerate() {
            let expected_chunk_offset = align_up(previous_chunk_end, chunk_alignment)?;
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
                let (_, (_, asset_alignment)) = entries
                    .get_key_value(&placement.guid)
                    .ok_or(ContainerError::InvalidDirectory)?;
                let required_alignment = chunk_alignment.max(u64::from(*asset_alignment));
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
        let payload_start = align_up(
            u64::try_from(directory_end).map_err(|_| ContainerError::IntegerOverflow)?,
            chunk_alignment,
        )?;

        let mut records = Vec::with_capacity(directory_len);
        let mut payload_ranges = Vec::with_capacity(entries.len());
        for (guid, (bytes, alignment)) in &entries {
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
            if offset % u64::from(*alignment) != 0 {
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
        out[8..10].copy_from_slice(&PLANNED_FORMAT_VERSION.to_le_bytes());
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
