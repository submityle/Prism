use alloc::collections::BTreeMap;
use alloc::sync::Arc;

use prism_asset::StableGuid;

use crate::format::{
    read_u16, read_u32, read_u64, Codec, ContainerError, ContainerInfo, EntryInfo, FORMAT_VERSION,
    HEADER_SIZE, MAGIC, RECORD_SIZE,
};

const PLANNED_FORMAT_VERSION: u16 = 2;

/// Strict, immutable in-memory container reader. Mount validates the complete
/// directory and every payload digest before the container can be mounted.
#[derive(Clone, Debug)]
pub struct ContainerReader {
    bytes: Arc<[u8]>,
    info: ContainerInfo,
    entries: BTreeMap<StableGuid, EntryInfo>,
}

impl ContainerReader {
    /// Validates a complete version-1 container and retains its bytes immutably.
    pub fn new(bytes: impl Into<Arc<[u8]>>) -> Result<Self, ContainerError> {
        let bytes = bytes.into();
        if bytes.len() < HEADER_SIZE {
            return Err(ContainerError::Truncated);
        }
        if bytes[0..8] != MAGIC {
            return Err(ContainerError::BadMagic);
        }
        let version = read_u16(&bytes, 8)?;
        if version != FORMAT_VERSION && version != PLANNED_FORMAT_VERSION {
            return Err(ContainerError::UnsupportedVersion(version));
        }
        if usize::from(read_u16(&bytes, 10)?) != HEADER_SIZE || read_u32(&bytes, 12)? != 0 {
            return Err(ContainerError::InvalidHeader);
        }
        let count = read_u32(&bytes, 16)?;
        let dir_off = read_u64(&bytes, 20)?;
        let dir_len = read_u64(&bytes, 28)?;
        let payload_off = read_u64(&bytes, 36)?;
        let total_len = read_u64(&bytes, 44)?;
        if bytes[84..96].iter().any(|v| *v != 0) {
            return Err(ContainerError::InvalidHeader);
        }
        let expected_dir_len = u64::from(count)
            .checked_mul(RECORD_SIZE as u64)
            .ok_or(ContainerError::IntegerOverflow)?;
        let dir_end = dir_off
            .checked_add(dir_len)
            .ok_or(ContainerError::IntegerOverflow)?;
        let byte_len = u64::try_from(bytes.len()).map_err(|_| ContainerError::IntegerOverflow)?;
        if total_len > byte_len {
            return Err(ContainerError::Truncated);
        }
        if dir_off != HEADER_SIZE as u64
            || dir_len != expected_dir_len
            || (version == FORMAT_VERSION && dir_end != payload_off)
            || (version == PLANNED_FORMAT_VERSION && dir_end > payload_off)
            || total_len != byte_len
            || payload_off > total_len
        {
            return Err(ContainerError::InvalidHeader);
        }
        let dir_range = usize::try_from(dir_off).map_err(|_| ContainerError::IntegerOverflow)?
            ..usize::try_from(dir_end).map_err(|_| ContainerError::IntegerOverflow)?;
        let directory = bytes.get(dir_range).ok_or(ContainerError::Truncated)?;
        if blake3::hash(directory).as_bytes() != &bytes[52..84] {
            return Err(ContainerError::HashMismatch);
        }
        let directory_padding = bytes
            .get(
                usize::try_from(dir_end).map_err(|_| ContainerError::IntegerOverflow)?
                    ..usize::try_from(payload_off).map_err(|_| ContainerError::IntegerOverflow)?,
            )
            .ok_or(ContainerError::Truncated)?;
        if directory_padding.iter().any(|byte| *byte != 0) {
            return Err(ContainerError::InvalidHeader);
        }
        let mut entries = BTreeMap::new();
        let mut previous_guid = None;
        for record in directory.chunks_exact(RECORD_SIZE) {
            let guid =
                StableGuid::from_u128(u128::from_le_bytes(record[0..16].try_into().unwrap()));
            if entries.contains_key(&guid) {
                return Err(ContainerError::DuplicateGuid(guid));
            }
            if previous_guid.is_some_and(|previous| previous >= guid) {
                return Err(ContainerError::InvalidDirectory);
            }
            previous_guid = Some(guid);
            let offset = read_u64(record, 16)?;
            let stored_len = read_u64(record, 24)?;
            let raw_len = read_u64(record, 32)?;
            let codec = Codec::decode(record[40])?;
            if record[41..44].iter().any(|v| *v != 0) {
                return Err(ContainerError::InvalidDirectory);
            }
            if codec == Codec::Raw && stored_len != raw_len {
                return Err(ContainerError::InvalidDirectory);
            }
            let end = offset
                .checked_add(stored_len)
                .ok_or(ContainerError::IntegerOverflow)?;
            if offset < payload_off || end > total_len {
                return Err(ContainerError::OutOfBounds);
            }
            let hash = record[44..76].try_into().unwrap();
            let alignment = u32::from_le_bytes(record[76..80].try_into().unwrap());
            if alignment == 0 || !alignment.is_power_of_two() {
                return Err(ContainerError::InvalidAlignment);
            }
            if offset % u64::from(alignment) != 0 {
                return Err(ContainerError::InvalidDirectory);
            }
            let payload_start =
                usize::try_from(offset).map_err(|_| ContainerError::IntegerOverflow)?;
            let payload_end = usize::try_from(end).map_err(|_| ContainerError::IntegerOverflow)?;
            let payload = bytes
                .get(payload_start..payload_end)
                .ok_or(ContainerError::Truncated)?;
            if blake3::hash(payload).as_bytes() != &hash {
                return Err(ContainerError::HashMismatch);
            }
            entries.insert(
                guid,
                EntryInfo {
                    guid,
                    offset,
                    stored_len,
                    raw_len,
                    codec,
                    hash,
                    alignment,
                },
            );
        }
        // The canonical writer emits GUID-ordered payloads with zero padding
        // between entries when alignment requires it. Requiring monotonic,
        // non-overlapping ranges prevents aliasing/overlap tricks.
        let mut ordered: Vec<_> = entries.values().collect();
        if version == FORMAT_VERSION {
            ordered.sort_by_key(|e| e.guid);
        } else {
            ordered.sort_by_key(|e| (e.offset, e.stored_len, e.guid));
        }
        let mut cursor = payload_off;
        for entry in ordered {
            if entry.offset < cursor {
                return Err(ContainerError::OutOfBounds);
            }
            let gap = bytes
                .get(
                    usize::try_from(cursor).map_err(|_| ContainerError::IntegerOverflow)?
                        ..usize::try_from(entry.offset)
                            .map_err(|_| ContainerError::IntegerOverflow)?,
                )
                .ok_or(ContainerError::Truncated)?;
            if gap.iter().any(|byte| *byte != 0) {
                return Err(ContainerError::InvalidDirectory);
            }
            cursor = entry
                .offset
                .checked_add(entry.stored_len)
                .ok_or(ContainerError::IntegerOverflow)?;
        }
        if cursor != total_len {
            return Err(ContainerError::InvalidDirectory);
        }
        Ok(Self {
            bytes,
            info: ContainerInfo {
                version,
                entry_count: count,
                total_len,
            },
            entries,
        })
    }

    #[must_use]
    /// Returns validated top-level metadata.
    pub const fn info(&self) -> ContainerInfo {
        self.info
    }
    #[must_use]
    /// Returns whether this container contains the specified asset.
    pub fn contains(&self, guid: StableGuid) -> bool {
        self.entries.contains_key(&guid)
    }
    #[must_use]
    /// Returns validated TOC metadata for an asset, if present.
    pub fn entry(&self, guid: StableGuid) -> Option<EntryInfo> {
        self.entries.get(&guid).copied()
    }

    /// GUIDs in canonical on-disk order.
    pub fn guids(&self) -> impl Iterator<Item = StableGuid> + '_ {
        self.entries.keys().copied()
    }

    /// Reads a byte range from a raw payload. Payload integrity was verified
    /// in full when this reader was constructed.
    pub fn read_range(
        &self,
        guid: StableGuid,
        offset: u64,
        out: &mut [u8],
    ) -> Result<(), ContainerError> {
        let entry = self
            .entries
            .get(&guid)
            .ok_or(ContainerError::MissingAsset(guid))?;
        let len = u64::try_from(out.len()).map_err(|_| ContainerError::IntegerOverflow)?;
        let end = offset
            .checked_add(len)
            .ok_or(ContainerError::IntegerOverflow)?;
        if end > entry.raw_len {
            return Err(ContainerError::InvalidRange);
        }
        let start = usize::try_from(
            entry
                .offset
                .checked_add(offset)
                .ok_or(ContainerError::IntegerOverflow)?,
        )
        .map_err(|_| ContainerError::IntegerOverflow)?;
        let finish = start
            .checked_add(out.len())
            .ok_or(ContainerError::IntegerOverflow)?;
        let payload = self
            .bytes
            .get(start..finish)
            .ok_or(ContainerError::OutOfBounds)?;
        out.copy_from_slice(payload);
        Ok(())
    }

    /// Reads and verifies the complete raw payload.
    pub fn read_all(&self, guid: StableGuid) -> Result<Vec<u8>, ContainerError> {
        let entry = self
            .entries
            .get(&guid)
            .ok_or(ContainerError::MissingAsset(guid))?;
        let len = usize::try_from(entry.raw_len).map_err(|_| ContainerError::IntegerOverflow)?;
        let mut out = vec![0; len];
        self.read_range(guid, 0, &mut out)?;
        Ok(out)
    }
}
