use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use prism_asset::StableGuid;

use crate::format::{
    read_u16, read_u32, read_u64, Codec, ContainerError, ContainerInfo, EntryInfo, FORMAT_VERSION,
    HEADER_SIZE, MAGIC, RECORD_SIZE,
};

const PLANNED_FORMAT_VERSION: u16 = 2;

const VALIDATION_CHUNK_SIZE: usize = 64 * 1024;

#[derive(Clone, Debug)]
enum Backing {
    Memory(Arc<[u8]>),
    Range(Arc<dyn crate::RangeSource>),
}

/// Strict container reader. Mount validates the complete directory and every
/// payload digest before the container can be mounted. A range-backed reader
/// retains only the source and validated metadata; it never caches the whole
/// file.
#[derive(Clone, Debug)]
pub struct ContainerReader {
    backing: Backing,
    info: ContainerInfo,
    entries: BTreeMap<StableGuid, EntryInfo>,
}

#[derive(Debug)]
struct ParsedContainer {
    info: ContainerInfo,
    entries: BTreeMap<StableGuid, EntryInfo>,
}

impl ContainerReader {
    /// Validates a complete container and retains its bytes immutably.
    pub fn new(bytes: impl Into<Arc<[u8]>>) -> Result<Self, ContainerError> {
        let bytes = bytes.into();
        let source = crate::MemoryRangeSource::new(bytes.clone());
        let parsed = parse_source(&source)?;
        Ok(Self {
            backing: Backing::Memory(bytes),
            info: parsed.info,
            entries: parsed.entries,
        })
    }

    /// Mounts and strictly validates a container through a seekable/range
    /// source. The source length must exactly match the header's `total_len`.
    /// Header, ToC, padding, and every raw payload are verified at mount time
    /// in bounded chunks; the source is then used for all payload reads.
    pub fn from_range_source<S>(source: Arc<S>) -> Result<Self, ContainerError>
    where
        S: crate::RangeSource + 'static,
    {
        let source: Arc<dyn crate::RangeSource> = source;
        Self::from_shared_range_source(source)
    }

    /// Mounts a range source that is already held behind a trait-object `Arc`.
    pub fn from_shared_range_source(
        source: Arc<dyn crate::RangeSource>,
    ) -> Result<Self, ContainerError> {
        let parsed = parse_source(source.as_ref())?;
        Ok(Self {
            backing: Backing::Range(source),
            info: parsed.info,
            entries: parsed.entries,
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
        let physical_offset = entry
            .offset
            .checked_add(offset)
            .ok_or(ContainerError::IntegerOverflow)?;
        match &self.backing {
            Backing::Memory(bytes) => {
                let start = usize::try_from(physical_offset)
                    .map_err(|_| ContainerError::IntegerOverflow)?;
                let finish = start
                    .checked_add(out.len())
                    .ok_or(ContainerError::IntegerOverflow)?;
                let payload = bytes
                    .get(start..finish)
                    .ok_or(ContainerError::OutOfBounds)?;
                out.copy_from_slice(payload);
                Ok(())
            }
            Backing::Range(source) => read_from_source(source.as_ref(), physical_offset, out),
        }
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

fn read_from_source(
    source: &dyn crate::RangeSource,
    offset: u64,
    out: &mut [u8],
) -> Result<(), ContainerError> {
    let end = offset
        .checked_add(u64::try_from(out.len()).map_err(|_| ContainerError::IntegerOverflow)?)
        .ok_or(ContainerError::IntegerOverflow)?;
    if end > source.len() {
        return Err(ContainerError::Truncated);
    }
    source
        .read_range(offset, out)
        .map_err(|error| ContainerError::Source(error.to_string()))
}

fn parse_source(source: &dyn crate::RangeSource) -> Result<ParsedContainer, ContainerError> {
    let mut header = [0_u8; HEADER_SIZE];
    read_from_source(source, 0, &mut header)?;
    if header[0..8] != MAGIC {
        return Err(ContainerError::BadMagic);
    }
    let version = read_u16(&header, 8)?;
    if version != FORMAT_VERSION && version != PLANNED_FORMAT_VERSION {
        return Err(ContainerError::UnsupportedVersion(version));
    }
    if usize::from(read_u16(&header, 10)?) != HEADER_SIZE || read_u32(&header, 12)? != 0 {
        return Err(ContainerError::InvalidHeader);
    }
    let count = read_u32(&header, 16)?;
    let dir_off = read_u64(&header, 20)?;
    let dir_len = read_u64(&header, 28)?;
    let payload_off = read_u64(&header, 36)?;
    let total_len = read_u64(&header, 44)?;
    if header[84..96].iter().any(|v| *v != 0) {
        return Err(ContainerError::InvalidHeader);
    }
    let expected_dir_len = u64::from(count)
        .checked_mul(RECORD_SIZE as u64)
        .ok_or(ContainerError::IntegerOverflow)?;
    let dir_end = dir_off
        .checked_add(dir_len)
        .ok_or(ContainerError::IntegerOverflow)?;
    let source_len = source.len();
    if total_len > source_len {
        return Err(ContainerError::Truncated);
    }
    if dir_off != HEADER_SIZE as u64
        || dir_len != expected_dir_len
        || (version == FORMAT_VERSION && dir_end != payload_off)
        || (version == PLANNED_FORMAT_VERSION && dir_end > payload_off)
        || total_len != source_len
        || payload_off > total_len
    {
        return Err(ContainerError::InvalidHeader);
    }

    let content_hash = digest_range(source, 0, total_len)?;
    let generation = u64::from_le_bytes(content_hash[..8].try_into().unwrap());

    let directory_len = usize::try_from(dir_len).map_err(|_| ContainerError::IntegerOverflow)?;
    let mut directory = vec![0_u8; directory_len];
    read_from_source(source, dir_off, &mut directory)?;
    if blake3::hash(&directory).as_bytes() != &header[52..84] {
        return Err(ContainerError::HashMismatch);
    }
    validate_zero_range_as(source, dir_end, payload_off, ContainerError::InvalidHeader)?;

    let mut entries = BTreeMap::new();
    let mut previous_guid = None;
    for record in directory.chunks_exact(RECORD_SIZE) {
        let guid = StableGuid::from_u128(u128::from_le_bytes(record[0..16].try_into().unwrap()));
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
        validate_hash_range(source, offset, end, hash)?;
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

    // Requiring monotonic, non-overlapping ranges prevents aliasing/overlap
    // tricks. Gaps are part of the canonical byte layout and must be zero.
    let mut ordered: Vec<_> = entries.values().collect();
    if version == FORMAT_VERSION {
        ordered.sort_by_key(|entry| entry.guid);
    } else {
        ordered.sort_by_key(|entry| (entry.offset, entry.stored_len, entry.guid));
    }
    let mut cursor = payload_off;
    for entry in ordered {
        if entry.offset < cursor {
            return Err(ContainerError::OutOfBounds);
        }
        validate_zero_range(source, cursor, entry.offset)?;
        cursor = entry
            .offset
            .checked_add(entry.stored_len)
            .ok_or(ContainerError::IntegerOverflow)?;
    }
    if cursor != total_len {
        return Err(ContainerError::InvalidDirectory);
    }
    Ok(ParsedContainer {
        info: ContainerInfo {
            version,
            entry_count: count,
            total_len,
            content_hash,
            generation,
        },
        entries,
    })
}

fn validate_zero_range(
    source: &dyn crate::RangeSource,
    start: u64,
    end: u64,
) -> Result<(), ContainerError> {
    validate_zero_range_as(source, start, end, ContainerError::InvalidDirectory)
}

fn validate_zero_range_as(
    source: &dyn crate::RangeSource,
    start: u64,
    end: u64,
    error: ContainerError,
) -> Result<(), ContainerError> {
    validate_range(source, start, end, |chunk| {
        if chunk.iter().any(|byte| *byte != 0) {
            Err(error.clone())
        } else {
            Ok(())
        }
    })
}

fn validate_hash_range(
    source: &dyn crate::RangeSource,
    start: u64,
    end: u64,
    expected: [u8; 32],
) -> Result<(), ContainerError> {
    let mut hasher = blake3::Hasher::new();
    validate_range(source, start, end, |chunk| {
        hasher.update(chunk);
        Ok(())
    })?;
    if hasher.finalize().as_bytes() != &expected {
        return Err(ContainerError::HashMismatch);
    }
    Ok(())
}

fn digest_range(
    source: &dyn crate::RangeSource,
    start: u64,
    end: u64,
) -> Result<[u8; 32], ContainerError> {
    let mut hasher = blake3::Hasher::new();
    validate_range(source, start, end, |chunk| {
        hasher.update(chunk);
        Ok(())
    })?;
    Ok(*hasher.finalize().as_bytes())
}

fn validate_range<F>(
    source: &dyn crate::RangeSource,
    start: u64,
    end: u64,
    mut validate: F,
) -> Result<(), ContainerError>
where
    F: FnMut(&[u8]) -> Result<(), ContainerError>,
{
    if end < start {
        return Err(ContainerError::IntegerOverflow);
    }
    if end == start {
        return Ok(());
    }
    let mut offset = start;
    let mut buffer = vec![0_u8; VALIDATION_CHUNK_SIZE];
    while offset < end {
        let remaining = end - offset;
        let amount = usize::try_from(remaining)
            .unwrap_or(VALIDATION_CHUNK_SIZE)
            .min(VALIDATION_CHUNK_SIZE);
        read_from_source(source, offset, &mut buffer[..amount])?;
        validate(&buffer[..amount])?;
        offset = offset
            .checked_add(u64::try_from(amount).map_err(|_| ContainerError::IntegerOverflow)?)
            .ok_or(ContainerError::IntegerOverflow)?;
    }
    Ok(())
}
