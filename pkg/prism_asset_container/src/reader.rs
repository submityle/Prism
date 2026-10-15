use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use flate2::{Decompress, FlushDecompress, Status};

use prism_asset::StableGuid;

use crate::format::{
    read_u16, read_u32, read_u64, ChunkInfo, Codec, ContainerError, ContainerInfo, EntryInfo,
    CHUNK_FORMAT_VERSION, CHUNK_HEADER_SIZE, CHUNK_RECORD_SIZE, CHUNK_TABLE_RECORD_SIZE,
    FORMAT_VERSION, HEADER_SIZE, MAGIC, RECORD_SIZE,
};
use crate::{RangeBatchPolicy, RangeRequest};

const VALIDATION_CHUNK_SIZE: usize = 64 * 1024;

#[derive(Clone, Debug)]
enum Backing {
    Memory(Arc<[u8]>),
    Range(Arc<dyn crate::RangeSource>),
}

/// Strict container reader. Mount validates the complete directory and every
/// payload digest before the container can be mounted. A range-backed reader
/// retains only the source and validated metadata; it never caches the whole
/// file. Range sources must keep their byte contents immutable for the reader's
/// lifetime; `read_all` additionally verifies the bytes it returns.
#[derive(Clone, Debug)]
pub struct ContainerReader {
    backing: Backing,
    info: ContainerInfo,
    entries: BTreeMap<StableGuid, EntryInfo>,
    chunks: BTreeMap<u32, ChunkInfo>,
}

#[derive(Debug)]
struct ParsedContainer {
    info: ContainerInfo,
    entries: BTreeMap<StableGuid, EntryInfo>,
    chunks: BTreeMap<u32, ChunkInfo>,
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
            chunks: parsed.chunks,
        })
    }

    /// Mounts and strictly validates a container through a seekable/range
    /// source. The source length must exactly match the header's `total_len`.
    /// Header, ToC, padding, and every raw payload are verified at mount time
    /// in bounded chunks; the source is then used for all payload reads. The
    /// caller must ensure source contents remain immutable for this reader's
    /// lifetime; range reads cannot re-verify a whole-payload digest without
    /// reading the entire payload.
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
            chunks: parsed.chunks,
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

    /// Returns validated metadata for one independently stored chunk.
    #[must_use]
    pub fn chunk(&self, id: u32) -> Option<ChunkInfo> {
        self.chunks.get(&id).copied()
    }

    /// GUIDs in canonical on-disk order.
    pub fn guids(&self) -> impl Iterator<Item = StableGuid> + '_ {
        self.entries.keys().copied()
    }

    /// Reads a byte range from a decoded payload. Raw payloads use direct
    /// physical range I/O; Deflate payloads are decoded and then sliced.
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
        if self.info.version == CHUNK_FORMAT_VERSION {
            let chunk = self
                .chunks
                .get(&entry.chunk_id)
                .copied()
                .ok_or(ContainerError::InvalidDirectory)?;
            if chunk.codec == Codec::Deflate {
                let decoded = self.decode_chunk(chunk)?;
                let start = usize::try_from(entry.raw_offset)
                    .and_then(|base| usize::try_from(offset).map(|local| base + local))
                    .map_err(|_| ContainerError::IntegerOverflow)?;
                let finish = start
                    .checked_add(out.len())
                    .ok_or(ContainerError::IntegerOverflow)?;
                out.copy_from_slice(
                    decoded
                        .get(start..finish)
                        .ok_or(ContainerError::OutOfBounds)?,
                );
                return Ok(());
            }
            let physical_offset = chunk
                .offset
                .checked_add(entry.raw_offset)
                .and_then(|value| value.checked_add(offset))
                .ok_or(ContainerError::IntegerOverflow)?;
            return self.read_physical_range(physical_offset, out);
        }
        if entry.codec == Codec::Deflate {
            let decoded = self.decode_entry(*entry)?;
            let start = usize::try_from(offset).map_err(|_| ContainerError::IntegerOverflow)?;
            let finish = start
                .checked_add(out.len())
                .ok_or(ContainerError::IntegerOverflow)?;
            out.copy_from_slice(
                decoded
                    .get(start..finish)
                    .ok_or(ContainerError::OutOfBounds)?,
            );
            return Ok(());
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

    /// Reads multiple payload ranges using bounded physical range coalescing.
    ///
    /// All requests are validated before the first source read. The source is
    /// read in deterministic physical order, while the returned vectors retain
    /// the caller's request order. Overlapping requests share one physical read;
    /// nearby requests may include a bounded gap to reduce I/O operations.
    pub fn read_ranges(
        &self,
        requests: &[RangeRequest],
        policy: RangeBatchPolicy,
    ) -> Result<Vec<Vec<u8>>, ContainerError> {
        if policy.max_batch_bytes == 0 {
            return Err(ContainerError::InvalidBatchPolicy);
        }

        if requests.iter().any(|request| {
            self.entries
                .get(&request.guid)
                .is_some_and(|entry| entry.codec == Codec::Deflate)
        }) {
            let mut outputs = Vec::with_capacity(requests.len());
            for request in requests.iter().copied() {
                let len = self.validate_range_request(request)?;
                policy.validate_request_len(request.len)?;
                let mut output = vec![0; len];
                self.read_range(request.guid, request.offset, &mut output)?;
                outputs.push(output);
            }
            return Ok(outputs);
        }

        let mut outputs = Vec::with_capacity(requests.len());
        let mut physical = Vec::new();
        for (index, request) in requests.iter().copied().enumerate() {
            policy.validate_request_len(request.len)?;
            let len = self.validate_range_request(request)?;
            outputs.push(vec![0; len]);
            if len != 0 {
                let entry = self.entries[&request.guid];
                let start = entry
                    .offset
                    .checked_add(request.offset)
                    .ok_or(ContainerError::IntegerOverflow)?;
                let end = start
                    .checked_add(request.len)
                    .ok_or(ContainerError::IntegerOverflow)?;
                physical.push(PhysicalRange { index, start, end });
            }
        }

        physical.sort_by_key(|range| (range.start, range.end, range.index));
        let mut batches = Vec::new();
        for range in physical {
            let Some(batch) = batches.last_mut() else {
                batches.push(PhysicalBatch::new(range));
                continue;
            };
            if batch.can_merge(range, policy) {
                batch.end = batch.end.max(range.end);
                batch.ranges.push(range);
            } else {
                batches.push(PhysicalBatch::new(range));
            }
        }

        for batch in batches {
            let batch_len = usize::try_from(batch.end - batch.start)
                .map_err(|_| ContainerError::IntegerOverflow)?;
            let mut bytes = vec![0; batch_len];
            self.read_physical_range(batch.start, &mut bytes)?;
            for range in batch.ranges {
                let start = usize::try_from(range.start - batch.start)
                    .map_err(|_| ContainerError::IntegerOverflow)?;
                let len = usize::try_from(range.end - range.start)
                    .map_err(|_| ContainerError::IntegerOverflow)?;
                let end = start
                    .checked_add(len)
                    .ok_or(ContainerError::IntegerOverflow)?;
                outputs[range.index]
                    .copy_from_slice(bytes.get(start..end).ok_or(ContainerError::OutOfBounds)?);
            }
        }
        Ok(outputs)
    }

    pub(crate) fn validate_range_request(
        &self,
        request: RangeRequest,
    ) -> Result<usize, ContainerError> {
        let entry = self
            .entries
            .get(&request.guid)
            .ok_or(ContainerError::MissingAsset(request.guid))?;
        let end = request
            .offset
            .checked_add(request.len)
            .ok_or(ContainerError::IntegerOverflow)?;
        if end > entry.raw_len {
            return Err(ContainerError::InvalidRange);
        }
        usize::try_from(request.len).map_err(|_| ContainerError::IntegerOverflow)
    }

    fn read_physical_range(&self, offset: u64, out: &mut [u8]) -> Result<(), ContainerError> {
        match &self.backing {
            Backing::Memory(bytes) => {
                let start = usize::try_from(offset).map_err(|_| ContainerError::IntegerOverflow)?;
                let end = start
                    .checked_add(out.len())
                    .ok_or(ContainerError::IntegerOverflow)?;
                let payload = bytes.get(start..end).ok_or(ContainerError::OutOfBounds)?;
                out.copy_from_slice(payload);
                Ok(())
            }
            Backing::Range(source) => read_from_source(source.as_ref(), offset, out),
        }
    }

    fn decode_chunk(&self, chunk: ChunkInfo) -> Result<Vec<u8>, ContainerError> {
        let stored_len =
            usize::try_from(chunk.stored_len).map_err(|_| ContainerError::IntegerOverflow)?;
        let mut stored = vec![0; stored_len];
        self.read_physical_range(chunk.offset, &mut stored)?;
        match chunk.codec {
            Codec::Raw => Ok(stored),
            Codec::Deflate => decode_deflate(&stored, chunk.raw_len),
        }
    }

    fn decode_entry(&self, entry: EntryInfo) -> Result<Vec<u8>, ContainerError> {
        let stored_len =
            usize::try_from(entry.stored_len).map_err(|_| ContainerError::IntegerOverflow)?;
        let mut stored = vec![0; stored_len];
        self.read_physical_range(entry.offset, &mut stored)?;
        decode_deflate(&stored, entry.raw_len)
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
        if blake3::hash(&out).as_bytes() != &entry.hash {
            return Err(ContainerError::HashMismatch);
        }
        Ok(out)
    }
}

#[derive(Clone, Copy, Debug)]
struct PhysicalRange {
    index: usize,
    start: u64,
    end: u64,
}

#[derive(Debug)]
struct PhysicalBatch {
    start: u64,
    end: u64,
    ranges: Vec<PhysicalRange>,
}

impl PhysicalBatch {
    fn new(range: PhysicalRange) -> Self {
        Self {
            start: range.start,
            end: range.end,
            ranges: vec![range],
        }
    }

    fn can_merge(&self, range: PhysicalRange, policy: RangeBatchPolicy) -> bool {
        let gap = range.start.saturating_sub(self.end);
        let end = self.end.max(range.end);
        gap <= policy.max_gap_bytes
            && end
                .checked_sub(self.start)
                .is_some_and(|size| size <= policy.max_batch_bytes)
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
    let mut prefix = [0_u8; 12];
    read_from_source(source, 0, &mut prefix)?;
    if prefix[0..8] != MAGIC {
        return Err(ContainerError::BadMagic);
    }
    return match read_u16(&prefix, 8)? {
        FORMAT_VERSION => parse_v1(source),
        CHUNK_FORMAT_VERSION => parse_v2(source),
        version => Err(ContainerError::UnsupportedVersion(version)),
    };
}

fn parse_v1(source: &dyn crate::RangeSource) -> Result<ParsedContainer, ContainerError> {
    let mut header = [0_u8; HEADER_SIZE];
    read_from_source(source, 0, &mut header)?;
    if header[0..8] != MAGIC {
        return Err(ContainerError::BadMagic);
    }
    let version = read_u16(&header, 8)?;
    if version != FORMAT_VERSION {
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
        || dir_end > payload_off
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
        if codec == Codec::Raw {
            validate_hash_range(source, offset, end, hash)?;
        } else {
            let stored_len =
                usize::try_from(stored_len).map_err(|_| ContainerError::IntegerOverflow)?;
            let mut stored = vec![0; stored_len];
            read_from_source(source, offset, &mut stored)?;
            validate_deflate(&stored, raw_len, hash)?;
        }
        entries.insert(
            guid,
            EntryInfo {
                guid,
                chunk_id: 0,
                raw_offset: 0,
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
    ordered.sort_by_key(|entry| (entry.offset, entry.stored_len, entry.guid));
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
            chunk_count: 0,
            total_len,
            content_hash,
            generation,
        },
        entries,
        chunks: BTreeMap::new(),
    })
}

fn parse_v2(source: &dyn crate::RangeSource) -> Result<ParsedContainer, ContainerError> {
    let mut header = [0_u8; CHUNK_HEADER_SIZE];
    read_from_source(source, 0, &mut header)?;
    if header[0..8] != MAGIC
        || read_u16(&header, 8)? != CHUNK_FORMAT_VERSION
        || usize::from(read_u16(&header, 10)?) != CHUNK_HEADER_SIZE
        || read_u32(&header, 12)? != 0
        || header[136..CHUNK_HEADER_SIZE].iter().any(|byte| *byte != 0)
    {
        return Err(ContainerError::InvalidHeader);
    }
    let count = read_u32(&header, 16)?;
    let chunk_count = read_u32(&header, 20)?;
    let dir_off = read_u64(&header, 24)?;
    let dir_len = read_u64(&header, 32)?;
    let table_off = read_u64(&header, 40)?;
    let table_len = read_u64(&header, 48)?;
    let payload_off = read_u64(&header, 56)?;
    let total_len = read_u64(&header, 64)?;
    let expected_dir_len = u64::from(count)
        .checked_mul(CHUNK_RECORD_SIZE as u64)
        .ok_or(ContainerError::IntegerOverflow)?;
    let expected_table_len = u64::from(chunk_count)
        .checked_mul(CHUNK_TABLE_RECORD_SIZE as u64)
        .ok_or(ContainerError::IntegerOverflow)?;
    let dir_end = dir_off
        .checked_add(dir_len)
        .ok_or(ContainerError::IntegerOverflow)?;
    let table_end = table_off
        .checked_add(table_len)
        .ok_or(ContainerError::IntegerOverflow)?;
    if total_len > source.len() {
        return Err(ContainerError::Truncated);
    }
    if dir_off != CHUNK_HEADER_SIZE as u64
        || dir_len != expected_dir_len
        || table_off < dir_end
        || table_off % 8 != 0
        || table_len != expected_table_len
        || payload_off < table_end
        || total_len != source.len()
        || payload_off > total_len
    {
        return Err(ContainerError::InvalidHeader);
    }
    let content_hash = digest_range(source, 0, total_len)?;
    let generation = u64::from_le_bytes(content_hash[..8].try_into().unwrap());
    let mut directory =
        vec![0; usize::try_from(dir_len).map_err(|_| ContainerError::IntegerOverflow)?];
    read_from_source(source, dir_off, &mut directory)?;
    if blake3::hash(&directory).as_bytes() != &header[72..104] {
        return Err(ContainerError::HashMismatch);
    }
    let mut table =
        vec![0; usize::try_from(table_len).map_err(|_| ContainerError::IntegerOverflow)?];
    read_from_source(source, table_off, &mut table)?;
    if blake3::hash(&table).as_bytes() != &header[104..136] {
        return Err(ContainerError::HashMismatch);
    }
    validate_zero_range_as(source, dir_end, table_off, ContainerError::InvalidHeader)?;
    validate_zero_range_as(
        source,
        table_end,
        payload_off,
        ContainerError::InvalidHeader,
    )?;

    let mut chunks = BTreeMap::new();
    for (index, record) in table.chunks_exact(CHUNK_TABLE_RECORD_SIZE).enumerate() {
        let id = u32::try_from(index).map_err(|_| ContainerError::IntegerOverflow)?;
        let offset = read_u64(record, 0)?;
        let stored_len = read_u64(record, 8)?;
        let raw_len = read_u64(record, 16)?;
        let codec = Codec::decode(record[24])?;
        if record[25..28].iter().any(|byte| *byte != 0)
            || record[60..64].iter().any(|byte| *byte != 0)
        {
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
        let hash = record[28..60].try_into().unwrap();
        if codec == Codec::Raw {
            validate_hash_range(source, offset, end, hash)?;
        } else {
            let mut stored =
                vec![0; usize::try_from(stored_len).map_err(|_| ContainerError::IntegerOverflow)?];
            read_from_source(source, offset, &mut stored)?;
            validate_deflate(&stored, raw_len, hash)?;
        }
        if chunks
            .insert(
                id,
                ChunkInfo {
                    id,
                    offset,
                    stored_len,
                    raw_len,
                    codec,
                    hash,
                },
            )
            .is_some()
        {
            return Err(ContainerError::InvalidDirectory);
        }
    }
    let mut ordered: Vec<_> = chunks.values().copied().collect();
    ordered.sort_by_key(|chunk| (chunk.offset, chunk.id));
    let mut cursor = payload_off;
    let mut raw_cursor = payload_off;
    for chunk in ordered {
        if chunk.offset < cursor || chunk.offset < raw_cursor {
            return Err(ContainerError::OutOfBounds);
        }
        let raw_end = chunk
            .offset
            .checked_add(chunk.raw_len)
            .ok_or(ContainerError::IntegerOverflow)?;
        validate_zero_range(source, cursor, chunk.offset)?;
        cursor = chunk
            .offset
            .checked_add(chunk.stored_len)
            .ok_or(ContainerError::IntegerOverflow)?;
        raw_cursor = raw_end;
    }
    if let Some(last) = chunks.values().max_by_key(|chunk| chunk.offset) {
        let expected_end = last
            .offset
            .checked_add(last.raw_len)
            .ok_or(ContainerError::IntegerOverflow)?;
        if expected_end != total_len {
            return Err(ContainerError::InvalidDirectory);
        }
    } else if payload_off != total_len {
        return Err(ContainerError::InvalidDirectory);
    }
    validate_zero_range(source, cursor, total_len)?;

    let mut entries = BTreeMap::new();
    let mut previous_guid = None;
    let mut decoded_chunks = BTreeMap::<u32, Vec<u8>>::new();
    for record in directory.chunks_exact(CHUNK_RECORD_SIZE) {
        let guid = StableGuid::from_u128(u128::from_le_bytes(record[0..16].try_into().unwrap()));
        if entries.contains_key(&guid) || previous_guid.is_some_and(|previous| previous >= guid) {
            eprintln!("v2 guid ordering");
            return Err(ContainerError::InvalidDirectory);
        }
        previous_guid = Some(guid);
        let chunk_id = read_u32(record, 16)?;
        let raw_offset = read_u64(record, 24)?;
        let raw_len = read_u64(record, 32)?;
        let chunk = chunks
            .get(&chunk_id)
            .copied()
            .ok_or(ContainerError::InvalidDirectory)?;
        let raw_end = raw_offset
            .checked_add(raw_len)
            .ok_or(ContainerError::IntegerOverflow)?;
        if record[20..24].iter().any(|byte| *byte != 0)
            || record[76..80].iter().any(|byte| *byte != 0)
            || raw_end > chunk.raw_len
        {
            eprintln!("v2 entry bounds guid={guid:?} chunk={chunk_id} raw={raw_offset}+{raw_len} chunkraw={}", chunk.raw_len);
            return Err(ContainerError::InvalidDirectory);
        }
        let hash = record[40..72].try_into().unwrap();
        let alignment = u32::from_le_bytes(record[72..76].try_into().unwrap());
        let absolute = chunk
            .offset
            .checked_add(raw_offset)
            .ok_or(ContainerError::IntegerOverflow)?;
        if alignment == 0 || !alignment.is_power_of_two() || absolute % u64::from(alignment) != 0 {
            eprintln!("v2 alignment guid={guid:?} absolute={absolute} alignment={alignment}");
            return Err(ContainerError::InvalidDirectory);
        }
        if chunk.codec == Codec::Raw {
            validate_hash_range(
                source,
                absolute,
                absolute
                    .checked_add(raw_len)
                    .ok_or(ContainerError::IntegerOverflow)?,
                hash,
            )?;
        } else {
            let decoded = decoded_chunks.entry(chunk_id).or_default();
            if decoded.is_empty() {
                *decoded = decode_chunk_source(source, chunk)?;
            }
            let start = usize::try_from(raw_offset).map_err(|_| ContainerError::IntegerOverflow)?;
            let end = start
                .checked_add(usize::try_from(raw_len).map_err(|_| ContainerError::IntegerOverflow)?)
                .ok_or(ContainerError::IntegerOverflow)?;
            if blake3::hash(decoded.get(start..end).ok_or(ContainerError::OutOfBounds)?).as_bytes()
                != &hash
            {
                return Err(ContainerError::HashMismatch);
            }
        }
        entries.insert(
            guid,
            EntryInfo {
                guid,
                chunk_id,
                raw_offset,
                offset: chunk.offset,
                stored_len: chunk.stored_len,
                raw_len,
                codec: chunk.codec,
                hash,
                alignment,
            },
        );
    }
    Ok(ParsedContainer {
        info: ContainerInfo {
            version: CHUNK_FORMAT_VERSION,
            entry_count: count,
            chunk_count,
            total_len,
            content_hash,
            generation,
        },
        entries,
        chunks,
    })
}

fn decode_chunk_source(
    source: &dyn crate::RangeSource,
    chunk: ChunkInfo,
) -> Result<Vec<u8>, ContainerError> {
    let mut stored =
        vec![0; usize::try_from(chunk.stored_len).map_err(|_| ContainerError::IntegerOverflow)?];
    read_from_source(source, chunk.offset, &mut stored)?;
    match chunk.codec {
        Codec::Raw => Ok(stored),
        Codec::Deflate => decode_deflate(&stored, chunk.raw_len),
    }
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

fn decode_deflate(stored: &[u8], raw_len: u64) -> Result<Vec<u8>, ContainerError> {
    let raw_len = usize::try_from(raw_len).map_err(|_| ContainerError::IntegerOverflow)?;
    let capacity = raw_len
        .checked_add(1)
        .ok_or(ContainerError::IntegerOverflow)?;
    let mut output = vec![0_u8; capacity];
    let mut decompressor = Decompress::new(false);
    let status = loop {
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
            break status;
        }
        if status == Status::BufError
            || (decompressor.total_in() == before_in && decompressor.total_out() == before_out)
        {
            return Err(ContainerError::CompressionFailed);
        }
    };
    if status != Status::StreamEnd
        || decompressor.total_in() != u64::try_from(stored.len()).unwrap_or(u64::MAX)
        || usize::try_from(decompressor.total_out()).unwrap_or(usize::MAX) != raw_len
    {
        return Err(ContainerError::CompressionFailed);
    }
    output.truncate(raw_len);
    Ok(output)
}

fn validate_deflate(
    stored: &[u8],
    raw_len: u64,
    expected_hash: [u8; 32],
) -> Result<(), ContainerError> {
    let mut decompressor = Decompress::new(false);
    let mut output = vec![0_u8; VALIDATION_CHUNK_SIZE];
    let mut hasher = blake3::Hasher::new();
    let status = loop {
        let consumed = usize::try_from(decompressor.total_in())
            .map_err(|_| ContainerError::CompressionFailed)?;
        let input = stored
            .get(consumed..)
            .ok_or(ContainerError::CompressionFailed)?;
        let before_in = decompressor.total_in();
        let before_out = decompressor.total_out();
        let status = decompressor
            .decompress(input, &mut output, FlushDecompress::Finish)
            .map_err(|_| ContainerError::CompressionFailed)?;
        let produced = decompressor
            .total_out()
            .checked_sub(before_out)
            .ok_or(ContainerError::CompressionFailed)?;
        let produced = usize::try_from(produced).map_err(|_| ContainerError::CompressionFailed)?;
        hasher.update(&output[..produced]);
        if decompressor.total_out() > raw_len {
            return Err(ContainerError::CompressionFailed);
        }
        if status == Status::StreamEnd {
            break status;
        }
        if (status == Status::BufError && produced == 0)
            || (decompressor.total_in() == before_in && decompressor.total_out() == before_out)
        {
            return Err(ContainerError::CompressionFailed);
        }
    };
    if status != Status::StreamEnd
        || decompressor.total_in() != u64::try_from(stored.len()).unwrap_or(u64::MAX)
        || decompressor.total_out() != raw_len
    {
        return Err(ContainerError::CompressionFailed);
    }
    if hasher.finalize().as_bytes() != &expected_hash {
        return Err(ContainerError::HashMismatch);
    }
    Ok(())
}
