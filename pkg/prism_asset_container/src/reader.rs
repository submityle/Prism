//! Validated mounting and bounded raw/compressed payload reads.

use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use flate2::{Decompress, FlushDecompress, Status};

use prism_asset::StableGuid;

use crate::budget::{zeroed, ReadBuffer};
use crate::format::{
    read_u16, read_u32, read_u64, ChunkInfo, Codec, ContainerError, ContainerInfo, EntryInfo,
    CHUNK_FORMAT_VERSION, CHUNK_HEADER_SIZE, CHUNK_RECORD_SIZE, CHUNK_TABLE_RECORD_SIZE,
    FORMAT_VERSION, HEADER_SIZE, MAGIC, RECORD_SIZE,
};
use crate::{ContainerReadBudget, RangeBatchPolicy, RangeRequest};

const VALIDATION_CHUNK_SIZE: usize = 64 * 1024;

#[cfg(feature = "mmap")]
#[path = "mmap_view.rs"]
mod mmap_view;

#[cfg(feature = "mmap")]
pub use mmap_view::MmapV2DirectoryView;

#[derive(Clone, Debug)]
enum Backing {
    Memory(Arc<[u8]>),
    Range(Arc<dyn crate::RangeSource>),
}

/// Normalizes a source's per-call cap once, before parsing or retaining it.
/// This also covers block-index/decoder reads that bypass `read_from_source`.
#[derive(Debug)]
struct BoundedSource {
    source: Arc<dyn crate::RangeSource>,
    maximum_read_bytes: usize,
}

impl crate::RangeSource for BoundedSource {
    fn len(&self) -> u64 {
        self.source.len()
    }

    fn read_range(&self, mut offset: u64, out: &mut [u8]) -> Result<(), crate::RangeSourceError> {
        let len = u64::try_from(out.len()).map_err(|_| crate::RangeSourceError::OutOfBounds)?;
        let end = offset
            .checked_add(len)
            .ok_or(crate::RangeSourceError::OutOfBounds)?;
        if end > self.source.len() {
            return Err(crate::RangeSourceError::OutOfBounds);
        }
        for chunk in out.chunks_mut(self.maximum_read_bytes) {
            self.source.read_range(offset, chunk)?;
            offset += chunk.len() as u64;
        }
        Ok(())
    }
}

/// Strict container reader. Mount validates the complete directory and every
/// payload digest before the container can be mounted. A range-backed reader
/// retains only the source and validated metadata; it never caches the whole
/// file. Range sources must keep their byte contents immutable for the reader's
/// lifetime; `read_all` additionally verifies the bytes it returns.
#[derive(Clone)]
pub struct ContainerReader {
    backing: Backing,
    info: ContainerInfo,
    entries: BTreeMap<StableGuid, EntryInfo>,
    chunks: BTreeMap<u32, ChunkInfo>,
    budget: ContainerReadBudget,
    block_decode_backend: Arc<dyn crate::block::BlockDecodeBackend>,
}

impl core::fmt::Debug for ContainerReader {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ContainerReader")
            .field("backing", &self.backing)
            .field("info", &self.info)
            .field("entries", &self.entries)
            .field("chunks", &self.chunks)
            .field("budget", &self.budget)
            .finish_non_exhaustive()
    }
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
        Self::with_budget(bytes, ContainerReadBudget::default())
    }

    /// Validates memory-backed bytes using a caller-owned shared budget.
    pub fn with_budget(
        bytes: impl Into<Arc<[u8]>>,
        budget: ContainerReadBudget,
    ) -> Result<Self, ContainerError> {
        Self::with_budget_and_block_decode_backend(
            bytes,
            budget,
            Arc::new(crate::block::CpuBlockDecodeBackend),
        )
    }

    /// Validates memory-backed bytes with a caller-provided `BlockDeflate`
    /// backend and shared budget.
    pub fn with_budget_and_block_decode_backend(
        bytes: impl Into<Arc<[u8]>>,
        budget: ContainerReadBudget,
        block_decode_backend: Arc<dyn crate::block::BlockDecodeBackend>,
    ) -> Result<Self, ContainerError> {
        let bytes = bytes.into();
        let source = crate::MemoryRangeSource::new(bytes.clone());
        let parsed = parse_source(&source, &budget, block_decode_backend.as_ref())?;
        Ok(Self {
            backing: Backing::Memory(bytes),
            info: parsed.info,
            entries: parsed.entries,
            chunks: parsed.chunks,
            budget,
            block_decode_backend,
        })
    }

    /// Validates memory-backed bytes with a caller-provided `BlockDeflate`
    /// backend and the default budget.
    pub fn new_with_block_decode_backend(
        bytes: impl Into<Arc<[u8]>>,
        block_decode_backend: Arc<dyn crate::block::BlockDecodeBackend>,
    ) -> Result<Self, ContainerError> {
        Self::with_budget_and_block_decode_backend(
            bytes,
            ContainerReadBudget::default(),
            block_decode_backend,
        )
    }

    /// Mounts and strictly validates a container through a seekable/range
    /// source. The source length must exactly match the header's `total_len`.
    /// Header, `ToC`, padding, and every raw payload are verified at mount time
    /// in bounded chunks; the source is then used for all payload reads. The
    /// caller must ensure source contents remain immutable for this reader's
    /// lifetime; range reads cannot re-verify a whole-payload digest without
    /// reading the entire payload.
    /// Reads larger than `RangeSource::maximum_read_bytes` are split into
    /// consecutive calls, including metadata, hashes and compressed blocks.
    /// Mount still verifies the entire container; a fetch cap does not make
    /// mounting lazy or authenticate an isolated runtime subrange.
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
        Self::from_range_source_with_budget_and_block_decode_backend(
            source,
            ContainerReadBudget::default(),
            Arc::new(crate::block::CpuBlockDecodeBackend),
        )
    }

    /// Mounts a range source under a budget that may also govern other mounts.
    pub fn from_range_source_with_budget(
        source: Arc<dyn crate::RangeSource>,
        budget: ContainerReadBudget,
    ) -> Result<Self, ContainerError> {
        Self::from_range_source_with_budget_and_block_decode_backend(
            source,
            budget,
            Arc::new(crate::block::CpuBlockDecodeBackend),
        )
    }

    /// Mounts a range source under a budget with a caller-provided `BlockDeflate`
    /// backend. The backend is used during mount validation and later reads.
    pub fn from_range_source_with_budget_and_block_decode_backend(
        source: Arc<dyn crate::RangeSource>,
        budget: ContainerReadBudget,
        block_decode_backend: Arc<dyn crate::block::BlockDecodeBackend>,
    ) -> Result<Self, ContainerError> {
        let maximum_read_bytes = source.maximum_read_bytes().get();
        let source: Arc<dyn crate::RangeSource> = if maximum_read_bytes < usize::MAX {
            Arc::new(BoundedSource {
                source,
                maximum_read_bytes,
            })
        } else {
            source
        };
        let parsed = parse_source(source.as_ref(), &budget, block_decode_backend.as_ref())?;
        Ok(Self {
            backing: Backing::Range(source),
            info: parsed.info,
            entries: parsed.entries,
            chunks: parsed.chunks,
            budget,
            block_decode_backend,
        })
    }

    /// Mounts a range source with a caller-provided `BlockDeflate` backend and
    /// the default budget.
    pub fn from_shared_range_source_with_block_decode_backend(
        source: Arc<dyn crate::RangeSource>,
        block_decode_backend: Arc<dyn crate::block::BlockDecodeBackend>,
    ) -> Result<Self, ContainerError> {
        Self::from_range_source_with_budget_and_block_decode_backend(
            source,
            ContainerReadBudget::default(),
            block_decode_backend,
        )
    }

    /// Returns the shared in-flight budget used by this reader and its clones.
    #[must_use]
    pub fn budget(&self) -> &ContainerReadBudget {
        &self.budget
    }

    /// Mounts a read-only memory-mapped local container file.
    #[cfg(feature = "mmap")]
    pub fn from_mmap_file(path: impl AsRef<std::path::Path>) -> Result<Self, ContainerError> {
        let source = crate::MmapRangeSource::open(path)
            .map_err(|error| ContainerError::Source(error.to_string()))?;
        Self::from_range_source(Arc::new(source))
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
    /// On error, `out` may have been partially written and must be discarded.
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
        let limits = self.budget.limits();
        limits.check("output bytes", len, limits.max_output_bytes)?;
        if out.is_empty() {
            return Ok(());
        }
        if self.info.version == CHUNK_FORMAT_VERSION {
            let chunk = self
                .chunks
                .get(&entry.chunk_id)
                .copied()
                .ok_or(ContainerError::InvalidDirectory)?;
            if chunk.codec == Codec::BlockDeflate {
                let raw_offset = entry
                    .raw_offset
                    .checked_add(offset)
                    .ok_or(ContainerError::IntegerOverflow)?;
                return self.read_block_chunk_range(chunk, raw_offset, out);
            }
            if chunk.codec == Codec::Deflate {
                let decoded = self.decode_chunk(chunk)?;
                let start = usize::try_from(entry.raw_offset)
                    .map_err(|_| ContainerError::IntegerOverflow)?
                    .checked_add(
                        usize::try_from(offset).map_err(|_| ContainerError::IntegerOverflow)?,
                    )
                    .ok_or(ContainerError::IntegerOverflow)?;
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
    /// All requests are validated before the first source read, and the
    /// returned vectors retain the caller's request order. Raw-only batches
    /// coalesce overlapping/nearby ranges in physical order. Batches containing
    /// compressed entries decode each requested unit once; raw entries in a
    /// mixed batch are read separately, without coalescing.
    pub fn read_ranges(
        &self,
        requests: &[RangeRequest],
        policy: RangeBatchPolicy,
    ) -> Result<Vec<Vec<u8>>, ContainerError> {
        if policy.max_batch_bytes == 0 {
            return Err(ContainerError::InvalidBatchPolicy);
        }
        let limits = self.budget.limits();
        limits.check(
            "batch requests",
            requests.len() as u64,
            u64::from(limits.max_records),
        )?;

        // Validate the complete batch before reading, including the compressed
        // path. Output allocations remain reserved until every read finishes.
        let mut total = 0_u64;
        for request in requests.iter().copied() {
            self.validate_range_request(request)?;
            policy.validate_request_len(request.len)?;
            total = total
                .checked_add(request.len)
                .ok_or(ContainerError::IntegerOverflow)?;
        }
        limits.check("output bytes", total, limits.max_output_bytes)?;
        let _outputs_reservation = self.budget.reserve(total)?;

        if requests.iter().any(|request| {
            self.entries
                .get(&request.guid)
                .is_some_and(|entry| matches!(entry.codec, Codec::Deflate | Codec::BlockDeflate))
        }) {
            let mut outputs = Vec::with_capacity(requests.len());
            let mut compressed = Vec::new();
            for (index, request) in requests.iter().copied().enumerate() {
                outputs.push(zeroed(request.len)?);
                if request.len == 0 {
                    continue;
                }
                let entry = self.entries[&request.guid];
                if entry.codec == Codec::Deflate {
                    let key = if self.info.version == CHUNK_FORMAT_VERSION {
                        (entry.chunk_id, None)
                    } else {
                        (0, Some(entry.guid))
                    };
                    compressed.push((key, index));
                } else {
                    self.read_range(request.guid, request.offset, &mut outputs[index])?;
                }
            }
            compressed.sort_unstable();
            let mut position = 0;
            while let Some(&(key, first)) = compressed.get(position) {
                let entry = self.entries[&requests[first].guid];
                let decoded = if self.info.version == CHUNK_FORMAT_VERSION {
                    self.decode_chunk(self.chunks[&entry.chunk_id])?
                } else {
                    self.decode_entry(entry)?
                };
                while position < compressed.len() && compressed[position].0 == key {
                    let index = compressed[position].1;
                    let request = requests[index];
                    let entry = self.entries[&request.guid];
                    let start = usize::try_from(
                        entry
                            .raw_offset
                            .checked_add(request.offset)
                            .ok_or(ContainerError::IntegerOverflow)?,
                    )
                    .map_err(|_| ContainerError::IntegerOverflow)?;
                    let end = start
                        .checked_add(outputs[index].len())
                        .ok_or(ContainerError::IntegerOverflow)?;
                    outputs[index].copy_from_slice(
                        decoded.get(start..end).ok_or(ContainerError::OutOfBounds)?,
                    );
                    position += 1;
                }
            }
            return Ok(outputs);
        }

        let mut outputs = Vec::with_capacity(requests.len());
        let mut physical = Vec::new();
        for (index, request) in requests.iter().copied().enumerate() {
            policy.validate_request_len(request.len)?;
            let len = self.validate_range_request(request)?;
            outputs.push(zeroed(len as u64)?);
            if len != 0 {
                let entry = self.entries[&request.guid];
                let base = entry
                    .offset
                    .checked_add(if self.info.version == CHUNK_FORMAT_VERSION {
                        entry.raw_offset
                    } else {
                        0
                    })
                    .ok_or(ContainerError::IntegerOverflow)?;
                let start = base
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
            let mut bytes = self.budget.buffer(batch_len as u64)?;
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

    fn with_source<T>(
        &self,
        f: impl FnOnce(&dyn crate::RangeSource) -> Result<T, ContainerError>,
    ) -> Result<T, ContainerError> {
        match &self.backing {
            Backing::Memory(bytes) => {
                let source = crate::MemoryRangeSource::new(bytes.clone());
                f(&source)
            }
            Backing::Range(source) => f(source.as_ref()),
        }
    }

    fn decode_chunk(&self, chunk: ChunkInfo) -> Result<ReadBuffer, ContainerError> {
        self.budget.limits().unit(chunk.stored_len, chunk.raw_len)?;
        if chunk.codec == Codec::BlockDeflate {
            return self.with_source(|source| {
                let index = crate::block::parse_index(
                    source,
                    chunk.offset,
                    chunk.stored_len,
                    0,
                    &self.budget,
                )?;
                if index.raw_len() != chunk.raw_len {
                    return Err(ContainerError::InvalidDirectory);
                }
                let mut decoded = self.budget.buffer(chunk.raw_len)?;
                index.decode_range_with_backend(
                    source,
                    0,
                    &mut decoded,
                    &self.budget,
                    self.block_decode_backend.as_ref(),
                )?;
                Ok(decoded)
            });
        }
        let mut stored = self.budget.buffer(chunk.stored_len)?;
        self.read_physical_range(chunk.offset, &mut stored)?;
        match chunk.codec {
            Codec::Raw => Ok(stored),
            Codec::Deflate => decode_deflate(&stored, chunk.raw_len, &self.budget),
            Codec::BlockDeflate => Err(ContainerError::InvalidDirectory),
        }
    }

    fn read_block_chunk_range(
        &self,
        chunk: ChunkInfo,
        raw_offset: u64,
        out: &mut [u8],
    ) -> Result<(), ContainerError> {
        self.with_source(|source| {
            let index =
                crate::block::parse_index(source, chunk.offset, chunk.stored_len, 0, &self.budget)?;
            if index.raw_len() != chunk.raw_len {
                return Err(ContainerError::InvalidDirectory);
            }
            index.decode_range_with_backend(
                source,
                raw_offset,
                out,
                &self.budget,
                self.block_decode_backend.as_ref(),
            )
        })
    }

    fn decode_entry(&self, entry: EntryInfo) -> Result<ReadBuffer, ContainerError> {
        self.budget.limits().unit(entry.stored_len, entry.raw_len)?;
        let mut stored = self.budget.buffer(entry.stored_len)?;
        self.read_physical_range(entry.offset, &mut stored)?;
        decode_deflate(&stored, entry.raw_len, &self.budget)
    }

    /// Reads an owned payload range after validating and reserving its output.
    ///
    /// Bridges returning a `Vec` should use this entry point instead of
    /// allocating an unchecked output before calling [`Self::read_range`].
    pub fn read_range_owned(
        &self,
        guid: StableGuid,
        offset: u64,
        len: u64,
    ) -> Result<Vec<u8>, ContainerError> {
        self.validate_range_request(RangeRequest::new(guid, offset, len))?;
        let limits = self.budget.limits();
        limits.check("output bytes", len, limits.max_output_bytes)?;
        let mut out = self.budget.buffer(len)?;
        self.read_range(guid, offset, &mut out)?;
        Ok(out.into_vec())
    }

    /// Reads and verifies the complete raw payload.
    pub fn read_all(&self, guid: StableGuid) -> Result<Vec<u8>, ContainerError> {
        let entry = self
            .entries
            .get(&guid)
            .ok_or(ContainerError::MissingAsset(guid))?;
        let limits = self.budget.limits();
        limits.check("output bytes", entry.raw_len, limits.max_output_bytes)?;
        let mut out = self.budget.buffer(entry.raw_len)?;
        self.read_range(guid, 0, &mut out)?;
        if blake3::hash(&out).as_bytes() != &entry.hash {
            return Err(ContainerError::HashMismatch);
        }
        Ok(out.into_vec())
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

fn parse_source(
    source: &dyn crate::RangeSource,
    budget: &ContainerReadBudget,
    block_decode_backend: &dyn crate::block::BlockDecodeBackend,
) -> Result<ParsedContainer, ContainerError> {
    let limits = budget.limits();
    limits.check("container bytes", source.len(), limits.max_container_bytes)?;
    let mut prefix = [0_u8; 12];
    read_from_source(source, 0, &mut prefix)?;
    if prefix[0..8] != MAGIC {
        return Err(ContainerError::BadMagic);
    }
    match read_u16(&prefix, 8)? {
        FORMAT_VERSION => parse_v1(source, budget),
        CHUNK_FORMAT_VERSION => parse_v2(source, budget, block_decode_backend),
        version => Err(ContainerError::UnsupportedVersion(version)),
    }
}

fn parse_v1(
    source: &dyn crate::RangeSource,
    budget: &ContainerReadBudget,
) -> Result<ParsedContainer, ContainerError> {
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

    let limits = budget.limits();
    limits.metadata(dir_len, 0, count, 0)?;
    let content_hash = digest_range(source, 0, total_len, budget)?;
    let generation = u64::from_le_bytes(content_hash[..8].try_into().unwrap());

    let mut directory = budget.buffer(dir_len)?;
    read_from_source(source, dir_off, &mut directory)?;
    if blake3::hash(&directory).as_bytes() != &header[52..84] {
        return Err(ContainerError::HashMismatch);
    }
    validate_zero_range_as(
        source,
        dir_end,
        payload_off,
        ContainerError::InvalidHeader,
        budget,
    )?;

    let mut entries = BTreeMap::new();
    let mut previous_guid = None;
    let mut validation_bytes = 0_u64;
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
        if codec == Codec::BlockDeflate {
            return Err(ContainerError::CodecNotSupportedForLayout(codec));
        }
        if record[41..44].iter().any(|v| *v != 0) {
            return Err(ContainerError::InvalidDirectory);
        }
        if codec == Codec::Raw && stored_len != raw_len {
            return Err(ContainerError::InvalidDirectory);
        }
        limits.unit(stored_len, raw_len)?;
        validation_bytes = validation_bytes
            .checked_add(raw_len)
            .ok_or(ContainerError::IntegerOverflow)?;
        limits.check(
            "validation decoded bytes",
            validation_bytes,
            limits.max_validation_decoded_bytes,
        )?;
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
            validate_hash_range(source, offset, end, hash, budget)?;
        } else {
            let mut stored = budget.buffer(stored_len)?;
            read_from_source(source, offset, &mut stored)?;
            validate_deflate(&stored, raw_len, hash, budget)?;
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
        validate_zero_range(source, cursor, entry.offset, budget)?;
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

fn parse_v2(
    source: &dyn crate::RangeSource,
    budget: &ContainerReadBudget,
    block_decode_backend: &dyn crate::block::BlockDecodeBackend,
) -> Result<ParsedContainer, ContainerError> {
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
    let limits = budget.limits();
    limits.metadata(dir_len, table_len, count, chunk_count)?;
    let content_hash = digest_range(source, 0, total_len, budget)?;
    let generation = u64::from_le_bytes(content_hash[..8].try_into().unwrap());
    let mut directory = budget.buffer(dir_len)?;
    read_from_source(source, dir_off, &mut directory)?;
    if blake3::hash(&directory).as_bytes() != &header[72..104] {
        return Err(ContainerError::HashMismatch);
    }
    let mut table = budget.buffer(table_len)?;
    read_from_source(source, table_off, &mut table)?;
    if blake3::hash(&table).as_bytes() != &header[104..136] {
        return Err(ContainerError::HashMismatch);
    }
    validate_zero_range_as(
        source,
        dir_end,
        table_off,
        ContainerError::InvalidHeader,
        budget,
    )?;
    validate_zero_range_as(
        source,
        table_end,
        payload_off,
        ContainerError::InvalidHeader,
        budget,
    )?;

    let mut chunks = BTreeMap::new();
    let mut validation_bytes = 0_u64;
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
        limits.unit(stored_len, raw_len)?;
        validation_bytes = validation_bytes
            .checked_add(raw_len)
            .ok_or(ContainerError::IntegerOverflow)?;
        limits.check(
            "validation decoded bytes",
            validation_bytes,
            limits.max_validation_decoded_bytes,
        )?;
        let end = offset
            .checked_add(stored_len)
            .ok_or(ContainerError::IntegerOverflow)?;
        if offset < payload_off || end > total_len {
            return Err(ContainerError::OutOfBounds);
        }
        let hash = record[28..60].try_into().unwrap();
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
        validate_zero_range(source, cursor, chunk.offset, budget)?;
        cursor = chunk
            .offset
            .checked_add(chunk.stored_len)
            .ok_or(ContainerError::IntegerOverflow)?;
        raw_cursor = raw_end;
    }
    if let Some(last) = chunks.values().max_by_key(|chunk| chunk.offset) {
        let expected_end = last
            .offset
            .checked_add(last.raw_len.max(last.stored_len))
            .ok_or(ContainerError::IntegerOverflow)?;
        if expected_end != total_len {
            return Err(ContainerError::InvalidDirectory);
        }
    } else if payload_off != total_len {
        return Err(ContainerError::InvalidDirectory);
    }
    validate_zero_range(source, cursor, total_len, budget)?;

    let mut entries = BTreeMap::new();
    let mut previous_guid = None;
    for record in directory.chunks_exact(CHUNK_RECORD_SIZE) {
        let guid = StableGuid::from_u128(u128::from_le_bytes(record[0..16].try_into().unwrap()));
        if entries.contains_key(&guid) || previous_guid.is_some_and(|previous| previous >= guid) {
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
            return Err(ContainerError::InvalidDirectory);
        }
        let hash = record[40..72].try_into().unwrap();
        let alignment = u32::from_le_bytes(record[72..76].try_into().unwrap());
        let absolute = chunk
            .offset
            .checked_add(raw_offset)
            .ok_or(ContainerError::IntegerOverflow)?;
        if alignment == 0 || !alignment.is_power_of_two() || absolute % u64::from(alignment) != 0 {
            return Err(ContainerError::InvalidDirectory);
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

    // Validate in chunk order rather than GUID order: one decode per chunk,
    // one bounded live scratch buffer, no repeated decode for interleaved GUIDs.
    let mut grouped: Vec<_> = entries.values().collect();
    grouped.sort_by_key(|entry| (entry.chunk_id, entry.raw_offset, entry.raw_len, entry.guid));
    let mut position = 0;
    for chunk in chunks.values().copied() {
        let begin = position;
        while position < grouped.len() && grouped[position].chunk_id == chunk.id {
            position += 1;
        }
        if position == begin {
            return Err(ContainerError::InvalidDirectory);
        }
        let members = &grouped[begin..position];
        let decoded = if matches!(chunk.codec, Codec::Deflate | Codec::BlockDeflate) {
            let metadata_bytes = dir_len
                .checked_add(table_len)
                .ok_or(ContainerError::IntegerOverflow)?;
            let decoded =
                decode_chunk_source(source, chunk, metadata_bytes, budget, block_decode_backend)?;
            if blake3::hash(&decoded).as_bytes() != &chunk.hash {
                return Err(ContainerError::HashMismatch);
            }
            Some(decoded)
        } else {
            validate_hash_range(
                source,
                chunk.offset,
                chunk
                    .offset
                    .checked_add(chunk.raw_len)
                    .ok_or(ContainerError::IntegerOverflow)?,
                chunk.hash,
                budget,
            )?;
            None
        };
        let mut cursor = 0_u64;
        for entry in members {
            if entry.raw_offset < cursor {
                return Err(ContainerError::InvalidDirectory);
            }
            let end = entry
                .raw_offset
                .checked_add(entry.raw_len)
                .ok_or(ContainerError::IntegerOverflow)?;
            if let Some(decoded) = &decoded {
                let start = usize::try_from(entry.raw_offset)
                    .map_err(|_| ContainerError::IntegerOverflow)?;
                let end_usize =
                    usize::try_from(end).map_err(|_| ContainerError::IntegerOverflow)?;
                if decoded[cursor as usize..start]
                    .iter()
                    .any(|byte| *byte != 0)
                {
                    return Err(ContainerError::InvalidDirectory);
                }
                if blake3::hash(&decoded[start..end_usize]).as_bytes() != &entry.hash {
                    return Err(ContainerError::HashMismatch);
                }
            } else {
                validate_zero_range(
                    source,
                    chunk.offset + cursor,
                    chunk.offset + entry.raw_offset,
                    budget,
                )?;
                validate_hash_range(
                    source,
                    chunk.offset + entry.raw_offset,
                    chunk.offset + end,
                    entry.hash,
                    budget,
                )?;
            }
            cursor = end;
        }
        if cursor != chunk.raw_len {
            return Err(ContainerError::InvalidDirectory);
        }
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
    existing_metadata_bytes: u64,
    budget: &ContainerReadBudget,
    block_decode_backend: &dyn crate::block::BlockDecodeBackend,
) -> Result<ReadBuffer, ContainerError> {
    budget.limits().unit(chunk.stored_len, chunk.raw_len)?;
    if chunk.codec == Codec::BlockDeflate {
        let index = crate::block::parse_index(
            source,
            chunk.offset,
            chunk.stored_len,
            existing_metadata_bytes,
            budget,
        )?;
        if index.raw_len() != chunk.raw_len {
            return Err(ContainerError::InvalidDirectory);
        }
        let mut decoded = budget.buffer(chunk.raw_len)?;
        index.decode_range_with_backend(source, 0, &mut decoded, budget, block_decode_backend)?;
        return Ok(decoded);
    }
    let mut stored = budget.buffer(chunk.stored_len)?;
    read_from_source(source, chunk.offset, &mut stored)?;
    match chunk.codec {
        Codec::Raw => Ok(stored),
        Codec::Deflate => decode_deflate(&stored, chunk.raw_len, budget),
        Codec::BlockDeflate => Err(ContainerError::InvalidDirectory),
    }
}

fn validate_zero_range(
    source: &dyn crate::RangeSource,
    start: u64,
    end: u64,
    budget: &ContainerReadBudget,
) -> Result<(), ContainerError> {
    validate_zero_range_as(source, start, end, ContainerError::InvalidDirectory, budget)
}

fn validate_zero_range_as(
    source: &dyn crate::RangeSource,
    start: u64,
    end: u64,
    error: ContainerError,
    budget: &ContainerReadBudget,
) -> Result<(), ContainerError> {
    validate_range(source, start, end, budget, |chunk| {
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
    budget: &ContainerReadBudget,
) -> Result<(), ContainerError> {
    let mut hasher = blake3::Hasher::new();
    validate_range(source, start, end, budget, |chunk| {
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
    budget: &ContainerReadBudget,
) -> Result<[u8; 32], ContainerError> {
    let mut hasher = blake3::Hasher::new();
    validate_range(source, start, end, budget, |chunk| {
        hasher.update(chunk);
        Ok(())
    })?;
    Ok(*hasher.finalize().as_bytes())
}

fn validate_range<F>(
    source: &dyn crate::RangeSource,
    start: u64,
    end: u64,
    budget: &ContainerReadBudget,
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
    let mut buffer = budget.buffer((end - start).min(VALIDATION_CHUNK_SIZE as u64))?;
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

fn decode_deflate(
    stored: &[u8],
    raw_len: u64,
    budget: &ContainerReadBudget,
) -> Result<ReadBuffer, ContainerError> {
    budget.limits().unit(stored.len() as u64, raw_len)?;
    let raw_len = usize::try_from(raw_len).map_err(|_| ContainerError::IntegerOverflow)?;
    let capacity = raw_len
        .checked_add(1)
        .ok_or(ContainerError::IntegerOverflow)?;
    let mut output = budget.buffer(capacity as u64)?;
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
    budget: &ContainerReadBudget,
) -> Result<(), ContainerError> {
    budget.limits().unit(stored.len() as u64, raw_len)?;
    let mut decompressor = Decompress::new(false);
    let mut output = budget.buffer(raw_len.saturating_add(1).min(VALIDATION_CHUNK_SIZE as u64))?;
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
            // Finish requires enough output space for the entire remainder.
            // This validator deliberately reuses a bounded streaming buffer.
            .decompress(input, &mut output, FlushDecompress::None)
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

#[cfg(test)]
mod deflate_validation_tests {
    use std::io::Write;

    use flate2::{write::DeflateEncoder, Compression};

    use super::{validate_deflate, ContainerError, ContainerReadBudget, VALIDATION_CHUNK_SIZE};
    use crate::ContainerReadLimits;

    fn fixture(len: usize) -> (Vec<u8>, Vec<u8>, [u8; 32]) {
        let raw: Vec<_> = (0..len)
            .map(|index| ((index.wrapping_mul(31) ^ (index >> 7)) & 255) as u8)
            .collect();
        let hash = *blake3::hash(&raw).as_bytes();
        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&raw).unwrap();
        (raw, encoder.finish().unwrap(), hash)
    }

    fn budget() -> ContainerReadBudget {
        ContainerReadBudget::new(ContainerReadLimits {
            max_in_flight_bytes: VALIDATION_CHUNK_SIZE as u64,
            ..ContainerReadLimits::default()
        })
    }

    #[test]
    fn deflate_validation_streams_across_output_buffer_boundaries() {
        for len in [
            0,
            1,
            VALIDATION_CHUNK_SIZE - 1,
            VALIDATION_CHUNK_SIZE,
            VALIDATION_CHUNK_SIZE + 1,
            2 * VALIDATION_CHUNK_SIZE,
            3 * VALIDATION_CHUNK_SIZE + 17,
        ] {
            let (_, stored, hash) = fixture(len);
            let budget = budget();
            assert_eq!(validate_deflate(&stored, len as u64, hash, &budget), Ok(()));
            assert_eq!(budget.in_flight_bytes(), 0);
            assert_eq!(
                budget.peak_in_flight_bytes(),
                (len + 1).min(VALIDATION_CHUNK_SIZE) as u64
            );
        }
    }

    #[test]
    fn deflate_validation_rejects_corruption_and_releases_bounded_buffer() {
        for len in [0, 1, VALIDATION_CHUNK_SIZE, 3 * VALIDATION_CHUNK_SIZE + 17] {
            let (_, stored, hash) = fixture(len);
            let budget = budget();
            for cut in [0, 1, stored.len() / 2, stored.len() - 1] {
                // A severely truncated input can hit the expansion gate before
                // entering the codec. Preserve that earlier safety rejection.
                let expected = budget
                    .limits()
                    .unit(cut as u64, len as u64)
                    .err()
                    .unwrap_or(ContainerError::CompressionFailed);
                assert_eq!(
                    validate_deflate(&stored[..cut], len as u64, hash, &budget),
                    Err(expected),
                    "incorrect truncation rejection: raw={len}, stored={cut}"
                );
                assert_eq!(budget.in_flight_bytes(), 0);
            }
            let mut trailing = stored.clone();
            trailing.push(0);
            assert_eq!(
                validate_deflate(&trailing, len as u64, hash, &budget),
                Err(ContainerError::CompressionFailed)
            );
            assert_eq!(budget.in_flight_bytes(), 0);
            for declared in [len.saturating_sub(1), len + 1] {
                if declared == len {
                    continue;
                }
                assert_eq!(
                    validate_deflate(&stored, declared as u64, hash, &budget),
                    Err(ContainerError::CompressionFailed)
                );
                assert_eq!(budget.in_flight_bytes(), 0);
            }
            let mut wrong_hash = hash;
            wrong_hash[0] ^= 1;
            assert_eq!(
                validate_deflate(&stored, len as u64, wrong_hash, &budget),
                Err(ContainerError::HashMismatch)
            );
            assert_eq!(budget.in_flight_bytes(), 0);
            assert!(budget.peak_in_flight_bytes() <= VALIDATION_CHUNK_SIZE as u64);
        }
    }
}
