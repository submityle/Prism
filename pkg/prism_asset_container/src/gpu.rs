//! Optional real-device `BlockDeflate` backend.
//!
//! `BlockDeflate` stores independently addressable raw-DEFLATE streams. This
//! adapter owns only the format boundary: the GPU crate performs the complete
//! raw-DEFLATE decode, while the container keeps range, budget, and BLAKE3
//! validation authoritative.

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt;
use core::ops::Range;
use std::sync::OnceLock;

use prism_gdeflate_gpu::{
    DecodePath, DecoderError, DeflateBatchLimits, DeflateTile, GDeflateDecoder, GpuContext,
    GpuDeflate, ResidentDeflateTile,
};
use prism_render_architecture::compression::validate_deflate_bounded;

use crate::block::{
    BlockCodec, BlockDecodeBackend, BlockDecodePath, BlockDecodeReport, BlockDecodeRequest,
};
use crate::budget::zeroed;
use crate::format::{ChunkInfo, Codec, ContainerError, CHUNK_RECORD_SIZE, CHUNK_TABLE_RECORD_SIZE};
#[cfg(feature = "mmap")]
use crate::MmapRangeSource;
use crate::{ContainerReadBudget, ContainerReader, MemoryRangeSource};

enum ResidentBacking {
    Owned(Arc<[u8]>),
    #[cfg(feature = "mmap")]
    Mapped(Arc<MmapRangeSource>),
}

impl ResidentBacking {
    fn bytes(&self) -> &[u8] {
        match self {
            Self::Owned(bytes) => bytes,
            #[cfg(feature = "mmap")]
            Self::Mapped(source) => source.mapped_bytes(),
        }
    }
}

/// A reusable GPU-preferred decoder for `BlockDeflate` blocks.
pub struct GpuBlockDecodeBackend {
    context: Arc<GpuContext>,
    decoder: GDeflateDecoder,
    resident_decoder: OnceLock<Arc<GpuDeflate>>,
}

impl fmt::Debug for GpuBlockDecodeBackend {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GpuBlockDecodeBackend")
            .field("adapter", &self.context.adapter_info().name)
            .finish_non_exhaustive()
    }
}

impl GpuBlockDecodeBackend {
    /// Compiles the GPU pipeline once and reuses it for all block reads.
    #[must_use]
    pub fn new(context: Arc<GpuContext>) -> Self {
        let decoder = GDeflateDecoder::new(&context);
        Self {
            context,
            decoder,
            resident_decoder: OnceLock::new(),
        }
    }

    /// Returns the shared GPU context.
    #[must_use]
    pub fn context(&self) -> &Arc<GpuContext> {
        &self.context
    }

    /// Binds a previously validated chunk to its exact immutable stored bytes.
    ///
    /// The reader must already have completed ordinary mount validation,
    /// including every block hash and the decoded chunk hash. This method
    /// hashes the complete *stored* container and compares its identity with
    /// that reader; caller-supplied metadata or a parsed index alone cannot
    /// create this capability. The retained `Arc` prevents subsequent mutation.
    /// As with ordinary reader access, range sources and injected decoders used
    /// at mount must obey their immutable-source and decode contracts.
    ///
    /// No decoded payload is produced here. Resident reads subsequently use
    /// GPU-only batch decoding, with no payload readback or CPU fallback.
    pub fn bind_validated_chunk(
        &self,
        reader: &ContainerReader,
        bytes: Arc<[u8]>,
        chunk_id: u32,
        budget: ContainerReadBudget,
    ) -> Result<GpuResidentBlockPayload, ContainerError> {
        let source = MemoryRangeSource::new(bytes.clone());
        self.bind_validated_source(
            reader,
            ResidentBacking::Owned(bytes),
            &source,
            chunk_id,
            budget,
        )
    }

    /// Binds a previously validated mmap-backed v2 chunk without copying the
    /// complete container into an `Arc<[u8]>`. The reader must be the source of
    /// the retained mapping, and mount-time container/block hashes remain the
    /// integrity authority. The mapping must remain immutable for its lifetime.
    #[cfg(feature = "mmap")]
    pub fn bind_validated_mmap_chunk(
        &self,
        reader: &ContainerReader,
        chunk_id: u32,
        budget: ContainerReadBudget,
    ) -> Result<GpuResidentBlockPayload, ContainerError> {
        let source = reader
            .validated_mmap_source()
            .ok_or(ContainerError::InvalidDirectory)?;
        self.bind_validated_source(
            reader,
            ResidentBacking::Mapped(source.clone()),
            source.as_ref(),
            chunk_id,
            budget,
        )
    }

    fn bind_validated_source(
        &self,
        reader: &ContainerReader,
        backing: ResidentBacking,
        source: &dyn crate::RangeSource,
        chunk_id: u32,
        budget: ContainerReadBudget,
    ) -> Result<GpuResidentBlockPayload, ContainerError> {
        let info = reader.info();
        let limits = budget.limits();
        limits.check(
            "container bytes",
            info.total_len,
            limits.max_container_bytes,
        )?;
        if let ResidentBacking::Owned(bytes) = &backing
            && (bytes.len() as u64 != info.total_len
                || blake3::hash(bytes).as_bytes() != &info.content_hash)
        {
            return Err(ContainerError::HashMismatch);
        }
        let chunk = reader
            .chunk(chunk_id)
            .ok_or(ContainerError::InvalidDirectory)?;
        if chunk.codec != Codec::BlockDeflate {
            return Err(ContainerError::CodecNotSupportedForLayout(chunk.codec));
        }
        let metadata_bytes = u64::from(info.entry_count)
            .checked_mul(CHUNK_RECORD_SIZE as u64)
            .and_then(|size| {
                u64::from(info.chunk_count)
                    .checked_mul(CHUNK_TABLE_RECORD_SIZE as u64)
                    .and_then(|table| size.checked_add(table))
            })
            .ok_or(ContainerError::IntegerOverflow)?;
        limits.check(
            "entry count",
            u64::from(info.entry_count),
            u64::from(limits.max_records),
        )?;
        limits.check(
            "chunk count",
            u64::from(info.chunk_count),
            u64::from(limits.max_records),
        )?;
        let index = crate::block::parse_index(
            source,
            chunk.offset,
            chunk.stored_len,
            metadata_bytes,
            &budget,
        )?;
        if index.raw_len() != chunk.raw_len {
            return Err(ContainerError::InvalidDirectory);
        }
        limits.check(
            "resident block size",
            u64::from(index.block_size()),
            64 * 1024,
        )?;
        // Mount checks the outer chunk digest and each block digest. Explicitly
        // bind the inner whole-payload digest to that authoritative chunk hash.
        if index.raw_hash() != chunk.hash {
            return Err(ContainerError::HashMismatch);
        }
        let decoder = self
            .resident_decoder
            .get_or_init(|| Arc::new(GpuDeflate::new(&self.context)))
            .clone();
        Ok(GpuResidentBlockPayload {
            backing,
            chunk,
            index,
            budget,
            context: self.context.clone(),
            decoder,
        })
    }
}

/// A private validation capability bound to an immutable stored container.
///
/// Created only by [`GpuBlockDecodeBackend::bind_validated_chunk`]. It retains
/// its context, decoder, bytes, and budgeted metadata independently of the
/// original reader/backend. Stored bytes are retained; decoded bytes are not.
pub struct GpuResidentBlockPayload {
    backing: ResidentBacking,
    chunk: ChunkInfo,
    index: crate::block::BlockIndex,
    budget: ContainerReadBudget,
    context: Arc<GpuContext>,
    decoder: Arc<GpuDeflate>,
}

/// One all-or-nothing resident range result, in increasing payload order.
#[derive(Debug)]
pub struct GpuResidentBlockRange {
    range: Range<u64>,
    slices: Vec<GpuResidentBlockSlice>,
}

impl GpuResidentBlockRange {
    /// Exact requested range, relative to the decoded chunk.
    #[must_use]
    pub fn range(&self) -> Range<u64> {
        self.range.clone()
    }

    /// Owned block outputs with explicit offsets for the requested bytes.
    #[must_use]
    pub fn slices(&self) -> &[GpuResidentBlockSlice] {
        &self.slices
    }

    /// Transfers the independent outputs to downstream consumers.
    #[must_use]
    pub fn into_slices(self) -> Vec<GpuResidentBlockSlice> {
        self.slices
    }
}

/// An owned decoded block and the precise bytes selected from it.
///
/// Offsets need not be word aligned. Consumers must use byte addressing or
/// align GPU copies themselves; allocation padding is never logical output.
#[derive(Debug)]
pub struct GpuResidentBlockSlice {
    tile: ResidentDeflateTile,
    block_index: u32,
    buffer_range: Range<u64>,
    output_offset: u64,
}

impl GpuResidentBlockSlice {
    /// Complete owned block output for binding or copying on the same device.
    #[must_use]
    pub fn tile(&self) -> &ResidentDeflateTile {
        &self.tile
    }

    /// Index in the original independent block table.
    #[must_use]
    pub fn block_index(&self) -> u32 {
        self.block_index
    }

    /// Exact requested byte interval within the block's GPU buffer.
    #[must_use]
    pub fn buffer_range(&self) -> Range<u64> {
        self.buffer_range.clone()
    }

    /// Byte offset within the complete requested output range.
    #[must_use]
    pub fn output_offset(&self) -> u64 {
        self.output_offset
    }

    /// Transfers the block while preserving its selected interval and offset.
    #[must_use]
    pub fn into_parts(self) -> (ResidentDeflateTile, Range<u64>, u64) {
        (self.tile, self.buffer_range, self.output_offset)
    }
}

impl GpuResidentBlockPayload {
    /// Validated chunk metadata from the reader that authorized this payload.
    #[must_use]
    pub fn chunk(&self) -> ChunkInfo {
        self.chunk
    }

    /// Device on which returned block buffers may be consumed.
    #[must_use]
    pub fn context(&self) -> &Arc<GpuContext> {
        &self.context
    }

    /// Decodes only independent blocks intersecting `offset..offset + len`.
    ///
    /// Admission includes complete touched blocks, lane padding, uniforms, and
    /// status staging. Host byte buffers use the shared container budget. All
    /// ranges and admission are checked before allocation or submission. The
    /// CPU only validates compressed syntax/counts and transposes stored words;
    /// it neither inflates nor receives decoded payload bytes.
    ///
    /// Integrity relies on the authoritative mount-time block/chunk BLAKE3
    /// checks bound to these exact immutable encoded bytes. GPU statuses must
    /// additionally validate every block before any result is published.
    pub fn read_range_resident(
        &self,
        offset: u64,
        len: u64,
        admission: DeflateBatchLimits,
    ) -> Result<GpuResidentBlockRange, ContainerError> {
        let end = offset
            .checked_add(len)
            .ok_or(ContainerError::IntegerOverflow)?;
        if offset > self.chunk.raw_len || end > self.chunk.raw_len {
            return Err(ContainerError::InvalidRange);
        }
        let limits = self.budget.limits();
        limits.check("output bytes", len, limits.max_output_bytes)?;
        if len == 0 {
            return Ok(GpuResidentBlockRange {
                range: offset..end,
                slices: Vec::new(),
            });
        }
        let block_size = u64::from(self.index.block_size());
        let first =
            u32::try_from(offset / block_size).map_err(|_| ContainerError::IntegerOverflow)?;
        let last =
            u32::try_from((end - 1) / block_size).map_err(|_| ContainerError::IntegerOverflow)?;
        let count = u64::from(last) - u64::from(first) + 1;
        limits.check(
            "resident tiles",
            count,
            u64::try_from(admission.max_tiles).map_err(|_| ContainerError::IntegerOverflow)?,
        )?;
        let mut device_bytes = 0u64;
        let mut padded_bytes = 0u64;
        // Separate admission from preparation: even the last block can reject
        // this request before any input buffer is allocated or work submitted.
        for number in first..=last {
            let record = resident_record(self.backing.bytes(), &self.index, number)?;
            limits.unit(record.stored.len() as u64, record.raw_len)?;
            let padded = padded_len(record.stream_len()?)?;
            padded_bytes = padded_bytes
                .checked_add(padded)
                .ok_or(ContainerError::IntegerOverflow)?;
            let output_bytes = record.raw_len.div_ceil(4) * 4;
            device_bytes = device_bytes
                .checked_add(padded)
                .and_then(|size| size.checked_add(output_bytes))
                // 32 uniform + 8 status + 8 packed status bytes per tile.
                .and_then(|size| size.checked_add(48))
                .ok_or(ContainerError::IntegerOverflow)?;
        }
        limits.check(
            "resident device bytes",
            device_bytes,
            admission.max_device_bytes,
        )?;
        let count_usize = usize::try_from(count).map_err(|_| ContainerError::IntegerOverflow)?;
        let descriptor_size = size_of::<Vec<u8>>()
            + size_of::<DeflateTile<'_>>()
            + size_of::<GpuResidentBlockSlice>()
            + size_of::<ResidentDeflateTile>();
        let host_bytes = count
            .checked_mul(descriptor_size as u64 + 8)
            // Adapter lane bytes plus the batch transport's copied u32 words.
            .and_then(|size| {
                padded_bytes
                    .checked_mul(2)
                    .and_then(|b| size.checked_add(b))
            })
            .ok_or(ContainerError::IntegerOverflow)?;
        let _reservation = self.budget.reserve(host_bytes)?;
        let mut inputs = Vec::new();
        let mut requests = Vec::new();
        let mut slices = Vec::new();
        inputs
            .try_reserve_exact(count_usize)
            .map_err(|_| ContainerError::BackendAllocationFailed)?;
        requests
            .try_reserve_exact(count_usize)
            .map_err(|_| ContainerError::BackendAllocationFailed)?;
        slices
            .try_reserve_exact(count_usize)
            .map_err(|_| ContainerError::BackendAllocationFailed)?;
        for number in first..=last {
            let record = resident_record(self.backing.bytes(), &self.index, number)?;
            if !record.raw {
                let decoded_len = validate_deflate_bounded(record.stored, record.raw_len as usize)
                    .map_err(|_| ContainerError::CompressionFailed)?;
                if decoded_len as u64 != record.raw_len {
                    return Err(ContainerError::LengthMismatch);
                }
            }
            inputs.push(interleave_record(&record)?);
        }
        for (input, number) in inputs.iter().zip(first..=last) {
            let record = resident_record(self.backing.bytes(), &self.index, number)?;
            requests.push(DeflateTile {
                interleaved: input,
                compressed_size: usize::try_from(record.stream_len()?)
                    .map_err(|_| ContainerError::IntegerOverflow)?,
                expected_output_size: record.raw_len as usize,
            });
        }
        let tiles = self
            .decoder
            .decompress_batch_resident(&self.context, &requests, admission)
            .map_err(|error| ContainerError::BlockDecodeFailed(error.to_string()))?;
        if tiles.len() != count_usize {
            return Err(ContainerError::InvalidDecodeReport);
        }
        let mut covered = 0u64;
        for (tile, number) in tiles.into_iter().zip(first..=last) {
            let record = resident_record(self.backing.bytes(), &self.index, number)?;
            if tile.byte_len() != record.raw_len {
                return Err(ContainerError::InvalidDecodeReport);
            }
            let start = offset.max(record.raw_offset);
            let block_end = record
                .raw_offset
                .checked_add(record.raw_len)
                .ok_or(ContainerError::IntegerOverflow)?;
            let stop = end.min(block_end);
            if start >= stop || start.checked_sub(offset) != Some(covered) {
                return Err(ContainerError::InvalidDirectory);
            }
            covered = covered
                .checked_add(stop - start)
                .ok_or(ContainerError::IntegerOverflow)?;
            slices.push(GpuResidentBlockSlice {
                tile,
                block_index: number,
                buffer_range: start - record.raw_offset..stop - record.raw_offset,
                output_offset: start - offset,
            });
        }
        if covered != len {
            return Err(ContainerError::InvalidDecodeReport);
        }
        Ok(GpuResidentBlockRange {
            range: offset..end,
            slices,
        })
    }
}

struct ResidentRecord<'a> {
    stored: &'a [u8],
    raw_offset: u64,
    raw_len: u64,
    raw: bool,
}

impl ResidentRecord<'_> {
    fn stream_len(&self) -> Result<u64, ContainerError> {
        let len = self.stored.len() as u64;
        if self.raw {
            // Stored DEFLATE framing lets raw and compressed container blocks
            // share one GPU batch without synthesizing decoded CPU payloads.
            len.checked_add(len.div_ceil(u64::from(u16::MAX)) * 5)
                .ok_or(ContainerError::IntegerOverflow)
        } else {
            Ok(len)
        }
    }
}

fn resident_record<'a>(
    bytes: &'a [u8],
    index: &crate::block::BlockIndex,
    number: u32,
) -> Result<ResidentRecord<'a>, ContainerError> {
    // The format module remains the sole owner of wire offsets and records.
    let record = index.encoded_block(bytes, number)?;
    Ok(ResidentRecord {
        stored: record.stored,
        raw_offset: record.raw_offset,
        raw_len: record.raw_len,
        raw: record.codec == BlockCodec::Raw,
    })
}

fn padded_len(len: u64) -> Result<u64, ContainerError> {
    len.checked_add(127)
        .map(|size| size / 128 * 128)
        .ok_or(ContainerError::IntegerOverflow)
}

fn interleave_record(record: &ResidentRecord<'_>) -> Result<Vec<u8>, ContainerError> {
    let mut output = zeroed(padded_len(record.stream_len()?)?)?;
    let rounds = output.len() / 128;
    let mut position = 0usize;
    let mut put = |byte| {
        let word = position / 4;
        let destination = (word % 32 * rounds + word / 32) * 4 + position % 4;
        output[destination] = byte;
        position += 1;
    };
    if record.raw {
        let count = record.stored.len().div_ceil(u16::MAX as usize);
        for (number, chunk) in record.stored.chunks(u16::MAX as usize).enumerate() {
            put(u8::from(number + 1 == count));
            let len = chunk.len() as u16;
            for byte in len.to_le_bytes().into_iter().chain((!len).to_le_bytes()) {
                put(byte);
            }
            for &byte in chunk {
                put(byte);
            }
        }
    } else {
        for &byte in record.stored {
            put(byte);
        }
    }
    Ok(output)
}

impl BlockDecodeBackend for GpuBlockDecodeBackend {
    fn decode_block(
        &self,
        request: BlockDecodeRequest<'_>,
        output: &mut [u8],
    ) -> Result<BlockDecodeReport, ContainerError> {
        if request.raw_len
            != u64::try_from(output.len()).map_err(|_| ContainerError::IntegerOverflow)?
        {
            return Err(ContainerError::LengthMismatch);
        }
        match request.codec {
            BlockCodec::Raw => {
                if request.stored.len() != output.len() {
                    return Err(ContainerError::InvalidDirectory);
                }
                output.copy_from_slice(request.stored);
                Ok(BlockDecodeReport {
                    path: BlockDecodePath::Raw,
                    gpu_attempted: false,
                    compressed_bytes: u64::try_from(request.stored.len())
                        .map_err(|_| ContainerError::IntegerOverflow)?,
                    uncompressed_bytes: u64::try_from(output.len())
                        .map_err(|_| ContainerError::IntegerOverflow)?,
                })
            }
            BlockCodec::Deflate => {
                let decoded = self
                    .decoder
                    .decode_raw_deflate_bounded(
                        &self.context,
                        request.stored,
                        output.len(),
                        output.len(),
                    )
                    .map_err(map_decoder_error)?;
                if decoded.bytes.len() != output.len() {
                    return Err(ContainerError::LengthMismatch);
                }
                output.copy_from_slice(&decoded.bytes);
                let path = match decoded.report.path {
                    DecodePath::Cpu => BlockDecodePath::CpuDeflate,
                    DecodePath::GpuDeflate => BlockDecodePath::GpuDeflate,
                    DecodePath::CpuFallback => BlockDecodePath::CpuFallback,
                };
                Ok(BlockDecodeReport {
                    path,
                    gpu_attempted: decoded.report.gpu_attempted,
                    compressed_bytes: u64::try_from(decoded.report.compressed_bytes)
                        .map_err(|_| ContainerError::IntegerOverflow)?,
                    uncompressed_bytes: u64::try_from(decoded.report.uncompressed_bytes)
                        .map_err(|_| ContainerError::IntegerOverflow)?,
                })
            }
        }
    }
}

fn map_decoder_error(error: DecoderError) -> ContainerError {
    match error {
        DecoderError::AllocationFailed => ContainerError::BackendAllocationFailed,
        DecoderError::Format(error) => ContainerError::BlockDecodeFailed(format!("{error:?}")),
    }
}
