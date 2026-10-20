//! Integration coverage for the standalone block-addressed codec.
//!
//! The tests exercise the public source/range contract, including physical
//! relocation of expanded chunks and rejection of malformed wire layouts.

use alloc::sync::Arc;
use std::sync::{Arc as StdArc, Mutex};

extern crate alloc;

use prism_asset::StableGuid;
use prism_asset_container::block::{
    encode, encode_with_block_size, parse_index, BlockDecodeBackend, BlockDecodePath,
    BlockDecodeReport, BlockDecodeRequest, BlockDecodeSummary, CpuBlockDecodeBackend,
    DEFAULT_BLOCK_SIZE,
};
use prism_asset_container::{
    ChunkInput, ChunkPlanner, ChunkPolicy, Codec, ContainerError, ContainerReadBudget,
    ContainerReadLimits, ContainerReader, ContainerWriter, MemoryRangeSource, RangeSource,
    RangeSourceError,
};

#[cfg(feature = "gpu")]
use prism_asset_container::GpuBlockDecodeBackend;
#[cfg(feature = "gpu")]
use prism_gdeflate_gpu::GpuContext;

#[derive(Clone, Debug)]
struct RecordingSource {
    bytes: Arc<[u8]>,
    reads: Arc<Mutex<Vec<(u64, usize)>>>,
}

impl RecordingSource {
    fn new(bytes: Vec<u8>) -> Self {
        Self {
            bytes: bytes.into(),
            reads: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn reads(&self) -> Vec<(u64, usize)> {
        self.reads
            .lock()
            .expect("test mutex is not poisoned")
            .clone()
    }
}

impl RangeSource for RecordingSource {
    fn len(&self) -> u64 {
        self.bytes.len() as u64
    }

    fn read_range(&self, offset: u64, out: &mut [u8]) -> Result<(), RangeSourceError> {
        self.reads
            .lock()
            .expect("test mutex is not poisoned")
            .push((offset, out.len()));
        let start = usize::try_from(offset).map_err(|_| RangeSourceError::OutOfBounds)?;
        let end = start
            .checked_add(out.len())
            .ok_or(RangeSourceError::OutOfBounds)?;
        let bytes = self
            .bytes
            .get(start..end)
            .ok_or(RangeSourceError::OutOfBounds)?;
        out.copy_from_slice(bytes);
        Ok(())
    }
}

#[derive(Debug, Default)]
struct RecordingDecodeBackend {
    requests: Mutex<Vec<(u32, u64, u64)>>,
    corrupt: bool,
}

impl BlockDecodeBackend for RecordingDecodeBackend {
    fn decode_block(
        &self,
        request: BlockDecodeRequest<'_>,
        output: &mut [u8],
    ) -> Result<BlockDecodeReport, ContainerError> {
        self.requests
            .lock()
            .expect("decode mutex is not poisoned")
            .push((request.block_index, request.raw_offset, request.raw_len));
        let report = CpuBlockDecodeBackend.decode_block(request, output)?;
        if self.corrupt && !output.is_empty() {
            output[0] ^= 1;
        }
        Ok(report)
    }
}

fn mixed_block_payload() -> Vec<u8> {
    let mut raw = vec![0xA5; 4096];
    let mut state = 0x3141_5926u32;
    for _ in 0..4096 {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        raw.push((state >> 24) as u8);
    }
    raw.extend((0..4096).map(|value| [b'P', b'r', b'i', b's', b'm'][value % 5]));
    raw
}

#[test]
fn decode_summary_counts_only_touched_blocks_and_empty_ranges_do_no_work() {
    let raw = mixed_block_payload();
    let wire = encode_with_block_size(&raw, 4096).unwrap();
    let source = RecordingSource::new(wire);
    let budget = ContainerReadBudget::default();
    let index = parse_index(&source, 0, source.len(), 0, &budget).unwrap();
    let baseline = budget.in_flight_bytes();
    let summary = index
        .validate_all_with_backend_report(&source, &budget, &CpuBlockDecodeBackend)
        .unwrap();
    assert_eq!(summary.blocks, 3);
    assert_eq!(summary.cpu_blocks, 2);
    assert_eq!(summary.raw_blocks, 1);
    assert_eq!(summary.gpu_blocks, 0);
    assert_eq!(summary.fallback_blocks, 0);
    assert_eq!(summary.uncompressed_bytes, raw.len() as u64);
    assert!(summary.compressed_bytes > 4096 && summary.compressed_bytes < raw.len() as u64);
    assert_eq!(budget.in_flight_bytes(), baseline);

    let mut out = [0; 73];
    let summary = index
        .decode_range_with_backend_report(&source, 4077, &mut out, &budget, &CpuBlockDecodeBackend)
        .unwrap();
    assert_eq!(out, raw[4077..4150]);
    assert_eq!(summary.blocks, 2);
    assert_eq!(summary.cpu_blocks, 1);
    assert_eq!(summary.raw_blocks, 1);
    assert_eq!(summary.uncompressed_bytes, 8192);
    assert_eq!(budget.in_flight_bytes(), baseline);

    let before = source.reads().len();
    assert_eq!(
        index
            .decode_range_with_backend_report(
                &source,
                raw.len() as u64,
                &mut [],
                &budget,
                &CpuBlockDecodeBackend,
            )
            .unwrap(),
        BlockDecodeSummary::default(),
    );
    assert_eq!(
        index.decode_range_with_backend_report(
            &source,
            raw.len() as u64 + 1,
            &mut [],
            &budget,
            &CpuBlockDecodeBackend,
        ),
        Err(ContainerError::InvalidRange),
    );
    assert_eq!(source.reads().len(), before);
}

struct InconsistentReportBackend(u8);

impl BlockDecodeBackend for InconsistentReportBackend {
    fn decode_block(
        &self,
        request: BlockDecodeRequest<'_>,
        output: &mut [u8],
    ) -> Result<BlockDecodeReport, ContainerError> {
        let mut report = CpuBlockDecodeBackend.decode_block(request, output)?;
        match self.0 {
            0 => report.compressed_bytes += 1,
            1 => report.uncompressed_bytes += 1,
            2 => report.gpu_attempted = true,
            3 => report.path = BlockDecodePath::Raw,
            4 => report.path = BlockDecodePath::GpuDeflate,
            5 => report.path = BlockDecodePath::CpuFallback,
            _ => unreachable!(),
        }
        Ok(report)
    }
}

#[test]
fn inconsistent_backend_reports_are_rejected_before_copying_and_release_budget() {
    let wire = encode(&vec![0xA5; 4096]).unwrap();
    let source = MemoryRangeSource::new(wire);
    let budget = ContainerReadBudget::default();
    let index = parse_index(&source, 0, source.len(), 0, &budget).unwrap();
    let baseline = budget.in_flight_bytes();
    for mutation in 0..6 {
        let backend = InconsistentReportBackend(mutation);
        let mut out = [0x42; 16];
        assert_eq!(
            index.decode_range_with_backend_report(&source, 0, &mut out, &budget, &backend),
            Err(ContainerError::InvalidDecodeReport),
        );
        assert_eq!(out, [0x42; 16]);
        assert_eq!(budget.in_flight_bytes(), baseline);
        assert_eq!(
            index.validate_all_with_backend_report(&source, &budget, &backend),
            Err(ContainerError::InvalidDecodeReport),
        );
        assert_eq!(budget.in_flight_bytes(), baseline);
    }
}

/// A multi-block payload must round-trip through both the index and range
/// helper; the request below crosses exactly two of the four blocks.
#[test]
fn round_trip_and_range_reads_only_touched_blocks() {
    let mut raw = Vec::new();
    for block in 0..4u8 {
        raw.extend(std::iter::repeat_n(block, 32));
    }
    let wire = encode_with_block_size(&raw, 32).expect("encoding succeeds");
    let source = RecordingSource::new(wire.clone());
    let budget = ContainerReadBudget::default();
    let index = parse_index(&source, 0, wire.len() as u64, 0, &budget).expect("index is valid");
    assert_eq!(index.block_count(), 4);
    assert_eq!(index.raw_len(), raw.len() as u64);

    let reads_before = source.reads().len();
    let mut range = [0u8; 40];
    index
        .decode_range(&source, 16, &mut range, &budget)
        .expect("range is valid");
    assert_eq!(&range[..16], &raw[16..32]);
    assert_eq!(&range[16..], &raw[32..56]);
    let reads = source.reads();
    assert_eq!(reads.len() - reads_before, 2);
    index
        .validate_all(&source, &budget)
        .expect("all hashes validate");
}

/// The optional adapter must be usable by the real container block contract:
/// full validation decodes every block, while a range request decodes only
/// the touched blocks and still checks their hashes and execution reports.
#[cfg(feature = "gpu")]
#[test]
fn gpu_backend_decodes_blockdeflate_mount_and_ranges() {
    let Some(context) = GpuContext::try_headless() else {
        eprintln!("[skip] gpu_backend_decodes_blockdeflate_mount_and_ranges: no GPU");
        return;
    };
    let backend = GpuBlockDecodeBackend::new(Arc::new(context));

    let raw = mixed_block_payload();
    let wire = encode_with_block_size(&raw, 4096).expect("BlockDeflate encoding succeeds");
    let source = MemoryRangeSource::new(wire.clone());
    let budget = ContainerReadBudget::default();
    let index = parse_index(&source, 0, wire.len() as u64, 0, &budget)
        .expect("BlockDeflate index is valid");

    let summary = index
        .validate_all_with_backend_report(&source, &budget, &backend)
        .expect("GPU backend validates all independently encoded blocks");
    assert_eq!(summary.blocks, 3);
    assert_eq!(summary.gpu_blocks, 2);
    assert_eq!(summary.raw_blocks, 1);
    assert_eq!(summary.cpu_blocks, 0);
    assert_eq!(summary.fallback_blocks, 0);
    assert_eq!(summary.uncompressed_bytes, raw.len() as u64);

    let offset = 4096 - 19;
    let expected = &raw[offset..offset + 73];
    let mut range = vec![0; expected.len()];
    let summary = index
        .decode_range_with_backend_report(&source, offset as u64, &mut range, &budget, &backend)
        .expect("range is decoded through the GPU backend");
    assert_eq!(range, expected);
    assert_eq!(summary.blocks, 2);
    assert_eq!(summary.gpu_blocks, 1);
    assert_eq!(summary.raw_blocks, 1);
    assert_eq!(summary.cpu_blocks, 0);
    assert_eq!(summary.fallback_blocks, 0);
    assert_eq!(summary.uncompressed_bytes, 8192);
}

#[cfg(feature = "gpu")]
struct ObservedGpuBackend {
    inner: GpuBlockDecodeBackend,
    reports: Mutex<Vec<BlockDecodeReport>>,
}

#[cfg(feature = "gpu")]
impl BlockDecodeBackend for ObservedGpuBackend {
    fn decode_block(
        &self,
        request: BlockDecodeRequest<'_>,
        output: &mut [u8],
    ) -> Result<BlockDecodeReport, ContainerError> {
        let report = self.inner.decode_block(request, output)?;
        self.reports.lock().unwrap().push(report);
        Ok(report)
    }
}

#[cfg(feature = "gpu")]
impl ObservedGpuBackend {
    fn assert_gpu_calls_and_clear(&self) {
        let mut reports = self.reports.lock().unwrap();
        assert!(
            !reports.is_empty(),
            "container must invoke the injected GPU backend"
        );
        for report in reports.drain(..) {
            assert_eq!(report.path, BlockDecodePath::GpuDeflate);
            assert!(report.gpu_attempted);
        }
    }
}

#[cfg(feature = "gpu")]
#[test]
fn gpu_backend_serves_real_container_mount_full_and_range_without_fallback() {
    let Some(context) = GpuContext::try_headless() else {
        eprintln!("[skip] gpu_backend_serves_real_container_mount_full_and_range_without_fallback: no GPU");
        return;
    };
    let backend = Arc::new(ObservedGpuBackend {
        inner: GpuBlockDecodeBackend::new(Arc::new(context)),
        reports: Mutex::new(Vec::new()),
    });
    let raw = mixed_block_payload();
    let guid = StableGuid::from_path("gpu-container.asset");
    let mut writer = ContainerWriter::new();
    writer.add_aligned(guid, raw.clone(), 1).unwrap();
    let plan = ChunkPlanner::new(ChunkPolicy {
        max_chunk_size: raw.len() as u64,
        chunk_alignment: 64,
        max_chunks: 1,
    })
    .plan([ChunkInput {
        guid,
        size: raw.len() as u64,
        alignment: 1,
        dependencies: Vec::new(),
        priority: 0,
    }])
    .unwrap();
    let bytes = writer
        .finish_with_chunk_compression(&plan, 64, Codec::BlockDeflate)
        .unwrap();
    for range_backed in [false, true] {
        let budget = ContainerReadBudget::default();
        let reader = if range_backed {
            ContainerReader::from_range_source_with_budget_and_block_decode_backend(
                Arc::new(MemoryRangeSource::new(bytes.clone())),
                budget.clone(),
                backend.clone(),
            )
        } else {
            ContainerReader::with_budget_and_block_decode_backend(
                bytes.clone(),
                budget.clone(),
                backend.clone(),
            )
        }
        .unwrap();
        backend.assert_gpu_calls_and_clear();
        assert_eq!(budget.in_flight_bytes(), 0);
        assert_eq!(reader.read_all(guid).unwrap(), raw);
        backend.assert_gpu_calls_and_clear();
        let mut range = [0; 73];
        reader.read_range(guid, 4077, &mut range).unwrap();
        assert_eq!(range, raw[4077..4150]);
        backend.assert_gpu_calls_and_clear();
        assert_eq!(budget.in_flight_bytes(), 0);
    }
}

/// The default is 64 KiB, while a caller-selected block size is carried in
/// the canonical header and still produces dense raw coverage.
#[test]
fn default_and_custom_block_sizes_are_explicit() {
    let raw = vec![7u8; DEFAULT_BLOCK_SIZE as usize + 3];
    let default_wire = encode(&raw).expect("default encoding succeeds");
    let custom_wire = encode_with_block_size(&raw, 1024).expect("custom encoding succeeds");
    let budget = ContainerReadBudget::default();
    let default_index = parse_index(
        &MemoryRangeSource::new(default_wire.clone()),
        0,
        default_wire.len() as u64,
        0,
        &budget,
    )
    .expect("default index is valid");
    let custom_index = parse_index(
        &MemoryRangeSource::new(custom_wire.clone()),
        0,
        custom_wire.len() as u64,
        0,
        &budget,
    )
    .expect("custom index is valid");
    assert_eq!(default_index.block_size(), DEFAULT_BLOCK_SIZE);
    assert_eq!(custom_index.block_size(), 1024);
    default_index
        .validate_all(&MemoryRangeSource::new(default_wire), &budget)
        .expect("default hashes validate");
}

/// Header/index mutations must fail closed before any block is decoded.  The
/// final assertion also proves that the declared wire length cannot hide a
/// trailing byte.
#[test]
fn malformed_header_index_and_trailing_bytes_are_rejected() {
    let raw = vec![3u8; 128];
    let wire = encode_with_block_size(&raw, 64).expect("encoding succeeds");
    let budget = ContainerReadBudget::default();

    let mut bad_reserved = wire.clone();
    bad_reserved[12] = 1;
    let source = MemoryRangeSource::new(bad_reserved);
    assert!(parse_index(&source, 0, source.len(), 0, &budget).is_err());

    let mut bad_index = wire.clone();
    bad_index[144 + 32] = 2;
    let source = MemoryRangeSource::new(bad_index);
    assert!(parse_index(&source, 0, source.len(), 0, &budget).is_err());

    let mut trailing = wire;
    trailing.push(0);
    let source = MemoryRangeSource::new(trailing);
    assert!(parse_index(&source, 0, source.len(), 0, &budget).is_err());
}

/// The public container writer/reader must preserve the block codec marker and
/// route a logical range directly through the block index.
#[test]
fn container_chunk_codec_round_trips_and_serves_ranges() {
    let guid = StableGuid::from_path("block-container.asset");
    let raw: Vec<u8> = (0..192).map(|value| value as u8).collect();
    let mut writer = ContainerWriter::new();
    writer
        .add_aligned(guid, raw.clone(), 1)
        .expect("entry is accepted");
    let plan = ChunkPlanner::new(ChunkPolicy {
        max_chunk_size: 512,
        chunk_alignment: 64,
        max_chunks: 1,
    })
    .plan([ChunkInput {
        guid,
        size: raw.len() as u64,
        alignment: 1,
        dependencies: Vec::new(),
        priority: 0,
    }])
    .expect("chunk plan is valid");
    let bytes = writer
        .finish_with_chunk_compression(&plan, 64, Codec::BlockDeflate)
        .expect("block container is encoded");
    let reader = ContainerReader::new(bytes).expect("block container is valid");
    assert_eq!(
        reader.entry(guid).expect("entry exists").codec,
        Codec::BlockDeflate
    );

    let mut range = vec![0; 37];
    reader
        .read_range(guid, 71, &mut range)
        .expect("range is readable");
    assert_eq!(range, raw[71..108]);
    assert_eq!(
        reader.read_all(guid).expect("full payload is readable"),
        raw
    );
}

#[test]
fn caller_provided_block_backend_is_used_for_mount_and_reads() {
    let guid = StableGuid::from_path("block-custom-backend.asset");
    let raw: Vec<u8> = (0..192).map(|value| value as u8).collect();
    let mut writer = ContainerWriter::new();
    writer
        .add_aligned(guid, raw.clone(), 1)
        .expect("entry is accepted");
    let plan = ChunkPlanner::new(ChunkPolicy {
        max_chunk_size: 512,
        chunk_alignment: 64,
        max_chunks: 1,
    })
    .plan([ChunkInput {
        guid,
        size: raw.len() as u64,
        alignment: 1,
        dependencies: Vec::new(),
        priority: 0,
    }])
    .expect("chunk plan is valid");
    let bytes = writer
        .finish_with_chunk_compression(&plan, 64, Codec::BlockDeflate)
        .expect("block container is encoded");
    let backend = StdArc::new(RecordingDecodeBackend::default());
    let reader = ContainerReader::new_with_block_decode_backend(bytes.clone(), backend.clone())
        .expect("custom backend validates the mount");
    assert_eq!(backend.requests.lock().unwrap().as_slice(), &[(0, 0, 192)]);

    let mut range = vec![0; 37];
    reader
        .read_range(guid, 71, &mut range)
        .expect("custom backend serves ranges");
    assert_eq!(range, raw[71..108]);
    assert_eq!(
        backend.requests.lock().unwrap().as_slice(),
        &[(0, 0, 192), (0, 0, 192)]
    );

    let calls = backend.requests.lock().unwrap().len();
    let mut invalid = [0; 1];
    assert_eq!(
        reader.read_range(guid, raw.len() as u64 + 1, &mut invalid),
        Err(ContainerError::InvalidRange)
    );
    assert_eq!(backend.requests.lock().unwrap().len(), calls);

    let corrupt_backend = StdArc::new(RecordingDecodeBackend {
        corrupt: true,
        ..RecordingDecodeBackend::default()
    });
    assert!(matches!(
        ContainerReader::new_with_block_decode_backend(bytes, corrupt_backend),
        Err(ContainerError::HashMismatch)
    ));
}

#[test]
fn container_writer_relocates_expanded_blocks_and_reader_rejects_physical_overlap() {
    let first = StableGuid::from_path("block-overlap-first.asset");
    let second = StableGuid::from_path("block-overlap-second.asset");
    let mut writer = ContainerWriter::new();
    writer
        .add_aligned(first, vec![1], 1)
        .expect("first entry is accepted");
    writer
        .add_aligned(second, vec![2], 1)
        .expect("second entry is accepted");
    let plan = ChunkPlanner::new(ChunkPolicy {
        max_chunk_size: 1,
        chunk_alignment: 1,
        max_chunks: 2,
    })
    .plan([
        ChunkInput {
            guid: first,
            size: 1,
            alignment: 1,
            dependencies: Vec::new(),
            priority: 1,
        },
        ChunkInput {
            guid: second,
            size: 1,
            alignment: 1,
            dependencies: Vec::new(),
            priority: 0,
        },
    ])
    .expect("two one-byte chunks are planned");

    let mut bytes = writer
        .finish_with_chunk_compression(&plan, 1, Codec::BlockDeflate)
        .expect("encoded chunk starts account for block-index overhead");
    let reader = ContainerReader::new(bytes.clone()).expect("expanded layout is valid");
    let first_chunk = reader.chunk(0).unwrap();
    let second_chunk = reader.chunk(1).unwrap();
    assert!(first_chunk.stored_len > first_chunk.raw_len);
    assert!(second_chunk.offset >= first_chunk.offset + first_chunk.stored_len);
    assert_eq!(reader.read_all(first).unwrap(), [1]);
    assert_eq!(reader.read_all(second).unwrap(), [2]);
    assert_eq!(reader.entry(first).unwrap().raw_offset, 0);
    assert_eq!(reader.entry(second).unwrap().raw_offset, 0);

    let table_offset = usize::try_from(u64::from_le_bytes(bytes[40..48].try_into().unwrap()))
        .expect("table offset fits");
    let table_len = usize::try_from(u64::from_le_bytes(bytes[48..56].try_into().unwrap()))
        .expect("table length fits");
    assert_eq!(table_len, 128);
    bytes[table_offset + 64..table_offset + 72].copy_from_slice(&first_chunk.offset.to_le_bytes());
    let table_hash = *blake3::hash(&bytes[table_offset..table_offset + table_len]).as_bytes();
    bytes[104..136].copy_from_slice(&table_hash);
    assert!(matches!(
        ContainerReader::new(bytes.clone()),
        Err(ContainerError::OutOfBounds)
    ));
    assert!(matches!(
        ContainerReader::from_range_source(Arc::new(MemoryRangeSource::new(bytes))),
        Err(ContainerError::OutOfBounds)
    ));
}

#[test]
fn range_backed_container_reads_only_the_requested_block() {
    let guid = StableGuid::from_path("block-range-backed.asset");
    let mut raw = Vec::with_capacity(DEFAULT_BLOCK_SIZE as usize * 3);
    let mut state = 0x1234_5678_u32;
    for _ in 0..(DEFAULT_BLOCK_SIZE as usize * 3) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        raw.push(state as u8);
    }
    let mut writer = ContainerWriter::new();
    writer
        .add_aligned(guid, raw.clone(), 1)
        .expect("entry is accepted");
    let plan = ChunkPlanner::new(ChunkPolicy {
        max_chunk_size: raw.len() as u64,
        chunk_alignment: 1,
        max_chunks: 1,
    })
    .plan([ChunkInput {
        guid,
        size: raw.len() as u64,
        alignment: 1,
        dependencies: Vec::new(),
        priority: 0,
    }])
    .expect("chunk plan is valid");
    let bytes = writer
        .finish_with_chunk_compression(&plan, 1, Codec::BlockDeflate)
        .expect("block container is encoded");
    let source = Arc::new(RecordingSource::new(bytes));
    let reader = ContainerReader::from_range_source(source.clone()).expect("mount succeeds");
    let mount_reads = source.reads().len();
    let mut range = vec![0; 31];
    reader
        .read_range(guid, DEFAULT_BLOCK_SIZE as u64 + 17, &mut range)
        .expect("range is readable");
    assert_eq!(range, raw[DEFAULT_BLOCK_SIZE as usize + 17..][..31]);
    let reads_after_mount = &source.reads()[mount_reads..];
    assert!(reads_after_mount
        .iter()
        .any(|(_, len)| *len == DEFAULT_BLOCK_SIZE as usize));
    assert!(reads_after_mount
        .iter()
        .all(|(_, len)| *len <= DEFAULT_BLOCK_SIZE as usize));
}

#[test]
fn block_index_respects_metadata_budget_before_reading_the_index() {
    let wire = encode_with_block_size(&[9u8; 128], 64).expect("encoding succeeds");
    let limits = ContainerReadLimits {
        max_metadata_bytes: 144 + 80 - 1,
        ..ContainerReadLimits::default()
    };
    let budget = ContainerReadBudget::new(limits);
    assert!(matches!(
        parse_index(
            &MemoryRangeSource::new(wire.clone()),
            0,
            wire.len() as u64,
            0,
            &budget,
        ),
        Err(ContainerError::ResourceLimit {
            resource: "metadata bytes",
            ..
        })
    ));
}

#[test]
fn malformed_outer_chunk_table_is_rejected_after_table_hash_update() {
    let guid = StableGuid::from_path("block-outer-table.asset");
    let raw = vec![4u8; 512];
    let mut writer = ContainerWriter::new();
    writer.add_aligned(guid, raw, 1).expect("entry is accepted");
    let plan = ChunkPlanner::new(ChunkPolicy {
        max_chunk_size: 1024,
        chunk_alignment: 64,
        max_chunks: 1,
    })
    .plan([ChunkInput {
        guid,
        size: 512,
        alignment: 1,
        dependencies: Vec::new(),
        priority: 0,
    }])
    .expect("chunk plan is valid");
    let mut bytes = writer
        .finish_with_chunk_compression(&plan, 64, Codec::BlockDeflate)
        .expect("block container is encoded");
    let table_offset = usize::try_from(u64::from_le_bytes(bytes[40..48].try_into().unwrap()))
        .expect("table offset fits");
    let stored_len_offset = table_offset + 8;
    let stored_len = u64::from_le_bytes(
        bytes[stored_len_offset..stored_len_offset + 8]
            .try_into()
            .unwrap(),
    );
    bytes[stored_len_offset..stored_len_offset + 8]
        .copy_from_slice(&stored_len.saturating_add(1).to_le_bytes());
    let table_hash = *blake3::hash(&bytes[table_offset..table_offset + 64]).as_bytes();
    bytes[104..136].copy_from_slice(&table_hash);
    let error = ContainerReader::new(bytes).expect_err("mutated table must be rejected");
    assert!(
        matches!(
            error,
            ContainerError::InvalidHeader
                | ContainerError::OutOfBounds
                | ContainerError::InvalidDirectory
        ),
        "unexpected error: {error:?}"
    );
}
