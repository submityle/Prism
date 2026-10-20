//! Integration coverage for the standalone block-addressed codec.
//!
//! The tests exercise the public source/range contract, including physical
//! relocation of expanded chunks and rejection of malformed wire layouts.

use alloc::sync::Arc;
use std::sync::Mutex;

extern crate alloc;

use prism_asset::StableGuid;
use prism_asset_container::block::{
    encode, encode_with_block_size, parse_index, DEFAULT_BLOCK_SIZE,
};
use prism_asset_container::{
    ChunkInput, ChunkPlanner, ChunkPolicy, Codec, ContainerError, ContainerReadBudget,
    ContainerReadLimits, ContainerReader, ContainerWriter, MemoryRangeSource, RangeSource,
    RangeSourceError,
};

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
