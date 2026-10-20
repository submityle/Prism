//! Resident BlockDeflate integration and immutable validation identity.

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use prism_asset::StableGuid;
use prism_asset_container::{
    BlockDecodeBackend, BlockDecodeReport, BlockDecodeRequest, ChunkInput, ChunkPlanner,
    ChunkPolicy, Codec, ContainerError, ContainerReadBudget, ContainerReadLimits, ContainerReader,
    ContainerWriter, CpuBlockDecodeBackend, GpuBlockDecodeBackend, MemoryRangeSource,
};
use prism_gdeflate_gpu::{buffer, DeflateBatchLimits, GpuContext};

const BLOCK: usize = 65536;

fn context() -> Option<Arc<GpuContext>> {
    let context = GpuContext::try_headless();
    assert!(
        context.is_some() || std::env::var_os("PRISM_REQUIRE_GPU").is_none(),
        "PRISM_REQUIRE_GPU requires a real device",
    );
    context.map(Arc::new)
}

fn admission() -> DeflateBatchLimits {
    DeflateBatchLimits {
        max_tiles: 16,
        max_device_bytes: 4 * 1024 * 1024,
    }
}

fn mixed_payload() -> Vec<u8> {
    let mut raw = vec![0xA5; BLOCK];
    let mut state = 0x3141_5926u32;
    for _ in 0..BLOCK {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        raw.push((state >> 24) as u8);
    }
    raw.extend_from_slice(b"Prism-resident-17");
    raw
}

fn container(raw: &[u8]) -> Arc<[u8]> {
    container_with_codec(raw, Codec::BlockDeflate)
}

fn container_with_codec(raw: &[u8], codec: Codec) -> Arc<[u8]> {
    let guid = StableGuid::from_path("resident.asset");
    let mut writer = ContainerWriter::new();
    writer.add_aligned(guid, raw.to_vec(), 1).unwrap();
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
    writer
        .finish_with_chunk_compression(&plan, 64, codec)
        .unwrap()
        .into()
}

struct CountBackend(AtomicUsize);

impl BlockDecodeBackend for CountBackend {
    fn decode_block(
        &self,
        request: BlockDecodeRequest<'_>,
        output: &mut [u8],
    ) -> Result<BlockDecodeReport, ContainerError> {
        self.0.fetch_add(1, Ordering::Relaxed);
        CpuBlockDecodeBackend.decode_block(request, output)
    }
}

#[test]
fn mixed_batch_preserves_exact_ranges_owned_outputs_and_prior_validation() {
    let Some(context) = context() else { return };
    let raw = mixed_payload();
    let mut bytes = container(&raw);
    let calls = Arc::new(CountBackend(AtomicUsize::new(0)));
    let reader =
        ContainerReader::new_with_block_decode_backend(bytes.clone(), calls.clone()).unwrap();
    let validated_calls = calls.0.load(Ordering::Relaxed);
    assert_eq!(validated_calls, 3);
    let chunk = reader.chunk(0).unwrap();
    let base = chunk.offset as usize;
    assert_eq!(bytes[base + 144 + 32], 1);
    assert_eq!(bytes[base + 144 + 80 + 32], 0);
    let budget = ContainerReadBudget::default();
    let backend = GpuBlockDecodeBackend::new(context.clone());
    let payload = backend
        .bind_validated_chunk(&reader, bytes.clone(), 0, budget.clone())
        .unwrap();
    assert_eq!(payload.chunk(), chunk);
    assert!(Arc::ptr_eq(payload.context(), &context));
    assert!(Arc::get_mut(&mut bytes).is_none());
    let baseline = budget.in_flight_bytes();
    assert_eq!(baseline, 3 * 80);
    let start = BLOCK as u64 - 3;
    let len = BLOCK as u64 + 11;
    let result = payload
        .read_range_resident(start, len, admission())
        .unwrap();
    assert_eq!(result.range(), start..start + len);
    assert_eq!(result.slices().len(), 3);
    let expected = [(0, 65533..65536, 0), (1, 0..65536, 3), (2, 0..8, 65539)];
    for (slice, (number, range, destination)) in result.slices().iter().zip(expected) {
        assert_eq!(slice.block_index(), number);
        assert_eq!(slice.buffer_range(), range);
        assert_eq!(slice.output_offset(), destination);
    }
    assert_eq!(calls.0.load(Ordering::Relaxed), validated_calls);
    assert_eq!(budget.in_flight_bytes(), baseline);
    // Outputs retain their buffers independently of all validation/decode state.
    drop(payload);
    drop(backend);
    drop(reader);
    drop(bytes);
    assert_eq!(budget.in_flight_bytes(), 0);
    let mut received = vec![0u8; len as usize];
    for slice in result.into_slices() {
        let (tile, selected, destination) = slice.into_parts();
        let copy_len = tile.byte_len().div_ceil(4) * 4;
        // The consumer can copy complete aligned words, then select exact odd
        // byte ranges. This readback is test evidence, outside the resident API.
        let stage = buffer::staging(context.device(), "container_resident_consumer", copy_len);
        let mut encoder = context.device().create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(tile.buffer(), 0, &stage, 0, copy_len);
        let submission = context.queue().submit([encoder.finish()]);
        let words = buffer::read_back::<u32>(&context, &stage, submission).unwrap();
        let decoded: Vec<u8> = words.into_iter().flat_map(u32::to_le_bytes).collect();
        let amount = (selected.end - selected.start) as usize;
        received[destination as usize..destination as usize + amount]
            .copy_from_slice(&decoded[selected.start as usize..selected.end as usize]);
    }
    assert_eq!(received, raw[start as usize..(start + len) as usize]);
}

#[cfg(feature = "mmap")]
#[test]
fn mmap_backing_survives_reader_drop_for_gpu_resident_ranges() {
    let Some(context) = context() else { return };
    static NEXT_PATH: AtomicUsize = AtomicUsize::new(1);
    let raw = mixed_payload();
    let bytes = container(&raw);
    let path = std::env::temp_dir().join(format!(
        "prism-resident-mmap-{}-{}.container",
        std::process::id(),
        NEXT_PATH.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&path, &bytes).unwrap();
    let reader = ContainerReader::from_mmap_file(&path).unwrap();
    let budget = ContainerReadBudget::default();
    let baseline = budget.in_flight_bytes();
    let backend = GpuBlockDecodeBackend::new(context.clone());
    let payload = backend
        .bind_validated_mmap_chunk(&reader, 0, budget.clone())
        .unwrap();
    assert_eq!(budget.in_flight_bytes(), baseline + 3 * 80);
    drop(reader);

    let start = BLOCK as u64 - 5;
    let len = BLOCK as u64 + 13;
    let result = payload
        .read_range_resident(start, len, admission())
        .unwrap();
    assert_eq!(result.slices().len(), 3);
    let mut received = vec![0_u8; len as usize];
    for slice in result.into_slices() {
        let (tile, selected, destination) = slice.into_parts();
        let copy_len = tile.byte_len().div_ceil(4) * 4;
        let stage = buffer::staging(context.device(), "mmap_resident_consumer", copy_len);
        let mut encoder = context.device().create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(tile.buffer(), 0, &stage, 0, copy_len);
        let submission = context.queue().submit([encoder.finish()]);
        let words = buffer::read_back::<u32>(&context, &stage, submission).unwrap();
        let decoded: Vec<u8> = words.into_iter().flat_map(u32::to_le_bytes).collect();
        let amount = (selected.end - selected.start) as usize;
        received[destination as usize..destination as usize + amount]
            .copy_from_slice(&decoded[selected.start as usize..selected.end as usize]);
    }
    assert_eq!(received, raw[start as usize..(start + len) as usize]);
    drop(payload);
    assert_eq!(budget.in_flight_bytes(), baseline);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn range_and_device_admission_fail_deterministically_then_valid_read_succeeds() {
    let Some(context) = context() else { return };
    let raw = mixed_payload();
    let bytes = container(&raw);
    let reader = ContainerReader::new(bytes.clone()).unwrap();
    let budget = ContainerReadBudget::default();
    let backend = GpuBlockDecodeBackend::new(context);
    let payload = backend
        .bind_validated_chunk(&reader, bytes, 0, budget.clone())
        .unwrap();
    let baseline = budget.in_flight_bytes();
    assert_eq!(
        payload
            .read_range_resident(u64::MAX, 1, admission())
            .unwrap_err(),
        ContainerError::IntegerOverflow,
    );
    for (offset, len) in [(raw.len() as u64 + 1, 0), (raw.len() as u64, 1)] {
        assert_eq!(
            payload
                .read_range_resident(offset, len, admission())
                .unwrap_err(),
            ContainerError::InvalidRange,
        );
    }
    assert_eq!(
        payload
            .read_range_resident(
                BLOCK as u64 - 1,
                2,
                DeflateBatchLimits {
                    max_tiles: 1,
                    ..admission()
                }
            )
            .unwrap_err(),
        ContainerError::ResourceLimit {
            resource: "resident tiles",
            requested: 2,
            limit: 1
        },
    );
    // A 64 KiB raw block needs two stored-DEFLATE frames, 128-byte lane
    // padding, a complete rounded output, and all 48 status/uniform bytes.
    let required = 65664 + 65536 + 48;
    assert_eq!(
        payload
            .read_range_resident(
                BLOCK as u64,
                1,
                DeflateBatchLimits {
                    max_device_bytes: required - 1,
                    ..admission()
                }
            )
            .unwrap_err(),
        ContainerError::ResourceLimit {
            resource: "resident device bytes",
            requested: required,
            limit: required - 1
        },
    );
    assert_eq!(budget.in_flight_bytes(), baseline);
    assert!(payload
        .read_range_resident(
            raw.len() as u64,
            0,
            DeflateBatchLimits {
                max_tiles: 0,
                max_device_bytes: 0
            }
        )
        .unwrap()
        .slices()
        .is_empty());
    let result = payload
        .read_range_resident(
            BLOCK as u64,
            1,
            DeflateBatchLimits {
                max_device_bytes: required,
                ..admission()
            },
        )
        .unwrap();
    assert_eq!(result.slices()[0].block_index(), 1);
    assert_eq!(result.slices()[0].buffer_range(), 0..1);
    assert_eq!(budget.in_flight_bytes(), baseline);
}

#[test]
fn validated_identity_rejects_changed_bytes_and_unrelated_mounts() {
    let Some(context) = context() else { return };
    let raw = mixed_payload();
    let bytes = container(&raw);
    let reader = ContainerReader::new(bytes.clone()).unwrap();
    let chunk = reader.chunk(0).unwrap();
    let backend = GpuBlockDecodeBackend::new(context);
    let budget = ContainerReadBudget::default();
    for position in [0, chunk.offset as usize + 144 + 36, bytes.len() - 1] {
        let mut changed = bytes.to_vec();
        changed[position] ^= 1;
        assert_eq!(
            backend
                .bind_validated_chunk(&reader, changed.into(), 0, budget.clone())
                .err()
                .unwrap(),
            ContainerError::HashMismatch,
        );
    }
    let mut different = raw.clone();
    different[0] ^= 1;
    let different_bytes = container(&different);
    let different_reader = ContainerReader::new(different_bytes).unwrap();
    assert_eq!(
        backend
            .bind_validated_chunk(&different_reader, bytes.clone(), 0, budget.clone())
            .err()
            .unwrap(),
        ContainerError::HashMismatch,
    );
    assert_eq!(
        backend
            .bind_validated_chunk(&reader, bytes.clone(), 99, budget.clone())
            .err()
            .unwrap(),
        ContainerError::InvalidDirectory,
    );
    // Identical content in a distinct Arc is safe: identity is a full BLAKE3
    // digest, not a forgeable caller token or an allocator pointer comparison.
    let independent: Arc<[u8]> = bytes.to_vec().into();
    assert!(!Arc::ptr_eq(&bytes, &independent));
    let bound = backend
        .bind_validated_chunk(&reader, independent, 0, budget.clone())
        .unwrap();
    drop(bound);
    assert_eq!(budget.in_flight_bytes(), 0);
}

#[test]
fn authoritative_block_and_inner_payload_hashes_cannot_be_skipped() {
    let raw = mixed_payload();
    let bytes = container(&raw);
    let reader = ContainerReader::new(bytes.clone()).unwrap();
    let base = reader.chunk(0).unwrap().offset as usize;
    let mut bad_block_hash = bytes.to_vec();
    bad_block_hash[base + 144 + 36] ^= 1;
    let index_len = 3 * 80;
    let index_hash = blake3::hash(&bad_block_hash[base + 144..base + 144 + index_len]);
    bad_block_hash[base + 104..base + 136].copy_from_slice(index_hash.as_bytes());
    assert_eq!(
        ContainerReader::new(bad_block_hash).unwrap_err(),
        ContainerError::HashMismatch
    );
    // Even a replacement raw block with a matching per-block hash/index hash
    // must still pass the authoritative decoded chunk hash at mount.
    let mut bad_chunk_hash = bytes.to_vec();
    let record = base + 144 + 80;
    let data = u64::from_le_bytes(bytes[base + 56..base + 64].try_into().unwrap()) as usize;
    let stored = u64::from_le_bytes(bytes[record + 16..record + 24].try_into().unwrap()) as usize;
    let start = base + data + stored;
    bad_chunk_hash[start] ^= 1;
    let hash = blake3::hash(&bad_chunk_hash[start..start + BLOCK]);
    bad_chunk_hash[record + 36..record + 68].copy_from_slice(hash.as_bytes());
    let hash = blake3::hash(&bad_chunk_hash[base + 144..base + 144 + index_len]);
    bad_chunk_hash[base + 104..base + 136].copy_from_slice(hash.as_bytes());
    assert_eq!(
        ContainerReader::new(bad_chunk_hash).unwrap_err(),
        ContainerError::HashMismatch
    );

    let Some(context) = context() else { return };
    let backend = GpuBlockDecodeBackend::new(context);
    let mut bad_payload_hash = bytes.to_vec();
    bad_payload_hash[base + 72] ^= 1;
    let bad_payload_hash: Arc<[u8]> = bad_payload_hash.into();
    // Some reader versions validate the outer chunk hash without comparing
    // the inner whole-payload hash. Either mount or binding must reject it.
    match ContainerReader::new(bad_payload_hash.clone()) {
        Err(error) => assert_eq!(error, ContainerError::HashMismatch),
        Ok(reader) => assert_eq!(
            backend
                .bind_validated_chunk(&reader, bad_payload_hash, 0, ContainerReadBudget::default())
                .err()
                .unwrap(),
            ContainerError::HashMismatch,
        ),
    }
}

#[test]
fn budgets_are_shared_and_released_on_rejection() {
    let Some(context) = context() else { return };
    let raw = mixed_payload();
    let bytes = container(&raw);
    let reader = ContainerReader::new(bytes.clone()).unwrap();
    let backend = GpuBlockDecodeBackend::new(context);
    let limits = ContainerReadLimits {
        max_output_bytes: 1,
        max_in_flight_bytes: 2048,
        ..Default::default()
    };
    let budget = ContainerReadBudget::new(limits);
    let payload = backend
        .bind_validated_chunk(&reader, bytes.clone(), 0, budget.clone())
        .unwrap();
    let baseline = budget.in_flight_bytes();
    let sibling = backend
        .bind_validated_chunk(&reader, bytes.clone(), 0, budget.clone())
        .unwrap();
    assert_eq!(budget.in_flight_bytes(), baseline * 2);
    drop(sibling);
    assert_eq!(budget.in_flight_bytes(), baseline);
    assert_eq!(
        payload.read_range_resident(0, 2, admission()).unwrap_err(),
        ContainerError::ResourceLimit {
            resource: "output bytes",
            requested: 2,
            limit: 1
        },
    );
    assert!(matches!(
        payload.read_range_resident(BLOCK as u64, 1, admission()),
        Err(ContainerError::ResourceLimit {
            resource: "in-flight bytes",
            ..
        }),
    ));
    assert_eq!(budget.in_flight_bytes(), baseline);
    // A smaller compressed block still fits the shared host-byte budget.
    assert_eq!(
        payload
            .read_range_resident(0, 1, admission())
            .unwrap()
            .slices()
            .len(),
        1
    );
    drop(payload);
    assert_eq!(budget.in_flight_bytes(), 0);
    let limited = ContainerReadBudget::new(ContainerReadLimits {
        max_metadata_bytes: 1,
        ..Default::default()
    });
    assert!(matches!(
        backend.bind_validated_chunk(&reader, bytes, 0, limited.clone()),
        Err(ContainerError::ResourceLimit {
            resource: "metadata bytes",
            ..
        }),
    ));
    assert_eq!(limited.in_flight_bytes(), 0);
}

#[test]
fn binding_enforces_container_metadata_and_complete_block_unit_limits() {
    let Some(context) = context() else { return };
    let raw = mixed_payload();
    let bytes = container(&raw);
    let reader = ContainerReader::new(bytes.clone()).unwrap();
    let backend = GpuBlockDecodeBackend::new(context);
    let policies = [
        (
            ContainerReadLimits {
                max_container_bytes: bytes.len() as u64 - 1,
                ..Default::default()
            },
            "container bytes",
        ),
        (
            ContainerReadLimits {
                max_records: 0,
                ..Default::default()
            },
            "entry count",
        ),
        (
            ContainerReadLimits {
                max_records: 2,
                ..Default::default()
            },
            "block count",
        ),
        (
            ContainerReadLimits {
                max_decoded_unit_bytes: BLOCK as u64 - 1,
                ..Default::default()
            },
            "decoded unit bytes",
        ),
        (
            ContainerReadLimits {
                max_stored_unit_bytes: BLOCK as u64 - 1,
                ..Default::default()
            },
            "stored unit bytes",
        ),
        (
            ContainerReadLimits {
                max_in_flight_bytes: 144 + 3 * 80 - 1,
                ..Default::default()
            },
            "in-flight bytes",
        ),
    ];
    for (limits, expected_resource) in policies {
        let budget = ContainerReadBudget::new(limits);
        let error = backend
            .bind_validated_chunk(&reader, bytes.clone(), 0, budget.clone())
            .err()
            .unwrap();
        assert!(
            matches!(error, ContainerError::ResourceLimit { resource, .. } if resource == expected_resource),
            "{error:?}"
        );
        assert_eq!(budget.in_flight_bytes(), 0);
    }
    for codec in [Codec::Raw, Codec::Deflate] {
        let bytes = container_with_codec(&raw, codec);
        let reader = ContainerReader::new(bytes.clone()).unwrap();
        assert_eq!(
            backend
                .bind_validated_chunk(&reader, bytes, 0, ContainerReadBudget::default())
                .err()
                .unwrap(),
            ContainerError::CodecNotSupportedForLayout(codec),
        );
    }
}

#[test]
fn range_backed_validated_reader_can_bind_exact_owned_content() {
    let Some(context) = context() else { return };
    let raw = mixed_payload();
    let bytes = container(&raw);
    let reader =
        ContainerReader::from_range_source(Arc::new(MemoryRangeSource::new(bytes.clone())))
            .unwrap();
    let backend = GpuBlockDecodeBackend::new(context);
    let payload = backend
        .bind_validated_chunk(&reader, bytes, 0, ContainerReadBudget::default())
        .unwrap();
    // An exact boundary selects only the next block, never its predecessor.
    let result = payload
        .read_range_resident(BLOCK as u64, 4, admission())
        .unwrap();
    assert_eq!(result.slices().len(), 1);
    assert_eq!(result.slices()[0].block_index(), 1);
    assert_eq!(result.slices()[0].buffer_range(), 0..4);
    assert_eq!(result.slices()[0].output_offset(), 0);
}
