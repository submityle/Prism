//! End-to-end logical manifest authorization and owned GPU-resident ranges.

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use prism_asset::StableGuid;
use prism_asset_container::{
    BlockDecodeBackend, BlockDecodeReport, BlockDecodeRequest, ChunkInput, ChunkPlanner,
    ChunkPolicy, Codec, ContainerError, ContainerIdentity, ContainerReadBudget,
    ContainerReadLimits, ContainerReader, ContainerWriter, CpuBlockDecodeBackend,
    GpuBlockDecodeBackend, GpuResidentAssetRange, GpuResidentManifestAsset, ManifestReadError,
    ManifestRoute, MemoryRangeSource, OverlaySet, RangeSource, RangeSourceError,
    RemoteRangeFetchError, RemoteRangeFetcher, RemoteRangeRequest, RemoteRangeResponse,
    RemoteRangeSource, RuntimeManifest,
};
use prism_gdeflate_gpu::{buffer, DeflateBatchLimits, GpuContext};

const BLOCK: usize = 65536;
const PREFIX: StableGuid = StableGuid::from_u128(1);
const LOCAL: StableGuid = StableGuid::from_u128(2);
const LOGICAL: StableGuid = StableGuid::from_u128(3);
const ABSENT: StableGuid = StableGuid::from_u128(4);
const NAME: &str = "resident-manifest";
const PRIORITY: i32 = 7;
const RAW_OFFSET: u64 = 192;
const INDEX_BYTES: u64 = 3 * 80;

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

fn zero_admission() -> DeflateBatchLimits {
    DeflateBatchLimits {
        max_tiles: 0,
        max_device_bytes: 0,
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

fn shared_container(raw: &[u8], codec: Codec) -> Arc<[u8]> {
    // Keep canonical TOC order independent of the dependency-driven placement.
    assert!(PREFIX < LOCAL);
    let mut writer = ContainerWriter::new();
    writer.add_aligned(PREFIX, vec![0x31; 137], 16).unwrap();
    writer.add_aligned(LOCAL, raw.to_vec(), 32).unwrap();
    let plan = ChunkPlanner::new(ChunkPolicy {
        max_chunk_size: raw.len() as u64 + 1024,
        chunk_alignment: 64,
        max_chunks: 1,
    })
    .plan([
        ChunkInput {
            guid: LOCAL,
            size: raw.len() as u64,
            alignment: 64,
            dependencies: vec![PREFIX],
            priority: 99,
        },
        ChunkInput {
            guid: PREFIX,
            size: 137,
            alignment: 16,
            dependencies: Vec::new(),
            priority: 0,
        },
    ])
    .unwrap();
    assert_eq!(plan.chunks.len(), 1);
    let placements = &plan.chunks[0].placements;
    assert_eq!(placements[0].guid, PREFIX);
    assert_eq!(placements[0].offset, 0);
    assert_eq!(placements[1].guid, LOCAL);
    assert_eq!(placements[1].offset, RAW_OFFSET);
    writer
        .finish_with_chunk_compression(&plan, 64, codec)
        .unwrap()
        .into()
}

fn identity(reader: &ContainerReader) -> ContainerIdentity {
    let info = reader.info();
    ContainerIdentity {
        byte_len: info.total_len,
        content_hash: info.content_hash,
        generation: info.generation,
    }
}

fn route_manifest(local: StableGuid, priority: i32) -> RuntimeManifest {
    let mut manifest = RuntimeManifest::new();
    manifest
        .insert(ManifestRoute {
            guid: LOGICAL,
            container_name: NAME.into(),
            priority,
            container_guid: local,
        })
        .unwrap();
    manifest
}

fn manifest(reader: &ContainerReader) -> RuntimeManifest {
    let mut manifest = route_manifest(LOCAL, PRIORITY);
    manifest.bind_container(NAME, identity(reader)).unwrap();
    manifest
}

fn overlays(reader: &ContainerReader) -> OverlaySet {
    let mut overlays = OverlaySet::new();
    overlays.mount(NAME, PRIORITY, reader.clone());
    overlays
}

fn bind(
    manifest: &RuntimeManifest,
    overlays: &OverlaySet,
    backend: &GpuBlockDecodeBackend,
) -> Result<GpuResidentManifestAsset, ManifestReadError> {
    manifest
        .bind(overlays)?
        .bind_gpu_resident_asset(LOGICAL, backend)
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

// Test-only consumer readback: production resident APIs never read payloads back.
fn readback(context: &GpuContext, result: GpuResidentAssetRange) -> Vec<u8> {
    assert_eq!(result.logical_guid(), LOGICAL);
    let range = result.range();
    let mut output = vec![0; (range.end - range.start) as usize];
    let mut copied = 0;
    for slice in result.into_slices() {
        assert!(slice.tile().byte_len() >= slice.buffer_range().end);
        let (tile, selected, destination) = slice.into_parts();
        let copy_len = tile.byte_len().div_ceil(4) * 4;
        let stage = buffer::staging(context.device(), "manifest_resident_consumer", copy_len);
        let mut encoder = context.device().create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(tile.buffer(), 0, &stage, 0, copy_len);
        let submission = context.queue().submit([encoder.finish()]);
        let words = buffer::read_back::<u32>(context, &stage, submission).unwrap();
        let decoded: Vec<u8> = words.into_iter().flat_map(u32::to_le_bytes).collect();
        let amount = (selected.end - selected.start) as usize;
        assert_eq!(destination as usize, copied);
        output[destination as usize..destination as usize + amount]
            .copy_from_slice(&decoded[selected.start as usize..selected.end as usize]);
        copied += amount;
    }
    assert_eq!(copied, output.len());
    output
}

fn check_empty_and_invalid_ranges(asset: &GpuResidentManifestAsset, budget: &ContainerReadBudget) {
    let len = asset.entry().raw_len;
    let baseline = budget.in_flight_bytes();
    for offset in [0, len] {
        let empty = asset
            .read_range_resident(offset, 0, zero_admission())
            .unwrap();
        assert_eq!(empty.logical_guid(), LOGICAL);
        assert_eq!(empty.range(), offset..offset);
        assert!(empty.slices().is_empty());
        assert!(empty.into_slices().is_empty());
    }
    // Asset bounds and checked arithmetic must win over zero GPU admission,
    // including ranges that would still fit inside the surrounding chunk.
    for (offset, amount) in [
        (len + 1, 0),
        (len, 1),
        (len - 1, 2),
        (0, len + 1),
        (u64::MAX, 0),
        (u64::MAX, 1),
        (1, u64::MAX),
    ] {
        assert_eq!(
            asset
                .read_range_resident(offset, amount, zero_admission())
                .unwrap_err(),
            ManifestReadError::InvalidRange {
                logical_guid: LOGICAL,
                offset,
                len: usize::try_from(amount).unwrap_or(usize::MAX),
            },
        );
        assert_eq!(budget.in_flight_bytes(), baseline);
    }
    assert!(matches!(
        asset.read_range_resident(0, 1, zero_admission()),
        Err(ManifestReadError::Container(
            ContainerError::ResourceLimit {
                resource: "resident tiles",
                requested: 1,
                limit: 0,
            }
        ))
    ));
    assert_eq!(budget.in_flight_bytes(), baseline);
}

fn odd_range(asset: &GpuResidentManifestAsset) -> GpuResidentAssetRange {
    let start = BLOCK as u64 - RAW_OFFSET - 3;
    let len = BLOCK as u64 + 11;
    let result = asset.read_range_resident(start, len, admission()).unwrap();
    assert_eq!(result.logical_guid(), LOGICAL);
    assert_eq!(result.range(), start..start + len);
    assert_ne!(result.range().start, RAW_OFFSET + start);
    assert_eq!(result.slices().len(), 3);
    let expected = [(0, 65533..65536, 0), (1, 0..65536, 3), (2, 0..8, 65539)];
    for (slice, (block, selected, destination)) in result.slices().iter().zip(expected) {
        assert_eq!(slice.block_index(), block);
        assert_eq!(slice.buffer_range(), selected);
        assert_eq!(slice.output_offset(), destination);
    }
    result
}

#[test]
fn memory_capability_and_outputs_outlive_manifest_reader_backend_and_bytes() {
    let Some(context) = context() else { return };
    let raw = mixed_payload();
    let bytes = shared_container(&raw, Codec::BlockDeflate);
    let weak_bytes = Arc::downgrade(&bytes);
    let calls = Arc::new(CountBackend(AtomicUsize::new(0)));
    let budget = ContainerReadBudget::default();
    let reader = ContainerReader::with_budget_and_block_decode_backend(
        bytes.clone(),
        budget.clone(),
        calls.clone(),
    )
    .unwrap();
    let validated_calls = calls.0.load(Ordering::Relaxed);
    assert_eq!(validated_calls, 3);
    assert_eq!(budget.in_flight_bytes(), 0);
    let entry = reader.entry(LOCAL).unwrap();
    assert_eq!(entry.raw_offset, RAW_OFFSET);
    assert_eq!(entry.chunk_id, reader.entry(PREFIX).unwrap().chunk_id);
    assert_eq!(entry.raw_len, raw.len() as u64);
    let manifest = manifest(&reader);
    let overlays = overlays(&reader);
    let backend = GpuBlockDecodeBackend::new(context.clone());
    let asset = bind(&manifest, &overlays, &backend).unwrap();
    assert_ne!(LOGICAL, LOCAL);
    assert_eq!(asset.logical_guid(), LOGICAL);
    assert_eq!(asset.entry(), entry);
    assert!(Arc::ptr_eq(asset.context(), &context));
    assert_eq!(budget.in_flight_bytes(), INDEX_BYTES);
    drop(manifest);
    drop(overlays);
    drop(backend);
    drop(reader);
    drop(bytes);
    assert!(weak_bytes.upgrade().is_some());

    check_empty_and_invalid_ranges(&asset, &budget);
    let result = odd_range(&asset);
    let head = asset.read_range_resident(0, 7, admission()).unwrap();
    assert_eq!(head.range(), 0..7);
    assert_eq!(head.slices()[0].block_index(), 0);
    assert_eq!(head.slices()[0].buffer_range(), 192..199);
    let tail_offset = raw.len() as u64 - 9;
    let tail = asset
        .read_range_resident(tail_offset, 9, admission())
        .unwrap();
    assert_eq!(tail.range(), tail_offset..raw.len() as u64);
    assert_eq!(tail.slices()[0].block_index(), 2);
    assert_eq!(tail.slices()[0].buffer_range(), 200..209);
    assert_eq!(calls.0.load(Ordering::Relaxed), validated_calls);
    assert_eq!(budget.in_flight_bytes(), INDEX_BYTES);
    drop(asset);
    assert_eq!(budget.in_flight_bytes(), 0);
    assert!(weak_bytes.upgrade().is_none());

    let range = result.range();
    assert_eq!(
        readback(&context, result),
        raw[range.start as usize..range.end as usize]
    );
    assert_eq!(readback(&context, head), raw[..7]);
    assert_eq!(readback(&context, tail), raw[tail_offset as usize..]);
    assert_eq!(budget.in_flight_bytes(), 0);
}

#[cfg(feature = "mmap")]
struct MappedFixtureFile(std::path::PathBuf);

#[cfg(feature = "mmap")]
impl MappedFixtureFile {
    fn new(bytes: &[u8]) -> Self {
        use std::io::Write;

        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "prism-manifest-resident-{}-{stamp}-{}.container",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed),
        ));
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        file.write_all(bytes).unwrap();
        file.sync_all().unwrap();
        Self(path)
    }
}

#[cfg(feature = "mmap")]
impl Drop for MappedFixtureFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[cfg(feature = "mmap")]
#[test]
fn mmap_capability_keeps_immutable_mapping_and_releases_index_before_output_readback() {
    let Some(context) = context() else { return };
    let raw = mixed_payload();
    let bytes = shared_container(&raw, Codec::BlockDeflate);
    let file = MappedFixtureFile::new(&bytes);
    let budget = ContainerReadBudget::default();
    let reader = ContainerReader::from_mmap_file_with_budget(&file.0, budget.clone()).unwrap();
    let baseline = budget.in_flight_bytes();
    assert_eq!(baseline, 0);
    let manifest = manifest(&reader);
    let overlays = overlays(&reader);
    let backend = GpuBlockDecodeBackend::new(context.clone());
    let asset = bind(&manifest, &overlays, &backend).unwrap();
    assert_eq!(asset.logical_guid(), LOGICAL);
    assert_eq!(asset.entry().raw_offset, RAW_OFFSET);
    assert_eq!(
        asset.entry().chunk_id,
        reader.entry(PREFIX).unwrap().chunk_id
    );
    assert!(Arc::ptr_eq(asset.context(), &context));
    assert_eq!(budget.in_flight_bytes(), baseline + INDEX_BYTES);
    drop(manifest);
    drop(overlays);
    drop(backend);
    drop(reader);
    drop(bytes);
    // Keep the file immutable until the final mapping owner has been dropped.
    check_empty_and_invalid_ranges(&asset, &budget);
    let result = odd_range(&asset);
    let range = result.range();
    drop(asset);
    assert_eq!(budget.in_flight_bytes(), baseline);
    drop(file);
    assert_eq!(
        readback(&context, result),
        raw[range.start as usize..range.end as usize]
    );
    assert_eq!(budget.in_flight_bytes(), baseline);
}

#[test]
fn manifest_routes_fail_closed_before_resident_reservations_or_decoding() {
    let Some(context) = context() else { return };
    let bytes = shared_container(&mixed_payload(), Codec::BlockDeflate);
    let calls = Arc::new(CountBackend(AtomicUsize::new(0)));
    let budget = ContainerReadBudget::default();
    let reader =
        ContainerReader::with_budget_and_block_decode_backend(bytes, budget.clone(), calls.clone())
            .unwrap();
    let validated_calls = calls.0.load(Ordering::Relaxed);
    let backend = GpuBlockDecodeBackend::new(context);
    let mounted = overlays(&reader);
    let valid = manifest(&reader);
    let check = |manifest: &RuntimeManifest, mounts: &OverlaySet, expected: ManifestReadError| {
        assert_eq!(bind(manifest, mounts, &backend).err().unwrap(), expected);
        assert_eq!(budget.in_flight_bytes(), 0);
        assert_eq!(calls.0.load(Ordering::Relaxed), validated_calls);
    };
    check(
        &RuntimeManifest::new(),
        &mounted,
        ManifestReadError::MissingRoute(LOGICAL),
    );
    check(
        &valid,
        &OverlaySet::new(),
        ManifestReadError::MissingContainer {
            logical_guid: LOGICAL,
            container_name: NAME.into(),
        },
    );
    check(
        &route_manifest(ABSENT, PRIORITY),
        &mounted,
        ManifestReadError::MissingContainerGuid {
            logical_guid: LOGICAL,
            container_name: NAME.into(),
            container_guid: ABSENT,
        },
    );
    check(
        &route_manifest(LOCAL, PRIORITY),
        &mounted,
        ManifestReadError::MissingContainerIdentity {
            logical_guid: LOGICAL,
            container_name: NAME.into(),
        },
    );
    let mut priority_drift = mounted.clone();
    priority_drift.mount(NAME, PRIORITY + 1, reader.clone());
    check(
        &valid,
        &priority_drift,
        ManifestReadError::PriorityMismatch {
            logical_guid: LOGICAL,
            container_name: NAME.into(),
            manifest_priority: PRIORITY,
            mounted_priority: PRIORITY + 1,
        },
    );
    let mut source_drift = mounted.clone();
    source_drift.mount("shadow", PRIORITY + 1, reader.clone());
    check(
        &valid,
        &source_drift,
        ManifestReadError::SourceMismatch {
            logical_guid: LOGICAL,
            expected: NAME.into(),
            actual: Some("shadow".into()),
        },
    );
    let actual = identity(&reader);
    for field in 0..3 {
        let mut expected = actual;
        match field {
            0 => expected.byte_len += 1,
            1 => expected.content_hash[0] ^= 1,
            _ => expected.generation ^= 1,
        }
        let mut drift = route_manifest(LOCAL, PRIORITY);
        drift.bind_container(NAME, expected).unwrap();
        check(
            &drift,
            &mounted,
            ManifestReadError::ContainerIdentityMismatch {
                logical_guid: LOGICAL,
                container_name: NAME.into(),
                expected_len: expected.byte_len,
                actual_len: actual.byte_len,
                expected_hash: expected.content_hash,
                actual_hash: actual.content_hash,
                expected_generation: expected.generation,
                actual_generation: actual.generation,
            },
        );
    }
    // Rejection must not poison a subsequent valid binding.
    let asset = bind(&valid, &mounted, &backend).unwrap();
    assert_eq!(budget.in_flight_bytes(), INDEX_BYTES);
    drop(asset);
    assert_eq!(budget.in_flight_bytes(), 0);
    assert_eq!(calls.0.load(Ordering::Relaxed), validated_calls);
}

#[test]
fn v1_and_v2_raw_and_deflate_are_not_resident_manifest_assets() {
    let Some(context) = context() else { return };
    let backend = GpuBlockDecodeBackend::new(context);
    for version in [1, 2] {
        for codec in [Codec::Raw, Codec::Deflate] {
            let raw = vec![0x65; 257];
            let bytes: Arc<[u8]> = if version == 1 {
                let mut writer = ContainerWriter::new();
                match codec {
                    Codec::Raw => writer.add(LOCAL, raw).unwrap(),
                    Codec::Deflate => writer.add_deflate(LOCAL, raw).unwrap(),
                    Codec::BlockDeflate => unreachable!(),
                }
                writer.finish().unwrap().into()
            } else {
                shared_container(&raw, codec)
            };
            let budget = ContainerReadBudget::default();
            let reader = ContainerReader::with_budget(bytes, budget.clone()).unwrap();
            assert_eq!(reader.info().version, version);
            assert_eq!(reader.entry(LOCAL).unwrap().codec, codec);
            let manifest = manifest(&reader);
            let overlays = overlays(&reader);
            for _ in 0..2 {
                assert_eq!(
                    bind(&manifest, &overlays, &backend).err().unwrap(),
                    ManifestReadError::Container(ContainerError::CodecNotSupportedForLayout(codec)),
                );
                assert_eq!(budget.in_flight_bytes(), 0);
            }
        }
    }
}

#[derive(Debug)]
struct CountRange {
    source: MemoryRangeSource,
    calls: Arc<AtomicUsize>,
}

impl RangeSource for CountRange {
    fn len(&self) -> u64 {
        self.source.len()
    }

    fn read_range(&self, offset: u64, out: &mut [u8]) -> Result<(), RangeSourceError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.source.read_range(offset, out)
    }
}

#[derive(Debug)]
struct CountRemote {
    bytes: Arc<[u8]>,
    identity: ContainerIdentity,
    calls: Arc<AtomicUsize>,
}

impl RemoteRangeFetcher for CountRemote {
    fn fetch_range(
        &self,
        request: RemoteRangeRequest,
    ) -> Result<RemoteRangeResponse, RemoteRangeFetchError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        assert_eq!(request.identity, self.identity);
        assert!(request.len <= 4093);
        let start = usize::try_from(request.offset)
            .map_err(|_| RemoteRangeFetchError::Rejected("offset overflow".into()))?;
        let end = start
            .checked_add(request.len)
            .ok_or_else(|| RemoteRangeFetchError::Rejected("length overflow".into()))?;
        let bytes = self
            .bytes
            .get(start..end)
            .ok_or_else(|| RemoteRangeFetchError::Rejected("out of bounds".into()))?;
        Ok(RemoteRangeResponse::new(
            request.offset,
            self.identity,
            bytes,
        ))
    }
}

fn reject_range_backing(
    source: Arc<dyn RangeSource>,
    calls: &AtomicUsize,
    context: Arc<GpuContext>,
) {
    let budget = ContainerReadBudget::default();
    let reader = ContainerReader::from_range_source_with_budget(source, budget.clone()).unwrap();
    let validated_reads = calls.load(Ordering::Relaxed);
    assert!(validated_reads > 0);
    assert_eq!(budget.in_flight_bytes(), 0);
    let manifest = manifest(&reader);
    let overlays = overlays(&reader);
    let backend = GpuBlockDecodeBackend::new(context);
    for _ in 0..3 {
        assert_eq!(
            backend
                .bind_validated_reader_chunk(&reader, reader.entry(LOCAL).unwrap().chunk_id)
                .err()
                .unwrap(),
            ContainerError::ResidentBackingUnsupported,
        );
        assert_eq!(calls.load(Ordering::Relaxed), validated_reads);
        assert_eq!(budget.in_flight_bytes(), 0);
        assert_eq!(
            bind(&manifest, &overlays, &backend).err().unwrap(),
            ManifestReadError::Container(ContainerError::ResidentBackingUnsupported),
        );
        assert_eq!(calls.load(Ordering::Relaxed), validated_reads);
        assert_eq!(budget.in_flight_bytes(), 0);
    }
    drop(overlays);
    drop(reader);
    assert_eq!(budget.in_flight_bytes(), 0);
    assert_eq!(calls.load(Ordering::Relaxed), validated_reads);
}

#[test]
fn generic_range_backing_rejects_without_new_reads_or_budget_leaks() {
    let Some(context) = context() else { return };
    let calls = Arc::new(AtomicUsize::new(0));
    let source = Arc::new(CountRange {
        source: MemoryRangeSource::new(shared_container(&mixed_payload(), Codec::BlockDeflate)),
        calls: calls.clone(),
    });
    reject_range_backing(source, &calls, context);
}

#[test]
fn actual_remote_range_backing_rejects_without_fetches_or_budget_leaks() {
    let Some(context) = context() else { return };
    let bytes = shared_container(&mixed_payload(), Codec::BlockDeflate);
    let identity = identity(&ContainerReader::new(bytes.clone()).unwrap());
    let calls = Arc::new(AtomicUsize::new(0));
    let fetcher = Arc::new(CountRemote {
        bytes,
        identity,
        calls: calls.clone(),
    });
    let source = Arc::new(RemoteRangeSource::new(fetcher, identity, 4093).unwrap());
    reject_range_backing(source, &calls, context);
}

#[test]
fn capability_uses_the_authorizing_readers_output_budget() {
    let Some(context) = context() else { return };
    let budget = ContainerReadBudget::new(ContainerReadLimits {
        max_output_bytes: 7,
        ..ContainerReadLimits::default()
    });
    let reader = ContainerReader::with_budget(
        shared_container(&mixed_payload(), Codec::BlockDeflate),
        budget.clone(),
    )
    .unwrap();
    let backend = GpuBlockDecodeBackend::new(context.clone());
    let asset = bind(&manifest(&reader), &overlays(&reader), &backend).unwrap();
    assert_eq!(
        asset.read_range_resident(0, 8, admission()).unwrap_err(),
        ManifestReadError::Container(ContainerError::ResourceLimit {
            resource: "output bytes",
            requested: 8,
            limit: 7,
        }),
    );
    assert_eq!(budget.in_flight_bytes(), INDEX_BYTES);
    let result = asset.read_range_resident(0, 7, admission()).unwrap();
    drop(asset);
    assert_eq!(budget.in_flight_bytes(), 0);
    assert_eq!(readback(&context, result), vec![0xA5; 7]);
}
