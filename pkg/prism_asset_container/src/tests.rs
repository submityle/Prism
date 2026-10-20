use alloc::sync::Arc;
use prism_asset::{AssetReader, StableGuid};
use std::fs::{self, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::sync::Mutex;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::{
    install_runtime_package, install_runtime_package_transaction, ChunkInput, ChunkPlanner,
    ChunkPolicy, Codec, ContainerError, ContainerIdentity, ContainerReader, ContainerWriter,
    FileRangeSource, InstallMount, InstallTombstones, ManifestAssetReader, ManifestError,
    ManifestMetadata, ManifestRangeRequest, ManifestReadError, ManifestRoute, MemoryRangeSource,
    OverlaySet, RangeBatchPolicy, RangeRequest, RangeSource, RangeSourceError,
    RemoteRangeFetchError, RemoteRangeFetcher, RemoteRangeRequest, RemoteRangeResponse,
    RemoteRangeSource, RuntimeInstallPolicy, RuntimeManifest, CHUNK_FORMAT_VERSION, FORMAT_VERSION,
    MAGIC, SUPPORTED_FORMAT_VERSIONS,
};
#[cfg(feature = "mmap")]
use crate::{MmapRangeSource, MmapV2DirectoryView};

fn guid(name: &str) -> StableGuid {
    StableGuid::from_path(name)
}

fn build(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let mut writer = ContainerWriter::new();
    for (name, bytes) in entries {
        writer.add(guid(name), bytes.to_vec()).unwrap();
    }
    writer.finish().unwrap()
}

#[test]
fn deterministic_deflate_payload_round_trips_and_supports_raw_ranges() {
    let id = guid("compressed.asset");
    let bytes = b"repeatable payload ".repeat(512);
    let mut first = ContainerWriter::new();
    first.add_deflate(id, bytes.clone()).unwrap();
    let first_bytes = first.finish().unwrap();

    let mut second = ContainerWriter::new();
    second.add_deflate(id, bytes.clone()).unwrap();
    let second_bytes = second.finish().unwrap();
    assert_eq!(first_bytes, second_bytes);

    let reader = ContainerReader::new(first_bytes.clone()).unwrap();
    let entry = reader.entry(id).unwrap();
    assert_eq!(entry.codec, Codec::Deflate);
    assert_eq!(entry.raw_len, bytes.len() as u64);
    assert!(entry.stored_len < entry.raw_len);
    assert_eq!(reader.read_all(id).unwrap(), bytes);

    let mut range = vec![0; 37];
    reader.read_range(id, 123, &mut range).unwrap();
    assert_eq!(range, bytes[123..160]);

    let ranges = reader
        .read_ranges(
            &[
                RangeRequest {
                    guid: id,
                    offset: 0,
                    len: 5,
                },
                RangeRequest {
                    guid: id,
                    offset: 500,
                    len: 11,
                },
            ],
            RangeBatchPolicy::default(),
        )
        .unwrap();
    assert_eq!(ranges, vec![bytes[0..5].to_vec(), bytes[500..511].to_vec()]);

    let range_reader =
        ContainerReader::from_range_source(Arc::new(MemoryRangeSource::new(first_bytes))).unwrap();
    assert_eq!(range_reader.read_all(id).unwrap(), bytes);
}

#[test]
fn deflate_payload_corruption_is_rejected_at_mount() {
    let id = guid("corrupt-compressed.asset");
    let mut writer = ContainerWriter::new();
    writer.add_deflate(id, b"payload".repeat(128)).unwrap();
    let mut bytes = writer.finish().unwrap();
    let entry = ContainerReader::new(bytes.clone())
        .unwrap()
        .entry(id)
        .unwrap();
    let offset = usize::try_from(entry.offset).unwrap();
    bytes[offset] ^= 0x80;
    assert!(matches!(
        ContainerReader::new(bytes),
        Err(ContainerError::CompressionFailed | ContainerError::HashMismatch)
    ));
}

fn identity(bytes: &[u8]) -> ContainerIdentity {
    let info = ContainerReader::new(bytes.to_vec()).unwrap().info();
    ContainerIdentity {
        byte_len: info.total_len,
        content_hash: info.content_hash,
        generation: info.generation,
    }
}

#[derive(Clone, Debug)]
enum RemoteFetchOutcome {
    Exact,
    Response(RemoteRangeResponse),
    Error(RemoteRangeFetchError),
}

#[derive(Debug)]
struct ControlledRemoteFetcher {
    bytes: Arc<[u8]>,
    outcome: RemoteFetchOutcome,
    requests: Arc<Mutex<Vec<RemoteRangeRequest>>>,
}

impl ControlledRemoteFetcher {
    fn new(
        bytes: impl Into<Arc<[u8]>>,
        outcome: RemoteFetchOutcome,
    ) -> (Arc<Self>, Arc<Mutex<Vec<RemoteRangeRequest>>>) {
        let requests = Arc::new(Mutex::new(Vec::new()));
        (
            Arc::new(Self {
                bytes: bytes.into(),
                outcome,
                requests: Arc::clone(&requests),
            }),
            requests,
        )
    }
}

impl RemoteRangeFetcher for ControlledRemoteFetcher {
    fn fetch_range(
        &self,
        request: RemoteRangeRequest,
    ) -> Result<RemoteRangeResponse, RemoteRangeFetchError> {
        self.requests.lock().unwrap().push(request);
        match &self.outcome {
            RemoteFetchOutcome::Exact => {
                let start = usize::try_from(request.offset)
                    .map_err(|_| RemoteRangeFetchError::Transport("test offset overflow".into()))?;
                let end = start.checked_add(request.len).ok_or_else(|| {
                    RemoteRangeFetchError::Transport("test range overflow".into())
                })?;
                let bytes = self.bytes.get(start..end).ok_or_else(|| {
                    RemoteRangeFetchError::Transport("test range out of bounds".into())
                })?;
                Ok(RemoteRangeResponse::new(
                    request.offset,
                    request.identity,
                    bytes,
                ))
            }
            RemoteFetchOutcome::Response(response) => Ok(response.clone()),
            RemoteFetchOutcome::Error(error) => Err(error.clone()),
        }
    }
}

#[test]
fn remote_range_source_binds_requests_to_identity_and_exact_bytes() {
    let bytes = b"immutable remote source".to_vec();
    let expected_identity = identity(&build(&[("remote-identity.asset", &bytes)]));
    let (fetcher, requests) =
        ControlledRemoteFetcher::new(bytes.clone(), RemoteFetchOutcome::Exact);
    let source = RemoteRangeSource::new(fetcher.clone(), expected_identity, 16).unwrap();

    assert_eq!(source.identity(), expected_identity);
    assert_eq!(source.maximum_request_bytes(), 16);
    assert_eq!(source.maximum_read_bytes().get(), 16);
    assert_eq!(source.len(), expected_identity.byte_len);

    let mut output = [0_u8; 6];
    source.read_range(4, &mut output).unwrap();
    assert_eq!(output, bytes[4..10]);
    assert_eq!(
        requests.lock().unwrap().as_slice(),
        &[RemoteRangeRequest::new(4, 6, expected_identity)]
    );
}

#[test]
fn remote_range_source_rejects_invalid_responses_and_transport_failures() {
    let object = build(&[("remote-errors.asset", b"payload")]);
    let expected_identity = identity(&object);
    let another_identity = identity(&build(&[("different.asset", b"payload")]));
    let requested_bytes = object[3..7].to_vec();

    let cases = [
        (
            RemoteFetchOutcome::Response(RemoteRangeResponse::new(
                3,
                another_identity,
                requested_bytes.clone(),
            )),
            RangeSourceError::RemoteIdentityMismatch,
        ),
        (
            RemoteFetchOutcome::Response(RemoteRangeResponse::new(
                2,
                expected_identity,
                requested_bytes.clone(),
            )),
            RangeSourceError::RemoteOffsetMismatch {
                expected: 3,
                actual: 2,
            },
        ),
        (
            RemoteFetchOutcome::Response(RemoteRangeResponse::new(
                3,
                expected_identity,
                requested_bytes[..3].to_vec(),
            )),
            RangeSourceError::RemoteLengthMismatch {
                expected: 4,
                actual: 3,
            },
        ),
        (
            RemoteFetchOutcome::Response(RemoteRangeResponse::new(
                3,
                expected_identity,
                [requested_bytes.clone(), vec![0]].concat(),
            )),
            RangeSourceError::RemoteLengthMismatch {
                expected: 4,
                actual: 5,
            },
        ),
    ];

    for (outcome, expected_error) in cases {
        let (fetcher, requests) = ControlledRemoteFetcher::new(object.clone(), outcome);
        let source = RemoteRangeSource::new(fetcher, expected_identity, 16).unwrap();
        let mut output = [0_u8; 4];
        assert_eq!(source.read_range(3, &mut output), Err(expected_error));
        assert_eq!(requests.lock().unwrap().len(), 1);
    }

    for error in [
        RemoteRangeFetchError::Rejected("stale generation".into()),
        RemoteRangeFetchError::Transport("connection reset".into()),
    ] {
        let (fetcher, requests) =
            ControlledRemoteFetcher::new(object.clone(), RemoteFetchOutcome::Error(error.clone()));
        let source = RemoteRangeSource::new(fetcher, expected_identity, 16).unwrap();
        let mut output = [0_u8; 4];
        assert_eq!(
            source.read_range(3, &mut output),
            Err(RangeSourceError::RemoteFetch(error))
        );
        assert_eq!(requests.lock().unwrap().len(), 1);
    }
}

#[test]
fn remote_range_source_rejects_invalid_requests_before_fetching() {
    let object = build(&[("remote-bounds.asset", b"payload")]);
    let expected_identity = identity(&object);
    let (fetcher, requests) = ControlledRemoteFetcher::new(object, RemoteFetchOutcome::Exact);
    let source = RemoteRangeSource::new(fetcher, expected_identity, 3).unwrap();

    assert!(matches!(
        RemoteRangeSource::new(
            ControlledRemoteFetcher::new(Vec::<u8>::new(), RemoteFetchOutcome::Exact).0,
            expected_identity,
            0,
        ),
        Err(RangeSourceError::InvalidRequestLimit { maximum: 0 })
    ));

    let mut too_large = [0_u8; 4];
    assert_eq!(
        source.read_range(0, &mut too_large),
        Err(RangeSourceError::RequestTooLarge {
            requested: 4,
            maximum: 3,
        })
    );
    let mut out_of_bounds = [0_u8; 1];
    assert_eq!(
        source.read_range(expected_identity.byte_len, &mut out_of_bounds),
        Err(RangeSourceError::OutOfBounds)
    );
    assert_eq!(
        source.read_range(u64::MAX, &mut out_of_bounds),
        Err(RangeSourceError::OutOfBounds)
    );
    let mut empty = [];
    source
        .read_range(expected_identity.byte_len, &mut empty)
        .unwrap();
    assert!(requests.lock().unwrap().is_empty());
}

#[test]
fn remote_range_source_mount_keeps_container_integrity_validation() {
    let payload = b"remote container payload".repeat(256);
    let bytes = build(&[("remote-mount.asset", &payload)]);
    let expected_identity = identity(&bytes);
    let (fetcher, _) = ControlledRemoteFetcher::new(bytes.clone(), RemoteFetchOutcome::Exact);
    let reader = ContainerReader::from_range_source(Arc::new(
        RemoteRangeSource::new(fetcher, expected_identity, 64 * 1024).unwrap(),
    ))
    .unwrap();
    assert_eq!(
        reader.read_all(guid("remote-mount.asset")).unwrap(),
        payload
    );

    let mut tampered = bytes;
    let payload_offset = u64::from_le_bytes(tampered[36..44].try_into().unwrap()) as usize;
    tampered[payload_offset] ^= 0x80;
    let (fetcher, _) = ControlledRemoteFetcher::new(tampered, RemoteFetchOutcome::Exact);
    assert_eq!(
        ContainerReader::from_range_source(Arc::new(
            RemoteRangeSource::new(fetcher, expected_identity, 64 * 1024).unwrap(),
        ))
        .unwrap_err(),
        ContainerError::HashMismatch
    );
}

#[derive(Debug)]
struct TrackingRangeSource {
    bytes: Arc<[u8]>,
    requests: Arc<Mutex<Vec<(u64, usize)>>>,
}

#[derive(Debug)]
struct ToggleFailureSource {
    bytes: Arc<[u8]>,
    fail_reads: Arc<std::sync::atomic::AtomicBool>,
}

impl RangeSource for ToggleFailureSource {
    fn len(&self) -> u64 {
        self.bytes.len() as u64
    }

    fn read_range(&self, offset: u64, out: &mut [u8]) -> Result<(), RangeSourceError> {
        if self.fail_reads.load(std::sync::atomic::Ordering::Relaxed) {
            return Err(RangeSourceError::Io("injected range failure".into()));
        }
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

impl TrackingRangeSource {
    fn new(bytes: Vec<u8>) -> (Self, Arc<Mutex<Vec<(u64, usize)>>>) {
        let requests = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                bytes: Arc::from(bytes),
                requests: Arc::clone(&requests),
            },
            requests,
        )
    }
}

impl RangeSource for TrackingRangeSource {
    fn len(&self) -> u64 {
        self.bytes.len() as u64
    }

    fn read_range(&self, offset: u64, out: &mut [u8]) -> Result<(), RangeSourceError> {
        self.requests.lock().unwrap().push((offset, out.len()));
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

#[derive(Debug)]
struct CappedRangeSource {
    bytes: Arc<[u8]>,
    maximum_read: usize,
    requests: Arc<Mutex<Vec<(u64, usize)>>>,
    fail_after: Mutex<Option<usize>>,
}

impl RangeSource for CappedRangeSource {
    fn len(&self) -> u64 {
        self.bytes.len() as u64
    }

    fn maximum_read_bytes(&self) -> std::num::NonZeroUsize {
        std::num::NonZeroUsize::new(self.maximum_read).unwrap()
    }

    fn read_range(&self, offset: u64, out: &mut [u8]) -> Result<(), RangeSourceError> {
        let mut requests = self.requests.lock().unwrap();
        let request_index = requests.len();
        requests.push((offset, out.len()));
        if self
            .fail_after
            .lock()
            .unwrap()
            .is_some_and(|limit| request_index >= limit)
        {
            return Err(RangeSourceError::Io("injected capped failure".into()));
        }
        assert!(out.len() <= self.maximum_read);
        let start = usize::try_from(offset).map_err(|_| RangeSourceError::OutOfBounds)?;
        let end = start
            .checked_add(out.len())
            .ok_or(RangeSourceError::OutOfBounds)?;
        out.copy_from_slice(
            self.bytes
                .get(start..end)
                .ok_or(RangeSourceError::OutOfBounds)?,
        );
        Ok(())
    }
}

fn capped_source(
    bytes: Vec<u8>,
    maximum_read: usize,
    fail_after: Option<usize>,
) -> (Arc<CappedRangeSource>, Arc<Mutex<Vec<(u64, usize)>>>) {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let source = Arc::new(CappedRangeSource {
        bytes: Arc::from(bytes),
        maximum_read,
        requests: Arc::clone(&requests),
        fail_after: Mutex::new(fail_after),
    });
    (source, requests)
}

#[test]
fn remote_range_large_container_mount_and_runtime_reads_respect_fetch_cap() {
    let id = guid("remote-large.asset");
    let payload: Vec<u8> = (0..2 * 1024 * 1024).map(|i| (i % 251) as u8).collect();
    let bytes = build(&[("remote-large.asset", &payload)]);
    let expected_identity = identity(&bytes);
    let (fetcher, requests) = ControlledRemoteFetcher::new(bytes, RemoteFetchOutcome::Exact);
    let cap = 4093;
    let source = Arc::new(RemoteRangeSource::new(fetcher, expected_identity, cap).unwrap());
    let budget = crate::ContainerReadBudget::new(crate::ContainerReadLimits {
        max_in_flight_bytes: 128 * 1024,
        ..crate::ContainerReadLimits::default()
    });
    let reader = ContainerReader::from_range_source_with_budget(source, budget.clone()).unwrap();
    assert_eq!(reader.info().content_hash, expected_identity.content_hash);
    assert_eq!(reader.info().generation, expected_identity.generation);
    assert_eq!(budget.in_flight_bytes(), 0);
    assert!(budget.peak_in_flight_bytes() <= 64 * 1024 + 80);

    // The digest pass covers every byte, sequentially, without an object-sized
    // request or allocation. The two preceding requests are prefix and header.
    let mount_requests = requests.lock().unwrap().clone();
    let mut end = 0;
    for request in &mount_requests[2..] {
        assert_eq!(request.offset, end);
        end += request.len as u64;
        if end == expected_identity.byte_len {
            break;
        }
    }
    assert_eq!(end, expected_identity.byte_len);

    requests.lock().unwrap().clear();
    let mut output = vec![0; 96 * 1024 + 11];
    reader.read_range(id, 13, &mut output).unwrap();
    assert_eq!(output, payload[13..13 + output.len()]);
    let physical_start = reader.entry(id).unwrap().offset + 13;
    let reads = requests.lock().unwrap().clone();
    let mut end = physical_start;
    for request in &reads {
        assert_eq!(request.offset, end);
        end += request.len as u64;
    }
    assert_eq!(end, physical_start + output.len() as u64);
    assert_eq!(reads.len(), output.len().div_ceil(cap));
    assert_eq!(
        reader.read_range_owned(id, 77, 8001).unwrap(),
        payload[77..8078]
    );
    assert_eq!(
        reader
            .read_ranges(
                &[
                    RangeRequest::new(id, 7000, 5000),
                    RangeRequest::new(id, 0, 8000)
                ],
                RangeBatchPolicy {
                    max_gap_bytes: 0,
                    max_batch_bytes: 16 * 1024,
                },
            )
            .unwrap(),
        vec![payload[7000..12000].to_vec(), payload[..8000].to_vec()]
    );
    let runtime_requests = requests.lock().unwrap().clone();
    assert!(mount_requests
        .iter()
        .chain(runtime_requests.iter())
        .all(|request| {
            request.len > 0
                && request.len <= cap
                && request.identity == expected_identity
                && request.offset + request.len as u64 <= expected_identity.byte_len
        }));
    assert_eq!(budget.in_flight_bytes(), 0);
}

#[test]
fn remote_range_caps_smaller_than_headers_and_directories_mount_and_read_all() {
    let payload = vec![42; 513];
    let bytes = build(&[("remote-tiny-cap.asset", &payload)]);
    let expected_identity = identity(&bytes);
    for cap in [1, 7, 79, 96, 64 * 1024] {
        let (fetcher, requests) =
            ControlledRemoteFetcher::new(bytes.clone(), RemoteFetchOutcome::Exact);
        let reader = ContainerReader::from_range_source(Arc::new(
            RemoteRangeSource::new(fetcher, expected_identity, cap).unwrap(),
        ))
        .unwrap();
        assert_eq!(reader.info().content_hash, expected_identity.content_hash);
        assert_eq!(
            reader.read_all(guid("remote-tiny-cap.asset")).unwrap(),
            payload
        );
        assert!(requests
            .lock()
            .unwrap()
            .iter()
            .all(|request| request.len <= cap));
    }
}

#[test]
fn capped_range_sources_cover_v1_deflate_and_all_v2_codecs() {
    // Spans multiple validation buffers and independently read blocks.
    let payload = b"remote codec payload ".repeat(8_000);
    let id = guid("remote-codecs.asset");
    let policy = ChunkPolicy {
        max_chunk_size: 256 * 1024,
        chunk_alignment: 4096,
        max_chunks: 1,
    };
    let plan = ChunkPlanner::new(policy)
        .plan([ChunkInput {
            guid: id,
            size: payload.len() as u64,
            alignment: 1,
            dependencies: vec![],
            priority: 0,
        }])
        .unwrap();
    for (version, codec) in [
        (FORMAT_VERSION, Codec::Deflate),
        (CHUNK_FORMAT_VERSION, Codec::Raw),
        (CHUNK_FORMAT_VERSION, Codec::Deflate),
        (CHUNK_FORMAT_VERSION, Codec::BlockDeflate),
    ] {
        let mut writer = ContainerWriter::new();
        let bytes = if version == FORMAT_VERSION {
            writer.add_deflate(id, payload.clone()).unwrap();
            writer.finish().unwrap()
        } else {
            writer.add_aligned(id, payload.clone(), 1).unwrap();
            writer
                .finish_with_chunk_compression(&plan, policy.chunk_alignment, codec)
                .unwrap()
        };
        let expected_identity = ContainerReader::new(bytes.clone())
            .unwrap_or_else(|error| panic!("invalid {version:?}/{codec:?} fixture: {error:?}"))
            .info();
        let expected_identity = ContainerIdentity {
            byte_len: expected_identity.total_len,
            content_hash: expected_identity.content_hash,
            generation: expected_identity.generation,
        };
        let (fetcher, requests) = ControlledRemoteFetcher::new(bytes, RemoteFetchOutcome::Exact);
        // Smaller than both outer and block headers, directory and chunk records.
        let cap = 61;
        let reader = ContainerReader::from_range_source(Arc::new(
            RemoteRangeSource::new(fetcher, expected_identity, cap).unwrap(),
        ))
        .unwrap();
        assert_eq!(reader.info().version, version);
        assert_eq!(reader.info().content_hash, expected_identity.content_hash);
        requests.lock().unwrap().clear();
        let mut output = vec![0; 5003];
        reader.read_range(id, 65_001, &mut output).unwrap();
        assert_eq!(output, payload[65_001..70_004]);
        assert_eq!(reader.read_all(id).unwrap(), payload);
        assert_eq!(
            reader
                .read_ranges(
                    &[
                        RangeRequest::new(id, 11, 2000),
                        RangeRequest::new(id, 70_000, 3000)
                    ],
                    RangeBatchPolicy::default(),
                )
                .unwrap(),
            vec![payload[11..2011].to_vec(), payload[70_000..73_000].to_vec()]
        );
        assert!(requests
            .lock()
            .unwrap()
            .iter()
            .all(|request| request.len <= cap));
        assert_eq!(reader.budget().in_flight_bytes(), 0);
    }
}

#[test]
fn capped_range_reads_fail_fast_and_release_budget_without_retrying_short_reads() {
    let bytes = build(&[("capped-failure.asset", &vec![23; 128 * 1024])]);
    let budget = crate::ContainerReadBudget::default();
    // Prefix + header + two successful digest fragments, then fail mid-hash.
    let (source, requests) = capped_source(bytes.clone(), 1024, Some(4));
    let error = ContainerReader::from_range_source_with_budget(source, budget.clone()).unwrap_err();
    assert_eq!(
        error,
        ContainerError::Source("range source I/O failed: injected capped failure".into())
    );
    assert_eq!(requests.lock().unwrap().len(), 5);
    assert_eq!(budget.in_flight_bytes(), 0);

    let (source, requests) = capped_source(bytes, 1024, None);
    let reader =
        ContainerReader::from_range_source_with_budget(source.clone(), budget.clone()).unwrap();
    requests.lock().unwrap().clear();
    *source.fail_after.lock().unwrap() = Some(2);
    let mut out = vec![0xCC; 4000];
    assert_eq!(
        reader.read_range(guid("capped-failure.asset"), 0, &mut out),
        Err(error.clone())
    );
    assert_eq!(requests.lock().unwrap().len(), 3);
    assert_eq!(&out[..2048], &[23; 2048]);
    assert!(out[2048..].iter().all(|byte| *byte == 0xCC));
    assert_eq!(budget.in_flight_bytes(), 0);
    requests.lock().unwrap().clear();
    assert_eq!(
        reader.read_range_owned(guid("capped-failure.asset"), 0, 4000),
        Err(error)
    );
    assert_eq!(requests.lock().unwrap().len(), 3);
    assert_eq!(budget.in_flight_bytes(), 0);
}

#[derive(Debug)]
struct ShortReadSource {
    bytes: Arc<[u8]>,
    maximum_read: usize,
}

impl RangeSource for ShortReadSource {
    fn len(&self) -> u64 {
        self.bytes.len() as u64
    }

    fn read_range(&self, offset: u64, out: &mut [u8]) -> Result<(), RangeSourceError> {
        let start = usize::try_from(offset).map_err(|_| RangeSourceError::OutOfBounds)?;
        let end = start
            .checked_add(out.len())
            .ok_or(RangeSourceError::OutOfBounds)?;
        let bytes = self
            .bytes
            .get(start..end)
            .ok_or(RangeSourceError::OutOfBounds)?;
        if out.len() > self.maximum_read {
            let actual = self.maximum_read.min(out.len());
            out[..actual].copy_from_slice(&bytes[..actual]);
            return Err(RangeSourceError::ShortRead {
                expected: out.len(),
                actual,
            });
        }
        out.copy_from_slice(bytes);
        Ok(())
    }
}

#[derive(Debug)]
struct DeclaredLengthSource {
    bytes: Arc<[u8]>,
    declared_len: u64,
}

impl RangeSource for DeclaredLengthSource {
    fn len(&self) -> u64 {
        self.declared_len
    }

    fn read_range(&self, offset: u64, out: &mut [u8]) -> Result<(), RangeSourceError> {
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

fn temporary_container_path(label: &str) -> std::path::PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "prism_asset_container_{label}_{}_{}",
        std::process::id(),
        nonce
    ))
}

#[cfg(feature = "mmap")]
#[test]
fn mmap_range_source_mounts_and_reads_an_immutable_container() {
    let id = guid("mmap.asset");
    let payload = b"memory mapped payload".repeat(4096);
    let bytes = build(&[("mmap.asset", &payload)]);
    let path = temporary_container_path("mmap");
    fs::write(&path, &bytes).unwrap();

    let source = MmapRangeSource::open(&path).unwrap();
    assert_eq!(source.len(), bytes.len() as u64);
    let mut range = vec![0; 37];
    source.read_range(96, &mut range).unwrap();
    assert_eq!(range, bytes[96..133]);

    let reader = ContainerReader::from_mmap_file(&path).unwrap();
    assert_eq!(reader.read_all(id).unwrap(), payload);
    fs::remove_file(path).unwrap();
}

#[cfg(feature = "mmap")]
#[test]
fn mmap_range_source_rejects_empty_files_before_mapping() {
    let path = temporary_container_path("mmap-empty");
    fs::write(&path, []).unwrap();
    assert!(matches!(
        MmapRangeSource::open(&path),
        Err(RangeSourceError::Io(message)) if message.contains("empty file")
    ));
    fs::remove_file(path).unwrap();
}

#[cfg(feature = "mmap")]
#[test]
fn mmap_v2_directory_view_matches_reader_for_all_chunk_codecs_without_tree_materialization() {
    let entries = [
        (guid("mapped/a.asset"), b"A".repeat(4096), 0),
        (guid("mapped/b.asset"), b"B".repeat(4096), 1),
    ];
    let policy = ChunkPolicy {
        max_chunk_size: 4096,
        chunk_alignment: 64,
        max_chunks: 2,
    };
    let plan = ChunkPlanner::new(policy)
        .plan(entries.iter().map(|(id, bytes, priority)| ChunkInput {
            guid: *id,
            size: bytes.len() as u64,
            alignment: 1,
            dependencies: vec![],
            priority: *priority,
        }))
        .unwrap();

    for codec in [Codec::Raw, Codec::Deflate, Codec::BlockDeflate] {
        let mut writer = ContainerWriter::new();
        for (id, bytes, _) in &entries {
            writer.add_aligned(*id, bytes.clone(), 1).unwrap();
        }
        let bytes = writer
            .finish_with_chunk_compression(&plan, policy.chunk_alignment, codec)
            .unwrap();
        let expected = ContainerReader::new(bytes.clone()).unwrap();
        let path = temporary_container_path("mmap-v2-view");
        fs::write(&path, &bytes).unwrap();
        let view = MmapV2DirectoryView::open(&path).unwrap();
        let mapped_reader = ContainerReader::from_mmap_file(&path).unwrap();

        assert_eq!(view.info(), expected.info());
        assert_eq!(mapped_reader.info(), expected.info());
        assert_eq!(
            view.guids().collect::<Vec<_>>(),
            expected.guids().collect::<Vec<_>>()
        );
        for (id, _, _) in &entries {
            assert_eq!(view.entry(*id), expected.entry(*id));
            assert_eq!(mapped_reader.entry(*id), expected.entry(*id));
            assert_eq!(
                mapped_reader.read_all(*id).unwrap(),
                expected.read_all(*id).unwrap()
            );
        }
        for chunk_id in 0..expected.info().chunk_count {
            assert_eq!(view.chunk(chunk_id), expected.chunk(chunk_id));
            assert_eq!(mapped_reader.chunk(chunk_id), expected.chunk(chunk_id));
        }
        assert!(!view.contains(guid("mapped/missing.asset")));
        assert_eq!(view.chunk(expected.info().chunk_count), None);
        fs::remove_file(path).unwrap();
    }
}

#[cfg(feature = "mmap")]
#[test]
fn mmap_v2_directory_view_rejects_payload_tampering_and_non_v2_layouts() {
    let id = guid("mapped/tamper.asset");
    let payload = b"mmap validation is strict".repeat(128);
    let policy = ChunkPolicy {
        max_chunk_size: payload.len() as u64,
        chunk_alignment: 64,
        max_chunks: 1,
    };
    let plan = ChunkPlanner::new(policy)
        .plan([ChunkInput {
            guid: id,
            size: payload.len() as u64,
            alignment: 1,
            dependencies: vec![],
            priority: 0,
        }])
        .unwrap();
    let mut writer = ContainerWriter::new();
    writer.add(id, payload).unwrap();
    let mut bytes = writer
        .finish_with_chunk_compression(&plan, 64, Codec::Raw)
        .unwrap();
    let reader = ContainerReader::new(bytes.clone()).unwrap();
    let chunk = reader.chunk(0).unwrap();
    bytes[chunk.offset as usize] ^= 0x80;
    let path = temporary_container_path("mmap-v2-tampered");
    fs::write(&path, bytes).unwrap();
    assert!(matches!(
        MmapV2DirectoryView::open(&path),
        Err(ContainerError::HashMismatch)
    ));
    fs::remove_file(path).unwrap();

    let path = temporary_container_path("mmap-v1-view");
    let v1_bytes = build(&[("mapped/v1.asset", b"legacy")]);
    fs::write(&path, &v1_bytes).unwrap();
    assert!(matches!(
        MmapV2DirectoryView::open(&path),
        Err(ContainerError::UnsupportedVersion(FORMAT_VERSION))
    ));
    let v1_reader = ContainerReader::from_mmap_file(&path).unwrap();
    assert_eq!(
        v1_reader.read_all(guid("mapped/v1.asset")).unwrap(),
        b"legacy"
    );
    fs::remove_file(path).unwrap();
}

#[cfg(feature = "mmap")]
#[test]
fn mmap_v2_directory_view_charges_validation_scratch_and_releases_it_on_failure() {
    let id = guid("mapped/budget.asset");
    let payload = b"bounded mmap directory".repeat(64);
    let policy = ChunkPolicy {
        max_chunk_size: payload.len() as u64,
        chunk_alignment: 64,
        max_chunks: 1,
    };
    let plan = ChunkPlanner::new(policy)
        .plan([ChunkInput {
            guid: id,
            size: payload.len() as u64,
            alignment: 1,
            dependencies: vec![],
            priority: 0,
        }])
        .unwrap();
    let mut writer = ContainerWriter::new();
    writer.add(id, payload).unwrap();
    let bytes = writer
        .finish_with_chunk_compression(&plan, 64, Codec::Raw)
        .unwrap();
    let path = temporary_container_path("mmap-v2-budget");
    fs::write(&path, bytes).unwrap();

    let limits = crate::ContainerReadLimits {
        max_in_flight_bytes: 0,
        ..crate::ContainerReadLimits::default()
    };
    let budget = crate::ContainerReadBudget::new(limits);
    assert!(matches!(
        MmapV2DirectoryView::open_with_budget(&path, budget.clone()),
        Err(ContainerError::ResourceLimit {
            resource: "in-flight bytes",
            ..
        })
    ));
    assert_eq!(budget.in_flight_bytes(), 0);
    assert!(matches!(
        ContainerReader::from_mmap_file_with_budget(&path, budget.clone()),
        Err(ContainerError::ResourceLimit {
            resource: "in-flight bytes",
            ..
        })
    ));
    assert_eq!(budget.in_flight_bytes(), 0);

    let limits = crate::ContainerReadLimits {
        max_metadata_bytes: 0,
        ..crate::ContainerReadLimits::default()
    };
    let budget = crate::ContainerReadBudget::new(limits);
    assert!(matches!(
        MmapV2DirectoryView::open_with_budget(&path, budget.clone()),
        Err(ContainerError::ResourceLimit {
            resource: "metadata bytes",
            ..
        })
    ));
    assert_eq!(budget.in_flight_bytes(), 0);
    fs::remove_file(path).unwrap();
}

#[test]
fn range_mount_reads_header_toc_and_payload_in_bounded_requests() {
    let payload = vec![0xA5; 256 * 1024];
    let bytes = build(&[("large-range.asset", &payload)]);
    let total_len = bytes.len();
    let (source, requests) = TrackingRangeSource::new(bytes);
    let source = Arc::new(source);
    let reader = ContainerReader::from_range_source(Arc::clone(&source)).unwrap();
    let mount_requests = requests.lock().unwrap().clone();

    assert!(!mount_requests.is_empty());
    assert!(mount_requests
        .iter()
        .all(|(_, length)| *length <= 64 * 1024));
    assert!(mount_requests.iter().all(|(_, length)| *length < total_len));
    assert!(mount_requests
        .iter()
        .any(|(offset, length)| *offset == 0 && *length == 96));
    assert!(mount_requests
        .iter()
        .any(|(_, length)| *length == 64 * 1024));

    let entry = reader.entry(guid("large-range.asset")).unwrap();
    let mut output = [0_u8; 17];
    reader
        .read_range(guid("large-range.asset"), 123, &mut output)
        .unwrap();
    assert_eq!(output, [0xA5; 17]);
    let requests = requests.lock().unwrap();
    assert_eq!(requests.last(), Some(&(entry.offset + 123, 17)));
}

#[test]
fn file_range_source_mounts_and_supports_concurrent_payload_ranges() {
    let payload: Vec<u8> = (0..(128 * 1024)).map(|index| (index % 251) as u8).collect();
    let bytes = build(&[("file-range.asset", &payload)]);
    let path = temporary_container_path("concurrent");
    fs::write(&path, &bytes).unwrap();

    let source = Arc::new(FileRangeSource::open(&path).unwrap());
    let reader = Arc::new(ContainerReader::from_range_source(source).unwrap());
    let asset = guid("file-range.asset");
    let handles: Vec<_> = (0..8)
        .map(|thread_index| {
            let reader = Arc::clone(&reader);
            thread::spawn(move || {
                let offset = (thread_index * 4096) as u64;
                let mut output = vec![0_u8; 4096];
                reader.read_range(asset, offset, &mut output).unwrap();
                output
            })
        })
        .collect();

    for (thread_index, handle) in handles.into_iter().enumerate() {
        let output = handle.join().unwrap();
        let start = thread_index * 4096;
        assert_eq!(output, payload[start..start + 4096]);
    }
    fs::remove_file(path).unwrap();
}

#[test]
fn read_all_rejects_payload_mutation_after_mount() {
    let bytes = build(&[("mutated-after-mount.asset", b"original payload")]);
    let path = temporary_container_path("mutated-after-mount");
    fs::write(&path, &bytes).unwrap();

    let reader =
        ContainerReader::from_range_source(Arc::new(FileRangeSource::open(&path).unwrap()))
            .unwrap();
    let entry = reader.entry(guid("mutated-after-mount.asset")).unwrap();
    let mut file = OpenOptions::new().write(true).open(&path).unwrap();
    file.seek(SeekFrom::Start(entry.offset)).unwrap();
    file.write_all(b"tampered payload").unwrap();
    file.sync_all().unwrap();

    assert_eq!(
        reader.read_all(entry.guid),
        Err(ContainerError::HashMismatch)
    );
    fs::remove_file(path).unwrap();
}

#[test]
fn range_backend_reports_short_read_truncation_and_backend_bounds() {
    let bytes = build(&[("range-errors.asset", b"payload")]);
    let short = Arc::new(ShortReadSource {
        bytes: Arc::from(bytes.clone()),
        maximum_read: 8,
    });
    assert!(matches!(
        ContainerReader::from_range_source(short),
        Err(ContainerError::Source(error)) if error.contains("short read")
    ));

    let truncated = Arc::new(MemoryRangeSource::new(Arc::<[u8]>::from(
        bytes[..bytes.len() - 1].to_vec(),
    )));
    assert_eq!(
        ContainerReader::from_range_source(truncated).unwrap_err(),
        ContainerError::Truncated
    );

    let declared = Arc::new(DeclaredLengthSource {
        bytes: Arc::from(bytes[..bytes.len() - 1].to_vec()),
        declared_len: bytes.len() as u64,
    });
    assert!(matches!(
        ContainerReader::from_range_source(declared),
        Err(ContainerError::Source(error)) if error.contains("out of bounds")
    ));
}

#[test]
fn file_range_source_rejects_out_of_bounds_and_detects_external_short_read() {
    let path = temporary_container_path("short-read");
    let mut file = OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(&path)
        .unwrap();
    file.write_all(b"12345678").unwrap();
    file.sync_all().unwrap();
    let source = FileRangeSource::from_file(file.try_clone().unwrap()).unwrap();

    let mut output = [0_u8; 1];
    assert_eq!(
        source.read_range(8, &mut output),
        Err(RangeSourceError::OutOfBounds)
    );
    file.set_len(3).unwrap();
    let mut output = [0_u8; 8];
    assert_eq!(
        source.read_range(0, &mut output),
        Err(RangeSourceError::ShortRead {
            expected: 8,
            actual: 3,
        })
    );
    drop(file);
    fs::remove_file(path).unwrap();
}

#[test]
fn range_mount_rejects_payload_tampering_before_exposing_reader() {
    let mut bytes = build(&[("tampered-range.asset", b"trusted")]);
    let payload_offset = u64::from_le_bytes(bytes[36..44].try_into().unwrap()) as usize;
    bytes[payload_offset] ^= 0xFF;
    let source = Arc::new(MemoryRangeSource::new(Arc::<[u8]>::from(bytes)));
    assert_eq!(
        ContainerReader::from_range_source(source).unwrap_err(),
        ContainerError::HashMismatch
    );
}

fn update_directory_hash(bytes: &mut [u8]) {
    let directory_start = u64::from_le_bytes(bytes[20..28].try_into().unwrap()) as usize;
    let directory_len = u64::from_le_bytes(bytes[28..36].try_into().unwrap()) as usize;
    let hash = blake3::hash(&bytes[directory_start..directory_start + directory_len]);
    bytes[52..84].copy_from_slice(hash.as_bytes());
}

#[test]
fn writer_is_deterministic_and_canonical_across_insertion_orders() {
    let mut first = ContainerWriter::new();
    first.add(guid("z.asset"), b"zed".to_vec()).unwrap();
    first.add(guid("a.asset"), b"alpha".to_vec()).unwrap();
    let mut second = ContainerWriter::new();
    second.add(guid("a.asset"), b"alpha".to_vec()).unwrap();
    second.add(guid("z.asset"), b"zed".to_vec()).unwrap();
    let first = first.finish().unwrap();
    let second = second.finish().unwrap();
    assert_eq!(first, second);
    assert_eq!(&first[..8], &MAGIC);
    assert_eq!(
        u16::from_le_bytes(first[8..10].try_into().unwrap()),
        FORMAT_VERSION
    );
    let reader = ContainerReader::new(first).unwrap();
    assert_eq!(reader.info().entry_count, 2);
    assert_eq!(reader.read_all(guid("a.asset")).unwrap(), b"alpha");
    assert_eq!(reader.read_all(guid("z.asset")).unwrap(), b"zed");
}

#[test]
fn writer_rejects_duplicate_guid_without_replacing_payload() {
    let id = guid("same.asset");
    let mut writer = ContainerWriter::new();
    writer.add(id, b"first".to_vec()).unwrap();
    assert_eq!(
        writer.add(id, b"second".to_vec()),
        Err(ContainerError::DuplicateGuid(id))
    );
    let reader = ContainerReader::new(writer.finish().unwrap()).unwrap();
    assert_eq!(reader.read_all(id).unwrap(), b"first");
}

#[test]
fn reader_handles_empty_payload_and_checks_ranges_without_overflow() {
    let bytes = build(&[("empty", b""), ("data", b"1234")]);
    let reader = ContainerReader::new(bytes).unwrap();
    assert_eq!(reader.read_all(guid("empty")).unwrap(), b"");
    let mut range = [0; 2];
    reader.read_range(guid("data"), 1, &mut range).unwrap();
    assert_eq!(&range, b"23");
    assert_eq!(
        reader.read_range(guid("data"), u64::MAX, &mut range),
        Err(ContainerError::IntegerOverflow)
    );
    assert_eq!(
        reader.read_range(guid("data"), 3, &mut range),
        Err(ContainerError::InvalidRange)
    );
    assert_eq!(
        reader.read_range(guid("absent"), 0, &mut range),
        Err(ContainerError::MissingAsset(guid("absent")))
    );
}

#[test]
fn reader_batches_overlapping_and_nearby_ranges_in_caller_order() {
    let bytes = build(&[
        ("first", b"abcdefgh"),
        ("second", b"ijklmnop"),
        ("far", b"qrstuvwx"),
    ]);
    let (source, requests) = TrackingRangeSource::new(bytes);
    let reader = ContainerReader::from_range_source(Arc::new(source)).unwrap();
    requests.lock().unwrap().clear();

    let output = reader
        .read_ranges(
            &[
                RangeRequest::new(guid("second"), 2, 3),
                RangeRequest::new(guid("first"), 1, 4),
                RangeRequest::new(guid("second"), 0, 2),
            ],
            RangeBatchPolicy {
                max_batch_bytes: 64,
                max_gap_bytes: 4096,
            },
        )
        .unwrap();
    assert_eq!(
        output,
        vec![b"klm".to_vec(), b"bcde".to_vec(), b"ij".to_vec()]
    );

    let physical_reads = requests.lock().unwrap().clone();
    assert_eq!(physical_reads.len(), 1);
    let first = reader.entry(guid("first")).unwrap();
    let second = reader.entry(guid("second")).unwrap();
    let expected_start = (first.offset + 1).min(second.offset);
    let expected_end = (first.offset + 5).max(second.offset + 5);
    assert_eq!(
        physical_reads[0],
        (expected_start, (expected_end - expected_start) as usize)
    );
}

#[test]
fn reader_batches_duplicate_requests_without_reordering_results() {
    let bytes = build(&[("duplicate", b"duplicate payload")]);
    let (source, requests) = TrackingRangeSource::new(bytes);
    let reader = ContainerReader::from_range_source(Arc::new(source)).unwrap();
    requests.lock().unwrap().clear();

    let request = RangeRequest::new(guid("duplicate"), 6, 6);
    let output = reader
        .read_ranges(
            &[request, RangeRequest::new(guid("duplicate"), 3, 8), request],
            RangeBatchPolicy {
                max_batch_bytes: 32,
                max_gap_bytes: 1,
            },
        )
        .unwrap();
    assert_eq!(
        output,
        vec![b"ate pa".to_vec(), b"licate p".to_vec(), b"ate pa".to_vec()]
    );
    assert_eq!(requests.lock().unwrap().len(), 1);
}

#[test]
fn reader_batches_ranges_within_the_configured_gap() {
    let payload: Vec<u8> = (0..32).map(|value| value as u8).collect();
    let bytes = build(&[("near-gap", &payload)]);
    let (source, requests) = TrackingRangeSource::new(bytes);
    let reader = ContainerReader::from_range_source(Arc::new(source)).unwrap();
    requests.lock().unwrap().clear();

    let output = reader
        .read_ranges(
            &[
                RangeRequest::new(guid("near-gap"), 8, 4),
                RangeRequest::new(guid("near-gap"), 0, 4),
            ],
            RangeBatchPolicy {
                max_batch_bytes: 12,
                max_gap_bytes: 4,
            },
        )
        .unwrap();
    assert_eq!(
        output,
        vec![payload[8..12].to_vec(), payload[0..4].to_vec()]
    );

    let entry = reader.entry(guid("near-gap")).unwrap();
    assert_eq!(requests.lock().unwrap().as_slice(), &[(entry.offset, 12)]);
}

#[test]
fn reader_batching_never_reads_past_the_source_end() {
    let payload: Vec<u8> = (0..32).map(|value| value as u8).collect();
    let bytes = build(&[("source-end", &payload)]);
    let source_len = bytes.len() as u64;
    let (source, requests) = TrackingRangeSource::new(bytes);
    let reader = ContainerReader::from_range_source(Arc::new(source)).unwrap();
    requests.lock().unwrap().clear();

    reader
        .read_ranges(
            &[
                RangeRequest::new(guid("source-end"), 0, 4),
                RangeRequest::new(guid("source-end"), 28, 4),
            ],
            RangeBatchPolicy {
                max_batch_bytes: 32,
                max_gap_bytes: 24,
            },
        )
        .unwrap();

    let entry = reader.entry(guid("source-end")).unwrap();
    let physical = requests.lock().unwrap().clone();
    assert_eq!(physical, vec![(entry.offset, 32)]);
    assert!(physical.iter().all(|(offset, length)| {
        offset
            .checked_add(*length as u64)
            .is_some_and(|end| end <= source_len)
    }));
}

#[test]
fn reader_batches_unsorted_overlaps_and_preserves_duplicate_results() {
    let payload: Vec<u8> = (0..64).map(|value| b'A' + value).collect();
    let bytes = build(&[("unsorted-overlap", &payload)]);
    let (source, requests) = TrackingRangeSource::new(bytes);
    let reader = ContainerReader::from_range_source(Arc::new(source)).unwrap();
    requests.lock().unwrap().clear();

    let output = reader
        .read_ranges(
            &[
                RangeRequest::new(guid("unsorted-overlap"), 20, 5),
                RangeRequest::new(guid("unsorted-overlap"), 10, 15),
                RangeRequest::new(guid("unsorted-overlap"), 20, 5),
                RangeRequest::new(guid("unsorted-overlap"), 12, 4),
            ],
            RangeBatchPolicy {
                max_batch_bytes: 32,
                max_gap_bytes: 0,
            },
        )
        .unwrap();
    assert_eq!(output[0], payload[20..25]);
    assert_eq!(output[1], payload[10..25]);
    assert_eq!(output[2], payload[20..25]);
    assert_eq!(output[3], payload[12..16]);
    let entry = reader.entry(guid("unsorted-overlap")).unwrap();
    assert_eq!(
        requests.lock().unwrap().as_slice(),
        &[(entry.offset + 10, 15)]
    );
}

#[test]
fn reader_batching_rejects_overflow_before_source_io() {
    let bytes = build(&[("batch-overflow", b"payload")]);
    let (source, requests) = TrackingRangeSource::new(bytes);
    let reader = ContainerReader::from_range_source(Arc::new(source)).unwrap();
    requests.lock().unwrap().clear();

    assert_eq!(
        reader.read_ranges(
            &[RangeRequest::new(guid("batch-overflow"), u64::MAX, 1)],
            RangeBatchPolicy::default(),
        ),
        Err(ContainerError::IntegerOverflow)
    );
    assert!(requests.lock().unwrap().is_empty());
}

#[test]
fn reader_batching_propagates_source_failure_without_partial_success() {
    let bytes = build(&[("batch-failure", b"payload")]);
    let fail_reads = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let source = ToggleFailureSource {
        bytes: Arc::from(bytes),
        fail_reads: Arc::clone(&fail_reads),
    };
    let reader = ContainerReader::from_range_source(Arc::new(source)).unwrap();
    fail_reads.store(true, std::sync::atomic::Ordering::Relaxed);

    assert!(matches!(
        reader.read_ranges(
            &[RangeRequest::new(guid("batch-failure"), 0, 4)],
            RangeBatchPolicy::default(),
        ),
        Err(ContainerError::Source(error))
            if error == "range source I/O failed: injected range failure"
    ));
}

#[test]
fn reader_batches_never_exceed_the_maximum_physical_read() {
    let payload: Vec<u8> = (0..64).map(|value| value as u8).collect();
    let bytes = build(&[("batch-bound", &payload)]);
    let (source, requests) = TrackingRangeSource::new(bytes);
    let reader = ContainerReader::from_range_source(Arc::new(source)).unwrap();
    requests.lock().unwrap().clear();

    reader
        .read_ranges(
            &[
                RangeRequest::new(guid("batch-bound"), 0, 8),
                RangeRequest::new(guid("batch-bound"), 12, 8),
                RangeRequest::new(guid("batch-bound"), 40, 8),
            ],
            RangeBatchPolicy {
                max_batch_bytes: 20,
                max_gap_bytes: 4,
            },
        )
        .unwrap();

    let physical = requests.lock().unwrap().clone();
    assert_eq!(physical.len(), 2);
    assert!(physical.iter().all(|(_, length)| *length <= 20));
}

#[test]
fn reader_batching_rejects_offset_overflow_before_source_io() {
    let bytes = build(&[("batch-overflow", b"payload")]);
    let (source, requests) = TrackingRangeSource::new(bytes);
    let reader = ContainerReader::from_range_source(Arc::new(source)).unwrap();
    requests.lock().unwrap().clear();

    assert_eq!(
        reader.read_ranges(
            &[RangeRequest::new(guid("batch-overflow"), u64::MAX, 1)],
            RangeBatchPolicy::default(),
        ),
        Err(ContainerError::IntegerOverflow)
    );
    assert!(requests.lock().unwrap().is_empty());
}

#[test]
fn manifest_batching_rejects_overflow_and_bounds_policy_before_io() {
    let logical = guid("logical-policy");
    let local = guid("local-policy");
    let bytes = build(&[("local-policy", b"0123456789")]);
    let mut overlays = OverlaySet::new();
    overlays.mount(
        "container-policy",
        0,
        ContainerReader::new(bytes.clone()).unwrap(),
    );
    let mut manifest = RuntimeManifest::new();
    manifest
        .insert(ManifestRoute {
            guid: logical,
            container_name: "container-policy".into(),
            priority: 0,
            container_guid: local,
        })
        .unwrap();
    manifest
        .bind_container("container-policy", identity(&bytes))
        .unwrap();
    let reader = manifest.bind(&overlays).unwrap();

    assert_eq!(
        reader.read_ranges(
            &[ManifestRangeRequest::new(logical, 0, 5)],
            RangeBatchPolicy {
                max_batch_bytes: 4,
                max_gap_bytes: 0,
            },
        ),
        Err(ManifestReadError::Container(
            ContainerError::InvalidBatchPolicy
        ))
    );
    assert!(matches!(
        reader.read_ranges(
            &[ManifestRangeRequest::new(logical, u64::MAX, 1)],
            RangeBatchPolicy::default(),
        ),
        Err(ManifestReadError::InvalidRange { .. })
    ));
}

#[test]
fn reader_batch_policy_bounds_coalescing_and_validates_before_io() {
    let bytes = build(&[("one", b"1234"), ("two", b"5678")]);
    let (source, requests) = TrackingRangeSource::new(bytes);
    let reader = ContainerReader::from_range_source(Arc::new(source)).unwrap();
    requests.lock().unwrap().clear();

    let output = reader
        .read_ranges(
            &[
                RangeRequest::new(guid("one"), 0, 4),
                RangeRequest::new(guid("two"), 0, 4),
            ],
            RangeBatchPolicy {
                max_batch_bytes: 4,
                max_gap_bytes: 4096,
            },
        )
        .unwrap();
    assert_eq!(output, vec![b"1234".to_vec(), b"5678".to_vec()]);
    assert_eq!(requests.lock().unwrap().len(), 2);

    let before = requests.lock().unwrap().len();
    assert_eq!(
        reader.read_ranges(
            &[RangeRequest::new(guid("one"), 3, 2)],
            RangeBatchPolicy::default(),
        ),
        Err(ContainerError::InvalidRange)
    );
    assert_eq!(requests.lock().unwrap().len(), before);
}

#[test]
fn reader_batch_policy_rejects_single_request_over_limit_before_io() {
    let payload = b"12345678";
    let bytes = build(&[("single-request-bound", payload)]);
    let (source, requests) = TrackingRangeSource::new(bytes);
    let reader = ContainerReader::from_range_source(Arc::new(source)).unwrap();
    requests.lock().unwrap().clear();

    let exact = reader
        .read_ranges(
            &[RangeRequest::new(guid("single-request-bound"), 0, 4)],
            RangeBatchPolicy {
                max_batch_bytes: 4,
                max_gap_bytes: 0,
            },
        )
        .unwrap();
    assert_eq!(exact, vec![b"1234".to_vec()]);
    assert_eq!(requests.lock().unwrap().as_slice().len(), 1);
    assert_eq!(requests.lock().unwrap().as_slice()[0].1, 4);

    requests.lock().unwrap().clear();
    assert_eq!(
        reader.read_ranges(
            &[RangeRequest::new(guid("single-request-bound"), 0, 5)],
            RangeBatchPolicy {
                max_batch_bytes: 4,
                max_gap_bytes: 0,
            },
        ),
        Err(ContainerError::InvalidBatchPolicy)
    );
    assert!(requests.lock().unwrap().is_empty());
}

#[test]
fn rejects_truncation_version_and_directory_digest_corruption() {
    let canonical = build(&[("one", b"one")]);
    assert_eq!(
        ContainerReader::new(canonical[..canonical.len() - 1].to_vec()).unwrap_err(),
        ContainerError::Truncated
    );

    let mut version = canonical.clone();
    version[8..10].copy_from_slice(&99_u16.to_le_bytes());
    assert_eq!(
        ContainerReader::new(version).unwrap_err(),
        ContainerError::UnsupportedVersion(99)
    );

    let mut directory = canonical;
    directory[96] ^= 1;
    assert_eq!(
        ContainerReader::new(directory).unwrap_err(),
        ContainerError::HashMismatch
    );
}

#[test]
fn rejects_payload_hash_mismatch_and_does_not_modify_output() {
    let mut bytes = build(&[("payload", b"hello")]);
    let payload_offset = u64::from_le_bytes(bytes[36..44].try_into().unwrap()) as usize;
    bytes[payload_offset] ^= 0x20;
    assert_eq!(
        ContainerReader::new(bytes).unwrap_err(),
        ContainerError::HashMismatch
    );
}

#[test]
fn rejects_noncanonical_or_invalid_toc_records() {
    // Reserved record bytes must be zero, even when the directory digest matches.
    let mut reserved = build(&[("item", b"x")]);
    reserved[96 + 76..96 + 80].copy_from_slice(&3_u32.to_le_bytes());
    update_directory_hash(&mut reserved);
    assert_eq!(
        ContainerReader::new(reserved).unwrap_err(),
        ContainerError::InvalidAlignment
    );

    // A payload offset that creates a gap is not accepted as a valid TOC.
    let mut gap = build(&[("item", b"x")]);
    let old = u64::from_le_bytes(gap[96 + 16..96 + 24].try_into().unwrap());
    gap[96 + 16..96 + 24].copy_from_slice(&(old + 1).to_le_bytes());
    update_directory_hash(&mut gap);
    assert_eq!(
        ContainerReader::new(gap).unwrap_err(),
        ContainerError::OutOfBounds
    );
}

#[test]
fn aligned_payloads_preserve_absolute_offsets_and_zero_padding() {
    let mut writer = ContainerWriter::new();
    writer
        .add_aligned(guid("unaligned"), b"a".to_vec(), 1)
        .unwrap();
    writer
        .add_aligned(guid("aligned"), b"payload".to_vec(), 4096)
        .unwrap();
    let bytes = writer.finish().unwrap();
    let reader = ContainerReader::new(bytes).unwrap();
    let entry = reader.entry(guid("aligned")).unwrap();
    assert_eq!(entry.alignment, 4096);
    assert_eq!(entry.offset % 4096, 0);
    assert_eq!(reader.read_all(guid("aligned")).unwrap(), b"payload");
}

#[test]
fn planned_layout_is_consumed_and_round_trips_in_version_one() {
    let first = guid("planned-first");
    let second = guid("planned-second");
    let policy = ChunkPolicy {
        max_chunk_size: 16,
        chunk_alignment: 8,
        max_chunks: 2,
    };
    let plan = ChunkPlanner::new(policy)
        .plan([
            ChunkInput {
                guid: first,
                size: 3,
                alignment: 1,
                dependencies: vec![],
                priority: 0,
            },
            ChunkInput {
                guid: second,
                size: 4,
                alignment: 4,
                dependencies: vec![],
                priority: 9,
            },
        ])
        .unwrap();
    assert_eq!(plan.chunks[0].placements[0].guid, second);

    let mut writer = ContainerWriter::new();
    writer.add_aligned(first, b"one".to_vec(), 1).unwrap();
    writer.add_aligned(second, b"four".to_vec(), 4).unwrap();
    let bytes = writer
        .finish_with_chunk_plan(&plan, policy.chunk_alignment)
        .unwrap();
    assert_eq!(
        u16::from_le_bytes(bytes[8..10].try_into().unwrap()),
        FORMAT_VERSION
    );
    let reader = ContainerReader::new(bytes).unwrap();
    let first_entry = reader.entry(first).unwrap();
    let second_entry = reader.entry(second).unwrap();
    assert_eq!(second_entry.offset % 4, 0);
    assert_eq!(first_entry.offset, second_entry.offset + 8);
    assert_eq!(reader.read_all(first).unwrap(), b"one");
    assert_eq!(reader.read_all(second).unwrap(), b"four");
}

#[test]
fn planned_layout_honors_payload_alignment_above_chunk_alignment() {
    let id = guid("high-alignment");
    let policy = ChunkPolicy {
        max_chunk_size: 16 * 1024,
        chunk_alignment: 4096,
        max_chunks: 1,
    };
    let plan = ChunkPlanner::new(policy)
        .plan([ChunkInput {
            guid: id,
            size: 3,
            alignment: 8192,
            dependencies: vec![],
            priority: 0,
        }])
        .unwrap();
    assert_eq!(plan.chunks[0].start_alignment, 8192);

    let mut writer = ContainerWriter::new();
    writer.add_aligned(id, b"abc".to_vec(), 8192).unwrap();
    let reader = ContainerReader::new(
        writer
            .finish_with_chunk_plan(&plan, policy.chunk_alignment)
            .unwrap(),
    )
    .unwrap();
    let entry = reader.entry(id).unwrap();
    assert_eq!(entry.offset % 8192, 0);
    assert_eq!(reader.read_all(id).unwrap(), b"abc");
}

#[test]
fn planned_layout_keeps_cross_chunk_dependencies_before_dependents() {
    let root = StableGuid::from_u128(1);
    let dependency = StableGuid::from_u128(2);
    let dependent = StableGuid::from_u128(3);
    let policy = ChunkPolicy {
        max_chunk_size: 8,
        chunk_alignment: 1,
        max_chunks: 2,
    };
    let plan = ChunkPlanner::new(policy)
        .plan([
            ChunkInput {
                guid: root,
                size: 5,
                alignment: 1,
                dependencies: vec![],
                priority: 0,
            },
            ChunkInput {
                guid: dependency,
                size: 4,
                alignment: 4,
                dependencies: vec![root],
                priority: 0,
            },
            ChunkInput {
                guid: dependent,
                size: 1,
                alignment: 1,
                dependencies: vec![dependency],
                priority: 0,
            },
        ])
        .unwrap();

    assert_eq!(plan.assignments[&root], 0);
    assert_eq!(plan.assignments[&dependency], 1);
    assert_eq!(plan.assignments[&dependent], 1);
    assert_eq!(
        plan.chunks[1]
            .placements
            .iter()
            .map(|placement| placement.guid)
            .collect::<Vec<_>>(),
        vec![dependency, dependent]
    );
    assert!(
        plan.chunks[plan.assignments[&dependency]].offset
            <= plan.chunks[plan.assignments[&dependent]].offset
    );
}

#[test]
fn planned_writer_rejects_a_plan_that_does_not_cover_exactly_its_payloads() {
    let id = guid("planned-only");
    let plan = ChunkPlanner::new(ChunkPolicy {
        max_chunk_size: 8,
        chunk_alignment: 4,
        max_chunks: 1,
    })
    .plan([ChunkInput {
        guid: id,
        size: 1,
        alignment: 1,
        dependencies: vec![],
        priority: 0,
    }])
    .unwrap();
    let mut writer = ContainerWriter::new();
    writer.add(id, b"different-size".to_vec()).unwrap();
    assert_eq!(
        writer.finish_with_chunk_plan(&plan, 4),
        Err(ContainerError::InvalidDirectory)
    );
}

#[test]
fn planned_writer_rejects_noncanonical_placement_and_chunk_extents() {
    let id = guid("planned-canonical-extent");
    let policy = ChunkPolicy {
        max_chunk_size: 16,
        chunk_alignment: 8,
        max_chunks: 1,
    };
    let canonical = ChunkPlanner::new(policy)
        .plan([ChunkInput {
            guid: id,
            size: 1,
            alignment: 1,
            dependencies: vec![],
            priority: 0,
        }])
        .unwrap();

    // A forged plan must not be able to force arbitrary zero-filled extents
    // around one byte of actual payload.
    let mut oversized_extent = canonical.clone();
    oversized_extent.chunks[0].placements[0].offset = 1024;
    oversized_extent.chunks[0].raw_size = 1025;
    let mut writer = ContainerWriter::new();
    writer.add(id, b"x".to_vec()).unwrap();
    assert_eq!(
        writer
            .finish_with_chunk_plan(&oversized_extent, policy.chunk_alignment)
            .unwrap_err(),
        ContainerError::InvalidDirectory
    );

    // Even a small hole is noncanonical: placements are packed at the first
    // offset satisfying the effective alignment, and raw_size ends at the
    // final placement rather than including attacker-chosen trailing bytes.
    let mut noncanonical_hole = canonical;
    noncanonical_hole.chunks[0].placements[0].offset = 8;
    noncanonical_hole.chunks[0].raw_size = 9;
    let mut writer = ContainerWriter::new();
    writer.add(id, b"x".to_vec()).unwrap();
    assert_eq!(
        writer
            .finish_with_chunk_plan(&noncanonical_hole, policy.chunk_alignment)
            .unwrap_err(),
        ContainerError::InvalidDirectory
    );
}

#[test]
fn planned_writer_rejects_noncanonical_chunk_gaps() {
    let first = guid("planned-gap-first");
    let second = guid("planned-gap-second");
    let policy = ChunkPolicy {
        max_chunk_size: 1,
        chunk_alignment: 8,
        max_chunks: 2,
    };
    let mut plan = ChunkPlanner::new(policy)
        .plan([first, second].map(|guid| ChunkInput {
            guid,
            size: 1,
            alignment: 1,
            dependencies: vec![],
            priority: 0,
        }))
        .unwrap();
    plan.chunks[1].offset += 8;

    let mut writer = ContainerWriter::new();
    writer.add(first, b"a".to_vec()).unwrap();
    writer.add(second, b"b".to_vec()).unwrap();
    assert_eq!(
        writer
            .finish_with_chunk_plan(&plan, policy.chunk_alignment)
            .unwrap_err(),
        ContainerError::InvalidDirectory
    );
}

#[test]
fn version_one_reader_rejects_nonzero_directory_alignment_padding() {
    let id = guid("padding-check");
    let plan = ChunkPlanner::new(ChunkPolicy {
        max_chunk_size: 8,
        chunk_alignment: 64,
        max_chunks: 1,
    })
    .plan([ChunkInput {
        guid: id,
        size: 1,
        alignment: 1,
        dependencies: vec![],
        priority: 0,
    }])
    .unwrap();
    let mut writer = ContainerWriter::new();
    writer.add(id, b"x".to_vec()).unwrap();
    let mut bytes = writer.finish_with_chunk_plan(&plan, 64).unwrap();
    let payload_start = u64::from_le_bytes(bytes[36..44].try_into().unwrap()) as usize;
    assert!(payload_start > 176);
    bytes[176] = 1;
    assert_eq!(
        ContainerReader::new(bytes).unwrap_err(),
        ContainerError::InvalidHeader
    );
}

#[test]
fn version_two_chunk_table_round_trips_and_is_a_closed_set() {
    assert_eq!(
        SUPPORTED_FORMAT_VERSIONS,
        &[FORMAT_VERSION, CHUNK_FORMAT_VERSION]
    );
    let first = guid("chunk-table-a");
    let second = guid("chunk-table-b");
    let policy = ChunkPolicy {
        max_chunk_size: 512,
        chunk_alignment: 64,
        max_chunks: 4,
    };
    let plan = ChunkPlanner::new(policy)
        .plan([
            ChunkInput {
                guid: first,
                size: 300,
                alignment: 1,
                dependencies: vec![],
                priority: 1,
            },
            ChunkInput {
                guid: second,
                size: 500,
                alignment: 1,
                dependencies: vec![],
                priority: 0,
            },
        ])
        .unwrap();
    let mut writer = ContainerWriter::new();
    let first_bytes = b"A".repeat(300);
    let second_bytes = b"B".repeat(500);
    writer.add(first, first_bytes.clone()).unwrap();
    writer.add(second, second_bytes.clone()).unwrap();
    let bytes = writer
        .finish_with_chunk_compression(&plan, policy.chunk_alignment, Codec::Deflate)
        .unwrap();
    let reader = ContainerReader::new(bytes.clone()).unwrap();
    assert_eq!(reader.info().version, CHUNK_FORMAT_VERSION);
    assert_eq!(reader.info().chunk_count, plan.chunks.len() as u32);
    assert_eq!(reader.read_all(first).unwrap(), first_bytes);
    assert_eq!(reader.read_all(second).unwrap(), second_bytes);
    let mut range = vec![0; 33];
    reader.read_range(second, 101, &mut range).unwrap();
    assert_eq!(range, second_bytes[101..134]);
}

#[test]
fn version_two_relocates_expanded_chunks_without_changing_raw_placements() {
    for codec in [Codec::Raw, Codec::Deflate, Codec::BlockDeflate] {
        for alignments in [[64, 64, 64], [64, 256, 128]] {
            let ids = [guid("expanded-a"), guid("expanded-b"), guid("expanded-c")];
            let payload_len = *alignments.iter().max().unwrap();
            let payloads: Vec<Vec<u8>> = (0..3)
                .map(|index| {
                    let mut state = 0x9e37_79b9_u32.wrapping_add(index);
                    (0..payload_len)
                        .map(|_| {
                            state ^= state << 13;
                            state ^= state >> 17;
                            state ^= state << 5;
                            state as u8
                        })
                        .collect()
                })
                .collect();
            let policy = ChunkPolicy {
                max_chunk_size: payload_len,
                chunk_alignment: 64,
                max_chunks: 3,
            };
            let plan = ChunkPlanner::new(policy)
                .plan((0..3).map(|index| ChunkInput {
                    guid: ids[index],
                    size: payloads[index].len() as u64,
                    alignment: alignments[index],
                    dependencies: vec![],
                    priority: (3 - index) as u32,
                }))
                .unwrap();
            assert_eq!(plan.chunks.len(), 3);
            let encode = |plan: &crate::ChunkPlan| {
                let mut writer = ContainerWriter::new();
                for index in 0..3 {
                    writer
                        .add_aligned(
                            ids[index],
                            payloads[index].clone(),
                            alignments[index] as u32,
                        )
                        .unwrap();
                }
                writer.finish_with_chunk_compression(plan, policy.chunk_alignment, codec)
            };
            let bytes = encode(&plan).unwrap();
            assert_eq!(bytes, encode(&plan).unwrap());
            let total_len = bytes.len() as u64;
            let source = CappedRangeSource {
                bytes: Arc::from(bytes.clone()),
                maximum_read: 17,
                requests: Arc::new(Mutex::new(Vec::new())),
                fail_after: Mutex::new(None),
            };
            let requests = Arc::clone(&source.requests);
            let reader = ContainerReader::from_range_source(Arc::new(source)).unwrap();
            let memory_reader = ContainerReader::new(bytes).unwrap();
            let mut end = 0;
            for index in 0..3 {
                let chunk = reader.chunk(index as u32).unwrap();
                assert!(chunk.offset >= end);
                assert_eq!(chunk.offset % alignments[index], 0);
                if codec != Codec::Raw {
                    assert!(chunk.stored_len > chunk.raw_len);
                }
                end = chunk.offset + chunk.stored_len.max(chunk.raw_len);
                let entry = reader.entry(ids[index]).unwrap();
                assert_eq!(entry.chunk_id, plan.assignments[&ids[index]] as u32);
                assert_eq!(entry.raw_offset, plan.chunks[index].placements[0].offset);
                assert_eq!(reader.read_all(ids[index]).unwrap(), payloads[index]);
                assert_eq!(memory_reader.read_all(ids[index]).unwrap(), payloads[index]);
                requests.lock().unwrap().clear();
                let mut range = [0; 19];
                reader.read_range(ids[index], 7, &mut range).unwrap();
                assert_eq!(range.as_slice(), &payloads[index][7..26]);
                for &(offset, len) in requests.lock().unwrap().iter() {
                    assert!(len <= 17);
                    assert!(offset >= chunk.offset);
                    assert!(offset + len as u64 <= chunk.offset + chunk.stored_len);
                }
            }
            assert_eq!(end, total_len);
            let mut invalid_plan = plan.clone();
            invalid_plan.chunks[1].offset += 64;
            assert_eq!(encode(&invalid_plan), Err(ContainerError::InvalidDirectory));
        }
    }
}

#[test]
fn version_two_raw_range_batching_uses_chunk_local_offsets() {
    let first = guid("chunk-range-first");
    let second = guid("chunk-range-second");
    let policy = ChunkPolicy {
        max_chunk_size: 64,
        chunk_alignment: 16,
        max_chunks: 1,
    };
    let plan = ChunkPlanner::new(policy)
        .plan([
            ChunkInput {
                guid: first,
                size: 5,
                alignment: 1,
                dependencies: vec![],
                priority: 0,
            },
            ChunkInput {
                guid: second,
                size: 7,
                alignment: 1,
                dependencies: vec![],
                priority: 0,
            },
        ])
        .unwrap();
    let mut writer = ContainerWriter::new();
    writer.add(first, b"first".to_vec()).unwrap();
    writer.add(second, b"second!".to_vec()).unwrap();
    let bytes = writer
        .finish_with_chunk_compression(&plan, policy.chunk_alignment, Codec::Raw)
        .unwrap();
    let (source, requests) = TrackingRangeSource::new(bytes);
    let reader = ContainerReader::from_range_source(Arc::new(source)).unwrap();
    let first_entry = reader.entry(first).unwrap();
    let second_entry = reader.entry(second).unwrap();
    assert_eq!(first_entry.chunk_id, second_entry.chunk_id);
    assert_ne!(first_entry.raw_offset, second_entry.raw_offset);
    requests.lock().unwrap().clear();

    let output = reader
        .read_ranges(
            &[
                RangeRequest::new(second, 1, 4),
                RangeRequest::new(first, 0, 5),
            ],
            RangeBatchPolicy {
                max_batch_bytes: 32,
                max_gap_bytes: 16,
            },
        )
        .unwrap();
    assert_eq!(output, vec![b"econd"[..4].to_vec(), b"first".to_vec()]);

    let first_start = first_entry
        .offset
        .checked_add(first_entry.raw_offset)
        .unwrap();
    let second_start = second_entry
        .offset
        .checked_add(second_entry.raw_offset)
        .unwrap();
    let expected_start = first_start.min(second_start);
    let expected_end = first_start
        .checked_add(first_entry.raw_len)
        .unwrap()
        .max(second_start.checked_add(5).unwrap());
    assert_eq!(
        requests.lock().unwrap().as_slice(),
        &[(
            expected_start,
            usize::try_from(expected_end - expected_start).unwrap()
        )]
    );
}

#[test]
fn version_two_rejects_chunk_payload_tampering() {
    let id = guid("chunk-table-corrupt");
    let policy = ChunkPolicy {
        max_chunk_size: 512,
        chunk_alignment: 64,
        max_chunks: 1,
    };
    let plan = ChunkPlanner::new(policy)
        .plan([ChunkInput {
            guid: id,
            size: 200,
            alignment: 1,
            dependencies: vec![],
            priority: 0,
        }])
        .unwrap();
    let mut writer = ContainerWriter::new();
    writer.add(id, vec![7; 200]).unwrap();
    let mut bytes = writer
        .finish_with_chunk_compression(&plan, policy.chunk_alignment, Codec::Deflate)
        .unwrap();
    let payload_offset = u64::from_le_bytes(bytes[56..64].try_into().unwrap()) as usize;
    bytes[payload_offset] ^= 0x80;
    assert!(matches!(
        ContainerReader::new(bytes),
        Err(ContainerError::CompressionFailed | ContainerError::HashMismatch)
    ));
}

#[test]
fn unknown_container_codec_is_a_structured_error() {
    let mut bytes = build(&[("unknown-codec", b"payload")]);
    bytes[96 + 40] = 0xff;
    update_directory_hash(&mut bytes);
    assert_eq!(
        ContainerReader::new(bytes).unwrap_err(),
        ContainerError::UnsupportedCodec(0xff)
    );
}

#[test]
fn chunk_plan_rejects_deflate_with_a_structured_layout_error() {
    let id = guid("compressed-planned");
    let plan = ChunkPlanner::new(ChunkPolicy::default())
        .plan([ChunkInput {
            guid: id,
            size: 1024,
            alignment: 1,
            dependencies: vec![],
            priority: 0,
        }])
        .unwrap();
    let mut writer = ContainerWriter::new();
    writer.add_deflate(id, vec![7; 1024]).unwrap();
    assert_eq!(
        writer.finish_with_chunk_plan(&plan, ChunkPolicy::default().chunk_alignment),
        Err(ContainerError::CodecNotSupportedForLayout(Codec::Deflate))
    );
}

#[test]
fn overlay_priority_ties_latest_mount_wins_and_unmount_falls_back() {
    let id = guid("shared.asset");
    let mut overlays = OverlaySet::new();
    overlays.mount(
        "base",
        0,
        ContainerReader::new(build(&[("shared.asset", b"base")])).unwrap(),
    );
    overlays.mount(
        "patch",
        1,
        ContainerReader::new(build(&[("shared.asset", b"patch")])).unwrap(),
    );
    overlays.mount(
        "same-priority-later",
        1,
        ContainerReader::new(build(&[("shared.asset", b"latest")])).unwrap(),
    );
    assert_eq!(overlays.source(id), Some("same-priority-later"));
    assert_eq!(overlays.read_all(id).unwrap(), b"latest");
    assert!(overlays.unmount("same-priority-later"));
    assert_eq!(overlays.read_all(id).unwrap(), b"patch");
    assert!(overlays.unmount("patch"));
    assert_eq!(overlays.read_all(id).unwrap(), b"base");
    assert!(!overlays.unmount("missing"));
}

#[test]
fn patch_tombstones_hide_lower_priority_assets_and_restore_on_unmount() {
    let id = guid("deleted-by-patch.asset");
    let mut overlays = OverlaySet::new();
    overlays.mount(
        "base",
        0,
        ContainerReader::new(build(&[("deleted-by-patch.asset", b"base")])).unwrap(),
    );
    overlays.mount_tombstones("patch", 1, [id]);
    assert!(!overlays.contains(id));
    assert_eq!(overlays.source(id), None);
    assert_eq!(overlays.read_all(id), Err(ContainerError::MissingAsset(id)));
    assert!(overlays.unmount("patch"));
    assert_eq!(overlays.read_all(id).unwrap(), b"base");
}

#[test]
fn later_higher_priority_mount_can_restore_a_tombstoned_asset() {
    let id = guid("restored-by-hotfix.asset");
    let mut overlays = OverlaySet::new();
    overlays.mount(
        "base",
        0,
        ContainerReader::new(build(&[("restored-by-hotfix.asset", b"base")])).unwrap(),
    );
    overlays.mount_tombstones("patch", 1, [id]);
    overlays.mount(
        "hotfix",
        2,
        ContainerReader::new(build(&[("restored-by-hotfix.asset", b"hotfix")])).unwrap(),
    );
    assert_eq!(overlays.source(id), Some("hotfix"));
    assert_eq!(overlays.read_all(id).unwrap(), b"hotfix");
}

#[test]
fn remounting_a_container_name_replaces_its_previous_guid_set() {
    let old_guid = guid("remount-old");
    let new_guid = guid("remount-new");
    let mut overlays = OverlaySet::new();
    overlays.mount(
        "shipping",
        1,
        ContainerReader::new(build(&[("remount-old", b"old")])).unwrap(),
    );
    overlays.mount(
        "shipping",
        1,
        ContainerReader::new(build(&[("remount-new", b"new")])).unwrap(),
    );
    assert!(!overlays.contains(old_guid));
    assert_eq!(overlays.read_all(new_guid).unwrap(), b"new");

    let mut manifest = RuntimeManifest::new();
    manifest
        .insert(ManifestRoute {
            guid: old_guid,
            container_name: "shipping".into(),
            priority: 1,
            container_guid: old_guid,
        })
        .unwrap();
    assert!(matches!(
        manifest.bind(&overlays),
        Err(ManifestReadError::MissingContainerGuid { .. })
    ));
}

#[test]
fn manifest_rejects_same_name_and_guid_with_different_container_bytes() {
    let asset = guid("stable-record");
    let old_bytes = build(&[("stable-record", b"old payload")]);
    let new_bytes = build(&[("stable-record", b"new payload")]);
    let mut overlays = OverlaySet::new();
    overlays.mount("shipping", 1, ContainerReader::new(old_bytes).unwrap());
    let manifest = RuntimeManifest::from_overlay(&overlays).unwrap();
    overlays.mount("shipping", 1, ContainerReader::new(new_bytes).unwrap());
    assert!(matches!(
        manifest.bind(&overlays),
        Err(ManifestReadError::ContainerIdentityMismatch { .. })
    ));
    assert!(overlays.contains(asset));
}

#[test]
fn runtime_manifest_round_trips_deterministically_and_rejects_bad_input() {
    let logical = guid("logical");
    let local = guid("container-local");
    let container_bytes = build(&[("container-local", b"payload")]);
    let mut manifest = RuntimeManifest::new();
    manifest
        .insert(ManifestRoute {
            guid: logical,
            container_name: "patch.container".into(),
            priority: 7,
            container_guid: local,
        })
        .unwrap();
    manifest
        .bind_container("patch.container", identity(&container_bytes))
        .unwrap();
    let encoded = manifest.encode().unwrap();
    assert_eq!(RuntimeManifest::decode(&encoded).unwrap(), manifest);
    assert_eq!(manifest.encode().unwrap(), encoded);
    assert_eq!(manifest.route(logical).unwrap().container_guid, local);
    for invalid in ["../escape", "a/b", "a\\b", ".", ".."] {
        assert_eq!(
            manifest.bind_container(invalid, identity(&container_bytes)),
            Err(ManifestError::InvalidName),
            "invalid container name {invalid:?} must be rejected",
        );
    }
    assert_eq!(
        RuntimeManifest::decode(&encoded[..encoded.len() - 1]),
        Err(ManifestError::Truncated)
    );

    let mut tampered = encoded.clone();
    tampered[HEADER_NAME_OFFSET] ^= 1;
    assert_eq!(
        RuntimeManifest::decode(&tampered),
        Err(ManifestError::HashMismatch)
    );

    let mut unknown = encoded;
    unknown[8..10].copy_from_slice(&99_u16.to_le_bytes());
    assert_eq!(
        RuntimeManifest::decode(&unknown),
        Err(ManifestError::UnsupportedVersion(99))
    );
}

#[test]
fn runtime_manifest_persists_route_dependencies_and_delivery_metadata() {
    let root = guid("metadata-root");
    let hard = guid("metadata-hard");
    let optional = guid("metadata-optional");
    let bytes = build(&[
        ("metadata-root", b"root"),
        ("metadata-hard", b"hard"),
        ("metadata-optional", b"optional"),
    ]);
    let mut manifest = RuntimeManifest::new();
    for (guid, name) in [(root, "root"), (hard, "hard"), (optional, "optional")] {
        manifest
            .insert(ManifestRoute {
                guid,
                container_name: name.into(),
                priority: 1,
                container_guid: guid,
            })
            .unwrap();
        manifest.bind_container(name, identity(&bytes)).unwrap();
    }
    manifest
        .set_metadata(
            root,
            ManifestMetadata {
                hard_dependencies: vec![hard],
                optional_dependencies: vec![optional],
                initial_residency: 2,
                streaming_level: 4,
                codec: Codec::Deflate,
                schema: 9,
                install_capabilities: 0x25,
            },
        )
        .unwrap();
    let encoded = manifest.encode().unwrap();
    let decoded = RuntimeManifest::decode(&encoded).unwrap();
    assert_eq!(decoded, manifest);
    assert_eq!(decoded.metadata(root).unwrap().schema, 9);
    assert_eq!(
        decoded.metadata(root).unwrap().hard_dependencies,
        vec![hard]
    );
}

#[test]
fn unknown_manifest_codec_is_a_structured_error() {
    let id = guid("manifest-unknown-codec");
    let bytes = build(&[("manifest-unknown-codec", b"payload")]);
    let mut manifest = RuntimeManifest::new();
    manifest
        .insert(ManifestRoute {
            guid: id,
            container_name: "asset".into(),
            priority: 0,
            container_guid: id,
        })
        .unwrap();
    manifest.bind_container("asset", identity(&bytes)).unwrap();
    let mut encoded = manifest.encode().unwrap();

    // One container and one route with no dependencies: metadata starts after
    // the fixed header, container record/name, route record/name.
    let codec_offset = 20 + 50 + "asset".len() + 38 + "asset".len() + 10;
    encoded[codec_offset] = 0xff;
    let body_end = encoded.len() - 32;
    let hash = *blake3::hash(&encoded[..body_end]).as_bytes();
    encoded[body_end..].copy_from_slice(&hash);

    assert_eq!(
        RuntimeManifest::decode(&encoded),
        Err(ManifestError::UnsupportedCodec(0xff))
    );
}

#[test]
fn runtime_manifest_rejects_dependency_cycles() {
    let first = guid("dependency-cycle-first");
    let second = guid("dependency-cycle-second");
    let bytes = build(&[("first", b"first"), ("second", b"second")]);
    let mut manifest = RuntimeManifest::new();
    for (guid, name) in [(first, "first"), (second, "second")] {
        manifest
            .insert(ManifestRoute {
                guid,
                container_name: name.into(),
                priority: 0,
                container_guid: guid,
            })
            .unwrap();
        manifest.bind_container(name, identity(&bytes)).unwrap();
    }
    manifest
        .set_metadata(
            first,
            ManifestMetadata {
                hard_dependencies: vec![second],
                ..ManifestMetadata::default()
            },
        )
        .unwrap();
    manifest
        .set_metadata(
            second,
            ManifestMetadata {
                hard_dependencies: vec![first],
                ..ManifestMetadata::default()
            },
        )
        .unwrap();
    assert_eq!(manifest.encode(), Err(ManifestError::InvalidFormat));
}

const HEADER_NAME_OFFSET: usize = 20 + 50 + "patch.container".len() + 38;

#[test]
fn runtime_manifest_rejects_duplicate_guid_records_even_with_valid_hash() {
    let mut manifest = RuntimeManifest::new();
    let container_bytes = build(&[("identity", b"payload")]);
    for name in ["a", "b"] {
        manifest
            .insert(ManifestRoute {
                guid: guid(name),
                container_name: name.into(),
                priority: 0,
                container_guid: guid(name),
            })
            .unwrap();
        manifest
            .bind_container(name, identity(&container_bytes))
            .unwrap();
    }
    let mut bytes = manifest.encode().unwrap();
    let route_table = 20 + 2 * (50 + 1);
    let first_guid = bytes[route_table..route_table + 16].to_vec();
    // Locate the second variable-size record and replace its logical GUID.
    let first_name_len = u16::from_le_bytes(
        bytes[route_table + 36..route_table + 38]
            .try_into()
            .unwrap(),
    ) as usize;
    let second_guid_at = route_table + 38 + first_name_len + 24;
    bytes[second_guid_at..second_guid_at + 16].copy_from_slice(&first_guid);
    let body_end = bytes.len() - 32;
    let digest = blake3::hash(&bytes[..body_end]);
    bytes[body_end..].copy_from_slice(digest.as_bytes());
    assert!(matches!(
        RuntimeManifest::decode(&bytes),
        Err(ManifestError::DuplicateGuid(_))
    ));
}

#[test]
fn overlay_manifest_exports_the_priority_winner_snapshot() {
    let id = guid("overlay-index");
    let mut overlays = OverlaySet::new();
    overlays.mount(
        "low",
        -2,
        ContainerReader::new(build(&[("overlay-index", b"low")])).unwrap(),
    );
    overlays.mount(
        "high",
        4,
        ContainerReader::new(build(&[("overlay-index", b"high")])).unwrap(),
    );
    let manifest = RuntimeManifest::from_overlay(&overlays).unwrap();
    let route = manifest.route(id).unwrap();
    assert_eq!(route.container_name, "high");
    assert_eq!(route.priority, 4);
    assert_eq!(route.container_guid, id);
    assert_eq!(
        RuntimeManifest::decode(&manifest.encode().unwrap()).unwrap(),
        manifest
    );
}

#[test]
fn manifest_reader_routes_logical_guid_to_container_guid() {
    let logical = guid("logical.asset");
    let local = guid("container-local.asset");
    let mut overlays = OverlaySet::new();
    let container_bytes = build(&[("container-local.asset", b"mapped bytes")]);
    overlays.mount(
        "patch",
        7,
        ContainerReader::new(container_bytes.clone()).unwrap(),
    );

    let mut manifest = RuntimeManifest::new();
    manifest
        .insert(ManifestRoute {
            guid: logical,
            container_name: "patch".into(),
            priority: 7,
            container_guid: local,
        })
        .unwrap();
    manifest
        .bind_container("patch", identity(&container_bytes))
        .unwrap();
    let reader = manifest.bind(&overlays).unwrap();
    assert_eq!(reader.read_all(logical).unwrap(), b"mapped bytes");
    let mut range = [0; 5];
    reader.read_range(logical, 7, &mut range).unwrap();
    assert_eq!(&range, b"bytes");
}

#[test]
fn manifest_reader_batches_ranges_per_container_and_preserves_logical_order() {
    let first_logical = guid("logical-first.batch");
    let second_logical = guid("logical-second.batch");
    let first_local = guid("local-first.batch");
    let second_local = guid("local-second.batch");
    let first_bytes = build(&[("local-first.batch", b"first payload")]);
    let second_bytes = build(&[("local-second.batch", b"second payload")]);
    let mut overlays = OverlaySet::new();
    overlays.mount(
        "first",
        1,
        ContainerReader::new(first_bytes.clone()).unwrap(),
    );
    overlays.mount(
        "second",
        1,
        ContainerReader::new(second_bytes.clone()).unwrap(),
    );

    let mut manifest = RuntimeManifest::new();
    manifest
        .insert(ManifestRoute {
            guid: first_logical,
            container_name: "first".into(),
            priority: 1,
            container_guid: first_local,
        })
        .unwrap();
    manifest
        .insert(ManifestRoute {
            guid: second_logical,
            container_name: "second".into(),
            priority: 1,
            container_guid: second_local,
        })
        .unwrap();
    manifest
        .bind_container("first", identity(&first_bytes))
        .unwrap();
    manifest
        .bind_container("second", identity(&second_bytes))
        .unwrap();

    let reader = manifest.bind(&overlays).unwrap();
    let output = reader
        .read_ranges(
            &[
                ManifestRangeRequest::new(second_logical, 7, 7),
                ManifestRangeRequest::new(first_logical, 0, 5),
            ],
            RangeBatchPolicy::default(),
        )
        .unwrap();
    assert_eq!(output, vec![b"payload".to_vec(), b"first".to_vec()]);
}

#[test]
fn manifest_reader_reads_ranges_from_a_version_two_chunk_container() {
    let logical = guid("logical-v2-range");
    let local = guid("local-v2-range");
    let policy = ChunkPolicy {
        max_chunk_size: 64,
        chunk_alignment: 16,
        max_chunks: 1,
    };
    let plan = ChunkPlanner::new(policy)
        .plan([ChunkInput {
            guid: local,
            size: 16,
            alignment: 1,
            dependencies: vec![],
            priority: 0,
        }])
        .unwrap();
    let mut writer = ContainerWriter::new();
    writer.add(local, b"manifest-v2-data".to_vec()).unwrap();
    let bytes = writer
        .finish_with_chunk_compression(&plan, policy.chunk_alignment, Codec::Raw)
        .unwrap();

    let mut overlays = OverlaySet::new();
    overlays.mount("v2-range", 3, ContainerReader::new(bytes.clone()).unwrap());
    let mut manifest = RuntimeManifest::new();
    manifest
        .insert(ManifestRoute {
            guid: logical,
            container_name: "v2-range".into(),
            priority: 3,
            container_guid: local,
        })
        .unwrap();
    manifest
        .bind_container("v2-range", identity(&bytes))
        .unwrap();
    let reader = manifest.bind(&overlays).unwrap();

    let mut range = [0; 6];
    reader.read_range(logical, 9, &mut range).unwrap();
    assert_eq!(&range, b"v2-dat");
    assert_eq!(
        reader
            .read_ranges(
                &[ManifestRangeRequest::new(logical, 0, 8)],
                RangeBatchPolicy::default(),
            )
            .unwrap(),
        vec![b"manifest".to_vec()]
    );
}

#[test]
fn manifest_reader_accepts_compressed_v2_entries_after_a_large_raw_offset() {
    let first = guid("manifest-v2-compressed-first");
    let logical = guid("logical-v2-compressed-range");
    let second = guid("local-v2-compressed-range");
    let policy = ChunkPolicy {
        max_chunk_size: 2048,
        chunk_alignment: 16,
        max_chunks: 1,
    };
    let plan = ChunkPlanner::new(policy)
        .plan([
            ChunkInput {
                guid: first,
                size: 1024,
                alignment: 1,
                dependencies: vec![],
                priority: 1,
            },
            ChunkInput {
                guid: second,
                size: 128,
                alignment: 1,
                dependencies: vec![],
                priority: 0,
            },
        ])
        .unwrap();
    let mut writer = ContainerWriter::new();
    writer.add(first, vec![b'a'; 1024]).unwrap();
    writer.add(second, vec![b'b'; 128]).unwrap();
    let bytes = writer
        .finish_with_chunk_compression(&plan, policy.chunk_alignment, Codec::Deflate)
        .unwrap();
    let container = ContainerReader::new(bytes.clone()).unwrap();
    let entry = container.entry(second).unwrap();
    assert_eq!(entry.codec, Codec::Deflate);
    assert!(entry.raw_offset > entry.stored_len);

    let mut overlays = OverlaySet::new();
    overlays.mount(
        "v2-compressed-range",
        4,
        ContainerReader::new(bytes.clone()).unwrap(),
    );
    let mut manifest = RuntimeManifest::new();
    manifest
        .insert(ManifestRoute {
            guid: logical,
            container_name: "v2-compressed-range".into(),
            priority: 4,
            container_guid: second,
        })
        .unwrap();
    manifest
        .bind_container("v2-compressed-range", identity(&bytes))
        .unwrap();

    let reader = manifest.bind(&overlays).unwrap();
    assert_eq!(
        reader
            .read_ranges(
                &[ManifestRangeRequest::new(logical, 0, 128)],
                RangeBatchPolicy::default(),
            )
            .unwrap(),
        vec![vec![b'b'; 128]]
    );
}

#[test]
fn manifest_reader_distinguishes_route_container_guid_and_range_errors() {
    let logical = guid("logical-errors.asset");
    let local = guid("local-errors.asset");
    let absent = guid("absent-errors.asset");

    let empty_manifest = RuntimeManifest::new();
    let overlays = OverlaySet::new();
    let reader = empty_manifest.bind(&overlays).unwrap();
    assert_eq!(
        reader.read_all(logical),
        Err(ManifestReadError::MissingRoute(logical))
    );

    let mut missing_container = RuntimeManifest::new();
    missing_container
        .insert(ManifestRoute {
            guid: logical,
            container_name: "not-mounted".into(),
            priority: 0,
            container_guid: local,
        })
        .unwrap();
    assert!(matches!(
        missing_container.bind(&overlays),
        Err(ManifestReadError::MissingContainer { .. })
    ));

    let mut overlays = OverlaySet::new();
    let mounted_bytes = build(&[("other.asset", b"other")]);
    overlays.mount(
        "mounted",
        0,
        ContainerReader::new(mounted_bytes.clone()).unwrap(),
    );
    let mut missing_guid = RuntimeManifest::new();
    missing_guid
        .insert(ManifestRoute {
            guid: logical,
            container_name: "mounted".into(),
            priority: 0,
            container_guid: absent,
        })
        .unwrap();
    assert!(matches!(
        missing_guid.bind(&overlays),
        Err(ManifestReadError::MissingContainerGuid { .. })
    ));

    let mut valid = RuntimeManifest::new();
    valid
        .insert(ManifestRoute {
            guid: logical,
            container_name: "mounted".into(),
            priority: 0,
            container_guid: guid("other.asset"),
        })
        .unwrap();
    valid
        .bind_container("mounted", identity(&mounted_bytes))
        .unwrap();
    let reader = valid.bind(&overlays).unwrap();
    let mut output = [0xA5; 2];
    assert!(matches!(
        reader.read_range(logical, 4, &mut output),
        Err(ManifestReadError::InvalidRange { .. })
    ));
    assert_eq!(output, [0xA5; 2]);
}

#[test]
fn manifest_reader_rejects_priority_and_source_drift() {
    let logical = guid("logical-drift.asset");
    let local = guid("local-drift.asset");
    let mut overlays = OverlaySet::new();
    overlays.mount(
        "base",
        1,
        ContainerReader::new(build(&[("local-drift.asset", b"base")])).unwrap(),
    );

    let mut wrong_priority = RuntimeManifest::new();
    wrong_priority
        .insert(ManifestRoute {
            guid: logical,
            container_name: "base".into(),
            priority: 2,
            container_guid: local,
        })
        .unwrap();
    assert!(matches!(
        wrong_priority.bind(&overlays),
        Err(ManifestReadError::PriorityMismatch { .. })
    ));

    overlays.mount(
        "patch",
        2,
        ContainerReader::new(build(&[("local-drift.asset", b"patch")])).unwrap(),
    );
    let mut wrong_source = RuntimeManifest::new();
    wrong_source
        .insert(ManifestRoute {
            guid: logical,
            container_name: "base".into(),
            priority: 1,
            container_guid: local,
        })
        .unwrap();
    assert!(matches!(
        wrong_source.bind(&overlays),
        Err(ManifestReadError::SourceMismatch { .. })
    ));
}

#[test]
fn manifest_asset_reader_implements_vfs_path_and_subasset_contract() {
    let parent = guid("scene.asset");
    let child = StableGuid::derive_sub(parent, "Mesh0");
    let mut writer = ContainerWriter::new();
    writer.add(parent, b"scene bytes".to_vec()).unwrap();
    writer.add(child, b"mesh bytes".to_vec()).unwrap();
    let container = ContainerReader::new(writer.finish().unwrap()).unwrap();
    let container_identity = {
        let info = container.info();
        ContainerIdentity {
            byte_len: info.total_len,
            content_hash: info.content_hash,
            generation: info.generation,
        }
    };

    let mut overlays = OverlaySet::new();
    overlays.mount("shipping", 3, container);
    let mut manifest = RuntimeManifest::new();
    manifest
        .insert(ManifestRoute {
            guid: parent,
            container_name: "shipping".into(),
            priority: 3,
            container_guid: parent,
        })
        .unwrap();
    manifest
        .bind_container("shipping", container_identity)
        .unwrap();
    manifest
        .insert(ManifestRoute {
            guid: child,
            container_name: "shipping".into(),
            priority: 3,
            container_guid: child,
        })
        .unwrap();

    let source = ManifestAssetReader::new(manifest, overlays);
    assert_eq!(source.read("scene.asset").unwrap(), b"scene bytes");
    assert_eq!(
        source.read_range("scene.asset#Mesh0", 5, 5).unwrap(),
        b"bytes"
    );
    assert_eq!(
        source.read_range("scene.asset#Mesh0", u64::MAX, u64::MAX),
        Err(prism_asset::ReadError::OutOfRange {
            offset: u64::MAX,
            len: u64::MAX,
            size: 10,
        })
    );
    assert_eq!(
        source.read_range("scene.asset#Mesh0", 9, 2),
        Err(prism_asset::ReadError::OutOfRange {
            offset: 9,
            len: 2,
            size: 10,
        })
    );
    assert_eq!(source.metadata("scene.asset#Mesh0").unwrap().size, 10);
    assert_eq!(
        source.read("../scene.asset"),
        Err(prism_asset::ReadError::InvalidPath)
    );
}

#[test]
fn runtime_manifest_patch_replaces_routes_removes_tombstones_and_is_transactional() {
    let base_guid = StableGuid::from_u128(1);
    let removed_guid = StableGuid::from_u128(2);
    let replacement_guid = StableGuid::from_u128(3);
    let base_bytes = build(&[("base", b"base")]);
    let patch_bytes = build(&[("replacement", b"patch")]);

    let mut base = RuntimeManifest::new();
    base.insert(ManifestRoute {
        guid: base_guid,
        container_name: "base".into(),
        priority: 1,
        container_guid: base_guid,
    })
    .unwrap();
    base.insert(ManifestRoute {
        guid: removed_guid,
        container_name: "base".into(),
        priority: 1,
        container_guid: removed_guid,
    })
    .unwrap();
    base.bind_container("base", identity(&base_bytes)).unwrap();

    let mut patch = RuntimeManifest::new();
    patch
        .insert_with_metadata(
            ManifestRoute {
                guid: base_guid,
                container_name: "patch".into(),
                priority: 2,
                container_guid: replacement_guid,
            },
            ManifestMetadata::default(),
        )
        .unwrap();
    patch
        .bind_container("patch", identity(&patch_bytes))
        .unwrap();

    base.apply_patch(&patch, &[removed_guid]).unwrap();
    assert_eq!(base.route(base_guid).unwrap().container_name, "patch");
    assert!(base.route(removed_guid).is_none());

    let before = base.clone();
    let mut conflicting_patch = RuntimeManifest::new();
    conflicting_patch
        .insert(ManifestRoute {
            guid: StableGuid::from_u128(4),
            container_name: "patch".into(),
            priority: 3,
            container_guid: StableGuid::from_u128(4),
        })
        .unwrap();
    conflicting_patch
        .bind_container("patch", identity(&base_bytes))
        .unwrap();
    assert!(matches!(
        base.apply_patch(&conflicting_patch, &[]),
        Err(ManifestError::ContainerIdentityConflict(name)) if name == "patch"
    ));
    assert_eq!(base, before);
}

#[test]
fn runtime_package_install_publishes_only_after_manifest_binding() {
    let existing_guid = guid("existing.asset");
    let package_guid = guid("package.asset");
    let existing_bytes = build(&[("existing.asset", b"existing")]);
    let package_bytes = build(&[("package.asset", b"package")]);

    let mut overlay = OverlaySet::new();
    overlay.mount("base", 1, ContainerReader::new(existing_bytes).unwrap());

    let mut manifest = RuntimeManifest::new();
    manifest
        .insert(ManifestRoute {
            guid: package_guid,
            container_name: "package".into(),
            priority: 5,
            container_guid: package_guid,
        })
        .unwrap();
    manifest
        .bind_container("package", identity(&package_bytes))
        .unwrap();

    install_runtime_package(
        &mut overlay,
        &manifest,
        [InstallMount::new(
            "package",
            5,
            ContainerReader::new(package_bytes.clone()).unwrap(),
        )],
        std::iter::empty::<InstallTombstones>(),
    )
    .unwrap();

    assert_eq!(overlay.read_all(package_guid).unwrap(), b"package");
    assert_eq!(overlay.read_all(existing_guid).unwrap(), b"existing");
}

#[test]
fn runtime_package_install_failure_keeps_previous_overlay_snapshot() {
    let existing_guid = guid("existing.asset");
    let package_guid = guid("package.asset");
    let existing_bytes = build(&[("existing.asset", b"existing")]);
    let expected_bytes = build(&[("package.asset", b"expected")]);
    let wrong_bytes = build(&[("package.asset", b"wrong")]);

    let mut overlay = OverlaySet::new();
    overlay.mount("base", 1, ContainerReader::new(existing_bytes).unwrap());

    let mut manifest = RuntimeManifest::new();
    manifest
        .insert(ManifestRoute {
            guid: package_guid,
            container_name: "package".into(),
            priority: 5,
            container_guid: package_guid,
        })
        .unwrap();
    manifest
        .bind_container("package", identity(&expected_bytes))
        .unwrap();

    let error = install_runtime_package(
        &mut overlay,
        &manifest,
        [InstallMount::new(
            "package",
            5,
            ContainerReader::new(wrong_bytes).unwrap(),
        )],
        std::iter::empty::<InstallTombstones>(),
    )
    .unwrap_err();
    assert!(matches!(error, crate::InstallError::Manifest(_)));
    assert_eq!(overlay.read_all(existing_guid).unwrap(), b"existing");
    assert!(!overlay.contains(package_guid));
}

#[test]
fn runtime_package_install_policy_rejects_before_publishing_overlay() {
    let existing_guid = guid("existing.policy");
    let package_guid = guid("package.policy");
    let existing_bytes = build(&[("existing.policy", b"existing")]);
    let package_bytes = build(&[("package.policy", b"package")]);

    let mut overlay = OverlaySet::new();
    overlay.mount("base", 1, ContainerReader::new(existing_bytes).unwrap());

    let mut manifest = RuntimeManifest::new();
    manifest
        .insert_with_metadata(
            ManifestRoute {
                guid: package_guid,
                container_name: "package".into(),
                priority: 5,
                container_guid: package_guid,
            },
            ManifestMetadata {
                initial_residency: 3,
                streaming_level: 2,
                install_capabilities: 0b101,
                ..ManifestMetadata::default()
            },
        )
        .unwrap();
    manifest
        .bind_container("package", identity(&package_bytes))
        .unwrap();

    let error = crate::install_runtime_package_with_policy(
        &mut overlay,
        &manifest,
        [InstallMount::new(
            "package",
            5,
            ContainerReader::new(package_bytes).unwrap(),
        )],
        std::iter::empty::<InstallTombstones>(),
        RuntimeInstallPolicy {
            available_capabilities: 0b001,
            max_initial_residency: 3,
            max_streaming_level: 2,
            allowed_codecs: 1,
        },
    )
    .unwrap_err();
    assert!(matches!(error, crate::InstallError::Policy { guid, .. } if guid == package_guid));
    assert_eq!(overlay.read_all(existing_guid).unwrap(), b"existing");
    assert!(!overlay.contains(package_guid));
}

#[test]
fn runtime_install_policy_can_reject_block_deflate_codec() {
    let package_guid = guid("package.block-policy");
    let mut writer = ContainerWriter::new();
    writer.add_aligned(package_guid, vec![7; 512], 1).unwrap();
    let plan = ChunkPlanner::new(ChunkPolicy {
        max_chunk_size: 1024,
        chunk_alignment: 64,
        max_chunks: 1,
    })
    .plan([ChunkInput {
        guid: package_guid,
        size: 512,
        alignment: 1,
        dependencies: vec![],
        priority: 0,
    }])
    .unwrap();
    let package_bytes = writer
        .finish_with_chunk_compression(&plan, 64, Codec::BlockDeflate)
        .unwrap();

    let mut overlay = OverlaySet::new();
    let mut manifest = RuntimeManifest::new();
    manifest
        .insert_with_metadata(
            ManifestRoute {
                guid: package_guid,
                container_name: "package-block-policy".into(),
                priority: 5,
                container_guid: package_guid,
            },
            ManifestMetadata {
                codec: Codec::BlockDeflate,
                ..ManifestMetadata::default()
            },
        )
        .unwrap();
    manifest
        .bind_container("package-block-policy", identity(&package_bytes))
        .unwrap();

    let error = crate::install_runtime_package_with_policy(
        &mut overlay,
        &manifest,
        [InstallMount::new(
            "package-block-policy",
            5,
            ContainerReader::new(package_bytes).unwrap(),
        )],
        std::iter::empty::<InstallTombstones>(),
        RuntimeInstallPolicy {
            allowed_codecs: (1 << Codec::Raw as u8) | (1 << Codec::Deflate as u8),
            ..RuntimeInstallPolicy::permissive()
        },
    )
    .unwrap_err();
    assert!(matches!(
        error,
        crate::InstallError::Policy { guid, .. } if guid == package_guid
    ));
    assert!(!overlay.contains(package_guid));
}

#[test]
fn runtime_package_transaction_rolls_back_and_recovers_from_bounded_record() {
    let existing_guid = guid("existing.rollback");
    let package_guid = guid("package.rollback");
    let existing_bytes = build(&[("existing.rollback", b"existing")]);
    let package_bytes = build(&[("package.rollback", b"package")]);
    let provenance = b"release-provenance-v1";

    let mut overlay = OverlaySet::new();
    overlay.mount("base", 1, ContainerReader::new(existing_bytes).unwrap());
    let previous = overlay.clone();

    let mut manifest = RuntimeManifest::new();
    manifest
        .insert(ManifestRoute {
            guid: package_guid,
            container_name: "package".into(),
            priority: 5,
            container_guid: package_guid,
        })
        .unwrap();
    manifest
        .bind_container("package", identity(&package_bytes))
        .unwrap();

    let mut handle = install_runtime_package_transaction(
        &mut overlay,
        &manifest,
        [InstallMount::new(
            "package",
            5,
            ContainerReader::new(package_bytes).unwrap(),
        )],
        std::iter::empty::<InstallTombstones>(),
        provenance,
    )
    .unwrap();
    let record = handle.recovery_record();
    let encoded = record.encode().unwrap();
    assert_eq!(
        crate::RuntimeInstallRecovery::decode(&encoded).unwrap(),
        record
    );
    assert_eq!(overlay.read_all(package_guid).unwrap(), b"package");

    handle.rollback(&mut overlay).unwrap();
    assert_eq!(overlay.read_all(existing_guid).unwrap(), b"existing");
    assert!(!overlay.contains(package_guid));

    let mut installed = previous.clone();
    installed.mount(
        "package",
        5,
        ContainerReader::new(build(&[("package.rollback", b"package")])).unwrap(),
    );
    let decoded = crate::RuntimeInstallRecovery::decode(&encoded).unwrap();
    decoded
        .recover_rollback(&mut installed, previous, &manifest, provenance)
        .unwrap();
    assert_eq!(installed.read_all(existing_guid).unwrap(), b"existing");
    assert!(!installed.contains(package_guid));
}

#[test]
fn runtime_package_recovery_rejects_tampering_without_mutating_live_overlay() {
    let package_guid = guid("package.recovery-tamper");
    let package_bytes = build(&[("package.recovery-tamper", b"package")]);
    let mut overlay = OverlaySet::new();
    let previous = overlay.clone();
    let mut manifest = RuntimeManifest::new();
    manifest
        .insert(ManifestRoute {
            guid: package_guid,
            container_name: "package".into(),
            priority: 5,
            container_guid: package_guid,
        })
        .unwrap();
    manifest
        .bind_container("package", identity(&package_bytes))
        .unwrap();
    let handle = install_runtime_package_transaction(
        &mut overlay,
        &manifest,
        [InstallMount::new(
            "package",
            5,
            ContainerReader::new(package_bytes).unwrap(),
        )],
        std::iter::empty::<InstallTombstones>(),
        b"provenance",
    )
    .unwrap();
    let mut record = handle.recovery_record().encode().unwrap();
    let last = record.len() - 1;
    record[last] ^= 1;
    assert!(matches!(
        crate::RuntimeInstallRecovery::decode(&record),
        Err(crate::RecoveryError::HashMismatch)
    ));
    assert!(overlay.contains(package_guid));
    handle.commit();
    assert!(!previous.contains(package_guid));
}
