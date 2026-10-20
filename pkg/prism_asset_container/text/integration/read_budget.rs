//! Admission must bound untrusted work before I/O and release all reservations.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

use prism_asset::StableGuid;
use prism_asset_container::{
    ChunkInput, ChunkPlanner, ChunkPolicy, Codec, ContainerError, ContainerReadBudget,
    ContainerReadLimits, ContainerReader, ContainerWriter, MemoryRangeSource, RangeBatchPolicy,
    RangeRequest, RangeSource, RangeSourceError,
};

fn fixture(compressed: bool, len: usize) -> (StableGuid, Vec<u8>) {
    let guid = StableGuid::from_path("budget.asset");
    let mut writer = ContainerWriter::new();
    if compressed {
        writer.add_deflate(guid, vec![42; len]).unwrap();
    } else {
        writer.add(guid, vec![42; len]).unwrap();
    }
    (guid, writer.finish().unwrap())
}

fn rejected(bytes: Vec<u8>, limits: ContainerReadLimits, resource: &str) {
    let budget = ContainerReadBudget::new(limits);
    let error = ContainerReader::with_budget(bytes, budget.clone()).unwrap_err();
    assert!(
        matches!(error, ContainerError::ResourceLimit { resource: actual, .. } if actual == resource),
        "{error:?}"
    );
    assert_eq!(budget.in_flight_bytes(), 0);
    assert!(budget.peak_in_flight_bytes() <= limits.max_in_flight_bytes);
}

/// Every independently configurable mount ceiling denies before unbounded work.
#[test]
fn mount_limits_cover_encoded_metadata_records_payloads_and_validation_work() {
    let (_, bytes) = fixture(true, 128 * 1024);
    let defaults = ContainerReadLimits::default();
    rejected(
        bytes.clone(),
        ContainerReadLimits {
            max_container_bytes: 10,
            ..defaults
        },
        "container bytes",
    );
    rejected(
        bytes.clone(),
        ContainerReadLimits {
            max_metadata_bytes: 79,
            ..defaults
        },
        "metadata bytes",
    );
    rejected(
        bytes.clone(),
        ContainerReadLimits {
            max_records: 0,
            ..defaults
        },
        "entry count",
    );
    rejected(
        bytes.clone(),
        ContainerReadLimits {
            max_stored_unit_bytes: 1,
            ..defaults
        },
        "stored unit bytes",
    );
    rejected(
        bytes.clone(),
        ContainerReadLimits {
            max_decoded_unit_bytes: 100,
            ..defaults
        },
        "decoded unit bytes",
    );
    rejected(
        bytes.clone(),
        ContainerReadLimits {
            max_validation_decoded_bytes: 100,
            ..defaults
        },
        "validation decoded bytes",
    );
    rejected(
        bytes.clone(),
        ContainerReadLimits {
            max_expansion_ratio: 1,
            ..defaults
        },
        "expansion bytes",
    );
    rejected(
        bytes,
        ContainerReadLimits {
            max_in_flight_bytes: 0,
            ..defaults
        },
        "in-flight bytes",
    );
}

/// Huge declarations are rejected from a validated directory before decoding.
#[test]
fn hostile_raw_length_is_rejected_without_reserving_the_advertised_payload() {
    let (_, mut bytes) = fixture(true, 8);
    bytes[128..136].copy_from_slice(&u64::MAX.to_le_bytes());
    let digest = *blake3::hash(&bytes[96..176]).as_bytes();
    bytes[52..84].copy_from_slice(&digest);
    let budget = ContainerReadBudget::default();
    assert!(matches!(
        ContainerReader::with_budget(bytes, budget.clone()),
        Err(ContainerError::ResourceLimit {
            resource: "decoded unit bytes",
            ..
        })
    ));
    assert_eq!(budget.in_flight_bytes(), 0);
    assert!(budget.peak_in_flight_bytes() < 64 * 1024);
}

#[derive(Debug)]
struct ObservedSource {
    source: MemoryRangeSource,
    reads: AtomicUsize,
    block_next: AtomicBool,
    entered: Mutex<mpsc::SyncSender<()>>,
    release: Mutex<mpsc::Receiver<()>>,
}

impl RangeSource for ObservedSource {
    fn len(&self) -> u64 {
        self.source.len()
    }

    fn read_range(&self, offset: u64, out: &mut [u8]) -> Result<(), RangeSourceError> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        if self.block_next.swap(false, Ordering::AcqRel) {
            self.entered.lock().unwrap().send(()).unwrap();
            self.release
                .lock()
                .unwrap()
                .recv_timeout(Duration::from_secs(5))
                .unwrap();
        }
        self.source.read_range(offset, out)
    }
}

fn observed(
    bytes: Vec<u8>,
) -> (
    Arc<ObservedSource>,
    mpsc::Receiver<()>,
    mpsc::SyncSender<()>,
) {
    let (entered_tx, entered_rx) = mpsc::sync_channel(1);
    let (release_tx, release_rx) = mpsc::sync_channel(1);
    (
        Arc::new(ObservedSource {
            source: MemoryRangeSource::new(bytes),
            reads: AtomicUsize::new(0),
            block_next: AtomicBool::new(false),
            entered: Mutex::new(entered_tx),
            release: Mutex::new(release_rx),
        }),
        entered_rx,
        release_tx,
    )
}

/// A late invalid compressed request must not issue I/O for an earlier valid one.
#[test]
fn compressed_batch_validates_all_requests_before_any_io() {
    let (guid, bytes) = fixture(true, 512);
    let (source, _, _) = observed(bytes);
    let reader = ContainerReader::from_range_source(source.clone()).unwrap();
    source.reads.store(0, Ordering::Relaxed);
    assert!(matches!(
        reader.read_ranges(
            &[
                RangeRequest::new(guid, 0, 1),
                RangeRequest::new(guid, 512, 1)
            ],
            RangeBatchPolicy::default()
        ),
        Err(ContainerError::InvalidRange)
    ));
    assert_eq!(source.reads.load(Ordering::Relaxed), 0);
    assert_eq!(reader.budget().in_flight_bytes(), 0);
}

/// Repeated v1 subranges share one stored read and preserve request order.
#[test]
fn compressed_entry_batch_decodes_once_and_skips_empty_requests() {
    let guid = StableGuid::from_path("pattern");
    let raw: Vec<_> = (0..2048).map(|value| (value % 251) as u8).collect();
    let mut writer = ContainerWriter::new();
    writer.add_deflate(guid, raw.clone()).unwrap();
    let (source, _, _) = observed(writer.finish().unwrap());
    let reader = ContainerReader::from_range_source(source.clone()).unwrap();
    source.reads.store(0, Ordering::Relaxed);
    let output = reader
        .read_ranges(
            &[
                RangeRequest::new(guid, 1000, 20),
                RangeRequest::new(guid, 0, 30),
                RangeRequest::new(guid, 1000, 20),
                RangeRequest::new(guid, 2048, 0),
            ],
            RangeBatchPolicy::default(),
        )
        .unwrap();
    assert_eq!(
        output,
        vec![
            raw[1000..1020].to_vec(),
            raw[..30].to_vec(),
            raw[1000..1020].to_vec(),
            vec![]
        ]
    );
    assert_eq!(source.reads.load(Ordering::Relaxed), 1);
    source.reads.store(0, Ordering::Relaxed);
    assert_eq!(
        reader
            .read_ranges(
                &[RangeRequest::new(guid, 2048, 0)],
                RangeBatchPolicy::default()
            )
            .unwrap(),
        vec![Vec::<u8>::new()]
    );
    assert_eq!(source.reads.load(Ordering::Relaxed), 0);
    assert_eq!(reader.budget().in_flight_bytes(), 0);
}

/// Different v2 assets in a shared compressed chunk decode that chunk once.
#[test]
fn compressed_chunk_batch_shares_decode_across_asset_boundaries() {
    let a = StableGuid::from_path("chunk-a");
    let b = StableGuid::from_path("chunk-b");
    let mut writer = ContainerWriter::new();
    writer.add(a, vec![7; 256]).unwrap();
    writer.add(b, vec![9; 512]).unwrap();
    let plan = ChunkPlanner::new(ChunkPolicy::default())
        .plan([
            ChunkInput {
                guid: a,
                size: 256,
                alignment: 16,
                dependencies: vec![],
                priority: 0,
            },
            ChunkInput {
                guid: b,
                size: 512,
                alignment: 16,
                dependencies: vec![],
                priority: 0,
            },
        ])
        .unwrap();
    assert_eq!(plan.chunks.len(), 1);
    let (source, _, _) = observed(
        writer
            .finish_with_chunk_compression(
                &plan,
                ChunkPolicy::default().chunk_alignment,
                Codec::Deflate,
            )
            .unwrap(),
    );
    let reader = ContainerReader::from_range_source(source.clone()).unwrap();
    source.reads.store(0, Ordering::Relaxed);
    let output = reader
        .read_ranges(
            &[
                RangeRequest::new(b, 400, 8),
                RangeRequest::new(a, 250, 6),
                RangeRequest::new(b, 2, 4),
            ],
            RangeBatchPolicy::default(),
        )
        .unwrap();
    assert_eq!(output, vec![vec![9; 8], vec![7; 6], vec![9; 4]]);
    assert_eq!(source.reads.load(Ordering::Relaxed), 1);
    assert_eq!(reader.budget().in_flight_bytes(), 0);
}

/// Duplicate requests consume output capacity independently, even if I/O merges.
#[test]
fn aggregate_output_limit_counts_duplicate_requests_before_io() {
    let (guid, bytes) = fixture(true, 512);
    let (source, _, _) = observed(bytes);
    let budget = ContainerReadBudget::new(ContainerReadLimits {
        max_output_bytes: 10,
        ..Default::default()
    });
    let reader =
        ContainerReader::from_range_source_with_budget(source.clone(), budget.clone()).unwrap();
    source.reads.store(0, Ordering::Relaxed);
    assert!(matches!(
        reader.read_ranges(
            &[RangeRequest::new(guid, 0, 6); 2],
            RangeBatchPolicy::default()
        ),
        Err(ContainerError::ResourceLimit {
            resource: "output bytes",
            requested: 12,
            limit: 10
        })
    ));
    assert!(matches!(
        reader.read_all(guid),
        Err(ContainerError::ResourceLimit {
            resource: "output bytes",
            ..
        })
    ));
    assert_eq!(source.reads.load(Ordering::Relaxed), 0);
    assert_eq!(budget.in_flight_bytes(), 0);
}

/// Manifest/VFS allocation must use the routed mount's shared byte budget.
#[test]
fn manifest_vfs_owned_ranges_enforce_mount_limits_before_io() {
    use prism_asset::AssetReader;
    use prism_asset_container::{ManifestAssetReader, OverlaySet};

    let (guid, bytes) = fixture(false, 512);
    let (source, _, _) = observed(bytes);
    let budget = ContainerReadBudget::new(ContainerReadLimits {
        max_output_bytes: 16,
        ..Default::default()
    });
    let reader =
        ContainerReader::from_range_source_with_budget(source.clone(), budget.clone()).unwrap();
    assert!(matches!(
        reader.read_range_owned(guid, 0, 32),
        Err(ContainerError::ResourceLimit {
            resource: "output bytes",
            ..
        })
    ));
    let mut overlays = OverlaySet::new();
    overlays.mount("base", 0, reader);
    let manifest = overlays.runtime_manifest().unwrap();
    let vfs = ManifestAssetReader::new(manifest, overlays);
    source.reads.store(0, Ordering::Relaxed);
    assert!(
        matches!(vfs.read_range("budget.asset", 0, 32), Err(prism_asset::ReadError::Io(message)) if message.contains("output bytes"))
    );
    assert_eq!(source.reads.load(Ordering::Relaxed), 0);
    assert_eq!(budget.in_flight_bytes(), 0);
    assert_eq!(vfs.read_range("budget.asset", 1, 16).unwrap(), vec![42; 16]);
    assert_eq!(source.reads.load(Ordering::Relaxed), 1);
    assert_eq!(budget.in_flight_bytes(), 0);
}

/// Separate mounts share admission while a clone performs a blocked real read.
#[test]
fn concurrent_mounts_share_budget_and_release_after_success_and_rejection() {
    let (guid, bytes) = fixture(false, 512);
    let (source, entered, release) = observed(bytes);
    let budget = ContainerReadBudget::new(ContainerReadLimits {
        max_in_flight_bytes: 1000,
        ..Default::default()
    });
    let reader =
        ContainerReader::from_range_source_with_budget(source.clone(), budget.clone()).unwrap();
    let other =
        ContainerReader::from_range_source_with_budget(source.clone(), budget.clone()).unwrap();
    source.block_next.store(true, Ordering::Release);
    let clone = reader.clone();
    let worker = std::thread::spawn(move || clone.read_all(guid));
    entered.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!(budget.in_flight_bytes(), 512);
    let denied = other.read_all(guid);
    release.send(()).unwrap();
    assert_eq!(worker.join().unwrap().unwrap(), vec![42; 512]);
    assert!(matches!(
        denied,
        Err(ContainerError::ResourceLimit {
            resource: "in-flight bytes",
            ..
        })
    ));
    assert_eq!(budget.in_flight_bytes(), 0);
    assert_eq!(reader.read_all(guid).unwrap(), vec![42; 512]);
    assert_eq!(budget.in_flight_bytes(), 0);
    assert!(budget.peak_in_flight_bytes() <= 1000);
}

/// Mount visits compressed chunks in physical order, not interleaved GUID order.
#[test]
fn many_compressed_chunks_mount_with_only_one_decoded_chunk_live() {
    let mut writer = ContainerWriter::new();
    let mut inputs = Vec::new();
    let raw_len = 128 * 1024;
    for ordinal in 0..8 {
        let guid = StableGuid::from_path(&format!("chunk-{ordinal}"));
        writer.add(guid, vec![ordinal as u8; raw_len]).unwrap();
        inputs.push(ChunkInput {
            guid,
            size: raw_len as u64,
            alignment: 1,
            dependencies: vec![],
            priority: ordinal,
        });
    }
    let plan = ChunkPlanner::new(ChunkPolicy {
        max_chunk_size: raw_len as u64,
        chunk_alignment: 4096,
        max_chunks: 8,
    })
    .plan(inputs)
    .unwrap();
    let bytes = writer
        .finish_with_chunk_compression(&plan, 4096, Codec::Deflate)
        .unwrap();
    let budget = ContainerReadBudget::new(ContainerReadLimits {
        max_in_flight_bytes: raw_len as u64 + 8192,
        ..Default::default()
    });
    let reader = ContainerReader::with_budget(bytes, budget.clone()).unwrap();
    assert_eq!(reader.guids().count(), 8);
    assert_eq!(budget.in_flight_bytes(), 0);
    assert!(budget.peak_in_flight_bytes() <= raw_len as u64 + 8192);
}

/// Failed source reads release both output and decode-scratch reservations.
#[test]
fn source_error_releases_output_reservation() {
    let (guid, bytes) = fixture(false, 512);
    // A separate failure source preserves the immutable bytes but denies I/O
    // after mount, modeling permission/device loss without mutating payloads.
    #[derive(Debug)]
    struct Failing {
        source: MemoryRangeSource,
        fail: AtomicBool,
    }
    impl RangeSource for Failing {
        fn len(&self) -> u64 {
            self.source.len()
        }
        fn read_range(&self, offset: u64, out: &mut [u8]) -> Result<(), RangeSourceError> {
            if self.fail.load(Ordering::Relaxed) {
                Err(RangeSourceError::Io("device unavailable".into()))
            } else {
                self.source.read_range(offset, out)
            }
        }
    }
    let source = Arc::new(Failing {
        source: MemoryRangeSource::new(bytes),
        fail: AtomicBool::new(false),
    });
    let budget = ContainerReadBudget::default();
    let reader =
        ContainerReader::from_range_source_with_budget(source.clone(), budget.clone()).unwrap();
    source.fail.store(true, Ordering::Relaxed);
    assert!(matches!(
        reader.read_all(guid),
        Err(ContainerError::Source(_))
    ));
    assert_eq!(budget.in_flight_bytes(), 0);
}
