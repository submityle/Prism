//! Real file ranges must decode through the coordinator before SDF publication.

extern crate alloc;

use alloc::sync::Arc;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use prism_world_physics_bridge::{
    SdfDistanceGradient, SdfPhysicsBridgeBudget, SdfQueryResult, WorldSdfSnapshot,
    WorldSdfTransform,
};
use prism_world_physics_cell_runtime::{
    SdfCellCodec, SdfCellCodecLimits, SdfCellSnapshot, SdfPhysicsCellHost, SdfResidencyBudget,
};
use prism_world_residency::{BrickCoord, SparseSdfBrickGrid};
use prism_world_runtime::{
    CellReadRequest, FailureStage, FileCellProvider, LoadedCell, ManifestHeader, SceneHost,
    TraceEvent, TraceOp, WorldFrameCoordinator, CELL_HEADER_SIZE,
};
use prism_world_stream::{
    AabbI64, CellManifest, DataLayerMask, DemandSource, GlobalPos, SectorId, StreamBudget,
};

struct TempFile(PathBuf);
impl TempFile {
    fn new(bytes: &[u8]) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "prism_sdf_file_{}_{}.bin",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        file.write_all(bytes).unwrap();
        Self(path)
    }
}
impl Drop for TempFile {
    fn drop(&mut self) {
        fs::remove_file(&self.0).expect("remove owned test file");
    }
}

struct Host {
    reject: Arc<AtomicBool>,
    loads: Arc<AtomicU64>,
}
impl SceneHost for Host {
    type Error = &'static str;
    fn load_cell(&mut self, _: &CellManifest, payload: &[u8]) -> Result<(), Self::Error> {
        assert_eq!(
            &payload[..4],
            b"PSDC",
            "coordinator must remove exactly the outer header"
        );
        self.loads.fetch_add(1, Ordering::Relaxed);
        if self.reject.load(Ordering::Relaxed) {
            Err("host rejected load")
        } else {
            Ok(())
        }
    }
    fn evict_cell(&mut self, _: SectorId) -> Result<(), Self::Error> {
        if self.reject.load(Ordering::Relaxed) {
            Err("host rejected eviction")
        } else {
            Ok(())
        }
    }
}

fn codec() -> SdfCellCodec {
    SdfCellCodec::new(SdfCellCodecLimits::new(1024, 1, 8, 80))
}

fn source(revision: u64, offset: i16) -> SdfCellSnapshot {
    let mut grid = SparseSdfBrickGrid::new(2, 1).unwrap();
    grid.insert_brick(
        BrickCoord::new(0, 0, 0, 0),
        (0..8).map(|i| i + offset).collect(),
    )
    .unwrap();
    SdfCellSnapshot::new(
        SectorId::new(0, 0, 0, 0),
        WorldSdfSnapshot::new(
            WorldSdfTransform::new([0; 3], 100, 10).unwrap(),
            revision,
            grid,
        ),
    )
}

fn manifest(version: u64, payload: &[u8]) -> CellManifest {
    CellManifest {
        sector: SectorId::new(0, 0, 0, 0),
        bounds: AabbI64::new(GlobalPos::new(0, 0, 0), GlobalPos::new(200, 200, 200)).unwrap(),
        scene_key: 1,
        content_version: version,
        derived_hash: u128::from(version),
        io_bytes: (CELL_HEADER_SIZE + payload.len()) as u64,
        data_layers: DataLayerMask::EMPTY,
    }
}

fn load_file(
    host: &mut SdfPhysicsCellHost<'_, Host, SdfCellCodec>,
    current: CellManifest,
    payload: &[u8],
    previous: Option<CellManifest>,
) -> WorldFrameCoordinator<FileCellProvider> {
    let mut bytes = b"archive-prefix".to_vec();
    let offset = bytes.len() as u64;
    bytes.extend_from_slice(&ManifestHeader::from_manifest(current).encode());
    bytes.extend_from_slice(payload);
    bytes.extend_from_slice(b"archive-trailer");
    let file = TempFile::new(&bytes);
    // Seven-byte reads split both the outer header and SDF records.
    let mut provider = FileCellProvider::with_chunk_size(&file.0, 1, 1, 7).unwrap();
    provider
        .register(CellReadRequest::new(current, 0).with_byte_offset(offset))
        .unwrap();
    let mut coordinator = WorldFrameCoordinator::new(provider, 1).unwrap();
    let loaded: Vec<_> = previous.map(LoadedCell::new).into_iter().collect();
    let demand = DemandSource::new(current.sector, 0, 1, 10).unwrap();
    assert_eq!(
        coordinator
            .plan_frame(
                &[demand],
                &[current],
                &loaded,
                StreamBudget {
                    io_bytes: current.io_bytes,
                    max_actions: 1
                },
                host
            )
            .submitted,
        1
    );
    assert!(coordinator.poll_io(host));
    assert!(!coordinator.poll_io(host));
    assert_eq!(coordinator.in_flight(), 0);
    coordinator
}

fn assert_value(host: &SdfPhysicsCellHost<'_, Host, SdfCellCodec>, distance: f64) {
    assert_eq!(
        host.query_world_distance_gradient_mm(SectorId::new(0, 0, 0, 0), [25, 50, 75], 0),
        Ok(SdfQueryResult::Value(SdfDistanceGradient {
            distance_mm: distance,
            gradient: [0.1, 0.2, 0.4]
        }))
    );
}

/// CRC, stale revision and downstream rejection preserve the last committed field.
#[test]
fn file_load_reload_failure_retry_and_eviction() {
    let reject = Arc::new(AtomicBool::new(false));
    let loads = Arc::new(AtomicU64::new(0));
    let mut downstream = Host {
        reject: Arc::clone(&reject),
        loads: Arc::clone(&loads),
    };
    let mut host = SdfPhysicsCellHost::new(
        &mut downstream,
        codec(),
        SdfPhysicsBridgeBudget::new(1, 8, 80).unwrap(),
        SdfResidencyBudget::new(1, 1, 8, 80),
    );
    let initial = codec().encode(&source(1, 0)).unwrap();
    let old = manifest(1, &initial);
    let coordinator = load_file(&mut host, old, &initial, None);
    assert!(coordinator
        .trace()
        .as_slice()
        .iter()
        .any(|event| matches!(event, TraceEvent::Loaded { .. })));
    assert_value(&host, 42.5);
    let usage = host.resident_usage();
    let before = host.resident(old.sector).unwrap().clone();

    let next = codec().encode(&source(2, 7)).unwrap();
    let new = manifest(2, &next);
    let mut corrupt = next.clone();
    corrupt[144] ^= 1;
    let coordinator = load_file(&mut host, new, &corrupt, Some(old));
    assert!(coordinator.trace().as_slice().iter().any(|event| matches!(
        event,
        TraceEvent::Failed {
            stage: FailureStage::LoadHost,
            operation: TraceOp::Reload,
            ..
        }
    )));
    assert_eq!(
        loads.load(Ordering::Relaxed),
        1,
        "decode failure cannot reach downstream"
    );
    assert_eq!(host.resident(old.sector), Some(&before));
    assert_eq!(host.resident_usage(), usage);
    assert_value(&host, 42.5);

    let stale = codec().encode(&source(1, 7)).unwrap();
    load_file(&mut host, new, &stale, Some(old));
    assert_eq!(
        loads.load(Ordering::Relaxed),
        1,
        "revision rejection cannot reach downstream"
    );
    assert_value(&host, 42.5);

    reject.store(true, Ordering::Relaxed);
    load_file(&mut host, new, &next, Some(old));
    assert_eq!(loads.load(Ordering::Relaxed), 2);
    assert_eq!(host.resident(old.sector), Some(&before));
    assert_eq!(host.resident_usage(), usage);
    assert_value(&host, 42.5);

    reject.store(false, Ordering::Relaxed);
    let mut coordinator = load_file(&mut host, new, &next, Some(old));
    assert_eq!(loads.load(Ordering::Relaxed), 3);
    assert_eq!(host.resident(new.sector).unwrap().revision(), 2);
    assert_eq!(host.resident_usage(), usage);
    assert_value(&host, 112.5);
    reject.store(true, Ordering::Relaxed);
    let budget = StreamBudget {
        io_bytes: 0,
        max_actions: 1,
    };
    assert_eq!(
        coordinator
            .plan_frame(&[], &[new], &[LoadedCell::new(new)], budget, &mut host)
            .failed,
        1
    );
    assert_value(&host, 112.5);
    reject.store(false, Ordering::Relaxed);
    assert_eq!(
        coordinator
            .plan_frame(&[], &[new], &[LoadedCell::new(new)], budget, &mut host)
            .failed,
        0
    );
    assert_eq!(host.resident_usage(), (0, 0, 0, 0));
    assert_eq!(
        host.query_world_distance_gradient_mm(new.sector, [25, 50, 75], 0),
        Ok(SdfQueryResult::Unknown)
    );
}

/// A failed reload restores scheduler identity so the same coordinator can retry.
#[test]
fn same_coordinator_retries_a_repaired_file_without_losing_old_field() {
    let reject = Arc::new(AtomicBool::new(false));
    let loads = Arc::new(AtomicU64::new(0));
    let mut downstream = Host {
        reject,
        loads: Arc::clone(&loads),
    };
    let mut host = SdfPhysicsCellHost::new(
        &mut downstream,
        codec(),
        SdfPhysicsBridgeBudget::new(1, 8, 80).unwrap(),
        SdfResidencyBudget::new(1, 1, 8, 80),
    );
    let initial = codec().encode(&source(1, 0)).unwrap();
    let old = manifest(1, &initial);
    load_file(&mut host, old, &initial, None);
    let payload = codec().encode(&source(2, 7)).unwrap();
    let new = manifest(2, &payload);
    let mut bytes = ManifestHeader::from_manifest(new).encode();
    bytes.extend_from_slice(&payload);
    let mut corrupt = bytes.clone();
    corrupt[CELL_HEADER_SIZE + 144] ^= 1;
    let file = TempFile::new(&corrupt);
    let mut provider = FileCellProvider::with_chunk_size(&file.0, 1, 1, 7).unwrap();
    provider.register(CellReadRequest::new(new, 0)).unwrap();
    let mut coordinator = WorldFrameCoordinator::new(provider, 1).unwrap();
    let demand = DemandSource::new(new.sector, 0, 1, 10).unwrap();
    let budget = StreamBudget {
        io_bytes: new.io_bytes,
        max_actions: 1,
    };
    for repaired in [false, true] {
        if repaired {
            // Replace only this test-owned cache file after IO is terminal.
            let mut output = OpenOptions::new()
                .write(true)
                .truncate(true)
                .open(&file.0)
                .unwrap();
            output.write_all(&bytes).unwrap();
        }
        assert_eq!(
            coordinator
                .plan_frame(
                    &[demand],
                    &[new],
                    &[LoadedCell::new(old)],
                    budget,
                    &mut host
                )
                .submitted,
            1
        );
        assert!(coordinator.poll_io(&mut host));
        assert_eq!(coordinator.in_flight(), 0);
        assert_value(&host, if repaired { 112.5 } else { 42.5 });
        assert_eq!(
            host.resident(new.sector).unwrap().revision(),
            if repaired { 2 } else { 1 }
        );
        assert_eq!(loads.load(Ordering::Relaxed), if repaired { 2 } else { 1 });
    }
    let events = coordinator.trace().as_slice();
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(
                event,
                TraceEvent::Failed {
                    operation: TraceOp::Reload,
                    stage: FailureStage::LoadHost,
                    ..
                }
            ))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, TraceEvent::Loaded { .. }))
            .count(),
        1
    );
}
