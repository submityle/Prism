use super::*;

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use prism_world_stream::{AabbI64, GlobalPos};

fn manifest(version: u64, io_bytes: u64) -> CellManifest {
    CellManifest {
        sector: SectorId::new(0, 2, -1, 4),
        bounds: AabbI64::new(GlobalPos::new(0, 0, 0), GlobalPos::new(1, 1, 1))
            .expect("valid bounds"),
        scene_key: 0xD3C0_DED,
        content_version: version,
        derived_hash: u128::from(version),
        io_bytes,
        data_layers: DataLayerMask::EMPTY,
    }
}

#[derive(Clone)]
struct TextDecoder {
    decode_calls: Arc<AtomicUsize>,
    underestimate: Arc<AtomicBool>,
}

impl CellPayloadDecoder for TextDecoder {
    type Cell = String;
    type Error = &'static str;

    fn preflight(&mut self, _: &CellManifest, payload: &[u8]) -> Result<u64, Self::Error> {
        if payload == b"decode-error" {
            return Err("rejected fixture payload");
        }
        let actual = payload.len() as u64;
        Ok(if self.underestimate.load(Ordering::Relaxed) {
            actual.saturating_sub(1)
        } else {
            actual
        })
    }

    fn decode(
        &mut self,
        _: &CellManifest,
        payload: &[u8],
    ) -> Result<DecodedCell<Self::Cell>, Self::Error> {
        self.decode_calls.fetch_add(1, Ordering::Relaxed);
        let text = String::from_utf8(payload.to_vec()).map_err(|_| "invalid UTF-8")?;
        Ok(DecodedCell::new(text, payload.len() as u64))
    }
}

#[derive(Default)]
struct TypedHost {
    cells: BTreeMap<SectorId, String>,
    fail_prepare: Option<&'static str>,
    fail_evict: bool,
}

impl DecodedCellHost for TypedHost {
    type Cell = String;
    type Prepared = (SectorId, String);
    type Error = &'static str;

    fn prepare(
        &mut self,
        manifest: &CellManifest,
        cell: Self::Cell,
    ) -> Result<Self::Prepared, Self::Error> {
        if let Some(error) = self.fail_prepare {
            return Err(error);
        }
        Ok((manifest.sector, cell))
    }

    fn commit(&mut self, (sector, cell): Self::Prepared) {
        self.cells.insert(sector, cell);
    }

    fn evict(&mut self, sector: SectorId) -> Result<(), Self::Error> {
        if self.fail_evict {
            return Err("eviction rejected");
        }
        self.cells.remove(&sector);
        Ok(())
    }
}

fn typed_host(
    limits: CellDecodeLimits,
) -> (DecodedSceneHost<TextDecoder, TypedHost>, Arc<AtomicUsize>) {
    let decode_calls = Arc::new(AtomicUsize::new(0));
    (
        DecodedSceneHost::new(
            TextDecoder {
                decode_calls: decode_calls.clone(),
                underestimate: Arc::new(AtomicBool::new(false)),
            },
            TypedHost::default(),
            limits,
        ),
        decode_calls,
    )
}

#[test]
fn decoded_host_admits_budgets_before_decode_and_rejects_preflight_underestimates() {
    let (mut host, decode_calls) = typed_host(CellDecodeLimits::new(3, 3, 1));
    let m = manifest(1, 0);
    assert!(matches!(
        host.load_cell(&m, b"four"),
        Err(DecodedSceneHostError::CellBudget {
            required: 4,
            limit: 3
        })
    ));
    assert_eq!(decode_calls.load(Ordering::Relaxed), 0);
    assert_eq!(host.resident_cells(), 0);
    assert_eq!(host.resident_bytes(), 0);

    host.decoder_mut()
        .underestimate
        .store(true, Ordering::Relaxed);
    assert!(matches!(
        host.load_cell(&m, b"abc"),
        Err(DecodedSceneHostError::PreflightUnderestimated {
            actual: 3,
            reserved: 2
        })
    ));
    assert!(host.host().cells.is_empty());
    assert_eq!(host.resident_bytes(), 0);
}

#[test]
fn decode_and_prepare_failures_keep_old_cell_and_residency_on_reload() {
    let (mut host, _) = typed_host(CellDecodeLimits::new(32, 32, 1));
    let old = manifest(1, 0);
    let updated = manifest(2, 0);
    host.load_cell(&old, b"old").expect("initial typed load");
    assert_eq!(host.resident_bytes(), 3);

    assert!(matches!(
        host.load_cell(&updated, b"decode-error"),
        Err(DecodedSceneHostError::Decode(_))
    ));
    assert_eq!(
        host.host().cells.get(&old.sector).map(String::as_str),
        Some("old")
    );
    assert_eq!(host.resident_bytes(), 3);

    host.host_mut().fail_prepare = Some("prepare rejected");
    assert!(matches!(
        host.load_cell(&updated, b"new-value"),
        Err(DecodedSceneHostError::Prepare("prepare rejected"))
    ));
    assert_eq!(
        host.host().cells.get(&old.sector).map(String::as_str),
        Some("old")
    );
    assert_eq!(host.resident_bytes(), 3);

    host.host_mut().fail_prepare = None;
    host.load_cell(&updated, b"replacement")
        .expect("replacement commits");
    assert_eq!(
        host.host().cells.get(&updated.sector).map(String::as_str),
        Some("replacement")
    );
    assert_eq!(host.resident_cells(), 1);
    assert_eq!(host.resident_bytes(), 11);
}

#[test]
fn failed_eviction_preserves_typed_cell_and_byte_accounting() {
    let (mut host, _) = typed_host(CellDecodeLimits::new(32, 32, 1));
    let m = manifest(1, 0);
    host.load_cell(&m, b"retained").expect("typed load");
    host.host_mut().fail_evict = true;
    assert!(matches!(
        host.evict_cell(m.sector),
        Err(DecodedSceneHostError::Evict("eviction rejected"))
    ));
    assert_eq!(host.resident_bytes(), 8);
    assert!(host.host().cells.contains_key(&m.sector));

    host.host_mut().fail_evict = false;
    host.evict_cell(m.sector).expect("eviction succeeds");
    assert_eq!(host.resident_bytes(), 0);
    assert_eq!(host.resident_cells(), 0);
    assert!(host.host().cells.is_empty());
}

struct MemoryCellProvider(Vec<u8>);

impl CellProvider for MemoryCellProvider {
    type Error = ();

    fn read_cell(
        &self,
        _: &CellReadRequest,
        cancel: CancellationToken,
    ) -> Result<Vec<u8>, Self::Error> {
        if cancel.is_cancelled() {
            return Err(());
        }
        Ok(self.0.clone())
    }
}

#[test]
fn runtime_validates_header_then_decodes_and_commits_typed_cell() {
    let payload = b"typed-world-cell";
    let mut raw = ManifestHeader::from_manifest(manifest(1, 0)).encode();
    let m = manifest(1, (raw.len() + payload.len()) as u64);
    raw = ManifestHeader::from_manifest(m).encode();
    raw.extend_from_slice(payload);
    let demand = DemandSource::new(m.sector, 0, 1, 10).expect("demand");
    let mut coordinator =
        WorldFrameCoordinator::new(MemoryCellProvider(raw), 1).expect("bounded IO runtime");
    let (mut typed, _) = typed_host(CellDecodeLimits::new(64, 64, 1));
    let report = coordinator.plan_frame(
        &[demand],
        &[m],
        &[],
        StreamBudget {
            io_bytes: m.io_bytes,
            max_actions: 1,
        },
        &mut typed,
    );
    assert_eq!(report.submitted, 1);
    assert!(coordinator.poll_io(&mut typed));
    assert_eq!(
        typed.host().cells.get(&m.sector).map(String::as_str),
        Some("typed-world-cell")
    );
    assert_eq!(typed.resident_bytes(), payload.len() as u64);
    assert!(coordinator
        .trace()
        .as_slice()
        .iter()
        .any(|event| matches!(event, TraceEvent::Loaded { sector, .. } if *sector == m.sector)));
}
