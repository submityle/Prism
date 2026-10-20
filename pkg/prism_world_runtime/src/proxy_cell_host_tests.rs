use super::*;
use alloc::vec;
use core::cell::Cell;
use prism_world_gpu::{ProxyInstance, ProxyRegistry, ProxyRegistryError};
use prism_world_stream::{AabbI64, GlobalPos};
use std::error::Error;
use std::panic::{catch_unwind, AssertUnwindSafe};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TestError(&'static str);

impl fmt::Display for TestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

impl Error for TestError {}

#[derive(Clone, Copy, Default)]
enum Behavior {
    #[default]
    Succeed,
    Reject,
    Panic,
}

#[derive(Default)]
struct Host {
    load: Behavior,
    evict: Behavior,
    // Record before returning/panicking to expose non-atomic host changes.
    loads: Vec<(CellManifest, Vec<u8>)>,
    evictions: Vec<SectorId>,
}

impl SceneHost for Host {
    type Error = TestError;

    fn load_cell(&mut self, manifest: &CellManifest, payload: &[u8]) -> Result<(), Self::Error> {
        self.loads.push((*manifest, payload.to_vec()));
        match self.load {
            Behavior::Succeed => Ok(()),
            Behavior::Reject => Err(TestError("load rejected")),
            Behavior::Panic => panic!("load panicked after partial host changes"),
        }
    }

    fn evict_cell(&mut self, sector: SectorId) -> Result<(), Self::Error> {
        self.evictions.push(sector);
        match self.evict {
            Behavior::Succeed => Ok(()),
            Behavior::Reject => Err(TestError("evict rejected")),
            Behavior::Panic => panic!("evict panicked after partial host changes"),
        }
    }
}

fn manifest(x: i64) -> CellManifest {
    CellManifest {
        sector: SectorId::new(0, x, 0, 0),
        bounds: AabbI64::new(GlobalPos::new(x, 0, 0), GlobalPos::new(x + 1, 1, 1))
            .expect("valid bounds"),
        scene_key: 42,
        content_version: 3,
        derived_hash: 17,
        io_bytes: (CELL_HEADER_SIZE + b"opaque payload".len()) as u64,
        data_layers: prism_world_stream::DataLayerMask::EMPTY,
    }
}

fn proxy(id: u32) -> ProxyInstance {
    ProxyInstance::new(id, 2, [1.0, 2.0, 3.0], 0.5, 1.0, 100.0)
}

fn decode(_: &CellManifest, _: &[u8]) -> Result<Vec<ProxyInstance>, TestError> {
    Ok(vec![proxy(7)])
}

#[test]
fn success_preserves_handles_until_eviction_and_does_not_evict_on_drop() {
    let m = manifest(0);
    let other = manifest(1);
    let mut registry = ProxyRegistry::new(4).unwrap();
    let survivor = registry.insert_sector(other.sector, &[proxy(99)]).unwrap()[0];
    let mut host = Host::default();
    let handle;
    {
        let mut adapter = ProxyCellHost::new(&mut host, &mut registry, decode);
        adapter.load_cell(&m, b"untouched").unwrap();
        handle = adapter.registry().sector_handles(m.sector).unwrap()[0];
        assert_eq!(adapter.registry().get(handle), Some(&proxy(7)));
        assert_eq!(adapter.registry().get(survivor), Some(&proxy(99)));
        assert_eq!(adapter.registry().len(), 2);
        assert!(adapter.registry().has_pending_uploads());
    }
    assert_eq!(host.loads, vec![(m, b"untouched".to_vec())]);
    assert!(host.evictions.is_empty());
    assert_eq!(registry.get(handle), Some(&proxy(7)));
    {
        let mut adapter = ProxyCellHost::new(&mut host, &mut registry, decode);
        adapter.evict_cell(m.sector).unwrap();
        assert!(adapter.registry().sector_handles(m.sector).is_none());
        assert!(adapter.registry().get(handle).is_none());
        assert_eq!(adapter.registry().get(survivor), Some(&proxy(99)));
        assert_eq!(adapter.registry().dense_slot(survivor), Some(0));
    }
    assert_eq!(host.evictions, vec![m.sector]);
}

#[test]
fn fn_mut_decoder_receives_exact_inputs_and_keeps_its_state() {
    let mut host = Host::default();
    let mut registry = ProxyRegistry::new(2).unwrap();
    let mut calls = 0;
    {
        let decoder = |m: &CellManifest, payload: &[u8]| {
            calls += 1;
            assert_eq!(*m, manifest((calls - 1) as i64));
            assert_eq!(payload, b"opaque");
            Ok::<_, TestError>(vec![proxy(calls)])
        };
        let mut adapter = ProxyCellHost::new(&mut host, &mut registry, decoder);
        adapter.load_cell(&manifest(0), b"opaque").unwrap();
        adapter.load_cell(&manifest(1), b"opaque").unwrap();
    }
    assert_eq!(calls, 2);
    assert_eq!(registry.proxies(), &[proxy(1), proxy(2)]);
    assert_eq!(host.loads.len(), 2);
}

#[test]
fn decode_error_does_not_touch_registry_or_host() {
    let mut host = Host::default();
    let mut registry = ProxyRegistry::new(1).unwrap();
    {
        let decoder =
            |_: &CellManifest, _: &[u8]| Err::<Vec<ProxyInstance>, _>(TestError("cannot decode"));
        let mut adapter = ProxyCellHost::new(&mut host, &mut registry, decoder);
        assert!(matches!(
            adapter.load_cell(&manifest(0), b"bad"),
            Err(ProxyCellHostError::Decoder(TestError("cannot decode")))
        ));
    }
    assert!(registry.sector_handles(manifest(0).sector).is_none());
    assert!(!registry.has_pending_uploads());
    assert!(host.loads.is_empty());
}

#[test]
fn decoder_panic_never_reaches_registry_or_host() {
    let mut host = Host::default();
    let mut registry = ProxyRegistry::new(1).unwrap();
    let decoder = |_: &CellManifest, _: &[u8]| -> Result<Vec<ProxyInstance>, TestError> {
        panic!("decoder panic")
    };
    {
        let mut adapter = ProxyCellHost::new(&mut host, &mut registry, decoder);
        assert!(catch_unwind(AssertUnwindSafe(|| {
            let _ = adapter.load_cell(&manifest(0), b"bad");
        }))
        .is_err());
    }
    assert!(registry.sector_handles(manifest(0).sector).is_none());
    assert!(!registry.has_pending_uploads());
    assert!(host.loads.is_empty());
}

#[test]
fn invalid_proxies_are_rejected_before_host_load() {
    for invalid in [
        ProxyInstance {
            position: [f32::NAN, 0.0, 0.0],
            ..proxy(7)
        },
        ProxyInstance {
            radius: -1.0,
            ..proxy(7)
        },
        ProxyInstance {
            geometric_error: f32::INFINITY,
            ..proxy(7)
        },
        ProxyInstance {
            max_distance: -1.0,
            ..proxy(7)
        },
    ] {
        let mut host = Host::default();
        let mut registry = ProxyRegistry::new(1).unwrap();
        {
            let decoder = |_: &CellManifest, _: &[u8]| Ok::<_, TestError>(vec![invalid]);
            let mut adapter = ProxyCellHost::new(&mut host, &mut registry, decoder);
            assert!(matches!(
                adapter.load_cell(&manifest(0), b"opaque"),
                Err(ProxyCellHostError::Registry(
                    ProxyRegistryError::InvalidProxy { .. }
                ))
            ));
        }
        assert!(registry.sector_handles(manifest(0).sector).is_none());
        assert!(!registry.has_pending_uploads());
        assert!(host.loads.is_empty());
    }
}

#[test]
fn proxy_capacity_is_rejected_before_host_load() {
    let mut host = Host::default();
    let mut registry = ProxyRegistry::new(0).unwrap();
    {
        let mut adapter = ProxyCellHost::new(&mut host, &mut registry, decode);
        assert!(matches!(
            adapter.load_cell(&manifest(0), b"opaque"),
            Err(ProxyCellHostError::Registry(
                ProxyRegistryError::CapacityExceeded { .. }
            ))
        ));
    }
    assert!(registry.sector_handles(manifest(0).sector).is_none());
    assert!(!registry.has_pending_uploads());
    assert!(host.loads.is_empty());
}

#[test]
fn duplicate_sector_preserves_original_handles_without_host_load() {
    let m = manifest(0);
    let mut host = Host::default();
    let mut registry = ProxyRegistry::new(2).unwrap();
    let handle = registry.insert_sector(m.sector, &[proxy(99)]).unwrap()[0];
    {
        let mut adapter = ProxyCellHost::new(&mut host, &mut registry, decode);
        assert!(matches!(
            adapter.load_cell(&m, b"opaque"),
            Err(ProxyCellHostError::Registry(ProxyRegistryError::SectorAlreadyExists { sector }))
                if sector == m.sector
        ));
    }
    assert_eq!(registry.sector_handles(m.sector), Some(&[handle][..]));
    assert_eq!(registry.get(handle), Some(&proxy(99)));
    assert_eq!(registry.len(), 1);
    assert!(host.loads.is_empty());
}

#[test]
fn duplicate_proxy_ids_are_rejected_before_host_load() {
    let mut host = Host::default();
    let mut registry = ProxyRegistry::new(2).unwrap();
    {
        let decoder = |_: &CellManifest, _: &[u8]| Ok::<_, TestError>(vec![proxy(7), proxy(7)]);
        let mut adapter = ProxyCellHost::new(&mut host, &mut registry, decoder);
        assert!(matches!(
            adapter.load_cell(&manifest(0), b"opaque"),
            Err(ProxyCellHostError::Registry(
                ProxyRegistryError::DuplicateProxyId { .. }
            ))
        ));
    }
    assert!(registry.is_empty());
    assert!(!registry.has_pending_uploads());
    assert!(host.loads.is_empty());
}

#[test]
fn sector_capacity_is_rejected_before_host_load_even_for_empty_cells() {
    let mut host = Host::default();
    let mut registry = ProxyRegistry::with_sector_capacity(2, 1).unwrap();
    registry.insert_sector(manifest(1).sector, &[]).unwrap();
    {
        let decoder = |_: &CellManifest, _: &[u8]| Ok::<_, TestError>(Vec::new());
        let mut adapter = ProxyCellHost::new(&mut host, &mut registry, decoder);
        assert!(matches!(
            adapter.load_cell(&manifest(0), b"opaque"),
            Err(ProxyCellHostError::Registry(
                ProxyRegistryError::SectorCapacityExceeded { capacity: 1 }
            ))
        ));
    }
    assert_eq!(registry.sector_handles(manifest(1).sector), Some(&[][..]));
    assert!(registry.sector_handles(manifest(0).sector).is_none());
    assert!(!registry.has_pending_uploads());
    assert!(host.loads.is_empty());
}

fn assert_load_rollback(behavior: Behavior) {
    let m = manifest(0);
    let other = manifest(1);
    let mut registry = ProxyRegistry::new(2).unwrap();
    let survivor = registry.insert_sector(other.sector, &[proxy(99)]).unwrap()[0];
    let mut host = Host {
        load: behavior,
        ..Host::default()
    };
    {
        let mut adapter = ProxyCellHost::new(&mut host, &mut registry, decode);
        match behavior {
            Behavior::Reject => assert!(matches!(
                adapter.load_cell(&m, b"partial"),
                Err(ProxyCellHostError::Host(TestError("load rejected")))
            )),
            Behavior::Panic => assert!(catch_unwind(AssertUnwindSafe(|| {
                let _ = adapter.load_cell(&m, b"partial");
            }))
            .is_err()),
            Behavior::Succeed => unreachable!(),
        }
        // The guard has run although the adapter itself is still alive.
        assert!(adapter.registry().sector_handles(m.sector).is_none());
        assert_eq!(adapter.registry().len(), 1);
        assert_eq!(adapter.registry().get(survivor), Some(&proxy(99)));
    }
    // Derived compensation must not pretend to undo partial host changes.
    assert_eq!(host.loads, vec![(m, b"partial".to_vec())]);
    assert!(host.evictions.is_empty());
    // The rejected insertion's slot and sector capacity can be reused.
    host.load = Behavior::Succeed;
    {
        let mut adapter = ProxyCellHost::new(&mut host, &mut registry, decode);
        adapter.load_cell(&m, b"retry").unwrap();
    }
    assert_eq!(registry.len(), 2);
}

#[test]
fn host_load_error_rolls_back_only_new_sector() {
    assert_load_rollback(Behavior::Reject);
}

#[test]
fn host_load_panic_unwinds_and_rolls_back_only_new_sector() {
    assert_load_rollback(Behavior::Panic);
}

fn assert_evict_preserves_registry(behavior: Behavior) {
    let m = manifest(0);
    let mut registry = ProxyRegistry::new(2).unwrap();
    let handle = registry.insert_sector(m.sector, &[proxy(7)]).unwrap()[0];
    let other = registry
        .insert_sector(manifest(1).sector, &[proxy(99)])
        .unwrap()[0];
    let dense_slot = registry.dense_slot(handle);
    let pending = registry.has_pending_uploads();
    let mut host = Host {
        evict: behavior,
        ..Host::default()
    };
    {
        let mut adapter = ProxyCellHost::new(&mut host, &mut registry, decode);
        match behavior {
            Behavior::Reject => assert!(matches!(
                adapter.evict_cell(m.sector),
                Err(ProxyCellHostError::Host(TestError("evict rejected")))
            )),
            Behavior::Panic => assert!(catch_unwind(AssertUnwindSafe(|| {
                let _ = adapter.evict_cell(m.sector);
            }))
            .is_err()),
            Behavior::Succeed => unreachable!(),
        }
        assert_eq!(
            adapter.registry().sector_handles(m.sector),
            Some(&[handle][..])
        );
        assert_eq!(adapter.registry().get(handle), Some(&proxy(7)));
        assert_eq!(adapter.registry().get(other), Some(&proxy(99)));
        assert_eq!(adapter.registry().dense_slot(handle), dense_slot);
        assert_eq!(adapter.registry().has_pending_uploads(), pending);
        assert_eq!(adapter.registry().len(), 2);
    }
    assert_eq!(host.evictions, vec![m.sector]);
    assert!(host.loads.is_empty());
}

#[test]
fn host_evict_error_preserves_registry() {
    assert_evict_preserves_registry(Behavior::Reject);
}

#[test]
fn host_evict_panic_preserves_registry() {
    assert_evict_preserves_registry(Behavior::Panic);
}

#[test]
fn empty_sector_is_owned_until_successful_eviction() {
    let m = manifest(0);
    let mut host = Host::default();
    let mut registry = ProxyRegistry::new(0).unwrap();
    {
        let decoder = |_: &CellManifest, _: &[u8]| Ok::<_, TestError>(Vec::new());
        let mut adapter = ProxyCellHost::new(&mut host, &mut registry, decoder);
        adapter.load_cell(&m, b"empty").unwrap();
        assert_eq!(adapter.registry().sector_handles(m.sector), Some(&[][..]));
        assert!(matches!(
            adapter.load_cell(&m, b"duplicate"),
            Err(ProxyCellHostError::Registry(
                ProxyRegistryError::SectorAlreadyExists { .. }
            ))
        ));
        adapter.evict_cell(m.sector).unwrap();
        assert!(adapter.registry().sector_handles(m.sector).is_none());
        assert!(!adapter.registry().has_pending_uploads());
    }
    assert_eq!(host.loads.len(), 1);
    assert_eq!(host.evictions, vec![m.sector]);
}

#[test]
fn empty_sector_is_rolled_back_on_host_error_or_panic() {
    for behavior in [Behavior::Reject, Behavior::Panic] {
        let m = manifest(0);
        let mut host = Host {
            load: behavior,
            ..Host::default()
        };
        let mut registry = ProxyRegistry::new(0).unwrap();
        {
            let decoder = |_: &CellManifest, _: &[u8]| Ok::<_, TestError>(Vec::new());
            let mut adapter = ProxyCellHost::new(&mut host, &mut registry, decoder);
            let outcome = catch_unwind(AssertUnwindSafe(|| adapter.load_cell(&m, b"empty")));
            match behavior {
                Behavior::Reject => {
                    assert!(matches!(outcome, Ok(Err(ProxyCellHostError::Host(_)))))
                }
                Behavior::Panic => assert!(outcome.is_err()),
                Behavior::Succeed => unreachable!(),
            }
            assert!(adapter.registry().sector_handles(m.sector).is_none());
        }
        assert_eq!(host.loads.len(), 1);
        assert!(host.evictions.is_empty());
    }
}

#[test]
fn eviction_without_derived_sector_still_calls_authoritative_host() {
    let m = manifest(0);
    let mut host = Host::default();
    let mut registry = ProxyRegistry::new(0).unwrap();
    {
        let mut adapter = ProxyCellHost::new(&mut host, &mut registry, decode);
        adapter.evict_cell(m.sector).unwrap();
    }
    assert_eq!(host.evictions, vec![m.sector]);
    assert!(!registry.has_pending_uploads());
}

#[test]
fn error_display_and_sources_keep_seam_identity_with_correct_bounds() {
    // Display must not require Debug, Error, or 'static on the inner errors.
    struct DisplayOnly<'a>(&'a str);
    impl fmt::Display for DisplayOnly<'_> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(self.0)
        }
    }
    let message = String::from("borrowed decoder detail");
    let error: ProxyCellHostError<DisplayOnly<'_>, DisplayOnly<'_>> =
        ProxyCellHostError::Decoder(DisplayOnly(&message));
    assert_eq!(
        error.to_string(),
        "cell proxy decoder: borrowed decoder detail"
    );

    type E = ProxyCellHostError<TestError, TestError>;
    let error = E::Decoder(TestError("decode"));
    assert_eq!(
        error.source().unwrap().downcast_ref::<TestError>(),
        Some(&TestError("decode"))
    );
    let error = E::Host(TestError("host"));
    assert_eq!(error.to_string(), "cell scene host: host");
    assert_eq!(
        error.source().unwrap().downcast_ref::<TestError>(),
        Some(&TestError("host"))
    );
    let error = E::Registry(ProxyRegistryError::CapacityExceeded {
        requested: 2,
        available: 1,
    });
    assert!(error.to_string().starts_with("cell proxy registry:"));
    assert!(error.source().unwrap().is::<ProxyRegistryError>());
}

#[test]
fn scene_host_impl_does_not_require_error_display_or_error_trait() {
    struct OpaqueHost;
    impl SceneHost for OpaqueHost {
        type Error = ();
        fn load_cell(&mut self, _: &CellManifest, _: &[u8]) -> Result<(), ()> {
            Ok(())
        }
        fn evict_cell(&mut self, _: SectorId) -> Result<(), ()> {
            Ok(())
        }
    }
    let mut host = OpaqueHost;
    let mut registry = ProxyRegistry::new(1).unwrap();
    let decoder = |_: &CellManifest, _: &[u8]| Ok::<_, ()>(vec![proxy(7)]);
    let mut adapter = ProxyCellHost::new(&mut host, &mut registry, decoder);
    adapter.load_cell(&manifest(0), b"opaque").unwrap();
    adapter.evict_cell(manifest(0).sector).unwrap();
}

struct MemoryProvider {
    sector: SectorId,
    bytes: Vec<u8>,
}

impl CellProvider for MemoryProvider {
    type Error = ();
    fn read_cell(
        &self,
        request: &CellReadRequest,
        cancel: CancellationToken,
    ) -> Result<Vec<u8>, ()> {
        if cancel.is_cancelled() || request.sector() != self.sector {
            return Err(());
        }
        assert_eq!(request.manifest().io_bytes, self.bytes.len() as u64);
        Ok(self.bytes.clone())
    }
}

fn coordinator(m: CellManifest, header: CellManifest) -> WorldFrameCoordinator<MemoryProvider> {
    let mut bytes = ManifestHeader::from_manifest(header).encode();
    bytes.extend_from_slice(b"opaque payload");
    WorldFrameCoordinator::new(
        MemoryProvider {
            sector: m.sector,
            bytes,
        },
        1,
    )
    .unwrap()
}

#[test]
fn real_coordinator_demands_loads_evicts_and_replays_trace() {
    let m = manifest(0);
    let mut coordinator = coordinator(m, m);
    let mut host = Host::default();
    let mut registry = ProxyRegistry::new(1).unwrap();
    let calls = Cell::new(0);
    {
        let decoder = |actual: &CellManifest, payload: &[u8]| {
            assert_eq!(*actual, m);
            assert_eq!(payload, b"opaque payload"); // Header was stripped by the coordinator.
            calls.set(calls.get() + 1);
            Ok::<_, TestError>(vec![proxy(7)])
        };
        let mut adapter = ProxyCellHost::new(&mut host, &mut registry, decoder);
        let budget = StreamBudget {
            io_bytes: m.io_bytes,
            max_actions: 1,
        };
        let report = coordinator.plan_frame(
            &[DemandSource::new(m.sector, 0, 1, 10).unwrap()],
            &[m],
            &[],
            budget,
            &mut adapter,
        );
        assert_eq!((report.actions, report.submitted, report.failed), (1, 1, 0));
        assert_eq!(coordinator.in_flight(), 1);
        assert!(adapter.registry().sector_handles(m.sector).is_none());
        assert_eq!(calls.get(), 0);
        assert!(coordinator.poll_io(&mut adapter));
        assert_eq!(coordinator.in_flight(), 0);
        assert_eq!(calls.get(), 1);
        let handle = adapter.registry().sector_handles(m.sector).unwrap()[0];
        assert_eq!(adapter.registry().get(handle), Some(&proxy(7)));
        assert!(adapter.registry().has_pending_uploads());
        assert!(!coordinator.poll_io(&mut adapter));
        let report = coordinator.plan_frame(&[], &[m], &[LoadedCell::new(m)], budget, &mut adapter);
        assert_eq!((report.actions, report.submitted, report.failed), (1, 0, 0));
        assert!(adapter.registry().get(handle).is_none());
        assert!(adapter.registry().sector_handles(m.sector).is_none());
    }
    assert_eq!(host.loads, vec![(m, b"opaque payload".to_vec())]);
    assert_eq!(host.evictions, vec![m.sector]);
    assert_eq!(coordinator.trace().replay(), Ok(2));
    assert!(matches!(coordinator.trace().as_slice(), [
        TraceEvent::FrameStarted { frame: 1 },
        TraceEvent::LoadSubmitted { frame: 1, sector: submitted, .. },
        TraceEvent::FrameFinished { frame: 1, actions: 1 },
        TraceEvent::Loaded { frame: 1, sector: loaded },
        TraceEvent::FrameStarted { frame: 2 },
        TraceEvent::Evicted { frame: 2, sector: evicted },
        TraceEvent::FrameFinished { frame: 2, actions: 1 },
    ] if *submitted == m.sector && *loaded == m.sector && *evicted == m.sector));
}

#[test]
fn real_coordinator_rejects_bad_header_before_decoder_or_host() {
    let m = manifest(0);
    let mut wrong = m;
    wrong.derived_hash += 1;
    let mut coordinator = coordinator(m, wrong);
    let mut host = Host::default();
    let mut registry = ProxyRegistry::new(1).unwrap();
    {
        let decoder = |_: &CellManifest, _: &[u8]| -> Result<Vec<ProxyInstance>, TestError> {
            panic!("invalid headers must not reach the proxy decoder")
        };
        let mut adapter = ProxyCellHost::new(&mut host, &mut registry, decoder);
        coordinator.plan_frame(
            &[DemandSource::new(m.sector, 0, 1, 10).unwrap()],
            &[m],
            &[],
            StreamBudget {
                io_bytes: m.io_bytes,
                max_actions: 1,
            },
            &mut adapter,
        );
        assert!(coordinator.poll_io(&mut adapter));
        assert!(!adapter.registry().has_pending_uploads());
    }
    assert!(host.loads.is_empty());
    assert!(registry.sector_handles(m.sector).is_none());
    assert_eq!(coordinator.trace().replay(), Ok(1));
    assert!(coordinator.trace().as_slice().iter().any(|event| matches!(
        event,
        TraceEvent::Failed {
            stage: FailureStage::ManifestHeader,
            ..
        }
    )));
}
