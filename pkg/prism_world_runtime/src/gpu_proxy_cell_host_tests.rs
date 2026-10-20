use super::*;

use core::fmt;
use prism_world_gpu::{GpuContext, GpuSectorProxies, ProxyInstance};
use prism_world_stream::{AabbI64, CellManifest, GlobalPos};
use std::error::Error;
use std::panic::{catch_unwind, AssertUnwindSafe};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TestError(&'static str);

impl fmt::Display for TestError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.0)
    }
}

impl Error for TestError {}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
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
    loads: Vec<SectorId>,
    evictions: Vec<SectorId>,
}

impl SceneHost for Host {
    type Error = TestError;

    fn load_cell(&mut self, manifest: &CellManifest, _: &[u8]) -> Result<(), Self::Error> {
        self.loads.push(manifest.sector);
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

fn gpu_context() -> GpuContext {
    let context = GpuContext::try_headless()
        .expect("GPU proxy host tests require successful device acquisition")
        .expect("GPU proxy host tests require a native adapter");
    assert!(
        matches!(
            context.evidence().device_type.as_str(),
            "IntegratedGpu" | "DiscreteGpu"
        ),
        "GPU proxy host tests require physical GPU evidence: {:?}",
        context.evidence()
    );
    context
}

fn manifest(x: i64) -> CellManifest {
    CellManifest {
        sector: SectorId::new(0, x, 0, 0),
        bounds: AabbI64::new(GlobalPos::new(x, 0, 0), GlobalPos::new(x + 1, 1, 1))
            .expect("valid bounds"),
        scene_key: 42,
        content_version: 3,
        derived_hash: 17,
        io_bytes: 1,
        data_layers: prism_world_stream::DataLayerMask::EMPTY,
    }
}

fn proxy(id: u32) -> ProxyInstance {
    ProxyInstance::new(id, 2, [1.0, 2.0, 3.0], 0.5, 1.0, 100.0)
}

fn decode(_: &CellManifest, _: &[u8]) -> Result<Vec<ProxyInstance>, TestError> {
    Ok(vec![proxy(7)])
}

fn decode_payload(_: &CellManifest, payload: &[u8]) -> Result<Vec<ProxyInstance>, TestError> {
    let id = payload
        .first()
        .copied()
        .ok_or(TestError("missing proxy id"))?;
    Ok(vec![proxy(u32::from(id))])
}

#[test]
#[ignore = "requires a native GPU adapter"]
fn successful_load_and_evict_share_one_gpu_sector_lifecycle() {
    let context = gpu_context();
    let sector_manifest = manifest(0);
    let mut host = Host::default();
    let mut gpu = GpuSectorProxies::new(&context, 2, 3).unwrap();

    {
        let mut adapter = GpuProxyCellHost::new(&mut host, &mut gpu, decode);
        adapter.load_cell(&sector_manifest, b"payload").unwrap();
        assert!(adapter
            .gpu()
            .registry()
            .sector_handles(sector_manifest.sector)
            .is_some());
        assert_eq!(adapter.gpu().registry().len(), 1);
        assert!(adapter.gpu().registry().has_pending_uploads());
        assert_eq!(adapter.stage_upload().unwrap().proxies, 1);

        adapter.evict_cell(sector_manifest.sector).unwrap();
        assert!(adapter
            .gpu()
            .registry()
            .sector_handles(sector_manifest.sector)
            .is_none());
        assert_eq!(adapter.stage_upload().unwrap().dense_count, 0);
    }

    assert_eq!(host.loads, vec![sector_manifest.sector]);
    assert_eq!(host.evictions, vec![sector_manifest.sector]);
}

#[test]
#[ignore = "requires a native GPU adapter"]
fn same_sector_reload_reuses_capacity_and_commits_only_after_host_success() {
    let context = gpu_context();
    let sector_manifest = manifest(0);
    let mut host = Host::default();
    let mut gpu = GpuSectorProxies::new(&context, 1, 1).unwrap();

    {
        let mut adapter = GpuProxyCellHost::new(&mut host, &mut gpu, decode_payload);
        adapter.load_cell(&sector_manifest, &[7]).unwrap();
        adapter.stage_upload().unwrap();
    }
    let old = gpu
        .registry()
        .sector_handles(sector_manifest.sector)
        .unwrap()[0];

    {
        let mut adapter = GpuProxyCellHost::new(&mut host, &mut gpu, decode_payload);
        adapter.load_cell(&sector_manifest, &[9]).unwrap();
        assert_eq!(adapter.gpu().registry().get(old), None);
        let replacement = adapter
            .gpu()
            .registry()
            .sector_handles(sector_manifest.sector)
            .unwrap()[0];
        assert_eq!(adapter.gpu().registry().get(replacement), Some(&proxy(9)));
        assert_eq!(adapter.gpu().registry().len(), 1);
        assert!(adapter.gpu().registry().has_pending_uploads());
    }

    assert_eq!(host.loads, vec![sector_manifest.sector; 2]);
}

#[test]
#[ignore = "requires a native GPU adapter"]
fn rejected_or_panicking_reload_preserves_old_gpu_generation_and_upload_state() {
    for behavior in [Behavior::Reject, Behavior::Panic] {
        let context = gpu_context();
        let sector_manifest = manifest(0);
        let mut host = Host::default();
        let mut gpu = GpuSectorProxies::new(&context, 1, 1).unwrap();
        {
            let mut adapter = GpuProxyCellHost::new(&mut host, &mut gpu, decode_payload);
            adapter.load_cell(&sector_manifest, &[7]).unwrap();
            adapter.stage_upload().unwrap();
        }
        let old = gpu
            .registry()
            .sector_handles(sector_manifest.sector)
            .unwrap()[0];
        host.load = behavior;

        {
            let mut adapter = GpuProxyCellHost::new(&mut host, &mut gpu, decode_payload);
            match behavior {
                Behavior::Reject => assert!(matches!(
                    adapter.load_cell(&sector_manifest, &[9]),
                    Err(GpuProxyCellHostError::Host(TestError("load rejected")))
                )),
                Behavior::Panic => assert!(catch_unwind(AssertUnwindSafe(|| {
                    let _ = adapter.load_cell(&sector_manifest, &[9]);
                }))
                .is_err()),
                Behavior::Succeed => unreachable!(),
            }
            assert_eq!(adapter.gpu().registry().get(old), Some(&proxy(7)));
            assert_eq!(
                adapter
                    .gpu()
                    .registry()
                    .sector_handles(sector_manifest.sector),
                Some([old].as_slice())
            );
            assert_eq!(adapter.gpu().registry().len(), 1);
            assert!(!adapter.gpu().registry().has_pending_uploads());
        }

        assert_eq!(host.loads, vec![sector_manifest.sector; 2]);
    }
}

#[test]
#[ignore = "requires a native GPU adapter"]
fn host_load_error_rolls_back_only_the_new_gpu_sector() {
    let context = gpu_context();
    let failed = manifest(0);
    let survivor = manifest(1);
    let mut host = Host {
        load: Behavior::Reject,
        ..Host::default()
    };
    let mut gpu = GpuSectorProxies::new(&context, 2, 3).unwrap();
    let survivor_handle = gpu
        .registry_mut()
        .insert_sector(survivor.sector, &[proxy(99)])
        .unwrap()[0];
    gpu.upload().unwrap();

    {
        let mut adapter = GpuProxyCellHost::new(&mut host, &mut gpu, decode);
        assert!(matches!(
            adapter.load_cell(&failed, b"payload"),
            Err(GpuProxyCellHostError::Host(TestError("load rejected")))
        ));
        assert!(adapter
            .gpu()
            .registry()
            .sector_handles(failed.sector)
            .is_none());
        assert_eq!(
            adapter.gpu().registry().get(survivor_handle),
            Some(&proxy(99))
        );
        assert_eq!(adapter.gpu().registry().len(), 1);
    }

    assert_eq!(host.loads, vec![failed.sector]);
}

#[test]
#[ignore = "requires a native GPU adapter"]
fn host_load_panic_rolls_back_the_new_gpu_sector() {
    let context = gpu_context();
    let failed = manifest(0);
    let mut host = Host {
        load: Behavior::Panic,
        ..Host::default()
    };
    let mut gpu = GpuSectorProxies::new(&context, 1, 3).unwrap();

    {
        let mut adapter = GpuProxyCellHost::new(&mut host, &mut gpu, decode);
        assert!(catch_unwind(AssertUnwindSafe(|| {
            let _ = adapter.load_cell(&failed, b"payload");
        }))
        .is_err());
        assert!(adapter
            .gpu()
            .registry()
            .sector_handles(failed.sector)
            .is_none());
        assert_eq!(adapter.gpu().registry().len(), 0);
    }

    assert_eq!(host.loads, vec![failed.sector]);
}

#[test]
#[ignore = "requires a native GPU adapter"]
fn host_evict_error_and_panic_preserve_the_gpu_sector() {
    for behavior in [Behavior::Reject, Behavior::Panic] {
        let context = gpu_context();
        let sector_manifest = manifest(0);
        let mut host = Host {
            evict: behavior,
            ..Host::default()
        };
        let mut gpu = GpuSectorProxies::new(&context, 1, 3).unwrap();
        gpu.registry_mut()
            .insert_sector(sector_manifest.sector, &[proxy(7)])
            .unwrap();

        {
            let mut adapter = GpuProxyCellHost::new(&mut host, &mut gpu, decode);
            match behavior {
                Behavior::Reject => assert!(matches!(
                    adapter.evict_cell(sector_manifest.sector),
                    Err(GpuProxyCellHostError::Host(TestError("evict rejected")))
                )),
                Behavior::Panic => assert!(catch_unwind(AssertUnwindSafe(|| {
                    let _ = adapter.evict_cell(sector_manifest.sector);
                }))
                .is_err()),
                Behavior::Succeed => unreachable!(),
            }
            assert!(adapter
                .gpu()
                .registry()
                .sector_handles(sector_manifest.sector)
                .is_some());
            assert_eq!(adapter.gpu().registry().len(), 1);
        }

        assert_eq!(host.evictions, vec![sector_manifest.sector]);
    }
}

#[test]
#[ignore = "requires a native GPU adapter"]
fn missing_gpu_sector_fails_before_authoritative_eviction() {
    let context = gpu_context();
    let sector_manifest = manifest(0);
    let mut host = Host::default();
    let mut gpu = GpuSectorProxies::new(&context, 0, 3).unwrap();

    {
        let mut adapter = GpuProxyCellHost::new(&mut host, &mut gpu, decode);
        assert!(matches!(
            adapter.evict_cell(sector_manifest.sector),
            Err(GpuProxyCellHostError::MissingGpuSector(sector)) if sector == sector_manifest.sector
        ));
    }

    assert!(host.evictions.is_empty());
}
