use super::*;
use prism_render_architecture::abi::AbiHash;
use prism_render_architecture::shader_package::{
    DeviceFingerprint, GraphicsBackend, PersistLoadError, PersistedPsoCache, PipelineReadiness,
    PipelineStateHash, PsoCacheKey, ShaderPackageId,
};
use prism_render_driver::{
    Adapter, DeviceDescriptor, Instance, MultisampleState, PrimitiveState, RequestAdapterOptions,
    ShaderModuleId, VertexState,
};
use prism_render_driver_null::{NullDevice, NullInstance};
use std::sync::atomic::AtomicUsize;
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

fn persistence_path(label: &str) -> std::path::PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    std::env::temp_dir().join(format!(
        "prism-pso-{label}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}

fn persistence_fingerprint(driver: u64) -> DeviceFingerprint {
    DeviceFingerprint::new(GraphicsBackend::Other(7), 1, 2, driver, AbiHash([3; 32]))
}

fn adapter_info() -> AdapterInfo {
    NullInstance::new()
        .request_adapter(&RequestAdapterOptions::default())
        .unwrap()
        .info()
        .clone()
}

// Keep the real adapter, package, permutation, pipeline descriptor and opaque
// binary unchanged. Only one compiled material ABI input varies, through the
// same producer used by startup/shutdown (never an injected final AbiHash).
fn assert_abi_source_change_rejects_persisted_cache(
    label: &str,
    revised_sources: MaterialAbiSources<'_>,
) {
    let info = adapter_info();
    let original = pso_device_fingerprint(&info);
    let revised = pso_device_fingerprint_for_abi_sources(&info, revised_sources);
    assert_ne!(original.abi_hash, revised.abi_hash);
    assert_eq!(original.backend, revised.backend);
    assert_eq!(original.vendor_id, revised.vendor_id);
    assert_eq!(original.device_id, revised.device_id);
    assert_eq!(original.driver_version, revised.driver_version);

    let path = persistence_path(label);
    let typed = key(1);
    let cold = cold_key_from_pipeline_key(&typed, ShaderPackageId::new("s5.same-program"));
    // This is a wire/import-boundary fixture, not a driver-produced binary.
    let blob = vec![0x53, 0x35, 0x50, 0x53, 0x4f];
    let mut old_image = PersistedPsoCache::new(original.clone());
    old_image.insert(cold.clone(), blob.clone());
    std::fs::write(&path, old_image.encode()).unwrap();

    let device = device();
    let mut cache = PipelineCache::new();
    // Positive control: the public production fingerprint admits its own file.
    assert_eq!(
        cache.load_persisted_cache_from_path(&path, &original, 1024),
        Ok(1)
    );
    cache.shutdown(&*device);
    let mut cache = PipelineCache::new();
    assert_eq!(
        cache.load_persisted_cache_from_path(&path, &revised, 1024),
        Err(PipelineCachePersistenceError::Decode(
            PersistLoadError::Fingerprint(
                prism_render_architecture::shader_package::FingerprintMismatch::Abi,
            ),
        ))
    );
    assert_eq!(cache.pending_persisted_len(), 0);
    assert_eq!(cache.last_binary_error(), None, "reject before lazy import");

    let mut rebuilt = PersistedPsoCache::new(revised.clone());
    rebuilt.insert(cold.clone(), blob.clone());
    std::fs::write(&path, rebuilt.encode()).unwrap();
    assert_eq!(
        cache.load_persisted_cache_from_path(&path, &revised, 1024),
        Ok(1)
    );
    assert_eq!(cache.persisted.get(&cold), Some(&blob));
    // Rejecting a stale ABI expectation must not clear an admitted image.
    assert_eq!(
        cache.load_persisted_cache_from_path(&path, &original, 1024),
        Err(PipelineCachePersistenceError::Decode(
            PersistLoadError::Fingerprint(
                prism_render_architecture::shader_package::FingerprintMismatch::Abi,
            ),
        ))
    );
    assert_eq!(cache.persisted.get(&cold), Some(&blob));
    cache.shutdown(&*device);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn schema_revision_only_change_rejects_app_persisted_cache() {
    let version = prism_render_material::SURFACE_SCHEMA_VERSION;
    let before = format!("pub const SURFACE_SCHEMA_VERSION: u32 = {version};");
    let after = format!(
        "pub const SURFACE_SCHEMA_VERSION: u32 = {};",
        version.checked_add(1).unwrap()
    );
    assert_eq!(
        MATERIAL_ABI_SOURCES.surface_layout.matches(&before).count(),
        1
    );
    let revised = MATERIAL_ABI_SOURCES
        .surface_layout
        .replacen(&before, &after, 1);
    assert_abi_source_change_rejects_persisted_cache(
        "schema-revision",
        MaterialAbiSources {
            surface_layout: &revised,
            ..MATERIAL_ABI_SOURCES
        },
    );
}

#[test]
fn generated_layout_only_change_rejects_app_persisted_cache() {
    let words = prism_render_material::SURFACE_CORE_WORDS;
    let before = format!("pub const SURFACE_CORE_WORDS: usize = {words};");
    let after = format!("pub const SURFACE_CORE_WORDS: usize = {};", words + 4);
    assert_eq!(
        MATERIAL_ABI_SOURCES.surface_layout.matches(&before).count(),
        1
    );
    let revised = MATERIAL_ABI_SOURCES
        .surface_layout
        .replacen(&before, &after, 1);
    assert_abi_source_change_rejects_persisted_cache(
        "surface-layout",
        MaterialAbiSources {
            surface_layout: &revised,
            ..MATERIAL_ABI_SOURCES
        },
    );
}

#[test]
fn render_class_encoding_only_change_rejects_app_persisted_cache() {
    let before = "    Opaque,\n    OpaqueTwoSided,";
    let after = "    OpaqueTwoSided,\n    Opaque,";
    assert_eq!(
        MATERIAL_ABI_SOURCES.material_record.matches(before).count(),
        1
    );
    let revised = MATERIAL_ABI_SOURCES
        .material_record
        .replacen(before, after, 1);
    assert_abi_source_change_rejects_persisted_cache(
        "render-class-encoding",
        MaterialAbiSources {
            material_record: &revised,
            ..MATERIAL_ABI_SOURCES
        },
    );
}

#[test]
fn response_abi_only_change_rejects_app_persisted_cache() {
    let before = "    base_weight: f32,";
    let after = "    base_weight: vec2<f32>,";
    assert_eq!(MATERIAL_ABI_SOURCES.response_abi.matches(before).count(), 2);
    let revised = MATERIAL_ABI_SOURCES.response_abi.replacen(before, after, 1);
    assert_abi_source_change_rejects_persisted_cache(
        "response-abi",
        MaterialAbiSources {
            response_abi: &revised,
            ..MATERIAL_ABI_SOURCES
        },
    );
}

#[test]
fn persisted_cache_rejects_legacy_version_only_app_fingerprint() {
    let current = pso_device_fingerprint(&adapter_info());
    // Reproduce the exact old producer, with the SAME three numeric versions.
    let legacy = DeviceFingerprint {
        abi_hash: AbiHash::combine(&[
            b"prism-render-app-pso-cache",
            &ARCHITECTURE_VERSION.to_le_bytes(),
            &prism_render_material::MATERIAL_ABI_VERSION.to_le_bytes(),
            &PSO_CACHE_ABI_VERSION.to_le_bytes(),
        ]),
        ..current.clone()
    };
    let mut persisted = PersistedPsoCache::new(legacy);
    persisted.insert(
        cold_key_from_pipeline_key(&key(1), ShaderPackageId::new("s5.same-program")),
        vec![1, 2, 3],
    );
    let path = persistence_path("legacy-version-only");
    std::fs::write(&path, persisted.encode()).unwrap();
    let device = device();
    let mut cache = PipelineCache::new();
    assert_eq!(
        cache.load_persisted_cache_from_path(&path, &current, 1024),
        Err(PipelineCachePersistenceError::Decode(
            PersistLoadError::Fingerprint(
                prism_render_architecture::shader_package::FingerprintMismatch::Abi,
            ),
        ))
    );
    assert_eq!(cache.pending_persisted_len(), 0);
    cache.shutdown(&*device);
    std::fs::remove_file(path).unwrap();
}

fn device() -> Arc<NullDevice> {
    let instance = NullInstance::new();
    let adapter = instance
        .request_adapter(&RequestAdapterOptions::default())
        .unwrap();
    Arc::new(
        instance
            .create_device(&adapter, &DeviceDescriptor::default())
            .unwrap()
            .0,
    )
}

fn descriptor() -> RenderPipelineDescriptor {
    RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: VertexState {
            module: ShaderModuleId::from_parts(0, 0),
            entry_point: "vs".into(),
            buffers: vec![],
        },
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        fragment: None,
    }
}

fn key(class: u32) -> PipelineKey {
    PipelineKey::from_descriptor(
        ExecutionFamily::FusedBuiltin,
        ModuleId(u64::from(class)),
        PermutationIndex::ZERO,
        &descriptor(),
    )
}

struct ControlledCompiler {
    device: Arc<NullDevice>,
    gate: (Mutex<bool>, Condvar),
    started: AtomicUsize,
    active: AtomicUsize,
    peak: AtomicUsize,
    fail_class: Option<u32>,
}

impl ControlledCompiler {
    fn new(device: Arc<NullDevice>, fail_class: Option<u32>) -> Arc<Self> {
        Arc::new(Self {
            device,
            gate: (Mutex::new(false), Condvar::new()),
            started: AtomicUsize::new(0),
            active: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            fail_class,
        })
    }

    fn release(&self) {
        let (gate, wake) = &self.gate;
        *gate.lock().unwrap() = true;
        wake.notify_all();
    }

    fn wait_started(&self, count: usize) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while self.started.load(Ordering::SeqCst) < count {
            assert!(
                Instant::now() < deadline,
                "compiler did not start {count} jobs"
            );
            std::thread::yield_now();
        }
    }
}

impl PipelineCompiler for ControlledCompiler {
    fn compile(&self, key: PipelineKey, descriptor: &RenderPipelineDescriptor) -> RenderPipelineId {
        let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(active, Ordering::SeqCst);
        self.started.fetch_add(1, Ordering::SeqCst);
        let (gate, wake) = &self.gate;
        let guard = gate.lock().unwrap();
        let (guard, timeout) = wake
            .wait_timeout_while(guard, Duration::from_secs(5), |open| !*open)
            .unwrap();
        assert!(!timeout.timed_out() || *guard, "test worker gate timed out");
        drop(guard);
        self.active.fetch_sub(1, Ordering::SeqCst);
        if self.fail_class.map(u64::from) == Some(key.response_module_id.0) {
            panic!("injected compile failure");
        }
        self.device.create_render_pipeline(descriptor)
    }
}

fn request(cache: &mut PipelineCache, device: &NullDevice, class: u32) -> PipelineResolution {
    cache.resolve_or_request(device, key(class), descriptor(), key(99), descriptor())
}

fn drain(cache: &mut PipelineCache, device: &NullDevice) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while cache.pending_len() != 0 {
        cache.pump(device);
        assert!(Instant::now() < deadline, "worker results did not arrive");
        std::thread::yield_now();
    }
}

#[test]
fn finish_pending_admits_successful_worker_results_without_disabling_worker() {
    let device = device();
    let compiler = ControlledCompiler::new(device.clone(), None);
    compiler.release();
    let mut cache = PipelineCache::with_compiler_config(
        compiler,
        PipelineCompileConfig {
            worker_count: 1,
            queue_capacity: 1,
        },
    );
    assert!(matches!(
        request(&mut cache, &device, 1),
        PipelineResolution::Fallback(_)
    ));
    cache.finish_pending(&*device);
    assert!(request(&mut cache, &device, 1).is_ready());
    assert!(cache.has_worker());
    cache.shutdown(&*device);
}

#[test]
fn persisted_load_rejects_fingerprint_mismatch_without_poisoning_pending_entries() {
    let path = persistence_path("fingerprint");
    let expected = persistence_fingerprint(11);
    let mut persisted = PersistedPsoCache::new(persistence_fingerprint(12));
    persisted.insert(
        PsoCacheKey::new(ShaderPackageId::new("pso-test"), 1, PipelineStateHash(2)),
        vec![1, 2, 3],
    );
    std::fs::write(&path, persisted.encode()).expect("write test cache");

    let device = device();
    let mut cache = PipelineCache::new();
    assert_eq!(
        cache.load_persisted_cache_from_path(&path, &expected, 1024),
        Err(PipelineCachePersistenceError::Decode(
            PersistLoadError::Fingerprint(
                prism_render_architecture::shader_package::FingerprintMismatch::Driver,
            ),
        ))
    );
    assert_eq!(cache.pending_persisted_len(), 0);
    cache.shutdown(&*device);
    let _ = std::fs::remove_file(path);
}

#[test]
fn persisted_binary_is_consumed_only_after_typed_mapping_and_import_failure_falls_back() {
    let device = device();
    let compiler = ControlledCompiler::new(device.clone(), None);
    compiler.release();
    let mut cache = PipelineCache::with_compiler(compiler);
    let package = ShaderPackageId::new("pso-test.lazy-import");
    let fallback_package = ShaderPackageId::new("pso-test.lazy-import.fallback");
    cache.register_module_package(ModuleId(1), package).unwrap();
    cache
        .register_module_package(ModuleId(99), fallback_package)
        .unwrap();

    let target = key(1);
    let cold = cache.remember_cold_key(&target).unwrap();
    let mut persisted = PersistedPsoCache::new(persistence_fingerprint(1));
    persisted.insert(cold, vec![1, 2, 3]);
    assert_eq!(cache.load_persisted_cache(persisted), 1);

    let first = cache.resolve_or_request(
        &*device,
        target.clone(),
        descriptor(),
        key(99),
        descriptor(),
    );
    assert!(matches!(first, PipelineResolution::Fallback(_)));
    assert_eq!(cache.pending_persisted_len(), 0);
    assert_eq!(
        cache.last_binary_error(),
        Some(PipelineBinaryCacheError::Unsupported)
    );
    assert_eq!(
        cache.lookup_status(&target),
        Some(PipelineLookupStatus::Miss)
    );

    drain(&mut cache, &device);
    assert!(cache
        .resolve_or_request(&*device, target, descriptor(), key(99), descriptor())
        .is_ready());
    cache.shutdown(&*device);
}

#[test]
fn persisted_load_enforces_encoded_size_budget() {
    let path = persistence_path("size");
    std::fs::write(&path, [0u8; 8]).expect("write oversized test cache");
    let device = device();
    let mut cache = PipelineCache::new();
    assert_eq!(
        cache.load_persisted_cache_from_path(&path, &persistence_fingerprint(1), 7),
        Err(PipelineCachePersistenceError::FileTooLarge {
            size: 8,
            max_bytes: 7,
        })
    );
    cache.shutdown(&*device);
    let _ = std::fs::remove_file(path);
}

#[test]
fn persistence_config_cannot_raise_global_file_budget() {
    assert_eq!(
        PipelineCachePersistenceConfig::new("cache")
            .with_max_bytes(u64::MAX)
            .max_bytes,
        MAX_PERSISTED_PSO_CACHE_BYTES
    );
}

#[test]
fn empty_persisted_save_is_atomic_and_round_trips() {
    let path = persistence_path("roundtrip");
    let device = device();
    let cache = PipelineCache::new();
    assert_eq!(
        cache.save_persisted_cache_to_path(
            &*device,
            persistence_fingerprint(1),
            &path,
            MAX_PERSISTED_PSO_CACHE_BYTES,
        ),
        Ok(0)
    );
    let bytes = std::fs::read(&path).expect("atomic save produced destination");
    assert_eq!(
        PersistedPsoCache::load_for(&bytes, &persistence_fingerprint(1))
            .expect("saved image is readable")
            .len(),
        0
    );
    let _ = std::fs::remove_file(path);
}

#[test]
fn persistence_rejects_non_idle_workers_without_replacing_the_previous_image() {
    let device = device();
    let compiler = ControlledCompiler::new(device.clone(), None);
    let mut cache = PipelineCache::with_compiler_config(
        compiler.clone(),
        PipelineCompileConfig {
            worker_count: 1,
            queue_capacity: 0,
        },
    );
    let path = persistence_path("not-idle");
    let previous = b"previous-valid-image";
    std::fs::write(&path, previous).expect("write previous image");
    request(&mut cache, &device, 1);
    compiler.wait_started(1);

    assert_eq!(
        cache.save_persisted_cache_to_path(
            &*device,
            persistence_fingerprint(1),
            &path,
            MAX_PERSISTED_PSO_CACHE_BYTES,
        ),
        Err(PipelineCachePersistenceError::NotIdle)
    );
    assert_eq!(std::fs::read(&path).unwrap(), previous);

    compiler.release();
    cache.shutdown(&*device);
    let _ = std::fs::remove_file(path);
}

#[test]
fn clear_invalidates_driver_local_package_and_typed_key_mappings() {
    let device = device();
    let mut cache = PipelineCache::new();
    let package = ShaderPackageId::new("pso-test.invalidate");
    let key = key(1);
    cache.register_module_package(ModuleId(1), package).unwrap();
    let cold = cache.remember_cold_key(&key).unwrap();
    assert_eq!(cache.typed_key_for(&cold), Ok(key.clone()));

    cache.clear(&*device);

    assert_eq!(
        cache.cold_key_for(&key),
        Err(PipelineKeyMappingError::UnregisteredModule(ModuleId(1)))
    );
    assert_eq!(
        cache.typed_key_for(&cold),
        Err(PipelineKeyMappingError::UnknownColdKey)
    );
}

#[test]
fn null_backend_reports_unsupported_binary_export_for_resident_pipeline() {
    let device = device();
    let mut cache = PipelineCache::new();
    let descriptor = descriptor();
    let key = key(1);
    cache
        .register_module_package(
            key.response_module_id,
            ShaderPackageId::new("test.pipeline"),
        )
        .unwrap();
    cache.get_or_create(&*device, key, &descriptor);
    let path = persistence_path("unsupported");
    assert_eq!(
        cache.save_persisted_cache_to_path(
            &*device,
            persistence_fingerprint(1),
            &path,
            MAX_PERSISTED_PSO_CACHE_BYTES,
        ),
        Err(PipelineCachePersistenceError::Binary(
            PipelineBinaryCacheError::Unsupported,
        ))
    );
    assert!(!path.exists(), "failed save must not replace destination");
    cache.shutdown(&*device);
}

#[test]
fn bounded_pool_is_parallel_nonblocking_and_deduplicates_saturation_retry() {
    let device = device();
    let compiler = ControlledCompiler::new(device.clone(), None);
    let config = PipelineCompileConfig {
        worker_count: 2,
        queue_capacity: 1,
    };
    let mut cache = PipelineCache::with_compiler_config(compiler.clone(), config);
    assert_eq!(cache.compile_config(), Some(config));
    assert!(matches!(
        request(&mut cache, &device, 1),
        PipelineResolution::Fallback(_)
    ));
    assert!(matches!(
        request(&mut cache, &device, 2),
        PipelineResolution::Fallback(_)
    ));
    compiler.wait_started(2);
    assert_eq!(cache.compiling_len(), 2);
    assert_eq!(cache.in_flight_len(), 2);
    assert_eq!(compiler.peak.load(Ordering::SeqCst), 2);
    assert!(matches!(
        request(&mut cache, &device, 3),
        PipelineResolution::Fallback(_)
    ));
    assert_eq!(cache.waiting_len(), 1);
    let start = Instant::now();
    for _ in 0..16 {
        assert!(matches!(
            request(&mut cache, &device, 4),
            PipelineResolution::Fallback(_)
        ));
        assert!(matches!(
            request(&mut cache, &device, 3),
            PipelineResolution::Fallback(_)
        ));
        assert!(matches!(
            request(&mut cache, &device, 1),
            PipelineResolution::Fallback(_)
        ));
    }
    assert!(
        start.elapsed() < Duration::from_secs(2),
        "saturated render path blocked"
    );
    assert_eq!(cache.pending_len(), 3);
    assert_eq!(cache.waiting_len(), 1);
    assert_eq!(compiler.started.load(Ordering::SeqCst), 2);

    compiler.release();
    drain(&mut cache, &device);
    assert_eq!(compiler.started.load(Ordering::SeqCst), 3);
    assert!(request(&mut cache, &device, 3).is_ready());
    assert!(matches!(
        request(&mut cache, &device, 4),
        PipelineResolution::Fallback(_)
    ));
    drain(&mut cache, &device);
    assert!(request(&mut cache, &device, 4).is_ready());
    assert_eq!(compiler.started.load(Ordering::SeqCst), 4);
    cache.shutdown(&*device);
}

#[test]
fn panic_is_failed_once_and_worker_recovers_for_next_key() {
    let device = device();
    let compiler = ControlledCompiler::new(device.clone(), Some(1));
    compiler.release();
    let mut cache = PipelineCache::with_compiler_config(
        compiler.clone(),
        PipelineCompileConfig {
            worker_count: 1,
            queue_capacity: 1,
        },
    );
    assert!(matches!(
        request(&mut cache, &device, 1),
        PipelineResolution::Fallback(_)
    ));
    drain(&mut cache, &device);
    assert_eq!(cache.failed_len(), 1);
    assert!(matches!(
        request(&mut cache, &device, 1),
        PipelineResolution::Fallback(_)
    ));
    assert_eq!(compiler.started.load(Ordering::SeqCst), 1);
    assert!(matches!(
        request(&mut cache, &device, 2),
        PipelineResolution::Fallback(_)
    ));
    drain(&mut cache, &device);
    assert_eq!(compiler.started.load(Ordering::SeqCst), 2);
    assert!(request(&mut cache, &device, 2).is_ready());
    cache.shutdown(&*device);
}

#[test]
fn shutdown_drains_full_result_channel_and_drop_joins_workers() {
    let device = device();
    let compiler = ControlledCompiler::new(device.clone(), None);
    let mut cache = PipelineCache::with_compiler_config(
        compiler.clone(),
        PipelineCompileConfig {
            worker_count: 2,
            queue_capacity: 1,
        },
    );
    for class in 1..=3 {
        request(&mut cache, &device, class);
    }
    compiler.wait_started(2);
    compiler.release();
    cache.shutdown(&*device);
    assert_eq!(cache.pending_len(), 0);
    assert_eq!(cache.in_flight_len(), 0);
    assert!(!cache.has_worker());
    assert!(compiler.started.load(Ordering::SeqCst) >= 2);
    // No background compiler after shutdown: normal synchronous lookup.
    assert!(request(&mut cache, &device, 4).is_ready());
    cache.clear(&*device);

    let compiler = ControlledCompiler::new(device.clone(), None);
    let mut cache = PipelineCache::with_compiler_config(
        compiler.clone(),
        PipelineCompileConfig {
            worker_count: 2,
            queue_capacity: 0,
        },
    );
    request(&mut cache, &device, 5);
    compiler.wait_started(1);
    compiler.release();
    drop(cache);
}

#[test]
fn cancel_clear_and_re_request_do_not_enqueue_duplicate_jobs() {
    let device = device();
    let compiler = ControlledCompiler::new(device.clone(), None);
    let mut cache = PipelineCache::with_compiler_config(
        compiler.clone(),
        PipelineCompileConfig {
            worker_count: 1,
            queue_capacity: 0,
        },
    );
    request(&mut cache, &device, 1);
    compiler.wait_started(1);
    assert!(!cache.release(&*device, &key(1)));
    assert_eq!(cache.pending_len(), 0);
    assert!(matches!(
        request(&mut cache, &device, 1),
        PipelineResolution::Fallback(_)
    ));
    assert_eq!(cache.pending_len(), 1);
    assert_eq!(cache.in_flight_len(), 1);
    cache.clear(&*device);
    assert!(matches!(
        request(&mut cache, &device, 1),
        PipelineResolution::Fallback(_)
    ));
    compiler.release();
    drain(&mut cache, &device);
    assert_eq!(compiler.started.load(Ordering::SeqCst), 1);
    assert!(request(&mut cache, &device, 1).is_ready());
    cache.shutdown(&*device);
}

#[test]
fn synchronous_compile_overtakes_waiting_and_dispatched_without_duplicate_admission() {
    let device = device();
    let compiler = ControlledCompiler::new(device.clone(), None);
    let mut cache = PipelineCache::with_compiler_config(
        compiler.clone(),
        PipelineCompileConfig {
            worker_count: 1,
            queue_capacity: 1,
        },
    );
    request(&mut cache, &device, 1);
    compiler.wait_started(1);
    request(&mut cache, &device, 2);
    assert_eq!(cache.waiting_len(), 1);
    let foreground = cache.get_or_create(&*device, key(1), &descriptor());
    let waiting = cache.get_or_create(&*device, key(2), &descriptor());
    assert_eq!(cache.waiting_len(), 0);
    assert_eq!(cache.pending_len(), 0);
    compiler.release();
    let deadline = Instant::now() + Duration::from_secs(5);
    while cache.in_flight_len() != 0 {
        cache.pump(&*device);
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
    assert_eq!(compiler.started.load(Ordering::SeqCst), 1);
    assert_eq!(
        cache.get_or_create(&*device, key(1), &descriptor()),
        foreground
    );
    assert_eq!(
        cache.get_or_create(&*device, key(2), &descriptor()),
        waiting
    );
    cache.shutdown(&*device);
}

#[test]
#[should_panic(expected = "pipeline worker count must be nonzero")]
fn worker_count_zero_is_rejected() {
    let device = device();
    let _ = PipelineCache::with_compiler_config(
        ControlledCompiler::new(device, None),
        PipelineCompileConfig {
            worker_count: 0,
            queue_capacity: 1,
        },
    );
}

#[test]
fn typed_key_maps_to_cold_key_without_hash_only_reverse_decoding() {
    let mut cache = PipelineCache::new();
    let key = key(7);
    let package = ShaderPackageId::new("app.pipeline-cache.mapping");

    assert_eq!(
        cache.cold_key_for(&key),
        Err(PipelineKeyMappingError::UnregisteredModule(ModuleId(7)))
    );
    cache
        .register_module_package(ModuleId(7), package)
        .expect("first package registration");
    let cold = cache
        .remember_cold_key(&key)
        .expect("registered module maps to cold key");
    assert_eq!(cold.package, package);
    assert_eq!(cold.permutation_index, 0);
    assert_eq!(cache.typed_key_for(&cold), Ok(key.clone()));
    assert_eq!(
        cache.typed_key_for(&PsoCacheKey::new(package, 999, cold.state)),
        Err(PipelineKeyMappingError::UnknownColdKey)
    );
    assert_eq!(
        cache.register_module_package(ModuleId(7), ShaderPackageId::new("other.package")),
        Err(PipelineKeyMappingError::ConflictingModulePackage {
            module_id: ModuleId(7),
            existing: package,
            requested: ShaderPackageId::new("other.package"),
        })
    );
    assert_eq!(
        cache.register_module_package(ModuleId(8), package),
        Err(PipelineKeyMappingError::PackageMappedToMultipleModules {
            package,
            existing: ModuleId(7),
            requested: ModuleId(8),
        })
    );
}

#[test]
fn unregistered_lookup_exposes_mapping_error_without_dropping_fallback() {
    let device = device();
    let mut cache = PipelineCache::new();
    let result = cache.resolve_or_request(&*device, key(7), descriptor(), key(99), descriptor());

    assert!(result.is_ready());
    assert_eq!(cache.len(), 2);
    assert_eq!(
        cache.last_mapping_error(),
        Some(PipelineKeyMappingError::UnregisteredModule(ModuleId(7)))
    );
}

#[test]
fn warm_runtime_follows_real_pipeline_cache_miss_fallback_and_ready() {
    let device = device();
    let compiler = ControlledCompiler::new(device.clone(), None);
    let mut cache = PipelineCache::with_compiler(compiler.clone());
    let package = ShaderPackageId::new("app.pipeline-cache.lifecycle");
    let fallback_package = ShaderPackageId::new("app.pipeline-cache.lifecycle.fallback");
    cache.register_module_package(ModuleId(1), package).unwrap();
    cache
        .register_module_package(ModuleId(99), fallback_package)
        .unwrap();

    let target = key(1);
    let fallback = key(99);
    let first = cache.resolve_or_request(
        &*device,
        target.clone(),
        descriptor(),
        fallback,
        descriptor(),
    );
    assert!(matches!(first, PipelineResolution::Fallback(_)));
    assert_eq!(
        cache.lookup_status(&target),
        Some(PipelineLookupStatus::Miss)
    );
    assert_eq!(cache.readiness(&target), Ok(PipelineReadiness::Compiling));
    assert_eq!(cache.warm_counters().runtime_misses, 1);
    assert_eq!(cache.warm_counters().compiles_started, 2);

    let repeat = cache.resolve_or_request(
        &*device,
        target.clone(),
        descriptor(),
        key(99),
        descriptor(),
    );
    assert!(matches!(repeat, PipelineResolution::Fallback(_)));
    assert_eq!(
        cache.lookup_status(&target),
        Some(PipelineLookupStatus::Fallback)
    );
    assert_eq!(cache.warm_counters().runtime_misses, 2);

    compiler.release();
    drain(&mut cache, &device);
    let second = cache.resolve_or_request(
        &*device,
        target.clone(),
        descriptor(),
        key(99),
        descriptor(),
    );
    assert!(second.is_ready());
    assert_eq!(
        cache.lookup_status(&target),
        Some(PipelineLookupStatus::Ready)
    );
    assert_eq!(cache.readiness(&target), Ok(PipelineReadiness::Ready));
    assert_eq!(cache.warm_counters().compiles_finished, 2);
    assert_eq!(cache.warm_counters().runtime_misses, 2);
    cache.shutdown(&*device);
}

#[test]
fn lru_uses_explicit_residency_bytes_and_binary_api_uses_driver_capability() {
    let device = device();
    let mut cache = PipelineCache::with_lru_budget(100);
    let first_package = ShaderPackageId::new("app.pipeline-cache.lru.first");
    let second_package = ShaderPackageId::new("app.pipeline-cache.lru.second");
    cache
        .register_module_package(ModuleId(1), first_package)
        .unwrap();
    cache
        .register_module_package(ModuleId(2), second_package)
        .unwrap();
    let first = key(1);
    let second = key(2);

    assert_eq!(
        cache.record_resident_size(&first, 60).unwrap().evicted,
        Vec::new()
    );
    let outcome = cache.record_resident_size(&second, 60).unwrap();
    assert!(outcome.admitted);
    assert_eq!(outcome.evicted.len(), 1);
    assert_eq!(cache.lru().unwrap().used_bytes(), 60);
    assert_eq!(
        cache.export_driver_binary(device.as_ref(), &cache.cold_key_for(&first).unwrap()),
        Err(PipelineBinaryCacheError::NotResident)
    );
    assert_eq!(
        cache.import_driver_binary(
            device.as_ref(),
            &cache.cold_key_for(&second).unwrap(),
            &[1, 2, 3],
        ),
        Err(PipelineBinaryCacheError::Unsupported)
    );
}

#[test]
fn repeated_hot_resolution_does_not_accumulate_cache_references() {
    let device = device();
    let mut cache = PipelineCache::new();
    let requested = key(7);
    let fallback = key(99);
    for _ in 0..64 {
        assert!(cache
            .resolve_or_request(
                device.as_ref(),
                requested.clone(),
                descriptor(),
                fallback.clone(),
                descriptor(),
            )
            .is_ready());
    }
    assert_eq!(cache.cache.refcount(&requested), 1);
    assert_eq!(cache.cache.refcount(&fallback), 1);
    cache.clear(device.as_ref());
}
