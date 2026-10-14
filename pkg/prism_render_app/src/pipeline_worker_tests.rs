use super::*;
use prism_render_driver::{
    DeviceDescriptor, Instance, MultisampleState, PrimitiveState, RequestAdapterOptions,
    ShaderModuleId, VertexState,
};
use prism_render_driver_null::{NullDevice, NullInstance};
use std::sync::atomic::AtomicUsize;
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

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
