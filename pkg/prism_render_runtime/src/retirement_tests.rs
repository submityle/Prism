//! Deterministic CPU retirement tests, not native GPU-completion evidence.
//!
//! NullDevice owns real generational resource tables. The queue wrapper delays
//! completion independently of submission and can fail waits or discard callbacks.
//! Each instance-created device/queue has separate state, including across recovery.

use alloc::sync::{Arc, Weak};
use core::any::Any;
use core::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};

use prism_render_architecture::frame_graph::ResourceLifetime;
use prism_render_architecture::render_fabric::{
    AccessKind, CapabilitySet, DeviceFabric, FeatureCatalog, FeatureDefinition, FeatureId,
    FeatureKind, FeatureVariant, FrameResourceBinding, FrozenRoots, GraphLimits, PassKey,
    PassTemplate, PhysicalResource, QueueClass, RenderDomain, RenderQualityTier, ResourceInput,
    ResourceKey, ResourceLayout, ResourceTemplate, ResourceUse, RuntimeImport, RuntimePassBinding,
    RuntimeResourceHandle, ServiceSet, VariantId,
};
use prism_render_driver as rhi;
use prism_render_driver_null::{NullAdapter, NullDevice, NullInstance, NullQueue, NullSurface};
use prism_render_graph::{
    BufferDesc, CompileError, ImportedBuffer, Lifetime, PassFlags, RenderGraph, TextureDesc,
};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use rhi::{FrameSync, Instance, RenderDevice, RenderQueue};

use crate::{FrameSubmission, RenderRuntime, SubmissionError};

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Allocation {
    Buffer(rhi::BufferId),
    Texture(rhi::TextureId),
    View(rhi::TextureViewId),
}

#[derive(Default)]
struct ResourceLog {
    created: Vec<Allocation>,
    live: Vec<Allocation>,
    destroyed: Vec<Allocation>,
}

#[derive(Clone)]
struct RetirementDevice {
    inner: NullDevice,
    resources: Arc<Mutex<ResourceLog>>,
    loss: Arc<Mutex<Option<rhi::DeviceLostReason>>>,
}

impl RetirementDevice {
    fn created(&self, allocation: Allocation) {
        let mut resources = lock(&self.resources);
        assert!(!resources.live.contains(&allocation));
        resources.created.push(allocation);
        resources.live.push(allocation);
    }

    fn destroying(&self, allocation: Allocation) {
        let mut resources = lock(&self.resources);
        let index = resources
            .live
            .iter()
            .position(|live| *live == allocation)
            .expect("destroy must target a live allocation on its creating device exactly once");
        resources.live.remove(index);
        resources.destroyed.push(allocation);
    }

    fn assert_counts(&self, created: usize, destroyed: usize) {
        let resources = lock(&self.resources);
        assert_eq!(resources.created.len(), created);
        assert_eq!(resources.destroyed.len(), destroyed);
        assert_eq!(resources.live.len(), created - destroyed);
        // Device inspectors retain the resource table after runtime teardown, so
        // zero must come from actual destroy calls, not from dropping the device.
        assert_eq!(self.inner.live_resource_count(), created - destroyed);
    }
}

// The fixture counts graph allocations. Other resource methods still delegate
// to the actual null backend, rather than manufacturing unrelated placeholder IDs.
macro_rules! forward_resources {
    ($($create:ident($descriptor:ty) -> $id:ty, $destroy:ident;)*) => {
        $(
            fn $create(&self, descriptor: &$descriptor) -> $id {
                self.inner.$create(descriptor)
            }
            fn $destroy(&self, id: $id) {
                self.inner.$destroy(id);
            }
        )*
    };
}

impl RenderDevice for RetirementDevice {
    fn capabilities(&self) -> &rhi::DeviceCapabilities {
        self.inner.capabilities()
    }

    fn create_buffer(&self, descriptor: &rhi::BufferDescriptor) -> rhi::BufferId {
        let id = self.inner.create_buffer(descriptor);
        self.created(Allocation::Buffer(id));
        id
    }

    fn create_texture(&self, descriptor: &rhi::TextureDescriptor) -> rhi::TextureId {
        let id = self.inner.create_texture(descriptor);
        self.created(Allocation::Texture(id));
        id
    }

    fn create_texture_view(
        &self,
        texture: rhi::TextureId,
        descriptor: &rhi::TextureViewDescriptor,
    ) -> rhi::TextureViewId {
        assert!(lock(&self.resources)
            .live
            .contains(&Allocation::Texture(texture)));
        let id = self.inner.create_texture_view(texture, descriptor);
        self.created(Allocation::View(id));
        id
    }

    fn destroy_buffer(&self, id: rhi::BufferId) {
        self.destroying(Allocation::Buffer(id));
        self.inner.destroy_buffer(id);
    }

    fn destroy_texture(&self, id: rhi::TextureId) {
        self.destroying(Allocation::Texture(id));
        self.inner.destroy_texture(id);
    }

    fn destroy_texture_view(&self, id: rhi::TextureViewId) {
        self.destroying(Allocation::View(id));
        self.inner.destroy_texture_view(id);
    }

    fn take_device_lost(&self) -> Option<rhi::DeviceLostReason> {
        lock(&self.loss).take()
    }

    forward_resources! {
        create_sampler(rhi::SamplerDescriptor) -> rhi::SamplerId, destroy_sampler;
        create_shader_module(rhi::ShaderModuleDescriptor) -> rhi::ShaderModuleId, destroy_shader_module;
        create_bind_group_layout(rhi::BindGroupLayoutDescriptor) -> rhi::BindGroupLayoutId, destroy_bind_group_layout;
        create_bind_group(rhi::BindGroupDescriptor) -> rhi::BindGroupId, destroy_bind_group;
        create_pipeline_layout(rhi::PipelineLayoutDescriptor) -> rhi::PipelineLayoutId, destroy_pipeline_layout;
        create_render_pipeline(rhi::RenderPipelineDescriptor) -> rhi::RenderPipelineId, destroy_render_pipeline;
        create_compute_pipeline(rhi::ComputePipelineDescriptor) -> rhi::ComputePipelineId, destroy_compute_pipeline;
    }
}

type Callback = Box<dyn FnOnce() + Send + 'static>;

struct QueueProgress {
    submitted: u64,
    completed: u64,
    command_buffers: usize,
    callbacks: Vec<(u64, Callback)>,
    waits: Vec<u64>,
    idle_waits: usize,
    confirm_waits: bool,
    confirm_idle: bool,
}

struct QueueControl(Mutex<QueueProgress>);

impl QueueControl {
    fn new() -> Self {
        Self(Mutex::new(QueueProgress {
            submitted: 0,
            completed: 0,
            command_buffers: 0,
            callbacks: Vec::new(),
            waits: Vec::new(),
            idle_waits: 0,
            confirm_waits: true,
            confirm_idle: true,
        }))
    }

    fn set_wait_results(&self, per_submission: bool, idle: bool) {
        let mut progress = lock(&self.0);
        progress.confirm_waits = per_submission;
        progress.confirm_idle = idle;
    }

    fn complete_through(&self, value: u64) {
        let ready = {
            let mut progress = lock(&self.0);
            assert!(value <= progress.submitted);
            progress.completed = progress.completed.max(value);
            let mut ready = Vec::new();
            for (target, callback) in core::mem::take(&mut progress.callbacks) {
                if target <= progress.completed {
                    ready.push(callback);
                } else {
                    progress.callbacks.push((target, callback));
                }
            }
            ready
        };
        // Never invoke or discard callbacks under the queue lock: callbacks may
        // destroy resources and must not observe an artificially held test lock.
        for callback in ready {
            callback();
        }
    }

    fn discard_callbacks(&self) {
        let callbacks = core::mem::take(&mut lock(&self.0).callbacks);
        drop(callbacks); // Deliberately do NOT invoke the closures.
    }

    fn assert_submissions(&self, count: u64) {
        let progress = lock(&self.0);
        assert_eq!(progress.submitted, count);
        assert_eq!(progress.command_buffers as u64, count);
    }
}

struct DelayedToken {
    queue: Weak<QueueControl>,
    value: u64,
}

struct DelayedQueue {
    inner: NullQueue,
    control: Arc<QueueControl>,
}

impl RenderQueue for DelayedQueue {
    fn write_buffer(&self, buffer: rhi::BufferId, offset: u64, data: &[u8]) {
        self.inner.write_buffer(buffer, offset, data);
    }

    fn write_texture(
        &self,
        destination: rhi::TextureWrite,
        data: &[u8],
        layout: rhi::ImageDataLayout,
    ) {
        self.inner.write_texture(destination, data, layout);
    }

    fn submit(&self, commands: &[rhi::CommandBuffer]) {
        self.inner.submit(commands);
        let mut progress = lock(&self.control.0);
        progress.submitted += 1;
        progress.command_buffers += commands.len();
    }
}

impl FrameSync for DelayedQueue {
    fn submit_tracked(&self, commands: &[rhi::CommandBuffer]) -> rhi::CompletionToken {
        self.submit(commands);
        Box::new(DelayedToken {
            queue: Arc::downgrade(&self.control),
            value: lock(&self.control.0).submitted,
        })
    }

    fn wait_for(&self, token: &(dyn Any + Send + Sync)) -> bool {
        let Some(token) = token.downcast_ref::<DelayedToken>() else {
            return false;
        };
        if !Weak::ptr_eq(&token.queue, &Arc::downgrade(&self.control)) {
            return false;
        }
        let confirm = {
            let mut progress = lock(&self.control.0);
            progress.waits.push(token.value);
            progress.confirm_waits
        };
        if confirm {
            self.control.complete_through(token.value);
        }
        confirm
    }

    fn wait_idle(&self) -> bool {
        let (confirm, value) = {
            let mut progress = lock(&self.control.0);
            progress.idle_waits += 1;
            (progress.confirm_idle, progress.submitted)
        };
        if confirm {
            self.control.complete_through(value);
        }
        confirm
    }

    fn on_submitted_work_done(&self, callback: Callback) {
        {
            let mut progress = lock(&self.control.0);
            let value = progress.submitted;
            if value > progress.completed {
                progress.callbacks.push((value, callback));
                return;
            }
        }
        callback();
    }
}

#[derive(Clone)]
struct DeviceInspector {
    device: RetirementDevice,
    queue: Arc<QueueControl>,
}

#[derive(Clone, Default)]
struct RetirementInstance {
    devices: Arc<Mutex<Vec<DeviceInspector>>>,
}

impl RetirementInstance {
    fn device(&self, index: usize) -> DeviceInspector {
        lock(&self.devices)[index].clone()
    }
}

impl Instance for RetirementInstance {
    type Adapter = NullAdapter;
    type Device = RetirementDevice;
    type Queue = DelayedQueue;
    type Surface = NullSurface;

    fn enumerate_adapters(&self) -> Vec<Self::Adapter> {
        NullInstance::new().enumerate_adapters()
    }

    fn request_adapter(&self, options: &rhi::RequestAdapterOptions) -> Option<Self::Adapter> {
        NullInstance::new().request_adapter(options)
    }

    fn request_adapter_for_surface(
        &self,
        options: &rhi::RequestAdapterOptions,
        surface: &Self::Surface,
    ) -> Option<Self::Adapter> {
        NullInstance::new().request_adapter_for_surface(options, surface)
    }

    fn create_device(
        &self,
        adapter: &Self::Adapter,
        descriptor: &rhi::DeviceDescriptor,
    ) -> Result<(Self::Device, Self::Queue), rhi::DeviceError> {
        let (inner, queue) = NullInstance::new().create_device(adapter, descriptor)?;
        let device = RetirementDevice {
            inner,
            resources: Arc::new(Mutex::new(ResourceLog::default())),
            loss: Arc::new(Mutex::new(None)),
        };
        let control = Arc::new(QueueControl::new());
        lock(&self.devices).push(DeviceInspector {
            device: device.clone(),
            queue: Arc::clone(&control),
        });
        Ok((
            device,
            DelayedQueue {
                inner: queue,
                control,
            },
        ))
    }

    fn create_surface<W>(
        &self,
        window: Arc<W>,
        device: &Self::Device,
        queue: &Self::Queue,
    ) -> Result<Self::Surface, rhi::SurfaceError>
    where
        W: HasWindowHandle + HasDisplayHandle + Send + Sync + 'static,
    {
        NullInstance::new().create_surface(window, &device.inner, &queue.inner)
    }
}

fn runtime(max_in_flight: usize) -> (RenderRuntime<RetirementInstance>, RetirementInstance) {
    let instance = RetirementInstance::default();
    let runtime = RenderRuntime::with_frames_in_flight(
        instance.clone(),
        rhi::Extent3d::new_2d(32, 16),
        max_in_flight,
    )
    .expect("null runtime creation");
    (runtime, instance)
}

// Two textures, two default views, two buffers; each lifetime class is represented.
const FRAME_ALLOCATIONS: usize = 6;
const FRAME_RECORDINGS: usize = 2;

fn frame(calls: &Arc<AtomicUsize>) -> RenderGraph {
    let mut graph = RenderGraph::new(rhi::Extent3d::new_2d(32, 16));
    let compute_calls = Arc::clone(calls);
    graph.add_compute_pass("owned buffers", PassFlags::SIDE_EFFECT, move |b| {
        let desc = BufferDesc::new(256, rhi::BufferUsages::STORAGE);
        let transient = b.create_buffer("frame buffer", desc.clone());
        let history = b.create_persistent_buffer("history buffer", desc);
        b.write_storage_buffer(transient);
        b.write_storage_buffer(history);
        move |ctx| {
            compute_calls.fetch_add(1, Ordering::SeqCst);
            ctx.dispatch(1, 1, 1);
        }
    });
    let raster_calls = Arc::clone(calls);
    graph.add_raster_pass("owned targets", PassFlags::SIDE_EFFECT, move |b| {
        let desc = TextureDesc::color(rhi::TextureFormat::Rgba8Unorm);
        let transient = b.create_texture("frame target", desc.clone());
        let history = b.create_persistent_texture("history target", desc);
        b.color_attachment(transient);
        b.color_attachment(history);
        move |ctx| {
            raster_calls.fetch_add(1, Ordering::SeqCst);
            ctx.draw(0..3, 0..1);
        }
    });
    graph
}

#[test]
fn fabric_compile_prepare_and_runtime_submission_share_one_frozen_selection() {
    const FEATURE: FeatureId = FeatureId(930);
    const VARIANT: VariantId = VariantId(930);
    const PASS: PassKey = PassKey(930);
    const SOURCE: ResourceKey = ResourceKey::of(FEATURE, 0);
    const TARGET: ResourceKey = ResourceKey::of(FEATURE, 1);
    const FRAME: u64 = 7;

    let accesses: &'static [ResourceUse] = Box::leak(
        vec![
            ResourceUse {
                input: ResourceInput::Initial,
                key: SOURCE,
                access: AccessKind::TransferRead,
                scope: prism_render_architecture::render_fabric::ResourceScope::Whole,
                frame_lag: None,
            },
            ResourceUse {
                input: ResourceInput::Automatic,
                key: TARGET,
                access: AccessKind::TransferWrite,
                scope: prism_render_architecture::render_fabric::ResourceScope::Whole,
                frame_lag: None,
            },
        ]
        .into_boxed_slice(),
    );
    let passes: &'static [PassTemplate] = Box::leak(
        vec![PassTemplate {
            id: PASS,
            accesses,
            queue: QueueClass::Transfer,
            depends_on: &[],
        }]
        .into_boxed_slice(),
    );
    let variants: &'static [FeatureVariant] = Box::leak(
        vec![FeatureVariant {
            id: VARIANT,
            rank: 0,
            min_quality: RenderQualityTier::Low,
            required_caps: CapabilitySet::EMPTY,
            requires_services_extra: ServiceSet::EMPTY,
            passes,
            shader_permutation: None,
        }]
        .into_boxed_slice(),
    );
    let resources: &'static [ResourceKey] = Box::leak(vec![SOURCE, TARGET].into_boxed_slice());
    let definitions: &'static [FeatureDefinition] = Box::leak(
        vec![FeatureDefinition {
            id: FEATURE,
            kind: FeatureKind::Subsystem,
            domain: Some(RenderDomain::Surface),
            requires_services: ServiceSet::EMPTY,
            variants,
        }]
        .into_boxed_slice(),
    );
    let catalog = FeatureCatalog::new(definitions, resources).unwrap();
    let roots = FrozenRoots::from_build_profile(&catalog, &[FEATURE], &[]).unwrap();
    let fabric = DeviceFabric::new(
        catalog,
        roots,
        CapabilitySet::WEBGPU_BASELINE,
        RenderQualityTier::Low,
    )
    .unwrap();
    assert_eq!(
        fabric.selection().feature(FEATURE).unwrap().variant_id(),
        VARIANT
    );

    let (runtime, instance) = runtime(4);
    let source_graph = {
        let mut graph = runtime.new_graph();
        graph.add_compute_pass("fabric source", PassFlags::SIDE_EFFECT, |builder| {
            let source = builder.create_persistent_buffer(
                "fabric source buffer",
                BufferDesc::new(
                    64,
                    rhi::BufferUsages::COPY_SRC | rhi::BufferUsages::COPY_DST,
                ),
            );
            builder.write_storage_buffer(source);
            move |commands| {
                commands.dispatch(1, 1, 1);
            }
        });
        graph
    };
    let producer = runtime.execute_owned(source_graph).unwrap();
    let source_id = producer
        .allocations()
        .buffers()
        .iter()
        .find_map(|(_, lifetime, id)| (*lifetime == Lifetime::Persistent).then_some(*id))
        .expect("producer must publish a persistent source buffer");
    instance.device(0).queue.complete_through(1);

    let buffer_layout = ResourceLayout::Buffer {
        bytes: 64,
        usage: rhi::BufferUsages::COPY_SRC | rhi::BufferUsages::COPY_DST,
    };
    let source_template = ResourceTemplate {
        key: SOURCE,
        lifetime: ResourceLifetime::Imported,
        layout: buffer_layout,
        alignment: 16,
    };
    let target_template = ResourceTemplate {
        key: TARGET,
        lifetime: ResourceLifetime::Transient,
        layout: buffer_layout,
        alignment: 16,
    };
    let declaration = fabric
        .compile(
            &[source_template, target_template],
            FRAME,
            &[FrameResourceBinding {
                key: SOURCE,
                frame: FRAME,
                version: 1,
                physical: PhysicalResource::Buffer {
                    id: source_id,
                    offset: 0,
                },
                layout: buffer_layout,
                alignment: 16,
            }],
            &[],
            GraphLimits::default(),
        )
        .unwrap();
    let source_resource = declaration.resource_id(SOURCE, FRAME).unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let recorder_calls = Arc::clone(&calls);
    let runtime_graph = declaration
        .prepare_runtime(
            rhi::Extent3d::new_2d(1, 1),
            &[RuntimeImport::Buffer {
                resource: source_resource,
                imported: ImportedBuffer {
                    buffer: source_id,
                    entry_state: rhi::BufferState::initial(),
                },
            }],
            vec![RuntimePassBinding::new(PASS, move |context| {
                let RuntimeResourceHandle::Buffer { handle: source, .. } =
                    context.accesses()[0].handle
                else {
                    panic!("source access must lower as a buffer");
                };
                let RuntimeResourceHandle::Buffer { handle: target, .. } =
                    context.accesses()[1].handle
                else {
                    panic!("target access must lower as a buffer");
                };
                recorder_calls.fetch_add(1, Ordering::SeqCst);
                context
                    .commands()
                    .copy_buffer_to_buffer(source, 0, target, 0, 64);
            })],
        )
        .unwrap();
    let owner = runtime
        .execute_prepared_owned(
            runtime_graph.into_prepared(),
            core::slice::from_ref(&producer),
        )
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    drop(producer);
    instance.device(0).queue.complete_through(2);
    assert!(owner.is_complete());
    assert!(runtime.wait_idle());
    assert_eq!(runtime.frames_in_flight(), 0);
    drop(owner);
    instance.device(0).device.assert_counts(2, 2);
}

fn history_frame(
    calls: &Arc<AtomicUsize>,
    owner: &FrameSubmission<RetirementDevice>,
) -> RenderGraph {
    let mut graph = frame(calls);
    let buffer = owner
        .allocations()
        .buffers()
        .iter()
        .find_map(|(_, lifetime, id)| (*lifetime == Lifetime::Persistent).then_some(*id))
        .expect("producer has a persistent buffer");
    let imported = graph.import_buffer(
        "previous submission history",
        BufferDesc::new(256, rhi::BufferUsages::STORAGE),
        ImportedBuffer {
            buffer,
            entry_state: rhi::BufferState::initial(),
        },
    );
    let calls = Arc::clone(calls);
    graph.add_compute_pass("consume history", PassFlags::SIDE_EFFECT, move |b| {
        b.read_storage_buffer(imported);
        move |ctx| {
            calls.fetch_add(1, Ordering::SeqCst);
            assert_eq!(ctx.buffer(imported), buffer);
            ctx.dispatch(1, 1, 1);
        }
    });
    graph
}

#[test]
fn prepared_submission_realizes_and_records_the_validated_plan_exactly_once() {
    let (runtime, instance) = runtime(2);
    let inspector = instance.device(0);
    let calls = Arc::new(AtomicUsize::new(0));
    let prepared = frame(&calls).prepare().unwrap();
    inspector.device.assert_counts(0, 0);
    inspector.queue.assert_submissions(0);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let owner = runtime.execute_prepared_owned(prepared, &[]).unwrap();
    inspector.device.assert_counts(FRAME_ALLOCATIONS, 0);
    inspector.queue.assert_submissions(1);
    assert_eq!(calls.load(Ordering::SeqCst), FRAME_RECORDINGS);
    drop(owner);
    inspector.queue.complete_through(1);
    inspector
        .device
        .assert_counts(FRAME_ALLOCATIONS, FRAME_ALLOCATIONS);
}

#[test]
fn consumer_retains_completed_history_until_its_own_actual_completion() {
    let (runtime, instance) = runtime(3);
    let inspector = instance.device(0);
    let calls = Arc::new(AtomicUsize::new(0));
    let source = runtime.execute_owned(frame(&calls)).unwrap();
    inspector.queue.complete_through(1);
    assert!(source.is_complete());
    let consumer = runtime
        .execute_prepared_owned(
            history_frame(&calls, &source).prepare().unwrap(),
            core::slice::from_ref(&source),
        )
        .unwrap();
    assert_eq!(consumer.allocations().buffers().len(), 2);
    drop(source);
    inspector.device.assert_counts(2 * FRAME_ALLOCATIONS, 0);
    assert!(!consumer.is_complete());
    inspector.queue.complete_through(2);
    assert!(consumer.is_complete());
    // Keeping the completed consumer does not pin its now-unused predecessor.
    inspector
        .device
        .assert_counts(2 * FRAME_ALLOCATIONS, FRAME_ALLOCATIONS);
    drop(consumer);
    inspector
        .device
        .assert_counts(2 * FRAME_ALLOCATIONS, 2 * FRAME_ALLOCATIONS);
    assert_eq!(calls.load(Ordering::SeqCst), 2 * FRAME_RECORDINGS + 1);
}

#[test]
fn shared_history_and_duplicate_import_owners_retire_after_the_last_consumer() {
    let (runtime, instance) = runtime(4);
    let inspector = instance.device(0);
    let calls = Arc::new(AtomicUsize::new(0));
    let source = runtime.execute_owned(frame(&calls)).unwrap();
    inspector.queue.complete_through(1);
    let first = runtime
        .execute_prepared_owned(
            history_frame(&calls, &source).prepare().unwrap(),
            &[source.clone(), source.clone()],
        )
        .unwrap();
    let second = runtime
        .execute_prepared_owned(
            history_frame(&calls, &source).prepare().unwrap(),
            core::slice::from_ref(&source),
        )
        .unwrap();
    drop(source);
    drop(first);
    inspector.queue.complete_through(2);
    inspector
        .device
        .assert_counts(3 * FRAME_ALLOCATIONS, FRAME_ALLOCATIONS);
    inspector.queue.complete_through(3);
    inspector
        .device
        .assert_counts(3 * FRAME_ALLOCATIONS, 2 * FRAME_ALLOCATIONS);
    drop(second);
    inspector
        .device
        .assert_counts(3 * FRAME_ALLOCATIONS, 3 * FRAME_ALLOCATIONS);
}

#[test]
fn completed_history_chain_does_not_pin_all_preceding_frames() {
    let (runtime, instance) = runtime(2);
    let inspector = instance.device(0);
    let calls = Arc::new(AtomicUsize::new(0));
    let mut source = runtime.execute_owned(frame(&calls)).unwrap();
    inspector.queue.complete_through(1);
    for value in 2..=33 {
        let next = runtime
            .execute_prepared_owned(
                history_frame(&calls, &source).prepare().unwrap(),
                core::slice::from_ref(&source),
            )
            .unwrap();
        drop(source);
        inspector.device.assert_counts(
            value as usize * FRAME_ALLOCATIONS,
            (value as usize - 2) * FRAME_ALLOCATIONS,
        );
        inspector.queue.complete_through(value);
        inspector.device.assert_counts(
            value as usize * FRAME_ALLOCATIONS,
            (value as usize - 1) * FRAME_ALLOCATIONS,
        );
        source = next;
    }
    drop(source);
    inspector
        .device
        .assert_counts(33 * FRAME_ALLOCATIONS, 33 * FRAME_ALLOCATIONS);
}

#[test]
fn foreign_device_owner_is_rejected_before_pacing_or_recording() {
    let (producer_runtime, producer_instance) = runtime(1);
    let (consumer_runtime, consumer_instance) = runtime(1);
    let producer = producer_instance.device(0);
    let consumer = consumer_instance.device(0);
    let calls = Arc::new(AtomicUsize::new(0));
    let source = producer_runtime.execute_owned(frame(&calls)).unwrap();
    let own = consumer_runtime.execute_owned(frame(&calls)).unwrap();
    assert_eq!(source.device_generation(), own.device_generation());
    consumer.queue.set_wait_results(false, false);
    let prepared = history_frame(&calls, &source).prepare().unwrap();
    assert!(matches!(
        consumer_runtime.execute_prepared_owned(prepared, &[own.clone(), source.clone()]),
        Err(SubmissionError::ForeignImportOwner { index: 1 })
    ));
    consumer.device.assert_counts(FRAME_ALLOCATIONS, 0);
    producer.device.assert_counts(FRAME_ALLOCATIONS, 0);
    assert!(lock(&consumer.queue.0).waits.is_empty());
    assert_eq!(consumer_runtime.frames_in_flight(), 1);
    assert_eq!(consumer_runtime.next_completion_value(), 2);
    assert_eq!(calls.load(Ordering::SeqCst), 2 * FRAME_RECORDINGS);
    consumer.queue.assert_submissions(1);
    producer.queue.assert_submissions(1);
    consumer.queue.set_wait_results(true, true);
}

#[test]
fn recovered_device_rejects_old_history_before_realization() {
    let (mut runtime, instance) = runtime(1);
    let old = instance.device(0);
    let calls = Arc::new(AtomicUsize::new(0));
    let source = runtime.execute_owned(frame(&calls)).unwrap();
    assert_eq!(runtime.recover_device().unwrap().generation, 1);
    let new = instance.device(1);
    assert!(matches!(
        runtime.execute_prepared_owned(
            history_frame(&calls, &source).prepare().unwrap(),
            core::slice::from_ref(&source),
        ),
        Err(SubmissionError::ForeignImportOwner { index: 0 })
    ));
    new.device.assert_counts(0, 0);
    new.queue.assert_submissions(0);
    assert!(lock(&new.queue.0).waits.is_empty());
    assert_eq!(runtime.frames_in_flight(), 0);
    assert_eq!(runtime.next_completion_value(), 1);
    assert_eq!(calls.load(Ordering::SeqCst), FRAME_RECORDINGS);
    drop(source);
    old.device
        .assert_counts(FRAME_ALLOCATIONS, FRAME_ALLOCATIONS);
}

#[test]
fn prepared_rejection_preserves_history_and_has_no_allocation_recording_or_submission() {
    let (runtime, instance) = runtime(1);
    let inspector = instance.device(0);
    let calls = Arc::new(AtomicUsize::new(0));
    let source = runtime.execute_owned(frame(&calls)).unwrap();
    inspector.queue.set_wait_results(false, false);
    assert!(matches!(
        runtime.execute_prepared_owned(
            history_frame(&calls, &source).prepare().unwrap(),
            core::slice::from_ref(&source),
        ),
        Err(SubmissionError::CompletionUnconfirmed)
    ));
    assert_eq!(lock(&inspector.queue.0).waits, vec![1]);
    assert_eq!(runtime.next_completion_value(), 2);
    runtime.submitted_value.store(u64::MAX, Ordering::Release);
    assert!(matches!(
        runtime.execute_prepared_owned(
            history_frame(&calls, &source).prepare().unwrap(),
            core::slice::from_ref(&source),
        ),
        Err(SubmissionError::CompletionValueExhausted)
    ));
    assert_eq!(lock(&inspector.queue.0).waits, vec![1]);
    assert_eq!(runtime.frames_in_flight(), 1);
    assert_eq!(calls.load(Ordering::SeqCst), FRAME_RECORDINGS);
    inspector.device.assert_counts(FRAME_ALLOCATIONS, 0);
    inspector.queue.assert_submissions(1);
    inspector.queue.complete_through(1);
    assert!(source.is_complete());
    drop(source);
    inspector
        .device
        .assert_counts(FRAME_ALLOCATIONS, FRAME_ALLOCATIONS);
    inspector.queue.set_wait_results(true, true);
}

#[test]
fn discarded_consumer_callback_cannot_destroy_completed_import_or_leak_device_arc() {
    let (runtime, instance) = runtime(3);
    let inspector = instance.device(0);
    let weak_device = Arc::downgrade(&runtime.device);
    let calls = Arc::new(AtomicUsize::new(0));
    let source = runtime.execute_owned(frame(&calls)).unwrap();
    inspector.queue.complete_through(1);
    let consumer = runtime
        .execute_prepared_owned(
            history_frame(&calls, &source).prepare().unwrap(),
            core::slice::from_ref(&source),
        )
        .unwrap();
    drop(source);
    drop(consumer);
    inspector.queue.discard_callbacks();
    // A completed producer can still be in use by an unconfirmed consumer.
    inspector.device.assert_counts(2 * FRAME_ALLOCATIONS, 0);
    drop(runtime);
    assert!(weak_device.upgrade().is_none());
    inspector.device.assert_counts(2 * FRAME_ALLOCATIONS, 0);
}

#[test]
fn discarded_multilevel_history_callbacks_do_not_destroy_or_leak_owner_arcs() {
    let (runtime, instance) = runtime(4);
    let inspector = instance.device(0);
    let weak_device = Arc::downgrade(&runtime.device);
    let calls = Arc::new(AtomicUsize::new(0));
    let first = runtime.execute_owned(frame(&calls)).unwrap();
    let second = runtime
        .execute_prepared_owned(
            history_frame(&calls, &first).prepare().unwrap(),
            core::slice::from_ref(&first),
        )
        .unwrap();
    let third = runtime
        .execute_prepared_owned(
            history_frame(&calls, &second).prepare().unwrap(),
            core::slice::from_ref(&second),
        )
        .unwrap();
    drop(first);
    drop(second);
    drop(third);
    inspector.queue.discard_callbacks();
    inspector.device.assert_counts(3 * FRAME_ALLOCATIONS, 0);
    assert_eq!(runtime.completed_value(), 0);
    drop(runtime);
    assert!(weak_device.upgrade().is_none());
    inspector.device.assert_counts(3 * FRAME_ALLOCATIONS, 0);
}

#[test]
fn early_owner_drop_keeps_allocations_until_the_completion_callback_runs() {
    let (runtime, instance) = runtime(2);
    let inspector = instance.device(0);
    let calls = Arc::new(AtomicUsize::new(0));
    let owner = runtime.execute_owned(frame(&calls)).unwrap();
    assert_eq!(owner.completion_value(), 1);
    assert_eq!(owner.device_generation(), 0);
    assert!(!owner.is_complete());
    assert_eq!(calls.load(Ordering::SeqCst), FRAME_RECORDINGS);
    assert_eq!(owner.allocations().textures().len(), 2);
    assert_eq!(owner.allocations().buffers().len(), 2);
    inspector.queue.assert_submissions(1);
    inspector.device.assert_counts(FRAME_ALLOCATIONS, 0);
    drop(owner);
    inspector.device.assert_counts(FRAME_ALLOCATIONS, 0);
    assert_eq!(runtime.completed_value(), 0);
    assert_eq!(lock(&inspector.queue.0).callbacks.len(), 1);

    inspector.queue.complete_through(1);
    assert_eq!(runtime.completed_value(), 1);
    inspector
        .device
        .assert_counts(FRAME_ALLOCATIONS, FRAME_ALLOCATIONS);
    assert!(lock(&inspector.queue.0).callbacks.is_empty());
    inspector.queue.complete_through(1);
    inspector
        .device
        .assert_counts(FRAME_ALLOCATIONS, FRAME_ALLOCATIONS);
}

#[test]
fn execute_and_execute_tracked_retire_only_their_completed_submissions() {
    let (runtime, instance) = runtime(2);
    let inspector = instance.device(0);
    let calls = Arc::new(AtomicUsize::new(0));
    runtime.execute(frame(&calls)).unwrap();
    assert_eq!(runtime.execute_tracked(frame(&calls)).unwrap(), 2);
    inspector.device.assert_counts(2 * FRAME_ALLOCATIONS, 0);
    inspector.queue.assert_submissions(2);

    inspector.queue.complete_through(1);
    assert_eq!(runtime.completed_value(), 1);
    inspector
        .device
        .assert_counts(2 * FRAME_ALLOCATIONS, FRAME_ALLOCATIONS);
    assert_eq!(lock(&inspector.queue.0).callbacks.len(), 1);
    inspector.queue.complete_through(2);
    assert_eq!(runtime.completed_value(), 2);
    inspector
        .device
        .assert_counts(2 * FRAME_ALLOCATIONS, 2 * FRAME_ALLOCATIONS);
}

#[test]
fn explicit_cloned_owner_preserves_transients_and_history_past_completion() {
    let (runtime, instance) = runtime(2);
    let inspector = instance.device(0);
    let owner = runtime
        .execute_owned(frame(&Arc::new(AtomicUsize::new(0))))
        .unwrap();
    let retained = owner.clone();
    for lifetime in [Lifetime::Transient, Lifetime::Persistent] {
        assert!(owner
            .allocations()
            .textures()
            .iter()
            .any(|(_, class, _)| *class == lifetime));
        assert!(owner
            .allocations()
            .buffers()
            .iter()
            .any(|(_, class, _)| *class == lifetime));
    }
    inspector.queue.complete_through(1);
    assert!(owner.is_complete());
    assert!(retained.is_complete());
    inspector.device.assert_counts(FRAME_ALLOCATIONS, 0);
    assert!(runtime.wait_idle());
    assert_eq!(runtime.frames_in_flight(), 0);
    inspector.device.assert_counts(FRAME_ALLOCATIONS, 0);
    drop(owner);
    inspector.device.assert_counts(FRAME_ALLOCATIONS, 0);
    assert!(!retained.allocations().is_empty());
    drop(retained);
    inspector
        .device
        .assert_counts(FRAME_ALLOCATIONS, FRAME_ALLOCATIONS);
}

#[test]
fn runtime_drop_drives_the_installed_queue_barrier_and_retires_unowned_frames() {
    let (runtime, instance) = runtime(2);
    let inspector = instance.device(0);
    runtime
        .execute(frame(&Arc::new(AtomicUsize::new(0))))
        .unwrap();
    assert_eq!(lock(&inspector.queue.0).idle_waits, 0);
    inspector.device.assert_counts(FRAME_ALLOCATIONS, 0);
    drop(runtime);
    assert_eq!(lock(&inspector.queue.0).idle_waits, 1);
    assert_eq!(lock(&inspector.queue.0).completed, 1);
    assert!(lock(&inspector.queue.0).callbacks.is_empty());
    inspector
        .device
        .assert_counts(FRAME_ALLOCATIONS, FRAME_ALLOCATIONS);
}

#[test]
fn explicit_owner_outlives_runtime_drop_and_releases_on_its_creating_device() {
    let (runtime, instance) = runtime(2);
    let inspector = instance.device(0);
    let owner = runtime
        .execute_owned(frame(&Arc::new(AtomicUsize::new(0))))
        .unwrap();
    drop(runtime);
    assert_eq!(lock(&inspector.queue.0).idle_waits, 1);
    assert!(owner.is_complete());
    inspector.device.assert_counts(FRAME_ALLOCATIONS, 0);
    drop(owner);
    inspector
        .device
        .assert_counts(FRAME_ALLOCATIONS, FRAME_ALLOCATIONS);
}

#[test]
fn failed_shutdown_wait_does_not_authorize_destruction() {
    let (runtime, instance) = runtime(2);
    let inspector = instance.device(0);
    inspector.queue.set_wait_results(false, false);
    runtime
        .execute(frame(&Arc::new(AtomicUsize::new(0))))
        .unwrap();
    drop(runtime);
    assert_eq!(lock(&inspector.queue.0).idle_waits, 1);
    assert_eq!(lock(&inspector.queue.0).completed, 0);
    inspector.device.assert_counts(FRAME_ALLOCATIONS, 0);
    // The inspector retains the old callback queue independently of the runtime.
    inspector.queue.complete_through(1);
    inspector
        .device
        .assert_counts(FRAME_ALLOCATIONS, FRAME_ALLOCATIONS);
}

#[test]
fn dropping_a_completion_callback_is_not_confirmation_or_retirement() {
    let (runtime, instance) = runtime(2);
    let inspector = instance.device(0);
    let owner = runtime
        .execute_owned(frame(&Arc::new(AtomicUsize::new(0))))
        .unwrap();
    inspector.queue.discard_callbacks();
    assert!(!owner.is_complete());
    assert_eq!(runtime.completed_value(), 0);
    drop(owner);
    inspector.device.assert_counts(FRAME_ALLOCATIONS, 0);
    // Even a later successful queue wait cannot synthesize the discarded callback.
    drop(runtime);
    assert_eq!(lock(&inspector.queue.0).idle_waits, 1);
    inspector.device.assert_counts(FRAME_ALLOCATIONS, 0);
    // These unconfirmed resources are intentionally left to device teardown.
}

#[test]
fn failed_pacing_waits_restore_the_oldest_token_before_allocation_record_or_submit() {
    let (runtime, instance) = runtime(2);
    let inspector = instance.device(0);
    let calls = Arc::new(AtomicUsize::new(0));
    runtime.execute(frame(&calls)).unwrap();
    runtime.execute(frame(&calls)).unwrap();
    inspector.queue.set_wait_results(false, false);

    assert_eq!(
        runtime.execute(frame(&calls)),
        Err(SubmissionError::CompletionUnconfirmed)
    );
    assert_eq!(
        runtime.execute_tracked(frame(&calls)),
        Err(SubmissionError::CompletionUnconfirmed)
    );
    assert!(matches!(
        runtime.execute_owned(frame(&calls)),
        Err(SubmissionError::CompletionUnconfirmed)
    ));
    assert_eq!(runtime.frames_in_flight(), 2);
    assert_eq!(runtime.next_completion_value(), 3);
    assert_eq!(runtime.completed_value(), 0);
    assert_eq!(calls.load(Ordering::SeqCst), 2 * FRAME_RECORDINGS);
    inspector.device.assert_counts(2 * FRAME_ALLOCATIONS, 0);
    inspector.queue.assert_submissions(2);
    assert_eq!(lock(&inspector.queue.0).waits, vec![1, 1, 1]);
    assert_eq!(lock(&inspector.queue.0).callbacks.len(), 2);

    inspector.queue.set_wait_results(true, true);
    let retried = runtime.execute_owned(frame(&calls)).unwrap();
    assert_eq!(retried.completion_value(), 3);
    assert_eq!(runtime.completed_value(), 1);
    assert_eq!(runtime.frames_in_flight(), 2);
    assert_eq!(calls.load(Ordering::SeqCst), 3 * FRAME_RECORDINGS);
    inspector
        .device
        .assert_counts(3 * FRAME_ALLOCATIONS, FRAME_ALLOCATIONS);
    inspector.queue.assert_submissions(3);
    assert_eq!(lock(&inspector.queue.0).waits, vec![1, 1, 1, 1]);
    inspector.queue.set_wait_results(false, false);
    assert_eq!(
        runtime.execute(frame(&calls)),
        Err(SubmissionError::CompletionUnconfirmed)
    );
    assert_eq!(lock(&inspector.queue.0).waits, vec![1, 1, 1, 1, 2]);
    assert_eq!(runtime.frames_in_flight(), 2);
    assert_eq!(runtime.next_completion_value(), 4);
    assert_eq!(calls.load(Ordering::SeqCst), 3 * FRAME_RECORDINGS);
    inspector.queue.assert_submissions(3);
    drop(retried);
    inspector.queue.complete_through(3);
    inspector
        .device
        .assert_counts(3 * FRAME_ALLOCATIONS, 3 * FRAME_ALLOCATIONS);
    inspector.queue.set_wait_results(true, true);
}

#[test]
fn failed_idle_wait_keeps_unconfirmed_tokens_in_original_order_for_retry() {
    let (runtime, instance) = runtime(3);
    let inspector = instance.device(0);
    let calls = Arc::new(AtomicUsize::new(0));
    for _ in 0..3 {
        runtime.execute(frame(&calls)).unwrap();
    }
    inspector.queue.set_wait_results(false, false);
    assert!(!runtime.wait_idle());
    assert_eq!(runtime.frames_in_flight(), 3);
    assert_eq!(lock(&inspector.queue.0).waits, vec![1, 2, 3]);
    assert_eq!(lock(&inspector.queue.0).idle_waits, 1);
    inspector.device.assert_counts(3 * FRAME_ALLOCATIONS, 0);
    assert_eq!(
        runtime.execute(frame(&calls)),
        Err(SubmissionError::CompletionUnconfirmed)
    );
    assert_eq!(runtime.frames_in_flight(), 3);
    assert_eq!(runtime.next_completion_value(), 4);
    assert_eq!(calls.load(Ordering::SeqCst), 3 * FRAME_RECORDINGS);
    inspector.queue.assert_submissions(3);
    inspector.queue.set_wait_results(true, true);
    assert!(runtime.wait_idle());
    assert_eq!(lock(&inspector.queue.0).waits, vec![1, 2, 3, 1, 1, 2, 3]);
    assert_eq!(lock(&inspector.queue.0).idle_waits, 2);
    assert_eq!(runtime.frames_in_flight(), 0);
    assert_eq!(runtime.completed_value(), 3);
    inspector
        .device
        .assert_counts(3 * FRAME_ALLOCATIONS, 3 * FRAME_ALLOCATIONS);
}

#[test]
fn recovery_waits_on_the_old_queue_and_retained_owners_do_not_destroy_reused_new_ids() {
    let (mut runtime, instance) = runtime(2);
    let old = instance.device(0);
    let calls = Arc::new(AtomicUsize::new(0));
    let old_owner = runtime.execute_owned(frame(&calls)).unwrap();
    *lock(&old.device.loss) = Some(rhi::DeviceLostReason::Unknown);
    let report = runtime
        .recover_if_lost()
        .unwrap()
        .expect("injected loss must recover");
    assert_eq!(report.generation, 1);
    assert_eq!(report.reason, Some(rhi::DeviceLostReason::Unknown));
    assert_eq!(lock(&old.queue.0).idle_waits, 1);
    assert!(old_owner.is_complete());
    assert_eq!(old_owner.device_generation(), 0);
    old.device.assert_counts(FRAME_ALLOCATIONS, 0);
    assert_eq!(runtime.completed_value(), 0);
    assert_eq!(runtime.next_completion_value(), 1);
    assert_eq!(runtime.frames_in_flight(), 0);

    let new = instance.device(1);
    let new_owner = runtime.execute_owned(frame(&calls)).unwrap();
    assert_eq!(new_owner.device_generation(), 1);
    assert_eq!(new_owner.completion_value(), 1);
    assert!(!new_owner.is_complete());
    // Independent NullDevices intentionally mint identical IDs. This catches
    // accidental destruction through the runtime's replacement device.
    assert_eq!(
        old_owner.allocations().textures(),
        new_owner.allocations().textures()
    );
    assert_eq!(
        old_owner.allocations().buffers(),
        new_owner.allocations().buffers()
    );
    drop(old_owner);
    old.device
        .assert_counts(FRAME_ALLOCATIONS, FRAME_ALLOCATIONS);
    new.device.assert_counts(FRAME_ALLOCATIONS, 0);
    assert_eq!(runtime.completed_value(), 0);
    new.queue.complete_through(1);
    assert!(new_owner.is_complete());
    new.device.assert_counts(FRAME_ALLOCATIONS, 0);
    drop(new_owner);
    new.device
        .assert_counts(FRAME_ALLOCATIONS, FRAME_ALLOCATIONS);
}

#[test]
fn late_old_generation_callback_cannot_complete_or_retire_the_replacement_frame() {
    let (mut runtime, instance) = runtime(2);
    let old = instance.device(0);
    let calls = Arc::new(AtomicUsize::new(0));
    let old_owner = runtime.execute_owned(frame(&calls)).unwrap();
    let old_textures = old_owner.allocations().textures().to_vec();
    let old_buffers = old_owner.allocations().buffers().to_vec();
    drop(old_owner);
    old.queue.set_wait_results(false, false);
    assert_eq!(runtime.recover_device().unwrap().generation, 1);
    assert_eq!(lock(&old.queue.0).idle_waits, 1);
    old.device.assert_counts(FRAME_ALLOCATIONS, 0);
    let new = instance.device(1);
    let new_owner = runtime.execute_owned(frame(&calls)).unwrap();
    assert_eq!(new_owner.allocations().textures(), old_textures.as_slice());
    assert_eq!(new_owner.allocations().buffers(), old_buffers.as_slice());

    old.queue.complete_through(1);
    old.device
        .assert_counts(FRAME_ALLOCATIONS, FRAME_ALLOCATIONS);
    new.device.assert_counts(FRAME_ALLOCATIONS, 0);
    assert!(!new_owner.is_complete());
    assert_eq!(runtime.completed_value(), 0);
    assert_eq!(runtime.frames_in_flight(), 1);
    assert_eq!(runtime.next_completion_value(), 2);
    drop(new_owner);
    new.queue.complete_through(1);
    assert_eq!(runtime.completed_value(), 1);
    new.device
        .assert_counts(FRAME_ALLOCATIONS, FRAME_ALLOCATIONS);
}

#[test]
fn completion_values_never_wrap_or_reuse_the_final_value() {
    let (runtime, instance) = runtime(1);
    let inspector = instance.device(0);
    let calls = Arc::new(AtomicUsize::new(0));
    runtime
        .submitted_value
        .store(u64::MAX - 1, Ordering::Release);
    assert_eq!(runtime.next_completion_value(), u64::MAX);
    let last = runtime.execute_owned(frame(&calls)).unwrap();
    assert_eq!(last.completion_value(), u64::MAX);
    assert_eq!(runtime.next_completion_value(), u64::MAX);

    assert_eq!(
        runtime.execute(frame(&calls)),
        Err(SubmissionError::CompletionValueExhausted)
    );
    assert_eq!(
        runtime.execute_tracked(frame(&calls)),
        Err(SubmissionError::CompletionValueExhausted)
    );
    assert!(matches!(
        runtime.execute_owned(frame(&calls)),
        Err(SubmissionError::CompletionValueExhausted)
    ));
    assert_eq!(calls.load(Ordering::SeqCst), FRAME_RECORDINGS);
    inspector.device.assert_counts(FRAME_ALLOCATIONS, 0);
    inspector.queue.assert_submissions(1);
    assert!(lock(&inspector.queue.0).waits.is_empty());
    assert_eq!(runtime.frames_in_flight(), 1);

    inspector.queue.complete_through(1);
    assert!(last.is_complete());
    assert_eq!(runtime.completed_value(), u64::MAX);
    drop(last);
    inspector
        .device
        .assert_counts(FRAME_ALLOCATIONS, FRAME_ALLOCATIONS);
}

#[test]
fn compile_failure_has_no_retirement_token_allocation_recording_or_submission() {
    let (runtime, instance) = runtime(1);
    let inspector = instance.device(0);
    let calls = Arc::new(AtomicUsize::new(0));
    let invalid = || {
        let mut graph = runtime.new_graph();
        let calls = Arc::clone(&calls);
        graph.add_compute_pass("overflow", PassFlags::SIDE_EFFECT, move |b| {
            let buffer = b.create_buffer(
                "overflowing footprint",
                BufferDesc::new(u64::MAX, rhi::BufferUsages::STORAGE),
            );
            b.write_storage_buffer(buffer);
            move |ctx| {
                calls.fetch_add(1, Ordering::SeqCst);
                ctx.dispatch(1, 1, 1);
            }
        });
        graph
    };
    assert_eq!(
        runtime.execute(invalid()),
        Err(SubmissionError::Compile(CompileError::MemoryOverflow))
    );
    assert_eq!(
        runtime.execute_tracked(invalid()),
        Err(SubmissionError::Compile(CompileError::MemoryOverflow))
    );
    assert!(matches!(
        runtime.execute_owned(invalid()),
        Err(SubmissionError::Compile(CompileError::MemoryOverflow))
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    inspector.device.assert_counts(0, 0);
    inspector.queue.assert_submissions(0);
    assert!(lock(&inspector.queue.0).callbacks.is_empty());
    assert!(lock(&inspector.queue.0).waits.is_empty());
    assert_eq!(runtime.frames_in_flight(), 0);
    assert_eq!(runtime.next_completion_value(), 1);
    drop(runtime);
    assert_eq!(lock(&inspector.queue.0).idle_waits, 0);
}
