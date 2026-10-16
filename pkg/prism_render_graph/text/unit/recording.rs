//! Record-only ownership and retirement contracts, using an instrumented RHI.
//! These are CPU unit tests, not native-queue/GPU-completion evidence.

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::cell::{Cell, RefCell};
use core::sync::atomic::{AtomicUsize, Ordering};

use prism_render_driver::{
    Backend, BindGroupDescriptor, BindGroupId, BindGroupLayoutDescriptor, BindGroupLayoutId,
    BufferDescriptor, BufferId, BufferState, BufferUsages, CommandBuffer,
    ComputePipelineDescriptor, ComputePipelineId, DeviceCapabilities, Extent3d, Features,
    ImageDataLayout, Limits, PipelineLayoutDescriptor, PipelineLayoutId, RenderDevice,
    RenderPipelineDescriptor, RenderPipelineId, RenderQueue, SamplerDescriptor, SamplerId,
    ShaderModuleDescriptor, ShaderModuleId, TextureDescriptor, TextureFormat, TextureId,
    TextureState, TextureViewDescriptor, TextureViewId, TextureWrite,
};

use crate::desc::{BufferDesc, TextureDesc};
use crate::graph::RenderGraph;
use crate::handle::BufferHandle;
use crate::pass::PassFlags;
use crate::plan::CompileError;
use crate::resource::{ImportedBuffer, ImportedTexture, Lifetime};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Allocation {
    Texture(TextureId),
    View(TextureViewId),
    Buffer(BufferId),
}

// Intentionally !Sync: recording must not send the device to worker threads.
struct TrackingDevice {
    caps: DeviceCapabilities,
    next: Cell<u32>,
    live: RefCell<Vec<Allocation>>,
    created: RefCell<Vec<Allocation>>,
    destroyed: RefCell<Vec<Allocation>>,
    views: RefCell<Vec<(TextureViewId, TextureId)>>,
    texture_extents: RefCell<Vec<Extent3d>>,
}

impl TrackingDevice {
    fn new() -> Self {
        Self {
            caps: DeviceCapabilities {
                backend: Backend::Noop,
                features: Features::NONE,
                limits: Limits::default(),
            },
            next: Cell::new(1),
            live: RefCell::new(Vec::new()),
            created: RefCell::new(Vec::new()),
            destroyed: RefCell::new(Vec::new()),
            views: RefCell::new(Vec::new()),
            texture_extents: RefCell::new(Vec::new()),
        }
    }

    fn mint(&self) -> u32 {
        let id = self.next.get();
        self.next.set(id + 1);
        id
    }

    fn created(&self, allocation: Allocation) {
        self.live.borrow_mut().push(allocation);
        self.created.borrow_mut().push(allocation);
    }

    fn destroy(&self, allocation: Allocation) {
        let mut live = self.live.borrow_mut();
        let index = live
            .iter()
            .position(|item| *item == allocation)
            .expect("must not destroy imports, foreign IDs, or an allocation twice");
        live.remove(index);
        self.destroyed.borrow_mut().push(allocation);
    }

    fn assert_fully_released(&self) {
        assert!(self.live.borrow().is_empty());
        assert_eq!(self.created.borrow().len(), self.destroyed.borrow().len());
    }
}

// These tests exercise only graph-owned textures, views and buffers. Unexpected
// pipeline/binding activity must fail rather than silently minting fake objects.
macro_rules! unsupported_resources {
    ($($create:ident($descriptor:ty) -> $id:ty, $destroy:ident;)*) => {
        $(
            fn $create(&self, _: &$descriptor) -> $id {
                panic!(concat!(stringify!($create), " is outside this recording fixture"));
            }
            fn $destroy(&self, _: $id) {
                panic!(concat!(stringify!($destroy), " is outside this recording fixture"));
            }
        )*
    };
}

impl RenderDevice for TrackingDevice {
    fn capabilities(&self) -> &DeviceCapabilities {
        &self.caps
    }

    fn create_texture(&self, descriptor: &TextureDescriptor) -> TextureId {
        let id = TextureId::from_parts(self.mint(), 0);
        self.texture_extents.borrow_mut().push(descriptor.size);
        self.created(Allocation::Texture(id));
        id
    }

    fn create_texture_view(&self, texture: TextureId, _: &TextureViewDescriptor) -> TextureViewId {
        let view = TextureViewId::from_parts(self.mint(), 0);
        assert!(self.live.borrow().contains(&Allocation::Texture(texture)));
        self.views.borrow_mut().push((view, texture));
        self.created(Allocation::View(view));
        view
    }

    fn create_buffer(&self, _: &BufferDescriptor) -> BufferId {
        let id = BufferId::from_parts(self.mint(), 0);
        self.created(Allocation::Buffer(id));
        id
    }

    fn destroy_texture_view(&self, id: TextureViewId) {
        self.destroy(Allocation::View(id));
    }

    fn destroy_texture(&self, id: TextureId) {
        for &(view, texture) in self.views.borrow().iter() {
            if texture == id {
                assert!(
                    !self.live.borrow().contains(&Allocation::View(view)),
                    "destroy default views before their textures"
                );
            }
        }
        self.destroy(Allocation::Texture(id));
    }

    fn destroy_buffer(&self, id: BufferId) {
        self.destroy(Allocation::Buffer(id));
    }

    unsupported_resources! {
        create_sampler(SamplerDescriptor) -> SamplerId, destroy_sampler;
        create_shader_module(ShaderModuleDescriptor) -> ShaderModuleId, destroy_shader_module;
        create_bind_group_layout(BindGroupLayoutDescriptor) -> BindGroupLayoutId, destroy_bind_group_layout;
        create_bind_group(BindGroupDescriptor) -> BindGroupId, destroy_bind_group;
        create_pipeline_layout(PipelineLayoutDescriptor) -> PipelineLayoutId, destroy_pipeline_layout;
        create_render_pipeline(RenderPipelineDescriptor) -> RenderPipelineId, destroy_render_pipeline;
        create_compute_pipeline(ComputePipelineDescriptor) -> ComputePipelineId, destroy_compute_pipeline;
    }
}

#[derive(Default)]
struct TrackingQueue {
    submitted: RefCell<Vec<CommandBuffer>>,
}

impl RenderQueue for TrackingQueue {
    fn write_buffer(&self, _: BufferId, _: u64, _: &[u8]) {
        panic!("unexpected upload");
    }
    fn write_texture(&self, _: TextureWrite, _: &[u8], _: ImageDataLayout) {
        panic!("unexpected upload");
    }
    fn submit(&self, commands: &[CommandBuffer]) {
        self.submitted.borrow_mut().extend_from_slice(commands);
    }
}

fn extent() -> Extent3d {
    Extent3d::new_2d(64, 32)
}

fn color() -> TextureDesc {
    TextureDesc::color(TextureFormat::Rgba8Unorm)
}

fn buffer() -> BufferDesc {
    BufferDesc::new(256, BufferUsages::NONE)
}

// Includes imports, both owned lifetimes, raster/compute/transfer commands and
// a cross-queue producer/consumer dependency. All three callbacks survive.
fn frame(calls: Arc<AtomicUsize>) -> RenderGraph {
    let mut graph = RenderGraph::new(extent());
    let imported_texture = graph.import_texture(
        "external target",
        color(),
        ImportedTexture {
            texture: TextureId::from_parts(1000, 0),
            default_view: TextureViewId::from_parts(1001, 0),
            entry_state: TextureState::initial(),
            exit_state: TextureState::present(),
            mip_levels: 1,
            array_layers: 1,
        },
    );
    let imported_buffer = graph.import_buffer(
        "external buffer",
        buffer(),
        ImportedBuffer {
            buffer: BufferId::from_parts(1002, 0),
            entry_state: BufferState::initial(),
        },
    );

    let compute_calls = calls.clone();
    graph.add_compute_pass(
        "produce",
        PassFlags::SIDE_EFFECT | PassFlags::ASYNC_COMPUTE,
        move |b| {
            let transient = b.create_buffer("frame buffer", buffer());
            let transient = b.write_storage_buffer(transient);
            let persistent = b.create_persistent_buffer("history buffer", buffer());
            b.write_storage_buffer(persistent);
            b.read_storage_buffer(imported_buffer);
            b.blackboard_mut().set(transient);
            move |ctx| {
                compute_calls.fetch_add(1, Ordering::SeqCst);
                ctx.dispatch(2, 1, 1);
            }
        },
    );

    let transfer_calls = calls.clone();
    graph.add_transfer_pass("consume", PassFlags::SIDE_EFFECT, move |b| {
        let source = *b.blackboard().get::<BufferHandle>().unwrap();
        let source = b.copy_src_buffer(source);
        let destination = b.create_buffer("copy destination", buffer());
        let destination = b.copy_dst_buffer(destination);
        move |ctx| {
            transfer_calls.fetch_add(1, Ordering::SeqCst);
            ctx.copy_buffer_to_buffer(source, 0, destination, 0, 256);
        }
    });

    graph.add_raster_pass("raster", PassFlags::EMPTY, move |b| {
        let transient = b.create_texture("frame target", color());
        let persistent = b.create_persistent_texture("history target", color());
        b.color_attachment(transient);
        b.color_attachment(persistent);
        b.color_attachment(imported_texture);
        move |ctx| {
            calls.fetch_add(1, Ordering::SeqCst);
            ctx.draw(0..3, 0..1);
        }
    });
    graph
}

#[test]
fn prepare_is_pure_and_record_keeps_the_exact_plan_without_submitting() {
    let device = TrackingDevice::new();
    let calls = Arc::new(AtomicUsize::new(0));
    let prepared = frame(calls.clone()).prepare().unwrap();
    let plan = prepared.plan().clone();
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(device.created.borrow().is_empty());
    assert!(plan
        .queue_batches
        .iter()
        .any(|batch| !batch.waits_for.is_empty()));
    assert!(plan
        .logical_barriers
        .iter()
        .any(|barrier| barrier.queue_transfer));

    let recorded = prepared.record(&device);
    assert_eq!(recorded.plan(), &plan);
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    assert_eq!(recorded.commands().passes.len(), 3);
    assert_eq!(recorded.allocations().textures().len(), 2);
    assert_eq!(recorded.allocations().buffers().len(), 3);
    assert!(device.destroyed.borrow().is_empty());
    // No RenderQueue is supplied to record; not even a submit capability exists.
    recorded.release(&device);
    device.assert_fully_released();
}

#[test]
fn external_submission_retains_allocations_until_explicit_retirement() {
    let device = TrackingDevice::new();
    let queue = TrackingQueue::default();
    let recorded = frame(Arc::new(AtomicUsize::new(0)))
        .record(&device)
        .unwrap();
    let (commands, plan, allocations) = recorded.into_parts();
    assert!(!allocations.is_empty());
    assert!(!plan.queue_batches.is_empty());
    queue.submit(core::slice::from_ref(&commands));
    assert_eq!(queue.submitted.borrow().len(), 1);
    assert!(
        device.destroyed.borrow().is_empty(),
        "submit is not completion"
    );
    assert_eq!(device.live.borrow().len(), 7);

    // The unit fixture models the caller retiring a completed batch. Production
    // must gate this call on the native submitting queue's completion signal.
    drop(commands);
    allocations.release(&device);
    device.assert_fully_released();
}

#[test]
fn transient_retirement_preserves_history_and_never_destroys_imports() {
    let device = TrackingDevice::new();
    let recorded = frame(Arc::new(AtomicUsize::new(0)))
        .record(&device)
        .unwrap();
    let (commands, _, mut allocations) = recorded.into_parts();
    drop(commands); // Cancelled before submission; safe to release frame resources.
    allocations.release_transients(&device);
    assert_eq!(allocations.textures().len(), 1);
    assert_eq!(allocations.buffers().len(), 1);
    assert_eq!(allocations.textures()[0].1, Lifetime::Persistent);
    assert_eq!(allocations.buffers()[0].1, Lifetime::Persistent);
    assert_eq!(device.live.borrow().len(), 3);
    let destroyed = device.destroyed.borrow().len();
    allocations.release_transients(&device);
    assert_eq!(
        device.destroyed.borrow().len(),
        destroyed,
        "no double destroy"
    );
    allocations.release(&device);
    // TrackingDevice rejects any destroy of an imported or already freed ID.
    device.assert_fully_released();
}

#[test]
fn dropping_packet_does_not_destroy_potentially_in_flight_resources() {
    let device = TrackingDevice::new();
    let queue = TrackingQueue::default();
    let recorded = frame(Arc::new(AtomicUsize::new(0)))
        .record(&device)
        .unwrap();
    queue.submit(core::slice::from_ref(recorded.commands()));
    let textures = recorded.allocations().textures().to_vec();
    let buffers = recorded.allocations().buffers().to_vec();
    drop(recorded);
    assert!(device.destroyed.borrow().is_empty());
    assert_eq!(device.live.borrow().len(), 7);
    // A dropped owner intentionally leaks rather than guessing completion.
    // Clean up this mock's copied IDs only after discarding its fake submission.
    queue.submitted.borrow_mut().clear();
    for (_, _, texture) in textures {
        device.destroy_texture_view(texture.default_view);
        device.destroy_texture(texture.texture);
    }
    for (_, _, buffer) in buffers {
        device.destroy_buffer(buffer);
    }
    device.assert_fully_released();
}

#[test]
fn execute_returns_the_owner_of_exactly_the_commands_it_submitted() {
    let device = TrackingDevice::new();
    let queue = TrackingQueue::default();
    let calls = Arc::new(AtomicUsize::new(0));
    let recorded = frame(calls.clone()).execute(&device, &queue).unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    assert_eq!(
        queue.submitted.borrow().as_slice(),
        core::slice::from_ref(recorded.commands())
    );
    assert!(device.destroyed.borrow().is_empty());
    queue.submitted.borrow_mut().clear();
    recorded.release(&device);
    device.assert_fully_released();
}

fn overflowing_frame(calls: Arc<AtomicUsize>) -> RenderGraph {
    let mut graph = RenderGraph::new(extent());
    graph.add_compute_pass("overflow", PassFlags::SIDE_EFFECT, move |b| {
        let resource = b.create_buffer(
            "overflowing footprint",
            BufferDesc::new(u64::MAX, BufferUsages::NONE),
        );
        b.write_storage_buffer(resource);
        move |ctx| {
            calls.fetch_add(1, Ordering::SeqCst);
            ctx.dispatch(1, 1, 1);
        }
    });
    graph
}

#[test]
fn compile_error_means_zero_allocation_recording_and_submission() {
    let device = TrackingDevice::new();
    let queue = TrackingQueue::default();
    let calls = Arc::new(AtomicUsize::new(0));
    assert!(matches!(
        overflowing_frame(calls.clone()).prepare(),
        Err(CompileError::MemoryOverflow)
    ));
    assert!(matches!(
        overflowing_frame(calls.clone()).record(&device),
        Err(CompileError::MemoryOverflow)
    ));
    assert!(matches!(
        overflowing_frame(calls.clone()).execute(&device, &queue),
        Err(CompileError::MemoryOverflow)
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(device.created.borrow().is_empty());
    assert!(device.destroyed.borrow().is_empty());
    assert!(queue.submitted.borrow().is_empty());
}

#[test]
fn cancelled_preparation_has_no_physical_allocations_or_recorded_callbacks() {
    let device = TrackingDevice::new();
    let calls = Arc::new(AtomicUsize::new(0));
    drop(frame(calls.clone()).prepare().unwrap());
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(device.created.borrow().is_empty());
}

#[test]
fn recording_uses_prepared_extent_and_resize_requires_a_new_preparation() {
    let device = TrackingDevice::new();
    let calls = Arc::new(AtomicUsize::new(0));
    let resized = Extent3d::new_2d(128, 64);
    let prepare = |size| {
        let mut graph = RenderGraph::new(size);
        let calls = calls.clone();
        graph.add_raster_pass("sized target", PassFlags::SIDE_EFFECT, move |b| {
            let target = b.create_texture("target", color());
            b.color_attachment(target);
            move |ctx| {
                calls.fetch_add(1, Ordering::SeqCst);
                ctx.draw(0..3, 0..1);
            }
        });
        graph.prepare().unwrap()
    };
    let original = prepare(extent());
    let new_size = prepare(resized);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(device.created.borrow().is_empty());

    let original_plan = original.plan().clone();
    let resized_plan = new_size.plan().clone();
    assert!(resized_plan.alias.heap_size > original_plan.alias.heap_size);
    let original = original.record(&device);
    let new_size = new_size.record_sequential(&device);
    assert_eq!(original.plan(), &original_plan);
    assert_eq!(new_size.plan(), &resized_plan);
    assert_eq!(
        device.texture_extents.borrow().as_slice(),
        &[extent(), resized]
    );
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert!(device.destroyed.borrow().is_empty());
    original.release(&device);
    new_size.release(&device);
    device.assert_fully_released();
}

#[test]
fn culled_resources_and_callbacks_are_not_realized_or_recorded() {
    let device = TrackingDevice::new();
    let calls = Arc::new(AtomicUsize::new(0));
    let dead_calls = calls.clone();
    let mut graph = RenderGraph::new(extent());
    graph.add_raster_pass("dead", PassFlags::EMPTY, move |b| {
        let texture = b.create_texture("unused", color());
        b.color_attachment(texture);
        move |_| {
            dead_calls.fetch_add(1, Ordering::SeqCst);
        }
    });
    let recorded = graph.record(&device).unwrap();
    assert_eq!(recorded.plan().alive, [false]);
    assert!(recorded.commands().passes.is_empty());
    assert!(recorded.allocations().is_empty());
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(device.created.borrow().is_empty());
    recorded.release(&device);
    device.assert_fully_released();
}

#[test]
fn public_sequential_recording_matches_the_default_strategy() {
    let sequential_device = TrackingDevice::new();
    let default_device = TrackingDevice::new();
    let calls = Arc::new(AtomicUsize::new(0));
    let sequential = frame(calls.clone())
        .prepare()
        .unwrap()
        .record_sequential(&sequential_device);
    let default = frame(calls.clone())
        .prepare()
        .unwrap()
        .record(&default_device);
    assert_eq!(calls.load(Ordering::SeqCst), 6);
    assert_eq!(sequential.commands(), default.commands());
    assert_eq!(sequential.plan(), default.plan());
    assert_eq!(
        sequential.allocations().textures(),
        default.allocations().textures()
    );
    assert_eq!(
        sequential.allocations().buffers(),
        default.allocations().buffers()
    );
    sequential.release(&sequential_device);
    default.release(&default_device);
    sequential_device.assert_fully_released();
    default_device.assert_fully_released();
}

#[test]
fn empty_recording_still_returns_an_owned_packet_without_allocations() {
    let device = TrackingDevice::new();
    let recorded = RenderGraph::new(extent()).record(&device).unwrap();
    assert!(recorded.plan().order.is_empty());
    assert!(recorded.commands().passes.is_empty());
    assert!(recorded.allocations().is_empty());
    recorded.release(&device);
    device.assert_fully_released();
}
