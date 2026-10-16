//! GPU-gated tests for the RHI render spine.
//!
//! Each test first attempts to bring up a real device. When no adapter is
//! present (the usual GPU-less CI sandbox) the test returns early and is a
//! no-op, so the suite always passes without hardware. On a machine with a GPU
//! it exercises the full facade -> graph -> readback path through the frozen
//! RHI, which is exactly the ground the Bevy render stack used to provide.

use prism_render_driver as rhi;
use prism_render_driver_wgpu::WgpuInstance;
use prism_render_graph::{ImportedTexture, PassFlags, TextureDesc};
use rhi::{
    Adapter, ColorTargetState, FragmentState, Instance, MultisampleState, PrimitiveState,
    RenderDevice, RenderQueue, ShaderModuleDescriptor, Surface, TextureState, TextureUsages,
    VertexFormat, VertexState, VertexStepMode,
};

use crate::RenderRuntime;

use prism_render_driver_null::NullInstance;
use raw_window_handle::{HandleError, HasDisplayHandle, HasWindowHandle};

use alloc::sync::{Arc, Weak};
use core::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// The offscreen target edge length used by the readback test.
const SIZE: u32 = 64;

/// A window handle that always reports "unavailable". The null surface never
/// inspects it, so the headless present loop needs no real OS window.
struct FakeWindow;

#[derive(Default)]
struct BringUpCalls {
    adapter_requests: Mutex<Vec<rhi::RequestAdapterOptions>>,
    surface_adapter_requests: AtomicUsize,
    device_descriptors: Mutex<Vec<rhi::DeviceDescriptor>>,
    device_creations: AtomicUsize,
    surface_creations: AtomicUsize,
    current_loss: Mutex<Option<Weak<Mutex<Option<rhi::DeviceLostReason>>>>>,
}

impl BringUpCalls {
    fn report_loss(&self, reason: rhi::DeviceLostReason) {
        let signal = self
            .current_loss
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .and_then(Weak::upgrade)
            .expect("a current test device exists");
        *signal
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(reason);
    }
}

#[derive(Clone)]
struct LossyNullDevice {
    inner: prism_render_driver_null::NullDevice,
    loss: Arc<Mutex<Option<rhi::DeviceLostReason>>>,
}

impl LossyNullDevice {
    fn live_resource_count(&self) -> usize {
        self.inner.live_resource_count()
    }
}

impl RenderDevice for LossyNullDevice {
    fn capabilities(&self) -> &rhi::DeviceCapabilities {
        self.inner.capabilities()
    }

    fn create_buffer(&self, descriptor: &rhi::BufferDescriptor) -> rhi::BufferId {
        self.inner.create_buffer(descriptor)
    }

    fn create_texture(&self, descriptor: &rhi::TextureDescriptor) -> rhi::TextureId {
        self.inner.create_texture(descriptor)
    }

    fn create_texture_view(
        &self,
        texture: rhi::TextureId,
        descriptor: &rhi::TextureViewDescriptor,
    ) -> rhi::TextureViewId {
        self.inner.create_texture_view(texture, descriptor)
    }

    fn create_sampler(&self, descriptor: &rhi::SamplerDescriptor) -> rhi::SamplerId {
        self.inner.create_sampler(descriptor)
    }

    fn create_shader_module(&self, descriptor: &ShaderModuleDescriptor) -> rhi::ShaderModuleId {
        self.inner.create_shader_module(descriptor)
    }

    fn create_bind_group_layout(
        &self,
        descriptor: &rhi::BindGroupLayoutDescriptor,
    ) -> rhi::BindGroupLayoutId {
        self.inner.create_bind_group_layout(descriptor)
    }

    fn create_bind_group(&self, descriptor: &rhi::BindGroupDescriptor) -> rhi::BindGroupId {
        self.inner.create_bind_group(descriptor)
    }

    fn create_pipeline_layout(
        &self,
        descriptor: &rhi::PipelineLayoutDescriptor,
    ) -> rhi::PipelineLayoutId {
        self.inner.create_pipeline_layout(descriptor)
    }

    fn create_render_pipeline(
        &self,
        descriptor: &rhi::RenderPipelineDescriptor,
    ) -> rhi::RenderPipelineId {
        self.inner.create_render_pipeline(descriptor)
    }

    fn create_compute_pipeline(
        &self,
        descriptor: &rhi::ComputePipelineDescriptor,
    ) -> rhi::ComputePipelineId {
        self.inner.create_compute_pipeline(descriptor)
    }

    fn take_device_lost(&self) -> Option<rhi::DeviceLostReason> {
        self.loss
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
    }

    fn destroy_buffer(&self, id: rhi::BufferId) {
        self.inner.destroy_buffer(id);
    }

    fn destroy_texture(&self, id: rhi::TextureId) {
        self.inner.destroy_texture(id);
    }

    fn destroy_texture_view(&self, id: rhi::TextureViewId) {
        self.inner.destroy_texture_view(id);
    }

    fn destroy_sampler(&self, id: rhi::SamplerId) {
        self.inner.destroy_sampler(id);
    }

    fn destroy_shader_module(&self, id: rhi::ShaderModuleId) {
        self.inner.destroy_shader_module(id);
    }

    fn destroy_bind_group_layout(&self, id: rhi::BindGroupLayoutId) {
        self.inner.destroy_bind_group_layout(id);
    }

    fn destroy_bind_group(&self, id: rhi::BindGroupId) {
        self.inner.destroy_bind_group(id);
    }

    fn destroy_pipeline_layout(&self, id: rhi::PipelineLayoutId) {
        self.inner.destroy_pipeline_layout(id);
    }

    fn destroy_render_pipeline(&self, id: rhi::RenderPipelineId) {
        self.inner.destroy_render_pipeline(id);
    }

    fn destroy_compute_pipeline(&self, id: rhi::ComputePipelineId) {
        self.inner.destroy_compute_pipeline(id);
    }
}

#[derive(Clone, Default)]
struct CountingInstance {
    calls: Arc<BringUpCalls>,
}

impl CountingInstance {
    fn new() -> Self {
        Self::default()
    }
}

impl Instance for CountingInstance {
    type Adapter = prism_render_driver_null::NullAdapter;
    type Device = LossyNullDevice;
    type Queue = prism_render_driver_null::NullQueue;
    type Surface = prism_render_driver_null::NullSurface;

    fn enumerate_adapters(&self) -> Vec<Self::Adapter> {
        <NullInstance as Instance>::enumerate_adapters(&NullInstance::new())
    }

    fn request_adapter(&self, options: &rhi::RequestAdapterOptions) -> Option<Self::Adapter> {
        self.calls
            .adapter_requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(*options);
        <NullInstance as Instance>::request_adapter(&NullInstance::new(), options)
    }

    fn request_adapter_for_surface(
        &self,
        options: &rhi::RequestAdapterOptions,
        surface: &Self::Surface,
    ) -> Option<Self::Adapter> {
        self.calls
            .surface_adapter_requests
            .fetch_add(1, Ordering::Relaxed);
        self.calls
            .adapter_requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(*options);
        <NullInstance as Instance>::request_adapter_for_surface(
            &NullInstance::new(),
            options,
            surface,
        )
    }

    fn create_device(
        &self,
        adapter: &Self::Adapter,
        descriptor: &rhi::DeviceDescriptor,
    ) -> Result<(Self::Device, Self::Queue), rhi::DeviceError> {
        self.calls.device_creations.fetch_add(1, Ordering::Relaxed);
        self.calls
            .device_descriptors
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(descriptor.clone());
        let (inner, queue) =
            <NullInstance as Instance>::create_device(&NullInstance::new(), adapter, descriptor)?;
        let loss = Arc::new(Mutex::new(None));
        *self
            .calls
            .current_loss
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Arc::downgrade(&loss));
        Ok((LossyNullDevice { inner, loss }, queue))
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
        self.calls.surface_creations.fetch_add(1, Ordering::Relaxed);
        <NullInstance as Instance>::create_surface(
            &NullInstance::new(),
            window,
            &device.inner,
            queue,
        )
    }
}

/// Whether a missing adapter must fail the test instead of skipping it.
///
/// Set `PRISM_REQUIRE_GPU` to opt into strictness: a run meant to exercise
/// hardware cannot silently degrade into a no-op.
fn require_gpu() -> bool {
    std::env::var_os("PRISM_REQUIRE_GPU").is_some()
}

impl HasWindowHandle for FakeWindow {
    fn window_handle(&self) -> Result<raw_window_handle::WindowHandle<'_>, HandleError> {
        Err(HandleError::Unavailable)
    }
}

impl HasDisplayHandle for FakeWindow {
    fn display_handle(&self) -> Result<raw_window_handle::DisplayHandle<'_>, HandleError> {
        Err(HandleError::Unavailable)
    }
}

#[test]
fn surface_present_loop_runs_headless_on_the_null_backend() {
    let mut rt = RenderRuntime::headless(NullInstance::new(), rhi::Extent3d::new_2d(SIZE, SIZE))
        .expect("the null device always comes up");

    // Presenting without a surface is a clear error, not a panic.
    assert_eq!(
        rt.render_frame(|_, _| {}),
        Err(crate::FrameError::NoSurface)
    );

    rt.create_surface(Arc::new(FakeWindow))
        .expect("the null surface is created for any handle");
    rt.configure_surface_default(SIZE, SIZE)
        .expect("the null surface advertises a usable format");
    assert_eq!(
        rt.extent(),
        rhi::Extent3d::new_2d(SIZE, SIZE),
        "configuring adopts the surface extent"
    );

    rt.render_frame(|graph, target| {
        graph.add_raster_pass("present-clear", PassFlags::SIDE_EFFECT, move |b| {
            let _ = b.color_attachment(target);
            |_ctx| {}
        });
    })
    .expect("the frame is acquired, drawn, and presented");

    // The null queue invokes completion immediately, so a confirmed frame is
    // not reported as in flight even though its owner remains in scope.
    assert_eq!(
        rt.frames_in_flight(),
        0,
        "completed frames are not in flight"
    );
    assert!(rt.wait_idle(), "the paced pipeline drains");
    assert_eq!(rt.frames_in_flight(), 0);
}

struct ResidentBuffer {
    id: rhi::BufferId,
    byte_len: u64,
}

#[test]
fn recovery_rebuilds_device_surface_and_real_resources() {
    let instance = CountingInstance::new();
    let calls = Arc::clone(&instance.calls);
    let adapter_options = rhi::RequestAdapterOptions {
        power_preference: rhi::PowerPreference::LowPower,
        force_fallback: true,
    };
    let device_descriptor = rhi::DeviceDescriptor {
        label: Some("recoverable-device".into()),
        ..Default::default()
    };
    let mut rt = RenderRuntime::with_device_configuration(
        instance,
        rhi::Extent3d::new_2d(SIZE, SIZE),
        3,
        adapter_options,
        device_descriptor.clone(),
    )
    .expect("the initial null device comes up");
    rt.create_surface(Arc::new(FakeWindow))
        .expect("the retained surface is created");
    let surface_config = rhi::SurfaceConfig {
        desired_maximum_frame_latency: 3,
        ..rhi::SurfaceConfig::new(rhi::TextureFormat::Bgra8Unorm, 91, 57)
    };
    rt.configure_surface(surface_config)
        .expect("the retained configuration is supported");

    let replay_calls = Arc::new(AtomicUsize::new(0));
    let replay_calls_for_recipe = Arc::clone(&replay_calls);
    let resident = rt
        .register_replay("resident-vertex-buffer", move |device, queue| {
            replay_calls_for_recipe.fetch_add(1, Ordering::Relaxed);
            let bytes = [3_u8, 1, 4, 1, 5, 9, 2, 6];
            let id = device.create_buffer(&rhi::BufferDescriptor {
                label: Some("resident-vertex-buffer".into()),
                size: bytes.len() as u64,
                usage: rhi::BufferUsages::VERTEX | rhi::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(id, 0, &bytes);
            Ok(ResidentBuffer {
                id,
                byte_len: bytes.len() as u64,
            })
        })
        .expect("initial resource creation succeeds");

    let old_device = rt.device_handle();
    let old_resident = resident.current();
    assert_eq!(old_resident.byte_len, 8);
    assert_eq!(rt.device().live_resource_count(), 1);

    let report = rt.recover_device().expect("full recovery succeeds");

    assert_eq!(report.generation, 1);
    assert_eq!(report.reason, None);
    assert_eq!(report.replayed_resources, 1);
    assert!(report.surface_recreated);
    assert_eq!(rt.device_generation(), 1);
    assert_eq!(rt.max_frames_in_flight(), 3);
    assert_eq!(rt.frames_in_flight(), 0);
    assert_eq!(rt.next_completion_value(), 1);
    assert_eq!(rt.completed_value(), 0);
    assert!(!Arc::ptr_eq(&old_device, &rt.device_handle()));

    let new_resident = resident.current();
    assert!(!Arc::ptr_eq(&old_resident, &new_resident));
    assert_eq!(new_resident.byte_len, 8);
    assert_eq!(new_resident.id.index(), 0);
    assert_eq!(rt.device().live_resource_count(), 1);
    assert_eq!(replay_calls.load(Ordering::Relaxed), 2);

    assert_eq!(rt.surface_config(), Some(&surface_config));
    assert_eq!(
        rt.surface().expect("replacement surface exists").size(),
        rhi::Extent3d::new_2d(91, 57)
    );
    assert_eq!(calls.device_creations.load(Ordering::Relaxed), 2);
    assert_eq!(calls.surface_creations.load(Ordering::Relaxed), 2);
    assert_eq!(calls.surface_adapter_requests.load(Ordering::Relaxed), 1);
    assert_eq!(
        *calls
            .adapter_requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        vec![adapter_options, adapter_options]
    );
    assert_eq!(
        *calls
            .device_descriptors
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        vec![device_descriptor.clone(), device_descriptor]
    );
}

#[test]
fn failed_replay_is_explicit_and_does_not_commit_a_partial_generation() {
    let instance = CountingInstance::new();
    let calls = Arc::clone(&instance.calls);
    let mut rt = RenderRuntime::headless(instance, rhi::Extent3d::new_2d(SIZE, SIZE))
        .expect("the initial null device comes up");
    let attempts = Arc::new(AtomicUsize::new(0));
    let attempts_for_recipe = Arc::clone(&attempts);
    let resident = rt
        .register_replay("fallible-resident-buffer", move |device, _queue| {
            if attempts_for_recipe.fetch_add(1, Ordering::Relaxed) != 0 {
                return Err(crate::ReplayError::new("asset payload is unavailable"));
            }
            Ok(ResidentBuffer {
                id: device.create_buffer(&rhi::BufferDescriptor {
                    label: Some("fallible-resident-buffer".into()),
                    size: 16,
                    usage: rhi::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                }),
                byte_len: 16,
            })
        })
        .expect("the initial resource is available");
    let old_device = rt.device_handle();
    let old_resident = resident.current();

    let error = rt.recover_device().expect_err("the replay must fail");
    let crate::RecoveryError::Replay(failure) = error else {
        panic!("unexpected recovery error: {error:?}");
    };
    assert_eq!(failure.label(), "fallible-resident-buffer");
    assert_eq!(
        failure.source_error().detail(),
        "asset payload is unavailable"
    );
    assert_eq!(rt.device_generation(), 0);
    assert!(Arc::ptr_eq(&old_device, &rt.device_handle()));
    assert!(Arc::ptr_eq(&old_resident, &resident.current()));
    assert_eq!(rt.device().live_resource_count(), 1);
    assert_eq!(calls.device_creations.load(Ordering::Relaxed), 2);
    assert_eq!(attempts.load(Ordering::Relaxed), 2);
}

#[test]
fn recovery_retries_sticky_loss_and_commits_all_live_entries_together() {
    use core::sync::atomic::AtomicBool;
    let instance = CountingInstance::new();
    let calls = instance.calls.clone();
    let mut rt = RenderRuntime::headless(instance, rhi::Extent3d::new_2d(SIZE, SIZE)).unwrap();
    let count = Arc::new(AtomicUsize::new(0));
    let counter = count.clone();
    let first = rt
        .register_replay("first", move |_, _| {
            Ok(counter.fetch_add(1, Ordering::SeqCst))
        })
        .unwrap();
    let fail = Arc::new(AtomicBool::new(false));
    let failing = fail.clone();
    let second = rt
        .register_replay("second", move |_, _| {
            if failing.load(Ordering::SeqCst) {
                Err("retry later".into())
            } else {
                Ok(42_u32)
            }
        })
        .unwrap();
    let obsolete = rt.register_replay("dropped", |_, _| Ok(0_u32)).unwrap();
    drop(obsolete);
    let original = first.current();
    let original_second = second.current();
    calls.report_loss(rhi::DeviceLostReason::Destroyed);
    assert_eq!(rt.device_lost(), Some(rhi::DeviceLostReason::Destroyed));
    fail.store(true, Ordering::SeqCst);
    for _ in 0..2 {
        assert!(rt.recover_if_lost().is_err());
        assert_eq!(rt.device_generation(), 0);
        assert!(Arc::ptr_eq(&first.current(), &original));
        assert!(Arc::ptr_eq(&second.current(), &original_second));
        assert_eq!(rt.device_lost(), Some(rhi::DeviceLostReason::Destroyed));
    }
    fail.store(false, Ordering::SeqCst);
    let report = rt.recover_if_lost().unwrap().unwrap();
    assert_eq!(report.reason, Some(rhi::DeviceLostReason::Destroyed));
    assert_eq!(
        report.replayed_resources, 2,
        "dropped handles are not replayed"
    );
    assert_eq!(*first.current(), 3);
    assert!(!Arc::ptr_eq(&second.current(), &original_second));
    assert!(rt.device_lost().is_none());
    assert!(rt.recover_if_lost().unwrap().is_none());
}

#[test]
fn recovery_explicit_replacement_clears_a_previously_polled_loss() {
    let instance = CountingInstance::new();
    let calls = instance.calls.clone();
    let mut rt = RenderRuntime::headless(instance, rhi::Extent3d::new_2d(SIZE, SIZE)).unwrap();
    calls.report_loss(rhi::DeviceLostReason::Destroyed);
    assert!(rt.device_lost().is_some());
    rt.recover_device().unwrap();
    assert!(rt.device_lost().is_none());
    assert!(rt.recover_if_lost().unwrap().is_none());
    assert_eq!(rt.device_generation(), 1);
}

#[test]
fn tracked_submission_exposes_monotonic_completion_values() {
    let rt = RenderRuntime::headless(NullInstance::new(), rhi::Extent3d::new_2d(SIZE, SIZE))
        .expect("the null device always comes up");
    let texture = rt.device().create_texture(&rhi::TextureDescriptor::new_2d(
        rhi::Extent3d::new_2d(SIZE, SIZE),
        rhi::TextureFormat::Rgba8Unorm,
        TextureUsages::RENDER_ATTACHMENT,
    ));
    let view = rt
        .device()
        .create_texture_view(texture, &rhi::TextureViewDescriptor::default());

    let mut graph = rt.new_graph();
    let target = graph.import_texture(
        "tracked-target",
        TextureDesc::color(rhi::TextureFormat::Rgba8Unorm),
        ImportedTexture {
            texture,
            default_view: view,
            entry_state: TextureState::initial(),
            exit_state: TextureState::initial(),
            mip_levels: 1,
            array_layers: 1,
        },
    );
    graph.add_raster_pass("tracked-clear", PassFlags::SIDE_EFFECT, |b| {
        let _ = b.color_attachment(target);
        |_ctx| {}
    });

    let first = rt
        .execute_tracked(graph)
        .expect("the tracked graph executes");
    assert_eq!(first, 1);
    assert_eq!(rt.completed_value(), first, "null completion is immediate");

    let mut next = rt.new_graph();
    let target = next.import_texture(
        "tracked-target-2",
        TextureDesc::color(rhi::TextureFormat::Rgba8Unorm),
        ImportedTexture {
            texture,
            default_view: view,
            entry_state: TextureState::initial(),
            exit_state: TextureState::initial(),
            mip_levels: 1,
            array_layers: 1,
        },
    );
    next.add_raster_pass("tracked-clear-2", PassFlags::SIDE_EFFECT, |b| {
        let _ = b.color_attachment(target);
        |_ctx| {}
    });
    let second = rt.execute_tracked(next).expect("the second graph executes");
    assert_eq!(second, 2);
    assert_eq!(rt.completed_value(), second);

    rt.wait_idle();
    rt.device().destroy_texture_view(view);
    rt.device().destroy_texture(texture);
}

#[test]
fn discarding_a_frame_reclaims_its_image_ids() {
    let mut rt = RenderRuntime::headless(NullInstance::new(), rhi::Extent3d::new_2d(SIZE, SIZE))
        .expect("the null device always comes up");
    rt.create_surface(Arc::new(FakeWindow))
        .expect("the null surface is created");
    rt.configure_surface_default(SIZE, SIZE)
        .expect("the null surface advertises a usable format");

    let baseline = rt.device().live_resource_count();
    let frame = rt
        .surface()
        .expect("surface exists")
        .acquire()
        .expect("acquire");
    assert!(
        rt.device().live_resource_count() > baseline,
        "an acquired frame registers its image on the device"
    );
    rt.surface().expect("surface exists").discard(frame);
    assert_eq!(
        rt.device().live_resource_count(),
        baseline,
        "discarding reclaims the image ids so a failed frame cannot leak"
    );
}

#[test]
fn instance_enumerates_adapters_through_the_rhi() {
    let instance = WgpuInstance::new();
    let adapters = <WgpuInstance as Instance>::enumerate_adapters(&instance);
    // A GPU-less sandbox legitimately reports nothing; skip in that case.
    for adapter in &adapters {
        let info = adapter.info();
        assert!(!info.name.is_empty(), "adapter reports a name");
        let caps = adapter.capabilities();
        assert_eq!(caps.backend, info.backend, "caps and info agree on backend");
    }
}

#[test]
fn rhi_graph_clears_imported_target() {
    let Some(rt) = RenderRuntime::wgpu(rhi::Extent3d::new_2d(SIZE, SIZE)).ok() else {
        assert!(
            !require_gpu(),
            "PRISM_REQUIRE_GPU is set but no adapter was found"
        );
        return;
    };
    let device = rt.device();

    // An offscreen target the graph imports (so it survives culling and its
    // result outlives the frame) and clears.
    let texture = device.create_texture(&rhi::TextureDescriptor::new_2d(
        rhi::Extent3d::new_2d(SIZE, SIZE),
        rhi::TextureFormat::Rgba8Unorm,
        TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
    ));
    let view = device.create_texture_view(texture, &rhi::TextureViewDescriptor::default());

    let mut graph = rt.new_graph();
    let target = graph.import_texture(
        "offscreen",
        TextureDesc::color(rhi::TextureFormat::Rgba8Unorm).with_usage(TextureUsages::COPY_SRC),
        ImportedTexture {
            texture,
            default_view: view,
            entry_state: TextureState::initial(),
            exit_state: TextureState::copy_src(),
            mip_levels: 1,
            array_layers: 1,
        },
    );
    graph.add_raster_pass("clear", PassFlags::SIDE_EFFECT, |b| {
        let _ = b.color_attachment(target);
        |_ctx| {}
    });

    rt.execute(graph).expect("graph executes on the RHI");
    assert!(rt.wait_idle(), "the submitted frame completes");

    let pixels = rt.queue().read_texture_rgba8(texture, SIZE, SIZE);
    assert_eq!(pixels.len(), (SIZE * SIZE * 4) as usize);
    assert!(
        pixels.chunks_exact(4).all(|p| p == [0, 0, 0, 255]),
        "a freshly written version-0 target is cleared to opaque black"
    );

    device.destroy_texture_view(view);
    device.destroy_texture(texture);
}

/// A back-to-front triangle in clip space, three `vec3<f32>` positions.
const TRIANGLE: [f32; 9] = [0.0, 0.6, 0.0, -0.6, -0.6, 0.0, 0.6, -0.6, 0.0];

/// Builds a one-pass graph that clears `target`, giving the frame pacer real
/// submissions to bound.
///
/// `target` is imported (rather than created transiently) so the same texture
/// can back every frame without the graph leaking a fresh allocation per call.
fn clear_graph(
    rt: &RenderRuntime<WgpuInstance>,
    texture: rhi::TextureId,
    view: rhi::TextureViewId,
) -> prism_render_graph::RenderGraph {
    let mut graph = rt.new_graph();
    let target = graph.import_texture(
        "paced-target",
        TextureDesc::color(rhi::TextureFormat::Rgba8Unorm).with_usage(TextureUsages::COPY_SRC),
        ImportedTexture {
            texture,
            default_view: view,
            entry_state: TextureState::initial(),
            exit_state: TextureState::copy_src(),
            mip_levels: 1,
            array_layers: 1,
        },
    );
    graph.add_raster_pass("paced-clear", PassFlags::SIDE_EFFECT, |b| {
        let _ = b.color_attachment(target);
        |_ctx| {}
    });
    graph
}

/// Submitting more frames than the configured maximum must never leave more
/// than `max` outstanding, and `wait_idle` must drain the pipeline to zero.
#[test]
fn pacer_bounds_frames_in_flight_and_drains() {
    const MAX_IN_FLIGHT: usize = 2;
    const FRAMES: usize = 5;

    let extent = rhi::Extent3d::new_2d(SIZE, SIZE);
    let Some(rt) = RenderRuntime::<WgpuInstance>::wgpu_paced(extent, MAX_IN_FLIGHT).ok() else {
        assert!(
            !require_gpu(),
            "PRISM_REQUIRE_GPU is set but no adapter was found"
        );
        return;
    };
    assert_eq!(
        rt.max_frames_in_flight(),
        MAX_IN_FLIGHT,
        "the configured bound is reported back"
    );
    assert_eq!(
        rt.frames_in_flight(),
        0,
        "a fresh runtime has no frames in flight"
    );

    let device = rt.device();
    let texture = device.create_texture(&rhi::TextureDescriptor::new_2d(
        extent,
        rhi::TextureFormat::Rgba8Unorm,
        TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
    ));
    let view = device.create_texture_view(texture, &rhi::TextureViewDescriptor::default());

    for frame in 0..FRAMES {
        rt.execute(clear_graph(&rt, texture, view))
            .expect("a paced frame executes on the RHI");
        assert!(
            rt.frames_in_flight() <= MAX_IN_FLIGHT,
            "frame {frame} left {} outstanding, bound is {MAX_IN_FLIGHT}",
            rt.frames_in_flight()
        );
    }
    assert_eq!(
        rt.frames_in_flight(),
        MAX_IN_FLIGHT,
        "after more frames than the bound, the pipeline stays exactly full"
    );

    assert!(rt.wait_idle(), "wait_idle confirms every outstanding frame");
    assert_eq!(
        rt.frames_in_flight(),
        0,
        "wait_idle leaves no outstanding frames"
    );

    // The queue is idle now, so the imported target is safe to release.
    device.destroy_texture_view(view);
    device.destroy_texture(texture);
}

/// Minimal pass-through vertex + constant-color fragment shader.
const TRIANGLE_WGSL: &str = "\
@vertex fn vs(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> {\n\
    return vec4<f32>(pos, 1.0);\n\
}\n\
@fragment fn fs() -> @location(0) vec4<f32> {\n\
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);\n\
}\n";

#[test]
fn rhi_graph_draws_a_triangle() {
    let Some(rt) = RenderRuntime::wgpu(rhi::Extent3d::new_2d(SIZE, SIZE)).ok() else {
        assert!(
            !require_gpu(),
            "PRISM_REQUIRE_GPU is set but no adapter was found"
        );
        return;
    };
    let device = rt.device();
    let queue = rt.queue();

    // Vertex buffer holding the triangle.
    let vertex_bytes: Vec<u8> = TRIANGLE.iter().flat_map(|v| v.to_le_bytes()).collect();
    let vertices = device.create_buffer(&rhi::BufferDescriptor {
        label: Some("triangle-vertices".into()),
        size: vertex_bytes.len() as u64,
        usage: rhi::BufferUsages::VERTEX | rhi::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(vertices, 0, &vertex_bytes);

    // Pipeline: pass-through vertex stage, constant-red fragment stage.
    let module = device.create_shader_module(&ShaderModuleDescriptor::wgsl(TRIANGLE_WGSL));
    let layout = device.create_pipeline_layout(&rhi::PipelineLayoutDescriptor {
        label: Some("triangle-pl".into()),
        bind_group_layouts: vec![],
        push_constant_ranges: vec![],
    });
    let pipeline = device.create_render_pipeline(&rhi::RenderPipelineDescriptor {
        label: Some("triangle".into()),
        layout: Some(layout),
        vertex: VertexState {
            module,
            entry_point: "vs".into(),
            buffers: vec![rhi::VertexBufferLayout::packed(
                VertexStepMode::Vertex,
                0,
                &[VertexFormat::Float32x3],
            )],
        },
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        fragment: Some(FragmentState {
            module,
            entry_point: "fs".into(),
            targets: vec![Some(ColorTargetState::opaque(
                rhi::TextureFormat::Rgba8Unorm,
            ))],
        }),
    });

    // Offscreen target the graph imports, clears, and draws into.
    let target_texture = device.create_texture(&rhi::TextureDescriptor::new_2d(
        rhi::Extent3d::new_2d(SIZE, SIZE),
        rhi::TextureFormat::Rgba8Unorm,
        TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
    ));
    let target_view =
        device.create_texture_view(target_texture, &rhi::TextureViewDescriptor::default());

    let mut graph = rt.new_graph();
    let target = graph.import_texture(
        "triangle-target",
        TextureDesc::color(rhi::TextureFormat::Rgba8Unorm).with_usage(TextureUsages::COPY_SRC),
        ImportedTexture {
            texture: target_texture,
            default_view: target_view,
            entry_state: TextureState::initial(),
            exit_state: TextureState::copy_src(),
            mip_levels: 1,
            array_layers: 1,
        },
    );
    graph.add_raster_pass("draw-triangle", PassFlags::SIDE_EFFECT, move |b| {
        let _ = b.color_attachment(target);
        move |ctx| {
            ctx.set_render_pipeline(pipeline);
            ctx.set_vertex_buffer(0, vertices, 0);
            ctx.draw(0..3, 0..1);
        }
    });
    rt.execute(graph)
        .expect("triangle frame executes on the RHI");
    assert!(rt.wait_idle(), "the submitted frame completes");

    // The compiler clears the fresh target to opaque black, then the triangle
    // paints the center red and leaves the corners black.
    let pixels = queue.read_texture_rgba8(target_texture, SIZE, SIZE);
    let at = |x: u32, y: u32| {
        let i = ((y * SIZE + x) * 4) as usize;
        [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
    };
    let center = at(SIZE / 2, SIZE / 2);
    assert!(
        center[0] > 200 && center[1] < 50 && center[2] < 50 && center[3] == 255,
        "the triangle paints the center red, got {center:?}"
    );
    let corner = at(1, 1);
    assert_eq!(
        corner,
        [0, 0, 0, 255],
        "a corner outside the triangle stays black"
    );

    device.destroy_render_pipeline(pipeline);
    device.destroy_pipeline_layout(layout);
    device.destroy_shader_module(module);
    device.destroy_buffer(vertices);
    device.destroy_texture_view(target_view);
    device.destroy_texture(target_texture);
}

/// A null/headless backend never reports device loss: the default
/// [`RenderDevice::take_device_lost`](prism_render_driver::RenderDevice::take_device_lost)
/// returns `None`, so the runtime's [`device_lost`](RenderRuntime::device_lost)
/// accessor does too. The accessor is the seam a host layer polls to drive
/// device-loss recovery.
#[test]
fn headless_backend_reports_no_device_loss() {
    let rt = RenderRuntime::headless(NullInstance::new(), rhi::Extent3d::new_2d(SIZE, SIZE))
        .expect("the null device always comes up");
    assert!(rt.device_lost().is_none());
}
