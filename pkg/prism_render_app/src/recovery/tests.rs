use super::*;
use prism_app::{
    lifecycle::{AppLifecycle, DeviceRestored},
    App,
};
use prism_ecs::event::{EventCursor, Events};
use prism_render_driver as rhi;
use prism_render_driver_null::{NullAdapter, NullDevice, NullInstance, NullQueue, NullSurface};
use rhi::{RenderDevice, RenderQueue};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc, Mutex,
};

use crate::{
    DeviceRecoveryStatus, GpuResourceAppExt, PrismRenderPlugin, RenderApp, RenderAppConfig,
    RenderFrameStats, SpineMeshPipeline,
};

#[derive(Default)]
struct Controls {
    fail_device: AtomicBool,
    created: AtomicUsize,
    loss: Mutex<Option<Arc<Mutex<Option<rhi::DeviceLostReason>>>>>,
}

impl Controls {
    fn lose(&self) {
        let loss = self.loss.lock().unwrap().as_ref().unwrap().clone();
        *loss.lock().unwrap() = Some(rhi::DeviceLostReason::Destroyed);
    }
}

struct TestDevice {
    inner: NullDevice,
    loss: Arc<Mutex<Option<rhi::DeviceLostReason>>>,
}

macro_rules! create {
    ($($method:ident($descriptor:ty) -> $id:ty;)*) => {$ (
        fn $method(&self, descriptor: &$descriptor) -> $id {
            self.inner.$method(descriptor)
        }
    )*};
}

macro_rules! destroy {
    ($($method:ident($id:ty);)*) => {$ (
        fn $method(&self, id: $id) { self.inner.$method(id); }
    )*};
}

impl RenderDevice for TestDevice {
    fn capabilities(&self) -> &rhi::DeviceCapabilities {
        self.inner.capabilities()
    }
    create! {
        create_buffer(rhi::BufferDescriptor) -> rhi::BufferId;
        create_texture(rhi::TextureDescriptor) -> rhi::TextureId;
        create_sampler(rhi::SamplerDescriptor) -> rhi::SamplerId;
        create_shader_module(rhi::ShaderModuleDescriptor) -> rhi::ShaderModuleId;
        create_bind_group_layout(rhi::BindGroupLayoutDescriptor) -> rhi::BindGroupLayoutId;
        create_bind_group(rhi::BindGroupDescriptor) -> rhi::BindGroupId;
        create_pipeline_layout(rhi::PipelineLayoutDescriptor) -> rhi::PipelineLayoutId;
        create_render_pipeline(rhi::RenderPipelineDescriptor) -> rhi::RenderPipelineId;
        create_compute_pipeline(rhi::ComputePipelineDescriptor) -> rhi::ComputePipelineId;
    }
    fn create_texture_view(
        &self,
        texture: rhi::TextureId,
        descriptor: &rhi::TextureViewDescriptor,
    ) -> rhi::TextureViewId {
        self.inner.create_texture_view(texture, descriptor)
    }
    fn take_device_lost(&self) -> Option<rhi::DeviceLostReason> {
        self.loss.lock().unwrap().take()
    }
    destroy! {
        destroy_buffer(rhi::BufferId);
        destroy_texture(rhi::TextureId);
        destroy_texture_view(rhi::TextureViewId);
        destroy_sampler(rhi::SamplerId);
        destroy_shader_module(rhi::ShaderModuleId);
        destroy_bind_group_layout(rhi::BindGroupLayoutId);
        destroy_bind_group(rhi::BindGroupId);
        destroy_pipeline_layout(rhi::PipelineLayoutId);
        destroy_render_pipeline(rhi::RenderPipelineId);
        destroy_compute_pipeline(rhi::ComputePipelineId);
    }
}

#[derive(Default)]
struct TestQueue {
    writes: Mutex<Vec<(rhi::BufferId, u64, Vec<u8>)>>,
}

impl RenderQueue for TestQueue {
    fn write_buffer(&self, buffer: rhi::BufferId, offset: u64, data: &[u8]) {
        self.writes
            .lock()
            .unwrap()
            .push((buffer, offset, data.to_vec()));
    }
    fn write_texture(
        &self,
        destination: rhi::TextureWrite,
        data: &[u8],
        layout: rhi::ImageDataLayout,
    ) {
        NullQueue.write_texture(destination, data, layout);
    }
    fn submit(&self, commands: &[rhi::CommandBuffer]) {
        NullQueue.submit(commands);
    }
}

impl FrameSync for TestQueue {
    fn submit_tracked(&self, commands: &[rhi::CommandBuffer]) -> rhi::CompletionToken {
        NullQueue.submit_tracked(commands)
    }
    fn wait_for(&self, token: &(dyn core::any::Any + Send + Sync)) -> bool {
        NullQueue.wait_for(token)
    }
    fn wait_idle(&self) -> bool {
        NullQueue.wait_idle()
    }
    fn on_submitted_work_done(&self, callback: Box<dyn FnOnce() + Send + 'static>) {
        NullQueue.on_submitted_work_done(callback);
    }
}

struct TestInstance(Arc<Controls>);

impl Instance for TestInstance {
    type Adapter = NullAdapter;
    type Device = TestDevice;
    type Queue = TestQueue;
    type Surface = NullSurface;
    fn enumerate_adapters(&self) -> Vec<NullAdapter> {
        NullInstance::new().enumerate_adapters()
    }
    fn request_adapter(&self, options: &rhi::RequestAdapterOptions) -> Option<NullAdapter> {
        NullInstance::new().request_adapter(options)
    }
    fn request_adapter_for_surface(
        &self,
        options: &rhi::RequestAdapterOptions,
        surface: &NullSurface,
    ) -> Option<NullAdapter> {
        NullInstance::new().request_adapter_for_surface(options, surface)
    }
    fn create_device(
        &self,
        adapter: &NullAdapter,
        descriptor: &rhi::DeviceDescriptor,
    ) -> Result<(TestDevice, TestQueue), rhi::DeviceError> {
        if self.0.fail_device.load(Ordering::SeqCst) {
            return Err(rhi::DeviceError::Request(
                "injected replacement failure".into(),
            ));
        }
        let (inner, _) = NullInstance::new().create_device(adapter, descriptor)?;
        let loss = Arc::new(Mutex::new(None));
        *self.0.loss.lock().unwrap() = Some(loss.clone());
        self.0.created.fetch_add(1, Ordering::SeqCst);
        Ok((TestDevice { inner, loss }, TestQueue::default()))
    }
    fn create_surface<W>(
        &self,
        _window: Arc<W>,
        _device: &TestDevice,
        _queue: &TestQueue,
    ) -> Result<NullSurface, rhi::SurfaceError> {
        // This loss-injecting backend is headless; surface replay belongs to
        // the runtime's retained-window tests.
        Err(rhi::SurfaceError::Unsupported)
    }
}

fn app() -> (App, Arc<Controls>) {
    let controls = Arc::new(Controls::default());
    let mut app = App::new();
    app.add_plugins(PrismRenderPlugin::with_config(
        TestInstance(controls.clone()),
        RenderAppConfig {
            extent: rhi::Extent3d::new_2d(16, 16),
            material_capacity: 2,
            material_parameter_words: 32,
            material_texture_rows: 2,
            ..Default::default()
        },
    ));
    (app, controls)
}

fn runtime(app: &App) -> &prism_render_runtime::RenderRuntime<TestInstance> {
    app.get_sub_app(RenderApp)
        .unwrap()
        .world
        .resource::<RenderRuntimeResource<TestInstance>>()
        .runtime()
}

#[derive(Resource)]
struct StartupGeneration(u64);

impl prism_ecs::resource::FromWorld for StartupGeneration {
    fn from_world(world: &mut World) -> Self {
        Self(
            world
                .resource::<RenderRuntimeResource<TestInstance>>()
                .runtime()
                .device_generation(),
        )
    }
}

#[test]
fn recovery_replays_resources_rebuilds_startup_and_resumes_frames() {
    let (mut app, controls) = app();
    app.sub_app_mut(RenderApp)
        .unwrap()
        .init_gpu_resource::<StartupGeneration>();
    let recipe_calls = Arc::new(AtomicUsize::new(0));
    let calls = recipe_calls.clone();
    let replay = app
        .sub_app_mut(RenderApp)
        .unwrap()
        .world
        .resource_mut::<RenderRuntimeResource<TestInstance>>()
        .runtime_mut()
        .register_replay("test resident", move |device, _| {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(device.create_buffer(&rhi::BufferDescriptor {
                label: None,
                size: 16,
                usage: rhi::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }))
        })
        .unwrap();
    app.update();
    let old_device = runtime(&app).device_handle();
    let old_count = old_device.inner.live_resource_count();
    let old_replay = replay.current();
    let before = app
        .get_sub_app(RenderApp)
        .unwrap()
        .world
        .resource::<RenderFrameStats>()
        .frames_submitted;
    controls.lose();
    // Polling beforehand must not drain the loss away from the recovery driver.
    assert_eq!(
        runtime(&app).device_lost(),
        Some(rhi::DeviceLostReason::Destroyed)
    );
    app.update();
    assert_eq!(runtime(&app).device_generation(), 0);
    assert_eq!(
        app.get_sub_app(RenderApp)
            .unwrap()
            .world
            .resource::<RenderFrameStats>()
            .frames_submitted,
        before
    );
    app.update();
    assert_eq!(runtime(&app).device_generation(), 1);
    assert!(!Arc::ptr_eq(&old_device, &runtime(&app).device_handle()));
    assert!(!Arc::ptr_eq(&old_replay, &replay.current()));
    assert_eq!(recipe_calls.load(Ordering::SeqCst), 2);
    assert_eq!(
        old_device.inner.live_resource_count(),
        old_count,
        "old ids were not destroyed on the replacement"
    );
    let render = &app.get_sub_app(RenderApp).unwrap().world;
    assert_eq!(render.resource::<StartupGeneration>().0, 1);
    assert_eq!(
        render.resource::<RenderFrameStats>().frames_submitted,
        before
    );
    // Target color/depth and views (4), three material buffers, shader, replay.
    assert_eq!(runtime(&app).device().inner.live_resource_count(), 9);
    app.update();
    assert!(!app.world().resource::<DeviceRecoveryStatus>().recovering);
    assert_eq!(
        app.world()
            .resource::<DeviceRecoveryStatus>()
            .last_report
            .unwrap()
            .replayed_resources,
        1
    );
    let mut restored = EventCursor::<DeviceRestored>::new();
    assert_eq!(
        restored
            .read(app.world().resource::<Events<DeviceRestored>>())
            .count(),
        1
    );
    app.update();
    assert_eq!(
        *app.world().resource::<AppLifecycle>(),
        AppLifecycle::Running
    );
    assert!(
        app.get_sub_app(RenderApp)
            .unwrap()
            .world
            .resource::<RenderFrameStats>()
            .frames_submitted
            > before
    );
}

#[test]
fn recovery_device_failure_stays_recovering_then_retries() {
    let (mut app, controls) = app();
    app.update();
    controls.lose();
    controls.fail_device.store(true, Ordering::SeqCst);
    for _ in 0..4 {
        app.update();
    }
    let status = app.world().resource::<DeviceRecoveryStatus>();
    assert!(status.recovering);
    assert!(status.attempts >= 1);
    assert!(status
        .last_error
        .as_ref()
        .unwrap()
        .contains("injected replacement failure"));
    assert_eq!(
        *app.world().resource::<AppLifecycle>(),
        AppLifecycle::Recovering
    );
    assert_eq!(runtime(&app).device_generation(), 0);
    let mut restored = EventCursor::<DeviceRestored>::new();
    assert_eq!(
        restored
            .read(app.world().resource::<Events<DeviceRestored>>())
            .count(),
        0
    );
    controls.fail_device.store(false, Ordering::SeqCst);
    app.update();
    app.update();
    assert_eq!(runtime(&app).device_generation(), 1);
    assert!(!app.world().resource::<DeviceRecoveryStatus>().recovering);
    assert_eq!(
        restored
            .read(app.world().resource::<Events<DeviceRestored>>())
            .count(),
        1
    );
}

#[test]
fn recovery_replay_failure_does_not_publish_restored_or_commit_generation() {
    let (mut app, controls) = app();
    let fail = Arc::new(AtomicBool::new(false));
    let fail_recipe = fail.clone();
    let handle = app
        .sub_app_mut(RenderApp)
        .unwrap()
        .world
        .resource_mut::<RenderRuntimeResource<TestInstance>>()
        .runtime_mut()
        .register_replay("fallible recipe", move |_, _| {
            if fail_recipe.load(Ordering::SeqCst) {
                Err(prism_render_runtime::ReplayError::new(
                    "payload unavailable",
                ))
            } else {
                Ok(42_u32)
            }
        })
        .unwrap();
    let old = handle.current();
    app.update();
    controls.lose();
    fail.store(true, Ordering::SeqCst);
    for _ in 0..4 {
        app.update();
    }
    assert_eq!(runtime(&app).device_generation(), 0);
    assert!(Arc::ptr_eq(&old, &handle.current()));
    assert!(app
        .world()
        .resource::<DeviceRecoveryStatus>()
        .last_error
        .as_ref()
        .unwrap()
        .contains("payload unavailable"));
    assert_eq!(
        *app.world().resource::<AppLifecycle>(),
        AppLifecycle::Recovering
    );
    fail.store(false, Ordering::SeqCst);
    app.update();
    app.update();
    assert_eq!(runtime(&app).device_generation(), 1);
    assert!(!app.world().resource::<DeviceRecoveryStatus>().recovering);
}

#[test]
fn recovery_custom_provider_fails_before_replacing_device() {
    let (mut app, controls) = app();
    let provider = SpineMeshPipeline::new(runtime(&app).device(), crate::RENDER_TARGET_FORMAT);
    app.sub_app_mut(RenderApp)
        .unwrap()
        .world
        .insert_resource(MeshPipelines::new(provider));
    app.update();
    controls.lose();
    for _ in 0..4 {
        app.update();
    }
    let status = app.world().resource::<DeviceRecoveryStatus>();
    assert!(status.recovering);
    assert!(status
        .last_error
        .as_ref()
        .unwrap()
        .contains("custom mesh pipeline provider"));
    assert_eq!(controls.created.load(Ordering::SeqCst), 1);
    assert_eq!(runtime(&app).device_generation(), 0);
}

#[test]
fn recovery_material_buffers_restore_bytes_and_preserve_allocations() {
    let (app, _) = app();
    let old_device = runtime(&app).device();
    let mut buffers = MaterialParameterBuffers::new(old_device, 2, 64, 4, rhi::BufferUsages::NONE);
    let record = prism_render_material::fallback_material_record(
        prism_render_architecture::abi::GenerationalHandle::new(0, 1),
        1,
    );
    buffers.sync(old_device, runtime(&app).queue(), [(0, record)], 7);
    let header = *buffers.header(0).unwrap();
    let words = buffers.parameters_of(0).unwrap().to_vec();
    let used = buffers.used_words();
    let instance = TestInstance(Arc::new(Controls::default()));
    let adapter = instance.request_adapter(&Default::default()).unwrap();
    let (device, queue) = instance
        .create_device(&adapter, &Default::default())
        .unwrap();
    buffers.recover(&device, &queue);
    assert_eq!(device.inner.live_resource_count(), 3);
    assert_eq!(buffers.header(0).unwrap(), &header);
    assert_eq!(buffers.parameters_of(0).unwrap(), words);
    assert_eq!(buffers.used_words(), used);
    let writes = queue.writes.lock().unwrap();
    assert_eq!(writes.len(), 3);
    let header_bytes: &[u8] = bytemuck::bytes_of(&header);
    assert_eq!(&writes[0].2[..header_bytes.len()], header_bytes);
    let word_bytes: &[u8] = bytemuck::cast_slice(&words);
    assert_eq!(&writes[1].2[..word_bytes.len()], word_bytes);
    assert_eq!(
        writes[2].2.len(),
        4 * core::mem::size_of::<prism_render_material::GpuMaterialTexture>()
    );
}

#[derive(Clone)]
struct RecoverySource {
    bytes: Vec<u8>,
    retry: Arc<AtomicBool>,
}

impl prism_asset::Asset for RecoverySource {
    const TYPE_NAME: &'static str = "device-recovery-test-source";
}

struct RecoveryAsset(rhi::BufferId);

impl crate::RenderAsset for RecoveryAsset {
    type SourceAsset = RecoverySource;

    fn prepare(
        source: &RecoverySource,
        id: prism_asset::AssetId<RecoverySource>,
        device: &dyn RenderDevice,
        queue: &dyn RenderQueue,
        previous: Option<&Self>,
    ) -> Result<Self, crate::PrepareAssetError<RecoverySource>> {
        if source.retry.load(Ordering::SeqCst) {
            return Err(crate::PrepareAssetError::RetryNextUpdate(id));
        }
        assert!(previous.is_none(), "old device assets must be invalidated");
        let buffer = device.create_buffer(&rhi::BufferDescriptor {
            label: None,
            size: source.bytes.len() as u64,
            usage: rhi::BufferUsages::STORAGE | rhi::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(buffer, 0, &source.bytes);
        Ok(Self(buffer))
    }

    fn unload(prepared: &Self, device: &dyn RenderDevice) {
        device.destroy_buffer(prepared.0);
    }
}

#[test]
fn recovery_reextracts_quiet_assets_and_waits_for_prepare_retry() {
    let (mut app, controls) = app();
    crate::register_render_assets::<TestInstance, RecoveryAsset>(&mut app);
    let retry = Arc::new(AtomicBool::new(false));
    let mut sources = prism_asset::Assets::<RecoverySource>::new();
    let handle = sources.insert(RecoverySource {
        bytes: vec![1, 2, 3, 4],
        retry: retry.clone(),
    });
    let id = handle.id();
    app.insert_resource(sources);
    // An unrelated server must not hide the compatibility source arena.
    app.insert_resource(prism_asset::AssetServer::new());
    app.update();
    assert!(app
        .get_sub_app(RenderApp)
        .unwrap()
        .world
        .resource::<crate::RenderAssets<RecoveryAsset>>()
        .contains(id));
    controls.lose();
    app.update();
    retry.store(true, Ordering::SeqCst);
    app.update();
    assert_eq!(runtime(&app).device_generation(), 1);
    assert!(!app
        .get_sub_app(RenderApp)
        .unwrap()
        .world
        .resource::<crate::RenderAssets<RecoveryAsset>>()
        .contains(id));
    app.update();
    app.update();
    assert!(app.world().resource::<DeviceRecoveryStatus>().recovering);
    assert_eq!(
        *app.world().resource::<AppLifecycle>(),
        AppLifecycle::Recovering
    );
    let mut restored = EventCursor::<DeviceRestored>::new();
    assert_eq!(
        restored
            .read(app.world().resource::<Events<DeviceRestored>>())
            .count(),
        0
    );
    retry.store(false, Ordering::SeqCst);
    app.update();
    assert!(app
        .get_sub_app(RenderApp)
        .unwrap()
        .world
        .resource::<crate::RenderAssets<RecoveryAsset>>()
        .contains(id));
    app.update();
    assert!(!app.world().resource::<DeviceRecoveryStatus>().recovering);
    assert_eq!(
        restored
            .read(app.world().resource::<Events<DeviceRestored>>())
            .count(),
        1
    );
    assert!(runtime(&app)
        .queue()
        .writes
        .lock()
        .unwrap()
        .iter()
        .any(|(_, _, bytes)| bytes == &[1, 2, 3, 4]));
}

#[test]
fn recovery_startup_panic_is_observable_and_can_retry_without_another_device() {
    let (mut app, controls) = app();
    let fail = Arc::new(AtomicBool::new(false));
    let fail_system = fail.clone();
    app.sub_app_mut(RenderApp)
        .unwrap()
        .world
        .resource_mut::<prism_ecs::schedule::Schedules>()
        .get_mut(crate::RenderStartup)
        .unwrap()
        .add_systems(move || {
            assert!(
                !fail_system.load(Ordering::SeqCst),
                "injected GPU startup failure"
            );
        });
    app.update();
    controls.lose();
    app.update();
    fail.store(true, Ordering::SeqCst);
    app.update();
    app.update();
    assert_eq!(runtime(&app).device_generation(), 1);
    let status = app.world().resource::<DeviceRecoveryStatus>();
    assert!(status.recovering);
    assert!(status
        .last_error
        .as_ref()
        .unwrap()
        .contains("injected GPU startup failure"));
    fail.store(false, Ordering::SeqCst);
    app.update();
    app.update();
    assert_eq!(controls.created.load(Ordering::SeqCst), 2);
    assert!(!app.world().resource::<DeviceRecoveryStatus>().recovering);
}

#[test]
fn recovery_replacement_lost_during_startup_never_publishes_restored() {
    let (mut app, controls) = app();
    let lose_replacement = controls.clone();
    app.sub_app_mut(RenderApp)
        .unwrap()
        .world
        .resource_mut::<prism_ecs::schedule::Schedules>()
        .get_mut(crate::RenderStartup)
        .unwrap()
        .add_systems(
            move |runtime: prism_ecs::prelude::Res<RenderRuntimeResource<TestInstance>>| {
                if runtime.runtime().device_generation() == 1 {
                    lose_replacement.lose();
                }
            },
        );
    app.update();
    controls.lose();
    app.update();
    app.update();
    assert_eq!(runtime(&app).device_generation(), 1);
    let mut restored = EventCursor::<DeviceRestored>::new();
    assert_eq!(
        restored
            .read(app.world().resource::<Events<DeviceRestored>>())
            .count(),
        0
    );
    app.update();
    assert!(app.world().resource::<DeviceRecoveryStatus>().recovering);
    assert!(app
        .world()
        .resource::<DeviceRecoveryStatus>()
        .last_error
        .as_ref()
        .unwrap()
        .contains("replacement device was lost"));
    assert_eq!(runtime(&app).device_generation(), 2);
    app.update();
    assert!(!app.world().resource::<DeviceRecoveryStatus>().recovering);
    assert_eq!(
        restored
            .read(app.world().resource::<Events<DeviceRestored>>())
            .count(),
        1
    );
}
