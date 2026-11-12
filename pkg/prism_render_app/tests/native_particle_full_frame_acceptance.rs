//! Native Particle production-chain acceptance.
//!
//! This drives the real application update/extract/render schedule with a live
//! emitter and a deferred queue completion boundary. It intentionally uses the
//! null RHI for deterministic graph/lifecycle evidence; shader execution is
//! covered by the focused physical-GPU particle tests.
#![forbid(unsafe_code)]

use std::sync::{Arc, Mutex, Weak};

use prism_app::App;
use prism_ecs::world::World;
use prism_math::{Frustum, Mat4, Plane, Vec3};
use prism_particle_app::{EmitterRuntimeComponent, ParticleRuntimePlugin};
use prism_render_app::visibility::{PrismCamera, PrismFrustum, PrismVisibilityPlugin};
use prism_render_app::{
    ExtractedRenderFabricFrame, NativeFrameExecution, NativeFramePhase, ParticleGpuState,
    ParticleNativeGraphReceipt, ParticleUploadStatus, PrismRenderPlugin, RenderApp,
    RenderAppConfig, RenderRecord, RenderRuntimeResource, SurfaceViewQueues,
};
use prism_render_architecture::particle::{
    emitter::{BurstSpawn, Emitter, EmitterShape, SpawnParams},
    runtime::{EmitterRuntime, EmitterRuntimeConfig},
    EmitterHandle,
};
use prism_render_driver as rhi;
use prism_render_driver::{
    Backend, DeviceCapabilities, Features, FrameSync, Instance, Limits, RenderQueue,
};
use prism_render_driver_null::{NullAdapter, NullDevice, NullInstance, NullQueue, NullSurface};
use prism_transform::GlobalTransform;

const EXTENT: rhi::Extent3d = rhi::Extent3d::new_2d(16, 12);
const PARTICLE_CAPACITY: u32 = 4;

#[derive(Default)]
struct DeferredWork {
    submitted: u64,
    completed: u64,
    callbacks: Vec<Box<dyn FnOnce() + Send + 'static>>,
}

#[derive(Default)]
struct DeferredCompletion {
    work: Mutex<DeferredWork>,
}

impl DeferredCompletion {
    fn complete_submitted(&self) {
        let callbacks = {
            let mut work = self.work.lock().expect("deferred queue lock");
            work.completed = work.submitted;
            std::mem::take(&mut work.callbacks)
        };
        for callback in callbacks {
            callback();
        }
    }
}

struct DeferredToken(Weak<DeferredCompletion>, u64);
struct DeferredQueue(Arc<DeferredCompletion>);

impl RenderQueue for DeferredQueue {
    fn write_buffer(&self, buffer: rhi::BufferId, offset: u64, data: &[u8]) {
        NullQueue.write_buffer(buffer, offset, data);
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
        drop(self.submit_tracked(commands));
    }
}

impl FrameSync for DeferredQueue {
    fn submit_tracked(&self, commands: &[rhi::CommandBuffer]) -> rhi::CompletionToken {
        NullQueue.submit(commands);
        let mut work = self.0.work.lock().expect("deferred queue lock");
        work.submitted += 1;
        Box::new(DeferredToken(Arc::downgrade(&self.0), work.submitted))
    }

    fn wait_for(&self, token: &(dyn core::any::Any + Send + Sync)) -> bool {
        token.downcast_ref::<DeferredToken>().is_some_and(|token| {
            token.0.ptr_eq(&Arc::downgrade(&self.0))
                && token.1 <= self.0.work.lock().expect("deferred queue lock").completed
        })
    }

    fn wait_idle(&self) -> bool {
        let work = self.0.work.lock().expect("deferred queue lock");
        work.completed == work.submitted
    }

    fn on_submitted_work_done(&self, callback: Box<dyn FnOnce() + Send + 'static>) {
        let mut work = self.0.work.lock().expect("deferred queue lock");
        if work.completed < work.submitted {
            work.callbacks.push(callback);
        } else {
            drop(work);
            callback();
        }
    }
}

struct DeferredInstance(Arc<DeferredCompletion>);

fn null_instance() -> NullInstance {
    NullInstance::with_capabilities(DeviceCapabilities {
        backend: Backend::Noop,
        features: Features::INDIRECT_DRAW,
        limits: Limits::desktop(),
    })
}

impl Instance for DeferredInstance {
    type Adapter = NullAdapter;
    type Device = NullDevice;
    type Queue = DeferredQueue;
    type Surface = NullSurface;

    fn enumerate_adapters(&self) -> Vec<Self::Adapter> {
        null_instance().enumerate_adapters()
    }

    fn request_adapter(&self, options: &rhi::RequestAdapterOptions) -> Option<Self::Adapter> {
        null_instance().request_adapter(options)
    }

    fn request_adapter_for_surface(
        &self,
        options: &rhi::RequestAdapterOptions,
        surface: &Self::Surface,
    ) -> Option<Self::Adapter> {
        null_instance().request_adapter_for_surface(options, surface)
    }

    fn create_device(
        &self,
        adapter: &Self::Adapter,
        descriptor: &rhi::DeviceDescriptor,
    ) -> Result<(Self::Device, Self::Queue), rhi::DeviceError> {
        let (device, _) = null_instance().create_device(adapter, descriptor)?;
        Ok((device, DeferredQueue(self.0.clone())))
    }

    fn create_surface<W>(
        &self,
        _window: Arc<W>,
        _device: &Self::Device,
        _queue: &Self::Queue,
    ) -> Result<Self::Surface, rhi::SurfaceError> {
        Err(rhi::SurfaceError::Unsupported)
    }
}

fn broad_frustum() -> PrismFrustum {
    PrismFrustum(Frustum::from_planes([
        Plane::new(Vec3::X, 10.0),
        Plane::new(-Vec3::X, 10.0),
        Plane::new(Vec3::Y, 10.0),
        Plane::new(-Vec3::Y, 10.0),
        Plane::new(Vec3::Z, 10.0),
        Plane::new(-Vec3::Z, 10.0),
    ]))
}

fn live_emitter() -> EmitterRuntime {
    let mut runtime = EmitterRuntime::new(
        Emitter::new(
            EmitterHandle(0),
            EmitterShape::Point,
            SpawnParams::default(),
        ),
        EmitterRuntimeConfig {
            capacity: 1,
            bursts: vec![BurstSpawn {
                time: 0.1,
                count: 1,
            }],
            lifetime: 10.0,
            ..Default::default()
        },
        prism_render_architecture::particle::Vec3::ZERO,
        prism_render_architecture::particle::Vec3::ZERO,
    )
    .expect("valid emitter configuration");
    runtime.step(0.1);
    runtime
}

fn acceptance_app() -> (App, Arc<DeferredCompletion>) {
    let completion = Arc::new(DeferredCompletion::default());
    let mut app = App::new();
    app.add_plugins(ParticleRuntimePlugin);
    let mut simulation = prism_particle_app::ParticleSimulationConfig::default();
    simulation.capacity = PARTICLE_CAPACITY;
    simulation.paused = true;
    app.insert_resource(simulation);
    app.world_mut()
        .spawn(EmitterRuntimeComponent::new(live_emitter()));
    app.add_plugins(PrismRenderPlugin::with_config(
        DeferredInstance(completion.clone()),
        RenderAppConfig::new(EXTENT).with_particle_capacity(PARTICLE_CAPACITY),
    ));
    app.add_plugins(PrismVisibilityPlugin);
    app.world_mut().spawn((
        PrismCamera::new(Mat4::IDENTITY, [0, 0, EXTENT.width, EXTENT.height]),
        broad_frustum(),
        GlobalTransform::IDENTITY,
    ));
    (app, completion)
}

fn render_world(app: &App) -> &World {
    &app.get_sub_app(RenderApp)
        .expect("native render sub-app")
        .world
}

fn install_particle_state(app: &mut App) -> ParticleNativeGraphReceipt {
    let render = app.sub_app_mut(RenderApp).expect("render sub-app");
    let runtime = render
        .world
        .resource::<RenderRuntimeResource<DeferredInstance>>()
        .runtime();
    let device = runtime.device_handle();
    let state = ParticleGpuState::new(
        device.as_ref(),
        device.clone(),
        runtime.device_generation(),
        PARTICLE_CAPACITY,
    )
    .expect("particle GPU state");
    let receipt = state.native_graph_receipt();
    render.world.insert_resource(state);
    receipt
}

#[test]
fn native_app_update_drives_particle_snapshot_to_completion() {
    let (mut app, completion) = acceptance_app();
    let before = install_particle_state(&mut app);
    assert_eq!(before, ParticleNativeGraphReceipt::default());

    // This single update drives snapshot extraction, particle preparation,
    // Fabric finalization, queue publication, native graph execution, record,
    // and submit through the production RenderApp schedule.
    app.update();

    let world = render_world(&app);
    let frame = world.resource::<RenderRecord>().frame_index;
    let snapshot = world.resource::<prism_particle_app::ExtractedParticleSnapshot>();
    assert_eq!(snapshot.frame(), frame);
    assert_eq!(
        snapshot.len(),
        1,
        "snapshot must contain the live emitter particle"
    );

    let state = world.resource::<ParticleGpuState>();
    assert_eq!(state.native_graph_receipt().last_frame, frame);
    assert_eq!(state.fabric_gpu_frame(), None);
    assert_eq!(state.fabric_gpu_views().count(), 1);
    assert!(state
        .fabric_gpu_views()
        .all(|(_, fabric)| !fabric.draws().is_empty()));

    let upload = world.resource::<ParticleUploadStatus>();
    assert_eq!(upload.error, None);
    let fabric = world.resource::<ExtractedRenderFabricFrame>();
    assert_eq!(fabric.error(), None);
    assert_eq!(fabric.frame().frame(), frame);
    assert!(fabric.frame().items().iter().any(
        |item| item.domain == prism_render_architecture::render_fabric::RenderDomain::Particle
    ));

    let queues = world.resource::<SurfaceViewQueues>();
    assert!(
        queues.is_ready(),
        "fill_queues did not publish a usable camera batch"
    );
    assert_eq!(queues.frame_index, frame);
    assert_eq!(queues.len(), 1);

    let lifecycle = world.resource::<NativeFrameExecution>();
    assert_eq!(lifecycle.frame_index(), frame);
    assert_eq!(lifecycle.phase(), NativeFramePhase::Submitted);
    assert_eq!(lifecycle.graph_count(), 1);
    assert!(lifecycle.imported_resources() > 0);
    assert!(lifecycle.compiled_passes() > 0);
    assert!(lifecycle.prepared_passes() > 0);
    assert!(lifecycle.recorded_passes() > 0);
    assert_eq!(lifecycle.submissions(), 1);
    let submission = lifecycle.last_submission().expect("native submit receipt");
    assert_eq!(lifecycle.completed_value(), 0);

    let receipt = state.native_graph_receipt();
    assert!(receipt.graph_appends > 0);
    assert!(receipt.renderer_passes > 0);
    assert_eq!(receipt.last_submitted_frame, frame);
    assert_eq!(receipt.submitted_graphs, 1);
    assert_eq!(receipt.completed_graphs, 0);

    let runtime = world
        .resource::<RenderRuntimeResource<DeferredInstance>>()
        .runtime();
    assert_eq!(runtime.completed_value(), 0);

    // Completion is driven by the queue callback, not inferred from submit.
    completion.complete_submitted();

    let world = render_world(&app);
    let state = world.resource::<ParticleGpuState>();
    let receipt = state.native_graph_receipt();
    assert_eq!(receipt.completed_graphs, 1);
    assert_eq!(receipt.last_completed_frame, frame);
    assert_eq!(receipt.submitted_graphs, 1);
    assert_eq!(
        world
            .resource::<RenderRuntimeResource<DeferredInstance>>()
            .runtime()
            .completed_value(),
        submission
    );
    let lifecycle = world.resource::<NativeFrameExecution>();
    assert_eq!(lifecycle.completed_value(), submission);
    assert_eq!(lifecycle.phase(), NativeFramePhase::Completed);
}
