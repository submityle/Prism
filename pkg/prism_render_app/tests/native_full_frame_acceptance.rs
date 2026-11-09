//! Native Prism full-frame acceptance.
//!
//! This is deliberately an application-level test: it drives `App::update()`
//! and observes the production render sub-app rather than inspecting source
//! text or calling individual graph helpers.  The deferred queue makes the
//! completion boundary visible so the second frame must fail closed until the
//! first submission is explicitly completed.
#![forbid(unsafe_code)]

use std::sync::{Arc, Mutex, Weak};

use prism_app::App;
use prism_ecs::world::World;
use prism_math::{Frustum, Mat4, Plane, Vec3};
use prism_render_app::prelude::{
    OffscreenFrameOutput, PrismRenderPlugin, RenderApp, RenderAppConfig, RenderFrameStats,
    SurfaceViewQueues, TextureTargetRegistry,
};
use prism_render_app::visibility::{PrismCamera, PrismFrustum, PrismVisibilityPlugin};
use prism_render_app::{
    ExtractedSurfaceViews, ExtractedVisibilityViews, RenderRuntimeResource, TaaHistoryCache,
    TextureTargetError,
};
use prism_render_driver as rhi;
use prism_render_driver::{FrameSync, Instance, RenderQueue};
use prism_render_driver_null::{NullAdapter, NullDevice, NullInstance, NullQueue, NullSurface};
use prism_transform::GlobalTransform;

const EXTENT: rhi::Extent3d = rhi::Extent3d::new_2d(8, 8);

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

impl Instance for DeferredInstance {
    type Adapter = NullAdapter;
    type Device = NullDevice;
    type Queue = DeferredQueue;
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
        let (device, _) = NullInstance::new().create_device(adapter, descriptor)?;
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

fn acceptance_app() -> (App, Arc<DeferredCompletion>) {
    let completion = Arc::new(DeferredCompletion::default());
    let mut app = App::new();
    app.add_plugins(PrismRenderPlugin::with_config(
        DeferredInstance(completion.clone()),
        RenderAppConfig::new(EXTENT),
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

#[test]
fn native_app_update_drives_full_frame_and_fail_closed_recovery() {
    let (mut app, completion) = acceptance_app();

    // First update drives the complete production path:
    // extract -> create/prepare -> queue -> graph record -> submit.
    app.update();
    let world = render_world(&app);
    let stats = world.resource::<RenderFrameStats>();
    assert_eq!(
        world
            .resource::<prism_render_app::RenderRecord>()
            .frame_index,
        1
    );
    assert_eq!(stats.frames_submitted, 1);
    assert_eq!(stats.frames_failed, 0);
    assert_eq!(stats.frames_skipped, 0);
    assert!(stats.passes_declared > 0, "the graph declared no passes");
    assert_eq!(stats.draws_queued, 0, "empty camera fixture has no draws");
    assert_eq!(stats.draws_recorded, 0);
    assert_eq!(world.resource::<ExtractedSurfaceViews>().len(), 1);
    assert_eq!(world.resource::<ExtractedVisibilityViews>().len(), 1);
    assert!(stats.lighting_plan_compiles > 0);

    let output = world
        .resource::<OffscreenFrameOutput>()
        .latest()
        .expect("submitted offscreen output");
    assert_eq!(output.frame_index(), 1);
    assert_ne!(output.submission(), 0);
    let runtime = world
        .resource::<RenderRuntimeResource<DeferredInstance>>()
        .runtime();
    assert_eq!(runtime.completed_value(), 0);
    assert_eq!(runtime.frames_in_flight(), 1);

    let queues = world.resource::<SurfaceViewQueues>();
    assert!(queues.is_ready(), "native camera queues were not published");
    assert_eq!(queues.frame_index, 1);
    assert_eq!(queues.len(), 1);
    assert!(world.resource::<TaaHistoryCache>().len() >= 1);
    assert_eq!(world.resource::<TaaHistoryCache>().retired_len(), 0);

    // The queue has not completed.  The next real App::update must reject the
    // old bindings, clear the camera queue snapshot, and leave the old output
    // as the last known submitted frame rather than republishing it.
    app.update();
    let world = render_world(&app);
    let stats = world.resource::<RenderFrameStats>();
    assert_eq!(stats.frames_submitted, 1);
    assert_eq!(stats.frames_failed, 1);
    assert_eq!(stats.frames_recorded(), 2);
    assert_eq!(
        world
            .resource::<prism_render_app::RenderRecord>()
            .frame_index,
        2
    );
    let queues = world.resource::<SurfaceViewQueues>();
    assert!(!queues.is_ready());
    assert!(queues.is_empty());
    assert_eq!(queues.frame_index, 2);
    assert!(queues.error().is_some());
    assert_eq!(
        world
            .resource::<OffscreenFrameOutput>()
            .latest()
            .expect("failed frame must not erase the last successful output")
            .frame_index(),
        1
    );

    // Completion is explicit, not inferred from submit.  Once it is observed,
    // the next update must rebuild the camera queue and submit a new frame.
    completion.complete_submitted();
    let world = render_world(&app);
    assert_eq!(
        world
            .resource::<RenderRuntimeResource<DeferredInstance>>()
            .runtime()
            .completed_value(),
        1
    );
    app.update();
    let world = render_world(&app);
    let stats = world.resource::<RenderFrameStats>();
    assert_eq!(stats.frames_submitted, 2);
    assert_eq!(stats.frames_failed, 1);
    assert!(world.resource::<SurfaceViewQueues>().is_ready());
    assert_eq!(world.resource::<SurfaceViewQueues>().frame_index, 3);
    let output = world
        .resource::<OffscreenFrameOutput>()
        .latest()
        .expect("recovered frame output");
    assert_eq!(output.frame_index(), 3);
    assert_eq!(output.submission(), 2);
    assert!(world.resource::<TaaHistoryCache>().len() >= 1);

    // A target generation is sealed against the actual submission callback.
    // It cannot be unregistered before completion, and is reclaimed only after
    // the callback has run. Re-registration then advances the logical handle.
    let first_target = {
        let world = &mut app.sub_app_mut(RenderApp).expect("render sub-app").world;
        let mut registry = world
            .remove_resource::<TextureTargetRegistry<NullDevice>>()
            .expect("target registry");
        let handle = {
            let runtime = world
                .resource::<RenderRuntimeResource<DeferredInstance>>()
                .runtime();
            registry
                .register(runtime, 41, EXTENT)
                .expect("register target generation")
        };
        let sealed = {
            let runtime = world
                .resource::<RenderRuntimeResource<DeferredInstance>>()
                .runtime();
            registry.seal(runtime, handle)
        };
        assert!(sealed);
        world.insert_resource(registry);
        handle
    };
    {
        let world = &mut app.sub_app_mut(RenderApp).expect("render sub-app").world;
        let mut registry = world
            .remove_resource::<TextureTargetRegistry<NullDevice>>()
            .expect("target registry");
        let runtime = world
            .resource::<RenderRuntimeResource<DeferredInstance>>()
            .runtime();
        assert_eq!(
            registry.unregister(runtime, first_target),
            Err(TextureTargetError::CompletionUnconfirmed)
        );
        assert_eq!(
            registry.current_generation(41),
            Some(first_target.generation)
        );
        world.insert_resource(registry);
    }

    completion.complete_submitted();
    let second_target = {
        let world = &mut app.sub_app_mut(RenderApp).expect("render sub-app").world;
        let mut registry = world
            .remove_resource::<TextureTargetRegistry<NullDevice>>()
            .expect("target registry");
        let second = {
            let runtime = world
                .resource::<RenderRuntimeResource<DeferredInstance>>()
                .runtime();
            assert!(registry
                .unregister(runtime, first_target)
                .expect("retire target"));
            assert_eq!(registry.collect_completed(), 1);
            registry
                .register(runtime, 41, rhi::Extent3d::new_2d(10, 6))
                .expect("register next target generation")
        };
        assert_eq!(second.generation, first_target.generation + 1);
        world.insert_resource(registry);
        second
    };
    assert_eq!(second_target.id, first_target.id);
    assert_ne!(second_target.route_id(), first_target.route_id());

    completion.complete_submitted();
}
