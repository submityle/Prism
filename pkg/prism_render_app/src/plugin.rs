//! The [`PrismRenderPlugin`]: assembly of the render sub-app.

use std::sync::Mutex;

use prism_app::{App, Last, Plugin, SubApp};
use prism_ecs::schedule::{IntoSystemConfigs, Schedule, Schedules};
use prism_render_driver::{FrameSync, Instance};
use prism_render_runtime::RenderRuntime;

use crate::config::RenderAppConfig;
use crate::intent::RenderIntent;
use crate::label::{Render, RenderApp, RenderStartup};
use crate::pipeline_cache::PipelineCache;
use crate::queue::RenderQueues;
use crate::record::RenderRecord;
use crate::runtime_resource::RenderRuntimeResource;
use crate::scene::SceneRegistry;
use crate::sets::{RenderSystems, configure_render_sets};
use crate::stats::RenderFrameStats;
use crate::startup::ensure_render_startup;
use crate::systems::{clear_queues, drive_render_schedule, execute_frame, extract_render_record};
use crate::target::RenderTarget;
use crate::texture_cache::TextureCache;
use crate::views::{RenderViews, build_view_family};

/// Wires the RHI render runtime into a [`prism_app::App`] as a dedicated render
/// sub-app.
///
/// Adding the plugin:
///
/// 1. brings up a [`RenderRuntime`] on the given [`Instance`] backend and stores
///    it in the render sub-app's world (the GPU state never enters the main
///    world);
/// 2. creates the offscreen [`RenderTarget`] and the record/stats resources;
/// 3. registers a [`Render`] schedule holding the frame system, driven each
///    frame from the render sub-app's `Last` phase;
/// 4. installs the one-way extract that copies the main world's [`RenderIntent`]
///    into the render world.
///
/// It also registers the [`RenderStartup`] schedule, which runs once before the
/// first frame so a plugin can register GPU resources (which must be built
/// against the device) from anywhere in the app.
///
/// The plugin is generic over the backend, so a real app passes the shipped
/// wgpu backend and a test passes the headless null backend; neither the plugin
/// nor any of its systems names a graphics API.
///
/// # Panics
///
/// [`Plugin::build`] panics if the backend cannot bring up a device (no usable
/// adapter, or the device request is rejected). Bring-up failure is fatal at
/// startup by design: a render plugin with no GPU has nothing to do, and
/// failing loudly beats a silently black window.
pub struct PrismRenderPlugin<I> {
    /// The backend instance, consumed by [`Plugin::build`].
    ///
    /// Held behind a [`Mutex`] because [`Plugin::build`] takes `&self` while the
    /// runtime must own the instance; a backend instance is not required to be
    /// `Clone` (the wgpu one is not), so the instance is moved out rather than
    /// cloned. `Plugin::is_unique` guarantees `build` runs at most once.
    instance: Mutex<Option<I>>,
    /// The render sub-app's startup configuration.
    config: RenderAppConfig,
}

impl<I: Instance> PrismRenderPlugin<I> {
    /// Creates the plugin, bringing up `instance` when the plugin is built.
    #[must_use]
    pub fn new(instance: I) -> Self {
        Self::with_config(instance, RenderAppConfig::default())
    }

    /// Creates the plugin with an explicit [`RenderAppConfig`].
    #[must_use]
    pub fn with_config(instance: I, config: RenderAppConfig) -> Self {
        Self {
            instance: Mutex::new(Some(instance)),
            config,
        }
    }

    /// The configuration this plugin will build the render sub-app with.
    #[must_use]
    pub fn config(&self) -> RenderAppConfig {
        self.config
    }

    /// Moves the backend instance out of the plugin, for [`Plugin::build`].
    fn take_instance(&self) -> I {
        self.instance
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
            .unwrap_or_else(|| {
                panic!(
                    "PrismRenderPlugin::build ran twice: the backend instance was already consumed"
                )
            })
    }
}

impl<I> Plugin for PrismRenderPlugin<I>
where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    fn build(&self, app: &mut App) {
        let config = self.config;
        let runtime = RenderRuntime::with_frames_in_flight(
            self.take_instance(),
            config.extent,
            config.max_frames_in_flight,
        )
        .unwrap_or_else(|error| {
            panic!("PrismRenderPlugin: render backend bring-up failed: {error}")
        });

        // The GPU state the render world owns for the app's lifetime.
        let target = RenderTarget::new(runtime.device(), config.extent);
        let mut render = SubApp::new();
        render
            .world
            .insert_resource(RenderRuntimeResource::new(runtime));
        render.world.insert_resource(target);
        render.world.insert_resource(RenderRecord::default());
        render.world.insert_resource(RenderFrameStats::default());
        render.world.insert_resource(RenderViews::new());
        render.world.insert_resource(PipelineCache::new());
        render.world.insert_resource(TextureCache::new());
        render.world.insert_resource(RenderQueues::new());
        render
            .world
            .insert_resource(crate::draw::canonical::registry());
        render.world.insert_resource(SceneRegistry::new());

        // The render schedule, plus the `Last` phase that runs it each frame.
        {
            let schedules = render.world.resource_mut::<Schedules>();
            schedules.insert(Render, Schedule::new());
            schedules.insert(RenderStartup, Schedule::new());
            schedules.insert(Last, Schedule::new());
        }
        {
            let schedule = render
                .world
                .resource_mut::<Schedules>()
                .get_mut(Render)
                .expect("the Render schedule was just inserted");
            // Order the render stages/phases first, then hang the frame system
            // off the record stage so later renderer plugins slot in by set.
            configure_render_sets(schedule);
            // Materialize the view family before the record stage consumes it.
            schedule.add_systems(build_view_family.in_set(RenderSystems::CreateViews));
            schedule.add_systems(execute_frame::<I>.in_set(RenderSystems::Phase));
            schedule.add_systems(clear_queues.in_set(RenderSystems::Cleanup));
        }
        render
            .world
            .resource_mut::<Schedules>()
            .get_mut(Last)
            .expect("the Last schedule was just inserted")
            .add_systems(drive_render_schedule);

        app.insert_sub_app(RenderApp, render);

        // The main world's authoritative render intent, and the one-way feed.
        app.insert_resource(RenderIntent::from_extent(config.extent));
        app.set_extract(RenderApp, extract_render_record);
    }

    fn finish(&self, app: &mut App) {
        // Every plugin has registered its `RenderStartup` systems by now, so the
        // render world's GPU resources can be built. The schedule runs lazily on
        // the first frame too, so a test that only calls `update` still gets it;
        // running here as well means an app that inspects its render world before
        // the first frame sees the resources.
        if let Some(render) = app.sub_app_mut(RenderApp) {
            ensure_render_startup(&mut render.world);
        }
    }

    fn shutdown(&self, app: &mut App) {
        let Some(render) = app.sub_app_mut(RenderApp) else {
            return;
        };
        // Drain the pipeline, then release the frame target. The device itself
        // is dropped with the runtime when the app's worlds are torn down.
        if let Some(runtime) = render.world.get_resource::<RenderRuntimeResource<I>>() {
            let _ = runtime.runtime().wait_idle();
        }
        if let (Some(runtime), Some(target)) = (
            render.world.get_resource::<RenderRuntimeResource<I>>(),
            render.world.get_resource::<RenderTarget>(),
        ) {
            target.destroy(runtime.runtime().device());
        }
    }

    fn needs_rendering(&self) -> bool {
        true
    }
}
