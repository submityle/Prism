//! The [`PrismRenderPlugin`]: assembly of the render sub-app.

use std::path::PathBuf;
use std::sync::Mutex;

use prism_app::{App, First, Last, Plugin, SubApp};
use prism_ecs::schedule::{IntoSystemConfigs, Schedule, Schedules};
use prism_ecs::world::World;
use prism_render_driver::{FrameSync, Instance};
use prism_render_runtime::RenderRuntime;

use crate::config::RenderAppConfig;
use crate::device_loss::{DeviceLossSignal, DeviceRecoveryStatus};
use crate::intent::RenderIntent;
use crate::label::{Render, RenderApp, RenderStartup};
use crate::material::{
    sync_material_buffers, MaterialParameterBuffers, MaterialUploadStats, RenderMaterials,
};
use crate::mesh_pipeline::MeshPipelineSpecializer;
use crate::mesh_pipelines::MeshPipelines;
use crate::pipeline_cache::{
    pso_device_fingerprint, PipelineCachePersistenceConfig, PipelineCachePersistenceError,
    PipelineCachePersistenceStatus,
};
use crate::queue::RenderQueues;
use crate::queue_stage::fill_queues;
use crate::record::RenderRecord;
use crate::runtime_resource::RenderRuntimeResource;
use crate::scene::{SceneRegistry, VisibilityWorkList};
use crate::sets::{configure_render_sets, RenderSystems};
use crate::startup::ensure_render_startup;
use crate::stats::RenderFrameStats;
use crate::systems::{
    clear_queues, drive_render_schedule, execute_frame, extract_render_record,
    pump_pipeline_compiles,
};
use crate::target::RenderTarget;
use crate::texture_cache::TextureCache;
use crate::views::{build_view_family, RenderViews};

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
    /// Optional driver-local PSO cache persistence.
    pso_cache_persistence: Option<PipelineCachePersistenceConfig>,
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
            pso_cache_persistence: None,
        }
    }

    /// Enables driver-local PSO persistence at `path`.
    #[must_use]
    pub fn with_pso_cache_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.pso_cache_persistence = Some(PipelineCachePersistenceConfig::new(path));
        self
    }

    /// Enables driver-local PSO persistence with an explicit configuration.
    #[must_use]
    pub fn with_pso_cache_config(mut self, config: PipelineCachePersistenceConfig) -> Self {
        self.pso_cache_persistence = Some(config);
        self
    }

    /// Creates a plugin with render and PSO-persistence configuration together.
    #[must_use]
    pub fn with_config_and_pso_cache(
        instance: I,
        config: RenderAppConfig,
        pso_cache: PipelineCachePersistenceConfig,
    ) -> Self {
        Self::with_config(instance, config).with_pso_cache_config(pso_cache)
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

fn load_persisted_pso_cache<I>(world: &mut World)
where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    let Some(config) = world
        .get_resource::<PipelineCachePersistenceConfig>()
        .cloned()
    else {
        return;
    };
    let fingerprint = pso_device_fingerprint(
        world
            .resource::<RenderRuntimeResource<I>>()
            .runtime()
            .adapter_info(),
    );
    let result = world
        .resource_mut::<MeshPipelineSpecializer>()
        .load_persisted_cache_from_path(&config, &fingerprint);
    world.resource_mut::<PipelineCachePersistenceStatus>().load = Some(result);
}

fn record_pso_save_result(world: &mut World, result: Result<usize, PipelineCachePersistenceError>) {
    if world
        .get_resource::<PipelineCachePersistenceConfig>()
        .is_some()
    {
        world.resource_mut::<PipelineCachePersistenceStatus>().save = Some(result);
    }
}

fn pso_save_rejection(
    recovering: bool,
    idle: bool,
    device_lost: bool,
) -> Option<PipelineCachePersistenceError> {
    if recovering {
        Some(PipelineCachePersistenceError::Recovering)
    } else if device_lost {
        Some(PipelineCachePersistenceError::DeviceLost)
    } else if !idle {
        Some(PipelineCachePersistenceError::NotIdle)
    } else {
        None
    }
}

fn persist_pso_cache_after_idle<I>(world: &mut World, idle: bool)
where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    let Some(config) = world
        .get_resource::<PipelineCachePersistenceConfig>()
        .cloned()
    else {
        return;
    };
    let recovering = crate::recovery::is_recovering(world);
    let device_lost = world
        .resource::<RenderRuntimeResource<I>>()
        .runtime()
        .device_lost()
        .is_some();
    let result = if let Some(error) = pso_save_rejection(recovering, idle, device_lost) {
        Err(error)
    } else {
        let fingerprint = pso_device_fingerprint(
            world
                .resource::<RenderRuntimeResource<I>>()
                .runtime()
                .adapter_info(),
        );
        let mut specializer = world
            .remove_resource::<MeshPipelineSpecializer>()
            .expect("the mesh pipeline specializer is installed");
        let result = specializer.save_persisted_cache(
            world
                .resource::<RenderRuntimeResource<I>>()
                .runtime()
                .device(),
            &config,
            fingerprint,
        );
        world.insert_resource(specializer);
        result
    };
    record_pso_save_result(world, result);
}

/// Saves the configured PSO cache at an explicit host-controlled lifecycle point.
///
/// The cache remains installed and its worker pool remains usable after this
/// call. The final plugin shutdown still owns worker and native-pipeline release.
pub fn save_persisted_pso_cache<I>(world: &mut World)
where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    if crate::recovery::is_recovering(world) {
        persist_pso_cache_after_idle::<I>(world, false);
        return;
    }
    if world
        .resource::<RenderRuntimeResource<I>>()
        .runtime()
        .device_lost()
        .is_some()
    {
        record_pso_save_result(world, Err(PipelineCachePersistenceError::DeviceLost));
        return;
    }
    let idle = world
        .resource::<RenderRuntimeResource<I>>()
        .runtime()
        .wait_idle();
    persist_pso_cache_after_idle::<I>(world, idle);
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
        let mut runtime = RenderRuntime::with_frames_in_flight(
            self.take_instance(),
            config.extent,
            config.max_frames_in_flight,
        )
        .unwrap_or_else(|error| {
            panic!("PrismRenderPlugin: render backend bring-up failed: {error}")
        });
        let lighting_gpu =
            crate::lighting_gpu::LightingGpuState::new(&mut runtime).unwrap_or_else(|error| {
                panic!("PrismRenderPlugin: lighting GPU setup failed: {error}")
            });

        // Device-loss bridge (design §12/§24.3): one queue, cloned into both the
        // render world (which owns the GPU runtime) and the main world (which
        // owns the app lifecycle).
        let device_loss = DeviceLossSignal::new();

        // The GPU state the render world owns for the app's lifetime.
        let target = RenderTarget::new(runtime.device(), config.extent);
        // The spine's default mesh pipelines are built against the device, so
        // they are created before the runtime moves into the render world.
        let mesh_pipelines = MeshPipelines::spine(runtime.device(), crate::RENDER_TARGET_FORMAT);
        // The material ABI buffers are created against the device too, so they are
        // built here as well.
        let material_buffers = MaterialParameterBuffers::new(
            runtime.device(),
            config.material_capacity,
            config.material_parameter_words,
            config.material_texture_rows,
            prism_render_driver::BufferUsages::NONE,
        );
        // Cold material permutations compile on a worker that owns a shareable
        // device handle; the specializer's cache serves the full-lobe fallback
        // until the exact PSO arrives.
        let pipeline_specializer = MeshPipelineSpecializer::with_worker(runtime.device_handle());
        let mut render = SubApp::new();
        render
            .world
            .insert_resource(RenderRuntimeResource::new(runtime));
        render.world.insert_resource(target);
        render.world.insert_resource(RenderRecord::default());
        render.world.insert_resource(RenderFrameStats::default());
        render
            .world
            .insert_resource(crate::ExtractedLightingInputs::default());
        let lighting_plan = crate::LightingPlanState::from_config(config);
        let lighting_debug = crate::lighting::LightingDebugViewState::from_plan(&lighting_plan);
        render.world.insert_resource(lighting_plan);
        render.world.insert_resource(lighting_debug);
        render.world.insert_resource(lighting_gpu);
        render.world.insert_resource(RenderViews::new());
        render.world.insert_resource(TextureCache::new());
        render.world.insert_resource(RenderQueues::new());
        render
            .world
            .insert_resource(crate::draw::canonical::registry());
        render.world.insert_resource(pipeline_specializer);
        render
            .world
            .insert_resource(PipelineCachePersistenceStatus::default());
        if let Some(pso_cache) = self.pso_cache_persistence.clone() {
            render.world.insert_resource(pso_cache);
        }
        render.world.insert_resource(mesh_pipelines);
        render.world.insert_resource(SceneRegistry::new());
        render.world.insert_resource(VisibilityWorkList::new());
        // The material ABI: the render world's registry plus the header/parameter
        // buffers built from it (sized for the scene's material budget).
        render
            .world
            .insert_resource(RenderMaterials::new(config.material_capacity));
        render.world.insert_resource(material_buffers);
        render.world.insert_resource(MaterialUploadStats::default());
        render
            .world
            .insert_resource(crate::RenderTransforms::default());
        render
            .world
            .insert_resource(crate::ExtractedTransforms::default());
        render.world.insert_resource(device_loss.clone());

        // The render schedule, plus the `Last` phase that runs it each frame.
        {
            let schedules = render.world.resource_mut::<Schedules>();
            schedules.insert(Render, Schedule::new());
            schedules.insert(RenderStartup, Schedule::new());
            schedules.insert(Last, Schedule::new());
        }
        if self.pso_cache_persistence.is_some() {
            render
                .world
                .resource_mut::<Schedules>()
                .get_mut(RenderStartup)
                .expect("the startup schedule was just inserted")
                .add_systems(load_persisted_pso_cache::<I>);
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
            schedule.add_systems(
                crate::transforms::prepare_transform_resources::<I>.in_set(RenderSystems::Extract),
            );
            // Materialize the view family before the record stage consumes it.
            schedule.add_systems(build_view_family.in_set(RenderSystems::CreateViews));
            schedule.add_systems(
                (
                    crate::lighting::compile_lighting_plan,
                    crate::lighting::sync_lighting_debug_view,
                )
                    .chain()
                    .in_set(RenderSystems::PrepareViews),
            );
            // The visible set becomes this frame's queue before anything records.
            // Materials register/retire → the ABI buffers, before anything binds them.
            schedule
                .add_systems(sync_material_buffers::<I>.in_set(RenderSystems::PrepareResources));
            schedule
                .add_systems(pump_pipeline_compiles::<I>.in_set(RenderSystems::PrepareResources));
            schedule.add_systems(fill_queues::<I>.in_set(RenderSystems::Queue));
            schedule.add_systems(execute_frame::<I>.in_set(RenderSystems::Phase));
            schedule.add_systems(clear_queues.in_set(RenderSystems::Cleanup));
        }
        render
            .world
            .resource_mut::<Schedules>()
            .get_mut(Last)
            .expect("the Last schedule was just inserted")
            .add_systems(drive_render_schedule::<I>);

        app.insert_sub_app(RenderApp, render);

        // The main-world end of the bridge: each frame, drain the signal and
        // emit `DeviceLost`, which `drive_app_lifecycle` folds into
        // `AppLifecycle::Recovering` (pausing simulation for the rebuild).
        app.insert_resource(device_loss);
        app.insert_resource(DeviceRecoveryStatus::default());
        app.add_event::<prism_app::lifecycle::DeviceLost>();
        app.add_event::<prism_app::lifecycle::DeviceRestored>();
        app.add_systems(First, crate::device_loss::forward_device_recovery);
        app.add_lifecycle_events();

        // The main world's authoritative render intent, and the one-way feeds.
        //
        // Both are *appended*: the render world's extract is a chain (the frame
        // record, then one step per render-asset family), and replacing it here
        // would drop a family registered by another plugin.
        app.insert_resource(RenderIntent::from_extent(config.extent));
        app.insert_resource(crate::LightingSceneInputs {
            stats: config.lighting_scene_stats,
            journal: prism_render_architecture::lighting::contract::ChangeJournal::new(),
            directionals: Vec::new(),
            punctuals: Vec::new(),
            gi_candidates: Vec::new(),
            surfaces: vec![crate::lighting::LightingSurfaceInput::default()],
        });
        app.add_extract(RenderApp, extract_render_record);
        app.add_extract(RenderApp, crate::lighting::extract_lighting_inputs);
        app.init_resource::<crate::PublishedTransforms>();
        app.add_systems(Last, crate::transforms::publish_transforms);
        app.add_extract(RenderApp, crate::transforms::extract_transforms);
        crate::ecs_columns::install_gpu_column_extraction::<I>(app);
        // The spine's frame draws meshes, so the plugin owns that family: its
        // extract, its prepared table, and its prepare stage. Registering it
        // again elsewhere is a no-op.
        crate::render_asset::register_render_assets::<I, crate::RenderMesh>(app);
        // Images reach the GPU the same way: the spine owns the family so a
        // material can bind a texture without the app wiring a second pipeline.
        crate::render_asset::register_render_assets::<I, crate::GpuImage>(app);
        crate::presentation::install::<I>(app);
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
        crate::presentation::shutdown::<I>(&mut render.world);
        // A partially rebuilt target may still name the previous generation.
        // Never wait on or destroy resources through a lost/replaced device.
        let recovering = crate::recovery::is_recovering(&render.world);
        let device_lost = render
            .world
            .get_resource::<RenderRuntimeResource<I>>()
            .is_some_and(|runtime| runtime.runtime().device_lost().is_some());
        let idle = if recovering || device_lost {
            false
        } else {
            // Drain the pipeline before releasing resources. The device itself
            // is dropped with the runtime when the app's worlds are torn down.
            render
                .world
                .get_resource::<RenderRuntimeResource<I>>()
                .is_some_and(|runtime| runtime.runtime().wait_idle())
        };
        persist_pso_cache_after_idle::<I>(&mut render.world, idle);
        if render
            .world
            .get_resource::<RenderRuntimeResource<I>>()
            .is_some()
        {
            let mut specializer = render
                .world
                .remove_resource::<MeshPipelineSpecializer>()
                .expect("the mesh pipeline specializer is installed");
            if idle && !recovering && !device_lost {
                let runtime = render.world.resource::<RenderRuntimeResource<I>>();
                specializer.shutdown(runtime.runtime().device());
            } else {
                specializer.shutdown_without_device();
            }
            render.world.insert_resource(specializer);
        }
        if idle && !recovering && !device_lost {
            if let (Some(runtime), Some(target)) = (
                render.world.get_resource::<RenderRuntimeResource<I>>(),
                render.world.get_resource::<RenderTarget>(),
            ) {
                target.destroy(runtime.runtime().device());
            }
        }
    }

    fn needs_rendering(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pso_save_rejection_prioritizes_recovery_then_loss_then_idle() {
        assert_eq!(
            pso_save_rejection(true, true, true),
            Some(PipelineCachePersistenceError::Recovering)
        );
        assert_eq!(
            pso_save_rejection(false, true, true),
            Some(PipelineCachePersistenceError::DeviceLost)
        );
        assert_eq!(
            pso_save_rejection(false, false, false),
            Some(PipelineCachePersistenceError::NotIdle)
        );
        assert_eq!(pso_save_rejection(false, true, false), None);
    }
}
