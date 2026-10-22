//! The [`PrismRenderPlugin`]: assembly of the render sub-app.

use std::path::PathBuf;
use std::sync::Mutex;

use prism_app::{App, First, Last, Plugin, SubApp};
use prism_ecs::prelude::{Res, ResMut};
use prism_ecs::schedule::{IntoSystemConfigs, Schedule, Schedules};
use prism_ecs::world::World;
use prism_render_driver::{FrameSync, Instance};
use prism_render_runtime::RenderRuntime;

use crate::config::RenderAppConfig;
use crate::device_loss::{DeviceLossSignal, DeviceRecoveryStatus};
use crate::intent::RenderIntent;
use crate::label::{Render, RenderApp, RenderStartup};
use crate::material::virtual_texture::{
    NativeMaterialVt, NativeVtScheduler, NativeVtSources, NativeVtUploadDriver,
};
use crate::material::{
    sync_material_buffers, MaterialParameterBuffers, MaterialUploadStats, RenderMaterials,
};
use crate::mesh_pipeline::MeshPipelineSpecializer;
use crate::mesh_pipelines::MeshPipelines;
use crate::particle::init_particle_gpu_state;
use crate::pipeline_cache::{
    pso_device_fingerprint, PipelineCachePersistenceConfig, PipelineCachePersistenceError,
    PipelineCachePersistenceStatus,
};
use crate::queue::{CanonicalMaterialBindings, RenderQueues};
use crate::queue_stage::{fill_queues, ClstDrawSubmissions};
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

/// Observe loss before recovery's first-frame early return, not just when it
/// eventually replaces particle state or the Ember registry. Close both build
/// gates and invalidate published particle leases before either can be dropped.
/// Poll the runtime's cached notification; workers must never consume the
/// backend's loss notification themselves.
pub(crate) fn abandon_lost_ember_device<I>(world: &mut World)
where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    let lost = world
        .get_resource::<RenderRuntimeResource<I>>()
        .is_some_and(|runtime| runtime.runtime().device_lost().is_some());
    if lost {
        if let Some(builds) = world.get_resource::<crate::particle::ParticleGpuBuildQueue>() {
            builds.abandon_device();
        }
        if let Some(state) = world.get_resource::<crate::particle::ParticleGpuState>() {
            state.abandon();
        }
        if let Some(registry) = world.get_resource::<crate::particle::EmberGpuAssetRegistry>() {
            registry.abandon_materialization_device();
        }
    }
}

/// Standalone legacy systems may omit VT entirely. The production plugin always
/// installs its config and all three VT resources: partial removal must never
/// silently enable uploads or queue stale virtual rows.
pub(crate) fn native_vt_resources_ready(
    required: bool,
    scheduler: Option<&NativeVtScheduler>,
    has_owner: bool,
    has_sources: bool,
) -> bool {
    match scheduler {
        Some(scheduler) => has_owner && has_sources && scheduler.is_ready(),
        None => !required && !has_owner && !has_sources,
    }
}

/// Publish/revoke VT rows before material-buffer sync. Do not retain a binding:
/// consumers must acquire a creator-device lease for their actual submission.
pub(crate) fn prepare_native_vt<I>(
    runtime: Res<RenderRuntimeResource<I>>,
    mut materials: ResMut<RenderMaterials>,
    mut stats: ResMut<MaterialUploadStats>,
    mut scheduler: Option<ResMut<NativeVtScheduler>>,
    mut owner: Option<ResMut<NativeMaterialVt<I::Device>>>,
    sources: Option<Res<NativeVtSources>>,
    mut uploader: Option<ResMut<NativeVtUploadDriver<I>>>,
) where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    stats.native_vt_ready = false;
    stats.native_vt_errors.clear();
    stats.native_vt_missing_resources = scheduler.is_none() || owner.is_none() || sources.is_none();
    let (Some(scheduler), Some(owner), Some(sources)) =
        (scheduler.as_mut(), owner.as_mut(), sources.as_ref())
    else {
        if let Some(scheduler) = scheduler.as_mut() {
            scheduler.invalidate();
        }
        if let Some(owner) = owner.as_mut() {
            owner.cancel();
        }
        return;
    };
    // The not-ready Queue gate cannot reach its usual completion barrier.
    // In particular wgpu callbacks need host progress before the first atlas
    // can publish. Only drain an existing upload; quiet frames do not wait.
    // A failed wait is not completion: publication still checks its callback.
    if owner.has_pending_upload() && runtime.runtime().device_lost().is_none() {
        let _ = runtime.runtime().wait_idle();
    }
    let report = match uploader.as_mut() {
        Some(uploader) => scheduler.prepare_with_uploader(
            runtime.runtime(),
            owner,
            sources,
            &mut materials,
            |runtime, owner, sources, heap, residency| {
                uploader.stage(runtime, owner, sources, heap, residency)
            },
        ),
        None => scheduler.prepare(runtime.runtime(), owner, sources, &mut materials),
    };
    stats.native_vt_ready = report.ready;
    stats.native_vt_errors = report.errors;
}

fn drive_render_schedule_with_ember_loss_guard<I>(world: &mut World)
where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    abandon_lost_ember_device::<I>(world);
    drive_render_schedule::<I>(world);
    // Also cover a loss first reported by startup, preparation or submission.
    // Recovery alone is not loss: a healthy replacement registry must stay open.
    abandon_lost_ember_device::<I>(world);
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
        app.init_resource::<crate::particle::ParticleViewBudget>();
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
        // The production surface-encode provider owns a device-rebuild recipe,
        // so its shader and pipeline layouts are recreated by the recovery
        // boundary when this device is replaced.
        let mesh_pipelines = MeshPipelines::surface_encode(runtime.device());
        // The material ABI buffers are created against the device too, so they are
        // built here as well.
        let material_buffers = MaterialParameterBuffers::new_owned(
            runtime.device_handle(),
            config.material_capacity,
            config.material_parameter_words,
            config.material_texture_rows,
            prism_render_driver::BufferUsages::NONE,
        );
        let native_vt = NativeMaterialVt::new(&runtime, config.material_virtual_texture)
            .unwrap_or_else(|error| {
                panic!("PrismRenderPlugin: native material VT setup failed: {error}")
            });
        // Cold material permutations compile on a worker that owns a shareable
        // device handle; the specializer's cache serves the full-lobe fallback
        // until the exact PSO arrives.
        let pipeline_specializer = MeshPipelineSpecializer::with_worker(runtime.device_handle());
        let mut render = SubApp::new();
        render
            .world
            .insert_resource(RenderRuntimeResource::new(runtime));
        render.world.insert_resource(config);
        render.world.insert_resource(target);
        // Both output paths consume the configuration copied from RenderAppConfig.
        render.world.insert_resource(
            crate::display_transform_gpu::OffscreenDisplayTransform::new(config.display_transform),
        );
        render
            .world
            .insert_resource(crate::display_transform_gpu::OffscreenFrameOutput::default());
        render.world.insert_resource(RenderRecord::default());
        render.world.insert_resource(RenderFrameStats::default());
        render
            .world
            .insert_resource(crate::ExtractedLightingInputs::default());
        render
            .world
            .insert_resource(crate::ExtractedVisibilityViews::default());
        render
            .world
            .insert_resource(crate::lighting_trace_gpu::LightingTraceState::default());
        let lighting_plan = crate::LightingPlanState::from_config(config);
        let lighting_debug = crate::lighting::LightingDebugViewState::from_plan(&lighting_plan);
        render.world.insert_resource(lighting_plan);
        render.world.insert_resource(lighting_debug);
        render.world.insert_resource(lighting_gpu);
        render.world.insert_resource(RenderViews::new());
        render
            .world
            .insert_resource(crate::surface_views::ExtractedSurfaceViews::default());
        render
            .world
            .init_resource::<crate::particle::ExtractedParticleViews>();
        render.world.insert_resource(
            *app.world()
                .resource::<crate::particle::ParticleViewBudget>(),
        );
        render.world.insert_resource(TextureCache::new());
        render.world.insert_resource(RenderQueues::new());
        render.world.init_resource::<crate::SurfaceViewQueues>();
        render
            .world
            .init_resource::<crate::surface_view_targets::SurfaceViewTargets<I::Device>>();
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
        render
            .world
            .insert_resource(crate::NativeSceneGpuState::new());
        render
            .world
            .insert_resource(crate::ExtractedNativeScene::default());
        render
            .world
            .insert_resource(crate::NativeGeometryGpuState::new());
        render
            .world
            .insert_resource(crate::ExtractedNativeGeometry::default());
        // The material ABI: the render world's registry plus the header/parameter
        // buffers built from it (sized for the scene's material budget).
        render
            .world
            .insert_resource(RenderMaterials::new(config.material_capacity));
        render.world.insert_resource(material_buffers);
        render
            .world
            .insert_resource(crate::material::NativeExecutionTableBuffers::new());
        render
            .world
            .insert_resource(CanonicalMaterialBindings::new(config.material_capacity));
        render.world.insert_resource(MaterialUploadStats::default());
        render.world.insert_resource(native_vt);
        render.world.insert_resource(NativeVtScheduler::default());
        render.world.insert_resource(NativeVtSources::default());
        render
            .world
            .insert_resource(crate::material::NativeVtSourceStatus::default());
        render
            .world
            .insert_resource(crate::RenderTransforms::default());
        render
            .world
            .insert_resource(crate::ExtractedTransforms::default());
        render
            .world
            .insert_resource(prism_particle_app::ExtractedParticleSnapshot::default());
        render
            .world
            .insert_resource(crate::particle::ExtractedEmberRenderAssets::default());
        render
            .world
            .insert_resource(crate::particle::ExtractedEmberExternalGpuResources::default());
        render
            .world
            .insert_resource(crate::particle::ParticleGpuBuildQueue::default());
        render
            .world
            .insert_resource(crate::particle::ParticleUploadStatus::default());
        render
            .world
            .insert_resource(crate::particle::ParticleGpuSimulationInputs::default());
        render
            .world
            .insert_resource(crate::particle::PreparedParticleFabricFrame::default());
        render
            .world
            .init_resource::<crate::ExtractedRenderFabricFrame>();
        render
            .world
            .insert_resource(crate::particle::EmberPrepareStatus::default());
        render
            .world
            .insert_resource(crate::particle::EmberGpuAssetRegistry::default());
        render
            .world
            .insert_resource(crate::frame::FrameGraphTextureImports::default());
        render.world.init_resource::<ClstDrawSubmissions>();
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
        render
            .world
            .resource_mut::<Schedules>()
            .get_mut(RenderStartup)
            .expect("the RenderStartup schedule was just inserted")
            .add_systems((
                init_particle_gpu_state::<I>,
                crate::startup::init_gpu_resource::<
                    crate::render_asset::VirtualGeometryGpuPublicationStore<I>,
                >,
            ));
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
            schedule
                .add_systems(crate::extract_native_geometry::<I>.in_set(RenderSystems::Extract));
            // Materialize the view family before the record stage consumes it.
            schedule.add_systems(build_view_family.in_set(RenderSystems::CreateViews));
            schedule.add_systems(
                (
                    crate::lighting::compile_lighting_plan::<I>,
                    crate::lighting::sync_lighting_debug_view,
                )
                    .chain()
                    .in_set(RenderSystems::PrepareViews),
            );
            // Queue validates the active ABI header against the extracted
            // registry revision. Publish material rows before that validation,
            // not in PrepareResources (which runs after Queue).
            schedule.add_systems(
                (prepare_native_vt::<I>, sync_material_buffers::<I>)
                    .chain()
                    .in_set(RenderSystems::PrepareAssets),
            );
            schedule.add_systems(
                crate::prepare_native_scene_gpu::<I>.in_set(RenderSystems::PrepareResources),
            );
            schedule.add_systems(
                crate::prepare_native_geometry_gpu::<I>.in_set(RenderSystems::PrepareResources),
            );
            schedule
                .add_systems(pump_pipeline_compiles::<I>.in_set(RenderSystems::PrepareResources));
            schedule.add_systems(
                (
                    crate::particle::prepare_particle_snapshot::<I>,
                    crate::fabric_frame::finalize_render_fabric_frame,
                )
                    .chain()
                    .in_set(RenderSystems::PrepareResources),
            );
            schedule.add_systems(
                crate::particle::prepare_ember_graphs::<I>.in_set(RenderSystems::PrepareResources),
            );
            schedule.add_systems(fill_queues::<I>.in_set(RenderSystems::Queue));
            schedule.add_systems(execute_frame::<I>.in_set(RenderSystems::Phase));
            schedule.add_systems(clear_queues.in_set(RenderSystems::Cleanup));
        }
        render
            .world
            .resource_mut::<Schedules>()
            .get_mut(Last)
            .expect("the Last schedule was just inserted")
            .add_systems(drive_render_schedule_with_ember_loss_guard::<I>);

        app.insert_sub_app(RenderApp, render);

        // Virtual-geometry publication is part of the default render spine.
        // The bridge is generation-bound and fail-closed; requiring an
        // application-specific opt-in here would leave the CLST queue with no
        // authoritative page-table lifecycle in the production plugin.
        crate::render_asset::register_virtual_geometry_publication_bridge::<I>(app);

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
        app.insert_resource(crate::LightingSceneInputs::with_stats(
            config.lighting_scene_stats,
        ));
        if app
            .world()
            .get_resource::<prism_particle_app::ParticleSimulationConfig>()
            .is_none()
        {
            let mut simulation = prism_particle_app::ParticleSimulationConfig::default();
            simulation.capacity = config.particle_capacity;
            app.insert_resource(simulation);
        }
        prism_particle_app::install_particle_simulation_inputs(app);
        app.add_extract(RenderApp, extract_render_record);
        app.add_extract(RenderApp, |main, secondary| {
            prism_particle_app::extract_particle_snapshot(main, secondary);
        });
        app.add_extract(RenderApp, crate::particle::extract_particle_fabric_frame);
        app.init_resource::<crate::particle::EmberRenderAssets>();
        app.init_resource::<crate::particle::EmberCompiledGraphAssets>();
        app.init_resource::<crate::particle::EmberExternalGpuResources>();
        app.init_resource::<crate::frame::FrameGraphTextureImports>();
        app.add_extract(RenderApp, crate::particle::extract_ember_render_assets);
        app.add_extract(
            RenderApp,
            crate::particle::extract_particle_simulation_inputs,
        );
        app.add_extract(
            RenderApp,
            crate::particle::extract_ember_external_gpu_resources,
        );
        app.add_extract(RenderApp, crate::frame::extract_frame_graph_texture_imports);
        app.add_extract(RenderApp, crate::lighting::extract_lighting_inputs);
        app.add_extract(RenderApp, crate::surface_views::extract_surface_views);
        app.add_extract(RenderApp, |main, render| {
            if let Some(budget) = main
                .get_resource::<crate::particle::ParticleViewBudget>()
                .copied()
            {
                render.insert_resource(budget);
            }
            crate::particle::extract_particle_views(main, render);
        });
        app.add_extract(RenderApp, crate::fabric_frame::extract_render_fabric_frame);
        app.add_extract(
            RenderApp,
            crate::visibility_bridge::extract_visibility_views,
        );
        app.add_extract(RenderApp, crate::extract_native_scene::<I>);
        app.init_resource::<crate::PublishedTransforms>();
        app.add_systems(Last, crate::transforms::publish_transforms);
        app.add_extract(RenderApp, crate::transforms::extract_transforms);
        crate::ecs_columns::install_gpu_column_extraction::<I>(app);
        // The forward-lit mesh frame owns this family: its extract, prepared
        // table, and prepare stage. Registering it again elsewhere is a no-op.
        crate::render_asset::register_render_assets::<I, crate::RenderMesh>(app);
        // Images reach the GPU through the same render-asset boundary; material
        // providers can bind them without installing a second mesh pipeline.
        crate::render_asset::register_render_assets::<I, crate::GpuImage>(app);
        app.add_extract_finalizer(RenderApp, crate::material::extract_native_vt_sources);
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
        abandon_lost_ember_device::<I>(&mut render.world);
        crate::presentation::shutdown::<I>(&mut render.world);
        // A partially rebuilt target may still name the previous generation.
        // Never wait on or destroy resources through a lost/replaced device.
        let recovering = crate::recovery::is_recovering(&render.world);
        let device_lost = render
            .world
            .get_resource::<RenderRuntimeResource<I>>()
            .is_some_and(|runtime| runtime.runtime().device_lost().is_some());
        if let Some(builds) = render
            .world
            .remove_resource::<crate::particle::ParticleGpuBuildQueue>()
        {
            // Disconnect the receiver even on healthy shutdown. A queued or
            // later unsent result owns its cleanup; only loss suppresses calls
            // to the old backend through the shared construction gate.
            if device_lost {
                builds.abandon_device();
            }
            drop(builds);
        }
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
        if !idle || recovering || device_lost {
            if let Some(owner) = render.world.get_resource::<NativeMaterialVt<I::Device>>() {
                owner.abandon_device();
            }
        }
        // Healthy shutdown must drain uploads before dropping their owner.
        // Pending callbacks/consumer leases retain their creator-device slice;
        // cancellation alone is not GPU completion.
        render.world.remove_resource::<NativeVtScheduler>();
        render.world.remove_resource::<NativeVtUploadDriver<I>>();
        render
            .world
            .remove_resource::<NativeMaterialVt<I::Device>>();
        render.world.remove_resource::<NativeVtSources>();
        if let Some(display) = render
            .world
            .get_resource::<crate::display_transform_gpu::OffscreenDisplayTransform>()
        {
            display.allow_destruction(idle && !recovering && !device_lost);
        }
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
            if let Some(mut bindings) = render.world.remove_resource::<CanonicalMaterialBindings>()
            {
                let runtime = render.world.resource::<RenderRuntimeResource<I>>();
                bindings.destroy(runtime.runtime().device());
            }
            if let Some(mut native_scene) =
                render.world.remove_resource::<crate::NativeSceneGpuState>()
            {
                let runtime = render.world.resource::<RenderRuntimeResource<I>>();
                native_scene.destroy(runtime.runtime().device());
            }
            if let Some(mut native_geometry) = render
                .world
                .remove_resource::<crate::NativeGeometryGpuState>()
            {
                let runtime = render.world.resource::<RenderRuntimeResource<I>>();
                native_geometry.destroy(runtime.runtime().device());
            }
            if let Some(ember) = render
                .world
                .remove_resource::<crate::particle::EmberGpuAssetRegistry>()
            {
                let runtime = render.world.resource::<RenderRuntimeResource<I>>();
                ember.destroy(runtime.runtime().device());
            }
            if let Some(particle) = render
                .world
                .remove_resource::<crate::particle::ParticleGpuState>()
            {
                particle.destroy();
            } else {
                render
                    .world
                    .remove_resource::<crate::particle::ParticleUploadStatus>();
            }
            render
                .world
                .remove_resource::<crate::particle::EmberPrepareStatus>();
            render
                .world
                .remove_resource::<crate::particle::PreparedParticleFabricFrame>();
            render
                .world
                .remove_resource::<crate::ExtractedRenderFabricFrame>();
            render
                .world
                .remove_resource::<crate::particle::ExtractedEmberRenderAssets>();
            render
                .world
                .remove_resource::<crate::particle::ExtractedEmberExternalGpuResources>();
            if let (Some(runtime), Some(target)) = (
                render.world.get_resource::<RenderRuntimeResource<I>>(),
                render.world.get_resource::<RenderTarget>(),
            ) {
                target.destroy(runtime.runtime().device());
            }
        } else {
            render.world.remove_resource::<CanonicalMaterialBindings>();
            render.world.remove_resource::<crate::NativeSceneGpuState>();
            render
                .world
                .remove_resource::<crate::NativeGeometryGpuState>();
            render
                .world
                .remove_resource::<crate::particle::EmberGpuAssetRegistry>();
            if let Some(state) = render
                .world
                .remove_resource::<crate::particle::ParticleGpuState>()
            {
                state.abandon();
                // Retire CPU/core ownership without destroying lost handles.
                state.destroy();
            }
            render
                .world
                .remove_resource::<crate::particle::ParticleUploadStatus>();
            render
                .world
                .remove_resource::<crate::particle::EmberPrepareStatus>();
            render
                .world
                .remove_resource::<crate::particle::PreparedParticleFabricFrame>();
            render
                .world
                .remove_resource::<crate::ExtractedRenderFabricFrame>();
            render
                .world
                .remove_resource::<crate::particle::ExtractedEmberRenderAssets>();
            render
                .world
                .remove_resource::<crate::particle::ExtractedEmberExternalGpuResources>();
        }
    }

    fn needs_rendering(&self) -> bool {
        true
    }
}

#[cfg(test)]
#[path = "plugin/native_vt_integration_tests.rs"]
mod native_vt_integration_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh_pipeline::{MeshPipelineKey, MeshPipelineProvider};
    use prism_render_architecture::shader_package::{ExecutionFamily, ModuleId, PermutationIndex};
    use prism_render_driver::{ColorTargetState, PrimitiveTopology, TextureFormat};
    use prism_render_driver_null::NullInstance;

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

    #[test]
    fn plugin_initializes_particle_views_and_default_budget() {
        let mut app = App::new();
        app.add_plugins(PrismRenderPlugin::new(NullInstance::new()));

        let expected = crate::ParticleViewBudget::default();
        assert_eq!(
            *app.world().resource::<crate::ParticleViewBudget>(),
            expected
        );
        let render = app.get_sub_app(RenderApp).expect("render sub-app");
        assert_eq!(
            *render.world.resource::<crate::ParticleViewBudget>(),
            expected
        );
        assert!(render
            .world
            .resource::<crate::ExtractedParticleViews>()
            .is_empty());
    }

    #[test]
    fn plugin_preserves_configured_particle_view_budget() {
        let mut app = App::new();
        let budget = crate::ParticleViewBudget {
            max_views: 3,
            max_retired_renderers: 5,
            max_work_items: 4096,
        };
        app.insert_resource(budget);
        app.add_plugins(PrismRenderPlugin::new(NullInstance::new()));

        assert_eq!(*app.world().resource::<crate::ParticleViewBudget>(), budget);
        let render = app.get_sub_app(RenderApp).expect("render sub-app");
        assert_eq!(
            *render.world.resource::<crate::ParticleViewBudget>(),
            budget
        );
    }

    #[test]
    fn healthy_shutdown_removes_particle_build_queue() {
        let mut app = App::new();
        app.add_plugins(PrismRenderPlugin::new(NullInstance::new()));
        assert!(app
            .get_sub_app(RenderApp)
            .unwrap()
            .world
            .get_resource::<crate::particle::ParticleGpuBuildQueue>()
            .is_some());

        PrismRenderPlugin::new(NullInstance::new()).shutdown(&mut app);

        assert!(app
            .get_sub_app(RenderApp)
            .unwrap()
            .world
            .get_resource::<crate::particle::ParticleGpuBuildQueue>()
            .is_none());
    }

    #[test]
    fn plugin_installs_surface_encode_as_the_default_mesh_provider() {
        let mut app = App::new();
        app.add_plugins(PrismRenderPlugin::new(NullInstance::new()));

        let render = app.get_sub_app(RenderApp).expect("render sub-app");
        let pipelines = render.world.resource::<MeshPipelines>();
        let descriptor = pipelines.descriptor(&MeshPipelineKey {
            execution_family: ExecutionFamily::FusedBuiltin,
            response_module_id: ModuleId::SPINE,
            permutation_index: PermutationIndex::ZERO,
            fallback_permutation: None,
            color_format: Some(TextureFormat::Rgba16Float),
            depth_format: Some(crate::RENDER_DEPTH_FORMAT),
            sample_count: 1,
            topology: PrimitiveTopology::TriangleList,
        });

        assert_eq!(descriptor.vertex.entry_point, "vs");
        assert!(descriptor.layout.is_some());
        assert_eq!(
            descriptor
                .fragment
                .as_ref()
                .expect("forward-lit fragment")
                .targets,
            vec![
                Some(ColorTargetState::opaque(
                    crate::target::POSITION_X_GBUFFER_FORMAT
                )),
                Some(ColorTargetState::opaque(crate::target::GBUFFER_FORMAT)),
                Some(ColorTargetState::opaque(
                    crate::target::ALBEDO_GBUFFER_FORMAT
                )),
                Some(ColorTargetState::opaque(
                    crate::target::SURFACE_ID_GBUFFER_FORMAT
                )),
            ]
        );
    }
}
