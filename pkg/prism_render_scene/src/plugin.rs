use crate::scene_lighting::{
    extract_canonical_lighting, CanonicalLightingDiagnostics, CanonicalLightingScene,
};
use bevy_app::{App, Plugin};
use bevy_asset::embedded_asset;
use bevy_ecs::resource::Resource;
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_ecs::system::SystemParam;
use bevy_render::view::ExtractedView;
use bevy_render::{
    init_gpu_resource,
    render_resource::PipelineCache,
    renderer::{RenderContext, RenderGraph, RenderGraphSystems, RenderQueue},
    renderer::{RenderDevice, ViewQuery},
    sync_world::SyncToRenderWorld,
    ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
};
use prism_particle_render::ParticleFabricCompletion;
use prism_render_architecture::gpu_scene::GpuCompletionValue;
use prism_render_architecture::hair::raster::HairSoftRasterAbi;
use prism_render_architecture::render_fabric::{PassKey, RenderDomain};
use prism_render_driver::Extent3d;

use crate::{
    buffers::{
        prepare_gpu_scene_bind_group, write_gpu_scene_buffers, GpuSceneBindGroup, GpuSceneBuffers,
    },
    compare::{compare_scene_mirror, GpuSceneParityDiagnostics},
    completion::{reclaim_completed_handles, GpuCompletionTracker},
    diagnostics::{GpuSceneDiagnostics, GpuSceneUploadSettings},
    extract::{
        apply_extracted_scene_changes, extract_scene_instances, retire_unused_geometry,
        PrismGpuSceneEntity,
    },
    fabric_runtime::{SceneFabricCoordinator, SceneFabricCoordinatorInput, SceneFabricRequest},
    geometry::{
        extract_shading_geometry, prepare_geometry_bind_group, sync_geometry_registry,
        sync_shading_geometry_registry, upload_geometry_buffers, upload_shading_geometry_buffers,
        GeometryBindGroup, RenderGeometryBuffers, RenderGeometryRegistry,
        RenderShadingGeometryBuffers, RenderShadingGeometryRegistry, ShadingGeometryStaging,
        VirtualGeometryRuntimeRegistry,
    },
    gpu::fabric::SceneFabric,
    scene::RenderGpuScene,
};

/// Installs the retained GPU Scene into Bevy's render sub-application.
pub struct PrismGpuScenePlugin;

/// Observable admission result for the single coordinator batch.
///
/// The semantic frame can contain Hair/Volume/Decal/Surface identities even
/// when their physical typed recorder has not been published. Keeping these
/// counters here makes that distinction explicit: semantic collection is not
/// execution evidence.
#[derive(Resource, Debug, Default)]
struct SceneFabricCoordinatorBatchStatus {
    frame: u64,
    surface_items: usize,
    particle_items: usize,
    hair_items: usize,
    volume_items: usize,
    decal_items: usize,
    particle_passes: usize,
    particle_resources: usize,
    unavailable_domains: Vec<&'static str>,
    last_error: Option<String>,
}

/// Particle completion owners retained until the same host queue submission
/// that contains the unified coordinator command buffer completes.
#[derive(Resource, Default)]
struct SceneFabricCoordinatorParticleCompletions(Vec<ParticleFabricCompletion>);

/// Temporarily removes the real prepared Water packet from the legacy Water
/// recorder's resource slot while the unified coordinator consumes it. The
/// packet is restored after the simulation phase so the Water surface raster
/// still sees the same resident buffers and bind groups.
#[derive(Resource, Default)]
struct SceneFabricCoordinatorWaterPacket(Option<crate::water::WaterGpuBodies>);

/// Decal publications and bind groups retained by the unified host until the
/// same queue submission that contains the coordinator command buffer retires.
#[derive(Default)]
struct SceneFabricCoordinatorDecalLeases {
    pending: Vec<crate::decal::DecalResolveBatchLease>,
    in_flight: Vec<(GpuCompletionValue, crate::decal::DecalResolveBatchLease)>,
}

impl PrismGpuScenePlugin {
    /// Component bundle that opts a mesh into the retained scene and Bevy's
    /// render-world synchronization without changing Bevy's required-component
    /// registrations.
    pub fn entity(config: PrismGpuSceneEntity) -> (PrismGpuSceneEntity, SyncToRenderWorld) {
        (config, SyncToRenderWorld::default())
    }
}

/// Controls whether opt-in entities use the retained GPU Scene or remain on
/// the legacy renderer-only path.
#[derive(Resource, Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum GpuSceneMode {
    /// Stops extraction/consumption and keeps the legacy renderer active.
    Disabled,
    /// Enables the GPU Scene; installed consumers replace their legacy phase.
    #[default]
    Enabled,
    /// Enables the same consumers while publishing CPU/GPU parity diagnostics.
    /// Pixel A/B rendering is intentionally owned by tooling, not duplicated
    /// into the production opaque phase.
    Compare,
}

impl Plugin for PrismGpuScenePlugin {
    fn build(&self, app: &mut App) {
        // Prism's canonical light scene is a main-world input.  The adapter
        // below copies it one way into the Bevy render world; no Bevy light
        // component or Bevy lighting registry is installed here.
        app.init_resource::<CanonicalLightingScene>();
        if !app.is_plugin_added::<crate::material::PrismMaterialPlugin>() {
            app.add_plugins(crate::material::PrismMaterialPlugin);
        }
        if !app.is_plugin_added::<crate::visibility::PrismVisibilityPlugin>() {
            app.add_plugins(crate::visibility::PrismVisibilityPlugin);
        }
        if !app.is_plugin_added::<crate::shading::PrismShadingPlugin>() {
            app.add_plugins(crate::shading::PrismShadingPlugin);
        }
        if !app.is_plugin_added::<crate::cloth::plugin::ClothPlugin>() {
            app.add_plugins(crate::cloth::plugin::ClothPlugin);
        }
        if !app.is_plugin_added::<crate::water::plugin::WaterPlugin>() {
            app.add_plugins(crate::water::plugin::WaterPlugin);
        }
        if !app.is_plugin_added::<crate::gpu::PrismGpuPlugin>() {
            app.add_plugins(crate::gpu::PrismGpuPlugin);
        }
        embedded_asset!(app, "shaders/gpu_scene.wesl");
        embedded_asset!(app, "shaders/decal_resolve.wesl");
        embedded_asset!(app, "shaders/hair_raster.wesl");
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .world_mut()
            .init_non_send::<SceneFabricCoordinatorDecalLeases>();
        render_app
            .world_mut()
            .init_non_send::<crate::hair_raster::HairRasterPipelineState>();
        render_app
            .world_mut()
            .init_non_send::<crate::hair_raster::HairRasterCoordinatorState>();

        render_app
            .init_resource::<GpuSceneMode>()
            .init_resource::<GpuSceneDiagnostics>()
            .init_resource::<GpuSceneParityDiagnostics>()
            .init_resource::<GpuSceneUploadSettings>()
            .init_resource::<CanonicalLightingDiagnostics>()
            .init_resource::<crate::fabric_frame::RenderFabricFrameClock>()
            .init_resource::<crate::fabric_frame::RenderFabricFrame>()
            .init_resource::<crate::fabric_frame::FabricDomainContributions>()
            .init_resource::<crate::fabric_frame::FabricDomainAggregationStatus>()
            .init_resource::<SceneFabricCoordinator>()
            .init_resource::<SceneFabricCoordinatorBatchStatus>()
            .init_resource::<SceneFabricCoordinatorParticleCompletions>()
            .init_resource::<SceneFabricCoordinatorWaterPacket>()
            .init_resource::<crate::particle::ParticleSceneFabricHost>()
            .init_resource::<crate::fabric_domains::owners::HairRenderWorldOwner>()
            .init_resource::<crate::fabric_domains::owners::VolumeRenderWorldOwner>()
            .init_resource::<crate::fabric_domains::owners::DecalRenderWorldOwner>()
            .init_resource::<crate::fabric_domains::owners::FabricDomainOwnerDiagnostics>()
            .init_resource::<crate::fabric_domains::owners::FabricDomainExtractionDiagnostics>()
            .init_resource::<RenderGeometryRegistry>()
            .init_resource::<VirtualGeometryRuntimeRegistry>()
            .init_resource::<crate::geometry::GeometryDrawPhases>()
            .init_resource::<crate::geometry::VirtualGeometryDrawPublications>()
            .init_resource::<RenderShadingGeometryRegistry>()
            .init_resource::<crate::raytrace::RenderWorldAcceleration>()
            .init_resource::<crate::raytrace::RayTraceFrameResources>()
            .init_resource::<ShadingGeometryStaging>()
            .init_resource::<crate::extract::lifecycle::ExtractionClock>()
            .init_resource::<RenderGpuScene>()
            .init_resource::<GpuCompletionTracker>()
            .add_systems(
                RenderStartup,
                (
                    init_gpu_resource::<GpuSceneBuffers>,
                    init_gpu_resource::<GpuSceneBindGroup>,
                    init_gpu_resource::<RenderGeometryBuffers>,
                    init_gpu_resource::<RenderShadingGeometryBuffers>,
                    init_gpu_resource::<GeometryBindGroup>,
                    rebuild_gpu_scene_after_device_startup,
                )
                    .chain(),
            )
            .add_systems(RenderStartup, crate::decal::init_decal_resolve_pipeline)
            .add_systems(
                RenderStartup,
                crate::hair_raster::init_hair_raster_pipeline
                    .after(crate::gpu::rhi::init_scene_rhi),
            )
            .add_systems(RenderStartup, init_raytrace_consumer)
            .add_systems(
                ExtractSchedule,
                (
                    extract_canonical_lighting,
                    crate::fabric_domains::owners::extract_hair_and_decal_fabric_inputs,
                    extract_scene_instances,
                    extract_shading_geometry,
                    crate::material::systems::invalidate_scene_materials
                        .after(crate::material::systems::extract_standard_materials)
                        .after(extract_scene_instances),
                    retire_unused_geometry,
                ),
            )
            .configure_sets(
                Render,
                (
                    crate::fabric_frame::RenderFabricSystems::CollectDomainContributions,
                    crate::fabric_frame::RenderFabricSystems::BuildFrame,
                )
                    .chain()
                    .in_set(RenderSystems::PrepareResources),
            )
            .add_systems(
                Render,
                (
                    apply_extracted_scene_changes.in_set(RenderSystems::PrepareResources),
                    compare_scene_mirror
                        .after(apply_extracted_scene_changes)
                        .in_set(RenderSystems::PrepareResources),
                    sync_geometry_registry
                        .after(apply_extracted_scene_changes)
                        .before(crate::visibility::systems::build_unified_visibility)
                        .in_set(RenderSystems::PrepareResources),
                    crate::fabric_domains::owners::collect_owned_domain_contributions_system
                        .in_set(
                            crate::fabric_frame::RenderFabricSystems::CollectDomainContributions,
                        ),
                    crate::fabric_frame::build_render_fabric_frame_system
                        .after(apply_extracted_scene_changes)
                        .after(sync_geometry_registry)
                        .in_set(crate::fabric_frame::RenderFabricSystems::BuildFrame),
                    begin_scene_fabric_coordinator_frame
                        .after(crate::fabric_frame::build_render_fabric_frame_system)
                        .in_set(RenderSystems::PrepareResources),
                    prepare_scene_fabric_hair
                        .after(begin_scene_fabric_coordinator_frame)
                        .in_set(RenderSystems::PrepareResources),
                    isolate_scene_fabric_water_packet
                        .after(RenderSystems::PrepareResources)
                        .before(RenderSystems::PrepareResourcesFlush),
                    sync_shading_geometry_registry
                        .after(apply_extracted_scene_changes)
                        .after(sync_geometry_registry)
                        .before(crate::visibility::systems::build_unified_visibility)
                        .in_set(RenderSystems::PrepareResources),
                    crate::geometry::build_geometry_draw_phases
                        .after(crate::visibility::systems::build_unified_visibility)
                        .after(sync_geometry_registry)
                        .in_set(RenderSystems::PrepareResources),
                    crate::raytrace::build_render_world_acceleration
                        .after(crate::visibility::systems::build_unified_visibility)
                        .after(sync_shading_geometry_registry)
                        .in_set(RenderSystems::PrepareResources),
                    write_gpu_scene_buffers.in_set(RenderSystems::PrepareResourcesFlush),
                    upload_geometry_buffers
                        .after(sync_geometry_registry)
                        .in_set(RenderSystems::PrepareResourcesFlush),
                    upload_shading_geometry_buffers
                        .after(sync_shading_geometry_registry)
                        .in_set(RenderSystems::PrepareResourcesFlush),
                    prepare_gpu_scene_bind_group
                        .after(write_gpu_scene_buffers)
                        .in_set(RenderSystems::PrepareBindGroups),
                    prepare_geometry_bind_group
                        .after(upload_geometry_buffers)
                        .in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            .add_systems(
                RenderGraph,
                (
                    reclaim_completed_handles,
                    fail_unrecorded_scene_fabric_frame.after(crate::completion::track_submission),
                    finish_scene_fabric_particle_completions
                        .after(crate::completion::track_submission),
                    advance_scene_fabric_coordinator_completion
                        .after(crate::completion::track_submission),
                )
                    .in_set(RenderGraphSystems::Finish),
            );
        render_app.add_systems(
            bevy_core_pipeline::Core3d,
            (
                crate::raytrace::raytrace_begin_frame,
                crate::raytrace::raytrace_probe_pass,
            )
                .chain()
                .before(bevy_core_pipeline::Core3dSystems::MainPass),
        );
        render_app.add_systems(
            bevy_core_pipeline::Core3d,
            record_scene_fabric_coordinator.before(bevy_core_pipeline::Core3dSystems::MainPass),
        );
        render_app.add_systems(
            bevy_core_pipeline::Core3d,
            restore_scene_fabric_water_packet
                .after(bevy_core_pipeline::Core3dSystems::MainPass)
                .before(crate::water::WaterSurfaceSystems::Draw),
        );
    }
}

/// Prepares the real Hair GPU packet that the unified Core3d consumer appends
/// to the frame input. Projection, allocation, upload and bind-group creation
/// all happen before recording; a missing pipeline or invalid owner snapshot
/// leaves an explicit error and no Hair pass.
fn prepare_scene_fabric_hair(
    frame: bevy_ecs::prelude::Res<crate::fabric_frame::RenderFabricFrame>,
    owner: bevy_ecs::prelude::Res<crate::fabric_domains::owners::HairRenderWorldOwner>,
    rhi: bevy_ecs::prelude::Res<crate::gpu::rhi::SceneRhi>,
    device: bevy_ecs::prelude::Res<RenderDevice>,
    views: bevy_ecs::prelude::Query<&ExtractedView>,
    pipelines: bevy_ecs::prelude::NonSend<crate::hair_raster::HairRasterPipelineState>,
    mut state: bevy_ecs::prelude::NonSendMut<crate::hair_raster::HairRasterCoordinatorState>,
) {
    state.last_error = None;
    if state.packet.is_some() || state.pending_adapter.is_some() {
        state.last_error = Some("Hair coordinator still owns the previous frame adapter".into());
        return;
    }
    if owner.is_empty() {
        return;
    }
    let Some(pipelines) = pipelines.pipelines.as_ref() else {
        state.last_error = Some(
            pipelines
                .error
                .clone()
                .unwrap_or_else(|| "Hair raster pipeline is unavailable".into()),
        );
        return;
    };
    let mut views = views.iter();
    let Some(view) = views.next() else {
        state.last_error = Some("Hair coordinator requires one active ExtractedView".into());
        return;
    };
    if views.next().is_some() {
        state.last_error = Some(
            "Hair coordinator currently requires a single active view; per-view aggregation is not yet implemented".into(),
        );
        return;
    }
    let viewport = [view.viewport.z, view.viewport.w];
    let clip_from_world = view
        .clip_from_world
        .unwrap_or_else(|| view.clip_from_view * view.world_from_view.to_matrix().inverse());
    match crate::hair_raster::HairRasterDomainPacket::prepare_from_owner(
        &mut state.gpu,
        rhi.as_ref(),
        frame.frame(),
        owner.as_ref(),
        &clip_from_world.to_cols_array_2d(),
        viewport,
        HairSoftRasterAbi::default(),
        device.limits().max_compute_workgroups_per_dimension,
        pipelines,
    ) {
        Ok(packet) => state.packet = Some(packet),
        Err(error) => {
            state.last_error = Some(format!("Hair coordinator preparation failed: {error}"))
        }
    }
}

/// All inputs for the one RenderApp/Core3d production consumer for the
/// SceneFabric batch. Keeping this as one derived system parameter avoids the
/// Bevy function-parameter ceiling while preserving one atomic coordinator
/// call and one host command-buffer handoff.
#[derive(SystemParam)]
struct SceneFabricCoordinatorParams<'w, 's> {
    frame: bevy_ecs::prelude::Res<'w, crate::fabric_frame::RenderFabricFrame>,
    fabric: bevy_ecs::prelude::Res<'w, SceneFabric>,
    rhi: bevy_ecs::prelude::Res<'w, crate::gpu::rhi::SceneRhi>,
    particle_host: bevy_ecs::prelude::ResMut<'w, crate::particle::ParticleSceneFabricHost>,
    coordinator: bevy_ecs::prelude::ResMut<'w, SceneFabricCoordinator>,
    status: bevy_ecs::prelude::ResMut<'w, SceneFabricCoordinatorBatchStatus>,
    particle_completions: bevy_ecs::prelude::ResMut<'w, SceneFabricCoordinatorParticleCompletions>,
    decal_leases: bevy_ecs::prelude::NonSendMut<'w, SceneFabricCoordinatorDecalLeases>,
    decal_host: bevy_ecs::prelude::ResMut<'w, crate::decal::DecalResolveHost>,
    decal_pipeline: bevy_ecs::prelude::Res<'w, crate::decal::DecalResolvePipeline>,
    device: bevy_ecs::prelude::Res<'w, RenderDevice>,
    cache: bevy_ecs::prelude::Res<'w, PipelineCache>,
    volumetrics_settings: bevy_ecs::prelude::Res<'w, crate::shading::PrismVolumetricsSettings>,
    volumetrics_pipeline: Option<bevy_ecs::prelude::Res<'w, crate::shading::VolumetricsPipeline>>,
    view: ViewQuery<
        'w,
        's,
        (
            &'static ExtractedView,
            Option<&'static crate::shading::ViewVisibilityBuffer>,
            Option<&'static crate::shading::ViewSsrTextures>,
            Option<&'static crate::shading::ViewVolumetrics>,
            Option<&'static crate::shading::ViewVolumetricsBindGroups>,
            Option<&'static mut crate::shading::ViewResponseComposition>,
            Option<&'static crate::shading::composition::ResponseCompositionTarget>,
            Option<&'static crate::shading::taa::ViewTaaBindGroup>,
            Option<&'static crate::shading::taa::ViewTaa>,
        ),
    >,
    water_pipelines: Option<bevy_ecs::prelude::Res<'w, crate::water::WaterComputePipelines>>,
    spray_pipelines:
        Option<bevy_ecs::prelude::Res<'w, crate::water::spray_history::SprayPipelines>>,
    spray_history: bevy_ecs::prelude::ResMut<'w, crate::water::spray_history::SprayHistory>,
    context: RenderContext<'w, 's>,
    hair_state: bevy_ecs::prelude::NonSendMut<'w, crate::hair_raster::HairRasterCoordinatorState>,
    water_packet: bevy_ecs::prelude::Res<'w, SceneFabricCoordinatorWaterPacket>,
    shading_settings: bevy_ecs::prelude::Res<'w, crate::shading::PrismShadingSettings>,
    composition_pipelines:
        bevy_ecs::prelude::Res<'w, crate::shading::composition::ResidentCompositionPipelines>,
    taa_pipeline: bevy_ecs::prelude::Res<'w, crate::shading::taa::TaaResolvePipeline>,
    taa_history_cache: bevy_ecs::prelude::ResMut<'w, crate::shading::taa::TaaHistoryCache>,
    scene_history: bevy_ecs::prelude::ResMut<'w, crate::gpu::fabric_history::SceneFabricHistory>,
}

/// The one RenderApp/Core3d production consumer for the SceneFabric batch.
///
/// The top-level SceneFabric consumer admits only producer-owned physical
/// packets. Particle, Hair, Volumetrics, Decal and Water each contribute their
/// real buffers, pipelines, bind groups and typed recorders; a missing packet
/// is left missing so the coordinator's strict coverage check rejects the
/// frame instead of accepting a synthetic pass.
fn record_scene_fabric_coordinator(params: SceneFabricCoordinatorParams) {
    let SceneFabricCoordinatorParams {
        frame,
        fabric,
        rhi,
        mut particle_host,
        mut coordinator,
        mut status,
        mut particle_completions,
        mut decal_leases,
        mut decal_host,
        decal_pipeline,
        device,
        cache,
        volumetrics_settings,
        volumetrics_pipeline,
        view,
        water_pipelines,
        spray_pipelines,
        spray_history,
        mut context,
        mut hair_state,
        water_packet,
        shading_settings,
        composition_pipelines,
        taa_pipeline,
        mut taa_history_cache,
        mut scene_history,
    } = params;
    let frame_number = frame.frame();
    *status = SceneFabricCoordinatorBatchStatus {
        frame: frame_number,
        ..Default::default()
    };
    for item in frame.items() {
        match item.domain {
            RenderDomain::Surface => status.surface_items += 1,
            RenderDomain::Particle => status.particle_items += 1,
            RenderDomain::Hair => status.hair_items += 1,
            RenderDomain::Volume => status.volume_items += 1,
            RenderDomain::Decal => status.decal_items += 1,
        }
    }

    let particle_input = match particle_host.take_for_frame(frame_number) {
        Ok(input) => input,
        Err(error) => {
            let message = format!("Particle coordinator publication rejected: {error}");
            status.last_error = Some(message.clone());
            bevy_log::error!("{message}");
            return;
        }
    };

    let (
        extracted_view,
        visibility,
        ssr,
        volumetrics,
        volumetrics_groups,
        responses,
        response_target,
        taa_bind_group,
        taa,
    ) = view.into_inner();
    let mut responses = responses;
    let extent = visibility
        .map(|visibility| Extent3d::new_2d(visibility.size.x, visibility.size.y))
        .or_else(|| particle_input.as_ref().map(|input| input.extent))
        .or_else(|| hair_state.packet.as_ref().map(|packet| packet.extent()))
        .unwrap_or_else(|| Extent3d::new_2d(1, 1));

    let mut items = Vec::new();
    let mut active_passes = Vec::<PassKey>::new();
    let mut resources = Vec::new();
    let mut recorders = Vec::new();
    let mut particle_appended = false;
    if let Some(input) = particle_input.as_ref() {
        particle_appended = match crate::particle::append_particle_fabric_parts(
            input,
            &mut items,
            &mut active_passes,
            &mut resources,
            &mut recorders,
        ) {
            Ok(appended) => appended,
            Err(error) => {
                let message = format!("Particle coordinator batch rejected: {error}");
                status.last_error = Some(message.clone());
                bevy_log::error!("{message}");
                return;
            }
        };
    }
    status.particle_passes = active_passes.len();
    status.particle_resources = resources.len();

    let mut batch = SceneFabricCoordinatorInput::new(frame_number, extent);
    if particle_appended {
        if let Err(error) = batch.push_request(SceneFabricRequest {
            frame: frame_number,
            extent,
            items: &items,
            active_passes: &active_passes,
            fulfilled_dependencies: &[],
            resources,
            recorders,
        }) {
            let message = format!("SceneFabric coordinator batch assembly failed: {error}");
            status.last_error = Some(message.clone());
            bevy_log::error!("{message}");
            return;
        }
    }

    if let Some(packet) = hair_state.packet.as_ref() {
        if packet.frame() != frame_number {
            hair_state.last_error = Some(format!(
                "Hair coordinator packet frame {} does not match frame {frame_number}",
                packet.frame()
            ));
            status.last_error = hair_state.last_error.clone();
            return;
        }
        if let Err(error) = packet.append_to(&mut batch) {
            hair_state.last_error = Some(format!("Hair coordinator batch rejected: {error}"));
            status.last_error = hair_state.last_error.clone();
            bevy_log::error!("Hair coordinator batch rejected: {error}");
            return;
        }
    } else if status.hair_items != 0 {
        let error = hair_state
            .last_error
            .clone()
            .unwrap_or_else(|| "Hair semantic input has no admitted GPU packet".into());
        status.last_error = Some(error.clone());
        bevy_log::error!("{error}");
        return;
    }

    let mut producer_templates = Vec::new();
    if let Some(pipeline) = volumetrics_pipeline.as_deref() {
        let scatter = cache.get_compute_pipeline(pipeline.scatter());
        let integrate = cache.get_compute_pipeline(pipeline.integrate());
        let apply = cache.get_compute_pipeline(pipeline.apply());
        if let Err(error) = crate::shading::append_volumetrics_batch(
            &mut batch,
            &mut producer_templates,
            volumetrics_settings.as_ref(),
            frame_number,
            volumetrics,
            volumetrics_groups,
            visibility,
            ssr,
            scatter,
            integrate,
            apply,
        ) {
            status.last_error = Some(format!("Volume coordinator batch rejected: {error}"));
            bevy_log::error!("Volume coordinator batch rejected: {error}");
            return;
        }
    }

    let publications = match decal_host.take_for_frame(frame_number) {
        Ok(publications) => publications,
        Err(error) => {
            status.last_error = Some(error.clone());
            bevy_log::error!("Decal coordinator publication rejected: {error}");
            return;
        }
    };
    let native_pipeline = cache.get_compute_pipeline(decal_pipeline.resolve());
    let mut decal_adapter = match (publications.is_empty(), native_pipeline) {
        (true, _) => None,
        (false, Some(native_pipeline)) => {
            match crate::decal::DecalResolveBatchAdapter::from_publications(
                rhi.as_ref(),
                device.as_ref(),
                &decal_pipeline,
                native_pipeline,
                publications,
            ) {
                Ok(adapter) => Some(adapter),
                Err(error) => {
                    status.last_error = Some(format!("Decal coordinator batch rejected: {error}"));
                    bevy_log::error!("Decal coordinator batch rejected: {error}");
                    return;
                }
            }
        }
        (false, None) => {
            status.last_error = Some("Decal resolve pipeline is not ready".into());
            bevy_log::error!("Decal resolve pipeline is not ready");
            return;
        }
    };
    let mut decal_object_resources = Vec::new();
    if let Some(adapter) = decal_adapter.as_ref() {
        if let Err(error) = adapter.append_to(&mut batch, &mut decal_object_resources) {
            status.last_error = Some(format!("Decal coordinator batch rejected: {error}"));
            bevy_log::error!("Decal coordinator batch rejected: {error}");
            return;
        }
        if let Err(error) =
            batch.append_object_parts(adapter.object_accesses(), decal_object_resources)
        {
            status.last_error = Some(format!(
                "Decal coordinator object resources rejected: {error}"
            ));
            bevy_log::error!("Decal coordinator object resources rejected: {error}");
            return;
        }
    }

    if let (Some(bodies), Some(pipelines), Some(spray_pipelines)) = (
        water_packet.0.as_ref(),
        water_pipelines.as_deref(),
        spray_pipelines.as_deref(),
    ) {
        if !bodies.is_empty() {
            let result = fabric.append_water_batch(
                &mut batch,
                &mut producer_templates,
                frame_number,
                bodies,
                pipelines,
                cache.as_ref(),
                spray_pipelines,
            );
            match result {
                Ok(result) => {
                    if result.requires_history_reset {
                        if let Some(body) = bodies.bodies.first() {
                            if let Some(spray) = body.spray.as_ref() {
                                spray_history.reset_missing(context.command_encoder(), spray);
                            }
                        }
                    }
                }
                Err(error) => {
                    status.last_error = Some(format!("Water coordinator batch rejected: {error}"));
                    bevy_log::error!("Water coordinator batch rejected: {error}");
                    return;
                }
            }
        }
    }

    let composition_appended = if let (Some(responses), Some(target)) =
        (responses.as_deref_mut(), response_target)
    {
        match crate::shading::composition::append_composition_fabric_parts(
            responses,
            composition_pipelines.as_ref(),
            target,
            frame.as_ref(),
            &mut batch,
            &mut producer_templates,
            shading_settings.frame_index,
        ) {
            Ok(true) => true,
            Ok(false) => false,
            Err(error) => {
                status.last_error = Some(format!(
                    "Composition coordinator batch rejected: {error}"
                ));
                bevy_log::error!("Composition coordinator batch rejected: {error}");
                return;
            }
        }
    } else {
        false
    };

    let taa_producer = if shading_settings.enable_taa {
        let Some(resolve) = cache.get_compute_pipeline(taa_pipeline.resolve()) else {
            status.last_error = Some("TAA resolve pipeline is not ready".into());
            bevy_log::error!("TAA resolve pipeline is not ready");
            return;
        };
        let (Some(visibility), Some(taa_group), Some(taa)) = (visibility, taa_bind_group, taa)
        else {
            status.last_error = Some("TAA coordinator is missing prepared view resources".into());
            bevy_log::error!("TAA coordinator is missing prepared view resources");
            return;
        };
        match crate::shading::taa::append_taa_fabric_parts(
            extracted_view,
            visibility,
            taa_group,
            taa,
            resolve,
            frame.as_ref(),
            taa_history_cache.as_ref(),
            scene_history.as_ref(),
            &mut batch,
            &mut producer_templates,
            composition_appended,
        ) {
            Ok(producer) => Some(producer),
            Err(error) => {
                status.last_error = Some(format!("TAA coordinator batch rejected: {error}"));
                bevy_log::error!("TAA coordinator batch rejected: {error}");
                return;
            }
        }
    } else {
        None
    };

    if batch.pass_count() == 0 {
        if let Some(input) = particle_input {
            input.completion.complete();
        }
        status.last_error = Some("SceneFabric coordinator received no real domain recorder".into());
        bevy_log::error!(
            "SceneFabric coordinator frame {frame_number} received no real domain recorder"
        );
        return;
    }

    if let Err(error) = coordinator.prepare_and_record_input(
        fabric.as_ref(),
        rhi.as_ref(),
        batch,
        &producer_templates,
        &mut context,
    ) {
        let message = format!("SceneFabric coordinator batch assembly failed: {error}");
        status.last_error = Some(message.clone());
        bevy_log::error!("{message}");
        return;
    }
    if let Err(error) = coordinator.finish_recording() {
        let message = format!("SceneFabric coordinator finish_recording failed: {error}");
        status.last_error = Some(message.clone());
        bevy_log::error!("{message}");
        return;
    }
    if let Some(input) = particle_input {
        particle_completions.0.push(input.completion);
    }
    if let Some(adapter) = decal_adapter.take() {
        decal_leases
            .pending
            .push(crate::decal::DecalResolveBatchLease::from_adapter(adapter));
    }
    if let Some(packet) = hair_state.packet.take() {
        hair_state.pending_adapter = Some(packet.into_adapter());
    }
    if composition_appended {
        if let Some(responses) = responses.as_deref_mut() {
            if !responses.consume_publication(shading_settings.frame_index) {
                status.last_error = Some("Composition publication was lost before consume".into());
                bevy_log::error!("Composition publication was lost before consume");
            }
        } else {
            status.last_error = Some("Composition publication was lost before consume".into());
            bevy_log::error!("Composition publication was lost before consume");
        }
    }
    if let Some(taa_producer) = taa_producer {
        if taa_history_cache.mark_encoded(taa_producer.owner) {
            if let Err(error) = scene_history.lifecycle.enqueue([taa_producer]) {
                status.last_error = Some(format!("TAA history enqueue failed: {error}"));
                bevy_log::error!("TAA history enqueue failed: {error}");
            }
        }
    }
}

/// Completes Particle's producer-owned publication only after the same host
/// queue that contains the coordinator command buffer has been submitted.
fn finish_scene_fabric_particle_completions(
    mut completions: bevy_ecs::prelude::ResMut<SceneFabricCoordinatorParticleCompletions>,
    mut decal_leases: bevy_ecs::prelude::NonSendMut<SceneFabricCoordinatorDecalLeases>,
    mut hair_state: bevy_ecs::prelude::NonSendMut<crate::hair_raster::HairRasterCoordinatorState>,
    tracker: bevy_ecs::prelude::Res<crate::completion::GpuCompletionTracker>,
    queue: Option<bevy_ecs::prelude::Res<RenderQueue>>,
) {
    let pending = core::mem::take(&mut completions.0);
    let Some(queue) = queue else {
        completions.0.extend(pending);
        return;
    };
    for completion in pending {
        queue.on_submitted_work_done(Box::new(move || completion.complete()));
    }
    let next = tracker.next_submission_value();
    if next.0 > 1 {
        let completion = GpuCompletionValue(next.0 - 1);
        if let Some(adapter) = hair_state.pending_adapter.take() {
            match hair_state.gpu.submit_batch_adapter(adapter, completion) {
                Ok(submission) => hair_state.in_flight.push(submission),
                Err(error) => {
                    hair_state.last_error = Some(format!(
                        "Hair coordinator host submission rejected: {error}"
                    ));
                }
            }
        }
        let pending = core::mem::take(&mut decal_leases.pending);
        decal_leases
            .in_flight
            .extend(pending.into_iter().map(|lease| (completion, lease)));
    }
    let completed = tracker.completed_value();
    let pending = core::mem::take(&mut hair_state.in_flight);
    for submission in pending {
        match hair_state.gpu.complete_batch_adapter(submission, completed) {
            Ok(()) => {}
            Err(submission) => hair_state.in_flight.push(submission),
        }
    }
}

/// Opens the one coordinator boundary after the shared Surface/domain frame is
/// built. Actual typed producer packets are admitted later through
/// [`SceneFabricCoordinator::prepare`]; no missing packet is represented as a
/// fabricated dispatch here.
fn begin_scene_fabric_coordinator_frame(
    frame: bevy_ecs::prelude::Res<crate::fabric_frame::RenderFabricFrame>,
    fabric: bevy_ecs::prelude::Res<SceneFabric>,
    mut coordinator: bevy_ecs::prelude::ResMut<SceneFabricCoordinator>,
) {
    if let Err(error) = coordinator.begin_frame(frame.frame(), fabric.as_ref()) {
        bevy_log::error!("SceneFabric coordinator frame admission failed: {error}");
    }
}

/// Hides the real prepared Water packet from the legacy Water simulation
/// recorder for the duration of the coordinator-owned Core3d phase. The
/// coordinator reads the moved packet from [`SceneFabricCoordinatorWaterPacket`]
/// and the packet is restored after MainPass for Water's raster consumer.
fn isolate_scene_fabric_water_packet(
    mut bodies: bevy_ecs::prelude::ResMut<crate::water::WaterGpuBodies>,
    mut packet: bevy_ecs::prelude::ResMut<SceneFabricCoordinatorWaterPacket>,
) {
    if packet.0.is_some() {
        bevy_log::error!("SceneFabric Water packet was not restored before the next prepare phase");
        return;
    }
    if bodies.is_empty() {
        return;
    }
    packet.0 = Some(core::mem::take(&mut *bodies));
}

/// Restores the exact prepared Water packet after the coordinator/legacy
/// simulation phase and before Water surface rasterization. No resource is
/// recreated or fabricated; this only moves the producer-owned packet back to
/// its original render-world slot.
fn restore_scene_fabric_water_packet(
    mut bodies: bevy_ecs::prelude::ResMut<crate::water::WaterGpuBodies>,
    mut packet: bevy_ecs::prelude::ResMut<SceneFabricCoordinatorWaterPacket>,
) {
    let Some(restored) = packet.0.take() else {
        return;
    };
    if !bodies.is_empty() {
        bevy_log::error!(
            "SceneFabric Water restore found an unexpected resident packet; preserving the coordinator packet"
        );
        packet.0 = Some(restored);
        return;
    }
    *bodies = restored;
}

/// Binds the coordinator's recorded frame to Bevy's actual host submission
/// token, then polls completion from the same tracker. This system never treats
/// command recording or callback registration as completion by itself.
fn advance_scene_fabric_coordinator_completion(
    tracker: bevy_ecs::prelude::Res<crate::completion::GpuCompletionTracker>,
    queue: Option<bevy_ecs::prelude::Res<RenderQueue>>,
    mut decal_leases: bevy_ecs::prelude::NonSendMut<SceneFabricCoordinatorDecalLeases>,
    mut coordinator: bevy_ecs::prelude::ResMut<SceneFabricCoordinator>,
) {
    if queue.is_some()
        && coordinator.phase() == crate::fabric_runtime::SceneFabricCoordinatorPhase::Recorded
    {
        let next = tracker.next_submission_value();
        if next.0 > 1 {
            if let Err(error) = coordinator.mark_host_submitted(GpuCompletionValue(next.0 - 1)) {
                bevy_log::error!(
                    "SceneFabric coordinator host submission admission failed: {error}"
                );
            }
        }
    }
    let completed = tracker.completed_value();
    decal_leases
        .in_flight
        .retain(|(completion, _)| *completion > completed);
    let _ = coordinator.poll_completion(completed);
}

/// Closes a coordinator frame that crossed the production frame boundary but
/// did not receive a complete typed runtime. This is intentionally a hard
/// failure: semantic frame aggregation is not execution evidence, and the
/// next frame must not inherit a stale `Compiled`/`Prepared` phase.
fn fail_unrecorded_scene_fabric_frame(
    mut coordinator: bevy_ecs::prelude::ResMut<SceneFabricCoordinator>,
) {
    match coordinator.phase() {
        crate::fabric_runtime::SceneFabricCoordinatorPhase::Compiled => {
            if let Err(error) = coordinator.fail_unprepared_frame() {
                bevy_log::error!("SceneFabric coordinator strict admission failure: {error}");
            }
        }
        crate::fabric_runtime::SceneFabricCoordinatorPhase::Prepared => {
            if let Err(error) = coordinator.fail_unrecorded_frame() {
                bevy_log::error!("SceneFabric coordinator strict recording failure: {error}");
            }
        }
        crate::fabric_runtime::SceneFabricCoordinatorPhase::Idle
        | crate::fabric_runtime::SceneFabricCoordinatorPhase::Recorded
        | crate::fabric_runtime::SceneFabricCoordinatorPhase::Submitted
        | crate::fabric_runtime::SceneFabricCoordinatorPhase::Completed
        | crate::fabric_runtime::SceneFabricCoordinatorPhase::Failed => {}
    }
}

fn init_raytrace_consumer(
    mut commands: bevy_ecs::prelude::Commands,
    device: bevy_ecs::prelude::Res<RenderDevice>,
) {
    commands.insert_resource(crate::raytrace::GpuRayTraversal::new(&device));
}

fn rebuild_gpu_scene_after_device_startup(
    mut scene: bevy_ecs::prelude::ResMut<RenderGpuScene>,
    mut buffers: bevy_ecs::prelude::ResMut<GpuSceneBuffers>,
    mut diagnostics: bevy_ecs::prelude::ResMut<GpuSceneDiagnostics>,
) {
    scene.rebuild_gpu_buffers(&mut buffers);
    diagnostics.buffer_version = scene.buffer_version();
    diagnostics.buffer_rebuilds = diagnostics.buffer_rebuilds.saturating_add(1);
}
