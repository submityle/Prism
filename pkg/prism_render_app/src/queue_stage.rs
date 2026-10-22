//! The queue stage: the visible set becomes the frame's draws.
//!
//! Between culling (which decides what is visible) and recording (which issues
//! the commands) sits the queue: it resolves each view-scoped work item against
//! the scene's tables, specializes its pipeline, and appends a phase item. The
//! record stage only ever walks a queue that is already final.

use prism_ecs::prelude::{Commands, Res, ResMut};
use prism_render_driver::{BindGroupLayoutId, FrameSync, Instance, RenderDevice, RenderQueue};

use crate::draw::canonical;
use crate::material::virtual_texture::{NativeMaterialVt, NativeVtScheduler, NativeVtSources};
use crate::material::MaterialParameterBuffers;
use crate::mesh_pipeline::{MeshPipelineProvider, MeshPipelineSpecializer};
use crate::mesh_pipelines::MeshPipelines;
use crate::queue::{CanonicalMaterialBindings, RenderQueues};
use crate::render_asset::RenderAssets;
use crate::runtime_resource::RenderRuntimeResource;
use crate::scene::SceneRegistry;
use crate::stats::{RenderFrameStats, ViewQueueAdmissionRejection};
use crate::surface_view_frame::{SurfaceViewQueue, SurfaceViewQueueError, SurfaceViewQueues};
use crate::surface_views::ExtractedSurfaceViews;
use crate::visibility_bridge::{ExtractedVisibilityView, ExtractedVisibilityViews};

use crate::frame::RENDER_DEPTH_FORMAT;
use crate::mesh::RenderMesh;

/// Resolves every validated camera into independent native queues.
///
/// Runs in [`RenderSystems::Queue`](crate::RenderSystems::Queue), before the
/// record stage. Four things happen, in order:
///
/// 1. every bound geometry's mesh class is synced from its prepared mesh, so the
///    layout and buffers the queue specializes on are the ones the mesh actually
///    has this frame;
/// 2. camera-local visibility/surface pairs are admitted in stable Entity order;
/// 3. all arena capacity, material buffers and frame slots are prepared before
///    any binding is emitted; no camera loop can resize a previous binding;
/// 4. each camera's work is resolved, specialized and sorted independently.
///
/// Publishes [`SurfaceViewQueues`] using deferred commands even on failure.
/// Explicit global [`RenderQueues`] and window overrides are never modified.
///
/// A work item whose geometry, material, mesh, or class is not resident is
/// skipped: the queue and the frame agree on what is drawable, rather than the
/// frame drawing with a class nothing declared.
pub(crate) fn fill_queues<I>(
    visibility_views: Res<ExtractedVisibilityViews>,
    mut registry: ResMut<SceneRegistry>,
    meshes: Res<RenderAssets<RenderMesh>>,
    mut specializer: ResMut<MeshPipelineSpecializer>,
    pipelines: Res<MeshPipelines>,
    mut commands: Commands,
    mut canonical_material: ResMut<CanonicalMaterialBindings>,
    material_buffers: Res<MaterialParameterBuffers>,
    surface_views: Res<ExtractedSurfaceViews>,
    (mut stats, runtime, mut transforms, record): (
        ResMut<RenderFrameStats>,
        Res<RenderRuntimeResource<I>>,
        ResMut<crate::RenderTransforms>,
        Res<crate::RenderRecord>,
    ),
    (config, scheduler, vt_owner, vt_sources): (
        Option<Res<crate::config::RenderAppConfig>>,
        Option<Res<NativeVtScheduler>>,
        Option<Res<NativeMaterialVt<I::Device>>>,
        Option<Res<NativeVtSources>>,
    ),
    #[cfg(feature = "physics-resident")] mut resident: (
        Option<Res<crate::physics_surface::resident::ResidentSurfaceHost>>,
        Option<ResMut<crate::physics_surface::resident::ResidentSurfaceImports>>,
        Option<Res<crate::physics_surface::resident::ResidentSurfaceDrawFunctions>>,
    ),
) where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    stats.view_queue_rejection = None;
    registry.sync_mesh_classes(&meshes);
    specializer.sync_response_modules(pipelines.response_modules());

    let runtime = runtime.runtime();
    #[cfg(feature = "physics-resident")]
    if let Some(imports) = resident.1.as_mut() {
        imports.queue_rejections.clear();
    }
    let mut frame = SurfaceViewQueues::pending(record.frame_index, runtime.device_generation());
    if !crate::plugin::native_vt_resources_ready(
        config.is_some(),
        scheduler.as_deref(),
        vt_owner.is_some(),
        vt_sources.is_some(),
    ) {
        // Replace the previous snapshot even on failure, before waiting,
        // preparing bindings or admitting any draw with stale VT metadata.
        frame.fail(SurfaceViewQueueError::MaterialBuffers);
        publish_frame(&mut commands, frame);
        return;
    }
    let views = frame.admit(visibility_views.iter(), surface_views.iter());
    if visibility_views.is_empty() {
        stats.view_queue_rejection = Some(ViewQueueAdmissionRejection::MissingVisibilityView);
    }
    let surface_layout = pipelines.surface_bind_group_layout();
    // The arena may destroy old allocations during layout/capacity preflight.
    // Drain previous submissions once, never inside the per-camera loop. This
    // conservative barrier can later be replaced by an owner retirement fence.
    if !runtime.wait_idle() {
        frame.fail(SurfaceViewQueueError::CompletionUnconfirmed);
        publish_frame(&mut commands, frame);
        return;
    }
    if let Err(error) = prepare_view_bindings(
        &mut canonical_material,
        runtime.device(),
        runtime.queue(),
        runtime.device_generation(),
        surface_layout,
        &material_buffers,
        &views,
    ) {
        frame.fail(error);
        publish_frame(&mut commands, frame);
        return;
    }
    let mut camera_queues = Vec::with_capacity(views.len());
    let mut queued = 0;
    for (slot, view) in views.iter().enumerate() {
        // The preflight checked that every dense slot is representable.
        let frame_slot = slot as u32;
        let mut queues = RenderQueues::new();
        for work_item in &view.work {
            let cpu_requests = || {
                registry
                    .resolve_passes(
                        work_item,
                        &meshes,
                        Some(RENDER_DEPTH_FORMAT),
                        1,
                        canonical::MESH,
                    )
                    .unwrap_or_default()
                    .into_iter()
                    .map(|request| (request, None))
                    .collect::<Vec<(
                        crate::queue::DrawRequest,
                        Option<crate::queue::ExternalIndirectDraw>,
                    )>>()
            };
            #[cfg(feature = "physics-resident")]
            let requests = if let Some(imports) = resident
                .1
                .as_mut()
                .filter(|imports| imports.contains(work_item.geometry))
            {
                use crate::physics_surface::resident::ResidentSurfaceRejection;
                let key = (view.camera, work_item.geometry);
                let (Some(host), Some(draws)) = (resident.0.as_ref(), resident.2.as_ref()) else {
                    imports
                        .queue_rejections
                        .insert(key, ResidentSurfaceRejection::HostMismatch);
                    continue;
                };
                if !host.matches_device_queue(
                    runtime.device(),
                    runtime.queue(),
                    runtime.device_generation(),
                ) {
                    imports
                        .queue_rejections
                        .insert(key, ResidentSurfaceRejection::WrongDevice);
                    continue;
                }
                let draw = if surface_layout.is_some() {
                    draws.surface
                } else {
                    draws.unbound
                };
                match imports.resolve_passes(
                    host,
                    &registry,
                    work_item,
                    Some(RENDER_DEPTH_FORMAT),
                    1,
                    draw,
                ) {
                    Ok(Some(requests)) => requests
                        .into_iter()
                        .map(|request| (request.request, Some(request.lease.queued_draw())))
                        .collect(),
                    Ok(None) => unreachable!("resident entry disappeared during queue admission"),
                    Err(error) => {
                        imports.queue_rejections.insert(key, error);
                        continue;
                    }
                }
            } else {
                cpu_requests()
            };
            #[cfg(not(feature = "physics-resident"))]
            let requests = cpu_requests();
            for (mut request, external) in requests {
                let resident_draw = external.is_some();
                let default_mesh_draw = request.draw == canonical::MESH;
                let transformed = if resident_draw {
                    // The producer emitted world positions. Source/proxy
                    // transforms must never add another transform stream.
                    false
                } else {
                    let Ok(transformed) = transforms.decorate(&mut request, &pipelines) else {
                        continue;
                    };
                    transformed
                };
                let pipeline = specializer
                    .specialize_with(
                        runtime.device(),
                        request.variant,
                        &request.layout,
                        |variant| {
                            if transformed {
                                pipelines
                                    .transformed_descriptor(variant)
                                    .expect("validated transform stream provider")
                            } else {
                                pipelines.descriptor(variant)
                            }
                        },
                    )
                    .id();
                let draw = if default_mesh_draw && surface_layout.is_none() {
                    canonical::MESH_UNBOUND
                } else {
                    request.draw
                };
                let bind_groups = if default_mesh_draw || resident_draw {
                    let Some(record) = registry.material(request.material) else {
                        #[cfg(feature = "physics-resident")]
                        if resident_draw {
                            resident.1.as_mut().expect("admitted resident imports")
                                .queue_rejections.insert((view.camera, work_item.geometry),
                                    crate::physics_surface::resident::ResidentSurfaceRejection::MissingMaterial);
                        }
                        continue;
                    };
                    if surface_layout.is_some() {
                        let Some(binding) = canonical_material.binding_for_frame_slot(
                            runtime.queue(),
                            frame_slot,
                            request.material,
                            record,
                            &material_buffers,
                        ) else {
                            #[cfg(feature = "physics-resident")]
                            if resident_draw {
                                resident.1.as_mut().expect("admitted resident imports")
                                    .queue_rejections.insert((view.camera, work_item.geometry),
                                        crate::physics_surface::resident::ResidentSurfaceRejection::MaterialBindingUnavailable);
                            }
                            continue;
                        };
                        alloc::vec![binding]
                    } else {
                        alloc::vec::Vec::new()
                    }
                } else {
                    alloc::vec::Vec::new()
                };
                queues.push_with_external_indirect(
                    request.phase,
                    crate::PhaseItem {
                        pipeline,
                        draw,
                        geometry: request.geometry,
                        vertices: request.vertices,
                        vertex_sizes: request.vertex_sizes,
                        indices: request.indices,
                        index_size: request.index_size,
                        first: request.first,
                        count: request.count,
                        base_vertex: request.base_vertex,
                        indirect: None,
                        lod_level: request.lod_level,
                        material: request.material,
                        scene: request.scene,
                        order: request.order,
                    },
                    bind_groups,
                    external,
                );
                queued += 1;
            }
        }
        queues.sort();
        camera_queues.push(SurfaceViewQueue {
            surface: view.surface,
            frame_slot,
            queues,
        });
    }
    frame.publish(camera_queues);
    publish_frame(&mut commands, frame);
    stats.draws_queued += queued as u64;
}

pub(crate) fn checked_frame_slots(count: usize) -> Result<u32, SurfaceViewQueueError> {
    u32::try_from(count).map_err(|_| SurfaceViewQueueError::FrameSlotOverflow)
}

/// No binding lookup is allowed until this entire function succeeds. All
/// cameras (including empty-work cameras) get a prepared dense frame slot.
fn prepare_view_bindings(
    arena: &mut CanonicalMaterialBindings,
    device: &dyn RenderDevice,
    queue: &dyn RenderQueue,
    device_generation: u64,
    layout: Option<BindGroupLayoutId>,
    materials: &MaterialParameterBuffers,
    views: &[&ExtractedVisibilityView],
) -> Result<(), SurfaceViewQueueError> {
    let frame_slots = checked_frame_slots(views.len())?;
    if !arena.ensure_layout(device, device_generation, layout) {
        return Err(SurfaceViewQueueError::Layout);
    }
    if layout.is_none() {
        return Ok(());
    }
    if !arena.prepare_capacities(device, materials.header_capacity(), frame_slots) {
        return Err(SurfaceViewQueueError::Capacity);
    }
    if !arena.prepare_material_buffers(device, materials) {
        return Err(SurfaceViewQueueError::MaterialBuffers);
    }
    for (slot, view) in views.iter().enumerate() {
        let slot = slot as u32;
        if !arena.prepare_frame_slot(queue, slot, Some(&view.surface)) {
            return Err(SurfaceViewQueueError::FrameUpload { slot });
        }
    }
    Ok(())
}

fn publish_frame(commands: &mut Commands, frame: SurfaceViewQueues) {
    commands.queue(move |world| {
        world.insert_resource(frame);
    });
}

#[cfg(test)]
#[path = "queue_stage/tests.rs"]
mod tests;
