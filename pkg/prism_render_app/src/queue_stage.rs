//! The queue stage: the visible set becomes the frame's draws.
//!
//! Between culling (which decides what is visible) and recording (which issues
//! the commands) sits the queue: it resolves each visible work item against the
//! scene's tables, specializes its pipeline, and appends a phase item. Bevy
//! splits this across `queue_material_meshes` and the phase machinery; here it is
//! one stage with one job, so the record stage only ever walks a queue that is
//! already final.

use prism_ecs::prelude::{Res, ResMut};
use prism_render_driver::{FrameSync, Instance};

use crate::draw::canonical;
use crate::mesh_pipeline::MeshPipelineSpecializer;
use crate::mesh_pipelines::MeshPipelines;
use crate::queue::RenderQueues;
use crate::render_asset::RenderAssets;
use crate::runtime_resource::RenderRuntimeResource;
use crate::scene::{SceneRegistry, VisibilityWorkList};
use crate::stats::RenderFrameStats;

use crate::frame::RENDER_DEPTH_FORMAT;
use crate::mesh::RenderMesh;

/// Resolves the frame's visible work into queued draws.
///
/// Runs in [`RenderSystems::Queue`](crate::RenderSystems::Queue), before the
/// record stage. Three things happen, in order:
///
/// 1. every bound geometry's mesh class is synced from its prepared mesh, so the
///    layout and buffers the queue specializes on are the ones the mesh actually
///    has this frame;
/// 2. each visible work item is resolved against the scene registry, specialized
///    through the mesh pipeline provider, and appended to its phase;
/// 3. the phases are sorted deterministically, so a frame records the same order
///    every run.
///
/// A work item whose geometry, material, mesh, or class is not resident is
/// skipped: the queue and the frame agree on what is drawable, rather than the
/// frame drawing with a class nothing declared.
pub(crate) fn fill_queues<I>(
    work: Res<VisibilityWorkList>,
    mut registry: ResMut<SceneRegistry>,
    meshes: Res<RenderAssets<RenderMesh>>,
    mut specializer: ResMut<MeshPipelineSpecializer>,
    pipelines: Res<MeshPipelines>,
    mut queues: ResMut<RenderQueues>,
    mut stats: ResMut<RenderFrameStats>,
    runtime: Res<RenderRuntimeResource<I>>,
) where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    registry.sync_mesh_classes(&meshes);
    if work.is_empty() {
        return;
    }
    let queued = queues.fill_from_visibility(
        runtime.runtime().device(),
        &mut specializer,
        &registry,
        &meshes,
        work.items(),
        // The frame has a depth attachment, so the variants must declare the
        // matching depth state: a pipeline and the pass it records into have to
        // agree on the attachment set.
        Some(RENDER_DEPTH_FORMAT),
        1,
        canonical::MESH,
        &*pipelines,
    );
    queues.sort();
    stats.draws_queued += queued as u64;
}
