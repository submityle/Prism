//! Device-side preparation of the resident `GPU` water bodies.
//!
//! This system runs in [`bevy_render::Render`] under
//! [`bevy_render::RenderSystems::PrepareResources`] and turns each extracted
//! [`WaterBody`](super::body::WaterBody) into a resident
//! [`WaterGpuBody`]. It is the device half of the prepare stage: the
//! device-free golden [`prepare`] does all the substep / iteration / cascade
//! expansion and per-kernel workgroup sizing, and this system only allocates
//! the `wgpu` buffers and storage textures, builds the twelve bind groups and
//! records the resident body plus its ordered dispatch schedule.
//!
//! Bodies are rebuilt every frame from scratch (clear-then-refill), matching
//! the extract stage: the resident resources are recreated and re-uploaded,
//! which is the honest baseline before a later slice folds in persistent
//! buffers with `queue`-side rewrites. A body whose live passes expand to no
//! dispatch contributes no resident body, so an empty or dormant body never
//! fabricates a solve and the dispatch node stays a genuine no-op for it.

use bevy_ecs::prelude::*;
use bevy_render::renderer::RenderDevice;

use prism_render_architecture::water::gpu::pipeline::prepare;

use super::bind_groups::{WaterBodyBindGroups, WaterBodyGpuBuffers};
use super::body::ExtractedWater;
use super::pipeline::WaterComputePipelines;
use super::resources::{WaterGpuBodies, WaterGpuBody};

/// Rebuilds the resident `GPU` water bodies from the extracted bodies.
///
/// Clears the existing bodies and, for every extracted body, expands its
/// float-free dispatch schedule with the golden [`prepare`], and — when that
/// schedule is non-empty — allocates the resident buffers and storage
/// textures, builds the twelve bind groups and pushes the resulting
/// [`WaterGpuBody`] with its ordered schedule. Bodies whose live passes expand
/// to no dispatch are skipped so the dispatch node never records an empty
/// solve.
pub(crate) fn prepare_water_bodies(
    mut bodies: ResMut<WaterGpuBodies>,
    extracted: Res<ExtractedWater>,
    pipelines: Option<Res<WaterComputePipelines>>,
    device: Res<RenderDevice>,
) {
    bodies.bodies.clear();

    // The pipelines are built once at `RenderStartup`; if that resource is not
    // present yet there is nothing to bind against, so skip this frame rather
    // than fabricate a body.
    let Some(pipelines) = pipelines else {
        return;
    };

    for body in &extracted.bodies {
        let plan = prepare(&body.as_extract());
        if plan.dispatches.is_empty() {
            continue;
        }

        let upload = body.as_upload();
        let buffers = WaterBodyGpuBuffers::create(&device, &upload);
        let bind_groups = WaterBodyBindGroups::create(&device, &pipelines, &buffers);
        bodies
            .bodies
            .push(WaterGpuBody::new(buffers, bind_groups, plan.dispatches));
    }
}
