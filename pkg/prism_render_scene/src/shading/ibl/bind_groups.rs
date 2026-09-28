//! Bind group for the DFG lookup-table precompute.
//!
//! The table is global (view-independent), so its single bind group is a
//! resource rather than a per-view component.  It is built once, as soon as
//! both the pipeline layout and the backing texture are resident, and then
//! reused every frame — the storage view it references never changes.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries},
    renderer::RenderDevice,
};

use super::pipeline::BrdfLutPipeline;
use super::resources::DfgLutTexture;

/// The single group-0 bind group binding the DFG storage texture as the compute
/// pass's write target.  Present once both the pipeline and texture exist.
#[derive(Resource)]
pub(crate) struct DfgLutBindGroup(pub(crate) BindGroup);

/// `PrepareBindGroups` system that builds [`DfgLutBindGroup`] exactly once.
///
/// The DFG table's storage view is allocated once and never resized, so there
/// is nothing to rebuild per frame; the early return keeps the system a no-op
/// after the first successful build.
pub(crate) fn prepare_dfg_lut_bind_group(
    mut commands: Commands,
    existing: Option<Res<DfgLutBindGroup>>,
    pipeline: Option<Res<BrdfLutPipeline>>,
    texture: Option<Res<DfgLutTexture>>,
    device: Res<RenderDevice>,
) {
    if existing.is_some() {
        return;
    }
    let (Some(pipeline), Some(texture)) = (pipeline, texture) else {
        return;
    };

    let bind_group = device.create_bind_group(
        "prism DFG LUT",
        &pipeline.layout,
        &BindGroupEntries::single(texture.view()),
    );
    commands.insert_resource(DfgLutBindGroup(bind_group));
}
