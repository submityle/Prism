//! Bind groups for the IBL precompute passes.
//!
//! Both tables are global (view-independent), so their bind groups are
//! resources rather than per-view components:
//!
//! * [`DfgLutBindGroup`] binds the write-only DFG storage texture.  It is built
//!   once, as soon as the pipeline layout and backing texture are resident, and
//!   then reused every frame — the storage view never changes.
//! * [`EnvPrefilterBindGroups`] binds, per output mip, the source radiance cube
//!   plus its sampler and that mip's write-only array target.  It is rebuilt
//!   only when the active environment probe changes, keyed by the source id.

use bevy_asset::AssetId;
use bevy_ecs::prelude::*;
use bevy_image::Image;
use bevy_render::{
    render_asset::RenderAssets,
    render_resource::{BindGroup, BindGroupEntries},
    renderer::RenderDevice,
    texture::GpuImage,
};

use super::extract::ExtractedIblSource;
use super::pipeline::{BrdfLutPipeline, EnvPrefilterPipeline};
use super::resources::{DfgLutTexture, PrefilteredEnvironmentMap};

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

/// One bind group per output mip for the prefilter pass, plus the source asset
/// id they were built against so they can be rebuilt when the probe changes.
#[derive(Resource, Default)]
pub(crate) struct EnvPrefilterBindGroups {
    /// group-0 bind group for each output mip, in mip order.
    pub(crate) mips: Vec<BindGroup>,
    /// Asset id of the source radiance cube the current groups were built for.
    pub(crate) source: Option<AssetId<Image>>,
}

/// `PrepareBindGroups` system that (re)builds [`EnvPrefilterBindGroups`] when
/// the active probe's source cube changes.
///
/// The groups are keyed by the source [`AssetId`]: while the same probe stays
/// resident the system is a no-op, and when the probe changes (or is cleared)
/// the groups are rebuilt (or dropped) so the prefilter pass reconvolves the
/// new environment.  Building is deferred until the source's [`GpuImage`] is
/// uploaded so the cube view actually exists.
pub(crate) fn prepare_env_prefilter_bind_groups(
    mut bind_groups: ResMut<EnvPrefilterBindGroups>,
    extracted: Res<ExtractedIblSource>,
    pipeline: Option<Res<EnvPrefilterPipeline>>,
    target: Option<Res<PrefilteredEnvironmentMap>>,
    images: Res<RenderAssets<GpuImage>>,
    device: Res<RenderDevice>,
) {
    let Some(source_id) = extracted.specular_map else {
        // No active probe: drop any stale groups so the pass idles.
        if bind_groups.source.is_some() {
            bind_groups.mips.clear();
            bind_groups.source = None;
        }
        return;
    };

    let (Some(pipeline), Some(target)) = (pipeline, target) else {
        return;
    };

    // Already built for this exact source — nothing to do.
    if bind_groups.source == Some(source_id) && !bind_groups.mips.is_empty() {
        return;
    }

    // Wait until the source cube's GPU image is uploaded.
    let Some(source_image) = images.get(source_id) else {
        return;
    };

    let mut mips = Vec::with_capacity(target.mip_count() as usize);
    for mip in 0..target.mip_count() {
        let Some(write_view) = target.mip_write_view(mip) else {
            continue;
        };
        let bind_group = device.create_bind_group(
            "prism env prefilter",
            &pipeline.layout,
            &BindGroupEntries::sequential((
                &source_image.texture_view,
                &source_image.sampler,
                write_view,
            )),
        );
        mips.push(bind_group);
    }

    bind_groups.mips = mips;
    bind_groups.source = Some(source_id);
}
