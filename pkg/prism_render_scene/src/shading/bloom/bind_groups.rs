//! Per-view group-0 bind groups for the five bloom compute passes.
//!
//! One [`ViewBloomBindGroups`] carries every group the [`bloom_pass`] node
//! records, built once per frame from the freshly allocated
//! [`ViewBloomTextures`] pyramid and the shared [`BloomPipelines`] layouts:
//!
//! * `copy` (`copy_layout`) — `scene_color` (sampled) into the full-res base.
//! * `prefilter` (`downsample_layout`) — `scene_color` behind the linear-clamp
//!   sampler into the half-res mip 0.
//! * `downs[k]` (`downsample_layout`) — `down[k]` sampled into `down[k + 1]`;
//!   one per deeper downsample, so `downs.len() == levels - 1`.
//! * `ups[i]` (`merge_layout`) — the coarser mip sampled + `down[i]` loaded
//!   into `up[i]`; `ups[i]` targets `up[i]`, so `ups.len() == levels - 1` and
//!   the coarsest upsample reads `down[levels - 1]` while the rest read the
//!   previous `up`.
//! * `combine` (`merge_layout`) — the full-res base loaded + the accumulated
//!   bloom (`up[0]`, or `down[0]` for a single-level pyramid) sampled, written
//!   back into `scene_color`.
//!
//! Gated on `enable_bloom`: a disabled frame drops any stale groups so nothing
//! is bound, matching the exposure / SSGI precedent.
//!
//! [`bloom_pass`]: super::passes::bloom_pass

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries},
    renderer::RenderDevice,
};

use super::super::resources::ViewVisibilityBuffer;
use super::super::runtime::PrismShadingSettings;
use super::pipeline::BloomPipelines;
use super::resources::ViewBloomTextures;

/// All group-0 bind groups the bloom chain records for one view. Present only
/// when `enable_bloom` is set and the view has both a resident `scene_color`
/// and its bloom pyramid.
#[derive(Component)]
pub(crate) struct ViewBloomBindGroups {
    /// `bloom_copy_scene`: `scene_color` into the full-res base.
    pub(crate) copy: BindGroup,
    /// `bloom_prefilter_downsample`: `scene_color` into the half-res mip 0.
    pub(crate) prefilter: BindGroup,
    /// `bloom_downsample`: `down[k]` into `down[k + 1]`; `downs.len() == levels - 1`.
    pub(crate) downs: Vec<BindGroup>,
    /// `bloom_upsample`: coarser mip + `down[i]` into `up[i]`; `ups[i]` targets
    /// `up[i]`, so `ups.len() == levels - 1`.
    pub(crate) ups: Vec<BindGroup>,
    /// `bloom_combine_pass`: base + accumulated bloom back into `scene_color`.
    pub(crate) combine: BindGroup,
}

/// `PrepareBindGroups` system building the bloom bind groups for every view
/// with a resident `scene_color` and bloom pyramid, gated on `enable_bloom`.
pub(crate) fn prepare_bloom_bind_groups(
    mut commands: Commands,
    settings: Res<PrismShadingSettings>,
    pipelines: Res<BloomPipelines>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ViewVisibilityBuffer, &ViewBloomTextures)>,
) {
    for (entity, visibility, textures) in &views {
        if !settings.enable_bloom {
            commands.entity(entity).remove::<ViewBloomBindGroups>();
            continue;
        }

        let scene = visibility.scene_color_view();
        let base = &textures.base().default_view;
        let down = textures.down();
        let up = textures.up();
        let levels = down.len();
        // `prepare_bloom_textures` never inserts an empty pyramid, but stay
        // defensive: without a mip there is nothing to bind.
        if levels == 0 {
            commands.entity(entity).remove::<ViewBloomBindGroups>();
            continue;
        }

        let copy = device.create_bind_group(
            "prism bloom copy",
            &pipelines.copy_layout,
            &BindGroupEntries::sequential((scene, base)),
        );

        let prefilter = device.create_bind_group(
            "prism bloom prefilter",
            &pipelines.downsample_layout,
            &BindGroupEntries::sequential((&pipelines.sampler, scene, &down[0].default_view)),
        );

        // One deeper downsample per gap between mips: `down[k]` -> `down[k + 1]`.
        let downs: Vec<BindGroup> = (0..levels - 1)
            .map(|k| {
                device.create_bind_group(
                    "prism bloom downsample",
                    &pipelines.downsample_layout,
                    &BindGroupEntries::sequential((
                        &pipelines.sampler,
                        &down[k].default_view,
                        &down[k + 1].default_view,
                    )),
                )
            })
            .collect();

        // One upsample per `up[i]` target (finest .. one-before-coarsest). The
        // coarse source is `down[levels - 1]` for the coarsest step and the
        // previous `up[i + 1]` for the rest; the finer detail is always
        // `down[i]`; the destination is `up[i]`.
        let ups: Vec<BindGroup> = (0..levels - 1)
            .map(|i| {
                let coarse = if i == levels - 2 {
                    &down[levels - 1].default_view
                } else {
                    &up[i + 1].default_view
                };
                device.create_bind_group(
                    "prism bloom upsample",
                    &pipelines.merge_layout,
                    &BindGroupEntries::sequential((
                        &pipelines.sampler,
                        coarse,
                        &down[i].default_view,
                        &up[i].default_view,
                    )),
                )
            })
            .collect();

        // The accumulated bloom is `up[0]` once there is anything to upsample;
        // a single-level pyramid has no `up` targets, so mip 0 is the bloom.
        let bloom_view = if levels == 1 {
            &down[0].default_view
        } else {
            &up[0].default_view
        };
        let combine = device.create_bind_group(
            "prism bloom combine",
            &pipelines.merge_layout,
            &BindGroupEntries::sequential((&pipelines.sampler, base, bloom_view, scene)),
        );

        commands.entity(entity).insert(ViewBloomBindGroups {
            copy,
            prefilter,
            downs,
            ups,
            combine,
        });
    }
}
