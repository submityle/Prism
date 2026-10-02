//! `Core3d` scheduling system recording the colour-grade dispatch for every
//! view.
//!
//! Mirrors [`super::super::vignette::dispatch`]: gate on the enable, resolve the
//! view, upload the immediate block, run one workgroup per 8x8 *pixel* block and
//! copy the graded output back over `scene_color`.
//!
//! The dispatch bounds-checks its invocations in the shader. Placed after the
//! composited HDR `scene_color` is ready and gated on
//! [`PrismColorGradeSettings::enabled`]. The output is copied back over
//! `scene_color` so the downstream post chain reads the graded image.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{
        ComputePassDescriptor, Extent3d, Origin3d, PipelineCache, TexelCopyTextureInfo,
        TextureAspect,
    },
    renderer::{RenderContext, ViewQuery},
};

use super::super::resources::ViewVisibilityBuffer;
use super::abi::COLOR_GRADE_WORKGROUP_SIZE;
use super::bind_groups::ViewColorGradeBindGroup;
use super::pipeline::ColorGradePipeline;
use super::resources::ViewColorGrade;
use super::settings::PrismColorGradeSettings;

/// `Core3d` scheduling system recording the `color_grade_main` dispatch for
/// every view whose grade texture and bind group are resident.
pub(crate) fn color_grade_pass(
    settings: Res<PrismColorGradeSettings>,
    view: ViewQuery<(
        &ViewColorGrade,
        &ViewColorGradeBindGroup,
        &ViewVisibilityBuffer,
    )>,
    pipeline: Res<ColorGradePipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enabled {
        return;
    }
    let (color_grade, group, visibility) = view.into_inner();

    // The pipeline must be resident before the pass runs.
    let Some(color_grade_pipeline) = cache.get_compute_pipeline(pipeline.pipeline()) else {
        return;
    };

    let size = color_grade.size;
    if size.x == 0 || size.y == 0 {
        return;
    }

    let params = settings.params(size);

    let groups_x = size.x.div_ceil(COLOR_GRADE_WORKGROUP_SIZE);
    let groups_y = size.y.div_ceil(COLOR_GRADE_WORKGROUP_SIZE);

    let encoder = ctx.command_encoder();

    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism color grade"),
            timestamp_writes: None,
        });
        pass.set_pipeline(color_grade_pipeline);
        pass.set_bind_group(0, group.group(), &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&params));
        pass.dispatch_workgroups(groups_x, groups_y, 1);
    }

    // Copy the graded result back over scene_color so the downstream post chain
    // (which samples scene_color, not color_grade_out) reads the graded image.
    // The pass cannot write scene_color in place because it binds scene_color as
    // a sampled read, so the chain ping-pongs into a dedicated target and then
    // blits back. Same extent + same format (SCENE_COLOR_FORMAT), single
    // mip/layer.
    encoder.copy_texture_to_texture(
        TexelCopyTextureInfo {
            texture: color_grade.color_grade_out_texture(),
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        TexelCopyTextureInfo {
            texture: visibility.scene_color_texture(),
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        },
    );
}
